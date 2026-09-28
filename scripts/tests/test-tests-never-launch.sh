#!/usr/bin/env bash
# Gate: no test in this suite may BOOT A MODEL.
#
# ⚠️⚠️ FOUND IN THE WILD 2026-09-18. `test-engine-kind-resolver.sh` drives the
# real `spec-sweep.sh` to check that an SGLang slug classifies as `sglang`. Its
# own comment said it runs "against a dead URL so it prints its classification
# and exits before touching a server". It did not: the `[spec-sweep] engine=`
# banner prints BEFORE the dry-run check, so for every non-llamacpp family the
# script sailed on into its real work — `switch.sh --force <slug>` per arm. A
# unit test for a string resolver was booting a 27B model on both GPUs.
#
# Three things that look like protection and are not:
#   • `timeout 120` — TERMs `spec-sweep` ONLY. The `switch.sh` GRANDCHILD is not
#     in the signalled process group, so it survives and keeps launching long
#     after the test has "finished". Observed: the timeout exceeded threefold,
#     with `sglang-qwen38-27b-max-dual` still serving on :8145.
#   • `| head -40` — nothing sends SIGPIPE while the child is quiet.
#   • a dead `URL=` — spec-sweep BOOTS the slug precisely BECAUSE nothing answers.
#
# The damage is silent: the test still passes (it got its banner), so the suite
# goes green while the rig quietly loses both GPUs and the next GPU-sensitive
# test flakes against a machine under load.
#
# SCOPE — deliberately narrow, to stay signal. Only `spec-sweep.sh` is policed
# for the dry-run knob, because it is the one harness script whose ORDINARY
# behaviour is to boot. `switch.sh` / `launch.sh` are policed for a neutralizer.
# Read-only sub-commands are not launches and are listed as such.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUTF8="${PYTHONUTF8:-1}"

python3 - <<'PY'
import io, glob, re, sys

# A launch is neutralized when the command carries one of these. Each is a real
# mechanism in this repo, not a guess:
NEUTRALIZERS = (
    "SWEEP_DRY=1",        # spec-sweep: prints the plan, boots/measures nothing
    "COMPOSE_BIN=:",      # switch.sh: `docker compose` becomes a shell no-op
    "SWITCH=",            # launch.sh: delegates to a mock switch in TMP_DIR
    "ESTATE_HELPER=",     # launch.sh: every estate boot goes through
                          # `python3 "$ESTATE_HELPER" boot --file …` (launch.sh
                          # :1274), so overriding it to a test fake replaces the
                          # boot itself. VERIFIED against test-parallel-boot.sh's
                          # helper, which prints "ARGS …" and exits — not assumed
                          # from the variable's name.
    "DRY_RUN=1", "--dry-run",
    'PATH="$T/bin:',      # test-gpu-mode-service-upgrade: `docker` and `sudo` are
                          # fixture-driven shims in $T/bin, prepended INLINE on each
                          # gpu-mode call. VERIFIED, not assumed: that test refuses
                          # to run unless `command -v docker/sudo` under that PATH
                          # resolves to the shims, and it asserts on the shims' call
                          # log, so a real docker would fail it rather than launch.
)
# Read-only sub-commands — these never reach `compose up`.
READONLY = (
    "--list", "--explain", "--defaults", "--set-default", "--clear-default",
    "--validate-estate", "--topology", "--check", "--json", "--help",
    "--version", "--down", "--print", "--emit",
    "service-images",     # gpu-mode: show_service_images → service_image_rows runs only
                          # `docker compose config` + `docker ps/inspect`, and prints.
                          # VERIFIED against gpu-mode.sh, not assumed from the name.
)
LAUNCHERS = re.compile(r"\b(spec-sweep|switch|launch|gpu-mode)\.sh\b")
# Execution, not mention: `bash <path>/<launcher>.sh` anywhere in the command.
#
# ⚠️ THIS PATTERN WAS ONCE CLEVERER AND THEREFORE WRONG. It used to anchor on a
# command boundary and step over env assignments with `(?:\w+=\S*\s+)*`, so
# `SWEEP_N="0 1"` — a VALUE CONTAINING A SPACE — broke the walk and the regex
# matched nothing. The gate then reported "all neutralized" while blind to the
# exact line it exists to catch. Caught only by running a negative control; the
# green was indistinguishable from real safety. Keep it dumb and broad.
EXEC = re.compile(r"\b(?:bash|sh)\s+\S*(?:spec-sweep|switch|launch|gpu-mode)\.sh\b")

def commands(path):
    """Yield (first_lineno, joined_command). Shell continuations are joined so a
    guard on the line ABOVE the invocation still counts — getting this wrong is
    how a naive scan reports false positives on correctly-guarded calls."""
    buf, start, heredoc = "", 0, None
    for i, raw in enumerate(io.open(path, encoding="utf-8"), 1):
        line = raw.rstrip("\n")
        # Heredoc BODIES are data, not shell. The docs-slug gate embeds strings
        # like "bash scripts/switch.sh <prod>" in a python heredoc and feeds them
        # to a TEXT PARSER — matching those would be a false positive that trains
        # people to add allowlist entries for things that never execute.
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
            continue
        m = re.search(r"<<-?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?\s*$", line)
        if m:
            heredoc = m.group(1)
            continue
        if not buf:
            start = i
        stripped = line.strip()
        if stripped.startswith("#"):          # a comment cannot launch anything
            buf = ""; continue
        if line.endswith("\\"):
            buf += line[:-1] + " "; continue
        buf += line
        yield start, buf
        buf = ""
    if buf:
        yield start, buf

# A launcher name inside an ASSERTION is a string being compared, not a command
# being run: `assert_contains "$OUT" "bash scripts/launch.sh"` checks that a
# report RECOMMENDS the launcher. Keyed on the command's leading word, so it
# cannot accidentally excuse a real invocation later in the line.
ASSERT_HELPERS = ("assert_contains", "assert_not_contains", "assert", "echo",
                  "printf", "grep", "command grep", ":")

bad = []
scanned = 0
for path in sorted(glob.glob("scripts/tests/*.sh")):
    if path.endswith("test-tests-never-launch.sh"):
        continue
    for lineno, cmd in commands(path):
        if not LAUNCHERS.search(cmd) or not EXEC.search(cmd):
            continue
        lead = cmd.strip().split()
        lead = " ".join(lead[:2]) if lead[:1] == ["command"] else (lead[0] if lead else "")
        if lead in ASSERT_HELPERS:
            continue
        scanned += 1
        if any(n in cmd for n in NEUTRALIZERS) or any(r in cmd for r in READONLY):
            continue
        # spec-sweep gets a pointed message: its dry knob is the whole fix.
        why = ("spec-sweep.sh BOOTS by design — set SWEEP_DRY=1 (the banner still "
               "prints; it is emitted before the dry check)"
               if "spec-sweep.sh" in cmd else
               "neutralize it (COMPOSE_BIN=: / a mock SWITCH=) or use a read-only sub-command")
        bad.append((path, lineno, cmd.strip()[:150], why))

if bad:
    print("FAIL: test(s) invoke a launcher with nothing to stop it booting a model:")
    for p, n, c, why in bad:
        print(f"  ⛔ {p}:{n}")
        print(f"       {c}")
        print(f"       → {why}")
    print()
    print("  A green suite is not proof: the leaked launch outlives `timeout` (it")
    print("  signals the direct child only, not the switch.sh grandchild), so the")
    print("  test passes while the rig keeps serving.")
    sys.exit(1)

print(f"  ✓ no test can boot a model ({scanned} launcher invocation(s) checked, "
      f"all neutralized or read-only)")
PY

echo "test-tests-never-launch: ok"
