#!/usr/bin/env bash
#
# Guard: a compose healthcheck may not probe LOOPBACK from inside its own
# container.
#
# WHY (club-3090#1295): the SGLang composes shipped
#
#     test: ["CMD-SHELL", "curl -sf http://localhost:30000/health || exit 1"]
#
# which runs INSIDE the container against 127.0.0.1 — the one path that stays up
# when the bind address is wrong. When #1292 silently dropped `--host 0.0.0.0`
# the server bound 127.0.0.1, docker's port forward had nothing to reach, and
# every external client got an instant connection reset (~0.5 ms, not a timeout).
# Meanwhile:
#
#     docker ps                -> "Up 27 minutes (healthy)"
#     container healthcheck    -> passing
#     engine log               -> 127.0.0.1 "GET /health HTTP/1.1" 200 OK
#     an actual client         -> connection refused
#
# The probe would have passed identically if the port mapping had been deleted
# outright. It tests that the process is alive and READS as "the service is
# reachable". That cost two misdiagnoses in one day (first a crash, then an IPv6
# resolution problem) before /proc/net/tcp settled it.
#
# THE RULE: probe the container's own ROUTABLE address — the address docker's
# port forward actually targets — so the check fails when the server is bound
# somewhere a client cannot reach:
#
#     test: ["CMD-SHELL", "curl -sf \"http://$${HOSTNAME:-$$(hostname)}:PORT/health\" || exit 1"]
#
# ⚠️ This is the FALSE-CLEAN class: the old probe's success was indistinguishable
# from its failure. Ask of any healthcheck: if the service were unreachable,
# would this output differ?
#
# ⚠️ Must FAIL against the pre-fix tree, or it is asserting the wrong thing.
#
# ⚠️ A healthcheck must never fall back to loopback when the address lookup
# fails — that reintroduces the silent pass. Failing loudly (unhealthy) is
# correct; nothing in this repo gates on container health (no depends_on:
# service_healthy, and switch.sh's ready probe curls the endpoint from the HOST),
# so an unhealthy mark is a visible signal, not a boot blocker.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
# #779: this gate shells out to python3, which decodes reads/argv with the
# LOCALE codec unless UTF-8 mode is on — and these composes are full of unicode.
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---------------------------------------------------------------------------
# DEBT REGISTER — same defect, NOT fixed by #1295, which was scoped to SGLang.
# These 31 llama.cpp-family composes carry the identical loopback probe. They
# are listed rather than silently exempted so the shape is visible and cannot
# grow: the gate FAILS if a listed compose no longer has a loopback probe (delete
# its line when you fix it) and FAILS if any compose NOT listed here has one.
# Shrink this list; never grow it.
# ---------------------------------------------------------------------------
KNOWN_LOOPBACK=(
  models/deepseek-v4-flash-0731/llama-cpp/compose/dual/unsloth-iq2-xxs/offload.yml
  models/deepseek-v4-flash-0731/llama-cpp/compose/dual/unsloth-q8-kxl/offload.yml
  models/deepseek-v4-flash-0731/llama-cpp/compose/multi4/unsloth-q8-kxl/offload.yml
  models/deepseek-v4-flash-0731/llamacpp-club3090/compose/dual/unsloth-q8-kxl/moecache.yml
  models/deepseek-v4-flash-0731/llamacpp-club3090/compose/multi4/unsloth-q8-kxl/moecache.yml
  models/deepseek-v4-flash-vision-exp/llamacpp-club3090/compose/dual/unsloth-ud-q8kxl/moecache.yml
  models/deepseek-v4-flash-vision-exp/llamacpp-club3090/compose/multi4/unsloth-ud-q8kxl/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q2k/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q2k/offload.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q3km/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q3km/offload.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/dual/unsloth-ud-iq3xxs/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/dual/unsloth-ud-iq4xs/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q2k/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q2k/offload.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q3km/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q3km/offload.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi4/unsloth-ud-iq3xxs/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi4/unsloth-ud-iq4xs/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q2k/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q2k/offload.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q3km/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q3km/offload.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi8/unsloth-ud-iq3xxs/moecache.yml
  models/glm-5.3-flash/llamacpp-club3090/compose/multi8/unsloth-ud-iq4xs/moecache.yml
  models/inkling-small/llamacpp-club3090/compose/dual/unsloth-ud-iq4xs/moecache.yml
  models/inkling-small/llamacpp-club3090/compose/dual/unsloth-ud-iq4xs/residency.yml
  models/inkling-small/llamacpp-club3090/compose/multi4/unsloth-ud-iq4xs/moecache.yml
  models/qwen3.8-flash-next/llamacpp-club3090/compose/dual/unsloth-ud-q4kxl/moecache.yml
  models/qwen3.8-flash-next/llamacpp-club3090/compose/multi4/unsloth-ud-q4kxl/moecache.yml
  models/qwen3.8-flash-next/llamacpp-club3090/compose/multi8/unsloth-ud-q4kxl/moecache.yml
)

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
scan="$tmp/scan.py"

# ONE scanner, used for both the control and the real run. A control that
# exercises different code proves nothing about the code that matters.
cat > "$scan" <<'PYEOF'
import glob, os, re, sys

root = sys.argv[1]
# Loopback in any spelling a curl URL can carry.
LOOPBACK = re.compile(r'//(?:localhost|127\.0\.0\.1|\[::1\])[:/]')

loopback, probes = [], 0
for f in sorted(glob.glob(os.path.join(root, "models/**/*.yml"), recursive=True)):
    raw = open(f, encoding="utf-8", errors="replace").read()
    rel = os.path.relpath(f, root)
    in_hc = False
    for line in raw.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            continue                      # prose, not a healthcheck
        if s.startswith("healthcheck:"):
            in_hc = True
            continue
        if not in_hc:
            continue
        # The healthcheck block ends at the first key that is not one of its own.
        if s.startswith("test:"):
            probes += 1
            if LOOPBACK.search(s):
                loopback.append(rel)
            in_hc = False
        elif s and not s.split(":")[0] in (
                "interval", "timeout", "retries", "start_period",
                "start_interval", "disable"):
            in_hc = False
print("PROBES %d" % probes)
for p in loopback:
    print("LOOPBACK " + p)
PYEOF

# --- positive control FIRST -------------------------------------------------
# A scan that silently matches nothing is indistinguishable from a clean tree.
mkdir -p "$tmp/probe/models/planted/eng/compose/dual/q"
printf '%s\n' \
  '    healthcheck:' \
  '      test: ["CMD-SHELL", "curl -sf http://localhost:30000/health || exit 1"]' \
  > "$tmp/probe/models/planted/eng/compose/dual/q/bad.yml"
printf '%s\n' \
  '    healthcheck:' \
  '      test: ["CMD-SHELL", "curl -sf \"http://$${HOSTNAME:-$$(hostname)}:30000/health\" || exit 1"]' \
  > "$tmp/probe/models/planted/eng/compose/dual/q/ok.yml"
printf '%s\n' \
  '    # healthcheck:' \
  '    #   test: ["CMD-SHELL", "curl -sf http://localhost:30000/health || exit 1"]' \
  > "$tmp/probe/models/planted/eng/compose/dual/q/commented.yml"

ctl="$(python3 "$scan" "$tmp/probe")"
[[ "$(printf '%s\n' "$ctl" | command grep -c '^LOOPBACK ')" == "1" ]] || {
  echo "FAIL: positive control — scanner should flag exactly 1 planted loopback probe" >&2
  printf '%s\n' "$ctl" >&2; exit 1; }
[[ "$ctl" == *"PROBES 2"* ]] || {
  echo "FAIL: positive control — scanner should have seen 2 real healthchecks (the commented one is prose)" >&2
  printf '%s\n' "$ctl" >&2; exit 1; }
echo "  ✓ scanner flags a loopback probe, passes a routable one, ignores commented YAML"

# --- the actual assertion ---------------------------------------------------
out="$(python3 "$scan" "$ROOT")"
probes="$(printf '%s\n' "$out" | command sed -n 's/^PROBES //p')"
found="$(printf '%s\n' "$out" | command sed -n 's/^LOOPBACK //p' | command sort)"

if [[ -z "$probes" || "$probes" -lt 1 ]]; then
  echo "FAIL: scanned no healthchecks at all — the search is wrong, not the tree" >&2
  exit 1
fi

known="$(printf '%s\n' "${KNOWN_LOOPBACK[@]}" | command sort)"
fails=0

# (a) anything loopback that is NOT registered debt
while IFS= read -r p; do
  [[ -n "$p" ]] || continue
  if ! printf '%s\n' "$known" | command grep -qxF "$p"; then
    echo "FAIL: $p healthcheck probes LOOPBACK from inside the container." >&2
    echo "      It cannot fail when the bind address is wrong (club-3090#1295)." >&2
    echo "      Probe the routable address instead:" >&2
    echo "        curl -sf \"http://\$\${HOSTNAME:-\$\$(hostname)}:<port>/health\"" >&2
    fails=$((fails+1))
  fi
done <<< "$found"

# (b) registered debt that is no longer broken — delete the line, don't let the
#     register rot into a list nobody trusts.
while IFS= read -r p; do
  [[ -n "$p" ]] || continue
  if ! printf '%s\n' "$found" | command grep -qxF "$p"; then
    echo "FAIL: $p is in KNOWN_LOOPBACK but no longer probes loopback — remove it from the register." >&2
    fails=$((fails+1))
  fi
done <<< "$known"

if [[ "$fails" -gt 0 ]]; then
  echo "$fails healthcheck-reachability check(s) failed" >&2
  exit 1
fi
echo "PASS: test-compose-healthcheck-reachability — $probes healthcheck(s) scanned, ${#KNOWN_LOOPBACK[@]} registered as known debt (#1295)"
