#!/usr/bin/env bash
# test-launch-knobs — the launch-knob catalogue (#1465 phase 3a) says what the composes really do.
#
# WHY THIS EXISTS
# ---------------
# #1465 lets a user persist launch settings (KV offload tiers, thinking and effort
# defaults, the drafter depth) globally or per slug. Two silent failures sit under that:
#   * a setting saved for a slug whose compose never reads it does nothing (gotcha 4);
#   * a compose can only use what docker delivers — a knob it reads as $${NAME} but
#     never forwards, or interpolates into a field that never reaches the process, is
#     dead, and a test that stops short of the delivery path goes green on it (gotcha 5).
# And the value domains differ per engine and model (gotcha 6): SGLang refuses a bad
# REASONING_EFFORT at boot, vLLM boots and fails every request, ENABLE_THINKING is
# spliced raw into JSON. So the catalogue (scripts/lib/profiles/launch-knobs.json)
# records the domains, and this guard holds it to the composes:
#   1. the real tree — shape; every reader covered by exactly one domain; every domain
#      run through the compose code it cites (shell fragment / JSON splice); every claim
#      rendered through `docker compose config` via BOTH delivery channels (process env
#      as switch.sh/launch.sh use, --env-file as gpu-mode.sh uses);
#   2. the registry emit's `knobs` field equals the scan, slug by slug;
#   3. a ratchet on `- NAME=${NAME:-}` forwards of catalogued knobs (they arrive EMPTY,
#      not absent — the atoi("")==0 class). Existing ones are benign (every read uses
#      ${NAME:-x}, checked in 1) and are counted; the count may only fall;
#   4. self-tests: fixture composes and deliberately broken catalogue copies, each of
#      which MUST make its check fail — a guard that cannot fail proves nothing.
# Read-only: `docker compose config` renders, nothing is started. Skips the docker legs
# cleanly when docker compose is unavailable.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fails=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fails=$((fails+1)); }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ── 1. the real tree ────────────────────────────────────────────────────────────
echo "[1] catalogue vs the registered composes"
python3 scripts/lib/profiles/launch_knobs_check.py --root "$ROOT" || bad "launch_knobs_check reported problems (above)"
if ! python3 scripts/lib/profiles/launch_knobs.py --check >/dev/null 2>&1; then
  echo "test-launch-knobs: the catalogue does not load (above) — nothing further can be checked" >&2
  exit 1
fi

# ── 2. the emit carries exactly the scan ────────────────────────────────────────
echo "[2] registry-emit.sh --json 'knobs' == the compose scan"
if python3 -c 'import yaml' 2>/dev/null; then
  if bash scripts/lib/registry-emit.sh --json "$ROOT" > "$TMP/emit.json" 2> "$TMP/emit.err"; then
    python3 - "$ROOT" "$TMP/emit.json" <<'PY' || bad "emit 'knobs' drifted from the scan (above)"
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root))
from scripts.lib.profiles import launch_knobs as lk
names = lk.knob_names(lk.load_catalogue())
variants = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["variants"]
bad = [(v["slug"], v["knobs"], lk.consumed_knobs(root / v["compose_path"], names))
       for v in variants if v["knobs"] != lk.consumed_knobs(root / v["compose_path"], names)]
for b in bad[:10]:
    print(f"  ✗ {b[0]}: emit {b[1]} != scan {b[2]}", file=sys.stderr)
if bad:
    raise SystemExit(1)
print(f"  ✓ {len(variants)} variants: emit knobs == scan")
PY
    command grep -q 'launch-knob' "$TMP/emit.err" && bad "emit warned about the catalogue: $(cat "$TMP/emit.err")"
  else
    bad "registry-emit.sh --json failed: $(tail -3 "$TMP/emit.err")"
  fi
else
  echo "  - SKIP: PyYAML missing (the --json path requires it; the scan itself is stdlib)"
fi

# ── 3. ratchet: NAME=${NAME:-} forwards of catalogued knobs ─────────────────────
echo "[3] empty-string forwards (- VAR=\${KNOB:-}) — a ratchet that only falls"
# Existing ones, 2026-09-28. All are read back with ${NAME:-default} (arm 1 checks
# that), so today empty == unset for them; the bare form (- NAME) is the target.
declare -A CEILING=([SPEC_N]=112 [KV_OFFLOAD_GB]=5 [KV_OFFLOAD_DISK]=5)
KNOBS="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from scripts.lib.profiles import launch_knobs as lk; print("|".join(lk.knob_names(lk.load_catalogue())))' "$ROOT")"
# empty_forwards DIR → "file:line:KNOB" for every list (- VAR=${KNOB:-}) or map
# (VAR: ${KNOB:-}) environment entry whose value is a catalogued knob with an EMPTY default.
empty_forwards() {
  command grep -rnE --include='*.yml' \
    "^[[:space:]]*(-[[:space:]]+[\"']?[A-Za-z_][A-Za-z0-9_]*=|[\"']?[A-Za-z_][A-Za-z0-9_]*[\"']?:[[:space:]]*[\"']?)\\\$\\{(${KNOBS})(:?-)\\}[\"']?[[:space:]]*\$" \
    "$1" 2>/dev/null | command grep -v '/_archive/' | sed -E 's/^([^:]+:[0-9]+):.*\$\{([A-Z0-9_]+):?-\}.*/\1:\2/'
}
EF="$(empty_forwards models)"
fails_before=$fails
for k in ${KNOBS//|/ }; do
  n="$(printf '%s\n' "$EF" | command grep -c ":${k}\$" || true)"
  c="${CEILING[$k]:-0}"
  if (( n > c )); then
    bad "$k: $n empty-string forward(s), ceiling $c — forward it bare (- $k) so unset stays unset:"
    printf '%s\n' "$EF" | command grep ":${k}\$" | sed 's/^/        /' >&2
  elif (( n < c )); then
    bad "$k: $n empty-string forward(s) now, ceiling still $c — lower CEILING[$k] to $n (the ratchet only falls)"
  fi
done
if (( fails == fails_before )); then
  ok "empty-string forwards at their ceilings ($(for k in "${!CEILING[@]}"; do printf '%s=%s ' "$k" "${CEILING[$k]}"; done)— none new)"
fi

# ── 4. self-tests: every check above can fail ──────────────────────────────────
echo "[4] self-tests — fixtures and broken catalogue copies must each fail"
mkdir -p "$TMP/fx"
cat > "$TMP/fx/good.yml" <<'YML'
services:
  probe:
    image: busybox
    environment:
      - SPEC_N
      # - KV_OFFLOAD_GB    (a commented-out entry is not a read)
    entrypoint:
      - bash
      - -c
      - |
        n="$${SPEC_N:-3}"   # $${KV_OFFLOAD_DISK} only in this trailing comment
        # $${ENABLE_THINKING} in a shell comment is dead code
        exec echo "$$n"
    command: ["--effort", "${REASONING_EFFORT:-low}"]
YML
cat > "$TMP/fx/undelivered.yml" <<'YML'
x-note: "effort ${REASONING_EFFORT:-low}"
services:
  probe:
    image: busybox
    labels:
      - "spec=${SPEC_N:-3}"
    command: ["true"]
YML
cat > "$TMP/fx/dead-read.yml" <<'YML'
services:
  probe:
    image: busybox
    entrypoint: ["bash", "-c", "echo $${KV_OFFLOAD_GB:-off}"]
YML
cat > "$TMP/fx/pinned.yml" <<'YML'
services:
  probe:
    image: busybox
    environment:
      ENABLE_THINKING: "true"
    command: ["true"]
YML
cat > "$TMP/fx/same-indent.yml" <<'YML'
services:
  probe:
    image: busybox
    environment:
    - KV_OFFLOAD_DISK
    - "ENABLE_THINKING=${ENABLE_THINKING:-true}"
    command: ["sh", "-c", "echo $${KV_OFFLOAD_DISK:-0} $${ENABLE_THINKING}"]
YML
cat > "$TMP/fx/empty-forward.yml" <<'YML'
services:
  probe:
    image: busybox
    environment:
      - KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-}
    command: ["sh", "-c", "echo $${KV_OFFLOAD_GB-unset}"]
YML
cat > "$TMP/fx/merge-key.yml" <<'YML'
x-knobs: &knobs
  SPEC_N:
services:
  probe:
    image: busybox
    environment:
      <<: *knobs
      OTHER: x
    command: ["true"]
YML

# The ratchet's detector must see a fixture's empty forward.
fx_ef="$(empty_forwards "$TMP/fx")"
[[ "$fx_ef" == *"empty-forward.yml:5:KV_OFFLOAD_GB" && "$(printf '%s\n' "$fx_ef" | wc -l)" -eq 1 ]] \
  && ok "the empty-forward detector finds the fixture's one \${KV_OFFLOAD_GB:-} forward" \
  || bad "the empty-forward detector missed the fixture (got: ${fx_ef:-nothing})"

python3 - "$ROOT" "$TMP/fx" <<'PY' || bad "a self-test failed (above)"
import copy, sys
from pathlib import Path
root, fx = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
from scripts.lib.profiles import launch_knobs as lk, launch_knobs_check as ck
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY

cat = lk.load_catalogue()
names = lk.knob_names(cat)
fails = 0


def expect(cond, msg):
    global fails
    print(("  ✓ " if cond else "  ✗ ") + msg, file=sys.stdout if cond else sys.stderr)
    fails += not cond


def claims(f):
    return sorted(n for n, u in lk.scan_compose(fx / f, names).items() if u.consumed)


# --- the scan -------------------------------------------------------------------
expect(claims("good.yml") == ["REASONING_EFFORT", "SPEC_N"],
       f"scan: good.yml claims the forwarded + interpolated knobs only, not comment mentions ({claims('good.yml')})")
expect(claims("same-indent.yml") == ["ENABLE_THINKING", "KV_OFFLOAD_DISK"],
       f"scan: an environment list at the key's own indent still counts ({claims('same-indent.yml')})")
dr = lk.scan_compose(fx / "dead-read.yml", names)["KV_OFFLOAD_GB"]
expect(dr.dead_read and not dr.consumed, "scan: $${KNOB} read without a forward is a dead read, not a claim")
pin = lk.scan_compose(fx / "pinned.yml", names)["ENABLE_THINKING"]
expect(pin.env_forms() == ["pinned"] and not pin.consumed, "scan: a constant environment value is pinned, not a claim")
try:
    lk.scan_compose(fx / "merge-key.yml", names)
    expect(False, "scan: a YAML merge key in environment: must be refused, not silently skipped")
except ValueError:
    expect(True, "scan: a YAML merge key in environment: is refused, not silently under-claimed")
expect(lk.value_error(cat["knobs"]["SPEC_N"], cat["knobs"]["SPEC_N"]["variants"][0], "") is not None,
       "values: an empty string is never a value (unset the knob instead)")
expect(lk.dependency_errors(cat, {"KV_OFFLOAD_DISK_GB": "3", "KV_OFFLOAD_DISK": "1"}) != [],
       "values: KV_OFFLOAD_DISK_GB + KV_OFFLOAD_DISK=1 without KV_OFFLOAD_GB breaks the chain")


def fx_consumer(f):
    text = (fx / f).read_text(encoding="utf-8")
    return ck.Consumer(f"fixture/{f}", str(fx / f), "vllm", "fixture-model", text, lk.scan_compose_text(text, names))


cov = ck.check_coverage(cat, [fx_consumer("dead-read.yml"), fx_consumer("pinned.yml")])
expect(any("dead-read.yml reads $${KV_OFFLOAD_GB}" in e for e in cov), "coverage: flags the dead read")
expect(any("pinned.yml pins it to a constant" in e for e in cov), "coverage: flags the pinned knob")
efe, _ = ck.check_empty_forwards([fx_consumer("empty-forward.yml")])
expect(any("empty-forward.yml forwards it as KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-}" in e for e in efe),
       "empty forwards: a colon-less read of an empty-forwarded knob fails")

# --- delivery ---------------------------------------------------------------------
if ck.docker_compose_available():
    good = lk.scan_compose(fx / "good.yml", names)
    errs, _ = ck.check_delivery(root, cat, [(str(fx / "good.yml"), good),
                                            (str(fx / "same-indent.yml"), lk.scan_compose(fx / "same-indent.yml", names))])
    expect(errs == [], f"delivery: the positive controls deliver through both channels ({errs[:2]})")
    errs, _ = ck.check_delivery(root, cat, [(str(fx / "undelivered.yml"), lk.scan_compose(fx / "undelivered.yml", names))])
    expect(sum("REASONING_EFFORT is interpolated" in e for e in errs) == 2
           and sum("SPEC_N is interpolated" in e for e in errs) == 2,
           "delivery: knobs interpolated into labels / an x- field fail on both channels (claimed, never delivered)")
    doctored = dict(good)
    doctored["REASONING_EFFORT"] = lk.KnobUse("REASONING_EFFORT")
    errs, _ = ck.check_delivery(root, cat, [(str(fx / "good.yml"), doctored)])
    expect(any("docker delivers REASONING_EFFORT but the scan does not claim it" in e for e in errs),
           "delivery: a knob docker delivers but the scan missed fails (under-claim)")
else:
    print("  - delivery self-tests: SKIP (docker compose unavailable)")

# --- broken catalogue copies: each must make its check fail ------------------------
consumers = ck.load_consumers(root, cat, COMPOSE_REGISTRY)


def only(c, *keep):
    c["knobs"] = {k: v for k, v in c["knobs"].items() if k in keep}
    return c


def var(c, knob, engine, model=None):
    for v in c["knobs"][knob]["variants"]:
        m = v["match"]
        if engine in m.get("engine", [engine]) and (model is None or model in m.get("model", [model])):
            return v
    raise LookupError((knob, engine, model))


def pop_glm(c):
    c["knobs"]["REASONING_EFFORT"]["variants"].remove(var(c, "REASONING_EFFORT", "llamacpp", "glm-5.3-flash"))


def break_from(c):
    var(c, "SPEC_N", "vllm")["checks"][0]["from"] = "^NO SUCH LINE$"


MUTATIONS = [  # (check, label, knobs kept, mutate, the error text that must appear)
    ("domains", "SGLang REASONING_EFFORT gains 'high' (the compose refuses it at boot)", ("REASONING_EFFORT",),
     lambda c: var(c, "REASONING_EFFORT", "sglang")["values"].append("high"), "reject 'high': the catalogue accepts it"),
    ("domains", "KV_OFFLOAD_DISK loses its requires rule", ("KV_OFFLOAD_DISK", "KV_OFFLOAD_GB"),
     lambda c: c["knobs"]["KV_OFFLOAD_DISK"].pop("requires"), "reject {'KV_OFFLOAD_DISK': '1'}: the catalogue accepts it"),
    ("domains", "SGLang KV_OFFLOAD_GB loses its 1-GB-per-GPU minimum", ("KV_OFFLOAD_GB",),
     lambda c: var(c, "KV_OFFLOAD_GB", "sglang").pop("min"), "reject '1': the catalogue accepts it"),
    ("domains", "vLLM KV_OFFLOAD_GB claims enforced=request though the compose refuses at boot", ("KV_OFFLOAD_GB",),
     lambda c: var(c, "KV_OFFLOAD_GB", "vllm").update(enforced="request"), "the compose validates now"),
    ("domains", "vLLM ENABLE_THINKING accepts 'on' (it breaks the JSON splice)", ("ENABLE_THINKING",),
     lambda c: var(c, "ENABLE_THINKING", "vllm", "qwen3.8-27b")["values"].append("on"), "reject 'on': the catalogue accepts it"),
    ("domains", "REASONING_EFFORT evidence no longer in the template", ("REASONING_EFFORT",),
     lambda c: var(c, "REASONING_EFFORT", "vllm")["evidence"][0].update(contains="{%- if nothing == 'here' %}"),
     "no longer contains"),
    ("domains", "a check's fragment anchor matches nothing", ("SPEC_N",), break_from, "cannot run the compose check"),
    ("coverage", "REASONING_EFFORT drops llamacpp from engines", ("REASONING_EFFORT",),
     lambda c: c["knobs"]["REASONING_EFFORT"]["engines"].remove("llamacpp"), "!= the engines whose composes read it"),
    ("coverage", "the GLM REASONING_EFFORT variant is removed", ("REASONING_EFFORT",), pop_glm, "is covered by no variant"),
    ("coverage", "SGLang ENABLE_THINKING default flipped to false", ("ENABLE_THINKING",),
     lambda c: var(c, "ENABLE_THINKING", "sglang").update(default="false"), "the catalogue says 'false'"),
    ("coverage", "SPEC_N requires text no compose carries", ("SPEC_N",),
     lambda c: var(c, "SPEC_N", "vllm")["every_consumer_contains"].append("NOT-IN-ANY-COMPOSE"), "lacks 'NOT-IN-ANY-COMPOSE'"),
]
for check, label, keep, mutate, needle in MUTATIONS:
    c = only(copy.deepcopy(cat), *keep)
    mutate(c)
    errs = ck.check_domains(root, c, consumers)[0] if check == "domains" else ck.check_coverage(c, consumers)
    expect(any(needle in e for e in errs), f"mutation caught ({check}): {label}")
c = copy.deepcopy(cat)
del c["knobs"]["REASONING_EFFORT"]["description"]
c["knobs"]["SPEC_N"]["variants"][0]["checks"][0]["kind"] = "regex"
expect(len(lk.catalogue_errors(c)) == 2, "mutation caught: schema (missing description, unknown check kind)")
raise SystemExit(1 if fails else 0)
PY

if (( fails > 0 )); then
  echo "test-launch-knobs: $fails failure(s)" >&2
  exit 1
fi
echo "test-launch-knobs: ok"
