#!/usr/bin/env bash
# test-report-resolved-config — report.sh must say what the ENGINE actually ran,
# and must never hand a secret to a GitHub issue while doing it (club-3090#1265).
#
# WHY THIS TEST EXISTS
# --------------------
# `report.sh --full` profiled the rig exhaustively and captured nothing about the
# engine's configuration, so an unexpected number in a community report was
# unattributable. #1261 is the worked example: a clean report, and the three
# findings that mattered (--mem-fraction-static, --chunked-prefill-size, the
# drafter identity) came only from a boot log the reporter happened to paste.
#
# THE ASSERTIONS HAVE TO DISCRIMINATE, AND MOST NAIVE ONES DO NOT
# ---------------------------------------------------------------
# Two traps this file is written around:
#
#   1. "the report mentions X" can pass PRE-FIX. report.sh has carried a
#      per-GPU wattage since its first commit, which is why the power-state
#      test asserts on interpretation rather than presence. Every assertion
#      here therefore keys on content that exists ONLY if the feature ran: the
#      literal verdict strings, the slug, and a flag's BOTH values.
#
#   2. ⚠️⚠️ "no token appears in the report" ALSO passes pre-fix — trivially,
#      because a pre-fix report never reads .Config.Env at all. A leak test
#      alone therefore proves nothing. So the redaction case asserts, in the
#      same run and against the same fixture: the env block IS present, it DOES
#      carry a known allowlisted knob with its value, AND the planted token is
#      absent. Only the conjunction discriminates — leg 5 is a live negative
#      control that removes the allowlist and proves the leak assertion can
#      actually fail.
#
# Contract asserted here:
#   - a stock slug says so in ONE line, naming the slug;
#   - a modified slug lists exactly which flag differs, with BOTH values;
#   - an override read only INSIDE the container (llama.cpp's SPEC=off, which
#     leaves argv byte-identical) is still caught — argv alone is a false clean;
#   - a hand-rolled compose says "unrecognised", and still prints the flags,
#     because for an unrecognised launch argv is the only evidence there is;
#   - the env allowlist is an ALLOWLIST: a planted token, an unknown var and a
#     credential living inside an allowed prefix family are all withheld, and
#     that holds under --no-redact;
#   - no-docker / daemon-down / missing-container each say so DISTINCTLY, so an
#     empty capture can never read like a clean one;
#   - the section is bounded — it must not grow by the size of the compose.
set -uo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0
# Rendered output is captured to files and only the PATH is echoed on failure:
# fifty assertions each dumping a report is unreadable, and the diagnostic still
# has to survive. RC_KEEP_DUMPS=<dir> keeps them on a green run too, for eyeballing
# the rendered section by hand.
DUMP_DIR="${RC_KEEP_DUMPS:-$(mktemp -d)}"
mkdir -p "$DUMP_DIR"
DUMP_IDX=0
CURRENT_DUMP="(none)"

ok() { PASS=$((PASS+1)); }
no() { echo "FAIL: $1" >&2; FAIL=$((FAIL+1)); }
has()   { if [[ "$2" == *"$3"* ]]; then ok; else no "$1: missing '$3' — output: $CURRENT_DUMP"; fi; }
hasnt() { if [[ "$2" != *"$3"* ]]; then ok; else no "$1: must NOT contain '$3' — output: $CURRENT_DUMP"; fi; }

dump() {  # capture long output to a file; print only its path on failure
  DUMP_IDX=$((DUMP_IDX+1))
  CURRENT_DUMP="${DUMP_DIR}/$(printf '%02d' "$DUMP_IDX")-$1.md"
  printf '%s\n' "$2" > "$CURRENT_DUMP"
}

RENDER="${ROOT_DIR}/scripts/lib/resolved_config.py"
if [[ ! -f "$RENDER" ]]; then
  # Recorded, not fatal: a run against a tree WITHOUT the feature must fail
  # loudly here AND still exercise everything below that can run, rather than
  # exiting on the first gap.
  no "scripts/lib/resolved_config.py is absent — the resolved-config capture does not exist"
fi

# The planted secrets. Fake, obviously shaped, and each one probes a different
# way the allowlist could fail open.
PLANT_HF="hf_PLANTED00000000000000000000000000000000"
PLANT_BARE="club3090-planted-bare-secret-value"
PLANT_PREFIX="sk-plantedinsideallowedprefix00000"

TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

# ===========================================================================
# Layer 1 — the env allowlist as a unit
# ===========================================================================
ENV_IN="$(printf '%s\n' \
  "HUGGING_FACE_HUB_TOKEN=${PLANT_HF}" \
  "HF_TOKEN=${PLANT_HF}" \
  "VLLM_API_KEY=${PLANT_PREFIX}" \
  "LLAMA_ARG_API_KEY=${PLANT_PREFIX}" \
  "MY_COMPANY_SECRET=${PLANT_BARE}" \
  "SOME_UNKNOWN_VAR=${PLANT_BARE}" \
  "MEM_FRACTION=0.82" \
  "CHUNKED_PREFILL_SIZE=2048" \
  "VLLM_USE_DEEP_GEMM=1" \
  "GENESIS_ENABLE_P100=1")"

if [[ -f "$RENDER" ]]; then
  ENV_OUT="$(printf '%s\n' "$ENV_IN" | python3 "$RENDER" env-filter 2>&1)"
else
  ENV_OUT=""
fi
dump "env-filter" "$ENV_OUT"

# It must let the real knobs through — otherwise "nothing leaked" is just
# "nothing was captured", which is the failure mode this whole file guards.
has   "allowlist passes a tuning knob"                "$ENV_OUT" "MEM_FRACTION=0.82"
has   "allowlist passes a second tuning knob"         "$ENV_OUT" "CHUNKED_PREFILL_SIZE=2048"
has   "allowlist passes an allowed VLLM_ prefix var"  "$ENV_OUT" "VLLM_USE_DEEP_GEMM=1"
has   "allowlist passes an allowed GENESIS_ var"      "$ENV_OUT" "GENESIS_ENABLE_P100=1"
# ...and withhold every secret shape.
hasnt "HUGGING_FACE_HUB_TOKEN value withheld"         "$ENV_OUT" "$PLANT_HF"
hasnt "HUGGING_FACE_HUB_TOKEN name withheld"          "$ENV_OUT" "HUGGING_FACE_HUB_TOKEN"
hasnt "HF_TOKEN withheld"                             "$ENV_OUT" "HF_TOKEN"
hasnt "an unknown var is withheld (allowlist, not denylist)" "$ENV_OUT" "SOME_UNKNOWN_VAR"
hasnt "a company secret is withheld"                  "$ENV_OUT" "$PLANT_BARE"
# The prefix families are the fail-open risk: VLLM_ and LLAMA_ARG_ are allowed
# wholesale, and BOTH contain a real credential var upstream.
hasnt "VLLM_API_KEY withheld despite the allowed VLLM_ prefix"       "$ENV_OUT" "VLLM_API_KEY"
hasnt "LLAMA_ARG_API_KEY withheld despite the allowed LLAMA_ARG_ prefix" "$ENV_OUT" "LLAMA_ARG_API_KEY"
hasnt "no planted prefix-family secret value leaks"   "$ENV_OUT" "$PLANT_PREFIX"
has   "withheld vars are counted, not silently dropped" "$ENV_OUT" "# withheld:"

# ===========================================================================
# Layer 2 — end to end through report.sh, against a stubbed docker
# ===========================================================================
source "${ROOT_DIR}/scripts/tests/fixtures/report-harness/report-env.sh"

REAL_DOCKER="$(command -v docker 2>/dev/null || true)"
HAVE_COMPOSE=0
if [[ -n "$REAL_DOCKER" ]] && "$REAL_DOCKER" compose version >/dev/null 2>&1; then
  HAVE_COMPOSE=1
fi

report_env_init "$ROOT_DIR"
# The renderer resolves a slug by matching the container's compose label against
# registry compose_paths UNDER THE REPO ROOT, and report.sh's root is the fake
# one — so the shipped recipes have to be reachable from there.
ln -s "${ROOT_DIR}/models" "${REPORT_FAKE_ROOT}/models"

install_docker_stub() {
  report_docker_stub <<STUB
#!/usr/bin/env bash
# Scripted docker for the resolved-config test. Answers inspect/ps/logs/info
# from fixtures; delegates ONLY \`compose\` to the real binary, because
# \`compose config\` is pure text rendering (no daemon) and stubbing it would
# make the recipe diff a fixture-vs-fixture tautology.
case "\${1:-}" in
  info)    [[ "\${RC_DAEMON_DOWN:-0}" == "1" ]] && exit 1; exit 0 ;;
  compose) [[ -n "${REAL_DOCKER}" ]] || exit 1; shift; exec "${REAL_DOCKER}" compose "\$@" ;;
  ps)      [[ "\${*}" == *"-a"* ]] && exit 0
           [[ -n "\${RC_CONTAINER:-}" ]] && echo "\${RC_CONTAINER}"; exit 0 ;;
  logs)    [[ -n "\${RC_LOGS:-}" && -f "\${RC_LOGS}" ]] && cat "\${RC_LOGS}"; exit 0 ;;
  inspect)
           name="\${2:-}"
           [[ "\$name" == "\${RC_CONTAINER:-}" ]] || exit 1
           [[ -n "\${RC_INSPECT:-}" && -f "\${RC_INSPECT}" ]] || exit 1
           case "\${4:-}" in
             '{{json .}}') cat "\${RC_INSPECT}" ;;
             *) exit 1 ;;
           esac ;;
  exec)    exit 1 ;;
  *)       exit 1 ;;
esac
STUB
}
install_docker_stub

# build_inspect <compose-path> <container> <out.json> [sed-expr] [extra-env...]
#
# Models what the DAEMON ends up holding for a compose-launched container:
# `docker compose config` rendered in a clean env, with `$$` unescaped to `$`.
# That unescaping is written out literally here rather than imported from the
# module under test — it is the MEASURED daemon behaviour (verified on this rig
# against a throwaway compose), so the test states it independently and the code
# has to agree with it.
build_inspect() {
  local compose="$1" container="$2" out="$3" mutate="${4:-}"; shift 4 2>/dev/null || shift $#
  RC_COMPOSE="$compose" RC_NAME="$container" RC_OUT="$out" RC_MUTATE="$mutate" \
  RC_EXTRA_ENV="$(printf '%s\n' "$@")" \
  python3 - <<'PY'
import json, os, re, subprocess, sys
compose = os.environ["RC_COMPOSE"]; name = os.environ["RC_NAME"]
clean = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp")}
p = subprocess.run(["docker", "compose", "--env-file", "/dev/null", "-f", compose,
                    "config", "--format", "json"],
                   capture_output=True, encoding="utf-8", env=clean)
if p.returncode != 0:
    sys.stderr.write(p.stderr); sys.exit(1)
doc = json.loads(p.stdout)
svc = None
for s in doc.get("services", {}).values():
    if s.get("container_name") == name:
        svc = s; break
if svc is None:
    sys.stderr.write("no service with container_name=%s\n" % name); sys.exit(1)


def argv(v):
    if v is None:
        return []
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def unescape(xs):
    return [x.replace("$$", "$") for x in xs]


cmd = unescape(argv(svc.get("command")))
mut = os.environ.get("RC_MUTATE") or ""
if mut:
    old, new = mut.split("=>", 1)
    if not any(old in c for c in cmd):
        sys.stderr.write("mutation source %r not present in Cmd\n" % old); sys.exit(1)
    cmd = [c.replace(old, new) for c in cmd]
env = ["%s=%s" % (k, "" if v is None else v) for k, v in (svc.get("environment") or {}).items()]
env += [e for e in os.environ.get("RC_EXTRA_ENV", "").splitlines() if e.strip()]
env += ["PATH=/usr/local/bin:/usr/bin:/bin", "NVIDIA_VISIBLE_DEVICES=all"]
out = {
    "Config": {
        "Image": svc.get("image"),
        "Entrypoint": unescape(argv(svc.get("entrypoint"))),
        "Cmd": cmd,
        "Env": env,
        "Labels": {
            "com.docker.compose.project.config_files": os.path.abspath(compose),
            "com.docker.compose.service": name,
        },
    },
    "State": {"Status": "running"},
}
with open(os.environ["RC_OUT"], "w", encoding="utf-8") as f:
    json.dump(out, f)
PY
}

run_report() {  # sets REPORT_OUT / REPORT_RC
  # ⚠️ Deliberately does NOT touch errexit. The obvious `set +e … set -e` pair
  # around the call TURNS ERREXIT ON for the rest of the file (it was never on —
  # this file runs `set -uo pipefail`), so the first non-zero command after the
  # first report run aborts the suite mid-way. That looks like "fewer failures",
  # which is the worst possible failure shape for a gate.
  REPORT_OUT="$(
    env PATH="${REPORT_FAKE_BIN}:$PATH" \
        HOME="${REPORT_ENV_DIR}/home" \
        REPORT_FULL_CALIBRATION=0 \
        RC_CONTAINER="${RC_CONTAINER:-}" \
        RC_INSPECT="${RC_INSPECT:-}" \
        RC_LOGS="${RC_LOGS:-}" \
        RC_DAEMON_DOWN="${RC_DAEMON_DOWN:-0}" \
        bash "${REPORT_FAKE_ROOT}/scripts/report.sh" "$@" 2>&1
  )"
  REPORT_RC=$?
}

section_of() {  # isolate the new section so assertions can't match elsewhere
  printf '%s\n' "$1" | awk '/^## Engine configuration \(resolved\)/{f=1} /^## Recent failed boot/{f=0} f'
}

VLLM_COMPOSE="${ROOT_DIR}/models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml"
LLAMA_COMPOSE="${ROOT_DIR}/models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml"
VLLM_SLUG="$(python3 - "$ROOT_DIR" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(sys.argv[1], "scripts/lib/profiles"))
from compose_registry import COMPOSE_REGISTRY
want = "models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml"
print(next((s for s, e in COMPOSE_REGISTRY.items() if e.get("compose_path") == want), ""))
PY
)"
[[ -n "$VLLM_SLUG" ]] || no "could not resolve the fixture compose to a registry slug"

if [[ "$HAVE_COMPOSE" -eq 0 ]]; then
  no "docker compose is unavailable — the recipe-diff legs (2a-2e) could not run. \
This is a SKIP recorded as a failure on purpose: a silently skipped diff test is \
indistinguishable from a passing one."
else

# --- 2a. stock slug: one line, and it names the slug ------------------------
RC_CONTAINER="vllm-qwen36-27b-minimal"
RC_INSPECT="${TMP}/stock.json"; RC_LOGS=""
build_inspect "$VLLM_COMPOSE" "$RC_CONTAINER" "$RC_INSPECT" "" \
  "HUGGING_FACE_HUB_TOKEN=${PLANT_HF}" "MY_COMPANY_SECRET=${PLANT_BARE}"
run_report --container "$RC_CONTAINER"
SEC="$(section_of "$REPORT_OUT")"
dump "stock" "$SEC"
has   "stock: verdict present"        "$SEC" "✅ **Stock recipe**"
has   "stock: names the slug"         "$SEC" "$VLLM_SLUG"
has   "stock: names the recipe path"  "$SEC" "single/autoround-int4/minimal.yml"
hasnt "stock: no spurious flag diff"  "$SEC" "flag(s) differ"
hasnt "stock: no spurious entrypoint diff" "$SEC" "Entrypoint differs"
has   "stock: engine image compared to the engine profile" "$SEC" "engine profile"

# --- 2b. modified slug: exactly what differs, with BOTH values --------------
RC_INSPECT="${TMP}/modified.json"
build_inspect "$VLLM_COMPOSE" "$RC_CONTAINER" "$RC_INSPECT" "0.92=>0.85" \
  "HUGGING_FACE_HUB_TOKEN=${PLANT_HF}"
run_report --container "$RC_CONTAINER"
SEC="$(section_of "$REPORT_OUT")"
dump "modified" "$SEC"
has   "modified: flags-differ banner" "$SEC" "**1 flag(s) differ from the shipped recipe:**"
has   "modified: names the flag"      "$SEC" "--gpu-memory-utilization"
has   "modified: shows the RUNNING value" "$SEC" "0.85"
has   "modified: shows the RECIPE value"  "$SEC" "0.92"
hasnt "modified: does not claim stock"    "$SEC" "✅ **Stock recipe**"

# --- 2c. the false-clean class: an override argv cannot see -----------------
# llama.cpp's composes read SPEC inside the container ($$SPEC), so SPEC=off
# disables the drafter with a byte-identical Cmd. A flags-only diff would call
# this stock.
RC_CONTAINER="llama-cpp-qwen36-27b"
RC_INSPECT="${TMP}/envonly.json"
build_inspect "$LLAMA_COMPOSE" "$RC_CONTAINER" "$RC_INSPECT" "" "SPEC=off"
run_report --container "$RC_CONTAINER"
SEC="$(section_of "$REPORT_OUT")"
dump "env-only-override" "$SEC"
hasnt "env-only override is NOT reported as stock" "$SEC" "✅ **Stock recipe**"
has   "env-only override names the knob"           "$SEC" "\`SPEC\`"
has   "env-only override says argv could not see it" "$SEC" "do not appear in the flags above"

# --- 2d. hand-rolled compose: says so, and still prints the flags -----------
RC_CONTAINER="sgl-myrig"
RC_INSPECT="${TMP}/handrolled.json"
python3 - "$RC_INSPECT" "${TMP}/not-a-recipe.yml" "$PLANT_HF" <<'PY'
import json, sys
out, path, token = sys.argv[1], sys.argv[2], sys.argv[3]
json.dump({
    "Config": {
        "Image": "lmsysorg/sglang:v0.5.19",
        "Entrypoint": ["/bin/bash", "-c"],
        "Cmd": ["exec python3 -m sglang.launch_server --model-path /models/x "
                "--mem-fraction-static 0.72 --tp-size 2\n"],
        "Env": ["HUGGING_FACE_HUB_TOKEN=%s" % token, "MEM_FRACTION=0.72"],
        "Labels": {"com.docker.compose.project.config_files": path},
    },
    "State": {"Status": "running"},
}, open(out, "w", encoding="utf-8"))
PY
run_report --container "$RC_CONTAINER"
SEC="$(section_of "$REPORT_OUT")"
dump "handrolled" "$SEC"
has   "hand-rolled: says unrecognised"            "$SEC" "Unrecognised — not a shipped recipe"
has   "hand-rolled: still prints the flags"       "$SEC" "--mem-fraction-static 0.72"
hasnt "hand-rolled: does not claim stock"         "$SEC" "✅ **Stock recipe**"

# --- 2e. no secret reaches the report, INCLUDING under --no-redact ----------
# ⚠️ On its own "the token is absent" passes pre-fix, because a pre-fix report
# never reads .Config.Env. The first two assertions are what make the third
# mean anything: they prove env WAS captured from this very fixture.
RC_CONTAINER="vllm-qwen36-27b-minimal"
RC_INSPECT="${TMP}/secrets.json"
build_inspect "$VLLM_COMPOSE" "$RC_CONTAINER" "$RC_INSPECT" "0.92=>0.85" \
  "HUGGING_FACE_HUB_TOKEN=${PLANT_HF}" "HF_TOKEN=${PLANT_HF}" \
  "VLLM_API_KEY=${PLANT_PREFIX}" "MY_COMPANY_SECRET=${PLANT_BARE}" \
  "VLLM_USE_DEEP_GEMM=1"
run_report --no-redact --container "$RC_CONTAINER"
SEC="$(section_of "$REPORT_OUT")"
dump "secrets-no-redact" "$SEC"
has   "--no-redact: env block IS rendered"              "$SEC" "**Env (allowlist"
has   "--no-redact: a real allowlisted value IS shown"  "$SEC" "VLLM_USE_DEEP_GEMM=1"
hasnt "--no-redact: HF token absent"                    "$SEC" "$PLANT_HF"
hasnt "--no-redact: prefix-family credential absent"    "$SEC" "$PLANT_PREFIX"
hasnt "--no-redact: unknown company secret absent"      "$SEC" "$PLANT_BARE"
hasnt "--no-redact: token var NAME absent"              "$SEC" "HUGGING_FACE_HUB_TOKEN"

# --- 2f. bounded size -------------------------------------------------------
# The section must not grow by the size of the compose. The recipe it diffs
# against here is ~9 KB of YAML; the section must stay far under that.
SEC_BYTES=${#SEC}
if (( SEC_BYTES > 0 && SEC_BYTES < 8000 )); then ok; else
  no "section is ${SEC_BYTES} bytes — expected a bounded block well under the compose it diffs (dump: $CURRENT_DUMP)"
fi

# --- 2g. a live NEGATIVE CONTROL for the leak assertions --------------------
# Everything above would also pass against a renderer that printed NO env at
# all. So: neuter the allowlist in a throwaway copy of the module, point the
# fake root at it, and require a plant to FLIP to leaking. If it does not, 2e
# was asserting nothing.
#
# ⚠️ The probe is PLANT_BARE, not the hf_-shaped one, and the choice is the
# whole point. `hf_…` and `sk-…` are also caught downstream by the value-shape
# scrub, so a neutered ALLOWLIST would still not leak them — probing with one
# would test the scrub and silently leave the allowlist unexercised.
# PLANT_BARE has no recognisable shape, so the allowlist is the ONLY thing
# standing between it and the report. That is exactly why #1265 requires an
# allowlist: a shape-matcher cannot recognise the next secret.
NEUTERED="${TMP}/neutered"
if [[ ! -f "$RENDER" ]]; then
  no "negative control could not run — scripts/lib/resolved_config.py is absent"
else
mkdir -p "${NEUTERED}/scripts"
cp -r "${ROOT_DIR}/scripts/lib" "${NEUTERED}/scripts/lib"
cp "${ROOT_DIR}/scripts/report.sh" "${NEUTERED}/scripts/report.sh"
ln -s "${ROOT_DIR}/models" "${NEUTERED}/models"
python3 - "${NEUTERED}/scripts/lib/resolved_config.py" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p, encoding="utf-8").read()
src = src.replace("def env_allowed(name):", "def env_allowed(name):\n    return True  # NEUTERED", 1)
open(p, "w", encoding="utf-8").write(src)
PY
NEUTERED_OUT="$(env PATH="${REPORT_FAKE_BIN}:$PATH" HOME="${REPORT_ENV_DIR}/home" \
  REPORT_FULL_CALIBRATION=0 RC_CONTAINER="$RC_CONTAINER" RC_INSPECT="$RC_INSPECT" \
  RC_DAEMON_DOWN=0 bash "${NEUTERED}/scripts/report.sh" --no-redact --container "$RC_CONTAINER" 2>&1)"
NEUTERED_SEC="$(section_of "$NEUTERED_OUT")"
dump "negative-control-neutered-allowlist" "$NEUTERED_SEC"
if [[ "$NEUTERED_SEC" == *"$PLANT_BARE"* ]]; then ok; else
  no "negative control did not leak with the allowlist removed — the 2e leak assertions are not discriminating (dump: $CURRENT_DUMP)"
fi
fi  # RENDER present

fi  # HAVE_COMPOSE

# ===========================================================================
# Layer 3 — the failure paths must be DISTINCT, never an empty-looking block
# ===========================================================================
# "If this silently did nothing, would the output differ?" — each of these
# renders a different sentence, so a reader can tell "nothing to report" from
# "could not look".

# 3a. no container at all
RC_CONTAINER=""; RC_INSPECT=""; RC_DAEMON_DOWN=0
run_report
SEC="$(section_of "$REPORT_OUT")"
dump "no-container" "$SEC"
has "no container: says so explicitly" "$SEC" "No engine container resolved"

# 3b. a named container docker does not know
RC_CONTAINER="ghost-container"; RC_INSPECT=""
run_report --container "definitely-not-running"
SEC="$(section_of "$REPORT_OUT")"
dump "missing-container" "$SEC"
has   "missing container: says it could not inspect" "$SEC" "Could not inspect"
hasnt "missing container: does not claim stock"      "$SEC" "✅ **Stock recipe**"

# 3c. docker daemon down
RC_DAEMON_DOWN=1; RC_CONTAINER=""
run_report
SEC="$(section_of "$REPORT_OUT")"
dump "daemon-down" "$SEC"
has "daemon down: says so explicitly" "$SEC" "docker daemon unreachable"
RC_DAEMON_DOWN=0

# 3d. no docker binary at all
printf '#!/usr/bin/env bash\nexit 127\n' > "${REPORT_FAKE_BIN}/docker"
chmod +x "${REPORT_FAKE_BIN}/docker"
run_report
SEC="$(section_of "$REPORT_OUT")"
dump "docker-broken" "$SEC"
if [[ "$SEC" == *"docker daemon unreachable"* || "$SEC" == *"docker not available"* ]]; then ok; else
  no "broken docker: expected an explicit unavailable/unreachable line (dump: $CURRENT_DUMP)"
fi
hasnt "broken docker: does not claim stock" "$SEC" "✅ **Stock recipe**"
install_docker_stub

# 3e. host engine build — CONTAINER=none must not inspect a phantom
RC_CONTAINER="none"; RC_INSPECT=""
run_report --container none
SEC="$(section_of "$REPORT_OUT")"
dump "container-none" "$SEC"
has   "CONTAINER=none: says there is no container" "$SEC" "Host engine build"
hasnt "CONTAINER=none: does not claim stock"       "$SEC" "✅ **Stock recipe**"

# 3f. the help text advertises the opt-in dump flag
HELP_OUT="$(bash "${REPORT_FAKE_ROOT}/scripts/report.sh" --help 2>&1)"
dump "help" "$HELP_OUT"
has "help mentions --engine-args" "$HELP_OUT" "--engine-args"

report_env_cleanup

echo "----------------------------------------"
echo "PASS: $PASS  FAIL: $FAIL"
if [[ "$FAIL" -ne 0 ]]; then
  echo "Saved output: $DUMP_DIR" >&2
  exit 1
fi
[[ -n "${RC_KEEP_DUMPS:-}" ]] || rm -rf "$DUMP_DIR"
echo "OK: report.sh resolved-config capture + env allowlist"
