#!/usr/bin/env bash
# selftest.sh [image] — offline proof that patch B (vllm-effort-budget) still hooks the pinned
# vLLM. This is the patch's drift_guard.check in patches.yml: run it at every vLLM pin bump.
#
#   bash models/qwen3.8-27b/vllm/patches/vllm-effort-budget/selftest.sh [vllm/vllm-openai:v0.31.0]
#
# No GPU, no network (`docker run --rm --network none`, the image's own python). Legs:
#   main  install.sh twice (the 2nd a verified no-op that leaves serving.py byte-identical);
#         the AST check (the hook is the statement right before request.to_sampling_params(
#         in OpenAIServingChat._create_chat_completion); the AST check FAILS on a copy with the
#         hook moved after that call (positive control); the glue against vLLM's real
#         ChatCompletionRequest / _effective_chat_template_kwargs / to_sampling_params and the
#         Anthropic converter (selftest_glue.py); the module's own test (test-effort-budget.sh).
#   neg   the AST check fails on the pristine file; a malformed map refuses before anything is
#         patched; a duplicated anchor and a removed anchor each refuse, naming the anchor and
#         the README's re-anchor steps.
# Each leg runs in its own throwaway container, so the negative controls can edit
# site-packages freely.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
IMAGE="${1:-vllm/vllm-openai:v0.31.0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../.." && pwd)"
MAP='{"high":256,"low":64,"max":256,"medium":128,"minimal":64,"xhigh":256}'

command -v docker >/dev/null 2>&1 || { echo "selftest: docker is required" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "selftest: image $IMAGE is not present locally — docker pull it first" >&2; exit 2; }
[[ -f "$ROOT/scripts/lib/effort_budget.py" ]] || { echo "selftest: $ROOT/scripts/lib/effort_budget.py missing" >&2; exit 2; }

leg() {  # leg <name> <script> [docker run args...] — prints the container output, indented
  local name="$1" script="$2"; shift 2
  echo "--- leg: $name ($IMAGE)"
  docker run --rm --network none --entrypoint bash \
    -v "$HERE:/etc/club3090/effort-budget:ro" \
    -v "$ROOT/scripts/lib/effort_budget.py:/etc/club3090/effort_budget.py:ro" \
    -v "$ROOT/scripts:/repo/scripts:ro" \
    "$@" "$IMAGE" -c "$script" 2>&1 | sed 's/^/    /'
}

# Shared prelude: locate serving.py without importing vllm.
PRE='set -u
SERVING="$(python3 -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec(\"vllm\").origin).parent / \"entrypoints/openai/chat_completion/serving.py\")")"
res() { if [ "$2" = 0 ]; then echo "RESULT $1 PASS"; else echo "RESULT $1 FAIL ${3:-}"; fi; }
'

MAIN="$PRE"'
o1="$(bash /etc/club3090/effort-budget/install.sh 2>&1)"; r1=$?
printf "%s\n" "$o1"
printf "%s\n" "$o1" | command grep -q "^\[vllm-effort-budget\] applied:"; res install-first $(( r1 | $? )) "rc=$r1"
n="$(printf "%s\n" "$o1" | command grep -c "^\[vllm-effort-budget\]")"; [ "$n" = 1 ]; res one-status-line $? "lines=$n"
h1="$(sha256sum "$SERVING")"
o2="$(bash /etc/club3090/effort-budget/install.sh 2>&1)"; r2=$?
printf "%s\n" "$o2"
h2="$(sha256sum "$SERVING")"
printf "%s\n" "$o2" | command grep -q "already applied"; a=$?
[ "$r2" = 0 ] && [ "$a" = 0 ] && [ "$h1" = "$h2" ]; res install-second-noop $? "rc=$r2"
python3 /etc/club3090/effort-budget/patch_effort_budget.py verify; res ast-hook-placement $?
python3 - "$SERVING" /tmp/moved.py <<"PY"
import sys
lines = open(sys.argv[1], encoding="utf-8").read().splitlines(keepends=True)
i = next(k for k, l in enumerate(lines) if "apply_effort_budget(" in l)
hook = lines.pop(i)
j = next(k for k, l in enumerate(lines) if l.strip() == "sampling_params = request.to_sampling_params(")
lines.insert(j + 4, hook)   # after the 4-line to_sampling_params(...) call
open(sys.argv[2], "w", encoding="utf-8").writelines(lines)
PY
! python3 /etc/club3090/effort-budget/patch_effort_budget.py verify /tmp/moved.py 2>/dev/null; res ast-catches-moved-hook $?
python3 /etc/club3090/effort-budget/selftest_glue.py 2>&1 | command grep -v "^W[0-9]"; res glue-vs-vllm-request ${PIPESTATUS[0]}
bash /repo/scripts/tests/test-effort-budget.sh; res module-unit-tests $?
'

NEG="$PRE"'
cp "$SERVING" /tmp/pristine.py
! python3 /etc/club3090/effort-budget/patch_effort_budget.py verify "$SERVING" 2>/dev/null; res ast-fails-on-pristine $?
o="$(CLUB3090_REASONING_EFFORT_BUDGETS="{\"low\": -1}" bash /etc/club3090/effort-budget/install.sh 2>&1)"; r=$?
printf "%s\n" "$o"
! command grep -q "club3090 vllm-effort-budget" "$SERVING"; u=$?
[ "$r" != 0 ] && [ "$u" = 0 ] && printf "%s\n" "$o" | command grep -q "REFUSE"; res bad-map-refuses-before-patching $? "rc=$r"
cp /tmp/pristine.py "$SERVING"
printf "\ndef _club3090_dup(request):\n    if True:\n        if True:\n            if True:\n                sampling_params = request.to_sampling_params(\n                    1)\n" >> "$SERVING"
o="$(bash /etc/club3090/effort-budget/install.sh 2>&1)"; r=$?
printf "%s\n" "$o"
[ "$r" != 0 ] && printf "%s\n" "$o" | command grep -q "found 2 times" && printf "%s\n" "$o" | command grep -q "Re-anchoring"; res duplicate-anchor-refuses $? "rc=$r"
cp /tmp/pristine.py "$SERVING"
sed -i "s/sampling_params = request.to_sampling_params(/sampling_params = request.to_sampling_params_moved(/" "$SERVING"
o="$(bash /etc/club3090/effort-budget/install.sh 2>&1)"; r=$?
printf "%s\n" "$o"
[ "$r" != 0 ] && printf "%s\n" "$o" | command grep -q "found 0 times" && printf "%s\n" "$o" | command grep -q "Re-anchoring"; res missing-anchor-refuses $? "rc=$r"
'

out="$( { leg main "$MAIN" -e "CLUB3090_REASONING_EFFORT_BUDGETS=$MAP"; leg neg "$NEG" -e "CLUB3090_REASONING_EFFORT_BUDGETS=$MAP"; } )"
printf '%s\n' "$out"

want=(install-first one-status-line install-second-noop ast-hook-placement ast-catches-moved-hook
      glue-vs-vllm-request module-unit-tests ast-fails-on-pristine bad-map-refuses-before-patching
      duplicate-anchor-refuses missing-anchor-refuses)
fails=0
echo "--- summary (vllm-effort-budget selftest, $IMAGE)"
for w in "${want[@]}"; do
  line="$(printf '%s\n' "$out" | command grep -E "RESULT $w (PASS|FAIL)" | tail -1)"
  if [[ "$line" == *"RESULT $w PASS"* ]]; then
    echo "  ✓ $w"
  else
    echo "  ✗ $w ${line:-(no result — the leg died before this check)}" >&2
    fails=$((fails + 1))
  fi
done
if (( fails )); then echo "vllm-effort-budget selftest: $fails failure(s)" >&2; exit 1; fi
echo "vllm-effort-budget selftest: ok (${#want[@]} checks)"
