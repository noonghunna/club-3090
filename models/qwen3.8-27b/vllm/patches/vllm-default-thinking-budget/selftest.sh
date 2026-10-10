#!/usr/bin/env bash
# selftest.sh [image] — offline proof that patch D (vllm-default-thinking-budget) still applies
# to the pinned vLLM. This is the patch's drift_guard.check in patches.yml: run it at every
# vLLM pin bump.
#
#   bash models/qwen3.8-27b/vllm/patches/vllm-default-thinking-budget/selftest.sh [vllm/vllm-openai:v0.31.0]
#
# No GPU, no network (`docker run --rm --network none`, the image's own python). Legs:
#   main  the compose's own entrypoint line exports the floor (effort_budget.py shell-env,
#         default effort low = 64); install.sh twice (the 2nd a verified no-op that leaves
#         sampling_params.py byte-identical); the AST check (the floor is the statement right
#         after validate_thinking_token_budget in SamplingParams.__post_init__); the AST check
#         FAILS on a copy whose floor lost its trace-replay guard (positive control);
#         selftest_floor.py — the floor on real SamplingParams / Responses / completions /
#         unhooked chat requests, and the real Model-Runner-V2 kernel showing a budget on a
#         thinking-OFF prompt forces nothing (with a thinking-ON positive control).
#   neg   the AST check fails on the pristine file; a non-integer floor refuses before
#         anything is patched; a duplicated and a removed anchor each refuse, naming the
#         anchor and the README's re-anchor steps.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
IMAGE="${1:-vllm/vllm-openai:v0.31.0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../.." && pwd)"

command -v docker >/dev/null 2>&1 || { echo "selftest: docker is required" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "selftest: image $IMAGE is not present locally — docker pull it first" >&2; exit 2; }
[[ -f "$ROOT/scripts/lib/effort_budget.py" ]] || { echo "selftest: $ROOT/scripts/lib/effort_budget.py missing" >&2; exit 2; }

leg() {  # leg <name> <script> — prints the container output, indented
  echo "--- leg: $1 ($IMAGE)"
  docker run --rm --network none --entrypoint bash \
    -v "$HERE:/etc/club3090/default-thinking-budget:ro" \
    -v "$ROOT/scripts/lib/effort_budget.py:/etc/club3090/effort_budget.py:ro" \
    "$IMAGE" -c "$2" 2>&1 | sed 's/^/    /'
}

PRE='set -u
SP="$(python3 -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec(\"vllm\").origin).parent / \"sampling_params.py\")")"
res() { if [ "$2" = 0 ]; then echo "RESULT $1 PASS"; else echo "RESULT $1 FAIL ${3:-}"; fi; }
'

MAIN="$PRE"'
_eb="$(python3 /etc/club3090/effort_budget.py shell-env --engine vllm --default-effort low --low 64 --medium 128 --xhigh 256)" || exit 1; eval "$_eb"
[ "${CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET:-}" = 64 ]; res shell-env-exports-floor $? "got=${CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET:-unset}"
o1="$(bash /etc/club3090/default-thinking-budget/install.sh 2>&1)"; r1=$?
printf "%s\n" "$o1"
printf "%s\n" "$o1" | command grep -q "^\[vllm-default-thinking-budget\] applied:.*floor=64"; res install-first $(( r1 | $? )) "rc=$r1"
n="$(printf "%s\n" "$o1" | command grep -c "^\[vllm-default-thinking-budget\]")"; [ "$n" = 1 ]; res one-status-line $? "lines=$n"
h1="$(sha256sum "$SP")"
o2="$(bash /etc/club3090/default-thinking-budget/install.sh 2>&1)"; r2=$?
printf "%s\n" "$o2"
h2="$(sha256sum "$SP")"
printf "%s\n" "$o2" | command grep -q "already applied"; a=$?
[ "$r2" = 0 ] && [ "$a" = 0 ] && [ "$h1" = "$h2" ]; res install-second-noop $? "rc=$r2"
python3 /etc/club3090/default-thinking-budget/patch_default_thinking_budget.py verify; res ast-floor-placement $?
sed "s/ and not self.trace_decode_token_ids:/:/" "$SP" > /tmp/unguarded.py
! python3 /etc/club3090/default-thinking-budget/patch_default_thinking_budget.py verify /tmp/unguarded.py 2>/dev/null; res ast-catches-unguarded-floor $?
python3 /etc/club3090/default-thinking-budget/selftest_floor.py 2>&1 | command grep -v "^W[0-9]"; res floor-and-thinking-off-inert ${PIPESTATUS[0]}
'

NEG="$PRE"'
cp "$SP" /tmp/pristine.py
! python3 /etc/club3090/default-thinking-budget/patch_default_thinking_budget.py verify "$SP" 2>/dev/null; res ast-fails-on-pristine $?
o="$(CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET=4k bash /etc/club3090/default-thinking-budget/install.sh 2>&1)"; r=$?
printf "%s\n" "$o"
! command grep -q "club3090 vllm-default-thinking-budget" "$SP"; u=$?
[ "$r" != 0 ] && [ "$u" = 0 ] && printf "%s\n" "$o" | command grep -q "REFUSE"; res bad-floor-refuses-before-patching $? "rc=$r"
cp /tmp/pristine.py "$SP"
printf "\nclass _Club3090Dup:\n    def f(self):\n        self.thinking_token_budget = validate_thinking_token_budget(\n            self.thinking_token_budget\n        )\n" >> "$SP"
o="$(CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET=64 bash /etc/club3090/default-thinking-budget/install.sh 2>&1)"; r=$?
printf "%s\n" "$o"
[ "$r" != 0 ] && printf "%s\n" "$o" | command grep -q "found 2 times" && printf "%s\n" "$o" | command grep -q "Re-anchoring"; res duplicate-anchor-refuses $? "rc=$r"
cp /tmp/pristine.py "$SP"
sed -i "s/self.thinking_token_budget = validate_thinking_token_budget(/self.thinking_token_budget = _validate_ttb(/" "$SP"
o="$(CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET=64 bash /etc/club3090/default-thinking-budget/install.sh 2>&1)"; r=$?
printf "%s\n" "$o"
[ "$r" != 0 ] && printf "%s\n" "$o" | command grep -q "found 0 times" && printf "%s\n" "$o" | command grep -q "Re-anchoring"; res missing-anchor-refuses $? "rc=$r"
'

out="$( { leg main "$MAIN"; leg neg "$NEG"; } )"
printf '%s\n' "$out"

want=(shell-env-exports-floor install-first one-status-line install-second-noop ast-floor-placement
      ast-catches-unguarded-floor floor-and-thinking-off-inert ast-fails-on-pristine
      bad-floor-refuses-before-patching duplicate-anchor-refuses missing-anchor-refuses)
fails=0
echo "--- summary (vllm-default-thinking-budget selftest, $IMAGE)"
for w in "${want[@]}"; do
  line="$(printf '%s\n' "$out" | command grep -E "RESULT $w (PASS|FAIL)" | tail -1)"
  if [[ "$line" == *"RESULT $w PASS"* ]]; then
    echo "  ✓ $w"
  else
    echo "  ✗ $w ${line:-(no result — the leg died before this check)}" >&2
    fails=$((fails + 1))
  fi
done
if (( fails )); then echo "vllm-default-thinking-budget selftest: $fails failure(s)" >&2; exit 1; fi
echo "vllm-default-thinking-budget selftest: ok (${#want[@]} checks)"
