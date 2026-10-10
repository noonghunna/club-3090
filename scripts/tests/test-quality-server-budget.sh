#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-quality-server-budget — guards the wrapper side of the server-side
# reasoning budget chosen by effort (scripts/lib/effort_budget.py):
#
#   quality-test.sh reads the serving container's boot log (`docker logs`, the
#   LAST `[effort-budget] v1` line — the log spans restarts) and, on a thinking
#   leg, raises the DEFAULT thinking cap to budget + headroom so the client cap
#   cannot cut off the answer after the server's forced close. It records the
#   budget on the Quality: line and, when benchlocal-cli advertises it, with
#   --server-thinking-budget.
#
# Each property below would fail SILENTLY if it broke:
#   1. budget found → cap raised (36864 for 32768), stamped, recorded; the line is
#      read from STDERR too (the compose prints it there) — `docker logs 2>&1`.
#   2. the LAST readback line wins (an earlier boot's budget must not be used).
#   3. untouched: an explicit THINKING_MAX_TOKENS / --thinking-max-tokens, and
#      --thinking-budget (whose own per-request budget wins — not recorded).
#   4. unknown / off / CONTAINER=none / --no-thinking / REASONING_EFFORT=none →
#      the default cap, nothing recorded as a number.
#   5. --resume never passes --server-thinking-budget (benchlocal restores the
#      original record and refuses a different value); --pack-budgets keeps
#      benchlocal's caps but still records the budget.
#   6. --server-thinking-budget only when `benchlocal-cli run --help` lists it.
#   7. a malformed readback warns and leaves the cap alone (never aborts a run).
#
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FAILED=0
fail_case() { echo "ASSERTION FAILED [$1]: $2" >&2; FAILED=1; }
assert_contains() {
  local label="$1" haystack="$2" needle="$3"
  [[ "$haystack" == *"$needle"* ]] || { fail_case "$label" "expected to contain: $needle"; echo "--- got ---" >&2; echo "$haystack" >&2; }
}
assert_not_contains() {
  local label="$1" haystack="$2" needle="$3"
  [[ "$haystack" != *"$needle"* ]] || { fail_case "$label" "expected NOT to contain: $needle"; echo "--- got ---" >&2; echo "$haystack" >&2; }
}

tmp_bin="$(mktemp -d)"
tmp_work="$(mktemp -d)"
ARGS_LOG="${tmp_work}/benchlocal-args.log"
before_list="$(mktemp)"
after_list="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before_list" || true
cleanup() {
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$after_list" || true
  comm -13 "$before_list" "$after_list" | xargs -r rm -f
  rm -rf "$tmp_bin" "$tmp_work"
  rm -f "$before_list" "$after_list"
}
trap cleanup EXIT

# ---- mocks -------------------------------------------------------------------
cat > "${tmp_bin}/curl" <<'MOCK_CURL'
#!/usr/bin/env bash
for arg in "$@"; do
  case "$arg" in
    */v1/models) printf '{"data":[{"id":"mock-model","owned_by":"vllm"}]}'; exit 0 ;;
    */server_info|*/get_server_info|*/props|*/get_model_info) exit 22 ;;
  esac
done
exit 0
MOCK_CURL

# docker: one serving container, vllm-mock. `logs` replays DOCKER_MOCK_LOGS — on
# STDERR when DOCKER_MOCK_LOGS_STDERR=1, which is where the compose entrypoint
# prints the readback line (so a wrapper without 2>&1 would never see it).
cat > "${tmp_bin}/docker" <<'MOCK_DOCKER'
#!/usr/bin/env bash
case "${1:-}" in
  inspect)
    [[ "${2:-}" == "vllm-mock" ]] || exit 1
    if [[ "${3:-}" == "--format" ]]; then
      case "${4:-}" in
        *RestartCount*) echo 0 ;;
        *Config.Image*) echo "vllm/vllm-openai:v0.31.0" ;;
        *) echo "" ;;
      esac
      exit 0
    fi
    printf '[{"Name":"/vllm-mock","Config":{"Image":"vllm/vllm-openai:v0.31.0","Entrypoint":["vllm","serve"],"Cmd":["/m","--reasoning-parser","qwen3"],"Env":[]},"State":{"RestartCount":0},"HostConfig":{}}]\n'
    exit 0 ;;
  logs)
    [[ "${2:-}" == "vllm-mock" ]] || exit 1
    if [[ -n "${DOCKER_MOCK_LOGS:-}" && -f "${DOCKER_MOCK_LOGS}" ]]; then
      if [[ "${DOCKER_MOCK_LOGS_STDERR:-0}" == "1" ]]; then cat "${DOCKER_MOCK_LOGS}" >&2; else cat "${DOCKER_MOCK_LOGS}"; fi
    fi
    exit 0 ;;
  ps) exit 0 ;;
esac
exit 1
MOCK_DOCKER

# benchlocal-cli: `run --help` lists --server-thinking-budget only when
# BL_MOCK_SERVER_BUDGET=1 (lane-B benchlocal); a run records the argv and writes
# a results JSON that carries server_thinking_budget the way that benchlocal does.
cat > "${tmp_bin}/benchlocal-cli" <<'MOCK_BENCHLOCAL'
#!/usr/bin/env bash
for a in "$@"; do
  if [[ "$a" == "--help" ]]; then
    echo "--reasoning-effort --progress"
    [[ "${BL_MOCK_SERVER_BUDGET:-0}" == "1" ]] && echo "--server-thinking-budget N"
    exit 0
  fi
done
printf '%s\n' "$@" >> "${BENCHLOCAL_ARGS_LOG}"
json_out=""; tm="pack-defaults"; stb="null"; prev=""
for a in "$@"; do
  case "$prev" in
    --save-json) json_out="$a" ;;
    --server-thinking-budget) stb="$a" ;;
  esac
  case "$a" in
    --enable-thinking) tm="force-on" ;;
    --no-thinking) tm="force-off" ;;
  esac
  prev="$a"
done
[[ "$tm" == "force-off" ]] && stb="null"
if [[ -n "$json_out" ]]; then
  mkdir -p "$(dirname "$json_out")"
  printf '{"thinking_mode":"%s","sampling_source":"server","server_thinking_budget":%s,"packs":[{"pack_id":"toolcall-15","status":"ok","passed":14,"total":15,"score":0.933,"version":"1.0.1"}]}\n' "$tm" "$stb" > "$json_out"
fi
exit 0
MOCK_BENCHLOCAL
chmod +x "${tmp_bin}"/*

# ---- readback fixtures (the line shell-env prints; the module owns its format) --
EB=scripts/lib/effort_budget.py
eb_line() {  # eb_line LOW MEDIUM XHIGH DEFAULT_EFFORT → one readback line
  python3 "$EB" shell-env --engine vllm --default-effort "$4" --low "$1" --medium "$2" --xhigh "$3" 2>&1 >/dev/null
}
L_BIG="$(eb_line 8192 16384 32768 xhigh)"
L_SMALL="$(eb_line 4096 8192 16384 low)"
L_OFF="$(THINKING_BUDGETS=off python3 "$EB" shell-env --engine vllm --default-effort low --low 1 --medium 2 --xhigh 3 2>&1 >/dev/null)"
[[ "$L_BIG" == "[effort-budget] v1 map="*"default_effort=xhigh floor=32768 engine=vllm" ]] || { echo "fixture: unexpected readback line: $L_BIG" >&2; exit 1; }
[[ "$L_OFF" == "[effort-budget] v1 off engine=vllm" ]] || { echo "fixture: unexpected off line: $L_OFF" >&2; exit 1; }
printf '%s\n' "INFO starting" "$L_BIG" "INFO serving" > "${tmp_work}/big.log"
printf '%s\n' "INFO starting" "$L_SMALL" > "${tmp_work}/small.log"
printf '%s\n' "INFO first boot" "$L_SMALL" "INFO restarted" "$L_BIG" > "${tmp_work}/restart.log"
printf '%s\n' "INFO first boot" "$L_BIG" "INFO restarted" "$L_SMALL" > "${tmp_work}/restart-rev.log"
printf '%s\n' "$L_OFF" > "${tmp_work}/off.log"
printf '%s\n' "INFO no budget line in this boot" > "${tmp_work}/none.log"
printf '%s\n' '[effort-budget] v1 map={"low":"lots"} default_effort=low floor=1 engine=vllm' > "${tmp_work}/bad.log"
printf '{}' > "${tmp_work}/prior.json"

# ---- runner --------------------------------------------------------------------
# run_wrapper [KEY=VAL ...] -- <wrapper args>
run_wrapper() {
  local extra_env=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do extra_env+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift
  : > "$ARGS_LOG"
  set +e
  # A developer shell's knobs must not leak in (env -u goes before the assignments).
  env -u THINKING_MAX_TOKENS -u THINKING_BUDGET -u REASONING_EFFORT -u PACK_BUDGETS -u THINKING_BUDGET_HEADROOM \
      -u ENABLE_THINKING -u NO_THINKING -u MAX_TOKENS \
      PATH="${tmp_bin}:$PATH" BENCHLOCAL_ARGS_LOG="$ARGS_LOG" \
      PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model CONTAINER=vllm-mock \
      ${extra_env[@]+"${extra_env[@]}"} \
      bash scripts/quality-test.sh "$@" > "${tmp_work}/run.out" 2>&1
  RUN_RC=$?
  set -e
  OUT="$(cat "${tmp_work}/run.out")"
  ARGS="$(tr '\n' ' ' < "$ARGS_LOG")"
  QLINE="$(command grep -m1 '^Quality:   ' "${tmp_work}/run.out" || true)"
}
arg_after() { awk -v f="$1" '$0 == f { getline; v = $0 } END { print v }' "$ARGS_LOG"; }
count_arg() { command grep -c -x -- "$1" "$ARGS_LOG" || true; }
expect_cap() {  # expect_cap LABEL N — exactly one --thinking-max-tokens, = N
  local label="$1" want="$2"
  [[ "$RUN_RC" == "0" ]] || { fail_case "$label" "wrapper exited $RUN_RC"; echo "$OUT" >&2; return; }
  [[ "$(count_arg --thinking-max-tokens)" == "1" && "$(arg_after --thinking-max-tokens)" == "$want" ]] \
    || fail_case "$label" "expected --thinking-max-tokens $want, got '$(arg_after --thinking-max-tokens)' (x$(count_arg --thinking-max-tokens))"
}

echo "--- 1. budget found on a thinking leg → cap raised, stamped, recorded ---"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" BL_MOCK_SERVER_BUDGET=1 -- --quick --enable-thinking
expect_cap 1 36864
assert_contains 1 "$OUT" "server budget 32768 reasoning tokens (the server's default effort, xhigh), read from vllm-mock's boot log — thinking cap raised to 36864 (= 32768 + 4096 answer headroom"
assert_contains 1 "$OUT" "thinking-max-tokens=server-budget+4096"
[[ "$(arg_after --server-thinking-budget)" == "32768" ]] || fail_case 1 "--server-thinking-budget was '$(arg_after --server-thinking-budget)'"
assert_contains 1 "$QLINE" ", sampling=server, budget=server 32768 (effort xhigh), packs tc1.0.1, "
# the readback line on STDERR — where the entrypoint prints it
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" DOCKER_MOCK_LOGS_STDERR=1 -- --quick --enable-thinking
expect_cap 1-stderr 36864
# the run's own effort picks the map entry (aliases folded by the module)
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" REASONING_EFFORT=low BL_MOCK_SERVER_BUDGET=1 -- --quick --enable-thinking
expect_cap 1-low 16384
assert_contains 1-low "$OUT" "server budget 8192 reasoning tokens (effort low)"
assert_contains 1-low "$OUT" "the default thinking cap 16384 already leaves ≥ 4096 tokens for the answer"
[[ "$(arg_after --server-thinking-budget)" == "8192" ]] || fail_case 1-low "--server-thinking-budget was '$(arg_after --server-thinking-budget)'"
assert_contains 1-low "$QLINE" "budget=server 8192 (effort low)"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/small.log" REASONING_EFFORT=high -- --quick --enable-thinking
expect_cap 1-alias 20480
assert_contains 1-alias "$QLINE" "budget=server 16384 (effort high)"
# pack-default thinking (no gate forced) is a thinking-capable leg too
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" -- --quick
expect_cap 1-packdefault 36864
# THINKING_BUDGET_HEADROOM is the same knob --thinking-budget uses
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" THINKING_BUDGET_HEADROOM=2048 -- --quick --enable-thinking
expect_cap 1-headroom 34816
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" THINKING_BUDGET_HEADROOM=abc -- --quick --enable-thinking
[[ "$RUN_RC" == "2" && ! -s "$ARGS_LOG" ]] || fail_case 1-badheadroom "a bad THINKING_BUDGET_HEADROOM must refuse before running (rc=$RUN_RC)"

echo "--- 2. the LAST readback line wins (docker logs span restarts) ---"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/restart.log" -- --quick --enable-thinking
expect_cap 2 36864
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/restart-rev.log" -- --quick --enable-thinking
expect_cap 2-rev 16384
assert_contains 2-rev "$OUT" "server budget 4096 reasoning tokens (the server's default effort, low)"

echo "--- 3. a choice is never overridden ---"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" THINKING_MAX_TOKENS=20000 BL_MOCK_SERVER_BUDGET=1 -- --quick --enable-thinking
expect_cap 3-env 20000
assert_contains 3-env "$OUT" "your thinking cap 20000 is kept"
assert_contains 3-env "$OUT" "WARN: thinking cap 20000 is not above the server budget 32768"
[[ "$(arg_after --server-thinking-budget)" == "32768" ]] || fail_case 3-env "the budget is still in effect and must still be recorded"
assert_contains 3-env "$QLINE" "budget=server 32768 (effort xhigh)"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" -- --quick --enable-thinking --thinking-max-tokens 40000
expect_cap 3-flag 40000
assert_not_contains 3-flag "$OUT" "WARN: thinking cap"
# --thinking-budget: the per-request budget wins on the server, so the server's is not the one in effect
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" BL_MOCK_SERVER_BUDGET=1 -- --quick --enable-thinking --thinking-budget 8192
expect_cap 3-tb 12288
assert_contains 3-tb "$OUT" "--thinking-budget 8192 is sent per request and wins over it"
[[ "$(count_arg --server-thinking-budget)" == "0" ]] || fail_case 3-tb "--server-thinking-budget must not be sent when --thinking-budget overrides the server"
assert_not_contains 3-tb "$QLINE" "budget=server"

echo "--- 4. nothing to apply → the default cap, no number recorded ---"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/none.log" BL_MOCK_SERVER_BUDGET=1 -- --quick --enable-thinking
expect_cap 4-unknown 16384
assert_contains 4-unknown "$OUT" "server budget: unknown (no [effort-budget] v1 line in vllm-mock's boot log) — thinking cap unchanged"
[[ "$(count_arg --server-thinking-budget)" == "0" ]] || fail_case 4-unknown "--server-thinking-budget sent for an unknown budget"
assert_not_contains 4-unknown "$QLINE" "budget="
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/off.log" BL_MOCK_SERVER_BUDGET=1 -- --quick --enable-thinking
expect_cap 4-off 16384
assert_contains 4-off "$OUT" "server budget: off (vllm-mock booted with THINKING_BUDGETS=off)"
[[ "$(count_arg --server-thinking-budget)" == "0" ]] || fail_case 4-off "--server-thinking-budget takes a number only"
assert_contains 4-off "$QLINE" ", sampling=server, budget=server off, packs "
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" CONTAINER=none -- --quick --enable-thinking
expect_cap 4-nocontainer 16384
assert_not_contains 4-nocontainer "$OUT" "server budget"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" BL_MOCK_SERVER_BUDGET=1 -- --quick --no-thinking
expect_cap 4-nothinking 16384
assert_not_contains 4-nothinking "$OUT" "server budget"
[[ "$(count_arg --server-thinking-budget)" == "0" ]] || fail_case 4-nothinking "--server-thinking-budget sent on a thinking-off leg"
assert_not_contains 4-nothinking "$QLINE" "budget="
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" REASONING_EFFORT=none -- --quick --enable-thinking
expect_cap 4-effortnone 16384
assert_contains 4-effortnone "$OUT" "server budget: not applied — REASONING_EFFORT=none"

echo "--- 5. --resume never sends it; --pack-budgets records but keeps benchlocal's caps ---"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" BL_MOCK_SERVER_BUDGET=1 -- --resume "${tmp_work}/prior.json"
[[ "$RUN_RC" == "0" ]] || { fail_case 5-resume "wrapper exited $RUN_RC"; echo "$OUT" >&2; }
[[ "$(count_arg --server-thinking-budget)" == "0" ]] || fail_case 5-resume "--server-thinking-budget sent with --resume (benchlocal refuses a value that differs from its journal)"
[[ "$(count_arg --thinking-max-tokens)" == "0" ]] || fail_case 5-resume "a thinking cap rode along with --resume"
assert_not_contains 5-resume "$OUT" "server budget"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" BL_MOCK_SERVER_BUDGET=1 -- --quick --enable-thinking --pack-budgets
[[ "$RUN_RC" == "0" ]] || { fail_case 5-pack "wrapper exited $RUN_RC"; echo "$OUT" >&2; }
[[ "$(count_arg --thinking-max-tokens)" == "0" ]] || fail_case 5-pack "--pack-budgets must keep benchlocal's own caps"
assert_contains 5-pack "$OUT" "--pack-budgets keeps benchlocal's own caps (not raised)"
[[ "$(arg_after --server-thinking-budget)" == "32768" ]] || fail_case 5-pack "the budget is in effect and must still be recorded"

echo "--- 6. --server-thinking-budget only when benchlocal-cli advertises it ---"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/big.log" BL_MOCK_SERVER_BUDGET=0 -- --quick --enable-thinking
expect_cap 6-old 36864
[[ "$(count_arg --server-thinking-budget)" == "0" ]] || fail_case 6-old "an unknown flag was sent to a benchlocal-cli that does not have it"
assert_contains 6-old "$OUT" "this benchlocal-cli predates --server-thinking-budget"
# the stamp does not depend on benchlocal recording it
assert_contains 6-old "$QLINE" "budget=server 32768 (effort xhigh)"

echo "--- 7. a malformed readback warns and changes nothing ---"
run_wrapper "DOCKER_MOCK_LOGS=${tmp_work}/bad.log" -- --quick --enable-thinking
expect_cap 7 16384
assert_contains 7 "$OUT" "WARN: could not read the server budget from vllm-mock's boot log"
assert_not_contains 7 "$QLINE" "budget="

if [[ "$FAILED" != "0" ]]; then
  echo "FAIL: test-quality-server-budget" >&2
  exit 1
fi
echo "PASS: test-quality-server-budget (server budget readback, auto thinking cap, stamp, benchlocal record)"
