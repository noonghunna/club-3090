#!/usr/bin/env bash
#
# test-effort-budget — guards scripts/lib/effort_budget.py, the engine-neutral half of the
# reasoning budget chosen by effort (vLLM + SGLang composes mount it at
# /etc/club3090/effort_budget.py; the engine hooks import it; the quality wrapper reads its
# readback line). Everything the engines do with a budget depends on this file, so the
# properties that would fail SILENTLY are pinned here:
#
#   1. The map: knobs override the compose defaults; an EMPTY knob counts as unset; a bad
#      knob is an error, never a quiet fallback; THINKING_BUDGETS=off turns it off;
#      high/max fold to xhigh and minimal to low.
#   2. The effort in use: top-level > chat_template_kwargs > the server's default
#      (reasoning_effort or default_reasoning_effort). Thinking off gets no map budget.
#   3. shell-env: stdout is ONLY shell (it is eval'd), the readback line goes to stderr,
#      a bad knob exits non-zero with nothing on stdout, and "off" still prints something
#      (so `_eb="$(…)" || exit 1; eval "$_eb"` can't mistake failure for success).
#   4. The readback: the LAST v1 line wins (docker logs span restarts); another version
#      tag is ignored.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
EB=scripts/lib/effort_budget.py

fail() { echo "✗ $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }

# Knobs a developer's shell may carry must not leak into the assertions.
unset THINKING_BUDGET_LOW THINKING_BUDGET_MEDIUM THINKING_BUDGET_XHIGH THINKING_BUDGETS REASONING_EFFORT \
      CLUB3090_REASONING_EFFORT_BUDGETS CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET SGLANG_MAX_THINK_TOKENS

echo "--- 1+2+4. library ---"
python3 - "$EB" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("effort_budget", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def raises(fn):
    try:
        fn()
    except m.BudgetConfigError:
        return True
    return False

D = {"low": 4096, "medium": 8192, "xhigh": 16384}
full = {"low": 4096, "minimal": 4096, "medium": 8192, "xhigh": 16384, "high": 16384, "max": 16384}
assert m.build_map({}, D) == full, m.build_map({}, D)
assert m.build_map({"THINKING_BUDGET_LOW": "1024"}, D)["minimal"] == 1024
assert m.build_map({"THINKING_BUDGET_LOW": ""}, D)["low"] == 4096, "an empty knob must count as unset"
assert m.build_map({"THINKING_BUDGET_LOW": "  "}, D)["low"] == 4096
for bad in ("-1", "abc", "1.5", "4k"):
    assert raises(lambda: m.build_map({"THINKING_BUDGET_XHIGH": bad}, D)), bad
assert raises(lambda: m.build_map({}, {"low": 1, "medium": 2})), "a level with no budget must be an error"
for off in ("off", "OFF", "0", "false", "none"):
    assert m.build_map({"THINKING_BUDGETS": off}, D) is None, off
assert m.build_map({"THINKING_BUDGETS": "on"}, D) == full
assert raises(lambda: m.build_map({"THINKING_BUDGETS": "maybe"}, D))
print("  ✓ map: knobs, empty = unset, bad = error, off, aliases")

assert m.parse_map(None) == {} and m.parse_map("") == {}
assert m.parse_map('{"LOW": 5}') == {"low": 5}
for bad in ("[1]", "{bad", '{"low": -1}', '{"low": "x"}', '{"low": true}'):
    assert raises(lambda: m.parse_map(bad)), bad
assert m.budgets_from_env({"CLUB3090_REASONING_EFFORT_BUDGETS": '{"low": 7}'}) == {"low": 7}
assert m.budgets_from_env({}) == {}
print("  ✓ parse_map: strict")

srv_vllm = {"enable_thinking": True, "reasoning_effort": "low"}
srv_sgl = {"enable_thinking": True, "default_reasoning_effort": "low"}
assert m.resolve_effort("xhigh", {"reasoning_effort": "medium"}, srv_vllm) == "xhigh"
assert m.resolve_effort(None, {"reasoning_effort": "Medium"}, srv_vllm) == "medium"
assert m.resolve_effort(None, {}, srv_vllm) == "low"
assert m.resolve_effort(None, None, srv_sgl) == "low"
assert m.resolve_effort("", {"reasoning_effort": " "}, None) is None
print("  ✓ effort: top-level > kwargs > server default (both keys)")

assert m.budget_for(full, "high", None, srv_vllm) == 16384, "alias high -> xhigh"
assert m.budget_for(full, None, {"reasoning_effort": "minimal"}, srv_vllm) == 4096
assert m.budget_for(full, None, None, srv_sgl) == 4096, "no effort sent -> the server default's budget"
assert m.budget_for(full, "none", None, srv_vllm) is None, "effort none = thinking off"
assert m.budget_for(full, "xhigh", {"enable_thinking": False}, srv_vllm) is None, "request turns thinking off"
assert m.budget_for(full, None, {"enable_thinking": True}, {"enable_thinking": False, "reasoning_effort": "low"}) == 4096, \
    "the request's enable_thinking wins over the server's"
assert m.budget_for(full, None, None, {"enable_thinking": "false", "reasoning_effort": "low"}) is None
assert m.budget_for(full, "turbo", None, srv_vllm) is None, "an unmapped effort gets no map budget"
assert m.budget_for({}, "xhigh", None, srv_vllm) is None, "no map, no budget"
assert m.budget_for(full, None, None, None) is None, "nothing to resolve -> no budget (the floor handles it)"
print("  ✓ budget_for: aliases, thinking off, precedence, unmapped")

line = m.readback_line(full, "low", 4096, engine="vllm")
assert line.startswith("[effort-budget] v1 map={") and "default_effort=low floor=4096 engine=vllm" in line, line
info = m.parse_readback("noise\n" + line + "\n")
assert info == {"off": False, "map": full, "default_effort": "low", "floor": 4096}, info
old = m.readback_line({"low": 1, "medium": 2, "xhigh": 3}, "xhigh", 3)
assert m.parse_readback(old + "\n" + line)["floor"] == 4096, "the LAST line must win (docker logs span restarts)"
assert m.parse_readback(line + "\n" + m.readback_line(None))["off"] is True
assert m.parse_readback(line.replace(" v1 ", " v2 ")) is None, "another version tag must be ignored, not misread"
assert m.parse_readback("no budget line here") is None
print("  ✓ readback: round-trip, last line wins, version-gated")
PY

echo "--- 3. shell-env (the entrypoint contract) ---"
ARGS=(shell-env --engine vllm --default-effort low --low 4096 --medium 8192 --xhigh 16384)
out="$(python3 "$EB" "${ARGS[@]}" 2>"${TMPDIR:-/tmp}/eb.err.$$")"; err="$(cat "${TMPDIR:-/tmp}/eb.err.$$")"; rm -f "${TMPDIR:-/tmp}/eb.err.$$"
while IFS= read -r l; do
  [[ "$l" == export\ * ]] || fail "shell-env stdout must be shell only, got: $l"
done <<<"$out"
[[ "$err" == "[effort-budget] v1 map="*"default_effort=low floor=4096 engine=vllm" ]] || fail "readback line on stderr wrong: $err"
pass "stdout is export lines only; the readback line is on stderr"

( _eb="$(python3 "$EB" "${ARGS[@]}" 2>/dev/null)" || exit 1; eval "$_eb"
  [[ "$CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET" == 4096 ]] || exit 3
  python3 -c 'import json,os,sys; m=json.loads(os.environ["CLUB3090_REASONING_EFFORT_BUDGETS"]); sys.exit(0 if m["high"]==16384 else 4)' ) \
  || fail "eval of shell-env did not export the map + vLLM floor"
pass "eval exports the map and the vLLM floor"

( _eb="$(python3 "$EB" shell-env --engine sglang --default-effort high --low 1 --medium 2 --xhigh 3 2>/dev/null)" || exit 1; eval "$_eb"
  [[ "$SGLANG_MAX_THINK_TOKENS" == 3 && -z "${CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET:-}" ]] ) \
  || fail "sglang must export SGLANG_MAX_THINK_TOKENS (= the default effort's budget, aliases folded) and not the vLLM floor"
pass "sglang exports SGLANG_MAX_THINK_TOKENS; default effort 'high' folds to xhigh"

( export THINKING_BUDGET_MEDIUM=""; _eb="$(python3 "$EB" "${ARGS[@]}" 2>/dev/null)" || exit 1; eval "$_eb"
  python3 -c 'import json,os,sys; sys.exit(0 if json.loads(os.environ["CLUB3090_REASONING_EFFORT_BUDGETS"])["medium"]==8192 else 1)' ) \
  || fail "an empty THINKING_BUDGET_MEDIUM must keep the compose default (the compose '- VAR=\${X:-}' trap)"
pass "an empty knob keeps the compose default"

set +e
bad_out="$(THINKING_BUDGET_XHIGH=lots python3 "$EB" "${ARGS[@]}" 2>/dev/null)"; rc=$?
set -e
[[ $rc -ne 0 && -z "$bad_out" ]] || fail "a bad knob must exit non-zero with EMPTY stdout (rc=$rc, out=$bad_out)"
if ( _eb="$(THINKING_BUDGET_XHIGH=lots python3 "$EB" "${ARGS[@]}" 2>/dev/null)" || exit 1; eval "$_eb"; exit 0 ); then
  fail "the entrypoint pattern must stop the boot on a bad knob"
fi
pass "a bad knob stops the boot (non-zero, nothing to eval)"

for args in "--engine vllm --low 1 --medium 2 --xhigh 3" "--engine vllm --default-effort none --low 1 --medium 2 --xhigh 3"; do
  # shellcheck disable=SC2086
  if python3 "$EB" shell-env $args >/dev/null 2>&1; then fail "shell-env $args must fail (no usable default effort)"; fi
done
pass "a missing or unmapped default effort is an error"

off_out="$(THINKING_BUDGETS=off python3 "$EB" "${ARGS[@]}" 2>"${TMPDIR:-/tmp}/eb.err.$$")"; off_err="$(cat "${TMPDIR:-/tmp}/eb.err.$$")"; rm -f "${TMPDIR:-/tmp}/eb.err.$$"
[[ -n "$off_out" ]] || fail "off must still print something on stdout"
[[ "$off_err" == "[effort-budget] v1 off engine=vllm" ]] || fail "off readback wrong: $off_err"
( export CLUB3090_REASONING_EFFORT_BUDGETS=stale CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET=9
  _eb="$(THINKING_BUDGETS=off python3 "$EB" "${ARGS[@]}" 2>/dev/null)" || exit 1; eval "$_eb"
  [[ -z "${CLUB3090_REASONING_EFFORT_BUDGETS:-}" && -z "${CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET:-}" ]] ) \
  || fail "off must unset the map and the floor"
pass "THINKING_BUDGETS=off: stdout non-empty, unsets the map + floor, readback says off"

echo "--- 4. readback-budget (the wrapper's view) ---"
log="$(printf '%s\n' "boot 1" \
  '[effort-budget] v1 map={"high":3,"low":1,"max":3,"medium":2,"minimal":1,"xhigh":3} default_effort=xhigh floor=3 engine=vllm' \
  "restart" \
  '[effort-budget] v1 map={"high":300,"low":100,"max":300,"medium":200,"minimal":100,"xhigh":300} default_effort=low floor=100 engine=vllm')"
[[ "$(python3 "$EB" readback-budget --line "$log")" == 100 ]] || fail "readback-budget must use the LAST line's default effort"
[[ "$(python3 "$EB" readback-budget --line "$log" --effort high)" == 300 ]] || fail "readback-budget --effort high must fold to xhigh"
[[ "$(printf '%s\n' "$log" | python3 "$EB" readback-budget --line - --effort medium)" == 200 ]] || fail "readback-budget must read stdin"
[[ "$(python3 "$EB" readback-budget --line '[effort-budget] v1 off engine=sglang')" == off ]] || fail "off must read as off"
[[ "$(python3 "$EB" readback-budget --line "$log" --effort none)" == none ]] || fail "effort none = thinking off: no budget, not the floor"
set +e; unk="$(python3 "$EB" readback-budget --line "nothing")"; rc=$?; set -e
[[ "$unk" == unknown && $rc -eq 2 ]] || fail "no readback line must be 'unknown' with rc 2 (got $unk / $rc)"
pass "last line wins, aliases, stdin, off, unknown"

echo "✓ test-effort-budget: all checks passed"
