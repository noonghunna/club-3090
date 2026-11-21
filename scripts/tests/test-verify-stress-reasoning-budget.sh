#!/usr/bin/env bash
# test-verify-stress-reasoning-budget.sh — verify-stress widens its short reply budgets
# for a model that cannot stop reasoning, by the same test verify-full uses.
#
# WHY: the needle checks reply in 30 tokens and the tool-prefill summary in 500. A
# model that reasons under every switch needs room for the reasoning first.
# verify-stress widened only on THINK_ALWAYS_ON (a dial with no off position), while
# verify-full's TOK_SCALE also widens on `none*` (no working off-switch). On vLLM,
# Inkling-2-Mini is detected as `none`: it kept 30 / 500, every needle "missed" with
# an empty answer and tool-prefill failed finish=length, on a server that recalled
# every needle with room to reason (2026-11-21).
#
# This runs the REAL `stress_set_reasoning_budgets` from scripts/verify-stress.sh
# (extracted, not copied), and checks the tool-prefill request reads its budget.
# NEGATIVE CONTROL: the pre-fix verify-stress has no such function (its inline rule
# keyed on THINK_ALWAYS_ON only), so case 2 below is the one it got wrong.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VS="${VERIFY_STRESS:-$ROOT/scripts/verify-stress.sh}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

awk '/^stress_set_reasoning_budgets\(\) \{/{p=1} p{print} p&&/^\}/{exit}' "$VS" > "$TMP/fn.sh"
command grep -q '^stress_set_reasoning_budgets() {' "$TMP/fn.sh" \
  || { echo "FAIL: could not extract stress_set_reasoning_budgets from $VS" >&2; exit 1; }

fails=0
case_() {   # <name> <want "needle/tool"> [VAR=VAL ...]
  local name=$1 want=$2; shift 2
  local got
  got=$(env -u THINK_CONTROL -u THINK_ALWAYS_ON -u VERIFY_NEEDLE_MAX_TOKENS -u VERIFY_TOOL_PREFILL_MAX_TOKENS "$@" \
        bash -c 'source "'"$TMP/fn.sh"'"; stress_set_reasoning_budgets; echo "$NEEDLE_MAX_TOKENS/$TOOL_PREFILL_MAX_TOKENS"')
  if [[ "$got" == "$want" ]]; then echo "  ok  $name → $got"
  else echo "FAIL: $name → $got, expected $want" >&2; fails=$((fails+1)); fi
}

case_ "1. a working switch (enable_thinking) keeps the tight budgets"        "30/500"    THINK_CONTROL=enable_thinking
case_ "2. ⭐ no working switch (none) widens — the Inkling-on-vLLM case"       "512/2048"  THINK_CONTROL=none
case_ "3. an always-on effort dial widens (as before)"                        "512/2048"  THINK_CONTROL=reasoning_effort THINK_ALWAYS_ON=1
case_ "4. a working effort dial keeps the tight budgets"                      "30/500"    THINK_CONTROL=reasoning_effort THINK_ALWAYS_ON=0
case_ "5. an overridden none still widens"                                   "512/2048"  "THINK_CONTROL=none (off overridden)"
case_ "6. the env knobs set the widened values"                               "900/3000"  THINK_CONTROL=none VERIFY_NEEDLE_MAX_TOKENS=900 VERIFY_TOOL_PREFILL_MAX_TOKENS=3000

# The tool-prefill request must take its budget from the function, not a literal.
command grep -qF '"max_tokens": int(os.environ.get("TOOL_PREFILL_MAX_TOKENS") or 500),' "$VS" \
  && echo "  ok  7. the tool-prefill request reads TOOL_PREFILL_MAX_TOKENS" \
  || { echo "FAIL: 7. the tool-prefill request does not read TOOL_PREFILL_MAX_TOKENS" >&2; fails=$((fails+1)); }
# …and the function must actually be called before the probes run.
command grep -qE '^stress_set_reasoning_budgets$' "$VS" \
  && echo "  ok  8. verify-stress calls stress_set_reasoning_budgets" \
  || { echo "FAIL: 8. verify-stress never calls stress_set_reasoning_budgets" >&2; fails=$((fails+1)); }

if (( fails > 0 )); then echo "test-verify-stress-reasoning-budget: ${fails} case(s) failed" >&2; exit 1; fi
echo "test-verify-stress-reasoning-budget: all cases passed"
