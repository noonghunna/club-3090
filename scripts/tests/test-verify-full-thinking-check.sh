#!/usr/bin/env bash
# test-verify-full-thinking-check.sh — verify-full [7/10] must pass a healthy concise
# thinker and still fail the real faults.
#
# WHY: [7/10] asked "What is 2+2?" and FAILED any reasoning under 50 chars. Concise or
# adaptive thinkers answer that with a line of thought or none, so healthy boots scored
# 9/10: ThinkingCap-Qwen3.8 (41 chars, #1418) and MiMo-V2.6 (empty, flaky). The check
# exists to prove thinking engaged and was parsed into its own field; short reasoning
# plus an answer proves exactly that.
#
# ADAPTIVE thinkers (2026-11-13): MiMo-V2.6 decides per request whether to reason, and
# reasoned on the train question 4 of 20 times at temperature 0.3, so the check failed
# most healthy boots. When the easy question gets no reasoning, check_thinking asks one
# harder question (MiMo: 10 of 10) and fails only if that one gets none either.
#
# This runs the REAL `check_thinking` from scripts/verify-full.sh (extracted, not
# copied) with `curl` stubbed to serve each response shape (a second body for the
# second request), and checks the verdict and how many requests were sent.
# NEGATIVE CONTROL: against the pre-fix verify-full the short-reasoning case fails, and
# against the one before the second question the adaptive case fails.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VF="${VERIFY_FULL:-$ROOT/scripts/verify-full.sh}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# The function body, from `check_thinking() {` to its closing `}` at column 0.
awk '/^check_thinking\(\) \{/{p=1} p{print} p&&/^\}/{exit}' "$VF" > "$TMP/check.sh"
command grep -q '^check_thinking() {' "$TMP/check.sh" || { echo "FAIL: could not extract check_thinking from $VF" >&2; exit 1; }

fails=0
mkbody() { python3 -c 'import json,sys; print(json.dumps({"choices":[{"message":{"reasoning_content":sys.argv[1],"content":sys.argv[2]},"finish_reason":sys.argv[3]}]}))' "$1" "$2" "$3"; }
# case <name> <expect pass|fail> <reasoning> <content> <finish> [raw-response]
# BODY2=<json> (env) is what a second request gets; without it every request gets the first body.
case_() {
  local name=$1 want=$2 reasoning=$3 content=$4 finish=$5 raw=${6:-}
  local body
  if [[ -n "$raw" ]]; then body=$raw
  else body=$(mkbody "$reasoning" "$content" "$finish")
  fi
  local out rc=0
  rm -f "$TMP"/req.json.* "$TMP/calls"
  out=$(BODY="$body" BODY2="${BODY2:-}" REQ="$TMP/req.json" CALLS="$TMP/calls" bash -c '
    pass() { echo "PASS: $1"; }
    fail() { echo "FAIL: $1"; return 1; }
    curl() {
      local n=$(( $(cat "$CALLS" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$CALLS"
      while [[ $# -gt 0 ]]; do [[ $1 == -d ]] && { printf "%s" "$2" > "$REQ.$n"; shift; }; shift; done
      if [[ $n -ge 2 && -n "$BODY2" ]]; then printf "%s" "$BODY2"; else printf "%s" "$BODY"; fi
    }
    URL=http://mock MODEL=m THINK_ON_STD="" THINK_ON_KW="{\"enable_thinking\": true}"
    source "'"$TMP/check.sh"'"
    check_thinking' 2>&1) || rc=$?
  local got=pass; [[ $rc -ne 0 || "$out" == *"FAIL: "* ]] && got=fail
  if [[ "$got" == "$want" ]]; then echo "  ok  $name → $got"
  else echo "FAIL: $name → $got, expected $want" >&2; echo "$out" | sed 's/^/      /' >&2; fails=$((fails+1)); fi
}

R41="14:35 + 3h = 17:35, + 13 min = 17:48."   # 37 chars: a concise thinker (ThinkingCap: 41)
R300=$(printf 'step %.0s' {1..60})
case_ "short reasoning + answer (concise thinker)" pass "$R41" "17:48" stop
case_ "long reasoning + answer"                    pass "$R300" "17:48" stop
case_ "reasoning, no answer, hit max_tokens"       pass "$R300" "" length
case_ "EMPTY reasoning on both questions"          fail "" "17:48" stop
case_ "reasoning but no answer, finish=stop"       fail "$R300" "" stop
case_ "unparseable response"                       fail "" "" "" "not json"

# Adaptive thinker: no reasoning on the easy question, reasoning on the harder one.
BODY2=$(mkbody "$R300" "1/2" stop)        case_ "adaptive: empty, then reasoning + answer"        pass "" "17:48" stop
BODY2=$(mkbody "$R300" "" length)         case_ "adaptive: empty, then reasoning hits max_tokens" pass "" "17:48" stop
BODY2=$(mkbody "$R300" "" stop)           case_ "adaptive: empty, then reasoning but no answer"   fail "" "17:48" stop
BODY2="not json"                          case_ "adaptive: empty, then unparseable"               fail "" "17:48" stop

# The easy question must need a few steps: "What is 2+2?" is what made this check flaky.
case_ "(request capture)" pass "$R300" "17:48" stop > /dev/null
if command grep -q '2+2' "$TMP/req.json.1"; then
  echo "FAIL: [7/10] still sends the 2+2 prompt, which concise/adaptive thinkers answer without reasoning" >&2; fails=$((fails+1))
else
  echo "  ok  prompt is not the trivial 2+2"
fi
# A model that reasons on the easy question pays for one request, not two.
if [[ "$(cat "$TMP/calls")" == "1" && ! -e "$TMP/req.json.2" ]]; then
  echo "  ok  reasoning on the easy question sends no second request"
else
  echo "FAIL: reasoning on the easy question still sent $(cat "$TMP/calls") requests" >&2; fails=$((fails+1))
fi
# The second question is a different, harder one, with the same thinking switch.
BODY2=$(mkbody "$R300" "1/2" stop) case_ "(second request capture)" pass "" "17:48" stop > /dev/null
if [[ -e "$TMP/req.json.2" ]] && ! cmp -s "$TMP/req.json.1" "$TMP/req.json.2" \
   && command grep -q 'pipe' "$TMP/req.json.2" && command grep -q 'enable_thinking' "$TMP/req.json.2"; then
  echo "  ok  the second request is the harder question, with thinking on"
else
  echo "FAIL: the second request is missing, the same question, or lost the thinking switch" >&2; fails=$((fails+1))
fi

if [[ $fails -gt 0 ]]; then echo "test-verify-full-thinking-check: $fails FAILED" >&2; exit 1; fi
echo "test-verify-full-thinking-check: PASS — short-but-present reasoning passes; an adaptive thinker passes on the harder second question; empty reasoning on both, a missing answer and an unparseable response still fail"
