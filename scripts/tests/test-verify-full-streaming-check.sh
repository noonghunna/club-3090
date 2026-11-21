#!/usr/bin/env bash
# test-verify-full-streaming-check.sh — verify-full [5/10] counts a live stream's
# reasoning chunks too, and still fails a stream that really is buffered or empty.
#
# WHY: [5] asks for a three-line haiku and fails under 5 chunks ("SSE may be
# buffering"). It counted CONTENT chunks only. With speculative decoding a step can
# emit several tokens in one chunk (DSpark: up to 8), so Inkling-2-Mini on one RTX 3090
# with its DSpark drafter streamed the haiku intact as 3 content chunks and failed a
# healthy boot (2026-11-21). The reasoning streamed before it (`reasoning` on vLLM
# >= 0.34, `reasoning_content` on llama.cpp / SGLang) is just as much proof that SSE
# is not buffered.
#
# This runs the REAL `check_streaming` from scripts/verify-full.sh (extracted, not
# copied) with `curl` stubbed to serve each stream shape.
# NEGATIVE CONTROL: against the pre-fix verify-full (VERIFY_FULL=<old copy>) cases 1
# and 2 fail; cases 3-5 hold on both.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VF="${VERIFY_FULL:-$ROOT/scripts/verify-full.sh}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# The function body, from `check_streaming() {` to its closing `}` at column 0.
awk '/^check_streaming\(\) \{/{p=1} p{print} p&&/^\}/{exit}' "$VF" > "$TMP/check.sh"
command grep -q '^check_streaming() {' "$TMP/check.sh" || { echo "FAIL: could not extract check_streaming from $VF" >&2; exit 1; }

# stream <n_reasoning> <reasoning field|-> <content pieces...>  → an SSE body
stream() {
  python3 - "$@" <<'PY'
import json, sys
n, field, pieces = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
out = []
for i in range(n):
    out.append({"choices": [{"index": 0, "delta": {field: f"thinking {i} "}}]})
for p in pieces:
    out.append({"choices": [{"index": 0, "delta": {"content": p}}]})
for o in out:
    print("data: " + json.dumps(o))
    print()
print("data: [DONE]")
PY
}

fails=0
case_() {   # <name> <want pass|fail> <body file>
  local name=$1 want=$2 body=$3 out rc=0
  out=$(BODY_FILE="$body" bash -c '
    pass() { echo "PASS: $1"; }
    fail() { echo "FAIL: $1"; return 1; }
    curl() { cat "$BODY_FILE"; }
    URL=http://mock MODEL=m TMO_STREAM=45 MT_STREAM=120 THINK_OFF_STD="" THINK_OFF_KW="{\"enable_thinking\": false}"
    source "'"$TMP/check.sh"'"
    check_streaming' 2>&1) || rc=$?
  local got=pass; [[ $rc -ne 0 || "$out" == *"FAIL: "* ]] && got=fail
  if [[ "$got" == "$want" ]]; then echo "  ok  $name → $got"
  else echo "FAIL: $name → $got, expected $want" >&2; echo "$out" | sed 's/^/      /' >&2; fails=$((fails+1)); fi
}

HAIKU=("The tiny bug is small. " "I trace the flickering line. " "The fix comes at last.")
stream 10 reasoning "${HAIKU[@]}" > "$TMP/s1"
stream 10 reasoning_content "${HAIKU[@]}" > "$TMP/s2"
stream 0 - "${HAIKU[@]}" > "$TMP/s3"
stream 0 - "The " "tiny " "bug " "is " "small. " "I " "trace " "the line." > "$TMP/s4"
stream 10 reasoning > "$TMP/s5"

case_ "1. 3 content chunks after 10 \`reasoning\` chunks (vLLM >= 0.34, spec decode)" pass "$TMP/s1"
case_ "2. 3 content chunks after 10 \`reasoning_content\` chunks (llama.cpp / SGLang)"  pass "$TMP/s2"
case_ "3. 3 content chunks and nothing else (buffered-looking) → still fails"           fail "$TMP/s3"
case_ "4. 8 content chunks, no reasoning (a plain model) → passes as before"            pass "$TMP/s4"
case_ "5. reasoning only, no content → still fails"                                     fail "$TMP/s5"

if (( fails > 0 )); then echo "test-verify-full-streaming-check: ${fails} case(s) failed" >&2; exit 1; fi
echo "test-verify-full-streaming-check: all cases passed"
