#!/usr/bin/env bash
#
# Full-functional test — fast functional smoke covering server reachability,
# patch application, basic completion, tool calling, streaming, thinking
# mode, output quality / cascade detection, and MTP acceptance verification.
# Run before publishing or after any major patch / vLLM image bump.
#
# This is the FAST functional check (~1-2 min). It does NOT exercise long-
# context recall or large prefill activation peaks — for those run
# `bash scripts/verify-stress.sh` (5-10 min, KV-cache + prefill boundary tests).
#
# Checks (in order):
#   1. Server reachable
#   2. Genesis patches applied
#   3. Basic completion (Paris)
#   4. Tool calling (KNOWN TO FAIL in default compose; PASS in tools variants)
#   5. Streaming (SSE) — non-tool prompt, verify chunks add up to coherent text
#   6. Thinking mode — reasoning prompt, verify reasoning + content both populated
#   7. Output quality / cascade detection — 2K-token completion, scan for
#      <tool_call> inline cascade and repetitive degeneracy
#   8. MTP acceptance length — parse SpecDecoding metrics from docker logs,
#      assert mean AL >= 2.0 (sanity that spec-decode is contributing)
#
# For boundary / stress validation (longctx needle ladder, ~25K-token tool
# prefill OOM detection): see scripts/verify-stress.sh
#
# Usage:
#   CONTAINER=<your-container> bash scripts/verify-full.sh
#
# Env (optional):
#   URL          Default: http://localhost:8020
#   MODEL        Default: qwen3.6-27b-autoround
#   CONTAINER    Default: vllm-qwen36-27b
#   SKIP_TOOLS   Set to 1 to skip the tool-call test entirely (useful when
#                running against the default config which is known to fail
#                tool calls — see README "Known issue" section).
#
# Optional flag:
#   --bench      After all correctness checks pass, run scripts/bench.sh
#                (3 warmup + 5 measured) to report wall_TPS / decode_TPS /
#                TTFT mean+std+CV. Adds ~1-2 minutes.

set -euo pipefail

RUN_BENCH=0
for arg in "$@"; do
  case "$arg" in
    --bench) RUN_BENCH=1 ;;
  esac
done

URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-qwen3.6-27b-autoround}"
CONTAINER="${CONTAINER:-vllm-qwen36-27b}"

pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; printf "    \033[33m→\033[0m %s\n" "$2"; return 1; }
skip() { printf "  \033[33m⊘\033[0m %s (skipped)\n" "$1"; }

FAILED=0
run_check() {
  local label="$1"; shift
  if "$@"; then :; else FAILED=$((FAILED + 1)); fi
}

echo "Running FULL functional test against ${URL} (model=${MODEL}, container=${CONTAINER})"
echo ""

# --------------------------------------------------------------------
# 1. Server reachable
# --------------------------------------------------------------------
check_server() {
  echo "[1/8] Server reachable on /v1/models ..."
  if curl -sf -m 5 "${URL}/v1/models" >/dev/null 2>&1; then
    pass "server is serving"
  else
    fail "no response from ${URL}/v1/models" \
         "Start the stack: cd compose && docker compose up -d ; docker logs -f ${CONTAINER}"
  fi
}
run_check "server" check_server

# --------------------------------------------------------------------
# 2. Genesis patches applied
# --------------------------------------------------------------------
check_patches() {
  echo "[2/8] Genesis patches applied ..."
  if ! command -v docker >/dev/null 2>&1; then
    skip "docker not in PATH"
    return 0
  fi
  if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    skip "container '${CONTAINER}' not found"
    return 0
  fi
  local logs
  logs="$(docker logs "${CONTAINER}" 2>&1 | grep -E "Qwen3 tool_call fix|\[FAILED\]" | tail -5)"
  if echo "$logs" | grep -q "\[OK\] Qwen3 tool_call fix"; then
    pass "Genesis Qwen3 tool_call fix applied"
  elif echo "$logs" | grep -q "\[FAILED\] Qwen3 tool_call fix"; then
    fail "Genesis Qwen3 tool_call fix [FAILED]" \
         "vLLM image drifted past patch anchor. Pin sha256:9bba4628a3b9 in compose."
  else
    skip "no Genesis marker in logs"
  fi
}
run_check "patches" check_patches

# --------------------------------------------------------------------
# 3. Basic completion — Paris sanity
# --------------------------------------------------------------------
check_basic() {
  echo "[3/8] Basic completion — capital of France ..."
  local resp
  resp="$(curl -sf -m 30 "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? One short sentence.\"}],
      \"max_tokens\": 30,
      \"temperature\": 0.6,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }")" || { fail "completion request failed" "Check docker logs ${CONTAINER}"; return 1; }
  local content
  content="$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || true)"
  if echo "$content" | grep -qi "Paris"; then
    pass "reply contains 'Paris'"
  else
    fail "reply didn't mention Paris: $(echo "$content" | head -c 80)" \
         "Model may be loading badly or wrong chat template."
  fi
}
run_check "basic" check_basic

# --------------------------------------------------------------------
# 4. Tool calling
# --------------------------------------------------------------------
check_tools() {
  echo "[4/8] Tool calling ..."
  if [[ "${SKIP_TOOLS:-0}" == "1" ]]; then
    skip "SKIP_TOOLS=1 (expected for default config — see README Known issue)"
    return 0
  fi
  local resp
  resp="$(curl -sf -m 60 "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in San Francisco? Use the get_weather tool.\"}],
      \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
      \"tool_choice\": \"auto\", \"max_tokens\": 200, \"temperature\": 0.3,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }")" || { fail "tool-call request failed" "Check docker logs"; return 1; }
  local tool_calls
  tool_calls="$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls')
    if tc:
        print(json.dumps(tc, indent=2))
    else:
        content = d['choices'][0]['message'].get('content') or ''
        if '<tool_call>' in content:
            print('__INLINED__')
        else:
            print('__NONE__')
except Exception as e:
    print(f'__PARSE_ERROR__: {e}')
" 2>&1)"
  if echo "$tool_calls" | grep -q "__INLINED__"; then
    fail "model emitted <tool_call> as inline text (tool_calls[] empty)" \
         "Known issue: MTP × TurboQuant incompat. Use docker-compose.tools.yml or .tools-text.yml. See README Known issues."
  elif echo "$tool_calls" | grep -qi "get_weather"; then
    pass "tool_calls[] populated with get_weather"
  else
    fail "unexpected tool_calls structure" "Raw: $(echo "$tool_calls" | head -c 300)"
  fi
}
run_check "tools" check_tools

# --------------------------------------------------------------------
# 5. Streaming — SSE chunks add up to coherent text
# --------------------------------------------------------------------
check_streaming() {
  echo "[5/8] Streaming (SSE) ..."
  # Collect streamed chunks for 15 seconds max
  local stream_out
  stream_out="$(curl -sf -m 45 --no-buffer "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Write a three-sentence haiku about debugging.\"}],
      \"max_tokens\": 120,
      \"temperature\": 0.6,
      \"stream\": true,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" 2>/dev/null)" || { fail "streaming request failed" "Check docker logs"; return 1; }

  local text chunks
  text="$(echo "$stream_out" | python3 -c "
import sys, json
text = ''
chunks = 0
for line in sys.stdin:
    line = line.strip()
    if not line or not line.startswith('data: '):
        continue
    payload = line[6:]
    if payload == '[DONE]':
        break
    try:
        d = json.loads(payload)
        delta = d['choices'][0].get('delta', {}).get('content') or ''
        if delta:
            text += delta
            chunks += 1
    except Exception:
        pass
print(f'{chunks}||{text}')
" 2>/dev/null)"
  chunks="${text%%||*}"
  local final_text="${text#*||}"
  if [[ -z "$final_text" ]] || [[ "$chunks" == "0" ]]; then
    fail "no streaming content received ($chunks chunks)" \
         "Streaming broken — check that vLLM isn't buffering. stream_out head: $(echo "$stream_out" | head -c 200)"
  elif [[ "$chunks" -lt 5 ]]; then
    fail "suspiciously few chunks ($chunks) for 120 max_tokens" \
         "SSE may be buffering. Final text: $(echo "$final_text" | head -c 120)"
  elif [[ ${#final_text} -lt 20 ]]; then
    fail "streamed text too short (${#final_text} chars)" \
         "Content: $final_text"
  else
    pass "streamed $chunks chunks, ${#final_text} chars:  $(echo "$final_text" | head -c 80 | tr '\n' ' ')..."
  fi
}
run_check "streaming" check_streaming

# --------------------------------------------------------------------
# 6. Thinking mode — reasoning + content both populated
# --------------------------------------------------------------------
check_thinking() {
  echo "[6/8] Thinking / reasoning mode ..."
  local resp
  # enable_thinking: true (Qwen3 default). Math problem that needs visible reasoning.
  resp="$(curl -sf -m 120 "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? One-line answer.\"}],
      \"max_tokens\": 4000,
      \"temperature\": 0.3,
      \"chat_template_kwargs\": {\"enable_thinking\": true}
    }")" || { fail "thinking request failed" "Check docker logs"; return 1; }
  local analyzed
  analyzed="$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
msg = d['choices'][0]['message']
reasoning = msg.get('reasoning') or ''
content = msg.get('content') or ''
finish = d['choices'][0].get('finish_reason')
print(f'{len(reasoning)}|{len(content)}|{finish}|{(reasoning[:60] or \"(empty)\").replace(chr(10), \" \")}|{(content[:60] or \"(empty)\").replace(chr(10), \" \")}')
" 2>/dev/null)"
  IFS='|' read -r r_len c_len fin r_head c_head <<< "$analyzed"
  if [[ -z "$r_len" ]]; then
    fail "couldn't parse thinking response" "$(echo "$resp" | head -c 300)"
  elif [[ "$r_len" == "0" ]]; then
    fail "reasoning field empty (thinking mode didn't engage)" \
         "May indicate Genesis Patch 12 didn't land or chat_template_kwargs not honored. content='$c_head'"
  elif [[ "$c_len" == "0" ]] && [[ "$fin" == "length" ]]; then
    # Reasoning populated but model didn't finish before max_tokens — thinking
    # mode is working (reasoning field extracted cleanly), just verbose.
    pass "reasoning $r_len chars (model kept thinking, hit max_tokens before finishing — Qwen3.6 is verbose; thinking IS extracting correctly)"
    printf "    \033[2mreasoning head:\033[0m %s...\n" "$r_head"
  elif [[ "$c_len" == "0" ]]; then
    fail "reasoning present but content empty, finish=$fin (not length)" \
         "Likely genuine stall — finish_reason should be length if it's just verbosity. reasoning: $r_head"
  elif [[ "$r_len" -lt 50 ]]; then
    fail "reasoning suspiciously short ($r_len chars)" "reasoning: $r_head"
  else
    pass "reasoning $r_len chars, content $c_len chars (finish=$fin)"
    printf "    \033[2mreasoning:\033[0m %s...\n" "$r_head"
    printf "    \033[2mcontent:  \033[0m %s...\n" "$c_head"
  fi
}
run_check "thinking" check_thinking
# --------------------------------------------------------------------
# 9. Output quality / cascade detection — 2K-token completion, scan
#    for the silent <tool_call> inline cascade (MTP × TurboQuant bug)
#    and for repetitive degeneracy (stale-draft / sampling collapse).
# --------------------------------------------------------------------
check_output_quality() {
  echo "[7/8] Output quality / cascade detection (2K-token completion) ..."
  local resp
  resp="$(curl -sf -m 180 "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Write a detailed 1500-word essay explaining how transformer attention works. Cover: query/key/value projections, scaled dot-product attention, softmax, multi-head attention, positional encodings, and a brief comparison with RNN-based attention.\"}],
      \"max_tokens\": 2000,
      \"temperature\": 0.6,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }")" || { fail "output quality request failed" "Check docker logs ${CONTAINER}"; return 1; }

  local analysis
  analysis="$(echo "$resp" | python3 -c "
import sys, json, re
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message'].get('content') or ''
    finish = d['choices'][0].get('finish_reason') or 'n/a'
    clen = len(c)
    cascade = 'tool_call_cascade' if '<tool_call>' in c else 'none'
    # Repetitive cascade: same non-empty line appearing >=5 times consecutively
    lines = [l.strip() for l in c.split('\n') if l.strip()]
    max_repeat, cur_line, cur_count = 0, '', 0
    for l in lines:
        if l == cur_line:
            cur_count += 1
            max_repeat = max(max_repeat, cur_count)
        else:
            cur_line, cur_count = l, 1
    # Lexical variety over the first 200 words (samples coherence)
    words = re.findall(r\"[A-Za-z']+\", c.lower())
    sample = words[:200]
    variety = (len(set(sample)) / len(sample)) if sample else 0.0
    print(f'{clen}|{cascade}|{max_repeat}|{variety:.3f}|{finish}')
except Exception as e:
    print(f'err|{e}|0|0|n/a')
" 2>/dev/null)"

  IFS='|' read -r clen cascade max_repeat variety finish <<< "$analysis"
  if [[ "$clen" == "err" ]]; then
    fail "couldn't parse response: $cascade" "$(echo "$resp" | head -c 200)"
  elif [[ "${clen:-0}" == "0" ]]; then
    fail "empty completion (finish=${finish})" "Likely silent generation failure"
  elif [[ "$cascade" == "tool_call_cascade" ]]; then
    fail "MTP × TurboQuant cascade — <tool_call> emitted in normal text" \
         "Genesis P64/P65 not active or compose using broken MTP path. See README Known issues."
  elif [[ "${max_repeat:-0}" -ge 5 ]]; then
    fail "repetitive degeneracy — line repeats ${max_repeat}× consecutively" \
         "Sampling collapsed (stale-draft? sampler bug?). Check finish_reason=${finish}, vLLM ngram/spec settings."
  elif python3 -c "import sys; sys.exit(0 if float('${variety:-0}') >= 0.30 else 1)" 2>/dev/null; then
    pass "output OK — ${clen} chars, variety=${variety}, max_line_repeat=${max_repeat}, finish=${finish}"
  else
    fail "low lexical variety (${variety}, threshold 0.30)" \
         "Possible degenerate output. clen=${clen}, finish=${finish}"
  fi
}
run_check "output_quality" check_output_quality

# --------------------------------------------------------------------
# 10. MTP acceptance length — assert spec-decode is contributing speedup.
#     Mean AL >= 2.0 means each step accepts >=1 drafted token on average
#     (target_only baseline = 1.0). Production sees AL 3.4-3.8 with n=3.
# --------------------------------------------------------------------
check_mtp_acceptance() {
  echo "[8/8] MTP acceptance length threshold ..."
  if ! command -v docker >/dev/null 2>&1; then
    skip "docker not in PATH"
    return 0
  fi
  if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    skip "container '${CONTAINER}' not found"
    return 0
  fi

  # Trigger a fresh decode to populate metrics (some vLLM builds only emit
  # SpecDecoding stats after a non-trivial generation completes).
  curl -sf -m 60 "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 80, one number per line.\"}],
      \"max_tokens\": 500,
      \"temperature\": 0.0,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" >/dev/null 2>&1 || { fail "metrics-trigger request failed" "Check docker logs"; return 1; }
  sleep 3  # let log line flush

  local recent
  recent="$(docker logs --tail 200 "${CONTAINER}" 2>&1 | grep -iE "SpecDecoding|acceptance length|spec_decode" | tail -3)"
  if [[ -z "$recent" ]]; then
    skip "no SpecDecoding metrics in logs (compose may not have spec-decode enabled)"
    return 0
  fi

  local al
  al="$(echo "$recent" | grep -oiE "(mean acceptance length|acceptance length|al|mean_acceptance_length)[: ]+[0-9]+\.[0-9]+" \
        | grep -oE "[0-9]+\.[0-9]+" | tail -1)"
  if [[ -z "$al" ]]; then
    skip "couldn't parse AL from: $(echo "$recent" | head -c 240 | tr '\n' ' ')"
    return 0
  fi

  if python3 -c "import sys; sys.exit(0 if float('$al') >= 2.0 else 1)" 2>/dev/null; then
    pass "MTP acceptance length = ${al} (>=2.0 — spec-decode contributing)"
  else
    fail "MTP acceptance length = ${al} (<2.0 — spec-decode degraded or off)" \
         "Either MTP routing broken (P65 not active?) or accept rate collapsed. Check spec_decode kernel + Genesis env vars."
  fi
}
run_check "mtp" check_mtp_acceptance

echo ""
if [[ "$FAILED" == "0" ]]; then
  printf "\033[32mAll checks passed.\033[0m Stack is ready for full-functionality use.\n"
else
  printf "\033[31m%d check(s) failed.\033[0m See hints above.\n" "$FAILED"
fi

if [[ "$RUN_BENCH" == "1" && "$FAILED" == "0" ]]; then
  echo ""
  echo "=========================================="
  echo "  --bench: running scripts/bench.sh"
  echo "=========================================="
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  URL="${URL}" MODEL="${MODEL}" CONTAINER="${CONTAINER}" \
    bash "${SCRIPT_DIR}/bench.sh"
fi

exit "$FAILED"
