#!/usr/bin/env bash
#
# Post-setup smoke test — confirms the stack is healthy before you start using it.
#
# Runs four checks, each short-circuits on failure with an actionable hint:
#   1. Server responds on /v1/models
#   2. Genesis patches applied cleanly (tool_call fix is the fragile one)
#   3. Basic text completion works (Paris sanity)
#   4. Tool calling works end-to-end (request includes tools → response has tool_calls[])
#
# If check 4 fails but checks 1-3 pass, your Genesis tool_call patch didn't apply
# and you're on a vLLM nightly that drifted past our pinned digest. See the README
# troubleshooting section.
#
# Env vars (optional):
#   URL          Override endpoint. Default: http://localhost:8020
#   MODEL        Served model name. Default: qwen3.6-27b-autoround
#   CONTAINER    Docker container name for log scraping. Default: vllm-qwen36-27b

set -euo pipefail

URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-qwen3.6-27b-autoround}"
CONTAINER="${CONTAINER:-vllm-qwen36-27b}"

pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; printf "    \033[33m→\033[0m %s\n" "$2"; exit 1; }

echo "Running smoke test against ${URL} (model=${MODEL}, container=${CONTAINER})"
echo ""

# --------------------------------------------------------------------
# 1. Server reachable
# --------------------------------------------------------------------
echo "[1/4] Server reachable on /v1/models ..."
if curl -sf -m 5 "${URL}/v1/models" >/dev/null 2>&1; then
  pass "server is serving"
else
  fail "no response from ${URL}/v1/models" \
       "Start the stack: cd compose && docker compose up -d ; docker logs -f ${CONTAINER}"
fi

# --------------------------------------------------------------------
# 2. Genesis patches applied cleanly — especially the fragile tool_call one
# --------------------------------------------------------------------
echo "[2/4] Genesis patches applied (Qwen3 tool_call fix in particular) ..."
if ! command -v docker >/dev/null 2>&1; then
  echo "  (skipped — docker not in PATH, cannot read container logs)"
elif ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "  (skipped — container '${CONTAINER}' not found; if your container has a different name, set CONTAINER=...)"
else
  logs="$(docker logs "${CONTAINER}" 2>&1 | grep -E "Qwen3 tool_call fix|\[FAILED\]" | tail -5)"
  if echo "$logs" | grep -q "\[OK\] Qwen3 tool_call fix"; then
    pass "Genesis Qwen3 tool_call fix applied"
  elif echo "$logs" | grep -q "\[FAILED\] Qwen3 tool_call fix"; then
    fail "Genesis Qwen3 tool_call fix [FAILED]" \
         "Your vLLM image drifted past the patch anchor. Pin to sha256:9bba4628a3b9... in compose/docker-compose.yml (already pinned by default). Re-pull if you bumped it manually."
  else
    echo "  (warn — no Genesis OK/FAILED marker for tool_call in logs; container may have been restarted. Continuing.)"
  fi
fi

# --------------------------------------------------------------------
# 3. Basic completion — Paris sanity
# --------------------------------------------------------------------
echo "[3/4] Basic completion — capital of France ..."
resp="$(curl -sf -m 30 "${URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? Reply in one short sentence.\"}],
    \"max_tokens\": 30,
    \"temperature\": 0.6,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }")" || fail "completion request failed" "Check docker logs ${CONTAINER}"

content="$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || true)"
if echo "$content" | grep -qi "Paris"; then
  pass "reply contains 'Paris': $(echo "$content" | head -c 70)..."
else
  fail "reply didn't mention Paris: $(echo "$content" | head -c 80)" \
       "Model may be loading badly or using wrong chat template. Check docker logs ${CONTAINER}."
fi

# --------------------------------------------------------------------
# 4. Tool calling end-to-end — request with tools[] → response with tool_calls[]
# --------------------------------------------------------------------
echo "[4/4] Tool calling — model should populate tool_calls[] ..."
tool_resp="$(curl -sf -m 60 "${URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"What is the weather in San Francisco right now? Use the get_weather tool.\"}
    ],
    \"tools\": [
      {
        \"type\": \"function\",
        \"function\": {
          \"name\": \"get_weather\",
          \"description\": \"Get the current weather for a given city.\",
          \"parameters\": {
            \"type\": \"object\",
            \"properties\": {
              \"city\": {\"type\": \"string\", \"description\": \"City name\"},
              \"units\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}
            },
            \"required\": [\"city\"]
          }
        }
      }
    ],
    \"tool_choice\": \"auto\",
    \"max_tokens\": 200,
    \"temperature\": 0.3,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }")" || fail "tool-call request failed" "Check docker logs ${CONTAINER}"

tool_calls="$(echo "$tool_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls')
    if tc:
        print(json.dumps(tc, indent=2))
    else:
        # Check if model inlined <tool_call> as plain text — the symptom of a broken patch
        content = d['choices'][0]['message'].get('content') or ''
        if '<tool_call>' in content:
            print('__INLINED__', content[:200], sep='\n')
        else:
            print('__NONE__', content[:200], sep='\n')
except Exception as e:
    print(f'__PARSE_ERROR__: {e}')
" 2>&1)"

if echo "$tool_calls" | grep -q "__INLINED__"; then
  fail "model emitted <tool_call> as inline text (tool_calls[] is empty)" \
       "Genesis Patch 12 (Qwen3 tool_call fix) did not apply. Re-check the container logs and pin the image digest. README § Troubleshooting has the full chain."
elif echo "$tool_calls" | grep -q "__NONE__"; then
  fail "model answered without invoking the tool" \
       "May be a model-behavior issue (it chose not to call) rather than a patch issue. Try rephrasing the prompt or lowering temperature. Raw content: $(echo "$tool_calls" | tail -1)"
elif echo "$tool_calls" | grep -q "__PARSE_ERROR__"; then
  fail "couldn't parse the response JSON" \
       "Response was: $(echo "$tool_resp" | head -c 400)"
elif echo "$tool_calls" | grep -qi "get_weather"; then
  pass "tool_calls[] populated, includes get_weather:"
  echo "$tool_calls" | head -20 | sed 's/^/      /'
else
  fail "unexpected tool_calls structure" \
       "Raw: $(echo "$tool_calls" | head -c 300)"
fi

echo ""
echo "All checks passed. Stack is ready for use."
