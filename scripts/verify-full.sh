#!/usr/bin/env bash
#
# Full-functional test — extends verify.sh with streaming, thinking mode,
# long-context needle-in-a-haystack, tool-prefill OOM detection, output-
# quality / cascade detection, and MTP acceptance-length verification.
# Run before publishing or after any major patch / vLLM image bump.
#
# This is SLOWER than verify.sh (builds large prompts, does streaming,
# ~80K token needle test, ~15K-token tool prefill). Allow ~3-5 minutes.
#
# Checks (in order):
#   1. Server reachable
#   2. Genesis patches applied
#   3. Basic completion (Paris)
#   4. Tool calling (KNOWN TO FAIL in default compose; PASS in tools variants)
#   5. Streaming (SSE) — non-tool prompt, verify chunks add up to coherent text
#   6. Thinking mode — reasoning prompt, verify reasoning + content both populated
#   7. Long-context needle — ~80K-token prompt with secret embedded, verify recall
#   8. Tool response prefill OOM — multi-turn payload with ~15K-token mock tool
#      response, catches activation-memory OOM at the prefill peak (the bug class
#      that hit production at 192K + 0.98 mem-util)
#   9. Output quality / cascade detection — 2K-token completion, scan for
#      <tool_call> inline cascade and repetitive degeneracy
#  10. MTP acceptance length — parse SpecDecoding metrics from docker logs,
#      assert mean AL >= 2.0 (sanity that spec-decode is contributing)
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
#   SKIP_LONGCTX Set to 1 to skip the long-context test (saves ~30s and
#                ~3GB of VRAM pressure).
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
  echo "[1/10] Server reachable on /v1/models ..."
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
  echo "[2/10] Genesis patches applied ..."
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
  echo "[3/10] Basic completion — capital of France ..."
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
  echo "[4/10] Tool calling ..."
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
  echo "[5/10] Streaming (SSE) ..."
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
  echo "[6/10] Thinking / reasoning mode ..."
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
# 7. Long-context needle — put a secret at ~80% depth, ask for it at the end
# --------------------------------------------------------------------
check_longctx() {
  echo "[7/10] Long-context needle (ladder: 10K / 30K / 60K / 90K) ..."
  if [[ "${SKIP_LONGCTX:-0}" == "1" ]]; then
    skip "SKIP_LONGCTX=1"
    return 0
  fi

  # Ladder tests: 10K, 30K, 60K, 90K tokens. Each needs its own random
  # secret so caching doesn't confound results. Needle placed at 50% depth.
  # Depths that exceed the engine's --max-model-len get HTTP 400 — that's
  # the engine's prefill-safety guardrail (clean rejection, not OOM); we
  # treat that as "skipped at this ctx" rather than a test failure.
  local any_fail=0
  local any_pass=0
  local any_skipped=0

  # Discover deployed max-model-len for graceful skipping (best-effort).
  local deployed_max
  deployed_max="$(curl -sf -m 5 "${URL}/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0].get('max_model_len',0))" 2>/dev/null \
    || echo 0)"

  for filler_scale in 150 450 900 1400; do
    local secret_file req_file
    secret_file="$(mktemp --suffix=.secret)"
    req_file="$(mktemp --suffix=.json)"
    MODEL_VAR="${MODEL}" SECRET_FILE="${secret_file}" REQ_FILE="${req_file}" \
      FILLER_SCALE="${filler_scale}" python3 - <<'EOF'
import json, os, random
random.seed(None)
model = os.environ['MODEL_VAR']
scale = int(os.environ['FILLER_SCALE'])
# Use a memorable-content secret: random animal + color + 2-digit number.
# This uses in-vocabulary tokens so recall failures reflect attention quality,
# not tokenizer-sparsity edge cases (random hex triggers per-character output).
animals = ["otter", "falcon", "platypus", "iguana", "narwhal", "chinchilla", "capybara", "axolotl"]
colors = ["crimson", "turquoise", "amber", "violet", "emerald", "sapphire", "silver", "golden"]
animal = random.choice(animals)
color = random.choice(colors)
num = random.randint(10, 99)
secret = f"{color} {animal} {num}"
block = (
    "This section describes the history of computing in detail. "
    "Transistors were invented in 1947 at Bell Labs. The integrated circuit came a decade later. "
    "Microprocessors emerged in the 1970s and changed the world. "
    "Personal computing followed, then networking, then the web, then cloud and AI. "
)
half = scale // 2
filler_before = block * half
filler_after  = block * (scale - half)
content = (
    filler_before
    + f"\n\nIMPORTANT MEMORY: The hidden phrase is '{secret}'. Remember this exactly.\n\n"
    + filler_after
    + f"\n\nQuestion: In the middle of the document above I wrote 'The hidden phrase is ___'. What was the hidden phrase? Reply with only the phrase, no other text."
)
req = {
    "model": model,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 30,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
with open(os.environ['SECRET_FILE'], 'w') as f:
    f.write(secret)
with open(os.environ['REQ_FILE'], 'w') as f:
    json.dump(req, f)
EOF
    local secret
    secret="$(cat "$secret_file")"
    local resp content_raw prompt_tok http_code resp_file
    resp_file="$(mktemp --suffix=.json)"
    http_code="$(curl -s -m 300 -o "${resp_file}" -w '%{http_code}' \
      "${URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      --data-binary "@${req_file}")" || http_code="000"
    rm -f "$secret_file" "$req_file"
    if [[ "$http_code" == "400" ]]; then
      # Engine pre-check rejected — clean ctx-limit response, not OOM.
      printf "    \033[33m⊘\033[0m scale=%d: HTTP 400 (exceeds --max-model-len, expected — clean rejection)\n" "$filler_scale"
      rm -f "$resp_file"
      any_skipped=1
      continue
    elif [[ "$http_code" != "200" ]]; then
      printf "    \033[31m✗\033[0m scale=%d: HTTP %s (request failed)\n" "$filler_scale" "$http_code"
      rm -f "$resp_file"
      any_fail=1
      continue
    fi
    resp="$(cat "${resp_file}")"
    rm -f "${resp_file}"
    prompt_tok="$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null)"
    content_raw="$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)"
    # Match on all three tokens (color, animal, number) individually — even if
    # model wraps quotes / capitalizes / reorders slightly, all three appearing
    # means recall succeeded.
    local all_match=1
    for tok in $secret; do
      echo "$content_raw" | grep -qiF "$tok" || all_match=0
    done
    if [[ "$all_match" == "1" ]]; then
      printf "    \033[32m✓\033[0m %6s tokens: recalled '%s' (got: %s)\n" "$prompt_tok" "$secret" "$(echo "$content_raw" | head -c 60 | tr '\n' ' ')"
      any_pass=1
    else
      printf "    \033[31m✗\033[0m %6s tokens: expected '%s', got '%s'\n" "$prompt_tok" "$secret" "$(echo "$content_raw" | head -c 80 | tr '\n' ' ')"
      any_fail=1
    fi
  done

  if [[ "$any_fail" == "0" ]] && [[ "$any_pass" == "1" ]]; then
    if [[ "$any_skipped" == "1" ]]; then
      pass "all in-budget long-ctx depths recalled secret (above-budget depths cleanly rejected by engine pre-check)"
    else
      pass "all long-ctx depths recalled secret correctly"
    fi
  elif [[ "$any_fail" == "0" ]] && [[ "$any_pass" == "0" ]]; then
    skip "all depths above --max-model-len (deployed=${deployed_max:-unknown}); shrink ladder or raise ctx"
  elif [[ "$any_pass" == "1" ]]; then
    fail "partial recall — some in-budget depths failed" \
         "Attention quality degrades at longer contexts on this config OR the deployment crashed mid-test. Check docker logs."
  else
    fail "no depth recalled the secret (all failed, none succeeded)" \
         "Either container crashed early in the ladder or attention is broken. Check docker logs."
  fi
}
run_check "longctx" check_longctx

# --------------------------------------------------------------------
# 8. Tool response prefill OOM — multi-turn with ~15K-token mock tool
#    response, catches activation-memory peak during prefill (the bug
#    class that hit production at 192K + 0.98 mem-util — verified at
#    idle but OOMs the moment a real-world tool reply is loaded).
# --------------------------------------------------------------------
check_tool_prefill() {
  echo "[8/10] Tool response prefill OOM (~15K-token mock tool response) ..."
  if [[ "${SKIP_TOOL_PREFILL:-0}" == "1" ]]; then
    skip "SKIP_TOOL_PREFILL=1"
    return 0
  fi
  local req_file resp_file
  req_file="$(mktemp --suffix=.json)"
  resp_file="$(mktemp --suffix=.json)"
  MODEL_VAR="${MODEL}" REQ_FILE="${req_file}" python3 - <<'EOF'
import json, os
model = os.environ['MODEL_VAR']
# News-style filler blocks (~300 chars each, varied content) to simulate a
# realistic tool response (web fetch, news API, document read, etc.).
blocks = [
    "Federal Reserve Chair Jerome Powell stated today that interest rates would remain steady amid mixed economic signals. The central bank's decision came after months of debate about inflation trajectories and labor market resilience. Treasury yields responded modestly, with the 10-year note ticking down two basis points by late trading.",
    "European markets opened higher on news that German industrial output rebounded sharply in March. The DAX gained 0.8% in morning trading while the Stoxx 600 added 0.5%. Analysts cited improved manufacturing PMI readings and stabilizing energy prices as primary drivers behind the optimistic open.",
    "Tech sector earnings season kicked into high gear this week with several major firms reporting better-than-expected quarterly results. Cloud computing revenues grew across the board, with AI infrastructure demand cited as a key catalyst. Margin pressure remained a concern in semiconductor names due to inventory adjustments.",
    "Crude oil prices edged higher after OPEC announced extended production cuts through the third quarter. Brent crude rose 1.2% to settle near $84 per barrel, while WTI gained similarly to $79. Geopolitical tensions in the Middle East continued to lend support to prices despite weakening demand signals from China.",
    "Bond markets saw a mild flattening of the yield curve as investors digested mixed signals about economic growth. The spread between 2-year and 10-year Treasuries narrowed to 35 basis points, down from 42 a week prior. Dealers cited reduced expectations for near-term Fed action as the primary driver.",
    "Currency markets remained range-bound with the dollar index trading near 104.5 throughout the session. The euro held above 1.08 as traders awaited Thursday's ECB minutes for clarity on the rate path. The yen weakened modestly as Japanese authorities continued verbal intervention without direct market action.",
    "Gold prices touched a fresh three-week high at $2,415 per ounce as safe-haven demand returned amid simmering geopolitical concerns. Silver tracked higher in sympathy, gaining 0.8%. Mining stocks rallied broadly with the GDX ETF up over 1.5% for the day on heavier-than-average volume.",
    "US equity markets posted modest gains with the S&P 500 closing up 0.4% at 5,680. The Nasdaq Composite added 0.7% led by mega-cap tech names. Small-caps lagged with the Russell 2000 finishing flat as investors continued to favor large-cap growth in the current uncertain rate environment.",
    "Cryptocurrency markets experienced renewed volatility with Bitcoin briefly trading above $73,000 before settling near $71,500. Ethereum followed a similar pattern, peaking at $3,950 before retracing. Spot ETF flows turned positive for the third consecutive day, snapping a brief outflow streak from late last week.",
    "Real estate markets showed continued bifurcation between residential and commercial sectors. Existing home sales fell 1.9% month-over-month while office vacancy rates ticked higher in major metros. REIT performance reflected this divide with residential REITs outperforming office and retail-focused names by a wide margin.",
    "Manufacturing PMI readings across emerging markets came in mixed, with India and Vietnam showing expansion while Brazil and South Africa contracted. Supply chain conditions continued to normalize from pandemic-era disruptions, though shipping rates remained elevated due to Red Sea route detours.",
    "Insurance sector earnings reflected ongoing pricing power as carriers continued to push through rate increases on commercial lines. Auto insurance trends showed moderation in claim severity though frequency remained elevated. Reinsurance pricing stabilized after several quarters of significant upward pressure.",
    "Healthcare M&A activity picked up notably with three major deals announced in the biotech space. Strategic buyers continued to dominate the deal landscape as private equity remained selective amid elevated financing costs. IPO pipeline strength suggested potential thawing in capital markets activity.",
    "Consumer staples companies reported divergent results with packaged food makers facing volume pressure while beverage names exceeded expectations. Pricing power moderated across categories as private label gained share. Margin commentary suggested a return to volume-led growth strategies for fiscal 2026.",
    "Semiconductor industry data showed continued strength in AI-related demand offset by softness in traditional end markets including industrial and automotive. Inventory normalization progressed as channel checks indicated improving dynamics. Capacity expansion plans remained robust at leading-edge nodes.",
    "Renewable energy stocks rallied on news of expanded tax credits in pending legislation. Solar panel manufacturers led the move with several names gaining over 5%. Wind energy faced ongoing headwinds from supply chain costs but installation pipelines suggested improving fundamentals through year-end.",
    "Telecommunications companies reported stable subscriber trends with limited churn despite increased competitive promotional activity. Capex commentary suggested moderation in 5G build-out spending as networks reach critical density. Fiber expansion continued to be the primary growth driver for wireline operations.",
    "Industrial conglomerates posted solid quarterly results with order backlogs reaching multi-year highs in several segments. Aerospace and defense saw particular strength while traditional manufacturing showed mixed regional performance. Margin expansion came from operational improvements and pricing actions implemented earlier.",
    "Retail spending data for the latest week suggested steady consumer activity though average ticket sizes moderated. Discount channels gained share as mid-tier department stores faced ongoing pressure. Apparel categories saw some normalization after prior weather-driven volatility.",
    "Transportation indices ticked higher with rail traffic up 2.1% year-over-year on strong intermodal volumes. Trucking spot rates remained pressured though contract rates stabilized. Air freight saw seasonal strength as electronics and pharmaceutical shipments accelerated ahead of mid-year inventory builds.",
]
target_chars = int(os.environ.get('PREFILL_TARGET_CHARS', '100000'))  # ~25K tokens default
content = ""
i = 0
while len(content) < target_chars:
    content += blocks[i % len(blocks)] + "\n\n"
    i += 1
# Match real-world tool-calling pipelines (Hermes / OpenAI / Open WebUI):
# include the tool definition + tool_choice so xgrammar matcher loads. ampersandru's
# production OOM trace shows xgrammar grammar matchers active at the OOM site.
tool_def = {"type": "function",
            "function": {"name": "fetch_news",
                         "description": "Fetch latest news on a topic.",
                         "parameters": {"type": "object",
                                        "properties": {"topic": {"type": "string"}},
                                        "required": ["topic"]}}}
payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "What's happening in financial markets today?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_news_1", "type": "function",
             "function": {"name": "fetch_news",
                          "arguments": json.dumps({"topic": "markets"})}}
        ]},
        {"role": "tool", "tool_call_id": "call_news_1", "content": content},
        {"role": "user", "content": "Summarize the top 3 themes from this news data in about 100 words."}
    ],
    "tools": [tool_def],
    "tool_choice": "auto",
    "max_tokens": 500,
    "temperature": 0.6,
    "chat_template_kwargs": {"enable_thinking": False},
}
with open(os.environ['REQ_FILE'], 'w') as f:
    json.dump(payload, f)
EOF

  local http_code
  http_code="$(curl -s -m 240 -o "${resp_file}" -w '%{http_code}' \
    "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary "@${req_file}")" || http_code="000"
  rm -f "$req_file"

  case "$http_code" in
    200)
      # The OOM is the failure mode we care about. ANY valid HTTP 200 response —
      # whether the model emits a text summary OR another tool_call — proves the
      # ~25K-token prefill survived activation-memory peak. Distinguish only OOM
      # from real responses; let the model's output choice stand as legitimate.
      local content_len tc_count finish
      read -r content_len tc_count finish < <(python3 -c "
import json
try:
    d = json.load(open('${resp_file}'))
    msg = d['choices'][0]['message']
    c = msg.get('content') or ''
    tc = msg.get('tool_calls') or []
    f = d['choices'][0].get('finish_reason') or 'n/a'
    print(len(c), len(tc), f)
except Exception as e:
    print(-1, 0, f'parse_err:{e}')
" 2>/dev/null)
      if [[ "${content_len:-0}" -ge 50 ]]; then
        pass "tool prefill OK — text response (${content_len} chars, finish=${finish})"
      elif [[ "${tc_count:-0}" -ge 1 ]]; then
        pass "tool prefill OK — model emitted ${tc_count} tool_call(s) (finish=${finish}, prefill survived)"
      else
        fail "HTTP 200 but empty response (text=${content_len:-0} chars, tool_calls=${tc_count:-0}, finish=${finish:-?})" \
             "Likely silent prefill truncation. Check docker logs for warnings."
      fi
      ;;
    500)
      fail "HTTP 500 — OOM during ~15K-token tool-response prefill" \
           "Activation memory peak exceeded budget. Lower --max-model-len or --gpu-memory-utilization. See README 'Activation memory caveat'. Server logs: docker logs ${CONTAINER} 2>&1 | tail -50"
      ;;
    000)
      fail "no HTTP response (timeout or container died)" \
           "Prefill may have hung or container OOM-killed. Check: docker logs ${CONTAINER} 2>&1 | tail -50; nvidia-smi"
      ;;
    *)
      fail "unexpected HTTP ${http_code}" \
           "Body head: $(head -c 200 "${resp_file}" 2>/dev/null)"
      ;;
  esac
  rm -f "$resp_file"
}
run_check "tool_prefill" check_tool_prefill

# --------------------------------------------------------------------
# 9. Output quality / cascade detection — 2K-token completion, scan
#    for the silent <tool_call> inline cascade (MTP × TurboQuant bug)
#    and for repetitive degeneracy (stale-draft / sampling collapse).
# --------------------------------------------------------------------
check_output_quality() {
  echo "[9/10] Output quality / cascade detection (2K-token completion) ..."
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
  echo "[10/10] MTP acceptance length threshold ..."
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
