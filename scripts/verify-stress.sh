#!/usr/bin/env bash
#
# Stress / boundary-case test — for KV-cache and prefill-activation-memory
# stress paths that take real time to run. Split out from verify-full.sh
# 2026-04-28 so the fast functional smoke (verify-full.sh) stays under
# ~2 min while these heavier boundary tests are run only when needed.
#
# Run before publishing, after any major patch / vLLM image bump, or when
# investigating prefill-OOM regressions specifically.
#
# This is SLOW (large prompts, long-ctx needle ladder up to ~115K tokens,
# ~25K-token tool prefill) — allow ~5–10 minutes on dual-card, longer on
# single-card configurations where the longest depths get rejected by the
# engine pre-check (HTTP 400, treated as graceful skip).
#
# Checks (in order):
#   1. Long-context needle — recall ladder at 4 depths (10K / 30K / 60K / 90K
#      tokens); each depth gets its own random secret to defeat caching.
#      Depths above the deployed --max-model-len are gracefully skipped via
#      the engine's HTTP 400 pre-check (clean rejection, not failure).
#   2. Tool response prefill OOM — multi-turn payload with ~25K-token mock
#      tool message + tool definition + auto tool_choice; catches the
#      activation-memory peak class of bug (the one that hit production
#      at 192K + 0.98 mem-util — see ampersandru's report in the predecessor
#      single-3090 repo issue #1).
#   3. IDE-agent one-shot (added 2026-05-01) — synthetic Cline/OpenCode-shape
#      prompt: ~5K-char sys preamble + 10 tool schemas + ~350-char user request
#      + max_tokens=2000. Catches Cliff 1 mech B (inductor compile-path FFN
#      intermediate buffer leak). Different prefill shape than check #2 — bulk
#      is in the SYSTEM message + tool schemas, not in a 25K-token tool RETURN.
#      Required because check #2 passes on v0.20 + Genesis v7.65 dev tip but
#      this shape still crashes (see club-3090#16). Repro of VolandBerlioz's
#      Reddit failure mode; takes ~10s per attempt (one request, fail-fast).
#
# Future probe ideas (TODO — not yet implemented):
#   - Multi-turn agent: sys + tools + user → assistant(tool_call) → tool reply
#     → user followup. Different inductor compile path from one-shot.
#   - LCB-coding shape: problem statement → ~300-700 think tokens → code
#     emission. Triggers DS conv state crash on some configs.
#   - Reasoning-heavy: math/algorithm prompt + max_tokens=8192. Stresses
#     spec-decode AL collapse + mamba_cache_mode='align' interactions.
#
# Usage:
#   CONTAINER=<your-container> bash scripts/verify-stress.sh
#
# Env (optional):
#   URL                    Default: http://localhost:8020
#   MODEL                  Default: qwen3.6-27b-autoround
#   CONTAINER              Default: vllm-qwen36-27b
#   SKIP_LONGCTX           Set to 1 to skip the long-context needle ladder.
#   SKIP_TOOL_PREFILL      Set to 1 to skip the tool-response prefill test.
#   PREFILL_TARGET_CHARS   Tool-response prefill payload size in chars
#                          (default: 100000 ≈ 25K tokens; set higher to
#                          push closer to the cliff under investigation).

set -euo pipefail

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

echo "Running STRESS / boundary test against ${URL} (model=${MODEL}, container=${CONTAINER})"
echo "  This script does the heavy stuff (longctx needle ladder + ~25K-token tool prefill)."
echo "  For the fast functional smoke (~2 min), use verify-full.sh instead."
echo ""

# --------------------------------------------------------------------
# 1. Long-context needle — put a secret at ~50% depth, ask for it at the end
# --------------------------------------------------------------------
check_longctx() {
  echo "[1/3] Long-context needle (ladder: 10K / 30K / 60K / 90K) ..."
  if [[ "${SKIP_LONGCTX:-0}" == "1" ]]; then
    skip "SKIP_LONGCTX=1"
    return 0
  fi

  local any_fail=0
  local any_pass=0
  local any_skipped=0

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
# 2. Tool response prefill OOM — multi-turn with ~25K-token mock tool
#    response, catches activation-memory peak during prefill (the bug
#    class that hit production at 192K + 0.98 mem-util — verified at
#    idle but OOMs the moment a real-world tool reply is loaded).
# --------------------------------------------------------------------
check_tool_prefill() {
  echo "[2/3] Tool response prefill OOM (~25K-token mock tool response) ..."
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
target_chars = int(os.environ.get('PREFILL_TARGET_CHARS', '100000'))
content = ""
i = 0
while len(content) < target_chars:
    content += blocks[i % len(blocks)] + "\n\n"
    i += 1
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
      fail "HTTP 500 — OOM during ~25K-token tool-response prefill" \
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
  local rc=$?
  rm -f "$resp_file"
  return "$rc"
}
run_check "tool_prefill" check_tool_prefill

# 3. IDE-agent one-shot — synthetic Cline/OpenCode shape (added 2026-05-01).
# Catches Cliff 1 mech B (inductor compile-path FFN intermediate buffer leak)
# that fires on real coding-agent prompts but NOT on the synthetic 25K tool
# prefill above. See club-3090#16. Fail-fast: one request, ~10s if green,
# instant HTTP 500 if the bug fires.
echo "[3/3] IDE-agent one-shot prompt (sys + tool schemas + user request) ..."
check_ide_agent() {
  local req_file resp_file http_code body
  req_file="$(mktemp --suffix=.json)"
  resp_file="$(mktemp --suffix=.json)"
  MODEL_VAR="${MODEL}" REQ_FILE="${req_file}" python3 - <<'PYEOF'
import json, os
model = os.environ['MODEL_VAR']
# Synthetic IDE-agent system prompt: realistic Cline/OpenCode preamble x5
# to bulk it up to ~5K chars, the shape that triggers the bug.
sys_text = (
    "You are a helpful AI coding assistant operating inside an IDE. You have access to "
    "a set of tools to read, write, search, and execute commands in the user's project. "
    "Always use the appropriate tool when the user requests file operations or code "
    "execution. Be concise in your reasoning, prefer minimal edits, and verify your "
    "changes by reading the file back after writing. When refactoring, preserve "
    "existing behavior unless explicitly asked to change it. When reasoning through "
    "complex changes, think step by step but keep the explanation focused on the "
    "specific change being made. Avoid restating the user's request. If a request is "
    "ambiguous, ask one focused clarifying question rather than guessing. When a task "
    "requires multiple file edits, plan the edits first, then execute them in order, "
    "verifying each before moving to the next. Never modify files outside the user's "
    "project root. Never run destructive commands without explicit confirmation. "
) * 5
tools = [
    {"type": "function", "function": {"name": n, "description": d,
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "pattern": {"type": "string"},
         "command": {"type": "string"}, "content": {"type": "string"},
         "recursive": {"type": "boolean"}, "encoding": {"type": "string", "default": "utf-8"},
     }, "required": ["path"]}}}
    for n, d in [
        ("read_file", "Read the contents of a file at the given path."),
        ("write_file", "Write content to a file at the given path."),
        ("list_directory", "List files at the given path, optionally recursive."),
        ("search_code", "Search for a regex pattern across the codebase."),
        ("run_command", "Execute a shell command in the project directory."),
        ("get_file_metadata", "Get metadata for a file."),
        ("create_directory", "Create a directory."),
        ("delete_file", "Delete a file."),
        ("git_status", "Get the current git status."),
        ("git_diff", "Get the diff for current changes."),
    ]
]
user_text = (
    "I have a Python function `compute_metrics` in `src/analytics/metrics.py` that "
    "currently calculates running statistics by re-iterating the entire data list "
    "every call. Refactor it to maintain a streaming aggregation state that updates "
    "incrementally. Preserve the public API. Show me the diff before applying it."
)
body = {
    "model": model,
    "messages": [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_text},
    ],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": 2000,
    "temperature": 0.6,
    "top_p": 0.95,
    "stream": False,
}
with open(os.environ['REQ_FILE'], 'w') as f:
    json.dump(body, f)
PYEOF
  http_code="$(curl -sS -o "${resp_file}" -w "%{http_code}" --max-time 180 \
    -H "Content-Type: application/json" -X POST \
    -d "@${req_file}" "${URL}/v1/chat/completions" 2>/dev/null || echo "000")"
  rm -f "$req_file"
  case "$http_code" in
    200)
      body="$(cat "${resp_file}")"
      local finish content_len
      finish="$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0].get('finish_reason') or '?')" 2>/dev/null || echo "?")"
      content_len="$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0].get('message') or {}; c=m.get('content') or ''; tc=m.get('tool_calls') or []; print(f'{len(c)},{len(tc)}')" 2>/dev/null || echo "0,0")"
      pass "IDE-agent one-shot OK — finish=${finish} content_chars=${content_len%,*} tool_calls=${content_len##*,}"
      ;;
    500)
      fail "HTTP 500 — likely Cliff 1 mech B (inductor FFN intermediate OOM)" \
           "This is club-3090#16. Real IDE-agent workloads will crash on this compose. Switch to tools-text.yml (fp8 KV path with PN8) until Genesis PN25 lands default-on. Server logs: docker logs ${CONTAINER} 2>&1 | grep -B5 -A5 empty_strided_cuda"
      ;;
    000)
      fail "no HTTP response (timeout or container died)" \
           "Engine likely crashed. Check: docker logs ${CONTAINER} 2>&1 | tail -50"
      ;;
    *)
      fail "unexpected HTTP ${http_code}" \
           "Body head: $(head -c 200 "${resp_file}" 2>/dev/null)"
      ;;
  esac
  local rc=$?
  rm -f "$resp_file"
  return "$rc"
}
run_check "ide_agent" check_ide_agent

echo ""
if [[ "$FAILED" == "0" ]]; then
  printf "\033[32mAll stress / boundary checks passed.\033[0m KV-cache and prefill paths are sound for the deployed config.\n"
else
  printf "\033[31m%d stress check(s) failed.\033[0m See hints above.\n" "$FAILED"
fi
exit "$FAILED"
