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
# This is SLOW (large prompts, long-ctx needle ladder from ~10K up to
# ~92% of CTX_SIZE, ~25K-token tool prefill) — allow ~10–20 minutes on
# dual-card, longer on single-card configurations where the longest depths
# get rejected by the engine pre-check (HTTP 400, treated as graceful skip).
#
# Checks (in order — Cliff 2 territory deferred to last, ceiling rung last):
#   1. Long-context needle SMALL rungs (10K + 30K) — recall ladder at 2
#      depths that DON'T hit Cliff 2. Each depth gets its own random secret
#      to defeat caching. Depths above the deployed --max-model-len are
#      gracefully skipped via the engine's HTTP 400 pre-check.
#   2. Tool response prefill OOM — multi-turn payload with ~25K-token mock
#      tool message + tool definition + auto tool_choice; catches the
#      activation-memory peak class of bug.
#   3. IDE-agent one-shot — synthetic Cline/OpenCode-shape prompt: ~5K-char
#      sys preamble + 10 tool schemas + ~350-char user request + max_tokens=2000.
#      Catches Cliff 1 mech B (inductor compile-path FFN intermediate leak).
#   4. Multi-turn agent — sys + tools + user → assistant tool_call → tool reply
#      → user followup. Different inductor compile path than single-turn (#3).
#   5. LCB-coding shape — LeetCode-style problem statement + structured plan
#      request + max_tokens=4096. Catches DS conv state crash class.
#   6. Reasoning-heavy — math/algorithm problem + max_tokens=8192 to give the
#      model real reasoning room. Stresses spec-decode AL collapse + mamba
#      cache_mode='align' interactions.
#   7. Long-context needle LARGE rungs (60K + 90K) — Cliff 2 territory.
#      On 24 GB single-card, 60K+ single-prompt may crash the engine (DeltaNet
#      GDN forward state OOM). Putting it late preserves engine liveness for
#      probes 2-6. On dual-card or higher-VRAM rigs that can carry 60K+ this
#      passes.
#   8. Context CEILING ladder (CTX_SIZE-scaled, #199) — staggers NIAH rungs
#      from ~95K tokens up to ~92% of n_ctx in ~30K increments (configurable
#      via CEILING_STEP_TOKENS). Each rung captures VRAM before/after so you
#      get a margin curve, not just pass/fail. Stops at the first failure —
#      that depth IS the real ceiling. Catches the false-ceiling class of bug
#      (#197: hermes agent OOM at 127K on a compose that "passed" at 90K).
#      Example ladder for a 262K compose: 95K → 125K → 155K → 185K → 215K → 241K.
#
# Usage:
#   CONTAINER=<your-container> bash scripts/verify-stress.sh
#
# Env (optional):
#   URL                    Default: http://localhost:8020
#   MODEL                  Default: auto-detected from /v1/models, else
#                          qwen3.6-27b-autoround
#   CONTAINER              Default: vllm-qwen36-27b
#   SKIP_LONGCTX           Set to 1 to skip the long-context needle ladder.
#   SKIP_TOOL_PREFILL      Set to 1 to skip the tool-response prefill test.
#   SKIP_CEILING           Set to 1 to skip the context ceiling ladder (#199).
#   PREFILL_TARGET_CHARS   Tool-response prefill payload size in chars
#                          (default: 100000 ≈ 25K tokens; set higher to
#                          push closer to the cliff under investigation).
#   CEILING_FRACTION       Fraction of n_ctx to target for the top ceiling
#                          rung (default: 0.92). Lower to test a safer fill.
#   CEILING_STEP_TOKENS    Token increment between ceiling ladder rungs
#                          (default: 30000). Smaller = more precise wall
#                          location but more rungs (each rung is a full
#                          NIAH request at that depth).
#   CEILING_START_TOKENS   First ceiling rung target in tokens (default:
#                          95000 — just above probe 7's ~90K). The ladder
#                          starts here and steps up by CEILING_STEP_TOKENS
#                          until CEILING_FRACTION × n_ctx.
#   VRAM_MARGIN_MB         Minimum free-VRAM (MB) required after a ceiling
#                          rung. Below this → warn even on HTTP 200.
#                          (default: 1024 = 1 GB; absorbs agent checkpoint
#                          overhead that single-shot NIAH doesn't exercise).
#   STRESS_LONGCTX_TIMEOUT_S       Curl timeout for long-context needle checks
#                          (default: 300, auto-bumped to 600 if container has
#                          VLLM_ENFORCE_EAGER=1 — eager prefill at 60K+ can
#                          take 200-290s, see easel #102 follow-up).
#   STRESS_TOOL_PREFILL_TIMEOUT_S  Curl timeout for tool-prefill OOM check
#                          (default: 240, auto-bumped to 480 if container has
#                          VLLM_ENFORCE_EAGER=1).

set -euo pipefail

# Auto-detect running container + port (URL/CONTAINER env vars still win).
# See scripts/preflight.sh::preflight_autodetect_endpoint.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
  # shellcheck source=preflight.sh
  source "${ROOT_DIR}/scripts/preflight.sh"
  preflight_autodetect_endpoint
fi
URL="${URL:-http://localhost:8020}"
# Resolve the served model from /v1/models when MODEL is unset (#372). The qwen
# literal below is only a last resort if detection no-ops (endpoint unreachable).
declare -F preflight_autodetect_model >/dev/null && preflight_autodetect_model
MODEL="${MODEL:-qwen3.6-27b-autoround}"
CONTAINER="${CONTAINER:-vllm-qwen36-27b}"

# Detect VLLM_ENFORCE_EAGER=1 in the running container's env. Eager-mode
# prefill at 60K-140K can take 200-290s (vs <60s with CUDA graphs); the
# default curl timeouts below would false-positive as HTTP 000 on rigs that
# need eager mode to fit (typical for WSL2 / laptop GPUs at long ctx, see
# easel #102). When detected, scale the long-ctx + tool-prefill timeouts up.
EAGER_MODE_DETECTED=0
if command -v docker >/dev/null 2>&1; then
  if docker inspect "${CONTAINER}" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
       | grep -qE '^VLLM_ENFORCE_EAGER=1$'; then
    EAGER_MODE_DETECTED=1
  fi
fi
if [[ "${EAGER_MODE_DETECTED}" == "1" ]]; then
  STRESS_LONGCTX_TIMEOUT_S="${STRESS_LONGCTX_TIMEOUT_S:-600}"
  STRESS_TOOL_PREFILL_TIMEOUT_S="${STRESS_TOOL_PREFILL_TIMEOUT_S:-480}"
else
  STRESS_LONGCTX_TIMEOUT_S="${STRESS_LONGCTX_TIMEOUT_S:-300}"
  STRESS_TOOL_PREFILL_TIMEOUT_S="${STRESS_TOOL_PREFILL_TIMEOUT_S:-240}"
fi

pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; printf "    \033[33m→\033[0m %s\n" "$2"; return 1; }
skip() { printf "  \033[33m⊘\033[0m %s (skipped)\n" "$1"; }

FAILED=0
run_check() {
  local label="$1"; shift
  if "$@"; then :; else FAILED=$((FAILED + 1)); fi
}

# ---- Engine detection (parallel to verify-full.sh::detect_engine, see #87) ---
# Used to emit engine-aware diagnostic hints in fail() messages instead of
# always saying "Check: docker logs $CONTAINER" — meaningless to a host-build
# llama.cpp user. Engine class is detected once at startup and cached.
detect_engine() {
  if curl -sf -m 3 "${URL}/props" >/dev/null 2>&1; then
    echo "llamacpp"; return 0
  fi
  local fp
  fp="$(curl -sf -m 5 "${URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('system_fingerprint','') or '')" 2>/dev/null)"
  case "$fp" in
    vllm-*)   echo "vllm"; return 0 ;;
    sglang-*) echo "sglang"; return 0 ;;
  esac
  case "$CONTAINER" in
    vllm-*)      echo "vllm"; return 0 ;;
    llama-cpp-*) echo "llamacpp"; return 0 ;;
  esac
  echo "unknown"
}
ENGINE_KIND="$(detect_engine)"

# Engine-aware "where to find logs" string for fail() diagnostic hints.
# vLLM users want "docker logs $CONTAINER"; llama.cpp host-build users want
# "stdout/stderr where you launched llama-server"; etc. Computed once.
case "$ENGINE_KIND" in
  vllm|sglang) LOG_CMD="docker logs ${CONTAINER} 2>&1 | tail -50" ;;
  llamacpp)
    if [[ "$CONTAINER" == "none" ]]; then
      LOG_CMD="check llama-server stdout/stderr where you launched it"
    else
      LOG_CMD="docker logs ${CONTAINER} 2>&1 | tail -50"
    fi ;;
  *) LOG_CMD="check your engine's stdout/stderr or container logs" ;;
esac

# --------------------------------------------------------------------
# Streaming NIAH helper — sends a needle-in-haystack request with
# stream:true, measures TTFT, extracts prefill throughput.
#
# Usage: send_streaming_niah <req_file> <result_file> <url> <timeout_s>
#
# Writes a JSON result to <result_file> with:
#   http_code, content, prompt_tokens, completion_tokens,
#   ttft_ms, total_wall_ms,
#   prefill_tps, prefill_ms, prefill_n
#
# Prefill throughput:
#   - Primary (llama.cpp): timings.prompt_per_second + timings.prompt_ms
#     from the final streaming chunk (confirmed against live endpoint).
#   - Fallback (cross-engine): prompt_tokens / TTFT_seconds.
#   - If neither available: prefill_tps=null.
# --------------------------------------------------------------------
send_streaming_niah() {
  local _req_file="$1" _result_file="$2" _url="$3" _timeout="$4"
  python3 - "$_req_file" "$_result_file" "$_url" "$_timeout" <<'PYEOF'
import json, os, sys, time, urllib.request, urllib.error

req_file, result_file, url, timeout_s = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

with open(req_file) as f:
    req = json.load(f)

# Enable streaming + usage in final chunk
req["stream"] = True
req["stream_options"] = {"include_usage": True}

body = json.dumps(req).encode("utf-8")
http_req = urllib.request.Request(
    f"{url}/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)

t_send = time.time()
ttft = None
content_parts = []
usage = None
timings = None
result = {"http_code": 0, "error": None}

try:
    with urllib.request.urlopen(http_req, timeout=timeout_s) as resp:
        result["http_code"] = resp.getcode() or 200
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").rstrip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # First content token → TTFT
            choices = chunk.get("choices") or []
            if choices and ttft is None:
                delta = choices[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft = time.time() - t_send

            # Accumulate streamed content
            if choices:
                c = (choices[0].get("delta", {}).get("content", "") or "")
                if c:
                    content_parts.append(c)

            # Final chunk carries usage + timings (llama.cpp)
            if "usage" in chunk:
                usage = chunk["usage"]
            if "timings" in chunk:
                timings = chunk["timings"]

    t_total = time.time() - t_send

    result["content"] = "".join(content_parts)
    result["prompt_tokens"] = (usage or {}).get("prompt_tokens", 0)
    result["completion_tokens"] = (usage or {}).get("completion_tokens", 0)
    result["ttft_ms"] = round(ttft * 1000) if ttft is not None else None
    result["total_wall_ms"] = round(t_total * 1000)

    # Prefill throughput: prefer llama.cpp timings (server-measured, precise)
    if timings and timings.get("prompt_per_second"):
        result["prefill_tps"] = round(timings["prompt_per_second"], 1)
        result["prefill_ms"] = round(timings["prompt_ms"], 1) if timings.get("prompt_ms") else None
        result["prefill_n"] = timings.get("prompt_n")
    elif ttft and usage and usage.get("prompt_tokens") and ttft > 0:
        # Cross-engine fallback: prompt_tokens / TTFT
        result["prefill_tps"] = round(usage["prompt_tokens"] / ttft, 1)
        result["prefill_ms"] = round(ttft * 1000)
        result["prefill_n"] = usage["prompt_tokens"]
    else:
        result["prefill_tps"] = None
        result["prefill_ms"] = None
        result["prefill_n"] = None

except urllib.error.HTTPError as e:
    result["http_code"] = e.code
    result["error"] = str(e)
    # Try to read the error body for diagnostics
    try:
        result["error_body"] = e.read().decode("utf-8", errors="replace")[:500]
    except Exception:
        pass
except Exception as e:
    result["http_code"] = 0
    result["error"] = str(e)

with open(result_file, "w") as f:
    json.dump(result, f)
PYEOF
}

echo "Running STRESS / boundary test against ${URL}"
echo "  model=${MODEL}  container=${CONTAINER}  engine=${ENGINE_KIND}"
echo "  This script does the heavy stuff (longctx needle ladder + ~25K-token tool prefill)."
echo "  For the fast functional smoke (~2 min), use verify-full.sh instead."
echo ""

# Some failure-mode hints in this script are vLLM-specific (Genesis env vars,
# club-3090 issue references, etc.). They're emitted regardless of engine but
# generic mode shows the same actionable info to non-vLLM users; only the
# "where to find logs" strings adapt to engine class above.

# --------------------------------------------------------------------
# 1. Long-context needle — put a secret at ~50% depth, ask for it at the end
# --------------------------------------------------------------------
check_longctx() {
  # Header only when called from probe 1 (default); probe 7 (large rungs)
  # prints its own header before calling us.
  if [[ -z "${LONGCTX_SCALES:-}" ]]; then
    echo "[1/8] Long-context needle small rungs (10K / 30K) ..."
  fi
  if [[ "${SKIP_LONGCTX:-0}" == "1" ]]; then
    skip "SKIP_LONGCTX=1"
    return 0
  fi

  local any_fail=0
  local any_pass=0
  local any_skipped=0
  local any_recall_miss=0

  local deployed_max
  deployed_max="$(curl -sf -m 5 "${URL}/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0].get('max_model_len',0))" 2>/dev/null \
    || echo 0)"

  # Split: small-rung needles (10K + 30K) run as probe 1 — they exercise
  # long-context attention quality at depths that DON'T hit Cliff 2. The
  # large-rung needles (60K + 90K) run last as probe 7, since hitting
  # Cliff 2 (DeltaNet GDN forward state OOM) on a 24 GB single card
  # crashes the engine and would cascade-fail all subsequent probes.
  # Override which set runs via $LONGCTX_SCALES env (default: small rungs).
  local _longctx_scales="${LONGCTX_SCALES:-150 450}"
  for filler_scale in $_longctx_scales; do
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
    local result_file http_code
    result_file="$(mktemp --suffix=.json)"
    send_streaming_niah "$req_file" "$result_file" "$URL" "$STRESS_LONGCTX_TIMEOUT_S"
    rm -f "$secret_file" "$req_file"
    http_code="$(python3 -c "import json; print(json.load(open('$result_file'))['http_code'])" 2>/dev/null || echo 0)"
    if [[ "$http_code" == "400" ]]; then
      printf "    \033[33m⊘\033[0m scale=%d: HTTP 400 (exceeds --max-model-len, expected — clean rejection)\n" "$filler_scale"
      rm -f "$result_file"
      any_skipped=1
      continue
    elif [[ "$http_code" != "200" ]]; then
      printf "    \033[31m✗\033[0m scale=%d: HTTP %s (request failed)\n" "$filler_scale" "$http_code"
      rm -f "$result_file"
      any_fail=1
      continue
    fi
    local prompt_tok content_raw prefill_tps prefill_ms
    prompt_tok="$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('prompt_tokens',0))" 2>/dev/null || echo 0)"
    content_raw="$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('content',''))" 2>/dev/null || echo '')"
    prefill_tps="$(python3 -c "import json; d=json.load(open('$result_file')); v=d.get('prefill_tps'); print(v if v is not None else '')" 2>/dev/null || echo '')"
    prefill_ms="$(python3 -c "import json; d=json.load(open('$result_file')); v=d.get('prefill_ms'); print(v if v is not None else '')" 2>/dev/null || echo '')"
    rm -f "$result_file"
    local prefill_str=""
    if [[ -n "$prefill_tps" ]]; then
      prefill_str="  prefill=${prefill_tps} t/s"
      if [[ -n "$prefill_ms" ]]; then
        local prefill_s
        prefill_s="$(python3 -c "print(f'{${prefill_ms}/1000:.0f}')" 2>/dev/null || echo '')"
        if [[ -n "$prefill_s" ]]; then
          prefill_str="  prefill=${prefill_tps} t/s (${prefill_s}s)"
        fi
      fi
    fi
    local all_match=1
    for tok in $secret; do
      echo "$content_raw" | grep -qiF "$tok" || all_match=0
    done
    if [[ "$all_match" == "1" ]]; then
      printf "    \033[32m✓\033[0m %6s tokens: recalled '%s' (got: %s)%s\n" "$prompt_tok" "$secret" "$(echo "$content_raw" | head -c 60 | tr '\n' ' ')" "$prefill_str"
      any_pass=1
    else
      # Recall miss at HTTP 200 = attention quality degradation, not a system
      # failure. Log it as informational (yellow △), break out of the ladder
      # — deeper rungs will only be worse — and pass the probe so the
      # pipeline moves to the next stage.
      printf "    \033[33m△\033[0m %6s tokens: recall MISS (expected '%s', got '%s') — system OK, quality ceiling reached%s\n" "$prompt_tok" "$secret" "$(echo "$content_raw" | head -c 80 | tr '\n' ' ')" "$prefill_str"
      any_recall_miss=1
      break
    fi
  done

  if [[ "$any_fail" == "0" ]] && [[ "$any_pass" == "1" ]]; then
    if [[ "$any_skipped" == "1" ]]; then
      pass "all in-budget long-ctx depths recalled secret (above-budget depths cleanly rejected by engine pre-check)"
    elif [[ "$any_recall_miss" == "1" ]]; then
      pass "all system requests succeeded (some recall misses at depth — attention quality, not system health)"
    else
      pass "all long-ctx depths recalled secret correctly"
    fi
  elif [[ "$any_fail" == "0" ]] && [[ "$any_pass" == "0" ]]; then
    if [[ "$any_recall_miss" == "1" ]]; then
      pass "all system requests succeeded (all recall missed — attention quality degraded, but system filled every depth)"
    else
      skip "all depths above --max-model-len (deployed=${deployed_max:-unknown}); shrink ladder or raise ctx"
    fi
  else
    fail "system-level failures during long-ctx needle ladder (HTTP 5xx / timeout / crash)" \
         "One or more depths returned non-200 (not a recall miss). Check: ${LOG_CMD}"
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
  echo "[2/8] Tool response prefill OOM (~25K-token mock tool response) ..."
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
  http_code="$(curl -s -m "${STRESS_TOOL_PREFILL_TIMEOUT_S}" -o "${resp_file}" -w '%{http_code}' \
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
             "Likely silent prefill truncation. Check warnings: ${LOG_CMD}"
      fi
      ;;
    500)
      fail "HTTP 500 — OOM during ~25K-token tool-response prefill" \
           "Activation memory peak exceeded budget. Lower --max-model-len or --gpu-memory-utilization. See README 'Activation memory caveat'. Server logs: ${LOG_CMD}"
      ;;
    000)
      fail "no HTTP response (timeout or container died)" \
           "Prefill may have hung or container OOM-killed. Check: ${LOG_CMD}; nvidia-smi"
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
echo "[3/8] IDE-agent one-shot prompt (sys + tool schemas + user request) ..."
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
    # tool_choice="none" forces content-only output (no tool_calls),
    # which makes the model go through long-reasoning + code emission —
    # the path that triggers Cliff 1 mech B inductor leak. With "auto"
    # the model can short-circuit by emitting a tool_call and exit
    # before hitting the inductor-compiled reasoning forward, hiding
    # the bug. We want the bug to surface deterministically.
    "tool_choice": "none",
    "max_tokens": 2000,
    "temperature": 0.0,
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
      local finish content_chars completion_tokens
      finish="$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0].get('finish_reason') or '?')" 2>/dev/null || echo "?")"
      content_chars="$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0].get('message') or {}; print(len(m.get('content') or ''))" 2>/dev/null || echo "0")"
      completion_tokens="$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens', 0))" 2>/dev/null || echo "0")"
      # The bug we care about (Cliff 1 mech B) crashes the engine — that's
      # HTTP 500. Any HTTP 200 means the inductor compile path actually
      # executed without ICE'ing. Token count low is fine; the model just
      # decided the request didn't need a long answer. Don't fail on length.
      pass "IDE-agent one-shot OK — ${completion_tokens} completion tokens (${content_chars} chars), finish=${finish}"
      ;;
    500)
      fail "HTTP 500 — likely Cliff 1 mech B (inductor FFN intermediate OOM)" \
           "This is club-3090#16. Real IDE-agent workloads will crash on this compose. Switch to tools-text.yml (fp8 KV path with PN8) until Genesis PN25 lands default-on. Server logs: docker logs ${CONTAINER} 2>&1 | grep -B5 -A5 empty_strided_cuda"
      ;;
    000)
      fail "no HTTP response (timeout or container died)" \
           "Engine likely crashed. Check: ${LOG_CMD}"
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

# 4. Multi-turn agent — sys + tools + user → assistant(tool_call) → tool reply
# → user followup. Different inductor compile path than check #3 (single-turn)
# because the assistant + tool messages reshape the prefill that gets compiled.
echo "[4/8] Multi-turn agent prompt (sys + tools + 4-turn history) ..."
check_multiturn_agent() {
  local req_file resp_file http_code body
  req_file="$(mktemp --suffix=.json)"
  resp_file="$(mktemp --suffix=.json)"
  MODEL_VAR="${MODEL}" REQ_FILE="${req_file}" python3 - <<'PYEOF'
import json, os
model = os.environ['MODEL_VAR']
sys_text = (
    "You are a coding assistant inside an IDE. Use the provided tools to read "
    "and edit files. Be concise. After each tool call, verify the result before "
    "proceeding to the next step. "
) * 8
tools = [
    {"type": "function", "function": {"name": n, "description": d,
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"},
         "pattern": {"type": "string"},
     }, "required": ["path"]}}}
    for n, d in [
        ("read_file", "Read a file."),
        ("write_file", "Write a file."),
        ("search_code", "Search for a regex pattern."),
        ("list_directory", "List a directory."),
    ]
]
# Realistic 4-turn agent history: user asks → assistant calls tool →
# tool returns content → user follow-up. The tool reply is ~3K chars
# of mock file content (smaller than check #2's 25K, larger than check
# #3's empty history).
mock_file = "\n".join([
    f"def function_{i}(arg{i}): return arg{i} * {i+1}  # line {i}"
    for i in range(80)
])
body = {
    "model": model,
    "messages": [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": "Read src/utils.py and tell me what functions are defined."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_read_1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "src/utils.py"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_read_1", "content": mock_file},
        {"role": "user", "content": "Now refactor function_5 to use a different multiplier."},
    ],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": 1500,
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
      pass "multi-turn agent OK"
      ;;
    500)
      fail "HTTP 500 — multi-turn prefill crashed engine" \
           "Different compile path than check #3 — assistant + tool messages reshape the prefill. May indicate a separate inductor bug or different shape of the same Cliff 1 issue. Check: ${LOG_CMD}"
      ;;
    000)
      fail "no HTTP response (timeout or container died)" \
           "Engine likely crashed. Check: ${LOG_CMD}"
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
run_check "multiturn_agent" check_multiturn_agent

# 5. LCB-coding shape — LeetCode-style problem statement requesting structured
# plan + code. Catches DS conv state crash (genesis-vllm-patches#17) on configs
# where VLLM_SSM_CONV_STATE_LAYOUT=DS + spec-decode + AL>1 + this prompt shape
# trip the NotImplementedError in vllm/model_executor/layers/mamba/mamba_utils.py.
echo "[5/8] LCB-coding shape (LeetCode-style problem + structured plan) ..."
check_lcb_coding() {
  local req_file resp_file http_code body
  req_file="$(mktemp --suffix=.json)"
  resp_file="$(mktemp --suffix=.json)"
  MODEL_VAR="${MODEL}" REQ_FILE="${req_file}" python3 - <<'PYEOF'
import json, os
model = os.environ['MODEL_VAR']
problem = (
    "You are given an integer array nums. Return the length of the longest "
    "subarray with a sum equal to a target value k. If no such subarray exists, "
    "return 0.\n\n"
    "Example 1:\n"
    "Input: nums = [1, -1, 5, -2, 3], k = 3\n"
    "Output: 4\n"
    "Explanation: The subarray [1, -1, 5, -2] sums to 3 and has length 4.\n\n"
    "Example 2:\n"
    "Input: nums = [-2, -1, 2, 1], k = 1\n"
    "Output: 2\n\n"
    "Constraints:\n"
    "- 1 <= nums.length <= 2 * 10^5\n"
    "- -10^4 <= nums[i] <= 10^4\n"
    "- -10^9 <= k <= 10^9\n\n"
    "Plan your approach in the format:\n"
    "GOAL: <one-line restatement>\n"
    "STATE: <data structures>\n"
    "ALGO: <key steps>\n"
    "EDGE: <edge cases>\n"
    "VERIFY: <how to test>\n\n"
    "Then implement the solution as `class Solution: def maxSubArrayLen(...)`."
)
body = {
    "model": model,
    "messages": [{"role": "user", "content": problem}],
    "max_tokens": 4096,
    "temperature": 0.0,
    "stream": False,
}
with open(os.environ['REQ_FILE'], 'w') as f:
    json.dump(body, f)
PYEOF
  http_code="$(curl -sS -o "${resp_file}" -w "%{http_code}" --max-time 240 \
    -H "Content-Type: application/json" -X POST \
    -d "@${req_file}" "${URL}/v1/chat/completions" 2>/dev/null || echo "000")"
  rm -f "$req_file"
  case "$http_code" in
    200)
      pass "LCB-coding shape OK"
      ;;
    500)
      fail "HTTP 500 — LCB-coding shape crashed engine" \
           "Likely DS conv state crash (genesis-vllm-patches#17). Check: docker logs ${CONTAINER} 2>&1 | grep -B2 -A5 'DS conv state\\|NotImplementedError'. Workaround: drop VLLM_SSM_CONV_STATE_LAYOUT=DS from compose env."
      ;;
    000)
      fail "no HTTP response (timeout or container died)" \
           "Engine likely crashed. Check: ${LOG_CMD}"
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
run_check "lcb_coding" check_lcb_coding

# 6. Reasoning-heavy — math/algorithm problem with max_tokens=8192 budget.
# Stresses spec-decode AL collapse and mamba_cache_mode='align' interactions
# over a long generation. Catches regressions where generation completes but
# AL collapses past a certain decode depth, or where long generations trigger
# state-copy bugs that don't fire on short outputs.
echo "[6/8] Reasoning-heavy (math problem + max_tokens=8192) ..."
check_reasoning_heavy() {
  local req_file resp_file http_code body
  req_file="$(mktemp --suffix=.json)"
  resp_file="$(mktemp --suffix=.json)"
  MODEL_VAR="${MODEL}" REQ_FILE="${req_file}" python3 - <<'PYEOF'
import json, os
model = os.environ['MODEL_VAR']
problem = (
    "Prove that for any positive integer n, the sum 1^3 + 2^3 + 3^3 + ... + n^3 "
    "equals (n(n+1)/2)^2. Show every step of your reasoning, including:\n"
    "1. The base case verification.\n"
    "2. The inductive hypothesis.\n"
    "3. The full algebraic manipulation in the inductive step.\n"
    "4. A geometric or visual interpretation if you can think of one.\n"
    "5. A verification by computing both sides for n=1, 2, 3, 4, 5.\n\n"
    "Be thorough; show every algebraic step rather than skipping any. After the "
    "proof, also derive a closed-form expression for the sum 1^4 + 2^4 + ... + n^4 "
    "using the same induction technique, and verify it for n=1, 2, 3."
)
body = {
    "model": model,
    "messages": [{"role": "user", "content": problem}],
    "max_tokens": 8192,
    "temperature": 0.0,
    "stream": False,
}
with open(os.environ['REQ_FILE'], 'w') as f:
    json.dump(body, f)
PYEOF
  http_code="$(curl -sS -o "${resp_file}" -w "%{http_code}" --max-time 600 \
    -H "Content-Type: application/json" -X POST \
    -d "@${req_file}" "${URL}/v1/chat/completions" 2>/dev/null || echo "000")"
  rm -f "$req_file"
  case "$http_code" in
    200)
      body="$(cat "${resp_file}")"
      local completion_tokens
      completion_tokens="$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens', 0))" 2>/dev/null || echo "0")"
      if [[ "$completion_tokens" -lt 500 ]]; then
        fail "reasoning-heavy returned only ${completion_tokens} tokens (expected >500 for max=8192)" \
             "Possible spec-decode AL collapse or early stop. Check finish_reason."
      else
        pass "reasoning-heavy OK — ${completion_tokens} completion tokens"
      fi
      ;;
    500)
      fail "HTTP 500 — long-generation crashed engine" \
           "Possible mamba state-copy bug at deeper decode positions. Check: ${LOG_CMD}"
      ;;
    000)
      fail "no HTTP response (timeout or container died)" \
           "Engine likely crashed during long generation. Check: ${LOG_CMD}"
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
run_check "reasoning_heavy" check_reasoning_heavy

# 7. Long-context needle large rungs (60K + 90K) — runs LAST because hitting
# Cliff 2 (DeltaNet GDN forward state OOM at 50-60K single-prompt) on a 24 GB
# single card crashes the engine. We want all the OTHER probes to run on a
# live engine first; this probe is the architectural ceiling check.
echo "[7/8] Long-context needle large rungs (60K / 90K — Cliff 2 territory) ..."
check_longctx_large() {
  if [[ "${SKIP_LONGCTX:-0}" == "1" ]]; then
    skip "SKIP_LONGCTX=1"
    return 0
  fi
  LONGCTX_SCALES="900 1400" check_longctx
}
# Engine health check after crash-prone probes (7, 8).
# If the engine died (OOM, crash), attempt to restart the container so
# subsequent pipeline steps (soak-test, quality-test) don't cascade-fail.
# No-op when CONTAINER=none (endpoint-first mode — can't restart a host server).
engine_healthy() {
  curl -sf -m 5 "${URL}/v1/models" >/dev/null 2>&1
}

ensure_engine_alive() {
  local probe_label="$1"
  if engine_healthy; then
    return 0
  fi

  echo "    \033[33m⚠\033[0m ${probe_label} crashed the engine — attempting restart…"

  if [[ "${CONTAINER:-}" == "none" ]] || ! command -v docker >/dev/null 2>&1; then
    echo "    \033[31m✗\033[0m Cannot restart (CONTAINER=none or docker unavailable)."
    echo "      Subsequent pipeline steps will fail against a dead engine."
    echo "      Restart manually before running soak-test / quality-test."
    return 1
  fi

  if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    echo "    \033[31m✗\033[0m Container '${CONTAINER}' not found. Cannot restart."
    return 1
  fi

  echo "    Restarting container '${CONTAINER}'…"
  docker restart "${CONTAINER}" >/dev/null 2>&1 || true

  # Poll for engine recovery (up to 120s — model reload can take a while)
  local waited=0 max_wait=120
  while [[ $waited -lt $max_wait ]]; do
    if engine_healthy; then
      echo "    \033[32m✓\033[0m Engine recovered after ${waited}s"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done

  echo "    \033[31m✗\033[0m Engine did not recover after ${max_wait}s"
  echo "      Check: ${LOG_CMD}"
  return 1
}

run_check "longctx_large" check_longctx_large
ensure_engine_alive "probe 7 (large rungs)" || true

# --------------------------------------------------------------------
# 8. Context CEILING ladder (staggered NIAH from ~95K → ~92% × n_ctx)
#    (#199). Staggers NIAH rungs from ~95K tokens up to ~92% of n_ctx
#    in ~30K increments. Each rung captures VRAM before/after so you
#    get a margin curve, not just pass/fail. Stops at the first failure
#    or recall miss. Catches the false-ceiling class of bug (#197).
# --------------------------------------------------------------------

# Detect the server's context window size from its API.
# llama.cpp: /props → default_generation_settings.n_ctx (nested, NOT top-level)
#            or docker inspect → -c / --ctx-size
# vLLM/SGLang: /v1/models → data[0].max_model_len
#              or docker inspect → --max-model-len
# Returns 0 when detection fails (caller should skip gracefully).
get_n_ctx() {
  local n_ctx=0

  # llama.cpp: GET /props → default_generation_settings.n_ctx
  # Real llama.cpp nests n_ctx inside default_generation_settings, NOT at
  # top-level. Top-level d.get('n_ctx') returns None → 0.
  if [[ "$ENGINE_KIND" == "llamacpp" ]] || curl -sf -m 3 "${URL}/props" >/dev/null 2>&1; then
    n_ctx="$(curl -sf -m 5 "${URL}/props" 2>/dev/null \
      | python3 -c "
import json, sys
d = json.load(sys.stdin)
# Try nested first (real llama.cpp shape), then top-level (some forks)
nested = d.get('default_generation_settings', {})
v = nested.get('n_ctx') if isinstance(nested, dict) else None
if v is None:
    v = d.get('n_ctx', 0)
print(v or 0)
" 2>/dev/null || echo 0)"
  fi

  # vLLM/SGLang: GET /v1/models → data[0].max_model_len
  if [[ "${n_ctx:-0}" -le 0 ]]; then
    n_ctx="$(curl -sf -m 5 "${URL}/v1/models" 2>/dev/null \
      | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0].get('max_model_len', 0))" 2>/dev/null \
      || echo 0)"
  fi

  # Docker fallback: scrape container args for context-size flags.
  # vLLM/SGLang: --max-model-len
  # llama.cpp:   -c <N> or --ctx-size <N>
  if [[ "${n_ctx:-0}" -le 0 && "${CONTAINER:-}" != "none" ]] \
     && command -v docker >/dev/null 2>&1 \
     && docker inspect "$CONTAINER" >/dev/null 2>&1; then
    n_ctx="$(docker inspect "$CONTAINER" 2>/dev/null \
      | python3 -c "
import json, sys, re
try:
    cfg = json.load(sys.stdin)
    args = ' '.join(cfg[0].get('Config',{}).get('Cmd',[]) or [])
    args += ' ' + ' '.join(cfg[0].get('Args',[]) or [])
    # vLLM/SGLang
    m = re.search(r'--max-model-len[=\s]+(\d+)', args)
    if m:
        print(m.group(1))
        sys.exit(0)
    # llama.cpp: -c NNNN or --ctx-size NNNN or --ctx-size=NNNN
    m = re.search(r'(?:^|\s)(?:-c|--ctx-size)[=\s]+(\d+)', args)
    if m:
        print(m.group(1))
        sys.exit(0)
    print(0)
except Exception:
    print(0)
" 2>/dev/null || echo 0)"
  fi

  echo "${n_ctx:-0}"
}

# Capture free VRAM for the model's GPU(s) only (sum, in MB).
# On a multi-GPU host running a single-card compose (CUDA_VISIBLE_DEVICES=0),
# summing ALL GPUs inflates the margin with idle card memory — defeating the
# gate. We read the container's DeviceRequests (Docker) or CUDA_VISIBLE_DEVICES
# env to identify which GPU(s) the model actually uses.
#
# Priority:
#   1. Docker HostConfig.DeviceRequests[0].DeviceIDs (compose device_ids)
#   2. Container env CUDA_VISIBLE_DEVICES
#   3. Container env NVIDIA_VISIBLE_DEVICES (if numeric, e.g. "0")
#   4. Fallback: sum all GPUs with a warning (multi-GPU host, can't determine subset)
# Returns 0 when nvidia-smi is unavailable.
get_vram_free_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi

  local gpu_ids=""

  # Try Docker DeviceRequests (compose device_ids: ["0"])
  if [[ "${CONTAINER:-}" != "none" ]] \
     && command -v docker >/dev/null 2>&1 \
     && docker inspect "$CONTAINER" >/dev/null 2>&1; then
    gpu_ids="$(docker inspect "$CONTAINER" 2>/dev/null \
      | python3 -c "
import json, sys
try:
    cfg = json.load(sys.stdin)[0]
    drs = cfg.get('HostConfig', {}).get('DeviceRequests', []) or []
    for dr in drs:
        ids = dr.get('DeviceIDs', [])
        if ids:
            print(','.join(ids))
            sys.exit(0)
    # Fallback: check env vars
    for e in cfg.get('Config', {}).get('Env', []) or []:
        if e.startswith('CUDA_VISIBLE_DEVICES='):
            val = e.split('=', 1)[1]
            if val and val != 'all':
                print(val)
                sys.exit(0)
        if e.startswith('NVIDIA_VISIBLE_DEVICES='):
            val = e.split('=', 1)[1]
            if val and val != 'all' and val.replace(',', '').isdigit():
                print(val)
                sys.exit(0)
    print('')
except Exception:
    print('')
" 2>/dev/null)"
  fi

  if [[ -n "$gpu_ids" ]]; then
    nvidia-smi -i "$gpu_ids" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
      | awk '{s+=$1} END {printf "%.0f\n", s}' 2>/dev/null || echo 0
  else
    # Can't determine which GPUs — sum all, but flag it
    local total_gpus
    total_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
    if [[ "$total_gpus" -gt 1 ]]; then
      echo "    [vram] WARN: could not determine model GPU(s) on ${total_gpus}-GPU host — summing all (margin may be inflated)" >&2
    fi
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
      | awk '{s+=$1} END {printf "%.0f\n", s}' 2>/dev/null || echo 0
  fi
}

echo "[8/8] Context ceiling ladder (staggered NIAH from ~${CEILING_START_TOKENS:-95000} → ~${CEILING_FRACTION:-0.92} × n_ctx) ..."
check_ceiling_ladder() {
  if [[ "${SKIP_CEILING:-0}" == "1" ]]; then
    skip "SKIP_CEILING=1"
    return 0
  fi
  if [[ "${SKIP_LONGCTX:-0}" == "1" ]]; then
    skip "SKIP_LONGCTX=1 (also skips ceiling ladder)"
    return 0
  fi

  local ceiling_fraction="${CEILING_FRACTION:-0.92}"
  local ceiling_step="${CEILING_STEP_TOKENS:-30000}"
  local ceiling_start="${CEILING_START_TOKENS:-95000}"
  local vram_margin_mb="${VRAM_MARGIN_MB:-1024}"

  # Step 1: detect context window
  local n_ctx
  n_ctx="$(get_n_ctx)"
  if [[ "${n_ctx:-0}" -le 0 ]]; then
    skip "could not detect n_ctx from endpoint (no /props, no /v1/models max_model_len, no docker inspect)"
    return 0
  fi

  # Step 2: build the ladder — from ceiling_start to ceiling_fraction*n_ctx
  local ceiling_top
  ceiling_top="$(python3 -c "print(int(${n_ctx} * ${ceiling_fraction}))")"
  # If the compose is small enough that probe 7 already covers it, skip
  if [[ "$ceiling_top" -le "$ceiling_start" ]]; then
    skip "n_ctx=${n_ctx} → ceiling target ${ceiling_top} ≤ start ${ceiling_start} (probe 7 already covers this range)"
    return 0
  fi

  # Compute rungs as a space-separated list
  local rungs
  rungs="$(python3 -c "
start, top, step = ${ceiling_start}, ${ceiling_top}, ${ceiling_step}
rungs = list(range(start, top, step))
if not rungs or rungs[-1] != top:
    rungs.append(top)
print(' '.join(str(r) for r in rungs))
")"
  local rung_count
  rung_count="$(echo "$rungs" | wc -w)"

  echo "    n_ctx=${n_ctx}  ladder: $(echo "$rungs" | sed 's/ / → /g') (${rung_count} rungs)"

  # Per-rung timeout: large contexts need more time for prefill
  local rung_timeout="${STRESS_CEILING_TIMEOUT_S:-600}"
  if [[ "$EAGER_MODE_DETECTED" == "1" ]]; then
    rung_timeout="${STRESS_CEILING_TIMEOUT_S:-900}"
  fi

  # Step 2b: calibrate filler→token ratio with a small probe.
  # The old heuristic (target_tokens / 3.5) was ~18x off for this tokenizer
  # (block is 290 chars × scale repetitions; the "3.5" treated scale as chars
  # not block-reps, producing prompts far exceeding n_ctx → HTTP 400).
  # Send a scale=100 probe, read back prompt_tokens, compute the real ratio.
  local tok_per_scale_unit=0
  local cal_req cal_result
  cal_req="$(mktemp --suffix=.json)"
  cal_result="$(mktemp --suffix=.json)"
  python3 -c "
import json
block = (
    'This section describes the history of computing in detail. '
    'Transistors were invented in 1947 at Bell Labs. The integrated circuit came a decade later. '
    'Microprocessors emerged in the 1970s and changed the world. '
    'Personal computing followed, then networking, then the web, then cloud and AI. '
) * 100
req = {
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': block + '\n\nHi.'}],
    'max_tokens': 5, 'temperature': 0.0,
    'chat_template_kwargs': {'enable_thinking': False},
}
with open('${cal_req}', 'w') as f:
    json.dump(req, f)
" 2>/dev/null
  send_streaming_niah "$cal_req" "$cal_result" "$URL" 60 2>/dev/null
  local cal_tokens
  cal_tokens="$(python3 -c "import json; print(json.load(open('$cal_result')).get('prompt_tokens', 0))" 2>/dev/null || echo 0)"
  rm -f "$cal_req" "$cal_result"
  if [[ "$cal_tokens" -gt 0 ]]; then
    # tokens_per_scale_unit = cal_tokens / 100 (we sent scale=100 worth of blocks)
    tok_per_scale_unit="$(python3 -c "print(round(${cal_tokens} / 100, 2))" 2>/dev/null || echo 0)"
    echo "    calibrated: scale=100 → ${cal_tokens} tokens (tok/scale_unit=${tok_per_scale_unit})"
  else
    # Fallback: use a conservative estimate (65 tok/scale_unit for Qwen tokenizers)
    tok_per_scale_unit=65
    echo "    calibration probe failed — using fallback tok/scale_unit=${tok_per_scale_unit}"
  fi

  # Step 3: run each rung
  local any_pass=0 any_fail=0 any_skipped=0 any_recall_miss=0 any_sizing_error=0
  local last_pass_tokens=0 last_pass_pct=0
  local first_fail_tokens=0 first_recall_miss_tokens=0 first_recall_miss_pct=0
  local vram_before_all vram_after_all
  vram_before_all="$(get_vram_free_mb)"
  if [[ "$vram_before_all" -gt 0 ]]; then
    echo "    VRAM free (ladder start): ${vram_before_all} MB"
  fi

  local rung_idx=0
  for target_tokens in $rungs; do
    rung_idx=$((rung_idx + 1))
    local filler_scale
    # scale = target_tokens / tok_per_scale_unit (calibrated, not hardcoded)
    filler_scale="$(python3 -c "print(max(100, int(${target_tokens} / ${tok_per_scale_unit})))")" 

    # Build + send the NIAH request
    local secret_file req_file http_code
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

    local result_file
    result_file="$(mktemp --suffix=.json)"
    send_streaming_niah "$req_file" "$result_file" "$URL" "$rung_timeout"
    rm -f "$secret_file" "$req_file"

    http_code="$(python3 -c "import json; print(json.load(open('$result_file'))['http_code'])" 2>/dev/null || echo 0)"

    # Capture VRAM after this rung
    local vram_after_rung
    vram_after_rung="$(get_vram_free_mb)"
    local vram_str=""
    if [[ "$vram_after_rung" -gt 0 ]]; then
      vram_str="  VRAM_free=${vram_after_rung}MB"
    fi

    # Extract prefill data from result
    local prefill_tps prefill_ms prefill_str
    prefill_tps="$(python3 -c "import json; d=json.load(open('$result_file')); v=d.get('prefill_tps'); print(v if v is not None else '')" 2>/dev/null || echo '')"
    prefill_ms="$(python3 -c "import json; d=json.load(open('$result_file')); v=d.get('prefill_ms'); print(v if v is not None else '')" 2>/dev/null || echo '')"
    prefill_str=""
    if [[ -n "$prefill_tps" ]]; then
      prefill_str="  prefill=${prefill_tps} t/s"
      if [[ -n "$prefill_ms" ]]; then
        local prefill_s
        prefill_s="$(python3 -c "print(f'{${prefill_ms}/1000:.0f}')" 2>/dev/null || echo '')"
        if [[ -n "$prefill_s" ]]; then
          prefill_str="  prefill=${prefill_tps} t/s (${prefill_s}s)"
        fi
      fi
    fi

    # Evaluate
    case "$http_code" in
      200)
        local prompt_tok content_raw all_match
        prompt_tok="$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('prompt_tokens',0))" 2>/dev/null || echo 0)"
        content_raw="$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('content',''))" 2>/dev/null || echo '')"
        rm -f "$result_file"
        all_match=1
        for tok in $secret; do
          echo "$content_raw" | grep -qiF "$tok" || all_match=0
        done
        local pct=0
        [[ "$prompt_tok" -gt 0 && "$n_ctx" -gt 0 ]] && pct=$(( prompt_tok * 100 / n_ctx ))
        if [[ "$all_match" == "1" ]]; then
          printf "    \033[32m✓\033[0m rung %d/%d: target=%dK  actual=%dK tok (%d%%)  recalled '%s'%s%s\n" \
            "$rung_idx" "$rung_count" "$((target_tokens / 1000))" "$((prompt_tok / 1000))" "$pct" "$secret" "$prefill_str" "$vram_str"
          any_pass=1
          last_pass_tokens="$prompt_tok"
          last_pass_pct="$pct"
        else
          # Recall miss = quality ceiling found. The system filled the context
          # but attention quality dropped — deeper rungs will only be worse.
          # Log it, break out of the ladder, and pass the probe so the
          # pipeline moves to the next stage without wasting time on
          # unreliable depths.
          printf "    \033[33m△\033[0m rung %d/%d: target=%dK  actual=%dK tok (%d%%)  recall MISS (got: '%s') — quality ceiling reached%s%s\n" \
            "$rung_idx" "$rung_count" "$((target_tokens / 1000))" "$((prompt_tok / 1000))" "$pct" \
            "$(echo "$content_raw" | head -c 60 | tr '\n' ' ')" "$prefill_str" "$vram_str"
          any_recall_miss=1
          if [[ "$first_recall_miss_tokens" -eq 0 ]]; then
            first_recall_miss_tokens="$prompt_tok"
            first_recall_miss_pct="$pct"
          fi
          break
        fi
        ;;
      400)
        # HTTP 400 can mean two things:
        #   (a) target > n_ctx → engine correctly rejected (legitimate skip)
        #   (b) target < n_ctx → our filler→token sizing overshot (BUG, not a skip)
        # Distinguish by comparing the rung target against n_ctx.
        if [[ "$target_tokens" -lt "$n_ctx" ]]; then
          printf "    \033[31m✗\033[0m rung %d/%d: target=%dK < n_ctx=%d but HTTP 400 — filler sizing overshot%s\n" \
            "$rung_idx" "$rung_count" "$((target_tokens / 1000))" "$n_ctx" "$vram_str"
          rm -f "$result_file"
          any_sizing_error=1
          any_fail=1
          if [[ "$first_fail_tokens" -eq 0 ]]; then
            first_fail_tokens="$target_tokens"
          fi
        else
          printf "    \033[33m⊘\033[0m rung %d/%d: target=%dK  HTTP 400 (exceeds engine limit — clean rejection)%s\n" \
            "$rung_idx" "$rung_count" "$((target_tokens / 1000))" "$vram_str"
          rm -f "$result_file"
          any_skipped=1
        fi
        # Either way — stop the ladder (deeper rungs will also 400)
        break
        ;;
      500)
        printf "    \033[31m✗\033[0m rung %d/%d: target=%dK  HTTP 500 (OOM at ~%d%% of n_ctx=%d)%s\n" \
          "$rung_idx" "$rung_count" "$((target_tokens / 1000))" \
          "$(( target_tokens * 100 / n_ctx ))" "$n_ctx" "$vram_str"
        rm -f "$result_file"
        any_fail=1
        if [[ "$first_fail_tokens" -eq 0 ]]; then
          first_fail_tokens="$target_tokens"
        fi
        # OOM = the wall. Stop the ladder.
        break
        ;;
      000)
        printf "    \033[31m✗\033[0m rung %d/%d: target=%dK  timeout/crash (>%ds)%s\n" \
          "$rung_idx" "$rung_count" "$((target_tokens / 1000))" "$rung_timeout" "$vram_str"
        rm -f "$result_file"
        any_fail=1
        if [[ "$first_fail_tokens" -eq 0 ]]; then
          first_fail_tokens="$target_tokens"
        fi
        # Engine crashed — no point continuing
        break
        ;;
      *)
        printf "    \033[31m✗\033[0m rung %d/%d: target=%dK  unexpected HTTP %s%s\n" \
          "$rung_idx" "$rung_count" "$((target_tokens / 1000))" "$http_code" "$vram_str"
        rm -f "$result_file"
        any_fail=1
        if [[ "$first_fail_tokens" -eq 0 ]]; then
          first_fail_tokens="$target_tokens"
        fi
        break
        ;;
    esac
  done

  # Step 4: summary
  vram_after_all="$(get_vram_free_mb)"
  echo ""
  if [[ "$any_fail" == "0" && "$any_pass" == "1" ]]; then
    if [[ "$any_skipped" == "1" ]]; then
      pass "ceiling ladder: all rungs recalled (engine rejected above ${last_pass_tokens} tok — ${last_pass_pct}% of n_ctx=${n_ctx})"
    elif [[ "$any_recall_miss" == "1" ]]; then
      pass "ceiling ladder: quality ceiling at ${first_recall_miss_tokens} tok (${first_recall_miss_pct:-?}% of n_ctx=${n_ctx}) — recall miss, passed up to ${last_pass_tokens} tok"
    else
      pass "ceiling ladder: all ${rung_count} rungs passed — fillable to ${last_pass_tokens} tok (${last_pass_pct}% of n_ctx=${n_ctx})"
    fi
    # VRAM margin check on the deepest successful rung
    if [[ "$vram_after_all" -gt 0 && "$vram_after_all" -lt "$vram_margin_mb" ]]; then
      echo "    \033[33m⚠\033[0m VRAM margin thin at ceiling: ${vram_after_all} MB free < ${vram_margin_mb} MB threshold"
      echo "      Recall succeeded at ${last_pass_pct}% fill, but sustained agent load also carries"
      echo "      prompt-cache + context-checkpoint overhead (~292 MiB, see #197)."
      echo "      Agent users should target a CTX_SIZE where margin ≥ ${vram_margin_mb} MB at this depth."
      FAILED=$((FAILED + 1))
    elif [[ "$vram_before_all" -gt 0 && "$vram_after_all" -gt 0 ]]; then
      local total_drop=$(( vram_before_all - vram_after_all ))
      echo "    VRAM: ${vram_before_all} → ${vram_after_all} MB (Δ -${total_drop} MB across ladder, margin threshold=${vram_margin_mb} MB)"
    fi
  elif [[ "$any_pass" == "1" && "$any_fail" == "1" ]]; then
    local recall_note=""
    if [[ "$any_recall_miss" == "1" ]]; then
      recall_note=", recall miss at ${first_recall_miss_tokens} tok"
    fi
    fail "ceiling ladder: wall at ~${first_fail_tokens} tok — filled up to ${last_pass_tokens} tok (${last_pass_pct}% of n_ctx=${n_ctx}), then OOMed${recall_note}" \
         "Real fillable ceiling is ~${last_pass_tokens} tok, not the advertised ${n_ctx}. Agent users will hit this wall (see #197). Lower CTX_SIZE or reduce KV dtype. Check: ${LOG_CMD}"
  elif [[ "$any_recall_miss" == "1" && "$any_fail" == "1" ]]; then
    fail "ceiling ladder: wall at ~${first_fail_tokens} tok — recall miss at ${first_recall_miss_tokens} tok, then OOMed" \
         "Attention quality dropped at ${first_recall_miss_tokens} tok, system filled to ~${first_fail_tokens} tok. Real usable ceiling is lower. Check: ${LOG_CMD}"
  elif [[ "$any_sizing_error" == "1" ]]; then
    fail "ceiling ladder: filler→token sizing error — rung target < n_ctx but got HTTP 400" \
         "Calibration probe measured tok/scale_unit=${tok_per_scale_unit} but rungs still overshot. " \
         "Check the calibration output above. This is a verify-stress.sh bug, not an engine issue."
  elif [[ "$any_fail" == "1" ]]; then
    fail "ceiling ladder: first rung at ${ceiling_start} tok already failed — ceiling may be below probe 7 range" \
         "Check whether the engine survived probe 7's 90K rung. If not, the ceiling is <90K. Check: ${LOG_CMD}"
  elif [[ "$any_pass" == "0" && "$any_skipped" == "1" ]]; then
    # All rungs got legitimate HTTP 400 (target > n_ctx) — ladder measured nothing.
    # This is NOT a pass. A ladder that tested nothing must warn.
    fail "ceiling ladder: no rungs tested — all targets exceeded engine limit" \
         "CEILING_START_TOKENS=${ceiling_start} is above the engine's max. Lower it or check n_ctx detection (detected n_ctx=${n_ctx})."
  else
    fail "ceiling ladder: no rung succeeded and none were skipped" \
         "Engine may have crashed early. Check: ${LOG_CMD}"
  fi
}
run_check "ceiling_ladder" check_ceiling_ladder
ensure_engine_alive "probe 8 (ceiling ladder)" || true

echo ""
if [[ "$FAILED" == "0" ]]; then
  printf "\033[32mAll stress / boundary checks passed.\033[0m KV-cache and prefill paths are sound for the deployed config.\n"
else
  printf "\033[31m%d stress check(s) failed.\033[0m See hints above.\n" "$FAILED"
fi
exit "$FAILED"
