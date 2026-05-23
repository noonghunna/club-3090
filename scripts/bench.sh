#!/usr/bin/env bash
#
# Canonical bench against the running vLLM service.
#   - Runs both the canonical narrative AND code prompts in one invocation.
#     This matches the README's narrative/code TPS pairing.
#   - 3 warmup + N measured runs per prompt (default 5 narrative + 5 code).
#   - per-run: wall time, TTFT (via streaming), completion tokens,
#     wall_TPS (= comp / wall), decode_TPS (= comp / (wall - TTFT))
#   - per-prompt summary: mean / std / CV for both TPS metrics + mean TTFT
#     + prompt-processing throughput (`PP tok/s`)
#   - shows MTP SpecDecoding metrics from docker logs at the end
#
# Why two TPS metrics:
#   - wall_TPS  = "user-perceived speed" (includes prefill cost)
#   - decode_TPS = "model decode rate" (excludes prefill)
#   For long prompts the two can differ a lot. For short prompts they
#   converge. Reporting both keeps comparisons honest across configs.
#
# Why narrative + code:
#   MTP acceptance varies wildly by prompt structure. Code (repetitive,
#   token-predictable) accepts at ~80% per position; prose (semantically
#   rich) at ~50%. Reporting only one half is misleading. README claims
#   like "66 / 85 TPS" pair them; bench should too.
#
# Prereq: stack is running and reports "Application startup complete".
#
# Env vars:
#   URL                Endpoint. Default: http://localhost:8020
#   MODEL              Served model name. Default: qwen3.6-27b-autoround
#   CONTAINER          Container for log scraping. Default: vllm-qwen36-27b
#   RUNS               Measured runs per prompt. Default: 5
#   WARMUPS            Warm-up runs (shared across both). Default: 3
#   PROMPT_NARR        Override narrative prompt
#   PROMPT_CODE        Override code prompt
#   MAX_TOKENS_NARR    Default: 1000
#   MAX_TOKENS_CODE    Default: 800
#   ONLY               Set to "narr" or "code" to skip the other. Default: both
#   QUIET              Set to 1 to skip per-run lines (just print summary)
#   PP                 Set to 1 to add the long-prompt PP fallback probe.
#                      llama.cpp containers enable this automatically.
#   PP_FALLBACK_TOKENS Approximate filler-token target for PP=1. Default: 10000
#   PP_MAX_TOKENS      Completion cap for the PP fallback request. Default: 16
#   ENABLE_THINKING    Set to 1 to send chat_template_kwargs.enable_thinking=true
#                      in bench requests. Default: 0.
#
# Usage:
#   bash scripts/bench.sh
#   ONLY=code bash scripts/bench.sh
#   PP=1 bash scripts/bench.sh
#   RUNS=10 bash scripts/bench.sh

set -euo pipefail

# Auto-detect running container + port (URL/CONTAINER env vars still win).
# See scripts/preflight.sh::preflight_autodetect_endpoint.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
  # shellcheck source=preflight.sh
  source "${ROOT_DIR}/scripts/preflight.sh"
  preflight_autodetect_endpoint || true
fi
URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-qwen3.6-27b-autoround}"
CONTAINER="${CONTAINER:-vllm-qwen36-27b}"
RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-3}"
MAX_TOKENS_NARR="${MAX_TOKENS_NARR:-1000}"
MAX_TOKENS_CODE="${MAX_TOKENS_CODE:-800}"
PROMPT_NARR="${PROMPT_NARR:-Write a detailed 800-word essay explaining transformer attention.}"
PROMPT_CODE="${PROMPT_CODE:-Write a Python implementation of quicksort with comments explaining each step.}"
ONLY="${ONLY:-both}"
QUIET="${QUIET:-0}"
PP="${PP:-0}"
PP_FALLBACK_TOKENS="${PP_FALLBACK_TOKENS:-10000}"
PP_MAX_TOKENS="${PP_MAX_TOKENS:-16}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }
}
need curl
need python3

ENGINE_KIND="${ENGINE_KIND:-unknown}"
if [[ "$ENGINE_KIND" == "unknown" && "${CONTAINER:-}" != "none" ]] && command -v docker >/dev/null 2>&1 && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  container_image="$(docker inspect --format '{{.Config.Image}}' "${CONTAINER}" 2>/dev/null || true)"
  container_name="$(docker inspect --format '{{.Name}}' "${CONTAINER}" 2>/dev/null || true)"
  if [[ "${container_image} ${container_name}" == *"llama.cpp"* || "${container_image} ${container_name}" == *"llama-cpp"* ]]; then
    ENGINE_KIND="llamacpp"
  elif [[ "${container_image} ${container_name}" == *"vllm"* ]]; then
    ENGINE_KIND="vllm"
  fi
fi

PP_MODE="log"
if [[ "$PP" == "1" || "$ENGINE_KIND" == "llamacpp" ]]; then
  PP_MODE="fallback"
fi

if [[ "$ENABLE_THINKING" == "1" ]]; then
  echo "[bench] thinking: enabled (request chat_template_kwargs.enable_thinking=true)" >&2
fi

if [[ "${BENCH_MOCK:-0}" == "1" ]]; then
  if [[ "$PP_MODE" == "fallback" ]]; then
    cat <<'EOF'

========== PROMPT-PROCESSING (fallback target=10000 prompt tokens, max_tokens=16) ==========
=== measured (1) ===
  run-1      wall=  3.20s  ttft= 2500ms  prompt_toks=  9876  PP_tok/s=3950.40

=== summary [prompt-processing] (n=1) ===
  PP tok/s       mean=3950.40   std=  0.00   CV= 0.0%   min=3950.40   max=3950.40
  TTFT          mean=  2500ms  std=    0ms  min=2500ms  max=2500ms
EOF
  else
    cat <<'EOF'

========== NARRATIVE (prompt=61 chars, max_tokens=1000) ==========
=== measured (1) ===
  run-1      wall=  4.20s  ttft=   120ms  toks=1000  wall_TPS=238.10  decode_TPS=245.10

=== summary [narrative] (n=1) ===
  wall_TPS       mean= 238.10   std=  0.00   CV= 0.0%   min=238.10   max=238.10
  decode_TPS     mean= 245.10   std=  0.00   CV= 0.0%   min=245.10   max=245.10
  TTFT          mean=   120ms  std=    0ms  min=120ms  max=120ms
  PP tok/s       mean=2843.21   std=  0.00   CV= 0.0%   min=2843.21   max=2843.21
EOF
  fi
  exit 0
fi

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  echo "  Start with: cd compose && docker compose up -d" >&2
  exit 1
fi

server_reasoning_on() {
  if curl -sf -m 3 "${URL}/props" 2>/dev/null | python3 -c '
import json, sys
try:
    obj = json.load(sys.stdin)
except Exception:
    sys.exit(1)

def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            lk = str(k).lower()
            if lk in {"reasoning", "enable_reasoning"}:
                if v is True or str(v).lower() in {"1", "true", "on", "yes"}:
                    return True
            if walk(v):
                return True
    elif isinstance(x, list):
        return any(walk(v) for v in x)
    return False
sys.exit(0 if walk(obj) else 1)
' >/dev/null 2>&1; then
    return 0
  fi
  if [[ -n "${CONTAINER:-}" && "${CONTAINER:-}" != "none" ]] \
     && command -v docker >/dev/null 2>&1 \
     && docker inspect "$CONTAINER" >/dev/null 2>&1; then
    docker inspect "$CONTAINER" 2>/dev/null \
      | grep -Eq -- '(--reasoning[= ]+on|"--reasoning"[[:space:]]*,[[:space:]]*"on")' && return 0
  fi
  return 1
}

if [[ "$ENABLE_THINKING" != "1" ]] && server_reasoning_on; then
  echo "[bench] WARN: server appears to have reasoning enabled, but bench requests send enable_thinking=false. Use ENABLE_THINKING=1 for reasoning-on TPS." >&2
fi

python3 - "$URL" "$MODEL" "$WARMUPS" "$RUNS" "$QUIET" "$ONLY" \
            "$CONTAINER" "$PP_MODE" "$PP_FALLBACK_TOKENS" "$PP_MAX_TOKENS" \
            "$ENABLE_THINKING" \
            "$PROMPT_NARR" "$MAX_TOKENS_NARR" \
            "$PROMPT_CODE" "$MAX_TOKENS_CODE" << 'PYEOF'
import json, re, shutil, subprocess, sys, time, urllib.request, statistics as s

(URL, MODEL, WARMUPS, RUNS, QUIET, ONLY,
 CONTAINER, PP_MODE, PP_FALLBACK_TOKENS, PP_MAX_TOKENS,
 ENABLE_THINKING, PROMPT_NARR, MAX_NARR, PROMPT_CODE, MAX_CODE) = sys.argv[1:]
WARMUPS = int(WARMUPS); RUNS = int(RUNS); QUIET = int(QUIET) == 1
MAX_NARR = int(MAX_NARR); MAX_CODE = int(MAX_CODE)
PP_FALLBACK_TOKENS = int(PP_FALLBACK_TOKENS); PP_MAX_TOKENS = int(PP_MAX_TOKENS)
ENABLE_THINKING = ENABLE_THINKING == "1"

def run_once(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", errors="ignore").rstrip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content and ttft is None:
                    ttft = time.time() - t_send
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    t_end = time.time()
    wall = t_end - t_send
    if ttft is None:
        ttft = wall
    if not prompt_tokens:
        prompt_tokens = max(1, len(prompt.split()))
    return wall, ttft, completion_tokens, prompt_tokens

def fmt(label, wall, ttft, toks):
    decode_t = max(wall - ttft, 1e-6)
    wtps = toks / wall if wall > 0 else 0
    dtps = toks / decode_t
    line = f"  {label:<10s} wall={wall:6.2f}s  ttft={ttft*1000:6.0f}ms  toks={toks:>4d}  wall_TPS={wtps:6.2f}  decode_TPS={dtps:6.2f}"
    return wtps, dtps, ttft, line

def fmt_pp(label, wall, ttft, prompt_tokens):
    pp = prompt_tokens / max(ttft, 1e-6)
    line = f"  {label:<10s} wall={wall:6.2f}s  ttft={ttft*1000:6.0f}ms  prompt_toks={prompt_tokens:>6d}  PP_tok/s={pp:7.2f}"
    return pp, ttft, line

def stats(name, xs, unit=""):
    m = s.mean(xs)
    sd = s.stdev(xs) if len(xs) > 1 else 0
    cv = (sd / m * 100) if m > 0 else 0
    return f"  {name:<14s} mean={m:7.2f}{unit}   std={sd:6.2f}   CV={cv:4.1f}%   min={min(xs):.2f}   max={max(xs):.2f}"

def scrape_prompt_throughput(container, n):
    if not container or container == "none" or shutil.which("docker") is None:
        return []
    try:
        proc = subprocess.run(
            ["docker", "logs", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception:
        return []
    vals = [
        float(m.group(1))
        for m in re.finditer(r"Avg prompt throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s", proc.stdout)
    ]
    return vals[-max(n, 1):]

def run_set(label, prompt, max_tokens):
    print(f"\n========== {label.upper()} (prompt={len(prompt)} chars, max_tokens={max_tokens}) ==========")
    print(f"=== warmups ({WARMUPS}) ===")
    for i in range(WARMUPS):
        try:
            w, t, k, _ = run_once(prompt, max_tokens)
            _, _, _, line = fmt(f"warm-{i+1}", w, t, k)
            if not QUIET:
                print(line)
        except Exception as e:
            print(f"  warm-{i+1}  FAIL: {e}")
    print(f"\n=== measured ({RUNS}) ===")
    walls, decodes, ttfts = [], [], []
    for i in range(RUNS):
        try:
            w, t, k, _ = run_once(prompt, max_tokens)
            wtps, dtps, ttft, line = fmt(f"run-{i+1}", w, t, k)
            if not QUIET:
                print(line)
            walls.append(wtps); decodes.append(dtps); ttfts.append(ttft)
        except Exception as e:
            print(f"  run-{i+1}  FAIL: {e}")
    if walls:
        print(f"\n=== summary [{label}] (n={len(walls)}) ===")
        print(stats("wall_TPS",   walls))
        print(stats("decode_TPS", decodes))
        print(f"  TTFT          mean={s.mean(ttfts)*1000:6.0f}ms  std={s.stdev(ttfts)*1000 if len(ttfts) > 1 else 0:5.0f}ms  min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms")
        if PP_MODE == "log":
            pp_vals = scrape_prompt_throughput(CONTAINER, len(walls))
            if pp_vals:
                print(stats("PP tok/s", pp_vals))
            else:
                print("  PP tok/s       n/a (vLLM log scrape unavailable; use PP=1 for long-prompt fallback)")
        else:
            print("  PP tok/s       n/a (long-prompt fallback below)")

def long_prompt(target_tokens):
    filler = (
        "club3090 prompt processing calibration filler with stable token shape. "
        "This sentence is intentionally plain so tokenizer variance stays modest. "
    )
    words_per_chunk = max(len(filler.split()), 1)
    chunks = max(1, target_tokens // words_per_chunk)
    return (
        "Read the following calibration text. Reply with one concise sentence summarizing its purpose.\n\n"
        + filler * chunks
    )

def run_pp_fallback():
    prompt = long_prompt(PP_FALLBACK_TOKENS)
    print(
        f"\n========== PROMPT-PROCESSING "
        f"(fallback target={PP_FALLBACK_TOKENS} prompt tokens, max_tokens={PP_MAX_TOKENS}) =========="
    )
    print("=== measured (1) ===")
    pp_vals, ttfts = [], []
    try:
        w, t, _k, prompt_tokens = run_once(prompt, PP_MAX_TOKENS)
        pp, ttft, line = fmt_pp("run-1", w, t, prompt_tokens)
        print(line)
        pp_vals.append(pp); ttfts.append(ttft)
    except Exception as e:
        print(f"  run-1      FAIL: {e}")
    if pp_vals:
        print("\n=== summary [prompt-processing] (n=1) ===")
        print(stats("PP tok/s", pp_vals))
        print(f"  TTFT          mean={s.mean(ttfts)*1000:6.0f}ms  std=    0ms  min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms")

if ONLY in ("both", "narr"):
    run_set("narrative", PROMPT_NARR, MAX_NARR)
if ONLY in ("both", "code"):
    run_set("code", PROMPT_CODE, MAX_CODE)
if PP_MODE == "fallback":
    run_pp_fallback()
PYEOF

# GPU state
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
fi

# MTP / spec-decode stats — only when running against a Docker container we own.
# Skipped silently in endpoint-first mode (CONTAINER=none).
if [[ "${CONTAINER:-}" != "none" ]] && command -v docker >/dev/null 2>&1 \
   && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo ""
  echo "=== Last 3 SpecDecoding metrics ==="
  docker logs "${CONTAINER}" 2>&1 | grep "SpecDecoding metrics" | tail -3 || true
fi
