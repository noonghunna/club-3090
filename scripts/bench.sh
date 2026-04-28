#!/usr/bin/env bash
#
# Canonical bench against the running vLLM service.
#   - Runs both the canonical narrative AND code prompts in one invocation.
#     This matches the README's narrative/code TPS pairing.
#   - 3 warmup + N measured runs per prompt (default 5 narrative + 5 code).
#   - per-run: wall time, TTFT (via streaming), completion tokens,
#     wall_TPS (= comp / wall), decode_TPS (= comp / (wall - TTFT))
#   - per-prompt summary: mean / std / CV for both TPS metrics + mean TTFT
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
#
# Usage:
#   bash scripts/bench.sh
#   ONLY=code bash scripts/bench.sh
#   RUNS=10 bash scripts/bench.sh

set -euo pipefail

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

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }
}
need curl
need python3

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  echo "  Start with: cd compose && docker compose up -d" >&2
  exit 1
fi

python3 - "$URL" "$MODEL" "$WARMUPS" "$RUNS" "$QUIET" "$ONLY" \
            "$PROMPT_NARR" "$MAX_TOKENS_NARR" \
            "$PROMPT_CODE" "$MAX_TOKENS_CODE" << 'PYEOF'
import json, sys, time, urllib.request, statistics as s

(URL, MODEL, WARMUPS, RUNS, QUIET, ONLY,
 PROMPT_NARR, MAX_NARR, PROMPT_CODE, MAX_CODE) = sys.argv[1:]
WARMUPS = int(WARMUPS); RUNS = int(RUNS); QUIET = int(QUIET) == 1
MAX_NARR = int(MAX_NARR); MAX_CODE = int(MAX_CODE)

def run_once(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    ttft = None
    completion_tokens = 0
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
    t_end = time.time()
    wall = t_end - t_send
    if ttft is None:
        ttft = wall
    return wall, ttft, completion_tokens

def fmt(label, wall, ttft, toks):
    decode_t = max(wall - ttft, 1e-6)
    wtps = toks / wall if wall > 0 else 0
    dtps = toks / decode_t
    line = f"  {label:<10s} wall={wall:6.2f}s  ttft={ttft*1000:6.0f}ms  toks={toks:>4d}  wall_TPS={wtps:6.2f}  decode_TPS={dtps:6.2f}"
    return wtps, dtps, ttft, line

def stats(name, xs, unit=""):
    m = s.mean(xs)
    sd = s.stdev(xs) if len(xs) > 1 else 0
    cv = (sd / m * 100) if m > 0 else 0
    return f"  {name:<14s} mean={m:7.2f}{unit}   std={sd:6.2f}   CV={cv:4.1f}%   min={min(xs):.2f}   max={max(xs):.2f}"

def run_set(label, prompt, max_tokens):
    print(f"\n========== {label.upper()} (prompt={len(prompt)} chars, max_tokens={max_tokens}) ==========")
    print(f"=== warmups ({WARMUPS}) ===")
    for i in range(WARMUPS):
        try:
            w, t, k = run_once(prompt, max_tokens)
            _, _, _, line = fmt(f"warm-{i+1}", w, t, k)
            if not QUIET:
                print(line)
        except Exception as e:
            print(f"  warm-{i+1}  FAIL: {e}")
    print(f"\n=== measured ({RUNS}) ===")
    walls, decodes, ttfts = [], [], []
    for i in range(RUNS):
        try:
            w, t, k = run_once(prompt, max_tokens)
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

if ONLY in ("both", "narr"):
    run_set("narrative", PROMPT_NARR, MAX_NARR)
if ONLY in ("both", "code"):
    run_set("code", PROMPT_CODE, MAX_CODE)
PYEOF

# GPU state
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
fi

# MTP / spec-decode stats
if command -v docker >/dev/null 2>&1 && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo ""
  echo "=== Last 3 SpecDecoding metrics ==="
  docker logs "${CONTAINER}" 2>&1 | grep "SpecDecoding metrics" | tail -3 || true
fi
