#!/usr/bin/env bash
# power-cap-sweep.sh — Power-cap A/B sweep for cross-rig efficiency-knee data
#
# Why this exists:
#   3090 sweet spot is ~330W (5% TPS loss vs ~388W stock for ~15% power
#   reduction — see @syangsao's three-cap data on issue #58). For other GPU
#   classes (4090, 5090, A5000, A6000, modded variants) the knee differs and
#   has to be measured. This script automates the sweep so contributors can
#   produce comparable cross-rig numbers without hand-editing nvidia-smi
#   commands and bench invocations.
#
# Usage:
#   sudo bash scripts/power-cap-sweep.sh                          # comprehensive sweep at 10W increments (matches @laurimyllari resolution)
#   sudo bash scripts/power-cap-sweep.sh --step-size 20           # coarser sweep (~half runtime)
#   sudo bash scripts/power-cap-sweep.sh --caps 260,280,300       # explicit caps (overrides auto-derive)
#   sudo bash scripts/power-cap-sweep.sh --gpu 1                  # specific GPU index
#   sudo bash scripts/power-cap-sweep.sh --cooling water          # tag the run as water-cooled
#   sudo bash scripts/power-cap-sweep.sh --cooling air            # tag as air-cooled
#   sudo bash scripts/power-cap-sweep.sh --cooling aio            # tag as AIO/closed-loop
#   sudo bash scripts/power-cap-sweep.sh --load-mode decode-concurrent --concurrency auto
#   sudo bash scripts/power-cap-sweep.sh --load-mode decode-concurrent --concurrency 8
#   sudo bash scripts/power-cap-sweep.sh --load-mode decode-concurrent --concurrency 8 --bench-runs 3
#   sudo bash scripts/power-cap-sweep.sh --load-mode prefill-heavy
#   sudo bash scripts/power-cap-sweep.sh --no-reset               # leave at last cap (you reset manually)
#   sudo bash scripts/power-cap-sweep.sh --include-commit         # stamp club-3090 git short SHA in report header
#
# Load modes:
#   decode-single:
#     Original single-stream bench.sh path. Best for continuity with existing
#     contributor data, and enough to expose the efficiency knee on cards where
#     this workload already loads compute well (3090 / 4090).
#
#   decode-concurrent:
#     Runs N concurrent chat completions and reports aggregate decode TPS. Use
#     this for realistic multi-request serving load, especially on larger cards
#     where decode-single is under-loaded and produces flat power curves.
#     Pass --concurrency auto to calibrate the stream count before the sweep:
#     the script probes increasing concurrency at the highest requested cap and
#     selects the first N that reaches --load-target, or the best non-failing N.
#
#     Pass --concurrency-stretch N (with --concurrency auto) to add N more streams
#     to whatever plateau-detect picked. Useful when your sweep shows draw plateau
#     below cap (e.g. 547W actual at 600W cap on a 5090) — bumping concurrency past
#     the auto-recommended pick can probe whether the card has compute headroom
#     unused by the plateau-safe N. Per-stream TPS will drop; aggregate TPS may
#     dip slightly; actual draw may rise. See docs/HARDWARE.md for interpretation.
#
#     ⚠️ VARIANCE CAVEAT: decode-concurrent defaults to n=1 measured batch per
#     cap (one batch of N concurrent requests for narr, one for code). Aggregate
#     TPS can vary 10-30% between back-to-back runs at the SAME cap because
#     vLLM's continuous-batching window is timing-sensitive — whether N
#     requests batch together vs queue sequentially depends on arrival jitter.
#     Single caps may show TPS going the "wrong direction" between adjacent
#     caps. For cross-rig anchor data, prefer one of: (a) bump --concurrency to
#     8 or 16 so per-stream noise averages out; (b) pass --bench-runs 3 to
#     median repeated batches per cap; (c) read curve shape across the full
#     30-cap sweep instead of comparing adjacent caps.
#
#   prefill-heavy:
#     Sends one large prompt with a tiny decode tail and reports prompt prefill
#     TPS. This is the cleanest intrinsic compute curve when decode workloads
#     are too small to move the card. Lower variance than decode-concurrent
#     (single request per cap, no batching jitter; nonce defeats prefix-cache
#     reuse between caps).
#
# Per-card starting points:
#   3090 / 4090: decode-single or decode-concurrent both usually surface a knee.
#   5090:        decode-concurrent N=8+ recommended.
#   RTX PRO 6000: prefill-heavy or decode-concurrent N=16+, preferably with a
#                 larger model than 27B if the endpoint can schedule it.
#
# Default sweep behavior:
#   Without --caps, the script reads power.min_limit and power.max_limit and
#   generates caps at 10W increments across the entire envelope. This matches
#   @laurimyllari's reference resolution that produced the cleanest 4090 curve.
#
#   Each cap runs a reduced bench (WARMUPS=1 RUNS=2 with 500/400 max_tokens),
#   targeting ~30s/cap of sustained load — enough for the power sampler to
#   collect 50+ under-load samples for a stable median. Per-card estimates:
#
#     3090 (100-388W) →  30 caps  ~15 min
#     4090 (150-450W) →  31 caps  ~16 min
#     5090 (250-575W) →  33 caps  ~17 min
#     A5000 (100-230W) → 14 caps  ~7 min
#
#   At heavily-throttled caps (e.g. 100W on a 3090), bench runs slower and
#   the per-cap time can stretch to ~50-60s, so total runtime is ~20 min on
#   a typical sweep. For zooming into a known-good region, use --caps
#   260,280,300 explicitly. For coarser sweeps, --step-size 20.
#
# Output:
#   - Per-cap bench logs at /tmp/power-cap-N{wattage}.log
#   - Markdown summary at /tmp/power-cap-summary.md (paste into GitHub issue/discussion)
#
# Per-cap summary columns:
#   Cap | Narr TPS | Code TPS | Actual W | Temp °C | SM clk | Mem clk | Pwr-throttle % | P-state | TPS/W
#
# - SM clk / Mem clk = median compute / memory clocks during in-load samples (MHz). Together
#   they distinguish compute-bound vs bandwidth-bound regimes: if SM clock varies with cap
#   while TPS plateaus → bandwidth-bound; if SM clock is pinned at max while TPS still climbs
#   → compute-bound.
# - Pwr-throttle % = % of in-load samples where firmware was actively capping draw at the
#   set limit (sw_power_cap=Active). 100% means power is the binding constraint at that
#   cap; <100% means workload undersupply or thermal-throttle taking over.
# - P-state = dominant firmware power state during in-load samples (P0=max boost,
#   P2=sustained-load pinned, higher numbers=idle). Boost-state plateaus appear when
#   adjacent caps share a P-state and draw the same wattage despite different cap settings.
#
# Plateau auto-detection:
#   At end of sweep, the script scans for 3+ consecutive caps with identical draw
#   (within ±2W) and TPS (within ±1%) — a firmware boost-clock plateau where caps
#   in the run are functionally equivalent. Detected plateaus are logged via
#   [plateau detected] lines and added as a "Detected boost-clock plateau(s)"
#   section in the summary file. Pick the LOWEST cap in a plateau to save power
#   for free; raising past the plateau end-cap is the only way to escape.
#
# Recommended sweep chain (full workload-class characterization on your rig):
#   Run two modes — decode and prefill sweet spots can differ. ~14 min total.
#     1. sudo bash scripts/power-cap-sweep.sh --cooling <class> --load-mode decode-single
#     2. sudo bash scripts/power-cap-sweep.sh --cooling <class> --load-mode prefill-heavy
#   For multi-tenant rigs, also:
#     3. sudo bash scripts/power-cap-sweep.sh --cooling <class> --load-mode decode-concurrent --concurrency auto
#   See docs/HARDWARE.md > Power > "Recommended sweep chain" for the full rationale.
#
# Requires sudo for `nvidia-smi -pl`. Auto-detects running container + URL +
# MODEL via the same logic as bench.sh.
#
# Why --cooling matters:
#   Air-cooled cards thermal-throttle around 80-83°C, capping effective
#   sustained power at ~310-340W on a 3090 regardless of the software cap.
#   Water-cooled / AIO cards hold lower temps (~50-65°C) and sustain full
#   board power. Same software cap on different cooling produces different
#   real curves — recording the cooling class is essential for cross-rig
#   comparison. The script does NOT auto-detect this; you must specify.

set -euo pipefail

# Defaults — override via flags
GPU_INDEX=0
GPU_INDEX_SET=0      # set to 1 once --gpu is explicitly passed (distinguishes from the default 0)
GPUS=""              # explicit --gpus a,b list; empty → auto-detect the workload's GPUs
CAPS=""              # empty → auto-derive from card's min/max power limits at STEP_SIZE granularity
RESET=1              # 1 = reset to stock at end; 0 = leave at last cap
COOLING="unspecified" # air|water|aio|unspecified — affects how to read the data
STEP_SIZE=10          # increment in W between caps when --caps not specified (10W matches @laurimyllari's resolution)
LOAD_MODE="decode-single"   # decode-single | decode-concurrent | prefill-heavy
CONCURRENCY=4         # parallel streams, or "auto", when LOAD_MODE=decode-concurrent
BENCH_RUNS=1          # repeated measured batches for decode-concurrent/prefill-heavy (median reported);
                      # default stays one batch for sweep shape. Use --bench-runs 3 only for anchor data.
MAX_CONCURRENCY_PROBE=16
LOAD_TARGET=0.92      # target actual-power/cap ratio for --concurrency auto
CONCURRENCY_STRETCH=0 # add N to auto-detected concurrency (probe headroom past plateau pick)
TARGET_CAP_SECONDS=10 # decode-single time-bounded streaming bench seconds per direction
                      # (narrative + code). This keeps per-cap wall stable
                      # across card classes while giving the sampler >=10s
                      # of util>50% data per cap.
TARGET_PREFILL_SECONDS="${TARGET_PREFILL_SECONDS:-10}" # prefill-heavy prompt is calibrated at highest cap
PREFILL_CALIBRATION_REPEATS="${PREFILL_CALIBRATION_REPEATS:-1000}"
PREFILL_FILLER_REPEATS=""
PREFILL_PROMPT_TOKENS=""
DECODE_CONCURRENT_RUN_SECONDS=""
CALIBRATION_NOTE=""
INCLUDE_COMMIT=0      # --include-commit stamps the club-3090 git short SHA in
                      # the report header. Off by default — `curl ... | bash`
                      # users have no clone, and stamping "n/a" is confusing
                      # (better to suppress the field entirely there).

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)         GPU_INDEX="$2"; GPU_INDEX_SET=1; shift 2 ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --caps)        CAPS="$2"; shift 2 ;;
    --cooling)     COOLING="$2"; shift 2 ;;
    --step-size)   STEP_SIZE="$2"; shift 2 ;;
    --load-mode)   LOAD_MODE="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --bench-runs)  BENCH_RUNS="$2"; shift 2 ;;
    --max-concurrency-probe) MAX_CONCURRENCY_PROBE="$2"; shift 2 ;;
    --load-target) LOAD_TARGET="$2"; shift 2 ;;
    --concurrency-stretch) CONCURRENCY_STRETCH="$2"; shift 2 ;;
    --target-cap-seconds) TARGET_CAP_SECONDS="$2"; shift 2 ;;
    --include-commit) INCLUDE_COMMIT=1; shift ;;
    --no-reset)    RESET=0; shift ;;
    -h|--help)
      sed -n '1,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *)             echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Validate --load-mode value
case "$LOAD_MODE" in
  decode-single|decode-concurrent|prefill-heavy) ;;
  *) echo "[error] --load-mode must be one of: decode-single, decode-concurrent, prefill-heavy" >&2; exit 1 ;;
esac
CONCURRENCY_AUTO=0
if [ "$CONCURRENCY" = "auto" ]; then
  CONCURRENCY_AUTO=1
elif ! [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --concurrency must be a positive integer or 'auto'" >&2
  exit 1
fi
if ! [[ "$BENCH_RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --bench-runs must be a positive integer" >&2
  exit 1
fi
if ! [[ "$MAX_CONCURRENCY_PROBE" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-concurrency-probe must be a positive integer" >&2
  exit 1
fi
if [ "$CONCURRENCY_AUTO" -eq 1 ] && [ "$MAX_CONCURRENCY_PROBE" -lt 4 ]; then
  echo "[error] --max-concurrency-probe must be at least 4 when --concurrency auto is used" >&2
  exit 1
fi
if ! [[ "$CONCURRENCY_STRETCH" =~ ^[0-9]+$ ]]; then
  echo "[error] --concurrency-stretch must be a non-negative integer" >&2
  exit 1
fi
if [ "$CONCURRENCY_STRETCH" -gt 0 ] && [ "$CONCURRENCY_AUTO" -ne 1 ]; then
  echo "[error] --concurrency-stretch only applies with --concurrency auto" >&2
  exit 1
fi
if ! [[ "$TARGET_CAP_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --target-cap-seconds must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TARGET_PREFILL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] TARGET_PREFILL_SECONDS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$PREFILL_CALIBRATION_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] PREFILL_CALIBRATION_REPEATS must be a positive integer" >&2
  exit 1
fi
if ! python3 - "$LOAD_TARGET" <<'PY' >/dev/null 2>&1
import sys
x = float(sys.argv[1])
raise SystemExit(0 if 0 < x <= 1 else 1)
PY
then
  echo "[error] --load-target must be a float in (0, 1]" >&2
  exit 1
fi

# Validate --cooling value
case "$COOLING" in
  air|water|aio|unspecified) ;;
  *) echo "[error] --cooling must be one of: air, water, aio (or omit for 'unspecified')" >&2; exit 1 ;;
esac

if [ "$COOLING" = "unspecified" ]; then
  echo "[warn] --cooling not specified. Cooling class is essential context for interpreting"
  echo "[warn] the efficiency knee (air-cooled cards thermal-throttle, water-cooled don't)."
  echo "[warn] Consider re-running with: --cooling air|water|aio"
  echo
fi

# --- Multi-GPU helpers -------------------------------------------------------
# A TP=N serving workload spans several cards, so the sweep operates on a SET of
# GPUs, not a single index. These helpers resolve that set, capture each card's
# envelope, aggregate per-tick power across cards, and restore caps on exit.

gpu_indices_from_container() {
  # Echo the comma-separated GPU indices a container is pinned to, or "" if it
  # uses all GPUs / can't be determined (caller then falls back to all GPUs).
  local c="$1" nvd
  nvd="$(docker inspect "$c" 2>/dev/null \
    | grep -o 'NVIDIA_VISIBLE_DEVICES=[^"]*' | head -1 | cut -d= -f2)"
  case "${nvd:-}" in
    ""|all|void|none) echo "" ;;
    *) echo "$nvd" | tr -d ' ' ;;
  esac
}

resolve_gpu_indices() {
  # Echo the comma-separated set of GPUs to sweep. Precedence:
  #   --gpus a,b  >  --gpu N  >  container-detected  >  all GPUs (warn).
  local csv=""
  if [ -n "${GPUS:-}" ]; then
    csv="$GPUS"
  elif [ "${GPU_INDEX_SET:-0}" -eq 1 ]; then
    csv="$GPU_INDEX"
  elif [ -n "${CONTAINER:-}" ] && [ "${CONTAINER}" != "none" ]; then
    csv="$(gpu_indices_from_container "$CONTAINER")"
  fi
  if [ -z "$csv" ]; then
    csv="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)"
    [ -n "$csv" ] && echo "[warn] could not scope the workload's GPUs — sweeping ALL GPUs (${csv}). Pass --gpus a,b to scope." >&2
  fi
  echo "$csv" | tr -d ' '
}

aggregate_gpu_sample() {
  # Collapse N per-GPU nvidia-smi CSV lines (one sampler tick) into ONE synthetic
  # line: power.draw SUMMED, utilization/temp MAX, throttle reasons OR'd,
  # clocks/pstate from the first card. Keeps the same 9-field schema the
  # downstream median/throttle post-processing expects — so multi-GPU needs no
  # change there. For a single GPU it is a pass-through (sum == that GPU).
  awk -F',' '
    { for (i = 1; i <= NF; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i)
      sumP += $3 + 0
      if (($2 + 0) > maxU) maxU = $2 + 0
      if (($4 + 0) > maxT) maxT = $4 + 0
      if (NR == 1) { sm = $5; mem = $6; ps = $7 }
      if ($8 == "Active") pwr = "Active"
      if ($9 == "Active") therm = "Active"
      n++ }
    END { if (n > 0) printf "agg, %d, %.2f, %d, %s, %s, %s, %s, %s\n",
            maxU, sumP, maxT, sm, mem, ps,
            (pwr == "Active" ? "Active" : "Not Active"),
            (therm == "Active" ? "Active" : "Not Active") }'
}

capture_envelopes() {
  # Populate INIT_ARR[idx] (current power.limit — for --no-reset restore) and
  # STOCK_ARR[idx] (factory default_limit — for the default reset) for every GPU
  # in GPU_INDICES, and derive the SYMMETRIC sweep range as the intersection of
  # all cards' envelopes: floor = max of per-card mins, ceiling = min of per-card
  # maxes, so a single cap value is valid on every card. STOCK_TDP (used only for
  # the 50%-of-stock floor) takes the lowest card's default.
  local idx mn mx df lim
  MIN_LIMIT=""; MAX_LIMIT=""; STOCK_TDP=""
  for idx in "${GPU_INDICES[@]}"; do
    df=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits -i "$idx" | head -1 | tr -d ' ')
    mn=$(nvidia-smi --query-gpu=power.min_limit     --format=csv,noheader,nounits -i "$idx" | head -1 | tr -d ' ')
    mx=$(nvidia-smi --query-gpu=power.max_limit     --format=csv,noheader,nounits -i "$idx" | head -1 | tr -d ' ')
    lim=$(nvidia-smi --query-gpu=power.limit        --format=csv,noheader,nounits -i "$idx" | head -1 | tr -d ' ')
    INIT_ARR[$idx]="$lim"
    STOCK_ARR[$idx]="$df"
    if [ -z "$MIN_LIMIT" ] || awk "BEGIN{exit !($mn > $MIN_LIMIT)}"; then MIN_LIMIT="$mn"; fi
    if [ -z "$MAX_LIMIT" ] || awk "BEGIN{exit !($mx < $MAX_LIMIT)}"; then MAX_LIMIT="$mx"; fi
    if [ -z "$STOCK_TDP" ] || awk "BEGIN{exit !($df < $STOCK_TDP)}"; then STOCK_TDP="$df"; fi
  done
}

restore_gpus() {
  # On end/exit: RESET=1 (default) → set each GPU to its factory stock; --no-reset
  # (RESET=0) → restore each GPU to the limit it had when the sweep started.
  local idx target
  for idx in "${GPU_INDICES[@]}"; do
    if [ "${RESET:-1}" -eq 1 ]; then target="${STOCK_ARR[$idx]:-}"; else target="${INIT_ARR[$idx]:-}"; fi
    [ -n "$target" ] && nvidia-smi -pl "$target" -i "$idx" >/dev/null 2>&1 || true
  done
}

# Sanity checks
if [ "$EUID" -ne 0 ]; then
  echo "[error] must run as root (nvidia-smi -pl requires sudo)" >&2
  echo "[hint]  rerun with: sudo bash scripts/power-cap-sweep.sh ..." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi not found in PATH" >&2; exit 1
fi

# Determine paths — script may be invoked from anywhere
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$REPO_ROOT/scripts/bench.sh"
if [ ! -x "$BENCH" ]; then
  echo "[error] expected $BENCH" >&2; exit 1
fi

# Auto-detect URL/CONTAINER/MODEL from the running engine.
# This must happen BEFORE we exec bench.sh under our sudo context — bench.sh's
# own autodetect doesn't reliably fire when re-invoked under sudo (env vars
# get stripped, defaults kick in, wrong MODEL → HTTP 404 against the server).
if [ -z "${CONTAINER:-}" ] || [ -z "${URL:-}" ]; then
  if [[ -f "$REPO_ROOT/scripts/preflight.sh" ]]; then
    # shellcheck source=preflight.sh
    source "$REPO_ROOT/scripts/preflight.sh"
    preflight_autodetect_endpoint || true
  fi
fi

# preflight_autodetect_endpoint only sets URL + CONTAINER, not MODEL.
# Query the live /v1/models endpoint to derive the served model name.
if [ -z "${MODEL:-}" ] && [ -n "${URL:-}" ]; then
  MODEL=$(curl -sf --max-time 5 "${URL}/v1/models" 2>/dev/null \
    | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
fi

# CONTAINER is OPTIONAL — host engine builds (e.g. llama.cpp host server, see
# club-3090#85, #87) have no container. URL + MODEL are the only hard
# requirements. If CONTAINER is unset we mark it "none" for display, which is
# also the value bench.sh expects to skip its docker-log scrape cleanly.
if [ -z "${CONTAINER:-}" ]; then
  CONTAINER="none"
fi

if [ -z "${URL:-}" ] || [ -z "${MODEL:-}" ]; then
  echo "[error] could not auto-detect a running URL + MODEL." >&2
  echo "[hint]  start a model server first (bash scripts/switch.sh <variant>)" >&2
  echo "[hint]  or pass URL=http://... MODEL=name as env vars" >&2
  echo "[hint]  CONTAINER is optional — set CONTAINER=none for host builds" >&2
  echo "[got]   URL='${URL:-}' CONTAINER='${CONTAINER:-}' MODEL='${MODEL:-}'" >&2
  exit 1
fi
export URL CONTAINER MODEL
echo "[setup] target:   container=$CONTAINER url=$URL model=$MODEL"

# Resolve the set of GPUs to sweep (a TP=N workload spans several cards).
GPU_LIST_CSV="$(resolve_gpu_indices)"
IFS=',' read -ra GPU_INDICES <<< "$GPU_LIST_CSV"
if [ "${#GPU_INDICES[@]}" -eq 0 ] || [ -z "${GPU_INDICES[0]:-}" ]; then
  echo "[error] could not determine which GPU(s) to sweep — pass --gpus 0,1 (or --gpu N)." >&2
  exit 1
fi
PRIMARY_GPU="${GPU_INDICES[0]}"
declare -A INIT_ARR=() STOCK_ARR=()
echo "[setup] GPUs:     ${GPU_LIST_CSV}"

# Capture each card's power envelope: per-GPU INIT/STOCK arrays + the intersected
# symmetric sweep range (MIN_LIMIT/MAX_LIMIT/STOCK_TDP). See capture_envelopes().
capture_envelopes
GPU_NAME=$(nvidia-smi --query-gpu=name         --format=csv,noheader        -i "$PRIMARY_GPU" | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$PRIMARY_GPU" | head -1 | tr -d ' ')

SAMPLER_PID=""
cleanup() {
  # Avoid trap recursion if cleanup itself is interrupted
  trap '' INT TERM EXIT

  if [ -n "${SAMPLER_PID:-}" ]; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=""
  fi

  # Kill any bench.sh / tee / curl children that might still be running.
  # Without this, Ctrl+C or SIGTERM from a parent shell leaves orphaned
  # subprocess writing to log files + holding the GPU cap setting.
  # (SIGKILL bypasses this trap entirely — that's a kernel-level guarantee.)
  pkill -TERM -P $$ 2>/dev/null || true

  if [ -n "${GPU_LIST_CSV:-}" ]; then
    restore_gpus
  fi
}
trap cleanup EXIT INT TERM

bench_decode_single_for_seconds() {
  local kind="$1"
  local seconds="$2"
  local cap="$3"
  local log_file="$4"
  local req_file out_file start_ns end_ns wall_s tokens tps prompt max_time

  req_file="/tmp/power-cap-N${cap}-${kind}.req.json"
  out_file="/tmp/power-cap-N${cap}-${kind}.sse"
  max_time="$seconds"

  case "$kind" in
    narrative) prompt="Write a detailed 800-word essay explaining transformer attention." ;;
    code)      prompt="Implement quicksort in Python with detailed comments." ;;
    *) echo "[error] unknown decode-single prompt kind: $kind" >&2; return 1 ;;
  esac

  python3 - "$req_file" "$MODEL" "$prompt" <<'PY'
import json
import sys

path, model, prompt = sys.argv[1:4]
body = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 99999,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "stream": True,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(body, f)
PY

  start_ns=$(date +%s%N)
  # curl exits 28 when --max-time cuts the stream. That is expected here: the
  # wall clock is the benchmark boundary, not a completed max_tokens response.
  curl -sS --no-buffer --max-time "$max_time" "${URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "@${req_file}" \
    -o "$out_file" 2>>"$log_file" || true
  end_ns=$(date +%s%N)

  wall_s=$(python3 - "$start_ns" "$end_ns" <<'PY'
import sys
start, end = map(int, sys.argv[1:3])
print(f"{(end - start) / 1e9:.3f}")
PY
)
  tokens=$(python3 - "$out_file" <<'PY'
import json
import sys

path = sys.argv[1]
chunks = 0
usage_tokens = None
chars = 0

try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            usage = obj.get("usage")
            if isinstance(usage, dict):
                completion = usage.get("completion_tokens")
                if isinstance(completion, int) and completion > 0:
                    usage_tokens = completion
            for choice in obj.get("choices", []):
                text = ""
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    # Sum content + reasoning_content + reasoning fields. Servers route
                    # thinking tokens to different field paths depending on config:
                    #   delta.content           — standard OpenAI chat-completions path
                    #   delta.reasoning_content — vLLM with --reasoning-parser qwen3 (most common)
                    #   delta.reasoning         — some vLLM versions / DeepSeek convention
                    # Counting only content silently produces 0 TPS readings even though
                    # the GPU is generating fine. See:
                    #   - disc club-3090#62 (laurimyllari 4090 sweep, 2026-05-08) — added reasoning_content
                    #   - issue club-3090#104 (alexpolo1 dual 3090 sweep, 2026-05-08) — added reasoning
                    text = (
                        (delta.get("content") or "")
                        + (delta.get("reasoning_content") or "")
                        + (delta.get("reasoning") or "")
                    )
                if not text:
                    text = choice.get("text") or ""
                if text:
                    chunks += 1
                    chars += len(text)
except FileNotFoundError:
    pass

if usage_tokens:
    print(usage_tokens)
elif chunks:
    print(chunks)
elif chars:
    print(max(1, round(chars / 4)))
else:
    print(0)
PY
)
  tps=$(python3 - "$tokens" "$wall_s" <<'PY'
import sys
tokens = int(sys.argv[1])
wall = float(sys.argv[2])
print(f"{tokens / max(wall, 0.001):.2f}")
PY
)

  echo "[$kind] ${tokens} streamed token-chunks in ${wall_s}s -> ${tps} TPS" | tee -a "$log_file"
  printf "%s\n" "$tps"
}

bench_decode_concurrent_for_seconds() {
  local kind="$1"
  local seconds="$2"
  local cap="$3"
  local run_idx="$4"
  local log_file="$5"
  local req_file out_file start_ns end_ns wall_s total_tokens tps prompt max_time
  local pids=()

  max_time="$seconds"
  case "$kind" in
    narrative) prompt="Write a detailed 800-word essay explaining transformer attention." ;;
    code)      prompt="Implement quicksort in Python with detailed comments." ;;
    *) echo "[error] unknown decode-concurrent prompt kind: $kind" >&2; return 1 ;;
  esac

  start_ns=$(date +%s%N)
  for i in $(seq 1 "$CONCURRENCY"); do
    req_file="/tmp/power-cap-N${cap}-r${run_idx}-${kind}-${i}.req.json"
    out_file="/tmp/power-cap-N${cap}-r${run_idx}-${kind}-${i}.sse"
    python3 - "$req_file" "$MODEL" "$prompt" "$run_idx" "$i" <<'PY'
import json
import sys
import time

path, model, prompt, run_idx, stream_idx = sys.argv[1:6]
nonce = f"power-cap concurrent timed nonce {time.time_ns()} run={run_idx} stream={stream_idx}. "
body = {
    "model": model,
    "messages": [{"role": "user", "content": nonce + prompt}],
    "max_tokens": 99999,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "stream": True,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(body, f)
PY
    curl -sS --no-buffer --max-time "$max_time" "${URL}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "@${req_file}" \
      -o "$out_file" 2>>"$log_file" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
  end_ns=$(date +%s%N)

  wall_s=$(python3 - "$start_ns" "$end_ns" <<'PY'
import sys
start, end = map(int, sys.argv[1:3])
print(f"{(end - start) / 1e9:.3f}")
PY
)
  total_tokens=0
  for i in $(seq 1 "$CONCURRENCY"); do
    out_file="/tmp/power-cap-N${cap}-r${run_idx}-${kind}-${i}.sse"
    local stream_tokens
    stream_tokens=$(python3 - "$out_file" <<'PY'
import json
import sys

path = sys.argv[1]
chunks = 0
usage_tokens = None
chars = 0
try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            usage = obj.get("usage")
            if isinstance(usage, dict):
                completion = usage.get("completion_tokens")
                if isinstance(completion, int) and completion > 0:
                    usage_tokens = completion
            for choice in obj.get("choices", []):
                text = ""
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    # Sum content + reasoning_content + reasoning fields. Servers route
                    # thinking tokens to different field paths depending on config:
                    #   delta.content           — standard OpenAI chat-completions path
                    #   delta.reasoning_content — vLLM with --reasoning-parser qwen3 (most common)
                    #   delta.reasoning         — some vLLM versions / DeepSeek convention
                    # Counting only content silently produces 0 TPS readings even though
                    # the GPU is generating fine. See:
                    #   - disc club-3090#62 (laurimyllari 4090 sweep, 2026-05-08) — added reasoning_content
                    #   - issue club-3090#104 (alexpolo1 dual 3090 sweep, 2026-05-08) — added reasoning
                    text = (
                        (delta.get("content") or "")
                        + (delta.get("reasoning_content") or "")
                        + (delta.get("reasoning") or "")
                    )
                if not text:
                    text = choice.get("text") or ""
                if text:
                    chunks += 1
                    chars += len(text)
except FileNotFoundError:
    pass

if usage_tokens:
    print(usage_tokens)
elif chunks:
    print(chunks)
elif chars:
    print(max(1, round(chars / 4)))
else:
    print(0)
PY
)
    total_tokens=$((total_tokens + stream_tokens))
  done
  tps=$(python3 - "$total_tokens" "$wall_s" <<'PY'
import sys
tokens = int(sys.argv[1])
wall = float(sys.argv[2])
print(f"{tokens / max(wall, 0.001):.2f}")
PY
)

  echo "[$kind r${run_idx}] ${CONCURRENCY} streams, ${total_tokens} streamed token-chunks in ${wall_s}s -> aggregate ${tps} TPS" | tee -a "$log_file"
  printf "%s\n" "$tps"
}

bench_prefill_once() {
  local repeats="$1"
  local cap="$2"
  local run_idx="$3"
  local log_file="$4"
  local max_time="${5:-180}"
  local req_file out_file start_ns end_ns wall_s prompt_tokens tps

  req_file="/tmp/power-cap-N${cap}-r${run_idx}-prefill.req.json"
  out_file="/tmp/power-cap-N${cap}-r${run_idx}-prefill.json"
  python3 - "$req_file" "$MODEL" "$repeats" <<'PY'
import json
import sys
import time

path, model = sys.argv[1:3]
filler_repeats = int(sys.argv[3])
filler = "The quick brown fox jumps over the lazy dog. " * filler_repeats
nonce = f" Unique power-cap sweep nonce: {time.time_ns()}."
body = {
    "model": model,
    "messages": [{"role": "user", "content": nonce + " " + filler + " Summarize."}],
    "max_tokens": 10,
    "temperature": 0,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(body, f)
PY
  start_ns=$(date +%s%N)
  if ! curl -sS -f --max-time "$max_time" "${URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "@${req_file}" \
    -o "$out_file" 2>>"$log_file"; then
    echo "[warn] prefill-heavy run ${run_idx} curl failed at ${cap}W" | tee -a "$log_file" >&2
  fi
  end_ns=$(date +%s%N)
  wall_s=$(python3 - "$start_ns" "$end_ns" <<'PY'
import sys
start, end = map(int, sys.argv[1:3])
print(f"{(end - start) / 1e9:.3f}")
PY
)
  prompt_tokens=$(python3 - "$out_file" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        print(json.load(f).get("usage", {}).get("prompt_tokens", 0))
except Exception:
    print(0)
PY
)
  tps=$(python3 - "$prompt_tokens" "$wall_s" <<'PY'
import sys
tokens = int(sys.argv[1])
wall = float(sys.argv[2])
print(f"{tokens / max(wall, 0.001):.2f}")
PY
)
  echo "[prefill r${run_idx}] ${prompt_tokens} prompt tokens in ${wall_s}s -> ${tps} prefill TPS" | tee -a "$log_file" >&2
  printf "%s %s %s\n" "$tps" "$prompt_tokens" "$wall_s"
}

run_concurrency_probe() {
  local n="$1"
  local cap="$2"
  local dir="$3"
  local sample_file="$dir/samples-N${n}.csv"
  local start_ns end_ns wall_s total_tokens fails tps stats actual_power ratio

  (
    while true; do
      nvidia-smi --query-gpu=index,utilization.gpu,power.draw,temperature.gpu \
        --format=csv,noheader,nounits -i "$PRIMARY_GPU" 2>/dev/null | head -1
      sleep 0.25
    done
  ) > "$sample_file" &
  local probe_sampler_pid=$!

  local pids=()
  start_ns=$(date +%s%N)
  for i in $(seq 1 "$n"); do
    local req_file="$dir/req-N${n}-${i}.json"
    python3 - "$req_file" "$MODEL" "$n" "$i" <<'PY'
import json
import sys
import time

path, model, n, i = sys.argv[1:5]
nonce = f"power-cap auto calibration nonce {time.time_ns()} N={n} stream={i}. "
body = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": nonce + "Write a detailed 300-word essay explaining transformer attention.",
    }],
    "max_tokens": 200,
    "temperature": 0.6,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(body, f)
PY
    curl -sS -f --max-time 90 "${URL}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "@${req_file}" \
      -o "$dir/out-N${n}-${i}.json" 2>>"$dir/probe-N${n}.log" &
    pids+=("$!")
  done

  fails=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fails=$((fails + 1))
    fi
  done
  end_ns=$(date +%s%N)
  kill "$probe_sampler_pid" 2>/dev/null || true
  wait "$probe_sampler_pid" 2>/dev/null || true

  wall_s=$(python3 - "$start_ns" "$end_ns" <<'PY'
import sys
start, end = map(int, sys.argv[1:3])
print((end - start) / 1e9)
PY
)
  total_tokens=0
  for i in $(seq 1 "$n"); do
    if [ -s "$dir/out-N${n}-${i}.json" ]; then
      local t
      t=$(python3 -c "import json; print(json.load(open('$dir/out-N${n}-${i}.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
      total_tokens=$((total_tokens + t))
    fi
  done
  tps=$(python3 - "$total_tokens" "$wall_s" <<'PY'
import sys
tokens = int(sys.argv[1])
wall = float(sys.argv[2])
print(f"{tokens / max(wall, 0.001):.2f}")
PY
)
  stats=$(python3 - "$sample_file" <<'PY'
import sys
samples = []
with open(sys.argv[1]) as f:
    for line in f:
        try:
            _, util, power, _ = [x.strip() for x in line.strip().split(",")]
            if int(util) > 50:
                samples.append(float(power))
        except Exception:
            pass
if not samples:
    print("?")
else:
    samples.sort()
    print(f"{samples[len(samples)//2]:.2f}")
PY
)
  actual_power="$stats"
  ratio=$(python3 - "$actual_power" "$cap" <<'PY'
import sys
try:
    power = float(sys.argv[1])
    cap = float(sys.argv[2])
    print(f"{power / max(cap, 0.001):.3f}")
except Exception:
    print("0.000")
PY
)
  printf "%s %s %s %s %s %s\n" "$n" "$tps" "$actual_power" "$ratio" "$fails" "$wall_s"
}

# If --caps not specified, derive a sweep at STEP_SIZE-W increments across the
# card's operating range. 10W default matches @laurimyllari's reference
# resolution that produced the cleanest 4090 curve. Works on any card class:
#   3090 (100-388W) →  30 caps  (~60 min runtime at 2 min/cap)
#   4090 (150-450W) →  31 caps  (~62 min runtime)
#   5090 (250-575W) →  33 caps  (~66 min runtime)
#   A5000 (100-230W) → 14 caps  (~30 min runtime)
# For a quicker first-look use --step-size 20 (cuts runtime in half) or
# --caps 260,280,300 (zoom into a known-good region).
if [ -z "$CAPS" ]; then
  CAPS=$(python3 -c "
min_l = int(float('${MIN_LIMIT%.*}'))
max_l = int(float('${MAX_LIMIT%.*}'))
stock = int(float('${STOCK_TDP%.*}'))
step = max(1, int('${STEP_SIZE}'))
# Smart floor: max(firmware-min, 50% of stock TDP). Below 50% of stock TDP,
# the GPU is so throttled that bench takes 3-5× longer per cap and produces
# uselessly low TPS. Skip that region by default. Override via --caps if you
# explicitly want sub-50%-stock data.
# Examples:
#   3090  (firmware 100W, stock 370W) → floor max(100, 185) = 185W
#   4090  (firmware 150W, stock 450W) → floor max(150, 225) = 225W
#   5090  (firmware 250W, stock 600W) → floor max(250, 300) = 300W
floor = max(min_l, int(stock * 0.5))
# Round floor UP to nearest step boundary, max DOWN — keeps caps clean multiples of step.
start = ((floor + step - 1) // step) * step
end   = (max_l // step) * step
caps = list(range(start, end + 1, step))
# Always include the exact max_limit at the end if rounding clipped it (so we
# capture the stock-or-near anchor).
if caps[-1] != max_l:
    caps.append(max_l)
print(','.join(str(c) for c in caps))
")
  AUTO_DERIVED=1
else
  AUTO_DERIVED=0
fi
NUM_CAPS=$(echo "$CAPS" | tr ',' '\n' | wc -l | tr -d ' ')
if [ "$LOAD_MODE" = "decode-single" ]; then
  EST_MIN=$(( (NUM_CAPS * (TARGET_CAP_SECONDS * 2 + 5) + 59) / 60 ))
  EST_MAX=$(( (NUM_CAPS * (TARGET_CAP_SECONDS * 2 + 10) + 59) / 60 ))
elif [ "$LOAD_MODE" = "decode-concurrent" ]; then
  DECODE_CONCURRENT_RUN_SECONDS=$(( (TARGET_CAP_SECONDS + BENCH_RUNS - 1) / BENCH_RUNS ))
  [ "$DECODE_CONCURRENT_RUN_SECONDS" -lt 3 ] && DECODE_CONCURRENT_RUN_SECONDS=3
  EST_MIN=$(( (NUM_CAPS * (BENCH_RUNS * DECODE_CONCURRENT_RUN_SECONDS * 2 + 5) + 59) / 60 ))
  EST_MAX=$(( (NUM_CAPS * (BENCH_RUNS * DECODE_CONCURRENT_RUN_SECONDS * 2 + 10) + 59) / 60 ))
elif [ "$LOAD_MODE" = "prefill-heavy" ]; then
  EST_MIN=$(( (NUM_CAPS * BENCH_RUNS * (TARGET_PREFILL_SECONDS + 5) + 59) / 60 ))
  EST_MAX=$(( (NUM_CAPS * BENCH_RUNS * (TARGET_PREFILL_SECONDS * 3 + 10) + 59) / 60 ))
else
  EST_MIN=$(( (NUM_CAPS * 30 + 59) / 60 ))
  EST_MAX=$(( EST_MIN * 3 ))
fi
HIGHEST_CAP=$(python3 - "$CAPS" <<'PY'
import sys
print(max(int(float(x.strip())) for x in sys.argv[1].split(",") if x.strip()))
PY
)

# Persistence mode (one-time; idempotent). Do this before optional
# auto-calibration so clocks/caps behave consistently during probes.
for _gi in "${GPU_INDICES[@]}"; do nvidia-smi -pm 1 -i "$_gi" >/dev/null 2>&1 || true; done

if [ "$LOAD_MODE" = "decode-concurrent" ] && [ "$CONCURRENCY_AUTO" -eq 1 ]; then
  echo "[calibrate] --concurrency auto: probing stream count at ${HIGHEST_CAP}W cap"
  echo "[calibrate] target load: actual power >= $(python3 - "$LOAD_TARGET" <<'PY'
import sys
print(f"{float(sys.argv[1]) * 100:.0f}%")
PY
) of cap; max probe concurrency: ${MAX_CONCURRENCY_PROBE}"
  for _gi in "${GPU_INDICES[@]}"; do nvidia-smi -pl "$HIGHEST_CAP" -i "$_gi" >/dev/null; done
  sleep 2

  CAL_DIR=$(mktemp -d /tmp/power-cap-autoload.XXXXXX)
  BEST_N=""
  BEST_TPS=0
  BEST_POWER="?"
  BEST_RATIO=0
  SELECTED_N=""
  PREV_N=""
  PREV_TPS=""
  PREV_POWER=""
  PREV_RATIO=""
  ANY_RATIO_GE_050=0
  for CANDIDATE in 4 6 8 12 16; do
    if [ "$CANDIDATE" -gt "$MAX_CONCURRENCY_PROBE" ]; then
      break
    fi
    read -r PROBE_N PROBE_TPS PROBE_POWER PROBE_RATIO PROBE_FAILS PROBE_WALL < <(
      run_concurrency_probe "$CANDIDATE" "$HIGHEST_CAP" "$CAL_DIR"
    )
    echo "[calibrate] N=${PROBE_N} draw=${PROBE_POWER}W/$HIGHEST_CAP (${PROBE_RATIO}) aggregate=${PROBE_TPS} TPS fails=${PROBE_FAILS} wall=${PROBE_WALL}s"
    if [ "$PROBE_FAILS" -gt 0 ]; then
      echo "[calibrate] N=${PROBE_N} had request failures; stopping probe growth."
      break
    fi
    RATIO_GE_050=$(python3 - "$PROBE_RATIO" <<'PY'
import sys
print("1" if float(sys.argv[1]) >= 0.50 else "0")
PY
)
    [ "$RATIO_GE_050" = "1" ] && ANY_RATIO_GE_050=1
    if [ -z "$BEST_N" ]; then
      BEST_N="$PROBE_N"
      BEST_TPS="$PROBE_TPS"
      BEST_POWER="$PROBE_POWER"
      BEST_RATIO="$PROBE_RATIO"
    fi
    if [ -z "$PREV_N" ]; then
      FAST_PATH=$(python3 - "$PROBE_RATIO" <<'PY'
import sys
print("1" if float(sys.argv[1]) >= 0.97 else "0")
PY
)
      if [ "$FAST_PATH" = "1" ]; then
        SELECTED_N="$PROBE_N"
        echo "[calibrate] N=${PROBE_N} ratio=${PROBE_RATIO} reached fast-path threshold (>=0.97); selecting N=${SELECTED_N}."
        break
      fi
      PREV_N="$PROBE_N"
      PREV_TPS="$PROBE_TPS"
      PREV_POWER="$PROBE_POWER"
      PREV_RATIO="$PROBE_RATIO"
      continue
    fi

    read -r TPS_DELTA DRAW_DELTA TPS_IMPROVED DRAW_IMPROVED < <(python3 - "$PREV_TPS" "$PROBE_TPS" "$PREV_POWER" "$PROBE_POWER" <<'PY'
import sys
prev_tps, cur_tps, prev_power, cur_power = map(float, sys.argv[1:5])
tps_delta = (cur_tps - prev_tps) / max(prev_tps, 1e-9)
draw_delta = (cur_power - prev_power) / max(prev_power, 1e-9)
print(f"{tps_delta * 100:.1f} {draw_delta * 100:.1f} {1 if tps_delta > 0.03 else 0} {1 if draw_delta > 0.03 else 0}")
PY
)
    if [ "$TPS_IMPROVED" = "1" ] && [ "$DRAW_IMPROVED" = "1" ]; then
      BEST_N="$PROBE_N"
      BEST_TPS="$PROBE_TPS"
      BEST_POWER="$PROBE_POWER"
      BEST_RATIO="$PROBE_RATIO"
      PREV_N="$PROBE_N"
      PREV_TPS="$PROBE_TPS"
      PREV_POWER="$PROBE_POWER"
      PREV_RATIO="$PROBE_RATIO"
      continue
    fi

    SELECTED_N="$BEST_N"
    REASON="plateau"
    if [ "$TPS_IMPROVED" != "1" ] && [ "$DRAW_IMPROVED" != "1" ]; then
      REASON="TPS and draw plateau"
    elif [ "$TPS_IMPROVED" != "1" ]; then
      REASON="TPS plateau"
    elif [ "$DRAW_IMPROVED" != "1" ]; then
      REASON="draw plateau"
    fi
    echo "[calibrate] plateau at N=${PROBE_N} (${REASON}; TPS ${PREV_TPS}→${PROBE_TPS} = ${TPS_DELTA}%, draw ${PREV_POWER}W→${PROBE_POWER}W = ${DRAW_DELTA}%); selecting N=${SELECTED_N}."
    break
  done
  if [ -z "$SELECTED_N" ]; then
    SELECTED_N="$BEST_N"
    if [ "$ANY_RATIO_GE_050" -eq 0 ]; then
      echo "[calibrate] selected N=${SELECTED_N}: best non-failing aggregate TPS before target/load limit (draw=${BEST_POWER}W ratio=${BEST_RATIO})."
      echo "[calibrate] If draw is still far below cap, increase --max-concurrency-probe or use --load-mode prefill-heavy."
    else
      echo "[calibrate] reached --max-concurrency-probe=${MAX_CONCURRENCY_PROBE}; selecting N=${SELECTED_N}."
    fi
  fi
  if [ "$CONCURRENCY_STRETCH" -gt 0 ]; then
    STRETCHED_N=$((SELECTED_N + CONCURRENCY_STRETCH))
    echo "[calibrate] --concurrency-stretch ${CONCURRENCY_STRETCH}: bumping N from ${SELECTED_N} to ${STRETCHED_N} to probe headroom past plateau pick."
    echo "[calibrate]   This intentionally pushes above what plateau-detect chose. Expect lower per-stream TPS but possibly higher actual draw at the same cap."
    echo "[calibrate]   Useful when sweep shows draw plateau below cap (e.g. 547W actual at 600W cap on a 5090). See docs/HARDWARE.md for interpretation."
    SELECTED_N="$STRETCHED_N"
    CALIBRATION_NOTE="auto-selected concurrency=${SELECTED_N} (plateau-pick + stretch ${CONCURRENCY_STRETCH}) at ${HIGHEST_CAP}W cap (target=${LOAD_TARGET}, max-probe=${MAX_CONCURRENCY_PROBE})"
  else
    CALIBRATION_NOTE="auto-selected concurrency=${SELECTED_N} at ${HIGHEST_CAP}W cap (target=${LOAD_TARGET}, max-probe=${MAX_CONCURRENCY_PROBE})"
  fi
  CONCURRENCY="$SELECTED_N"
  rm -rf "$CAL_DIR"
  echo
fi

if [ "$LOAD_MODE" = "prefill-heavy" ]; then
  if [ -n "${BENCH_PREFILL_FILLER_REPEATS:-}" ]; then
    PREFILL_FILLER_REPEATS="$BENCH_PREFILL_FILLER_REPEATS"
    CALIBRATION_NOTE="prefill filler_repeats=${PREFILL_FILLER_REPEATS} from BENCH_PREFILL_FILLER_REPEATS override"
  else
    # Detect model's max context window for prompt-size clamping. If the
    # calibrated target prompt exceeds the context, llama.cpp truncates
    # silently → wall time becomes meaningless → bogus TPS readings.
    # Prefer llama.cpp's /props endpoint (exposes n_ctx); fall back to
    # vLLM's /v1/models (exposes max_model_len). 0 = unable to detect,
    # skip the clamp (preserve existing behavior).
    MODEL_MAX_CTX=$(
      curl -sf --max-time 5 "${URL}/props" 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('default_generation_settings',{}).get('n_ctx',0))" 2>/dev/null \
        || curl -sf --max-time 5 "${URL}/v1/models" 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); m=d.get('data',[{}])[0]; print(m.get('max_model_len', m.get('context_length',0)))" 2>/dev/null \
        || echo 0
    )
    MODEL_MAX_CTX="${MODEL_MAX_CTX:-0}"
    if [ "$MODEL_MAX_CTX" -gt 0 ]; then
      echo "[calibrate] detected model context: ${MODEL_MAX_CTX} tokens (will clamp prompt to 90%)"
    else
      echo "[calibrate] could not detect model context — proceeding without clamp (set MODEL_MAX_CTX env if needed)"
    fi
    echo "[calibrate] prefill-heavy: sizing prompt at ${HIGHEST_CAP}W cap for ~${TARGET_PREFILL_SECONDS}s at the fastest cap"
    for _gi in "${GPU_INDICES[@]}"; do nvidia-smi -pl "$HIGHEST_CAP" -i "$_gi" >/dev/null; done
    sleep 2
    CAL_LOG="/tmp/power-cap-prefill-calibration.log"
    : > "$CAL_LOG"
    read -r PROBE_TPS PROBE_TOKENS PROBE_WALL < <(
      bench_prefill_once "$PREFILL_CALIBRATION_REPEATS" "$HIGHEST_CAP" calibration "$CAL_LOG" 90
    )
    if ! [[ "$PROBE_TPS" =~ ^[0-9]+\.?[0-9]*$ ]] || ! [[ "$PROBE_TOKENS" =~ ^[0-9]+$ ]] || [ "$PROBE_TOKENS" -le 0 ]; then
      echo "[error] prefill calibration failed; see $CAL_LOG" >&2
      exit 1
    fi
    read -r PREFILL_FILLER_REPEATS PREFILL_PROMPT_TOKENS PREFILL_CLAMPED < <(python3 - "$PROBE_TPS" "$PROBE_TOKENS" "$TARGET_PREFILL_SECONDS" "$PREFILL_CALIBRATION_REPEATS" "$MODEL_MAX_CTX" <<'PY'
import math
import sys

tps = float(sys.argv[1])
probe_tokens = int(sys.argv[2])
target_seconds = int(sys.argv[3])
tokens_per_repeat = max(probe_tokens / float(sys.argv[4]), 1.0)
max_ctx = int(sys.argv[5])
target_tokens = max(1000, int(round(tps * target_seconds)))
clamped = 0
if max_ctx > 0:
    safe_max = int(max_ctx * 0.9)
    # Subtract some headroom for the chat template + max_tokens overhead
    safe_max = max(1000, safe_max - 256)
    if target_tokens > safe_max:
        target_tokens = safe_max
        clamped = 1
repeats = max(1, int(math.floor(target_tokens / tokens_per_repeat)))
print(repeats, int(round(repeats * tokens_per_repeat)), clamped)
PY
)
    if [ "${PREFILL_CLAMPED:-0}" = "1" ]; then
      EFFECTIVE_SECONDS=$(awk "BEGIN{printf \"%.1f\", ${PREFILL_PROMPT_TOKENS} / ${PROBE_TPS}}")
      CALIBRATION_NOTE="prefill probe at ${HIGHEST_CAP}W: ${PROBE_TOKENS} tok in ${PROBE_WALL}s = ${PROBE_TPS} TPS; target=${TARGET_PREFILL_SECONDS}s would have needed >${MODEL_MAX_CTX}-token prompt, clamped to 90% of model ctx → filler_repeats=${PREFILL_FILLER_REPEATS} (~${PREFILL_PROMPT_TOKENS} prompt tok, ~${EFFECTIVE_SECONDS}s effective at high cap)"
      echo "[calibrate] ⚠️  clamped: ${CALIBRATION_NOTE}"
      echo "[calibrate] note: per-cap walls at the high end will be < ${TARGET_PREFILL_SECONDS}s. For longer wall, restart engine with bigger context (e.g. -c $((MODEL_MAX_CTX * 2))) or lower --target-prefill-seconds."
    else
      CALIBRATION_NOTE="prefill probe at ${HIGHEST_CAP}W: ${PROBE_TOKENS} tok in ${PROBE_WALL}s = ${PROBE_TPS} TPS; target=${TARGET_PREFILL_SECONDS}s -> filler_repeats=${PREFILL_FILLER_REPEATS} (~${PREFILL_PROMPT_TOKENS} prompt tok)"
      echo "[calibrate] ${CALIBRATION_NOTE}"
    fi
    echo
  fi
fi

echo "[setup] GPUs ${GPU_LIST_CSV}: $GPU_NAME ($GPU_VRAM MiB)"
echo "[setup] power envelope: ${MIN_LIMIT}W (min) → ${STOCK_TDP}W (default) → ${MAX_LIMIT}W (max)"
echo "[setup] cooling:   $COOLING"
if [ "$AUTO_DERIVED" -eq 1 ]; then
  echo "[setup] sweep caps: $NUM_CAPS caps in ${STEP_SIZE}W increments (override via --caps or --step-size)"
  echo "[setup]            $CAPS W"
else
  echo "[setup] sweep caps: $NUM_CAPS caps (user-specified)"
  echo "[setup]            $CAPS W"
fi
echo "[setup] load mode: $LOAD_MODE$([ "$LOAD_MODE" = "decode-single" ] && echo " (${TARGET_CAP_SECONDS}s × 2 timed streams)")$([ "$LOAD_MODE" = "decode-concurrent" ] && echo " (concurrency=$CONCURRENCY, ${DECODE_CONCURRENT_RUN_SECONDS}s/run × ${BENCH_RUNS} runs × 2 timed batches)")$([ "$LOAD_MODE" = "prefill-heavy" ] && echo " (target-prefill=${TARGET_PREFILL_SECONDS}s, filler_repeats=${PREFILL_FILLER_REPEATS})")$([ "$LOAD_MODE" != "decode-single" ] && echo " (bench-runs=$BENCH_RUNS)")"
[ -n "$CALIBRATION_NOTE" ] && echo "[setup] calibration: $CALIBRATION_NOTE"
echo "[setup] estimated runtime: ${EST_MIN}-${EST_MAX} min (${NUM_CAPS} caps; range varies with cap throttle + bench shape)"
echo "[setup] reset at end: $([ $RESET -eq 1 ] && echo yes || echo no)"

# Warn on configurations known to produce biased data
if [ "${BENCH_WARMUPS:-1}" = "0" ]; then
  echo "[warn] BENCH_WARMUPS=0: the FIRST cap of the sweep will have cold-cache bias"
  echo "[warn]   (model weights not warm in any sense — narrative bench runs first,"
  echo "[warn]    so first 250-500 narrative tokens absorb cold-start cost)."
  echo "[warn]   Subsequent caps are fine (cache warm from previous cap)."
  echo "[warn]   Recommend BENCH_WARMUPS=1 minimum unless you can discard first-cap data."
fi

if [ "$LOAD_MODE" != "decode-single" ] && [ "$BENCH_RUNS" -gt 1 ]; then
  echo "[warn] --bench-runs=${BENCH_RUNS}: this is anchor-grade mode, not the fast default."
  echo "[warn]   Sweep time scales linearly with --bench-runs; use --bench-runs 1 for ≤15 min full sweeps."
fi

echo

if [ "$LOAD_MODE" = "decode-concurrent" ]; then
  echo "[check] decode-concurrent scheduling check (best effort, N=${CONCURRENCY})"
  echo "[check] /v1/models usually exposes max_model_len, not max_num_seqs; probing with tiny concurrent requests."
  PROBE_DIR=$(mktemp -d /tmp/power-cap-concurrency-probe.XXXXXX)
  PROBE_PIDS=()
  for i in $(seq 1 "$CONCURRENCY"); do
    REQ_FILE="$PROBE_DIR/req-${i}.json"
    python3 - "$REQ_FILE" "$MODEL" <<'PY'
import json
import sys

path, model = sys.argv[1:3]
body = {
    "model": model,
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 1,
    "temperature": 0,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(body, f)
PY
    (
      curl -sS -o "$PROBE_DIR/out-${i}.json" -w "%{http_code}" --max-time 30 \
        "${URL}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "@${REQ_FILE}" > "$PROBE_DIR/code-${i}.txt"
    ) &
    PROBE_PIDS+=("$!")
  done
  PROBE_FAILS=0
  for pid in "${PROBE_PIDS[@]}"; do
    if ! wait "$pid"; then
      PROBE_FAILS=$((PROBE_FAILS + 1))
    fi
  done
  PROBE_BAD_CODES=$(python3 - "$PROBE_DIR" "$CONCURRENCY" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
n = int(sys.argv[2])
bad = []
for i in range(1, n + 1):
    p = root / f"code-{i}.txt"
    code = p.read_text().strip() if p.exists() else "curl-failed"
    if code != "200":
        bad.append(f"{i}:{code}")
print(",".join(bad))
PY
)
  if [ "$PROBE_FAILS" -gt 0 ] || [ -n "$PROBE_BAD_CODES" ]; then
    echo "[warn] concurrency probe had failures/non-200 responses: pids=${PROBE_FAILS}, http=${PROBE_BAD_CODES:-none}"
    echo "[warn] If the sweep reports 503/timeouts, lower --concurrency or raise compose --max-num-seqs."
  else
    echo "[check] concurrency probe passed at N=${CONCURRENCY}"
  fi
  rm -rf "$PROBE_DIR"
  echo
fi

# Sweep
RESULTS_FILE=/tmp/power-cap-summary.md
{
  echo "# Power-cap sweep — $GPU_NAME (GPUs $GPU_LIST_CSV)"
  echo ""
  echo "**GPU:** $GPU_NAME &nbsp; **VRAM:** ${GPU_VRAM} MiB &nbsp; **Stock TDP:** ${STOCK_TDP}W &nbsp; **Cooling:** ${COOLING}"
  echo "**Model:** \`${MODEL}\` &nbsp; **Engine:** \`${CONTAINER}\` &nbsp; **Endpoint:** ${URL}"
  echo "**Load mode:** \`${LOAD_MODE}\`$([ "$LOAD_MODE" = "decode-single" ] && echo " (${TARGET_CAP_SECONDS}s × 2 timed streams)")$([ "$LOAD_MODE" = "decode-concurrent" ] && echo " (concurrency=${CONCURRENCY}, ${DECODE_CONCURRENT_RUN_SECONDS}s/run × ${BENCH_RUNS} runs × 2 timed batches)")$([ "$LOAD_MODE" = "prefill-heavy" ] && echo " (target-prefill=${TARGET_PREFILL_SECONDS}s, filler_repeats=${PREFILL_FILLER_REPEATS})")$([ "$LOAD_MODE" != "decode-single" ] && echo " (bench-runs=${BENCH_RUNS})")"
  [ -n "$CALIBRATION_NOTE" ] && echo "**Calibration:** ${CALIBRATION_NOTE}"
  # --include-commit: stamp club-3090 git short SHA next to the date if requested.
  # Suppress entirely (rather than show "n/a") when run from a non-clone or
  # when git isn't reachable — closes the curl-pipe-from-docs UX hole.
  COMMIT_FRAGMENT=""
  if [ "${INCLUDE_COMMIT:-0}" = "1" ]; then
    COMMIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true)
    if [ -n "$COMMIT_SHA" ]; then
      COMMIT_FRAGMENT=" &nbsp; **club-3090 commit:** \`${COMMIT_SHA}\`"
    fi
  fi
  echo "**Date:** $(date -u +%Y-%m-%dT%H:%M:%S)Z${COMMIT_FRAGMENT}"
  echo ""
  if [ "$COOLING" = "unspecified" ]; then
    echo "> ⚠️  Cooling class not specified at run time. Add **air / water / AIO** when posting"
    echo "> this data — water-cooled cards sustain full board power; air-cooled thermal-throttle"
    echo "> at ~80-83 °C and may cap below the software limit regardless of \`-pl\` setting."
    echo ""
  fi
  echo "> Cross-rig comparisons require **matching model + engine class** — TPS scales with"
  echo "> model size and quant (e.g. Qwen3.6-27B-AutoRound at 30 TPS, Gemma-4-31B-AutoRound +"
  echo "> MTP at 100 TPS). The *shape* of the efficiency knee is the cross-rig signal; absolute"
  echo "> numbers only compare like-to-like."
  echo ""
  echo "| Cap (W) | Narr wall TPS | Code wall TPS | Actual power (W) | GPU temp (°C) | SM clk (MHz) | Mem clk (MHz) | Pwr-throttle % | P-state | TPS/W (narr) |"
  echo "|--------:|--------------:|--------------:|-----------------:|--------------:|-------------:|--------------:|---------------:|:-------:|-------------:|"
} > "$RESULTS_FILE"

IFS=',' read -ra CAP_ARRAY <<< "$CAPS"

for CAP in "${CAP_ARRAY[@]}"; do
  CAP=$(echo "$CAP" | tr -d ' ')
  CAP_START_NS=$(date +%s%N)
  CAP_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "================================================"
  echo "=== Cap: ${CAP}W (GPUs $GPU_LIST_CSV) @ ${CAP_START_UTC} ==="
  echo "================================================"

  # Apply cap to every participating GPU (symmetric sweep)
  _cap_ok=1
  for _gi in "${GPU_INDICES[@]}"; do
    nvidia-smi -pl "$CAP" -i "$_gi" >/dev/null 2>&1 || _cap_ok=0
  done
  if [ "$_cap_ok" -ne 1 ]; then
    echo "[warn] failed to set ${CAP}W on one or more GPUs — skipping"
    continue
  fi

  # Verify cap applied
  ACTUAL_LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "$PRIMARY_GPU" | head -1 | tr -d ' ')
  echo "[verify] limit set to: ${ACTUAL_LIMIT}W"
  echo

  # Brief settle (let driver re-clock)
  sleep 3

  # Start background power-draw sampler at 0.5s intervals.
  # Capturing under-load power requires sampling DURING bench runs — bench.sh's
  # final "GPU state" line samples after all runs complete and may catch the
  # card mid-idle (~40W) instead of under load (~330W). The sampler writes
  # CSV: index, utilization%, power.draw_W, temp_C — we post-process for
  # median power across samples where utilization > 50% (under-load median).
  SAMPLE_FILE="/tmp/power-cap-N${CAP}-samples.csv"
  (
    while true; do
      nvidia-smi --query-gpu=index,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_thermal_slowdown \
        --format=csv,noheader,nounits -i "$GPU_LIST_CSV" 2>/dev/null | aggregate_gpu_sample
      sleep 0.5
    done
  ) > "$SAMPLE_FILE" &
  SAMPLER_PID=$!

  # Run bench at reduced precision for sweep efficiency.
  # Canonical bench.sh uses WARMUPS=3 RUNS=5 + 1000/800 max_tokens =
  # 8 × (1000+800) tokens = ~14k tokens × ~10ms/token = ~2 min/cap.
  # For sweep purposes we don't need ±0.5% TPS precision — we need the curve
  # shape and stable under-load power readings. With WARMUPS=1 RUNS=2 +
  # 500/400 max_tokens = 3 × 900 tokens = 2,700 tokens → ~25-35s/cap on a
  # mid-range card, ~50s on a heavily-power-starved cap. That's enough
  # sustained load for the sampler to collect 50+ under-load samples for
  # a stable median.
  LOG_FILE="/tmp/power-cap-N${CAP}.log"
  case "$LOAD_MODE" in
    decode-single)
      # Single-stream decode is time-bounded instead of token-bounded. Fixed
      # token counts make low caps take 2-4× longer than high caps; fixed wall
      # seconds keep sweep runtime portable across 3090/4090/5090/A-series while
      # still providing sustained under-load samples for the power median.
      echo "[bench] decode-single @ ${CAP}W cap (${TARGET_CAP_SECONDS}s narrative + ${TARGET_CAP_SECONDS}s code, output: $LOG_FILE)"
      : > "$LOG_FILE"
      if ! NARR_TPS=$(bench_decode_single_for_seconds narrative "$TARGET_CAP_SECONDS" "$CAP" "$LOG_FILE"); then
        kill $SAMPLER_PID 2>/dev/null || true
        wait $SAMPLER_PID 2>/dev/null || true
        SAMPLER_PID=""
        echo "[warn] narrative timed bench failed at ${CAP}W"
        continue
      fi
      NARR_TPS=$(echo "$NARR_TPS" | tail -1)
      if ! CODE_TPS=$(bench_decode_single_for_seconds code "$TARGET_CAP_SECONDS" "$CAP" "$LOG_FILE"); then
        kill $SAMPLER_PID 2>/dev/null || true
        wait $SAMPLER_PID 2>/dev/null || true
        SAMPLER_PID=""
        echo "[warn] code timed bench failed at ${CAP}W"
        continue
      fi
      CODE_TPS=$(echo "$CODE_TPS" | tail -1)
      kill $SAMPLER_PID 2>/dev/null || true
      wait $SAMPLER_PID 2>/dev/null || true
      SAMPLER_PID=""
      echo
      ;;

    decode-concurrent)
      # Concurrent decode is time-bounded like decode-single: each measured
      # batch runs N streaming requests until curl's wall timer cuts them off.
      # Aggregate TPS is total streamed token-chunks across all streams divided
      # by batch wall time.
      echo "[bench] decode-concurrent @ ${CAP}W cap, N=${CONCURRENCY}, runs=${BENCH_RUNS}, ${DECODE_CONCURRENT_RUN_SECONDS}s/run narrative + ${DECODE_CONCURRENT_RUN_SECONDS}s/run code (output: $LOG_FILE)"
      : > "$LOG_FILE"
      NARR_TPS_VALUES=()
      CODE_TPS_VALUES=()
      for RUN_IDX in $(seq 1 "$BENCH_RUNS"); do
        if ! NARR_RUN_TPS=$(bench_decode_concurrent_for_seconds narrative "$DECODE_CONCURRENT_RUN_SECONDS" "$CAP" "$RUN_IDX" "$LOG_FILE"); then
          echo "[warn] narrative timed concurrent run ${RUN_IDX} failed at ${CAP}W" | tee -a "$LOG_FILE"
          continue
        fi
        NARR_RUN_TPS=$(echo "$NARR_RUN_TPS" | tail -1)
        NARR_TPS_VALUES+=("$NARR_RUN_TPS")
        if ! CODE_RUN_TPS=$(bench_decode_concurrent_for_seconds code "$DECODE_CONCURRENT_RUN_SECONDS" "$CAP" "$RUN_IDX" "$LOG_FILE"); then
          echo "[warn] code timed concurrent run ${RUN_IDX} failed at ${CAP}W" | tee -a "$LOG_FILE"
          continue
        fi
        CODE_RUN_TPS=$(echo "$CODE_RUN_TPS" | tail -1)
        CODE_TPS_VALUES+=("$CODE_RUN_TPS")
      done

      NARR_TPS=$(python3 - "${NARR_TPS_VALUES[@]}" <<'PY'
import statistics
import sys
vals = [float(x) for x in sys.argv[1:]]
print(f"{statistics.median(vals):.2f}" if vals else "?")
PY
)
      CODE_TPS=$(python3 - "${CODE_TPS_VALUES[@]}" <<'PY'
import statistics
import sys
vals = [float(x) for x in sys.argv[1:]]
print(f"{statistics.median(vals):.2f}" if vals else "?")
PY
)
      echo "[summary] median aggregate TPS across ${BENCH_RUNS} run(s): narr=${NARR_TPS}, code=${CODE_TPS}" | tee -a "$LOG_FILE"

      kill $SAMPLER_PID 2>/dev/null || true
      wait $SAMPLER_PID 2>/dev/null || true
      SAMPLER_PID=""
      echo
      ;;

    prefill-heavy)
      # Prefill-heavy: send a calibrated prompt with max_tokens=10.
      # Prefill is compute-bound by definition (single forward pass through
      # all layers on the entire prompt). Exposes compute-knee on any card,
      # since prefill TPS scales directly with tensor-core throughput.
      # Less commonly useful than decode-concurrent for "real workload"
      # framing, but produces a clean compute-only curve for diagnostic.
      echo "[bench] prefill-heavy @ ${CAP}W cap, runs=${BENCH_RUNS}, filler_repeats=${PREFILL_FILLER_REPEATS}, target-fast-cap=${TARGET_PREFILL_SECONDS}s (output: $LOG_FILE)"
      : > "$LOG_FILE"
      PREFILL_TPS_VALUES=()
      for RUN_IDX in $(seq 1 "$BENCH_RUNS"); do
        if ! PREFILL_RESULT=$(bench_prefill_once "$PREFILL_FILLER_REPEATS" "$CAP" "$RUN_IDX" "$LOG_FILE" 180); then
          echo "[warn] prefill-heavy run ${RUN_IDX} failed at ${CAP}W" | tee -a "$LOG_FILE"
          continue
        fi
        PREFILL_RUN_TPS=$(echo "$PREFILL_RESULT" | awk '{print $1}')
        PREFILL_TPS_VALUES+=("$PREFILL_RUN_TPS")
      done
      NARR_TPS=$(python3 - "${PREFILL_TPS_VALUES[@]}" <<'PY'
import statistics
import sys
vals = [float(x) for x in sys.argv[1:]]
print(f"{statistics.median(vals):.2f}" if vals else "?")
PY
)
      CODE_TPS="$NARR_TPS"   # use same column; prefill doesn't differentiate narr/code
      echo "[summary] median prefill TPS across ${BENCH_RUNS} run(s): ${NARR_TPS}" | tee -a "$LOG_FILE"

      kill $SAMPLER_PID 2>/dev/null || true
      wait $SAMPLER_PID 2>/dev/null || true
      SAMPLER_PID=""
      echo
      ;;
  esac

  # NARR_TPS / CODE_TPS are populated per-mode inside the case above.

  # Compute median under-load power and peak temp from the sampler.
  # Filter to samples where GPU utilization > 50% (i.e. actively decoding).
  # Falls back to bench.sh's GPU-state line if sampler captured no under-load
  # samples (rare; only happens if bench.sh failed silently or finished before
  # the sampler took its first reading).
  if [ -s "$SAMPLE_FILE" ]; then
    UNDER_LOAD_STATS=$(python3 -c "
import sys
from collections import Counter
samples = []
with open('$SAMPLE_FILE') as f:
    for line in f:
        try:
            parts = [x.strip() for x in line.strip().split(',')]
            if len(parts) < 9:
                continue
            idx, util, power, temp, sm_clk, mem_clk, pstate, pwr_thr, therm_thr = parts
            if int(util) > 50:
                samples.append((
                    float(power),
                    int(temp),
                    int(sm_clk),
                    int(mem_clk),
                    pstate,
                    pwr_thr == 'Active',
                    therm_thr == 'Active',
                ))
        except Exception:
            continue
if not samples:
    print('? ? ? ? ? ? ?')
else:
    powers = sorted(s[0] for s in samples)
    temps  = [s[1] for s in samples]
    sm_clks = sorted(s[2] for s in samples)
    mem_clks = sorted(s[3] for s in samples)
    pstates = [s[4] for s in samples]
    n = len(samples)
    pwr_thr_pct = sum(1 for s in samples if s[5]) / n * 100
    therm_thr_pct = sum(1 for s in samples if s[6]) / n * 100
    median_power = powers[len(powers)//2]
    peak_temp    = max(temps)
    median_sm    = sm_clks[len(sm_clks)//2]
    median_mem   = mem_clks[len(mem_clks)//2]
    dom_pstate   = Counter(pstates).most_common(1)[0][0]
    print(f'{median_power:.2f} {peak_temp} {median_sm} {median_mem} {dom_pstate} {pwr_thr_pct:.0f} {therm_thr_pct:.0f}')
" 2>/dev/null || echo "? ? ? ? ? ? ?")
    ACTUAL_POWER=$(echo "$UNDER_LOAD_STATS" | awk '{print $1}')
    GPU_TEMP=$(echo "$UNDER_LOAD_STATS"      | awk '{print $2}')
    SM_CLK=$(echo "$UNDER_LOAD_STATS"        | awk '{print $3}')
    MEM_CLK=$(echo "$UNDER_LOAD_STATS"       | awk '{print $4}')
    PSTATE=$(echo "$UNDER_LOAD_STATS"        | awk '{print $5}')
    PWR_THR_PCT=$(echo "$UNDER_LOAD_STATS"   | awk '{print $6}')
    THERM_THR_PCT=$(echo "$UNDER_LOAD_STATS" | awk '{print $7}')
  else
    ACTUAL_POWER="?"; GPU_TEMP="?"
    SM_CLK="?"; MEM_CLK="?"; PSTATE="?"; PWR_THR_PCT="?"; THERM_THR_PCT="?"
  fi

  # Fallback to bench.sh GPU-state line if sampler returned ?
  if [ "$ACTUAL_POWER" = "?" ]; then
    GPU_STATE_LINE=$(grep -A2 "GPU state" "$LOG_FILE" | grep ",$PRIMARY_GPU," | head -1 || grep -A2 "GPU state" "$LOG_FILE" | grep "^${PRIMARY_GPU}," | head -1 || echo "")
    ACTUAL_POWER=$(echo "$GPU_STATE_LINE" | awk -F', ' '{print $5}' | grep -oE '[0-9]+\.?[0-9]*' | head -1 || echo "?")
    GPU_TEMP=$(echo "$GPU_STATE_LINE"     | awk -F', ' '{print $6}' | tr -d ' ' || echo "?")
  fi

  # TPS/W efficiency calc (if both numeric)
  if [[ "$NARR_TPS" =~ ^[0-9]+\.[0-9]+$ && "$ACTUAL_POWER" =~ ^[0-9]+\.?[0-9]*$ && "$ACTUAL_POWER" != "0" ]]; then
    EFFICIENCY=$(awk "BEGIN{printf \"%.3f\", $NARR_TPS / $ACTUAL_POWER}")
  else
    EFFICIENCY="?"
  fi

  CAP_END_NS=$(date +%s%N)
  CAP_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  CAP_WALL_S=$(python3 - "$CAP_START_NS" "$CAP_END_NS" <<'PY'
import sys
start, end = map(int, sys.argv[1:3])
print(f"{(end - start) / 1e9:.1f}")
PY
)

  printf "[result] cap=%sW actual_W=%s temp=%s sm_clk=%s mem_clk=%s pstate=%s throttle_pwr=%s throttle_thermal=%s narr=%s code=%s tps_per_w=%s\n" \
    "$CAP" "$ACTUAL_POWER" "$GPU_TEMP" "$SM_CLK" "$MEM_CLK" "$PSTATE" "$PWR_THR_PCT" "$THERM_THR_PCT" "$NARR_TPS" "$CODE_TPS" "$EFFICIENCY"
  printf "[time] %sW cap wall=%ss start=%s end=%s\n\n" \
    "$CAP" "$CAP_WALL_S" "$CAP_START_UTC" "$CAP_END_UTC"

  printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
    "$CAP" "$NARR_TPS" "$CODE_TPS" "$ACTUAL_POWER" "$GPU_TEMP" "$SM_CLK" "$MEM_CLK" "$PWR_THR_PCT" "$PSTATE" "$EFFICIENCY" \
    >> "$RESULTS_FILE"
done

# Detect boost-clock plateaus from the captured data.
# A plateau is 3+ consecutive caps where SM clock is identical, actual draw is
# within ±2W, and TPS is within ±1%. This pattern signals firmware boost-state
# locking — raising the cap doesn't push past until the firmware decides to
# step to a new operating point. See learnings/qwen3.6-35b-a3b.md for examples.
PLATEAU_LINES=$(python3 - "$RESULTS_FILE" <<'PY'
import re
import sys

path = sys.argv[1]
rows = []
row_re = re.compile(
    r'^\|\s*(\d+)\s*'                         # cap
    r'\|\s*([\d.?]+)\s*'                      # narr TPS
    r'\|\s*([\d.?]+)\s*'                      # code TPS
    r'\|\s*([\d.?]+)\s*'                      # actual_W
    r'\|\s*(\d+|\?)\s*'                       # temp
    r'\|\s*(\d+|\?)\s*'                       # sm_clk
    r'\|\s*(\d+|\?)\s*'                       # mem_clk
    r'\|\s*([\d.?]+)\s*'                      # pwr_throttle %
    r'\|\s*([P\d?]+)\s*'                      # pstate
    r'\|\s*([\d.?]+)\s*\|\s*$'                # tps/w
)

with open(path) as f:
    for line in f:
        m = row_re.match(line.strip())
        if not m:
            continue
        try:
            cap = int(m.group(1))
            narr = float(m.group(2))
            draw = float(m.group(4))
            sm = int(m.group(6))
            rows.append((cap, narr, draw, sm))
        except ValueError:
            continue

# A plateau is a run where draw and TPS are functionally identical (within ±2W
# and ±1% TPS) regardless of cap. SM clock often locks to a single value across
# the plateau, but firmware can use slightly different SM setpoints (e.g. 1605
# vs 1620 MHz) within the same operating-point envelope — so we report SM as a
# range when it varies, single value when locked.
plateaus = []
i = 0
while i < len(rows):
    j = i + 1
    base_draw = rows[i][2]
    base_tps = rows[i][1]
    sm_min = sm_max = rows[i][3]
    while j < len(rows):
        draw_match = abs(rows[j][2] - base_draw) <= 2.0
        tps_match = abs(rows[j][1] - base_tps) / max(base_tps, 0.1) <= 0.01
        if draw_match and tps_match:
            sm_min = min(sm_min, rows[j][3])
            sm_max = max(sm_max, rows[j][3])
            j += 1
        else:
            break
    if j - i >= 3:
        sm_label = str(sm_min) if sm_min == sm_max else f"{sm_min}-{sm_max}"
        plateaus.append((rows[i][0], rows[j-1][0], sm_label, base_draw, base_tps))
        i = j
    else:
        i += 1

for start, end, sm, draw, tps in plateaus:
    # tab-separated so "1605-1620" stays as one field
    print(f"{start}\t{end}\t{sm}\t{draw:.2f}\t{tps:.2f}")
PY
)

if [ -n "$PLATEAU_LINES" ]; then
  echo "================================================"
  echo "Boost-clock plateau(s) detected"
  echo "================================================"
  while IFS=$'\t' read -r START END SM DRAW TPS; do
    [ -z "$START" ] && continue
    SPAN=$((END - START))
    printf "[plateau detected] caps %sW–%sW (%dW span) → SM %s MHz, %sW draw, %s TPS — firmware boost-clock lock; raise cap past %sW to escape\n" \
      "$START" "$END" "$SPAN" "$SM" "$DRAW" "$TPS" "$END"
  done <<< "$PLATEAU_LINES"
  echo
fi

# Reset
if [ "$RESET" -eq 1 ]; then
  echo "[reset] restoring GPUs ${GPU_LIST_CSV} to stock TDP"
else
  echo "[reset] --no-reset: restoring GPUs ${GPU_LIST_CSV} to their pre-sweep limits"
fi
restore_gpus

# Append context to results file
{
  echo ""
  if [ -n "$PLATEAU_LINES" ]; then
    echo "**Detected boost-clock plateau(s):**"
    echo ""
    while IFS=$'\t' read -r START END SM DRAW TPS; do
      [ -z "$START" ] && continue
      SPAN=$((END - START))
      echo "- Caps **${START}W–${END}W** (${SPAN}W span) → SM **${SM} MHz**, **${DRAW}W** draw, **${TPS} TPS** — firmware boost-clock plateau. Caps in this range are functionally equivalent; raise past **${END}W** to step to the next firmware operating point."
    done <<< "$PLATEAU_LINES"
    echo ""
  fi
  echo "**Reset:** $([ $RESET -eq 1 ] && echo "auto-reset to per-GPU stock" || echo "restored to pre-sweep limits (--no-reset)")"
  echo ""
  echo "**Notes:**"
  case "$LOAD_MODE" in
    decode-single)
      echo "- Load mode: \`decode-single\` — time-bounded streaming requests: ${TARGET_CAP_SECONDS}s narrative + ${TARGET_CAP_SECONDS}s code per cap."
      echo "- TPS columns are streamed token-chunks / wall seconds. If an engine emits final streaming usage before timeout, completion_tokens is used instead."
      ;;
    decode-concurrent)
      echo "- Load mode: \`decode-concurrent\` — ${CONCURRENCY} parallel streaming chat completions for ${DECODE_CONCURRENT_RUN_SECONDS}s/run narr, then ${CONCURRENCY} for ${DECODE_CONCURRENT_RUN_SECONDS}s/run code."
      echo "- \`--target-cap-seconds=${TARGET_CAP_SECONDS}\` is treated as the per-direction budget across \`--bench-runs\`; each run gets at least 3s."
      echo "- TPS columns are **median aggregate** throughput across ${BENCH_RUNS} measured timed batch(es): total streamed token-chunks across streams / batch wall time."
      echo "- ⚠️ **Variance caveat**: with \`--bench-runs 1\`, each cap is a **single batch of ${CONCURRENCY} concurrent requests**."
      echo "  Aggregate TPS can vary 10-30% between back-to-back runs at the same cap because vLLM's"
      echo "  continuous-batching window is timing-sensitive — adjacent caps may show TPS going the"
      echo "  \"wrong direction\" without that being a real signal. Read **curve shape across the full"
      echo "  sweep**, not adjacent-cap deltas. For tighter cross-rig anchors, use \`--bench-runs 3\`,"
      echo "  and/or bump \`--concurrency\` to 8 or 16 so per-stream noise averages out."
      ;;
    prefill-heavy)
      echo "- Load mode: \`prefill-heavy\` — prompt size calibrated at the highest cap for ~${TARGET_PREFILL_SECONDS}s, then reused across all caps with \`max_tokens=10\`; both TPS columns show prompt prefill TPS."
      echo "- Prefill TPS = median of ${BENCH_RUNS} run(s), each computed as response \`usage.prompt_tokens\` / request wall time."
      ;;
  esac
  echo "- Actual power = **median** of 0.5s samples taken DURING the workload where util > 50% (i.e. under-load)."
  echo "- GPU temp = **peak** during workload (not a single post-bench point sample)."
  echo "- **SM clk / Mem clk** = **median** clock speeds during in-load samples. SM clock is the compute-tier clock; memory clock is the HBM/GDDR clock."
  echo "  - If SM clock varies with cap while TPS plateaus → workload is **bandwidth-bound** (more compute headroom unused)."
  echo "  - If SM clock is pinned at max (~1.9 GHz on 3090, ~2.5+ GHz on 4090/5090) while TPS still climbs → workload is **compute-bound**."
  echo "  - Memory clock should normally pin at the card's spec max (9501 MHz on 3090, 10501 MHz on 4090, 14001 MHz on 5090). If it drops, that's a memory power-state transition worth investigating."
  echo "- **Pwr-throttle %** = % of in-load samples where firmware was actively capping draw at the set limit (\`clocks_throttle_reasons.sw_power_cap=Active\`)."
  echo "  - **100%** → power is the binding constraint at this cap; raising the cap would draw more and produce more TPS."
  echo "  - **<100%** → either workload is undersupplying the card (lift via concurrency/prefill), or thermal-throttle is taking over (check temp + cooling)."
  echo "- **P-state** = dominant firmware power state during in-load samples. P0 = max boost, P2 = sustained-load pinned, higher numbers = idle/low-load."
  echo "  - Boost-state plateaus appear when several adjacent caps all sit in the same P-state and draw identical wattage despite different cap settings (e.g. 3090s pin P2 across ~340-370W, then escape to P0 at 380W cap)."
  echo "- TPS/W efficiency lets you spot the knee — typically the highest cap before efficiency drops."
  echo "- If actual power < cap consistently and TPS is flat, the workload is **under-loading** this hardware:"
  echo "  the card can't use the extra power because it's not the bottleneck (smaller models on bigger"
  echo "  GPUs commonly land here). Use \`decode-concurrent\` or \`prefill-heavy\` to surface a useful curve."
  echo "- **Cooling class affects interpretation:** air-cooled cards thermal-throttle at ~80-83 °C, capping"
  echo "  effective sustained power below the software limit. Water-cooled / AIO cards stay at lower temps"
  echo "  and sustain the full software cap. Cross-rig comparisons should match cooling class for fairness."
} >> "$RESULTS_FILE"

echo
echo "================================================"
echo "Sweep complete. Summary at: $RESULTS_FILE"
echo "Raw bench logs at: /tmp/power-cap-N*.log"
echo "================================================"
echo
cat "$RESULTS_FILE"
