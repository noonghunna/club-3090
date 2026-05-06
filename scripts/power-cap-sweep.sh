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
#   sudo bash scripts/power-cap-sweep.sh                          # default sweep + auto-reset
#   sudo bash scripts/power-cap-sweep.sh --caps 300,340,380       # custom caps
#   sudo bash scripts/power-cap-sweep.sh --gpu 1                  # specific GPU index
#   sudo bash scripts/power-cap-sweep.sh --cooling water          # tag the run as water-cooled
#   sudo bash scripts/power-cap-sweep.sh --cooling air            # tag as air-cooled
#   sudo bash scripts/power-cap-sweep.sh --cooling aio            # tag as AIO/closed-loop
#   sudo bash scripts/power-cap-sweep.sh --no-reset               # leave at last cap (you reset manually)
#
# Output:
#   - Per-cap bench logs at /tmp/power-cap-N{wattage}.log
#   - Markdown summary at /tmp/power-cap-summary.md (paste into GitHub issue/discussion)
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
CAPS="300,320,340,360,380"
RESET=1   # 1 = reset to stock at end; 0 = leave at last cap
COOLING="unspecified"  # air|water|aio|unspecified — affects how to read the data

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)        GPU_INDEX="$2"; shift 2 ;;
    --caps)       CAPS="$2"; shift 2 ;;
    --cooling)    COOLING="$2"; shift 2 ;;
    --no-reset)   RESET=0; shift ;;
    -h|--help)
      sed -n '1,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *)            echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

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

# Auto-detect URL/CONTAINER/MODEL from the running vllm container.
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

if [ -z "${URL:-}" ] || [ -z "${MODEL:-}" ] || [ -z "${CONTAINER:-}" ]; then
  echo "[error] could not auto-detect a running container + URL + MODEL." >&2
  echo "[hint]  start a model server first (bash scripts/switch.sh <variant>)" >&2
  echo "[hint]  or pass URL=http://... CONTAINER=name MODEL=name as env vars" >&2
  echo "[got]   URL='${URL:-}' CONTAINER='${CONTAINER:-}' MODEL='${MODEL:-}'" >&2
  exit 1
fi
export URL CONTAINER MODEL
echo "[setup] target:   container=$CONTAINER url=$URL model=$MODEL"

# Capture stock TDP (so we can reset cleanly even on non-3090/4090/5090 cards)
STOCK_TDP=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits -i "$GPU_INDEX" | head -1 | tr -d ' ')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_INDEX" | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_INDEX" | head -1 | tr -d ' ')

echo "[setup] GPU $GPU_INDEX: $GPU_NAME ($GPU_VRAM MiB)"
echo "[setup] stock TDP: ${STOCK_TDP}W"
echo "[setup] cooling:   $COOLING"
echo "[setup] sweep caps: $CAPS W"
echo "[setup] reset at end: $([ $RESET -eq 1 ] && echo yes || echo no)"
echo

# Persistence mode (one-time; idempotent)
nvidia-smi -pm 1 -i "$GPU_INDEX" >/dev/null 2>&1 || true

# Sweep
RESULTS_FILE=/tmp/power-cap-summary.md
{
  echo "# Power-cap sweep — $GPU_NAME (GPU $GPU_INDEX)"
  echo ""
  echo "**GPU:** $GPU_NAME &nbsp; **VRAM:** ${GPU_VRAM} MiB &nbsp; **Stock TDP:** ${STOCK_TDP}W &nbsp; **Cooling:** ${COOLING}"
  echo "**Model:** \`${MODEL}\` &nbsp; **Engine:** \`${CONTAINER}\` &nbsp; **Endpoint:** ${URL}"
  echo "**Date:** $(date -u +%Y-%m-%dT%H:%M:%S)Z"
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
  echo "| Cap (W) | Narr wall TPS | Code wall TPS | Actual power (W) | GPU temp (°C) | TPS/W (narr) |"
  echo "|--------:|--------------:|--------------:|-----------------:|--------------:|-------------:|"
} > "$RESULTS_FILE"

IFS=',' read -ra CAP_ARRAY <<< "$CAPS"
for CAP in "${CAP_ARRAY[@]}"; do
  CAP=$(echo "$CAP" | tr -d ' ')
  echo "================================================"
  echo "=== Cap: ${CAP}W (GPU $GPU_INDEX) ==="
  echo "================================================"

  # Apply cap
  if ! nvidia-smi -pl "$CAP" -i "$GPU_INDEX" 2>&1 | tail -1; then
    echo "[warn] failed to set ${CAP}W — skipping"
    continue
  fi

  # Verify cap applied
  ACTUAL_LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "$GPU_INDEX" | head -1 | tr -d ' ')
  echo "[verify] limit set to: ${ACTUAL_LIMIT}W"
  echo

  # Brief settle (let driver re-clock)
  sleep 3

  # Run canonical bench
  LOG_FILE="/tmp/power-cap-N${CAP}.log"
  echo "[bench] running bench.sh @ ${CAP}W cap (output: $LOG_FILE)"
  if ! bash "$BENCH" 2>&1 | tee "$LOG_FILE" | tail -8; then
    echo "[warn] bench.sh failed at ${CAP}W"
    continue
  fi
  echo

  # Extract metrics
  NARR_TPS=$(grep -A1 "summary \[narrative\]" "$LOG_FILE" | grep "wall_TPS" | head -1 | grep -oE 'mean= *[0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+' || echo "?")
  CODE_TPS=$(grep -A1 "summary \[code\]"      "$LOG_FILE" | grep "wall_TPS" | head -1 | grep -oE 'mean= *[0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+' || echo "?")
  GPU_STATE_LINE=$(grep -A2 "GPU state" "$LOG_FILE" | grep ",$GPU_INDEX," | head -1 || grep -A2 "GPU state" "$LOG_FILE" | grep "^${GPU_INDEX}," | head -1 || echo "")
  ACTUAL_POWER=$(echo "$GPU_STATE_LINE" | awk -F', ' '{print $5}' | grep -oE '[0-9]+\.?[0-9]*' | head -1 || echo "?")
  GPU_TEMP=$(echo "$GPU_STATE_LINE" | awk -F', ' '{print $6}' | tr -d ' ' || echo "?")

  # TPS/W efficiency calc (if both numeric)
  if [[ "$NARR_TPS" =~ ^[0-9]+\.[0-9]+$ && "$ACTUAL_POWER" =~ ^[0-9]+\.?[0-9]*$ && "$ACTUAL_POWER" != "0" ]]; then
    EFFICIENCY=$(awk "BEGIN{printf \"%.3f\", $NARR_TPS / $ACTUAL_POWER}")
  else
    EFFICIENCY="?"
  fi

  printf "[result] %sW cap → %s narr / %s code TPS @ %sW actual draw, %s°C, eff %s TPS/W\n\n" \
    "$CAP" "$NARR_TPS" "$CODE_TPS" "$ACTUAL_POWER" "$GPU_TEMP" "$EFFICIENCY"

  printf "| %s | %s | %s | %s | %s | %s |\n" \
    "$CAP" "$NARR_TPS" "$CODE_TPS" "$ACTUAL_POWER" "$GPU_TEMP" "$EFFICIENCY" \
    >> "$RESULTS_FILE"
done

# Reset
if [ "$RESET" -eq 1 ]; then
  echo "[reset] restoring GPU $GPU_INDEX to stock TDP (${STOCK_TDP}W)"
  nvidia-smi -pl "$STOCK_TDP" -i "$GPU_INDEX" 2>&1 | tail -1
else
  echo "[reset] --no-reset specified; GPU $GPU_INDEX left at last cap"
fi

# Append context to results file
{
  echo ""
  echo "**Reset:** $([ $RESET -eq 1 ] && echo "auto-reset to ${STOCK_TDP}W stock" || echo "left at last cap (--no-reset)")"
  echo ""
  echo "**Notes:**"
  echo "- Each row: 3 warm + 5 measured runs of canonical narr (800-word essay) + code (quicksort) prompts."
  echo "- Actual power = mid-bench sample; transient peaks may exceed cap by up to ~10W on some boards."
  echo "- TPS/W efficiency lets you spot the knee — typically the highest cap before efficiency starts dropping."
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
