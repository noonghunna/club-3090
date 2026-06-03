#!/usr/bin/env bash
#
# rebench-runtime.sh — the "runtime" tier of rebench: verify-full (preflight)
# + bench + verify-stress + soak, skipping the two generation-quality legs
# (quality-full think-OFF + quality-thinking think-ON).
#
# WHEN TO USE THIS instead of rebench-full.sh:
#   When you changed a serving-config knob that affects throughput / VRAM /
#   stability but NOT per-token generation quality, so re-scoring the 8-pack
#   twice (think-OFF + think-ON) would be ~1.5-2.5 hr of wasted cycles that can
#   only reproduce the prior scores. Examples:
#     - context-size ceiling bump (KV *type* unchanged)
#     - --ubatch-size / --batch-size tuning
#     - --gpu-memory-utilization / --mem-fraction-static
#     - power-limit sweep
#     - MTP n / draft-p-min retune (TPS + accept-rate, not answer text)
#   If the change DOES touch generation (quant swap, KV-cache *type* change,
#   chat-template change, model swap) → run full rebench-full.sh instead.
#
# Preflight + three legs (≈30-40 min on single-card, vs ~2.5-3.5 hr for the
# full matrix):
#   0. verify-full.sh     — functional preflight, FAIL-FAST (inherited from
#                           rebench-full; NOT skipped — a runtime knob can break
#                           serving, so we still gate on it)
#   1. bench.sh           — TPS narrative + code
#   2. verify-stress.sh   — long-context needle ladder + prefill-OOM boundary
#   3. soak-test.sh       — accumulating-context endurance (Cliff 2b)
#
# Usage (identical surface to rebench-full.sh — all flags pass through):
#   bash scripts/rebench-runtime.sh                       # auto-detect running compose
#   bash scripts/rebench-runtime.sh --tag ik-262k         # explicit tag
#   bash scripts/rebench-runtime.sh --skip soak           # ALSO skip soak (merged)
#
# llama.cpp / ik_llama (non-vllm container names) — soak-test.sh's container
# auto-detect only matches vllm-*; pass the endpoint + container explicitly
# (see #403):
#   CONTAINER=ik-llama-qwen36-27b URL=http://localhost:8020 MODEL=ik-iq4ks-mtp \
#     bash scripts/rebench-runtime.sh --engine llama-cpp
#
# This is a thin preset over rebench-full.sh: it injects
# `--skip quality-full,quality-thinking` and merges any --skip you pass.
# (verify-full is intentionally NOT skipped — it's the fail-fast preflight.)

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_SKIP="quality-full,quality-thinking"
USER_SKIP=""
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip) USER_SKIP="${2:-}"; shift 2 ;;
    *)      PASS_ARGS+=("$1"); shift ;;
  esac
done

SKIP="$BASE_SKIP${USER_SKIP:+,$USER_SKIP}"

echo "[rebench-runtime] runtime tier (verify-full + bench + verify-stress + soak); --skip=$SKIP"
exec bash "$ROOT_DIR/scripts/rebench-full.sh" --skip "$SKIP" "${PASS_ARGS[@]}"
