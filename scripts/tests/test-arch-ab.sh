#!/usr/bin/env bash
# test-arch-ab — plan/refusal contract of scripts/arch-ab.sh (issue #246).
# Hermetic: GPUs faked via CLUB3090_FAKE_GPUS; only --dry-run/--help paths run
# (they exit before any docker/switch/rebench side effect).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() { echo "FAIL: $1" >&2; exit 1; }
assert_contains() { [[ "$1" == *"$2"* ]] || { echo "FAIL: missing '$2' in:" >&2; echo "$1" >&2; exit 1; }; }

GPU_4090="0:NVIDIA_GeForce_RTX_4090:24564:8.9"
GPU_3090="0:NVIDIA_GeForce_RTX_3090:24576:8.6"
GPU_5090x2="0:NVIDIA_GeForce_RTX_5090:32607:12.0,1:NVIDIA_GeForce_RTX_5090:32607:12.0"

# --- help exits 0 and names the arms -----------------------------------------
out="$(bash scripts/arch-ab.sh --help)" || fail "--help nonzero"
assert_contains "$out" "e5m2"
assert_contains "$out" "fp8w"

# --- single-card auto-pick + default arms ------------------------------------
out="$(CLUB3090_FAKE_GPUS="$GPU_4090" bash scripts/arch-ab.sh --dry-run)"
assert_contains "$out" "variant=vllm/minimal"
assert_contains "$out" "arm e5m2: vllm/minimal + KV_CACHE_DTYPE=fp8_e5m2"
assert_contains "$out" "arm e4m3: vllm/minimal + KV_CACHE_DTYPE=fp8_e4m3"
assert_contains "$out" "dry run — nothing executed"

# --- dual auto-pick + all four arms on 2x blackwell ---------------------------
out="$(CLUB3090_FAKE_GPUS="$GPU_5090x2" bash scripts/arch-ab.sh --arms e5m2,e4m3,nvfp4,fp8w --dry-run)"
assert_contains "$out" "variant=vllm/dual"
assert_contains "$out" "arm nvfp4: vllm/dual + KV_CACHE_DTYPE=nvfp4"
assert_contains "$out" "arm fp8w: vllm/qwen-27b-dual-max STOCK"

# --- refusals (fail-loud, each names the fix) ---------------------------------
if out="$(CLUB3090_FAKE_GPUS="$GPU_3090" bash scripts/arch-ab.sh --arms e5m2,nvfp4 --dry-run 2>&1)"; then
  fail "nvfp4 on sm_8.6 must refuse"
fi
assert_contains "$out" "needs Blackwell"
if out="$(CLUB3090_FAKE_GPUS="$GPU_4090" bash scripts/arch-ab.sh --arms fp8w --dry-run 2>&1)"; then
  fail "fp8w on 1 GPU must refuse"
fi
assert_contains "$out" "needs 2 GPUs"
if out="$(CLUB3090_FAKE_GPUS="$GPU_4090" bash scripts/arch-ab.sh --variant vllm/dual --dry-run 2>&1)"; then
  fail "vllm/dual on 1 GPU must refuse"
fi
if out="$(CLUB3090_FAKE_GPUS="$GPU_4090" bash scripts/arch-ab.sh --arms bogus --dry-run 2>&1)"; then
  fail "unknown arm must refuse"
fi
assert_contains "$out" "unknown arm"

echo "test-arch-ab: ok"
