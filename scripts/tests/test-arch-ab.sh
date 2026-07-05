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

# --- dual auto-pick + the arms that RUN on 2x consumer Blackwell --------------
# (nvfp4 is excluded here — sm_120 has no nvfp4-KV FMHA kernel; covered below.)
out="$(CLUB3090_FAKE_GPUS="$GPU_5090x2" bash scripts/arch-ab.sh --arms e5m2,e4m3,fp8w --dry-run)"
assert_contains "$out" "variant=vllm/dual"
assert_contains "$out" "arm e4m3: vllm/dual + KV_CACHE_DTYPE=fp8_e4m3"
assert_contains "$out" "arm fp8w: vllm/qwen-27b-dual-max STOCK"

# --- refusals (fail-loud, each names the fix) ---------------------------------
# default arms include e4m3 -> a bare run on an Ampere rig must refuse with the
# "no arch delta to measure" explanation, NOT hang in the boot ready-wait
if out="$(CLUB3090_FAKE_GPUS="$GPU_3090" bash scripts/arch-ab.sh --dry-run 2>&1)"; then
  fail "default arms on sm_8.6 must refuse (e4m3 boot-rejects on Ampere)"
fi
assert_contains "$out" "no arch delta to measure"
# ...but the explicit control-only run stays possible on Ampere
out="$(CLUB3090_FAKE_GPUS="$GPU_3090" bash scripts/arch-ab.sh --arms e5m2 --dry-run)"
assert_contains "$out" "arm e5m2"
# nvfp4 needs DATACENTER Blackwell (sm_100/103) — refuse on Ampere AND on
# CONSUMER Blackwell (sm_120), which is a higher number but has no FMHA kernel
# (the disc #571 crash: a plain ">=10" wrongly passed sm_120).
if out="$(CLUB3090_FAKE_GPUS="$GPU_3090" bash scripts/arch-ab.sh --arms e5m2,nvfp4 --dry-run 2>&1)"; then
  fail "nvfp4 on sm_8.6 must refuse"
fi
assert_contains "$out" "DATACENTER Blackwell"
GPU_5090="0:NVIDIA_GeForce_RTX_5090:32607:12.0"
if out="$(CLUB3090_FAKE_GPUS="$GPU_5090" bash scripts/arch-ab.sh --arms e5m2,nvfp4 --dry-run 2>&1)"; then
  fail "nvfp4 on consumer sm_12.0 must refuse (no FMHA kernel)"
fi
assert_contains "$out" "DATACENTER Blackwell"
# ...but datacenter Blackwell (sm_100) is allowed
out="$(CLUB3090_FAKE_GPUS="0:NVIDIA_B200:186000:10.0" bash scripts/arch-ab.sh --arms nvfp4 --dry-run 2>&1)"
assert_contains "$out" "arm nvfp4"
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
