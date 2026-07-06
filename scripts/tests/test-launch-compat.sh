#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="${ROOT_DIR}/scripts/lib/profiles/launch_compat.py"
GPU_3090='0|RTX_3090|24576|8.6'
MTP_SHA="01d4d1ad375dc5854779c593eee093bcebb0cada"
CLEAN_SHA="bf610c2f56764e1b30bc6065f4ceace3d6e59036"
DFLASH_SHA="e47c98ef7a38792996e452ef53914e21e41928e9"

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" == *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output not to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

out="$(python3 "$HELPER" filter-candidates \
  --variants vllm/dual,vllm/minimal,llamacpp/default \
  --model qwen3.6-27b \
  --gpu-spec "$GPU_3090" \
  --tp 1 \
  --pp 1 \
  --workload fast-chat)"
assert_contains "$out" "vllm/minimal"
assert_not_contains "$out" "vllm/dual"

out="$(python3 "$HELPER" filter-candidates \
  --variants vllm/minimal,llamacpp/default,llamacpp/mtp \
  --model qwen3.6-27b \
  --gpu-spec "$GPU_3090" \
  --tp 1 \
  --pp 1 \
  --stable)"
assert_contains "$out" "vllm/minimal"
assert_contains "$out" "llamacpp/default"
assert_contains "$out" "llamacpp/mtp"

if out="$(python3 "$HELPER" validate-variant \
  --variant vllm/gemma-mtp-tp1 \
  --gpu-spec "$GPU_3090" \
  --tp 2 \
  --pp 1 \
  --no-project-vram 2>&1)"; then
  echo "ASSERTION FAILED: invalid Gemma single-card profile unexpectedly passed" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "C1: tp=2 * pp=1 = 2 != 1 cards selected"
assert_contains "$out" "C5: kv_format=fp8_e4m3 not supported by hardware: rtx-3090"

# fp8_e4m3 KV is WEIGHTS-CONDITIONAL on Ampere (sm_86): fp8-weights checkpoints route to
# FlashInfer (native fp8 storage) and ARE supported; non-fp8 weights (Gemma W4A16, above)
# route to Triton (needs SM89+) and are NOT. Validated #594 (Qwen fp8 dual-max on 2x 3090:
# boot/decode/NIAH/quality/soak green) + learnings/gemma-4-31b.md 2026-07-01 (gemma fp8_e4m3
# fails at KV-init on the same stack). Prove the ALLOW direction so the two stay in sync.
GPU_3090_X2="0|RTX_3090|24576|8.6;1|RTX_3090|24576|8.6"
out="$(python3 "$HELPER" validate-variant \
  --variant vllm/qwen-27b-dual-max \
  --gpu-spec "$GPU_3090_X2" \
  --tp 2 \
  --pp 1 \
  --no-project-vram 2>&1)" || {
  echo "ASSERTION FAILED: Qwen fp8-weights dual-max (fp8_e4m3 KV) rejected on 2x rtx-3090" >&2
  echo "$out" >&2
  exit 1
}
assert_not_contains "$out" "kv_format=fp8_e4m3 not supported"

out="$(python3 "$HELPER" validate-variant \
  --variant vllm/minimal \
  --gpu-spec "$GPU_3090" \
  --tp 1 \
  --pp 1 \
  --no-project-vram \
  --verbose 2>&1)"
assert_contains "$out" "Pass 1 fits()"
assert_contains "$out" "Resolved compose: vllm/minimal"
assert_contains "$out" "Pass 2 fits()"

out="$(python3 "$HELPER" resolve-engine-pin --engine-id vllm-nightly-mtp --format shell)"
assert_contains "$out" "VLLM_NIGHTLY_SHA=${MTP_SHA}"

if out="$(python3 "$HELPER" resolve-engine-pin --engine-id vllm-pip-baseline --format shell 2>&1)"; then
  echo "ASSERTION FAILED: pip-only vllm-pip-baseline unexpectedly resolved as a docker nightly" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "install.spec is not a docker image"

out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell)"
assert_contains "$out" "VLLM_IMAGE=vllm/vllm-openai:v0.24.0"


out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/gemma-int8-mtp --format shell)"
assert_contains "$out" "VLLM_IMAGE=vllm/vllm-openai:v0.22.0"

# --- #246 arch-aware KV injection matrix (resolve-variant-pin --gpu-spec) ----
GPU_4090='0|NVIDIA GeForce RTX 4090|24564|8.9'
GPU_5090X2='0|NVIDIA GeForce RTX 5090|32607|12.0;1|NVIDIA GeForce RTX 5090|32607|12.0'

# ampere -> NOTHING injected (compose defaults; the no-op is data equality)
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_3090")"
assert_not_contains "$out" "KV_CACHE_DTYPE"
# ada / blackwell pilot slugs -> native-fp8 swap
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_4090")"
assert_contains "$out" "KV_CACHE_DTYPE=fp8_e4m3"
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/minimal --format shell --gpu-spec "$GPU_5090X2")"
assert_contains "$out" "KV_CACHE_DTYPE=fp8_e4m3"
# non-pilot slug (same kv_format) -> no injection until the #246 A/B expands the pilot
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/qwen-27b-dual-fast --format shell --gpu-spec "$GPU_4090")"
assert_not_contains "$out" "KV_CACHE_DTYPE"
# quant-specific KV slug -> never overridden (compressed-tensors reject fp8 KV)
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/gemma-int8-mtp --format shell --gpu-spec "$GPU_4090")"
assert_not_contains "$out" "KV_CACHE_DTYPE"
# explicit user env pin wins
out="$(KV_CACHE_DTYPE=fp8_e5m2 python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_4090")"
assert_not_contains "$out" "KV_CACHE_DTYPE"
# heterogeneous rig -> no single right answer -> no injection
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "${GPU_3090};1|NVIDIA GeForce RTX 4090|24564|8.9")"
assert_not_contains "$out" "KV_CACHE_DTYPE"
# unmapped card -> degrade to compose defaults, never an error
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "0|Weird GPU|8192|7.0")"
assert_contains "$out" "VLLM_IMAGE=vllm/vllm-openai:v0.24.0"
assert_not_contains "$out" "KV_CACHE_DTYPE"
echo "  ok: #246 arch-aware KV injection matrix (8 cases)"

# --- detector: Blackwell family must not collapse to rtx-5090 (#576 wrinkle) --
# The sm>=12 bucket used to map every Blackwell to rtx-5090, so a 96 GB PRO 6000
# and a 128 GB GB10 both mis-detected as a 32 GB 5090. Lock the split.
det="$(python3 - <<'PY'
import sys; sys.path.insert(0, "scripts/lib/profiles")
from launch_compat import _hardware_id_from_gpu as m
cases = [
    ("NVIDIA RTX PRO 6000 Blackwell", 98304, 12.0, "rtx-6000-pro-blackwell"),
    ("Unnamed Blackwell 96GB",        98304, 12.0, "rtx-6000-pro-blackwell"),  # alias-miss fallback
    ("NVIDIA GB10",                  131072, 12.1, "dgx-spark"),
    ("NVIDIA GeForce RTX 5090",       32607, 12.0, "rtx-5090"),
    ("NVIDIA RTX 6000 Ada Generation",49140,  8.9, "rtx-4090"),                # 'pro 6000' must NOT catch Ada
]
bad = [f"{n}->{m(n,v,s)} want {e}" for n, v, s, e in cases if m(n, v, s) != e]
print("FAIL: " + " | ".join(bad) if bad else "OK")
PY
)"
[[ "$det" == "OK" ]] || { echo "  FAIL: detector: $det"; exit 1; }
echo "  ok: Blackwell detector split (PRO 6000 / GB10 / 5090 / Ada — 5 cases)"

# --- #246 Phase 2 mem-fraction floor (DOWNWARD only) --------------------------
# A unified-memory card (Spark, mem_util_safe 0.85) can't safely give the 0.92
# compose default -> inject the floor. Discrete cards (0.95/0.96 > 0.92) are
# never raised (that touches Cliff margin -> validated opt-in, not automatic).
GPU_SPARK='0|NVIDIA GB10|131072|12.1'
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_SPARK")"
assert_contains "$out" "GPU_MEMORY_UTILIZATION=0.85"
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_5090X2")"
assert_not_contains "$out" "GPU_MEMORY_UTILIZATION"
# heterogeneous: the lowest ceiling (Spark 0.85) forces the whole rig down
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "${GPU_SPARK};1|NVIDIA GeForce RTX 5090|32607|12.0")"
assert_contains "$out" "GPU_MEMORY_UTILIZATION=0.85"
# explicit user pin wins
out="$(GPU_MEMORY_UTILIZATION=0.7 python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_SPARK")"
assert_not_contains "$out" "GPU_MEMORY_UTILIZATION=0.85"
echo "  ok: #246 mem-fraction floor (Spark down · discrete no-raise · het-min · user-pin — 4 cases)"

# --- fp8-weights DeepGEMM disable on consumer cards (disc #571) ----------------
# DeepGEMM has no recipe on consumer Blackwell (sm_120/121, hard-fails) and is
# unused on Ada (sm_89, harmless no-op) -> disable for fp8-weights slugs. Hopper
# (sm_90) keeps it; non-fp8 slugs are untouched.
DMAX=vllm/qwen-27b-dual-max
GPU_H100X2='0|NVIDIA H100|81920|9.0;1|NVIDIA H100|81920|9.0'
out="$(python3 "$HELPER" resolve-variant-pin --variant "$DMAX" --format shell --gpu-spec "$GPU_5090X2")"
assert_contains "$out" "VLLM_USE_DEEP_GEMM=0"
out="$(python3 "$HELPER" resolve-variant-pin --variant "$DMAX" --format shell --gpu-spec "${GPU_4090};1|NVIDIA GeForce RTX 4090|24564|8.9")"
assert_contains "$out" "VLLM_USE_DEEP_GEMM=0"          # Ada proactively covered
out="$(python3 "$HELPER" resolve-variant-pin --variant "$DMAX" --format shell --gpu-spec "$GPU_H100X2")"
assert_not_contains "$out" "VLLM_USE_DEEP_GEMM"        # Hopper keeps the fast path
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_5090X2")"
assert_not_contains "$out" "VLLM_USE_DEEP_GEMM"        # int4-weights slug: not the DeepGEMM path
# fp8-DYNAMIC (compressed-tensors) is fp8-family too — agents-a1 must also disable
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/agents-a1-dual --format shell --gpu-spec "$GPU_5090X2")"
assert_contains "$out" "VLLM_USE_DEEP_GEMM=0"          # fp8-dynamic weights route FP8 GEMM
out="$(VLLM_USE_DEEP_GEMM=1 python3 "$HELPER" resolve-variant-pin --variant "$DMAX" --format shell --gpu-spec "$GPU_5090X2")"
assert_not_contains "$out" "VLLM_USE_DEEP_GEMM=0"      # explicit user pin wins
echo "  ok: fp8w DeepGEMM disable (5090/Ada down · Hopper keep · non-fp8 skip · fp8-dynamic · user-pin — 6 cases)"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  out="$(VLLM_NIGHTLY_SHA="$CLEAN_SHA" docker compose -f "$ROOT_DIR/models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml" config 2>/dev/null)"
  assert_contains "$out" "image: vllm/vllm-openai:v0.24.0"

  out="$(VLLM_NIGHTLY_SHA="$CLEAN_SHA" VLLM_IMAGE=vllm/vllm-openai:latest docker compose -f "$ROOT_DIR/models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml" config 2>/dev/null)"
  assert_contains "$out" "image: vllm/vllm-openai:latest"
fi

out="$(python3 - <<'PY'
from scripts.lib.profiles.compat import InstanceSpec
from scripts.lib.profiles.estate_cli import compose_env

clean = compose_env(InstanceSpec(name="qwen", compose_name="vllm/dual", gpu_indices=(0, 1), port=8010))
gemma = compose_env(InstanceSpec(name="gemma", compose_name="vllm/gemma-int8-mtp", gpu_indices=(0, 1), port=8032))
print(clean["VLLM_IMAGE"])
print(gemma["VLLM_IMAGE"])
PY
)"
assert_contains "$out" "vllm/vllm-openai:v0.24.0"   # clean (vllm/dual → vllm-stable) bumped to v0.24.0
assert_contains "$out" "vllm/vllm-openai:v0.22.0"   # gemma (vllm/gemma-int8-mtp → vllm-gemma-stable) stays v0.22.0

echo "test-launch-compat: ok"
