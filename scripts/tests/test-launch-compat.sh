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
assert_contains "$out" "VLLM_IMAGE=vllm/vllm-openai:v0.22.0"


out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/gemma-int8-mtp --format shell)"
assert_contains "$out" "VLLM_IMAGE=vllm/vllm-openai:v0.22.0"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  out="$(VLLM_NIGHTLY_SHA="$CLEAN_SHA" docker compose -f "$ROOT_DIR/models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml" config 2>/dev/null)"
  assert_contains "$out" "image: vllm/vllm-openai:v0.22.0"

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
assert_contains "$out" "vllm/vllm-openai:v0.22.0"

echo "test-launch-compat: ok"
