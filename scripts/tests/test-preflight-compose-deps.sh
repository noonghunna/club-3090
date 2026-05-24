#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

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

run_deps() {
  local compose="$1"
  local model_dir="$2"
  (
    export ROOT_DIR MODEL_DIR="$model_dir"
    source "${ROOT_DIR}/scripts/preflight.sh"
    preflight_compose_deps "$compose"
  ) 2>&1
}

expect_missing() {
  local compose="$1"
  local model_dir="$2"
  local output

  if output="$(run_deps "$compose" "$model_dir")"; then
    echo "ASSERTION FAILED: expected missing-model failure for $compose" >&2
    echo "--- output ---" >&2
    echo "$output" >&2
    exit 1
  fi
  printf '%s' "$output"
}

ik_compose="${TMP_DIR}/ik.yml"
cat > "$ik_compose" <<'YAML'
services:
  ik:
    image: ${IK_LLAMA_IMAGE:-ghcr.io/ikawrakow/ik-llama-cpp:cu13-server}
    command: >-
      --model /models/${GGUF_FILE:-qwen3.6-27b-gguf/ubergarm-mtp-iq4ks/Qwen3.6-27B-MTP-IQ4_KS.gguf}
YAML

out="$(expect_missing "$ik_compose" "${TMP_DIR}/empty-models")"
assert_contains "$out" "qwen3.6-27b-gguf/ubergarm-mtp-iq4ks/Qwen3.6-27B-MTP-IQ4_KS.gguf"
assert_not_contains "$out" "qwen3.6-27b-autoround-int4"

llama_compose="${TMP_DIR}/llama.yml"
cat > "$llama_compose" <<'YAML'
services:
  llama:
    image: ${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda-b9246}
    command: >-
      -m /models/${GGUF_FILE:-qwen3.6-27b-gguf/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf}
YAML
mkdir -p "${TMP_DIR}/models/qwen3.6-27b-gguf/unsloth-mtp-q4km"
touch "${TMP_DIR}/models/qwen3.6-27b-gguf/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf"
out="$(run_deps "$llama_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]

vllm_compose="${TMP_DIR}/gemma.yml"
cat > "$vllm_compose" <<'YAML'
services:
  vllm:
    image: ghcr.io/noonghunna/vllm-club3090:latest
    command:
      - --model
      - /root/.cache/huggingface/gemma-4-31b-autoround-int4
      - --speculative-config
      - '{"model":"/root/.cache/huggingface/gemma-4-31b-it-assistant","num_speculative_tokens":4}'
YAML
out="$(expect_missing "$vllm_compose" "${TMP_DIR}/empty-models")"
assert_contains "$out" "gemma-4-31b-autoround-int4/config.json"
assert_contains "$out" "gemma-4-31b-it-assistant/config.json"
assert_not_contains "$out" "qwen3.6-27b-autoround-int4"
mkdir -p "${TMP_DIR}/models/gemma-4-31b-autoround-int4" "${TMP_DIR}/models/gemma-4-31b-it-assistant"
touch "${TMP_DIR}/models/gemma-4-31b-autoround-int4/config.json" "${TMP_DIR}/models/gemma-4-31b-it-assistant/config.json"
out="$(run_deps "$vllm_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]


extends_base="${TMP_DIR}/base.yml"
extends_stub="${TMP_DIR}/stub.yml"
cat > "$extends_base" <<'YAML'
services:
  vllm-base:
    image: ghcr.io/noonghunna/vllm-club3090:latest
    command:
      - --model
      - /root/.cache/huggingface/qwen3.6-27b-autoround-int4
YAML
cat > "$extends_stub" <<'YAML'
services:
  vllm-stub:
    extends:
      file: base.yml
      service: vllm-base
YAML
out="$(expect_missing "$extends_stub" "${TMP_DIR}/empty-models")"
assert_contains "$out" "qwen3.6-27b-autoround-int4/config.json"

sglang_compose="${TMP_DIR}/sglang.yml"
cat > "$sglang_compose" <<'YAML'
services:
  sglang:
    image: ghcr.io/sgl-project/sglang:v0.5.12
    volumes:
      - "${MODEL_DIR:?MODEL_DIR is required}/qwen3.6-27b-autoround-int4:/models/target:ro"
      - "${MODEL_DIR}/qwen3.6-27b-prism-eagle3/compressed:/models/drafter:ro"
YAML
out="$(expect_missing "$sglang_compose" "${TMP_DIR}/empty-models")"
assert_contains "$out" "qwen3.6-27b-autoround-int4"
assert_contains "$out" "qwen3.6-27b-prism-eagle3/compressed"
mkdir -p "${TMP_DIR}/models/qwen3.6-27b-autoround-int4" "${TMP_DIR}/models/qwen3.6-27b-prism-eagle3/compressed"
out="$(run_deps "$sglang_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]

echo "test-preflight-compose-deps: ok"
