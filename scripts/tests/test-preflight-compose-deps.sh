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

run_deps_unset_model_dir() {
  local compose="$1"
  (
    unset MODEL_DIR
    export ROOT_DIR
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
    image: ${IK_LLAMA_IMAGE:-ghcr.io/ikawrakow/ik-llama-cpp@sha256:5f914f1ccade922417af58c94bd1cbb558052c8852d86678ead3fe693eec0143}
    command: >-
      --model /models/${GGUF_FILE:-qwen3.6-27b-gguf/ubergarm-mtp-iq4ks/Qwen3.6-27B-MTP-IQ4_KS.gguf}
YAML

out="$(expect_missing "$ik_compose" "${TMP_DIR}/empty-models")"
assert_contains "$out" "qwen3.6-27b-gguf/ubergarm-mtp-iq4ks/Qwen3.6-27B-MTP-IQ4_KS.gguf"
assert_contains "$out" "hf download ubergarm/Qwen3.6-27B-GGUF Qwen3.6-27B-MTP-IQ4_KS.gguf"
assert_contains "$out" "WEIGHTS=iq4ks bash scripts/setup.sh qwen3.6-27b"
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
    image: vllm/vllm-openai:v0.21.0
    command:
      - --model
      - /root/.cache/huggingface/gemma-4-31b-autoround-int4
      - --speculative-config
      - '{"model":"/root/.cache/huggingface/gemma-4-31b-it-assistant","num_speculative_tokens":4}'
YAML
out="$(expect_missing "$vllm_compose" "${TMP_DIR}/empty-models")"
assert_contains "$out" "gemma-4-31b-autoround-int4/config.json"
assert_contains "$out" "gemma-4-31b-it-assistant/config.json"
assert_contains "$out" "hf download Intel/gemma-4-31B-it-int4-AutoRound"
assert_contains "$out" "hf download google/gemma-4-31B-it-assistant"
assert_contains "$out" "bash scripts/setup.sh gemma-4-31b"
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
    image: vllm/vllm-openai:v0.21.0
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

if out="$(run_deps_unset_model_dir "$extends_stub")"; then
  echo "ASSERTION FAILED: expected missing-model failure with unset MODEL_DIR" >&2
  echo "--- output ---" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "MODEL_DIR not set"
assert_contains "$out" "defaulting to ${ROOT_DIR}/models-cache"

out="$(CLUB3090_WEIGHTS_READER_DISABLE=1 expect_missing "$extends_stub" "${TMP_DIR}/empty-models")"
assert_contains "$out" "qwen3.6-27b-autoround-int4/config.json"
assert_contains "$out" "Check the compose header for its model-specific hf download command."
assert_not_contains "$out" "hf download Lorbus/Qwen3.6-27B-int4-AutoRound"

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
assert_contains "$out" "hf download Ex0bit/Qwen3.6-27B-PRISM-EAGLE3"
assert_contains "$out" "WITH_PRISM_EAGLE3=1 bash scripts/setup.sh qwen3.6-27b"
mkdir -p "${TMP_DIR}/models/qwen3.6-27b-autoround-int4" "${TMP_DIR}/models/qwen3.6-27b-prism-eagle3/compressed"
out="$(run_deps "$sglang_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]

# beellama.cpp DFlash: the image is NOT ggml-org/ik-llama (so it must still be
# detected as a llama.cpp-family GGUF server), and the drafter is named via
# --spec-draft-model (not -m). A present target + MISSING drafter must refuse
# with the drafter path + its hf-download hint — the #288 George report, where
# a missing drafter otherwise crashed cryptically in-container.
beellama_compose="${TMP_DIR}/beellama.yml"
cat > "$beellama_compose" <<'YAML'
services:
  beellama:
    image: ${BEELLAMA_IMAGE:-ghcr.io/anbeeld/beellama.cpp:server-cuda-v0.3.0-e0663be2713c}
    command: >-
      -m /models/${GGUF_FILE:-qwen3.6-27b-gguf/unsloth-q5ks/Qwen3.6-27B-Q5_K_S.gguf}
      --spec-draft-model /models/${DRAFT_FILE:-qwen3.6-27b-gguf/anbeeld-dflash-iq4xs/Qwen3.6-27B-DFlash-IQ4_XS.gguf}
YAML
mkdir -p "${TMP_DIR}/bl-models/qwen3.6-27b-gguf/unsloth-q5ks"
touch "${TMP_DIR}/bl-models/qwen3.6-27b-gguf/unsloth-q5ks/Qwen3.6-27B-Q5_K_S.gguf"
out="$(expect_missing "$beellama_compose" "${TMP_DIR}/bl-models")"
assert_contains "$out" "qwen3.6-27b-gguf/anbeeld-dflash-iq4xs/Qwen3.6-27B-DFlash-IQ4_XS.gguf"
assert_contains "$out" "speculative drafter GGUF"
assert_contains "$out" "hf download Anbeeld/Qwen3.6-27B-DFlash-GGUF Qwen3.6-27B-DFlash-IQ4_XS.gguf"
# the target is present, so it must NOT be reported missing
assert_not_contains "$out" "unsloth-q5ks/Qwen3.6-27B-Q5_K_S.gguf (llama.cpp GGUF weights)"
# add the drafter → must pass
mkdir -p "${TMP_DIR}/bl-models/qwen3.6-27b-gguf/anbeeld-dflash-iq4xs"
touch "${TMP_DIR}/bl-models/qwen3.6-27b-gguf/anbeeld-dflash-iq4xs/Qwen3.6-27B-DFlash-IQ4_XS.gguf"
out="$(run_deps "$beellama_compose" "${TMP_DIR}/bl-models")"
[[ -z "$out" ]]

echo "test-preflight-compose-deps: ok"
