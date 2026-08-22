#!/usr/bin/env bash
set -euo pipefail

# Guard: #1042 weight-shard preflight. A checkout whose
# model.safetensors.index.json references shards that aren't on disk (or whose
# numbered -000NN-of-000NN GGUF part-set is incomplete) must REFUSE pre-launch
# naming the absent files + the exact WEIGHT_KEY re-fetch command — instead of
# a 53 KB vLLM traceback with the real cause buried at the bottom.
#
# Asserts:
#   - missing safetensors shard detected, filename + WEIGHT_KEY setup hint printed
#   - complete checkout passes silently
#   - no index → skip cleanly (not an error)
#   - GGUF part-pattern: incomplete set detected; complete set passes
#   - FORCE=1 (--force) bypasses the shard check
#   - PREFLIGHT_NO_SHARD_CHECK=1 skips it too
#
# Fixtures are tmp dirs; no downloads, no docker, no GPU.

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
    echo "ASSERTION FAILED: expected output NOT to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

run_deps() {
  local compose="$1"
  local model_dir="$2"
  shift 2
  (
    export ROOT_DIR MODEL_DIR="$model_dir" "$@"
    source "${ROOT_DIR}/scripts/preflight.sh"
    preflight_compose_deps "$compose"
  ) 2>&1
}

expect_fail() {
  local compose="$1"
  local model_dir="$2"
  shift 2
  local output
  if output="$(run_deps "$compose" "$model_dir" "$@")"; then
    echo "ASSERTION FAILED: expected preflight failure for $compose" >&2
    echo "--- output ---" >&2
    echo "$output" >&2
    exit 1
  fi
  printf '%s' "$output"
}

# --- fixture: vLLM compose mounting one HF subdir ---------------------------
vllm_compose="${TMP_DIR}/vllm.yml"
cat > "$vllm_compose" <<'YAML'
services:
  vllm:
    image: vllm/vllm-openai:nightly
    command:
      - '--model'
      - '/root/.cache/huggingface/qwen3.6-27b-autoround-int4'
YAML

slug="${TMP_DIR}/models/qwen3.6-27b-autoround-int4"
mkdir -p "$slug"
touch "${slug}/config.json"
cat > "${slug}/model.safetensors.index.json" <<'JSON'
{"weight_map": {"l0.weight": "model-00001-of-00002.safetensors",
                "l1.weight": "model-00001-of-00002.safetensors",
                "l2.weight": "model-00002-of-00002.safetensors"}}
JSON

# 1 — a referenced-but-absent shard refuses the launch, names the file, and
#     prints the exact resumable re-fetch command (weights.py recipe join).
out="$(expect_fail "$vllm_compose" "${TMP_DIR}/models")"
assert_contains "$out" "model-00002-of-00002.safetensors"
assert_contains "$out" "download is incomplete"
assert_contains "$out" "WEIGHT_KEY=qwen3.6-27b:autoround-int4 bash scripts/setup.sh qwen3.6-27b"

# 2 — complete checkout passes silently.
touch "${slug}/model-00001-of-00002.safetensors" "${slug}/model-00002-of-00002.safetensors"
out="$(run_deps "$vllm_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]

# 3 — no index → skip cleanly (single-file checkout is NOT an error).
rm "${slug}/model.safetensors.index.json"
out="$(run_deps "$vllm_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]

# 4 — FORCE=1 (--force) bypasses the shard check; restore a broken checkout.
cat > "${slug}/model.safetensors.index.json" <<'JSON'
{"weight_map": {"l0.weight": "model-00001-of-00002.safetensors",
                "l2.weight": "model-00002-of-00002.safetensors"}}
JSON
rm "${slug}/model-00002-of-00002.safetensors"
out="$(run_deps "$vllm_compose" "${TMP_DIR}/models" FORCE=1)"
[[ -z "$out" ]]
out="$(run_deps "$vllm_compose" "${TMP_DIR}/models" PREFLIGHT_NO_SHARD_CHECK=1)"
[[ -z "$out" ]]
out="$(expect_fail "$vllm_compose" "${TMP_DIR}/models")"
assert_contains "$out" "model-00002-of-00002.safetensors"

# --- fixture: llama.cpp GGUF part-pattern (no index) -------------------------
gguf_compose="${TMP_DIR}/gguf.yml"
cat > "$gguf_compose" <<'YAML'
services:
  llama:
    image: ghcr.io/ggml-org/llama.cpp:server
    command:
      - '-m'
      - '/models/qwen3.6-27b-gguf/unsloth-mtp-q4km/shardguard-00001-of-00003.gguf'
YAML

parts="${TMP_DIR}/models/qwen3.6-27b-gguf/unsloth-mtp-q4km"
mkdir -p "$parts"
touch "${parts}/shardguard-00001-of-00003.gguf" "${parts}/shardguard-00003-of-00003.gguf"

# 5 — an incomplete -000NN-of-000NN set names the absent part.
out="$(expect_fail "$gguf_compose" "${TMP_DIR}/models")"
assert_contains "$out" "shardguard-00002-of-00003.gguf"
assert_contains "$out" "missing numbered GGUF part"

# 6 — the complete part-set passes silently.
touch "${parts}/shardguard-00002-of-00003.gguf"
out="$(run_deps "$gguf_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]

# 7 — single-file GGUF (no part pattern, no index) still passes.
rm "${parts}"/shardguard-*.gguf
touch "${parts}/Qwen3.6-27B-Q4_K_M.gguf"
out="$(run_deps "$gguf_compose" "${TMP_DIR}/models" GGUF_FILE="qwen3.6-27b-gguf/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf")"
[[ -z "$out" ]]

echo "test-preflight-shards: ok"
