#!/usr/bin/env bash
# Full 262K context for Qwen3.6-27B on a single RTX 3090 via llama.cpp.
# This is the standout recipe — vLLM single-card caps at 48K safe (192K opt-in
# with caveats); llama.cpp + Q4_K_M + q4_0 KV reaches the model's natural max.
#
# Memory math:
#   Model (Q4_K_M):           ~16 GB
#   KV at 262K (q4_0 K + V):  ~5 GB
#   Total:                    ~21 GB
#   Headroom:                 ~3 GB for activation peaks + prompts
#
# Sustained throughput at 262K: 35-45 tok/s on a stock 3090 (community-reported
# flat curve at any in-budget context, including 4K vs 262K).
#
# The KV cache type (--cache-type-k q4_0 --cache-type-v q4_0) is THE unlock.
# fp16 KV doesn't fit at all. q8 KV fits at ~23 GB but runs ~3× slower per
# community reports. q4_0 fits at full speed.
#
# Prereqs:
#   - llama.cpp built with -DGGML_CUDA=ON at /opt/llama.cpp
#   - Q4_K_M GGUF at /mnt/models/gguf/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf
#
# Override defaults via env: LLAMA_DIR, MODEL_PATH, PORT, CTX, KV_TYPE

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"
MODEL_PATH="${MODEL_PATH:-/mnt/models/gguf/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf}"
PORT="${PORT:-8020}"
CTX="${CTX:-262144}"
KV_TYPE="${KV_TYPE:-q4_0}"

if [[ ! -x "${LLAMA_DIR}/build/bin/llama-server" ]]; then
  echo "ERROR: llama-server not found at ${LLAMA_DIR}/build/bin/llama-server"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: model not found at ${MODEL_PATH}"
  echo "Download: hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir $(dirname "${MODEL_PATH}")"
  exit 1
fi

exec "${LLAMA_DIR}/build/bin/llama-server" \
  -m "${MODEL_PATH}" \
  -ngl 99 \
  -c "${CTX}" \
  -np 1 \
  -fa on \
  --cache-type-k "${KV_TYPE}" --cache-type-v "${KV_TYPE}" \
  --host 0.0.0.0 --port "${PORT}" \
  --jinja
