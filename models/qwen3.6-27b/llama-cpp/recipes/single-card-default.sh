#!/usr/bin/env bash
# Sane mid-context default for Qwen3.6-27B on llama.cpp + 1× RTX 3090.
# 65K ctx, plenty for chat + light agent work. Q4_K_M GGUF.
#
# Prereqs:
#   - llama.cpp built with -DGGML_CUDA=ON at /opt/llama.cpp
#   - Q4_K_M GGUF at /mnt/models/gguf/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf
#     (download via: hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir /mnt/models/gguf/qwen3.6-27b/)
#
# Override defaults via env: LLAMA_DIR, MODEL_PATH, PORT, CTX

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"
MODEL_PATH="${MODEL_PATH:-/mnt/models/gguf/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf}"
PORT="${PORT:-8020}"
CTX="${CTX:-65536}"

if [[ ! -x "${LLAMA_DIR}/build/bin/llama-server" ]]; then
  echo "ERROR: llama-server not found at ${LLAMA_DIR}/build/bin/llama-server"
  echo "Build llama.cpp first: git clone https://github.com/ggerganov/llama.cpp ${LLAMA_DIR} && cd ${LLAMA_DIR} && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: model not found at ${MODEL_PATH}"
  echo "Download: hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir $(dirname "${MODEL_PATH}")"
  exit 1
fi

exec "${LLAMA_DIR}/build/bin/llama-server" \
  -m "${MODEL_PATH}" \
  -c "${CTX}" \
  -ngl 99 \
  -fa on \
  --host 0.0.0.0 --port "${PORT}" \
  --jinja
