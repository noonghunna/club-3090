#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Lorbus/Qwen3.6-27B-int4-AutoRound}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-0}"
INTERACTIVE="${INTERACTIVE:-0}"

export HF_HOME="${HF_HOME:-/workspace/hf_home}"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No GPUs detected by nvidia-smi"
    exit 1
fi
echo "=== club-3090 RunPod entrypoint ==="
echo "  GPUs detected: ${GPU_COUNT}"
echo "  Model:         ${MODEL_NAME_OR_PATH}"
echo "  HF_HOME:       ${HF_HOME}"
echo "  NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset}"

if [ "$TENSOR_PARALLEL_SIZE" -eq 0 ]; then
    TENSOR_PARALLEL_SIZE="$GPU_COUNT"
fi
if [ "$TENSOR_PARALLEL_SIZE" -gt "$GPU_COUNT" ]; then
    echo "WARNING: TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} > GPU_COUNT=${GPU_COUNT}, capping"
    TENSOR_PARALLEL_SIZE="$GPU_COUNT"
fi

if [ "$INTERACTIVE" = "1" ]; then
    echo "=== INTERACTIVE mode — launching jupyter lab on port 8080 ==="
    pip install -q jupyterlab notebook 2>/dev/null
    cat > /tmp/jupyter_config.py << 'PYEOF'
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8080
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_origin = "*"
c.ServerApp.disable_check_xsrf = True
c.ServerApp.trust_xheaders = True
c.ServerApp.token = ""
c.ServerApp.password = ""
c.ServerApp.root_dir = "/workspace"
PYEOF
    exec jupyter lab --config=/tmp/jupyter_config.py
fi

VLLM_ARGS=(
    --model "${MODEL_NAME_OR_PATH}"
    --served-model-name qwen3.6-27b-autoround
    --quantization auto_round
    --dtype float16
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --trust-remote-code
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --enable-prefix-caching
    --enable-chunked-prefill
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    --host 0.0.0.0
    --port "${VLLM_PORT}"
)

if [ "$TENSOR_PARALLEL_SIZE" -ge 2 ]; then
    echo "=== Dual GPU — dual.yml path (262K, fp8 KV, vision + tools, Genesis-less) ==="
    VLLM_ARGS+=(
        --disable-custom-all-reduce
        --max-model-len "${MAX_MODEL_LEN:-262144}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}"
        --max-num-seqs "${MAX_NUM_SEQS:-2}"
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
        --kv-cache-dtype fp8_e5m2
    )
else
    echo "=== Single GPU — tools-text path (75K, fp8 KV, text-only, Genesis P64+PN8) ==="
    python3 -m vllm._genesis.patches.apply_all 2>/dev/null || echo "  Genesis apply_all skipped"
    python3 /patches/patch_tolist_cudagraph.py 2>/dev/null || true
    export GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1
    export GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1
    VLLM_ARGS+=(
        --max-model-len "${MAX_MODEL_LEN:-75000}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
        --max-num-seqs "${MAX_NUM_SEQS:-1}"
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
        --kv-cache-dtype fp8_e5m2
        --language-model-only
    )
fi

echo "  TP=${TENSOR_PARALLEL_SIZE}"
echo "  max_model_len=${MAX_MODEL_LEN:-default}  mem_util=${GPU_MEMORY_UTILIZATION:-default}  max_seqs=${MAX_NUM_SEQS:-default}"
echo ""

exec vllm serve "${VLLM_ARGS[@]}"
