# Build from repo root: docker build -f runpod/Dockerfile -t club-3090-vllm .
FROM vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08

ARG GENESIS_PIN=917519b

RUN python3 -m pip install --no-cache-dir xxhash pandas scipy

RUN mkdir -p /tmp/genesis && \
    curl -sSL https://github.com/Sandermage/genesis-vllm-patches/archive/${GENESIS_PIN}.tar.gz | \
    tar xz -C /tmp/genesis --strip-components=1 && \
    cp -r /tmp/genesis/vllm/_genesis /usr/local/lib/python3.12/dist-packages/vllm/_genesis && \
    rm -rf /tmp/genesis

RUN curl -sSL -o /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/marlin.py \
    https://raw.githubusercontent.com/noonghunna/vllm/marlin-pad-sub-tile-n/vllm/model_executor/kernels/linear/mixed_precision/marlin.py && \
    curl -sSL -o /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py \
    https://raw.githubusercontent.com/noonghunna/vllm/marlin-pad-sub-tile-n/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py

COPY models/qwen3.6-27b/vllm/patches/patch_tolist_cudagraph.py /patches/patch_tolist_cudagraph.py
COPY runpod/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV VLLM_WORKER_MULTIPROC_METHOD=fork \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_P2P_DISABLE=1 \
    VLLM_NO_USAGE_STATS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
    VLLM_USE_FLASHINFER_SAMPLER=1 \
    OMP_NUM_THREADS=1 \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    VLLM_FLOAT32_MATMUL_PRECISION=high \
    CUDA_DEVICE_MAX_CONNECTIONS=8 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    HF_HOME=/workspace/hf_home \
    MODEL_NAME_OR_PATH=Lorbus/Qwen3.6-27B-int4-AutoRound

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
