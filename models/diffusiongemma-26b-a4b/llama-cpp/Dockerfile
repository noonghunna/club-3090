# ===========================================================================
# Custom image for DiffusionGemma-26B-A4B on llama.cpp (single 3090, OpenAI shim).
#
# WHY a custom image (not a volume-mount like our vLLM paths): the diffusion arch
# only exists in llama.cpp DRAFT PR #24423 and there is no stock image that carries
# `llama-diffusion-cli`. So we BUILD it — from the existing draft-PR branch, with our
# DG_NDJSON patch applied — and bundle the FastAPI OpenAI shim. See docs/UPSTREAM.md
# (PR #24423 row) and patches/README.md.
#
# Build context = the llama-cpp/ dir (so COPY can reach patches/ and serve/):
#   docker build -t club3090/diffusiongemma-shim:pr24423-c84e85a \
#     -f models/diffusiongemma-26b-a4b/llama-cpp/Dockerfile \
#        models/diffusiongemma-26b-a4b/llama-cpp
# Or just `docker compose -f compose/single/q4-k-m/openai-shim.yml build`.
# ===========================================================================

# --- build stage: compile the patched llama-diffusion-cli from the PR branch -------
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS build

# The "existing branch" we rely on: danielhanchen's DiffusionGemma draft PR #24423.
ARG DG_REPO=https://github.com/danielhanchen/llama.cpp.git
ARG DG_SHA=c84e85af61011f9fbfcf41479381d5ed1661a564
# RTX 3090 = Ampere SM 8.6. Override for other cards (89=4090, 120=5090).
ARG CUDA_ARCH=86

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY patches/diffusion-ndjson-server.patch /tmp/diffusion-ndjson-server.patch
RUN git init -q llama.cpp && cd llama.cpp && \
    git remote add origin "${DG_REPO}" && \
    git fetch -q --depth 1 origin "${DG_SHA}" && \
    git checkout -q FETCH_HEAD && \
    git apply /tmp/diffusion-ndjson-server.patch && \
    cmake -B build -G Ninja \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j --target llama-diffusion-cli

# --- runtime stage: CUDA runtime + the shim ---------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/llama.cpp/build/bin/llama-diffusion-cli /app/llama-diffusion-cli
COPY serve/ /app/serve/
RUN pip3 install --no-cache-dir -r /app/serve/requirements.txt

ENV DG_CLI_BIN=/app/llama-diffusion-cli \
    DG_GGUF=/models/diffusiongemma-26b-a4b-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
    DG_PORT=8060 DG_NGL=99 DG_CTX=3072 DG_MAXGEN=1024 \
    DG_MODEL_ID=diffusiongemma-26b-a4b

EXPOSE 8060
ENTRYPOINT ["python3", "/app/serve/diffusion_openai_server.py"]
