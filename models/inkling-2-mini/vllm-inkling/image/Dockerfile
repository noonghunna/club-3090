# vllm-inkling-club3090: Inkling-2-Mini on RTX 3090s (sm_86), on vLLM's own Inkling image.
#
# vLLM's `vllm/vllm-openai:inkling-2-mini` refuses anything below Hopper, and its compiled ops carry
# sm_100 code (plus Marlin's sm_80), so on a 3090 they silently return unwritten buffers. This image
# changes two things and nothing else:
#   1. `_C_stable_libtorch` + `_moe_C_stable_libtorch` rebuilt for sm_86 from the vLLM source the
#      image's Python was built from (stage 1);
#   2. a diff to vllm/models/inkling2 (inkling2-sm86.diff: sm_8x takes the existing SM12x / Hopper
#      paths, PDL off, software E4M3 for the FP8 KV cache, and from r2 int8 per-channel weights at load
#      behind VLLM_INKLING_W8, from lulakien/inkling2-mini-3090-recipe), checked against the tested files
#      (inkling2-sm86.sha256).
# A bridge until a vLLM release runs Inkling-2-Mini on sm_86 itself (README.md). Build with build.sh.

# vLLM's Inkling launch image, by digest (vLLM 0.1.dev25855+g7cb5dae22, torch 2.14.0, CUDA 13.0, Python 3.14).
ARG BASE=vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e

# --- stage 1: the two op libraries, for sm_86 (no GPU needed) -------------------------------------
FROM ${BASE} AS ops
# The Inkling PR head the image's Python matches; it is in vllm-project/vllm's network.
ARG VLLM_SRC_COMMIT=fc1443152c0eb655f5c2762f56b540bebe93b196
ARG VLLM_SRC_SHA256=cba822b25ce5e99d37ab11f22d14fa0f6d8a6eaa98234c5ca3613a32f0f7e0c8
ARG MAX_JOBS=16
# sm_86 only: the launch image's other compiled code (FA2 for vision: sm_80 cubins) does not run on
# sm_120 either, so a fat build here would not make a 5090 work. sm_86 cubins also run on sm_89.
ENV TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=${MAX_JOBS} NVCC_THREADS=2
# git: CMake's FetchContent clones the third-party sources (CUTLASS etc.).
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null \
 && rm -rf /var/lib/apt/lists/* \
 && pip install -q "cmake>=3.26.1,<4"
WORKDIR /src
RUN curl -sSL -f -o /tmp/vllm-src.tar.gz "https://codeload.github.com/vllm-project/vllm/tar.gz/${VLLM_SRC_COMMIT}" \
 && echo "${VLLM_SRC_SHA256}  /tmp/vllm-src.tar.gz" | sha256sum -c - \
 && tar xzf /tmp/vllm-src.tar.gz --strip-components=1 -C /src \
 && rm /tmp/vllm-src.tar.gz
# The PR's csrc/inkling2 kernels are CMake-gated to 9.0a / 10.x, so they drop out; the diff routes
# around them on sm_8x.
RUN PY="$(command -v python3)" \
 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
      -DVLLM_PYTHON_EXECUTABLE="$PY" -DVLLM_PYTHON_PATH="$("$PY" -c 'import sys; print(":".join(sys.path))')" \
      -DFETCHCONTENT_BASE_DIR=/src/.deps -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DNVCC_THREADS=2 \
      -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING=compile=${MAX_JOBS} \
 && cmake --build build --target _C_stable_libtorch _moe_C_stable_libtorch -j ${MAX_JOBS} \
 && ls -la build/_C_stable_libtorch.abi3.so build/_moe_C_stable_libtorch.abi3.so

# --- stage 2: the launch image + the two libraries + the diff -------------------------------------
FROM ${BASE}
ARG VLLM_SRC_COMMIT=fc1443152c0eb655f5c2762f56b540bebe93b196
ARG RECIPE_COMMIT=unknown
ARG VLLM_PKG=/usr/local/lib/python3.14/dist-packages/vllm
COPY --from=ops /src/build/_C_stable_libtorch.abi3.so /src/build/_moe_C_stable_libtorch.abi3.so ${VLLM_PKG}/
COPY inkling2-sm86.diff inkling2-sm86.sha256 /opt/club3090/
RUN cd ${VLLM_PKG}/models \
 && patch -p1 --no-backup-if-mismatch --forward < /opt/club3090/inkling2-sm86.diff \
 && sha256sum -c /opt/club3090/inkling2-sm86.sha256
# Every run behind the published numbers had this set; the image carries sm_86 code only anyway.
ENV TORCH_CUDA_ARCH_LIST=8.6
LABEL org.opencontainers.image.title="vllm-inkling-club3090" \
      org.opencontainers.image.description="vLLM's Inkling-2-Mini image with its op libraries rebuilt for sm_86 and an sm_8x diff to vllm/models/inkling2 (club-3090)" \
      org.opencontainers.image.source="https://github.com/noonghunna/club-3090" \
      org.opencontainers.image.revision="${RECIPE_COMMIT}" \
      org.opencontainers.image.base.name="vllm/vllm-openai:inkling-2-mini" \
      org.opencontainers.image.base.digest="sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e" \
      club3090.recipe="models/inkling-2-mini/vllm-inkling/image" \
      club3090.vllm-src-commit="${VLLM_SRC_COMMIT}" \
      club3090.cuda-archs="8.6"
