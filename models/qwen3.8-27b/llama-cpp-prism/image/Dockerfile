# llama-cpp-prism — wraps PrismML-Eng/llama.cpp's OWN Linux CUDA 12.8 release tarball.
# Nothing is compiled or patched here: the engine is upstream's binary, run in the CUDA
# runtime it was built against. (Recovered from the b10685 image's layers and checked in
# 2026-09-24, so the next bump does not have to reverse-engineer it.)
#
# Build (from this directory):
#   TAG=prism-b10735-842b188
#   curl -sSLO https://github.com/PrismML-Eng/llama.cpp/releases/download/$TAG/llama-$TAG-bin-linux-cuda-12.8-x64.tar.gz
#   sha256sum llama-$TAG-bin-linux-cuda-12.8-x64.tar.gz   # must equal the release asset's published digest
#   mkdir prism && tar xzf llama-$TAG-*.tar.gz -C prism --strip-components=1 llama-$TAG
#   docker build -t ghcr.io/noonghunna/llamacpp-prism:${TAG#prism-} .
#
# ⚠️ Pick a tag whose asset list HAS `linux-cuda-12.8-x64` — some releases ship Windows only
#    (prism-b10687-5d80cff did).
# ⚠️ CUDA 12.8 deliberately, matching our other serving images; the container is the runtime.
# ⚠️ libgomp1 is REQUIRED: the runtime base lacks it and every binary fails at exec without it.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY prism/ /app/
ENV LD_LIBRARY_PATH=/app
EXPOSE 8080
ENTRYPOINT ["/app/llama-server"]
