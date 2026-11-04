# syntax=docker/dockerfile:1
#
# ghcr.io/noonghunna/strata-maya-club3090: Project Maya (github.com/mw00/project-maya, MIT), a fork of
# Strata for GLM-5.3-Flash, built by club-3090 from one release. Upstream publishes no image and no
# Linux engine (its setup compiles on the user's PC); from v1.1.53 it has a recipe for building an
# image yourself (docker/, adapted from this file, mw00/project-maya#808). This file is ours. build.sh
# fetches the release's commit into the build context and runs it; see README.md.
#
# What it builds, the way maya.py's compile_engine does, with two differences:
#   - STRATA_PORTABLE=ON: ggml's CPU code for an AVX2 baseline instead of the build machine's CPU
#     (upstream's CMake option, which maya.py passes only for its HIP build);
#   - CUDA archs from CUDA_ARCHITECTURES (a fat binary) instead of the user's own cards.
# Text only: the vision encoder and the DFlash2 drafter are not built or fetched.
# Three engines: `maya` (GLM-5.3-Flash) and, when the release has them, `maya-ds4` (DeepSeek-V4.1-Flash-1017,
# Maya's src/ds4 engine with its own tools/ds4 pack builder) and `maya-mm3` (MiniMax-M3.1-Flash, src/mm3, with
# tools/mm3's pack builders; v1.1.29 on).

ARG CUDA_VERSION=13.0.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates ccache cmake curl git ninja-build python3 unzip \
    && rm -rf /var/lib/apt/lists/*
# ccache in front of nvcc and the C/C++ compilers, its cache in a BuildKit cache mount that outlives the build: a new
# release recompiles only the files whose inputs changed (a shared header or ggml change still fans out). The paths
# are fixed (/src, /build), so the hashes match from one build to the next.
ENV CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=30G CCACHE_COMPILERCHECK=content

ARG CUDA_ARCHITECTURES="75;80;86;89;120"
# llama.cpp's source at the commit setup.py pins (LLAMA_CPP_COMMIT): ggml's CPU backend comes from it
ARG LLAMA_CPP_COMMIT
# empty: half the CPU threads, as maya.py does (nvcc takes 2-4 GB per job on the big kernels)
ARG JOBS=""

COPY . /src
RUN test -n "$LLAMA_CPP_COMMIT" \
    && curl -fsSL -o /tmp/llama.zip "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.zip" \
    && unzip -q /tmp/llama.zip -d /tmp \
    && mv "/tmp/llama.cpp-${LLAMA_CPP_COMMIT}" /llama.cpp \
    && rm /tmp/llama.zip

RUN cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
        -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON \
        -DSTRATA_GGML_DIR=/llama.cpp \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
# The engines' targets: `maya` since v1.1.16, `strata` before (v1.1.16 renamed the programs after Maya),
# and DeepSeek's `maya-ds4` and MiniMax's `maya-mm3` beside it in the releases that define them (build.sh
# checks they are there)
# (the ccache statistics of this build go to the build log: build.sh's caller can see what the cache saved)
RUN --mount=type=cache,id=strata-maya-ccache,target=/root/.cache/ccache \
    ccache --zero-stats >/dev/null; \
    target=strata; grep -q -E '^ *add_executable\(maya ' /src/CMakeLists.txt && target=maya; \
    grep -q -E '^ *add_executable\(maya-ds4 ' /src/CMakeLists.txt && target="$target maya-ds4"; \
    grep -q -E '^ *add_executable\(maya-mm3 ' /src/CMakeLists.txt && target="$target maya-mm3"; \
    echo "$target" > /tmp/engine-targets; \
    jobs="${JOBS:-$(( $(nproc) / 2 ))}"; [ "$jobs" -ge 1 ] || jobs=1; \
    for t in $target; do cmake --build /build --target "$t" -j "$jobs" || exit 1; done; \
    ccache --show-stats
# the CPU settings and archs the tree was configured with, read by build.sh's checks
RUN mkdir -p /out/bin && for t in $(cat /tmp/engine-targets); do cp "/build/$t" "/out/bin/$t" || exit 1; done \
    && grep -E '^(STRATA_PORTABLE|GGML_NATIVE|GGML_AVX[A-Z0-9_]*|GGML_FMA|GGML_F16C|GGML_BMI2|GGML_AMX_[A-Z0-9_]*):' \
         /build/CMakeCache.txt | sed 's/^/build /' > /out/maya-cpu-flags.txt \
    && sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[A-Z]*=//p' /build/CMakeCache.txt > /out/cuda-architectures.txt

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 \
    MAYA_UPDATE_CHECK=0
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# The release's own files, less what only the setup, the tests or the HIP build use. The server
# (serve/server.py) and the pack builders (tools/iq_pack.py; DeepSeek's tools/ds4/pack_gguf.py; MiniMax's
# tools/mm3/pack_gguf.py) run from this tree.
COPY --from=build /src /opt/maya
# (data/glm5-synth.gguf, 12 MB, stays: build.sh packs it to check the pack builder works)
RUN rm -rf /opt/maya/.git /opt/maya/tests /opt/maya/bench /opt/maya/ref /opt/maya/third_party/wheels \
           /opt/maya/data/glm5-*-dumps*
# llama.cpp's gguf-py, where setup.py puts it (tools/_paths.py looks there): the pack builder reads the
# tokenizer with it
COPY --from=build /llama.cpp/gguf-py /opt/maya/third_party/llama.cpp/gguf-py
# setup.py's PY_PACKAGES, read from the release's own setup.py so a package a release adds reaches the
# image (v1.1.18 added jsonschema, which the Responses API's structured output needs; a hand-kept list
# here missed it), less cmake / ninja (build tools). The list and the versions pip took are recorded.
RUN mkdir -p /opt/club3090 && python3 - > /opt/club3090/maya-py-packages.txt <<'PY'
import ast, re
tree = ast.parse(open("/opt/maya/setup.py", encoding="utf-8").read())
found = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == "PY_PACKAGES" for t in node.targets)]
assert len(found) == 1, "setup.py: expected one PY_PACKAGES list"
packages = [p for p in ast.literal_eval(found[0])
            if re.split(r"[<>=!~;\[ ]", p, maxsplit=1)[0].lower() not in ("cmake", "ninja")]
assert "numpy" in packages, packages
print("\n".join(packages))
PY
RUN python3 -m venv /opt/maya/.venv \
    && /opt/maya/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/maya/.venv/bin/pip install --no-cache-dir -r /opt/club3090/maya-py-packages.txt \
    && /opt/maya/.venv/bin/pip freeze > /opt/club3090/maya-pip-freeze.txt
# the engines under the release's own names: /opt/maya/build/maya (v1.1.16 on) or /opt/maya/build/strata,
# /opt/maya/build/maya-ds4 and /opt/maya/build/maya-mm3
COPY --from=build /out/bin/ /opt/maya/build/
COPY --from=build /out/maya-cpu-flags.txt /out/cuda-architectures.txt /opt/club3090/

WORKDIR /opt/maya
# No entrypoint of our own in the image: the composes mount the launcher
# (models/glm-5.3-flash/strata-maya/launcher/) and run it.
CMD ["/opt/maya/.venv/bin/python", "/opt/maya/serve/server.py", "--help"]
