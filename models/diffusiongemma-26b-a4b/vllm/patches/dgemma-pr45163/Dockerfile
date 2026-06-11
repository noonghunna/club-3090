# =============================================================================
# vLLM PR #45163 (DiffusionGemma — `dgemma` branch) overlaid on the June-10 nightly.
#
# COMPOSE-ONLY, untracked, fully ISOLATED from the vllm-stable / vllm-gemma-stable
# engine pins and the compose_registry. Build a local image, point the standalone
# compose at it; nothing in the launchers/registry/tests references this.
#
# Method — cp-over-python overlay (NO recompile):
#   PR #45163 changes 49 files, but ALL are .py / Triton kernels except one
#   flash-attn `.cmake` bump (irrelevant: we serve with TRITON_ATTN, a JIT Triton
#   path). No .cu/csrc changes → the nightly's compiled _C.so stays compatible.
#   So we just copy the branch's vllm/ python tree over the installed package and
#   bump transformers to main (for models/diffusion_gemma). The final RUN asserts
#   the arch registered, so the build fails fast if the overlay didn't take.
#
# Build:  docker build -t vllm-dgemma:pr45163 \
#           models/diffusiongemma-26b-a4b/vllm/patches/dgemma-pr45163/
# Re-pin: bump BASE to a newer nightly SHA if the branch rebases.
# Retire: delete this dir + the compose once #45163 merges into a nightly we pin.
# =============================================================================
ARG BASE=vllm/vllm-openai:nightly-2c9c07c85e56c799afffd5a671a8a0bace377a39
FROM ${BASE}
# git isn't in the slim runtime image; needed for the branch clone + pip git+ transformers.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# Optional: pin DGEMMA_SHA to a specific dgemma commit for reproducibility
# (the branch is actively moving). Default empty = clone current HEAD.
ARG DGEMMA_SHA=
RUN set -eux; \
    git clone --branch dgemma https://github.com/vllm-project/vllm.git /tmp/dgemma; \
    if [ -n "${DGEMMA_SHA}" ]; then git -C /tmp/dgemma checkout "${DGEMMA_SHA}"; fi; \
    git -C /tmp/dgemma rev-parse HEAD | tee /etc/dgemma-commit.sha; \
    git -C /tmp/dgemma log -1 --oneline | tee -a /etc/dgemma-commit.sha; \
    VLLM_PKG="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"; \
    echo "overlaying dgemma vllm/ tree ($(cat /etc/dgemma-commit.sha | head -1)) onto $VLLM_PKG"; \
    cp -r /tmp/dgemma/vllm/. "$VLLM_PKG"/; \
    rm -rf /tmp/dgemma "$VLLM_PKG"/__pycache__; \
    pip install --no-cache-dir -U "transformers @ git+https://github.com/huggingface/transformers.git@main"; \
    python3 -c "from vllm.model_executor.models.registry import ModelRegistry as R; \
archs=R.get_supported_archs(); \
assert 'DiffusionGemmaForBlockDiffusion' in archs, 'FAIL: arch not registered after overlay'; \
print('OK: DiffusionGemmaForBlockDiffusion registered after overlay')"
