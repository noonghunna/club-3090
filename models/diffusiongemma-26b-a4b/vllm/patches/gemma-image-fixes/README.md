# DiffusionGemma — Ampere/TP fixes for the official `vllm/vllm-openai:gemma` image

vLLM publishes an official **`vllm/vllm-openai:gemma`** image (pushed 2026-06-10,
same day as the DiffusionGemma blog) that has the dLLM arch baked in — it's a stock
vLLM build of the `dgemma` branch commit `74b5964f` (`DiffusionGemmaForBlockDiffusion`
registers natively; `transformers 5.10.2`). So we **pin that image** and no longer
sideload the model code.

But three fixes are **NOT upstream** (not in PR #45163), so they're not in `:gemma` —
vLLM builds/tests on H100/B200 and their recipe is TP=1, so neither our Ampere fp8
path nor our TP=2 path is exercised upstream. The compose **bind-mounts these 3 files**
over the image's vllm package (`site_package_overlay`, `wired_at: volumes`):

| File → mounts over | Fix |
|---|---|
| `marlin.py` → `vllm/model_executor/kernels/linear/scaled_mm/marlin.py` | sm_86 fp8 Marlin **sub-tile-K pad** (dense). Without it, `:gemma` dies in warmup: `Invalid thread config … num_bits=8 … max_shared_mem=101376` for K=352/1056 (can't tile in Ampere's 99 KB shared mem). |
| `marlin_utils_fp8.py` → `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py` | the FP8-MoE half of the K-pad. |
| `diffusion_gemma.py` → `vllm/model_executor/models/diffusion_gemma.py` | **TP-vocab soft-embed fix** (slice probs to the rank's vocab shard → local-embed matmul → TP all-reduce) + `:656` dtype cast. `:gemma`'s copy is the pristine branch file → hits the TP=2 vocab/dtype bug without this. |

Validated live on 2× RTX 3090 (2026-06-11): `:gemma` clean dies on the Marlin wall in
warmup; `:gemma` + these 3 mounts boots, serves coherent output, 262K, ~177/180 TPS
typical (~1100 peak on low-entropy), 23.1 GB/card.

## Provenance + rebase

These 3 are the surviving fixes from the original sideload overlay (Codex's marlin-K-pad
= the K-axis analogue of our PR #40361 sub-tile-N pad; + the TP-vocab/dtype fix). The
marlin pair were authored against a stock-nightly marlin (identical stock==dgemma-branch),
and apply cleanly onto `:gemma`'s newer base (verified — no skew).

**Rebase when `:gemma` is re-pinned** (vLLM may re-push the tag): pull the new image,
diff its `vllm/model_executor/.../marlin.py` / `marlin_utils_fp8.py` / `models/diffusion_gemma.py`
against these, re-apply the K-pad + TP-vocab deltas, re-validate (boot + serve + the gate).

**Retire entirely** when the K-pad lands upstream (our PR #40361 / an Ampere Marlin fix)
*and* the TP-vocab fix merges into `:gemma` — then mount nothing.
