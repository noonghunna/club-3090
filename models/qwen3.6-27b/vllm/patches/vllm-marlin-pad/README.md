# vLLM Marlin pad-sub-tile-n vendored patch

**What's here:** the two patched files from our [vLLM PR #40361](https://github.com/vllm-project/vllm/pull/40361),
vendored into the repo so dual-card composes don't need a host clone of
the patched fork.

```
marlin.py            ← from vllm/model_executor/kernels/linear/mixed_precision/marlin.py
MPLinearKernel.py    ← from vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py
```

## Why this exists

vLLM's Marlin INT4 GEMM kernel requires `out_features ≥ GPTQ_MARLIN_MIN_THREAD_N`
(64). On TP=2 with the Lorbus AutoRound INT4 quant, some MTP-related layers
have `out_features = 128` which split into 64-per-card — exactly at the
boundary. Other shapes split below 64 entirely and the kernel crashes:

```
RuntimeError: GPTQ_MARLIN_MIN_THREAD_N (64) > out_features
```

The fix pads the tensor's out-dim up to the kernel minimum at load time, then
slices the result back during `apply_weights`. Trade-off: one extra memcpy
per Marlin call. Measured cost: <0.5% TPS overhead. Gain: works on any TP
shape that produces sub-tile layers.

## Compose mounts

All four dual-card vLLM composes mount these files into the running
container, replacing vLLM's shipped (unpatched) versions:

```yaml
volumes:
  - ../patches/vllm-marlin-pad/marlin.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:ro
  - ../patches/vllm-marlin-pad/MPLinearKernel.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:ro
```

**No host filesystem dependency** — paths are relative to the compose dir,
files ship in this repo.

## Provenance + sync

- **Source:** [`noonghunna/vllm` `marlin-pad-sub-tile-n` branch, commit
  `67f8c2b`](https://github.com/noonghunna/vllm/tree/marlin-pad-sub-tile-n)
- **License:** Apache 2.0 (vLLM's license — patched files retain it)
- **Pinned image SHA this matches:** `vllm/vllm-openai:nightly-7a1eb8ac2ec4ea69338c51dc7afd4b15010abfa8`
- **Last sync date:** 2026-05-03

If the pinned vLLM image bumps **and** upstream changes `marlin.py` /
`MPLinearKernel.py` between our base and the new image, the vendored copies
need to be rebased on the new upstream and re-copied. Steps:

```bash
# Verify upstream files haven't changed since our fork base
cd /opt/ai/vllm-src    # or wherever your local clone lives
git log <our-fork-base>..<new-image-sha> -- \
  vllm/model_executor/kernels/linear/mixed_precision/marlin.py \
  vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py

# If empty: re-copy our patched files (no rebase needed)
cp marlin.py MPLinearKernel.py /path/to/club-3090/models/qwen3.6-27b/vllm/patches/vllm-marlin-pad/

# If non-empty: rebase the patch onto the new upstream first, then re-copy
```

This vendored copy drops out entirely when [vllm#40361](https://github.com/vllm-project/vllm/pull/40361)
merges upstream — at which point we'd delete this directory and the four
compose mount lines.

## Why not a runtime text-patch sidecar?

Most of our other patches (`patch_tolist_cudagraph.py`,
`patch_inputs_embeds_optional.py`) are runtime regex-based text-patches that
modify ~5-10 lines per file. The Marlin patch is **~120 lines of substantive
code** — a new `_maybe_pad_n` method, edits to `process_weights_after_loading`
and `apply_weights`, plus imports/logger/class-field declarations. A
text-patch covering that surface would be brittle (every upstream marlin.py
change risks breaking it) and harder to review.

Vendoring two files with a clear sync procedure is the cleaner trade.
