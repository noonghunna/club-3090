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

## Sanity-check before ANY image bump or sync

Before vendoring updated patched files (or before assuming the patch is
still needed at all), run these three checks:

### 1. Is vllm#40361 still open upstream?

```bash
gh pr view 40361 --repo vllm-project/vllm --json state,mergedAt
```

- `"state":"OPEN", "mergedAt":null` → patch still needed; continue
- `"state":"MERGED"` (or `mergedAt` non-null) → upstream landed it.
  **Delete this entire directory + the four compose mount lines** and
  bump the vLLM image SHA past the merge commit. Patch obsolete.

### 2. Does the pinned image actually still have the bug?

Check the upstream marlin.py at our pinned image's commit SHA for the
pad-sub-tile-n symbols. If they're absent, the patch is doing real work:

```bash
# Replace <pinned-sha> with the actual SHA from the docker image tag
git -C /path/to/local/vllm-clone show <pinned-sha>:vllm/model_executor/kernels/linear/mixed_precision/marlin.py \
  | grep -E '_maybe_pad_n|GPTQ_MARLIN_MIN_THREAD_N|round_up|_marlin_orig_n'
```

- Empty output → upstream marlin.py doesn't have the pad logic; the
  patch is still doing real work
- Non-empty output → upstream has SOMETHING related; investigate
  whether vllm#40361's design merged under a different PR number,
  or whether a different fix landed. Don't blindly delete — verify
  the bug doesn't fire on a fresh dual-card boot first

### 3. Have the patched files diverged from upstream?

```bash
cd /opt/ai/vllm-src    # or wherever your local clone lives
git log <our-fork-base>..<new-image-sha> -- \
  vllm/model_executor/kernels/linear/mixed_precision/marlin.py \
  vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py
```

- Empty: re-copy our patched files (no rebase needed):
  ```bash
  cp marlin.py MPLinearKernel.py /path/to/club-3090/models/qwen3.6-27b/vllm/patches/vllm-marlin-pad/
  ```
- Non-empty: rebase the patch onto the new upstream first, then re-copy

### Last-checked log

When syncing, append a row here so future readers can see the
verification trail without re-running everything:

| Date | Image SHA | PR state | Bug present? | Action taken |
|---|---|---|---|---|
| 2026-05-03 | `7a1eb8ac2ec...` | OPEN | yes (grep returned empty) | Initial vendor |

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
