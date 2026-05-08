# vLLM PR #40391 — Gemma 4 per-token-head fp8 KV cache (rebased overlay)

**Author:** [@lisp19](https://github.com/lisp19) (original)
**Upstream PR:** [vllm-project/vllm#40391](https://github.com/vllm-project/vllm/pull/40391)
**Vendored on:** 2026-05-08 (post-Gemma4-MTP-merge nightly base)
**Local rebase by:** noonghunna for club-3090 (Ampere consumer cross-rig validation path)

## What this fixes

Gemma 4 has interleaved attention layers with two different head dimensions:
- Sliding-window / local layers: `head_dim = 256`
- Global / full layers: `head_dim = 512`

When per-token-head fp8 KV (`--kv-cache-dtype fp8_e5m2`) is requested, the layout adds
8 bytes of scale metadata per token per head. This breaks vLLM's
`unify_kv_cache_spec_page_size()` because the resulting page sizes are 520 vs 1032 —
they don't share a clean 1:2 ratio that the allocator's slab structure can absorb.

Without this PR, `gemma-mtp.yml` and friends are forced to `--kv-cache-dtype auto`
(BF16 KV) on Ampere, which caps usable context at ~32K (vs Gemma 4's natural 1M).

The PR's fix:
1. Pre-pads the global layers' KV cache spec to a 1040-byte factor at the model spec level
2. Updates the worker-runtime view to read the padded layout
3. Makes `unify_kv_cache_spec_page_size()` accept the new ratio
4. Routes standard attention backends (non-MLA) through a new `get_padded_attention_kv_cache_shape()` helper for the contiguous-view path

## Why we vendor a rebased copy

Upstream PR is open + has cross-rig validation (cferra on sm_120 Blackwell, noonghunna
on sm_86 Ampere) but is `mergeable_state: dirty` against current main and stalled on
maintainer review. The PR's branch was last updated 2026-05-06 with a merge from main;
since then, vLLM main landed:
- Mamba hybrid attention support (renamed `raw_tensor` → `kv_raw_tensor`, added
  `has_attn` / `has_mamba` dispatch in `gpu/attn_utils.py:_reshape_kv_cache`)
- Various other refactors

Our rebase resolves the conflict in `vllm/v1/worker/gpu/attn_utils.py` by combining:
- main's hybrid attention/mamba dispatch structure
- main's variable rename (`kv_raw_tensor` / `kv_tensor` / `cache_dtype_str`)
- PR #40391's MLA-vs-standard-attention split for `page_size_padded` handling
- PR #40391's `get_padded_attention_kv_cache_shape` import

Resolution preserved both PRs' intents:
- Hybrid Mamba models continue to work (main's path)
- Gemma 4 per-token-head fp8 KV becomes available on Ampere (PR #40391's path)

## Files vendored

7 source files (test files from the PR not vendored — we don't run vLLM tests in serving):

```
model_executor/layers/attention/attention.py    # +5 lines: kv_cache_shape pad routing
model_executor/models/gemma4.py                 # +23 lines: per-token-head spec setup
v1/core/kv_cache_utils.py                       # +2/-2: page_size_padded acceptance
v1/kv_cache_interface.py                        # +17 lines: page_size_padded field
v1/worker/gpu/attn_utils.py                     # MERGE-RESOLVED w/ main's hybrid dispatch
v1/worker/gpu_model_runner.py                   # +27/-18: KV cache build call signature
v1/worker/kv_cache_shape_utils.py               # +78 lines (new file): the helper itself
```

## Validation status (as of vendor date)

- ✅ All 7 files Python-parse OK
- ✅ Conflict resolution preserves intent of both PRs
- ⏳ **Boot validation pending** — needs Qwen container down on dual 3090 to validate
- ⏳ **verify-stress (91K needle)** — critical gate. Codex's earlier overlay attempts on
  this code area produced decode-TPS decay turn-1 33 → turn-5 10 (30% retention).
  Long-context stress test catches this regression class.
- ⏳ **Soak (continuous, 5 sessions)** — confirms 0 VRAM growth + no per-turn decode decay
- ⏳ **Bench parity** — should match the PR's claimed 4× context lift (32K → ~120K)

## Drop trigger

When PR #40391 merges to vLLM main AND propagates to a nightly tag,
this overlay can be removed entirely. Track upstream merge status:

```bash
gh api repos/vllm-project/vllm/pulls/40391 --jq '.state, .merged_at'
```

## Companion overlay

Used in tandem with [`../vllm-gemma4-tool-parser-fixes/`](../vllm-gemma4-tool-parser-fixes/)
which stacks PR #42006 (MTP streaming multi-tool calls) + PR #41991 (parser bounds).
Both target `gemma4_tool_parser.py` at non-overlapping regions.

## Composes that mount this

Currently: `docker-compose.gemma-mtp-fp8.yml` (the fp8-KV variant introduced for this overlay).

The bf16-KV variants (`docker-compose.gemma-mtp.yml`, `docker-compose.gemma-mtp-tp1.yml`)
do NOT mount this overlay — they don't need per-token-head fp8 KV path. Keep them
overlay-free as the safe fallback.
