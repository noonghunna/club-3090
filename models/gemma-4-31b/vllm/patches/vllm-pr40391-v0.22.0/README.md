# vLLM PR #40391 — Gemma 4 per-token-head KV cache (lean diff-apply overlay)

**Upstream PR:** [vllm-project/vllm#40391](https://github.com/vllm-project/vllm/pull/40391) (open)
**Target image:** pinned stock `vllm/vllm-openai:v0.22.0`
**Delivery:** boot-time diff-apply (`install_script`) — see `install.sh`

## What this fixes

Gemma 4 has interleaved attention with two head dims (sliding/local **256**, global
**512**). Per-token-head KV quantization (`int8_per_token_head`) adds per-token scale
metadata, so the local vs global per-block page sizes (520 vs 1032 bytes) don't share
the clean ratio vLLM's `unify_kv_cache_spec_page_size()` requires → stock vLLM refuses
to boot. #40391 pads global layers to a 1040-byte factor and routes the non-MLA padded
layout through a new `get_padded_attention_kv_cache_shape()` helper. This is the **only**
reason `vllm/gemma-int8-mtp` is pinned: it unlocks INT8-PTH KV (→ 262K) on Ampere.
(`bf16` KV needs no patch — it never hits the per-head page-size problem.)

## Why a diff-apply, not full-module mounts

This **replaces** the old `vllm-pr40391-rebased/` overlay, which mounted **7 full v0.22.0
modules** (~13K lines) over the stock files. That style is heavyweight and drift-prone — a
full-module mount silently *reverts* any other v0.22.0.x change to those files on a re-pin,
and the v0.21.0-era copies ImportError'd on v0.22.0 (they lacked v0.22.0's
`get_kv_cache_spec_kind`). This dir carries **only the genuine ~240-line delta**:

```
pr40391-v0.22.0.patch     # 6-file unified diff (stock v0.22.0 → +#40391)
kv_cache_shape_utils.py   # the one new helper file #40391 adds
install.sh                # idempotent, fail-loud boot-time applier
```

The diff was generated against stock `v0.22.0` and includes two club-3090 resolutions the
raw PR diff needs on v0.22.0: (1) the `gpu/attn_utils.py` `is_mla()` branch hand-merged onto
v0.22.0's `_reshape_kv_cache` (the PR's base predated v0.22.0's refactor), and (2) the PR's
removal of the `replace` import from `kv_cache_utils.py` is **dropped** — v0.22.0 still uses
`replace` at another site, so removing it would `NameError`.

## How it's wired

`int8.yml` mounts this dir read-only at `/etc/club3090/pr40391` and runs `install.sh` from
the entrypoint **before** `vllm serve`. The script no-ops cleanly if #40391 is already
present (future merged image) and `exit 1`s if the diff fails to apply (never ships a
half-patched engine).

## Drop trigger

Delete this overlay + unpin when #40391 merges upstream and lands in the pinned release:

```
gh api repos/vllm-project/vllm/pulls/40391 --jq '.state, .merged_at'
```

Tracked in [`docs/UPSTREAM.md`](../../../../../../docs/UPSTREAM.md).
