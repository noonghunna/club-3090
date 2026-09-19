# vllm-dflash2-mamba-align-offload — DFlash2 offload/prefix-cache hit fix (vLLM 0.29.0)

Ports upstream vLLM commit [`fa5017a5`](https://github.com/vllm-project/vllm/commit/fa5017a5f93e029e58e7d41f906ea6e5adaee323)
*"[Spec Decode] Scope prefix-cache last-block drop to eagle-family drafters (DFlash2 offload fix)"*
onto **`vllm/vllm-openai:v0.29.0`**.

## Why this patch exists
`fa5017a5` landed on vLLM main **2026-09-10, one day after the v0.29.0 cut (09-09)**, so it is
**not in the `v0.29.0` image** (verified: `use_eagle_preserves_target_kv_cache()` has 0 occurrences
in the v0.29.0 tag's `config/speculative.py`). It is also not yet in any released tag
(`v0.29.1rc0` is still 27 commits behind it). This vendored patch is the vLLM half of what the
FIELD_NOTES "0.28.0 + #54165" row did with a local port of [PR #54165](https://github.com/vllm-project/vllm/pull/54165)
(still OPEN); `fa5017a5` is the same intent shipped to main.

**The bug it fixes ([vllm#53505](https://github.com/vllm-project/vllm/issues/53505), closed 2026-08-30):**
`Scheduler._mamba_block_aligned_split()` backs the last cache position off by one mamba block
whenever `use_eagle` is true. `use_eagle()` returns True for `dflash`/`dspark` too, so DFlash2
receives the EAGLE volatile-trailing-block drop on the mamba group → the final block-aligned
mamba snapshot never materializes → **every prefix-cache and offload-tier lookup converges to 0**
("stores but never serves a hit"). The fix scopes that back-off to eagle-family drafters only
(`eagle`/`eagle3`/`mtp`), which share the target's full-attention KV cache groups; DFlash/DSpark
draft from their own KV cache and never write target blocks.

> This is the blocker for **LMCache (or vLLM native KV offload) with DFlash2** on 0.29.0. Without
> it, the offload store is non-zero but reads never hit. [PR #48375](https://github.com/vllm-project/vllm/pull/48375)
> (mounted separately in the compose) is **not** sufficient — it is `single_type_kv_cache_manager`
> only, a different code path.

## Adaptation for 0.29.0
`fa5017a5`'s parent already had `use_eagle_block_drop()`; v0.29.0 is older and still does the drop
**inline** in `_mamba_block_aligned_split` gated on `self.use_eagle`. So this patch (a) adds the
`use_eagle_preserves_target_kv_cache()` capability bit to `SpecDecodingConfig` and (b) scopes the
inline back-off to a `self.use_eagle_preserves_target_kv_cache` attribute. Same two files, same effect:
- `config/speculative.py` — +`use_eagle_preserves_target_kv_cache()` method.
- `v1/core/sched/scheduler.py` — set the attr in `__init__`; gate the mamba-align back-off on it.

## Verification (2026-09-14)
Clean `patch -p1 --dry-run` **and** real apply onto pristine `v0.29.0`; `py_compile` both files;
reverse dry-run clean (idempotency); **and** sequential apply *after* `vllm-dflash2-backport.patch`
(the real entrypoint order) also clean + compiles.

## Delivery
Idempotent `install.sh` (mount at `/etc/club3090/dflash2-mamba-align`, call in the entrypoint
**after** the dflash2-backport and **before** serve): `_check_applied.py` fingerprint, else
`patch -p1 --forward` from the `vllm/` package dir. Refuses boot on anchor drift.

## Drop-when
When a released vLLM tag contains `fa5017a5` (watch `v0.29.1` / `v0.30.0`). Confirm by checking
`use_eagle_preserves_target_kv_cache` is present in that tag's `config/speculative.py` — then the
reverse-apply of this patch should be clean (drop it).
