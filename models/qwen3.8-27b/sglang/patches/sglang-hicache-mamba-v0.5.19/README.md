# SGLang HiCache L2 / Mamba Radix-Cache Patches (v0.5.19)

Two independent SGLang engine patches applied before launch: `apply_39342.py`
is **fail-closed** (validated, load-bearing — refuses boot on drift) and
`apply_33713.py` is **fail-open** (hypothesis-only — warns + continues on
drift, never blocks a boot).

## Status

| Patch | Upstream ref | Status | Validation |
|-------|-------------|--------|------------|
| `apply_39342.py` | sgl-project/sglang#39342 | **VALIDATED** | 90/90 clean on 4x3090 TP4 W8A8 rig (2026-09-15). The exact trigger that crashed the unpatched stack is neutralized. v2 (2026-09-16) relocates the rollback to the head of `mix_with_running` (F2 fix: the v1 tail placement over-reached onto running decode reqs and silently dropped their KV from the cache under sustained load — a cache-hit regression invisible to a crash/correctness test). |
| `apply_33713.py` | sgl-project/sglang#33713 | **HYPOTHESIS-ONLY** | NOT reproducible on the GDN model this compose serves (Qwen3.8-27B). The KV-level symptom reproduces on the issue's KDA model (Ling-3.0-flash). Fail-closed within itself + idempotent; included per request, not as a confirmed fix. The prod wrapper is **fail-open** (a drift warns + continues; it never blocks a boot) and it is bench-only. |

## What they fix

### #39342 — Mixed-chunk mamba radix-cache corruption

`prepare_for_extend` stamps `req.kv.mamba_last_track_seqlen` for each incoming
prefill. `merge_batch` (called from `mix_with_running`) then unconditionally
nulls the batch `mamba_track_*` tensors, so the GDN checkpoint write is
skipped. The per-req claim is never rolled back, and
`mamba_component.prepare_for_caching_req` reads `mamba_last_track_seqlen`
to size the mamba donation. Result: a stale ping-pong slot (never written for
this seqlen) is donated into the unified radix tree under a mismatched key.

The 5-edit fix (A1/A2/B on `schedule_batch.py`, C/D on `mamba_component.py`)
is a load-bearing unit. Edit B is at the **head** of `mix_with_running`
(before `merge_batch`), so it rolls back only the **incoming prefill** claims,
not the running decode reqs (whose stamps remain valid). Edit C returns **0**
(not `None`): the caller's `if cl is not None:
effective_cache_len = min(len(token_ids), cl)` then truncates to 0. In
`cache_unfinished_req` that hits the existing `effective_cache_len <= 0`
early-return guard; in `cache_finished_req` (which has no such guard) the
empty-key `insert()` is short-circuited by `unified_tree_core.begin_insert`.
Either way: no insert, no assertion, no stale mamba donation. This is what
makes the patch safe under mixed-chunk retraction.

**Cost**: the rolled-back (mixed) prefill's KV prefix is not cached (cold
re-prefill on next reuse). This is the safe analogue of pristine, which cached
a stale mamba slot. (Carrying the mamba track through the merge so mixed
prefills still checkpoint is a follow-up, upstream sgl#39526.)

### #33713 — MAMBA host-leaf pruned instead of downgraded

The eviction fork in `unified_tree_core.py` keys off `node.backuped`, which
is the FULL (KV) component's `host_value is not None` ONLY. Mamba is a
separate component; a node can demote on KV but lack a mamba host copy.
The 2-edit match-side change treats a node as a host leaf if **any** component
(FULL or MAMBA) has a host copy.

**NOT the evict-fork fix**: the evict decision still keys off the FULL
component. The KV-level "no loadback" symptom does not reproduce on the GDN
model. This patch addresses the match-side half only.

## Files patched

| File | Patch |
|------|-------|
| `python/sglang/srt/managers/schedule_batch.py` | #39342 (edits A1, A2, B) |
| `python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py` | #39342 (edits C, D) |
| `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py` | #33713 (edits E1, E2) |

All three are in the v0.5.19 unified-cache path. The patches are
content-anchored (byte-unique anchors), version-gated (warn on version
drift, don't hard-fail), and idempotent.

## Reversibility

Each script supports `--revert`:

```
python3 apply_39342.py --revert   # undo #39342
python3 apply_33713.py --revert   # undo #33713
```

Reverting is gated on the per-edit marker: it only inverts edits that were
actually applied, so a partial state won't be corrupted by a revert.

## F2 fix — v1 tail rollback over-reached (corrected 2026-09-16)

v1 placed the `mix_with_running` rollback loop at the **tail** (after
`merge_batch`), where `self.reqs` = incoming prefills **plus** running decode
reqs. A running decode req carries a valid `mamba_last_track_seqlen` (stamped
in its own earlier prefill, cleared only at finish), so the loop cleared that
stamp and set `mamba_mixed_rollback = True` on **every** running decode req in
a mixed batch. Edit C then made `prepare_for_caching_req` return 0 for them, so
at finish their KV prefix was **not cached** — a cold re-prefill on every reuse.
Under sustained high concurrency with mixed chunking (the deployed scenario)
that is a pervasive TTFT / cache-hit-rate regression, invisible to an
answer-only or crash-only test.

**Fix:** relocate the loop to the **head** of `mix_with_running`, before
`merge_batch`, where `self.reqs` is only the incoming prefills. The decode
reqs' stamps stay intact. The script carries a transient migration edit that
reverts the v1 tail rollback before adding the head one, so an existing v1 tree
lands cleanly on v2.

**Remaining field validation:** a cache-hit-rate / TTFT A/B against the
unpatched stack on the mixed-chunk workload (the check the old answer-only A/B
missed). The *intentional* cost of not caching a rolled-back mixed prefill is a
separate follow-up (upstream sgl#39526 — carry the mamba track through the merge
so mixed prefills still checkpoint).

## Provenance

- #39342: authored by @A1RM4X, validated 2026-09-15 on a 4x RTX 3090
  (GA102, TP4, PCIe) rig running Qwen3.8-27B W8A8 + DFlash2 + HiCache L2
  + mixed-chunk. SGLang v0.5.19.
- #33713: authored by @A1RM4X, 2026-09-15. Hypothesis-driven, not
  validated against the model this compose serves.

## Compose mount

Mounted at `/etc/club3090/hicache-mamba` (read-only). The compose entrypoint
runs `bash /etc/club3090/hicache-mamba/install.sh` before launching the
server. The compose also requires `--enable-hierarchical-cache` +
`--enable-mixed-chunk` + `--radix-eviction-policy slru` +
`--mamba-max-states-per-path 1` for the #39342 patch to be meaningful
(those flags are the trigger condition).
