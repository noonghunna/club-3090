# Upstream tracker

Issues and PRs in upstream repos that affect this stack — what we depend on, what we've filed, what unblocks for us when each lands.

This file is the **single source of truth** for upstream status. When you file or notice an upstream issue / PR / commit relevant to club-3090, add a row here. When status changes (closed, merged, propagated), update it. Don't scatter the same link across multiple docs without coming back here first.

If you're adding a new compose that depends on an unmerged upstream patch (volume-mount of a fork, monkey-patch script), it MUST link to a row in this file so future readers know when the workaround can drop.

---

## How rows work

Each row covers one upstream link with: **title • status • our dependency / impact • workaround (if any)**.

**Status vocabulary:**
- 🟢 **Landed** — merged upstream + propagated to our pinned versions (pin-bump done)
- 🔵 **Merged, awaiting propagation** — merged upstream but our nightly / commit pin hasn't picked it up yet
- 🟡 **Open / in review** — PR open, no merge yet; we depend on it landing
- 🟠 **Open / blocked or stalled** — PR exists but progress stalled
- 🔴 **Open, no PR yet** — issue acknowledged but no fix in progress (us or upstream)
- ⚫ **Workaround locally, no plan to merge** — fixed in our patches, upstream not pursuing
- ✅ **Resolved** — closed and resolved (kept for historical context)
- ❌ **Closed without fix** — closed, won't fix, kept for context

---

## Active follow-ups (next-week revisit queue) 🗓️

Items deferred for review next week (week of 2026-05-10). Audit at the start of that week — most should be either ready to action or have new upstream signal worth re-evaluating.

| Item | Why deferred | Trigger to revisit |
|---|---|---|
| **🟡 [Sandermage/genesis-vllm-patches#22](https://github.com/Sandermage/genesis-vllm-patches/issues/22)** — PN59 streaming-GDN doesn't engage on chunked-prefill (single-card 24 GB Cliff 2b stays open under v7.72.2). Filed 2026-05-05 with reproducer + 4 fix proposals. | Awaiting Sander's review. The cleanest of our 4 proposals is making `has_no_chunk_metadata` rejection optional (env-gated), letting single-seq chunked-prefill take the streaming path. | Sander posts a candidate fix or comment on the issue; or we run a one-line A/B if he requests it. **Until resolved, single-card 24 GB long-context users should run `dual.yml` / `dual-turbo.yml` / `llamacpp/default`.** |
| ~~**Genesis pin bump `2db18df` → `f2147ad`**~~ | ✅ **Done 2026-05-05** — bumped to `7b9fd319` (v7.72.2) on branch `v7.72.2-uplift`. Drops `patch_inputs_embeds_optional.py` (PN35 native), `patch_pn30_dst_shaped_temp_fix.py` (PN30 v7.68), `patch_pn25_genesis_register_fix.py` (PN25), `patch_tolist_cudagraph.py` (P78), `patch_workspace_lock_disable.py` (PN34), `patch_pr40798_workspace.py` (research artifact). | — |
| ~~**Enable P68/P69 across composes**~~ | Superseded by v7.72.1 P68 auto-skip + v7.72.2 PN70 schema-subset filter — both ship default-aware behavior. Closed [#57](https://github.com/noonghunna/club-3090/issues/57) along the way. | — |
| **Rebase + ping vllm#40361** (our Marlin pad-sub-tile-n PR) | 13 days stale on vLLM upstream as of 2026-05-03; not blocking anything locally (we vendor the patched files in-repo) but worth keeping on the maintainer queue. | Open a "still relevant" comment + rebase if behind main, link to the JusefPol NVLink merge as additional cross-rig evidence the patch is in real use. |

See the platform-specific tables below for the rows these reference.

---

## vLLM (`vllm-project/vllm`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [#40361](https://github.com/vllm-project/vllm/pull/40361) — Marlin pad-sub-tile-n | 🟡 Open, mergeable, **stale 13d** (last update 2026-04-20) | All 4 dual-card composes + `dual-nvlink.yml` mount the patched files vendored in-repo at `models/qwen3.6-27b/vllm/patches/vllm-marlin-pad/`. Drops out as a setup dependency when this merges + propagates. | Vendored mount: see [`models/qwen3.6-27b/vllm/patches/vllm-marlin-pad/README.md`](../models/qwen3.6-27b/vllm/patches/vllm-marlin-pad/README.md). Queued for rebase + ping next week (see "Active follow-ups" table above). |
| [#40807](https://github.com/vllm-project/vllm/issues/40807) — `.tolist()` cudagraph crash on continuation-prefill | ⚫ Local workaround | Single-card TQ3 + spec-decode + chunked-prefill blocked without it. We ship a file-edit patch. | `patch_tolist_cudagraph.py` runs in `setup.sh`. Drop when upstream fixes the sync. |
| [#40849](https://github.com/vllm-project/vllm/pull/40849) — MTP draft online-quant propagation | 🟡 Open / Genesis backport active | Closes Cliff 1 on FP8+MTP path (`tools-text.yml`). | Genesis PN8 backport: `GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1`. |
| [#40914](https://github.com/vllm-project/vllm/pull/40914) — Sandermage K+1 verify routing | 🟡 Open | Recovers ~22 TPS narrative regression on TQ3 + spec-decode (P65 cost). When it lands, default compose's narrative TPS catches up to ampersandru's pre-P65 numbers. | None local — wait for merge. |
| [#40334](https://github.com/vllm-project/vllm/pull/40334) — DFlash `combine_hidden_states` dtype mismatch | 🟡 Open | All `dual-dflash*.yml` need `--dtype bfloat16` flag to work around. | Composes set `--dtype bfloat16`. Drop when this lands. |
| [#40382](https://github.com/vllm-project/vllm/issues/40382) — Gemma-4 + DFlash unservable on Ampere | 🟠 Open, no fix in progress | Blocks DFlash on Gemma-4 family. Not directly our problem (we serve Qwen3.6) but tracked because future model adds may hit it. | None — different attention backend selection. |
| [#40354](https://github.com/vllm-project/vllm/issues/40354) — Marlin TP=2 W4A16 < 64 | ✅ Same root-cause as #40361 | Our PR #40361 resolves this. | See #40361 row. |
| [#39931](https://github.com/vllm-project/vllm/issues/39931) — DeltaNet rollback support | 🔴 Open, architectural | Blocks **all** spec-decode (EAGLE / DFlash) on Qwen3-Next family across engines. The reason "speculative decoding doesn't work" on this stack. | Use MTP (no rollback needed) until this lands. |
| [#40124](https://github.com/vllm-project/vllm/issues/40124) — related architectural | 🔴 Open | Pairs with #39931 for DeltaNet rollback. | Same as above. |
| [#40880](https://github.com/vllm-project/vllm/issues/40880) — MTP × TQ × cudagraph cascade | ✅ Closed (P65 fix) | Fixed by Genesis P65 (cudagraph PIECEWISE downgrade for spec-decode). | Auto-on in default compose. |
| [#40831](https://github.com/vllm-project/vllm/issues/40831) — TQ × spec-decode corruption | ✅ Closed | Resolved by P65 (MTP) + ngram-mod `prompt_lookup_min=8` (ngram). | See P65 / ngram routing. |
| [#40798](https://github.com/vllm-project/vllm/pull/40798) — workspace-manager refactor | ❌ Negative result | Hypothesized fix for #40831 / #40880; backporting it (Probe 8) didn't resolve the bug. Kept for context — saved future time on the same dead end. | n/a |
| [#40875](https://github.com/vllm-project/vllm/issues/40875) — ngram + MTP coexistence | ✅ Closed | Routed via `prompt_lookup_min=8` flag. | Set in compose where applicable. |
| [#41142](https://github.com/vllm-project/vllm/pull/41142) — Quentin-M streaming tool-call IndexError | 🟡 Open / Genesis backport active | Closes a streaming tool-call crash on Hermes / similar templates. | Genesis PN11 backport (auto-enabled where REC). |
| [#39598](https://github.com/vllm-project/vllm/pull/39598) — kotori-yan qwen3coder MTP streaming early-return | 🟡 Open / Genesis backport active | Empty `tool_calls[]` when MTP bundles last param + `</function>` in same delta. | Genesis P64 backport: `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1` (default-on in our composes). |
| [#40961](https://github.com/vllm-project/vllm/pull/40961) — Preserve max_seq_len in ubatch metadata during CUDA graph capture | 🟡 Open PR | Confirms the cap-leak pattern: cudagraph capture passes `max_model_len` as `max_seq_len` through ubatch metadata. PR is *fixing a missing pass-through* for SWA models (where seqlen=1 at capture broke kernel selection) — by establishing that `max_model_len` is what gets carried through capture metadata, it cements the source of Cliff 1's max-ctx-dependent FA2 workspace sizing. | Stay at `default` 48K — see FA2 #1011 row + INTERNALS.md Cliff 1 mechanism. |
| [#40069](https://github.com/vllm-project/vllm/issues/40069) — [Tracking] TurboQuant / HIGGS Attention follow-ups | 🟡 Open tracker | Umbrella tracking for TurboQuant + attention backend issues on our stack class. | Watch for cross-references when Cliff 1/2 work lands upstream. |
| [#25543](https://github.com/vllm-project/vllm/pull/25543) — [V0 Deprecation] Remove `max_seq_len_to_capture` | ✅ Merged 2025-09-24 | Important to know: the `--max-seq-len-to-capture` flag (commonly suggested as a Cliff 1 mitigation) **does not exist in V1**. Don't recommend it. | n/a — flag removed. |
| [#39226](https://github.com/vllm-project/vllm/pull/39226) — workspace-resize GPU memory leak fix | 🔵 Merged into v0.20.0; covered by sidecar | Strict `WorkspaceManager.lock()` semantics. After our 2026-05-01 v0.20 + Genesis v7.65 dev tip migration, the surfaces that locked at 0 MB on our config are largely covered by v0.20's revised TQ FA paths ([#40092](https://github.com/vllm-project/vllm/pull/40092)). For the residual cases, our local `patch_workspace_lock_disable.py` sidecar (mounted on every TQ3 compose) downgrades the strict assertion to a one-shot WARNING. P98 covers the same surface but auto-skips on v0.20 due to a drift-marker false-positive (filed as side-note, awaiting Sandermage marker fix). | Drop the sidecar when Sandermage ships the marker fix that re-enables P98 on v0.20. |
| [#40092](https://github.com/vllm-project/vllm/pull/40092) — TurboQuant FA3/FA4 prefill paths | 🔵 Merged into v0.20.0 | TQ + flash-attention 3/4 prefill support. Relevant if/when v0.20 unblocks for us — the FA varlen workspace allocator behavior may change under FA3/FA4 vs the FA2 path we currently hit. | Track. Re-evaluate the `flash_attn_interface.py:300` cliff (Genesis #15) once v0.20 unblocks since FA3/FA4 may have different workspace semantics. FA3/FA4 not enabled on Ampere SM 8.6 anyway (Hopper+ only). |
| [#40941](https://github.com/vllm-project/vllm/pull/40941) — TurboQuant share buffers | 🔵 Merged into v0.20.0 | Sandermage's bare_metal_27b_int4_TQ_k8v4.sh comments call out P98 as the workaround for "WorkspaceManager fix vs vllm#40941". Same WorkspaceManager class that vllm#39226 made strict. | Sandermage's **P98** is the workaround — required for TQ k8v4 on hybrid. Worth a focused investigation: enable P98 on the v0.20-experimental compose to see if it also unblocks vllm#39226's path. |
| [#35975](https://github.com/vllm-project/vllm/pull/35975) — Skip `inputs_embeds` GPU buffer for text-only models ⭐ | 🟡 Open upstream / **local backport active** / **Genesis PN35 lands same fix on dev `f2147ad`** (2026-05-03) | Frees ~444 MiB at boot on Qwen3.6-27B (both `gpu_model_runner.py` + `llm_base_proposer.py` call sites compound; PR claims ~64 MiB but our config has multiple residency points that benefit). **Critical for Cliff 2 closure** at 60K on TP=1 + 24GB — combined with mem-util 0.93, closes the late-stage 50 MiB activation peak. Diagnosed by ChatGPT/Codex as the missing margin; cross-rig validated 2026-05-02 PM. Upstream PR last updated 2026-03-13 (51d stale); Sandermage's PN35 is the practical replacement. | `patch_inputs_embeds_optional.py` ships at compose-entrypoint time (mounted on `long-text.yml` and `long-text-no-mtp.yml`). **Drops out when we bump GENESIS_PIN to dev tip — queued for next-week revisit.** |
| [#37429](https://github.com/vllm-project/vllm/pull/37429) — Hybrid Mamba/attention KV cache sizing | 🟡 Open | Could free more residency without trading mem-util. Larger/riskier than #35975 (architectural Mamba allocation change). Untested on this stack. | Not currently backported. Test on a separate branch when CI signals stabilize. |
| [#37521](https://github.com/vllm-project/vllm/pull/37521) — Spec-decode warmup memory accounting | 🟡 Open | Profiling/KV sizing leaves less false headroom. Genesis PN33 already extends this beyond the original `use_eagle()` gate — so most of the surface is covered, but watch for upstream refinement. | n/a — Genesis PN33 covers the path. |
| [#36598](https://github.com/vllm-project/vllm/issues/36598) — Triton autotuner OOM on Qwen3.5/Qwen3-Next GDN layers (non-SM90 GPUs) | ✅ Closed 2026-03-12, fix shipped via #36599 | Original report of first-inference OOM during Triton autotuning on non-SM90 hardware. Closed because the warmup fix landed. Reading thread is useful context for understanding the GDN kernel autotuner pressure on our hardware class. | n/a — fix in our image. |
| [#36599](https://github.com/vllm-project/vllm/pull/36599) — Warm up Triton autotuner for GDN layers during V1 profiling | ✅ Merged 2026-03-12 (in image SHA `7a1eb8ac`) | Adds `_warmup_triton_kernels()` at V1 profile phase. Warms with B=1, T=64 dummy tensors. Closes the boot-time first-inference autotuner OOM that #36598 reported. **DOES NOT close Cliff 2b** (multi-turn accumulated context) — the warmup uses T=64 but FLA kernels use `do_not_specialize=["T"]` so production T=4128 is the same autotune key, meaning runtime fragmentation isn't from missed autotune; it's from the per-shape Triton kernel binaries staying resident in CUDA context. Confirmed by Codex memo 2026-05-03. | n/a — fix in image; doesn't help our remaining cliff. |
| [#36973](https://github.com/vllm-project/vllm/issues/36973) — `_warmup_prefill_kernels` leaks ~3.4 GiB despite empty_cache | 🟡 Open, RTX 5090-specific | jhsmith409's report — Triton autotuner cubin retention initially suspected but haosdent comment #18-19 traced the bulk to **TMA overhead** scaling with SM count (~22 MiB/SM × 170 SMs on 5090 = 3.7 GiB). Closed via #37700 (TMA-disable for SM12x). **Doesn't apply to Ampere SM86** — no TMA hardware. Useful context though: thread comment #5 explicitly notes Triton autotuner keeps all variants loaded; `empty_cache()` only releases PyTorch's caching allocator, not CUDA-context cubins. | n/a — RTX 3090 doesn't have TMA. |
| [#37700](https://github.com/vllm-project/vllm/pull/37700) — Fix FLA Hopper/TMA misclassification on SM12x desktop Blackwell | 🟡 Open / closes #36973 for SM12x | Uses shared-memory threshold instead of `major >= 9` checks for TMA path selection. SM12x desktop Blackwell only — RTX 5090, DGX Spark GB10. Doesn't apply to Ampere SM86 (no TMA hardware). | n/a — different hardware family. |
| **Cliff 2b — multi-turn accumulated-context OOM (we filed)** | 🟡 Open, [Sandermage genesis-vllm-patches#19](https://github.com/Sandermage/genesis-vllm-patches/issues/19) | DeltaNet `chunk_gated_delta_rule_fwd` holds ~500 MiB of simultaneous live tensors at T=4128. Under multi-turn agent traffic (hermes/openhands/etc.), accumulated KV + this kernel's working set + model + workspace exceeds 24 GiB on 1× 3090. Cliff fires at ~21-26K accumulated context. We tested mem-util tuning, MTP-off, max-num-batched-tokens reduction, TRITON_CACHE_AUTOTUNING, expandable_segments, empty_cache between turns — none close it. Validated 2026-05-03: 6 single-card vLLM variants FAIL v2 continuous soak; only TP=2 / dual.yml passes. Filed with Sandermage proposing streaming refactor of GDN forward intermediates. | **`bash scripts/switch.sh vllm/dual` (TP=2)** for 2× rigs, **`llamacpp/default`** for 1× rigs. See [club-3090#41](https://github.com/noonghunna/club-3090/issues/41) + [docs/CLIFFS.md](CLIFFS.md) "Why TP=2 escapes" / "Why llama.cpp escapes" sections. |

---

## Genesis (`Sandermage/genesis-vllm-patches`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [#22](https://github.com/Sandermage/genesis-vllm-patches/issues/22) — PN59 streaming-GDN never engages on chunked-prefill ⚠️ | 🟡 **Open, filed 2026-05-05 by us** | Genesis v7.72.2 advertises PN59 as the structural Cliff 2b fix on 24 GB single cards, but its eligibility check rejects calls with `chunk_indices`/`chunk_offsets` populated — which vLLM's mandatory `--max-num-batched-tokens 4128` always sets. PN59 falls back to vanilla, OOMs at the same `chunk_o.py:161` site. Single-card 24 GB long-context (`long-text.yml` / `long-text-no-mtp.yml` / `long-vision.yml`) regresses vs the prior workarounds. | **Use `dual.yml` / `dual-turbo.yml` (TP=2)** or `llamacpp/default` (different engine, no Cliff 2b). Reproducer + 4 fix proposals in the issue body; awaiting Sander review. |
| [#5](https://github.com/Sandermage/genesis-vllm-patches/issues/5) — P8 ImportError on vLLM v0.20.0 | ✅ Closed (now on v0.20 pin since 2026-05-01) | Originally about P8 ImportError on the v0.20.0 GA tag. We migrated master to v0.20.1rc1.dev16 + Genesis v7.65 dev tip — P8 path no longer fires on our configs. | n/a — pin already moved. |
| [#6](https://github.com/Sandermage/genesis-vllm-patches/issues/6) — P65 PIECEWISE cost quantified | ✅ Closed | We characterized the +22 TPS narrative cost of P65 on Qwen3.6-27B + MTP. Sandermage acknowledged. Will recover when vllm#40914 lands. | Accept the cost on substrate-current; ampersandru's pre-P65 stack avoids it. |
| [#7](https://github.com/Sandermage/genesis-vllm-patches/issues/7) — P67 Triton CompilationError on Qwen3.6-27B | ✅ Closed | Resolved in v7.64 — P67 generalized to non-power-of-2 GQA via `BLOCK_QH = triton.next_power_of_2(HEADS_PER_KV)` + lane_valid mask. Tool-call 0/5 → 7/7 on 2× A5000 validation. | Now safe to enable on 27B configs with v7.64+. |
| [#9](https://github.com/Sandermage/genesis-vllm-patches/issues/9) — P68/P69 8000-char threshold breaks IDE agents | ✅ Closed 2026-05-01 — fix shipped in v7.65+ (50K-char default); we're on v7.69 so the fix lives in our pin | P68 silently rewrites `tool_choice: auto → required`; P69 injects "must use a tool" hint. New 50K threshold clears typical IDE-agent contexts (Cline ~30K / Cursor ~25K / Copilot ~20-25K) while genuine long-history sessions still trigger the reminder. | Composes still have P68/P69 env vars commented out — **enabling them across composes is queued for next-week revisit** (see "Active follow-ups" table above). Until then, manual override: `GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1`. |
| [#11](https://github.com/Sandermage/genesis-vllm-patches/issues/11) — Cliff 1 mech A FA2 softmax_lse clamp request | ✅ Closed (PN17 in v7.64; default-on across all TQ3 composes since the v0.20 migration) | Sandermage's PN17 lands the clamp at `flash_attn.py`. Active on every TQ3 compose. | n/a — default-on. |
| **Local P104 FA max_seqlen_k runtime clamp** | ✅ Dropped during v0.20 migration | Built 2026-04-30 as `patch_fa_max_seqlen_clamp.py`. Sandermage's PN17 + P15B together cover both layers (FA wrapper + TQ wrapper). Sidecar removed from compose mounts on 2026-05-01. | n/a — Genesis-native. |
| [PR #12](https://github.com/Sandermage/genesis-vllm-patches/pull/12) — P101 anchor drift fix | ✅ Closed; on v0.20 pin since 2026-05-01 | P101 anchor matches on `0.20.1rc1.dev16+g7a1eb8ac2`. | n/a — pin matches. |
| [PR #13](https://github.com/Sandermage/genesis-vllm-patches/pull/13) — PN12 anchor drift fix | ✅ Closed; on v0.20 pin since 2026-05-01 | PN12 anchors match natively. Local `patch_pn12_ffn_pool_anchor.py` sidecar removed. | n/a — pin matches. |
| [#14](https://github.com/Sandermage/genesis-vllm-patches/issues/14) — P38 silently no-op'd on TurboQuant KV path (we filed 2026-05-01) | ✅ Closed via P38B in Genesis v7.65 dev tip | Sandermage shipped **P38B** — text-patches `turboquant_attn.py` source to inject a delegate hook at the start of `_continuation_prefill` body. Active via `GENESIS_ENABLE_P38B_COMPILE_SAFE=1` on every TQ3 compose since 2026-05-01. | n/a — Genesis-native. |
| [#15](https://github.com/Sandermage/genesis-vllm-patches/issues/15) — FA varlen kernel workspace cliff at flash_attn_interface.py:300 (we filed 2026-05-01) | ✅ Closed via P15B in Genesis v7.65 dev tip | Sandermage shipped **P15B** — direct backport of our suggestion. Active via `GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP=1` on every TQ3 compose since 2026-05-01. Empirically the cliff also doesn't reproduce on v0.20 (vllm#40092 changed workspace allocator behavior) — covered from two directions. | n/a — Genesis-native. |
| [#16](https://github.com/Sandermage/genesis-vllm-patches/issues/16) — PN25 worker-spawn registration | 🟡 Sander shipped d92bcb3 (v7.65) + Library refactor (v7.66); both fail on TP=1 | v7.65 used `@torch.library.custom_op` (failed at `infer_schema` inside dynamo trace). v7.66 refactored to `direct_register_custom_op` + `Library("genesis", "FRAGMENT")` — fails at `instantiate_user_defined_class_object` inside dynamo trace. Same root cause: torch.library construction inside trace context disallowed on TP=1 spawn. Cross-rig data on Sander's [discussion #19 reply](https://github.com/noonghunna/club-3090/discussions/19#discussioncomment-16785590). | Local `patch_pn25_genesis_register_fix.py` v3 — text-patches `activation.py` to register at module-import time, BEFORE any trace. Survives both v7.65 and v7.66 mechanisms. PR-ready upstream. |
| [#17](https://github.com/Sandermage/genesis-vllm-patches/issues/17) — DS conv state spec-decode crash | 🟡 Sander shipped a9977d8 (PN30) but `.contiguous()` is layout-incorrect | Sander's PN30 materializes `state[src, :, offset:].contiguous()` (compact 10240×5) and raw-memcpys into `state[dest]` (strided 10240×6) → corrupts DS row strides → eventual TQ store CUDA assert several layers downstream. Diagnosis credit: ChatGPT/Codex CLI cross-check 2026-05-02. Sent corrected fix to Sander. | Local `patch_pn30_dst_shaped_temp_fix.py` — patches `collect_mamba_copy_meta` to build dst-shaped temp instead of compact. Reuses Sander's `_GENESIS_PN30_TEMP_TENSORS` lifecycle. Validated on all 4 TQ3 composes; probes 4 + 5 pass cleanly. PR-ready upstream. |
| [#15](https://github.com/Sandermage/genesis-vllm-patches/issues/15) — PN31 FA varlen persistent out | 🟡 Sander shipped 753344b (PN31, default OFF); doesn't fit on 24 GB | Per-shape persistent buffer growth + PN12+PN25 pool residence outpaces activation budget at DeltaNet `chunk_fwd_o` on 24 GB single GPU. Sander explicitly flagged he couldn't validate on 24 GB. | Lower mem-util to 0.95 — gives enough activation headroom to close the 25K tool-RETURN path PN31 was meant to fix, without needing PN31. Cross-rig data on Sander #15 comment. |
| **PN33 spec-decode warmup K-aware** (Sander v7.66 fc89395, default ON) | 🟡 Partial close on TP=1 | Backport of vllm#37521 EXTENDED to MTP/ngram. Sander claimed it closes both ampersandru's mid-stream OOM AND our workspace_lock AssertionError. Cross-rig 2026-05-02: closes BOOT-time profile_run workspace_lock ✅, but runtime decode `turboquant_attn.py:1350:_decode_attention` AssertionError still fires ❌. | Local `patch_workspace_lock_disable.py` sidecar still required for runtime decode. Drop when upstream covers the runtime path. |
| **P98 marker false-positive on v0.20** (we filed 2026-05-01 in [#9 thread](https://github.com/Sandermage/genesis-vllm-patches/issues/9#issuecomment-4359541875)) | 🟡 Side-noted to Sandermage; awaiting his call on fix | P98's drift detection auto-skips on v0.20 (`UNIFORM_SINGLE_TOKEN_DECODE` marker false-positive) but the strict workspace lock still fires rare paths P98 was supposed to revert. | Local `patch_workspace_lock_disable.py` sidecar (mounted on every TQ3 compose) relaxes the strict assertion to a one-shot WARNING. Drop when Sandermage ships either a marker fix or P98 with explicit env-override. |
| **PN30 v7.68 part3 drift-marker false-positive** (we filed in noonghunna/club-3090#19 cross-rig retest) | ✅ Closed in v7.69 (commit 2db18df) | Part3's `upstream_drift_markers=["[Genesis PN30"]` (generic prefix) matched markers parts 1+2 wrote on the same file. Part3 skipped as `upstream_merged` → apply_all FAILS → vLLM aborts. v7.69 tightened to `[Genesis PN30 v7.68 dst-shaped]` (specific). | n/a — fixed in v7.69. |
| **P103 setattr lost on `exec vllm serve`** (we filed in noonghunna/club-3090#19) | ✅ Closed in v7.69 | v7.68 P103's `setattr` ran in entrypoint shell but was lost on `exec vllm serve` worker spawn (process image replaced). v7.69 ships chunk.py self-install hook appended to end-of-file — survives any startup mechanism. | n/a — fixed in v7.69. |
| **PN32 v1 chunked at wrong level** (we filed in noonghunna/club-3090#19) | ✅ Closed in v7.69 (PN32 v2) | PN32 v1 chunked outer-level inputs but inner FLA call still got full-prompt cu_seqlens, allocating full h tensor regardless. v7.69 PN32 v2 patches `_forward_core` directly + threads `last_recurrent_state` between chunks. | n/a — fixed in v7.69. |
| [#18](https://github.com/Sandermage/genesis-vllm-patches/issues/18) — P103 cu_seqlens=[0,T] single-seq case is bypassed (we filed 2026-05-02 PM) | 🟡 Open / v7.70 proposal | P103's gate currently bypasses chunking for ANY non-None cu_seqlens, but `cu_seqlens.shape[0] == 2` (single sequence boundary) is semantically dense B=1, not multi-seq varlen. Fix admits the chunked path on real serving. Diagnosis: ChatGPT/Codex CLI. Cross-rig observation: P103 chunked path never engages on real config because vLLM's outer chunked-prefill caps T at `max_num_batched_tokens=4128` (well below `_MAX_T=16384`), so the gate-fix is semantically correct but doesn't independently close 60K Cliff 2 on TP=1+24GB. | n/a yet — gate fix queued for v7.70. Real Cliff 2 closure on this config comes from [vllm#35975 backport](https://github.com/vllm-project/vllm/pull/35975) + mem-util 0.93 (see vLLM section above + [`docs/CLIFFS.md`](CLIFFS.md)). |

---

## FlashAttention 2 (`Dao-AILab/flash-attention`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [#1011](https://github.com/Dao-AILab/flash-attention/issues/1011) — Variable memory allocation with varlen kernels | 🔴 Open since 2024, no fix | **Cliff 1 root cause.** `softmax_lse` is allocated as `[num_seqs, num_heads, max_seqlen]` — sized by `max_seqlen` parameter, NOT actual `cu_seqlens`. So a 25K-token chunked-prefill at `max_model_len=86K` allocates softmax_lse for 86K, not 25K. This is why Cliff 1 fires harder at higher max-ctx even when the actual prompt is the same. | None. Stay at `default` 48K (or `tools-text` 75K with PN8 mitigation). FA2 redesign of softmax_lse format would be the upstream fix. |

## flash-linear-attention (`fla-org/flash-linear-attention`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| **Cliff 2 — DeltaNet GDN forward OOM at 50–60K single-prompt** | 🔴 Open, **no upstream issue filed yet**. **Confirmed cleared on dual TP=2** (this rig, 2026-04-29 — see DUAL_CARD.md "237K single-prompt verified"). | The `chunk_gated_delta_rule_fwd` kernel allocates intermediate buffers proportional to `seq_len`. Fires on single-card regardless of mem-util. On dual TP=2 the activation memory splits across cards and the cliff doesn't fire — verified at 237K single-prompt prefill on `dual.yml` (~830 tok/s prefill, matches Sandermage's 262K @ 311s on 2× A5000). Sandermage explicitly punted on the single-card fix (genesis-vllm-patches issue #1: *"can't fix this short of multi-GPU TP=2 or upstream fla.ops changes"*). Likely the same architectural pattern as FA#1011 — recurrent state buffer pre-allocated by max_seq_len. | Single-card: use `tools-text.yml` (75K cap) or `llamacpp/default` (262K, different engine). Dual: `dual.yml` clears at ≥237K. |

---

## FlashQLA (`QwenLM/FlashQLA`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| **Ampere SM 8.6 / Ada SM 8.9 port** | 🔴 No issue filed; tweet to @QwenLM drafted but not yet posted | FlashQLA is QwenLM's TileLang DeltaNet kernels — would fix Cliff 2 if it ran on Ampere. Currently SM90+ only. | None. Watch the repo for Ampere support; revisit when an issue is filed and a port is on the roadmap. |

---

## Luce DFlash (`Luce-Org/lucebox-hub`) — separate llama.cpp fork (NOT our vLLM dual-dflash)

**Heads-up — naming clarification:**
- This section tracks **`Luce-Org/lucebox-hub`** (a llama.cpp fork from Luce). **As of 2026-05-04 this is no longer single-card-only** — see "Dual-GPU split landed" below.
- Our **`docker-compose.dual-dflash.yml` / `dual-dflash-noviz.yml`** (vLLM TP=2 dual-card) IS shipping and is the recommended DFlash path on this stack today. Both consume the **same draft model** (`z-lab/Qwen3.6-27B-DFlash`), but the engine + topology differ. Don't confuse the two.

### 🆕 Dual-GPU split landed (2026-05-02 + 2026-05-04)

Two @weicj PRs shipped that change the lucebox-hub serving topology. **Target weights on one GPU + DFlash draft (or PFlash drafter) on a separate GPU** — heterogeneous spec-decode, not weight-sharded TP. Each model lives entirely on its own card; they communicate at spec-decode boundaries via peer copies.

- [**lucebox-hub PR #80** — `bench(dflash): add dual-GPU target/draft split harness`](https://github.com/Luce-Org/lucebox-hub/pull/80) (merged 2026-05-04). New flags `--target-gpu` / `--draft-gpu` (also `DFLASH_TARGET_GPU` / `DFLASH_DRAFT_GPU` env). Validation on dual RTX 2080 Ti 22 GB: HumanEval 10-prompt at **51.86 tok/s, AL 7.09, 44.3% accept** on Qwen3.5-27B Q4 target + z-lab DFlash draft.
- [**lucebox-hub PR #78** — `bench(pflash): add dual-GPU PFlash phase-split harness`](https://github.com/Luce-Org/lucebox-hub/pull/78) (merged 2026-05-02). New flag `--pflash-gpu` + persistent `pflash_daemon`. Validation on same hardware: **single-GPU co-resident passes NIAH at 24,573 source tokens; dual-GPU phase split passes at 262,125 source tokens (10.7×).** Compressed context reaches 13,229 tokens at 262K source.

**Implication for our 2× 3090 stack:** the single-card limitations we documented (65K max_ctx, draft VRAM competing with target activations) are addressed by dual-GPU split. Target Qwen3.5-27B Q4_K_M gets a full 24 GB on GPU 0; DFlash draft + PFlash drafter live on GPU 1. **No NCCL/allreduce overhead per token** since each model lives entirely on its own card — should be faster per-stream than SGLang TP=2 + DFlash for single-stream workloads. Bench tracked at task #229 (queued, not yet executed locally — PR #80 is hours old as of this entry). **Qwen3.6-27B draft remains under training** so the dual-GPU benefit applies primarily to the stable Qwen3.5-27B + DFlash pair today.

Re-benched 2026-04-30 PM on Qwen3.6-27B Q4_K_M + matched z-lab/Qwen3.6-27B-DFlash draft (under training). Open issues against single-card lucebox-hub follow:

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) — draft model still under training | 🟡 Snapshot 2026-04-26 | Narrative AL ~3.7, code AL ~7.0 on `Luce-Org/lucebox-hub` (single-card llama.cpp). When training finishes, expected to climb toward Qwen3.5 reference (8.31 HE, 7.04 Math). The same caveat applies to vLLM `dual-dflash.yml` — published 82/125 TPS in [`docs/DUAL_CARD.md`](DUAL_CARD.md) was measured against this 2026-04-26 snapshot at peak code-prompt conditions; AL on real agent traffic will be lower until z-lab tags training-complete. | Re-test when z-lab tags training-complete. The vLLM dual-dflash path remains shipping — see [DUAL_CARD.md](DUAL_CARD.md) — but treat its numbers as a snapshot. For autonomous coding agents on dual-3090 today, `dual.yml` (FP8 + MTP) is the recommended robust path. |
| **Build fragility on `dflash` main HEAD** | 🔴 Reproducible 2026-04-30 PM | `cmake --build` errors with `ggml_turbo_wht` and `GGML_TYPE_TQ3_0` undefined. Required submodule commit `b6ffab4a9` not auto-fetched. Cross-rig signal — fresh clone fails. | After clone: `cd dflash/deps/llama.cpp && git fetch origin && cd ../../.. && git submodule update --init`. |
| **Daemon-mode "empty prompt" regression** | 🔴 Reproducible 2026-04-30 PM | After streaming requests, subsequent requests return `"empty prompt"` from the test_dflash daemon. Server keeps accepting requests but generates 0 tokens. Forces restart. | Restart server between request flavors; avoid mixing streaming + non-streaming. |
| **`enable_thinking` chat_template_kwargs honored differently than vLLM** | 🟡 Behavioural difference | Test sends `enable_thinking=true` and expects `reasoning_content` populated. Luce returns `content` directly. Not a missing feature, but breaks our `verify-full.sh` check 6. | Don't treat the thinking-mode test as a Luce-correctness signal until the chat-template path is documented. |
| **Greedy only** | 🟡 Documented limitation | `temperature` / `top_p` accepted but ignored. Real downside for creative-writing workloads. | Use vLLM long-text/long-vision when sampling matters. |
| **Prefill OOM in `fattn-chunked.cu` on 25K+ prompts at Q8_0 KV** | 🟡 Open (configuration trade) | Chunked flash-attention CUDA OOMs on large prefill at default Q8_0. **TQ3 KV (`DFLASH27B_KV_TQ3=1`) closes it** at max_ctx=65K — verify-stress passes 791 chars / finish=stop. Higher max_ctx (131K) reopens it. | Always set `DFLASH27B_KV_TQ3=1` for stress-test-passing config. Cap max_ctx at ~65K. |
| [**PFlash — long-context prefill accelerator**](https://www.lucebox.com/blog/pflash) (sibling tech to DFlash, same Luce-Org/lucebox-hub repo) | 🟢 **Public release 2026-04 + dual-GPU split shipped 2026-05-02 (PR #78)** | **Speculative prefill + block-sparse attention.** Compresses 128K prompts to ~6.5K tokens (`keep_ratio=0.05`) before target prefill. Single-card claimed: TTFT 24.8s vs 257s vanilla llama.cpp at 128K (~10.4× speedup). **Dual-GPU phase split (PR #78) extends the passing source-context ceiling from ~24K (single-card co-resident) to 262K (~10.7×) on dual 22 GB cards** — NIAH key/answer retained at 262K. C++/CUDA only, lives inside the lucebox-hub server stack. PFlash sits *in front of* DFlash decode: PFlash accelerates prefill, DFlash accelerates generation. **For 2× 3090 deployments**: pin PFlash drafter to GPU 1 via `--pflash-gpu`, target on GPU 0. The single-card-coresident limit (was the binding blocker for our use) no longer applies. MIT license. **Open exploration**: bench PFlash + DFlash dual-GPU vs vLLM `dual-dflash.yml` (185K, 82/125 TPS on 2× 3090) on TTFT-bound workloads. Tracked at task #229. | Re-evaluate as a club-3090 shipping option once we (a) reproduce the 262K passing source-ctx claim on 2× 3090 with verify-stress + soak-continuous + bench, OR (b) an upstream-vLLM port lands. The dual-GPU split removes the single-card co-residency blocker; remaining blockers are daemon-mode bugs (greedy-only, no vision, "empty prompt" regression) carried over from the single-card history. |

---

## llama.cpp (`ggml-org/llama.cpp`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [PR #21089](https://github.com/ggerganov/llama.cpp/pull/21089) — TurboQuant KV mainline | 🟡 Open (CPU first, CUDA follow-on) | When CUDA path lands, `turbo3` becomes a first-class option on llama.cpp. Naming will migrate from `turbo3` → `tbq3_0`. | Use Tom's fork for now: [llama-cpp-turboquant](https://github.com/tdraxl/llama-cpp-turboquant). |
| **Q3_K_XL TPS regression (28.5 TPS @ 262K → 21 TPS today)** | 🔴 Suspected, no upstream issue filed | Measured 2026-04-23 vs 2026-04-28: same model, same hardware, 28.5 TPS dropped to 21 TPS between commits `9ab47e7d8` and `0d0764dfd`. Bisect or file. | None — we're on the slower commit. Tracked in [club-3090 TODO](https://github.com/noonghunna/club-3090) (private). |

---

## transformers (`huggingface/transformers`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [#45283](https://github.com/huggingface/transformers/issues/45283) — Qwen3.5 GGUF support | ❌ Closed without fix 2026-04-28 (no associated PR; closing event has `source: null`, last comment was just `cc @SunMarc` — looks won't-fix or stale-bot) | Was tracked as the missing piece (alongside vllm#38140 / vllm#37797) for Qwen3.5/3.6 GGUF on vLLM/SGLang. Won't be picked up via transformers — **llama.cpp remains the only GGUF path** for this model family. | llama.cpp path. Don't expect a vLLM/SGLang GGUF route for Qwen3-Next family. |

---

## SGLang (`sgl-project/sglang`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| **Same Marlin pad-sub-tile-n bug as vllm#40361** | 🔴 Not filed; same kernel-line fix applies | Blocks Lorbus INT4 + EAGLE on SGLang. We haven't filed an SGLang PR. | None on SGLang. Use vLLM (with our patched fork) or wait for SGLang to pick up the upstream Marlin fix. |
| **DeltaNet KV rollback (vllm#39931 cross-engine)** | 🔴 Same architectural issue | Blocks EAGLE on Qwen3-Next family in SGLang too. | None — see vllm#39931. |

---

## Filing conventions

When you file or learn of a new upstream issue:

1. **Add a row** to the appropriate section of this file. Include the link, status emoji, one-line "why it matters," and the local workaround (if any).
2. **Cross-link** from any code, compose comment, or doc that depends on the workaround back to the row in this file (e.g., `# See docs/UPSTREAM.md — vllm#40361`).
3. **Update the row** when status changes — closed, merged, propagated, replaced. Don't delete; if a row is no longer load-bearing, mark it ✅ Resolved or ❌ Closed without fix and leave it as historical context.
4. **Bump the relevant pin** when an upstream lands (Genesis commit, vLLM nightly, llama.cpp commit). Add a CHANGELOG entry citing the upstream PR.

When you file an issue against an upstream repo from this work, **link back to club-3090** in the body so the upstream maintainer can see the affected user surface and re-test if needed.

---

## Related reading

- [`models/qwen3.6-27b/INTERNALS.md`](../models/qwen3.6-27b/INTERNALS.md) — model-specific deep dives (DFlash forensics, MTP head, AutoRound rationale)
- [`models/qwen3.6-27b/vllm/patches/README.md`](../models/qwen3.6-27b/vllm/patches/README.md) — local patches (tolist, Marlin pad fork, Genesis env-var matrix)
- [`AGENTS.md`](../AGENTS.md) — repo-wide conventions, including the rule that this file is the upstream-tracking single source of truth
