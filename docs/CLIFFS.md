# Prefill cliffs on Qwen3.6-27B / single 3090 — full synopsis

Comprehensive deep-dive into Cliff 1 and Cliff 2: what they are, why they fire, what's actually happening at the hardware/library level, what we've tried, what works, and what's left. Last revised 2026-04-29 after FA2 root-cause bisection.

This document supersedes earlier characterizations in CHANGELOG and FAQ where they conflict — those have been corrected to match what's documented here.

> **Cross-reference:** Sandermage's Genesis tree maintains a broader catalog of [8 cliffs](https://github.com/Sandermage/genesis-vllm-patches/blob/main/docs/CLIFFS.md) covering patch behavior across configs and pin versions: Cliff 1 (FA2 lse), Cliff 2 (GDN fwd_h), Cliff 3 (TQ + spec-verify K+1 + FULL cudagraph), Cliff 4 (non-pow-2 GQA + P67), Cliff 5 (ngram strict prompt_lookup_min), Cliff 6 (MoE v0.20+ for non-FP8, [vllm#41306](https://github.com/vllm-project/vllm/pull/41306)), Cliff 7 (DFlash 24 GB OOM at >80K), Cliff 8 (anchor drift on vLLM pin bumps). Our doc here focuses on Cliff 1 + Cliff 2 with the Qwen3.6-27B / RTX 3090 forensics that motivated those Genesis cliffs.

---

## TL;DR

**Three distinct OOM "cliffs"** fire on a single 24 GB RTX 3090 when serving Qwen3.6-27B with vLLM + Genesis patches. They affect different workload patterns:

| | **Cliff 1** | **Cliff 2a (single-prompt)** | **Cliff 2b (multi-turn)** ⭐ NEW |
|---|---|---|---|
| Trigger | 25K+ token tool messages → chunked prefill | Single prompt > ~50–60K tokens | ~21-26K **accumulated** multi-turn context (hermes/openhands/Cline/etc.) |
| OOM site | `_vllm_fa2_C.varlen_fwd` (FlashAttention 2) | `fla.ops.chunk_gated_delta_rule_fwd → chunk_fwd_o → empty_like(v)` | Same kernel as 2a, fires earlier under multi-turn KV pressure |
| Root cause | `softmax_lse` padded to max_seqlen | GDN forward live-tensor cascade (~500 MiB at T=4128) | Same cascade; multi-turn KV accumulation eats activation headroom |
| Allocation requested at OOM | varies | 50 MiB | 38-50 MiB (per chunk, not scaled by ctx) |
| Affects | TQ3 paths at large prompts | All single-card vLLM at long single prompts | **All single-card vLLM under accumulating-context agentic traffic** |
| Closed status | **CLOSED** ✅ — PN25 + PN30 in v7.69, PN17 (FA clamp) in v7.64 | **MOSTLY CLOSED** ✅ — v7.69 + vllm#35975 + 0.93 mem-util passes 60K | **NOT CLOSED** ❌ — fires in 4-5 turns of agentic traffic regardless of config |
| Our mitigation | Genesis PN17/PN25 (default-on) | mem-util 0.93, max-model-len ≤180K | **`vllm/dual` (TP=2)** OR **`llamacpp/default`** — see "Why those escape" below |
| Real fix | Already shipped | Already shipped | **Streaming refactor of `chunk_gated_delta_rule_fwd`** — being filed with Sandermage |

**Practical impact today:** Cliff 1 is closed. Cliff 2a is closed for typical workloads (single prompts up to ~60K). **Cliff 2b is open and bites every user running an agentic coding client on single-card vLLM** — see [#41](https://github.com/noonghunna/club-3090/issues/41) for the full validation matrix.

---

## Why TP=2 escapes Cliff 2b (and Cliff 2a above 60K)

`chunk_gated_delta_rule_fwd` allocates intermediate tensors shaped `[1, T, H, D]` where `H=48` value heads at TP=1. With TP=2, the heads dimension is sharded — each card sees `H=24`. **Same kernel runs on each card, but per-card live-tensor sizes are halved.**

| Tensor | Size at TP=1 (H=48) | Size at TP=2 (H=24 per card) |
|---|---|---|
| v, u, v_new, o, w | 48 MiB each (×5 = 240) | **24 MiB each (×5 = 120)** |
| A | 48 MiB | **24 MiB** |
| Ai | 24 MiB | **12 MiB** |
| h | 97 MiB | **49 MiB** |
| Total per-card live FLA set | ~500 MiB | **~250 MiB** |

Plus per-card model weights drop from ~14 GB to ~7 GB (sharded), KV cache from full to half, etc. Per-card peak stays comfortably below 24 GB even with 25K accumulated multi-turn context. **Validated 2026-05-03**: dual.yml passed 5×5 v2 continuous soak with 0 errors / 0 MiB growth, 23.5 GB per-card steady, 111 TPS p50 decode.

Cost paid for this: NCCL allreduce per layer between cards (~30-50µs/token on PCIe), ~10-20% TPS overhead vs single-card if single-card actually worked.

## Why llama.cpp escapes Cliff 2b on a single card

llama.cpp uses **different kernels and a different memory allocator** than vLLM. Three concrete differences:

1. **Different GDN kernel implementation.** vLLM uses `flash-linear-attention` (fla-org) Triton kernels — those are the ones with the simultaneous live-tensor cascade. llama.cpp ships its own hand-written CUDA implementation of DeltaNet attention with smaller per-step working buffers and more sequential intermediate computation.

2. **Different memory allocator.** vLLM uses **PyTorch's caching allocator** (designed for training where shapes change a lot — caches freed blocks for fast reuse, which causes our fragmentation under fixed-shape inference). llama.cpp uses **ggml's manual memory management** — pre-allocates fixed buffers per op type at boot, no caching layer, no fragmentation accumulation. For fixed-shape inference, ggml's flat allocator is more predictable on tight VRAM.

3. **No JIT / no Triton autotune.** vLLM relies on `torch.compile` + Triton autotune + lazy cudagraph capture — new shapes at runtime trigger new compilations that pin memory. llama.cpp ships pre-compiled CUDA kernels — memory layout is static from boot.

Trade: llama.cpp gives up ~3× decode speed (21 TPS vs 67 on vLLM single when single works) for cliff-immunity at long context. Different engineering posture for different workload shapes.

---

## vLLM pin compatibility status (master shipped on v0.20 + Genesis v7.66 — 2026-05-02)

**Master now ships on `vllm/vllm-openai:nightly-7a1eb8ac2ec4ea69338c51dc7afd4b15010abfa8` (`0.20.1rc1.dev16+g7a1eb8ac2`) + Genesis v7.66 dev tip (commit `fc89395`).** Cliff 1 mech B (inductor compile-path FFN intermediate buffer leak) is **closed** across all 4 TQ3 composes via PN25 v3 (local import-time backport for TP=1) + PN30 dst-shaped temp fix (local correction for Sander's compact `.contiguous()`). verify-stress.sh 7-probe ladder: 6/7 pass on every TQ3 variant; only Cliff 2 architectural fails.

### Cliff 1 mech B — what closed it

---

## vLLM pin compatibility status (master shipped on v0.20 + Genesis v7.66 — 2026-05-02)

**Master now ships on `vllm/vllm-openai:nightly-7a1eb8ac2ec4ea69338c51dc7afd4b15010abfa8` (`0.20.1rc1.dev16+g7a1eb8ac2`) + Genesis v7.66 dev tip (commit `fc89395`).** Cliff 1 mech B (inductor compile-path FFN intermediate buffer leak) is **closed** across all 4 TQ3 composes via PN25 v3 (local import-time backport for TP=1) + PN30 dst-shaped temp fix (local correction for Sander's compact `.contiguous()`). verify-stress.sh 7-probe ladder: 6/7 pass on every TQ3 variant; only Cliff 2 architectural fails.

### Cliff 1 mech B — what closed it

Two compounding fixes:

1. **PN25 v3 import-time backport** (`patch_pn25_genesis_register_fix.py`) — text-patches `vllm/model_executor/layers/activation.py` to register `silu_and_mul` as a `torch.library.custom_op` at module-import time, BEFORE any dynamo trace context exists. Required because Sander's upstream PN25 (both v7.65 `@custom_op` and v7.66 `direct_register_custom_op` mechanisms) fails inside the dynamo trace on TP=1 spawn with `infer_schema` / `instantiate_user_defined_class_object`. Our approach moves registration entirely outside the trace and works regardless of which underlying mechanism is used.

2. **PN30 dst-shaped temp fix** (`patch_pn30_dst_shaped_temp_fix.py`) — corrects Sander's PN30 `a9977d8` which materialized a compact `.contiguous()` source-tail and raw-memcpy'd it into a strided destination, corrupting DS conv state row strides on every offset>0 copy. Our fix builds a dst-shaped temp inside `collect_mamba_copy_meta` (where both source AND destination block IDs are known) and does the strided copy correctly. Diagnosis credit: ChatGPT/Codex CLI cross-check.

Both fixes apply at setup time via `bash scripts/setup.sh qwen3.6-27b`. They survive Genesis pin bumps as long as the upstream code anchors don't shift dramatically.

### Genesis v7.66 dev tip — relevant patches active

| Patch | What it does |
|---|---|
| **PN12** | FFN intermediate scratch pool (Cliff 1 mech B fix on TQ3 — eager path) |
| **PN17** | FA2 softmax_lse runtime clamp — frees 50-100 MiB on long-ctx |
| **PN25** (Sander v7.66) | direct_register_custom_op refactor — superseded on TP=1 by our local v3 import-time patch |
| **PN26b** | First public sparse-V Triton kernel for SM86 (Ampere consumer) |
| **PN30** (Sander a9977d8) | DS conv state spec-decode fix — superseded by our local dst-shaped temp fix (Sander's compact `.contiguous()` corrupts DS row strides) |
| **PN33** (NEW v7.66, default ON) | Spec-decode warmup K-aware (vllm#37521 backport extended to MTP/ngram). Closes boot-time workspace_lock issue but NOT the runtime decode path on TP=1 |
| **P38B** | In-source hook for `_continuation_prefill` (Genesis #14 fix) |
| **P15B** | FA varlen `max_seqlen_k` clamp at TQ wrapper boundary (Genesis #15 fix) |

### Local sidecars retained on master

| Sidecar | Reason |
|---|---|
| `patch_pn25_genesis_register_fix.py` | Sander's PN25 mechanisms (v7.65 `@custom_op` AND v7.66 `Library`) both fail inside dynamo trace on TP=1. Our v3 registers at activation.py import time, outside any trace. Drop when upstream provides a TP=1-compatible registration path. |
| `patch_pn30_dst_shaped_temp_fix.py` | Replaces Sander's compact `.contiguous()` with a dst-shaped temp that preserves DS row strides. Drop when upstream lands the corrected approach. |
| `patch_workspace_lock_disable.py` | PN33 closes boot-time `profile_run` workspace_lock but not runtime `turboquant_attn.py:1350:_decode_attention`. Drop when upstream has a fix that covers the runtime decode path. |
| `patch_tolist_cudagraph.py` | cudagraph capture fix, unchanged from earlier rounds. |

### Genesis v7.68 dev tip (`18e65e3`) — TESTED 2026-05-02, NOT ADOPTED

Cross-rig validation of v7.68 on the `v7.68-cliff2-test` branch (pushed to origin for reference). Sander accepted our 3 cross-rig sidecars as Genesis-native; we attempted the drop but hit 3 regressions on TP=1 + 24GB:

| v7.68 patch | Result | Detail |
|---|---|---|
| **PN25 v7.68** (replaces our `patch_pn25_genesis_register_fix.py`) | ✅ works on TP=1 | `direct_register_custom_op` + `Library("genesis", "FRAGMENT")` survives worker spawn. FFN intermediate pool active across all dynamo traces. |
| **PN34** (replaces our `patch_workspace_lock_disable.py`) | ✅ works (env-opt-in) | Default OFF; needs `GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX=1`. Without it, runtime decode hits `AssertionError: Workspace is locked` at `turboquant_attn.py:1350`. |
| **PN30 v7.68** (replaces our `patch_pn30_dst_shaped_temp_fix.py`) | ❌ broken | part3's `upstream_drift_markers=["[Genesis PN30"]` (generic prefix) matches markers parts 1+2 wrote on the same file. Part3 skips as "upstream_merged" → apply_all FAILS → vLLM aborts. Needs part3-specific marker. |
| **P103** (FLA Cliff 2 chunked fwd_h+fwd_o) | ❌ broken | Wrap reports "rebound at 0 caller sites" — never intercepts. Cliff 2 OOM trace passes through `chunk_gated_delta_rule_fwd` directly. v7.68 commit `5743c03` fixed a NameError but the binding mechanism still has 0 callers on Qwen3.6-27B. |
| **PN32** (GDN chunked-prefill) | ❌ insufficient on TP=1 | Chunks `gdn_linear_attn.forward_cuda` but the inner FLA `chunk_gated_delta_rule_fwd` still allocates the full h tensor. Without PN30 (broken above) the activation budget is so tight Cliff 2 fires at 30K instead of 50-60K. |

**Verdict:** master stays on v7.66 (`fc89395`) + our 3 sidecars. Branch `v7.68-cliff2-test` retained as a snapshot for cross-rig data. Re-evaluate when Sander cuts v7.69 with these 3 fixes applied.

Full findings in [`results/v0.20-migration/v768-cliff2-test.summary`](../results/v0.20-migration/v768-cliff2-test.summary).

### Genesis v7.69 dev tip (`2db18df`) + vllm#35975 backport — Cliff 2 60K CLOSED 2026-05-02 PM ⭐

After v7.68's regressions, Sander cut v7.69 (commit `2db18df`) with proper fixes for all 3 of our cross-rig findings. We retested + did a 6-round bisect with ChatGPT/Codex CLI to find the actual Cliff 2 closure recipe. **60K Cliff 2 is now closed on TP=1 + 24GB.**

#### v7.69 patch status (verified working)

| v7.69 patch | Result | Detail |
|---|---|---|
| **PN30 v7.68 part3 drift-marker** | ✅ FIXED | v7.69 tightened part3's `upstream_drift_markers` to `[Genesis PN30 v7.68 dst-shaped]` (specific to part3's own marker). DS layout now active throughout, all 3 sub-patches APPLY clean. |
| **P103 self-install hook** | ✅ FIXED | v7.69 ships chunk.py self-install hook appended to end-of-file. Survives `exec vllm serve` worker spawn. The "rebound at 0 caller sites" log message in v7.68 was misleading. |
| **PN32 v2 _forward_core chunked-prefill** | ✅ FIXED | Rewritten to chunk at `_forward_core` directly + thread initial_state via `last_recurrent_state`. Composes with P103. |
| **Codex P103 cu_seqlens gate fix** | 🟡 Sent to Sander as v7.70 proposal | [Genesis issue #18](https://github.com/Sandermage/genesis-vllm-patches/issues/18). Semantically correct (cu_seqlens=[0,T] is dense single-seq, not multi-seq varlen) but doesn't independently close real-config Cliff 2 because vLLM's outer chunked-prefill caps T at 4128 — well below P103's MAX_T=16384 — so the chunked path never fires regardless. |

#### The actual Cliff 2 closure: vllm#35975 backport + mem-util tuning

ChatGPT/Codex round 2 diagnosed that the 50 MiB OOM at `chunk_fwd_o:torch.empty_like(v)` after ~394 successful T=4128 chunks on a 60K prompt is **steady cumulative state** (model weights + KV pool + Mamba conv state + Genesis pools + activation residue), not a single allocation that needs splitting. Closure requires freeing residency, not chunking activation.

[vllm#35975](https://github.com/vllm-project/vllm/pull/35975) (open) skips `inputs_embeds` GPU buffer for text-only models — saves ~64 MiB GPU + 64 MiB pinned CPU. We backported it as a setup-time text-patch (`patch_inputs_embeds_optional.py`) — frees ~444 MiB at boot on Qwen3.6-27B (both `gpu_model_runner.py` and `llm_base_proposer.py` call sites compound).

Combined with mem-util tuning, the matrix:

| Config | Boot resident | 60K MTP-on | Wall | Notes |
|---|---|---|---|---|
| 0.95 (baseline) | 23,164 MiB | ❌ OOM 50/24.5 free | n/a | Cliff 2 fires |
| 0.95 + #35975 | 22,720 MiB | ❌ OOM 50/46.5 free | n/a | 444 MiB freed at boot, only 22 MiB at peak |
| 0.92 + #35975 | 21,980 MiB | ✅ HTTP 200 | 689s | Cliff 2 closed, ~580 MiB end-of-run margin |
| **0.93 + #35975** | **22,260 MiB** | **✅ HTTP 200** | **623s** | **Best balanced point.** ~494 MiB margin, AL=4.00 |

Plus MTP-off + 0.95: 60K passes in 504s with full 5+ GiB KV pool — separate "max-context safety" variant for users who want max KV admission over decode TPS.

#### Two shippable variants on master post-v7.69

| Variant | File | max_model_len | mem-util | MTP | Cliff 2 60K | Use case |
|---|---|---|---|---|---|---|
| **Balanced MTP** | [`long-text.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.long-text.yml) | 180K | 0.93 | ✅ K=3 | ✅ 623s | Default — multi-turn agentic coding |
| **Max-context safety** | [`long-text-no-mtp.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.long-text-no-mtp.yml) (NEW) | 200K | 0.95 | ❌ off | ✅ 537s | Long single-shot RAG / codebase analysis |

Both top out at 60K Cliff 2 ceiling — that's the hardware-physical wall on 24 GB single-card. 90K probes hit OOM (Balanced MTP) or fail to complete in 25-min budget (Max-context). For prompts >60K, route to `dual.yml` (TP=2 splits state) or `llamacpp/default` (262K, different engine).

#### Local sidecars on master (post-v7.69)

| Sidecar | Reason |
|---|---|
| `patch_inputs_embeds_optional.py` | Backports vllm#35975 (~444 MiB savings on text-only Qwen3.6-27B). **Drop when PR merges upstream.** |
| `patch_tolist_cudagraph.py` | cudagraph capture fix (vllm#40807). Unchanged. |

The 3 prior sidecars (`patch_pn25_*`, `patch_pn30_dst_shaped_*`, `patch_workspace_lock_disable.py`) are **dropped** — Sander accept-and-folded all three into v7.68/v7.69 (PN25 v7.68 + PN30 v7.68 + PN34).

Full diagnostic trail (6-round bisect): [`results/v0.20-migration/v769-codex-r1-test.summary`](../results/v0.20-migration/v769-codex-r1-test.summary).

### Validation across all 4 TQ3 variants (2026-05-02 on Genesis v7.66 + local sidecars)

| Variant | Ctx | mem-util | verify-stress.sh probes | Notes |
|---|---|---|---|---|
| `long-text.yml` | 180K | 0.95 | 6/7 (Cliff 2 only fail) | IDE-agent recommended |
| `long-vision.yml` | 145K | 0.95 | 6/7 (Cliff 2 only fail) | Vision tower tightens budget further |
| `bounded-thinking.yml` | 180K | 0.95 | 6/7 (Cliff 2 only fail) | Parity with long-text. Recommended grammar: DeepSeek scratchpad. |
| `dual-turbo.yml` | 262K | 0.85 (TP=2) | 6/7 (Cliff 2 only fail) | TP=2 doesn't avoid Cliff 2 |
| `tools-text.yml` | 75K | 0.97 | ✅ | ✅ 8/8 | ✅ | ✅ | 69.66 (CV 1.4%) |
| `dual-turbo.yml` | 262K | 0.85 | ✅ | ✅ 8/8 | ✅ | ✅ | 76.01 (CV 4.5%, 269 TPS aggregate at n=4 streams) |

### Context restored on v0.20

- `long-vision.yml`: 140K → **198K** (+41%)
- `long-text.yml`: 185K → **214K** (+16%)
- `bounded-thinking.yml`: 185K → **214K** (+16%)

### v0.20 baseline checklist (verified)

- PyTorch 2.11 + CUDA 13.0.2 default ([#34644](https://github.com/vllm-project/vllm/pull/34644), [#40669](https://github.com/vllm-project/vllm/pull/40669)) — host on 595.58.03 + CUDA 13.2, fine
- Transformers v5 baseline ([#30566](https://github.com/vllm-project/vllm/pull/30566)) — fine
- CUDAGraph memory profiling default-ON ([#38284](https://github.com/vllm-project/vllm/pull/38284)) — disabled via `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` to recover ~120 MiB KV pool

---

## How we characterize "a cliff"

Both cliffs share a profile:

1. **Sudden CUDA OOM mid-prefill** (not at boot). Engine boots clean, accepts the request, processes the first ~N tokens fine, then dies on a small allocation — typically 50–138 MiB.
2. **Free VRAM at fault is < the requested allocation**, with a few hundred MB of "reserved but unallocated" PyTorch memory.
3. **Lowering `--gpu-memory-utilization` doesn't always fix it** — can shift the threshold or fail to boot entirely.
4. **The same prompt that fires on long-ctx config passes cleanly on shorter-ctx config** — same hardware, same model, same Genesis patches.

The two cliffs differ in *which workload triggers them* and *which library handles the failing allocation*.

---

## Cliff 1 — FA2 softmax_lse padded by max_seqlen

### What you see

Symptom on `long-vision.yml` (192K + 0.98) when an IDE agent like Cline, Cursor, OpenCode, or Continue.dev returns a tool message exceeding 25K tokens (a `read_file` of a big source file, a `web_fetch` of a long page, a `grep_search` returning many matches):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 50.00 MiB.
GPU 0 has a total capacity of 23.56 GiB of which 30.50 MiB is free.
22.67 GiB allocated, 24.00 MiB in private pools (CUDA Graphs),
508 MiB reserved by PyTorch but unallocated.

  File ".../vllm/v1/attention/backends/turboquant_attn.py", line 855, in _continuation_prefill
    return self._flash_attn_varlen(...)
  File ".../vllm/v1/attention/backends/turboquant_attn.py", line 340, in _flash_attn_varlen
    return flash_attn_varlen_func(...)
  File ".../vllm_flash_attn/flash_attn_interface.py", line 300, in flash_attn_varlen_func
    out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(...)
```

The container OOM-dies, the streaming response truncates mid-flight, and the user sees an empty or partial response.

### When it fires

| Pattern | Fires? |
|---|---|
| Steady chat accumulation to 150K+ via many small turns | ❌ no — each turn's prefill is small |
| Tool-using agent: 30K-token `read_file` result | ✅ yes — chunked prefill of the tool message hits FA workspace allocation |
| RAG: stuffing a 100K-token document into the user message | ⚠️ Cliff 2 fires first at ~50–60K |
| Single 25K-token user message (no tool) | ✅ yes — same chunked prefill mechanics |

The unifying trigger: **any prefill batch large enough that vLLM does chunked prefill (>= max-num-batched-tokens = 4128 in our default) on a config with high max-model-len.**

### Empirical bisection (2026-04-29 on RTX 3090)

| Config | Boot VRAM | KV pool | 25K tool prefill |
|---|---|---|---|
| 48K + 0.92 + TQ3 + vision (default) | 20.8 GB used, 1.87 GiB / 148K tok | ✅ passes |
| 86K + 0.92 + TQ3 + vision (engine ceiling for 0.92) | 20.8 GB used, 1.87 GiB / 148K tok | ❌ fails — 50 MiB allocate, 30.5 MiB free |
| 96K + 0.98 + TQ3 + vision | 22.3 GB used, 3.28 GiB / 260K tok | ❌ fails at 30K longctx rung |
| 128K + 0.98 + TQ3 + vision | 22.3 GB used, 3.28 GiB / 260K tok (same!) | ❌ fails |
| 192K + 0.98 + TQ3 + vision (current `long-vision.yml`) | 22.3 GB used, 3.28 GiB / 260K tok | ❌ fails |
| 96K + 0.92 + TQ3 + vision | **boot fails** — 96K exceeds engine ceiling at 0.92 |  |
| 192K + 0.95 + TQ3 + vision | **boot fails** — KV pool can't fit 192K at 0.95 | |

**The puzzling result:** 48K + 0.92 and 86K + 0.92 have **identical boot stats** (same VRAM used, same KV pool size) yet behave differently on the *same* 25K tool prefill. That's what cracked the diagnosis open — the difference can't be in static allocation.

### Root cause — dual mechanism (revised 2026-04-29 PM after P101+P103 testing)

Cliff 1 has **two mechanisms**; whichever has the larger allocation fires first under tight activation budget:

**Mechanism A — FA2 softmax_lse cap-leak.** vLLM's FA2 backend calls `flash_attn_varlen_func`, which internally allocates `softmax_lse` as `[num_seqs, num_heads, max_seqlen]` — sized by the `max_seqlen` parameter passed into the function call, NOT by the actual `cu_seqlens` of the current batch. vLLM passes `attn_metadata.max_seq_len`, which during cudagraph capture gets set to `max_model_len`. Tracked at [Dao-AILab/flash-attention#1011](https://github.com/Dao-AILab/flash-attention/issues/1011) (open since 2024).

**Mechanism B — FFN intermediate buffer.** The inductor-compiled FFN forward allocates the up_proj output as `[max_num_batched_tokens, intermediate_size]` per chunked-prefill batch. For Qwen3.6-27B with `max_num_batched_tokens=4128`, `intermediate_size=17408`: `4128 × 17408 × 2 bytes ≈ 138 MiB` per chunk. Stack-trace site:

```
File ".../inductor_cache/.../call.py", line 1208, in call
    buf9 = empty_strided_cuda((s18, 17408), (17408, 1), torch.float16)
```

`max_num_batched_tokens` is bounded below by `block_size` (4128 on hybrid Qwen3-Next due to Mamba cache constraint — asserted at boot if you try to go lower).

**How the two interact under our tested configs (2026-04-29):**

| Config | Dominant cliff | Allocate / Free |
|---|---|---|
| 192K + 0.98 + TQ3 + vision (current `long-vision.yml`, no P101/P103) | A (FA2 softmax_lse) | 50 MiB / 30 MiB |
| 192K + 0.98 + TQ3 + vision + P101 + P103 | **B** (FFN buffer; P101 reroutes around A) | 138 MiB / 130 MiB |
| 175K + 0.97 + TQ3 + vision + P101 + P103 | B (FFN buffer) | 138 MiB / 110 MiB |
| 205K + 0.98 + TQ3 + **no-vision** + P101 + P103 | A (FA2 softmax_lse) — vision drop frees ~500 MiB so FFN clears | 50 MiB / 50 MiB |
| 86K + 0.92 + TQ3 + vision (no P101/P103) | A (FA2 softmax_lse) | 50 MiB / 30 MiB |
| 48K + 0.92 + TQ3 + vision (default — no P101/P103) | neither — both fit in budget | ✅ passes |

**Implications:**

- Sandermage's P101 (already exists, opt-in) reroutes long-cached-prefix continuation prefill from FA2 → TQ decode kernel — closes Mechanism A but exposes Mechanism B.
- P103 (already exists, opt-in) addresses Cliff 2 (different code path entirely; not a Cliff 1 mitigation).
- A hypothetical FA call-site `max_seqlen_k` clamp (asked in [Genesis #11](https://github.com/Sandermage/genesis-vllm-patches/issues/11)) would close Mechanism A. Wouldn't close Mechanism B.
- A complete fix on TQ3 long-ctx with vision would need: clamp + chunked FFN forward (or activation checkpointing in FFN) + something to relieve the ~500 MiB pressure that vision tower adds.

vLLM passes `attn_metadata.max_seq_len` as the `max_seqlen` argument. During cudagraph capture in vLLM V1, [`gpu_model_runner` sets `max_seq_len = self.max_model_len`](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_model_runner/) so captured graphs have shape stability across all possible request sizes.

Result: at `max-model-len = 192K`, even a 4128-token chunked-prefill batch reserves `softmax_lse` for 192K. **Memory math:**

- `softmax_lse` element type: float32 (4 bytes)
- per-layer at 192K: `1 × num_heads × 192_000 × 4 = ~24 MiB per attention layer`
- across ~16 attention layers (Qwen3-Next is hybrid — 16 attention + 48 GDN): **~380 MiB total softmax_lse pre-allocation**
- at 86K: `~170 MiB total`
- at 48K: `~95 MiB total`

The difference between 86K and 48K is roughly the 50–138 MiB OOM allocation we observe. The leak is real and quantitatively explains the bisection.

This is exactly DeepSeek's hypothesis #4 in our consultation — which I initially dismissed as "tiny, probably not the culprit." Wrong. It IS the culprit.

### Why our earlier characterization was wrong

Until 2026-04-29 we described Cliff 1 as "FFN intermediate buffer activation peak (138 MiB at intermediate_size × max-num-batched-tokens)." That was based on the *amount* of memory failing to allocate (138 MiB) matching `17408 × 4128 × 2 bytes ≈ 144 MiB` — a coincidental match.

The actual stack trace (which we'd been observing all along but interpreting differently) shows the OOM site is `_vllm_fa2_C.varlen_fwd`, not the FFN. The 138 MiB allocation is FA2's softmax_lse + minor workspaces, not the FFN intermediate.

The FFN math was reasonable but wrong. Today's bisection (identical boot stats but different behavior at 48K vs 86K) couldn't be explained by FFN buffer math (chunk size is `max-num-batched-tokens`, which is identical across configs — so FFN buffer would be identical). Only `max-model-len` differed, which only affects FA2's softmax_lse padding.

This corrected understanding is now in [FAQ.md](FAQ.md#whats-a-prefill-cliff), [SINGLE_CARD.md](SINGLE_CARD.md), and the prefill-cliffs memory entry.

### Why mem-util doesn't help

Two coupled knobs:

| Action | Effect |
|---|---|
| Lower mem-util at fixed max-ctx | Engine ceiling drops (vLLM caps max-ctx by what the KV pool budget can hold). 192K + 0.95 doesn't boot. |
| Lower max-ctx at fixed mem-util | KV pool size unchanged (vLLM allocates max possible KV pool the budget allows). Activation budget identical. Same Cliff 1 firing. |

The two knobs are coupled — you can't get more activation budget without dropping the engine ceiling proportionally. Going from 192K + 0.98 to 48K + 0.92 isn't "more headroom at high mem-util" — it's "smaller engine budget overall, which forces smaller KV pool, which leaves more activation budget."

### Why PN8 closes Cliff 1 on `tools-text.yml`

PN8 (Sandermage's backport of vllm#40849, "MTP draft online-quant propagation") makes the MTP draft head inherit the target's online-quant config. On FP8+MTP paths, the draft loads in FP8 instead of BF16 default — saving ~600–900 MiB of draft-model footprint.

That freed memory becomes part of the activation budget at runtime. At `tools-text.yml` (75K + FP8 + PN8):

- Leaked `softmax_lse` at 75K: ~150 MiB
- Activation peak at prefill: ~50–138 MiB
- PN8's freed: ~900 MiB
- Net: enough for both leak and peak with margin to spare → Cliff 1 closes

PN8 doesn't fix the underlying FA2 leak — it just provides enough headroom that the leak fits without breaking anything.

PN8 doesn't reach TQ3 paths because:
1. PN8 propagates *weight* quant config (FP8 → FP8). On TQ3 paths the weights are AutoRound INT4 (not FP8); there's no FP8 quant to propagate.
2. TQ3 itself is a *KV-cache* format with custom Genesis kernels (P3/P4/P5/P6) that are target-side only — the draft model loader doesn't have those kernels wired in.

So at 192K + TQ3 + 0.98, even if we wanted PN8-style headroom, there's no draft-model footprint to free. The leak (~380 MiB) is bigger than any plausible alternative-knob fix.

---

## Cliff 2 — DeltaNet GDN forward intermediate buffer

### What you see

Symptom on `long-text.yml` (205K + 0.98 + TQ3 no-vision) when given a single user message exceeding ~50–60K tokens (RAG ingest, single-shot document summarization, repo-wide grep dump):

```
torch.OutOfMemoryError: CUDA out of memory.

  File ".../fla/ops/chunk/chunk_gated_delta_rule.py", line 312, in chunk_gated_delta_rule_fwd
    h = torch.empty(B, NT, H, V, K, dtype=..., device=...)
```

`NT = ceil(seq_len / chunk_size)` — grows linearly with sequence length. At `seq_len = 60K`, this allocation can exceed the available VRAM regardless of mem-util.

### When it fires

| Pattern | Fires? |
|---|---|
| Steady chat accumulation to 150K via many small turns | ❌ no — each turn's prefill is small (Cliff 2 is about *single-prompt* depth, not accumulated context) |
| Tool-using agent (≤25K tool returns, normal-size user messages) | ❌ no |
| RAG: stuffing a 100K-token document in one shot | ✅ yes |
| Big-doc summarization at 80K single user message | ✅ yes |
| Single 50K user message | borderline — sometimes fires, sometimes doesn't |

Trigger: **single-prompt sequence length crosses ~50–60K** (engine + workload-dependent threshold).

### Root cause

`fla.ops.chunk.chunk_gated_delta_rule_fwd` is the forward implementation of DeltaNet attention for the GDN (Gated Delta Network) layers in Qwen3-Next's hybrid architecture. The forward pass allocates an intermediate state tensor `h` shaped `(B, NT, H, V, K)`:

- B = batch size
- NT = number of chunks = `ceil(seq_len / chunk_size)`
- H = number of heads
- V, K = head dim

For Qwen3.6-27B (estimated):
- 48 GDN layers
- num_heads ≈ 32 (varies by layer config)
- chunk_size: ~256 in fla's default
- per-element: 4 bytes (float32) or 2 bytes (bf16)

At seq_len = 60K with chunk_size = 256: NT = 235. The total `h` allocation across layers is multi-GB — plausibly exceeding free VRAM on a single 24 GB card.

This is **not** a max_seqlen cap-leak like Cliff 1. The allocation is sized by *actual* seq_len. The cliff is in the algorithm itself: GDN forward materializes an O(seq_len × chunk_size) intermediate that doesn't fit on consumer Ampere VRAM beyond ~50–60K.

The architectural fix is a **streaming/tiled forward** — process the sequence in tiles, fold intermediate state at each tile boundary instead of materializing the whole thing. This is exactly what FlashAttention does for attention, and what FlashQLA does for DeltaNet on Hopper. There's no Ampere implementation today.

### Why TP=2 splits Cliff 2 in half

Under tensor parallelism with 2 cards, the GDN state buffer's per-card allocation is roughly halved (state is sharded across cards along the head dim). At seq_len = 240K, each card holds ~120K worth of state — fits in 24 GB. **Verified at 237K single-prompt prefill on `dual.yml`** (2026-04-29; ~830 tok/s prefill, peak 23.5 GB / card, no OOM).

This is why our long-prompt single-card recommendation is "use llama.cpp 262K (no cliff at all) or move to dual-card."

### KV format choice tunes the boundary

The Cliff 2 boundary isn't just "context length" — it's the sum of (KV pool bytes) + (DeltaNet GDN forward intermediate at the activation peak), and **the KV format choice trades these against each other**:

- `turboquant_3bit_nc` (TQ3 KV): 0.375 bytes/cached-token — smallest KV pool → most concurrency. Activation peak is *higher* (TQ3 dequant happens during the GDN forward, adding ~1 GB/card to the working set).
- `fp8_e5m2`: 1 byte/cached-token — larger KV pool but smaller activation peak. Single-prompt long-ctx is roomier; concurrency at full ctx drops.
- `q4_0` / `q8_0` / `k8v4` (vLLM's other KV options): in between, with their own dequant patterns.

On 24 GB / 3090 the per-card budget absorbs TQ3's activation peak and the smaller KV pool wins. **On 20 GB Ampere (modded 3080 / cap'd cards) the trade flips** — TQ3's activation peak after TP=2 split exceeds the per-card budget at 90K and Cliff 2 fires there. Override to fp8_e5m2 and the full 262K window opens. Validated cross-rig 2026-05-04 by [@efschu](https://github.com/noonghunna/club-3090/issues/47); see [HARDWARE.md](HARDWARE.md#note-for-sub-24-gb-cards) for the byte math.

The general principle: **the variant matrix is per-card-budget × KV-format-tradeoff aware**. Compose defaults are tuned for 24 GB / 3090; users on different VRAM classes may need to override `--kv-cache-dtype` to relocate the activation/pool balance for their hardware.

### Why llama.cpp doesn't have Cliff 2

llama.cpp's Qwen3-Next implementation processes DeltaNet/GDN layers with **online state updates** (incremental) rather than materializing the full intermediate. State is updated per-token or per-tile, never as a single multi-GB tensor. Different algorithm, different memory profile.

This is the same design difference that explains why llama.cpp doesn't have Cliff 1 either — see [LLAMA_CPP.md](engines/LLAMA_CPP.md) "Why llama.cpp doesn't hit the prefill cliffs vLLM does."

---

## Why llama.cpp dodges both cliffs structurally

Three architectural differences between the engines (full discussion in [docs/engines/LLAMA_CPP.md](engines/LLAMA_CPP.md)):

1. **Different attention library.** vLLM links `_vllm_fa2_C.varlen_fwd` (Dao-AILab FA2). llama.cpp uses ggml-cuda kernels (`fattn-mma-f16.cu`, `fattn-tile-f16.cu`, `fattn-vec-f16.cu`). No `max_seqlen` parameter to leak.
2. **Different KV/workspace model.** vLLM = paged attention + varlen kernel pre-allocating worst-case workspace. llama.cpp = static contiguous KV slab + dynamic per-call workspace sized by actual tokens.
3. **Cudagraph capture is decode-only in llama.cpp.** Prefill goes through the imperative ggml graph. No path for `max_model_len` to leak through capture metadata.

Bonus: **Cliff 2 also dodged** because llama.cpp's GDN is online-streaming, not tile-materializing.

The 3–4× TPS gap (vLLM ~70 TPS vs llama.cpp ~21 TPS on this stack) is the cost of these differences — vLLM optimizes for batched serving with fixed-shape kernels (faster steady-state, has cliffs); llama.cpp optimizes for single-request serving with dynamic shapes (slower steady-state, no cliffs). Neither is wrong; they're different design points on the same Pareto frontier.

This is why our launch frame is **two routes, not one**: vLLM dual-card for max throughput in environments where you control prompt shape, llama.cpp single-card for max robustness when prompts can balloon unpredictably.

---

## What we tried (workarounds and dead ends)

### Workarounds that work

| Mitigation | Closes which cliff? | Where shipped |
|---|---|---|
| Cap `max-model-len` at 48K (TQ3) | Cliff 1 (under threshold) | `docker-compose.yml` (default) |
| FP8 KV + PN8 + cap at 75K | Cliff 1 (PN8 absorbs leak) | `tools-text.yml` |
| TP=2 (dual-card) | Cliff 2 (state splits across cards) | `dual.yml`, `dual-turbo.yml` |
| llama.cpp engine swap | Both (different library entirely) | `llamacpp/default`, `llamacpp/concurrent` |

### Workarounds that don't work or are unavailable

| Mitigation | Why it fails |
|---|---|
| `--max-seq-len-to-capture` < `max-model-len` | Removed in V1 ([vllm#25543](https://github.com/vllm-project/vllm/pull/25543), merged 2025-09-24). Doesn't exist on our nightly. |
| `--enforce-eager` | Disables ALL cudagraphs, ~30% TPS hit, may break MTP. Partial fix at best — FA2 still receives `attn_metadata.max_seq_len` in eager paths. |
| `--max-num-batched-tokens 2048` (from 4128) | Halves chunk-size workspace; doesn't fix `softmax_lse[:, :, max_seqlen]` padding (which is sized by max_model_len, not chunk size). Marginal at best. **Don't pursue this as a primary fix** — it touches the Q dimension while the cap-leak is on the K dimension. |
| Lower mem-util (e.g. 0.92 → 0.88) | Coupled with max-ctx — going lower makes the engine ceiling drop too. No standalone benefit. |
| Extending PN8 to TQ3 paths | PN8 propagates *weight* quant config (FP8 → FP8); TQ3 is *KV* format with target-side-only kernels. Mechanism mismatch — can't naively port. |

### Alternative attention backends we evaluated

| Backend | Available on Ampere? | Avoids cap leak? | Realistic? |
|---|---|---|---|
| FlashAttention 2 (current) | ✅ | ❌ — has the leak | Status quo |
| FlashAttention 3 | ❌ Hopper-only (sm_90+) | ✅ | No |
| FlashInfer | Mostly Hopper; some Ampere paths | Different design — likely ✅ | Doesn't support TurboQuant 3-bit KV; doesn't support Qwen3-Next hybrid GDN+attention split. **No.** |
| xformers (`memory_efficient_attention`) | ✅ | Likely ✅ | xformers is essentially superseded; doesn't support TurboQuant or paged KV the way our TURBOQUANT backend needs. **Loses our entire feature stack.** |
| TRITON_ATTN | ✅ | ✅ (Triton kernels allocate dynamically) | ~30–40% TPS hit, may not support all our paths. **Last resort.** |
| Genesis TURBOQUANT (current) | ✅ | ❌ — internally calls `flash_attn_varlen_func` | What we're on |

There is no clean "swap the backend" workaround for our specific feature stack (Qwen3-Next hybrid + TurboQuant 3-bit KV + MTP spec-decode + paged KV + Ampere SM 8.6). Every alternative either doesn't run on our hardware, drops a feature we depend on, or trades a 30%+ TPS hit for the cliff fix.

---

## The fix landscape — who can address each cliff

### Cliff 1

| Actor | Fix | Likelihood / status |
|---|---|---|
| **Sandermage (Genesis)** | vLLM-side text-patch clamping `attn_metadata.max_seq_len` to actual current chunk seqlen at the FA call site, runtime-only (not capture-time) | **Currently asked** at [genesis-vllm-patches#11](https://github.com/Sandermage/genesis-vllm-patches/issues/11). Most efficient path — he has the patch infrastructure and the SWA-aware test harness. ~1-2 weeks reasonable wait for response. |
| **Tri Dao (FA2 maintainer)** | Change `softmax_lse` allocation in FA2 from `[num_seqs, num_heads, max_seqlen]` → `[total_q, num_heads]` (unpacked) | [Dao-AILab/flash-attention#1011](https://github.com/Dao-AILab/flash-attention/issues/1011) tracking it since 2024. Unlikely to be accepted — the padded shape is intentional for backward-pass shape stability and cudagraph capture. |
| **vLLM maintainers** | vLLM-side clamp in `vllm/v1/attention/backends/flash_attn.py` and per-backend variants. Same idea as Sandermage but landed upstream. | Possible but slow — would need careful PR with capture-correctness guarantees and SWA tests. [vllm#40961](https://github.com/vllm-project/vllm/pull/40961) is moving in the opposite direction (PRESERVING max_seq_len through capture). |
| **Us, if Sandermage declines** | Genesis-style text-patch in our `vllm/patches/` tree, similar to `patch_tolist_cudagraph.py` | 1–2 days dev + 1–2 days bench. In scope. Backup plan. |

### Cliff 2

| Actor | Fix | Likelihood / status |
|---|---|---|
| **`fla-org/flash-linear-attention`** maintainers | Streaming/tiled GDN forward — process sequence in tiles, fold intermediate state at boundaries | No upstream effort underway. Would be a substantial library rewrite. **No issue filed yet** — we should file one with our specific bisection data and Cliff 2 stack trace. |
| **QwenLM (FlashQLA)** | Port FlashQLA's TileLang kernels to Ampere SM 8.6 | FlashQLA is currently SM90+ only. Porting requires rewriting kernels using Ampere primitives instead of Hopper warp-specialization async tensor cores. **Out of QwenLM's stated scope** — we [tweet-drafted asking](https://github.com/noonghunna/club-3090/blob/master/docs/UPSTREAM.md) but no expectation. |
| **Sandermage** | Has explicitly punted on Cliff 2 in [single-3090#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1#issuecomment-4321094428): *"can't fix this short of multi-GPU TP=2 or upstream fla.ops changes"* | Not in scope. Confirmed. |
| **vLLM maintainers** | Could surface tile size as a knob, expose alternative GDN backend selectors | No tracking issue; would need us to file. |
| **Us** | Theoretically: rewrite GDN forward in tiled fashion. Practically: months of CUDA + research-level work. | **Out of scope.** Even if we attempted it, we'd be reimplementing FlashQLA without the TileLang infrastructure. |

---

## What we could do at any difficulty level

Ordered from cheapest to most aggressive, with realistic effort/reward.

### Trivial — already done

- [x] **Document and route.** SINGLE_CARD.md + DUAL_CARD.md TL;DRs surface the right config per workload pattern. FAQ explains cliffs. UPSTREAM.md tracks every related issue/PR.
- [x] **Cap default at 48K** to stay below Cliff 1. Most users land here without thinking about it.
- [x] **Genesis PN8 default-on for FP8+MTP.** Closes Cliff 1 on `tools-text.yml` for IDE-agent workloads.
- [x] **TP=2 verified at 237K single-prompt** for users with dual-card budget.
- [x] **llama.cpp 262K shipped as bulletproof fallback** for users with prompt unpredictability.

### Cheap (1-2 days, no novel CUDA work)

- [x] **Built ✓** — Codex agent shipped **P104 FA max_seqlen_k runtime clamp** (2026-04-30, branch `club-3090-cliff1-prep` in our local Genesis clone). Closes Cliff 1 **mechanism A** (FA2 softmax_lse). Also fixed silent-no-op bug in **P101** anchor — upstream `_arange_cache → torch.arange` change broke the old pattern; P101 was reporting "applied" but actually no-op'd. **P101 anchor fix opened as [Sandermage PR #12](https://github.com/Sandermage/genesis-vllm-patches/pull/12) on 2026-04-30; P104 held back pending Sandermage's response on issue #11.** Empirically validated via diagnostic log (`GENESIS_FA_CLAMP_DEBUG=1`); confirmed reroute past FA2 site on long-text.yml + 175K config.

  **Caveat: P104 alone doesn't close Cliff 1 on TQ3 + long-ctx + MTP at 24GB single-card.** Mechanism B (FFN intermediate buffer at `empty_strided_cuda((s18, 17408))` ≈ 138 MiB per chunk) fires next regardless of max_model_len — measured at 205K, 175K, all hit 138 MiB / 130.5 MiB free, same buffer site. The FFN buffer is sized by `max_num_batched_tokens × intermediate_size`; `max_num_batched_tokens` is pinned at 4128 by Mamba block_size constraint. Architecturally bounded.

  **Implementation shape (refined via ChatGPT consultation):**

  - Patch targets: `vllm/v1/attention/backends/flash_attn.py` AND `models/qwen3.6-27b/vllm/patches/genesis/.../turboquant_attn.py` (the `_flash_attn_varlen` wrapper around the FA call).
  - Clamp formula: `max_seqlen_k = min(attn_metadata.max_seq_len, actual_max_seq_len_for_this_batch)`. **NOT** chunk size (`max_num_batched_tokens=4128`) — chunk size is the Q dimension, but `softmax_lse` is shaped by the K dimension which spans accumulated prompt. Using chunk size would break continuation prefill.
  - Guards (all must hold simultaneously):
    1. FA2 / Ampere path only (skip on FA3 / Hopper / FlashInfer paths).
    2. Outside CUDA-graph capture only (capture metadata stays unchanged for shape stability — clamping during capture would break captured graphs).
    3. Never set below `max(seqused_k)` / actual current KV length (per-sequence; under-clamping = correctness bug).
  - Env gate: `GENESIS_FA2_CLAMP_MAX_SEQLEN=1` (default-off until validated).
  - Diagnostic logging at the patched call site: `num_actual_tokens`, `max_query_len`, `attn_metadata.max_seq_len`, `seq_lens.max()`. Useful both for verifying the patch fires correctly and for confirming the cap leak in the first place.

  **Test progression:**
  1. 86K + 0.92 + TQ3 + vision (known-fail case) — confirm clamp closes Cliff 1
  2. 75K + FP8 + MTP + PN8 + clamp (`tools-text.yml`) — verify regression doesn't break the existing PN8 mitigation
  3. 48K + 0.92 + TQ3 + vision (default) — verify no regression on the safe baseline
  4. 128K + 0.98 + TQ3 + vision — push the new ceiling and bisect to find new long-vision-safe value
  5. With MTP enabled at each: verify spec-decode AL stays in expected range (3.4-3.8)
  6. SWA-aware test: ensure no regression on attention-window models (none in our stack but worth verifying — Sandermage's fix also has to not break them)
  7. CUDA-graph capture validation: confirm capture-time metadata stays at `max_model_len` (clamp is runtime-only)

- [ ] **File `fla-org/flash-linear-attention` issue** with our Cliff 2 bisection data. Doesn't fix the cliff, but raises upstream signal that real users on consumer Ampere are affected. Increases the chance of someone tackling streaming GDN.

### Moderate (1-2 weeks, vendoring required)

- [ ] **Custom FA2 build** patching `softmax_lse` allocation to unpacked layout `[total_q, num_heads]`. Maintainable as a fork until upstream accepts (probably never). Need to vendor + rebuild on every nightly bump. Higher maintenance burden, no real upside vs the vLLM-side clamp above.
- [ ] **Custom Triton attention kernel** mirroring FA2 varlen but with dynamic workspace allocation. Substantial time investment for ~30% TPS regression vs FA2.

### Hard (2+ weeks, novel research)

- [ ] **Tiled GDN forward in `fla.ops`.** Implement streaming chunked_gated_delta_rule_fwd that doesn't materialize the full intermediate. Conceptually similar to FlashAttention's online softmax but for the gated delta rule. Days of CUDA prototyping + careful correctness validation against the reference forward + likely upstream-rejected for performance regression on Hopper. **Probably worth filing as a research-track upstream issue rather than implementing.**

### Out of scope

- [ ] **Port FlashQLA to Ampere.** TileLang's CUDA-DSL targets Hopper hardware features (warp specialization, async tensor cores). Porting would mean rewriting the entire kernel implementation in Ampere primitives. Months of full-time CUDA expert work. QwenLM team's territory.
- [ ] **Rewrite FA2 to use unpacked softmax_lse layout.** Would be a substantial change throughout the FA / PyTorch / vLLM stack. Tri Dao won't accept it (shape stability is intentional). We'd be vendoring forever.

---

## Update 2026-04-30 PM — PN12 anchor drift was the real bug; Cliff 1 closes at 205K

**Initial hypothesis (wrong):** PN12 only pools `SiluAndMul` output (1 of 3 FFN allocations) so the cliff fires at the unpooled `gate_up_proj` upstream. We thought extending PN12 to cover gate_up_proj would be required.

**Actual finding:** PN12 was **silently no-op'd** on dev205+ — same anchor-drift bug class as P101. Genesis's `apply_all` reported "PN12 applied" while the live `vllm/model_executor/layers/activation.py` retained the vanilla `SiluAndMul.forward_cuda` body (no `Genesis PN12` marker, no `FFNIntermediateCache` import). PN12's anchor expects the next decorator after `SiluAndMul` to be `@CustomOp.register("silu_and_mul_with_clamp")`; in dev205+ that section is `MulAndSilu`, so the text patch skipped without raising.

**Repair + result:** Once a local sidecar (`patch_pn12_ffn_pool_anchor.py`) actually patches `SiluAndMul.forward_cuda` with PN12's pooled-output body, **Cliff 1 closes at 205K** with the existing stack. No `gate_up_proj` extension needed. Sandermage's PN12 design intent was correct all along — the text-patch anchor was just stale.

### Verified shipped configs (2026-04-30 PM, RTX 3090 single-card)

| Variant | max_model_len | mem-util | Vision | Override | KV pool | Verified |
|---|---|---|---|---|---|---|
| **long-text.yml** | **218K** | 0.985 | ❌ | none | 280K tokens | verify-stress + verify-full pass, MTP AL 2.66, VRAM 23.7/24 GB |
| **long-vision.yml** | **198K** | 0.98 | ✅ | none | 260K tokens | verify-stress pass, MTP AL 2.63, VRAM 24/24 GB |

Both rely on the same local sidecars wired in via compose:
- `patch_pn12_ffn_pool_anchor.py` — repairs PN12 anchor on dev205+ (idempotent: skips if Genesis-side PN12 already applied via the bundled tree carrying PR #13's fix).
- `patch_fa_max_seqlen_clamp.py` — local P104 FA softmax_lse clamp.

Both also enable the runtime gates: `GENESIS_ENABLE_P101 / P103 / PN12_FFN_INTERMEDIATE_POOL / PN13_CUDA_GRAPH_LAMBDA_ARITY / FA_MAX_SEQLEN_CLAMP=1`.

Bisection that established the ceilings:

| Config | Result |
|---|---|
| long-text 220K + 0.985 + no vision | engine refuses (estimated max 218K) |
| long-text 218K + 0.985 + no vision | ✅ pass (shipped) |
| long-text 214K + 0.985 + no vision | ✅ pass |
| long-text 206K + 0.98 + no vision | ✅ pass |
| long-vision 220K + 0.985 + vision | engine refuses (estimated max 206K) |
| long-vision 205K + 0.985 + vision | ❌ Cliff 1 reopens (mem-util shifts budget away from activations) |
| long-vision 200K + 0.985 + vision | ❌ Cliff 1 reopens |
| long-vision 198K + 0.98 + vision | ✅ pass (shipped) |
| 240K + 0.99 + anything | hardware OOM at startup (driver reserves ~440 MiB; vLLM's 0.99 check fails) |

Full diagnostic log: [`models/qwen3.6-27b/vllm/diagnostics/cliff1-attack.md`](../models/qwen3.6-27b/vllm/diagnostics/cliff1-attack.md).

### Why our prior diagnosis was wrong

We had verifiable evidence the cliff fired at `empty_strided_cuda((s18, 17408))` with PN12 nominally enabled. We assumed PN12 was applying and concluded its surface area must be too narrow. We didn't verify the live `activation.py` content. Lesson: **for any Genesis text-patch on a fresh upstream pin, grep the live file for the patch marker before drawing implementation conclusions.** The same trap caught us with P101 on the prior cycle. Anchor health verification belongs ahead of implementation analysis.

### What's still architecturally bounded

- **Cliff 2** (DeltaNet GDN forward OOM at 50–60K single-prompt) is unchanged on single-card. `long-text.yml` remains "use for steady-state accumulation across many turns, not for stuffing >50K of fresh tokens in one request." Dual TP=2 (`dual.yml` at 237K) and llama.cpp (`llamacpp/default` at 262K) stay the paths for big single-shot prompts.
- **`--num-gpu-blocks-override 50`** caps usable concurrency at ~0.77x at 205K. Acceptable for single-stream long-text workloads (max_num_seqs=1); not suitable if multi-seq concurrency matters.
- **Local sidecars are required**: `patch_pn12_ffn_pool_anchor.py` and `patch_fa_max_seqlen_clamp.py` must be wired in until Genesis ships an anchor-corrected PN12 + P104 (or equivalent).

---

## Update 2026-05-01 PM — Cliff 1 mech B closed; FA varlen workspace cliff surfaces

**What changed:** Genesis pin bumped v7.62 → v7.64 (commit `64dd18b`). New patches in this cycle:
- **Sandermage's PN17** — anchored FA softmax_lse runtime clamp (replaces our P104 at the `flash_attn.py` layer; we keep P104 sidecar mounted because it covers the `turboquant_attn.py` wrapper layer that PN17 doesn't reach).
- **Sandermage's P38** — `_continuation_prefill` persistent K_full/V_full workspace replacing per-call `torch.cat` peaks at `turboquant_attn.py:903`. Activated via `GENESIS_ENABLE_P37=1`.
- **Our local `patch_pn12_compile_safe_custom_op.py`** — registers `club3090::pn12_silu_and_mul` as opaque `torch.library.custom_op` so Inductor-compiled `forward_native` routes through the FFN intermediate pool (which the eager `forward_cuda` PN12 patch couldn't reach under `custom_ops=["none"]`). Sandermage shipped his own version (PN25) on the `dev` branch — drop our local version when PN25 lands in stable.

**P38 result on long-text 205K + 0.985:** verify-full 8/8 (MTP AL 3.22). 130K-char (33K-token) tool-prefill stress passes cleanly. **But 200K-char (50K-token) single-shot tool-prefill stress crashes** — the OOM moves from `turboquant_attn.py:903` (`torch.cat` peak, P38's target) to a downstream allocation site at `flash_attn_interface.py:300:flash_attn_varlen_func`. The trace:

```
File ".../turboquant_attn.py", line 909, in _continuation_prefill
File ".../turboquant_attn.py", line 394, in _flash_attn_varlen
File ".../flash_attn_interface.py", line 300, in flash_attn_varlen_func
torch.OutOfMemoryError: Tried to allocate 50.00 MiB. ... 50.50 MiB is free.
```

50 MiB allocation, 50.5 MiB free. **None of our patches reach this allocation site** — it's inside the FA Python wrapper around the C extension, before any text-patch we apply. Mamba cache align mode forbids dropping `max_num_batched_tokens` below `block_size` (4128 on this model + TQ3) so the chunk size lever is unavailable.

### Bisection sweep (2026-05-01 PM, all with full v7.64 + sidecar stack)

| Config | Boots | verify-full | 130K stress | 200K stress | Notes |
|---|---|---|---|---|---|
| 200K + 0.97 | ❌ | — | — | — | Engine ceiling at 0.97 (text) is ~177K — KV pool short |
| 200K + 0.92 | ❌ | — | — | — | Engine ceiling at 0.92 is ~85K — far short |
| 175K + 0.97 | ✅ | 8/8 | ✅ | ❌ | Same FA varlen cliff at 50/50 MiB |
| 185K + 0.97 | ❌ | — | — | — | Engine ceiling at 0.97 is 177K |
| 185K + 0.975 | ✅ | 8/8 (AL 2.66) | ✅ | (untested in sweep) | Shipped on long-text + bounded-thinking |
| 130K + 0.95 | ✅ | 8/8 (AL 3.22) | ✅ | ✅ | Earlier intermediate config |
| 218K + 0.985 (original) | ✅ | 8/8 (AL 2.66) | ✅ | ❌ | Same FA varlen cliff |

### Update 2026-05-01 PM — P38 silently no-op'd on TurboQuant KV path

After shipping the 185K + 0.975 / 140K + 0.95 configs, instrumented `_genesis_continuation_prefill` (P38's replacement body) with a call counter + log line. Booted long-text and ran the 33K-token tool-prefill stress (which forces chunked continuation prefills). Result: **the patched body's log NEVER fires** despite the dispatcher's "rebound" line appearing at boot. Confirmed by inspecting the live patched `turboquant_attn.py:903` — it's still the original `v_full = torch.cat([v_cached_trim.to(qdtype), val_chunk], dim=0)` line, exactly the OOM site we observed at 50K-token stress.

**Architectural cause (same class as PN12 forward_native problem we fixed via the compile-safe sidecar):** vLLM's `aot_compile_fullgraph` decorator on the model `forward` captures the call chain `forward → _prefill_attention → _continuation_prefill` at compile time, baking in the ORIGINAL method bodies. P38's class-attribute rebind (`TurboQuantAttentionImpl._continuation_prefill = _genesis_continuation_prefill`) updates the live class but does NOT update the compiled artifact. Subsequent forward calls execute the pre-compiled original code, not the rebound method.

**Why Sandermage may not hit this in PROD:** his documented PROD configs target 35B-A3B-FP8 and 27B-Lorbus-fp8_e5m2. Both use **fp8 KV**, not TurboQuant — which means the entire `TurboQuantAttentionImpl._continuation_prefill` path is inactive there. P38 reports "applied" but never had a chance to take effect because the call site doesn't fire on fp8 paths. Our 27B AutoRound INT4 + TurboQuant 3-bit KV configs are precisely the paths that exercise `_continuation_prefill` — and discover the silent no-op.

**Fix path (mirrors what we did for PN12 → PN25):** convert `_continuation_prefill` to a `torch.library.custom_op` so Inductor treats it as opaque and dispatches via the registered op (which CAN be replaced/redefined). We have a working reference in `models/qwen3.6-27b/vllm/patches/patch_pn12_compile_safe_custom_op.py` for the FFN forward_native case. **Filed as [Genesis #14](https://github.com/Sandermage/genesis-vllm-patches/issues/14) 2026-05-01 PM.**

**Practical impact on shipped config:** none — long-text + bounded-thinking + long-vision shipped configs are bench-validated at 33K-token tool prefills with the existing patch stack (which doesn't actually include functional P38 on TQ paths). Removing the `GENESIS_ENABLE_P37=1` env var on long-text + bounded-thinking would simplify the config without changing behavior, but we leave it on to track when Sandermage fixes P38 — at that point the cliff at line 903 would close and we could push 50K stress.

**Update 2026-05-01 PM (later) — Sandermage shipped P38B + P15B on `dev`, pending v7.65.** Within hours of filing #14 + #15, Sandermage published two companion patches:
- **P38B** (Genesis #14 fix) — text-patches `turboquant_attn.py` source to inject a delegate hook at the start of `_continuation_prefill`. Source-level edit survives `aot_compile_fullgraph` capture because the compiler reads our modified source at engine init. Different mechanism from our PN12 → PN25 `torch.library.custom_op` route; both reach the same compile-time visibility. Composes with existing P38 for eager-mode coverage. Env: `GENESIS_ENABLE_P38B_COMPILE_SAFE=1`.
- **P15B** (Genesis #15 fix) — direct backport of our suggestion path 1. Text-patches `turboquant_attn.py:_flash_attn_varlen` to clamp `max_seqlen_k` from actual `cu_seqlens_k` before invoking the FA wrapper. Trade-off: ONE GPU→CPU sync per call (acceptable on continuation-prefill path which is infrequent). Env: `GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP=1`.

**Cross-validation on v0.20 (separate path):** while the Genesis-side fixes were pending, we tested the v0.20 pin upgrade across all 5 single-card / dual-card variants and found that **the 50K-token cliff doesn't reproduce on v0.20 at all** — different memory profile (likely vllm#40092's TQ FA3/FA4 prefill paths). So we have two independent paths to the same outcome:
1. **Stay on dev205, adopt v7.65 when it lands** — enable `P38B + P15B + PN25` env-gates, drop our local sidecars one-by-one.
2. **Migrate to v0.20.1rc1.dev16 + workspace_lock_disable sidecar** — empirically passes 50K stress today; needs Sandermage's marker fix on v7.65 for clean adoption.

Holding both paths open until v7.65 ships so we can do the full migration as one coherent PR (pin bump + Genesis bump + sidecar cleanup + context restoration to 218K/198K).

---

**Decision:** ship **long-text.yml at 185K + 0.975 / bounded-thinking.yml at 185K + 0.975 / long-vision.yml at 140K + 0.95**. The 175K → 185K bump on text-only recovers usable ctx vs intermediate configs; 0.985 → 0.975 drop frees ~240 MiB activation budget. Vision compose ships at the more conservative 140K + 0.95 because the vision tower's persistent ~1 GiB plus the new patches' persistent allocations (P38 K_full/V_full ~750 MiB at 185K + compile-safe sidecar ~138 MiB) reopen Cliff 2 (DeltaNet GDN forward buffer) on the vision compose at higher ctx; tested 185K + 0.98, 185K + 0.975, 160K + 0.97 vision configs — all reopened Cliff 2 at the 130K-char stress class until ctx dropped to 140K + 0.95. P37 disabled on vision because P37's MoE intermediate cache pool is no-op on dense Qwen3.6-27B and the env-gate doesn't free memory anyway. The synthetic 200K-char single-shot stress remains a known failure on every config — that's beyond what realistic agent workloads emit (ampersandru and VolandBerlioz repros were both ~30K real tokens).

### Re-push criteria

1. Upstream FA adds varlen workspace clamping at the call site OR
2. Sandermage's next pin extends PN17 coverage to the FA kernel entry (currently PN17 patches `flash_attn.py` not `flash_attn_interface.py`'s C-extension wrapper).

Until then 185K + 0.975 is the validated text-only ceiling on a single 24 GB 3090 with this patch stack.

---

## Our recommended path forward (revised post-2026-04-30 PM)

1. **Status:** [P101 PR #12](https://github.com/Sandermage/genesis-vllm-patches/pull/12) and [PN12 PR #13](https://github.com/Sandermage/genesis-vllm-patches/pull/13) opened 2026-04-30. Both are narrow anchor-drift fixes (same bug class). P104 still held back until Sandermage responds on issue #11 (P104 is new functionality, not just an anchor fix — different scoping decision).

2. **Update shipped configs (next):** `long-text.yml` can now ship a verified-cliff-safe 205K mode using the two local sidecars + `--num-gpu-blocks-override 50`. Default 48K stays as the conservative production option; 205K becomes the documented frontier-text variant. Pending decision on whether to flip the default or add as a variant.

3. **Dual.yml / llama.cpp paths unchanged** — both remain correct for their respective workloads (multi-stream + max-ctx single-prompt).

4. **For users who genuinely need cliff-safe long-context with vision or 70+ TPS:** route to dual-card `dual.yml` (TP=2, 237K verified single-prompt at ~830 tok/s prefill).

5. **Don't pursue chunked FFN forward, FA2 source patching, or FlashQLA Ampere port.** All are above-budget work for incremental improvement when better hardware paths already exist.

6. **Continue tracking upstream fixes.** [Dao-AILab/flash-attention#1011](https://github.com/Dao-AILab/flash-attention/issues/1011) (softmax_lse layout), [vllm#40961](https://github.com/vllm-project/vllm/pull/40961) (capture metadata flow), Sandermage's response to [Genesis #11](https://github.com/Sandermage/genesis-vllm-patches/issues/11). Re-test when any lands.

6. **Independently, file `fla-org/flash-linear-attention` issue** with Cliff 2 bisection data + our 237K-on-TP=2 result. Doesn't unblock us, but raises signal that consumer Ampere users hit this on Qwen3-Next class models.

7. **Cliff 2 stays documented as "not solved on single-card vLLM."** Route those workloads to TP=2 or llama.cpp. Don't pursue FA2 source patching, FlashQLA porting, or GDN rewriting — all above-budget for our team.

---

## Open questions and re-test triggers

| If this happens | Re-test |
|---|---|
| Sandermage ships a Cliff-1 clamp in Genesis | Re-bench `long-vision.yml` and `long-text.yml` — if Cliff 1 closes, we can drop the ⚠️ warning from SINGLE_CARD.md. Cliff 2 still applies though. |
| Dao-AILab merges some form of FA#1011 | Re-bench, may obsolete the Genesis clamp |
| `fla-org/flash-linear-attention` adds streaming GDN | Re-bench long single prompts on single-card — Cliff 2 might close |
| QwenLM ports FlashQLA to Ampere SM 8.6 | Investigate as a hot-swap for the GDN attention path — would close Cliff 2 with potential TPS gain |
| `vllm-project/vllm` adopts a backend selector that exposes per-call max_seqlen | Use it to clamp at the call site without text-patching |
| RTX 5090 / Blackwell consumer tier becomes targetable | FlashQLA might run there (sm_120 likely supports the necessary primitives), opening a different upgrade path |

---

## References

### Upstream issues / PRs (full list in [UPSTREAM.md](UPSTREAM.md))

- **Cliff 1:**
  - [Dao-AILab/flash-attention#1011](https://github.com/Dao-AILab/flash-attention/issues/1011) — softmax_lse padded by max_seqlen (root cause, open since 2024)
  - [vllm#40961](https://github.com/vllm-project/vllm/pull/40961) — preserve max_seq_len in ubatch metadata during CUDA graph capture (confirms the cap-leak pattern, going opposite direction of what we want for non-capture path)
  - [vllm#40849](https://github.com/vllm-project/vllm/pull/40849) — MTP draft online-quant propagation (the source of PN8, our current Cliff 1 mitigation on FP8+MTP)
  - [vllm#25543](https://github.com/vllm-project/vllm/pull/25543) — V0 deprecation removed `max_seq_len_to_capture` (kills one commonly-suggested mitigation)
  - [genesis-vllm-patches#11](https://github.com/Sandermage/genesis-vllm-patches/issues/11) — our request for a Genesis-style clamp at the FA call site

- **Cliff 2:**
  - No tracking issue filed yet at fla-org/flash-linear-attention — should file
  - [genesis-vllm-patches#1](https://github.com/Sandermage/genesis-vllm-patches/issues/1) — Sandermage's "can't fix without TP=2 or fla.ops changes" punt
  - [QwenLM/FlashQLA Ampere port request](https://github.com/noonghunna/club-3090/blob/master/docs/UPSTREAM.md) (tweet drafted, not posted)

### Our internal references

- [docs/UPSTREAM.md](UPSTREAM.md) — single source of truth for upstream tracking
- [docs/FAQ.md "What's a prefill cliff?"](FAQ.md#whats-a-prefill-cliff)
- [docs/SINGLE_CARD.md](SINGLE_CARD.md) — workload routing
- [docs/DUAL_CARD.md](DUAL_CARD.md) — TP=2 verification
- [docs/engines/LLAMA_CPP.md](engines/LLAMA_CPP.md) — why llama.cpp dodges both cliffs
- [models/qwen3.6-27b/INTERNALS.md](../models/qwen3.6-27b/INTERNALS.md) — engineering deep dive
- [Cross-rig data on club-3090 issue #2](https://github.com/noonghunna/club-3090/issues/2) — HoodOG1 + tenitram repro thread
- [Cross-rig data on single-3090 issue #1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1) — ampersandru's original Cliff 1 OOM trace
