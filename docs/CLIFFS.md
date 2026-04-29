# Prefill cliffs on Qwen3.6-27B / single 3090 — full synopsis

Comprehensive deep-dive into Cliff 1 and Cliff 2: what they are, why they fire, what's actually happening at the hardware/library level, what we've tried, what works, and what's left. Last revised 2026-04-29 after FA2 root-cause bisection.

This document supersedes earlier characterizations in CHANGELOG and FAQ where they conflict — those have been corrected to match what's documented here.

---

## TL;DR

**Two distinct OOM "cliffs"** fire during prefill on a single 24 GB RTX 3090 when serving Qwen3.6-27B with vLLM + Genesis patches. They affect different workload patterns and live in different libraries:

| | **Cliff 1** | **Cliff 2** |
|---|---|---|
| Trigger | 25K+ token tool messages → chunked prefill | Single prompt > ~50–60K tokens |
| OOM site | `_vllm_fa2_C.varlen_fwd` (FlashAttention 2) | `fla.ops.chunk_gated_delta_rule_fwd` |
| Root cause | `softmax_lse` allocated as `[num_seqs, num_heads, max_seqlen]` — padded to **max_seqlen parameter, not actual cu_seqlens** | DeltaNet/GDN intermediate state buffer non-streaming, sized by actual sequence length |
| Affects | TQ3 paths primarily (`docker-compose.yml` at higher max-ctx, `long-vision.yml`, `long-text.yml`) | All single-card vLLM configs at long single prompts |
| Our mitigation | **PN8** (Genesis patch) closes it on `tools-text.yml` (FP8+MTP); cap max-ctx at 48K on TQ3 default | **TP=2** (`dual.yml`) splits the GDN state across cards (verified at 237K); **llama.cpp** uses a different GDN implementation entirely |
| Real fix | vLLM-side clamp at FA call site (asked Sandermage in [Genesis #11](https://github.com/Sandermage/genesis-vllm-patches/issues/11)) OR FA2 source fix ([Dao-AILab/flash-attention#1011](https://github.com/Dao-AILab/flash-attention/issues/1011)) | Streaming GDN forward in `fla.ops` — no upstream effort underway. Or FlashQLA Ampere port (QwenLM, currently SM90+ only) |

**Practical impact today:** users following the SINGLE_CARD.md TL;DR table land on cliff-safe configs by default. The cliffs only bite users who explicitly opt into long-context variants AND have specific workload patterns (big tool returns, RAG single-shot prompts).

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

- [ ] **Build a vLLM-side `max_seq_len` clamp** ourselves if Sandermage doesn't take it. ~50 lines of Python text-patch in `turboquant_attn.py` or `flash_attn.py`. Same code shape as our existing `patch_tolist_cudagraph.py`. Closes Cliff 1 on TQ3 paths.

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

## Our recommended path forward

Ordered by efficiency and likelihood of working:

1. **Wait on Sandermage #11** for ~1-2 weeks. He has the most efficient path (existing patch infrastructure, SWA-aware tests, knows the vLLM internals). He just shipped v7.62.x with 5 days of intense work and explicitly said he needs rest before tackling the cliff next.

2. **Meanwhile, do prep work, not implementation:** read `vllm/v1/attention/backends/flash_attn.py` and `turboquant_attn.py` to map the patch surface. Identify the exact call sites where `attn_metadata.max_seq_len` flows into FA2, and verify the runtime-vs-capture distinction is reachable from there. Document findings — useful for Sandermage if he asks.

3. **Keep `default 48K` capped until clamp is verified.** Don't pre-emptively bump max-ctx based on the patch theory; only bump after measurement confirms Cliff 1 closes on each rung (86K → 128K → 192K).

4. **If Sandermage declines or doesn't respond after 2 weeks:** ship the vLLM-side clamp ourselves with the [refined implementation shape above](#cheap-1-2-days-no-novel-cuda-work) — env gate, dual-target patch (`flash_attn.py` + `turboquant_attn.py`), `min(attn_metadata.max_seq_len, actual_max_seq_len_for_this_batch)` clamp, runtime-not-capture only, never below `max(seqused_k)`. ~1-2 days dev + ~1 day bench progression (86K known-fail → 75K PN8 regression → 48K baseline → 128K push). Add patch to `setup.sh` parallel to `patch_tolist_cudagraph.py`.

5. **After clamp passes:** cautiously re-open 75K / 86K / 128K TQ3 configs as new variants, with the clamp env-gated on. Update SINGLE_CARD.md TL;DR.

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
