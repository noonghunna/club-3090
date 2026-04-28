# Qwen3.6-27B — Internals

Deep technical companion to this model's [README](README.md). Read this when you want to understand:

- Why a 27 B-parameter model with vision works on a single 24 GB consumer card at all
- What "Genesis patches" do under the hood and which patch fixes which bug
- Why TP=2 (dual-card) doesn't double single-stream TPS
- The Marlin pad-sub-tile-n patch we filed upstream
- DFlash N=5 vs MTP — when each wins
- The 9-probe forensics trail that isolated the upstream bugs we worked around
- Current upstream-fix status

If you just want to use the stack, the model README is enough.

For engine-general docs (vLLM tuning, llama.cpp tradeoffs, SGLang status), see [`/docs/engines/`](../../docs/engines/).

---

## Single-card: why this works where other recipes don't

Three hurdles had to be cleared for this config to run on a single consumer 24 GB card:

### 1. The published int4-AutoRound quant preserves `mtp.fc` at full precision

A vanilla `auto-round` run on Qwen3.6-27B packs the MTP fusion layer (`mtp.fc`) as INT4. In that form, vLLM's `Qwen3_5MTP` loader silently skips loading it (param name mismatch: expects `fc.weight`, finds `fc.qweight`). Result: MTP "loads" with zero parameters and produces **0% draft acceptance**.

Both [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) and [`Intel/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Intel/Qwen3.6-27B-int4-AutoRound) work around this — they ship `mtp.fc.weight` as a plain unquantized BF16 tensor. We use Lorbus because it's what we tested end-to-end.

Quick check that whichever quant you use has the fix: look for `mtp.fc.weight` (not `mtp.fc.qweight`) in the safetensors index.

### 2. Genesis patches bypass the TurboQuant hybrid gate

`Qwen3.6-27B` is a Qwen3-Next hybrid model: interleaved DeltaNet (Gated Linear Attention) + standard attention layers. vLLM's TurboQuant KV cache refuses to initialize on hybrid models:

```
NotImplementedError: TurboQuant KV cache is not supported for hybrid
(attention + Mamba) models. Boundary layer protection requires uniform
attention layers.
```

[Sandermage's Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) are a 20+-patch runtime monkey-patcher that, among other things, rewrites the hybrid gate to compute boundary protection only over attention layers. Works on Ampere SM 80–86.

### 3. Our `patch_tolist_cudagraph.py` fixes CUDA graph capture

Even with the hybrid gate bypassed, vLLM still crashed during engine warmup:

```
turboquant_attn.py:570  qsl = query_start_loc.tolist()
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

The continuation-prefill branch of `_prefill_attention` forces a GPU→CPU sync via `.tolist()`, which is illegal during CUDA graph capture. Our patch (`vllm/patches/patch_tolist_cudagraph.py`) is a disk-edit that wraps both `.tolist()` sites with `torch.cuda.is_current_stream_capturing()` guards — fall back to graph-safe fast path during capture, run the original at inference.

Without this patch, the documented workaround is `--compilation-config.cudagraph_mode=none`, which costs **−55% short-prompt TPS** and makes the whole setup net-negative vs plain fp8 KV.

---

## Genesis v7.14 — the surgical fix for the silent tool-call cascade

[Sandermage's Genesis v7.14](https://github.com/Sandermage/genesis-vllm-patches) shipped 2026-04-25 with the **P65** patch root-causing [vllm#40880](https://github.com/vllm-project/vllm/issues/40880):

`TurboQuantAttentionImpl._prefill_attention`'s cudagraph-capture bypass treats spec-decode K+1 verify batches as first-chunk prefill (sets `cu_seqlens_k = cu_seqlens_q`), so the captured kernel ignores cached KV. Drafter and verifier both produce noise from the kernel-without-context path; for tool-call prompts they converge on the same high-bias special token (`<tool_call>`) and cascade.

P65 downgrades `_cudagraph_support` from `UNIFORM_BATCH` to `UNIFORM_SINGLE_TOKEN_DECODE`. vLLM's compilation auto-detects and forces `cudagraph_mode=PIECEWISE` for spec-decode → eager continuation runs the correct branch. 1-token decode batches still get piecewise capture; only K+1 spec-verify batches go eager.

This is a workaround. The proper fix is a custom multi-query Triton kernel (P67) that handles K+1 query against compressed cached KV under cudagraph capture — designed-but-not-implemented as of v7.14.

---

## Single-card: forensics trail — the 9-probe bug isolation

**TurboQuant KV is frontier-level.** It landed in vLLM mainline only weeks before this repo's predecessor was published and is still under active development.

Initial symptom: under the originally-shipped 125K config (TurboQuant KV + MTP spec-decode + cudagraph on), the model produces degenerate token loops on tool calls, long-context recall, and occasionally streaming. We isolated the bug through nine probes:

| # | turboquant | spec-dec | cudagraph | torch.compile | result | TPS |
|---|---|---|---|---|---|---|
| 1 | ✅ | off | ✅ | ✅ | ✅ all tests pass | 40 |
| 2 | ✅ | ngram n=3 | ✅ | ✅ | ✗ same loops as MTP | -- |
| 3 (MiMo dense) | ✅ | MTP n=1 | ✅ | ✅ | ✗ first-token collapse | -- |
| 4 | ✅ | MTP n=3 | ✅ | + `_CONTINUATION_DECODE_THRESHOLD=0` | ✗ | -- |
| 5 | ✅ | MTP n=3 | ❌ | ❌ | ✅ all tests pass | 23 |
| **6** | ✅ | MTP n=3 | **❌** | ✅ | **✅ all tests pass** | **33** |
| 7 | ✅ | MTP n=3 (9-prompt structured-output sweep) | ❌ | ✅ | ✅ all 9 prompts pass | 33 |
| 8 | ✅ | MTP n=3 + PR #40798 backport | ✅ | ✅ | ✗ same loops | 96 |
| **9A** | ✅ | MTP n=3 + Genesis v7.13 (#40738 + parser fixes) | ✅ | ✅ | ✗ tool calls fail, recall truncates | -- |
| **9C** | ✅ | ngram n=3 + `prompt_lookup_min=8` + Genesis v7.13 | ✅ | ✅ | ✅ short-ctx clean | 35 |

**What this isolates:**

- Probe 1 → TurboQuant alone is fine.
- Probes 2-3 → bug isn't MTP-specific; isn't hybrid-attention-specific.
- Probe 4 → bug isn't in the within-batch decode-fast-path routing.
- Probe 5 → disabling **both** torch.compile and cudagraph fixes the bug — compilation machinery is the culprit.
- Probe 6 → disabling **only** cudagraph (keeping torch.compile inductor on) also fixes the bug — **isolating to CUDA graph capture/replay specifically**.
- Probe 7 → confirmed against [Sander's 9-prompt corruption-detection suite](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4317214311) — all clean.
- Probe 8 → backported PR #40798 (workspace-manager refactor). Bug persists.
- **Probe 9A → Sander's v7.13 backports do NOT fix MTP × TurboQuant × cudagraph on Qwen3.6-27B.** Filed as [#40880](https://github.com/vllm-project/vllm/issues/40880).
- **Probe 9C → ngram + `prompt_lookup_min=8` + v7.13 backports DO work** at short context (cross-confirmation of [#40875](https://github.com/vllm-project/vllm/issues/40875)).

**The Triton kernels are correct when invoked dynamically. torch.compile inductor output is correct.** What corrupts the output is how the captured CUDA graph handles spec-decode's runtime shapes vs warmup-shape capture for the TurboQuant attention path.

---

## Dual-card: why TP=2 doesn't double single-stream TPS

A common expectation: "I'm doubling the GPUs, I should get double the speed." Reality on PCIe-only consumer Ampere: per-stream TPS gain from TP=2 is small (~5%).

**Why:** decode on a single batch is memory-bandwidth-bound, not compute-bound. With TP=2:

- Each card holds half the weights (good — halves the per-card memory bandwidth requirement)
- But after each layer's matmul, partial results must combine via NCCL **all-reduce** across both cards
- All-reduce on **PCIe Gen 4** (~32 GB/s practical) is ~3-5× slower than NVLink (~600 GB/s on H100, ~200 GB/s on 3090 with bridge)
- All-reduce overhead approximately cancels the memory-bandwidth halving

Net: single-stream TPS per card ≈ same as single-card TPS. The TP=2 win is **concurrent** throughput — when 2-4 streams run simultaneously, all-reduce overhead amortizes across the larger batch and aggregate scales near-linearly to ~4 streams.

This is why we measure both single-stream TPS and "concurrent throughput at full ctx" — the latter is what you'd get serving 4 simultaneous users.

**When this would matter less:**
- If you have NVLink (e.g., A6000 + bridge, or H100 SXM): single-stream gain from TP=2 is closer to 1.6-1.8×
- If you're running attention-bound prefill, not decode, the TP win is bigger (prefill is more compute-bound)

---

## Dual-card: the Marlin pad-sub-tile-n patch (vllm#40361)

**Symptom:** vLLM crashes during model-load on TP=2 with:
```
RuntimeError: GPTQ_MARLIN_MIN_THREAD_N (64) > out_features
```

**Root cause:** vLLM's Marlin INT4 GEMM kernel requires `out_features ≥ GPTQ_MARLIN_MIN_THREAD_N` (64). Lorbus's Qwen3.6-27B-int4-AutoRound has tensors with `out_features = 128` for some MTP-related layers. With TP=2, those split into 64 per card — exactly at the boundary. Some users hit smaller post-split shapes and crash instantly.

**Our fix (PR #40361):** Pad the tensor's `out_features` dimension up to the kernel minimum (64) before dispatch, then slice the result back. Trade-off: one extra memcpy per Marlin call. Measured cost: <0.5% TPS overhead. Gain: works on any TP-shape that produces sub-tile layers.

```python
# Pseudo-code of the patch
if out_features < GPTQ_MARLIN_MIN_THREAD_N:
    pad_to = GPTQ_MARLIN_MIN_THREAD_N
    weight_padded = pad_tensor(weight, pad_to)
    output = marlin_gemm(weight_padded)
    output = output[..., :out_features]  # slice back
else:
    output = marlin_gemm(weight)         # original fast path
```

Status: PR is **OPEN, MERGEABLE**, labeled `bug`, sitting in maintainer queue. When it lands, drop the `/opt/ai/vllm-src/` mount from the dual compose. Until then, the volume-mount path lets users get the fix without forking themselves.

---

## Dual-card: DFlash N=5 — what it does differently than MTP

**MTP (default in single + dual fp8):** Qwen3.6 ships an integrated 1-token MTP head trained jointly with the main model. We run it with `num_speculative_tokens=3` — predicts 3 tokens forward, verifies, accepts what's correct. AL ~3.4 means 3.4 tokens accepted per spec-decode step on average.

**DFlash (z-lab fork, dual-card only):** A separately-trained, larger draft model (5-token forward window) optimized specifically for **code workloads**. Runs in parallel with the main model on the same GPUs. AL ~4.7 on code prompts (vs 3.4 for MTP).

Why DFlash beats MTP on code:
- **Bigger draft model** = more accurate predictions per position (vs MTP's tiny ~1B head)
- **Trained on code-heavy data** = higher AL on structured tokens (function names, syntax, brackets)
- **N=5 vs N=3** = more shots per verify step when accept rate is high (which it is on code)

Cost of DFlash:
- Draft model adds ~500 MB VRAM (negligible on dual-3090)
- ~20% extra compute per forward (the parallel draft) — but accept rate gain pays it back
- Vision support is preserved in `dual-dflash.yml`, dropped in `dual-dflash-noviz.yml` (text-only path slightly faster + 200K vs 185K ctx)

When MTP wins:
- Narrative / chat workloads (code-token bias of DFlash doesn't help, AL drops to ~2.8)
- Tool-call generation (structured but different patterns than code)
- When you need full 262K ctx (DFlash caps at 185K with vision / 200K without due to draft-model VRAM)

So the dual-card path ships both: default = fp8 + MTP for breadth, DFlash for code-heavy peak performance.

---

## Why we configure consumer-Ampere knobs differently than Sandermage's reference

Sandermage tests on 2× A5000 (32 GB each). Their default `gpu-memory-utilization=0.92` and `max-num-batched-tokens=8192` work great there because A5000 has 33% more VRAM. On 24 GB 3090s, those defaults don't leave enough activation headroom for chunked-prefill peaks at 60K+ context.

Our adjustments in the dual-Turbo variant:
- `gpu-memory-utilization 0.92 → 0.85` — frees ~1.7 GB activation headroom per card
- `max-num-batched-tokens 8192 → 4128` — smaller chunked-prefill chunks, smaller activation peak per forward

Without these, deep-prefill (60K+) requests OOM. With them, the Turbo variant runs cleanly at full 262K with 4-stream concurrency.

For Sandermage's documented numbers on his A5000 setup, see his [MODELS.md](https://github.com/Sandermage/genesis-vllm-patches). For our adjusted numbers on 3090s, see this model's USE_CASES.md.

---

## Upstream status

| Issue | Status | Notes |
|---|---|---|
| [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) | **OPEN, MERGEABLE** | Our Marlin pad-sub-tile-n patch (dual-card dependency). Drops out as a dependency when this lands. |
| [vllm#40334](https://github.com/vllm-project/vllm/pull/40334) | **OPEN, NOT MERGED** | DFlash `combine_hidden_states` dtype mismatch fix. Workaround in our DFlash composes: `--dtype bfloat16`. |
| [vllm#40807](https://github.com/vllm-project/vllm/issues/40807) | Worked around locally | CUDA graph `.tolist()` crash (single-card). Our `patch_tolist_cudagraph.py` ships the fix. |
| [vllm#40831](https://github.com/vllm-project/vllm/issues/40831) | Closed (ngram + MTP) | TurboQuant × spec-decode corruption. Ngram via [#40875](https://github.com/vllm-project/vllm/issues/40875) `prompt_lookup_min=8`; MTP via Genesis v7.14 P65. |
| [vllm#40880](https://github.com/vllm-project/vllm/issues/40880) | Worked around (Genesis v7.14) | MTP × TurboQuant × cudagraph (single-card). P65 PIECEWISE downgrade. Proper P67 multi-query Triton kernel TBD. |
| [vllm#40914](https://github.com/vllm-project/vllm/pull/40914) | **OPEN** | Sandermage's upstream synthetic-args trick for spec-decode K+1 verify. Would close P67 design gap. |
| [PR #40798](https://github.com/vllm-project/vllm/pull/40798) | Hypothesized fix that didn't pan out | Workspace-manager refactor. Probe 8 backported it; bug persisted. Useful negative result documented on the PR thread. |

When PR #40361 lands, we drop the `/opt/ai/vllm-src/` mount from dual composes. When PR #40914 lands, the dual-Turbo variant's TPS regression vs fp8 narrows substantially.

---

## See also

- [Qwen3.6-27B README](README.md) — model overview, recommended configs, quick start
- [Qwen3.6-27B USE_CASES](USE_CASES.md) — per-workload guides (chat, agents, code, RAG, vision, frontier ctx, multi-tenant)
- [Qwen3.6-27B CHANGELOG](CHANGELOG.md) — dated history
- [Cross-engine docs](../../docs/engines/) — comparison + per-engine deep dives
- [Hardware notes](../../docs/HARDWARE.md) — Ampere SM 8.6+, NVLink, power, etc.
