# KV Cache Math — predicting per-card VRAM budget on Qwen3.6-27B

This page documents the math behind [`tools/kv-calc.py`](../tools/kv-calc.py) — the predictor that helps you decide whether a config will fit on your hardware *before* booting it. It also explains why predictions are estimates (±1.5 GB error band) rather than precise allocations.

## TL;DR

```bash
# What's my budget if I run dual-turbo on 20 GB cards?
bash tools/kv-calc.py --compose dual-turbo --vram 20 --mem-util 0.82

# What's the largest max_ctx that fits on 16 GB cards with TP=2 + fp8 KV?
bash tools/kv-calc.py --solve-max-ctx --tp 2 --kv-format fp8_e5m2 --vram 16 --mem-util 0.95

# How accurate is the model? Show predicted vs measured for our shipped composes:
bash tools/kv-calc.py --calibration
```

The predictor is a directional estimator, not a precise allocator. The vLLM engine's `gpu_worker.py` boot-log report is authoritative — the calculator is for *before* boot.

## Per-card budget components

For Qwen3.6-27B AutoRound INT4 at TP=N, the per-card VRAM peak during bench is composed of:

```
peak ≈ weights/N + kv_pool + activation_peak + cudagraph_workspace + dflash_draft/N
```

Each term has a well-defined formula or empirical anchor:

### 1. Model weights (`weights / N`)

AutoRound INT4 weights total ~17.5 GB on disk. Under tensor parallelism, weights split across cards:

- TP=1: 17.5 GB / card
- TP=2: 8.75 GB / card
- TP=4: 4.4 GB / card

This term is exact (the model checkpoint is a fixed size). DeltaNet's `linear_attn.in_proj_a` / `in_proj_b` layers stay at fp16 in the AutoRound quantization (per `extra_config` in `config.json`), but the byte budget is included in the 17.5 GB total.

### 2. KV pool (attention layers only)

In the Qwen3-Next hybrid architecture, **only the 16 full_attention layers contribute to the growing KV cache**. The 48 GDN (Gated DeltaNet) layers maintain a fixed-size recurrent state instead of a growing KV cache (Yang et al., [Gated Delta Networks ICLR 2025](https://github.com/NVlabs/GatedDeltaNet)).

Per-token KV cache bytes (across all attention layers, K + V):

```
per_token_bytes = num_attn_layers × num_kv_heads × head_dim × 2 × bytes_per_kv_element
                = 16        × 4              × 256      × 2 × bpe
                = 32,768 × bpe bytes
```

Where `bpe` depends on the KV format:

| KV format | `bytes_per_kv_element` | per-token KV (full) | per-token KV (TP=2) |
|---|---:|---:|---:|
| `fp16` / `bf16` | 2.0 | 65,536 B | 32,768 B |
| `fp8_e5m2` / `fp8_e4m3` | 1.0 | 32,768 B | 16,384 B |
| `q4_0` | ~0.56 | 18,350 B | 9,175 B |
| `k8v4` | 0.75 | 24,576 B | 12,288 B |
| `turboquant_3bit_nc` (TQ3) | ~0.425 | 13,927 B | 6,963 B |

Total KV pool (per card) = `per_token_bytes / TP × max_ctx × max_num_seqs`. PagedAttention ([Kwon et al., arxiv 2309.06180](https://arxiv.org/abs/2309.06180)) wastes <4% of this in fragmentation.

**Caveat**: this formula computes *requested* KV pool. vLLM's actual allocation is bounded by `mem_util × VRAM - other_components`. If requested exceeds available, vLLM emits `estimated max model length is N` and refuses to boot — that's the trigger for FAIL verdict.

**Caveat #2**: `max_num_seqs > 1` over-predicts in `kv-calc.py`. Real vLLM may rate-limit internally; my calculator doesn't model that. If your config uses `max_num_seqs=2` or higher and the calculator predicts FAIL but you've seen it boot, the calculator is over-counting. See "Known limitations" below.

### 3. Activation peak (GDN forward, the Cliff 2 mechanism)

The 48 GDN layers materialize a block-wise intermediate state during prefill. This is the source of [Cliff 2](CLIFFS.md#cliff-2--deltanet-gdn-forward-intermediate-buffer).

The PerfMamba paper ([arxiv 2511.22849](https://arxiv.org/html/2511.22849)) measures this directly on the parent architecture: at sequence length 2048, **Mamba-2 SSM consumes 33.5% more memory than Mamba-1 (115.68 GB vs 86.64 GB) due to "block-wise state materialization."** The asymptotic scaling per the paper:

```
activation_peak ∝ γ × D × N × L
```

where γ = expansion factor, D = hidden dim, N = state dim, L = sequence length.

For Qwen3.6-27B's GDN layers specifically, `fla.ops.chunk.chunk_gated_delta_rule_fwd` allocates an intermediate `h` shaped `(B, NT, H, V, K)`:
- B = batch, NT = `ceil(seq_len / chunk_size)` chunks (chunk_size=256)
- H = number of heads (linear_num_k_heads = 16, linear_num_v_heads = 48)
- V, K = head dim (linear_v_head_dim = linear_k_head_dim = 128)
- per-element 4 bytes (mamba_ssm_dtype = fp32 on this stack)

The published O(γDNL) gives the asymptotic scaling but not the absolute coefficient — that depends on `fla.ops.chunk` implementation details (tiling, streaming, register reuse). We use an **empirical coefficient** calibrated against the 8-10 measured BENCHMARKS rows:

| KV format | bytes/layer/token coefficient | Why this differs from fp8 |
|---|---:|---|
| `fp16` / `bf16` | ~135 | baseline (no KV dequant during forward) |
| `fp8_e5m2` / `fp8_e4m3` | ~130 | small dequant overhead |
| `q4_0` / `k8v4` | ~155 | larger dequant + scale ops |
| `turboquant_3bit_nc` | ~165 | TQ3 dequant during the materialized block adds ~20-25% activation pressure |

The TQ3 → fp8 difference (~25%) is what causes the [20 GB Ampere Cliff 2 fire at 90K](HARDWARE.md#note-for-sub-24-gb-cards) — TQ3's larger activation peak exceeds the per-card budget after TP=2 split on smaller-VRAM cards. Cross-rig validated by [@efschu](https://github.com/noonghunna/club-3090/issues/47) on 2× 3080 modded.

### 4. Cudagraph + workspace overhead

vLLM's torch.compile pass captures multiple cudagraph variants (one per `(batch_size, seq_len_bucket)` combination). Each capture costs ~50-100 MB. FlashInfer adds a 394 MB workspace per card. NCCL allreduce buffers cost ~200-300 MB on TP > 1.

Empirical fit:
```
overhead = 0.5 + 1.0 × mem_util + 0.3 × (TP - 1)  # GB
```

This is rough — actual overhead depends on how many graphs vLLM captures, which depends on `max_num_seqs`, `compile_sizes`, and other internals.

### 5. DFlash draft model

Only present on `dual-dflash*.yml` composes. `z-lab/Qwen3.6-27B-DFlash` is a ~1.75 GB draft model (per card, FP16). With TP > 1, the draft itself is sharded.

## Known limitations

**The model is empirically calibrated, not first-principles.** Specifically:

1. **`max_num_seqs > 1` over-predicts**. My KV pool formula is `max_ctx × max_num_seqs × per_token_bytes` — but vLLM doesn't always allocate the full pool. The engine internally rate-limits based on `mem_util × VRAM - other_stuff`. For configs with `max_num_seqs=2-4`, the calculator may say FAIL when reality is TIGHT-PASS.

2. **Activation coefficient varies by `chunk_size` and `dtype`**. We use the fla default `chunk_size=256` and `mamba_ssm_dtype=float32` (per Qwen3.6-27B config.json). If those change, the coefficient needs re-calibration.

3. **No driver/allocator overhead modeling**. snoby's 4090 needed `max-model-len` 200K → 180K vs 3090 baseline. The driver-class delta isn't modeled here. We hand-wave with the `±1.5 GB` error band.

4. **No Cliff 2b accumulation modeling**. The multi-turn fragmentation cliff at ~25K accumulated tokens is empirical-only and not in this calculator. Use `SOAK_MODE=continuous` to probe it.

5. **Single-model**. The math is Qwen3.6-27B-specific. Adding a model means deriving a new `MODEL_SPEC` block (architecture params + weights size) and re-calibrating the activation coefficient against new measured points.

## Calibration

Run `bash tools/kv-calc.py --calibration` to see predicted vs measured for all shipped composes. Current verdict accuracy: **9/11 = 82%** with the ±1.5 GB error band. The two ⨯ cases are over-predictions on `max_num_seqs > 1` configs (limitation #1 above).

## When to trust the calculator vs vLLM's boot log

| Question | Use this |
|---|---|
| "Will it boot?" — for a *shipped* compose on canonical 24 GB | We've already validated; check BENCHMARKS.md |
| "Will it boot?" — for a *novel* config (custom ctx, kv format, or VRAM class) | `kv-calc.py --compose <X>` for a directional answer; then boot and read `gpu_worker.py` |
| "What's my max ctx?" — given my hardware | `kv-calc.py --solve-max-ctx ...` for an estimate; vLLM's pre-check `estimated max model length is N` line at boot is authoritative |
| "Is TQ3 or fp8 better for my hardware?" | `kv-calc.py` with both options to see the trade-off; cross-check against [HARDWARE.md](HARDWARE.md#note-for-sub-24-gb-cards) for the published guidance |

## References

- [PerfMamba: Performance Analysis and Pruning of Selective State Space Models (arxiv 2511.22849)](https://arxiv.org/html/2511.22849) — block-wise state materialization scaling
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (arxiv 2504.19874, ICLR 2026)](https://arxiv.org/abs/2504.19874) — TQ3 byte savings + technique
- [Gated Delta Networks: Improving Mamba2 with Delta Rule (NVlabs ICLR 2025)](https://github.com/NVlabs/GatedDeltaNet) — Qwen3-Next architecture
- [Mamba: Linear-Time Sequence Modeling (arxiv 2312.00752)](https://arxiv.org/abs/2312.00752) — Mamba-1 baseline for PerfMamba's deltas
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (arxiv 2309.06180)](https://arxiv.org/abs/2309.06180) — vLLM's KV pool allocator
- [An Investigation of FP8 Across Accelerators for LLM Inference (arxiv 2502.01070)](https://arxiv.org/html/2502.01070v1) — FP8 e5m2/e4m3 KV cache analysis
- [docs/CLIFFS.md](CLIFFS.md) — Cliff 2 mechanism + KV-format-tunability section
- [docs/HARDWARE.md](HARDWARE.md) — 20 GB Ampere TQ3→fp8 swap rule (cross-rig validated by @efschu)

## See also

- [`tools/kv-calc.py`](../tools/kv-calc.py) — the predictor itself
- [BENCHMARKS.md](../BENCHMARKS.md) — measured cross-rig data, the calibration anchors
