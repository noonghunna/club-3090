# Qwen3.8-27B

Next-gen 27B model from the Qwen team with refined DeltaNet architecture.

## Quick start

```bash
# 1. Download weights (~15GB)
hf download bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-IQ4_KS.gguf \
  --local-dir $MODEL_DIR/qwen3.8-27b-gguf/iq4ks

# 2. (Optional) Download vision projector
hf download bartowski/Qwen3.8-27B-GGUF mmproj-Qwen3.8-27B-bf16.gguf \
  --local-dir $MODEL_DIR/qwen3.8-27b-gguf

# 3. Launch (text-only)
MODEL_DIR=/path/to/models docker compose -f llama-cpp/compose/single/iq4ks.yml up -d

# 4. Launch (with vision)
MODEL_DIR=/path/to/models docker compose -f llama-cpp/compose/single/iq4ks-vision.yml up -d

# 5. Test
curl http://localhost:8020/v1/models
```

## Available configs

| Config | Path | Vision | Context | Notes |
|---|---|---|---|---|
| `iq4ks` | `llama-cpp/compose/single/iq4ks.yml` | no | 200K | Text-only; **200K requires IQ4_NL weights** — OOMs with Q4_K_M (see Benchmarks) |
| `iq4ks-vision` | `llama-cpp/compose/single/iq4ks-vision.yml` | yes | 150K | Multimodal @ 1M-px default; bench rig ran 131K with Q4_K_M |

## Weights sources

### bartowski/Qwen3.8-27B-GGUF (default)

Broad quant selection. The configs above default to **IQ4_KS** (~14.8 GB).

| Quant | Size | Quality |
|---|---|---|
| `IQ4_NL` | ~16.3 GB | **Recommended** - 4-bit with largest super-block; fits 200K ctx on 24 GB (untested — see Benchmarks) |
| `Q4_K_M` | ~17.8 GB | **Bench-validated on this rig** — 63/72 TPS @ 131K + vision (see Benchmarks); 200K OOMs |
| `Q8_0` | ~29.1 GB | Near-lossless — does not fit 24 GB with context |

> ⚠️ **No IQ4_KS in this repo.** Qwen3.8's hybrid DeltaNet/attention arch ships
> without an IQ4_KS from bartowski (IQK quants for this arch are IQ2_XS→IQ4_NL
> and IQ4_XS only). The `iq4ks*.yml` names are a quant-label placeholder for
> "4-bit" — the compose's effective default is IQ4_NL.

### unsloth/Qwen3.8-27B-GGUF (alternative — bench-validated)

Unsloth Dynamic V3.0 architecture with native MTP. **The rig behind the
BENCHMARKS row below ran unsloth `Q4_K_M` (17.7 GB) + `mmproj-Qwen3.8-27B-f16.gguf`.**

| Quant | Size |
|---|---|
| `Q4_K_M` | ~17.8 GB (bench rig) |
| `Q3_K_M` / `Q3_K_S` | ~13 / ~12 GB (tighter fit, smaller ctx headroom) |

To use unsloth weights, override the path at launch:
```bash
GGUF_FILE=qwen3.8-27b-gguf/unsloth/Qwen3.8-27B-IQ4_KS.gguf \
  MODEL_DIR=/path/to/models docker compose -f llama-cpp/compose/single/iq4ks.yml up -d
```
(Download unsloth weights into `$MODEL_DIR/qwen3.8-27b-gguf/unsloth/` first.
For the vision compose, also set `MMPROJ_FILE=` if unsloth's projector name differs.)

## Architecture

Qwen3.8-27B uses a mixed architecture:
- 16 blocks of: 3x Gated DeltaNet (linear attention) -> FFN, then 1x Gated Attention -> FFN
- 64 layers total, 5120 hidden dim, 24Q/4KV attention heads
- Native context: 262K tokens, extensible with YaRN
- MTP (multi-token prediction) trained with n=3 steps

## VRAM on 24 GB (RTX 3090)

Measured facts (bench rig, unsloth Q4_K_M + mmproj-F16, q4_0 KV, `-c 131072`):
boot ≈ **22 GB**. KV cost is 16 gated-attention layers × 4 KV heads × 256 × 2
= 32 768 elems/token ≈ **18 KB/token** at q4_0 → 131K ≈ 2.4 GB, 200K ≈ 3.7 GB.

| Config | Weights | KV | Fits 24 GB? |
|---|---:|---:|---|
| Q4_K_M + mmproj @ 131K | 17.8 + 0.93 | ~2.4 | ✅ **measured, 22 GB boot** |
| Q4_K_M + mmproj @ 200K | 17.8 + 0.93 | ~3.7 | ❌ **~22.4+ GB → OOM** |
| IQ4_NL @ 200K | ~16.3 | ~3.7 | ⚠️ ~21 GB — should fit, **untested** |
| IQ4_NL + mmproj @ 150K | ~16.3 + 0.93 | ~2.7 | ⚠️ ~20 GB — should fit, **untested** |

**The 200K default in the text compose is unvalidated** — it only works with
IQ4_NL weights and needs a fill-ladder verification before graduating the
compose out of incubating.

## Troubleshooting

**OOM at boot?** Lower `CTX_SIZE` (e.g. `CTX_SIZE=131072`) or try `UBATCH_SIZE=1024`.

**Slow prefill?** Try `CTX_SIZE=131072 UBATCH_SIZE=1024` for faster prefill at the cost of max context.

**Vision not working?** Make sure `mmproj-Qwen3.8-27B-bf16.gguf` is downloaded and the `--mmproj` flag is present in the compose.
