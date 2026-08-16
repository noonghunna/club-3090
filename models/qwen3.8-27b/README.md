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
| `iq4ks` | `llama-cpp/compose/single/iq4ks.yml` | no | 200K | Text-only, max context |
| `iq4ks-vision` | `llama-cpp/compose/single/iq4ks-vision.yml` | yes | 150K | Multimodal @ 1M-px default |

## Weights sources

### bartowski/Qwen3.8-27B-GGUF (default)

Broad quant selection. The configs above default to **IQ4_KS** (~14.8 GB).

| Quant | Size | Quality |
|---|---|---|
| `IQ4_KS` | ~14.8 GB | **Recommended** - best quality/size for 24 GB |
| `IQ4_NL` | ~15.6 GB | Nearly identical quality, slightly larger |
| `Q4_K_M` | ~16.9 GB | Well-tested, slightly larger |
| `Q8_0` | ~27.8 GB | Near-FP8 - may OOM with large context on 24 GB |

### unsloth/Qwen3.8-27B-GGUF (alternative)

Unsloth Dynamic V3.0 architecture - state-of-the-art 4-bit quant with native MTP.

| Quant | Size |
|---|---|
| `IQ4_KS` | ~15.0 GB |
| `IQ4_NL` | ~15.6 GB |
| `Q3_K_M` | ~12.9 GB |
| `Q3_K_S` | ~11.7 GB |

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

| Component | Size |
|---|---|
| IQ4_KS weights | ~14.8 GB |
| KV cache (200K, q4_0) | ~6.0 GB |
| MTP overhead | ~0.5 GB |
| **Total (text-only)** | **~21.3 GB** |
| mmproj BF16 | ~0.9 GB |
| Image buffers | ~0.5 GB |
| **Total (vision)** | **~22.7 GB** |

## Troubleshooting

**OOM at boot?** Lower `CTX_SIZE` (e.g. `CTX_SIZE=131072`) or try `UBATCH_SIZE=1024`.

**Slow prefill?** Try `CTX_SIZE=131072 UBATCH_SIZE=1024` for faster prefill at the cost of max context.

**Vision not working?** Make sure `mmproj-Qwen3.8-27B-bf16.gguf` is downloaded and the `--mmproj` flag is present in the compose.
