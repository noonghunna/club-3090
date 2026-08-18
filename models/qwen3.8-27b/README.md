# Qwen3.8-27B

Next-gen 27B model from the Qwen team with refined DeltaNet architecture.

## Quick start

```bash
# 1. Download weights (~16 GB — IQ4_NL is the 4-bit default; no IQ4_KS for this arch)
hf download bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-IQ4_NL.gguf \
  --local-dir $MODEL_DIR/qwen3.8-27b-gguf

# 2. (Optional) Download vision projector
hf download bartowski/Qwen3.8-27B-GGUF mmproj-Qwen3.8-27B-f16.gguf \
  --local-dir $MODEL_DIR/qwen3.8-27b-gguf/mmproj

# 3. Launch (with vision) - defaults equal the benched config
MODEL_DIR=/path/to/models docker compose -f llama-cpp/compose/single/bartowski-q4km/q4kv-vision.yml up -d

# For text-only, use the upstream compose:
#   docker compose -f llama-cpp/compose/single/unsloth-iq4nl/q8kv.yml up -d

# 5. Test
curl http://localhost:8020/v1/models
```

## Available configs

| Config | Path | Vision | Context | Notes |
|---|---|---|---|---|
| `bartowski-q4km/q4kv-vision` | `llama-cpp/compose/single/bartowski-q4km/q4kv-vision.yml` | **yes** | **131K** | Multimodal. Defaults equal the benched config: bartowski Q4_K_M + mmproj-F16, q4_0 KV, MTP n=2. NIAH-filled to 120,320 tok (91%) |
| `unsloth-iq4nl/q8kv` (upstream) | `llama-cpp/compose/single/unsloth-iq4nl/q8kv.yml` | no | 131K | Text-only, shipped by upstream master — not part of this PR |

## Weights sources

### bartowski/Qwen3.8-27B-GGUF (default)

Four-bit default is **IQ4_NL** (see the note below — this arch has no IQ4_KS).

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
GGUF_FILE=qwen3.8-27b-gguf/unsloth/Qwen3.8-27B-Q4_K_M.gguf \
  MODEL_DIR=/path/to/models docker compose -f llama-cpp/compose/single/bartowski-q4km/q4kv-vision.yml up -d
```
(Download unsloth weights into `$MODEL_DIR/qwen3.8-27b-gguf/unsloth/` first. Note
unsloth's Q4_K_M is a **different file**: 17,106,775,008 B vs bartowski's
17,772,537,440 B. Its projectors are also named differently (`mmproj-F16.gguf`,
927,607,488 B), so set `MMPROJ_FILE=` too. Only the bartowski pairing is benched.)

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

## What's working

- **llama.cpp single-card boot, with and without vision** - bench-validated
  config is unsloth `Q4_K_M` weights + `mmproj-Qwen3.8-27B-f16.gguf`, q4_0 KV,
  `-c 131072`, MTP `--spec-draft-n-max 2`, single RTX 3090 @ 370 W.
- **MTP spec-decode** on the bench-validated config above.
- **Vision** via the `-vision` compose variant, tested at the bench-validated
  context, not at the compose's higher default.

## What's not working today

- **The compose files' shipped default** - IQ4_NL weights @ 200K ctx, MTP
  n_max 3, `mmproj-Qwen3.8-27B-bf16.gguf` - has been **downloaded to the host
  but never booted or benchmarked**. Treat it as unvalidated until it's run
  through the same fill-ladder as the Q4_K_M path.
- **No IQ4_KS for this architecture** (see Weights sources above) - the
  `iq4ks*.yml` compose names are placeholder labels, not literal IQ4_KS
  quants.
- **200K context end-to-end** - only the Q4_K_M @ 131K + vision path has been
  measured on this rig; the IQ4_NL @ 200K path is untested (see VRAM table
  above).

## See also

- [BENCHMARKS.md](../../BENCHMARKS.md) - measured TPS / TTFT for the
  bench-validated config on this rig.
- [`llama-cpp/compose/single/bartowski-q4km/q4kv-vision.yml`](llama-cpp/compose/single/bartowski-q4km/q4kv-vision.yml) - the vision compose added by this PR.
- `llama-cpp/compose/single/unsloth-iq4nl/q8kv.yml` - upstream's text-only
  single-3090 compose (not part of this PR).

## Troubleshooting

**OOM at boot?** Lower `CTX_SIZE` (e.g. `CTX_SIZE=131072`) or try `UBATCH_SIZE=1024`.

**Slow prefill?** Try `CTX_SIZE=131072 UBATCH_SIZE=1024` for faster prefill at the cost of max context.

**Vision not working?** Make sure `mmproj-Qwen3.8-27B-bf16.gguf` is downloaded and the `--mmproj` flag is present in the compose.
