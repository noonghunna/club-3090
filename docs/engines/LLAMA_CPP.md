# llama.cpp — quick recipe + gotchas

If you want a lighter-weight setup, run on non-NVIDIA hardware, or just prefer llama.cpp's ergonomics, here's how to run Qwen3.6-27B on it. **Note:** mainline llama.cpp's Qwen3-Next support is still landing (4 PRs open). The fastest path today is via [Luce z-lab's DFlash fork](https://github.com/luce-spec/llama-cpp-dflash), which adds a custom DFlash spec-decode draft model.

---

## TL;DR

- ✅ Runs on this stack (with caveats)
- ✅ Vision via `mmproj` model
- ✅ Fastest cold start of any engine (~30 sec)
- ✅ Smallest footprint (single binary, ~50 MB)
- ✅ **Best path for 262K context on a single 3090** — see "Going to 262K" recipe below
- ⚠️ Server feature parity behind vLLM (no auto-tool-choice in upstream `server` binary; need a wrapper)
- ⚠️ Concurrent serving is single-threaded (forks per request) → sluggish UX under load
- ⚠️ No spec-decode in mainline (Luce DFlash fork has it; mainline doesn't yet)

---

## Why pick llama.cpp over vLLM on a single 3090?

The honest answer: **disk + VRAM size**.

| Quant + format | Disk size | VRAM at idle | Headroom for KV at 262K |
|---|---|---|---|
| Lorbus int4-AutoRound (vLLM) | ~18-19 GB | ~19-20 GB | **~3-4 GB** (can't fit f16 KV at 262K — needs TQ3 KV compression on vLLM) |
| GGUF Q4_K_M (llama.cpp) | ~16.8 GB | ~17-18 GB | ~6 GB |
| GGUF UD-Q3_K_XL (llama.cpp) | **~14.5 GB** | ~15-16 GB | **~8-9 GB** |
| GGUF Q3_K_M (llama.cpp) | ~13.6 GB | ~14-15 GB | ~9-10 GB |

The ~4 GB savings going from AutoRound (vLLM) to UD-Q3_K_XL (llama.cpp) translates **directly** into more KV cache room. That's the difference between "fits 262K with quantized KV" (llama.cpp) and "fits 48K with quantized KV" (vLLM single-card default).

**Quality cost** of going from int4-AutoRound to UD-Q3_K_XL is small on Qwen3.6-27B (the model is quantization-friendly), but real on harder reasoning benchmarks. Trade-off is yours to weigh.

---

## Pros

| Pro | Detail |
|---|---|
| **Lightweight** | Single binary, ~50 MB. No Docker required (though Docker images exist). |
| **Fastest cold start** | ~30 sec from launch to first token. vLLM takes ~2 min. |
| **Lowest VRAM overhead** | No inference-framework taxes — pure model + KV cache. |
| **GGUF format** | Many quant options (Q4_K_M, Q5_K_S, IQ4_XS, etc.). Easy to swap. |
| **Cross-platform** | Works on AMD (ROCm), Intel (oneAPI), Apple Silicon (Metal), CPU-only. vLLM is NVIDIA-only. |
| **Active community** | Lots of distros — Ollama, LM Studio, LocalAI, koboldcpp, etc. |
| **Luce DFlash fork available** | If you want spec-decode equivalent to MTP, [Luce's fork](https://github.com/luce-spec/llama-cpp-dflash) ships DFlash N=5 for Qwen3.6-27B. |

## Cons

| Con | Detail |
|---|---|
| **Qwen3-Next support still landing** | Need the right binary build. Mainline `llama.cpp` works but lags vLLM on edge cases (some attention variants, MTP head loading). |
| **Server feature parity behind vLLM** | Upstream `llama-server` doesn't expose `--enable-auto-tool-choice`. Need a wrapper (Open WebUI, LM Studio, Ollama with custom modelfile) for tool-call extraction. |
| **No TurboQuant equivalent** | KV cache is fp16 / fp8 / q4_0 / q5_1 / q8_0 / **turbo3 (in Tom's fork)**. None as compact as vLLM's TQ3 → max usable ctx is ~64K with Q4_K_M on a single 3090. |
| **Concurrent serving is sluggish** | `llama-server` forks per request. Two simultaneous requests → second waits or both slow. Not designed for multi-tenant. |
| **DFlash needs a fork** | The Luce DFlash fork is server-only and forks per request — sluggish chat UX, fine for long generation. Mainline llama.cpp doesn't have spec-decode for Qwen3-Next family yet. |

---

## Quick recipe — mainline llama.cpp + GGUF

### 1. Get a GGUF quant

[Unsloth ships Qwen3.6-27B GGUFs](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) (Q4_K_M ~16 GB, IQ4_XS ~14 GB, Q5_K_S ~18 GB). Or download the full BF16 and quantize yourself with `llama-quantize`.

> ⚠️ **Don't use `aria2c` to download multi-GB GGUFs.** It silently corrupts files during stall cycles — they'll have the right size but wrong bytes. Use `hf download` instead. Always `sha256sum` verify if available.

```bash
# Use hf CLI (pip install 'huggingface-hub[hf_transfer]')
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir /mnt/models/gguf/qwen3.6-27b/
```

Confirm size matches the HuggingFace listing. If a `sha256` is published, verify it.

### 2. Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

For ROCm: `-DGGML_HIPBLAS=ON`. For Apple Silicon: builds with Metal by default.

### 3. Launch the server

For a sane mid-context default (65K, plenty for chat + light agent work):

```bash
/opt/llama.cpp/build/bin/llama-server \
  -m /mnt/models/gguf/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf \
  -c 65536 \
  --host 0.0.0.0 --port 8020 \
  -ngl 999 \
  --jinja
```

`-ngl 999` puts all layers on GPU (use `-ngl 35` or similar to split with CPU if you have less VRAM).
`--jinja` enables chat template processing.

#### Going to 262K (full Qwen3.6 context) on a single 3090

This works on llama.cpp because Q4_K_M is ~16 GB on disk + a quantized KV cache at q4_0 leaves comfortable headroom. The vLLM stack can't get here on one card without splitting prompts — this is genuinely llama.cpp's win.

Memory math at 262K context with Q4_K_M + q4_0 KV (single 3090):
- Model on disk: ~16 GB
- KV cache at 262K (q4_0 K + q4_0 V): ~5 GB
- **Total VRAM: ~21 GB, leaving ~3 GB headroom** for prompts and activation peaks

Recipe (community-reported, validated by multiple users on r/LocalLLaMA):

```bash
/opt/llama.cpp/build/bin/llama-server \
  -m /mnt/models/gguf/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf \
  -ngl 99 \
  -c 262144 \
  -np 1 \
  -fa on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --host 0.0.0.0 --port 8020 \
  --jinja
```

What each flag does:
- `-ngl 99` — offload all layers to GPU (some users use 999 instead, equivalent)
- `-c 262144` — full 262K context window
- `-np 1` — single user slot. **Don't enable multi-slot here** — the KV pool gets divided across slots, eating your headroom.
- `-fa on` — flash attention on (memory + speed both win on Ampere+)
- `--cache-type-k q4_0 --cache-type-v q4_0` — **the unlock** — see KV cache type table below

Sustained throughput at 262K with this config is typically **35-45 tok/s** on a stock 3090 (community-reported flat curve at any in-budget context).

### 4. Vision (optional)

Download the `mmproj` model:
```bash
hf download unsloth/Qwen3.6-27B-GGUF mmproj-F16.gguf --local-dir /mnt/models/gguf/qwen3.6-27b/
```

Add to launch:
```bash
--mmproj /mnt/models/gguf/qwen3.6-27b/mmproj-F16.gguf
```

### 5. Tool calls (limited)

`llama-server` doesn't have built-in `--enable-auto-tool-choice`. Workarounds:

- **Ollama** wraps llama.cpp and adds tool-call extraction (uses Modelfile's `TEMPLATE` directive). Easiest path.
- **Open WebUI** can do client-side tool-call extraction from `<tool_call>...</tool_call>` strings.
- **Custom wrapper** — proxy that parses tool-call XML out of completions before returning to client.

For first-class tool calls in OpenAI format, vLLM is still the easiest option.

---

## Recipe — DFlash N=5 via Luce fork (for code workloads)

If you want spec-decode equivalent to vLLM's MTP path:

```bash
git clone https://github.com/luce-spec/llama-cpp-dflash /opt/llama-cpp-dflash
cd /opt/llama-cpp-dflash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# Download draft model (~500 MB)
hf download luce-spec/dflash-qwen3.6-27b-N5 --local-dir /mnt/models/gguf/qwen3.6-27b-dflash/

# Launch
/opt/llama-cpp-dflash/build/bin/llama-server \
  -m /mnt/models/gguf/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf \
  --draft /mnt/models/gguf/qwen3.6-27b-dflash/dflash-N5.gguf \
  --draft-max 5 \
  --draft-min 1 \
  -c 65536 \
  -ngl 999 \
  --host 0.0.0.0 --port 8004 \
  --jinja
```

Measured on this stack (single 3090, Q4_K_M main + DFlash N=5 draft, code prompts): **~106 TPS mean code TPS**, AL 4.74, accept 30.6%. Matches Luce's published README numbers.

**Trade-off:** the server forks per request, so chat UX feels sluggish (second request waits on first). For long generation tasks (single-shot code synthesis, document summarization), the per-request fork is fine.

---

## Tuning levers

### `--cache-type-k` / `--cache-type-v` — the biggest single lever

This is the most consequential knob on a 24 GB card. Most tutorials don't cover it.

| KV type | Per-token bytes | Fits at 262K on 24 GB? | Decode speed (vs q4_0) | Notes |
|---|---|---|---|---|
| `f16` (default) | ~12 KB | ❌ doesn't fit | n/a | Default in many guides — wrong choice for max-ctx on consumer cards |
| `q8_0` | ~6 KB | ⚠️ fits at ~23 GB but slow | **~3× slower** | Community-reported on Qwen3.6 27B + 3090; flash-attention path doesn't optimize q8 the way it does q4_0 |
| **`q4_0`** ⭐ | ~3 KB | ✅ fits at ~21 GB | full speed (40 tok/s flat) | The right pick for max context on consumer hardware |
| `turbo3` (Tom's fork) | ~2 KB | ✅ fits with margin | full speed (with the fork) | Even more compact; not yet upstream — watch [llama.cpp PR #21089](https://github.com/ggerganov/llama.cpp/pull/21089) |

The q8 → q4_0 jump is **counter-intuitive** because q8 is "higher precision" — you'd expect it to be slower for higher quality. In practice on this hardware:
- q8 KV cache hits a slower kernel path on flash-attention
- The "more precision" gain on KV is invisible at quant-noise levels of a 4-bit model
- q4_0 is the dominated choice on every axis except theoretical KV precision

**Test it on your own rig** before trusting the 3× claim — but if you're on `f16` or `q8` KV at 262K and seeing slow decode, swap to q4_0 first.

### Other levers

- **`--ctx-size`** — 65K is the comfortable default ceiling with f16 KV; 262K only works with q4_0 KV (see table above).
- **`--threads N`** — CPU threads for non-GPU ops. Set to physical-cores / 2 typically.
- **`-fa on` (flash attention)** — usually faster on modern GPUs. Required for the q4_0 KV path to perform well.

---

## When llama.cpp is the right pick

- ✅ You want minimal setup (single binary, no Docker)
- ✅ You're on AMD / Intel / Apple Silicon (vLLM is NVIDIA-only)
- ✅ You're embedding inference in another tool (LM Studio, Ollama, Faraday, etc.)
- ✅ You don't need concurrent multi-tenant serving
- ✅ You're OK with no first-class tool-call extraction (or use Ollama as a wrapper)

## When to use vLLM instead

- You need full OpenAI API parity (tools, streaming, structured output)
- You want max context (192K+) on a single 3090 — TurboQuant only available on vLLM
- You need concurrent serving (multi-tenant)
- You want MTP spec-decode (the integrated head, not DFlash)
- You're hitting llama.cpp's Qwen3-Next limitations and want the actively-developed path

---

## Watch list (when llama.cpp catches up)

- [llama.cpp PR #21089](https://github.com/ggerganov/llama.cpp/pull/21089) — TurboQuant KV cache landing (CPU first, CUDA follow-on). When CUDA path lands, `turbo3` becomes a first-class option on llama.cpp.
- Mainline Qwen3-Next dense / hybrid attention support — track upstream issues if you're hitting bugs.
- DFlash mainline integration — currently fork-only.

---

## See also

- [VLLM.md](VLLM.md) — what this repo ships
- [SGLANG.md](SGLANG.md) — the third option (currently blocked)
- [Luce z-lab's llama-cpp-dflash](https://github.com/luce-spec/llama-cpp-dflash) — DFlash fork
- [Unsloth Qwen3.6-27B GGUFs](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) — pre-quantized weights
