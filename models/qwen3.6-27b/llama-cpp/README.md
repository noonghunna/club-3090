# Qwen3.6-27B on llama.cpp

The lightweight path. Best for: max context on a single 3090, lightest cold-start, non-NVIDIA hardware, embedded use, anything where you'd rather skip Docker.

## When to pick llama.cpp over vLLM for this model

- ✅ You want **262K context on a single 3090** (vLLM caps at 48K safe / 192K opt-in with caveats)
- ✅ You're on AMD / Intel / Apple Silicon (vLLM is NVIDIA-only)
- ✅ You're embedding inference in another tool (LM Studio, Ollama, Faraday)
- ✅ You don't need concurrent multi-tenant serving
- ✅ You're OK with no first-class tool-call extraction (or use Ollama as a wrapper)

## When NOT to pick llama.cpp

- ❌ You need MTP spec-decode (only DFlash N=5 via [Luce z-lab fork](https://github.com/luce-spec/llama-cpp-dflash); mainline doesn't have it)
- ❌ You need full OpenAI API parity for tool calling, structured output
- ❌ You're serving multi-user (llama-server forks per request — sluggish under concurrent load)

For full pros/cons + general llama.cpp tuning, see [`/docs/engines/LLAMA_CPP.md`](../../../docs/engines/LLAMA_CPP.md).

---

## Recipes

[`recipes/`](recipes/) contains shell scripts that launch `llama-server` with the right flags for this model.

### `single-card-default.sh`

Sane mid-context default (65K, plenty for chat + light agent work). Q4_K_M GGUF.

### `single-card-max-ctx.sh`

**The standout recipe.** Full 262K context on a single 3090 via Q4_K_M + q4_0 KV. Memory math:
- Model: ~16 GB
- KV at 262K (q4_0 K + q4_0 V): ~5 GB
- Total: ~21 GB; ~3 GB headroom

Sustained throughput at 262K: **35-45 tok/s** on a stock 3090 (community-reported flat curve at any in-budget context).

### `dual-card.sh`

TBD — llama.cpp supports multi-GPU but we haven't validated configs for this model. Open an issue if you have one working.

---

## Quick start

```bash
# 1. Get a GGUF quant (recommended: Unsloth's Q4_K_M)
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir /mnt/models/gguf/qwen3.6-27b/

# 2. Build llama.cpp with CUDA support
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j

# 3. Run a recipe
cd <repo>/models/qwen3.6-27b/llama-cpp/recipes
bash single-card-max-ctx.sh
```

---

## Quant recommendations

GGUFs of this model are at [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF). Sizes and trade-offs:

| Quant | Disk | Quality | When to pick |
|---|---|---|---|
| Q4_K_M | ~16.8 GB | Strong baseline | Default; pairs well with q4_0 KV at 262K |
| Q5_K_S | ~19 GB | Slightly higher quality | If you have ~3 GB extra headroom |
| UD-Q3_K_XL ([Unsloth dynamic](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)) | ~14.5 GB | Small quality cost on Qwen3.6 (quantization-friendly); real on harder reasoning | When you want even more KV headroom for huge ctx + multi-shot |
| Q3_K_M | ~13.6 GB | More aggressive 3-bit | When you absolutely need every spare GB for KV |

**⚠️ Don't use `aria2c` to download multi-GB GGUFs.** It silently corrupts files during stall cycles — they'll have the right size but wrong bytes. Use `hf download` instead, and `sha256sum` verify if a hash is published.

---

## Vision (mmproj)

```bash
hf download unsloth/Qwen3.6-27B-GGUF mmproj-F16.gguf --local-dir /mnt/models/gguf/qwen3.6-27b/

# Add to launch: --mmproj /mnt/models/gguf/qwen3.6-27b/mmproj-F16.gguf
```

Vision works via the mmproj model. Sample text+image queries are OpenAI-compat.

---

## Tool calls (limited)

`llama-server` doesn't have built-in `--enable-auto-tool-choice`. Workarounds:

- **Ollama** wraps llama.cpp and adds tool-call extraction. Easiest.
- **Open WebUI** can extract `<tool_call>` from completions client-side.
- **Custom wrapper** — proxy that parses tool-call XML before returning.

For first-class tool calls in OpenAI format, vLLM is still the easier option. See [`../vllm/`](../vllm/).

---

## DFlash spec-decode (Luce z-lab fork)

If you want spec-decode equivalent to vLLM's MTP, build [Luce's fork](https://github.com/luce-spec/llama-cpp-dflash) and download the DFlash N=5 draft. See [`/docs/engines/LLAMA_CPP.md`](../../../docs/engines/LLAMA_CPP.md#recipe--dflash-n5-via-luce-fork-for-code-workloads) for the full recipe. Measured ~106 TPS code on this stack.
