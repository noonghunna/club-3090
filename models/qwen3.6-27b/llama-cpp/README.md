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

## Docker compose (recommended)

Two compose files in [`compose/`](compose/) — both use the official `ghcr.io/ggml-org/llama.cpp:server-cuda` image, no custom build needed.

### `docker-compose.yml` — max context, single slot, vision

Showcase: full **262K context** on one 3090 with vision + q4_0 KV.

```bash
cd models/qwen3.6-27b/llama-cpp/compose
MODEL_DIR=/mnt/models/gguf docker compose up -d
```

Memory budget: 14.5 GB (Q3_K_XL) + 4.5 GB KV @ 262K + 0.8 GB mmproj ≈ 20 GB / 24 GB.

### `docker-compose.concurrent.yml` — 4 parallel slots, vision

Trade max context for parallelism. Same image, `--parallel 4` + smaller ctx pool.

---

## Recipes (host-binary alternative)

[`recipes/`](recipes/) contains shell scripts that launch a host-built `llama-server` with the same flags. Use these if you've built llama.cpp natively (e.g. for AMD/Intel/Apple Silicon) and don't want Docker.

- **`single-card-default.sh`** — 65K ctx, Q4_K_M
- **`single-card-max-ctx.sh`** — 262K ctx, Q4_K_M + q4_0 KV
- `dual-card.sh` — TBD; llama.cpp supports multi-GPU but we haven't validated configs for this model

---

## Measured TPS (2026-04-28, club-3090 substrate)

| Config | Quant | KV | Ctx | Vision | Narr TPS | Code TPS | Notes |
|---|---|---|---|---|---|---|---|
| docker-compose.yml | UD-Q3_K_XL | q4_0 | 262K | ✅ | 21 | 21 | Flat across context depth — same TPS at 65K and 262K |
| `+ --spec-type ngram-mod` (recipe) | Q4_K_M | q8_0 | 32K | ❌ | 22 | **26** | +25% on code via draftless n-gram spec-decode |

The Q3_K_XL number at 262K is **lower than community-reported 35-45 tok/s** ([Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1sx8uok/) + earlier 2026-04-23 measurements showing 28.5 TPS on Q4_K_M). We're investigating whether mainline llama.cpp regressed between commits `9ab47e7d8` (2026-04-23) and `0d0764dfd` (current). For absolute speed today, **vLLM patched is ~2.5× faster** on the same hardware (51-55 narr / 67-70 code) — see [BENCHMARKS](../../../BENCHMARKS.md). llama.cpp's value proposition here is **simplicity + max context + multi-platform**, not throughput.

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
| **UD-Q3_K_XL** ⭐ ([Unsloth dynamic](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)) | ~14.5 GB | Small quality cost on Qwen3.6 (quantization-friendly); real on harder reasoning | **Our default** — picked for huge ctx + multi-shot headroom. Independently validated as the best accuracy / token-efficiency / footprint balance by Benjamin Marie's eval (see below). |
| Q3_K_M | ~13.6 GB | More aggressive 3-bit | When you absolutely need every spare GB for KV |

**Independent third-party eval — Q3_K_XL is the right pick.** Benjamin Marie ([@bnjmn_marie](https://x.com/bnjmn_marie)) ran an H100 GGUF benchmark sweep on Qwen3.6-27B (Q2_K_XL / IQ3_XXS / Q3_K_XL / IQ2_XXS, plus abliterated variants) and concludes Q3_K_XL is the optimal balance between accuracy, token efficiency, and memory footprint — performance drops sharply below 10 GB, and IQ2_XXS produces server errors. Charts + methodology in *[Summary of Qwen3.6 GGUF Evals](https://kaitchup.substack.com/p/summary-of-qwen36-gguf-evals-updating)* (Kaitchup #139, 2026-04-24). We use those findings as our quality lens; our number on this hardware is the speed lens (21 TPS @ 262K + vision via Docker compose).

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
