# Qwen3.6-27B on llama.cpp

The lightweight path. Best for: max context on a single 3090, lightest cold-start, non-NVIDIA hardware, embedded use, anything where you'd rather skip Docker.

## When to pick llama.cpp over vLLM for this model

- ✅ You want **262K context on a single 3090** (vLLM caps at 48K safe / 192K opt-in with caveats)
- ✅ You're on AMD / Intel / Apple Silicon (vLLM is NVIDIA-only)
- ✅ You're embedding inference in another tool (LM Studio, Ollama, Faraday)
- ✅ You don't need concurrent multi-tenant serving
- ✅ You're OK with no first-class tool-call extraction (or use Ollama as a wrapper)

## When NOT to pick llama.cpp

- ❌ You need MTP spec-decode (only DFlash N=5 via [Luce z-lab fork](https://github.com/Luce-Org/lucebox-hub); mainline doesn't have it)
- ❌ You need full OpenAI API parity for tool calling, structured output
- ❌ You're serving multi-user (llama-server forks per request — sluggish under concurrent load)

For full pros/cons + general llama.cpp tuning, see [`/docs/engines/LLAMA_CPP.md`](../../../docs/engines/LLAMA_CPP.md).

---

## Docker compose (recommended)

Two compose variants in [`compose/single/`](compose/single/) — both use the official `ghcr.io/ggml-org/llama.cpp` image (CUDA), **no custom build needed**, **no club-3090 patches** (unlike our vLLM track). MTP PR #22673 has merged upstream so this image has it natively. The composes are **pinned to build `server-cuda-b9246`** (validated 2026-05-20) — *not* the rolling `:server-cuda` tag, because that tag regressed at `b9282` (broken lib packaging → crash loop, [#187](https://github.com/noonghunna/club-3090/issues/187)). To follow a newer build, override `LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-bXXXX` (validate it first). Bench numbers were measured on `b9246`; expect ±5% drift on newer builds.

### `single/mtp.yml` — MTP n=2, 262K ctx, no vision

The single-card speed + context workhorse: ~51/60 TPS (narr/code), 262K ctx (full native @ `-ub 512`; 131K @ `-ub 1024` for faster prefill), 7/7 verify-stress boundary checks (incl. 60K + 91K needle recall), 102/150 (68%) on the 8-pack quality matrix. Best for IDE agents, opencode, Hermes, long-multi-turn agentic. Q4_K_M MTP GGUF (`unsloth/Qwen3.6-27B-MTP-GGUF` Q4_K_M).

### `single/mtp-vision.yml` — MTP n=2, 160K ctx, vision on

Multimodal profile — combines MTP + vision (validated on build 9235, 2026-05-19). 160K default context on 24 GB with mmproj F16 mounted. Supports up to 192K with `UBATCH_SIZE=512`.

### Tuning knobs

Both composes expose llama.cpp's batch-size + KV controls without editing YAML:

| Env var | llama.cpp flag | Default | Sensible range on 24 GB | Notes |
|---|---|---:|---:|---|
| `CTX_SIZE` | `-c` | varies by variant | up to ~256K (q4_0 KV) | KV pool size. See per-variant defaults below. |
| `BATCH_SIZE` | `-b` | `4096` | `2048`-`8192` | Logical prompt-processing batch. Higher can improve prefill throughput if VRAM headroom allows. |
| `UBATCH_SIZE` | `-ub` | `1024` | `512`-`4096` | Physical microbatch. **Lower this first if long prompts OOM during prefill** — but it also has a major impact on max-context (see next section). |
| `KV_TYPE` | `--cache-type-k/-v` | `q4_0` | `q4_0`, `q5_0`, `q8_0` | Lower KV bits-per-value = more ctx fits at same VRAM (quality trade-off is small at q4_0 for this model). |

These are throughput-tuning knobs inside llama.cpp. They are orthogonal to
`ESTATE_GPUS` and `ESTATE_PORT`, which only isolate GPU assignment and host port
when `scripts/launch.sh --estate` boots multiple instances.

### Speed vs context — pick your trade-off

`UBATCH_SIZE` (the `-ub` chunked-prefill chunk) is doing two jobs at once: it caps the **per-pass activation buffer** (cliff-survival for tool prefill) AND it eats into the **VRAM budget that could otherwise go to KV cache**. We ship `1024` as the default sweet spot, but you can rebalance:

**For `llamacpp/mtp-vision` specifically** — the vision encoder (mmproj F16, ~0.8 GB) competes for the same VRAM budget. The shipped 49K ctx + ub=1024 is the **speed-optimal** point on a single 3090. If you need more ctx for agentic vision workloads (UI navigation, multi-step tool use, long screenshots-in-context), drop `-ub` to 512 and you can push context up to 192K with full cliff coverage:

```bash
# Tested 2026-05-20 on single 3090, verify-stress 7/7 (incl. 60K + 91K needle):
UBATCH_SIZE=512 CTX_SIZE=196608 bash scripts/switch.sh llamacpp/mtp-vision
```

| Config | ctx | VRAM | narr TPS | verify-stress | When to pick |
|---|---|---:|---:|:---:|---|
| shipped: `ub=1024` | 49K | 22.0 GB | **56.5** | 7/7 ✓ | speed-first, short context |
| override: `ub=512 CTX=131072` | 131K | 21.0 GB | 50.0 | 7/7 ✓ | balanced (extra headroom) |
| override: `ub=512 CTX=196608` | **192K** | 22.5 GB | 50.9 | 7/7 ✓ | **max ctx with cliff coverage** |

So ~10% TPS hit (56.5 → 50.9 narr) buys ~4× more context (49K → 192K). For pure-chat / short-prompt workloads, keep the default. For agentic vision, override.

**For `llamacpp/mtp` (no vision)** — the same `-ub` 512 trade applies but with smaller margins (no mmproj competing for VRAM). Probe with `UBATCH_SIZE=512 CTX_SIZE=196608 bash scripts/switch.sh llamacpp/mtp` if you need more than the shipped 131K — we haven't shipped this as a default but the lever is there.

**For `llamacpp/default`** — already at the model's training-max 262K ctx; `-ub` is not a useful lever (no ctx upside, only TPS cost). Keep the default `1024`.

---

## Measured TPS (2026-04-28, club-3090 substrate)

| Config | Quant | KV | Ctx | Vision | Narr TPS | Code TPS | Notes |
|---|---|---|---|---|---|---|---|
| docker-compose.yml | UD-Q3_K_XL | q4_0 | 262K | ✅ | 21 | 21 | Flat across context depth — same TPS at 65K and 262K |
| `+ --spec-type ngram-mod` | Q4_K_M | q8_0 | 32K | ❌ | 22 | **26** | +25% on code via draftless n-gram spec-decode |

The Q3_K_XL number at 262K is **lower than community-reported 35-45 tok/s** ([Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1sx8uok/) + earlier 2026-04-23 measurements showing 28.5 TPS on Q4_K_M). We're investigating whether mainline llama.cpp regressed between commits `9ab47e7d8` (2026-04-23) and `0d0764dfd` (current). For absolute speed today, **vLLM patched is ~2.5× faster** on the same hardware (51-55 narr / 67-70 code) — see [BENCHMARKS](../../../BENCHMARKS.md). llama.cpp's value proposition here is **simplicity + max context + multi-platform**, not throughput.

---

## Quick start

```bash
# 1. Get the MTP-enabled GGUF
hf download unsloth/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-Q4_K_M.gguf \
  --local-dir $MODEL_DIR/qwen3.6-27b-gguf/unsloth-mtp-q4km

# 2. Launch via Docker compose (recommended)
cd <repo>/models/qwen3.6-27b/llama-cpp/compose
MODEL_DIR=$MODEL_DIR docker compose -f single/mtp.yml up -d
curl http://localhost:8020/v1/models
```

For host-built llama.cpp (AMD/Intel/Apple Silicon without Docker), use the
same flags from `compose/single/mtp.yml` adapted to your binary. Key flags:
`-ngl 99 -fa on -c 262144 -ub 512 --cache-type-k q4_0 --cache-type-v q4_0
--spec-type draft-mtp --spec-draft-n-max 2 --jinja --reasoning off`.

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
hf download unsloth/Qwen3.6-27B-GGUF mmproj-F16.gguf --local-dir $MODEL_DIR/qwen3.6-27b-gguf/

# Add to launch: --mmproj $MODEL_DIR/qwen3.6-27b-gguf/mmproj-F16.gguf
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

If you want spec-decode equivalent to vLLM's MTP, build [Luce's fork](https://github.com/Luce-Org/lucebox-hub) and download the DFlash N=5 draft. See [`/docs/engines/LLAMA_CPP.md`](../../../docs/engines/LLAMA_CPP.md#recipe--dflash-n5-via-luce-fork-for-code-workloads) for the full recipe. Measured ~106 TPS code on this stack.
