# Qwen3.6-27B on llama.cpp

The lightweight path. Best for: max context on a single 3090, lightest cold-start, non-NVIDIA hardware, embedded use, anything where you'd rather skip Docker.

## When to pick llama.cpp over vLLM for this model

- ✅ You want **262K context on a single 3090** (vLLM caps at 48K safe / 192K opt-in with caveats)
- ✅ You're on AMD / Intel / Apple Silicon (vLLM is NVIDIA-only)
- ✅ You're embedding inference in another tool (LM Studio, Ollama, Faraday)
- ✅ You don't need concurrent multi-tenant serving
- ✅ You're OK with no first-class tool-call extraction (or use Ollama as a wrapper)

## When NOT to pick llama.cpp

- ⚠️ You need spec-decode *and* maximum context on one card — DFlash is native on mainline now (see below), but enabling it costs context (the drafter inherits `-c`)
- ❌ You need full OpenAI API parity for tool calling, structured output
- ❌ You're serving multi-user (llama-server forks per request — sluggish under concurrent load)

For full pros/cons + general llama.cpp tuning, see [`/docs/engines/LLAMA_CPP.md`](../../../docs/engines/LLAMA_CPP.md).

---

## Docker compose (recommended)

Three compose variants in [`compose/single/`](compose/single/) — all use the official `ghcr.io/ggml-org/llama.cpp` image (CUDA), **no custom build needed**, **no club-3090 patches** (unlike our vLLM track). MTP PR #22673 has merged upstream so this image has it natively. The composes are **pinned to build `server-cuda-b9967`** (validated 2026-07-11 via a full quality A/B on Tess: think-OFF total tie, think-ON +4 — the improvement is thinking-path-specific; earlier pin b9246 validated 2026-05-20) — *not* the rolling `:server-cuda` tag, because that tag regressed at `b9282` (broken lib packaging → crash loop, [#187](https://github.com/noonghunna/club-3090/issues/187)). To follow a newer build, override `LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-bXXXX` (validate it first). Bench numbers in BENCHMARKS were measured on the pin current at their date; expect ±5% drift across builds.

### `single/unsloth-q4km/mtp.yml` — MTP n=2, 200K ctx, no vision

The single-card speed + context workhorse: ~51/60 TPS (narr/code), **200K ctx** (max-safe default @ `-ub 512` — fills cleanly with ~1.1 GB margin; 131K @ `-ub 1024` for faster prefill; 262K is the native max but *boots-not-fills*, see [`docs/CLIFFS.md`](../../../docs/CLIFFS.md)), 7/7 verify-stress boundary checks (incl. 60K + 91K needle recall), 102/150 (68%) on the 8-pack quality matrix. Best for IDE agents, opencode, Hermes, long-multi-turn agentic. Q4_K_M MTP GGUF (`unsloth/Qwen3.6-27B-MTP-GGUF` Q4_K_M).

### `single/unsloth-q4km/bounded-thinking.yml` — MTP n=2, 200K ctx, reasoning on, grammar per request

Structured-CoT variant for llama.cpp. It is intentionally the same runtime envelope as `single/unsloth-q4km/mtp.yml` — Q4_K_M MTP GGUF, q4_0 KV, `-ub 512`, 200K context, no vision — with `REASONING=on` as the default. The grammar is not baked into the server; clients pass GBNF in the OpenAI-compatible request body `grammar` field. Use [`tools/grammar-eval/deepseek-scratchpad.llamacpp.gbnf`](../../../tools/grammar-eval/deepseek-scratchpad.llamacpp.gbnf) as the recommended default, or pass the original andthattoo / Holiday alternates client-side. See [`docs/STRUCTURED_COT.md`](../../../docs/STRUCTURED_COT.md) for request examples and validation status.

### `single/unsloth-q4km/mtp-vision.yml` — MTP n=2, 160K ctx, vision on

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
| unsloth-q4km/mtp.yml | UD-Q3_K_XL | q4_0 | 262K | ✅ | 21 | 21 | Flat across context depth — same TPS at 65K and 262K |
| `+ --spec-type ngram-mod` | Q4_K_M | q8_0 | 32K | ❌ | 22 | **26** | +25% on code via draftless n-gram spec-decode |

The Q3_K_XL number at 262K is **lower than community-reported 35-45 tok/s** ([Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1sx8uok/) + earlier 2026-04-23 measurements showing 28.5 TPS on Q4_K_M). We're investigating whether mainline llama.cpp regressed between commits `9ab47e7d8` (2026-04-23) and `0d0764dfd` (current). For absolute speed today, **vLLM patched is ~2.5× faster** on the same hardware (51-55 narr / 67-70 code) — see [BENCHMARKS](../../../BENCHMARKS.md). llama.cpp's value proposition here is **simplicity + max context + multi-platform**, not throughput.

---

## Quick start

```bash
# 1. Get the MTP-enabled GGUF
#    Easiest: WEIGHTS=gguf bash scripts/setup.sh qwen3.6-27b   (downloads Q4_K_M + mmproj,
#    SHA-verified, into the path below; skips Genesis). Or download it directly:
hf download unsloth/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-Q4_K_M.gguf \
  --local-dir $MODEL_DIR/qwen3.6-27b-gguf/unsloth-mtp-q4km

# 2. Launch via Docker compose (recommended)
cd <repo>/models/qwen3.6-27b/llama-cpp/compose
MODEL_DIR=$MODEL_DIR docker compose -f single/unsloth-q4km/mtp.yml up -d
curl http://localhost:8020/v1/models
```

For host-built llama.cpp (AMD/Intel/Apple Silicon without Docker), use the
same flags from `compose/single/unsloth-q4km/mtp.yml` adapted to your binary. Key flags:
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

## DFlash spec-decode — native on mainline (no fork needed)

**Updated 2026-07-26 (#759).** DFlash is a native spec type on mainline llama.cpp — `--spec-type draft-dflash`, implemented in `common/speculative.cpp`. The previous "Luce fork only" guidance here was stale. Verified present on **b10066**, **b10088** and **b10103**; it loads the published [Anbeeld DFlash GGUF](https://huggingface.co/Anbeeld/Qwen3.6-27B-DFlash-GGUF) drafters with no fork.

```bash
llama-server \
  -m   .../unsloth-q5ks/Qwen3.6-27B-Q5_K_S.gguf \
  --spec-type draft-dflash \
  --spec-draft-model .../anbeeld-dflash-q4km/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 15 \
  -ngl 99 -ngld 99 --ctx-size 32768 -b 2048 -ub 512 -np 1 \
  --flash-attn on --cache-type-k q5_0 --cache-type-v q4_1 --jinja \
  --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0.0
```

**Community-measured on 1× RTX 3090 Ti, TP=1** (@superbiche, #759 — code workload, 3 warm + 5 measured):

| config | gen tok/s (mean / median) | acceptance |
|---|--:|--:|
| no-spec (Q5_K_S target only) | 28.79 / 28.92 | — |
| DFlash, mainline b10066 | **68.16 / 64.03** | 0.238 mean |
| speedup | **~2.37× mean, ~2.22× median** | 3.6–6.1 tok/accepted-run |

Confirm engagement from the server log rather than inferring it — you should see:

```
common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'
  block_size=16, mask_token_id=248070, n_extract=5
```

plus a `draft acceptance = X (A accepted / G generated)` line per request.

### ⚠️ Spec-decode costs you context — budget for it

**The drafter inherits the main `-c`**, and mainline has no flag to cap the draft context independently (buun's fork auto-sets `-cd 256`). So enabling DFlash reduces the context you can serve, on any card size:

- **1× 24 GB, Q5_K_S target (18.6 GB):** usable ctx caps around **32768** with DFlash. The `102400` default OOMs on compute buffers at prefill (#759).
- **2× 24 GB, IQ4_NL 27B target + MTP:** 262144 serves fine *without* spec-decode; the ceiling *with* it is **196608** (measured on this stack).

⚠️ **Quantizing the draft KV does not recover it.** `--spec-draft-type-k/-v q4_0` did not rescue the 2-card case — the binding constraint is the drafter's **compute buffers**, not its KV.

### `--spec-draft-n-max` is clamped to 15

DFlash was trained with `block_size=16`, so values above **15** are clamped. Asking for 16 wastes a run. (Poolside's own Laguna config likewise specifies `num_speculative_tokens: 15`.)

### Alternative: the Luce z-lab fork

[Luce's fork](https://github.com/Luce-Org/lucebox-hub) remains an option and measured ~106 TPS code on this stack, but it is **no longer required** — mainline gets most of that on a single card. Prefer mainline unless you specifically need something the fork adds.

### ⚠️ Acceptance is model-specific — verify, don't assume

DFlash acceptance varies enormously by target model, and a bad pairing looks like a catastrophic slowdown rather than a mild one. On this stack, **Laguna-S-2.1** with poolside's own fork and their official drafter measured **0.25% acceptance / −85% decode**, against Qwen3.6-27B's healthy 23.8%. Always read the `draft acceptance` line before trusting a spec-decode config: below ~10% you are paying for drafts nobody uses.
