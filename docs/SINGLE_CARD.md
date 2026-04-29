# Single 3090 — what fits, how to run it

You have **one RTX 3090 (24 GB VRAM)**. This page is the front door for picking a config and knowing what to expect. The model-specific deep dives (quants, Genesis patches, engine internals) live elsewhere — links at the bottom.

---

## TL;DR — pick by workload

| What you're doing | Compose | Narr / Code TPS | Why |
|---|---|---|---|
| Chat / Q&A (≤20K) | [`fast-chat.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.fast-chat.yml) | **52 / 67** | Max TPS, fp8 KV, plus PN8 = ~−800 MiB |
| Tool-using IDE agents (Cline / Cursor / Copilot Gateway) | [`tools-text.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.tools-text.yml) | **51 / 65** | 75K ctx, **Cliff 1 closed** via Genesis PN8 (2026-04-29) |
| General-purpose default (≥20K, vision + tools) | [`docker-compose.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.yml) | **50 / 67** | 48K, TQ3 KV, prefill-safe |
| Long single prompts (RAG / summarization, no vision) | [`tools-text.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.tools-text.yml) (vLLM 75K) **or** [llama.cpp recipe](../models/qwen3.6-27b/llama-cpp/) (262K) | 51/65 (vLLM) · 21/21 (llama.cpp) | fp8 KV avoids GDN cliff up to ~60K; llama.cpp avoids cliffs entirely |
| Frontier 192K + vision | [`long-vision.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.long-vision.yml) | 51 / 68 | Engine ceiling; ⚠️ Cliff 1 still fires on >25K tool prefills |
| Frontier 205K text-only | [`long-text.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.long-text.yml) | 50 / 66 | Engine ceiling |
| Easy mode (one Docker line, 262K, no patches) | [`llamacpp/default`](../models/qwen3.6-27b/llama-cpp/compose/docker-compose.yml) | 21 / 21 | Q3_K_XL + q4_0 KV; no prefill cliffs anywhere |

Run any of these via `bash scripts/launch.sh` (interactive) or `bash scripts/switch.sh <variant>`.

---

## VRAM budget on 24 GB

![Per-card VRAM allocation, single-card section](img/vram-budget-dual.png)

What this says about single-card constraints:

- **Model weights** consume ~14 GB (AutoRound INT4 / GGUF Q3_K_XL). Half the card.
- **KV cache** is the next biggest line; its size depends on `--kv-cache-dtype` × ctx. fp8 ≈ 1 byte/token/(layer×head); TQ3 ≈ 0.4 bytes/token/(layer×head); fp16 ≈ 2 bytes/token/(layer×head).
- **Vision tower** (mmproj) costs ~0.5–1.0 GB extra when on.
- **Activations + cudagraph pools** is what's left. At `--gpu-memory-utilization 0.92` (default 48K) you have 2-3 GB of activation headroom — comfortable. At 0.98 (long-vision / long-text), <0.5 GB — that's where prefill cliffs fire.

For the cross-card TP=2 picture, see [`DUAL_CARD.md`](DUAL_CARD.md).

---

## Pick a config

### Chat / Q&A — `fast-chat.yml`

**Workload:** single user, short conversations, browser UIs (Open WebUI, LibreChat). Intermittent.

20K context with fp8 KV. Smallest, fastest, no prefill cliff at this depth. PN8 enabled (2026-04-29) — frees ~800 MiB headroom. **Caveat:** 20K fills faster than people expect with system prompts + tool defs + history. A 30-turn coding chat can exceed 20K. Bump to default 48K if conversations grow.

### Tool-using IDE agents — `tools-text.yml`

**Workload:** Cline, Cursor, GitHub Copilot LLM Gateway, Continue.dev, Hermes — anything that calls tools (`read_file`, `run_in_terminal`, `web_fetch`) and expects structured `tool_calls[]` responses.

75K context with fp8 KV + Genesis MTP n=3 + PN8. As of 2026-04-29 this compose's **`verify-stress.sh` 25K-token tool-prefill check passes clean** — Cliff 1 closed via PN8 freeing ~900 MiB. The **only single-card path that's safe with big tool returns**.

**Two gotchas worth surfacing:**

- **VS Code Copilot LLM Gateway sends ~20K tokens of tool schema** in every request. `fast-chat.yml` (20K cap) won't work for Copilot — this is why we recommend `tools-text.yml` (75K).
- **Truncated `max_tokens`** (some clients send 64) cuts tool-call JSON mid-string — produces malformed output that some gateways report as "empty response." That's a client config issue, not the server. See [FAQ: Copilot Gateway](FAQ.md#will-this-work-with-vs-code-github-copilot-llm-gateway).

### General-purpose default — `docker-compose.yml`

**Workload:** anything that doesn't fit the above narrowly. Mixed chat + light tools + occasional images.

48K + TQ3 KV + Genesis P65/P66/P64 + MTP n=3 + vision tower. Production-safe — below both prefill cliffs at 0.92 mem-util. **Verify-full's** 10K/30K/60K/90K needle ladder passes; tool-prefill check at 15K passes.

### Long single prompts — `tools-text.yml` (vLLM) or `llama.cpp` recipe

**Workload:** Loading a long document or repo in one shot, asking questions about it. RAG ingest, single-shot summarization. Cold prefill cost is the dominant factor.

- **vLLM `tools-text.yml`** (75K + fp8 + no vision): tested up to 60K-token single prompts. Beyond that, Cliff 2 (DeltaNet GDN forward) fires regardless of mem-util.
- **llama.cpp** (262K + Q4_K_M + q4_0 KV): the only single-card path to the model's natural max. `bash scripts/switch.sh llamacpp/default`. ~21 TPS, but **no prefill cliffs anywhere** — this is the robust choice for unpredictable input sizes. See [llama-cpp/README](../models/qwen3.6-27b/llama-cpp/README.md) for quant + KV options.

**Cold prefill at 60K+ is genuinely slow** — 30-60 seconds for a fresh 50K-token doc on vLLM single-card. Use prefix caching aggressively if you'll re-query the same doc.

### Vision-heavy — default 48K with vision on

**Workload:** Multimodal pipelines. Code-screenshot review, document OCR-style tasks, visual Q&A.

`docker-compose.yml` ships with vision tower (mmproj) active. Tower is small (~1 GB VRAM), comfortable headroom. For more context with vision, opt into `long-vision.yml` (192K) — but read the prefill caveat.

### Frontier context — `long-vision.yml` (192K + vision) or `long-text.yml` (205K text-only)

**Workload:** "I need 100K+ context for whole-codebase / long-document workflows." Steady-state context accumulation across many small turns, NOT stuffing 192K of new tokens in one request.

- `long-vision.yml`: 192K + vision (engine ceiling at 0.98 mem-util). PN8 testing showed +6K headroom potential (192→198K), opt-in via uncommenting the env var.
- `long-text.yml`: 205K text-only (engine ceiling capped by attention block-size divisor at ~206K).

**Critical caveat — Cliff 1 still fires on TQ3 paths:** ≥25K-token tool prefills will OOM regardless of mem-util tuning, because the 138 MiB allocate is an FFN intermediate-buffer activation peak (not draft-model footprint that PN8 fixes). For tool-using agents that return large blobs, **drop back to `tools-text.yml` or `docker-compose.yml`** — those are below the cliff. See [FAQ: prefill cliff](FAQ.md#whats-a-prefill-cliff).

### Easy mode — llama.cpp Q3_K_XL

**Workload:** First-time users, "just give me something that works." No Genesis, no AutoRound, no patched vLLM source. One Docker pull + one GGUF.

`bash scripts/switch.sh llamacpp/default`. Q3_K_XL (Unsloth dynamic) + q4_0 KV at 262K + vision (mmproj). All `verify-stress.sh` checks pass clean — **no prefill cliffs anywhere on this engine.** Trade is throughput: ~21 TPS, ~2.5× slower than vLLM. Quant validated independently by [Benjamin Marie's Kaitchup eval](https://kaitchup.substack.com/p/summary-of-qwen36-gguf-evals-updating).

---

## What single-card can't do

| Want | Why not on 1× | What you'd need |
|---|---|---|
| 4 concurrent streams at 262K + vision | KV pool too small for 4 × full ctx | TP=2 (see DUAL_CARD.md) |
| Peak code TPS (>100 TPS on quicksort prompt) | DFlash N=5 needs head_size=256 + non-causal — vLLM head-dim split | TP=2 + DFlash |
| ≥25K tool prefills + 192K context together | Cliff 1 (FFN activation peak), no fix yet | TP=2 splits activation memory across cards |
| Single-prompt >60K tokens on vLLM | Cliff 2 (DeltaNet GDN forward), no fix yet | TP=2 OR llama.cpp 262K (different engine) |

---

## Common pitfalls (single-card specifics)

### Prefill cliffs

- **Cliff 1** — FFN intermediate-buffer activation peak (138 MiB allocate at `intermediate_size × max-num-batched-tokens`). Fires on long-ctx composes at >0.95 mem-util when prefill batch needs the buffer. **Closes on `tools-text.yml`** (FP8 KV path) since 2026-04-29 via Genesis PN8. Still fires on TQ3 paths (`long-vision.yml`, `long-text.yml`).
- **Cliff 2** — DeltaNet GDN forward OOM at 50-60K single-prompt regardless of mem-util. In `fla.ops` upstream, no file-replacement patch available. Watch [vllm#40914](https://github.com/vllm-project/vllm/pull/40914) and [FlashQLA](https://github.com/QwenLM/FlashQLA) for upstream fixes.

### VRAM peak vs idle

`nvidia-smi` at boot ≠ peak. Boot shows weights + KV pool reservation. **Peak adds activation buffers during prefill** — typically +500-1500 MiB. If `nvidia-smi` shows 23.5/24 GB at idle, you have ~500 MiB for prefill activations — not enough for the 138 MiB-class buffer at long ctx. Drop mem-util by 0.03 if you need the headroom.

### Tool-call extraction needs `--enable-auto-tool-choice`

vLLM ships this off by default. Our composes set `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice`. If you're rolling your own compose, both are required.

---

## Quick start

```bash
# 1. Setup (downloads model, clones Genesis, ~20 min cold)
bash scripts/setup.sh qwen3.6-27b

# 2. Pick + boot via wizard (asks engine + workload)
bash scripts/launch.sh

# 3. Or skip the wizard:
bash scripts/launch.sh --variant vllm/tools-text   # IDE agent path
bash scripts/launch.sh --variant llamacpp/default  # easy mode

# 4. Sanity test
curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":30}'

# 5. Switch later without re-running setup
bash scripts/switch.sh vllm/long-vision    # for example
bash scripts/switch.sh --list              # show all variants
```

---

## Models supported on single 3090

- **[Qwen3.6-27B](../models/qwen3.6-27b/)** — primary model. Quant choices, Genesis patches, engine internals all in the model directory.
- More models coming. As they're added, this section will list which single-card configs each one supports.

---

## Deep dives

- **[Model README](../models/qwen3.6-27b/)** — quant choices (AutoRound INT4 / GGUF Q3_K_XL), Genesis patch surface, what's working / what's not.
- **[INTERNALS.md](../models/qwen3.6-27b/INTERNALS.md)** — engineering rationale (Genesis P65/P66/PN8, Marlin pad fork, MTP, the cascade bug, upstream tracker).
- **[VRAM allocation diagram](../models/qwen3.6-27b/README.md#vram-allocation-across-configs)** — full per-config breakdown across single + dual.
- **[FAQ.md](FAQ.md)** — common questions (4090 / 5090 support, why MTP not EAGLE, Copilot Gateway, what's a cliff, etc.).
- **[EXAMPLES.md](EXAMPLES.md)** — Python / TS / curl client snippets + IDE connection settings.
- **[HARDWARE.md](HARDWARE.md)** — Ampere SM 8.6 specifics, NVLink (declined), power caps.
- **[DUAL_CARD.md](DUAL_CARD.md)** — when you need what single-card can't deliver.
