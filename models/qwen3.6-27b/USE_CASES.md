# Qwen3.6-27B — Use cases

Practical guide matched to common workloads. Single-card and dual-card recipes side-by-side. For each: which compose to boot, why it fits, gotchas, limitations, tuning levers.

> If you're new to the stack, read the model [README](README.md) first. This doc assumes you've got it running and want to dial it in.

---

## Quick map

| Workload | 1× 3090 | 2× 3090 | TPS narr / code |
|---|---|---|---|
| General chat / Q&A (≤20K) | [`fast-chat.yml`](vllm/compose/docker-compose.fast-chat.yml) | [`dual.yml`](vllm/compose/docker-compose.dual.yml) | 55/70 (1×) · 69/89 (2×) |
| Tool-using agents (Cline / Roo / Cursor / Hermes) | [`yml`](vllm/compose/docker-compose.yml) (default) | [`dual.yml`](vllm/compose/docker-compose.dual.yml) | 51/68 (1×) · 69/89 (2×) |
| Coding (single-file or repo with tool truncation) | [`yml`](vllm/compose/docker-compose.yml) (default) | [`dual-dflash.yml`](vllm/compose/docker-compose.dual-dflash.yml) | 51/68 (1× MTP) · 82/125 (2× DFlash) |
| Long single prompts (RAG, summarization) | [`tools-text.yml`](vllm/compose/docker-compose.tools-text.yml) (75K) or [llama-cpp 262K](llama-cpp/recipes/) | [`dual-dflash-noviz.yml`](vllm/compose/docker-compose.dual-dflash-noviz.yml) (200K) | 53/70 (1× fp8) · 78/127 (2× DFlash text-only) |
| Vision (images in prompts) | [`yml`](vllm/compose/docker-compose.yml) (default) | [`dual.yml`](vllm/compose/docker-compose.dual.yml) | 51/68 (1×) · 69/89 (2×) |
| Frontier 192K (vision) | [`long-vision.yml`](vllm/compose/docker-compose.long-vision.yml) | [`dual.yml`](vllm/compose/docker-compose.dual.yml) (262K native) | 51/68 (1×) · 69/89 (2×) |
| Frontier 205K (text-only) | [`long-text.yml`](vllm/compose/docker-compose.long-text.yml) | [`dual.yml`](vllm/compose/docker-compose.dual.yml) | 50/66 (1×) · 69/89 (2×) |
| **Multi-tenant** (2-4 concurrent users) | n/a — single-card serializes | [`dual.yml`](vllm/compose/docker-compose.dual.yml) (2 streams) or [`dual-turbo.yml`](vllm/compose/docker-compose.dual-turbo.yml) (4 streams, 54/73 per-stream) | aggregate ~180-220 TPS |
| **Max context** (262K with vision on 1 card) | [llama-cpp recipe](llama-cpp/recipes/) | [`dual.yml`](vllm/compose/docker-compose.dual.yml) | 35-45 (1×) · 71/89 (2×) |

---

## General chat / Q&A

**Single-user, short to medium-length conversations. Browser UIs (Open WebUI, LibreChat). Intermittent use.**

### 1× 3090
- **`fast-chat.yml`** — 20K + fp8 KV — ~5-7% faster than the default; fp8 sidesteps the cudagraph bug entirely.
- Pick the default 48K if conversations might grow long (history + tool defs eat 20K faster than expected).

### 2× 3090
- **`dual.yml`** — 262K + fp8 + MTP. Single-stream is virtually identical to fast-chat 1× (~71 vs 55 narr) once you account for PCIe allreduce overhead.
- Honestly: if you only have one user at a time, the dual stack's win is small. The dual stack shines on **concurrent** users, not solo chat.

**Gotcha:** 20K fills up faster than you'd think with system prompts + tool definitions + history. A 30-turn coding conversation can exceed 20K. Switch to default (48K) or dual (262K) if users let chats grow long.

---

## Tool-using agents (Cline, Roo, Cursor, Hermes, OpenAI Assistants)

**AI coding/research assistants that call tools (read_file, run_command, web_fetch). Multi-turn conversations with structured tool returns.**

### 1× 3090
- **Default `docker-compose.yml`** (48K + TQ3 + Genesis P65). Below both prefill cliffs; `verify-stress.sh` (tool prefill OOM check) passes at 25-40K tool responses.
- **Critical:** verify your agent framework truncates tool responses to ≤20K tokens. **The single most common production crash** is a single tool returning 50K+ tokens. Most frameworks (Cline, OpenAI Assistants, LangChain) do this by default — confirm yours is on.

### 2× 3090
- **`dual.yml`** (default fp8, 262K). TP=2 splits activation memory across cards — neither cliff is an active failure mode here.
- For 4-stream concurrency at full ctx (multiple agents simultaneously), upgrade to **`dual-turbo.yml`**. ~25% per-stream TPS regression but unlocks concurrent serving.

**Gotcha (1×):** tool calls need Genesis v7.14. Default compose loads it; verify via `Genesis Results: 27 applied` in container logs. Without it, the model emits `<tool_call>` as plain text instead of populating `tool_calls[]` cleanly (the silent cascade bug).

**Gotcha (2×):** dual-Turbo requires Genesis P65 to apply correctly; same verification step.

---

## Coding (single-file or repo-aware)

**Code generation, code review, refactoring, debugging. With or without tool access.**

### 1× 3090
- **Default 48K**. Code TPS at 67-68 is the second-fastest config (only fast-chat at 70 beats it). Handles ~3-5 source files at typical sizes plus reasoning history.
- For larger repos: opt into 96K-128K (edit `max-model-len`). Be aware of the prefill cliff at 50-60K single-prompt tokens.

### 2× 3090
- **Default `dual.yml`** for breadth — 71 narr / 89 code TPS, 262K ctx.
- **`dual-dflash.yml`** for **peak code TPS** — 78 / 128 narr/code TPS via Luce z-lab's DFlash N=5 draft model. Code AL ~4.7 vs MTP's 3.4. ~44% faster code generation.
- Note DFlash trade-offs: 185K ctx (vs 262K), single-stream only, vision retained.

**Gotcha:** Code TPS > narrative TPS on this model — the MTP/DFlash draft has higher acceptance on structured tokens. Don't be confused by the asymmetry.

---

## Long single prompts (RAG, document summarization)

**Loading a long document or document-set in one shot, asking questions about it. Single-shot summarization. Cold-prefill cost is the dominant factor.**

### 1× 3090
- **Best path: `tools-text.yml`** (75K + fp8 + no vision). fp8 KV avoids the GDN cliff at 50-60K-token single prompts. Tested up to 60K-token single-prompt depth.
- **Alternative: llama.cpp + Q4_K_M + q4_0 KV at 262K** (see [llama-cpp recipes](llama-cpp/recipes/)). 262K is the model's natural max — only achievable on single-card via llama.cpp. ~35-45 TPS sustained.
- vLLM single-card maxes at ~50K safe single prompts (Cliff 2 hardware-bound).

### 2× 3090
- **`dual-dflash-noviz.yml`** (200K + DFlash + text-only). Longest context any dual variant offers; fastest TPS on text-heavy work.
- **`dual.yml`** (262K + vision + MTP) for mixed-modal docs (PDFs with charts, etc.).

**Gotcha (universal):** **Cold prefill at 200K-262K is GENUINELY slow** — 3-5 minutes for a fresh 150K-token doc on dual-card; longer on single-card llama.cpp. Use prefix caching aggressively. Pre-warm during off-peak.

**Gotcha (universal):** **Recall quality at 100K+ tokens degrades** — like most current LLMs, attention thins toward the document middle. Test recall on YOUR actual corpus before trusting.

---

## Vision-heavy workloads

**Multimodal pipelines where users frequently send images alongside text. Code-screenshot review, document OCR-style tasks, visual Q&A.**

### 1× 3090
- **Default 48K** with vision tower active. Tower is small (~1 GB VRAM), comfortable headroom.

### 2× 3090
- **`dual.yml`** (default — vision + 262K + MTP).
- DFlash variants have vision but the draft's code-bias hurts vision-task quality. Default is the right pick for vision.

**Gotchas (universal):**
- Each image consumes 640-1280 tokens at default resolution. 5-image conversations chew through several thousand tokens before any text gets processed.
- Vision quality is good for charts, screenshots, natural images. Less reliable for OCR on dense text — Qwen team didn't optimize for OCR specifically.
- High-resolution images (2048×2048+) get downsampled internally.

**No image generation** — this model is vision-input-only. For "draw me a picture" use a separate diffusion model (ComfyUI / SD).

---

## Frontier context (192K-262K)

**Whole-codebase agents. Long research papers in one shot. Multi-step reasoning over very large inputs.**

### 1× 3090

Two dedicated composes for the frontier-context tier (R3' / R3''' from the v714 formal bench round):

- **[`long-vision.yml`](vllm/compose/docker-compose.long-vision.yml)** — 192K context with vision tower active. 51/68 TPS narr/code. Pick when you need image input + long ctx.
- **[`long-text.yml`](vllm/compose/docker-compose.long-text.yml)** — 205K context, vision dropped (engine ceiling for single-card TQ3). 50/66 TPS narr/code.
- **OR llama.cpp at 262K** ([recipe](llama-cpp/recipes/)) — actually achieves the model's natural max on a single card. Trade-off: slower TPS, no MTP spec-decode.

**Read [INTERNALS.md](INTERNALS.md) Activation-memory caveat first** — both long-* composes have prefill caveats:
- ≥25K-token tool prefills will OOM (Cliff 1) — same root cause as ampersandru's #1
- ≥50-60K single prompts trigger DeltaNet GDN forward OOM (Cliff 2 — hardware-bound on 24 GB)
- The full 192K/205K is for *steady-state context accumulation* across many small turns, NOT for stuffing 192K of fresh tokens in one request

If your workload includes either pattern, use the default 48K instead.

### 2× 3090

- **`dual.yml`** (262K native — no opt-ins needed). TP=2 splits activation memory, so cliffs are not active failure modes here. Full context with vision + tools + MTP.

**Gotcha (universal):** Recall degrades past 100K. Test on your actual corpus.

---

## Multi-tenant serving (dual-card only)

**2-4 concurrent users / agents on the same backend. Production-like serving on dev hardware.**

### 1× 3090
- Don't. Single-card serializes; concurrent requests wait or both run slow.

### 2× 3090
- **`dual.yml`** (default fp8) — 2 streams at full 262K ctx. Aggregate ~180 TPS at 2 concurrent.
- **`dual-turbo.yml`** (TQ3 + Genesis v7.14) — **4 streams at full 262K ctx**. ~25% per-stream TPS regression but aggregate ~257 TPS at 4 concurrent (~5× single-card cap).
- Crossover where Turbo wins on aggregate is ~3 concurrent streams.

**Gotcha:** Turbo loads Genesis v7.14 patches at boot. Verify via container logs. We adjusted `gpu-memory-utilization=0.85` and `max-num-batched-tokens=4128` from Sandermage's A5000 reference defaults — don't push these without testing.

---

## Advanced mode / contributing

For folks doing spec-decode bench experiments, validating Sandermage's PRs, or contributing back to upstream / Genesis:

- Read [INTERNALS.md](INTERNALS.md) for what each Genesis patch does
- Use `bench.sh` with `RUNS=10 WARMUPS=3` for tight CV measurements
- Capture VRAM, GPU util, and AL alongside TPS in your reports
- Cross-link to Sandermage's MODELS.md for A5000-class baselines

**Things to try:**
- Different DFlash N values (N=3, N=5, N=7) on dual-card
- Genesis P67 / P78 enabled (env-gated)
- TurboQuant variants: `turboquant_4bit_nc` vs `turboquant_3bit_nc` — context vs activation pressure trade-off

**Things that won't work on this model:**
- GGUF on vLLM — Qwen3-Next family unsupported upstream
- TP=4 — only 2 cards
- EAGLE on hybrid attention — DeltaNet rollback issue (cross-engine architectural)

---

## When something is wrong

- **Container dies at boot with `GPTQ_MARLIN_MIN_THREAD_N (64) > out_features`** — dual-card vllm#40361 patch didn't apply. Check `/opt/ai/vllm-src/` exists.
- **Container dies during DFlash boot** — vllm#40334 dtype mismatch. Verify compose has `--dtype bfloat16`.
- **Tool calls return `<tool_call>` as plain text** — Genesis v7.14 didn't apply. Check `Genesis Results: 27 applied` in logs.
- **OOM during prefill at 60K+** — single-card Cliff 2 (GDN forward). Switch to lower max-model-len OR dual-card OR llama.cpp + q4_0 KV.
- **OOM during prefill at 25K+ tool response** — single-card Cliff 1. Lower mem-util OR use default 48K.
- **Per-stream TPS lower than expected** — re-run `bench.sh` with 3+ warmups + 5 measured runs first. Run-to-run variance is 5%.

If none match, open an issue with `docker logs <container> 2>&1 | tail -200` + `nvidia-smi`.

---

## See also

- [Model README](README.md) — overview, recommended path, quick start
- [Model INTERNALS](INTERNALS.md) — engineering deep dive (forensics, Marlin patch, DFlash, upstream tracker)
- [Cross-engine docs](../../docs/engines/) — vLLM / llama.cpp / SGLang comparison
- [Hardware notes](../../docs/HARDWARE.md) — Ampere, NVLink, power
