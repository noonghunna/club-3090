# Muse Glimmer 30B on llama.cpp — DFlash + 131K (single RTX 3090)

Config for running **Meta's Muse Glimmer 30B** (dense, SWA+GQA, native vision + tool calling) on a single RTX 3090 with DFlash speculative decoding.

> Single-rig data: RTX 3090 (sm_86, 24 GB), WSL2 Ubuntu (llama.cpp build 10349, commit 62bf73d25). Also runs on native Windows with equivalent performance. Measured 2026-08-11.

---

## What this is

- **30B parameter dense LLM** from Meta Superintelligence Lab (Apache 2.0)
- Architecture: 2B ViT perceiver + 28B decoder, GQA 32Q/2KV heads, Sliding Window Attention (2048), native 131K context
- **DFlash spec-decode:** separate drafter model (`dflash-kquant.gguf`), `--spec-type draft-dflash --spec-draft-n-max 15`
- **Native tool calling:** no parser/proxy needed. `tools=[...]` → `finish_reason=tool_calls` with parsed JSON args
- **Vision:** `--mmproj` fits alongside DFlash (22.6 GB total)

---

## Profile: `dflash-131k`

```bash
llama-server \
  -m muse-glimmer-30B-kquant-17gb.gguf \
  -md dflash-kquant.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 15 \
  --alias muse-glimmer \
  --port 18080 --host 127.0.0.1 \
  -ngl 99 \
  -fa on \
  --jinja \
  -c 131072 \
  -np 1 \
  --metrics \
  --no-webui \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 64 \
  --reasoning-budget 2048 \
  --chat-template-kwargs '{"reasoning_strength":"high"}'
```

### Measured performance (RTX 3090, verified 2026-08-11)

| Metric | Value | Notes |
|--------|-------|-------|
| VRAM (with DFlash) | 20,886 MiB / 24,576 | Stable at full 131K context |
| VRAM (without DFlash) | 18,882 MiB | DFlash costs ~2 GB |
| Prefill @ 80–100K ctx | ~860 tok/s | |
| Prefill @ 114K fresh | ~679 tok/s | KV cache reuse makes follow-ups near-instant |
| Decode (DFlash on, long gen) | **39.4 t/s** @ 131K ctx | |
| Decode (DFlash on, short) | **44.3 t/s** @ 114K ctx | |
| Decode (DFlash off, long) | 35.5 t/s | DFlash adds +10–25% |
| Draft acceptance | 0.14 (179/1245 accepted) | Mean draft length: 3.16 tokens |
| Post-sweep (with `--no-webui`) | **47.3 t/s long / 67.1 short** | ~20.8 GB VRAM |

### Reasoning behavior

- Default: `reasoning_strength=high`, routed to `message.reasoning_content`
- Per-request control: `"chat_template_kwargs": {"reasoning_strength": "low|medium|high|xhigh"}`
- `low` ≈ 2× faster (99.7 vs 53.3 t/s on coding) without losing correctness
- `--reasoning off` is a no-op for this template (always thinks, just less)
- `--reasoning-budget 2048` caps thinking per step; guarantees content from `max_tokens >= 512`

### Tool calling (verified)

- Native: no `--enable-auto-tool-choice` needed (build doesn't have the flag)
- `tools=[...]`, `tool_choice="auto"` → `finish_reason=tool_calls` with parsed function name + JSON args
- Tool-result injection → final answer: correct
- No DFlash suppression on tool markers (unlike BeeLlama/Qwen3.6 — see below)

### Multi-turn cache

- In-slot KV reuse is automatic
- Turns 2–5 at ~220 ms with growing `cached_tokens` (49 → 124)
- No `--cache-reuse` flag needed

---

## DFlash: when it works and when it doesn't

### Muse Glimmer: DFlash pays off

Dense architecture + SWA = drafter predictions are highly accurate. Acceptance rate 0.14 with mean length 3.16 → net speedup of +10–25% for the cost of ~2 GB VRAM. Keep it on.

### Qwen3.6-27B (BeeLlama): DFlash regresses under agent workloads

On BeeLlama's fork, long agent prompts with tool-style output trigger:
```
raw tool marker observed ... suppressing DFlash
```
This forces full prompt re-processing and drops from ~97 t/s (clean generation) to ~27 t/s (agent loops). Do not use DFlash for agentic workloads on Qwen3.6/BeeLlama.

### vLLM/SGLang: DFlash only works with BF16 weights

DFlash in vLLM/SGLang requires BF16 model weights (~54 GB for 30B). Does not fit on 24 GB. llama.cpp + GGUF is the only DFlash path on consumer hardware.

---

## Gotchas

1. **Never quantize the draft KV.** `--spec-draft-type-k/v q4_0` (or `-ctkd`/`-ctvd`) collapses DFlash acceptance to near-zero. Upstream issue #25725; fix #25823 confirmed present in build 10349. Keep draft KV at f16 (default).

2. **`--jinja` is required.** Muse Glimmer's chat template uses Jinja2 syntax. Without `--jinja`, tool calling and reasoning routing break silently.

3. **`-np 1` at 131K.** KV cache is per-slot. At 131K, one slot fills ~20 GB. Concurrent requests queue (fine for single-user). If you need concurrency, drop context to 64K and use `-np 2`.

4. **WSL2 launch must include trailing `sleep 5`.** The `wsl.exe` client detaches from the process after the command returns. Without the sleep, the server gets killed when the WSL session closes. Pattern:
   ```bash
   setsid "$BIN" ... > "$LOG" 2>&1 < /dev/null &
   echo "started pid=$!"
   sleep 5
   ```

5. **Startup warning is harmless.** `[spec] failed to measure draft model memory` appears during boot. Known timing issue in the memory fitting step. The server boots correctly and DFlash works.

---

## Model files

| File | Size | Source |
|------|------|--------|
| `muse-glimmer-30B-kquant-17gb.gguf` | ~17 GB | Official Meta GGUF quant (kquant) |
| `dflash-kquant.gguf` | ~2 GB | DFlash drafter (kquant) |
| `mmproj-muse-glimmer.gguf` | ~1.8 GB | Vision projector (optional) |

---

## Build info

- **llama.cpp:** build 10349 (commit 62bf73d25), CUDA, WSL2 Ubuntu
- **GPU:** NVIDIA GeForce RTX 3090, 24576 MiB
- **OS:** Windows 10 host + WSL2 (or native Windows with equivalent binary)
