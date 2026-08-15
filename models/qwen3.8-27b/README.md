# Qwen3.8-27B on llama.cpp — single RTX 3090

Configs for running **Qwen 3.8 27B** (dense, hybrid GDN+FA, native vision + MTP) on a single RTX 3090 (24 GB VRAM) with llama.cpp. Validated on both native Windows and WSL2.

> Single-rig data: RTX 3090 (sm_86, 230W cap), Ryzen 5 5600G, 32 GB DDR4, Windows 10, llama.cpp build **b10435** (CUDA 13). Numbers are streaming decode TPS with vision loaded.

---

## Model files

- **Weights:** [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) → `Qwen3.8-27B-UD-Q4_K_XL.gguf` (16.69 GiB, dynamic Q4_K_XL quant)
- **Vision:** `mmproj-Qwen3.8-27B-Q8_0.gguf` (~0.9 GB)
- **MTP head:** Embedded in GGUF (`blk.*.nextn.*`) — no external drafter needed

Architecture: 27B dense, Qwen3-Next hybrid (GDN + full attention layers), native 262K context, trained MTP head, vision (image + video), tool calling.

---

## Profiles

### `mtp-170k` — speed-first daily (MTP spec-decode + vision)

Best for: multi-turn agentic with tool calls (Hermes Agent, OpenCode, Cline). Speed-first.

```bash
llama-server.exe ^
  --model Qwen3.8-27B-UD-Q4_K_XL.gguf ^
  --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf ^
  --alias qwen3.8-27b-mtp-q4xl-170k ^
  --host 0.0.0.0 --port 18080 ^
  --ctx-size 174080 ^
  --gpu-layers 99 ^
  --flash-attn on ^
  --cache-type-k q4_0 --cache-type-v q4_0 ^
  --batch-size 2048 --ubatch-size 512 ^
  --mlock ^
  --parallel 1 ^
  --spec-type draft-mtp --spec-draft-n-max 2 ^
  --reasoning off ^
  --temp 0.7 --top-p 0.8 --top-k 20 ^
  --presence-penalty 1.5 --repeat-penalty 1.0 ^
  --timeout 3600 --metrics --slots --no-ui
```

**Measured (RTX 3090, streaming):**
- ~40–44 t/s decode @ 170K context + MTP n=2 + vision
- VRAM: ~24 GB (170K + MTP is the ceiling on 24 GB)
- MTP @ 131K: ~44–46 t/s (~23.3 GB, more headroom)
- MTP @ 262K: does not fit on 24 GB

### `max-262k` — max context (no MTP)

Best for: long single-shot RAG, codebase analysis, document Q&A.

```bash
llama-server.exe ^
  --model Qwen3.8-27B-UD-Q4_K_XL.gguf ^
  --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf ^
  --alias qwen3.8-27b-ud-q4xl-262k ^
  --host 0.0.0.0 --port 18080 ^
  --ctx-size 262144 ^
  --gpu-layers 99 ^
  --flash-attn on ^
  --cache-type-k q4_0 --cache-type-v q4_0 ^
  --batch-size 512 --ubatch-size 128 ^
  --mlock ^
  --parallel 1 ^
  --reasoning off ^
  --temp 0.7 --top-p 0.8 --top-k 20 ^
  --presence-penalty 1.5 --repeat-penalty 1.0 ^
  --timeout 3600 --metrics --slots --no-ui
```

**Measured:**
- ~30–33 t/s decode @ 262K context + vision
- VRAM: ~24 GB (fills cleanly with small margin)
- First request after cold start is slow (disk page-in); warm requests hit full speed

### `think-262k` — reasoning mode

Best for: complex analysis, architecture decisions, multi-step problem solving. Slower but deeper.

```bash
llama-server.exe ^
  --model Qwen3.8-27B-UD-Q4_K_XL.gguf ^
  --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf ^
  --alias qwen3.8-27b-think-ud-q4xl-262k ^
  --host 0.0.0.0 --port 18080 ^
  --ctx-size 262144 ^
  --gpu-layers 99 ^
  --flash-attn on ^
  --cache-type-k q4_0 --cache-type-v q4_0 ^
  --batch-size 256 --ubatch-size 64 ^
  --mlock ^
  --parallel 1 ^
  --reasoning on --reasoning-format deepseek --reasoning-budget 4096 ^
  --temp 1.0 --top-p 0.95 --top-k 20 ^
  --presence-penalty 0.0 --repeat-penalty 1.0 ^
  --timeout 3600 --metrics --slots --no-ui
```

**Notes:**
- Reasoning routed to `message.reasoning_content` (OpenAI-compatible)
- Per-request override: `chat_template_kwargs: {"enable_thinking": false}` for direct answers
- Estimated ~20–25 t/s (reasoning budget consumes throughput; not separately benched)

### `balanced-128k` — general daily

Best for: general use where 262K is overkill but you want headroom.

```bash
llama-server.exe ^
  --model Qwen3.8-27B-UD-Q4_K_XL.gguf ^
  --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf ^
  --alias qwen3.8-27b-ud-q4xl-128k ^
  --host 0.0.0.0 --port 18080 ^
  --ctx-size 131072 ^
  --gpu-layers 99 ^
  --flash-attn on ^
  --cache-type-k q4_0 --cache-type-v q4_0 ^
  --batch-size 512 --ubatch-size 128 ^
  --mlock ^
  --parallel 1 ^
  --reasoning off ^
  --temp 0.7 --top-p 0.8 --top-k 20 ^
  --presence-penalty 1.5 --repeat-penalty 1.0 ^
  --timeout 3600 --metrics --slots --no-ui
```

---

## Sampling parameters (official)

| Mode | temp | top_p | top_k | presence_penalty |
|------|------|-------|-------|-----------------|
| Non-thinking (daily) | 0.7 | 0.8 | 20 | 1.5 |
| Thinking (reasoning) | 1.0 | 0.95 | 20 | 0.0 |

---

## VRAM budget breakdown (single 3090, 24 GB)

| Component | mtp-170k | max-262k |
|-----------|---------:|---------:|
| Model weights (UD-Q4_K_XL) | ~16.7 GB | ~16.7 GB |
| mmproj Q8_0 (vision tower) | ~0.9 GB | ~0.9 GB |
| KV cache (q4_0, 170K / 262K) | ~3.5 GB | ~5.5 GB |
| MTP draft KV (f16, mandatory) | ~1.5 GB | — |
| Activations + cudagraph | ~1.0 GB | ~0.9 GB |
| **Total** | **~23.6 GB** | **~24.0 GB** |

> MTP draft KV must stay f16. `--spec-draft-type-k/v q4_0` hangs every request (no progress past prefill). Known llama.cpp limitation, not a config error.

---

## Gotchas

1. **`--mlock` matters on Windows.** Without it, the OS can evict model pages under RAM pressure; decode drops from ~40 t/s to ~10–15 t/s after sustained mixed usage. With `--mlock`, pages are pinned and performance is stable. (Less critical on Linux/WSL2 where the page cache behaves differently.)

2. **`--parallel 1`.** The default (`auto`) resolves to 4 concurrent slots, multiplying KV cache by 4. On 24 GB at high context, only single-slot works. Concurrent requests queue (FIFO) — invisible for single-user/agent use.

3. **`--gpu-layers 99` (hardcoded).** With `auto` + `--fit on`, the fit logic logs a confusing "failed to fit params" message and skips its calculation. Hardcoding 99 avoids the ambiguity. All layers fit in 24 GB at Q4.

4. **ubatch sizing is the VRAM cliff lever.** At 170K + MTP, the default `ubatch 1024` saturates VRAM (24.07 GB) and decode collapses to ~10 t/s. Use `512`. At 262K without MTP, use `128`. Same mechanism as the vLLM "Cliff 2" in [`docs/CLIFFS.md`](../../docs/CLIFFS.md) — per-pass activation peak — but tunable via `-ub` in llama.cpp.

5. **Cold start is slow.** First request after boot pages the model in from disk (~5–15 s on NVMe for a 17 GB file). Not a bug; don't benchmark on the first token.

6. **`--image-min-tokens` default (1024) reserves VRAM for vision even when no images are sent.** At 170K+ where you're already near the ceiling, consider `--image-min-tokens 512`, or omit `--mmproj` entirely if you don't need vision.

---

## Comparison with Qwen3.6-27B (same hardware)

| Metric | Qwen3.6-27B (club-3090 baseline) | Qwen3.8-27B (this page) |
|--------|----------------------------------|------------------------|
| Quant | Q4_K_M / IQ4_KS | UD-Q4_K_XL (dynamic) |
| Max ctx (single 3090) | 200K (llamacpp/mtp, retired) | **262K** (native max) |
| MTP speed @ ~170K | ~51/60 TPS (narr/code) | ~40–44 t/s (MTP n=2 + vision) |
| Max ctx speed (no MTP) | ~33/40 TPS @ 200K | ~30–33 t/s @ 262K |
| Vision | Separate config (160K, slower) | Integrated at full ctx |

The speed delta vs. Qwen3.6 is expected: UD-Q4_K_XL is a slightly larger quant than Q4_K_M, and the 3.8 architecture has more GDN layers. The context gain (+62K) and integrated vision are the trade-off.

---

## Operational notes

- **API:** OpenAI-compatible at `http://localhost:18080/v1`. No API key required (open port).
- **Metrics:** `--metrics` exposes a Prometheus-style `/metrics` endpoint.
- **Health check:** `curl http://localhost:18080/v1/models` should return the alias within ~30 s of boot.
- **Recommended client config for agents:**
  ```
  base_url: http://127.0.0.1:18080/v1
  model: qwen3.8-27b-mtp-q4xl-170k
  max_tokens: 2048–4096 (not unlimited)
  stream: true
  ```

---

## Build info

- **llama.cpp:** build b10435, CUDA 13, Windows x64
- **GPU:** NVIDIA GeForce RTX 3090, 24576 MiB, driver 610.88
- **OS:** Windows 10 (native; WSL2 available but not used for inference)
- **RAM:** 32 GB DDR4 (model pages pinned via `--mlock`)
