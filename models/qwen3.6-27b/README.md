# Qwen3.6-27B

**Run [Qwen3.6-27B](https://huggingface.co/Qwen) — with vision and tool calling — on 1 or 2 RTX 3090s.** Full OpenAI-compatible API, drop-in replacement for ChatGPT/Claude in any tool that uses the OpenAI SDK.

---

## TL;DR

- **27B parameter model** with vision support, on consumer hardware
- Quant: [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — INT4 weights with BF16 `mtp.fc` head preserved (lets vLLM use MTP spec-decode)
- 1× 3090: 48K-262K context (depending on engine + config), 51-70 TPS (vLLM)
- 2× 3090: full 262K + 4-stream concurrency, 71/89 TPS single-stream (vLLM)
- Engines supported: **vLLM** (full features) · **llama.cpp** (max context, lighter footprint) · SGLang (currently blocked, watch list)

---

## Quick start (vLLM, single card — most common)

```bash
# from repo root
bash scripts/setup.sh qwen3.6-27b
cd models/qwen3.6-27b/vllm/compose && docker compose up -d
docker logs -f vllm-qwen36-27b
# wait for "Application startup complete"

curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":30}'
```

For dual-card: `docker compose -f docker-compose.dual.yml up -d` from the same directory.

For llama.cpp: `cd ../llama-cpp && cat README.md` (different setup; useful for max-context single-card).

---

## Recommended config — pick by workload

| Workload | 1× 3090 | 2× 3090 | Notes |
|---|---|---|---|
| **General chat ≤20K** | `fast-chat.yml` | `dual.yml` | Single-card 55/70 TPS; dual single-stream similar |
| **Tool-using agents** | `yml` (default) | `dual.yml` | 48K is prefill-safe under realistic tool flows |
| **Coding** | `yml` (default) | `dual-dflash.yml` | DFlash adds 44% code TPS on dual |
| **Long docs / RAG** | `tools-text.yml` (75K) or [llama.cpp 262K](llama-cpp/recipes/) | `dual-dflash-noviz.yml` (200K) | fp8 KV avoids the GDN cliff |
| **Vision-heavy** | `yml` (default) | `dual.yml` | Vision tower on, fp8 KV |
| **Multi-tenant** | n/a | `dual.yml` (2 streams) or `dual-turbo.yml` (4 streams) | Single-card serializes |
| **Frontier 192K (vision)** | `long-vision.yml` (read caveat) | `dual.yml` (262K native) | Dual unlocks safely |
| **Frontier 205K (text-only)** | `long-text.yml` or [llama.cpp 262K](llama-cpp/recipes/) | `dual.yml` | Single-card engine ceiling |

See [USE_CASES.md](USE_CASES.md) for per-workload recommended config + gotchas + tuning levers.

---

## Compose variants (vLLM)

All under [`vllm/compose/`](vllm/compose/):

### Single-card

| File | Context | KV | Spec-decode | Vision | Tools | Narr/Code TPS | Best for |
|---|---|---|---|---|---|---|---|
| **`docker-compose.yml`** ⭐ | 48K | TQ3 | MTP n=3 | ✅ | ✅ | 51/68 | **Default for ≥20K + tool-using agents** |
| `docker-compose.long-vision.yml` | **192K** | TQ3 | MTP n=3 | ✅ | ✅ | 51/68 | Long ctx + vision (read prefill caveat — single-prompt unsafe ≥16K tool / ≥50K text) |
| `docker-compose.long-text.yml` | **205K** | TQ3 | MTP n=3 | ❌ | ✅ | 50/66 | Engine-ceiling text-only; same caveats as long-vision |
| `docker-compose.fast-chat.yml` | 20K | fp8 | MTP n=3 | ✅ | ✅ | 55/70 | Chat-only, max TPS |
| `docker-compose.tools-text.yml` | 75K | fp8 | MTP n=3 | ❌ | ✅ | 53/70 | Long single prompts (fp8 KV avoids GDN cliff up to ~60K) |
| `docker-compose.no-genesis-mtp.yml` | 20K | fp8 | MTP n=3 | ✅ | ✅ | 55/68 | Same as fast-chat minus Genesis (control) |
| `docker-compose.minimal.yml` | 32K | fp8 | none | ✅ | ✅ | 32/33 | Simplest stack, no patches |

### Dual-card

| File | Context | KV | Spec-decode | Vision | Streams | Narr/Code TPS | Best for |
|---|---|---|---|---|---|---|---|
| **`docker-compose.dual.yml`** ⭐ | 262K | fp8 | MTP n=3 | ✅ | 2 | **69/89** | **Default dual** |
| `docker-compose.dual-turbo.yml` | 262K | TQ3 + Genesis v7.14 | MTP n=3 | ✅ | **4** | **54/73** | Multi-tenant (4 concurrent) |
| `docker-compose.dual-dflash.yml` | 185K | FP16 | DFlash N=5 | ✅ | 1 | **82/125** | Peak code TPS |
| `docker-compose.dual-dflash-noviz.yml` | 200K | FP16 | DFlash N=5 | ❌ | 1 | **78/127** | Long-doc text-only, peak speed |

---

## What's working

- **Vision** — images in messages via OpenAI-compat format
- **Tool calling** — `tools=[...]` + `tool_choice="auto"`, parsed cleanly into `tool_calls[]` (with Genesis v7.14)
- **Streaming** — SSE chunks add up to coherent text; tool-call deltas stream too
- **Reasoning mode** — `chat_template_kwargs.enable_thinking=true` for chain-of-thought
- **Spec-decode** — MTP n=3 default (~83% per-position-1 accept on code); DFlash N=5 on dual-card for code-heavy
- **All standard sampling** — temperature, top_p, top_k, repetition_penalty, JSON-mode, structured output

## What's not working today

- **GGUF on vLLM** for Qwen3-Next family — not supported upstream. Use llama.cpp for GGUF on this model.
- **EAGLE spec-decode on hybrid attention** — DeltaNet rollback issue (cross-engine architectural). Watch upstream.
- **Single-card 192K with big tool prefills** — Cliff 1 OOMs above ~16K tool-response prefill at 0.98 mem-util. Use dual-card or default 48K.
- **Single-card single prompts >50-60K** — Cliff 2 (DeltaNet GDN forward state). Use dual-card or llama.cpp.

---

## See also

- [INTERNALS.md](INTERNALS.md) — engineering deep dive (Genesis patches, forensics, Marlin pad, DFlash, upstream tracker)
- [USE_CASES.md](USE_CASES.md) — per-workload guide with gotchas and tuning levers
- [CHANGELOG.md](CHANGELOG.md) — dated history (combines single + dual timelines)
- [vllm/](vllm/) — vLLM-specific recipes for this model
- [llama-cpp/](llama-cpp/) — llama.cpp recipes (max context on single card)
- [sglang/](sglang/) — SGLang status (currently blocked)
- [/docs/engines/](../../docs/engines/) — cross-model engine comparison + per-engine deep dives
- [/docs/HARDWARE.md](../../docs/HARDWARE.md) — hardware notes (Ampere, NVLink, power)
