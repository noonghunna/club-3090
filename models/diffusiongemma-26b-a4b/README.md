# DiffusionGemma-26B-A4B (experimental)

Google's **DiffusionGemma** — a *block-diffusion* language model (non-autoregressive) on a
Gemma-4 26B-A4B MoE backbone (128+1 experts, ~3.8B active, 256K train ctx, canvas 256). vLLM's
first dLLM (blog 2026-06-10). It denoises a whole 256-token "canvas" in parallel via bidirectional
attention, with an entropy-bound sampler — so throughput is content-dependent, not a flat tok/s.

It's a true diffusion LM, so neither stock `llama-server` nor `vllm serve` runs it out of the box;
both routes below depend on unmerged upstream work (tracked in [`docs/UPSTREAM.md`](../../docs/UPSTREAM.md)).

## Two routes (pick by card count)

| | **vLLM / fp8 — dual card** | **llama.cpp / GGUF — single card** |
|---|---|---|
| Path | `vllm/` · slug `vllm/diffusiongemma-dual` | `llama-cpp/` · `compose/single/q4-k-m/openai-shim.yml` |
| Topology | 2× 3090 (TP=2) | **1× 3090** |
| Weights | RedHatAI FP8-dynamic (~25 GB) | unsloth Q4_K_M GGUF (~16 GB) |
| Throughput | ~150–200 tok/s typical (up to ~1100 low-entropy) | ~75 tok/s effective (thinking-off coding) |
| Max ctx | 262K (NIAH ~250K) | ~2800 tok (DG_CTX=3072 ubatch ceiling) |
| Tool calls | yes (gemma4 parsers) | content only (v1) |
| Upstream dep | vLLM [#45163](https://github.com/vllm-project/vllm/pull/45163) + `vllm/vllm-openai:gemma` image + 3 vendored Ampere/TP fixes | llama.cpp draft [#24423](https://github.com/ggml-org/llama.cpp/pull/24423) + our `DG_NDJSON` patch |
| Status | 🧪 Experimental (launch `--force`) | ⏸️ Upstream-gated |

**Why two:** the fp8 weights that let vLLM fit need ~25 GB + fp8 Marlin — that's a **2-card** path
on Ampere (the team's `vllm/` route, with sub-tile-K Marlin + TP-vocab fixes; see
[`vllm/patches/gemma-image-fixes/`](vllm/patches/gemma-image-fixes/README.md)). On a **single
3090**, fp8 won't fit and the fitting paths (FP8/NVFP4) need Hopper/Blackwell — so the only
single-card option is **GGUF-Q4 via the llama.cpp draft CLI**, documented below. If you have two
cards, prefer the vLLM route: it's faster and goes to 262K context.

---

# llama.cpp / GGUF — single-3090 route

The draft PR ships `llama-diffusion-cli` (interactive) + a logits microservice but **no HTTP
server**. We add a small `DG_NDJSON` server mode to the CLI and wrap it in a FastAPI shim:

```
coding harness ──/v1──▶ serve/diffusion_openai_server.py ──NDJSON──▶ patched llama-diffusion-cli
                         (FastAPI, channel parsing,                    (DG_NDJSON=1, model resident,
                          413 on overflow, np=1)                        entropy-bound denoise, KV-cache on)
```

- `llama-cpp/Dockerfile` — **primary delivery**: builds the patched CLI from the draft-PR branch
  and bundles the shim into a custom image (no stock image carries `llama-diffusion-cli`).
- `llama-cpp/patches/diffusion-ndjson-server.patch` — the env-gated `DG_NDJSON` mode.
- `llama-cpp/serve/` — the OpenAI shim (`diffusion_openai_server.py`, `serve.sh`).
- `llama-cpp/compose/single/q4-k-m/openai-shim.yml` — builds + runs the image.

## Run it (single 3090)

**Docker (primary)** — builds from the existing PR branch, bakes in our patch:
```bash
cd models/diffusiongemma-26b-a4b/llama-cpp/compose/single/q4-k-m
docker compose -f openai-shim.yml build      # ~2 GB CUDA build of PR #24423 @ c84e85a
docker compose -f openai-shim.yml up -d
curl http://localhost:8060/v1/models
```

**Host (no-Docker fallback — validated on the authoring rig)**:
```bash
# build the CLI once (see llama-cpp/patches/README.md), then:
DG_CLI_BIN=/path/to/llama-diffusion-cli \
  bash models/diffusiongemma-26b-a4b/llama-cpp/serve/serve.sh
```

Either way, point a coding harness at `http://localhost:8060/v1` (model id `diffusiongemma-26b-a4b`):
```bash
curl http://localhost:8060/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python quicksort. Code only."}],"max_tokens":512}'
```

## Measured (1× 3090, Q4_K_M, `-ngl 99`, host path, 2026-06-11)

| Metric | Value | Notes |
|---|---|---|
| Builds + runs on Ampere | ✅ | draft PR #24423 `llama-diffusion-cli`, SM 86 |
| Effective throughput (thinking **off**, coding) | **~75 tok/s** (5–6 steps, ~1–2 s / 256-tok block) | step count scales with content + block count |
| Effective throughput (thinking **on**) | ~40–61 tok/s (30–98 steps) | thinking inflates denoising steps |
| Peak VRAM | **22.3–23.0 GB / 24** | tight; grows with prompt length |
| Prompt ceiling @ `DG_CTX=3072` | ~2800 tokens | over that → HTTP 413 (prompt+canvas must fit one ubatch) |

**Speed verdict:** on a single 3090, diffusion is **not** a speed win — ~75 tok/s effective is in
the same ballpark as (often below) our autoregressive MoEs, and VRAM is on a knife's edge. The
headline ~700–1100 tok/s figures are FP8 on Hopper/Blackwell (or the dual-card vLLM route). The
value on one card is capability/experimentation, not throughput.

## Known limitations (single-card route)

- **Draft-PR dependency** — unmerged, AI-assisted (logit-verified). May change/break on rebase;
  the image pins the branch sha.
- **VRAM ceiling** — raising `DG_CTX` for longer prompts risks OOM near 24 GB. Watch `nvidia-smi`.
- **Trivial prompts → empty** — ultra-short conversational prompts ("say ready in one word") can
  return empty content across seeds (entropy-bound sampler collapses the canvas). Real coding
  prompts are unaffected.
- **v1 cutouts** — no native tool-call JSON (assistant content only; the GGUF template *does*
  support tools, parse-able later); "streaming" emits the final answer as one SSE chunk
  (diffusion is block-wise, not token-by-token).
- **Not in the curated catalog** — no `compose_registry.py` entry / DEFAULTS promotion; can't pass
  `verify-full`/`bench` as a normal OpenAI server until upstream ships one.

Design rationale: [`docs/superpowers/specs/2026-06-11-diffusiongemma-serving-design.md`](../../docs/superpowers/specs/2026-06-11-diffusiongemma-serving-design.md).
