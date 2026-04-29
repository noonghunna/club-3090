# Comparisons — club-3090 vs alternatives

Honest read on when this repo is the right answer and when it isn't. We have a real bias toward this stack because we built it, but we'll try to call the trade-offs straight.

---

## vs Ollama

**Ollama is a great place to start. We don't replace it; we serve a different need.**

| | Ollama | club-3090 |
|---|---|---|
| Setup time | ~2 min (one binary install) | ~5-10 min (Docker pull + model download) |
| Model registry | curated `ollama pull` registry | direct from Hugging Face (Unsloth UD-Q3_K_XL exact build) |
| Engine | llama.cpp under a thin wrapper | vLLM **and** llama.cpp, picked per workload |
| Configurability | limited — can't expose `--cache-type-k q4_0`, `--mmproj`, `--spec-type ngram-mod`, `--parallel`, custom KV quants | full engine flag access via Docker compose |
| Reproducibility | model versions move with Ollama's registry | pinned image SHA + Genesis commit + GGUF SHA |
| GUI / system integration | system tray, desktop app | none — CLI / Docker / OpenAI-compat API only |
| Tool calling on Qwen3.6-27B | works (Ollama parses `<tool_call>` blocks) | works (vLLM `--tool-call-parser qwen3_coder`, llama.cpp `--jinja`) |

**Pick Ollama if:** chat-only use, you want a system-tray GUI, you don't care which exact GGUF quant runs, you're switching between many models often.

**Pick club-3090 if:** you want a tested config that another club-3090 user can reproduce exactly, you need vLLM features (MTP n=3, Marlin pad fork, AutoRound INT4, Genesis patches), or you've measured that Ollama's defaults aren't fast enough on your hardware.

**Honest take:** if you're new to local LLMs and "does this run?" is the question, start with Ollama. Come back here when you hit a configuration limit.

---

## vs LM Studio

**LM Studio is a desktop app with a GUI. Different audience.**

| | LM Studio | club-3090 |
|---|---|---|
| UX | GUI (model selection, chat history, settings panels) | CLI / Docker compose |
| Backend | llama.cpp | vLLM and llama.cpp |
| Headless deployment | not really — desktop app expects a UI | designed for it (homelab racks, dev backends) |
| Bench reproducibility | hard — version + setting state lives in app config | easy — pinned everything |
| Quant control | dropdown of common quants | exact Unsloth UD-Q3_K_XL with q4_0 KV |
| Multi-engine | no | yes |

**Pick LM Studio if:** you want a GUI, you're doing exploratory chat with different models, you don't need pinned reproducibility.

**Pick club-3090 if:** you're embedding the model into a service, you need bench numbers another rig can match, you want vLLM's higher TPS ceiling.

---

## vs raw llama.cpp (build from source)

**Same engine as our llama.cpp compose, just without the Docker wrapper.**

| | Raw llama.cpp build | club-3090 llama.cpp compose |
|---|---|---|
| Engine binary | you build (`cmake -DGGML_CUDA=ON`) | `ghcr.io/ggml-org/llama.cpp:server-cuda` (multi-arch, rebuilds on master push) |
| Toolchain dependency | CUDA toolkit, gcc, cmake, etc. on host | Docker only |
| Version drift | you `git pull` and rebuild whenever | image tag re-pulled when you re-up |
| Model file location | wherever you download | `models-cache/` by default, override via `MODEL_DIR` |
| Flags | full control | full control (compose forwards every llama-server flag) |
| Vision setup | manual mmproj path | mmproj wired into compose |
| Recipe sharing | "here are my exact flags" → comment with config | `bash scripts/switch.sh llamacpp/default` |

**Pick raw llama.cpp if:** you're already a llama.cpp committer, you need a specific PR branch, you don't trust Docker, or your platform isn't supported by the official image.

**Pick our compose if:** you want a one-shot tested config without the build chain. The compose is just a thin layer — every flag is documented in the YAML header, drop the wrapper any time you want.

---

## vs cloud APIs (Together, Fireworks, Anthropic)

**This is the comparison most users actually care about. Self-host vs API call.**

### Pricing landscape (rough, 2026-04 — verify before quoting)

| Service | Model | Input ($/M tok) | Output ($/M tok) | Notes |
|---|---|---|---|---|
| Together AI | Qwen3-Next-class hosted | ~$0.20 | ~$0.60 | exact Qwen3.6-27B availability varies |
| Fireworks AI | Qwen / DeepSeek / similar | ~$0.20 | ~$0.80 | Q-LoRA serving, latency-optimized |
| Anthropic | Claude Haiku 4.5 | ~$1.00 | ~$5.00 | bigger / smarter model, not apples-to-apples |
| Anthropic | Claude Sonnet 4.6 | ~$3.00 | ~$15.00 | reference for "what cloud-frontier costs" |
| **club-3090** (you host) | **Qwen3.6-27B** | **$0** marginal | **$0** marginal | hardware + power amortized |

### Self-host break-even math

A 2× RTX 3090 rig:

```
Hardware:    ~$1,500 / card secondhand × 2 = $3,000
Plus:        case + PSU + motherboard + RAM = ~$1,000
Total cap:   ~$4,000

Power:       ~500 W × 24h × 30 days = 360 kWh / month
             × $0.15 / kWh           = ~$54 / month

5-year amortization (typical depreciation horizon):
             $4,000 / 60 months = $67 / month hardware
             + $54 power
             = ~$120 / month operating cost
```

At ~$0.50 / M tokens blended cloud rate (Qwen-class, mixed input + output), $120 / month is **240 M tokens / month** of equivalent cloud spend.

```
240 M tokens / 30 days = 8 M tokens / day
                       = ~333 K tokens / hour
                       = ~93 tokens / second sustained
```

**Crossover lens: if you're consistently driving ~100 TPS of sustained generation, self-host pays back at month 60. If you're driving 10 TPS, you'd be better off on cloud (~$12 / month vs $120).**

### Self-host wins outside of pure cost

Cost isn't always the question. Reasons people self-host even when cloud is cheaper at their volume:

- **Latency floor.** ~120 ms TTFT (our measured) vs cloud's ~300-800 ms typical (network + queue). Tight for IDE-agent flows.
- **No rate limits.** vLLM dual-card serves 4 concurrent streams indefinitely; cloud APIs throttle at ~10-100 RPM on most plans.
- **Data residency.** Health, legal, financial workloads where the data can't leave your network.
- **Customization.** Run with Genesis patches, your own LoRA, your own KV-quant choice. Cloud APIs don't expose these.
- **Predictable cost.** $120 / month flat vs token-metered surprises during a big batch run.
- **Offline / disconnected.** Demos behind a firewall, edge deployments, dev rigs without reliable internet.
- **Learning lens.** If you want to *understand* how serving works (cliffs, KV math, spec-decode acceptance), the abstraction in API providers hides this.

### Cloud wins outside of pure cost

Be honest about when cloud is the right call:

- **Bursty / low-volume.** If your usage is "a few hundred requests per week", cloud is dramatically cheaper.
- **Multi-region serving.** Self-host doesn't replicate; cloud does.
- **Frontier-quality output.** Claude Sonnet / Opus are smarter than Qwen3.6-27B on complex reasoning. If the work needs the smartest model, cloud is the only path right now.
- **Maintenance budget.** Cloud handles upstream-bug routing, model updates, hardware failures. Self-host needs you (or someone) to read the GitHub issue tracker and roll updates.

### When club-3090 is genuinely better than both

A specific class of workload tilts hardest toward this stack:

- **Heavy agentic IDE flows** (Cursor / Cline / Aider / Continue). High request rate, large tool-call payloads, want sub-second TTFT, OK with Qwen-class quality.
- **Privacy-sensitive analytical workloads** that fit Qwen3.6-27B's capability bar.
- **Long-context experimentation** where the cloud's 128K-256K context tier costs significantly more per million tokens, but you have the hardware to run 262K locally.

For these, the cost math shifts strongly in favor of self-host even at moderate volumes, and the latency / privacy story is independent.

---

## Bottom line

| If you want… | Pick |
|---|---|
| GUI desktop chat, exploratory model browsing | LM Studio |
| Easiest "does this run?" first contact | Ollama |
| Same llama.cpp engine without Docker | raw llama.cpp build |
| Bursty low-volume work, smart-model quality | cloud API |
| Tested OpenAI-compat endpoint with vLLM features, full ctx, no cliffs on tools | **club-3090** |
| Heavy agentic IDE work, sub-second TTFT, no rate limits | **club-3090** |
| Building / learning the serving stack itself | **club-3090** |

---

## See also

- [README](../README.md) — top-level overview
- [docs/FAQ.md](FAQ.md) — common questions
- [docs/EXAMPLES.md](EXAMPLES.md) — client snippets for the OpenAI-compat endpoint
- [docs/HARDWARE.md](HARDWARE.md) — hardware notes
- [models/qwen3.6-27b/USE_CASES.md](../models/qwen3.6-27b/USE_CASES.md) — which compose for which workload
