# Glossary

Plain-language definitions for terms used throughout the docs. Roughly grouped by topic.

> Want the *narrative* version — how hardware / engines / sizes / quants fit together — instead of isolated definitions? See [LOCAL_AI_PRIMER.md](LOCAL_AI_PRIMER.md).

> Coming from a different background and don't see something here? Open an issue and we'll add it.

---

## Universal `pull` (v0.8.0)

| Term | What it means |
|---|---|
| **`pull`** | `scripts/pull.sh <hf-repo> --profile-like <key>` — evaluates *any* safetensors HF repo against this stack's KV math and, if it passes the gates, downloads + generates a minimal compose + boots it. The model-agnostic front door; the curated catalog still works unchanged. See [PULL.md](PULL.md). |
| **dry-run** | `pull … --dry-run` — evaluate only: never downloads, never boots, just prints the verdict. |
| **confidence tier** | How much the fit verdict is trusted: `exact` (a measured/curated profile) vs `estimated-lower-bound` (derived from the repo's own config — a floor, likely under-modeled). Always shown with the verdict. |
| **boot-fit ≠ runtime-stability** | A "fits" verdict is a *boot-time* allocation check. It's necessary-not-sufficient: a config that boots clean can still degrade/OOM under sustained accumulated-context agent workloads (see [CLIFFS.md](CLIFFS.md)). Validate with soak-continuous before relying on it. |
| **calibration backbone** | The curated catalog's role under v0.8.0 — the measured anchor set the KV math is calibrated against (vs. being the only supported models). |

## Throughput / latency

| Term | What it means |
|---|---|
| **TPS** | Tokens per second — how fast the model generates output. ~70 TPS is roughly conversational speed; ChatGPT cloud is ~80-120. |
| **Wall TPS** | `completion_tokens / wall_time` — user-perceived total speed (includes prefill cost). |
| **Decode TPS** | `completion_tokens / (wall_time − TTFT)` — pure model decode rate, excludes prefill. |
| **TTFT** | Time to first token. Dominated by prefill cost on long prompts. |
| **CV** | Coefficient of variation across measured runs. Lower = more predictable. We aim for <5% in benches. |

---

## Memory / context

| Term | What it means |
|---|---|
| **Prefill** | The phase where the model processes the entire input (system prompt + user message + history) before generating the first output token. Slow on first request, fast on follow-ups via prefix cache. |
| **Decode** | The phase after prefill — generating output tokens one at a time. |
| **KV cache** | "Key-value" cache — the model's working memory of the conversation so far. Larger context = bigger KV cache = more VRAM. |
| **Prefix cache** | When two requests share a leading prompt, vLLM (and llama.cpp) serve the second from cache (skip re-prefill). Especially useful for long-document workflows. |
| **Context window** | Total tokens the model can hold in working memory at once. Set via `--max-model-len` (vLLM) or `-c` (llama.cpp). |
| **Activation memory** | Memory used during forward pass (intermediate tensor outputs at each layer). Distinct from KV cache (long-lived) and model weights (fixed). Activation peaks during prefill cause the OOMs we document. |

---

## Quantization

> For the full per-GPU-arch hardware-acceleration matrix — which dtypes / quant schemes run on Tensor Cores natively vs in software emulation — see [DTYPE_MATRIX.md](DTYPE_MATRIX.md).

| Term | What it means |
|---|---|
| **Quantization** | Compressing model weights from 16-bit floats to 4-bit or 8-bit ints. Lets a 27B model fit in 18 GB instead of 54 GB, with small quality loss. |
| **AutoRound** | Intel's 4-bit quantization method using signed gradient descent. Strong on Qwen-family models. |
| **GPTQ** | Layer-wise Hessian-based 4-bit quantization. Mature, broadly supported. |
| **AWQ** | Activation-aware salience-scaled 4-bit quantization. Strong baseline. |
| **GGUF** | Standardized binary format used by llama.cpp / Ollama / LM Studio. Many quant types: Q4_K_M, Q5_K_S, IQ4_XS, etc. |
| **TurboQuant** | A 3-bit KV cache compression scheme used by vLLM. Lets us fit 192K+ context where fp8 KV would only fit ~32K. |
| **fp8 / fp8_e5m2** | An 8-bit float KV cache format. Larger per-token bytes than TurboQuant but dodges several bugs. |

---

## Speculative decoding

| Term | What it means |
|---|---|
| **Spec-decode / speculative decoding** | The model predicts several tokens ahead, then verifies. Roughly 2-3× faster than greedy decoding when accept rate is high. |
| **MTP** | Multi-Token Prediction — built-in spec-decode head that ships with Qwen3.6. We run it with `num_speculative_tokens=3`. |
| **DFlash N=5** | A custom 5-token draft model from z-lab specialized for Qwen3.6 code workloads. Replaces MTP with a parallel external draft. |
| **EAGLE** | SGLang's MTP equivalent; currently blocked on hybrid attention. |
| **AL (acceptance length)** | Average number of tokens accepted per spec-decode step. AL 3.5 means the model usually gets 3-4 tokens right per round. Higher is better. Theoretical max for n=3 is 4. |
| **Per-position acceptance** | The accept rate at each position 1, 2, 3 of the spec-decode draft. e.g., 92% / 86% / 71% on code means position-1 is almost always right; position-3 is right 71% of the time. |

---

## Engines & infrastructure

| Term | What it means |
|---|---|
| **vLLM** | A production-grade GPU LLM inference engine. Open source, NVIDIA-focused. Powers many cloud inference services. |
| **llama.cpp** | A lightweight CPU-and-GPU inference engine. Works on every platform. Smaller binary, less feature-rich than vLLM. |
| **SGLang** | A high-throughput serving engine with RadixAttention prefix sharing. Often beats vLLM on multi-tenant aggregate. |
| **Genesis patches** 🗑️ | A third-party vLLM patch tree this stack used through mid-2026 to fix Qwen3-Next bugs. **No shipped compose enables it any more** — kept in the glossary only so older discussions and CHANGELOG entries stay readable. The one finding that outlived it is upstream at [vllm#40914](https://github.com/vllm-project/vllm/pull/40914). Anything at runtime. We mount it into vLLM's site-packages. |
| **Cudagraph** | A CUDA optimization that records GPU operation sequences and replays them. Faster than dispatching ops individually. |
| **OpenAI API** | The HTTP API spec (`/v1/chat/completions`, etc.) used by ChatGPT, Claude (via proxy), and many OSS chat tools. We serve this on `localhost:8020` (single-card) or `localhost:8010` (dual-card). |

---

## Multi-card concepts

| Term | What it means |
|---|---|
| **TP=2 / tensor parallelism** | Splits each model layer's weights across both GPUs; layers compute together, results combined via NCCL all-reduce. Doubles effective VRAM (48 GB total). |
| **PP / pipeline parallelism** | Different layers go on different GPUs; not used in this stack. |
| **NVLink** | NVIDIA's high-bandwidth GPU-to-GPU interconnect (~600 GB/s on H100, ~200 GB/s on 3090 with bridge). Not required by this stack — we run PCIe-only. |
| **All-reduce** | The collective op TP uses to combine partial results. PCIe-only consumer Ampere is ~3-5× slower than NVLink. |
| **Concurrent streams** | Multiple users/agents serving simultaneously. KV pool is shared; each stream gets a slice. |

---

## Model architecture

| Term | What it means |
|---|---|
| **Qwen3-Next** | Qwen team's hybrid attention architecture used in Qwen3.5/3.6. Interleaves DeltaNet (linear attention) layers with standard attention layers. |
| **DeltaNet / GDN** | "Gated DeltaNet" — a linear-attention layer type. Qwen3.6-27B has 48 GDN + 16 standard attention layers (3:1 ratio). |
| **Hybrid attention** | Architectures mixing standard attention with linear-attention or state-space layers. Qwen3-Next, Mamba-class models, Jamba. |
| **MTP head / `mtp.fc`** | Multi-Token Prediction head — a small extra network in the model that drafts speculative tokens. Lorbus's quant preserves it in BF16 (rather than INT4) so vLLM can load and use it. |

---

## Tool calling / API features

| Term | What it means |
|---|---|
| **Tool calling** | The model emits structured calls to external functions you define (e.g., `get_weather(...)`); your code runs them and feeds results back. |
| **Reasoning / thinking mode** | The model emits intermediate reasoning steps before its final answer. Set `chat_template_kwargs.enable_thinking=true`. |
| **Streaming** | Tokens arrive incrementally via Server-Sent Events. Faster perceived UX. |
| **Vision** | The model can accept images alongside text. Powered by an integrated vision tower. |
| **Tool prefill** | When an agent calls a tool and feeds the (potentially huge) tool response back, the next inference call has to "prefill" all that history. Big tool returns can OOM if context tier isn't set right. |
