# Inference engine comparison — vLLM, llama.cpp, SGLang, ktransformers, ik_llama.cpp

Pragmatic comparison of the five engines this stack interacts with most. **Versions captured 2026-05-07** — all five projects move fast; verify against upstream release notes before making a long-horizon decision. Linked sources at the bottom.

This doc isn't an evangelism piece. It's a *picker* — which engine to reach for when the workload demands a specific feature, and where each engine still has structural gaps. Coverage is biased toward what matters on **club-3090's hardware class** (consumer Ampere/Ada/Blackwell + 24-32 GB VRAM cards + community workstations), not enterprise H100/B200 deployments.

**Note on ik_llama.cpp**: it's a [Iwan Kawrakow](https://github.com/ikawrakow) fork of llama.cpp. Most rows mirror mainline because it inherits the codebase. Differences are flagged where they matter — chiefly **MTP merged on main** (vs mainline's open PR), **fused MoE kernels for DeepSeek-R1 / Kimi**, and the **IQ_K quant series** (IQ4_KT, IQ3_K_R4) that mainline doesn't have.

---

## TL;DR — pick by workload

| Workload | Engine | Why |
|---|---|---|
| **Production multi-tenant chat / API** (24-48 GB VRAM, fits-VRAM models) | **vLLM** | Continuous batching + paged attention + extensive tool-call/structured-output support; Genesis patches close most Qwen3-Next gaps |
| **Single-card max-quality on small-VRAM rigs** (12-24 GB), bulletproof | **llama.cpp** | Engine-agnostic memory model; no Cliff 1/2 footguns; broadest model coverage; q4_0/q8_0 KV |
| **Hyper-optimized prefix caching, structured outputs, multimodal at scale** | **SGLang** | RadixAttention-class prefix cache + FSM-native structured outputs + best-in-class spec-decode V2 |
| **Big MoE (>VRAM) on consumer hardware** (M2/Kimi/DeepSeek-class) | **ktransformers** (or SGLang+kt-kernel) | Router-aware hot-expert caching; 1.5-2× decode TPS over llama.cpp's layer-uniform `--n-cpu-moe` |
| **Apple Silicon** | **llama.cpp** (Metal) or **SGLang** (MLX backend, v0.5.10+) | Both work; MLX backend is newer but native |
| **Qwen3.x + MTP on llama.cpp without PR-branch building** | **ik_llama.cpp** | Qwen MTP merged on main (vs mainline's open [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)); GLM-4.x MTP also working |
| **DeepSeek-R1 / Kimi-K2 / large MoE on consumer hardware (alternative to ktransformers)** | **ik_llama.cpp** | Fused MoE kernels + smart expert reduction (`-ser`) + on-the-fly MLA tensors; ships as a llama.cpp fork rather than separate engine |

---

## Versions + release cadence (2026-05-07)

| Engine | Latest | Release cadence | Maturity | License |
|---|---|---|---|---|
| **vLLM** | v0.20.1 (May 4) + nightlies | Major/month, nightlies/day | Production | Apache 2.0 |
| **llama.cpp** | b9050 (May 7) | Tagged releases multi-daily | Production | MIT |
| **SGLang** | v0.5.11 (May 5) | ~Monthly minors, weekly patches | Production | Apache 2.0 |
| **ktransformers** | v0.6.2 (May 3) | ~Monthly minors | Research → graduating | Apache 2.0 |
| **ik_llama.cpp** | rolling (active main, no tagged releases) | PR-by-PR, smaller community | Production fork | MIT |

All five are actively developed. ktransformers is positioned as research but production paths exist (kt-kernel + SGLang). ik_llama.cpp tracks mainline llama.cpp's API surface but adds quant types (IQ_K series), MTP-merged-on-main, and fused MoE kernels for big-MoE workloads — chiefly used as a drop-in `llama-server` replacement when those features matter.

---

## Hardware support

| Feature | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **NVIDIA CUDA (CC 7.0 Volta)** | ❌ (needs ≥7.5) | ✅ | ❌ (needs ≥8.0) | ❌ (needs ≥8.0) | ✅ |
| **NVIDIA CC 7.5 Turing** | ✅ | ✅ | ⚠️ Limited | ❌ | ✅ |
| **NVIDIA CC 8.0+ Ampere (3090)** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **NVIDIA CC 8.6 Ampere consumer** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **NVIDIA CC 8.9 Ada (4090)** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **NVIDIA CC 9.0+ Hopper (H100)** | ✅ + FA3 | ✅ | ✅ + TRT-LLM NSA | ✅ | ✅ |
| **NVIDIA CC 12.0 Blackwell (5090)** | ✅ | ✅ | ✅ + 8× 5090 validated | ✅ + kt-kernel validated | ✅ |
| **AMD ROCm** | ✅ | ✅ | ✅ + DFLASH on ROCm | ⚠️ Limited | ✅ |
| **Apple Silicon (Metal)** | ❌ | ✅ first-class | ✅ MLX backend (v0.5.10) | ❌ | ✅ first-class |
| **Intel GPU (XPU/SYCL)** | ⚠️ | ✅ | ⚠️ | ❌ | ✅ |
| **CPU-only inference** | ⚠️ Slow path | ✅ | ⚠️ | ❌ (kernel library) | ✅ |
| **Vulkan (cross-platform)** | ❌ | ✅ | ❌ | ❌ | ✅ |

**Recommendation for our stack** (RTX 3090 sm_86 + 2× consumer-board PCIe 4.0): all five work. Pick by feature, not hardware support.

---

## Quantization formats

### Weight quants

| Format | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **GGUF (Q*, K*, IQ*)** | ⚠️ **Experimental** ([official docs](https://docs.vllm.ai/en/stable/features/quantization/gguf/) flag "under-optimized"; single-file only — multi-file needs gguf-split merge; UD-* prefixes via [PR #39471](https://github.com/vllm-project/vllm/pull/39471) merged 2026-04-10; tokenizer conversion unstable on large-vocab models like Qwen3.6) | ✅ Native | ⚠️ Recent | ⚠️ | ✅ Native + IQ_K series |
| **AutoRound INT4 (Marlin)** | ✅ + our [PR #40361](https://github.com/vllm-project/vllm/pull/40361) | ❌ | ✅ | ⚠️ | ❌ |
| **GPTQ INT4** | ✅ | ❌ | ✅ | ✅ | ❌ |
| **AWQ INT4** | ✅ | ❌ | ✅ | ⚠️ | ❌ |
| **FP8 (e4m3, e5m2)** | ✅ multi-backend (Marlin/TRTLLM/MXFP8) | ⚠️ Limited | ✅ FlashInfer MXFP8 | ✅ Native (M2.x) | ⚠️ Limited (inherits llama.cpp) |
| **FP4 / MXFP4** | ✅ (online MoE quant) | ⚠️ | ✅ MXFP4 kernels | ✅ DeepSeek-V4 / kt-kernel | ⚠️ Limited |
| **NVFP4 (Blackwell)** | ✅ rescaled weight scales | ❌ | ✅ | ⚠️ | ❌ |
| **INT8** | ✅ | ✅ | ✅ | ✅ AVX2-VNNI RAWINT4 (consumer CPU) | ✅ |
| **BF16** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FP16** | ✅ | ✅ | ✅ | ✅ | ✅ |

### KV cache types

| KV format | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **FP16 / BF16** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FP8 e4m3 / e5m2** | ✅ | ⚠️ | ✅ | ✅ | ⚠️ (inherits llama.cpp) |
| **INT8 per-token-head** | ⚠️ ([PR #40391](https://github.com/vllm-project/vllm/pull/40391) for hybrid pages) | ❌ | ✅ | ⚠️ | ✅ |
| **Q4_0 (4.5 bit)** | ❌ | ✅ | ❌ | ❌ | ✅ |
| **Q8_0 (8.5 bit)** | ❌ | ✅ | ❌ | ❌ | ✅ |
| **TurboQuant 3-bit (TQ3)** | ✅ via Genesis | ✅ ([PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) WIP) | ⚠️ Same kernel bug as vLLM PR #40361 | ❌ | 🟡 [Issue #1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509) — CPU complete + CUDA written, awaiting merge |
| **2-bit KV** | ✅ ([PR #38479](https://github.com/vllm-project/vllm/pull/38479)) | ❌ | ❌ | ❌ | ❌ |
| **CPU KV offload** | ✅ pluggable policies ([PR #37160](https://github.com/vllm-project/vllm/pull/37160)) | ⚠️ via mmap | ✅ Decode Radix Cache (v0.5.11) | N/A | ⚠️ via mmap |
| **Disk KV offload** | ✅ FlexKV ([PR #34328](https://github.com/vllm-project/vllm/pull/34328)) + LMCache | ⚠️ | ⚠️ | ❌ | ⚠️ |

**Notes**: llama.cpp's q4_0 KV is the *only* engine choice when you need 4-bit KV at 200K+ ctx on 24 GB VRAM. vLLM's TurboQuant 3-bit KV (TQ3) is comparable ratio but Qwen3-Next-only via Sandermage's [Genesis patches](https://github.com/Sandermage/genesis-vllm-patches).

---

## Speculative decoding

| Method | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **MTP (Multi-Token Prediction)** | ✅ broad model coverage (Qwen3.x, Gemma4, MiniMax-M2, DeepSeek) | ✅ **Merged on main** ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), am17an, merged 2026-05-16) | ✅ Spec V2 default (overlap scheduling) | ⚠️ via SGLang front-end | ✅ **Merged on main** (GLM-4.x + Qwen MTP) + **two-stage ngram+MTP** ([PR #1789](https://github.com/ikawrakow/ik_llama.cpp/pull/1789)) |
| **EAGLE / EAGLE-3** | ✅ Eagle3 (Qwen3.5, Gemma4, MiniMax-M2) | ❌ | ✅ Day-0 for newest models | ❌ | ❌ |
| **DFlash (block-diffusion)** | ✅ ([PR #41703](https://github.com/vllm-project/vllm/pull/41703) Codex-rebased; Luce z-lab) | ⚠️ Luce fork (server-only) | ✅ DFLASH cross-backend incl. ROCm (v0.5.11) | ❌ | ❌ |
| **N-gram prompt-lookup** | ✅ GPU impl + async scheduler ([PR #29184](https://github.com/vllm-project/vllm/pull/29184)) | ✅ | ✅ | ❌ | ✅ + **ngram-mod** (enhanced self-spec, chainable with MTP via `--spec-stage`) |
| **Draft model (separate small)** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **PFlash (prompt-compression)** | ❌ | ⚠️ Luce experimental | ❌ | ❌ | ❌ |
| **Async / overlap scheduling** | ✅ Zero-bubble ([PR #32951](https://github.com/vllm-project/vllm/pull/32951)) | ❌ | ✅ Spec V2 with overlap (default v0.5.11) | ⚠️ via SGLang | ❌ |

**Notes on Qwen3-Next family** (DeltaNet hybrid attention): **only MTP works** today. EAGLE/DFlash/draft/ngram all blocked on KV rollback support — see vLLM [#39931](https://github.com/vllm-project/vllm/issues/39931). MTP is the rolling default for our shipped Qwen3.6-27B composes.

---

## MoE features

| Feature | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **Tensor Parallel** | ✅ | ✅ | ✅ | ✅ | ⚠️ Limited (inherits llama.cpp) |
| **Expert Parallel (EP)** | ✅ Elastic EP M2 ([PR #35627](https://github.com/vllm-project/vllm/pull/35627)) | ❌ | ✅ Independent MoE/attention tuning (v0.5.11) | ✅ kt-kernel | ❌ |
| **Layer-uniform expert offload to CPU** | ⚠️ generic `--cpu-offload-gb` (catastrophic for MoE) | ✅ `--n-cpu-moe N` / `-ot` regex | ⚠️ via kt-kernel | N/A | ✅ `-ot` regex + `-ser` smart expert reduction (key strength) |
| **Router-aware hot-expert caching** | ❌ | ❌ ([feature request #20757](https://github.com/ggml-org/llama.cpp/issues/20757) open) | ✅ via kt-kernel integration | ✅ Native (purpose-built) | ❌ (same gap as mainline) |
| **Disk-offload for cold experts** | ⚠️ via FlexKV / LMCache | ⚠️ via mmap | ⚠️ via kt-kernel | ⚠️ | ⚠️ via mmap |
| **Shared-expert handling** | ✅ ([PR #35153](https://github.com/vllm-project/vllm/pull/35153) Oracle Flow) | ✅ | ✅ | ✅ | ✅ + fused MoE kernels (DeepSeek-R1 / Kimi optimized) ⭐ |
| **MXFP4 MoE online quant** | ✅ | ❌ | ✅ MXFP8/MXFP4 | ✅ DeepSeek-V4-Flash (kt-kernel MXFP4 op) | ❌ |
| **Disaggregated prefill/decode** | ✅ ~5% overhead reduction | ❌ | ✅ Decode Radix Cache (v0.5.11) | ⚠️ via SGLang | ❌ |

**Bottom line on big-MoE**: ktransformers (or kt-kernel via SGLang) is the only engine with router-aware hot-expert caching. Sustainable benchmark on DeepSeek-V2 (2× 3090): **9 TPS @ 45% VRAM** (kt) vs **7.5 TPS @ 95% VRAM** (llama.cpp `--n-cpu-moe`) — +20% TPS at half the VRAM.

---

## Distributed / multi-card

| Feature | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **Tensor Parallel (TP)** | ✅ | ✅ `-sm tensor` | ✅ | ✅ | ⚠️ Limited (inherits llama.cpp) |
| **Pipeline Parallel (PP)** | ✅ + cudagraphs ([PR #35162](https://github.com/vllm-project/vllm/pull/35162)) | ⚠️ Limited | ✅ | ✅ | ⚠️ Limited |
| **Expert Parallel (EP)** | ✅ Elastic EP M2 | ❌ | ✅ + all-reduce fusion | ✅ kt-kernel | ❌ |
| **Data Parallel (DP)** | ✅ | ❌ | ✅ | ⚠️ | ❌ |
| **Context Parallel (long-ctx)** | ✅ DeepSeek context-parallel | ❌ | ✅ Enhanced (v0.5.11) | ❌ | ❌ |
| **Disaggregated PD** | ✅ ~5% reduction | ❌ | ✅ Decode Radix Cache | ❌ | ❌ |
| **NVLink awareness** | ✅ | ⚠️ Generic NCCL | ✅ | ⚠️ | ⚠️ Generic NCCL |
| **Custom all-reduce** | ✅ (must disable on PCIe-only) | N/A | ✅ | N/A | N/A |
| **Patched-P2P (consumer GPUs)** | ✅ (Sam McLeod's guide) | ✅ | ✅ | ✅ | ✅ |

**Cross-rig validation on club-3090's 2× 3090 PCIe**: NVLink lift = +15-19% on DFlash paths ([disc #19](https://github.com/noonghunna/club-3090/discussions/19)); patched-P2P captures ~60-80% of NVLink's lift on DFlash, ~13% on plain dual ([issue #91/#95](https://github.com/noonghunna/club-3090/issues/91)).

---

## Memory / KV cache features

| Feature | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **Paged attention** | ✅ Original (Berkeley) | ⚠️ via continuous-batch slots | ✅ | ✅ via SGLang | ⚠️ via parallel slots (inherits) |
| **Prefix caching (in-VRAM)** | ✅ default-on | ✅ slot-based | ✅ **RadixAttention** (best-in-class) | ✅ via SGLang | ✅ slot-based (inherits) |
| **CPU-tier prefix cache (warm restart)** | ✅ via [LMCache](https://github.com/LMCache/LMCache) connector | ⚠️ via mmap | ✅ Decode Radix Cache (v0.5.11) | ⚠️ | ⚠️ via mmap |
| **Disk-tier prefix cache (cold restart)** | ✅ LMCache + FlexKV | ⚠️ | ✅ Decode Radix Cache | ⚠️ | ⚠️ |
| **Sparse attention (long ctx)** | ⚠️ HiSparse research | ⚠️ | ✅ HiSparse (v0.5.10) | ❌ | ⚠️ |
| **Mamba/SSM hybrid handling** | ✅ Mamba state corruption fixes ([PR #37728](https://github.com/vllm-project/vllm/pull/37728)) | ⚠️ Limited | ✅ SSM-FA hybrid via NIXL | ⚠️ | ⚠️ Limited |
| **TMA / async copies (Hopper+)** | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| **Chunked prefill** | ✅ default | ⚠️ via parallel slots | ✅ | ✅ + Layerwise Prefill | ⚠️ via parallel slots |
| **Continuous batching** | ✅ | ✅ via parallel slots | ✅ | ✅ via SGLang | ✅ via parallel slots |

---

## Multimodal

| Modality | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **Vision (image)** | ✅ broad VLM coverage (Gemma4, Granite Vision, Hunyuan v3, ViT cudagraphs) | ✅ via `--mmproj` | ✅ optimized encoders | ⚠️ Limited | ✅ via `--mmproj` + on-the-fly MLA tensors for DeepSeek |
| **Audio** | ✅ Nemotron / Qwen3-Omni | ✅ Granite Speech (b9045) | ⚠️ Limited | ❌ | ✅ inherits Granite Speech support |
| **Video** | ⚠️ Frame-by-frame | ⚠️ | ✅ Diffusion (LTX-2, FLUX) | ❌ | ⚠️ |
| **Image generation** | ❌ | ❌ | ✅ Diffusion (FLUX, Qwen-Image fused kernels) | ❌ | ❌ |

**Practical for our stack**: Gemma 4 vision + Qwen3.6 vision both work via vLLM (see `models/gemma-4-31b/` and `models/qwen3.6-27b/long-vision.yml`); llama.cpp `--mmproj` is the single-card fallback.

---

## Structured output / tool calling

| Feature | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **OpenAI tool-call API compat** | ✅ | ✅ via `--jinja` | ✅ | ✅ via SGLang | ✅ via `--jinja` |
| **Custom tool parsers** | ✅ Per-model (Gemma4, Kimi-K2.5, GigaChat 3.1, qwen3_coder) | ⚠️ Generic | ✅ | ⚠️ | ⚠️ Generic |
| **Grammar (GBNF / EBNF)** | ✅ via outlines/lm-format-enforcer/xgrammar | ✅ Native GBNF | ✅ FSM-based | ⚠️ | ✅ Native GBNF |
| **JSON schema mode** | ✅ | ✅ | ✅ Best-in-class FSM | ⚠️ | ✅ |
| **Function calling enforcement** | ✅ tool-choice | ⚠️ Soft | ✅ Strict | ⚠️ | ⚠️ Soft |
| **Reasoning-channel separation** | ✅ qwen3 reasoning parser | ⚠️ Default ON via peg-native + `<think>` parsing → routes to `reasoning_content` field. **Most clients (incl. opencode) ignore this and hang** ([issue #97](https://github.com/noonghunna/club-3090/issues/97)). Workaround: `--reasoning-format none` flag (now default in our `llamacpp/default` compose). | ✅ | ⚠️ | ⚠️ Inherits llama.cpp default |
| **Streaming tool-call deltas** | ✅ + Anthropic API compat | ✅ | ✅ + Responses API streaming | ⚠️ | ✅ |

**Notes**: SGLang's RadixAttention + native FSM makes structured output the headline strength. Our [bounded-thinking compose](../models/qwen3.6-27b/vllm/compose/single/autoround-int4/bounded-thinking.yml) uses vLLM's xgrammar; could re-do on SGLang for a probable speedup but vLLM is the daily-driver here.

---

## Model coverage (latest architectures, 2026 lens)

| Model family | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **Qwen3.5 / Qwen3.6 (incl. 80B-A3B)** | ✅ | ✅ | ✅ Day-0 (Qwen3.6 v0.5.11) | ✅ | ✅ |
| **Qwen3-Next family (DeltaNet hybrid)** | ✅ via Genesis patches | ✅ | ✅ | ⚠️ | ✅ |
| **Gemma 4 / Gemma-4 31B** | ✅ + MTP ([PR #41745](https://github.com/vllm-project/vllm/pull/41745)) + DFlash ([PR #41703](https://github.com/vllm-project/vllm/pull/41703)) | ✅ via mmproj | ✅ Day-0 | ⚠️ | ✅ via mmproj |
| **DeepSeek V3 / R1 / V4-Flash** | ✅ | ✅ | ✅ + TRT-LLM NSA (3-5× on Blackwell) | ✅ Native (kt-kernel MXFP4 for V4-Flash) | ✅ |
| **Kimi-K2.5 / K2.6** | ✅ tool parser ([PR #37438](https://github.com/vllm-project/vllm/pull/37438)) | ✅ | ✅ Day-0 K2.6 (v0.5.11) | ✅ | ✅ |
| **MiniMax-M2 / M2.5 / M2.7 (incl. REAP)** | ⚠️ via custom code | ✅ via REAP'd GGUF | ✅ Day-0 M2.5 / M2.6 | ✅ Day-0 + native FP8 | ✅ via REAP'd GGUF |
| **GLM-4.5 / GLM-5 / GLM-5.1** | ✅ | ✅ | ✅ Day-0 GLM-5.1 + MoE | ✅ GLM-5 (v0.6.x) | ✅ |
| **Mistral Medium 3.5** | ⚠️ | ✅ | ✅ Day-0 | ⚠️ | ✅ |
| **GPT-OSS-120B** | ✅ | ✅ | ✅ | ✅ via kt-kernel | ✅ |
| **Llama 3 / Llama 4** | ✅ | ✅ | ✅ | ⚠️ Experimental L4 | ✅ |
| **EXAONE-4.5 / Phi-4-reasoning-vision** | ✅ (v0.20) | ⚠️ | ⚠️ | ❌ | ⚠️ |

**Day-0 wins for new MoE releases**: SGLang and ktransformers both ship same-day support for new MiniMax / Kimi / GLM releases. vLLM tends to lag 1-2 weeks on new MoE architectures (PR-driven). llama.cpp typically lags 2-4 weeks (community-driven).

---

## API / serving surface

| Feature | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **OpenAI-compatible HTTP** | ✅ | ✅ (`llama-server`) | ✅ | ✅ via SGLang | ✅ (`llama-server` compatible CLI) |
| **Anthropic API compat** | ⚠️ | ❌ | ✅ Direct (v0.5.9) | ⚠️ | ❌ |
| **Native Python SDK** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **gRPC** | ✅ ([PR #36169](https://github.com/vllm-project/vllm/pull/36169)) | ❌ | ⚠️ | ❌ | ❌ |
| **Responses API streaming** | ✅ | ⚠️ | ✅ | ⚠️ | ⚠️ |
| **Docker images (official)** | ✅ `vllm/vllm-openai` | ✅ `ghcr.io/ggml-org/llama.cpp` | ✅ `lmsysorg/sglang` | ❌ **No official Docker image** | ✅ `ghcr.io/ikawrakow/ik-llama-cpp` (cu12/cu13 tags) |
| **Pluggable backends** | ✅ | ✅ | ✅ kt-kernel pluggable | ✅ kt-kernel as backend | ✅ + runtime quant repacking |

**Note on ktransformers Docker absence**: this is a real friction point for our compose-based stack. The pip install path works in a conda env but breaks the "every service is a compose dir" convention.

---

## When to pick which engine

Decision tree for a new model deployment on club-3090's hardware class:

```
Q1: Does the model fit your VRAM at desired quant?
├── YES, fits comfortably → 
│      Q2: Need max throughput + multi-tenant?
│      ├── YES → vLLM (production daily-driver)
│      └── NO → 
│             Q3: Want bulletproof / no-cliffs?
│             ├── YES → llama.cpp (single-card)
│             └── NO → vLLM still
│
└── NO, doesn't fit → 
       Q4: Is it MoE?
       ├── NO (dense doesn't fit) → No good options on consumer rig.
       │      Lower quant, smaller model, or rent cloud.
       └── YES → 
              Q5: Big MoE (>2× VRAM, < 96 GB RAM)?
              ├── YES → ktransformers (or SGLang+kt-kernel)
              └── NO (just slightly over) → llama.cpp `--n-cpu-moe`
                  OR ik_llama.cpp (better DeepSeek/Kimi MoE kernels)
```

### Specific picks for club-3090's shipped models

| Model | Daily driver | Reason |
|---|---|---|
| **Qwen3.6-27B** (dense hybrid, fits VRAM) | **vLLM** + Genesis patches | Multi-tenant, full feature set, Cliff 1/2 closed on TP=2 |
| **Qwen3.6-27B** (single-card no-cliffs path) | **llama.cpp** | Different memory model; no Cliff 2b under multi-turn |
| **Qwen3.6-27B** (single-card with MTP, no PR-branch building) | **ik_llama.cpp** | MTP merged on main — get the ~+34% TPS lift without rebuilding from PR #22673 |
| **Gemma 4 31B** (dual-card) | **vLLM** + MTP/DFlash overlays | Best spec-decode story, vision support |
| **Gemma 4 31B** (single-card long-ctx + spec-dec) | **beellama.cpp** (experimental, unofficial sm_86 image) | Only single-card engine with Gemma-4 windowed KV *and* DFlash spec-dec in one GGUF |
| **Carnice / Qwopus / variants** | **vLLM** | Same daily-driver path |
| **(future) MiniMax-M2.7-REAP-172B** | **ktransformers** + SGLang | Big-MoE > VRAM, router-aware caching is the unlock |
| **(future) GPT-OSS-120B** | **ktransformers** or **llama.cpp** `--n-cpu-moe` | Either works; ktransformers ~+20% TPS but harder to deploy |
| **(future) DeepSeek-V4-Flash** | **ktransformers** (kt-kernel native MXFP4) | Best support for V4-Flash architecture |

---

## Honest gaps / what each *isn't* great at

### vLLM
- **Single-card cliffs on Qwen3-Next**: Cliff 2 / Cliff 2b on long-ctx single-card (24 GB) — see [docs/CLIFFS.md](CLIFFS.md). Mitigated by Genesis but not fully closed.
- **Layer-uniform CPU offload only** (no router-aware MoE caching).
- **Patch-heavy for Qwen3-Next family** — needs Genesis-vllm-patches for production-grade behavior.

### llama.cpp
- **Single-stream-only on dense models** (parallel slots exist but TP/EP are limited).
- **No FP8 weight quants** — stuck with GGUF Q* / K* / IQ*.
- **MTP merged on main** ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), 2026-05-16) — no longer needs PR-branch building.
- **Layer-uniform `--n-cpu-moe`** — no router-aware caching upstream yet ([feature request #20757](https://github.com/ggml-org/llama.cpp/issues/20757)).

### SGLang
- **More fragile than vLLM on production** — fewer cross-rig anchor data points in the wild.
- **Marlin pad-sub-tile-n bug** mirrors vLLM's [PR #40361](https://github.com/vllm-project/vllm/pull/40361) — blocks AutoRound INT4 + EAGLE on TP=2 sub-tile-N shards (same kernel-line fix applies).
- **Less mature on Apple Silicon** despite MLX backend.

### ktransformers
- **No Docker image** — breaks compose-based deployment patterns.
- **Smaller community** — issues/PRs move slower than vLLM/llama.cpp.
- **NVIDIA-only** — no AMD/Apple/Vulkan paths.
- **Yaml-based layer-placement config per model** — non-trivial first-time setup.
- **Targets 200 GB RAM + Sapphire Rapids CPU** for headline numbers; consumer rigs land in degraded-mode (AVX2 backend, REAP'd quants).

### ik_llama.cpp
- **Smaller community than mainline llama.cpp** — bugs take longer to surface, fewer cross-rig data points.
- **No tagged releases** — rolls on main; no version pinning story for production users.
- **Official Docker image** ships (`ghcr.io/ikawrakow/ik-llama-cpp:cu13-server`, digest-pinned in our composes). This stack uses it for the **[advanced-quant track](engines/IK_LLAMA.md)** — fork-exclusive **IQK imatrix quants** (`IQ4_KS`), the best quality-per-bit in the GGUF world. See [QUANTIZATION.md](QUANTIZATION.md).
- **Diverging quant naming from mainline** — IQ_K series flags differ; cross-engine GGUF compatibility caveats.
- **Spec-decode coverage** — MTP merged + two-stage ngram+MTP ([PR #1789](https://github.com/ikawrakow/ik_llama.cpp/pull/1789)) + hybrid-aware recurrent checkpoints ([PR #1774](https://github.com/ikawrakow/ik_llama.cpp/pull/1774)). No EAGLE3 / DFlash; lags mainline on those research paths.
- **Smaller maintainer surface** — primarily Iwan Kawrakow + a handful of contributors. Not the right pick for production where you need >1 person to debug a kernel issue.

### beellama.cpp

[beellama.cpp](https://github.com/Anbeeld/beellama.cpp) is Anbeeld's llama.cpp fork. Its two fork-exclusive levers are **DFlash** (cross-attention speculative decoding — a small DFlash draft GGUF reads the target's recent hidden states and predicts multiple tokens ahead, verified in one target pass) and **TurboQuant / TCQ KV cache types** on top of mainline-lineage **SWA-aware windowed KV**. On this stack it is the single-card answer for two cases the other engines can't cover together:

- **Gemma-4-31B single-card long-context + spec-dec** — the only engine that does Gemma-4 windowed KV (big context) *and* a Gemma-4 spec-dec arch in one GGUF. ik_llama has the MTP arch but allocates full KV (≈24K single-card wall); mainline llama.cpp windows KV but its plain Gemma-4 decode is slow (~10 TPS).
- **Qwen3.6-27B tool-grammar-neutral spec-dec** — the external DFlash drafter does not share the target distribution, so it does not amplify the CodeAct attractor the built-in MTP head does ([club-3090#237](https://github.com/noonghunna/club-3090/discussions/237)).

Honest gaps:

- **No *official* upstream Docker image yet (upstream ships Windows binaries + source only); we publish an unofficial *multi-arch* build.** We compile `Anbeeld/beellama.cpp` (MIT) for **sm_86 / sm_89 / sm_120 (RTX 3090 / 4090 / 5090)** and publish it as **`ghcr.io/noonghunna/beellama-cpp:multiarch-b9459-07ac3ce`**. Users on any of those cards can **pull and run**:

  ```bash
  docker pull ghcr.io/noonghunna/beellama-cpp:multiarch-b9459-07ac3ce
  ```

  `beellama/dflash` (Qwen3.6-27B) is now the **single-card default** (⚠️ caveats); `beellama/gemma-dflash` stays 🧪 **Experimental**. **Caveat: sm_89 / sm_120 are compiled but UNVALIDATED** — only sm_86 / RTX 3090 is verified on club-3090's rig, so 4090 / 5090 users should treat it as unverified and report back via the *numbers-from-your-rig* issue template. Anbeeld's official v0.3.0 (CI + Docker) is in progress ([discussion #239](https://github.com/noonghunna/club-3090/discussions/239)); when it lands we drop our unofficial image. **To build your own image** (a different arch, or to self-host), from a clone of the fork (Linux GCC + CUDA):

  ```bash
  cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  ```

  Use `-DCMAKE_CUDA_ARCHITECTURES=89` for an RTX 4090, `120` for a 5090, etc. To produce a Docker image for your arch, bake the fork's `.devops/cuda.Dockerfile` with `CUDA_DOCKER_ARCH=<arch>` (a **semicolon list** like `"86;89;120"` builds one multi-arch image — that is exactly how our published image is built) and the `-DGGML_CUDA_FA_ALL_QUANTS=ON` flag — the stock Dockerfile omits it, and **`FA_ALL_QUANTS` is required for the TurboQuant / TCQ cache types** (and for the `q5_0`/`q4_1` KV the composes default to). Point the composes at it per-launch with `BEELLAMA_IMAGE=...`.
- **Server target ENTRYPOINT is `/app/llama-server`** — compose `command:` is server flags only (same shape as the ik-llama / llama-cpp composes).
- **Single-stream / `-np 1`** — DFlash is single-slot by default; same compute-bound single-card caveat as the other llama.cpp-family composes.

---

## Cross-engine bug / feature parity tracker

| Issue | vLLM | llama.cpp | SGLang | ktransformers | ik_llama.cpp |
|---|---|---|---|---|---|
| **Marlin pad-sub-tile-n** (output-dim shards <64 on TP=2 W4A16) | 🟡 [PR #40361](https://github.com/vllm-project/vllm/pull/40361) open | N/A | 🟡 Same bug, same fix applies | N/A | N/A |
| **DeltaNet rollback** (blocks EAGLE/DFlash/draft on Qwen3-Next) | 🔴 [#39931](https://github.com/vllm-project/vllm/issues/39931) | 🔴 | 🔴 | 🔴 | 🔴 |
| **MTP for Gemma 4** | 🟢 [PR #41745](https://github.com/vllm-project/vllm/pull/41745) merged | ❌ | ✅ Day-0 | ⚠️ | ❌ |
| **DFlash for Gemma 4** | 🟡 [PR #41703](https://github.com/vllm-project/vllm/pull/41703) Codex-rebased | ⚠️ Luce fork | ✅ Day-0 ROCm + CUDA | ❌ | ⚠️ Luce fork |
| **qwen3coder `<tool_call>`-in-prose silent SSE** | ⚫ Local sidecar (issue #72) | N/A | ⚫ Same root cause | N/A | N/A |
| **Per-token-head KV on hybrid pages** | 🟡 [PR #40391](https://github.com/vllm-project/vllm/pull/40391) | N/A | 🟡 | N/A | N/A |
| **Workspace lock strictness (vLLM 0.20)** | 🟢 [PR #39226](https://github.com/vllm-project/vllm/pull/39226) merged | N/A | N/A | N/A | N/A |
| **TurboQuant CPU + CUDA** | 🟢 ✅ via Genesis | 🟡 [PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) (CPU first; CUDA no PR) | 🟡 ⚠️ Same kernel bug as PR #40361 | ❌ | 🟡 [Issue #1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509) — CPU complete + CUDA written, awaiting validation/merge |
| **MTP merged on llama.cpp family** | N/A | 🟢 ✅ [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) merged 2026-05-16 | N/A | N/A | 🟢 ✅ **Merged on main** (Qwen + GLM-4.x) + two-stage ngram+MTP |

🟢 Landed / 🟡 Open / 🔴 Blocked / ⚫ Local workaround

---

## Sources

- **vLLM**: [GitHub releases](https://github.com/vllm-project/vllm/releases) (v0.20.1, May 4 2026 + nightlies)
- **llama.cpp**: [GitHub releases](https://github.com/ggml-org/llama.cpp/releases) (b9050, May 7 2026)
- **SGLang**: [GitHub releases](https://github.com/sgl-project/sglang/releases) (v0.5.11, May 5 2026)
- **ktransformers**: [GitHub releases](https://github.com/kvcache-ai/ktransformers/releases) (v0.6.2, May 3 2026) + [docs site](https://ktransformers.net/en/docs)
- **ik_llama.cpp**: [GitHub repo](https://github.com/ikawrakow/ik_llama.cpp) (rolling main, no tagged releases) + [Discussion #258](https://github.com/ikawrakow/ik_llama.cpp/discussions/258) (vs llama.cpp / ktransformers positioning) + [Issue #1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509) (TurboQuant ready-for-review)
- **Genesis-vllm-patches** (Qwen3-Next ergonomics layer): [GitHub](https://github.com/Sandermage/genesis-vllm-patches)
- **LMCache** (CPU+disk-tier KV connector for vLLM): [GitHub](https://github.com/LMCache/LMCache)
- **Cross-rig anchors on club-3090's hardware**: [BENCHMARKS.md](../BENCHMARKS.md), [disc #86](https://github.com/noonghunna/club-3090/discussions/86)

---

## Maintenance note

This file ages quickly — all five projects ship frequently (some multiple releases per month, ik_llama.cpp rolls on main). **Refresh quarterly** (or whenever a major release substantively changes the comparison shape). When updating:

1. Re-fetch each project's latest release notes
2. Update version + date row at top
3. Update any 🟡 → 🟢 transitions in the bug-parity tracker
4. Add new feature rows where a project ships something the others don't
5. Don't preserve historical "this used to be ❌, now ✅" — that lives in each project's release notes, not here

Last refresh: **2026-05-07**.
