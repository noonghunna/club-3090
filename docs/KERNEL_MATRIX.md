# Kernel & Attention Backend Matrix (Mid-2026)

**Last updated:** 2026-05-15  
**Focus:** Consumer / Prosumer GPUs (RTX 3090 → 5090) + hybrid/MoE models (Qwen3.6, Gemma4)

This matrix helps decide which inference engine + kernel combination to route composes to.

## Core Modern Kernels

| Kernel / Backend              | Primary Purpose                        | Best Hardware              | Key Strengths                              | Maturity on Ampere (3090) |
|-------------------------------|----------------------------------------|----------------------------|--------------------------------------------|---------------------------|
| **FlashAttention-2**         | Standard attention                    | Ampere+                   | Excellent balance, wide compatibility     | Very High |
| **FlashAttention-3**         | Hopper/Blackwell optimized            | H100/B200+                | FP8, asynchrony, TMA/WGMMA                | Medium (falls back) |
| **FlashInfer**               | Flexible paged / custom attention     | All NVIDIA                | High performance, Triton-based, very tunable | Very High |
| **RadixAttention**           | Prefix sharing (radix tree)           | All                       | Best for chat, RAG, agents                | High |
| **PagedAttention** (vLLM)    | Memory-efficient KV management        | All                       | Low fragmentation, high concurrency       | Very High |
| **Triton Custom Kernels**    | Rapid prototyping / specialized ops   | All                       | Easy to write, high flexibility           | High |
| **TensorRT-LLM Kernels**     | Deep fusion + hardware-specific       | NVIDIA (best on Ada+)     | Highest raw speed on supported hardware   | High |
| **MLA / FlashMLA**           | Multi-head Latent Attention (DeepSeek/Qwen) | All                  | Optimized for compressed KV               | High |

## KV Cache Impact

Different kernels affect KV cache size/efficiency differently. Some only speed up attention computation (no KV size change); others fundamentally change how the cache is laid out, paged, or shared across requests.

| Kernel / System                   | Impacts KV Cache Size? | Main Impact on KV Cache                                                                                                                          | Best For                              | Notes for 3090 / Consumer                |
|-----------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|------------------------------------------|
| **FlashAttention-2 / -3**        | No                    | Reduces HBM traffic during attention computation (IO-aware tiling). Makes attention faster without materializing full attention matrix.         | Speed (especially prefill)           | FA2 is excellent on Ampere. FA3 falls back. |
| **FlashInfer**                   | No                    | Highly optimized paged + custom attention kernels. Excellent at handling non-contiguous KV blocks with minimal overhead.                        | Flexibility + speed on all NVIDIA    | Often the fastest backend on 3090.       |
| **PagedAttention** (vLLM)        | Yes (effective)       | Breaks KV cache into small pages/blocks. Dramatically reduces fragmentation → much higher real-world memory utilization.                        | High concurrency + long context      | Core reason vLLM can serve many users.   |
| **RadixAttention** (SGLang)      | Yes (very strong)     | Stores KV cache in a prefix tree (radix trie). Shares common prefixes across requests → massive memory savings in chat/RAG/agent workloads.    | Prefix-heavy workloads               | Biggest win for multi-turn / agents.     |
| **TensorRT-LLM kernels**         | No                    | Highly fused, hardware-specific kernels. Excellent with FP8/TQ3 but same base KV size.                                                          | Raw speed on new NVIDIA cards        | Less flexible on 3090.                   |
| **Block Diffusion** (DFlash/Zaya) | Indirect but big      | Generates multiple tokens per forward pass → fewer total KV updates per output token. KV cache grows slower in practice.                       | Throughput on bandwidth-limited cards| Very promising for 3090.                 |

## Engine Support Matrix

| Feature / Kernel                  | **vLLM**                          | **SGLang**                          | **TensorRT-LLM**                   | **llama.cpp**                  | Notes for 3090 / Consumer |
|-----------------------------------|-----------------------------------|-------------------------------------|------------------------------------|--------------------------------|---------------------------|
| **Default Attention**            | FlashAttention / FlashInfer      | FlashInfer (preferred)             | Custom TRT kernels + FA3          | Basic FA2 / custom            | FlashInfer best on 3090 |
| **RadixAttention / Prefix Cache**| APC (Automatic Prefix Caching)   | **Native RadixAttention** (best)   | Limited                           | Basic                         | SGLang wins for agents/RAG |
| **Paged KV Cache**               | **Native PagedAttention**        | Yes                                | Yes                               | Basic                         | vLLM strongest |
| **FlashAttention-3**             | Good (Hopper+)                   | Good                               | **Best**                          | No                            | Falls back on Ampere |
| **DFlash (Block Diffusion)**     | Good                             | **Excellent** (early + deep)       | Emerging                          | Limited                       | SGLang currently strongest |
| **MTP / Speculative Decoding**   | Strong (but TQ3 issues)          | Strong                             | Excellent                         | Good                          | vLLM + MTP works well |
| **TQ3 / Advanced KV Quant**      | Yes (but MTP broken)             | Partial / WIP                      | Excellent                         | Good (GGUF)                   | vLLM best but buggy |
| **MoE / Hybrid Models**          | Very Good                        | Excellent (esp. Qwen/DeepSeek)     | Excellent                         | Decent                        | SGLang & TRT-LLM shine |
| **Structured Output**            | Good                             | **Best-in-class**                  | Good                              | Basic                         | SGLang wins |
| **3090 / Ampere Optimization**   | Solid                            | **Strong**                         | Good                              | Very Good                     | SGLang + FlashInfer often fastest |

## Recommendations for club-3090 / Composes

### Primary Routing Guidance

- **Interactive / Chat / Agents / RAG** → **SGLang** (RadixAttention + DFlash)
- **Max raw throughput on new cards (Ada/Blackwell)** → **TensorRT-LLM**
- **Broad compatibility + stability on 3090** → **vLLM** (with FlashInfer backend)
- **Low VRAM / single card / maximum compression** → **llama.cpp** (accept GGUF tradeoffs)
- **Hybrid MoE models (Qwen3.6 35B-A3B, Gemma4 26B)** → **SGLang** or **vLLM** with FlashInfer

### 3090-Specific Tips
- Use **FlashInfer** backend wherever possible (`--attention-backend flashinfer` in vLLM/SGLang).
- TQ3 + MTP is currently unstable in vLLM → prefer FP8/INT8 KV or DFlash.
- SGLang + RadixAttention + DFlash often gives the best real-world experience on consumer hardware.

---

**Cross-references**:
- See [DTYPE_MATRIX.md](DTYPE_MATRIX.md) for quantization + hardware accelerators
- See [KV_MATH.md](KV_MATH.md) for cache calculations per model
- See [INFERENCE_ENGINES.md](INFERENCE_ENGINES.md) for high-level engine comparison & setup

Contributions & updates welcome — this space moves fast!
