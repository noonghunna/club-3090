# Hardware dtype & quant capability matrix

*Last verified: 2026-05-12. Kernel landscape moves fast — when in doubt, cross-check against the [vLLM nightly](https://docs.vllm.ai/), [Intel IPEX-LLM](https://github.com/intel-analytics/ipex-llm), [ROCm](https://rocm.docs.amd.com/), and the vendor arch docs cited at the bottom.*

What each GPU class accelerates *natively* in hardware vs *emulates* in software, across **NVIDIA, Intel, and AMD**. Use this as the input when picking KV-cache dtype, weight quant, and kernel path for a compose targeting a specific rig.

This stack ships **vLLM-CUDA-first** composes (NVIDIA), with **llama.cpp as the cross-vendor fallback** that runs on AMD (ROCm) and Intel (SYCL) too. The non-NVIDIA sections below are forward-looking — they tell you what's *hardware-supported* on those archs, with notes on which inference stack actually uses each path today.

## Table of contents

- [NVIDIA](#nvidia)
- [Intel](#intel)
- [AMD](#amd)
- [Cross-vendor routing](#cross-vendor-routing)
- [Per-arch recommendations for club-3090 composes](#per-arch-recommendations-for-club-3090-composes) (NVIDIA-focused — the primary target)

---

# NVIDIA

> **TL;DR:**
> - **3090/A100 (Ampere)** → AutoRound/AWQ/GPTQ INT4 weights + BF16/INT8 compute + TQ3 KV. fp8 works but is software-emulated (no speedup vs fp16).
> - **4090/L40 (Ada)** → same as Ampere + native FP8 on Tensor Cores (lower throughput than Hopper but real).
> - **H100/H200 (Hopper)** → first arch with full FP8 transformer-engine path. Native FP8 weights + FP8 KV are the fast path.
> - **5090/5080 (Blackwell consumer)** → adds NVFP4 / FP4 / MXFP8 on top of Hopper's FP8.
> - **B100/B200 (Blackwell datacenter)** → same as consumer Blackwell plus enterprise features (TMA, SVE).
>
> See [GLOSSARY.md](GLOSSARY.md) for what AWQ / GPTQ / AutoRound / TurboQuant mean as quant schemes (vs raw dtypes).

---

## At-a-glance — compute dtype support on Tensor Cores

✓ = native hardware Tensor Core support · 🟡 = software-emulated (correct but no speedup vs fp16) · ✗ = not supported · — = arch predates feature

| Compute dtype | Pascal (sm_60/61) | Volta (sm_70) | Turing (sm_75) | Ampere consumer (sm_86) | Ampere DC (sm_80) | Ada (sm_89) | Hopper (sm_90) | Blackwell DC (sm_10x) | Blackwell consumer (sm_120) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FP32 (CUDA cores) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TF32 (TC) | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP16 (TC) | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BF16 (TC) | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| INT8 (TC) | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| INT4 (TC) | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8 E4M3 / E5M2 (TC) | — | — | — | 🟡 SW | 🟡 SW | ✓ | ✓ | ✓ | ✓ |
| FP6 (TC) | — | — | — | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| FP4 / NVFP4 / MXFP4 (TC) | — | — | — | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| MXFP8 (TC) | — | — | — | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

**The big inflection points:**

- **Turing (sm_75)** — first to add **INT8/INT4 Tensor Cores**. T4 was the data-center workhorse for INT8 inference for years. INT4 throughput on Turing is modest; the real volume push came one gen later.
- **Ampere (sm_80/86)** — adds **BF16** and **TF32** TCs. **Roughly 2× the INT8/INT4 throughput of Turing** + new **2:4 structured sparsity** path that doubles peak again on supported kernels. **No native FP8** (despite what some marketing suggested) — FP8 on Ampere is software-only and runs at ≤ fp16 speed. *(Note: TF32 here is a training/mixed-precision convenience format — it's rarely the active path during LLM inference, which generally runs in FP16/BF16/INT8/FP8 instead.)*
- **Ada (sm_89)** — first consumer arch with **native FP8 Tensor Cores** (E4M3 + E5M2). The 4090 / L40 get a real FP8 path. Peak FP8 TFLOPS is lower than Hopper's and — crucially — Ada **lacks Hopper's full transformer-engine integration** (faster on-die format conversion between FP8↔FP16, smarter mixed-precision accumulation, native FP8 KV). So Ada's FP8 is a real inference path but with a thinner kernel ecosystem than H100. Expect ~half the FP8 TFLOPS/clock of Hopper at comparable TC count.
- **Hopper (sm_90)** — first **transformer engine** with FP8 weights + FP8 KV both on-die, plus block-scaling support for higher-accuracy FP8 paths. The fast path for FP8 inference. H100 also adds TMA (Tensor Memory Accelerator).
- **Blackwell (sm_10x DC / sm_120/121 consumer)** — adds **NVFP4 / FP4 / MXFP4** (4-bit float on TCs), **MXFP8**, and **FP6**. The first arch where 4-bit *compute* (not just storage) is a hardware feature. **⚠️ The raw Tensor Cores are similar across datacenter (sm_10x) and consumer (sm_12x) Blackwell, but the *attention/KV kernels are NOT* — see the section below; this is the #1 "it has the hardware so it should work" trap.** *(Current vLLM on consumer Blackwell prefers FP8 KV in practice — NVFP4 *weights* work, but NVFP4 *KV* needs a datacenter-only kernel.)*

---


## Having the Tensor Cores ≠ using them: the two axes that decide real behavior

The matrix above is *silicon capability* — what the Tensor Cores can multiply. It does **not** tell you what vLLM's kernels actually run, and two independent splits decide that. Getting this wrong is how "the 5090 has FP4, so `nvfp4` KV should work" turns into a boot crash.

### Axis 1 — weights vs KV are gated separately

- **Weight quant** (FP8/FP4/INT4 weights → the GEMM) is available wherever the Tensor Cores exist: FP8 weights on **sm_89+** (Ada 4090, Hopper, all Blackwell), FP4/NVFP4 weights on **all Blackwell** — consumer (sm_120/121) *and* datacenter (sm_100/103). If your card is in the matrix column, the weights path works.
- **KV quant** is *not* the same story, because KV compression interacts with the **attention (FMHA) kernel**, which has far narrower arch coverage than the raw Tensor Cores.

### Axis 2 — KV "compute" vs KV "storage": the FMHA gate

A KV dtype is used one of two ways:

- **Storage-only** — KV is stored quantized (e.g. 1 byte/token for FP8), then **dequantized to BF16/FP16 for the attention matmul**. Saves VRAM (bigger pool / longer context), **no compute speedup**. Works on basically any arch vLLM runs on.
- **Native compute** — the attention matmul itself runs in the quantized domain (FP8/FP4). This needs a purpose-built FMHA kernel, and there are only two: **FlashAttention-3** (Hopper **sm_90 only**) and the **trtllm-gen FMHA** (datacenter Blackwell **sm_100/103 only**).

**So on every consumer GPU we care about — Ada (4090, sm_89), consumer Blackwell (5090 / RTX PRO 6000 Blackwell, sm_120), and the DGX Spark (GB10, sm_121) — FP8 KV is storage-only, and `nvfp4` KV does not work at all.** Neither FA3 nor the trtllm-gen FMHA builds for those arches.

Two consequences, both empirically confirmed on this program:

1. **`fp8_e4m3` vs `fp8_e5m2` is speed-neutral on consumer cards.** Both are 1 byte/token stored, both dequant to BF16 for attention — identical throughput. Measured on an RTX 5090: **86.73 vs 86.66 decode TPS** (<0.5%, [disc #571](https://github.com/noonghunna/club-3090/discussions/571)). Their only difference is *numeric* (e4m3 = more mantissa → better KV precision; e5m2 = wider range). Prefer **e4m3 on sm_89+ for precision, not speed**.
2. **`nvfp4` KV crashes on consumer Blackwell** (sm_120/121) — the trtllm-gen FP4 FMHA has no build there ([vLLM #43562](https://github.com/vllm-project/vllm/issues/43562) / [TRT-LLM #10241](https://github.com/NVIDIA/TensorRT-LLM/issues/10241)). NVFP4 *weights* are fine; only the KV/attention path fails.

**Consumer Blackwell is one family (sm_12x).** The 5090 (sm_120), RTX PRO 6000 Blackwell (sm_120), and DGX Spark GB10 (sm_121) share this behavior — do **not** conflate them with datacenter Blackwell (B100/B200/GB200, sm_100/103), which *does* ship the trtllm-gen FMHA and therefore native FP8/FP4 KV compute. On consumer Blackwell the useful KV lever is **compression for capacity** (fit more context — e.g. INT8-PTH / TQ3), not dtype-for-compute.

### Does this apply to the other engines? No — it's a vLLM-family story

Everything above is about the **vLLM family (vLLM + SGLang)**, and it's a *consequence of* those engines having a native-quantized-KV-attention path at all. The **llama.cpp family (mainline / ik-llama / beellama)** — which is where most of our single-card composes live — is a completely different regime with **no consumer-vs-datacenter split**:

- **llama.cpp KV quant (`q4_0`/`q8_0`/`q5_0`/`iq4_nl`/TQ) is storage-only on *every* arch.** The quantized KV is dequantized on-the-fly *inside* the flash-attention kernel; the matmul runs in FP16/FP32. It never touches FP8/FP4 tensor cores — [not even on Hopper](https://github.com/ggml-org/llama.cpp/discussions/22411), where the hardware could. So `q4_0` KV behaves identically on a 3090, 4090, 5090, or DGX Spark — **which is exactly why our single-3090 ik/beellama configs hit 262K, and why they're the right tool for a consumer card that wants big context.**
- **The flip side:** GGUF engines never get a *native-KV-compute* speedup on *any* hardware — the memory saving is the whole benefit, universally. There's no `e4m3`-vs-`e5m2`-for-speed question in GGUF-land because there's no native-KV-compute path. (The only "fast path" nuance is that symmetric K/V quant enables the *fused* FA kernel vs a slower fallback — arch-agnostic, not a tensor-core-precision effect.)
- **Weights too:** GGUF weight quant (K-quants / IQ-quants) is dequant-to-FP16 — a Q4/Q8 GGUF on a 5090 does **not** use the FP8/FP4 tensor cores. The "native FP8/NVFP4 weights" win is vLLM/CUTLASS-only; there's no GGUF equivalent.

**Division of labor, then:** the *native low-precision compute* wins (FP8/FP4 weights, FP8 KV compute) are **vLLM-only, and mostly datacenter-only for KV**. The *capacity* win (KV compression for long context) is delivered **universally and arch-agnostically by the llama.cpp family**. So a consumer card (4090 / 5090 / Spark) that wants long context is best served by the GGUF composes — which were never in the "consumer can't do native KV compute" trap, because they don't do native KV compute anywhere.



## Weight-quantization schemes — storage format vs compute path

Weight quants like AWQ / GPTQ / AutoRound are **storage formats**, not hardware features. The runtime kernel reads the quantized weights, dequantizes on-the-fly, and feeds Tensor Cores in a *different* compute dtype.

| Scheme | Storage | Compute on Ampere/Ada | Compute on Hopper+ | Notes |
|---|---|---|---|---|
| FP16 (no quant) | FP16 | FP16 TC | FP16 TC | Baseline. Largest weight footprint. |
| BF16 (no quant) | BF16 | BF16 TC | BF16 TC | Same TC cost as FP16, larger numerical range. |
| GPTQ INT4 | INT4 + group scales | Marlin → FP16 TC | Marlin → FP16 TC | Layer-wise Hessian-based. Mature. |
| AWQ INT4 | INT4 + scales | Marlin → FP16 TC | Marlin → FP16 TC | Activation-aware salience scaling. |
| AutoRound INT4 ⭐ | INT4 + scales | Marlin → FP16 TC | Marlin → FP16 TC | Intel's signed-gradient quant. Strongest on Qwen-family per @Lorbus's recipes. |
| AutoRound INT8 | INT8 + scales | INT8 TC | INT8 TC | Less common; usually only for serving INT8 directly. |
| NF4 (NormalFloat4) | 4-bit float (normal-distribution optimized) | bitsandbytes kernels → FP16 TC | same | bitsandbytes / QLoRA path. Better accuracy than uniform INT4 on normally-distributed weights. Mostly used for **fine-tuning**, not inference. |
| SmoothQuant W8A8 | INT8 weights + INT8 activations + per-channel scales | INT8 TC (full path) | INT8/FP8 TC | Smooths activation outliers via per-channel scaling, enabling W8A8. **Weight + activation quant** — needs strong INT8/FP8 hardware. |
| FP8 weights (FBGEMM / INC) | FP8 | 🟡 SW → FP16 TC | FP8 TC | Native fast path only on Ada/Hopper/Blackwell. |
| MXFP8 weights | FP8 + per-32-element E8M0 block scale | ✗ no native path | ✗ no native path | OCP microscaling standard. Native on Blackwell only. |
| MXFP4 weights | FP4 + per-32-element E8M0 block scale | ✗ | ✗ | Blackwell-native. OCP community format. |
| NVFP4 weights | E2M1 (4-bit) + per-16-element E4M3 scale + optional global scale | ✗ | ✗ | NVIDIA-proprietary 4-bit format. Native on Blackwell (sm_10x + sm_120). Smaller scaling blocks than MXFP4 → typically better accuracy. |
| GGUF (Q4_0 / Q4_K_M / Q5_K_M / IQ4_XS / Q8_0 / …) | 4-8 bit mixed, llama.cpp's grouped/importance-aware variants | llama.cpp's own kernels → FP16 TC | same | Runs on any arch with FP16 TC support. K-quants use grouped blocks; IQ-quants are importance-aware. |
| HQQ / AQLM / SqueezeLLM | 2-4 bit, codebook or outlier-aware | software kernels | software kernels | Niche / research-grade. Less universal hardware path. |

**Key takeaway for Ampere users**: INT4 *weight* quant (AutoRound / AWQ / GPTQ) compute paths all converge through Marlin → FP16 TC on Ampere. The choice of *which* INT4 scheme is about **calibration quality**, not hardware acceleration. NF4, NVFP4, and MXFP4 require newer hardware (NF4 = bitsandbytes software, NVFP4/MXFP4 = Blackwell-only).

### Weight-only vs weight + activation quantization

This is a second axis on top of the storage format — *what* gets quantized:

| Axis | Weight-only (W*A16 / W*A8) | Weight + activation (W8A8 / W4A8) |
|---|---|---|
| What's quantized | Just the weights, activations stay FP16/BF16 | Both weights and activations are low-precision |
| Hardware needed | Any Tensor-Core-capable arch (Marlin dequants → FP16 compute) | Strong INT8/FP8 TCs needed for full speed |
| Best on | Ampere+ (universal) | Hopper+ (FP8) or Ada (FP8 with caveats) |
| Examples | AutoRound INT4, AWQ, GPTQ, NF4, FP8 W8A16 | SmoothQuant W8A8, INT8 W8A8, FP8 W8A8, NVFP4 W4A4 |
| Trade | Easier — broad compat. Activation tensors stay full-precision so accuracy is mostly preserved. | Harder — activations are harder to quantize cleanly (outliers). Bigger throughput wins when it works. |

For a 24 GB consumer-Ampere rig (this stack's primary target), weight-only INT4 (AutoRound / AWQ / GPTQ) is the sweet spot: best accuracy retention, no special hardware needed. Weight+activation paths are mostly a Hopper/Blackwell-and-up game today.

---

## KV-cache dtype support

KV cache is a separate concern from weights — vLLM ships several KV-quant schemes, most of which are software-only and work on any arch where vLLM runs.

| KV format | Implementation | Native HW? | Works on Ampere? | Notes |
|---|---|:--:|:--:|---|
| FP16 / BF16 | Standard cast | ✓ (FP16/BF16 TC) | ✓ | Baseline. Largest KV pool footprint. |
| FP8 (E4M3 / E5M2) | **Storage-only on all consumer GPUs** (stored 1 B/tok, dequant→BF16 for attention). Native FP8 *attention* compute only on Hopper (FA3) / datacenter Blackwell (trtllm-gen) | consumer: storage-only; Hopper/DC-Blackwell: compute | ✓ (storage) | Half the bytes/token vs FP16. The default in `dual/autoround-int4/fp8-mtp.yml`. e4m3≡e5m2 in speed on consumer cards (precision vs range only). |
| INT8 PTH (per-token-head) | Custom kernel with per-(token, head) scales | ✓ (INT8 TC) | ✓ | High single-stream TPS; **doesn't scale at concurrency** (see [FAQ](FAQ.md#int8-pth-gives-me-150-tps-single-stream-but-doesnt-scale-with-concurrency--is-that-a-bug)). **Native in stock vLLM ≥ v0.22.0 for uniform-head-dim models** (Qwen etc.) — no overlay; Gemma-4 needs the #40391 overlay (see "KV-quant × checkpoint compatibility" below). |
| TurboQuant 3-bit (TQ3) | Custom Triton kernels | ✗ (Triton soft) | ✓ | ~5× the KV pool of FP16. Hybrid-attention models (Qwen3-Next) need a multi-query verify kernel for spec-decode (only Genesis P67 today). |
| TurboQuant 4-bit (TQ4) | Custom Triton kernels | ✗ | ✓ | ~4× the KV pool of FP16. Slightly higher quality than TQ3. |
| TurboQuant k8v4 | 8-bit K + 4-bit V mixed | ✗ | ✓ | Asymmetric — K matters more for attention precision. BF16-equivalent quality per the TQ paper. |

**Only Hopper (FA3) and datacenter Blackwell (trtllm-gen) get a compute win on FP8 KV** — those are the only two backends that run the *attention matmul* in FP8. Everywhere else — Ampere, **and every consumer card including Ada 4090, the 5090, and the DGX Spark** — FP8 KV is **storage-only**: it saves VRAM (bigger pool / longer context) but the attention matmul still upcasts to BF16, so there's no per-stream speedup and no `e4m3`-vs-`e5m2` throughput difference (see the two-axes section above). On such cards, if you want a KV format that actually multiplies faster on the Tensor Core, that's **INT8-PTH** (native INT8 TC), not FP8.

**The launchers inject `KV_CACHE_DTYPE=fp8_e4m3` on sm_89+ since [#246](https://github.com/noonghunna/club-3090/issues/246) Phase 1** (pilot slugs `vllm/dual`/`vllm/minimal`; Ampere keeps its compose default; explicit env wins; quant-specific KV slugs untouched). **NOTE the reframe from the first cross-rig data:** e4m3 is *not* a throughput win on consumer cards (it's storage-only there, ≡ e5m2 — see above). It's still the right default on sm_89+ as the **better-precision** FP8 KV format at zero speed cost — but the #246 "e4m3 for native FP8 compute" hypothesis only holds on Hopper / datacenter Blackwell, which the community doesn't run.

**Emerging KV recipes** (worth watching, not yet defaults):
- **Block-scaled FP8 KV** — per-block scales like MXFP8 but applied to KV cache rather than weights. Recovers accuracy at long-context where flat FP8 can lose precision in deep layers. Lands on Hopper / Blackwell first.
- **NVFP4 KV** — **DATACENTER Blackwell only (sm_100/sm_103: B100/B200/GB200).** It forces vLLM's trtllm-gen FP4 FMHA, which has **no consumer-Blackwell (sm_120/121) build** — so it **crashes on RTX 5090s** despite their FP4 hardware ([vLLM #43562](https://github.com/vllm-project/vllm/issues/43562) / [TRT-LLM #10241](https://github.com/NVIDIA/TensorRT-LLM/issues/10241); empirically hit on two 5090s via #246, disc #571). NVFP4 *weights* run on consumer Blackwell — only the KV/attention path doesn't. Consumer-Blackwell FP4-era KV = **fp8_e4m3**. (A textbook 'literal exists ≠ kernel exists' — the exact trap the #246 A/B was built to catch.)
- **TensorRT-LLM W4A8 / W4A4** — Hopper/Blackwell weight+activation quant recipes that pair INT4/NVFP4 weights with FP8/FP4 activations. Out of scope for the vLLM-first composes here but worth knowing if you cross-shop.

### KV-quant × checkpoint compatibility — the two Ampere traps

Beyond "does the kernel exist", two non-obvious rules decide which KV dtype you can actually *use*:

1. **fp8 KV is rejected for *compressed-tensors* checkpoints.** `--kv-cache-dtype fp8_e5m2` on an AWQ / FP8-weight / INT8-weight checkpoint loaded via the **compressed-tensors** path dies with `ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints` — and it fires whether `--quantization compressed-tensors` is explicit *or* auto-detected (the guard keys off the *detected* method, not the flag). It is **not** caused by a real fp8 weight component — a pure-int4 AWQ (e.g. `cyankiwi/...-AWQ-BF16-INT4`, which has zero fp8 in its config) trips it too. The **`auto_round` / GPTQ loaders are *not* affected** — that's why `vllm/dual` (AutoRound) runs fp8 KV fine. **So for a compressed-tensors checkpoint, the 1-byte-per-token KV path is INT8-PTH, not fp8.**

2. **INT8-PTH is native in stock vLLM (≥ v0.22.0) — but only for *uniform-head-dim* models.** `--kv-cache-dtype int8_per_token_head` works out-of-the-box on Qwen and other standard-attention models (defined in `config/cache.py` + `v1/kv_cache_interface.py`; **no overlay**). The catch is **heterogeneous head dims**: Gemma-4 interleaves head_dim 256 (sliding) / 512 (global), so the per-token-head page sizes (520 B vs 1032 B) don't satisfy vLLM's `unify_kv_cache_spec_page_size()` and the engine refuses to boot. The vendored **#40391** overlay pads global layers to a 1040-B factor to fix *only that*. So #40391 is **Gemma-4-specific — do not copy it to other models**, and don't assume INT8-PTH needs a patch in general.

**Putting it together — picking a KV dtype on Ampere (sm_86):**

| Your checkpoint / goal | KV dtype | Overlay? |
|---|---|---|
| `auto_round` / GPTQ weights | `fp8_e5m2` | none |
| compressed-tensors (AWQ / FP8 / INT8 weights), **long context** | `int8_per_token_head` | none — **except Gemma-4: +#40391** |
| anything, **context ≤ ~half the model-max** | `bf16` (default) | none |

The "INT8-PTH needs no overlay" point was upstreamed in v0.22.0 and is easy to miss — **engine capability flags that predate it can be wrong; verify against the running image**, not the profile. (The reason most people never hit any of this: they run **bf16 KV at the auto-fit context**, well under the model max — the int8-PTH/overlay question only arises when you stack *max context* AND *a compressed-tensors checkpoint*.)

---

# Intel

Intel's Tensor-Core equivalent is **XMX (Xe Matrix Extensions)** on Xe-architecture GPUs. The lineage from old to new:

- **Xe-HPG (Arc A-series — Alchemist, 2023)** — first consumer XMX. INT8/INT4/FP16/BF16 supported. No FP8.
- **Xe-HPC / Ponte Vecchio (Flex Series, Max 1100/1550, datacenter)** — high-density XMX. INT8/INT4/FP16/BF16. No FP8 in original silicon.
- **Xe2 (Arc B-series — Battlemage, Lunar Lake iGPU, 2025)** — XMX gen-2. Adds **INT2** support, ~2× INT8 throughput vs Alchemist, partial FP8 emerging. MXFP4 / microscaling formats landing via OpenVINO.

## Intel XMX feature matrix

| Format | Xe-HPG (Arc A) | Xe-HPC (Flex/Max DC) | Xe2 (Arc B / Lunar Lake iGPU) |
|---|:--:|:--:|:--:|
| FP32 (vector) | ✓ | ✓ | ✓ |
| FP16 (XMX) | ✓ | ✓ | ✓ |
| BF16 (XMX) | ✓ | ✓ | ✓ |
| INT8 (XMX) | ✓ | ✓ | ✓ (~2× HPG throughput) |
| INT4 (XMX) | ✓ | ✓ | ✓ |
| INT2 (XMX) | ✗ | ✗ | ✓ |
| FP8 E4M3 / E5M2 | ✗ | ✗ | 🟡 partial (silicon support; production stacks maturing) |
| MXFP4 / microscaling | ✗ | ✗ | 🟡 emerging via OpenVINO |
| FP4 / NVFP4 | ✗ | ✗ | ✗ |

**FP8 maturity caveat (mid-2026)**: Xe2 silicon supports FP8, but **full FP8 acceleration in production inference stacks (vLLM-SYCL, IPEX-LLM)** is still being plumbed through. OpenVINO has the most coverage today; expect 6-12 month lag behind NVIDIA's transformer-engine path for equivalent throughput. Track [oneAPI release notes](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) for kernel landing dates.

## Intel KV cache options

| KV format | Xe-HPG (Arc A) | Xe-HPC (DC) | Xe2 (Arc B) | Stack |
|---|:--:|:--:|:--:|---|
| FP16 / BF16 | ✓ | ✓ | ✓ | universal — IPEX-LLM / OpenVINO default |
| INT8 via XMX | ✓ | ✓ | ✓ | IPEX-LLM has the most coverage |
| FP8 | ✗ | ✗ | 🟡 emerging | not production-ready in vLLM-SYCL as of 2026-05 |
| llama.cpp `q4_0` / `q5_0` / `q8_0` | ✓ | ✓ | ✓ | SYCL backend — most portable path on Intel GPUs today |

## What this means for LLM serving on Intel

| GPU class | Best path today | Stack |
|---|---|---|
| Arc A-series (A770 16 GB / A580 / A380) | GPTQ/AWQ INT4 + FP16 compute | llama.cpp via SYCL backend; IPEX-LLM for transformer-style serving; OpenVINO for batch-low-latency |
| Arc B-series (Battlemage B580 / B570) | Same as A + INT4/INT8 throughput uplift; FP8 paths landing | Same stacks, Xe2-tuned kernels |
| Max Series / Flex (data center) | INT8 with sparsity; some BF16 / FP16 mixed | OpenVINO + oneAPI |

**Key strengths**: cost-effective low-bit inference (INT4/INT8). Arc A770 16 GB has been a popular budget cross-vendor option for small LLMs via llama.cpp.

**Key weaknesses**: ecosystem maturity. FP8 kernels are still landing; vLLM has a SYCL backend but it's significantly less mature than the CUDA path; advanced quants like NVFP4 / MXFP4 are mostly research at this point.

**For club-3090 composes**: Intel users today go through **llama.cpp** (`ghcr.io/ggml-org/llama.cpp:server-intel` image or build with SYCL). The KV-quant menu shrinks to llama.cpp's offerings (`q4_0`, `q5_0`, `q8_0`); spec-decode menu shrinks to llama.cpp's MTP PR and ngram.

---

# AMD

AMD splits matrix acceleration across two distinct architecture lines:

- **RDNA (Radeon consumer)** — uses **WMMA** (Wave Matrix Multiply-Accumulate) instructions
- **CDNA (Instinct datacenter)** — uses **MFMA** (Matrix Fused Multiply-Add) — much denser; the actual AMD AI silicon

The two are *not* the same hardware path and have very different LLM-serving characteristics.

## RDNA (Radeon consumer) — WMMA

| Format | RDNA 2 (RX 6000) | RDNA 3 (RX 7000) | RDNA 4 (RX 8000 / RX 9000) |
|---|:--:|:--:|:--:|
| FP32 (vector) | ✓ | ✓ | ✓ |
| FP16 (WMMA) | — | ✓ | ✓ |
| BF16 (WMMA) | — | ✓ | ✓ |
| INT8 (WMMA) | — | ✓ | ✓ |
| INT4 (WMMA) | — | ✓ | ✓ |
| FP8 E4M3 / E5M2 (WMMA) | — | 🟡 limited | ✓ native |
| 2:4 structured sparsity | — | — | ✓ |

**RDNA 3 (RX 7900 XTX / 7900 XT / 7800 XT / 7700 XT)** — first WMMA generation. FP16/BF16/INT8/INT4 work; FP8 path is limited and depends on kernel availability.

**RDNA 4 (RX 8000 / RX 9000 series, 2025+)** — WMMA gen-3. Early roadmaps used the RX 8000 name; AMD officially shipped under the RX 9000 brand. Adds **native FP8** (E4M3 + E5M2), **2:4 structured sparsity** (effectively doubling peak TFLOPS on supported kernels), and improved INT4/INT8 throughput. The first consumer Radeon arch that's seriously competitive for LLM inference.

## CDNA (Instinct datacenter) — MFMA

| Format | CDNA 2 (MI250/MI210) | CDNA 3 (MI300X/MI300A) | CDNA 4 (MI350/MI400) |
|---|:--:|:--:|:--:|
| FP16 (MFMA) | ✓ | ✓ | ✓ |
| BF16 (MFMA) | ✓ | ✓ | ✓ |
| INT8 (MFMA) | ✓ | ✓ | ✓ |
| INT4 (MFMA) | ✓ | ✓ | ✓ |
| FP8 E4M3 / E5M2 (MFMA) | ✗ | ✓ | ✓ |
| FP6 (MFMA) | ✗ | ✗ | ✓ |
| FP4 / MXFP4 / NVFP4-style (MFMA) | ✗ | ✗ | ✓ |
| MXFP8 (MFMA) | ✗ | ✗ | ✓ |
| 2:4 structured sparsity | ✗ | ✓ | ✓ |

**CDNA 3 (MI300X)** — AMD's Hopper competitor. 192 GB HBM3 per card, FP8 MFMA. Massive memory + competitive FP8 throughput; the option to look at if you want to escape NVIDIA's data-center pricing.

**CDNA 4 (MI350 / MI400)** — AMD's Blackwell competitor. Adds FP6 / FP4 / MXFP* native paths. **Note: NVFP4 specifically is NVIDIA-proprietary (E2M1 data + per-16-element E4M3 scaling)** and AMD doesn't support it. What AMD supports on CDNA 4 are the *OCP-standard* block-scaled FP4/FP6 formats (per-32-element E8M0 scales — MXFP4 / MXFP6) plus AMD-defined variants. For cross-vendor portability the safer 4-bit target is MXFP4; NVFP4 is a NVIDIA-only optimization.

## AMD KV cache options

| KV format | RDNA 3 (RX 7000) | RDNA 4 (RX 8000/9000) | CDNA 2 (MI200) | CDNA 3 (MI300) | CDNA 4 (MI350+) | Stack |
|---|:--:|:--:|:--:|:--:|:--:|---|
| FP16 / BF16 | ✓ | ✓ | ✓ | ✓ | ✓ | universal — vLLM ROCm / llama.cpp |
| FP8 | ✗ | ✓ native | ✗ | ✓ native | ✓ native | vLLM ROCm on CDNA 3+ / RDNA 4 |
| INT8 | ✓ | ✓ | ✓ | ✓ | ✓ | quant scheme dependent |
| MXFP* block-scaled | ✗ | ✗ | ✗ | 🟡 emerging | ✓ | AMD's NVFP4 equivalent; Composable Kernel landing |
| llama.cpp `q4_0` / `q5_0` / `q8_0` | ✓ | ✓ | ✓ | ✓ | ✓ | ROCm backend — most portable on consumer Radeon |

## What this means for LLM serving on AMD

| GPU class | Best path today | Stack |
|---|---|---|
| RDNA 3 (RX 7900 XTX 24 GB) | GPTQ/AWQ INT4 + FP16 compute via ROCm | llama.cpp ROCm image (`ghcr.io/ggml-org/llama.cpp:server-rocm`); vLLM ROCm (improving) |
| RDNA 4 (RX 9000) | Add FP8 paths + 2:4 sparsity | Same stacks, RDNA4-tuned kernels landing in 2025-26 |
| CDNA 3 (MI300X) | **FP8 weights + FP8 KV** | vLLM ROCm fork; AMD's Composable Kernel; production-grade |
| CDNA 4 (MI350+) | FP4 / MXFP4 / FP6 paths emerging | vLLM ROCm + AMD ML libs |

**Key strengths**: huge HBM on Instinct (192 GB on MI300X — a single card holds models that need TP=4 on H100s); RDNA 4 brings consumer FP8 to AMD for the first time.

**Key weaknesses**: software fragmentation between ROCm (datacenter-quality) and consumer Radeon drivers; fewer production-grade quant kernels than NVIDIA's ecosystem; vLLM's ROCm backend is real but trails CUDA in feature parity (no DFlash on AMD, fewer KV-quant formats, etc.).

**For club-3090 composes**: AMD users today go through **llama.cpp** for consumer Radeon (`ghcr.io/ggml-org/llama.cpp:server-rocm`), or **vLLM ROCm** for Instinct rigs. If you have an MI300X, the relevant comparisons are against H100, not against this stack's 3090 baseline.

---

# Cross-vendor routing

When future composes want to detect the host GPU and pick an optimal path, here's the rough decision tree:

```
1. Detect vendor + arch
   - NVIDIA: torch.cuda.is_available() and torch.version.hip is None
             → torch.cuda.get_device_capability()  →  (major, minor)
   - AMD:    torch.cuda.is_available() and torch.version.hip is not None
             → torch.cuda.get_device_properties(0).gcnArchName  →  e.g. "gfx1100" (RDNA3)
             → confirm via `rocm-smi --showproductname` or env $ROCM_PATH
             ⚠ HIP-via-PyTorch appears as `cuda` — distinguish via torch.version.hip, NOT torch.cuda
   - Intel:  torch.xpu.is_available()  →  sycl device query, parse arch via clinfo or sycl-ls
   - Apple:  torch.backends.mps.is_available()  (out of scope for these composes)

2. Map to capability tier
   - NVIDIA sm_120 / sm_10x   → Blackwell tier (FP4 / NVFP4 / MXFP* available)
   - NVIDIA sm_90              → Hopper tier (FP8 + transformer engine)
   - NVIDIA sm_89              → Ada tier (FP8 native, thinner kernel set)
   - NVIDIA sm_80 / sm_86      → Ampere tier (BF16/INT8, FP8 emulated)
   - AMD CDNA 4                → MI350+ tier (FP4/FP6/MXFP*)
   - AMD CDNA 3                → MI300X tier (FP8 + huge HBM)
   - AMD RDNA 4                → RX 9000 tier (FP8 consumer)
   - AMD RDNA 3                → RX 7000 tier (INT8/INT4 only)
   - Intel Xe2                 → Battlemage tier (INT2/INT4/INT8 + emerging FP8)
   - Intel Xe-HPG              → Alchemist tier (INT4/INT8/FP16)

3. Pick KV + weight quant per tier
   - Blackwell  → NVFP4 weights (when artifact exists) or FP8 weights + FP8 KV
   - Hopper     → FP8 weights + FP8 KV
   - Ada        → AutoRound INT4 weights + FP8 KV
   - Ampere     → AutoRound INT4 weights + TQ3 / INT8 PTH / fp8 KV
   - CDNA 4     → MXFP4 / FP8 weights + FP8 KV
   - CDNA 3     → FP8 weights + FP8 KV
   - RDNA 4     → AWQ/GPTQ INT4 + FP8 KV
   - RDNA 3     → AWQ/GPTQ INT4 + FP16 KV (no FP8)
   - Xe2        → AWQ/GPTQ INT4 + FP16 KV (FP8 in progress)
   - Xe-HPG     → AWQ/GPTQ INT4 + FP16 KV

4. Pick engine
   - vLLM-CUDA      for NVIDIA Ampere+
   - vLLM-ROCm      for AMD Instinct (CDNA)
   - llama.cpp-ROCm for AMD consumer (RDNA)
   - llama.cpp-SYCL for Intel
   - llama.cpp-CPU  fallback (anywhere)
```

This stack codifies the NVIDIA half of that tree today via `scripts/switch.sh` + the `models/<model>/<engine>/compose/` tree. Cross-vendor routing is a future-direction item — happy to entertain PRs that add `compose/intel/` or `compose/amd/` paths once we have benchmark numbers from those rigs to ground them in.

**Ecosystem maturity caveat (mid-2026)**: NVIDIA's stack (vLLM + Marlin + Genesis + TensorRT-LLM + cuDNN) has 2-3 years more polish than AMD ROCm or Intel oneAPI. Expect day-zero kernel coverage on NVIDIA for new model architectures, and 1-3 month lag on AMD/Intel for the same models to reach equivalent throughput. For mission-critical inference, NVIDIA stays the default; for cost-sensitive or vendor-diversification scenarios, the Instinct (MI300X+) and Battlemage paths are increasingly viable.

---

## Per-arch recommendations for club-3090 composes

What to ship as the default for each GPU class, given the matrix above:

| GPU class | Best weight quant | Best KV dtype | Best spec-decode | Compose pattern |
|---|---|---|---|---|
| **Pascal (10x0)** | FP16 only — no INT8 TC | FP16 | none (vLLM unsupported, CC<7.5) | llama.cpp only |
| **Volta (V100)** | FP16 — no BF16 TC, weak INT8 | FP16 | none | llama.cpp — see [@efschu's V100 bench](../BENCHMARKS.md#qwen36-27b) |
| **Turing (T4, 20-series)** | INT8 native; GPTQ INT4 via Marlin works | FP16 / INT8 | n-gram only | not a primary target |
| **Ampere consumer (3090/3080/3060)** ⭐ | **AutoRound INT4** | **TQ3 (Genesis) / INT8 PTH / fp8** | **MTP n=3** + Genesis P67 | `dual/autoround-int4/turbo.yml`, `dual/autoround-int4/tq3-mtp-genesis.yml`, `single/autoround-int4/long-text.yml` — primary target of this stack |
| **Ampere DC (A100)** | AutoRound INT4 or FP16 | TQ3 / fp8 | MTP n=3 | Same composes as 3090, more VRAM headroom |
| **Ada (4090 / L40)** | AutoRound INT4 **OR** FP8 weights (native FP8 GEMM on sm_89) | **INT8-PTH** (native INT8 TC) / TQ3 for a compute win; `fp8_e4m3` = storage-only (≡e5m2, launcher-injected #246, precision not speed) | MTP n=3 / DFlash | FP8 KV here is **storage-only** — no FA3 (Hopper-only); the arch win is FP8 *weights*, not FP8 KV |
| **Hopper (H100/H200)** | FP8 weights (FBGEMM/INC) | **fp8 (HW transformer engine)** | MTP / DFlash | Not a primary target — these cards are usually already running their own optimised stacks |
| **Blackwell consumer (5090 / PRO 6000 Blackwell / DGX Spark, sm_12x)** | AutoRound INT4 or **NVFP4 *weights*** (native FP4 GEMM) | **fp8_e4m3** = storage-only (≡e5m2; launcher-injected #246 for precision). **`nvfp4` KV does NOT work** (datacenter-only trtllm-gen FMHA, [#43562](https://github.com/vllm-project/vllm/issues/43562)). Compute-win KV = **INT8-PTH / TQ3** | MTP n=3 | **Measured (disc #571): e4m3≡e5m2, no KV-dtype perf lever.** The arch win is raw silicon + FP4 *weights*, not KV. DGX Spark (sm_121) = same family, plus low-bandwidth → favor KV *compression* for capacity |
| **Blackwell DC (B100/B200/GB200)** | NVFP4 / FP8 weights | NVFP4 / fp8 | MTP / DFlash | Out of scope for club-3090 |

The starred row (Ampere consumer) is the actual target of this stack; everything else is "should work, here's the data point we have".

---

## How to detect your GPU's capabilities at runtime

The Genesis patch stack already encodes most of this — the guard functions in `models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/guards.py` give you the runtime answer:

```python
import vllm._genesis.guards as g

g.get_compute_capability()    # → (8, 6) on 3090, (8, 9) on 4090, (9, 0) on H100, (12, 0) on 5090, (10, 0) on B200
g.is_ampere_consumer()         # 3090/3080/...
g.is_ampere_datacenter()       # A100
g.is_ada_lovelace()            # 4090/L40
g.is_hopper()                  # H100/H200 — exactly sm_90
g.is_blackwell()               # any sm_10x or sm_120
g.is_blackwell_datacenter()    # B100/B200/GB200 — sm_10x
g.is_blackwell_consumer()      # 5090/5080/5070 — sm_120
g.has_native_fp8()             # True on Ada+ (sm_89+)
```

For the same info outside Python, `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` returns `8.6` / `8.9` / `9.0` / `12.0` etc.

---

## Notes on the corner cases

**FP8 on Ada vs Hopper.** Both have FP8 Tensor Cores. The numerical formats (E4M3 + E5M2) are the same. The difference is peak throughput: Hopper's transformer engine has higher FP8 TFLOPS/clock and on-die format-conversion units that Ada doesn't. For Qwen3.6-27B inference, the practical difference between Ada and Hopper FP8 KV is in the single-digit-% range — both are real hardware paths, both meaningfully faster than Ampere's software emulation.

**Blackwell consumer vs Blackwell datacenter.** Genesis treats them as one family for has_native_fp8 / has_native_fp4 purposes, but different regimes for kernel tuning. Per club-3090#51 (apnar 2026-05-04): fp8_e4m3 + 96K ctx on RTX 5090 regressed 2-6% TPS vs fp8_e5m2 + 48K — vLLM's Blackwell e4m3 codepath is newly added and undertuned. Use e5m2 on consumer Blackwell until the pin catches up. Sm_120 also surfaced a `CUDAGraphMode.FULL_AND_PIECEWISE not supported with spec-decode for FlashInferBackend` issue — Genesis ships P100 patch (PR #41127 backport) for this.

**NVFP4 on RTX 5090 today.** The hardware supports it. vLLM's NVFP4 quant kernels are landing in 2026-05/06 nightlies but aren't broadly used yet — most public 27-31B weight artifacts are still INT4 (AutoRound/AWQ/GPTQ), not NVFP4. Once NVFP4-quantized weight artifacts ship for Qwen3.6 / Gemma 4 / etc., Blackwell consumer rigs gain a meaningful TPS lift on top of FP8.

**NVFP4 vs MXFP4 — same bit-width, different block size.** Both are 4-bit float formats Blackwell accelerates natively. They differ in how often the scale changes:

| | NVFP4 (NVIDIA) | MXFP4 (OCP / community) |
|---|---|---|
| Element format | E2M1 (4-bit) | E2M1 (4-bit) |
| Scale block size | **16 elements** | **32 elements** |
| Scale format | E4M3 (8-bit) per block + optional global FP32 | E8M0 (8-bit exponent only) per block |
| Implication | Finer-grained scaling → typically better accuracy | Coarser scaling → slightly larger throughput, slightly worse accuracy |
| Vendor support | NVIDIA-proprietary, Blackwell-native | Open standard from OCP — supported across Blackwell + future AMD MI3xx/MI4xx |

For inference workloads, NVFP4 is usually preferred when both are available. MXFP4 wins on cross-vendor portability.

**MXFP* (microscaling) family.** The OCP microscaling standard adds **per-32-element block scales** (E8M0 = 8-bit power-of-two scale) on top of FP4 / FP6 / FP8 element formats: **MXFP4 / MXFP6 / MXFP8**. Native on Blackwell Tensor Cores; emulation possible on older arches but slow. The block-scale trick lets you keep low-bit element storage while recovering most of the dynamic range you'd lose with plain low-bit floats — so MXFP* formats typically score **measurably better on perplexity / downstream accuracy than plain low-bit floats at the same bit width**, which is exactly why Blackwell hardware leans into them. Expect MXFP8 to be the new "FP8 with better accuracy" default once kernel paths mature.

**FP6 on Blackwell.** Blackwell adds native FP6 Tensor Cores — same theoretical throughput as FP8, smaller storage. Useful middle-ground between FP8 (highest accuracy at low precision) and FP4 (highest density). Kernel ecosystem is still maturing; not yet a common production choice for LLM serving as of 2026-05.

**INT4 Tensor Cores.** All Turing+ archs have INT4 TCs in hardware — but they're almost never used as compute targets. Most "INT4 weights" workloads (AWQ/GPTQ/AutoRound) dequant the INT4 weights to FP16 on the fly and use the FP16 TCs. Marlin is the canonical kernel for this; Machete is a newer alternative. Direct INT4 TC compute is mainly a special-case path inside CUTLASS / cuBLASLt and not how most LLM inference works today.

**INT8 KV PTH anti-scaling.** Per-token-head INT8 stores a separate scale per (token, head) pair. Single-stream is fast (the scale lookup is cheap), but at concurrency the per-(token, head) dequant becomes a serialization point. fp8 has a single global scale and scales near-linearly. See [FAQ](FAQ.md#int8-pth-gives-me-150-tps-single-stream-but-doesnt-scale-with-concurrency--is-that-a-bug).

---

## References

**NVIDIA**:
- [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) — official mapping of sm_XY to features
- [Ampere whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) — first BF16/TF32/INT8 TC mainstream
- [Hopper whitepaper](https://www.nvidia.com/en-us/data-center/h100/) — transformer engine, FP8
- [Blackwell announcement](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) — FP4/NVFP4
- [Marlin](https://github.com/IST-DASLab/marlin) — INT4 weight × FP16 compute kernel; the dominant Ampere INT4 path

**Intel**:
- [oneAPI docs](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) — SYCL/DPC++ programming model
- [XMX (Xe Matrix Extensions) overview](https://www.intel.com/content/www/us/en/developer/articles/technical/xe-matrix-extensions.html) — the matrix-compute path on Xe GPUs
- [IPEX-LLM](https://github.com/intel-analytics/ipex-llm) — Intel's LLM-on-Xe runtime
- [OpenVINO](https://docs.openvino.ai/) — Intel's broader inference toolkit with INT4/INT8 quant support

**AMD**:
- [ROCm docs](https://rocm.docs.amd.com/) — primary entry point for AMD GPU compute
- [WMMA on RDNA](https://gpuopen.com/learn/wmma_on_rdna3/) — consumer-Radeon matrix instructions
- [MFMA / Matrix Cores on CDNA](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/) — datacenter Instinct matrix path
- [Composable Kernel](https://github.com/ROCm/composable_kernel) — AMD's high-perf kernel framework
- [vLLM ROCm fork](https://github.com/ROCm/vllm) — ROCm-specific vLLM patches

**Cross-vendor**:
- [vLLM kernel/dtype matrices](https://docs.vllm.ai/) — vLLM source `vllm/model_executor/layers/quantization/` is the source of truth for which schemes have working kernels per arch
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — most portable inference engine; CUDA, ROCm, SYCL, Metal, CPU backends
- [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) — community standard for MXFP4/6/8

**This repo**:
- **Genesis guards** — `models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/guards.py` encodes the runtime per-arch feature detection
- **club-3090 cross-rig data** — [BENCHMARKS.md](../BENCHMARKS.md) measures these matrices in practice (3090 / 4090 / 5090 / V100 / A5000 / mixed-arch eGPU)
