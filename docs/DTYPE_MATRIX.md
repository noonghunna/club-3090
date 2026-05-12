# Hardware dtype & quant capability matrix

What each NVIDIA GPU class accelerates *natively* in hardware vs *emulates* in software. Use this as the input when picking KV-cache dtype, weight quant, and kernel path for a compose targeting a specific rig.

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

- **Turing (sm_75)** — first to add **INT8/INT4 Tensor Cores**. T4 was the data-center workhorse for INT8 inference for years.
- **Ampere (sm_80/86)** — adds **BF16** and **TF32** TCs. Doubles INT8 throughput vs Turing. **No native FP8** (despite what some marketing suggested) — FP8 on Ampere is software-only and runs at ≤ fp16 speed.
- **Ada (sm_89)** — first consumer arch with **native FP8 Tensor Cores** (E4M3 + E5M2). The 4090 / L40 get a real FP8 path, though peak FP8 throughput is lower than Hopper's transformer engine.
- **Hopper (sm_90)** — first **transformer engine** with FP8 weights + FP8 KV both on-die. The fast path for FP8 inference. H100 also adds TMA (Tensor Memory Accelerator).
- **Blackwell (sm_10x DC / sm_120 consumer)** — adds **NVFP4 / FP4 / MXFP4** (4-bit float on TCs), **MXFP8**, and **FP6**. The first arch where 4-bit *compute* (not just storage) is a hardware feature. Genesis treats sm_10x and sm_120 as one family with the same FP8/FP4 hardware capabilities; performance per watt differs.

---

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
| FP8 (E4M3 / E5M2) | Software cast on Ampere; HW TC on Ada+ | partial | ✓ (SW) | Half the bytes/token vs FP16. The default in `dual/docker-compose.yml`. |
| INT8 PTH (per-token-head) | Custom kernel with per-(token, head) scales | ✓ (INT8 TC) | ✓ | High single-stream TPS; **doesn't scale at concurrency** (see [FAQ](FAQ.md#int8-pth-gives-me-150-tps-single-stream-but-doesnt-scale-with-concurrency--is-that-a-bug)). |
| TurboQuant 3-bit (TQ3) | Custom Triton kernels | ✗ (Triton soft) | ✓ | ~5× the KV pool of FP16. Hybrid-attention models (Qwen3-Next) need a multi-query verify kernel for spec-decode (only Genesis P67 today). |
| TurboQuant 4-bit (TQ4) | Custom Triton kernels | ✗ | ✓ | ~4× the KV pool of FP16. Slightly higher quality than TQ3. |
| TurboQuant k8v4 | 8-bit K + 4-bit V mixed | ✗ | ✓ | Asymmetric — K matters more for attention precision. BF16-equivalent quality per the TQ paper. |

**Ada / Hopper / Blackwell get a real win on FP8 KV** because the Tensor Cores can multiply FP8 directly without an upcast. On Ampere, FP8 KV is just smaller storage — the matmul still happens in FP16, so you save VRAM but not compute time. That's why the 3090's fp8 KV gives lower per-stream TPS than INT8 PTH (which *can* multiply in INT8 on the Tensor Core directly).

---

## Per-arch recommendations for club-3090 composes

What to ship as the default for each GPU class, given the matrix above:

| GPU class | Best weight quant | Best KV dtype | Best spec-decode | Compose pattern |
|---|---|---|---|---|
| **Pascal (10x0)** | FP16 only — no INT8 TC | FP16 | none (vLLM unsupported, CC<7.5) | llama.cpp only |
| **Volta (V100)** | FP16 — no BF16 TC, weak INT8 | FP16 | none | llama.cpp — see [@efschu's V100 bench](../BENCHMARKS.md#qwen36-27b) |
| **Turing (T4, 20-series)** | INT8 native; GPTQ INT4 via Marlin works | FP16 / INT8 | n-gram only | not a primary target |
| **Ampere consumer (3090/3080/3060)** ⭐ | **AutoRound INT4** | **TQ3 (Genesis) / INT8 PTH / fp8** | **MTP n=3** + Genesis P67 | `dual/turbo.yml`, `dual/tq3-mtp-genesis.yml`, `single/long-text.yml` — primary target of this stack |
| **Ampere DC (A100)** | AutoRound INT4 or FP16 | TQ3 / fp8 | MTP n=3 | Same composes as 3090, more VRAM headroom |
| **Ada (4090 / L40)** | AutoRound INT4 (preserve INT4 path) **OR** FP8 weights for full TC use | **fp8 (HW)** / TQ3 / INT8 PTH | MTP n=3 / DFlash | Same composes work; FP8 KV is now a real perf win |
| **Hopper (H100/H200)** | FP8 weights (FBGEMM/INC) | **fp8 (HW transformer engine)** | MTP / DFlash | Not a primary target — these cards are usually already running their own optimised stacks |
| **Blackwell consumer (5090/5080/5070)** | AutoRound INT4 today; NVFP4 when vLLM kernels mature | fp8_e5m2 advisory (e4m3 path undertuned per #51) | MTP n=3 | Genesis treats Blackwell consumer as a separate regime — some Hopper-targeted patches don't apply |
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

**MXFP* (microscaling) family.** The OCP microscaling standard adds **per-32-element block scales** (E8M0 = 8-bit power-of-two scale) on top of FP4 / FP6 / FP8 element formats: **MXFP4 / MXFP6 / MXFP8**. Native on Blackwell Tensor Cores; emulation possible on older arches but slow. The block-scale trick lets you keep low-bit element storage while recovering most of the dynamic range you'd lose with plain low-bit floats. Expect MXFP8 to be the new "FP8 with better accuracy" default once kernel paths mature.

**FP6 on Blackwell.** Blackwell adds native FP6 Tensor Cores — same theoretical throughput as FP8, smaller storage. Useful middle-ground between FP8 (highest accuracy at low precision) and FP4 (highest density). Kernel ecosystem is still maturing; not yet a common production choice for LLM serving as of 2026-05.

**INT4 Tensor Cores.** All Turing+ archs have INT4 TCs in hardware — but they're almost never used as compute targets. Most "INT4 weights" workloads (AWQ/GPTQ/AutoRound) dequant the INT4 weights to FP16 on the fly and use the FP16 TCs. Marlin is the canonical kernel for this; Machete is a newer alternative. Direct INT4 TC compute is mainly a special-case path inside CUTLASS / cuBLASLt and not how most LLM inference works today.

**INT8 KV PTH anti-scaling.** Per-token-head INT8 stores a separate scale per (token, head) pair. Single-stream is fast (the scale lookup is cheap), but at concurrency the per-(token, head) dequant becomes a serialization point. fp8 has a single global scale and scales near-linearly. See [FAQ](FAQ.md#int8-pth-gives-me-150-tps-single-stream-but-doesnt-scale-with-concurrency--is-that-a-bug).

---

## References

- **NVIDIA architecture docs**:
  - [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) — official mapping of sm_XY to features
  - [Ampere whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) — first BF16/TF32/INT8 TC mainstream
  - [Hopper whitepaper](https://www.nvidia.com/en-us/data-center/h100/) — transformer engine, FP8
  - [Blackwell announcement](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) — FP4/NVFP4
- **vLLM kernel/dtype matrices** — vLLM source `vllm/model_executor/layers/quantization/` is the source of truth for which schemes have working kernels per arch
- **Genesis guards** — `models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/guards.py` in this repo encodes the runtime per-arch feature detection used here
- **Marlin** — [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) — INT4 weight × FP16 compute kernel; the dominant Ampere INT4 path
- **club-3090 cross-rig data** — [BENCHMARKS.md](../BENCHMARKS.md) measures these matrices in practice (3090 / 4090 / 5090 / V100 / A5000 / mixed-arch eGPU)
