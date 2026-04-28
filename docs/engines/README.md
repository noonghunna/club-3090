# Inference engines for Qwen3.6-27B — comparison + quick recipes

This repo's main path is **vLLM** because it has the deepest support for Qwen3-Next features (vision, MTP, TurboQuant, full OpenAI API parity). But the model also runs on **llama.cpp** and **SGLang** with different trade-offs. This page compares the three; per-engine pages have setup instructions.

> 🔁 **Coming from the README's Quick start?** It already shipped you the vLLM path. Skim this comparison to see what the alternatives look like, then pick a per-engine page if you want to try one.

---

## At a glance

| Engine | Status on this stack | Per-stream TPS (1× 3090) | Max ctx (1× 3090) | Vision | Tool calls | Spec-decode | OpenAI API parity |
|---|---|---|---|---|---|---|---|
| **[vLLM](VLLM.md)** ⭐ | **Validated, production-grade** (this repo) | 51-55 narr / 67-70 code | 48K safe (192K-205K opt-in with caveats) | ✅ | ✅ | ✅ MTP n=3 | ✅ Full |
| **[llama.cpp](LLAMA_CPP.md)** | Works mainline + [Luce DFlash fork](https://github.com/luce-spec/llama-cpp-dflash) for spec-decode | 35-60 (varies by quant + KV type) | **262K** (Q4_K_M + q4_0 KV) | ✅ (via mmproj) | ⚠️ Limited (no auto-tool-choice in server) | ✅ DFlash N=5 in fork | ⚠️ Partial |
| **[SGLang](SGLANG.md)** | **Blocked** by same Marlin pad-sub-tile-n bug (vllm#40361 / sglang equivalent); EAGLE spec-decode separately blocked by GDN/DeltaNet rollback | n/a (untested at this state) | n/a | ✅ | ✅ | ⚠️ EAGLE blocked on hybrid | ✅ Full |

---

## Pros / cons matrix

### vLLM ⭐

**Pros:**
- Deepest Qwen3-Next feature support upstream
- TurboQuant 3-bit KV cache (lets us reach 192K+ context on a single 3090)
- MTP speculative decoding works out of the box
- Genesis patch ecosystem (Sandermage's tree fixes many compatibility edges)
- Full OpenAI API parity (chat, vision, tools, streaming, reasoning, structured output)
- Active development — bugs we hit get triaged within days

**Cons:**
- Heavyweight — Docker image is ~9 GB
- Longer cold-start (~2 min for compile + cudagraph capture)
- Sensitive to upstream API drift across nightly versions (we pin to dev205 to avoid this)
- Frontier features sometimes ship with bugs we have to patch around (the whole reason this repo exists)

**When to pick:** Production / serious local work / anything that needs the full feature set.

---

### llama.cpp

**Pros:**
- Lightweight — single binary, ~50 MB
- Fastest cold-start (~30 sec)
- Lowest VRAM overhead (no inference framework taxes)
- GGUF support for many quant formats (Q4_K_M, Q5_K_S, IQ4_XS, etc.)
- Works on AMD + Intel + Apple Silicon (vLLM is NVIDIA-only)
- Active community, lots of distros / wrappers (Ollama, LM Studio, LocalAI, etc.)

**Cons:**
- Qwen3-Next family support is a moving target — needs the right binary build
- Server feature parity behind vLLM (no auto-tool-choice in upstream `server`; need wrapper)
- DFlash spec-decode requires a fork ([Luce's llama-cpp-dflash](https://github.com/luce-spec/llama-cpp-dflash))
- Concurrent serving is single-threaded by default (the server forks per request — sluggish under concurrent load)
- No TurboQuant equivalent → max usable context is much lower (~64K with Q4_K_M on 24 GB)

**When to pick:** Quick experiments, embedded use, non-NVIDIA hardware, when you want simplicity over feature completeness.

---

### SGLang

**Pros:**
- Designed for high-throughput serving — RadixAttention prefix sharing, structured-output-aware scheduling
- Often beats vLLM by 10-30% on multi-tenant throughput when both work
- First-class OpenAI API
- Good support for batched structured output (constraint decoding)

**Cons:**
- **Currently blocked on this stack by the same Marlin pad-sub-tile-n bug we hit on vLLM TP=2.** Same kernel-line fix applies (would need a similar patch on SGLang's side or for them to pick up the upstream fix).
- EAGLE spec-decode (their MTP equivalent) is separately blocked by the DeltaNet/GDN hybrid layer not supporting KV rollback — this is a Qwen3-Next architectural issue, not SGLang-specific.
- Smaller community than vLLM; fewer eyes on Qwen3-Next bugs.

**When to pick:** Production multi-tenant serving on models that work cleanly on it (not yet Qwen3.6-27B-int4-AutoRound — track the unblock list below).

**Watch list to unblock SGLang on this stack:**
- Marlin pad-sub-tile-n landing (we [filed PR #40361 on vLLM](https://github.com/vllm-project/vllm/pull/40361); the same fix applies to SGLang's Marlin call site)
- DeltaNet KV rollback support upstream (vllm#39931 / issue #40124 land would unblock EAGLE on Qwen3-Next family across engines)

---

## How to choose

| Your priority | Pick | Why |
|---|---|---|
| **Full feature set, MTP spec-decode, OpenAI API parity** | vLLM + Lorbus AutoRound | This repo's path. 51-70 TPS depending on workload, all features, prefill-safe at 48K default. |
| **Maximum context (262K) on one 3090** | llama.cpp + UD-Q3_K_XL or Q4_K_M + q4_0 KV | Smaller quants leave 8-10 GB headroom for KV at 262K. ~35-45 TPS sustained. |
| **Best concurrent throughput on dual 3090** | vLLM TP=2 + Turbo (TQ3) | 4 streams at full 262K, ~200 TPS aggregate. See [companion repo](https://github.com/noonghunna/qwen36-dual-3090). |
| **Non-NVIDIA hardware (AMD / Intel / Apple)** | llama.cpp | Only engine with cross-platform support. |
| **Lightest setup, fastest cold start** | llama.cpp | Single binary, ~30s cold start. Good for embedded use, quick experiments. |
| **High-throughput multi-tenant serving** | SGLang (when unblocked — currently blocked on Qwen3.6) | RadixAttention prefix sharing wins at scale. Watch list in SGLANG.md. |

---

## Quant choice (orthogonal to engine choice)

The model itself comes in several quant formats. Engine-quant compatibility:

| Quant | Disk size | Engine fit | Notes |
|---|---|---|---|
| **AutoRound int4** ([Lorbus](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound)) | ~18-19 GB | vLLM ✅ · llama.cpp ❌ · SGLang (when unblocked) | This repo's choice. W4A16, group_size=128, BF16 mtp.fc head. **Required** for vLLM's MTP spec-decode. |
| GPTQ int4 | ~16.5-17 GB | vLLM ✅ · llama.cpp ❌ · SGLang ✅ | Mature, broadly supported. Slightly smaller disk than AutoRound. |
| AWQ int4 | ~16-17 GB | vLLM ✅ · llama.cpp ❌ · SGLang ✅ | Strong baseline, compatible with Marlin kernels. |
| GGUF Q4_K_M | ~16.8 GB | llama.cpp ✅ · vLLM ⚠️ experimental · SGLang ❌ | The default GGUF mid-range quant. Strong quality, broad ecosystem (Ollama, LM Studio, etc). |
| GGUF UD-Q3_K_XL ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)) | **~14.5 GB** | llama.cpp ✅ | Smaller than 4-bit options. Quality cost is small on Qwen3.6 (quantization-friendly), buys substantial KV cache room. |
| GGUF Q3_K_M | ~13.6 GB | llama.cpp ✅ | More aggressive 3-bit; quality cost real but acceptable for many workloads. |

### AutoRound vs GPTQ vs AWQ (within vLLM)

All three are 4-bit weight-only quantization for vLLM. Differences:

| Aspect | AutoRound | GPTQ | AWQ |
|---|---|---|---|
| **Method** | Signed gradient descent jointly optimizing rounding + scaling | Layer-wise Hessian-based error minimization | Activation-aware salience scaling, then RTN |
| **Calibration set** | Small (~128-512 samples) | Larger (~1024-2048) | Small-medium |
| **Quantization time** | Minutes to ~1-2 hours for 27B | Slower for same model | Fast |
| **Accuracy at 4-bit** | Typically slightly best on hard reasoning (MMLU/GPQA/Math style) | Strong baseline; 0.5-2% behind AutoRound on average | Comparable to GPTQ; depends on tuning |
| **Ultra-low bits (3, 2)** | Strongest at <4 bit | Degrades faster below 4 bit | Middle of the pack |
| **Marlin kernel support** | ✅ (via the kernel-line fix in our [vllm#40361](https://github.com/vllm-project/vllm/pull/40361)) | ✅ (mature) | ✅ |
| **Ecosystem** | Newer, growing fast (Intel-maintained) | Most mature, broadest tool support | Strong vLLM/SGLang support |

**Why we picked AutoRound for this repo:** Lorbus's AutoRound quant ships `mtp.fc.weight` as BF16 (preserved at higher precision), which lets vLLM's `Qwen3_5MTP` loader actually load the head and run multi-token prediction at high acceptance rates (~80% per-position-1, AL ~3.5). GPTQ-quantized variants of the MTP head silently fail to load → 0% draft acceptance. So AutoRound isn't just "slightly better quality" here — it's the only path to working MTP spec-decode in vLLM today.

If MTP isn't a priority for your workload, GPTQ or AWQ are equally valid.

---

## Per-engine pages

- **[VLLM.md](VLLM.md)** — current setup (what this repo ships). Brief recap + tuning levers.
- **[LLAMA_CPP.md](LLAMA_CPP.md)** — quick GGUF recipe, vision via mmproj, Luce DFlash fork pointer for spec-decode, gotchas around server feature parity.
- **[SGLANG.md](SGLANG.md)** — current blocked state, what would unblock, when to revisit. TBD recipe placeholder until either Marlin pad lands upstream or DeltaNet rollback lands.

---

## See also

- [docs/INTERNALS.md](../INTERNALS.md) — why this repo picked vLLM specifically (the 9-probe forensics + upstream tracker)
- [docs/USE_CASES.md](../USE_CASES.md) — workload-specific configs for the vLLM path
- [LEARNINGS.md (parent stack)](https://github.com/noonghunna/qwen36-27b-single-3090/blob/master/docs/INTERNALS.md) — why vLLM, why these patches
