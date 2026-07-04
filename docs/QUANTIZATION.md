# Quantization — a field guide for the club-3090 community

Quantization is how a 27B model that would need ~54 GB at FP16 fits in 24 GB of VRAM. This page explains the **quant families** you'll see in the wild, what actually differs between them, and which ones this stack ships — including the **IQK imatrix quants** that exist only in [ik_llama.cpp](engines/IK_LLAMA.md).

> **The one idea to take away:** at the same *bits-per-weight*, not all quants are equal. The two levers that separate good from bad are (1) **non-linear levels** that match the weight distribution and (2) **calibration** (an "importance matrix") that spends bits where they matter. The best quants use both.

See also: [GLOSSARY.md](GLOSSARY.md) · [DTYPE_MATRIX.md](DTYPE_MATRIX.md) (KV/compute dtypes) · [engines/IK_LLAMA.md](engines/IK_LLAMA.md).

---

## 1. The vocabulary

- **bpw (bits per weight):** the headline number. FP16 = 16 bpw. A "4-bit" quant is ~4-4.5 bpw once you count the per-block scale/zero-point overhead. Lower bpw = smaller file = more context room, but more quality risk.
- **Block / group:** quants don't store one scale for the whole tensor — they chunk weights into blocks (e.g. 32 weights) and store a scale per block. Smaller blocks = finer = more accurate, but more overhead.
- **imatrix (importance matrix):** a calibration pass over real text that records *which weights matter most* for the model's outputs, so the quantizer protects those and compresses the rest harder. "i-quant" / "IQ" prefixes signal imatrix use.
- **Weight quant vs KV-cache quant:** two independent knobs. One shrinks the *model*; the other shrinks the *context* (see §5). You pick both.

---

## 2. The GGUF ladder (llama.cpp + ik_llama.cpp)

GGUF is the llama.cpp-family weight format. Roughly in order of quality-per-bit (worst → best at a given bpw):

| Family | Examples | Calibrated? | Where | Notes |
|---|---|---|---|---|
| **Legacy** | `Q4_0`, `Q4_1`, `Q5_0`, `Q8_0` | ❌ | mainline | Simple round-to-nearest, one scale/block. `Q8_0` is still a great near-lossless choice; the low-bit legacy ones are superseded. |
| **K-quants** | `Q3_K_M`, **`Q4_K_M`**, `Q5_K_M`, `Q6_K` | ❌ (data-free) | mainline | Mixed precision per tensor-type + 2-level block scales. The mainstream default. **`Q4_K_M` is what our shipped `llamacpp/mtp` runs.** Good, but data-free — no calibration. |
| **i-quants** | `IQ2_XXS` … `IQ3_M`, `IQ4_XS` | ✅ imatrix | mainline | Non-linear lattice codebooks + importance matrix. Clearly better quality-per-bit than k-quants, *especially below 4 bpw*. Slightly slower dequant than k-quants. |
| **IQK quants** ⭐ | **`IQ4_KS`**, `IQ5_KS`, `IQ4_K`, `IQ2_K` … | ✅ imatrix | **[ik_llama.cpp](engines/IK_LLAMA.md) only** | Refined grids + imatrix + **kernels co-designed for those grids**. Best quality-per-bit in the GGUF world *and* fast (the dequant path is hand-tuned). Fork-exclusive. |

**The progression that matters:** `Q4_K_M` (data-free) → `IQ4_XS` (imatrix, mainline) → `IQ4_KS` (imatrix + co-designed kernels, ik fork). Each step is better quality at similar bpw. Our shipped `llamacpp/mtp` is at the *first* rung (`Q4_K_M`); the [ik_llama track](engines/IK_LLAMA.md) is at the *last* (`IQ4_KS`).

---

## 3. What "imatrix" actually buys you

A data-free quant treats every weight as equally important and rounds uniformly. But in a trained model, a small fraction of weights carry most of the signal. An **importance matrix** is computed by running calibration text through the model and measuring how much each weight influences activations. The quantizer then:
- protects high-importance weights (more bits / closer grid points), and
- compresses low-importance weights harder.

Result: at 4 bpw, an imatrix quant loses noticeably less quality than a data-free one — and the gap *widens* as you go lower (at 2-3 bpw, imatrix is the difference between usable and broken). The cost is a one-time calibration step when *building* the quant; inference is the same speed.

> **Calibration corpus matters.** An imatrix calibrated on chat+code+tool-calling preserves those skills; one calibrated on Wikipedia may quietly drop tool-call formatting. This is exactly the kind of thing that shows up in our 8-pack quality tests (see [QUALITY_TEST.md](QUALITY_TEST.md)).

---

## 4. The vLLM / safetensors side (not GGUF)

vLLM and SGLang don't use GGUF — they load **safetensors** with these quant schemes:

| Quant | Bits | Calibrated? | Notes |
|---|---|---|---|
| **AutoRound** ⭐ | INT4 | ✅ (sign-gradient) | Intel's method; **what our shipped `vllm/dual` runs** (`qwen3.6-27b-autoround-int4`). Strong 4-bit quality. |
| **AWQ** | INT4 | ✅ (activation-aware) | Protects salient channels by inspecting activations. Widely available. |
| **GPTQ** | INT3/4/8 | ✅ (second-order) | Older, well-supported; AWQ/AutoRound usually edge it at 4-bit. |
| **FP8 (e4m3/e5m2)** | 8 | ❌ | **FP8 *weights* DO run on Ampere — via Marlin W8A16** (weights dequant for the matmul: real VRAM saving, no compute speedup; only Hopper+ multiply FP8 directly). Among the **highest-fidelity** practical quants (see §4a). An FP8 weight checkpoint is compressed-tensors → it **can't also use fp8 *KV*** — pair it with int8-PTH ([DTYPE_MATRIX](DTYPE_MATRIX.md)). Don't confuse FP8 *weights* with the fp8 *KV cache* (§5). |
| **bitsandbytes** | 4/8 | ❌ | Easy/on-the-fly; lower quality-per-bit than AWQ/AutoRound. |

These are conceptually the same idea as imatrix i-quants (calibrate, protect what matters) in a different ecosystem. There is **no GGUF↔safetensors interchange** — a quant is tied to its engine family.

---

## 4a. Picking a quant by fidelity (KLD) — and where QAT fits

Method labels don't rank fidelity — **measure it.** The cleanest signal is **KL-divergence vs the BF16 model** (lower = closer to full precision). [Phaelon74's logit-capture sweep](https://github.com/vllm-project/vllm/pull/35961) on Qwen3.6-27B (one consistent instrument across quants):

| Quant | Mean KLD | ~Size | Ampere path |
|---|---|---|---|
| INT8 (W8A16) | **0.009** | 34 GiB | Marlin W8A16 |
| **FP8** | **0.023** | 29 GiB | Marlin W8A16 |
| AWQ-BF16-INT4 | 0.042 | 27 GiB | Marlin int4 |
| AWQ-INT4 | 0.051 | 20 GiB | Marlin int4 |
| **AutoRound INT4** (our `vllm/dual`) | 0.063 | 18 GiB | Marlin int4 |
| NVFP4 variants | 0.06–0.19 | — | **Blackwell-only — Ampere-dead** |

- **Fidelity rises with bits; 8-bit PTQ (FP8/INT8) is already near-lossless** — hence "accuracy → FP8" as a default. INT4 trades fidelity for size/speed.
- **Our shipped AutoRound INT4 is near the *bottom* of the practical pack** — not because AutoRound is bad, but because it's the *smallest* (18 GiB). A method label ("calibrated"/"QAT-adjacent") does **not** guarantee best KLD; the bigger/higher-bit quants win. Always measure for the specific model.
- **KLD here is *weights-only*** (measured with full/BF16 KV). **KV-quant (fp8 / int8-PTH) adds a *separate* error** this number doesn't include → deployed fidelity = weight-KLD + KV-quant drift. The purest accuracy config is best-weights **+ BF16 KV**; a 1-byte KV (to reach max context) spends a little of that back.

**Where QAT fits:** quantization-*aware* training fine-tunes *with* quantization simulated, so weights adapt → near-BF16 at low bits. Its advantage is **concentrated at ≤4-bit** — at 8-bit, PTQ is already near-lossless, so QAT adds little. So: want **4-bit + max fidelity** → a **QAT-int4** (e.g. Google's Gemma QAT) beats AutoRound/AWQ int4; want **max fidelity, size-flexible** → just use **8-bit PTQ (FP8/INT8)**, no QAT needed. PTQ methods that *approach* QAT without retraining: AutoRound (learns rounding), AWQ (activation-aware), GPTQ (2nd-order), QuIP#/AQLM (codebook, 2-bit), SpinQuant/QuaRot (rotation); the GGUF analogue is **imatrix** (§3).

**Tiering principle:** **dual-card = fidelity tier, single-card = fit/speed tier.** A dual's distinct value is *capacity* — spend it on a higher-fidelity quant (FP8/INT8/AWQ) than the single's small INT4, not the same one. This is the **topology projection** of the tier trade-space in **§4b** — a single card is forced into the fit/speed corner; the second card is what *buys* the higher-fidelity (and prefill) corners as real options.

---

## 4b. The tier trade-space — what fast / balanced / max actually optimize

§4a ranks quants by **fidelity** (KLD vs bf16). That's the right instrument for *one* question — "how close to full precision?" — but it's a single axis, and the shipped **fast / balanced / max** composes are **not** three rungs on it. They're **corners of a trade-space**. On this stack's hardware (Ampere SM 8.6, PCIe, no native FP8, bandwidth-bound decode) there are **three independent axes**:

| Axis | Bound by | Direction | Measured by | The tier it defines |
|---|---|---|---|---|
| **Weight fidelity** | bits/weight | more bits → closer to bf16 | **KLD** (§4a) | **max** |
| **Decode speed + context** | memory **bandwidth** | fewer weight bytes → faster decode **and** bigger KV pool | **decode TPS**, KV-pool tokens | **fast** |
| **Prefill / TTFT** | tensor-core **compute** | native INT8 skips the dequant step | **TTFT** | **prefill** |

Two consequences fall straight out of this:

- **Decode speed and context are the *same* axis here.** Decode is bandwidth-bound, so the quant with the fewest weight bytes (INT4) is *both* the fastest to decode *and* the one that leaves the most VRAM for KV → the biggest context pool. "fast" being the headroom leader too isn't a coincidence — it's the same byte count paying off twice.
- **Prefill is a *different* axis from decode.** Prefill is compute-bound, and Ampere has native INT8 tensor cores but no native FP8. So an **INT8 W8A8** quant (8-bit *weights and activations*) runs prefill on the IMMA cores with no dequant → fast TTFT, while FP8 and INT4 must dequant to 16-bit first. A quant can be a mediocre decoder and the best prefiller; decode TPS tells you nothing about it.

### Scheme vs algorithm — don't confuse the two

A recurring trap: **`W4A16` / `W8A8` / `W8A16` / `FP8` are *schemes*** — how many bits, and whether the *activations* are quantized too. The scheme sets **which axis** a quant lives on. **`AutoRound` / `AWQ` / `GPTQ` are *algorithms*** — how the weights are rounded — and they set **accuracy *within*** a scheme. So "AutoRound vs AWQ vs a GPTQ-int4" is one *algorithm* comparison at a fixed scheme (all W4A16); "W4A16 vs W8A8" is a *scheme* comparison across axes. **The tier is set by the scheme; the algorithm only decides how good that tier's fidelity is.** (A practical corollary: a "better" W4 — say a code-calibrated AutoRound — is a *fast-tier* swap, not a move up a tier. It can't change axis by being marginally larger.)

### The corner map

Each tier maximizes one axis and accepts the cost on the others:

| Tier | Optimizes | Measured differentiator | Sacrifices | Typical scheme |
|---|---|---|---|---|
| **fast** ⭐ | decode + context | TPS, KV-pool size | weight fidelity (lowest KLD rank) | W4A16 (AutoRound) |
| **max** | weight fidelity | KLD vs bf16 | decode, pool, prefill | FP8 / INT8 (W8A16) |
| **prefill** | TTFT | TTFT bench | some activation fidelity | INT8 **W8A8** |
| **balanced** | *(see note)* | *(under review)* | *(see note)* | — |

> The §4a *single-card vs dual-card* split is the **topology projection** of this map: a single 24 GB card is forced into the **fast** corner (VRAM picks the smallest quant), and the **second card is what *buys*** the **max** and **prefill** corners as real options. "Add a card" = "unlock the other corners." (See [`DUAL_CARD.md`](DUAL_CARD.md) for the per-model dual-card pick-table this maps onto.)

### The rule that makes a tier legitimate

**A tier earns its place only if it owns a corner no other tier dominates — with a *measurable* differentiator.** The instrument has to be able to actually *see* the thing the tier claims:

- weight fidelity → **KLD** — but note KLD is **weights-only**; it cannot see activation-quant or KV-quant error,
- decode / context → **TPS + KV-pool size**,
- prefill → **TTFT**,
- anything fidelity-related KLD can't see (activation quant, KV quant, calibration drift) → the **8-pack** quality suite ([QUALITY_TEST.md](QUALITY_TEST.md)).

**A claimed differentiator no instrument can measure is not a tier — it's a guess.** This is the test that decides whether a candidate quant ships as its own tier or folds into an existing one.

> **Worked example — why "balanced" is provisional.** "balanced" has historically meant *an INT4 weight + a higher-fidelity KV* (int8-per-token-head instead of fp8), betting the KV fidelity is worth a tier. On the measurements we have, it's **dominated by fast**: same-or-slower decode, a *smaller* KV pool (heavier weights leave less room), and a *tie* on the 8-pack. Its only claimed edge is a KV cache that is the **same byte size** as fast's and that the short-context 8-pack can't see. Until an instrument that *can* see it (long-context recall / NIAH) separates them, "balanced" isn't a real corner — it's fast with a more expensive KV. The honest options are three: prove it with the right instrument, redefine it onto a real axis, or retire it.

### The prefill corner, filled — **INT8 W8A8** (measured 2026-06-30, vLLM v0.24.0)

The third axis — **prefill / TTFT** — now has an occupant. An **INT8 W8A8** quant (INT8 weights *and* dynamic INT8 activations, vision kept fp16) was benched head-to-head against **FP8** at the 8-bit slot, *all else equal* (same int8-per-token-head KV → same 295K pool, same MTP n=3, same TP=2, same 262K ctx, same harness). vLLM picks the native **CUTLASS INT8** GEMM for it on Ampere sm_86; FP8 falls back to **Marlin weight-only dequant** (it logs *"may degrade performance for compute-heavy workloads"*).

| | **W8A8** (CUTLASS INT8) | **FP8** (Marlin dequant) |
|---|---|---|
| 8-pack quality (`--full`, 150) | **107/150** | **107/150** |
| Prefill t/s @10K → 90K | **2062 → 1021** | 1364 → 875 (**W8A8 +17–51%**) |
| Short-prompt TTFT | **122 ms** | 158 ms |
| Decode TPS (narr/code) | 77 / 96 | **83 / 108** |
| KV pool | 295K / 1.13× | 295K / 1.13× |

**Verdict: W8A8 *qualifies* as a legitimate new tier (by the legitimacy rule above), not a replacement** — but *qualifying* (this analysis) is separate from a *shipping* decision (a catalog call, deferred pending cross-rig numbers + the thinking-on arm; not committed as a shipped slug). It's a clean **prefill-vs-decode tradeoff at the 8-bit slot** — *equal quality (107=107, pack-for-pack ±1), equal pool*, but W8A8 wins **prefill/TTFT** (native INT8 IMMA, the compute-bound path FP8's Marlin dequant is explicitly worse at) while FP8 wins **decode** (its Marlin kernel is better-tuned for single-token). So:

- **W8A8 = the prefill / agentic 8-bit corner** (long-prompt · RAG · tool-chains — TTFT-bound).
- **FP8 = the decode-throughput 8-bit corner.**
- Neither dominates; they're different corners. This is the legitimacy rule in action — W8A8 earns its corner with a **measurable** differentiator (prefill t/s + TTFT), exactly what "balanced" lacked.

> **The activation-quant fear was unfounded.** W8A8 quantizes activations — §4a's weights-only KLD can't see that cost, which is *why* this needed the 8-pack. The 8-pack saw it: **zero cost** (107 = 107). 8-bit weight fidelity + dynamic per-token activation scales held tool-call / structured / numeric / agentic quality identical to near-lossless FP8.

**Open refinements (don't block the analysis):** this run was **thinking-off** (`--full` default) — a thinking-**on** arm is the remaining two-arm check; and a **bf16-KV** W8A8 likely pushes prefill/TTFT further still, trading max-ctx. Both are follow-ups, not gates.

---

## 5. KV-cache quantization (a separate knob)

Independent of the weight quant, you can quantize the **KV cache** — this is what sets your max context, not your model quality:

| KV type | Bits | Engine | Notes |
|---|---|---|---|
| `f16` | 16 | all | Lossless, biggest. Rarely needed. |
| `q8_0` | 8 | llama.cpp / ik | Near-lossless; good default when context is moderate. |
| `q4_0` | 4 | llama.cpp / ik | Halves KV vs q8_0 → enables **262K on one 3090** (ik IQ4_KS). Tiny quality cost. |
| `fp8_e5m2` | 8 | vLLM | Our `vllm/dual` default (AutoRound weights) — the Ampere-safe storage-only fp8. |
| `fp8_e4m3` | 8 | vLLM | Same bytes as e5m2 but **native FP8 Tensor-Core compute on sm_89+** (Ada/Hopper/Blackwell; hard-rejected on Ampere). Since [#246](https://github.com/noonghunna/club-3090/issues/246) the launchers inject it automatically on those cards for the pilot slugs — you don't hand-pick it; `KV_CACHE_DTYPE=` in your env overrides. |
| `nvfp4` | 4 | vLLM ≥ v0.24.0 | Blackwell-only (sm ≥ 10.0) FP4 KV — a valid dtype literal in our pin, **unvalidated on this stack** (candidate on the Blackwell hardware profiles; #246 A/B arm 3). Needs a NIAH-clean cross-rig gate before it's anyone's default. |
| **`int8_per_token_head`** | 8 | vLLM | ~1 byte/tok like fp8; **native in stock v0.22.0** for standard models (Gemma-4 needs the #40391 overlay). The KV path for **compressed-tensors weights (AWQ/FP8/INT8) at long context** — those can't use fp8 KV. |
| **TQ3 (TurboQuant)** | 3 | vLLM (Genesis) | 3-bit KV — beats fp8 on long-context memory; powers our `dual-turbo`. See [TQ3_MTP_GENESIS.md](TQ3_MTP_GENESIS.md) + [CLIFFS.md](CLIFFS.md). |
| `-khad` (modifier) | — | **ik only** | Hadamard transform on the K-cache → recovers accuracy lost to KV quantization, so you keep quality at q4_0/q8_0. |

> ⚠️ **fp8 KV is rejected for compressed-tensors checkpoints** (AWQ / FP8 / INT8 weights): `--kv-cache-dtype fp8_e5m2` → `ValueError: … not supported with fp8 checkpoints`, regardless of the `--quantization` flag. Use **`int8_per_token_head`** there — `auto_round`/GPTQ weights are unaffected (they take fp8 KV fine). Full picker + the Gemma-4 #40391 caveat: [DTYPE_MATRIX](DTYPE_MATRIX.md).

---

## 6. Engine × quant support

| Quant family | vLLM | mainline llama.cpp | ik_llama.cpp | SGLang |
|---|---|---|---|---|
| K-quants (`Q4_K_M`…) | ❌ | ✅ | ✅ | ❌ |
| i-quants (`IQ4_XS`…) | ❌ | ✅ | ✅ | ❌ |
| **IQK (`IQ4_KS`…)** | ❌ | ❌ | ✅ **only** | ❌ |
| AutoRound / AWQ / GPTQ | ✅ | ❌ | ❌ | ✅ |
| FP8 weights | ✅ | ❌ | ❌ | ✅ |

---

## 7. Why doesn't every GGUF repo ship IQK?

If IQK is the best quality-per-bit, why are most community GGUFs still `Q4_K_M`?

1. **It's fork-locked.** IQK quants run *only* on ik_llama.cpp. A `Q4_K_M` runs on mainline llama.cpp, Ollama, LM Studio, LocalAI, Jan — everything. Quant authors optimize for reach.
2. **Kernel co-design.** IQK's quality comes partly from kernels written *for* its grids. Porting that to mainline isn't a small patch, and upstreaming has been slow.
3. **Inertia + tooling.** `Q4_K_M` is the well-trodden default; build pipelines, docs, and "recommended download" buttons all point at it.

So IQK is a deliberate "I'll run the fork to get the better quant" choice — which is exactly the niche the [ik_llama track](engines/IK_LLAMA.md) fills on this stack.

---

## 8. What this stack ships (and why)

| Path | Quant (weights) | KV | Rationale |
|---|---|---|---|
| `vllm/dual` | AutoRound INT4 | fp8_e5m2 | Production dual-card; deepest Qwen3-Next feature support |
| `vllm/dual-turbo` | AutoRound INT4 | **TQ3** | Max throughput + long context (3-bit KV) |
| `llamacpp/mtp` | **Q4_K_M** | q4_0 | Conservative, mainline image, cliff-immune single-card |
| `ik-llama/iq4ks-mtp` ⭐ | **IQ4_KS** (imatrix) | q4_0 + `-khad` | Advanced-quant track: best quality-per-bit + 262K single-card |

**Rule of thumb for your own rig:**
- Tightest VRAM / lowest bpw → reach for an **imatrix quant** (`IQ4_XS` mainline, or `IQ4_KS` on ik_llama), not a data-free `Q4_K_M`.
- Want maximum quality-per-bit and willing to run the fork → **ik_llama + IQK**.
- Multi-tenant / vision / tools at scale → **vLLM + AutoRound**.
- "Just works everywhere, no fork" → mainline **llama.cpp + Q4_K_M**.

---

## See also
- [engines/IK_LLAMA.md](engines/IK_LLAMA.md) — the engine that unlocks IQK
- [INFERENCE_ENGINES.md](INFERENCE_ENGINES.md) — engine comparison
- [DTYPE_MATRIX.md](DTYPE_MATRIX.md) — compute/KV dtype matrix
- [CLIFFS.md](CLIFFS.md) + [TQ3_MTP_GENESIS.md](TQ3_MTP_GENESIS.md) — KV-cache quant deep-dives
- [BENCHMARKS.md](../BENCHMARKS.md) — measured quality + TPS per quant/engine
