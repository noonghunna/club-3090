# Qwen3.6-27B on SGLang — re-test pending

SGLang is a strong alternative to vLLM for high-throughput multi-tenant serving — RadixAttention prefix sharing, structured-output-aware scheduling. Often beats vLLM by 10-30% on aggregate throughput **when both work**.

**Status (2026-05-04):** the historical block (Marlin pad-sub-tile-n + EAGLE/DeltaNet rollback) is partially out of date — SGLang upstream has moved. We need to re-test before we know whether AutoRound INT4 + TP=2 still hits the kernel bug, or whether the path is now clear for a polished SGLang variant on this stack.

---

## TL;DR

| Feature | SGLang upstream status (May 2026) | Verified on this stack? |
|---|---|---|
| **DFlash spec-decode** | ✅ Native, recent ([z-lab confirmed](https://github.com/z-lab/dflash) — `--speculative-algorithm DFLASH`). Qwen3.6-27B draft published at [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash). | ❌ untested |
| **MTP** | ✅ Native, first-class for Qwen3-Next family (per [LMSYS Jul 2025 blog](https://www.lmsys.org/blog/2025-07-17-mtp/)). | ❌ untested |
| **EAGLE-2/EAGLE-3** | ✅ Mainline | ❌ untested. Was previously blocked by DeltaNet/GDN rollback issue — unclear if upstream has resolved this. |
| **TurboQuant 3-bit KV** | ⚠️ WIP — [Issue #21618](https://github.com/sgl-project/sglang/issues/21618) (core) + [Issue #23134](https://github.com/sgl-project/sglang/issues/23134) (4-bit fused Triton). Not in mainline yet. | n/a (not available) |
| **Marlin pad-sub-tile-n fix** (AutoRound INT4 + TP=2 boot) | ⚠️ Unknown — was the binding blocker last we checked. Same kernel-line fix as our [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) applies; needs verification on current SGLang main. | ❌ unverified — **this is the gating question for re-test** |

The `sglang/` directory exists in the repo for organizational symmetry with `vllm/` and `llama-cpp/`, but **no working compose ships here yet**. Re-test plan below decides whether one can.

---

## Re-test plan

If a contributor wants to land an SGLang variant, here's the order we'd attempt it:

### Step 1 — Pull latest SGLang main + smoke-boot

```bash
# Use the most recent SGLang container/build to pick up any recent kernel landings
docker pull lmsysorg/sglang:latest

# Smoke-boot at TP=2 with our AutoRound INT4 weights, no spec-decode, no exotic KV:
docker run --gpus all -it --rm \
  -v ${MODEL_DIR}:/root/.cache/huggingface \
  -p 30000:30000 \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
    --model-path /root/.cache/huggingface/qwen3.6-27b-autoround-int4 \
    --tp 2 \
    --port 30000
```

**Decision point:**
- ✅ **Boots cleanly** → proceed to step 2
- ❌ **Marlin pad-sub-tile-n error** ("size_n must be divisible by 64" or similar) → SGLang still has the bug; file an SGLang PR mirroring [vllm#40361](https://github.com/vllm-project/vllm/pull/40361)'s fix and stop here

### Step 2 — Validate working baseline (if step 1 boots)

Test with **fp8 or q4 KV**, NOT TurboQuant (TurboQuant is WIP on SGLang per #21618). Two candidate baselines, in order of preference:

```bash
# Preferred: fp8 KV (matches our vLLM dual.yml ship config)
--kv-cache-dtype fp8_e5m2 --max-context-len 262144

# Or if fp8 isn't available: q4 KV
--kv-cache-dtype q4 --max-context-len 262144
```

Run `bash scripts/verify-stress.sh` against the SGLang endpoint — same probes we use on vLLM. If 7/7 pass, we have a working baseline and can move to spec-decode.

### Step 3 — Add DFlash spec-decode (preferred over MTP)

DFlash gets priority because:
- Higher acceptance rate per published benchmarks (~6× vs MTP's ~3-4×)
- z-lab explicitly maintains the SGLang integration
- We already have the Qwen3.6-27B draft model on disk if `WITH_DFLASH_DRAFT=1` was used in setup.sh

```bash
--speculative-algorithm DFLASH \
--speculative-draft-model-path z-lab/Qwen3.6-27B-DFlash \
--speculative-num-draft-tokens 16
```

Run `verify-stress.sh` again + `scripts/bench.sh` for canonical TPS comparison against our vLLM `dual-dflash.yml` (185K, 82 narr / 125 code TPS on 2× 3090).

**If DFlash blocks** (e.g. because the DeltaNet rollback issue is still binding on SGLang's spec-decode path): fall back to MTP:

```bash
--speculative-algorithm MTP
# (assumes the model's mtp.safetensors head is loaded automatically)
```

### Step 4 — Land the variant if benches are competitive

If SGLang + DFlash beats or matches vLLM `dual-dflash.yml` on the canonical bench (78-82 narr / 125-127 code TPS on 2× 3090) AND `verify-stress.sh` 7/7 passes AND `SOAK_MODE=continuous` passes, ship as `models/qwen3.6-27b/sglang/compose/docker-compose.dual.yml` and add a BENCHMARKS row.

If SGLang underperforms vLLM — keep this README as historical context, document the measured delta, and pick vLLM.

---

## Why we don't ship one yet

**Last attempt (early 2026):** AutoRound INT4 + TP=2 hit the Marlin pad-sub-tile-n bug ([vllm#40361](https://github.com/vllm-project/vllm/pull/40361)'s SGLang equivalent). EAGLE spec-decode was separately blocked by DeltaNet/GDN hybrid layer not supporting KV rollback. We deferred until either upstream blocker cleared.

**Since then (May 2026):** DFlash + MTP support has landed in SGLang mainline per public docs/blogs. The Marlin pad-sub-tile-n fix may or may not have propagated — that's the gating question for re-test. The pad-sub-tile-n bug is INT4-specific (FP16/bf16 weights work fine on SGLang); we're affected because AutoRound INT4 is the default ship quant on this repo.

If a cross-rig contributor with FP16 or bf16 weights wants to bench SGLang on Qwen3.6-27B today, that's also useful and would fill a different gap.

---

## Watch list

When ANY of the following happens, the SGLang re-test becomes more concrete:

- vLLM PR [#40361](https://github.com/vllm-project/vllm/pull/40361) merges upstream → SGLang likely picks up the fix via shared Marlin
- SGLang Issue [#21618](https://github.com/sgl-project/sglang/issues/21618) (TurboQuant KV) merges → unblocks parity with our `dual-turbo.yml` capacity story
- A community contributor reports SGLang + Qwen3.6-27B-INT4 + TP=2 boots cleanly on current main → green light for the re-test plan above

---

## Cross-links

- [docs/engines/README.md](../../../docs/engines/README.md) — engine comparison
- [docs/UPSTREAM.md](../../../docs/UPSTREAM.md) — upstream issue tracking
- [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) — the Marlin pad-sub-tile-n fix that should propagate to SGLang's Marlin call site
- [z-lab/dflash](https://github.com/z-lab/dflash) — DFlash spec-decode (with SGLang support confirmed)
- [LMSYS MTP blog](https://www.lmsys.org/blog/2025-07-17-mtp/) — SGLang's native MTP integration
- [`../vllm/`](../vllm/) — the validated default path on this stack
