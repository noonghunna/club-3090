# Benchmarks — measured numbers, by model

This file is the consolidated cross-rig table for every compose variant we
ship, with **measured** numbers (not derived estimates). It's intentionally
**append-friendly** — every row carries an explicit `Rig` cell so multiple
contributors can publish numbers for the same compose without rewriting the
file.

Rows land here:
- when a contributor opens a PR adding a new compose variant, OR
- when a contributor supplies canonical bench output via the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) issue template.

Per-model qualitative findings, framework comparisons, and "why we picked
this quant" rationale live in `models/<model>/INTERNALS.md` (or the local
`learnings/` tree). This file is **just the numbers, anchored to (rig, date)**.

---

## Canonical bench

All `Narr / Code TPS` rows come from `bash scripts/bench.sh`, which runs:

> **Narrative:** "Write a detailed 800-word essay explaining transformer attention." (`max_tokens=1000`)
>
> **Code:** "Write a Python implementation of quicksort with comments explaining each step." (`max_tokens=800`)
>
> Sampling: `temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0, enable_thinking=false`. Three warmups + five measured runs per prompt. Mean wall TPS reported.

Cross-rig numbers are comparable because the prompt + sampling are pinned. Variations against your rig usually trace back to power caps, PCIe lane counts, or pin (vLLM image SHA / Genesis commit) — see [`scripts/report.sh`](scripts/report.sh) which captures all three.

## How to add a row for your rig

1. Run `bash scripts/report.sh --full > my-rig.md` — captures hardware (incl. power caps + PCIe lanes), stack version (vLLM image SHA, Genesis commit), verify-full + verify-stress + **SOAK_MODE=continuous** + canonical bench numbers in one ~35-min pass. (Or `--bench` for the fast subset; soak-continuous catches Cliff 2b which the others don't.)
2. Open the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) issue template, paste the report, mention which compose variant you ran.
3. We'll append your numbers as a row in the appropriate table here, with `Rig` cell formatted `@your-handle (rig-shape)` — e.g. `@whamp (4× 3090 PCIe x4/x8/x16/x16, 300 W)`.

If the same compose has multiple rig rows showing different numbers, that's a feature — it tells future readers what's portable vs rig-specific.

---

## Qwen3.6-27B

Primary serving model. Hybrid Qwen3-Next architecture (DeltaNet GDN + standard attention). Quants used: AutoRound INT4 (vLLM), Unsloth Q5_K_XL GGUF (llama.cpp).

### Single-card (1× RTX 3090) — vLLM

| Compose | Rig | KV | Max ctx | Narr / Code TPS | Peak VRAM | Date | Notes |
|---|---|---|---:|---:|---:|---|---|
| `minimal.yml` (`mem-util 0.95 max-model-len 65536`) | @noonghunna (1× 3090, x16, 350 W) | TQ3 | 64K | ~32 / ~33 | ~22.4 GB | 2026-05-03 | no MTP. [stiggy2k16](https://github.com/noonghunna/club-3090/issues/43) cross-rig data point — short-prompt vLLM-safe path when llama.cpp is too slow. |
| `long-vision.yml` | @noonghunna (1× 3090) | TQ3 | 145K | 50 / 66 | ~23.0 GB | 2026-04-30 | vision + tools + thinking. mem-util 0.95. |
| `long-text.yml` ⭐ | @noonghunna (1× 3090) | TQ3 | 180K | 50 / 67 | ~22.3 GB | 2026-04-30 | text-only (vision tower dropped). MTP n=3. mem-util 0.93. **Default for RAG / IDE agents below 25K accumulated ctx**. |
| `long-text-no-mtp.yml` | @noonghunna (1× 3090) | TQ3 | 200K | TBD | ~21.0 GB | — | max-context single-shot, no MTP. Slow decode but biggest ctx window. |
| `bounded-thinking.yml` | @noonghunna (1× 3090) | TQ3 | 180K | 50 / 66 | ~21.7 GB | 2026-05-04 | structured-CoT FSM in reasoning channel; **recommended grammar: DeepSeek scratchpad** (PLAN/NOTE×0-15/VERDICT). Phase 3 final: **93.9% HE+ / 66.0% LCB v6** (87.4% combined, +1 net vs the andthattoo G/A/E baseline). Andthattoo G/A/E grammar also works (94.5% HE+ / 62.0% LCB / 86.9% combined, ~4× tighter think budget — pass via `extra_body`). See [STRUCTURED_COT.md](docs/STRUCTURED_COT.md). |
| `tools-text.yml` | @noonghunna (1× 3090) | fp8 | 75K | TBD | TBD | — | IDE-agent path that escapes the long-text Cliff 1 mech B leak (see [#16](https://github.com/noonghunna/club-3090/issues/16)). |

### Single-card (1× RTX 3090) — llama.cpp

| Compose | Rig | Quant | Max ctx | Narr / Code TPS | Peak VRAM | Date | Notes |
|---|---|---|---:|---:|---:|---|---|
| `llamacpp/default` | @noonghunna (1× 3090) | Unsloth Q5_K_XL | 262K | 21 / 21 | ~20 GB | 2026-04-21 | bulletproof — different engine, different memory allocator, no Cliff 1 / Cliff 2. Slow decode but cliff-immune. |
| `llamacpp/concurrent` | @noonghunna (1× 3090) | Unsloth Q5_K_XL | 262K | TBD | TBD | — | concurrent-serving variant. |

### Dual-card (2× RTX 3090, TP=2)

| Compose | Rig | KV | Max ctx | Narr / Code TPS | Peak VRAM | Date | Notes |
|---|---|---|---:|---:|---:|---|---|
| `dual.yml` ⭐ | @noonghunna (2× 3090 PCIe, no NVLink) | fp8 | 262K (237K single-prompt verified) | 69 / 89 | ~23.6 GB | 2026-04-29 | tested 2-card baseline. fp8 KV, 2 streams, full feature set. **PASSES v2 continuous soak** (Cliff 2b clean). |
| `dual-turbo.yml` | @noonghunna (2× 3090 PCIe) | TQ3 | 262K | 58 / 76 per-stream (**269 TPS aggregate at 4 streams**) | ~19.8 GB | 2026-04-29 | TQ3 KV — 4.67× concurrency for multi-tenant agent workloads. |
| `dual-dflash.yml` | @noonghunna (2× 3090 PCIe) | fp8 | 185K | 82 / **125** | ~23.6 GB | 2026-04-29 | DFlash N=5 + 1.75 GB draft / card. AL ~4.4. Fastest 2-card short-prompt code path. |
| `dual-dflash-noviz.yml` | @noonghunna (2× 3090 PCIe) | fp8 | 200K | 78 / **127** | ~23.8 GB | 2026-04-29 | DFlash + no vision tower. +15K ctx vs `dual-dflash`. |
| `dual-dflash-noviz.yml` | @snoby (2× **4090** PCIe — 5-GPU rig, GPUs 2,3, no NVLink, [#46](https://github.com/noonghunna/club-3090/issues/46)) | fp8 | **180K** | 92.55 / **148.99** | ~21.8 GB | 2026-05-04 | First non-3090 cross-rig data. **Required `max-model-len` drop from 200K→180K** vs 3090 baseline (boot OOM at 200K) — 4090 ctx-ceiling gotcha pending investigation. +17% TPS lift vs same compose on 3090 (78→92.55 narr / 127→148.99 code). |
| `dual-nvlink.yml` | @JusefPol (2× 3090 PCIe x8 + **NVLink 4× bonded**, i7-11700K, 365 W/card) | fp8 | 262K | **108.81 / 138.55** | ~23.7 GB | 2026-05-04 | First NVLink cross-rig data. **+58% narr / +56% code TPS vs `dual.yml` PCIe-only baseline (69 / 89)** — NVLink reduces the per-token NCCL allreduce latency floor; compounds at multi-stream. verify-stress 7/7 PASS incl. 91K needle. **PASSES v2 continuous soak** (5 sessions × 5 turns, 0 MiB growth, 100% TPS retention). MTP n=3, 65–98% per-position accept. PR [#31](https://github.com/noonghunna/club-3090/pull/31). |
| `carnice-bf16mtp.yml` | @noonghunna (2× 3090 PCIe, no NVLink) | fp8 | 65K | 65 / 81 | ~22.25 GB | 2026-05-04 | **Carnice-V2-27B (Hermes agentic fine-tune) + BF16 MTP overlay**. ~95% narr / ~91% code TPS of base-Qwen `dual.yml`. Patched chat template for Hermes JSON tool calls. verify-full 7/8 PASS (thinking test lenient due to Carnice short-reasoning style). verify-stress PAS-6/7 (needle recall failures at ≥60K — model-level GDN attention ceiling). |

### Quad-card (4× RTX 3090, TP=4)

| Compose | Rig | KV | Max ctx | Narr / Code TPS | Peak VRAM | Date | Notes |
|---|---|---|---:|---:|---:|---|---|
| `dual4.yml` | @whamp (4× 3090 PCIe x4/x16/x8/x16, 300 W cap, no NVLink) | fp8 | 262K | 63 / 76 | ~23.5 GB | 2026-05-03 | TP=4 capacity king. **6.77× concurrency at 262K**. PASSES v2 continuous soak (20 sessions, 0 MiB growth, 90.8% TPS retention). PR [#44](https://github.com/noonghunna/club-3090/pull/44). |
| `dual4-dflash.yml` | @whamp (4× 3090 PCIe x4/x16/x8/x16, 300 W cap) | fp8 | 262K | 64 / **104** | ~22.0 GB | 2026-05-03 | TP=4 + DFlash. 2.27× concurrency at 262K. PASSES v2 continuous soak (5 sessions, 0 MiB growth, 100% TPS retention). **Bench-vs-soak inversion**: bench shows DFlash wins by 37% on short-prompt code, soak shows DFlash *loses* by 47% on multi-turn agent — DFlash AL likely collapses on mixed prompts. PR [#44](https://github.com/noonghunna/club-3090/pull/44). |

### Verify-stress + soak-continuous matrix

Not TPS, but load-bearing. Every shipped variant is validated against:

- `bash scripts/verify-full.sh` — fast functional smoke (8 checks)
- `bash scripts/verify-stress.sh` — boundary tests including Cliff 2 needle recall (probe 7: 60K + 90K needles)
- `SOAK_MODE=continuous bash scripts/soak-test.sh` — multi-turn accumulating-context cliff (Cliff 2b at ~25K)

| Variant | Rig | verify-full | verify-stress 7/7 | soak-continuous | Date |
|---|---|---|---|---|---|
| `minimal.yml` (single-card vLLM) | @noonghunna | PASS | PASS at 64K | **FAIL** — Cliff 2b fires | 2026-05-03 |
| `long-text.yml` | @noonghunna | PASS | PASS at 180K | **FAIL** — Cliff 2b fires | 2026-05-03 |
| `long-vision.yml` | @noonghunna | PASS | PASS at 145K | **FAIL** — Cliff 2b fires | 2026-05-03 |
| `bounded-thinking.yml` | @noonghunna | PASS | PASS at 180K | **FAIL** — Cliff 2b fires | 2026-05-03 |
| `tools-text.yml` | @noonghunna | PASS | PASS at 75K | **FAIL** — Cliff 2b fires | 2026-05-03 |
| `llamacpp/default` | @noonghunna | PASS | PASS at 262K | **PASS** — different engine, no cliff | 2026-04-21 |
| `dual.yml` (TP=2) | @noonghunna | PASS | PASS at 262K (237K single-prompt) | **PASS** | 2026-05-03 |
| `dual-turbo.yml` (TP=2) | @noonghunna | PASS | PASS at 262K | **PASS** (assumed by activation-split argument; not yet measured cross-rig) | 2026-04-29 |
| `dual-dflash.yml` (TP=2) | @noonghunna | PASS | PASS at 185K | TBD | — |
| `dual-dflash-noviz.yml` (TP=2) | @noonghunna | PASS | PASS at 200K | TBD | — |
| `dual4.yml` (TP=4) | @whamp | PASS | PASS at 262K (incl. 58K + 91K needles) | **PASS** (20 sessions, 0 MiB growth, 90.8% retention) | 2026-05-03 |
| `dual4-dflash.yml` (TP=4) | @whamp | PASS | PASS at 262K (incl. 58K + 91K needles) | **PASS** (5 sessions, 0 MiB growth, 100% retention; ⚠ 4 turns >30s; n=5 small) | 2026-05-03 |

The single-card vLLM Cliff 2b status is canonicalized in [#41](https://github.com/noonghunna/club-3090/issues/41) — fix is gated on upstream [Sandermage genesis-vllm-patches#19](https://github.com/Sandermage/genesis-vllm-patches/issues/19). See [docs/CLIFFS.md](docs/CLIFFS.md) for the byte-level explanation.

### Cross-engine — Luce DFlash (lucebox-hub) on Qwen3.5-27B

Not directly comparable to vLLM rows above (different engine, different bench script, different model — Qwen3.5-27B not 3.6 because the 3.6 DFlash draft is still under training as of 2026-05-04). Bench harness: `lucebox-hub/dflash/scripts/bench_he.py`, HumanEval 10 prompts, n_gen=128.

| Config | Rig | Mean tok/s | AL | Accept % | Notes |
|---|---|---:|---:|---:|---|
| Same-card, default KV | @noonghunna (1× 3090) | 73.97 | 6.39 | 41.3% | Range 52.7–108.7 across 10 HE prompts. Bench 2026-05-04. |
| Same-card, **K8V4** (`-ctk q8_0 -ctv q4_0`) | @noonghunna (1× 3090) | 74.68 | 6.38 | 41.4% | Range 54.6–109.1. **+1% over default KV** — basically identical. KV-format optimization doesn't help at HE-scale (<150-tok prompts × 128-tok gen) where KV pool isn't the bottleneck. Asymmetric quant via [PR #56/#54](https://github.com/Luce-Org/lucebox-hub/pull/56) merged 2026-04-28. |
| Dual-GPU split ([PR #80](https://github.com/Luce-Org/lucebox-hub/pull/80) `--target-gpu 0 --draft-gpu 1 --draft-feature-mirror`) | @noonghunna (2× 3090, no NVLink, **P2P "Chipset Not Supported"**) | 75.24 | 6.39 | 41.3% | Range 54.2–110.0. **+1.7% over same-card** — but **NOT a fair test of the split's value**. CUDA P2P access is disabled at the chipset level on this rig (PHB topology, consumer-board limitation). The lucebox dual-GPU code path requires P2P for direct draft-feature transfers; without it, falls back to host-staging copies (CPU↔GPU bouncing). The published 51.86 tok/s on dual 2080 Ti 22GB ([PR #80](https://github.com/Luce-Org/lucebox-hub/pull/80)) presumably ran with P2P available. **Verdict for our hardware class**: dual-GPU split needs a P2P-capable interconnect (NVLink or peer-supported chipset) to deliver its value. PHB+CNS rigs see no benefit. |

### PFlash long-context compression on 1× 3090 — measured ceiling 131K source

Bench harness: `lucebox-hub/dflash/scripts/phase_split_dual_gpu.py bench-niah` (PFlash drafter only, no target loaded — measures the prefill compression phase). Drafter: `Qwen3-0.6B-BF16.gguf`, BSA enabled, keep_ratio=0.05.

| Source ctx | Compressed | Ratio | PFlash time | tok/s | Key + answer retained |
|---:|---:|---:|---:|---:|:---:|
| 16,372 | 788 | 0.048 | 1.08 s | 15,117 | ✓ ✓ |
| 32,764 | 1,628 | 0.050 | 1.80 s | 18,205 | ✓ ✓ |
| 65,524 | 3,252 | 0.050 | 4.37 s | 15,009 | ✓ ✓ |
| **131,068** | **6,524** | **0.050** | **10.80 s** | **12,135** | **✓ ✓** |
| 199,996 | OOM at layer 25 (390 MiB ephemeral alloc) | — | — | — | ✗ ✗ |
| 259,996 | OOM at layer 18 (507 MiB ephemeral alloc) | — | — | — | ✗ ✗ |

**Compression-phase result**: PFlash drafter scoring works up to **131K source on 1× 24 GB / 3090** — compresses to 6.5K (5%) in 10.8s with NIAH key + answer retained. Vanilla llama.cpp pp131072 takes ~257s per Luce's published numbers, so the compression phase alone is **~24× faster** at this context. Adding target prefill on the compressed 6.5K would estimated ~1-2s (untested), suggesting ~12-13s end-to-end TTFT vs ~257s vanilla.

Above 131K source the drafter's **ephemeral forward-pass tensors** (`K_curr/V_curr/Q_last` per layer at full sequence length) exceed 24 GB. K-cache quantization (`--pflash-k-type q8_0`) didn't help — the failing allocs are forward-pass not cache. Bench `lucebox-pflash-niah-q8k-20260504-150600/` confirmed identical OOM at 200K and 260K with both BF16 and q8_0 K cache.

**On the @weicj 24K → 262K phase-split claim** ([PR #78](https://github.com/Luce-Org/lucebox-hub/pull/78)): not refuted but not reproduced on our hardware class either — their setup was 2× 22 GB Ti with **target also loaded** on one card; "24K single-card" was target+drafter co-resident. Our 131K is drafter-alone on 24 GB, which already passes their dual-GPU 262K-style scaling sanity-check. Reproducing 262K specifically would need investigation of their drafter config (chunk_size, lookahead, BSA window) — drafter activation footprint at 200K+ is the binding constraint regardless of how many GPUs are present.

#### What we have NOT validated — gates before "shippable"

This is **directional evidence** (TTFT compression + NIAH retention at 131K), not a complete validation. The gates we hold every other shipped compose to are still open for PFlash:

| Gate | Status | Notes |
|---|---|---|
| TTFT speedup at long context | ✅ measured (~24× compression alone) | Single test — needs reproduction across prompt shapes |
| NIAH single-needle retrieval | ✅ measured at every ctx ≤131K | Synthetic test only — single key+answer pair per prompt |
| Target prefill on compressed tokens | ❌ unmeasured | Bench harness measures PFlash phase only |
| Decode TPS after compressed prefill | ❌ unmeasured | End-to-end TTFT + decode pipeline not tested |
| **HumanEval+ / LCB v6 pass@1** | ❌ **not applicable** — those benches have <2K-token prompts; PFlash's compression path wouldn't even engage | Need long-context coding benches (repo-understanding, RULER+code) |
| **Long-context QA accuracy** (RULER, LongBench, multi-needle) | ❌ unmeasured | The actual quality gate — does compression preserve task performance, not just synthetic needle retrieval? |
| `verify-stress.sh` 7/7 | ❌ **0/7 PASS** (2026-05-04, see below) | OpenAI server gate — multiple distinct failures |
| `SOAK_MODE=continuous` | ❌ blocked | Daemon dies during stress; soak can't run on a dead daemon |
| Multi-turn compression stability | ❌ blocked by 3rd-cycle CUDA bug | Daemon hits illegal-mem-access on 3rd compress regardless of GPU layout |

#### verify-stress on PFlash-enabled lucebox server (2026-05-04, single-card and dual-GPU)

Tested via `URL=http://localhost:8004 MODEL=luce-dflash bash scripts/verify-stress.sh` against a `lucebox-hub/dflash/scripts/server.py` boot with `--prefill-compression auto --prefill-threshold 8000 --prefill-keep-ratio 0.05` + Qwen3-0.6B-BF16 drafter. Two configurations:

- **Single-GPU**: target+dflash draft+pflash drafter all on GPU 0 → OOM at 75 MiB on 2nd request, daemon exits, all subsequent probes 503. Logs: `results/lucebox-pflash-verify-stress-20260504-152713/`.
- **Dual-GPU**: local `server.py` patch reading `LUCEBOX_TARGET_GPU=0 LUCEBOX_DRAFT_GPU=1 LUCEBOX_DRAFT_FEATURE_MIRROR=1` to pin dflash draft to GPU 1. Probes 1 (10K + 30K) survive but **return wrong needle answers** because PFlash @ keep=0.05 drops the needle phrase. Probe 2 (25K tool prefill) crashes the daemon with `CUDA error: an illegal memory access was encountered` on the 3rd compress cycle. Probes 3-7 all 503. Logs: `results/lucebox-pflash-verify-stress-dualgpu-20260504-153220/`.

| Probe | Single-GPU | Dual-GPU | Failure mode |
|---|---|---|---|
| 1. 10K + 30K needle | ✗ blank reply | ✗ wrong content | PFlash drops needle phrase from top-5% kept chunks |
| 2. 25K tool prefill | ✗ daemon dead | ✗ HTTP 500 → daemon dies | CUDA illegal-mem-access on 3rd pflash compress |
| 3. IDE-agent | ✗ HTTP 503 | ✗ HTTP 500 | Cliff 1 mech B class on lucebox path; daemon already dead in single-GPU run |
| 4-6. Multi-turn / LCB / reasoning | ✗ HTTP 503 | ✗ HTTP 503 | Daemon died at probe 2 |
| 7. 60K + 90K needle | ✗ HTTP 500 | ✗ HTTP 500 | Daemon dead |

**Two distinct failure classes in the integrated PFlash + DFlash + OpenAI server path:**
1. **Compression-vs-retrieval at keep=0.05**: short factual needles (color animal num) don't survive top-15-chunks selection in 10-30K contexts. The standalone NIAH bench (#230) used a needle/filler pattern PFlash's importance scorer favors; verify-stress's pattern doesn't replicate that. Means PFlash's "key+answer retained" claim is filler-pattern-dependent at moderate contexts.
2. **3rd-cycle multi-cycle daemon stability**: lucebox-hub upstream bug — CUDA illegal-mem-access in `ggml_backend_buffer_free` after the 3rd pflash compress cycle, regardless of single- vs dual-GPU. Daemon doesn't recover; subsequent requests 503.

**Honest read**: PFlash gives us a compelling single-stat win (24× TTFT compression + NIAH-retention) at 131K source on 1× 3090 in the **standalone bench harness**, but the **integrated OpenAI server path fails the verify-stress gate**. PFlash is **not a shippable club-3090 path today**. Re-evaluate when (a) lucebox-hub fixes the 3rd-cycle CUDA stability bug, (b) `--pflash-gpu` lands in `server.py` (currently only standalone bench has it), and (c) the importance scorer / keep ratio reliably preserves arbitrary short needles. Long-context QA harness (RULER or similar) is the meaningful next investment if any of those three land.

**Setup gotcha** for anyone re-running on consumer rigs: check `nvidia-smi topo -p2p r` before configuring `--target-gpu` / `--draft-gpu`. If the matrix shows `CNS` (Chipset Not Supported), the dual-GPU split won't deliver its claimed uplift on that hardware regardless of whether you have multiple GPUs. NVLink-bonded setups would also typically expose P2P (a different cross-rig contributor would need to confirm on lucebox specifically; @JusefPol's [#31](https://github.com/noonghunna/club-3090/pull/31) NVLink win was measured on vLLM TP=2, not lucebox). PHB-only consumer boards typically lack P2P.

---

## See also

- [docs/SINGLE_CARD.md](docs/SINGLE_CARD.md) — single-card variant picker
- [docs/DUAL_CARD.md](docs/DUAL_CARD.md) — 2-card variant picker
- [docs/MULTI_CARD.md](docs/MULTI_CARD.md) — 4+ card variant picker
- [docs/STRUCTURED_COT.md](docs/STRUCTURED_COT.md) — bounded-thinking benchmark on HumanEval+ + LiveCodeBench v6
- [docs/CLIFFS.md](docs/CLIFFS.md) — known failure modes and which variants escape them
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add a row
