# Benchmarks — measured numbers, by model

This file is the consolidated cross-rig table for every compose variant we
ship, with **measured** numbers (not derived estimates). Rows land here
when a contributor opens a PR adding a new compose variant or supplies
canonical bench output via the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml)
issue template.

Per-model qualitative findings, framework comparisons, and "why we picked
this quant" rationale live in `models/<model>/INTERNALS.md` (or the local
`learnings/` tree). This file is **just the numbers, anchored to dates**.

---

## Canonical bench

All `Narr / Code TPS` rows come from `bash scripts/bench.sh`, which runs:

> **Narrative:** "Write a detailed 800-word essay explaining transformer attention." (`max_tokens=1000`)
>
> **Code:** "Write a Python implementation of quicksort with comments explaining each step." (`max_tokens=800`)
>
> Sampling: `temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0, enable_thinking=false`. Three warmups + five measured runs per prompt. Mean wall TPS reported.

Cross-rig numbers are comparable because the prompt + sampling are pinned. Variations against your rig usually trace back to power caps, PCIe lane counts, or pin (vLLM image SHA / Genesis commit) — see [`scripts/report.sh`](scripts/report.sh) which captures all three.

---

## Qwen3.6-27B

Primary serving model. Hybrid Qwen3-Next architecture (DeltaNet GDN + standard attention). Quants used here: AutoRound INT4 (vLLM), Unsloth Q5_K_XL GGUF (llama.cpp).

### Single-card (1× RTX 3090)

vLLM:

| Compose | KV | Max ctx | Narr / Code TPS | Peak VRAM | Date | Notes |
|---|---|---:|---:|---:|---|---|
| `minimal.yml` (`mem-util 0.95 max-model-len 65536`) | TQ3 | 64K | ~32 / ~33 | ~22.4 GB | 2026-05-03 | no MTP. [stiggy2k16](https://github.com/noonghunna/club-3090/issues/43) data point — short-prompt vLLM-safe path when llama.cpp is too slow. |
| `long-vision.yml` | TQ3 | 145K | 50 / 66 | ~23.0 GB | 2026-04-30 | vision + tools + thinking. mem-util 0.95. |
| `long-text.yml` | TQ3 | 180K | 50 / 67 | ~22.3 GB | 2026-04-30 | text-only (vision tower dropped). MTP n=3. mem-util 0.93. **Default for RAG / IDE agents below 25K accumulated ctx**. |
| `long-text-no-mtp.yml` | TQ3 | 200K | TBD | ~21.0 GB | — | max-context single-shot, no MTP. Slow decode but biggest ctx window. |
| `bounded-thinking.yml` | TQ3 | 180K | 50 / 66 (~10% slower than long-text per-token; per-problem 10× faster on coding bench because thinking is 30× shorter) | ~21.7 GB | 2026-04-30 | structured-CoT FSM + grammar enforcement. See [STRUCTURED_COT.md](docs/STRUCTURED_COT.md). 92.7% pass@1 HE+, 66% LCB v6. |
| `tools-text.yml` | fp8 | 75K | TBD | TBD | — | IDE-agent path that escapes the long-text Cliff 1 mech B leak (see [#16](https://github.com/noonghunna/club-3090/issues/16)). |

llama.cpp:

| Compose | Quant | Max ctx | Narr / Code TPS | Peak VRAM | Date | Notes |
|---|---|---:|---:|---:|---|---|
| `llamacpp/default` | Unsloth Q5_K_XL GGUF | 262K | 21 / 21 | ~20 GB | 2026-04-21 | bulletproof — different engine, different memory allocator, no Cliff 1 / Cliff 2. Slow decode but cliff-immune. |
| `llamacpp/concurrent` | Unsloth Q5_K_XL GGUF | 262K | TBD | TBD | — | concurrent-serving variant. |

### Dual-card (2× RTX 3090, TP=2)

| Compose | KV | Max ctx | Narr / Code TPS | Peak VRAM | Date | Notes |
|---|---|---:|---:|---:|---|---|
| `dual.yml` ⭐ | fp8 | 262K (237K single-prompt verified) | 69 / 89 | ~23.6 GB | 2026-04-29 | tested 2-card baseline. fp8 KV, 2 streams, full feature set. **PASSES v2 continuous soak** (Cliff 2b clean). |
| `dual-turbo.yml` | TQ3 | 262K | 58 / 76 per-stream (**269 TPS aggregate at 4 streams**) | ~19.8 GB | 2026-04-29 | TQ3 KV — 4.67× concurrency for multi-tenant agent workloads. |
| `dual-dflash.yml` | fp8 | 185K | 82 / **125** | ~23.6 GB | 2026-04-29 | DFlash N=5 + 1.75 GB draft / card. AL ~4.4. Fastest 2-card short-prompt code path. |
| `dual-dflash-noviz.yml` | fp8 | 200K | 78 / **127** | ~23.8 GB | 2026-04-29 | DFlash + no vision tower. +15K ctx vs `dual-dflash`. |
| `dual-nvlink.yml` | fp8 | 262K | TBD | TBD | — | NVLink topology variant ([@JusefPol](https://github.com/JusefPol) PR #31). |

### Quad-card (4× RTX 3090, TP=4)

Cross-rig data from [@Whamp](https://github.com/Whamp)'s rig (Threadripper 2950X, 4× RTX 3090 PCIe — GPU 0 at x4 lanes, GPU 2 at x8, GPU 1+3 at x16; 300W power caps; PR [#44](https://github.com/noonghunna/club-3090/pull/44)).

| Compose | KV | Max ctx | Narr / Code TPS | Peak VRAM | Date | Notes |
|---|---|---:|---:|---:|---|---|
| `dual4.yml` | fp8 | 262K | 63 / 76 | ~23.5 GB | 2026-05-03 | TP=4 capacity king. **6.77× concurrency at 262K**. PASSES v2 continuous soak (20 sessions, 0 MiB growth, 90.8% TPS retention). |
| `dual4-dflash.yml` | fp8 | 262K | 64 / **104** | ~22.0 GB | 2026-05-03 | TP=4 + DFlash. 2.27× concurrency at 262K. PASSES v2 continuous soak (5 sessions, 0 MiB growth, 100% TPS retention). **Bench-vs-soak inversion noted**: bench shows DFlash wins by 37% on short-prompt code, soak shows DFlash *loses* by 47% on multi-turn agent workloads — DFlash AL likely collapses on mixed prompts. Pick `dual4` for multi-turn, `dual4-dflash` for full-262K single-stream code. |

### Notes on attribution + reproducibility

- All `Date` entries reference the canonical-bench run date. Re-bench triggers (vLLM image SHA bump, Genesis pin bump, Marlin patch update) update the date on affected rows.
- Per-row VRAM is `peak / card` during the canonical bench (3 warm + 5 measured), captured via `nvidia-smi --query-gpu=memory.used --format=csv` sampled at 1 s during the run.
- Cross-rig contributions land with attribution to the contributor's GitHub handle. The [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) issue template is the canonical entry path.

### Verify-stress + soak-continuous (not TPS, but load-bearing)

Every shipped variant has been validated against:

- `bash scripts/verify-full.sh` — fast functional smoke (8 checks)
- `bash scripts/verify-stress.sh` — boundary tests including Cliff 2 needle recall (probe 7: 60K + 90K needles)
- `SOAK_MODE=continuous bash scripts/soak-test.sh` — multi-turn accumulating-context cliff (Cliff 2b at ~25K)

| Variant | verify-full | verify-stress 7/7 | soak-continuous |
|---|---|---|---|
| `minimal.yml` (single-card vLLM) | PASS | PASS at 64K | **FAIL** — Cliff 2b fires |
| `long-text.yml` | PASS | PASS at 180K | **FAIL** — Cliff 2b fires |
| `long-vision.yml` | PASS | PASS at 145K | **FAIL** — Cliff 2b fires |
| `bounded-thinking.yml` | PASS | PASS at 180K | **FAIL** — Cliff 2b fires |
| `tools-text.yml` | PASS | PASS at 75K | **FAIL** — Cliff 2b fires |
| `llamacpp/default` | PASS | PASS at 262K | **PASS** — different engine, no cliff |
| `dual.yml` (TP=2) | PASS | PASS at 262K (237K single-prompt) | **PASS** |
| `dual-turbo.yml` (TP=2) | PASS | PASS at 262K | **PASS** (assumed by activation-split argument; not yet measured cross-rig) |
| `dual-dflash.yml` (TP=2) | PASS | PASS at 185K | TBD |
| `dual-dflash-noviz.yml` (TP=2) | PASS | PASS at 200K | TBD |
| `dual4.yml` (TP=4) | PASS | PASS at 262K (incl. 58K + 91K needles) | **PASS** (Whamp 2026-05-03) |
| `dual4-dflash.yml` (TP=4) | PASS | PASS at 262K (incl. 58K + 91K needles) | **PASS** (Whamp 2026-05-03) |

The single-card vLLM Cliff 2b status is canonicalized in [#41](https://github.com/noonghunna/club-3090/issues/41) — fix is gated on upstream [Sandermage genesis-vllm-patches#19](https://github.com/Sandermage/genesis-vllm-patches/issues/19). See [docs/CLIFFS.md](docs/CLIFFS.md) for the byte-level explanation.

---

## See also

- [docs/SINGLE_CARD.md](docs/SINGLE_CARD.md) — single-card variant picker
- [docs/DUAL_CARD.md](docs/DUAL_CARD.md) — 2-card variant picker
- [docs/MULTI_CARD.md](docs/MULTI_CARD.md) — 4+ card variant picker
- [docs/STRUCTURED_COT.md](docs/STRUCTURED_COT.md) — bounded-thinking benchmark on HumanEval+ + LiveCodeBench v6
- [docs/CLIFFS.md](docs/CLIFFS.md) — known failure modes and which variants escape them
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add a row
