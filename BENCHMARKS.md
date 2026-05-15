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

`PP tok/s` is prompt-processing throughput. For vLLM rows, `bench.sh` scrapes
the most recent `Avg prompt throughput` lines from container logs. For
llama.cpp or engines without that log shape, run `PP=1 bash scripts/bench.sh`
to add a single long-prompt fallback probe that computes prompt tokens over
TTFT. Existing rows show `—` until re-benched with v0.7.1+ tooling.

Cross-rig numbers are comparable because the prompt + sampling are pinned. Variations against your rig usually trace back to power caps, PCIe lane counts, or pin (vLLM image SHA / Genesis commit) — see [`scripts/report.sh`](scripts/report.sh) which captures all three.

## How to add a row for your rig

1. Run `bash scripts/report.sh --full > my-rig.md` — captures hardware (incl. power caps + PCIe lanes), stack version (vLLM image SHA, Genesis commit), verify-full + verify-stress + **SOAK_MODE=continuous** + canonical bench numbers in one ~35-min pass. (Or `--bench` for the fast subset; soak-continuous catches Cliff 2b which the others don't.)
2. Open the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) issue template, paste the report, mention which compose variant you ran.
3. We'll append your numbers as a row in the appropriate table here, with `Rig` cell formatted `@your-handle (rig-shape)` — e.g. `@whamp (4× 3090 PCIe x4/x8/x16/x16, 300 W)`.

If the same compose has multiple rig rows showing different numbers, that's a feature — it tells future readers what's portable vs rig-specific.

### Notes column convention — soak verdict

Every row's Notes cell should start (or include) an explicit soak verdict so readers can grep at a glance:

- `Soak: ✓ PASS` — clean (no errors, 0 silent-empty turns, <200 MiB VRAM growth across 5×5 continuous sessions)
- `Soak: ⚠ borderline` — within thresholds but worth flagging (e.g. 240 MiB growth, slow turns >30s on x4 PCIe, etc.)
- `Soak: ✗ FAIL` — Cliff 2b suspect, silent-empty turns, mid-soak errors, or growth-threshold overshoot
- `Soak: —` — not run (acceptable but discouraged; row is "bench-only" until a follow-up brings the soak verdict)

Soak-continuous is the only signal that catches Cliff 2b. If your row says `Soak: —`, future readers will assume the worst; if your row says `Soak: ✓ PASS`, that's the 30-second answer to "is this rig+compose stable for hermes/openhands traffic." Use `bash scripts/soak-test.sh --continuous` (~25 min, auto-detects endpoint + container) to add the verdict if you initially submitted bench-only.

---

## Qwen3.6-27B

Primary serving model. Hybrid Qwen3-Next architecture (DeltaNet GDN + standard attention). Quants used: AutoRound INT4 (vLLM), Unsloth Q5_K_XL GGUF (llama.cpp).

### Single-card (1× RTX 3090) — vLLM

> ⚠️ **Cliff 2b open on `long-text*` / `long-vision` (2026-05-05)** — Genesis v7.72.2's PN59 streaming-GDN orchestrator doesn't engage on the chunked-prefill path 24 GB single-card configs are forced to take. Single-prompt prefill at >~50K may OOM. Filed at [Sandermage/genesis-vllm-patches#22](https://github.com/Sandermage/genesis-vllm-patches/issues/22). **Safe single-card paths**: `llamacpp/default` (no Cliff 2b) or single-prompt context capped at <50K. **TP=2 paths escape the cliff** entirely (see Dual-card section).

| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP tok/s | Peak VRAM | Date | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `minimal.yml` (`mem-util 0.95 max-model-len 65536`) | @noonghunna (1× 3090, x16, 350 W) | TQ3 | 64K | ~32 / ~33 | — | ~22.4 GB | 2026-05-03 | no MTP. [stiggy2k16](https://github.com/noonghunna/club-3090/issues/43) cross-rig data point — short-prompt vLLM-safe path when llama.cpp is too slow. |
| `long-vision.yml` | @noonghunna (1× 3090) | TQ3 | 145K | 50 / 66 | — | ~23.0 GB | 2026-04-30 | vision + tools + thinking. mem-util 0.95. |
| `long-text.yml` ⭐ | @noonghunna (1× 3090) | TQ3 | 180K | 50 / 67 | — | ~22.3 GB | 2026-04-30 | text-only (vision tower dropped). MTP n=3. mem-util 0.93. **Default for RAG / IDE agents below 25K accumulated ctx**. |
| `long-text.yml` | @laurimyllari (1× **4090**, AMD Ryzen 7 7800X3D, 230W cap) | TQ3 | **90K** (forced by KV-pool fit on 4090 — see Notes) | **102.96 / 103.09** | — | ~23.7 GB | 2026-05-05 | **First 4090 single-card vLLM bench** on club-3090. **Required `max-model-len` drop from 180K→90K** at default mem-util 0.92 (KV cache budget on his 24 GB 4090 is tighter than the 3090s the compose was calibrated against — likely 4090 driver/desktop overhead consumes more idle VRAM). MTP n=3 active, AL 3.34-3.45 narr / per-pos accept 92-95% / 79-84% / 62-67%. CV 2.2%/2.2%. Verify-stress hit Cliff 2b OOM at long-vision 50 MiB (sidesteps via long-text). [Issue #71](https://github.com/noonghunna/club-3090/issues/71) + [disc #62](https://github.com/noonghunna/club-3090/discussions/62#discussioncomment-16821619). |
| `long-text-no-mtp.yml` | @noonghunna (1× 3090) | TQ3 | 200K | TBD | — | ~21.0 GB | — | max-context single-shot, no MTP. Slow decode but biggest ctx window. |
| `bounded-thinking.yml` | @noonghunna (1× 3090) | TQ3 | 180K | 50 / 66 | — | ~21.7 GB | 2026-05-04 | structured-CoT FSM in reasoning channel; **recommended grammar: DeepSeek scratchpad** (PLAN/NOTE×0-15/VERDICT). Phase 3 final: **93.9% HE+ / 66.0% LCB v6** (87.4% combined, +1 net vs the andthattoo G/A/E baseline). Andthattoo G/A/E grammar also works (94.5% HE+ / 62.0% LCB / 86.9% combined, ~4× tighter think budget — pass via `extra_body`). See [STRUCTURED_COT.md](docs/STRUCTURED_COT.md). |
| `tools-text.yml` | @noonghunna (1× 3090) | fp8 | 75K | TBD | — | TBD | — | IDE-agent path that escapes the long-text Cliff 1 mech B leak (see [#16](https://github.com/noonghunna/club-3090/issues/16)). |
| `dual-dflash.yml`-shape forced TP=1 (DFlash N=5, fp8 KV, mem-util 0.96, custom_all_reduce disabled) | @efschu (1× **RTX 5090** 32 GB, AMD Ryzen 9 5950X, Debian trixie, PCIe x8, 575 W cap) | fp8 | 49K (KV-fit at 0.96 mem-util) | **126.53 / 200.11** (decode 127.98 / 204.80) | — | 31.5 GB | 2026-05-07 | **First single-5090 DFlash data point** on club-3090. AutoRound INT4 weights + DFlash N=5 draft. CV 3.0%/2.0%. **Code TPS 200 is the highest single-card number measured on the matrix** — beats single-3090 (50/67 long-text) by ~3× on code, single-4090 (102/103 at 90K) by ~2× code. Trade is ctx ceiling: 49K vs 90K-180K on 24 GB cards, due to KV-pool fit at fp8 + 32 GB total VRAM. vLLM `nightly-01d4d1ad3` (post-v7.72.2 uplift). [Issue #93](https://github.com/noonghunna/club-3090/issues/93). |
| `vllm/default` (single, MAX_MODEL_LEN=48000, mem-util 0.92, MTP n=3) | @ygafarov (1× 3090 via **oculink eGPU on PCIe x4**, AMD Ryzen AI MAX+ 395 / Strix Halo miniPC, CachyOS, 124 GB RAM, 290W cap) | TQ3 | 48K | **68.86 / 91.70** (decode 69.27 / 92.76) | — | 23.6 GB | 2026-05-09 | **Soak: ⚠ borderline** (VRAM grew 240 MiB > 200 MiB threshold, 3 turns >30s; 100% TPS retention + 0 errors + 0 silent-empty turns — x4-PCIe accretion + bus-latency under prefill, not a leak. Threshold may need an "eGPU bus class" allowance.) **First Strix-Halo-miniPC + oculink-eGPU bench** on club-3090. Single 3090 over PCIe x4 (oculink) instead of x8/x16 internal. CV 1.6%/2.7%. MTP AL 3.31, accept 76.9% (per-pos 0.918/0.772/0.616). `scheduler_reserve_full_isl=False`. Driver 595.71.05 (very new, CUDA 13.2). vLLM `nightly-01d4d1ad3`. [Issue #113](https://github.com/noonghunna/club-3090/issues/113). |

### Single-card (1× RTX 3090) — llama.cpp

| Compose | Rig | Quant | Max ctx | Narr / Code TPS | PP tok/s | Peak VRAM | Date | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `llamacpp/default` | @noonghunna (1× 3090) | Unsloth Q5_K_XL | 262K | 21 / 21 | — | ~20 GB | 2026-04-21 | bulletproof — different engine, different memory allocator, no Cliff 1 / Cliff 2. Slow decode but cliff-immune. |
| `llamacpp/concurrent` | @noonghunna (1× 3090) | Unsloth Q5_K_XL | 262K | TBD | — | TBD | — | concurrent-serving variant. |
| llama.cpp PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) MTP, custom build (`Qwen3.6-27B-MTP-Q4_K_M-GGUF` + `--spec-type mtp --spec-draft-n-max 3`) | @efschu (**2× Tesla V100-SXM2-16GB**, Xeon Gold 6154, Debian 13, custom-built llama-server docker) | Q4_K_M MTP | 100K | **49.96 / 62.46** | — | 15.6 GB/card (15,596 MiB at 100K ctx) | 2026-05-06 | **First V100 (sm_70 Volta) cross-rig data on the matrix** — only non-3090/4090/5090 GPU class tested. vLLM blocked (V100=CC 7.0, vLLM needs ≥7.5); fell back to llama.cpp via am17an's PR #22673 with a custom-built docker. **All 7 stress checks PASS including 90K NIAH** (Cliff 2 territory). 2× cards via tensor split (`-sm tensor`). MTP n=3, accept rates not in log. ~80 W/card (V100 max 300 W). [Issue #80](https://github.com/noonghunna/club-3090/issues/80). |
| llama.cpp PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) MTP, host build (`havenoammo/Qwen3.6-27B-MTP-UD-GGUF` + `--spec-type mtp --spec-draft-n-max 3` + q4_0 KV) | @lamentofhighborne (1× RTX 3090, PCIe x8, 350W) | UD-Q4_K_XL + Q8_0 MTP head | **131K** | **47.12 / 60.42** | — | ~23.1 GiB | 2026-05-07 | **First 1× 3090 llama.cpp MTP data point** on Qwen3.6-27B. Decode 47.60 / 61.71 TPS, TTFT 212 / 194 ms. **`verify-full-mtp.sh` PASS 8/8** (locally-adapted), **`verify-stress-mtp.sh` PASS 7/7 including 91K needle at 131K ctx** — pushes the documented llama.cpp MTP ctx ceiling from ~64-80K (q8_0 KV) to 131K (q4_0 KV). MTP acceptance 78.7%; recurrent 65-layer bug from froggeric's earlier MTP GGUF did **NOT** reproduce on havenoammo's UD GGUF. Native host build (no Docker), surfaced engine-coupling shortcomings in our verify/soak harness — see [Issue #85](https://github.com/noonghunna/club-3090/issues/85). |
| llama.cpp PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) MTP, host build (`froggeric/Qwen3.6-27B-MTP-GGUF` + `--spec-type mtp --spec-draft-n-max 3` + q4_0 KV) | @lamentofhighborne (1× RTX 3090, PCIe x8, 350 W) | Q4_K_M MTP | **164K** | **47.49 / 55.09** | — | ~22.2 GiB | 2026-05-07 | **Second 1× 3090 llama.cpp MTP data point on same rig** — froggeric's Q4_K_M MTP GGUF vs havenoammo's UD-Q4_K_XL above. Decode 47.91 / 55.81 TPS, TTFT 96 / 98 ms. `verify-full-mtp.sh` PASS 8/8, `verify-stress-mtp.sh` PASS 7/7 incl. 91K needle at 164K ctx. Functional MTP acceptance **86.7%**; canonical acceptance 55.3% narr / 71.2% code. **Ctx-fit ladder**: 262K OOMed MTP, 229K served without MTP, 196K initialized MTP but daemon died at 90K stress; 164K was the stable stress-passing ceiling on this rig. **Beats havenoammo on narr (47.49 vs 47.12, +0.8%) and ctx ceiling (164K vs 131K) but trails on code (55.09 vs 60.42, −9%)**. Manual long-context needles also passed at **120K** (39.39 decode TPS, 81% MTP accept) and **150K** (35.44 decode TPS, 80% MTP accept). MTP+vision incompat (per froggeric's model card); separate no-MTP+vision path passed 65K and 150K. [Issue #94](https://github.com/noonghunna/club-3090/issues/94). |

### Dual-card (2× RTX 3090, TP=2)

> NVLink auto-detection: dual-card composes now detect NVLink presence automatically. The `dual-nvlink*.yml` files are deprecated stubs that extend the unified compose with `NVLINK_MODE=force_on`. All NVLink bench rows below were measured with NVLink enabled (either via auto-detection or the deprecated stub). PCIe rows used `NCCL_P2P_DISABLE=1`.

| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP tok/s | Peak VRAM | Date | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `dual.yml` ⭐ | @noonghunna (2× 3090 PCIe, no NVLink) | fp8 | 262K (237K single-prompt verified) | 69 / 89 | — | ~23.6 GB | 2026-04-29 | tested 2-card baseline. fp8 KV, 2 streams, full feature set. **PASSES v2 continuous soak** (Cliff 2b clean). |
| `dual-turbo.yml` | @noonghunna (2× 3090 PCIe) | TQ3 | 262K | 58 / 76 per-stream (**269 TPS aggregate at 4 streams**) | — | ~19.8 GB | 2026-04-29 | TQ3 KV — 4.67× concurrency for multi-tenant agent workloads. |
| `dual-turbo.yml` ⭐ | @noonghunna (2× 3090 PCIe) | TQ3 | 262K | **81.21 / 108.20** single-stream | — | **20.0 GB** | 2026-05-05 | **v7.72.2 uplift**: Genesis pin `7b9fd319` + vLLM `01d4d1ad3` (Sander's PROD pin). 6 redundant local sidecars dropped (PN35/PN30/PN25/P78/PN34 supersede). 5 measured runs each, CV 2.3%/0.9%. AL 3.46. **VRAM −2.1 GB/card vs v7.69 baseline** (PN35 native + PN59 fold value). All 8/8 verify-full checks pass. |
| `dual-dflash.yml` | @noonghunna (2× 3090 PCIe) | fp8 | 185K | 82 / **125** | — | ~23.6 GB | 2026-04-29 | DFlash N=5 + 1.75 GB draft / card. AL ~4.4. Fastest 2-card short-prompt code path. |
| `dual-dflash.yml` | @apriori (2× 3090 + EPYC 7302P, Arch Linux, 230 W cap, NODE topology, no NVLink) | fp8 | 185K | **78.44 / 122.71** | — | ~24.0 GB | 2026-05-05 | **First EPYC + Arch cross-rig data on `dual-dflash`** — matches @noonghunna baseline within run-to-run CV (78/127 reference, narr drift +0.4 / code −3.4%). **PASSES continuous soak** (0 MiB VRAM growth, 0 errors, 0/25 silent-empty, 100% TPS retention) — first independent confirmation `dual-dflash` is Cliff 2b clean cross-rig. 3 turns >30s TTFT warning (informational). [Discussion #18](https://github.com/noonghunna/club-3090/discussions/18#discussioncomment-16819551). |
| `dual-dflash-noviz.yml` | @noonghunna (2× 3090 PCIe) | fp8 | 200K | 78 / **127** | — | ~23.8 GB | 2026-04-29 | DFlash + no vision tower. +15K ctx vs `dual-dflash`. |
| `dual-dflash-noviz.yml` | @snoby (2× **4090** PCIe — 5-GPU rig, GPUs 2,3, no NVLink, [#46](https://github.com/noonghunna/club-3090/issues/46)) | fp8 | **180K** | 92.55 / **148.99** | — | ~21.8 GB | 2026-05-04 | First non-3090 cross-rig data. **Required `max-model-len` drop from 200K→180K** vs 3090 baseline (boot OOM at 200K) — 4090 ctx-ceiling gotcha pending investigation. +17% TPS lift vs same compose on 3090 (78→92.55 narr / 127→148.99 code). |
| `dual-nvlink.yml` | @JusefPol (2× 3090 PCIe x8 + **NVLink 4× bonded**, i7-11700K, 365 W/card) | fp8 | 262K | **108.81 / 138.55** | — | ~23.7 GB | 2026-05-04 | First NVLink cross-rig data. **+58% narr / +56% code TPS vs `dual.yml` PCIe-only baseline (69 / 89)** — NVLink reduces the per-token NCCL allreduce latency floor; compounds at multi-stream. verify-stress 7/7 PASS incl. 91K needle. **PASSES v2 continuous soak** (5 sessions × 5 turns, 0 MiB growth, 100% TPS retention). MTP n=3, 65–98% per-position accept. PR [#31](https://github.com/noonghunna/club-3090/pull/31). |
| `dual-nvlink-turbo.yml` ⭐ | @danbedford (2× 3090 NVLink, 230W cap) | TQ3 | 262K | **102.34 / 133.98** | — | ~22.3 GB | 2026-05-05 | **v7.72.2-rebench** (image `nightly-01d4d1ad3`). 4-stream TurboQuant KV + NVLink. **+11% narr / +12% code vs same-rig PCIe `dual-turbo` (#73 below)** — controlled A/B on identical hardware, only `NCCL_P2P_LEVEL` differs. Custom all-reduce ENABLED (disabled on PCIe). CV 3.1% narr / 1.8% code. PR [#56](https://github.com/noonghunna/club-3090/pull/56) + [Issue #69](https://github.com/noonghunna/club-3090/issues/69). |
| `dual.yml` | @danbedford (2× 3090 NVLink-cable-attached, run as PCIe via `NCCL_P2P_DISABLE=1`, 230W cap) | fp8 | 262K | **89.24 / 114.57** | — | ~23.7 GB | 2026-05-06 | **First controlled PCIe-vs-NVLink A/B on same rig** — pair with `dual-nvlink.yml` row immediately above. **+15% narr / +15% code lift from NVLink** (#74 102/132 vs this 89/115). CV 3.8%/2.5%. **Note: this corrects the "+58% narr / +56% code" claim from JusefPol's row** — that comparison conflated NVLink lift with v7.72.2 lift (his baseline was 2026-04-29 dual.yml at 69/89 on the older image). On a strictly v7.72.2-controlled comparison NVLink adds ~15%, not ~58%. [Issue #77](https://github.com/noonghunna/club-3090/issues/77). |
| `dual-turbo.yml` | @danbedford (2× 3090 NVLink-cable-attached, run as PCIe via `NCCL_P2P_DISABLE=1`, 230W cap) | TQ3 | 262K | 91.58 / 120.00 | — | ~22.0 GB | 2026-05-06 | Companion to `dual-nvlink-turbo` row above for the controlled A/B. NVLink lift on TQ3 path: **+11% / +12%**. CV 3.2%/1.9%. [Issue #73](https://github.com/noonghunna/club-3090/issues/73). |
| `dual-nvlink.yml` | @danbedford (2× 3090 NVLink, 230W cap) | fp8 | 262K | **102.09 / 131.59** | — | ~24.0 GB | 2026-05-06 | Second cross-rig data on `dual-nvlink.yml` (vs JusefPol's earlier 108.81/138.55). Lower than JusefPol partly explained by his lower power cap (365 W/card vs 230) — on memory-bandwidth-bound decode, 2 GB/card more thermal headroom doesn't compound much, so close-but-lower at half the wattage is consistent. CV 2.6%/1.4%. [Issue #74](https://github.com/noonghunna/club-3090/issues/74). |
| `dual-dflash.yml` | @danbedford (2× 3090 PCIe NVLink-cable-attached but `NCCL_P2P_DISABLE=1`, 230W cap) | FP16 | 185K | 86.62 / **141.02** | — | ~24.0 GB | 2026-05-06 | Third cross-rig DFlash data point (after @noonghunna 82/125 + @lolren 87/142). **Code TPS 141 ties lolren's 142** as the highest measured on club-3090. CV 2.4%/5.0%. [Issue #75](https://github.com/noonghunna/club-3090/issues/75). |
| `dual-dflash-noviz.yml` | @danbedford (2× 3090 PCIe NVLink-cable-attached but `NCCL_P2P_DISABLE=1`, 230W cap) | FP16 | 200K | 88.31 / **142.79** | — | ~23.9 GB | 2026-05-06 | DFlash + no vision tower. Beats @noonghunna baseline 78/127 (+13%/+12%). CV 2.3%/2.9%. [Issue #76](https://github.com/noonghunna/club-3090/issues/76). |
| `dual-nvlink-dflash.yml` ⭐ NEW | @danbedford (2× 3090 NVLink, 230W cap, i9-11900KF) | FP16 | 185K | **101.55 / 163.33** | — | 24.06 GB/card | 2026-05-07 | **First NVLink-enabled DFlash row.** Mirrors `dual-dflash.yml` shape but enables NCCL P2P over NVLink + custom_all_reduce. **+17% narr / +16% code over his own PCIe `dual-dflash` row above** (86.62 / 141.02 — same rig with `NCCL_P2P_DISABLE=1`). Decode 102.43 / 166.54 TPS, CV 1.8%/1.9%. **PASSES continuous soak** (0 errors, 0 silent-empty, 0 MiB growth, 100% TPS retention, p50 66.71). verify-full 8/8 + verify-stress 7/7 incl. 91K Cliff 2 needle. PR [#92](https://github.com/noonghunna/club-3090/pull/92). |
| `dual-nvlink-dflash-noviz.yml` ⭐ NEW | @danbedford (2× 3090 NVLink, 230W cap) | FP16 | **188K** | **103.24 / 167.45** | — | ~23.97 GB/card | 2026-05-07 | **NVLink + DFlash + no vision** — pushes the with-vision 185K ctx ceiling to **188K** by dropping MoonViT (~0.78 GB freed). Empirically determined: 189K had only 1/3 success rate (flaky on freshly rebooted system), 188K is the stable ceiling. **+17% narr / +17% code over his own PCIe `dual-dflash-noviz` row above** (88.31 / 142.79). Decode 104.07 / 171.01 TPS, CV 2.2%/3.6%. **PASSES continuous soak** (p50 66.75, 100% retention). verify-full 8/8 + verify-stress 7/7. PR [#96](https://github.com/noonghunna/club-3090/pull/96). |
| `dual.yml`-shape **+ patched P2P drivers** (no NVLink hardware) | @aaronlockhartdev (2× 3090 PCIe x16, EPYC 7F52, Arch Linux, custom Dockerfile via [Sam McLeod's guide](https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/) — patched `aikitoria/open-gpu-kernel-modules` + vLLM `cuda.py` `return True` patch) | fp8 | 262K | **93 / 125** | — | n/a | 2026-05-07 | **First patched-driver P2P cross-rig data point** — answers the question raised in [disc #70](https://github.com/noonghunna/club-3090/discussions/70). Same-rig controlled A/B: unpatched baseline 91 narr / 114 code → patched P2P 93 / 125 = **+2% narr / +9% code**. Compared to NVLink hardware lift (+15% / +15% per @danbedford's controlled A/B): patched P2P captures **~60% of NVLink's code gain but ~13% of NVLink's narr gain** — code workloads (spec-decode K+1 verify is heavily cross-card matmul) benefit more from cross-card bandwidth than narr decode (more sequential per-token). For ~95% of dual-3090 owners without NVLink, the trade is small TPS lift vs custom kernel module + DKMS maintenance burden. [Issue #91](https://github.com/noonghunna/club-3090/issues/91). |
| `dual-dflash-noviz.yml`-shape **+ patched P2P drivers** (no NVLink hardware, custom_all_reduce ENABLED) | @aaronlockhartdev (2× 3090 PCIe x16, EPYC 7F52, Arch Linux, patched kernel module + `NCCL_P2P_LEVEL=PHB`) | fp8 | 200K | **100.47 / 160.15** (decode 101.53 / 164.44) | — | ~22.2 GB/card | 2026-05-07 | **Second patched-P2P cross-rig data point** — extends [#91 dual.yml result](https://github.com/noonghunna/club-3090/issues/91) to the DFlash + no-vision path. Same-rig controlled A/B: unpatched baseline 82.55 narr / 134.45 code → patched P2P 100.47 / 160.15 = **+22% narr / +19% code**. **Significantly larger lift than `dual.yml`-shape** (+22%/+19% here vs +2%/+9% on `dual.yml`) — DFlash's K+1 cross-card verify pattern stresses peer-bandwidth more than fp8-only `dual.yml`. **Important methodology update**: `NCCL_P2P_LEVEL=PHB` alone with the default vLLM image produced the same lift as the full vLLM `cuda.py` patch — **the in-container vLLM source patch is unnecessary**, only the kernel module patch matters. CV 4.6%/2.4%. custom_all_reduce ENABLED (vs disabled on the `dual.yml` row). [Issue #95](https://github.com/noonghunna/club-3090/issues/95) + [disc #70](https://github.com/noonghunna/club-3090/discussions/70). |
| `carnice-bf16mtp.yml` | @noonghunna (2× 3090 PCIe, no NVLink) | fp8 | 262K | **72** / **80** | — | ~22.25 GB | 2026-05-04 | **Carnice-V2-27B (Hermes agentic fine-tune) + BF16 MTP overlay**. Full 262K context, 2 streams. 71.75 narr / 80.35 code wall TPS (n=5 each, CV ~11%), MTP AL 3.02-3.14, TTFT 141ms. Patched chat template for Hermes JSON tool calls. verify-full 7/8 PASS. soak PASS. |
| `dual.yml` ⭐ | @lolren (2× 3090 PCIe + Ryzen 9 5950X, **250W/card cap**) | fp8 | 262K | **89.78 / 117.60** | — | ~22.3 GB | 2026-05-05 | **First cross-rig data on the v7.72.2 uplift** (image `nightly-01d4d1ad3`, post-PR #59). +30% narr / +32% code over @noonghunna 2026-04-29 baseline (69/89 on older image) — confirms the v7.72.2 dividend cross-rig. CV 3.3%/2.0%. MTP AL ~3.5, per-pos accept 94/84/72%. [Disc #18](https://github.com/noonghunna/club-3090/discussions/18#discussioncomment-16820303). |
| `dual-dflash.yml` | @lolren (2× 3090 PCIe + Ryzen 9 5950X, 250W cap) | FP16 | 185K | 87.10 / **142.0** | — | ~22.1 GB | 2026-05-05 | Older image `nightly-7a1eb8ac2`. +6% narr / +14% code over @noonghunna baseline (82/125) — likely Ryzen 5950X advantage on prefill. DFlash AL ~4.5, per-pos accept 93/81/68/56/48%, avg accept 69%. [Disc #18](https://github.com/noonghunna/club-3090/discussions/18#discussioncomment-16820303). |
| `bounded-thinking.yml` | @lolren (2× 3090 PCIe + Ryzen 9 5950X, 250W cap, **MTP-disabled-suspected**) | TQ3 | 180K | 64.86 / 64.96 (CV **0.1%**) | — | ~22.3 GB | 2026-05-05 | **Anomaly:** lolren reports "no spec-decode" on this run despite `bounded-thinking.yml` shipping `--speculative-config mtp n=3` by default. Near-identical narr=code TPS + extreme CV stability (0.1%) suggests MTP was inactive — likely because his image was older `nightly-7a1eb8ac2` (pre-v7.72.2 + pre-PN35). Re-test on `nightly-01d4d1ad3` should restore MTP path → expect ~50/66 narr/code with normal CV. Tracked. [Disc #18](https://github.com/noonghunna/club-3090/discussions/18#discussioncomment-16820303). |
| `dual.yml` | @JDWarner (**Mixed RTX A5000 + RTX 3090**, both **Razer Core X eGPU enclosures over Thunderbolt 3**, Intel NUC11TNH i5-1135G7, **16 GB RAM**, headless, A5000=230W cap / 3090=290W cap, PCIe **x4 Gen 3** per card) | fp8 | 262K | **56.83 / 72.47** (soak p50 93.09) | — | ~23.6 GB/card | 2026-05-09 | **Soak: ✓ PASS** (5×5, 0 errors, 0 silent-empty, 100% TPS retention, 0 MiB growth). **First TB3 dual-eGPU + mixed-arch cross-rig data**. The setup that "shouldn't work": each card on a separate TB3 controller → ~3.94 GB/s effective per card vs ~32 GB/s on PCIe x16 Gen 4 (~8× cut), mixed Ampere SKUs (workstation A5000 + consumer 3090 with different mem bandwidth + clocks), 16 GB system RAM total. **Result: matches `dual.yml` PCIe x16 baseline within run-to-run noise** — confirms decode on Qwen3.6-27B is per-card-bandwidth bound, cross-card NCCL allreduce is small enough that even an 8× link cut doesn't dominate. Extends @aaronlockhartdev's #91/#95 finding (patched-P2P only +2%/+9% on `dual.yml`) in the opposite direction: even with 8× *less* cross-card bandwidth, decode holds. MTP AL 3.39-3.52, per-pos accept 0.93/0.83/0.70 (89% avg). verify-full + verify-stress all PASS. Genesis pin `7b9fd319` (v7.72.2). [Issue #107](https://github.com/noonghunna/club-3090/issues/107). |
| `dual/docker-compose.yml` (default) | @ygafarov (**3090 via USB4 eGPU dock + 5070 Ti via OCuLink** — heterogeneous Ampere + Blackwell consumer dual-eGPU, AMD Ryzen AI MAX+ 395 / Strix Halo miniPC, CachyOS, 123 GB RAM, 290 W cap both cards, PCIe **x4** per card — USB4 ≈ 3.94 GB/s, OCuLink ≈ 7.88 GB/s) | fp8 | 200K | **65.10 / 85.81** | — | 17.1 / 15.7 GB | 2026-05-12 | **First heterogeneous Ampere + Blackwell consumer dual-eGPU on the matrix.** TP=2 bound by the slower USB4 link in allreduce + sm_86 kernels (5070 Ti spends back-half of step waiting — 91% util but only 125 W out of 290 W cap). KV pool 200K @ 1.00× concurrency — VRAM cap from the 5070 Ti's 16 GiB (model takes 13.8 GiB/card → only ~2.2 GiB left for KV on the smaller card). verify-stress 7/7 incl. **91K needle recall** (Cliff 2 clean). Soak ⚠ borderline (360 MiB > 200 MiB threshold — same eGPU-bus accretion as ygafarov's own #113 single-card row above at 240 MiB; 100% TPS retention + 0 silent-empty + 0 errors so not a leak). MTP AL 3.50, per-pos accept 0.94/0.86/0.70. CV 4.5%/1.8%. **Slower than ygafarov's own single-3090 #113 row** (68.86/91.70 at 48K) — on this rig the single-card path is recommended; the 5070 Ti adds VRAM cap pain without TPS gain. Driver 595.71.05, vLLM `nightly-1acd67a79`, no Genesis (Blackwell consumer not on allowlist). [Issue #120](https://github.com/noonghunna/club-3090/issues/120). |

### Quad-card (4× RTX 3090, TP=4)

| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP tok/s | Peak VRAM | Date | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `multi4.yml` | @whamp (4× 3090 PCIe x4/x16/x8/x16, 300 W cap, no NVLink) | fp8 | 262K | 63 / 76 | — | ~23.5 GB | 2026-05-03 | TP=4 capacity king. **6.77× concurrency at 262K**. PASSES v2 continuous soak (20 sessions, 0 MiB growth, 90.8% TPS retention). PR [#44](https://github.com/noonghunna/club-3090/pull/44). |
| `multi4.yml` | [@alanspires #127](https://github.com/noonghunna/club-3090/issues/127) (**6× 3090 VFIO-passthrough**, AMD EPYC 7313 host, all cards 250W cap, no NVLink) | fp8 | 262K | **74.93 / 92.85** | — | ~21.7 GB | 2026-05-14 | TP=4 on a 6-card rig — GPUs 4-5 free (Qwen `num_kv_heads=4` doesn't divide 6, so TP=6 invalid). **First VFIO-passthrough + virtualized data point** — scripts (verify-full / verify-stress / soak) all PASS on the virt envelope without modification. KV pool 1,774,963 tokens at 6.77× concurrency. Soak p50 122.84 / p95 161.14 across 5 multi-turn sessions, 0 errors. |
| `multi4-dflash.yml` | @whamp (4× 3090 PCIe x4/x16/x8/x16, 300 W cap) | fp8 | 262K | 64 / **104** | — | ~22.0 GB | 2026-05-03 | TP=4 + DFlash. 2.27× concurrency at 262K. PASSES v2 continuous soak (5 sessions, 0 MiB growth, 100% TPS retention). **Bench-vs-soak inversion**: bench shows DFlash wins by 37% on short-prompt code, soak shows DFlash *loses* by 47% on multi-turn agent — DFlash AL likely collapses on mixed prompts. PR [#44](https://github.com/noonghunna/club-3090/pull/44). |

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
| `multi4.yml` (TP=4) | @whamp | PASS | PASS at 262K (incl. 58K + 91K needles) | **PASS** (20 sessions, 0 MiB growth, 90.8% retention) | 2026-05-03 |
| `multi4-dflash.yml` (TP=4) | @whamp | PASS | PASS at 262K (incl. 58K + 91K needles) | **PASS** (5 sessions, 0 MiB growth, 100% retention; ⚠ 4 turns >30s; n=5 small) | 2026-05-03 |

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

---

## Gemma 4 31B (community-experimental)

Cross-rig data on Google's official Gemma 4 MTP "assistant" drafter (released 2026-05-05). PR [#41745](https://github.com/vllm-project/vllm/pull/41745) merged 2026-05-06 → today's nightly contains it natively (overlay dropped 2026-05-08). The companion compose [`dual/int8.yml`](models/gemma-4-31b/vllm/compose/dual/int8.yml) (added 2026-05-08) vendors PR [#40391](https://github.com/vllm-project/vllm/pull/40391) (rebased) + PR #42006 + PR #41991 to unlock per-token-head INT8 KV → 8.2× context lift on Ampere (32K → 262K). See announcement [discussion #67](https://github.com/noonghunna/club-3090/discussions/67) for the original Gemma 4 setup story; Phase 2 INT8 PTH validation in progress 2026-05-08.

| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP tok/s | AL | Per-pos accept (code) | Peak VRAM | Date | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| `dual.yml` (TP=2) | @noonghunna (2× 3090 PCIe, no NVLink, 230W cap) | bf16 | 32K | **108.87 / 142.25** | — | **3.94-4.04** | 92 / 79 / 68 / 59 % | 22.5 GB/card | 2026-05-05 | First Ampere consumer cross-rig data on Google MTP drafters. **+1.79× narr / +2.31× code** over baseline (61 TPS no-spec-decode same TP). **PASSES continuous soak** (100 turns, 0 errors / 0 silent-empty / 0 MiB growth, 98.3% TPS retention). bf16 KV (fp8 blocked on Ampere — see TP=1 row). PR [#41745](https://github.com/vllm-project/vllm/pull/41745) overlay + transformers 5.8.0 entrypoint. |
| `dual.yml` (TP=2) re-bench post-#41745 merge | @noonghunna (2× 3090 PCIe, 230W cap) | bf16 | 32K | **105.91 / 141.11** | — | 3.94 | (warming) | 21.5 GB/card | 2026-05-08 | Re-validated on post-merge nightly `1acd67a795...` (PR #41745 overlay dropped, transformers entrypoint upgrade dropped). Within CV of 109/142 baseline → cleanup is parity-clean. KV pool 99K tokens, 3.03× concurrency at 32K. |
| **`dual-int8.yml` (TP=2, max-num-seqs=4)** ⭐ | @noonghunna (2× 3090 PCIe, 230W cap) | **int8_per_token_head** | **98K** | **96.16 / 127.11** | — | 3.79 | (warming) | 22.2 GB/card | 2026-05-08 | **3.07× context lift over bf16 ceiling on Ampere — INT8 PTH KV unblocks Gemma 4 long-context.** Vendors PR #40391 rebased + PR #42006 + PR #41991 stacked (see `models/gemma-4-31b/vllm/patches/`). KV pool **354K tokens, 3.6× concurrency**. ~10% TPS cost vs bf16 / 32K. **PASSES verify-stress 7/7** incl. 91K Cliff-2 needle. PR #40391's per-token-head page-size fix routes via `get_padded_attention_kv_cache_shape()`; INT8 (not fp8) is the right Ampere dtype because Triton `fp8e4nv` kernel is not supported on sm_86 (Ada/Blackwell only). |
| **`dual-int8.yml` (TP=2, max-num-seqs=1, MAX_MODEL_LEN=262144)** ⭐⭐ | @noonghunna (2× 3090 PCIe, 230W cap) | int8_per_token_head | **262K (model native max)** | **95.27 / 125.93** | — | 3.93 | (warming) | 22.1 GB/card | 2026-05-08 | **8.2× context lift vs dual.yml — full Gemma 4 native context (262144) unblocked on dual 3090 Ampere.** KV pool 455K tokens, 1.74× concurrency at full 262K. **PASSES verify-stress 7/7** + **137K NIAH PASS** (correctly recalled needle from 137,557-token prompt, 5min wall, ~458 prefill TPS). Per-token TPS preserved at full max-model-len (95/126 at 262K vs 96/127 at 98K — bench prompt size dominates, not max-model-len). Override `MAX_MODEL_LEN=262144 MAX_NUM_SEQS=1`. |
| `dual-dflash.yml` (TP=2, n=7) | @noonghunna (2× 3090 PCIe, no NVLink, 230W cap) | bf16 | 32K | **95.16 / 167.55** | — | **~3.0 narr / 5.23 code** | 89 / 78 / 66 / 57 / 50 / 43 / 39 % | 22.7 GB/card | 2026-05-06 | First Ampere consumer cross-rig data on **z-lab Gemma 4 DFlash** block-diffusion drafter (vLLM PR [#41703](https://github.com/vllm-project/vllm/pull/41703) — Codex-rebased onto upstream/main). **+2.74× code / +1.56× narr** over baseline. **PASSES continuous soak** (100 turns, 0 errors / 0 silent-empty / 0 MiB growth, 98.6% TPS retention, p50 55.8 TPS — 5.8% higher than n=5). DFlash dominates MTP on **code (+18%)**; MTP wins on narrative (+15%). n-sweep: n=5 109/141 (best narr) → n=6 99/161 (knee) → **n=7 95/168 (code-optimal default)** → n=8 91/167 (dominated) → n=15 82/172 (past knee). PR #41703 overlay (12 RO-mounted files) + transformers 5.8.0 + nightly `e47c98ef`. |
| `dual-dflash.yml` (TP=2, n=7) re-bench | @noonghunna (2× 3090 PCIe, 230W cap) | bf16 | 32K | **104.48 / 176.66** (CV 2.1% / 3.6%) | — | 2.85 narr / 4.11-4.94 code | (warm) | 22.3 GB/card | 2026-05-08 | Re-validated after Phase 2 INT8 PTH session. Same overlay + same `e47c98ef` pin — modest uplift over 2026-05-06 (warm-cache + ambient variance — CV ranges overlap at +1σ). KV pool 42,848 tokens, 1.31× concurrency at 32K. Ampere upper-bound for Gemma 4 + DFlash: code-optimal at 177 TPS. Combining DFlash drafter with PR #40391 INT8 PTH KV (Phase 3, dual-dflash-int8.yml) is the next structural step — would unlock long-context code-optimal. |
| **`dual-dflash-int8.yml` (TP=2, n=7, MAX_MODEL_LEN=262144, MAX_NUM_SEQS=1)** ⭐⭐⭐ | @noonghunna (2× 3090 PCIe, 230W cap) | int8_per_token_head + drafter bf16 | **262K (model native max)** | **86.86 / 145.96** (CV 0.9% / 2.2%) | — | 5.0-5.3 long-ctx code | (warm) | 22.0 GB/card | 2026-05-08 | **8.2× context lift over `dual-dflash.yml` 32K bf16 baseline — DFlash + INT8 PTH KV unblocked on Ampere via [vLLM PR #42102](https://github.com/vllm-project/vllm/pull/42102) (our patch).** Matches the 32K bf16 baseline's code TPS within CV at 8× more context (146 vs 168 = -13% perf cost for 8× ctx). KV pool 168,178 tokens, 0.64× concurrency at full 262K — effective single-stream serving ceiling ~168K. **NIAH PASS at 98,444 tokens** (`bronze octopus 17` recalled cleanly, 157s wall = ~625 effective prefill TPS). DFlash drafter uses BF16 KV in independent pool (target uses INT8 PTH); the patch partitions them at unify-time, drafter cache_dtype overridden to "auto" in qwen3_dflash.py, FA metadata scheduler reads per-spec dtype. **n-sweep at 262K config 2026-05-08**: n=5 81.91/138.87 (-6/-5%), n=7 86.86/145.96 (default), n=8 86.63/152.06 (+0/+4% but CV 5.7% — within noise). n=7 retained as default — sweet spot didn't shift meaningfully from the 32K bf16 baseline. **DFlash code-optimal advantage preserved at long ctx**: 146 code TPS vs dual-int8.yml's 126 at 262K = +16% code (offset: -10% narr). Pin nightly `e47c98ef`; needs `dual-dflash-int8.yml` compose (still ⚠️ flagged DOES NOT BOOT until PR #42102 lands — currently requires the vllm-src patch mounts). |
| `single.yml` (TP=1) | @noonghunna (1× 3090) | bf16 / fp8 | — | **boot OOM** | — | — | — | — | 2026-05-05 | **Upstream-blocked on Ampere consumer.** bf16 KV: weights+drafter+profiling at 8K ctx + mem-util 0.95 leaves zero KV pool ("No available memory for the cache blocks"). fp8 KV: Triton `fp8e4nv not supported in this architecture` on sm_86 (Ampere supports `fp8e4b15`/`fp8e5` only); but `fp8_e5m2` is rejected by `gemma4_mm.py:1336` allowlist. Compose preserved for re-test when (a) vLLM adds Ampere-aware fp8 dispatch OR (b) PR #41745 relaxes the assert. Gemma 4 26B-A4B MoE single-card is the obvious follow-up. |
| **`dual-awq.yml` (TP=2, MAX_NUM_SEQS=4, MTP n=4)** ⭐ | @noonghunna (2× 3090 PCIe, 230W cap) | bf16 | **65K** | **104.59 / 130.56** (CV 1.7% / 0.4%) | — | 3.07 narr / 3.55-3.88 code | 0.79 / 0.59 / 0.43 / 0.31 | 19.8 GB/card | 2026-05-08 | **Cross-rig reproducer of @3dluvr's [#103 bench](https://github.com/noonghunna/club-3090/issues/103) — AWQ-4bit weights instead of AutoRound INT4.** Bypasses PR #40391 (per-token-head bug) entirely because AWQ doesn't use FP8 KV — trades weight quant precision for ctx instead of trading KV precision. Vendor: `cyankiwi/gemma-4-31B-it-AWQ-4bit` (~17 GB on disk, AWQ-pack-quantized group_size=32, asymmetric, MSE observer; routed via vLLM compressed-tensors loader → Marlin kernel). KV pool 89,228 tokens, 1.36× concurrency at 65K. Numbers comparable to dual.yml (BF16 INT4 AutoRound, 105.91/141.11) on narrative; slightly slower code at this n. Default config — multi-stream agent / RAG. |
| **`dual-awq.yml` (TP=2, MAX_NUM_SEQS=1, MTP n=8, MAX_MODEL_LEN=118304)** ⭐⭐ | @noonghunna (2× 3090 PCIe, 230W cap) | bf16 | **118K** | **101.16 / 141.90** (CV 2.5% / 0.4%) | — | 3.7 narr / 5.13 code | 0.89 / 0.77 / 0.64 / 0.54 / 0.45 / 0.37 / 0.26 / 0.21 | 19.8 GB/card | 2026-05-08 | **3.7× context lift over dual.yml's BF16 32K ceiling.** Closer match to @3dluvr's #103 anchor (113/163 at 195K with `--dtype half --async-scheduling --cudagraph_capture_sizes [9]`). Our config: `--dtype bfloat16`, default cudagraph capture sizes `[1,2,4,8,16]` (auto-clamped to single-stream). vLLM auto-estimated 195K won't fit at 0.85 mem-util (10.18 GiB KV needed vs 7.25 GiB available); 118K is the achievable ceiling at this mem-util / cudagraph budget. **NIAH PASS at 88K** (recalled "bronze octopus 17" from 88,527-token prompt, 11-tok completion in 127.1s wall, ~696 effective prefill TPS). KV pool 118,304 tokens, 1.00× concurrency. Code AL 5.13 (n=8 saturates well on code, less so on narr where AL is 3.7). **n=8 vs n=4 trade**: code +9% (130 → 142), narrative -3% (104 → 101) — n=8 dominates n=4 for code. Override `MAX_MODEL_LEN=118304 MAX_NUM_SEQS=1 MTP_N=8`. **A/B finding 2026-05-08** — three tuning flags tested individually on this rig (with bench n=3 each, sync-baseline 101.16/141.90): |
| Flag | Narr Δ | Code Δ | Verdict |
|---|---:|---:|---|
| `--dtype half` (vs bfloat16) | -3% | -2% | Ampere has NO fp16 hardware accel on sm_86 — bfloat16 is canonical |
| `--async-scheduling` | ~0% | ~0% | within CV noise (CV 6.3% on narr) |
| `cudagraph_capture_sizes [9]` | -3% | +2.5% | single-shape capture optimizes n=8 hot batch (1 base + 8 spec = 9), hurts narrative-dynamic patterns |
None close the **-13% narr / -11% code gap to 3dluvr's anchor**. Remaining gap likely rig-specific (3dluvr's EPYC 7J13 / different PCIe topology / 275W cap / etc) rather than tunable via flags. Cross-rig productionizable settings: stick with default `--dtype bfloat16`, default cudagraph, default scheduling. Custom override `cudagraph_capture_sizes [9]` worth it ONLY if you serve >95% n=8-MTP-single-stream code traffic (e.g. dedicated coding-agent endpoint) where the +2.5% code lift exceeds the -3% narrative loss. |
| `dual.yml`-shape forced TP=1 | @apnar (1× **RTX 5090** 32 GB, air-cooled, 600 W) | bf16 | 32K | **159.67 / 215.10** (decode 160.71 / 217.30) | 27.5 GB | 2026-05-07 | **First single-5090 Gemma 4 MTP data point.** First non-OOM single-card Gemma 4 result on the matrix — the 32 GB Blackwell envelope clears the 24 GB Ampere boot OOM. CV 1.9%/1.8%, peak 426 W. **+46% narr / +51% code over @noonghunna's 2× 3090 TP=2 baseline (109/142)** — single-card 5090 beats dual-3090 on Gemma 4. [Disc #67](https://github.com/noonghunna/club-3090/discussions/67#discussioncomment-16832042). |
| `dual-dflash.yml`-shape forced TP=1 (mem-util 0.96, max-model-len 12000) | @apnar (1× **RTX 5090** 32 GB, air-cooled, 600 W) | bf16 | **12K** | **150.40 / 261.06** (decode 151.16 / 264.62) | 28.8 GB | 2026-05-07 | **First single-5090 Gemma 4 DFlash data point.** Trade vs MTP row above: ~6% narr loss, **+21% code lift** (215→261). 1st-warmup TTFT outlier (73 s) suggests cudagraph warmup taking longer on first request; subsequent warmups stable at <40 ms. CV 3.6%/2.8%, peak 440 W. **Required mem-util 0.96 + max-model-len 12K** to fit BF16 weights + DFlash N=5 drafter on 32 GB — DFlash drafter footprint pushes out ctx ceiling vs MTP's 32K. [Disc #67](https://github.com/noonghunna/club-3090/discussions/67#discussioncomment-16832042). |


---

## MoE models (v0.7.3 — preview track)

First-pass numbers for the two MoE models onboarded in v0.7.3. **Preview** track = `vllm-nightly-clean` engine (no Genesis patches, no TQ3 KV, no MTP yet) — exercises the upstream MoE loader [PR #42521](https://github.com/vllm-project/vllm/pull/42521) (qwen3_5_moe weight loading, merged 2026-05-14) without other overlays. Production track (Genesis-anchored, MTP, longer context) lands in v0.7.4 after Genesis v7.73.x re-anchors on a post-#42521 nightly.

| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP tok/s | AL | Per-pos accept | Peak VRAM | Date | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| **`qwen3.6-35b-a3b/dual/preview.yml` (TP=2)** ⭐ | @noonghunna (2× 3090 PCIe, 230 W cap) | fp8_e5m2 | 16K | **182.68 / 177.45** (decode 186.98 / 186.90) | — | n/a (no drafter) | n/a | **21.94 GB/card** | 2026-05-15 | **First v0.7.3 MoE preview row on the matrix.** Engine `vllm-nightly-clean` (nightly `bf610c2f`, post-#42521). No Genesis, no MTP — exercises the upstream qwen3_5_moe loader cleanly. CV **0.3% / 0.9%** (very stable). TTFT 126 ms. GPU 0 at 99% util / 292 W, GPU 1 at 85% / 244 W. Decode TPS basically identical narr vs code (187 / 187) — characteristic of MoE memory-bandwidth-bound decode (3 B active params per forward, weights fit cache). **~2× the Qwen 3.6 27B dense `dual.yml` baseline** (89/118 wall) on the same hardware — MoE's active-params advantage on Ampere. Next steps: TQ3 KV + MTP (built-in head) after Genesis v7.73.x lands; longer ctx after upstream MoE expert dispatch overhead is measured. Compose: `models/qwen3.6-35b-a3b/vllm/compose/dual/preview.yml`. |
| `qwen3.6-35b-a3b/dual/preview-mtp.yml` (TP=2, MTP n=3, built-in head) ⚠️ | @noonghunna (2× 3090 PCIe, 230 W cap) | fp8_e5m2 | 16K | **90.36 / 115.33** (decode 91.48 / 119.65) | — | **3.44** (narr) | 0.927 / 0.810 / 0.698 | 22.72 GB/card | 2026-05-15 | **MTP MAKES THINGS SLOWER on Qwen MoE preview path.** A/B vs preview.yml above (same config, same nightly, same hardware): **−51% narr / −35% code wall TPS** despite 81.2% avg draft acceptance and AL 3.44. Per-position accept (0.927 / 0.810 / 0.698) is healthy — the draft head works correctly. Surfacing cause: asymmetric GPU util (GPU 0 at **39%** / 233 W vs GPU 1 at 79% / 191 W; non-MTP run had both at 85-99%) indicates the draft forward pass on MoE imposes inter-GPU sync overhead the acceptance gain can't amortize. Vendor warning at boot: `max_num_scheduled_tokens=4096 ... suboptimal performance ... increase max_num_batched_tokens to accommodate the additional draft token slots`. **CV 5.4% / 2.0%** (less stable than no-MTP). VRAM +0.78 GB/card vs no-MTP (MTP head workspace). **Practical implication**: for v0.7.3, route Qwen 35B-A3B users to `preview.yml` (no MTP) as the default — MTP gives worse latency despite high acceptance. Re-test after (a) Genesis v7.73.x re-anchors on a post-#42521 nightly (Cliff 2 mitigations may unblock the underlying scheduler overhead), (b) max_num_batched_tokens raised to ~12K to satisfy the vLLM warning, or (c) MTP n=2 to see if smaller spec depth changes the calculus. Compose: `models/qwen3.6-35b-a3b/vllm/compose/dual/preview-mtp.yml`. |
| **`gemma-4-26b-a4b/dual/awq.yml` (TP=2)** ⭐ | @noonghunna (2× 3090 PCIe, 230 W cap) | bf16 | 32K | **138.88 / 138.67** (decode 139.92 / 139.98) | — | n/a (no drafter) | n/a | **23.45 GB/card** | 2026-05-15 | **First v0.7.3 Gemma MoE production-track row.** Engine `vllm-nightly-clean` (nightly `bf610c2f`) + vendored [vLLM PR #40886](https://github.com/vllm-project/vllm/pull/40886) overlay (compressed-tensors AWQ MoE key remapping — applied at boot via anchor-based Python patcher in `models/gemma-4-26b-a4b/vllm/patches/vllm-pr40886-awq-moe-keys/install.sh`). Weights: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (17 GB). CV **0.2% / 0.0%** — extraordinarily stable, characteristic of MoE memory-bandwidth-bound decode. TTFT 53 ms. GPU 0 at 98% util / 360 W, GPU 1 at 59% / 299 W. **Identical narr/code wall TPS (139 / 139)** — same MoE signature as Qwen 3.6 35B-A3B preview above (per-token weight reads dominate, prompt content distribution doesn't matter). ~76% of Qwen 35B-A3B preview's narr TPS — consistent with 4 B vs 3 B active params per forward. Compose: `models/gemma-4-26b-a4b/vllm/compose/dual/awq.yml`. |
| **`gemma-4-26b-a4b/dual/awq-mtp.yml` (TP=2, MTP n=4)** ⭐⭐ | @noonghunna (2× 3090 PCIe, 230 W cap) | bf16 | 32K | **155.05 / 207.02** (decode 156.68 / 210.61) | — | **3.04 narr / 3.79 code** | 0.77 / 0.55 / 0.40 / 0.29 (narr); 50.9% avg (narr), 73.5% avg (code) | 23.50 GB/card | 2026-05-15 | **MTP boosts Gemma MoE: +12% narr / +49% code over awq.yml**. Engine `vllm-nightly-clean` + PR #40886 overlay + external `google/gemma-4-26B-A4B-it-assistant` drafter (832 MB BF16). Both GPUs symmetric at 98% util — **the external-drafter path doesn't pay the inter-GPU sync penalty that the Qwen 35B-A3B built-in MTP head pays** (see `preview-mtp.yml` row above where MTP made things 50% slower). Mechanism: external drafter is a small dense model (`Gemma4AssistantForCausalLM`, ~0.5 B params) — the draft forward bypasses MoE expert routing entirely. The 32-49% code lift confirms MTP is structurally compatible with Gemma's hybrid-SWA attention on the AWQ compressed-tensors path. **Recommended default for Gemma 26B-A4B going forward; awq.yml stays as the no-drafter A/B reference.** Compose: `models/gemma-4-26b-a4b/vllm/compose/dual/awq-mtp.yml`. |
| `gemma-4-26b-a4b/dual/docker-compose.yml` (TP=2, Intel AutoRound INT4) | @noonghunna (2× 3090 PCIe) | — | — | **boot fail (SM86)** | — | — | — | — | 2026-05-15 | **Ampere-blocked on Intel AutoRound INT4.** `moe_intermediate_size=704` is not a multiple of `group_size=128` (5.5×); Marlin K-dim alignment fails. SM86 has no WNA16 kernel for unaligned K-dim — only SM90+ Cutlass W4A8 / Machete handle arbitrary shapes. Same failure on TP=1 (no split) and TP=2 (split to 352 per rank). Both Intel quant variants (`int4-mixed-AutoRound` and `int4-AutoRound`) hit the same error. **AWQ is the Ampere path** (row above). AutoRound compose preserved here as a documented-blocker for SM90+ rigs (RTX 5090 / Pro 6000 should boot it). |


---

## Quality benches — Aider Polyglot 30

Pass rate on a curated 30-exercise subset of [aider-polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) (5 per language across cpp/go/java/javascript/python/rust, mix of easy/medium/hard). Tests **edit-format reliability** AND **algorithmic correctness** — does the model emit diffs aider can apply, AND do the resulting tests pass.

Run via [`benchlocal-cli`](https://github.com/noonghunna/benchlocal-cli) `aider-polyglot-30` pack. Different from the TPS rows above — this is a quality / agentic-coding signal, not a throughput measurement.

| Model | Compose | Rig | Pass / Total | % | Wall (real) | Wall (sum-dur) | Tokens (P+C) | Date | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| **Qwen 3.6 27B** (AutoRound INT4) | `dual.yml` (TP=2) | @noonghunna (2× 3090 PCIe, 230W cap) | **20 / 30** | **66.7%** | 19.0 min | 34.0 min | 436K + 111K = 547K | 2026-05-10 | **`enable_thinking=false`** (server-side `--default-chat-template-kwargs '{"enable_thinking": false}'` + per-request `extra_body` belt). With thinking ON: 0/30 (1500s timeout exceeded before any exercise completed — Qwen burns the token budget on hidden CoT). Per-language: cpp 3/5 · go 4/5 · **java 4/5** · js 4/5 · python 2/5 · rust 3/5. `threads=2`. |
| **Gemma 4 31B** (Intel AutoRound INT4) | `dual.yml` (TP=2) | @noonghunna (2× 3090 PCIe, 230W cap) | 17 / 30 | 56.7% | 19.2 min | 19.2 min | 380K + 72K = 452K | 2026-05-10 | Default thinking off (Gemma 4's chat template requires explicit `enable_thinking=true` to enable). Per-language: cpp 2/5 · **go 4/5** · java 1/5 · **js 4/5** · python 3/5 · rust 3/5. `threads=2`. |

**Notable**:
- Qwen 3.6 27B beats Gemma 4 31B by **+10 pp** despite 4 GB fewer parameters. Java is the biggest swing (4/5 vs 1/5 — `affine-cipher` specifically tripped Gemma).
- Gemma is meaningfully faster wall-clock (sum-of-exercise-durations 19 vs 34 min), suggesting Qwen produces longer per-turn answers but they convert to passes more reliably.
- Qwen with thinking ON is unusable for this kind of bench on club-3090 hardware: hits the 1500s subprocess timeout cap (now bumped to 2700s) before completing any exercises — the hidden CoT eats the per-exercise token budget. **Set `enable_thinking=false` for any agentic / multi-turn workload.**

**To re-run cross-rig**: `bash scripts/quality-test.sh --pack aider-polyglot-30 --enable-sandboxed-packs` against your endpoint. See [docs/QUALITY_TEST.md](docs/QUALITY_TEST.md) for the harness setup.

---

## Head-to-head — Qwen 3.6 27B vs Gemma 4 31B on dual 3090 (TP=2, MTP)

Single-rig comparison run 2026-05-10 on @noonghunna's 2× 3090 PCIe rig (230 W cap each). Both models served via `dual.yml` compose with MTP speculative decode and `enable_thinking=false`. Aim: agentic-workload pick guidance.

### Config & pin delta — what differs between the two runs

| | Qwen 3.6 27B | Gemma 4 31B |
|---|---|---|
| Weights quant | Lorbus AutoRound INT4 (mtp.fc preserved BF16) | Intel AutoRound INT4 |
| MTP type | Built-in head, `method: mtp` n=3 | External 0.5B BF16 drafter (`gemma-4-31b-it-assistant`) via [vllm#41745](https://github.com/vllm-project/vllm/pull/41745), n=4 |
| KV cache | `fp8_e5m2` | BF16 |
| Max ctx | 262144 (262K) | 32768 (32K — BF16 KV ceiling on 2× 24 GB; use `gemma-int8` for 262K) |
| `--gpu-memory-utilization` | 0.92 | 0.92 |
| TP | 2 | 2 |
| vLLM nightly | `01d4d1ad` (2026-05-04) | `1acd67a7` (2026-05-08) — newer, post Gemma 4 MTP merge |
| Genesis pin | v7.64 (loaded via setup.sh) | n/a (Gemma 4 doesn't use Genesis patches) |
| thinking | OFF (server-side `--default-chat-template-kwargs`) | OFF (Gemma 4 default; needs explicit `enable_thinking=true` to enable) |

### TPS (canonical `bench.sh`: 800-word essay narrative + quicksort C++ code)

| Bench | Qwen 3.6 27B | Gemma 4 31B | Gemma vs Qwen |
|---|---:|---:|---:|
| Narrative decode TPS | 68.83 | **108.46** | **+57%** |
| Code decode TPS | 87.54 | **139.88** | **+60%** |
| TTFT narrative | 151 ms | **70 ms** | ~2× faster |
| TTFT code | 125 ms | **62 ms** | ~2× faster |
| VRAM/card | 23.7 GiB | **22.5 GiB** | −0.8 GiB |
| MTP mean accept length | 3.30–3.56 | 3.05–3.96 (warming) | similar |
| MTP avg accept rate | 76.5–85.2% | 51.2–73.9% (warming) | Qwen slightly higher when warm |
| CV (narr / code) | 3.9% / 4.1% | 1.4% / 1.0% | Gemma noticeably more stable |
| GPU power (per card) | 308 W / 263 W | 351 W / 296 W | Gemma uses ~13% more power |

### Quality — 8-pack `quality-test.sh --full` (150 scenarios)

Same endpoint, same thinking-off default. **Note** — `BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1` env var required for hermesagent-20 to reach host vLLM from the sandbox container (auto-set by `scripts/quality-test.sh` for localhost URLs since [83bf73d](https://github.com/noonghunna/club-3090/commit/83bf73d)).

| Pack | Qwen 3.6 27B | Gemma 4 31B | Δ (Gemma − Qwen) | Workload type |
|---|---:|---:|---:|---|
| toolcall-15 | 10/15 (67%) | 9/15 (60%) | **−7 pp** | Multi-tool call sequencing |
| instructfollow-15 | 13/15 (87%) | 13/15 (87%) | tied | Format-constraint following |
| structoutput-15 | 13/15 (87%) | 13/15 (87%) | tied | JSON / CSV schema emission |
| dataextract-15 | 15/15 (**100%**) | 15/15 (**100%**) | tied | Information extraction |
| reasonmath-15 | 6/15 (40%) | 6/15 (40%) | tied | Math (thinking-off cost = identical) |
| bugfind-15 | 12/15 (80%) | **14/15 (93%)** | **+13 pp** | Code debugging |
| hermesagent-20 | 10/20 (50%) | **12/20 (60%)** | **+10 pp** | Multi-turn agentic (Hermes harness, v0.7.4 grader) |
| cli-40 | 21/40 (52%) | 20/40 (50%) | −2 pp | CLI task completion |
| **TOTAL** | **100/150 (67%)** | **102/150 (68%)** | **+1 pp** | |

### Quality — Aider Polyglot 30 (per-language breakdown)

Cross-reference [Quality benches — Aider Polyglot 30](#quality-benches--aider-polyglot-30) above. Same `dual.yml` configs, same compose, single-shot.

| Language | Qwen 3.6 27B | Gemma 4 31B | Δ |
|---|---:|---:|---:|
| C++ | 3/5 (60%) | 2/5 (40%) | −1 |
| Go | 4/5 (80%) | 4/5 (80%) | tied |
| Java | **4/5 (80%)** | 1/5 (20%) | **−3** (Gemma fell apart on `affine-cipher`) |
| JavaScript | 4/5 (80%) | 4/5 (80%) | tied |
| Python | 2/5 (40%) | **3/5 (60%)** | **+1** |
| Rust | 3/5 (60%) | 3/5 (60%) | tied |
| **Total** | **20/30 (66.7%)** | 17/30 (56.7%) | **−10 pp** |

### Matched-config rebench (later same day) — most of the "Gemma +60% TPS" gap was config drift

After the first run, the two legs differed on (a) vLLM nightly, (b) KV dtype, (c) max context, (d) MTP n. Re-ran both with everything matched (vLLM `1acd67a7` · `int8_per_token_head` KV · 262K ctx · MTP n=4 · TP=2 · mem-util 0.92 · max-num-seqs=2 · AutoRound INT4 W4A16 group_size 128). Compose files added: `models/qwen3.6-27b/vllm/compose/dual/int8.yml` (new, this work) + `models/gemma-4-31b/vllm/compose/dual/int8.yml` (existing, `SPEC_N_MAX` parameterized).

**Notable side-finding:** Qwen3-Next + INT8 PTH KV (vllm#40391) **works end-to-end on our stack** — first validation. Real chat completions, MTP acceptance ~75%, **+27%/+33% TPS** vs Qwen's prior fp8 KV row above (most of which is the newer vLLM nightly + INT8 PTH KV combo, not architectural). Qwen `int8.yml` is now a candidate shipping path.

#### Matched-config performance (2026-05-10 PM)

| Metric | Qwen 3.6 27B int8.yml | Gemma 4 31B int8.yml | Gemma vs Qwen |
|---|---:|---:|---:|
| Narrative decode TPS | 88.05 | **98.12** | **+11%** |
| Code decode TPS | 120.17 | **129.34** | **+8%** |
| TTFT narrative | 152 ms | **79 ms** | **−48%** |
| TTFT code | 137 ms | **79 ms** | **−42%** |
| CV narrative / code | 1.6% / 5.3% | 2.1% / 0.6% | both clean |
| MTP avg accept (warm) | 75.3% | 70.6% | Qwen +5 pp |
| MTP per-pos rates (n=4) | 0.94 / 0.83 / 0.69 / 0.55 | 0.89 / 0.75 / 0.65 / 0.55 | Qwen earlier-pos better |

#### Concurrency + VRAM (matched config)

| Metric | Qwen | Gemma | Δ |
|---|---:|---:|---:|
| VRAM used / card | 21.4 GiB | 22.5 GiB | +1.0 GiB Gemma |
| Available KV cache / card | 11.10 GiB | 10.82 GiB | -0.28 GiB Gemma |
| **GPU KV cache size (tokens)** | **605,495** | 466,892 | **+30% Qwen** |
| max_num_seqs (configured) | 2 | 2 | tied |
| **Max concurrency @ 262K/req** | **2.31×** | 1.78× | +30% Qwen |
| Practical concurrency @ 100K/req | ~6.0× | ~4.7× | +28% Qwen |
| Practical concurrency @ 32K/req | ~18.9× | ~14.6× | +30% Qwen |

#### Matched-config headline

- **Per-stream speed:** Gemma wins **~10% decode TPS + ~45% TTFT** — that's the model-architecture-attributable gap. Dense attention prefills faster than DeltaNet hybrid.
- **Throughput at scale:** Qwen wins **~30% more KV pool** — supports more concurrent streams at the same per-stream context budget. Lighter weights + smaller per-token KV from the hybrid attention.
- **Workload pick:** Single-agent / chat → Gemma. Multi-tenant fleet → Qwen.
- **Original mismatched table below preserved** for the "shipping defaults" comparison. The +60% gap from this morning was nearly all vLLM-nightly + KV-class drift, not model intrinsics.

### Headline takeaways (original mismatched-config bench, kept for reference)

- **TPS:** Gemma 4 31B is **~60% faster** across both narrative and code, plus ~2× faster TTFT and 0.8 GiB less VRAM per card. For throughput-bound workloads at ≤32K ctx, Gemma 4 wins decisively at this config. For long-context (>32K), Qwen 3.6 27B's 262K ceiling + fp8 KV is the only option in this matchup (use `gemma-int8` for Gemma at 262K — separate compose, not benched here).
- **Quality aggregate:** Effectively tied — **180-scenario combined total: Qwen 120/180 (66.7%), Gemma 119/180 (66.1%)**. Within noise. Neither model is generically "better"; the choice is workload-shaped.
- **Quality by workload:** Gemma wins agentic + bug-fix (`bugfind +13 pp`, `hermesagent +10 pp`). Qwen wins polyglot code editing (`aider-polyglot +10 pp`, driven mostly by Java). Tool-call ordering favors Qwen (`toolcall +7 pp`). Five packs are dead-even.
- **reasonmath 6/15 (40%) on BOTH** — the thinking-off cost is identical and model-independent. Math problems with strict-format expectations need thinking-on (and the `aider-polyglot-30` row above shows what Qwen-with-thinking does to wall time — 0/30 hit the 1500s subprocess cap, now 2700s).
- **Pin/config delta:** Gemma's vLLM nightly is 4 days newer (post Gemma 4 MTP merge), KV is BF16 vs Qwen's fp8_e5m2, and Gemma uses an external 0.5B drafter vs Qwen's built-in MTP head. Same `dual.yml` topology, same `--gpu-memory-utilization 0.92`, same TP=2.

### How to reproduce on your rig

```bash
# Qwen leg
gpu-mode 27b                                                    # bring up at :8010
RUNS=3 WARMUPS=1 bash scripts/bench.sh                          # TPS
bash scripts/quality-test.sh --full                             # 8-pack quality
benchlocal-cli run --pack aider-polyglot-30 \
  --endpoint http://localhost:8010 --model qwen3.6-27b-autoround

# Gemma leg (cycle GPU)
gpu-mode gemma                                                  # bring up at :8030
URL=http://localhost:8030 MODEL=gemma-4-31b-autoround \
  RUNS=3 WARMUPS=1 bash scripts/bench.sh
URL=http://localhost:8030 MODEL=gemma-4-31b-autoround \
  bash scripts/quality-test.sh --full
benchlocal-cli run --pack aider-polyglot-30 \
  --endpoint http://localhost:8030 --model gemma-4-31b-autoround
```

End-to-end takes ~3 hours on dual 3090. quality-test.sh auto-detects the endpoint and served-model-id, so `URL`/`MODEL` overrides are only needed when running against non-default ports.
