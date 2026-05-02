# Qwen3.6-27B — Changelog

Dated history for Qwen3.6-27B configs in this repo. Combines the single-card and dual-card timelines (both were previously separate repos; consolidated here 2026-04-28).

## 2026-05-02 — Genesis v7.66 + Cliff 1 mech B closed ⭐

Genesis pin bump `753344b` → `fc89395` (v7.66 dev tip). Cliff 1 mech B closed across all 4 TQ3 composes via two local backports:

**PN25 v3 import-time backport** (`patch_pn25_genesis_register_fix.py`):
Sander's PN25 mechanisms — both v7.65 `@torch.library.custom_op` (which crashes on `infer_schema` inside dynamo trace) AND v7.66 `direct_register_custom_op` + `Library("genesis", "FRAGMENT")` (which crashes on `instantiate_user_defined_class_object` inside dynamo trace) — fail on TP=1 spawn. Our v3 text-patches `vllm/model_executor/layers/activation.py` to register the op at module-import time, BEFORE any trace context exists. Survives both Sander mechanisms because it sidesteps registration-during-trace entirely.

**PN30 dst-shaped temp fix** (`patch_pn30_dst_shaped_temp_fix.py`):
Sander's PN30 `a9977d8` corrupts DS conv state row strides by raw-memcpying a compact `.contiguous()` source-tail into a strided destination. Our fix builds a destination-shaped temp inside `collect_mamba_copy_meta` (where both source AND destination block IDs are known) and does the strided copy correctly. Diagnosis credit: ChatGPT/Codex CLI cross-check.

**Validation matrix (v7.66 + local sidecars, verify-stress.sh 7-probe ladder):**

| Compose            | Ctx | mem-util | Probes | Failure          |
|--------------------|-----|----------|--------|------------------|
| long-text          | 180K | 0.95 | 6/7 | Cliff 2 architectural |
| long-vision        | 145K | 0.95 | 6/7 | Cliff 2 architectural |
| bounded-thinking   | 180K | 0.95 | 6/7 | Cliff 2 architectural |
| dual-turbo (TP=2)  | 262K | 0.85 | 6/7 | Cliff 2 architectural |

Backoff from 214K + 0.985 → 180K + 0.95 (long-text/bounded-thinking) and 198K + 0.98 → 145K + 0.95 (long-vision) was needed to give activation headroom for the PN12+PN25 FFN pool residence + PN30 dst-shaped temp lifecycle. Vision tower's persistent ~1 GB tightens long-vision further.

**Sander's v7.66 PN33 partial:** PN33 (default ON) closes BOOT-time profile_run workspace_lock issue, but the runtime decode workspace_lock at `turboquant_attn.py:1350:_decode_attention` still fires on TP=1. Cross-rig data sent to Sander via [discussion #19 reply](https://github.com/noonghunna/club-3090/discussions/19#discussioncomment-16785590).

**Sander's v7.66 PN31:** still doesn't fit on 24 GB. Per-shape persistent buffers + PN12+PN25 pool residence outpace activation budget at `chunk_fwd_o`. Lower mem-util (0.95) is sufficient to close the 25K tool-RETURN path PN31 was designed to fix without needing PN31 itself.

**Sidecars retained on master (4):**
- `patch_pn25_genesis_register_fix.py` (PN25 v3 import-time, TP=1 only)
- `patch_pn30_dst_shaped_temp_fix.py` (replaces Sander's compact `.contiguous()`)
- `patch_workspace_lock_disable.py` (PN33 narrowed but didn't close runtime decode path)
- `patch_tolist_cudagraph.py` (cudagraph capture fix, unchanged)

Per-config + cross-rig results in `results/v0.20-migration/v766-pin-results.summary` and the per-compose `*-pn30.summary` files.

## 2026-05-01 PM — vLLM v0.20 + Genesis v7.65 dev tip migration ⭐

Master pin migration from `vllm-openai:nightly-07351e088...` (`0.19.2rc1.dev205`) + Genesis v7.64 (`64dd18b`) to `vllm-openai:nightly-7a1eb8ac2ec...` (`0.20.1rc1.dev16+g7a1eb8ac2`) + Genesis v7.65 dev tip (commit `d89a089`). v0.20's revised TQ FA prefill paths ([vllm#40092](https://github.com/vllm-project/vllm/pull/40092)) and Genesis v7.65's PN26b sparse-V kernel + PN17 FA2 lse-clamp + P38B/P15B in-source hooks together close Cliff 1 mech B sub-mechanisms that forced the dev205 backoffs. Three of our local sidecars (`patch_pn12_ffn_pool_anchor.py`, `patch_pn12_compile_safe_custom_op.py`, `patch_fa_max_seqlen_clamp.py`) replaced by Genesis-native equivalents.

**Sidecars retained:**
- `patch_workspace_lock_disable.py` (NEW) — relaxes vllm#39226 strict assertion to one-shot WARNING. Sandermage's P98 covers the same surface but auto-skips on v0.20 (drift-marker false-positive). Drop when Sandermage ships marker fix.
- `patch_tolist_cudagraph.py` — unchanged.

**Sidecars dropped:**
- `patch_pn12_ffn_pool_anchor.py` → covered natively by PN12 on v0.20
- `patch_pn12_compile_safe_custom_op.py` → covered by Genesis PN25
- `patch_fa_max_seqlen_clamp.py` → covered by PN17 + P15B

**Mamba block_size cap fix:** v0.20 enforces `long_prefill_token_threshold >= block_size`; on hybrid Mamba+TQ3 the engine forces `block_size=4128`. Bumped `GENESIS_PROFILE_RUN_CAP_M` and `GENESIS_PREALLOC_TOKEN_BUDGET` from 4096 → 4128 across all 5 main composes.

**Restored ceilings (vs dev205 backoff):**

| Variant | Before (dev205+v7.64) | After (v0.20+v7.65 dev) | Δ |
|---|---|---|---|
| `long-text.yml` | 185K + 0.975 | **214K + 0.985** | +29K (+16%) |
| `long-vision.yml` | 140K + 0.95 | **198K + 0.98** | +58K (+41%) |
| `bounded-thinking.yml` | 185K + 0.975 | **214K + 0.985** | +29K (+16%) |
| `tools-text.yml` | 75K + 0.97 (fp8) | **75K + 0.97** (unchanged) | flat |
| `dual-turbo.yml` | 262K + 0.85 | **262K + 0.85** (full v7.65 PROD env-vars) | flat ctx, +9% TPS |

**Bench results (n=5, 3 warmups + 5 measured, canonical narr+code prompts):**

| Variant | Narr wall_TPS (CV) | Code wall_TPS (CV) | TTFT | AL | Avg accept | VRAM | KV pool tokens |
|---|---|---|---|---|---|---|---|
| `long-text.yml` 214K | 49.74 (2.6%) | 67.39 (2.7%) | 154/155 ms | 3.34-3.51 | 78-84% | 23.4 GB | 284,832 (1.03×) |
| `long-vision.yml` 198K | 50.32 (2.3%) | 66.12 (4.1%) | 159/158 ms | 3.40-3.56 | 79-85% | 22.3 GB | 264,192 (1.02×) |
| `bounded-thinking.yml` 214K | 49.77 (1.4%) | 65.80 (2.3%) | 155/154 ms | 3.25-3.61 | 75-87% | 21.7 GB | 284,832 (1.03×) |
| `tools-text.yml` 75K (fp8) | 53.32 (2.3%) | 69.66 (1.4%) | 150/153 ms | 3.53-3.59 | 84-87% | 22.2 GB | 104,000 (1.05×) |
| `dual-turbo.yml` 262K (TP=2) | 58.33 (2.9%) | 76.01 (4.5%) | 112/110 ms | 3.39-3.51 | 79-84% | 19.8 GB/card | **1,523,232 (4.67×)** |

**Concurrent throughput on dual-turbo** (canonical code prompt, 2 runs per stream):

| Streams | Total TPS | Per-stream mean | Per-stream CV | Speedup |
|---|---|---|---|---|
| 1 | 74.03 | 73.99 | 3.7% | 1.00× |
| 2 | 128.74 | 65.57 | 14.1% | 1.74× |
| 3 | 126.52 | 55.41 | 31.9% | 1.71× |
| **4** | **269.03** | **74.05** | **3.1%** | **3.63×** |

n=4 lands at near-single-stream per-stream TPS — true parallel decoding of full-context streams, not interleaved. The n=2/n=3 dips are scheduler artifacts on small bench sizes (high CV at n=3 confirms interleave behavior). Practically: dual-turbo serves either 1 stream at 76 TPS or 4 streams at 269 TPS aggregate.

**Validation:** verify-full ✅ 8/8 on every variant. verify-stress 33K AND 50K tool-prefill ✅ PASS on every variant (the 50K cliff that fired on EVERY dev205 config no longer reproduces). Branch `v0.20-migration`; bench captures at `results/v0.20-migration/`.

## 2026-04-30 PM — Cliff 1 closes; long-text 218K + long-vision 198K

PN12 was silently no-op'd on dev205+ via anchor drift (same bug class as P101). Genesis `apply_all` reported "PN12 applied" while live `vllm/model_executor/layers/activation.py` retained the vanilla `SiluAndMul.forward_cuda`. Local sidecar `patch_pn12_ffn_pool_anchor.py` repairs it; bundled Genesis tree carries the fix via [PR #13](https://github.com/Sandermage/genesis-vllm-patches/pull/13). Combined with local `patch_fa_max_seqlen_clamp.py` (P104 FA softmax_lse clamp), Cliff 1 closes on TQ3 paths.

**New shipped ceilings:**
- `long-text.yml`: 205K → **218K** at 0.985 mem-util (no vision, no override). Engine ceiling vLLM-reported 218K. Verify-stress + verify-full pass; MTP AL 2.66; VRAM 23.7/24 GB.
- `long-vision.yml`: 192K → **198K** at 0.98 mem-util (vision on). Engine ceiling vLLM-reported 198K. 0.985 + vision reopens Cliff 1 (more goes to KV at the cost of activation budget; vision tower's persistent ~1 GB makes 0.98 the right balance).
- `--num-gpu-blocks-override 50` no longer needed at 0.985 — anchor-fixed PN12 cuts allocator churn enough that natural activation budget at higher mem-util is sufficient on text-only path.
- 0.99 mem-util ruled out — driver/system reserves ~440 MiB; vLLM startup check fails at 0.99.

**Cliff 2 unchanged.** Single-prompt >50–60K still OOMs in DeltaNet GDN. Both long-* variants stay "steady-state accumulation across many turns, not single-shot big prompts."

**Variants stay distinct:** `docker-compose.yml` (48K, below both cliffs, fast boot) and `tools-text.yml` (FP8 path for IDE agents) remain valuable for their respective use cases. Four-variant menu kept; the long-* options now ship at higher ceilings.

Branch `cliff1-fa-clamp` carries the changeset; commits `41eabac` (PN12 sidecar) → `f3e5b52` (218K bisection) → `26e5f65` (docs).

## 2026-04-29 — Genesis v7.62.x + PN8 enabled on FP8 paths

Sandermage shipped Genesis v7.62.x (commit `917519b`) on 2026-04-29 with PN8 (MTP draft online-quant propagation — backport of vllm#40849) targeting the FP8+MTP memory-headroom problem. We benched the patch across all 5 single-card composes that use Genesis:

| Compose | KV | mem-util | PN8 effect | TPS Δ | Verdict |
|---|---|---|---|---|---|
| `tools-text.yml` (75K, fp8) | fp8 | 0.97 | **−900 MiB at boot · Cliff 1 closes** ⭐ | −7% code | **PN8 enabled** |
| `fast-chat.yml` (20K, fp8) | fp8 | 0.95 | **−800 MiB at boot** | −4.7% code | **PN8 enabled** |
| `docker-compose.yml` (48K, TQ3) | TQ3 | 0.92 | no-op (already plenty of headroom) | −3% / −5% | PN8 not enabled |
| `long-vision.yml` (192K, TQ3) | TQ3 | 0.98 | KV pool +230 MiB, engine ceiling 192K → 198K, but Cliff 1 still fires | −5% | PN8 not enabled (commented in env, opt-in) |
| `long-text.yml` (205K, TQ3) | TQ3 | 0.98 | no effect (engine ceiling capped by block-size divisor at 206K) | not benched | PN8 not enabled |

Why split-decision: the **Cliff 1 OOM that ampersandru hit on `long-vision.yml`** is an FFN intermediate-buffer activation peak (138 MiB allocate at `intermediate_size=17408 × max-num-batched-tokens=4128`), not a draft-model footprint. PN8's quant-config propagation doesn't reach that buffer on TQ3 paths. On FP8 paths the draft head's own footprint shrinks meaningfully — that's where the win is.

**Cross-rig data + analysis posted to Sandermage**: [single-3090 #1 comment 4343317153](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1#issuecomment-4343317153).

Other v7.62.x items relevant to us (not yet benched here):
- **PN11** (Quentin-M, vllm#41142 streaming tool-call IndexError fix) — applies cleanly via the auto-detected REC; planned to enable in tools-text + fast-chat next pass.
- **TurboQuant k8v4 unlocked on hybrid GDN via P4 + P98** — Sandermage's A5000 measurement +1.9%; we'll bench on dual.
- **Per-GPU recommendation system** (`vllm/_genesis/gpu_profile.py`) — boot log now lists `[REC]/[OFF]` per patch on this card. Nice ergonomics.

## 2026-04-28 (post-launch) — llama.cpp Q3_K_XL + Docker compose + stress-test findings + VRAM diagram

- **First measured TPS for UD-Q3_K_XL on this stack:** 21.22 narr / 20.79 code @ 262K context + vision (single 3090, q4_0 KV). VRAM 20.17 GB / 24 GB at boot. Lower than memory's 28.5 baseline (Q4_K_M, 2026-04-23 on llama.cpp commit `9ab47e7d8`) — investigating mainline regression vs current `0d0764dfd`. ngram-mod path measured at 22.04 / 26.11 (+25% on code, draftless via `--spec-type ngram-mod`).
- **llama.cpp Docker compose** at `models/qwen3.6-27b/llama-cpp/compose/`:
  - `docker-compose.yml` — single slot, 262K ctx, q4_0 KV, vision via mmproj. Uses `ghcr.io/ggml-org/llama.cpp:server-cuda`.
  - `docker-compose.concurrent.yml` — 4 parallel slots, 192K ctx pool, vision. Multi-tenant variant.
- **All three llama.cpp configs pass verify-full + verify-stress** on this stack. Crucial finding: llama.cpp R1 (Q4_K_M @ 262K + q4_0 KV), Q3_K_XL @ 262K + vision, and Q4_K_M + ngram-mod @ 32K all clear the 90K needle ladder + 25K tool-prefill checks. **No Cliff 1, no Cliff 2** — the prefill OOMs that bite vLLM single-card 192K configs don't fire in llama.cpp on this model. Trade is the ~2-3× lower TPS (21 vs 51-55 vLLM). Reframes our launch positioning around "vLLM dual = max throughput, llama.cpp single = max robustness." Single feature gap: llama.cpp doesn't peel `<think>` into `reasoning_content` (parser issue, not model). Tool calling, streaming, vision, output quality all clean on `--jinja`.
- **`models/qwen3.6-27b/README.md`** — added "VRAM allocation across configs" section with embedded `docs/img/vram-budget-dual.svg`. Per-card stacked bars across 7 configs (3 single, 4 dual) showing weights / KV / vision / DFlash draft / activations / free headroom on the 24 GB budget. Visualizes the TP=2 unlock concretely.
- **`models/qwen3.6-27b/llama-cpp/README.md`** — quant table updated. UD-Q3_K_XL marked ⭐ as our default with citation to Benjamin Marie's [Kaitchup Q3.6-27B GGUF eval](https://kaitchup.substack.com/p/summary-of-qwen36-gguf-evals-updating) — independent H100-validated pick of Q3_K_XL as the optimal accuracy/efficiency/footprint balance, complementary to our 3090 speed measurements.

## 2026-04-28 — Split verify-full.sh → verify-full.sh (fast) + verify-stress.sh (boundary)

Recent additions to `verify-full.sh` (#8 tool-prefill OOM, #9 cascade detection, #10 MTP AL) made the script slow — the longctx needle ladder (#7) alone could run 5+ min, and the full 10-check suite was approaching 10 min. Awkward for "is the stack functional" iteration during dev work.

Split into two scripts:

- **`verify-full.sh`** — fast functional smoke, 8 checks, ~1-2 min. Contains: server reachability, Genesis patches applied, basic completion (Paris), tool calling, streaming, thinking mode, output quality / cascade detection, MTP acceptance length. **Run after every config change** to confirm the stack still serves cleanly.

- **`verify-stress.sh`** — boundary-case stress test, 2 checks, ~5-10 min. Contains: long-context needle ladder (4 depths up to 90K tokens) + tool-response prefill OOM (~25K-token mock tool message). **Run before publishing or when investigating prefill-OOM regressions** specifically.

Same env-var conventions (URL, MODEL, CONTAINER, SKIP_LONGCTX, SKIP_TOOL_PREFILL, PREFILL_TARGET_CHARS). Both pass on the new club-3090 default + dual.yml + dual-turbo.

## 2026-04-28 — Dual-card re-bench on club-3090 substrate (revised TPS numbers)

The published dual-card TPS numbers were measured pre-v714 formalization (April 24-25 timeframe), on a different vLLM nightly + Genesis tree. Re-benched all 4 dual composes on the club-3090 unified substrate (dev205 + Genesis v7.51-stable + Marlin pad fork mounted) to reconcile.

Also: caught a stale mount path in `dual-turbo.yml` — predecessor mounted `patch_tolist_cudagraph.py` from `../patches/genesis/` (where it lived in the old qwen36-dual-3090 layout); club-3090 has it at `../patches/` (top-level). Fixed before measurement; container booted clean. All other composes already had correct paths.

**Measured numbers (3 warmup + 5 measured per prompt arm, narr 1000 tok + code 800 tok):**

| Compose | Narr TPS (CV) | Code TPS (CV) | TTFT | MTP/DFlash AL | VRAM/card | Was claimed | Δ% |
|---|---|---|---|---|---|---|---|
| dual.yml | 69.05 (2.3%) | 88.58 (3.4%) | 145ms | 3.38-3.48 | 23.6 GB | 71/89 | -3% / -1% |
| dual-turbo.yml (now TQ3) | 53.65 (2.7%) | 72.93 (2.7%) | 113ms | 3.41-3.42 | 24.1 GB | 58/69 (k8v4) | -8% / +6% |
| dual-dflash.yml | 81.94 (4.3%) | 124.93 (5.8%) | 138ms | 4.10-4.35 | 23.6 GB | 78/128 | +5% / -2% |
| dual-dflash-noviz.yml | 78.19 (2.5%) | 126.99 (2.2%) | 143ms | 4.24-4.37 | 23.8 GB | 77/124 | +2% / +2% |

Net: most numbers within run-to-run variance. The dual.yml fp8 path is essentially unchanged. dual-turbo's TQ3 swap (from k8v4) cost ~8% narrative but recovered ~6% code — net trade for ~9× the KV pool capacity.

All 4 composes pass `verify-full.sh` functional checks (skipped longctx ladder on the DFlash variants for time; fp8 + MTP variants pass full 10/10 including the 90K-token needle). Updated all docs (README compose table, USE_CASES.md, dual.yml header, dual-turbo.yml header) with the measured numbers.

## 2026-04-28 — Add long-vision + long-text composes (R3' / R3''' from formal v714 round)

Previously the 192K and 205K opt-in tiers were documented as "edit max-model-len + mem-util in docker-compose.yml" — fragile for reproducibility against published bench numbers. Promoted both to dedicated compose files:

- **`docker-compose.long-vision.yml`** — TQ3 + Genesis P65 + MTP n=3 + 192K + 0.98 mem-util + vision tower active. Matches R3' bench row (50.93 narr / 67.69 code TPS, AL 3.40-3.58 80-86% accept). Container name: `vllm-qwen36-27b-long-vision`. Same prefill caveats as edit-the-default did.
- **`docker-compose.long-text.yml`** — Same config + `--language-model-only` + max-model-len 205K. Matches R3''' (50.11 narr / 65.84 code TPS). Container name: `vllm-qwen36-27b-long-text`.

Trade-off: 2 more compose files (now 11 vs 9). Net: every published bench row from the v714 formalization round (R2, R3, R3', R3''', R4, R6, R7) now boots cleanly with one `-f` flag — no error-prone editing for users who want to reproduce. R1 (eager) and R5 (longctx) stay deleted (obsolete, not niche).

Header references updated: model README compose table, USE_CASES.md frontier-context section, default's variant matrix, vllm/README.md "Pick a compose" code block.

## 2026-04-28 — Repo migration to club-3090

Configs migrated from the predecessor repos (`qwen36-27b-single-3090`, `qwen36-dual-3090`) into this repo's `models/qwen3.6-27b/vllm/compose/` directory. File renames:

| Old path | New path |
|---|---|
| `qwen36-27b-single-3090/compose/docker-compose.yml` | `models/qwen3.6-27b/vllm/compose/docker-compose.yml` |
| `qwen36-27b-single-3090/compose/docker-compose.fast-chat.yml` | `models/qwen3.6-27b/vllm/compose/docker-compose.fast-chat.yml` |
| `qwen36-27b-single-3090/compose/docker-compose.tools-text.yml` | `models/qwen3.6-27b/vllm/compose/docker-compose.tools-text.yml` |
| `qwen36-27b-single-3090/compose/docker-compose.no-genesis-mtp.yml` | `models/qwen3.6-27b/vllm/compose/docker-compose.no-genesis-mtp.yml` |
| `qwen36-27b-single-3090/compose/docker-compose.minimal.yml` | `models/qwen3.6-27b/vllm/compose/docker-compose.minimal.yml` |
| `qwen36-dual-3090/compose/docker-compose.yml` | `models/qwen3.6-27b/vllm/compose/docker-compose.dual.yml` |
| `qwen36-dual-3090/compose/docker-compose.turbo.yml` | `models/qwen3.6-27b/vllm/compose/docker-compose.dual-turbo.yml` |
| `qwen36-dual-3090/compose/docker-compose.dflash.yml` | `models/qwen3.6-27b/vllm/compose/docker-compose.dual-dflash.yml` |
| `qwen36-dual-3090/compose/docker-compose.dflash-noviz.yml` | `models/qwen3.6-27b/vllm/compose/docker-compose.dual-dflash-noviz.yml` |
| `qwen36-27b-single-3090/patches/patch_tolist_cudagraph.py` | `models/qwen3.6-27b/vllm/patches/patch_tolist_cudagraph.py` |

Functional content identical — only paths changed. Anyone with scripts referencing the old paths needs to update; the old repos still serve the old paths read-only.

## 2026-04-28 — Compose rename: v7.14 is the zero-arg vLLM default (single-card)

**Breaking change** at the time (mitigated by being on a small-audience repo).

- `docker-compose.v714.yml` → **`docker-compose.yml`**. Running `docker compose up -d` (with no `-f` flag) now boots the production-safe TQ3 + Genesis v7.14 + MTP n=3 + 48K + 0.92 config.
- The previous zero-arg default (fp8 + MTP n=3 + 20K) → **`docker-compose.fast-chat.yml`**. Pick this one when you only need ≤20K context and want the maximum-TPS chat path (~5-7% faster than the new default).
- `docker-compose.longctx-experimental.yml` → **deleted**. Superseded by the default's opt-in 128K + 0.95 tier.

## 2026-04-28 — Prefill-OOM tests + safe v714 default

Triggered by ampersandru's production OOM report ([noonghunna/qwen36-27b-single-3090#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1)) — a Hermes-class agent fetching ~25K tokens of news as a tool reply at 192K context crashed the engine.

**Discovered two distinct activation-memory cliffs** on this hardware:
- **Cliff 1** — TurboQuant attention scratch + tool-response prefill, fires on ≥25K-token tool messages at high `--gpu-memory-utilization`. OOM site: TurboQuant forward (dequant scratch + mid_o/output buffers), ~138 MiB allocate.
- **Cliff 2** — DeltaNet/GLA recurrent state buffer, fires on any single prompt above ~50-60K tokens regardless of mem-util. OOM site: `fla.ops.chunk.chunk_gated_delta_rule_fwd_h.h.new_empty(...)`. NT grows linearly with prompt length; chunked-prefill doesn't help.

**Shipped:**
- `verify-full.sh` extended from 7 → 10 checks: #8 tool-response prefill OOM, #9 output quality / cascade detection, #10 MTP acceptance length threshold.
- `verify-full.sh #7` long-context needle ladder treats engine HTTP 400 (oversize ctx rejection) as a clean "skipped at this depth" rather than a failure.
- vLLM single-card default lowered to **48K + 0.92** — below both cliffs. All 10 checks pass.
- README/docs document the full opt-in matrix (64K → 205K) with safe single-prompt + tool-prefill envelopes per tier.
- Three-layer defense documented: vLLM `--max-model-len` HTTP 400 rejection + agent-framework truncation + system-prompt limits.

TPS unchanged at the new default: 51 narr / 68 code TPS (CV ~2.3%). Hardware-bound.

## 2026-04-28 — Inherited prefill-OOM tests on dual-card

The dual-card stack adopted the new `verify-full.sh` checks for safety even though TP=2 + fp8 KV (the dual-card default) gives much wider safety margins than single-card TQ3 KV — the cliffs are not active failure modes on dual hardware.

All compose variants pinned to `vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08` (= vLLM `dev205+g07351e088`). Previously some tracked `:nightly` and drifted with upstream.

## 2026-04-27 — Full-matrix re-bench + substrate unification (single-card)

Discovered and fixed four real compose drift bugs during a complete re-bench cycle:

- **Image split**: composes had drifted across two different vLLM image pins. All six unified to `vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08`.
- **`eager.yml` config drift**: shipped with `gpu-memory-utilization=0.92` and `max-model-len=131072` while [@ampersandru](https://github.com/ampersandru)'s actual measurement was `0.97` + `125000`. As-shipped failed to boot. Compose deleted entirely.
- **`v714.yml` mount path**: `patch_tolist_cudagraph.py` was mounted from a wrong path. Fixed.
- **Bench harness regression**: `scripts/bench.sh` had silently dropped the code-prompt arm. Restored.
- **Genesis exoneration**: A/B between default + Genesis vs no-Genesis confirmed Genesis is performance-neutral on the fp8+MTP path. Cross-rig confirmed by [u/sudeposutemizligi](https://www.reddit.com/r/LocalLLaMA/) on TP=2 + dev45 + no Genesis (55 narrative / 68 code, same hardware class).

## 2026-04-27 — Removed: docker-compose.eager.yml (single-card)

Originally proposed by [@ampersandru](https://github.com/ampersandru) as a 125K path that bypasses the cudagraph bug class via `--enforce-eager`. Re-bench cycle measured 25.5 narr / 32.3 code — strictly dominated by `longctx-experimental.yml` at the same 125K context (38/50 TPS). Compose removed.

## 2026-04-27 — Patch hardening (single-card)

- `patches/patch_tolist_cudagraph.py` was silently failing on (a) any non-docker setup (hardcoded `dist-packages` path) and (b) any vLLM nightly past the one we initially tested against. Fixed: patcher auto-discovers vLLM via `import vllm` and uses single-line regex anchors. Bug reported by [@3dluvr](https://github.com/3dluvr) in single-3090 #1.

## 2026-04-25 — Genesis v7.14 (Sandermage upstream)

Genesis v7.14 shipped with the **P65** patch root-causing [vllm#40880](https://github.com/vllm-project/vllm/issues/40880) — the silent tool-call cascade bug under MTP × TurboQuant × cudagraph. P65 forces `cudagraph_mode=PIECEWISE` for spec-decode → eager continuation runs the correct branch.

This shipped as a workaround. The proper fix is a custom multi-query Triton kernel (P67) that handles K+1 query against compressed cached KV under cudagraph capture — designed-but-not-implemented as of v7.14.

The dual-card **Turbo variant** (`docker-compose.dual-turbo.yml`) loads Genesis v7.14 with P64/P65/P66/P68/P69 enabled via env vars. ~25% per-stream TPS regression vs fp8 default but **4.59× concurrency at full 262K vs fp8's 2.36×** — aggregate throughput exceeds fp8 above ~3 concurrent streams.

We adjusted two consumer-Ampere knobs vs Sandermage's A5000-class defaults: `gpu-memory-utilization 0.92 → 0.85` and `max-num-batched-tokens 8192 → 4128`. Without these, deep-prefill (60K+) requests OOM on 24 GB cards.

## 2026-04-22 — DFlash N=5 + Qwen3.6-27B (Luce z-lab fork)

[Luce z-lab](https://github.com/luce-spec)'s DFlash spec-decode draft model for Qwen3.6-27B clears verify-full.sh on dual-3090. Single-stream **78 / 128 TPS narr/code** — substantially faster than MTP n=3's 71 / 89.

Two DFlash variants ship in the dual-card path:
- `docker-compose.dual-dflash.yml` — vision + DFlash N=5 + 185K context
- `docker-compose.dual-dflash-noviz.yml` — text-only + DFlash N=5 + 200K context

Required workaround: vllm#40334 (DFlash `combine_hidden_states` dtype mismatch) is open. Compose sets `--dtype bfloat16` to match the draft's training dtype.

## 2026-04-15 — Marlin pad-sub-tile-n (PR #40361 — our patch)

Filed [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) — fixes a crash in vLLM's Marlin INT4 kernel where output features < 64 cause `GPTQ_MARLIN_MIN_THREAD_N (64) > out_features` on TP=2.

Status: **OPEN, MERGEABLE**, awaiting maintainer review. Until it lands, dual-card composes volume-mount our [patched fork](https://github.com/noonghunna/vllm) at `/opt/ai/vllm-src/`.

## 2026-04-08 — Initial dual-card release

vLLM-based dual-3090 recipe shipping at TP=2 with fp8 KV + MTP n=3, full feature parity with the single-card project plus the Marlin pad workaround. Was its own repo at the time; now lives here.

## Earlier — Initial single-card release

Initial single-card release shipped a `docker-compose.longctx-experimental.yml` at 125K with `cudagraph_mode=NONE` as the long-context option. v7.14 superseded this; deprecated and removed during 2026-04-27 cleanup.
