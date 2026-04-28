# Qwen3.6-27B — Changelog

Dated history for Qwen3.6-27B configs in this repo. Combines the single-card and dual-card timelines (both were previously separate repos; consolidated here 2026-04-28).

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
