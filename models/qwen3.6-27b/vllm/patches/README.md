# Patches for Qwen3.6-27B on vLLM

This directory contains the model + engine-specific patches that compose variants apply at boot. As of **2026-05-05 (v7.72.2-uplift branch)** the local-sidecar surface has shrunk significantly — Genesis v7.72.2 absorbed most of what we used to carry here.

## Active patches

| Path | Used by | Purpose |
|---|---|---|
| `genesis/` | every Genesis-loaded compose | Sandermage's [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches) tree (gitignored; checked out at the pin in `scripts/setup.sh` — currently `7b9fd319`, v7.72.2) |
| `vllm-marlin-pad/marlin.py` + `MPLinearKernel.py` | every TP=2 compose with AutoRound INT4 | Vendored vLLM PR #40361 fix for Marlin `GPTQ_MARLIN_MIN_THREAD_N=64` blocking sub-tile-N output dim shards |
| `carnice-chat-template.jinja` | `docker-compose.carnice-bf16mtp.yml` only | Patched chat template for Carnice-V2-27B's Hermes-style tool format |

## Composes that load Genesis

7 composes currently bootstrap the Genesis tree + apply_all entrypoint:

- `docker-compose.yml` (single-card default, 48K ctx)
- `docker-compose.dual-turbo.yml` (TP=2, TQ3 KV, MTP — daily-driver)
- `docker-compose.long-text.yml` (TP=1, 180K ctx, MTP)
- `docker-compose.long-text-no-mtp.yml` (TP=1, 200K ctx, no MTP)
- `docker-compose.long-vision.yml` (TP=1 with vision tower)
- `docker-compose.bounded-thinking.yml` (TP=1, FSM bounded-thinking)
- `docker-compose.tools-text.yml` (TP=1, 75K ctx, no MTP)

Composes that do **not** mount Genesis (intentionally — Genesis-free fallback / different attention path / minimal config): `docker-compose.dual.yml` (fp8 KV TP=2 — kept Genesis-free as a debugging fallback for cross-engine bisect), `dual4.yml`, `dual4-dflash.yml`, `dual-dflash.yml`, `dual-dflash-noviz.yml`, `dual-nvlink.yml`, `minimal.yml`, `carnice-bf16mtp.yml`, `qwopus-bf16mtp.yml`.

## What was retired in v7.72.2-uplift (2026-05-05)

The following local sidecars were **deleted** because Genesis v7.72.2 natives supersede them:

| Retired sidecar | Genesis native that supersedes |
|---|---|
| `patch_inputs_embeds_optional.py` | **PN35** (vllm#35975 backport, default-on since v7.69) |
| `patch_pn30_dst_shaped_temp_fix.py` | **PN30 v7.68** dst-shaped temp (default-on since v7.69) |
| `patch_pn25_genesis_register_fix.py` | **PN25** opaque-op pool (default-on since v7.66) |
| `patch_tolist_cudagraph.py` | **P78** TQ tolist capture-guard (Sander explicitly noted in v7.72 CHANGELOG: "Deprecated external probes removed from 4 launch scripts; P78/PN14 supersede") |
| `patch_workspace_lock_disable.py` | **PN34** workspace-lock relax (default-on since v7.66) |
| `patch_pr40798_workspace.py` | (negative-result research artifact, no compose ever mounted it) |

The dual-turbo bench on v7.72.2 with these sidecars dropped is within noise of the version that still mounted them: 81.21 vs 82.09 narr wall TPS, 108.20 vs 109.91 code wall TPS, VRAM 20.0 GB/card both ways. Genesis's natives carry the load.

## vLLM PR #40361 — Marlin pad-sub-tile-n (still vendored)

**What it fixes:** Marlin's `GPTQ_MARLIN_MIN_THREAD_N=64` blocks any W4A16 shard where per-rank out-dim falls below 64. Hits on Ampere SM 8.6 with AutoRound INT4 quants under TP=2.

**Status:** PR open at https://github.com/vllm-project/vllm/pull/40361, awaiting maintainer review. Until it lands, we vendor the patched files in `vllm-marlin-pad/` and RO-mount them over the container's copies. No host filesystem dependency.

```yaml
volumes:
  - ../patches/vllm-marlin-pad/marlin.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:ro
  - ../patches/vllm-marlin-pad/MPLinearKernel.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:ro
```

When PR #40361 lands upstream, the entire `vllm-marlin-pad/` directory + the four compose mount lines get deleted, and TP=2 composes just use upstream nightly. Drift recovery procedure documented in `vllm-marlin-pad/README.md`.

## Genesis tree (`genesis/`)

Sandermage's [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches) — a runtime monkey-patcher for vLLM that fixes Qwen3-Next architectural bugs (hybrid TurboQuant gate, GDN streaming, MTP propagation, tool-parser edge cases, structured-output spec-decode timing, etc).

Setup:
- Cloned by `bash scripts/setup.sh qwen3.6-27b` at the pinned SHA (currently **`7b9fd319`** = v7.72.2)
- Override the pin via `GENESIS_PIN=<sha-or-tag>` env var
- Gitignored from this repo (we don't vendor someone else's tree)
- Pin-gate at boot enforces compatibility with the running vLLM version (allowlist-clean for both `0.20.1rc1.dev16+g7a1eb8ac2` and `0.20.2rc1.dev9+g01d4d1ad3`; this branch ships the latter)

## Cross-rig findings tracker

- **PN59 streaming-GDN** doesn't engage on chunked-prefill on Ampere consumer (1× RTX 3090). Eligibility check rejects calls with non-None `chunk_indices`/`chunk_offsets`, which vLLM's `--max-num-batched-tokens 4128` always populates. Cliff 2b unchanged on `long-text.yml`. Filed as [Sandermage/genesis-vllm-patches#22](https://github.com/Sandermage/genesis-vllm-patches/issues/22) with reproducer + 4 fix proposals.
