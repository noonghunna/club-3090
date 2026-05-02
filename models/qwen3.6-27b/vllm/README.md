# Qwen3.6-27B on vLLM

The recommended path for this model. Full features, validated end-to-end via `verify-full.sh`.

## What's here

- [`compose/`](compose/) — docker-compose files for single-card, dual-card, and quad-card configs
- [`patches/`](patches/) — engine-specific patches (CUDA graph capture fix, research artifacts, Marlin pad notes)

## Pick a compose

```bash
# Single-card default — 48K + Genesis v7.66 + TurboQuant 3-bit + vision
cd compose && docker compose up -d

# Single-card frontier 145K — vision on, Cliff 2 still applies single-prompt >50K
cd compose && docker compose -f docker-compose.long-vision.yml up -d

# Single-card frontier 180K — text-only, same Cliff 2 caveat
cd compose && docker compose -f docker-compose.long-text.yml up -d

# Single-card bounded-thinking — 180K + structured-CoT grammar in reasoning (~30× cheaper think on coding)
cd compose && docker compose -f docker-compose.bounded-thinking.yml up -d

# Single-card IDE-agent / long-prompt — 75K + fp8 + no vision (Cline / Cursor / Copilot / RAG)
cd compose && docker compose -f docker-compose.tools-text.yml up -d

# Dual-card default — 262K + fp8 + MTP + vision (best general dual-card)
cd compose && docker compose -f docker-compose.dual.yml up -d

# Dual-card multi-tenant — 4 streams at 262K
cd compose && docker compose -f docker-compose.dual-turbo.yml up -d

# Dual-card peak code TPS — DFlash N=5 (78/128 narr/code)
cd compose && docker compose -f docker-compose.dual-dflash.yml up -d

# Quad-card single endpoint — two NVLink pairs, PP=2 x TP=2, 262K + vision, no MTP
cd compose && docker compose -f docker-compose.quad.yml up -d

# Quad-card paired replicas — router on :8020, direct pairs on :8021/:8022
cd compose && docker compose -f docker-compose.quad-pairs.yml up -d
```

See [the model README](../README.md) and the hardware pages for the full matrix. `quad-pairs.yml` is measured at 370 / 445 aggregate narrative/code TPS under direct pair load; `quad.yml` remains unpublished until benchmarked on the inspected four-card host.

## Patches

vLLM doesn't ship cleanly out of the box for this model + this hardware combination. We mount three layers of patches at container boot:

1. **`patches/patch_tolist_cudagraph.py`** — fixes a CUDA graph capture crash in TurboQuant continuation prefill (single-card stacks). Auto-applied by container entrypoint.
2. **`patches/genesis/`** (gitignored, fetched by `setup.sh`) — Sandermage's [Genesis patch tree](https://github.com/Sandermage/genesis-vllm-patches). Loads at boot via `python3 -m vllm._genesis.patches.apply_all`.
3. **Marlin pad-sub-tile-n** (`/opt/ai/vllm-src/`) — our [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) patch fork, volume-mounted for multi-card composes. See [`patches/README.md`](patches/) for setup instructions. Drops out as a dependency when our PR lands upstream.

## Tuning

See [`/docs/engines/VLLM.md`](../../../docs/engines/VLLM.md) for engine-general tuning (mem-util, KV type, spec-decode config, power cap, Genesis env-opts).

For model-specific quirks (the prefill cliffs, why we pick AutoRound, etc.), see [`../INTERNALS.md`](../INTERNALS.md).
