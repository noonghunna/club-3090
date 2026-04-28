# Qwen3.6-27B on vLLM

The recommended path for this model. Full features, validated end-to-end via `verify-full.sh`.

## What's here

- [`compose/`](compose/) — docker-compose files for single-card and dual-card configs
- [`patches/`](patches/) — engine-specific patches (CUDA graph capture fix, research artifacts, Marlin pad notes)

## Pick a compose

```bash
# Single-card default — 48K + Genesis v7.14 + TurboQuant 3-bit + vision (recommended for ≥20K + tool agents)
cd compose && docker compose up -d

# Single-card chat-only — 20K + fp8 + MTP (fastest TPS at small ctx)
cd compose && docker compose -f docker-compose.fast-chat.yml up -d

# Single-card long-prompt — 75K + fp8 + no vision (best for RAG / summarization on 1 card)
cd compose && docker compose -f docker-compose.tools-text.yml up -d

# Dual-card default — 262K + fp8 + MTP + vision (best general dual-card)
cd compose && docker compose -f docker-compose.dual.yml up -d

# Dual-card multi-tenant — 4 streams at 262K
cd compose && docker compose -f docker-compose.dual-turbo.yml up -d

# Dual-card peak code TPS — DFlash N=5 (78/128 narr/code)
cd compose && docker compose -f docker-compose.dual-dflash.yml up -d
```

See [the model README's compose table](../README.md#compose-variants-vllm) for the full matrix with TPS numbers and use cases.

## Patches

vLLM doesn't ship cleanly out of the box for this model + this hardware combination. We mount three layers of patches at container boot:

1. **`patches/patch_tolist_cudagraph.py`** — fixes a CUDA graph capture crash in TurboQuant continuation prefill (single-card stacks). Auto-applied by container entrypoint.
2. **`patches/genesis/`** (gitignored, fetched by `setup.sh`) — Sandermage's [Genesis patch tree](https://github.com/Sandermage/genesis-vllm-patches). Loads at boot via `python3 -m vllm._genesis.patches.apply_all`.
3. **Marlin pad-sub-tile-n** (`/opt/ai/vllm-src/`) — our [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) patch fork, volume-mounted for dual-card composes only. See [`patches/README.md`](patches/) for setup instructions. Drops out as a dependency when our PR lands upstream.

## Tuning

See [`/docs/engines/VLLM.md`](../../../docs/engines/VLLM.md) for engine-general tuning (mem-util, KV type, spec-decode config, power cap, Genesis env-opts).

For model-specific quirks (the prefill cliffs, why we pick AutoRound, etc.), see [`../INTERNALS.md`](../INTERNALS.md).
