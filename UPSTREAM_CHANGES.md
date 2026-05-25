# Upstream Changes for Club-3090

This branch ports the live `/opt/ai/club-3090` custom ik-llama work onto the current upstream `master`/`v0.8.5` tree.

## Summary

- Adds the PRISM-PRO-DQ preset family for `qwen3.6-27b` on the ik-llama path.
- Adds the APEX-MTP preset family for `qwen3.6-35b-a3b` on the ik-llama path.
- Wires every new preset into `scripts/launch.sh` so `launch.sh` can discover, select, port-map, and name the containers correctly.
- Extends `scripts/lib/profiles/compose_registry.py` so the upstream profile catalog exposes the new presets with the right topology, workload, context, and KV metadata.
- Fixes the `ik-llama/apex-mtp-compact` registry metadata to report `kv_format="q4_0"` so the catalog matches the compose header and the actual single-card compact lane.
- Reclassifies `ik-llama/iq4ks-two-stage` as `tool-heavy` instead of `fast-chat`, which matches the compose header and the intended code-first/two-stage use case.

## New PRISM-PRO-DQ Presets

### Single-card lanes

- `ik-llama/prism-pro-dq-mtp`
  - File: `models/qwen3.6-27b/ik-llama/compose/single/prism-pro-dq-mtp.yml`
  - Topology: single 3090
  - Drafter: native MTP `n=4`, `p-min=0.0`
  - KV: `q4_0`
  - Default context: `122880`
  - Positioning: lean text-first PRISM lane for fast single-card MTP evaluation

- `ik-llama/prism-pro-dq-long`
  - File: `models/qwen3.6-27b/ik-llama/compose/single/prism-pro-dq-long.yml`
  - Topology: single 3090
  - Drafter: native MTP `n=5`, `p-min=0.0`
  - KV: `q4_0`
  - Default context: `180000`
  - Positioning: single-card long-context compromise with better context headroom than the lean MTP lane

- `ik-llama/prism-pro-dq-two-stage`
  - File: `models/qwen3.6-27b/ik-llama/compose/single/prism-pro-dq-two-stage.yml`
  - Topology: single 3090
  - Drafter: two-stage `ngram-mod + MTP fallback`
  - KV: `q4_0`
  - Default context: `200000`
  - Positioning: code-heavy PRISM evaluation on the two-stage speculative path

### Dual-card lanes

- `ik-llama/prism-pro-dq-dual`
  - File: `models/qwen3.6-27b/ik-llama/compose/dual/prism-pro-dq-mtp.yml`
  - Topology: dual 3090
  - Drafter: native MTP `n=4`, `p-min=0.0`
  - KV: `q4_0`
  - Default context: `196608`
  - Positioning: text-first dual-card PRISM serving with more context headroom than the multimodal lane

- `ik-llama/prism-pro-dq-dual-vision`
  - File: `models/qwen3.6-27b/ik-llama/compose/dual/prism-pro-dq-mtp-vision.yml`
  - Topology: dual 3090
  - Drafter: native MTP `n=5`, `p-min=0.0`
  - KV: `q4_0`
  - Default context: `196608`
  - Vision: enabled via `mmproj-F16`
  - Positioning: dual-card multimodal PRISM lane with embedded vision preserved

## New APEX-MTP Presets

- `ik-llama/apex-mtp-compact`
  - File: `models/qwen3.6-35b-a3b/ik-llama/compose/single/apex-mtp-compact.yml`
  - Topology: single 3090
  - Drafter: native MTP `n=2`, `p-min=0.0`
  - KV: `q4_0`
  - Default context: `200000`
  - Positioning: practical single-card APEX entry point when the `I-Quality` file is too tight for a usable 24 GB speculative/KV budget

- `ik-llama/apex-mtp-compact-long`
  - File: `models/qwen3.6-35b-a3b/ik-llama/compose/single/apex-mtp-compact-long.yml`
  - Topology: single 3090
  - Drafter: native MTP `n=2`, `p-min=0.0`
  - KV: `q8_0`
  - Default context: `262144`
  - Positioning: highest-context single-card APEX compact lane that still stayed fast on the test rig

- `ik-llama/apex-mtp-quality-dual`
  - File: `models/qwen3.6-35b-a3b/ik-llama/compose/dual/apex-mtp-quality.yml`
  - Topology: dual 3090
  - Drafter: native MTP `n=5`, `p-min=0.0`
  - KV: `q8_0`
  - Default context: `196608`
  - Positioning: long-context dual-card APEX Quality lane that keeps full-context `q8` KV on the paired rig

## Launcher and Registry Integration

- `scripts/launch.sh`
  - Adds compose-path entries for every PRISM and APEX lane.
  - Adds model ownership for the new `qwen3.6-27b` PRISM and `qwen3.6-35b-a3b` APEX variants.
  - Adds engine-family registration so the new presets route through the llama.cpp/ik-llama launch path.
  - Adds `KVCALC=SKIP` coverage so the new ik-llama presets don't fall into unsupported KV calculator paths.
  - Adds launch ordering, default ports, and default container names for the entire PRISM/APEX family.
  - Carries forward the already-existing `ik-llama/iq4ks-two-stage` launch wiring that was missing from upstream `launch.sh`.

- `scripts/lib/profiles/compose_registry.py`
  - Registers all PRISM/APEX presets so they show up in profile-backed selection flows.
  - Sets the new presets' workload classes, topology, context limits, and default ports.
  - Corrects `ik-llama/apex-mtp-compact` to `kv_format="q4_0"`.
  - Changes `ik-llama/iq4ks-two-stage` to `workload="tool-heavy"` to match its two-stage/code-oriented purpose.

## Files Added

- `models/qwen3.6-27b/ik-llama/compose/single/prism-pro-dq-mtp.yml`
- `models/qwen3.6-27b/ik-llama/compose/single/prism-pro-dq-long.yml`
- `models/qwen3.6-27b/ik-llama/compose/single/prism-pro-dq-two-stage.yml`
- `models/qwen3.6-27b/ik-llama/compose/dual/prism-pro-dq-mtp.yml`
- `models/qwen3.6-27b/ik-llama/compose/dual/prism-pro-dq-mtp-vision.yml`
- `models/qwen3.6-35b-a3b/ik-llama/compose/single/apex-mtp-compact.yml`
- `models/qwen3.6-35b-a3b/ik-llama/compose/single/apex-mtp-compact-long.yml`
- `models/qwen3.6-35b-a3b/ik-llama/compose/dual/apex-mtp-quality.yml`
