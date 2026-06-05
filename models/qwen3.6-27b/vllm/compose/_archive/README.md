# Archived vLLM composes (Genesis / purged-nightly path)

Archived 2026-06-05 (#254 Genesis cleanup). These composes were pinned to the
purged `vllm-nightly-clean` / `vllm-nightly-mtp` / `vllm-nightly-full` nightly
images (404 on Docker Hub) and rode the Genesis patch stack, which is on hold
upstream. Kept here verbatim for possible future revival; NOT maintained, NOT in
the registry, hidden from the launchers.

## Revival

1. `git mv` the file back to its original `compose/<topology>/<quant>/<file>` path
   (restores the `../../../patches/` relative mounts).
2. Re-add its `_entry(...)` to `scripts/lib/profiles/compose_registry.py` (repoint
   `engine=` to a functional v0.22.0 engine + re-validate).

## Archived entries

| Slug | Orig compose path | Engine | Drafter | KV | TP | Status |
|---|---|---|---|---|---|---|
| `vllm/bounded-thinking` | `models/qwen3.6-27b/vllm/compose/single/autoround-int4/bounded-thinking.yml` | vllm-nightly-mtp | qwen-mtp-builtin | turboquant_3bit_nc | 1 | deprecated |
| `vllm/default` | `models/qwen3.6-27b/vllm/compose/single/autoround-int4/tq3-mtp.yml` | vllm-nightly-mtp | qwen-mtp-builtin | turboquant_3bit_nc | 1 | deprecated |
| `vllm/dual-bf16` | `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/bf16.yml` | vllm-nightly-clean | qwen-mtp-builtin | bf16 | 2 | deprecated |
| `vllm/dual-carnice-bf16mtp` | `models/qwen3.6-27b/vllm/compose/dual/carnice-bf16mtp/bf16-mtp.yml` | vllm-nightly-clean | qwen-mtp-builtin | fp8_e5m2 | 2 | caveats |
| `vllm/dual-int8` | `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/int8.yml` | vllm-nightly-full | qwen-mtp-builtin | int8_per_token_head | 2 | deprecated |
| `vllm/dual-qwopus-bf16mtp` | `models/qwen3.6-27b/vllm/compose/dual/qwopus-bf16mtp/bf16-mtp.yml` | vllm-nightly-clean | qwen-mtp-builtin | fp8_e5m2 | 2 | preview |
| `vllm/dual-tq3-mtp` | `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/tq3-mtp.yml` | vllm-nightly-mtp | qwen-mtp-builtin | turboquant_3bit_nc | 2 | deprecated |
| `vllm/dual-tq3-mtp-genesis` | `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/tq3-mtp-genesis.yml` | vllm-nightly-mtp | qwen-mtp-builtin | turboquant_3bit_nc | 2 | deprecated |
| `vllm/dual-tq3-nomtp` | `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/tq3-nomtp.yml` | vllm-nightly-mtp | - | turboquant_3bit_nc | 2 | deprecated |
| `vllm/dual-turbo` | `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/turbo.yml` | vllm-nightly-mtp | qwen-mtp-builtin | turboquant_3bit_nc | 2 | deprecated |
| `vllm/dual4` | `models/qwen3.6-27b/vllm/compose/multi4/autoround-int4/fp8-mtp.yml` | vllm-nightly-clean | qwen-mtp-builtin | fp8_e5m2 | 4 | production |
| `vllm/long-text` | `models/qwen3.6-27b/vllm/compose/single/autoround-int4/long-text.yml` | vllm-nightly-mtp | qwen-mtp-builtin | turboquant_3bit_nc | 1 | deprecated |
| `vllm/long-text-no-mtp` | `models/qwen3.6-27b/vllm/compose/single/autoround-int4/long-text-no-mtp.yml` | vllm-nightly-mtp | - | turboquant_3bit_nc | 1 | deprecated |
| `vllm/long-vision` | `models/qwen3.6-27b/vllm/compose/single/autoround-int4/long-vision.yml` | vllm-nightly-mtp | qwen-mtp-builtin | turboquant_3bit_nc | 1 | deprecated |
| `vllm/tools-text` | `models/qwen3.6-27b/vllm/compose/single/autoround-int4/tools-text.yml` | vllm-nightly-clean | qwen-mtp-builtin | fp8_e5m2 | 1 | deprecated |

## DFlash path — archived 2026-06-05 (#327)

`vllm-nightly-dflash` rode the same purged `nightly-e47c98ef` SHA as the deprecated
`vllm-nightly-full` (404 on Docker Hub). DFlash spec-decode is also blocked on Qwen3-Next
(DeltaNet rollback) per the repo hardware truths. Revive only if both lift.

| Slug | Orig compose path | Engine | Drafter | KV | TP | Status |
|---|---|---|---|---|---|---|
| `vllm/dual-dflash` | `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/dflash.yml` | vllm-nightly-dflash | zlab-qwen-dflash | fp16 | 2 | deprecated |
| `vllm/dual-dflash-noviz` | `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/dflash-noviz.yml` | vllm-nightly-dflash | zlab-qwen-dflash | fp16 | 2 | deprecated |
| `vllm/dual4-dflash` | `models/qwen3.6-27b/vllm/compose/multi4/autoround-int4/dflash.yml` | vllm-nightly-dflash | zlab-qwen-dflash | fp16 | 4 | production |
