# Three or more GPUs — what to run

For **4 or 8 cards**. With 3, 5, 6 or 7 cards, run the 4-card slugs (or the 2-card ones) and leave
the rest idle: see [card counts](#card-counts-that-work) below.

⚠️ The maintainer's rig has two GPUs, so every slug on this page was validated on a community rig
or not yet at all. If you have the hardware, your run is the validation:
[Run the evals yourself](RUN_EVALS.md).

> 📣 **Newest first:** new slugs, numbers and caveats are posted in
> [Announcements](https://github.com/noonghunna/club-3090/discussions/categories/announcements)
> before this page catches up. Each slug below links the thread that introduced it.

## Pick a slug

**Quick picks** (2026-09-23):
- **Qwen3.8-27B** has the same tiers as on two cards (max, fast, super, ultra; see
  [#1076](https://github.com/noonghunna/club-3090/discussions/1076)), plus SGLang variants
  ([#1245](https://github.com/noonghunna/club-3090/discussions/1245)).
- **Qwen3.6-27B:** `vllm/qwen-27b-multi-fast` and `vllm/qwen-27b-multi-max` (⚠️) are the older
  generation, validated on several community rigs.
- **Very large MoE models on 4–8 cards** (GLM-5.3-Flash, DeepSeek-V4-Flash, Qwen3.8-Flash-Next):
  [#1117](https://github.com/noonghunna/club-3090/discussions/1117). They still need a lot of system
  RAM (Notes column).

### 4 cards

<!-- BEGIN GENERATED: slug-table multi4 -->
| Model | Slug | Status | Max ctx | Compose | Announced in | Notes |
|---|---|---|--:|---|---|---|
| **DeepSeek-V4-Flash-0731 (284B MoE)** | `llamacpp-club3090/deepseek-flash-multi4-q8-moecache` | 🧪 experimental | 204800 | [multi4/unsloth-q8-kxl/moecache.yml](../models/deepseek-v4-flash-0731/llamacpp-club3090/compose/multi4/unsloth-q8-kxl/moecache.yml) | [#951](https://github.com/noonghunna/club-3090/discussions/951) | host RAM ≥ 156 GB |
|  | `llamacpp/deepseek-flash-multi4-q8` | 🧪 experimental | 204800 | [multi4/unsloth-q8-kxl/offload.yml](../models/deepseek-v4-flash-0731/llama-cpp/compose/multi4/unsloth-q8-kxl/offload.yml) | [#909](https://github.com/noonghunna/club-3090/discussions/909) | host RAM ≥ 120 GB |
| **DeepSeek-V4-Flash-Vision-Exp (deepseek4 MoE)** | `llamacpp-club3090/deepseek-flash-vision-multi4-q8-moecache` | 🧪 experimental | 204800 | [multi4/unsloth-ud-q8kxl/moecache.yml](../models/deepseek-v4-flash-vision-exp/llamacpp-club3090/compose/multi4/unsloth-ud-q8kxl/moecache.yml) | — | host RAM ≥ 156 GB |
| **Gemma 4 31B** | `vllm/gemma-31b-multi-google-qat-w4a16` | ⚠️ caveats | 262144 | [multi4/google-qat-w4a16/base.yml](../models/gemma-4-31b/vllm/compose/multi4/google-qat-w4a16/base.yml) | — |  |
| **GLM-5.3-Flash (320.76B MoE, Z.ai)** | `llamacpp-club3090/glm53-flash-multi4-iq3xxs-moecache` | 🧪 experimental | 204800 | [multi4/unsloth-ud-iq3xxs/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi4/unsloth-ud-iq3xxs/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `llamacpp-club3090/glm53-flash-multi4-iq4xs-moecache` | 🧪 experimental | 204800 | [multi4/unsloth-ud-iq4xs/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi4/unsloth-ud-iq4xs/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 142 GB |
|  | `llamacpp-club3090/glm53-flash-multi4-q2k-moecache` | 🧪 experimental | 204800 | [multi4/devquasar-q2k/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q2k/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `llamacpp-club3090/glm53-flash-multi4-q2k-offload` | 🧪 experimental | 204800 | [multi4/devquasar-q2k/offload.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q2k/offload.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `llamacpp-club3090/glm53-flash-multi4-q3km-moecache` | 🧪 experimental | 204800 | [multi4/devquasar-q3km/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q3km/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 145 GB |
|  | `llamacpp-club3090/glm53-flash-multi4-q3km-offload` | 🧪 experimental | 204800 | [multi4/devquasar-q3km/offload.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q3km/offload.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 145 GB |
| **Inkling-Small (263B MoE, Thinking Machines)** | `llamacpp-club3090/inkling-small-multi4-iq4xs-moecache` | 🧪 experimental | 262144 | [multi4/unsloth-ud-iq4xs/moecache.yml](../models/inkling-small/llamacpp-club3090/compose/multi4/unsloth-ud-iq4xs/moecache.yml) | [#959](https://github.com/noonghunna/club-3090/discussions/959) | host RAM ≥ 121 GB |
| **Nemotron-3 Puzzle 75B-A9B (hybrid Mamba2+attn MoE)** | `vllm/nemotron-75b-multi-mtp` | ⚠️ caveats | 200000 | [multi4/nvfp4/mtp.yml](../models/nemotron-3-puzzle-75b/vllm/compose/multi4/nvfp4/mtp.yml) | [#706](https://github.com/noonghunna/club-3090/discussions/706) | needs sm 9.0+ (not a 3090) |
| **Qwen 3.6 27B** | `vllm/qwen-27b-multi-fast` | ⚠️ caveats | 262144 | [multi4/autoround-int4/mtp.yml](../models/qwen3.6-27b/vllm/compose/multi4/autoround-int4/mtp.yml) | [#733](https://github.com/noonghunna/club-3090/discussions/733) |  |
|  | `vllm/qwen-27b-multi-max` | ⚠️ caveats | 262144 | [multi4/fp8/mtp.yml](../models/qwen3.6-27b/vllm/compose/multi4/fp8/mtp.yml) | — |  |
| **Qwen 3.8 27B** | `sgl/qwen38-27b-multi4-fast` | 🧪 experimental | 262144 | [multi4/autoround-int4/mtp.yml](../models/qwen3.8-27b/sglang/compose/multi4/autoround-int4/mtp.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `sgl/qwen38-27b-multi4-max` | 🧪 experimental | 262144 | [multi4/fp8/mtp.yml](../models/qwen3.8-27b/sglang/compose/multi4/fp8/mtp.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `sgl/qwen38-27b-multi4-superfast` | 🧪 experimental | 262144 | [multi4/autoround-int4/dflash2.yml](../models/qwen3.8-27b/sglang/compose/multi4/autoround-int4/dflash2.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `sgl/qwen38-27b-multi4-supermax` | 🧪 experimental | 262144 | [multi4/fp8/dflash2.yml](../models/qwen3.8-27b/sglang/compose/multi4/fp8/dflash2.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `vllm/qwen38-27b-multi4-fast` | 🧪 experimental | 262144 | [multi4/autoround-int4/mtp.yml](../models/qwen3.8-27b/vllm/compose/multi4/autoround-int4/mtp.yml) | [#1024](https://github.com/noonghunna/club-3090/discussions/1024) |  |
|  | `vllm/qwen38-27b-multi4-max` | 🧪 experimental | 262144 | [multi4/fp8/mtp.yml](../models/qwen3.8-27b/vllm/compose/multi4/fp8/mtp.yml) | [#993](https://github.com/noonghunna/club-3090/discussions/993) |  |
|  | `vllm/qwen38-27b-multi4-superfast` | 🧪 experimental | 262144 | [multi4/autoround-int4/dflash2-fp8.yml](../models/qwen3.8-27b/vllm/compose/multi4/autoround-int4/dflash2-fp8.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
|  | `vllm/qwen38-27b-multi4-supermax` | 🧪 experimental | 262144 | [multi4/fp8/dflash2-fp8.yml](../models/qwen3.8-27b/vllm/compose/multi4/fp8/dflash2-fp8.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
|  | `vllm/qwen38-27b-multi4-ultrafast` | 🧪 experimental | 262144 | [multi4/autoround-int4/dflash2.yml](../models/qwen3.8-27b/vllm/compose/multi4/autoround-int4/dflash2.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
|  | `vllm/qwen38-27b-multi4-ultramax` | 🧪 experimental | 262144 | [multi4/fp8/dflash2.yml](../models/qwen3.8-27b/vllm/compose/multi4/fp8/dflash2.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
| **Qwen3.8-Flash-Next (MoE, Qwen)** | `llamacpp-club3090/qwen38-flash-next-multi4-q4kxl-moecache` | 🧪 experimental | 204800 | [multi4/unsloth-ud-q4kxl/moecache.yml](../models/qwen3.8-flash-next/llamacpp-club3090/compose/multi4/unsloth-ud-q4kxl/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
| **ThinkingCap Qwen3.8-27B** | `sgl/thinkingcap38-27b-multi4-fast` | 🐣 incubating | 262144 | [multi4/autoround-int4/mtp.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/multi4/autoround-int4/mtp.yml) | — |  |
|  | `sgl/thinkingcap38-27b-multi4-max` | 🐣 incubating | 262144 | [multi4/fp8/mtp.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/multi4/fp8/mtp.yml) | — |  |
|  | `sgl/thinkingcap38-27b-multi4-superfast` | 🐣 incubating | 262144 | [multi4/autoround-int4/dflash2.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/multi4/autoround-int4/dflash2.yml) | — |  |
|  | `sgl/thinkingcap38-27b-multi4-supermax` | 🐣 incubating | 262144 | [multi4/fp8/dflash2.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/multi4/fp8/dflash2.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi4-fast` | 🐣 incubating | 262144 | [multi4/autoround-int4/mtp.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi4/autoround-int4/mtp.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi4-max` | 🐣 incubating | 262144 | [multi4/fp8/mtp.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi4/fp8/mtp.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi4-superfast` | 🐣 incubating | 262144 | [multi4/autoround-int4/dflash2-fp8.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi4/autoround-int4/dflash2-fp8.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi4-supermax` | 🐣 incubating | 262144 | [multi4/fp8/dflash2-fp8.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi4/fp8/dflash2-fp8.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi4-ultrafast` | 🐣 incubating | 262144 | [multi4/autoround-int4/dflash2.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi4/autoround-int4/dflash2.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi4-ultramax` | 🐣 incubating | 262144 | [multi4/fp8/dflash2.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi4/fp8/dflash2.yml) | — |  |

35 slugs. Generated from the registry by `tools/docs/slug_tables.py`; don't edit by hand.
<!-- END GENERATED: slug-table multi4 -->

### 8 cards

<!-- BEGIN GENERATED: slug-table multi8 -->
| Model | Slug | Status | Max ctx | Compose | Announced in | Notes |
|---|---|---|--:|---|---|---|
| **GLM-5.3-Flash (320.76B MoE, Z.ai)** | `llamacpp-club3090/glm53-flash-multi8-iq3xxs-moecache` | 🧪 experimental | 204800 | [multi8/unsloth-ud-iq3xxs/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi8/unsloth-ud-iq3xxs/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `llamacpp-club3090/glm53-flash-multi8-iq4xs-moecache` | 🧪 experimental | 204800 | [multi8/unsloth-ud-iq4xs/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi8/unsloth-ud-iq4xs/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 142 GB |
|  | `llamacpp-club3090/glm53-flash-multi8-q2k-moecache` | 🧪 experimental | 204800 | [multi8/devquasar-q2k/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q2k/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `llamacpp-club3090/glm53-flash-multi8-q2k-offload` | 🧪 experimental | 204800 | [multi8/devquasar-q2k/offload.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q2k/offload.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `llamacpp-club3090/glm53-flash-multi8-q3km-moecache` | 🧪 experimental | 204800 | [multi8/devquasar-q3km/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q3km/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 145 GB |
|  | `llamacpp-club3090/glm53-flash-multi8-q3km-offload` | 🧪 experimental | 204800 | [multi8/devquasar-q3km/offload.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q3km/offload.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 145 GB |
| **Qwen 3.8 27B** | `sgl/qwen38-27b-multi8-fast` | 🧪 experimental | 262144 | [multi8/autoround-int4/mtp.yml](../models/qwen3.8-27b/sglang/compose/multi8/autoround-int4/mtp.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `sgl/qwen38-27b-multi8-max` | 🧪 experimental | 262144 | [multi8/fp8/mtp.yml](../models/qwen3.8-27b/sglang/compose/multi8/fp8/mtp.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `sgl/qwen38-27b-multi8-superfast` | 🧪 experimental | 262144 | [multi8/autoround-int4/dflash2.yml](../models/qwen3.8-27b/sglang/compose/multi8/autoround-int4/dflash2.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `sgl/qwen38-27b-multi8-supermax` | 🧪 experimental | 262144 | [multi8/fp8/dflash2.yml](../models/qwen3.8-27b/sglang/compose/multi8/fp8/dflash2.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `vllm/qwen38-27b-multi8-fast` | 🧪 experimental | 262144 | [multi8/autoround-int4/mtp.yml](../models/qwen3.8-27b/vllm/compose/multi8/autoround-int4/mtp.yml) | [#1024](https://github.com/noonghunna/club-3090/discussions/1024) |  |
|  | `vllm/qwen38-27b-multi8-max` | 🧪 experimental | 262144 | [multi8/fp8/mtp.yml](../models/qwen3.8-27b/vllm/compose/multi8/fp8/mtp.yml) | [#993](https://github.com/noonghunna/club-3090/discussions/993) |  |
|  | `vllm/qwen38-27b-multi8-superfast` | 🧪 experimental | 262144 | [multi8/autoround-int4/dflash2-fp8.yml](../models/qwen3.8-27b/vllm/compose/multi8/autoround-int4/dflash2-fp8.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
|  | `vllm/qwen38-27b-multi8-supermax` | 🧪 experimental | 262144 | [multi8/fp8/dflash2-fp8.yml](../models/qwen3.8-27b/vllm/compose/multi8/fp8/dflash2-fp8.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
|  | `vllm/qwen38-27b-multi8-ultrafast` | 🧪 experimental | 262144 | [multi8/autoround-int4/dflash2.yml](../models/qwen3.8-27b/vllm/compose/multi8/autoround-int4/dflash2.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
|  | `vllm/qwen38-27b-multi8-ultramax` | 🧪 experimental | 262144 | [multi8/fp8/dflash2.yml](../models/qwen3.8-27b/vllm/compose/multi8/fp8/dflash2.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
| **Qwen3.8-Flash-Next (MoE, Qwen)** | `llamacpp-club3090/qwen38-flash-next-multi8-q4kxl-moecache` | 🧪 experimental | 204800 | [multi8/unsloth-ud-q4kxl/moecache.yml](../models/qwen3.8-flash-next/llamacpp-club3090/compose/multi8/unsloth-ud-q4kxl/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
| **ThinkingCap Qwen3.8-27B** | `sgl/thinkingcap38-27b-multi8-fast` | 🐣 incubating | 262144 | [multi8/autoround-int4/mtp.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/multi8/autoround-int4/mtp.yml) | — |  |
|  | `sgl/thinkingcap38-27b-multi8-max` | 🐣 incubating | 262144 | [multi8/fp8/mtp.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/multi8/fp8/mtp.yml) | — |  |
|  | `sgl/thinkingcap38-27b-multi8-superfast` | 🐣 incubating | 262144 | [multi8/autoround-int4/dflash2.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/multi8/autoround-int4/dflash2.yml) | — |  |
|  | `sgl/thinkingcap38-27b-multi8-supermax` | 🐣 incubating | 262144 | [multi8/fp8/dflash2.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/multi8/fp8/dflash2.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi8-fast` | 🐣 incubating | 262144 | [multi8/autoround-int4/mtp.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi8/autoround-int4/mtp.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi8-max` | 🐣 incubating | 262144 | [multi8/fp8/mtp.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi8/fp8/mtp.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi8-superfast` | 🐣 incubating | 262144 | [multi8/autoround-int4/dflash2-fp8.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi8/autoround-int4/dflash2-fp8.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi8-supermax` | 🐣 incubating | 262144 | [multi8/fp8/dflash2-fp8.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi8/fp8/dflash2-fp8.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi8-ultrafast` | 🐣 incubating | 262144 | [multi8/autoround-int4/dflash2.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi8/autoround-int4/dflash2.yml) | — |  |
|  | `vllm/thinkingcap38-27b-multi8-ultramax` | 🐣 incubating | 262144 | [multi8/fp8/dflash2.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/multi8/fp8/dflash2.yml) | — |  |

27 slugs. Generated from the registry by `tools/docs/slug_tables.py`; don't edit by hand.
<!-- END GENERATED: slug-table multi8 -->

## Launch

```bash
bash scripts/setup.sh <model>            # first time: downloads the weights
bash scripts/switch.sh --force <slug>    # most multi-card slugs are 🧪 and need --force
bash scripts/switch.sh --list            # what your machine can run (--all adds retired slugs)
```

## Know this first

- **Several copies can beat one big split.** If the model fits on one or two cards, running one copy
  per card (or per pair) gives far more total throughput than splitting one copy across all of them:
  3.4× for Gemma-4-12B and 2.1–2.4× for Qwen3.6-27B on 4× 3090
  ([#773](https://github.com/noonghunna/club-3090/discussions/773)). Splitting (TP=4) wins for one
  user and for one very long prompt. Each copy needs its own port: [PODS.md](PODS.md).
- **More cards buy prefill and headroom, not generation speed.** On PCIe, going from 2 to 4 cards
  made prefill at 90K tokens 62% faster, while generation speed per stream barely moved (#773).
- **Pick the best-connected cards.** Check `nvidia-smi topo -m` (prefer `NV#` > `PIX` > `PXB` >
  `PHB` > `SYS`) and pin them with `CUDA_VISIBLE_DEVICES`. Mixing VRAM sizes wastes the bigger cards,
  because the KV pool is sized from the smallest one. More in [PCIE_P2P.md](PCIE_P2P.md).
- **Custom all-reduce is off at 3+ cards on PCIe** ([#786](https://github.com/noonghunna/club-3090/issues/786));
  the composes handle it.
- **fp8 KV cache with two or more streams** on the FlashAttention-plugin composes crashed the engine
  until [#1389](https://github.com/noonghunna/club-3090/pull/1389) (2026-09-23). Update before setting
  `KV_CACHE_DTYPE=fp8_e4m3` on a multi-card slug.

### Card counts that work

Tensor parallelism has to divide the model's attention heads. Qwen3.6/3.8-27B has 24 attention heads
and 4 KV heads, so it runs at **TP = 1, 2, 4 or 8**. With 3 cards use 2; with 5–7 use 4; with 9 or 10
use 8. Other models have other head counts: check the model's `config.json`.

## More

- **[Run the evals yourself](RUN_EVALS.md)**: measure a slug on your rig and post the numbers.
- [BENCHMARKS.md](../BENCHMARKS.md) (every measured row, including community 4-card runs) ·
  [PCIE_P2P.md](PCIE_P2P.md) · [KV_MATH.md](KV_MATH.md) · [PODS.md](PODS.md)
- [MULTI_CARD.history.md](MULTI_CARD.history.md): the previous long version of this page, including
  the scaling tables and the recipe for deriving your own TP=N compose from a 2-card one.
- One card → [SINGLE_CARD.md](SINGLE_CARD.md) · two → [DUAL_CARD.md](DUAL_CARD.md)
