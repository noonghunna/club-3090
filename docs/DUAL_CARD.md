# Two 3090s — what to run

For **2× RTX 3090**. Pick a slug from the table, launch it, and read the notes below. NVLink is
detected at boot, so you don't choose a compose for it.

> 📣 **Newest first:** new slugs, numbers and caveats are posted in
> [Announcements](https://github.com/noonghunna/club-3090/discussions/categories/announcements)
> before this page catches up. Each slug below links the thread that introduced it.

## Pick a slug

**Quick picks** (2026-09-25):
- **Qwen3.8-27B** has four vLLM tiers, all 🧪: `dual-max` (official FP8, MTP drafter), `dual-fast`
  (INT4 with int8 activations, MTP), `dual-superfast` / `dual-supermax` (DFlash2 drafter), and
  `dual-ultrafast` / `dual-ultramax` (DFlash2 on the FlashAttention fp8-KV plugin, full 262K). What each tier trades
  and measured: [#1076](https://github.com/noonghunna/club-3090/discussions/1076). The same tiers on
  SGLang: [#1245](https://github.com/noonghunna/club-3090/discussions/1245).
- **ThinkingCap-Qwen3.8-27B** (bottlecapai's reasoning fine-tune): the same tiers on both engines,
  `thinkingcap38-27b-*`, at base-model speed. ⚠️ Its license is PolyForm Small Business, not Apache:
  [#1418](https://github.com/noonghunna/club-3090/discussions/1418).
- **Two full 262K sessions at once:** `sgl/qwen38-27b-dual-fast` or `vllm/qwen38-27b-dual-fast` (and the
  ThinkingCap twins). Which slugs serve concurrency: [#1092](https://github.com/noonghunna/club-3090/discussions/1092).
- **Keep idle sessions warm:** opt-in KV offload to host RAM and disk (`KV_OFFLOAD_GB`) on the dual MTP
  slugs of both engines: [#1419](https://github.com/noonghunna/club-3090/discussions/1419).
- **Qwen3.6-27B:** `vllm/dual` (the model's default) or `vllm/qwen-27b-dual-max` (FP8 weights).
- **Many agents at once:** `vllm/qwen-35b-a3b-dual` (✅). The MoE's total throughput holds steady up
  to 16 streams, where the dense 27B's peaks at 2 and halves by 8.
- **Very large MoE models** (GLM-5.3-Flash, DeepSeek-V4-Flash, Inkling-Small, Qwen3.8-Flash-Next)
  run with most experts in system RAM, so they need a lot of it: see the Notes column, and
  [#840](https://github.com/noonghunna/club-3090/discussions/840) for how CPU offload works here.

<!-- BEGIN GENERATED: slug-table dual -->
| Model | Slug | Status | Max ctx | Compose | Announced in | Notes |
|---|---|---|--:|---|---|---|
| **Agents-A1 (InternScience 35B agentic MoE)** | `vllm/agents-a1-dual` | ⚠️ caveats | 262144 | [dual/fp8-dynamic/fp8.yml](../models/agents-a1/vllm/compose/dual/fp8-dynamic/fp8.yml) | [#547](https://github.com/noonghunna/club-3090/discussions/547) |  |
| **DeepSeek-V4-Flash-0731 (284B MoE)** | `llamacpp-club3090/deepseek-flash-dual-q8-moecache` | 🧪 experimental | 204800 | [dual/unsloth-q8-kxl/moecache.yml](../models/deepseek-v4-flash-0731/llamacpp-club3090/compose/dual/unsloth-q8-kxl/moecache.yml) | [#951](https://github.com/noonghunna/club-3090/discussions/951) | host RAM ≥ 156 GB |
|  | `llamacpp/deepseek-flash-dual-iq2` | 🧪 experimental | 204800 | [dual/unsloth-iq2-xxs/offload.yml](../models/deepseek-v4-flash-0731/llama-cpp/compose/dual/unsloth-iq2-xxs/offload.yml) | [#909](https://github.com/noonghunna/club-3090/discussions/909) | host RAM ≥ 86 GB |
|  | `llamacpp/deepseek-flash-dual-q8` | 🧪 experimental | 204800 | [dual/unsloth-q8-kxl/offload.yml](../models/deepseek-v4-flash-0731/llama-cpp/compose/dual/unsloth-q8-kxl/offload.yml) | [#909](https://github.com/noonghunna/club-3090/discussions/909) | host RAM ≥ 146 GB |
| **DeepSeek-V4-Flash-Vision-Exp (deepseek4 MoE)** | `llamacpp-club3090/deepseek-flash-vision-dual-q8-moecache` | 🧪 experimental | 204800 | [dual/unsloth-ud-q8kxl/moecache.yml](../models/deepseek-v4-flash-vision-exp/llamacpp-club3090/compose/dual/unsloth-ud-q8kxl/moecache.yml) | — | host RAM ≥ 156 GB |
| **DeepSeek-V4.1-Flash-1017 (MoE with Engram tables)** | `strata-maya/deepseek41-flash-dual-ista-iq2-xxs` | 🐣 incubating | 65536 | [dual/ista-iq2-xxs/dspark.yml](../models/deepseek-v4.1-flash-1017/strata-maya/compose/dual/ista-iq2-xxs/dspark.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 106 GB |
| **Gemma 4 12B (unified)** | `vllm/gemma-12b-dual-bf16-mtp` | ⚠️ caveats | 262144 | [dual/bf16/mtp.yml](../models/gemma-4-12b/vllm/compose/dual/bf16/mtp.yml) | [#412](https://github.com/noonghunna/club-3090/discussions/412) |  |
| **Gemma 4 26B-A4B (MoE)** | `vllm/gemma-26ba4b-dual` | 🧪 experimental | 262144 | [dual/awq/mtp.yml](../models/gemma-4-26b-a4b/vllm/compose/dual/awq/mtp.yml) | — |  |
| **Gemma 4 31B** | `vllm/gemma-31b-dual` ⭐ | ⚠️ caveats | 229376 | [dual/qat-awq-int4/base.yml](../models/gemma-4-31b/vllm/compose/dual/qat-awq-int4/base.yml) | [#67](https://github.com/noonghunna/club-3090/discussions/67) |  |
| **GLM-5.3-Flash (320.76B MoE, Z.ai)** | `llamacpp-club3090/glm53-flash-dual-iq3xxs-moecache` | 🧪 experimental | 204800 | [dual/unsloth-ud-iq3xxs/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/dual/unsloth-ud-iq3xxs/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `llamacpp-club3090/glm53-flash-dual-iq4xs-moecache` | 🧪 experimental | 204800 | [dual/unsloth-ud-iq4xs/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/dual/unsloth-ud-iq4xs/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 142 GB |
|  | `llamacpp-club3090/glm53-flash-dual-q2k-moecache` | 🧪 experimental | 204800 | [dual/devquasar-q2k/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q2k/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `llamacpp-club3090/glm53-flash-dual-q2k-offload` | 🧪 experimental | 204800 | [dual/devquasar-q2k/offload.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q2k/offload.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `llamacpp-club3090/glm53-flash-dual-q3km-moecache` | 🧪 experimental | 204800 | [dual/devquasar-q3km/moecache.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q3km/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 145 GB |
|  | `llamacpp-club3090/glm53-flash-dual-q3km-offload` | 🧪 experimental | 204800 | [dual/devquasar-q3km/offload.yml](../models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q3km/offload.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 145 GB |
|  | `strata-maya/glm53-flash-dual-maya-l` | 🐣 incubating | 131072 | [dual/maya-l/base.yml](../models/glm-5.3-flash/strata-maya/compose/dual/maya-l/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 129 GB |
|  | `strata-maya/glm53-flash-dual-maya-m` | 🐣 incubating | 131072 | [dual/maya-m/base.yml](../models/glm-5.3-flash/strata-maya/compose/dual/maya-m/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 86 GB |
|  | `strata-maya/glm53-flash-dual-maya-s` | 🐣 incubating | 131072 | [dual/maya-s/base.yml](../models/glm-5.3-flash/strata-maya/compose/dual/maya-s/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 66 GB |
| **Inkling-Small (263B MoE, Thinking Machines)** | `llamacpp-club3090/inkling-small-dual-iq4xs-moecache` | 🧪 experimental | 262144 | [dual/unsloth-ud-iq4xs/moecache.yml](../models/inkling-small/llamacpp-club3090/compose/dual/unsloth-ud-iq4xs/moecache.yml) | [#959](https://github.com/noonghunna/club-3090/discussions/959) | host RAM ≥ 121 GB |
|  | `llamacpp-club3090/inkling-small-dual-iq4xs-residency` | 🧪 experimental | 262144 | [dual/unsloth-ud-iq4xs/residency.yml](../models/inkling-small/llamacpp-club3090/compose/dual/unsloth-ud-iq4xs/residency.yml) | [#959](https://github.com/noonghunna/club-3090/discussions/959) | host RAM ≥ 99 GB |
| **MiniMax-M3.1-Flash (428B MoE, MiniMax)** | `strata-maya/minimax31-flash-dual-maya-s` | 🐣 incubating | 204800 | [dual/maya-s/dspark.yml](../models/minimax-m3.1-flash/strata-maya/compose/dual/maya-s/dspark.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 102 GB |
| **Nanbeige4.3-33B** | `sgl/nanbeige43-33b-dual` | 🐣 incubating | 204800 | [dual/autoround-int4/fp8.yml](../models/nanbeige4.3-33b/sglang/compose/dual/autoround-int4/fp8.yml) | — |  |
|  | `vllm/nanbeige43-33b-dual` | 🐣 incubating | 204800 | [dual/autoround-int4/fp8.yml](../models/nanbeige4.3-33b/vllm/compose/dual/autoround-int4/fp8.yml) | — |  |
| **Nemotron-3 Puzzle 75B-A9B (hybrid Mamba2+attn MoE)** | `vllm/nemotron-75b-dual-w4a16` | 🧪 experimental | 262144 | [dual/w4a16/turbo.yml](../models/nemotron-3-puzzle-75b/vllm/compose/dual/w4a16/turbo.yml) | [#706](https://github.com/noonghunna/club-3090/discussions/706) |  |
| **Qwen 3.6 27B** | `vllm/dual` ⭐ | ⚠️ caveats | 262144 | [dual/autoround-int4/fp8-mtp.yml](../models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml) | [#733](https://github.com/noonghunna/club-3090/discussions/733) |  |
|  | `vllm/qwen-27b-dual-fast` | ⚠️ caveats | 262144 | [dual/autoround-int4/fp8-mtp.yml](../models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml) | [#733](https://github.com/noonghunna/club-3090/discussions/733) |  |
|  | `vllm/qwen-27b-dual-max` | ⚠️ caveats | 262144 | [dual/fp8/mtp.yml](../models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml) | [#733](https://github.com/noonghunna/club-3090/discussions/733) |  |
|  | `vllm/qwen-27b-dual-nvfp4` | ⚠️ caveats | 262144 | [dual/nvfp4/mtp.yml](../models/qwen3.6-27b/vllm/compose/dual/nvfp4/mtp.yml) | [#608](https://github.com/noonghunna/club-3090/discussions/608) | needs sm 9.0+ (not a 3090) |
|  | `vllm/qwen-27b-dual-lmcache` | 🐣 incubating | 262144 | [dual/fp8/lmcache.yml](../models/qwen3.6-27b/vllm-lmcache/compose/dual/fp8/lmcache.yml) | [#423](https://github.com/noonghunna/club-3090/discussions/423) |  |
| **Qwen 3.6 35B-A3B (MoE)** | `vllm/qwen-35b-a3b-dual` ⭐ | ✅ production | 262144 | [dual/autoround-int4/fp8.yml](../models/qwen3.6-35b-a3b/vllm/compose/dual/autoround-int4/fp8.yml) | — |  |
|  | `vllm/qwen-35b-a3b-dual-nvfp4-fast` | ⚠️ caveats | 262144 | [dual/nvfp4-fast/fp8.yml](../models/qwen3.6-35b-a3b/vllm/compose/dual/nvfp4-fast/fp8.yml) | [#608](https://github.com/noonghunna/club-3090/discussions/608) | needs sm 9.0+ (not a 3090) |
|  | `llamacpp/hauhaucs-35ba3b-dual` | 🧪 experimental | 262144 | [dual/morikomorizz-q6kp/mtp.yml](../models/qwen3.6-35b-a3b/llama-cpp/compose/dual/morikomorizz-q6kp/mtp.yml) | [#411](https://github.com/noonghunna/club-3090/discussions/411) |  |
|  | `vllm/qwen-35b-a3b-dual-nvfp4` | 🧪 experimental | 262144 | [dual/nvfp4/fp8.yml](../models/qwen3.6-35b-a3b/vllm/compose/dual/nvfp4/fp8.yml) | [#608](https://github.com/noonghunna/club-3090/discussions/608) | needs sm 9.0+ (not a 3090) |
| **Qwen 3.8 27B** | `qengine/qwen38-27b-dual` | ⚠️ caveats | 262144 | [dual/redhat-int4/mtp.yml](../models/qwen3.8-27b/qengine/compose/dual/redhat-int4/mtp.yml) | — |  |
|  | `llamacpp/qwen38-27b-dual-q8kxl` | 🧪 experimental | 262144 | [dual/unsloth-q8kxl/q8kv.yml](../models/qwen3.8-27b/llama-cpp/compose/dual/unsloth-q8kxl/q8kv.yml) | [#993](https://github.com/noonghunna/club-3090/discussions/993) |  |
|  | `sgl/qwen38-27b-dual-fast` | 🧪 experimental | 262144 | [dual/autoround-int4/mtp.yml](../models/qwen3.8-27b/sglang/compose/dual/autoround-int4/mtp.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `sgl/qwen38-27b-dual-max` | 🧪 experimental | 163840 | [dual/fp8/mtp.yml](../models/qwen3.8-27b/sglang/compose/dual/fp8/mtp.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `sgl/qwen38-27b-dual-superfast` | 🧪 experimental | 262144 | [dual/autoround-int4/dflash2.yml](../models/qwen3.8-27b/sglang/compose/dual/autoround-int4/dflash2.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `vllm/qwen38-27b-dual-fast` | 🧪 experimental | 262144 | [dual/autoround-int4/mtp.yml](../models/qwen3.8-27b/vllm/compose/dual/autoround-int4/mtp.yml) | [#1024](https://github.com/noonghunna/club-3090/discussions/1024) |  |
|  | `vllm/qwen38-27b-dual-max` | 🧪 experimental | 262144 | [dual/fp8/mtp.yml](../models/qwen3.8-27b/vllm/compose/dual/fp8/mtp.yml) | [#993](https://github.com/noonghunna/club-3090/discussions/993) |  |
|  | `vllm/qwen38-27b-dual-nvfp4` | 🧪 experimental | 262144 | [dual/nvfp4/mtp.yml](../models/qwen3.8-27b/vllm/compose/dual/nvfp4/mtp.yml) | [#1024](https://github.com/noonghunna/club-3090/discussions/1024) |  |
|  | `vllm/qwen38-27b-dual-superfast` | 🧪 experimental | 262144 | [dual/autoround-int4/dflash2-fp8.yml](../models/qwen3.8-27b/vllm/compose/dual/autoround-int4/dflash2-fp8.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
|  | `vllm/qwen38-27b-dual-supermax` | 🧪 experimental | 147456 | [dual/fp8/dflash2-fp8.yml](../models/qwen3.8-27b/vllm/compose/dual/fp8/dflash2-fp8.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
|  | `vllm/qwen38-27b-dual-ultrafast` | 🧪 experimental | 262144 | [dual/autoround-int4/dflash2.yml](../models/qwen3.8-27b/vllm/compose/dual/autoround-int4/dflash2.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
|  | `vllm/qwen38-27b-dual-ultramax` | 🧪 experimental | 262144 | [dual/fp8/dflash2.yml](../models/qwen3.8-27b/vllm/compose/dual/fp8/dflash2.yml) | [#1076](https://github.com/noonghunna/club-3090/discussions/1076) |  |
| **Qwen-AgentWorld 35B-A3B (language world model)** | `vllm/qwen-agentworld-35b-a3b-dual-awq-int4` | ✅ production | 262144 | [dual/cyankiwi-awq-int4/fp8.yml](../models/qwen-agentworld-35b-a3b/vllm/compose/dual/cyankiwi-awq-int4/fp8.yml) | — |  |
| **Qwen3.8-Flash-Next (MoE, Qwen)** | `llamacpp-club3090/qwen38-flash-next-dual-q4kxl-moecache` | 🧪 experimental | 204800 | [dual/unsloth-ud-q4kxl/moecache.yml](../models/qwen3.8-flash-next/llamacpp-club3090/compose/dual/unsloth-ud-q4kxl/moecache.yml) | [#1117](https://github.com/noonghunna/club-3090/discussions/1117) | host RAM ≥ 110 GB |
|  | `vllm/qwen38-flash-next-dual` | 🧪 experimental | 262144 | [dual/albucino-autoround-int4-fp8ple/mtp.yml](../models/qwen3.8-flash-next/vllm/compose/dual/albucino-autoround-int4-fp8ple/mtp.yml) | — | host RAM ≥ 19 GB |
|  | `bucko-vllm/qwen3.8-flash-next-ple-dual` | 🐣 incubating | 262144 | [dual/albucino-autoround-int4-fp8ple/base.yml](../models/qwen3.8-flash-next/bucko-vllm/compose/dual/albucino-autoround-int4-fp8ple/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 135 GB |
|  | `bucko-vllm/qwen3.8-flash-next-swift15-dual` | 🐣 incubating | 262144 | [dual/ukisai-swift15-autoround-int4/base.yml](../models/qwen3.8-flash-next/bucko-vllm/compose/dual/ukisai-swift15-autoround-int4/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 170 GB |
|  | `exllamav3/qwen38-flash-next-dual-exl3-305-cpumoe` | 🐣 incubating | 204800 | [dual/exl3-3.05bpw/cpumoe.yml](../models/qwen3.8-flash-next/exllamav3/compose/dual/exl3-3.05bpw/cpumoe.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 96 GB |
|  | `exllamav3/qwen38-flash-next-dual-exl3-405-cpumoe` | 🐣 incubating | 204800 | [dual/exl3-4.05bpw/cpumoe.yml](../models/qwen3.8-flash-next/exllamav3/compose/dual/exl3-4.05bpw/cpumoe.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 118 GB |
|  | `qengine/qwen38-flash-next-dual` | 🐣 incubating | 262144 | [dual/qengine-int4/mtp.yml](../models/qwen3.8-flash-next/qengine/compose/dual/qengine-int4/mtp.yml) | — | host RAM ≥ 88 GB |
|  | `strata/qwen3.8-flash-next-iq3s-dual` | 🐣 incubating | 131072 | [dual/ista-iq3s/base.yml](../models/qwen3.8-flash-next/strata/compose/dual/ista-iq3s/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 93 GB |
|  | `strata/qwen3.8-flash-next-swift15-abliterated-iq3s-dual` | 🐣 incubating | 131072 | [dual/sc117-swift15-abliterated-iq3s/base.yml](../models/qwen3.8-flash-next/strata/compose/dual/sc117-swift15-abliterated-iq3s/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 93 GB |
| **ThinkingCap Qwen3.6-27B** | `vllm/thinkingcap-dual-w4a8` ⭐ | ⚠️ caveats | 262144 | [dual/w4a16/w4a8.yml](../models/thinkingcap-27b/vllm/compose/dual/w4a16/w4a8.yml) | [#749](https://github.com/noonghunna/club-3090/discussions/749) |  |
| **ThinkingCap Qwen3.8-27B** | `sgl/thinkingcap38-27b-dual-fast` | 🧪 experimental | 262144 | [dual/autoround-int4/mtp.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/dual/autoround-int4/mtp.yml) | [#1418](https://github.com/noonghunna/club-3090/discussions/1418) |  |
|  | `sgl/thinkingcap38-27b-dual-max` | 🧪 experimental | 163840 | [dual/fp8/mtp.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/dual/fp8/mtp.yml) | [#1418](https://github.com/noonghunna/club-3090/discussions/1418) |  |
|  | `sgl/thinkingcap38-27b-dual-superfast` | 🧪 experimental | 262144 | [dual/autoround-int4/dflash2.yml](../models/thinkingcap-qwen3.8-27b/sglang/compose/dual/autoround-int4/dflash2.yml) | [#1418](https://github.com/noonghunna/club-3090/discussions/1418) |  |
|  | `vllm/thinkingcap38-27b-dual-fast` | 🧪 experimental | 262144 | [dual/autoround-int4/mtp.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/dual/autoround-int4/mtp.yml) | [#1418](https://github.com/noonghunna/club-3090/discussions/1418) |  |
|  | `vllm/thinkingcap38-27b-dual-max` | 🧪 experimental | 262144 | [dual/fp8/mtp.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/dual/fp8/mtp.yml) | [#1418](https://github.com/noonghunna/club-3090/discussions/1418) |  |
|  | `vllm/thinkingcap38-27b-dual-superfast` | 🧪 experimental | 262144 | [dual/autoround-int4/dflash2-fp8.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/dual/autoround-int4/dflash2-fp8.yml) | [#1418](https://github.com/noonghunna/club-3090/discussions/1418) |  |
|  | `vllm/thinkingcap38-27b-dual-supermax` | 🧪 experimental | 147456 | [dual/fp8/dflash2-fp8.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/dual/fp8/dflash2-fp8.yml) | [#1418](https://github.com/noonghunna/club-3090/discussions/1418) |  |
|  | `vllm/thinkingcap38-27b-dual-ultrafast` | 🧪 experimental | 262144 | [dual/autoround-int4/dflash2.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/dual/autoround-int4/dflash2.yml) | [#1418](https://github.com/noonghunna/club-3090/discussions/1418) |  |
|  | `vllm/thinkingcap38-27b-dual-ultramax` | 🧪 experimental | 262144 | [dual/fp8/dflash2.yml](../models/thinkingcap-qwen3.8-27b/vllm/compose/dual/fp8/dflash2.yml) | [#1418](https://github.com/noonghunna/club-3090/discussions/1418) |  |

65 slugs. ⭐ = the model's default for this topology (`bash scripts/switch.sh <model>/default`). Generated from the registry by `tools/docs/slug_tables.py`; don't edit by hand.
<!-- END GENERATED: slug-table dual -->

## Launch

```bash
bash scripts/setup.sh <model>            # first time: downloads the weights
bash scripts/switch.sh <slug>            # ✅ / ⚠️ slugs
bash scripts/switch.sh --force <slug>    # 🧪 / 🐣 slugs
bash scripts/switch.sh --list            # what your machine can run (--all adds retired slugs)
```

## Before you rely on it

- **Drafter + long agent sessions:** the Qwen3.8 slugs ship a speculative-decoding drafter, which is
  exposed to an open vLLM bug that can kill workers under sustained multi-turn traffic
  ([UPSTREAM.md](UPSTREAM.md), vllm#50021). `SPEC_N=0` turns the drafter off.
- **NVLink rigs:** on `dual-ultrafast`, the custom all-reduce kernel that NVLink enables cut the
  fillable context from 262K to ~185K on one rig; `DISABLE_CUSTOM_ALL_REDUCE=1` restored it
  ([#1339](https://github.com/noonghunna/club-3090/issues/1339)).
- **A long prompt stalls the other streams.** A big prefill (a large file or tool result) arriving
  while another stream is generating can drop that stream to near zero until the prefill finishes.
  Aggregate TPS figures measure throughput, not latency under agent traffic.
- **Context over concurrency.** `--max-num-seqs` caps how many requests run at once; it doesn't
  reserve memory for each. The KV pool is fixed by VRAM, so a higher context limit costs short
  requests nothing.
- **One user?** Single-stream generation barely uses the second card; one card is often the better
  deal ([SINGLE_CARD.md](SINGLE_CARD.md)).

## More

- **[Run the evals yourself](RUN_EVALS.md)**: measure a slug on your rig and post the numbers.
- [BENCHMARKS.md](../BENCHMARKS.md) (every measured row) · [PCIE_P2P.md](PCIE_P2P.md) (topology and
  P2P) · [KV_MATH.md](KV_MATH.md) · [FAQ.md](FAQ.md) · [PULL.md](PULL.md) (bring any Hugging Face
  model)
- [DUAL_CARD.history.md](DUAL_CARD.history.md): the previous long version of this page, with the
  sampling notes, VRAM budget and older measurements.
- One card → [SINGLE_CARD.md](SINGLE_CARD.md) · three or more → [MULTI_CARD.md](MULTI_CARD.md)
