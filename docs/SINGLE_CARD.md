# One 3090 — what to run

For **one RTX 3090 (24 GB)**. Pick a slug from the table, launch it, and read the notes below
before you put an agent on it.

> 📣 **Newest first:** new slugs, numbers and caveats are posted in
> [Announcements](https://github.com/noonghunna/club-3090/discussions/categories/announcements)
> before this page catches up. Each slug below links the thread that introduced it.

## Pick a slug

**Quick picks** (2026-09-23):
- **Long context + vision on one card:** `llamacpp/mimo9b-single-vision` (MiMo 9B, ⚠️) or
  `llamacpp/qwen38-27b-single-iq4xs` (Qwen3.8-27B on q4_0 KV, 🧪). Both serve the full 262,144 tokens
  with vision and leave a second card free.
- **Gemma:** `vllm/gemma-12b-single-int8-mtp` (full context) or `vllm/gemma-26ba4b-single` (176K).
- **Qwen3.6-27B** on one card is `vllm/minimal`: 32K, no vision. For long context on Qwen3.6, use two
  cards ([DUAL_CARD.md](DUAL_CARD.md)).
- ⚠️ The **NVFP4** slugs don't run on a 3090: they need Hopper/Blackwell (sm 9.0+), and
  `vllm/qwen38-27b-single-nvfp4` needs a 32 GB card.

<!-- BEGIN GENERATED: slug-table single -->
| Model | Slug | Status | Max ctx | Compose | Announced in | Notes |
|---|---|---|--:|---|---|---|
| **Gemma 4 12B (unified)** | `vllm/gemma-12b-single-int8-mtp` | ⚠️ caveats | 262144 | [single/autoround-int8/mtp.yml](../models/gemma-4-12b/vllm/compose/single/autoround-int8/mtp.yml) | [#412](https://github.com/noonghunna/club-3090/discussions/412) |  |
|  | `llamacpp/gemma-12b-single-q8kxl` | 🧪 experimental | 262144 | [single/unsloth-q8kxl/base.yml](../models/gemma-4-12b/llama-cpp/compose/single/unsloth-q8kxl/base.yml) | [#412](https://github.com/noonghunna/club-3090/discussions/412) |  |
|  | `vllm/gemma-12b-qat-w4a16-single` | 🧪 experimental | 262144 | [single/qat-w4a16/mtp.yml](../models/gemma-4-12b/vllm/compose/single/qat-w4a16/mtp.yml) | [#412](https://github.com/noonghunna/club-3090/discussions/412) |  |
| **Gemma 4 26B-A4B (MoE)** | `vllm/gemma-26ba4b-single` ⭐ | ⚠️ caveats | 176000 | [single/awq/int8.yml](../models/gemma-4-26b-a4b/vllm/compose/single/awq/int8.yml) | — |  |
| **GLM-5.3-Flash (320.76B MoE, Z.ai)** | `strata-maya/glm53-flash-single-maya-l` | 🐣 incubating | 131072 | [single/maya-l/base.yml](../models/glm-5.3-flash/strata-maya/compose/single/maya-l/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 148 GB |
|  | `strata-maya/glm53-flash-single-maya-m` | 🐣 incubating | 131072 | [single/maya-m/base.yml](../models/glm-5.3-flash/strata-maya/compose/single/maya-m/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 105 GB |
|  | `strata-maya/glm53-flash-single-maya-s` | 🐣 incubating | 131072 | [single/maya-s/base.yml](../models/glm-5.3-flash/strata-maya/compose/single/maya-s/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 87 GB |
| **MiMo V2.6 Distill Qwen 9B (Xiaomi)** | `llamacpp/mimo9b-single-vision` ⭐ | ⚠️ caveats | 262144 | [single/bartowski-q8/full-vision.yml](../models/mimo-v2.6-9b/llama-cpp/compose/single/bartowski-q8/full-vision.yml) | — |  |
| **Qwen 3.6 27B** | `vllm/minimal` ⭐ | ✅ production | 32768 | [single/autoround-int4/minimal.yml](../models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml) | [#733](https://github.com/noonghunna/club-3090/discussions/733) |  |
|  | `vllm/qwen-27b-single-nvfp4` | ⚠️ caveats | 65536 | [single/nvfp4/mtp.yml](../models/qwen3.6-27b/vllm/compose/single/nvfp4/mtp.yml) | [#608](https://github.com/noonghunna/club-3090/discussions/608) | needs sm 9.0+ (not a 3090) |
| **Qwen 3.6 35B-A3B (MoE)** | `vllm/qwen-35b-a3b-single-nvfp4` | ⚠️ caveats | 131072 | [single/nvfp4/fp8.yml](../models/qwen3.6-35b-a3b/vllm/compose/single/nvfp4/fp8.yml) | [#608](https://github.com/noonghunna/club-3090/discussions/608) | needs sm 9.0+ (not a 3090) |
|  | `vllm/qwen-a3b-preview-single` | 👁️ preview | 8192 | [single/autoround-int4/preview.yml](../models/qwen3.6-35b-a3b/vllm/compose/single/autoround-int4/preview.yml) | — |  |
| **Qwen 3.8 27B** | `llamacpp/qwen38-27b-swift15-iq3s-mtp` | ✅ production | 262144 | [single/ukisai-swift15-gsq-rco-iq3s-mtp/base.yml](../models/qwen3.8-27b/llama-cpp/compose/single/ukisai-swift15-gsq-rco-iq3s-mtp/base.yml) | [#993](https://github.com/noonghunna/club-3090/discussions/993) |  |
|  | `llama-cpp-prism/qwen38-27b-single-ternary-mtp` | 🧪 experimental | 262144 | [single/prism-ternary-mtp/mtp.yml](../models/qwen3.8-27b/llama-cpp-prism/compose/single/prism-ternary-mtp/mtp.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) |  |
|  | `llamacpp/qwen38-27b-single-iq4xs` | 🧪 experimental | 262144 | [single/unsloth-iq4xs/q4kv-vision.yml](../models/qwen3.8-27b/llama-cpp/compose/single/unsloth-iq4xs/q4kv-vision.yml) | [#993](https://github.com/noonghunna/club-3090/discussions/993) |  |
|  | `sgl/qwen38-27b-single-fast` | 🧪 experimental | 262144 | [single/autoround-int4/mtp.yml](../models/qwen3.8-27b/sglang/compose/single/autoround-int4/mtp.yml) | [#1245](https://github.com/noonghunna/club-3090/discussions/1245) |  |
|  | `vllm/qwen38-27b-single-fast` | 🧪 experimental | 262144 | [single/autoround-int4/mtp.yml](../models/qwen3.8-27b/vllm/compose/single/autoround-int4/mtp.yml) | [#1024](https://github.com/noonghunna/club-3090/discussions/1024) |  |
|  | `vllm/qwen38-27b-single-nvfp4` | 🧪 experimental | 65536 | [single/nvfp4/mtp.yml](../models/qwen3.8-27b/vllm/compose/single/nvfp4/mtp.yml) | [#1024](https://github.com/noonghunna/club-3090/discussions/1024) |  |
|  | `llama-cpp-prism/qwen38-27b-single-ternary-pq2` | 🐣 incubating | 262144 | [single/prism-ternary-pq2/ternary.yml](../models/qwen3.8-27b/llama-cpp-prism/compose/single/prism-ternary-pq2/ternary.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) |  |
|  | `vllm-hyperqwen/qwen38-27b-single-ultrafast` | 🐣 incubating | 65536 | [single/hyperqwen-w4a16/dflash2.yml](../models/qwen3.8-27b/vllm-hyperqwen/compose/single/hyperqwen-w4a16/dflash2.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) |  |
| **Qwen3.8-Flash-Next (MoE, Qwen)** | `strata/qwen3.8-flash-next-iq3s-single` | 🐣 incubating | 131072 | [single/ista-iq3s/base.yml](../models/qwen3.8-flash-next/strata/compose/single/ista-iq3s/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 84 GB |
|  | `strata/qwen3.8-flash-next-swift15-abliterated-iq3s-single` | 🐣 incubating | 131072 | [single/sc117-swift15-abliterated-iq3s/base.yml](../models/qwen3.8-flash-next/strata/compose/single/sc117-swift15-abliterated-iq3s/base.yml) | [#1700](https://github.com/noonghunna/club-3090/discussions/1700) | host RAM ≥ 84 GB |

22 slugs. ⭐ = the model's default for this topology (`bash scripts/switch.sh <model>/default`). Generated from the registry by `tools/docs/slug_tables.py`; don't edit by hand.
<!-- END GENERATED: slug-table single -->

## Launch

```bash
bash scripts/setup.sh <model>            # first time: downloads the weights
bash scripts/switch.sh <slug>            # ✅ / ⚠️ slugs
bash scripts/switch.sh --force <slug>    # 🧪 / 🐣 slugs
bash scripts/switch.sh --list            # what your machine can run (--all adds retired slugs)
```

New to this? [GETTING_STARTED.md](GETTING_STARTED.md) walks through the first run.

## Before you rely on it

- **Agents that keep their context** (Cline, OpenCode, Roo, hermes and similar): single-card vLLM on
  Qwen3.6-27B (`vllm/minimal`) hits a cliff at ~21–26K accumulated tokens
  ([Cliff 2b](CLIFFS.md)). It's specific to single-card vLLM on Qwen3-Next models; the llama.cpp slugs
  and Gemma don't have it. For long Qwen3.6 agent sessions, use two cards.
- **Idle VRAM isn't peak VRAM.** `nvidia-smi` at boot doesn't show the prefill peak (another
  0.5–1.5 GB). If a slug boots and then dies on the first long prompt, lower `MAX_MODEL_LEN`, or
  `GPU_MEMORY_UTILIZATION` by 0.03.
- **Sharing the card with a desktop:** lower `MAX_MODEL_LEN` first; it's a clean KV cut.
  `GPU_MEMORY_UTILIZATION=0.80` is usually too low for vLLM's startup profiling.
- **Retired slugs still run.** `switch.sh --list --all` shows them and `--force` launches them,
  unmaintained. For example `llamacpp/default`: Qwen3.6-27B at 200K, immune to the cliff above.

## More

- **[Run the evals yourself](RUN_EVALS.md)**: measure a slug on your rig and post the numbers.
- [BENCHMARKS.md](../BENCHMARKS.md) (every measured row) · [CLIFFS.md](CLIFFS.md) ·
  [KV_MATH.md](KV_MATH.md) (predict VRAM before you boot) · [FAQ.md](FAQ.md) ·
  [HARDWARE.md](HARDWARE.md) · [PULL.md](PULL.md) (bring any Hugging Face model) ·
  [PODS.md](PODS.md) (two instances side by side)
- [SINGLE_CARD.history.md](SINGLE_CARD.history.md): the previous long version of this page, with the
  derivations and the numbers for retired slugs.
- Two cards → [DUAL_CARD.md](DUAL_CARD.md) · three or more → [MULTI_CARD.md](MULTI_CARD.md)
