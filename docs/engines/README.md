# Inference engines

club-3090 serves every model through one of a few engines. **Which engine a slug runs is set by its registry entry, not by this page.** The slug's prefix names the engine (`vllm/`, `sgl/`, `llamacpp/`, …), and its entry's `engine:` field names the engine profile, `scripts/lib/profiles/engines/<id>.yml`, which pins the image.

This page is the index: what each engine is for here, which profiles it has, and where its page is. It carries no throughput numbers or image pins on purpose, because both drift. Read them from [BENCHMARKS.md](../../BENCHMARKS.md), the card pages and the engine profiles.

For a given slug:

```bash
bash scripts/switch.sh --list                # launchable slugs (--all adds other topologies and incubating, gated and deprecated slugs)
bash scripts/switch.sh --explain <slug>      # its engine profile, compose, fit verdict and measured rows
bash scripts/diagnose-profile.sh <slug>      # does it fit this hardware and the catalog
```

---

## The engines

| Engine | Slug prefix | Engine profiles | Image (pinned in the profile) | What it's for here | Page |
|---|---|---|---|---|---|
| **vLLM** | `vllm/` | `vllm-stable`; per-model `vllm-gemma-stable` and `vllm-gemma4-unified`; `vllm-lmcache` (KV offload through LMCache) | a release tag of the stock `vllm/vllm-openai`; any patches are volume-mounted into it, never baked in. `vllm-lmcache` pins `lmcache/vllm-openai` by digest | The widest catalog: Qwen3.8-27B, Qwen3.6, Gemma 4, Tess, Nemotron and more, from one card to multi-card tensor parallel, with MTP speculative decoding and vision. Most `defaults:` rows point here. | [VLLM.md](VLLM.md) |
| **SGLang** | `sgl/` | `sglang-stable` | a release tag of `lmsysorg/sglang` | Qwen3.8-27B and ThinkingCap-Qwen3.8-27B, from one card to eight (tensor parallel). Every slug is 🧪 Experimental. | [SGLANG.md](SGLANG.md) |
| **llama.cpp** (mainline) | `llamacpp/` | `llama-cpp-local` | a digest of ggml-org's official image | GGUF models that fit in VRAM, with mainline's MTP. Holds the MiMo single-card default and the Deckard and Tess dual-card defaults. | [LLAMA_CPP.md](LLAMA_CPP.md) |
| **llama.cpp club-3090** (MoE expert cache) | `llamacpp-club3090/` | one per release: `llamacpp-club3090-v1.5`, `llamacpp-club3090-v1.6` | a digest of `ghcr.io/noonghunna/llamacpp-club3090` | MoE models bigger than VRAM: experts live in CPU RAM behind a GPU expert cache ([@leloch](https://github.com/leloch/llama.cpp)'s work, not yet upstream). GLM-5.3-Flash, Qwen3.8-Flash-Next, DeepSeek-V4-Flash, Inkling. Every slug is 🧪 Experimental. | [LLAMACPP_CLUB3090.md](LLAMACPP_CLUB3090.md) |
| **ExLlamaV3 + TabbyAPI** | `exllamav3/` | `exllamav3` | a digest of TabbyAPI's own `ghcr.io/theroyallab/tabbyapi:cu13` | exl3 quants of Qwen3.8-Flash-Next on two cards, with the MoE experts computed on the CPU. 🐣 Incubating. | [EXLLAMAV3.md](EXLLAMAV3.md) |
| **bucko-vllm** | `bucko-vllm/` | `bucko-vllm` | a digest of DominikBucko's official `ghcr.io/dominikbucko/qwen38-flash-next-2x3090` | Qwen3.8-Flash-Next on two cards with experts and the PLE table in host RAM: a vLLM derivative with its own expert cache and MTP3. 🐣 Incubating. | [BUCKO_VLLM.md](BUCKO_VLLM.md) |
| **Strata** | `strata/` | `strata` | a digest of upstream's official `ghcr.io/niko1221/strata` image (`<release>-cuda13`; we built `strata-club3090` until 2026-10-20) | Qwen3.8-Flash-Next on one or two cards (ISTA's GSQ-RCO of the original model, or SC117's abliterated Swift 1.5 fine-tune), all experts in host RAM behind a GPU expert cache: its own engine for this model (not a llama.cpp fork). 🐣 Incubating. | [STRATA.md](STRATA.md) |
| **Project Maya** | `strata-maya/` | `strata-maya` | a digest of `ghcr.io/noonghunna/strata-maya-club3090`, which we build from each upstream release | GLM-5.3-Flash on one, two or four cards with Maya's own quants (Maya-S, Maya-M, Maya-L): a Strata fork rewritten for this model, experts spread over a VRAM pool, a pinned RAM tier and the model files. 🐣 Incubating. | [STRATA_MAYA.md](STRATA_MAYA.md) |
| **HyperQwen** | `vllm-hyperqwen/` | `vllm-hyperqwen` | a digest of syv-ai's official `ghcr.io/syv-ai/hyperqwen`, which their CI builds per commit | Qwen3.8-27B on one 24 GB card: syv-ai's vLLM derivative (a patch series, its own launchers and a model-preparation step at the first start) with the DFlash2 drafter. 🐣 Incubating. | [HYPERQWEN.md](HYPERQWEN.md) |
| **QEngine** | `qengine/` | `qengine` | a digest of the author's own `ghcr.io/bpx03/qengine` | Qwen3.8-27B INT4 (RedHatAI's checkpoint only) and Qwen3.8-Flash-Next (the author's own INT4 layout) on RTX 3090s, tuned for three with tensor parallel over PCIe: its own C++/CUDA engines. Two two-card slugs: the 27B ⚠️ Production w/ caveats, Flash-Next 🐣 Incubating. MIT license from 1.3.5.8 (PolyForm Noncommercial up to 1.3.5.7). | [QENGINE.md](QENGINE.md) |
| **PrismML llama.cpp** | `llama-cpp-prism/` | `llama-cpp-prism` | a digest of `ghcr.io/noonghunna/llamacpp-prism`, a wrapper we build around PrismML's release | Ternary (PQ2_0) quants of Qwen3.8-27B on one card: PrismML's llama.cpp fork, which mainline can't stand in for. 🧪 / 🐣. | [LLAMA_CPP_PRISM.md](LLAMA_CPP_PRISM.md) |

The two images we build ourselves (`llamacpp-prism` and `strata-maya-club3090`) are exceptions to the image-provenance rule: a derivative is otherwise pinned to an image its own maintainer publishes ([ADDING_ENGINES.md](../ADDING_ENGINES.md), "CONTRACT — image provenance"). [AGENTS.md](../../AGENTS.md) → "Engine image pinning" says which engines pin a tag and which a digest, and why.

**Defaults.** `<engine>/default` reads the `defaults:` map in `scripts/lib/profiles/registry.yaml`. `<model>/default` walks `engine_preference:` there and takes the first engine with a functional `defaults:` row for that model and topology. Today that's llama.cpp, then vLLM, on one card, and vLLM, then llama.cpp, on two or more. Only vLLM and llama.cpp hold defaults today; slugs on the other engines are always picked by name.

---

## No longer in the catalog

- **ik_llama.cpp** is out of the catalog. Every `ik-llama/` slug is 🗑️ Deprecated; they ran under the `llama-cpp-local` profile, and there is no ik-llama engine profile. [IK_LLAMA.md](IK_LLAMA.md) stays as a reference.
- **beellama.cpp** (`beellama-local`) was retired on 2026-07-27 (Anbeeld #98, won't fix). Its slugs are 🗑️ Deprecated and launch only by name with `--force`.
- **Engine profiles with no live slug**, still in `scripts/lib/profiles/engines/`:
  - the four `vllm-nightly-*` profiles;
  - `vllm-pip-baseline` and `vllm-stable-next`;
  - `vllm-diffusion-gemma`;
  - the first two moe-cache releases, `llamacpp-club3090` and `llamacpp-club3090-v1.1`.

---

## Choosing

- **For a model:** start from the card pages: [SINGLE_CARD.md](../SINGLE_CARD.md), [DUAL_CARD.md](../DUAL_CARD.md) and [MULTI_CARD.md](../MULTI_CARD.md). Their tables are generated from the registry and list every slug with its status and max context; ⭐ marks the model's default.
- **For engine trade-offs in general:** [INFERENCE_ENGINES.md](../INFERENCE_ENGINES.md) compares the upstream projects feature by feature.
- **For what a flag does:** [ENGINE_FLAGS.md](../ENGINE_FLAGS.md) lists every flag our composes pass.
- **For quantization:** [QUANTIZATION.md](../QUANTIZATION.md) covers the formats and which engine runs which. Qwen3.6-27B on vLLM uses AutoRound because it keeps the MTP head in BF16, which is what lets MTP load; see [models/qwen3.6-27b/README.md](../../models/qwen3.6-27b/README.md).
- **For upstream bugs that gate a slug:** [UPSTREAM.md](../UPSTREAM.md).

---

## Adding an engine

[ADDING_ENGINES.md](../ADDING_ENGINES.md) covers the whole process: proposal first, lineage, image provenance, the engine profile, and everything in the tooling that has to recognise it. A new engine adds a row to the table above and a page in this directory; the engine PR template asks for both.
