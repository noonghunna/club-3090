# club-3090 docs index

Two tracks. Pick the one that matches what you're doing.

- **User track** — *"I have GPUs and a model; how do I serve it?"*
- **Contributor / maintainer track** — *"I'm working on the v0.8.0 pull
  pipeline, patches, or the calibration loop."*

Every link below resolves to a file in this repo.

---

## User track

Start here if you want to run a model.

| Doc | What it is |
|---|---|
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | **Start here — 5-minute clone-to-curl path.** No decisions, no menus. |
| [`SINGLE_CARD.md`](SINGLE_CARD.md) | 1× RTX 3090 — workload → curated config → quick start. |
| [`DUAL_CARD.md`](DUAL_CARD.md) | 2× RTX 3090 (PCIe / NVLink auto-detected) — workload → config → quick start. |
| [`MULTI_CARD.md`](MULTI_CARD.md) | 3+ GPUs — TP scaling math, derivation from `dual.yml`, valid TP values. |
| [`PULL.md`](PULL.md) | Any HF safetensors repo — evaluate against the KV math, honest about confidence. |
| [`BRING_YOUR_OWN.md`](BRING_YOUR_OWN.md) | Serve + tune + validate **your own** model/compose (any engine, single or dual) without touching the catalog. |
| [`HARDWARE.md`](HARDWARE.md) | Card-class questions — 4090/5090, power caps, NVLink, laptop EC power. |
| [`GLOSSARY.md`](GLOSSARY.md) | TPS / KV / MTP / TP and the rest of the vocabulary. |
| [`FAQ.md`](FAQ.md) | Common setup and operational questions. |
| [`COMPARISONS.md`](COMPARISONS.md) | Self-host vs cloud APIs — cost crossover and when each wins. |
| [`EXAMPLES.md`](EXAMPLES.md) | Worked end-to-end usage examples. |
| [`ai-studio/`](ai-studio/README.md) | **Club 3090 AI Studio** — chat-driven, all-modality creative studio (text · image · video · audio) on 2× 3090. **Start here** for the architecture + the full 8-lane matrix. |
| [`ai-studio/image.md`](ai-studio/image.md) | **Image** lanes — HiDream-O1 (top-quality/photoreal) · Ideogram-4 (design/logo/text) · Chroma (uncensored). |
| [`ai-studio/video.md`](ai-studio/video.md) | **Video** — LTX-2.3 (video+audio) + Sulphur (uncensored), text/image→video, 60 s+ chaining. |
| [`ai-studio/audio.md`](ai-studio/audio.md) | **Audio** — Step-Audio-EditX (voice clone+edit) · Kokoro (narration) · ACE-Step (music) · Stable Audio (SFX). |

---

## Contributor / maintainer track

### The v0.8.0 pull pipeline (in pipeline order)

A model slug flows through these stages. Read them in order to understand
the whole.

| Stage | Doc | What it owns |
|---|---|---|
| `[D]` | [`COMPOSE_GENERATOR.md`](COMPOSE_GENERATOR.md) | The #141 compose generator — the substrate that owns the arch→patches matrix. |
| Gate | [`PULL_GATE.md`](PULL_GATE.md) | `scripts/pull.sh` — the locked 6-stratum abort taxonomy, `[C0]`/`[C2a]`/`[B]`/`[C1]` gates, §4.1 confidence×verdict table. |
| `[E]` | [`PULL_EMIT_DERIVED.md`](PULL_EMIT_DERIVED.md) | Download → boot → smoke for a download-eligible derived model; writes the §6 capture artifacts. |
| `[F]` | [`LOOP.md`](LOOP.md) | The calibration loop — reads the capture bundle, classifies, runs the inbound-trust pipeline, dedups failures into the tracker. |

### Patch & model contribution

| Doc | What it is |
|---|---|
| [`PATCH_POLICY.md`](PATCH_POLICY.md) | When/how a patch ships, the local-overlay vs upstream rules. |
| [`PATCH_ATTRIBUTION.md`](PATCH_ATTRIBUTION.md) | The Phase-A patch-attribution matrix — arch → engine-pin → required patches. |
| [`ADDING_MODELS.md`](ADDING_MODELS.md) | How a new model gets added to the curated catalog. |
| [`KV_MATH.md`](KV_MATH.md) | The KV-cache math the `[B]` fit verdict is computed from. |

### Stack reference & ops

| Doc | What it is |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Repo/stack architecture overview. |
| [`UPSTREAM.md`](UPSTREAM.md) | Upstream PR / issue tracker for this stack. |
| [`NIGHTLY_BUMP_RUNBOOK.md`](NIGHTLY_BUMP_RUNBOOK.md) | Procedure for bumping the vLLM nightly pin. |
| [`CONTAINER_RUNTIMES.md`](CONTAINER_RUNTIMES.md) | Docker / container runtime notes. |

---

## Reference matrices & deep dives

These are cross-cutting references both tracks reach for.

| Doc | What it is |
|---|---|
| **`scripts/switch.sh --list`** *(runtime command, not a doc)* | **The authoritative compose × slug matrix.** Registry-derived from `scripts/lib/profiles/compose_registry.py`, so it's always current — every launchable slug with its topology, model, engine, KV format, and max ctx. Run this rather than trusting any hand-maintained table; the static lists in the per-topology docs are illustrative, this is the source of truth. |
| [`engines/`](engines/) | Per-engine deep dives — [vLLM](engines/VLLM.md), [llama.cpp](engines/LLAMA_CPP.md), [SGLang](engines/SGLANG.md). |
| [`INFERENCE_ENGINES.md`](INFERENCE_ENGINES.md) | Engine picker — which engine for which workload, and structural gaps. |
| [`CLIFFS.md`](CLIFFS.md) | The accumulated-context / prefill failure modes (Cliff 2, Cliff 2b) and how to detect them. |
| [`DTYPE_MATRIX.md`](DTYPE_MATRIX.md) | Supported dtype × model × engine matrix. |
| [`KERNEL_MATRIX.md`](KERNEL_MATRIX.md) | Quant-kernel availability and alignment constraints. |
| [`QUALITY_TEST.md`](QUALITY_TEST.md) | The quality-test harness and what it measures. |
| [`RESULTS_CARD.md`](RESULTS_CARD.md) | The standard 3-panel format (Serving · Quality · Takeaways) for sharing a config's measured results. |
| [`ANNOUNCEMENT_TEMPLATE.md`](ANNOUNCEMENT_TEMPLATE.md) | The "we shipped X" Announcements-post skeleton that wraps a Results Card (intro+credits · Results Card · getting it · run it · credits). |
| [`STRUCTURED_COT.md`](STRUCTURED_COT.md) | The bounded-thinking / structured-CoT compose path. |
| [`TQ3_MTP_GENESIS.md`](TQ3_MTP_GENESIS.md) | TQ3 KV × MTP × Genesis-patch results and config. |
