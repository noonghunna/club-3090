# PrismML llama.cpp — the fork that runs ternary quants of Qwen3.8-27B on one card

**Engine id:** `llama-cpp-prism` · **Slug prefix:** `llama-cpp-prism/` · **Status of its slugs:** 🐣 Incubating (plain) and 🧪 Experimental (MTP) · **Requires:** NVIDIA compute capability 8.6+ (the binary carries native code for sm_86, sm_89 and sm_120), one 24 GB card

## What it is

[PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) adds ternary weight types: weights in {−1, 0, +1} with one FP16 scale per 128, stored in a Hadamard-rotated basis, plus the activation transform that undoes the rotation at run time. There are two packings, PQ2_0 (2.13 bpw) and PTQ1_0 (1.75 bpw).

prism-ml's **Bonsai 2 27B** is a ternary quant of Qwen3.8-27B with the architecture unchanged (the GGUF declares `qwen35`), so the catalog treats it as a weights variant of `qwen3.8-27b`, not a new model. At PQ2_0 the weights are 7.21 GB, so the whole dense model sits on one card: no offload, no expert cache.

⚠️ **Mainline llama.cpp can't run these files, and fails silently.** It rejects PQ2_0 and PTQ1_0 as unknown types, and it loads Q2_0 with no warning and produces garbage, because it has no Hadamard activation runtime. Never point a Bonsai slug at `llama-cpp-local`, `llamacpp-club3090` or an ik_llama build.

Its profile declares `type: llama.cpp` and `lineage.relation: fork`; the tooling reads it as llama.cpp.

## Image

`ghcr.io/noonghunna/llamacpp-prism` wraps PrismML's own Linux CUDA 12.8 release tarball unchanged: nothing is compiled or patched, and the image only adds `libgomp1`, without which every binary in the tarball fails at exec. That makes it an image-provenance exception, one that predates the rule ([ADDING_ENGINES.md](../ADDING_ENGINES.md), "CONTRACT — image provenance"). Recipe: [models/qwen3.8-27b/llama-cpp-prism/image/Dockerfile](../../models/qwen3.8-27b/llama-cpp-prism/image/Dockerfile).

The profile pins release `prism-b12178-03ec600` by digest, and the launchers inject it as `LLAMACPP_PRISM_IMAGE`. A prism tag counts the fork's own builds, so `b10776` is not a mainline build number.

**Bumping:**
1. Pick a release whose assets include `linux-cuda-12.8-x64`; some ship Windows assets only. Check the tarball's sha256 against the release's published digest, then build as the Dockerfile header says.
2. Boot both slugs on the new image through `switch.sh` and run `verify-full` on each. Its `[9/10]` step fails the MTP slug if nothing was drafted.
3. Run `scripts/engine-pin-bump.sh llama-cpp-prism ghcr.io/noonghunna/llamacpp-prism@sha256:<digest>`.
4. Re-check every PrismML row in [UPSTREAM.md](../UPSTREAM.md) against the new release, not just the ones we carry. A PR we had dropped (#189) still arrived through `prism-b10770` and corrupted MTP output at `SPEC_N=2`, until #355 in `prism-b10776` fixed it.

## Slugs

| Slug | Cards | Port | Status | Notes |
|---|---|---|---|---|
| `llama-cpp-prism/qwen38-27b-single-ternary-pq2` | 1 | 8182 | 🐣 | prism-ml's PQ2_0 GGUF. No drafter: prism-ml's conversion dropped the MTP head that Qwen3.8-27B's other GGUFs carry |
| `llama-cpp-prism/qwen38-27b-single-ternary-mtp` | 1 | 8183 | 🧪 | the same target re-bundled by ProCreations with the MTP head put back (7.66 GB), loaded with `-m` and no `-md`; MTP at `SPEC_N=4` |

What both share:
- **KV and context:** q8_0 KV at 262144, the model's full context, on one card (17,108 MiB in use on the plain slug).
- **Vision:** the Q8_0 projector (0.63 GB) runs in host RAM (`-mmdev none`). Vision passes `verify-full` 4/4 on ground truth on the pinned build.
- **Effort and budget:** thinking is on by default, at effort `medium`, because upstream says `low` isn't supported and behaves like xhigh. The server's reasoning budget is 32,768 tokens. [RUN_EVALS.md](../RUN_EVALS.md#engines-with-their-own-defaults) has the eval settings.

`setup.sh` fetches the GGUF and the projector.

**Context is cheap here.** Everything is on the GPU, so there's no offload budget for context to eat. The whole ladder from 32768 to 262144 boots on one card, and decode is flat across it: 46.15 / 46.58 tok/s at 32768 against 45.66 / 45.28 at 262144, on the engine's own counter. q8_0 KV at 262144 uses 17,108 MiB against f16's 23,806 for a 0.2% decode difference, which is why it ships.

**MTP depth is a choice.** On a greedy sweep (2026-09-19), n=4 was fastest for code and structured output: 1.84× over no drafter. Prose peaked at n=2. Set `SPEC_N=2` for prose-heavy use; past n=4 throughput falls off sharply (n=5 was −18%). Under `bench.sh`'s sampler the gain is smaller: 1.62× on code, 1.26× on prose. llama-server logs a `draft acceptance` line per request, which `verify-full`, `bench.sh`, `health.sh` and `report.sh` read.

Only the bundled MTP head is supported as a drafter (`supported_drafters` in the profile); an external draft model is untried.

Measurements, with builds, dates and power caps, are in [BENCHMARKS.md](../../BENCHMARKS.md) (Qwen3.8-27B, the single-card PrismML table). Neither slug has been benched on the pinned build; the `prism-b10776` bump was boot-tested with `verify-full`.

## Known gaps

- **Quality:** one 8-pack, on the MTP slug's thinking leg at xhigh, on an earlier build (2026-09-19): 124/150 pass@1, 134/150 pass@3. It ran without a reasoning budget, so its cases were bounded by the clock rather than by tokens. There's no instruct leg, and nothing on the plain slug. The vendor's quality figure was measured with vLLM on an H100 and doesn't describe this fork's output.
- **Pack choice:** only PQ2_0 is catalogued. Upstream finds PQ2_0 faster on H100, A100 and Blackwell and PTQ1_0 faster on Ada and the L4; the 3090 is in neither list, so which pack wins here is unknown.
- **Not run:** `verify-stress`, soak and any concurrency above one request.

⚠️ **Attribution:** the ternary kernels, the quantization method and both packings are PrismML's work (Apache 2.0); the MTP re-bundle is ProCreations'. We wrap PrismML's release binary and patch nothing; what we add is measurement.
