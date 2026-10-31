# HyperQwen — syv-ai's vLLM build for Qwen3.8-27B on one 24 GB card

**Engine id:** `vllm-hyperqwen` · **Slug prefix:** `vllm-hyperqwen/` · **Status of its slugs:** 🐣 Incubating · **Requires:** NVIDIA compute capability 8.6+ (measured on sm_86), one 24 GB card, the network on the first start

## What it is

[HyperQwen](https://github.com/syv-ai/HyperQwen) (Apache-2.0, syv-ai) is a vLLM derivative for Qwen3.8-27B. It has three parts:
- **a patch series on a pinned vLLM** (0.32.0 at our pin; its `patches/series` lists 56, one line each in `PATCHES.md`), among them int8 Marlin GEMM layer selection, the MTP draft vocabulary, GDN spec-decode bounds (vllm#50021) and a hybrid-KV CUDA-graph fix, plus the KVarN 4/2-bit KV cache;
- **its own launchers**, `single-user/` (speculative decoding, for one or a few users) and `batch/` (throughput), with their own env names (`SPEC`, `CTX`, `DFLASH_TOKENS`, `GPU_UTIL`, `EXTRA_ARGS`);
- **a model-preparation step** (`prepare/`) that runs at every container start. The first time, it rewrites the W4A16 checkpoint's `lm_head`, embeddings and MTP module to int8 in place, builds a draft vocabulary, and assembles the single-user "fast" variant from syv-ai's own Hugging Face repo. After that it takes seconds.

The image runs prepare, then HyperQwen's own `verify.sh --no-server`, then the launcher.

## Image

syv-ai's CI builds `ghcr.io/syv-ai/hyperqwen` on every push to main and tags it `:sha-<7>`; there are no versioned releases. The profile pins a digest (`:sha-f382342`, commit `f382342d`, 2026-10-30) and the launchers inject it as `VLLM_IMAGE`, as they do for `vllm-lmcache`. It pulls without credentials.

**Bumping:** boot the slug on the new digest through `switch.sh`, run `verify-full`, then `scripts/engine-pin-bump.sh vllm-hyperqwen ghcr.io/syv-ai/hyperqwen@sha256:<digest>`. Update `lineage.source_ref` and `base_version`. [NIGHTLY_BUMP_RUNBOOK.md](../NIGHTLY_BUMP_RUNBOOK.md) lists the HyperQwen interfaces the compose leans on, which are what can move.

## Slugs

| Slug | Cards | Port | Notes |
|---|---|---|---|
| `vllm-hyperqwen/qwen38-27b-single-ultrafast` | 1 | 8023 | upstream's setup B: DFlash2 W4A16 drafter at 7 tokens per verify, bf16 KV, 64K |

**Why "ultrafast":** it runs the drafter of our `*-ultrafast` tiers (syv-ai's DFlash2 W4A16) on an INT4 target. It differs from them in KV and context: bf16 at 64K on HyperQwen's kernels, where our ultrafast tiers run fp8 KV at 262K through our FlashAttention fp8-KV plugin. Its 8-bit option is `CTX=long`, at 128K. Since commit `ce9615c4` the image picks that backend on a 3090: by default int8 per-token-head KV on HyperQwen's own pth-flash-attn kernel (`PTH_FA=1`); with any `SPEC_N` but 7, or `PTH_FA=0`, fp8 KV through the FA2 fp8-KV plugin our ultrafast tiers use (`FA2_FP8KV=1`); and with both set to 0, int8 KV on Triton, which is what `CTX=long` ran before. That last arm is the one measured on the reference rig: decode the same as bf16, prefill 15% slower at 10K (the compose header has the numbers). The two newer arms aren't measured here. The bf16 default was the maintainer's call (2026-10-14).

**Weights:** `hyperqwen-w4a16`, `dbirks/Qwen3.8-27B-W4A16-AutoRound` (19.47 GB, the export HyperQwen's prepare is written for), in its own directory, `qwen3.8-27b-hyperqwen/`, because prepare rewrites it in place. The DFlash2 drafter is our existing `dflash2` weights, mounted read-only. `setup.sh` fetches both.

**First start:** prepare, then `torch.compile` and CUDA-graph capture: 368 s on the reference rig with the image already pulled (the image is 9.5 GB). Give it `READY_TIMEOUT=1800`. Later starts: 88 s.

Measurements, with dates and settings, are in the compose header and in [BENCHMARKS.md](../../BENCHMARKS.md) (Qwen3.8-27B, the single-card vLLM-derivatives table).

## What club-3090 adds

- **Our conventions on upstream's setup B:** internal port 8000 (so our tools find it), `BIND_HOST` / `ESTATE_PORT` / `ESTATE_CONTAINER`, the launcher's GPU choice, the `CLUB3090_RESTART` knob, and the host GID plus `umask 0002`, so what prepare and the compile caches write stays removable.
- **Our launch knobs, mapped onto HyperQwen's names** in the compose's entrypoint: `SPEC_N` sets `DFLASH_TOKENS` (0 turns speculation off with `SPEC=off`), and `REASONING_EFFORT` becomes `--default-chat-template-kwargs` (low by default, as on the other Qwen3.8 slugs). `MAX_NUM_SEQS`, our name on the vLLM composes, sets HyperQwen's `MAX_SEQS`.
- **HyperQwen's long-prompt prefill path, passed through:** `PTH_FA2_SLAB_PREFILL=1` (syv-ai/HyperQwen#386, off by default) prefills a long prompt through FA2 over bounded BF16 slabs of the INT8 KV cache, for one sequence on the pth-flash-attn backend. So it takes `CTX=long` (whose default backend that is) and `MAX_NUM_SEQS=1`: `CTX=long MAX_NUM_SEQS=1 PTH_FA2_SLAB_PREFILL=1 bash scripts/switch.sh --force vllm-hyperqwen/qwen38-27b-single-ultrafast`. It keeps a 136 MiB workspace; upstream measured cold prefill +6.5% on a 3090 (+7.8% at 150K). The entrypoint refuses a value other than 0 or 1, and says so when it's set where it won't engage.
- **The tooling contract:** it resolves to the vLLM kind by its id, container name, image (`engine-kind.sh` gained a `hyperqwen` arm) and profile `type`; vLLM's `SpecDecoding metrics` and `Avg prompt throughput` lines are intact, so `verify-full`, `bench.sh` and `health.sh` read it as vLLM.

Credit: community PR #1634 by VykosX did the first port of HyperQwen to this catalog and validated it on a 3090.

## Known gaps

- **Quality:** not validated here (no 8-pack, `verify-stress` or soak). #1634's 8-pack ran on VykosX's own build of the image.
- **Template:** HyperQwen serves its own hardened copy of the checkpoint's template, not ours. A system message after the first raises `System message must be at the beginning.`, and tool schemas are rendered without sorting their keys, so a client that reorders keys misses the prefix cache. Efforts map like ours (minimal to low, high and max to xhigh).
- **Thinking budgets:** none. `THINKING_BUDGET_*` and `ENABLE_THINKING` aren't wired.
- **Other setups:** only setup B is catalogued. HyperQwen also has C (`DFLASH_TOKENS=15`, quoting), D (`SPEC=mtp CTX=long`, 150K), E (`CTX=huge`, the KVarN cache, 240K), `batch`, and two cards with `EXTRA_ARGS="--tensor-parallel-size 2"`. The K8/V4 KVarN format from #1634 isn't in syv-ai's image.
- **`setup.sh` re-runs:** they re-download the shards prepare rewrote; prepare redoes them at the next start.
- **VRAM projection:** kv-calc counts the 19 GB download rather than the 15 GiB that's served, so `diagnose-profile` reports YELLOW on a 3090 (registered as expected in `test-diagnose-profile`).
