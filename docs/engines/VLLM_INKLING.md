# vllm-inkling — vLLM's Inkling image, rebuilt to run Inkling-2-Mini on RTX 3090s

**Engine id:** `vllm-inkling` · **Slug prefix:** `vllm/` · **Status of its slugs:** 🐣 Incubating · **Requires:** compute capability 8.6 exactly (RTX 3090 / 3090 Ti, A5000 class), one or two 24 GB cards

## What it is

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is Thinking Machines' looped MoE with
vision, shipped as one mixed NVFP4 + MXFP8 ModelOpt checkpoint (19.5 GB of weights plus its 0.64 GB DSpark
drafter). vLLM published a launch image for it, `vllm/vllm-openai:inkling-2-mini`, which refuses anything below
Hopper. Lowering that gate isn't enough: the image's compiled ops carry sm_100 code (plus Marlin's sm_80) only, so on
a 3090 `moe_align_block_size` returns its buffers unwritten, with no error, and other ops fail at warm-up.

This engine's image is that launch image (by digest) with two changes:
- **`_C_stable_libtorch` and `_moe_C_stable_libtorch` rebuilt for sm_86**, from the vLLM source the image's Python
  was built from (the Inkling PR head, `fc144315`);
- **our `inkling2-sm86.diff` to `vllm/models/inkling2`**: sm_8x takes the existing SM12x and Hopper paths (no TMA,
  the Triton conv/mix fallback, the Humming experts with BF16 activations), PDL is off below sm_90, the vision
  encoder uses FlashAttention 2, and the FP8 (E4M3) KV cache is converted in software, because Triton has no
  `fp8e4nv` below sm_89. From image r2 the diff also stores a few layers as int8 per output channel, quantized at
  load (attention `qkvr`, the LM head, DSpark's Markov fold and draft layers): the code is from
  lulakien/inkling2-mini-3090-recipe, and the freed memory goes to the KV pool.

The recipe, the file-by-file diff table and the build checks are in
[`models/inkling-2-mini/vllm-inkling/image/README.md`](../../models/inkling-2-mini/vllm-inkling/image/README.md).

## Why we build it

It is an exception to two rules, by maintainer decision (2026-11-21):
- **Image provenance** ([ADDING_ENGINES.md](../ADDING_ENGINES.md), "CONTRACT — image provenance"): the image is ours,
  and the diff is baked in rather than mounted, because the op libraries have to be rebuilt.
- **Upstream first** for engine patches: the diff is ours.
  [vllm#69165](https://github.com/vllm-project/vllm/pull/69165) does the same job on vLLM's `main`, which the launch
  image predates ([UPSTREAM.md](../UPSTREAM.md) has both rows). From image r2 the diff also carries community code,
  the int8 weights from lulakien/inkling2-mini-3090-recipe: a capacity and speed change, not a critical fix
  (maintainer decisions 2026-11-27 to port it, 2026-11-28 to turn it on by default).

**When the slugs leave it.** The two-card slug (BF16 KV) moves to `vllm-stable` once a release carries Inkling-2-Mini
with sm_86 ops and vllm#69165. vLLM 0.35.0 is the first candidate: `main` has the model (vllm#69028) and builds its ops
for 8.6 and 12.0. **The one-card slugs stay here, on FP8 KV, for now** (maintainer decision, 2026-11-22), rather than
moving to BF16 at about half the context. Stock vLLM gates Inkling's FP8 KV to SM12x, and #69165's own software E4M3
decode made prefill 3.6–8× slower than ours when we opened that gate (2026-11-21). So this engine outlives v0.35
unless a release fixes both.

## Image

`ghcr.io/noonghunna/vllm-inkling-club3090`, tagged `:r<N>` (the recipe revision) and `:sha-<7>` (the recipe commit).
The profile pins a digest. The maintainer builds it on the reference rig from master
(`bash models/inkling-2-mini/vllm-inkling/image/build.sh`), validates every slug on that local image through
`switch.sh`, then pushes that same image with `build.sh --push`, because a build isn't reproducible bit for bit.
`--push` refuses unless the image was built from the commit checked out, that commit is on `origin/master` and the
recipe folder is clean, and it won't move a published tag. sm_86 cubins also run on sm_89 (RTX 40xx), but the
slugs pin `supported_sm: [8.6]` because nothing on that architecture has been tested. A 5090 can't run it: the
launch image's FlashAttention 2 library (used for vision) has sm_80 cubins only.

**Bumping:** there's no upstream stream to follow. The image changes only when the recipe does (a fix to the diff,
or a new launch image): bump `REV` in `build.sh`, merge, build from master, validate every slug, push, and pin the new
digest ([NIGHTLY_BUMP_RUNBOOK.md](../NIGHTLY_BUMP_RUNBOOK.md), "vllm-inkling").

## Slugs

| Slug | Cards | Port | Notes |
|---|---|---|---|
| `vllm/inkling-2-mini-dual-fast` | 2 | 8226 | TP=2, 262K (the maintainer's cap), BF16 KV, DSpark 7 tokens, `--max-num-seqs 8` |
| `vllm/inkling-2-mini-single-max` | 1 | 8227 | 240K, FP8 KV, no drafter, util 0.94 |
| `vllm/inkling-2-mini-single-fast` | 1 | 8228 | 150K, FP8 KV, DSpark 7 tokens, util 0.94 |

The one-card slugs run `--max-num-batched-tokens 512 --max-num-seqs 2 --no-async-scheduling`: the sliding-window and
conv-state caches reserve window plus in-flight tokens per request, so small batches are what leave room for the
window. BF16 KV topped out at about 97K on one card on image r1. Every slug passes `--no-enable-extensible-kv-cache`,
because vLLM 0.34's extensible KV cache holds back about 2.9 GiB a card on this model.

**Why not 262K on one card.** The one-card windows are the most that fits at util 0.94 on image r2 (2026-11-28):
one 262K request needs 4.51 GiB of KV on `single-max` against 4.24 GiB there (vLLM's estimate: 244,480 tokens), and
5.54 GiB on `single-fast`, whose drafter keeps its own KV in BF16, against 3.48 GiB (154,240 tokens). Util 0.95 would
leave less than verify-stress's 1,024 MB of free VRAM (954 MB on `single-fast` at 160K). With the int8 weights off
(`VLLM_INKLING_W8=`), the windows have to come back down to image r1's 180K and 100K.

**Why BF16 on two cards:** with FP8 KV the same config was slower (105.80 / 158.25 against 113.98 / 164.07 on
2026-11-21), and one verify-full request looped on both boots tried, token for token, because vLLM's default seed
hands it the same sample. BF16 passes.

Measurements, with dates and settings, are in the compose headers and in [BENCHMARKS.md](../../BENCHMARKS.md)
(Inkling-2-Mini).

## What club-3090 adds

- **Our launch knobs:** `SPEC_N` sets the DSpark depth (0, or `SPEC=off`, turns the drafter off) and
  `DRAFT_SAMPLE_METHOD` picks `probabilistic` (the default, vLLM's recipe) or `greedy`. `REASONING_EFFORT` becomes
  `--default-chat-template-kwargs`, the effort a request that sends none gets (`high` by default, the template's own).
  A request's own `chat_template_kwargs.reasoning_effort` or top-level `reasoning_effort` wins.
  `VLLM_INKLING_W8` (image r2 on) picks the layers stored as int8 per output channel, quantized at load: `attn` (the
  attention `qkvr` projection), `head` (the LM head, shared by the drafter), `markov` (DSpark's folded Markov head) and
  `draft` (the draft layers). The composes default to all four; `VLLM_INKLING_W8=` keeps them bf16. The experts are
  untouched. The code is from lulakien/inkling2-mini-3090-recipe (Apache-2.0), credited in the image README.
  With it empty the one-card windows don't fit: set `MAX_MODEL_LEN` back to 180000 / 100000 as well.
- **Our compose conventions** on vLLM's flags: internal port 8000, `BIND_HOST` / `ESTATE_PORT` / `ESTATE_CONTAINER`,
  the launcher's GPU choice, the restart knob, the cache mounts with the host GID and `umask 0002`, and on two cards
  `detect_nvlink.sh` for the peer transport (it turned PCIe P2P on for the measured boots).
- **The tooling contract:** it's a `type: vllm` profile, so `engine-kind.sh`, `verify-full`, `bench.sh` and
  `health.sh` read it as vLLM. This vLLM returns reasoning in the `reasoning` field, which the harness reads since
  #2132.

## Known gaps

- **Quality:** not measured (no 8-pack), and no soak. The int8 weights (on by default from image r2) rest on
  lulakien's own check, MMLU 911 vs 908 in bf16, not on a measurement of ours.
- **Reasoning loops:** the model card warns of long reasoning loops, "particularly when debugging code", and
  recommends output-token and agent-turn limits. Cap `max_tokens` on agent traffic.
- **Always reasons:** the template reads `reasoning_effort` (minimal … xhigh), not `enable_thinking`, and its
  minimum doesn't switch reasoning off. `none` reaches the template as 0.0, which reasoned more than `minimal` in the
  2026-11-21 probe.
- **Sampling:** the checkpoint's `generation_config` sets temperature 1.0, which is what a request that names none
  gets. The bench ran at the stack's 0.6 / 0.95 / 20, and DSpark's acceptance depends on the temperature, so treat the
  drafter's measured gain as an upper bound.
- **KV calculator:** kv-calc can't model the looped core, so the slugs are `kvcalc_key: SKIP` and `switch.sh
  --explain` gives no fit verdict.
