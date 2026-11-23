# vllm-inkling-club3090 image

`ghcr.io/noonghunna/vllm-inkling-club3090` runs [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
on RTX 3090s (sm_86). It is vLLM's own Inkling image, `vllm/vllm-openai:inkling-2-mini`, with two changes. The
vLLM image itself refuses anything below Hopper.

| File | What it does |
|---|---|
| [`Dockerfile`](Dockerfile) | Two stages on the launch image (by digest): build the two op libraries for sm_86, then copy them in and apply the diff. |
| [`inkling2-sm86.diff`](inkling2-sm86.diff) | Ours: the sm_8x changes to `vllm/models/inkling2` (below). |
| [`inkling2-sm86.sha256`](inkling2-sm86.sha256) | The patched files the slugs were validated with; the build and `build.sh` check against it. |
| [`build.sh`](build.sh) | Build, check, push (the maintainer runs it on the rig; see Publishing). |

## Why it is an image of ours

This image is an exception to two of the repo's rules, by maintainer decision (2026-11-21), for as long as the
slugs need it ("When it ends", below):

- **Image provenance** ([`docs/ADDING_ENGINES.md`](../../../../docs/ADDING_ENGINES.md)): an engine's image comes
  from its own maintainer, and vLLM patches are mounted into the stock image rather than baked. A mount can't
  carry what this needs: the launch image's compiled ops have sm_100 code (plus Marlin's sm_80) only, so on a
  3090 `moe_align_block_size` returns its buffers unwritten, with no error, and other ops fail at warm-up. The
  ops have to be rebuilt.
- **Upstream first** for engine patches: the diff is ours. vLLM PR
  [#69165](https://github.com/vllm-project/vllm/pull/69165) does the same job on vLLM's `main`, which the launch
  image predates.

**When it ends.** The two-card slug moves to a stock vLLM engine once a release carries Inkling-2-Mini with sm_86 ops
and vLLM PR #69165 (vLLM 0.35.0 is the first candidate: `main` builds its ops for 8.6 and 12.0, and Mini merged in
#69028). The one-card slugs stay on this image, on FP8 KV, for now (maintainer decision, 2026-11-22): stock vLLM gates
Inkling's FP8 KV to SM12x, and #69165's own E4M3 decode was 3.6–8× slower in prefill than this diff's. So the image
lasts until a release runs FP8 KV on sm_86 at that speed.

## What the image is

1. **Ops** (no GPU needed): vLLM's source at
   [`fc144315`](https://github.com/vllm-project/vllm/commit/fc1443152c0eb655f5c2762f56b540bebe93b196), the Inkling
   PR head the image's Python was built from, from vllm-project/vllm's archive and checked by sha256. Two CMake
   targets are built, `_C_stable_libtorch` and `_moe_C_stable_libtorch`, with `TORCH_CUDA_ARCH_LIST=8.6`, inside
   the launch image (same torch 2.14.0, CUDA 13.0, Python 3.14). The PR's `csrc/inkling2` kernels are CMake-gated
   to 9.0a / 10.x and drop out; the diff routes around them.
2. **Runtime**: the launch image, the two libraries copied over its own, `inkling2-sm86.diff` applied to
   `vllm/models/inkling2` and checked against `inkling2-sm86.sha256`, and `TORCH_CUDA_ARCH_LIST=8.6`.

The diff, file by file (each change is marked `club-3090 sm86 overlay` in the code):

| File | Change |
|---|---|
| `quant_config.py` | Minimum capability 80, not 90: sm_8x runs the Hopper path's Humming experts (BF16 activations). |
| `nvidia/ops/__init__.py` | `pre_hopper()`; on sm_8x the Triton `gdc_wait` / `gdc_launch_dependents` (griddepcontrol, which ptxas rejects below sm_90) become no-ops and launches get `launch_pdl=False`. |
| `nvidia/ops/linear.py`, `nvidia/ops/conv_mix.py` | sm_8x takes the SM12x branches (no TMA; the Triton `short_conv` + `attn_res` fallback for the csrc conv/mix kernels). |
| `nvidia/vision.py` | FlashAttention 2 for the vision encoder (FA3 is Hopper-only). |
| `nvidia/ops/e4m3_sw.py` (new), `nvidia/ops/attention_sm90.py`, `nvidia/ops/sconv.py`, `nvidia/attention.py` | FP8 (E4M3) KV cache on sm_8x, in software: Triton has no `fp8e4nv` below sm_89, so the cache reaches the kernels as bytes. Decode: `(b & 0x7F) << 7` as fp16 × 256 (exact, subnormals included), NaN for 0x7F; encode: round to nearest even, saturate at ±448. |
| `speculator_config.py` | The DSpark draft keeps a BF16 KV cache on sm_8x too, as upstream does on SM12x. |

**sm_86 only.** The launch image's other compiled code doesn't run on a 5090 either: its FlashAttention 2 library
(used for vision) has sm_80 cubins only. A fat build of the two libraries would not change that. sm_86 cubins also
run on sm_89 (RTX 40xx), but nothing on that architecture has been tested.

## Publishing

```bash
bash models/inkling-2-mini/vllm-inkling/image/build.sh            # from master: build + check
# boot and verify every vllm-inkling slug on that local image, then:
bash models/inkling-2-mini/vllm-inkling/image/build.sh --push     # push THAT image, print its digest
```

A build is not reproducible bit for bit (CMake fetches third-party sources, and apt and pip resolve at build
time), so `--push` pushes the image already built and validated instead of building again. It refuses unless that
image was built from the commit checked out, the commit is on `origin/master` and the recipe folder is clean
(`scripts/lib/image-push-guard.sh`). It also refuses to move a published tag: bump `REV` in `build.sh` with any
change to the recipe. After a push, pin the digest in `scripts/lib/profiles/engines/vllm-inkling.yml`. The
catalog needs an anonymous pull, and a new GHCR package starts private, so make it public.
