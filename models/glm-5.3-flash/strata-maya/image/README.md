# strata-maya-club3090 image

`ghcr.io/noonghunna/strata-maya-club3090` is [Project Maya](https://github.com/mw00/project-maya) (MIT),
a fork of [Strata](https://github.com/Niko1221/Strata) for GLM-5.3-Flash (and, with engines of their own,
DeepSeek-V4.1-Flash-1017 on `maya-ds4` and MiniMax-M3.1-Flash on `maya-mm3`), built by us for each release.
Maya publishes no image and no Linux engine (its `maya.sh` compiles on the user's PC). From v1.1.53 its
repository has a recipe for building an image yourself ([`docs/DOCKER.md`](https://github.com/mw00/project-maya/blob/main/docs/DOCKER.md),
adapted from this one in [mw00/project-maya#808](https://github.com/mw00/project-maya/pull/808)), but no published
image, so this image is an exception to the image-provenance contract
([`docs/ADDING_ENGINES.md`](../../../../docs/ADDING_ENGINES.md)) by maintainer decision (2026-10-15).

| File | What it does |
|---|---|
| [`Dockerfile`](Dockerfile) | Ours: compiles the engine the way `maya.py` does and installs the server's Python packages. |
| [`build.sh`](build.sh) | Resolves a release, builds it, checks the image, pushes with `--push` (the maintainer runs it on the rig; see Publishing). |

## What the image is

Two stages on NVIDIA's CUDA 13.0 images (so the host needs a driver ≥ 580):

1. **Build** (`-devel`): the release's tree at its commit, llama.cpp's source at the commit `setup.py`
   pins (`LLAMA_CPP_COMMIT`; ggml's CPU backend computes the CPU-side experts), then the GLM engine's
   target (`maya`, `strata` before v1.1.16), DeepSeek's `maya-ds4` and MiniMax's `maya-mm3` (when the release
   defines them) with the
   flags `maya.py`'s `compile_engine` passes, plus two of ours:
   - **`-DSTRATA_PORTABLE=ON`**: ggml's CPU code for an AVX2 baseline. Without it the CMake sets
     `GGML_NATIVE=ON`, which builds for the build machine's CPU: right for a PC that compiles its own
     engine, wrong for an image (built on an AVX-512 runner, it would stop with an illegal
     instruction on a Ryzen 5000). It is upstream's own option; `maya.py` passes it only for its HIP
     build. Maya's own AVX-512 kernels are compiled per file and chosen at run time either way.
   - **`CMAKE_CUDA_ARCHITECTURES=75;80;86;89;120`** (RTX 20 / 30 / 40 / 50 and the A-series) instead of the
     user's own cards: a fat binary, one cubin per arch.
2. **Runtime** (`-runtime`, which carries cudart and cuBLAS): the release's files in `/opt/maya`
   (less `.git`, tests, benches and test dumps), llama.cpp's `gguf-py` in `third_party/llama.cpp/` where `setup.py` puts it (the pack builder reads the tokenizer with it), the engines at `/opt/maya/build/maya` (`build/strata` for releases before v1.1.16, which renamed its programs after Maya) `/opt/maya/build/maya-ds4` and `/opt/maya/build/maya-mm3`, and a
   `.venv` with `setup.py`'s `PY_PACKAGES` less the build tools, read from the release's own `setup.py` at
   build time, so a package a release adds is installed with it (the list is in
   `/opt/club3090/maya-py-packages.txt`, the versions pip took in `/opt/club3090/maya-pip-freeze.txt`).
   `MAYA_UPDATE_CHECK=0`: the dashboard's self-update does a
   `git` fast-forward, which an image can't do.

**Text only.** The vision encoder and the DFlash2 drafter are not built or fetched (the drafter's
weights are CC BY-NC-ND). The image has no entrypoint of its own: the composes mount our launcher
(`models/glm-5.3-flash/strata-maya/launcher/` for GLM; DeepSeek's slugs carry their own), which builds Maya's pack on the data volume, writes the config and starts `serve/server.py`.
Maya's own setup (`maya.sh`) is not used in the image. DeepSeek's vision tower is never fetched, so its
slugs run `--vision off`.

| Card | Arch | |
|---|---|---|
| H100 | `90` | not built (Maya's CMake refuses only archs below 75, so a build with `CUDA_ARCHITECTURES` adding 90 should work; untested) |
| B200 / GB200 | `100` | not built (no club hardware class) |
| DGX Spark (GB10) | `121` | **can't run**: ARM CPU, while the image is x86-64 and the CPU kernels are x86 SIMD |
| P100 / V100 | `60` / `70` | not built: CUDA 13 dropped them. Maya is developed on V100s; run `maya.sh` natively with CUDA 12 there |

Not measured: decode on an AVX-512 CPU against a natively compiled engine (the Strata image notes the
same open question).

## What a build checks

`build.sh` stops at the first failure, before anything is pushed:

| Check | Catches |
|---|---|
| tag → commit, built at that commit; VERSION file = the tag | a tag moved after the fact; a release whose files don't match its tag |
| `setup.py`'s llama.cpp pin = `CMakeLists.txt`'s `GIT_TAG` | upstream's two pins disagreeing (we build from `setup.py`'s, as `maya.py` does) |
| skip when `:<tag>` is already published (only "not found" = unpublished) | re-pushing a tag over an image users may have pinned |
| `/opt/club3090/maya-cpu-flags.txt`: `STRATA_PORTABLE` on, `GGML_NATIVE` off, `GGML_AVX2` on, no AVX-512 / VNNI / AMX | a native build slipping through (read from `CMakeCache.txt`, not the build log) |
| `/opt/club3090/cuda-architectures.txt` = the archs asked for | a build for the wrong cards |
| `ldd` on the engine: everything resolves except `libcuda` / `libnvidia-*` | a runtime library missing from the runtime stage |
| `build/maya-ds4` is there and resolves; `maya-ds4 --help` and `tools/ds4/pack_gguf.py --help` still offer every flag the DeepSeek launcher passes; `data/expert-profile-ds4.bin` is there | a release that drops or renames DeepSeek's engine target or a flag its launcher depends on |
| `build/maya-mm3` is there and resolves; every flag the MiniMax launcher passes is one of its strings exactly (its usage text leaves out `--profile` and `--serve`, which its parser takes); `tools/mm3/pack_gguf.py --help` offers its pack flags; `data/expert-profile-mm3.bin` and `tools/mm3/chat_template.jinja` are there | the same, for MiniMax's engine and its launcher |
| `serve/server.py --help` and `tools/iq_pack.py --help` still offer every flag the launcher passes; the expert profile is there | a release renaming a flag that every launch depends on |
| llama.cpp's `gguf-py` imports the way Maya's tools look for it, and `iq_pack.py` indexes the release's 12 MB synthetic GLM-5 model (`data/glm5-synth.gguf`; it carries no tokenizer, so the check wants the index and dense weights) | a pack builder that can't run in the image (the first local build lacked `gguf-py` and failed exactly there) |
| every package in `setup.py`'s `PY_PACKAGES` (less `cmake` / `ninja`), read from the image's own copy of `setup.py`, is installed in the `.venv` at a version the release accepts | a package a release adds missing from the image (v1.1.18's `jsonschema`, which the Responses API's structured output needs, was missing from the hand-kept list the recipe had until then) |

None needs a GPU, so they run on any build machine. **A published image is not a pin.** The `strata-maya` engine
profile moves to a new digest only after the checks in
[the bump runbook](../../../../docs/NIGHTLY_BUMP_RUNBOOK.md#strata-and-project-maya-the-bump-rule).

## Run it by hand

```bash
bash models/glm-5.3-flash/strata-maya/image/build.sh                       # newest release: build + check
bash models/glm-5.3-flash/strata-maya/image/build.sh --tag <vX.Y.Z> --push --jobs 32  # publish (see below)
bash models/glm-5.3-flash/strata-maya/image/build.sh --tag <vX.Y.Z> --verify ghcr.io/noonghunna/strata-maya-club3090:<vX.Y.Z>
CUDA_ARCHITECTURES=86 STRATA_MAYA_IMAGE=strata-maya-local \
  bash models/glm-5.3-flash/strata-maya/image/build.sh --tag <vX.Y.Z> --jobs 16 --force   # a quick local build for one card
```

A local image name needs `--force`: the "already published?" probe can't tell an unpublished local
name from a registry error.

## Publishing a release

The maintainer builds and pushes each release we take from the reference rig (GitHub CI built it until
2026-10-17), from a clean checkout of master:

```bash
git switch master && git pull
bash models/glm-5.3-flash/strata-maya/image/build.sh --tag <vX.Y.Z> --push --jobs 32
```

- **The push guard** ([`scripts/lib/image-push-guard.sh`](../../../../scripts/lib/image-push-guard.sh)) refuses `--push`
  when the recipe folder has uncommitted or untracked changes, or when HEAD isn't on `origin/master`. The image's
  recipe labels (`org.opencontainers.image.revision`, `club3090.recipe`) then name the code that built it. Without
  it, a rig build was once labelled with its worktree's base commit while it ran an uncommitted edit.
- **No provenance attestation.** CI's runs signed one through GitHub's own identity token; a rig push can't.
  The labels (upstream commit, llama.cpp pin, archs, CPU baseline, recipe commit) are the record.
- **The rig needs a GHCR login with push rights** (`docker login ghcr.io`, a token with `write:packages`).
- Nothing builds new releases automatically any more: run it when we want a release.
- The first push from the rig uploads every layer (the rig's layers have other digests than CI's);
  later pushes upload only the layers that changed.
- After the push, `build.sh` prints the digest and a line of facts for the pin PR.
