# Project Maya — a Strata fork for GLM-5.3-Flash on one, two or four cards, DeepSeek-V4.1-Flash on two, and MiniMax-M3.1-Flash on one or two

**Engine id:** `strata-maya` · **Slug prefix:** `strata-maya/` · **Status of its slugs:** 🐣 Incubating · **Requires:** NVIDIA compute capability 7.5+, driver ≥ 580 (CUDA 13), an x86-64 CPU with AVX2, and enough host RAM to hold the experts that don't fit in VRAM (~65 GB for Maya-S on two cards up to ~148 GB for Maya-L on one, measured; less RAM still runs, reading the rest from the model files)

## What it is

[Project Maya](https://github.com/mw00/project-maya) (MIT, mw00) is a fork of [Strata](https://github.com/Niko1221/Strata) rewritten for one model, Z.ai's GLM-5.3-Flash (320.76B MoE), with its own quants ([peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)). It keeps Strata's shape: a C++/CUDA engine (`build/maya`; `build/strata` before v1.1.16) with ggml's CPU backend as a library, and a Python server (`serve/server.py`) on port 8080 with an OpenAI API (`/v1/chat/completions`), an Anthropic one (`/v1/messages`) and a dashboard.

Where the experts live:
- **VRAM:** a per-layer expert pool on each card, filled from a shipped routing profile and promoted as it runs;
- **host RAM:** a pinned tier sized from the memory free at start;
- **the model files:** whatever fits in neither, read while it answers.

Each token's RAM-tier experts are split between the CPU (computed in place) and PCIe copies to the GPU, priced per expert at start. On several cards it splits the **layers** (`--layer-split auto`, a placement search over the cards' free VRAM), and on two or more it drafts with the model's own MTP (NextN) block, the cards working on consecutive tokens. No NCCL and no P2P.

Its profile declares `type: strata` and `lineage.relation: fork` (base `strata`); its id carries the family prefix, so the tooling knows it as engine kind `strata` (`scripts/lib/engine-kind.sh`): container names `strata-maya-…`, image `strata-maya-club3090`.

## Image

Maya publishes no image, no Dockerfile and no Linux engine: `maya.sh` compiles on the user's PC. So club-3090 builds `ghcr.io/noonghunna/strata-maya-club3090` for each release, an image-provenance exception recorded in [ADDING_ENGINES.md](../ADDING_ENGINES.md).
- **Build:** our own Dockerfile compiles the engines (`maya` for GLM, `maya-ds4` for DeepSeek since #1801, `maya-mm3` for MiniMax-M3.1-Flash in the releases that define it, v1.1.29 on) the way `maya.py` does, with upstream's `STRATA_PORTABLE=ON` (ggml's CPU code for an AVX2 baseline instead of the build machine's CPU) and a fixed arch list. The runtime stage carries the release's files, llama.cpp's `gguf-py` and the server's Python packages.
- **Archs:** `75;80;86;89;120` (RTX 20–50 and the A-series). V100 needs CUDA 12, which this image can't build; Maya's own setup builds it natively.
- **Text only:** neither the vision encoder nor the DFlash2 drafter (CC BY-NC-ND weights) is in the image.
- **Recipe, checks, publishing and pin flow:** [models/glm-5.3-flash/strata-maya/image/](../../models/glm-5.3-flash/strata-maya/image/README.md).

The profile pins a digest, and the launchers inject it as `STRATA_MAYA_IMAGE`. Export `STRATA_MAYA_IMAGE` in the shell to try another build for one launch; a saved value loses to the pin.

**Bumping:** the maintainer publishes a release from the rig with `build.sh --tag <vX.Y.Z> --push` (`:<tag>`, `:sha-<commit>`), but publishing doesn't pin. What moving the pin checks, and when to skip a release: [the bump runbook](../NIGHTLY_BUMP_RUNBOOK.md#strata-and-project-maya-the-bump-rule).

A release that renames a flag the launchers pass (`serve/server.py`, `tools/iq_pack.py`; for DeepSeek, `maya-ds4` and `tools/ds4/pack_gguf.py`; for MiniMax, `maya-mm3` and `tools/mm3/pack_gguf.py`) fails the image build, so the recipe and the launchers move together.

## Slugs

| Quant | Single (1 card) | Dual (2 cards) | Multi4 (4 cards) |
|---|---|---|---|
| **Maya-S v2** (`maya-s`, IQ2_XXS experts, 96.5 GB) | `strata-maya/glm53-flash-single-maya-s` · 8027 | `strata-maya/glm53-flash-dual-maya-s` · 8026 | `strata-maya/glm53-flash-multi4-maya-s` · 8028 |
| **Maya-M** (`maya-m`, IQ2_S experts, 116 GB) | `strata-maya/glm53-flash-single-maya-m` · 8035 | `strata-maya/glm53-flash-dual-maya-m` · 8029 | `strata-maya/glm53-flash-multi4-maya-m` · 8037 |
| **Maya-L** (`maya-l`, IQ3_S experts, 156.3 GB) | `strata-maya/glm53-flash-single-maya-l` · 8044 | `strata-maya/glm53-flash-dual-maya-l` · 8043 | `strata-maya/glm53-flash-multi4-maya-l` · 8045 |

- **Weights:** Maya's own quants from `peasantsmith/GLM-5.3-Flash-Maya-GGUF` (revision `f16ee4ba`), one folder per quant under `glm-5.3-flash-maya-gguf/`. Each carries the MTP draft block. Upstream's quality figures (KL against the FP8 model on held-out text, its own harness): Maya-L 0.188, Maya-M 0.291, Maya-S 0.428. Smaller quants fit more experts in VRAM, so they decode faster.
- **multi4 is never booted on the reference rig** (two cards): community-validated by design. Its numbers come from community reports.
- **One card has no drafting:** the MTP block runs only on two or more GPUs.

**Data volume:** `strata-maya-data` under the data directory (`~/.local/share/club-3090` by default; `STRATA_MAYA_DATA_DIR` overrides), one folder per quant: a symlink per shard, Maya's pack (its index of the model files, ~1 GB, written next to the shards as the engine expects) and the config.

**First start:** the launcher checks each shard's SHA-256 against Maya's own table (a few minutes for 100–156 GB, once; again only when a file's size or time changes) and builds the pack (a minute or a few, once). Later starts take about 30 seconds on the reference rig. That is with the shards in the page cache: from a hard disk with a cold cache, every start reads the whole pack, which took 930–1,010 s for Maya-L and Maya-M on a community 2× RTX 5090 rig. That is longer than `switch.sh`'s 10-minute wait, so give it `READY_TIMEOUT=1800` (or re-run `switch.sh` once it times out; the container keeps loading). Dropping the page cache before a boot (advice below, so Maya sizes its RAM tier from real free memory) makes the next start a cold one. `SETUP_ONLY=1` runs just that; `REPACK=1` builds the pack again; `VERIFY=0` skips the hash check.

**RAM:** Maya sizes its RAM tier from the memory free at start. Drop the page cache first on a box that just read big files, or the tier comes out smaller than it could be. On a host with less RAM than the measured figure, the experts that don't fit are read from the model files while it answers (slower; not measured here).

**Requests and readiness:** one request at a time. `switch.sh`'s readiness probe (one completion) passes once the server prints `ready: http`.

Measurements, with dates and settings, are in each compose's header and in [BENCHMARKS.md](../../BENCHMARKS.md) (GLM-5.3-Flash, the Project Maya table).

## What club-3090 adds

- **A launcher for every compose** (`models/glm-5.3-flash/strata-maya/launcher/`: `maya-entrypoint.sh` and `maya_prepare.py`). Maya's own setup isn't used in the image. The model folder is mounted read-only, but Maya writes its pack next to the shards, so the launcher links the shards into a folder of their own on the data volume, checks their SHA-256 there, builds the pack with Maya's `tools/iq_pack.py` (and stamps it as `maya.py` does, so a changed shard rebuilds it), writes the config and starts `serve/server.py` on the cards `MAYA_GPUS` names. The compose passes the quant as the launcher's one argument, not as a setting, so no saved value can point a slug at another quant's pack.
- **Calibration knobs:** `MAYA_PCIE_SHARE` and `MAYA_CPU_LANE` become the engine's `STRATA_GLM_PCIE_SHARE` / `STRATA_GLM_CPU_LANE` when set. They are what Maya's own calibration tunes for a PC; the composes ship the engine's own choice. See [Tuning for your PC](#tuning-for-your-pc).
- **Launch settings, written into Maya's config at every boot** (`launcher/maya_settings.py`). The launcher writes `maya.json` afresh at each start, so a setting you remove is gone at the next boot. A bad value stops the boot with the reason, before the SHA-256 check. `REASONING_EFFORT` and `ENABLE_THINKING` are catalogued, so `bash scripts/switch.sh --set <slug> REASONING_EFFORT=high` saves them for one slug; save the others with `bash scripts/settings.sh set TEMP=0.7` (they apply to every slug that reads them) or give them as a shell prefix.

  | Setting | Maya config | Values | Unset |
  |---|---|---|---|
  | `REASONING_EFFORT` | `reasoning_effort` | `none`, `low`, `high`, `max`; `medium` runs as `high` and `xhigh` as `max` (GLM-5.3 has four levels, and Maya's server maps the names so) | `max`, as on the other GLM-5.3-Flash slugs (Maya's own setup writes `high`) |
  | `ENABLE_THINKING` | `reasoning_effort: none` when `false` | `true`, `false`; `false` wins over `REASONING_EFFORT` | thinking on |
  | `THINKING_BUDGET` | `thinking_budget` | tokens, `0` = no budget; one cap for every request. A request's own `max_tokens` lowers it to leave the answer a quarter of that (at least 1,024 tokens) | Maya's own, 32,768 |
  | `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `PRESENCE_PENALTY`, `REPEAT_PENALTY` | `sampling.temperature`, `.top_p`, `.top_k`, `.min_p`, `.presence_penalty`, `.repetition_penalty` | ≥ 0 / 0–1 / 1–64 / 0–1 / ≥ 0 / > 0 | temperature 1.0, top_p 0.95 (GLM-5.3's vendor sampling); the rest the engine's |
  | `KV` | `--kv` in the engine args | `fp16`, `int8` (the attention's latent cache in INT8, about half the memory). The engine also parses `q4_0` and `k8v4`, but only for Strata's Qwen path; a GLM pack runs them as `fp16`, so the launcher refuses them | `fp16`, the engine's own |
  | `SERVED_NAME` | `model_name` | one name, no spaces: the name `/v1/models` reports and every answer carries (Maya answers a request whatever model it names) | `glm-5.3-flash` |
  | `LOOP_GUARD` | `loop_guard` | `true`, `false`. From v1.1.30 Maya's server watches each answer and its thinking for one exact block of tokens repeated back to back (3 times, or 4 for a block under 32 tokens, covering at least 96 tokens). The first time, it drops the repeats and applies a short repetition penalty; the second, it closes the thinking or ends the answer (finish `stop`). A request's own `"loop_guard"` wins | on, Maya's own |

  A request's own fields always win over these defaults, and so do the Chat settings saved from Maya's web page (`maya.shared-settings.json` next to the config): they fill effort and sampling before the config does. Two settings don't apply. `SPEC_N`: Maya's `--spec` is documented for its DFlash2 drafter, which the image doesn't carry, not for GLM's MTP block. `MAX_TOKENS`: the server takes a default output cap only from those saved Chat settings; `max_tokens` in `maya.json` is the vision encoder's. A request with no `max_tokens` runs to the end of the context.
- **Model load/unload endpoints (`MAYA_MODEL_MANAGEMENT`).** From v1.1.47 Maya serves llama.cpp's `/models/load`, `/models/unload` and `/models/sse` by default (Open WebUI's Models page uses them; so can any API client or command). These composes set no API key and publish on `BIND_HOST` (default `0.0.0.0`), so anyone who can reach the port can unload the model. The composes forward `MAYA_MODEL_MANAGEMENT`, which Maya's server reads ahead of its config: `MAYA_MODEL_MANAGEMENT=0` turns the endpoints off (they answer 404), and unset keeps Maya's default, on.
- **Runtime numbers from `/metrics`.** Maya serves Strata's `/metrics` document (its version in `engine.maya_version`), but its draft counts are only in its engine log (`/data/<quant>/maya.log`, "glm spec: N of M drafts accepted"). Given the container, `scripts/lib/strata_metrics.py` sums them, so `health.sh`, `verify-full` (its MTP acceptance check), `bench.sh` and `report.sh` read Maya's drafts the way they read Strata's.
- **The club conventions:** the `CLUB3090_RESTART` knob, `BIND_HOST` / `ESTATE_PORT` / `ESTATE_CONTAINER` / `ESTATE_GPUS`, and the host GID plus `umask 0002` on the data volume.

## DeepSeek-V4.1-Flash (maya-ds4)

Maya also runs DeepSeek-V4.1-Flash-1017 on a second engine of its own, `maya-ds4` (`src/ds4`): tensor parallel or a layer split, every CPU-side expert copied into RAM at load when it fits (`--mmap-experts auto`, no demand store), a DSpark drafter and the model's two Engram hash-table layers. From v1.1.27 it reads [ISTA-DASLab's GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), which no other engine in the catalog serves.

| Quant | Dual (2 cards) |
|---|---|
| **ISTA GSQ-RCO IQ2_XXS** (`ista-iq2-xxs`: experts IQ1_M / IQ2_XXS / IQ2_XS, dense Q5_K, 138.2 GB + the 110.6 GB IQ4_NL Engram file) | `strata-maya/deepseek41-flash-dual-ista-iq2-xxs` · 8047 |

- **Layer split, not tensor parallel.** The slug runs `--tp 1` on two GPUs: each card holds its own layers with whole experts. With `--tp 2` both cards hold the same experts, each split between them, so the card that also carries the drafter and the head fills first and the other keeps ~4 GB it can't use. Measured on two 3090s (2026-10-25, hand-run local test images, the same pack, drafter and expert profile; tok/s narrative / code / 6K prompt): `--tp 2` 44.71 / 51.03 / 488, layer split 41.08 / 48.54 / 755. Prompts read 55% faster for 5–8% less decode, which favours agents that re-read long prompts. `MAYA_TP=2` runs tensor parallel instead.
- **Measured on the slug** (2026-10-25, v1.1.28 image, through `switch.sh`, `bench.sh` on its own boot, 65,536 context, a freshly seeded expert profile, 24 CPU threads, the cards capped at 230 W): 45.34 / 52.46 tok/s decode (narrative / code), 799 / 911 / 1,055 tok/s prefill @6K / @10K / @60K, 82.7% of drafts kept, 23,700 MiB VRAM a card, host RAM 106 GB (MemAvailable fall, decimal). First start ready in 183 s (the SHA-256 check took 2 min), a later one in 56 s from a cold page cache. Tensor parallel was not re-measured on the slug. **At stock power** (370 / 420 W, 2026-10-26, its own boot): prefill 1,045 / 1,186 / 1,390 tok/s (+30–32%), decode 45.79 / 50.97 (the same within noise). Prompt reading is GPU-bound; decode is set by the CPU lane, so the power cap costs only prefill.
- **Drafter:** peasantsmith's `dspark-xs2` (Maya-XS's 2-bit DSpark drafter, 4.25 GB, from [peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)), mounted at `/dspark`. Maya's pack reads Maya's drafter format only, not unsloth's dspark GGUF. ISTA's GGUF carries no DSpark geometry, and Maya v1.1.27–v1.1.28's pack builder then writes 0, so the drafter would load and never draft. The launcher writes DeepSeek-V4.1-Flash's own values into the pack when it says 0 ([UPSTREAM.md](../UPSTREAM.md)). The draft block is 5, from that geometry; `maya-ds4` has no depth flag, so `SPEC_N` doesn't apply here either.
- **Launcher** (`models/deepseek-v4.1-flash-1017/strata-maya/launcher/`: `ds4-entrypoint.sh`, `ds4_prepare.py`, `ds4_settings.py`). Maya's DeepSeek pack can live away from the model files (its `ds4.txt` names the folder it indexes), so nothing is linked: the launcher checks the 7 files' SHA-256 against its own table (Maya's tables don't list ISTA's quant; minutes for ~255 GB, once), builds the pack with `tools/ds4/pack_gguf.py` on the data volume (seconds: an index, no weights copied), seeds the expert profile from `data/expert-profile-ds4.bin`, writes `maya-ds4.json` and starts `serve/server.py`. The data volume is the GLM slugs' (`strata-maya-data`), in a folder of its own (`ista-iq2-xxs/`).
- **Settings:** the same as the GLM slugs' table above, with DeepSeek's defaults: effort `high` (Maya's own for DeepSeek; DeepSeek's template turns low / high / max into its 1–100 "Reasoning Effort" budget), temperature 1.0 / top_p 1.0, the name `deepseek-v4.1-flash`. `KV` doesn't apply (`maya-ds4`'s cache is fp16). One more: `THREADS`, the CPU lane's threads (`--cpu-threads`). Unset, the launcher uses half the CPUs the container may use (nproc / 2, at least 1): a safe default on a host we know nothing about. A rig that has measured its own sets `THREADS` (a shell prefix, or `bash scripts/settings.sh set THREADS=<n>`; it isn't a per-slug setting, so the saved value also reaches the llama.cpp CPU-offload slugs, which read the same variable). The reference rig runs 24 of 32 vCPUs, the best count in Strata's port of this engine, where 16 cost 4–5% of decode and the engine's own default (a thread a core, two left spare: ~30 on a 32-core host) ~40%; it wasn't re-measured on `maya-ds4`, which ran 24 throughout.
- **RAM:** Maya copies every CPU-side expert into RAM when they fit beside 9 GB for the rest (~100 GB with this quant and drafter on two 24 GB cards); with less RAM it reads them from the files while it answers (slower, not measured).
- **Maya's `--calibrate` doesn't help here:** on two equal GPUs it tries only `--tp 2` and fewer CPU threads, never the layer split.

## MiniMax-M3.1-Flash (maya-mm3)

Maya runs MiniMaxAI's [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) (428B MoE, 23B active a token: 128 routed experts top-4 plus a shared one, 3 leading dense layers, a block-sparse attention indexer in every layer, 1M native context) on a third engine, `maya-mm3` (`src/mm3`, from v1.1.29; mw00/project-maya#477). The slugs serve Project Maya's own GGUF quant of it, from v1.1.30.

| Quant | Single (1 card) | Dual (2 cards) |
|---|---|---|
| **Maya-S** (`maya-s`: routed experts IQ2_XXS, the most sensitive IQ2_XS / IQ2_S / IQ3_XXS, ~2.1 bits; attention and indexers Q8_0, the rest Q6_K; 3 shards + the indexers side file, 122.5 GB) | `strata-maya/minimax31-flash-single-maya-s` · 8118 | `strata-maya/minimax31-flash-dual-maya-s` · 8059 |

- **⚠️ License:** the MiniMax Community License (`LICENSE` in the weights' repo): free for non-commercial use, commercial use has conditions.
- **Weights:** [peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF) @`4b40cc79`, `Maya-S/` (mounted at `/gguf`) and `dspark/` (mounted at `/companions/dspark`). The dense layers 0–2's sparse-attention indexers sit in a side file, `MiniMax-M3.1-Flash-Maya-S-maya-indexers.gguf`, beside the shards: Maya's pack reads both, and stock llama.cpp loads the shards alone. Maya's own setup takes the drafter from Mia-AiLab's EXL3 repo (`tools/mm3/download.json`); the GGUF repo's `dspark/` is the same files, every one hashed equal (2026-10-27).
- **Launcher** (`models/minimax-m3.1-flash/strata-maya/launcher/`: `mm3-entrypoint.sh`, `mm3_prepare.py`, `mm3_settings.py`). The launcher:
  - checks the 14 files' SHA-256 against Maya's own manifests (`tools/mm3/download_gguf.json` and `download.json`'s `dspark`), once;
  - builds the pack with `tools/mm3/pack_gguf.py --companions /companions` into `/data/minimax-maya-s/pack`. The pack names the files by their path from the model folder, so nothing is linked. A repack keeps the engine's prefill calibration for the PC (`prefill-tune.txt`);
  - writes `maya-mm3.json` and starts the server.

  The quant's folder is `minimax-maya-s`, so it never meets GLM's `maya-s` on the shared data volume.
- **Layer split:** a GGUF pack runs as a layer split over the cards (`maya-mm3`'s tensor parallel is its EXL3 path's). The experts VRAM doesn't hold are computed on the CPU from RAM (pinned).
- **K / V:** INT8 with rotated rows by default (the engine logs `K int8, V int8 (rows rotated)`), streamed: 32K positions of each layer in VRAM, every one in pinned RAM (63,360 bytes a token: ~13.0 GB at the slugs' 204,800 context; `KV=bf16` doubles it).
- **Drafter:** DSpark n=7, in the pack, **on**. Maya's own setup leaves it off until its speed calibration says drafting pays (`tools/calibrate_ds4.py`: the engine started twice, ~1,024 sustained tokens each, the drafter kept above +5%), since with most experts on the CPU a draft can cost more than it saves. On these cards it pays on both slugs (v1.1.31, one start of each kind):
  - two cards: 22.5 against 18.7 tok/s (+20.5%, 73% of drafts kept);
  - one card: 18.8 against 17.4 (+8.2%, 70%).

  Its 2.3 GB comes out of an expert pool: 1,625 experts resident instead of 1,800 on two cards, 146 instead of 320 on one. `DSPARK=off` turns it off.
- **Prefill calibration:** none needed on 3090s. Maya's `tools/calibrate_mm3_prefill.py` tries a prompt chunk above 1,024 only where every routed expert is in VRAM, and hipBLASLt only on AMD, so on these cards it measures the defaults alone (chunk 1024, cuBLAS) and keeps them. Without its `prefill-tune.txt` a start only logs a NOTE.
- **⚠️ The start's CPU-lane probe can misread.** At load the engine times one expert on the CPU lane against a copy to VRAM (`CPU lane vs staging (measured): an expert 0.31 ms on the CPU lane … 1 staged`) and keeps that choice for the session. One start of the single slug read 3.98 ms instead of the usual 0.25–0.35, staged every expert, and decoded ~21% slower. From v1.1.34 the engine measures a reading like that one (four experts no slower than one) again, up to three rounds, and the start line says so. A reading off the other way isn't caught, so the line is still worth a look: anything but `1 staged` is worth a restart ([mw00/project-maya#548](https://github.com/mw00/project-maya/issues/548), [UPSTREAM.md](../UPSTREAM.md)).
- **Settings:** the same as the GLM slugs' table above, with MiniMax's defaults and levels.
  - **Effort:** `auto` by default (Maya's own: MiniMax's adaptive thinking, the model decides whether to think). `low`, `medium`, `high`, `xhigh` and `max` reach the template as they are (MiniMax has five levels; for GLM, `medium` and `xhigh` fold into `high` and `max`). `none`, or `ENABLE_THINKING=false`, turns thinking off.
  - **Sampling and name:** temperature 1.0 / top_p 0.95, the model card's. The name is `minimax-m3.1-flash`.
  - **Three more settings:**
    - `KV` (`--kv-cache`): `int8` or `bf16`. fp8 isn't validated on these cards, and fp4 is below the stack's KV floor; both are refused.
    - `DSPARK` (`on` / `off`).
    - `THREADS` (`--cpu-threads`): half the CPUs by default, as DeepSeek's. From v1.1.34 that's the lane's threads in all; earlier images started one more (mw00/project-maya#552).
- **Context:** 204,800. Maya's own setup defaults to 65,536 and recommends more where the PC's memory holds it; the model reaches 1M, and Maya has run 256K end to end.
- **Measured on the slugs** (2026-10-27, v1.1.31 image `c94aa0e1`, through `switch.sh`, `bench.sh` on each slug's own boot, DSpark on, thinking off, the cards capped at 230 W):

  | Slug | Decode narrative | Decode code | Prefill @10K / @60K | Drafts kept | Experts in VRAM | VRAM | Host RAM |
  |---|---|---|---|---|---|---|---|
  | dual | 20.13 tok/s (CV 2.6%) | 20.30 (18.31–24.56) | 167 / 147 tok/s | 80.6% | 1,625 of 7,296 (layers 0–33 / 34–59) | 23,376 / 23,360 MiB | 102 GB |
  | single | 18.12 (CV 0.9%) | 20.67 (15.70–22.41) | 181 / 170 | 80.3% | 146 | 23,410 MiB | 124 GB (`/metrics`: 7,150 expert slots, 100.4 GB, in RAM) |

  - **Code decode varies within a boot.** On two cards it fell from 24.56 to ~18.3 tok/s over the five runs while GPU0's clock held at ~1,770 MHz and the short prompt's first token went from 67 to 85 ms. Narrative stays within a few percent, so compare on that: 18.37 tok/s on v1.1.30 with DSpark on, 20.13 on v1.1.31.
  - **One card reads prompts faster than two** (8% at 10K, 16% at 60K). Not explained yet; upstream has noted it for a closer look ([mw00/project-maya#549](https://github.com/mw00/project-maya/issues/549), closed on its `--bench` fix; [UPSTREAM.md](../UPSTREAM.md)).
  - **Drafts kept** is the bench's figure under `bench.sh`'s sampler (0.6 / 0.95 / top-k 20); the model card's 1.0 / 0.95, which the slugs serve, may keep fewer.
  - Host RAM is the `MemAvailable` fall, decimal, at its peak through the bench on a 224 GB host. `verify-full` passes every applicable check on both.
  - The first starts (v1.1.30, DSpark off) were ready in 72 s (dual) and 48 s (single).
  - Not measured yet: quality.

## Tuning for your PC

Maya's calibration measures decode at a few splits of each token's RAM-tier experts between the CPU and the PCIe link, and at fewer CPU threads, then keeps the fastest. It takes about 10–15 minutes, because the model loads first. Boot the slug you want to tune first and let it come up: the calibration reads the config that boot wrote, and all the topologies of one quant share its folder on the data volume, so a config last written by the one-card slug tunes one card. Then stop it, since the calibration needs the GPUs to itself. For `strata-maya/glm53-flash-single-maya-s`:

```bash
bash scripts/switch.sh --down
f="$(bash scripts/settings.sh compose-env-file)"
CLUB3090_DATA_DIR="${CLUB3090_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/club-3090}" \
  docker compose --env-file "$f" -f models/glm-5.3-flash/strata-maya/compose/single/maya-s/base.yml \
  run --rm --entrypoint /bin/sh strata-maya -c \
  'umask 0002; cd /opt/maya && exec .venv/bin/python maya.py --calibrate --no-start --yes --plain --config /data/maya-s/maya.json'
docker compose --env-file "$f" -f models/glm-5.3-flash/strata-maya/compose/single/maya-s/base.yml down
rm -f "$f"
```

For another slug, use its compose and its quant's folder (`/data/maya-m/…`, `/data/maya-l/…`).
- **Keep `--no-start`.** Without it Maya starts a server when it finishes, from the config our launcher wrote: listening on every address in the container, with no API key.
- **It prints** each split and thread count it tried, then the settings it changed, e.g. `tuned for this PC: STRATA_GLM_PCIE_SHARE=0.10 (32.3 tok/s)`, or that the engine's own settings are already the fastest here.
- **The `down`** removes the network the one-off run created; without it, each tuned quant leaves one behind.
- **Save what it printed as settings**, because the launcher writes Maya's config afresh at every boot and drops what the tuning wrote there. `STRATA_GLM_PCIE_SHARE` becomes `MAYA_PCIE_SHARE` and `STRATA_GLM_CPU_LANE` becomes `MAYA_CPU_LANE`: `bash scripts/settings.sh set MAYA_PCIE_SHARE=0.10`. Every Maya slug reads a saved value, whatever its quant or card count, so save the result of the slug you run. `switch.sh --set` doesn't accept these two settings yet.

## Known gaps

- **Quality:** not validated here (no 8-pack yet); the KL figures above are upstream's.
- **Reasoning:** GLM-5.3's template thinks by default. Maya's server closes the thinking block when a request sends `enable_thinking: false` or `reasoning_effort: none`, unlike the llama.cpp GLM slugs, where thinking can't be turned off. The server default for requests that name none is effort `max`, as on the other GLM-5.3-Flash slugs (Maya's own setup writes `high`), with Maya's own 32K reasoning budget. `REASONING_EFFORT`, `ENABLE_THINKING` and `THINKING_BUDGET` change these server-side (above). `thinking-budget.sh` (per-effort budgets) doesn't apply: Maya has one budget for every effort.
- **Vision and DFlash2:** not in the image. Maya's DFlash2 drafter (+12% on two cards, +19% on one 3090, by upstream's measurements) has CC BY-NC-ND weights.
- **Platforms:** x86-64 only. DGX Spark (ARM) can't run it. V100s need a native build (CUDA 12).
