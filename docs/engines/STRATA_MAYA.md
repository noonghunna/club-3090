# Project Maya — a Strata fork for GLM-5.3-Flash on one, two or four cards, DeepSeek-V4.1-Flash on two, MiniMax-M3.1-Flash on one or two, and Hy4-Flash on one to four

**Engine id:** `strata-maya` · **Slug prefix:** `strata-maya/` · **Status of its slugs:** 🐣 Incubating · **Requires:** NVIDIA compute capability 7.5+, driver ≥ 580 (CUDA 13), an x86-64 CPU with AVX2, and enough host RAM to hold the experts that don't fit in VRAM (~65 GB for Maya-S on two cards up to ~148 GB for Maya-L on one, measured; less RAM still runs, reading the rest from the model files)

## What it is

[Project Maya](https://github.com/mw00/project-maya) (MIT, mw00) is a fork of [Strata](https://github.com/Niko1221/Strata) rewritten for one model, Z.ai's GLM-5.3-Flash (320.76B MoE), with its own quants ([peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)). It keeps Strata's shape: a C++/CUDA engine (`build/maya`; `build/strata` before v1.1.16) with ggml's CPU backend as a library, and a Python server (`serve/server.py`) on port 8080 with an OpenAI API (`/v1/chat/completions`), an Anthropic one (`/v1/messages`) and a dashboard.

Where the experts live:
- **VRAM:** a per-layer expert pool on each card, filled from a shipped routing profile and promoted as it runs;
- **host RAM:** a pinned tier sized from the memory free at start;
- **the model files:** whatever fits in neither, read while it answers.

Each token's RAM-tier experts are split between the CPU (computed in place) and PCIe copies to the GPU, priced per expert at start. On several cards it splits the **layers** (`--layer-split auto`, a placement search over the cards' free VRAM), and for GLM on two or more it drafts with the model's own MTP (NextN) block, the cards working on consecutive tokens (Hy4-Flash's `maya-hy4` drafts with its MTP block on one card too). No NCCL and no P2P.

Its profile declares `type: strata` and `lineage.relation: fork` (base `strata`); its id carries the family prefix, so the tooling knows it as engine kind `strata` (`scripts/lib/engine-kind.sh`): container names `strata-maya-…`, image `strata-maya-club3090`.

## Image

Maya publishes no image and no Linux engine: `maya.sh` compiles on the user's PC. From v1.1.53 its repository has a recipe for building an image yourself ([`docs/DOCKER.md`](https://github.com/mw00/project-maya/blob/main/docs/DOCKER.md), adapted from ours in [mw00/project-maya#808](https://github.com/mw00/project-maya/pull/808)), but no published image. So club-3090 builds `ghcr.io/noonghunna/strata-maya-club3090` for each release, an image-provenance exception recorded in [ADDING_ENGINES.md](../ADDING_ENGINES.md).
- **Build:** our own Dockerfile compiles the engines (`maya` for GLM, `maya-ds4` for DeepSeek since #1801, `maya-mm3` for MiniMax-M3.1-Flash in the releases that define it, v1.1.29 on, `maya-hy4` for Hy4-Flash, v1.1.68 on) the way `maya.py` does, with upstream's `STRATA_PORTABLE=ON` (ggml's CPU code for an AVX2 baseline instead of the build machine's CPU) and a fixed arch list. The runtime stage carries the release's files, llama.cpp's `gguf-py` and the server's Python packages.
- **Archs:** `75;80;86;89;120` (RTX 20–50 and the A-series). V100 needs CUDA 12, which this image can't build; Maya's own setup builds it natively.
- **GLM is text only:** neither GLM's vision encoder nor its DFlash2 drafter (CC BY-NC-ND weights) is in the image. Hy4-Flash's vision tower is part of `maya-hy4`.
- **Recipe, checks, publishing and pin flow:** [models/glm-5.3-flash/strata-maya/image/](../../models/glm-5.3-flash/strata-maya/image/README.md).

The profile pins a digest, and the launchers inject it as `STRATA_MAYA_IMAGE`. Export `STRATA_MAYA_IMAGE` in the shell to try another build for one launch; a saved value loses to the pin.

**Bumping:** the maintainer publishes a release from the rig with `build.sh --tag <vX.Y.Z> --push` (`:<tag>`, `:sha-<commit>`), but publishing doesn't pin. What moving the pin checks, and when to skip a release: [the bump runbook](../NIGHTLY_BUMP_RUNBOOK.md#strata-and-project-maya-the-bump-rule).

A release that renames a flag the launchers pass (`serve/server.py`, `tools/iq_pack.py`; for DeepSeek, `maya-ds4` and `tools/ds4/pack_gguf.py`; for MiniMax, `maya-mm3` and `tools/mm3/pack_gguf.py`; for Hy4-Flash, `maya-hy4` and `tools/hy4/pack_gguf.py`) fails the image build, so the recipe and the launchers move together.

## Slugs

| Quant | Single (1 card) | Dual (2 cards) | Multi4 (4 cards) |
|---|---|---|---|
| **Maya-S v2** (`maya-s`, IQ2_XXS experts, 96.5 GB) | `strata-maya/glm53-flash-single-maya-s` · 8027 | `strata-maya/glm53-flash-dual-maya-s` · 8026 | `strata-maya/glm53-flash-multi4-maya-s` · 8028 |
| **Maya-S24** (`maya-s24`, Maya-S v2's IQ2_XXS experts with Q4_K attention and shared experts, 94.7 GB) | — | `strata-maya/glm53-flash-dual-maya-s24` · 8194 | `strata-maya/glm53-flash-multi4-maya-s24` · 8207 |
| **Maya-M** (`maya-m`, IQ2_S experts, 116 GB) | `strata-maya/glm53-flash-single-maya-m` · 8035 | `strata-maya/glm53-flash-dual-maya-m` · 8029 | `strata-maya/glm53-flash-multi4-maya-m` · 8037 |
| **Maya-L** (`maya-l`, IQ3_S experts, 156.3 GB) | `strata-maya/glm53-flash-single-maya-l` · 8044 | `strata-maya/glm53-flash-dual-maya-l` · 8043 | `strata-maya/glm53-flash-multi4-maya-l` · 8045 |

- **Weights:** Maya's own quants from `peasantsmith/GLM-5.3-Flash-Maya-GGUF` (revision `f16ee4ba`), one folder per quant under `glm-5.3-flash-maya-gguf/`. Each carries the MTP draft block. Upstream's quality figures (KL against the FP8 model on held-out text, its own harness): Maya-L 0.188, Maya-M 0.291, Maya-S 0.428. Smaller quants fit more experts in VRAM, so they decode faster.
- **multi4 is never booted on the reference rig** (two cards): community-validated by design. Its numbers come from community reports.
- **One card has no drafting:** GLM's MTP block runs only on two or more GPUs (Hy4-Flash's drafts on one card too).

**Data volume:** `strata-maya-data` under the data directory (`~/.local/share/club-3090` by default; `STRATA_MAYA_DATA_DIR` overrides), one folder per quant: a symlink per shard, Maya's pack (its index of the model files, ~1 GB, written next to the shards as the engine expects) and the config.

**First start:** the launcher checks each shard's SHA-256 against Maya's own table (a few minutes for 100–156 GB, once; again only when a file's size or time changes) and builds the pack (a minute or a few, once). Later starts take about 30 seconds on the reference rig. That is with the shards in the page cache: from a hard disk with a cold cache, every start reads the whole pack, which took 930–1,010 s for Maya-L and Maya-M on a community 2× RTX 5090 rig. That is longer than `switch.sh`'s 10-minute wait, so give it `READY_TIMEOUT=1800` (or re-run `switch.sh` once it times out; the container keeps loading). Dropping the page cache before a boot (advice below, so Maya sizes its RAM tier from real free memory) makes the next start a cold one. `SETUP_ONLY=1` runs just that; `REPACK=1` builds the pack again; `VERIFY=0` skips the hash check.

**RAM:** Maya sizes its RAM tier from the memory free at start. Drop the page cache first on a box that just read big files, or the tier comes out smaller than it could be. On a host with less RAM than the measured figure, the experts that don't fit are read from the model files while it answers (slower; not measured here).

**Requests and readiness:** one request at a time. `switch.sh`'s readiness probe (one completion) passes once the server prints `ready: http`.

Measurements, with dates and settings, are in each compose's header and in [BENCHMARKS.md](../../BENCHMARKS.md) (GLM-5.3-Flash, the Project Maya table).

## What club-3090 adds

- **A launcher for every compose** (`models/glm-5.3-flash/strata-maya/launcher/`: `maya-entrypoint.sh` and `maya_prepare.py`). Maya's own setup isn't used in the image. The model folder is mounted read-only, but Maya writes its pack next to the shards, so the launcher links the shards into a folder of their own on the data volume, checks their SHA-256 there, builds the pack with Maya's `tools/iq_pack.py` (and stamps it as `maya.py` does, so a changed shard rebuilds it), writes the config and starts `serve/server.py` on the cards `MAYA_GPUS` names. The compose passes the quant as the launcher's one argument, not as a setting, so no saved value can point a slug at another quant's pack.
- **Calibration knobs:** `MAYA_PCIE_SHARE` and `MAYA_CPU_LANE` become the engine's `MAYA_GLM_PCIE_SHARE` / `MAYA_GLM_CPU_LANE` when set (the launcher exports them under those names; until 2026-11-17 it used the older `STRATA_GLM_*` names, which Maya maps to these). They are what Maya's own calibration tunes for a PC; the composes ship the engine's own choice. See [Tuning for your PC](#tuning-for-your-pc).
- **Launch settings, written into Maya's config at every boot** (`launcher/maya_settings.py`). The launcher writes `maya.json` afresh at each start, so a setting you remove is gone at the next boot. A bad value stops the boot with the reason, before the SHA-256 check. `REASONING_EFFORT`, `ENABLE_THINKING` and `CONTEXT` are catalogued, so `bash scripts/switch.sh --set <slug> REASONING_EFFORT=high` saves them for one slug; save the others with `bash scripts/settings.sh set TEMP=0.7` (they apply to every slug that reads them) or give them as a shell prefix.

  | Setting | Maya config | Values | Unset |
  |---|---|---|---|
  | `REASONING_EFFORT` | `reasoning_effort` | `none`, `low`, `high`, `max`; `medium` runs as `high` and `xhigh` as `max` (GLM-5.3 has four levels, and Maya's server maps the names so) | `max`, as on the other GLM-5.3-Flash slugs (Maya's own setup writes `high`) |
  | `ENABLE_THINKING` | `reasoning_effort: none` when `false` | `true`, `false`; `false` wins over `REASONING_EFFORT` | thinking on |
  | `THINKING_BUDGET` | `thinking_budget` | tokens, `0` = no budget; one cap for every request. A request's own `max_tokens` lowers it to leave the answer a quarter of that (at least 1,024 tokens) | Maya's own, 32,768 |
  | `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `PRESENCE_PENALTY`, `REPEAT_PENALTY` | `sampling.temperature`, `.top_p`, `.top_k`, `.min_p`, `.presence_penalty`, `.repetition_penalty` | ≥ 0 / 0–1 / 1–64 / 0–1 / ≥ 0 / > 0 | temperature 1.0, top_p 0.95 (GLM-5.3's vendor sampling); the rest the engine's |
  | `KV` | `--kv` in the engine args | `fp16`, `int8` (the attention's latent cache in INT8, about half the memory). The engine also parses `q4_0` and `k8v4`, but only for Strata's Qwen path; a GLM pack runs them as `fp16`, so the launcher refuses them | `int8`, the engine's own from v1.1.55 (`fp16` before): the engine logs `latent cache INT8`. `KV=fp16` keeps 16 bits |
  | `SERVED_NAME` | `model_name` | one name, no spaces: the name `/v1/models` reports and every answer carries (Maya answers a request whatever model it names) | `glm-5.3-flash` |
  | `LOOP_GUARD` | `loop_guard` | `true`, `false`. From v1.1.30 Maya's server watches each answer and its thinking for one exact block of tokens repeated back to back (3 times, or 4 for a block under 32 tokens, covering at least 96 tokens). The first time, it drops the repeats and applies a short repetition penalty; the second, it closes the thinking or ends the answer (finish `stop`). A request's own `"loop_guard"` wins | on, Maya's own |
  | `CONTEXT` | `--max-context` in the engine args | a whole number ≥ 4096 (the same floor the Strata recipe checks); past the compose header's Max ctx unvalidated — the headers carry the measured ceilings. A value above the compose's default needs more host RAM than the header's figures, which are at the default context (MiniMax keeps every position's K/V in RAM: ~63,360 B/token at INT8, ~12 GiB at 204,800, ~66 GB at 1M). Catalogued: `switch.sh --set <slug> CONTEXT=65536` saves it for one slug, and `--explain` shows where the value comes from | the compose's own: 131,072 on the GLM slugs (262,144 on Maya-S24), 262,144 on DeepSeek's ISTA and Maya-XS slugs (65,536 on UD-IQ3_S), 204,800 on MiniMax |

  A request's own fields always win over these defaults, and so do the Chat settings saved from Maya's web page (`maya.shared-settings.json` next to the config): they fill effort and sampling before the config does. Two settings don't apply. `SPEC_N`: Maya's `--spec` is documented for its DFlash2 drafter, which the image doesn't carry, not for GLM's MTP block. `MAX_TOKENS`: the server takes a default output cap only from those saved Chat settings; `max_tokens` in `maya.json` is the vision encoder's. A request with no `max_tokens` runs to the end of the context.
- **Model load/unload endpoints (`MAYA_MODEL_MANAGEMENT`).** From v1.1.47 Maya serves llama.cpp's `/models/load`, `/models/unload` and `/models/sse` by default (Open WebUI's Models page uses them; so can any API client or command). These composes set no API key and publish on `BIND_HOST` (default `0.0.0.0`), so anyone who can reach the port can unload the model. The composes forward `MAYA_MODEL_MANAGEMENT`, which Maya's server reads ahead of its config: `MAYA_MODEL_MANAGEMENT=0` turns the endpoints off (they answer 404), and unset keeps Maya's default, on. From v1.1.50 the same setting also covers Maya's model API (`/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull`, `/api/delete`). Without an API key, that model API takes a download, a delete or a model switch only from a client on the same machine as the server, which in a container is the container itself, so through the published port those answer 403. Maya's web page has its own routes for the same jobs (`/api/models/switch`, `/api/models/delete`, `/api/models/leftovers`): `MAYA_MODEL_MANAGEMENT` doesn't cover them, and without an API key they take any JSON request that doesn't come from another website. In these containers they find nothing to act on, and answer 400. A delete or a switch names a `maya-*.json` config other than the running one, and each slug's data folder holds only its own config (the GLM slugs' is `maya.json`). Deleting the running model, or switching away from it, needs Maya's own start script, which the image doesn't use. And "files no model uses" are looked for only in the models folders Maya's setup downloads into, which these containers don't have. An `/api/chat` request's `keep_alive` sets how long the model stays loaded once idle, so a client of that API can make it unload; the next request loads it again.
- **v1.1.54 to v1.1.56 (pinned 2026-11-07), what reaches these slugs:**
  - **GLM on two cards:** v1.1.56 fixes the drafting decode that since v1.1.52 could crash ("illegal memory access") or slow to a crawl on some two-GPU PCs. A two-GPU split drafts as in v1.1.51 again (mw00/project-maya#860).
  - **GLM KV:** from v1.1.55 the attention's latent cache is INT8 by default (`MAYA_GLM_KV_INT8`, which Maya's own setup already wrote for new installs): half the cache's VRAM. Upstream checked it against the full model: the same quality as 16 bits, and no looping. With `KV` unset the GLM slugs now run INT8; `KV=fp16` keeps 16 bits.
  - **GLM sampling:** a sampled request without a `seed` gets a fresh random seed (v1.1.56, #845), so asking again gives a different answer. A `seed` in the request still repeats exactly.
  - **DeepSeek:** the loop guard now also catches DeepSeek's repeated tool calls; before, it never fired there (#844).
  - **The server, every model:**
    - `/v1/responses` accepts what Codex CLI sends (v1.1.55, #726): grouped tools, built-in tools Maya can't run left out, `reasoning.summary`, encrypted reasoning.
    - Files arrive as text through OpenAI `file` / Responses `input_file` parts and Anthropic `document` blocks (v1.1.56, #837).
    - Web-chat answers run to the end and are kept for a day in `answers/` next to the config, on the data volume (v1.1.55).
    - A re-sent conversation is prepared by increment, with the same token ids (`MAYA_PROMPT_MEMO`, on).
  - **Packs:** from v1.1.56 the pack tool keeps GLM's expert selection bias in F32 (#861) for packs built from then on. A pack already on the data volume keeps the rounded bias until `REPACK=1`.
  - **Newly forwarded:** `MAYA_GLM_P2P` on the GLM dual and multi4 composes (unset = off). With `=1`, the layer split sends its token states GPU to GPU on any pair the driver grants peer access, not only the MI50. Upstream says the answers are the same. On the reference rig (P2P on) the engine logged `glm settings from the environment: MAYA_GLM_P2P=1` and verify-full passed, with no speed gain visible: the best decode stat of one verify-full run was 53.4 tok/s against 56.3 without it.
  - **Not forwarded:**
    - AMD-only, EXL3-only and Pascal-only switches, which don't reach these GGUF slugs on NVIDIA.
    - DeepSeek's new prompt and loading defaults (`DS4_AI_RUN`, `MAYA_DS4_AR2`, `MAYA_DS4_LOAD_INTERLEAVE`): speed only, the same answers.
    - `MAYA_PROMPT_MEMO`: the same token ids.
- **v1.1.57 (pinned 2026-11-09), what reaches these slugs:**
  - **The server, every model:** a SIGTERM or Ctrl+C right as the server was waiting could hang it for good; it now stops. `docker stop`, which the slugs' teardown runs, sends SIGTERM.
  - **Maya's own model API** (`/api/chat`, `/api/generate`): `"think": false` now turns thinking off; before, thinking stayed on for every model. The OpenAI, Responses and Anthropic APIs weren't affected.
  - **DeepSeek:** a video in a request gets a clear 400 ("reads pictures, not videos") instead of an encoder error, or being dropped beside a picture.
  - **Prompt path:** the release also changes headers the GLM prompt path shares (`moe_mmq.hpp`, `glm_kv_stream.hpp`), so the bump ran a long-prompt needle: 72,139 tokens on the dual Maya-L slug, found.
  - **Not forwarded:** the release's new settings (`MAYA_GLM_IDX_FP16`, `MAYA_GLM_DSA_GRAPHS`, `MAYA_GLM_LANE_ROW4`, `MAYA_GLM_NGRAM_DRAFT` / `_MIN` / `_MAX`, `MAYA_GLM_PROMPT_CACHE_*`, `MAYA_GLM_SPLIT_CARDS`) are GLM-5.3 (744B)'s own, in upstream's `docs/SETTINGS.md`. No slug serves that model: its UD-IQ2_M files are 238.6 GB, and upstream recommends about 256 GB of VRAM and RAM together.
- **v1.1.58 to v1.1.61 (pinned 2026-11-13), what reaches these slugs:**
  - **GLM-5.3-Flash drafting (v1.1.58):** Q5_K weights take the fast multi-token path (the same answers), and answers in Chinese, Japanese, Korean and other scripts are drafted again.
  - **The server, every model:**
    - The loop guard no longer keeps a finished short answer going, and an answer it does keep continues on a new paragraph (v1.1.58, #921).
    - Responses API pictures with any `detail` value are accepted (v1.1.58, #914).
    - With `--parallel` and fair slices, one waiting request pauses one running answer, not all of them (v1.1.59). It doesn't reach these slugs: the launcher writes no `--parallel` (one conversation at a time).
  - **MiniMax (v1.1.61):** on one GPU the prompt lookup is on by default: where an answer repeats earlier text (code edits, tool-call arguments, quotes), the tokens that followed it are checked in one step. Upstream measured code and tool calls about 6% faster with DSpark on. It reaches the single MiniMax slug, which this bump didn't boot. On two GPUs it stays opt-in (`--suffix-draft 3`), which the dual slug doesn't set.
  - **DeepSeek:**
    - On NVIDIA tensor-parallel pairs, the prompt all-reduce goes GPU to GPU again, as in v1.1.54 (v1.1.59). Our DeepSeek slugs run a layer split unless `MAYA_TP=2` is set, and `MAYA_DS4_HIP_PEER_AR` isn't forwarded.
    - A failed `DS4_LENT_RAM_GB` copy no longer makes every request fail with HTTP 400 (v1.1.59, #884).
  - **Packs:** v1.1.58 keeps the 744B GLM-5.3's Q8_0 attention weights in Q8_0. That is `tools/strata/glm_dsa_pack.py`; GLM-5.3-Flash's packer (`tools/iq_pack.py`) is unchanged, so a Flash pack on the data volume is reused (the bump's boot logged `pack up to date`).
  - **The API key variable:** the server's `--api-key` now defaults to `$MAYA_API_KEY` (it was `$STRATA_API_KEY`). Our launcher passes neither, so the slugs stay keyless as before.
  - **Not forwarded:**
    - The release's new settings:
      - `MAYA_GLM_PREFILL_STABLE`: GLM prompts that don't depend on what the expert cache holds, for A/B checks; opt-in, about 2-5% slower prompts.
      - `MAYA_GLM_ROUTE_AFFINITY`: Turbo's routing on the 744B GLM-5.3; it changes the answers.
      - `MAYA_GLM_PIN_ROOM_MB`: Windows only.
      - `MAYA_DS4_LOOKUP_TRUST` / `MAYA_MM3_LOOKUP_TRUST`: on by default, and they apply only with `--suffix-draft`.
      - `DS4_STREAM_ONE_WARM`: DeepSeek's one-GPU path, on by default.
    - DeepSeek's opt-in fast prompt reading (`--fast-prefill on`; the answers can shift).
    - The opt-in refusal ablation.
    - The AMD-only and EXL3-only changes, and v1.1.60 as a whole (AMD, an EXL3 quant, DFlash2 for the 744B GLM-5.3).
- **v1.1.66 and v1.1.67 (pinned 2026-11-18), what reaches these slugs:**
  - **GLM correctness fix (v1.1.67):** on a PC where part of GLM's experts sit in RAM, an expert moved into VRAM twice at the same moment could later be looked up in a RAM slot that by then held another expert, and computed with that one's weights. There was no error, only slightly wrong answers, more often the larger the share kept in RAM. On 3090s every GLM slug here keeps experts in RAM: the smallest quant, Maya-S24, is 94.7 GB, as much as four 3090s' VRAM, and Maya-L is 156.3 GB. Upstream says answers can differ from v1.1.66 because they are now correct, at the same speed.
  - **DeepSeek (v1.1.66):** the drafter's experts now count in the expert profile (mw00/project-maya#501), so which experts sit in VRAM can change as the profile fills, and answers can differ slightly from v1.1.65.
  - **Newly forwarded** (unset = Maya's own, which is what the slugs ran before):
    - **The idle sleeps** (v1.1.66, mw00/project-maya#951): `MAYA_GLM_SERVICE_NAP_US` / `MAYA_GLM_SERVICE_SLEEP_MS` on the 11 GLM composes, `MAYA_DS4_SERVICE_NAP_US` / `MAYA_DS4_SERVICE_SLEEP_MS` on the 3 DeepSeek ones. They set how long an idle engine's service thread sleeps between looks for work: the short nap (microseconds, 1 to 10,000; default 100), then the longer sleep after a longer quiet (milliseconds, 1 to 1,000; default 2). A value outside the range is brought to the nearer end, and a non-number keeps the default. Longer sleeps lower a waiting server's CPU use; the first token after a pause can wait up to one sleep. For example `MAYA_GLM_SERVICE_SLEEP_MS=20 bash scripts/switch.sh --force strata-maya/glm53-flash-dual-maya-l`, or `bash scripts/settings.sh set MAYA_GLM_SERVICE_SLEEP_MS=20` to keep it for every launch. `switch.sh --set` takes only the catalogued launch settings, so it refuses these, as it does the other engine pass-throughs.
    - **Two GLM diagnostics**, off by default, on the 11 GLM composes. `MAYA_GLM_PREFILL_CHECK=1` (v1.1.66) checks every stage of a prompt layer and names the one that faulted; prompts are slower while it's on, and the engine log says `MAYA_GLM_PREFILL_CHECK is on`. `MAYA_GLM_TIER_VERIFY=1` (v1.1.67) checks GLM's expert tables after every token and reports an entry that points at the wrong place. Both only report.
  - **Not forwarded:**
    - `MAYA_PAR_MAX` (v1.1.66): it lowers a config's `--parallel` before the engine starts, and the launchers write no `--parallel` (one conversation at a time), so it would change nothing.
    - `MAYA_GLM_HELPER_WARM` (v1.1.67) and v1.1.66's faster decode on the AMD APU + NVIDIA pair: both need Maya's expert helper on a second, integrated GPU (`MAYA_GLM_EXPERT_HELPER`), which these composes don't start.
    - Setup and Windows changes (the fit estimate before a download, the model menu, the long-path unpack fix): Maya's own setup, which the launcher doesn't use.
- **Runtime numbers from `/metrics`.** Maya serves Strata's `/metrics` document (its version in `engine.maya_version`), but its draft counts are only in its engine log (`/data/<quant>/maya.log`, "glm spec: N of M drafts accepted"). Given the container, `scripts/lib/strata_metrics.py` sums them, so `health.sh`, `verify-full` (its MTP acceptance check), `bench.sh` and `report.sh` read Maya's drafts the way they read Strata's.
- **The club conventions:** the `CLUB3090_RESTART` knob, `BIND_HOST` / `ESTATE_PORT` / `ESTATE_CONTAINER` / `ESTATE_GPUS`, and the host GID plus `umask 0002` on the data volume.

## DeepSeek-V4.1-Flash (maya-ds4)

Maya also runs DeepSeek-V4.1-Flash-1017 on a second engine of its own, `maya-ds4` (`src/ds4`): tensor parallel or a layer split, every CPU-side expert copied into RAM at load when it fits (`--mmap-experts auto`, no demand store), a DSpark drafter and the model's two Engram hash-table layers. From v1.1.27 it reads [ISTA-DASLab's GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), which no other engine in the catalog serves, and unsloth's GGUF quants, of which Maya's own setup lists [UD-IQ3_S](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (`--ds4-quant gguf-iq3_s`). From v1.1.22 it lists its **own** two quants of it too — `--ds4-quant maya-s` and `maya-xs` ([peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)) — and Maya-XS is the one a ~92 GB host can hold.

| Quant | Dual (2 cards) | Multi4 (4 cards) |
|---|---|---|
| **ISTA GSQ-RCO IQ2_XXS** (`ista-iq2-xxs`: experts IQ1_M / IQ2_XXS / IQ2_XS, dense Q5_K, 138.2 GB + the 110.6 GB IQ4_NL Engram file) | `strata-maya/deepseek41-flash-dual-ista-iq2-xxs` · 8047 | — |
| **unsloth UD-IQ3_S** (`ud-iq3-s`: experts IQ2_S / IQ3_XXS / IQ3_S / Q3_K / MXFP4 by layer, dense Q8_0 / BF16, 8 shards, 423.6 GB with the 161.3 GB Q6_K Engram shard; needs ~229 GB of host RAM) | `strata-maya/deepseek41-flash-dual-ud-iq3-s` · 8089 | `strata-maya/deepseek41-flash-multi4-ud-iq3-s` · 8208 |
| **Maya Maya-XS v2** (`maya-xs`: every routed expert IQ1_S with error-feedback rounding, dense Q8_0, 3 shards, 116.8 GB + the 161.3 GB Q6_K Engram pair it shares with Maya-S; Maya's manifest says 121 GB of VRAM + RAM hold it, its 4.3 GB drafter included — the DeepSeek quant for a host with ~92 GB of RAM) | `strata-maya/deepseek41-flash-dual-maya-xs` · 8146 | — |

- **Layer split, not tensor parallel.** The slug runs `--tp 1` on two GPUs: each card holds its own layers with whole experts. With `--tp 2` both cards hold the same experts, each split between them, so the card that also carries the drafter and the head fills first and the other keeps ~4 GB it can't use. Measured on two 3090s (2026-10-25, hand-run local test images, the same pack, drafter and expert profile; tok/s narrative / code / 6K prompt): `--tp 2` 44.71 / 51.03 / 488, layer split 41.08 / 48.54 / 755. Prompts read 55% faster for 5–8% less decode, which favours agents that re-read long prompts. `MAYA_TP=2` runs tensor parallel instead.
- **Four cards** (`strata-maya/deepseek41-flash-multi4-ud-iq3-s`; never booted on the reference rig: multi4 is community-validated by design). It runs the same layer split over four cards (`MAYA_TP=1`). Tensor parallel there is opt-in, and with UD-IQ3_S `MAYA_TP=4` alone does not form a group: the experts' down projections include Q3_K (256-value blocks, layers 10–14), which can't split DeepSeek's FFN width of 2,304 into four whole parts (nor in two, and the 64 attention heads rule out three), so the engine logs `tensor parallel 4 -> 1` and runs the layer split (`ds4_model.cu`; the width and types from the pack's `--pack-info`). `MAYA_TP=4 MAYA_DS4_TP_GGUF_UNEVEN=1` forms the four-wide group, whole blocks 3 / 2 / 2 / 2 a member: the shape Hazelhof measured for Maya-XS and ISTA on four RTX 5080s ([#2047](https://github.com/noonghunna/club-3090/issues/2047), [#2054](https://github.com/noonghunna/club-3090/issues/2054)), not with this quant. Its host RAM figure is the dual slug's 229 GB, as an upper bound: four cards hold more experts in VRAM, so the CPU lane to copy is smaller.
- **Measured on the slug** (2026-10-25, v1.1.28 image, through `switch.sh`, `bench.sh` on its own boot, 65,536 context, a freshly seeded expert profile, 24 CPU threads, the cards capped at 230 W): 45.34 / 52.46 tok/s decode (narrative / code), 799 / 911 / 1,055 tok/s prefill @6K / @10K / @60K, 82.7% of drafts kept, 23,700 MiB VRAM a card, host RAM 106 GB (MemAvailable fall, decimal). First start ready in 183 s (the SHA-256 check took 2 min), a later one in 56 s from a cold page cache. Tensor parallel was not re-measured on the slug. **At stock power** (370 / 420 W, 2026-10-26, its own boot): prefill 1,045 / 1,186 / 1,390 tok/s (+30–32%), decode 45.79 / 50.97 (the same within noise). Prompt reading is GPU-bound; decode is set by the CPU lane, so the power cap costs only prefill.
- **Measured on the UD-IQ3_S slug** (2026-11-02, v1.1.48 image, through `switch.sh`, `bench.sh` on its own boot, 65,536 context, 24 CPU threads, the cards capped at 230 W, no drafter): 34.68 / 33.44 tok/s decode, 557 tok/s prefill @10K, 24,102 / 24,104 MiB VRAM, host RAM 229 GB (all 218.2 GB of the CPU lane's experts copied). First start ready in 352 s (the SHA-256 check of 424 GB), a later one in 112 s. **Closer to the original than ISTA's quant:** on one 4,096-token sequence, ISTA's KL from UD-IQ3_S was 0.151 against UD-IQ3_S's own run-to-run 0.008, its top token differed at 14.8% of positions (3.5% run to run), and its perplexity was 4.97 against 4.56. That uses UD-IQ3_S as the reference, so it doesn't place ISTA on Maya's own chart (UD-IQ3_S 0.061, Q8 0.047 from the full-precision model; its scorer isn't public).
- **Drafter:** peasantsmith's `dspark-xs2` (Maya-XS's 2-bit DSpark drafter, 4.25 GB, from [peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)), mounted at `/dspark`. Maya's pack reads Maya's drafter format only, not unsloth's dspark GGUF. ISTA's GGUF carries no DSpark geometry, and Maya v1.1.27–v1.1.28's pack builder then writes 0, so the drafter would load and never draft. The launcher writes DeepSeek-V4.1-Flash's own values into the pack when it says 0 ([UPSTREAM.md](../UPSTREAM.md)). The draft block is 5, from that geometry; `maya-ds4` has no depth flag, so `SPEC_N` doesn't apply here either. The UD-IQ3_S slug runs **no drafter**: on two 3090s neither Maya's pinned `dspark.bin` (MXFP4 experts) nor `dspark-xs2` sped it up (20-token prompt, 256 tokens, warm pass, 2026-11-02: off 38.6, `dspark.bin` 39.5, `dspark-xs2` 38.5 tok/s, ~80% of drafts kept), and the drafter's experts cost 4–8 GB more RAM. Its pack is built without one and the launcher passes `--dspark off`.
- **Launcher** (`models/deepseek-v4.1-flash-1017/strata-maya/launcher/`: `ds4-entrypoint.sh`, `ds4_prepare.py`, `ds4_settings.py`). Maya's DeepSeek pack can live away from the model files (its `ds4.txt` names the folder it indexes), so nothing is linked: the launcher checks the files' SHA-256 against its own table once (ISTA's 7, which Maya's tables don't list, ~255 GB; UD-IQ3_S's 8, the SHA-256s Maya pins, ~424 GB: about 4 minutes on the reference rig; Maya-XS's 7 — its 5 GGUF and the 2 drafter files — ~282 GB, not timed here), builds the pack with `tools/ds4/pack_gguf.py` on the data volume (seconds: an index, no weights copied), seeds the expert profile from `data/expert-profile-ds4.bin`, writes `maya-ds4.json` and starts `serve/server.py`. The data volume is the GLM slugs' (`strata-maya-data`), a folder per quant (`ista-iq2-xxs/`, `ud-iq3-s/`, `maya-xs/`).
- **Settings:** the same as the GLM slugs' table above, with DeepSeek's defaults: effort `high` (Maya's own for DeepSeek; DeepSeek's template turns low / high / max into its 1–100 "Reasoning Effort" budget), temperature 1.0 / top_p 1.0, the name `deepseek-v4.1-flash`. `KV` doesn't apply (`maya-ds4`'s cache is fp16). One more: `THREADS`, the CPU lane's threads (`--cpu-threads`). Unset, the launcher uses half the CPUs the container may use (nproc / 2, at least 1): a safe default on a host we know nothing about. A rig that has measured its own sets `THREADS` (a shell prefix, or `bash scripts/settings.sh set THREADS=<n>`; it isn't a per-slug setting, so the saved value also reaches the llama.cpp CPU-offload slugs, which read the same variable). The reference rig runs 24 of 32 vCPUs, the best count in Strata's port of this engine, where 16 cost 4–5% of decode and the engine's own default (a thread a core, two left spare: ~30 on a 32-core host) ~40%; it wasn't re-measured on `maya-ds4`, which ran 24 throughout.
- **RAM:** with `--mmap-experts auto` Maya copies every CPU-side expert into RAM, or none: only when all of them fit with 8 GiB to spare beside 8 GiB for the rest (`ds4_model.cu`). ISTA's quant with its drafter needs ~100 GB of experts copied; UD-IQ3_S 218 GB, so ~237 GB must be available at start. Below that it reads them in place from the files, through the page cache. On the reference rig at 241 GB of RAM, ~6 GB short, UD-IQ3_S read a 4K prompt 7× slower than ISTA (8.3 vs 57.5 tok/s in Maya's logit-dump path, 2026-11-02); raising the VM to 257.6 GB made it copy. From v1.1.51 `DS4_RAM_MARGIN_GB` sets the 8 GiB for the rest (in GiB, 0 to 4096; every DeepSeek compose passes it through, and unset keeps 8). The copy test reads it, so a lower value lowers the RAM the copy needs by as much, and leaves that much less to the page cache (the Engram reads) and other programs.
- **Maya-XS is the one a 92 GB host can hold.** Built from its pack's tensor table: 121.09 GB has to be held — 109.57 GB of routed experts (40 layers × 384 experts × w1/w2/w3, IQ1_S 95.55 + IQ2_XXS 14.01), 7.27 GB of dense / attention / head, 4.25 GB of drafter — which is exactly Maya's manifest `memory_gb: 121` for the quant, and the 161.28 GB of Q6_K Engram tables it shares with Maya-S never enter RAM at all. Two 24 GB cards hold at most ~51.5 of the 121, and not all of that (KV, activations, the CUDA context), so the host carries 73-85 GB — inside a 92 GB machine, outside the two sibling slugs': ISTA's IQ2_XXS measured 106 GB of host RAM and UD-IQ3_S 229.
  - Its first boot (2026-11-09, v1.1.56, a 91.9 GiB rig, through `switch.sh`) split it that way: the two pools filled to 19.21 / 19.62 GB of experts (2,558 of 7,680 and 2,759 of 8,064, 582 / 584 MiB left free), the CPU lane kept the other 10,427 (74.5 GB), and the engine's own reading of the RAM free for them was 85.0 GB — inside the band, and above the 80 GB the host gate asks for. ⚠️ It left every one of them **in place in the files rather than pinned**: 85.0 GB free against the 74.1 GB the engine prices the copy with did not clear `--mmap-experts auto`'s margin (see RAM above), so this rig runs the page-cache mode. The two figures are the engine's own, not two readings of one thing: 74.1 GB is its read-ahead set — its line is "the most-routed 74.1 GB read ahead" — while the 74.5 GB above is the lane's 10,427 experts together. `DS4_RAM_MARGIN_GB` is the knob that decides which mode a host gets; what the mode costs this quant is unmeasured, since no pinned boot of it exists here to compare against.
  - **Quality** (2026-11-09, `quality-test.sh --full`, thinking ON at `REASONING_EFFORT=low`, the server's sampler 1.0 / 1.0, the wrapper's budgets): **130/150 pass@1, 137/150 pass@3**. The UD-IQ3_S slug's thinking leg totalled the same 130 and 137 at effort **high**, so this is not a controlled A/B — but Maya's smallest quant of the model holds its 423.6 GB one on the harness with a quarter of the host RAM. No `bench.sh` row and no thinking-off leg yet.
- **Prompt reading on the layer split:** from v1.1.53 every DeepSeek compose passes `DS4_FLOW_COPY_DMA` through (unset = off). With `=1`, a prompt's relay through the GPUs DMAs each streamed expert straight from its pinned RAM copy, where a reader otherwise copies it into the pinned staging first (`ds4_model.cu`; the same bytes, [mw00/project-maya#811](https://github.com/mw00/project-maya/pull/811)). It needs the experts copied into RAM (see RAM above) and does nothing under `MAYA_TP=2`. Upstream measured it on 4× 3090 over PCIe 3.0, only together with three other settings; it isn't measured here on two GPUs. Of those three, `DS4_AI_RUN` has been on by default since v1.1.55, so it needs no pass-through. It sizes the indexer's rows by the prompt's own keys rather than by the context (the same answers, fewer slabs). From the pinned source (`ds4_model.cu`, not measured): at a 65,536 context the context's rows are already 128, its cap, so it changes nothing there; at 262,144 they are 32, and a prompt with fewer keys gets more, up to 128. `DS4_CHUNK_AUTO` / `DS4_YOCO_FLOW` change the answers slightly, so neither is forwarded. Nor is `MAYA_DS4_Q8MV_TC` (tensor-core Q8_0 calls in DSpark's verify windows): in ISTA's quant only the two smallest attention matrices are Q8_0, and UD-IQ3_S runs no drafter.
- **More cards, and the expert mode by hand ([#2042](https://github.com/noonghunna/club-3090/issues/2042)):** every DeepSeek compose passes `MAYA_DS4_TP_GGUF_UNEVEN` and `DS4_TP_SHARES` through (unset = Maya's own). With `MAYA_TP` of 2 or more, `MAYA_DS4_TP_GGUF_UNEVEN=1` splits each GGUF expert's FFN in whole blocks as even as they go, so a group whose members can't take equal parts still forms (Maya-XS on four cards), and `DS4_TP_SHARES=w0,w1,..` sets each member's share by hand (`ds4_model.cu`). More cards also need `MAYA_GPUS` and the launcher's `--gpus`. `MMAP_EXPERTS` (`auto` / `on` / `off`) writes maya-ds4's `--mmap-experts`: `on` reads the CPU lane's experts in place, `off` copies them into RAM, `auto` (the default) copies them when they all fit, so a host picks the mode directly rather than through `DS4_RAM_MARGIN_GB` (whose large values also shrink the read-ahead). None of these is measured on the reference rig: on four RTX 5080 16 GB with 125 GB of RAM, Hazelhof's TP4 on Maya-XS prefilled 1,813 / 2,743 tok/s at 10K / 90K with the experts copied but failed the swap check, and 727 / 1,339 read in place, which passed it.
- **More engine settings, passed through unset (v1.1.65):** every DeepSeek compose also passes these to maya-ds4, each found in the pinned image; unset, Maya's own default applies:
  - `DS4_STREAM_TIE`: the PCIe expert lane on more than one GPU ([mw00/project-maya#956](https://github.com/mw00/project-maya/issues/956), v1.1.64). `keep` streams as before v1.1.64, `off` turns the lane off. Unset, a layer split turns the lane off unless it measures faster. On the reference rig's layer split (v1.1.65), the lane was timed at load and stayed on for 16 layers.
  - `DS4_FLOW_TP=1`: under `MAYA_TP` of 2 or more with `MAYA_DS4_TP_GGUF_UNEVEN`, the group's members read prompts on the flow too, each streaming its own rows. Off the flow, a box that streams experts from the disk reads prompts much slower.
  - `DS4_TP_UNEVEN_HEADS=1`: a tensor-parallel group of 3 on a GGUF pack keeps a width the 64 attention heads don't split into evenly (experimental upstream).
  - `MAYA_DS4_SLOT_KEEP=1` keeps a conversation's state when another one takes the model, so an agent's parent and workers each continue their own. `MAYA_DS4_KEEP_THINKING=1` keeps earlier turns' reasoning in the prompt, which a chat without tools needs beside it; a request with tools keeps it already. Maya's server reads both.
  - None of them is measured here.
- **Maya's `--calibrate` doesn't help here:** on two equal GPUs it tries only `--tp 2` and fewer CPU threads, never the layer split.

## MiniMax-M3.1-Flash (maya-mm3)

Maya runs MiniMaxAI's [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) (428B MoE, 23B active a token: 128 routed experts top-4 plus a shared one, 3 leading dense layers, a block-sparse attention indexer in every layer, 1M native context) on a third engine, `maya-mm3` (`src/mm3`, from v1.1.29; mw00/project-maya#477). The slugs serve Project Maya's own GGUF quant of it, from v1.1.30.

| Quant | Single (1 card) | Dual (2 cards) | Multi4 (4 cards) |
|---|---|---|---|
| **Maya-S** (`maya-s`: routed experts IQ2_XXS, the most sensitive IQ2_XS / IQ2_S / IQ3_XXS, ~2.1 bits; attention and indexers Q8_0, the rest Q6_K; 3 shards + the indexers side file, 122.5 GB) | `strata-maya/minimax31-flash-single-maya-s` · 8118 | `strata-maya/minimax31-flash-dual-maya-s` · 8059 | `strata-maya/minimax31-flash-multi4-maya-s` · 8209 |

- **Four cards** (`strata-maya/minimax31-flash-multi4-maya-s`; never booted on the reference rig: multi4 is community-validated by design): the same layer split over four cards (a GGUF pack has no other), DSpark on the last card. Its host RAM figure is the dual slug's 102 GB, as an upper bound: four cards hold more experts in VRAM, and the ~13 GB of pinned K / V at 204,800 is the same on any card count.
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
- **Context:** 204,800. Maya's own setup defaults to 65,536 and recommends more where the PC's memory holds it; the model reaches 1M, and Maya has run 256K end to end. Catalogued like the GLM table's `CONTEXT` row above: `switch.sh --set <slug> CONTEXT=...` saves it per slug.
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

## Hy4-Flash (maya-hy4)

Maya runs Tencent's [Hy4-Flash](https://huggingface.co/tencent/Hy4-flash) on a fifth engine, `maya-hy4` (`src/hy4`, from v1.1.68), using Project Maya's own GGUF quants. The model is a 220B MoE with 42 layers:
- layer 0 dense, the others with 512 routed experts (16 a token) plus a shared one;
- MLA attention, with a sliding window on most layers and 12 indexed sparse-attention layers;
- four Engram n-gram layers;
- an MTP block, vision, and 1M native context.

| Quant | Single (1 card) | Dual (2 cards) | Multi3 (3 cards) | Multi4 (4 cards) |
|---|---|---|---|---|
| **Maya-XS** (`maya-xs`, IQ2_XXS experts, Q6_K attention; model shard 44.8 GB) | `strata-maya/hy4-flash-single-maya-xs` · 8210 | `strata-maya/hy4-flash-dual-maya-xs` · 8214 | `…-multi3-maya-xs` · 8218 | `…-multi4-maya-xs` · 8222 |
| **Maya-S** (`maya-s`, IQ2_XXS experts on layers 1–23, IQ3_XXS on 24–41; 53.0 GB) | `strata-maya/hy4-flash-single-maya-s` · 8211 | `strata-maya/hy4-flash-dual-maya-s` · 8215 | `…-multi3-maya-s` · 8219 | `…-multi4-maya-s` · 8223 |
| **Maya-M** (`maya-m`, IQ3_XXS experts, IQ4_XS on 7 layers, Q8_0 attention; 68.4 GB) | `strata-maya/hy4-flash-single-maya-m` · 8212 | `strata-maya/hy4-flash-dual-maya-m` · 8216 | `…-multi3-maya-m` · 8220 | `…-multi4-maya-m` · 8224 |
| **Maya-L** (`maya-l`, IQ4_XS experts, Q5_K on 5 layers, Q8_0 attention; 89.5 GB) | `strata-maya/hy4-flash-single-maya-l` · 8213 | `strata-maya/hy4-flash-dual-maya-l` · 8217 | `…-multi3-maya-l` · 8221 | `…-multi4-maya-l` · 8225 |

- **⚠️ License:** the Tencent Hy License (`LICENSE` in the weights' repo, `LICENSE.hy4-flash` in Maya's).
- **Weights:** [peasantsmith/Hy4-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF) @`9d73294d`. Each size's own file is its model shard (00001). Every size's folder also holds the same three shared files:
  - shard 00002, the n-gram table (63.4 GB);
  - the MTP block (`mtp/Hy4-Flash-mtp-BF16.gguf_file`, 7.5 GB);
  - the vision tower (`Maya-XS/mmproj-Hy4-Flash-Maya-XS.gguf`, 0.8 GB).

  Keep one copy of each. The catalog names the table by its `Maya-XS/` copy, and every compose mounts the shared files under its own size's names. All four sizes with the shared files take 327 GB.
- **Not downloaded:** the repo also has a DFly2 drafter for each size (0.7–0.9 GB). `maya-hy4` doesn't read it yet, and Maya's own setup doesn't fetch it either.
- **Launcher** (`models/hy4-flash/strata-maya/launcher/`: `hy4-entrypoint.sh`, `hy4_prepare.py`, `hy4_settings.py`). The launcher:
  - checks the four files' SHA-256 against Maya's manifest, once;
  - builds the pack with `tools/hy4/pack_gguf.py` on the data volume (`/data/hy4-maya-<size>/pack`, an index: no weights are copied);
  - writes `maya-hy4.json` afresh at every boot and starts the server.

  With the files in the page cache, the first start was ready in 60–80 s and later ones in 16 s.
- **Placement:** everything that isn't a routed expert sits in VRAM with the cache. The card fills the rest with the most-routed experts, and a CPU lane computes the others from the memory-mapped model file.
- **Tensor parallel:** on several cards `maya-hy4` (`--tp auto`) runs tensor parallel where one member fits on every card, and splits the layers where it doesn't. On 2 × 3090 it never fits: a member needs 24.3 / 28.5 / 36.3 / 47.1 GB for XS / S / M / L. So the dual slugs run the layer split.
- **Drafter:** the MTP block, **on**, including on one card (GLM's `maya` drafts only on two or more). `DRAFTING=off` turns it off.
- **K / V:** `KV=auto` (F16 latent rows, 8-bit where F16 doesn't fit), or `f16` / `q8`. They live in VRAM only: this engine has no RAM path, so a longer context leaves fewer experts in VRAM.
- **The n-gram table** is read from the disk a row at a time while the model answers, so Maya says to keep it on an SSD. `HY4_EG_POPULATE=1` maps it into the page cache at start instead.
- **Settings:** as for the other Maya slugs, with Hy4-Flash's:
  - **Effort:** `high` by default. Hy4-Flash thinks at high or not at all, so `low` / `medium` / `xhigh` / `max` run at high (the boot log says so). `none`, or `ENABLE_THINKING=false`, turns thinking off.
  - **Sampling and name:** temperature 0.9 / top_p 1.0; the name is `hy4-flash`.
  - **More:** `KV`, `DRAFTING` (`on` / `off`), `VISION` (`off` gives the tower's room to the model), `THREADS` (`--cpu-threads`, half the CPUs by default), `HY4_EG_POPULATE` (`0` / `1`).
- **Context:** 32,768. Maya has run its quants end to end up to 32K; the model reaches 1M. `CONTEXT` sets more.
- **One request at a time:** `maya-hy4` has no `--parallel` yet.
- **Checked on the reference rig** (2026-11-20, v1.1.68 image `3815b467`, 230 W): every single and dual slug, each on its own boot, booted through `switch.sh` and passed `verify-full` (9 passed, 0 failed, vision 4/4).
  - Not benched yet.
  - Quality per Maya's model card (same top token / KL against full precision): XS 88.5% / 0.16, S 91.8% / 0.072, M 93.6% / 0.046, L 96.6% / 0.012.
  - multi3 and multi4 are community-validated by design.

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
- **It prints** each split and thread count it tried, then the settings it changed, e.g. `tuned for this PC: MAYA_GLM_PCIE_SHARE=0.10 (32.3 tok/s)`, or that the engine's own settings are already the fastest here.
- **The `down`** removes the network the one-off run created; without it, each tuned quant leaves one behind.
- **Save what it printed as settings**, because the launcher writes Maya's config afresh at every boot and drops what the tuning wrote there. `MAYA_GLM_PCIE_SHARE` becomes `MAYA_PCIE_SHARE` and `MAYA_GLM_CPU_LANE` becomes `MAYA_CPU_LANE`: `bash scripts/settings.sh set MAYA_PCIE_SHARE=0.10`. Every Maya slug reads a saved value, whatever its quant or card count, so save the result of the slug you run. `switch.sh --set` doesn't accept these two settings yet.

## Known gaps

- **Quality:** not validated here (no 8-pack yet); the KL figures above are upstream's.
- **Reasoning:** GLM-5.3's template thinks by default. Maya's server closes the thinking block when a request sends `enable_thinking: false` or `reasoning_effort: none`, unlike the llama.cpp GLM slugs, where thinking can't be turned off. The server default for requests that name none is effort `max`, as on the other GLM-5.3-Flash slugs (Maya's own setup writes `high`), with Maya's own 32K reasoning budget. `REASONING_EFFORT`, `ENABLE_THINKING` and `THINKING_BUDGET` change these server-side (above). `thinking-budget.sh` (per-effort budgets) doesn't apply: Maya has one budget for every effort.
- **GLM's vision and DFlash2:** not in the image (Hy4-Flash's vision is). Maya's DFlash2 drafter (+12% on two cards, +19% on one 3090, by upstream's measurements) has CC BY-NC-ND weights.
- **Platforms:** x86-64 only. DGX Spark (ARM) can't run it. V100s need a native build (CUDA 12).
