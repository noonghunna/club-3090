# Strata — an engine written for Qwen3.8-Flash-Next on one or two cards

**Engine id:** `strata` · **Slug prefix:** `strata/` · **Status of its slugs:** 🐣 Incubating · **Requires:** NVIDIA compute capability 7.5+, driver ≥ 580 (CUDA 13), an x86-64 CPU with AVX2, ~86 GB of host RAM for IQ3_S when the model disk is rotational (setup keeps the 28.8 GB n-gram table in RAM; less on an SSD)

## What it is

[Strata](https://github.com/Niko1221/Strata) (MIT, Niko1221) is its own C++/CUDA engine, written for one model, plus a Python server and a Terminal (`strata launch`) that set it up and serve it. It is not a llama.cpp or vLLM fork. It uses three pieces of llama.cpp as libraries:
- ggml's CPU backend, which computes the expert rows that run on the CPU;
- mtmd, for its image encoder;
- gguf-py, to read the GGUF files.

The scheduler, the expert cache, the CUDA kernels, the MTP draft layer and the server are its own.

Every expert is page-locked in host RAM, and the GPU keeps a cache of the hot ones, filled from a routing profile and swapped as it runs. On two cards it splits the **layers** (`--gpus 0,1`): each card caches the experts of its own layers, with one hand-off per verify window. That needs no NCCL and no P2P. It serves an OpenAI API (chat, completions, `/v1/responses`) and an Anthropic one (`/v1/messages`) on port 8080.

Its profile declares `type: strata` and `lineage.relation: new`, and the tooling knows it as engine kind `strata` (`scripts/lib/engine-kind.sh`).

## Image

From 0.2.1 Strata publishes its own Docker images, built by its `docker-images` workflow from the signed engine packages of each release: `ghcr.io/niko1221/strata:<release>-<flavor>`. The slugs pin the digest of `<release>-cuda13` (RTX 20–50 and the A-series, NVIDIA driver 580 or newer), and the launchers inject it as `STRATA_IMAGE`. Export `STRATA_IMAGE` in the shell to try another image for one launch; a saved value loses to the pin.

- **Pinned:** 0.2.3 (engine 0.1.48), since 2026-10-23 (0.2.2, engine 0.1.47, from 2026-10-22).
- **Before 0.2.1** upstream published no Linux image, and club-3090 built `ghcr.io/noonghunna/strata-club3090` from upstream's Dockerfile, with ggml's CPU code for an AVX2 baseline (the last one, v0.1.45, until 2026-10-20). Upstream's image now builds that way itself.
- x86-64 with AVX2 only (no DGX Spark).

**Bumping:** pin the digest of the new release's `<release>-cuda13` image. What a bump checks, and when to skip a release: [the bump runbook](../NIGHTLY_BUMP_RUNBOOK.md#strata-and-project-maya-the-bump-rule).

## Slugs

| Slug | Cards | Port | Notes |
|---|---|---|---|
| `strata/qwen3.8-flash-next-iq3s-dual` | 2, layer split | 8018 | the fastest offloaded Flash-Next measured on the reference rig |
| `strata/qwen3.8-flash-next-iq3s-single` | 1 | 8019 | the same model, on one card (its own topology folder) |
| `strata/qwen3.8-flash-next-swift15-abliterated-iq3s-dual` | 2, layer split | 8024 | SC117's abliterated Swift 1.5 |
| `strata/qwen3.8-flash-next-swift15-abliterated-iq3s-single` | 1 | 8025 | the same model, on one card (its own topology folder) |
| `strata/qwen3.8-flash-next-swift15-abliterated-iq3s-multi4` | 4, layer split | 8046 | the same model on cards 0–3; never booted here (two cards): community-validated |

**Weights:**
- The first two run ISTA-DASLab's GSQ-RCO IQ3_S of the original model (`ista-iq3s`, 83.6 GB in two shards).
- The `swift15-abliterated` pair runs SC117's abliterated GSQ-RCO IQ3_S of UkisAI's Swift 1.5 (`sc117-swift15-abliterated-iq3s`, 83.55 GB in two shards, pinned to revision `2b89cf3f`). Swift 1.5 is a fine-tune of Qwen3.8-Flash-Next that thinks much shorter. SC117 replaced its 144 residual-writer tensors with ones from orcarouter's abliterated release, which removes the refusal direction. Its README says it has no reliable built-in guardrails, so put your own moderation in front of anything user-facing. It is under the Swift Open License 1.0 (commercial use is free only below US$1M annual revenue) as well as the Qwen licence. Strata serves it as its `swift` model family.

**Data volume:** `strata-data-v2` under the data directory (`~/.local/share/club-3090` by default; `STRATA_DATA_DIR` overrides), in Strata 0.2's layout; 0.1.x's `strata-data` isn't read by it. It has:
- one folder per topology (`dual`, `single`, `multi4`): the run configs setup writes, the presets and Strata's state. Setup writes the cards into a model's run config, so a dual and a single start sharing one would set the model up again each time;
- `Strata-data`, shared by every slug: one prepared pack per model family (`iq3_s` for ISTA, `swift-iq3_s` for Swift) and the MTP draft layer.

**First start:** the first start of a model in a topology runs Strata's setup. It checks the shards, prepares that family's pack and, unless the volume already has it, range-reads about 5 GB of MTP tensors from `Qwen/Qwen3.8-Flash-Next`, so it needs the network. Swift 1.5 ships no draft layer of its own, so its slugs use the same one. While setup runs, the server answers on a port the compose doesn't publish, so nothing outside talks to it before the launch settings are in. On a fresh volume it took 511 seconds on the reference rig with the image already pulled. Give the first launch `READY_TIMEOUT=1800`. A model already prepared for another topology starts in about 45 seconds, later starts in about 25.

**API key:** Strata 0.2+ answers nothing but `GET /health` without a key, so every slug has one: club-3090's default, `club-3090-strata-local-key`, unless you set `STRATA_API_KEY` (`bash scripts/settings.sh set STRATA_API_KEY=…`; 16 characters or more). Clients send `Authorization: Bearer <key>`. Our scripts read the key from the running container and send it themselves (`scripts/lib/engine-auth.sh`); for a remote URL set `ENGINE_API_KEY` — the gateway's sync never uses that override: its probe only ever asks local ports, where the container decides. The gateway's sync works the same way: its probe starts keyless, and the 401 makes it resolve the container's key through that same rule and retry — which is how a running slug gets a gateway route at all (#1794). The rendered route references `os.environ/STRATA_API_KEY`; only the public default is written into the runtime file as a literal, and if the container runs a key your settings never saved, the sync says so and tells you to save it; if the saved key differs from the container's, the sync renders no route for that engine rather than send the gateway a key its server does not use. A gateway recreated while no Strata slug is running drops `STRATA_API_KEY` from its environment — the next Strata boot's sync says so, and the same `bash scripts/gpu-mode.sh gateway` recreate puts it back. The default is public, so set your own before you publish the port beyond this machine.

**Requests and readiness:** Strata picks how many requests run at once (`--parallel 3` on the reference rig, on one card and on two). `switch.sh`'s readiness probe sends one completion, so its ready means the engine generates.

Measurements, with dates and settings, are in each compose's header and in [BENCHMARKS.md](../../BENCHMARKS.md) (Qwen3.8-Flash-Next, the Strata table).

## What club-3090 adds

- **One launcher for every compose** (`models/qwen3.8-flash-next/strata/launcher/strata-entrypoint.sh`), run under the image's `tini`. It hands the image's own entrypoint our model folder (read-only, `MODEL_PATH=/gguf`), the model family, the cards `STRATA_START_GPUS` names, `CONTEXT`, `KV` and the API key, and writes the launch settings below into setup's run config. The compose passes the family as the launcher's one argument (`qwen` when absent, or `swift`), not as a setting, so no saved value can point a slug at the other model's files. Pictures are off (`VISION=off`), as before.
- **Runtime numbers from `/metrics`.** Strata writes its per-request lines (throughput, drafts accepted, expert-cache hits) to a log file inside the container, not to `docker logs`. `health.sh`, `verify-full` (its MTP acceptance check), `bench.sh` and `report.sh` read them from `GET /metrics` through `scripts/lib/strata_metrics.py` instead.
- **Launch settings, written into Strata's own config at every boot** (`launcher/strata_config.py`). Strata reads these from the run config setup writes (`/data/install/strata-<family>.json`), and setup writes none of them. Its launcher rebuilds the live config from that file at every start, so that file is where they go, before the server starts. Each apply first takes out what the last one wrote, so a setting you remove is gone at the next boot. Strata 0.2's setup keeps keys it doesn't write itself and folds an `--spec` of ours into its own record. When setup runs during a start anyway (after `CONTEXT` or `KV` changed), the launcher checks what the server runs from once it answers, and restarts Strata once if a setting didn't survive. A bad value stops the boot with the reason. `REASONING_EFFORT`, `ENABLE_THINKING` and `SPEC_N` are catalogued, so `bash scripts/switch.sh --set <slug> REASONING_EFFORT=low` saves them for one slug; `--set` refuses the others, so save those with `bash scripts/settings.sh set TEMP=0.7` (they apply to every slug that reads them) or give them as a shell prefix.

  | Setting | Strata config | Values | Unset |
  |---|---|---|---|
  | `REASONING_EFFORT` | `reasoning_effort` | `none`, `low`, `medium`, `high` (`xhigh` and `max` mean `high`, the template's xhigh) | the template's default, xhigh |
  | `ENABLE_THINKING` | `reasoning_effort: none` when `false` | `true`, `false`; `false` wins over `REASONING_EFFORT` | thinking on |
  | `THINKING_BUDGET` | `reasoning_budget_tokens` | tokens, `0` = no budget; one cap for every request (Strata has no per-effort budgets) | no cap |
  | `SPEC_N` | `--spec N` in the engine args | `1`–`8`: the MTP drafter's verify window; the drafter can't be turned off from here | `4` (setup's) |
  | `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `PRESENCE_PENALTY`, `REPEAT_PENALTY` | `sampling.temperature`, `.top_p`, `.top_k`, `.min_p`, `.presence_penalty`, `.repetition_penalty` | ≥ 0 / 0–1 / 1–64 / 0–1 / ≥ 0 / > 0 | the model card's (see Known gaps) |
  | `SERVED_NAME` | `aliases: [name]` | one name, no spaces: another name the model answers to. `/v1/models` lists setup's own name first and this one after it, and an answer to a request that names it carries it (Strata answers a request whatever model it names, so this changes what it lists and reports) | setup's name only (`qwen3.8-flash-next-iq3_s`, `swift-1.5-iq3_s`) |
  | `CONTEXT`, `KV` | setup's `--max-context` and `--kv` | a context ≥ 4096; `fp16`, `int8`, `q4_0`, `k8v4` | 131072 / setup's choice (int8 here) |

  `CONTEXT` and `KV` size the KV cache and the expert cache, so setup owns them: a value that differs from what setup was given runs setup again at the next boot (it says why). A request's own fields always win over these defaults, and so do the Chat settings saved from Strata's web page: they fill effort and sampling before the config does. `MAX_TOKENS` doesn't apply: Strata takes a default output cap only from those saved Chat settings (`max_tokens` in its run config is the vision encoder's), so a request with no `max_tokens` runs to the end of the context. Until 2026-10-17 the launcher wrote it anyway, to no effect.
- **The club conventions:** the `CLUB3090_RESTART` knob, `BIND_HOST` / `ESTATE_PORT` / `ESTATE_CONTAINER` / `ESTATE_GPUS`, and the host GID plus `umask 0002` on the data volume.

## Known gaps

- **Quality:** not validated on these slugs (no 8-pack yet). For the abliterated Swift slugs, SC117 published measurements for its IQ2_XS tier only: decode about 2% slower than upstream Swift, and MTP draft acceptance 62.5% against 66.8% with the base model's draft layer.
- **Reasoning and sampling defaults:** with no settings, a request that names no effort runs at the template's default, `xhigh` (`high` and `max` also map to `xhigh`). That is the checkpoint's own chat-template default (`reasoning_effort|default('xhigh')`), and the `bucko-vllm` slugs keep it too: they also run at xhigh, with a 16,384-token default thinking budget on top. Only the Qwen3.8-27B vLLM and SGLang composes default to `low`. `REASONING_EFFORT` / `ENABLE_THINKING` change that server-side (above). Since v0.1.45, a sampler field a request leaves out comes from the model card unless the `TEMP` … `REPEAT_PENALTY` settings fill it: thinking requests get temperature 1.0, top_p 0.95, top_k 20; thinking-off requests get 0.7, 0.8, 20 and presence_penalty 1.5. Before v0.1.45 a request with no sampler decoded greedy, and only `temperature: 0` does now. A field the request sends always wins; for the 8-pack, [RUN_EVALS.md](../RUN_EVALS.md) has the commands. `THINKING_BUDGET` sets one thinking cap for every request; `thinking-budget.sh` (per-effort budgets) still refuses Strata.
- **`spec-sweep.sh`** still refuses Strata slugs, although `SPEC_N` now applies: a sweep would need it to reboot per depth through the setting, which it doesn't do for Strata yet.
- **Prompt-cache share:** `cache-share.sh` lists Strata but doesn't measure it. Strata's prefix-cache counters are only in its Prometheus output (`GET /metrics` with `Accept: text/plain`).
- **Platforms:** x86-64 only. Strata's CPU kernels are x86 SIMD, so DGX Spark (ARM) can't run it. A CPU without AVX2 makes setup compile Strata's older-CPU engine at the first start; that's untested.
