# QEngine — bpx03's engines for Qwen3.8-27B and Qwen3.8-Flash-Next INT4 on RTX 3090s

**Engine id:** `qengine` · **Slug prefix:** `qengine/` · **Status:** two two-card slugs, 🐣 Incubating · **Requires:** NVIDIA RTX 3090 (sm_86 only), ideally three; a driver with CUDA 13.1 support

> ⚠️ **License: PolyForm Noncommercial 1.0.0.** QEngine is free for non-commercial use only; commercial use needs a licence from its author. Check this before you use it.

## What it is

[QEngine](https://github.com/bpx03/QEngine) (Krzysztof Jagus) is an independent C++/CUDA inference engine (`libqengine.so`, about 26k lines) with its own Python server. It is built for one model on one kind of machine: **Qwen3.8-27B INT4** (RedHatAI's compressed-tensors W4A16 checkpoint and its MTP head) on **three RTX 3090s** with tensor parallel 3 over PCIe. It isn't a fork of any engine we ship, so its lineage class is `new` ([ADDING_ENGINES.md](../ADDING_ENGINES.md)).

What it does, from its README:
- **INT4 weights**, repacked at load into tensor-core layouts, with an INT4 GEMV for decode and an INT4 GEMM for prefill.
- **Tensor parallel over PCIe**, with the all-reduces inside the kernels. It uses GPU-to-GPU P2P when the driver grants it, and otherwise a slower path through host memory.
- **MTP speculative decoding**, with 1–7 drafts a step chosen from the measured acceptance. The author also publishes an MTP head fine-tuned on the engine's own answers (`model_mtp_ft.safetensors`, a release asset). It adds **lookup drafts** from the token history.
- **FP8 (e4m3) K / V** in a shared pool per GPU, up to 262,144 tokens a conversation, with prefix and KV reuse and a host-RAM archive of evicted conversations.
- **Continuous batching** of up to 32 sequences, structured output (llguidance), tool calls, and thinking on or off with reasoning-effort levels.
- **OpenAI- and Anthropic-compatible APIs** (`/v1/chat/completions`, `/v1/models`, `/v1/messages`) on port 8080, `/health`, and a dashboard with Prometheus metrics on port 9393.

**Cards:** three are the tuned configuration. The author has measured two (72–85% of the three-card throughput, full 262K context, `QE_MAX_SEQS=32`). One card is experimental: it starts only with `QE_MAX_SEQS` 1–4, for about 85K–112K tokens of context.

**Models:** Qwen3.8-27B, RedHatAI's INT4 only. It reads `weight_packed` / `weight_scale` (compressed-tensors `pack-quantized`), so other INT4 checkpoints don't load, including our AutoRound one (`auto_gptq` packing: `qweight` / `qzeros` / `scales`). That one has the same scheme (int4, symmetric, group 128, the same layers) but a different file layout and different rounding. Its 1.3 release adds a second engine for **Qwen3.8-Flash-Next** (`libqflash.so`), which reads only the author's own INT4 layout ([bpx03/Qwen3.8-Flash-Next-QEngine](https://huggingface.co/bpx03/Qwen3.8-Flash-Next-QEngine), converted from Intel's AutoRound checkpoint). Upstream calls it a preview: tested on one machine, no quality benchmark yet. 1.3-rc4 fixes that engine's INT8 MoE prefill kernel, which now and then gave a few tokens wrong activations, mostly on two GPUs.

## Image

The author publishes `ghcr.io/bpx03/qengine` in two forms per release: `:<release>`, text only (about 0.3 GB), and `:<release>-vision`, which adds PyTorch for the vision encoder (about 3 GB to pull). Each image's `org.opencontainers.image.revision` label names its source commit. The profile pins **`1.3.2-vision`** by digest (`sha256:3375a425…`, source `fb1da72d` = git tag `v1.3.2`), and the launchers inject it as `QENGINE_IMAGE`. Its server, engine libraries and entrypoint are byte-identical to the text image's, so text serving is the same in either; the vision image is what lets the slugs take images.

**Vision (`VISION`, default 1):** both slugs accept images (OpenAI `image_url`, data URLs or http(s); Anthropic `image` blocks) through QEngine's `--vision 1`. Both weights folders carry the checkpoint's vision tower. The encoder runs in PyTorch on the last card, with its weights pinned in host RAM. Measured on the reference rig (2026-10-31, verify-full vision 4/4 on both): on the 27B, the encoder and PyTorch's CUDA context on GPU1 shrink the KV pool from 374,784 to 289,792 tokens (-23%), because tensor parallel sizes it by the smaller card; it still holds a full 262K context, and `VISION=0` gives the pool back. On Flash-Next, GPU1's expert budget goes from 15.73 to 15.05 GiB (1,485 fewer resident expert groups, ~2.4% of both cards'), the KV pool is unchanged, and MemAvailable at ready fell 88.2 GiB against 85.5 on the text image. Upstream measured text output unchanged by the feature and agent throughput the same (536 vs 537 tok/s, three cards); it has no vision quality benchmark yet, and video isn't supported. `VISION=0` turns it off; the compose stops with the reason if `VISION=1` meets an image without PyTorch.

## Settings

Its own environment variables (the image's entrypoint passes `QE_EXTRA_ARGS` to `server/app.py`):

| Variable | Default | What |
|---|---|---|
| `QE_GPUS` | `0,1,2` | the GPUs, as the container sees them |
| `QE_MAX_SEQS` | 16 | sequence slots, up to 32 |
| `QE_RESERVE_GB` | 1.5 | VRAM left outside the KV pool, per card |
| `QE_KV_ARCHIVE_GB` | 32 | the host-RAM archive of evicted conversations |
| `QE_P2P` | auto | 0 forces the host-memory path |
| `QE_PRESENCE_PENALTY`, `QE_FREQUENCY_PENALTY` | 0 | the penalties of sampled requests that send none (from 1.3-rc4) |
| `QE_LOOP_GUARD_TURNS` | 3 | the cross-turn loop guard (from 1.3-rc4): after this many identical (reply, tool result) turns in a row, the next turn gets a presence penalty on the repeated reply (`QE_LOOP_GUARD_PENALTY`, 1.5) and no lookup drafts; 0 turns it off |
| `QE_LOOP_GUARD_NOTE` | 1 | on a guarded turn, also append a short server note to the latest tool result; 0 drops the note |
| `QE_MTP_FILE` | auto | the MTP head (`model_mtp_ft.safetensors` in the weights folder, else the server's downloaded copy, from 1.3.2) |
| `QE_TOKEN_RANK_FILE` | auto | the MTP draft vocabulary (`token_rank.bin`, found the same way; from 1.3.2) |
| `QE_FETCH_ASSETS` | 1 | 27B: download the release assets missing from the weights folder at start, into `~/.cache/qengine/assets`; 0 only warns (from 1.3.2) |
| `QE_FETCH_WAIT_S` | 120 | how long a start waits for that download; slower, it starts without them and the next start uses them |
| `QE_MEM_MARGIN_GB` | max(8 GB, 5% of RAM) | the host RAM the archive always leaves free; the archive gives memory back before the OOM killer would end the server (from 1.3) |
| `QE_ALLOW_SHARED_GPU` | 0 | 1 skips the one-engine-per-GPU check (from 1.3). The lock is `~/.cache/qengine/locks/gpu-<uuid>.lock` under the container's own home, so containers only see each other's when they share it |
| `QE_METRICS_PORT` | 9393 | the port of its dashboard, `/metrics` (Prometheus) and `/api/stats` (its `qetop`), bound to 0.0.0.0 inside the container; 0 turns them off. On our slugs it also publishes them on the host, on 127.0.0.1 only (below) |
| `QE_LOG_FILE` | auto | the server's own log, `~/.cache/qengine/logs/<served-name>.log` (64 MB × 4); 0 turns it off (from 1.3) |
| `QE_EXTRA_ARGS` | | `server/app.py` flags, e.g. `--think 0`, `--max-ctx 131072`, `--served-name` |

⚠️ The loop guard is on by default, and it writes into the conversation: on a guarded turn, an agent finds the server's note in its latest tool result. A request with `"loop_guard": false` turns it off for that request.

## Slugs

| Slug | Model and weights | Port | |
|---|---|---|---|
| `qengine/qwen38-27b-dual` | Qwen3.8-27B, `RedHatAI/Qwen3.8-27B-INT4` (`redhat-int4`) | 8197 | the README's two-card setting: `QE_MAX_SEQS=32`, 262K context, MTP up to 7 drafts a step |
| `qengine/qwen38-flash-next-dual` | Qwen3.8-Flash-Next, `bpx03/Qwen3.8-Flash-Next-QEngine` (`qengine-int4`, 128 GB) | 8198 | 16 slots, 262K context, MTP up to 3 drafts a step; about half the expert groups in pinned host RAM, ~130 GB of host RAM at the defaults (upstream's figure) |

Both are 🐣 Incubating. The 27B slug was booted and benched on 2026-10-28 on two 3090s at 230 W: 147.2 / 216.3 tok/s decode (narrative / code), 2,430 prefill @10K and 1,786 @90K, MTP acceptance length 3.4, 23.4 GB a card, verify-full 8 + 2 n/a ([BENCHMARKS.md](../../BENCHMARKS.md)). The Flash-Next slug was booted and benched the same day: 164.0 / 170.6 tok/s decode, 1,921 prefill @10K and 2,373 @90K, MTP acceptance length 2.8, 24.1 GB a card, host RAM 88 GB at ready (up to ~114 GB as the archive fills), verify-full 8 + 2 n/a.

Community PR #1836 (eddietheengineer) validated both slugs on another two-card rig and found two Flash-Next problems there: a NaN on a ~76K-token prefill with `FLASH_COMM=auto`, and an out-of-memory in the 8-pack's agent pack at 8 slots. Its fixes are `FLASH_COMM=ce` and `FLASH_MAX_SEQS=4`. Neither reproduced on the reference rig: 90K and 107K needles on `auto`, which picked the P2P path here, and 8 concurrent multi-turn conversations to 27.7K tokens at the default 16 slots. So the defaults stay, and the compose header names both settings. With both, an 8-pack run there later aborted on a CUDA-graph out-of-memory (#1846): QEngine logs `graph instantiate out of memory, other graphs dropped` first, then a failed allocation (`flash_seq.cu:298`) ends the engine. `FLASH_EC_MARGIN_MB=1024`, the VRAM a card keeps free beside the expert cache (256 MB by default), fixed it there, for about 10% of decode and 5–7% of prefill. The first line also appears on the reference rig, in 3 of 6 boots and at the default 16 slots too, never followed by a crash so far. The default stays QEngine's: the abort is the engine's to fix. QEngine 1.3.1 drops its cached graphs and retries instead of aborting, and 1.3.2 fixes the root cause (the sampling kernel of batched requests needed more GPU stack than a CUDA context reserves); neither has been re-run on that rig.

On the 27B, the engine is faster with its two release assets (`token_rank.bin`, `model_mtp_ft.safetensors`, unchanged since v1.0; upstream: ~8-9% more accepted drafts). `setup.sh` doesn't fetch them, and the compose header and the weights entry give the `gh release download` line for the weights folder. From 1.3.2 the server downloads the missing ones itself at start (0.85 GB, checked against the release's `SHA256SUMS`) into the engine cache, which is kept per image, so each new pin downloads them again; in the weights folder they're found first.

The composes give the container only the two chosen cards (`device_ids`, so `launch.sh --gpus` picks them) and set `QE_GPUS=0,1`, the cards as the container numbers them. QEngine's own entrypoint overrides `CUDA_VISIBLE_DEVICES` with `QE_GPUS` (default `0,1,2`), so a forwarded mask would not reach it. The engine cache (`~/.cache/qengine`: the GEMM tune table, stats, logs and the one-engine-per-GPU locks) is mounted from the engine cache directory, so the GEMMs are tuned once per image.

## Thinking and effort

Our knobs reach QEngine through its server flags (in `QE_EXTRA_ARGS`) or its own variables:

| Knob | QEngine | |
|---|---|---|
| `ENABLE_THINKING` | `--think 1` / `0` | the default for requests that set none; `true` unless set |
| `SPEC_N` | `--mtp-k <n>` | the most MTP drafts a step (the engine adapts below it); 0 turns MTP off, lookup drafts stay on |
| `REASONING_EFFORT` | `QE_REASONING_EFFORT` | `low` unless set; `low`, `medium` or `xhigh` |

QEngine serves its own copy of the Qwen3.8 chat template (`server/templates/qwen38_syssafe.jinja`), whose default effort is `xhigh`. From 1.3.1 its server takes a default for requests that send none, `QE_REASONING_EFFORT` (or `--reasoning-effort`; #1842), on chat completions, messages and count_tokens alike, and the compose sets it from `REASONING_EFFORT`. It refuses to start on an image whose `server/app.py` doesn't read that variable, so an older image can't quietly go back to `xhigh`. Until 2026-10-30 the compose copied the template at boot with its one `reasoning_effort|default('xhigh')` replaced, and served the copy with `--template`. A request's own effort still goes to QEngine's template, which maps `high` to `xhigh` and anything else outside the three, `minimal` and `max` included, to `medium` (our patched template on the vLLM and SGLang slugs maps `minimal` to `low` and `max` to `xhigh`).

QEngine has no thinking budget: no flag, variable or request field caps the thinking, so `THINKING_BUDGET_*` and `THINKING_BUDGETS` don't apply to these slugs (`knob-conformance.json`).

## What's wired

- the engine profile (`scripts/lib/profiles/engines/qengine.yml`);
- engine kind `qengine` in `scripts/lib/engine-kind.sh`, from its id, its profile `type`, a `qengine-` container name, its image and `/v1/models` `owned_by: "qengine"` (it sends no `system_fingerprint`);
- container discovery (`qengine-` in the prefix set; port 8080 counts with that evidence);
- the `QENGINE_IMAGE` arm in both launchers;
- the label in `health.sh`, the Genesis skip in `verify-full.sh`, and the kind in `report.sh`, `run_context.py` and the local registration types;
- the two slugs, their weights variants and the launch knobs above.

QEngine prints `[stats]` lines for log parsers, in llama.cpp's timing format and vLLM's `SpecDecoding metrics` format, so `bench.sh`'s engine-side timings and acceptance capture and `verify-full.sh`'s MTP acceptance check read it with no QEngine-specific code (checked on the 27B slug, 2026-10-28).

QEngine reports neither its slot count nor its context over HTTP, so `served-slots.sh` (the concurrency step), `verify-stress.sh` and `concurrency_probe.py` read them from its boot log, when the container can be read (#1850). A host-run QEngine (`CONTAINER=none`) is recognised by `/v1/models`' `owned_by`. `verify-stress.sh` doesn't apply its absolute VRAM-margin check to QEngine, as it doesn't to Strata: QEngine allocates its KV pool (and, on Flash-Next, its expert cache) at load, so free VRAM is set at start and doesn't move with the context (763 → 763 MB across a 120K ladder on the 27B, 3 → 3 MB across 240K on Flash-Next, 2026-10-29). It prints the ladder's drop instead (#1826).

`health.sh` has a QEngine section: slots and context from the boot log, the latest `[chat]` requests, decode rates, MTP + lookup acceptance and, on Flash-Next, the expert cache's hit rate. `bench.sh`'s interconnect block reads QEngine's own transport lines (it links no NCCL): the 27B's `P2P: decode all-reduce GPU to GPU` or its host-memory path, Flash-Next's `no peer access` line, and both slugs' `[comm] prefill transfers`. c3 classifies a `qengine-` container as QEngine, not as llama.cpp from its 8080 port, and matches a rebench REPORT.md's container line to it.

The composes don't publish QEngine's metrics port: the dashboard has no auth and a write endpoint (`POST /api/reset`). `QE_METRICS_PORT=<port>` opts in. The compose passes it to QEngine, which then listens on that port, and `switch.sh` (so `launch.sh` and c3 too) adds an override at launch that publishes it on the host's loopback only, `127.0.0.1:<port>`. Compose can't publish a port conditionally, hence the override (`scripts/lib/qengine_metrics.py`); a compose you run with `docker compose` yourself gets no mapping. Set it for one launch (`QE_METRICS_PORT=9393 bash scripts/switch.sh --force <slug>`) or for every QEngine slug (`bash scripts/settings.sh set QE_METRICS_PORT=9393`); `switch.sh` refuses a value that isn't a port before it stops what is running. Two QEngine slugs running at once need two ports. Unpublished doesn't mean unreachable: inside the container the port is bound to 0.0.0.0, so the host can still reach it on the container's bridge address, as before. From another machine, use an SSH tunnel (`ssh -L 9393:127.0.0.1:9393 <rig>`).

**Not wired yet** ([ADDING_ENGINES.md](../ADDING_ENGINES.md), "CONTRACT — wiring it into the tooling"):
- the gateway's native `/v1/messages` passthrough. QEngine serves `/v1/messages` and `/v1/messages/count_tokens` itself, but the passthrough admits only `owned_by` vllm / sglang, so LiteLLM translates instead.
