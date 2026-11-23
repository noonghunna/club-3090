# Engine flags — inventory

**What this is.** One table per engine family listing every engine flag our shipped composes actually pass (and, for the launcher-driven engines, every env setting the compose hands to a launcher), what the engine's *own* documentation says it does, how many registered slugs pass it (and through which `${ENV:-default}`), and — the column that did not exist anywhere before — whether the flag's validity or value depends on the detected card, and by which mechanism. Look things up here instead of re-deriving them from 163 composes.

**Generated 2026-11-16** against commit `e36e6d40` (`master`, "BENCHMARKS: multi4 Maya-S24 row from #2073"). `scripts/lib/profiles/registry.yaml` = **164 non-deprecated curated slugs over 163 composes**: 67 vLLM (4 engine ids) · 24 SGLang · 2 exllamav3 · 39 llama.cpp-family (4 engine ids) · 3 vLLM derivatives (HyperQwen 1, bucko-vllm 2) · 7 Strata · 16 Project Maya · 6 QEngine (§§9–10 count at `baef2cb6`: next paragraph). Counts exclude `compose/_archive/`, unregistered compose files, the gitignored `profiles-local` layer, and the **42 `🗑️ Deprecated` slugs** (15 `ik-llama/*` + 8 `llamacpp/*` on `llama-cpp-local`, 10 `beellama/*`, 8 vLLM, 1 DiffusionGemma). ⚠️ The 2026-09-21 page's "138" **included** deprecated slugs (all 15 ik and 10 beellama sat inside its 70 llama.cpp-family), so 138 → 164 is not a like-for-like comparison.

**§§9–10 re-read 2026-11-20** against commit `baef2cb6` (`master`, "Merge pull request #2129 from noonghunna/strata-maya/hy4-tp-settings"), for the Strata 0.2.17 and Project Maya v1.1.70 pins. There the registry has **181 non-deprecated slugs over 180 composes** and **43 deprecated**. The 18 new slugs are all Project Maya's (the DeepSeek and MiniMax multi4s of 2026-11-19 and 16 Hy4-Flash slugs: 34 in all), and `vllm/qwen38-flash-next-dual` was deprecated (vLLM 66). Strata is still 7. §§1–8 and 11 keep their `e36e6d40` numbers, the vLLM count and §2's slug counts included; in §7 only the Strata and Maya rows were brought up to date.

**ik_llama.cpp and beellama.cpp are out of the catalog** — every `ik-llama/*` and `beellama/*` slug is deprecated (ik has no engine profile; beellama retired 2026-07-27), so neither is tabled here; their 09-21 rows remain in this file's git history.

**Section order.** §§1–7 keep their 2026-09-21 numbers (other pages cite §2 and §7). The four launcher-driven families added since then follow the summary as §§8–11; §7 covers them too.

**Upstream flag sets drift.** Every row carries a source tag (legend below). Where no authoritative source could be found the row says `UNVERIFIED` rather than guessing. A flag *we* pass that upstream does *not* document is reported as such — that is a finding, not an error to hide. Re-generate when an engine pin moves (the pin of record is the `spec:` line in `scripts/lib/profiles/engines/<id>.yml`, with its `# pin-current:` comment where the profile has one).

**Not a tuning guide.** Per-engine levers and their measured effects live in [`engines/`](engines/); cliffs in [`CLIFFS.md`](CLIFFS.md); the KV math in [`KV_MATH.md`](KV_MATH.md); the offload sweep order in [`OFFLOAD_MATRIX.md`](OFFLOAD_MATRIX.md).

---

## 1. How to read the columns

| Column | Meaning |
|---|---|
| **Flag** | The engine's own spelling. Short aliases (`-c`, `-ngl`, `-ot`) are the spelling our composes use. |
| **Env (compose)** / **→ becomes** | §§8–11 only: the env name as the compose sets it, and what the launcher turns it into (argv flag, config key, engine env), with the file that does it. |
| **Upstream status** | What the engine documents, condensed, with a source tag. `[…-help …]` = the pinned image's own `--help`; `[…-src …]` = source read at a named commit (used when the help is silent or the image is not local). |
| **Used by us** | `N` = registered slugs whose compose passes it, out of that family's total. `${VAR:-default}` = the env var the compose reads, with the shipped default(s) and how many slugs use each. `LIT` = hard-coded, no env var — the launcher cannot change it. **opt-in** = the flag appears only when an env var is set. |
| **Hardware-gated?** | One of the mechanism tags from §2, or `bare 24 GB` (a hand-tuned default nothing adapts) or `—` (not hardware-related). A `⚠` marks the intersection this document exists for: **hardware-dependent in reality, hard-coded in the compose.** |

### Source legend

Every help dump was run as `docker run --rm --network none --entrypoint <binary> <pinned digest> --help` — never a server. Only vLLM needed `--gpus all`: without CUDA its parser aborts (`Failed to infer device type`) before printing help.

| Tag | Source |
|---|---|
| `[vllm-help 0.33]` | `vllm serve --help=all`, image `vllm/vllm-openai:v0.33.0` (the `vllm-stable` pin, 2 455 lines) |
| `[vllm-help 0.32]` | same, `vllm/vllm-openai:v0.32.0` (the previous `vllm-stable` pin, local) — used only for the drift notes. v0.29/v0.30, the dumps the 09-21 page cited, are not local and were not re-diffed |
| `[vllm-envs 0.33]` | `vllm.envs.environment_variables` in the v0.33.0 image (env-var names only) |
| `[vllm-src 0.33]` | Python source read inside the v0.33.0 image |
| `[vllm-probe 0.33]` | `make_arg_parser(FlexibleArgumentParser()).parse_args([...])` inside the v0.33.0 image, with a bogus flag as the negative control (rejected, exit 2) |
| `[sgl-help 0.5.23]` | `python3 -m sglang.launch_server --help`, image `lmsysorg/sglang:v0.5.23` (the `sglang-stable` pin) |
| `[sgl-help 0.5.20]` | same, `lmsysorg/sglang:v0.5.20` (the 09-21 pin, local) — drift notes only |
| `[tabby-help 4728158]` | `python3 main.py --help`, image `ghcr.io/theroyallab/tabbyapi@sha256:4d5484ee…` (the `exllamav3` pin: tabbyAPI `47281587`, exllamav3 1.6.7) |
| `[tabby-help 09-21]` | same, the 09-21 pin `ghcr.io/noonghunna/tabbyapi-club3090@sha256:87490d21…` (local) — drift notes only |
| `[exl3-src 1.6.7]` | `grep` over the pinned image's `exllamav3` package (env-var names only) |
| `[lcpp-help b12359]` | `llama-server --help`, image `ghcr.io/ggml-org/llama.cpp@sha256:a8cf841e…` (the `llama-cpp-local` pin = `v1.0.0-b12359-b1b770d`) |
| `[lcpp-help b10920]` | same, the 09-21 pin `ghcr.io/ggml-org/llama.cpp@sha256:6ac92152…` (local) — drift notes only |
| `[lcpp-bin b12359]` | `grep -a` over the same image's binaries (env-var names only) |
| `[c3090-help v1.6]` / `[c3090-help v1.5]` | `llama-server --help`, `ghcr.io/noonghunna/llamacpp-club3090@sha256:69b833ee…` / `@sha256:1c228a52…` (unchanged since 09-21) |
| `[c3090-bin v1.6]` | `grep -a` over the v1.6 image's binaries (env-var names only); the v1.5 image carries the same `GGML_CUDA_MOE_CACHE_*` set |
| `[prism-help b12128]` | `llama-server --help`, `ghcr.io/noonghunna/llamacpp-prism@sha256:c68dd5a7…` (tag `b12128-d7b2896`). It printed without `--gpus` — the 09-21 page fell back to `[prism-src 9a9394a]` because that older build aborted without `libcuda.so.1` |
| `[lmcache-docs]` | LMCache/LMCache `docs/source/cli/server.rst` — carried from 09-21; the pinned `lmcache/vllm-openai@sha256:663f9b2f…` is not local, not re-checked |
| `[hq-help 3d6c954]` / `[hq-envs 3d6c954]` | `/app/venv/bin/vllm serve --help=all` / `vllm.envs` in `ghcr.io/syv-ai/hyperqwen@sha256:e0e4f9db…` (vLLM 0.33.0 + syv-ai's patches) |
| `[hq-src 3d6c954]` | HyperQwen's own scripts read from that image (`single-user/start_qwen.sh`, `resolve_config.sh`, `launcher_common.sh`, `docker/entrypoint.sh`) |
| `[hq-argv 3d6c954]` | the argv `start_qwen.sh` emits at our compose's defaults, rendered inside the image against a stub `vllm` that prints its argv (HyperQwen's own `bench/test_tp_size.sh` method; no GPU, so compute-capability defaults that apply only at `CTX=long` did not fire) |
| `[bucko-help v0.5.0]` / `[bucko-envs v0.5.0]` | `vllm serve --help=all` / `vllm.envs` in `ghcr.io/dominikbucko/qwen38-flash-next-2x3090@sha256:75adfc64…` (base reports vLLM `0.1.dev20073+g8e685d198`) |
| `[bucko-src v0.5.0]` | `/usr/local/bin/qwen38-serve`, `/opt/qwen38/configs/*.env` and bucko's Python modules in that image |
| `[strata-help 0.2.17]` | `/opt/strata/components/engine/0.1.63-linux-x86_64-cuda-cuda13/strata --help`, `ghcr.io/niko1221/strata` at digest `sha256:51d39e3f…` (294 lines, no GPU) |
| `[strata-launch-help 0.2.17]` | `/opt/strata/terminal/bin/strata launch --help`, same image (171 lines) |
| `[strata-entry 0.2.17]` | `/opt/strata/entrypoint.sh` read from the same image |
| `[strata-src 12d491da]` | Python sources in that image at upstream `12d491da` (`serve/server.py`, `serve/server_parts/http_api.py`, `strata_control/{drafting,split_engine,speed_switches}.py`, `setup.py`) |
| `[strata-bin 0.1.63]` | `grep -a` over the same image's two NVIDIA engine binaries, `strata` and `strata-split-2-sm86`: env-var names (`grep -aow` inside the image, which has no `strings`; `STRATA_DEFAULTS`, in both, is the positive control and a made-up name, in neither, the negative), and the engine's own log strings (read from a copy of `strata`) |
| `[strata-help 0.2.13]` / `[strata-launch-help 0.2.13]` / `[strata-entry 0.2.13]` / `[strata-src d2f09647]` / `[strata-bin 0.1.59]` | the same reads on the previous pin, `ghcr.io/niko1221/strata` at digest `sha256:b6e18765…` (0.2.13, engine 0.1.59; 285 / 175 lines of help), still local — drift notes only |
| `[strata-rig 2026-11-20]` | the reference rig's Strata data volume: the run configs setup wrote (`/data/install/strata-<tag>.json`) and the engine log of the 0.2.17 boots of `strata/qwen3.8-flash-next-ud-iq3xxs-dual` (`/data/state/strata/launch/club-unsloth-ud-iq3_xxs.engine.log`) |
| `[strata-launcher]` | our `models/qwen3.8-flash-next/strata/launcher/{strata-entrypoint.sh,strata_config.py}` at `baef2cb6` (unchanged since `e36e6d40`) |
| `[maya-help v1.1.70]` | `/opt/maya/serve/server.py --help` (run with the image's own `/opt/maya/.venv/bin/python`; the system `python3` has no `jinja2`), `ghcr.io/noonghunna/strata-maya-club3090` at digest `sha256:d129c0c1…`. One sentence differs from v1.1.67's (digest `sha256:98da6b3f…`, itself byte-identical to v1.1.65's and v1.1.66's) |
| `[maya-bin v1.1.70]` | `/opt/maya/build/{maya,maya-mm3,maya-ds4} --help`, same image (no GPU; `maya-mm3` prints usage and exits 2). All three byte-identical to v1.1.67's. `maya-hy4` has no `--help`: see `[hy4-src b6632a0b]` |
| `[maya-src b6632a0b]` | sources read inside that image (`/opt/maya/{serve,tools,src,include}`; upstream mw00/project-maya at `b6632a0b`, the v1.1.70 tag). Every claim first read at v1.1.67 (`56b5216a`) was re-checked here on 2026-11-20 |
| `[hy4-src b6632a0b]` | `maya-hy4` in the same tree, the source of record for an engine with no `--help`: the parser's `a == "--flag"` comparisons in `src/hy4/hy4_main.cpp`, the option defaults in `src/hy4/hy4_model.hpp`, and the `getenv("HY4_…")` reads in `src/hy4/*` |
| `[maya-reads]` | `scripts/lib/profiles/engine-config-reads.json` entries `strata-maya`, `strata-maya:mm3`, `strata-maya:ds4`, `strata-maya:hy4` (where the pinned server reads each run-config key) |
| `[maya-launcher]` | our `models/<model>/strata-maya/launcher/*` at `baef2cb6` |
| `[qe-help 1.3.6]` | `python3 server/app.py --help`, `ghcr.io/bpx03/qengine@sha256:5064093b…` (source `444b2e49`) |
| `[qe-entry 1.3.6]` | `/usr/local/bin/qengine-entrypoint` + `/app/server/run.sh` in that image |
| `[qe-bin 1.3.6]` | the env name is present (`grep -aw`) in that image's `build/{libqengine.so,libqflash.so,flash_run}` or `server/*.py` — presence only |
| `[qe-src 444b2e49]` / `[qe-docs 444b2e49]` | bpx03/QEngine source (`src/`) / `README.md` + `docs/advanced.md` at the image's revision label, read through the GitHub API (the image ships no docs and only built libraries) |
| `UNVERIFIED` | no authoritative source found. `UNVERIFIED (pinned image not local)` = the pin could not be inspected: `vllm/vllm-openai:v0.22.0` (`vllm-gemma-stable`), `vllm/vllm-openai:gemma4-unified`, `lmcache/vllm-openai@sha256:663f9b2f…` — not pulled (GHCR/Docker Hub throughput and disk) |

---

## 2. The hardware-gating mechanisms — do not conflate them

Six distinct things decide whether a flag's value tracks the card. They have different reach, different owners and different failure modes; a row's tag names which one applies.

| # | Tag | Mechanism | Where | Reach at `e36e6d40` |
|---|---|---|---|---|
| 1 | `SM floor` | A compute-capability gate on the *whole compose*, not a value. Three layers, checked by `compat.fits()` gate **C3** and by the bash `compose_hw_compose_status`: (a) engine profile `min_sm` (`engines/*.yml`: vLLM stable/gemma/gemma4-unified 7.5, lmcache 8.0, HyperQwen/bucko 8.6, SGLang 8.0, exllamav3 8.6, mainline llama.cpp 6.0, prism + all `llamacpp-club3090-*` 8.6, Strata/Maya 7.5, QEngine 8.6); (b) registry `required_sm` (8.6 on 48 slugs, 9.0 on 6, 7.5 on 1), `supported_sm` (an explicit list, 5 slugs: the four `…-27b-dual-ultra{fast,max}` Qwen3.8/ThinkingCap slugs `[8.6, 8.9, 9.0, 10.0, 12.0]`, `vllm/qwen38-flash-next-dual` `[8.0]`), `fallback_sm` (8 slugs — the weight-only Marlin fallback floor 7.5 that *replaces* `required_sm` as the hard floor); (c) the compose header `Requires-sm:` (6 vLLM composes: `7.5+` ×5, `8.0+` ×1), parsed by `scripts/lib/compose-meta.sh`. No flag *value* changes; the slug is accepted or refused. | `scripts/lib/profiles/compat.py` (C3), `scripts/lib/compose-meta.sh` | All 164 slugs (wizard/`switch.sh --list`) |
| 2 | `inject:<KEY>` | **#246 Phase 1/2 envelope injection.** `launch_compat.resolve_variant_pin` computes env exports from the detected card class (`scripts/lib/profiles/hardware/*.yml`, 12 classes, resolved by `_hardware_id_from_gpu`) and `envelopes.yml`; the launcher exports them and the compose consumes `${KEY:-default}`. Keys: `MAX_NUM_SEQS` / `MAX_RUNNING_REQUESTS` (concurrency, keyed by engine *type*: `vllm` / `sglang` only), `GPU_MEMORY_UTILIZATION` / `MEM_FRACTION` (downward-only floor, unified-memory cards), `VLLM_USE_DEEP_GEMM` (fp8/nvfp4 weights on sm 8.9/12.0/12.1 → `0`), `MOE_RESERVE_MB` (upward-only on >24 GB cards, `moe_cache` slugs), `DECODE_GRANULARITY` (a model property; its only user is deprecated). A user-set value always wins. Engine types `strata` and `qengine` are in neither table, so nothing injects into §§9–11. | `scripts/lib/profiles/launch_compat.py`, `envelopes.yml`, `scripts/launch.sh:1181-1320`, `scripts/switch.sh:1464-1609` | `resolve_variant_pin` is total (164/164 resolve, 0 raise). Per-rig reach: §7. |
| 3 | `launcher-bash:<fn>` | The bash-side hardware path in `scripts/lib/compose-meta.sh` + `scripts/preflight.sh`, keyed on compose **header comments** and live `nvidia-smi`: `resolve_offload_residency` exports `OT_G<i>` from *free* VRAM per card (additive fit model, headers `CPU-Offload-Bundle-MiB` / `-MoE-Layers` / `-GPU-Reserve-MiB` / `-First-MoE-Layer` / `-Draft-Reserve-MiB` / `-Draft-Card`; ≥2 cards only); `resolve_offload_threads` exports `THREADS=nproc/2` for any compose matching `=CPU`/`--cpu-moe`/`--n-cpu-moe`; `resolve_cpu_moe_split` derives exl3 `MOE_SPLIT` as a VRAM fit (#1373); `resolve_cpu_offload_gb` picks bucko's `CPU_OFFLOAD_GB` from **measured** points only (header `CPU-Offload-GB-Measured`, never extrapolated); `preflight_offload_split_mode` refuses `-sm row`/`tensor` under offload; `preflight_cpu_offload_ram` gates host RAM on `CPU-Offload-Host-RAM-GB` minus the residency grant; `preflight_single_card_util` warns when a user raises `GPU_MEMORY_UTILIZATION` above the single-card default. (`preflight_ik_llama_image` still exists; only deprecated ik slugs reach it.) | `scripts/lib/compose-meta.sh:495-1000`, `scripts/preflight.sh:2406-2800`; all called from `scripts/switch.sh:1700-1767` | `OT_G*`: 10 `residency` composes (7 club3090 + 3 mainline). `THREADS`: 31 llama.cpp composes + both exl3. `MOE_SPLIT`: 2 exl3. `CPU_OFFLOAD_GB`: 1 bucko compose. `CPU-Offload-Host-RAM-GB` on 60 composes (club3090 26, Maya 16, Strata 7, mainline 3, QEngine 3, exl3 2, bucko 2, vLLM 1). `Requires-min-vram-gb` / `Tensor-parallel` on 38 composes each (vLLM 30, QEngine 6, SGLang 1, HyperQwen 1) feed the fit display. |
| 4 | `boot-detect` | Decided **inside the container at boot**. (a) `scripts/detect_nvlink.sh` (bind-mounted at `/etc/club3090/`): NVLink or driver-confirmed PCIe P2P → `NCCL_P2P_LEVEL`, custom all-reduce **on** for vLLM (flag omitted), `expandable_segments` stripped; otherwise `NCCL_P2P_DISABLE=1`, `--disable-custom-all-reduce` **passed**. SGLang composes use it more strictly (#2008): the kernel is allowed only on a **full NVLink mesh**, so a P2P-only rig keeps it off. `NVLINK_MODE=auto\|force_on\|force_off\|pcie_p2p` and `DISABLE_CUSTOM_ALL_REDUCE=1` are the operator overrides (#1332). (b) `SIZE_TIER` (new): the entrypoint reads `nvidia-smi --query-gpu=memory.total`; below 30 720 MiB it switches to a "lean" tier of defaults (`SIZE_TIER=full\|lean` overrides). | `scripts/detect_nvlink.sh`; the two `single-fast` composes | (a) 59 of 67 vLLM slugs (every TP>1 compose; the 8 without are TP=1 or PP-only) and 23 of 24 SGLang slugs (the 24th is single-card). Not HyperQwen, bucko, Strata, Maya or QEngine — QEngine runs its own link probe (§11). (b) `vllm/qwen38-27b-single-fast`, `sgl/qwen38-27b-single-fast`. |
| 5 | `engine-auto` | The engine sizes itself from the device at load time; the compose only sets policy. llama.cpp `--fit` (default `on`: adjusts *unset* `-ngl`/`-c` to fit) and `-ngl auto`; club3090 `--moe-cache auto` (grants free − `GGML_CUDA_MOE_CACHE_RESERVE_MB` per device); exllamav3 `--gpu-split-auto` + `--autosplit-reserve`; vLLM's `--gpu-memory-utilization` is a *fraction* so the KV pool scales with VRAM even though the number is fixed, and `--max-model-len -1` fits the context; Strata, Maya and QEngine fit slots, context, KV residency, expert pools and link paths themselves (§§9–11). | engine | Per row |
| 6 | `bare 24 GB` | `${FLAG:-value}` hand-tuned on the reference 2× RTX 3090 with nothing adapting it. The largest class. Marked `⚠` when the right value demonstrably depends on the card (a bigger card wastes VRAM, a smaller one OOMs). | compose | Per row |

Two things that are **not** hardware gating but look like it: the `SPEC_N`/`SPEC=off` drafter contract (a user toggle that adds or removes the spec-dec flags: 54 vLLM, 31 llama.cpp, 23 SGLang, 2 exl3 slugs, plus the launcher families' own mappings in §§8–11) and `VLLM_ENFORCE_EAGER` (a cudagraph toggle, 30 vLLM slugs) — both are operator knobs, present regardless of card.

---

## 3. vLLM — 67 registered slugs, 4 engine ids

Engine pins in play: `vllm-stable` = `v0.33.0` (63 slugs), `vllm-gemma-stable` = `v0.22.0` (1: `vllm/gemma-26ba4b-single`), `vllm-gemma4-unified` = `vllm/vllm-openai:gemma4-unified` (2: `vllm/gemma-12b-single-int8-mtp`, `vllm/gemma-12b-qat-w4a16-single`), `vllm-lmcache` = `lmcache/vllm-openai@sha256:663f9b2f…` (1: `vllm/qwen-27b-dual-lmcache`). `vllm-diffusion-gemma` now has only a deprecated slug; the 4 `vllm-nightly-*` ids and `vllm-pip-baseline`/`vllm-stable-next` have **zero** registry users. HyperQwen and bucko-vllm are vLLM derivatives with their own launchers — §8. **Only v0.33.0 is local**, so the 4 slugs on the other three pins were checked against `[vllm-help 0.33]`, a different version: their rows are `UNVERIFIED (pinned image not local)` wherever the answer could differ. Every flag below is present in `[vllm-help 0.33]` except where noted.

**Headline findings for this engine**

- **`--mamba-cache-mode` is a deprecated no-op on v0.33.0.** `[vllm-help 0.33]`: "*--mamba-cache-mode is deprecated and will be removed in v0.33.0. It has no effect: Mamba prefix caching follows --enable-prefix-caching.*" (v0.32.0 still documented `align`/`none` as live strategies). 16 `vllm-stable` slugs pass `${MAMBA_CACHE_MODE:-align}`, so `MAMBA_CACHE_MODE` no longer does anything there — and when upstream finishes the removal the flag would likely become a parse error on all 16. **Dropped from those 16 composes 2026-11-17** (v0.32.0 picks align itself with prefix caching on, so the rollback pin is unaffected; `docs/UPSTREAM.md`, mamba-align row). (The lmcache slug passes `LIT align` on its own image — `UNVERIFIED (pinned image not local)`; HyperQwen's launcher passes it too, §8a.)
- **The `--no-enable-*` spellings left the help.** v0.33.0 pairs every `--enable-x` with `--disable-x` (`EnableDisableAction`, `vllm/utils/argparse_utils.py`: "`--no-enable-x` and `--no-disable-x` are still accepted for backwards compatibility") `[vllm-src 0.33]`. 3 slugs pass `--no-enable-prefix-caching` (`vllm/qwen-27b-single-nvfp4`, `vllm/qwen-27b-dual-nvfp4`, `vllm/nemotron-75b-dual-w4a16`): it **still parses** — `[vllm-probe 0.33]` gives `enable_prefix_caching=False` for both spellings and rejects a bogus flag — but it is now an undocumented alias. The positive spellings we use (`--enable-prefix-caching`, `--enable-chunked-prefill`, `--enable-prompt-tokens-details`, `--enable-auto-tool-choice`, `--enable-expert-parallel`, `--disable-custom-all-reduce`) are unchanged.
- **`int8_per_token_head` is now a stock `--kv-cache-dtype` value** (listed by both `[vllm-help 0.32]` and `[vllm-help 0.33]`). The 09-21 finding "overlay-only (PR #40391)" now holds only for the v0.22.0 gemma pin (`vllm/gemma-26ba4b-single`, `${KV_DTYPE:-int8_per_token_head}` — `UNVERIFIED (pinned image not local)`); `vllm/nemotron-75b-dual-w4a16` passes it `LIT` on stock v0.33.0. No non-deprecated slug carries `required_engine_features` any more.
- **9 slugs hard-code `--max-num-seqs`** (was 10): `vllm/qwen-27b-dual-max`, `-multi-fast`, `-multi-max`, `-dual-nvfp4` (`2`); `vllm/qwen-27b-single-nvfp4`, `vllm/qwen-a3b-preview-single`, `vllm/qwen38-27b-dual-ultramax`, `vllm/thinkingcap38-27b-dual-ultramax` (`1`); `vllm/gemma-26ba4b-dual` (`256`). The one concurrency knob that injects cannot reach them.
- ⚠ **The two `…-27b-dual-ultramax` slugs size the KV pool in absolute bytes** (`${KV_CACHE_MEMORY_BYTES:-6335076762}`) and pass **no** `--gpu-memory-utilization`, so `inject:GPU_MEMORY_UTILIZATION` has nothing to act on and the pool stays the 24 GB size on any card — while their registry `supported_sm` lists 8.6 → 12.0. They also hard-code `--tensor-parallel-size 2`, `--max-num-seqs 1`, `--kv-cache-dtype fp8_e4m3`, `--attention-backend FLASH_ATTN`, `--long-prefill-token-threshold 0` and the cudagraph capture sizes.
- **`VLLM_ATTENTION_BACKEND` is still not read** — absent from `[vllm-envs 0.33]`, and `grep -r` of the v0.33 package finds no reader `[vllm-src 0.33]`. One compose sets it (`vllm/nemotron-75b-dual-w4a16`, `=TRITON_ATTN`), and ⚠️ **correction to the 09-21 page**: that compose does **not** also pass `--attention-backend TRITON_ATTN` (it did not at `3126bc31` either), so the env is inert and the backend vLLM then selects for that compose's `int8_per_token_head` KV is **UNVERIFIED** (the compose comment calls the dtype Triton-native; nothing on this page proves which backend runs). The launchers' `case` allowlists still carry an arm for it (`launch.sh:1309`, `switch.sh:1587`).
- **New: an in-container VRAM tier.** `vllm/qwen38-27b-single-fast` reads `nvidia-smi memory.total` at boot (`boot-detect:SIZE_TIER`): under 30 720 MiB → GMU 0.95, 1 seq, 2048 batched, threshold 0, `max-model-len 81920`, MTP and vision off; otherwise 0.90 / 8 / 8192 / 4096 / `-1` (fit). Its `${MAX_NUM_SEQS:-8}`-style command-list defaults are overridden by the tier; a user value still wins.
- `VLLM_SKIP_P2P_CHECK` is **no longer hard-coded**: 56 composes forward it bare (unset → vLLM's own P2P check runs). The 09-21 page listed it as "PCIe topology assumption, hard-coded" on 43.
- `--prefix-match-unit` exists from v0.29.0 on, not in v0.22.0. All 16 users are on `vllm-stable` — safe today, would break silently on a pin roll-back.

### 3a. Placement, memory, concurrency

| Flag | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|
| `--tensor-parallel-size` | Number of tensor-parallel groups (default 1). `[vllm-help 0.33]` | 67/67 — `${TP:-2}` ×26, `${TP:-4}` ×16, `${TP:-8}` ×12, `${TP:-1}` ×9; `LIT 2` ×3 (`vllm/thinkingcap-dual-w4a8`, both `…-dual-ultramax`), `LIT 1` ×1 (`vllm/qwen38-flash-next-dual`, PP=2) | Topology. `launch.sh` exports `TP`/`PP` from the wizard's card selection (`pick_parallelism`), and the compose default equals its `<topology>/` dir. `compat.fits()` refuses a TP the selected cards cannot host. The 4 `LIT` slugs ignore the launcher. |
| `--pipeline-parallel-size` | Number of pipeline-parallel groups (default 1). `[vllm-help 0.33]` | 39/67 — `${PP:-1}` ×38, `LIT 2` ×1 (`vllm/qwen38-flash-next-dual`, with `VLLM_PP_LAYER_PARTITION=${…:-26,22}`) | Topology (launcher exports `PP`). ⚠ the PP=2 slug's layer partition is a 2-card literal. |
| `--gpu-memory-utilization` | Fraction of GPU memory for the model executor, 0–1, per-instance; default 0.92. `[vllm-help 0.33]` From v0.34.0 the default is 1.0 while the extensible KV cache is on; our composes turn that off (next row but one). `[vllm-src 0.34]` | 65/67 — `${GPU_MEMORY_UTILIZATION:-0.92}` ×32, `:-0.95` ×15, `:-0.90` ×11, `:-0.85` ×4, `:-0.91`, `:-0.94`, `:-0.96`. Absent on both `…-dual-ultramax` (absolute KV bytes instead) | `engine-auto` (fraction scales with VRAM) + `inject:GPU_MEMORY_UTILIZATION` — **downward only**, fires only for a card marked `unified_memory: true` whose `mem_util_safe < default` (today `dgx-spark`, 0.85). Never raised on bigger cards by design (Cliff-2b margin). `launcher-bash:preflight_single_card_util` warns on user increases (TP=1). Registry `mem_util` mirrors the default for `fits()`. |
| `--max-num-seqs` | Max sequences per iteration. `[vllm-help 0.33]` | 67/67 — `${MAX_NUM_SEQS:-2}` ×28, `:-1` ×14, `:-8` ×10, `:-4` ×5, `:-256` ×1; **`LIT`** ×9 (list above) | `inject:MAX_NUM_SEQS` — `envelopes.yml` rows reach 2 slugs on 2×5090, 3 on 2×A6000, 1 on DGX Spark (§7). Every other slug and card class: `bare 24 GB` ⚠ (the 9 `LIT` slugs cannot be reached at all). Heterogeneous rigs clamp to the smallest card. `vllm/qwen38-27b-single-fast`: `boot-detect:SIZE_TIER`. |
| `--max-model-len` | Context length (prompt + output); `-1`/`auto` = the largest that fits in GPU memory. `[vllm-help 0.33]` | 67/67 — `${MAX_MODEL_LEN:-262144}` ×51, `${MAX_MODEL_LEN:-${CTX:-262144}}` ×4, `:-65536` ×2, `:-147456` ×2, and one each of `:-32768`, `${CTX:-229376}`, `:-176000`, `:-8192`, `:-131072`, `:-200000`, `:-204800`, `:--1` (single-fast; its tier sets 81920 on a <30 GB card) | `bare 24 GB` ⚠ — the context lever is **deliberately not injected** (`envelopes.yml` header). Fit is *checked* by kv-calc in `fits()`, never *adapted*; the one `-1` is `engine-auto`. |
| `--max-num-batched-tokens` | Max tokens per scheduler iteration (chunk size). `[vllm-help 0.33]` | 64/67 — `${MAX_NUM_BATCHED_TOKENS:-8192}` ×54, `:-4096` ×6, `:-2048` ×2, `:-1584` (lmcache: must be ≥ the int8-PTH block), `:-1024` | `bare 24 GB` ⚠ — sets the prefill activation peak that decides the single-card prefill cliff (`CLIFFS.md`); a 32 GB card could chunk larger, a 12 GB card needs smaller. |
| `--long-prefill-token-threshold` | Chunked prefill: "long" above N tokens; 0 disables the cap (default); not applied when the request is alone in the batch. `[vllm-help 0.33]` | 47/67 — `${LONG_PREFILL_TOKEN_THRESHOLD:-4096}` ×39, `:-2048` ×5, `:-0` ×1, `LIT 0` ×2 | `bare 24 GB` ⚠ (prefill scheduling shares the activation budget with `--max-num-batched-tokens`) |
| `--kv-cache-memory-bytes` | Absolute KV pool size per GPU; **ignores `gpu_memory_utilization` when set**. `[vllm-help 0.33]` | 4/67 — `${KV_CACHE_MEMORY_BYTES:-6335076762}` ×2 (`…-dual-ultramax`), `:-820000000` ×1, computed in-entrypoint ×1 (`vllm/qwen-27b-dual-nvfp4`) | `bare 24 GB` ⚠ — an absolute byte count does **not** scale with VRAM the way the fraction does. |
| `--enable-extensible-kv-cache` (we pass `--no-…`) | New in v0.34.0, on by default on CUDA unless `--kv-cache-memory-bytes` or `num_gpu_blocks_override` is set: sizes KV after warmup and keeps the larger of the activation peak and torch's reservation growth free in full, plus a margin of at least 1 GiB, on top of the memory fraction. `[vllm-src 0.34]` | 58 composes — `LIT --no-enable-extensible-kv-cache` (every non-deprecated vllm-stable compose that sets `--gpu-memory-utilization`) | — (costs 262K its fit on 24 GB cards: vllm#68184) |
| `--kv-cache-dtype` | `auto` / `fp8` (= `fp8_e4m3`) / `fp8_e5m2` / `bfloat16` / `float16` / `int8_per_token_head` / `*_per_token_head` / `nvfp4*` / `octave_*` / `turboquant_*` / …; "an explicit float16 or bfloat16 must match the model data type". `[vllm-help 0.33]` | 64/67 — `${KV_CACHE_DTYPE:-fp8_e4m3}` ×33, `:-fp8` ×10, `:-bfloat16` ×8, `:-auto` ×4, `${KV_DTYPE:-auto}` ×2, `LIT fp8_e4m3` ×2, `${KV_CACHE_DTYPE:-int8_per_token_head}` ×1 (lmcache), `${KV_DTYPE:-int8_per_token_head}` ×1 (gemma v0.22), `LIT auto`, `LIT fp8`, `LIT int8_per_token_head` ×1 each | `SM floor` in reality: on sm_86 fp8 KV is **storage only** (no FP8 compute — `docs/HARDWARE.md`); `hardware/*.yml` `supported_kv_formats` gate it in `fits()` (C4). No injector (`inject:KV_CACHE_DTYPE` retired in #1371). What decides e4m3-on-Ampere is the backend route, encoded as `_fp8w_ampere_kv` in `compat.py`'s C5 gate. |
| `--dtype` | `auto`/`bfloat16`/`float`/`float16`/`float32`/`half` (default auto). `[vllm-help 0.33]` | 60/67 — `LIT bfloat16` ×52, `LIT float16` ×8 | `SM floor` in reality (bf16 needs sm_80+; every supported class qualifies) — hard-coded, harmless. |
| `--quantization` | Weight quantization method; `None` reads `quantization_config`. `[vllm-help 0.33]` | 57/67 — `LIT auto_round` ×27, `fp8` ×21, `modelopt` ×6, `compressed-tensors` ×3 | `SM floor` — fp8 weights on sm_86 run the Marlin W8A16 path (no native FP8); `modelopt` (NVFP4) slugs carry registry `required_sm: 9.0` + `fallback_sm: 7.5`. `inject:VLLM_USE_DEEP_GEMM=0` fires for `fp8*`/`nvfp4` weights on sm 8.9/12.0/12.1 (31 composes read the env). |
| `--enable-expert-parallel` | Expert parallelism instead of TP for MoE layers. `[vllm-help 0.33]` | 2/67 — bare (both Nemotron) | — |
| `--attention-backend` | Attention backend; `auto`/None = automatic. `[vllm-help 0.33]` | 12/67 — `${ATTN_BACKEND:-FLASH_ATTN}` ×10, `LIT FLASH_ATTN` ×2 (all `…-27b-*-ultra*` DFlash2 slugs) | `SM floor` in reality ⚠ — FA2 is the sm_80+ path; FA3 is sm_90-only and never selected here. Hard-coded; a Hopper rig gets FA2 unless the operator overrides. |
| `--enforce-eager` | Always eager PyTorch; disables CUDA graphs and `torch.compile`. `[vllm-help 0.33]` | 30/67 — `${VLLM_ENFORCE_EAGER:+--enforce-eager}` in the entrypoint (opt-in) | — (operator toggle) |
| `--compilation-config` | JSON `torch.compile`/cudagraph config. `[vllm-help 0.33]` | 4/67 — `LIT` capture sizes ×3, entrypoint-built `FULL_DECODE_ONLY` ×1 (`vllm/qwen38-flash-next-dual`) | — |
| `--async-scheduling` / `--no-async-scheduling` | Async scheduling on/off. `[vllm-help 0.33]` | `LIT` ×1 / opt-in ×6 (`ASYNC_SCHED=off`, the vllm#50021 mitigation) | — |
| `--disable-custom-all-reduce` | Disable the custom all-reduce kernel, fall back to NCCL; inverse now spelled `--enable-custom-all-reduce`. `[vllm-help 0.33]` | 59/67 — **never in `command:`**; added by the entrypoint as `$AR` when `_CUSTOM_AR_ENABLED != 1` | `boot-detect` — `detect_nvlink.sh` decides at boot (NVLink / driver-confirmed P2P → kernel on; PCIe → flag passed). vLLM additionally vetoes its kernel at world > 2 without a full NVLink mesh (#786). Override: `DISABLE_CUSTOM_ALL_REDUCE=1`, `NVLINK_MODE=…`. |
| `--enable-chunked-prefill` | Prefill chunking on `max_num_batched_tokens`. `[vllm-help 0.33]` | 60/67 — bare | — |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | Prefix caching on/off; the off spelling in help is now `--disable-prefix-caching`. `[vllm-help 0.33]` `[vllm-probe 0.33]` | 59 (`${PREFIX_CACHE_ARG:---enable-prefix-caching}` ×51, bare ×8) / 3 (`--no-…`: `${PREFIX_CACHE_ARG:-…}` ×2, bare ×1) | — |
| `--prefix-match-unit` | Finest token boundary a prefix-cache hit can land on; must divide every KV group's block size. `[vllm-help 0.33]`; **absent in 0.22** | 16/67 — `${PREFIX_MATCH_UNIT:-16}` (all on `vllm-stable`) | — (pin-gated, not hardware) |
| `--mamba-cache-mode` | **Deprecated, no effect** on v0.33.0 (headline). `[vllm-help 0.33]` | 17/67 — `${MAMBA_CACHE_MODE:-align}` ×16, `LIT align` ×1 (lmcache, `UNVERIFIED (pinned image not local)`); since 2026-11-17 only the lmcache one | — |
| `--mamba-ssm-cache-dtype` | SSM-state dtype `{auto,bfloat16,float16,float32}`; `auto` follows `mamba_cache_dtype`. `[vllm-help 0.33]` | 41/67 — entrypoint `MAMBA_ARGS` from `${MAMBA_SSM_CACHE_DTYPE:-bfloat16}` ×40 (omitted when `auto`), `LIT float16` ×1 | — |
| `--mamba-block-size`, `--mamba-cache-dtype` | Mamba block size (the multiple-of-8 rule was dropped from the v0.33 help) / conv + SSM cache dtype. `[vllm-help 0.33]` | 16 / 15 — opt-in (`MAMBA_BLOCK_SIZE` / `MAMBA_CACHE_DTYPE` set) | — |
| `--mamba-backend`, `--enable-mamba-cache-stochastic-rounding`, `--mamba-cache-philox-rounds`, `--enable-mamba-shared-prefix-checkpoint` | SSU backend (default triton); stochastic rounding to the fp16 cache; PRNG rounds; an extra Mamba checkpoint at the shared-prefix junction. `[vllm-help 0.33]` | 2 (`flashinfer` ×1, `triton` ×1) / 1 / 1 / 1 (`${SHARED_PREFIX_CKPT:-1}`, `vllm/qwen38-flash-next-dual`) | `--mamba-backend flashinfer` (1 slug): FlashInfer must support the active GPU + dtype per `--help`; not gated by us. |
| `--kv-offloading-size` + `--kv-offloading-backend native` / `--kv-transfer-config` | Host-RAM KV offload buffer in GiB, **total across TP ranks** / KV connector JSON. `[vllm-help 0.33]` | 5 / 5 opt-in (`KV_OFFLOAD_GB`; `KV_OFFLOAD_DISK=1` switches to a `TieringOffloadingSpec` connector) + `LIT` `LMCacheMPConnector` ×1 (lmcache) | Host RAM — `bare`; the entrypoint refuses a size larger than free `/dev/shm`. |
| `--language-model-only`, `--engram-config`, `--block-size` | Disable all multimodal inputs; n-gram-embedding storage (`{"cpu_offload":true}`); KV block size. `[vllm-help 0.33]` | 3 (`LIT` ×1, `VISION=off` / tier ×2) / 1 / 1 opt-in (`BLOCK_SIZE`) | — |

### 3b. Model, API and sampling (not hardware-related)

| Flag | Upstream status | Used by us | HW |
|---|---|---|---|
| `--model`, `--served-model-name`, `--host`, `--port` | `[vllm-help 0.33]` | 66 (the PP=2 Flash-Next compose passes the path positionally) / 67 / 67 / 67 — `LIT` (port 8000) | — |
| `--trust-remote-code` | `[vllm-help 0.33]` | 64/67 | — |
| `--reasoning-parser` | `[vllm-help 0.33]` (`none` now documented as "the same as leaving it unset") | 67/67 — `qwen3` ×58, `gemma4` ×7, `nemotron_v3` ×2 | — |
| `--tool-call-parser`, `--enable-auto-tool-choice` | `[vllm-help 0.33]` | 67 / 67 — `qwen3_coder` ×59 (+ `${TOOL_CALL_PARSER:-qwen3_coder}` ×1), `gemma4` ×7 | — |
| `--chat-template` | `[vllm-help 0.33]` | 60/67 — the froggeric / reasoning-effort Qwen templates, the canonical Gemma template | — |
| `--default-chat-template-kwargs` | `[vllm-help 0.33]` | 60/67 — `{"enable_thinking": ${ENABLE_THINKING:-true}, "reasoning_effort": "${REASONING_EFFORT:-low}"}` ×40, `{"enable_thinking": false}` ×15, others ×5 | — |
| `--override-generation-config` | `[vllm-help 0.33]` | 66/67 — sampler JSON from `TEMP`/`TOP_P`/`TOP_K`/`MIN_P`/`PRESENCE_PENALTY`/`REPEAT_PENALTY` (40 entrypoint-built with a thinking/instruct row, 26 in `command:`) | — |
| `--enable-prompt-tokens-details` | `[vllm-help 0.33]` (off spelling now `--disable-prompt-tokens-details`) | 41/67 — `${PROMPT_TOKENS_DETAILS_ARG:---enable-prompt-tokens-details}` | — |
| `--speculative-config` | JSON spec-decode config (`method`, `model`, `num_speculative_tokens`). `[vllm-help 0.33]` | 54/67 — entrypoint `SPEC_ARGS` under the `SPEC_N`/`SPEC=off` contract: `mtp` ×26, `dflash` (DFlash2) ×24, Gemma assistant draft model ×4 | — (model-gated: EAGLE/original DFlash inside vLLM blocked on Qwen3-Next — `docs/UPSTREAM.md` vllm#39931) |
| `--limit-mm-per-prompt`, `--mm-processor-kwargs` | `[vllm-help 0.33]` | 10 / 2 | — |

### 3c. `lmcache server` (a second process in the lmcache compose, not vLLM)

The pinned `lmcache/vllm-openai@sha256:663f9b2f…` is not local; these rows are carried from 09-21 (`[lmcache-docs]`) with the compose re-read at `e36e6d40` — same flags, same defaults, entrypoint-built as `LM_ARGS`.

| Flag | Upstream status | Used by us | HW |
|---|---|---|---|
| `--chunk-size` | Chunk size in tokens. `[lmcache-docs]` | 1 — `${LMCACHE_CHUNK_SIZE:-1584}` (must be a multiple of vLLM's block) | — |
| `--l1-size-gb` | L1 (CPU RAM) tier size. `[lmcache-docs]` | 1 — `${LMCACHE_L1_GB:-30}` | Host RAM: `preflight.sh` (`preflight_lmcache_ram`) gates available RAM ≥ l1 + a vLLM reserve; `shm_size` must be ≥ this. `bare` for the size itself. |
| `--eviction-policy` | `[lmcache-docs]` | 1 — `LIT LRU` | — |
| `--l2-adapter` | **UNVERIFIED** — `server.rst` documents L2 adapters as config-file settings and points to `lmcache server --help`; the flag itself is not listed. | 1 — opt-in (`LMCACHE_L2=1` → an `fs` adapter on `/lmcache-kv`, or `LMCACHE_L2_ADAPTER=…`) | — |

### 3d. vLLM environment knobs our composes set (names checked against `[vllm-envs 0.33]`)

| Env | In `vllm.envs` 0.33? | Slugs | HW |
|---|---|---|---|
| `VLLM_USE_DEEP_GEMM` | yes | 31 (bare pass-through) | `inject:VLLM_USE_DEEP_GEMM` → `0` on sm 8.9 / 12.0 / 12.1 for fp8/nvfp4 weights (DeepGEMM is Hopper + datacenter-Blackwell only; disc #571) |
| `VLLM_SKIP_P2P_CHECK` | yes | 56 (bare pass-through, unset) | — (was hard-coded at 09-21) |
| `VLLM_USE_FLASHINFER_SAMPLER` | yes | 50 — `${…:-0}` ×38, `LIT 0` ×7, `LIT 1` ×5 | — |
| `VLLM_HUMMING_INPUT_QUANT_CONFIG` (+ our `W4A8`) | yes `[vllm-src 0.34]` | 20 (bare ×19, `LIT {"dtype":"int8"}` ×1); `W4A8` gate as before | — (W4A8 path on vLLM v0.34.0: the entrypoint sets `{"dtype":"int8"}` and `--quantization humming` when `W4A8≠0`. `VLLM_MARLIN_INPUT_DTYPE` is a startup error on v0.34.0 (#65542); only the retired `vllm/qwen38-flash-next-dual`, held on v0.33.0, still forwards it) |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | yes | 7 — `LIT 1` | — |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, `VLLM_LOG_STATS_INTERVAL`, `VLLM_CACHE_ROOT`, `VLLM_PP_LAYER_PARTITION` | yes | 3 / 4 / 4 / 1 | `VLLM_PP_LAYER_PARTITION` `${…:-26,22}` ⚠ — a 2-stage literal |
| `VLLM_NO_USAGE_STATS`, `VLLM_WORKER_MULTIPROC_METHOD` | yes | 67 each — `LIT 1`, `LIT spawn` | — |
| `VLLM_ATTENTION_BACKEND` | **no** — and no reader in the package `[vllm-src 0.33]` | 1 (`environment: …=TRITON_ATTN`, `vllm/nemotron-75b-dual-w4a16`) | inert (headline) |
| `VLLM_PLE_MMAP_REBUILD`, `VLLM_PLE_PREFETCH` (+ `VLLM_PLE_MMAP_PATH`, entrypoint-set) | **no** — read by our `qwen4exp-ple-mmap` overlay (`ple-mmap-v0.33.0.diff`) | 1 (`vllm/qwen38-flash-next-dual`) | — |
| `NCCL_P2P_DISABLE`, `NCCL_CUMEM_ENABLE`, `PYTORCH_CUDA_ALLOC_CONF` | n/a (NCCL / PyTorch) | 65 / 65 / 66 — `NCCL_P2P_DISABLE` `${…:-1}` ×46, `LIT 1` ×19 | `boot-detect` — `detect_nvlink.sh` rewrites `NCCL_P2P_*` and strips `expandable_segments` on the P2P path |
| `FA2_ARTIFACT_ID` | ours (the FA2 FP8-KV sidecar artifact, `test-fa2-artifact`) | 12 | — |
| `VLLM_ENFORCE_EAGER`, `SPEC_N`, `SPEC`, `DRAFT_SAMPLE_METHOD`, `MAMBA_*`, `ASYNC_SCHED`, `KV_OFFLOAD_*`, `SIZE_TIER`, `VISION`, `PLE_PLACEMENT`, `SHARED_PREFIX_CKPT`, `THINKING_BUDGET*` | ours, not vLLM's | — | entrypoint contracts |

---

## 4. SGLang — 24 registered slugs, 1 engine id

All on `sglang-stable` = `lmsysorg/sglang:v0.5.23`: 23 Qwen3.8-27B / ThinkingCap-Qwen3.8-27B composes (`🧪 Experimental`) and `sgl/nanbeige43-33b-dual` (`🐣 Incubating`, not hybrid — no mamba, spec-dec or strict-thinking flags). All are `bash -c` script bodies (no `command:` list), so every flag below is entrypoint-built. Every flag below is in `[sgl-help 0.5.23]`; drift was checked against `[sgl-help 0.5.20]`, the 09-21 pin.

**Headline findings for this engine**

- **The 09-21 "no `detect_nvlink.sh`" headline is gone.** 23 of 24 composes source it (#2008) and drop `--disable-custom-all-reduce` only when the cards form a **full NVLink mesh** (`_NVLINK_FOUND=1`, and world ≤ 2 or no partial mesh). Stricter than vLLM: a PCIe-P2P grant alone keeps the kernel off. The 24th, `sgl/qwen38-27b-single-fast`, is single-card and keeps the literal.
- **TP is still a literal on 23 of 24** (`--tp-size 2|4|8` from the topology dir); only `sgl/qwen38-27b-single-fast` reads `${TP:-1}`.
- **`MEM_FRACTION` injection reaches 14 slugs on a unified-memory card** (§7), and `MAX_RUNNING_REQUESTS` is plumbed but has no `envelopes.yml` row. The 09-21 registry mismatches are fixed: `max_num_seqs` and `mem_util` now equal each compose's shipped default on all 24.
- **New since 09-21:** a host-RAM KV tier (opt-in `KV_OFFLOAD_GB`, 4 dual slugs → HiCache), `--mamba-radix-cache-strategy`, `--enable-strict-thinking`, `--sleep-on-idle`, and the `SIZE_TIER` VRAM tier on `sgl/qwen38-27b-single-fast`.
- **Drift 0.5.20 → 0.5.23 for flags we pass:** choice lists only grew (`--attention-backend` +`llada2_cfg_flashinfer`, `--radix-eviction-policy` +`tlru`, `--hicache-storage-backend` +`tensorcast`/`seaweedfs`, new parsers); `--kv-cache-dtype` **dropped `fp4_e2m1`** (unused here); `--enable-hierarchical-cache` gained the alias `--enable-hicache`. Nothing we pass disappeared.

| Flag | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|
| `--tp-size` | Tensor parallelism size (alias `--tensor-parallel-size`). `[sgl-help 0.5.23]` | 24/24 — **`LIT`** `2` ×7, `4` ×8, `8` ×8; `${TP:-1}` ×1 | Topology ⚠ — hard-coded on 23; `compat.fits()` still refuses a mismatch, but nothing adapts it. |
| `--disable-custom-all-reduce` | Disable the custom all-reduce kernel, fall back to NCCL. `[sgl-help 0.5.23]` | 24/24 — `$_AR` ×23 (omitted on a full NVLink mesh), `LIT` ×1 (single-card) | `boot-detect` (NVLink-only, #2008). Override: `DISABLE_CUSTOM_ALL_REDUCE=1`, `NVLINK_MODE`. |
| `--mem-fraction-static` | Fraction for static allocation (weights + KV pool); lower on OOM. `[sgl-help 0.5.23]` | 24/24 — `${MEM_FRACTION:-0.82}` ×6, `:-0.86` ×4, `:-0.84` ×4, `:-0.90` ×4, `:-0.95` ×3, `:-0.92` ×2, `:-0.88` ×1; `SPEC_N=0` on an MTP compose sets 0.90 when unset; single-fast lean tier 0.88 | `engine-auto` (fraction) + `inject:MEM_FRACTION` (downward, unified-memory cards) + `bare 24 GB`. ⚠️ the injected value is the card's own `mem_util_safe` ceiling — unvalidated on SGLang. |
| `--max-running-requests` | Max running requests. `[sgl-help 0.5.23]` | 24/24 — `${MAX_RUNNING_REQUESTS:-2}` ×18, `:-1` ×4, `:-8` ×1 (nanbeige), tier ×1 (lean 1 / full 2) | `inject:MAX_RUNNING_REQUESTS` plumbed, **inert** (no `envelopes.yml` row). Otherwise `bare 24 GB` ⚠. |
| `--context-length` | Model max context; SI/IEC suffixes. `[sgl-help 0.5.23]` | 24/24 — `${CONTEXT_LENGTH:-262144}` ×20, `:-163840` ×2, `:-204800` ×1, tier ×1 (lean 32768 / full 262144) | `bare 24 GB` ⚠ (single-fast: `boot-detect:SIZE_TIER`) |
| `--chunked-prefill-size` | Max tokens per prefill chunk; −1 disables. `[sgl-help 0.5.23]` | 24/24 — `${CHUNKED_PREFILL_SIZE:-2048}` | `bare 24 GB` ⚠ (same role as vLLM `--max-num-batched-tokens`) |
| `--kv-cache-dtype` | `{auto, fp8_e5m2, fp8_e4m3, mxfp8, bf16, bfloat16, nvfp4, fp4_mx_block16}`; fp8 needs CUDA 11.8+. `[sgl-help 0.5.23]` | 24/24 — `${KV_CACHE_DTYPE:-fp8_e4m3}` ×16, `:-auto` ×8 | `SM floor` in reality — storage-only on sm_86; `hardware/*.yml` `supported_kv_formats` gates in `fits()`. No injector for this engine. |
| `--attention-backend` | Kernel choice; 25 backends incl. `fa3`, `fa4`, `flashinfer`, `triton`. `[sgl-help 0.5.23]` | 24/24 — `${ATTENTION_BACKEND:-flashinfer}` | `SM floor` in reality ⚠ — `fa3`/`fa4` are Hopper+/Blackwell paths never selected here; hard-coded to the sm_86-safe backend. |
| `--quantization` | Weight quantization method. `[sgl-help 0.5.23]` | 24/24 — `LIT auto-round` ×14, `${QUANTIZATION:-fp8}` ×10 | `SM floor` in reality — fp8 weights on sm_86 take the non-native path; `W4A8` env selects our W4A8 kernel patch (23 composes). |
| `--mamba-ssm-dtype`, `--mamba-radix-cache-strategy` | SSM state dtype (else model config); `{auto, no_buffer, extra_buffer, extra_buffer_lazy}`. `[sgl-help 0.5.23]` | 23 / 23 — `${MAMBA_SSM_DTYPE:-bfloat16}`, `${MAMBA_RADIX_CACHE_STRATEGY:-auto}` (`no_buffer` refused: it needs `--disable-overlap-schedule`, which we don't expose) | — |
| `--max-mamba-cache-size` | Max mamba cache entries. `[sgl-help 0.5.23]` | 5/24 — `${MAX_MAMBA_CACHE_SIZE:-20}` ×3, `:-10` ×2 (`auto` omits it) | `bare 24 GB` |
| `--enable-hierarchical-cache`, `--hicache-size`, `--mamba-max-states-per-path 1`, `--hicache-storage-backend file`, `--hicache-storage-prefetch-policy wait_complete` | HiCache host tier (size in GB, overrides `--hicache-ratio`); per-path Mamba state cap; storage backend + prefetch policy. `[sgl-help 0.5.23]` | 4/24 — opt-in (`KV_OFFLOAD_GB`, per-GPU size = GiB × 2³⁰ / 10⁹ / 2; disk flags with `KV_OFFLOAD_DISK=1`, `SGLANG_HICACHE_FILE_BACKEND_*` env) | Host RAM — `bare` (the `/ 2` is the dual topology the 4 slugs are in) |
| `--enable-session-radix-cache`, `--radix-eviction-policy` | Per-session radix refs; eviction `{lru,lfu,slru,priority,tlru}`. `[sgl-help 0.5.23]` | 23 / 23 — opt-in (`SESSION_RADIX_CACHE=1`, `RADIX_EVICTION_POLICY`; `slru` default under HiCache) | — |
| `--speculative-algorithm` | Builtins `EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK, UNO`. `[sgl-help 0.5.23]` | 23/24 — `EAGLE` (MTP head, 13) or `DFLASH` (10 `dflash2*.yml`), under `SPEC_N` | — |
| `--speculative-draft-model-path` | Draft model path. `[sgl-help 0.5.23]` | 23/24 | — |
| `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens` | EAGLE depth / top-k / draft tokens. `[sgl-help 0.5.23]` | 13/24 each — `SPEC_N`, `1`, `SPEC_N+1` | — |
| `--speculative-dflash-block-size` | DFLASH only; alias of `--speculative-num-draft-tokens`. `[sgl-help 0.5.23]` | 10/24 — from `SPEC_N` | — |
| `--speculative-draft-model-quantization` | Drafter quantization. `[sgl-help 0.5.23]` | 10/24 — `${DRAFT_QUANT:-unquant}` | — |
| `--model-path`, `--served-model-name`, `--host`, `--port`, `--trust-remote-code` | `[sgl-help 0.5.23]` | 24/24 (`--trust-remote-code` 23) — `LIT` (port 30000) | — |
| `--reasoning-parser`, `--tool-call-parser`, `--default-chat-template-kwargs`, `--preferred-sampling-params`, `--sampling-defaults model` | `[sgl-help 0.5.23]`. `--preferred-sampling-params` is documented only as "settings that will be returned in /get_model_info"; `--sampling-defaults model` takes request defaults from `generation_config.json` | 24/24 each — `${REASONING_PARSER:-qwen3}`, `${TOOL_CALL_PARSER:-qwen3_coder}` | — |
| `--enable-strict-thinking`, `--enable-cache-report`, `--enable-metrics`, `--sleep-on-idle` | `[sgl-help 0.5.23]` | 23 / 24 / 24 / 24 | — |

Script-body artefacts excluded: `--tensor-parallel-size 2` (×23) is the argument the composes pass to `detect_nvlink.sh`, not SGLang; `--disable-overlap-schedule` appears only in an error message; `--verify` (×23) is the vendored w4a8 install script's flag; `--engine/--default-effort/--low/--medium/--xhigh` belong to `effort_budget.py`.

---

## 5. exllamav3 (TabbyAPI) — 2 registered slugs, 1 engine id

Both `🐣 Incubating` (`exllamav3/qwen38-flash-next-dual-exl3-305-cpumoe`, `-405-cpumoe`). The pin moved from our own `tabbyapi-club3090` build to theroyallab's official `ghcr.io/theroyallab/tabbyapi@sha256:4d5484ee…` (`:cu13`, tabbyAPI `47281587`, exllamav3 1.6.7; pinned 2026-11-16, not booted here — no EXL3 weights on the reference rig). `min_sm: 8.6`. TabbyAPI is config-file driven; the compose passes CLI overrides on top of `--override-preset safe_defaults`. Every flag below is in `[tabby-help 4728158]`. `registry.offload = cpu-moe-split`; `moe_cache = False`; no `mem_util`.

**Headline findings:** `--cpu-moe-split-experts` is no longer a bare default — `launcher-bash:resolve_cpu_moe_split` (#1373) exports `MOE_SPLIT` as a VRAM fit from the safetensors tensor table (switch.sh only; one calibration point per tier). **Drift `[tabby-help 09-21]` → `[tabby-help 4728158]` for flags we pass:** `--reasoning` now defaults to **True** (was False) and `--reasoning-start-token`/`--reasoning-end-token` and `--tool-format` to **auto** (were `<think>`/`</think>`/blank) — we pass all four explicitly, so nothing moved; `--autosplit-reserve` now treats a negative value as "exclude this GPU"; `--override-preset safe_defaults` applies temperature last. No flag we pass disappeared.

| Flag | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|
| `--cpu-moe-split-experts` | Routed experts per MoE layer computed on the CPU (default 0); splits *every* MoE layer, hot experts stay in VRAM; not with TP. `[tabby-help 4728158]` | 2/2 — `${MOE_SPLIT:-120}` (3.05 bpw), `${MOE_SPLIT:-256}` (4.05 bpw) | `launcher-bash:resolve_cpu_moe_split` — a fit, not a fraction. ⚠️ one calibration point per tier, so other rigs are labelled extrapolations at boot. Not in any injector. |
| `--cpu-moe-threads` | Worker threads for CPU MoE; unset = library default. `[tabby-help 4728158]` | 2/2 — from `THREADS`, else `nproc/2` computed in the entrypoint | `launcher-bash:resolve_offload_threads` exports `THREADS=nproc/2`; the entrypoint computes the same floor. Measured good zone is an absolute ~24-32 (`docs/engines/EXLLAMAV3.md`) — CPU-keyed, not VRAM-keyed. |
| `--gpu-split-auto` | Auto-allocate across GPUs (default True); ignored on 1 GPU. `[tabby-help 4728158]` | 2/2 — `LIT true` | `engine-auto` |
| `--autosplit-reserve` | VRAM reserved when loading, MB per GPU (default 96 on GPU 0); a negative value excludes that GPU. `[tabby-help 4728158]` | 2/2 — `${RESERVE:-512}` `${RESERVE:-512}` (two entries) | `bare 24 GB` ⚠ — **two** list entries hard-code a 2-card rig; a 4-card rig gets no reserve on cards 2-3. |
| `--cache-size` / `--max-seq-len` | KV cache size / max sequence length (tokens). `[tabby-help 4728158]` | 2/2 — `${CTX:-204800}` for both | `bare 24 GB` ⚠ (204800 is where it was left, not a ceiling) |
| `--cache-mode` | KV quantization `k_bits,v_bits` 2-8, or legacy FP16/Q8/Q6/Q4. `[tabby-help 4728158]` | 2/2 — `${KV_TYPE:-Q4}` | `bare` |
| `--chunk-size` | Prefill chunk size. `[tabby-help 4728158]` | 2/2 — `${CHUNK:-4096}` | `bare 24 GB` |
| `--sysmem-kv-cache`, `--sysmem-recurrent-cache` | Host-RAM KV / recurrent-state caches (MB). `[tabby-help 4728158]` | 2/2 — `${SYSMEM_KV:-4096}`, `${SYSMEM_RECURRENT:-8192}` | Host RAM — `bare` |
| `--ngram-ram` | Load the n-gram embedding table into system RAM (default False; PLE models only). `[tabby-help 4728158]` | 2/2 — `${NGRAM_RAM:-true}` | Host RAM (the `CPU-Offload-Host-RAM-GB` header gates it) |
| `--vision-offload` | Keep vision weights in pinned host RAM, stream to GPU. `[tabby-help 4728158]` | 2/2 — `${VISION_OFFLOAD:-true}` | Host RAM — `bare` |
| `--tensor-parallel` | TP on/off (default off). `[tabby-help 4728158]` | 2/2 — `LIT false` (autosplit, not TP) | Topology — hard-coded off by design |
| `--draft-mode`, `--draft-num-tokens`, `--dynamic-draft` | Built-in MTP drafter; with dynamic draft, `--draft-num-tokens` is a ceiling. `[tabby-help 4728158]` | 2/2 — `mtp`, `SPEC_N` (default 4), `${DYNAMIC_DRAFT:-true}` | — |
| `--backend exllamav3`, `--api-servers OAI`, `--disable-auth true`, `--host`, `--port 5000`, `--model-dir`, `--model-name`, `--override-preset safe_defaults` | `[tabby-help 4728158]` | 2/2 — `LIT` | — |
| `--vision true`, `--tool-format qwen3_coder`, `--tool-calls-in-reasoning true`, `--reasoning`, `--reasoning-start-token`, `--reasoning-end-token`, `--start-in-reasoning auto` | `[tabby-help 4728158]` | 2/2 | — |
| env `EXL3_MOE_PINNED_ARENA`, `EXL3_MOE_ARENA_HUGE`, `EXL3_MOE_ARENA_DEBUG` | read by `exllamav3/model/moe_cpu_host.py` / `config.py` `[exl3-src 1.6.7]` | 2/2 — pass-through, unset (the entrypoint checks `2m`/`1g` and reports hugepage state) | Host RAM — `bare` |

---

## 6. llama.cpp family — 39 registered slugs, 4 engine ids

| Sub-engine | Slugs | Engine id(s) | Image | Help source |
|---|---|---|---|---|
| mainline | 9 (`llamacpp/*`) | `llama-cpp-local` | `ghcr.io/ggml-org/llama.cpp@sha256:a8cf841e…` (v1.0.0-b12359) | `[lcpp-help b12359]` |
| club3090 (moe-cache) | 28 (`llamacpp-club3090/*`) | `llamacpp-club3090-v1.5` (5), `-v1.6` (23); `llamacpp-club3090`, `-v1.1` have 0 users | `ghcr.io/noonghunna/llamacpp-club3090@sha256:1c228a52…` / `@sha256:69b833ee…` | `[c3090-help v1.5]` / `[c3090-help v1.6]` |
| prism | 2 (`llama-cpp-prism/*`) | `llama-cpp-prism` | `ghcr.io/noonghunna/llamacpp-prism@sha256:c68dd5a7…` (b12128-d7b2896) | `[prism-help b12128]` |

Every flag a compose passes was checked against its **own** pinned image's help; none is missing (single-letter aliases `-b`/`-c`/`-m`/`-n` included). The club3090 help differs from mainline b12359 by `--moe-cache`, `--tensor-read-lazy`, `-dio`, `-dt` and the deprecated `--mlock`/`--mmap`/`--no-mmap` (present in club3090), and by 17 mainline-only flags. Prism adds `--kv-mean-center`, `--reasoning-effort-allow`, `--reasoning-effort-fallback`, `--reasoning-max-tokens-floor`, `--spec-draft-depth-max`.

**Headline findings for this family**

- **Mainline grew its own expert cache.** b12359 has `--moe-cache-mib N` ("GPU cache size in MiB for the MoE experts kept in the CPU … split among them like the layers") and `--moe-cache-cpu-share F` `[lcpp-help b12359]`, neither in b10920. **No compose uses them**; our club3090 fork's `--moe-cache MODE` is a different flag with different semantics, and the fork's `GGML_CUDA_MOE_CACHE_*` env set is absent from mainline `[lcpp-bin b12359]`.
- **Mainline's default port moved 8080 → 9931** between b10920 and b12359 `[lcpp-help b12359]`. All 39 composes pass `--port 8080` explicitly, so nothing moved — a compose that ever drops the flag would listen elsewhere. `--host` now takes a comma-separated list.
- **Other drift b10920 → b12359 on flags we pass:** `--cache-type-k/-v` gained `f8_e4m3`; sampling flags gained `LLAMA_ARG_*` env names. `-dt` was removed (unused). `--fit` / `--load-mode` are unchanged. The 09-21 `--no-mmap` work is complete: **0** live composes pass it (its one holdout was the now-deprecated `ik-llama/apex-fit-q8q5`).
- **`-t`/`--threads` is never passed as a flag.** 31 composes (28 club3090 + 3 mainline offload) export `LLAMA_ARG_THREADS` from `THREADS` inside the entrypoint, defaulting to `nproc/2` — the same floor `resolve_offload_threads` computes; the compose comments say the measured optimum is an absolute ~24-32.
- **`MOE_RESERVE_MB` now reaches the moe-cache slugs.** It is consumed as `GGML_CUDA_MOE_CACHE_RESERVE_MB=${MOE_RESERVE_MB:-1536}` (28 composes, name in `[c3090-bin v1.6]`), and `inject:MOE_RESERVE_MB` fires for the 22 slugs with `MOE_CACHE` on, on every class above 24 GB (§7). The 6 `*-offload` GLM slugs ship `MOE_CACHE:-off`.
- **`-ot` is where hardware adaptation actually lives** for 10 residency slugs: `resolve_offload_residency` sizes `OT_G<i>` from free VRAM per card with an additive, field-calibrated model, and the RAM gate prices exactly what it grants. The other 21 `-ot` composes pin *all* experts to CPU (`…exps\.weight=CPU`) — the config that runs anywhere.
- **22 of 39 composes pass no `--cache-type-k/-v`** (all 13 GLM-5.3 and both DeepSeek-vision club3090 v1.6 composes, the 5 v1.5 composes, 2 mainline DeepSeek) and so run the engine default `f16`.

### 6a. Placement, memory, threads

| Flag | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|
| `-ngl` / `--gpu-layers` | Max layers in VRAM: number, `auto` or `all` (default `auto`). `[lcpp-help b12359]` | 39/39 — `LIT 99` | `engine-auto` exists (`auto` + `--fit on`) but we opt out: full offload is hard-coded; the offload composes move experts with `-ot` instead. |
| `-c` / `--ctx-size` | Context size (0 = from model). `[lcpp-help b12359]` | 39/39 — `${CTX:-204800}` ×28, `${CTX_SIZE:-262144}` ×6, `${CTX:-262144}` ×5 | `bare 24 GB` ⚠ — on the offload slugs the KV pool competes with the expert cache for the same VRAM; nothing re-sizes it per card. |
| `-b` / `--batch-size` | Logical max batch (default 2048). `[lcpp-help b12359]` | 39/39 — `${UBATCH:-4096}` ×17, `${UBATCH:-2048}` ×11, `${BATCH_SIZE:-4096}` ×5, `LIT 4096` ×3, `${UBATCH:-512}` ×2, `${BATCH_SIZE:-2048}` ×1 (30 composes set `-b` and `-ub` from one var) | `bare 24 GB` ⚠ (moves with `-ub`, next row) |
| `-ub` / `--ubatch-size` | Physical max batch (default 512); sets the compute-buffer peak. `[lcpp-help b12359]` | 39/39 — `${UBATCH:-4096}` ×17, `${UBATCH:-2048}` ×11, `${UBATCH_SIZE:-512}` ×5, `LIT 4096` ×3, `${UBATCH:-512}` ×2, `${UBATCH_SIZE:-4096}` ×1 | `bare 24 GB` ⚠ — the compute-buffer swing this sets is what `GGML_CUDA_MOE_CACHE_RESERVE_MB` was measured against (1 128 MiB at ub4096); changing one without the other is the documented 11 % regression. |
| `-np` / `--parallel` | Server slots (default −1 = auto). `[lcpp-help b12359]` | 39/39 — `${NPARALLEL:-1}` ×33, `${NP:-1}` ×6 | `bare` — the family exposes **no concurrency knob to the injector by design** (`_ENGINE_TYPE_CONCURRENCY_ENV` omits llama.cpp). Two env names for the same knob. |
| `-t` / `--threads` | CPU threads for generation (default −1); env `LLAMA_ARG_THREADS`. `[lcpp-help b12359]` | **0 as a flag**; 31 via `LLAMA_ARG_THREADS=${THREADS}` in the entrypoint | `launcher-bash:resolve_offload_threads` → `nproc/2` (CPU-count keyed, not VRAM). Explicit `THREADS` always wins. |
| `-ts` / `--tensor-split` | Fraction of the model per GPU, e.g. `3,1`. `[lcpp-help b12359]` | 33/39 — `LIT 1,1` ×13, `${TENSOR_SPLIT:-1,1,1,1}` ×9, `:-1,1,1,1,1,1,1,1` ×7, `:-0.55,0.45` ×2, `LIT 1,1,1,1` ×2 | `bare 24 GB` ⚠ — even splits assume matched cards; `launch.sh`'s topology advisory *recommends* `--tensor-split` for VRAM-mismatched rigs but injects nothing. The 15 `LIT` composes cannot be corrected without editing. |
| `-sm` / `--split-mode` | `none`/`layer`/`row`/`tensor` (default `layer`). `[lcpp-help b12359]` | 32/39 — `LIT layer` ×31, `${SPLIT_MODE:-layer}` ×1 | `launcher-bash:preflight_offload_split_mode` **refuses** `row`/`tensor` under CPU offload (measured −85 % prefill) — a guard, not an adapter. |
| `-ot` / `--override-tensor` | `<pattern>=<buffer type>,…`; env `LLAMA_ARG_OVERRIDE_TENSOR`. `[lcpp-help b12359]` | 31/39 — 21 static `blk\.[0-9]+\.ffn_(gate\|up\|down)_exps\.weight=CPU` (registry `offload: tensor-override`), 10 with leading `${OT_G0}…${OT_G<n>}` slots (registry `offload: residency`) | `launcher-bash:resolve_offload_residency` for the 10 residency slugs (6 club3090 v1.6 + 1 v1.5 + 3 mainline): `OT_G<i>` from detected free VRAM, ≥2 cards, outer-edge layer selection. User `OT_G<i>` always wins. The 21 static ones: `bare`, but the safe direction. ⚠ `-ot` force-disables pipeline parallelism (engine property — `learnings/moe-cache-engine.md` §11). |
| `--moe-cache` | club3090 only: cache the hottest CPU-resident experts in spare VRAM — `auto` (preserve repacking) / `on` / `off` / `N` MiB per device. `[c3090-help v1.6]` — **not in mainline**, which has its own `--moe-cache-mib` (headline) | 28/39 — `${MOE_CACHE:-auto}` ×22, `${MOE_CACHE:-off}` ×6 | `engine-auto` (grants free − reserve per device) + `inject:MOE_RESERVE_MB` via `GGML_CUDA_MOE_CACHE_RESERVE_MB`. The binary also carries `GGML_CUDA_MOE_CACHE_MIN_CC` (an SM floor inside the engine; semantics not verified) `[c3090-bin v1.6]`. |
| `--cache-type-k` / `-ctk`, `--cache-type-v` / `-ctv` | KV dtype: `f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` (default f16; `f8_e4m3` new in mainline since b10920). `[lcpp-help b12359]` | 17/39 — `${KV_TYPE:-q8_0}` ×9, `${KV_TYPE:-f16}` ×5, `${KV_TYPE:-q4_0}` ×2, `LIT q8_0` ×1 | `bare` — quality floor, not hardware |
| `-fa` / `--flash-attn` | `on`/`off`/`auto` (default auto). `[lcpp-help b12359]` | 39/39 — `LIT on` (38 as `-fa`, 1 as `--flash-attn`) | — (hard-coded on; every supported class has the CUDA FA path) |
| `--load-mode` | `auto` (mmap unless a device can't) / `none` / `mmap` / `mmap+mlock` / …. `[lcpp-help b12359]` `[c3090-help v1.6]` `[prism-help b12128]` | 31/39 — `LIT none` (all offload composes) | Host RAM — `bare` |
| `--fit` | `[on\|off]` default on: adjust unset args to fit device memory (`--fit-target` per device, `--fit-ctx`). `[lcpp-help b12359]` | 31/39 — `LIT off` | `engine-auto` — deliberately **switched off** on every offload compose so the explicit residency plan stands. |
| `--cache-prompt` | Prompt caching (default on). `[lcpp-help b12359]` | 31/39 — bare | — |
| `-devd` / `--spec-draft-device` | Devices for the draft model (`none` = don't offload; default follows `--device`). `[lcpp-help b12359]` | 20/39 (club3090) — `${DRAFT_DEVICE:-CUDA1}` ×18 (GLM v1.6), `LIT none` ×2 (v1.5) | `bare 24 GB` ⚠ — assumes a second card named `CUDA1`; the residency sizer charges the drafter's VRAM to that card via `CPU-Offload-Draft-Card` (#1233). |
| `-mmdev` / `--mmproj-device` | Device for the multimodal projector (`none` = don't offload; default follows `--device`); env `MTMD_BACKEND_DEVICE`. `[lcpp-help b12359]` | 23/39 — `${MMPROJ_DEVICE:-none}` | `bare` (projector kept on CPU to protect the expert pool) |
| `--mmproj` | Multimodal projector file. `[lcpp-help b12359]` | 26/39 | — |
| `--override-kv` | Override model metadata `KEY=TYPE:VALUE`. `[lcpp-help b12359]` | 1/39 — `gemma4.context_length=int:${CTX_SIZE:-262144}` | — |

### 6b. Speculative decoding

| Flag | Upstream status | Used by us | HW |
|---|---|---|---|
| `--spec-type` | `none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache` (comma list). `[lcpp-help b12359]` `[c3090-help v1.6]` `[prism-help b12128]` | 31/39 — `draft-dflash` ×18 (GLM v1.6), `draft-dspark` ×5 (DeepSeek: v1.5 2, mainline 3), `draft-mtp` ×4 (mainline), `ngram-mod` ×3 (Flash-Next v1.6, entrypoint-built), `${SPEC_TYPE:-draft-mtp}` ×1 (prism) — all under the `SPEC_N`/`SPEC=off` contract | — |
| `--spec-draft-n-max` | Draft tokens (default 3). `[lcpp-help b12359]` | 28/39 — `${SPEC_N:-3}` ×12, `:-2` ×6, `:-5` ×5, `${SPEC_N:-${MTP_DRAFT_N_MAX:-2}}` ×3, `${SPEC_N:-${MTP_DRAFT_N_MAX:-3}}` ×1, entrypoint ×1 | — (`spec-dec is single-stream-only on CPU offload` is a workload finding, not a card gate) |
| `-md` / `--spec-draft-model` | Draft model file. `[lcpp-help b12359]` | 23/39 — GLM-5.3 DFlash2 ×18, DeepSeek DSpark ×5 | — |
| `--spec-ngram-mod-n-min` / `-n-max` / `-n-match` | ngram-mod parameters (defaults 1 / 64 / 24). `[lcpp-help b12359]` | 3/39 (club3090 v1.6 Flash-Next) — `${NGRAM_N_MIN:-1}`, `SPEC_N`, `${NGRAM_N_MATCH:-24}` | — |

Script-body artefacts excluded: `--spec-type`, `--spec-draft-n-max`, `-md`, `-devd`, `--mmproj` also appear in every drafter entrypoint as `case` patterns that *strip* those flags when `SPEC=off`.

### 6c. Model, API and sampling (not hardware-related)

| Flag | Upstream status | Used by us |
|---|---|---|
| `-m` / `--model`, `--alias`, `--host`, `--port 8080` | `[lcpp-help b12359]` (`--port` default is now 9931 — headline) | 39 / 38 / 39 / 39 |
| `--jinja`, `--chat-template-file`, `--chat-template-kwargs` (+ env `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` ×5) | `[lcpp-help b12359]` | 6 / 24 / 21 (`{"clear_thinking": ${CLEAR_THINKING:-false}}` ×18, `reasoning_effort` ×3) |
| `--reasoning`, `--reasoning-format`, `--reasoning-effort`, `--reasoning-budget`, `--reasoning-budget-message`, `--reasoning-preserve`/`--no-…` | `[lcpp-help b12359]` (`--reasoning-effort`: `default` or a level `minimal…max`) | 11 / 25 (`${REASONING_FORMAT:-deepseek}`) / 18 (`${REASONING_EFFORT:-max}`, GLM) / 6 / 6 / 5 |
| env `LLAMA_ARG_REASONING_EFFORT_ALLOW`, `LLAMA_ARG_REASONING_EFFORT_FALLBACK` | `[prism-help b12128]` only (`--reasoning-effort-allow` / `-fallback`); **not** in mainline or club3090 | 2 (prism) — `:-xhigh,medium,low`, `:-xhigh` |
| `-n` / `--predict` | `[lcpp-help b12359]` | 18 — `${MAX_TOKENS:-65536}` |
| `--image-min-tokens` | `[lcpp-help b12359]` | 2 — opt-in (`${IMAGE_MIN_TOKENS:+…}`) |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--repeat-last-n`, `--presence-penalty` | `[lcpp-help b12359]` | 32 / 32 / 11 / 14 / 8 / 1 / 9 |

### 6d. llama.cpp-family environment knobs

| Env | Composes | Source | HW |
|---|---|---|---|
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` = `${MOE_RESERVE_MB:-1536}` | 28 | `[c3090-bin v1.6]` | `inject:MOE_RESERVE_MB` — 22 slugs above 24 GB; 1536 was the 24 GB sweep optimum, 1024 measured ~11 % slower |
| `GGML_CUDA_MOE_CACHE_ADMIT_AFTER` = `${MOE_ADMIT_AFTER:-64}`, `GGML_CUDA_MOE_CACHE_STATS` = `${MOE_STATS:-0}`, `GGML_EXPERT_PREFETCH` (`${EXPERT_PREFETCH:-0}` ×26), `GGML_EXPERT_PREFETCH_SLOTS` | 28 | `[c3090-bin v1.6]` | `bare` |
| `LLAMA_ARG_THREADS` (entrypoint, from `THREADS`) | 31 | `[lcpp-help b12359]` | `launcher-bash:resolve_offload_threads` |
| `LLAMA_ARG_LOG_VERBOSITY`, `LLAMA_ARG_CACHE_REUSE`, `LLAMA_GRAPH_REUSE_DISABLE`, `LLAMA_ARG_MMPROJ_OFFLOAD` | 29 / 28 / 28 / 2 | help-documented except `LLAMA_GRAPH_REUSE_DISABLE`, which is in the binaries only (`[c3090-bin v1.6]`, `[lcpp-bin b12359]`) | — |
| `GGML_OP_OFFLOAD_MIN_BATCH` | 0 — present in every binary; **must not be set** on a cache compose (silently prevents the cache allocating, −4.2× — engine profile notes) | `[c3090-bin v1.6]` | — |

---

## 7. Cross-engine summary — the intersection this page exists for

Flags that are hardware-dependent in reality and hard-coded (or hard-tuned) in our composes:

| Engine | Flag | Why it depends on the card | What adapts it today |
|---|---|---|---|
| vLLM | `--max-num-seqs` on 9 `LIT` slugs | KV pool ∝ VRAM | nothing (env var absent from those 9) |
| vLLM | `--max-model-len` | KV pool ∝ VRAM | nothing (deferred by design); `SIZE_TIER` on 1 slug |
| vLLM | `--max-num-batched-tokens`, `--long-prefill-token-threshold` | prefill activation peak vs free VRAM (`CLIFFS.md`) | nothing (`SIZE_TIER` on 1 slug) |
| vLLM | `--kv-cache-memory-bytes` (4), incl. both `…-dual-ultramax` with no `--gpu-memory-utilization` | absolute bytes, overrides the fraction | nothing |
| vLLM | `--attention-backend FLASH_ATTN` (12) | FA3 exists on sm_90 | nothing |
| vLLM | `--kv-cache-dtype` | backend route, **not** card class | the compose default, full stop (the card-keyed injector was retired in #1371; the e4m3-on-Ampere decision is `_fp8w_ampere_kv` in `compat.py`'s C5 gate) |
| SGLang | `--tp-size` (23 `LIT`) | card count | nothing — no `${TP}` outside the single-card compose |
| SGLang | `--disable-custom-all-reduce` | NVLink/P2P | ✅ `boot-detect` since #2008 — but NVLink-mesh only; a P2P-granted PCIe rig keeps the kernel off (vLLM would turn it on) |
| SGLang | `--mem-fraction-static` | VRAM, but **not the same quantity as vLLM's utilisation** | ✅ `MEM_FRACTION`, keyed per engine type — fires on 14 slugs on unified-memory cards. ⚠️ the value is the card's own `mem_util_safe` ceiling, downward-only — **unvalidated on SGLang** |
| SGLang | `--max-running-requests` | VRAM | key mapping exists, **inert: no `envelopes.yml` row for any sgl slug**. Do not add one from a vLLM measurement — envelopes require a `validated` or `computed` basis |
| SGLang | `--context-length`, `--chunked-prefill-size` | VRAM | nothing (`SIZE_TIER` on 1 slug) |
| exllamav3 | `--cpu-moe-split-experts` | it *is* a VRAM fit (#1360) | ✅ `resolve_cpu_moe_split` (#1373) — a fit from the safetensors tensor table + a calibrated per-card reserve. ⚠️ one calibration point per tier |
| exllamav3 | `--autosplit-reserve` ×2 entries | per-GPU list | nothing — 2-card literal |
| llama.cpp | `-c`, `-ub`, `-b` on offload slugs | KV + compute buffer share VRAM with the expert cache | nothing |
| llama.cpp | `-ts` even splits (15 `LIT`) | mismatched VRAM | advisory only |
| llama.cpp | `-devd CUDA1` (18) | needs a second card | nothing |
| club3090 | `GGML_CUDA_MOE_CACHE_RESERVE_MB` | allocator working room ∝ card | ✅ `MOE_RESERVE_MB` — 22 moe-cache slugs above 24 GB. ⚠️ upward-only **safety heuristic, not a tuned optimum**: measured on 24 GB Ampere only |
| HyperQwen | `--kv-cache-memory=5583457484` (launcher `KV_MEM`), `--max-num-batched-tokens 2048` | absolute per-GPU KV bytes and the prefill peak, sized for 24 GB | nothing — not forwarded by the compose; `GPU_UTIL` (reached by `inject:GPU_MEMORY_UTILIZATION`) is ignored while the pool is pinned |
| bucko-vllm | `CUDA_VISIBLE_DEVICES "0,1"`, `--tensor-parallel-size 2`, `--disable-custom-all-reduce` | which cards, how many, NVLink/P2P | nothing — compose and image literals, no `detect_nvlink.sh`, `launch.sh --gpus` cannot reach it |
| bucko-vllm | `--kv-cache-memory-bytes 4429185024`, `VLLM_WNA16_STATIC_HOT_CACHE_SIZE 84`, `MAX_NUM_BATCHED_TOKENS 4096` | one VRAM budget shared by KV, hot cache, resident experts and prefill | nothing (the `GPU_MEMORY_UTILIZATION` injection is emitted but neither compose forwards it) |
| bucko-vllm | `--cpu-offload-gb 30` | VRAM | ✅ `resolve_cpu_offload_gb` on `-ple-dual` (30 → 22 when every card has ≥ 31 130 MiB free, measured points only); nothing on `-swift15-dual` |
| Strata | drafter, split engine, `--spec`/`--draft-cap` (`SPEC_N` unset) | model family × compute capability × card count (and from 0.2.17 a 12 GB-class card) picks DFlash2 / MTP / the 2-card split engine | ✅ `engine-auto` inside Strata; a set `SPEC_N` fixes the window on every card |
| Strata | concurrency (`PARALLEL`, never passed); `CONTEXT` → KV residency | VRAM / RAM | ✅ `engine-auto` ("fit to your graphics card; one request at a time with the DFlash2 drafter", `--kv-resident`); no `inject:` key for `type: strata` |
| Strata / Maya | `STRATA_START_GPUS` (7) / `MAYA_GPUS` (34) | card count; container ordinals must match the device mask | nothing — a per-topology default no launcher exports, handed to the engine as its card list (`--gpu` / `CUDA_VISIBLE_DEVICES`) |
| Maya | `CONTEXT` → `--max-context`, `--kv-streaming on` (32K resident; GLM, mm3) or the whole cache in VRAM (Hy4, 262144 on all 16) | KV shares VRAM with the expert pool and pinned host RAM | nothing for the KV; the expert pool and RAM tier fit themselves to what is left |
| Maya (ds4) / (mm3) | `MAYA_TP=1` / `KV` refuses `fp8` | interconnect / FP8 is native on sm_89+ | nothing — chosen on PCIe 3090s / refused on every card |
| QEngine | `QE_GPUS` (6, `LIT`) | it is the tensor-parallel width | nothing — the host-card choice reaches the container only through `device_ids` |
| QEngine | `FLASH_MAX_SEQS=8` (Flash-Next `dual`) | slot state competes with the expert cache for VRAM | nothing: the engine lowers context to fit, not slots |
| all | `THREADS` = `nproc/2` | CPU count (documented optimum is absolute 24-32) | `resolve_offload_threads` (a floor); Maya computes the same in-container |

### Injection reach, re-measured on `e36e6d40`

Re-measured **statically**: `resolve_variant_pin(profiles, slug, gpu_spec)` called for every non-deprecated slug with synthetic `idx|name|MiB|sm` specs for each rig (the format the launchers build from `nvidia-smi`), in a clean environment so no user value suppresses a key. All 164 resolve, 0 raise. Slugs receiving at least one **non-image** env, by rig:

| Rig | Slugs | Keys |
|---|--:|---|
| 2×3090 | 0 | — |
| 2×5090 | 53 | `VLLM_USE_DEEP_GEMM` 29 · `MOE_RESERVE_MB` 22 · `MAX_NUM_SEQS` 2 |
| 2×A6000 | 25 | `MOE_RESERVE_MB` 22 · `MAX_NUM_SEQS` 3 |
| 2×H100 | 22 | `MOE_RESERVE_MB` 22 |
| DGX Spark | 106 | `GPU_MEMORY_UTILIZATION` 64 · `VLLM_USE_DEEP_GEMM` 29 · `MOE_RESERVE_MB` 22 · `MEM_FRACTION` 14 · `MAX_NUM_SEQS` 1 |

Not comparable row-for-row with the 09-21 table: that one ran over 138 slugs **including** deprecated ones (its 2×3090 row's 2 hits were the now-deprecated DiffusionGemma slug's `DECODE_GRANULARITY` and a `GPU_MEMORY_UTILIZATION` floor). ⚠️ **An emitted key is not a delivered key**: the 64 `GPU_MEMORY_UTILIZATION` hits on DGX Spark include both `bucko-vllm` slugs, whose composes do not list the variable in `environment:` (docker forwards only listed vars), and HyperQwen, whose launcher pins the KV pool in bytes so the fraction does not size it (§8). Strata, Maya and QEngine receive nothing on any rig (re-checked for the 7 Strata and 34 Maya slugs at `baef2cb6`).

⭐ **The reference rig is the one that gets nothing**, which is the honest reading of these numbers: the knobs were tuned *on* 2×24 GB Ampere, so there is nothing to correct there. Everything above is the correction other hardware gets.

### What **does** adapt, for contrast

`--disable-custom-all-reduce` (vLLM boot-detect; SGLang NVLink-only) · `SIZE_TIER` on the two `single-fast` slugs · `-ot` residency slots on 10 slugs (free-VRAM fit) · `--cpu-moe-split-experts` (exl3 VRAM fit) · bucko `CPU_OFFLOAD_GB` (measured points) · `MOE_RESERVE_MB` on 22 moe-cache slugs above 24 GB · `MAX_NUM_SEQS` / `MAX_RUNNING_REQUESTS` per engine dialect · `GPU_MEMORY_UTILIZATION` / `MEM_FRACTION` downward on unified-memory cards · `VLLM_USE_DEEP_GEMM` on sm 8.9/12.x · `--fit` / `--gpu-split-auto` / `--moe-cache auto` inside the engines · Strata's, Maya's (maya-hy4's `--tp auto` included) and QEngine's own start-up fitting (§§9–11) · every SM-floor refusal.

### Pin-drift findings surfaced while sourcing rows

Not hardware, but the same "we pass it — does the pinned engine still take it?" question:

| Engine | Finding | Section |
|---|---|---|
| vLLM 0.32 → 0.33 | `--mamba-cache-mode` deprecated and inert (16 slugs + HyperQwen's launcher; dropped from the 16 on 2026-11-17) | §3, §8a |
| vLLM 0.32 → 0.33 | `--no-enable-*` → `--disable-*` in help; `--no-enable-prefix-caching` (3 slugs) still parses, undocumented | §3 |
| vLLM | `int8_per_token_head` is stock now; overlay-only only on the v0.22 gemma pin (UNVERIFIED) | §3 |
| vLLM | `VLLM_ATTENTION_BACKEND` still unread; the 09-21 claim that the Nemotron compose also passes `--attention-backend` was wrong | §3d |
| SGLang 0.5.20 → 0.5.23 | only additions on flags we pass, plus `fp4_e2m1` dropped from `--kv-cache-dtype` (unused) | §4 |
| TabbyAPI | `--reasoning` default False → True; reasoning tokens and `--tool-format` default to auto (we pass all explicitly) | §5 |
| llama.cpp b10920 → b12359 | `--port` default 8080 → 9931 (we pass 8080); mainline `--moe-cache-mib` / `--moe-cache-cpu-share` added (unused); `--no-mmap` users 1 → 0 | §6 |
| Strata 0.2.12 → 0.2.13, Maya v1.1.61 → v1.1.65, QEngine 1.3.5.9 → 1.3.6 | no flag or env we pass was removed or renamed | §§9–11 |
| Maya v1.1.65 → v1.1.67 | help byte-identical (server + 3 engines); no env read removed; 10 new upstream env reads, none set by our composes (`MAYA_PAR_MAX` caps a `--parallel` our launchers never write) | §10 |
| Maya | GLM `MAYA_PCIE_SHARE` / `MAYA_CPU_LANE` exported under `STRATA_GLM_*` names the engine reads only through a legacy alias (the launcher exports `MAYA_GLM_*` since 2026-11-17) | §10 |
| Strata 0.2.13 → 0.2.17 | no flag or env we pass was removed or renamed. The launch help lost its DeepSeek flags (unused) and gained `--draft-vocab` / `--listen`; the engine's dense-weights trim is now on by default on a layer split; the binary gained `STRATA_SPLIT_CPU_SHARE`, and the route tail skip is off by default (both forwarded since #2114) | §9 |
| Strata | `STRATA_ROUTE_TAIL_SKIP=7` is replaced by the `0` setup writes into the Unsloth run configs' `env` (2 slugs), and the nine `STRATA_*` pass-throughs are absent from the all-cards split engine `strata-split-2-sm86` (also at 0.1.59) | §9 |
| Maya v1.1.67 → v1.1.70 | a fourth engine, `maya-hy4` (no `--help`; its parser takes every flag we write). The server help changed in one sentence; the other three engines' help is byte-identical. No env read was removed; 101 new reads (93 `HY4_*`, 8 DeepSeek / GLM kernel switches), none set by our composes except the 4 `HY4_*` the Hy4 composes forward | §10 |

⚠️ **The reach table will rot the same way the rows above do.** It describes *injection*, which changes whenever a resolver or a launcher gate does — unlike §§1–6 and 8–11, which change when an engine pin or a compose moves. Re-derive it with `resolve_variant_pin` across the registry rather than trusting the numbers here.

---


## 8. vLLM derivatives — 3 registered slugs, 2 engine ids

Two third-party vLLM builds whose composes pass settings as env to the image's **own launcher**, not as `vllm serve` argv. Both profiles carry `min_sm: 8.6` (`SM floor` on the whole compose) and are pinned by digest to the image their maintainer publishes.

### 8a. HyperQwen — 1 registered slug, engine id `vllm-hyperqwen`

`vllm-hyperqwen/qwen38-27b-single-ultrafast` (🐣 Incubating), image `ghcr.io/syv-ai/hyperqwen@sha256:e0e4f9db…` (`:sha-3d6c954`, commit `3d6c9543`, vLLM 0.33.0 + syv-ai's patch series, built by their CI). Our compose's `bash -c` wrapper validates `SPEC_N` / `REASONING_EFFORT` / `MAX_NUM_SEQS` / `PTH_FA2_SLAB_PREFILL`, translates them to HyperQwen's names (`DFLASH_TOKENS`, `SPEC=off`, `MAX_SEQS`, `EXTRA_ARGS`) and execs `docker/entrypoint.sh single` → `prepare.sh` → `verify.sh --no-server` → `single-user/start_qwen.sh` → `venv/bin/vllm serve`. `[hq-help 3d6c954]` has **exactly the flag names of `[vllm-help 0.33]`**; the only difference in a choice list we can reach is `--kv-cache-dtype` adding `kvarn_*` (`CTX=huge`, not shipped).

**Headline findings for this engine**

- ⚠ **The KV pool is an absolute byte count our compose cannot see.** At the shipped `SPEC=dflash2 CTX=fast` the launcher always appends `--kv-cache-memory=5583457484` (its `KV_MEM` default) `[hq-argv 3d6c954]`; the compose does not forward `KV_MEM`. Per `[hq-help 3d6c954]` that value is per GPU and "ignores gpu_memory_utilization" — so `GPU_UTIL`, which the compose fills from `${GPU_MEMORY_UTILIZATION:-0.93}` and which `inject:GPU_MEMORY_UTILIZATION` therefore reaches, does **not** size the pool on the default path. The launcher's own note says these memory defaults "are sized for the shipped 1.2 GiB head and a 24 GiB card" `[hq-src 3d6c954]`.
- **`--kv-cache-memory=` is a spelling no `--help` lists.** It resolves only by argparse prefix abbreviation of `--kv-cache-memory-bytes` (vLLM's parser sources set no `allow_abbrev`); a future `--kv-cache-memory-*` flag would make it ambiguous. It boots today (pin-current: verify-full all checks).
- ⚠ `--max-num-batched-tokens 2048` is a literal in `start_qwen.sh`; only `EXTRA_ARGS` reaches it.
- GPU selection is the standard vLLM knob (`count: all` + `NVIDIA_VISIBLE_DEVICES` + bare `CUDA_VISIBLE_DEVICES`), so `launch.sh --gpus` reaches it.

| Env (compose) | → becomes | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|---|
| `CTX` | `fast` → `--max-model-len 65536 --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16`; `long` → 131072 (dflash2) with int8 / fp8 KV per backend arm; `huge` → `--kv-cache-dtype kvarn_k4v2_g128 --block-size 128` `[hq-src 3d6c954]` | flags `[hq-help 3d6c954]`; `kvarn_*` dtypes only in this image | 1/1 — `${CTX:-fast}` | ⚠ `bare 24 GB` — a context tier picked for one 24 GB card |
| `PTH_FA`, `FA2_FP8KV`, `PTH_FA2_SLAB_PREFILL` | `CTX=long` backend arms: pth-flash-attn int8 KV (`PTH_FA=1`), `--attention-backend FLASH_ATTN --kv-cache-dtype fp8` (`FA2_FP8KV=1`), else int8 on `TRITON_ATTN`; slab prefill read by `pthfa/…/fa2_slab.py` `[hq-src 3d6c954]` | HyperQwen's own env, not vLLM's | 1/1 — `${PTH_FA:-}`, `${FA2_FP8KV:-}` (empty → launcher default), slab pass-through, unset | `engine-auto` — `resolve_config.sh` picks the arm from the device's compute capability (8.6 defaults). Inert at `CTX=fast` |
| `GPU_UTIL` | `--gpu-memory-utilization` `[hq-argv 3d6c954]` | fraction; ignored when `kv_cache_memory_bytes` is set `[hq-help 3d6c954]` | 1/1 — `${GPU_UTIL:-${GPU_MEMORY_UTILIZATION:-0.93}}` | `inject:GPU_MEMORY_UTILIZATION` reaches it, **inert** while the pool is pinned (next row) |
| — (`KV_MEM`, not forwarded) | `--kv-cache-memory=5583457484` `[hq-argv 3d6c954]` | abbreviation of `--kv-cache-memory-bytes`, KV per GPU `[hq-help 3d6c954]` | 1/1 — `LIT` (launcher default) | ⚠ `bare 24 GB` — absolute bytes do not scale with VRAM |
| `MAX_NUM_SEQS` | our wrapper → `MAX_SEQS` → `--max-num-seqs` (launcher default 8 on dflash2) | `[hq-help 3d6c954]` | 1/1 — pass-through, unset | `inject:MAX_NUM_SEQS` plumbed (engine type `vllm`), no `envelopes.yml` row → `bare 24 GB`. The launcher itself warns the pool holds ~5-6 resident ~1k-token requests, 1 at 64K |
| — | `--max-num-batched-tokens 2048` | `[hq-help 3d6c954]` | 1/1 — `LIT` (image) | ⚠ `bare 24 GB` (prefill activation peak, `CLIFFS.md`) |
| `SPEC`, `DFLASH_TOKENS`, `SPEC_N` | `--speculative-config {"method":"dflash","model":<DFlash2 W4A16>,"num_speculative_tokens":7,"draft_sample_method":"probabilistic"}`; `SPEC_N=0` → `SPEC=off` | `[hq-help 3d6c954]` | 1/1 — `${SPEC:-dflash2}`, `${DFLASH_TOKENS:-7}`, `SPEC_N` pass-through | — |
| `PREFIX_CACHE` | `1` → `--enable-prefix-caching --mamba-cache-mode align --prefix-cache-retention-interval None`; `0` → `--no-enable-prefix-caching` `[hq-argv 3d6c954]` | `[hq-help 3d6c954]`, as stock 0.33: `--mamba-cache-mode` is **deprecated with no effect** (§3 headline) and the off spelling in help is `--disable-prefix-caching` (`--no-enable-…` still parses as a back-compat alias, §3) | 1/1 — `${PREFIX_CACHE:-1}` | — |
| `REASONING_EFFORT`, `EXTRA_ARGS` | wrapper prepends `--default-chat-template-kwargs {"reasoning_effort":"low"}`; `EXTRA_ARGS` is appended verbatim | `[hq-help 3d6c954]` | 1/1 — pass-through (wrapper default `low`); `${EXTRA_ARGS:-}` | — |
| launcher literals | `--mamba-ssm-cache-dtype float16`, `--async-scheduling`, `--compilation-config {"max_cudagraph_capture_size":64,"custom_ops":["+rms_norm","+silu_and_mul"]}`, `--api-server-count 1`, `--language-model-only` (`VISION` not forwarded), `--reasoning-parser qwen3`, `--enable-auto-tool-choice --tool-call-parser qwen3_coder --tool-strict-level parameter`, `--enable-prompt-tokens-details`, `--sse-keep-alive-interval 30`, `--served-model-name qwen3.8-27b`; exports `VLLM_USE_FLASHINFER_SAMPLER=0` | `[hq-help 3d6c954]` | 1/1 — `LIT` (image) | — |
| `PORT`, `BIND_HOST`, `HOME`, `HF_HUB_OFFLINE`, `HUGGING_FACE_HUB_TOKEN` | `--host 0.0.0.0 --port 8000`; HF client env (`HUGGING_FACE_HUB_TOKEN` is read by none of HyperQwen's scripts) | `[hq-help 3d6c954]` / n/a | 1/1 — `LIT` | — |

### 8b. bucko-vllm — 2 registered slugs, engine id `bucko-vllm`

`bucko-vllm/qwen3.8-flash-next-ple-dual` and `-swift15-dual` (both 🐣 Incubating), image `ghcr.io/dominikbucko/qwen38-flash-next-2x3090@sha256:75adfc64…` (v0.5.0, revision `e7279ab7`, DominikBucko's release image). Its base, `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece…`, reports vLLM `0.1.dev20073+g8e685d198` — a dev build, not a release, so comparisons with 0.33 are **by flag name only**. Our entrypoint checks `MTP_DEPTH`, runs our `default-thinking-budget/apply.py`, and execs `/usr/local/bin/qwen38-serve <model> [<mtp model>]`, which sources `configs/2x3090-128gb.env` (`: "${X:=default}"`, so a compose value wins) and runs one fixed `vllm serve`. Every flag it emits is in `[bucko-help v0.5.0]` and, by name, in `[vllm-help 0.33]`.

**Headline findings for this engine**

- ⚠ **Topology is hard-coded in three places.** The image launcher passes `--tensor-parallel-size 2 --enable-expert-parallel` literally, and the compose sets `CUDA_VISIBLE_DEVICES: "0,1"` as a literal under `gpus: all`. A literal in `environment:` is not interpolated, so the UUIDs `launch.sh --gpus` exports never reach the container; this shape matches neither arm of `test-compose-gpu-mask-passthrough` (the `NVIDIA_VISIBLE_DEVICES` env line, `device_ids`), and the post-boot `gpu_select_assert_placement` can only warn. Static reading, not boot-tested.
- ⚠ **Custom all-reduce is off on every rig.** `DISABLE_CUSTOM_ALL_REDUCE:-1` → `--disable-custom-all-reduce`; no `detect_nvlink.sh`. The image refuses `0` while `PYTORCH_CUDA_ALLOC_CONF` holds `expandable_segments:True` (the compose default), so an NVLink/P2P rig needs two overrides.
- ⚠ **Memory is set in absolute units, not a fraction.** `--kv-cache-memory-bytes 4429185024` (per GPU, ignores `gpu_memory_utilization` — the registry's `mem_util: 0.9` is nominal; `resolve_variant_pin` emits `GPU_MEMORY_UTILIZATION` for both slugs on a DGX Spark, but neither compose lists it in `environment:`, so that injection never reaches the container) and `--cpu-offload-gb 30` (per GPU) `[bucko-help v0.5.0]`. Only the second adapts: `resolve_cpu_offload_gb` (`switch.sh` only; `launch.sh` does not call it) lowers 30 → 22 when every card has ≥ 31,130 MiB free — on `-ple-dual` only; the Swift-1.5 compose has no `CPU-Offload-GB-Measured` header.
- **Env this image's `vllm.envs` does not register**, read straight from `os.environ` by bucko's own modules `[bucko-src v0.5.0]`: `VLLM_WNA16_STATIC_HOT_CACHE_SIZE`/`_MAX_TOKENS`, `VLLM_QSA_EXACT_TOPK`, `VLLM_MTP_DRAFT_VOCAB_RANGES`, `QWEN38_*`. None is in `[vllm-envs 0.33]` either.
- **`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` is an env here and a flag in stock.** `[bucko-envs v0.5.0]` registers it (default `None` = dense); stock 0.33 has `--prefix-cache-retention-interval` instead, which `[bucko-help v0.5.0]` lacks. The compose ships `1600` and itself flags it SUSPECT (sparse retention on a hybrid + MTP model), untested.

| Env (compose) | → becomes | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` | exported unchanged by `qwen38-serve` | n/a (CUDA) | 2/2 — `LIT "0,1"` (+ `gpus: all`) | ⚠ Topology — `launch.sh --gpus` cannot reach it (headline) |
| — | `--tensor-parallel-size 2 --enable-expert-parallel --all2all-backend allgather_reducescatter` `[bucko-src v0.5.0]` | `[bucko-help v0.5.0]` | 2/2 — `LIT` (image) | ⚠ Topology — 2 cards |
| `DISABLE_CUSTOM_ALL_REDUCE` | `1` → `--disable-custom-all-reduce`; `0` → omitted, refused under `expandable_segments:True` `[bucko-src v0.5.0]` | `[bucko-help v0.5.0]` | 2/2 — `${DISABLE_CUSTOM_ALL_REDUCE:-1}` | ⚠ `boot-detect` in reality, hard-coded off |
| `KV_CACHE_MEMORY_BYTES` | `--kv-cache-memory-bytes` | KV per GPU in bytes; ignores `gpu_memory_utilization` `[bucko-help v0.5.0]` | 2/2 — `${KV_CACHE_MEMORY_BYTES:-4429185024}` | ⚠ `bare 24 GB` |
| `CPU_OFFLOAD_GB` | `--offload-backend uva --cpu-offload-gb N --cpu-offload-params experts` (128 GB profile; `QWEN38_HOT_ONLY>0`, not forwarded, would replace it) | "space in GiB to offload to CPU, per GPU"; `uva` = zero-copy UVA `[bucko-help v0.5.0]` | 2/2 — `${CPU_OFFLOAD_GB:-30}` | `launcher-bash:resolve_cpu_offload_gb` on 1/2 (measured points only, never extrapolated); ⚠ `bare 24 GB` on `-swift15-dual` |
| `VLLM_WNA16_STATIC_HOT_CACHE_SIZE`, `…_MAX_TOKENS` | `os.getenv` in `compressed_tensors_moe_wna16.py` / `auto_gptq.py` (defaults 0 / 16) `[bucko-src v0.5.0]` | not in `[bucko-envs v0.5.0]` | 2/2 — `:-84`, `:-16` | ⚠ `bare 24 GB` — a VRAM-side expert copy sharing one budget with KV and resident experts; the compose records 88 OOMing at KV allocation on 2× 5090 |
| `MAX_NUM_BATCHED_TOKENS`, `MAX_MODEL_LEN` | `--max-num-batched-tokens`, `--max-model-len` | `[bucko-help v0.5.0]` | 2/2 — `:-4096`, `:-262144` | ⚠ `bare 24 GB` (prefill activation peak) / `bare 24 GB` |
| `MAX_NUM_SEQS` | `--max-num-seqs` | `[bucko-help v0.5.0]` | 2/2 — `:-1` | `inject:MAX_NUM_SEQS` plumbed, no `envelopes.yml` row → `bare` (single-stream by design) |
| `MAX_PARALLEL_LOADING_WORKERS` | `--max-parallel-loading-workers` | `[bucko-help v0.5.0]` | 2/2 — `:-1` | Host RAM — `bare` |
| `MTP_DEPTH` (← `SPEC_N`) | `--speculative-config {"method":"mtp","num_speculative_tokens":N,"use_local_argmax_reduction":true,"model":<compact MTP>}`; `0` refused by our entrypoint | `--speculative-config` `[bucko-help v0.5.0]`; `use_local_argmax_reduction` is a `SpeculativeConfig` field here and in `[vllm-src 0.33]` | 2/2 — `${MTP_DEPTH:-${SPEC_N:-3}}` | — |
| `QWEN38_ASYNC_SCHEDULING` | `1` → `--async-scheduling`, `0` → `--no-async-scheduling` `[bucko-src v0.5.0]` | `[bucko-help v0.5.0]` | 2/2 — `:-1` (image default 0) | — |
| `QWEN38_STREAM_STAGE` (+`_MIN_TOKENS`), `QWEN38_STAGE_OVERLAP`, `QWEN38_TRITON_SKINNY`, `QWEN38_PLE_PREFAULT` | read by bucko's overlay modules (`stream_stage.py`, `tiered_runtime.py`, `compressed_tensors_moe_wna16.py`, `ops/hc.py`, `v1/ple_offload/worker.py`); image defaults 0 / 3000 / 0 / 0 / 0 `[bucko-src v0.5.0]` | bucko's own; semantics UNVERIFIED beyond the module names | 2/2 — `:-1`, `:-1536`, `:-1`, `:-1`, `:-1` | — |
| `VLLM_QSA_EXACT_TOPK`, `VLLM_MTP_DRAFT_VOCAB_RANGES` | `os.environ` in `qwen3_8_flash_next/nvidia/ops/qsa.py`, `…/mtp.py` `[bucko-src v0.5.0]` | not in `[bucko-envs v0.5.0]` | 2/2 — `:-0`, `:-[[0,65536],[248044,248320]]` | — |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT`, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | engine env | in `[bucko-envs v0.5.0]` (defaults 600, `None`) | 2/2 — `:-1200`, `:-1600` | — |
| `THINKING_TOKEN_BUDGET` → `CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET` | read only by our `patches/default-thinking-budget/apply.py` (`SamplingParams` fallback) | ours (`patches.yml`) | 2/2 — `${THINKING_TOKEN_BUDGET:-16384}` | — |
| not forwarded: `KV_CACHE_DTYPE`, `ENABLE_VISION`, `QWEN38_HOT_ONLY` | `--kv-cache-dtype auto` (bf16 on this model), `--language-model-only`, the 128 GB profile | `[bucko-help v0.5.0]` | 2/2 — image defaults; a shell value never reaches the container | — |
| launcher literals | `--moe-backend humming`, `--dtype bfloat16`, `--load-format safetensors --safetensors-load-strategy lazy`, `--enable-chunked-prefill`, `--enable-prefix-caching --mamba-cache-mode align`, `--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}`, `--trust-remote-code`, `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`, `--host 0.0.0.0 --port 8000`; exports `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_FORCE_DYNAMIC_SPEC_SCHEDULING=1`, `VLLM_WNA16_DYNAMIC_LRU=1`, `VLLM_WNA16_MIXED_VMM_HOT_CACHE=1` | `[bucko-help v0.5.0]`; `humming` is also a `--moe-backend` choice in `[vllm-help 0.33]` | 2/2 — `LIT` (image) | — |
| `PYTORCH_CUDA_ALLOC_CONF`, `SERVED_MODEL_NAME`, `PORT` | allocator (also gates the all-reduce check above); `--served-model-name` | PyTorch / `[bucko-help v0.5.0]` | 2/2 — `:-expandable_segments:True`; per-slug names; `LIT 8000` | — |

---


## 9. Strata — 7 registered slugs, engine id `strata`

The pin is `ghcr.io/niko1221/strata` at digest `sha256:51d39e3f…`, which is 0.2.17 (upstream `12d491da`, engine 0.1.63), pinned 2026-11-20. 0.2.14 and 0.2.15 never got a `-cuda13` image and 0.2.16 was skipped, so the pin before it is 0.2.13 (digest `sha256:b6e18765…`, engine 0.1.59). Both images are local: everything below was read on 0.2.17 (re-read 2026-11-20) and diffed against 0.2.13. It is upstream's own image, published by Strata's `docker-images` workflow since 0.2.1. All 7 slugs are `🐣 Incubating`, and all serve Qwen3.8-Flash-Next (`models/qwen3.8-flash-next/strata/`). The profile's `min_sm` is 7.5 (`SM floor`).

**No compose puts a flag in `command:`.** Settings reach Strata as env, by three paths:
1. Our `strata-entrypoint.sh` sets the image entrypoint's own env (`MODEL`, `FAMILY`, `MODEL_PATH`, `CONTEXT`, `KV`, `GPUS`, `VISION`, `API_KEY`, `PRESET`). That entrypoint turns them into `strata launch --no-tui …` flags `[strata-entry 0.2.17]`.
2. `strata_config.py apply` writes run-config keys and engine args (`--spec`, `--rope-*`) into `/data/install/strata-<tag>.json`. `forget` removes ours from the launch config's `user_args` before every start (Strata 0.2.11+ carries `user_args` over from one launch to the next). It never writes the run config's `env` `[strata-launcher]`.
3. The `STRATA_*` switches reach the engine process as env. `serve/server.py` `child_env()` starts from the container's environment, lays the run config's `env` over it, and then gives `STRATA_ROUTE_TAIL_SKIP` the value `0` when neither names it `[strata-src 12d491da]`.

Upstream status comes from the engine's and the launcher's `--help` in the pinned image, the image's own Python sources, and the engine binaries' own strings.

**Headline findings for this engine**

- **Nothing injects, and almost nothing needs to.**
  - Strata has no `inject:` key: `_ENGINE_TYPE_CONCURRENCY_ENV` and `_ENGINE_TYPE_MEM_UTIL_ENV` map only `vllm` and `sglang`. Re-checked at `baef2cb6`: `resolve_variant_pin` gives the 7 slugs no non-image env on 2×3090, 2×5090, 2×A6000, 2×H100 or DGX Spark (positive control: 53 slugs get one on 2×5090).
  - No compose passes `PARALLEL`, so concurrency is Strata's own: "fit to your graphics card; one request at a time with the DFlash2 drafter" `[strata-launch-help 0.2.17]` (0.2.13's text ended at "fit to your graphics card").
  - Setup sizes the VRAM-resident KV (`--kv-resident`), the expert cache and the conversation cache (from free RAM), and places the n-gram table (by disk type and free RAM).
  - The one hardware-coupled value a compose ships is the card list: container-relative, so host-card choice still works, but its count is a per-topology default no launcher exports — `⚠` by the same rule as SGLang's literal `--tp-size` (§4).
  - This is the most self-sizing engine in the catalog.
- **The engine decides at start which drafter runs and which engine binary is used. The compose does not.**
  - ⚠️ Correction to the 0.2.13 re-read, which said all 7 headers name the MTP: 5 headers say `Drafter: MTP (--spec 4, min-p 0.5)`, and the two Unsloth ones say DFlash2.
  - Auto (the default; no compose overrides it) is "DFlash2 wherever the engine has it and the model is Qwen3.8-Flash-Next, Gyro or an Unsloth quantization, the MTP everywhere else" (`drafting.py`). `SUPPORTED_FAMILIES` is `qwen`, `gyro`, `unsloth`, and `_EXCLUDED_NAME` keeps any `swift` file out whatever its header says, so **Auto gives the 3 Swift slugs the MTP**, as their headers say `[strata-src 12d491da]`.
  - `SET_DEFAULTS` names `("cuda","iq3_s")` and `("cuda","unsloth-ud-iq4_xs-v31")` as DFlash2 sets. New since 0.2.13: `CLASS_DEFAULTS` puts IQ3_S back on the MTP on a "streaming" CUDA card (fewer than 3000 engine expert slots, the 12 GB class) `[strata-src 12d491da]`.
  - "Auto uses the MTP when several requests run at once" `[strata-launch-help 0.2.17]`.
  - On exactly 2 cards of cc 8.6 or 12.0, Auto may instead start the all-cards split engine (`strata-split-2-sm86`) with the MTP: `AUTO_SPLIT_BEATS_DFLASH2 = (86, 120)`. It declines when the run config has `user_args` or replicas, when the KV is not int8, above `TESTED_CONTEXT` (131072), on any card count but 2 (`TESTED_CARD_COUNTS`, so never on the multi4 slug), or when `decide()` rejects the model file's tensor types, the VRAM or the RAM `[strata-src 12d491da]`.
  - Observed: on 0.2.13, `ud-iq4xs-dual` ran `--draft dflash2 --spec 6 --draft-conf 0.35`. On 0.2.17, `ud-iq3xxs-dual` ran the normal `strata` binary on a layer split with the DFlash2 drafter ("DFlash2 drafter on … layer split: the drafter is on CUDA1") `[strata-rig 2026-11-20]`.
  - **UNVERIFIED on 0.2.17: the ISTA IQ3_S slugs' drafter, request concurrency and split-engine use, and whether the Swift dual lands on the split engine.** No boot of them was read. Their headers' "up to 3 requests" predates these rules.
  - Whether a set `SPEC_N` ends up in the `user_args` the split rule reads, after `forget` runs, was not traced.
- **What `SPEC_N` controls depends on the drafter.**
  - `--spec` is "the MTP drafter's verify window" `[strata-help 0.2.17]`.
  - Under DFlash2, Auto writes `--spec 6`, or `--spec 8` on Strix Halo and on Blackwell cc 12.0 with ≥ 48 GB `[strata-src 12d491da]`.
  - `--draft-cap` defaults by card class: "8 on Strix Halo / iGPU and on Blackwell cc 12.0 with 48 GB or more, 6 on other Blackwell, --spec as given elsewhere" `[strata-help 0.2.17]`, unchanged from 0.2.13 (0.2.12 gave "5 on Strix Halo / iGPU, --spec elsewhere").
  - Left unset, the window is `engine-auto`. A set `SPEC_N` fixes one window for every card.
- **The nine pass-through `STRATA_*` switches are in the `strata` engine binary, and none of them is in `strata-split-2-sm86`** `[strata-bin 0.1.63]`.
  - Controls: `STRATA_DEFAULTS` is in both binaries, and the made-up name is in neither. 0.1.59's split binary had none of them either `[strata-bin 0.1.59]`, so this is not new at this pin.
  - So when Auto starts the all-cards split engine (above), the nine have no reader. Which of our slugs that happens to is UNVERIFIED; the one 0.2.17 dual boot read ran the layer split.
  - Three now have semantics from the engine's own log strings: `STRATA_ROUTE_TAIL_SKIP`, `STRATA_SPLIT_CPU_SHARE` and `STRATA_PLE_AUTO_RAM`. The other six still come from upstream's release notes as quoted in [`engines/STRATA.md`](engines/STRATA.md), not from source, and are tagged `UNVERIFIED` below.
- **⚠ `STRATA_ROUTE_TAIL_SKIP` cannot turn the skip on for the 2 Unsloth slugs.**
  - Setup writes `STRATA_ROUTE_TAIL_SKIP: "0"` into the `env` of every Unsloth run config (0.2.17's `unsloth_default_env`). 0.2.13's wrote it wherever every expert is in RAM rather than in its RAM-budget mode `[strata-src 12d491da]` `[strata-src d2f09647]`.
  - On the reference rig both Unsloth run configs carry it (written by 0.2.11's setup on 2026-11-07); the IQ3_S and Swift ones have no `env` `[strata-rig 2026-11-20]`.
  - `child_env()` lays that `env` over the container's environment, so a compose `STRATA_ROUTE_TAIL_SKIP=7` becomes `0` on those two slugs. Read from source; no boot with `=7` was made.
  - It also means those two already ran with the skip off on 0.2.13. The compose comment "0.2.13 to 0.2.16 ran it at rank 7 on CUDA" holds for the 5 IQ3_S and Swift slugs.
  - No other forwarded switch is in any of the rig's run configs' `env`, so for those the container's value reaches the engine.
- **Pin drift, 0.2.13 → 0.2.17 (both images local; both `--help`s diffed):**
  - The engine help only gained and reworded entries. New: `--conversation-cache-rewind-keep N` (default 0, park) and `--no-trim-stage-weights`. `--trim-stage-weights` is now the default on a layer split: each card loads only its own layers' dense weights, and the VRAM this frees goes to its expert cache. That reaches our 5 multi-card slugs whenever they run a layer split.
  - The launch help lost `--dsv41-package`, `--speed`, `--full-quality` and `--dspark`: DeepSeek left Strata, and we passed none of them. It gained `--draft-vocab`, `--listen` and `--access address`, and `--parallel`'s default now says DFlash2 answers one request at a time.
  - The image entrypoint reads two new settings, `DRAFTING` and `DRAFT_VOCAB` `[strata-entry 0.2.17]`. Our launcher sets neither, so Auto and Strata's own vocabulary pick apply.
  - The engine binary gained `STRATA_SPLIT_CPU_SHARE`. Two defaults that reach our slugs changed: the route tail skip is now off, and the two-card CPU prefill share is now on. The composes forward both since 2026-11-20 (#2114).
  - No flag or env we use was removed or renamed.

| Env (compose) | → becomes | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|---|
| `STRATA_START_GPUS` | image env `GPUS` → `strata launch --gpu LIST` `[strata-launcher]` `[strata-entry 0.2.17]`; a saved run config that names other cards is dropped and set up again | "graphics card numbers to use, e.g. 0 or 0,1 (default: the engine's choice)"; 0.2.17 adds that with more than one card the engine decides which is the Main card (`--gpu-roles` sets the roles instead) `[strata-launch-help 0.2.17]`; engine `--gpu LIST` sets `CUDA_VISIBLE_DEVICES` inside the engine `[strata-help 0.2.17]` | 7/7 — `${STRATA_START_GPUS:-0,1}` ×4, `:-0` ×2, `:-0,1,2,3` ×1 | Topology ⚠ — a per-topology default no launcher exports (same rule as SGLang `--tp-size`, §4). It is **container-relative**: `device_ids` maps the host cards from `ESTATE_GPUS` / `CUDA_VISIBLE_DEVICES`, so `0,1` stays right whichever two cards the launcher picks; only the count is fixed. The layer split itself is `engine-auto` (`layer_split: auto` in setup's run config). |
| `CONTEXT` | `strata launch --context N` → setup's `--max-context` (a change re-runs setup) `[strata-entry 0.2.17]`; strata_config.py only checks it (≥ 4096) | "32k, 64k, 128k, 256k or a number of tokens" `[strata-launch-help 0.2.17]`; `--max-context N` "KV/state capacity (default 4096)" `[strata-help 0.2.17]` | 7/7 — `${CONTEXT:-131072}` ×7 | `bare`, but the limit is host RAM, not VRAM. Setup sizes the VRAM-resident KV (32K cells on the reference rig) and streams the rest to RAM. That RAM is gated by `launcher-bash:preflight_cpu_offload_ram` (the `CPU-Offload-Host-RAM-GB` header). A CONTEXT above 131072 also rules out the auto split engine. |
| `KV` | `strata launch --kv FORMAT` `[strata-entry 0.2.17]` | `int8` (default) · `q4_0` · `k8v4` · `fp16` `[strata-launch-help 0.2.17]`; the split engine "keeps int8" `[strata-src 12d491da]` | 7/7 — `${KV:-}` ×7 (unset → int8) | `—`. A non-int8 KV also takes the auto split engine off the table. |
| *(not passed)* `PARALLEL` | — | "1 to 8 (default: fit to your graphics card; one request at a time with the DFlash2 drafter)" `[strata-launch-help 0.2.17]`; the image entrypoint would read it `[strata-entry 0.2.17]` | 0/7 | `engine-auto`; no `inject:` for this engine type |
| `STRATA_PLE_AUTO_RAM` | env to the engine, as set | the engine's own line: "ple-io ram chosen by the residency policy (…; STRATA_PLE_AUTO_RAM=0 keeps the direct reads, --ple-io direct forces them)" `[strata-bin 0.1.63]` | 7/7 — pass-through, unset | Host RAM. `launcher-bash:preflight_cpu_offload_ram` drops the `Strata-PLE-Table-GB` share on an SSD below `Strata-PLE-Setup-Pins-From-GiB`. Setup's own `--ple-io ram` is chosen by disk type (`engine-auto`), and `=0` does not override it. |
| `STRATA_PCIE_AUTO_SPLIT`, `STRATA_PCIE_K_RULE`, `STRATA_PCIE_SHARE_MAX` | env to the engine | names present `[strata-bin 0.1.63]`; the 0.2.11 CPU-vs-PCIe share of missed-expert work, which follows the CPU pool's measured speed. **UNVERIFIED** in source | 7/7 each — pass-through, unset | `engine-auto` (measured at runtime) |
| `STRATA_GDN_TC`, `STRATA_GDN_CONVL2` | env to the engine | names present `[strata-bin 0.1.63]`; the tensor-core GDN scan and the fused conv pass, on by default on Linux + NVIDIA from 0.2.9, with a fallback where the card doesn't qualify. **UNVERIFIED** in source | 7/7 each — pass-through, unset | `engine-auto` (the engine's own card check) |
| `STRATA_ROUTE_TAIL_SKIP` | env to the engine; unset, `child_env()` sets `0` (`speed_switches.ENGINE_START_DEFAULTS`) `[strata-src 12d491da]`. On the 2 Unsloth slugs the run config's `env` `0` replaces any value (headline) | the engine's own lines: "STRATA_ROUTE_TAIL_SKIP not set: off (the default since 0.1.63)"; set to N, "a missed expert that every token of a verify window routes at rank N or lower is skipped (answers differ slightly)" `[strata-bin 0.1.63]`. Strata's switch writes `7` for on; 1–9 is the rank, and "Only 7 is measured" `[strata-src 12d491da]` | 7/7 — pass-through, unset (= off); forwarded since 2026-11-20 (#2114) | — . ⚠ inert on 2/7 (the Unsloth slugs) |
| `STRATA_SPLIT_CPU_SHARE` | env to the engine | new in the 0.1.63 binary (absent from 0.1.59's). The engine's own line: "layer split of two cards: the CPU prefill share is armed by default, as on one card (STRATA_SPLIT_CPU_SHARE=0 turns it off)" `[strata-bin 0.1.63]`. The 0.2.17 `ud-iq3xxs-dual` boot logged it; with `=0` the line was gone `[strata-rig 2026-11-20]` | 7/7 — pass-through, unset (= armed on a two-card layer split); forwarded since 2026-11-20 (#2114) | — . What it does on the 2 single slugs and the 4-card multi4 is **UNVERIFIED**: the line names only a two-card layer split |
| `SPEC_N` | `--spec N` (1–8) in the run config's `args` `[strata-launcher]` | MTP verify window, "setup writes 4" `[strata-help 0.2.17]`; DFlash2 Auto writes 6, or 8 on Strix Halo and big Blackwell `[strata-src 12d491da]` | 7/7 — pass-through, unset | `engine-auto` while unset; a set value is the same on every card |
| *(not passed)* drafter, split engine, draft vocabulary | — | `--drafting auto\|mtp\|dflash2\|off`, `--split-engine auto\|on\|off`, `--draft-vocab cjk\|en\|cyrillic\|fr\|latin` ("Without it Strata picks from this computer's languages") `[strata-launch-help 0.2.17]`; the image entrypoint reads `DRAFTING` and `DRAFT_VOCAB` from 0.2.17 `[strata-entry 0.2.17]`; the Auto rules in the headline `[strata-src 12d491da]` | 0/7 — Auto ×7 | `engine-auto`, keyed on the model family, compute capability × card count and (from 0.2.17) the card's expert-slot class |
| `ROPE_SCALING`, `ROPE_SCALE` | `--rope-scaling T`, `--rope-scale F` in `args` `[strata-launcher]`; unset gives setup's choice (none ≤ 262144, yarn with factor ctx/262144 past it) | `none`/`linear`/`yarn`, "Fixed at startup"; the factor defaults to the model file's, else 1 `[strata-help 0.2.17]` | 7/7 each — pass-through, unset | — |
| family + set (launcher **argv**, not env) | image env `MODEL`/`FAMILY` → `--model`/`--family`; `MODEL_PATH=/gguf` → `--model-path`; `PRESET` → `--preset`/`--save-preset club-<…>`. The image adds `--access lan --port 8080 --data-dir /data/Strata-data` `[strata-entry 0.2.17]` | `[strata-launch-help 0.2.17]` | 7/7 — `LIT`: `qwen IQ3_S` ×2 (the default when the args are absent), `swift IQ3_S` ×3, `unsloth UD-IQ4_XS-v31` ×1, `unsloth UD-IQ3_XXS` ×1 | — |
| `VISION` | `LIT off` in our launcher → `--vision off` | default "on when the model can read pictures and the engine has the picture reader"; 'on' picks the reader by graphics memory (the CPU reader under 24 GB) `[strata-launch-help 0.2.17]` | 7/7 — `LIT off` | — (the VRAM-keyed reader choice never runs) |
| `API_KEY` | `--api-key-env STRATA_LAUNCH_KEY` `[strata-entry 0.2.17]`; strata_config.py checks only the length | `[strata-launch-help 0.2.17]` | 7/7 — `${STRATA_API_KEY:-club-3090-strata-local-key}` | — |
| `REASONING_EFFORT`, `ENABLE_THINKING`, `THINKING_BUDGET`, `SERVED_NAME` | run-config keys `reasoning_effort` (`ENABLE_THINKING=false` → `none`), `reasoning_budget_tokens`, `aliases` `[strata-launcher]` | read in `server.py` `_main()` (Niko1221/Strata#1641, Niko1221/Strata#123, Niko1221/Strata#297) `[strata-src 12d491da]` | 7/7 each — pass-through, unset | — |
| `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `PRESENCE_PENALTY`, `REPEAT_PENALTY` | `sampling.{temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty}` `[strata-launcher]` | `sampling_defaults_from_config()`, in `serve/server_parts/http_api.py` since 0.2.17 split `server.py` (`_main()` still calls it): defaults for fields a request leaves out; a bad value refuses to start `[strata-src 12d491da]` | 7/7 each — pass-through, unset | — |
| `STRATA_PENALTY_PROMPT` | env to the engine | name present `[strata-bin 0.1.63]`; the 0.2.10 penalty-window change. **UNVERIFIED** in source | 7/7 — pass-through, unset | — |

---


## 10. Project Maya — 34 registered slugs, engine id `strata-maya`

Pin `ghcr.io/noonghunna/strata-maya-club3090` at digest `sha256:d129c0c1…` = Maya **v1.1.70**, upstream mw00/project-maya at `b6632a0b`, recipe `e9a16343`, archs `75;80;86;89;120` (pinned 2026-11-20). Everything below was read on this pin (re-read 2026-11-20) and diffed against v1.1.67 (digest `sha256:98da6b3f…`, upstream `56b5216a`), the pin of the previous re-read; v1.1.68 and v1.1.69 were pinned in between and not diffed on their own. We build the image from each upstream release (an image-provenance exception, `docs/ADDING_ENGINES.md`). All 34 slugs are `🐣 Incubating`, four engines behind one image:
- **GLM-5.3-Flash on `maya`:** 11 (single 3 · dual 4 · multi4 4).
- **MiniMax-M3.1-Flash on `maya-mm3`:** 3 (single 1 · dual 1 · multi4 1).
- **DeepSeek-V4.1-Flash-1017 on `maya-ds4`:** 4 (dual 3 · multi4 1).
- **Hy4-Flash on `maya-hy4`:** 16 (single 4 · dual 4 · multi3 4 · multi4 4), the sizes Maya-XS / S / M / L. The engine is in the image from v1.1.68.

The composes pass **no argv**. Each sets env and runs our `launcher/<x>-entrypoint.sh <quant>`. That script calls `<x>_prepare.py`, which validates the settings through `<x>_settings.py` and rewrites `/data/<quant>/maya{,-mm3,-ds4,-hy4}.json` at every boot. The JSON holds `exe`, `args`, `gpu`, `sampling`, the run-config keys the server reads (`reasoning_effort`, `model_name`, …) and, for Hy4, an `env` block. The entrypoint then execs `serve/server.py --engine strata` (GLM, mm3, ds4) or `--engine maya` (Hy4) with `--config … --host 0.0.0.0 --port 8080`. The server spawns `build/<engine> --serve <args>` with its own environment, so pass-through env reaches the engine. It adds `CUDA_VISIBLE_DEVICES=<gpu>`, the config's `env`, and `--layer-split auto` when there is more than one GPU (`serve/server.py` `engine_args`/`child_env`, `[maya-src b6632a0b]`). Help sources: the server's `--help` and the `maya`, `maya-mm3` and `maya-ds4` binaries' `--help`; `maya-hy4` has none, so its parser was read (`[hy4-src b6632a0b]`); source read inside the image wherever the help is silent.

**Headline findings for this engine**

- **`MAYA_PCIE_SHARE` / `MAYA_CPU_LANE` (GLM, 11) reach the engine by its own names since 2026-11-17.**
  - `maya-entrypoint.sh` exports `MAYA_GLM_PCIE_SHARE` / `MAYA_GLM_CPU_LANE`, the names `src/core/glm_fast_path.cu` reads `[maya-launcher]` `[maya-src b6632a0b]`.
  - Until 2026-11-17 it exported `STRATA_GLM_PCIE_SHARE` / `STRATA_GLM_CPU_LANE`, which no engine since v1.1.61 reads by name. They arrived only because `server.py` imports `tools/maya_names.py`, whose `alias_env` copies every `STRATA_X` to `MAYA_X` (still there at v1.1.70) `[maya-src b6632a0b]`.
- **`--engine strata` (18/34: GLM, mm3, ds4) is a legacy spelling.** `server.py` keeps it as "the old name, kept for older scripts" and rewrites it to `maya` `[maya-src b6632a0b]`. The help lists `{mock,maya,strata}`. The Hy4 launcher passes `--engine maya` (16/34).
- **Upstream's help contradicts its own source on GLM `--kv`.**
  - `build/maya --help` says "default fp16 until gate G-C accepts int8".
  - `src/program/generate.cpp:5013` defaults the GLM path to int8 when `--kv` is unset (v1.1.55, `MAYA_GLM_KV_INT8`).
  - Our launcher's "unset = int8" matches the source, not the help.
- **⚠ `MAYA_GPUS` is a per-topology literal.**
  - Defaults: `0` ×8, `0,1` ×12, `0,1,2` ×4, `0,1,2,3` ×10.
  - It becomes the engine's `CUDA_VISIBLE_DEVICES` inside a container whose device mask is the separate `device_ids: ['${ESTATE_GPUS:-${CUDA_VISIBLE_DEVICES:-…}}']`.
  - Nothing derives one from the other, and no launcher sets `MAYA_GPUS` (no hit in `scripts/` outside tests). The count is matched by hand.
- **Placement is `engine-auto`; the hard-coded values are the KV/context ones.**
  - Adapts by itself: the layer split (`--layer-split auto`, a placement search over free VRAM), the VRAM expert pool (GLM `--expert-cache auto` = free VRAM less a 700 MiB reserve; never passed), the RAM tier (sized from free host memory at start), and maya-hy4's tensor-parallel choice (below).
  - Hard-coded: `CONTEXT`, `--kv-streaming on` (GLM, mm3), `MAYA_TP=1` (ds4), and Hy4's whole cache in VRAM.
- **No injector reaches it.** There is no `envelopes.yml` row and no `inject:` key in these composes. Re-checked at `baef2cb6`: `resolve_variant_pin` gives the 34 slugs no non-image env on 2×3090, 2×5090, 2×A6000, 2×H100 or DGX Spark. `resolve_offload_threads` does not fire, because the composes have no `=CPU` / `--cpu-moe` token.
  - `THREADS` defaults to nproc/2 of the container's CPU affinity, computed in-container by `mm3_settings.py` / `ds4_settings.py` / `hy4_settings.py` `threads()`.
  - GLM's CPU lane is the engine's own choice: one thread per physical core.
- **maya-hy4 (16 slugs, new since the last re-read).**
  - **It has no `--help`.** It answers "unknown argument --help" and exits 2. Its flags are the 42 `a == "-…"` comparisons in `src/hy4/hy4_main.cpp`, and any other argument stops it the same way, so a flag dropped upstream fails the boot instead of being ignored.
  - Every flag our launcher and the server write is in that parser: `--pack`, `--max-context`, `--cpu-threads`, `--mtp`, `--kv`, `--vision`, `--serve` and `--layer-split auto` `[hy4-src b6632a0b]`.
  - Not passed, so the engine's own default applies:
    - `--tp`: auto;
    - `--reserve-gb`: 0.75;
    - `--vram-experts`: 1.0;
    - `--chunk`: auto;
    - `--kv-streaming`: off;
    - `--profile`: `hy4_prepare.py` adds it only when `data/expert-profile-hy4.bin` exists, and v1.1.70 ships none.
  - There is no `--parallel`: one request at a time.
  - **`--tp auto` decides the layout** (v1.1.70, `src/hy4/hy4_model.cu`), in this order:
    1. tensor parallel over all N cards where every routed expert fits;
    2. one card fewer, down to two;
    3. for a group of three or more, tensor parallel with the CPU lane;
    4. otherwise the layer split.

    The Hy4 composes forward the three switches that steer it (next table) `[hy4-src b6632a0b]`.
  - ⚠ **The whole latent cache sits in VRAM.** `kv_streaming` defaults to 0 and the launcher does not set it, so `CONTEXT=262144` (all 16 slugs) competes with the experts for VRAM on every card.
  - **`THREADS` unset still writes `--cpu-threads`**, at half the container's CPUs. The engine's own default, one worker per physical core but the first (`hy4_model.hpp`), never applies.
  - `src/hy4/*` reads 93 `HY4_*` names, 87 of them through a direct `getenv`. The composes pass 4, and the engine reads all 4: `HY4_EG_POPULATE` ×16, `HY4_TP_LANE` ×12, `HY4_TP_DOWN` ×8, `HY4_TP_EXPERTS` ×8. The server sets one more itself, `HY4_Q4X_FILE` (where the CPU lane's verdict on its 4-bit expert copies is kept) `[maya-src b6632a0b]`.
- **Our own docs and comments that disagree with the code:**
  - `maya-mm3` is configured with `--profile data/expert-profile-mm3.bin`. Its usage line documents only `--expert-profile`, but the parser takes `--profile` (`src/mm3/mm3_main.cpp:1734`, the shipped profile; `--expert-profile` is the learned one), and Maya's own `maya.py` writes it the same way.
  - `mm3-entrypoint.sh`'s header says DSpark is off by default, but `mm3_settings.py` writes `--dspark on` when `DSPARK` is unset, as the compose header says.
- **Pin drift v1.1.61 → v1.1.65:** both images are local. `server.py --help` is byte-identical, and the three engines' `--help` flag sets are identical.
- **Pin drift v1.1.65 → v1.1.67 (re-read 2026-11-19; v1.1.65, v1.1.66 and v1.1.67 images all local):**
  - `server.py --help` and all three engines' `--help` are byte-identical across the three releases.
  - No env name the v1.1.65 sources read is gone from v1.1.67's.
  - 16 env reads are new. Six are the idle-sleep and diagnostic settings first forwarded on 2026-11-18 (rows below).
  - The other ten are not set by any compose of ours:
    - `MAYA_PAR_MAX` (server): caps the `--parallel N` asked of the engine, and the requests beyond it wait in the server's queue. Our launchers write no `--parallel`, so it has nothing to cap.
    - `MAYA_GLM_HELPER_WARM`, `MAYA_GLM_HELPER_WIN`, `MAYA_GLM_HELPER_WIN_CHECK`, `MAYA_GLM_LINK_MULTI`, `MAYA_GLM_MV_EXACT_COLS`, `MAYA_GLM_NO_POOL`, `MAYA_GLM_WIN_HC_ROWS`, `MAYA_GLM_WIN_ONCE`, `MAYA_GLM_WIN_WAIT_ROWS`: GLM kernel and expert-helper A/B switches and bit-for-bit checks, going by the comments beside each read (`src/core/{glm_fast_path.cu,glm_window.cu,remote_experts.cpp}`, `src/kernels/cuda/glm_fast.cu`). Their defaults were not traced.
- **Pin drift v1.1.67 → v1.1.70 (re-read 2026-11-20; both images local):**
  - `server.py --help` changed in one sentence: the vision readers now include Hy4-Flash's (`serve/hy4_vision.py`). The `maya`, `maya-mm3` and `maya-ds4` `--help` are byte-identical.
  - New engine: `maya-hy4` (above).
  - No env name the v1.1.67 sources read is gone from v1.1.70's.
  - 101 env reads are new, counted as name literals passed to a reader (`getenv`, `ds4_env`, `os.environ.get` and the engines' `env_*` helpers) in `src`, `include`, `serve` and `tools`, tests left out:
    - 93 are `HY4_*` names, the new engine's own.
    - 8 are not, and no compose sets them: `DS4_ARENA_THP` (the DeepSeek RAM arena's transparent huge pages, read through `ds4_env`, so `MAYA_DS4_ARENA_THP` first), `DS4_IQ2V`, `DS4_IQ3V`, `DS4_IQ4V`, `DS4_IQ_Q4X`, `DS4_Q4XV` (DeepSeek's CPU-lane IQ kernels, `src/ds4/ds4_iq_cpu.cpp`), `MAYA_IQ_MV8` (`src/ds4/ds4_iq.cu`) and `MAYA_GLM_PIN_MEASURE_FIRST` (`src/core/glm_fast_path.cu`). Their defaults were not traced.

| Env (compose) | → becomes | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|---|
| `MAYA_GPUS` | config `"gpu"` → engine `CUDA_VISIBLE_DEVICES` (+ `CUDA_DEVICE_ORDER=PCI_BUS_ID`), and `--layer-split auto` when >1 card `[maya-launcher]` `[maya-src b6632a0b]` | `--gpu`: "the GPU to run on … or several for a layer split", also `"gpu"` in the config `[maya-help v1.1.70]`; maya-hy4 takes every card it sees (no `--gpus` passed) `[hy4-src b6632a0b]` | 34/34 — `${MAYA_GPUS:-0}` ×8, `:-0,1` ×12, `:-0,1,2` ×4 (Hy4 multi3), `:-0,1,2,3` ×10 | Topology ⚠ — must equal the docker device mask's count; nothing ties them (headline) |
| `MAYA_TP` | `--tp N` (ds4) `[maya-launcher]` | `--tp N` `[maya-bin v1.1.70]`; on two GPUs 1 = layer split with whole experts, 2 = tensor parallel (our compose comment, hand-measured on 3090s: decode +9/+5 %, 6K prompt −35 %) | ds4 4/4 — `${MAYA_TP:-1}`. Hy4 passes no `--tp` (auto: next row) | Topology ⚠ — picked on PCIe 3090s; nothing detects the interconnect |
| `HY4_TP_LANE`, `HY4_TP_DOWN`, `HY4_TP_EXPERTS` | engine env (pass-through) | `--tp auto`'s switches (`src/hy4/hy4_model.cu`) `[hy4-src b6632a0b]`: `HY4_TP_DOWN=0` tries only the full group (else one card fewer, down to two: nothing to try on two cards); `HY4_TP_LANE` set and nonzero takes tensor parallel with the CPU lane on any group, `0` on none, unset only on a group of three or more; `HY4_TP_EXPERTS=unsplit` / `slices` fixes how a group shares an expert (unset: whole experts where slicing would pad one by 10 % or more), and `slices` also rules out the CPU-lane plan | Hy4: `HY4_TP_LANE` 12 (dual, multi3, multi4), `HY4_TP_DOWN` and `HY4_TP_EXPERTS` 8 (multi3, multi4) — pass-through, unset; forwarded since 2026-11-20 (#2129) | `engine-auto` (a fit over free VRAM); a set value is the same on every card |
| `CONTEXT` | `--max-context N` (all four); `<x>_settings.py` refuses < 4096 `[maya-launcher]` | "KV/state capacity (default 4096)" `[maya-bin v1.1.70]`; maya-hy4's own default is 32768 (`hy4_model.hpp`) `[hy4-src b6632a0b]` | 34/34 — GLM `${CONTEXT:-131072}` ×10, `:-262144` ×1 (dual maya-s24; its multi4 sibling ships 131072); mm3 `:-204800` ×3; ds4 `:-262144` ×2 (ista-iq2-xxs, maya-xs), `:-65536` ×2 (ud-iq3-s dual and multi4); Hy4 `:-262144` ×16 | `bare 24 GB` ⚠ — KV competes with the VRAM expert pool and, streamed, with pinned host RAM (mm3: 63,360 B/token at INT8, ~13 GB at 204800 — entrypoint comment). Hy4 keeps all of it in VRAM (next row) |
| — (`LIT`) | `--kv-streaming on` (GLM, mm3) `[maya-launcher]` | "on keeps `--kv-resident`'s N cells in VRAM, 32768 when it gives none; off keeps all of the KV in VRAM" `[maya-bin v1.1.70]`; maya-hy4 parses `--kv-streaming on\|off\|auto` too, default off (`kv_streaming = 0`) `[hy4-src b6632a0b]` | GLM 11 + mm3 3 — `LIT on`; `--kv-resident` never passed. Hy4 16 — not passed (off) | `bare 24 GB` ⚠ — a 32K-position window whatever the card; on Hy4, the whole context in VRAM whatever the card |
| — (not passed) | expert pool in VRAM: GLM `--expert-cache auto` / `--vram-reserve-mib` (default 700); mm3 `--cache-gb` / `--reserve-gb`; ds4 `--reserve-mib` / `--vram-experts`; Hy4 `--reserve-gb` (0.75) / `--vram-experts` (1.0) / `--chunk` (auto) | `[maya-bin v1.1.70]` `[hy4-src b6632a0b]` | 0/34 — engine defaults | `engine-auto` — "takes the free VRAM less the reserve" (GLM help) |
| — (`LIT`) | `--prefill auto` (GLM) | "the largest chunk up to 8192 whose buffers the expert cache can lend" `[maya-bin v1.1.70]` | GLM 11 | `engine-auto` |
| — (`LIT`) | `--expert-profile data/expert-profile-glm.bin` (GLM) · `--profile data/expert-profile-mm3.bin` (mm3) | GLM: "pre-load the VRAM tier from a profile.bin" `[maya-bin v1.1.70]`; mm3 `--profile`: parsed, not in the usage line (`mm3_main.cpp`) `[maya-src b6632a0b]`; maya-hy4 parses `--profile`, but v1.1.70 ships no `data/expert-profile-hy4.bin`, so `hy4_prepare.py` leaves it out `[hy4-src b6632a0b]` | 11 · 3 · Hy4 0/16 | — (which experts first; how many is the pool's fit) |
| `MAYA_PCIE_SHARE` · `MAYA_CPU_LANE` | exported as `MAYA_GLM_PCIE_SHARE` / `MAYA_GLM_CPU_LANE` when non-empty `[maya-launcher]` (until 2026-11-17 as `STRATA_GLM_*`, which reached the engine only through `maya_names.alias_env`) | share of a route's RAM-tier experts sent over PCIe (0..1); CPU-lane threads (0 = off), default one per physical core; unset = timed at start (`lane_plan`) `[maya-src b6632a0b]` | GLM 11/11 — `${…:-}` (unset) | `engine-auto` (measured per machine at start) |
| `THREADS` | `--cpu-threads N`; unset = nproc/2 of the container's affinity (`threads()`), written either way `[maya-launcher]` | `--cpu-threads N` `[maya-bin v1.1.70]`; maya-hy4's own default (one worker per physical core but the first) is never reached `[hy4-src b6632a0b]` | mm3 3 + ds4 4 + Hy4 16 — pass-through, unset | CPU count, computed in-container (same floor as `launcher-bash:resolve_offload_threads`, which does not fire here) |
| `MMAP_EXPERTS` | `--mmap-experts on\|off\|auto` (ds4) `[maya-launcher]` | `--mmap-experts on\|off\|auto` `[maya-bin v1.1.70]`; auto = copy into RAM when they all fit (compose comment) | ds4 4 — pass-through, unset (= engine's auto) | `engine-auto` (host RAM) |
| `DS4_RAM_MARGIN_GB` | engine env (pass-through) | RAM left to the rest, default 8 GiB, 0–4096 (`src/ds4/ds4_model.cu`) `[maya-src b6632a0b]` | ds4 4 — pass-through, unset | Host RAM — `bare` |
| `DS4_FLOW_COPY_DMA`, `DS4_FLOW_TP`, `DS4_STREAM_TIE`, `DS4_TP_SHARES`, `DS4_TP_UNEVEN_HEADS`, `MAYA_DS4_TP_GGUF_UNEVEN` | engine env (pass-through) | all read by `maya-ds4` (`ds4_model.cu`; names in the binary's strings) `[maya-src b6632a0b]`; semantics per the compose comments, not re-derived. `DS4_TP_SHARES` unset = shares from each member's free VRAM | ds4 4 each — pass-through, unset | `engine-auto` (TP shares); the rest — |
| `MAYA_GLM_P2P` | engine env (pass-through) | `is_p2p_enabled()`: nonzero = on (`src/core/glm_window.cu`) `[maya-src b6632a0b]`; "peer-granted pairs only, else host memory" is the compose comment's, **UNVERIFIED** beyond the getenv | GLM 8 (dual + multi4) — pass-through, unset = off | — (opt-in; not `boot-detect`: nothing probes the P2P grant for it) |
| `KV` | GLM `--kv int8\|fp16`; mm3 `--kv-cache int8\|bf16`; Hy4 `--kv auto\|f16\|q8` (only when set); ds4 none (latent cache fp16) `[maya-launcher]` | GLM: help says default fp16, **source says int8** (headline); mm3: "bf16 int8 fp8 fp4; default int8" `[maya-bin v1.1.70]`; Hy4: `auto` (the default) is F16 rows, 8-bit rows where F16 does not fit (`hy4_model.hpp`) `[hy4-src b6632a0b]` | GLM 11 + mm3 3 + Hy4 16 — pass-through, unset = the engine's default (int8 · int8 · auto) | `SM floor` in reality ⚠ — the launcher refuses mm3 `fp8` on **every** card ("not validated on these cards"), Ada/Hopper included. Hy4's `auto` is `engine-auto` |
| `DSPARK` / — | mm3 `--dspark on\|off` (unset = on); ds4 `--dspark` `LIT` per quant `[maya-launcher]` | `--dspark on\|off` (ds4 also `none`) `[maya-bin v1.1.70]` | mm3 3 — pass-through → `on` ×3; ds4 `LIT on` ×2 (ista-iq2-xxs, maya-xs), `off` ×2 (ud-iq3-s dual and multi4) | — (its 2.3 GB comes out of the expert pool) |
| `DRAFTING` (Hy4) | `--mtp` when unset or `on`; `off` leaves it out `[maya-launcher]` | `--mtp`: "load the MTP block (the GGUF's nextn layer, its BF16 rounded to Q8_0) for drafts"; without it no drafter is loaded (`--dfly2`, the other one, is not passed) `[hy4-src b6632a0b]` | Hy4 16 — pass-through, unset = on | — (about 4 GB of VRAM beside the model, per our compose header) |
| — (`LIT`) / `VISION` | `--vision off` (mm3, ds4); Hy4 `--vision off` only when `VISION=off` `[maya-launcher]` | `--vision on\|off` `[maya-bin v1.1.70]`; maya-hy4: on by default when the pack has the tower `[hy4-src b6632a0b]` | mm3 + ds4 7 — `LIT off`; Hy4 16 — pass-through, unset = on | — |
| `HY4_EG_POPULATE` | config `env` (`hy4_settings.py`) → the server's `child_env` → engine env `[maya-launcher]` | nonzero: the n-gram table's pages mapped into the process in the background after the load; unset or `0`: each looked-up row's page read from the disk at first use, and its "INFO Engram" line says which (`src/hy4/hy4_model.cu`) `[hy4-src b6632a0b]` | Hy4 16 — pass-through, unset = off | Host RAM / disk — the 63.4 GB table is not counted in the `CPU-Offload-Host-RAM-GB` header |
| `REASONING_EFFORT`, `ENABLE_THINKING`, `THINKING_BUDGET`, `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `PRESENCE_PENALTY`, `REPEAT_PENALTY`, `SERVED_NAME`, `LOOP_GUARD` | run-config keys `reasoning_effort`, `thinking_budget`, `sampling.*`, `model_name`, `loop_guard`; club defaults per engine: effort GLM `max` / mm3 `auto` / ds4 `high` / Hy4 `high`, sampling 1.0/0.95 (GLM, mm3), 1.0/1.0 (ds4) or 0.9/1.0 (Hy4) `[maya-launcher]` | read by `serve/server.py` `[maya-reads]`; not CLI flags. Hy4 has two levels, `frontend.py`'s `["none", "high"]`: our launcher folds every other level to `high`, as the server does `[maya-src b6632a0b]` | 34/34 each — pass-through, unset | — |
| `MAYA_DS4_SLOT_KEEP`, `MAYA_DS4_KEEP_THINKING` | engine / server env (pass-through) | read by `maya-ds4` (`ds4_main.cpp`) and `serve/frontend.py`; `KEEP_THINKING` by the server only `[maya-src b6632a0b]` | ds4 4 — unset = off | — |
| `MAYA_GLM_SERVICE_NAP_US`, `MAYA_GLM_SERVICE_SLEEP_MS` · `MAYA_DS4_SERVICE_NAP_US`, `MAYA_DS4_SERVICE_SLEEP_MS` | engine env (pass-through) | the idle service thread's two sleeps (v1.1.66, mw00/project-maya#951): GLM `src/core/glm_fast_path.cu`, DeepSeek `src/ds4/ds4_model.cu` (through `ds4_env`, which reads the `MAYA_DS4_` name first); `idle_sleep_setting()` takes a whole number, clamps it to 1..10000 µs / 1..1000 ms, and falls back to the default for an empty or non-numeric value (`include/strata/platform/idle_gate.hpp`) `[maya-src b6632a0b]` | GLM 11 · ds4 4 — pass-through, unset = 100 µs / 2 ms (1000 µs on Windows builds only) | — (idle CPU use against the first token after a pause) |
| `MAYA_GLM_PREFILL_CHECK`, `MAYA_GLM_TIER_VERIFY` | engine env (pass-through) | diagnostics, nonzero = on: every stage of a prompt layer checked, with the log line `MAYA_GLM_PREFILL_CHECK is on` (`src/core/glm_prefill.cu`, v1.1.66); the expert tables read back and checked after every token (`src/core/glm_fast_path.cu`, v1.1.67) `[maya-src b6632a0b]`; both only report | GLM 11 — pass-through, unset = off | — |
| `MAYA_MODEL_MANAGEMENT` | server env (pass-through) | `model_management_setting()`: `/models/load`, `/unload`, `/sse` on unless 0/off/false/no `[maya-src b6632a0b]` | 34/34 — unset = on | — |
| `SETUP_ONLY`, `REPACK`, `VERIFY` | read only by our launcher (pack build, SHA-256 check) | ours | 34/34 — `:-0`, `:-0`, `:-1` | — |
| header `CPU-Offload-Host-RAM-GB` | `launcher-bash:preflight_cpu_offload_ram` (keyed on the header, not on `is_cpu_offload_compose`) | — | 34/34 — GLM L/M/S/S24 131/88/68/68 (dual and multi4: multi4 reuses dual's as a documented, unmeasured upper bound), single 150/107/89; mm3 102/124 (dual / single; multi4 reuses the dual's 102); ds4 106/80/229 (ista-iq2-xxs / maya-xs / ud-iq3-s; the ud-iq3-s multi4 reuses its dual's 229), each multi4 figure a documented, unmeasured upper bound; Hy4 XS/S/M/L 52/64/76/100 on every topology, Maya's own `memory_gb` (`tools/hy4/download.json`), not measured, an upper bound | Host RAM — the engine sizes its RAM tier from free memory at start; the header only gates |

---


## 11. QEngine — 6 registered slugs, engine id `qengine`

Pin `ghcr.io/bpx03/qengine@sha256:5064093b…` = `1.3.6-vision`, source `444b2e49`, pinned 2026-11-15. It is the author's own image (bpx03), and `min_sm: 8.6` (the image description reads "sm_86; sm_89 and sm_120 built in, not tuned"). Slugs: 3 Qwen3.8-27B (`dual` ⚠️ Production w/ caveats, `multi3` and `multi4` 🧪) and 3 Qwen3.8-Flash-Next (`dual`, `multi3`, `multi4`, all 🐣). **How settings travel:** the compose `environment:` forwards 76 of QEngine's own `QE_*` / `FLASH_*` names. All but 5 (`QE_GPUS`, `QE_PORT`, `QE_MODEL`, `QE_MAX_SEQS`, `FLASH_MAX_SEQS`) are bare pass-through, so unset means QEngine's default. A `bash -c` entrypoint then maps our 10 knobs to `server/app.py` flags, appends them to `QE_EXTRA_ARGS` and execs the image's `/usr/local/bin/qengine-entrypoint`. That runs `server/run.sh`, which runs `python3 server/app.py --host … --port … --devices 0..n-1 --model … $QE_EXTRA_ARGS` `[qe-entry 1.3.6]`. Most `QE_*` / `FLASH_*` names are read by the C++ engine through `getenv`, not by `app.py` `[qe-bin 1.3.6]`. Flags were checked in `[qe-help 1.3.6]`, env semantics in `[qe-docs 444b2e49]`.

**Headline findings for this engine**

- **Every name we pass exists in the pinned build.** All 76 `QE_*`/`FLASH_*` names are found in the image `[qe-bin 1.3.6]`, and all 12 `app.py` flags that reach the server are in `[qe-help 1.3.6]`: 6 from our entrypoint (`--served-name --think --mtp-k --vision --disk-cache --disk-cache-gb`) and 6 from the image's launcher (`--host --port --metrics-host --metrics-port --model --devices`). **Two are undocumented upstream:** `FLASH_LEND_CHECK` and `FLASH_QSA_PIPE_CE` are in `libqflash.so`/`flash_run` but appear in neither README nor `advanced.md`. Their meaning comes only from our compose header.
- **The engine sizes itself; our composes hard-code almost nothing about the card.** Every VRAM, host-RAM and link setting is forwarded unset. At start QEngine does the following:
  - lowers the 27B's slot count to fit (`QE_MAX_SEQS` is only a ceiling: "sequence slots: N -> M", `[qe-src 444b2e49]` `src/engine.cpp` `plan_slots`);
  - shrinks the 27B's reserve on a small pool when `QE_RESERVE_GB` is unset;
  - lowers Flash-Next's context to the smallest card (`FLASH_CTX_FIT`);
  - loads DFlash2 only if it fits (`QE_DRAFTER=auto`);
  - probes and caches the link path;
  - picks the parallel mode (`FLASH_PAR=auto`).
- **⚠ `QE_GPUS` is a literal** (`- QE_GPUS=0,1` / `0,1,2` / `0,1,2,3`, with no `${}`). It is the tensor-parallel width, fixed by the slug's topology; no setting can change it. The image default `0,1,2`, three-card TP (`[qe-entry 1.3.6]`), therefore never applies. Host-card choice still works through `device_ids: ${ESTATE_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}`, because the container then sees ordinals `0..n-1`.
- **Our P2P machinery does not reach QEngine.** No `detect_nvlink.sh` mount, and `NVLINK_MODE` / `DISABLE_CUSTOM_ALL_REDUCE` are not forwarded. QEngine runs its own start-up link probe and P2P data check (`QE_COMM_CHECK`, 4 rounds), and `QE_P2P=0` is the only way to force it off.
- **No injector key reaches these slugs.** Engine type `qengine` is in neither `_ENGINE_TYPE_CONCURRENCY_ENV` nor `_ENGINE_TYPE_MEM_UTIL_ENV` (`launch_compat.py`), and `envelopes.yml` has no `qengine/` row.
- **Shipped values differ from upstream defaults:**
  - `FLASH_MAX_SEQS` is 8 on `dual`; upstream's default is 16.
  - `--served-name` is `qwen3.8-27b` / `qwen3.8-flash-next`; upstream's default is `…-qengine`.
- **Pin drift 1.3.5.9 → 1.3.6** (both images are local):
  - `app.py` gained `--fill-passes`, `--long-share` and `--pause-wait`, and lost `--main-bonus`. We pass none of them.
  - `QE_KV_BITS` and `FLASH_KV_BITS` are absent from the 1.3.5.9 image. They are forwarded, unset.

| Env (compose) | → becomes | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|---|
| *(whole compose)* | — | engine profile `min_sm: 8.6`; headers `Requires-min-vram-gb` 24 (`dual`) / 16 (`multi3`/`multi4`), `Requires-min-gpu-count`, `Tensor-parallel` 2/3/4 | 6/6 | `SM floor` |
| `QE_GPUS` | `QE_DEVICES=${QE_DEVICES:-${QE_GPUS:-0,1,2}}` → `CUDA_VISIBLE_DEVICES` + `--devices 0..n-1` `[qe-entry 1.3.6]` | Docker GPU pick, default `0,1,2`; `QE_DEVICES` wins `[qe-docs 444b2e49]` | 6/6 — `LIT 0,1` ×2, `LIT 0,1,2` ×2, `LIT 0,1,2,3` ×2 | ⚠ Topology — literal TP width (headline) |
| `FLASH_PAR` | engine env | `auto`/`tp`/`pp`/`pptp`. `auto` = TP on 2 equal cards; on 4 cards, TP only with peer access between all of them, else PP `[qe-docs 444b2e49]` `[qe-help 1.3.6]` | 3/6 — pass-through, unset | `engine-auto` |
| `QE_NUMA` | engine env | nodes of the pinned host buffers; unset = each card's node from sysfs `[qe-docs 444b2e49]` | 4/6 (`multi3`/`multi4`) — pass-through, unset | `engine-auto` (host topology) |
| `QE_P2P`, `QE_COMM` (27B) / `FLASH_COMM` (Flash-Next), `QE_DECODE_COMM`, `QE_COMM_CACHE`, `QE_COMM_CHECK`, `QE_COMM_TAX_SEVERE`, `QE_COMM_TAX_1NODE` | engine env | a start-up probe picks p2p / ce / host for prefill and times decode's path. The result is cached per hardware, and the cache key includes NVLink links. `QE_P2P=0` forces host memory `[qe-docs 444b2e49]` | 6/6 — pass-through, unset (`QE_COMM` 3, `FLASH_COMM` 3) | `engine-auto` — QEngine's own link probe; `boot-detect` does not reach it |
| `QE_MAX_SEQS` | engine env `[qe-bin 1.3.6]` | conversations in the KV pool, default 16, up to 32 `[qe-docs 444b2e49]`; lowered at start to fit the cards `[qe-src 444b2e49]` | 3/6 — `${QE_MAX_SEQS:-16}` ×3 (QEngine's default; 32 until 2026-11-23) | `engine-auto` (downward; 32 = the engine's maximum) |
| `FLASH_MAX_SEQS` | engine env `[qe-bin 1.3.6]` | 1–16, default 16; fewer slots leave VRAM to the expert cache `[qe-docs 444b2e49]` | 3/6 — `${FLASH_MAX_SEQS:-8}` ×1 (`dual`), `:-16` ×2 | `bare 24 GB` ⚠ — 8 comes from the 16/8/4 sweep on 3090s (2026-11-01). Flash-Next lowers the context to fit, not the slot count |
| `QE_RESERVE_GB`, `QE_GRAPH_RESERVE_MB` (27B) / `FLASH_EC_MARGIN_MB`, `FLASH_GRAPH_MB` (Flash-Next) | engine env | VRAM per card outside the KV pool: 1.5 GB (a list sets each card) / 128 MB for graphs. Next to the expert cache: 256 MB free / 256 MB for graphs `[qe-docs 444b2e49]` | 3 + 3 — pass-through, unset | `engine-auto` — the 27B lowers the reserve on a small pool when it is unset `[qe-src 444b2e49]` |
| `QE_VRAM_ADAPT`, `_LOW_MB`, `_TARGET_MB` / `FLASH_VRAM_ADAPT`, `_LOW_MB`, `_TARGET_MB` | engine env | when another process takes VRAM: 27B shrinks the KV pool, Flash-Next moves expert groups to host; 1, marks 128/256 MB `[qe-docs 444b2e49]` | 3 + 3 — pass-through, unset | `engine-auto` (runtime) |
| `FLASH_CTX_FIT` | engine env | 1: at start, lower `--max-ctx` to what fits the smallest card `[qe-docs 444b2e49]` | 3/6 — pass-through, unset | `engine-auto` |
| `QE_DRAFTER`, `QE_DFLASH_URL` (27B) | `dflash_fetch.py` + engine | `auto` loads the DFlash2 drafter only when the requested context and slots still fit (0.82 GB a card on two cards) `[qe-docs 444b2e49]` | 3/6 — pass-through, unset | `engine-auto` (VRAM fit) |
| `QE_KV_BITS` / `FLASH_KV_BITS`, `FLASH_KV`, `FLASH_KV_RAM_TOKENS` | engine env | `8` (FP8, QEngine's only default format); `4` = opt-in 4-bit KV, not lossless; `FLASH_KV=ram` = KV in pinned host RAM `[qe-docs 444b2e49]`. `*_KV_BITS` are new in 1.3.6 | 3 + 3 — pass-through, unset | — (opt-in) |
| `QE_KV_ARCHIVE_GB`/`FLASH_KV_ARCHIVE_GB`, `QE_PREFIX_GB`/`FLASH_PREFIX_GB`, `QE_MEM_MARGIN_GB`, `QE_HOST_RESERVE_GB`, `QE_RAM_WAIT_S`, `FLASH_NGRAM_PIN`, `QE_LOAD_DROP`, `QE_SNAPSHOTS` | engine env | host-RAM plan at start: archive 32 GB (27B) / 48 GB (Flash-Next), pinned prompts 8 GB, both "less on smaller machines"; margin max(8 GB, 5 % of RAM) `[qe-docs 444b2e49]` | pass-through, unset (shared ×6, per-model ×3) | `engine-auto` (host RAM) + `launcher-bash:preflight_cpu_offload_ram` on Flash-Next (`CPU-Offload-Host-RAM-GB` 94 on `dual`, 75 on `multi3`/`multi4`) |
| `FLASH_PFB_BIG`, `FLASH_PFB_ALIAS`, `FLASH_MAX_PREFILLS`, `FLASH_PFGRAPH_WINDOW`, `FLASH_ATTN_DMIN`, `FLASH_ATTN_CHUNK`, `FLASH_PRELOAD`, `FLASH_YARN` | engine env | prefill batch and graph tuning; `FLASH_PFB_BIG` is 8192 on 2 cards and 4096 on 3 or with `FLASH_KV=ram` `[qe-docs 444b2e49]` | 3/6 — pass-through, unset | `engine-auto` (`FLASH_PFB_BIG`, by card count); otherwise — |
| `QE_HANG_S`, `FLASH_RUN_TIMEOUT_S`, `FLASH_GUARD_MAX`, `QE_NO_ROOM_S`, `QE_PAUSE_TIMEOUT_S`, `FLASH_EC_RECLAIM_S`, `FLASH_EC_RECLAIM_ABOVE_S` | engine env | watchdogs and idle reclaim `[qe-docs 444b2e49]` | pass-through, unset | — |
| `FLASH_LEND_CHECK`, `FLASH_QSA_PIPE_CE` | engine env | **not in README or `advanced.md`**; names present in `libqflash.so` + `flash_run` `[qe-bin 1.3.6]`. Our header calls them a debug overlap check and the off switch for the copy-engine prefill pipeline | 3/6 — pass-through, unset | — |
| `SPEC_N` (ours) | `--mtp-k N`, only when set | most MTP drafts a step, adaptive; default 27B 7, Flash-Next 3 `[qe-help 1.3.6]`. Our 27B header says `0` also stops the DFlash2 block steps: **UNVERIFIED** (not in upstream docs) | 6/6 — pass-through, unset | — |
| `QE_LOOKUP` / `FLASH_LOOKUP`, `QE_FETCH_ASSETS`, `QE_FETCH_WAIT_S`, `QE_TOKEN_RANK_FILE`, `QE_LSA` (27B) | engine / `assets.py` / `lsa_map.py` | lookup drafts on/off; download the tuned MTP head, draft ranking and LSA map `[qe-docs 444b2e49]` `[qe-help 1.3.6]` (`--lsa`) | pass-through, unset | — |
| `ENABLE_THINKING`, `REASONING_EFFORT`, `THINKING_BUDGET_{LOW,MEDIUM,XHIGH}`, `THINKING_BUDGETS` (ours) | `--think 1\|0`; `QE_REASONING_EFFORT` (the entrypoint refuses an `app.py` without the name); `effort_budget.py` → `QE_THINKING_BUDGET='low:4096,medium:16384,xhigh:32768'` (a hand-set value wins) | = `--think`, `--reasoning-effort` (unset = the template's own, xhigh), `--thinking-budget` per-effort map `[qe-help 1.3.6]` | 6/6 — our defaults `true`, `low`, 4096/16384/32768 | — |
| `VISION` (ours) | `--vision 1\|0` (refuses `1` on an image without torch) | loads the encoder on the last GPU `[qe-help 1.3.6]` `[qe-docs 444b2e49]` | 6/6 — our default `1` | — (it costs the encoder card's KV, and TP sizes every card's pool by the smallest: −23 % on the 27B, our measurement) |
| `KV_OFFLOAD_DISK`, `KV_OFFLOAD_DISK_GB` (ours) | `--disk-cache /kv-offload/qengine-<model>`, `--disk-cache-gb N` | persistent prompt cache; off by default, 200 GB cap `[qe-docs 444b2e49]` (`--disk-cache-gb` has no help text `[qe-help 1.3.6]`) | 6/6 — off | Disk — `bare` |
| `QE_PORT`, `QE_METRICS_PORT`, `QE_MODEL`, `QE_EXTRA_ARGS`; `--served-name` | `--port` via `run.sh`, `--metrics-port` (default 9393), `--model`; extra args go after ours `[qe-entry 1.3.6]` | `--served-name` default `qwen3.8-27b-qengine` / `qwen3.8-flash-next-qengine` `[qe-help 1.3.6]` | `QE_PORT` `LIT 8080` 6/6. `QE_METRICS_PORT` pass-through (`switch.sh` publishes it on 127.0.0.1 when set). `QE_MODEL` `LIT` ×3 (Flash-Next; the 27B uses the image default path). `--served-name` `LIT` ×6 | — |
| `QE_PRESENCE_PENALTY`, `QE_FREQUENCY_PENALTY`, `QE_LOOP_GUARD_{TURNS,NOTE,PENALTY,THINK}`, `QE_THINK_TAIL`, `QE_EMPTY_RETRY`, `QE_THINKING_BUDGET` | `app.py` / `think_budget.py` env | = `--presence-penalty` … `--empty-retry`, each "env QE_…" `[qe-help 1.3.6]`; `QE_THINK_TAIL` `[qe-docs 444b2e49]` | 6/6 — pass-through, unset (`QE_THINKING_BUDGET` is set by our entrypoint) | — |
