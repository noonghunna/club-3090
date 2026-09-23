# Engine flags — inventory

**What this is.** One table per engine family listing every engine flag our shipped composes actually pass, what the engine's *own* documentation says it does, how many registered slugs pass it (and through which `${ENV:-default}`), and — the column that did not exist anywhere before — whether the flag's validity or value depends on the detected card, and by which mechanism. Look things up here instead of re-deriving them from 138 composes.

**Generated 2026-09-21** against commit `3126bc31` (`master` after #1368) — **§7 refreshed the same day against `4dc310a6`**, after #1369 and #1373 landed the injection work this page was written alongside; §§1–6 still describe `3126bc31`, `scripts/lib/profiles/compose_registry.py` = **138 curated slugs** (55 vLLM · 11 SGLang · 2 exllamav3 · 70 llama.cpp-family). Counts exclude `compose/_archive/`, the unregistered files under `models/*/sglang/` and `models/qwen3-omni-30b-a3b/vllm-omni/`, and the gitignored `profiles-local` layer. The in-flight, uncommitted #1365 work that was present in the working tree at generation time was **not** used as a source; it has since merged, and §7 is the only section it touched.

**Upstream flag sets drift.** Every row carries a source tag (legend below). Where no authoritative source could be found the row says `UNVERIFIED` rather than guessing. A flag *we* pass that upstream does *not* document is reported as such — that is a finding, not an error to hide. Re-generate when an engine pin moves (`docs/UPSTREAM.md` tracks the pins).

**Not a tuning guide.** Per-engine levers and their measured effects live in [`engines/`](engines/); cliffs in [`CLIFFS.md`](CLIFFS.md); the KV math in [`KV_MATH.md`](KV_MATH.md); the offload sweep order in [`OFFLOAD_MATRIX.md`](OFFLOAD_MATRIX.md).

---

## 1. How to read the columns

| Column | Meaning |
|---|---|
| **Flag** | The engine's own spelling. Short aliases (`-c`, `-ngl`, `-ot`) are the spelling our composes use. |
| **Upstream status** | What the engine documents, condensed, with a source tag. `[…-help …]` = the pinned image's own `--help`; `[…-src …]` = a fork's argument parser at a named commit (used when the pinned image's `--help` could not be run or the image is not local). |
| **Used by us** | `N` = registered slugs whose compose passes it, out of that family's total. `${VAR:-default}` = the env var the compose reads, with the shipped default(s) and how many slugs use each. `LIT` = hard-coded, no env var — the launcher cannot change it. |
| **Hardware-gated?** | One of the mechanism tags from §2, or `bare 24 GB` (a hand-tuned default nothing adapts) or `—` (not hardware-related). A `⚠` marks the intersection this document exists for: **hardware-dependent in reality, hard-coded in the compose.** |

### Source legend

| Tag | Source |
|---|---|
| `[vllm-help 0.29]` | `vllm serve --help=all`, image `vllm/vllm-openai:v0.29.0` (the `vllm-stable` pin, 2 011 lines) |
| `[vllm-help 0.22]` | same, `vllm/vllm-openai:v0.22.0` (the `vllm-gemma-stable` pin) |
| `[vllm-envs 0.29]` | `vllm.envs` module in the v0.29.0 image (env-var names only) |
| `[sgl-help 0.5.20]` | `python3 -m sglang.launch_server --help`, image `lmsysorg/sglang:v0.5.20` (the `sglang-stable` pin) |
| `[tabby-help]` | `python3 main.py --help`, image `ghcr.io/noonghunna/tabbyapi-club3090@sha256:87490d21…` (the `exllamav3` pin) |
| `[lcpp-help b10920]` | `llama-server --help`, image `ghcr.io/ggml-org/llama.cpp@sha256:6ac92152…` (the `llama-cpp-local` pin = build b10920) |
| `[c3090-help v1.6]` | `llama-server --help`, image `ghcr.io/noonghunna/llamacpp-club3090@sha256:69b833ee…` (the `llamacpp-club3090-v1.6` pin) |
| `[c3090-bin v1.6]` | `strings` over the same image's binaries (env-var names only) |
| `[prism-src 9a9394a]` | PrismML-Eng/llama.cpp `common/arg.cpp` @ `9a9394a` — the pinned prism image's `--help` aborts without `libcuda.so.1`, so the parser source was used |
| `[ik-src baac291]` | ikawrakow/ik_llama.cpp `common/common.cpp` @ `baac291` (HEAD on 2026-09-21). ⚠ The two pinned ik digests (`5f914f1…`, `b35e062…`) are older builds and are not local; presence was checked **by name at HEAD**, not against the pinned binary |
| `[bee-src 98caf25]` | Anbeeld/beellama.cpp `common/arg.cpp` @ `98caf25` — the commit the pinned v0.3.2-preview digest (`858e7cfb…`) was built from |
| `[lmcache-docs]` | LMCache/LMCache `docs/source/cli/server.rst` |
| `UNVERIFIED` | no authoritative source found |

---

## 2. The hardware-gating mechanisms — do not conflate them

Six distinct things decide whether a flag's value tracks the card. They have different reach, different owners and different failure modes; a row's tag names which one applies.

| # | Tag | Mechanism | Where | Reach at `3126bc31` |
|---|---|---|---|---|
| 1 | `SM floor` | A compute-capability gate on the *whole compose*, not a value. Three layers, checked by `compat.fits()` gate **C3** and by the bash `compose_hw_compose_status`: (a) engine profile `min_sm` (`engines/*.yml`: vLLM 7.5, gemma/diffusion/lmcache/SGLang 8.0, exllamav3 8.6, mainline llama.cpp + beellama 6.0, prism + all `llamacpp-club3090-*` 8.6); (b) registry `required_sm` (8.6 on 37 slugs, 9.0 on 8, 7.5 on 1), `supported_sm` (an explicit list, 2 slugs: `vllm/qwen38-27b-dual-ultrafast`/`-ultramax`), `fallback_sm` (10 slugs — the weight-only Marlin fallback floor 7.5 that *replaces* `required_sm` as the hard floor); (c) the compose header `Requires-sm:` (8 vLLM composes: `7.5+` ×5, `8.0+` ×2, `9.0+` ×1), parsed by `scripts/lib/compose-meta.sh`. No flag *value* changes; the slug is accepted or refused. | `scripts/lib/profiles/compat.py` (C3), `scripts/lib/compose-meta.sh` | All 138 slugs (wizard/`switch.sh --list`) |
| 2 | `inject:<KEY>` | **#246 Phase 1/2 envelope injection.** `launch_compat.resolve_variant_pin` computes env exports from the detected card class (`scripts/lib/profiles/hardware/*.yml`, 11 classes, resolved by `_hardware_id_from_gpu`) and `envelopes.yml`; the launcher exports them and the compose consumes `${KEY:-default}`. Keys: `MAX_NUM_SEQS` / `MAX_RUNNING_REQUESTS` (Phase 2 concurrency, keyed by engine *type* since #1362), `GPU_MEMORY_UTILIZATION` (downward-only floor), `VLLM_USE_DEEP_GEMM` (fp8/nvfp4 weights on sm 8.9/12.0/12.1 → `0`), `MOE_RESERVE_MB` (upward-only on >24 GB cards, `moe_cache` slugs), `DECODE_GRANULARITY` (a model property, not hardware). A user-set value always wins. | `scripts/lib/profiles/launch_compat.py`, `envelopes.yml`, `scripts/launch.sh:1195-1275`, `scripts/switch.sh:1044-1120` | **65 of 138 slugs.** Both launchers gate the call on `[[ $variant == vllm/* \|\| $variant == beellama/* ]]`; for every other engine `resolve_engine_pin` raises *"install.spec is not a docker image"* (it is one — the raise is the missing `_ENGINE_IMAGE_ENV` entry). Verified by simulation on a HEAD worktree: `sgl/*`, `llamacpp/*`, `llamacpp-club3090/*`, `ik-llama/*`, `exllamav3/*` all error out. Tracked as #1361/#1365. Within the 65: `MAX_NUM_SEQS` fires for 5 slugs on 4 card classes, `GPU_MEMORY_UTILIZATION` only on `dgx-spark` (0.85), `VLLM_USE_DEEP_GEMM=0` on every fp8/nvfp4 vLLM slug on sm 8.9/12.x, `MOE_RESERVE_MB` never (no vLLM slug has `moe_cache`, and it is absent from the launcher `case` allowlist — #1363). |
| 3 | `launcher-bash:<fn>` | The second, bash-side hardware path in `scripts/lib/compose-meta.sh` + `scripts/preflight.sh`, keyed on compose **header comments** and live `nvidia-smi`: `resolve_offload_residency` exports `OT_G<i>` from *free* VRAM per card (additive fit model: `(free − reserve − first_card_extra − margin) / bundle`, headers `CPU-Offload-Bundle-MiB` / `-MoE-Layers` / `-GPU-Reserve-MiB` / `-First-MoE-Layer` / `-Draft-Reserve-MiB` / `-Draft-Card`; ≥2 cards only); `resolve_offload_threads` exports `THREADS=nproc/2` for any compose matching `=CPU`/`--cpu-moe`/`--n-cpu-moe`; `preflight_ik_llama_image` swaps `IK_LLAMA_IMAGE` to the cu12 build when the driver's CUDA < 13.2; `preflight_offload_split_mode` refuses `-sm row`/`tensor` under offload; `preflight_cpu_offload_ram` gates host RAM on `CPU-Offload-Host-RAM-GB` minus the residency grant; `preflight_single_card_util` warns when a user raises `GPU_MEMORY_UTILIZATION` above the single-card default. | `scripts/lib/compose-meta.sh:386-721`, `scripts/preflight.sh` | `OT_G*`: the 10 `residency` composes (7 club3090 + 3 mainline). `THREADS`: 31 llama.cpp composes + both exl3 (which compute the same `nproc/2` in-container). Headers `Requires-min-vram-gb`/`-gpu-count`/`Tensor-parallel` on 36 composes (33 vLLM, 1 SGLang, 1 llama.cpp, 1 lmcache) feed the `--list` fit display only. |
| 4 | `boot-detect` | Decided **inside the container at boot** by `scripts/detect_nvlink.sh` (bind-mounted at `/etc/club3090/`): NVLink or driver-confirmed PCIe P2P → `NCCL_P2P_LEVEL`, custom all-reduce **on** (flag omitted), `expandable_segments` stripped from `PYTORCH_CUDA_ALLOC_CONF`; otherwise `NCCL_P2P_DISABLE=1`, `--disable-custom-all-reduce` **passed**. `NVLINK_MODE=auto\|force_on\|force_off\|pcie_p2p` and `DISABLE_CUSTOM_ALL_REDUCE=1` are the operator overrides (#1332). | `scripts/detect_nvlink.sh` | 46 vLLM composes source it. **0 of 11 SGLang composes do** — they hard-code `--disable-custom-all-reduce`. |
| 5 | `engine-auto` | The engine sizes itself from the device at load time; the compose only sets policy. llama.cpp `--fit` (default `on`: adjusts *unset* `-ngl`/`-c` to fit) and `-ngl auto`; club3090 `--moe-cache auto` (grants free − `GGML_CUDA_MOE_CACHE_RESERVE_MB` per device); exllamav3 `--gpu-split-auto` + `--autosplit-reserve`; vLLM's `--gpu-memory-utilization` is a *fraction* so the KV pool scales with VRAM even though the number is fixed. | engine | Per row |
| 6 | `bare 24 GB` | `${FLAG:-value}` hand-tuned on the reference 2× RTX 3090 with nothing adapting it. The largest class. Marked `⚠` when the right value demonstrably depends on the card (a bigger card wastes VRAM, a smaller one OOMs). | compose | Per row |

Two things that are **not** hardware gating but look like it: the `SPEC_N`/`SPEC=off` drafter contract (a user toggle that adds or removes the spec-dec flags, 39 vLLM + 59 llama.cpp + 11 SGLang + 2 exl3 composes) and `VLLM_ENFORCE_EAGER` (a cudagraph toggle, 23 composes) — both are operator knobs, present regardless of card.

---

## 3. vLLM — 55 registered slugs, 11 engine ids

Engine pins in play: `vllm-stable` = `v0.30.0` (the vast majority), `vllm-gemma-stable` = `v0.22.0` (6 slugs), `vllm-diffusion-gemma` = `v0.24.0` (1), `vllm-gemma4-unified` = ephemeral tag (2), `vllm-lmcache` = `lmcache/vllm-openai@sha256:663f9b2f…` (1). The 4 `vllm-nightly-*` ids and `vllm-pip-baseline`/`vllm-stable-next` have **zero** registry users. Every flag below is present in *both* `--help=all` dumps (v0.29.0 and v0.22.0) except where noted. **Re-checked at the v0.30.0 bump (2026-09-23):** every `--flag` and `VLLM_*` env name used by the 43 `vllm-stable` composes resolves identically in the v0.30.0 dumps (`--help=all` 2 093 lines; `vllm.envs`); the only flag v0.30.0 removed, `--enable-bf16x3-router-gemm`, has no users. The `[vllm-help 0.29]` / `[vllm-envs 0.29]` citations below were not re-derived — they name the dump each claim was read from.

**Headline findings for this engine**

- **Phase 1 `KV_CACHE_DTYPE` injection is GONE** (retired in #1371; at this page's original anchor it was still present but inert). `_ARCH_KV_ALLOWED = {"fp8_e5m2": {"fp8_e4m3"}}` was keyed by the slug's registry `kv_format`, and **zero of 138 slugs** declare `fp8_e5m2` — both pilots (`vllm/dual`, `vllm/minimal`) migrated to `fp8_e4m3`, so the function returned `{}` for every card. Removal verified behaviour-neutral across 1,242 (slug, rig) pairs.
- **10 slugs hard-code `--max-num-seqs`** (`vllm/qwen-27b-dual-max`, `-dual-nvfp4`, `-multi-fast`, `-multi-max`, `-single-nvfp4`, `vllm/qwen-a3b-preview-single`, `vllm/qwen38-27b-dual-ultramax`, `vllm/gemma-26ba4b-dual`, plus 2 deprecated). The one Phase 2 knob that *does* inject cannot reach them — five are `⚠️ Production w/ caveats`.
- **`--kv-cache-dtype int8_per_token_head` is not a stock value.** Neither `--help` lists it; it is supplied by the vendored PR #40391 overlay on `vllm-gemma-stable` (registry `required_engine_features: [int8_per_token_head]` on 2 slugs). Those 5 gemma composes also read `${KV_DTYPE:-…}`, not `${KV_CACHE_DTYPE:-…}`.
- `VLLM_ATTENTION_BACKEND` (set in `environment:` by 2 composes) **is not a recognised env var in v0.29.0** (`vllm.envs` has no such name). On `vllm/nemotron-75b-dual-w4a16` (vllm-stable) it is inert; the same compose also passes `--attention-backend TRITON_ATTN`, which is what actually takes effect. The launcher's `case` allowlist still carries an arm for it. (Not checked on the v0.24.0 diffusiongemma pin.)
- `--prefix-match-unit` exists in v0.29.0 but **not** in v0.22.0. All 8 users are on `vllm-stable`, so this is safe today — it would break silently on a pin roll-back.

### 3a. Placement, memory, concurrency

| Flag | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|
| `--tensor-parallel-size` | Number of tensor-parallel replicas. `[vllm-help 0.29]` | 55/55 — `${TP:-2}` ×27, `${TP:-1}` ×10, `${TP:-4}` ×10, `${TP:-8}` ×6; `LIT 2` ×2 (`qwen38-27b-dual-ultramax`, `thinkingcap-dual-w4a8`) | Topology. `launch.sh` exports `TP`/`PP` from the wizard's card selection (`pick_parallelism`), and the compose default equals its `<topology>/` dir. `compat.fits()` refuses a TP the selected cards cannot host. The 2 `LIT` slugs ignore the launcher. |
| `--pipeline-parallel-size` | Number of pipeline stages. `[vllm-help 0.29]` | 39/55 — `${PP:-1}` ×39 | Topology (launcher exports `PP`; no PP compose ships — `docs/MULTI_CARD.md`) |
| `--gpu-memory-utilization` | Fraction of GPU memory for the model executor, 0–1, per-instance; default 0.92. `[vllm-help 0.29]` | 54/55 — `${GPU_MEMORY_UTILIZATION:-0.92}` ×26, `:-0.95` ×12, `:-0.90` ×7, `:-0.85` ×4, `:-0.94`, `:-0.82`, `:-0.40` | `engine-auto` (fraction scales with VRAM) + `inject:GPU_MEMORY_UTILIZATION` — **downward only**, fires when `mem_util_safe < default`, i.e. today only `dgx-spark` (0.85). Never raised on bigger cards by design (Cliff-2b margin). `launcher-bash:preflight_single_card_util` warns on user increases (TP=1). Registry `mem_util` mirrors the default for `fits()`. |
| `--max-num-seqs` | Max sequences per iteration (cap, graceful preemption). `[vllm-help 0.29]` | 55/55 — `${MAX_NUM_SEQS:-2}` ×18, `:-1` ×14, `:-4` ×8, `:-8` ×4; **`LIT`** ×10 (list above) | `inject:MAX_NUM_SEQS` — `envelopes.yml` rows: `vllm/dual` (rtx-5090→4, rtx-6000-pro→8), `vllm/minimal` (rtx-5090→9, rtx-6000-pro→16, dgx-spark→8), `vllm/qwen38-27b-dual-fast`/`-multi4-fast`/`-multi8-fast` (rtx-a6000→5/13/14, `method: measured-pool-scaling`, reachable only since #1364). Every other slug, and every card class not listed: `bare 24 GB`. Heterogeneous rigs clamp to the smallest card. |
| `--max-model-len` | Model context length; derived from config if unset. `[vllm-help 0.29]` | 55/55 — `${MAX_MODEL_LEN:-262144}` ×36, `${MAX_MODEL_LEN:-${CTX:-262144}}` ×6, `:-131072` ×4, `:-8192` ×2, `:-65536` ×2, and one each of `:-176000`, `:-200000`, `:-147456`, `:-32768`, `${CTX:-229376}` | `bare 24 GB` ⚠ — the context lever is **deliberately not injected** (`envelopes.yml` header: single-card vLLM only, Cliff-2b-capped, deferred). Fit is *checked* by kv-calc in `fits()`, never *adapted*. |
| `--max-num-batched-tokens` | Max tokens per scheduler iteration (chunk size). `[vllm-help 0.29]` | 51/55 — `${MAX_NUM_BATCHED_TOKENS:-8192}` ×37, `:-4096` ×10, `:-2048` ×2, `:-1024`, `:-1584` (lmcache: must be ≥ the int8-PTH block) | `bare 24 GB` ⚠ — sets the prefill activation peak that decides the single-card prefill cliff (`CLIFFS.md`); a 32 GB card could chunk larger, a 12 GB card needs smaller. |
| `--long-prefill-token-threshold` | Chunked prefill: a request is "long" above N tokens; 0 disables. `[vllm-help 0.29]` | 26/55 — `${LONG_PREFILL_TOKEN_THRESHOLD:-4096}` ×19, `:-2048` ×5, `:-0` ×1, `LIT 0` ×1 | `bare 24 GB` |
| `--kv-cache-memory-bytes` | Absolute KV pool size per GPU; **ignores `gpu_memory_utilization` when set**. `[vllm-help 0.29]` | 3/55 — `${KV_CACHE_MEMORY_BYTES:-820000000}` ×1, `:-6335076762` ×1, computed in-entrypoint ×1 | `bare 24 GB` ⚠ — an absolute byte count does **not** scale with VRAM the way the fraction does; on a bigger card the pool stays the 24 GB size. |
| `--kv-cache-dtype` | `auto` / `fp8` (= `fp8_e4m3`) / `fp8_e5m2` / `bfloat16` (fp8-default models); CUDA 11.8+. `[vllm-help 0.29]` `[vllm-help 0.22]` | 51/55 — `${KV_CACHE_DTYPE:-fp8_e4m3}` ×22, `:-fp8` ×10, `:-bfloat16` ×4, `${KV_DTYPE:-int8_per_token_head}` ×3, `${KV_DTYPE:-auto}` ×2, `LIT auto` ×2, `LIT fp8`/`fp8_e4m3` ×2 | `SM floor` in reality: on sm_86 fp8 KV is **storage only** (no FP8 compute — `docs/HARDWARE.md`); `hardware/*.yml` `supported_kv_formats` gate it in `fits()` (C4). No injector: `inject:KV_CACHE_DTYPE` was wired for `vllm/dual`+`vllm/minimal` and retired in #1371 (see headline). The compose default is the whole story. `int8_per_token_head` = overlay-only value. |
| `--dtype` | Weight/activation dtype: `auto`/`half`/`float16`/`bfloat16`/`float32`. `[vllm-help 0.29]` | 43/55 — `LIT bfloat16` ×33, `LIT float16` ×9, `${DTYPE:-bfloat16}` ×1 | `SM floor` in reality (bf16 needs sm_80+; every supported class qualifies) — hard-coded, harmless. |
| `--quantization` | Weight quantization method; `None` reads `quantization_config`. `[vllm-help 0.29]` | 40/55 — `LIT auto_round` ×18, `fp8` ×12, `modelopt` ×6, `compressed-tensors` ×4 | `SM floor` — fp8 weights on sm_86 run the Marlin/CUTLASS W8A16 path (no native FP8); `modelopt` (NVFP4) slugs carry registry `required_sm: 9.0` + `fallback_sm: 7.5` (Marlin W4A16 fallback). `inject:VLLM_USE_DEEP_GEMM=0` fires for `fp8*`/`nvfp4` weights on sm 8.9/12.0/12.1 (27 composes read the env). |
| `--enable-expert-parallel` | Expert parallelism instead of TP for MoE layers. `[vllm-help 0.29]` | 2/55 — bare | — |
| `--attention-backend` | Attention backend; `auto`/None = automatic. `[vllm-help 0.29]` | 7/55 — `${ATTN_BACKEND:-FLASH_ATTN}` ×5, `LIT FLASH_ATTN` ×1, `LIT TRITON_ATTN` ×1 | `SM floor` in reality ⚠ — FA2 is the sm_80+ path; FA3 is sm_90-only and never selected here. Hard-coded; a Hopper rig gets FA2 unless the operator overrides. |
| `--enforce-eager` | Always eager PyTorch; disables CUDA graphs. `[vllm-help 0.29]` | 24/55 — `${VLLM_ENFORCE_EAGER:+--enforce-eager}` in 23 entrypoints, bare ×1 (diffusiongemma) | — (operator toggle) |
| `--compilation-config` | JSON compilation/cudagraph config. `[vllm-help 0.29]` | 2/55 — `LIT` cudagraph capture sizes | — |
| `--async-scheduling` / `--no-async-scheduling` | Async scheduling on/off (avoids GPU-utilisation gaps). `[vllm-help 0.29]` | 1 + 2 /55 | — |
| `--disable-custom-all-reduce` | Disable the custom all-reduce kernel, fall back to NCCL. `[vllm-help 0.29]` | 46/55 — **never in `command:`**; added by the entrypoint when `_CUSTOM_AR_ENABLED != 1` | `boot-detect` — `detect_nvlink.sh` decides at boot (NVLink / driver-confirmed P2P → kernel on; PCIe → flag passed). vLLM additionally vetoes its kernel at world > 2 without a full NVLink mesh (#786). Override: `DISABLE_CUSTOM_ALL_REDUCE=1`, `NVLINK_MODE=…`. |
| `--enable-chunked-prefill` | Prefill chunking on `max_num_batched_tokens`. `[vllm-help 0.29]` | 43/55 — bare | — |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | Prefix caching on/off. `[vllm-help 0.29]` | 9 bare + 13 via `${PREFIX_CACHE_ARG:---enable-prefix-caching}` + 1 `--no-…` | — |
| `--prefix-match-unit` | Finest token boundary a prefix-cache hit can land on; must divide every KV group's block size. `[vllm-help 0.29]` **absent in 0.22** | 8/55 — `${PREFIX_MATCH_UNIT:-16}` (all on `vllm-stable`) | — (pin-gated, not hardware) |
| `--mamba-cache-mode` | Mamba prefix-cache strategy `none`/`all`/`align`. `[vllm-help 0.29]` (0.22 differs on the default) | 9/55 — `${MAMBA_CACHE_MODE:-align}` ×8, `LIT align` ×1 | — |
| `--mamba-block-size`, `--mamba-cache-dtype` | Mamba block size / cache dtype. `[vllm-help 0.29]` | 8/55 each — entrypoint `MAMBA_ARGS` from `MAMBA_BLOCK_SIZE`/`MAMBA_CACHE_DTYPE` | — |
| `--mamba-ssm-cache-dtype`, `--mamba-backend`, `--enable-mamba-cache-stochastic-rounding`, `--mamba-cache-philox-rounds` | SSM-state dtype; SSU backend `triton`/`flashinfer`; stochastic rounding to fp16 cache; PRNG rounds. `[vllm-help 0.29]` | 1 / 2 / 1 / 1 | `--mamba-backend flashinfer` (1 slug): FlashInfer must support the active GPU + dtype per `--help`; not gated by us. |

### 3b. Model, API and sampling (not hardware-related)

| Flag | Upstream status | Used by us | HW |
|---|---|---|---|
| `--model`, `--served-model-name`, `--host`, `--port` | `[vllm-help 0.29]` | 55 / 55 / 54 / 54 — `LIT` | — |
| `--trust-remote-code` | `[vllm-help 0.29]` | 54/55 | — |
| `--reasoning-parser` | `[vllm-help 0.29]` | 54/55 — `qwen3` ×41, `gemma4` ×11, `nemotron_v3` ×2 | — |
| `--tool-call-parser`, `--enable-auto-tool-choice` | `[vllm-help 0.29]` | 54 / 54 — `qwen3_coder` ×42, `gemma4` ×12 | — |
| `--chat-template` | `[vllm-help 0.29]` | 28/55 — the froggeric Qwen / canonical Gemma templates | — |
| `--default-chat-template-kwargs` | `[vllm-help 0.29]` | 42/55 — `enable_thinking` via `${ENABLE_THINKING}` / `${REASONING_EFFORT}` | — |
| `--override-generation-config` | `[vllm-help 0.29]` | 54/55 — sampler JSON from `TEMP`/`TOP_P`/`TOP_K`/`MIN_P`/`REPEAT_PENALTY` | — |
| `--generation-config` | `[vllm-help 0.29]` | 1/55 (deprecated slug) | — |
| `--speculative-config` | JSON spec-decode config (`method`, `model`, `num_speculative_tokens`). `[vllm-help 0.29]` | 40/55 — entrypoint `SPEC_ARGS` under the `SPEC_N`/`SPEC=off` contract; methods `mtp`, `dflash`, draft-model | — (model-gated: EAGLE/DFlash inside vLLM blocked on Qwen3-Next — `docs/UPSTREAM.md` vllm#39931) |
| `--limit-mm-per-prompt`, `--mm-processor-kwargs`, `--language-model-only` | `[vllm-help 0.29]` | 10 / 1 / 1 | — |
| `--hf-overrides` | `[vllm-help 0.29]` | 1/55 (deprecated diffusiongemma) | — |
| `--kv-transfer-config` | KV connector config JSON. `[vllm-help 0.29]` | 1/55 — `LMCacheMPConnector` (`vllm/qwen-27b-dual-lmcache`, incubating) | — |

### 3c. `lmcache server` (a second process in the lmcache compose, not vLLM)

| Flag | Upstream status | Used by us | HW |
|---|---|---|---|
| `--chunk-size` | Chunk size in tokens. `[lmcache-docs]` | 1 — `${LMCACHE_CHUNK_SIZE:-1584}` (must be a multiple of vLLM's block) | — |
| `--l1-size-gb` | L1 (CPU RAM) tier size. `[lmcache-docs]` | 1 — `${LMCACHE_L1_GB:-30}` | Host RAM: `preflight.sh` gates available RAM ≥ l1 + a vLLM reserve; `shm_size` must be ≥ this. `bare` for the size itself. |
| `--eviction-policy` | `[lmcache-docs]` | 1 — `LRU` | — |
| `--l2-adapter` | **UNVERIFIED** — `server.rst` documents L2 adapters as config-file settings and points to `lmcache server --help`; the flag itself is not listed. | 1 — off by default (`LMCACHE_L2=1` / `LMCACHE_L2_ADAPTER=…`) | — |

### 3d. vLLM environment knobs our composes set (names verified against `vllm.envs` v0.29.0)

| Env | In `vllm.envs` 0.29? | Composes | HW |
|---|---|---|---|
| `VLLM_USE_DEEP_GEMM` | yes | 27 (as `${VLLM_USE_DEEP_GEMM:-…}` pass-through) | `inject:VLLM_USE_DEEP_GEMM` → `0` on sm 8.9 / 12.0 / 12.1 for fp8/nvfp4 weights (DeepGEMM is Hopper + datacenter-Blackwell only; disc #571) |
| `VLLM_SKIP_P2P_CHECK` | yes | 43 | PCIe topology assumption, hard-coded |
| `VLLM_USE_FLASHINFER_SAMPLER` | yes | 30 | — |
| `VLLM_MARLIN_INPUT_DTYPE` | yes | 13 | — (W4A8 path, sm_86 Marlin) |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | yes | 12 | — |
| `VLLM_USE_V2_MODEL_RUNNER` | yes | diffusiongemma | — |
| `VLLM_ATTENTION_BACKEND` | **no** | 2 (`environment: …=TRITON_ATTN`) | inert on v0.29.0 (see headline) |
| `NCCL_P2P_DISABLE`, `NCCL_CUMEM_ENABLE`, `PYTORCH_CUDA_ALLOC_CONF` | n/a (NCCL / PyTorch) | 53 each | `boot-detect` — `detect_nvlink.sh` rewrites `NCCL_P2P_*` and strips `expandable_segments` on the P2P path |
| `VLLM_ENFORCE_EAGER`, `SPEC_N`, `SPEC`, `MAMBA_*`, `W4A8` | ours, not vLLM's | — | entrypoint contracts |

---

## 4. SGLang — 11 registered slugs, 1 engine id

All 11 are `🧪 Experimental`, all on `sglang-stable` = `lmsysorg/sglang:v0.5.20`, all `models/qwen3.8-27b/sglang/…`, all `bash -c` script bodies (no `command:` list). Every flag below is in `[sgl-help 0.5.20]`. The two unregistered `eagle3-experimental.yml` files additionally pass `--mamba-scheduler-strategy`, which is **not** in the v0.5.20 help — they are not launchable and are excluded here.

**Headline findings for this engine**

- **No `detect_nvlink.sh`, no `${TP}`.** All 11 hard-code `--tp-size 2|4|8` and `--disable-custom-all-reduce`. An NVLink or P2P rig runs SGLang with the custom kernel off, and the launcher cannot change the width.
- **`MAX_RUNNING_REQUESTS` is wired but dead.** The compose reads it and `_ENGINE_TYPE_CONCURRENCY_ENV["sglang"]` emits it (#1362), but (a) the launcher never calls the injector for `sgl/*` and (b) `envelopes.yml` has no `sgl/` row. `MEM_FRACTION` has no injector at all — `_mem_util_env` emits `GPU_MEMORY_UTILIZATION`, a key no SGLang compose reads (#1363).

| Flag | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|
| `--tp-size` | Tensor parallelism size. `[sgl-help 0.5.20]` | 11/11 — **`LIT`** `2` ×4, `4` ×4, `8` ×3 (from the topology dir) | Topology ⚠ — hard-coded; `compat.fits()` still refuses a mismatch, but nothing adapts it. |
| `--disable-custom-all-reduce` | Disable the custom all-reduce kernel, fall back to NCCL. `[sgl-help 0.5.20]` | 11/11 — bare, unconditional | `boot-detect` in reality ⚠ — hard-coded OFF; no NVLink detection (contrast vLLM §3a). |
| `--mem-fraction-static` | Fraction for static allocation (weights + KV pool); lower on OOM. `[sgl-help 0.5.20]` | 11/11 — `${MEM_FRACTION:-0.82}` ×3 (autoround dflash2), `:-0.84` ×2, `:-0.86` ×2, `:-0.90` ×2, `:-0.92` ×1, `:-0.95` ×1 (`sgl/qwen38-27b-dual-fast` — the registry records `mem_util: 0.9` for it, so `fits()` reasons about a number the compose does not ship) | `engine-auto` (fraction) + `bare 24 GB`. The `dgx-spark` floor injector emits the wrong key for this engine (§2 #2). |
| `--max-running-requests` | Max running requests. `[sgl-help 0.5.20]` | 11/11 — `${MAX_RUNNING_REQUESTS:-2}` ×9, `:-1` ×2 (`dual-fast` mtp on fp8, `dual-superfast`). ⚠ The registry carries `max_num_seqs: 1` for **all 11** — the kv-calc fit runs at half the shipped concurrency on 9 of them. | `inject:MAX_RUNNING_REQUESTS` — plumbed, **unreachable** (headline). Otherwise `bare 24 GB`. |
| `--context-length` | Model max context; SI/IEC suffixes. `[sgl-help 0.5.20]` | 11/11 — `${CONTEXT_LENGTH:-262144}` / `:-196608` / `:-163840` | `bare 24 GB` ⚠ |
| `--chunked-prefill-size` | Max tokens per prefill chunk; −1 disables. `[sgl-help 0.5.20]` | 11/11 — `${CHUNKED_PREFILL_SIZE:-2048}` | `bare 24 GB` ⚠ (same role as vLLM `--max-num-batched-tokens`) |
| `--kv-cache-dtype` | `{auto, fp8_e5m2, fp8_e4m3, mxfp8, bf16, bfloat16, nvfp4, fp4_mx_block16, fp4_e2m1}`; fp8 needs CUDA 11.8+. `[sgl-help 0.5.20]` | 11/11 — `${KV_CACHE_DTYPE:-fp8_e4m3}` ×7, `:-auto` ×4 | `SM floor` in reality — storage-only on sm_86; `hardware/*.yml` `supported_kv_formats` gates in `fits()`. No injector for this engine. |
| `--attention-backend` | Kernel choice; 24 backends incl. `fa3`, `fa4`, `flashinfer`, `triton`. `[sgl-help 0.5.20]` | 11/11 — `${ATTENTION_BACKEND:-flashinfer}` | `SM floor` in reality ⚠ — `fa3`/`fa4` are Hopper+/Blackwell paths never selected here; hard-coded to the sm_86-safe backend. |
| `--quantization` | Weight quantization method. `[sgl-help 0.5.20]` | 11/11 — `${QUANTIZATION:-auto-round}` / `:-fp8` / `LIT auto-round` | `SM floor` in reality — fp8 weights on sm_86 take the non-native path; `W4A8` env selects the W4A8 kernel patch (13 composes). |
| `--max-mamba-cache-size` | Max mamba cache entries. `[sgl-help 0.5.20]` | 1/11 — `${MAX_MAMBA_CACHE_SIZE:-10}` | `bare 24 GB` |
| `--mamba-ssm-dtype` | SSM state dtype (else model config). `[sgl-help 0.5.20]` | 11/11 — `${MAMBA_SSM_DTYPE:-bfloat16}` | — |
| `--enable-session-radix-cache`, `--radix-eviction-policy` | Per-session radix refs; eviction `{lru,lfu,slru,priority}`. `[sgl-help 0.5.20]` | 11 / 11 — `${SESSION_RADIX_CACHE}`, `${RADIX_EVICTION_POLICY}` | — |
| `--speculative-algorithm` | Builtins `EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK, UNO`. `[sgl-help 0.5.20]` | 11/11 — `EAGLE` (MTP head, 6 `mtp.yml`) or `DFLASH` (5 `dflash2*.yml`), chosen in-script | — |
| `--speculative-draft-model-path` | Draft model path. `[sgl-help 0.5.20]` | 11/11 | — |
| `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens` | EAGLE depth / top-k / draft tokens. `[sgl-help 0.5.20]` | 6/11 each — from `SPEC_N` | — |
| `--speculative-dflash-block-size` | DFLASH only; alias of `--speculative-num-draft-tokens`. `[sgl-help 0.5.20]` | 5/11 — from `SPEC_N` | — |
| `--speculative-draft-model-quantization` | Drafter quantization. `[sgl-help 0.5.20]` | 5/11 — `${DRAFT_QUANT:-unquant}` | — |
| `--model-path`, `--served-model-name`, `--host`, `--port`, `--trust-remote-code` | `[sgl-help 0.5.20]` | 11/11 — `LIT` (port 30000) | — |
| `--reasoning-parser`, `--tool-call-parser`, `--default-chat-template-kwargs`, `--preferred-sampling-params`, `--sampling-defaults model` | `[sgl-help 0.5.20]` | 11/11 each | — |
| `--enable-cache-report`, `--enable-metrics` | `[sgl-help 0.5.20]` | 11/11 each | — |

Script-body artefact excluded: `--verify` (11) is the vendored w4a8 install script's own flag, not SGLang's.

---

## 5. exllamav3 (TabbyAPI) — 2 registered slugs, 1 engine id

Both `🐣 Incubating` (`exllamav3/qwen38-flash-next-dual-exl3-305-cpumoe`, `-405-cpumoe`), image `ghcr.io/noonghunna/tabbyapi-club3090@sha256:87490d21…`, `min_sm: 8.6`. TabbyAPI is config-file driven; the compose passes CLI overrides on top of `--override-preset safe_defaults`. Every flag below is in `[tabby-help]`. `registry.offload = cpu-moe-split`; `moe_cache = False`; no `mem_util`.

**Headline finding:** `--cpu-moe-split-experts` is the #1360 case — a 24 GB-calibrated expert count that a 32 GB card inherits verbatim and runs half CPU-resident with VRAM idle. #1366 proposes deriving it as a VRAM fit; at this commit it is a bare default.

| Flag | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|
| `--cpu-moe-split-experts` | Routed experts per MoE layer to run on CPU (default 0); splits *every* MoE layer. `[tabby-help]` | 2/2 — `${MOE_SPLIT:-144}` (3.05 bpw), `${MOE_SPLIT:-256}` (4.05 bpw) | `bare 24 GB` ⚠⚠ — the value *is* a VRAM fit; nothing computes it (#1360 / #1366). Not in any injector (#1361). |
| `--cpu-moe-threads` | Worker threads for CPU MoE; unset = library default. `[tabby-help]` | 2/2 — from `THREADS`, else `nproc/2` computed in the entrypoint | `launcher-bash:resolve_offload_threads` exports `THREADS=nproc/2` (compose matches `--cpu-moe`); the entrypoint computes the same floor. Measured good zone is an absolute ~24-32 (`docs/engines/EXLLAMAV3.md`) — CPU-keyed, not VRAM-keyed. |
| `--gpu-split-auto` | Auto-allocate across GPUs (default True); ignored on 1 GPU. `[tabby-help]` | 2/2 — `LIT true` | `engine-auto` |
| `--autosplit-reserve` | VRAM reserved for autosplit, one MB value per GPU (default 96 on GPU 0). `[tabby-help]` | 2/2 — `${RESERVE:-512}` `${RESERVE:-512}` (two entries) | `bare 24 GB` ⚠ — **two** list entries hard-codes a 2-card rig; a 4-card rig gets no reserve on cards 2-3. |
| `--cache-size` / `--max-seq-len` | KV cache size / max sequence length (tokens). `[tabby-help]` | 2/2 — `${CTX:-204800}` for both | `bare 24 GB` ⚠ (204800 is where it was left, not a ceiling — profile notes) |
| `--cache-mode` | KV quantization, `k_bits,v_bits` 2-8 or FP16/Q4 etc. `[tabby-help]` | 2/2 — `${KV_TYPE:-Q4}` | `bare` |
| `--chunk-size` | Prefill chunk size. `[tabby-help]` | 2/2 — `${CHUNK:-4096}` | `bare 24 GB` |
| `--sysmem-kv-cache`, `--sysmem-recurrent-cache` | Host-RAM KV / recurrent-state caches (MB). `[tabby-help]` | 2/2 — `${SYSMEM_KV:-4096}`, `${SYSMEM_RECURRENT:-8192}` | Host RAM — `bare` |
| `--ngram-ram` | Load the n-gram/PLE table into system RAM (PLE models only). `[tabby-help]` | 2/2 — `${NGRAM_RAM:-true}` | Host RAM (31 GB table; `CPU-Offload-Host-RAM-GB` header gates it) |
| `--vision-offload` | Keep vision weights in pinned host RAM, stream to GPU. `[tabby-help]` | 2/2 — `${VISION_OFFLOAD:-true}` | Host RAM — `bare` |
| `--tensor-parallel` | TP on/off (default off). `[tabby-help]` | 2/2 — `LIT false` (autosplit, not TP) | Topology — hard-coded off by design |
| `--draft-mode`, `--draft-num-tokens`, `--dynamic-draft` | Built-in MTP drafter; with dynamic draft, `--draft-num-tokens` is a ceiling. `[tabby-help]` | 2/2 — `mtp`, `SPEC_N` (default 4), `${DYNAMIC_DRAFT:-true}` | — |
| `--backend exllamav3`, `--api-servers OAI`, `--disable-auth true`, `--host`, `--port 5000`, `--model-dir`, `--model-name`, `--override-preset safe_defaults` | `[tabby-help]` | 2/2 — `LIT` | — |
| `--vision true`, `--tool-format qwen3_coder`, `--tool-calls-in-reasoning true`, `--reasoning`, `--reasoning-start-token`, `--reasoning-end-token`, `--start-in-reasoning auto` | `[tabby-help]` | 2/2 | — |

---

## 6. llama.cpp family — 70 registered slugs, 8 engine ids (+ ik_llama)

| Sub-engine | Slugs | Engine id(s) | Image | Help source |
|---|---|---|---|---|
| mainline | 15 (`llamacpp/*`) | `llama-cpp-local` | `ghcr.io/ggml-org/llama.cpp@sha256:6ac92152…` (b10920) | `[lcpp-help b10920]` |
| club3090 (moe-cache) | 28 (`llamacpp-club3090/*`) | `llamacpp-club3090-v1.5` (7), `-v1.6` (21); `llamacpp-club3090`, `-v1.1` have 0 users | `ghcr.io/noonghunna/llamacpp-club3090@sha256:…` | `[c3090-help v1.6]` |
| ik_llama | 15 (`ik-llama/*`, 13 deprecated) | **`llama-cpp-local`** — there is no ik-llama engine profile; these slugs are registered against the *mainline* profile, whose `supported_kv_formats`/`supported_drafters` therefore do not describe them | `${IK_LLAMA_IMAGE:-ghcr.io/ikawrakow/ik-llama-cpp@sha256:5f914f1…}` (13) / `…b35e062…` (2) | `[ik-src baac291]` |
| beellama | 10 (`beellama/*`, all deprecated) | `beellama-local` | `ghcr.io/anbeeld/beellama.cpp@sha256:858e7cfb…` | `[bee-src 98caf25]` |
| prism | 2 (`llama-cpp-prism*/`) | `llama-cpp-prism`, `llama-cpp-prism-mtp` | `ghcr.io/noonghunna/llamacpp-prism*@sha256:…` | `[prism-src 9a9394a]` |

Rows are tagged with the sub-engines that pass them. Flags shared with mainline were checked in both `[lcpp-help b10920]` and `[c3090-help v1.6]`; the two help outputs differ only in `--moe-cache`, `--direct-io`, `--tensor-read-lazy`, `--mlock/--mmap/--no-mmap` (present in club3090) and `--lazy-mode`, `--log-jsonl` (present in mainline).

**Headline findings for this family**

- **`--no-mmap` no longer exists on the pinned mainline build — and every compose has now moved off it.** b10920 (`ghcr.io/ggml-org/llama.cpp@sha256:6ac92152…`, the `llama-cpp-local` pin since 2026-09-12) lists only `--load-mode` `[lcpp-help b10920]`, and its parser rejects the flag: `error: invalid argument: --no-mmap` (probed 2026-09-21, position-independent; a flag it *does* have fails on the VALUE, which is the positive control that makes the rejection specific). Three `llamacpp/deepseek-flash-*` slugs could not get past argument parsing between that pin bump and 2026-09-21 — **fixed in #1374** (`--load-mode none`, upstream's "no special loading mode"). The remaining **28 `llamacpp-club3090` composes** migrated in the follow-up: they still *worked*, because the fork accepts the flag with a DEPRECATED warning, but that is a fact about when the fork last rebased, not about the flag. Both club3090 pins (v1.5, v1.6) were verified to accept `--load-mode none` before the migration, and the rendered argv of one compose per pin was run against its own image. ⭐ **One holdout, and it must NOT be migrated:** `ik-llama/apex-fit-q8q5` (`🗑️ Deprecated`). Its compose pins `${IK_LLAMA_IMAGE:-…}`, a different fork from the engine profile it registers against. That image was pulled and probed rather than reasoned about: the pinned ik build **accepts `--no-mmap` and documents it un-deprecated**, and **has no `--load-mode` at all** (`error: unknown argument: --load-mode`). Migrating it would break a slug that works. ⚠️ Note ik's `--help` exits **1** normally, so the exit code is not the discriminator on this fork — the error text is; a probe keying on rc alone would call every ik flag broken. The guard skips these composes explicitly (it would otherwise test ik flags against the mainline binary) and will keep skipping them until **ik-llama gets an engine profile of its own** — it has none today, which is why its 15 slugs register against the *mainline* profile and then override the image. ⚠️ The earlier note here said `llama-cpp-local` should be "split"; that was wrong. `llama-cpp-local` **is** the mainline engine and is correctly used by 15 `llamacpp/*` slugs — the missing piece is an ik profile, not a split. ik is plainly a separate engine: our own ik composes pass `--merge-qkv` (14), `--fit` (10), `--fit-margin` (5) and `--cache-ram` (3), none of which mainline has, and the two forks disagree on `--no-mmap`/`--load-mode` in opposite directions.
- **`-t`/`--threads` is never passed as a flag.** 31 composes (28 club3090 + 3 mainline offload) export `LLAMA_ARG_THREADS` from `THREADS` inside the entrypoint, defaulting to `nproc/2` — the same floor `resolve_offload_threads` computes. The compose comments themselves say the measured optimum is an absolute ~24-32, so the launcher and the compose agree on a value both document as a floor, not an optimum.
- **`MOE_RESERVE_MB` never reaches a llama.cpp slug.** It is consumed as `GGML_CUDA_MOE_CACHE_RESERVE_MB=${MOE_RESERVE_MB:-1536}` (28 composes, name verified in the v1.6 binary), and `_moe_cache_env` scales it on >24 GB cards — but the launcher never calls the injector for `llamacpp-club3090/*`, and the key is missing from the launcher `case` allowlist (a fired row would `exit 2`; #1363). The `rtx-a6000.yml` note that says it scales here describes code that cannot run.
- **`--fit` means two different things.** Mainline/club3090: `--fit [on|off]`, default **on** — the engine auto-adjusts *unset* `-ngl`/`-c` to fit device memory; 31 offload composes pass `off` deliberately so the explicit `-ot`/`-c` stand. ik_llama: `--fit` is a bare boolean plus `--fit-margin N` (5 deprecated composes pass `--fit --fit-margin 256`). Same spelling, different grammar.
- **`-ot` is where hardware adaptation actually lives** for 10 residency slugs: `resolve_offload_residency` sizes `OT_G<i>` from free VRAM per card with an additive, field-calibrated model (`compose-meta.sh:396-428`), and the RAM gate prices exactly what it grants. The other 21 `-ot` composes pin *all* experts to CPU (`…exps\.weight=CPU`) — the config that runs anywhere.

### 6a. Placement, memory, threads

| Flag | Upstream status | Used by us | Hardware-gated? |
|---|---|---|---|
| `-ngl` / `--gpu-layers` | Layers in VRAM: number, `auto`, or `all` (default `auto`). `[lcpp-help b10920]` `[ik-src]` `[bee-src]` | 69/70 — `LIT 99` ×59, `LIT all` ×10 | `engine-auto` exists (`auto` + `--fit on`) but we opt out: full offload is hard-coded; the offload composes move experts with `-ot` instead. |
| `-c` / `--ctx-size` | Context size (0 = from model). `[lcpp-help b10920]` `[ik-src]` `[bee-src]` | 68/70 — `${CTX:-…}` ×33, `${CTX_SIZE:-…}` ×35 (values 65536 → 262144) | `bare 24 GB` ⚠ — on the offload slugs the KV pool competes with the expert cache for the same VRAM (`learnings/moe-cache-engine.md`); nothing re-sizes it per card. |
| `-b` / `--batch-size` | Logical max batch (default 2048). `[lcpp-help b10920]` | 70/70 — `${BATCH_SIZE:-4096}` ×21, `${UBATCH:-4096}` ×17, `${BATCH_SIZE:-2048}` ×13, `${UBATCH:-2048}` ×11, `:-1024` ×3, `LIT 4096` ×3 | `bare 24 GB` |
| `-ub` / `--ubatch-size` | Physical max batch (default 512); sets the compute-buffer peak. `[lcpp-help b10920]` | 70/70 — `${UBATCH_SIZE:-512}` ×23, `${UBATCH:-4096}` ×17, `${UBATCH_SIZE:-1024}` ×12, `${UBATCH:-2048}` ×11, `:-2048` ×2, `LIT 4096` ×3 | `bare 24 GB` ⚠ — the compute-buffer swing this sets is what `GGML_CUDA_MOE_CACHE_RESERVE_MB` was measured against (1 128 MiB at ub4096); changing one without the other is the documented 11 % regression. |
| `-np` / `--parallel` | Server slots (default −1 = auto). `[lcpp-help b10920]` | 70/70 — `${NP:-1}` ×36, `${NPARALLEL:-1}` ×33, `${NP:-3}` ×1 | `bare` — the family exposes **no concurrency knob to the injector by design** (`_ENGINE_TYPE_CONCURRENCY_ENV` omits llama.cpp). Two different env names for the same knob. |
| `-t` / `--threads` | CPU threads for generation (default −1); env `LLAMA_ARG_THREADS`. `[lcpp-help b10920]` | **0 as a flag**; 31 via `LLAMA_ARG_THREADS=${THREADS}` in the entrypoint (28 club3090 + 3 mainline offload) | `launcher-bash:resolve_offload_threads` → `nproc/2` (CPU-count keyed, not VRAM). Explicit `THREADS` always wins. |
| `-ts` / `--tensor-split` | Fraction of the model per GPU, e.g. `3,1`. `[lcpp-help b10920]` `[ik-src]` `[bee-src]` | 41/70 — `LIT 1,1` ×13, `${TENSOR_SPLIT:-1,1,1,1}` ×9, `:-1,1,1,1,1,1,1,1` ×7, `:-1,1` ×5, `LIT 1,1,1,1` ×2, `:-0.55,0.45` ×4, `:-0.575,0.425` ×1 | `bare 24 GB` ⚠ — even splits assume matched cards; `launch.sh`'s topology advisory *recommends* `--tensor-split` for VRAM-mismatched rigs but injects nothing. The 15 `LIT` composes cannot be corrected without editing. |
| `-sm` / `--split-mode` | `none`/`layer`/`row`/`tensor` (default `layer`). `[lcpp-help b10920]`; ik_llama adds `graph` `[ik-src]` | 40/70 — `LIT layer` ×33, `${SPLIT_MODE:-layer}` ×6, `LIT graph` ×1 (ik) | `launcher-bash:preflight_offload_split_mode` **refuses** `row`/`tensor` under CPU offload (measured −85 % prefill) — a guard, not an adapter. |
| `--main-gpu` | GPU for KV/intermediates with `row`/`none`. `[lcpp-help b10920]` `[bee-src]` | 5/70 (beellama) — `${MAIN_GPU:-0}` | `bare` |
| `--max-gpu` | ik_llama only: cap on GPUs used. `[ik-src]` — **not in mainline** | 1/70 (deprecated ik) — `LIT 2` | Topology ⚠ — hard-coded 2 |
| `-ot` / `--override-tensor` | `<pattern>=<buffer type>,…`; env `LLAMA_ARG_OVERRIDE_TENSOR`. `[lcpp-help b10920]` `[ik-src]` | 31/70 — 21 static `blk\.[0-9]+\.ffn_(gate\|up\|down)_exps\.weight=CPU` (all experts on CPU; registry `offload: tensor-override`), 10 with leading `${OT_G0}…${OT_G<n>}` slots (registry `offload: residency`) | `launcher-bash:resolve_offload_residency` for the 10 residency slugs (7 club3090 + 3 mainline): `OT_G<i>` from detected free VRAM, ≥2 cards, outer-edge layer selection so experts sit on the card owning the layer. User `OT_G<i>` always wins. The 21 static ones: `bare`, but the safe direction. ⚠ `-ot` force-disables pipeline parallelism (engine property — `learnings/moe-cache-engine.md` §11). |
| `--moe-cache` | club3090 only: cache the hottest CPU-resident experts in spare VRAM — `auto` (preserve repacking) / `on` / `off` / `N` MiB per device. `[c3090-help v1.6]` — **not in mainline** | 28/70 — `${MOE_CACHE:-auto}` ×22, `${MOE_CACHE:-off}` ×6 | `engine-auto` (grants free − reserve per device) + `inject:MOE_RESERVE_MB` via `GGML_CUDA_MOE_CACHE_RESERVE_MB` — **unreachable** (headline). The binary also carries `GGML_CUDA_MOE_CACHE_MIN_CC` (an SM floor inside the engine; semantics not verified) `[c3090-bin v1.6]`. |
| `--cache-type-k` / `-ctk`, `--cache-type-v` / `-ctv` | KV dtype: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` (default f16). `[lcpp-help b10920]` `[ik-src]`; beellama adds pseudo-types `kvarn2…kvarn8` `[bee-src 98caf25]` | 48/70 — `${KV_TYPE:-q8_0}` ×14, `${KV_TYPE:-q4_0}` ×17, `${KV_TYPE_K:-q5_0}`/`${KV_TYPE_V:-q4_1}` ×7, `${KV_TYPE:-f16}` ×5, `${KV_TYPE_K:-kvarn4}` ×2, `${KV_TYPE_K:-q8_0}` ×1 | `bare` — quality floor, not hardware (`hardware/*.yml` lists them as supported on every class; `kvarn4` only on rtx-3090). |
| `-fa` / `--flash-attn` | `on`/`off`/`auto` (default auto). `[lcpp-help b10920]` `[ik-src]` `[bee-src]` | 70/70 — `LIT on` (59 as `-fa`, 11 as `--flash-attn`) | — (hard-coded on; every supported class has the CUDA FA path) |
| `--no-mmap` | **Removed in mainline b10920** — only `-lm`/`--load-mode {auto,none,mmap,mmap+mlock,…}` remains and the parser rejects `--no-mmap` `[lcpp-help b10920]` + probe. Still accepted as DEPRECATED on club3090 v1.5/v1.6 `[c3090-help v1.6]`; present in ik_llama `[ik-src]`. | **1/70** — only `ik-llama/apex-fit-q8q5` (🗑️ Deprecated), which is **correct**: its ik pin has no `--load-mode`. Was 32: the 3 mainline slugs fixed in #1374, the 28 club3090 composes migrated after. | Host RAM — `bare`. Pin-drift, not hardware. |
| `--cache-prompt`, `--cache-ram` | Prompt caching (default on); host-RAM prompt-cache cap in MiB (default 8192). `[lcpp-help b10920]` | 31 / 11 — bare; `LIT 0` ×10 (beellama), `${CACHE_RAM_MB:-4096}` ×1 | Host RAM — `bare` |
| `--kv-unified`, `--no-host`, `--no-context-shift` | Unified KV buffer across slots; bypass host buffer; no context shift. `[lcpp-help b10920]` `[bee-src 98caf25]` | 10 / 10 / 2 (beellama, deprecated) | — |
| `--fit` | Mainline: `[on\|off]` default on, adjust unset args to fit device memory (`--fit-target MiB0,MiB1,…` per device). `[lcpp-help b10920]` — ik_llama: bare boolean + `--fit-margin N` `[ik-src]` | 36/70 — `LIT off` ×31 (mainline/club3090 offload), bare ×5 + `--fit-margin 256` ×5 (ik, deprecated) | `engine-auto` — deliberately **switched off** on every offload compose so the explicit residency plan stands. |
| `-devd` / `--spec-draft-device` | Devices for the draft model (`none` = don't offload). `[lcpp-help b10920]` | 20/70 (club3090) — `${DRAFT_DEVICE:-CUDA1}` ×18, `LIT none` ×2 | `bare 24 GB` ⚠ — assumes a second card named `CUDA1`; the residency sizer charges the drafter's VRAM to that card via `CPU-Offload-Draft-Card` (#1233). |
| `-ngld` / `--spec-draft-ngl` | Draft layers in VRAM. `[lcpp-help b10920]` `[ik-src]` `[bee-src]` | 12 (ik, `LIT 99`) + 5 (beellama, `LIT all`) | `bare` |
| `-mmdev` / `--mmproj-device` | Device for the multimodal projector (`none` = don't offload; default auto); env `MTMD_BACKEND_DEVICE`. `[lcpp-help b10920]` `[prism-src]` | 23/70 — `${MMPROJ_DEVICE:-none}` | `bare` (projector kept on CPU to protect the expert pool) |
| `--mmproj` | Multimodal projector file. `[lcpp-help b10920]` `[ik-src]` | 27/70 | — |
| `--merge-qkv`, `-khad`/`-vhad`, `--recurrent-ckpt-mode` | ik_llama only: merge QKV; K/V-cache Hadamard rotation; recurrent checkpoint mode. `[ik-src baac291]` — **not in mainline** | 13 / 12 / 12 (all deprecated ik) | — |
| `--override-kv` | Override model metadata `KEY=TYPE:VALUE`. `[lcpp-help b10920]` `[bee-src]` | 2/70 — `gemma4.context_length=int:${CTX_SIZE:-262144}` | — |

### 6b. Speculative decoding

| Flag | Upstream status | Used by us | HW |
|---|---|---|---|
| `--spec-type` | Mainline: `none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache` (comma list). `[lcpp-help b10920]` `[c3090-help v1.6]`; ik_llama: stage syntax `mtp:n_max=…` `[ik-src]`; beellama: `dflash` `[bee-src 98caf25]` | 60/70 — `draft-dflash` ×18, `draft-mtp` ×14, ik `mtp:n_max=${SPEC_N…}` ×13, beellama `dflash` ×5, `draft-dspark` ×5, `ngram-mod` ×3, `ngram-map-k` ×1 — all under the `SPEC_N`/`SPEC=off` contract | — |
| `--spec-draft-n-max` | Draft tokens (default 3). `[lcpp-help b10920]` `[bee-src]` | 36/70 — `${SPEC_N:-3}` ×12, `${SPEC_N:-${MTP_DRAFT_N_MAX:-…}}` ×10, `:-2` ×6, `:-5` ×5, … | — (`spec-dec is single-stream-only on CPU offload` is a workload finding, not a card gate) |
| `-md` / `--spec-draft-model` (alias `--model-draft`) | Draft model file. `[lcpp-help b10920]` `[bee-src]` | 23 (`-md`, mainline/club3090) + 6 (`--spec-draft-model`, beellama/mainline) | — |
| `--spec-dflash-cross-ctx` | beellama only: DFlash cross-attention context. `[bee-src 98caf25]` — **absent from beellama HEAD `arg.cpp`** (removed after v0.3.2) and from mainline | 5/70 (deprecated beellama) — `${CROSS_CTX:-1024}` | — |
| `--spec-ngram-mod-n-min` / `-n-max` / `-n-match` | ngram-mod parameters (defaults 1 / 64 / 24). `[lcpp-help b10920]` | 3/70 (club3090) — `${NGRAM_N_MIN:-1}`, `SPEC_N`, `${NGRAM_N_MATCH:-24}` | — |

Script-body artefacts excluded: `--model-draft`, `--spec-draft-n-min`, `--spec-draft-device`, `--mmproj-device` appear in every drafter entrypoint only as `case` patterns that *strip* those flags when `SPEC=off`; `--include`/`--local-dir` are hf-download hints in an `echo`.

### 6c. Model, API and sampling (not hardware-related)

| Flag | Upstream status | Used by us |
|---|---|---|
| `-m` / `--model`, `--alias`, `--host`, `--port 8080` | `[lcpp-help b10920]` `[ik-src]` `[bee-src]` | 70 / 41 / 70 / 70 |
| `--jinja`, `--chat-template-file`, `--chat-template-kwargs` | `[lcpp-help b10920]` `[ik-src]` | 37 / 22 / 21 |
| `--reasoning`, `--reasoning-format`, `--reasoning-effort`, `--reasoning-budget`, `--reasoning-budget-message`, `--reasoning-preserve`/`--no-…` | `[lcpp-help b10920]` (`--reasoning-effort`, `--reasoning-preserve` are mainline-only — not in ik `[ik-src]`) | 42 / 48 / 18 / 7 / 4 / 4 |
| `--parallel-tool-calls` | ik_llama only `[ik-src]` | 12 (deprecated ik) |
| `-n` / `--predict` | `[lcpp-help b10920]` | 18 — `${MAX_TOKENS:-65536}` |
| `--image-min-tokens`, `--image-max-tokens` | `[lcpp-help b10920]` `[ik-src]` | 4 / 3 |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--repeat-last-n`, `--presence-penalty`, `--dry-multiplier`/`--dry-base`/`--dry-allowed-length` | `[lcpp-help b10920]` `[ik-src]` `[bee-src]` | 63 / 63 / 42 / 45 / 29 / 2 / 8 / 2 |

### 6d. llama.cpp-family environment knobs (names verified in the v1.6 binary, `[c3090-bin v1.6]`)

| Env | Composes | HW |
|---|---|---|
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` = `${MOE_RESERVE_MB:-1536}` | 28 | `inject:MOE_RESERVE_MB` — unreachable (headline); 1536 was the 24 GB sweep optimum, 1024 measured ~11 % slower |
| `GGML_CUDA_MOE_CACHE_ADMIT_AFTER` = `${MOE_ADMIT_AFTER:-64}`, `GGML_CUDA_MOE_CACHE_STATS`, `GGML_EXPERT_PREFETCH`, `GGML_EXPERT_PREFETCH_SLOTS` | 28 | `bare` |
| `LLAMA_ARG_THREADS` (entrypoint, from `THREADS`) | 31 | `launcher-bash:resolve_offload_threads` |
| `LLAMA_ARG_LOG_VERBOSITY`, `LLAMA_ARG_CACHE_REUSE`, `LLAMA_GRAPH_REUSE_DISABLE`, `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` | 28 / 28 / 28 / 4 | — |
| `GGML_OP_OFFLOAD_MIN_BATCH` | 0 — present in the binary; **must not be set** on a cache compose (silently prevents the cache allocating, −4.2× — engine profile notes) | — |
| `IK_LLAMA_IMAGE` | 15 ik composes | `launcher-bash:preflight_ik_llama_image` — driver CUDA < 13.2 → `ghcr.io/ikawrakow/ik-llama-cpp:cu12-server-4574` (cu13 forward-compat error 804 → CPU fallback crash loop, #633) |

---

## 7. Cross-engine summary — the intersection this page exists for

> **§7 refreshed 2026-09-21 against `4dc310a6`** (`master` after #1373); §§1–6 are still as generated at `3126bc31`. #1369 and #1373 landed between the two and changed exactly the rows below — nothing in §§1–6 moved, because none of it is about injection.

Flags that are hardware-dependent in reality and hard-coded (or hard-tuned) in our composes:

| Engine | Flag | Why it depends on the card | What adapts it today |
|---|---|---|---|
| vLLM | `--max-num-seqs` on 10 `LIT` slugs | KV pool ∝ VRAM | nothing (env var absent from those 10) |
| vLLM | `--max-model-len` | KV pool ∝ VRAM | nothing (deferred by design) |
| vLLM | `--max-num-batched-tokens`, `--long-prefill-token-threshold` | prefill activation peak vs free VRAM (`CLIFFS.md`) | nothing |
| vLLM | `--kv-cache-memory-bytes` (3) | absolute bytes, overrides the fraction | nothing |
| vLLM | `--attention-backend FLASH_ATTN` (7) | FA3 exists on sm_90 | nothing |
| vLLM | `--kv-cache-dtype` | backend route, **not** card class | the compose default, full stop. The #246 Phase 1 card-keyed injector was **retired in #1371** — it had been inert on every card since the composes migrated to `fp8_e4m3`. What actually decides e4m3-on-Ampere is whether the checkpoint routes to FlashInfer or Triton, encoded as `_fp8w_ampere_kv` in `compat.py`'s C5 gate |
| SGLang | `--tp-size`, `--disable-custom-all-reduce` (11 each) | card count; NVLink/P2P | nothing — no `${TP}`, no `detect_nvlink.sh`. An NVLink rig runs SGLang with the kernel off |
| SGLang | `--mem-fraction-static` | VRAM, but **not the same quantity as vLLM's utilisation** (static weights+KV share vs total-VRAM budget) | ✅ **`MEM_FRACTION`, reachable since #1369** and keyed per engine type, so vLLM's spelling can never land here. Fires on 6 slugs on unified-memory cards. ⚠️ the value is the card's own `mem_util_safe` ceiling, downward-only — **unvalidated on SGLang** (no sgl soak at a clamped fraction) |
| SGLang | `--max-running-requests` | VRAM | key mapping exists (`MAX_RUNNING_REQUESTS`, #1362) and the launcher accepts it, but **inert: no `envelopes.yml` row for any sgl slug**. Do not add one from a vLLM measurement — envelopes require a `validated` or `computed` basis |
| SGLang | `--context-length`, `--chunked-prefill-size` | VRAM | nothing |
| exllamav3 | `--cpu-moe-split-experts` | it *is* a VRAM fit (#1360) | ✅ **`resolve_cpu_moe_split` (#1373)** — a fit, not a fraction, from the safetensors tensor table + a calibrated per-card reserve. ⚠️ one calibration point per tier, so other rigs are labelled extrapolations at boot |
| exllamav3 | `--autosplit-reserve` ×2 entries | per-GPU list | nothing — 2-card literal |
| llama.cpp | `-c`, `-ub`, `-b` on offload slugs | KV + compute buffer share VRAM with the expert cache | nothing |
| llama.cpp | `-ts` even splits (15 `LIT`) | mismatched VRAM | advisory only |
| llama.cpp | `-devd CUDA1` (18) | needs a second card | nothing |
| club3090 | `GGML_CUDA_MOE_CACHE_RESERVE_MB` | allocator working room ∝ card | ✅ **`MOE_RESERVE_MB`, reachable since #1369** — 22 moe-cache slugs above 24 GB. ⚠️ upward-only **safety heuristic, not a tuned optimum**: measured on 24 GB Ampere only, and it preserves the reference rig's reserve/VRAM ratio. The boot line says so |
| all | `THREADS` = `nproc/2` | CPU count (documented optimum is absolute 24-32) | `resolve_offload_threads` (a floor) |

### Injection reach, measured on `4dc310a6`

`resolve_variant_pin` is now **total: 138 of 138 slugs resolve, 0 raise** (it was 65 reachable / 73 raising at `3126bc31`, because the launchers gated the whole call behind a `vllm/*｜beellama/*` prefix test). Slugs receiving at least one **non-image** env, by rig:

| Rig | Slugs | Keys |
|---|--:|---|
| 2×3090 | 2 | `GPU_MEMORY_UTILIZATION`, `DECODE_GRANULARITY` |
| 2×5090 | 46 | `MOE_RESERVE_MB` 22 · `VLLM_USE_DEEP_GEMM` 22 · `MAX_NUM_SEQS` 2 · `DECODE_GRANULARITY` 1 |
| 2×A6000 | 27 | `MOE_RESERVE_MB` 22 · `MAX_NUM_SEQS` 3 · `GPU_MEMORY_UTILIZATION` 1 · `DECODE_GRANULARITY` 1 |
| 2×H100 | 23 | `MOE_RESERVE_MB` 22 · `DECODE_GRANULARITY` 1 |
| DGX Spark | 82 | `GPU_MEMORY_UTILIZATION` 49 · `MOE_RESERVE_MB` 22 · `VLLM_USE_DEEP_GEMM` 22 · `MEM_FRACTION` 6 · `MAX_NUM_SEQS` 1 · `DECODE_GRANULARITY` 1 |

⭐ **The reference rig is the one that gets almost nothing**, which is the honest reading of these numbers: the knobs were tuned *on* 2×24 GB Ampere, so there is nothing to correct there. Everything above is the correction other hardware was silently not getting. Verified additive across 966 (slug, rig) pairs before/after the flip — nothing removed, no pre-existing value changed, and the effective *image* byte-identical for all 138 slugs.

### What **does** adapt, for contrast

`--disable-custom-all-reduce` (vLLM, boot-detect) · `-ot` residency slots on 10 slugs (free-VRAM fit) · `--cpu-moe-split-experts` (exl3 VRAM fit) · `MOE_RESERVE_MB` on 22 moe-cache slugs above 24 GB · `MAX_NUM_SEQS` / `MAX_RUNNING_REQUESTS` per engine dialect · `GPU_MEMORY_UTILIZATION` / `MEM_FRACTION` downward on unified-memory cards · `VLLM_USE_DEEP_GEMM` on sm 8.9/12.x · the ik-llama cu12 image swap · `--fit` / `--gpu-split-auto` / `--moe-cache auto` inside the engines · every SM-floor refusal.

### Tracker state

| Issue | State |
|---|---|
| #1361 injection reach (umbrella) · #1363 matrix guard · #1364 detector VRAM · #1365 image pin vs injection · #1366 exl3 `MOE_SPLIT` fit | ✅ all closed — merged as #1367, #1368, #1369, #1373 |
| ~~#1370 `--no-mmap` boot-blocks 3 `llamacpp/deepseek-flash-*` slugs~~ | ✅ closed — #1374, plus `test-compose-flags-accepted` for the class |
| ~~#1371 `KV_CACHE_DTYPE` pilot inert on every card~~ | ✅ closed — retired, not repointed |

**Pin-drift findings surfaced while sourcing rows** (not hardware, but the same "we pass it, does the pinned engine still take it?" question): ~~`--no-mmap` rejected by the mainline b10920 pin on 3 `llamacpp/deepseek-flash-*` slugs~~ — **fixed in #1374** (`--load-mode none`), and the whole class is now guarded by `test-compose-flags-accepted`, which checks every compose's flags against its own pinned image's `--help` (33 composes, 9.2 s). the 28 club3090 composes that still passed it have since migrated too; the one remaining user, `ik-llama/apex-fit-q8q5`, stays as-is because its ik pin has no `--load-mode` (probed, not assumed); `--prefix-match-unit` exists only from v0.29.0 (§3, safe today); `--kv-cache-dtype int8_per_token_head` is overlay-supplied, not stock (§3); `VLLM_ATTENTION_BACKEND` is not read by v0.29.0 (§3d); `--spec-dflash-cross-ctx` is gone from beellama HEAD but present at the pinned commit (§6b); the registry's `max_num_seqs`/`mem_util` disagree with what 10 SGLang composes ship (§4). Only the first had a tracker row, and it is now closed.

⚠️ **This section will rot the same way the rows above did.** It describes *injection*, which changes whenever a resolver or a launcher gate does — unlike §§1–6, which change only when an engine pin moves. Re-derive the reach table with `resolve_variant_pin` across the registry rather than trusting the numbers here.
