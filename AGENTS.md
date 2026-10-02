# Agent guide

Guidance for AI coding agents (Claude Code, Cursor, Copilot, Continue, etc.) working in this repo. Focused — only conventions an agent wouldn't infer from the code itself.

> **One file, two names:** this is `AGENTS.md` (canonical); **`CLAUDE.md` is a symlink → it.** Edit either — both names resolve to the same guide so any agent that looks for either finds it, and the two can't drift.

## Read first

Before making non-trivial changes:

- [`README.md`](README.md) — what the repo is, two-routes framing, repo layout
- [`docs/README.md`](docs/README.md) — **the docs index** (user track + contributor track); start here to find any guide
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — current stack state (services, ports, paths, model + KV config)
- [`docs/SINGLE_CARD.md`](docs/SINGLE_CARD.md) / [`docs/DUAL_CARD.md`](docs/DUAL_CARD.md) — pick-by-workload guidance + the cliffs they reference
- [`docs/ADDING_MODELS.md`](docs/ADDING_MODELS.md) — add a model: serve-locally vs the curated-catalog workflow (see "Adding a model" below)
- [`docs/UPSTREAM.md`](docs/UPSTREAM.md) — every upstream issue / PR we depend on or have filed (see "Upstream issues" section below for why this matters)
- [`models/qwen3.6-27b/INTERNALS.md`](models/qwen3.6-27b/INTERNALS.md) — DFlash forensics, AutoRound rationale, Marlin pad fork, MTP head
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — what kinds of PRs land cleanly + benchmark + verify protocol

## Hardware truths

**The reference rig** (the maintainer's, where all first-party numbers come from): 2× RTX 3090 Ampere SM 8.6, PCIe-only, **no NVLink** (and we won't add it). Its constraints:

- Custom all-reduce must be disabled in vLLM/SGLang configs on PCIe-only multi-GPU topologies (NVLink-assumed paths break).
- **Ampere (sm_86) has no native FP8 compute** — there, FP8 KV is a storage optimization only. On Ada/Hopper/Blackwell, FP8 compute paths are real (fp8-weights composes are 5090-safe via the launcher's `VLLM_USE_DEEP_GEMM` pass-through — an arch-conditional guard, not a blanket "everything works on 5090").
- Speculative decoding using EAGLE / DFlash **inside vLLM** is blocked on Qwen3-Next family on every arch (DeltaNet rollback). MTP works, and DFlash works via the beellama engine (external drafter). See [`docs/UPSTREAM.md`](docs/UPSTREAM.md) — vllm#39931.

**The rig you're running on may not be the reference rig.** Supported hardware classes live in `scripts/lib/profiles/hardware/*.yml` (3060 → 5090, A5000, A100, H100, DGX Spark, …); the profile-catalog compat layer and launchers key on the *detected* class and inject arch-aware env. Before assuming any constraint above applies, check which class you're on (`nvidia-smi` + the matching hardware YAML) — and never hand-copy a reference-rig workaround (e.g. disabling all-reduce) onto an NVLink-equipped or non-Ampere rig without checking it's still warranted.

## Working on a rig — yours or a user's

Most agent sessions on a rig are "make it serve / find out why it doesn't", not repo changes. What keeps a rig healthy:

- **Change behaviour through settings, never by editing tracked files.** A saved setting survives `git pull`; an edited compose conflicts with the next pull and misleads the next bug report. In order of preference: a one-off shell prefix (`KV_OFFLOAD_GB=64 bash scripts/switch.sh <slug>`) → a setting for one slug (`bash scripts/switch.sh --set <slug> KEY=VALUE`) → a global setting (`bash scripts/settings.sh set KEY=VALUE`) → a model or compose of your own in the gitignored local layer ([`docs/ADDING_MODELS.md`](docs/ADDING_MODELS.md), "Run a local GGUF without the catalog" / the local layer).
- **Where per-machine state lives** — outside the checkout since #1466; the repo `.env` is legacy (still read, last; never write it):

  | What | Where |
  |---|---|
  | settings | `~/.config/club-3090/club3090.env` |
  | tokens and keys (HF token, gateway key, cloud route keys) | `~/.config/club-3090/secrets.env` (0600) — `settings.sh set` files credential-looking names there itself |
  | settings for one slug | `~/.config/club-3090/slugs.json` (`switch.sh --set / --unset`) |
  | the gateway's own routes | `~/.config/club-3090/litellm/config.local.yaml` |
  | compile caches | `~/.cache/club-3090/<engine image>-<image id>/`, shared by every checkout |
  | KV-offload disk tier | `~/.local/share/club-3090/kv-offload` (`KV_OFFLOAD_DIR` overrides; vLLM's tier has no size cap) |

  `CLUB3090_CONFIG_DIR` / `CLUB3090_CACHE_DIR` / `CLUB3090_DATA_DIR` move them (the `XDG_*` variables too).
- **Precedence, highest first:** the shell > this slug > the model's pin > `club3090.env` > `secrets.env` > the repo `.env` > the compose default. A variable exported in the shell wins **even when empty**, so check `env` before blaming a file. Values are literal: no `$HOME` or `~` expansion.
- **Never print a secret.** `settings.sh show` hides them — don't add `--show-secrets` to anything whose output is shared or logged, never `cat secrets.env`, and when you call the gateway send the key on stdin (`curl -H @-`), not on the command line. `gateway-key.sh status` answers "default or own key, and does the running gateway accept it?" without printing it.
- **Running `docker compose` yourself** reads none of the settings: `f="$(bash scripts/settings.sh compose-env-file)"; docker compose --env-file "$f" -f <compose> up -d; rm -f "$f"`. Without them a compose also falls back to in-repo cache paths.
- **Worktrees** share settings and compile caches with the main checkout: a boot from a worktree needs nothing exported and leaves nothing root-owned. Stop its container before removing the worktree — the container bind-mounts files from it.

| Question | Command |
|---|---|
| What settings are in effect, and where does each come from? | `bash scripts/settings.sh show` (`path` lists the files) |
| Why does this slug launch with this value? | `bash scripts/switch.sh --explain <slug>` (`--json` for scripts) |
| What's running, on which ports; GPU / RAM / disk? | `bash scripts/gpu-mode.sh status` |
| Is the running server healthy right now (container, KV pool, spec-decode, recent errors)? | `bash scripts/health.sh` (finds the running container like `verify.sh`; `URL=` / `CONTAINER=` override) |
| Does it work (tools, long context, …)? | `bash scripts/verify.sh`, then `bash scripts/verify-full.sh` |
| Does this slug fit this hardware and catalogue? | `bash scripts/diagnose-profile.sh <slug>` |
| Is the gateway on a key of its own, and does the running one accept it? | `bash scripts/gateway-key.sh status` |
| Recreate only the gateway (after a key, route or route-key change) | `bash scripts/gpu-mode.sh gateway` |
| What does the gateway send and receive? | `bash scripts/litellm-log.sh on` … `off` |
| Is the prompt cache hitting? | `bash scripts/cache-share.sh` |
| Where are the compile caches and the KV disk tier, and how big? | `bash scripts/settings.sh caches` |
| A redacted bundle to paste into an issue | `bash scripts/report.sh` (`--full` adds verify / stress / soak / bench) |

## Upstream issues — single source of truth

`docs/UPSTREAM.md` tracks every upstream issue / PR we depend on, have filed, or use as context. **Before filing a new upstream issue or referencing an existing one in code/docs, check this file.** When status changes (issue closed, PR merged, pin bumped), update the row.

This rule exists because we previously had upstream links scattered across CHANGELOG, INTERNALS, FAQ, and per-compose comment headers — and they drifted. The tracker file is the canonical place; cross-link to it from anywhere else.

When filing a fresh upstream issue from this work:
1. Add the row to `docs/UPSTREAM.md` first
2. Link the issue back to `noonghunna/club-3090` in the body (helps maintainers see affected user surface)
3. If the upstream eventually merges + propagates, update the row to ✅ Resolved and bump the relevant pin (Genesis commit, vLLM nightly, etc.) in the same commit / PR

## Conventions on this repo

### Bench protocol
3 warm + 5 measured runs. Canonical prompts: 800-word essay (narrative, max_tokens=1000) + quicksort code (max_tokens=800). Sampler: **`temperature=0.6, top_p=0.95, top_k=20, min_p=0.0`** — all four are sent EXPLICITLY by `bench.sh`, and it prints them at start (`[bench] sampler: …`).

> ⚠️ **Why all four, explicitly (#962):** until 2026-08-12 only `temperature` and `top_p` were sent, so `top_k`/`min_p` fell through to **per-engine defaults** — llama.cpp applies `top_k 40 / min_p 0.05`, vLLM `top_k` off / `min_p 0`. The "canonical" protocol therefore resolved *differently on each engine*, which defeats the cross-engine `BENCHMARKS.md` table it exists for, and matched its own documentation on neither. Throughput is essentially sampler-insensitive at fixed `max_tokens`, so historical TPS rows remain comparable — but **quality-shaped results measured before this change were taken under different sampling** and should not be diffed against post-change runs. Override for sampler A/Bs with `BENCH_TEMP` / `BENCH_TOP_P` / `BENCH_TOP_K` / `BENCH_MIN_P`.

Capture both wall-time TPS and engine-internal `gen throughput` from logs. **Always capture per-card peak VRAM** alongside TPS.

### Genesis opt-in env vars
**Status (2026-07-06): no shipped compose currently enables Genesis** — the Genesis-pinned production paths were retired (their composes live under `compose/_archive/`). The guidance below stays because it applies verbatim if a Genesis-pinned compose is reintroduced, and the incident it encodes is the canonical example of why behavioral patches need repro-gating.

Genesis ships ~50 env-gated patches. Some are **targeted bugfixes** (P64 streaming, PN8 memory savings, P3/P5/P6 KV); others are **behavioral mitigations** that silently rewrite the request (P68 = `tool_choice → required`, P69 = inject "must use tool" reminder). Behavioral mitigations need a streaming + large-prompt repro before shipping default-on. We learned this the hard way on 2026-04-29 — see [`docs/UPSTREAM.md`](docs/UPSTREAM.md) → Genesis #9 row + the [club-3090 #2 thread](https://github.com/noonghunna/club-3090/issues/2#issuecomment-4346740245).

If you're considering enabling a new Genesis env var by default in a shipped compose:
1. Read the patch's header docstring in `vllm/_genesis/wiring/`. Does it modify `request.tool_choice`, `request.messages`, or rewrite output?
2. If yes (behavioral): run a streaming repro with prompt > the patch's threshold + a casual user message ("hi") + `tool_choice: auto`. If `finish_reason=stop` with empty content, don't ship default-on.
3. Pure bugfixes (no behavioral override) are fine to ship default-on once they pass `verify-full.sh`.

### Engine image pinning
Pin engine images when **either** trigger holds: (1) we vendor patches into the running container, or (2) the engine's rolling tag has burned us with a regression (a crash-loop or behavior break shipped under the same tag we'd validated — llama.cpp earned this pin via #187). Otherwise track upstream's rolling tag and accept that the compose YAML may need maintenance when upstream changes flags.

**Why:** patches hook into specific upstream code paths — a silent upstream change drifts those hooks and breaks the patched container in production. Pinning ensures the bytes we tested against are the bytes users get. Unpatched engines have no such hook, so upstream changes are upstream's problem to fix (or the YAML's, which is cheap to maintain) — *unless* the tag itself has proven mutable-under-validation, which is trigger (2).

**Pins are a maintenance liability — bump them on a cadence.** A stale stability-pin silently costs quality, not just features: the b9246→b9967 llama.cpp A/B (2026-07-11, #680) was think-OFF-neutral but **+4 think-ON** with 3 scenario-level flips. `scripts/engine-pin-bump.sh <engine-id> <tag> --check` does the mechanical half (spec + compose defaults + a bucketed report of fixtures/claims needing hands); the judgment half (re-validation, live boot) stays with the pin-bump checklist.

| Engine | Patches we vendor | Tag policy |
|---|---|---|
| `llama.cpp` | none | pinned **build tag** (e.g. `server-cuda-b9967`) — trigger (2): stability-pinned after the #187 rolling-tag crash-loop; zero patches, so bumps are cheap (`engine-pin-bump.sh llama-cpp-local <tag>`) |
| `vLLM` | Marlin pad, KV/loader overlays — `scripts/lib/profiles/patches.yml` is the source of truth | pinned **release tag** (e.g. `v0.22.0`, `v0.24.0`) — **never `nightly-*`/`latest`**: upstream purges nightly tags, orphaning the pin |
| `beellama.cpp` | none (the fork *is* the delta) | **digest-pinned** to Anbeeld's official image (`engines/beellama-local.yml` `install.spec`) |
| `SGLang` | per-compose decision (pin if we vendor a patch, rolling otherwise) | per-compose |

When adding the first vendored patch to a previously-rolling engine: pin in the same commit. When dropping the last patch: unpin in the same commit — unless the engine holds a trigger-(2) stability pin, which outlives its patches. Bump pins via PR with a `verify-full.sh` + `bench.sh` re-run, never silently.

**Delivery model (vLLM):** patches reach the container by **volume-mounting into the pinned *stock* `vllm/vllm-openai` image** (python sidecars / site-package overlays / install scripts — see `delivery_mechanism` in `scripts/lib/profiles/patches.yml`), **not** by baking a custom image. The older baked-image path (`ghcr.io/noonghunna/vllm-club3090`, which shipped the release images through `club-v0.8.3`) is **retired** — no compose or engine-pin references it, and the `dockerfile_bake` `delivery:` block in `patches.yml` is legacy/test-only. The GHCR package is kept as historical release artifacts (users pinned to a `club-v0.8.x` tag can still pull); it is not deleted and not produced by anything in-repo.

### File encoding — UTF-8 mode is the guarantee; `encoding="utf-8"` is the backstop

Repo sources are full of unicode (`— × → ⚠` in compose headers), and Python decodes file reads, stdout **and `sys.argv`** with the **locale** codec. On a rig whose locale is neither UTF-8 nor C that broke most of the script layer (#779: 46 pass / 38 fail on a real `en_US.ISO-8859-1`).

**The systemic fix — every script that runs python3 carries this, above its first call:**

```bash
export PYTHONUTF8="${PYTHONUTF8:-1}"
```

Python's UTF-8 mode (PEP 540) overrides the locale for reads, writes, stdout and argv in one move. Exported, so nested scripts and child processes inherit it. **Defaulted, not forced** (`:-1`, never `=1`) so a user who deliberately sets `PYTHONUTF8=0` keeps control. `test-locale-utf8.sh` enforces all of that — presence, placement above the first call, and default-not-force — plus a functional leg on a real single-byte locale built with `localedef`. **Add the line when you add a script that shells out to python3**; the gate is there because this invariant is what decays.

**⚠️ `LC_ALL=C` is NOT the at-risk case, and this guide used to say it was.** Python auto-enables UTF-8 mode for the C/POSIX locale, so C-locale rigs were always fine:

| environment | result |
|---|---|
| `LC_ALL=C` | `utf8_mode=1  stdout=utf-8` — protected |
| `LC_ALL=C PYTHONCOERCECLOCALE=0 PYTHONUTF8=0` | `utf8_mode=0  stdout=ascii` — synthetic repro only |
| a real `en_US.ISO-8859-1` / `de_DE.iso88591` | `utf8_mode=0  stdout=iso8859-1` — **the genuine at-risk case** |

Reproduce with a **real** single-byte locale (`localedef -f ISO-8859-1 -i en_US "$dir/en_US.ISO-8859-1"` + `LOCPATH`), not `PYTHONUTF8=0`. And note the failure shape differs: a single-byte locale **decodes any byte happily and corrupts quietly** — only the encode side raises. Compare output byte-for-byte across locales; don't probe for one character.

**Still write `encoding="utf-8"` explicitly** — belt and braces if the env var is ever overridden, and mandatory in these cases:

- **Writes, always paired with their reads.** `Path.write_text()` opens with mode `w`, which **truncates on open** — so pinning a read without pinning its paired write converts a clean crash into **data loss** (#777/#780: a `BENCHMARKS.md` copy went 180693 bytes → 0). For anything overwriting a file that matters, write a temp sibling and `os.replace()`: atomic, and a failed encode leaves the original intact.
- **`subprocess` output.** `subprocess.run(..., text=True)` decodes the child's stdout with the locale codec too. `patch_attribution.py` pinned every one of its own reads, was fully compliant with this rule, and still died reading `docker compose config` (#781). Pass `encoding="utf-8"` on the call.
- **`sys.argv`.** Under a non-UTF-8 locale argv is decoded with ASCII + `surrogateescape`, so unicode arguments arrive as *lone surrogates* that no `encoding=` pin can write. Recover with `os.fsencode(sys.argv[i]).decode("utf-8", "replace")` — correct under any locale — or pass long unicode payloads via stdin/file (#777).
- **Emit blocks that print unicode to a pipe:** `sys.stdout.reconfigure(encoding="utf-8")`.

Other corollaries:
- **The launcher table path is python-STDLIB-ONLY** — no PyYAML, no pip deps (community VMs ship bare python3; #584's `ModuleNotFoundError: yaml`). The `--json` contract path may require PyYAML but must fail with a `Fix:` hint, not a traceback. `test-registry-emit-no-yaml` guards both plus the locale cases.
- **A `.py` tool invoked directly** (`python3 tools/kv-calc.py`) gets no shell script to export the var. In-repo callers all go through the scripts; a user doing this on a single-byte locale is still exposed.
- **Don't blind-`2>/dev/null` launcher derive paths** — swallowing the traceback hid this exact class for months; capture stderr and surface it on failure instead.

### CHANGELOG
- `CHANGELOG.md` (cross-cutting) and `models/<name>/CHANGELOG.md` (per-model) are **append-only history**. Don't rewrite past entries even when a finding is superseded — add a new entry. The historical trail is load-bearing for "why did we do X."
- Old entries can reference files / patches that no longer exist. That's fine — leave them.

### Compose variants
- Each variant ships with a **header table** comparing it against sibling composes in the same directory. Update both directions when adding/removing variants.
- Keep the variant set lean. Three composes that overlap badly (similar TPS + similar context + same KV) is worse than two with a clean differentiation. Removed `fast-chat.yml` 2026-04-29 for exactly this reason.

#### Compose layout — `<model>/<engine>/compose/<topology>/<quant>/<serving>.yml`

The directory hierarchy encodes model, engine, topology, and the weights artifact. The filename encodes only the serving stack. No level repeats information from another.

| Path level | Encodes | Examples |
|---|---|---|
| `models/<model>/` | Model | `qwen3.6-27b` · `gemma-4-31b` |
| `models/<model>/<engine>/` | Inference engine | `vllm` · `llama-cpp` · `sglang` |
| `compose/<topology>/` | Hardware topology | `single` · `dual` · `multi3` · `multi4` · `multi8` |
| `<quant>/` | Weights artifact / `weights_variant` slug | `autoround-int4` · `ubergarm-iq4ks` · `awq` |
| `<serving>.yml` (filename) | Serving stack | `fp8-mtp.yml` · `turbo.yml` · `dflash.yml` · `int8.yml` |

**Topology rule**: `single` (TP=1) and `dual` (TP=2) have no count ambiguity. `multi<N>` requires the count because N varies (3 / 4 / 5 / 6 / 8). Aligns with `docs/SINGLE_CARD.md` / `DUAL_CARD.md` / `MULTI_CARD.md` doc framing.

**Default rule**: there is no filesystem default and no `default.yml`. Defaults live in `scripts/lib/profiles/registry.yaml` (`defaults:`; flattened to the `DEFAULTS` map by the `compose_registry.py` loader) and are selected by registry tag (`scripts/launch.sh`, `scripts/switch.sh`, estate planner). Direct Docker use must pass `-f <path>`.

**Default-resolver knobs** (maintainer-owned, in `registry.yaml`'s `defaults:` / `engine_preference:` / `recommended_default_models:` sections):
- `DEFAULTS[(model, engine, topology)] → slug` — the `<engine>/default` map (club-3090's recommended config per engine; reason can evolve, edited by PR).
- `ENGINE_PREFERENCE[topology] → [engine, …]` — the curated `<model>/default` policy. The resolver walks this list and picks the first engine with a **functional** (`status ∉ {experimental, preview, upstream-gated, deprecated}`) `DEFAULTS` entry. **Reorder a row to change a recommendation — no code change, any topology.** single = `[llamacpp, vllm]`; dual/multi = `[vllm, llamacpp]`. **`beellama` was removed from every walk 2026-07-27** (engine retired: Anbeeld #98 won't-fix; all 10 slugs `deprecated`, launchable by name with `--force`), and **`ik-llama` later** (its `qwen3.6-27b` slugs are `deprecated`). The single-card default for `qwen3.6-27b` is `vllm/minimal` (the only `DEFAULTS` entry the single walk finds); **`gemma-4-31b` single-card has NO functional default** (resolver honestly degrades to "pick explicitly") until a mainline Gemma single compose lands — see the retire plan in the stack tracker.
- `RECOMMENDED_DEFAULT_MODELS` — a **short opt-in shortlist** (today `["gemma-4-31b"]`; read `recommended_default_models:` in `registry.yaml` rather than trusting this line) of models eligible to be the *bare-`launch.sh`* default (first installed → its `<model>/default`). **NOT** an exhaustive ranking; absent models are runnable by name but never auto-default; **new models are NOT auto-added** — promote one explicitly.
- The shared resolver `model_default_target(root, model, topology)` (in `registry-emit.sh`) is the single injection point for both launchers. Precedence: `--variant` → user pin (`CLUB3090_DEFAULT_<MODELID, non-alnum→_>`, read from the environment the settings loader filled: shell > `~/.config/club-3090/club3090.env` > `secrets.env` > legacy repo `.env`) → community seam (`community_default_target` → `None` today) → curated walk → degradation (nearest-lower topology, else "pick explicitly"). `X/default` dispatch: `X ∈ engine-set` → engine rec; `X ∈ model-set` → model default; else error. Users pin/clear via `switch.sh --set-default <slug>` / `--clear-default <model>` (saved in `club3090.env`; clear also drops a legacy copy from the repo `.env`).

**Feature suffix order** (when stacking): interconnect → drafter → KV → vision modifier. Examples:
- `dual/autoround-int4/turbo.yml` — TP=2 + AutoRound INT4 weights + TurboQuant KV
- `dual/autoround-int4/dflash.yml` — TP=2 + AutoRound INT4 weights + DFlash drafter
- _(the `nvlink-` interconnect prefix is reserved but currently unused — NVLink is auto-detected at boot via `NVLINK_MODE`, not encoded in the filename)_
- `dual/autoround-int4/int8.yml` — TP=2 + AutoRound INT4 weights + INT8 PTH KV
- `dual/awq/bf16-mtp.yml` — TP=2 + AWQ weights + BF16 KV + MTP
- `multi4/autoround-int4/dflash.yml` — TP=4 + DFlash

**Plain / default cases**: a compose with the engine-**default** KV and no drafter is `base.yml` — do NOT name the default KV (no `bf16.yml` when bf16 is the default; that's `base.yml`); name only a **non-default** KV (`int8.yml`, `fp8.yml`, `tq3.yml`). **Workload-tuned** variants (use-case ctx/sampling, not a feature delta) keep a descriptive name (`long-text.yml`, `tools-text.yml`, `bounded-thinking.yml`, `minimal.yml`). Names predating this are grandfathered — don't rename (it re-paths the registry `compose_path`). Full filename + registry-slug conventions: [`docs/ADDING_MODELS.md`](docs/ADDING_MODELS.md) → "Build the first compose."

Concrete examples (checked against the tree 2026-09-28 — `switch.sh --list` is the live list):
- `models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml` — Qwen single-card default (`vllm/minimal`)
- `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml` — Qwen dual default (`vllm/dual`, fp8 KV + MTP)
- `models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml` — Qwen dual, FP8 weights + MTP
- `models/qwen3.6-27b/vllm/compose/multi4/autoround-int4/mtp.yml` — Qwen 4-card + MTP
- `models/gemma-4-31b/vllm/compose/dual/qat-awq-int4/base.yml` — Gemma dual default (`vllm/gemma-31b-dual`)
- `models/gemma-4-31b/vllm/compose/dual/autoround-int4/int8.yml` — Gemma dual + INT8 PTH KV

Filename collisions across topology and quant dirs (e.g. `dual/autoround-int4/dflash.yml` vs `multi4/autoround-int4/dflash.yml`) are fine — the path disambiguates. Registry tags in `scripts/switch.sh` decouple from filesystem paths; rename only the file path in the registry / VARIANTS map and keep tags backward-compatible.

**Fine-tune convention**: model-specific fine-tunes that share the canonical model's compose directory get their own quant slug (for example a `dual/<finetune>-bf16mtp/bf16-mtp.yml` under `models/qwen3.6-27b/`, as the Carnice and Qwopus fine-tunes once had). Long-term those fine-tunes can graduate to their own model directories (`models/carnice-v2-27b/`, `models/qwopus3.6-27b/`) when their variant set warrants it.

#### Patches and caches stay engine-level (NOT under a topology)

`<model>/<engine>/patches/` and `<model>/<engine>/cache/` sit parallel to `compose/`, not under any topology subdirectory (`cache/` is now only the raw-compose fallback — see 3). Three reasons:

1. **Patches are reused across topologies and quant variants.** `vllm-marlin-pad/` is mounted by dual and multi-card AutoRound composes. Putting it under one topology or quant dir would force the others to symlink or duplicate.
2. **Patches are scoped by (model, engine), not by topology.** A vLLM source override doesn't change based on TP=1 vs TP=4; it's an in-container engine-internal patch that applies the same way regardless of mount layout.
3. **Caches (`torch_compile/`, `triton/`) warm-start across composes** — and since #1466 phase 4 they no longer live in the checkout. The launchers (`switch.sh`/`launch.sh`, `gpu-mode`, the estate planner, c3's generated-compose serve) mount them from `${CLUB3090_CACHE_DIR:-${XDG_CACHE_HOME:-~/.cache}/club-3090}/<engine image key>/`, shared by every checkout and every model on that image; `<engine>/cache/` is only what a raw `docker compose up` with nothing set falls back to. Write a cache mount as `${CLUB3090_ENGINE_CACHE_DIR:-../../../cache}/<subdir>:<container path>` (with `user: "0:${DOCKER_GID:-1000}"`), and the KV disk tier as `${KV_OFFLOAD_DIR:-${CLUB3090_DATA_DIR:-../../../../../..}/kv-offload}:/kv-offload`; `test-compose-cache-ownership` holds every compose to both. Why the launcher and not the compose: the key is the image ID, which compose can't compute, and a missing bind-mount source is created by docker as root:root, so the launcher creates each mounted dir as the user first. Mechanism, key format and fallbacks: `scripts/lib/engine_cache.py`.

Relative paths from a compose to its sibling patches/caches: `../../../patches/...` and `../../../cache/...` (one `..` from `<quant>/` to `<topology>/`, second to `compose/`, third to the engine dir, then `patches/` or `cache/` — for caches, as the fallback inside `${CLUB3090_ENGINE_CACHE_DIR:-…}`). Repo-root mounts such as `scripts/` and `models-cache/` need one extra `../` for the quant layer too.

**If a future patch is genuinely topology-specific** (e.g., a kernel rewrite that only applies to TP=2), keep it at `<engine>/patches/<patch-name>/` and document the topology constraint in the patch's README. Discoverability ("one `patches/` per engine, search there") trumps the marginal benefit of a topology partition.

#### Engine patch provenance — upstream first

Engine-code patches enter the catalog from an **upstream PR or upstream commit** (merged or open) as the standard path. A user- or community-authored engine patch is vendored only when **both** hold: (1) **critical** — a measured crash, corruption, or severe perf/cliff defect with a repro, not a capability or tuning preference; (2) **not yet available upstream** — no open upstream PR/commit covers the fix. Every `patches.yml` entry carries `upstream.status` + ref; entries in the exception class additionally carry a one-line criticality justification and a drop trigger (file-upstream tracking / merge-inherit). Chat templates and other data files are not engine code and are out of scope; diagnostic overlays and negative-result records are not shipped fixes and are out of scope. When an exception-class patch's fix lands upstream in a pinned release, drop the vendored copy in the same pin bump (existing drop-trigger convention).

#### Profile schema header (every compose, every time)

Every compose starts with a `Profile (at-a-glance)` block declaring the (Model, Topology, Drafter, KV, Vision, Max-ctx, Genesis) tuple in structured form. Free-form description follows below the schema, not in place of it.

```yaml
# ===========================================================================
# Profile (at-a-glance):
#   Model:     <name + quant — e.g. "Qwen3.6-27B (Lorbus AutoRound INT4)">
#   Topology:  <e.g. "Dual 3090 PCIe (TP=2, no NVLink)">
#   Drafter:   <none | MTP n=N | DFlash n=N | ngram K=N>
#   KV:        <fp8_e5m2 | bf16 | int8_per_token_head | turboquant_3bit_nc>
#   Vision:    <yes | no>
#   Max ctx:   <e.g. 262K>
#   Genesis:   <none | v7.72.2 | N/A — Genesis is Qwen3-Next-specific>
#   Status:    <REQUIRED — exactly one of the enum values below>
#   Caveats:   <REQUIRED if Status is ⚠️ / 🐣 / 👁️ / ⏸️ / 🗑️; otherwise omit>
#   Quality:   <OPTIONAL — populated by `bash scripts/quality-test.sh --medium`>
#                e.g. "ToolCall-15 14/15 (93%) · InstructFollow-15 13/15 (87%)
#                      · StructOutput-15 15/15 (100%) · DataExtract-15 12/15 (80%)
#                      (--medium, thinking OFF, sampling=server, validity=valid,
#                      packs tc1.0.1·if1.0.0·so1.1.0·de1.2.0·rm1.0.0, 2026-05-09)"
#                Paste the wrapper-generated line VERBATIM — it carries explicit
#                per-pack versions (#981). Never hand-write a `packs v1.0.x`
#                wildcard: the eight packs span six distinct versions.
#   Best for:  <one short phrase — what workload this serves; ⭐ for canonical>
# ---------------------------------------------------------------------------
# (existing free-form description continues below)
```

**Status enum** — pick exactly one:

| Value | Meaning | Validation gate |
|---|---|---|
| `✅ Production` | Recommended for users. | verify-full 8/8 + verify-stress 7/7 + bench (BENCHMARKS row) + soak-continuous PASS + `quality-test.sh --quick` PASS (no ≥10pp regression on ToolCall / InstructFollow vs the pre-change baseline). Quality numbers on the compose's `Quality:` schema field when `--medium` has been run. |
| `⚠️ Production w/ caveats` | Works under documented constraints; not the same as broken. | Same gates as Production, but a known-and-disclosed limitation exists (e.g., Cliff 2b at >50K, or a >10pp drop on a specific quality pack). Caveats line MUST list the constraint. |
| `🧪 Experimental` | Under active validation; may not boot or pass all tests. | Typically untracked in git. No production guarantee. |
| `🐣 Incubating` | Pre-experimental: works, but not ready for the actionable list — a niche specialist or one that fails the standard gate by design (e.g. an always-reasoning model with no tool-calling). | **HIDDEN from `switch.sh --list` by default** (revealed by `--list --all`), and launch requires `--force` (non-functional). Caveats line MUST state why it's not gate-passing. Promote to 🧪/✅ when it earns the actionable list. |
| `👁️ Preview` | Known quality issues; tracked but not for production. | E.g., quality regressions in soak / NIAH. Caveats line MUST list specific issues. |
| `⏸️ Upstream-gated` | Exists but blocked by external action (PR merge, driver fix, hardware ceiling). | Boots only with vendored override OR doesn't boot until external dep lands. Caveats line MUST point at the external dep. |
| `🗑️ Deprecated` | Kept for historical reference; will be removed. | N/A — flagged for cleanup. |

**Why this enum exists**: the previous "Status optional, only when not production" convention left readers guessing whether absence-of-status meant "validated production" or "author forgot to fill it in." Making Status required + enumerated removes that ambiguity. Users picking a config can scan to one field and know the lifecycle stage instantly; new contributors must consciously declare it when authoring.

The `Caveats:` line is REQUIRED whenever Status is ⚠️ / 🐣 / 👁️ / ⏸️ / 🗑️, OMITTED for ✅ / 🧪. Format: a single-line summary or a short bullet list, with links to issues / discussions / upstream PRs where relevant.

This rule applies to **shipped composes AND local-only test composes** — apply the convention even before deciding whether to ship; it avoids a rename later if the experiment graduates.

When testing a new model, create the directory hierarchy from the start: `models/<new-model>/<engine>/compose/<topology>/<quant-slug>/<serving>.yml`. The quant slug must match the `weights_variant` key in `scripts/lib/profiles/models/<model>.yml` and `scripts/lib/profiles/weights.py`. When the model isn't Qwen3-Next, write `Genesis: N/A — Genesis is Qwen3-Next-specific` in the profile schema so readers don't expect Genesis-style perf folds where they don't apply.

#### Adding a model — full workflow → [`docs/ADDING_MODELS.md`](docs/ADDING_MODELS.md)

Read that doc before catalog work; the at-a-glance for agents:

- **⚠️ FIRST — ASK WHICH REGISTRY, and never write the curated catalog on a user's checkout.** Three destinations, not interchangeable. c3's Bring & Validate lane already states them verbatim — reuse its wording: `⏎ Write LOCAL layer · C WRITE CORE REGISTRY (needs C3_ALLOW_CORE_PROMOTE=1) · E Export as PR bundle`.
  - **This rig only** — the default, and right for nearly all BYOM. `python3 scripts/lib/profiles/promote.py --spec-file <spec>` writes the **gitignored** `scripts/lib/profiles-local/` layer (`models.d/<id>.yml` + `composes/<id>/…` + `registry.local.json`, slugs namespaced `local/`). Nothing tracked is touched, so `git pull` and branch switches can never conflict with it. `get_registry()` merges it into the catalog every launcher reads, and a local entry can never become a curated default.
  - **Into the club catalog** — promote LOCAL first, validate on the rig, then `python3 scripts/lib/profiles/export_pr.py --spec-file <spec> --out <dir>` emits the ready-to-commit bundle (`models/<id>.yml` + core-layout compose + `registry-entry.yaml`), writing only under `--out`. It **refuses (exit 3)** when the model lacks what a maintainer would bounce a PR for — real `display_name`/`family`, a well-formed weights map, complete registry kwargs, a vLLM `kvcalc_key`, the compose's `Status:` header; `--check` runs that validation and writes nothing. **The local layer is the staging ground for a contribution, not a dead end.**
  - **Maintainer, on the canonical rig** — `promote.py --layer core`, gated on **both** `--layer core` and `C3_ALLOW_CORE_PROMOTE=1`. That gate is a plain setting: `promote.py` reads it from the shell or, failing that, from the saved settings (`~/.config/club-3090/`, then a legacy repo `.env`) through the one loader, and prints where it came from when a file set it. `export` it or pass it per-invocation; a saved `C3_ALLOW_CORE_PROMOTE=1` leaves the gate open for every run from every checkout. **The curated-catalog steps below are this row only.**
- **Just serving, not cataloging?** Safetensors → `scripts/pull.sh <org/Model> --profile-like vllm/minimal`; a self-grabbed **GGUF** → copy an ik/llama compose and point `--model` at it (no registry/profile needed — see ADDING_MODELS "Run a local GGUF without the catalog"). The steps below are only for promoting a model into the **curated catalog**.
- **Adding a new QUANT/SLUG of an existing model?** That's the most common catalog change and it has its own checklist: [`docs/ADDING_MODELS.md`](docs/ADDING_MODELS.md) → "The lightweight path". Follow it verbatim — the three items agents skip (and shouldn't) are the FULL test suite (a new slug IS a catalog-shape change), `diagnose-profile.sh <slug>`, and booting the ACTUAL compose via `switch.sh` + verify-full (an equivalent hand `docker run` does not count).
- **Catalog steps the compose alone doesn't cover:** (1) `scripts/lib/profiles/models/<id>.yml` — `weights:` is a **map keyed by quant-slug**, not a list; (2) a `scripts/lib/profiles/registry.yaml` entry under `entries:` (`weights_variant`=slug · `kvcalc_key` — vLLM `"<model>:<profile>"`, ik/llama `"SKIP"` · `default_port` == the compose's `${PORT:-NNNN}`); (3) launchers **auto-derive** from the registry — never edit `launch.sh`/`switch.sh`; promote a default via the `defaults:` map. After any hand edit run `python3 scripts/lib/profiles/migrate_registry_to_yaml.py --check`.
- **Profile-catalog compatibility (easy to miss — hotfix #236):** the new `(model, engine, KV-format)` combo must validate or `test-profiles-compat` / `diagnose-profile` go red. Add the model's `family` to the engine's `supported_model_families` (`scripts/lib/profiles/engines/*.yml`), the KV format to the hardware profiles' `supported_kv_formats` (`scripts/lib/profiles/hardware/*.yml`); register any vendored chat-template in `scripts/lib/profiles/patches.yml` (with the symmetric-protocol `drift_guard`); bump the `test-compose-registry-disk` size-count.
- **Run the FULL catalog test suite for catalog-shape changes** (new model / engine / slug / profile-schema change), not just the serving tests in [Tests](#tests): `for t in scripts/tests/*.sh; do bash "$t"; done`. Key gates: `test-compose-registry-disk`, `test-compose-mounts-resolve` (the `../` depth), `test-model-weights-registry`, `test-switch-registry-parity` + `test-launch-registry-parity`, `test-profiles-compat`, `test-patch-attribution`, plus `tools/kv-calc.py --calibration`. A narrow subset shipped a model with two real catalog gaps (#236) — and some failures are pre-existing/env, so **baseline against the last release tag** before treating one as a blocker. **Scoping rule:** the full sweep is for catalog-shape changes only — for a scoped change (one compose's flags, a doc, a single script) run just the guards that touch what you changed; most gates are irrelevant to it and the sweep wastes the signal.
  ⭐ **Tier 0 — the tree-wide guards run on EVERY change, whatever the diff touches.** A handful of guards assert a property of the *whole* `scripts/` tree rather than of a named file, so "which area did this touch?" cannot find them and they get skipped exactly when they matter. They cost seconds, not the sweep. Run them all with:
  ```bash
  for t in $(command grep -lE "grep -rn|grep -rE|grep -rl" scripts/tests/*.sh) scripts/tests/test-{script-permissions,locale-utf8,no-bare-grep,config-single-parser,tests-never-launch}.sh; do bash "$t" || echo "FAIL $t"; done
  ```
  The first part finds every guard that greps the tree; the five named ones are tree-wide too but don't grep. Examples of what they hold: `test-engine-kind-resolver` (no private engine classifier anywhere — it greps all of `scripts/`), `test-script-permissions` (git mode bits, which an untracked new file does not have yet), `test-compose-bind-host`, `test-compose-cache-ownership`, `test-compose-gpu-mask-passthrough`, `test-compose-nparallel-knob`, `test-custom-ar-knob`, `test-local-slug-shape`, `test-studio-derig`, `test-config-single-parser` (settings are parsed only by `scripts/lib/club-config.sh` / `club_config.py`, #1466 — a ratchet: the allowlist of old readers only shrinks; it also fails any `scripts/tests/*.sh` that doesn't `export CLUB3090_CONFIG_DIR=/nonexistent/…` at the top, so no test reads the user's real settings), `test-launch-knobs` (every compose's launch knobs match `scripts/lib/profiles/launch-knobs.json` and reach the container through both delivery channels, #1465). A hand-kept list here drifted (10 guards were missing by 2026-09-28), so derive it with the command above — a guard that greps the tree is invariant-shaped.
  ⚠️ **This is how `master` went red for five merges (2026-09-21).** Community PR #1341 touched only `scripts/lib/litellm-emit.sh`; the litellm-area guards passed; it added `prov = "hosted_vllm" if engine.startswith("vllm") else "openai"` — the **seventh** private engine classifier, which is the exact thing `scripts/lib/engine-kind.sh` and its guard exist to prevent (#1282). Nothing in the diff pointed at a guard called `engine-kind`. **Merging someone else's PR raises this bar rather than lowering it:** a contributor writes to the conventions visible from their diff, and the tree-wide ones are invisible from inside one file. Fixed in #1372.
  **Which guards, for the common scoped changes:** a compose header / default-value edit → `test-compose-status-drift`, `test-compose-registry-disk`, `test-profiles-compat`, `test-launch-compat`. A registry `status`/`status_note` edit → the same four. A script-logic change → the guard named after it, plus its own test if it has one.
  **Branch far behind `master`?** Merge `master` into a worktree copy and run the **targeted** guards on that combination — that is what catches a cross-merge interaction (e.g. an exact-count assertion another PR moved). Running the other ~176 unrelated tests does not add signal, and merging without testing the combination does.
  **For a runtime/config default, live validation beats any static gate.** Boot the actual compose and exercise the thing that changed; no guard asserts on what the engine admitted at runtime. #1338 (`MAX_MODEL_LEN` default 204800 → 262144) was settled by serving a 240,030-token prompt — past the old ceiling, so only the new default admits it — not by the suite.
  ⚠️ **The sweep is not free or neutral:** ~28 min wall, it pins a worktree for its whole run (never clean up underneath a running sweep — doing so turned 82 passes into 98 phantom `getcwd` failures), and it [flakes under GPU load](#tests). Reflexively sweeping has its own failure modes; it is not the safe default.

#### Where do experimental / unvalidated composes live?

**Same directory as shipped composes, but kept untracked until validation passes.** Don't create a separate `experimental/` subdirectory — the relative paths to `../patches/...` and `../cache/...` are calibrated to the compose dir, and promoting an experiment from a sub-folder would require re-pathing every mount.

Workflow:

1. **Author the compose** in `models/<model>/<engine>/compose/<topology>/<quant-slug>/<serving>.yml` with the standard profile schema header. Mark `Status: ⚠️ EXPERIMENTAL` (or `⚠️ PREVIEW` if quality issues are known) so readers know it's not validated.
2. **Don't `git add`** until validation passes. The file shows up in `git status` as `??` — that's the signal. `git ls-tree -r HEAD` lists only shipped composes; the gap between that and `ls compose/*.yml` tells you what's pending validation.
3. **Validation gates** before promoting: `verify-full.sh` 8/8, `verify-stress.sh` 7/7 (or documented failures with rationale), `bench.sh` (numbers added to BENCHMARKS.md), `soak-test.sh SOAK_MODE=continuous` (catches Cliff 2b), `quality-test.sh --quick` (no major regression on ToolCall / InstructFollow). For pin bumps and new quants, run `quality-test.sh --medium` and add the result line to the compose's `Quality:` schema field.
4. **Promote**: drop the `Status: ⚠️ EXPERIMENTAL` line from the profile schema, `git add`, commit. Cross-rig validation can come later via the `numbers-from-your-rig` issue template.

For **entirely new models** under validation (e.g. "let's try MiniMax-M2.7"): keep the whole `models/<new-model>/` directory untracked until at least one compose validates. Avoid pushing `models/<new-model>/README.md` etc. before there's a working compose to back it up — empty model directories on master signal capability we don't actually have.

(The 2026-05-09 orphan set has since shipped or been removed — don't expect specific untracked files; the `git status` `??` gap is the live signal.)

**Retired composes → `compose/_archive/`.** A compose that's fully superseded but still referenced by history (CHANGELOG entries, learnings, old discussions) moves to `models/<model>/<engine>/compose/_archive/<topology>/...` instead of being deleted: it keeps old links resolving while staying **out of the registry** (no slug, not launchable, invisible to `switch.sh --list`). This is one step beyond `🗑️ Deprecated` (which keeps the registry entry — see the Status enum): deprecate when users may still reference the slug; archive when nothing but history points at it.

### Documentation
- Don't create new docs proactively. Most non-obvious things belong in `INTERNALS.md`, `FAQ.md`, `SINGLE_CARD.md`, or `DUAL_CARD.md`. New top-level files only when there's a recurring search miss.
- Charts: source `.svg` + exported `.png` at retina resolution (≥1500px wide). Markdown embeds use `.png` (clicking opens a viewable image; SVG opens raw XML). Re-generate with `python3 tools/charts/gen-perf.py` and `gen-vram.py` after editing data.
- For any change that adds a footnote to "this depends on upstream X" — the answer is to link the row in `docs/UPSTREAM.md`, not to inline-cite the upstream URL.

### Tests
- `verify.sh` — fast smoke (~15s). Confirms the stack is responding; runs after `setup.sh`.
- `verify-full.sh` — functional (~1-2 min). Runs on every compose change.
- `verify-stress.sh` — boundary cases (longctx ladder + tool-prefill OOM ~5-10 min). Runs on cliff-related changes.
- `bench.sh` — canonical TPS bench (~3-5 min). Run when you change anything that could move TPS (compose flags, Genesis env vars, vLLM pin).
- `quality-test.sh` — behavioral quality (~10-30 min depending on `--quick` / `--medium` / `--full`). Wraps [`benchlocal-cli`](https://github.com/noonghunna/benchlocal-cli) — runs verifier-backed bench packs (ToolCall-15, InstructFollow-15, StructOutput-15, etc) against the running endpoint. Catches what operational tests miss: a compose can pass verify + stress + bench + soak and still ship with degraded tool-call accuracy or instruction-follow drift from quantization or Genesis env-flips. Run before promoting `Status: ✅ Production` and before any pin bump that could shift behavior. **Run via this wrapper, not raw `benchlocal-cli`** — it auto-detects endpoint/model and, for localhost URLs, sets `BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1` so the sandboxed HermesAgent can reach the host model (direct `benchlocal-cli` skips that → hermes silently scores ~0/20). **Live per-scenario `[N/M]` progress is on by default** (the wrapper forwards `--progress` to benchlocal-cli) so long runs don't go dark; pass `--no-progress` only for CI / log-volume contexts. **Timeout sizing:** benchlocal auto-scales per-scenario timeouts (startup decode-TPS probe × token-budget multiplier, on both arms), deliberately over-budgeting — the fix for the thinking-on spurious-timeout class (benchlocal-cli #54/#59). **Don't hand-set `--timeout-per-case` to "fix" a slow run unless you've confirmed the probe measured wrong.** **Budgets default to the published recipe** (since 2026-10-02): 4,096 completion tokens, 16,384 on thinking packs, 900 s per sandbox model call, 600 s per hermes episode — benchlocal's per-pack ~1,024 tokens cut long answers into `token_limit` rows that read as wrong answers. Each is overridable; `--pack-budgets` restores benchlocal's own. The per-case timeout is deliberately not defaulted, because benchlocal scales it with the token budget. Quality results from before that date used whatever budgets their command named, so check before diffing across it. A planned opt-in tier will size from a soak-derived per-depth TPS curve (#114). Full precedence + flags → [`docs/QUALITY_TEST.md`](docs/QUALITY_TEST.md) "Per-scenario timeouts".
- `soak-test.sh` — stability (30-60 min). Run before shipping config / Genesis / memory-policy changes — catches Cliff 2b.
- `rebench-full.sh` — **the canonical full-eval orchestrator: use this instead of hand-sequencing the scripts above.** verify-full preflight (fail-fast) + 5 measured steps in the `docs/QUALITY_TEST.md` pipeline order, with the recurring manual-run mistakes guarded (wrong cwd, missing `--save-json`, forgotten `MODEL=` override, missing hermes localhost env, wrong port) and `--resume` idempotency so an interrupt doesn't redo the matrix.

**Long-running tests: redirect full output to a log file — NEVER pipe through `tail`/`head`/`grep`.** A pipe buffers until the process exits, so a `--full` quality run piped to `tail -12` is a ~2 h black box: no live `[N/M]` progress, no partial scores, and an interrupt leaves nothing readable (benchlocal also writes its results JSON only at completion — noonghunna/benchlocal-cli#82 tracks scenario-level resume). Do `bash scripts/quality-test.sh --full ... > /path/run.log 2>&1` (or `| tee`) and summarize from the file; `tail -f` the file for live progress. Learned 2026-07-11 on a template A/B.

**Model resolution (all serving tests): they auto-detect — don't hand-pass `MODEL=` out of superstition.** Every serving script resolves the served-model id from the endpoint when `MODEL` is unset: `verify.sh`, `verify-full.sh`, `verify-stress.sh`, `bench.sh` (via `preflight_autodetect_model`), plus `quality-test.sh`, `rebench-full.sh`, `bench-agentic.sh` and `switch.sh`'s ready probe — all through `club_served_model_id` (`scripts/lib/served-model.sh`). It asks **`GET /v1/model` first and trusts only a genuine model card**, because TabbyAPI (exllamav3) answers `/v1/models` with *every folder in its model directory* — a report resolved a folder called `modules` ([#1360](https://github.com/noonghunna/club-3090/issues/1360)); vLLM / llama.cpp / SGLang 404 there and get `data[0].id` exactly as before. `soak-test.sh`, `concurrency-probe.sh` and `spec-sweep.sh` still read `data[0].id` inline — pass `MODEL=` to them on TabbyAPI. The endpoint itself is found the same way for every engine, including TabbyAPI's internal port 5000 (`club_engine_port_lines` in `scripts/lib/club-containers.sh`; 5000 counts only for a container that is ours by name, since unrelated apps use it too). `bench.sh` announces it: `[autodetect] served model='…'`. An explicit `MODEL=` **always wins and is never clobbered**. The `qwen3.6-27b` literal in some of them is a last-resort fallback, and since [#1330](https://github.com/noonghunna/club-3090/issues/1330) it is **refused exactly where it would mislead**: when the endpoint is unreachable, or when we autodetected the container ourselves and it reports no model, those four scripts stop with *"could not resolve which model to request, and guessing would be worse"* rather than 404-ing every request against a name the server never heard of. That fallback used to fire on a server that was merely still **loading**, and the result — `rc=8`, 8/8 red — read exactly like the config under test being broken. A user-supplied URL that reports no model still keeps the literal, because llama.cpp ignores the request's model field entirely. `PREFLIGHT_MODEL_WAIT_S` (default 10) bounds a short readiness wait, since mid-boot is the common case.

⚠️ **Do pin `MODEL=` on multi-model endpoints** (llama-swap, or a compose registering several `--served-model-name` aliases): detection takes `data[0].id`, which may not be the alias you mean.

> *This paragraph used to read "`verify-*`/`bench.sh` default `MODEL=qwen3.6-27b-autoround` — against any other served model that's a silent HTTP 404." That stopped being true when #372 landed autodetect, and the stale wording caused a wrong support answer on [#873](https://github.com/noonghunna/club-3090/issues/873) (a contributor was told his bench run would 404 when it would have auto-resolved). Verified against all ten scripts 2026-08-04.*

The pipeline is layered: each script has a different question it answers ("does it serve / work / survive / fast / behave correctly / stay healthy"). Skipping any layer can mask regressions.

#### Running a full eval — two non-overlapping passes

**Behavioral quality** (the 8-pack) and **operational health** (verify / stress / soak / bench / agentic) split cleanly. Run one of each — together they cover everything with nothing run twice:

**One-time setup (quality):** the suite wraps `benchlocal-cli`; three of the eight packs (bugfind-15, cli-40, hermesagent-20) run inside Docker sandboxes that build once:

```bash
pip install git+https://github.com/noonghunna/benchlocal-cli.git
git clone https://github.com/noonghunna/benchlocal-cli
bash benchlocal-cli/tools/build-sandboxes.sh   # ~30 GB free; `docker system prune` if tight
```

(Without the images the 5 deterministic packs still run; the 3 sandboxed ones skip with a warning.)

1. **Behavioral quality — the 8-pack, both reasoning modes:**
   ```bash
   bash scripts/quality-test.sh --full --no-thinking       # reasoning OFF
   bash scripts/quality-test.sh --full --enable-thinking   # reasoning ON
   ```
   ⚠️ **Match the reasoning mode to the leg — in both directions.** Reasoning-ON leg: boot the compose with reasoning parsing on (`REASONING=on` for llama.cpp composes, `--reasoning-parser` for vLLM) so `<think>` lands in `reasoning_content`, not the graded answer. Reasoning-OFF leg: boot with reasoning parsing off — leaving `REASONING=on` up makes the server force reasoning on every request, so the "no-thinking" leg silently becomes a second thinking leg (both arms score alike and the A/B reads as a clean, legitimate null). benchlocal-cli flags both failure modes automatically: per-pack `thinking_validity` in the saved JSON, and `--strict-thinking` for a CI exit code (a first-class wrapper flag — `quality-test.sh --full --no-thinking --strict-thinking`; any other benchlocal-cli flag the wrapper doesn't name goes after `--`, e.g. `-- --retry-runaways`).
2. **Operational health:** `bash scripts/report.sh --full` (~43 min; redacted, paste-ready bundle — verify + stress + soak + bench + agentic).

**Don't pair `rebench-full.sh` with `report.sh --full`** — rebench re-runs the same operational gates (verify/bench/agentic/concurrency/stress/soak), so it *replaces* `report.sh --full` rather than complementing it. Pick by goal: `rebench-full --with-8pack-thinking=both` when you want one synthesized `REPORT.md` (quant A/B, BENCHMARKS row); the two-pass split above when you want the paste-ready cross-rig bundle. Since #805, rebench runs the **agentic curve and the concurrency rungs itself** (steps 1b/1c), so there is nothing left to top up with `report.sh --agentic` — it would just re-measure. The user-facing version is [`docs/RUN_EVALS.md`](docs/RUN_EVALS.md) (per-engine thinking switches, the SGLang instruct-leg trap, why each flag); announcements link it via [`docs/ANNOUNCEMENT_TEMPLATE.md`](docs/ANNOUNCEMENT_TEMPLATE.md) §7 "Run the evals". Keep the three in step when the recipe changes.

### serve-cockpit (c3)
`tools/serve-cockpit/` is the Textual TUI cockpit — a separate Python app with its **own venv and pytest suite**, NOT covered by `scripts/tests/*.sh`. See its `README.md`. For agents:
- Run tests with `tools/serve-cockpit/.venv/bin/python -m pytest tools/serve-cockpit/tests/ -q`. `test_services.py` + `test_registry_parser.py` are fast — run them on every c3 change; `test_app_headless.py` boots the full app and is slow — prefer targeted `-k` selection while iterating. Judge a change by the **full** suite before merging (~15 min): a failure that shows only when a file runs on its own is a test-isolation bug, not your change. `tests/conftest.py` removes any temp dir a test leaves on `sys.path` — one such leak made 11 tests fail whenever `test_services.py` ran alone.
- ⚠️ **The c3 suite is NOT in `scripts/tests/*.sh`, so the full sweep being green says nothing about c3.** A c3 change needs both, run separately. #905 shipped two red c3 tests behind a green `99/0` sweep for exactly this reason.
- ⚠️⚠️ **Never run this venv from a git worktree — it silently tests the WRONG TREE.** The editable install pins **absolute** paths (`.venv/lib/python*/site-packages/_editable_impl_club3090_*.pth` → `/…/club-3090/tools/serve-cockpit` and `/…/tools/tui-core` in the **main checkout**). So `import club3090_cockpit` resolves to the main tree no matter your cwd or which worktree you are in: pytest collects *your* test files and runs them against **master's** application code. That mismatch produced **495 failures where 2 were real**, and — far worse — it *masked* a genuine bug in the change under test. Do c3 work in the **main checkout**, build a venv inside the worktree, or put the worktree first on the path: from the worktree's `tools/serve-cockpit`, run the main venv's python with `PYTHONPATH=$PWD:$PWD/../tui-core`. Verify which code you are actually testing before trusting a result:
  ```bash
  cd tools/serve-cockpit && .venv/bin/python -c "import club3090_cockpit; print(club3090_cockpit.__file__)"
  ```
- c3 consumes the `registry-emit.sh --json` contract. Adding a field to the emit means threading it through `services.py` (`_variant_row_from_dict`) and, if displayed, `app.py` — and emit changes also need the `test-switch-registry-parity` / `test-launch-registry-parity` guards green.
- DataTable cells render in terminals: avoid U+FE0F variation-selector emoji (`⚠️ 👁️ ⏸️ 🗑️`) in fixed-width columns — Rich reserves 2 cells but many terminals draw 1, misaligning every column after it. Use `Emoji_Presentation=Yes` glyphs (see `_STATUS_GLYPH` in `app.py`).

### Commits
- New commit per logical change. Don't amend published commits.
- Commit messages: subject ≤72 chars, imperative ("Disable P68/P69..." not "Disabled..."), optional body for "why."
- Don't push without local verify-full + verify-stress passing for the affected compose.

### Hooks / verification
- Pre-flight checks live in `scripts/preflight.sh` and run from `setup.sh` + `launch.sh`. They check docker / GPU / disk before any heavy work. If a check fails, print an actionable `Fix:` hint, never a cryptic mid-run crash.

## Things to NOT do

- Don't add NVLink suggestions. The user has explicitly declined.
- Don't recommend EAGLE / DFlash spec-decode on Qwen3-Next single-card. It's blocked by DeltaNet rollback (see [`docs/UPSTREAM.md`](docs/UPSTREAM.md) → vllm#39931). MTP works.
- Don't enable Genesis behavioral patches (P68/P69 class) by default. They override user intent. If a user wants them, they can flip the env var.
- Don't claim a TPS number you didn't measure. "Should be ~80" labeled as estimate is fine; "is 80" needs a bench.
- Don't compress historical CHANGELOG entries. Append-only.
- Don't scatter upstream issue links across multiple docs. Link to the row in `docs/UPSTREAM.md` instead.
