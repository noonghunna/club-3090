# Adding an inference engine to the club-3090 stack

An **engine** here is everything between a model's weights and the OpenAI-compatible port: the server image, its launcher, its KV and spec-decode support, and the log lines our tools read. Adding one has two halves, and this page covers both:

1. **The catalog knows it.** There is an engine profile at `scripts/lib/profiles/engines/<id>.yml`, plus the registry, hardware and launcher entries that point at it.
2. **The tooling recognises it.** `verify-full.sh`, `bench.sh`, `health.sh`, `quality-test.sh`, the launchers, the gateway and c3 all decide what to do from the engine's family. An engine the catalog knows but the tooling doesn't is still second-class, because every check reads it as `unknown` and quietly takes a fallback path. Some fallbacks refuse, some assume vLLM, and some just skip.

Pairs with:
- [ADDING_MODELS.md](ADDING_MODELS.md) for models and slugs.
- [PATCH_POLICY.md](PATCH_POLICY.md) for the patches we deliver into an engine.
- [engines/](engines/README.md), one page per engine we ship.

> **Adding something engine-shaped? Pick the lightest path that fits:**
>
> 1. **A new slug on an engine we already ship**, for a model we already serve or a new one. Not this page; follow "Which case is this?" in [ADDING_MODELS.md](ADDING_MODELS.md). The slug names the engine's id.
> 2. **The same engine on a newer image.** Bump it in place: `scripts/engine-pin-bump.sh <engine-id> <tag> --check` does the mechanical half, and the PR re-runs `verify-full.sh` and `bench.sh` on the new image (`AGENTS.md` → "Engine image pinning"). Add a *second* profile only when slugs must stay on different builds of the same engine, as the `llamacpp-club3090-v1.x` profiles do. Never repoint an existing profile at an image its slugs weren't validated on (gate **M9** in ADDING_MODELS.md).
> 3. **Your own engine build, on your rig only.** `bash scripts/catalog.sh register --compose … --engine <id> --engine-type <family>` writes a local engine profile under `scripts/lib/profiles-local/engines.d/` and touches nothing tracked. See [BRING_YOUR_OWN.md](BRING_YOUR_OWN.md).
> 4. **A new engine in the curated catalog.** This page. Propose it before you build it (next section).

## Propose first

A new engine is a maintenance commitment: every pin bump, every tooling change and every new model has to consider it. So decide whether to carry it before the work starts, not after a 20-file PR. Open an [engine proposal](https://github.com/noonghunna/club-3090/issues/new?template=engine-proposal.yml) and wait for the go-ahead; the PR then uses `?template=engine.md`. The proposal answers:

- **Lineage:** its class (next section), its base engine and version, and its public source.
- **Image:** the image the engine's maintainer publishes, and its digest. See [Image provenance](#contract--image-provenance).
- **Patches:** every change on top of its upstream, and whether each one is upstream already.
- **Runtime contract:**
  - its internal port;
  - its launcher (the stock CLI, or a wrapper with its own env names);
  - its log lines for throughput and spec-decode acceptance;
  - how it handles the chat template, reasoning effort and thinking budgets;
  - its tool-call parser.
- **Why:** what it does that no shipped engine does, measured on a real card.

The engine's first slugs should preferably be on a model we already serve, so it is measured against an existing slug. If they need a model we don't have, open a [model proposal](https://github.com/noonghunna/club-3090/issues/new?template=model-proposal.yml) as well, and send one PR for each, engine first.

## CONTRACT — engine lineage

Every engine declares how it relates to the engines we already ship. Lineage decides how much of our tooling the engine inherits and what has to be built for it, so it is recorded in the profile and not only in prose.

| Class | Meaning | Inherits from its base | Must add | On master |
|---|---|---|---|---|
| **stock** | upstream's official release image, unmodified | everything; it *is* the base | the profile | `vllm-stable`, `sglang-stable`, `llama-cpp-local` |
| **patched** | a stock image plus **our** patches, mounted at boot | everything; the patches fall under [PATCH_POLICY.md](PATCH_POLICY.md) | `patches.yml` entries and drift guards | `vllm-gemma-stable`, `vllm-diffusion-gemma` |
| **derivative** | upstream engine pinned at a version, plus a **third party's** patch set or launcher, shipped as its own image | log format, metrics and API, but only for what it keeps unchanged | every difference: launcher env names, port, log wording, template handling | `vllm-lmcache` |
| **fork** | a separately maintained source tree branched from an engine, with its own releases | whatever it hasn't diverged on; check, don't assume | everything it changed, plus the fork chain | `llamacpp-club3090-*`, `llama-cpp-prism` |
| **new** | an independent codebase | nothing | the whole tooling contract below | `exllamav3` (TabbyAPI), `strata` |

The loader validates the field (`compat.py → _lineage`): unknown keys, an unknown relation, an unquoted numeric version and a malformed `chain` are refused. `test-engine-lineage` holds every curated profile to it.

The field:

```yaml
lineage:
  relation: derivative            # stock | patched | derivative | fork | new
  base: vllm                      # the family it comes from; null for new
  base_version: "0.30.0"          # release or commit of the base it is pinned to
  source: https://github.com/<org>/<repo>
  source_ref: <full commit sha>   # the exact source the image was built from
  chain: []                       # forks of forks, oldest first, e.g. [llama.cpp, <org>/<intermediate-fork>]
  delta: "<one line: what differs from the base>"
```

Rules that follow from the class:

- **Anything that isn't stock names a public `source` and `source_ref`.** An image we can't trace to source isn't accepted.
- **A derivative or fork pinned to an old base misses the base's fixes.** State which base version it tracks, and re-check it whenever we bump the base.
- **"Upstream first" applies to every upstream in the chain** (`AGENTS.md` → "Engine patch provenance — upstream first"). A fix to a derivative's own changes belongs in the derivative's repo. A capability that touches the base belongs in the base.
- **Credit follows the chain.** A fork that carries someone else's work keeps their authorship in its commits and names them on its engine page. For example, the moe-cache in `llamacpp-club3090` is leloch's work.

## CONTRACT — naming

Several of our tools identify an engine's family by **prefix**. The engine-kind resolver falls back to the profile's `type` when an engine *id* has no family prefix, so checks that classify by id or slug still work: the gateway emitter, launch-knob coverage, launch settings, `spec-sweep` and, since 2026-10-14, switch.sh's vLLM preflights. Container discovery and the classification that serving scripts do from container names and image refs still key on prefixes. A derivative or fork that takes the family's prefix inherits all of that family's handling; one that doesn't inherits only the id-based part.

| Thing | Rule | Precedent |
|---|---|---|
| Engine id | `<family>-<name>` for a derivative or fork. `scripts/lib/engine-kind.sh` matches ids by prefix (`vllm*`, `sglang*`/`sgl-*`, `exllamav3*`/`exl3*`, `llama-cpp*`/`llamacpp*`/`ik-llama*`/`beellama*`), and falls back to the profile's `type` for any other id | `vllm-lmcache`, `llamacpp-club3090-v1.6` |
| Profile filename | equals the id: `engines/<id>.yml`. Several tools open a profile by filename (c3, `generate_compose.py`, `list-image-pins.sh`, `test-engine-label-matches-compose`), so a misnamed profile is silently skipped by all of them; the `profile-filename` catalog rule refuses one | every core profile (the last exception, `llama-cpp-mainline.yml` → `llama-cpp-local.yml`, was renamed 2026-10-12) |
| Slug prefix | a derivative or fork may keep its family's prefix or take its engine id. switch.sh's hardware and GPU-fit preflights follow the slug's engine kind (`engine_kind_from_slug`), not its prefix (since 2026-10-14; before that they ran only for `vllm/` slugs) | `vllm/…` on `vllm-lmcache`; `vllm-hyperqwen/…`, `bucko-vllm/…`; `llamacpp-club3090/…`, `llama-cpp-prism/…` |
| Compose directory | `models/<model>/<engine-dir>/compose/…`, one directory per engine profile family | `vllm-lmcache/`, `llamacpp-club3090/`, `llama-cpp-prism/` |
| `container_name` | starts with the family's container prefix (`vllm-`, `sglang-`, `llama-cpp-`, `tabbyapi-`, …), so discovery and the kind resolver recognise it | — |
| Internal port | the family's standard port: vLLM 8000, SGLang 30000, llama.cpp 8080, TabbyAPI 5000 (see [Discovery](#2-discovery--container-name-and-internal-port)) | — |

## CONTRACT — image provenance

- **The engine must come with a published image. We don't build or publish images for new engines.**
  - **The image is published by the engine's own maintainer, from its public source.** Ideally the source repo's CI publishes it, so each tag maps to a commit. HyperQwen's workflow, for example, pushes `ghcr.io/syv-ai/hyperqwen:sha-<7>` per commit.
  - **The publisher matters, not the namespace.** `ghcr.io/anbeeld/beellama.cpp` sits in a personal account, but it's the maintainer's own image.
  - **Not accepted: an image re-published by anyone other than the maintainer**, even unmodified, and that includes a contributor's own build of someone else's engine.
  - **No image, no engine.** If the maintainer publishes none, ask them to; the proposal waits until one exists.
- **It pulls without credentials.** `DOCKER_CONFIG=$(mktemp -d) docker buildx imagetools inspect <install.spec>` must succeed on a machine with no registry login. A private package fails for every user who isn't logged in.
- **Pin by digest.** Anything that isn't upstream's own release image (derivatives and forks) is pinned `@sha256:` ("tags move, digests do not", #940). Stock and patched engines may pin an upstream release tag (`vllm/vllm-openai:v0.31.0`), but never `nightly-*` or `latest`, because upstream purges nightly tags.
- **Bumps depend on the maintainer.** We bump only to images the maintainer has published. An engine whose maintainer stops publishing can't be bumped and heads for deprecation, so weigh that in the proposal.
- **Images we build ourselves** are exceptions:
  - `llamacpp-prism` (a wrapper around PrismML's release) predates this rule.
  - `tabbyapi-club3090` (TabbyAPI / exllamav3, 2026-09-17 to 2026-10-30) is no longer one: TabbyAPI's CI publishes `ghcr.io/theroyallab/tabbyapi:cu13` from the same `docker/Dockerfile.cu13` we built, and the `exllamav3` engine pins it.
  - `strata-club3090` (Strata, 2026-10-13 to 2026-10-20) is no longer one: from 0.2.1 Strata publishes its own Docker images, and the `strata` engine pins them. While Strata published no Linux image, the maintainer built upstream's Dockerfile on the reference rig at each release commit we took, with ggml's CPU code built for an AVX2 baseline.
  - `strata-maya-club3090` (Project Maya, a Strata fork for GLM-5.3-Flash, 2026-10-15): Maya publishes no image and no Linux engine (from v1.1.53 its repository has a recipe for building an image yourself, adapted from ours), and releases several times a day. The maintainer builds each release we take on the reference rig with our own Dockerfile (the engine compiled as Maya's setup compiles it, with upstream's `STRATA_PORTABLE` option for an AVX2 baseline and a fixed arch list) and checks the result before pushing. Recipe and checks: [`models/glm-5.3-flash/strata-maya/image/`](../models/glm-5.3-flash/strata-maya/image/README.md).
  - `llamacpp-club3090` is our own fork. We're its maintainer, so it already meets the rule.
  - `ghcr.io/noonghunna/fa2-fp8kv-sm86` (2026-11-10) isn't an engine but a patch artifact: the FA2 FP8-KV kernels and vLLM plugin that the Qwen3.8-27B and ThinkingCap DFlash2 composes install on `vllm-stable`. Its author published images only for vLLM 0.29's PyTorch and has been idle since 2026-10-04, so for vLLM 0.33 we forked it ([noonghunna/fa2-fp8kv-sm86](https://github.com/noonghunna/fa2-fp8kv-sm86), branch `club-3090`) and build it with the author's own Dockerfile and checks. The changes are kept as PR-ready branches for the author. `docs/UPSTREAM.md` (FA2 rows) records what is in it.
- **Our patches are mounted, not baked.** Changes we carry on top of a stock image go through [PATCH_POLICY.md](PATCH_POLICY.md) (mounted onto the pinned image, tracked in `patches.yml`, guarded against drift). A community-authored engine patch is carried only under the "upstream first" rule in `AGENTS.md`.
- **The profile is the source of truth for the image.** The launchers inject `install.spec` through the engine's `image_env`, overriding the compose's `image:` default. Write `install.spec` as one unquoted token on exactly one `spec:` line, because `engine-pin-bump.sh` and `resolved_config.py` parse it that way. The compose's first `image:` must read `${<IMAGE_ENV>:-<the same spec>}`; `test-engine-label-matches-compose` checks it.
- **A new `image_env` variable must be allowlisted in both launchers.** That's the `case` arm in `export_variant_engine_pin()`, present in both `scripts/switch.sh` and `scripts/launch.sh`. Without it, the launch exits with `unexpected engine pin export`.

## CONTRACT — the engine profile

The file lives at `scripts/lib/profiles/engines/<id>.yml`, loaded by `scripts/lib/profiles/compat.py → EngineProfile` / `_engine()`. Copy the closest profile of the same class and adapt it.

**Required:** `schema_version: 1`, `id`, `display_name`, `type`, `stability`, `min_sm`. A missing one fails the load.

| Field | Meaning | Read by |
|---|---|---|
| `type` | what the engine *behaves like*: `vllm`, `llama.cpp`, `sglang`, `exllamav3` or `strata`. **Not validated.** | compat (C6 Genesis, C9 drafter match; C12 kv-calc projection runs only for `vllm`); `launch_compat` hardware injection (mem-util and concurrency env exist only for `vllm`/`sglang`); DeepGEMM; the derived-compose generator. `engine-kind.sh` reads it when the engine id has no family prefix; see [Engine kind](#1-engine-kind--scriptslibengine-kindsh). |
| `stability` | `stable`, `experimental` or `deprecated` | `launch_compat --stable` |
| `min_sm` | lowest compute capability | C3 |
| `image_env`, `install.method`, `install.spec` | the env var the image is injected as; `docker_image` or `pip`; the image ref | the launchers (`resolve_engine_pin`), `registry-emit`, `engine-pin-bump.sh` |
| `supported_model_families` | must contain the model YAML's `family:` exactly | C10 |
| `supported_kv_formats` | every KV format a slug on this engine uses | C4 |
| `supported_drafters` | **`spec_method` values** (`mtp`, `dflash`, `eagle3`, `ngram-mod`, …), not drafter ids | C7 |
| `supported_weight_formats` | every weights `format:` its slugs use | C14 |
| `valid_tp_by_family` | optional `{family: [tp, …]}`: the tensor-parallel sizes the engine runs for a model family, in place of the model's own `valid_tp` (vLLM's list). Only for an engine whose own tensor parallel differs (QEngine: three cards) | C2 |
| `features` | `{name: value}`, tested for truthiness against a slug's `required_engine_features` | C15 |
| `feature_provenance` | `{feature: {author, source, status}}`, for attribution | (record only) |
| `vendored_overlays` | overlays baked into the image (`image_baked: true`) or delivered from `patches/` | diagnose-profile [6/6] |
| `required_genesis`, `required_overlays` | vLLM/Genesis legacy | C6, diagnose-profile |
| `lineage` | see above | (draft; this proposal) |
| `notes` | prose: why this engine, how the image was built, traps | (record only) |

⚠️ Only **top-level** keys are strict-checked for engines. A typo nested inside `install:` or `features:` (for example `install.sepc`) is silently dropped.

Everything else that has to agree with the profile:

| Where | What | Check |
|---|---|---|
| `registry.yaml` entries | `engine: <id>` (must exist), plus `required_engine_features` | cross-reference check at load; C15 |
| `hardware/*.yml` | each new KV format in `supported_kv_formats` of every hardware profile its slugs must fit | C5 |
| `tools/kv-calc.py` | for `type: vllm`: the KV format in `KV_FORMAT_BYTES`, plus the activation-coefficient tables for `qwen3-next-*` families | C12 |
| `scripts/lib/profiles/canonical_scenarios.py` | each slug must fit at least one canonical scenario, so a Hopper-only engine can't pass today | `test-profiles-compat` |
| `arch_patches.yml` | `engine_pin` rows, only if the engine should be a base for the derived-compose generator and Pull-Gate (vLLM-type only). Most new engines don't | `generate_compose.validate_engine_pin` |

## CONTRACT — wiring it into the tooling

A derivative or fork that resolves to its base family's kind, and keeps the base's port, flags and log lines, inherits most of this section. A new family has to add every part.

### 1. Engine kind — `scripts/lib/engine-kind.sh`

This is the one place engine classification lives. Each entry point prints `vllm | llamacpp | sglang | exllamav3 | strata | unknown`. Consumers gather their own evidence and decide only through this file.

| Entry point | Evidence |
|---|---|
| `engine_kind_from_engine_id` | the registry `engine:` id, matched by prefix; otherwise the engine profile's `type` (core profiles, then the local layer) |
| `engine_kind_from_container` | the container name, matched by prefix |
| `engine_kind_from_image` | the image ref, matched as a substring |
| `engine_kind_from_fingerprint` | the response's `system_fingerprint` |
| `engine_kind_from_owned_by` | `/v1/models` `owned_by` |
| `engine_kind_from_slug` | the registry entry (core and local) → the id function |

**Make every one of them resolve your engine.** A derivative or fork needs to map to its base: its profile's `type` covers the id-based entry points, and the [naming](#contract--naming) rules cover the container- and image-based ones. A new family needs a new kind value, arms in all six functions, a `type` mapping in `_engine_kind_from_profile_type`, and probes added to `scripts/tests/test-engine-kind-resolver.sh`. Never add a classifier anywhere else: that test fails on a private `startswith("vllm")` or on a hand-listed container prefix in `scripts/`.

What `unknown` costs, by consumer:

| Consumer | On `unknown` |
|---|---|
| `spec-sweep.sh` | refuses: "engine id is unrecognised — add it to scripts/lib/engine-kind.sh" |
| `quality-test.sh` thinking budgets (`lib/thinking-budget.sh`) | refuses: "an unknown engine is a refusal, not a guess" |
| `probe-effort-budget.sh` | dies: vllm, sglang and qengine only |
| `test-launch-knobs` | **fails** if the compose reads any catalogued knob (`SPEC_N`, `REASONING_EFFORT`, …) |
| `health.sh` | no runtime details (KV usage, acceptance, throughput) |
| `verify-full.sh`, `report.sh`, `bench.sh` | treat it as "might be vLLM": vLLM probes, vLLM log greps |
| `report.sh` resolved config, `quality-test.sh` run metadata | no engine startup dump parsed |
| `cache-share.sh`, `litellm_sync.py` | no cached-token counters; no native `/v1/messages` passthrough |

### 2. Discovery — container name and internal port

`scripts/lib/club-containers.sh` finds the serving container in two steps:
- **Names** are registry-derived (the compose's first `container_name:`), with a prefix fallback.
- **Ports:** only containers publishing internal port **8000** or **30000**, or **8080** or **5000** with evidence (a recognised name or image), count as an engine.

A container on any other internal port is invisible to `verify*.sh`, `bench.sh`, `health.sh` and `soak-test.sh` unless the user passes both `URL=` and `CONTAINER=`.

- **Prefer the family's standard internal port.** Most wrapper launchers take a `PORT` env var.
- **If you genuinely need another port,** add it to every mirrored copy of the port set: `club-containers.sh`, `tools/tui-core/.../detect.py`, `scripts/catalog.sh`, and the port loops in `report.sh` and `soak-test.sh`. `test-engine-port-set-drift` checks they agree.
- **A new family's container prefix** goes into every mirrored copy of the prefix set too: the fallback regex in `club-containers.sh`, `ENGINE_PREFIXES` in `tools/tui-core/.../detect.py` and `_DETECT_PREFIXES` in `scripts/catalog.sh` (plus the `engine_kind_from_container` arm, [Engine kind](#1-engine-kind--scriptslibengine-kindsh)). `test-catalog-detectability` checks that `catalog.sh` matches `detect.py`. QEngine's registration missed the `catalog.sh` copy, and that test was red on master until it was added.

### 3. Throughput and spec-decode parsing

Two kinds of parser read the logs.

**Some match on wording, whatever the kind:**
- `scripts/lib/capture.sh` acceptance parsing knows the vLLM `SpecDecoding metrics:` line, SGLang `accept len:`, `(N accepted / M drafted)`, and exl3 `draft N/M`.
- `soak-helper.py` signatures.
- `p2p-state.sh` boot markers.

**Some branch on kind:**
- `health.sh` (one branch per family);
- `verify-full.sh` `check_mtp_acceptance`;
- `spec-sweep.sh` arms;
- `bench.sh`'s prefill scrape (`Avg prompt throughput:`);
- `rebench-full.sh`'s boot-log capture;
- `resolved_config.py` `_DUMP_PATTERNS`;
- `run_context.py`.

A vLLM derivative inherits all of it **only if it keeps upstream vLLM's log lines and `vllm:` Prometheus metric names**. A renamed line produces no error, just "no metrics" or a skipped check. When nothing parses, `bench.sh` says so ("this engine words its acceptance line differently"); teach the parser rather than ignoring it. A new family has to add an arm to each of the files above.

### 4. Launch knobs

Our standard knobs are catalogued in `scripts/lib/profiles/launch-knobs.json`, one `engines:` list of kinds per knob: `REASONING_EFFORT`, `ENABLE_THINKING`, `THINKING_BUDGET_*`, `SPEC_N`, `KV_OFFLOAD_*` and others. `switch.sh --set <slug> KNOB=…` refuses a knob the slug's compose doesn't read.

- **A wrapper launcher with its own env names** (for example `MAX_LEN` instead of `MAX_MODEL_LEN`) must map each catalogued knob it supports onto them in the compose's `environment:`.
- **Which knobs a compose must read** is in `scripts/lib/profiles/knob-conformance.json`. A knob applies by the model's own chat template (`enable_thinking`, `reasoning_effort`, Qwen3.8's effort levels for the per-effort budgets) or by the slug's drafter (`SPEC_N`). A new model needs a classification there, with the template as evidence. A compose that doesn't read an applicable knob fails `test-launch-knobs` unless the manifest says why: an `unsupported` entry when the engine can't take it, citing a repo file that says so (it matches by engine and model), or an entry in the `not_wired` backlog, which only shrinks. Don't add to the backlog: wire the knob.
- **Hardware injection is keyed by profile `type`.** `launch_compat` exports `GPU_MEMORY_UTILIZATION` (vLLM) or `MEM_FRACTION` (SGLang). A wrapper that reads a different name needs a mapping such as `GPU_UTIL: ${GPU_UTIL:-${GPU_MEMORY_UTILIZATION:-0.93}}`.
- **A launcher that writes the engine's own config file** (Strata's `strata_config.py`, Maya's `maya_settings.py`) needs an entry in `scripts/lib/profiles/engine-config-reads.json`: the probe value for each setting, and, for each key a setting writes, why the pinned engine reads it. `test-engine-config-reads` runs the launcher once per setting and checks every key it writes in the pinned image: read from the config object in an approved function, or listed in the engine's `--help`. A key that arrives in the file but that the engine never reads goes unnoticed by every other gate; Strata's `MAX_TOKENS` shipped that way (#1723, fixed in #1726). After a pin bump, `python3 scripts/lib/profiles/engine_config_reads.py verify <engine> --record` re-checks the evidence on the new image.
- **The spec-decode toggle contract** (`test-spec-toggle-contract`) runs entrypoints against a stub `vllm` binary. A launcher that isn't `vllm serve` isn't exercised by it, so test `SPEC_N`/`SPEC=off` on a live boot instead.

### 5. Reasoning, effort, chat template

- **Thinking budgets are per engine.** `lib/thinking-budget.sh` verifies and shapes them per kind, and `lib/effort_budget.py` (`FLOOR_ENV`, `shell-env --engine`) supports vllm and sglang.
- **The vLLM effort-budget hook patches the vLLM source by anchor.** A derivative whose vLLM source lacks the anchor can't use it, and the boot refuses. Ship without budgets and say so in `Caveats:`, rather than skipping the line.
- **The registry `chat_template` must name what the compose actually serves.**
  - `native` means the checkpoint's own template.
  - A named template (e.g. `qwen38-reasoning-effort`) must be mounted *and* passed with `--chat-template`; the template patch is registered in `patches.yml`.
  - `test-qwen38-template` only scans `models/*/vllm/compose`, so a compose in another engine directory isn't checked.
- **Match the model family's server-side defaults**, such as Qwen3.8's default `reasoning_effort=low`, or document the difference. A request that names no effort must behave the same on every slug of a model, or say why not.

### 6. Gateway

- **Every LiteLLM route is `openai/…`.** The engine kind only decides the Responses bridge (`NO_RESPONSES_KINDS` in `litellm-emit.sh`, today `exllamav3`).
- **The served name is parsed from `--served-model-name` / `--alias` in the compose.** A launcher that sets it some other way needs `served_name=` on the registry entry.
- **Native `/v1/messages` passthrough** needs `owned_by` to resolve to vllm or sglang.

### 7. Launchers and c3

- **switch.sh's hardware and GPU-fit preflights** run for every slug whose engine resolves to the vllm kind (`engine_kind_from_slug`), whatever its prefix.
- **c3** has its own family logic (`tools/tui-core/.../detect.py`, `club3090_cockpit/data.py` `_canon_engine_family`), which decides the Bring funnel and BENCHMARKS matching. It isn't covered by `scripts/tests`; run c3's own suite.

## CONTRACT — composes on a new engine

The compose layout, header and registry contract are the same as in [ADDING_MODELS.md](ADDING_MODELS.md). On top of that:

**Guarded for any engine directory:**

| Guard | Requires |
|---|---|
| `test-compose-registry-disk` | every compose under a new engine directory is registered (only parked SGLang composes and the vllm-omni compose may be disk-only), and the hard-coded counts are bumped |
| `test-compose-bind-host`, `test-compose-estate-port` | ports read `"${BIND_HOST:-0.0.0.0}:${ESTATE_PORT:-${PORT:-NNNN}}:<internal port>"` |
| `test-compose-healthcheck-reachability` | a healthcheck probes the container's own hostname, never loopback (a healthcheck itself is optional) |
| `test-compose-mounts-resolve` | relative mounts resolve; a compose that mounts caches ships `<model>/<engine>/cache/` with its `.gitignore` |
| `test-compose-env-declared-unused` | every declared env var is read. A launcher that consumes vars inside the image needs its prefix added to `ENGINE_PREFIXES` (as `EXL3_`/`TABBY_` were) |
| `test-compose-status-drift` | the header `Status:` matches the registry |
| `test-compose-port-conflicts` | no `default_port` collides across models, every curated one is on 8010–8199 (202xx is the local layer's), and none is a port the repo's own `services/` listen on |
| `test-engine-label-matches-compose` | the compose's `image:` default equals the profile's `install.spec` |
| `test-launch-knobs` | catalogued knobs the compose reads are covered for its kind (see §4) |

**Repo conventions the guards miss under a new engine directory.** Follow them anyway:

- **Restart policy:** `restart: ${CLUB3090_RESTART:-unless-stopped}`. `test-compose-restart-policy` scans every engine's composes.
- **Cache ownership:**
  - Mount caches as `${CLUB3090_ENGINE_CACHE_DIR:-../../../cache}/<subdir>:<container path>`, with `user: "0:${DOCKER_GID:-1000}"` and `umask 0002` at the top of the entrypoint. Otherwise the files the container writes into `~/.cache/club-3090/` come out `root:root` and need sudo to clear.
  - `test-compose-cache-ownership` only matches mounts whose container path starts with `/root/`, and nothing checks for the umask.
  - All cache-mounting services in a compose must share one image (`engine_cache.py` keys the directory by image id).
- **GPU selection:** copy your family's sibling compose, so the launcher's GPU choice reaches the container. Most families use `device_ids: ['${ESTATE_GPUS:-…}']`. vLLM composes use `- NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}` plus a bare `- CUDA_VISIBLE_DEVICES`; `test-compose-gpu-mask-passthrough` checks that form only where it appears.
- **Multi-GPU:** a compose that sources `detect_nvlink.sh` carries the custom all-reduce knob, `DISABLE_CUSTOM_ALL_REDUCE=${…}` (`test-custom-ar-knob`).
- **Engine-specific guards:** several guards are hard-wired to `models/*/vllm/compose` (KV-offload knobs, the mamba-SSM knob, the template check, draft sample method, registry compose defaults). Check which apply to your family and follow them.
- **Untracked composes are skipped by `test-compose-entrypoint-env-declared`,** which uses `git ls-files`. Run the full suite after `git add`, not before.

## GATES — which guard proves what

| Guard | Proves |
|---|---|
| `test-engine-kind-resolver` | your engine resolves to a kind, and no private classifier crept in |
| `test-engine-lineage` | every curated engine declares a valid lineage; non-stock engines name source and commit; every engine id resolves to a family |
| `test-engine-port-set-drift` | every copy of the engine-port set agrees |
| `scripts/lib/profiles/catalog_rules.py --slug <slug> --network` (or `--engine <id>`) | the part of this contract a machine can check: engine and container family, a discoverable port, the Status header, cache ownership, lineage, an anonymous image pull and the Hugging Face download. Advisory in `catalog.sh register`; required in `preflight-add-model.sh` for a curated slug and in `export_pr.py --check` (`test-catalog-rules`). A network rule that could not run counts as unchecked, never as a pass |
| `test-profiles-compat` | the profile loads, cross-references resolve, and every slug fits a canonical scenario |
| `test-engine-label-matches-compose` | the compose image default equals `install.spec` |
| `test-launch-knobs` | catalogued knobs reach the container through both delivery channels, and every live compose reads each knob that applies to it, or `knob-conformance.json` says why not |
| `test-engine-config-reads` | every key a launcher writes into the engine's own config is one the pinned engine reads (checked in the image when it's local) |
| `test-compose-*` (above) | the compose conventions that cover any engine directory |
| `bash scripts/preflight-add-model.sh <slug>` | scoped triage per slug |
| full `for t in scripts/tests/*.sh; do bash "$t"; done` | authoritative before merge; baseline failures against master |
| c3 suite | c3's own engine handling (not in `scripts/tests`) |

## MANUAL judgment calls — no test decides these

| Call | How |
|---|---|
| Carry it at all? | the proposal issue: what it does that no shipped engine does, measured |
| Lineage class | from the source repo, not the image name |
| Does a derivative keep its base's contract? | compare its port, flags, log lines and metric names with the base on a live boot |
| Base staleness | how far behind our base pin it is, and which fixes it therefore lacks |
| Quality parity | `rebench-full --with-8pack-thinking=both` against a stock-engine slug of the same model, with the effort level stated |
| Server defaults | default effort, thinking budget, template: the same as the model's other slugs, or documented |

## The checklist

1. ☐ Engine proposal issue answered; go-ahead from the maintainer
2. ☐ Lineage class and fields; naming (id, filename, slug prefix, compose directory, container name, internal port)
3. ☐ Image published by the engine's maintainer, digest-pinned, pullable without credentials (`catalog_rules.py --engine <id> --network`); `install.spec` and `image_env`; allowlist arm in **both** launchers
4. ☐ `engines/<id>.yml`, plus hardware KV formats and kv-calc tables (vLLM-type). `test-profiles-compat` counts the profile files itself, so it needs no edit
5. ☐ The engine resolves in `engine-kind.sh` from its id, container name and image (`test-engine-lineage` checks the id); arms and resolver probes for a new family
6. ☐ Discovery: standard internal port, or every mirrored port set updated
7. ☐ Throughput and acceptance parsing verified on a live boot (`bench.sh` and `health.sh` show acceptance)
8. ☐ Launch knobs mapped or declared unsupported; hardware-injection names mapped
9. ☐ Template, default effort and thinking budgets match the model's other slugs, or are documented
10. ☐ Composes follow the conventions above, including the unguarded ones
11. ☐ Registry entries for the first slugs ([ADDING_MODELS.md](ADDING_MODELS.md)), then `preflight-add-model.sh` green for each. Registered locally first (`catalog.sh register-engine`, [BRING_YOUR_OWN.md §4](BRING_YOUR_OWN.md#4-catalog-it-on-your-own-rig--the-local-layer))? `export_pr.py` with `{"engine_id": "<id>"}` bundles the profile, the first slugs and a filled `engine.md` body ([BRING_YOUR_OWN.md §5](BRING_YOUR_OWN.md#5-share-it--contribute-it-back))
12. ☐ Boot each slug with `switch.sh <slug>` from the branch; record `docker inspect --format '{{.Image}}'`; `verify-full`, then `rebench-full`
13. ☐ Full guard suite and the c3 suite green
14. ☐ Docs:
    - `docs/engines/<ENGINE>.md` (template: `EXLLAMAV3.md` / `LLAMACPP_CLUB3090.md`)
    - `docs/engines/README.md`
    - `docs/README.md`
    - the `AGENTS.md` engine-pinning table
    - `docs/UPSTREAM.md` (a pinned-images row: why it's pinned, which composes use it, when it can retire; plus an upstream section). The pin itself goes in the profile, as `install.spec` and a `# pin-current:` line beside it, not in that row
    - an engine-specific section in `docs/NIGHTLY_BUMP_RUNBOOK.md`
    - BENCHMARKS rows and the model CHANGELOG

## Common pitfalls (each shipped a real gap)

| Pitfall | Symptom | Fix |
|---|---|---|
| Engine id and containers without the family prefix | id-based checks fall back to the profile's `type`, but serving scripts that classify by container or image read `unknown`: `health.sh` blank, vLLM log parsing lost | `<family>-<name>` for the id and the containers |
| Non-standard internal port | `verify*.sh`, `bench.sh` and `health.sh` can't find the server without `URL=` and `CONTAINER=` | the family's standard port |
| Classifier written outside `engine-kind.sh` | the seventh and eighth private classifiers (#1341; #1247 skipped the weights check on 30 slugs) | delegate to `engine-kind.sh` |
| Registry `chat_template` doesn't match what the compose serves | a later system message raises; tool keys unsorted; effort defaults differ | mount and `--chat-template`, or label it `native` and document it |
| Profile filename ≠ id | four tools open profiles by filename and silently skip it | filename == id |
| Trialling an image through a *saved* `*_IMAGE` setting | the profile pin overrides saved values, so an A/B "on v0.30.0" actually ran v0.29.0 | export it in the shell for that launch (switch.sh announces it), or bump `install.spec` in a worktree; `docker inspect` the container either way |
| Root-owned compile caches | `~/.cache/club-3090` needs sudo to clean | `user:` GID line plus `umask 0002` |
| "The engine commit is done" | exl3 needed about a dozen follow-ups across discovery, parsers, ports and the gateway | work the tooling contract above before the first PR |

## Worked examples

- **New family: `exllamav3` (TabbyAPI).**
  - The engine commit `87574338` covered the profile, registry, three composes, two test count bumps and the engine page.
  - The tooling followed in later commits:
    - `engine-kind.sh` arms and parsers (`90d44788`, `fbadccff`);
    - the env-prefix guard (`3086161a`);
    - registry-derived container discovery (`5fb41c24`);
    - port 5000 and `/v1/model` detection (`0dab6d9c`, #1360);
    - the gateway Responses bridge.
  - That follow-up list is this page's checklist in practice.
- **Stock, registered first: `sglang-stable` (#1228).** The profile and patch contract landed with no compose targeting it yet: "registered but not wired".
- **Fork, our own: `llamacpp-club3090-*`.** Digest-pinned images built from the fork branch, one profile per build (`-v1.5`, `-v1.6`) because slugs stay on different builds; attribution of the upstream moe-cache work in the engine page.
- **Derivative, third party: `vllm-lmcache`.** The maintainer's own image, pinned by digest because its tag is mutable; id `vllm-…`, slugs `vllm/…`, its own compose directory.
