# Adding a model to the club-3090 stack

End-to-end workflow for onboarding a new model into club-3090's **central registry** — `scripts/lib/profiles/compose_registry.py` (the single source of truth) plus the profile / engine / drafter / calibration YAMLs it points at. Registering a model here is what makes it a first-class catalog citizen: **`launch.sh` and `switch.sh` both resolve it by slug** (both are registry-derived — you never edit the launchers), the VRAM/KV projection (`kv-calc`) knows it, and the guard tests cover it. Pairs with [KV_MATH.md](KV_MATH.md) (math reference) and [ARCHITECTURE.md](ARCHITECTURE.md) (current stack state).

> **Adding a model? Three paths — pick the lightest that fits:**
>
> 1. **Serve any safetensors repo locally (no catalog).** `scripts/pull.sh <org/Model> --profile-like vllm/minimal --dry-run` evaluates *any* safetensors HF repo against this stack's KV math (no download) and tells you whether it fits + at what confidence; drop `--dry-run` and add `--yes` to download + generate a minimal compose + boot. vLLM / safetensors only. See [PULL.md](PULL.md).
> 2. **Run your own GGUF locally (no catalog).** `pull.sh` doesn't take GGUF — use the [local-GGUF recipe](#run-a-local-gguf-without-the-catalog) below (copy an existing compose, 3 steps, llama.cpp / ik-llama).
> 3. **Promote a model into the curated catalog — i.e. register it in the central `compose_registry.py`** — *this page*. The heavier task: validated composes, a registry entry per compose (the SoT that `launch.sh`/`switch.sh` derive from), profile-compat coverage, calibration anchors, real benchmarks, per-model gotchas. The high-confidence backbone — **not** a prerequisite for serving.
>
> Paths 1–2 (serve + **tune** + **validate** your own model without touching core files) are walked end-to-end in [BRING_YOUR_OWN.md](BRING_YOUR_OWN.md) — start there if you're not yet cataloging. This page is the promotion step *after* you've validated a config there.

## Run a local GGUF without the catalog

Want to serve a GGUF you grabbed yourself (a community quant, your own conversion) on llama.cpp or ik-llama, *without* the full catalog workflow? Three steps:

1. **Drop the GGUF** at `/mnt/models/huggingface/<your-name>-gguf/<file>.gguf` (weights live on `/mnt/models` — disk-hygiene rule).
2. **Copy an existing compose** for the engine as a starting point:
   - llama.cpp: `models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml`
   - ik-llama:  `models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp.yml`

   Copy it **outside the repo tree** (e.g. `/tmp/my-model.yml`) so you don't have to re-figure the `../` mount depth, point its `--model` / `GGUF_FILE` default at your `.gguf`, and tune `CTX_SIZE`, `KV_TYPE` (`q4_0` = max ctx · `q8_0` = higher fidelity, ~half the ctx), container name + port.
3. **Boot it directly:** `MODEL_DIR=/mnt/models/huggingface docker compose -f /tmp/my-model.yml up`.

No `compose_registry.py` entry, no profile YAML, no calibration. You give up `launch.sh`/`switch.sh` discovery, the VRAM projection, and the guard tests — but you get a one-off local serve in minutes. When you want it discoverable + measured, do the full workflow below.

## What's automatic now

A large fraction of this workflow used to be hand-edited boilerplate. It isn't anymore — **each row below is machinery you must NOT hand-edit**, guarded by its own test:

| Concern | How it's automatic | You write | Guard |
|---|---|---|---|
| Profile-key typos | **Strict-key loader** (`compat.py → _check_profile_keys`): any YAML key outside the dataclass fields + deliberate `*_EXTRA_KEYS` allowlist is rejected loudly at load — with a did-you-mean suggestion — instead of being silently ignored. Covers `models/ hardware/ engines/ drafters/ workloads/ calibration/` **and** `profiles-local/models.d/`, including the nested `weights.<variant>` dicts and the `setup:` block. | nothing — a typo fails fast | load-time, every consumer |
| Download front door | **`setup.sh` is registry-derived**: usage list, labels, picker, dispatch, `WEIGHTS=` aliases, and the `"Supported: …"` error are all generated from `weights.py catalog --json`; per-model "next steps" come from `registry-emit.sh --json`. A new `models/<id>.yml` gets its front door free. | optional `setup:` block, ONLY when dispatch differs from default (see [ModelProfile schema](#contract--the-modelprofile-schema)) | `test-setup-registry-derived` |
| Launchers | `launch.sh` / `switch.sh` derive variants, defaults, topology-autodetect, and `--list` from the registry (since v0.8.x). Defaults live in the registry's `DEFAULTS` map, never a `default.yml`. | nothing | `test-launch-registry-parity` · `test-switch-registry-parity` · `test-default-resolver` |
| Gateway + URLs | `litellm-emit.sh` derives `services/litellm/config.yaml` routes from the registry's **`gateway=True`** entries (#1078) — no more hand-written `:<port>` route stanzas to forget (#1062). Extra served-names ride `serve_aliases`. | set `gateway=True` on the ONE slug LAN clients should hit | `test-litellm-ports-resolve` · `test-default-url-gateway` |
| Weight-shard integrity | `scripts/preflight.sh` (sourced by `setup.sh` **and** `launch.sh`) runs the #1042 **shard check**: diffs each sharded-safetensors `index.json` weight_map against disk and hard-fails on absent shards *before* boot surfaces it as a 53 KB vLLM traceback. Bypassed by `FORCE=1` / `PREFLIGHT_NO_SHARD_CHECK=1`. | nothing | runs on every setup/launch |
| Reasoning-mode sampler | Models whose card publishes per-mode sampler rows declare `sampler_profiles={"instruct": {…}, "thinking": {…}}` on their `_entry`. Composes DERIVE the sampler from that data: `ENABLE_THINKING=true` (vLLM) / `THINKING=1` (llama.cpp) alone selects the card's thinking row; explicit env still wins. Discussion #993's four-variable ritual is retired. | the `sampler_profiles` kwarg (see [_entry kwargs reference](#contract--compose_registry-entry-kwargs)) | `test-compose-sampler-profiles` |
| Local (community) layer | `scripts/lib/profiles-local/` is a gitignored layer (except its README) loaded with the SAME schema/factories as core: `models.d/<id>.yml` + `composes/<id>/…` + plain-dict `registry.local.json` entries under the enforced `local/` slug namespace. Nothing core is ever touched; delete the files to revert. | run `promote.py --layer local` (the default) | loader refuses non-`local/` slugs; invalid local profiles raise loudly |
| PR export | Validated LOCAL model → ready-to-commit CORE bundle via `export_pr.py`: emits `models/<id>.yml`, the core-layout compose, and a `compose_registry.py` patch into `--out`. | run it after LOCAL validation | `--check` mode; `test_export_pr.py` |

And the **c3 Bring→Promote lane** ties the whole producer path together: the cockpit's Bring & Validate pipeline (**① Bring** fit-check → ② Serve → ③ Tune → ④ Validate → **⑤ Promote**) computes the promotion scaffold from your measured config, writes the chosen layer via `promote.py`, then chains `diagnose-profile` + `preflight-add-model.sh` automatically as one confirmed action plan. Scaffold→write→diagnose→preflight is one button press; the *judgment* that remains is exactly what the MANUAL sections below mark.

Everything else in this doc is either a **CONTRACT** (machine-checked shapes you must satisfy) or a **MANUAL judgment call** (things no test can decide for you).

## CONTRACT — the ModelProfile schema

Drop the file at `scripts/lib/profiles/models/<id>.yml` (core) or `scripts/lib/profiles-local/models.d/<id>.yml` (local layer). Loaded automatically by `load_profiles()`; cross-references validated at startup. Live schema: `scripts/lib/profiles/compat.py → ModelProfile`.

**Strict-key rule (read before inventing a field).** The loader validates every key against the dataclass fields plus a deliberate `*_EXTRA_KEYS` allowlist (`compat.py`, "Strict profile-key validation"). A typo like `verify_glb` next to `verify_glob` is **rejected loudly with a closest-match suggestion**, not silently dropped — and adding any new key means extending the matching `*_EXTRA_KEYS` set in code first. That friction is the point. The nested `weights.<variant>` dicts and the `setup:` block get the same treatment.

The block below is ILLUSTRATIVE, not a universal required set — **the architecture fields are FAMILY-SPECIFIC.** Copy the closest *current* profile and adapt, never this verbatim:

- `gemma-4-12b.yml` / `gemma-4-31b.yml` — dense-SWA: use `num_attn_heads`, `num_full_attn_layers`, `num_sliding_attn_layers`, `head_dim_sliding`, `global_head_dim`, `sliding_window` — NOT a flat `head_dim`/`attention_type`.
- `qwen3.6-27b.yml` — DeltaNet hybrid: `num_gdn_layers`, `linear_*`.
- `qwen3.6-35b-a3b.yml` — MoE.

Family tags are the real ones (`gemma4-swa-dense`, `gemma4-unified`, `qwen3-next-hybrid`, `qwen3-next-moe`), not `gemma-4`/`qwen3-next`. Generic skeleton:

```yaml
schema_version: 1
id: <model-id>                          # e.g. "qwen3.6-35b-a3b"
display_name: <Human-readable name>
family: <family-tag>                    # e.g. "qwen3-next-moe"

# Architecture (drives kv-calc.py + fits() C2/C10/C11)
num_hidden_layers: <int>
num_growing_layers: <int>               # the KV-growing subset (== num_hidden_layers for non-hybrid)
num_kv_heads: <int>
num_attention_heads: <int>
head_dim: <int>
max_position_embeddings: <int>
valid_tp: [1, 2, 4]                     # which TP values the architecture supports (head divisibility)

# Hybrid quirks (omit when not applicable)
sliding_window: <int>                   # SWA only
global_head_dim: <int>                  # when global layers use a different head_dim
k_v_tensors: <1 | 2>                    # 1 when K=V tied, 2 otherwise
recurrent_state_dim: <int>              # DeltaNet/Mamba models

# MoE quirks (omit when not applicable)
num_experts: <int>
num_experts_per_tok: <int>
active_params_b: <float>                # documentation only; NOT in fits()

# Weight variants — a MAP keyed by quant-slug, NOT a list. The slug is the SAME
# string in three places: this map key == the compose `<quant>/` dir ==
# compose_registry `weights_variant`. A provider repo with N quant files → N
# sibling slugs sharing one hf_repo, differing by `files:`.
weights:
  autoround-int4:
    path: <id>-autoround-int4           # relative to /mnt/models/huggingface
    local_subdir: <id>-autoround-int4
    size_gb: <float>                    # honest size — setup.sh's disk gate is sized from it
    format: autoround                   # autoround | awq | gguf | …
    status: production                  # production | experimental | community-experimental
    hf_repo: <Org/Repo>
    revision: <sha-or-tag>              # OPTIONAL (#319): pin the fetch; unset = track HEAD.
                                        #   Set it to reproduce the bytes a BENCHMARKS row measured.
    files: ["*.safetensors"]            # REQUIRED — see "Two silent traps"
    engine: vllm                        # vllm | ik-llama | llama-cpp | beellama (filesystem dir name)
    kind: main                          # main | draft | mmproj | gguf
    verify_glob: "*.safetensors"        # REQUIRED on GGUF ("*.gguf") — default is safetensors

default_weight_variant: autoround-int4
compatible_drafters: [<drafter-id>, …]  # drives fits() C7-C9; must exist as drafter YAMLs
vision_capable: <bool>                  # drives fits() workload matching
```

### Two silent traps in `files:` and `verify_glob`

Both fields are **well-formed YAML when wrong**, so nothing downstream complains — and both burned real users on the DeepSeek-Flash ship (#910, #911).

**1. `verify_glob` defaults to `*.safetensors`.** In a GGUF directory that matches nothing, and the two consumers then blame the *weights*: c3's `weights_state_for()` reports `PARTIAL` (offering a pointless re-download) while every shard is present, and `setup.sh` post-download verify exits 1 with *"download may have failed"* after the pull **succeeded**. A user's rational response is to delete and re-pull hundreds of GB. **Declare `verify_glob: "*.gguf"` on every GGUF entry.**

**2. No `files:` means the fetch is the WHOLE repo.** Correct for a single-artifact bucket; catastrophic for one quant of a multi-quant repo — the DeepSeek GGUF repo holds 13 quants / 1,537 GB, and asking for the 85 GB tier started pulling all of it on a real user's machine.

And `hf download --local-dir` **preserves repo folder structure**, so if the artifact lives in a repo subfolder: `local_subdir` must be the **bucket** (not the quant dir — otherwise it nests twice), `files:` carries the subfolder prefix, and `verify_glob` carries it too (it globs relative to `local_subdir`). A repo-root artifact lands flat — that's how the DSpark drafter gets its own directory. `test-model-weights-registry` guards all of it.

### Cross-reference validation

Every `compatible_drafters` entry must have a matching `scripts/lib/profiles/drafters/<id>.yml`; otherwise you get `CrossReferenceError: models/<your-id>.yml references unknown drafter '<id>'.` Fix by adding the drafter YAML or removing the reference.

### CONTRACT — the `setup:` block (only when dispatch differs from default)

Default policy = primary fetch is `default_weight_variant`, no aliases, no drafters — **add nothing**. Add a `setup:` block ONLY when the model needs something else. Its keys are strict-key validated too:

```yaml
setup:
  primary: fp8                            # ONLY when setup.sh's default fetch must differ from default_weight_variant
  weights_aliases: {gguf: unsloth-q4km}   # WEIGHTS=<alias> → weight-variant key
  alias_extras: {gguf: [gguf_mmproj_f16]} # EXTRA_WEIGHT_KEYS fetched alongside an alias primary
  alias_resets_genesis: true              # a matching alias forces NEEDS_GENESIS=0 (GGUF paths never clone Genesis)
  always_draft: dspark                    # mandatory drafter (fetched on every run)
  assistant_draft: assistant              # WITH_ASSISTANT_DRAFT=1 target
  dflash: dflash2                         # WITH_DFLASH_DRAFT=1 target
  vision: gguf_mmproj_f16                 # WITH_VISION=1 target
  prism_eagle3: prism_eagle3              # WITH_PRISM_EAGLE3=1 target
```

Reference blocks: `qwen3.6-27b.yml` (aliases + extras + drafters), `qwen3.8-27b.yml` (primary override + dflash + vision), `deepseek-v4-flash-0731.yml` (primary override + always-draft), `gemma-4-31b.yml` / `gemma-4-26b-a4b.yml` (assistant + awq alias).

## CONTRACT — compose layout & header

One compose per validated config at `models/<id>/<engine>/compose/<topology>/<quant-slug>/<serving>.yml`:

- **Engine dirs** (filesystem): `vllm` · `llama-cpp` · `ik-llama` · `beellama`. ⚠️ Slug prefixes differ for llama.cpp-family: `llamacpp` / `ik-llama` / `beellama` — never `llama-cpp`.
- **Topologies**: `single` · `dual` · `multi4`. **Quant-slug dir** == the ModelProfile `weights:` map key exactly.
- **Serving filename** = the feature delta from a plain boot (`base.yml`, `fp8.yml`, `mtp.yml`, `fp8-mtp-vision.yml`; drafter→KV→vision order). Never `docker-compose.yml` / `default.yml` — defaults are registry pointers. Workload tunings keep descriptive names (`long-text.yml`). Grandfathered names stay.
- **Relative mounts need one more `../` than flat layouts** (6× for `single/<quant>/`) — `test-compose-mounts-resolve` catches wrong depth.
- **Env hooks mandatory**: `${ESTATE_GPUS:-…}`, `${ESTATE_PORT:-${PORT:-NNNN}}`, `${ESTATE_CONTAINER:-…}` with single-mode fallbacks — estate mode breaks without them.
- **Profile header mandatory**: the `# Profile (at-a-glance):` block with `Status:` — exactly one of ✅ Production · ⚠️ Production w/ caveats · 🧪 Experimental · 🐣 Incubating · 👁️ Preview · ⏸️ Upstream-gated · 🗑️ Deprecated, plus a `Caveats:` line whenever non-✅. **New models start at 🐣 Incubating** (`status="incubating"`, hidden from `--list`, launch needs `--force`) and graduate up as they clear gates. `test-compose-status-drift` fails CI on header↔registry mismatch.

## CONTRACT — `compose_registry._entry` kwargs reference

```python
"vllm/<slug>": _entry(
    model="<model-id>",                # matches models/<id>.yml
    weights_variant="autoround-int4",  # == weights map key == <quant>/ dir
    workload="long-ctx-single",        # long-ctx-single | multi-stream-tenant |
                                       # tool-heavy | vision-coding | fast-chat
    engine="vllm-stable",              # EngineProfile id (supported_model_families must cover the family!)
    drafter="qwen-mtp-builtin",        # or None — display label only
    kv_format="fp8_e5m2",              # must be in hardware profile supported_kv_formats
    tp=2,
    max_ctx=180000,
    max_num_seqs=1,
    mem_util=0.92,
    compose_path="models/<id>/vllm/compose/dual/autoround-int4/<serving>.yml",
    default_port=8050,                 # MUST equal the compose ${PORT:-NNNN} fallback
    kvcalc_key="<model>:<kvcalc-profile>",  # vLLM only; llama family uses "SKIP"
    weights_companions=("<id>:mmproj",),    # REQUIRED when the compose mounts a drafter/mmproj
    status="incubating",               # NEW MODELS START HERE
    status_note=None,                  # REQUIRED string when caveats/preview/upstream-gated/deprecated
    # ── newer kwargs ──
    served_name=None,                  # override; else parsed from the compose (--served-model-name)
    gateway=False,                     # opt-in: LiteLLM gateway route generated for this slug (#1078)
    serve_aliases=(),                  # extra stable served-names on the same route (#1073)
    sampler_profiles=None,             # {"instruct": {…}, "thinking": {…}} — enables sampler coupling + c3 toggle
    weights_companions=(),
)
```

Policy maps live beside the entries and are **core-only**: `DEFAULTS` (one functional row per `(model, engine, topology)` you want `<model>/default` to resolve), `ENGINE_PREFERENCE`, `RECOMMENDED_DEFAULT_MODELS`. Local-layer entries can never appear in them.

## CONTRACT — moe-cache / CPU-offload models (extra gates)

Applies to any model served with `-ot …ffn_(gate|up|down)_exps\.weight=CPU` on the
`llamacpp-club3090` engines. These are gates only — the **mechanism** for every one lives in
[`learnings/moe-cache-engine.md`](../../learnings/moe-cache-engine.md); do not restate it here.

| # | Gate | Why it exists |
|---|---|---|
| M1 | `kv_calc_supported: false` in the model YAML **and** `kvcalc_key: "SKIP"` on every entry | kv-calc has no model for per-layer hybrids, MLA compression, or a DSA indexer budget. Precedent: `deepseek4-moe`, `inkling-moe`, `glm5next-moe` |
| M2 | **The KV-growing layer count must be COUNTED, not assumed** | On hybrids `attention.head_count_kv` is a PER-LAYER ARRAY. GLM-5.3-Flash: 12 growing of 46 blocks — using `block_count` over-predicts the pool ~3.8× |
| M3 | **Expert byte-size READ from the GGUF tensor table**, never inferred from the quant name | Unsloth *Dynamic* quants mix expert types (GLM UD-IQ4_XS carries five: `iq3_s`/`iq4_xs`/`q6_K`/`q3_K`/`q4_K`). "IQ4_XS" does not give you a byte size |
| M4 | **Pool census captured at `LLAMA_ARG_LOG_VERBOSITY=4`, slots SUMMED per device** | `pool[i]` lines print at NO lower verbosity, and a device holds 2-3 pools. Reading one pool's `used=N/N` as the device total under-reported GLM ~4× (518 vs 2,143) |
| M5 | **Drafter placement (`-devd`) MEASURED for this model, never inherited** | It inverts between models: DeepSeek needs `none` (host), GLM needs `CUDA1` (+18.4% code; host is −11%). Takes a device NAME — `1` is invalid |
| M6 | `RESERVE_MB` **re-derived for this card's VRAM** | 1536 was derived against a MEASURED 1,128 MiB compute-buffer swing on 2×24 GB. Too low is NOT an OOM — it is *slower* |
| M7 | **Never benchmark the first request after boot** | Cold-start pool fill. Steady state arrives after one generation |
| M8 | **Tune on wall-clock; cache counters are diagnostics** | Hit rate is not the objective function — three independent sightings, incl. a 27%-smaller pool with a HIGHER hit rate and 9% LOWER throughput |
| M9 | **New engine profile file when the image changes — never repoint an existing one** | The launchers resolve the image from `engines/<engine>.yml` and INJECT it, overriding the compose's `image:` line. Repointing silently migrates every model on that profile onto a build it was not validated on |
| M10 | **Thinking control VERIFIED, not assumed** | A template that NAMES a switch may not honour it, and an unsupported enum value fails silently. `preflight_detect_thinking_control` probes behaviourally under `THINK_PROBE=1`; an unverified switch cost verify-full 4 false failures on GLM |
| M11 | **Host THP verified by COVERAGE (`ShmemHugePages ÷ Shmem`), not by the knob** | Offloaded experts are CUDA pinned host memory, accounted as **Shmem** — so `transparent_hugepage/enabled` does nothing for them and `shmem_enabled` is the governing knob. Setting `enabled=always` alone LOOKS like it worked (AnonHugePages climbs, the heap) while the model stays on 4 KiB pages. ⚠️ **Latency only:** measured −56% short-prompt TTFT; decode and prefill are unchanged, so never quote it as tok/s. `scripts/hugepages.sh` checks and applies; `report.sh` captures it; `preflight` warns (never blocks). |

⚠️ **Multi-shard GGUFs may ship a METADATA-ONLY shard 1** (GLM's is 9,429,859 B with
`tensor_count = 0`; all tensors are in shards 2-5). A probe that reads shard 1 for tensor info
finding nothing is **not** evidence of a broken download.

⚠️ **`-ot` force-disables pipeline parallelism, silently** — so prefill is GPU-asymmetric and
`n_copies`/TurboPrefill are unavailable. Do not plan prefill-overlap work on the assumption they
exist; it was closed NO-GO. This is an engine property, not a per-model defect.

## GATES — which guard proves what

| Guard | Proves |
|---|---|
| `test-compose-registry-disk` | every compose_path exists · descriptive filenames · quant-slug ↔ weights_variant · catalog count |
| `test-compose-mounts-resolve` | relative mount depth resolves |
| `test-model-weights-registry` | every weights_variant has a profile entry · files:/verify_glob contract |
| `test-switch-registry-parity` · `test-launch-registry-parity` | variant launchable from both launchers · default_port parity |
| `test-default-resolver` · `test-model-default-resolver` | `<engine>/default` + `<model>/default` resolve |
| `test-profiles-compat` | every entry fits a canonical scenario (strict loader active here too) |
| `test-compose-status-drift` | header Status emoji ↔ registry status |
| `test-docs-slugs-resolve` | every slug named in docs exists (non-functional carry `--force`) |
| `test-setup-registry-derived` | setup.sh output ≡ catalog for EVERY model+alias |
| `test-compose-sampler-profiles` | shipped sampler defaults == instruct row; THINKING render == thinking row |
| `test-litellm-generate` · `test-litellm-ports-resolve` | gateway block ≡ registry derivation; ports resolve |
| `test-preflight-shards` | absent shards refuse before boot |
| `python3 tools/kv-calc.py --calibration` | VRAM predictions ≥80% within ±1.5 GB |

Run the scoped triage via `bash scripts/preflight-add-model.sh <slug>`; the FULL `for t in scripts/tests/*.sh; do bash "$t"; done` stays authoritative before any commit.

## MANUAL judgment calls — no test decides these

| Call | How |
|---|---|
| Growing-attention layer split | README pattern / model code — NOT `num_hidden_layers`. Wrong split ⇒ KV pool off 3–6× |
| K=V tying (`k_v_tensors`) | Boot log vs prediction; 2× delta ⇒ tying suspect. Wrong ⇒ 2× error |
| Asymmetric head dims | Gemma-style global-vs-sliding splits need `global_head_dim` etc. |
| Calibration anchors | ≥4 measured boots per model, varying KV/ctx/TP; tune the coefficient in the profile YAML, never kv-calc itself |
| Status honesty | Start 🐣 Incubating; promote up ONLY as gates clear (verify-full → stress → soak → bench → quality) |
| Benchmarks + learnings | `rebench-full --with-8pack-thinking=both`; BENCHMARKS.md row (TPS/ctx/VRAM/KV/drafter/engine pin/date) same session; `learnings/<id>.md` per template |

## The checklist (what's actually left to do)

1. ☐ Architecture-facts table filled (config.json + README + model code)
2. ☐ `scripts/lib/profiles/models/<id>.yml` (+ optional `setup:` block)
3. ☐ First compose under `models/<id>/<engine>/compose/...` with ESTATE hooks + honest Status header
4. ☐ Registry `_entry(...)` (+ `DEFAULTS` row per engine×topology ONLY when promoting a default)
5. ☐ `bash scripts/preflight-add-model.sh <slug>` → GREEN
5b. ☐ **moe-cache models only:** gates M1-M10 above cleared, and the pool census captured at
    `LLAMA_ARG_LOG_VERBOSITY=4` with slots summed per device (record it in the `status_note`)
6. ☐ Run `bash scripts/setup.sh <id>` FOR REAL (hand-placed weights don't count)
7. ☐ Boot via `launch.sh --variant` → `verify-full.sh` → capture boot log → compare vs prediction
8. ☐ Calibration rows (vLLM) + `kv-calc.py --calibration` ≥80%
9. ☐ FULL guard suite green (baseline pre-existing failures against last release tag)
10. ☐ `rebench-full --with-8pack-thinking=both` → BENCHMARKS row + `learnings/<id>.md`
11. ☐ CLAUDE.md / ARCHITECTURE.md cross-references

**Or do steps 2–5 from the c3 app:** Bring & Validate lane → ⑤ Promote writes the scaffold for real and chains diagnose + preflight automatically.

## Common pitfalls (each shipped a real gap)

| Pitfall | Symptom | Fix |
|---|---|---|
| `num_hidden_layers` instead of `num_growing_layers` | KV pool predicted 3–6× large | encode the split in YAML |
| Missed K=V tying | prediction 2× large | `k_v_tensors: 1` |
| Flat `head_dim` on asymmetric model | 2× error on global layers | `global_head_dim` + split formula |
| Activation coefficient copied from another model | wrong across all KV formats | own anchors, own coefficient |
| Missing `default_port` | estate wizard can't suggest | add it; next free 20-slot block |
| Missing ESTATE hooks | single-mode fine, estate broken | audit with grep |
| Drafter referenced but no YAML | CrossReferenceError at load | add drafter YAML or drop ref |
| MoE active params counted as loaded budget | ~3 GB predicted, ~22 GB real | TOTAL params for budget |

## Worked example

Qwen 3.6 35B-A3B (MoE) is the canonical walkthrough: 40 layers, 10 growing (README pattern), 2 KV heads × 256 head dim, K/V untied, 128 experts @ 8 active — predicted dual-card peak ~19.7 GB at fp8 TP=2. See [KV_MATH.md](KV_MATH.md) §per-card budget components for the arithmetic and `learnings/qwen3.6-35b-a3b.md` for the as-shipped result.
