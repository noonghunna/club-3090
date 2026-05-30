# Adding a model to the club-3090 stack

End-to-end workflow for onboarding a new model into the **curated profile catalog** + serving infrastructure. Pairs with [KV_MATH.md](KV_MATH.md) (math reference) and [ARCHITECTURE.md](ARCHITECTURE.md) (current stack state).

> **Adding a model? Three paths — pick the lightest that fits:**
>
> 1. **Serve any safetensors repo locally (no catalog).** `scripts/pull.sh <org/Model> --profile-like vllm/minimal --dry-run` evaluates *any* safetensors HF repo against this stack's KV math (no download) and tells you whether it fits + at what confidence; drop `--dry-run` and add `--yes` to download + generate a minimal compose + boot. vLLM / safetensors only. See [PULL.md](PULL.md).
> 2. **Run your own GGUF locally (no catalog).** `pull.sh` doesn't take GGUF — use the [local-GGUF recipe](#run-a-local-gguf-without-the-catalog) below (copy an existing compose, 3 steps, llama.cpp / ik-llama).
> 3. **Promote a model into the curated catalog** — *this page*. The heavier task: validated composes, profile-compat coverage, calibration anchors, real benchmarks, per-model gotchas. The high-confidence backbone — **not** a prerequisite for serving.

## Run a local GGUF without the catalog

Want to serve a GGUF you grabbed yourself (a community quant, your own conversion) on llama.cpp or ik-llama, *without* the full catalog workflow? Three steps:

1. **Drop the GGUF** at `/mnt/models/huggingface/<your-name>-gguf/<file>.gguf` (weights live on `/mnt/models` — disk-hygiene rule).
2. **Copy an existing compose** for the engine as a starting point, e.g.:
   - llama.cpp: `models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml`
   - ik-llama:  `models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp.yml`

   Copy it **outside the repo tree** (e.g. `/tmp/my-model.yml`) so you don't have to re-figure the `../` mount depth, point its `--model` / `GGUF_FILE` default at your `.gguf`, and tune `CTX_SIZE`, `KV_TYPE` (`q4_0` = max ctx · `q8_0` = higher fidelity, ~half the ctx), container name + port.
3. **Boot it directly:** `MODEL_DIR=/mnt/models/huggingface docker compose -f /tmp/my-model.yml up`.

No `compose_registry.py` entry, no profile YAML, no calibration. You give up `launch.sh`/`switch.sh` discovery, the VRAM projection, and the guard tests — but you get a one-off local serve in minutes. When you want it discoverable + measured, do the full workflow below.

## When to add a new model vs a new quant of an existing one

| Scenario | Action |
|---|---|
| New base model (different family, different params, different architecture) | Full workflow below — new ModelProfile YAML, new calibration anchors, possibly new compose layouts |
| New quant of an existing model (e.g. AWQ alongside AutoRound INT4) | Lightweight — add weight variant to existing ModelProfile, new compose, no new calibration anchors needed (KV math is unchanged) |
| New drafter for an existing model | Just a DrafterProfile YAML + COMPOSE_REGISTRY entries; no model-level changes |
| New engine for an existing model | EngineProfile YAML; no model-level changes |

This doc covers the **new base model** case. The others are addressed in the v0.7.0 profile model design.

## Workflow at a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Source the architecture facts (config.json + README + code)     │
│  2. Author ModelProfile YAML  (scripts/lib/profiles/models/<id>.yml) │
│  3. Build the first compose  (.../compose/<topology>/<quant>/*.yml) │
│  4. Add COMPOSE_REGISTRY entries  (scripts/lib/profiles/...)        │
│ 4b. Profile-catalog compatibility (engine/hardware/patches.yml)     │
│  5. Boot + verify-full + capture the boot log                       │
│  6. Author CalibrationData YAML  (scripts/lib/profiles/calibration/) │
│  7. Validate: diagnose-profile + the FULL guard suite               │
│  8. Run rebench-full to populate BENCHMARKS.md                      │
│  9. Update CLAUDE.md, ARCHITECTURE.md, learnings/<model>.md         │
└─────────────────────────────────────────────────────────────────────┘
```

Steps 1-2 are pure data; steps 3-5 are infra; steps 6-9 are integration + measurement.

## Step 1 — Source the architecture facts

Some fields come from `config.json` automatically. Others require reading the model card README or model code. See [KV_MATH.md → Extracting parameters from Hugging Face config.json](KV_MATH.md#extracting-parameters-from-hugging-face-configjson) for the full matrix.

### Auto-extractable (config.json)

```python
import json
config = json.load(open(f"/mnt/models/huggingface/<model-id>/config.json"))
# Reliable:
config["num_hidden_layers"]
config["num_attention_heads"]
config["num_key_value_heads"]
config["head_dim"]                  # or hidden_size / num_attention_heads
config.get("sliding_window")        # when present
config.get("num_experts")           # MoE only
config.get("num_experts_per_tok")   # MoE only
```

### README-only or empirical

- **Growing-layer count** — for hybrid architectures (DeltaNet, SWA), the SPLIT between attention-with-growing-KV vs fixed-window/recurrent is usually only in the model card README.
- **K=V tying** — rarely a config field. Check model code (`modeling_*.py`) for `attn_kv_tie` / `attention_k_eq_v` or empirically verify against the boot log.
- **Recurrent state dim** for DeltaNet / Mamba (e.g. `linear_num_k_heads`, `linear_key_head_dim` — present in newer Qwen configs but model-family-specific).
- **Asymmetric head_dim** between layer types (e.g. Gemma 4: sliding `head_dim=256`, global `global_head_dim=512`).

### Write the architecture facts down before touching code

Before authoring the YAML, fill in this table for the new model:

| Field | Value | Source |
|---|---|---|
| Total layers | | config.json `num_hidden_layers` |
| Growing-attention layers | | README pattern |
| Sliding / recurrent layers | | inferred (total - growing) |
| KV heads | | config.json `num_key_value_heads` |
| Head dim | | config.json `head_dim` |
| Asymmetric global head dim? | | config.json or README |
| K=V tied? | | model code or boot log empirically |
| Sliding window (if SWA) | | config.json `sliding_window` |
| MoE? num_experts? active? | | config.json or README |
| Quant formats available | | HF model page |

This becomes your **single source of truth** for the YAML.

## Step 2 — Author the ModelProfile YAML

Drop the file at `scripts/lib/profiles/models/<id>.yml`. Loaded automatically by `load_profiles()`. Cross-references validated at startup.

### Schema reference

See `scripts/lib/profiles/compat.py → ModelProfile` for the live schema. Required fields (as of v0.7.0):

```yaml
schema_version: 1
id: <model-id>                          # e.g. "qwen3.6-35b-a3b"
display_name: <Human-readable name>
family: <family-tag>                    # e.g. "qwen3-next", "gemma-4"

# Architecture (drives kv-calc.py + fits() C2/C10/C11)
num_hidden_layers: <int>
num_growing_layers: <int>               # the KV-growing subset (== num_hidden_layers for non-hybrid)
num_kv_heads: <int>
num_attention_heads: <int>
head_dim: <int>
attention_type: <full | sliding | hybrid>
max_position_embeddings: <int>
valid_tp: [1, 2, 4]                     # which TP values the architecture supports (head divisibility)

# Hybrid quirks (omit when not applicable)
sliding_window: <int>                   # SWA only
global_head_dim: <int>                  # when global layers use a different head_dim
k_v_tensors: <1 | 2>                    # 1 when K=V tied, 2 otherwise
recurrent_state_dim: <int>              # DeltaNet/Mamba models
num_global_layers: <int>                # SWA hybrids
num_sliding_layers: <int>               # SWA hybrids
num_recurrent_layers: <int>             # DeltaNet hybrids

# MoE quirks (omit when not applicable)
num_experts: <int>
num_experts_per_tok: <int>
active_params_b: <float>                # for documentation; not in fits()

# Weight variants (drives fits() C14) — a MAP keyed by quant-slug, NOT a list.
# The slug is the SAME string in three places: this key == the compose
# `<quant>/` dir == compose_registry `weights_variant`. A provider repo with
# N quant files → N sibling slugs sharing one `hf_repo`, differing by `files:`.
weights:
  autoround-int4:                          # ← quant-slug = the map key
    path: <id>-autoround-int4              # relative to /mnt/models/huggingface
    local_subdir: <id>-autoround-int4
    size_gb: <float>
    format: autoround                      # autoround | awq | gguf | …
    status: production                     # production | experimental | community-experimental
    hf_repo: <Org/Repo>
    files: ["*.safetensors"]               # or explicit GGUF filenames
    engine: vllm                           # vllm | ik-llama | llama-cpp
    kind: main                             # main | draft | mmproj | gguf
    verify_glob: "*.safetensors"           # or "*.gguf"
  ubergarm-iq4ks:                          # second quant of the same model (own slug dir)
    path: <id>-gguf/ubergarm-mtp-iq4ks
    local_subdir: <id>-gguf/ubergarm-mtp-iq4ks
    size_gb: <float>
    format: gguf
    status: production
    hf_repo: <Org/GGUF-Repo>
    files: ["<File>.gguf"]
    engine: ik-llama
    kind: gguf
    verify_glob: "*.gguf"
default_weight_variant: autoround-int4

# Drafter compatibility (drives fits() C7-C9)
compatible_drafters:
  - <drafter-id>
  - <drafter-id>

# Vision support (drives fits() workload matching)
vision_capable: <bool>
```

### Critical: cross-reference validation

`load_profiles()` validates that every `compatible_drafters` entry has a matching `scripts/lib/profiles/drafters/<id>.yml`. If you reference a drafter that doesn't exist, you'll get:

```
CrossReferenceError: models/<your-id>.yml references unknown drafter `<id>`.
Available drafters: ...
```

Fix by either adding the drafter YAML or removing the reference.

## Step 3 — Build the first compose

Place at `models/<model-id>/<engine>/compose/<topology>/<quant-slug>/<serving>.yml`:

- Engines: `vllm`, `llama-cpp` today (others when we add them)
- Topologies: `single`, `dual`, `multi4`
- Quant slug: exactly matches the `weights_variant` key (`autoround-int4`, `awq`, `unsloth-q4km`, etc.)
- Serving filename: the feature stack only (`fp8-mtp.yml`, `turbo.yml`, `dflash.yml`, etc.). Do not create `docker-compose.yml` or `default.yml`; defaults are registry pointers.

### Required env-var hooks (post-v0.7.0)

Every compose under `models/*/<engine>/compose/**/*.yml` must accept the estate-planner env overrides with single-mode fallback defaults:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=${ESTATE_GPUS:-<default-gpu-list>}
ports:
  - "${BIND_HOST:-0.0.0.0}:${ESTATE_PORT:-${PORT:-<default-port>}}:8000"
container_name: ${ESTATE_CONTAINER:-<default-name>}
```

Fallback defaults preserve single-mode boot. The estate orchestrator overrides per instance.

### vLLM-specific compose conventions

- Match other model composes for the same engine (look at `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/` or `models/gemma-4-31b/vllm/compose/dual/autoround-int4/` as templates)
- Genesis pin must match the rest of the stack (currently v7.72.2)
- Set `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`, `--kv-cache-dtype` based on KV math projections
- Note any required vendored overlays (Marlin pad, DFlash + KV-quant, qwen3coder tool parser, etc.)

### llama.cpp-specific compose conventions

- Look at `models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/` for the template
- GGUF path under `/mnt/models/huggingface/<id>-gguf/`
- `--ctx-size`, `--n-gpu-layers`, `--parallel`

## Step 4 — Add COMPOSE_REGISTRY entries

In `scripts/lib/profiles/compose_registry.py`, add one entry per compose. Use the `_entry(...)` helper that's already defined:

```python
COMPOSE_REGISTRY = {
    # ... existing entries ...
    "vllm/<model-slug>": _entry(
        model="<model-id>",                          # matches models/<id>.yml
        weights_variant="autoround-int4",            # the quant-slug (weights map key == compose <quant>/ dir)
        workload="long-ctx-single",                  # one of the 5 workload IDs
        engine="vllm-nightly-mtp",                   # matches engines/<id>.yml (and its supported_model_families!)
        drafter="qwen-mtp-builtin",                  # or None
        kv_format="turboquant_3bit_nc",              # must be in the hardware profile's supported_kv_formats
        tp=2,
        max_ctx=180000,
        max_num_seqs=1,
        mem_util=0.92,
        compose_path="models/<model-id>/vllm/compose/dual/autoround-int4/<serving>.yml",
        default_port=8040,                           # MUST equal the compose's ${PORT:-NNNN} fallback (parity test)
        kvcalc_key="<model-id>:dual",                # vLLM: "<model>:<kvcalc-profile>"; llama.cpp/ik: "SKIP"
        required_engine_features=["turboquant_3bit_nc"],
    ),
}
```

### Picking `default_port`

Look at what other models use:

- Qwen 3.6 27B vLLM: 8010 / 8011
- Qwen 3.6 27B llama.cpp: 8020 (8011 also seen)
- Gemma 4 31B vLLM: 8030

New models pick the next free 20-slot block (8050, 8070, ...). Wizard uses this as the suggested next-free port when planning estates.

### Workload selection (5 options)

| Workload | Use when |
|---|---|
| `long-ctx-single` | max_num_seqs=1, max_ctx ≥ 180K, throughput-priority is depth |
| `multi-stream-tenant` | max_num_seqs ≥ 4, max_ctx ~100K, throughput is concurrency |
| `tool-heavy` | max_num_seqs=1, max_ctx 50-100K, prefill-optimized for IDE-agent flows |
| `vision-coding` | vision required, max_ctx 130-185K |
| `fast-chat` | max_num_seqs ≥ 4, max_ctx ≤ 32K, decode-throughput-priority |

If unsure, run `fits()` against the proposed combination first — `compat.fits()` will tell you which workloads validate.

## Step 4b — Profile-catalog compatibility (do NOT skip — the easy-to-miss class)

A registry entry isn't enough: every entry must **validate against the profile catalog**, or `test-profiles-compat` / `diagnose-profile` go red. New `(model, engine, KV-format)` combos commonly trip one of these — each is a one-line data fix *once you know it exists* (this whole section is the lesson of hotfix #236):

| Symptom (constraint) | Fix |
|---|---|
| `C10: engine <e> supported_model_families=[…] excludes <family>` | Add the model's `family` to `scripts/lib/profiles/engines/<engine>.yml` → `supported_model_families` (only if the engine genuinely serves it — e.g. llama.cpp does serve `qwen3-next-moe`). |
| `C5: kv_format=<fmt> not supported by hardware: <card>` | Add `<fmt>` to `supported_kv_formats` in the relevant `scripts/lib/profiles/hardware/*.yml`. KV quants like `q5_0`/`q8_0` are *software* (valid on any CUDA card) — add them to **all** hardware profiles, not just yours. |
| `composes with no fitting canonical scenario: [...]` | Usually a *downstream* symptom of the two above — fix the engine/hardware constraint and the entry validates on an existing scenario in `scripts/lib/profiles/canonical_scenarios.py` (the 9 are hardware topologies; you rarely add new ones). |

**If the model ships a vendored chat-template** (a `.jinja` mounted into the container), register it in `scripts/lib/profiles/patches.yml` with `delivery_mechanism: chat_template`, the `load_bearing_when` composes, and a `drift_guard.check` that **encodes the symmetric restart+settle protocol** (the check string must contain `symmetric`, `docker restart`, `settle`, `>=3`, `grand mean` — copy the `qwen-froggeric-chat-template` entry as the template). Otherwise `test-patch-attribution` flags it as an orphan. `status` must be one of `verified | unverified | suspect` (use `unverified` until live-validated).

**Two more easy-to-forget mechanics:**
- **Catalog-size guard** — `scripts/tests/test-compose-registry-disk.sh` asserts a hardcoded compose/registry count; bump it by the number of composes you added (`47 → 55`-style).
- **`+1 ../` mount depth** — composes live at `<topology>/<quant>/<serving>.yml`, one level deeper than the old flat layout, so relative mounts need an extra `../` (e.g. `${MODEL_DIR:-../../../../../../models-cache}`, 6× for `single/<quant>/`). `test-compose-mounts-resolve` catches a wrong depth.

**Launchers are registry-derived (since v0.8.x) — do NOT edit `launch.sh`/`switch.sh`.** Adding the registry entry makes the variant launchable automatically; `<engine>/default` and `<engine>/<topology>/default` resolve via the registry's `DEFAULTS` map (topology-autodetect). Promoting a new default = one line in `DEFAULTS`, never a `default.yml` file.

**`<model>/default` resolves via `ENGINE_PREFERENCE` — add a `DEFAULTS` row per engine you ship.** Your new model is immediately runnable by name (`--model <id>` / `<id>/default`). For `<id>/default` to resolve cleanly on a given topology, that `(model, engine, topology)` must have a **functional** `DEFAULTS` entry (status `production`/`caveats` — a `preview`/`experimental`/`upstream-gated`/`deprecated` config is skipped, never auto-defaulted). The resolver walks `ENGINE_PREFERENCE[topology]` (single = `[beellama, ik-llama, llamacpp, vllm]`; dual/multi = `[vllm, ik-llama, llamacpp, beellama]`) and picks the first engine with a functional `DEFAULTS` slug, so:
- Add a `DEFAULTS` row for **each engine × topology** you want `<model>/default` to cover. With no functional default at the detected topology the resolver emits a notice and falls back to the nearest-lower topology, else a clear "pick explicitly" message (it never crashes) — fine while a model is still validating, but the bare-launch UX wants at least one production row.
- **`RECOMMENDED_DEFAULT_MODELS` is intentionally NOT auto-grown** — a new model is runnable + resolvable but is never the bare-`launch.sh` auto-default until you explicitly add its id to that shortlist. Leave it alone unless you mean to promote the model to a first-class default.

## Step 5 — Boot + verify-full + capture the boot log

First-boot validation:

```bash
bash scripts/launch.sh --variant <model-slug>/<variant>
# or via the estate wizard:
bash scripts/launch.sh --estate
```

Then:

```bash
# Capture the boot log — needed for calibration
docker logs <container-name> 2>&1 | grep -iE "kv cache|model length|gpu memory|allocated" | tee /tmp/<model-id>-boot.log

# Run the verify suite
bash scripts/verify-full.sh

# Stress-test long context
bash scripts/verify-stress.sh
```

### Critical numbers to capture from boot log

```
Available KV cache / card = X GiB
```

Back-solve `per_token_bytes` from this against your `max_ctx` and `max_num_seqs`. Compare against the KV_MATH formula:

```
predicted_per_token_bytes = num_growing_layers × num_kv_heads × head_dim × k_v_tensors × bpe
measured_per_token_bytes  = (Available_KV × 1024^3) / (max_ctx × max_num_seqs / TP)
```

If they differ by 2×, suspect K=V tying (either the model is tied and you missed it, or vice versa). Fix the YAML and re-validate.

## Step 6 — Author CalibrationData YAML

After 4+ measured boot peaks (varying KV format / max_ctx / TP), author `scripts/lib/profiles/calibration/<model-id>.yml`:

```yaml
schema_version: 1
model: <model-id>
rows:
  - compose: vllm/<variant>
    vram_gb: 24
    measured_peak_gb: <X.X>
    ctx_override: null                  # or specific override
    status: active                      # active | stale | historical
    engine_pin: vllm-nightly-<sha>
    genesis_pin: v7.72.2
    source: "BENCHMARKS.md#<model-id> <variant> @<user> <date>"
  # ... more rows ...
```

### Status field

- `active`: current pin still matches; calibration is fresh
- `stale`: engine moved, recalibration recommended but not urgent
- `historical`: kept for the audit trail; predictions use newer rows

### Calibration accuracy gate

After authoring the calibration YAML, run:

```bash
bash tools/kv-calc.py --calibration
```

Look at the verdict accuracy per model. Target: **≥80% within ±1.5 GB**.

If accuracy is poor, the activation coefficient in `tools/kv-calc.py` needs tuning for this model. The coefficient lives in `MODEL_SPECS` or activation coefficient dicts (see Phase 3 refactor for the current home).

## Step 7 — Validate: per-compose triage + the FULL guard suite

```bash
# Per-compose triage: registry → cross-ref → fits() vs canonical scenarios →
# kv-calc projection → calibration freshness → vendored-overlay matching.
bash scripts/diagnose-profile.sh <engine>/<your-variant>

# Run the WHOLE suite — not just the one test you think is relevant.
for t in scripts/tests/*.sh; do echo "== $t =="; bash "$t" >/tmp/$(basename "$t").log 2>&1 && echo "  PASS" || echo "  FAIL — see /tmp/$(basename "$t").log"; done
```

The catalog-relevant gates and what each guards:

| Test | Guards |
|---|---|
| `test-compose-registry-disk` | every `compose_path` exists · descriptive filename (no `docker-compose.yml`/`default.yml`) · quant-slug ↔ `weights_variant` · the **catalog-size count** |
| `test-compose-mounts-resolve` | every relative mount resolves (the **`../` depth**) |
| `test-model-weights-registry` | every `weights_variant` has a weights entry in `models/<id>.yml` |
| `test-switch-registry-parity` + `test-launch-registry-parity` | the variant is launchable from both launchers · `default_port` matches the compose's `PORT` fallback |
| `test-default-resolver` | `<engine>/default` topology-autodetect resolves |
| `test-profiles-compat` | every entry fits a canonical scenario (Step 4b) |
| `test-patch-attribution` | any vendored chat-template is registered (Step 4b) |
| `tools/kv-calc.py --calibration` | VRAM projection accuracy ≥80% |

> **Run the full suite, not the single test you assume is relevant.** A narrow gate-subset shipped a model with two real catalog gaps that only `test-profiles-compat` + `test-patch-attribution` caught (#236). Conversely, some failures are **pre-existing or environmental** (fixture-absent tests, or `.pull-captures` cross-contamination between tests) — **baseline against the last release tag** (`git worktree add /tmp/base vX.Y.Z && run there`) to separate a real regression from pre-existing noise before you treat it as a blocker.

## Step 8 — Run rebench-full + update BENCHMARKS.md

```bash
bash scripts/rebench-full.sh
```

This runs the canonical 5-step bench matrix (bench, verify-stress, quality, soak, aider). Result goes into BENCHMARKS.md per the existing per-model section pattern.

Required BENCHMARKS.md columns per row: TPS (narrative + code), context, VRAM peak per card, KV format, drafter, AL (if spec-decode), engine pin, Genesis pin, date.

## Step 9 — Update CLAUDE.md, ARCHITECTURE.md, learnings/<model>.md

### CLAUDE.md

Already has a 7-step "When the user adds a new model" checklist at stack level (quant format → download → LiteLLM route → ARCHITECTURE.md update → serve test → learnings doc → record TPS). Cross-reference this doc from the v0.7.0-specific extension if you want to deepen it.

### ARCHITECTURE.md

Add the new model to the Storage Layout + GPU Mode Switcher sections. If it gets a new LiteLLM route, add that under the LiteLLM Routing section.

### learnings/<model>.md

Create per the canonical template in CLAUDE.md:

- Role on this stack
- Key architectural properties
- Quant decision table
- Speculative decoding status
- Framework comparison summary
- Serving mode selection
- Model-specific gotchas
- KV cache comparison
- Context ceilings
- Future re-tests

This is the **append-only history** for everything you learn about this model. Don't delete old findings even when superseded — add new sections.

## Worked example: adding Qwen 3.6 35B-A3B (MoE)

Suppose we're adding Qwen 3.6 35B-A3B. Per [KV_MATH.md](KV_MATH.md#qwen-36-35b-a3b-moe--per-card-budget-components):

**Architecture facts** (sourced from config.json + model card):

| Field | Value | Source |
|---|---|---|
| Total layers | 40 | `num_hidden_layers` |
| Growing-attention layers | 10 | README pattern: `10 × (3× GDN → MoE → 1× Gated Attn → MoE)` |
| Recurrent layers | 30 (GDN) | inferred |
| KV heads | 2 | `num_key_value_heads` |
| Head dim | 256 | `head_dim` (for gated-attention) |
| K=V tied? | No | Qwen3-Next family convention |
| MoE: num_experts | 128 (verify) | `num_experts` |
| MoE: experts_per_tok | 8 (verify) | `num_experts_per_tok` |

**ModelProfile YAML** (`scripts/lib/profiles/models/qwen3.6-35b-a3b.yml`):

```yaml
schema_version: 1
id: qwen3.6-35b-a3b
display_name: Qwen 3.6 35B-A3B (MoE)
family: qwen3-next
num_hidden_layers: 40
num_growing_layers: 10
num_recurrent_layers: 30
num_kv_heads: 2
num_attention_heads: 16          # estimate; verify from config.json
head_dim: 256
attention_type: hybrid
max_position_embeddings: 262144
valid_tp: [1, 2, 4]
k_v_tensors: 2
num_experts: 128
num_experts_per_tok: 8
active_params_b: 3
weights:
  - id: autoround-int4
    format: hf_safetensors
    path: /mnt/models/huggingface/qwen3.6-35b-a3b-autoround-int4
    size_gb: 23                  # estimate; update after download
    status: production
default_weight_variant: autoround-int4
compatible_drafters:
  - qwen-mtp-builtin             # check if MoE has matching drafter
vision_capable: false
```

**First compose** (`models/qwen3.6-35b-a3b/vllm/compose/dual/autoround-int4/preview.yml`):

Mirror `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml` shape, swap model path + adjust `--max-model-len` based on KV_MATH projections.

> _Illustrative add-time snapshot. This model was later validated + promoted out of preview: the live dual compose is now `dual/autoround-int4/fp8.yml` (slug `vllm/qwen-35b-a3b-dual`, **262K + vision**, no MTP, vLLM v0.22.0) — see BENCHMARKS.md._

**Per-card budget projection from KV_MATH**:

- Weights / 2 = ~11.5 GB (INT4)
- KV pool @ 200K, fp8, seqs=1 = ~1 GB (MoE is KV-light)
- Activation peak = ~6 GB (estimate; 30 GDN layers × per-layer coefficient ~115 fp8)
- Overhead = ~1.2 GB
- **Predicted peak = ~19.7 GB** → fits comfortably on 24 GB

**COMPOSE_REGISTRY entry**:

```python
"vllm/qwen36-35b-a3b": _entry(
    model="qwen3.6-35b-a3b",
    weights_variant="autoround-int4",
    workload="long-ctx-single",
    engine="vllm-nightly-mtp",
    drafter="qwen-mtp-builtin",
    kv_format="fp8_e5m2",
    tp=2,
    max_ctx=200000,
    max_num_seqs=1,
    mem_util=0.92,
    compose_path="models/qwen3.6-35b-a3b/vllm/compose/dual/autoround-int4/preview.yml",
    default_port=8050,
    required_engine_features=[],
),
```

**Boot, then calibrate**: capture `Available KV cache / card`, back-solve `per_token_bytes`, compare to predicted `2 × 2 × 256 × 2 × 1 = 2048 bytes` at fp8 TP=2. If measured matches → ship the calibration row. If 2× off → check K=V tying. If completely off → check the growing-layer count assumption.

## Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Used `num_hidden_layers` instead of `num_growing_layers` | KV pool predicted 3-6× larger than measured | Read model card README; encode the split in YAML |
| Missed K=V tying | KV pool predicted 2× larger than measured | Check boot log; set `k_v_tensors: 1` |
| Used `head_dim` for asymmetric model | Gemma 4 31B predictions off by 2× on global-layer rows | Add `global_head_dim` to YAML; split KV formula |
| Hardcoded activation coefficient from another model | Wrong predictions across all KV formats | Each model needs ≥4 calibration anchors + its own coefficient |
| Forgot `default_port` in COMPOSE_REGISTRY | Estate wizard fails to suggest a port | Add the field; use next-free 20-slot block |
| New compose missing `ESTATE_GPUS` / `ESTATE_PORT` env hooks | Single-mode works; estate-mode breaks on multi-instance | Audit + patch; verify via `grep -L 'ESTATE_GPUS' <compose>` |
| Cross-reference to undefined drafter | `CrossReferenceError` at load time | Add the drafter YAML or remove the reference |
| MoE model: counted active params as loaded budget | Predicted ~3 GB weights, measured ~22 GB | All experts live in VRAM; use TOTAL params for budget |

## Final PR checklist

When the new model is ready for review:

- [ ] `scripts/lib/profiles/models/<id>.yml` lands with all required fields + `schema_version: 1`
- [ ] At least one compose at `models/<id>/<engine>/compose/...` with `ESTATE_GPUS` + `ESTATE_PORT` + `ESTATE_CONTAINER` env hooks + sensible fallback defaults
- [ ] COMPOSE_REGISTRY entries added with `default_port` (== compose `PORT` fallback) + `kvcalc_key`
- [ ] **Profile-catalog compat (Step 4b):** engine `supported_model_families` covers the family · hardware `supported_kv_formats` covers the KV format · any vendored chat-template registered in `patches.yml` · `test-compose-registry-disk` size-count bumped · `../` mount depth correct
- [ ] The **full** `scripts/tests/*.sh` suite is green (or only pre-existing/env fails, baselined vs the last release tag)
- [ ] `bash scripts/launch.sh --variant <slug>` boots cleanly + `verify-full.sh` 8/8 PASS
- [ ] Boot log captured + reviewed against KV_MATH projections (per_token_bytes within 10% of formula)
- [ ] `scripts/lib/profiles/calibration/<id>.yml` populated with ≥4 measured rows
- [ ] `bash tools/kv-calc.py --calibration` verdict accuracy ≥80% on this model
- [ ] `bash scripts/diagnose-profile.sh <slug>` GREEN
- [ ] BENCHMARKS.md row added (TPS + ctx + VRAM + KV + drafter + AL + engine pin + Genesis pin + date)
- [ ] `learnings/<id>.md` populated per the canonical template
- [ ] CLAUDE.md + ARCHITECTURE.md cross-references updated

## See also

- [KV_MATH.md](KV_MATH.md) — KV cache math reference (formulas, per-model derivations, error bands)
- [AGENTS.md](../AGENTS.md) — the in-repo agent guide; its "Adding a model / quant" section is the entry point an AI agent working in a clone should read first, and links back here for the full workflow
- [ARCHITECTURE.md](ARCHITECTURE.md) — current stack state to update
- [BENCHMARKS.md](../BENCHMARKS.md) — the measured-data home for new calibration rows
- `scripts/lib/profiles/compat.py` — live ModelProfile schema definition
- [`tools/kv-calc.py`](../tools/kv-calc.py) — the predictor's actual implementation
