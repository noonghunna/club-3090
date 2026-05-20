# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, Copilot, Continue, etc.) working in this repo. Focused — only conventions an agent wouldn't infer from the code itself.

## Read first

Before making non-trivial changes:

- [`README.md`](README.md) — what the repo is, two-routes framing, repo layout
- [`docs/SINGLE_CARD.md`](docs/SINGLE_CARD.md) / [`docs/DUAL_CARD.md`](docs/DUAL_CARD.md) — pick-by-workload guidance + the cliffs they reference
- [`docs/UPSTREAM.md`](docs/UPSTREAM.md) — every upstream issue / PR we depend on or have filed (see "Upstream issues" section below for why this matters)
- [`models/qwen3.6-27b/INTERNALS.md`](models/qwen3.6-27b/INTERNALS.md) — DFlash forensics, AutoRound rationale, Marlin pad fork, MTP head
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — what kinds of PRs land cleanly + benchmark + verify protocol

## Hardware truths

- 2× RTX 3090 Ampere SM 8.6, PCIe-only, **no NVLink** (and we won't add it).
- Custom all-reduce must be disabled in vLLM/SGLang configs (PCIe topology breaks NVLink-assumed paths).
- No native FP8 compute; FP8 KV is a storage optimization only.
- Speculative decoding using EAGLE / DFlash is blocked on Qwen3-Next family (DeltaNet rollback). MTP works. See [`docs/UPSTREAM.md`](docs/UPSTREAM.md) — vllm#39931.

## Upstream issues — single source of truth

`docs/UPSTREAM.md` tracks every upstream issue / PR we depend on, have filed, or use as context. **Before filing a new upstream issue or referencing an existing one in code/docs, check this file.** When status changes (issue closed, PR merged, pin bumped), update the row.

This rule exists because we previously had upstream links scattered across CHANGELOG, INTERNALS, FAQ, and per-compose comment headers — and they drifted. The tracker file is the canonical place; cross-link to it from anywhere else.

When filing a fresh upstream issue from this work:
1. Add the row to `docs/UPSTREAM.md` first
2. Link the issue back to `noonghunna/club-3090` in the body (helps maintainers see affected user surface)
3. If the upstream eventually merges + propagates, update the row to ✅ Resolved and bump the relevant pin (Genesis commit, vLLM nightly, etc.) in the same commit / PR

## Conventions on this repo

### Bench protocol
3 warm + 5 measured runs. Canonical prompts: 800-word essay (narrative, max_tokens=1000) + quicksort code (max_tokens=800). `temperature=0.6, top_p=0.95, top_k=20`. Capture both wall-time TPS and engine-internal `gen throughput` from logs. **Always capture per-card peak VRAM** alongside TPS.

### Genesis opt-in env vars
Genesis ships ~50 env-gated patches. Some are **targeted bugfixes** (P64 streaming, PN8 memory savings, P3/P5/P6 KV); others are **behavioral mitigations** that silently rewrite the request (P68 = `tool_choice → required`, P69 = inject "must use tool" reminder). Behavioral mitigations need a streaming + large-prompt repro before shipping default-on. We learned this the hard way on 2026-04-29 — see [`docs/UPSTREAM.md`](docs/UPSTREAM.md) → Genesis #9 row + the [club-3090 #2 thread](https://github.com/noonghunna/club-3090/issues/2#issuecomment-4346740245).

If you're considering enabling a new Genesis env var by default in a shipped compose:
1. Read the patch's header docstring in `vllm/_genesis/wiring/`. Does it modify `request.tool_choice`, `request.messages`, or rewrite output?
2. If yes (behavioral): run a streaming repro with prompt > the patch's threshold + a casual user message ("hi") + `tool_choice: auto`. If `finish_reason=stop` with empty content, don't ship default-on.
3. Pure bugfixes (no behavioral override) are fine to ship default-on once they pass `verify-full.sh`.

### Engine image pinning
Pin engine images only when we vendor patches into the running container. Otherwise track upstream's rolling tag and accept that the compose YAML may need maintenance when upstream changes flags.

**Why:** patches hook into specific upstream code paths — a silent upstream change drifts those hooks and breaks the patched container in production. Pinning ensures the bytes we tested against are the bytes users get. Unpatched engines have no such hook, so upstream changes are upstream's problem to fix (or the YAML's, which is cheap to maintain).

| Engine | Patches we vendor | Tag policy |
|---|---|---|
| `llama.cpp` | none | rolling `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| `vLLM` | Genesis sidecars, Marlin pad, INT8 PTH, DFlash overlays | pinned to a specific nightly digest |
| `SGLang` | per-compose decision (pin if we vendor a patch, rolling otherwise) | per-compose |

When adding the first vendored patch to a previously-rolling engine: pin in the same commit. When dropping the last patch: unpin in the same commit. Bump pins via PR with a `verify-full.sh` + `bench.sh` re-run, never silently.

### CHANGELOG
- `CHANGELOG.md` (cross-cutting) and `models/<name>/CHANGELOG.md` (per-model) are **append-only history**. Don't rewrite past entries even when a finding is superseded — add a new entry. The historical trail is load-bearing for "why did we do X."
- Old entries can reference files / patches that no longer exist. That's fine — leave them.

### Compose variants
- Each variant ships with a **header table** comparing it against sibling composes in the same directory. Update both directions when adding/removing variants.
- Keep the variant set lean. Three composes that overlap badly (similar TPS + similar context + same KV) is worse than two with a clean differentiation. Removed `fast-chat.yml` 2026-04-29 for exactly this reason.

#### Compose layout — `<model>/<engine>/compose/<topology>/<feature>.yml`

The directory hierarchy encodes (model, engine, topology); the filename encodes the feature stack. No level repeats information from another.

| Path level | Encodes | Examples |
|---|---|---|
| `models/<model>/` | Model | `qwen3.6-27b` · `gemma-4-31b` |
| `models/<model>/<engine>/` | Inference engine | `vllm` · `llama-cpp` · `sglang` |
| `compose/<topology>/` | Hardware topology | `single` · `dual` · `multi3` · `multi4` · `multi8` |
| `<feature>.yml` (filename) | Feature stack | `docker-compose.yml` (recommended default) · `turbo.yml` · `dflash.yml` · `nvlink-dflash-noviz.yml` |

**Topology rule**: `single` (TP=1) and `dual` (TP=2) have no count ambiguity. `multi<N>` requires the count because N varies (3 / 4 / 5 / 6 / 8). Aligns with `docs/SINGLE_CARD.md` / `DUAL_CARD.md` / `MULTI_CARD.md` doc framing.

**Default-per-topology rule**: each topology subdirectory has a `docker-compose.yml` — the recommended starter. Bare `cd <topology> && docker compose up` works because docker compose finds `docker-compose.yml` automatically. Variants drop the `docker-compose.` prefix because they're explicitly invoked with `-f`.

**Feature suffix order** (when stacking): interconnect → drafter → KV → vision modifier. Examples:
- `dual/turbo.yml` — TP=2 + TurboQuant KV
- `dual/dflash.yml` — TP=2 + DFlash drafter
- `dual/nvlink-dflash-noviz.yml` — TP=2 + NVLink + DFlash + no vision
- `dual/int8.yml` — TP=2 + INT8 PTH KV
- `dual/awq.yml` — TP=2 + AWQ-4bit weights
- `multi4/dflash.yml` — TP=4 + DFlash

Concrete examples (post-2026-05-09 restructure):
- `models/qwen3.6-27b/vllm/compose/single/docker-compose.yml` — Qwen single-card default
- `models/qwen3.6-27b/vllm/compose/single/long-text.yml` — Qwen single-card text-only long-ctx variant
- `models/qwen3.6-27b/vllm/compose/dual/docker-compose.yml` — Qwen dual default (fp8 + MTP)
- `models/qwen3.6-27b/vllm/compose/dual/turbo.yml` — Qwen dual + TQ3
- `models/qwen3.6-27b/vllm/compose/multi4/dflash.yml` — Qwen 4-card + DFlash
- `models/gemma-4-31b/vllm/compose/dual/docker-compose.yml` — Gemma dual default (bf16 + MTP)
- `models/gemma-4-31b/vllm/compose/dual/int8.yml` — Gemma dual + INT8 PTH KV

Filename collisions across topology dirs (e.g. `dual/dflash.yml` vs `multi4/dflash.yml`) are fine — the path disambiguates. Registry tags in `scripts/switch.sh` decouple from filesystem paths; rename only the file path in the VARIANTS map and keep tags backward-compatible.

**Fine-tune exception**: model-specific fine-tunes that share the canonical model's compose directory keep the fine-tune name as a filename prefix (e.g. `dual/carnice-bf16mtp.yml`, `dual/qwopus-bf16mtp.yml` under `models/qwen3.6-27b/`). Long-term those fine-tunes can graduate to their own model directories (`models/carnice-v2-27b/`, `models/qwopus3.6-27b/`) when their variant set warrants it.

#### Patches and caches stay engine-level (NOT under a topology)

`<model>/<engine>/patches/` and `<model>/<engine>/cache/` sit parallel to `compose/`, not under any topology subdirectory. Three reasons:

1. **Patches are reused across topologies.** `vllm-marlin-pad/` is mounted by `dual/docker-compose.yml`, `multi4/docker-compose.yml`, and every `dual/nvlink-*.yml`. Putting it under one topology dir would force the others to symlink or duplicate.
2. **Patches are scoped by (model, engine), not by topology.** A vLLM source override doesn't change based on TP=1 vs TP=4; it's an in-container engine-internal patch that applies the same way regardless of mount layout.
3. **Caches (`torch_compile/`, `triton/`) warm-start across composes.** Sharing them at the engine level means switching from `single/docker-compose.yml` to `single/long-text.yml` reuses the JIT'd kernels instead of recompiling.

Relative paths from a compose to its sibling patches/caches: `../../patches/...` and `../../cache/...` (one `..` from `<topology>/` up to `compose/`, second `..` up to the engine dir, then `patches/` or `cache/`). All 27 shipped composes follow this.

**If a future patch is genuinely topology-specific** (e.g., a kernel rewrite that only applies to TP=2), keep it at `<engine>/patches/<patch-name>/` and document the topology constraint in the patch's README. Discoverability ("one `patches/` per engine, search there") trumps the marginal benefit of a topology partition.

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
#   Caveats:   <REQUIRED if Status is ⚠️ / 👁️ / ⏸️ / 🗑️; otherwise omit>
#   Quality:   <OPTIONAL — populated by `bash scripts/quality-test.sh --medium`>
#                e.g. "ToolCall-15 14/15 (93%) · InstructFollow-15 13/15 (87%)
#                      · StructOutput-15 15/15 (100%) · DataExtract-15 12/15 (80%)
#                      (--medium, packs v1.0.x, 2026-05-09)"
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
| `👁️ Preview` | Known quality issues; tracked but not for production. | E.g., quality regressions in soak / NIAH. Caveats line MUST list specific issues. |
| `⏸️ Upstream-gated` | Exists but blocked by external action (PR merge, driver fix, hardware ceiling). | Boots only with vendored override OR doesn't boot until external dep lands. Caveats line MUST point at the external dep. |
| `🗑️ Deprecated` | Kept for historical reference; will be removed. | N/A — flagged for cleanup. |

**Why this enum exists**: the previous "Status optional, only when not production" convention left readers guessing whether absence-of-status meant "validated production" or "author forgot to fill it in." Making Status required + enumerated removes that ambiguity. Users picking a config can scan to one field and know the lifecycle stage instantly; new contributors must consciously declare it when authoring.

The `Caveats:` line is REQUIRED whenever Status is ⚠️ / 👁️ / ⏸️ / 🗑️, OMITTED for ✅ / 🧪. Format: a single-line summary or a short bullet list, with links to issues / discussions / upstream PRs where relevant.

This rule applies to **shipped composes AND local-only test composes** — apply the convention even before deciding whether to ship; it avoids a rename later if the experiment graduates.

When testing a new model, create the directory hierarchy from the start: `models/<new-model>/<engine>/compose/<topology>/docker-compose.yml`. The hierarchy enforces the convention; filenames encode only the feature stack within that topology. When the model isn't Qwen3-Next, write `Genesis: N/A — Genesis is Qwen3-Next-specific` in the profile schema so readers don't expect Genesis-style perf folds where they don't apply.

#### Where do experimental / unvalidated composes live?

**Same directory as shipped composes, but kept untracked until validation passes.** Don't create a separate `experimental/` subdirectory — the relative paths to `../patches/...` and `../cache/...` are calibrated to the compose dir, and promoting an experiment from a sub-folder would require re-pathing every mount.

Workflow:

1. **Author the compose** in `models/<model>/<engine>/compose/<topology>/<feature>.yml` (or `docker-compose.yml` for the topology default) with the standard profile schema header. Mark `Status: ⚠️ EXPERIMENTAL` (or `⚠️ PREVIEW` if quality issues are known) so readers know it's not validated.
2. **Don't `git add`** until validation passes. The file shows up in `git status` as `??` — that's the signal. `git ls-tree -r HEAD` lists only shipped composes; the gap between that and `ls compose/*.yml` tells you what's pending validation.
3. **Validation gates** before promoting: `verify-full.sh` 8/8, `verify-stress.sh` 7/7 (or documented failures with rationale), `bench.sh` (numbers added to BENCHMARKS.md), `soak-test.sh SOAK_MODE=continuous` (catches Cliff 2b), `quality-test.sh --quick` (no major regression on ToolCall / InstructFollow). For pin bumps and new quants, run `quality-test.sh --medium` and add the result line to the compose's `Quality:` schema field.
4. **Promote**: drop the `Status: ⚠️ EXPERIMENTAL` line from the profile schema, `git add`, commit. Cross-rig validation can come later via the `numbers-from-your-rig` issue template.

For **entirely new models** under validation (e.g. "let's try MiniMax-M2.7"): keep the whole `models/<new-model>/` directory untracked until at least one compose validates. Avoid pushing `models/<new-model>/README.md` etc. before there's a working compose to back it up — empty model directories on master signal capability we don't actually have.

References (orphan composes / patches sitting in this state as of 2026-05-09):
- `models/gemma-4-31b/vllm/compose/dual-awq.yml` — local-only AWQ-4bit weights variant, Status: ⚠️ EXPERIMENTAL
- `models/gemma-4-31b/vllm/compose/dual-dflash-int8.yml` — Status: ⚠️ EXPERIMENTAL (boots only with vLLM PR #42102 vendored mounts)
- `models/qwen3.6-27b/vllm/compose/qwopus-bf16mtp.yml` — Status: ⚠️ PREVIEW (41× line repetition + NIAH drop, not production)

### Documentation
- Don't create new docs proactively. Most non-obvious things belong in `INTERNALS.md`, `FAQ.md`, `SINGLE_CARD.md`, or `DUAL_CARD.md`. New top-level files only when there's a recurring search miss.
- Charts: source `.svg` + exported `.png` at retina resolution (≥1500px wide). Markdown embeds use `.png` (clicking opens a viewable image; SVG opens raw XML). Re-generate with `python3 tools/charts/gen-perf.py` and `gen-vram.py` after editing data.
- For any change that adds a footnote to "this depends on upstream X" — the answer is to link the row in `docs/UPSTREAM.md`, not to inline-cite the upstream URL.

### Tests
- `verify.sh` — fast smoke (~15s). Confirms the stack is responding; runs after `setup.sh`.
- `verify-full.sh` — functional (~1-2 min). Runs on every compose change.
- `verify-stress.sh` — boundary cases (longctx ladder + tool-prefill OOM ~5-10 min). Runs on cliff-related changes.
- `bench.sh` — canonical TPS bench (~3-5 min). Run when you change anything that could move TPS (compose flags, Genesis env vars, vLLM pin).
- `quality-test.sh` — behavioral quality (~10-30 min depending on `--quick` / `--medium` / `--full`). Wraps [`benchlocal-cli`](https://github.com/noonghunna/benchlocal-cli) — runs verifier-backed bench packs (ToolCall-15, InstructFollow-15, StructOutput-15, etc) against the running endpoint. Catches what operational tests miss: a compose can pass verify + stress + bench + soak and still ship with degraded tool-call accuracy or instruction-follow drift from quantization or Genesis env-flips. Run before promoting `Status: ✅ Production` and before any pin bump that could shift behavior. See [`docs/QUALITY_TEST.md`](docs/QUALITY_TEST.md).
- `soak-test.sh` — stability (30-60 min). Run before shipping config / Genesis / memory-policy changes — catches Cliff 2b.

The pipeline is layered: each script has a different question it answers ("does it serve / work / survive / fast / behave correctly / stay healthy"). Skipping any layer can mask regressions.

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
