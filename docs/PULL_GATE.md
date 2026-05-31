# Pull-Gate — `scripts/pull.sh` (v0.8.0, #147)

Contributor/maintainer guide for the v0.8.0 Pull-Gate. `pull` takes one HF
repo slug, derives a ModelProfile-shaped spec from the repo's own files,
runs it through the **locked 6-stratum abort taxonomy**, and — only for a
curated, `[D]`-emittable model — hands the validated profile to the #141
compose generator. It is honest about confidence and **never silently
gate-passes**.

The honest scope of this phase is exactly:

> evaluate any safetensors HF repo; pull-and-emit only curated
> (Tier-1) models — and only when the gates pass (or an explicit
> override is accepted)

> ### The authoritative spec is the test, not this doc
> Per the locked v0.8.0 stop-condition, **`scripts/tests/test-pull.sh` is
> the canonical state-machine specification** — an exhaustive,
> network-mocked truth table over the §4.1 nine cells, all six strata,
> stratum ordering, every flag interaction, and golden cases g0–g15. This
> document is an explanatory companion. Where this prose and the test ever
> disagree, **the test is correct**; fix the doc.

For the `[D]` substrate this hands off to, see
[`docs/COMPOSE_GENERATOR.md`](COMPOSE_GENERATOR.md). For the underlying
patch/arch/profile data model see
[`docs/PATCH_ATTRIBUTION.md`](PATCH_ATTRIBUTION.md); for the locked design,
the v0.8.x consolidated design (§3/§4.1/§5.1/§5.2/§7).

---

## 1. Purpose & scope — the two paths

`pull` runs in exactly one of two paths. Path selection in
`run_pull` (`scripts/lib/profiles/pull.py`): an explicit driver `path=`
wins; else `--dry-run` forces Path B; else Path A iff the deriver returns a
Tier-1 curated hit **and** an `--out` target is given; otherwise Path B.

| Path | Trigger | What it does | Calls `[D]`? | Downloads? |
|---|---|---|---|---|
| **A — curated pull-and-emit** | curated (Tier-1) slug + `--out`, not `--dry-run` | full gate → stratum-6 `[D]` dry-run → real `[D]` emit of the validated registry key | yes (read-only dry-run, then real `gc.generate`) | no (download itself deferred — see below) |
| **B — universal evaluate** | any slug, or `--dry-run` | full gate → print the §7-caveated verdict → **STOP** | never | never |

Path A is the only path that ever invokes the #141 generator, and it does
so only after a download-eligible `[C1]` terminal **and** a clean stratum-6
`[D]` dry-run. Path B evaluates *any* safetensors repo (curated or not) and
stops at a verdict — it is structurally incapable of emit/download
(`test-pull.sh` sweeps `vllm/minimal`/`vllm/dual`/`vllm/tools-text` to
assert `emitted` and `compose_text` stay unset on Path B).

### Explicitly OUT of v0.8.0 Pull-Gate (named deferred phases)

| Deferred capability | Deferred to |
|---|---|
| Pull-Emit-Derived: dynamic-`[D]`-from-derived, the `[E]` weight download, static-route fallback for non-emittable profiles | **Pull-Emit-Derived** phase |
| `[F]` capture / classify / trust / loop; `override-accepted` telemetry capture; `--experimental-arch` outcome capture | **Loop** phase |
| The two UX doc tracks + the end-user `recommend` nudge | **post-Pull-Gate** (after this phase) |
| GGUF / `.bin` backend, multi-quant auto-pick, whichllm hardware slice | **v0.8.1** |
| `derived` confidence tier (override-registry promotion/consensus) | **Loop** phase (the `derived` row is RESERVED, still total in §4.1) |

`override-accepted` is a first-class terminal but **not** a gate-pass: this
phase records the state + a telemetry notice and downloads nothing
(`run_pull` returns `ok=True` with a "download deferred to the Loop phase"
notice). `--force-download` is therefore a **NO-OP + notice** this phase.

---

## 2. CLI reference

Observed from `python3 scripts/lib/profiles/pull.py --help` (`pull.sh` is a
thin argv pass-through to it):

```
pull.sh [-h] --profile-like PROFILE_LIKE [--dry-run] [--yes]
        [--force-download] [--experimental-arch] [--trust-remote-code]
        [--hf-home HF_HOME] [--out OUT] [--hardware HARDWARE]
        slug
```

| Flag / arg | Required | Semantics (as shipped) |
|---|---|---|
| `slug` (positional) | yes | HF repo slug, e.g. `org/Model-Name`. |
| `--profile-like PROFILE_LIKE` | **yes** | A curated `COMPOSE_REGISTRY` key supplying the runtime shape. Path A: must name the curated model+variant **and** be `[D]`-emittable. Path B: any vLLM profile, runtime *shape only*. |
| `--dry-run` | no | Force Path B (evaluate only; never emit/download). |
| `--yes` | no | Accept a `confirm→proceed` terminal (§4.1). Without it, a `confirm→proceed` cell is an honest non-pass. |
| `--force-download` | no | Advisory low-confidence `wont-fit` → `override-accepted`. **NO-OP + notice this phase**; the download is deferred to the Loop phase. |
| `--experimental-arch` | no | Bypass **ONLY** `[C0] engine-support-unknown/no-arch-row`. Never bypasses `runtime-incompatible`; Path B only this phase. |
| `--trust-remote-code` | no | Bypass `[C0] needs-trust-remote-code-ack`. |
| `--hf-home HF_HOME` | no | Override the `HF_HOME` resolution chain (`--hf-home > $HF_HOME > $XDG_CACHE_HOME/huggingface > ~/.cache/huggingface`). |
| `--out OUT` | no | Path A only: write the emitted compose here. Its presence (with a curated slug, no `--dry-run`) is what selects Path A. |
| `--hardware HARDWARE` | no | Override detected GPU compute capability (e.g. `8.6` for RTX 3090). Default = `nvidia-smi --query-gpu=compute_cap` detection (the **min** SM across visible GPUs). |

### Exit codes (CLI `main`)

| Code | Meaning |
|---|---|
| `0` | `ok` — download-eligible (Path A emitted, or Path-B clean verdict, or `override-accepted`). |
| `2` | Any honest hard-stop: a stratum-1→6 abort, or a `[C1]` `hard-block`. |
| `3` | A `confirm→proceed` / `override-accepted` terminal not yet satisfied (re-run adding the named flag). |
| `64` | argparse usage error. |

If no GPU is detected and no `--hardware` is given, `run_pull` **fails
closed** at stratum-3 with `hardware-sm-undetermined` rather than fabricate
a fit (the §1 honesty rule).

---

## 3. The 6-stratum abort taxonomy

Evaluated **strictly in order** and **monotonically**: each stratum depends
only on prior strata; nothing downstream rewrites an upstream verdict. The
earliest applicable stratum wins (`test-pull.sh` Section 8 stacks failing
conditions and asserts earliest-wins). `Stratum.DECIDED` (value `0`) means
the run reached a `[C1]` terminal / Path-B verdict with no abort.

| # | Stratum | Code | States / sub-reasons | Bypass flag (if any) |
|---|---|---|---|---|
| 1 | Deriver structured errors | `deriver.derive` | `repo-not-found`, `gated-no-token`, `unsupported-format`, `ambiguous-weight-set`, `quant-dtype-unknown` | none — pick a different repo |
| 2 | `--profile-like` validity precondition | `gates.stratum2_profile_like` | `unknown-profile-like`, `unsupported-runtime-engine`, `profile-not-emittable` (Path A), `profile-mismatch` (Path A) | none — pick another profile |
| 3 | `[C0]` engine-support / runtime / SM | `gates.c0_engine_support` | `engine-supported` · `engine-support-unknown` (+ sub-reason `no-arch-row` \| `runtime-incompatible`) · `needs-trust-remote-code-ack` | see below |
| 4 | `[C2a]` disk pre-gate | `gates.c2a_disk` | `disk-ok` \| `disk-short` | **none — non-negotiable** |
| 5 | Pre-`[B]` generic-dense eligibility | `pull.run_pull` (P4) | `no-fit-model` | **none — non-bypassable** |
| 6 | Path-A `[D]` dry-run | `pull.run_pull` (P4) | `d-refused:<pin-mismatch \| tp-or-kv \| trc \| foundational-drift \| degraded-drift \| scope \| other>` | none |

Then `[B]` (`kv-calc raw_verdict`) produces the raw fit verdict and `[C1]`
(§4.1) maps it to a terminal. On Path A only, a download-eligible terminal
is followed by stratum-6.

### Stratum-3 `[C0]` bypass scoping (exact)

`[C0]` tags which flags (if any) can bypass each non-`engine-supported`
verdict on `.bypassable_by`; `run_pull` only bypasses when **every** tagged
condition is covered by a provided flag (`tags.issubset(provided)`).

| `[C0]` outcome | `.bypassable_by` | Cleared by |
|---|---|---|
| `engine-support-unknown/no-arch-row` | `(--experimental-arch,)` | `--experimental-arch` (Path B only this phase) |
| `engine-support-unknown/runtime-incompatible` | `()` — empty | **nothing** (non-bypassable; includes hardware-SM mismatch) |
| `needs-trust-remote-code-ack` (known arch + `auto_map`) | `(--trust-remote-code,)` | `--trust-remote-code` only |
| `needs-trust-remote-code-ack` + no arch row | `(--trust-remote-code, --experimental-arch)` | **both** flags (subset rule) |

`--experimental-arch` therefore bypasses **only** `no-arch-row`. It does
**not** bypass `runtime-incompatible` (`bypassable_by=()`, asserted by g10
and g14), and it does **not** bypass stratum-5 `no-fit-model` (g11 runs
*with* the flag and still aborts). `disk-short` (stratum-4) and
`no-fit-model` (stratum-5) are non-negotiable: there is no flag, and for
stratum-5 "there is no model to force."

---

## 4. `[C0]` and `[C1]` are design-locked state sets

`[C0]` emits **exactly three** top-level states —
`engine-supported`, `engine-support-unknown`,
`needs-trust-remote-code-ack` (locked design §5.1). `no-arch-row` and
`runtime-incompatible` are **sub-reasons** carried on `C0Result.sub_reason`
(a structured *side field*), **not** new top-level states. Likewise `[C1]`
has **exactly four** terminals — `proceed`, `confirm→proceed`,
`hard-block`, `override-accepted` — frozen in `pull.LOCKED_TERMINALS`;
`test-pull.sh` asserts `LOCKED_TERMINALS == {proceed, confirm→proceed,
hard-block, override-accepted}` and that the enum adds nothing.

`[C0]`'s determination order (`gates.c0_engine_support`) is monotonic:
trust-remote-code (fail-closed: `requires_trust_remote_code ∈ {true,
unverified}` **or** `config.json` `auto_map`) → no arch row → arch row
exists but runtime/hardware not loadable → else `engine-supported`.

---

## 5. `[C1]` — the §4.1 confidence × raw-verdict total function

`[B]` (`kv-calc.raw_verdict`) is pure measurement: `FAIL → wont-fit`,
`TIGHT → fits-constrained`, `PASS → fits-clean`. `[C1]`
(`pull.c1_terminal`, backed by the data table `_C1_TABLE`) is the **single**
authoritative mapping, reproduced verbatim from the locked design §4.1:

| Confidence | `fits-clean` | `fits-constrained` | `wont-fit` |
|---|---|---|---|
| `exact` | **proceed** (silent) | **confirm→proceed** (constraint changed the requested config — user must accept the applied ctx/KV constraint even though math is trusted) | **hard-block** (math trusted; suggest closest-fit) |
| `derived` | **confirm→proceed** (`--yes`; "best-effort, validate post-boot") | **confirm→proceed** (`--yes` + constraint notice) | advisory → `--force-download` → **override-accepted** |
| `estimated-lower-bound` | **confirm→proceed** (`--yes` + "VRAM is a floor; likely under-modeled") | **confirm→proceed** (`--yes` + floor + constraint notice) | advisory → `--force-download` → **override-accepted** |

The table is **total** over `{exact, derived, estimated-lower-bound} ×
{fits-clean, fits-constrained, wont-fit}` (nine cells; `test-pull.sh`
asserts totality and exactly nine entries). Per the locked design §4.1
footnote: **only `exact × fits-clean` is silent**; every other cell needs
`--yes` (a `confirm→proceed` cell) or `--force-download` (a low-confidence
`wont-fit` advisory) to be satisfied. `exact × wont-fit` is an
unconditional `hard-block` — **no flag clears it** (g3 asserts this even
with `--yes --force-download`).

Critically, a flag-bypassed low-confidence model is **not** a silent pass.
After `--trust-remote-code` clears stratum-3 for an
`estimated-lower-bound` model (g4), `[C1]` still lands on
`confirm→proceed` which itself requires `--yes`. Without `--yes` it is an
honest non-pass (`abort_reason` begins `confirm→proceed`, exit 3); the
caveat verdict is only printed once the cell is satisfied. `derived` is
RESERVED for the future override-registry (Loop) phase and is never
assigned in v0.8.0, but its §4.1 row is present so the function stays
total.

Every download-eligible terminal and every Path-B verdict carries the
locked design §7 caveat verbatim (`pull.CAVEAT_S7`):

> boot-fit satisfied; this does NOT guarantee stability under sustained /
> accumulated-context workloads — validate with soak-continuous before
> relying on it (recommend: scripts/soak.sh SOAK_MODE=continuous).

This is presentation-only and changes no decision.

---

## 6. Hardware-SM gating (why a 3090 cannot get a confidently-wrong "fits")

Stratum-3 enforces, as part of runtime loadability:

```
need_sm = max(engine.min_sm, registry.required_sm, arch-kernel SM rule)
```

(`gates._required_sm`). The arch-kernel rule (`gates._ARCH_KERNEL_SM`)
pins `fp8_e4m3` and `turboquant_3bit_nc` (Gemma-TQ3) at SM 9.0. If the
detected/`--hardware` SM is below `need_sm`, `[C0]` returns
`engine-support-unknown/runtime-incompatible` — **non-bypassable**.

Concretely (g14): a curated Gemma-4-31B `fp8_e4m3` / `required_sm:9.0`
profile on a detected RTX 3090 (`sm_86`) aborts at stratum-3
`runtime-incompatible`; `--experimental-arch` does **not** bypass it; the
same profile on `sm_90` clears the SM gate (proving it *is* the SM gate).
Without this check a 3090 would receive a false "fits" verdict — exactly
the §1 confidently-wrong outcome the design forbids.

---

## 7. How each path consumes `--profile-like`

`--profile-like` is REQUIRED on both paths but consumed differently:

| Aspect | Path A (curated pull-and-emit) | Path B (universal evaluate) |
|---|---|---|
| What the profile supplies | the **full** registry key (the validated key is handed to `[D]`) | runtime **shape only**: `engine`, `kv_format`, `tp`, `max_ctx`, `max_num_seqs`, `mem_util`, topology |
| Model/variant match | profile `model` must equal the curated resolved model **and** `weights_variant` must equal the matched variant, else stratum-2 `profile-mismatch` (g3b) | not checked — the pulled repo's own files determine weight format/quant |
| `[D]`-emittable required | yes — reuses `[D]`'s scope-gate (`engine.type==vllm` ∧ `profile_runtime` entry exists ∧ `genesis_equipped==false`); a Genesis/TQ3 profile (`vllm/dual-turbo`) → stratum-2 `profile-not-emittable` (g0) | no |
| `weights_variant` compat | Path-A only (`[C0]` checks the curated variant against arch constraints) | n/a — uses deriver-resolved `weight_format`/quant |
| `drafter` | from the curated profile | `none` for non-curated (drafter profiles expose `model_compat`, not arch-compat) |
| Non-vLLM `--profile-like` | refused stratum-2 `unsupported-runtime-engine` (g13: `llamacpp/default`, `engine=llama-cpp-local`, `mem_util=None`) | same — refused on both paths |

---

## 8. `[D]` handoff & the `--project-directory` requirement

On Path A, after a satisfied download-eligible terminal, `run_pull` runs
the stratum-6 `[D]` dry-run via the real `generate_compose.generate`
(pure — no container). If `[D]` refuses at one of its later points (pin
mismatch / TP·KV / trc / foundational-or-degraded patch drift), that is a
**stratum-6 Path-A abort** (`d-refused:<token>`) and the run is **not**
reported download-eligible — stratum-2's scope-gate is necessary but not
sufficient for `[D]` emit (g15: an injected refusing runner →
`d-refused:foundational-drift`, `ok=False`, no file written).

On a clean dry-run, the validated key's compose text is written to
`--out`. The emitted compose's relative overlay mounts resolve from the
**compose file's own directory**, so generated composes are **not
relocatable** — `run_pull` surfaces a notice that the consumer must run:

```
docker compose --project-directory <repo-root> -f <out> up
```

See [`docs/COMPOSE_GENERATOR.md`](COMPOSE_GENERATOR.md) for the full
`--project-directory` rationale and the governed-slot invariant (the
in-scope Path-A emitted body contains zero `--trust-remote-code`, asserted
by `test-pull.sh` via `patch_attribution.service_body()==0`). For a
`fits-constrained` Path-A run, `[D]` emits the chosen registry profile
**unchanged**; the run prints a "known effective-cap warning" (vLLM
internally caps effective KV on this hardware) — **no compose config is
rewritten** (g2 asserts the notice says "no compose config rewritten" and
not "applied constraint").

---

## 9. Golden cases (the test's truth table)

`scripts/tests/test-pull.sh` is the canonical spec. Golden cases as
asserted there:

| Case | Path | Asserted behavior |
|---|---|---|
| g0 | A | `vllm/dual-turbo` (genesis_equipped) → stratum-2 `profile-not-emittable` before `[C0]` |
| g1 | A | curated + emittable, gate-passing, clean `[D]` dry-run → compose emitted (integration analogue of the satisfied g2) |
| g2 | A | curated effective-capped → `exact × fits-constrained` → `confirm→proceed`; `--yes` → "known effective-cap warning" (no rewrite) → `[D]` emit; without `--yes` → honest non-pass, no file |
| g3 | A | curated too big → `exact × wont-fit` → `hard-block`; no `[D]` even with `--yes --force-download` |
| g3b | A | model/variant mismatch (`vllm/gemma-bf16-mtp` vs curated Qwen) → stratum-2 `profile-mismatch` |
| g4 | B | Llama, `trc:unverified` → `needs-trust-remote-code-ack`; `--trust-remote-code` → `[C1]` `confirm→proceed` (still needs `--yes`); `+ --yes` → Path-B caveat verdict, no `[D]` |
| g5 | B | `no-arch-row` → stratum-3; `--experimental-arch` (+ `--yes`, eligible) → Path-B verdict + bypass notice |
| g6a | B | known arch + `auto_map` → `needs-trc-ack`, `bypassable_by == ['--trust-remote-code']` only; `--experimental-arch` alone does not clear it |
| g6b | B | no-arch-row + `auto_map` → `bypassable_by == {--trust-remote-code, --experimental-arch}`; needs **both** (subset rule) |
| g7 | B | 200 GiB model, 2 GiB free, clean `[C0]` → stratum-4 `disk-short`; `[B]` never ran (no fit verdict) |
| g8 | B | curated + `--dry-run` (with `--out`) → Path B, no `[D]` emit, no file written |
| g9 | B | derived dense → `estimated-lower-bound` confidence, real verdict; `--calibration` stays 22/22 |
| g10 | B | `loads:false` pin → `runtime-incompatible`; `--experimental-arch` does **not** bypass |
| g11 | B | SWA-only / MoE-on-known-arch → `[C0]` stays `engine-supported` → stratum-5 `no-fit-model`; `--experimental-arch` does **not** bypass |
| g12 | — | no `*.safetensors` → `unsupported-format`; multiple sets → `ambiguous-weight-set`; HF 404 → `repo-not-found`; stratum-1 emits **no** fit verdict |
| g13 | both | `--profile-like llamacpp/default` → stratum-2 `unsupported-runtime-engine` before `[C0]`/`[B]` |
| g14 | A | Gemma `fp8_e4m3` / `required_sm:9.0` on `sm_86` → stratum-3 `runtime-incompatible`; non-bypassable; clears on `sm_90` |
| g15 | A | stratum-2 + `[C1]` download-eligible but `[D]` dry-run refuses → stratum-6 Path-A abort, **not** download-eligible |

Run the canonical spec from the repo root:

```
bash scripts/tests/test-pull.sh
```
