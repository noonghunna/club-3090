# Patch Policy — the contributor contract for load-bearing patches (v0.8.0, #141)

This is the **durable contract** for anyone (human or agent) adding,
modifying, or removing a patch that the v0.8.0 compose generator
([`scripts/generate-compose.sh`](../scripts/generate-compose.sh)) consumes.
It is prescriptive and self-contained: follow it exactly, and
`scripts/tests/test-patch-attribution.sh` + `scripts/tests/test-generate-compose.sh`
will gate your PR cleanly.

It exists for one reason, stated up front so it frames every rule below:

> ### The #145 lesson
> A patch can be present in the repo, declared "load-bearing", and *still
> never reach the running engine* — silently. That is the
> `#72` / `#145`-class failure: a load-bearing fix that the compose does not
> actually deliver, with no error, just a quietly-wrong server. **Every
> mechanism in this document exists to make a silently-undelivered
> load-bearing patch impossible to introduce without it failing loud.**

For the data model these rules sit on, see
[`docs/PATCH_ATTRIBUTION.md`](PATCH_ATTRIBUTION.md). For how the generator
*uses* this metadata, see [`docs/COMPOSE_GENERATOR.md`](COMPOSE_GENERATOR.md).

---

## 1. Where patches and metadata live

| File | Role |
|---|---|
| `models/<model>/<engine>/patches/<patch-dir>/` | the patch artifact itself (overlay file(s), sidecar `.py`, or `install.sh`) |
| `scripts/lib/profiles/patches.yml` | the patch entry — what it fixes, how it is delivered, where it is load-bearing, its drift-guard |
| `scripts/lib/profiles/arch_patches.yml` | declarative arch → engine-pin / valid-TP / trc gate (C0 authority; **strict closed key-set** — do not add keys) |
| `scripts/lib/profiles/profile_runtime.yml` | per-profile captured template + `genesis_equipped`; `arch_model_xref` trc evidence ledger |

**Add the `patches.yml` entry in the same change as the patch artifact.**
Never land a patch file without its entry, and never land an entry that
points at a missing artifact.

---

## 2. Required per-patch metadata

A load-bearing patch entry in `patches.yml` MUST carry all of:

| Field | Requirement |
|---|---|
| `id` | unique, stable, descriptive |
| `model` | list of model slugs the patch applies to |
| `files` | repo-relative path(s) to the patch artifact |
| `load_bearing_when[].composes` | the **exact** list of profile keys this patch is load-bearing for. This is the *only* thing the generator keys patch selection off (`X ∈ P.load_bearing_when[].composes`). Each entry also carries `reason` + `evidence`. |
| `delivery_mechanism` | one of `python_sidecar`, `site_package_overlay`, `install_script`, or `none` (diagnostics / negative-result / Genesis-env patches) |
| `delivery_spec` | the concrete wiring: mount target(s) (`overlay_files[].dest`, `overlay_dir`+`dest_root`, `mounted_at`), sidecar/script path, `invoke` command, and `wired_at` (`volumes`, `entrypoint`, or both). This is the **single source** both the generator's emit and `reaches()`'s reachability validation read — they must agree by construction. |
| `drift_guard` | **mandatory** for any load-bearing patch (`kind`, `check`, `on_fail`). See §4. |
| `capability` | the capability the patch delivers (e.g. `tp-weight-load`, `tool-choice-required`, `tool-call-stream`, `dflash-spec-decode`) |
| `foundational` | `true` if weights/boot cannot proceed without it; `false` if absence merely loses a capability. Drives the §4.1 grade. |
| `upstream` | `ref` + `status` + `drop_when` — the condition under which the patch should be removed |
| `status` | `verified`, `unverified`, or `suspect`. Use `unverified` honestly — never claim `verified` without evidence. The `load_bearing_when[].evidence` string must point at a real doc/README anchor. |
| `delivery_gaps[]` | declare here, do **not** silently fix, any profile where the patch is load-bearing but cannot reach the compose today (the #145-class boundary). Each gap carries `composes` + an `issue` string. |

The legacy `delivery:` boolean block
(`dockerfile_bake`/`entrypoint_invoke`/`genesis`) is **DEPRECATED and
READ-ONLY** — retained verbatim only because `test-patch-attribution.sh`
still reads it. Do **not** put new wiring decisions there; use
`delivery_mechanism` / `delivery_spec`.

Diagnostics / no-op / negative-local-result / Genesis-env patches set
`delivery_mechanism: none` and have no `delivery_spec`.

---

## 3. Hard rule: guarded `python_sidecar` is the default delivery mechanism

Choose the delivery mechanism in this order of preference:

1. **`python_sidecar` (DEFAULT).** A guarded sidecar `.py` invoked from the
   entrypoint before `vllm` imports/serves. This is the locked default for
   any new patch unless there is a concrete reason it cannot work.
2. **`install_script`.** A mounted `install.sh` invoked from the entrypoint
   (`invoked_before: vllm-import`). Use when the change is a multi-step
   install rather than a single Python shim.
3. **`site_package_overlay` (LAST RESORT).** A bind-mount over a
   `dist-packages/...` file. Permitted **only** for an isolated, single-file
   emergency. It carries the **#145-class risk** (a stale/conflicting overlay
   silently shadows upstream — overlays of `speculative.py` /
   `gpu_model_runner.py` etc. conflict with each other; only one variant can
   run at a time) and **must be re-homed** to a sidecar or upstream fix as
   soon as the emergency is over. A multi-file overlay must record its
   `file_count` and a `note` flagging the #145-risk and any conflicting
   overlay it is mutually exclusive with.

If you reach for `site_package_overlay`, justify it in the PR and file the
re-home follow-up.

---

## 4. Hard rule: no `drift_guard` ⇒ not load-bearing in a generated compose

A patch with **no `drift_guard`** is, by definition, **not load-bearing in a
generated compose**. The generator only wires a patch whose state is
"applies cleanly" — and "applies cleanly" *means* drift-guard-tested (a
locked decision). If you believe a patch is load-bearing, you MUST give it a
`drift_guard`; if you cannot articulate a guard, the patch is not eligible to
be wired by the generator.

`drift_guard` fields:

- `kind` — `import-and-boot` or `behavioral`.
- `check` — a concrete, testable statement of what "still applies" means on
  the selected nightly.
- `on_fail` — the §4.1 grade (see below).

### 4.1 Drift fails loud — never silently repaired

The drift-guard is a runtime probe. The generator **cannot** run it at
generation time (no engine container), so it wires the maintainer-tested
patch and surfaces the guard for the boot leg / operator to re-run. When a
guard **fails**, the outcome is graded — and the generator **never repairs**:

| `on_fail` / `foundational` | Outcome | Rationale |
|---|---|---|
| `capability-degraded` (foundational: false) | patch **OMITTED**, compose flagged **DEGRADED**; run needs `--accept-degraded` (else exit 3). Patch is **never wired**. | The model still serves, minus one capability. Honest degradation beats a silently-broken capability. |
| `hard-refuse` / `foundational: true` | **hard-refuse** (exit 2), even *with* `--accept-degraded`. | Weights/boot depend on it; there is nothing safe to emit. |

**Gap-before-guard ordering:** a declared `delivery_gaps[]` entry covering
the profile is evaluated *before* the drift-guard — the patch is
selected-but-undelivered (wiring omitted, header WARNING) and the guard is
skipped. This is the structural acknowledgement of a known coverage
boundary; it is loud (header category [2]) by construction, never silent.

The contract: **drift never gets repaired, only flagged.** A failed
capability-scoped guard degrades; a failed foundational guard refuses;
neither edits the patch.

---

## 5. Hard rule: Genesis patches are arch-data only, never generator-emitted

The compose generator is **non-Genesis, vLLM-only** (a permanently locked
v0.8.x scope decision). Genesis patches:

- live in `patches.yml`/`arch_patches.yml` as **data only** (so the
  attribution audit stays complete);
- set `delivery_mechanism: none` (they are env-gated, not generator-wired);
- are **never emitted** by the generator — any profile that is
  `genesis_equipped: true` (compose has a `GENESIS_*` / `_genesis` token, or
  `kv_format` starts `turboquant`) is a clean refusal.

Do not add a Genesis patch expecting the generator to wire it. It will not,
and that is intentional.

---

## 6. Contributor workflow

1. **Land the artifact + entry together.** Add the patch under
   `models/<model>/<engine>/patches/<patch-dir>/` and its `patches.yml`
   entry (all §2 fields) in the same commit.
2. **Pick the delivery mechanism per §3** (default `python_sidecar`).
   Write the `delivery_spec` so `wired_at` and the mount/invoke markers
   describe the *actual* wiring in the compose(s) you list under
   `load_bearing_when[].composes`.
3. **Write a real `drift_guard` (§4).** No guard ⇒ the generator will not
   wire it (§4 hard rule) — that is a design signal, not a workaround.
4. **Wire it into the shipped compose(s)** you declared, OR — if you cannot
   reach a listed compose today — declare a `delivery_gaps[]` entry with an
   honest `issue` string. **Do not silently fix runtime files instead of
   declaring the gap** (that *is* the #145 failure mode).
5. **If the patch gates an arch/engine path**, extend the matching
   `arch_patches.yml` row (`required_patches`, `engine_pin`, `valid_tp`,
   `requires_trust_remote_code`). Do **not** add new top-level keys —
   `arch_patches.yml` is a strict closed key-set; arch-scoped fold-ins go in
   `profile_runtime.yml`'s `arch_model_xref`.
6. **Run the gates** (next section). Both must be green; the
   patch-attribution summary must stay exactly
   `61 patch / 11 arch / 18 calibration` *plus your delta* (the audit
   re-counts — keep the known-gaps list intentional).

---

## 7. How the tests gate a patch PR

Two tests are the contract. They are the contract; patches are fixed to them,
never the reverse.

### `scripts/tests/test-patch-attribution.sh`

Imports `scripts/lib/profiles/patch_attribution.py` and audits every
`patches.yml` / `arch_patches.yml` / `calibration_seed.yml` entry. It checks:
schema (all §2 required keys present, valid `status`/`confidence`/trc
tri-state), the closed arch key-set, artifact coverage, and **reachability**:

> `reaches(root, patch, name_or_path)` parses the **comment-stripped service
> body only** (everything from the top-level `services:` key onward, header
> banner and `#` comments removed) and validates the patch's **actual
> `delivery_spec` wiring** — the declared mount target(s) and/or entrypoint
> invoke at the `wired_at` insertion point(s) — *not* a substring of the
> patch ID.

Consequences for a contributor:

- A patch ID merely *named in a comment/header* does **not** count as
  reached — you must wire the real `delivery_spec` artifact.
- A patch declared load-bearing for a compose it does not actually reach
  fails the audit **unless** you declared the `delivery_gaps[]` entry. The
  audit prints the gap list explicitly — a silently-undelivered load-bearing
  patch cannot pass.

### `scripts/tests/test-generate-compose.sh`

Generates the five golden triples and asserts the golden-parity invariant
(semantic diff confined to the two insertion points; constants + image
expression verbatim), that wired patches `reaches()` the *generated* compose,
that selected-but-undelivered patches are **not** reachable, that no
`--trust-remote-code` is emitted, and the full refusal/degraded matrix.

If your patch is load-bearing for a golden triple, it must wire cleanly into
the generated compose (or be a correctly-declared gap). If it changes the
captured service body of a golden profile, update the shipped compose so the
parity invariant holds — the generator captures, it never synthesizes.

**Both tests must be RC=0 before a patch PR merges.** A silently-undelivered
load-bearing patch is precisely the failure these gates exist to prevent — if
either test goes red, the contract has caught exactly what it is for.
