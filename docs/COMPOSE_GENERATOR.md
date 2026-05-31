# Compose Generator — `scripts/generate-compose.sh` (v0.8.0, #141)

Operator/user guide for the v0.8.0 compose generator. It emits a
**minimal-reproduction** `docker-compose` for one in-scope profile by
replaying the *whole shipped service definition* that profile was captured
from, wiring (or deliberately omitting) the patches that profile depends on,
and refusing — loudly and cleanly — anything outside its scope.

This is the `[D]` substrate of the v0.8.x design. Its job is the narrow one
described by the v0.8.x headline scope:

> evaluate any safetensors HF repo; pull only vLLM-loadable supported ones,
> and only when the gates pass (or an explicit override is accepted)

The generator itself implements the *generation* slice of that: it does not
download, derive, or VRAM-gate (those are other v0.8.x phases). It takes an
already-resolved in-scope profile and produces the compose that reproduces it.

For the **contributor contract** governing the patch metadata this generator
consumes, see [`docs/PATCH_POLICY.md`](PATCH_POLICY.md). For the underlying
patch/arch/profile data model see
[`docs/PATCH_ATTRIBUTION.md`](PATCH_ATTRIBUTION.md).

---

## 1. Mission — reproduce + flag, NEVER repair

The single locked rule that explains every other behaviour:

> **The generator reproduces a known config and flags what is missing. It
> never repairs anything.**

Concretely, the shipped code guarantees:

- It **never rewrites the engine image** — the captured
  `image: ${VLLM_IMAGE:-vllm/vllm-openai:nightly-${VLLM_NIGHTLY_SHA}}`
  expression is passed through verbatim (see §5).
- It **never wires a patch whose drift-guard failed** — a failed
  capability-scoped guard *omits* the patch (DEGRADED), a failed foundational
  guard *refuses* the whole compose. It never edits the patch to make it
  apply.
- It **never blind-passes `--trust-remote-code`** — a governed security slot
  is stripped from the captured body and only ever emitted if the trc gate
  explicitly permits, which it never does in-scope (see §6).
- It **synthesizes nothing** — every non-insertion-point line, including all
  constants and param-slot `${VAR:-default}` expressions, reproduces
  byte-for-byte from the captured shipped compose (see §4).

A generated compose that won't boot is a *true signal* that something in the
captured config drifted, not a bug to be silently patched around. Repair is a
separate, out-of-scope loop.

---

## 2. Scope and the two clean-refuse classes

The generator handles **non-Genesis, vLLM-only** profiles. Anything else is a
clean refusal (stderr message + non-zero exit), never a partial or
best-effort emit.

Scope is decided by two gates that run **first**, before any patch or
template work:

| # | Gate | Discriminator | Refuse message (verbatim shape) | Exit |
|---|---|---|---|---|
| 1 | Engine type | `engine.type != "vllm"` | `engine <id> type='<t>' != vllm; the #141 generator is non-Genesis vLLM only -> refuse (out of scope)` | `2` |
| 2 | Genesis-equipped | profile `genesis_equipped: true` | `profile <p> is genesis_equipped:true (<evidence>); Genesis-flag generation is permanently out of scope -> refuse` | `2` |

### `genesis_equipped` discriminator

`genesis_equipped` is captured per profile in
`scripts/lib/profiles/profile_runtime.yml` using the locked v0.8.x §6
discriminator: a profile is `genesis_equipped: true` iff its shipped compose
contains a `_genesis` / `GENESIS_PIN` / `GENESIS_ENABLE` token **OR** its
`kv_format` starts with `turboquant`. (It is *not* keyed off an engine's
`required_genesis` flag.) Observed examples:

```text
$ scripts/generate-compose.sh --profile vllm/tools-text
[generate-compose] REFUSE: profile vllm/tools-text is genesis_equipped:true
(compose contains GENESIS_* / _genesis token); Genesis-flag generation is
permanently out of scope -> refuse

$ scripts/generate-compose.sh --profile vllm/dual-tq3-mtp
[generate-compose] REFUSE: profile vllm/dual-tq3-mtp is genesis_equipped:true
(kv_format 'turboquant_3bit_nc' starts with 'turboquant'); Genesis-flag
generation is permanently out of scope -> refuse

$ scripts/generate-compose.sh --profile llamacpp/default
[generate-compose] REFUSE: engine llama-cpp-local type='llama.cpp' != vllm;
the #141 generator is non-Genesis vLLM only -> refuse (out of scope)
```

Genesis-flag compose generation is **permanently** out of scope (a locked
v0.8.x decision), not a deferred feature.

---

## 3. Input — `--profile` is authoritative

```
scripts/generate-compose.sh --profile <COMPOSE_REGISTRY key> [--out FILE] [--accept-degraded]
```

`--profile` is the **only** authoritative input. It must be a key in
`scripts/lib/profiles/compose_registry.py` (`COMPOSE_REGISTRY`). An unknown
profile is a usage error:

```text
$ scripts/generate-compose.sh --profile vllm/does-not-exist
[generate-compose] REFUSE: unknown profile 'vllm/does-not-exist' (not in COMPOSE_REGISTRY)   # exit 64
```

### Convenience tuple (discovery aid only — NOT authoritative)

`--model` / `--engine` / `--kv` / `--tp` form a convenience tuple that
**lists candidate `--profile` values and exits non-zero (4)**. It never
generates — it is a discovery aid so you can find the authoritative profile
name:

```text
$ scripts/generate-compose.sh --model gemma-4-31b --engine vllm-gemma-stable
convenience tuple matched these profiles (re-run with an authoritative --profile):
  --profile vllm/gemma-int8-mtp
  --profile vllm/gemma-bf16-mtp
  --profile vllm/gemma-mtp-tp1
                                                                  # exit 4
```

No args at all → usage error (exit `64`).

### Output

`--out FILE` writes the compose there (parent dirs created) and prints a
one-line summary to stderr; without `--out` the compose goes to stdout. The
3-category provenance header (§7) is always part of the emitted file.

### Capacity values are the reference profile's — NOT fit-adapted to your hardware

The generated compose's capacity knobs — `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, and the KV cache dtype — are copied **verbatim from the captured reference profile** (the shipped compose it reproduces). The generator does **not** run a fit solve, does **not** size context to *your* GPU's VRAM, and does **not** down-cast the KV dtype for capacity. So:

- On a card **smaller** than the reference target, the emitted `--max-model-len` may not boot — it is not shrunk for you.
- On a card **larger** than the reference target, you are leaving context/throughput on the table — it is not grown for you.
- For a **derived** (non-curated) model the KV dtype is whatever the model ships (often `bf16`); it is not optimised toward a denser hardware-legal format.

This is deliberate (Mission §1 — reproduce + flag, NEVER repair): the generated file is a **known-safe starting point**, not a hardware-tuned config. For the actual fit on *your* hardware, run `scripts/pull.sh <slug> --profile-like <key> --recommend` (or `tools/kv-calc.py --solve-max-ctx ...`) and tune the emitted `${MAX_MODEL_LEN}` (it is intentionally an env-overridable default) accordingly. An opt-in capacity optimiser is planned for a later release; until then, right-sizing is a deliberate user step.

### Running a generated compose — it is NOT relocatable

Per the whole-service-template model (§4), the generator copies the shipped
service definition **verbatim** — including its bind-mount *sources*, which are
**repo-relative** (e.g. `../../patches/<patch>/...`, `../../cache/...`, and the
`${MODEL_DIR:-../../../../../models-cache}` default). The generator deliberately
does **not** rewrite these (it synthesizes nothing). They resolve correctly only
when Docker Compose's project directory is the shipped compose's own directory.

Consequence: a generated file run from an arbitrary location (e.g.
`docker compose -f /somewhere/else/out.yml up`) has **dangling bind-mount
sources**. Docker silently creates empty directories at the missing source
paths, so e.g. a mounted chat-template file becomes a directory and vLLM aborts
at boot with `... looks like a file path, but it failed to be opened ...
Is a directory`.

Run a generated compose with `--project-directory` anchored to the shipped
compose's directory (the value of `compose_service_template.source` in
`profile_runtime.yml`), and set the same `MODEL_DIR` / `VLLM_IMAGE` the repo
launcher sets:

```
docker compose \
  --project-directory models/<model>/vllm/compose/<topology> \
  -f <generated-file> up -d
```

Equivalently, write `--out` into that shipped compose directory and run it
there. This is consistent with §10 (coexistence with the pre-baked tree): the
generated artifact is a drop-in *for that location*, not a portable standalone.

---

## 4. The whole-service-template model

The capture unit is the **entire shipped service definition**, recorded per
profile in `profile_runtime.yml` under `compose_service_template`. The
generator loads the file named at `compose_service_template.source` and
classifies every token into exactly one of three classes:

| Class | What it is | What the generator does |
|---|---|---|
| **param-slot** | `${TP}`, `${MAX_MODEL_LEN}`, `${GPU_MEMORY_UTILIZATION}`, `${KV_CACHE_DTYPE}`, `--max-num-seqs`, `--speculative-config`, ports, NVLink env | Reproduced as the shipped `${VAR:-default}` expression (the registry carries the resolved value; the env-default mechanism does the substitution at runtime). `--speculative-config` is **omitted entirely** when the profile has no drafter. |
| **governed-slot** | `--trust-remote-code` (and any future security/policy flag) | Captured but flagged *governed*. Emitted **only** when the gate state explicitly permits — which never happens in-scope (§6). |
| **constant** | The `image:` expression, `shm_size`, `ipc`, `deploy.resources`, model-cache volume, base `environment`/`entrypoint`, `--model`, `--served-model-name`, `--quantization`, `--dtype`, `--tool-call-parser`, `--chat-template`, `--reasoning-parser`, `--default-chat-template-kwargs`, `--enable-prefix-caching`, `--enable-chunked-prefill`, `--enforce-eager`, … | Reproduced **verbatim, byte-for-byte**. |

The **only** transformation the generator applies to the body is at **two
named insertion points**:

- `volumes:` — overlay / sidecar bind-mount lines.
- `entrypoint:` — sidecar / install-script invoke lines.

A selected+wired patch keeps its mount/invoke lines; a selected-but-omitted
patch (delivery-gap or failed-guard) has *its* lines stripped. **Everything
else is reproduced exactly** — the generator synthesizes nothing. For a
golden profile the compose-keyed selection reproduces precisely the patch set
the maintainer shipped, so the generated service body is byte-identical to
the shipped file outside the insertion points (the golden-parity invariant,
§9).

---

## 5. The image expression is preserved, never rewritten

The shipped composes use
`image: ${VLLM_IMAGE:-vllm/vllm-openai:nightly-${VLLM_NIGHTLY_SHA}}`. This is
a **captured constant** and is passed through **verbatim**. The engine pin
flows through the existing `VLLM_NIGHTLY_SHA` / `VLLM_IMAGE` env mechanism —
the generator does **not** rewrite the image line.

The engine pin is **validated** (§8), not substituted. Validation confirms
the captured config is internally consistent; it does not edit the artifact.

---

## 6. `--trust-remote-code` is a governed slot, never emitted in-scope

`--trust-remote-code` is governed under the locked v0.8.x §88 security model.
It is treated as a governed-slot, not a constant:

1. **Gate (step 5):** the arch's `requires_trust_remote_code` is read from
   `arch_patches.yml`. If it is `true` or `unverified`, the generator
   **refuses** (security refusal, exit `2`) — trc acknowledgement is the
   pull-gate's job, not the generator's.
2. **Suppression:** when generation proceeds (arch trc = evidence-cited
   `false`), `trc_emit` is provably `False`, and the generator **strips** any
   `--trust-remote-code` token from the captured body before emitting.

The combination guarantees an in-scope generated compose **never
blind-passes `--trust-remote-code`**. Every golden triple is asserted to emit
no `--trust-remote-code` and report `meta.trc_emitted == False`. The header
states this explicitly:

```text
#   NOTE: --trust-remote-code is a GOVERNED slot (locked §88); it is
#   NOT emitted for any in-scope profile (arch trc gate = false).
```

Any future security/policy flag joins the governed-slot set and gets the same
treatment.

---

## 7. Patch selection, delivery gaps, drift guards (§4.1 graded drift)

Patch selection is **compose-keyed only**: patch `P` is selected for profile
`X` iff `X ∈ P.load_bearing_when[].composes`. Every other patch is
**EXCLUDED** (not load-bearing for this compose) — listed in header category
[3], emitted nowhere.

For each selected patch the generator classifies it, in this order:

1. **Delivery-gap first.** If the patch declares a `delivery_gaps[]` entry
   covering this profile, it is **selected-but-undelivered**: its wiring is
   omitted, a header WARNING records the gap's `issue` string, and the
   drift-guard is **skipped entirely**. This is the structural acknowledgement
   of a known coverage boundary (the #145 class — see
   [`docs/PATCH_POLICY.md`](PATCH_POLICY.md)).
2. **Drift-guard (graded, §4.1).** For a will-be-wired patch the generator
   records its `drift_guard`. The guard is a runtime import/boot/behavioral
   probe that **cannot run at generation time** (no engine container here);
   the contract is that the shipped, maintainer-tested state is "applies
   cleanly" (= drift-guard-tested, a locked decision), so the patch is wired
   and the guard is surfaced for the boot leg / operator to re-run. When a
   guard *does* fail (the test harness drives this deterministically via
   `CLUB3090_FORCE_GUARD_FAIL`; there is no other way to force a runtime
   probe failure from a unit test), the **gap-before-guard** ordering and the
   §4.1 grade decide the outcome:
   - **capability-scoped** fail → patch **OMITTED**, compose flagged
     **DEGRADED**. The run requires `--accept-degraded` to proceed; without
     it the generator refuses with exit `3`. A failed patch is **never
     wired**.
   - **foundational** fail (or `on_fail: hard-refuse`) → **hard-refuse**
     (exit `2`) — even *with* `--accept-degraded`. Weights/boot depend on it;
     there is nothing safe to emit.

Observed degraded path (forced fail, no ack):

```text
$ scripts/generate-compose.sh --profile vllm/gemma-int8-mtp    # CLUB3090_FORCE_GUARD_FAIL=gemma-vllm-gemma4-tool-parser-fixes
[generate-compose] REFUSE: DEGRADED: capability-scoped patch(es)
[gemma-vllm-gemma4-tool-parser-fixes] omitted after a failed drift-guard; re-run with
--accept-degraded to proceed                                  # exit 3
```

With `--accept-degraded` the same case proceeds: the patch is in
`degraded_omitted` (never `wired`), its mount lines are physically absent
from the body, and the header carries a `WARNING: DEGRADED` block.

The generator **never repairs** — capability-scoped failure degrades,
foundational failure refuses; neither edits the patch.

---

## 8. Engine-pin `<id>@<sha>` validation

Step 3 validates the engine pin (it does **not** rewrite the image, §5). The
pin is derived as `<engine_id>@<sha>` from the engine's `install.spec` (the
concrete image tag), and must match an entry in the resolved arch row's
`engine_pin[]` with `loads: true`:

- `<id>` (no sha) or `<id>@<sha>` forms are both accepted.
- A pin that matches the engine but is `loads: false` → refuse, surfacing the
  pin's `reason`.
- No `loads: true` match for the engine → refuse.

The arch row is resolved via `arch_model_xref` (`E.model ∈ model_slugs`), and
two further validations gate generation: `E.tp` must be in the arch's
`valid_tp.tp_divisors`, and `E.kv_format` must be in the engine's
`supported_kv_formats`. All are clean refusals (exit `2`) on failure. The
validated pin is recorded in the header:

```text
#   engine-pin (validated, image NOT rewritten): vllm-nightly-clean@bf610c2f...
```

The `kv_format → --kv-cache-dtype` map (non-Genesis only) is exercised on
every run and unit-tested:

| `kv_format` | emitted `--kv-cache-dtype` |
|---|---|
| `bf16`, `fp16` | *(no arg)* |
| `fp8_e5m2` | `fp8_e5m2` |
| `fp8_e4m3` | `fp8` |
| `int8_per_token_head` | `auto+PTH` |
| `q4_0` | `q4_0` |
| `k8v4` | `k8v4` |

---

## 9. The 3-category header and the 5-triple golden-parity guarantee

Every generated compose carries a provenance header **above** the top-level
`services:` key (so the patch-attribution `service_body()` parser discards
it — header text can never create a false-positive reachability hit). The
header has three categories:

1. **selected + WIRED** — drift-guard-tested, wired at `volumes:`/`entrypoint:`.
2. **selected + UNDELIVERED** — delivery-gap or failed-guard; wiring omitted,
   with the reason.
3. **EXCLUDED** — not load-bearing for this compose.

Plus the validated engine pin, the mission line, an optional `WARNING:
DEGRADED` block, and the governed-trc note.

**Golden-parity guarantee.** `scripts/tests/test-generate-compose.sh` asserts
five golden triples, all verified `genesis_equipped: false`, spanning every
in-scope engine class:

| Profile | Engine | Notes |
|---|---|---|
| `vllm/minimal` | `vllm-nightly-clean` | tp1, fp8, drafter=None; surfaces the qwen3coder delivery-gap as undelivered |
| `vllm/dual` | `vllm-nightly-clean` | tp2, fp8, MTP drafter |
| `vllm/gemma-bf16-mtp` | `vllm-gemma-stable` | gemma, bf16 |
| `vllm/gemma-int8-mtp` | `vllm-gemma-stable` | int8-PTH, multi-file overlay |
| `vllm/gemma-mtp-tp1` | `vllm-gemma-stable` | single-card fp8 risk path |

Per triple the test asserts: the semantic diff vs the shipped compose is
confined to the two patch insertion points (image expression + every constant
reproduce verbatim); selected+wired ⊆ shipped; wired patches pass `reaches()`
on the *generated* compose (actual wiring, not header text);
selected-but-undelivered patches are **not** reachable; the 3-category header
is present; and no `--trust-remote-code` is emitted. The refusal/degraded
matrix (genesis, llama.cpp, foundational hard-refuse, capability-scoped
degrade ± ack, convenience tuple) is asserted alongside. At least one triple
is booted on-rig.

---

## 10. Coexistence with the pre-baked compose tree

The generator does **not** replace or delete the existing hand-maintained
`models/<model>/<engine>/compose/...` tree. The shipped composes remain the
ground truth — they are exactly what the generator captures from and
reproduces. Operators who want the curated, maintainer-tested file keep using
it directly; the generator is for *deriving a minimal reproduction* of an
in-scope profile (e.g. to bisect drift, or as the `[D]` substrate of the
v0.8.x pull flow). There is no big-bang tree deletion in v0.8.0.

---

## 11. Exit codes

| Code | Meaning |
|---|---|
| `0` | compose emitted (possibly DEGRADED with `--accept-degraded`) |
| `2` | clean scope / validation / foundational refusal |
| `3` | capability-scoped DEGRADED, `--accept-degraded` absent |
| `4` | convenience tuple matched profiles (not authoritative — re-run with `--profile`) |
| `64` | argv / lookup misuse (no `--profile`, unknown profile) |
