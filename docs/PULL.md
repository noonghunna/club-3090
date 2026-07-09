# Pull — serve any HF safetensors model (v0.8.0)

**User guide.** You have a Hugging Face model and you want to know: *will it
run on my GPUs, and if so, how?* `scripts/pull.sh` answers that — it
evaluates the repo against this stack's precise KV math before you download
anything, and is honest about how much it trusts the answer.

> **v0.8.0 headline:** *"Evaluate any safetensors HF repo; pull only
> vLLM-loadable supported ones, and only when the gates pass (or an explicit
> override is accepted)."*
>
> **v0.8.2 adds** (additive only — no v0.8.0 decision logic changed): a
> failure on-ramp (a redacted-diagnostics submit path when a pull fails),
> arch-registry expansion (materially more safetensors models reach
> `engine-supported`), an optional hardware-detect slice for non-NVIDIA
> enumeration, and the `--recommend` UX (an honest aggregated
> recommendation over the same verdict). This release also **bundles two
> non-`pull` items that shipped on the same branch**: N-GPU NVLink
> auto-detection (wired into the multi-4 and Gemma-4-26B dual composes)
> and a documentation restructure (a quick-start-first README, a new
> `GETTING_STARTED.md`, model READMEs, and an updated docs index) — both
> are independent of and orthogonal to the `pull` decision path. **GGUF
> is deferred** — it is not a v0.8.2 item: cross-engine generation (GGUF
> / llama.cpp serving via `pull`) is a separate **§9 cross-engine
> design-unlock proposal**, not this release. Pointing `pull` at a
> GGUF-only repo is a known scope boundary, not a stack failure (see [the
> readiness ledger](#release-readiness-ledger--honest-deferrals) below).

This is the **user front door**. For the contributor/maintainer internals
of the same pipeline (gate strata, classifier, trust pipeline) start at
[`docs/README.md`](README.md) → the Contributor track.

---

## Quickstart

One command. Replace the slug with your model; `--profile-like` borrows a
curated runtime shape (a `COMPOSE_REGISTRY` key like `vllm/minimal` —
see [Usage](#usage) below for what the keys mean).

```bash
# Just check — never downloads, never boots:
scripts/pull.sh <org/Model> --profile-like vllm/minimal --dry-run

# Evaluate, then (if it passes) download + emit a compose + boot it:
scripts/pull.sh <org/Model> --profile-like vllm/minimal --yes
```

What you'll see — exactly one of:

| Outcome | Exit | Means |
|---|---|---|
| `proceed` / `confirm→proceed` | `0` / `3` | Fits. `0` = clean; `3` = re-run with the named flag (e.g. `--yes`) to continue. |
| `hard-block` | `2` | Honest stop with a precise reason (unsupported engine/arch, won't-fit, disk, needs `--trust-remote-code`). Nothing downloaded. |
| `override-accepted` | `0` | You explicitly accepted a non-pass path (e.g. `--force-download`); proceeds with the caveat recorded. |

> **First-run heads-up:** many common models (anything `Qwen2ForCausalLM` — Qwen2.5 & a large family, plus other custom-code archs) hard-block at `[C0] needs-trust-remote-code-ack` on the *very first* try — **even with `--dry-run`**. That's the gate working, not a failure. After you've checked what code the repo would run, add **`--trust-remote-code`** to that same command to clear it. See [`--trust-remote-code` — a security decision](#--trust-remote-code--a-security-decision) below.

It is **honest about confidence and never silently passes.** A "fits"
verdict is a *boot-time* check — read [Boot-fit ≠ runtime-stability](#boot-fit--runtime-stability--read-this)
before relying on it for sustained agent workloads. Full detail below.

---

## What changed in v0.8.0

Older releases worked one way: the repo *formally supported* a fixed list
of models, and you picked from that list. That still works and still ships
— see [`docs/SINGLE_CARD.md`](SINGLE_CARD.md) /
[`docs/DUAL_CARD.md`](DUAL_CARD.md) / [`docs/MULTI_CARD.md`](MULTI_CARD.md),
the curated catalog is unchanged.

v0.8.0 **adds** a model-agnostic front door. The stack no longer needs to
"formally support model X" per release to be useful for X. Instead:

- You hand `pull` *any* safetensors HF repo slug.
- It derives the model's shape from the repo's own `config.json` and runs
  it through this stack's KV math.
- It returns a verdict **with an explicit confidence tier**, and tells you
  which gate decided.

The curated catalog doesn't go away — it becomes the **calibration
backbone**: the measured corpus the math is anchored against. Curated
models get an `exact` confidence tier; arbitrary repos get an honest
lower-bound estimate. Both are first-class; the difference is stated, never
hidden.

---

## Usage

```
scripts/pull.sh <hf-slug> --profile-like <COMPOSE_REGISTRY-key> [opts]
```

`--profile-like` is **required**: it names a curated registry key that
supplies the runtime shape (engine, KV format, TP) to evaluate against.

### Path A — curated pull-and-emit

The slug is a curated, generator-emittable model. On a gate-passing run
`pull` hands the validated key to the #141 compose generator and emits a
ready compose.

```
scripts/pull.sh Lorbus/Qwen3.6-27B-int4-AutoRound \
    --profile-like vllm/minimal --out qwen.yml
```

> **The emitted compose carries the reference profile's capacity values, not a fit tuned to your GPU.** `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs` and the KV dtype are copied from the captured profile — *not* re-solved for your card. It is a known-safe starting point: add `--recommend` (or run `tools/kv-calc.py --solve-max-ctx`) for the honest fit on your hardware, and tune the emitted env-overridable `${MAX_MODEL_LEN}` accordingly. See `docs/COMPOSE_GENERATOR.md` § "Capacity values are the reference profile's".

### Path B — universal evaluate (never downloads, never emits)

Any non-curated slug, or `--dry-run` on anything, takes Path B. It prints
a confidence-tiered verdict and **never calls the generator and never
downloads weights**.

```
scripts/pull.sh some-org/Some-Llama-7B --profile-like vllm/minimal --dry-run
```

### Options

| Opt | Meaning |
|---|---|
| `--yes` | Accept a `confirm→proceed` terminal (§4.1 — see "Reading the verdict"). |
| `--force-download` | Advisory low-confidence `wont-fit` → `override-accepted`. **No-op + notice this phase** (download deferred to a later phase). |
| `--experimental-arch` | Bypass *only* a `[C0] engine-support-unknown` (no arch row) hard-block; attempt with default vLLM settings. Path B only this phase. |
| `--trust-remote-code` | Bypass a `[C0] needs-trust-remote-code-ack` hard-block (security decision — see below). |
| `--hf-home DIR` | Override the `HF_HOME` resolution chain (where disk is checked / weights would land). |
| `--out FILE` | Path A: write the emitted compose here. |
| `--hardware SM` | Override detected GPU compute capability (e.g. `8.6` for RTX 3090); default = `nvidia-smi`. |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Download-eligible / clean verdict. |
| `3` | Needs a flag — a `confirm→proceed` or advisory terminal that is not yet satisfied (re-run with the named flag). |
| `2` | Honest hard-stop — a gate aborted, or a `hard-block` terminal. |
| `64` | Usage error — missing/unknown argument (distinct from `2`, so a typo is distinguishable from an honest gate-block). |

> *Note: the `64` usage-vs-`2` hard-stop split is a post-`v0.8.0` fix — present on `master`/the next release; the `v0.8.0` release tag still exits `2` for argument errors.*

---

## Reading the verdict

Every run prints the **confidence tier** and **which gate decided** — this
is non-negotiable: the gate never silently passes anything except an
`exact × fits-clean` case.

Two axes combine:

- **confidence** ∈ `{exact, estimated-lower-bound}` (a `derived` tier is
  reserved for a future phase). `exact` = a curated calibration anchor;
  `estimated-lower-bound` = a derived estimate where the modelled VRAM is a
  *floor* and is likely under-modelled.
- **raw_verdict** ∈ `{fits-clean, fits-constrained, wont-fit}` — the KV
  math's pure measurement, no policy.

The two map to a **terminal** ∈ `{proceed, confirm→proceed, hard-block,
override-accepted}`:

| confidence | `fits-clean` | `fits-constrained` | `wont-fit` |
|---|---|---|---|
| `exact` | **proceed** (silent — the only silent pass) | **confirm→proceed** (`--yes`; a constraint changed your requested config) | **hard-block** (math trusted; closest-fit suggested) |
| `estimated-lower-bound` | **confirm→proceed** (`--yes`; VRAM is a floor, likely under-modelled) | **confirm→proceed** (`--yes` + floor + constraint notice) | advisory → `--force-download` → **override-accepted** |

The output names the stratum/gate that decided. An illustrative Path B line
(shape derived from the tool's actual print sites — your values will
differ):

```
[pull] OK path=B stratum=DECIDED slug=some-org/Some-Llama-7B profile-like=vllm/minimal
[pull] confidence=estimated-lower-bound raw_verdict=fits-clean terminal=confirm→proceed
[pull] Path B verdict: [C1] estimated-lower-bound×fits-clean → confirm→proceed (VRAM is a floor; likely under-modeled)
[pull] note: boot-fit satisfied; this does NOT guarantee stability under sustained / accumulated-context workloads — validate with soak-continuous before relying on it (recommend: scripts/soak.sh SOAK_MODE=continuous).
```

`override-accepted` is **not** a gate-pass. It is the deliberate, explicit
path for forcing a low-confidence `wont-fit`: it records the outcome as a
calibration signal, it does not record "fit validated".

---

## Boot-fit ≠ runtime-stability — read this

The KV math is a **static, boot-time allocation** check. Passing it is
*necessary but not sufficient* for real workloads. On this hardware class,
measured failure modes exist that a static check cannot see:

- **Cliff 2** — degradation/OOM at roughly **21–26K accumulated context**
  under accumulated-context agent workloads (hermes/openhands style).
- **Prefill cliffs** at single-prompt sizes well below the static ceiling.
- **Cliff 2b** — only detectable under a *continuous soak*, not a single
  request.

So a `fits-clean` / `proceed` config can still degrade or OOM once a real
agent accumulates context. Honesty is non-negotiable on this stack, so the
verdict output always carries this caveat verbatim:

> *boot-fit satisfied; this does NOT guarantee stability under sustained /
> accumulated-context workloads — validate with soak-continuous before
> relying on it (recommend: scripts/soak.sh SOAK_MODE=continuous).*

**Before you rely on any `fits-clean` / `proceed` config in production,
run** `scripts/soak.sh SOAK_MODE=continuous` — it is the only test that
catches Cliff 2b. A `fits-clean` that silently dies under soak is exactly
the confidently-wrong outcome this design forbids; the soak makes the
*predicted* side honest about its scope. See [`docs/CLIFFS.md`](CLIFFS.md)
for the full diagnosis of these failure modes.

---

## `--trust-remote-code` — a security decision

Some HF repos ship custom modelling code that the loader executes. If the
architecture's matrix entry requires it, the gate **hard-blocks** with
`needs-trust-remote-code-ack` and prints what code origin would run. It
does not proceed until you explicitly pass `--trust-remote-code`. This is a
deliberate fail-closed security gate — *do not reflexively pass the flag to
clear an error*; understand what code you are authorizing first.

A genuinely unknown architecture (no entry in the patch matrix at all)
hard-blocks differently — with `engine-support-unknown`. Pass
`--experimental-arch` to attempt it anyway with default vLLM settings; the
outcome is captured to inform support coverage.

---

## What happens after a pass

`pull` itself stays user-level — it evaluates and (Path A) emits. For the
depth behind a passing run:

- **The download → boot → smoke path** for a download-eligible derived
  model: [`docs/PULL_EMIT_DERIVED.md`](PULL_EMIT_DERIVED.md) (the `[E]`
  stage).
- **The contribution loop** — how a boot/OOM outcome becomes a classified,
  deduped, consensus-keyable calibration signal:
  [`docs/LOOP.md`](LOOP.md) (the `[F]` stage).
- **The compose the generator emits** and how it is shaped:
  [`docs/COMPOSE_GENERATOR.md`](COMPOSE_GENERATOR.md).

---

## Where weights land

`MODEL_DIR` is the single storage root — set it once (the repo `.env` ships a
default) and every path below derives from it, so a model downloaded by one tool
is found by the others and you never re-download the same weights.

| Tier | Path | Notes |
|---|---|---|
| **Curated catalog** | `$MODEL_DIR/<model>/` (safetensors) · `$MODEL_DIR/<model>-gguf/<variant>/<file>.gguf` (GGUF) | Human-readable, registry-tracked, referenced by the compose files. The ⑤ *Promote* target — a validated model in the catalog. |
| **BYO / brought pulls** | `$MODEL_DIR/.cache/huggingface/club3090/pulls/<repo-slug>/` | `<repo-slug>` = the HF repo id, lowercased, with `/` and other non-`[a-z0-9._-]` runs collapsed to `-` (`migtissera/Tess-4-27B-GGUF` → `migtissera-tess-4-27b-gguf`). **`pull.sh` and the c3 Bring `[D]` button compute the SAME path**, so a pull by either is discovered by both. Staging until you promote it. |
| **HF hub cache + token** | `$MODEL_DIR/.cache/huggingface/` (`HF_HOME`) | Shared download staging for all pulls. |

Two deliberate tiers: the **curated catalog** (top-level, compose-referenced) and
the derivable **BYO staging** the Bring funnel operates on. Point the root
elsewhere per-run with `--hf-home DIR` (or the `HF_HOME` env); everything else
follows from it.

---

## `--recommend` — the honest one-line answer

Add `--recommend` to any `pull` invocation and, after the gate runs, you
get an aggregated plain-language recommendation: does it fit, on which
profile/variant, at what confidence, and **which gate decided** — plus the
boot-fit≠runtime caveat verbatim when the verdict reached the fit math.

```bash
scripts/pull.sh <org/Model> --profile-like vllm/minimal --dry-run --recommend
```

It is **presentation only**: every line is read straight off the same
verdict the gate already produced — `--recommend` never changes the
decision, the exit code, or what gets downloaded/emitted. It is honest by
construction:

- It echoes the real confidence tier; an `estimated-lower-bound` fit is
  stated as a floor, never dressed up as a guarantee.
- It is **vLLM-only** (the gate is vLLM-only; `--recommend` only echoes
  that).
- It **never implies an artifact that was not produced** — the
  "compose emitted" line appears only when a compose was actually emitted
  (Path A); a Path B / `--dry-run` recommendation says so explicitly.
- A `FITS` verdict carries the §7 caveat and the
  `scripts/soak.sh SOAK_MODE=continuous` pointer; a pre-fit-math
  hard-block does **not** (it never reached the fit math, so it makes no
  soak claim).

When the verdict is *blocked*, `--recommend` points you at the failure
on-ramp below.

---

## Report a failed pull

When a `pull` fails (a gate hard-block, or a `fits-clean` that then
fails to boot), the run leaves a **redacted diagnostics bundle** on disk
and prints exactly where it is and the one command to send it back:

```
[pull] Diagnostics captured (redacted, no paths/tokens): .pull-captures/<slug>/<ts>
[pull] Help improve the fit math — submit with: scripts/pull.sh --submit-last
```

This is the success-path's mirror image: `--recommend` tells you what to
run; this tells you "that failed — help us fix the fit math." It is
entirely opt-in and the `pull` run itself does **no** network — capture is
a local file write only.

### The on-ramp, step by step

1. **Capture is automatic and redacted.** On any failure terminal `pull`
   writes a bundle under `.pull-captures/<slug>/<ts>` and records it as
   the most-recent capture. The bundle is already scrubbed of paths and
   tokens — you never paste terminal scrollback (your console output is
   *not* a safe source; the artifact is).

2. **You submit it deliberately, in a separate command.** Submission is a
   distinct, explicit, consented step — never automatic, never a phone
   home:

   ```bash
   # Submit the most-recent capture:
   scripts/pull.sh --submit-last

   # Or submit a specific bundle by directory:
   scripts/pull.sh --submit .pull-captures/<slug>/<ts>
   ```

   `--submit-last` and `--submit <dir>` need **neither a slug nor
   `--profile-like`** — they are a different verb from a gate run.

3. **You see the exact payload, then consent.** Before anything leaves
   your machine the command prints the resolved bundle identity and the
   **exact already-redacted payload that would be sent**, then asks:

   ```
   [submit] resolved bundle: <abs dir>
   [submit] identity: slug='<org/Model>' utc_ts='<ts>' outcome='hard-block' schema=2 ...
   [submit] this is the EXACT already-redacted payload that will be sent (no terminal scrollback, no paths/tokens):
   ------------------------------------------------------------------------
   ... the redacted report ...
   ------------------------------------------------------------------------
   [submit] §6.1 class=<class> should_file=<bool>
   [submit] submit this redacted report? [y/N]
   ```

   Anything other than a leading `y`/`Y` (including just Enter, or EOF in
   a non-interactive shell) is a decline:

   ```
   [submit] declined — nothing sent (no network performed).
   ```

   Network happens **only** after an explicit `y`.

4. **With `gh` (the GitHub CLI) installed and authenticated**, a
   consented submit reuses the deduped contribution path: equivalent
   reports are coalesced (a duplicate adds a `+1` rather than opening a
   new issue), and a correct-refusal class is spooled to the maintainer
   triage queue instead of opening a public issue. You'll see a line like
   `[submit] F5 action=... dedup_hash=... issue=...` (and a
   `[submit] spool: ...` line when it was queued, not filed).

5. **Without `gh`**, the same consented submit degrades cleanly: for a
   solicited (actionable) class it prints a prefilled GitHub issue URL you
   can paste into a browser; for a correct-refusal / unactionable class it
   prints the **local** triage-spool path and a "captured for maintainer
   triage; not a public issue" line — and **no** public-issue URL. Either
   way, the only thing you are ever asked to share is the already-redacted
   artifact, never a filesystem path or terminal output.

If `--submit-last` finds no recent capture it tells you plainly and points
at the explicit-directory form:

```
[submit] no recent capture; use `scripts/pull.sh --submit <dir>`
```

---

## Release readiness ledger — honest deferrals

`pull` evaluates and serves **safetensors via vLLM only**. This is a
deliberate scope boundary, stated up front so a §9 reader is not
surprised. v0.8.2's shipped scope is the four `pull` items below **plus**
two orthogonal, non-`pull` items that landed on the same release branch
(N-GPU NVLink auto-detection and a documentation restructure) — listed
here so the bundled scope is stated honestly, not under-claimed:

| Item | Status in v0.8.2 |
|---|---|
| Safetensors evaluate + (Path A) emit + boot | shipped (v0.8.0) |
| Failure on-ramp (capture + consented `--submit*`) | shipped (v0.8.2) |
| Arch-registry expansion (more models reach `engine-supported`) | shipped (v0.8.2) |
| Optional non-NVIDIA hardware-detect slice | shipped (v0.8.2, optional) |
| `--recommend` UX | shipped (v0.8.2) |
| N-GPU NVLink auto-detection (multi-4 + Gemma-4-26B dual composes) | shipped (v0.8.2, bundled — not a `pull` item) |
| Documentation restructure (quick-start README, `GETTING_STARTED.md`, model READMEs, docs index) | shipped (v0.8.2, bundled — not a `pull` item) |
| **GGUF** repos (evaluate **and** serve) | **deferred — see below** |
| **`.bin`** / non-safetensors weight layouts | deferred |
| Cross-engine generation (llama.cpp serving via `pull`) | deferred — see below |

**GGUF is deferred to a §9 cross-engine design-unlock proposal**, not a
later patch release of this line. The reason is structural, not a backlog
slip: `pull` emits and serves vLLM by design, so a GGUF path would be
either a thin evaluate-only calculator (no launcher) or full llama.cpp
serving — and **cross-engine generation is §9 "deferred indefinitely"**,
i.e. it requires its own design-unlock and review round before any
implementation. If GGUF matters to you, the next artifact is that
design-unlock proposal, not a GGUF feature in this line.

If you point `pull` at a GGUF-only or `.bin`-only repo today, that is this
documented scope boundary, **not** a stack failure.
