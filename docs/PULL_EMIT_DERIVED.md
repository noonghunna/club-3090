# Pull-Emit-Derived — the `[E]` stage of `scripts/pull.sh` (v0.8.0, #147)

Contributor/maintainer guide for the v0.8.0 Pull-Emit-Derived stage. The
Pull-Gate ([`docs/PULL_GATE.md`](PULL_GATE.md)) evaluates *any* safetensors
HF repo and pull-and-emits **curated** (Tier-1) models only. `[E]` is the
deferred extension that makes a **download-eligible derived (non-curated)**
model real: it emits a patchless default-vLLM compose, downloads the weights,
boots them, smokes the endpoint, and writes the §6 capture artifacts the
later Loop phase consumes.

`[E]` is **post-Pull-Gate, not a new gate**: it runs as a stage *inside*
`run_pull()`, strictly after a `[C1]` terminal, and never alters the locked
6-stratum taxonomy, the `[C1]` table, or curated Path-A behaviour. Those are
*consumed*, never modified.

> ### The authoritative spec is the test, not this doc
> Per the locked v0.8.0 stop-condition, the executable specs are
> authoritative. For `[E]` those are
> **`scripts/tests/test-generate-from-profile.sh`** (CONTRACT-2/5: the
> derived template + the eligibility gate),
> **`scripts/tests/test-pullgate-download.sh`** (CONTRACT-3: the download
> stage), **`scripts/tests/test-pullemit-capture.sh`** (CONTRACT-4: the §6
> capture emitters + manifest), and the `[E]` continuations / trigger matrix
> in **`scripts/tests/test-pull.sh`** (g16–g22). Where this prose and any of
> those tests ever disagree, **the test is correct**; fix the doc. This
> document is the explanatory companion.

For the curated `[D]` substrate `[E]` reuses for the engine-pin resolver and
the structured-refusal style, see
[`docs/COMPOSE_GENERATOR.md`](COMPOSE_GENERATOR.md); for the patch/arch/
profile data model, [`docs/PATCH_ATTRIBUTION.md`](PATCH_ATTRIBUTION.md) and
[`docs/PATCH_POLICY.md`](PATCH_POLICY.md).

> **On-rig status:** the mocked suites listed above are green and are the
> shipped E1–E4 contract. The real small-non-curated end-to-end run
> (`pull <slug> --profile-like <vLLM key>` → derive → gates → `[C1]` →
> emit → download → boot → smoke → teardown) is **pending on-rig validation
> (E5)** and is driven separately. Nothing below describes the on-rig run as
> validated; it describes the shipped, unit-verified E1–E4 behaviour.

---

## 1. What `[E]` adds on top of Pull-Gate

| Class of model | Pull-Gate behaviour | With `[E]` |
|---|---|---|
| **curated (Tier-1)**, Path A | full gate → stratum-6 `[D]` dry-run → real `[D]` emit of the validated registry key; **no download stage** (curated weights are local) | **unchanged** — curated Path-A is `[D]`-emit only, byte-for-byte. `[E]` never runs for a curated slug. |
| **derived (non-curated)**, Path B | full gate → print the §7-caveated verdict → **STOP** (structurally incapable of emit/download) | a download-eligible derived verdict now **continues into `[E]`**: CONTRACT-5 eligibility gate → `generate_from_profile` → `download_model` → `boot_derived` → `smoke_derived` → `emit_capture` (+ pt5 on override) |

`[E]` is wired into `run_pull()` (`scripts/lib/profiles/pull.py`) from
**exactly two** post-`[C1]` return points, and only when
`not is_curated and not dry_run`:

1. the satisfied `proceed` / `confirm→proceed` Path-B verdict, and
2. the `override-accepted` terminal that `--force-download` satisfied.

The `[E]` stage mutates **only** the additive `PullResult` fields
(`download_ok`, `boot_ok`, `smoke`, `capture_paths`) plus
`notices`/`diagnostics`. `ok` / `terminal` / `raw_verdict` / `emitted` /
`stratum` / `abort_reason` are left exactly as the `[C1]` decision set them,
so every pre-`[E]` `test-pull.sh` assertion is byte-unaffected. Every stage
function is injectable; all stage exceptions are caught and recorded
structurally — the `[E]` stage **never** raises out of `run_pull()`.

---

## 2. Trigger semantics (exactly as shipped)

This is the new user-facing contract. It *supersedes* the pre-`[E]`
"Path B never emits/downloads" / "`--force-download` is a NO-OP" rules —
but **only** for the non-curated derived download-eligible no-`--dry-run`
case. (`scripts/tests/test-pull.sh` g16–g22 is the truth table.)

| Run shape | Enters `[E]`? | Behaviour |
|---|---|---|
| non-curated + download-eligible `[C1]` terminal (`proceed` / satisfied `confirm→proceed`) + **no** `--dry-run` | **yes** | CONTRACT-5 gate → emit → download → boot → smoke → capture (pt1–4 + manifest). `is_override_accepted=False` (no pt5). |
| non-curated + `override-accepted` (the terminal `--force-download` satisfied) + **no** `--dry-run` | **yes** | same chain, **plus** pt5 override force-capture (`is_override_accepted=True`). `--force-download` activates `[E]` **only** for this terminal (it was a no-op pre-`[E]`). |
| `--dry-run` (any slug) | **no** | verdict-only, unchanged. `--dry-run` *never* enters `[E]`. A curated slug only reaches Path B via `--dry-run`, which this guard excludes. |
| curated slug (Path A) | **no** | curated Path-A `[D]`-emit, unchanged. Curated weights are local — there is no download stage. |
| non-curated, `confirm→proceed` **without** `--yes`; or low-confidence `wont-fit` **without** `--force-download` | **no** | honest non-pass (`[C1]` not satisfied); `[E]` is gated behind a *satisfied* download-eligible terminal. |

`override-accepted` is **not** a fit. The §5.3 rule is preserved verbatim in
the notices: a forced low-confidence download is a *calibration signal*,
never recorded as fit-validated. On the `override-accepted` `[E]` path the
pt5 force-capture (§4 below) carries the literal
`calibration_signal_not_validated:true`.

If the run is download-eligible non-curated but the local GPU topology
cannot be determined (no `nvidia-smi`, no `--hardware-gpus` override),
`[E]` does **not** run — it refuses honestly (`gpu-topology-undetermined`
notice) rather than fabricate a topology; the Pull-Gate verdict stands.

---

## 3. CONTRACT-5 — the derived-emittable eligibility gate

`generate_compose.derived_emittable(einput) -> (bool, reason|None)` is a
**pure** pre-`[E]` precondition run *before* any emit/download. It is **not**
a new gate state — it is a structured-refuse precondition, printed like
Pull-Gate's honest non-answer terminals. On refuse: a
`derived-runtime-unsupported:<reason>` notice + diagnostics, **no download,
no boot**.

The derived template is deliberately **patchless** (no Genesis / curated
overlays). Many curated profiles only work *because of* those overlays, so
inheriting an arbitrary `--profile-like` runtime shape into a patchless
compose would be a guaranteed-broken boot. CONTRACT-5 rejects, by
construction, anything that would need a patch/overlay/Genesis or that the
local hardware cannot run. A model is derived-emittable iff **all** hold
(checked in this short-circuit order; first failing clause names the
reason):

| Clause | Refuse reason token | Property checked |
|---|---|---|
| engine clean + Docker-image install | `derived-runtime-unsupported:engine-install-method` | the resolved engine YAML must have `required_genesis == false` **and** `vendored_overlays == []` **and** `install.method == "docker_image"`. Property-driven (not a hard engine list) so it is future-proof; an unresolvable engine YAML also fails here. (`vllm-stable`/`vllm-stable-next` are clean but `install.method: pip` — this clause keeps them out so the gate never emits an invalid `image: vllm==...`.) |
| no overlay features | `derived-runtime-unsupported:overlay-feature` | the `--profile-like` **COMPOSE_REGISTRY runtime entry** has `required_engine_features == []` (the registry runtime entry, *not* the engine YAML). |
| KV in the explicit safe set | `derived-runtime-unsupported:kv` | `kv_format ∈ {bf16, fp16, fp8_e5m2, fp8_e4m3}` (the literal `DERIVED_SAFE_KV` set). This is **not** "`kv_arg()` returns a defined result" — `kv_arg()` is *also* defined for `int8_per_token_head` / `turboquant_3bit_nc`, the values that must be rejected, so that predicate is self-contradictory. `kv_arg()` is emission-only (see §4 / §5). |
| no drafter | `derived-runtime-unsupported:drafter` | `runtime["drafter"] is None` — no DFlash / MTP / speculative / drafter (those need patches `[E]` does not include). |
| local GPUs can run this `tp` | `derived-runtime-unsupported:gpu-count` | `visible_gpu_count >= runtime["tp"]` **and** `len(selected_gpu_indices) == tp`. A clean no-drafter TP=2 shape on a 1-GPU rig is rejected here, *before* a multi-GB download. |
| quant resolvable | `derived-runtime-unsupported:unsupported-quant-for-derived[:autoround\|:missing-torch-dtype]` | `weight_format` must be resolvable by the CONTRACT-2 dispatch table (§4). `autoround` is an explicit reject (`:autoround`); a quantized repo with no usable compute dtype is `:missing-torch-dtype`. |

`derived_emittable()` is called twice on the success path — once by
`run_pull()` as the pre-`[E]` precondition, and once again as the first
thing `generate_from_profile()` does (it raises `Refuse(reason)`, mirroring
`[D]`'s structured-refusal style, on any reject) so the emitter can never be
called past a reject. The patchless `--kv-cache-dtype` / no-patch rows of
the derived template are therefore *safe by construction*.

`scripts/tests/test-generate-from-profile.sh` carries the negative matrix:
`vllm/gemma-int8-mtp` → `overlay-feature`; a TQ3-KV shape → `kv`; a drafter/MTP
shape → `drafter`; a clean TP=2 shape on a simulated 1-GPU `einput` →
`gpu-count` — each asserting **no** download and **no** emit, plus one
positive clean/no-drafter shape that proceeds.

---

## 4. The derived-vllm template + quant/dtype dispatch (CONTRACT-2)

`generate_compose.generate_from_profile(root, einput) -> (compose_text,
meta)` is an **additive** entry point. It emits a **separate** `derived-vllm`
base template — **not** the curated `compose_service_template`. (The curated
template hardcodes Qwen-specific constants — `--quantization auto_round`,
`--reasoning-parser qwen3`, the Qwen chat template, `--tool-call-parser
qwen3_coder` — so reusing it for an arbitrary derived model would emit a
broken boot.) The registry-key `generate(profile)` path is **byte-unchanged**
(`test-generate-compose.sh` byte-identical-proven, the P3 precedent).

Per-arg policy of the derived template (as shipped):

| Arg / field | Derived-template source |
|---|---|
| `image` | the **resolved** `engine.install.spec` from the clean engine YAML — a pinned image string, recorded in the capture manifest. **Not** a launch-time `${VLLM_NIGHTLY_SHA}` expansion (`[E]` boot does not go through `launch.sh`; an unresolved image risks a wrong-image boot). |
| `--model` | the **container** path `/models/club3090/pulls/<slug-sanitized>`. The host dir `<HF_HOME>/club3090/pulls/<slug-sanitized>/` is bind-mounted **`:ro`** to that container path via a `host:container:ro` `volumes:` entry. **Not** `${MODEL_DIR}`, not the HF hash-cache snapshot path. |
| `--served-model-name` | the slug sanitized: lowercased, every non-`[a-z0-9._-]` → `-`, collapse repeats, trim/strip, capped to vLLM's name limit. |
| `--quantization` / `--dtype` | the CONTRACT-2 dispatch table below. |
| `--tensor-parallel-size` / `--max-model-len` / `--gpu-memory-utilization` / `--max-num-seqs` / port | from `einput.runtime` (the `--profile-like` shape) — faithful inheritance. If the profile's `--max-num-seqs` is unsafe for the derived model, `[B]` kv-calc already prices it and the on-rig run catches OOM (pending E5). |
| `--kv-cache-dtype` | `kv_arg(kv_format)` is **emission-only** here, mapping the *already-eligible* value (CONTRACT-5 already gated it): `bf16`/`fp16` → flag omitted, `fp8_e4m3` → `fp8`, `fp8_e5m2` → `fp8_e5m2`. |
| `NVIDIA_VISIBLE_DEVICES` (env) | exactly `einput.selected_gpu_indices` — the same GPUs `[B]` priced. Never unset/`all` for derived. |
| `--trust-remote-code` | emitted **only if** the `[C0]` trc gate already resolved permitted (signalled on `einput.diagnostics["trc_permitted"] is True`); absent / not-True → never emitted. |
| `--chat-template`, `--reasoning-parser`, `--default-chat-template-kwargs`, `--enable-auto-tool-choice`, `--tool-call-parser` | **not emitted** — curated-specific. Derived uses vLLM defaults; derived tool/chat-template/reasoning support is explicitly OUT of scope. |
| patch insertion points | **patchless** — no Genesis / curated patches (CONTRACT-5 guarantees the runtime never needs them). |

**Quant/dtype dispatch table** (`_QUANT_DISPATCH`):

| `der.profile.weight_format` | `--quantization` | `--dtype` |
|---|---|---|
| `awq` | `awq` | from compute-dtype resolution |
| `gptq` | `gptq` | from compute-dtype resolution |
| `compressed-tensors` | `compressed-tensors` | from compute-dtype resolution |
| `fp8` | `fp8` | from compute-dtype resolution |
| `autoround` | **REJECT** `unsupported-quant-for-derived:autoround` | — |
| `float16` (pure dtype, no quant) | omitted | `float16` |
| `bfloat16` (pure dtype, no quant) | omitted | `bfloat16` |
| anything else | **REJECT** `unsupported-quant-for-derived` — never guess a `--quantization` | — |

`autoround` is rejected because the curated AutoRound composes explicitly
pass `--quantization auto_round`; "omit" is *not* demonstrably safe for an
arbitrary derived AutoRound repo and could silently mis-load. It is deferred
honestly rather than confidently-wrong.

**Compute-dtype resolution for the quantized rows** (`_resolve_compute_dtype`,
one defined order — not implementer's choice): (1) `der.profile["torch_dtype"]`
(an additive E1 deriver surface — the deriver already fetched `config.json`,
so this is an additive field, not a frozen-module rewrite) if it normalizes
to a real vLLM compute dtype; else (2) the deriver's **existing** bounded
safetensors header probe (`deriver.probe_safetensors_dtype`, never
reimplemented), **then normalize**: accept **only** a real compute dtype
(`float16`/`half`, `bfloat16`, `float32`). The probe returns a *storage*
dtype, and quantized safetensors expose int/storage dtypes (`I8`/`U8`/`I4`/…)
which are invalid as a vLLM `--dtype` — any such value is treated as "no
usable compute dtype"; else (3) **fail closed**
`unsupported-quant-for-derived:missing-torch-dtype`. Never default-guess or
pass a raw storage dtype. Pure-`float16`/`bfloat16` rows are unaffected —
their dtype *is* the weight_format. The live header-probe fetcher is wired
by E2's `downloader.set_probe_fetcher()` (E4 calls it inside `run_pull()`);
when no fetcher/inputs are present resolution fail-closes at step (3) rather
than guess.

The emitted derived compose is **patchless default-vLLM**: it carries zero
Genesis/curated patch insertions, a single absolute-host-path `:ro` volume,
the pinned engine image, and `NVIDIA_VISIBLE_DEVICES` pinned to the priced
GPUs.

---

## 5. The download contract (CONTRACT-3)

`downloader.download_model(einput, *, fetcher=None) -> DownloadResult`. One
shared allowlist, used identically by `[C2a]` sizing, the E2 download, and
E3 smoke — there is no parallel list that can drift:

```
download_set(api) = select_weight_files(api)            # *.safetensors (adapters excluded)
                  + the *.safetensors.index.json (if present, top-level)
                  + REQUIRED_METADATA siblings that exist # config/generation_config/
                                                          # tokenizer*/special_tokens_map/
                                                          # vocab.json/merges.txt
                  + every top-level *.jinja               # chat templates
```

`deriver.sized_download_gb(api)` sizes **exactly** `download_set(api)`, so
the `[C2a]` disk pre-gate prices precisely the set E2 fetches; a test
asserts fetched-set == sized-set. The fetcher is `huggingface_hub`
(`snapshot_download(repo_id, local_dir, allow_patterns=download_set)`) — the
established tool, **not** a hand-rolled fetcher and **not** the hash-oriented
snapshot cache. `allow_patterns` is mandatory and is literally the shared
`download_set` list. The real `huggingface_hub` import is lazy and the
fetcher is injectable so the test suite is hermetic (no live multi-GB pull
in CI).

**SHA verification — no-etag is a HARD fail** (the deliberate contrast):
every `*.safetensors` is verified by HF HEAD → `x-linked-etag` → local
SHA256 == etag. This reuses `setup.sh`'s *pattern*, but **`setup.sh:434/437`
prints `SKIP` and does not count a failure on a missing etag** — `[E]`
does the **opposite**: a missing/empty `x-linked-etag` is a hard
`failure:"no-etag"` abort. `[E]` never silently trusts an unverifiable
multi-GB weight (the `aria2c` corruption-incident lesson). Metadata files
are presence + size only.

`DownloadResult.failure` is one of `null | "no-etag" | "sha-mismatch" |
"gated-401" | "disk"`. This struct **is** the §6 capture-point-2 payload
shape; E2 only returns it (E3 emits the artifact).

**Atomic staging + cleanup:** download into
`<HF_HOME>/club3090/pulls/<slug-sanitized>/.incomplete/`; on full success
*and* all SHA verified, atomically move the verified tree onto
`<HF_HOME>/club3090/pulls/<slug-sanitized>/` (the CONTRACT-2 host `--model`
dir). On **any** failure — gated-401, no-etag, sha-mismatch, disk — the
`.incomplete` (and any sibling staged temp) tree is deleted: no corrupt or
partial residue, ever. `scripts/tests/test-pullgate-download.sh` carries the
no-etag fail-closed golden, the gated-401 mid-download case, and the
fetched-set == sized-set assertion.

---

## 6. The boot path (CONTRACT-4 boot half)

`booter.boot_derived(einput, compose_text, *, runner=None) -> BootResult`.
A derived compose boots **only** with the proven discipline:

* `docker compose --project-directory <project_dir> -f <compose> up -d` —
  the same `--project-directory` rule as `[D]` (relative-path resolution is
  project-dir-relative);
* the weights bind-mounted from the **HF_HOME host dir**
  `<HF_HOME>/club3090/pulls/<slug-sanitized>/` → container
  `/models/club3090/pulls/<slug-sanitized>` `:ro` — the exact mount
  `generate_from_profile()` already wrote into `compose_text`. This is
  **NOT** `MODEL_DIR=/mnt/models` (that is the curated Path-A convention; a
  derived model's weights live under HF_HOME per CONTRACT-2/3).

The boot is **always torn down** (`runner.down`) in a `finally`, success or
failure — no orphaned container/project state (the Pull-Gate on-rig harness
rule). `runner` is injectable; the default shells real `docker compose`,
tests pass a fixture runner (no Docker/GPU in CI). `BootResult.failure`
carries the *raw* container-died reason — `[E]` does **not** classify it
into a §6.1 class (that is `[F]`'s job). The real on-rig boot is **pending
on-rig validation (E5)**.

---

## 7. The §6 capture artifacts (CONTRACT-4 capture half)

`capture.emit_capture(...)` writes, schema **v1**, JSON, redacted, under:

```
<repo>/.pull-captures/<slug-sanitized>/<utc-ts>/
```

Four capture points + a top-level manifest:

| File | Point | Fields |
|---|---|---|
| `pt1-gate.json` | pre-download gate verdict | `schema, point, slug, confidence, raw_verdict, terminal, profile_like, hardware_sm` |
| `pt2-download.json` | download | `point, ok, files, bytes, sha_verified, failure` (the `DownloadResult` shape) |
| `pt3-boot.json` | boot | `point, ok, seconds, failure` (raw reason; **not** classified) |
| `pt4-smoke.json` | post-boot capability-aware smoke | `point, smoke_capability_set, results, partial` |
| `manifest.json` | §6.2/§6.3 manifest | see below |

The manifest carries the §6.2 success-consensus key inputs as **first-class
fields** (not only hashed inside the fingerprint — `[F]` must *reason over*
them; a hash is opaque): `model`, `quant_label`, `arch_family`,
`topology_class`, `engine_pin`/`engine_version` (the resolved CONTRACT-2
image spec), `kv_calc_version`, `selected_ctx`, `kv_format`,
`smoke_capability_set`, and `topology_summary_canonical` (the deterministic
serialization of **sorted `(gpu_name, vram_mib)` tuples**, §6.2 verbatim).
It also carries the `submission_fingerprint` (the §6.2 stage-2 hash, for
quick correlation) and the §6.3 dedup-key inputs as first-class fields
(`model_id`, `arch_family`, `kv_calc_version`, `engine_version`,
`topology_class`, `club3090_commit`). **`failure_class` is left `null`** by
`[E]` — the §6.1 classifier is `[F]`'s job; `[E]` emits the inputs, `[F]`
classifies/trusts/dedups/consumes.

**Capture-point 5 — override-accepted force-capture**
(`capture.emit_override_capture(...)`): a **separate** function, written into
the *same* capture directory, emitted **only** when
`einput.is_override_accepted` (the §5.3 trigger — the `override-accepted`
`[E]` path). Its fields: `point:"override_capture"`,
`predicted_b_breakdown` (the full `[B]` kv-calc GB breakdown that produced
the verdict), `actual:{boot_peak_mib, gpu_worker_reported_mib}` (or `null`
if the boot never reached allocation), `predicted_vs_actual_delta_mib` (or
`null`), `exit_error_summary` (or `null`), and the **mandatory literal**
`calibration_signal_not_validated: true` — §5.3: a forced low-confidence
download is a calibration *signal*, never recorded as fit-validated. The
artifact is emitted even when the boot never allocated (then `actual` is
`null` and `exit_error_summary` carries why). `emit_capture()` itself never
writes pt5 (`test-pullemit-capture.sh` asserts pt1–4 + manifest only).

**Redaction:** every artifact is written via `_redact_text()`, which reuses
the `report.sh --redact` convention (the identical sed expression set,
driven by the same `USER`/hostname/HF-token keys) and hardens it further —
any absolute internal mount path (`/opt/*`, `/mnt/*`, `/data/*`) is
additionally scrubbed to `<PATH>` (defence-in-depth, the "don't leak
internal paths in public artifacts" stack rule). The capture schema carries
only slugs / verdicts / relative filenames; it must never carry an absolute
internal host path.

---

## 8. The derived capability-smoke floor (CONTRACT-4)

`capture.smoke_derived(einput, endpoint, *, client=None) -> SmokeResult`. A
derived generic profile declares no capabilities, so the conservative floor
is:

* **plain-chat ALWAYS + streaming ALWAYS** — cheap, and catches the #145
  class (a model that boots and answers plain chat while streaming/tools are
  silently dead).
* `tool-call` / `reasoning-streaming` / `structured-output` / `vision` /
  `long-context` are smoked **only if** `der`'s `config.json` *positively*
  declares them (read-only positive signals; absence is never inferred as a
  capability). A generic dense derived model declares none → **floor only**.
* every optional capability not smoked → recorded `"unsmoked"`, and
  `partial = any(result == "unsmoked")`. Per §6.2 an anchor with un-smoked
  capabilities is `partial` and cannot graduate to Tier-1 for those
  capabilities.

`client` is injectable (fixture client in CI; the live server is the on-rig
E5 leg, pending). `test-pullemit-capture.sh` asserts a derived model that
declares no tools records `tool-call:"unsmoked", partial:true`.

---

## 9. Explicitly OUT of `[E]` scope

| Out | Where it lives |
|---|---|
| The `[F]` Loop: §6.1 2-tier failure classifier, §6.2 inbound-trust pipeline, §6.3 dedup, consensus, promotion. `[E]` **emits** the §6 artifacts; `[F]` **consumes** them. `[E]` does not classify (`failure_class` stays null), trust, dedup, or promote. | **Loop phase** |
| Derived tool-call / chat-template / reasoning / curated-or-Genesis-patch support (the CONTRACT-2 deferral — derived is patchless default-vLLM). | a future scoped phase |
| GGUF / `.bin` backends, multi-quant auto-pick, whichllm hardware slice. | **v0.8.1** |
| Curated Path-A behaviour, the §4.1 strata / `[C0]` / `[C1]` state sets. | **frozen** — `[E]` is strictly post-`[C1]`, consumed not modified |
| UX doc tracks (§7). | post-Pull-Gate |

---

## 10. Running the executable specs

From the repo root:

```
bash scripts/tests/test-generate-from-profile.sh   # CONTRACT-2 / CONTRACT-5
bash scripts/tests/test-pullgate-download.sh        # CONTRACT-3
bash scripts/tests/test-pullemit-capture.sh         # CONTRACT-4 (pt1-5 + manifest)
bash scripts/tests/test-pull.sh                     # the [E] continuations + g16-g22 trigger matrix
python3 tools/kv-calc.py --calibration              # must stay Overall: 22/22 (100%)
```

These are the authoritative spec; this document is the explanatory
companion. Where they disagree, the test wins.
