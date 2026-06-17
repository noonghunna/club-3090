"""Compose generator core (v0.8.0 STEP 3 — club-3090 #141 / PR #147).

Implements the brief's "STEP 3 — `scripts/generate-compose.sh`" decision
sequence (steps 1-10). The shell wrapper (`scripts/generate-compose.sh`)
only parses argv and shells into :func:`main`.

Mission (locked decision #2): minimal-reproduction + flag, NEVER repair.
The generator NEVER rewrites the engine image, NEVER repairs a failed
patch, NEVER wires a patch whose drift-guard failed, and NEVER blind-
passes a governed security flag (`--trust-remote-code`).

Emit model (brief §STEP 3 step 9 / Architecture)
-------------------------------------------------
The capture unit is the *whole shipped service definition* — recorded per
profile in ``profile_runtime.yml`` as ``compose_service_template`` with the
file at ``source`` being the verbatim ground truth. The generator:

* loads that shipped service body verbatim (it synthesizes nothing —
  param-slots remain the shipped ``${VAR:-default}`` expressions, the
  ``image:`` line is a captured *constant* passed through verbatim
  (correction #2), every other constant reproduces byte-for-byte);
* re-derives the **two named insertion points** (``volumes:`` overlay /
  sidecar mounts, ``entrypoint:`` sidecar invoke lines) from the
  compose-keyed patch selection: a selected+wired patch keeps its mount /
  invoke lines; a selected-but-undelivered (delivery-gap) or omitted
  (capability-degraded) patch has its mount / invoke lines stripped;
* emits ``--trust-remote-code`` ONLY if the trc governed-slot gate
  resolves evidence-cited-permitted — which, combined with step 5's
  refusal of ``{true, unverified}``, means an in-scope generated compose
  NEVER blind-passes it (correction #1);
* prepends a 3-category provenance header (selected+wired /
  selected+undelivered+reason / excluded). The header lives ABOVE the
  top-level ``services:`` key so STEP-2 ``service_body()`` discards it —
  header text can never create a false-positive reachability hit
  (correction #4).

For a golden profile the compose-keyed selection reproduces exactly the
patch set the maintainer shipped, so the re-derived insertion points are
byte-identical to the shipped file and the only service-body delta is
nil — the golden-parity invariant the STEP-4 test asserts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path so `scripts.lib.profiles.*` imports resolve whether
# this module is run as a script or imported (mirrors the test's import site).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse — never reimplement — the STEP-2 patch-attribution core.
from scripts.lib.profiles import patch_attribution as pa  # noqa: E402


# --------------------------------------------------------------------------
# Exit codes (stable contract for the STEP-4 test).
# --------------------------------------------------------------------------
EXIT_OK = 0
EXIT_REFUSE = 2          # clean scope / validation / foundational refusal
EXIT_DEGRADED_NOACK = 3  # capability-scoped guard fail, --accept-degraded absent
EXIT_AMBIGUOUS = 4       # convenience-tuple matched >0 profiles (not authoritative)
EXIT_USAGE = 64          # argv / lookup misuse


class Refuse(Exception):
    """Clean refusal — printed to stderr, mapped to an exit code."""

    def __init__(self, message: str, code: int = EXIT_REFUSE):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# Data loaders.
# --------------------------------------------------------------------------
def _load_yaml(root: Path, rel: str) -> dict:
    return pa.load(root / rel)


def load_runtime(root: Path) -> dict:
    return _load_yaml(root, "scripts/lib/profiles/profile_runtime.yml")


def load_patches(root: Path) -> list[dict]:
    return _load_yaml(root, "scripts/lib/profiles/patches.yml").get("patches", [])


def load_arches(root: Path) -> list[dict]:
    return _load_yaml(root, "scripts/lib/profiles/arch_patches.yml").get("arches", [])


def load_engine(root: Path, engine_id: str) -> dict:
    path = root / f"scripts/lib/profiles/engines/{engine_id}.yml"
    if not path.exists():
        raise Refuse(f"engine profile not found: {engine_id}")
    return pa.load(path)


def load_drafter(root: Path, drafter_id: str) -> dict:
    path = root / f"scripts/lib/profiles/drafters/{drafter_id}.yml"
    if not path.exists():
        raise Refuse(f"drafter profile not found: {drafter_id}")
    return pa.load(path)


def get_registry(root: Path) -> dict:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402

    return COMPOSE_REGISTRY


# --------------------------------------------------------------------------
# kv_format -> --kv-cache-dtype arg (non-Genesis map, brief §STEP 3 step 9).
# --------------------------------------------------------------------------
def kv_arg(kv_format: str):
    """Return the documented --kv-cache-dtype value (or None for 'no arg').

    Unit-tested by the STEP-4 test. Non-Genesis only: this generator never
    emits Genesis-equipped composes, but the map is exhaustive over the
    formats the in-scope engines list in supported_kv_formats.
    """
    table = {
        "bf16": None,
        "fp16": None,
        "fp8_e5m2": "fp8_e5m2",
        "fp8_e4m3": "fp8",
        "int8_per_token_head": "auto+PTH",
        "q4_0": "q4_0",
        "k8v4": "k8v4",
        "turboquant_3bit_nc": "turboquant_3bit_nc",  # Genesis-only; never emitted in-scope
    }
    if kv_format not in table:
        raise Refuse(f"unmapped kv_format: {kv_format}")
    return table[kv_format]


# --------------------------------------------------------------------------
# Arch resolution via model_slugs / arch_model_xref.
# --------------------------------------------------------------------------
def resolve_arch(root: Path, runtime: dict, arches: list[dict], model: str) -> tuple[str, dict]:
    """Map E.model -> (arch_string, arch_row) via arch_model_xref.model_slugs.

    arch_patches.yml is a strict closed key-set (RED-LINE: not editable);
    the brief stores the model_slugs fold-in in profile_runtime.yml keyed
    by the same arch string, so we join the two here.
    """
    xref = runtime.get("arch_model_xref") or {}
    matched_arch = None
    for arch_name, meta in xref.items():
        if model in (meta.get("model_slugs") or []):
            matched_arch = arch_name
            break
    if matched_arch is None:
        raise Refuse(
            f"no arch in arch_model_xref maps model_slug {model!r}; "
            f"out of the patch matrix -> refuse"
        )
    arch_row = next((r for r in arches if r.get("arch") == matched_arch), None)
    if arch_row is None:
        raise Refuse(f"arch {matched_arch!r} has no arch_patches.yml row -> refuse")
    return matched_arch, arch_row


def resolve_arch_from_config(runtime: dict, arches: list[dict], config_arch: str):
    """Map a config.json architectures[0] string -> (canonical_arch, arch_row)
    via arch_model_xref. Matches either the canonical (inner) arch key itself or
    any `config_architectures` alias (the OUTER multimodal wrapper class a weights
    repo reports — e.g. Qwen3_5ForConditionalGeneration -> Qwen3NextForCausalLM).
    Used by the [C0] pull gate so an uncurated derive (abliterated / fine-tune) of
    an already-supported arch is not false-aborted with "no arch_patches matrix row".
    arch_patches.yml stays a closed key-set; the alias lives in the editable xref.
    Returns (None, None) when the config arch maps to no row."""
    if not config_arch:
        return None, None
    xref = runtime.get("arch_model_xref") or {}
    for canonical, meta in xref.items():
        aliases = (meta or {}).get("config_architectures") or []
        if config_arch == canonical or config_arch in aliases:
            row = next((r for r in arches if r.get("arch") == canonical), None)
            if row is not None:
                return canonical, row
    return None, None


# --------------------------------------------------------------------------
# Step 3 — engine-pin validation (validation ONLY; never rewrites image).
# --------------------------------------------------------------------------
def scope_gate(engine_id: str, engine: dict, runtime: dict, profile: str) -> dict:
    """The #141 in-scope predicate (extracted verbatim from :func:`generate`
    step 2 — behaviour/messages/codes byte-identical).

    Raises :class:`Refuse` with the SAME message + code as the inlined
    sequence did when the profile is out of scope; returns the resolved
    ``profile_runtime`` block on success. This is pure (data in, data out /
    Refuse) so the Pull-Gate stratum-2 path can REUSE this exact predicate
    read-only instead of reimplementing it (locked design `[D]` reuse).
    """
    if engine.get("type") != "vllm":
        raise Refuse(
            f"engine {engine_id} type={engine.get('type')!r} != vllm; the #141 "
            f"generator is non-Genesis vLLM only -> refuse (out of scope)"
        )
    prof_rt = (runtime.get("profiles") or {}).get(profile)
    if prof_rt is None:
        raise Refuse(
            f"profile {profile!r} has no profile_runtime.yml capture "
            f"(only in-scope vLLM profiles are captured) -> refuse"
        )
    if prof_rt.get("genesis_equipped") is True:
        raise Refuse(
            f"profile {profile} is genesis_equipped:true "
            f"({prof_rt.get('genesis_equipped_evidence')}); Genesis-flag "
            f"generation is permanently out of scope -> refuse"
        )
    return prof_rt


def validate_engine_pin(engine_id: str, engine: dict, arch_row: dict) -> str:
    """Confirm <engine>@<sha-from-install.spec> matches a loads:true pin.

    Returns the resolved ``<engine_id>@<sha>`` string for the header. This
    is pure validation — the emitted compose keeps the shipped
    ``image: ${VLLM_IMAGE:-...}`` expression untouched (correction #2).
    """
    spec = (engine.get("install") or {}).get("spec", "")
    # spec looks like 'vllm/vllm-openai:nightly-<sha>' or 'vllm-pip-baseline@0.20.2'
    sha = ""
    if ":" in spec and "nightly-" in spec:
        sha = spec.split("nightly-", 1)[1].strip()
    elif "@" in spec:
        sha = spec.split("@", 1)[1].strip()
    elif ":" in spec:
        sha = spec.split(":", 1)[1].strip()
    resolved = f"{engine_id}@{sha}" if sha else engine_id

    for pin in arch_row.get("engine_pin") or []:
        pin_str = pin.get("pin", "")
        pin_engine, _, pin_sha = pin_str.partition("@")
        if pin_engine != engine_id:
            continue
        # Accept <id> (no sha) OR <id>@<sha>; require the matching pin loads.
        if (not sha or not pin_sha or pin_sha == sha) and pin.get("loads") is True:
            return resolved
        if (not sha or not pin_sha or pin_sha == sha) and pin.get("loads") is not True:
            reason = pin.get("reason", "pin not marked loads:true")
            raise Refuse(
                f"engine pin {resolved} maps to arch {arch_row['arch']} pin "
                f"{pin_str!r} with loads != true ({reason}) -> refuse"
            )
    raise Refuse(
        f"engine pin {resolved} has no loads:true match on arch "
        f"{arch_row['arch']} engine_pin[] -> refuse"
    )


# --------------------------------------------------------------------------
# Step 6-8 — patch selection / delivery-gap / drift-guard.
# --------------------------------------------------------------------------
def select_patches(patches: list[dict], profile: str) -> list[dict]:
    """Compose-keyed selection ONLY: include P iff profile in a
    P.load_bearing_when[].composes list (brief §STEP 3 step 6)."""
    selected = []
    for p in patches:
        for lb in p.get("load_bearing_when") or []:
            if profile in (lb.get("composes") or []):
                selected.append(p)
                break
    return selected


def classify_patch(root: Path, patch: dict, profile: str) -> dict:
    """Return the per-patch decision record (brief steps 7-8).

    state ∈ {wired, undelivered, omitted-degraded, refuse-foundational}.
    The generator NEVER repairs and NEVER wires a failed-guard patch.
    """
    pid = patch["id"]

    # Step 7 — delivery-gap BEFORE drift-guard. A declared gap covering this
    # profile -> selected-but-undelivered: omit wiring, header WARNING,
    # SKIP the drift-guard entirely.
    if pa.gap_declared(patch, profile):
        gap_reason = "delivery gap declared for this compose"
        for gap in patch.get("delivery_gaps") or []:
            if profile in set(gap.get("composes") or []):
                gap_reason = gap.get("issue", gap_reason)
                break
        return {"id": pid, "state": "undelivered", "reason": gap_reason, "patch": patch}

    # Step 8 — drift-guard on a will-be-wired patch (§4.1 graded).
    guard = patch.get("drift_guard") or {}
    on_fail = guard.get("on_fail")
    foundational = bool(patch.get("foundational"))
    capability = patch.get("capability")

    # The drift-guard is a runtime import-and-boot / behavioral probe that
    # cannot be executed at generation time (no engine container here). The
    # generator's contract: wire the patch (the shipped, maintainer-tested
    # state is "applies cleanly" = drift-guard-tested, locked decision #4),
    # and record the guard so the boot leg / operator can re-run it. A guard
    # whose grade is structurally a hard-refuse on a foundational patch is
    # surfaced; capability-scoped guards degrade only on an *observed* fail
    # (injected by the test harness via FORCE_GUARD_FAIL), never speculatively.
    forced_fail = _forced_guard_fail(pid)
    if forced_fail:
        if foundational or on_fail == "hard-refuse":
            return {
                "id": pid,
                "state": "refuse-foundational",
                "reason": f"foundational drift-guard failed ({capability}); "
                f"generator never repairs -> hard-refuse",
                "patch": patch,
            }
        # capability-scoped fail -> OMIT (never wire a failed patch) + DEGRADED
        return {
            "id": pid,
            "state": "omitted-degraded",
            "reason": f"capability-scoped drift-guard failed ({capability}); "
            f"omitted, compose is DEGRADED",
            "patch": patch,
        }

    return {"id": pid, "state": "wired", "reason": guard.get("check", ""), "patch": patch}


def _forced_guard_fail(pid: str) -> bool:
    """Test-only hook. The STEP-4 test exercises the failed-guard paths by
    setting CLUB3090_FORCE_GUARD_FAIL to a comma-list of patch ids; there
    is no other way to deterministically drive a runtime probe failure from
    a unit test. Empty / unset in all real runs."""
    import os

    forced = os.environ.get("CLUB3090_FORCE_GUARD_FAIL", "")
    return pid in {x.strip() for x in forced.split(",") if x.strip()}


# --------------------------------------------------------------------------
# Step 9 — emit from the captured compose_service_template.
# --------------------------------------------------------------------------
def _patch_wiring_markers(patch: dict) -> list[str]:
    """All concrete body strings (mount targets + invoke command) the patch
    contributes at the two insertion points. Derived from delivery_spec —
    the same source STEP-2 reaches() validates against, so emit and
    reachability stay in lock-step.

    Shipped composes mount overlays via the compose-relative path
    (``../../patches/<patch-dir>/...``) while delivery_spec records the
    repo-relative path (``models/.../patches/<patch-dir>``). The shared,
    stable token is the patch directory basename plus its container-side
    ``dest``; both are included so a strip of an undelivered/omitted patch
    matches the shipped relative mount line."""
    spec = patch.get("delivery_spec") or {}
    markers: list[str] = []

    def _dir_token(p: str) -> str:
        # ".../patches/<name>" or ".../patches/<name>/file.py" -> "patches/<name>/"
        parts = Path(p).as_posix().split("/")
        if "patches" in parts:
            i = parts.index("patches")
            if i + 1 < len(parts):
                return "patches/" + parts[i + 1] + "/"
        return p

    for of in spec.get("overlay_files") or []:
        if of.get("dest"):
            markers.append(of["dest"])
        if of.get("src"):
            markers.append(_dir_token(of["src"]))
    if spec.get("overlay_dir"):
        # The compose-relative src token (`patches/<name>/`) is unique per
        # patch; dest_root is shared across patches so it is NOT a marker.
        markers.append(_dir_token(spec["overlay_dir"]))
    if spec.get("mounted_at"):
        markers.append(spec["mounted_at"])
    if spec.get("sidecar"):
        markers.append(Path(spec["sidecar"]).name)
        markers.append(_dir_token(spec["sidecar"]))
    if spec.get("script"):
        markers.append(_dir_token(spec["script"]))
        markers.append(Path(spec["script"]).name)
    # chat_template: the vendored `.jinja` is mounted via the
    # compose-relative `patches/<name>/...jinja` path; `mounted_at` (added
    # above) is the container side. For the explicit `--chat-template`
    # wiring style add both serving tokens so a strip of an omitted/
    # undelivered chat_template patch removes the --chat-template arg too
    # (emit stays in lock-step with reaches()).
    if patch.get("delivery_mechanism") == "chat_template":
        if spec.get("jinja"):
            markers.append(_dir_token(spec["jinja"]))
            markers.append(Path(spec["jinja"]).name)
        if "--chat-template" in (spec.get("invoke") or "") and spec.get("mounted_at"):
            markers.append("--chat-template")
    inv = spec.get("invoke")
    if inv and "/" in inv and not inv.endswith((".", "serving", "import")):
        markers.append(inv)
    return [m for m in markers if m]


def _strip_unwired_lines(body_text: str, drop_markers: list[str]) -> str:
    """Remove only the volume-mount / entrypoint-invoke lines that belong to
    a selected-but-undelivered or omitted-degraded patch. Every other line
    (constants, the image expression, param-slots, base env/entrypoint) is
    reproduced verbatim — the generator synthesizes nothing."""
    if not drop_markers:
        return body_text
    kept = []
    for line in body_text.splitlines():
        if any(m in line for m in drop_markers):
            continue
        kept.append(line)
    return "\n".join(kept) + ("\n" if body_text.endswith("\n") else "")


def render_header(
    profile: str,
    resolved_pin: str,
    wired: list[dict],
    undelivered: list[dict],
    excluded: list[str],
    degraded: bool,
) -> str:
    lines = [
        "# ===========================================================================",
        f"# GENERATED by scripts/generate-compose.sh — club-3090 #141 (v0.8.0).",
        f"#   profile: {profile}",
        f"#   engine-pin (validated, image NOT rewritten): {resolved_pin}",
        "#   Mission: minimal-reproduction + flag — this file is NEVER hand-repaired.",
        "#",
        "#   [1] selected + WIRED (drift-guard tested; wired at volumes/entrypoint):",
    ]
    if wired:
        for w in wired:
            lines.append(f"#       - {w['id']}")
    else:
        lines.append("#       - (none)")
    lines.append("#   [2] selected + UNDELIVERED (gap / failed-guard — wiring omitted):")
    if undelivered:
        for u in undelivered:
            lines.append(f"#       - {u['id']}: {u['reason']}")
    else:
        lines.append("#       - (none)")
    lines.append("#   [3] EXCLUDED (not load-bearing for this compose):")
    if excluded:
        for e in excluded:
            lines.append(f"#       - {e}")
    else:
        lines.append("#       - (none)")
    if degraded:
        lines.append("#")
        lines.append(
            "#   WARNING: DEGRADED — a capability-scoped patch was omitted after a "
            "failed drift-guard."
        )
        lines.append("#   Re-run required --accept-degraded to acknowledge.")
    lines.append("#   NOTE: --trust-remote-code is a GOVERNED slot (locked §88); it is")
    lines.append("#   NOT emitted for any in-scope profile (arch trc gate = false).")
    lines.append("# ===========================================================================")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------
def generate(
    root: Path,
    profile: str,
    accept_degraded: bool = False,
) -> tuple[str, dict]:
    """Run the full STEP-3 decision sequence. Returns (compose_text, meta).

    Raises Refuse on any scope / validation / foundational / degraded-no-ack
    refusal (caller maps .code to the process exit code).
    """
    registry = get_registry(root)
    runtime = load_runtime(root)
    patches = load_patches(root)
    arches = load_arches(root)

    # Step 1 — resolve --profile -> E.
    if profile not in registry:
        raise Refuse(f"unknown profile {profile!r} (not in COMPOSE_REGISTRY)", EXIT_USAGE)
    E = registry[profile]
    engine_id = E["engine"]
    engine = load_engine(root, engine_id)

    # Step 2 — SCOPE GATES FIRST (before any capture lookup).
    prof_rt = scope_gate(engine_id, engine, runtime, profile)

    # Step 4 (model->arch) feeds step 3's pin validation.
    arch_name, arch_row = resolve_arch(root, runtime, arches, E["model"])

    # Step 3 — engine-pin validation (image NEVER rewritten).
    resolved_pin = validate_engine_pin(engine_id, engine, arch_row)

    # Step 4 — valid_tp / supported_kv_formats.
    tp_divisors = (arch_row.get("valid_tp") or {}).get("tp_divisors") or []
    if E["tp"] not in tp_divisors:
        raise Refuse(
            f"tp={E['tp']} not in arch {arch_name} valid_tp.tp_divisors "
            f"{tp_divisors} -> refuse"
        )
    if E["kv_format"] not in (engine.get("supported_kv_formats") or []):
        raise Refuse(
            f"kv_format {E['kv_format']!r} not in engine {engine_id} "
            f"supported_kv_formats {engine.get('supported_kv_formats')} -> refuse"
        )

    # Step 5 — trc deferral. {true, unverified} -> refuse. Combined with the
    # governed-slot rule this guarantees an in-scope compose never blind-
    # passes --trust-remote-code.
    trc = arch_row.get("requires_trust_remote_code")
    if trc in {"true", "unverified"}:
        raise Refuse(
            f"arch {arch_name} requires_trust_remote_code={trc!r} "
            f"(evidence: {arch_row.get('requires_trust_remote_code_evidence')}) "
            f"-> security refusal (deferred; pull-gate handles trc ack)"
        )
    # trc == 'false' here. --trust-remote-code is a GOVERNED slot: it is
    # NEVER emitted for an in-scope profile (correction #1).
    trc_emit = False

    # kv_arg is unit-tested even when its result is unused (bf16/fp16 -> no arg).
    _ = kv_arg(E["kv_format"])

    # Step 6 — compose-keyed patch selection.
    selected = select_patches(patches, profile)
    selected_ids = {p["id"] for p in selected}
    excluded = sorted(p["id"] for p in patches if p["id"] not in selected_ids)

    # Steps 7-8 — classify each selected patch.
    decisions = [classify_patch(root, p, profile) for p in selected]
    foundational_refusals = [d for d in decisions if d["state"] == "refuse-foundational"]
    if foundational_refusals:
        d = foundational_refusals[0]
        raise Refuse(f"{d['id']}: {d['reason']}")

    wired = [d for d in decisions if d["state"] == "wired"]
    undelivered = [d for d in decisions if d["state"] == "undelivered"]
    degraded_omitted = [d for d in decisions if d["state"] == "omitted-degraded"]
    degraded = bool(degraded_omitted)

    if degraded and not accept_degraded:
        names = ", ".join(d["id"] for d in degraded_omitted)
        raise Refuse(
            f"DEGRADED: capability-scoped patch(es) [{names}] omitted after a "
            f"failed drift-guard; re-run with --accept-degraded to proceed",
            EXIT_DEGRADED_NOACK,
        )

    # Step 9 — emit from compose_service_template. The shipped service body
    # at `source` is the captured unit: param-slots stay as the shipped
    # ${VAR:-default} expressions, constants (incl. the image expression)
    # reproduce verbatim, and we synthesize nothing. The ONLY transformation
    # is at the two insertion points: strip mount/invoke lines of patches
    # that are NOT selected+wired (undelivered or degraded-omitted).
    src_rel = (prof_rt["compose_service_template"] or {}).get("source") or E["compose_path"]
    src_path = root / src_rel
    raw = src_path.read_text(encoding="utf-8")
    body = pa.service_body(raw)  # drops the shipped at-a-glance banner; comment-free

    drop_markers: list[str] = []
    for d in undelivered + degraded_omitted:
        drop_markers.extend(_patch_wiring_markers(d["patch"]))
    body = _strip_unwired_lines(body, drop_markers)

    # Defensive scope assertion: the governed trc slot must never reach the
    # emitted body for an in-scope profile. trc_emit is provably False above.
    if not trc_emit:
        # The shipped body for in-scope profiles may carry --trust-remote-code
        # as a captured governed token; strip it (it is emitted ONLY when the
        # gate permits, which never happens in-scope per correction #1).
        body = _strip_trc(body)

    header = render_header(
        profile, resolved_pin,
        wired, undelivered + degraded_omitted, excluded, degraded,
    )
    out_text = header + body
    if not out_text.endswith("\n"):
        out_text += "\n"

    meta = {
        "profile": profile,
        "engine_pin": resolved_pin,
        "wired": [d["id"] for d in wired],
        "undelivered": [d["id"] for d in undelivered],
        "degraded_omitted": [d["id"] for d in degraded_omitted],
        "excluded": excluded,
        "degraded": degraded,
        "trc_emitted": trc_emit,
        "source": src_rel,
    }
    return out_text, meta


# ===========================================================================
# v0.8.0 [E] STEP E1 — derived (non-curated) emission.
#
# ADDITIVE ONLY. Everything ABOVE this banner — incl. generate() and its
# helpers — is byte-unchanged from the pre-E1 [D] registry-key generator
# (test-generate-compose.sh proves the registry path is byte-identical, P3
# precedent). generate_from_profile() / derived_emittable() are a SEPARATE
# entry point with a SEPARATE template; they NEVER call generate() and
# generate() never calls them, so the two paths are structurally isolated.
#
# CONTRACT-2 derived-vllm per-arg policy (docstring-as-spec, DeepSeek L1):
#
#   image                 -> resolved engine install.spec (NOT a bare
#                            ${VLLM_NIGHTLY_SHA}); recorded in meta.
#   --model               -> CONTAINER path /models/club3090/pulls/<slug-san>
#                            (+ a `<HF_HOME>/club3090/pulls/<slug-san>:
#                            <container>:ro` volume; host path from
#                            einput.hf_home — E1 emits the mount only, does
#                            not create/download it).
#   --served-model-name   -> slug sanitized (lower; non-[a-z0-9._-]->'-';
#                            collapse repeats; trim to vLLM 256 cap).
#   --quantization/--dtype-> CONTRACT-2 quant/dtype dispatch table (below).
#   --tensor-parallel-size/--max-model-len/--gpu-memory-utilization/
#     --max-num-seqs/port -> einput.runtime (the --profile-like shape).
#   --kv-cache-dtype      -> kv_arg() EMISSION-ONLY on the already-CONTRACT-5-
#                            eligible kv_format (bf16/fp16 -> flag omitted;
#                            fp8_e4m3 -> 'fp8'; fp8_e5m2 -> 'fp8_e5m2').
#   NVIDIA_VISIBLE_DEVICES-> exactly einput.selected_gpu_indices.
#   --trust-remote-code   -> ONLY if einput indicates the [C0] trc gate
#                            already resolved permitted.
#   NOT emitted           -> --chat-template / --reasoning-parser /
#                            --default-chat-template-kwargs /
#                            --enable-auto-tool-choice / --tool-call-parser
#                            (derived = vLLM defaults; deferred).
#   patches               -> PATCHLESS (no volumes/entrypoint patch insert;
#                            CONTRACT-5 guarantees none are needed).
#
# Every CONTRACT-5 / dispatch-table reject raises Refuse (mirrors [D]'s
# structured-refusal style) — NEVER a broken / default-guessed compose.
# ===========================================================================

# CONTRACT-5 explicit derived-safe KV set (NOT "kv_arg() is defined": kv_arg
# is ALSO defined for int8_per_token_head / turboquant_3bit_nc, the values
# CONTRACT-5 rejects, so that predicate is self-contradictory).
DERIVED_SAFE_KV = frozenset({"bf16", "fp16", "fp8_e5m2", "fp8_e4m3"})

# CONTRACT-2 quant/dtype dispatch table. weight_format -> (--quantization
# value | None=omit | "REJECT", needs_dtype). autoround is an explicit
# CONTRACT-5 reject (curated AutoRound composes pass --quantization
# auto_round; "omit" is NOT demonstrably safe for an arbitrary derived
# AutoRound repo — deferred, honesty-over-confidently-wrong).
_QUANT_DISPATCH = {
    "awq": ("awq", True),
    "gptq": ("gptq", True),
    "compressed-tensors": ("compressed-tensors", True),
    "fp8": ("fp8", True),
    "autoround": ("REJECT:autoround", True),
    # pure dtype, no quant: --quantization omitted, --dtype = the weight_format
    "float16": (None, False),
    "bfloat16": (None, False),
}

# torch_dtype-string -> vLLM --dtype compute value. CONTRACT-2 v8: accept
# ONLY a real compute dtype. Any int/storage dtype (I8/U8/I4/...) from the
# header probe is NOT a usable --dtype → fail-closed.
_COMPUTE_DTYPE = {
    "FLOAT16": "float16",
    "HALF": "float16",
    "F16": "float16",
    "FP16": "float16",
    "BFLOAT16": "bfloat16",
    "BF16": "bfloat16",
    "FLOAT32": "float32",
    "F32": "float32",
    "FP32": "float32",
}


def _sanitize_slug(slug: str, cap: int = 256) -> str:
    """slug -> vLLM-safe name: lowercase, every non-[a-z0-9._-] -> '-',
    collapse runs of '-', strip leading/trailing '-', trim to `cap`
    (vLLM served-model-name cap; DeepSeek-r2 L2)."""
    out = []
    for ch in slug.strip().lower():
        out.append(ch if (ch.isascii() and (ch.isalnum() or ch in "._-")) else "-")
    s = "".join(out)
    while "--" in s:
        s = s.replace("--", "-")
    s = s.strip("-")
    return s[:cap]


def _engine_install(engine: dict) -> dict:
    return engine.get("install") or {}


def _resolve_compute_dtype(einput) -> tuple[str, str | None]:
    """CONTRACT-2 v8 --dtype resolution order for a quantized derived repo.

    Returns (dtype | "", reject_token | None). dtype == "" with a token means
    fail-closed `unsupported-quant-for-derived:<token>`.

    (1) der.profile["torch_dtype"] (the additive E1 deriver surface) — if a
        recognized compute dtype.
    (2) the deriver's EXISTING bounded safetensors header probe THEN
        normalize: accept ONLY {float16/half, bfloat16, float32}; any
        int/storage dtype (I8/U8/I4/...) is NOT a usable --dtype.
    (3) fail-closed `missing-torch-dtype`.

    Header-probe wiring: E1 calls the deriver's existing
    `probe_safetensors_dtype()` (never reimplemented) ONLY when it is cleanly
    callable — i.e. einput.der carries `slug` + `profile["selected_weight_files"]`
    and a usable fetcher in `einput.diagnostics["fetcher"]`. If those inputs
    are not present on the EInput, step (2) is skipped and resolution
    fail-closes at step (3); full live header-probe wiring is an E2 follow
    (E2 owns the download/fetcher plumbing). This is the brief's explicitly
    permitted "torch_dtype-or-fail + note header-probe wiring as an E2 follow".
    """
    der = einput.der
    prof = getattr(der, "profile", None) or {}

    # (1) config.json torch_dtype (additive E1 deriver surface).
    td = prof.get("torch_dtype")
    if isinstance(td, str) and td.strip():
        norm = _COMPUTE_DTYPE.get(td.strip().upper())
        if norm is not None:
            return norm, None

    # (2) bounded header probe via the deriver's EXISTING fn (not reimplemented).
    fetcher = (einput.diagnostics or {}).get("fetcher")
    sel = prof.get("selected_weight_files") or []
    slug = getattr(der, "slug", None)
    if fetcher is not None and sel and slug:
        from scripts.lib.profiles import deriver as _D  # noqa: E402

        probed = _D.probe_safetensors_dtype(
            slug, sorted(sel)[0], fetcher,
            (einput.diagnostics or {}).get("hf_token"),
        )
        if isinstance(probed, str) and probed.strip():
            norm = _COMPUTE_DTYPE.get(probed.strip().upper())
            if norm is not None:
                return norm, None
        # any int/storage dtype (or unprobeable) -> NOT a usable --dtype.

    # (3) fail-closed.
    return "", "missing-torch-dtype"


def _trc_permitted(einput) -> bool:
    """--trust-remote-code is emitted ONLY if einput indicates the [C0] trc
    gate already resolved permitted. The signal is carried on
    einput.diagnostics["trc_permitted"] is True (E4 wires it from the [C0]
    resolution); absent / not-True -> NEVER emitted."""
    return (einput.diagnostics or {}).get("trc_permitted") is True


def derived_emittable(einput) -> tuple[bool, str | None]:
    """CONTRACT-5 derived-emission runtime-shape eligibility gate.

    Returns (True, None) iff ALL hold; else (False,
    "derived-runtime-unsupported:<reason>"). Pure: data in, verdict out — no
    download / no emit. The order below short-circuits on the FIRST failing
    clause (the brief's negative-matrix reason tokens):

      engine-install-method  required_genesis!=false OR vendored_overlays!=[]
                             OR install.method!="docker_image"
      overlay-feature        runtime["required_engine_features"] != []
      kv                     kv_format ∉ {bf16,fp16,fp8_e5m2,fp8_e4m3}
      drafter                runtime["drafter"] is not None
      gpu-count              visible_gpu_count < tp OR
                             len(selected_gpu_indices) != tp
      unsupported-quant-for-derived[:autoround|:missing-torch-dtype]
                             weight_format not resolvable by the CONTRACT-2
                             quant/dtype dispatch table
    """
    root = einput.diagnostics.get("_root") if einput.diagnostics else None
    rt = einput.runtime or {}
    engine_id = rt.get("engine")

    # --- engine + runtime clean, correct objects, AND docker_image install --
    engine = None
    if root is not None and engine_id:
        try:
            engine = load_engine(Path(root), engine_id)
        except Refuse:
            engine = None
    if engine is None:
        # cannot resolve the engine YAML -> cannot prove it is clean.
        return False, "derived-runtime-unsupported:engine-install-method"
    install = _engine_install(engine)
    if (
        engine.get("required_genesis") is not False
        or (engine.get("vendored_overlays") or []) != []
        or install.get("method") != "docker_image"
    ):
        return False, "derived-runtime-unsupported:engine-install-method"
    if (rt.get("required_engine_features") or []) != []:
        return False, "derived-runtime-unsupported:overlay-feature"

    # --- kv_format in the EXPLICIT derived-safe set -----------------------
    if rt.get("kv_format") not in DERIVED_SAFE_KV:
        return False, "derived-runtime-unsupported:kv"

    # --- no drafter / DFlash / MTP / speculative --------------------------
    if rt.get("drafter") is not None:
        return False, "derived-runtime-unsupported:drafter"

    # --- local GPU topology can actually run this tp ----------------------
    tp = int(rt.get("tp") or 1)
    if int(einput.visible_gpu_count) < tp or len(einput.selected_gpu_indices) != tp:
        return False, "derived-runtime-unsupported:gpu-count"

    # --- weight_format resolvable by the CONTRACT-2 dispatch table --------
    wf = (getattr(einput.der, "profile", None) or {}).get("weight_format")
    disp = _QUANT_DISPATCH.get(str(wf).lower()) if wf is not None else None
    if disp is None:
        return False, "derived-runtime-unsupported:unsupported-quant-for-derived"
    quant_val, needs_dtype = disp
    if isinstance(quant_val, str) and quant_val.startswith("REJECT:"):
        # autoround (the only current REJECT row).
        return (
            False,
            f"derived-runtime-unsupported:unsupported-quant-for-derived:"
            f"{quant_val.split(':', 1)[1]}",
        )
    if needs_dtype:
        dtype, reject = _resolve_compute_dtype(einput)
        if not dtype:
            return (
                False,
                f"derived-runtime-unsupported:unsupported-quant-for-derived:{reject}",
            )

    return True, None


def generate_from_profile(root: Path, einput) -> tuple[str, dict]:
    """ADDITIVE [E] entry point — emit the SEPARATE patchless `derived-vllm`
    base template per CONTRACT-2's per-arg policy. NEVER touches generate()
    or the curated compose_service_template (CONTRACT-2: a Llama model
    through the Qwen-constant curated template = broken boot).

    Runs CONTRACT-5 `derived_emittable()` FIRST; on any reject (or any
    dispatch-table reject) raises Refuse (mirrors [D] style/code) — never an
    emit. Returns (compose_text, meta) on success.
    """
    ok, reason = derived_emittable(einput)
    if not ok:
        raise Refuse(reason or "derived-runtime-unsupported:unknown")

    rt = einput.runtime or {}
    engine_id = rt["engine"]
    engine = load_engine(root, engine_id)
    install = _engine_install(engine)

    # image = resolved engine install.spec (NOT a bare ${VLLM_NIGHTLY_SHA}).
    image = install.get("spec")
    if not image:
        raise Refuse(
            "derived-runtime-unsupported:engine-install-method "
            f"(engine {engine_id} install.spec missing)"
        )

    san = _sanitize_slug(einput.slug)
    if not san:
        raise Refuse(
            "derived-runtime-unsupported:unsupported-quant-for-derived "
            f"(slug {einput.slug!r} sanitizes to empty served-model-name)"
        )
    container_model_dir = f"/models/club3090/pulls/{san}"
    host_model_dir = f"{Path(einput.hf_home)}/club3090/pulls/{san}"

    # --quantization / --dtype via the CONTRACT-2 dispatch table.
    wf = (getattr(einput.der, "profile", None) or {}).get("weight_format")
    quant_val, needs_dtype = _QUANT_DISPATCH[str(wf).lower()]
    if needs_dtype:
        dtype, reject = _resolve_compute_dtype(einput)
        if not dtype:  # pragma: no cover — derived_emittable already rejected
            raise Refuse(
                f"derived-runtime-unsupported:unsupported-quant-for-derived:{reject}"
            )
    else:
        # pure float16/bfloat16 rows: --dtype IS the weight_format.
        dtype = "float16" if str(wf).lower() == "float16" else "bfloat16"

    tp = int(rt.get("tp") or 1)
    max_ctx = int(rt.get("max_ctx") or 131072)
    mem_util = float(rt.get("mem_util") or 0.90)
    max_num_seqs = int(rt.get("max_num_seqs") or 1)
    port = int(rt.get("default_port") or rt.get("port") or 8000)

    # --kv-cache-dtype: kv_arg() EMISSION-ONLY on the already-eligible value.
    kv_cli = kv_arg(rt["kv_format"])  # bf16/fp16 -> None (flag omitted)

    gpu_idx = ",".join(str(i) for i in einput.selected_gpu_indices)

    svc = f"vllm-derived-{san}"[:63]

    cmd: list[str] = ["      - --model", f"      - {container_model_dir}"]
    cmd += ["      - --served-model-name", f"      - {san}"]
    if quant_val is not None:
        cmd += ["      - --quantization", f"      - {quant_val}"]
    cmd += ["      - --dtype", f"      - {dtype}"]
    cmd += ["      - --tensor-parallel-size", f'      - "{tp}"']
    cmd += ["      - --max-model-len", f'      - "{max_ctx}"']
    cmd += ["      - --gpu-memory-utilization", f'      - "{mem_util}"']
    cmd += ["      - --max-num-seqs", f'      - "{max_num_seqs}"']
    if kv_cli is not None:
        cmd += ["      - --kv-cache-dtype", f"      - {kv_cli}"]
    if _trc_permitted(einput):
        cmd += ["      - --trust-remote-code"]
    cmd += ["      - --host", '      - "0.0.0.0"', "      - --port", '      - "8000"']

    header = (
        "# ===========================================================================\n"
        "# GENERATED by scripts/pull.sh — club-3090 [E] derived-vllm "
        "(v0.8.0 #147).\n"
        f"#   derived (non-curated) slug: {einput.slug}\n"
        f"#   --profile-like runtime shape: engine={engine_id} kv="
        f"{rt.get('kv_format')} tp={tp}\n"
        f"#   resolved image (engine install.spec, NOT rewritten): {image}\n"
        "#   PATCHLESS default-vLLM: no Genesis / curated patches "
        "(CONTRACT-5 guarantees none needed).\n"
        "#   NOT emitted: --chat-template / --reasoning-parser / "
        "--tool-call-parser (vLLM defaults).\n"
        "# ===========================================================================\n"
    )
    body_lines = [
        "services:",
        f"  {svc}:",
        f"    image: {image}",
        f'    container_name: "${{ESTATE_CONTAINER:-{svc}}}"',
        "    restart: ${CLUB3090_RESTART:-unless-stopped}",
        "    ports:",
        f'      - "${{BIND_HOST:-0.0.0.0}}:${{ESTATE_PORT:-${{PORT:-{port}}}}}:8000"',
        "    volumes:",
        f"      - {host_model_dir}:{container_model_dir}:ro",
        "    environment:",
        f"      - NVIDIA_VISIBLE_DEVICES={gpu_idx}",
        "      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}",
        "      - VLLM_WORKER_MULTIPROC_METHOD=spawn",
        "      - NCCL_CUMEM_ENABLE=0",
        "      - NCCL_P2P_DISABLE=1",
        "      - VLLM_NO_USAGE_STATS=1",
        "    shm_size: \"16gb\"",
        "    ipc: host",
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        "              count: all",
        "              capabilities: [gpu]",
        "    entrypoint:",
        "      - /bin/bash",
        "      - -c",
        "      - |",
        "        exec vllm serve \"$@\"",
        "      - --",
        "    command:",
    ]
    body = header + "\n".join(body_lines + cmd) + "\n"

    meta = {
        "derived": True,
        "slug": einput.slug,
        "served_model_name": san,
        "resolved_image": image,
        "engine": engine_id,
        "container_model_dir": container_model_dir,
        "host_model_dir": host_model_dir,
        "quantization": quant_val,
        "dtype": dtype,
        "kv_format": rt.get("kv_format"),
        "kv_cache_dtype_arg": kv_cli,
        "tp": tp,
        "max_model_len": max_ctx,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": mem_util,
        "port": port,
        "nvidia_visible_devices": gpu_idx,
        "trc_emitted": _trc_permitted(einput),
        "patchless": True,
    }
    return body, meta


def _strip_trc(body: str) -> str:
    """Remove the --trust-remote-code governed token (and only it) from the
    captured body. vLLM `command:` lists render it as its own list element
    line ``- --trust-remote-code``; nothing else legitimately matches that
    exact token, so this is a precise governed-slot suppression, not a
    synthesis."""
    kept = []
    for line in body.splitlines():
        if line.strip() in ("- --trust-remote-code", "--trust-remote-code"):
            continue
        kept.append(line)
    return "\n".join(kept) + ("\n" if body.endswith("\n") else "")


# --------------------------------------------------------------------------
# Convenience-tuple resolver (authoritative input is --profile).
# --------------------------------------------------------------------------
def resolve_convenience(registry: dict, model=None, engine=None, kv=None, tp=None) -> list[str]:
    matches = []
    for name, E in registry.items():
        if model and E.get("model") != model:
            continue
        if engine and E.get("engine") != engine:
            continue
        if kv and E.get("kv_format") != kv:
            continue
        if tp is not None and E.get("tp") != tp:
            continue
        matches.append(name)
    return sorted(matches)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(
        prog="generate-compose.sh",
        description="v0.8.0 #141 — generate a minimal-reproduction vLLM compose "
        "for an in-scope (non-Genesis) profile. Mission: reproduce + flag, never repair.",
    )
    ap.add_argument("--profile", help="authoritative profile name (COMPOSE_REGISTRY key)")
    ap.add_argument("--model", help="convenience tuple: model slug")
    ap.add_argument("--engine", help="convenience tuple: engine id")
    ap.add_argument("--kv", help="convenience tuple: kv_format")
    ap.add_argument("--tp", type=int, help="convenience tuple: tensor-parallel size")
    ap.add_argument("--accept-degraded", action="store_true",
                    help="acknowledge a capability-scoped DEGRADED compose")
    ap.add_argument("--out", help="write the compose here (default: stdout)")
    args = ap.parse_args(argv)

    registry = get_registry(root)

    if not args.profile:
        # Convenience tuple: print matches + exit non-zero (authoritative
        # input is --profile; the tuple is a discovery aid only).
        if not any([args.model, args.engine, args.kv, args.tp is not None]):
            print("error: --profile is required (or a convenience "
                  "--model/--engine/--kv/--tp tuple to list candidates)",
                  file=sys.stderr)
            return EXIT_USAGE
        matches = resolve_convenience(
            registry, args.model, args.engine, args.kv, args.tp
        )
        if matches:
            print("convenience tuple matched these profiles "
                  "(re-run with an authoritative --profile):", file=sys.stderr)
            for m in matches:
                print(f"  --profile {m}", file=sys.stderr)
        else:
            print("convenience tuple matched no profiles", file=sys.stderr)
        return EXIT_AMBIGUOUS

    try:
        out_text, meta = generate(
            root, args.profile, accept_degraded=args.accept_degraded
        )
    except Refuse as r:
        print(f"[generate-compose] REFUSE: {r}", file=sys.stderr)
        return r.code

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
        print(f"[generate-compose] wrote {args.out} "
              f"(profile={meta['profile']}, pin={meta['engine_pin']}, "
              f"wired={len(meta['wired'])}, undelivered={len(meta['undelivered'])}, "
              f"degraded={meta['degraded']})", file=sys.stderr)
    else:
        sys.stdout.write(out_text)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
