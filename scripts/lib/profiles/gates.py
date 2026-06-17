"""v0.8.0 Pull-Gate — STEP P3: stratum-2 precondition + `[C0]`
engine-support/runtime/hardware gate + `[C2a]` disk pre-gate.

These are PURE, injectable, testable functions. They run AFTER P2's
`deriver.derive()` succeeds (stratum-1 errors are already carried on
`DeriveResult.error`). P3 implements ONLY abort strata 2, 3, 4 of the
6-stratum taxonomy. It does NOT orchestrate, does NOT implement `[C1]`,
stratum-5 abort wiring, the `pull` orchestrator, or any download — those
are P4. P4 consumes the structured outputs here and owns the CLI flags
(`--experimental-arch`, `--trust-remote-code`, `--hardware`, ...).

Public API (stable for P4)
--------------------------

    from scripts.lib.profiles import gates

    s2 = gates.stratum2_profile_like(
        registry_key, *, derive_result=None, path,
        root=None, registry=None, runtime=None,
    )                                   # -> Stratum2Result
    #   .ok           -> bool
    #   .refusal      -> Refusal | None  (refuse-reason, non-bypassable)
    #   .engine_id    -> str | None
    #   .registry_entry -> dict | None

    c0 = gates.c0_engine_support(
        registry_key, derive_result, *, path,
        hardware_sm,                    # injectable float (sm_86 -> 8.6)
        root=None, registry=None, runtime=None, arches=None,
    )                                   # -> C0Result
    #   .state        -> C0State  (EXACTLY one of the locked 3)
    #   .sub_reason   -> C0SubReason | None  (structured SIDE field)
    #   .bypassable_by -> tuple[str, ...]    (structured SIDE field; P4 acts)
    #   .detail       -> str

    c2a = gates.c2a_disk(
        derive_result, *, hf_home=None, statvfs=None,
    )                                   # -> C2aResult
    #   .state        -> C2aState  ({disk-ok, disk-short})
    #   .required_gb / .available_gb / .detail

Design-lock (do not violate)
----------------------------
`[C0]` emits EXACTLY one of {engine-supported, engine-support-unknown,
needs-trust-remote-code-ack}. `no-arch-row` / `runtime-incompatible` are
sub-reasons carried in `.sub_reason` — NOT new top-level states. Whether
P4 may bypass a verdict is carried in `.bypassable_by` (a structured side
field); P3 only TAGS, P4 acts on flags.

`[D]` reuse (read-only, never reimplemented)
--------------------------------------------
- engine-pin resolver  -> `generate_compose.validate_engine_pin`
- scope-gate predicate -> `generate_compose.scope_gate`
Both imported as functions. `generate_compose.py` got a small
behaviour-preserving refactor in this STEP solely to expose `scope_gate`
(the engine.type==vllm ∧ profile_runtime-exists ∧ genesis_equipped==false
predicate) as an importable pure function — `test-generate-compose.sh`
output is byte-identical pre/post.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

# Arch-kernel SM rule (locked design [C0] / brief): these KV formats require
# an SM-9.0-class kernel and are NOT loadable on Ampere sm_86 (RTX 3090).
# (fp8_e4m3 native compute + Gemma-TQ3 are the §1 confidently-wrong-on-3090
# risks Codex-r5 High-1 closes; mirrors compose_registry required_sm:9.0 +
# arch_patches Gemma kernel_constraints "fp8_e4m3 is not supported on sm_86".)
_ARCH_KERNEL_SM = {
    "fp8_e4m3": 9.0,
    "turboquant_3bit_nc": 9.0,
}


# ---------------------------------------------------------------------------
# [D] reuse — import the existing resolver + scope-gate as functions.
# ---------------------------------------------------------------------------
def _gc():
    """Import the [D] generator module (read-only reuse of its
    engine-pin resolver + scope-gate predicate + loaders)."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.lib import generate_compose as gc  # noqa: E402

    return gc


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Refusal:
    """A stratum-2 (or hard) refusal. Not bypassable by any flag unless a
    later stratum explicitly tags it so."""

    reason: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.reason}: {self.detail}" if self.detail else self.reason


@dataclass
class Stratum2Result:
    ok: bool
    refusal: Optional[Refusal] = None
    engine_id: Optional[str] = None
    registry_entry: Optional[dict[str, Any]] = None


class C0State(str, Enum):
    """LOCKED design [C0] state set — exactly these three, never extended."""

    ENGINE_SUPPORTED = "engine-supported"
    ENGINE_SUPPORT_UNKNOWN = "engine-support-unknown"
    NEEDS_TRC_ACK = "needs-trust-remote-code-ack"


class C0SubReason(str, Enum):
    """Structured SIDE field — NOT a top-level state (design-lock)."""

    NO_ARCH_ROW = "no-arch-row"
    RUNTIME_INCOMPATIBLE = "runtime-incompatible"


# Bypass-flag tokens P4 will act on (P3 only tags).
BYPASS_EXPERIMENTAL_ARCH = "--experimental-arch"
BYPASS_TRUST_REMOTE_CODE = "--trust-remote-code"


@dataclass
class C0Result:
    state: C0State
    sub_reason: Optional[C0SubReason] = None
    bypassable_by: tuple[str, ...] = ()
    detail: str = ""
    resolved_pin: Optional[str] = None
    arch: Optional[str] = None


class C2aState(str, Enum):
    DISK_OK = "disk-ok"
    DISK_SHORT = "disk-short"


@dataclass
class C2aResult:
    state: C2aState
    required_gb: Optional[float] = None
    available_gb: Optional[float] = None
    detail: str = ""


# ---------------------------------------------------------------------------
# Shared loaders (reuse [D]'s loaders so data parsing never drifts)
# ---------------------------------------------------------------------------
def _load(root: Path, registry, runtime, arches):
    gc = _gc()
    root = root or REPO_ROOT
    if registry is None:
        registry = gc.get_registry(root)
    if runtime is None:
        runtime = gc.load_runtime(root)
    if arches is None:
        arches = gc.load_arches(root)
    return gc, root, registry, runtime, arches


# ---------------------------------------------------------------------------
# STRATUM 2 — --profile-like validity precondition (both paths, pre-[C0]).
# ---------------------------------------------------------------------------
def stratum2_profile_like(
    registry_key: str,
    *,
    derive_result=None,
    path: str = "B",
    root: Optional[Path] = None,
    registry=None,
    runtime=None,
) -> Stratum2Result:
    """Validate the REQUIRED `--profile-like <COMPOSE_REGISTRY key>`.

    Both paths: the named profile's `engine.type` must be `vllm` else
    structured refuse `unsupported-runtime-engine` (covers
    `llamacpp/default` engine=`llama-cpp-local`, mem_util=None).

    Path A additionally:
      - the profile must be `[D]`-emittable: REUSE `[D]`'s scope-gate
        predicate (engine.type==vllm ∧ profile_runtime entry exists ∧
        genesis_equipped==false). Non-emittable (Genesis/TQ3/llama.cpp,
        e.g. `vllm/dual-turbo`) → refuse `profile-not-emittable` BEFORE
        `[C0]`.
      - `--profile-like`.model == resolved curated model AND
        .weights_variant == matched variant (else refuse `profile-mismatch`).

    Not bypassable by any flag (pick another profile).
    """
    gc = _gc()
    root = root or REPO_ROOT
    if registry is None:
        registry = gc.get_registry(root)
    if runtime is None:
        runtime = gc.load_runtime(root)

    if registry_key not in registry:
        return Stratum2Result(
            ok=False,
            refusal=Refusal(
                "unknown-profile-like",
                f"{registry_key!r} not in COMPOSE_REGISTRY",
            ),
        )
    entry = registry[registry_key]
    engine_id = entry["engine"]
    try:
        engine = gc.load_engine(root, engine_id)
    except gc.Refuse as r:
        return Stratum2Result(
            ok=False,
            refusal=Refusal("unsupported-runtime-engine", str(r)),
            engine_id=engine_id,
        )

    # --- both paths: engine.type must be vLLM ------------------------------
    if engine.get("type") != "vllm":
        return Stratum2Result(
            ok=False,
            refusal=Refusal(
                "unsupported-runtime-engine",
                f"--profile-like {registry_key!r} engine={engine_id} "
                f"type={engine.get('type')!r} (mem_util="
                f"{entry.get('mem_util')!r}); only vLLM profiles supply a "
                f"valid Pull-Gate runtime shape",
            ),
            engine_id=engine_id,
            registry_entry=entry,
        )

    if path == "A":
        # --- Path A: must be [D]-emittable (REUSE [D] scope-gate) ----------
        try:
            gc.scope_gate(engine_id, engine, runtime, registry_key)
        except gc.Refuse as r:
            return Stratum2Result(
                ok=False,
                refusal=Refusal("profile-not-emittable", str(r)),
                engine_id=engine_id,
                registry_entry=entry,
            )
        # --- Path A: model/variant must match the curated resolution ------
        t1 = getattr(derive_result, "tier1", None) if derive_result else None
        if t1 is None:
            return Stratum2Result(
                ok=False,
                refusal=Refusal(
                    "profile-mismatch",
                    "Path A requires a curated (tier-1) deriver hit to "
                    "validate --profile-like model/variant",
                ),
                engine_id=engine_id,
                registry_entry=entry,
            )
        if entry.get("model") != t1.model_id or entry.get(
            "weights_variant"
        ) != t1.weights_variant:
            return Stratum2Result(
                ok=False,
                refusal=Refusal(
                    "profile-mismatch",
                    f"--profile-like {registry_key!r} is "
                    f"({entry.get('model')!r},{entry.get('weights_variant')!r}) "
                    f"but the curated slug resolves to "
                    f"({t1.model_id!r},{t1.weights_variant!r})",
                ),
                engine_id=engine_id,
                registry_entry=entry,
            )

    return Stratum2Result(
        ok=True, engine_id=engine_id, registry_entry=entry
    )


# ---------------------------------------------------------------------------
# [C0] runtime SM resolution helpers.
# ---------------------------------------------------------------------------
def _resolve_arch_row(gc, root, runtime, arches, model_slug: str):
    """REUSE [D]'s arch resolution (arch_model_xref.model_slugs join).
    Returns (arch_name, arch_row) or (None, None) when there is no row."""
    try:
        return gc.resolve_arch(root, runtime, arches, model_slug)
    except gc.Refuse:
        return None, None


def _required_sm(engine: dict, entry: dict, kv_format: str) -> float:
    """hardware SM ≥ max(engine.min_sm, registry required_sm,
    arch-kernel SM rule for the requested runtime).

    - engine.min_sm        : engines/<id>.yml `min_sm`
    - registry required_sm : compose_registry `required_sm` (9.0 rows)
    - arch-kernel SM rule  : _ARCH_KERNEL_SM[kv_format] (fp8_e4m3 / Gemma-TQ3)
    """
    eng_sm = float(engine.get("min_sm") or 0.0)
    reg_sm = float(entry.get("required_sm") or 0.0)
    arch_sm = float(_ARCH_KERNEL_SM.get(kv_format, 0.0))
    return max(eng_sm, reg_sm, arch_sm)


# ---------------------------------------------------------------------------
# STRATUM 3 — [C0] engine-support gate.
# ---------------------------------------------------------------------------
def c0_engine_support(
    registry_key: str,
    derive_result,
    *,
    path: str = "B",
    hardware_sm: float,
    root: Optional[Path] = None,
    registry=None,
    runtime=None,
    arches=None,
) -> C0Result:
    """`[C0]` — emit EXACTLY one locked design state + optional structured
    sub-reason + structured bypassable-tag.

    Order of determination (monotonic):
      1. trust-remote-code (fail-closed): arch_patches
         `requires_trust_remote_code ∈ {true, unverified}` OR config.json
         `auto_map` present (P2 surfaces `profile.auto_map`) →
         NEEDS_TRC_ACK. Bypassable only by `--trust-remote-code`; if there
         is ALSO no arch row, P4 will additionally require
         `--experimental-arch` (we tag both).
      2. no arch row → ENGINE_SUPPORT_UNKNOWN / NO_ARCH_ROW
         (bypassable by `--experimental-arch`).
      3. arch row exists but runtime/hardware not loadable →
         ENGINE_SUPPORT_UNKNOWN / RUNTIME_INCOMPATIBLE (NON-bypassable):
           - REUSE [D] engine-pin resolver, assert loads:true
           - tp ∈ arch valid_tp.tp_divisors
           - kv_format ∈ engine supported_kv_formats AND vs arch
             quant-kernel/MoE constraints
           - hardware_sm ≥ max(engine.min_sm, registry required_sm,
             arch-kernel SM rule)
      4. else ENGINE_SUPPORTED.

    Path A checks the curated profile's `weights_variant` against arch
    constraints; Path B uses the deriver-resolved `weight_format`/quant
    (NOT the cloned profile's weights_variant).
    """
    gc, root, registry, runtime, arches = _load(root, registry, runtime, arches)
    entry = registry[registry_key]
    engine_id = entry["engine"]
    engine = gc.load_engine(root, engine_id)
    kv_format = entry.get("kv_format")

    profile = getattr(derive_result, "profile", None) or {}
    t1 = getattr(derive_result, "tier1", None)

    # The arch / model under evaluation.
    if path == "A" and t1 is not None:
        model_slug = entry.get("model")
        arch_from_config = None
    else:
        model_slug = derive_result.slug if derive_result else entry.get("model")
        arch_from_config = profile.get("arch")

    arch_name, arch_row = _resolve_arch_row(
        gc, root, runtime, arches, model_slug
    )
    # Path B: the registry model_slug join won't resolve an uncurated slug;
    # fall back to the config-derived architectures[0] for the matrix lookup.
    if arch_row is None and arch_from_config:
        arch_row = next(
            (r for r in arches if r.get("arch") == arch_from_config), None
        )
        if arch_row is not None:
            arch_name = arch_from_config
        else:
            # Resolve the config-reported OUTER wrapper arch to its canonical
            # matrix arch via arch_model_xref config_architectures (e.g.
            # Qwen3_5ForConditionalGeneration -> Qwen3NextForCausalLM), so an
            # uncurated derive (abliterated / fine-tune) of an already-supported
            # arch is not false-aborted with NO_ARCH_ROW.
            canon, arch_row = gc.resolve_arch_from_config(
                runtime, arches, arch_from_config
            )
            if arch_row is not None:
                arch_name = canon

    has_auto_map = bool(profile.get("auto_map"))

    # --- 1. trust-remote-code fail-closed ---------------------------------
    trc = (arch_row or {}).get("requires_trust_remote_code")
    trc_blocks = trc in {"true", "unverified"} or has_auto_map
    if trc_blocks:
        bypass = [BYPASS_TRUST_REMOTE_CODE]
        why = []
        if trc in {"true", "unverified"}:
            why.append(f"requires_trust_remote_code={trc!r}")
        if has_auto_map:
            why.append("config.json auto_map present")
        # If there is also no arch row, P4 additionally requires
        # --experimental-arch (tag BOTH conditions; design-lock: still
        # the single top-level NEEDS_TRC_ACK state).
        if arch_row is None:
            bypass.append(BYPASS_EXPERIMENTAL_ARCH)
            why.append("no arch matrix row")
        return C0Result(
            state=C0State.NEEDS_TRC_ACK,
            bypassable_by=tuple(bypass),
            detail=(
                f"arch {arch_name or arch_from_config or model_slug!r}: "
                + "; ".join(why)
            ),
            arch=arch_name or arch_from_config,
        )

    # --- 2. no arch row → unknown / no-arch-row (experimental-arch only) ---
    if arch_row is None:
        return C0Result(
            state=C0State.ENGINE_SUPPORT_UNKNOWN,
            sub_reason=C0SubReason.NO_ARCH_ROW,
            bypassable_by=(BYPASS_EXPERIMENTAL_ARCH,),
            detail=(
                f"no arch_patches matrix row for "
                f"{arch_from_config or model_slug!r}; a pull would likely "
                f"fail to boot"
            ),
            arch=arch_from_config,
        )

    # --- 3. arch row exists: runtime + hardware loadability ----------------
    def _incompat(detail: str) -> C0Result:
        return C0Result(
            state=C0State.ENGINE_SUPPORT_UNKNOWN,
            sub_reason=C0SubReason.RUNTIME_INCOMPATIBLE,
            bypassable_by=(),  # NON-bypassable
            detail=detail,
            arch=arch_name,
        )

    # 3a. engine-pin resolver (REUSE [D]) — must map to a loads:true pin.
    try:
        resolved_pin = gc.validate_engine_pin(engine_id, engine, arch_row)
    except gc.Refuse as r:
        return _incompat(
            f"engine-pin resolver: {r} (arch {arch_name})"
        )

    # 3b. tp ∈ arch valid_tp.tp_divisors
    tp_divisors = (arch_row.get("valid_tp") or {}).get("tp_divisors") or []
    if entry.get("tp") not in tp_divisors:
        return _incompat(
            f"tp={entry.get('tp')} not in arch {arch_name} "
            f"valid_tp.tp_divisors {tp_divisors}"
        )

    # 3c. kv_format vs engine supported_kv_formats (quant-kernel constraint)
    if kv_format not in (engine.get("supported_kv_formats") or []):
        return _incompat(
            f"kv_format {kv_format!r} not in engine {engine_id} "
            f"supported_kv_formats {engine.get('supported_kv_formats')}"
        )

    # 3c-bis. weight ARTIFACT vs engine — the GGUF axis (structural & mutually
    # exclusive). GGUF-serving engines (llama.cpp / ik-llama / beellama) and
    # safetensors-serving engines (vLLM / SGLang) cannot load each other's
    # artifact. The deriver already rejects GGUF on the --profile-like *derive*
    # path (safetensors-only header probe), but the curated registry / the
    # curated-swap path (clone a vLLM compose, point --model at your weights)
    # have no such probe — so enforce it here too, off the engine's declared
    # `supported_weight_formats`. Match the unambiguous `gguf` token only (NOT
    # full membership: the dtype-derived weight_format is raw-spelled
    # "bfloat16"/"float16", which a naive `in` test would false-reject).
    weight_format = str(profile.get("weight_format") or "").lower()
    _eng_wfmts = [
        str(f).lower() for f in (engine.get("supported_weight_formats") or [])
    ]
    if weight_format == "gguf" and "gguf" not in _eng_wfmts:
        return _incompat(
            f"weight_format 'gguf' not loadable by engine {engine_id} "
            f"(safetensors-only; supported_weight_formats="
            f"{engine.get('supported_weight_formats')}) — serve GGUF with a "
            f"llama.cpp / ik-llama / beellama compose, not a vLLM one"
        )

    # 3d. hardware SM ≥ max(engine.min_sm, registry required_sm, arch-kernel)
    need_sm = _required_sm(engine, entry, kv_format)
    if float(hardware_sm) < need_sm:
        return _incompat(
            f"hardware sm_{float(hardware_sm):g} < required sm_{need_sm:g} "
            f"for runtime (engine.min_sm={engine.get('min_sm')}, "
            f"registry.required_sm={entry.get('required_sm')}, "
            f"arch-kernel[{kv_format}]="
            f"{_ARCH_KERNEL_SM.get(kv_format, 0.0):g}); "
            f"e.g. fp8_e4m3 / Gemma-TQ3 need SM 9.0 — NOT loadable on sm_86"
        )

    # --- 4. supported -----------------------------------------------------
    return C0Result(
        state=C0State.ENGINE_SUPPORTED,
        detail=(
            f"arch {arch_name} loadable on {resolved_pin} "
            f"(tp={entry.get('tp')}, kv={kv_format}, "
            f"sm_{float(hardware_sm):g} >= sm_{need_sm:g})"
        ),
        resolved_pin=resolved_pin,
        arch=arch_name,
    )


# ---------------------------------------------------------------------------
# STRATUM 4 — [C2a] disk pre-gate (ordered AFTER [C0], before [B]).
# ---------------------------------------------------------------------------
def c2a_disk(
    derive_result,
    *,
    hf_home: Optional[str] = None,
    statvfs: Optional[Callable[[str], Any]] = None,
) -> C2aResult:
    """`[C2a]` — Σ(P2-selected weight blobs + required config/tokenizer
    siblings) × 1.2 vs resolved HF_HOME free space.

    `disk-short` → non-negotiable hard-abort (NO bypass tag — there is no
    flag that can clear it; per design §5.2 / §4.1).

    The footprint is the v0.8.0 [E] CONTRACT-3 SHARED `download_set()` total:
    `[C2a]` sizes EXACTLY the union E2 fetches + E3 smokes — one function,
    no parallel lists that can drift. When the derived profile carries the
    raw HF siblings API (`profile["_hf_api"]`, the deriver's additive E2
    surface) this gate sizes `deriver.sized_download_gb(api)` directly (which
    internally calls the shared `download_set(api)`). Tier-1 curated hits
    carry no API (the curated footprint is the variant `size_gb`) — fall
    back to the precomputed footprint / variant size there. ×1.2 safety
    margin (design §5.2). `resolve_hf_home()` is reused from P2's deriver
    (the documented `--hf-home > $HF_HOME > $XDG_CACHE_HOME/huggingface > ~`
    chain).
    """
    from . import deriver as D

    profile = getattr(derive_result, "profile", None) or {}
    # v0.8.0 [E] CONTRACT-3: size the SHARED download_set() — the exact set
    # E2 fetches. `sized_download_gb` calls `download_set(api)` internally,
    # so [C2a] / E2 / E3 are provably the same set.
    hf_api = profile.get("_hf_api")
    if hf_api is not None:
        footprint = D.sized_download_gb(hf_api)
    else:
        # tier-1 curated hit (no derived API) or a fixture without the API:
        # fall back to the precomputed footprint / variant size.
        footprint = profile.get("footprint_gb")
        if footprint is None:
            footprint = (
                profile.get("weights_variant_size_gb")
                or profile.get("weights_total_gb")
                or 0.0
            )
    required_gb = round(float(footprint) * 1.2, 4)

    target = D.resolve_hf_home(hf_home)
    # Resolve free space against the nearest existing ancestor of the
    # (possibly not-yet-created) target dir.
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    sv = statvfs or os.statvfs
    try:
        st = sv(str(probe))
        avail_bytes = st.f_bavail * st.f_frsize
    except (OSError, AttributeError) as exc:  # pragma: no cover - defensive
        # Cannot determine free space → fail-closed to disk-short (honest:
        # never print a fabricated fit per §1).
        return C2aResult(
            state=C2aState.DISK_SHORT,
            required_gb=required_gb,
            available_gb=None,
            detail=f"could not stat {probe} for free space ({exc})",
        )
    available_gb = round(avail_bytes / (1024 ** 3), 4)

    if available_gb < required_gb:
        return C2aResult(
            state=C2aState.DISK_SHORT,
            required_gb=required_gb,
            available_gb=available_gb,
            detail=(
                f"need {required_gb:g} GiB (footprint {float(footprint):g} "
                f"GiB ×1.2) but only {available_gb:g} GiB free at {target} "
                f"-> hard-abort (no bypass)"
            ),
        )
    return C2aResult(
        state=C2aState.DISK_OK,
        required_gb=required_gb,
        available_gb=available_gb,
        detail=(
            f"{available_gb:g} GiB free >= {required_gb:g} GiB required "
            f"at {target}"
        ),
    )
