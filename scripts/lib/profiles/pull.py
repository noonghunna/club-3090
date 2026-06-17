"""v0.8.0 Pull-Gate — STEP P4: the `pull` orchestrator.

This is the keystone STEP. It chains the *frozen* predecessor slices into
the locked 6-stratum abort taxonomy, implements the two P4-owned decision
units (stratum-5 `no-fit-model` + the `[C1]` §4.1 total function), runs the
Path-A stratum-6 `[D]` dry-run, and emits via the *existing* `[D]`
generator. It owns the CLI flags. It NEVER edits a frozen module — it
imports P1 (`tools/kv-calc.py`), P2 (`deriver`), P3 (`gates`) and `[D]`
(`generate_compose`) read-only.

pull.py / pull.sh split
-----------------------
`scripts/pull.sh` is a thin argv pass-through (the established
`generate-compose.sh` / `diagnose-profile.sh` pattern): it resolves
`ROOT_DIR` and `exec`s `python3 scripts/lib/profiles/pull.py "$@"`. ALL
decision logic lives here in `pull.py` so it is unit-testable hermetically
(injected hardware-SM + injected fetcher + injected statvfs + injected
`[D]` runner — no live network, no GPU, no real emit in tests).

Public API (stable; the test consumes `run_pull`)
-------------------------------------------------

    from scripts.lib.profiles import pull

    res = pull.run_pull(
        slug, profile_like, *,
        path=None,                  # None -> auto (A if curated+--out else B)
        dry_run=False,              # force Path B
        yes=False,                  # satisfy `confirm→proceed` --yes
        force_download=False,       # no-op + notice this phase
        experimental_arch=False,    # bypass ONLY no-arch-row
        trust_remote_code=False,    # bypass needs-trust-remote-code-ack
        hf_home=None,
        out=None,                   # Path A emit target
        hardware_sm=None,           # INJECTABLE (real detect when None)
        fetcher=None,               # INJECTABLE (real HTTP when None)
        statvfs=None,               # INJECTABLE (real os.statvfs when None)
        d_runner=None,              # INJECTABLE [D] dry-run/emit (real gc.generate)
        profiles=None,
        root=None,
    ) -> PullResult

`PullResult` is a frozen-ish dataclass carrying the terminal outcome, the
stratum at which the run stopped, the structured reason, and (Path A only,
on success) the emitted compose text. The truth-table test asserts against
its fields; the CLI renders it to stdout + an exit code.

`[C1]` §4.1
-----------
`c1_terminal(confidence, raw_verdict, flags)` reproduces the
`v0.8.x-design.md` §4.1 3×3 table EXACTLY as DATA (`_C1_TABLE`, a
`dict[(confidence, raw_verdict)] -> _Cell`). It is TOTAL over
`{exact, derived, estimated-lower-bound} × {fits-clean, fits-constrained,
wont-fit}`. The table is reproduced from `v0.8.x-design.md` lines 62-66
(the `### 4.1` table block). No cell was ambiguous.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

# Repo root on sys.path so `scripts.lib.profiles.*` absolute imports resolve
# whether this is imported as a module (tests) OR exec'd as a script
# (pull.sh) — same bootstrap pattern as generate_compose.py.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.profiles import deriver as D  # noqa: E402 (P2, frozen — RO)
from scripts.lib.profiles import gates as G  # noqa: E402  (P3, frozen — RO)


# ---------------------------------------------------------------------------
# P1 — tools/kv-calc.py via the documented sys.modules contract.
# ---------------------------------------------------------------------------
_KV = None


def _kv():
    """Load `tools/kv-calc.py` per the in-file import contract: register in
    `sys.modules["kv_calc"]` BEFORE `exec_module` (kv-calc.py uses
    @dataclass, which resolves `cls.__module__` via sys.modules during class
    creation)."""
    global _KV
    if _KV is not None:
        return _KV
    if "kv_calc" in sys.modules:
        _KV = sys.modules["kv_calc"]
        return _KV
    kv_path = REPO_ROOT / "tools" / "kv-calc.py"
    spec = importlib.util.spec_from_file_location("kv_calc", kv_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kv_calc"] = mod  # MUST precede exec_module
    spec.loader.exec_module(mod)
    _KV = mod
    return _KV


# ---------------------------------------------------------------------------
# [D] — generate_compose, imported read-only (engine-pin resolver, scope-gate,
# and the full generate() path for the stratum-6 dry-run + real emit).
# ---------------------------------------------------------------------------
def _gc():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.lib import generate_compose as gc  # noqa: E402

    return gc


# ===========================================================================
# Terminal + stratum vocabulary (design-locked — never extended)
# ===========================================================================
class Terminal(str, Enum):
    """LOCKED §4.1 / §5.3 terminal set — EXACTLY these four, never more."""

    PROCEED = "proceed"
    CONFIRM_PROCEED = "confirm→proceed"
    HARD_BLOCK = "hard-block"
    OVERRIDE_ACCEPTED = "override-accepted"


# Frozen design-lock assertion target.
LOCKED_TERMINALS = frozenset(t.value for t in Terminal)


class Stratum(int, Enum):
    """Where a run stopped. 0 == ran to a [C1] terminal / Path-B verdict."""

    DERIVER = 1            # stratum-1: deriver structured errors
    PROFILE_LIKE = 2       # stratum-2: --profile-like precondition
    C0 = 3                 # stratum-3: [C0] engine-support / runtime / SM
    C2A_DISK = 4           # stratum-4: [C2a] disk pre-gate
    ELIGIBILITY = 5        # stratum-5: pre-[B] generic-dense eligibility
    D_DRY_RUN = 6          # stratum-6: Path-A [D] dry-run refusal
    DECIDED = 0            # reached [C1] / Path-B verdict (no abort)


@dataclass
class PullResult:
    slug: str
    profile_like: str
    path: str                                   # "A" | "B"
    ok: bool                                    # download-eligible / clean verdict
    stratum: Stratum                            # where it stopped (DECIDED=ran through)
    abort_reason: Optional[str] = None          # structured machine reason
    detail: str = ""
    confidence: Optional[str] = None
    raw_verdict: Optional[str] = None
    terminal: Optional[str] = None              # [C1] terminal (when [B] reached)
    emitted: bool = False                       # Path A only: [D] actually emitted
    compose_text: Optional[str] = None          # Path A only, on emit
    notices: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # ----- v0.8.0 [E] E4 additive outcome fields (CONTRACT-1) -------------
    # None == the [E] stage did NOT run (curated Path-A, --dry-run Path-B,
    # not-download-eligible, or CONTRACT-5 reject). Existing PullResult
    # fields/semantics are UNCHANGED — these are purely additive so every
    # pre-E4 test-pull.sh assertion is byte-unaffected.
    download_ok: Optional[bool] = None          # E2 download stage outcome
    boot_ok: Optional[bool] = None              # E3 derived boot outcome
    smoke: Optional[dict[str, Any]] = None      # E3 derived smoke result
    capture_paths: list[str] = field(default_factory=list)  # §6 artifacts


# ===========================================================================
# [C1] — §4.1 3×3 confidence × raw-verdict → terminal TOTAL FUNCTION.
#
# Reproduced EXACTLY (as DATA, not branching prose) from the locked
# v0.8.x design §4.1 (the confidence × raw-verdict table):
#
#   | Confidence            | fits-clean      | fits-constrained | wont-fit  |
#   | exact                 | proceed(silent) | confirm→proceed  | hard-block|
#   | derived               | confirm→proceed | confirm→proceed  | advisory→ |
#   |                       | (--yes)         | (--yes + notice) | --force-  |
#   |                       |                 |                  | download→ |
#   |                       |                 |                  | override- |
#   |                       |                 |                  | accepted  |
#   | estimated-lower-bound | confirm→proceed | confirm→proceed  | advisory→ |
#   |                       | (--yes + floor) | (--yes + floor + | --force-  |
#   |                       |                 |  notice)         | download→ |
#   |                       |                 |                  | override- |
#   |                       |                 |                  | accepted  |
#
# §4.1 footnote (design line 68): "Never silently gate-pass means precisely:
# only exact × fits-clean reaches proceed without --yes."  No cell was
# ambiguous — every (confidence, raw_verdict) pair has exactly one row text.
# ===========================================================================
class _Need(str, Enum):
    """The flag a cell requires to reach its terminal."""

    NONE = "none"                 # silent (only exact×fits-clean)
    YES = "--yes"                 # confirm→proceed gate: --yes accepts
    FORCE = "--force-download"    # advisory: --force-download → override-accepted
    BLOCK = "block"               # unconditional hard-block (no flag clears it)


@dataclass(frozen=True)
class _Cell:
    """One §4.1 table cell, as data."""

    base_terminal: Terminal       # terminal the cell resolves to *when satisfied*
    need: _Need                   # what flag (if any) the cell requires
    note: str                     # the exact §4.1 parenthetical, surfaced to UX


_C = D.Confidence  # exact / estimated-lower-bound (derived RESERVED, still mapped)

# The 9-cell table. KEY = (confidence-value, raw-verdict-string).
# This dict IS the spec; c1_terminal() is a pure lookup + flag interaction.
_C1_TABLE: dict[tuple[str, str], _Cell] = {
    # --- exact -------------------------------------------------------------
    (_C.EXACT.value, "fits-clean"): _Cell(
        Terminal.PROCEED, _Need.NONE, "proceed (silent)"
    ),
    (_C.EXACT.value, "fits-constrained"): _Cell(
        Terminal.CONFIRM_PROCEED, _Need.YES,
        "constraint changed the requested config — user must accept the "
        "applied ctx/KV constraint even though math is trusted",
    ),
    (_C.EXACT.value, "wont-fit"): _Cell(
        Terminal.HARD_BLOCK, _Need.BLOCK,
        "math trusted; suggest closest-fit",
    ),
    # --- derived (RESERVED for the future override-registry phase; still a
    #     total-function row per §4.1 so the table is exhaustive) -----------
    (_C.DERIVED.value, "fits-clean"): _Cell(
        Terminal.CONFIRM_PROCEED, _Need.YES,
        "best-effort, validate post-boot",
    ),
    (_C.DERIVED.value, "fits-constrained"): _Cell(
        Terminal.CONFIRM_PROCEED, _Need.YES,
        "best-effort + constraint notice",
    ),
    (_C.DERIVED.value, "wont-fit"): _Cell(
        Terminal.OVERRIDE_ACCEPTED, _Need.FORCE,
        "advisory → --force-download → override-accepted",
    ),
    # --- estimated-lower-bound --------------------------------------------
    (_C.ESTIMATED_LOWER_BOUND.value, "fits-clean"): _Cell(
        Terminal.CONFIRM_PROCEED, _Need.YES,
        "VRAM is a floor; likely under-modeled",
    ),
    (_C.ESTIMATED_LOWER_BOUND.value, "fits-constrained"): _Cell(
        Terminal.CONFIRM_PROCEED, _Need.YES,
        "VRAM is a floor + constraint notice",
    ),
    (_C.ESTIMATED_LOWER_BOUND.value, "wont-fit"): _Cell(
        Terminal.OVERRIDE_ACCEPTED, _Need.FORCE,
        "advisory → --force-download → override-accepted",
    ),
}

# Domain (used by the test to assert totality without re-deriving it here).
C1_CONFIDENCE_DOMAIN = (
    _C.EXACT.value,
    _C.DERIVED.value,
    _C.ESTIMATED_LOWER_BOUND.value,
)
C1_RAW_VERDICT_DOMAIN = ("fits-clean", "fits-constrained", "wont-fit")


@dataclass(frozen=True)
class C1Outcome:
    terminal: Terminal
    satisfied: bool          # did the present flags satisfy the cell?
    note: str
    needs: str               # the flag still required (or "" when satisfied/blocked)


def c1_terminal(confidence: str, raw_verdict: str, flags: dict) -> C1Outcome:
    """The §4.1 total function. Pure: (confidence, raw_verdict, flags) ->
    C1Outcome. `flags` carries booleans `yes` / `force_download`.

    Resolution per §4.1 (the dict above is the authority — this only
    encodes the flag interaction the table prescribes, never new policy):

      - `_Need.NONE`  : reaches `proceed` with no flag (only exact×clean).
      - `_Need.YES`   : `confirm→proceed`; reached when `--yes` present,
                        else NOT satisfied (advisory: "re-run with --yes").
      - `_Need.FORCE` : low-confidence wont-fit advisory → `override-accepted`
                        ONLY with `--force-download`; else NOT satisfied.
      - `_Need.BLOCK` : `hard-block`, no flag clears it (exact×wont-fit).
    """
    key = (confidence, raw_verdict)
    cell = _C1_TABLE.get(key)
    if cell is None:  # pragma: no cover — totality is test-asserted
        raise KeyError(f"§4.1 has no cell for {key!r} (table is TOTAL)")

    if cell.need is _Need.NONE:
        return C1Outcome(cell.base_terminal, True, cell.note, "")
    if cell.need is _Need.BLOCK:
        # hard-block is itself the terminal; not a gate-pass, no flag clears.
        return C1Outcome(Terminal.HARD_BLOCK, False, cell.note, "")
    if cell.need is _Need.YES:
        if flags.get("yes"):
            return C1Outcome(cell.base_terminal, True, cell.note, "")
        return C1Outcome(cell.base_terminal, False, cell.note, "--yes")
    if cell.need is _Need.FORCE:
        if flags.get("force_download"):
            # override-accepted is NOT a gate-pass (§5.3 / design line 106):
            # state + telemetry notice only, NO download this phase.
            return C1Outcome(
                Terminal.OVERRIDE_ACCEPTED, True, cell.note, ""
            )
        return C1Outcome(
            Terminal.OVERRIDE_ACCEPTED, False, cell.note, "--force-download"
        )
    raise AssertionError(f"unreachable _Need {cell.need!r}")  # pragma: no cover


# ===========================================================================
# §7 boot-fit ≠ runtime caveat (printed on every download-eligible AND every
# Path-B verdict; presentation-only, decision-logic-neutral).
# ===========================================================================
CAVEAT_S7 = (
    "boot-fit satisfied; this does NOT guarantee stability under "
    "sustained / accumulated-context workloads — validate with "
    "soak-continuous before relying on it (recommend: scripts/soak.sh "
    "SOAK_MODE=continuous)."
)


# ===========================================================================
# Hardware-SM detection (real path, INJECTABLE so tests are hermetic).
# ===========================================================================
def detect_hardware_sm() -> Optional[float]:
    """Real detection via the existing preflight path
    (`nvidia-smi --query-gpu=...,compute_cap`). Returns the MIN sm across
    visible GPUs (the binding constraint for a multi-GPU runtime), or None
    when nvidia-smi is unavailable. NEVER called in tests (they inject)."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:
        return None
    caps = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            caps.append(float(line))
        except ValueError:  # pragma: no cover
            continue
    return min(caps) if caps else None


# ===========================================================================
# GPU / topology detection (v0.8.0 [E] E4 — additive; reuses the SAME
# nvidia-smi detection discipline as detect_hardware_sm(); INJECTABLE so
# tests are hermetic — NO real GPU in CI). gates.py has no GPU-count/VRAM
# helper (only the compute_cap SM path), so the additive-behaviour-preserving
# option per the brief is a new pull.py helper here (frozen modules
# untouched; curated Path-A unaffected — this is only consulted on the
# post-[C1] derived [E] path).
#
# Returns (visible_gpu_count, per_gpu_vram_mib, gpu_names) over ALL visible
# GPUs; None when nvidia-smi is unavailable AND no override was given (the
# derived [E] path then refuses honestly — never fabricates a topology).
# ===========================================================================
def detect_gpu_topology() -> Optional[tuple[int, list[int], list[str]]]:
    """Real detection via `nvidia-smi --query-gpu=memory.total,name`.
    Returns (count, per_gpu_vram_mib, gpu_names) for ALL visible GPUs, or
    None when nvidia-smi is absent. NEVER called in tests (they inject)."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):  # pragma: no cover - env dependent
        return None
    try:  # pragma: no cover - exercised on-rig (E5), injected in CI
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:  # pragma: no cover
        return None
    vram: list[int] = []
    names: list[str] = []
    for line in out.stdout.splitlines():  # pragma: no cover - on-rig
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        try:
            vram.append(int(float(parts[0])))
        except (ValueError, IndexError):
            continue
        names.append(parts[1] if len(parts) > 1 else "GPU")
    if not vram:  # pragma: no cover
        return None
    return len(vram), vram, names  # pragma: no cover


def _topology_summary_canonical(names: list[str], vram_mib: list[int]) -> str:
    """§6.2 verbatim — deterministic serialization of SORTED
    (gpu_name, vram_mib) tuples. Stable across runs/rigs for the
    capture consensus key."""
    tuples = sorted(
        (str(n), int(v)) for n, v in zip(names, vram_mib)
    )
    return "[" + ", ".join(f"({n}, {v})" for n, v in tuples) + "]"


def _weights_oversize_advisory(weights_total_gb, gpu_topology) -> str:
    """If the raw weights ALONE exceed the detected topology's total VRAM,
    return a one-line advisory explaining the won't-fit. Used to enrich the
    eligibility `no-fit-model` abort for an uncurated derive — e.g. a bf16
    full-weight abliterated repo of a model we serve only quantized — so the
    user sees WHY their specific repo can't run, not just a generic stop.

    Pure / total: empty string when it fits, when size or topology is unknown
    (headless / injected None), or on any malformed input — never raises."""
    try:
        wgb = float(weights_total_gb)
    except (TypeError, ValueError):
        return ""
    if wgb <= 0 or not gpu_topology:
        return ""
    try:
        gcount, vram_mib, gnames = gpu_topology
        total_vram_gb = sum(float(v) for v in vram_mib) / 1024.0
    except (TypeError, ValueError):
        return ""
    if total_vram_gb <= 0 or wgb <= total_vram_gb:
        return ""
    gpu_label = gnames[0] if gnames else "GPU"
    return (
        f" SIZE: the weights are ~{wgb:.0f} GB but the detected topology has "
        f"~{total_vram_gb:.0f} GB total VRAM ({gcount}× {gpu_label}) — they "
        f"won't fit even before KV cache. A bf16 full-weight repo is too large "
        f"for this rig; use an int4 (AWQ / AutoRound, ~1/4 the size) or GGUF variant."
    )


# ===========================================================================
# v0.8.2 CONTRACT-1.1 — capture-on-hard-block PASS-THROUGH.
#
# Emits a pt1-gate-only redacted bundle on a terminal hard-block, BEFORE the
# existing `return res`. This is a PURE PASS-THROUGH: it changes NONE of the
# `[C1]`/6-stratum decision logic — every `return res` decision (ok / stratum
# / abort_reason / detail) is exactly what it was; we only emit a bundle (a
# local file op, no network) and record the bundle dir on `res`. ANY emit
# failure is swallowed structurally — a capture problem must NEVER change a
# gate decision or raise out of `run_pull` (the locked `[F]`-offline / I/O-
# free-gate boundary; pre-existing test-pull.sh assertions byte-unaffected).
#
# `res.capture_paths` (an existing additive PullResult field) is appended so
# the surface step (V2) + tests can find the bundle; nothing else on `res`
# is touched.
# ===========================================================================
def _gate_capture_passthrough(
    res, *, slug, profile_like, der, hardware_sm, gpu_topology, root,
    gate_capture_fn,
):
    """Emit the pt1-gate-only bundle for a terminal hard-block, then return
    `res` UNCHANGED (pass-through). Swallows every exception — a capture
    failure never alters a gate decision."""
    try:
        from scripts.lib.profiles import capture as _CAP

        fn = gate_capture_fn or _CAP.emit_gate_capture

        # club3090 commit captured on the HOST (mirrors _build_einput; never
        # git-in-container) — best-effort, "unknown" on any failure.
        commit = "unknown"
        try:
            import subprocess

            cp = subprocess.run(
                ["git", "-C", str(root or REPO_ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if cp.returncode == 0 and cp.stdout.strip():
                commit = cp.stdout.strip()
        except Exception:  # pragma: no cover - git always present on rig
            commit = "unknown"

        out = fn(
            slug=slug,
            profile_like=profile_like,
            abort_reason=res.abort_reason or "",
            confidence=(
                SimpleNamespace(name=res.confidence)
                if res.confidence else None
            ),
            raw_verdict=res.raw_verdict,
            detail=res.detail,
            der=der,
            hardware_sm=hardware_sm,
            gpu_topology=gpu_topology,
            club3090_commit=commit,
            predicted_b_breakdown=res.diagnostics.get("b_breakdown"),
            repo_root=root or REPO_ROOT,
        )
        d = out.get("dir") if isinstance(out, dict) else None
        if d:
            res.capture_paths.append(str(d))
            res.diagnostics["gate_capture"] = {"dir": str(d)}
    except Exception as exc:  # pragma: no cover - capture is best-effort
        res.diagnostics["gate_capture"] = {"error": repr(exc)}
    return res


# ===========================================================================
# The orchestrator — chains the frozen slices in the LOCKED stratum order.
# ===========================================================================
def run_pull(
    slug: str,
    profile_like: str,
    *,
    path: Optional[str] = None,
    dry_run: bool = False,
    yes: bool = False,
    force_download: bool = False,
    experimental_arch: bool = False,
    trust_remote_code: bool = False,
    hf_home: Optional[str] = None,
    out: Optional[str] = None,
    hardware_sm: Optional[float] = None,
    fetcher=None,
    statvfs: Optional[Callable[[str], Any]] = None,
    d_runner: Optional[Callable[[Path, str, bool], tuple]] = None,
    profiles=None,
    root: Optional[Path] = None,
    # ----- v0.8.0 [E] E4 — INJECTABLE GPU/topology + [E]-stage hooks ------
    # gpu_topology: (count, per_gpu_vram_mib, gpu_names) — None -> real
    # detect_gpu_topology(); tests inject. Mirrors hardware_sm's
    # inject-or-detect pattern (the brief's --hardware-gpus-style override).
    gpu_topology: Optional[tuple[int, list[int], list[str]]] = None,
    # The 5 [E]-stage funcs are injectable so test-pull.sh stays mock-only
    # (NO real Docker / GPU / network — real on-rig is E5). None -> the real
    # shipped E1/E2/E3 functions.
    emit_fn: Optional[Callable] = None,         # E1 generate_from_profile
    download_fn: Optional[Callable] = None,     # E2 download_model
    # E3 boot lifecycle CM factory (the E3/E4-fix seam): a context-manager
    # factory `boot_cm(ei, compose_text, runner=...) -> ctx[BootResult]`
    # that keeps the server ALIVE for the `with` body (boot -> smoke ->
    # capture) and tears it down on __exit__. Default = booted_derived.
    boot_cm: Optional[Callable] = None,         # E3 booted_derived (CM)
    smoke_fn: Optional[Callable] = None,        # E3 smoke_derived
    capture_fn: Optional[Callable] = None,      # E3 emit_capture
    override_capture_fn: Optional[Callable] = None,  # E4 emit_override_capture
    # v0.8.2 CONTRACT-1.1 — the pt1-gate-only emitter, wired as a pure
    # PASS-THROUGH on the terminal hard-block return paths (it emits a
    # bundle BEFORE the existing `return res`; the decision is UNCHANGED).
    # Injectable so test-pull.sh stays mock-only; None -> the real shipped
    # capture.emit_gate_capture.
    gate_capture_fn: Optional[Callable] = None,
    # v0.8.2 CONTRACT-3 — OPTIONAL non-NVIDIA hardware-enumeration augment.
    # Consulted ONLY when `nvidia-smi` already returned nothing (so the
    # NVIDIA majority NEVER enters this seam — that path is byte-identical
    # with this absent OR present-but-degrading). hw-detect-ONLY: it yields
    # an SM-equivalent for the [C0] SM gate; it NEVER feeds kv-calc. None
    # -> the real shipped hwdetect.detect_non_nvidia_sm (optional bounded
    # `whichllm` subprocess; absent/failed -> graceful degrade to today's
    # exact `hardware-sm-undetermined` behaviour). Injectable so
    # test-pull.sh / test-hwdetect.sh stay hermetic (no `whichllm`, no
    # non-NVIDIA GPU in CI).
    hwdetect_fn: Optional[Callable] = None,
) -> PullResult:
    """Execute the 6-stratum Pull-Gate state machine.

    Order is STRICT and monotonic (the design's locked taxonomy):
      stratum-1  deriver structured errors (P2; already structured)
      stratum-2  --profile-like precondition (P3 gates.stratum2)
      stratum-3  [C0] engine-support/runtime/SM (P3 gates.c0) + flag bypass
      stratum-4  [C2a] disk pre-gate (P3 gates.c2a)
      stratum-5  pre-[B] generic-dense eligibility (P4; this module)
      [B]        kv.raw_verdict (P1)
      [C1]       §4.1 total function (P4; this module)
      stratum-6  Path A only: [D] dry-run (read-only existing generator)
      emit       Path A only, on a clean dry-run: real [D] generate()

    Path B (universal evaluate / --dry-run) NEVER reaches stratum-6, NEVER
    calls [D], NEVER downloads. `--force-download` is a no-op + notice this
    phase (emit/download deferred to the Loop phase).
    """
    root = root or REPO_ROOT
    gc = _gc()
    kv = _kv()
    flags = {"yes": yes, "force_download": force_download}

    # ----- Path selection -------------------------------------------------
    # Explicit `path=` wins (test driver). Else: --dry-run forces Path B;
    # otherwise Path A iff a curated tier-1 hit AND an --out target, else B.
    forced_path = path
    if profiles is None:
        from scripts.lib.profiles.compat import load_profiles

        profiles = load_profiles()

    # ----- stratum-1: deriver (P2, frozen) --------------------------------
    der = D.derive(
        slug, hf_home=hf_home, fetcher=fetcher, profiles=profiles
    )
    if der.error is not None:
        _der_res = PullResult(
            slug=slug, profile_like=profile_like, path="?",
            ok=False, stratum=Stratum.DERIVER,
            abort_reason=der.error.kind.value, detail=str(der.error),
        )
        # v0.8.2 CONTRACT-1.1 pass-through (pre-deriver: no der.profile ->
        # model/arch/quant null). Decision UNCHANGED.
        return _gate_capture_passthrough(
            _der_res, slug=slug, profile_like=profile_like, der=None,
            hardware_sm=hardware_sm, gpu_topology=gpu_topology, root=root,
            gate_capture_fn=gate_capture_fn,
        )

    is_curated = der.tier1 is not None
    if forced_path in ("A", "B"):
        eff_path = forced_path
    elif dry_run:
        eff_path = "B"
    elif is_curated and out is not None:
        eff_path = "A"
    else:
        eff_path = "B"

    res = PullResult(
        slug=slug, profile_like=profile_like, path=eff_path,
        ok=False, stratum=Stratum.DECIDED,
        confidence=der.confidence.value if der.confidence else None,
    )

    # ----- stratum-2: --profile-like precondition (P3, frozen) ------------
    s2 = G.stratum2_profile_like(
        profile_like, derive_result=der, path=eff_path, root=root,
    )
    if not s2.ok:
        res.ok = False
        res.stratum = Stratum.PROFILE_LIKE
        res.abort_reason = s2.refusal.reason
        res.detail = s2.refusal.detail
        return _gate_capture_passthrough(
            res, slug=slug, profile_like=profile_like, der=der,
            hardware_sm=hardware_sm, gpu_topology=gpu_topology, root=root,
            gate_capture_fn=gate_capture_fn,
        )

    # ----- stratum-3: [C0] engine-support / runtime / SM (P3, frozen) -----
    if hardware_sm is None:
        hardware_sm = detect_hardware_sm()
    if hardware_sm is None:
        # v0.8.2 CONTRACT-3 (hw-detect-ONLY) — OPTIONAL non-NVIDIA augment.
        # Reached ONLY when `nvidia-smi` gave nothing, so the NVIDIA
        # majority never executes this and that path is byte-identical
        # whether this augment is absent or present-but-degrading. An
        # absent/failed/non-applicable `whichllm` returns None and we fall
        # straight through to the unchanged `hardware-sm-undetermined`
        # terminal below (safety half). A recognised non-NVIDIA device
        # (AMD ROCm / Apple) yields an SM-equivalent so the eval path can
        # run the [C0] SM gate instead of refusing blind (delivery half).
        # This SM-equivalent feeds ONLY the [C0] gate — NEVER kv-calc
        # (kv-calc stays the sole fit authority; CONTRACT-3 is
        # enumeration-only). Any exception is swallowed by hwdetect itself
        # (leaf module never raises out).
        if hwdetect_fn is None:
            from scripts.lib.profiles.hwdetect import detect_non_nvidia_sm

            hwdetect_fn = detect_non_nvidia_sm
        _aug_sm = hwdetect_fn()
        if _aug_sm is not None:
            # Additive-only: mutate the existing `res` (created above;
            # stratum-2 already passed). NO decision field is touched —
            # only `hardware_sm` (so the EXISTING [C0] gate now runs
            # instead of the unchanged blind-refuse terminal) + an
            # additive notice/diagnostic. kv-calc is NOT consulted.
            hardware_sm = float(_aug_sm)
            res.notices.append(
                "[hw] nvidia-smi absent; non-NVIDIA accelerator "
                "enumerated via the optional hardware-detect subprocess "
                "(enumeration only — kv-calc remains the sole fit "
                "authority; validate with soak-continuous)"
            )
            res.diagnostics["hwdetect"] = {
                "augmented": True, "sm_equiv": float(_aug_sm),
            }
    if hardware_sm is None:
        # No GPU detected and none injected: cannot honestly run the SM
        # gate. Fail closed (never fabricate a fit per §1).
        res.ok = False
        res.stratum = Stratum.C0
        res.abort_reason = "hardware-sm-undetermined"
        res.detail = (
            "could not detect GPU compute capability (nvidia-smi absent) "
            "and no --hardware override given; refusing to run the SM gate "
            "blind"
        )
        # topology GUARANTEED null here (nvidia-smi absent — CONTRACT-1.1):
        # pass gpu_topology=None so the bundle's topology is null even if
        # an unrelated override was supplied.
        return _gate_capture_passthrough(
            res, slug=slug, profile_like=profile_like, der=der,
            hardware_sm=None, gpu_topology=None, root=root,
            gate_capture_fn=gate_capture_fn,
        )

    c0 = G.c0_engine_support(
        profile_like, der, path=eff_path, hardware_sm=float(hardware_sm),
        root=root,
    )
    if c0.state != G.C0State.ENGINE_SUPPORTED:
        # Apply ONLY the bypasses [C0] explicitly tagged on `.bypassable_by`.
        # --experimental-arch bypasses ONLY no-arch-row (never
        # runtime-incompatible — its bypassable_by is () so the membership
        # test below can never let it through).
        bypassed = False
        provided = set()
        if experimental_arch:
            provided.add(G.BYPASS_EXPERIMENTAL_ARCH)
        if trust_remote_code:
            provided.add(G.BYPASS_TRUST_REMOTE_CODE)
        tags = set(c0.bypassable_by)
        if tags and tags.issubset(provided):
            # Every condition [C0] flagged is covered by a provided flag.
            # (auto_map+no-arch-row tags BOTH; requires BOTH flags — the
            # subset test enforces that automatically.)
            bypassed = True
        if not bypassed:
            res.ok = False
            res.stratum = Stratum.C0
            res.abort_reason = c0.state.value + (
                "/" + c0.sub_reason.value if c0.sub_reason else ""
            )
            res.detail = c0.detail
            res.diagnostics["c0_bypassable_by"] = list(c0.bypassable_by)
            return _gate_capture_passthrough(
                res, slug=slug, profile_like=profile_like, der=der,
                hardware_sm=hardware_sm, gpu_topology=gpu_topology,
                root=root, gate_capture_fn=gate_capture_fn,
            )
        res.notices.append(
            f"[C0] {c0.state.value}"
            + (f"/{c0.sub_reason.value}" if c0.sub_reason else "")
            + f" bypassed by {sorted(provided & tags)} (Path B only this "
            f"phase; outcome capture deferred to Loop)"
        )

    # ----- stratum-4: [C2a] disk pre-gate (P3, frozen) — AFTER [C0] -------
    c2a = G.c2a_disk(der, hf_home=hf_home, statvfs=statvfs)
    if c2a.state != G.C2aState.DISK_OK:
        res.ok = False
        res.stratum = Stratum.C2A_DISK
        res.abort_reason = c2a.state.value          # "disk-short"
        res.detail = c2a.detail
        return _gate_capture_passthrough(
            res, slug=slug, profile_like=profile_like, der=der,
            hardware_sm=hardware_sm, gpu_topology=gpu_topology, root=root,
            gate_capture_fn=gate_capture_fn,
        )

    # ----- stratum-5: pre-[B] generic-dense eligibility (P4) --------------
    # Separate pre-[B] abort — NOT a [C0] rewrite ([C0] already emitted
    # engine-supported / a bypassed unknown and stays). Non-bypassable:
    # there is no fit model to force. Tier-1 curated hits are eligible by
    # construction (the curated profile encodes a priced model); for a
    # derived model, ineligible iff deriver said NOT tier-1 AND
    # kv.is_generic_dense_eligible(config) is False (deriver already ran the
    # predicate into .generic_dense_eligible).
    if not is_curated:
        eligible = bool(der.generic_dense_eligible)
        if not eligible:
            res.ok = False
            res.stratum = Stratum.ELIGIBILITY
            res.abort_reason = "no-fit-model"
            _arch = (der.profile or {}).get("arch")
            # If the (uncurated) arch resolves via the xref alias to a family we
            # serve CURATED (hybrid/MoE — not generic-dense, so unfit-priceable
            # here), point at the curated-swap path instead of dead-ending. The
            # generic derive genuinely can't fit these; the curated compose can.
            _hint = ""
            try:
                # `gc` (the generate_compose module) is already loaded above via
                # `gc = _gc()`; reuse it — do NOT re-import as a local (that
                # shadows the module-level `_gc` loader -> UnboundLocalError).
                _rt = gc._load_yaml(
                    root, "scripts/lib/profiles/profile_runtime.yml"
                )
                _canon, _row = gc.resolve_arch_from_config(
                    _rt, gc.load_arches(root), _arch
                )
                if _row is not None:
                    _slugs = (
                        ((_rt.get("arch_model_xref") or {}).get(_canon) or {})
                        .get("model_slugs") or []
                    )
                    if _slugs:
                        _hint = (
                            f" NOTE: this arch is the curated '{_slugs[0]}' "
                            f"({_row.get('family')}) — generic derive can't fit it, "
                            f"but the curated path can: clone that model's compose and "
                            f"point --model at your weights (use a quantized + MTP "
                            f"variant; the bf16 base won't fit and lacks the MTP head). "
                            f"See docs/BRING_YOUR_OWN.md → 'Swap a curated model for a "
                            f"fine-tune / abliterated variant'."
                        )
            except Exception:
                _hint = ""
            # Coarse weights-only VRAM verdict: even when we can't kv-calc-price
            # a hybrid/MoE, if the raw weights exceed total VRAM it won't fit at
            # ANY KV — surface that concretely (the huihui abliterated bf16 ~54GB
            # vs 2×24GB case). Empty when it fits / size unknown / headless.
            _size_hint = _weights_oversize_advisory(
                (der.profile or {}).get("weights_total_gb"), gpu_topology
            )
            res.detail = (
                f"{slug}: not Tier-1 curated and not generic-dense eligible "
                f"(arch {_arch!r}); no fit model "
                f"to price — pre-[B] hard-stop (non-bypassable; "
                f"--experimental-arch does NOT apply — there is no model)."
                + _size_hint
                + _hint
            )
            return _gate_capture_passthrough(
                res, slug=slug, profile_like=profile_like, der=der,
                hardware_sm=hardware_sm, gpu_topology=gpu_topology,
                root=root, gate_capture_fn=gate_capture_fn,
            )

    # ----- [B]: raw fit verdict (P1 kv.raw_verdict) -----------------------
    entry = s2.registry_entry or {}
    spec = der.spec
    if spec is None:
        # Tier-1 curated hit: build the curated-exact kv-calc spec (real
        # model_family preserved) so [B] prices it through P1's authoritative
        # family branch, not generic-dense's conservative lower-bound.
        spec = _curated_spec(profiles, der)
    rv = kv.raw_verdict(
        spec=spec,
        kv_format=entry.get("kv_format", "fp8_e5m2"),
        max_ctx=int(entry.get("max_ctx") or spec.get("max_ctx_supported")
                    or 131072),
        max_num_seqs=int(entry.get("max_num_seqs") or 1),
        tp=int(entry.get("tp") or 1),
        mem_util=float(entry.get("mem_util") or 0.95),
    )
    raw = rv["raw_verdict"]
    res.raw_verdict = raw
    res.diagnostics["b_breakdown"] = rv.get("breakdown_gb")

    # ----- [C1]: §4.1 total function (P4) ---------------------------------
    conf = der.confidence.value
    c1 = c1_terminal(conf, raw, flags)
    res.terminal = c1.terminal.value
    res.diagnostics["c1_note"] = c1.note

    if c1.terminal is Terminal.HARD_BLOCK:
        res.ok = False
        res.stratum = Stratum.DECIDED
        res.abort_reason = "hard-block"
        res.detail = f"[C1] {conf}×{raw} → hard-block ({c1.note})"
        # v0.8.2 CONTRACT-1.1 — the C1 exact×wont-fit terminal (the genuine
        # kv-calc VRAM refusal). predicted_b_breakdown is in scope
        # (res.diagnostics["b_breakdown"] set above) — the pass-through
        # carries it. Decision UNCHANGED.
        return _gate_capture_passthrough(
            res, slug=slug, profile_like=profile_like, der=der,
            hardware_sm=hardware_sm, gpu_topology=gpu_topology, root=root,
            gate_capture_fn=gate_capture_fn,
        )

    if not c1.satisfied:
        # confirm→proceed without --yes, or low-conf wont-fit advisory
        # without --force-download. Honest non-pass: state + the flag the
        # user must add. NEVER a silent gate-pass.
        res.ok = False
        res.stratum = Stratum.DECIDED
        res.abort_reason = f"{c1.terminal.value}:needs {c1.needs}"
        res.detail = (
            f"[C1] {conf}×{raw} → {c1.terminal.value} ({c1.note}); "
            f"re-run with {c1.needs} to accept"
        )
        return res

    if c1.terminal is Terminal.OVERRIDE_ACCEPTED:
        # NOT a fit (design line 106). v0.8.0 [E] E4 trigger semantics:
        # --force-download (which is what SATISFIED this terminal) now
        # ACTIVATES the derived [E] stage for a non-curated model that is
        # NOT --dry-run (was a no-op pre-[E]). The §5.3 override-accepted
        # force-capture (pt5) is emitted on this path. Curated / --dry-run
        # stay verdict-only (unchanged).
        res.ok = True
        res.stratum = Stratum.DECIDED
        res.abort_reason = None
        res.detail = (
            f"[C1] {conf}×{raw} → override-accepted ({c1.note}); "
            f"override-accepted is NOT a fit (calibration signal, never "
            f"recorded as fit-validated — §5.3)"
        )
        res.notices.append(
            "override-accepted: forced low-confidence download is a "
            "calibration signal, NOT a validated fit (§5.3)"
        )
        res.notices.append(CAVEAT_S7)
        if not is_curated and not dry_run:
            # Trigger C: --force-download activates [E] ONLY for the
            # override-accepted terminal. The §5.3 pt5 force-capture is
            # emitted (is_override_accepted=True).
            _run_derived_e_stage(
                res, slug=slug, profile_like=profile_like,
                der=der, s2=s2, c2a=c2a, conf=conf, raw=raw,
                terminal=c1.terminal.value, is_override_accepted=True,
                hf_home=hf_home, hardware_sm=float(hardware_sm),
                gpu_topology=gpu_topology, root=root, fetcher=fetcher,
                emit_fn=emit_fn, download_fn=download_fn, boot_cm=boot_cm,
                smoke_fn=smoke_fn, capture_fn=capture_fn,
                override_capture_fn=override_capture_fn,
            )
        else:
            res.notices.append(
                "override-accepted: download/telemetry capture deferred "
                "(curated or --dry-run — verdict-only this run)"
            )
        return res

    # ----- terminal is proceed / confirm→proceed (satisfied) --------------
    if raw == "fits-constrained" and eff_path == "A":
        res.notices.append(
            "known effective-cap warning: vLLM internally caps effective "
            "KV on this hardware; [D] emits the chosen registry profile "
            "UNCHANGED (no compose config rewritten)"
        )

    # ----- Path B: print the §7-caveated verdict ---------------------------
    if eff_path == "B":
        res.ok = True
        res.stratum = Stratum.DECIDED
        res.detail = (
            f"Path B verdict: [C1] {conf}×{raw} → {c1.terminal.value} "
            f"({c1.note})"
        )
        res.notices.append(CAVEAT_S7)
        # v0.8.0 [E] E4 trigger semantics (the new user-facing contract,
        # ADDITIVE — supersedes "Path B never emits/downloads" ONLY for the
        # non-curated derived download-eligible no-dry-run case):
        #   * non-curated + download-eligible [C1] terminal + NO --dry-run
        #     -> continue into the derived [E] stage;
        #   * --dry-run stays verdict-only (NEVER enters [E]);
        #   * curated forced-B (e.g. --dry-run on a curated slug) is
        #     verdict-only — curated weights are local, no download stage.
        # is_curated is True for a Tier-1 curated hit; a curated slug only
        # reaches Path B here via --dry-run, which the guard below excludes.
        if not is_curated and not dry_run:
            _run_derived_e_stage(
                res, slug=slug, profile_like=profile_like,
                der=der, s2=s2, c2a=c2a, conf=conf, raw=raw,
                terminal=c1.terminal.value, is_override_accepted=False,
                hf_home=hf_home, hardware_sm=float(hardware_sm),
                gpu_topology=gpu_topology, root=root, fetcher=fetcher,
                emit_fn=emit_fn, download_fn=download_fn, boot_cm=boot_cm,
                smoke_fn=smoke_fn, capture_fn=capture_fn,
                override_capture_fn=override_capture_fn,
            )
        else:
            if force_download:
                res.notices.append(
                    "--force-download is a no-op here (curated / --dry-run "
                    "verdict-only; download deferred)"
                )
        return res

    # ----- Path A: stratum-6 [D] dry-run, then real emit ------------------
    runner = d_runner or (lambda r, p, ad: gc.generate(r, p, accept_degraded=ad))
    try:
        compose_text, meta = runner(root, profile_like, False)
    except gc.Refuse as r:
        # [D] refused at one of its LATER points (pin mismatch / TP·KV /
        # trc / foundational-or-degraded patch drift). Surface as a Path-A
        # abort — do NOT report download-eligible (Codex-r5 Med-1: the
        # stratum-2 scope-gate is necessary-not-sufficient for [D] emit).
        res.ok = False
        res.stratum = Stratum.D_DRY_RUN
        res.abort_reason = f"d-refused:{_short_refuse(str(r))}"
        res.detail = (
            f"Path-A [D] dry-run refused: {r} — NOT reported "
            f"download-eligible (stratum-2 scope-gate is necessary but not "
            f"sufficient for [D] emit)"
        )
        return res

    # Clean dry-run: the validated registry key is handed to the existing
    # [D] for real emission. The dry-run already produced the exact compose
    # text (gc.generate is pure); honor --out (COMPOSE_GENERATOR.md
    # --project-directory correctness: the emitted compose's relative
    # overlay mounts resolve from the compose file's own directory, so the
    # consumer must `docker compose --project-directory <repo-root>` — we
    # surface that requirement as a notice and write where --out points).
    res.ok = True
    res.stratum = Stratum.DECIDED
    res.emitted = True
    res.compose_text = compose_text
    res.diagnostics["d_meta"] = meta
    res.detail = (
        f"Path A download-eligible: [C1] {conf}×{raw} → "
        f"{c1.terminal.value}; [D] dry-run clean, compose emitted "
        f"(pin={meta.get('engine_pin')})"
    )
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(compose_text, encoding="utf-8")
        res.diagnostics["out_written"] = str(out_path)
    res.notices.append(
        "run the emitted compose with `docker compose "
        "--project-directory <repo-root> -f <out> up` so the relative "
        "overlay mounts resolve (see COMPOSE_GENERATOR.md)"
    )
    res.notices.append(CAVEAT_S7)
    return res


# ===========================================================================
# v0.8.0 [E] E4 — the post-`[C1]` derived `[E]` stage.
#
# ADDITIVE: wired ONLY from the two post-`[C1]` non-curated download-eligible
# return points (override-accepted-with-force-download, and the satisfied
# proceed/confirm→proceed Path-B verdict), and ONLY when `not is_curated and
# not dry_run`. It NEVER touches the `[C1]` table, the 6-stratum logic, or
# curated Path-A — those are CONSUMED, not modified. It mutates ONLY the
# additive `PullResult` fields (download_ok / boot_ok / smoke /
# capture_paths) + notices/diagnostics; ok/terminal/raw_verdict/emitted/
# stratum/abort_reason are left exactly as the [C1] decision set them.
#
# Sequence (CONTRACT-1..5):
#   1. populate EInput from in-scope run_pull state (CONTRACT-1)
#   2. derived_emittable(einput)  — CONTRACT-5 pre-[E] precondition; on
#      refuse -> structured `derived-runtime-unsupported:<reason>` notice +
#      diagnostics, NO download / NO boot.
#   3. generate_from_profile -> download_model -> `with booted_derived(...)
#      as bt:`  [server UP] -> smoke_derived -> emit_capture (pt1-4 +
#      manifest) [-> pt5 if override]  -> CM __exit__: teardown (ALWAYS,
#      AFTER capture). The E3/E4-fix: smoke + capture run against a LIVE
#      server inside the `with`; teardown is on context-manager exit, not
#      before the boot call returns (on-rig E5 caught the prior teardown-
#      in-finally-before-smoke -> ConnectionRefused defect).
#   4. if is_override_accepted: emit_override_capture (§5.3 pt5) — also
#      inside the `with` (before teardown).
#
# Every stage func is INJECTABLE (test-pull.sh mocks them; real on-rig is
# E5). All stage exceptions are caught + recorded structurally — the [E]
# stage NEVER raises out of run_pull and NEVER mutates the [C1]-owned
# fields, so every pre-existing test-pull.sh assertion is byte-unaffected.
# ===========================================================================
def _build_einput(
    *, slug, der, s2, c2a, terminal, is_override_accepted,
    hf_home, hardware_sm, gpu_topology, root, fetcher,
):
    """CONTRACT-1 — construct the EInput from the in-scope run_pull state.
    GPU/topology via the injectable detect_gpu_topology() (None when no
    nvidia-smi and no override -> the caller refuses honestly)."""
    from scripts.lib.profiles.einput import EInput

    runtime = dict(s2.registry_entry or {})
    der_prof = getattr(der, "profile", None) or {}
    selected = list(
        der_prof.get("download_set")
        or der_prof.get("selected_weight_files")
        or []
    )
    resolved_hf_home = D.resolve_hf_home(hf_home)

    topo = gpu_topology
    if topo is None:
        topo = detect_gpu_topology()
    if topo is None:
        return None, "gpu-topology-undetermined"
    vis_count, per_gpu_vram_mib, gpu_names = topo

    tp = int(runtime.get("tp") or 1)
    # The EXACT tp GPUs the derived compose binds — the contiguous first-tp
    # (the curated `gpu_assignment_mode: contiguous` convention; CONTRACT-5
    # also asserts len(selected_gpu_indices) == tp).
    selected_gpu_indices = list(range(min(tp, vis_count)))
    selected_gpu_vram_mib = [
        per_gpu_vram_mib[i] for i in selected_gpu_indices
        if i < len(per_gpu_vram_mib)
    ]
    sel_names = [
        gpu_names[i] for i in selected_gpu_indices if i < len(gpu_names)
    ]
    topology_summary = _topology_summary_canonical(
        sel_names, selected_gpu_vram_mib
    )

    # club3090 commit captured on the HOST (NEVER git-in-container).
    commit = "unknown"
    try:
        import subprocess

        cp = subprocess.run(
            ["git", "-C", str(root or REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            commit = cp.stdout.strip()
    except (OSError, Exception):  # pragma: no cover - git always present
        commit = "unknown"

    ei = EInput(
        slug=slug,
        terminal=terminal,
        is_override_accepted=is_override_accepted,
        der=der,
        runtime=runtime,
        selected_files=selected,
        hf_home=resolved_hf_home,
        c2a=c2a,
        hardware_sm=float(hardware_sm),
        visible_gpu_count=int(vis_count),
        per_gpu_vram_mib=list(per_gpu_vram_mib),
        selected_gpu_indices=selected_gpu_indices,
        selected_gpu_vram_mib=selected_gpu_vram_mib,
        topology_summary=topology_summary,
        club3090_commit=commit,
        diagnostics={"_root": str(root or REPO_ROOT)},
    )
    return ei, None


def _run_derived_e_stage(
    res, *, slug, profile_like, der, s2, c2a, conf, raw, terminal,
    is_override_accepted, hf_home, hardware_sm, gpu_topology, root, fetcher,
    emit_fn, download_fn, boot_cm, smoke_fn, capture_fn,
    override_capture_fn,
):
    """Run the post-`[C1]` derived `[E]` stage. Mutates ONLY the additive
    `res` fields + notices/diagnostics — NEVER ok/terminal/raw_verdict/
    emitted/stratum/abort_reason (those are the [C1] decision, CONSUMED).
    Every stage func is injectable; all exceptions are recorded
    structurally so this never raises out of run_pull (pre-existing
    test-pull.sh assertions byte-unaffected)."""
    from datetime import datetime, timezone

    # Lazy imports — the real E1/E2/E3 funcs (only when not injected). This
    # keeps the import surface of pull.py unchanged for callers that never
    # reach [E] (curated Path-A / --dry-run).
    from scripts.lib import generate_compose as gc
    from scripts.lib.profiles import booter as _B
    from scripts.lib.profiles import capture as _CAP
    from scripts.lib.profiles import downloader as _DL

    emit_fn = emit_fn or gc.generate_from_profile
    download_fn = download_fn or _DL.download_model
    # E3/E4-fix: the boot lifecycle is a CONTEXT MANAGER factory — the
    # server stays ALIVE for the `with` body (smoke + capture run against a
    # live server); teardown is on __exit__, AFTER capture. Default =
    # booted_derived. Injectable so test-pull.sh stays mock-only.
    boot_cm = boot_cm or _B.booted_derived
    smoke_fn = smoke_fn or _CAP.smoke_derived
    capture_fn = capture_fn or _CAP.emit_capture
    override_capture_fn = override_capture_fn or _CAP.emit_override_capture
    root = root or REPO_ROOT

    # ----- CONTRACT-1: populate EInput from in-scope state ----------------
    ei, topo_err = _build_einput(
        slug=slug, der=der, s2=s2, c2a=c2a, terminal=terminal,
        is_override_accepted=is_override_accepted, hf_home=hf_home,
        hardware_sm=hardware_sm, gpu_topology=gpu_topology, root=root,
        fetcher=fetcher,
    )
    if ei is None:
        res.notices.append(
            f"[E] not run: {topo_err} (no nvidia-smi and no GPU-topology "
            f"override; refusing to fabricate a topology — verdict stands)"
        )
        res.diagnostics["e_stage"] = {"ran": False, "reason": topo_err}
        return

    # ----- CONTRACT-5: derived-emittable pre-[E] precondition -------------
    try:
        ok, reason = gc.derived_emittable(ei)
    except Exception as exc:  # pragma: no cover - gate is pure
        ok, reason = False, f"derived-runtime-unsupported:gate-error:{exc!r}"
    if not ok:
        res.notices.append(
            f"[E] CONTRACT-5 refuse: {reason} — NO download / NO boot"
        )
        res.diagnostics["e_stage"] = {
            "ran": False, "contract5_refuse": reason,
        }
        return

    # E2's standardized header-probe fetcher wiring (no-op on the dtype
    # resolution ORDER; just supplies the object E1 looks for).
    try:
        _DL.set_probe_fetcher(ei, fetcher)
    except Exception:  # pragma: no cover - defensive
        pass

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage = {"ran": True, "contract5_refuse": None}

    # ----- E1: generate_from_profile --------------------------------------
    try:
        compose_text, compose_meta = emit_fn(root, ei)
    except gc.Refuse as r:
        res.notices.append(f"[E] generate refuse: {r}")
        stage["emit_ok"] = False
        res.diagnostics["e_stage"] = stage
        return
    except Exception as exc:
        res.notices.append(f"[E] generate failed: {exc!r}")
        stage["emit_ok"] = False
        res.diagnostics["e_stage"] = stage
        return
    stage["emit_ok"] = True

    # ----- E2: download_model ---------------------------------------------
    try:
        dl = download_fn(ei, fetcher=fetcher)
    except TypeError:
        # an injected mock may not accept the fetcher kwarg.
        dl = download_fn(ei)
    except Exception as exc:
        res.notices.append(f"[E] download failed: {exc!r}")
        res.download_ok = False
        stage["download_failure"] = repr(exc)
        res.diagnostics["e_stage"] = stage
        return
    res.download_ok = bool(getattr(dl, "ok", False))
    if not res.download_ok:
        res.notices.append(
            f"[E] download not ok: failure="
            f"{getattr(dl, 'failure', None)!r}"
        )

    # ----- E3/E4-fix: boot -> smoke -> capture, ALL inside the live-server
    #       context manager. The server is ALIVE for the entire `with` body
    #       (booted_derived yields the handle while UP); teardown (down +
    #       rmtree) happens on `__exit__`, AFTER capture+pt5, ALWAYS (the
    #       CM's own finally — no-orphan guarantee preserved at the CORRECT
    #       scope). On-rig E5 caught the prior teardown-in-finally-before-
    #       smoke defect: smoke ran against an already-destroyed server ->
    #       ConnectionRefused, every time. Now smoke runs while UP.
    #
    #       The whole `with` is wrapped so a smoke/capture exception is
    #       caught structurally here (never raises out of run_pull — every
    #       pre-existing test-pull.sh assertion byte-unaffected); the CM's
    #       teardown STILL runs on the exceptional __exit__ before this
    #       except is reached. ------------------------------------------
    try:
        with boot_cm(ei, compose_text) as bt:
            res.boot_ok = bool(getattr(bt, "ok", False))

            # --- E3: smoke_derived (server is UP here) -------------------
            sm = _CAP.SmokeResult()
            endpoint = getattr(bt, "endpoint", None)
            if res.boot_ok and endpoint:
                try:
                    sm = smoke_fn(ei, endpoint)
                except Exception as exc:  # pragma: no cover - smoke defensive
                    res.notices.append(f"[E] smoke failed: {exc!r}")
            res.smoke = {
                "smoke_capability_set": list(
                    getattr(sm, "smoke_capability_set", []) or []
                ),
                "results": dict(getattr(sm, "results", {}) or {}),
                "partial": bool(getattr(sm, "partial", False)),
            }

            # --- E3: emit_capture (pt1-4 + manifest) — emitted while/just
            #     after smoke, BEFORE teardown (which is the CM __exit__) --
            try:
                cap = capture_fn(
                    ei,
                    confidence=SimpleNamespace(name=str(conf)),
                    raw_verdict=raw,
                    profile_like=profile_like,
                    download_result=dl,
                    boot_result=bt,
                    smoke_result=sm,
                    compose_meta=compose_meta,
                    kv_calc_version=_KV_CALC_VERSION,
                    repo_root=root,
                    ts=ts,
                    # F3/G6-A — PURE pass-through wiring (NO decision-logic
                    # change; gate/terminal/[B]/[C1]/is_override_accepted
                    # untouched). A-i: the predicted [B] breakdown that
                    # already exists at pull.py:638 (== raw_verdict()
                    # ["breakdown_gb"]), now persisted into pt1 on every
                    # post-[B] capture (was disk-only in pt5). A-ii: the
                    # booter's bounded redacted container-log excerpt
                    # (None on success / when unavailable — emit_capture
                    # writes failure_log_excerpt + parses pt3.actual ONLY
                    # when boot is not ok).
                    predicted_b_breakdown=res.diagnostics.get(
                        "b_breakdown"
                    ),
                    failure_log_excerpt=getattr(
                        bt, "failure_log_excerpt", None
                    ),
                )
                res.capture_paths = list((cap.get("paths") or {}).values())
                res.diagnostics["capture_dir"] = cap.get("dir")
            except Exception as exc:
                res.notices.append(f"[E] capture emit failed: {exc!r}")
                stage["capture_ok"] = False
                res.diagnostics["e_stage"] = stage
                # Leave the `with` -> CM teardown still runs on __exit__.
                return
            stage["capture_ok"] = True

            # --- CONTRACT-4 pt5: override-accepted force-capture (§5.3) --
            # Emitted ONLY when is_override_accepted (the §5.3 trigger).
            # The literal calibration_signal_not_validated:true is enforced
            # inside the emitter. Still inside the `with` (before teardown).
            if is_override_accepted:
                boot_peak = None
                gpu_worker = None
                exit_summary = None
                if not res.boot_ok:
                    # boot never reached allocation -> actual null; the WHY
                    # is the boot failure reason (CONTRACT-4 pt5).
                    exit_summary = getattr(bt, "failure", None) or (
                        "boot did not reach allocation"
                    )
                try:
                    p5 = override_capture_fn(
                        ei,
                        predicted_b_breakdown=res.diagnostics.get(
                            "b_breakdown"
                        ),
                        boot_peak_mib=boot_peak,
                        gpu_worker_reported_mib=gpu_worker,
                        exit_error_summary=exit_summary,
                        repo_root=root,
                        ts=ts,
                    )
                    res.capture_paths.append(p5)
                    res.notices.append(
                        "[E] §5.3 override-accepted force-capture (pt5) "
                        "emitted (calibration signal, NOT fit-validated)"
                    )
                except Exception as exc:  # pragma: no cover - emitter
                    res.notices.append(
                        f"[E] pt5 override-capture failed: {exc!r}"
                    )
        # <- CM __exit__ HERE: runner.down + rmtree ALWAYS ran (success OR
        #    exception in the with-body), AFTER smoke + capture + pt5.
    except Exception as exc:
        # Boot CM raised (defensive — booted_derived yields an ok=False
        # handle for an EXPECTED boot failure, so this is the unexpected
        # path). The CM's finally already tore down. Record structurally;
        # NEVER raise out of run_pull.
        res.notices.append(f"[E] boot failed: {exc!r}")
        if res.boot_ok is None:
            res.boot_ok = False

    res.diagnostics["e_stage"] = stage


# kv-calc has no version constant; this is the stable label E4 stamps into
# the §6.2 capture manifest's `kv_calc_version` (a [F] consensus-key input).
# Kept here (pull.py, the orchestrator) — NOT a frozen-module edit.
#
# CONTRACT-5 (ii) — G2: a calibration-affecting kv-calc math/corpus edit
# that does NOT change this string lets materially-different runs collide on
# the §6.2 consensus key / §6.3 dedup key / submission_fingerprint. Removing
# that defect by *deriving* the version from the actual content surface
# instead of a hand-bumped literal: `kvcalc-v0.8.0+<sha256(content)[:12]>`
# where <content> = the bytes of `tools/kv-calc.py` (the kv-calc math) + the
# sorted concatenation of `scripts/lib/profiles/calibration/*.yml` (the
# calibration corpus). A math OR corpus edit now changes the hash
# automatically — no manual bump discipline needed (closes G2; satisfies the
# (iii) regression). This is purely how the version STRING is computed: zero
# decision-logic touch (gate states / terminals / [B] / [C1] / verdicts /
# is_override_accepted / the §4.1 table are all untouched).
#
# Hashing convention is parity with `[E]`/§6.2's `submission_fingerprint`
# (`capture.py` `_fingerprint`): sha256 over `\x1f`-joined parts, utf-8.
# Computed ONCE at import and cached in the module-level constant (never
# recomputed per call). If the content files are unreadable for ANY reason
# the producer degrades to the bare `"kvcalc-v0.8.0"` and NEVER raises —
# `[E]`/pull must not crash over a provenance-label refinement.
_KV_CALC_VERSION_BASE = "kvcalc-v0.8.0"


def _kv_calc_content_hash(repo_root: Path) -> Optional[str]:
    """sha256[:12] over the kv-calc math + sorted calibration corpus.

    <content> parts, in this fixed order:
      1. raw bytes of `tools/kv-calc.py`
      2. for each `scripts/lib/profiles/calibration/*.yml` sorted by name:
         the file NAME then its raw bytes (name included so a rename / add /
         remove of a corpus file also moves the hash).
    Parts joined with the `\\x1f` unit-separator (same convention as
    `capture.py`'s `submission_fingerprint`), sha256, first 12 hex.
    Returns None (caller degrades to the bare base) if anything is
    unreadable — robust, never raises."""
    import hashlib

    try:
        kv_calc = (repo_root / "tools" / "kv-calc.py").read_bytes()
        calib_dir = repo_root / "scripts" / "lib" / "profiles" / "calibration"
        calib_files = sorted(
            calib_dir.glob("*.yml"), key=lambda p: p.name
        )
        parts: list[bytes] = [kv_calc]
        for cf in calib_files:
            parts.append(cf.name.encode("utf-8"))
            parts.append(cf.read_bytes())
        h = hashlib.sha256()
        h.update(b"\x1f".join(parts))
        return h.hexdigest()[:12]
    except Exception:  # pragma: no cover - robust degrade, never crash [E]
        return None


def _derive_kv_calc_version(repo_root: Path) -> str:
    """CONTRACT-5 (ii): `kvcalc-v0.8.0+<12hex>` from content, or the bare
    base when the content surface is unreadable (honest degrade)."""
    digest = _kv_calc_content_hash(repo_root)
    if digest is None:
        return _KV_CALC_VERSION_BASE
    return f"{_KV_CALC_VERSION_BASE}+{digest}"


# Computed once at import from the REAL repo content surface; cached here so
# every [E] emit stamps the identical derived label (the per-call site at
# the `capture_fn(... kv_calc_version=_KV_CALC_VERSION ...)` call is byte-
# unchanged — it still reads this module-level name).
_KV_CALC_VERSION = _derive_kv_calc_version(REPO_ROOT)


def _short_refuse(msg: str) -> str:
    """Compact a [D] Refuse message into a stable machine token."""
    low = msg.lower()
    if "loads" in low or "engine pin" in low:
        return "pin-mismatch"
    if "tp=" in low or "kv_format" in low:
        return "tp-or-kv"
    if "trust_remote_code" in low or "security refusal" in low:
        return "trc"
    if "foundational" in low:
        return "foundational-drift"
    if "degraded" in low:
        return "degraded-drift"
    if "out of scope" in low or "genesis" in low:
        return "scope"
    return "other"


# Per-family map: which weight field(s) `kv._weights_per_card_gb` reads on
# the DEFAULT (weights_variant="default") path that `[B]`/raw_verdict uses.
# We substitute the actually-resolved curated-variant blob size onto exactly
# these field(s) so pricing reflects the resolved variant WITHOUT ever
# touching `model_family` (which selects the authoritative family pricing
# branch in kv-calc). Keyed by the curated-exact spec's own model_family.
_FAMILY_WEIGHT_FIELDS = {
    "qwen3-next-hybrid": ("weights_total_gb",),
    "qwen3-next-moe": ("weights_total_gb",),
    "gemma4-swa-dense": ("weights_int4_gb",),
    "gemma4-swa-moe": ("weights_int4_gb", "weights_awq_gb"),
}


def _curated_spec(profiles, der) -> dict:
    """Build the **curated-exact** kv-calc spec for a Tier-1 curated hit so
    P1's raw_verdict prices it through the model's authoritative
    family-specific branch (NOT generic-dense's conservative lower-bound).

    Start from `kv.MODEL_SPECS[der.tier1.model_id]` — the SAME specs
    `tools/kv-calc.py --calibration` validates at 22/22 — preserving its real
    `model_family` and every family-specific field, then substitute ONLY the
    weight size for the actually-resolved variant (`der.tier1.weights_variant`)
    read from the curated ModelProfile (the ModelProfile remains the weight
    authority; we never recompute weight size). NEVER emit
    `model_family="generic-dense"` for a curated model."""
    t1 = der.tier1
    kv = _kv()
    model_specs = getattr(kv, "MODEL_SPECS", None)
    if not isinstance(model_specs, dict) or t1.model_id not in model_specs:
        raise RuntimeError(
            f"_curated_spec: Tier-1 model_id {t1.model_id!r} not in "
            f"kv.MODEL_SPECS (keys={sorted((model_specs or {}).keys())}). "
            f"This is a hard error — refusing to fall back to generic-dense "
            f"for a curated model."
        )
    # Defensive copy of the authoritative curated-exact spec (preserve
    # model_family + all family fields verbatim — do NOT recompute/adjust).
    spec = dict(model_specs[t1.model_id])
    family = spec.get("model_family")
    if family == "generic-dense" or not family:
        raise RuntimeError(
            f"_curated_spec: curated-exact spec for {t1.model_id!r} has "
            f"unexpected model_family={family!r}; refusing to price a "
            f"curated model as generic-dense."
        )

    # Resolved-variant weight blob size from the curated ModelProfile
    # (authoritative; same source the old code read).
    model = profiles.models[t1.model_id]
    vmeta = model.weights.get(t1.weights_variant, {}) or {}
    size_gb = (
        vmeta.get("size_gb")
        or (der.profile or {}).get("weights_variant_size_gb")
    )
    weight_fields = _FAMILY_WEIGHT_FIELDS.get(family)
    if weight_fields is None:
        raise RuntimeError(
            f"_curated_spec: no weight-field mapping for curated family "
            f"{family!r} ({t1.model_id!r}). STOP — do not silently fall "
            f"back to generic-dense."
        )
    # Only substitute when the curated ModelProfile gives a usable numeric
    # blob size for the resolved variant. If it does not (e.g. "variable"
    # GGUF), keep the authoritative spec's own validated weight size rather
    # than zeroing it — never weaken the curated-exact pricing.
    if isinstance(size_gb, (int, float)) and float(size_gb) > 0:
        for f in weight_fields:
            spec[f] = float(size_gb)
    spec["model_id"] = t1.slug
    return spec


# ===========================================================================
# v0.8.2 CONTRACT-4 — the `recommend` UX (success-path counterpart to the
# CONTRACT-1 failure on-ramp).
#
# This is PURE PRESENTATION / AGGREGATION over the SHIPPED `run_pull`
# verdict. It introduces NO decision logic: every field it renders is read
# straight off the real `PullResult` (`ok` / `confidence` / `raw_verdict` /
# `terminal` / `stratum` / `abort_reason` / `notices` / `emitted`) that the
# locked 6-stratum + §4.1 machinery produced for THIS run. The recommendation
# therefore tracks the real verdict by construction — a model that fits, one
# that hard-blocks, and one that is `confirm→proceed` each yield a different
# block because each carries a different real `res`. It is NOT a static or
# templated summary: remove the gate and the recommendation changes with it.
#
# Honesty invariants (CONTRACT-4):
#   * vLLM-only by construction (the shipped gate is vLLM-only; we add
#     nothing — we only echo what the gate decided).
#   * Carries the §7 boot-fit≠runtime caveat verbatim (CAVEAT_S7) + the
#     soak-continuous pointer, on any verdict the gate itself marked
#     boot-fit-satisfied (detected via CAVEAT_S7 ∈ res.notices — the SAME
#     signal the gate already emits; we do not re-derive it).
#   * States WHICH gate decided (`res.stratum`) — never silently summarised.
#   * Honest about the confidence tier (echoes res.confidence; an
#     estimated-lower-bound floor is stated, never hidden).
#   * NEVER implies a non-emitted artifact: a compose line is printed ONLY
#     when `res.emitted` is true (Path A actually emitted one).
# ===========================================================================
def _render_recommendation(res: "PullResult") -> list[str]:
    """Aggregate the SHIPPED `run_pull` verdict into an honest, human
    recommendation. Pure presentation: derives every line from `res`;
    introduces no decision logic. Returns the lines to print (so the
    test can assert the recommendation tracks a *real* differing verdict
    without driving a tty)."""
    out: list[str] = ["[recommend] ----- recommendation -----"]
    out.append(
        f"[recommend] model={res.slug} profile-like={res.profile_like} "
        f"engine=vLLM (the pull gate is vLLM-only by construction)"
    )

    if res.ok:
        # The gate said download-eligible / clean. Echo the real terminal +
        # confidence; never upgrade the gate's honesty.
        verdict = res.terminal or "proceed"
        out.append(
            f"[recommend] verdict: FITS → {verdict} "
            f"(confidence={res.confidence or 'n/a'}, "
            f"raw_verdict={res.raw_verdict or 'n/a'})"
        )
        if res.confidence == "estimated-lower-bound":
            out.append(
                "[recommend] note: this is an ESTIMATED LOWER BOUND — the "
                "modelled VRAM is a floor and is likely under-modelled; "
                "treat the fit as the optimistic edge, not a guarantee."
            )
        if res.emitted:
            out.append(
                "[recommend] a ready compose was emitted for this config "
                "(see the --out path / stdout above)."
            )
        else:
            # Honesty: do NOT imply a compose that was not produced.
            out.append(
                "[recommend] no compose was emitted this run (Path B "
                "evaluate / --dry-run / not download-eligible) — this is a "
                "verdict only, not a launchable artifact."
            )
    else:
        # `res.ok` is False, but that means "the run did not produce a
        # launchable artifact" — NOT necessarily "does not fit". Distinguish
        # the gate's *acceptance* terminals (it fits the modelled budget; the
        # run merely needs an explicit flag) from a genuine hard-block.
        # Honesty (CONTRACT-4): never label a fits-clean model "DOES NOT FIT".
        # Pure presentation: still derives only from `res`, no decision logic.
        reason = res.abort_reason or "see the gate output above"
        _rv = res.raw_verdict or ""
        _ar = res.abort_reason or ""
        _needs_accept = _rv == "fits-clean" and (
            _ar.startswith("confirm→proceed")
            or _ar.startswith("override-accepted")
            or "needs --yes" in _ar
            or "needs --force-download" in _ar
        )
        if _needs_accept:
            out.append(
                f"[recommend] verdict: FITS (estimated) — NOT YET ACCEPTED: "
                f"{reason} (confidence={res.confidence or 'n/a'}, "
                f"raw_verdict={res.raw_verdict or 'n/a'})"
            )
            out.append(
                "[recommend] the model fits the gate's modelled budget — this "
                "is an ACCEPTANCE gate, not a fit failure. Re-run with the "
                "flag named in the reason above to proceed."
            )
            if res.confidence == "estimated-lower-bound":
                out.append(
                    "[recommend] note: this is an ESTIMATED LOWER BOUND — the "
                    "modelled VRAM is a floor and is likely under-modelled; "
                    "treat the fit as the optimistic edge, not a guarantee."
                )
        else:
            # Genuine hard-block / needs-a-flag. Carry the precise reason and
            # WHICH gate decided; point at the failure on-ramp (CONTRACT-1).
            out.append(
                f"[recommend] verdict: DOES NOT FIT / BLOCKED — {reason}"
            )
            out.append(
                "[recommend] this is an honest stop, not a stack failure. The "
                "diagnostics (if a redacted bundle was captured) can be sent "
                "back with `scripts/pull.sh --submit-last` — see the failure "
                "on-ramp section of docs/PULL.md."
            )

    # WHICH gate decided — always stated (CONTRACT-4: never silently
    # summarised). Read straight off the real result.
    out.append(
        f"[recommend] decided at: stratum={res.stratum.name}"
    )

    # §7 boot-fit ≠ runtime caveat + soak-continuous pointer — carried
    # verbatim ONLY when the gate itself marked this run boot-fit-satisfied
    # (CAVEAT_S7 ∈ res.notices is the gate's own signal; we echo, not
    # re-derive). A blocked verdict that never reached [B] has no such
    # notice and (correctly) gets no false soak pointer.
    if CAVEAT_S7 in res.notices:
        out.append(f"[recommend] §7 caveat: {CAVEAT_S7}")
        out.append(
            "[recommend] before relying on this in production run: "
            "scripts/soak.sh SOAK_MODE=continuous (the only test that "
            "catches Cliff 2b)."
        )
    out.append("[recommend] -------------------------------")
    return out


# ===========================================================================
# CLI (thin; pull.sh execs this). Renders PullResult to stdout + exit code.
# ===========================================================================
_EXIT_OK = 0
_EXIT_ABORT = 2          # any honest hard-stop (stratum 1-6 / hard-block)
_EXIT_NEEDS_FLAG = 3     # confirm→proceed / advisory not yet satisfied
_EXIT_USAGE = 64


def _maybe_submit_verb(argv: list[str]) -> Optional[int]:
    """v0.8.2 CONTRACT-1.3 — the `--submit-last` / `--submit <dir>` verb is
    a DISTINCT top-level verb, parsed BEFORE the positional-slug /
    `--profile-like` requirement (so `--submit*` needs NEITHER a slug NOR
    `--profile-like`). Returns an exit code when the submit verb was
    invoked, else `None` (fall through to the normal gate CLI).

    This is intentionally a pre-argparse intercept: the shipped gate parser
    hard-requires `slug` + `--profile-like`, and the submit verb is a
    different invocation grammar entirely (the on-ramp, not a gate run).
    The submit path is the ONLY place network ever happens, and only after
    an explicit `y` (the locked `[F]`-offline boundary — `run_pull` itself
    stays I/O-free).
    """
    if "--submit-last" not in argv and "--submit" not in argv:
        return None

    from scripts.lib.profiles.submit_pull import submit_pull

    submit_last = "--submit-last" in argv
    capture_dir: Optional[str] = None
    if "--submit" in argv:
        i = argv.index("--submit")
        if i + 1 < len(argv):
            capture_dir = argv[i + 1]
    return submit_pull(
        capture_dir=capture_dir,
        submit_last=submit_last,
        repo_root=REPO_ROOT,
    )


def main(argv: list[str]) -> int:
    import argparse

    # v0.8.2 CONTRACT-1.3 — the submit verb is parsed BEFORE the gate
    # parser's slug/--profile-like requirement (a distinct top-level verb).
    _sub_rc = _maybe_submit_verb(argv)
    if _sub_rc is not None:
        return _sub_rc

    class _UsageExit64Parser(argparse.ArgumentParser):
        """argparse's default `error()` hard-exits 2 — which collides with
        `_EXIT_ABORT` (honest gate hard-stop), so a typo and a legitimate
        block are indistinguishable to callers/automation. Override to exit
        `_EXIT_USAGE` (64) on argument/usage errors, restoring the
        documented contract. `--help` is unaffected (it goes through
        `exit()`, not `error()`, and still returns 0)."""

        def error(self, message):
            self.print_usage(sys.stderr)
            self.exit(_EXIT_USAGE, f"{self.prog}: error: {message}\n")

    ap = _UsageExit64Parser(
        prog="pull.sh",
        description="v0.8.0 Pull-Gate — derive an HF repo, gate it through "
        "the locked 6-stratum taxonomy, and (Path A, curated+emittable) "
        "emit a compose via the #141 generator. Honest about confidence; "
        "never silently gate-passes.",
    )
    ap.add_argument("slug", help="HF repo slug (e.g. org/Model-Name)")
    ap.add_argument(
        "--profile-like", required=True, dest="profile_like",
        help="REQUIRED curated COMPOSE_REGISTRY key supplying the runtime "
        "shape (Path A: must name the curated model+variant & be "
        "[D]-emittable; Path B: any vLLM profile, shape only)",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="force Path B (evaluate only; never emit/download)")
    ap.add_argument("--yes", action="store_true",
                    help="accept a confirm→proceed terminal (§4.1)")
    ap.add_argument(
        "--force-download", action="store_true",
        help="advisory low-confidence wont-fit → override-accepted "
        "(NO-OP + notice this phase; download deferred to Loop)",
    )
    ap.add_argument(
        "--experimental-arch", action="store_true",
        help="bypass ONLY [C0] engine-support-unknown/no-arch-row "
        "(never runtime-incompatible; Path B only this phase)",
    )
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="bypass [C0] needs-trust-remote-code-ack")
    ap.add_argument("--hf-home", help="override the HF_HOME resolution chain")
    ap.add_argument("--out", help="Path A: write the emitted compose here")
    ap.add_argument(
        "--hardware", type=float, default=None,
        help="override detected GPU compute capability (e.g. 8.6 for "
        "RTX 3090); default = nvidia-smi detection",
    )
    ap.add_argument(
        "--hardware-gpus", default=None,
        help="v0.8.0 [E]: override detected GPU topology for the derived "
        "[E] stage as 'VRAM_MIB:NAME,VRAM_MIB:NAME' (e.g. "
        "'24576:RTX 3090,24576:RTX 3090'); default = nvidia-smi detection",
    )
    ap.add_argument(
        "--recommend", action="store_true",
        help="v0.8.2 CONTRACT-4: after the gate runs, print an honest "
        "aggregated recommendation (fits? which quant/variant; the §7 "
        "boot-fit≠runtime caveat; which gate decided). Presentation only "
        "— derives entirely from the real gate verdict, changes no "
        "decision; vLLM-only by construction",
    )
    args = ap.parse_args(argv)

    gpu_topology = None
    if args.hardware_gpus:
        vram: list[int] = []
        names: list[str] = []
        for tok in args.hardware_gpus.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                v, n = tok.split(":", 1)
            else:
                v, n = tok, "GPU"
            vram.append(int(float(v)))
            names.append(n.strip() or "GPU")
        if vram:
            gpu_topology = (len(vram), vram, names)

    res = run_pull(
        args.slug, args.profile_like,
        dry_run=args.dry_run, yes=args.yes,
        force_download=args.force_download,
        experimental_arch=args.experimental_arch,
        trust_remote_code=args.trust_remote_code,
        hf_home=args.hf_home, out=args.out,
        hardware_sm=args.hardware,
        gpu_topology=gpu_topology,
    )

    tag = "OK" if res.ok else "ABORT"
    print(f"[pull] {tag} path={res.path} stratum={res.stratum.name} "
          f"slug={res.slug} profile-like={res.profile_like}")
    if res.confidence:
        print(f"[pull] confidence={res.confidence} "
              f"raw_verdict={res.raw_verdict} terminal={res.terminal}")
    if res.abort_reason:
        print(f"[pull] reason={res.abort_reason}")
    if res.detail:
        print(f"[pull] {res.detail}")
    for n in res.notices:
        print(f"[pull] note: {n}")
    if res.emitted and not args.out:
        sys.stdout.write(res.compose_text or "")

    # v0.8.2 CONTRACT-1.2 — surface the failure on-ramp pointer (gate path,
    # I/O-FREE: a single print; NO network, NO blocking prompt, NO auto-send
    # — submission is the separate, explicit, consented `--submit-last`
    # step). The TRIGGER is "a gate bundle was emitted for this run", keyed
    # on the capture dir the V1 pass-through recorded on `res.diagnostics`
    # — EXPLICITLY NOT gated on the exit code. The single most important
    # §10-R9 lever (`engine-support-unknown/no-arch-row`) is the
    # `--experimental-arch`-bypassable advisory path that exits 0 yet emits
    # a bundle; gating this pointer on `exit==2` would silently suppress the
    # submit prompt for exactly that class. It does NOT classify (that is
    # loop-side at submit) — it prints unconditionally on any emitted gate
    # bundle; suppression (no public issue) is enforced loop-side. The
    # surfaced message points ONLY at the redacted `.pull-captures/`
    # artifact + the exact one-command — never "paste your terminal output"
    # (console is not a safe submission source).
    _gc = res.diagnostics.get("gate_capture") if res.diagnostics else None
    _gc_dir = _gc.get("dir") if isinstance(_gc, dict) else None
    if _gc_dir:
        print(
            f"[pull] Diagnostics captured (redacted, no paths/tokens): "
            f"{_gc_dir}"
        )
        print(
            "[pull] Help improve the fit math — submit with: "
            "scripts/pull.sh --submit-last"
        )

    # v0.8.2 CONTRACT-4 — the `recommend` UX. Pure presentation/aggregation
    # over the SHIPPED verdict (`res`); changes NO decision and NOT the exit
    # code (the recommendation only RENDERS what the gate already decided).
    # Emitted last so it summarises the full verdict the user just saw.
    if args.recommend:
        for ln in _render_recommendation(res):
            print(ln)

    if res.ok:
        return _EXIT_OK
    if res.abort_reason and (
        res.abort_reason.startswith("confirm→proceed")
        or res.abort_reason.startswith("override-accepted")
    ):
        return _EXIT_NEEDS_FLAG
    return _EXIT_ABORT


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
