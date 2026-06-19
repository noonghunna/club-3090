"""Cockpit data models — the typed shapes the panes consume.

This module is **pure**: no subprocess, no I/O, no Textual.  It defines the
dataclasses produced by ``services.CockpitData`` and the small parsing helpers
that turn raw contract output (JSON dicts / health.sh text) into those shapes.

Keeping these here (separate from ``services.py``) lets the panes and the tests
import the shapes without dragging in the subprocess machinery, and lets the
service layer be fully dependency-injected against a fake runner.

The enriched catalog row (``CatalogEntry``) wraps the shared-core ``VariantRow``
(never re-implements it) and layers on the join results: the local-card fit
verdict, measured TPS / 8-pack, and provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from club3090_tui_core.registry import VariantRow

# ── Fit verdict ───────────────────────────────────────────────────────────────


@dataclass
class FitVerdict:
    """Result of kv-calc --fit / switch.sh --explain's fit block for one slug."""

    # Real kv-calc --fit verdict enum (verified live):
    #   fits-clean | fits-constrained | wont-fit | unknown
    # plus the cockpit-internal "skip" (ik/llama kvcalc_key=SKIP — no vLLM fit).
    verdict: str = "unknown"          # fits-clean | fits-constrained | wont-fit | unknown | skip
    vram_est_gb: Optional[float] = None
    band_gb: Optional[float] = None
    max_ctx: Optional[int] = None
    card: str = ""
    error: str = ""

    # Compact glyph for the Catalog "fit" column.
    @property
    def glyph(self) -> str:
        return {
            "fits-clean": "●",
            "fits-constrained": "◐",
            "wont-fit": "○",
            "skip": "·",
            "unknown": "·",
        }.get(self.verdict, "·")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, card: str = "") -> "FitVerdict":
        if not d:
            return cls(card=card)
        return cls(
            verdict=str(d.get("verdict", "unknown")),
            vram_est_gb=_as_float(d.get("vram_est_gb")),
            band_gb=_as_float(d.get("band_gb")),
            max_ctx=_as_int(d.get("max_ctx")),
            card=card,
            error=str(d.get("error", "")),
        )


# ── Measurement (TPS / 8-pack) ──────────────────────────────────────────────────


@dataclass
class Measurement:
    """A measured result for a slug, joined from a structured corpus or parsed
    coarsely from BENCHMARKS.md.  ``source`` records provenance so the UI can
    distinguish a structured record from a best-effort markdown parse."""

    narr_tps: Optional[float] = None
    code_tps: Optional[float] = None
    quality_8pk: Optional[str] = None   # e.g. "107/150"
    max_ctx_label: str = ""
    date: str = ""
    source: str = ""                    # "explain" | "corpus" | "benchmarks.md" | ""

    @property
    def tps_label(self) -> str:
        if self.narr_tps is None and self.code_tps is None:
            return "—"
        n = f"{self.narr_tps:.0f}" if self.narr_tps is not None else "—"
        c = f"{self.code_tps:.0f}" if self.code_tps is not None else "—"
        return f"{n}/{c}"

    @property
    def quality_label(self) -> str:
        return self.quality_8pk or "—"


# ── Enriched catalog entry ──────────────────────────────────────────────────────


@dataclass
class CatalogEntry:
    """A registry VariantRow enriched with fit + measurement + provenance.

    ``row`` is the shared-core dataclass verbatim; the cockpit never mutates it.
    """

    row: VariantRow
    fit: FitVerdict = field(default_factory=FitVerdict)
    measurement: Measurement = field(default_factory=Measurement)

    # Convenience pass-throughs (so panes can read entry.slug, not entry.row.slug)
    @property
    def slug(self) -> str:
        return self.row.slug

    @property
    def engine(self) -> str:
        return self.row.engine

    @property
    def model(self) -> str:
        return self.row.model

    @property
    def status(self) -> str:
        return self.row.status

    @property
    def status_note(self) -> str:
        return self.row.status_note

    @property
    def ctx_label(self) -> str:
        return self.row.ctx_label

    @property
    def port(self) -> int:
        return self.row.port

    @property
    def source(self) -> str:
        """Provenance string for the catalog 'source' column (registry source
        field, e.g. 'curated' / 'community' / 'local')."""
        return getattr(self.row, "source", "") or "·"


# ── Estate / Scene / Container / Doctor ─────────────────────────────────────────


@dataclass
class Scene:
    """One gpu-mode scene from --list-modes --json."""

    name: str
    group: str = ""
    description: str = ""
    services: list[str] = field(default_factory=list)
    ports: list[str] = field(default_factory=list)
    gpus: str = ""                      # "none" | "0" | "both" | etc.

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scene":
        return cls(
            name=str(d.get("name", "")),
            group=str(d.get("group", "")),
            description=str(d.get("description", "")),
            services=list(d.get("services", []) or []),
            ports=[str(p) for p in (d.get("ports", []) or [])],
            gpus=str(d.get("gpus", "")),
        )


@dataclass
class ContainerInfo:
    """A running stack container that can hold a GPU, from docker ps.

    ``kind`` is one of:
      - ``"engine"``  — a core inference engine (``vllm-`` / ``llama-cpp-`` /
        ``ik-llama-`` / ``sglang-`` / ``beellama-``); ``slug`` is registry-matched.
      - ``"estate"``  — an estate-planner container (``club3090-<name>``).
      - ``"service"`` — a GPU-holding rig service (ComfyUI / Step-Audio).
    """

    name: str
    kind: str = "service"               # "engine" | "estate" | "service"
    host_port: int = 0
    internal_port: int = 0
    engine: str = ""                    # for engine containers
    slug: str = ""                      # registry slug if matched
    gpus: str = ""                      # "0,1" if known, else ""
    status: str = "running"             # "running" | "stopped" (known-but-down service)

    @property
    def is_running(self) -> bool:
        return self.status != "stopped"


@dataclass
class DoctorRead:
    """Parsed runtime-state summary from health.sh (text-only contract).

    health.sh has no --json mode, so this is a deliberately coarse text parse —
    ``raw`` keeps the full output for the pane to render verbatim, and the
    booleans/strings are best-effort signals for the rail/summary line.
    """

    reachable: bool = False
    serving: bool = False
    summary: str = ""                   # one-line condensed status
    kv_pool_pct: Optional[int] = None
    spec_dec: str = ""                  # e.g. "MTP n=2, 73% accept" or ""
    recent_errors: Optional[int] = None
    raw: str = ""
    parse_source: str = "health.sh-text"


@dataclass
class EstateState:
    """Live estate snapshot: detect + doctor + scene catalog + estate-planner."""

    target: Any = None                  # core ServingTarget (or None)
    gpus: list[Any] = field(default_factory=list)   # core GpuInfo list
    containers: list[ContainerInfo] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    doctor: DoctorRead = field(default_factory=DoctorRead)
    estate_report: dict[str, Any] = field(default_factory=dict)   # estate_cli report-state
    matched_slug: str = ""              # slug the running engine matched, if any
    error: str = ""


# ── Reconcile gate ──────────────────────────────────────────────────────────────


@dataclass
class GpuConflict:
    """A live GPU user that a pending write would collide with."""

    gpu_index: int
    mem_used_mib: int
    container: str = ""                 # container occupying it, if known
    note: str = ""


@dataclass
class ReconcileResult:
    """Result of reconcile_before_write() — the dual-writer safety gate.

    ``safe`` is True only when no running container / GPU user would collide
    with the pending action.  ``conflicts`` and ``gpu_conflicts`` enumerate
    exactly what's in the way so the UI can show "this will tear down X".
    """

    safe: bool
    action: str = ""                    # "serve:<slug>" | "scene:<mode>" | ...
    pending_gpus: list[int] = field(default_factory=list)   # GPUs the action wants
    conflicts: list[ContainerInfo] = field(default_factory=list)
    gpu_conflicts: list[GpuConflict] = field(default_factory=list)
    estate_claims: list[dict[str, Any]] = field(default_factory=list)  # estate instances in the way
    pending_claim_tokens: list[str] = field(default_factory=list)  # in-flight writes (HARD block, non-forceable)
    note: str = ""

    @property
    def conflict_summary(self) -> str:
        parts: list[str] = []
        for c in self.conflicts:
            g = f" (GPU {c.gpus})" if c.gpus else ""
            parts.append(f"{c.name}{g}")
        for e in self.estate_claims:
            parts.append(f"estate:{e.get('name', '?')}")
        return ", ".join(parts) if parts else "none"


# ── BYO check ────────────────────────────────────────────────────────────────────


@dataclass
class ByoResult:
    """Result of pull.sh --profile-like <repo> --dry-run --json."""

    repo: str
    profile_like: str
    arch: str = ""
    eligible: bool = False
    fit_verdict: str = ""
    note: str = ""
    # swap_path block
    route: Optional[str] = None
    sibling_slug: Optional[str] = None
    quant_match: Optional[str] = None
    drop_spec_config: bool = False
    error: str = ""

    @classmethod
    def from_dict(cls, repo: str, profile_like: str, d: dict[str, Any] | None) -> "ByoResult":
        if not d:
            return cls(repo=repo, profile_like=profile_like, error="no output")
        swap = d.get("swap_path") or {}
        return cls(
            repo=repo,
            profile_like=profile_like,
            arch=str(d.get("arch", "")),
            eligible=bool(d.get("eligible", False)),
            fit_verdict=str(d.get("fit_verdict", "")),
            note=str(d.get("note", "")),
            route=swap.get("route"),
            sibling_slug=swap.get("sibling_slug"),
            quant_match=swap.get("quant_match"),
            drop_spec_config=bool(swap.get("drop_spec_config", False)),
        )


# ── Action plans (wired but execution-gated) ─────────────────────────────────────


@dataclass
class ActionPlan:
    """A constructed-but-not-executed write command.

    Action builders return this; runtime execution (only when actually invoked,
    NEVER in tests / this phase) feeds ``cmd`` to the core SubprocessRunner.
    The reconcile gate is consulted BEFORE execution.
    """

    kind: str                           # "serve" | "set_default" | "clear_default" | "scene" | "estate_down" | "container" | "validation" | "submit_bench" | "power_cap" | "power_cap_sweep" | "prune" | "container_rm"
    cmd: list[str]
    description: str = ""
    is_write: bool = True
    requires_reconcile: bool = True
    force: bool = False
    force_reason: str = ""              # required when force=True
    # Phase 4: destructive non-GPU writes (prune, power-cap, submit-bench POST)
    # don't contend for a GPU so requires_reconcile=False, but they MUST still go
    # through a confirm modal.  This flag tells the UI "confirm even though the
    # reconcile gate doesn't apply".  ``network`` flags an outward-facing write
    # (submit-bench POST/PR) so the confirm copy can warn it leaves the rig.
    requires_confirm: bool = True
    network: bool = False


# ── Phase 4: Doctor (estate + profile triage reads) ──────────────────────────────


@dataclass
class EstateDiagnose:
    """Parsed ``diagnose-estate.sh --json`` (estate_cli.py diagnose --json).

    REAL shape (verified live): a top-level object with ``valid``, ``summary``
    (GREEN/AMBER/RED), ``estate_file``, ``live`` (bool), and a ``checks`` block
    holding ``schema`` / ``registry`` / ``per_instance_fits`` / ``cross_checks``
    / ``calibration`` / ``live``.  We keep the raw dict and surface the
    load-bearing top-level signals + a per-instance fit summary for the card.
    """

    valid: bool = False
    summary: str = ""                   # GREEN | AMBER | RED | ""
    estate_file: str = ""
    live: bool = False
    instance_count: int = 0
    instances_valid: int = 0            # how many per_instance_fits are valid
    cross_checks_ok: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "EstateDiagnose":
        if not d:
            return cls(error="no output")
        checks = d.get("checks") or {}
        schema = checks.get("schema") or {}
        fits = checks.get("per_instance_fits") or []
        cross = checks.get("cross_checks") or {}
        return cls(
            valid=bool(d.get("valid", False)),
            summary=str(d.get("summary", "")),
            estate_file=str(d.get("estate_file", "")),
            live=bool(d.get("live", False)),
            instance_count=_as_int(schema.get("instance_count")) or len(fits),
            instances_valid=sum(1 for f in fits if isinstance(f, dict) and f.get("valid")),
            cross_checks_ok=bool(cross.get("ok", False)),
            raw=d,
        )

    @property
    def summary_glyph(self) -> str:
        return {"GREEN": "●", "AMBER": "◐", "YELLOW": "◐", "RED": "○"}.get(self.summary.upper(), "·")


@dataclass
class ProfileTriageStep:
    """One ``[N/6]`` step from diagnose-profile.sh's text output."""

    num: int
    total: int
    name: str
    status: str = "passed"              # passed | failed | warn
    detail: str = ""


@dataclass
class ProfileTriage:
    """Parsed ``diagnose-profile.sh <slug>`` text output.

    diagnose-profile has NO --json mode (verified live) — it is a 6-step text
    triage with ``[N/6] <name>`` headers, ``✓/✗/⚠`` check glyphs, and a final
    ``Triage summary: GREEN|AMBER|RED`` line.  This is a deliberately coarse
    text parse; ``raw`` keeps the full output for verbatim rendering.
    """

    slug: str = ""
    summary: str = ""                   # GREEN | AMBER | RED | ""
    steps: list[ProfileTriageStep] = field(default_factory=list)
    raw: str = ""
    error: str = ""

    @property
    def summary_glyph(self) -> str:
        return {"GREEN": "●", "AMBER": "◐", "YELLOW": "◐", "RED": "○"}.get(self.summary.upper(), "·")

    @property
    def passed(self) -> int:
        return sum(1 for s in self.steps if s.status == "passed")


@dataclass
class DoctorReport:
    """The full Doctor read: health.sh DoctorRead + estate diagnose + profile
    triage.  Each leg is best-effort; a failed leg carries its own error and
    does not fail the others."""

    health: DoctorRead = field(default_factory=DoctorRead)
    estate: EstateDiagnose = field(default_factory=EstateDiagnose)
    profile: Optional[ProfileTriage] = None   # None when no target slug to triage


# ── Phase 4: Benchmarks explorer ──────────────────────────────────────────────────


@dataclass
class BenchRow:
    """One filterable benchmarks row for the explorer.

    Sourced from either the structured #249 measurement-record corpus
    (``source='corpus'`` — authoritative TPS/ctx, no 8-pack) or a coarse
    BENCHMARKS.md scrape (``source='benchmarks.md'`` — carries the 8-pack).
    The pane filters on (model, engine, topology)."""

    model: str = ""
    engine: str = ""
    topology: str = ""
    narr_tps: Optional[float] = None
    code_tps: Optional[float] = None
    max_ctx: str = ""
    quality_8pk: str = ""               # e.g. "109/150" or ""
    date: str = ""
    source: str = ""                    # "corpus" | "benchmarks.md"
    tag: str = ""                       # corpus _tag / md compose token

    @property
    def tps_label(self) -> str:
        if self.narr_tps is None and self.code_tps is None:
            return "—"
        n = f"{self.narr_tps:.0f}" if self.narr_tps is not None else "—"
        c = f"{self.code_tps:.0f}" if self.code_tps is not None else "—"
        return f"{n}/{c}"

    @property
    def quality_label(self) -> str:
        return self.quality_8pk or "—"


# ── Phase 4: Evidence (rebench run tags) ──────────────────────────────────────────


@dataclass
class EvidenceTag:
    """One ``results/rebench/<tag>/`` run directory the Evidence pane lists."""

    tag: str
    path: str = ""
    has_report: bool = False            # REPORT.md present
    has_internal: bool = False          # _internal.json present (#249-shaped)
    has_soak: bool = False              # soak.log / soak-artifacts present
    date: str = ""                      # from REPORT.md Meta or dir mtime
    # A coarse one-line TL;DR scraped from REPORT.md if present.
    tldr: str = ""


@dataclass
class EvidenceReport:
    """A generated paste-ready report for one evidence tag."""

    tag: str
    report_path: str = ""               # REPORT.md path (generated/located)
    body: str = ""                      # the markdown body
    error: str = ""


# ── Phase R / R3b-2: ④ Measure-vs-curated-bar (READ) ──────────────────────────────


@dataclass
class MeasuredNumbers:
    """The producer's MEASURED numbers parsed from a rebench tag dir.

    Best-effort: from ``_internal.json`` (authoritative — the rebench sidecar)
    with a ``REPORT.md`` TL;DR scrape as fallback.  ``narr_tps`` / ``code_tps``
    are decode TPS (the per-token rate, matching how the catalog bar reports);
    ``quality_8pk`` is "<passed>/<total>" when a quality battery was run."""

    narr_tps: Optional[float] = None
    code_tps: Optional[float] = None
    quality_8pk: str = ""               # e.g. "118/150" or ""
    model: str = ""                     # resolved from REPORT.md Meta, best-effort
    engine: str = ""                    # engine-FAMILY of the RUN, from REPORT.md
                                        # Meta (Container / vLLM-image), best-effort
    source: str = ""                    # "_internal.json" | "REPORT.md" | ""

    @property
    def tps_label(self) -> str:
        if self.narr_tps is None and self.code_tps is None:
            return "—"
        n = f"{self.narr_tps:.0f}" if self.narr_tps is not None else "—"
        c = f"{self.code_tps:.0f}" if self.code_tps is not None else "—"
        return f"{n}/{c}"

    @property
    def quality_label(self) -> str:
        return self.quality_8pk or "—"

    @property
    def has_any(self) -> bool:
        return (
            self.narr_tps is not None
            or self.code_tps is not None
            or bool(self.quality_8pk)
        )


@dataclass
class MeasureVsBar:
    """Apples-to-apples comparison of the producer's MEASURED numbers against the
    curated catalog's published bar for the same (model, engine-family) class.

    HONESTY (R3b-2): the cockpit FLAGS the protocol, it does NOT fabricate a
    "catalog-grade" verdict — ``verdict`` is a simple heuristic over whatever
    numbers parsed, and ``protocol_caveats`` lists what the cockpit cannot verify
    (matched power? same harness? same prompts?).  ``bar_source`` discloses where
    the bar came from (the #249 corpus vs the BENCHMARKS.md scrape)."""

    tag: str
    measured: MeasuredNumbers = field(default_factory=MeasuredNumbers)
    bar: Optional[BenchRow] = None      # the matched curated row (None if no match)
    bar_source: str = ""                # "corpus" | "benchmarks.md" | ""
    # The engine-family resolved for THIS run (from REPORT.md Meta / variants),
    # used to discriminate the bar.  "" when it could not be resolved → the bar
    # was picked deterministically on model alone (surfaced as a caveat).
    run_engine: str = ""
    engine_resolved: bool = False       # True when run_engine drove bar selection
    # Per-metric measured−bar deltas (None when either side is missing).
    narr_tps_delta: Optional[float] = None
    code_tps_delta: Optional[float] = None
    verdict: str = "insufficient data"  # see _measure_verdict for the enum
    protocol_caveats: list[str] = field(default_factory=list)
    error: str = ""


# ── Phase 4: Power cap ────────────────────────────────────────────────────────────


@dataclass
class PowerCapGpu:
    """One GPU's power-limit row from ``gpu-mode power-cap status``."""

    index: int
    limit_w: Optional[float] = None
    default_w: Optional[float] = None
    min_w: Optional[float] = None
    max_w: Optional[float] = None


@dataclass
class PowerCapState:
    """Parsed ``gpu-mode power-cap status`` (READ — safe to call live).

    REAL shape (verified live): a banner line then a CSV-ish table
    ``index, power.limit [W], power.default_limit [W], power.min_limit [W],
    power.max_limit [W]`` with one row per GPU (values like ``370.00 W``)."""

    gpus: list[PowerCapGpu] = field(default_factory=list)
    raw: str = ""
    error: str = ""


# ── Phase 4: Container top (docker top — READ) ────────────────────────────────────


@dataclass
class ContainerTop:
    """Parsed ``docker top <name>`` (READ — never mutates the container)."""

    name: str
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    error: str = ""


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 5 — the three v2 hooks (Evaluate · Promote-to-catalog · Optimize)
# ════════════════════════════════════════════════════════════════════════════════


# ── Hook 1: Evaluate (hand the shared ServingTarget to c3t) ───────────────────────


@dataclass
class EvaluateHandoff:
    """The c3t launch hand-off (Estate → ▸ Evaluate).

    Carries the SHARED ``club3090_tui_core.detect.ServingTarget`` (the SAME
    dataclass the test-console speaks — design §4/§6.6) plus the built launch
    ``ActionPlan``.  The launch is a HEAVY, confirm-gated, MOCK-ONLY action this
    phase: c3t runs tests against the live serving model, so the write runner is
    never executed live (conftest blocks the spawn; tests inject a fake).

    ``target`` is held by IDENTITY — the cockpit passes the very object the
    Estate poll detected, so c3t evaluates exactly what's running (and a test can
    assert ``handoff.target is <the detected ServingTarget>``).
    """

    target: Any                          # the SHARED ServingTarget (by identity)
    plan: "ActionPlan"
    reason: str = ""                     # why no target / why blocked, if any
    available: bool = True               # False when no running target to evaluate


# ── Hook 2: Promote to catalog (SCAFFOLD + GATE — design §3.5b) ────────────────────


@dataclass
class PromoteScaffold:
    """The computed catalog-promotion scaffold for a served/validated BYO model.

    Design §3.5b — a SCAFFOLD + GATE, not a YAML IDE.  COMPUTED from facts the
    app already holds (the BYO pull-gate arch facts in ``ByoResult`` + the
    Evidence ``Measurement`` numbers); PREVIEWED here.  The actual write into
    ``scripts/lib/profiles/`` + the guard-suite run is a separate gated
    ``ActionPlan`` (``write_plan``) that is MOCKED / never-executed this phase and
    NEVER auto-fires.

    Shapes match reality (verified against ``scripts/lib/profiles/models/*.yml``
    + ``compose_registry.py`` ``_entry(...)`` + ``docs/ADDING_MODELS.md``):
      - ``profile_yaml``   — the ``models/<id>.yml`` ModelProfile skeleton;
      - ``registry_entry`` — the ``compose_registry.py`` ``_entry(...)`` row;
      - new models START at ``status="incubating"`` (ADDING_MODELS.md rule).
    """

    model_id: str = ""
    repo: str = ""                       # the BYO HF repo this came from
    profile_path: str = ""               # scripts/lib/profiles/models/<id>.yml
    registry_slug: str = ""              # the proposed compose_registry key
    profile_yaml: str = ""               # the previewed ModelProfile YAML skeleton
    registry_entry: str = ""             # the previewed _entry(...) row
    guard_suite_cmd: list[str] = field(default_factory=list)  # for t in scripts/tests/*.sh
    write_plan: Optional["ActionPlan"] = None   # the gated, mock-only write+guard action
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def computed(self) -> bool:
        return bool(self.profile_yaml and self.registry_entry and not self.error)


# ── Hook 3: Optimize for my card (DORMANT v0.10.0 seam — design §5.2 seam 1) ───────


@dataclass
class OptimizerReport:
    """Result of the ▸ Optimize-for-my-card seam.

    The v0.10.0 optimizer (``recommend --optimize`` / ``generate_compose.py
    --optimize``) does NOT exist yet — this is a DORMANT seam.  When invoked it
    detects the optimizer's absence and reports ``available=False`` with the
    honest ``'optimizer not available (v0.10.0)'`` message.  The honesty-gate
    fields below are the INTERFACE reserved for when the engine lands; they stay
    empty / ``None`` while dormant (never fabricated — design §5.2).

    Honesty gates (rendered only once the optimizer is live):
      - ``boot_fit``        : 'predicted' | 'measured'  (boot-fit provenance)
      - ``runtime``         : 'soak-validated' | 'unvalidated'  (runtime claim)
      - ``confidence``      : a tier label (e.g. 'high' / 'cross-rig')
      - ``cliff_class``     : a cliff-class config needs ``--accept-runtime-risk``
      - ``accept_runtime_risk_required`` : True when the rec is cliff-class.
    """

    available: bool = False
    message: str = "optimizer not available (v0.10.0)"
    # Reserved honesty-gate interface (dormant — populated only when live):
    recommended_slug: str = ""
    boot_fit: str = ""                   # 'predicted' | 'measured'
    runtime: str = ""                    # 'soak-validated' | 'unvalidated'
    confidence: str = ""                 # confidence tier label
    cliff_class: bool = False
    accept_runtime_risk_required: bool = False


# ── Parse helpers (pure) ─────────────────────────────────────────────────────────


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# Strip ANSI color codes (health.sh / gpu-mode emit them).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def parse_health_text(text: str) -> DoctorRead:
    """Best-effort parse of health.sh stdout into a DoctorRead.

    health.sh has no --json contract; this scans the human-readable output for
    the load-bearing signals.  It is intentionally tolerant — any line it can't
    recognize is ignored, and ``raw`` always preserves the full text.
    """
    clean = strip_ansi(text or "")
    dr = DoctorRead(raw=text or "", parse_source="health.sh-text")

    lower = clean.lower()
    dr.reachable = "not reachable" not in lower and "✗ api not reachable" not in lower
    # "✓ serving" / "serving" markers
    dr.serving = "serving" in lower and "not serving" not in lower

    # KV pool percent: "KV pool 61%" / "KV cache ... 61%"
    m = re.search(r"kv\s*(?:pool|cache)[^0-9]*([0-9]{1,3})\s*%", clean, re.IGNORECASE)
    if m:
        dr.kv_pool_pct = int(m.group(1))

    # spec-dec firing: "MTP n=2, 73% accept" / "spec-dec firing (DFlash ...)"
    m = re.search(r"(spec[- ]?dec[^\n]*|MTP\s*n=\d+[^\n]*|DFlash[^\n]*)", clean, re.IGNORECASE)
    if m:
        dr.spec_dec = m.group(1).strip()

    # recent errors: "0 recent errors" / "3 errors"
    m = re.search(r"([0-9]+)\s+(?:recent\s+)?errors?", clean, re.IGNORECASE)
    if m:
        dr.recent_errors = int(m.group(1))

    # Condensed one-liner: first non-empty content line after the banner.
    if not dr.reachable:
        dr.summary = "API not reachable"
    else:
        bits: list[str] = []
        if dr.serving:
            bits.append("serving")
        if dr.kv_pool_pct is not None:
            bits.append(f"KV pool {dr.kv_pool_pct}%")
        if dr.spec_dec:
            bits.append(dr.spec_dec)
        if dr.recent_errors is not None:
            bits.append(f"{dr.recent_errors} errors")
        dr.summary = " · ".join(bits) if bits else "reachable"

    return dr


# Coarse BENCHMARKS.md row parse — provenance-flagged so the UI never mistakes
# a markdown scrape for a structured measurement record.
#
# Real BENCHMARKS.md "Narr / Code TPS" column shapes (verified live):
#   bold:      ``**81.21 / 108.20** single-stream`` / ``**59.67 / 68.78** (decode …)``
#   non-bold:  ``50 / 67`` / ``~32 / ~33``
#   absent:    ``TBD`` / ``—``  (must yield no TPS, NOT a bogus pair)
# The leading ``**`` and ``~`` are optional; trailing prose after the pair is
# ignored.  We anchor on the FIRST ``N / M`` pair (the canonical narr/code), so
# the parenthetical ``(decode 60.39 / 72.40)`` doesn't shadow the headline.
_BENCH_TPS_RE = re.compile(
    r"^\s*\*{0,2}\s*~?\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*~?\s*([0-9]+(?:\.[0-9]+)?)\s*\*{0,2}"
)
_BENCH_8PK_RE = re.compile(r"8-pack\s+([0-9]+/150)")
_BENCH_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def _tps_from_cell(cell: str) -> tuple[Optional[float], Optional[float]]:
    """Parse a 'Narr / Code TPS' table cell into (narr, code).

    Handles bold (``**X / Y**``), non-bold (``50 / 67``), tilde-prefixed
    (``~32 / ~33``) and trailing prose (``… single-stream`` / ``(decode …)``).
    Returns (None, None) for ``TBD`` / ``—`` / anything without a leading pair.
    """
    m = _BENCH_TPS_RE.match(cell or "")
    if not m:
        return None, None
    return _as_float(m.group(1)), _as_float(m.group(2))


def _bench_row_cells(line: str) -> list[str]:
    """Split a markdown table row into trimmed cell strings (no leading/trailing
    empties from the surrounding pipes)."""
    if "|" not in line:
        return []
    parts = [c.strip() for c in line.split("|")]
    # A '| a | b |' row splits to ['', 'a', 'b', ''] — drop the bookend empties.
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def parse_benchmarks_md_for_slug(md_text: str, slug: str) -> Optional[Measurement]:
    """Best-effort: scan BENCHMARKS.md for the row whose first cell names the
    slug's serving file.

    Returns a Measurement with source='benchmarks.md' (coarse) if a matching
    benchmark row yields a narr/code TPS pair, else None.  The registry keys
    composes by serving file (e.g. ``llamacpp/mtp``); the BENCHMARKS table keys
    by compose filename in the FIRST column, often with a ``.yml`` extension
    (``minimal.yml``).  The match is **anchored to the first cell** and exact on
    the serving-file token — a substring match would let ``dual`` hit
    ``dual-dflash.yml`` and pull the wrong row (the bug this fixes).
    """
    if not md_text or not slug:
        return None
    stem = slug.split("/")[-1]
    # Backtick-quoted tokens that must appear as a standalone word in the first
    # cell: the bare stem, the stem + .yml, or the full slug.  Word-boundary
    # anchored so 'dual' does not match 'dual-dflash'.
    tokens = {stem, f"{stem}.yml", slug}
    cell_token_res = [
        re.compile(r"(?<![\w./-])" + re.escape(t) + r"(?![\w.-])")
        for t in tokens
    ]
    for line in md_text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _bench_row_cells(line)
        if not cells:
            continue
        first = cells[0]
        if not any(r.search(first) for r in cell_token_res):
            continue
        # The benchmark TPS column is "Narr / Code TPS" — index 4 in the canonical
        # 9-col layout (Compose|Rig|KV|Max ctx|TPS|PP|VRAM|Date|Notes).  Stress /
        # soak rows have a different layout and no TPS cell; for robustness we
        # scan cells for the first parseable 'N / M' pair, skipping the header.
        narr = code = None
        for cell in cells[1:]:
            n, c = _tps_from_cell(cell)
            if n is not None:
                narr, code = n, c
                break
        if narr is None:
            # Matched the row but it carries no TPS (TBD / — / a non-TPS row) →
            # honestly report no measurement rather than a bogus pair.
            continue
        m = Measurement(narr_tps=narr, code_tps=code, source="benchmarks.md")
        q = _BENCH_8PK_RE.search(line)
        if q:
            m.quality_8pk = q.group(1)
        d = _BENCH_DATE_RE.search(line)
        if d:
            m.date = d.group(1)
        return m
    return None


def measurement_from_explain_columns(rec: dict[str, Any]) -> Measurement:
    """Build a Measurement from ONE explain ``benchmarks[]`` record.

    The REAL shape of switch.sh --explain --json ``benchmarks`` (verified live)
    is ``[{"row": "<markdown>", "columns": [<cell>, …]}]`` — the raw scraped
    BENCHMARKS.md row plus its split cells.  This is NOT the invented
    ``{"narr_tps": …}`` corpus shape; the TPS lives in ``columns[]`` and must be
    parsed by position (the canonical "Narr / Code TPS" column is index 4).

    Stress / soak rows have a different column layout and no TPS — those yield an
    empty Measurement (the caller then falls through to no measurement).
    """
    cols = rec.get("columns") or []
    if not isinstance(cols, list) or not cols:
        return Measurement()
    # Canonical bench layout: index 4 is "Narr / Code TPS".  Fall back to a scan
    # of all cells if index 4 isn't a TPS pair (layout drift / non-bench row).
    narr = code = None
    if len(cols) > 4:
        narr, code = _tps_from_cell(str(cols[4]))
    if narr is None:
        for cell in cols[1:]:
            n, c = _tps_from_cell(str(cell))
            if n is not None:
                narr, code = n, c
                break
    if narr is None:
        return Measurement()
    row_text = str(rec.get("row", ""))
    m = Measurement(narr_tps=narr, code_tps=code, source="explain")
    q = _BENCH_8PK_RE.search(row_text)
    if q:
        m.quality_8pk = q.group(1)
    d = _BENCH_DATE_RE.search(row_text)
    if d:
        m.date = d.group(1)
    # Max-ctx is the 4th canonical column ("Max ctx"); keep it if present.
    if len(cols) > 3:
        m.max_ctx_label = str(cols[3])
    return m


def measurement_from_explain_benchmarks(benchmarks: list[dict[str, Any]]) -> Measurement:
    """Build a Measurement from the structured ``benchmarks`` array of
    switch.sh --explain --json.

    The array is ``[{"row": "<md>", "columns": [...]}]``.  We walk it newest-row-
    last and return the FIRST record that yields a real TPS pair (a benchmark
    row), skipping stress / soak rows that carry no TPS.  Returns an empty
    Measurement (tps_label '—') when nothing parseable is present.
    """
    if not benchmarks:
        return Measurement()
    best = Measurement()
    for rec in benchmarks:
        if not isinstance(rec, dict):
            continue
        m = measurement_from_explain_columns(rec)
        if m.narr_tps is not None:
            best = m  # keep walking → newest TPS-bearing row wins
    return best


# ── Phase 4 parsers (pure) ────────────────────────────────────────────────────────

# diagnose-profile.sh step header:  [1/6] Compose registry entry exists
_PROFILE_STEP_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+?)\s*$")
# check glyph line:  ✓ vllm/dual found ...   /   ✗ ...   /   ⊘ / ⚠ / △ note
# (verified live: diagnose-profile uses ✓ pass, ✗ fail, ⊘ skipped/projection-FAIL).
_PROFILE_CHECK_RE = re.compile(r"^\s+([✓✗⊘⚠△])\s+(.+)")
# final verdict:  Triage summary: GREEN | YELLOW | RED  (verified live enum).
_PROFILE_SUMMARY_RE = re.compile(
    r"Triage summary:\s*(GREEN|YELLOW|AMBER|RED)", re.IGNORECASE
)


def parse_profile_triage(text: str, slug: str = "") -> ProfileTriage:
    """Parse diagnose-profile.sh's 6-step text output into a ProfileTriage.

    diagnose-profile has no --json (verified live): scan the ``[N/6] <name>``
    step headers, attribute the first ``✓/✗/⚠`` check glyph after each to that
    step's status, and capture the ``Triage summary: <COLOR>`` verdict.  Any
    line it can't recognize is ignored; ``raw`` preserves the full text.
    """
    clean = strip_ansi(text or "")
    tri = ProfileTriage(slug=slug, raw=text or "")
    cur: Optional[ProfileTriageStep] = None
    for line in clean.splitlines():
        m = _PROFILE_STEP_RE.match(line)
        if m:
            cur = ProfileTriageStep(
                num=int(m.group(1)), total=int(m.group(2)), name=m.group(3).strip()
            )
            tri.steps.append(cur)
            continue
        m = _PROFILE_CHECK_RE.match(line)
        if m and cur is not None:
            glyph, detail = m.group(1), m.group(2).strip()
            # First glyph on a step sets its status; a later ✗/⊘ downgrades it.
            status = {
                "✓": "passed", "✗": "failed",
                "⊘": "warn", "⚠": "warn", "△": "warn",
            }[glyph]
            if not cur.detail:
                cur.detail = detail
            # Worst-glyph wins per step (failed > warn > passed) — don't upgrade.
            order = {"passed": 0, "warn": 1, "failed": 2}
            if order[status] >= order.get(cur.status, 0):
                cur.status = status
            continue
        m = _PROFILE_SUMMARY_RE.search(line)
        if m:
            tri.summary = m.group(1).upper()
    return tri


# gpu-mode power-cap status row:
#   0, 370.00 W, 370.00 W, 100.00 W, 390.00 W
_POWER_CAP_ROW_RE = re.compile(
    r"^\s*(\d+)\s*,\s*([0-9.]+)\s*W\s*,\s*([0-9.]+)\s*W\s*,"
    r"\s*([0-9.]+)\s*W\s*,\s*([0-9.]+)\s*W\s*$"
)


def parse_power_cap_status(text: str) -> PowerCapState:
    """Parse ``gpu-mode power-cap status`` output (verified live shape).

    A banner line then a CSV-ish table with a header line and one
    ``index, limit W, default W, min W, max W`` row per GPU.  The header line
    (non-numeric first cell) is ignored by the numeric row regex."""
    clean = strip_ansi(text or "")
    st = PowerCapState(raw=text or "")
    for line in clean.splitlines():
        m = _POWER_CAP_ROW_RE.match(line)
        if not m:
            continue
        st.gpus.append(
            PowerCapGpu(
                index=int(m.group(1)),
                limit_w=_as_float(m.group(2)),
                default_w=_as_float(m.group(3)),
                min_w=_as_float(m.group(4)),
                max_w=_as_float(m.group(5)),
            )
        )
    if not st.gpus:
        st.error = "no power-limit rows parsed"
    return st


def parse_docker_top(name: str, text: str) -> ContainerTop:
    """Parse ``docker top <name>`` output (READ).

    docker top prints a ``ps``-style table: first line is the header, the rest
    are process rows (whitespace-split).  Best-effort split — keeps the raw
    columns so the pane can render them verbatim."""
    top = ContainerTop(name=name)
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        top.error = "no docker top output"
        return top
    top.header = lines[0].split()
    for ln in lines[1:]:
        # Limit the split to len(header) so the trailing CMD (which can contain
        # spaces) stays as one cell.
        ncols = max(len(top.header), 1)
        top.rows.append(ln.split(None, ncols - 1))
    return top


# ── #249 measurement-record corpus → BenchRow ────────────────────────────────────


def bench_row_from_corpus_record(rec: dict[str, Any]) -> Optional[BenchRow]:
    """Build a BenchRow from one #249 measurement-record JSONL line.

    REAL shape (verified live via measurement_record.py): a frozen-schema object
    with ``model_slug`` / ``engine_id`` / ``topology`` / ``kv_dtype`` /
    ``max_model_len`` and a ``measured_extensions`` block holding
    ``decode_tps_by_ctx`` (a ctx→TPS ladder; ``"canonical-short"`` is the bench
    point) and ``wall_tps``.  A bench-only record carries NO 8-pack (that lives
    in BENCHMARKS.md / the rebench _internal.json), so ``quality_8pk`` is "".

    ``decode_tps_by_ctx['canonical-short']`` is the model decode rate measured
    on the canonical short prompts; we surface it as ``code_tps`` (the bench's
    decode number maps to the per-token decode rate, matching how the rebench
    REPORT.md reports it) and leave ``narr_tps`` from ``wall_tps`` if present so
    the row shows a representative pair.  Returns None for a record with no
    usable TPS (an honest empty corpus row would mislead the explorer)."""
    if not isinstance(rec, dict):
        return None
    ext = rec.get("measured_extensions") or {}
    ladder = ext.get("decode_tps_by_ctx") or {}
    decode = None
    if isinstance(ladder, dict):
        # Prefer the canonical-short point; else the first numeric value.
        decode = _as_float(ladder.get("canonical-short"))
        if decode is None:
            for v in ladder.values():
                decode = _as_float(v)
                if decode is not None:
                    break
    wall = _as_float(ext.get("wall_tps"))
    if decode is None and wall is None:
        return None
    prov = rec.get("provenance") or {}
    return BenchRow(
        model=str(rec.get("model_slug", "")),
        engine=str(rec.get("engine_id", "")),
        topology=str(rec.get("topology", "")),
        narr_tps=wall,
        code_tps=decode,
        max_ctx=_ctx_label(rec.get("max_model_len")),
        quality_8pk="",                 # bench-only record carries no 8-pack
        date=str(prov.get("last_confirmed") or ""),
        source="corpus",
        tag=str(rec.get("_tag", "")),
    )


def _ctx_label(max_model_len: Any) -> str:
    """Render a max_model_len int as a compact ctx label (262144 → '256K')."""
    n = _as_int(max_model_len)
    if n is None:
        return ""
    if n >= 1024:
        k = n / 1024.0
        return f"{k:.0f}K" if k == int(k) else f"{k:.1f}K"
    return str(n)


# ── Phase R / R3b-2: ④ Measure — parse a rebench tag's MEASURED numbers ───────────


def measured_from_internal_json(blob: Any) -> MeasuredNumbers:
    """Parse the rebench ``_internal.json`` sidecar into MeasuredNumbers.

    REAL shape (verified against ``scripts/rebench-report.py``'s sidecar write):
    ``{"bench":{"narrative":{"decode_tps_mean","wall_tps_mean",...},
    "code":{...}},"quality":{"total_passed","total_total","total_pct"}}``.
    We surface the DECODE TPS (the per-token rate the catalog bar reports), not
    wall TPS.  Returns an empty MeasuredNumbers (``has_any`` False) when nothing
    parses — the honest "insufficient data" signal."""
    m = MeasuredNumbers(source="_internal.json")
    if not isinstance(blob, dict):
        return MeasuredNumbers()
    bench = blob.get("bench") or {}
    if isinstance(bench, dict):
        narr = bench.get("narrative") or {}
        code = bench.get("code") or {}
        if isinstance(narr, dict):
            m.narr_tps = _as_float(narr.get("decode_tps_mean"))
        if isinstance(code, dict):
            m.code_tps = _as_float(code.get("decode_tps_mean"))
    quality = blob.get("quality") or {}
    if isinstance(quality, dict):
        passed = _as_int(quality.get("total_passed"))
        total = _as_int(quality.get("total_total"))
        if passed is not None and total:
            m.quality_8pk = f"{passed}/{total}"
    return m if m.has_any else MeasuredNumbers()


# REPORT.md TL;DR / per-section fallbacks (when _internal.json is absent/empty).
_REPORT_TLDR_TPS_RE = re.compile(
    r"narrative\s*\*\*([\d.]+)\*\*\s*/\s*code\s*\*\*([\d.]+)\*\*", re.IGNORECASE
)
_REPORT_DECODE_ROW_RE = re.compile(
    r"^\|\s*(narrative|code)\s*\|.*?\|\s*\*\*([\d.]+)\*\*\s*\|", re.IGNORECASE
)
_REPORT_QUALITY_RE = re.compile(
    r"\*\*\s*(\d+)\s*/\s*(\d+)\s*\*\*", re.IGNORECASE
)
# REPORT.md Meta line:  - **Served as:** `qwen3.6-27b-autoround` from `…`
_REPORT_SERVED_RE = re.compile(r"\*\*Served as:\*\*\s*`([^`]+)`")
# REPORT.md Meta line:  - **Model arch:** qwen3_next (Qwen3NextForCausalLM)
_REPORT_ARCH_RE = re.compile(r"\*\*Model arch:\*\*\s*([^\s(]+)")
# REPORT.md Meta line:  - **Container:** `vllm-qwen36-dual`  (the engine prefix
# is the most reliable run-engine signal: vllm- / llama-cpp- / ik-llama- / …)
_REPORT_CONTAINER_RE = re.compile(r"\*\*Container:\*\*\s*`([^`]+)`")
# REPORT.md Meta line:  - **vLLM image:** `vllm/vllm-openai:vX` — its presence is
# itself a strong vLLM signal even when the Container line is absent.
_REPORT_VLLM_IMAGE_RE = re.compile(r"\*\*vLLM image:\*\*\s*`([^`]+)`")


def measured_from_report_md(text: str) -> MeasuredNumbers:
    """Parse measured numbers from a rebench ``REPORT.md`` (the fallback).

    Prefers the headline ``## TL;DR`` ``narrative **N** / code **M**`` bullet;
    falls back to the per-bench ``| narrative | … | **decode** |`` table rows.
    Quality is the ``## Quality`` TOTAL row (``**P / T**``).  Also resolves the
    model best-effort from the Meta ``Served as`` / ``Model arch`` lines."""
    m = MeasuredNumbers(source="REPORT.md")
    if not text:
        return MeasuredNumbers()
    tldr = _REPORT_TLDR_TPS_RE.search(text)
    if tldr:
        m.narr_tps = _as_float(tldr.group(1))
        m.code_tps = _as_float(tldr.group(2))
    else:
        for line in text.splitlines():
            row = _REPORT_DECODE_ROW_RE.match(line.strip())
            if row:
                val = _as_float(row.group(2))
                if row.group(1).lower() == "narrative":
                    m.narr_tps = val
                else:
                    m.code_tps = val
    # Quality TOTAL row (scoped to the Quality section so a stray bold N/M in
    # prose doesn't masquerade as the 8-pack).
    in_quality = False
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("## quality"):
            in_quality = True
            continue
        if in_quality:
            if s.startswith("## "):
                break
            if "total" in s.lower():
                q = _REPORT_QUALITY_RE.search(s)
                if q:
                    m.quality_8pk = f"{q.group(1)}/{q.group(2)}"
                    break
    served = _REPORT_SERVED_RE.search(text)
    if served:
        m.model = served.group(1).strip()
    else:
        arch = _REPORT_ARCH_RE.search(text)
        if arch:
            m.model = arch.group(1).strip()
    m.engine = _engine_family_from_report_md(text)
    return m if (m.has_any or m.model) else MeasuredNumbers()


def _engine_family_from_report_md(text: str) -> str:
    """Resolve the RUN's engine-FAMILY from a rebench REPORT.md Meta block.

    The rebench sidecar (_internal.json) carries NO engine info — it is only in
    the rendered REPORT.md Meta.  The most reliable signal is the ``Container``
    name's engine prefix (``vllm-`` / ``llama-cpp-`` / ``ik-llama-`` /
    ``beellama-`` / ``sglang-``); a present ``vLLM image`` line is itself a vLLM
    tell.  Returns a coarse family token (``vllm`` / ``llama-cpp`` / …) via
    ``_canon_engine_family``, or "" when nothing resolves (the caller then picks
    the bar deterministically and flags that it could not discriminate engine)."""
    if not text:
        return ""
    cm = _REPORT_CONTAINER_RE.search(text)
    if cm:
        from club3090_tui_core.detect import _classify_engine_from_container

        fam = _canon_engine_family(_classify_engine_from_container(cm.group(1).strip()))
        if fam:
            return fam
    if _REPORT_VLLM_IMAGE_RE.search(text):
        return "vllm"
    return ""


def _canon_engine_family(label: str) -> str:
    """Collapse an engine label to a coarse FAMILY key so the registry's
    slug-engine space and the BENCHMARKS.md scrape space compare equal.

    The registry emits e.g. ``llama-cpp-local`` (for BOTH llamacpp/* and
    ik-llama/* slugs), ``vllm-stable``, ``vllm-gemma-stable``, ``beellama-local``;
    the BENCHMARKS.md scrape derives ``llamacpp`` / ``ik-llama`` / ``vllm`` /
    ``beellama`` from the compose-cell prefix.  A raw substring test misses the
    llama.cpp family entirely (``"llama-cpp-local" in "llamacpp"`` is False in
    both directions), silently dropping every llama.cpp cross-rig row — so reduce
    both sides to a shared family token instead.  Returns "" for blank/unknown."""
    norm = (label or "").strip().lower().replace("_", "").replace("-", "")
    if not norm:
        return ""
    if "ikllama" in norm or "llamacpp" in norm:
        return "llama-cpp"
    if norm.startswith("vllm"):
        return "vllm"
    if norm.startswith("beellama"):
        return "beellama"
    if norm.startswith("sglang"):
        return "sglang"
    return norm


def _bench_row_matches(row: "BenchRow", model: str, engine: str) -> bool:
    """Whether a cross-rig BenchRow belongs to a slug's (model, engine).

    Model must match exactly (the catalog slug and the BENCHMARKS.md scrape use
    the same model id).  Engine is matched by FAMILY (see _canon_engine_family):
    the registry label (``llama-cpp-local`` / ``vllm-stable``) and the scrape
    engine (``llamacpp`` / ``ik-llama`` / ``vllm``) live in different label
    spaces, so both are reduced to a coarse family key before comparing.  A blank
    engine on either side (e.g. a bare ``*.yml`` scrape cell with no prefix)
    matches on model alone — the row is still that model's data, shown in the
    clearly-labelled cross-rig section."""
    if model and row.model != model:
        return False
    rf, sf = _canon_engine_family(row.engine), _canon_engine_family(engine)
    if not rf or not sf:
        return True
    return rf == sf


def _canon_model_key(model: str) -> str:
    """Reduce a model id / served-name to a coarse comparable token.

    The curated bar's model is a registry slug (``qwen3.6-27b``); a rebench
    REPORT.md's ``Served as`` is a served-model-name (``qwen3.6-27b-autoround``).
    Strip non-alnum + a trailing weights-variant tail so both reduce to the same
    family token for the loose match."""
    norm = re.sub(r"[^a-z0-9.]", "", (model or "").lower())
    # Drop a trailing quant/variant tail (autoround / awq / int4 / fp8 / gguf …).
    norm = re.sub(
        r"(autoround|awq|gptq|int4|int8|fp8|fp16|bf16|gguf|mtp|dflash|q\d.*)$",
        "",
        norm,
    )
    return norm


def _measure_verdict(vsbar: "MeasureVsBar") -> str:
    """A SIMPLE, honest verdict heuristic (never claims 'catalog-grade').

    Enum:
      - ``"insufficient data"`` — no measured numbers OR no curated bar matched;
      - ``"under the bar"``     — measured decode TPS materially below the bar
                                  (>10% lower on a comparable metric);
      - ``"within tolerance of the bar"`` — measured ≥ ~90% of the bar (incl.
                                  meeting/beating it).
    Deliberately coarse — it is a flag, not a grade; protocol_caveats carries the
    "we can't actually certify this" disclosure."""
    m = vsbar.measured
    bar = vsbar.bar
    if bar is None or not m.has_any:
        return "insufficient data"
    # Compare on whichever TPS metric both sides carry.  A CORPUS bar's narr_tps
    # is WALL (not decode), so it is NOT comparable to the measured narrative
    # DECODE — drop the narrative pair for a corpus bar (Fix 2) and grade on the
    # decode pair (corpus code_tps is the canonical-short decode).
    pairs: list[tuple[Optional[float], Optional[float]]] = [
        (m.code_tps, bar.code_tps),
    ]
    if vsbar.bar_source != "corpus":
        pairs.insert(0, (m.narr_tps, bar.narr_tps))
    ratios = [meas / ref for meas, ref in pairs if meas is not None and ref]
    if not ratios:
        # No comparable TPS pair — fall back to the 8-pack when BOTH sides carry
        # one, applying the SAME ~0.90 tolerance gate.  Never return "within
        # tolerance" without an actual comparison (a 50/150 measured vs 140/150
        # bar is materially UNDER the bar, not within it).
        mq = _quality_fraction(m.quality_8pk)
        bq = _quality_fraction(bar.quality_8pk)
        if mq is not None and bq:
            return "under the bar" if (mq / bq) < 0.90 else "within tolerance of the bar"
        return "insufficient data"
    worst = min(ratios)
    if worst < 0.90:
        return "under the bar"
    return "within tolerance of the bar"


def _quality_fraction(quality_8pk: str) -> Optional[float]:
    """Parse a ``"<passed>/<total>"`` 8-pack string to a pass-rate fraction.

    Returns ``passed/total`` (0..1), or None when unparseable / total is 0 — so
    the verdict can compare two 8-packs on the same scale instead of treating
    "both have a pack" as "within tolerance"."""
    if not quality_8pk:
        return None
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)\s*$", quality_8pk)
    if not m:
        return None
    passed, total = int(m.group(1)), int(m.group(2))
    if total <= 0:
        return None
    return passed / total


# Section header → (model, topology) for the BENCHMARKS.md fallback parse.
# REAL section headers (verified live): "## Qwen3.6-27B", "### Dual-card (2× RTX
# 3090, TP=2)", "### Single-card (1× RTX 3090) — vLLM", "## Gemma 4 31B ...".
_BENCH_MODEL_HDR_RE = re.compile(r"^#{2,3}\s+(.+?)\s*(?:\(community-experimental\)|—.*)?$")
_BENCH_TOPO_HDR_RE = re.compile(
    r"(single-card|dual-card|quad-card|multi)", re.IGNORECASE
)


def _bench_md_topology(header: str) -> str:
    """Map a BENCHMARKS.md sub-section header to a topology slug."""
    low = header.lower()
    if "single" in low:
        return "single"
    if "dual" in low:
        return "dual"
    if "quad" in low:
        return "multi4"
    m = re.search(r"tp\s*=\s*(\d+)", low)
    if m:
        tp = int(m.group(1))
        return {1: "single", 2: "dual"}.get(tp, f"multi{tp}")
    return ""


def _normalize_model_slug(header: str) -> str:
    """Normalize a BENCHMARKS.md model section header into a registry-like slug.

    'Qwen3.6-27B' → 'qwen3.6-27b'; 'Gemma 4 31B' → 'gemma-4-31b'; strips a
    trailing parenthetical and an em-dash clause.  Best-effort: the explorer
    filters on this loosely (substring), so exactness isn't load-bearing."""
    h = header.strip()
    h = re.split(r"\s+[—-]\s+|\s*\(", h, 1)[0].strip()
    return re.sub(r"\s+", "-", h).lower()


def bench_rows_from_benchmarks_md(md_text: str) -> list[BenchRow]:
    """Parse BENCHMARKS.md into a list of BenchRow (the explorer's fallback).

    Walks the doc tracking the current model (``## <Model>``) and topology
    (``### <Single|Dual|Quad>-card …``) section headers, and parses each
    canonical 9-column table row (``Compose | Rig | KV | Max ctx | Narr / Code
    TPS | PP | VRAM | Date | Notes``) into a BenchRow.  Reuses the same
    cell-parsing primitives as the per-slug scrape, so it inherits the bold /
    tilde / decode-paren / TBD handling.  Rows whose TPS cell carries no pair
    (TBD / —) are skipped (no bogus pair).  ``source='benchmarks.md'``; the
    8-pack is scraped from the row's Notes cell when present."""
    rows: list[BenchRow] = []
    if not md_text:
        return rows
    cur_model = ""
    cur_topo = ""
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            m = _BENCH_MODEL_HDR_RE.match(stripped)
            if m:
                title = m.group(1).strip()
                # Skip non-model sections ("How to add a row", "See also", etc.)
                if _looks_like_model_header(title):
                    cur_model = _normalize_model_slug(title)
                    cur_topo = ""
            continue
        if stripped.startswith("### "):
            cur_topo = _bench_md_topology(stripped)
            continue
        if not stripped.startswith("|") or not cur_model:
            continue
        cells = _bench_row_cells(line)
        if len(cells) < 5:
            continue
        # Skip the header + separator rows.
        first = cells[0].strip().strip("`").lower()
        if first in ("compose", "") or set(cells[0].strip()) <= {"-", ":", " "}:
            continue
        # Find the first parseable 'N / M' pair (the Narr / Code TPS column).
        narr = code = None
        for cell in cells[1:]:
            n, c = _tps_from_cell(cell)
            if n is not None:
                narr, code = n, c
                break
        if narr is None:
            continue
        row = BenchRow(
            model=cur_model,
            engine=_engine_from_compose_cell(cells[0]),
            topology=cur_topo,
            narr_tps=narr,
            code_tps=code,
            max_ctx=_first_ctx_cell(cells),
            source="benchmarks.md",
            tag=_compose_token(cells[0]),
        )
        q = _BENCH_8PK_RE.search(line)
        if q:
            row.quality_8pk = q.group(1)
        d = _BENCH_DATE_RE.search(line)
        if d:
            row.date = d.group(1)
        rows.append(row)
    return rows


# Model section headers we recognize (avoids treating "How to add a row" etc.
# as a model).  A header "looks like a model" if it has a digit (size/version)
# or matches a known family token.
_MODEL_HDR_TOKENS = ("qwen", "gemma", "llama", "mistral", "deckard", "diffusion")


def _looks_like_model_header(title: str) -> bool:
    low = title.lower()
    if any(tok in low for tok in _MODEL_HDR_TOKENS):
        return True
    return False


def _compose_token(cell: str) -> str:
    """Extract the leading compose/file token from a BENCHMARKS.md first cell.

    First cells look like ``` `minimal.yml` (`mem-util 0.95 …`) ``` or
    ``` `ik-llama/iq4ks-mtp` ⭐ ``` — the load-bearing token is the FIRST
    backtick-quoted span (or the first whitespace token if unquoted).  Trailing
    parentheticals / decorations are dropped."""
    s = cell.strip()
    m = re.match(r"`([^`]+)`", s)
    if m:
        return m.group(1).strip()
    return s.split()[0].strip("`") if s else ""


def _engine_from_compose_cell(cell: str) -> str:
    """Best-effort engine slug from a BENCHMARKS.md first cell.

    First cells look like ``llamacpp/mtp`` / ``ik-llama/iq4ks-mtp`` /
    ``beellama/dflash`` / ``minimal.yml`` / ``vllm/dual``.  When the cell is an
    ``engine/variant`` slug, take the engine prefix; a bare ``*.yml`` filename
    has no engine token → "" (the topology section + model still filter)."""
    tok = _compose_token(cell)
    if "/" in tok:
        eng = tok.split("/")[0]
        # 'llamacpp' → 'llama-cpp' is the registry's launch engine; keep the
        # markdown token as-is (the pane filters loosely).
        return eng
    return ""


def _first_ctx_cell(cells: list[str]) -> str:
    """Pull the 'Max ctx' cell (canonical index 3) from a parsed bench row.

    Falls back to scanning for the first cell that looks like a ctx label
    (``262K`` / ``131072`` / ``~188K``) when the layout drifts."""
    if len(cells) > 3:
        c = cells[3]
        if re.search(r"\d", c):
            return _clean_ctx_cell(c)
    for c in cells[1:]:
        if re.search(r"\b~?\d+\s*[Kk]?\b", c) and "/" not in c and "W" not in c:
            if re.search(r"[Kk]\b|\d{4,}", c):
                return _clean_ctx_cell(c)
    return ""


def _clean_ctx_cell(c: str) -> str:
    """Strip markdown bold/notes from a ctx cell, keeping the first ctx token."""
    m = re.search(r"~?\*{0,2}\s*(\d[\d.,]*\s*[Kk]?)", c)
    return m.group(1).strip() if m else c.strip()[:12]


# ── Phase 5: promote-to-catalog scaffold computation (pure) ───────────────────────


def _slug_from_repo(repo: str) -> str:
    """Derive a candidate model-id from an HF ``org/Model`` repo string.

    Best-effort, lower-kebab — the maintainer renames before committing.  This is
    a SCAFFOLD placeholder, not an authoritative id."""
    tail = (repo or "").rsplit("/", 1)[-1]
    tail = re.sub(r"-(gguf|awq|int4|int8|fp8|autoround|bf16)$", "", tail, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z0-9.]+", "-", tail).strip("-").lower()
    return s or "new-model"


def _quant_slug_for_arch(byo: Optional["ByoResult"]) -> str:
    """Pick a plausible weights-variant quant-slug from the BYO arch facts.

    Mirrors the real weights-map keys (``autoround-int4`` / ``awq`` / ``fp8`` /
    ``gguf``).  The BYO dry-run only reports a coarse ``quant_match`` hint, so
    this is a placeholder the maintainer confirms — NOT an inferred ground truth.
    """
    qm = (getattr(byo, "quant_match", "") or "").lower() if byo else ""
    if "int4" in qm or "autoround" in qm:
        return "autoround-int4"
    if "awq" in qm:
        return "awq"
    if "fp8" in qm:
        return "fp8"
    if "gguf" in qm or "iq" in qm or "q4" in qm or "q8" in qm:
        return "gguf"
    return "autoround-int4"


def compute_promote_scaffold(
    *,
    byo: Optional["ByoResult"],
    measurement: Optional["Measurement"],
    model_id: str = "",
    sibling_compose_path: str = "",
) -> "PromoteScaffold":
    """COMPUTE (never write) the catalog-promotion scaffold from facts the app
    already holds — the BYO pull-gate arch facts (``ByoResult``) + the Evidence
    measured numbers (``Measurement``).  Design §3.5b: a SCAFFOLD + GATE, not a
    YAML IDE.

    Returns a ``PromoteScaffold`` carrying:
      - the ``models/<id>.yml`` ModelProfile YAML skeleton (real schema keys:
        ``schema_version`` / ``id`` / ``display_name`` / ``family`` / a ``weights``
        MAP keyed by quant-slug / ``vision_capable`` — per ADDING_MODELS.md);
      - the ``compose_registry.py`` ``_entry(...)`` row (real kwargs: ``model`` /
        ``weights_variant`` / ``workload`` / ``engine`` / ``drafter`` /
        ``kv_format`` / ``tp`` / ``max_ctx`` / ``compose_path`` / ``default_port`` /
        ``kvcalc_key`` / ``status``);
      - the guard-suite command (``for t in scripts/tests/*.sh; do bash "$t"; done``).

    New models START at ``status="incubating"`` (the ADDING_MODELS.md rule).  The
    scaffold is a STARTING POINT the maintainer edits + validates — the field
    values it can't know (exact arch dims, real family tag) are left as REQUIRED
    `<...>` placeholders so the maintainer must fill them, never a fabricated
    number.  The actual write + guard run is attached by the service layer as a
    gated ``write_plan`` (mock-only this phase).
    """
    repo = getattr(byo, "repo", "") if byo else ""
    if byo is not None and getattr(byo, "error", ""):
        return PromoteScaffold(repo=repo, error=f"BYO check failed: {byo.error}")

    mid = model_id or _slug_from_repo(repo)
    quant = _quant_slug_for_arch(byo)
    arch = (getattr(byo, "arch", "") or "") if byo else ""
    fit_verdict = (getattr(byo, "fit_verdict", "") or "") if byo else ""
    sibling = (getattr(byo, "sibling_slug", "") or "") if byo else ""
    drop_spec = bool(getattr(byo, "drop_spec_config", False)) if byo else False

    # Registry slug mirrors the path: <engine>/<model>-<topology>-<quant>.
    short = mid.replace("qwen3.6-", "qwen-").replace("gemma-4-", "gemma-")
    registry_slug = f"vllm/{short}-dual-{quant}"
    profile_path = f"scripts/lib/profiles/models/{mid}.yml"
    compose_path = (
        sibling_compose_path
        or f"models/{mid}/vllm/compose/dual/{quant}/base.yml"
    )

    # Measured numbers (Evidence) → the registry status_note + a BENCHMARKS hint.
    tps = measurement.tps_label if measurement else "—"
    q8 = (measurement.quality_8pk if measurement else "") or ""

    profile_yaml = (
        "schema_version: 1\n"
        f"id: {mid}\n"
        f"display_name: <Human-readable name — from {repo or '<repo>'}>\n"
        "family: <family-tag>                    # REQUIRED — real tag "
        "(qwen3-next-hybrid / gemma4-swa-dense / …), NOT inferred\n"
        f"# arch reported by pull-gate: {arch or '<unknown>'} "
        "— fill the FAMILY-SPECIFIC dims from config.json (see ADDING_MODELS.md)\n"
        "# Architecture (drives kv-calc.py + fits()) — FAMILY-SPECIFIC keys:\n"
        "num_hidden_layers: <int>\n"
        "num_kv_heads: <int>\n"
        "num_attention_heads: <int>\n"
        "head_dim: <int>\n"
        "max_position_embeddings: <int>\n"
        "valid_tp: [1, 2]\n"
        "weights:\n"
        f"  {quant}:                                 # quant-slug == compose <quant>/ dir == weights_variant\n"
        f"    path: {mid}-{quant}\n"
        f"    local_subdir: {mid}-{quant}\n"
        "    size_gb: <float>\n"
        f"    format: {quant if quant != 'autoround-int4' else 'autoround'}\n"
        "    status: incubating\n"
        f"    hf_repo: {repo or '<Org/Repo>'}\n"
        f"    engine: vllm\n"
        "    kind: main\n"
        "    verify_glob: \"*.safetensors\"\n"
        f"default_weight_variant: {quant}\n"
        "compatible_drafters: []\n"
        "vision_capable: <bool>\n"
    )

    note_bits: list[str] = []
    if fit_verdict:
        note_bits.append(f"pull-gate fit={fit_verdict}")
    if tps and tps != "—":
        note_bits.append(f"measured ~{tps} TPS")
    if q8:
        note_bits.append(f"8-pack {q8}")
    if sibling:
        note_bits.append(f"BYO Route-C sibling of {sibling}")
    note_bits.append("scaffolded from cockpit Promote-to-catalog — VALIDATE before promoting")
    status_note = "; ".join(note_bits)

    registry_entry = (
        f'    "{registry_slug}": _entry(\n'
        f'        model="{mid}",\n'
        f'        weights_variant="{quant}",\n'
        f'        workload="long-ctx-single",\n'
        f'        engine="vllm-stable",\n'
        f'        drafter=None,'
        + ("  # BYO fine-tune has no MTP head — drop --speculative-config\n" if drop_spec else "\n")
        + f'        kv_format="fp8_e5m2",\n'
        f'        tp=2, max_ctx=<int>, max_num_seqs=2, mem_util=0.92,\n'
        f'        compose_path="{compose_path}",\n'
        f'        default_port=<NNNN>,                       # MUST equal the compose ${{PORT:-NNNN}}\n'
        f'        kvcalc_key="{mid}:dual",\n'
        f'        status="incubating",                       # NEW MODELS START HERE\n'
        f'        status_note="{status_note}",\n'
        f'    ),\n'
    )

    notes = [
        "New models start at status='incubating' (ADDING_MODELS.md): hidden from "
        "switch.sh --list, --force to launch; promote up the enum as it validates.",
        "Fill every <...> placeholder from config.json + a boot log — the scaffold "
        "never fabricates arch dims, ports, or sizes.",
        "After writing: run the FULL guard suite (below) + author CalibrationData "
        "for the vLLM entry, then verify-full / bench / soak / quality.",
    ]
    if drop_spec:
        notes.append("BYO swap_path flagged drop_spec_config — the row drops the drafter.")

    return PromoteScaffold(
        model_id=mid,
        repo=repo,
        profile_path=profile_path,
        registry_slug=registry_slug,
        profile_yaml=profile_yaml,
        registry_entry=registry_entry,
        guard_suite_cmd=["bash", "-c", 'for t in scripts/tests/*.sh; do bash "$t"; done'],
        notes=notes,
    )
