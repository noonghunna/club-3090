"""club3090 serve cockpit — main Textual application.

The two modes (Run & Operate · Bring & Validate) are wired to the real data layer
(``services.CockpitData`` + ``data.py`` shapes), reusing the shared core
(``club3090_tui_core``) for detect / streaming / widgets.  (R1 folded the former
Discover + Serve + Benchmarks modes into a single Run mode; R2a renamed the
Estate mode to Operate and moved the Doctor surface into it.)

  - Run / Catalog        : real enriched rows from ``CockpitData.load_catalog``;
                          ``e`` opens ``explain`` (incl. the folded-in cross-rig
                          benchmark rows), ``/`` filters, ``⏎`` builds the GATED
                          ``serve(slug)`` plan and opens the reconcile-gated
                          ConfirmActionScreen — on confirm the boot streams into
                          the transient Run LivePane (#serve-live).
  - Run / BYO            : ``CockpitData.byo_check`` → fit verdict + swap_path.
  - Operate / Orch       : ``estate_state`` live (GPU cards, scenes, services,
                          power-cap); scene-switch → confirm modal that FIRST calls
                          ``reconcile_before_write`` then ``scene_switch``; ``c``
                          opens the power-cap menu (default 230W / clear / custom
                          W), each gated.  (prune removed; cap-sweep → Doctor.)
  - Operate / Containers : ``containers`` real list; drill into Logs/Top/Config;
                          restart/stop/rm behind the reconcile-gated confirm.
  - Operate / Doctor     : "is it serving correctly?" — health.sh (live) + verify
                          ([v] verify.sh test query) + verify-full ([V]) cards, all
                          READ-only; [R] basic report, [F] full battery (warned);
                          ``w`` power-cap sweep (gated tuning bench).
  - Validate / Run       : launchable ladder + extra tools (``run_validation``,
                          confirm-gated, streamed into a LivePane) + §3.5 *tune*
                          gotchas inline.
  - Validate / Evidence  : ``evidence_list()`` run tags; ``⏎`` opens the
                          ``evidence_report()`` modal; ``s`` stages the gated
                          submit-to-localmaxxing (outward NETWORK write, never auto).

EVERY GPU-claiming write (serve / scene-switch / estate-down / container
restart|stop|rm) goes through ``CockpitData.execute_action``, which re-runs the
reconcile gate first and refuses when unsafe (unless an explicit, reasoned force
override).  Heavy/destructive non-GPU writes (validation launches, submit-bench
POST, power-cap, cap-sweep) are confirm-gated too.  The write runner / network are
NEVER executed live — tests inject fakes and conftest blocks the real spawn.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
    Tabs,
)
# ContentSwitcher is the inner pane host of a TabbedContent.  Reading a
# TabbedContent's DIRECT ContentSwitcher children gives ONLY its own TabPanes —
# unlike the recursive ``tc.query(TabPane)``, which also returns the panes of a
# NESTED TabbedContent (e.g. the Containers drill Logs/Top/Config sub-tabs).
from textual.widgets._tabbed_content import ContentSwitcher, ContentTabs
from textual import work

from rich.text import Text

from club3090_tui_core.registry import VariantRow
from club3090_tui_core.widgets.live_pane import LivePane

from .data import (
    ActionPlan,
    BenchRow,
    ByoResult,
    CatalogEntry,
    ContainerInfo,
    EstateState,
    EstateTelemetry,
    EvidenceReport,
    EvidenceTag,
    GATE_STEPS,
    Measurement,
    MeasureVsBar,
    OptimizerReport,
    PowerCapState,
    PromoteScaffold,
    ReconcileResult,
    Scene,
    SceneServiceState,
    WEIGHTS_ABSENT,
    WEIGHTS_DOWNLOADING,
    WEIGHTS_PARTIAL,
    WEIGHTS_PRESENT,
    _bench_row_matches,
    _canon_engine_family,
    _ctx_label,
    _human_gb,
    downgrade_fit_glyph,
    measurement_from_explain_columns,
    parse_ctx_label,
)
from .services import CockpitData

# ── Status glyph mapping ──────────────────────────────────────────────────────

_STATUS_GLYPH: dict[str, str] = {
    "production": "✅",
    "caveats": "⚠️",
    "experimental": "🧪",
    "incubating": "🐣",
    "preview": "👁️",
    "upstream-gated": "⏸️",
    "deprecated": "🗑️",
}


def _error_headline(err: str) -> str:
    """MUST-FIX 3: condense an estate READ error to a short headline for the rail
    / Containers strip.  ``state.error`` has multiple producers — a docker failure
    ("docker unreachable — daemon running? …"), a detect failure ("detect failed:
    …", docker fine), and an nvidia-smi failure — so a HARDCODED "docker
    unreachable" mislabels the others.  Render the ACTUAL text, truncated to the
    part before " — " (the OrchPane error strip renders the full text)."""
    head = (err or "").strip().split(" — ", 1)[0].strip()
    return head[:80] if head else "read failed"


def _status_glyph(status: str) -> str:
    return _STATUS_GLYPH.get(status.lower(), status)


def _weights_glyph(e: CatalogEntry) -> str:
    """Download-state prefix for the catalog slug cell (Download UX).  ⏳NN%
    downloading · ⬇ absent (not on disk) · ⚠ partial (interrupted/wrong).
    present / unknown → "" (no clutter: present needs no badge, unknown means we
    can't tell — e.g. a self-grabbed GGUF compose)."""
    st = getattr(e, "weights_state", "") or ""
    if st == WEIGHTS_DOWNLOADING:
        pct = getattr(e, "download_pct", None)
        return f"[yellow]⏳{pct}%[/yellow]" if pct is not None else "[yellow]⏳[/yellow]"
    if st == WEIGHTS_ABSENT:
        return "[dim]⬇[/dim]"
    if st == WEIGHTS_PARTIAL:
        return "[yellow]⚠[/yellow]"
    return ""


# ── Profile-template derivation (#6 / A12) ──────────────────────────────────────
#
# The BYO / ① Bring profile-like inputs are a SELECT of (engine, topology)
# "template" slugs derived from the loaded registry variants (one representative
# latest compose per (engine, topology)) so the user PICKS rather than free-types
# a profile-like.  The selected value is the SAME profile-like string byo_check
# consumes — the dropdown is a typo-proof front door, not a behaviour change.

_TOPO_ORDER = {"single": 0, "dual": 1, "multi3": 2, "multi4": 3, "multi8": 4}

# FIX 2 (escape hatch) — the curated (family, topology) dropdown is the primary
# affordance, but it deliberately lists only ~7-12 representatives, not all 53
# slugs.  A trailing sentinel option reveals a companion free-text Input so ANY
# registry slug stays expressible (validated by the existing byo_check
# unknown-profile path).  The sentinel value is a fixed marker, NOT a real slug.
PROFILE_CUSTOM_SENTINEL = "__custom__"


def _variant_topology(row: "VariantRow") -> str:
    """Extract the topology token (single/dual/multiN) from a variant's compose
    path — the path encodes it as ``…/compose/<topology>/<quant>/<file>``."""
    path = (getattr(row, "compose_path", "") or getattr(row, "compose_dir", "") or "")
    for part in path.replace("\\", "/").split("/"):
        if part in _TOPO_ORDER:
            return part
    return ""


class ProfileOption(NamedTuple):
    """One profile-template dropdown option.

    A 4-field NamedTuple so it still unpacks as ``(label, value)``-compatible
    in 2-element contexts is NOT assumed; callers that need the Select's
    ``(label, value)`` pairs use :func:`profile_select_options`.  ``topology``
    is carried THROUGH from the registry variant (NOT re-derived by splitting
    the label) so the default picker can filter by topology directly.  ``status``
    is the registry status word of the chosen representative slug (FIX-2
    status-aware pick) so :func:`default_profile_template` can apply the
    functional-status floor without re-walking the variant rows."""

    label: str
    slug: str
    topology: str
    status: str = "production"


# FIX 2 — functional-status floor (mirrors compose_registry.FUNCTIONAL_STATUSES,
# kept local so the cockpit data layer has no import dep on the scripts/ tree).
# A slug is "functional" (launches without --force) iff its status is one of
# these; experimental / incubating / preview / upstream-gated / deprecated are
# NON-functional (incubating needs --force).  Used to keep the profile-template
# representatives + the rig default off non-launchable slugs.
_FUNCTIONAL_STATUSES = frozenset({"production", "caveats"})


def _status_is_functional(status: str) -> bool:
    return (status or "").strip().lower() in _FUNCTIONAL_STATUSES


def profile_select_options(
    options: list["ProfileOption"],
) -> list[tuple[str, str]]:
    """Project ``ProfileOption``s down to the ``(label, value)`` pairs a Textual
    ``Select`` consumes, with the FIX-2 escape-hatch sentinel appended last so any
    non-curated registry slug stays reachable via a companion free-text Input.
    Pure."""
    pairs = [(o.label, o.slug) for o in options]
    pairs.append(("✎ custom slug…", PROFILE_CUSTOM_SENTINEL))
    return pairs


def _curated_default_map(
    defaults: Optional[list[dict]],
) -> dict[tuple[str, str], str]:
    """Project the registry's top-level ``defaults`` array (from
    ``registry-emit.sh --json``) down to ``{(family, topology): slug}`` — the
    registry's OWN curated recommendation per (engine-family, topology).  Engine
    is collapsed to a FAMILY (so ``vllm`` covers vllm-stable / vllm-gemma-stable),
    matching :func:`profile_templates`' grouping.  Empty when ``defaults`` is
    absent (the raw-tab fallback load path doesn't carry it) — callers then fall
    back to the status floor.  Pure."""
    out: dict[tuple[str, str], str] = {}
    for d in defaults or []:
        engine = (d.get("engine") or "").strip()
        topo = (d.get("topology") or "").strip() or "—"
        slug = (d.get("slug") or "").strip()
        if not slug:
            continue
        family = _canon_engine_family(engine) or (engine or "—")
        out.setdefault((family, topo), slug)
    return out


def profile_templates(
    variants: list["VariantRow"],
    defaults: Optional[list[dict]] = None,
) -> list["ProfileOption"]:
    """#6 — derive the profile-template dropdown options from the loaded variants.

    FIX 2 (maintainer directive) — ONE representative option per UNIQUE
    (engine-FAMILY, topology) pair — a short curated list (~7-12 entries), NOT one
    per slug.  The maintainer's call: "profile-like was meant to have only the
    unique and latest engine/topology composes and not all the list."  The escape
    hatch (a trailing "✎ custom slug…" sentinel + companion Input) keeps every
    other registry slug reachable, so the short dropdown doesn't strand them.

    Grouping is by canonical engine FAMILY (``_canon_engine_family``) so
    ``vllm-stable`` and ``vllm-gemma-stable`` collapse to ONE "vllm" per topology
    (otherwise "vllm / dual" would appear twice).

    **Representative resolution (FIX-2 status-aware — the earlier rule was
    status-BLIND, so 4 of 7 reps were non-functional, e.g. (vllm, single) →
    vllm/vibethinker-3b-single [incubating]).** Per (family, topology), in order:
      a. the slug literally named ``<family>/<topology>`` if present (the
         canonical/recommended template, e.g. ``vllm/dual``);
      b. else the registry's curated ``defaults`` slug for that (family,
         topology) if available (its literal recommendation — exactly the
         "latest/recommended" the maintainer wants);
      c. else the LAST variant in registry order whose status is FUNCTIONAL
         (production / ⚠️ caveats);
      d. else ``slugs[-1]`` — the group is ENTIRELY non-functional (e.g.
         (vllm, multi4) and (beellama, dual) on the live registry: every member
         is experimental), so there is no functional rep to choose; the escape
         hatch still reaches every member.

    Sorted by (topology order, family).  ``topology`` + the chosen slug's
    ``status`` are carried THROUGH on the option (status feeds the default
    picker's floor).  Pure — no I/O."""
    curated = _curated_default_map(defaults)
    # Group by (family, topology), preserving registry order within each group.
    groups: "OrderedDict[tuple[str, str], list[tuple[str, str]]]" = OrderedDict()
    for row in variants:
        engine = (getattr(row, "engine", "") or "").strip()
        slug = (getattr(row, "slug", "") or "").strip()
        if not slug:
            continue
        family = _canon_engine_family(engine) or (engine or "—")
        topo = _variant_topology(row) or "—"
        status = (getattr(row, "status", "") or "").strip().lower()
        groups.setdefault((family, topo), []).append((slug, status))
    out: list[ProfileOption] = []
    for (family, topo), members in groups.items():
        slugs = [s for (s, _st) in members]
        status_by_slug = {s: st for (s, st) in members}
        canonical = f"{family}/{topo}"
        if canonical in slugs:
            rep_slug = canonical                                  # (a) literal
        elif curated.get((family, topo)) in status_by_slug:
            rep_slug = curated[(family, topo)]                    # (b) curated default
        else:
            functional = [s for (s, st) in members if _status_is_functional(st)]
            rep_slug = functional[-1] if functional else slugs[-1]  # (c) / (d)
        rep_status = status_by_slug.get(rep_slug, "")
        # family + topology is the prominent, readable part; the chosen slug shows.
        label = f"{family} / {topo}  ·  {rep_slug}"
        out.append(
            ProfileOption(label=label, slug=rep_slug, topology=topo, status=rep_status)
        )
    out.sort(key=lambda o: (_TOPO_ORDER.get(o.topology, 99), o.label))
    return out


def default_profile_template(
    options: list["ProfileOption"], num_gpus: int
) -> Optional[str]:
    """A12 — pick the dropdown's default value for the rig's own topology.

    Rule (deterministic, meaningful): prefer the registry's CANONICAL slug for
    the rig topology — a slug literally named ``<engine>/<topo>`` (``vllm/dual``
    for ≥2 cards, ``vllm/single`` for 1 card), preferring a ``vllm/``-prefixed
    slug; then any literal ``<engine>/<topo>`` slug; then any slug whose
    topology matches; finally the first option.  NEVER an arbitrary alphabetical
    (e.g. Gemma/beellama) slug.  Topology comes from the carried-through
    ``ProfileOption.topology`` — never re-derived from the label.

    **FIX 2 status floor — a Select default MUST be launchable.** The earlier
    rule returned the FIRST vllm-single option for a 1-card rig, which (since the
    reps were status-blind) was ``vllm/vibethinker-3b-single`` [incubating] — a 3B
    that needs ``--force`` to launch.  Now the picker is restricted to FUNCTIONAL
    options (production / ⚠️ caveats) FIRST; with the status-aware reps that lands
    the rig default on ``vllm/minimal`` for a single-card rig (the registry's own
    curated single default) and keeps ``vllm/dual`` for ≥2 cards.  Only if NO
    functional option exists at all does it fall back to the prior order over the
    full option set — and the return is ALWAYS one of ``options`` (a Select can't
    default to an absent value)."""
    if not options:
        return None
    want = "single" if num_gpus <= 1 else "dual"

    def _pick(pool: list["ProfileOption"]) -> Optional[str]:
        """The original topology-preference order, applied to a pre-filtered
        option pool (functional-only, then the full set)."""
        same_topo = [o for o in pool if o.topology == want]
        # 1. the canonical vllm slug literally named "vllm/<topo>".
        canonical_vllm = f"vllm/{want}"
        for o in same_topo:
            if o.slug == canonical_vllm:
                return o.slug
        # 2. any slug literally named "<engine>/<topo>", vllm-prefixed first.
        literal = [o for o in same_topo if o.slug.endswith(f"/{want}")]
        for o in literal:
            if o.slug.startswith("vllm/"):
                return o.slug
        if literal:
            return literal[0].slug
        # 3. any slug of the right topology, vllm-prefixed first.
        for o in same_topo:
            if o.slug.startswith("vllm/"):
                return o.slug
        if same_topo:
            return same_topo[0].slug
        return None

    # Status floor: a functional default first.  Fall back to the full set, then
    # to the first option, so the return is always a real (selectable) value.
    functional = [o for o in options if _status_is_functional(o.status)]
    return _pick(functional) or _pick(options) or (
        functional[0].slug if functional else options[0].slug
    )


def _set_select_options(
    select: "Select", options: list[tuple[str, str]], default: Optional[str]
) -> None:
    """Replace a Select's options and select ``default`` (or the first option).
    A pure widget update — no I/O.  Shared by Run · BYO and the producer ① Bring
    stage so both pick from the SAME registry-derived templates."""
    if not options:
        return
    values = {v for (_l, v) in options}
    chosen = default if (default in values) else options[0][1]
    # NICE-TO-HAVE 2 — suppress Select.Changed for this PROGRAMMATIC update so the
    # app-level on_select_changed only ever sees GENUINE user picks (set_options
    # momentarily selects the first option, which would otherwise false-flag the
    # "user touched the profile" guard and block the rig-default reapply).
    try:
        with select.prevent(Select.Changed):
            select.set_options(options)
            select.value = chosen
    except Exception:
        # Fallback (no prevent available): best-effort update without the guard.
        try:
            select.set_options(options)
            select.value = chosen
        except Exception:
            pass


# NOTE (R3b-2): _canon_engine_family / _bench_row_matches moved to data.py (pure)
# so the data layer's measure_vs_bar can reuse them without a services→app import
# cycle.  They are re-exported here (imported above) so existing app-level callers
# and tests that reference app._bench_row_matches keep working.


# ── Help modal ────────────────────────────────────────────────────────────────


class HelpScreen(ModalScreen):
    """Help overlay showing keybindings and current phase status."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen > Vertical {
        width: 76;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    HelpScreen .help-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
    ]

    # Surface-threaded: the LEAN help OMITS every producer-lane token (the [2]
    # Bring & Validate mode, [P] Promote, [v] Evaluate, the lane section) so it
    # shows only the consumer affordances (Run & Operate + share-back).  The FULL
    # (default) help INCLUDES the lane section.

    # The mode line: lean sees only the merged mode; full additionally sees [2].
    _MODE_LINE_CONSUMER = "  [cyan]1[/cyan]  Run & Operate"
    _MODE_LINE_PRODUCER = (
        "  [cyan]1[/cyan]  Run & Operate    "
        "[cyan]2[/cyan]  Bring & Validate"
    )

    # The producer Bring & Validate lane section — rendered ONLY on producer.
    _LANE_SECTION = """\
[bold]Bring & Validate[/bold] (producer lane — the ① → ⑤ pipeline)
  ① Bring:   fit-check an HF model (pull.sh --dry-run)
  ② Serve:   [cyan]⏎[/cyan]/[cyan]g[/cyan] generate a compose + serve it untested (reconcile-gated)
  ③ Gate:    [cyan]⏎[/cyan] launch validation step (gated)   [cyan]F[/cyan] full battery report.sh --full (~43-min · confirm · uses serving model)
  ④ Measure: [cyan]⏎[/cyan] open report   [cyan]m[/cyan] vs catalog bar (read · flags protocol)   [cyan]s[/cyan] submit to localmaxxing (gated · never auto)
  ⑤ Promote: [cyan]P[/cyan] ▸ Promote a fit-checked model to the catalog (scaffold + gated write)
  [cyan]v[/cyan] ▸ Evaluate the running target via c3t (confirm-gated · mock-only this phase)
"""

    def __init__(self, *, surface: str = "consumer", **kwargs):
        super().__init__(**kwargs)
        self._surface = surface if surface in ("consumer", "producer") else "consumer"

    @property
    def help_text(self) -> str:
        producer = self._surface == "producer"
        mode_line = self._MODE_LINE_PRODUCER if producer else self._MODE_LINE_CONSUMER
        parts: list[str] = [
            "[bold]Keybindings[/bold]",
            "",
            mode_line,
            "  [cyan]r[/cyan]  Refresh (re-reads the live data layer for the active mode)",
            "  [cyan]/[/cyan]  Filter (Catalog tab)",
            "  [cyan]e[/cyan]  Explain selected slug (Catalog tab — incl. cross-rig benchmarks)",
            "  [cyan]⏎[/cyan]  Primary action (serve / switch scene / run step / open report)",
            "  [cyan]?[/cyan]  This help        [cyan]q[/cyan]  Quit",
            "",
            # A5: the navigation keys that are otherwise undiscoverable — the
            # sub-tab cycle has show=False bindings (so it never reaches the
            # footer) and [C] (the lean toggle) is show=False too.  This help is
            # their ONLY teaching surface; surface BOTH on lean AND full ([C] is
            # always-on — a lean-view user needs it to restore the full view).
            "[bold]Navigation[/bold]",
            "  [cyan]\\[[/cyan] / [cyan]][/cyan]  previous / next sub-tab (the only no-mouse tab move)",
            "  [cyan]Tab[/cyan]      cycle focus (tables · inputs · the footer keys)",
            "  [cyan].[/cyan]        toggle the left rail (Modes + Estate) — full-width content",
            "  [cyan]C[/cyan]        toggle lean view (hide / restore the Bring & Validate mode)",
            "  [cyan]Ctrl+p[/cyan]   command palette — fuzzy-search + run any action",
            "  [cyan]S[/cyan]        download Settings — model dir (where weights live) + HuggingFace token",
            "  [cyan]Y[/cyan] / right-click  copy the highlighted slug / open report / selection to the clipboard",
            "  [dim]    (for arbitrary on-screen text: Shift-drag — or Option-drag on macOS — to use the[/dim]",
            "  [dim]     terminal's OWN selection, bypassing the app's mouse capture, then your terminal's copy)[/dim]",
            "  [cyan]Shift+←/→[/cyan] page-scroll a wide table sideways (faster than the ←/→ column cursor)",
            "",
            "[bold]Run & Operate · Catalog[/bold]",
            "  [cyan]⏎[/cyan] serve selected slug (reconcile-gated confirm; F to Force the teardown)",
            "  [cyan]d[/cyan] set-default   [cyan]D[/cyan] clear-default",
            "  [cyan]O[/cyan] ▸ Optimize for my card (v0.10.0 seam — not available yet)",
            "[bold]Run & Operate · Orchestration[/bold]",
            "  [cyan]⏎[/cyan] switch scene   [cyan]k[/cyan] stop THIS model   [cyan]b[/cyan] restart serving   [cyan]n[/cyan] switch model (→ Catalog tab)   (writes gated)",
            "  [cyan]o[/cyan] stop ALL (tears down the whole estate)   [cyan]c[/cyan] power cap… (default 230W / clear / custom W)   (all gated)",
            "[bold]Run & Operate · Containers[/bold]",
            "  [cyan]l[/cyan] logs   [cyan]t[/cyan] top (read)   [cyan]s[/cyan] restart   [cyan]x[/cyan] stop   [cyan]X[/cyan] rm   (writes gated)",
            "[bold]Run & Operate · Doctor[/bold]  — is it serving correctly?",
            "  [cyan]r[/cyan] health (read)   [cyan]v[/cyan] verify   [cyan]V[/cyan] verify-full   [cyan]R[/cyan] report   [cyan]F[/cyan] report --full   [cyan]w[/cyan] power-cap sweep (gated)",
        ]
        # Producer-only lane section — OMITTED on the lean surface (clean help).
        if producer:
            parts.append(self._LANE_SECTION.rstrip("\n"))
        parts.extend([
            "",
            "[bold]Share back[/bold] (Run & Operate — lightweight, no surface switch)",
            "  [cyan]R[/cyan] rig report — paste-ready rig/bench snapshot (read · no network)",
            "  [cyan]B[/cyan] submit bench — submit the latest benched result (gated · never auto)",
            "  [cyan]![/cyan] report a problem — paste-ready issue from the failure context (read · surfaced at a failed serve)",
            "",
            "[bold]Safety — the reconcile gate[/bold]",
            "",
            "  Every write (serve, scene-switch, estate-down, container restart/stop/rm,",
            "  power-cap, cap-sweep, submit-bench) goes through a confirm modal.  GPU-claiming",
            "  writes re-run a FRESH detect immediately before executing and refuse if a",
            "  running container / busy GPU / active estate claim would collide; the modal",
            "  shows exactly what a write would tear down.  Validation launches and the",
            "  outward submit are heavy / network — confirmed, never auto-fired.  Nothing is",
            "  ever forced silently — F surfaces the override with its reason.",
            "",
            "[bold]Status glyphs[/bold]",
            "",
            "  ✅ production   ⚠️  caveats   🧪 experimental",
            "  🐣 incubating  👁️  preview   ⏸️  upstream-gated   🗑️  deprecated",
            "",
            "[bold]Fit glyphs (local card)[/bold]",
            "",
            "  ● fits-clean   ◐ fits-constrained   ○ won't-fit   · skip / unknown",
        ])
        return "\n".join(parts)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("club3090 serve cockpit — Help", classes="help-title")
            yield Static(self.help_text)

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ── Run · Catalog ─────────────────────────────────────────────────────────


# Catalog model-scope dropdown — a "model name only" filter (toggle-hidden, [\\]).
# The model→slugs grouping the flat variant table drops is restored ALWAYS by the
# table's "model" column + group-by-model sort (the switch.sh --list view); this
# dropdown just narrows to one model.  The sentinel = "no scope" (all lanes show).
_ALL_MODELS_VALUE = "__all_models__"
_ALL_MODELS_LABEL = "All models"


class CatalogPane(Container):
    """Catalog tab: DataTable populated from the enriched registry catalog."""

    DEFAULT_CSS = """
    CatalogPane {
        height: 1fr;
    }
    CatalogPane #catalog-status {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    CatalogPane #catalog-model-select {
        height: 3;
        width: 1fr;
        margin: 0 1;
        display: none;
    }
    CatalogPane #catalog-model-select.visible {
        display: block;
    }
    CatalogPane #catalog-filter {
        height: 3;
        display: none;
        margin: 0 1;
    }
    CatalogPane #catalog-filter.visible {
        display: block;
    }
    CatalogPane DataTable {
        height: 1fr;
    }
    CatalogPane #catalog-preview {
        height: auto;
        /* F2 — the border eats 2 rows, so max-height 6 capped content at 4
           lines and a WRAPPING caveat line clipped (the dual-fast case).
           8 fits the 3 base lines + a caveat wrapped to 2-3; anything
           pathological scrolls instead of silently clipping. */
        max-height: 8;
        overflow-y: auto;
        border: solid $primary;
        padding: 0 1;
        margin: 0 1;
        color: $text;
    }
    CatalogPane #catalog-hint {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Loading catalog…", id="catalog-status")
        # Model-scope dropdown — a "model name only" filter.  TOGGLE-HIDDEN like
        # the text filter ([\\] reveals it) so it stays out of the Tab/focus chain;
        # an always-visible focusable Select destabilised the suite.  The model
        # grouping is shown ALWAYS via the table's "model" column + group-by-model
        # sort (the switch.sh --list view); this dropdown is just the scope filter.
        # Seeded with "All models"; populate() refills it in registry order.
        yield Select(
            [(_ALL_MODELS_LABEL, _ALL_MODELS_VALUE)],
            id="catalog-model-select",
            value=_ALL_MODELS_VALUE,
            allow_blank=False,
        )
        yield Input(placeholder="filter slug / topology / engine / model / status… (space = AND)", id="catalog-filter")
        table: DataTable = DataTable(id="catalog-table", zebra_stripes=True)
        table.cursor_type = "row"
        yield table
        # #9/A8 — a compact preview strip for the highlighted row.  All LOCAL
        # reads off the CatalogEntry (status_note caveat + fit + ctx + measured
        # row) — the pick-decision input, updated on every cursor move (mirrors
        # the Operate·Containers·Config highlight pattern).  The full cross-rig
        # fold stays behind Explain ([e]).
        yield Static(
            "[dim]highlight a variant (move cursor) to preview it[/dim]",
            id="catalog-preview",
        )
        yield Label(
            "[dim]\\[\\\\] model   \\[/] filter   \\[⏎] serve   \\[e] explain   "
            "\\[d] set-default   \\[D] clear-default[/dim]",
            id="catalog-hint",
        )

    def on_mount(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        # Fold 3: the TPS / 8-pack columns are OUR-RIG measurements (cross-rig
        # rows live in the explain drill-down) — label them so cross-rig ambiguity
        # is gone now that the standalone Benchmarks tab is retired.
        # The Fit column moved OUT of the table into the serve confirm pop-up (the
        # fit verdict is a pick-the-serve decision input, shown when you ⏎ a row).
        # Fit is STILL computed (it feeds the pop-up + the serving-row exemption);
        # it just no longer occupies a Catalog column.
        # F6 — column budget: the money columns (ctx · TPS · 8pk · status) come
        # RIGHT after the identity (model · slug); topology/engine — largely
        # redundant with the slug, which encodes both — moved to the tail so a
        # 120-140-col terminal folds THEM, not the numbers a user picks by.
        # "(rig)" keeps the our-rig provenance at 4 chars ("our rig" cost 8 more).
        table.add_columns("model", "slug", "ctx", "TPS (rig)", "8pk (rig)", "status", "topo", "engine")
        # Full enriched catalog, and the current filter substring.
        self._entries: list[CatalogEntry] = []
        self._filter: str = ""
        # Model-scope dropdown selection ("" = all models); AND-combined with _filter.
        self._model_filter: str = ""
        # [h] toggle: hide 🗑️ deprecated slugs by default (mirrors `switch.sh --list`).
        self._show_deprecated: bool = False
        # Distinct model names (registry order) backing the dropdown — refreshed on load.
        self._models: list[str] = []
        # N3: the slug currently live-serving (from the estate's matched_slug),
        # so its Run-catalog row carries a "● serving" badge.  "" → none serving.
        self._serving_slug: str = ""
        # F9: the match grade for _serving_slug ("identity" | "shape" | "").
        self._serving_confidence: str = ""
        # A6: live per-GPU free-VRAM (GB) from the last estate poll, used to
        # DOWNGRADE a "● fits-clean" glyph that would actually OOM right now (e.g.
        # GPU0 holding ComfyUI).  None → unknown (the fit column is then labelled
        # "vs empty card" so the glyph is never read as a live verdict).
        self._free_gb_by_index: Optional[dict[int, float]] = None
        # Download UX banner: a non-empty note (set by the app when the model dir
        # is unset / missing) is prepended to the status line so the user is
        # prompted to set it ([S]).
        self._model_dir_note: str = ""

    def set_model_dir_note(self, note: str) -> None:
        """Set/clear the model-dir banner (e.g. '⚠ model dir not found — [S]')."""
        new = (note or "").strip()
        if new != self._model_dir_note:
            self._model_dir_note = new
            try:
                self.refresh_enriched()
            except Exception:
                pass

    # ── data ────────────────────────────────────────────────────────────────────

    def populate(self, entries: list[CatalogEntry], error: Optional[str]) -> None:
        """Fill the table with enriched catalog entries."""
        status_label = self.query_one("#catalog-status", Label)
        table = self.query_one("#catalog-table", DataTable)

        if error:
            self._entries = []
            table.clear()
            status_label.update(f"[red]Catalog error:[/red] {error}")
            table.add_row("—", "—", "—", "—", "—", "—", "—", "—")
            return

        self._entries = list(entries)
        self._refresh_model_options()
        self._render_rows()

    def _refresh_model_options(self) -> None:
        """Refill the model-scope dropdown with the DISTINCT model names in
        registry order (first-seen — same source switch.sh --list groups by), an
        "All models" sentinel first.  Uses _set_select_options so the programmatic
        refill doesn't fire Select.Changed (which would re-trigger the filter).
        Preserves the current scope if that model still exists, else falls to All."""
        models: list[str] = []
        for e in self._entries:
            if e.model and e.model not in models:
                models.append(e.model)
        self._models = models
        options = [(_ALL_MODELS_LABEL, _ALL_MODELS_VALUE)] + [(m, m) for m in models]
        keep = self._model_filter if self._model_filter in models else _ALL_MODELS_VALUE
        if self._model_filter and self._model_filter not in models:
            self._model_filter = ""  # the scoped model vanished from the catalog
        try:
            _set_select_options(
                self.query_one("#catalog-model-select", Select), options, default=keep
            )
        except Exception:
            pass

    def _render_rows(self) -> None:
        status_label = self.query_one("#catalog-status", Label)
        table = self.query_one("#catalog-table", DataTable)
        table.clear()

        rows = self._filtered_entries()
        serving = (self._serving_slug or "").strip()
        prev_model: Optional[str] = None  # blank-on-repeat → the switch.sh --list grouped look
        for e in rows:
            # source provenance — flag a coarse markdown scrape so a measurement
            # from BENCHMARKS.md is never mistaken for a structured record.
            meas_src = e.measurement.source
            tps = e.measurement.tps_label
            if meas_src == "benchmarks.md" and tps != "—":
                tps = f"{tps}*"
            # N3: mark the live-serving row so the running model is visible at a
            # glance in Run.  Driven by the estate's matched_slug.
            slug_cell = e.slug
            is_serving_row = bool(serving and e.slug == serving)
            if is_serving_row:
                if (self._serving_confidence or "") == "shape":
                    # F9: port/substring match — what's serving merely has this
                    # row's SHAPE (its port / a name fragment); it is NOT verified
                    # to BE this slug.  Never claim "serving" for a guess.
                    slug_cell = f"[yellow]👤[/yellow] {e.slug} [yellow]port in use[/yellow]"
                else:
                    slug_cell = f"[green]●[/green] {e.slug} [green]serving[/green]"
            else:
                # Download UX — a glyph for the on-disk state: ⏳NN% downloading,
                # ⬇ absent (not downloaded), ⚠ partial.  present/unknown → clean.
                dg = _weights_glyph(e)
                if dg:
                    slug_cell = f"{dg} {e.slug}"
            # Model cell — shown on the FIRST row of each model group only (rows are
            # grouped by model, registry order), blank on the repeats → the
            # switch.sh --list "model header, then its slugs" look.
            model_cell = f"[cyan]{e.model}[/cyan]" if e.model != prev_model else ""
            prev_model = e.model
            # The fit glyph is NO LONGER a Catalog column (it moved into the serve
            # confirm pop-up — see ConfirmActionScreen._render_serve_card).  Fit is
            # STILL computed here for the serving-row exemption + the pop-up; we
            # just don't emit a fit cell.
            table.add_row(
                model_cell,
                slug_cell,
                e.ctx_label or "—",
                tps,
                e.measurement.quality_label,
                _status_glyph(e.status),
                e.topology,
                e.engine,
            )

        banner = f"[yellow]{self._model_dir_note}[/yellow]  ·  " if self._model_dir_note else ""
        # Surface an active model scope (dropdown) the same way the text filter is shown.
        scope = f"model: [cyan]{self._model_filter}[/cyan]  ·  " if self._model_filter else ""
        # [h] hint: N 🗑️ deprecated slugs hidden (0 when revealed).
        dep_n = self._deprecated_hidden_count()
        dep_note = f"  ·  [dim]+{dep_n} deprecated hidden — h[/dim]" if dep_n else ""
        if self._filter or self._model_filter:
            tail = f"  ·  filter: {self._filter!r}" if self._filter else ""
            status_label.update(
                f"{banner}{scope}{len(rows)} / {len(self._entries)} variants{tail}{dep_note}"
            )
        else:
            star = "  ([dim]*[/dim] = BENCHMARKS.md scrape)" if self._has_md_scrape() else ""
            status_label.update(f"{banner}{len(self._entries)} variants loaded from registry{star}{dep_note}")

        # #9/A8 — keep the preview strip in sync with the cursor after a (re-)render
        # (enrichment mutates fit/measurement in place; the preview must reflect it).
        try:
            self.render_preview(self.selected_entry())
        except Exception:
            pass

    def refresh_enriched(self) -> None:
        """Re-render after background enrichment mutated the shared entries in
        place (fit / measurement), preserving the cursor row + active filter."""
        table = self.query_one("#catalog-table", DataTable)
        saved = table.cursor_row
        self._render_rows()
        if table.row_count:
            try:
                table.move_cursor(row=max(0, min(saved, table.row_count - 1)))
            except Exception:
                pass

    def set_serving_slug(self, slug: str, confidence: str = "") -> None:
        """N3: set (or clear, with "") the live-serving slug + re-render so the
        Run-catalog row badge stays fresh.  Cheap: only re-renders when the slug
        actually changed (so the periodic Operate poll doesn't churn the Run
        table on every tick).  Cursor + filter preserved via refresh_enriched.

        F9: ``confidence`` is the match grade ("identity" | "shape" | "") — a
        shape match renders the row badge as a guess, never as "serving"."""
        new = (slug or "").strip()
        new_conf = (confidence or "").strip()
        if new == (self._serving_slug or "").strip() and new_conf == (
            self._serving_confidence or ""
        ):
            return
        self._serving_slug = new
        self._serving_confidence = new_conf
        # Re-render only if rows are present (mount-order safe).
        if self._entries:
            self.refresh_enriched()

    def set_live_free_vram(self, free_gb_by_index: Optional[dict[int, float]]) -> None:
        """A6: feed the live per-GPU free-VRAM (GB) from the estate poll so the
        fit column reflects what actually fits RIGHT NOW.  Re-renders only when
        the value meaningfully changed (so the periodic Operate poll doesn't churn
        the Run table every tick).  Cursor + filter preserved via refresh_enriched."""
        new = free_gb_by_index or None

        def _key(d: Optional[dict[int, float]]) -> Optional[tuple]:
            if not d:
                return None
            # Round to whole GB so sub-GB jitter doesn't trigger a re-render.
            return tuple(sorted((i, round(v)) for i, v in d.items()))

        if _key(new) == _key(self._free_gb_by_index):
            return
        self._free_gb_by_index = new
        if self._entries:
            self.refresh_enriched()

    def _has_md_scrape(self) -> bool:
        return any(e.measurement.source == "benchmarks.md" for e in self._entries)

    def _filtered_entries(self) -> list[CatalogEntry]:
        # Hide 🗑️ deprecated slugs by default (mirrors `switch.sh --list`); [h] reveals them.
        pool = (
            self._entries
            if self._show_deprecated
            else [e for e in self._entries if (e.status or "").strip().lower() != "deprecated"]
        )
        # Model-scope dropdown first — AND-combined with the text filter below.
        base = (
            [e for e in pool if e.model == self._model_filter]
            if self._model_filter
            else pool
        )
        if not self._filter:
            return self._grouped_by_model(base)
        # Multi-word filter is AND-of-substrings: split the query on whitespace and
        # keep a row only when EVERY term is a (case-insensitive) substring of the
        # row's searchable text.  A single word reduces to the old contiguous
        # substring test; "gemma dual" now matches a gemma dual row (it did not
        # before, when the whole query was tested as one contiguous substring).
        terms = self._filter.lower().split()
        if not terms:
            return self._grouped_by_model(base)
        out: list[CatalogEntry] = []
        for e in base:
            hay = (
                f"{e.slug} {e.topology} {e.engine} {e.model} {e.status} {e.source}"
            ).lower()
            if all(t in hay for t in terms):
                out.append(e)
        return self._grouped_by_model(out)

    def _grouped_by_model(self, entries: list[CatalogEntry]) -> list[CatalogEntry]:
        """Stable group-by-model: a model's lanes sit together, models in registry
        (first-seen) order — the switch.sh --list view.  Intra-model order kept,
        so it stays consistent with the dropdown options and the cursor mapping."""
        rank = {m: i for i, m in enumerate(self._models)}
        return sorted(entries, key=lambda e: rank.get(e.model, len(rank)))

    def set_filter(self, text: str) -> None:
        self._filter = (text or "").strip()
        self._render_rows()

    def set_model_filter(self, model: str) -> None:
        """Scope the catalog to one model name ("" / the All sentinel = all models).
        AND-combines with the text filter.  Re-renders (cursor resets to the top of
        the now-narrowed set)."""
        new = (model or "").strip()
        if new in ("", _ALL_MODELS_VALUE):
            new = ""
        if new == self._model_filter:
            return
        self._model_filter = new
        self._render_rows()

    def toggle_deprecated(self) -> None:
        """[h] show/hide 🗑️ deprecated slugs (hidden by default, mirroring
        `switch.sh --list`).  Re-renders (cursor resets to the top of the
        now-widened / narrowed set)."""
        self._show_deprecated = not self._show_deprecated
        self._render_rows()

    def _deprecated_hidden_count(self) -> int:
        """How many 🗑️ deprecated slugs are currently hidden (0 once revealed)."""
        if self._show_deprecated:
            return 0
        return sum(
            1 for e in self._entries if (e.status or "").strip().lower() == "deprecated"
        )

    def toggle_model_select(self) -> None:
        """[\\] toggle the model-scope dropdown (hidden by default, like the text
        filter): reveal + focus, or hide + refocus the table.  Hidden → out of the
        Tab/focus chain.  Hiding keeps the scope (it shows in the model column +
        status line); re-open and pick "All models" to clear it."""
        sel = self.query_one("#catalog-model-select", Select)
        if "visible" in sel.classes:
            sel.remove_class("visible")
            self.query_one("#catalog-table", DataTable).focus()
        else:
            sel.add_class("visible")
            sel.focus()

    def selected_entry(self) -> Optional[CatalogEntry]:
        """The CatalogEntry under the table cursor, or None."""
        table = self.query_one("#catalog-table", DataTable)
        rows = self._filtered_entries()
        idx = table.cursor_row
        if 0 <= idx < len(rows):
            return rows[idx]
        return None

    def render_preview(self, entry: Optional[CatalogEntry]) -> None:
        """#9/A8 — render the compact preview strip for the highlighted entry.

        A pure LOCAL read off the CatalogEntry (no I/O — no kv-calc / explain
        re-run): the caveat ``status_note``, the fit (~VRAM/band/glyph) folded
        with the live-vs-empty-card note from B3, the max-ctx, and the last
        measured row.  Updated on every cursor move — the pick-decision input."""
        try:
            body = self.query_one("#catalog-preview", Static)
        except Exception:
            return
        if entry is None:
            body.update("[dim]highlight a variant (move cursor) to preview it[/dim]")
            return
        # Fit line — the B3 live-downgrade (now that the Catalog table has no fit
        # column, this preview strip + the serve pop-up are its only surfaces), so
        # the preview never reads "fits-clean" for a row that would OOM right now.
        serving = (self._serving_slug or "").strip()
        is_serving_row = bool(serving and entry.slug == serving)
        if is_serving_row:
            fit_glyph, fit_note = entry.fit.glyph, ""
        else:
            fit_glyph, fit_note = downgrade_fit_glyph(
                entry.fit, entry.row, self._free_gb_by_index
            )
        fit_line = f"{fit_glyph} {entry.fit.verdict}"
        vram = entry.fit.vram_est_gb
        if vram is not None:
            fit_line += f"  ~{float(vram):.1f} GiB"
            band = entry.fit.band_gb
            if band is not None:
                fit_line += f" / {float(band):.1f} GiB band"
        # N3 — only append the inline fit_note on a GENUINE live downgrade
        # (⚠/✗).  The "vs empty card" basis note is already shown by the trailing
        # "({fit_basis})" below, so appending it here too doubled it ("… vs empty
        # card (vs empty card)") when live free-VRAM is unknown.
        if fit_note and fit_glyph in ("⚠", "✗"):
            color = "yellow" if fit_glyph == "⚠" else "red"
            fit_line += f"  [{color}]{fit_note}[/{color}]"
        # B3 basis label so the glyph is never silently read as a live verdict.
        fit_basis = "vs live free-VRAM" if self._free_gb_by_index else "vs empty card"
        lines = [
            f"  [bold]{entry.slug}[/bold]  [dim]·[/dim]  {entry.engine}"
            f"  [dim]·[/dim]  {_status_glyph(entry.status)} {entry.status or '—'}",
            f"  [bold]fit[/bold]  {fit_line}  [dim]({fit_basis})[/dim]",
            f"  [bold]ctx[/bold]  {entry.ctx_label or '—'}"
            f"   [bold]measured[/bold]  {entry.measurement.tps_label} TPS"
            f"  ·  8pk {entry.measurement.quality_label}",
        ]
        note = (entry.status_note or "").strip()
        if note:
            lines.append(f"  [bold]caveat[/bold]  [yellow]{note}[/yellow]")
        body.update("\n".join(lines))

    def toggle_filter(self) -> None:
        inp = self.query_one("#catalog-filter", Input)
        if "visible" in inp.classes:
            inp.remove_class("visible")
            self.query_one("#catalog-table", DataTable).focus()
        else:
            inp.add_class("visible")
            inp.focus()

    def close_filter_if_open(self) -> bool:
        """Esc/cancel: hide the text filter and/or the model dropdown overlay and
        refocus the table. Returns True if one was open (so the app swallows the
        Esc). Hiding the model dropdown KEEPS the scope (only hides the control)."""
        closed = False
        sel = self.query_one("#catalog-model-select", Select)
        if "visible" in sel.classes:
            sel.remove_class("visible")
            closed = True
        inp = self.query_one("#catalog-filter", Input)
        if "visible" in inp.classes:
            inp.remove_class("visible")
            inp.value = ""
            self.set_filter("")
            closed = True
        if closed:
            self.query_one("#catalog-table", DataTable).focus()
        return closed


# ── Copy-to-clipboard support (Batch 4) ──────────────────────────────────────────


class _CopyableModal:
    """Mixin — a modal that holds paste-ready text exposes it to the global
    ``[Y]`` copy action via ``copyable_text()``.  The subclass stashes the RAW
    (markup-free) text in ``_copy_payload`` when its body is rendered.  This is
    how the report / explain / evidence overlays become one-keystroke copyable
    (their bodies are ``Static`` which CAN be drag-selected, but a context copy
    that doesn't depend on a mouse selection is more reliable).

    NOTE: a ``ModalScreen`` caps the binding chain at the modal (the whole point
    of a modal), so the app-level ``[Y]`` never reaches an open modal — each
    copyable modal binds ``Y`` → ``app.copy_context`` EXPLICITLY in its own
    BINDINGS (Textual does not merge BINDINGS from a non-DOMNode mixin)."""

    _copy_payload: str = ""

    def copyable_text(self) -> str:
        return self._copy_payload or ""


# ── Explain detail modal ────────────────────────────────────────────────────────


class ExplainScreen(_CopyableModal, ModalScreen):
    """Tier-3 detail overlay for one slug (switch.sh --explain --json)."""

    DEFAULT_CSS = """
    ExplainScreen {
        align: center middle;
    }
    ExplainScreen > Vertical {
        width: 84;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    ExplainScreen .explain-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    ExplainScreen #explain-body {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("e", "dismiss", "Close"),
        Binding("Y", "app.copy_context", "Copy", show=True),
    ]

    def __init__(self, slug: str, *, model: str = "", engine: str = "", **kwargs):
        super().__init__(**kwargs)
        self._slug = slug
        # (model, engine) drive the cross-rig benchmark fold (Fold 3).
        self._model = model
        self._engine = engine
        # Cached detail (our-rig story) so the cross-rig benchmark rows folded in
        # from the retired Benchmarks tab can be appended once they arrive.
        self._detail: Optional[dict] = None
        self._detail_error: Optional[str] = None
        self._cross_rig: list[BenchRow] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Explain · {self._slug}", classes="explain-title")
            yield Static("Loading detail…", id="explain-body")

    def on_mount(self) -> None:
        # Load detail + cross-rig AFTER mount (so the body query resolves) —
        # mirrors ConfirmActionScreen / EvidenceReportScreen.  Avoids the race
        # where a worker's set_detail runs before compose() mounts #explain-body.
        self.app.run_explain(self, self._slug)  # type: ignore[attr-defined]
        self.app.load_cross_rig_for_explain(  # type: ignore[attr-defined]
            self, self._model, self._engine
        )

    def set_detail(self, detail: Optional[dict], error: Optional[str]) -> None:
        self._detail = detail
        self._detail_error = error
        self._rerender()

    def set_cross_rig(self, rows: list[BenchRow]) -> None:
        """Fold the cross-rig benchmark rows (from the retired Validate ·
        Benchmarks tab / ``benchmarks_explorer``) for this slug's (model, engine)
        into the drill-down so that cross-rig data is never silently dropped."""
        self._cross_rig = list(rows)
        # Only re-render once detail has loaded (set_detail drives the body); if
        # cross-rig arrives first this is a no-op until set_detail fires.
        if self._detail is not None or self._detail_error is not None:
            self._rerender()

    def _rerender(self) -> None:
        body = self.query_one("#explain-body", Static)
        detail, error = self._detail, self._detail_error
        if error or detail is None:
            body.update(f"[red]explain failed:[/red] {error or 'no data'}")
            return
        reg = detail.get("registry", {}) or {}
        fit = detail.get("fit", {}) or {}
        benches = detail.get("benchmarks", []) or []
        lines: list[str] = []
        lines.append(f"  [bold]Model[/bold]   {reg.get('model', '—')}")
        lines.append(f"  [bold]Engine[/bold]  {reg.get('engine', '—')}")
        lines.append(f"  [bold]Status[/bold]  {_status_glyph(str(reg.get('status', '')))} {reg.get('status', '—')}")
        if reg.get("status_note"):
            lines.append(f"  [bold]Caveat[/bold]  [yellow]{reg.get('status_note')}[/yellow]")
        lines.append(f"  [bold]Card[/bold]    {detail.get('card', '—')}")
        verdict = str(fit.get("verdict", "—"))
        vram = fit.get("vram_est_gb")
        band = fit.get("band_gb")
        fit_line = verdict
        if vram is not None:
            fit_line += f"  ~{float(vram):.2f} GiB"
            if band is not None:
                fit_line += f" / {float(band):.1f} GiB band"
        lines.append(f"  [bold]Fit[/bold]     {fit_line}")
        if fit.get("max_ctx"):
            lines.append(f"  [bold]Max ctx[/bold] {fit.get('max_ctx')}")
        if benches:
            lines.append("")
            lines.append("  [bold]Measured (our rig)[/bold]")
            # Fix 3: the REAL shape is [{"row","columns"}]; TPS lives in
            # columns[4] — NOT invented {"narr_tps":…} keys.  Parse each
            # record via measurement_from_explain_columns so the modal shows
            # real numbers and never literal 'None/None'.
            for b in benches[-3:]:
                if not isinstance(b, dict):
                    continue
                m = measurement_from_explain_columns(b)
                n = f"{m.narr_tps:.0f}" if m.narr_tps is not None else "—"
                c = f"{m.code_tps:.0f}" if m.code_tps is not None else "—"
                q = m.quality_8pk or "—"
                d = m.date or ""
                lines.append(f"    {n}/{c} TPS · 8pk {q}  [dim]{d}[/dim]")
        else:
            lines.append("")
            lines.append("  [dim]no structured benchmarks for this slug[/dim]")
        # Cross-rig benchmark rows folded in from the retired Benchmarks tab — the
        # explorer corpus + BENCHMARKS.md scrapes for this (model, engine).  These
        # are NOT our-rig numbers; label them so cross-rig data isn't mistaken for
        # the local measurement.
        if self._cross_rig:
            lines.append("")
            lines.append("  [bold]Cross-rig benchmarks[/bold] [dim](other rigs / scrapes)[/dim]")
            for r in self._cross_rig[:6]:
                topo = r.topology or "—"
                ctx = r.max_ctx or "—"
                q = r.quality_label
                src = "md" if r.source == "benchmarks.md" else (r.source or "—")
                d = r.date or ""
                lines.append(
                    f"    {topo}: {r.tps_label} TPS · {ctx} · 8pk {q}  "
                    f"[dim]{src} {d}[/dim]"
                )
        lines.append("")
        lines.append("  [dim]Esc / e to close[/dim]")
        body.update("\n".join(lines))
        # Stash a markup-free copy of the detail for the global [Y] copy action.
        self._copy_payload = "\n".join(Text.from_markup(ln).plain for ln in lines[:-2])

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# Phase R-realign: the standalone Run · Bring-your-own tab (ByoPane) is REMOVED.
# The producer lane's ① Bring (LaneBringPane, defined below) is now the SINGLE
# bring-an-arbitrary-repo entry point.  The underlying ``byo_check`` data method +
# the shared verdict renderer (_byo_result_text) + the curated profile dropdown
# stay — ① Bring reuses them.


# ── Serve context (state-aware SERVE confirm) ────────────────────────────────────


class ServeContext(NamedTuple):
    """The state-aware context for a SERVE-of-a-catalog-slug confirm.

    Built by the app (``_serve_context_for``) when ⏎ stages a registry slug, this
    tells the confirm modal which controls to present:

      - ``mode == "stop"``   — the TARGET slug is the one ALREADY serving → the
        modal offers **Stop** (tear down THIS model, via ``stop_plan``) + Cancel.
        NO Start, NO Force.
      - ``mode == "start"``  — the target is NOT serving → the modal offers
        **Start** (the gated serve ``plan``) + Cancel.  Whether a CONFLICTING
        model holds the GPU is learned from the reconcile result once it lands:
        a conflict turns Start into a **warned Start** ("Starting this will STOP
        <model> on GPU N"), which performs teardown-then-serve — this is the old
        Force semantics FOLDED INTO Start (the maintainer's call), so there is no
        separate Force control.

    ``entry`` is the staged CatalogEntry (drives the card summary + fit verdict —
    rendered immediately from cached fields, BEFORE the background detect lands).
    ``stop_plan`` is the targeted-stop ActionPlan (``docker stop <container>``),
    present only for ``mode == "stop"``.
    """

    mode: str                                   # "stop" | "start" | "download" | "downloading"
    entry: Optional[CatalogEntry] = None
    stop_plan: Optional[ActionPlan] = None
    serving_container: str = ""                 # the container Stop tears down
    # Download UX modes (when the slug's weights aren't present):
    #   "download"    — weights ABSENT/partial + downloadable (hf_repo) → the modal
    #                   offers **Download** (Enter) instead of Start.  No reconcile
    #                   gate (a disk write, not a GPU claim).
    #   "downloading" — a download is already in flight for this slug → the modal
    #                   shows live ⏳NN% progress + **Cancel** (Esc closes, leaving
    #                   it running in the background).
    # Start is gated OFF until the weights are present.
    # force_required — the staged slug is NON-functional (experimental / preview /
    # incubating / upstream-gated / deprecated), which ``switch.sh`` REFUSES to
    # launch without ``--force``.  So a plain Start would just fail; the modal
    # presents a **Force Start** (passes ``--force``) with an "unvalidated config"
    # warning instead.  Production / caveats slugs leave this False (plain Start;
    # ``--force`` folds in only on a GPU conflict, per the warned-Start path).
    force_required: bool = False
    weights_state: str = ""                     # present|partial|absent|downloading|unknown


# ── Confirm modal (used for serve + scene + container writes) ────────────────────


class ConfirmActionScreen(ModalScreen):
    """The reconcile-gated confirm modal (design §7 #8 / §3.2).

    Shows the plan + the FRESH reconcile result (what this write would collide
    with / tear down), and only on confirm dispatches the write through the
    app's gated executor.

    Two presentations share ONE reconcile gate:

      - **SERVE of a catalog slug** (``serve_ctx`` set): a state-aware card
        (slug summary · fit verdict · reconcile/teardown status) with
        Start/Stop + Cancel as FOOTER key-hints — no on-screen button row, no
        phantom placeholder, no separate Force.  The card paints IMMEDIATELY
        from the staged entry's cached fields; the destructive Start/Stop key is
        disabled until the background detect (the dual-writer gate) resolves.
      - **Every other write** (scene-switch / estate-off / container / default /
        validation / generated serve — ``serve_ctx`` None): the original
        Confirm/Force/Cancel semantics, UNCHANGED.  When the gate is unsafe the
        primary action is disabled and the explicit Force override (routed back
        with a reason) is the only path.
    """

    DEFAULT_CSS = """
    ConfirmActionScreen {
        align: center middle;
    }
    ConfirmActionScreen > Vertical {
        width: 80;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    ConfirmActionScreen .confirm-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    ConfirmActionScreen #confirm-body {
        height: auto;
        margin-bottom: 1;
    }
    ConfirmActionScreen #confirm-btn-row {
        height: 3;
        align: left middle;
    }
    ConfirmActionScreen Button {
        margin-right: 1;
    }
    """

    # The commit affordances live ONLY in the modal footer (show=True BINDINGS),
    # gated per-state by check_action — there is no duplicate on-screen button row
    # and no phantom/empty placeholder.  priority=True so these win over any
    # focused widget's own ``enter``→press.
    #
    # SERVE (serve_ctx set):
    #   - mode "stop"  → ``Stop`` (k) + Cancel.  No start/confirm, no force.
    #   - mode "start" → ``Start`` (enter) + Cancel.  When the reconcile result
    #     shows a CONFLICT the Start is a WARNED start (teardown-then-serve, the
    #     old force semantics folded in) — still the SAME ``start`` action, just
    #     relabelled "Start (stops <model>)" in the footer (set_reconcile swaps
    #     this instance's per-instance ``start`` Binding description, so the
    #     destructive affordance is visible in the footer, not only the card).
    # NON-SERVE (serve_ctx None): the original ``confirm`` / ``force`` / Cancel.
    #
    # SAFETY: the destructive keys (confirm/start/stop/force) are gated off until
    # reconcile RESOLVES (rec is not None) — Enter/Stop on an unresolved or unsafe
    # gate never fires a write (check_action False + the on_key guard).
    BINDINGS = [
        Binding("enter", "confirm", "Confirm", show=True, priority=True),
        Binding("enter", "start", "Start", show=True, priority=True),
        # Download UX: a 3rd ``enter`` (Download — weights absent) + ``k`` Cancel
        # for a download in flight.  Disjoint with start/stop via check_action
        # (mode-gated), same proven pattern as the dual confirm/start enter.
        Binding("enter", "download", "Download", show=True, priority=True),
        Binding("k", "stop", "Stop", show=True, priority=True),
        Binding("k", "cancel_download", "Cancel download", show=True, priority=True),
        Binding("f", "force", "Force", show=True, priority=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, plan: ActionPlan, *, on_confirm=None, serve_ctx=None, **kwargs):
        super().__init__(**kwargs)
        self._plan = plan
        self._reconcile: Optional[ReconcileResult] = None
        # Optional alternate commit path.  Default (None) → the app's gated
        # ``dispatch_action`` (execute_action).  Set for launches that don't go
        # through execute_action — notably validation runs, which stream via
        # ``run_validation`` into the Run LivePane and never claim a GPU.  The
        # callback receives the (possibly force-reissued) plan.
        self._on_confirm = on_confirm
        # The state-aware SERVE context (None → the legacy Confirm/Force modal).
        self._serve_ctx: Optional[ServeContext] = serve_ctx

    # ── presentation predicates ───────────────────────────────────────────────────

    @property
    def _is_serve(self) -> bool:
        """True iff this is a state-aware SERVE-of-a-catalog-slug confirm."""
        return self._serve_ctx is not None

    @property
    def _is_stop_mode(self) -> bool:
        return bool(self._serve_ctx is not None and self._serve_ctx.mode == "stop")

    @property
    def _serve_mode(self) -> str:
        return self._serve_ctx.mode if self._serve_ctx is not None else ""

    @property
    def _is_download_modal(self) -> bool:
        """Download UX mode — 'download' (offer Download) or 'downloading' (show
        progress + Cancel).  A DISK action: NO reconcile gate (claims no GPU)."""
        return self._serve_mode in ("download", "downloading")

    @property
    def _has_conflict(self) -> bool:
        """Reconcile landed AND it is unsafe (a live model holds the GPU)."""
        rec = self._reconcile
        return bool(rec is not None and not rec.safe)

    def compose(self) -> ComposeResult:
        if self._is_serve:
            ctx = self._serve_ctx
            verb = {
                "stop": "Stop", "download": "Download", "downloading": "Downloading",
            }.get(ctx.mode, "Start")
            slug = getattr(ctx.entry, "slug", "") or self._serve_slug()
            with Vertical():
                yield Label(f"{verb} · {slug}", classes="confirm-title")
                yield Static("", id="confirm-body")
                # Controls live ONLY in the footer (no on-screen button row, no
                # phantom placeholder) — matches the rest of the app.
                yield Footer()
            return
        with Vertical():
            yield Label(f"Confirm · {self._plan.description}", classes="confirm-title")
            yield Static("Re-running reconcile gate (fresh detect)…", id="confirm-body")
            # Controls live ONLY in the footer (Confirm / the override / Cancel),
            # gated per-state by check_action — same as the serve modal.  No
            # duplicate on-screen button row (the old buttons + footer showed the
            # same keys twice).
            yield Footer()

    def on_mount(self) -> None:
        # FASTER PAINT: render the card IMMEDIATELY from cached fields, THEN run
        # the fresh reconcile detect in the BACKGROUND (run_reconcile_for_modal is
        # a @work worker).  The card never blocks on the detect; only the
        # destructive Start/Stop is disabled until the detect lands (set_reconcile
        # enables it).  The WRITE still waits for the detect — paint un-blocks, the
        # write never does.
        if self._is_serve:
            self._render_serve_card()
        # Download modes are a DISK action — no GPU claim, so NO reconcile gate.
        # Render the card + (while downloading) tick live ⏳NN% progress; the
        # Download/Cancel keys are enabled immediately (no detect to wait for).
        if self._is_download_modal:
            if self._serve_mode == "downloading":
                self._dl_timer = self.set_interval(1.0, self._render_serve_card)
            self.refresh_bindings()
            return
        # Re-run the gate (fresh detect) before enabling any commit affordance.
        self._set_force_binding_label(self._force_label())   # footer override label
        self.app.run_reconcile_for_modal(self, self._plan)  # type: ignore[attr-defined]
        self.refresh_bindings()

    def _serve_slug(self) -> str:
        ctx = self._serve_ctx
        if ctx is not None and ctx.entry is not None:
            return getattr(ctx.entry, "slug", "") or ""
        return ""

    def _download_card_text(self, entry) -> str:
        """The Download-UX card body — what fetching this slug's weights entails
        (mode 'download') or live progress (mode 'downloading')."""
        w = getattr(entry, "weights", None) if entry is not None else None
        slug = getattr(entry, "slug", "") or self._serve_slug()
        if self._serve_mode == "downloading":
            pct = getattr(entry, "download_pct", None)
            # Prefer the slug-keyed tracker: it survives a catalog refresh that
            # rebuilt `entry` (whose download_pct is briefly None until the worker's
            # next tick re-stamps it), so the card never falls back to "starting…".
            try:
                info = self.app._active_downloads().get(slug)  # type: ignore[attr-defined]
                if info is not None and info.get("pct") is not None:
                    pct = info["pct"]
            except Exception:
                pass
            bar = ""
            if pct is not None:
                fill = max(0, min(20, round(pct / 5)))
                bar = f"  [{'█' * fill}{'░' * (20 - fill)}] {pct}%"
            repo = (w.hf_repo if w else "") or "?"
            return (
                f"  [bold]{slug}[/bold]\n"
                f"  downloading [dim]{repo}[/dim]\n"
                f"{bar or '  ⏳ starting…'}\n\n"
                "  [dim]⏳ runs in the background — Esc closes this (download "
                "continues) · [cyan]k[/cyan] cancels the download[/dim]"
            )
        # mode "download"
        if w is None or not w.hf_repo:
            return (
                f"  [bold]{slug}[/bold]\n\n"
                "  [yellow]⚠ no direct download recipe[/yellow] — these weights are "
                "manual (no HF repo wired).\n  See the model profile's manual_note."
            )
        size = f"~{w.size_gb:.0f} GiB" if w.size_gb else "size unknown"
        fits, free_gb, need_gb = self.app._data.weights_fits_disk(w)  # type: ignore[attr-defined]
        disk = (
            f"  [green]● fits[/green] — {free_gb:.0f} GiB free / ~{need_gb:.0f} GiB needed"
            if fits else
            f"  [red]✗ may not fit[/red] — {free_gb:.0f} GiB free / ~{need_gb:.0f} GiB needed"
        )
        warn = ""
        if (getattr(entry, "weights_state", "") or "") == WEIGHTS_PARTIAL:
            warn = "  [yellow]⚠ partial download on disk — Download resumes it[/yellow]\n"
        return (
            f"  [bold]{slug}[/bold]   [dim]weights not on disk[/dim]\n"
            f"  repo   [dim]{w.hf_repo}[/dim]   ({size})\n"
            f"{disk}\n"
            f"{warn}\n"
            "  [dim]⏎ Download (streams in the background, progress in the list) · "
            "Esc Cancel[/dim]"
        )

    # ── SERVE card (immediate, then reconcile-enriched) ───────────────────────────

    def _render_serve_card(self) -> None:
        """Render the 'what am I about to do' card from the staged CatalogEntry's
        CACHED fields (model · quant/engine · KV/ctx · measured TPS/8-pack · status
        · fit verdict).  Painted on mount — does NOT wait for the detect.  The
        reconcile/teardown line is filled in by set_reconcile once it lands."""
        try:
            body = self.query_one("#confirm-body", Static)
        except Exception:
            return
        ctx = self._serve_ctx
        entry = ctx.entry if ctx is not None else None
        if self._is_download_modal:
            body.update(self._download_card_text(entry))
            return
        lines: list[str] = []
        if entry is not None:
            lines.append(
                f"  [bold]{entry.slug}[/bold]   "
                f"{_status_glyph(entry.status)} {entry.status or '—'}"
            )
            lines.append(
                f"  [bold]model[/bold]  {entry.model or '—'}"
                f"   [bold]engine[/bold] {entry.engine or '—'}"
                f"   [bold]topo[/bold] {entry.topology or '—'}"
            )
            lines.append(
                f"  [bold]ctx[/bold]    {entry.ctx_label or '—'}"
                f"   [bold]measured[/bold] {entry.measurement.tps_label} TPS"
                f"  ·  8pk {entry.measurement.quality_label}"
            )
            # Fit verdict — the fit that moved OUT of the Catalog column now lives
            # here.  Glyph + verdict + ~VRAM/band for THIS rig.
            fv = entry.fit
            fit_line = f"{fv.glyph} {fv.verdict}"
            if fv.vram_est_gb is not None:
                fit_line += f"  ~{float(fv.vram_est_gb):.1f} GiB"
                if fv.band_gb is not None:
                    fit_line += f" / {float(fv.band_gb):.1f} GiB band"
            lines.append(f"  [bold]fit[/bold]    {fit_line}")
            note = (entry.status_note or "").strip()
            if note:
                lines.append(f"  [bold]caveat[/bold] [yellow]{note}[/yellow]")
        else:
            lines.append(f"  [bold]{self._serve_slug() or self._plan.description}[/bold]")
        # Non-functional slug → Force Start serves an UNVALIDATED config (switch.sh
        # refuses it without --force).  Make that explicit, not just implied by the
        # status badge.
        ctx = self._serve_ctx
        if ctx is not None and ctx.force_required and ctx.mode == "start":
            lines.append(
                "  [yellow]⚠ experimental — Force Start serves an UNVALIDATED "
                "config (not gate-passed).[/yellow]"
            )
        # The reconcile / teardown line — a placeholder until the detect lands.
        lines.append("")
        lines.append(self._reconcile_line())
        body.update("\n".join(lines))

    def _reconcile_line(self) -> str:
        """The reconcile/teardown status line for the SERVE card.

        - before reconcile resolves: "checking rig state…" (Start/Stop disabled).
        - STOP mode: the running container being torn down.
        - START · gate clear: nothing to tear down.
        - START · conflict: the conflicting model(s) Start would stop, with GPU.
        """
        rec = self._reconcile
        ctx = self._serve_ctx
        if rec is None:
            return "  [dim]checking rig state…[/dim]"
        if ctx is not None and ctx.mode == "stop":
            con = ctx.serving_container or self._serve_slug() or "the running model"
            return f"  [yellow]⏹ stops[/yellow] container [red]{con}[/red] (frees its GPUs)"
        if rec.safe:
            return "  [green]● gate clear[/green] — GPU free, nothing to tear down."
        # START with a conflict → warn that Start performs teardown-then-serve.
        parts: list[str] = []
        for c in rec.conflicts:
            g = f" on GPU {c.gpus}" if c.gpus else ""
            parts.append(f"[red]{c.slug or c.name}[/red]{g}")
        for gc in rec.gpu_conflicts:
            parts.append(f"[red]GPU{gc.gpu_index}[/red] ({gc.mem_used_mib} MiB)")
        for inst in rec.estate_claims:
            parts.append(f"estate [red]{inst.get('name', '?')}[/red] (GPU {inst.get('gpus', [])})")
        who = ", ".join(parts) if parts else "the running model"
        return f"  [yellow]⚠ Starting this will STOP[/yellow] {who} — then serve."

    def _primary_conflict_name(self) -> str:
        """A SHORT, markup-free name for the model a warned-Start would stop —
        for the footer relabel (the card's _reconcile_line carries the full,
        marked-up list).  Same source order: container slug/name → estate name →
        bare GPU → generic."""
        rec = self._reconcile
        if rec is None:
            return "running model"
        for c in rec.conflicts:
            if c.slug or c.name:
                return c.slug or c.name
        for inst in rec.estate_claims:
            name = inst.get("name")
            if name:
                return name
        for gc in rec.gpu_conflicts:
            return f"GPU{gc.gpu_index}"
        return "running model"

    def _set_start_binding_label(self, label: str) -> None:
        """Relabel THIS modal's footer Start key in place.

        ``self._bindings`` is a per-instance copy of the class BINDINGS (DOMNode
        copies it at init), so swapping the frozen ``start`` Binding for one with
        a new ``description`` affects ONLY this short-lived modal — never the
        class or a sibling modal.  The Footer renders ``binding.description``, so
        this is how the warned conflict-Start surfaces "Start (stops <model>)" in
        the footer.  Caller must refresh_bindings() afterwards (set_reconcile
        does)."""
        for bindings in self._bindings.key_to_bindings.values():
            for i, b in enumerate(bindings):
                if b.action == "start" and b.description != label:
                    bindings[i] = dataclasses.replace(b, description=label)
                    return

    def set_reconcile(self, rec: ReconcileResult) -> None:
        """Render the reconcile verdict + enable the appropriate commit path.

        Two presentations:
          - SERVE card → refresh the teardown line + enable Start/Stop now that
            the dual-writer detect has resolved.
          - legacy Confirm/Force modal → the original safe/unsafe rendering.
        """
        self._reconcile = rec
        if self._is_serve:
            self._render_serve_card()
            # Surface the teardown in the FOOTER too (not only the card body): on a
            # conflicting Start, relabel the footer key "Start (stops <model>)" so
            # the destructive affordance is visible where the user reads controls.
            ctx = self._serve_ctx
            if ctx is not None and ctx.mode == "start":
                # "Force Start" when the slug is non-functional (experimental →
                # switch.sh needs --force); "(stops <model>)" appended when a live
                # model holds the GPU.  Both are the SAME --force re-issue.
                verb = "Force Start" if ctx.force_required else "Start"
                label = (
                    f"{verb} (stops {self._primary_conflict_name()})"
                    if self._has_conflict
                    else verb
                )
                self._set_start_binding_label(label)
            # The destructive Start/Stop is now safe to ENABLE (the fresh detect
            # landed).  Footer reflects it via check_action.
            self.refresh_bindings()
            # Focus nothing destructive: there is no button row.  The footer keys
            # (Start/Stop/Cancel) carry the affordance; a stray Enter on an
            # unresolved gate is already guarded (check_action False before rec).
            return

        body = self.query_one("#confirm-body", Static)
        force_label = self._force_label()

        lines: list[str] = [f"  [bold]Command[/bold]  {' '.join(self._plan.cmd)}"]
        wanted = ", ".join(str(g) for g in rec.pending_gpus) if rec.pending_gpus else "—"
        lines.append(f"  [bold]GPUs[/bold]     {wanted}")

        if rec.safe:
            lines.append("")
            lines.append("  [green]● gate clear[/green] — nothing live overlaps the requested GPUs.")
            lines.append("  [dim]⏎ Confirm (streams below) · Esc Cancel[/dim]")
        else:
            lines.append("")
            lines.append("  [yellow]⚠ this will tear down / collide with:[/yellow]")
            for c in rec.conflicts:
                g = f" (GPU {c.gpus})" if c.gpus else ""
                slug = f"  [{c.slug}]" if c.slug else ""
                lines.append(f"    • container [red]{c.name}[/red]{g}{slug}")
            for gc in rec.gpu_conflicts:
                lines.append(
                    f"    • GPU{gc.gpu_index} busy ([red]{gc.mem_used_mib} MiB[/red])"
                )
            for inst in rec.estate_claims:
                name = inst.get("name", "?")
                gpus = inst.get("gpus", [])
                lines.append(f"    • estate instance [red]{name}[/red] (GPU {gpus})")
            if rec.note:
                lines.append(f"  [dim]{rec.note}[/dim]")
            lines.append("")
            lines.append(
                f"  [dim]Confirm is disabled — press [cyan]f[/cyan] to "
                f"[yellow]{force_label}[/yellow] (stops the above).[/dim]"
            )

        body.update("\n".join(lines))
        # Footer-only controls (no button row): relabel the override key per plan
        # ('Switch anyway' / 'Stop everything' / 'Force') + refresh so check_action
        # shows Confirm/override/Cancel for the resolved safe/unsafe verdict.  A
        # stray Enter on an unsafe gate is inert (check_action False + the on_key
        # guard); the override is the explicit `f` key.
        self._set_force_binding_label(force_label)
        self.refresh_bindings()

    # ── key handlers (footer-only — no buttons) ───────────────────────────────────

    def _force_label(self) -> str:
        """Context-appropriate label for the override key (was always 'Force').

        A scene switch tears down what's running, so 'Switch anyway' reads
        clearer; the comprehensive 'off'/stop_all → 'Stop everything'; estate-down
        → 'Stop anyway'; anything else keeps 'Force'."""
        desc = (self._plan.description or "").lower()
        if "stop all" in desc:
            return "Stop everything"
        if self._plan.kind == "scene":
            return "Switch anyway"
        if self._plan.kind == "estate_down":
            return "Stop anyway"
        return "Force"

    def _set_force_binding_label(self, label: str) -> None:
        """Relabel THIS modal's footer override key in place (per-instance copy,
        same mechanism as _set_start_binding_label)."""
        for bindings in self._bindings.key_to_bindings.values():
            for i, b in enumerate(bindings):
                if b.action == "force" and b.description != label:
                    bindings[i] = dataclasses.replace(b, description=label)
                    return

    def on_key(self, event) -> None:
        """SAFETY belt-and-suspenders: stop a stray ``enter`` whenever the
        destructive primary (``confirm`` / ``start``) is gated off (gate unsafe in
        the legacy modal, or not yet resolved).

        check_action returning False for the destructive action makes its
        run_action a no-op but does NOT stop the key event, so Enter would
        otherwise propagate to whatever is focused (in the legacy modal that could
        be the Force button → forcing the very teardown the gate guards).  We stop
        it here so an unresolved/unsafe-gate Enter is genuinely inert.  The
        explicit `f` (legacy force) / `k` (serve stop) keys are untouched."""
        if event.key != "enter":
            return
        # Stop Enter iff NO enter-bound action is currently enabled.  In the serve
        # modal the active enter-action depends on mode (start | download); the
        # legacy modal uses confirm.  Stop/downloading modes have NO enter-action
        # (their key is `k`), so Enter is correctly inert there too.
        enter_actions = ("confirm", "start", "download") if self._is_serve else ("confirm",)
        if not any(self.check_action(a, ()) is True for a in enter_actions):
            event.stop()
            event.prevent_default()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Gate the footer visibility + run_action of every commit key on state.

        SERVE (serve_ctx set) — only the right one verb shows, and ONLY after the
        dual-writer detect resolves (rec is not None):
          - mode "stop"  → ``stop`` shown; confirm/start/force hidden.
          - mode "start" → ``start`` shown; stop/confirm/force hidden.  (The
            conflict warned-Start is the SAME ``start`` action — the teardown is
            folded into _commit, so there is no separate Force.)
        LEGACY (serve_ctx None) — the original Confirm/Force gating:
          - rec None: neither commit key shown (buttons disabled too).
          - gate SAFE   → ``confirm`` shown, ``force`` hidden.
          - gate UNSAFE → ``confirm`` hidden, ``force`` shown.
        Esc/Cancel is always shown.  Inertness on an unresolved/unsafe gate is NOT
        free: it relies on (1) the legacy unsafe branch focusing Cancel (not
        Force), and (2) the on_key guard stopping a stray ``enter`` whenever the
        destructive primary is gated off."""
        rec = self._reconcile
        resolved = rec is not None
        if self._is_serve:
            ctx = self._serve_ctx
            mode = ctx.mode if ctx is not None else ""
            # Download UX modes are a DISK action (no reconcile gate) — enabled
            # immediately, no ``resolved`` requirement.
            if action == "download":
                return mode == "download"
            if action == "cancel_download":
                return mode == "downloading"
            # Start/Stop need the fresh detect to have resolved (GPU writes).
            if action == "stop":
                return bool(resolved and mode == "stop")
            if action == "start":
                return bool(resolved and mode == "start")
            # confirm/force never apply in serve mode.
            if action in ("confirm", "force"):
                return False
            return True
        if action == "confirm":
            return bool(resolved and rec.safe)
        if action == "force":
            return bool(resolved and not rec.safe)
        # stop/start/download/cancel_download never apply in the legacy modal.
        if action in ("stop", "start", "download", "cancel_download"):
            return False
        return True

    def action_confirm(self) -> None:
        """Enter → commit (legacy modal; only reachable when the gate is safe —
        check_action disables this binding otherwise)."""
        self._commit(force=False)

    def action_force(self) -> None:
        """f → FORCE the teardown (legacy modal; only reachable when the gate is
        unsafe — check_action disables this binding otherwise).  Routes through
        the SAME _commit(force=True) path the old raw on_key handler used."""
        self._commit(force=True)

    def action_start(self) -> None:
        """Enter → START the serve (serve mode).  Only reachable once the
        dual-writer detect RESOLVES (check_action gates it off until then).

        ``--force`` is added in TWO cases: (1) a conflicting model holds the GPU →
        teardown-then-serve (the old Force semantics folded into Start), and
        (2) the slug is NON-functional (experimental/…) → switch.sh refuses it
        without --force, so a Force Start is the only way to launch it (the card +
        footer warn it's an unvalidated config).  Both are the SAME --force
        re-issue; on a clear gate for a functional slug this is a plain serve."""
        ctx = self._serve_ctx
        force_required = bool(ctx is not None and ctx.force_required)
        self._commit(force=self._has_conflict or force_required)

    def action_stop(self) -> None:
        """k → STOP the targeted serving model (serve mode, stop-context only).
        Dispatches the targeted-stop ActionPlan (``docker stop <container>``) — the
        SAME reconcile-gated B3 serving-panel stop path.  Only reachable once the
        detect resolves (check_action gates it off until then)."""
        ctx = self._serve_ctx
        if ctx is None or ctx.stop_plan is None:
            return
        self.app.pop_screen()
        self.app.dispatch_action(ctx.stop_plan)  # type: ignore[attr-defined]

    def action_download(self) -> None:
        """⏎ in download mode → start the weights download (DISK action, no GPU
        gate).  Closes the modal; progress shows live in the catalog listing."""
        ctx = self._serve_ctx
        if ctx is None or ctx.entry is None:
            return
        self.app.pop_screen()
        self.app.start_download(ctx.entry)  # type: ignore[attr-defined]

    def action_cancel_download(self) -> None:
        """k in downloading mode → cancel the in-flight download (Esc just closes,
        leaving it running)."""
        ctx = self._serve_ctx
        if ctx is None or ctx.entry is None:
            return
        self.app.pop_screen()
        self.app.cancel_download(ctx.entry.slug)  # type: ignore[attr-defined]

    def _commit(self, *, force: bool) -> None:
        plan = self._plan
        if force and not plan.force:
            # Re-issue the plan as a forced one (with a surfaced reason) so the
            # executor's force path is taken explicitly — never silently.  In serve
            # mode this is the warned-Start teardown-then-serve; in the legacy
            # modal it is the Force override.
            reason = (
                "user accepted teardown via warned Start (stop conflicting model, then serve)"
                if self._is_serve
                else "user accepted teardown via Force override"
            )
            plan = ActionPlan(
                kind=plan.kind,
                cmd=_with_force(plan),
                description=plan.description + " (FORCED)",
                is_write=plan.is_write,
                requires_reconcile=plan.requires_reconcile,
                force=True,
                force_reason=reason,
            )
        self.app.pop_screen()
        if self._on_confirm is not None:
            # Alternate commit path (e.g. a validation launch that streams via
            # run_validation rather than the gated execute_action).
            self._on_confirm(plan)
            return
        # Hand the actual (gated, mocked-in-test) execution back to the app.
        self.app.dispatch_action(plan)  # type: ignore[attr-defined]

    def action_cancel(self) -> None:
        self.app.pop_screen()


def _with_force(plan: ActionPlan) -> list[str]:
    """Insert --force into a serve switch.sh command for the forced re-issue.

    Only the serve (switch.sh) plan supports --force; for other kinds the
    command is unchanged (the force flag just relaxes the gate refusal)."""
    cmd = list(plan.cmd)
    if plan.kind == "serve" and "scripts/switch.sh" in cmd and "--force" not in cmd:
        # switch.sh --force <slug>: insert before the slug (last positional).
        cmd.insert(len(cmd) - 1, "--force")
    return cmd


# ── Operate · Orchestration ───────────────────────────────────────────────────────


class OperateOrchPane(Container):
    """Operate / Orchestration tab: GPU cards, Doctor, scene table, services."""

    DEFAULT_CSS = """
    OperateOrchPane {
        height: 1fr;
    }
    OperateOrchPane #orch-scroll {
        height: 1fr;
    }
    OperateOrchPane .gpu-card {
        border: solid $primary;
        padding: 0 1;
        margin: 0 1 1 1;
        height: auto;
    }
    OperateOrchPane .gpu-card-title {
        text-style: bold;
        color: $accent;
    }
    OperateOrchPane #estate-error-strip {
        display: none;
        padding: 0 1;
        margin: 0 1 0 1;
        color: $error;
        text-style: bold;
    }
    OperateOrchPane #estate-error-strip.visible {
        display: block;
    }
    OperateOrchPane #serving-line {
        padding: 0 1;
        margin: 0 1 0 1;
        text-style: bold;
    }
    OperateOrchPane #doctor-line {
        padding: 0 1;
        margin: 0 1 1 1;
        color: $text;
    }
    OperateOrchPane #scene-heading {
        text-style: bold;
        padding: 0 1;
        margin: 0 1 0 1;
    }
    OperateOrchPane DataTable {
        height: auto;
        margin: 0 1 1 1;
    }
    OperateOrchPane #scene-preview {
        height: auto;
        border: solid $primary;
        padding: 0 1;
        margin: 0 1 1 1;
        color: $text;
    }
    OperateOrchPane #orch-hint {
        padding: 0 1;
        margin: 0 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        # FOLD 3 — the scroll wrapper must NOT be a Tab stop: with can_focus left at
        # its ScrollableContainer default (True), descending into Orchestration via
        # Tab lands on the scroll container first (a dead stop needing a 2nd Tab to
        # reach #scene-table).  can_focus=False removes it from the Tab chain so a
        # single Tab from the tab bar reaches the scene list; it still scrolls via
        # its inner content / the mouse wheel.
        orch_scroll = ScrollableContainer(id="orch-scroll")
        orch_scroll.can_focus = False
        with orch_scroll:
            # A2/N2: a one-line red strip that only shows when the estate READ
            # hit a docker / nvidia-smi failure (state.error).  DISTINCT from the
            # calm "○ no model serving" idle line below — a read failure is not an
            # idle rig.  Hidden (display:none) until populate() finds an error.
            yield Static("", id="estate-error-strip")
            with Container(classes="gpu-card", id="gpu0-card"):
                yield Label("GPU0", classes="gpu-card-title")
                yield Static("[dim]querying nvidia-smi…[/dim]", id="gpu0-bar")
            with Container(classes="gpu-card", id="gpu1-card"):
                yield Label("GPU1", classes="gpu-card-title")
                yield Static("[dim]querying nvidia-smi…[/dim]", id="gpu1-bar")
            yield Static("[dim]reading estate…[/dim]", id="serving-line")
            yield Static("[dim]reading health.sh…[/dim]", id="doctor-line")
            yield Label("Scenes  [dim](⏎ to switch — gated)[/dim]", id="scene-heading")
            scene_table: DataTable = DataTable(
                id="scene-table", zebra_stripes=True, show_cursor=True
            )
            scene_table.cursor_type = "row"
            yield scene_table
            # #11 — a compact preview of the highlighted scene (what switching to
            # it brings up): its description + services + ports + GPUs.  A pure
            # LOCAL read off the Scene the gpu-mode --list-modes poll carried —
            # mirrors the catalog/containers highlight-preview pattern.
            yield Static(
                "[dim]highlight a scene (move cursor) to preview what it brings up[/dim]",
                id="scene-preview",
            )
            yield Label(
                "[dim]\\[k] stop this model (gated)   \\[b] restart serving (gated)   "
                "\\[n] switch model   \\[⏎] switch scene (gated)   \\[o] stop all (gated)   "
                "\\[c] power cap… (default / clear / custom, gated)   "
                "[dim](per-service start/stop → Containers tab)[/dim][/dim]",
                id="orch-hint",
            )
            # FIX 3 — the host disk-usage bars + system-RAM line MOVED out of this
            # sub-tab to the global left-rail (HostStatsRail / #host-stats-rail),
            # per the maintainer's "estate column on the left" directive.

    def on_mount(self) -> None:
        t = self.query_one("#scene-table", DataTable)
        t.add_columns("Scene", "Group", "GPUs", "Services")
        self._scenes: list[Scene] = []
        # #11-ext — per-service running states for the HIGHLIGHTED scene, so the
        # preview can render a live status bullet (● running · ○ stopped) per
        # service.  Read-only for now (per-service actions live in the Containers
        # tab); kept on the pane in case an interactive surface is added later.
        self._scene_svc_states: list[SceneServiceState] = []
        # FIX 1 — last-rendered scene-row signature (skip-if-unchanged guard).
        self._scene_rows_sig: Optional[tuple] = None
        # #10: GPU-index → active cap (W) so the GPU cards can show "(cap NNNW)".
        # Populated from the power-cap READ; only set when a card is below its
        # default (genuinely capped) so an uncapped card shows no spurious cap.
        self._gpu_cap: dict[int, float] = {}
        # Cache the last estate state so a later power-cap read can re-render the
        # GPU cards with the cap note (power-cap is read AFTER the estate poll).
        self._last_state: Optional[EstateState] = None
        # #12/N5 + attribution: the last Batch-5 telemetry (disk / RAM / GPU-VRAM
        # owners).  Cached so a re-render (e.g. after the power-cap read) keeps the
        # GPU-card "held by:" line, which is derived from this not the estate poll.
        self._last_telemetry: Optional[EstateTelemetry] = None
        # A7: matched catalog slug's claimed ctx (set by populate()).
        self._catalog_ctx_label: str = ""
        self._catalog_ctx: Optional[int] = None

    # ── data ────────────────────────────────────────────────────────────────────

    def populate(
        self,
        state: EstateState,
        *,
        catalog_ctx_label: str = "",
        catalog_ctx: Optional[int] = None,
    ) -> None:
        self._last_state = state
        # A7: the matched catalog slug's CLAIMED ctx — both a display label (e.g.
        # "262K") and the NUMERIC claim (e.g. 262144).  The numeric one drives the
        # divergence comparison so a colloquial label doesn't trip a false badge;
        # the label is for display only.
        self._catalog_ctx_label = catalog_ctx_label or ""
        self._catalog_ctx = catalog_ctx
        self._populate_error(state)
        self._populate_gpus(state)
        self._populate_serving(state)
        self._populate_doctor(state)
        self._populate_scenes(state.scenes)
        # #11-ext — re-render the scene preview every poll (NOT only when the scene
        # list changes, which _populate_scenes guards): the per-service status
        # bullets track live container state, so they must refresh off each estate
        # read.  (The global Services strip was removed as a duplicate of this
        # scene-scoped view — the full running list lives in the Containers tab.)
        try:
            self.render_scene_preview(self.selected_scene())
        except Exception:
            pass

    def _populate_error(self, state: EstateState) -> None:
        """A2/N2: render the estate READ error (docker / nvidia-smi failure) as a
        distinct red strip — NOT the calm idle line.  Hidden when there's no
        error so a healthy rig shows no scary strip."""
        try:
            strip = self.query_one("#estate-error-strip", Static)
        except Exception:
            return
        err = (getattr(state, "error", "") or "").strip()
        if err:
            strip.update(f"[red]⚠ {err}[/red]")
            strip.add_class("visible")
        else:
            strip.update("")
            strip.remove_class("visible")

    def _populate_serving(self, state: EstateState) -> None:
        """#1 (Batch 1): surface WHAT'S SERVING — the captured serving target
        (matched slug + model + port).  When nothing is matched, say so plainly.

        This reads from the estate snapshot the pane is populated with (the same
        ``matched_slug`` / ``target`` the app captures into ``_target_*``); it
        does NOT fabricate — a non-LLM GPU user (ComfyUI / studio) is correctly
        NOT a served model and surfaces only as the container holding the VRAM."""
        line = self.query_one("#serving-line", Static)
        slug = (state.matched_slug or "").strip()
        tgt = state.target
        model = (getattr(tgt, "model", "") or "").strip()
        url = (getattr(tgt, "url", "") or "").strip()
        port = getattr(tgt, "host_port", 0) or 0
        if not (slug or model):
            line.update("[dim]○ no model serving[/dim]")
            return
        parts: list[str] = []
        conf = (getattr(tgt, "match_confidence", "") or "").strip()
        if model:
            parts.append(f"[green]{model}[/green]")
        if slug:
            if conf == "shape":
                # F9: port/substring match — the slug is the matched SHAPE, not
                # the verified identity of what's serving.  Say so.
                parts.append(f"[yellow]👤[/yellow] [dim]on {slug} shape[/dim]")
            else:
                parts.append(f"[dim]{slug}[/dim]")
        if port:
            # F3: the USABLE endpoint — full LAN URL + auth status, not a bare
            # port. Same derivation as switch.sh's ready-line (services.lan_ip).
            try:
                lan = self.app._data.lan_ip()
            except Exception:
                lan = "localhost"
            parts.append(f"http://{lan}:{port}/v1 [dim]· no auth · \\[u] copy[/dim]")
        elif url:
            parts.append(f"[dim]{url}[/dim]")
        head = "[green]▶[/green] Serving: " + "  ·  ".join(parts)
        # A7: render the ACTUAL probed running config (ctx + engine image), NOT
        # the catalog slug's claim — and BADGE a divergence when the probed ctx
        # differs from the matched slug's claimed ctx.  A field the probe did not
        # return falls back to the catalog claim, clearly labelled "(per catalog
        # slug)"; we NEVER present a claim as a measured value.
        line.update(head + self._serving_config_suffix(state))

    def _serving_config_suffix(self, state: EstateState) -> str:
        """A7: the second line of the serving panel — the PROBED running config
        (ctx + image) with a divergence badge vs the catalog slug's claim.

        Honesty rules:
          - PROBED ctx is shown as a measured value; the catalog ctx is shown
            only as a fallback labelled "(per catalog slug)" when the probe gave
            nothing.
          - a divergence badge fires only when BOTH a probed ctx AND a catalog
            claim exist and they differ — never on a missing probe."""
        served = getattr(state, "served", None)
        probed_ctx = getattr(served, "max_model_len", None) if served else None
        image = (getattr(served, "image", "") or "").strip() if served else ""
        claim_label = (self._catalog_ctx_label or "").strip()
        # Prefer the NUMERIC catalog claim (fit.max_ctx) for the comparison; fall
        # back to parsing the display label only when no numeric claim was passed.
        claim_ctx = self._catalog_ctx
        if claim_ctx is None:
            claim_ctx = parse_ctx_label(claim_label)

        bits: list[str] = []
        # ── context ──────────────────────────────────────────────────────────
        if probed_ctx is not None:
            bits.append(f"ctx [cyan]{_ctx_label(probed_ctx)}[/cyan] [dim](running)[/dim]")
            # Divergence badge — only when there's a real NUMERIC claim to compare
            # to AND the probed running ctx genuinely differs from it (within a
            # small tolerance so a 1-token rounding artefact doesn't fire a false
            # badge).  Comparing the exact ints (not the colloquial labels) means
            # a slug labelled "262K" for a 262144 ctx does NOT trip divergence.
            if claim_ctx is not None and abs(int(claim_ctx) - int(probed_ctx)) > 1024:
                bits.append(
                    f"[yellow]⚠ config differs from catalog slug "
                    f"{state.matched_slug or '?'}[/yellow] "
                    f"[dim](slug claims {claim_label or _ctx_label(claim_ctx)})[/dim]"
                )
        elif claim_label:
            # No probe value — show the claim, clearly labelled as the claim.
            bits.append(f"ctx [dim]{claim_label} (per catalog slug)[/dim]")
        # ── engine image (ground truth — vllm.__version__ lags the tag) ────────
        if image:
            bits.append(f"image [dim]{image}[/dim]")
        if not bits:
            return ""
        return "\n   " + "  ·  ".join(bits)

    def populate_power_cap(self, st: PowerCapState) -> None:
        # #10(a): cache the active cap per GPU so the GPU cards can annotate it.
        # Only a card BELOW its default counts as capped.  (The dedicated power-cap
        # strip was removed — the cap shows on the GPU cards at the TOP of the pane,
        # so a bottom strip just duplicated it; the [c] menu still adjusts it.)
        cap_map: dict[int, float] = {}
        for g in st.gpus:
            capped = (
                g.limit_w is not None and g.default_w is not None and g.limit_w < g.default_w
            )
            if capped:
                cap_map[g.index] = g.limit_w  # type: ignore[assignment]
        self._gpu_cap = cap_map
        # Re-render the GPU cards now that the cap is known (power-cap is read
        # AFTER the estate poll, so the first card paint had no cap note).
        if getattr(self, "_last_state", None) is not None:
            try:
                self._populate_gpus(self._last_state)
            except Exception:
                pass

    def populate_telemetry(self, tel: EstateTelemetry) -> None:
        """#12 / N5 + attribution: re-render the GPU cards with the VRAM-owner line.

        FIX 3 — the host disk bars + system-RAM line MOVED to the left-rail
        ``HostStatsRail`` (the "estate column"); this pane only owns the GPU-card
        "held by:" attribution now.  The telemetry is read by ``load_estate``
        piggybacked on the Operate tick (no new timer).  Cached so a later GPU-card
        re-render (e.g. after the power-cap read) keeps the "held by:" attribution."""
        self._last_telemetry = tel
        # Re-render the GPU cards now that the attribution map is known (telemetry
        # is read AFTER the estate poll, so the first card paint had no owner line).
        if getattr(self, "_last_state", None) is not None:
            try:
                self._populate_gpus(self._last_state)
            except Exception:
                pass

    def refresh_gpu_cards(self, gpus) -> None:
        """Fast GPU-only repaint: swap a fresh docker-free nvidia-smi read into the cached
        estate state and re-render just the GPU cards (the App's _refresh_gpu_bars calls
        this every poll tick). No-op until the first full estate poll has seeded _last_state
        — and the 'held by: <container>' attribution stays on the slower estate batch."""
        st = getattr(self, "_last_state", None)
        if st is None or not gpus:
            return
        st.gpus = gpus
        try:
            self._populate_gpus(st)
        except Exception:
            pass

    def _populate_gpus(self, state: EstateState) -> None:
        # N2: when nvidia-smi returned NOTHING at all (no cards in the snapshot),
        # say so honestly on the first card rather than a calm "not present" per
        # slot — a totally-empty GPU read usually means nvidia-smi failed, not a
        # GPU-less rig.  A per-index gap (one card present, the other not) still
        # uses the calm "not present".
        no_gpus_at_all = not state.gpus
        for i, bar_id, title_id in ((0, "#gpu0-bar", "#gpu0-card"), (1, "#gpu1-bar", "#gpu1-card")):
            bar = self.query_one(bar_id, Static)
            gpu = next((g for g in state.gpus if getattr(g, "index", -1) == i), None)
            if gpu is None:
                if no_gpus_at_all and i == 0:
                    bar.update("[red]no GPUs — nvidia-smi returned nothing[/red]")
                else:
                    bar.update("[dim]not present[/dim]")
                continue
            used = getattr(gpu, "mem_used_mib", 0)
            total = getattr(gpu, "mem_total_mib", 0) or 1
            util = getattr(gpu, "utilization", 0)
            pwr = getattr(gpu, "power_draw_w", 0.0)
            pwr_lim = getattr(gpu, "power_limit_w", 0.0)
            temp = getattr(gpu, "temp_c", 0)
            pct = int(used / total * 100) if total else 0
            filled = max(0, min(20, round(pct / 5)))
            color = "green" if pct < 80 else "yellow" if pct < 95 else "red"
            bar_str = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (20 - filled)}[/dim]"
            # #10(a): show power draw + the cap on the card.
            cap_note = ""
            cap_w = self._gpu_cap.get(i) if getattr(self, "_gpu_cap", None) else None
            if cap_w is not None:
                cap_note = f" (cap {cap_w:.0f}W)"
            lines = [
                f"  {bar_str}  {used / 1024:.1f} / {total / 1024:.1f} GiB · {pct}%",
                f"  power: {pwr:.0f} / {pwr_lim:.0f} W{cap_note} · {temp}°C · util {util}%",
            ]
            # Batch 5 (GPU-VRAM → container attribution): WHO holds this card's
            # VRAM — closes the Batch-1 "GPU0's 22GB owner is invisible" loop.
            # Derived from the cached telemetry (nvidia-smi --query-compute-apps +
            # pid→cgroup→docker), NOT the estate poll.  Best-effort: a holder whose
            # container couldn't be resolved shows "pid <N>" (never a fabricated
            # owner); an empty card shows nothing extra.
            attrib = self._gpu_attribution_line(i)
            if attrib:
                lines.append(attrib)
            # Holders whose physical card couldn't be resolved (uuid→index read
            # skewed) bucket under the None key — render them on GPU0's card under
            # a NEUTRAL heading (NOT pinned to a specific card), never mis-pinned.
            if i == 0:
                unpinned = self._gpu_attribution_line(None)
                if unpinned:
                    lines.append(unpinned)
            bar.update("\n".join(lines))

    def _gpu_attribution_line(self, index: Optional[int]) -> str:
        """Batch 5: the "held by: <container> (<vram>)" line for GPU ``index``.

        ``index`` is a physical GPU index, or ``None`` for holders whose card the
        uuid→index read couldn't resolve — those render under a neutral "VRAM held
        (card unknown)" heading rather than being mis-pinned to GPU0.

        Reads the cached telemetry's per-card compute-apps.  Honest degradation:
        a holder whose pid→container map failed renders as ``pid <N>`` (no
        fabricated name), and if ANY holder on the card is nameless the line
        appends ``(names unavailable)`` so the user knows the *VRAM total* is
        honest even when a name is missing.  Returns '' when no holder is known
        (an idle card shows no spurious owner line)."""
        tel = getattr(self, "_last_telemetry", None)
        if tel is None:
            return ""
        apps = (tel.gpu_apps or {}).get(index, [])
        if not apps:
            return ""
        bits: list[str] = []
        any_nameless = False
        for app in apps:
            vram = f"{app.used_mib / 1024:.1f}G" if app.used_mib else "—"
            if app.container:
                bits.append(f"{app.container} ({vram})")
            else:
                bits.append(f"pid {app.pid} ({vram})")
                any_nameless = True
        heading = "held by:" if index is not None else "VRAM held (card unknown):"
        line = f"  [dim]{heading}[/dim] " + ", ".join(bits)
        if any_nameless:
            line += " [dim](names unavailable)[/dim]"
        return line

    def _populate_doctor(self, state: EstateState) -> None:
        dr = state.doctor
        line = self.query_one("#doctor-line", Static)
        if not dr.reachable:
            if dr.booting:
                line.update("[yellow]⏳[/yellow] API booting — model loading")
            else:
                line.update("[red]○[/red] API not reachable")
            return
        glyph = "[green]●[/green]" if dr.serving else "[yellow]○[/yellow]"
        line.update(f"{glyph} {dr.summary}")

    def _scene_rows_signature(self, scenes: list[Scene]) -> tuple:
        """FIX 1 — a cheap "did the data change?" signature for the scene set.
        When the next poll's scenes are byte-identical to what's already rendered
        we skip the whole clear/re-add (removes both flicker AND the cursor churn
        on the periodic Operate refresh)."""
        return tuple(
            (s.name, s.group, s.gpus or "—",
             ", ".join(s.services[:3]) + ("…" if len(s.services) > 3 else ""))
            for s in scenes
        )

    def _populate_scenes(self, scenes: list[Scene]) -> None:
        t = self.query_one("#scene-table", DataTable)
        new_sig = self._scene_rows_signature(scenes)
        # FIX 1 (skip-if-unchanged guard) — the B2 periodic refresh re-populates
        # every 4s; when nothing changed, leave the table (and its cursor) alone.
        if new_sig == getattr(self, "_scene_rows_sig", None):
            self._scenes = list(scenes)
            return
        # FIX 1 (cursor preserve) — capture the selected row's STABLE KEY (scene
        # name) BEFORE the clear, so a re-populate that changes the row set (a
        # scene appears/disappears) restores the cursor to the same scene, not a
        # bare index.  Do NOT call .focus() (must not steal focus from another
        # widget).  Guarded — a 0-row table must not raise.
        sel_name = ""
        old_idx = 0
        try:
            old_idx = max(0, t.cursor_row)
            sel = self.selected_scene()
            if sel is not None:
                sel_name = sel.name
        except Exception:
            pass
        self._scenes = list(scenes)
        self._scene_rows_sig = new_sig
        t.clear()
        for s in scenes:
            svc = ", ".join(s.services[:3]) + ("…" if len(s.services) > 3 else "")
            t.add_row(s.name, s.group, s.gpus or "—", svc or "—")
        # FIX 1 — restore the cursor by key: if the selected scene still exists,
        # move to its new index; if it's gone, clamp the OLD index; if the table
        # was at row 0 / unselected, leave it.  animate=False so no visible jump.
        try:
            if t.row_count and (sel_name or old_idx > 0):
                new_idx = next(
                    (i for i, s in enumerate(scenes) if s.name == sel_name), None
                )
                if new_idx is None:
                    new_idx = min(old_idx, t.row_count - 1)
                if new_idx > 0:
                    t.move_cursor(row=new_idx, animate=False)
        except Exception:
            pass
        # #11 — keep the preview in sync with the cursor after a (re-)populate.
        try:
            self.render_scene_preview(self.selected_scene())
        except Exception:
            pass

    def selected_scene(self) -> Optional[Scene]:
        t = self.query_one("#scene-table", DataTable)
        idx = t.cursor_row
        if 0 <= idx < len(self._scenes):
            return self._scenes[idx]
        return None

    def _scene_running_names(self) -> list[str]:
        """The running-container names from the LAST estate poll (sync, no I/O) —
        the match source for the scene-preview service bullets, so the highlight
        path never shells docker."""
        st = self._last_state
        if st is None:
            return []
        return [c.name for c in st.containers if getattr(c, "is_running", True)]

    def render_scene_preview(self, scene: Optional[Scene]) -> None:
        """#11/#11-ext — render the highlighted scene's preview: name · group ·
        GPUs · ACTIVE/inactive · description · its services each with a LIVE status
        bullet (● green = running · ○ red = stopped) · ports.  A pure LOCAL read off
        the Scene the gpu-mode poll carried + the running-container set from the last
        estate poll — no per-keystroke I/O.  Read-only (per-service start/stop lives
        in the Containers tab)."""
        try:
            body = self.query_one("#scene-preview", Static)
        except Exception:
            return
        if scene is None:
            body.update("[dim]highlight a scene (move cursor) to preview what it brings up[/dim]")
            self._scene_svc_states = []
            return
        from rich.markup import escape

        states = CockpitData.scene_service_states_sync(scene, self._scene_running_names())
        self._scene_svc_states = states
        active = any(s.running for s in states)
        tag = "[green]● ACTIVE[/green]" if active else "[dim]○ inactive[/dim]"
        lines = [
            f"  [bold]{escape(scene.name)}[/bold]"
            + (f"  [dim]·[/dim]  {escape(scene.group)}" if scene.group else "")
            + f"  [dim]·[/dim]  GPUs {escape(scene.gpus or '—')}"
            + f"  [dim]·[/dim]  {tag}",
        ]
        if scene.description:
            lines.append(f"  [dim]{escape(scene.description)}[/dim]")
        if states:
            # Show every service (running first so active ones lead). The preview box
            # auto-grows (no max-height) — a busy scene like ai-studio (~7 services) wraps
            # across rows without clipping; the pane scrolls if it ever needs to.
            ordered = sorted(states, key=lambda s: not s.running)
            svcs = "   ".join(
                (f"[green]●[/green] {escape(s.name)}" if s.running
                 else f"[red]○[/red] {escape(s.name)}")
                for s in ordered
            )
            lines.append(f"  [bold]services[/bold]  {svcs}")
        else:
            lines.append("  [bold]services[/bold]  [dim]— none (frees GPUs)[/dim]")
        if scene.ports:
            lines.append(f"  [bold]ports[/bold]  {', '.join(escape(p) for p in scene.ports)}")
        body.update("\n".join(lines))


# ── Operate · Containers ──────────────────────────────────────────────────────────


class OperateContainersPane(Container):
    """Operate / Containers tab: container list + drill-down area."""

    DEFAULT_CSS = """
    OperateContainersPane {
        height: 1fr;
    }
    OperateContainersPane #containers-heading {
        text-style: bold;
        padding: 0 1;
        margin: 0 1 0 1;
    }
    OperateContainersPane #containers-table {
        height: auto;
        margin: 0 1 0 1;
        max-height: 12;
    }
    OperateContainersPane #drill-tabs {
        height: 1fr;
        margin: 1 1 0 1;
        border: solid $primary;
    }
    OperateContainersPane #drill-logs {
        height: 1fr;
    }
    OperateContainersPane #drill-stats {
        padding: 1;
        color: $text;
    }
    OperateContainersPane #drill-config {
        padding: 1;
        color: $text-muted;
    }
    OperateContainersPane #containers-hint {
        padding: 0 1;
        margin: 0 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Containers", id="containers-heading")
        ct: DataTable = DataTable(
            id="containers-table", zebra_stripes=True, show_cursor=True
        )
        ct.cursor_type = "row"
        yield ct
        with TabbedContent(id="drill-tabs"):
            with TabPane("Logs", id="drill-tab-logs"):
                # F8 — pane-specific idle copy (the shared LivePane default once
                # leaked test-runner wording into this docker-logs drill).
                yield LivePane(
                    id="drill-logs",
                    placeholder="Select a running container — its docker logs stream here.",
                )
            with TabPane("Top", id="drill-tab-stats"):
                yield Static("[dim]highlight a container (move cursor) or press [t] — docker top loads[/dim]", id="drill-stats")
            with TabPane("Config", id="drill-tab-config"):
                yield Static("[dim]highlight a container (move cursor) to load its config[/dim]", id="drill-config")
        yield Label(
            "[dim]move cursor or \\[l]/\\[t] to load detail · \\[l] logs   \\[t] top   "
            "\\[s] restart · start if stopped (gated)   \\[x] stop (gated)   "
            "\\[X] rm (reconcile-gated)[/dim]",
            id="containers-hint",
        )

    def on_mount(self) -> None:
        t = self.query_one("#containers-table", DataTable)
        t.add_columns("Name", "Kind", "Engine", "Port", "Slug")
        self._containers: list[ContainerInfo] = []
        # FIX 1 — last-rendered container-row signature (skip-if-unchanged guard).
        self._container_rows_sig: Optional[tuple] = None
        # FIX 1 (clamp echo) — set by populate(): True when the prior selection was
        # CLAMPED to a DIFFERENT container (the selected one vanished), so the
        # caller can swallow the spurious move_cursor RowHighlighted echo.
        self.last_populate_clamped: bool = False

    def _container_rows_signature(
        self, containers: list[ContainerInfo], error: str
    ) -> tuple:
        """FIX 1 — a cheap "did the data change?" signature for the container set
        (incl. the error/empty sentinel rows) so an unchanged periodic poll skips
        the clear/re-add — removing flicker AND the cursor snap-to-row-0 churn."""
        if not containers:
            return ("__sentinel__", (error or "").strip())
        return tuple(
            (
                c.name,
                getattr(c, "kind", ""),
                getattr(c, "status", "running"),
                getattr(c, "engine", "") or "—",
                getattr(c, "host_port", 0) or 0,
                getattr(c, "slug", "") or "—",
            )
            for c in containers
        )

    def populate(self, containers: list[ContainerInfo], error: str = "") -> bool:
        """Re-render the container table.  Returns True when the table was actually
        cleared + rebuilt (the row set changed), False when an unchanged poll was
        skipped — the caller uses this to avoid re-arming the row-0 suppression on
        a no-op poll (which would otherwise drop a pending user drill).

        FIX 1 (clamp echo) — also sets ``self.last_populate_clamped`` to record
        whether the prior selection was PRESERVED (the same container name still
        present → the cursor moved to follow it) or CLAMPED to a DIFFERENT
        container (the selected container vanished → the cursor landed on a row
        the user never picked).  The caller reads this to suppress the spurious
        ``move_cursor`` RowHighlighted echo on the CLAMP case (which would
        otherwise auto-load a docker drill for a container the user didn't
        select).  Reset to False every call; only True on a CLAMP-to-different."""
        t = self.query_one("#containers-table", DataTable)
        self.last_populate_clamped = False
        new_sig = self._container_rows_signature(containers, error)
        # FIX 1 (skip-if-unchanged guard) — the B2 periodic refresh re-populates
        # every 4s; when the container set is byte-identical, leave the table (and
        # its cursor) untouched.  This also means the [r]-re-jump suppression in
        # load_estate is a no-op on an unchanged poll (the cursor never moved).
        if new_sig == getattr(self, "_container_rows_sig", None):
            self._containers = list(containers)
            return False
        # FIX 1 (cursor preserve) — capture the selected row's STABLE KEY (container
        # name) BEFORE the clear so a re-populate whose row set changed (a container
        # started/stopped) restores the cursor to the same container, not a bare
        # index.  Do NOT .focus() (must not steal focus).  Guarded — 0-row safe.
        sel_name = ""
        old_idx = 0
        try:
            old_idx = max(0, t.cursor_row)
            sel = self.selected_container()
            if sel is not None:
                sel_name = sel.name
        except Exception:
            pass
        self._containers = list(containers)
        self._container_rows_sig = new_sig
        t.clear()
        if not containers:
            # N2: a READ failure must NOT read as a calm empty estate.
            err = (error or "").strip()
            if err:
                # MUST-FIX 3: surface the ACTUAL error headline, not a hardcoded
                # "docker unreachable" — a detect failure (docker fine) or an
                # nvidia-smi failure would otherwise be mislabeled.
                t.add_row(f"[red]{_error_headline(err)}[/red]", "—", "—", "—", "—")
            else:
                t.add_row("[dim]no stack containers[/dim]", "—", "—", "—", "—")
            return True
        for c in containers:
            stopped = getattr(c, "status", "running") == "stopped"
            if stopped:
                # Known-but-not-running supporting service — greyed, ○ red status
                # bullet.  Engine (e.g. "web") is still shown so you can tell what
                # the down service is; the port is unknown while it's down.
                t.add_row(
                    f"[red]○[/red] [dim]{c.name}[/dim]",
                    f"[dim]{c.kind}[/dim]",
                    f"[dim]{c.engine or '—'}[/dim]",
                    "[dim]—[/dim]",
                    "[dim]stopped[/dim]",
                )
            else:
                # Running — ● green status bullet so a live service reads clearly
                # as running (the slug column carries the registry match / —).
                t.add_row(
                    f"[green]●[/green] {c.name}",
                    c.kind,
                    c.engine or "—",
                    str(c.host_port) if c.host_port else "—",
                    # F9: badge a port/substring (shape) match — the slug is a
                    # guess for this container, not a verified identity.
                    (f"👤 {c.slug}" if getattr(c, "match_confidence", "") == "shape" and c.slug else (c.slug or "—")),
                )
        # FIX 1 — restore the cursor by key: if the selected container still
        # exists, move to its new index; if it's gone, clamp the OLD index; if the
        # table was at row 0 / unselected, leave it.  animate=False so no visible
        # jump.  Guarded — 0-row safe.  When the selection was CLAMPED to a
        # DIFFERENT container (the original vanished), record it so the caller can
        # swallow the move_cursor echo (else a docker drill auto-loads for a
        # container the user never selected — the re-introduced [r]-re-jump
        # footgun, now firing on every periodic tick).
        try:
            if t.row_count and (sel_name or old_idx > 0):
                new_idx = next(
                    (i for i, c in enumerate(containers) if c.name == sel_name), None
                )
                preserved = new_idx is not None
                if new_idx is None:
                    new_idx = min(old_idx, t.row_count - 1)
                # CLAMP-to-different = the user HAD a selection (sel_name set) that
                # is now gone, and the cursor lands on a non-zero row that is NOT
                # that container.  (Row 0 / unselected keeps the existing behavior.)
                if (not preserved) and sel_name and new_idx > 0:
                    self.last_populate_clamped = True
                if new_idx > 0:
                    t.move_cursor(row=new_idx, animate=False)
        except Exception:
            pass
        return True

    def populate_top(self, top) -> None:
        """Render a ContainerTop into the Top drill tab (READ)."""
        body = self.query_one("#drill-stats", Static)
        if top.error:
            body.update(f"[red]docker top failed:[/red] {top.error}")
            return
        from rich.markup import escape

        lines = ["  " + "  ".join(escape(h) for h in top.header)]
        for row in top.rows[:30]:
            lines.append("  " + "  ".join(escape(c) for c in row))
        if not top.rows:
            lines.append("  [dim](no processes)[/dim]")
        body.update("\n".join(lines))

    def populate_config(self, con: Optional[ContainerInfo], variant) -> None:
        """Render the selected container's registry/compose info into Config
        (a local READ — uses the cached registry row matched to the container)."""
        body = self.query_one("#drill-config", Static)
        if con is None:
            body.update("[dim]select a container to read its config[/dim]")
            return
        lines = [
            f"  [bold]Container[/bold]  {con.name}",
            f"  [bold]Kind[/bold]       {con.kind}",
            f"  [bold]Port[/bold]       {con.host_port or '—'} → {con.internal_port or '—'}",
            f"  [bold]Engine[/bold]     {con.engine or '—'}",
            f"  [bold]Slug[/bold]       "
            + (
                f"👤 {con.slug} [dim](shape match — port/substring, not an exact container)[/dim]"
                if getattr(con, "match_confidence", "") == "shape" and con.slug
                else (con.slug or "[dim]unmatched[/dim]")
            ),
        ]
        if variant is not None:
            lines.append(f"  [bold]Compose[/bold]    [dim]{getattr(variant, 'compose_path', '') or '—'}[/dim]")
            if getattr(variant, "status", ""):
                lines.append(f"  [bold]Status[/bold]     {variant.status}")
        body.update("\n".join(lines))

    def selected_container(self) -> Optional[ContainerInfo]:
        t = self.query_one("#containers-table", DataTable)
        idx = t.cursor_row
        if 0 <= idx < len(self._containers):
            return self._containers[idx]
        return None


# ── Validate panes (Phase 4 — wired to the data layer) ───────────────────────────


# The §3.5 *tune* gotchas surfaced inline on the Run pane.  These are the
# "judge the numbers right" warnings the maintainer learned the slow way; they
# are advisory text, NOT data — shown so a launch isn't misread.
_TUNE_GOTCHAS = (
    "[bold]Reading the results — gotchas[/bold]\n"
    "  • [yellow]Cliffs[/yellow]: single-card long-ctx configs degrade at ~21-26K accumulated "
    "ctx (Cliff 2) — soak-continuous catches it, a one-shot bench won't.\n"
    "  • [yellow]NIAH ≠ allocation[/yellow]: a passing needle at depth D does not prove the KV "
    "pool fits D tokens of real traffic — verify-stress ladders the allocation.\n"
    "  • [yellow]Spec-dec[/yellow]: judge MTP/DFlash on the bench TPS [italic]delta[/italic] (on vs off), "
    "never the accept-rate alone — a high accept can still net-regress on this MoE.\n"
    "  • [yellow]A/B at matched power[/yellow]: the rig systemd-caps to 230W; compare two configs "
    "only at the SAME power cap, or a power artifact masquerades as a config win."
)


# The launchable ladder + extra tools, in display order.  Each row is
# (kind, label, blurb).  ``kind`` is the CockpitData.run_validation kind.
_RUN_LADDER: list[tuple[str, str, str]] = [
    ("verify-full", "verify-full", "functional smoke (8/8) — does it serve + work"),
    ("verify-stress", "verify-stress", "boundary matrix (7/7) — long-ctx + tool-prefill OOM"),
    ("bench", "bench", "canonical TPS bench (3 warm + 5 measured)"),
    ("quality-test", "quality-test", "behavioral 8-pack (--quick) — tool / instruct / struct"),
    ("soak-test", "soak-test", "stability (continuous) — catches Cliff 2b"),
    ("rebench-full", "rebench-full", "the 5-step orchestrator (bench→stress→quality→soak→aider)"),
]
_RUN_EXTRAS: list[tuple[str, str, str]] = [
    ("quality-baseline", "quality-baseline", "regression diff vs the curated baseline (#252)"),
    ("bench-agentic", "bench-agentic", "multi-turn prefill stress"),
    ("stream-toolcall-probe", "stream-toolcall-probe", "silent-streaming tool-call check"),
]


class ValidateRunPane(Container):
    """Validate / Run tab: launchable ladder steps + extra tools + a live
    output pane, with the §3.5 *tune* gotchas inline.

    Each step launches a heavy validation script via ``CockpitData`` —
    confirm-gated (these stress / hit a serving model).  In this phase the
    write runner is NEVER executed live; tests inject a fake.  Output streams
    into the core LivePane below.
    """

    DEFAULT_CSS = """
    ValidateRunPane {
        height: 1fr;
    }
    ValidateRunPane #run-heading {
        text-style: bold;
        padding: 0 1;
        margin: 0 1 0 1;
    }
    ValidateRunPane #run-ladder-table {
        height: auto;
        max-height: 14;
        margin: 0 1 1 1;
    }
    ValidateRunPane #run-step-preview {
        height: auto;
        max-height: 5;
        border: solid $primary;
        padding: 0 1;
        margin: 0 1 1 1;
        color: $text;
    }
    ValidateRunPane #run-gotchas {
        border: solid $warning;
        padding: 0 1;
        margin: 0 1 1 1;
        height: auto;
        color: $text-muted;
    }
    ValidateRunPane LivePane {
        height: 1fr;
        margin: 0 1;
    }
    ValidateRunPane #run-hint {
        padding: 0 1;
        margin: 0 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Run  [dim](⏎ launches the selected step — confirm-gated)[/dim]", id="run-heading")
        t: DataTable = DataTable(id="run-ladder-table", zebra_stripes=True, show_cursor=True)
        t.cursor_type = "row"
        yield t
        # N8 — a compact preview of the highlighted validation step: what it runs
        # + its blurb (not only on ⏎-launch).  A pure LOCAL read off the ladder
        # row — mirrors the catalog / scene / evidence highlight-preview pattern.
        yield Static(
            "[dim]highlight a step (move cursor) to preview what it runs[/dim]",
            id="run-step-preview",
        )
        yield Static(_TUNE_GOTCHAS, id="run-gotchas")
        yield LivePane(
            id="run-output",
            placeholder="Ready. Launch a validation run (⏎ on a step) — output streams here.",
        )
        yield Label(
            "[dim]\\[⏎] launch selected (heavy — confirm) · streams below[/dim]",
            id="run-hint",
        )

    # A9: outcome glyph vocabulary — reuses DoctorPane's step_glyph language so a
    # cleared gate reads the same everywhere.  ·(unrun) / ⟳(running) / ✓ / ✗ / ⚠.
    _OUTCOME_GLYPH: dict[str, str] = {
        "unrun": "[dim]·[/dim]",
        "running": "[cyan]⟳[/cyan]",
        "passed": "[green]✓[/green]",
        "failed": "[red]✗[/red]",
        "warn": "[yellow]⚠[/yellow]",
    }

    def on_mount(self) -> None:
        t = self.query_one("#run-ladder-table", DataTable)
        # A9: leading "last" column shows each step's last-run outcome glyph so the
        # producer can answer "have I cleared the gate?" without scrolling the log.
        t.add_columns("last", "step", "kind", "what it checks")
        # (kind) in cursor order — the selected row maps back to a run kind.
        self._kinds: list[str] = []
        # A9: per-kind last-run outcome, cached in the pane (decision input for
        # ⑤ Promote).  Defaults to "unrun" until a run reports an outcome.
        self._outcomes: dict[str, str] = {}
        self._rows: list[tuple[str, str, str]] = list(_RUN_LADDER) + [
            (k, l, b) for (k, l, b) in _RUN_EXTRAS
        ]
        self._render_ladder()

    def _render_ladder(self) -> None:
        t = self.query_one("#run-ladder-table", DataTable)
        # Preserve the cursor across the re-render (outcome updates don't move it).
        saved = t.cursor_row
        t.clear()
        self._kinds = []
        for kind, label, blurb in _RUN_LADDER:
            t.add_row(self._outcome_glyph(kind), f"[cyan]▷[/cyan] {label}", "ladder", blurb)
            self._kinds.append(kind)
        for kind, label, blurb in _RUN_EXTRAS:
            t.add_row(self._outcome_glyph(kind), f"[cyan]▷[/cyan] {label}", "extra", blurb)
            self._kinds.append(kind)
        if t.row_count:
            try:
                t.move_cursor(row=max(0, min(saved, t.row_count - 1)))
            except Exception:
                pass
        # N8 — keep the step preview in sync with the cursor after a (re-)render
        # (outcome updates re-render the ladder; the preview must reflect them).
        try:
            self.render_step_preview(self.selected_kind())
        except Exception:
            pass

    def _outcome_glyph(self, kind: str) -> str:
        return self._OUTCOME_GLYPH.get(self._outcomes.get(kind, "unrun"), self._OUTCOME_GLYPH["unrun"])

    def set_run_outcome(self, kind: str, status: str) -> None:
        """A9: record (and render) the last-run outcome for a ladder kind.

        ``status`` ∈ {unrun, running, passed, failed, warn}.  Cached per kind so
        the leading glyph survives across re-renders; the decision input for ⑤
        Promote ("have I cleared the gate?")."""
        if status not in self._OUTCOME_GLYPH:
            return
        self._outcomes[kind] = status
        try:
            self._render_ladder()
        except Exception:
            pass

    def selected_kind(self) -> Optional[str]:
        t = self.query_one("#run-ladder-table", DataTable)
        idx = t.cursor_row
        if 0 <= idx < len(self._kinds):
            return self._kinds[idx]
        return None

    def render_step_preview(self, kind: Optional[str]) -> None:
        """N8 — render the compact preview for the highlighted validation step:
        what it runs (label + ladder/extra classification) + its blurb + the
        last-run outcome.  A pure LOCAL read off the ladder rows — no I/O."""
        try:
            body = self.query_one("#run-step-preview", Static)
        except Exception:
            return
        if kind is None:
            body.update("[dim]highlight a step (move cursor) to preview what it runs[/dim]")
            return
        meta = {
            k: (l, b, "ladder") for (k, l, b) in _RUN_LADDER
        }
        meta.update({k: (l, b, "extra") for (k, l, b) in _RUN_EXTRAS})
        if kind not in meta:
            body.update("[dim]—[/dim]")
            return
        label, blurb, cls = meta[kind]
        outcome = self._outcomes.get(kind, "unrun")
        lines = [
            f"  [bold]{label}[/bold]  [dim]·[/dim]  {cls}"
            f"  [dim]·[/dim]  last: {self._OUTCOME_GLYPH.get(outcome, self._OUTCOME_GLYPH['unrun'])} {outcome}",
            f"  [dim]{blurb}[/dim]",
        ]
        body.update("\n".join(lines))


class DoctorPane(Container):
    """Operate / Doctor tab: "is the running model serving correctly?".

    Six cards.  The first three render a parsed verdict back INTO the pane:
      - **health.sh** — live reachable/serving line (updates from the Operate
        estate poll via ``populate``, and on the [y] re-run via ``populate_health``);
      - **verify** ([v]) — verify.sh sends a real test query to the serving model
        (reachable → Genesis patch → completion → tool-call, ~15s);
      - **verify-full** ([V]) — verify-full.sh, the ~1-2 min functional battery.

    The last three are descriptive LAUNCHER cards — same visual treatment, but
    their output lands elsewhere (not a Doctor-pane verdict):
      - **report** ([R]) — report.sh ~2s snapshot → paste-ready copy modal;
      - **report --full** ([F]) — report.sh --full ~43-min battery (confirm-gated,
        uses the serving GPUs) → streams into ③ Gate / artifact in results/;
      - **power-cap sweep** ([w]) — power-cap-sweep.sh (gated · mutates the cap +
        benches each) → finds the TPS-per-watt knee.

    diagnose-estate / diagnose-profile moved to the producer Bring & Validate
    lane (they triage a validation target, not the live consumer rig)."""

    DEFAULT_CSS = """
    DoctorPane {
        height: 1fr;
    }
    DoctorPane #doctor-scroll {
        height: 1fr;
        padding: 1 2;
    }
    DoctorPane #doctor-heading {
        text-style: bold;
        margin-bottom: 1;
    }
    DoctorPane .doctor-card {
        border: solid $primary;
        padding: 1 2;
        margin-bottom: 1;
        height: auto;
    }
    DoctorPane .doctor-card-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    DoctorPane #doctor-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="doctor-scroll"):
            yield Label(
                "Doctor  [dim]— is the running model serving correctly?[/dim]",
                id="doctor-heading",
            )
            with Container(classes="doctor-card", id="doctor-card-health"):
                yield Label("health.sh  [dim](live)[/dim]", classes="doctor-card-title")
                yield Static("[dim]reading health.sh…[/dim]", id="doctor-health-body")
            with Container(classes="doctor-card", id="doctor-card-verify"):
                yield Label(
                    "verify  [dim]— send a test query to the serving model[/dim]",
                    classes="doctor-card-title",
                )
                yield Static(
                    "[dim]press [cyan]v[/cyan] — verify.sh (~15s): reachable → Genesis "
                    "patch → completion → tool-call[/dim]",
                    id="doctor-verify-body",
                )
            with Container(classes="doctor-card", id="doctor-card-verifyfull"):
                yield Label(
                    "verify-full  [dim]— functional battery[/dim]",
                    classes="doctor-card-title",
                )
                yield Static(
                    "[dim]press [cyan]V[/cyan] — verify-full.sh (~1-2 min): 9 steps "
                    "(streaming / thinking / cascade / MTP)[/dim]",
                    id="doctor-verifyfull-body",
                )
            with Container(classes="doctor-card", id="doctor-card-report"):
                yield Label(
                    "report  [dim]— paste-ready rig/stack snapshot[/dim]",
                    classes="doctor-card-title",
                )
                yield Static(
                    "[dim]press [cyan]R[/cyan] — report.sh (~2s): redacted rig + "
                    "stack snapshot; opens a paste-ready copy modal (no network, "
                    "no GPU)[/dim]",
                    id="doctor-report-body",
                )
            with Container(classes="doctor-card", id="doctor-card-fullreport"):
                yield Label(
                    "report --full  [dim]— full validation battery[/dim]",
                    classes="doctor-card-title",
                )
                yield Static(
                    "[dim]press [cyan]F[/cyan] — report.sh --full "
                    "([yellow]~43 min · uses the serving GPUs · confirm-gated[/yellow]): "
                    "streams into the ③ Gate; the artifact lands in results/ "
                    "(viewable in Validate · Evidence)[/dim]",
                    id="doctor-fullreport-body",
                )
            with Container(classes="doctor-card", id="doctor-card-sweep"):
                yield Label(
                    "power-cap sweep  [dim]— benches at each power cap[/dim]",
                    classes="doctor-card-title",
                )
                yield Static(
                    "[dim]press [cyan]w[/cyan] — power-cap-sweep.sh "
                    "([yellow]gated · mutates the GPU cap + benches each[/yellow]): "
                    "finds the TPS-per-watt knee for this rig[/dim]",
                    id="doctor-sweep-body",
                )
            yield Label(
                "[dim][cyan]y[/cyan] re-run health   ·   [cyan]v[/cyan] verify   ·   "
                "[cyan]V[/cyan] verify-full   ·   [cyan]R[/cyan] report   ·   "
                "[cyan]F[/cyan] report --full   ·   [cyan]w[/cyan] power-cap sweep[/dim]",
                id="doctor-hint",
            )

    def populate(self, state: EstateState) -> None:
        """Live health line from the Operate estate poll (the cheap per-poll probe)."""
        self._render_health(state.doctor)

    def _render_health(self, dr) -> None:
        body = self.query_one("#doctor-health-body", Static)
        if not dr.reachable:
            if dr.booting:
                # A model container is up but hasn't bound its port yet — mid-boot,
                # NOT down. Soft message + a re-check pointer (no remediation needed).
                body.update(
                    "[yellow]⏳[/yellow]  API booting — the model is loading\n"
                    "   [dim]engines bind the port only AFTER loading weights "
                    "(~10s-3min, longest on vLLM). Give it a moment, then "
                    "[cyan]y[/cyan] to re-check.[/dim]"
                )
                return
            # N7: OFFER the obvious remediation, not just the symptom.  No write
            # here — a navigation pointer to the gated serve path.
            body.update(
                "[red]✗[/red]  API not reachable\n"
                "   [dim]→ fix: serve a model — Run · Catalog ([cyan]1[/cyan]), pick a "
                "variant, [cyan]⏎[/cyan] (reconcile-gated)[/dim]"
            )
            return
        glyph = "[green]✓[/green]" if dr.serving else "[yellow]○[/yellow]"
        line = f"{glyph}  {dr.summary}"
        if not dr.serving:
            # Reachable endpoint but nothing served — point at the serve path.
            line += (
                "\n   [dim]→ fix: serve a model — Run · Catalog ([cyan]1[/cyan]) "
                "[cyan]⏎[/cyan][/dim]"
            )
        body.update(line)

    def populate_health(self, dr) -> None:
        """Refresh the live health card from a dedicated health.sh read (the
        on-demand [y] re-run + the per-poll Operate refresh both land here)."""
        self._render_health(dr)

    # ── verify (the "is it serving correctly?" smoke) ─────────────────────────────

    def set_verify_running(self) -> None:
        self.query_one("#doctor-verify-body", Static).update(
            "[dim]running verify.sh — sending a test query to the model (~15s)…[/dim]"
        )

    def populate_verify(self, vs) -> None:
        body = self.query_one("#doctor-verify-body", Static)
        if vs.error:
            body.update(
                f"[red]✗[/red]  {vs.error}\n"
                "   [dim]→ fix: serve a model — Run · Catalog ([cyan]1[/cyan]) "
                "[cyan]⏎[/cyan] (reconcile-gated)[/dim]"
            )
            return
        glyph = "[green]✓[/green]" if vs.ok else "[red]✗[/red]"
        color = "green" if vs.ok else "red"
        verdict = "serving correctly" if vs.ok else "NOT serving correctly"
        lines = [f"  {glyph} [{color}]{verdict}[/{color}]  ({vs.passed}/{vs.total} checks)"]
        step_glyph = {"passed": "[green]✓[/green]", "failed": "[red]✗[/red]"}
        for c in vs.checks:
            lines.append(f"    {step_glyph.get(c.status, '·')} {c.name}")
            if c.status == "failed" and c.hint:
                lines.append(f"        [dim]→ {c.hint}[/dim]")
        if not vs.reachable:
            lines.append(
                "   [dim]→ fix: serve a model — Run · Catalog ([cyan]1[/cyan]) "
                "[cyan]⏎[/cyan][/dim]"
            )
        body.update("\n".join(lines))

    # ── verify-full (the ~1-2 min functional battery) ─────────────────────────────

    def set_verify_full_running(self) -> None:
        self.query_one("#doctor-verifyfull-body", Static).update(
            "[dim]running verify-full.sh — functional battery (~1-2 min)…[/dim]"
        )

    def populate_verify_full(self, vf) -> None:
        body = self.query_one("#doctor-verifyfull-body", Static)
        if vf.error:
            body.update(f"[red]✗[/red]  {vf.error}")
            return
        glyph = {"●": "[green]✓[/green]", "◐": "[yellow]◐[/yellow]", "○": "[red]✗[/red]"}.get(
            vf.verdict_glyph, "·"
        )
        color = "green" if vf.ok else "yellow"
        head = f"  {glyph} [{color}]{vf.passed}/{vf.total} checks passed[/{color}]"
        if not vf.ok and any(c.status == "failed" for c in vf.checks):
            head += "  [dim](step 4 tool-calling fails on the default compose — expected)[/dim]"
        lines = [head]
        step_glyph = {"passed": "[green]✓[/green]", "failed": "[red]✗[/red]"}
        for c in vf.checks:
            lines.append(f"    {step_glyph.get(c.status, '·')} {c.name}")
            if c.status == "failed" and c.hint:
                lines.append(f"        [dim]→ {c.hint}[/dim]")
        body.update("\n".join(lines))


def _age_label(secs: int) -> str:
    """Compact 'how long ago' label for the F10 live/stale evidence rows."""
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"


def _gate_ladder(et: EvidenceTag) -> str:
    """F10 — one-line gate ladder for a live/incomplete run.

    ``✓`` done (with duration) · ``▶`` the step running now · ``·`` not yet
    reached.  Steps rendered positionally from GATE_STEPS; a step that was
    skipped (--skip/--resume) simply never turns ✓ — the ▶ marker carries the
    truth of what is actually executing."""
    done = {s: secs for s, secs in et.steps_done}
    parts: list[str] = []
    for s in GATE_STEPS:
        if s in done:
            parts.append(f"[green]✓[/green]{s} [dim]{_age_label(int(done[s]))}[/dim]")
        elif s == et.live_step:
            parts.append(f"[yellow]▶{s}[/yellow]")
        else:
            parts.append(f"[dim]·{s}[/dim]")
    return "  ".join(parts)


class ValidateEvidencePane(Container):
    """Validate / Evidence tab: real ``results/rebench/<tag>/`` run list from
    ``evidence_list()``; ``⏎`` opens the paste-ready report (``evidence_report``)
    in a modal (reuses the history_view pattern), ``s`` stages the gated
    submit-to-localmaxxing for the selected tag (confirm modal; never auto).

    F10 — doubles as the live gate-run OBSERVER: a tag dir with no REPORT.md is
    a run in flight (badged ▶, ladder + live tail in the preview, 4s self
    refresh) or an aborted one (⚠ incomplete).  Renders script-owned artifacts
    only (timings.json + step logs) — works for CLI-launched runs, never
    executes anything."""

    DEFAULT_CSS = """
    ValidateEvidencePane {
        height: 1fr;
    }
    ValidateEvidencePane #evidence-heading {
        text-style: bold;
        padding: 0 1;
        margin: 0 1 0 1;
    }
    ValidateEvidencePane #evidence-status {
        height: 1;
        color: $text-muted;
        padding: 0 1;
        margin: 0 1;
    }
    ValidateEvidencePane #evidence-table {
        height: 1fr;
        margin: 0 1 0 1;
    }
    ValidateEvidencePane #evidence-preview {
        height: auto;
        max-height: 6;
        border: solid $primary;
        padding: 0 1;
        margin: 0 1;
        color: $text;
    }
    ValidateEvidencePane #evidence-hint {
        padding: 0 1;
        margin: 0 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Evidence", id="evidence-heading")
        yield Label("Loading run tags…", id="evidence-status")
        et: DataTable = DataTable(id="evidence-table", zebra_stripes=True, show_cursor=True)
        et.cursor_type = "row"
        yield et
        # N8 — a compact preview of the highlighted run tag (its numbers/summary
        # on highlight, not only on ⏎-open): the artifacts present + the scraped
        # TL;DR.  A pure LOCAL read off the EvidenceTag — mirrors the catalog /
        # scene highlight-preview pattern.  The full report stays behind ⏎.
        yield Static(
            "[dim]highlight a run tag (move cursor) to preview its artifacts + TL;DR[/dim]",
            id="evidence-preview",
        )
        yield Label(
            "[dim]\\[⏎] open report   \\[m] vs catalog bar   \\[s] submit to localmaxxing (gated · never auto)[/dim]",
            id="evidence-hint",
        )

    def on_mount(self) -> None:
        t = self.query_one("#evidence-table", DataTable)
        t.add_columns("tag", "date", "report", "internal", "soak", "TL;DR")
        self._tags: list[EvidenceTag] = []
        # F10 — live gate-run observer: while a run is in flight the list
        # re-reads itself so the ladder + tail stay fresh (a producer facing a
        # silent 3-hr gate otherwise assumes a hang).  Paused whenever nothing
        # is live — populate() resumes/pauses it from the data, so CLI-launched
        # runs picked up by any refresh start the ticking too.
        self._live_timer = self.set_interval(4.0, self._live_tick, pause=True)

    def _live_tick(self) -> None:
        """F10 — periodic re-read while a run is live (READ; observer only)."""
        try:
            self.app.load_evidence()  # type: ignore[attr-defined]
        except Exception:
            pass

    def populate(self, tags: list[EvidenceTag]) -> None:
        status = self.query_one("#evidence-status", Label)
        t = self.query_one("#evidence-table", DataTable)
        saved_row = t.cursor_row  # F10: periodic refresh must not yank the cursor
        t.clear()
        self._tags = list(tags)
        # F10 — tick only while a run is in flight.
        try:
            if any(et.live for et in tags):
                self._live_timer.resume()
            else:
                self._live_timer.pause()
        except Exception:
            pass
        if not tags:
            status.update("[dim]no runs under results/rebench/[/dim]")
            t.add_row("[dim]—[/dim]", "—", "—", "—", "—", "—")
            return
        for et in tags:
            yn = lambda b: "[green]✓[/green]" if b else "[dim]·[/dim]"
            tag_cell = et.tag
            tldr = (et.tldr[:48] + "…") if len(et.tldr) > 49 else (et.tldr or "—")
            if et.live:
                # F10 — a run in flight: badge the row + say what's running now.
                tag_cell = f"[green]▶[/green] {et.tag}"
                step = et.live_step or "starting"
                tldr = f"[green]running[/green] — {step} · {len(et.steps_done)}/{len(GATE_STEPS)} steps done"
            elif et.stale:
                tag_cell = f"[yellow]⚠[/yellow] {et.tag}"
                tldr = f"[yellow]incomplete[/yellow] [dim](no REPORT.md · quiet {_age_label(et.age_secs)})[/dim]"
            t.add_row(tag_cell, et.date or "—", yn(et.has_report), yn(et.has_internal), yn(et.has_soak), tldr)
        status.update(f"{len(tags)} run tag(s) under results/rebench/")
        try:
            if t.row_count and saved_row > 0:
                t.move_cursor(row=min(saved_row, t.row_count - 1), animate=False)
        except Exception:
            pass
        # N8 — keep the preview in sync with the cursor after a (re-)populate.
        try:
            self.render_preview(self.selected_tag())
        except Exception:
            pass

    def selected_tag(self) -> Optional[EvidenceTag]:
        t = self.query_one("#evidence-table", DataTable)
        idx = t.cursor_row
        if 0 <= idx < len(self._tags):
            return self._tags[idx]
        return None

    def render_preview(self, tag: Optional[EvidenceTag]) -> None:
        """N8 — render the compact preview for the highlighted run tag: which
        artifacts are present + the scraped TL;DR.  A pure LOCAL read off the
        EvidenceTag — no I/O (the full report generation stays behind ⏎)."""
        try:
            body = self.query_one("#evidence-preview", Static)
        except Exception:
            return
        if tag is None:
            body.update("[dim]highlight a run tag (move cursor) to preview its artifacts + TL;DR[/dim]")
            return
        from rich.markup import escape

        if tag.live:
            # F10 — a run in flight: the ladder + the active step's live tail.
            lines = [
                f"  [green]▶ RUNNING[/green]  [bold]{escape(tag.tag)}[/bold]"
                f"  [dim]· last write {_age_label(tag.age_secs)} ago[/dim]",
                f"  {_gate_ladder(tag)}",
            ]
            for tl in (tag.live_tail or "").splitlines()[-4:]:
                lines.append(f"  [dim]│[/dim] {escape(tl)}")
            if not tag.live_tail:
                lines.append("  [dim]│ (no step output yet)[/dim]")
            body.update("\n".join(lines))
            return
        if tag.stale:
            lines = [
                f"  [yellow]⚠ INCOMPLETE[/yellow]  [bold]{escape(tag.tag)}[/bold]"
                f"  [dim]· no REPORT.md · quiet {_age_label(tag.age_secs)}[/dim]",
                f"  {_gate_ladder(tag)}",
                "  [dim]aborted or orphaned — re-run with --resume to continue from its artifacts[/dim]",
            ]
            body.update("\n".join(lines))
            return
        yn = lambda b: "[green]✓[/green]" if b else "[dim]·[/dim]"
        lines = [
            f"  [bold]{escape(tag.tag)}[/bold]"
            + (f"  [dim]·[/dim]  {escape(tag.date)}" if tag.date else ""),
            f"  [bold]artifacts[/bold]  REPORT.md {yn(tag.has_report)}"
            f"   _internal.json {yn(tag.has_internal)}   soak {yn(tag.has_soak)}",
        ]
        tldr = (tag.tldr or "").strip()
        lines.append(
            f"  [bold]TL;DR[/bold]  {escape(tldr)}" if tldr else "  [bold]TL;DR[/bold]  [dim]—[/dim]"
        )
        body.update("\n".join(lines))


# ── Evidence report modal (reuses the history_view read pattern) ─────────────────


class EvidenceReportScreen(_CopyableModal, ModalScreen):
    """Paste-ready report overlay for one rebench tag (READ — reads results)."""

    DEFAULT_CSS = """
    EvidenceReportScreen {
        align: center middle;
    }
    EvidenceReportScreen > Vertical {
        width: 96;
        height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    EvidenceReportScreen .evidence-report-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    EvidenceReportScreen #evidence-report-scroll {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("Y", "app.copy_context", "Copy", show=True),
    ]

    def __init__(self, tag: str, **kwargs):
        super().__init__(**kwargs)
        self._tag = tag

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Report · {self._tag}", classes="evidence-report-title")
            with ScrollableContainer(id="evidence-report-scroll"):
                yield Static("Generating report (rebench-report.py — reads results)…", id="evidence-report-body")
            yield Label("[dim]Esc to close · Y to copy[/dim]")

    def on_mount(self) -> None:
        # Load the report once the modal is mounted (so set_report's query
        # resolves) — mirrors ConfirmActionScreen's reconcile-on-mount.
        self.app.run_evidence_report(self, self._tag)  # type: ignore[attr-defined]

    def set_report(self, report: EvidenceReport) -> None:
        body = self.query_one("#evidence-report-body", Static)
        if report.error and not report.body:
            body.update(f"[red]report unavailable:[/red] {report.error}")
            return
        # Render the markdown body verbatim (escape Rich markup so [..] in the
        # report text isn't parsed as a tag).
        from rich.markup import escape

        body.update(escape(report.body))
        self._copy_payload = report.body   # raw markdown (the [Y] copy target)

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ── Phase R / R3b-2 · ④ Measure-vs-curated-bar modal (design §3.3 ④) ──────────────


class MeasureVsBarScreen(ModalScreen):
    """The ④ Measure "vs catalog bar" view for a selected evidence tag (READ).

    Shows the producer's MEASURED numbers next to the curated catalog's published
    bar for the same class — measured-vs-bar side by side + deltas + the honest
    verdict + the protocol caveats.  READ-only: it loads on mount via a worker
    (mirrors EvidenceReportScreen) and renders the comparison.  NO GPU / network /
    write — the cockpit FLAGS the protocol, it does not fabricate "catalog-grade"."""

    DEFAULT_CSS = """
    MeasureVsBarScreen {
        align: center middle;
    }
    MeasureVsBarScreen > Vertical {
        width: 92;
        height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    MeasureVsBarScreen .vsbar-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    MeasureVsBarScreen #vsbar-scroll {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(self, tag: str, **kwargs):
        super().__init__(**kwargs)
        self._tag = tag

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Measure vs catalog bar · {self._tag}", classes="vsbar-title")
            with ScrollableContainer(id="vsbar-scroll"):
                yield Static("Comparing measured numbers to the curated bar (local read)…", id="vsbar-body")
            yield Label("[dim]Esc to close · this FLAGS the protocol — it is not a catalog-grade certification[/dim]")

    def on_mount(self) -> None:
        # Load once mounted (so set_result's query resolves) — mirrors
        # EvidenceReportScreen's load-on-mount.
        self.app.run_measure_vs_bar(self, self._tag)  # type: ignore[attr-defined]

    def set_result(self, vsbar: MeasureVsBar) -> None:
        body = self.query_one("#vsbar-body", Static)
        if vsbar.error:
            body.update(f"[red]unavailable:[/red] {vsbar.error}")
            return
        m = vsbar.measured
        bar = vsbar.bar
        lines: list[str] = []

        verdict_color = {
            "within tolerance of the bar": "green",
            "under the bar": "yellow",
            "insufficient data": "dim",
        }.get(vsbar.verdict, "dim")
        lines.append(f"  Verdict: [{verdict_color}]{vsbar.verdict}[/{verdict_color}]")
        if m.model:
            lines.append(f"  Model:   [bold]{m.model}[/bold]")
        lines.append("")

        # Side-by-side table.
        bar_src = vsbar.bar_source or "—"
        lines.append("  [bold]Metric        Measured        Catalog bar      Δ[/bold]")
        lines.append("  " + "─" * 56)
        m_tps = m.tps_label
        b_tps = bar.tps_label if bar else "—"

        def _d(v):
            if v is None:
                return "—"
            sign = "+" if v >= 0 else ""
            color = "green" if v >= 0 else "red"
            return f"[{color}]{sign}{v:.0f}[/{color}]"

        lines.append(
            f"  Narr TPS      {self._cell(m.narr_tps)}{self._cell(bar.narr_tps if bar else None)}{_d(vsbar.narr_tps_delta)}"
        )
        lines.append(
            f"  Code TPS      {self._cell(m.code_tps)}{self._cell(bar.code_tps if bar else None)}{_d(vsbar.code_tps_delta)}"
        )
        lines.append(
            f"  8-pack        {m.quality_label:<16}{(bar.quality_label if bar else '—'):<17}—"
        )
        lines.append("")
        # Surface WHICH bar was matched (engine + topology) so the comparison is
        # legible — and whether the run's engine actually drove the selection.
        bar_eng = (bar.engine if bar else "") or "—"
        bar_topo = (bar.topology if bar else "") or "—"
        match_note = "engine-matched" if vsbar.engine_resolved else "[yellow]engine NOT matched[/yellow]"
        lines.append(
            f"  Bar:          [dim]{bar_eng} · {bar_topo}[/dim] ({match_note})"
        )
        lines.append(f"  Bar source:   [dim]{bar_src}[/dim]   Measured from: [dim]{m.source or '—'}[/dim]")
        lines.append("")

        # Protocol caveats — the honesty section.
        lines.append("  [bold yellow]What the cockpit cannot verify (flags, not a grade):[/bold yellow]")
        if vsbar.protocol_caveats:
            for c in vsbar.protocol_caveats:
                lines.append(f"    [yellow]•[/yellow] {c}")
        else:
            lines.append("    [dim](none)[/dim]")
        body.update("\n".join(lines))

    @staticmethod
    def _cell(v: Optional[float]) -> str:
        return f"{v:.0f}".ljust(16) if v is not None else "—".ljust(16)

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ── Phase R / R2b · Share-back paste-ready report modal (design §3.2 / §9.4 R2) ───


class ShareBackReportScreen(_CopyableModal, ModalScreen):
    """Generic paste-ready report overlay for the consumer share-back affordances
    (rig report [R] / problem report [!]).  READ-only — it loads its body on
    mount via a worker (mirrors EvidenceReportScreen) and renders it verbatim for
    the user to copy.  NO ConfirmActionScreen, NO network: these are reads that
    gather LOCAL context; the user copies the text and posts it themselves.

    ``loader`` is an async callable returning ``{"report", "error"}`` (e.g.
    ``CockpitData.rig_report`` / ``problem_report``); the app's worker invokes it
    and pushes the result back via ``set_report``."""

    DEFAULT_CSS = """
    ShareBackReportScreen {
        align: center middle;
    }
    ShareBackReportScreen > Vertical {
        width: 96;
        height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    ShareBackReportScreen .share-report-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    ShareBackReportScreen #share-report-scroll {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("Y", "app.copy_context", "Copy", show=True),
    ]

    def __init__(self, title: str, kind: str, **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._kind = kind  # "rig" | "problem" — selects the loader in the app

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, classes="share-report-title")
            with ScrollableContainer(id="share-report-scroll"):
                yield Static("Generating report (local read — no network)…", id="share-report-body")
            yield Label("[dim]Esc to close · [cyan]Y[/cyan] to copy the report for sharing[/dim]")

    def on_mount(self) -> None:
        # Load once the modal is mounted (so set_report's query resolves) —
        # mirrors EvidenceReportScreen's load-on-mount.
        self.app.run_share_back_report(self, self._kind)  # type: ignore[attr-defined]

    def set_report(self, report: str, error: Optional[str]) -> None:
        body = self.query_one("#share-report-body", Static)
        if error and not report:
            body.update(f"[red]report unavailable:[/red] {error}")
            return
        from rich.markup import escape

        body.update(escape(report))
        self._copy_payload = report   # raw, paste-ready (the [Y] copy target)

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ── Settings modal (Download UX · MODEL_DIR + HF_TOKEN) ───────────────────────────


class SettingsScreen(ModalScreen):
    """Edit the download settings — the MODEL DIR (weights live under
    ``<dir>/huggingface/``) and the HF TOKEN (gated/private repos).  Persisted to
    ``c3-settings.json`` and applied LIVE (re-stats the catalog against the new
    dir).  HF_HOME is auto-derived under the model dir — not a user field."""

    DEFAULT_CSS = """
    SettingsScreen {
        align: center middle;
    }
    SettingsScreen > Vertical {
        width: 84;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    SettingsScreen .settings-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    SettingsScreen .settings-field {
        margin-top: 1;
    }
    SettingsScreen Input {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=True, priority=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, model_dir: str, hf_token_set: bool, director_device: str = "gpu0", **kwargs):
        super().__init__(**kwargs)
        self._model_dir = model_dir or ""
        self._hf_token_set = hf_token_set
        self._director_device = director_device if director_device in ("gpu0", "gpu1", "cpu") else "gpu0"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Settings", classes="settings-title")
            yield Label("Model dir  [dim](weights live under <dir>/huggingface/)[/dim]",
                        classes="settings-field")
            yield Input(value=self._model_dir, placeholder="/mnt/models/huggingface", id="set-model-dir")
            tok_ph = ("hf_…  (leave blank to keep the current token)"
                      if self._hf_token_set else "hf_…  (for gated / private repos)")
            yield Label("HuggingFace token", classes="settings-field")
            yield Input(value="", password=True, placeholder=tok_ph, id="set-hf-token")
            yield Label("Director placement  [dim](ai-studio prompt-crafter · :8090)[/dim]",
                        classes="settings-field")
            yield Select(
                [("GPU 0 — fast craft, ~4.6 GB (default)", "gpu0"),
                 ("GPU 1 — only when free, not during video", "gpu1"),
                 ("CPU — frees GPU0 for long video, slow craft", "cpu")],
                value=self._director_device, allow_blank=False, id="set-director-device",
            )
            yield Label(
                "[dim]Ctrl+S save · Esc cancel · HF_HOME auto-derived under the model dir · "
                "director change applies on next ai-studio start[/dim]",
                classes="settings-field",
            )
            yield Footer()

    def action_save(self) -> None:
        mdir = self.query_one("#set-model-dir", Input).value.strip()
        tok = self.query_one("#set-hf-token", Input).value.strip()
        dev = str(self.query_one("#set-director-device", Select).value)
        self.app.pop_screen()
        self.app.apply_settings(model_dir=mdir, hf_token=tok, director_device=dev)  # type: ignore[attr-defined]

    def action_cancel(self) -> None:
        self.app.pop_screen()


class PowerCapMenuScreen(ModalScreen):
    """[c] power-cap menu (Orchestration): apply the default 230W cap · clear the
    cap (uncap to hardware default) · set a custom wattage.  The choice is returned
    via ``dismiss`` and routed through the standard confirm gate by the app — this
    modal never touches the rig itself.  dismiss value: ``("on"|"off", None)`` or
    ``("custom", <watts:int>)``; ``None`` on cancel."""

    DEFAULT_CSS = """
    PowerCapMenuScreen {
        align: center middle;
    }
    PowerCapMenuScreen > Vertical {
        width: 80;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    PowerCapMenuScreen .pc-title {
        text-style: bold;
        color: $accent;
    }
    PowerCapMenuScreen .pc-row { margin-top: 1; }
    PowerCapMenuScreen Input { margin-top: 1; }
    """

    # Don't auto-focus the wattage Input — otherwise it would swallow the d/u/m
    # shortcut keys.  Nothing is focused on open, so d/u/m reach the screen
    # bindings; [m] (or clicking) focuses the Input for custom-wattage entry.
    AUTO_FOCUS = None

    BINDINGS = [
        # priority so the menu keys fire on the screen even when nothing is
        # focused; check_action below frees d/u/m to be typed once the Input is.
        Binding("d", "apply_default", "Default 230W", show=True, priority=True),
        Binding("u", "clear_cap", "Clear", show=True, priority=True),
        Binding("m", "focus_custom", "Custom", show=True, priority=True),
        Binding("enter", "apply_custom", "Apply custom", show=True, priority=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, status_summary: str = "", **kwargs):
        super().__init__(**kwargs)
        self._status_summary = status_summary or ""

    def on_mount(self) -> None:
        # A focused Input swallows printable keys (d/u/m) before any binding sees
        # them, so blur it on open — the menu shortcuts then reach the screen, and
        # [m] focuses the Input when the user actually wants a custom wattage.
        self.call_after_refresh(self.set_focus, None)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Power cap", classes="pc-title")
            yield Static(f"  [dim]{self._status_summary}[/dim]", classes="pc-row")
            yield Label(
                "  [cyan]d[/cyan] apply default cap (230W)    "
                "[cyan]u[/cyan] clear cap (uncap to hw default)",
                classes="pc-row",
            )
            yield Label("  [cyan]m[/cyan] custom — focus the field, type a wattage, then [cyan]Enter[/cyan]:",
                        classes="pc-row")
            yield Input(placeholder="watts, e.g. 280", id="pc-watts")
            yield Label(
                "  [dim]each choice is confirmed before it touches the rig · Esc cancel[/dim]",
                classes="pc-row",
            )
            yield Footer()

    def action_apply_default(self) -> None:
        self.dismiss(("on", None))

    def action_clear_cap(self) -> None:
        self.dismiss(("off", None))

    def action_focus_custom(self) -> None:
        try:
            self.query_one("#pc-watts", Input).focus()
        except Exception:
            pass

    def action_apply_custom(self) -> None:
        raw = self.query_one("#pc-watts", Input).value.strip()
        if raw.isdigit() and int(raw) > 0:
            self.dismiss(("custom", int(raw)))
        else:
            self.app.notify("Enter a positive integer wattage (e.g. 280).",
                            title="Power cap", severity="warning", timeout=4)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Phase 5 · Promote-to-catalog scaffold preview modal (design §3.5b) ────────────


class StudioSetupScreen(ModalScreen):
    """ai-studio first-install setup — lists the MISSING studio models (grouped by
    modality, with sizes + total) and offers [d] "download all missing", launched
    DETACHED with polled progress (close it and the download keeps going).  Models
    only; if the comfyui-local IMAGE isn't built it also points at setup-*.sh (which
    builds it).  Reached from ⏎ on the ai-studio scene when it isn't set up."""

    BINDINGS = [
        Binding("d", "download", "Download all missing", show=True, priority=True),
        Binding("escape", "dismiss", "Close", show=True),
    ]

    DEFAULT_CSS = """
    StudioSetupScreen { align: center middle; }
    StudioSetupScreen #studio-setup-card {
        width: 86; max-width: 90%; height: auto; max-height: 80%;
        border: round $primary; background: $surface; padding: 1 2;
    }
    StudioSetupScreen #studio-setup-title { text-style: bold; color: $accent; margin-bottom: 1; }
    StudioSetupScreen #studio-setup-progress { margin-top: 1; }
    StudioSetupScreen #studio-setup-foot { margin-top: 1; color: $text-muted; }
    """

    _MOD_ORDER = ("director", "image", "video", "audio")
    _MOD_LABEL = {"director": "Director", "image": "🖼️ Image", "video": "🎬 Video", "audio": "🔊 Audio"}

    def __init__(self, missing, image_built: bool, data) -> None:
        super().__init__()
        self._missing = list(missing)
        self._image_built = image_built
        self._data = data
        self._downloading = False
        self._poll = None

    def compose(self) -> ComposeResult:
        total = sum(m.size_gb for m in self._missing)
        with Container(id="studio-setup-card"):
            yield Label(
                f"AI-Studio setup — {len(self._missing)} model(s) missing (~{total:.0f} GB)",
                id="studio-setup-title",
            )
            yield Static(self._body_text(), id="studio-setup-body")
            yield Static("", id="studio-setup-progress")
            yield Label(
                "[cyan]d[/cyan] download all missing   ·   [cyan]escape[/cyan] close",
                id="studio-setup-foot",
            )

    def _body_text(self) -> str:
        lines: list[str] = []
        if not self._image_built:
            lines.append(
                "[yellow]⚠ ComfyUI image not built — run [b]bash scripts/setup-image-studio.sh[/b] "
                "(builds it, then downloads).[/yellow]\n"
            )
        if not self._missing:
            lines.append("[green]✓ all ai-studio models present.[/green]")
            return "\n".join(lines)
        for mod in self._MOD_ORDER:
            ms = [m for m in self._missing if m.modality == mod]
            if not ms:
                continue
            sub = sum(m.size_gb for m in ms)
            lines.append(f"[b]{self._MOD_LABEL.get(mod, mod)}[/b]  [dim](~{sub:.0f} GB)[/dim]")
            for m in ms:
                lines.append(f"   ○ {m.label}  [dim](~{m.size_gb:.1f} GB)[/dim]")
        return "\n".join(lines)

    def action_download(self) -> None:
        if self._downloading or not self._missing:
            return
        self._downloading = True
        self.app.run_studio_download_worker()
        self.query_one("#studio-setup-progress", Static).update(
            "[yellow]⏳ downloading… (runs in the background — you can close this)[/yellow]"
        )
        self._poll = self.set_interval(2.0, self._refresh)

    def _refresh(self) -> None:
        try:
            self._missing = self._data.studio_missing_models()
            pct = self._data.studio_models_progress()
        except Exception:
            return
        prog = self.query_one("#studio-setup-progress", Static)
        if not self._missing:
            prog.update("[green]✓ all ai-studio models present — ready to launch ai-studio.[/green]")
            if self._poll is not None:
                self._poll.stop()
                self._poll = None
        else:
            prog.update(
                f"[yellow]⏳ downloading… {pct}%  ({len(self._missing)} model(s) remaining)[/yellow]"
            )
        try:
            self.query_one("#studio-setup-body", Static).update(self._body_text())
        except Exception:
            pass

    def action_dismiss(self) -> None:
        if self._poll is not None:
            self._poll.stop()
            self._poll = None
        self.dismiss()


class PromoteScaffoldScreen(ModalScreen):
    """Preview the computed catalog-promotion scaffold (SCAFFOLD + GATE).

    Shows the ModelProfile YAML skeleton + the compose_registry _entry(...) row
    COMPUTED from the BYO arch facts + Evidence numbers, plus the guard suite the
    gated write would run.  ``⏎`` stages the GATED write+guard ActionPlan — which
    is MOCK-ONLY this phase (it writes into scripts/ + runs the guard suite, so it
    NEVER auto-fires / executes live).  ``Esc`` just closes the preview."""

    DEFAULT_CSS = """
    PromoteScaffoldScreen {
        align: center middle;
    }
    PromoteScaffoldScreen > Vertical {
        width: 100;
        max-width: 100%;
        height: 84%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    PromoteScaffoldScreen .promote-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    PromoteScaffoldScreen #promote-scroll {
        height: 1fr;
    }
    PromoteScaffoldScreen #promote-btn-row {
        height: 3;
        margin-top: 1;
    }
    PromoteScaffoldScreen Button {
        margin-right: 1;
    }
    """

    # FIX 3 — the Enter verb is a DECLARED binding (was a raw on_key("enter")), so
    # Help / binding-introspection advertise it.  NOTE this modal renders no Footer
    # of its own — the visible affordance is the "Stage write" button label + body
    # text; show=True just keeps the binding in active_bindings (Help parity).
    # ``stage_write`` is gated in check_action on ``self._scaffold.computed``
    # (mirroring the disabled stage button), so the key is inert when staging isn't
    # possible.  priority=True so it wins over the focused button's own enter→press
    # (both route through the SAME _stage_write path).
    BINDINGS = [
        Binding("enter", "stage_write", "Stage write", show=True, priority=True),
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(self, scaffold: PromoteScaffold, *, on_stage_write=None, **kwargs):
        super().__init__(**kwargs)
        self._scaffold = scaffold
        self._on_stage_write = on_stage_write

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(
                f"Promote to catalog · {self._scaffold.model_id or self._scaffold.repo or '—'}",
                classes="promote-title",
            )
            with ScrollableContainer(id="promote-scroll"):
                yield Static(self._body_text(), id="promote-body")
            with Horizontal(id="promote-btn-row"):
                yield Button(
                    "⏎ Stage write (gated · mock-only)",
                    id="promote-stage-btn",
                    variant="warning",
                    disabled=not self._scaffold.computed,
                )
                yield Button("Esc Close", id="promote-close-btn")

    def _body_text(self) -> str:
        from rich.markup import escape

        s = self._scaffold
        if s.error:
            return f"[red]cannot scaffold:[/red] {escape(s.error)}"
        lines: list[str] = []
        lines.append("[dim]Design §3.5b — a SCAFFOLD + GATE, not a YAML IDE.  COMPUTED from the[/dim]")
        lines.append("[dim]BYO pull-gate arch facts + the Evidence measured numbers.  Compute +[/dim]")
        lines.append("[dim]preview ONLY — the write into scripts/ + guard run is gated & mock-only.[/dim]")
        lines.append("")
        lines.append(f"  [bold]ModelProfile[/bold]  [cyan]{escape(s.profile_path)}[/cyan]")
        lines.append("")
        for ln in s.profile_yaml.splitlines():
            lines.append("    " + escape(ln))
        lines.append("")
        lines.append("  [bold]compose_registry.py[/bold]  entry "
                     f"[green]{escape(s.registry_slug)}[/green]")
        lines.append("")
        for ln in s.registry_entry.splitlines():
            lines.append("    " + escape(ln))
        lines.append("")
        lines.append("  [bold]Guard suite[/bold] (the gated write would run, never auto):")
        lines.append("    [yellow]" + escape(" ".join(s.guard_suite_cmd)) + "[/yellow]")
        if s.notes:
            lines.append("")
            lines.append("  [bold]Notes[/bold]")
            for n in s.notes:
                lines.append(f"    • [dim]{escape(n)}[/dim]")
        lines.append("")
        lines.append("  [dim]⏎ Stage the gated write+guard (MOCK-ONLY this phase) · Esc Close[/dim]")
        return "\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "promote-stage-btn":
            self._stage_write()
        elif event.button.id == "promote-close-btn":
            self.action_dismiss()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Gate the ⏎ Stage-write binding on whether the scaffold computed — so the
        footer only advertises it when staging is actually possible (mirrors the
        disabled state of #promote-stage-btn)."""
        if action == "stage_write":
            return bool(self._scaffold.computed)
        return True

    def action_stage_write(self) -> None:
        """⏎ — stage the gated write+guard plan (only reachable when the scaffold
        computed; check_action gates the binding otherwise).  Same path the
        #promote-stage-btn press takes."""
        if self._scaffold.computed:
            self._stage_write()

    def _stage_write(self) -> None:
        """Hand the GATED write+guard plan back to the app's confirm gate.  The
        write is NEVER executed live this phase — it routes through the standard
        ConfirmActionScreen (mock-only) and never auto-fires."""
        self.app.pop_screen()
        if self._on_stage_write is not None and self._scaffold.write_plan is not None:
            self._on_stage_write(self._scaffold.write_plan)

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ── Phase 5 · Optimize-for-my-card seam modal (DORMANT v0.10.0 — design §5.2) ──────


class OptimizeScreen(ModalScreen):
    """The ▸ Optimize-for-my-card seam — DORMANT until the v0.10.0 optimizer lands.

    On open it invokes the seam, which detects the optimizer's absence and shows
    'optimizer not available (v0.10.0)'.  The honesty-gate rendering (boot-fit
    predicted|measured · runtime soak-validated · confidence tier · cliff-class
    --accept-runtime-risk) is built into ``set_report`` but stays dormant — it
    renders ONLY once the engine reports ``available=True``.  Never fabricates
    optimizer output."""

    DEFAULT_CSS = """
    OptimizeScreen {
        align: center middle;
    }
    OptimizeScreen > Vertical {
        width: 80;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    OptimizeScreen .optimize-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    OptimizeScreen #optimize-body {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(self, slug: str = "", **kwargs):
        super().__init__(**kwargs)
        self._slug = slug

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(
                f"Optimize for my card{(' · ' + self._slug) if self._slug else ''}",
                classes="optimize-title",
            )
            yield Static("Querying the optimizer seam…", id="optimize-body")

    def on_mount(self) -> None:
        self.app.run_optimize_for_modal(self, self._slug)  # type: ignore[attr-defined]

    def set_report(self, report: OptimizerReport) -> None:
        body = self.query_one("#optimize-body", Static)
        if not report.available:
            # DORMANT seam — honest "not available", never a fabricated rec.
            body.update(
                f"  [yellow]{report.message}[/yellow]\n"
                "\n"
                "  [dim]The per-card optimizer (recommend --optimize /\n"
                "  generate_compose.py --optimize) lands in v0.10.0.  When it does,\n"
                "  this seam will show its honesty gates:[/dim]\n"
                "    [dim]· boot-fit  predicted | measured[/dim]\n"
                "    [dim]· runtime   soak-validated | unvalidated[/dim]\n"
                "    [dim]· confidence tier[/dim]\n"
                "    [dim]· cliff-class recs require --accept-runtime-risk[/dim]\n"
                "\n"
                "  [dim]Esc to close[/dim]"
            )
            return
        # Reserved — rendered only once the engine lands (dormant today).
        risk = (
            "  [red]cliff-class — requires --accept-runtime-risk[/red]\n"
            if report.accept_runtime_risk_required
            else ""
        )
        body.update(
            f"  [bold]Recommended[/bold]  [green]{report.recommended_slug or '—'}[/green]\n"
            f"  [bold]boot-fit[/bold]    {report.boot_fit or '—'}\n"
            f"  [bold]runtime[/bold]     {report.runtime or '—'}\n"
            f"  [bold]confidence[/bold]  {report.confidence or '—'}\n"
            + risk
            + "\n  [dim]Esc to close[/dim]"
        )

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ── Producer lane ② Serve — untested-compose preview modal (R3b-1) ─────────────────


class UntestedComposePreviewScreen(ModalScreen):
    """Preview a GENERATED compose VERBATIM, badged as an untested config
    reproduction of a CATALOG slug, then a confirm to serve it through the
    reconcile-gated path (producer lane ② Serve).

    ⚠️  HONESTY (R3b-1): the previewed compose is a verbatim, UNTESTED reproduction
    of the resolved CATALOG profile ``<slug>``'s compose — NOT the fit-checked
    brought model's weights.  generate-compose.sh has no --repo / weight-swap yet;
    that is a deferred follow-up.  The badge reads "untested config reproduction of
    <slug>", not "your brought model".

    Mission (generate-compose.sh locked decision #2): reproduce + flag, NEVER
    repair — the compose is shown EXACTLY as generated; we do NOT fit-adapt it.
    ``⏎`` hands the ``serve_generated`` ActionPlan to the app's reconcile gate
    (the SAME ConfirmActionScreen every serve uses); ``Esc`` closes the preview
    (and unlinks the temp compose, since it was NOT served)."""

    DEFAULT_CSS = """
    UntestedComposePreviewScreen {
        align: center middle;
    }
    UntestedComposePreviewScreen > Vertical {
        width: 100;
        max-width: 100%;
        height: 84%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    UntestedComposePreviewScreen .untested-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    UntestedComposePreviewScreen #untested-scroll {
        height: 1fr;
    }
    UntestedComposePreviewScreen #untested-btn-row {
        height: 3;
        margin-top: 1;
    }
    UntestedComposePreviewScreen Button {
        margin-right: 1;
    }
    """

    # FIX 3 — the Enter verb is a DECLARED binding (was a raw on_key("enter")), so
    # Help / binding-introspection advertise it.  NOTE this modal renders no Footer
    # of its own — the visible affordance is the "Serve" button label + body; show=True
    # just keeps the binding in active_bindings (Help parity).  priority=True so it
    # wins over the focused button's own enter→press (both route through the SAME
    # _serve path).
    BINDINGS = [
        Binding("enter", "serve_untested", "Serve untested", show=True, priority=True),
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(self, slug: str, compose_path: str, compose_yaml: str, *,
                 on_serve=None, **kwargs):
        super().__init__(**kwargs)
        self._slug = slug
        self._compose_path = compose_path
        self._compose_yaml = compose_yaml
        self._on_serve = on_serve

    def compose(self) -> ComposeResult:
        from rich.markup import escape

        with Vertical():
            yield Label(
                f"② Serve · [yellow]untested config reproduction of "
                f"{self._slug}[/yellow]",
                classes="untested-title",
            )
            with ScrollableContainer(id="untested-scroll"):
                header = (
                    f"[yellow]⚠ This is an UNTESTED reproduction of the catalog\n"
                    f"profile {escape(self._slug)}'s compose — NOT your brought\n"
                    "model's weights (the bring-your-own weight-swap is a deferred\n"
                    "follow-up).[/yellow]\n"
                    "[dim]Generated VERBATIM by generate-compose.sh — reproduce +\n"
                    "flag, NEVER repair.  This compose is shown exactly as emitted;\n"
                    "it is NOT fit-adapted.  Serving it claims the GPU → the confirm\n"
                    "below runs the reconcile gate like every serve.[/dim]\n"
                    f"\n[dim]path:[/dim] {escape(self._compose_path)}\n\n"
                )
                yield Static(header + escape(self._compose_yaml), id="untested-body")
            with Horizontal(id="untested-btn-row"):
                yield Button(
                    "⏎ Serve (untested · reconcile-gated)",
                    id="untested-serve-btn",
                    variant="warning",
                )
                yield Button("Esc Close", id="untested-close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "untested-serve-btn":
            self._serve()
        elif event.button.id == "untested-close-btn":
            self.action_dismiss()

    def action_serve_untested(self) -> None:
        """⏎ — hand the serve_generated plan to the app's reconcile gate.  Same
        path the #untested-serve-btn press takes (NEVER auto-fired / live)."""
        self._serve()

    def _serve(self) -> None:
        """Hand the serve_generated plan to the app's reconcile gate.  Routes
        through the standard ConfirmActionScreen — NEVER auto-fired, NEVER live.
        Does NOT unlink the temp compose: a served plan's `docker compose -f
        <path>` references it, so it must persist (it's gitignored)."""
        self.app.pop_screen()
        if self._on_serve is not None and self._compose_path:
            self._on_serve(self._compose_path)

    def _cleanup_temp(self) -> None:
        """Unlink the generated temp compose — DECLINE path only (Esc / Close).
        A serve keeps the file (see _serve)."""
        import os
        if self._compose_path:
            try:
                os.unlink(self._compose_path)
            except OSError:
                pass

    def action_dismiss(self) -> None:
        # Declined without serving → remove the stray c3-genc temp compose so it
        # doesn't accumulate on disk (git-pollution is already handled by
        # .gitignore; this is the disk-cleanup tail of the temp-file fix).
        self._cleanup_temp()
        self.app.pop_screen()


# ── Producer "Bring & Validate" lane stage panes (R3b-1) ──────────────────────────
#
# The producer lane (mode 1) presents the ADDING_MODELS stage machine as an
# ORDERED, numbered pipeline: ① Bring → ② Serve → ③ Gate → ④ Measure → ⑤ Promote.
# It reuses the existing TabbedContent pattern (lighter than a full wizard widget)
# with numbered tab labels so it reads as an ordered pipeline.  ① Bring REUSES the
# byo_check fit-check; ③ Gate is the existing ValidateRunPane ladder; ④ Measure is
# the existing ValidateEvidencePane; ⑤ Promote hosts the [P] promote action.


class LaneBringPane(Container):
    """① Bring — the producer lane's fit-check entry, and (since the 2-mode merge)
    the SINGLE bring-an-arbitrary-repo entry point in the app.

    REUSES ``byo_check`` (pull.sh --dry-run --json → ByoResult: supported? fits?
    the swap_path route) as the lane's first stage: paste an HF repo / slug,
    Fit-check, read the route + sibling_slug + quant_match.  (The standalone
    Run · Bring-your-own tab + its ByoPane were removed in the merge — this pane's
    widget IDs are the only fit-check widgets now.)  The cached ``_last_byo`` it
    produces feeds ② Serve and ⑤ Promote."""

    DEFAULT_CSS = """
    LaneBringPane {
        height: 1fr;
        padding: 1 2;
    }
    LaneBringPane #lane-bring-heading {
        text-style: bold;
        margin-bottom: 1;
    }
    LaneBringPane #lane-bring-input-row {
        height: 3;
        margin-bottom: 1;
    }
    LaneBringPane #lane-bring-url-input {
        width: 1fr;
    }
    LaneBringPane #lane-bring-profile-input {
        width: 40;
        margin-left: 1;
    }
    LaneBringPane #lane-bring-profile-custom {
        width: 40;
        margin-left: 1;
    }
    LaneBringPane .profile-custom-hidden {
        display: none;
    }
    LaneBringPane #lane-bring-fit-btn {
        width: 14;
        margin-left: 1;
    }
    LaneBringPane #lane-bring-result-card {
        border: solid $primary;
        padding: 1 2;
        margin-top: 1;
        height: auto;
    }
    LaneBringPane #lane-bring-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("① Bring — fit-check an HF model", id="lane-bring-heading")
        with Horizontal(id="lane-bring-input-row"):
            yield Input(
                placeholder="org/Model  (e.g. unsloth/Qwen3-27B-abliterated-GGUF)",
                id="lane-bring-url-input",
            )
            # #6/A12 — same registry-derived (engine, topology) template Select as
            # Run · BYO (populated by set_profile_options after the catalog loads).
            yield Select(
                [("vllm/dual  ·  loading templates…", "vllm/dual")],
                value="vllm/dual",
                allow_blank=False,
                id="lane-bring-profile-input",
            )
            # FIX 2 (escape hatch) — companion free-text override, hidden until the
            # "✎ custom slug…" sentinel is chosen (same idiom as Run · BYO).
            yield Input(
                placeholder="profile-like slug — e.g. ik-llama/iq4ks-mtp",
                id="lane-bring-profile-custom",
                classes="profile-custom-hidden",
            )
            yield Button("Fit-check", id="lane-bring-fit-btn", variant="primary")
        yield Static(
            "[dim]Stage ① of the Bring & Validate pipeline.  Enter an HF repo + a\n"
            "profile-like slug, then Fit-check — pull.sh --dry-run (Path B, never\n"
            "downloads).  A successful fit-check unlocks ② Serve (generate + serve\n"
            "the untested compose) and ⑤ Promote.[/dim]",
            id="lane-bring-result-card",
        )
        yield Label(
            "[dim]Routes:  A = new curated profile   ·   B = serve-locally   ·   "
            "C = reuse a sibling compose + swap weights\n"
            "next: \\[2/]] ② Serve   ·   ③ Gate   ·   ④ Measure   ·   "
            "\\[P] ⑤ Promote[/dim]",
            id="lane-bring-hint",
        )

    def set_checking(self, repo: str) -> None:
        self.query_one("#lane-bring-result-card", Static).update(
            f"[dim]Checking[/dim] [cyan]{repo}[/cyan] [dim](pull.sh --dry-run --json)…[/dim]"
        )

    def set_profile_options(
        self, options: list[tuple[str, str]], default: Optional[str]
    ) -> None:
        """#6/A12 — fill the ① Bring profile-template Select (same registry-derived
        templates + rig-topology default as Run · BYO)."""
        _set_select_options(
            self.query_one("#lane-bring-profile-input", Select), options, default
        )

    def populate(self, res: ByoResult) -> None:
        card = self.query_one("#lane-bring-result-card", Static)
        card.update(_byo_result_text(res))


def _byo_result_text(res: ByoResult) -> str:
    """Render a ByoResult into the verdict card text (shared by Run · BYO + the
    producer lane's ① Bring stage)."""
    if res.error:
        return f"[red]Fit-check failed:[/red] {res.error}"
    lines: list[str] = []
    elig = "[green]eligible[/green]" if res.eligible else "[red]not eligible[/red]"
    lines.append(f"  [bold]{res.repo}[/bold]   {elig}")
    lines.append(f"  [bold]arch[/bold]     [cyan]{res.arch or '—'}[/cyan]")
    fitc = {
        "fits-clean": "[green]● fits-clean[/green]",
        "fits-constrained": "[yellow]◐ fits-constrained[/yellow]",
        "wont-fit": "[red]○ won't-fit[/red]",
    }.get(res.fit_verdict, res.fit_verdict or "—")
    lines.append(f"  [bold]fit[/bold]      {fitc}")
    if res.route:
        route_label = {
            "A": "Route A — author a new curated profile",
            "B": "Route B — serve locally (no catalog entry)",
            "C": "Route C — reuse a sibling compose + swap weights",
        }.get(str(res.route).upper(), f"Route {res.route}")
        lines.append("")
        lines.append(f"  [bold]{route_label}[/bold]")
        if res.sibling_slug:
            lines.append(f"    • reuse compose for [green]{res.sibling_slug}[/green]")
        if res.quant_match:
            lines.append(f"    • match [yellow]--quantization[/yellow] → {res.quant_match}")
        if res.drop_spec_config:
            lines.append("    • drop [yellow]--speculative-config[/yellow] (no MTP head in fine-tune)")
    if res.note:
        lines.append("")
        lines.append(f"  [dim]{res.note}[/dim]")
    # N9 — point the producer forward: a successful fit-check that resolved a
    # servable catalog target hands straight off to ② Serve (now pre-armed).
    if not res.error and (res.sibling_slug or res.profile_like):
        target = res.sibling_slug or res.profile_like
        lines.append("")
        lines.append(
            f"  [green]→ ② Serve[/green] is armed with [green]{target}[/green] "
            "[dim](no re-entry needed)[/dim]"
        )
    return "\n".join(lines)


class LaneServePane(Container):
    """② Serve — generate a minimal compose for the resolved CATALOG profile, then
    serve it (untested) through the reconcile-gated path (R3b-1, the critical new
    link).

    ⚠️  HONESTY (R3b-1): this serves a verbatim, UNTESTED reproduction of the
    resolved CATALOG slug's compose (the Route-C sibling, else the profile-like the
    fit-check ran against) — NOT the brought model's weights.  generate-compose.sh
    has no --repo / weight-swap yet; the full brought-model serve is a deferred
    follow-up.

    After a successful ① Bring fit-check, ⏎ here (action_serve_untested) runs
    ``generate-compose.sh`` for the resolved catalog slug, previews the compose
    VERBATIM badged "untested config reproduction of <slug>", and a confirm serves
    it through the SAME reconcile gate every serve uses (the generated compose
    CLAIMS the GPU).  Mission: reproduce + flag, never repair — the compose is
    shown as generated, NOT fit-adapted."""

    DEFAULT_CSS = """
    LaneServePane {
        height: 1fr;
        padding: 1 2;
    }
    LaneServePane #lane-serve-heading {
        text-style: bold;
        margin-bottom: 1;
    }
    LaneServePane #lane-serve-body {
        border: solid $primary;
        padding: 1 2;
        margin-top: 1;
        height: 1fr;
    }
    LaneServePane #lane-serve-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("② Serve — reproduce + serve the resolved catalog compose (untested)", id="lane-serve-heading")
        yield Static(
            "[dim]Stage ② of the Bring & Validate pipeline.\n"
            "\n"
            "Run ① Bring first to fit-check a model.  Then ⏎ here generates a\n"
            "minimal compose (generate-compose.sh — reproduce + flag, never\n"
            "repair) for the RESOLVED CATALOG slug (the Route-C sibling, else the\n"
            "profile-like the fit-check ran against), previews it VERBATIM, and\n"
            "serves it through the reconcile-gated confirm (the generated compose\n"
            "claims the GPU like any serve).\n"
            "\n"
            "[yellow]Note: this serves an UNTESTED reproduction of the catalog\n"
            "profile's compose — NOT your brought model's weights.  The bring-your-\n"
            "own weight-swap (generate-compose.sh --repo) is a deferred follow-up.\n"
            "[/yellow][/dim]",
            id="lane-serve-body",
        )
        yield Label(
            "[dim]\\[⏎] generate + preview + serve (reconcile-gated · untested)[/dim]",
            id="lane-serve-hint",
        )

    def set_status(self, text: str) -> None:
        self.query_one("#lane-serve-body", Static).update(text)

    def set_armed(self, byo: "Optional[ByoResult]") -> None:
        """N9 — pre-arm ② Serve from the cached ① Bring fit-check: show the
        resolved servable catalog target so ⏎ here serves it WITHOUT re-entering
        ① Bring.  Pure render off the cached ByoResult (no I/O).  When there's no
        usable fit-check yet, restore the calm "run ① Bring first" placeholder."""
        body = self.query_one("#lane-serve-body", Static)
        if byo is None or getattr(byo, "error", ""):
            body.update(
                "[dim]Stage ② of the Bring & Validate pipeline.\n"
                "\n"
                "Run ① Bring first to fit-check a model.  Then ⏎ here generates +\n"
                "previews + serves the resolved catalog compose (reconcile-gated,\n"
                "untested).[/dim]"
            )
            return
        slug = (
            getattr(byo, "sibling_slug", "")
            or getattr(byo, "profile_like", "")
        )
        repo = getattr(byo, "repo", "") or "—"
        lines = [
            "[green]● armed from ① Bring[/green] — ⏎ serves the resolved catalog compose (untested):",
            "",
            f"  [bold]brought[/bold]   [cyan]{repo}[/cyan]",
        ]
        if slug:
            lines.append(f"  [bold]serves[/bold]    [green]{slug}[/green]  [dim](resolved catalog profile)[/dim]")
        else:
            lines.append(
                "  [yellow]no servable catalog target resolved[/yellow] — the fit-check found "
                "no sibling/profile slug (the bring-your-own weight-swap is a deferred follow-up)."
            )
        lines.append("")
        lines.append(
            "[yellow]Note: serves an UNTESTED reproduction of the catalog profile's "
            "compose — NOT your brought model's weights.[/yellow]"
        )
        body.update("\n".join(lines))


class LanePromotePane(Container):
    """⑤ Promote — promote the fit-checked + measured model into the catalog.

    Hosts the [P] promote affordance relocated out of Run · Catalog (R3b-1).  The
    action (``action_promote_catalog`` → PromoteScaffoldScreen) is unchanged and
    producer-gated; this stage is its home in the lane."""

    DEFAULT_CSS = """
    LanePromotePane {
        height: 1fr;
        padding: 1 2;
    }
    LanePromotePane #lane-promote-heading {
        text-style: bold;
        margin-bottom: 1;
    }
    LanePromotePane #lane-promote-body {
        border: solid $primary;
        padding: 1 2;
        margin-top: 1;
        height: 1fr;
    }
    LanePromotePane #lane-promote-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("⑤ Promote — scaffold a curated catalog entry", id="lane-promote-heading")
        yield Static(
            "[dim]Final stage of the Bring & Validate pipeline.\n"
            "\n"
            "Once the model is fit-checked (① Bring), served (② Serve), gated\n"
            "(③ Gate) and measured (④ Measure), \\[P] computes a SCAFFOLD + GATE:\n"
            "a ModelProfile YAML skeleton + a compose_registry entry COMPUTED from\n"
            "the BYO arch facts + Evidence numbers, previewed before the gated\n"
            "(mock-only this phase) write into scripts/ + the guard suite.[/dim]",
            id="lane-promote-body",
        )
        yield Label(
            "[dim]\\[P] compute + preview the catalog-promotion scaffold (gated write)[/dim]",
            id="lane-promote-hint",
        )


# ── Mode switcher (left rail) ─────────────────────────────────────────────────────


# Phase R-realign (2-mode merge): the app serves BOTH consumers and producers,
# and every producer is a consumer.  Mode 0 MERGES the old Run + Operate (Catalog
# · Orchestration · Containers · Doctor in one TabbedContent); mode 1 is the
# producer "Bring & Validate" lane.  BOTH modes are visible by DEFAULT (the "full"
# surface); the LEAN surface hides mode 1 for a consumer who wants the minimal
# rig view (the C toggle).
MODES = [
    ("Run & Operate", "1"),
    # Mode 1 is the producer "Bring & Validate" lane (key 2, producer-gated so the
    # lean surface hides it).  Renumbered 2→1 in the 2-mode merge.
    ("Bring & Validate", "2"),
]

# Per-mode primary action (what ⏎ does), by mode index.  Mode 0's ⏎ is
# context-specific PER TAB (serve on Catalog, switch-scene on Orchestration, …);
# "Serve" is the headline hint.
PRIMARY_ACTIONS = ["Serve", "Run stage"]


class RailStatus(Static):
    """Persistent left-rail status card — mirrors c3t's TargetPane.

    Wired in Phase 3: shows the live detect / doctor read.  Until the first
    estate poll completes it shows a 'detecting…' placeholder."""

    PLACEHOLDER = (
        "[bold]Estate[/bold]\n"
        "\n"
        "[dim]detecting…[/dim]\n"
        "\n"
        "[dim]open the Orchestration tab to poll[/dim]"
    )

    def __init__(self, **kwargs):
        super().__init__(self.PLACEHOLDER, **kwargs)

    def update_from_state(self, state: EstateState, *, as_of: str = "") -> None:
        lines: list[str] = ["[bold]Estate[/bold]", ""]
        # A2/N2: a READ error (docker / nvidia-smi failure) shows as a distinct
        # red line at the top of the rail — the always-visible card must not
        # quietly read as a healthy idle rig when the read actually failed.
        err = (getattr(state, "error", "") or "").strip()
        if err:
            # MUST-FIX 3: render the ACTUAL error text (truncated to the part
            # before " — "), NOT a hardcoded "docker unreachable".  state.error
            # has two producers — a docker failure ("docker unreachable — …") AND
            # a detect failure ("detect failed: …", docker fine) AND an nvidia-smi
            # failure — so the literal "docker unreachable" mislabels the others.
            lines.append(f"[red]⚠ {_error_headline(err)}[/red]")
            lines.append("")
        for i in (0, 1):
            gpu = next((g for g in state.gpus if getattr(g, "index", -1) == i), None)
            if gpu is None:
                continue
            used = getattr(gpu, "mem_used_mib", 0) / 1024
            total = (getattr(gpu, "mem_total_mib", 0) or 1) / 1024
            pct = int(used / total * 100) if total else 0
            filled = max(0, min(10, round(pct / 10)))
            color = "green" if pct < 80 else "yellow" if pct < 95 else "red"
            bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (10 - filled)}[/dim]"
            lines.append(f"{bar} GPU{i} {used:.0f}/{total:.0f}G")
        lines.append("")
        if state.matched_slug:
            # F9: a port/substring registry match is a SHAPE guess, not a verified
            # identity — a brought model on a sibling's port masquerades as the
            # sibling otherwise.  Lead with the PROBED served id + 👤 badge and
            # demote the slug to "shape"; only an exact-container match may claim
            # the slug as the identity.
            _conf = (getattr(state.target, "match_confidence", "") or "") if state.target is not None else ""
            _served = (getattr(state.target, "model", "") or "").strip() if state.target is not None else ""
            if _conf == "shape":
                lines.append(f"model   {_served or state.matched_slug} 👤")
                if _served:
                    lines.append(f"[dim]shape   {state.matched_slug}[/dim]")
            else:
                lines.append(f"model   {state.matched_slug}")
        elif state.target is not None and getattr(state.target, "model", ""):
            lines.append(f"model   {state.target.model}")
        dr = state.doctor
        if dr.reachable:
            glyph = "[green]●[/green]" if dr.serving else "[yellow]○[/yellow]"
            lines.append(f"{glyph} {dr.summary}")
            # F3: the endpoint, compactly (the rail is ~30 cols — the FULL URL
            # lives on the Orchestration serving card; [u] copies it anywhere).
            _port = getattr(state.target, "host_port", 0) or 0
            if _port:
                lines.append(f"[dim]api :{_port}/v1 · \\[u] copy[/dim]")
        elif dr.booting:
            lines.append("[yellow]⏳[/yellow] booting…")
        else:
            lines.append("[red]○[/red] not reachable")
        # A3: stamp the freshness so the always-visible card is honest between
        # the periodic polls ("as of <Nm/Ns ago>").
        if as_of:
            lines.append("")
            lines.append(f"[dim]as of {as_of}[/dim]")
        self.update("\n".join(lines))


class HostStatsRail(Static):
    """FIX 3 — host disk + RAM usage in the LEFT RAIL (the "estate column").

    The maintainer's directive: "host repo/models disk and ram usage was meant to
    show in the estate column on the left but appears in the Orchestration tab."
    B5 rendered these into ``#disk-rail`` INSIDE the Orchestration sub-tab; this
    widget moves them to the global left rail (below RailStatus) where they persist
    across Run/Operate/Validate.  Telemetry is still FETCHED only on the Operate
    (mode-1) tick (no new subprocess churn elsewhere); the rail simply shows the
    last-known values — host disk/RAM move slowly, matching RailStatus's
    persist-last-state pattern.

    The bar-rendering math is the SAME as the former orch-pane ``_populate_disk_rail``
    (moved verbatim, not rewritten) so the disk/RAM bars are pixel-identical."""

    PLACEHOLDER = "[bold]Host[/bold]\n[dim]reading disk / RAM…[/dim]"

    def __init__(self, **kwargs):
        super().__init__(self.PLACEHOLDER, **kwargs)

    def populate_telemetry(self, tel: EstateTelemetry) -> None:
        """Render the disk bars (repo + /mnt/models) and the RAM line into the rail.

        A read failure surfaces an honest cue (the B2 "A2" rule) rather than a
        silent false-zero."""
        lines: list[str] = ["[bold]Host[/bold]"]

        def _bar_markup(pct: int) -> str:
            color = "green" if pct < 80 else "yellow" if pct < 95 else "red"
            full = max(0, min(10, round(pct / 100 * 10)))
            return f"▕[{color}]{'█' * full}[/{color}][dim]{'░' * (10 - full)}[/dim]▏"

        bar_lines: list[str] = []
        for d in tel.disks or []:
            bar_lines.append(
                f"[bold]{d.mount_label:<7}[/bold] {_bar_markup(d.pct)} "
                f"{d.pct:>3}%  {_human_gb(d.used)}/{_human_gb(d.total)}"
            )
        ram = tel.ram
        if ram and ram.total > 0 and not ram.error:
            bar_lines.append(
                f"[bold]{'RAM':<7}[/bold] {_bar_markup(ram.pct)} "
                f"{ram.pct:>3}%  {_human_gb(ram.used)}/{_human_gb(ram.total)}"
            )
        elif ram and ram.error:
            bar_lines.append(f"[dim]RAM: {ram.error}[/dim]")
        if not tel.disks and not (ram and ram.total > 0):
            # Honest failure cue — never a silent blank/false-zero (A2 rule).
            err = (tel.error or "host telemetry unavailable").strip()
            self.update(f"[bold]Host[/bold]\n[dim]{err}[/dim]")
            return
        if tel.error and (not tel.disks or (ram and ram.error)):
            bar_lines.append(f"[dim]⚠ {tel.error}[/dim]")
        self.update("\n".join(lines + bar_lines))


class ModeSwitcher(Static):
    """Left-rail mode selector — the 1/2 digit bindings still jump modes
    directly, AND (BUG 2) the widget is now keyboard-navigable: it is a focus
    stop (the FIRST stop in the Tab chain, being first in the left-rail DOM), and
    when FOCUSED its arrow keys switch the active mode.

    Focus model (vertical: Modes (rail) ↕ tab bar ↕ list):
      • Focusable with a VISIBLE focus style (the :focus accent border below).
      • ``↑``/``↓`` SELECT the active mode (prev / next visible mode), clamped to
        the modes VISIBLE on the current surface (the lean view has one mode, so
        ↑/↓ are inert there).  Focus STAYS on the ModeSwitcher so the user can keep
        arrowing.  Switching reuses the app's real mode-switch path so content +
        highlight + bindings update exactly as 1/2 do.
      • ``→`` (and ``Enter`` / ``Tab``) DESCEND into the selected mode's content —
        "go operate the mode you picked".  ``→``/``Enter`` route to descend; ``Tab``
        descends via the natural focus chain to the tab bar.
      • ``←`` is INERT (round-4 — left/right are no longer mode-switchers).
    The arrows act ONLY while the ModeSwitcher is focused (key events reach the
    focused widget first), so they never leak to the app from a list/tab bar."""

    # BUG 2 — a focus stop (and the FIRST one: it is first in the left-rail DOM).
    can_focus = True

    DEFAULT_CSS = """
    ModeSwitcher {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
    }
    /* BUG 2 — a VISIBLE focus ring so the Modes rail is an obvious focus stop
       (focus must never be stranded invisibly). */
    ModeSwitcher:focus, ModeSwitcher:focus-within {
        border: solid $accent;
    }
    ModeSwitcher .mode-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    ModeSwitcher .mode-item {
        color: $text;
    }
    ModeSwitcher .mode-item-active {
        color: $accent;
        text-style: bold;
    }
    ModeSwitcher .mode-action-hint {
        color: $text-muted;
        margin-top: 1;
    }
    ModeSwitcher .mode-hidden {
        display: none;
    }
    """

    def __init__(self, *, surface: str = "producer", **kwargs):
        super().__init__("", **kwargs)
        self._active = 0
        # Surface-aware (2-mode merge — inverted default): the FULL/default
        # ("producer") surface renders BOTH modes (Run & Operate + Bring &
        # Validate); the LEAN ("consumer") surface renders only mode 0.  Both mode
        # Labels are always composed; the lean surface just HIDES the producer-only
        # second (.mode-hidden) so the runtime [C] lean toggle can show it again
        # without an async re-mount (set_surface flips the class only).
        self._surface = surface if surface in ("consumer", "producer") else "producer"

    @property
    def _modes(self) -> list[tuple[str, str]]:
        """The modes VISIBLE for this surface — BOTH on the full/default surface,
        only the merged Run & Operate on the lean surface.  (Both Labels are always
        mounted; this is the visible slice — see set_surface.)"""
        return list(MODES) if self._surface == "producer" else list(MODES[:1])

    def compose(self) -> ComposeResult:
        yield Label("Modes", classes="mode-title")
        # Always compose BOTH mode Labels; hide the producer-only second on the
        # lean surface so the runtime [C] toggle can reveal it class-only.
        for i, (name, digit) in enumerate(MODES):
            classes = "mode-item-active" if i == 0 else "mode-item"
            if i >= len(self._modes):
                classes += " mode-hidden"
            yield Label(f"▸ {name} [{digit}]" if i == 0 else f"  {name} [{digit}]",
                        id=f"mode-{i}", classes=classes)
        yield Label(f"⏎ {PRIMARY_ACTIONS[0]}", id="mode-action-hint",
                    classes="mode-action-hint")

    def set_active(self, index: int) -> None:
        self._active = index
        for i, (name, digit) in enumerate(MODES):
            try:
                lbl = self.query_one(f"#mode-{i}", Label)
                lbl.remove_class("mode-item-active")
                lbl.add_class("mode-item")
                if i == index:
                    lbl.remove_class("mode-item")
                    lbl.add_class("mode-item-active")
                    lbl.update(f"▸ {name} [{digit}]")
                else:
                    lbl.update(f"  {name} [{digit}]")
            except Exception:
                pass
        try:
            self.query_one("#mode-action-hint", Label).update(
                f"⏎ {PRIMARY_ACTIONS[index]}"
            )
        except Exception:
            pass

    def set_surface(self, surface: str) -> None:
        """Re-render the rail for a runtime surface change (the [C] lean toggle).

        The full/default ("producer") surface shows BOTH modes; the lean
        ("consumer") surface shows only the merged Run & Operate.  Both mode Labels
        are always mounted, so this is a pure class flip — show/hide the
        producer-only second Label via ``.mode-hidden`` (no async re-mount, which
        raced the pilot under the headless harness)."""
        new = surface if surface in ("consumer", "producer") else "producer"
        if new == self._surface:
            return
        self._surface = new
        for i in range(len(MODES)):
            try:
                lbl = self.query_one(f"#mode-{i}", Label)
                if i >= len(self._modes):
                    lbl.add_class("mode-hidden")
                else:
                    lbl.remove_class("mode-hidden")
            except Exception:
                pass

    # BUG 2 — keyboard navigation while the ModeSwitcher is focused.  Handled in
    # the widget's OWN on_key so these arrows reach the app ONLY when this widget
    # holds focus (a focused widget sees key events first); when focus is on a
    # list / tab bar / Input / Select / modal the app's existing arrow model runs
    # unchanged.
    def on_key(self, event) -> None:
        key = event.key
        # Maintainer round-4 — the Modes arrow map is now AXIS-SPLIT:
        #   ↑/↓  → SELECT the active mode (prev / next visible mode), clamped.
        #          Focus STAYS here so the user can keep arrowing.
        #   →    → DESCEND into the selected mode's content (Enter alias).
        #   ←    → INERT (consumed so it doesn't leak, but does nothing).
        # The old behaviour switched the mode on ←/→ too — removed: in Modes you
        # pick with ↑/↓ and step INTO the flow with → (or Enter / Tab).
        if key in ("up", "down"):
            visible = len(self._modes)
            # LEAN surface (one visible mode) → arrows are inert (nothing to switch).
            if visible <= 1:
                event.stop()
                event.prevent_default()
                return
            delta = -1 if key == "up" else 1
            target = self._active + delta
            if target < 0 or target >= visible:
                # Clamp: already at the first/last visible mode → no-op.
                event.stop()
                event.prevent_default()
                return
            event.stop()
            event.prevent_default()
            # Reuse the app's REAL mode-switch path so content + highlight +
            # bindings update exactly as the 1/2 digit keys do, then re-home focus
            # on the ModeSwitcher (the switch focuses the mode's primary widget).
            app = self.app
            switch = getattr(app, "_switch_mode_keep_modes_focus", None)
            if switch is not None:
                switch(target)
            return
        # ← is INERT — consumed (so it never leaks to the app) but does nothing.
        if key == "left":
            event.stop()
            event.prevent_default()
            return
        # → (right) and Enter both DESCEND into the selected mode's content (Tab
        # also descends via the natural focus chain — the tab bar is the next
        # focusable).  ↓ is NOT overloaded to exit downward (it switches mode above).
        if key in ("right", "enter"):
            event.stop()
            event.prevent_default()
            app = self.app
            descend = getattr(app, "_descend_from_modes", None)
            if descend is not None:
                descend()
            return


# ── Keyboard-traversable footer (#5) ───────────────────────────────────────────────


class FocusableFooter(Footer):
    """The main-screen footer — a HINT BAR, deliberately OUT of the Tab focus chain.

    Stock ``Footer`` is ``can_focus_children=False`` and its ``FooterKey`` items
    are mouse-only (``on_mouse_down`` → ``simulate_key``).  An earlier batch made
    every footer key Tab-focusable so a keyboard user could Tab onto a footer
    affordance — but footer keys render NO visible focus ring, so Tab-stepping
    through ~10 of them created an invisible-focus black hole between the tab bar
    and the lists.  The footer's actions are already reachable via their
    advertised HOTKEYS (app/screen BINDINGS, independent of footer focus) and via
    mouse click (stock ``Footer`` behaviour), so footer keys add no value to the
    Tab chain.  We therefore keep the stock ``can_focus_children=False`` and do
    NOT mark any ``FooterKey`` focusable — Tab cycles only the VISIBLE, meaningful
    stops (the tab bar + the active list).

    Recompose guard (retained): the stock Footer recomposes on EVERY
    ``bindings_changed`` signal, even when the DISPLAYED key set is unchanged
    (e.g. a pure focus move between tables re-fires the signal).  We suppress the
    redundant rebuild when the displayed binding signature (the ordered
    (key, action, enabled) tuples) is UNCHANGED — a real mode/context change
    flips the signature and DOES recompose, so the footer re-renders correctly;
    only no-op rebuilds are skipped."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_binding_sig: Optional[tuple] = None

    def _binding_signature(self) -> tuple:
        try:
            active = self.screen.active_bindings
        except Exception:
            return ()
        return tuple(
            (binding.key, binding.action, enabled)
            for (_node, binding, enabled, _tooltip) in active.values()
            if binding.show
        )

    def compose(self) -> ComposeResult:
        # Snapshot the signature we are composing for, so bindings_changed can
        # skip a redundant recompose that wouldn't change the rendered key set.
        self._last_binding_sig = self._binding_signature()
        yield from super().compose()

    def bindings_changed(self, screen) -> None:
        # Only recompose when the DISPLAYED key signature actually changed; a pure
        # focus move (which also fires this signal) leaves the signature identical
        # and must not trigger a rebuild.  A genuine mode/context change flips the
        # signature → super().bindings_changed runs → footer re-renders correctly.
        self._bindings_ready = True
        if self._binding_signature() == self._last_binding_sig:
            return
        super().bindings_changed(screen)


# ── Command palette (N6) ──────────────────────────────────────────────────────────


# N6: the user-facing actions exposed in the Textual command palette (Ctrl+P).
# Each entry is (action_method, title, help) — title is what the user fuzzy-types,
# help is the secondary line.  These invoke the SAME ``action_*`` methods the key
# bindings call, so the palette is a discoverability layer, never a parallel code
# path.  Producer-only actions (``_palette_is_producer_only``) are FILTERED OUT on
# the consumer surface — the palette respects the same surface gate as
# ``check_action`` (no producer action runnable on the consumer surface).  Mode-
# gated actions stay listed: invoking one out of its mode no-ops EXACTLY as the
# binding does (the ``action_*`` methods guard internally on ``_active_mode``), so
# the palette behaviour is consistent with the keyboard.
_PALETTE_COMMANDS: tuple[tuple[str, str, str], ...] = (
    # Always-on navigation / global verbs.
    ("mode_run", "Run & Operate mode", "Catalog · Orchestration · Containers · Doctor"),
    ("mode_operate", "Operate (Orchestration tab)", "Jump to the live estate in the merged mode"),
    ("mode_validate", "Bring & Validate mode", "Producer lane ① Bring → ⑤ Promote"),
    ("toggle_contribute", "Toggle lean view", "Hide / restore the Bring & Validate mode"),
    ("toggle_rail", "Toggle left rail", "Collapse / restore Modes + Estate rail"),
    ("copy_context", "Copy current view", "Copy the highlighted slug / open report / selection to the clipboard ([Y])"),
    ("settings", "Settings (model dir · HF token)", "Edit where weights download to + the HuggingFace token ([S])"),
    ("refresh", "Refresh", "Re-read the live data layer for the active mode"),
    ("help", "Help", "Show the keybindings + phase help overlay"),
    # Run & Operate · Catalog tab.
    ("primary_action", "Serve selected / primary action", "⏎ — serve the selected slug (reconcile-gated)"),
    ("explain", "Explain selected slug", "Catalog — detail + cross-rig benchmarks"),
    ("filter_catalog", "Filter catalog", "Catalog — filter by slug / engine / status"),
    ("toggle_catalog_deprecated", "Show/hide deprecated", "Catalog — reveal 🗑️ deprecated slugs (hidden by default)"),
    ("copy_endpoint", "Copy the serving API URL", "Run & Operate — copy http://<lan>:<port>/v1 for your agent/client (no auth by default)"),
    ("set_default", "Set default", "Catalog — pin the selected slug as model default"),
    ("clear_default", "Clear default", "Catalog — clear the model default pin"),
    ("optimize_card", "Optimize for my card", "Catalog — v0.10.0 seam (not available yet)"),
    # Run & Operate · Orchestration / Containers / Doctor tabs.
    ("serving_stop", "Stop this model", "Orchestration — stop JUST the serving container (gated)"),
    ("serving_restart", "Restart serving", "Orchestration — restart the serving container (gated)"),
    ("serving_switch", "Switch model", "Orchestration — flip to the Catalog tab to pick another"),
    ("estate_off", "Stop ALL (estate down)", "Orchestration — tear down the whole estate (gated)"),
    ("power_cap", "Power cap…", "Orchestration — power-cap menu: default 230W / clear / custom W (gated)"),
    ("power_cap_sweep", "Power cap sweep", "Doctor — sweep power caps + bench at each (gated)"),
    ("container_logs", "Container logs", "Containers — stream the selected container's logs"),
    ("doctor_rerun", "Re-run Doctor health", "Doctor — re-run the live health read (read-only)"),
    ("doctor_verify", "Verify serving", "Doctor — send a test query to the model (verify.sh · read)"),
    ("doctor_verify_full", "Verify-full battery", "Doctor — functional battery (verify-full.sh · ~1-2 min · read)"),
    ("full_report", "Full system report (report.sh --full)", "Doctor / ③ Gate — ~43-min battery (uses the serving GPUs · gated)"),
    # Share-back (consumer-resident — NOT producer-gated).
    ("rig_report", "Rig report", "Paste-ready rig/bench snapshot (read · no network)"),
    ("submit_bench", "Submit bench", "Submit the latest benched result (gated · never auto)"),
    ("report_problem", "Report a problem", "Paste-ready issue from the failure context (read)"),
    # Producer lane (Bring & Validate) — filtered out on the lean surface.
    ("serve_untested", "Serve untested (② Serve)", "Producer lane — generate a compose + serve it untested"),
    ("measure_vs_bar", "Compare vs catalog bar (④ Measure)", "Producer lane — read · flags protocol"),
    ("evaluate_target", "Evaluate running target", "Producer lane — c3t evaluate (confirm-gated)"),
    ("promote_catalog", "Promote to catalog (⑤ Promote)", "Producer lane — scaffold + gated write"),
)

# The producer-only subset — kept in sync with ``CockpitApp._PRODUCER_ONLY`` (a
# guard test asserts the two agree).  Used to FILTER the palette on consumer.
_PALETTE_PRODUCER_ONLY: frozenset[str] = frozenset({
    "mode_validate", "promote_catalog", "evaluate_target", "serve_untested",
    "measure_vs_bar",
})

# Single source of truth: each mode-level tab → the CSS id of its PRIMARY content
# list (the DataTable focus descends into / ascends out of).  The single map read
# by ALL THREE focus consumers — tab-activation focus (on_tabbed_content_tab_activated),
# mode-switch focus (_focus_mode_primary via _primary_list_for_active_tab), and the
# arrow-key focus-descent actions (action_descend_to_content / action_ascend_to_tabbar)
# — so the tab→list association never drifts.  Tabs absent from this map have NO primary
# list (mode 0 · Doctor; lane ① Bring / ② Serve / ⑤ Promote) — descend is a no-op
# there and there is nothing to ascend from.
_TAB_PRIMARY_LIST: dict[str, str] = {
    "tab-catalog":       "#catalog-table",
    "tab-run":           "#run-ladder-table",
    "tab-evidence":      "#evidence-table",
    "tab-orchestration": "#scene-table",
    "tab-containers":    "#containers-table",
}


class CockpitCommands(Provider):
    """N6 — fuzzy command-palette provider for the cockpit's user-facing actions.

    Surface-gated: producer-only actions are NOT offered on the consumer surface
    (mirrors ``CockpitApp._PRODUCER_ONLY`` / ``check_action``'s surface gate).
    Selecting a command dispatches the SAME ``action_*`` method the key binding
    would — so the palette is a pure discoverability layer.
    """

    def _available(self) -> list[tuple[str, str, str]]:
        app = self.app
        producer = getattr(app, "_surface", "consumer") == "producer"
        out: list[tuple[str, str, str]] = []
        for action, title, help_text in _PALETTE_COMMANDS:
            if not producer and action in _PALETTE_PRODUCER_ONLY:
                continue
            out.append((action, title, help_text))
        return out

    def _run(self, action: str):
        app = self.app

        def _do() -> None:
            method = getattr(app, f"action_{action}", None)
            if method is not None:
                method()

        return _do

    async def discover(self) -> Hits:
        """Surface a sensible default set when the palette opens (no query yet)."""
        for action, title, help_text in self._available():
            yield DiscoveryHit(title, self._run(action), help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for action, title, help_text in self._available():
            score = matcher.match(title)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(title),
                    self._run(action),
                    help=help_text,
                )


# ── Main application ──────────────────────────────────────────────────────────────


class CockpitApp(App):
    """club3090 serve cockpit — both modes (Run & Operate · Bring & Validate) wired to the live data layer."""

    TITLE = "club3090 cockpit"
    SUB_TITLE = "wired"

    # N6 — register the cockpit's action provider alongside Textual's built-in
    # system commands so Ctrl+P fuzzy-searches our verbs too.
    COMMANDS = App.COMMANDS | {CockpitCommands}

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("question_mark", "help", "Help", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        # Sub-tab cycle — shown only in modes that have sub-tabs (check_action gates).
        Binding("left_square_bracket", "prev_subtab", "Prev tab", show=False),
        Binding("right_square_bracket", "next_subtab", "Next tab", show=False),
        # Arrow-key focus descent (keyboard-nav enhancement):
        #   [down] on the tab bar → descend INTO the active tab's primary list.
        #     A NON-priority binding: the tab bar (ContentTabs) does NOT consume
        #     `down`, so the key bubbles to this app binding; the DataTable DOES
        #     consume `down` for its row cursor, so when focus is already on the
        #     list this binding's check_action returns falsey and the list keeps it.
        #   [up] at row 0 of a primary list → ascend back UP to the tab bar.
        #     PRIORITY so it's checked BEFORE the focused DataTable (which consumes
        #     `up` even at row 0).  check_action returns True ONLY at cursor_row==0
        #     on a primary list; above row 0 it's inert → the DataTable moves the
        #     cursor up as normal.  (Same priority+check_action shape the confirm
        #     modal uses for enter/f.)
        Binding("down", "descend_to_content", show=False),
        Binding("up", "ascend_to_tabbar", show=False, priority=True),
        # #8 — collapse / restore the left rail (Modes + Estate) so the content
        # area uses the full terminal width.  Always-on (no mode/surface gate).
        Binding("full_stop", "toggle_rail", "Rail", show=False),
        # Batch 4 — [Y] yank the contextually-relevant text to the system clipboard
        # (DataTable cells can't be drag-selected — ALLOW_SELECT is False — so a
        # context-aware copy is the reliable path).  shift+←/→ page-scroll a wide
        # table horizontally (faster than the per-column ←/→ cursor).
        Binding("Y", "copy_context", "Copy", show=False),
        Binding("shift+left", "hscroll_left", "Scroll ◀", show=False),
        Binding("shift+right", "hscroll_right", "Scroll ▶", show=False),
        # Download UX — [S] edit the model dir + HF token (where weights are
        # fetched to / gated-repo auth).  Always-on, persisted.  show=True: it's a
        # setup prerequisite for downloads, so it earns a permanent footer slot —
        # the "press [S]" banner only appears when the model dir is MISSING, which
        # leaves it invisible on a rig where the dir is already set.
        Binding("S", "settings", "Settings", show=True),
        # Context-sensitive — check_action enables/shows them only in the right mode.
        Binding("slash", "filter_catalog", "Filter", show=False),
        Binding("backslash", "toggle_catalog_model", "Model", show=False),
        Binding("h", "toggle_catalog_deprecated", "Deprecated", show=False),
        Binding("u", "copy_endpoint", "API URL", show=False),
        Binding("e", "explain", "Explain", show=False),
        # 2-mode merge: [1] = merged Run & Operate, [2] = Bring & Validate lane.
        Binding("1", "mode_run", "Run & Operate", show=True),
        Binding("2", "mode_validate", "Bring & Validate", show=True),
        Binding("enter", "primary_action", "Select", show=True),
        # Catalog (Run) — default pin management (.env write, gated=no GPU).
        Binding("d", "set_default", "Set default", show=False),
        Binding("D", "clear_default", "Clear default", show=False),
        # Operate · Containers — logs (read) + restart/stop (gated writes).
        # [s] is context-sensitive: restart (Operate · Containers) vs submit
        # (Validate · Evidence) — routed by mode/tab in action_s_key.
        Binding("l", "container_logs", "Logs", show=False),
        Binding("s", "s_key", "Restart / Submit", show=False),
        Binding("x", "container_stop", "Stop", show=False),
        Binding("X", "container_rm", "Remove", show=False),
        # Operate · Orchestration — stop all (estate down, gated write).
        Binding("o", "estate_off", "Stop all", show=False),
        # Operate · Orchestration — power-cap menu (default / clear / custom W).
        Binding("c", "power_cap", "Power cap", show=False),
        # Operate · Doctor — power-cap sweep (heavy A/B bench; gated rig write).
        Binding("w", "power_cap_sweep", "Cap sweep", show=False),
        # Operate · Containers / Validate — context-sensitive read keys.
        Binding("t", "context_t", "Top / Sort", show=False),
        # Phase 5 — the three v2 hooks:
        #   [v] Operate · evaluate the running target via c3t (confirm-gated, mock-only)
        #   [P] Run · promote the BYO model to the catalog (scaffold + gated write)
        #   [O] Run · optimize for my card (dormant v0.10.0 seam)
        Binding("v", "evaluate_target", "Evaluate", show=False),
        Binding("P", "promote_catalog", "Promote", show=False),
        # R3b-1 — producer lane ② Serve: generate a compose + serve it untested
        # (also reachable via ⏎ on the ② Serve stage).
        Binding("g", "serve_untested", "Serve untested", show=False),
        Binding("O", "optimize_card", "Optimize", show=False),
        # R3b-2 — producer lane ④ Measure: compare the selected tag's measured
        # numbers to the curated catalog bar (READ · producer-only).
        Binding("m", "measure_vs_bar", "vs catalog bar", show=False),
        # R3b-2 — producer lane: the ~43-min FULL validation battery
        # (report.sh --full) — confirm-gated, bg-streamed, producer-only, uses the
        # serving model (claims no GPU); NEVER auto-fired.
        Binding("F", "full_report", "Full report", show=False),
        # Phase R / R2b — consumer share-back affordances (NOT producer-gated):
        #   [R] rig report (READ · paste-ready)     — Run + Operate
        #   [B] submit bench (OUTWARD write · gated) — Operate
        #   [!] report a problem (READ · paste-ready, surfaced at a failed serve)
        Binding("R", "rig_report", "Rig report", show=False),
        Binding("B", "submit_bench", "Submit bench", show=False),
        Binding("exclamation_mark", "report_problem", "Report problem", show=False),
        # 2-mode merge: [C] = the LEAN-view toggle.  Both modes show by default;
        # [C] hides the Bring & Validate mode (the minimal rig view) + back.
        # Persists the choice for next launch.  Always available — NOT in
        # _PRODUCER_ONLY (it's how a lean-view user restores the full view).
        Binding("C", "toggle_contribute", "Lean view", show=False),
        # A4 — TARGETED serving verbs on Operate · Orchestration's #serving-line
        # (the most-looked-at panel).  Unlike [o] stop-ALL (which tears the whole
        # estate down, killing co-resident ComfyUI / studio), these act on JUST
        # the serving model's container.  All three writes are CONFIRM-gated
        # through ConfirmActionScreen → the reconcile gate (never auto-fired):
        #   [k] stop just this model        (resolve container by matched slug)
        #   [b] restart the serving container
        #   [n] switch — jump to Run · Catalog to pick another
        Binding("k", "serving_stop", "Stop this model", show=False),
        Binding("b", "serving_restart", "Restart serving", show=False),
        Binding("n", "serving_switch", "Switch model", show=False),
        # #4 — Operate · Doctor: RE-RUN the live health read on demand (READ-only).
        # [r] also refreshes Doctor (via load_estate→load_doctor), but [y] is the
        # discoverable, Doctor-resident re-run verb.
        # FIX 2 — show=True so the real Doctor verb surfaces in the footer (the
        # OLD ⏎ "Select" no-op on Doctor is now hidden by check_action).  It is
        # context-gated to mode 0 · Doctor in _CONTEXT_KEYS, so it only renders
        # there.
        Binding("y", "doctor_rerun", "Re-run health", show=True),
        # Batch 3 — Operate · Doctor: "is the model serving correctly?".  [v] sends
        # a test query (verify.sh), [V] runs the ~1-2 min functional battery
        # (verify-full.sh).  Both READ-only.  [v] DELIBERATELY shares its key with
        # the producer-lane [v] evaluate_target — the contexts are disjoint (mode 0
        # Doctor vs mode 1 ② Serve), so check_action activates exactly one (the
        # same proven pattern as the confirm modal's dual ``enter`` bindings).
        Binding("v", "doctor_verify", "Verify serving", show=True),
        Binding("V", "doctor_verify_full", "Verify-full", show=False),
    ]

    CSS = """
    /* Per-pane control/hint lines must stay WITHIN the viewport — a Label is
       `width: auto` (sizes to its content), so a long control hint runs off the
       right edge and pushes the page into horizontal scroll, hiding the controls.
       Constrain every hint to the available width so it WRAPS (all controls stay
       visible, no h-scroll).  An id selector beats Label's type-level default. */
    #catalog-hint, #orch-hint, #containers-hint, #run-hint, #doctor-hint,
    #evidence-hint, #lane-bring-hint, #lane-serve-hint, #lane-promote-hint {
        width: 1fr;
    }
    #main-layout {
        height: 1fr;
    }
    #left-rail {
        width: 32;
        height: 1fr;
        padding: 0 0;
    }
    /* #8 — collapsed rail: the content area then claims the full width. */
    #left-rail.rail-hidden {
        display: none;
    }
    #rail-status {
        width: 1fr;
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
        margin-top: 1;
        color: $text;
    }
    /* FIX 3 — host disk/RAM card at the rail bottom (auto-height, below status). */
    #host-stats-rail {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
        margin-top: 1;
        color: $text;
    }
    #content-area {
        width: 1fr;
        height: 1fr;
    }
    .mode-panel {
        width: 1fr;
        height: 1fr;
        display: none;
    }
    .mode-panel.active {
        display: block;
    }
    /* Transient Run boot-output pane — hidden until a serve commits, then
       revealed (and given height) so the boot log streams below the catalog. */
    #panel-run > #serve-live {
        display: none;
    }
    #panel-run > #serve-live.serving {
        display: block;
        height: 12;
        margin: 0 1 1 1;
    }
    /* FIX 1 — while a ModalScreen is topmost, suppress the BASE FocusableFooter
       (the main screen's footer renders UNDER/around the modal otherwise, showing
       global keys + "Enter Select" that are NOT active).  A modal either renders its
       own Footer (only ConfirmActionScreen does) or advertises its keys in-body —
       either way the BASE footer's hints are stale.  Toggled by the screen-stack
       watcher (_sync_base_footer_visibility), restored when the modal pops. */
    FocusableFooter.base-footer-hidden {
        display: none;
    }
    """

    # ── Dynamic binding visibility ─────────────────────────────────────────────────

    # Actions that are always active regardless of mode or focused widget.
    _ALWAYS_ON: frozenset[str] = frozenset({
        "quit", "help", "refresh",
        "mode_run", "mode_operate", "mode_validate",
        # FIX 2 — ``primary_action`` (⏎) was always-on, so the footer ALWAYS
        # advertised "Enter Select" even on tabs where ⏎ is a no-op (Doctor /
        # Containers).  It is now context-gated in check_action (only the surfaces
        # whose action_primary_action does real work show it).  NOT in _ALWAYS_ON.
        # #8 — the left-rail toggle is a pure view control (no write, no mode
        # dependency), reachable everywhere.
        "toggle_rail",
        # The lean-view toggle [C] must stay reachable on BOTH surfaces — always-on,
        # NOT in _PRODUCER_ONLY (a lean-view user that can't toggle could never
        # restore the full view).
        "toggle_contribute",
        # Batch 4 — [Y] copy the contextually-relevant text (highlighted row's
        # slug / the open report's body / any active selection) to the system
        # clipboard (OSC52).  Reachable everywhere; no-ops with a notify when
        # there's nothing copyable.  shift+←/→ page-scroll the active wide table.
        "copy_context", "hscroll_left", "hscroll_right",
        # [S] download Settings (MODEL_DIR + HF_TOKEN) — reachable everywhere.
        "settings",
    })

    # Context key → (modes, subtabs) where it should be enabled.
    # modes: set of _active_mode integers.  subtabs: set of active tab IDs, or
    # None meaning "any sub-tab in those modes" (used for whole-mode keys).
    # The sub-tab cycle keys are handled separately below.
    # Phase R-realign (2-mode merge): the OLD Operate mode (was 1) folded into the
    # merged mode 0.  So every context key that used to gate to mode 1 now gates to
    # mode 0 + its (still-present) Operate sub-tab; the Catalog keys gate to mode 0
    # + tab-catalog so they don't leak into the Orchestration/Containers/Doctor
    # tabs that now share the merged mode.  The producer lane (was 2) is now 1.
    _CONTEXT_KEYS: dict[str, tuple[set[int], Optional[set[str]]]] = {
        # Merged mode 0 · Catalog tab
        "filter_catalog":   ({0}, {"tab-catalog"}),  # Catalog
        "toggle_catalog_deprecated": ({0}, {"tab-catalog"}),  # Catalog — [h] hide/show deprecated
        # [u] copy the serving API URL — the endpoint is rig-global, so it's
        # live on EVERY merged-mode tab (F3; guards inside the action when
        # nothing is serving).
        "copy_endpoint":    ({0}, {"tab-catalog", "tab-orchestration", "tab-containers", "tab-doctor"}),
        "explain":          ({0}, {"tab-catalog"}),  # Catalog (guards inside action)
        "set_default":      ({0}, {"tab-catalog"}),  # Catalog
        "clear_default":    ({0}, {"tab-catalog"}),  # Catalog
        "optimize_card":    ({0}, {"tab-catalog"}),  # Catalog
        # Producer Bring & Validate lane (mode 1 in the 2-mode merge).  Both
        # producer-gated (hidden on the lean surface).
        "promote_catalog":  ({1}, None),          # Bring & Validate lane (⑤ Promote)
        "evaluate_target":  ({1}, None),          # Bring & Validate lane (the c3t hook)
        "serve_untested":   ({1}, {"tab-serve"}), # Bring & Validate lane (② Serve)
        # [m] vs-bar on ④ Measure (tab-evidence); [F] full battery on the lane's
        # ③ Gate (tab-run) AND on Operate · Doctor (Batch 3 — a consumer can run
        # the full system battery from Doctor; it's warned + confirm-gated).
        "measure_vs_bar":   ({1}, {"tab-evidence"}),
        "full_report":      ({0, 1}, {"tab-doctor", "tab-run"}),
        # Merged mode 0 · Orchestration tab
        "estate_off":       ({0}, {"tab-orchestration"}),
        "power_cap":        ({0}, {"tab-orchestration"}),
        # power-cap sweep lives on Doctor now (a tuning/diagnostic bench, not a
        # live-estate control) — prune was removed from the cockpit entirely.
        "power_cap_sweep":  ({0}, {"tab-doctor"}),
        # A4 — targeted serving verbs on the #serving-line (Orchestration).
        "serving_stop":     ({0}, {"tab-orchestration"}),
        "serving_restart":  ({0}, {"tab-orchestration"}),
        "serving_switch":   ({0}, {"tab-orchestration"}),
        # Merged mode 0 · Doctor tab (#4 — re-run the READ-only health read;
        # Batch 3 — [v] verify / [V] verify-full test queries to the serving model)
        "doctor_rerun":     ({0}, {"tab-doctor"}),
        "doctor_verify":    ({0}, {"tab-doctor"}),
        "doctor_verify_full": ({0}, {"tab-doctor"}),
        # Merged mode 0 · Containers tab
        "container_logs":   ({0}, {"tab-containers"}),
        # [s] is handled by a DEDICATED branch in check_action (FIX 2) — Containers
        # (restart) on mode 0 + the lane's ④ Measure (submit) on mode 1 — so it is
        # NOT listed here (the old `({0, 1}, None)` entry was over-broad: it showed
        # [s] on every mode-0 tab and every lane stage).
        "container_stop":   ({0}, {"tab-containers"}),
        "container_rm":     ({0}, {"tab-containers"}),
        # [t] = docker top on the Containers tab.
        "context_t":        ({0}, {"tab-containers"}),
        # Phase R / R2b — consumer share-back (CONSUMER-resident — NOT producer-
        # gated; absent from _PRODUCER_ONLY so they work on the lean surface too).
        # All three live in the merged mode 0 (the whole consumer rig surface):
        #   rig_report     — any merged-mode tab (catalog OR the live estate).
        #   submit_bench   — merged mode (you submit measured results once a bench
        #                    exists; the live estate / evidence is here).
        #   report_problem — any merged-mode tab (surfaced AT a failed serve in
        #                    Catalog, and reachable while operating).
        "rig_report":       ({0}, None),
        "submit_bench":     ({0}, None),
        "report_problem":   ({0}, None),
    }

    # Producer-only actions — hidden on the LEAN surface (the 2-mode merge keeps
    # the gate machinery, just inverts the default: the FULL/default surface shows
    # everything, the lean surface hides the producer "Bring & Validate" lane).
    # The gate fires in check_action BEFORE _ALWAYS_ON, so listing ``mode_validate``
    # here hides the ENTIRE producer lane (its mode switch + ladder + evidence) on
    # the lean surface; the default/full surface falls through to the normal
    # context result.
    #   NOT gated: the consumer share-back (rig_report / submit_bench /
    #   report_problem) is consumer-resident and stays reachable on the lean
    #   surface too.
    #   Lane-resident producer verbs: [v] evaluate_target + [serve_untested]
    #   (② Serve) + [P] promote_catalog (⑤ Promote) + [m] measure_vs_bar
    #   (④ Measure).
    #   NOT producer-only: [F] full_report — Batch 3 surfaced the ~43-min battery
    #   on the consumer Operate · Doctor too (context-gated to tab-doctor + the
    #   lane's ③ Gate tab-run), so it must stay reachable on the lean surface.
    _PRODUCER_ONLY: frozenset[str] = frozenset({
        "mode_validate", "promote_catalog", "evaluate_target", "serve_untested",
        "measure_vs_bar",
    })

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Return True (enabled + shown in footer), False (disabled + hidden in footer).

        Rules (in priority order):
        0. Surface gate — producer-only actions are hidden on the consumer
           surface, checked BEFORE the always-on set so it wins for EVERY action
           class (including mode_* switches, which live in _ALWAYS_ON; R3a hides
           the producer Bring & Validate MODE (mode_validate) + [P] promote on
           consumer this way).
        1. Always-on set — True unconditionally.
        2. A filter Input is focused — Textual's Input.is_printable already calls
           event.stop() for letter/digit keys, so they never reach app bindings.
           We still return False for all context keys here to hide them from the
           footer (avoids misleading `e Explain` hint while typing a query).
           Mode-switch keys (1–4) and sub-tab keys are kept visible/active because
           digits are printable and are stopped by Input, making this safe.
        3. Context key — True only in the (mode, subtab) set defined above;
           False otherwise (hidden from footer so the footer is mode-accurate).
        4. Sub-tab cycle keys — True only in modes that have sub-tabs.
        5. Everything else — True (pass-through; modals handle their own capture).
        """
        from textual.widgets import Input as _Input

        # Surface gate (R3a): producer-only actions are hidden on the consumer
        # surface — checked BEFORE _ALWAYS_ON so it wins for EVERY action class,
        # including mode_* switches (which live in _ALWAYS_ON). R3a hides the
        # producer Bring & Validate MODE (mode_validate) + [P] promote on
        # consumer via this gate, so it MUST beat _ALWAYS_ON.
        if self._surface != "producer" and action in self._PRODUCER_ONLY:
            return False

        # Arrow-key focus descent (tab bar ↔ primary list).  These are gated here —
        # BEFORE _ALWAYS_ON — so the result is fully controlled (neither is in
        # _ALWAYS_ON / _PRODUCER_ONLY).  show=False, so the bool only governs whether
        # the binding FIRES, never footer visibility.
        if action in ("descend_to_content", "ascend_to_tabbar"):
            # Under a modal (screen_stack deeper than the main screen): NEVER fire —
            # modals own their own keys, including arrow keys inside their widgets.
            if len(self.screen_stack) > 1:
                return False
            focused = self.focused
            if action == "descend_to_content":
                # [down] descends ONLY when focus is the active mode's tab bar AND
                # the active tab maps to a primary list.  When focus is the list
                # itself (or an Input/Select in the lane), return False so [down]
                # reaches the DataTable cursor / the widget's own handler.
                bar = self._active_tab_bar()
                if bar is None or focused is not bar:
                    return False
                return self._primary_list_for_active_tab() is not None
            # ascend_to_tabbar: [up] ascends in EXACTLY ONE case (priority binding):
            #   focus is the active tab's PRIMARY DataTable at cursor row 0 →
            #   ascend to the tab bar (above row 0 / off a primary list → the
            #   binding is inert and the DataTable handles [up] = cursor up).
            #
            # Maintainer round-4: [up] on the TAB BAR is now INERT — it must NOT
            # jump to the ModeSwitcher (that round-3 behaviour is removed).  Return
            # False when focus is the tab bar so the key is not consumed / no-ops;
            # the ModeSwitcher stays reachable via Shift+Tab (focus chain).
            bar = self._active_tab_bar()
            if bar is not None and focused is bar:
                return False
            table = self._primary_list_for_active_tab()
            if table is None or focused is not table:
                return False
            try:
                return table.cursor_row == 0
            except Exception:
                return False

        if action in self._ALWAYS_ON:
            return True

        # When ANY text Input is focused, hide the context-key bindings from the
        # footer.  The Input's own _on_key stops printable characters before they
        # reach app bindings, but we hide them for footer accuracy (the filter +
        # the lane repo inputs share letters with hotkeys q/e/s/w/c/p/o, so
        # a stale `e Explain` hint while typing a query is misleading).
        #
        # R4 (folds R3b-1 LOW item b): reverted to the blanket "any Input focused"
        # gate.  The R3b-1-era scoping to catalog-filter ONLY guarded against the
        # producer lane's ① Bring auto-focusing its HF-repo Input — but the lane
        # was deliberately built to land on the tab bar (NOT its input — see
        # _focus_mode_primary / test_switch_to_validate_lands_on_bring_stage), so
        # the lane's own context keys ([P]/[v]/② serve_untested) are only hidden
        # if the user explicitly TABs into the input, which is the correct
        # footer-accuracy behaviour (the digit/bracket keys still route fine).
        focused = self.focused
        if isinstance(focused, _Input):
            if action in self._CONTEXT_KEYS:
                return False
            if action in ("prev_subtab", "next_subtab"):
                return False

        # FIX 2 — ⏎ (primary_action) is shown ONLY on surfaces whose
        # action_primary_action does real work: merged mode 0 · Catalog (serve) +
        # Orchestration (scene switch), and EVERY Bring & Validate stage (mode 1).
        # On Doctor / Containers (mode 0) ⏎ is a no-op → hide it so the footer never
        # advertises a key that does nothing.
        if action == "primary_action":
            return self._primary_action_enabled()

        # FIX 2 — [s] is context-sensitive (action_s_key routes restart vs submit)
        # but the binding was over-broad ({0,1}, no sub-tab).  Constrain it to the
        # ONLY two surfaces it acts on: merged mode 0 · Containers (docker restart)
        # and the Bring & Validate lane's ④ Measure stage (submit-to-localmaxxing).
        if action == "s_key":
            if self._active_mode == 0:
                return self._current_subtab() == "tab-containers"
            if self._active_mode == 1:
                return self._active_validate_tab() == "tab-evidence"
            return False

        # Sub-tab cycle keys: both modes have sub-tabs (merged 0 = 4 tabs; lane 1
        # = the ①→⑤ stages).
        if action in ("prev_subtab", "next_subtab"):
            return self._active_mode in (0, 1)

        # Context keys.
        if action in self._CONTEXT_KEYS:
            modes, subtabs = self._CONTEXT_KEYS[action]
            if self._active_mode not in modes:
                return False
            if subtabs is not None:
                active_tab = self._current_subtab()
                if active_tab not in subtabs:
                    return False
            return True

        return True

    @staticmethod
    def _direct_pane_ids(tc: TabbedContent) -> list[str]:
        """The DIRECT TabPane ids of a TabbedContent, in display order.

        MUST-FIX 2 — the recursive ``tc.query(TabPane)`` also returns the panes of
        any NESTED TabbedContent (the Containers drill Logs/Top/Config sub-tabs),
        which leaks ``drill-tab-*`` ids into the mode-level ``[`` / ``]`` cycle and
        into ``_current_subtab``.  Reading the TabbedContent's OWN ContentSwitcher
        child returns only its direct panes, so the cycle / current-subtab stay
        scoped to the 4 real Operate tabs (or the 5 lane stages)."""
        try:
            switcher = tc.get_child_by_type(ContentSwitcher)
        except Exception:
            # Fallback: recursive query (pre-fix behaviour) if the switcher can't be
            # located — better a leaky cycle than a crash.
            return [p.id for p in tc.query(TabPane) if p.id]
        return [c.id for c in switcher.children if isinstance(c, TabPane) and c.id]

    def _current_subtab(self) -> str:
        """Return the active tab ID for the current mode's TabbedContent, or ''."""
        tab_ids = {
            0: "#operate-tabs",   # merged Run & Operate (Catalog · Orch · … · Doctor)
            1: "#validate-tabs",  # Bring & Validate lane
        }
        tc_id = tab_ids.get(self._active_mode, "")
        if not tc_id:
            return ""
        try:
            tc = self.query_one(tc_id, TabbedContent)
        except Exception:
            return ""
        active = tc.active
        # MUST-FIX 2 — never report a NESTED drill pane (drill-tab-*) as the mode's
        # current sub-tab: if the active pane isn't one of the TabbedContent's DIRECT
        # panes (a transient mid-activation read), fall back to '' rather than a
        # garbage id the context-key / primary-action gates would mis-evaluate.
        if active and active not in self._direct_pane_ids(tc):
            return ""
        return active

    # FIX 2 — the (mode, tab) surfaces where ⏎ (action_primary_action) does real
    # work.  Kept in lock-step with action_primary_action / _validate_primary:
    #   mode 0 · tab-catalog        → serve the selected slug (reconcile-gated)
    #   mode 0 · tab-orchestration  → scene switch (confirm-gated)
    #   mode 1 · ANY stage          → per-stage lane verb (fit-check … promote)
    # Doctor / Containers (mode 0) have NO ⏎ primary, so ⏎ is hidden there.
    def _primary_action_enabled(self) -> bool:
        if self._active_mode == 0:
            return self._current_subtab() in ("tab-catalog", "tab-orchestration")
        if self._active_mode == 1:
            return True
        return False

    def __init__(self, repo_root: Path, *, data: Optional[CockpitData] = None,
                 surface: str = "producer", **kwargs):
        super().__init__(**kwargs)
        self._repo_root = repo_root
        # Audience surface (2-mode merge — surface INVERSION).  The app serves BOTH
        # consumers and producers; every producer is a consumer; so BOTH modes show
        # by DEFAULT.  Internal values are kept ("producer"/"consumer") to reuse the
        # existing _PRODUCER_ONLY gate machinery, but the MEANING is inverted:
        #   "producer" = FULL  (default — Run & Operate + Bring & Validate, all verbs)
        #   "consumer" = LEAN  (the minimal rig view — hides Bring & Validate + the
        #                       producer-only verbs; opt-in via the [C] lean toggle)
        # The lean surface carries a sub-title indicator; full is the unmarked norm.
        self._surface = surface if surface in ("consumer", "producer") else "producer"
        if self._surface == "consumer":
            self.sub_title = f"{self.SUB_TITLE} · ▸ LEAN"
        # Injectable service layer — defaults to the real (live-read) impl.
        self._data: CockpitData = data or CockpitData(repo_root)
        self._active_mode = 0  # 0=Run & Operate (merged) · 1=Bring & Validate
        # Cache the last-loaded variants so detect/match + containers can match
        # running engines back to registry slugs.
        self._variants: list[VariantRow] = []
        # The slug staged for serve (selected from the catalog).
        self._staged_entry: Optional[CatalogEntry] = None
        # The live target (running engine), captured from the last estate poll,
        # used to point Doctor's profile-triage + the validation launches at the
        # currently-serving model.  None until a poll resolves a running engine.
        self._target_slug: str = ""
        self._target_model: str = ""
        self._target_url: str = ""
        # F9: the slug match GRADE ("identity" | "shape" | "") from the last
        # estate poll — a shape match must never be presented as an identity.
        self._target_confidence: str = ""
        # Phase 5: the SHARED ServingTarget OBJECT from the last estate poll —
        # held by identity so the c3t Evaluate hand-off passes the SAME dataclass
        # instance c3t speaks (design §4/§6.6), not a reconstructed copy.
        self._target_obj = None
        # #3/NH1: the Containers tab must be CALM on entry AND on [r]-refresh —
        # no forced selection / auto-load of the first row's drill detail.  This
        # flag is load-bearing: True means "settled — a RowHighlighted that
        # reaches the handler is a genuine USER cursor move → load the drill".
        # It is re-armed to False by the populate path ONLY when an [r]-refresh
        # repopulates WHILE the Containers tab is active (the one case where the
        # programmatic row-0 echo reaches the handler past its tab guard); that
        # one echo is then consumed (flag→True, no load) and subsequent real user
        # moves load again.  On tab-ENTRY the row-0 echo fires while Orchestration
        # is active and is guarded out, so no arming is needed there — the tab is
        # calm without a flag flip.  See on_data_table_row_highlighted.
        self._containers_user_navigated: bool = True
        # FIX 1 (clamp echo) — a one-shot to swallow the SECOND RowHighlighted echo
        # that a CLAMP-to-different populate emits (the move_cursor that lands on a
        # container the user never selected).  The row-0 re-arm above swallows the
        # first echo (the t.clear() reset); this swallows the follow-up move echo so
        # a periodic poll NEVER auto-loads a docker drill for an unselected
        # container.  Set True by load_estate on a clamped re-render; consumed once.
        self._containers_suppress_clamp_echo: bool = False
        # Studio guard — is the studio bundle set up (comfyui-local image built AND
        # the director GGUF on disk)?  Cached off each estate poll (load_estate).
        # Defaults True so an action before the first poll never false-blocks; the
        # first poll sets the real value.  When False, starting comfyui / switching
        # to a studio scene guides the user to setup-image-studio.sh.
        self._studio_ready: bool = True
        # Phase 5: the last BYO fit-check result (Run · BYO) — the arch facts
        # the Promote-to-catalog scaffold computes from.
        self._last_byo: Optional[ByoResult] = None
        # Phase R / R2b: failure context for the [!] problem report — captured AT
        # a failed serve in dispatch_action (the slug + the boot-log lines that
        # were streamed into the serve-live pane) so problem_report can assemble
        # a paste-ready issue with the readily-available context.
        self._problem_slug: str = ""
        self._problem_boot_log: str = ""
        # A3: monotonic timestamp of the last completed estate poll (for the
        # rail "as of <ago>" freshness stamp).  None until the first poll.
        self._last_estate_poll_mono: Optional[float] = None
        # Adaptive estate-poll state (see _docker_poll_due). _last_activity_mono seeded
        # "ancient" (0.0); on_mount stamps it to now so a fresh launch starts in steady.
        self._last_activity_mono: float = 0.0
        self._estate_tick: int = 0
        self._docker_burst_until: float = 0.0   # monotonic deadline of the post-action burst
        # A3: the periodic-refresh interval handle (created once on mount, gated
        # at fire time to Operate + the main screen so it never churns elsewhere).
        self._estate_interval = None
        # A1/A10: the serve-pending watcher.  When a serve is dispatched the boot
        # is ASYNC, so we DEFER the re-poll: poll the estate every few seconds
        # until the booted slug matches (✓) or we run out of attempts (still
        # booting).  These fields drive _resolve_pending_serve.
        self._pending_serve_slug: str = ""        # slug we're waiting to come up
        self._pending_serve_model: str = ""       # human label for the ✓ line
        self._pending_serve_port: int = 0
        self._pending_serve_timer = None          # the deferred re-poll timer
        self._pending_serve_attempts: int = 0     # polls done so far
        # MUST-FIX 1: a GENERATED/BYO serve (`docker compose -f <path> up -d`) has
        # NO registry slug — `cmd[-1]` is `-d`, not a slug.  So instead of a
        # slug-match we resolve its terminal state from a NEW container appearing
        # in the estate.  These two fields drive that container-appearance path
        # (set only for the generated lane; empty/false for a registry serve).
        self._pending_serve_generated: bool = False   # generated/BYO serve?
        # Container names present BEFORE the generated launch.  ``None`` = not yet
        # seeded → the FIRST post-launch poll establishes the true baseline (so a
        # stale/empty cached snapshot can't false-✓ a pre-existing container).
        self._pending_serve_baseline: Optional[set[str]] = None
        # A3: the last estate snapshot, cached so the periodic as-of re-render can
        # re-stamp the rail's freshness WITHOUT a fresh subprocess poll.
        self._last_estate_state: Optional[EstateState] = None
        # Last GPU/host telemetry (cached for the video⊕voice guard — read in the
        # sync container-write path without a fresh async poll).
        self._last_telemetry: Optional[EstateTelemetry] = None
        # #6/A12: the registry-derived profile-template options + whether the
        # rig-topology default has already been applied (so a later estate poll
        # re-defaults the dropdown only ONCE — never clobbering a user's pick).
        self._profile_options: list["ProfileOption"] = []
        self._profile_default_applied: bool = False
        self._profile_topology_defaulted: bool = False
        # NICE-TO-HAVE 2: once the user manually picks a profile in either Select,
        # the estate-poll rig-default reapply must not silently clobber it.
        self._profile_user_touched: bool = False
        self._last_applied_profile_default: Optional[str] = None

    # A1/A10 deferred-serve re-poll knobs.
    _SERVE_REPOLL_SECS = 3.0
    _SERVE_REPOLL_MAX_ATTEMPTS = 10               # ~30s before "still booting"

    # Adaptive estate poll (perf: the docker `ps`/`inspect` batch is the daemon-CPU sink;
    # nvidia-smi GPU reads are cheap + the only thing worth watching live). The timer fires
    # at _POLL_TICK_SECS; every tick refreshes the GPU bars (docker-free), but the heavy
    # docker+host batch runs only every Nth tick by REGIME (see _docker_poll_due):
    #   • burst  — within _DOCKER_BURST_SECS of a docker-affecting action → every tick (live)
    #   • steady — normal → _DOCKER_STEADY_STRIDE ticks (~20s)
    #   • idle   — no input for _ESTATE_IDLE_AFTER_SECS → _DOCKER_IDLE_STRIDE ticks (~30s)
    # Any input snaps idle→steady (_note_activity); a docker-affecting action snaps→burst.
    _POLL_TICK_SECS = 3.0            # base tick = the live GPU cadence
    _DOCKER_STEADY_STRIDE = 7       # 3s × 7 ≈ 21s
    _DOCKER_IDLE_STRIDE = 10        # 3s × 10 = 30s
    _ESTATE_IDLE_AFTER_SECS = 90.0  # no input this long → idle regime
    _DOCKER_BURST_SECS = 90.0       # fast-docker window after a docker-affecting action
    # ActionPlan kinds that change what's running (→ trigger the burst). NOT set_default /
    # clear_default / download / power_cap (those don't start/stop containers).
    _DOCKER_BURST_KINDS = frozenset(
        {"serve", "scene", "estate_down", "container", "container_rm", "service-up"}
    )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-layout"):
            with Vertical(id="left-rail"):
                yield ModeSwitcher(id="mode-switcher", surface=self._surface)
                yield RailStatus(id="rail-status")
                # FIX 3 — host disk + RAM usage lives in the LEFT RAIL (the "estate
                # column"), BELOW RailStatus — not inside the Orchestration sub-tab.
                yield HostStatsRail(id="host-stats-rail")
            with Container(id="content-area"):
                # Mode 0 — Run & Operate (MERGED).  ONE TabbedContent (#operate-tabs)
                # hosts the consumer's whole rig surface in four tabs, in order:
                #   Catalog       — discover + serve (was Run · Catalog)
                #   Orchestration — live estate / scene switch (was Operate)
                #   Containers    — docker drill (was Operate)
                #   Doctor        — health + diagnose reads (was Operate)
                # Serving from Catalog streams into the re-homed LivePane below the
                # tabs, and the user flips to the Orchestration tab IN THE SAME MODE
                # to watch the estate.  (The panel keeps the id #panel-run — it is
                # already the first + active panel and already hosts #serve-live —
                # and the TabbedContent keeps the id #operate-tabs so the ~30
                # Operate write-gates / focus-map entries keep resolving.)
                with Container(id="panel-run", classes="mode-panel active"):
                    with TabbedContent(id="operate-tabs"):
                        with TabPane("Catalog", id="tab-catalog"):
                            yield CatalogPane(id="catalog-pane")
                        with TabPane("Orchestration", id="tab-orchestration"):
                            yield OperateOrchPane(id="operate-orch-pane")
                        with TabPane("Containers", id="tab-containers"):
                            yield OperateContainersPane(id="operate-containers-pane")
                        with TabPane("Doctor", id="tab-doctor"):
                            yield DoctorPane(id="doctor-pane")
                    # Transient boot-output pane — re-homed here so a serve staged
                    # from the Catalog tab streams its boot log below the tabs, and
                    # the user can flip to Orchestration (SAME mode) to watch.
                    # Hidden until ⏎ on a Catalog row stages a serve and the
                    # reconcile-gated confirm commits.
                    # Hidden until a serve streams — no idle placeholder.
                    yield LivePane(id="serve-live", placeholder="")

                # Mode 1 — Bring & Validate (producer lane).  Renumbered 2→1 in the
                # 2-mode merge.  An ORDERED, numbered pipeline reusing the
                # TabbedContent pattern:
                #   ① Bring  → LaneBringPane (reuses byo_check fit-check)
                #   ② Serve  → LaneServePane (generate compose + serve untested)
                #   ③ Gate   → ValidateRunPane (the existing 9-step ladder)
                #   ④ Measure→ ValidateEvidencePane (the existing evidence list)
                #   ⑤ Promote→ LanePromotePane (hosts the [P] promote action)
                # Tab IDs encode the ordinal so the focus map / sub-tab cycle read
                # the stages in pipeline order.
                with Container(id="panel-validate", classes="mode-panel"):
                    with TabbedContent(id="validate-tabs"):
                        with TabPane("① Bring", id="tab-bring"):
                            yield LaneBringPane(id="lane-bring-pane")
                        with TabPane("② Serve", id="tab-serve"):
                            yield LaneServePane(id="lane-serve-pane")
                        with TabPane("③ Gate", id="tab-run"):
                            yield ValidateRunPane(id="validate-run-pane")
                        with TabPane("④ Measure", id="tab-evidence"):
                            yield ValidateEvidencePane(id="validate-evidence-pane")
                        with TabPane("⑤ Promote", id="tab-promote"):
                            yield LanePromotePane(id="lane-promote-pane")
        # #5 — a Tab-traversable footer so keyboard users can reach the footer
        # affordances (in addition to the hotkeys).
        yield FocusableFooter()

    # ── Mount / startup ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        import time as _t
        self._note_activity()                              # fresh launch = user present
        self._docker_burst_until = _t.monotonic() + 15.0   # 15s startup burst → immediate + live
        self.load_catalog()
        # A3: ONE periodic refresh interval, created once.  It is GATED at fire
        # time (_periodic_estate_refresh) to the MERGED Run & Operate mode
        # (_active_mode == 0 — the live estate tabs + host-stats rail + catalog
        # live-VRAM all live here now) AND the main screen (no modal) — so it never
        # churns subprocesses behind the Bring & Validate lane or a confirm modal.
        # set_interval is the only timer.
        self._estate_interval = self.set_interval(
            self._POLL_TICK_SECS, self._periodic_estate_refresh, pause=False
        )
        # A3: a lightweight as-of re-stamp timer.  The full poll (above) resets
        # the freshness clock every tick → the rail would always read "just now".
        # This SEPARATE, more-frequent timer re-renders ONLY the rail's as-of line
        # from the CACHED last EstateState (a pure read — NO subprocess), so the
        # "Ns/Nm ago" branches actually surface between/behind polls (e.g. when a
        # poll is skipped behind a modal).  Same merged-mode gating as the poll.
        self._asof_interval = self.set_interval(
            1.0, self._refresh_rail_as_of, pause=False
        )
        # FIX 1 — sync the base-footer suppression to the initial (modal-free)
        # stack so it starts visible.
        self._sync_base_footer_visibility()
        # MUST-FIX 1 — boot with focus on mode 0's primary list (#catalog-table),
        # NOT the tab bar.  Textual auto-focuses the ContentTabs (a Tabs subclass)
        # at startup BEFORE the startup tab-catalog TabActivated fires, and FIX B's
        # ``isinstance(self.focused, Tabs)`` guard then short-circuits the auto
        # focus-into-the-list — so without this the app would boot focused on the
        # tab bar.  ``_focus_mode_primary`` focuses the table DIRECTLY (it doesn't
        # route through on_tabbed_content_tab_activated), so the guard never sees it;
        # call_after_refresh ensures it lands AFTER the startup TabActivated.
        self._focus_mode_primary(0)

    # ── FIX 1 · base-footer suppression under a modal ────────────────────────────
    #
    # The main screen docks ONE FocusableFooter (compose()).  When a ModalScreen is
    # topmost, that base footer still renders under/around the modal advertising
    # global keys + "Enter Select" — misleading, because ONLY the modal's own
    # bindings are live.  (A modal either yields its own Footer — only
    # ConfirmActionScreen does — or advertises its keys in-body via button labels;
    # either way the BASE footer's hints are stale.)  We suppress the base
    # footer whenever the screen stack is deeper than the main screen, and restore
    # it when the last modal pops.  ``screen_stack`` mutates SYNCHRONOUSLY in
    # push_screen / pop_screen (verified), so toggling right after super() is
    # correct; ``dismiss`` also routes through App.pop_screen, so it's covered too.

    def _sync_base_footer_visibility(self) -> None:
        """Show the base FocusableFooter ONLY when no modal is topmost.

        Toggles the ``base-footer-hidden`` class (CSS ``display:none``) so that a
        modal's own footer is the single source of footer truth while it is up,
        and the base footer comes back the instant the stack returns to the main
        screen.  Never strands focus: the suppressed footer is a docked,
        non-focused widget — modals manage their own focus, and on restore the
        main screen's existing focus is untouched."""
        try:
            footer = self.query_one(FocusableFooter)
        except Exception:
            return
        modal_topmost = len(self.screen_stack) > 1
        footer.set_class(modal_topmost, "base-footer-hidden")

    def push_screen(self, *args, **kwargs):
        result = super().push_screen(*args, **kwargs)
        self._sync_base_footer_visibility()
        return result

    def pop_screen(self, *args, **kwargs):
        result = super().pop_screen(*args, **kwargs)
        self._sync_base_footer_visibility()
        return result

    def _refresh_rail_as_of(self) -> None:
        """A3: re-stamp the rail's freshness line from CACHED state — no poll.

        Gated to the merged Run & Operate mode + the main screen (same as the
        periodic poll) and a PURE read of self._last_estate_state, so it never
        spawns a subprocess and never runs outside the live panel.  This is what
        makes _estate_as_of()'s "Ns/Nm ago" branches reachable (the full poll
        resets the clock every tick, so on its own the rail would read 'just now'
        forever)."""
        if self._active_mode != 0:
            return
        try:
            if len(self.screen_stack) > 1:
                return
        except Exception:
            pass
        state = self._last_estate_state
        if state is None:
            return
        try:
            self.query_one("#rail-status", RailStatus).update_from_state(
                state, as_of=self._estate_as_of()
            )
        except Exception:
            pass

    def _periodic_estate_refresh(self) -> None:
        """A3: the gated periodic poll.  Fires ONLY while the user is in the merged
        Run & Operate mode and looking at the main screen — otherwise it's a no-op
        (no read churn behind the Bring & Validate lane or a modal).  The live
        estate tabs + host-stats rail + catalog live-VRAM all live in this mode now,
        so firing across the whole merged mode keeps them all fresh."""
        if self._active_mode != 0:
            return
        # Don't poll behind a modal (confirm gate / help / explain) — the live
        # panel isn't visible and a re-poll could race the gate's own reads.
        try:
            if len(self.screen_stack) > 1:
                return
        except Exception:
            pass
        self._estate_tick += 1
        # GPU bars refresh EVERY tick (cheap, docker-free) → live at _POLL_TICK_SECS.
        self._refresh_gpu_bars()
        # The heavy docker+host batch runs only on a due tick (burst/steady/idle).
        if self._docker_poll_due():
            self.load_estate()

    def _note_activity(self) -> None:
        """Reset the idle clock — any key/mouse/action keeps the docker poll out of the
        idle regime (snaps idle→steady); it backs off when the cockpit is left idle."""
        import time as _t
        self._last_activity_mono = _t.monotonic()

    @work(exclusive=True, group="gpu-rail")
    async def _refresh_gpu_bars(self) -> None:
        """Fast, docker-FREE GPU refresh (nvidia-smi only) — runs every tick so the GPU
        cards stay live between the slower docker+host polls. The docker-dependent
        'held by: <container>' attribution still rides the slow batch."""
        try:
            gpus = await self._data.gpu_info()
        except Exception:
            return
        if not gpus:
            return
        try:
            self.query_one("#operate-orch-pane", OperateOrchPane).refresh_gpu_cards(gpus)
        except Exception:
            pass
        # F3b: readiness fast-flip. api_booting only clears when the HEAVY
        # docker+health batch re-runs — so "⏳ booting" outlived actual readiness
        # by a poll cycle (audit: ≥25s stale). While booting, piggyback a CHEAP
        # direct /v1/models probe on this fast tick; on 200, pull the next heavy
        # poll forward (burst regime) so every "booting" surface flips promptly.
        # Bounded: fires ONLY in the booting state, 1.5s timeout, best-effort.
        state = getattr(self, "_last_estate_state", None)
        if state is not None and getattr(state.doctor, "booting", False):
            url = (getattr(state.target, "url", "") or "").strip()
            if not url:
                port = getattr(state.target, "host_port", 0) or 0
                url = f"http://localhost:{port}" if port else ""
            if url:
                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=1.5) as client:
                        r = await client.get(f"{url.rstrip('/')}/v1/models")
                    if r.status_code == 200:
                        import time as _t
                        self._docker_burst_until = _t.monotonic() + 6.0
                except Exception:
                    pass

    def _docker_poll_due(self) -> bool:
        """Whether THIS tick runs the heavy docker+host batch, by regime: burst (within
        _DOCKER_BURST_SECS of a docker-affecting action) → every tick; idle (no input for
        _ESTATE_IDLE_AFTER_SECS) → _DOCKER_IDLE_STRIDE; else steady → _DOCKER_STEADY_STRIDE.
        Pure (reads _estate_tick, no I/O) → unit-testable."""
        import time as _t
        now = _t.monotonic()
        if now < self._docker_burst_until:
            return True
        stride = (
            self._DOCKER_IDLE_STRIDE
            if (now - self._last_activity_mono) > self._ESTATE_IDLE_AFTER_SECS
            else self._DOCKER_STEADY_STRIDE
        )
        return (self._estate_tick % stride) == 0

    def _estate_as_of(self) -> str:
        """A3: a human "N s/m ago" string for the last estate poll, or "just now".
        Empty when no poll has completed yet (rail shows its placeholder)."""
        import time as _time
        t = getattr(self, "_last_estate_poll_mono", None)
        if t is None:
            return "just now"
        delta = max(0.0, _time.monotonic() - t)
        if delta < 5:
            return "just now"
        if delta < 90:
            return f"{int(delta)}s ago"
        return f"{int(delta // 60)}m ago"

    # ── Catalog loading ──────────────────────────────────────────────────────────────

    @work(exclusive=True, group="catalog")
    async def load_catalog(self) -> None:
        """Load the catalog (real read): paint the registry rows immediately,
        then enrich fit + TPS in the background so the table appears in ~1s
        instead of blocking on the full enrichment."""
        rows, error = await self._data.load_catalog_rows()
        if not error and not rows:
            error = "No variants returned — registry may be empty"
        # Cache variants for detect/match + container slug-matching.
        self._variants = [e.row for e in rows]
        # #6/A12 — derive the profile-template dropdown options from the variants
        # now that the registry has loaded (default applied with whatever GPU count
        # is known so far; the first estate poll re-defaults to the rig topology).
        if self._variants:
            self._refresh_profile_templates()
        try:
            pane = self.query_one("#catalog-pane", CatalogPane)
        except Exception:
            return
        pane.populate(rows, error)          # instant first paint (stub fit/TPS)
        if error or not rows:
            return
        # Progressive enrichment — re-render after each phase (cursor preserved).
        # rows are the SAME CatalogEntry objects the pane holds, so in-place
        # mutation of e.fit / e.measurement is visible to refresh_enriched().
        await self._data.enrich_fits(rows)
        pane.refresh_enriched()
        await self._data.enrich_measurements(rows)
        pane.refresh_enriched()
        # Download UX — which slugs' weights are on disk (one weights.py call +
        # a stat per row).  Last enrichment phase; feeds the Download-vs-Start
        # control + the listing's download glyph.
        await self._data.enrich_weights(rows)
        # In-flight downloads must survive a refresh: enrich_weights just re-stat'd
        # the rebuilt rows (a downloading slug would read partial/absent), so
        # re-apply the live DOWNLOADING state from the slug-keyed tracker.
        self._reapply_active_downloads(rows)
        pane.refresh_enriched()
        # Prompt to set the model dir if it's missing (weights-state can't be
        # trusted otherwise) — a banner on the catalog status line ([S] to set).
        from pathlib import Path as _Path
        mdir = self._data.weights_model_dir()
        note = "" if _Path(mdir).is_dir() else f"⚠ model dir not found ({mdir}) — press [S] to set it"
        pane.set_model_dir_note(note)

    # ── Download UX · fetch a slug's weights (setup.sh WEIGHT_KEY=…, streamed) ────────

    def _active_downloads(self) -> dict:
        """slug → {entry, meta, pct} for in-flight downloads (lazy).  Slug-keyed so
        the in-flight state SURVIVES a catalog refresh that rebuilds the entry
        objects; ``pct`` is the live aggregate %, ``entry`` is re-pointed to the
        rendered entry by _reapply_active_downloads on each reload."""
        d = getattr(self, "_downloads", None)
        if d is None:
            d = {}
            self._downloads = d
        return d

    def _reapply_active_downloads(self, rows: "list[CatalogEntry]") -> None:
        """Re-apply in-flight download state onto freshly-rebuilt catalog rows.

        A refresh (``r`` → load_catalog) builds NEW CatalogEntry objects, which
        would orphan a running download — its DOWNLOADING state + ⏳% live on the
        OLD entry, so the rebuilt row re-stats to partial/absent (⏳ vanishes) and
        the pop-up shows 'starting…' (the new entry's download_pct is None) while
        the worker keeps mutating the orphan.  ``_active_downloads`` is slug-keyed
        and survives the rebuild, so re-stamp the new entries from it and re-point
        the tracker at the rendered entry (run_download reads ``info['entry']``
        each tick → its next update lands on THIS one, and the % keeps moving)."""
        dls = self._active_downloads()
        if not dls:
            return
        by_slug = {e.slug: e for e in rows}
        for slug, info in dls.items():
            e = by_slug.get(slug)
            if e is None:
                continue
            e.weights_state = WEIGHTS_DOWNLOADING
            e.download_pct = info.get("pct")
            info["entry"] = e

    def _refresh_catalog_rows_safe(self) -> None:
        try:
            self.query_one("#catalog-pane", CatalogPane).refresh_enriched()
        except Exception:
            pass

    def start_download(self, entry: CatalogEntry) -> None:
        """Begin downloading ``entry``'s weights.  Disk pre-check + manual-only
        gate, then launch the streamed worker; the listing shows ⏳NN% live.
        No-op if already downloading this slug."""
        slug = entry.slug
        dls = self._active_downloads()
        if slug in dls:
            return
        meta = entry.weights
        if meta is None or not meta.hf_repo:
            self.notify(
                "No direct download recipe for this variant (manual weights).",
                title="Download", severity="warning", timeout=4,
            )
            return
        fits, free_gb, need_gb = self._data.weights_fits_disk(meta)
        if not fits:
            self.notify(
                f"Not enough disk for {slug}: need ~{need_gb:.0f} GB, {free_gb:.0f} GB free "
                "on the model dir.",
                title="Download", severity="error", timeout=6,
            )
            return
        entry.weights_state = WEIGHTS_DOWNLOADING
        entry.download_pct = 0
        dls[slug] = {"entry": entry, "meta": meta, "pct": 0}
        self._refresh_catalog_rows_safe()
        size = f" (~{meta.size_gb:.0f} GB)" if meta.size_gb else ""
        self.notify(f"Downloading {meta.hf_repo}{size}…", title="Download", timeout=4)
        self.run_download(entry)

    @work(group="weights-download", exclusive=False)
    async def run_download(self, entry: CatalogEntry) -> None:
        """Stream the download; poll bytes-on-disk for ⏳NN% until the process
        completes; then re-stat to confirm 'present' (authoritative over rc)."""
        import asyncio
        slug = entry.slug
        meta = entry.weights
        try:
            run = await self._data.run_weights_download(
                entry.model, entry.weights_variant, companions=entry.weights_companions
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._active_downloads().pop(slug, None)
            entry.weights_state = WEIGHTS_ABSENT
            entry.download_pct = None
            self._refresh_catalog_rows_safe()
            self.notify(f"Download failed to start: {exc}", title="Download",
                        severity="error", timeout=6)
            return
        # Aggregate progress over the WHOLE download set (core + companions) so the
        # ⏳NN% MOVES even when the core is already on disk and only a companion is
        # being fetched — a core-only signal would sit at a static ~99% off the
        # core subdir while the companion lands in its own subdir uncounted.
        try:
            _index = await self._data.weights_index()
            dl_metas = self._data.download_set_metas(entry, _index)
        except Exception:
            dl_metas = [meta] if meta is not None else []
        # Progress loop: refresh ⏳NN% every ~2s until the run signals done.  Read
        # the CURRENT entry from the tracker each tick — a catalog refresh rebuilds
        # entries and _reapply_active_downloads re-points info['entry'] at the
        # rendered one, so updates always land on what's on screen (not the orphan).
        while not run.done.is_set():
            info = self._active_downloads().get(slug)
            if info is None:
                return  # cancelled — cancel_download already reset + re-stat'd
            if dl_metas:
                pct = self._data.weights_download_progress_set(dl_metas)
                if pct is not None:
                    info["pct"] = pct                        # slug-keyed → survives refresh
                    cur = info.get("entry", entry)
                    if cur.download_pct != pct:
                        cur.download_pct = pct
                        self._refresh_catalog_rows_safe()
            try:
                await asyncio.wait_for(run.done.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
        info = self._active_downloads().get(slug)
        if info is None:
            return  # cancelled at the tail
        # Done — re-stat (verify_glob) is authoritative over the exit code.  Re-stat
        # the CURRENT (rendered) entry, not the possibly-orphaned original.
        cur = info.get("entry", entry)
        self._active_downloads().pop(slug, None)
        cur.download_pct = None
        await self._data.enrich_weights([cur])
        self._refresh_catalog_rows_safe()
        if cur.weights_state == WEIGHTS_PRESENT:
            self.notify(f"✓ {slug} weights downloaded.", title="Download", timeout=4)
        else:
            self.notify(
                f"✗ {slug} download incomplete ({cur.weights_state}) — "
                "check disk / HF_TOKEN, then retry.",
                title="Download", severity="warning", timeout=7,
            )

    def cancel_download(self, slug: str) -> None:
        """Cancel an in-flight download: stop the poll worker, kill the process
        group, reset + re-stat the entry."""
        info = self._active_downloads().pop(slug, None)
        if info is None:
            return
        try:
            self.workers.cancel_group(self, "weights-download")
        except Exception:
            pass
        self._kill_download()
        entry = info["entry"]
        entry.download_pct = None
        self._restat_after_cancel(entry)
        self.notify(f"Cancelled download of {slug}.", title="Download", timeout=3)

    @work(group="weights-download-ctl")
    async def _kill_download(self) -> None:
        await self._data.cancel_weights_download()

    @work(group="weights-download-restat")
    async def _restat_after_cancel(self, entry: CatalogEntry) -> None:
        await self._data.enrich_weights([entry])   # → partial / absent
        entry.download_pct = None
        self._refresh_catalog_rows_safe()

    # ── #6/A12 · profile-template dropdown ───────────────────────────────────────────

    def _known_profile_likes(self) -> list[str]:
        """A12 — the known profile-like slugs (every loaded registry variant slug),
        for the unknown-profile guard + its "known: …" hint.  Pure (cached
        variants); empty before the registry loads."""
        out: list[str] = []
        seen: set[str] = set()
        for row in (self._variants or []):
            slug = (getattr(row, "slug", "") or "").strip()
            if slug and slug not in seen:
                seen.add(slug)
                out.append(slug)
        return out

    def _known_gpu_count(self) -> Optional[int]:
        """Best-known live GPU count (from the last estate poll), or None when the
        estate hasn't been polled yet (the dropdown default then degrades to the
        first matching/`dual` template)."""
        st = self._last_estate_state
        if st is not None and getattr(st, "gpus", None):
            return len(st.gpus)
        return None

    def _refresh_profile_templates(self, *, reapply_default: bool = False) -> None:
        """#6/A12 — (re)derive the profile-template options from the loaded variants
        and push them into the producer lane's ① Bring profile-template Select,
        defaulting to the rig's own topology.  (The standalone Run · BYO pane was
        removed in the 2-mode merge — ① Bring is the single bring-a-repo entry.)

        The default is applied ONCE (first time options become available, or when
        ``reapply_default`` forces a re-default after the estate poll first learns
        the GPU count) so a later poll never clobbers a user's manual pick.  Pure —
        no I/O (reads cached variants + cached estate gpu-count)."""
        # FIX 2 — pass the registry's curated `defaults` array (surfaced by the
        # catalog load) so each (family, topology) representative is the
        # registry's own recommendation when available, never a status-blind
        # last-in-insertion-order slug.  Empty on the raw-tab fallback load path
        # → profile_templates degrades to the status floor.
        defaults = list(getattr(self._data, "catalog_defaults", None) or [])
        options = profile_templates(self._variants or [], defaults)
        if not options:
            return
        self._profile_options = options
        # NICE-TO-HAVE 2 — once a user manually picks a profile (Run/BYO or ①
        # Bring), the rig-default reapply (estate-poll re-default) must NOT
        # clobber it.  Skip the re-default when the user has touched the Select.
        apply_default = reapply_default or not self._profile_default_applied
        if apply_default and reapply_default and getattr(self, "_profile_user_touched", False):
            apply_default = False
        select_opts = profile_select_options(options)
        default = (
            default_profile_template(options, self._known_gpu_count() or 2)
            if apply_default
            else None
        )
        if apply_default:
            self._last_applied_profile_default = default
        try:
            self.query_one("#lane-bring-pane", LaneBringPane).set_profile_options(
                select_opts, default
            )
        except Exception:
            pass
        if apply_default:
            self._profile_default_applied = True

    # ── Estate polling ───────────────────────────────────────────────────────────────

    def _catalog_entry_for(self, slug: str) -> Optional[CatalogEntry]:
        """A7: the enriched CatalogEntry for a slug (for its numeric fit.max_ctx
        claim), or None.  Reads the already-loaded catalog pane entries."""
        try:
            entries = self.query_one("#catalog-pane", CatalogPane)._entries
        except Exception:
            return None
        for e in entries:
            if getattr(e, "slug", "") == slug:
                return e
        return None

    @staticmethod
    def _live_free_gb_by_index(state: EstateState) -> Optional[dict[int, float]]:
        """A6: derive live per-GPU FREE VRAM (GB) from the estate poll's GpuInfo.

        free = (mem_total_mib − mem_used_mib) / 1024.  Returns None when no GPU
        was read (nvidia-smi gave nothing) — the honest "unknown" signal so the
        catalog labels the fit column "vs empty card" rather than fabricating a
        free figure."""
        gpus = getattr(state, "gpus", None) or []
        out: dict[int, float] = {}
        for g in gpus:
            idx = getattr(g, "index", None)
            total = getattr(g, "mem_total_mib", 0) or 0
            used = getattr(g, "mem_used_mib", 0) or 0
            if idx is None or total <= 0:
                continue
            out[int(idx)] = max(0.0, (total - used) / 1024.0)
        return out or None

    @work(exclusive=True, group="estate")
    async def load_estate(self, *, explicit_refresh: bool = False) -> None:
        """Poll the live estate snapshot + push into the orch/doctor panes + rail.

        Also captures the live target (matched slug / model / url) so Doctor's
        profile-triage and the validation launches point at the running model,
        and reads the power-cap status (a safe READ) for the orch pane.

        ``explicit_refresh`` is True only for the [r]-driven action_refresh (a
        deliberate user re-read) — NOT the 4s periodic Operate tick.  FIX 1 uses it
        to decide whether to re-arm the containers row-0 suppression + cancel a
        pending drill timer even when the container set was unchanged: an explicit
        [r] should kill a stale drill timer (the [r]-re-jump footgun), but the
        periodic tick must leave the user's selection + pending drill alone."""
        state = await self._data.estate_state(variants=self._variants or None)
        # A3: stamp the poll time so the rail can render "as of <ago>".
        import time as _time
        self._last_estate_poll_mono = _time.monotonic()
        # A3: cache the snapshot so the periodic as-of re-render can re-stamp the
        # rail's freshness from CACHED state (a pure read, no subprocess).  Also
        # feeds the generated-serve container-appearance baseline (MUST-FIX 1).
        self._last_estate_state = state
        # Studio guard — refresh the studio-ready flag each poll (comfyui-local
        # image + director GGUF), so the user can run setup-image-studio.sh while
        # the cockpit is open and have the guard clear within a tick.
        try:
            self._studio_ready = await self._data.studio_ready()
        except Exception:
            pass
        # #6/A12 — the first estate poll that learns the real GPU count re-defaults
        # the profile-template dropdown to the rig topology (1 card → single, ≥2 →
        # dual).  reapply_default=True forces the re-default exactly once; a later
        # poll won't fire it again (the flag below), so a user's manual pick stands.
        if not getattr(self, "_profile_topology_defaulted", False) and getattr(state, "gpus", None):
            self._profile_topology_defaulted = True
            try:
                self._refresh_profile_templates(reapply_default=True)
            except Exception:
                pass
        # Capture the live target for profile-triage / validation launches.
        self._target_slug = state.matched_slug or ""
        tgt = state.target
        self._target_model = getattr(tgt, "model", "") or ""
        self._target_url = getattr(tgt, "url", "") or ""
        self._target_confidence = getattr(tgt, "match_confidence", "") or ""
        # Hold the SHARED ServingTarget object (by identity) for the c3t Evaluate
        # hand-off — design §4/§6.6 requires passing the SAME dataclass instance.
        self._target_obj = tgt
        # A7: look up the matched slug's CONFIGURED ctx so the serving panel can
        # badge a divergence between the probed running ctx and the catalog slug's
        # CONFIGURED ctx.  Both a display label (ctx_label, e.g. "262K") and the
        # NUMERIC configured ctx (the registry max_ctx int, e.g. 262144) are passed
        # — the numeric one drives the divergence comparison so a colloquial label
        # ("262K" for 262144) doesn't trip a false divergence.
        #
        # MUST-FIX 2: this is the slug's CONFIGURED ctx (registry max_ctx), NOT the
        # kv-calc CAPACITY ceiling (fit.max_ctx, e.g. ~295K for a 262K-configured
        # qwen).  Comparing the probe against the capacity ceiling false-fired the
        # badge on an honest 262144 serve (|295000−262144|>1024).  "" / None when
        # unmatched or the registry row didn't carry the int.
        catalog_ctx_label = ""
        catalog_ctx = None
        if self._target_slug:
            mrow = next(
                (v for v in (self._variants or []) if getattr(v, "slug", "") == self._target_slug),
                None,
            )
            if mrow is not None:
                catalog_ctx_label = getattr(mrow, "ctx_label", "") or ""
                catalog_ctx = getattr(mrow, "configured_ctx", None)
            # The enriched catalog entry is the authoritative source for the
            # configured int (it carries the registry row); prefer it when present.
            mentry = self._catalog_entry_for(self._target_slug)
            if mentry is not None:
                ent_cfg = mentry.configured_ctx
                if ent_cfg is not None:
                    catalog_ctx = ent_cfg
                if not catalog_ctx_label:
                    catalog_ctx_label = mentry.ctx_label or ""
        try:
            self.query_one("#operate-orch-pane", OperateOrchPane).populate(
                state, catalog_ctx_label=catalog_ctx_label, catalog_ctx=catalog_ctx
            )
        except Exception:
            pass
        try:
            containers_pane = self.query_one(
                "#operate-containers-pane", OperateContainersPane
            )
            rerendered = containers_pane.populate(
                state.containers, getattr(state, "error", "") or ""
            )
            # FIX 1 (clamp echo) — did this re-render CLAMP the cursor to a
            # DIFFERENT container (the user's selection vanished)?  The benign
            # case (selection PRESERVED, index merely shifted) re-loads the SAME
            # container and is harmless; the CLAMP case would auto-load a docker
            # drill for a container the user never picked.
            clamped_to_other = bool(getattr(containers_pane, "last_populate_clamped", False))
            # #3/NH1: a (re)populate clears the table and resets the cursor to
            # row 0, firing a PROGRAMMATIC RowHighlighted.  That echo only REACHES
            # on_data_table_row_highlighted (past its tab guard) when the populate
            # ran while the Containers tab was already active — i.e. an [r]-refresh
            # ON the tab.  Re-arm the suppression in exactly that case so the echo
            # does NOT auto-load row 0 (the [r]-re-jump footgun); a later real user
            # arrow-move re-sets the flag and DOES load.  (On tab-ENTRY the echo
            # fires while Orchestration is active → guarded out → no arming needed,
            # and the tab-focus handler re-arms separately for the calm-entry case.)
            # FIX 1: the row-0 suppression + drill-timer cancel are split now.
            #  • Re-arm the row-0 suppression (flag→False) ONLY when the table was
            #    actually RE-RENDERED — that re-render resets the cursor to row 0 and
            #    fires the PROGRAMMATIC echo that consumes the False flag.  Doing it
            #    on a SKIPPED poll (cursor preserved, no echo) would wrongly swallow
            #    the user's NEXT genuine move (nothing consumes the flag).
            #  • Cancel a stale drill timer on a re-render OR an explicit [r]-refresh
            #    (kills a pending drill from a prior move — the [r]-re-jump footgun),
            #    but NOT on an unchanged periodic tick (that must leave the user's
            #    in-flight drill alone — the whole point of preserving the cursor).
            on_containers = self._active_operate_tab() == "tab-containers"
            if rerendered and on_containers:
                self._containers_user_navigated = False
                # FIX 1 (clamp echo) — when the cursor was CLAMPED onto a DIFFERENT
                # container, the re-arm above swallows the t.clear() row-0 echo, but
                # the follow-up move_cursor echo would still arrive with the flag
                # already True and auto-load a drill the user never asked for.  Arm
                # the one-shot so that SECOND echo is swallowed too — the net
                # invariant: a periodic poll never starts a docker logs/top for a
                # container the user didn't actively select.
                self._containers_suppress_clamp_echo = clamped_to_other
            if (rerendered or explicit_refresh) and on_containers:
                timer = getattr(self, "_drill_timer", None)
                if timer is not None:
                    try:
                        timer.stop()
                    except Exception:
                        pass
                    self._drill_timer = None
        except Exception:
            pass
        try:
            self.query_one("#doctor-pane", DoctorPane).populate(state)
        except Exception:
            pass
        try:
            self.query_one("#rail-status", RailStatus).update_from_state(
                state, as_of=self._estate_as_of()
            )
        except Exception:
            pass
        # N3: tell the Run Catalog which slug is live-serving so the running row
        # is marked.  Driven by the SAME matched_slug the estate poll captured
        # (cleared to "" when nothing serves) — refreshed every load_estate, so
        # Operate entry / the periodic refresh / the post-serve re-poll all keep
        # the Run marker fresh.
        try:
            cat = self.query_one("#catalog-pane", CatalogPane)
            cat.set_serving_slug(self._target_slug, self._target_confidence)
            # A6: feed the Run catalog the LIVE per-GPU free-VRAM so a
            # "fits-clean" row that would OOM right now (e.g. GPU0 holding
            # ComfyUI) is downgraded.  Derived from THIS poll's GpuInfo
            # (free = total - used); empty/none when nvidia-smi gave nothing
            # (→ the column honestly reads "vs empty card").
            cat.set_live_free_vram(self._live_free_gb_by_index(state))
        except Exception:
            pass
        # A1/A10: if a serve is awaiting confirmation that the booted model came
        # up, resolve the serve LivePane terminal state from THIS poll's match.
        self._resolve_pending_serve(state)
        # Power-cap status (READ) for the orch pane.
        st = await self._data.power_cap_get()
        try:
            self.query_one("#operate-orch-pane", OperateOrchPane).populate_power_cap(st)
        except Exception:
            pass
        # Batch 5 (#12 / N5 / attribution): host telemetry (disk bars + RAM line +
        # GPU-VRAM → container owners).  Read AFTER the power-cap read so the final
        # GPU-card paint carries BOTH the cap note and the "held by:" line (both
        # re-render the cards; telemetry runs last so it has the cap map too).  All
        # reads are batched in ONE estate_telemetry() call — no per-tick storm.
        # Confined to the merged Run & Operate mode (mode 0, same guard as the
        # adjacent load_doctor read): the full host-telemetry battery (df, meminfo,
        # 2× nvidia-smi, docker ps, per-pid cgroup cats) renders to the
        # #operate-orch-pane + #host-stats-rail, both of which are hidden behind the
        # Bring & Validate lane (mode 1) — no point firing it there.
        if self._active_mode == 0:
            try:
                tel = await self._data.estate_telemetry()
                self._last_telemetry = tel   # cache for the video⊕voice start guard
                # FIX 3 — disk + RAM bars render into the LEFT RAIL (estate column),
                # NOT the Orchestration sub-tab.  The GPU-VRAM "held by:" attribution
                # stays on the orch GPU cards (populate_telemetry re-renders those).
                self.query_one("#host-stats-rail", HostStatsRail).populate_telemetry(tel)
                self.query_one("#operate-orch-pane", OperateOrchPane).populate_telemetry(tel)
            except Exception:
                pass
        # Doctor lives in the merged Run & Operate mode (the Doctor tab) and its
        # profile-triage consumes the _target_slug/_target_url THIS poll just
        # captured — so chain the doctor read here rather than racing it as a
        # sibling worker off the mode switch (which read _target_slug before this
        # wrote it → empty profile card on first entry). Guarded to mode 0; this
        # also means action_refresh ([r], which re-runs load_estate) refreshes the
        # Doctor cards too.
        if self._active_mode == 0:
            self.load_doctor()

    # ── Validate-mode loaders ──────────────────────────────────────────────────────

    @work(exclusive=True, group="doctor")
    async def load_doctor(self) -> None:
        """Refresh the live Doctor health card (health.sh — READ-only).  The
        heavier verify / verify-full reads are on-demand ([v] / [V]); the
        estate + profile triage moved to the producer Bring & Validate lane."""
        dr = await self._data.doctor_read(url=self._target_url or None)
        try:
            self.query_one("#doctor-pane", DoctorPane).populate_health(dr)
        except Exception:
            pass

    def action_doctor_verify(self) -> None:
        """[v] on Operate · Doctor: send a test query to the serving model
        (verify.sh) and render the per-check verdict.  READ-only (no GPU claim);
        no confirm gate — like [y] doctor_rerun, it only reads."""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-doctor":
            return
        try:
            self.query_one("#doctor-pane", DoctorPane).set_verify_running()
        except Exception:
            pass
        self.run_doctor_verify()
        self.notify(
            "Sending a test query to the serving model (verify.sh)…",
            title="Verify", severity="information", timeout=3,
        )

    @work(exclusive=True, group="doctor-verify")
    async def run_doctor_verify(self) -> None:
        vs = await self._data.verify_smoke()
        try:
            self.query_one("#doctor-pane", DoctorPane).populate_verify(vs)
        except Exception:
            pass

    def action_doctor_verify_full(self) -> None:
        """[V] on Operate · Doctor: run the ~1-2 min functional battery
        (verify-full.sh) and render the per-step verdict.  READ-only."""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-doctor":
            return
        try:
            self.query_one("#doctor-pane", DoctorPane).set_verify_full_running()
        except Exception:
            pass
        self.run_doctor_verify_full()
        self.notify(
            "Running the functional battery (verify-full.sh · ~1-2 min)…",
            title="Verify-full", severity="information", timeout=4,
        )

    @work(exclusive=True, group="doctor-verify-full")
    async def run_doctor_verify_full(self) -> None:
        vf = await self._data.verify_full()
        try:
            self.query_one("#doctor-pane", DoctorPane).populate_verify_full(vf)
        except Exception:
            pass

    @work(group="benchmarks")
    async def load_cross_rig_for_explain(
        self, screen: ExplainScreen, model: str, engine: str
    ) -> None:
        """Load cross-rig benchmark rows (corpus → BENCHMARKS.md fallback) and
        fold the ones matching this slug's (model, engine) into the open Explain
        modal.  This is the home of the data the retired Validate · Benchmarks tab
        used to show (Fold 3) — surfaced per-slug in the explain drill-down."""
        rows, _error = await self._data.benchmarks_explorer()
        matched = [r for r in rows if _bench_row_matches(r, model, engine)]
        try:
            screen.set_cross_rig(matched)
        except Exception:
            pass

    @work(exclusive=True, group="evidence")
    async def load_evidence(self) -> None:
        """Enumerate the rebench run tags for the Evidence pane (filesystem READ)."""
        tags = await self._data.evidence_list()
        try:
            self.query_one("#validate-evidence-pane", ValidateEvidencePane).populate(tags)
        except Exception:
            pass

    # ── BYO fit-check ────────────────────────────────────────────────────────────────

    @work(exclusive=True, group="byo")
    async def run_byo_check(self, repo: str, profile_like: str) -> None:
        # The fit-check is the producer lane's ① Bring stage (the standalone Run ·
        # BYO tab was removed in the 2-mode merge).  Render the verdict into the
        # lane pane via the shared byo_check + verdict text.
        lane_pane = None
        try:
            lane_pane = self.query_one("#lane-bring-pane", LaneBringPane)
            lane_pane.set_checking(repo)
        except Exception:
            lane_pane = None
        # A12 — a free-text / legacy profile-like that isn't a known registry slug
        # gets a precise "unknown profile <X> — known: <list>" instead of a generic
        # pull.sh dry-run error.  Only enforced once we have a non-empty known set
        # (the registry loaded); an empty catalog falls through to the live check.
        known = self._known_profile_likes()
        if known and profile_like and profile_like not in known:
            shown = ", ".join(known[:12]) + ("…" if len(known) > 12 else "")
            res = ByoResult(
                repo=repo,
                profile_like=profile_like,
                error=f"unknown profile {profile_like} — known: {shown}",
            )
            self._last_byo = res
            if lane_pane is not None:
                lane_pane.populate(res)
            # N9 — a failed re-Bring must clear any STALE "● armed …" left by a
            # prior valid ① Bring: mirror the success path's set_armed so ② Serve
            # restores the "run ① Bring first" placeholder, consistent with the
            # error _last_byo (the serve ACTION is already gate-safe).
            try:
                self.query_one("#lane-serve-pane", LaneServePane).set_armed(None)
            except Exception:
                pass
            return
        res = await self._data.byo_check(repo, profile_like)
        # Cache the arch facts for the lane ② Serve + the Promote scaffold (Phase 5).
        self._last_byo = res
        if lane_pane is not None:
            lane_pane.populate(res)
        # N9 — carry the fit-check result forward: pre-arm ② Serve with the
        # resolved target so the producer pipeline flows ① → ② without re-entry.
        try:
            self.query_one("#lane-serve-pane", LaneServePane).set_armed(
                res if not getattr(res, "error", "") else None
            )
        except Exception:
            pass

    # ── Explain ──────────────────────────────────────────────────────────────────────

    @work(group="explain")
    async def run_explain(self, screen: ExplainScreen, slug: str) -> None:
        detail, err = await self._data.explain(slug)
        try:
            screen.set_detail(detail, err)
        except Exception:
            pass

    # ── The reconcile gate (called by the confirm modal on mount) ────────────────────

    @work(group="reconcile")
    async def run_reconcile_for_modal(self, screen: ConfirmActionScreen, plan: ActionPlan) -> None:
        """Re-run the FRESH reconcile gate for a pending write, push verdict back
        into the confirm modal.  Pending GPUs are inferred from the plan kind
        (None = conservative both-cards for a serve/scene).

        A plan that does NOT claim a GPU (``requires_reconcile=False`` — a
        validation launch, the c3t Evaluate hand-off, submit-bench, power-cap,
        cap-sweep, the promote write) is reported trivially-safe: the reconcile gate
        only models GPU contention, and these actions legitimately run WHILE a
        model is serving (busy GPUs are EXPECTED — gating Evaluate on 'GPU free'
        would wrongly disable it against the very target it evaluates).  These
        still go through the confirm modal (``requires_confirm``), just not the
        GPU gate."""
        if not plan.requires_reconcile:
            try:
                screen.set_reconcile(ReconcileResult(safe=True, action=f"{plan.kind}:{plan.description}"))
            except Exception:
                pass
            return
        pending = self._pending_gpus_for(plan)
        rec = await self._data.reconcile_before_write(
            f"{plan.kind}:{plan.description}",
            pending_gpus=pending,
            variants=self._variants or None,
        )
        try:
            screen.set_reconcile(rec)
        except Exception:
            pass

    def _pending_gpus_for(self, plan: ActionPlan) -> Optional[list[int]]:
        """Best-effort GPUs a write wants.  Conservative None → both cards for
        serve / scene; container ops target whatever the named container holds
        (unknown → conservative None)."""
        return None

    # ── Write dispatch (GATED · execution mocked in tests, NEVER live this phase) ────

    @work(exclusive=True, group="dispatch")
    async def dispatch_action(self, plan: ActionPlan) -> None:
        """Execute a confirmed write ActionPlan through the gated executor.

        ⚠️  WRITE PATH.  ``execute_action`` re-runs the reconcile gate itself and
        refuses if unsafe (unless the plan carries an explicit force + reason).
        The actual command is streamed via the core SubprocessRunner — in this
        phase that runner is NEVER executed live; tests inject a fake.

        Serialized two ways: ``exclusive=True`` on this worker group means a
        second dispatch cancels/queues rather than racing; and
        ``execute_action`` holds ``CockpitData._write_lock`` across the
        gate→write window so even direct concurrent calls can't TOCTOU the gate.
        """
        self._note_activity()
        if plan.kind in self._DOCKER_BURST_KINDS:
            import time as _t
            self._docker_burst_until = _t.monotonic() + self._DOCKER_BURST_SECS
        live = self._serve_live_pane()
        executed, rec, _state = await self._data.execute_action(
            plan, variants=self._variants or None
        )
        if not executed:
            summary = rec.conflict_summary if rec else "unknown"
            self.notify(
                f"Refused — gate unsafe (collides with: {summary}). Use Force to override.",
                title="Reconcile gate",
                severity="warning",
                timeout=6,
            )
            if live is not None and plan.kind == "serve":
                self._reveal_serve_live()
                live.append_line(f"[red]✗ refused[/red] — {plan.description} (gate unsafe: {summary})")
                # Surface the problem-report affordance AT the failure (design:
                # "surfaced *at* the failure") and capture the failure context so
                # [!] assembles a paste-ready issue with the readily-available log.
                self._capture_serve_failure(plan, f"✗ refused — {plan.description} (gate unsafe: {summary})")
                live.append_line("[dim]press ! to report this[/dim]")
            return
        self.notify(
            f"{plan.description} dispatched.",
            title="Action",
            severity="information",
            timeout=4,
        )
        if plan.kind == "serve":
            # A successful serve clears any stale failed-serve context so a later
            # [!] reports THIS state, not a PRIOR failure (R2b verify fix).
            self._problem_slug = ""
            self._problem_boot_log = ""
        # A1: re-poll the estate after EVERY successful GPU-mutating write so the
        # Operate panes / rail / GPU bars / the Run catalog marker reflect the new
        # rig state — not just power_cap.  These are READS that run AFTER the
        # reconcile gate passed (execute_action returned executed=True); a REFUSED
        # write returned above and never reaches here.
        #
        #  - SYNCHRONOUS kinds (power_cap, scene-switch, estate-down, container
        #    restart/stop/rm) take effect by the time execute_action returns →
        #    re-poll immediately, like power_cap always has.
        #  - SERVE is ASYNC (the engine boots over ~tens of seconds) → a single
        #    immediate re-poll would still show the OLD/empty state.  Instead arm
        #    a DEFERRED watcher that re-polls every few seconds until the booted
        #    slug matches (or it times out).  See _start_pending_serve_watch.
        _SYNC_REPOLL_KINDS = {
            "power_cap",      # power-cap set/sweep
            "power_cap_sweep",
            "scene",          # gpu-mode scene switch
            "estate_down",    # estate_cli down (stop all)
            "container",      # docker restart/stop <name>
            "container_rm",   # docker rm <name>
        }
        if plan.kind in _SYNC_REPOLL_KINDS:
            self.load_estate()
        if plan.kind == "serve":
            if live is not None:
                # Reveal the transient Run boot pane (Fold 2).  Do NOT print the
                # old inert "(boot log streams here)" — nothing actually streams
                # into this pane yet; be honest that we're WATCHING for the model
                # to come up (the deferred re-poll resolves the terminal state).
                self._reveal_serve_live()
                live.clear_log()
                slug = self._serve_slug_for(plan)
                live.append_line(
                    f"[green]▶ launching[/green] {slug or plan.description} — "
                    f"watching for it to come up…"
                )
            # A1/A10: arm the deferred re-poll that resolves the LivePane terminal
            # state (✓ serving / still booting / ✗ did not come up).
            self._start_pending_serve_watch(plan)

    @staticmethod
    def _is_generated_serve(plan: ActionPlan) -> bool:
        """A generated/BYO serve launches via ``docker compose -f <path> up -d``
        (serve_generated), NOT a registry ``switch.sh <slug>``.  Such a plan has
        no registry slug — `cmd[-1]` is `-d`, so we must NOT slug-match it."""
        cmd = getattr(plan, "cmd", None) or []
        return bool(cmd) and cmd[0] == "docker"

    def _serve_slug_for(self, plan: ActionPlan) -> str:
        """Best-effort slug for a REGISTRY serve plan: the staged entry, else the
        last cmd arg of ``switch.sh <slug>`` (handles ``--force <slug>``).

        Returns "" for a generated/BYO serve (``docker compose … up -d``): it has
        no registry slug, and deriving one from `cmd[-1]` would yield `-d` (which
        never matches a registry `matched_slug` → the LivePane would hang forever).
        The generated lane resolves its terminal state from a NEW container
        appearing, not a slug match — see _start_pending_serve_watch."""
        if self._is_generated_serve(plan):
            return ""
        if self._staged_entry is not None:
            s = getattr(self._staged_entry, "slug", "") or ""
            if s:
                return s
        if plan.cmd:
            return plan.cmd[-1]
        return ""

    # ── A1/A10: deferred serve re-poll (the boot is ASYNC) ───────────────────────────

    def _start_pending_serve_watch(self, plan: ActionPlan) -> None:
        """Arm the deferred re-poll for a just-dispatched serve.

        The engine boots asynchronously, so we can't re-poll once and know the
        result.  Record the slug we're waiting for, then schedule a repeating
        re-poll (cheap — one load_estate per tick).  Each poll calls
        _resolve_pending_serve; when the booted slug matches we stamp
        "✓ serving …" and stop, on timeout "… still booting", and the failure
        path (_capture_serve_failure) handles "✗ did not come up".

        Don't stack duplicate watchers: cancel any prior pending-serve timer
        first so a rapid second serve replaces (not duplicates) the watch."""
        self._cancel_pending_serve_timer()
        # MUST-FIX 1: a generated/BYO serve has no registry slug — resolve its
        # terminal state from a NEW container appearing rather than a slug match.
        self._pending_serve_generated = self._is_generated_serve(plan)
        if self._pending_serve_generated:
            self._pending_serve_slug = ""
            # Baseline is seeded from the FIRST post-launch poll (None until then)
            # so a stale/empty cached snapshot can't make a pre-existing container
            # look "new".  The new container coming up is the ✓ launched signal.
            self._pending_serve_baseline = None
        else:
            self._pending_serve_slug = self._serve_slug_for(plan)
            self._pending_serve_baseline = None
        self._pending_serve_model = ""
        self._pending_serve_port = 0
        self._pending_serve_attempts = 0
        # NH5: a generated serve has no slug AND (until the first poll) an empty
        # baseline-relative match — but it IS pending; track that we're watching
        # so the timer-arm-failure path below can honestly bail it out too.
        watching = bool(self._pending_serve_slug) or self._pending_serve_generated
        # Kick an immediate poll (so a fast boot resolves at once), then repeat.
        self.load_estate()
        try:
            self._pending_serve_timer = self.set_interval(
                self._SERVE_REPOLL_SECS, self._poll_pending_serve
            )
        except Exception:
            # NH5: _pending_serve_slug was committed BEFORE the timer armed; if
            # set_interval raised, there's no timer to ever resolve the LivePane
            # → it would hang on "watching…".  Clear the pending state and stamp
            # an honest line so the user knows to refresh manually.
            self._pending_serve_timer = None
            self._pending_serve_slug = ""
            self._pending_serve_generated = False
            self._pending_serve_baseline = None
            if watching:
                live = self._serve_live_pane()
                if live is not None:
                    live.append_line(
                        "[yellow]…[/yellow] could not arm watcher — "
                        "press r to refresh"
                    )

    def _poll_pending_serve(self) -> None:
        """One deferred re-poll tick for a pending serve: bump the attempt count,
        re-poll the estate (which calls _resolve_pending_serve with the fresh
        match), and on timeout stamp the 'still booting' line + stop."""
        # A registry serve is identified by its pending slug; a generated/BYO
        # serve has no slug and is identified by the generated flag.
        if not self._pending_serve_slug and not self._pending_serve_generated:
            self._cancel_pending_serve_timer()
            return
        self._pending_serve_attempts += 1
        if self._pending_serve_attempts > self._SERVE_REPOLL_MAX_ATTEMPTS:
            # Timed out without a match — honest "still booting", offer a refresh.
            live = self._serve_live_pane()
            if live is not None:
                live.append_line(
                    f"[yellow]…[/yellow] still booting — press r to refresh"
                )
            self._pending_serve_slug = ""
            self._pending_serve_generated = False
            self._pending_serve_baseline = None
            self._cancel_pending_serve_timer()
            return
        self.load_estate()

    def _resolve_pending_serve(self, state: EstateState) -> None:
        """A1/A10: called from load_estate with each fresh snapshot.  Resolve the
        serve LivePane terminal state from THIS poll.  A no-op when no serve is
        pending.

        Two lanes:
          - REGISTRY serve (`switch.sh <slug>`): wait for the estate's
            `matched_slug` to equal the served slug → "✓ serving <model> · :<port>".
          - GENERATED/BYO serve (`docker compose … up -d`): there is NO registry
            slug (MUST-FIX 1), so wait for a NEW container — one not present at
            launch — to appear in `state.containers` → "✓ launched <name> · :<port>".
            If it happens to registry-match we still surface the model/port."""
        if self._pending_serve_generated:
            self._resolve_pending_generated_serve(state)
            return
        slug = self._pending_serve_slug
        if not slug:
            return
        matched = (state.matched_slug or "").strip()
        if matched and matched == slug:
            live = self._serve_live_pane()
            if live is not None:
                tgt = state.target
                model = (getattr(tgt, "model", "") or "").strip() or matched
                port = getattr(tgt, "host_port", 0) or 0
                tail = f" · :{port}" if port else ""
                live.append_line(f"[green]✓ serving[/green] {model}{tail}")
            self._pending_serve_slug = ""
            self._cancel_pending_serve_timer()

    def _resolve_pending_generated_serve(self, state: EstateState) -> None:
        """MUST-FIX 1: terminal-state resolver for a generated/BYO serve.  Stamps
        "✓ launched <name>" the moment a container NOT in the launch-time baseline
        appears in the estate.  No registry slug is involved, so a STALE staged
        catalog slug can NEVER drive a false "✓ serving <that model>"."""
        containers = list(getattr(state, "containers", None) or [])
        names = {
            getattr(c, "name", "") for c in containers if getattr(c, "name", "")
        }
        # Seed the baseline from the FIRST post-launch poll: containers already
        # present now are NOT the freshly-launched one.  (A real boot takes
        # seconds, so the engine container won't be up on this first immediate
        # poll — and even if it raced, we'd just need one more tick.)
        if self._pending_serve_baseline is None:
            self._pending_serve_baseline = names
            return
        new = [
            c for c in containers
            if getattr(c, "name", "") and getattr(c, "name", "") not in self._pending_serve_baseline
        ]
        if not new:
            return
        # Prefer an engine container (it's the served model); else the first new.
        cont = next((c for c in new if getattr(c, "kind", "") == "engine"), new[0])
        live = self._serve_live_pane()
        if live is not None:
            name = getattr(cont, "name", "") or "container"
            # If the estate registry-matched the running engine, surface the
            # model + port; otherwise just the container name (honest — a BYO
            # compose may not match any registry slug).
            tgt = state.target
            model = (getattr(tgt, "model", "") or "").strip()
            port = getattr(cont, "host_port", 0) or getattr(tgt, "host_port", 0) or 0
            tail = f" · :{port}" if port else ""
            label = f"{name}" + (f" ({model})" if model else "")
            live.append_line(f"[green]✓ launched[/green] {label}{tail}")
        self._pending_serve_generated = False
        self._pending_serve_baseline = None
        self._pending_serve_slug = ""
        self._cancel_pending_serve_timer()

    def _cancel_pending_serve_timer(self) -> None:
        timer = getattr(self, "_pending_serve_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._pending_serve_timer = None

    def serve_failed(self, plan: ActionPlan, last_line: str) -> None:
        """A10: a detected serve failure → stamp the ✗ terminal line + capture the
        context for the [!] report.  Stops any pending-serve watcher (the boot is
        resolved — it failed).  Reuses _capture_serve_failure for the report
        context.  (Hook for the live SubprocessRunner failure callback in R5;
        tests drive it directly.)"""
        self._pending_serve_slug = ""
        self._pending_serve_generated = False
        self._pending_serve_baseline = None
        self._cancel_pending_serve_timer()
        live = self._serve_live_pane()
        if live is not None:
            self._reveal_serve_live()
            live.append_line(
                f"[red]✗ did not come up[/red] — press ! to report"
            )
        self._capture_serve_failure(plan, last_line)

    def _serve_live_pane(self) -> Optional[LivePane]:
        try:
            return self.query_one("#serve-live", LivePane)
        except Exception:
            return None

    def _reveal_serve_live(self) -> None:
        """Show the transient Run boot-output LivePane (hidden until a serve)."""
        try:
            self.query_one("#serve-live", LivePane).add_class("serving")
        except Exception:
            pass

    def _capture_serve_failure(self, plan: ActionPlan, last_line: str) -> None:
        """Capture a failed serve's context for the [!] problem report (R2b).

        Records the slug being served + the boot-log line(s) that describe the
        failure so ``problem_report`` can assemble a paste-ready issue from the
        readily-available context.  Best-effort: slug from the staged entry, else
        the last cmd arg of the serve plan (``switch.sh <slug>``)."""
        slug = ""
        if self._staged_entry is not None:
            slug = getattr(self._staged_entry, "slug", "") or ""
        if not slug:
            slug = self._target_slug or ""
        if not slug and plan.cmd:
            # The slug is the LAST arg of a serve plan — handles both
            # `switch.sh <slug>` and `switch.sh --force <slug>` (where cmd[2]
            # would wrongly be "--force").
            slug = plan.cmd[-1]
        self._problem_slug = slug
        self._problem_boot_log = last_line

    # ── Mode switching ───────────────────────────────────────────────────────────────

    def _switch_mode(self, index: int) -> None:
        # 2-mode merge: mode 0 = merged Run & Operate (#panel-run), mode 1 = the
        # Bring & Validate lane (#panel-validate).  #panel-operate is gone (its
        # tabs folded into #panel-run).
        panel_ids = ["panel-run", "panel-validate"]
        for i, pid in enumerate(panel_ids):
            try:
                panel = self.query_one(f"#{pid}")
                if i == index:
                    panel.add_class("active")
                else:
                    panel.remove_class("active")
            except Exception:
                pass
        try:
            self.query_one("#mode-switcher", ModeSwitcher).set_active(index)
        except Exception:
            pass
        self._active_mode = index
        # Refresh the footer so bindings shown/hidden update immediately.
        self.refresh_bindings()
        # Move focus to the mode's primary interactive widget so context
        # keys and ⏎ act on the right thing immediately.
        self._focus_mode_primary(index)
        # The merged Run & Operate mode hosts the live estate (Orchestration /
        # Containers / Doctor tabs) — poll the estate on entry.  load_estate chains
        # the Doctor read at its END (its profile-triage consumes the target the
        # poll captures, so it must run AFTER the poll, not race it).
        if index == 0:
            self.load_estate()
        # Bring & Validate lane is live too — load the evidence read AND prime the
        # live target: [v] Evaluate consumes _target_obj, which only load_estate
        # writes.  Without this, entering the lane directly (key 2) leaves
        # _target_obj=None → "nothing to evaluate" even with a model live.
        # load_estate best-effort-guards every merged-mode pane query_one, so it is
        # safe to call when those panes aren't shown.
        elif index == 1:
            self._load_validate()
            self.load_estate()

    def _focus_mode_primary(self, index: int) -> None:
        """Move focus to the mode's primary interactive widget after a mode switch.

        Deferred via call_after_refresh so this runs AFTER any pending
        call_after_refresh callbacks from on_tabbed_content_tab_activated (which
        are enqueued during mount) — ensuring mode-switch focus wins."""
        def _do() -> None:
            # DRY: resolve the active tab's primary list via the SHARED
            # _TAB_PRIMARY_LIST (the same helper the arrow-descent actions use)
            # rather than a hand-coded tab→table chain.  ``index`` is the mode being
            # entered; _switch_mode / on_mount set self._active_mode = index BEFORE
            # this deferred body runs, so the helpers resolve the right mode.
            try:
                tbl = self._primary_list_for_active_tab()
                if tbl is not None:
                    tbl.focus()
                elif index == 1:
                    # Lane table-less stage (① Bring / ② Serve / ⑤ Promote): focus
                    # the lane tab bar (a ContentTabs — not an Input — so 1/2 + [ ]
                    # still route, and there's always a VISIBLE focus, not None)
                    # rather than leaving a focus black hole now the footer is out of
                    # the Tab chain.
                    bar = self._active_tab_bar()
                    if bar is not None:
                        bar.focus()
                # mode-0 Doctor (also no table) is read-only → leave focus unset.
            except Exception:
                pass
        self.call_after_refresh(_do)

    # ── BUG 2: ModeSwitcher keyboard navigation (Modes rail ↕ tab bar ↕ list) ──────

    def _switch_mode_keep_modes_focus(self, index: int) -> None:
        """Switch the active mode (real path — content + highlight + bindings all
        update as 1/2 do) but KEEP focus on the ModeSwitcher, so the user can keep
        arrowing through modes.  _switch_mode → _focus_mode_primary moves focus to
        the mode's primary widget; we re-home it on the ModeSwitcher afterwards
        (deferred so it wins over _focus_mode_primary's own call_after_refresh)."""
        self._switch_mode(index)

        def _refocus_modes() -> None:
            try:
                self.query_one("#mode-switcher", ModeSwitcher).focus()
            except Exception:
                pass
        self.call_after_refresh(_refocus_modes)

    def _descend_from_modes(self) -> None:
        """Tab/Enter from the focused ModeSwitcher → DESCEND into the content:
        focus the active tab's primary list, or (no list) the tab bar.  Reuses the
        same resolution the mode-switch / arrow-descent paths use so the target
        never drifts."""
        try:
            tbl = self._primary_list_for_active_tab()
            if tbl is not None:
                tbl.focus()
                return
            bar = self._active_tab_bar()
            if bar is not None:
                bar.focus()
        except Exception:
            pass

    def _load_validate(self) -> None:
        """Kick the Validate read workers (evidence).
        Best-effort — a failing leg doesn't block the rest.  The Run pane is
        launch-driven (no background read).  R2a moved Doctor to Operate, so
        Validate no longer kicks load_doctor here (that fires on Operate entry)."""
        self.load_evidence()

    # ── Actions ──────────────────────────────────────────────────────────────────────

    def action_mode_run(self) -> None:
        # Mode 0 is the MERGED Run & Operate (Catalog · Orchestration · Containers
        # · Doctor in one TabbedContent).
        self._switch_mode(0)

    def action_mode_operate(self) -> None:
        # 2-mode merge: Operate folded into the merged mode 0.  Kept as a harmless
        # alias (the palette "Operate mode" command + any edge reference) — lands on
        # the merged mode, then jumps to the Orchestration tab so the verb still
        # means "show me the live estate".
        self._switch_mode(0)
        try:
            self.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
        except Exception:
            pass

    def action_mode_validate(self) -> None:
        # Belt-and-suspenders surface guard: Bring & Validate is the producer lane,
        # hidden on the LEAN surface.  The binding is already gated by check_action's
        # surface gate, but guard the action too so no programmatic / edge path can
        # land a lean-surface user in the producer mode.
        if self._surface != "producer":
            return
        self._switch_mode(1)

    def action_toggle_contribute(self) -> None:
        """[C] — the LEAN-view toggle (surface inversion).

        Both modes show by DEFAULT (the full/"producer" surface).  This toggle
        flips to a LEAN view ("consumer" surface) that HIDES the Bring & Validate
        mode + the producer-only verbs, for a consumer who wants the minimal rig
        view — and back.  The split gates the producer ENGINE, never the consumer
        share-back (rig report / submit bench / report problem stay reachable lean).

        On toggle we (1) flip self._surface, (2) re-render the ModeSwitcher so it
        shows 1↔2 modes (set_surface), (3) refresh_bindings() so the _PRODUCER_ONLY
        actions gate/ungate immediately, (4) persist via save_surface_setting so
        resolve_surface picks it up next launch, and (5) notify the new view.  The
        Bring & Validate lane panel is always mounted (gated by check_action, not
        conditionally composed), so there is nothing to mount/unmount — only the
        gating + the rail change.

        EDGE: switching to LEAN while the user is IN the now-hidden Bring & Validate
        mode (mode 1) would strand them, so we first switch them to the merged mode
        0.  Going back to full from the merged mode just re-reveals mode 1 — no
        forced switch."""
        new_surface = "consumer" if self._surface == "producer" else "producer"
        lean = new_surface == "consumer"
        # EDGE: going lean while stranded in the now-hidden lane → merged mode 0.
        if lean and self._active_mode == 1:
            self._switch_mode(0)
        self._surface = new_surface
        # The LEAN view carries a sub-title indicator; the full view is unmarked.
        if lean:
            self.sub_title = f"{self.SUB_TITLE} · ▸ LEAN"
        else:
            self.sub_title = self.SUB_TITLE
        # Re-render the rail (show/hide mode 1 — a pure .mode-hidden class flip),
        # then re-assert the active highlight on the still-valid current mode
        # (defensive/idempotent — set_surface doesn't touch the active index).
        try:
            ms = self.query_one("#mode-switcher", ModeSwitcher)
            ms.set_surface(new_surface)
            ms.set_active(self._active_mode)
        except Exception:
            pass
        # Re-gate the producer-only actions/modes for the new surface.
        self.refresh_bindings()
        # Persist for next launch (test-injectable via C3_CONFIG_DIR).
        from .__main__ import save_surface_setting
        save_surface_setting(new_surface)
        if lean:
            self.notify(
                "▸ Lean view — Bring & Validate hidden (press C to restore)",
                title="Lean view",
                timeout=4,
            )
        else:
            self.notify(
                "Full view — Bring & Validate restored",
                title="Lean view",
                timeout=3,
            )

    def action_toggle_rail(self) -> None:
        """[.] — collapse / restore the left rail (Modes + Estate).

        When hidden, the content area expands to the full terminal width; when
        shown, the rail returns.  This is a pure view toggle — it never touches
        the active mode, the surface, or any write.  We must NOT strand focus:
        the rail's only focusable child is the ModeSwitcher (a Static — not
        focusable), and the mode keys 1/2 are app-level BINDINGS (not widget
        focus dependent), so hiding the rail can't break mode switching.  But if
        focus happened to land inside the rail subtree we move it back to the
        active mode's primary widget so a hidden-but-focused widget can't swallow
        keys."""
        try:
            rail = self.query_one("#left-rail", Vertical)
        except Exception:
            return
        hide = "rail-hidden" not in rail.classes
        if hide:
            # If focus is somewhere in the rail subtree, re-home it before hiding.
            focused = self.focused
            if focused is not None:
                node = focused
                in_rail = False
                while node is not None:
                    if node is rail:
                        in_rail = True
                        break
                    node = node.parent
                if in_rail:
                    self._focus_mode_primary(self._active_mode)
            rail.add_class("rail-hidden")
        else:
            rail.remove_class("rail-hidden")

    # ── Batch 4 · copy-to-clipboard + horizontal scroll ──────────────────────────

    def _serving_endpoint_url(self) -> str:
        """F3: the serving endpoint URL (``http://<lan>:<port>/v1``) from the
        CACHED estate state — '' when nothing is serving / port unknown. Same
        LAN-IP derivation as switch.sh's ready-line (services.lan_ip)."""
        state = getattr(self, "_last_estate_state", None)
        port = getattr(getattr(state, "target", None), "host_port", 0) or 0
        if not port:
            return ""
        try:
            lan = self._data.lan_ip()
        except Exception:
            lan = "localhost"
        return f"http://{lan}:{port}/v1"

    def action_copy_endpoint(self) -> None:
        """[u] — copy the serving API URL (the "point your agent here" string).
        No-ops with a notify when nothing is serving."""
        url = self._serving_endpoint_url()
        if not url:
            self.notify(
                "No model serving — nothing to point an agent at yet.",
                title="API URL", severity="warning", timeout=3,
            )
            return
        self.copy_to_clipboard(url)
        self.notify(
            f"Copied API URL: {url}  (OpenAI-compatible · no auth)",
            title="API URL", severity="information", timeout=4,
        )

    def action_copy_context(self) -> None:
        """[Y] — yank the contextually-relevant text to the system clipboard.

        DataTable cells can't be drag-selected (Textual ``ALLOW_SELECT`` is False
        on tables), so a context-aware copy is the reliable path.  Priority:
        (1) an active text selection (a Static body the user dragged), (2) the
        open modal's paste-ready body (report / explain / evidence), (3) the
        highlighted row's primary id (catalog slug / scene / container / run tag).
        Uses ``copy_to_clipboard`` (OSC52) — works over SSH on an OSC52-capable
        terminal; no-ops with a notify when there's nothing to copy."""
        text, what = self._resolve_copy_text()
        if not text:
            self.notify(
                "Nothing to copy here — highlight a row or open a report first.",
                title="Copy", severity="warning", timeout=3,
            )
            return
        self.copy_to_clipboard(text)
        preview = " ".join(text.split())
        if len(preview) > 56:
            preview = preview[:55] + "…"
        self.notify(
            f"Copied {what} to clipboard: {preview}",
            title="Copy", severity="information", timeout=3,
        )

    def action_settings(self) -> None:
        """[S] — open Settings (MODEL_DIR + HF_TOKEN + director placement)."""
        import os as _os
        self.push_screen(
            SettingsScreen(self._data.weights_model_dir(), bool(_os.environ.get("HF_TOKEN")),
                           self._data.director_device())
        )

    def apply_settings(self, *, model_dir: str, hf_token: str, director_device: str = "gpu0") -> None:
        """Persist + apply Settings.  MODEL_DIR / HF_TOKEN persist to c3-settings.json;
        the director placement persists to the repo .env (STUDIO_DIRECTOR_DEVICE — what
        gpu-mode reads, applied on the next ai-studio start).  Empty text fields are
        no-ops; a model-dir change re-stats the catalog."""
        import os as _os
        from .__main__ import load_settings, save_settings
        s = load_settings()
        json_changed = False
        if model_dir and model_dir != self._data.weights_model_dir():
            self._data._model_dir = model_dir
            s["model_dir"] = model_dir
            json_changed = True
        if hf_token:
            _os.environ["HF_TOKEN"] = hf_token
            s["hf_token"] = hf_token
            json_changed = True
        if json_changed:
            save_settings(s)
        env_changed = False
        if director_device and director_device != self._data.director_device():
            env_changed = self._data.set_repo_env_var("STUDIO_DIRECTOR_DEVICE", director_device)
        if json_changed or env_changed:
            msg = (f"Saved · director → {director_device} (next ai-studio start)."
                   if env_changed else "Settings saved.")
            self.notify(msg, title="Settings", timeout=4)
        else:
            self.notify("No changes.", title="Settings", timeout=2)
        if model_dir:
            self.load_catalog()   # re-stat weights against the (possibly new) dir

    def on_mouse_down(self, event) -> None:
        """RIGHT-CLICK → copy.  A TUI captures the mouse, so the terminal's native
        select-and-right-click-copy is suppressed (the app receives the events).
        We restore the expectation: right-click copies the active Textual selection
        (drag across a card/report/preview — a Static — to select), or, when there
        is none (e.g. over a DataTable, which can't be drag-selected), the
        contextual text — the highlighted row's slug.  The hard, cross-terminal
        path for copying ARBITRARY on-screen text stays the terminal's own
        selection: hold Shift (or Option on macOS) while dragging to bypass the
        app's mouse capture, then the terminal's copy."""
        self._note_activity()
        if getattr(event, "button", 0) != 3:   # 3 = right button
            return
        self.action_copy_context()

    def _resolve_copy_text(self) -> tuple[str, str]:
        """Return ``(text, label)`` for [Y] — '' text means nothing copyable."""
        # 1. an explicit text selection (any selectable Static the user dragged).
        try:
            sel = self.screen.get_selected_text()
        except Exception:
            sel = None
        if sel:
            return sel, "selection"
        # 2. a modal holding paste-ready text (ShareBack / Explain / Evidence).
        getter = getattr(self.screen, "copyable_text", None)
        if callable(getter):
            try:
                payload = getter() or ""
            except Exception:
                payload = ""
            if payload:
                return payload, "details"
        # 2.5 (F4) — the live-log pane the user is looking at (Containers Logs
        # drill / ③ Gate output).  RichLog-family widgets don't implement text
        # selection (Top/Config are selectable Statics), so [Y] copies the
        # pane's streamed tail.  An idle pane has an empty tail (placeholders
        # aren't buffered) → falls through to the row-primary copy below.
        pane, label = self._visible_live_pane()
        if pane is not None:
            tail = pane.tail_text()
            if tail:
                return tail, label
        # 3. the highlighted row's primary id on the active table.
        return self._active_row_primary()

    def _visible_live_pane(self) -> tuple[Optional[LivePane], str]:
        """F4 — the LivePane the user is currently LOOKING at, if any: the
        Containers Logs drill or the ③ Gate run output.  Main screen only (a
        modal on top owns [Y] via its own binding / copyable_text).  The
        transient #serve-live pane is deliberately NOT routed: it shares the
        screen with the primary tables, whose [Y]-copies-the-slug semantics
        are established."""
        if len(self.screen_stack) > 1:
            return None, ""
        try:
            tab = self._current_subtab()
            if tab == "tab-containers" and self._active_drill_tab() == "drill-tab-logs":
                return self.query_one("#drill-logs", LivePane), "log tail"
            if tab == "tab-run":
                return self.query_one("#run-output", LivePane), "run-output tail"
        except Exception:
            pass
        return None, ""

    def _active_row_primary(self) -> tuple[str, str]:
        """The most-pasteable id of the highlighted row on the active primary
        table (clean — via each pane's typed accessor, not a markup-laden cell)."""
        tab = self._current_subtab()
        try:
            if tab == "tab-catalog":
                e = self.query_one("#catalog-pane", CatalogPane).selected_entry()
                if e:
                    return e.slug, "slug"
            elif tab == "tab-orchestration":
                sc = self.query_one("#operate-orch-pane", OperateOrchPane).selected_scene()
                if sc:
                    return sc.name, "scene"
            elif tab == "tab-containers":
                c = self.query_one(
                    "#operate-containers-pane", OperateContainersPane
                ).selected_container()
                if c:
                    return c.name, "container"
            elif tab == "tab-evidence":
                t = self.query_one(
                    "#validate-evidence-pane", ValidateEvidencePane
                ).selected_tag()
                if t:
                    return t.tag, "run tag"
        except Exception:
            pass
        return "", ""

    def action_hscroll_left(self) -> None:
        self._hscroll(-1)

    def action_hscroll_right(self) -> None:
        self._hscroll(1)

    def _hscroll(self, direction: int) -> None:
        """shift+←/→ — page-scroll the active wide table horizontally (faster than
        the per-column ←/→ cursor).  A no-op (silently) when nothing scrollable is
        focused or under a modal."""
        if len(self.screen_stack) > 1:
            return
        table = self._primary_list_for_active_tab()
        if table is None:
            return
        try:
            if direction < 0:
                table.scroll_page_left(animate=False)
            else:
                table.scroll_page_right(animate=False)
        except Exception:
            pass

    def action_refresh(self) -> None:
        """Re-read the live data layer for the active mode.

        Merged mode 0: re-poll the live estate (feeds the rail + the Orchestration
        / Containers / Doctor tabs + the catalog's live-VRAM column), AND re-read
        the catalog when the Catalog tab is active.  Mode 1 (lane): re-read the
        evidence + re-prime the live target so [v] Evaluate stays wired."""
        if self._active_mode == 0:
            self.load_estate(explicit_refresh=True)
            if self._current_subtab() == "tab-catalog":
                try:
                    self.query_one("#catalog-pane", CatalogPane).query_one(
                        "#catalog-status", Label
                    ).update("Refreshing catalog…")
                except Exception:
                    pass
                self.load_catalog()
        elif self._active_mode == 1:
            self._load_validate()
            # Re-prime the live target so [v] Evaluate stays wired to the
            # currently-serving model on a lane refresh (mirrors _switch_mode).
            self.load_estate(explicit_refresh=True)

    def action_filter_catalog(self) -> None:
        """[/] filters the catalog (merged mode 0 · Catalog tab)."""
        if self._active_mode == 0 and self._current_subtab() == "tab-catalog":
            try:
                self.query_one("#catalog-pane", CatalogPane).toggle_filter()
            except Exception:
                pass

    def action_toggle_catalog_model(self) -> None:
        """[\\] toggles the model-scope dropdown (merged mode 0 · Catalog tab)."""
        if self._active_mode == 0 and self._current_subtab() == "tab-catalog":
            try:
                self.query_one("#catalog-pane", CatalogPane).toggle_model_select()
            except Exception:
                pass

    def action_toggle_catalog_deprecated(self) -> None:
        """[h] show/hide 🗑️ deprecated slugs (merged mode 0 · Catalog tab)."""
        if self._active_mode == 0 and self._current_subtab() == "tab-catalog":
            try:
                self.query_one("#catalog-pane", CatalogPane).toggle_deprecated()
            except Exception:
                pass

    def _active_validate_tab(self) -> str:
        try:
            return self.query_one("#validate-tabs", TabbedContent).active
        except Exception:
            return ""

    def _active_operate_tab(self) -> str:
        try:
            return self.query_one("#operate-tabs", TabbedContent).active
        except Exception:
            return ""

    # ── Arrow-key focus descent (tab bar ↔ primary list) ───────────────────────────
    #
    # The focus chain is `tab bar (ContentTabs) ↔ active content list (DataTable)`.
    # These helpers resolve the active mode's TabbedContent / its tab bar / the
    # active tab's primary list, and back the [down]/[up] descend/ascend actions +
    # their check_action gates.  They reuse the SHARED _TAB_PRIMARY_LIST map so the
    # tab→list association never drifts from the tab-activation focus logic.

    def _active_tabbed_content(self) -> Optional["TabbedContent"]:
        """The TabbedContent for the ACTIVE mode (mode 0 = #operate-tabs, mode 1 =
        #validate-tabs), or None if it can't be resolved."""
        tc_id = {0: "#operate-tabs", 1: "#validate-tabs"}.get(self._active_mode, "")
        if not tc_id:
            return None
        try:
            return self.query_one(tc_id, TabbedContent)
        except Exception:
            return None

    def _active_tab_bar(self) -> Optional[ContentTabs]:
        """The active mode's tab bar (the ContentTabs that holds focus while
        arrow/[ ] cycle tabs), or None."""
        tc = self._active_tabbed_content()
        if tc is None:
            return None
        try:
            return tc.get_child_by_type(ContentTabs)
        except Exception:
            return None

    def _primary_list_for_active_tab(self) -> Optional[DataTable]:
        """The DataTable that is the active mode's active-tab PRIMARY list, or None
        when the active tab has no primary list (mode 0 · Doctor; lane ① Bring /
        ② Serve / ⑤ Promote — none are in _TAB_PRIMARY_LIST)."""
        active_tab = self._current_subtab()
        widget_id = _TAB_PRIMARY_LIST.get(active_tab, "")
        if not widget_id:
            return None
        try:
            return self.query_one(widget_id, DataTable)
        except Exception:
            return None

    def action_descend_to_content(self) -> None:
        """[down] on the tab bar → move focus INTO the active tab's primary list.

        Does NOT reset the list's cursor — the preserved row stays selected.  A
        no-op when there is no primary list (check_action also gates this away)."""
        table = self._primary_list_for_active_tab()
        if table is not None:
            try:
                table.focus()
            except Exception:
                pass

    def action_ascend_to_tabbar(self) -> None:
        """[up] ascent — focus on a primary list (at cursor row 0) ascends to the
        active mode's tab bar.

        Maintainer round-4: [up] on the TAB BAR no longer jumps to the
        ModeSwitcher (that round-3 branch is removed — check_action returns False
        for that case so this action never fires while focus is the tab bar).  The
        ModeSwitcher stays reachable via Shift+Tab.  This action therefore only
        handles the list-row0 → tab bar ascent."""
        bar = self._active_tab_bar()
        if bar is not None:
            try:
                bar.focus()
            except Exception:
                pass

    def action_explain(self) -> None:
        """Open the explain detail modal for the selected catalog slug (merged
        mode 0 · Catalog tab)."""
        if self._active_mode != 0 or self._current_subtab() != "tab-catalog":
            return
        try:
            entry = self.query_one("#catalog-pane", CatalogPane).selected_entry()
        except Exception:
            entry = None
        if entry is None:
            return
        # The screen loads its own detail + cross-rig on mount (so the body query
        # resolves against a fully-mounted modal — Fold 3 cross-rig fold included).
        self.push_screen(ExplainScreen(entry.slug, model=entry.model, engine=entry.engine))

    def action_primary_action(self) -> None:
        """⏎ — context-specific per mode AND (in the merged mode 0) per tab.

        Merged mode 0:  Catalog → serve the selected slug (reconcile-gated);
        Orchestration → switch to the selected scene (confirm-gated); Containers /
        Doctor have no ⏎ primary (their verbs are explicit keys).
        Mode 1 (lane):  per-stage (① Bring fit-check … ⑤ Promote scaffold)."""
        if self._active_mode == 0:
            tab = self._current_subtab()
            if tab == "tab-catalog":
                self._run_primary()
            elif tab == "tab-orchestration":
                self._operate_primary()
        elif self._active_mode == 1:
            self._validate_primary()

    def _validate_primary(self) -> None:
        """⏎ in the Bring & Validate lane — context-specific per stage (R3b-1):
          - ① Bring   : trigger the lane fit-check (byo_check).
          - ② Serve   : generate the compose for the fit-checked slug + serve it
                        untested (reconcile-gated).
          - ③ Gate    : launch the selected ladder/extra step (confirm-gated).
          - ④ Measure : open the paste-ready report for the selected tag.
          - ⑤ Promote : compute + preview the catalog scaffold ([P] also does this).
          (Doctor moved to Operate in R2a — it's a read-only view with no
          primary action.)"""
        tab = self._active_validate_tab()
        if tab == "tab-bring":
            self._trigger_lane_bring()
        elif tab == "tab-serve":
            self.action_serve_untested()
        elif tab == "tab-run":
            self._run_validation_selected()
        elif tab == "tab-evidence":
            self._open_evidence_report()
        elif tab == "tab-promote":
            self.action_promote_catalog()

    def _run_validation_selected(self) -> None:
        """Stage the selected Run step as a confirm-gated validation launch."""
        try:
            kind = self.query_one("#validate-run-pane", ValidateRunPane).selected_kind()
        except Exception:
            kind = None
        if kind is None:
            self.notify("No validation step selected.", title="Validate", severity="warning", timeout=3)
            return
        slug = self._target_slug or (self._staged_entry.slug if self._staged_entry else None)
        plan = self._data.validation_plan(
            kind,
            model=self._target_model or None,
            url=self._target_url or None,
            slug=slug,
        )
        self.push_screen(ConfirmActionScreen(plan, on_confirm=lambda p: self.run_validation_launch(kind)))

    def _open_evidence_report(self) -> None:
        try:
            tag = self.query_one("#validate-evidence-pane", ValidateEvidencePane).selected_tag()
        except Exception:
            tag = None
        if tag is None:
            self.notify("No run tag selected.", title="Evidence", severity="warning", timeout=3)
            return
        if tag.live:
            # F10 — report generation WRITES REPORT.md into the tag dir; doing
            # that mid-run would corrupt the observer's completed/live signal
            # (and the observer never executes against a run in flight).
            self.notify(
                "Run in flight — REPORT.md generates when it completes. "
                "Watch the ladder + live tail in the preview below the list.",
                title="Evidence",
                severity="information",
                timeout=5,
            )
            return
        # The screen loads its own report on mount (run_evidence_report), so the
        # set_report query resolves against a fully-mounted modal.
        self.push_screen(EvidenceReportScreen(tag.tag))

    def _run_primary(self) -> None:
        """⏎ on the merged mode's Catalog tab: stage the selected slug and open the
        reconcile-gated serve confirm directly.  The serve ActionPlan goes through
        the SAME ConfirmActionScreen → run_reconcile_for_modal → dispatch_action
        gate as every other GPU-mutating write; on confirm the boot streams into the
        re-homed LivePane (#serve-live) below the tabs."""
        if self._current_subtab() != "tab-catalog":
            return
        try:
            entry = self.query_one("#catalog-pane", CatalogPane).selected_entry()
        except Exception:
            entry = None
        if entry is None:
            return
        self._staged_entry = entry
        plan = self._data.serve(entry.slug)  # gated, NOT --force
        self.push_screen(
            ConfirmActionScreen(plan, serve_ctx=self._serve_context_for(entry))
        )

    def _serve_context_for(self, entry: CatalogEntry) -> ServeContext:
        """Build the state-aware SERVE context for a staged catalog ``entry``.

        Resolves the mode from the LAST estate poll's matched serving slug:
          - the entry IS the live-serving slug → ``"stop"`` (offer a targeted Stop
            of THIS model, via the SAME ``container_action(stop)`` B3 path); the
            stop_plan tears down the matched serving container.
          - else → ``"start"`` (the conflict warned-Start is decided later from the
            reconcile result the modal runs — we don't pre-judge it here).

        The card itself is rendered from ``entry``'s cached fields; this only picks
        the controls + the stop target.  ALL writes still flow through the modal's
        background detect + the lease (Stop dispatches its own gated plan)."""
        serving = (self._target_slug or "").strip()
        if serving and entry.slug == serving:
            con = self._serving_container()
            if con is not None:
                # A targeted stop of THE SERVING container would otherwise
                # reconcile-UNSAFE against ITSELF (it holds the GPU → the gate
                # reads it as a conflict).  A stop FREES a card, it doesn't claim
                # one, and the user explicitly chose Stop — so build the stop as an
                # explicit, reasoned force: the dual-writer gate still runs its
                # FRESH detect + holds the lease/claim (serialization preserved),
                # it just doesn't refuse the user's named teardown of the very
                # container they targeted.  (A concurrent claim_token is still a
                # HARD block — force can't override an in-flight write.)
                stop_plan = ActionPlan(
                    kind="container",
                    cmd=["docker", "stop", con.name],
                    description=f"docker stop {con.name}",
                    requires_reconcile=True,
                    force=True,
                    force_reason=f"user chose Stop for the serving model {entry.slug}",
                )
                return ServeContext(
                    mode="stop",
                    entry=entry,
                    stop_plan=stop_plan,
                    serving_container=con.name,
                )
            # Serving slug matched but we couldn't resolve the container — fall
            # back to a Start presentation (no stop target to offer).
        # Download UX takes precedence over a (doomed) Start when the weights
        # aren't on disk: a Start with no weights would just boot-fail.
        ws = getattr(entry, "weights_state", "") or ""
        if entry.slug in self._active_downloads():
            return ServeContext(mode="downloading", entry=entry, weights_state=WEIGHTS_DOWNLOADING)
        if ws in (WEIGHTS_ABSENT, WEIGHTS_PARTIAL):
            # Download mode (start_download gates manual/no-hf_repo with a notify).
            return ServeContext(mode="download", entry=entry, weights_state=ws)
        # present / unknown → Start.  A non-functional slug (experimental/preview/…)
        # needs --force to launch (switch.sh refuses it) → the modal presents Force
        # Start; functional (production/caveats) slugs get a plain Start.
        return ServeContext(
            mode="start",
            entry=entry,
            weights_state=ws,
            force_required=not _status_is_functional(entry.status),
        )

    def _operate_primary(self) -> None:
        """⏎ on the merged mode's Orchestration tab: confirm-gated scene switch.

        The 'off' scene routes to the COMPREHENSIVE stop_all (gpu-mode off + estate
        down; a forced 'off' also tears down any other detected running container),
        not the bare ``gpu-mode off`` which can't stop a cockpit-served slug."""
        try:
            scene = self.query_one("#operate-orch-pane", OperateOrchPane).selected_scene()
        except Exception:
            scene = None
        if scene is None:
            return
        # Studio guard — the studio scenes need the comfyui-local image; if it's
        # absent (setup-image-studio.sh not run), guide instead of a doomed switch.
        if scene.name in self._STUDIO_SCENES and not self._studio_ready:
            self.open_studio_setup()       # missing-models setup modal (download all)
            return
        plan = self._data.stop_all() if scene.name == "off" else self._data.scene_switch(scene.name)
        self.push_screen(ConfirmActionScreen(plan))

    # Scenes that require the locally-built comfyui-local image.  Starting one
    # before `setup-image-studio.sh` has built it just fails, so the cockpit
    # intercepts and points the user at the setup script.  (The standalone comfyui
    # scene was removed — comfyui runs via ai-studio; the comfyui SERVICE start in
    # Containers is guarded separately by name.)
    _STUDIO_SCENES: frozenset = frozenset({"ai-studio"})

    def _studio_not_ready_notice(self) -> None:
        """Tell the user the studio bundle isn't set up + how to set it up."""
        self.notify(
            "Image/Video Studio isn't set up yet — the comfyui-local image or the "
            "director model is missing.\nRun in a terminal:  bash scripts/"
            "setup-image-studio.sh\n(builds ComfyUI + downloads the models, then "
            "starts the bundle.)",
            title="Studio setup needed",
            severity="warning",
            timeout=12,
        )

    @work(exclusive=True, group="studio-setup-open")
    async def open_studio_setup(self) -> None:
        """Gather the ai-studio missing-model state (image build + the manifest scan),
        then push the StudioSetupScreen — the first-install "what's missing + download
        all" modal (replaces the bare not-ready notify on the scene switch)."""
        image_built = await self._data.comfyui_image_present()
        missing = self._data.studio_missing_models()
        self.push_screen(StudioSetupScreen(missing, image_built, self._data))

    @work(exclusive=True, group="studio-download")
    async def run_studio_download_worker(self) -> None:
        """Launch the DETACHED 'download all missing ai-studio models' run.  Progress is
        polled by the StudioSetupScreen (disk), not streamed here — so the user can close
        the modal and the download keeps going.  Mock-only in tests (conftest blocks the
        real spawn)."""
        try:
            await self._data.run_studio_download()
        except Exception as exc:  # pragma: no cover - defensive
            self.notify(f"Studio download failed to start: {exc}", title="Studio setup",
                        severity="error", timeout=6)
            return
        self.notify("Downloading ai-studio models in the background…",
                    title="Studio setup", timeout=5)

    def action_help(self) -> None:
        # Thread the surface so the consumer help OMITS the producer lane (R3b-1).
        self.push_screen(HelpScreen(surface=self._surface))

    # ── Default-pin management (Run · Catalog) ──────────────────────────────────

    def action_set_default(self) -> None:
        """[d] in Run · Catalog: pin the selected slug as its model default.

        A ``.env`` write — no GPU contention — but still routed through the same
        ConfirmActionScreen → dispatch_action → execute_action gate so every
        write has one path.  The plan's ``requires_reconcile=False`` makes the
        gate report clear immediately."""
        if self._active_mode != 0 or self._current_subtab() != "tab-catalog":
            return
        entry = self._selected_catalog_entry()
        if entry is None:
            return
        plan = self._data.set_default(entry.slug)
        self.push_screen(ConfirmActionScreen(plan))

    def action_clear_default(self) -> None:
        """[D] on the merged mode's Catalog tab: clear the model default pin for
        the selected slug's model (gated path, .env write)."""
        if self._active_mode != 0 or self._current_subtab() != "tab-catalog":
            return
        entry = self._selected_catalog_entry()
        if entry is None:
            return
        plan = self._data.clear_default(entry.model)
        self.push_screen(ConfirmActionScreen(plan))

    def _selected_catalog_entry(self) -> Optional[CatalogEntry]:
        try:
            return self.query_one("#catalog-pane", CatalogPane).selected_entry()
        except Exception:
            return None

    # ── Containers (Operate · Containers) ──────────────────────────────────────────────

    def action_container_logs(self) -> None:
        """[l] on the merged mode's Containers tab: stream `docker logs` for the
        selected container into the drill Logs LivePane.  This is a READ — safe to
        run live (the conftest blocks an accidental write, not this read)."""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-containers":
            return
        con = self._selected_container()
        if con is None:
            self.notify("No container selected.", title="Logs", severity="warning", timeout=3)
            return
        if self._is_stopped_service(con):
            self.notify(f"{con.name} is not running.", title="Logs", severity="warning", timeout=3)
            return
        try:
            tabs = self.query_one("#drill-tabs", TabbedContent)
            tabs.active = "drill-tab-logs"
        except Exception:
            pass
        self.stream_container_logs(con.name)

    def action_s_key(self) -> None:
        """[s] is context-sensitive:
          - merged mode 0 · Containers : gated `docker restart <name>`.
          - Bring & Validate · ④ Measure : gated submit-to-localmaxxing for the tag.
        Other contexts ignore it."""
        if self._active_mode == 1 and self._active_validate_tab() == "tab-evidence":
            self.action_evidence_submit()
            return
        self.action_container_restart()

    def action_container_restart(self) -> None:
        """Gated `docker restart <name>` (Operate · Containers)."""
        self._container_write("restart")

    def action_container_stop(self) -> None:
        """[x] in Operate · Containers: gated `docker stop <name>`."""
        self._container_write("stop")

    def _container_write(self, op: str) -> None:
        # Merged mode 0 · Containers tab ONLY.  [s] (restart) falls through here
        # from action_s_key without a sub-tab gate, so guard the WRITE itself —
        # Catalog / Orchestration / Doctor are not container-write surfaces, and a
        # stray [s] there must not pop a `docker restart` confirm.
        if self._active_mode != 0 or self._active_operate_tab() != "tab-containers":
            return
        con = self._selected_container()
        if con is None:
            self.notify(
                f"No container selected to {op}.", title="Containers", severity="warning", timeout=3
            )
            return
        if self._is_stopped_service(con):
            if op == "restart":
                # Studio guard — starting comfyui needs the bundle set up (image +
                # director GGUF); if not, guide to setup-image-studio.sh, don't fail.
                if con.name == "comfyui" and not self._studio_ready:
                    self._studio_not_ready_notice()
                    return
                # GPU1 video⊕voice mutex — premium voice (step-audio-editx) needs ~14 GB
                # on GPU1; if a video render already holds that card, refuse instead of
                # OOMing.  Telemetry-honest (no telemetry ⇒ allowed).
                if con.name == "studio-step-voice":
                    reason = self._data.studio_voice_blocked(getattr(self, "_last_telemetry", None))
                    if reason:
                        self.notify(reason, title="Premium voice", severity="warning", timeout=8)
                        return
                # [s] on a STOPPED service = bring it UP.  A top-level service dir
                # → compose up (reconcile-gated, GPU-safe); an existing scene-managed
                # container without such a dir (studio-* sidecar) → docker restart.
                plan = self._data.service_start_or_restart(con.name)
                self.push_screen(ConfirmActionScreen(plan))
                return
            self.notify(
                f"{con.name} is not running — nothing to {op}.",
                title="Containers", severity="warning", timeout=3,
            )
            return
        plan = self._data.container_action(con.name, op)
        self.push_screen(ConfirmActionScreen(plan))

    def _selected_container(self) -> Optional[ContainerInfo]:
        try:
            return self.query_one(
                "#operate-containers-pane", OperateContainersPane
            ).selected_container()
        except Exception:
            return None

    def _is_stopped_service(self, con: Optional[ContainerInfo]) -> bool:
        """A known-but-not-running supporting service (#2) — there is no live
        container, so logs / top / restart / stop / rm have nothing to act on."""
        return con is not None and getattr(con, "status", "running") == "stopped"

    @work(group="container-logs", exclusive=True)
    async def stream_container_logs(self, name: str) -> None:
        """Read `docker logs --tail <N> <name>` and push lines into the drill
        Logs LivePane.  READ-only; goes through the injected read runner so
        tests stay subprocess-free."""
        try:
            live = self.query_one("#drill-logs", LivePane)
        except Exception:
            live = None
        if live is not None:
            live.clear_log()
            live.append_line(f"[dim]$ docker logs --tail 200 {name}[/dim]")
        res = await self._data.container_logs(name)
        if live is None:
            return
        if res.get("error"):
            live.append_line(f"[red]logs unavailable:[/red] {res['error']}")
            return
        for ln in res.get("lines", []):
            live.append_line(ln)

    def action_container_rm(self) -> None:
        """[X] on the merged mode's Containers tab: reconcile-gated `docker rm
        <name>`.

        Removing a container frees a GPU it held → the plan requires_reconcile,
        so it routes through the SAME ConfirmActionScreen → dispatch_action gate
        as stop.  rm of a live container needs Force (which adds -f)."""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-containers":
            return
        con = self._selected_container()
        if con is None:
            self.notify("No container selected to remove.", title="Containers", severity="warning", timeout=3)
            return
        if self._is_stopped_service(con):
            self.notify(f"{con.name} is not running.", title="Containers", severity="warning", timeout=3)
            return
        plan = self._data.container_rm(con.name)
        self.push_screen(ConfirmActionScreen(plan))

    def action_context_t(self) -> None:
        """[t] reads `docker top` for the selected container (merged mode 0 ·
        Containers tab)."""
        if self._active_mode == 0 and self._active_operate_tab() == "tab-containers":
            self._container_top()

    def _container_top(self) -> None:
        con = self._selected_container()
        if con is None:
            self.notify("No container selected.", title="Top", severity="warning", timeout=3)
            return
        if self._is_stopped_service(con):
            self.notify(f"{con.name} is not running.", title="Top", severity="warning", timeout=3)
            return
        try:
            self.query_one("#drill-tabs", TabbedContent).active = "drill-tab-stats"
        except Exception:
            pass
        self.read_container_top(con.name)

    @work(group="container-top", exclusive=True)
    async def read_container_top(self, name: str) -> None:
        """docker top <name> (READ) → the Top drill tab.  Also fills the Config
        tab from the cached registry row matched to the selected container."""
        top = await self._data.container_top(name)
        con = self._selected_container()
        variant = None
        if con is not None and con.slug:
            variant = next((v for v in self._variants if getattr(v, "slug", "") == con.slug), None)
        try:
            pane = self.query_one("#operate-containers-pane", OperateContainersPane)
            pane.populate_top(top)
            pane.populate_config(con, variant)
        except Exception:
            pass

    # ── Validate · Run launch (streams via run_validation — MOCKED in tests) ──────────

    @work(exclusive=True, group="validation-run")
    async def run_validation_launch(self, kind: str) -> None:
        """Launch a confirmed validation step, streamed into the Run LivePane.

        ⚠️  WIRED-BUT-MOCK-ONLY.  These scripts stress / hit a serving model and
        are heavy; the write runner is NEVER executed live this phase — conftest
        blocks the real spawn and tests inject a FakeWriteRunner."""
        live = self._run_output_pane()
        if live is not None:
            live.clear_log()
            live.append_line(f"[green]▶ launching[/green] {kind} (streams below)")
        slug = self._target_slug or (self._staged_entry.slug if self._staged_entry else None)
        # A9: mark the ladder row "running" while the step is in flight.
        self._set_run_outcome(kind, "running")

        def _on_line(text: str) -> None:
            if live is not None:
                live.append_line(text)

        run_state = await self._data.run_validation(
            kind,
            model=self._target_model or None,
            url=self._target_url or None,
            slug=slug,
            on_line=_on_line,
        )
        # A9/MUST-FIX 4: run_validation returns the core run state RIGHT AFTER
        # spawning (verdict=='', exit_code=None) — the real verdict is written only
        # when the detached _read_output task finishes and sets state.done.  Without
        # awaiting that, a COMPLETED real run stays stuck at ⟳ (the spawn-time
        # unknown verdict).  AWAIT the per-run completion (the established pattern,
        # services.py _release_claim_when_done) before reading the verdict.  The
        # worker is async — a long soak just keeps the row at ⟳ while it streams,
        # then resolves; the script owns its own timeout so we use none here.  A
        # mock/stub state with no ``done`` event (the FakeWriteRunner dict) skips
        # the await and resolves immediately via _run_verdict's unknown→running.
        done = getattr(run_state, "done", None)
        if done is not None:
            await done.wait()
        # A9: record the last-run OUTCOME so the ③ Gate ladder shows ✓/✗ per kind
        # (decision input for ⑤ Promote) — read from the core run state's verdict
        # / exit_code.  An unknown outcome (no verdict, no exit_code) leaves the row
        # at "running" rather than claiming a pass it didn't measure.
        self._set_run_outcome(kind, self._run_verdict(run_state))
        self.notify(f"{kind} launched.", title="Validate", severity="information", timeout=4)

    @staticmethod
    def _run_verdict(run_state: Any) -> str:
        """A9: map a core run state to a ladder outcome glyph key.  Honest about
        the unknown case — a mock/None state with no verdict stays 'running' (we
        never fabricate a pass)."""
        verdict = (getattr(run_state, "verdict", "") or "").strip().lower()
        if verdict in ("passed", "failed", "warn"):
            return verdict
        exit_code = getattr(run_state, "exit_code", None)
        if exit_code is not None:
            return "passed" if exit_code == 0 else "failed"
        return "running"

    def _set_run_outcome(self, kind: str, status: str) -> None:
        try:
            self.query_one("#validate-run-pane", ValidateRunPane).set_run_outcome(kind, status)
        except Exception:
            pass

    def _run_output_pane(self) -> Optional[LivePane]:
        try:
            return self.query_one("#run-output", LivePane)
        except Exception:
            return None

    # ── Validate · Evidence report (READ — reads results) ─────────────────────────────

    @work(group="evidence-report")
    async def run_evidence_report(self, screen: EvidenceReportScreen, tag: str) -> None:
        """Generate (reads results) + load the paste-ready report for a tag."""
        report = await self._data.evidence_report(tag)
        try:
            screen.set_report(report)
        except Exception:
            pass

    # ── Phase R / R3b-2 · ④ Measure-vs-curated-bar (READ — producer-only) ──────────

    def action_measure_vs_bar(self) -> None:
        """[m] in the lane's ④ Measure tab: open the "vs catalog bar" view for the
        selected evidence tag.  PRODUCER-only (gated in check_action); READ-only —
        no ConfirmActionScreen, no GPU / network.  Compares the producer's measured
        numbers to the curated catalog bar + flags the protocol it can't verify."""
        if self._active_mode != 1 or self._active_validate_tab() != "tab-evidence":
            return
        try:
            tag = self.query_one("#validate-evidence-pane", ValidateEvidencePane).selected_tag()
        except Exception:
            tag = None
        if tag is None:
            self.notify("No run tag selected.", title="Measure", severity="warning", timeout=3)
            return
        self.push_screen(MeasureVsBarScreen(tag.tag))

    @work(group="measure-vs-bar")
    async def run_measure_vs_bar(self, screen: "MeasureVsBarScreen", tag: str) -> None:
        """Compute the measured-vs-bar comparison for a tag (READ) + push it to
        the modal.  No GPU / network / write — pure filesystem reads + the
        benchmarks explorer."""
        vsbar = await self._data.measure_vs_bar(tag, variants=self._variants or None)
        try:
            screen.set_result(vsbar)
        except Exception:
            pass

    # ── Phase R / R3b-2 · Full validation battery (report.sh --full — producer) ────

    def action_full_report(self) -> None:
        """[F] — launch the ~43-min FULL validation battery (report.sh --full).
        Reachable from the lane's ③ Gate tab AND (Batch 3) Operate · Doctor.
        CONFIRM-gated (heavy + long-running, and it needs a model serving), then
        bg-streamed into the ③ Gate LivePane when that pane exists (off-lane —
        e.g. from Doctor on the lean surface — it runs in the background and the
        artifact lands in results/, viewable in Evidence).  It uses the SERVING
        model and does NOT claim a GPU (requires_confirm=True,
        requires_reconcile=False) → NEVER auto-fired."""
        on_gate = self._active_mode == 1 and self._active_validate_tab() == "tab-run"
        on_doctor = self._active_mode == 0 and self._active_operate_tab() == "tab-doctor"
        if not (on_gate or on_doctor):
            return
        # Guard on a resolved serving target — the ~43-min battery hits the
        # SERVING model; with nothing serving it would run against an empty
        # MODEL=/URL=.  Refuse + tell the user, never open the confirm.
        if not self._target_url and not self._target_model:
            self.notify(
                "No serving model — start a model before the full battery.",
                title="Full report",
                severity="warning",
                timeout=4,
            )
            return
        plan = self._data.full_validation_report_plan(
            model=self._target_model or None,
            url=self._target_url or None,
        )
        self.push_screen(ConfirmActionScreen(plan, on_confirm=lambda p: self.run_full_report_launch()))

    @work(exclusive=True, group="full-report")
    async def run_full_report_launch(self) -> None:
        """Launch the confirmed report.sh --full battery, streamed into the ③ Gate
        LivePane.

        ⚠️  WIRED-BUT-MOCK-ONLY.  The ~43-min battery hits the serving model; the
        write runner is NEVER executed live this phase — conftest blocks the real
        spawn and tests inject a FakeWriteRunner.  Uses the serving model; claims
        no GPU."""
        live = self._run_output_pane()
        if live is not None:
            live.clear_log()
            live.append_line(
                "[green]▶ launching[/green] report.sh --full "
                "(~43-min full battery · streams below)"
            )

        def _on_line(text: str) -> None:
            if live is not None:
                live.append_line(text)

        await self._data.run_full_validation_report(
            model=self._target_model or None,
            url=self._target_url or None,
            on_line=_on_line,
        )
        self.notify(
            "report.sh --full launched (~43-min battery).",
            title="Full report",
            severity="information",
            timeout=4,
        )

    # ── Phase R / R2b · Consumer share-back (READ paste-ready + outward submit) ────

    @work(group="share-back-report")
    async def run_share_back_report(self, screen: "ShareBackReportScreen", kind: str) -> None:
        """Load a consumer share-back report (READ — local context, no network).

        ``kind`` selects the loader: ``"rig"`` → rig_report (bare report.sh, a
        ~2 s snapshot); ``"problem"`` → problem_report (boot-log + compose + rig
        snapshot from the failure context the app captured).  Neither touches the
        network or a GPU."""
        if kind == "rig":
            res = await self._data.rig_report()
        elif kind == "problem":
            res = await self._data.problem_report(
                self._problem_slug,
                boot_log=self._problem_boot_log,
                url=self._target_url or None,
                variants=self._variants or None,
            )
        else:  # pragma: no cover - defensive
            res = {"report": "", "error": f"unknown report kind {kind!r}"}
        try:
            screen.set_report(res.get("report", ""), res.get("error"))
        except Exception:
            pass

    def action_rig_report(self) -> None:
        """[R] (Run + Operate): open the paste-ready rig/bench report.

        CONSUMER-resident share-back — NOT producer-gated.  It is a READ (bare
        report.sh generates a redacted ~2 s rig/stack snapshot; no network, no
        GPU write — the heavy --full validation battery is the producer Gate's
        job, R3), so there is NO ConfirmActionScreen: the user copies the text
        and posts it themselves."""
        self.push_screen(ShareBackReportScreen("Rig report · paste-ready", "rig"))

    def action_report_problem(self) -> None:
        """[!] (Run + Operate): open the paste-ready problem report.

        CONSUMER-resident share-back — NOT producer-gated.  READ-only: it gathers
        LOCAL failure context (the last serve's slug + captured boot-log + a rig
        snapshot) into a paste-ready issue; the user copies + opens the issue.
        Surfaced AT a failed serve via the affordance line in dispatch_action."""
        self.push_screen(ShareBackReportScreen("Report a problem · paste-ready", "problem"))

    def action_submit_bench(self) -> None:
        """[B] (merged mode 0): stage the OUTWARD submit-to-localmaxxing for the
        most recent benched run tag.  This is the ONLY outward write of the three
        share-back affordances — it keeps its confirm + network gate (the existing
        submit_bench ActionPlan: requires_confirm + network=True).  The network is
        mocked in tests; NEVER auto-fired."""
        if self._active_mode != 0:
            return
        self.resolve_and_submit_bench()

    @work(group="submit-bench-resolve")
    async def resolve_and_submit_bench(self) -> None:
        """Resolve the most-recent evidence tag, then stage the gated submit."""
        tags = await self._data.evidence_list()
        if not tags:
            self.notify(
                "No benched results to submit — run a bench first.",
                title="Submit bench",
                severity="warning",
                timeout=4,
            )
            return
        tag = tags[0].tag  # evidence_list is newest-first
        plan = self._data.submit_bench(tag)
        self.push_screen(ConfirmActionScreen(plan))

    def action_evidence_submit(self) -> None:
        """[s] in Bring & Validate · ④ Measure: stage the gated submit-to-
        localmaxxing for the selected run tag.  OUTWARD-FACING NETWORK WRITE —
        confirm-gated, NEVER auto-fired; the network is mocked in tests."""
        if self._active_mode != 1 or self._active_validate_tab() != "tab-evidence":
            return
        try:
            tag = self.query_one("#validate-evidence-pane", ValidateEvidencePane).selected_tag()
        except Exception:
            tag = None
        if tag is None:
            self.notify("No run tag selected.", title="Evidence", severity="warning", timeout=3)
            return
        if tag.live or tag.stale:
            # F10 — an in-flight run has incomplete artifacts; an aborted one
            # has no REPORT.md.  Neither is submittable evidence.
            self.notify(
                "This run has no completed REPORT.md — submit when the gate finishes.",
                title="Evidence",
                severity="warning",
                timeout=4,
            )
            return
        plan = self._data.submit_bench(tag.tag)
        self.push_screen(ConfirmActionScreen(plan))

    # ── Operate · Orchestration: power cap (gated rig write) ────────────────────────────

    def action_power_cap(self) -> None:
        """[c] in Operate · Orchestration: open the power-cap MENU — apply the
        default 230W cap / clear the cap (uncap) / set a custom wattage.  The
        chosen WRITE routes through the standard confirm gate; NEVER auto-fired."""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-orchestration":
            return
        self._open_power_cap_menu()

    @work(group="power-cap-menu")
    async def _open_power_cap_menu(self) -> None:
        st = await self._data.power_cap_get()
        self.push_screen(PowerCapMenuScreen(self._power_cap_summary(st)),
                         self._on_power_cap_choice)

    @staticmethod
    def _power_cap_summary(st) -> str:
        """One-line current per-GPU cap state for the menu header."""
        if getattr(st, "error", ""):
            return f"[red]{st.error}[/red]"
        parts = []
        for g in getattr(st, "gpus", []) or []:
            lim = f"{g.limit_w:.0f}W" if g.limit_w is not None else "?"
            dflt = f"{g.default_w:.0f}W" if g.default_w is not None else "?"
            capped = (g.limit_w is not None and g.default_w is not None
                      and g.limit_w < g.default_w)
            mark = "capped" if capped else "uncapped"
            parts.append(f"GPU{g.index}: {lim} ({mark}, default {dflt})")
        return "  ·  ".join(parts) if parts else "power-cap state unavailable"

    def _on_power_cap_choice(self, result) -> None:
        """Menu dismiss → build the chosen power-cap WRITE + route through the gate.
        ``result`` is ``("on"|"off", None)`` or ``("custom", <watts:int>)``; ``None``
        on cancel."""
        if not result:
            return
        kind, watts = result
        plan = self._data.power_cap_set(watts if kind == "custom" else kind)
        self.push_screen(ConfirmActionScreen(plan))

    def action_power_cap_sweep(self) -> None:
        """[w] in Operate · Doctor: confirm-gated power-cap sweep (heavy + mutating
        — runs benches at each cap).  A tuning/diagnostic bench, so it lives on
        Doctor (not the live-estate Orchestration tab).  NEVER auto-fired."""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-doctor":
            return
        plan = self._data.power_cap_sweep()
        self.push_screen(ConfirmActionScreen(plan))

    # ── Estate stop-all (Operate · Orchestration) ──────────────────────────────────────

    def action_estate_off(self) -> None:
        """[o] on the merged mode's Orchestration tab: gated estate-down (stop
        all)."""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-orchestration":
            return
        plan = self._data.estate_down()
        self.push_screen(ConfirmActionScreen(plan))

    # ── A4 · targeted serving verbs (Operate · Orchestration #serving-line) ────────

    def _serving_container(self) -> Optional[ContainerInfo]:
        """A4: resolve the container running the matched serving model.

        Matches the last estate poll's ``matched_slug`` against the running
        engine containers (``ContainerInfo.slug``) so the targeted stop/restart
        acts on JUST this model — not the whole estate ([o]).  Falls back to the
        detected target's container name when the slug join is empty.  Returns
        None when nothing is serving."""
        state = self._last_estate_state
        if state is None:
            return None
        slug = (getattr(state, "matched_slug", "") or "").strip()
        cons = [c for c in (getattr(state, "containers", []) or []) if getattr(c, "kind", "") == "engine"]
        if slug:
            for c in cons:
                if (getattr(c, "slug", "") or "").strip() == slug and getattr(c, "is_running", True):
                    return c
        # Fallback: the detected target's container name.
        tgt = getattr(state, "target", None)
        tgt_name = (getattr(tgt, "container", "") or "").strip()
        if tgt_name:
            for c in cons:
                if getattr(c, "name", "") == tgt_name and getattr(c, "is_running", True):
                    return c
        return None

    def _serving_write(self, op: str) -> None:
        """A4: confirm-gated `docker <op> <serving-container>` (op ∈ stop|restart).

        Resolves the serving model's container and routes the write through the
        SAME ConfirmActionScreen → reconcile gate as every other GPU-mutating
        write (NEVER auto-fired).  No-ops with a notify when nothing is serving."""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-orchestration":
            return
        con = self._serving_container()
        if con is None:
            self.notify(
                "No model serving — nothing to "
                f"{op}.",
                title="Serving",
                severity="warning",
                timeout=3,
            )
            return
        plan = self._data.container_action(con.name, op)
        self.push_screen(ConfirmActionScreen(plan))

    def action_serving_stop(self) -> None:
        """[k] in Operate · Orchestration: stop JUST the serving model's container
        (confirm-gated) — unlike [o] which tears down the whole estate."""
        self._serving_write("stop")

    def action_serving_restart(self) -> None:
        """[b] in Operate · Orchestration: restart the serving model's container
        (confirm-gated)."""
        self._serving_write("restart")

    def action_doctor_rerun(self) -> None:
        """[y] on the merged mode's Doctor tab (#4): re-run the live health read on
        demand (health.sh).  READ-only — no gate (nothing is mutated).  Distinct
        from the global [r] refresh in that it is the Doctor-resident, discoverable
        re-run verb.  (verify / verify-full are the deeper on-demand reads, [v]/[V].)"""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-doctor":
            return
        self.load_doctor()
        self.notify(
            "Re-running Doctor health read (health.sh)…",
            title="Doctor",
            severity="information",
            timeout=3,
        )

    def action_serving_switch(self) -> None:
        """[n] on the merged mode's Orchestration tab: switch model — flip to the
        Catalog tab (SAME mode now) to pick another (the serve itself is the
        existing reconcile-gated ⏎ path).  A pure navigation verb; no write here."""
        if self._active_mode != 0 or self._active_operate_tab() != "tab-orchestration":
            return
        try:
            self.query_one("#operate-tabs", TabbedContent).active = "tab-catalog"
        except Exception:
            pass
        self.notify(
            "Pick a variant and press ⏎ to switch — the serve is reconcile-gated.",
            title="Switch model",
            severity="information",
            timeout=4,
        )

    # ── Phase 5 · Hook 1: Evaluate the running target via c3t (design §4) ──────────────

    def action_evaluate_target(self) -> None:
        """[v] in the Bring & Validate lane: hand the SHARED ServingTarget to c3t
        (▸ Evaluate).  R3b-1 relocated the c3t hook into the lane (design: the c3t
        hook lives here); the live target is captured by the Operate estate poll
        and remains available via ``_target_obj``.

        Confirm-gated, MOCK-ONLY launch — c3t runs the post-boot evaluator
        against the live serving model (heavy).  The hand-off carries the SAME
        ``ServingTarget`` object the Estate poll detected (design §4/§6.6); the
        launch streams via ``launch_evaluate`` (write runner, NEVER live this
        phase — conftest blocks the spawn, tests fake it)."""
        if self._active_mode != 1:
            return
        handoff = self._data.evaluate_handoff(self._target_obj)
        if not handoff.available:
            self.notify(
                f"Evaluate: {handoff.reason}",
                title="Evaluate",
                severity="warning",
                timeout=4,
            )
            return
        # Confirm-gated; the commit launches c3t scoped to the shared target.
        self.push_screen(
            ConfirmActionScreen(
                handoff.plan,
                on_confirm=lambda _p: self.launch_c3t_evaluate(),
            )
        )

    @work(exclusive=True, group="evaluate")
    async def launch_c3t_evaluate(self) -> None:
        """Launch c3t scoped to the SHARED ServingTarget, streamed (MOCK-ONLY).

        ⚠️  WIRED-BUT-MOCK-ONLY.  c3t runs tests against the live serving model;
        the write runner is NEVER executed live this phase (conftest blocks the
        spawn; tests inject a FakeWriteRunner).  The SAME ``ServingTarget`` the
        Estate poll captured is passed by identity so c3t evaluates exactly what
        is running."""
        live = self._serve_live_pane()
        if live is not None:
            tgt = self._target_obj
            label = getattr(tgt, "model", "") or getattr(tgt, "url", "") or "target"
            live.append_line(f"[green]▶ c3t evaluate[/green] {label} (mock-only this phase)")

        def _on_line(text: str) -> None:
            if live is not None:
                live.append_line(text)

        await self._data.launch_evaluate(self._target_obj, on_line=_on_line)
        self.notify("c3t evaluate launched.", title="Evaluate", severity="information", timeout=4)

    # ── Phase R / R3b-1 · Bring & Validate lane ① Bring + ② Serve ──────────────────────

    def _trigger_lane_bring(self) -> None:
        """⏎ / Fit-check on the lane's ① Bring stage: run the lane fit-check
        (reuses byo_check) from the lane's own inputs."""
        try:
            repo = self.query_one("#lane-bring-url-input", Input).value.strip()
        except Exception:
            return
        profile = self._selected_profile_like("#lane-bring-profile-input")
        if not repo:
            self.notify("Enter an HF repo (org/Model).", title="① Bring", severity="warning", timeout=3)
            return
        self.run_byo_check(repo, profile)

    def action_serve_untested(self) -> None:
        """[g] / ⏎ in the Bring & Validate lane ② Serve: serve an untested
        REPRODUCTION of the resolved CATALOG profile's compose (R3b-1).

        ⚠️  HONESTY (R3b-1 fix): this does NOT serve the brought model's weights.
        ``generate-compose.sh`` has no --repo / weights-swap, so ② Serve generates
        + serves a verbatim reproduction of the *resolved catalog slug*'s compose
        (the Route-C sibling, else the profile-like the fit-check ran against) —
        the BYO repo / quant_match / drop_spec_config are NOT applied.  The full
        brought-model serve (pull-to-disk + a generate-compose.sh --repo extension)
        is a DEFERRED follow-up.

        Requires a successful ① Bring fit-check first (the cached ``_last_byo``).
        If no servable catalog slug resolves we do NOT fall back to a generic
        profile — we notify that this route has no servable target yet.  Otherwise
        we generate the catalog slug's minimal compose via ``generate_compose``
        (reproduce + flag, never repair), preview it VERBATIM badged "untested
        config reproduction", and — on confirm — serve it through the SAME
        reconcile-gated path every serve uses (the generated compose claims the
        GPU)."""
        if self._active_mode != 1:
            return
        if self._last_byo is None or getattr(self._last_byo, "error", ""):
            self.notify(
                "Run ① Bring fit-check first — no fit-checked model to serve.",
                title="② Serve",
                severity="warning",
                timeout=4,
            )
            return
        # The CATALOG slug whose compose we reproduce: the Route-C sibling, else
        # the profile-like the fit-check was run against.  We do NOT swap in the
        # brought model's weights (no --repo on generate-compose.sh yet) and we do
        # NOT fall back to a generic profile — if neither resolves, this route has
        # no servable target yet (the bring-your-own weight-swap is a pending
        # follow-up).
        slug = (
            getattr(self._last_byo, "sibling_slug", "")
            or getattr(self._last_byo, "profile_like", "")
        )
        if not slug:
            self.notify(
                "② Serve has no servable catalog target yet — the fit-check "
                "resolved no sibling/profile slug, and the bring-your-own "
                "weight-swap is a pending follow-up.",
                title="② Serve",
                severity="warning",
                timeout=5,
            )
            return
        self.generate_and_preview_compose(slug)

    @work(exclusive=True, group="generate-compose")
    async def generate_and_preview_compose(self, slug: str) -> None:
        """Generate the compose for ``slug`` (read-ish — writes only a temp file)
        and open the untested-compose preview modal.  The preview's confirm serves
        it through the reconcile gate; nothing auto-fires."""
        try:
            self.query_one("#lane-serve-pane", LaneServePane).set_status(
                f"[dim]Generating compose for[/dim] [cyan]{slug}[/cyan] "
                "[dim](generate-compose.sh)…[/dim]"
            )
        except Exception:
            pass
        res = await self._data.generate_compose(slug)
        if res.get("error") or not res.get("compose_yaml"):
            err = res.get("error") or "generator emitted no compose"
            try:
                self.query_one("#lane-serve-pane", LaneServePane).set_status(
                    f"[red]generate-compose failed:[/red] {err}"
                )
            except Exception:
                pass
            self.notify(f"② Serve: {err}", title="② Serve", severity="warning", timeout=5)
            return
        try:
            self.query_one("#lane-serve-pane", LaneServePane).set_status(
                f"[green]✓ generated[/green] compose for [cyan]{slug}[/cyan] — "
                "preview open (👤 untested)"
            )
        except Exception:
            pass
        self.push_screen(
            UntestedComposePreviewScreen(
                slug,
                res["compose_path"],
                res["compose_yaml"],
                on_serve=self._serve_generated_compose,
            )
        )

    def _serve_generated_compose(self, compose_path: str) -> None:
        """Stage the serve of a GENERATED compose through the reconcile gate.

        The serve_generated plan claims the GPU (``requires_reconcile=True``), so
        it routes through the SAME ConfirmActionScreen → run_reconcile_for_modal →
        dispatch_action gate as every serve — the dual-writer lease holds."""
        # MUST-FIX 1(b): a generated/BYO serve has NO registry slug.  Clear any
        # entry staged by a PRIOR catalog serve so that stale slug can NEVER drive
        # a false "✓ serving <that model>" via _serve_slug_for / failure capture.
        self._staged_entry = None
        plan = self._data.serve_generated(compose_path)
        self.push_screen(ConfirmActionScreen(plan))

    # ── Phase 5 · Hook 2: Promote the BYO model to the catalog (design §3.5b) ──────────

    def action_promote_catalog(self) -> None:
        """[P] in the Bring & Validate lane (⑤ Promote): compute + preview the
        catalog-promotion scaffold (R3b-1 relocated it out of Run · Catalog).

        Design §3.5b — a SCAFFOLD + GATE, not a YAML IDE.  Computes a ModelProfile
        YAML skeleton + a compose_registry row from the last BYO fit-check arch
        facts + any measured Evidence numbers, and previews them.  The write into
        scripts/ + the guard suite is the GATED write_plan on the scaffold —
        MOCK-ONLY this phase, never auto-fired."""
        if self._active_mode != 1:
            return
        if self._last_byo is None:
            self.notify(
                "No model to promote — run a ① Bring fit-check first.",
                title="Promote",
                severity="warning",
                timeout=4,
            )
            return
        meas = self._measurement_for_promote()
        scaffold = self._data.promote_scaffold(byo=self._last_byo, measurement=meas)
        if not scaffold.computed:
            self.notify(
                f"Cannot scaffold: {scaffold.error or 'incomplete BYO facts'}",
                title="Promote",
                severity="warning",
                timeout=5,
            )
            return
        # Preview only — the gated write routes through the same confirm gate
        # (mock-only), never auto-fires.
        self.push_screen(
            PromoteScaffoldScreen(
                scaffold,
                on_stage_write=lambda plan: self.push_screen(ConfirmActionScreen(plan)),
            )
        )

    def _measurement_for_promote(self) -> Optional[Measurement]:
        """Best-effort Evidence measurement for the Promote scaffold: the matched
        catalog entry's measurement (e.g. when a Route-C sibling already serves),
        else None.  Pure local lookup — no I/O."""
        sib = getattr(self._last_byo, "sibling_slug", "") if self._last_byo else ""
        if not sib:
            return None
        try:
            pane = self.query_one("#catalog-pane", CatalogPane)
        except Exception:
            return None
        for e in getattr(pane, "_entries", []) or []:
            if e.slug == sib:
                return e.measurement
        return None

    # ── Phase 5 · Hook 3: Optimize for my card (DORMANT v0.10.0 seam) ──────────────────

    def action_optimize_card(self) -> None:
        """[O] in Run: open the (dormant) per-card optimizer seam.

        The v0.10.0 optimizer does not exist yet — the modal shows 'optimizer not
        available (v0.10.0)'.  Available from Run · Catalog (selected slug); falls
        back to the last staged serve slug if no catalog row is selected."""
        if self._active_mode != 0:
            return
        entry = self._selected_catalog_entry()
        slug = entry.slug if entry else (self._staged_entry.slug if self._staged_entry else "")
        self.push_screen(OptimizeScreen(slug))

    @work(group="optimize")
    async def run_optimize_for_modal(self, screen: OptimizeScreen, slug: str) -> None:
        """Invoke the dormant optimizer seam + push the verdict into the modal.
        Detects the optimizer's absence → 'not available (v0.10.0)'; never
        fabricates output."""
        report = await self._data.optimize_for_card(slug=slug)
        try:
            screen.set_report(report)
        except Exception:
            pass

    # ── Sub-tab cycle actions ─────────────────────────────────────────────────────────

    def action_prev_subtab(self) -> None:
        """[ — cycle to the previous sub-tab in the active mode."""
        self._cycle_subtab(-1)

    def action_next_subtab(self) -> None:
        """] — cycle to the next sub-tab in the active mode."""
        self._cycle_subtab(1)

    def _cycle_subtab(self, direction: int) -> None:
        """Cycle the TabbedContent for the current mode by direction (+1 / -1)."""
        tab_widget_ids = {
            0: "#operate-tabs",   # merged Run & Operate (4 tabs)
            1: "#validate-tabs",  # Bring & Validate lane (①→⑤)
        }
        tc_id = tab_widget_ids.get(self._active_mode, "")
        if not tc_id:
            return
        try:
            tc = self.query_one(tc_id, TabbedContent)
            # MUST-FIX 2 — DIRECT panes only, so the cycle visits the 4 real Operate
            # tabs (catalog→orchestration→containers→doctor) / 5 lane stages and never
            # the nested Containers drill (drill-tab-logs/stats/config) sub-tabs.
            panes = self._direct_pane_ids(tc)
            if not panes:
                return
            current = tc.active
            try:
                idx = panes.index(current)
            except ValueError:
                idx = 0
            new_idx = (idx + direction) % len(panes)
            tc.active = panes[new_idx]
        except Exception:
            pass

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Refresh footer bindings whenever a sub-tab changes so context keys
        show/hide correctly (e.g. [/] appears only on Run·Catalog).  Also move
        focus to the new tab's primary widget.

        Focus is deferred via call_after_refresh because the new TabPane's content
        is not yet visible at the point the event fires, so an immediate .focus()
        call is silently lost.  Deferring one render cycle ensures the widget is
        fully displayed before we ask for focus.

        Note: Textual prefixes the Tab widget's id with `--content-tab-` when the
        TabbedContent creates the tab bar (e.g. the TabPane id `tab-run` becomes
        `--content-tab-tab-run` on the Tab object).  We strip that prefix so the
        focus map can use the clean TabPane IDs.

        We only apply focus for the TabbedContent that belongs to the *active* mode
        panel.  Events from mode panels that are currently hidden (display:none) are
        ignored so startup/background activations don't steal focus."""
        self.refresh_bindings()
        # Nested Operate·Containers drill-tabs (Logs/Top/Config): load the newly
        # active tab's content for the selected container, then stop — these are
        # NOT mode-level tabs and must not run the mode focus logic below.
        try:
            if event.tabbed_content.id == "drill-tabs":
                self._load_active_drill_tab()
                return
        except Exception:
            pass
        raw_tab_id = event.tab.id if event.tab else ""
        # Strip the Textual internal prefix if present.
        _PREFIX = "--content-tab-"
        tab_id = raw_tab_id[len(_PREFIX):] if raw_tab_id.startswith(_PREFIX) else raw_tab_id
        # N9 — entering ② Serve re-arms it from the cached ① Bring fit-check so the
        # resolved target is shown WITHOUT re-entering ① Bring (the pipeline flows).
        if tab_id == "tab-serve":
            try:
                self.query_one("#lane-serve-pane", LaneServePane).set_armed(self._last_byo)
            except Exception:
                pass
        # Only respond to tabs that belong to the current mode's active panel.
        # 2-mode merge: mode 0 is the MERGED Run & Operate (the 4 tabs); mode 1 is
        # the Bring & Validate lane stages ①→⑤.
        _mode_tabs: dict[int, set[str]] = {
            0: {"tab-catalog", "tab-orchestration", "tab-containers", "tab-doctor"},
            1: {"tab-bring", "tab-serve", "tab-run", "tab-evidence", "tab-promote"},
        }
        allowed_tabs = _mode_tabs.get(self._active_mode, set())
        if tab_id not in allowed_tabs:
            return
        # NOTE: the lane's ① Bring / ② Serve / ⑤ Promote stages + the merged mode's
        # Doctor tab are NOT in _TAB_PRIMARY_LIST on purpose — ① Bring's only
        # focusable widget is an Input (which would swallow the global digit (1/2) +
        # bracket ([ ]) keys), and Doctor is read-only.  Leaving focus on the tab bar
        # there keeps those keys routed to the app; the user Tab/clicks into the
        # input.  The map is the shared module constant so the tab-activation focus
        # logic + the arrow-key descend/ascend actions can never drift.
        _focus_map = _TAB_PRIMARY_LIST
        # FIX B — don't yank focus DOWN into the list when the user is browsing
        # the TAB BAR.  When focus is on the tab bar (a Tabs/ContentTabs widget —
        # that is what holds focus while arrow/[ ] cycle tabs; the individual Tab
        # children are not focusable), switching tabs must leave focus ON the tab
        # bar so the user can step Catalog→Orchestration→Containers→Doctor freely.
        # Only when focus is NOT on the tab bar (e.g. on a list, where [/] cycles
        # sub-tabs) do we move focus to the newly active tab's primary list — so
        # the list-operators keep operating lists.
        #
        # BUG 3 — the "don't yank focus while browsing the tab bar" guard applies
        # in BOTH modes now (was scoped to mode 0).  In the mode 1 producer lane,
        # activating ③ Gate / ④ Measure via the tab bar used to auto-focus their
        # list — the exact yank the user wants gone, now in the lane.  With the
        # mode-0 scope dropped, browsing the lane tab bar keeps focus on the tab
        # bar (consistent with Run & Operate).  A [/] cycle FROM a LIST (focus is
        # the DataTable, NOT a Tabs) still falls through and moves to the next
        # list — the table-less lane stages (① Bring / ② Serve / ⑤ Promote) aren't
        # in _focus_map, so a cycle landing there is a no-op below (focus stays
        # wherever it was), and a cycle landing on ③ Gate / ④ Measure focuses
        # their table as before.
        if isinstance(self.focused, Tabs):
            return
        widget_id = _focus_map.get(tab_id, "")
        if widget_id:
            def _do_focus() -> None:
                try:
                    self.query_one(widget_id, DataTable).focus()
                except Exception:
                    pass
                # #3/NH1: the Containers tab is CALM on entry — focus the table
                # but do NOT auto-load the highlighted row's drill detail.  No
                # arming is needed here: the row-0 echo from the entry populate
                # fired while Orchestration was active (guarded out of the
                # highlight handler), and focusing the table does not re-fire a
                # reaching RowHighlighted.  The flag is managed entirely by the
                # populate path ([r]-refresh) + the highlight handler.
            self.call_after_refresh(_do_focus)

    # ── Widget event handlers ─────────────────────────────────────────────────────────

    def on_select_changed(self, event: "Select.Changed") -> None:
        """NICE-TO-HAVE 2 — flag a GENUINE user pick of a profile template so the
        estate-poll rig-default reapply never clobbers it.

        Programmatic default-apply (``_set_select_options`` setting ``.value``)
        also fires ``Select.Changed``; we treat a change as user-driven ONLY when
        the new value differs from the last default we applied — so seeding the
        dropdown with the rig default doesn't count as a touch."""
        try:
            sel_id = event.select.id
        except Exception:
            return
        if sel_id == "catalog-model-select":
            val = event.value
            try:
                self.query_one("#catalog-pane", CatalogPane).set_model_filter(
                    "" if val in (Select.BLANK, _ALL_MODELS_VALUE) else str(val)
                )
            except Exception:
                pass
            return
        if sel_id != "lane-bring-profile-input":
            return
        new_val = event.value
        # FIX 2 (escape hatch) — the "✎ custom slug…" sentinel reveals + focuses the
        # companion free-text Input so any non-curated registry slug is reachable.
        # Selecting a curated slug again hides it.  (Done BEFORE the default-applied
        # guard so the toggle works on the very first user pick.)
        custom_id = "#lane-bring-profile-custom"
        try:
            custom = self.query_one(custom_id, Input)
            if new_val == PROFILE_CUSTOM_SENTINEL:
                custom.remove_class("profile-custom-hidden")
                custom.focus()
            else:
                custom.add_class("profile-custom-hidden")
        except Exception:
            pass
        # Before the registry-derived default has been applied, any Changed is the
        # initial-mount/placeholder seeding — not a user pick.
        if not getattr(self, "_profile_default_applied", False):
            return
        # Blank / no-selection sentinel isn't a meaningful pick.
        if new_val is None or new_val is Select.BLANK:
            return
        if new_val != getattr(self, "_last_applied_profile_default", None):
            self._profile_user_touched = True

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """DataTable emits RowSelected when the user presses Enter (select_cursor).
        Route it to the app's primary action so focusing a DataTable doesn't break
        the ⏎ → primary_action contract that the tests (and help text) document."""
        event.stop()
        self.action_primary_action()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select → inline preview, consistently across the tables (#9/#11/N8).

        Each preview is a pure LOCAL read (CatalogEntry / Scene / EvidenceTag /
        ladder-step blurb) rendered into a compact strip under its table — the
        same lazydocker-style highlight pattern Operate·Containers·Config uses.
        No I/O is done here (no subprocess per keystroke).  The container drill
        (Logs / Top) below is the ONE exception that does a debounced docker read;
        it stays gated to ``containers-table``."""
        try:
            tid = event.data_table.id
        except Exception:
            return

        # #9/A8 — Run · Catalog preview (consumer surface, READ).
        if tid == "catalog-table":
            try:
                pane = self.query_one("#catalog-pane", CatalogPane)
                pane.render_preview(pane.selected_entry())
            except Exception:
                pass
            return

        # #11 — Operate · Orchestration scene preview (READ).
        if tid == "scene-table":
            try:
                pane = self.query_one("#operate-orch-pane", OperateOrchPane)
                pane.render_scene_preview(pane.selected_scene())
            except Exception:
                pass
            return

        # N8 — Validate · ④ Measure evidence-tag preview (READ).
        if tid == "evidence-table":
            try:
                pane = self.query_one("#validate-evidence-pane", ValidateEvidencePane)
                pane.render_preview(pane.selected_tag())
            except Exception:
                pass
            return

        # N8 — Validate · ③ Gate validation-step preview (READ).
        if tid == "run-ladder-table":
            try:
                pane = self.query_one("#validate-run-pane", ValidateRunPane)
                pane.render_step_preview(pane.selected_kind())
            except Exception:
                pass
            return

        if tid != "containers-table":
            return
        if self._active_mode != 0 or self._active_operate_tab() != "tab-containers":
            return
        # #3/NH1: consult the load-bearing flag.  When an [r]-refresh repopulates
        # WHILE on this tab, the populate path re-arms the flag to False; the
        # immediately-following PROGRAMMATIC row-0 echo reaches here, is consumed
        # (flag→True) and is NOT auto-loaded — this kills the [r]-re-jump footgun
        # (spawning docker logs/top on row 0 off the user's prior selection).  A
        # genuine user arrow-move arrives with the flag already True → it loads,
        # and re-affirms the flag so further moves keep loading.
        if not self._containers_user_navigated:
            self._containers_user_navigated = True
            return
        # FIX 1 (clamp echo) — a CLAMP-to-different populate fires TWO programmatic
        # echoes: the t.clear() row-0 reset (swallowed by the gate above) AND the
        # follow-up move_cursor onto a container the user never selected.  The
        # latter arrives here with the navigated flag already True, so it would
        # auto-load a docker drill for that unselected container — re-introducing
        # the [r]-re-jump footgun on every periodic tick.  Swallow it via the
        # one-shot armed by load_estate; a later GENUINE user move clears nothing
        # (the flag stays True) and loads normally.
        if self._containers_suppress_clamp_echo:
            self._containers_suppress_clamp_echo = False
            return
        self._refresh_container_config()
        timer = getattr(self, "_drill_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._drill_timer = self.set_timer(0.25, self._load_active_drill_tab)

    def _active_drill_tab(self) -> str:
        try:
            return self.query_one("#drill-tabs", TabbedContent).active
        except Exception:
            return ""

    def _refresh_container_config(self) -> None:
        """Render the selected container's config tab from the cached registry
        (a local read — safe to run on every highlight)."""
        con = self._selected_container()
        variant = None
        if con is not None and con.slug:
            variant = next((v for v in self._variants if getattr(v, "slug", "") == con.slug), None)
        try:
            self.query_one("#operate-containers-pane", OperateContainersPane).populate_config(con, variant)
        except Exception:
            pass

    def _load_active_drill_tab(self) -> None:
        """Load the live content for whichever drill tab is active + the selected
        container. Logs/Top are docker reads (workers); Config is local (already
        refreshed on highlight)."""
        con = self._selected_container()
        if con is None:
            return
        if self._is_stopped_service(con):
            # No live container — don't spawn a docker logs/top read for a
            # known-but-stopped service (#2).  BUG 1: but DO clear the Logs/Top
            # drill panes to an explicit stopped placeholder so the previously-
            # selected RUNNING container's logs/stats don't stay on screen,
            # mislabeled to this stopped row.  (Config is a local read and is
            # refreshed correctly on highlight — leave it.)
            self._clear_drill_for_stopped(con)
            return
        tab = self._active_drill_tab()
        if tab == "drill-tab-logs":
            self.stream_container_logs(con.name)
        elif tab == "drill-tab-stats":
            self.read_container_top(con.name)

    def _clear_drill_for_stopped(self, con: "ContainerInfo") -> None:
        """BUG 1 — reset the Logs (#drill-logs LivePane) + Top (#drill-stats)
        drill panes to an explicit "stopped · no live logs/stats" placeholder so
        the panes always reflect the SELECTED container, never a stale other one.

        Reuses the existing populate methods (the logs pane's clear+write and the
        Top Static's update) — no new widgets.  Config is untouched: it's a local
        read refreshed on highlight (_refresh_container_config) and CAN show a
        stopped service's registry/compose info."""
        # Cancel any in-flight logs/top read from a PREVIOUSLY-selected running
        # container first: its exclusive worker isn't re-triggered by a stopped
        # row, so without this its slow `docker logs`/`top` result would resolve
        # AFTER the placeholder and append the FOREIGN container's lines below it
        # — the exact "stopped row shows another container's logs" symptom, via a
        # race on a slow real read.
        for grp in ("container-logs", "container-top"):
            try:
                self.workers.cancel_group(self, grp)
            except Exception:
                pass
        placeholder = f"▸ {con.name} — stopped · no live logs/stats"
        try:
            live = self.query_one("#drill-logs", LivePane)
            live.clear_log()
            # buffer=False: a display-only note — [Y] on an idle/stopped drill
            # must fall through to the container name, not copy this line (F4).
            live.append_line(f"[dim]{placeholder}[/dim]", buffer=False)
        except Exception:
            pass
        try:
            self.query_one("#operate-containers-pane", OperateContainersPane).query_one(
                "#drill-stats", Static
            ).update(f"[dim]{placeholder}[/dim]")
        except Exception:
            pass

    def on_key(self, event) -> None:
        """App-level Esc: close an open filter (Run·Catalog) and refocus the
        table. Modal screens capture their own Esc (they have escape→dismiss
        bindings), so this only runs on the main screen — Esc otherwise no-ops
        and NEVER quits."""
        self._note_activity()
        if event.key != "escape":
            return
        if isinstance(self.screen, ModalScreen):
            return
        if self._close_open_filter():
            event.stop()
            event.prevent_default()

    def _close_open_filter(self) -> bool:
        if self._active_mode == 0:
            pane_id, cls = "#catalog-pane", CatalogPane
        else:
            return False
        try:
            return self.query_one(pane_id, cls).close_filter_if_open()
        except Exception:
            return False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "lane-bring-fit-btn":
            self._trigger_lane_bring()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "catalog-filter":
            try:
                self.query_one("#catalog-pane", CatalogPane).set_filter(event.value)
                self.query_one("#catalog-pane", CatalogPane).query_one(
                    "#catalog-table", DataTable
                ).focus()
            except Exception:
                pass
        elif event.input.id == "lane-bring-url-input":
            self._trigger_lane_bring()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "catalog-filter":
            try:
                self.query_one("#catalog-pane", CatalogPane).set_filter(event.value)
            except Exception:
                pass

    def _selected_profile_like(self, select_id: str) -> str:
        """#6 — the profile-like string from a profile-template Select.  Reads the
        Select's value (a registry-derived slug); falls back to "vllm/dual" if the
        widget is blank/unresolved.

        FIX 2 (escape hatch) — when the "✎ custom slug…" sentinel is selected, read
        the companion free-text Input instead so any non-curated registry slug is
        expressible (byo_check then validates it via the unknown-profile path).  A
        blank custom Input falls back to the default (never the sentinel marker)."""
        try:
            val = self.query_one(select_id, Select).value
        except Exception:
            return "vllm/dual"
        if val is None or val is Select.BLANK:
            return "vllm/dual"
        if val == PROFILE_CUSTOM_SENTINEL:
            custom_id = "#lane-bring-profile-custom"
            try:
                typed = self.query_one(custom_id, Input).value.strip()
            except Exception:
                typed = ""
            return typed or "vllm/dual"
        return str(val).strip() or "vllm/dual"
