"""club3090 serve cockpit — main Textual application.

The three modes (Run · Operate · Validate) are wired to the real data layer
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
  - Operate / Orch       : ``estate_state`` live (GPU cards, Doctor, scenes,
                          services, power-cap); scene-switch → confirm modal that
                          FIRST calls ``reconcile_before_write`` then ``scene_switch``;
                          ``c`` cap on/off, ``w`` cap sweep, ``p`` prune (all gated).
  - Operate / Containers : ``containers`` real list; drill into Logs/Top/Config;
                          restart/stop/rm behind the reconcile-gated confirm.
  - Operate / Doctor     : real cards from ``doctor()`` (health + diagnose-estate
                          + diagnose-profile) — live-state, READ-only.
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
POST, power-cap, prune) are confirm-gated too.  The write runner / network are
NEVER executed live — tests inject fakes and conftest blocks the real spawn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)
from textual import work

from club3090_tui_core.registry import VariantRow
from club3090_tui_core.widgets.live_pane import LivePane

from .data import (
    ActionPlan,
    BenchRow,
    ByoResult,
    CatalogEntry,
    ContainerInfo,
    DoctorReport,
    EstateState,
    EvidenceReport,
    EvidenceTag,
    Measurement,
    MeasureVsBar,
    OptimizerReport,
    PowerCapState,
    PromoteScaffold,
    ReconcileResult,
    Scene,
    _bench_row_matches,
    _canon_engine_family,
    measurement_from_explain_columns,
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


def _status_glyph(status: str) -> str:
    return _STATUS_GLYPH.get(status.lower(), status)


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

    # Surface-threaded (R3b-1): the consumer help OMITS every producer-lane token
    # (the [3] Bring & Validate mode, [P] Promote, [v] Evaluate, the lane section)
    # so it shows only the consumer affordances (Run + Operate + share-back).  The
    # producer help INCLUDES the lane section.

    # The mode line: consumer sees Run + Operate; producer additionally sees [3].
    _MODE_LINE_CONSUMER = "  [cyan]1[/cyan]  Run    [cyan]2[/cyan]  Operate"
    _MODE_LINE_PRODUCER = (
        "  [cyan]1[/cyan]  Run    [cyan]2[/cyan]  Operate    "
        "[cyan]3[/cyan]  Bring & Validate"
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
            "  [cyan]/[/cyan]  Filter (Run · Catalog)",
            "  [cyan]e[/cyan]  Explain selected slug (Run · Catalog — incl. cross-rig benchmarks)",
            "  [cyan]⏎[/cyan]  Primary action (serve / switch scene / run step / open report)",
            "  [cyan]?[/cyan]  This help        [cyan]q[/cyan]  Quit",
            "",
            "[bold]Run · Catalog[/bold]",
            "  [cyan]⏎[/cyan] serve selected slug (reconcile-gated confirm; F to Force the teardown)",
            "  [cyan]d[/cyan] set-default   [cyan]D[/cyan] clear-default",
            "  [cyan]O[/cyan] ▸ Optimize for my card (v0.10.0 seam — not available yet)",
            "[bold]Operate · Orchestration[/bold]",
            "  [cyan]o[/cyan] stop all   [cyan]c[/cyan] power-cap on/off   [cyan]w[/cyan] cap sweep   [cyan]p[/cyan] prune images   (all gated)",
            "[bold]Operate · Containers[/bold]",
            "  [cyan]l[/cyan] logs   [cyan]t[/cyan] top (read)   [cyan]s[/cyan] restart   [cyan]x[/cyan] stop   [cyan]X[/cyan] rm   (writes gated)",
            "[bold]Operate · Doctor[/bold]",
            "  read-only — health + diagnose-estate + diagnose-profile cards ([cyan]r[/cyan] refreshes)",
        ]
        # Producer-only lane section — OMITTED on consumer (clean consumer help).
        if producer:
            parts.append(self._LANE_SECTION.rstrip("\n"))
        parts.extend([
            "",
            "[bold]Share back[/bold] (Run + Operate — lightweight, no surface switch)",
            "  [cyan]R[/cyan] rig report — paste-ready rig/bench snapshot (read · no network)",
            "  [cyan]B[/cyan] submit bench — submit the latest benched result (Operate · gated · never auto)",
            "  [cyan]![/cyan] report a problem — paste-ready issue from the failure context (read · surfaced at a failed serve)",
            "",
            "[bold]Safety — the reconcile gate[/bold]",
            "",
            "  Every write (serve, scene-switch, estate-down, container restart/stop/rm,",
            "  power-cap, prune, submit-bench) goes through a confirm modal.  GPU-claiming",
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
    CatalogPane #catalog-hint {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Loading catalog…", id="catalog-status")
        yield Input(placeholder="filter slug / engine / model / status…", id="catalog-filter")
        table: DataTable = DataTable(id="catalog-table", zebra_stripes=True)
        table.cursor_type = "row"
        yield table
        yield Label(
            "[dim]\\[/] filter   \\[⏎] serve   \\[e] explain   "
            "\\[d] set-default   \\[D] clear-default[/dim]",
            id="catalog-hint",
        )

    def on_mount(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        # Fold 3: the TPS / 8-pack columns are OUR-RIG measurements (cross-rig
        # rows live in the explain drill-down) — label them so cross-rig ambiguity
        # is gone now that the standalone Benchmarks tab is retired.
        table.add_columns("slug", "engine", "fit", "ctx", "TPS (our rig)", "8pk (our rig)", "status", "source")
        # Full enriched catalog, and the current filter substring.
        self._entries: list[CatalogEntry] = []
        self._filter: str = ""

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
        self._render_rows()

    def _render_rows(self) -> None:
        status_label = self.query_one("#catalog-status", Label)
        table = self.query_one("#catalog-table", DataTable)
        table.clear()

        rows = self._filtered_entries()
        for e in rows:
            # source provenance — flag a coarse markdown scrape so a measurement
            # from BENCHMARKS.md is never mistaken for a structured record.
            meas_src = e.measurement.source
            tps = e.measurement.tps_label
            if meas_src == "benchmarks.md" and tps != "—":
                tps = f"{tps}*"
            table.add_row(
                e.slug,
                e.engine,
                e.fit.glyph,
                e.ctx_label or "—",
                tps,
                e.measurement.quality_label,
                _status_glyph(e.status),
                e.source,
            )

        if self._filter:
            status_label.update(
                f"{len(rows)} / {len(self._entries)} variants  ·  filter: {self._filter!r}"
            )
        else:
            star = "  ([dim]*[/dim] = BENCHMARKS.md scrape)" if self._has_md_scrape() else ""
            status_label.update(f"{len(self._entries)} variants loaded from registry{star}")

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

    def _has_md_scrape(self) -> bool:
        return any(e.measurement.source == "benchmarks.md" for e in self._entries)

    def _filtered_entries(self) -> list[CatalogEntry]:
        if not self._filter:
            return self._entries
        f = self._filter.lower()
        out: list[CatalogEntry] = []
        for e in self._entries:
            hay = f"{e.slug} {e.engine} {e.model} {e.status} {e.source}".lower()
            if f in hay:
                out.append(e)
        return out

    def set_filter(self, text: str) -> None:
        self._filter = (text or "").strip()
        self._render_rows()

    def selected_entry(self) -> Optional[CatalogEntry]:
        """The CatalogEntry under the table cursor, or None."""
        table = self.query_one("#catalog-table", DataTable)
        rows = self._filtered_entries()
        idx = table.cursor_row
        if 0 <= idx < len(rows):
            return rows[idx]
        return None

    def toggle_filter(self) -> None:
        inp = self.query_one("#catalog-filter", Input)
        if "visible" in inp.classes:
            inp.remove_class("visible")
            self.query_one("#catalog-table", DataTable).focus()
        else:
            inp.add_class("visible")
            inp.focus()

    def close_filter_if_open(self) -> bool:
        """Esc/cancel: hide + clear the filter and refocus the table. Returns
        True if a filter was actually open (so the app can swallow the Esc)."""
        inp = self.query_one("#catalog-filter", Input)
        if "visible" in inp.classes:
            inp.remove_class("visible")
            inp.value = ""
            self.set_filter("")
            self.query_one("#catalog-table", DataTable).focus()
            return True
        return False


# ── Explain detail modal ────────────────────────────────────────────────────────


class ExplainScreen(ModalScreen):
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

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ── Run · Bring-your-own ────────────────────────────────────────────────────


class ByoPane(Container):
    """Bring-your-own tab: real HF fit-check via pull.sh --dry-run --json."""

    DEFAULT_CSS = """
    ByoPane {
        height: 1fr;
        padding: 1 2;
    }
    ByoPane #byo-heading {
        text-style: bold;
        margin-bottom: 1;
    }
    ByoPane #byo-input-row {
        height: 3;
        margin-bottom: 1;
    }
    ByoPane #byo-url-input {
        width: 1fr;
    }
    ByoPane #byo-profile-input {
        width: 28;
        margin-left: 1;
    }
    ByoPane #byo-fit-btn {
        width: 14;
        margin-left: 1;
    }
    ByoPane #byo-result-card {
        border: solid $primary;
        padding: 1 2;
        margin-top: 1;
        height: auto;
    }
    ByoPane #byo-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Bring-your-own HF model", id="byo-heading")
        with Horizontal(id="byo-input-row"):
            yield Input(
                placeholder="org/Model  (e.g. unsloth/Qwen3-27B-abliterated-GGUF)",
                id="byo-url-input",
            )
            yield Input(
                placeholder="profile-like (vllm/dual)",
                value="vllm/dual",
                id="byo-profile-input",
            )
            yield Button("Fit-check", id="byo-fit-btn", variant="primary")
        yield Static(
            "[dim]Enter an HF repo + a profile-like slug, then Fit-check.\n"
            "Runs pull.sh --dry-run (Path B — evaluates only, never downloads).[/dim]",
            id="byo-result-card",
        )
        yield Label(
            "[dim]Routes:  A = new curated profile   ·   B = serve-locally   ·   "
            "C = reuse a sibling compose + swap weights\n"
            "\\[O] ▸ Optimize for my card (v0.10.0 seam)\n"
            "[dim](Promote to the catalog lives in the producer Bring & Validate "
            "lane — c3 --contribute)[/dim][/dim]",
            id="byo-hint",
        )

    def set_checking(self, repo: str) -> None:
        self.query_one("#byo-result-card", Static).update(
            f"[dim]Checking[/dim] [cyan]{repo}[/cyan] [dim](pull.sh --dry-run --json)…[/dim]"
        )

    def populate(self, res: ByoResult) -> None:
        # The verdict-card render is shared with the producer lane's ① Bring stage
        # (LaneBringPane) — see _byo_result_text (defined below near the lane panes).
        self.query_one("#byo-result-card", Static).update(_byo_result_text(res))


# ── Confirm modal (used for serve + scene + container writes) ────────────────────


class ConfirmActionScreen(ModalScreen):
    """The reconcile-gated confirm modal (design §7 #8 / §3.2).

    Shows the plan + the FRESH reconcile result (what this write would collide
    with / tear down), and only on confirm dispatches the write through the
    app's gated executor.  When the gate is unsafe, the primary action is
    disabled and the user must surface the explicit Force override (which is
    routed back to the app with a reason).
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

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, plan: ActionPlan, *, on_confirm=None, **kwargs):
        super().__init__(**kwargs)
        self._plan = plan
        self._reconcile: Optional[ReconcileResult] = None
        # Optional alternate commit path.  Default (None) → the app's gated
        # ``dispatch_action`` (execute_action).  Set for launches that don't go
        # through execute_action — notably validation runs, which stream via
        # ``run_validation`` into the Run LivePane and never claim a GPU.  The
        # callback receives the (possibly force-reissued) plan.
        self._on_confirm = on_confirm

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Confirm · {self._plan.description}", classes="confirm-title")
            yield Static("Re-running reconcile gate (fresh detect)…", id="confirm-body")
            with Horizontal(id="confirm-btn-row"):
                yield Button("⏎ Confirm", id="confirm-ok-btn", variant="success", disabled=True)
                yield Button("F Force", id="confirm-force-btn", variant="warning", disabled=True)
                yield Button("Esc Cancel", id="confirm-cancel-btn")

    def on_mount(self) -> None:
        # Re-run the gate (fresh detect) before enabling any commit button.
        self.app.run_reconcile_for_modal(self, self._plan)  # type: ignore[attr-defined]

    def set_reconcile(self, rec: ReconcileResult) -> None:
        """Render the reconcile verdict + enable the appropriate commit path."""
        self._reconcile = rec
        body = self.query_one("#confirm-body", Static)
        ok_btn = self.query_one("#confirm-ok-btn", Button)
        force_btn = self.query_one("#confirm-force-btn", Button)

        lines: list[str] = [f"  [bold]Command[/bold]  {' '.join(self._plan.cmd)}"]
        wanted = ", ".join(str(g) for g in rec.pending_gpus) if rec.pending_gpus else "—"
        lines.append(f"  [bold]GPUs[/bold]     {wanted}")

        if rec.safe:
            lines.append("")
            lines.append("  [green]● gate clear[/green] — nothing live overlaps the requested GPUs.")
            lines.append("  [dim]⏎ Confirm to launch (streams below) · Esc Cancel[/dim]")
            ok_btn.disabled = False
            force_btn.disabled = True
            ok_btn.focus()
        else:
            lines.append("")
            lines.append("  [yellow]⚠ this write would tear down / collide with:[/yellow]")
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
            lines.append("  [dim]Confirm is disabled — F to FORCE this teardown (override).[/dim]")
            ok_btn.disabled = True
            force_btn.disabled = False
            force_btn.focus()

        body.update("\n".join(lines))

    # ── button / key handlers ────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-ok-btn":
            self._commit(force=False)
        elif event.button.id == "confirm-force-btn":
            self._commit(force=True)
        elif event.button.id == "confirm-cancel-btn":
            self.action_cancel()

    def on_key(self, event) -> None:
        if event.key == "f":
            force_btn = self.query_one("#confirm-force-btn", Button)
            if not force_btn.disabled:
                event.stop()
                self._commit(force=True)
        elif event.key == "enter":
            ok_btn = self.query_one("#confirm-ok-btn", Button)
            if not ok_btn.disabled:
                event.stop()
                self._commit(force=False)

    def _commit(self, *, force: bool) -> None:
        plan = self._plan
        if force and not plan.force:
            # Re-issue the plan as a forced one (with a surfaced reason) so the
            # executor's force path is taken explicitly — never silently.
            plan = ActionPlan(
                kind=plan.kind,
                cmd=_with_force(plan),
                description=plan.description + " (FORCED)",
                is_write=plan.is_write,
                requires_reconcile=plan.requires_reconcile,
                force=True,
                force_reason="user accepted teardown via Force override",
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
    OperateOrchPane #services-heading {
        text-style: bold;
        padding: 0 1;
        margin: 0 1 0 1;
    }
    OperateOrchPane #services-strip {
        padding: 0 1;
        margin: 0 1 1 1;
    }
    OperateOrchPane #powercap-heading {
        text-style: bold;
        padding: 0 1;
        margin: 0 1 0 1;
    }
    OperateOrchPane #powercap-strip {
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
        with ScrollableContainer(id="orch-scroll"):
            with Container(classes="gpu-card", id="gpu0-card"):
                yield Label("GPU0", classes="gpu-card-title")
                yield Static("[dim]querying nvidia-smi…[/dim]", id="gpu0-bar")
            with Container(classes="gpu-card", id="gpu1-card"):
                yield Label("GPU1", classes="gpu-card-title")
                yield Static("[dim]querying nvidia-smi…[/dim]", id="gpu1-bar")
            yield Static("[dim]reading health.sh…[/dim]", id="doctor-line")
            yield Label("Scenes  [dim](⏎ to switch — gated)[/dim]", id="scene-heading")
            scene_table: DataTable = DataTable(
                id="scene-table", zebra_stripes=True, show_cursor=True
            )
            scene_table.cursor_type = "row"
            yield scene_table
            yield Label("Services", id="services-heading")
            yield Static("[dim]reading estate…[/dim]", id="services-strip")
            yield Label("Power cap", id="powercap-heading")
            yield Static("[dim]reading power-cap status…[/dim]", id="powercap-strip")
            yield Label(
                "[dim]\\[⏎] switch scene (gated)   \\[o] stop all (gated)   "
                "\\[c] cap on/off (gated)   \\[w] cap sweep (gated)   "
                "\\[p] prune images (gated)[/dim]",
                id="orch-hint",
            )

    def on_mount(self) -> None:
        t = self.query_one("#scene-table", DataTable)
        t.add_columns("Scene", "Group", "GPUs", "Services")
        self._scenes: list[Scene] = []

    # ── data ────────────────────────────────────────────────────────────────────

    def populate(self, state: EstateState) -> None:
        self._populate_gpus(state)
        self._populate_doctor(state)
        self._populate_scenes(state.scenes)
        self._populate_services(state)

    def populate_power_cap(self, st: PowerCapState) -> None:
        strip = self.query_one("#powercap-strip", Static)
        if st.error and not st.gpus:
            strip.update(f"[dim]{st.error}[/dim]")
            return
        bits: list[str] = []
        for g in st.gpus:
            lim = f"{g.limit_w:.0f}W" if g.limit_w is not None else "—"
            dflt = f"{g.default_w:.0f}W" if g.default_w is not None else "—"
            capped = (
                g.limit_w is not None and g.default_w is not None and g.limit_w < g.default_w
            )
            tag = "[yellow]capped[/yellow]" if capped else "[green]uncapped[/green]"
            bits.append(f"GPU{g.index} {lim}/{dflt} {tag}")
        strip.update("  " + "   ·   ".join(bits) if bits else "[dim]no GPUs[/dim]")

    def _populate_gpus(self, state: EstateState) -> None:
        for i, bar_id, title_id in ((0, "#gpu0-bar", "#gpu0-card"), (1, "#gpu1-bar", "#gpu1-card")):
            bar = self.query_one(bar_id, Static)
            gpu = next((g for g in state.gpus if getattr(g, "index", -1) == i), None)
            if gpu is None:
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
            bar.update(
                f"  {bar_str}  {used / 1024:.1f} / {total / 1024:.1f} GiB · {pct}%\n"
                f"  {pwr:.0f} / {pwr_lim:.0f} W · {temp}°C · util {util}%"
            )

    def _populate_doctor(self, state: EstateState) -> None:
        dr = state.doctor
        line = self.query_one("#doctor-line", Static)
        if not dr.reachable:
            line.update("[red]○[/red] API not reachable")
            return
        glyph = "[green]●[/green]" if dr.serving else "[yellow]○[/yellow]"
        line.update(f"{glyph} {dr.summary}")

    def _populate_scenes(self, scenes: list[Scene]) -> None:
        self._scenes = list(scenes)
        t = self.query_one("#scene-table", DataTable)
        t.clear()
        for s in scenes:
            svc = ", ".join(s.services[:3]) + ("…" if len(s.services) > 3 else "")
            t.add_row(s.name, s.group, s.gpus or "—", svc or "—")

    def _populate_services(self, state: EstateState) -> None:
        strip = self.query_one("#services-strip", Static)
        # Services come from the running-container view + scene catalog.
        svc_names: list[str] = []
        for c in state.containers:
            if c.kind == "service":
                svc_names.append(c.name)
        if not svc_names:
            ae = (state.estate_report or {}).get("active_estate") or {}
            insts = ae.get("instances") or []
            if insts:
                strip.update(
                    "  "
                    + "   ".join(
                        f"[green]●[/green] {i.get('name', '?')} (GPU {i.get('gpus', [])})"
                        for i in insts
                    )
                )
                return
            strip.update("[dim]no stack services detected[/dim]")
            return
        strip.update("  " + "   ".join(f"[green]●[/green] {n}" for n in svc_names))

    def selected_scene(self) -> Optional[Scene]:
        t = self.query_one("#scene-table", DataTable)
        idx = t.cursor_row
        if 0 <= idx < len(self._scenes):
            return self._scenes[idx]
        return None


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
                yield LivePane(id="drill-logs")
            with TabPane("Top", id="drill-tab-stats"):
                yield Static("[dim]select a container — docker top loads automatically[/dim]", id="drill-stats")
            with TabPane("Config", id="drill-tab-config"):
                yield Static("[dim]select a container — config loads automatically[/dim]", id="drill-config")
        yield Label(
            "[dim]detail auto-loads on select · \\[l] logs   \\[t] top   \\[s] restart (gated)   "
            "\\[x] stop (gated)   \\[X] rm (reconcile-gated)[/dim]",
            id="containers-hint",
        )

    def on_mount(self) -> None:
        t = self.query_one("#containers-table", DataTable)
        t.add_columns("Name", "Kind", "Engine", "Port", "Slug")
        self._containers: list[ContainerInfo] = []

    def populate(self, containers: list[ContainerInfo]) -> None:
        self._containers = list(containers)
        t = self.query_one("#containers-table", DataTable)
        t.clear()
        if not containers:
            t.add_row("[dim]no stack containers[/dim]", "—", "—", "—", "—")
            return
        for c in containers:
            t.add_row(
                c.name,
                c.kind,
                c.engine or "—",
                str(c.host_port) if c.host_port else "—",
                c.slug or "—",
            )

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
            f"  [bold]Slug[/bold]       {con.slug or '[dim]unmatched[/dim]'}",
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
        yield Static(_TUNE_GOTCHAS, id="run-gotchas")
        yield LivePane(id="run-output")
        yield Label(
            "[dim]\\[⏎] launch selected (heavy — confirm) · streams below[/dim]",
            id="run-hint",
        )

    def on_mount(self) -> None:
        t = self.query_one("#run-ladder-table", DataTable)
        t.add_columns("step", "kind", "what it checks")
        # (kind) in cursor order — the selected row maps back to a run kind.
        self._kinds: list[str] = []
        for kind, label, blurb in _RUN_LADDER:
            t.add_row(f"[cyan]▷[/cyan] {label}", "ladder", blurb)
            self._kinds.append(kind)
        for kind, label, blurb in _RUN_EXTRAS:
            t.add_row(f"[cyan]▷[/cyan] {label}", "extra", blurb)
            self._kinds.append(kind)

    def selected_kind(self) -> Optional[str]:
        t = self.query_one("#run-ladder-table", DataTable)
        idx = t.cursor_row
        if 0 <= idx < len(self._kinds):
            return self._kinds[idx]
        return None


class DoctorPane(Container):
    """Operate / Doctor tab: real health / diagnose-estate / diagnose-profile
    cards from ``CockpitData.doctor()``.  Mode-agnostic Doctor surface — R2a
    moved it out of Validate into Operate (it reports live state, not a
    validation artifact).

    The health line also updates live from the Operate estate poll
    (``populate``); the estate + profile cards fill from the dedicated
    ``doctor()`` read (``r`` / on entering Operate) since diagnose-estate /
    diagnose-profile are heavier reads than the per-poll health probe."""

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
            yield Label("Doctor  [dim](r refreshes — runs the three diagnose reads)[/dim]", id="doctor-heading")
            with Container(classes="doctor-card", id="doctor-card-health"):
                yield Label("health.sh", classes="doctor-card-title")
                yield Static("[dim]reading health.sh…[/dim]", id="doctor-health-body")
            with Container(classes="doctor-card", id="doctor-card-estate"):
                yield Label("diagnose-estate", classes="doctor-card-title")
                yield Static("[dim]reading diagnose-estate…[/dim]", id="doctor-estate-body")
            with Container(classes="doctor-card", id="doctor-card-profile"):
                yield Label("diagnose-profile", classes="doctor-card-title")
                yield Static("[dim]reading diagnose-profile…[/dim]", id="doctor-profile-body")
            yield Label(
                "[dim]all three legs are READ-only (safe to run live)[/dim]",
                id="doctor-hint",
            )

    def populate(self, state: EstateState) -> None:
        """Live health line from the Operate estate poll (the cheap per-poll probe)."""
        self._render_health(state.doctor)

    def _render_health(self, dr) -> None:
        body = self.query_one("#doctor-health-body", Static)
        if not dr.reachable:
            body.update("[red]✗[/red]  API not reachable")
            return
        glyph = "[green]✓[/green]" if dr.serving else "[yellow]○[/yellow]"
        line = f"{glyph}  {dr.summary}"
        body.update(line)

    def populate_report(self, report: DoctorReport) -> None:
        """Full Doctor read — health + diagnose-estate + diagnose-profile cards."""
        self._render_health(report.health)
        self._render_estate(report.estate)
        self._render_profile(report.profile)

    def _render_estate(self, est) -> None:
        body = self.query_one("#doctor-estate-body", Static)
        if est.error:
            body.update(f"[red]✗[/red]  {est.error}")
            return
        verdict_color = {"GREEN": "green", "AMBER": "yellow", "YELLOW": "yellow", "RED": "red"}.get(
            est.summary.upper(), "dim"
        )
        lines = [
            f"  {est.summary_glyph} [{verdict_color}]{est.summary or '—'}[/{verdict_color}]"
            f"  ([{'green' if est.valid else 'red'}]{'valid' if est.valid else 'invalid'}[/])",
            f"  instances   {est.instances_valid}/{est.instance_count} fit"
            f"   ·   cross-checks {'[green]ok[/green]' if est.cross_checks_ok else '[red]fail[/red]'}",
            f"  estate file [dim]{est.estate_file or '—'}[/dim]"
            f"   ·   live {'yes' if est.live else 'no'}",
        ]
        body.update("\n".join(lines))

    def _render_profile(self, tri) -> None:
        body = self.query_one("#doctor-profile-body", Static)
        if tri is None:
            body.update("[dim]no target slug — serve a model or pick one in Run to triage[/dim]")
            return
        if tri.error and not tri.steps:
            body.update(f"[red]✗[/red]  {tri.error}")
            return
        verdict_color = {"GREEN": "green", "AMBER": "yellow", "YELLOW": "yellow", "RED": "red"}.get(
            tri.summary.upper(), "dim"
        )
        lines = [
            f"  [bold]{tri.slug}[/bold]   {tri.summary_glyph} "
            f"[{verdict_color}]{tri.summary or '—'}[/{verdict_color}]"
            f"   ({tri.passed}/{len(tri.steps)} steps)",
        ]
        step_glyph = {"passed": "[green]✓[/green]", "failed": "[red]✗[/red]", "warn": "[yellow]⚠[/yellow]"}
        for s in tri.steps:
            g = step_glyph.get(s.status, "·")
            lines.append(f"    {g} [{s.num}/{s.total}] {s.name}")
        body.update("\n".join(lines))


class ValidateEvidencePane(Container):
    """Validate / Evidence tab: real ``results/rebench/<tag>/`` run list from
    ``evidence_list()``; ``⏎`` opens the paste-ready report (``evidence_report``)
    in a modal (reuses the history_view pattern), ``s`` stages the gated
    submit-to-localmaxxing for the selected tag (confirm modal; never auto)."""

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
        yield Label(
            "[dim]\\[⏎] open report   \\[m] vs catalog bar   \\[s] submit to localmaxxing (gated · never auto)[/dim]",
            id="evidence-hint",
        )

    def on_mount(self) -> None:
        t = self.query_one("#evidence-table", DataTable)
        t.add_columns("tag", "date", "report", "internal", "soak", "TL;DR")
        self._tags: list[EvidenceTag] = []

    def populate(self, tags: list[EvidenceTag]) -> None:
        status = self.query_one("#evidence-status", Label)
        t = self.query_one("#evidence-table", DataTable)
        t.clear()
        self._tags = list(tags)
        if not tags:
            status.update("[dim]no runs under results/rebench/[/dim]")
            t.add_row("[dim]—[/dim]", "—", "—", "—", "—", "—")
            return
        for et in tags:
            yn = lambda b: "[green]✓[/green]" if b else "[dim]·[/dim]"
            tldr = (et.tldr[:48] + "…") if len(et.tldr) > 49 else (et.tldr or "—")
            t.add_row(et.tag, et.date or "—", yn(et.has_report), yn(et.has_internal), yn(et.has_soak), tldr)
        status.update(f"{len(tags)} run tag(s) under results/rebench/")

    def selected_tag(self) -> Optional[EvidenceTag]:
        t = self.query_one("#evidence-table", DataTable)
        idx = t.cursor_row
        if 0 <= idx < len(self._tags):
            return self._tags[idx]
        return None


# ── Evidence report modal (reuses the history_view read pattern) ─────────────────


class EvidenceReportScreen(ModalScreen):
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
    ]

    def __init__(self, tag: str, **kwargs):
        super().__init__(**kwargs)
        self._tag = tag

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Report · {self._tag}", classes="evidence-report-title")
            with ScrollableContainer(id="evidence-report-scroll"):
                yield Static("Generating report (rebench-report.py — reads results)…", id="evidence-report-body")
            yield Label("[dim]Esc to close[/dim]")

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


class ShareBackReportScreen(ModalScreen):
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
            yield Label("[dim]Esc to close · copy the text above to share[/dim]")

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

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ── Phase 5 · Promote-to-catalog scaffold preview modal (design §3.5b) ────────────


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

    BINDINGS = [
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

    def on_key(self, event) -> None:
        if event.key == "enter":
            btn = self.query_one("#promote-stage-btn", Button)
            if not btn.disabled:
                event.stop()
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

    BINDINGS = [
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

    def on_key(self, event) -> None:
        if event.key == "enter":
            event.stop()
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
# The producer lane (mode 2) presents the ADDING_MODELS stage machine as an
# ORDERED, numbered pipeline: ① Bring → ② Serve → ③ Gate → ④ Measure → ⑤ Promote.
# It reuses the existing TabbedContent pattern (lighter than a full wizard widget)
# with numbered tab labels so it reads as an ordered pipeline.  ① Bring REUSES the
# byo_check fit-check; ③ Gate is the existing ValidateRunPane ladder; ④ Measure is
# the existing ValidateEvidencePane; ⑤ Promote hosts the [P] promote action.


class LaneBringPane(Container):
    """① Bring — the producer lane's own fit-check entry.

    REUSES ``byo_check`` (pull.sh --dry-run --json → ByoResult: supported? fits?
    the swap_path route) exactly like Run · BYO, but as the lane's first stage:
    paste an HF repo / slug, Fit-check, read the route + sibling_slug + quant_match.
    Distinct widget IDs from Run · ByoPane so both can coexist (the consumer
    Run · BYO 'run-another' stays as-is; this is the producer lane's Bring entry).
    The cached ``_last_byo`` it produces feeds ② Serve and ⑤ Promote."""

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
        width: 28;
        margin-left: 1;
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
            yield Input(
                placeholder="profile-like (vllm/dual)",
                value="vllm/dual",
                id="lane-bring-profile-input",
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


MODES = [
    ("Run", "1"),
    ("Operate", "2"),
    # R3b-1: mode 2 is the producer "Bring & Validate" lane (key 3, producer-only).
    # The mode index stays 2 (NO renumber) — only the LABEL changed from "Validate".
    ("Bring & Validate", "3"),
]

# Per-mode primary action (what ⏎ does), by mode index.
PRIMARY_ACTIONS = ["Serve", "Switch scene", "Run stage"]


class RailStatus(Static):
    """Persistent left-rail status card — mirrors c3t's TargetPane.

    Wired in Phase 3: shows the live detect / doctor read.  Until the first
    estate poll completes it shows a 'detecting…' placeholder."""

    PLACEHOLDER = (
        "[bold]Estate[/bold]\n"
        "\n"
        "[dim]detecting…[/dim]\n"
        "\n"
        "[dim]press 2 (Operate) to poll[/dim]"
    )

    def __init__(self, **kwargs):
        super().__init__(self.PLACEHOLDER, **kwargs)

    def update_from_state(self, state: EstateState) -> None:
        lines: list[str] = ["[bold]Estate[/bold]", ""]
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
            lines.append(f"model   {state.matched_slug}")
        elif state.target is not None and getattr(state.target, "model", ""):
            lines.append(f"model   {state.target.model}")
        dr = state.doctor
        if dr.reachable:
            glyph = "[green]●[/green]" if dr.serving else "[yellow]○[/yellow]"
            lines.append(f"{glyph} {dr.summary}")
        else:
            lines.append("[red]○[/red] not reachable")
        self.update("\n".join(lines))


class ModeSwitcher(Static):
    """Left-rail mode selector — navigation is driven by CockpitApp via the
    1–3 digit bindings; this is the visual highlight."""

    DEFAULT_CSS = """
    ModeSwitcher {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
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

    def __init__(self, *, surface: str = "consumer", **kwargs):
        super().__init__("", **kwargs)
        self._active = 0
        # Surface-aware (R3a): the CONSUMER surface renders Run + Operate only;
        # PRODUCER additionally renders Validate (the Bring & Validate lane, key
        # 3).  Keys 1/2 = Run/Operate on both; key 3 is only meaningful on
        # producer (gated by check_action's surface gate on consumer).
        #
        # R4: all THREE mode Labels are always composed; the consumer surface just
        # HIDES the third (.mode-hidden) so the runtime Contribute toggle can show
        # it again without an async re-mount (set_surface flips the class only).
        self._surface = surface if surface in ("consumer", "producer") else "consumer"

    @property
    def _modes(self) -> list[tuple[str, str]]:
        """The modes VISIBLE for this surface — all three on producer, the first
        two (Run + Operate) on consumer.  (All three Labels are always mounted;
        this is the visible slice — see set_surface.)"""
        return list(MODES) if self._surface == "producer" else list(MODES[:2])

    def compose(self) -> ComposeResult:
        yield Label("Modes", classes="mode-title")
        # Always compose all three mode Labels; hide the producer-only third on
        # the consumer surface so the runtime toggle can reveal it class-only.
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
        """Re-render the rail for a runtime surface change (R4 Contribute door).

        Consumer shows Run + Operate (2 items); producer additionally shows Bring
        & Validate (3 items).  All three mode Labels are always mounted, so this
        is a pure class flip — show/hide the producer-only third Label via
        ``.mode-hidden`` (no async re-mount, which raced the pilot under the
        headless harness)."""
        new = surface if surface in ("consumer", "producer") else "consumer"
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


# ── Main application ──────────────────────────────────────────────────────────────


class CockpitApp(App):
    """club3090 serve cockpit — all three modes (Run · Operate · Validate) wired to the live data layer."""

    TITLE = "club3090 cockpit"
    SUB_TITLE = "wired"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("question_mark", "help", "Help", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        # Sub-tab cycle — shown only in modes that have sub-tabs (check_action gates).
        Binding("left_square_bracket", "prev_subtab", "Prev tab", show=False),
        Binding("right_square_bracket", "next_subtab", "Next tab", show=False),
        # Context-sensitive — check_action enables/shows them only in the right mode.
        Binding("slash", "filter_catalog", "Filter", show=False),
        Binding("e", "explain", "Explain", show=False),
        Binding("1", "mode_run", "Run", show=True),
        Binding("2", "mode_operate", "Operate", show=True),
        Binding("3", "mode_validate", "Validate", show=True),
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
        # Operate · Orchestration — power cap + prune (gated rig writes).
        Binding("c", "power_cap_toggle", "Cap on/off", show=False),
        Binding("w", "power_cap_sweep", "Cap sweep", show=False),
        Binding("p", "prune_images", "Prune", show=False),
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
        # Phase R / R4 — the in-app "Contribute" DOOR: toggle consumer ↔ producer
        # at runtime + persist the choice for next launch.  Always available (it
        # is the consumer's opt-in into producer mode) — NOT in _PRODUCER_ONLY.
        Binding("C", "toggle_contribute", "Contribute mode", show=False),
    ]

    CSS = """
    #main-layout {
        height: 1fr;
    }
    #left-rail {
        width: 32;
        height: 1fr;
        padding: 0 0;
    }
    #rail-status {
        width: 1fr;
        height: 1fr;
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
    """

    # ── Dynamic binding visibility ─────────────────────────────────────────────────

    # Actions that are always active regardless of mode or focused widget.
    _ALWAYS_ON: frozenset[str] = frozenset({
        "quit", "help", "refresh",
        "mode_run", "mode_operate", "mode_validate",
        "primary_action",
        # The Contribute door (R4) is the consumer's opt-in INTO producer mode,
        # so it must stay reachable on the consumer surface — always-on, NOT in
        # _PRODUCER_ONLY (a consumer that can't toggle could never contribute).
        "toggle_contribute",
    })

    # Context key → (modes, subtabs) where it should be enabled.
    # modes: set of _active_mode integers.  subtabs: set of active tab IDs, or
    # None meaning "any sub-tab in those modes" (used for whole-mode keys).
    # The sub-tab cycle keys are handled separately below.
    _CONTEXT_KEYS: dict[str, tuple[set[int], Optional[set[str]]]] = {
        # Run / Catalog only
        "filter_catalog":   ({0}, {"tab-catalog"}),  # Run · Catalog
        "explain":          ({0}, None),          # Run (any sub-tab — no-ops on BYO, harmless)
        "set_default":      ({0}, None),          # Run · Catalog (guards inside action)
        "clear_default":    ({0}, None),          # Run · Catalog
        # R3b-1: [P] promote + [v] evaluate relocated OUT of consumer modes INTO
        # the producer Bring & Validate lane (mode 2).  Both producer-gated.
        "promote_catalog":  ({2}, None),          # Bring & Validate lane (⑤ Promote)
        "evaluate_target":  ({2}, None),          # Bring & Validate lane (the c3t hook)
        "serve_untested":   ({2}, {"tab-serve"}), # Bring & Validate lane (② Serve)
        # R3b-2: [m] vs-bar on ④ Measure (tab-evidence); [F] full battery on
        # ③ Gate (tab-run — the lane's gate stage hosts the heavy validation).
        "measure_vs_bar":   ({2}, {"tab-evidence"}),
        "full_report":      ({2}, {"tab-run"}),
        "optimize_card":    ({0}, None),          # Run
        # Operate · Orchestration
        "estate_off":       ({1}, {"tab-orchestration"}),
        "power_cap_toggle": ({1}, {"tab-orchestration"}),
        "power_cap_sweep":  ({1}, {"tab-orchestration"}),
        "prune_images":     ({1}, {"tab-orchestration"}),
        # Operate · Containers
        "container_logs":   ({1}, {"tab-containers"}),
        # [s] restart only on Operate (any tab, action guards internally) +
        # [s] submit on Validate·Evidence; no sub-tab constraint at this level.
        "s_key":            ({1, 2}, None),  # Containers (restart) + Evidence (submit)
        "container_stop":   ({1}, {"tab-containers"}),
        "container_rm":     ({1}, {"tab-containers"}),
        # [t] only has the Containers (docker top) role now — the Benchmarks tab
        # and its sort-cycle are gone (folded into Run); not wired on Run rows.
        "context_t":        ({1}, {"tab-containers"}),
        # Phase R / R2b — consumer share-back (CONSUMER-resident — NOT producer-
        # gated; absent from _PRODUCER_ONLY so they work on the default surface):
        #   rig_report   — Run + Operate (a rig/bench snapshot is meaningful from
        #                  either the catalog or the live estate, any sub-tab).
        #   submit_bench — Operate only (you submit measured results once a bench
        #                  exists; Operate is where the live estate / evidence is).
        #   report_problem — Run + Operate (surfaced AT a failed serve in Run, and
        #                  reachable while operating; any sub-tab).
        "rig_report":       ({0, 1}, None),
        "submit_bench":     ({1}, None),
        "report_problem":   ({0, 1}, None),
    }

    # Producer-only actions — hidden on the consumer surface (R3a makes the
    # consumer/producer split REAL).  The gate fires in check_action BEFORE
    # _ALWAYS_ON, so listing ``mode_validate`` here hides the ENTIRE producer
    # "Bring & Validate" lane (its mode switch + ladder + evidence) on the
    # consumer surface; the producer surface (``c3 --contribute``) falls through
    # to the normal context result.  ``promote_catalog`` ([P]) is a producer
    # activity still surfaced in Run · Catalog today — hidden on consumer, still
    # reachable on producer (it relocates into the lane in R3b).
    #   NOT gated: the consumer share-back (rig_report / submit_bench /
    #   report_problem) is CONSUMER-resident and stays reachable; evaluate_target
    #   stays in Operate for now (R3b relocates it).
    #   R3b-1: [v] evaluate_target + [serve_untested] (② Serve) joined the lane,
    #   so they are producer-only too ([P] promote was already here).
    #   R3b-2: [m] measure_vs_bar (④ Measure, a READ) + [F] full_report (③ Gate,
    #   the ~43-min battery) are producer-lane actions too.
    _PRODUCER_ONLY: frozenset[str] = frozenset({
        "mode_validate", "promote_catalog", "evaluate_target", "serve_untested",
        "measure_vs_bar", "full_report",
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

        if action in self._ALWAYS_ON:
            return True

        # When ANY text Input is focused, hide the context-key bindings from the
        # footer.  The Input's own _on_key stops printable characters before they
        # reach app bindings, but we hide them for footer accuracy (the filter +
        # the BYO / lane repo inputs share letters with hotkeys q/e/s/w/c/p/o, so
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

        # Sub-tab cycle keys: only meaningful in modes with sub-tabs (0, 1, 2).
        if action in ("prev_subtab", "next_subtab"):
            return self._active_mode in (0, 1, 2)

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

    def _current_subtab(self) -> str:
        """Return the active tab ID for the current mode's TabbedContent, or ''."""
        tab_ids = {
            0: "#run-tabs",
            1: "#operate-tabs",
            2: "#validate-tabs",
        }
        tc_id = tab_ids.get(self._active_mode, "")
        if not tc_id:
            return ""
        try:
            return self.query_one(tc_id, TabbedContent).active
        except Exception:
            return ""

    def __init__(self, repo_root: Path, *, data: Optional[CockpitData] = None,
                 surface: str = "consumer", **kwargs):
        super().__init__(**kwargs)
        self._repo_root = repo_root
        # Audience surface (R0): "consumer" (default — Run + Operate) or "producer"
        # (+ Bring & Validate, R3). Gates producer-only actions/modes via
        # _PRODUCER_ONLY in check_action, and surfaces a CONTRIBUTE indicator.
        self._surface = surface if surface in ("consumer", "producer") else "consumer"
        if self._surface == "producer":
            self.sub_title = f"{self.SUB_TITLE} · ⚒ CONTRIBUTE"
        # Injectable service layer — defaults to the real (live-read) impl.
        self._data: CockpitData = data or CockpitData(repo_root)
        self._active_mode = 0  # 0=Run 1=Operate 2=Validate
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
        # Phase 5: the SHARED ServingTarget OBJECT from the last estate poll —
        # held by identity so the c3t Evaluate hand-off passes the SAME dataclass
        # instance c3t speaks (design §4/§6.6), not a reconstructed copy.
        self._target_obj = None
        # Phase 5: the last BYO fit-check result (Run · BYO) — the arch facts
        # the Promote-to-catalog scaffold computes from.
        self._last_byo: Optional[ByoResult] = None
        # Phase R / R2b: failure context for the [!] problem report — captured AT
        # a failed serve in dispatch_action (the slug + the boot-log lines that
        # were streamed into the serve-live pane) so problem_report can assemble
        # a paste-ready issue with the readily-available context.
        self._problem_slug: str = ""
        self._problem_boot_log: str = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-layout"):
            with Vertical(id="left-rail"):
                yield ModeSwitcher(id="mode-switcher", surface=self._surface)
                yield RailStatus(id="rail-status")
            with Container(id="content-area"):
                # Mode 0 — Run (Discover + Serve + Benchmarks folded in)
                with Container(id="panel-run", classes="mode-panel active"):
                    with TabbedContent(id="run-tabs"):
                        with TabPane("Catalog", id="tab-catalog"):
                            yield CatalogPane(id="catalog-pane")
                        with TabPane("Bring-your-own", id="tab-byo"):
                            yield ByoPane(id="byo-panel")
                    # Transient boot-output pane — re-homed from the retired Serve
                    # mode.  Hidden until ⏎ on a Catalog row stages a serve and the
                    # reconcile-gated confirm commits; then the boot log streams here.
                    yield LivePane(id="serve-live")

                # Mode 1 — Operate (Orchestration + Containers + Doctor)
                with Container(id="panel-operate", classes="mode-panel"):
                    with TabbedContent(id="operate-tabs"):
                        with TabPane("Orchestration", id="tab-orchestration"):
                            yield OperateOrchPane(id="operate-orch-pane")
                        with TabPane("Containers", id="tab-containers"):
                            yield OperateContainersPane(id="operate-containers-pane")
                        with TabPane("Doctor", id="tab-doctor"):
                            yield DoctorPane(id="doctor-pane")

                # Mode 2 — Bring & Validate (producer lane, R3b-1).  Presented as
                # an ORDERED, numbered pipeline reusing the TabbedContent pattern:
                #   ① Bring  → LaneBringPane (reuses byo_check fit-check)
                #   ② Serve  → LaneServePane (NEW — generate compose + serve untested)
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
        yield Footer()

    # ── Mount / startup ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.load_catalog()

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

    # ── Estate polling ───────────────────────────────────────────────────────────────

    @work(exclusive=True, group="estate")
    async def load_estate(self) -> None:
        """Poll the live estate snapshot + push into the orch/doctor panes + rail.

        Also captures the live target (matched slug / model / url) so Doctor's
        profile-triage and the validation launches point at the running model,
        and reads the power-cap status (a safe READ) for the orch pane."""
        state = await self._data.estate_state(variants=self._variants or None)
        # Capture the live target for profile-triage / validation launches.
        self._target_slug = state.matched_slug or ""
        tgt = state.target
        self._target_model = getattr(tgt, "model", "") or ""
        self._target_url = getattr(tgt, "url", "") or ""
        # Hold the SHARED ServingTarget object (by identity) for the c3t Evaluate
        # hand-off — design §4/§6.6 requires passing the SAME dataclass instance.
        self._target_obj = tgt
        try:
            self.query_one("#operate-orch-pane", OperateOrchPane).populate(state)
        except Exception:
            pass
        try:
            self.query_one("#operate-containers-pane", OperateContainersPane).populate(
                state.containers
            )
        except Exception:
            pass
        try:
            self.query_one("#doctor-pane", DoctorPane).populate(state)
        except Exception:
            pass
        try:
            self.query_one("#rail-status", RailStatus).update_from_state(state)
        except Exception:
            pass
        # Power-cap status (READ) for the orch pane.
        st = await self._data.power_cap_get()
        try:
            self.query_one("#operate-orch-pane", OperateOrchPane).populate_power_cap(st)
        except Exception:
            pass
        # Doctor lives in Operate (R2a) and its profile-triage consumes the
        # _target_slug/_target_url THIS poll just captured — so chain the doctor
        # read here rather than racing it as a sibling worker off the mode switch
        # (which read _target_slug before this wrote it → empty profile card on
        # first entry). Guarded to Operate; this also means action_refresh ([r],
        # which re-runs load_estate) now refreshes the Doctor cards too.
        if self._active_mode == 1:
            self.load_doctor()

    # ── Validate-mode loaders ──────────────────────────────────────────────────────

    @work(exclusive=True, group="doctor")
    async def load_doctor(self) -> None:
        """Run the full Doctor read (health + diagnose-estate + diagnose-profile)
        and push it into the Doctor pane.  ALL three legs are READ-only."""
        slug = self._target_slug or (self._staged_entry.slug if self._staged_entry else None)
        report = await self._data.doctor(url=self._target_url or None, slug=slug)
        try:
            self.query_one("#doctor-pane", DoctorPane).populate_report(report)
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
        # The fit-check can be triggered from EITHER Run · BYO (consumer) OR the
        # producer lane's ① Bring stage; render the verdict into whichever panes
        # exist (both share byo_check + the verdict text).  R3b-1.
        run_pane = lane_pane = None
        try:
            run_pane = self.query_one("#byo-panel", ByoPane)
            run_pane.set_checking(repo)
        except Exception:
            run_pane = None
        try:
            lane_pane = self.query_one("#lane-bring-pane", LaneBringPane)
            lane_pane.set_checking(repo)
        except Exception:
            lane_pane = None
        res = await self._data.byo_check(repo, profile_like)
        # Cache the arch facts for the lane ② Serve + the Promote scaffold (Phase 5).
        self._last_byo = res
        if run_pane is not None:
            run_pane.populate(res)
        if lane_pane is not None:
            lane_pane.populate(res)

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
        prune, the promote write) is reported trivially-safe: the reconcile gate
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
        if live is not None and plan.kind == "serve":
            # Reveal the transient Run boot pane (Fold 2) and stream the boot log.
            self._reveal_serve_live()
            live.clear_log()
            live.append_line(f"[green]▶ launching[/green] {plan.description}")
            live.append_line("[dim](boot log streams here)[/dim]")

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
        panel_ids = ["panel-run", "panel-operate", "panel-validate"]
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
        # Operate is live — poll the estate on entry.  load_estate chains the
        # Doctor read at its END (Doctor lives in Operate as of R2a, and its
        # profile-triage consumes the target the poll captures — so it must run
        # AFTER the poll, not race it as a sibling worker).
        if index == 1:
            self.load_estate()
        # Bring & Validate lane is live too — load the evidence read AND prime the
        # live target (R3b-1 fix): [v] Evaluate consumes _target_obj, which only
        # load_estate writes.  Without this, entering the lane directly (key 3,
        # never visiting Operate) leaves _target_obj=None → "nothing to evaluate"
        # even with a model live.  load_estate best-effort-guards every Operate
        # pane query_one, so it is safe to call when those panes aren't mounted.
        elif index == 2:
            self._load_validate()
            self.load_estate()

    def _focus_mode_primary(self, index: int) -> None:
        """Move focus to the mode's primary interactive widget after a mode switch.

        Deferred via call_after_refresh so this runs AFTER any pending
        call_after_refresh callbacks from on_tabbed_content_tab_activated (which
        are enqueued during mount) — ensuring mode-switch focus wins."""
        def _do() -> None:
            try:
                if index == 0:  # Run — catalog table
                    self.query_one("#catalog-table", DataTable).focus()
                elif index == 1:  # Operate — focus the active tab's table; Doctor
                    # is read-only with no focusable table, so leave focus unset
                    # there rather than grabbing the hidden Orchestration table.
                    try:
                        tc = self.query_one("#operate-tabs", TabbedContent)
                        if tc.active == "tab-containers":
                            self.query_one("#containers-table", DataTable).focus()
                        elif tc.active == "tab-orchestration":
                            self.query_one("#scene-table", DataTable).focus()
                    except Exception:
                        pass
                elif index == 2:  # Bring & Validate lane — focus the active stage's
                    # primary DATA TABLE (③ Gate ladder / ④ Measure list).  ① Bring /
                    # ② Serve / ⑤ Promote have no focusable table (① Bring's only
                    # focusable is an Input, which would swallow the global digit /
                    # bracket keys), so leave focus on the tab bar there — matching
                    # how Doctor (read-only) leaves focus unset — so 1/2/3 + [ ]
                    # still route to the app.
                    try:
                        tc = self.query_one("#validate-tabs", TabbedContent)
                        if tc.active == "tab-run":
                            self.query_one("#run-ladder-table", DataTable).focus()
                        elif tc.active == "tab-evidence":
                            self.query_one("#evidence-table", DataTable).focus()
                    except Exception:
                        pass
            except Exception:
                pass
        self.call_after_refresh(_do)

    def _load_validate(self) -> None:
        """Kick the Validate read workers (evidence).
        Best-effort — a failing leg doesn't block the rest.  The Run pane is
        launch-driven (no background read).  R2a moved Doctor to Operate, so
        Validate no longer kicks load_doctor here (that fires on Operate entry)."""
        self.load_evidence()

    # ── Actions ──────────────────────────────────────────────────────────────────────

    def action_mode_run(self) -> None:
        self._switch_mode(0)

    def action_mode_operate(self) -> None:
        self._switch_mode(1)

    def action_mode_validate(self) -> None:
        # Belt-and-suspenders surface guard (R3a): Validate is the producer
        # Bring & Validate lane.  The binding is already gated by check_action's
        # surface gate, but guard the action too so no programmatic / edge path
        # can land a consumer in the producer mode.
        if self._surface != "producer":
            return
        self._switch_mode(2)

    def action_toggle_contribute(self) -> None:
        """The in-app Contribute DOOR (R4): toggle consumer ↔ producer at runtime
        and persist the choice for next launch.

        On toggle we (1) flip self._surface, (2) re-render the ModeSwitcher so it
        shows 2↔3 modes (set_surface), (3) refresh_bindings() so the
        _PRODUCER_ONLY actions gate/ungate immediately, (4) persist via
        save_surface_setting so resolve_surface picks it up next launch, and (5)
        notify the new mode.  The Bring & Validate lane panel is always mounted
        (gated by check_action, not conditionally composed), so there is nothing
        to mount/unmount — only the gating + the rail change.

        EDGE: toggling producer → consumer while the user is IN the producer lane
        (mode 2, now hidden) would strand them, so we first switch them back to a
        consumer-visible mode (Run).  Toggling consumer → producer from Run/Operate
        just unlocks mode 2 — no forced switch."""
        new_surface = "consumer" if self._surface == "producer" else "producer"
        # EDGE: leaving producer while stranded in the now-hidden lane → Run.
        if new_surface == "consumer" and self._active_mode == 2:
            self._switch_mode(0)
        self._surface = new_surface
        # Keep the CONTRIBUTE sub-title indicator in sync with the live surface.
        if new_surface == "producer":
            self.sub_title = f"{self.SUB_TITLE} · ⚒ CONTRIBUTE"
        else:
            self.sub_title = self.SUB_TITLE
        # Re-render the rail (show/hide mode 2 — a pure .mode-hidden class flip),
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
        if new_surface == "producer":
            self.notify(
                "⚒ Contributor mode — Bring & Validate unlocked",
                title="Contribute",
                timeout=4,
            )
        else:
            self.notify("Consumer mode", title="Contribute", timeout=3)

    def action_refresh(self) -> None:
        """Re-read the live data layer for the active mode."""
        if self._active_mode == 1:
            self.load_estate()
        elif self._active_mode == 2:
            self._load_validate()
            # R3b-1: re-prime the live target so [v] Evaluate stays wired to the
            # currently-serving model on a lane refresh (mirrors _switch_mode).
            self.load_estate()
        else:
            try:
                self.query_one("#catalog-pane", CatalogPane).query_one(
                    "#catalog-status", Label
                ).update("Refreshing catalog…")
            except Exception:
                pass
            self.load_catalog()

    def action_filter_catalog(self) -> None:
        """[/] filters the Run catalog (the Benchmarks tab — and its own filter —
        was folded into Run · Catalog / explain)."""
        if self._active_mode == 0:
            try:
                self.query_one("#catalog-pane", CatalogPane).toggle_filter()
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

    def action_explain(self) -> None:
        """Open the explain detail modal for the selected catalog slug."""
        if self._active_mode != 0:
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
        """⏎ — context-specific per mode."""
        if self._active_mode == 0:
            self._run_primary()
        elif self._active_mode == 1:
            self._operate_primary()
        else:
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
        # The screen loads its own report on mount (run_evidence_report), so the
        # set_report query resolves against a fully-mounted modal.
        self.push_screen(EvidenceReportScreen(tag.tag))

    def _run_primary(self) -> None:
        """⏎ in Run · Catalog (Fold 2): stage the selected slug and open the
        reconcile-gated serve confirm directly — no Serve-mode hop.  The serve
        ActionPlan goes through the SAME ConfirmActionScreen → run_reconcile_for_modal
        → dispatch_action gate as every other GPU-mutating write; on confirm the
        boot streams into the transient Run LivePane (#serve-live).  ⏎ on the BYO
        tab no-ops (BYO has its own Fit-check button)."""
        if self._active_run_tab() != "tab-catalog":
            return
        try:
            entry = self.query_one("#catalog-pane", CatalogPane).selected_entry()
        except Exception:
            entry = None
        if entry is None:
            return
        self._staged_entry = entry
        plan = self._data.serve(entry.slug)  # gated, NOT --force
        self.push_screen(ConfirmActionScreen(plan))

    def _active_run_tab(self) -> str:
        try:
            return self.query_one("#run-tabs", TabbedContent).active
        except Exception:
            return ""

    def _operate_primary(self) -> None:
        """⏎ in Operate · Orchestration: confirm-gated scene switch."""
        try:
            scene = self.query_one("#operate-orch-pane", OperateOrchPane).selected_scene()
        except Exception:
            scene = None
        if scene is None:
            return
        plan = self._data.scene_switch(scene.name)
        self.push_screen(ConfirmActionScreen(plan))

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
        if self._active_mode != 0:
            return
        entry = self._selected_catalog_entry()
        if entry is None:
            return
        plan = self._data.set_default(entry.slug)
        self.push_screen(ConfirmActionScreen(plan))

    def action_clear_default(self) -> None:
        """[D] in Run · Catalog: clear the model default pin for the
        selected slug's model (gated path, .env write)."""
        if self._active_mode != 0:
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
        """[l] in Operate · Containers: stream `docker logs` for the selected
        container into the drill Logs LivePane.  This is a READ — safe to run
        live (the conftest blocks an accidental write, not this read)."""
        if self._active_mode != 1:
            return
        con = self._selected_container()
        if con is None:
            self.notify("No container selected.", title="Logs", severity="warning", timeout=3)
            return
        try:
            tabs = self.query_one("#drill-tabs", TabbedContent)
            tabs.active = "drill-tab-logs"
        except Exception:
            pass
        self.stream_container_logs(con.name)

    def action_s_key(self) -> None:
        """[s] is context-sensitive:
          - Operate · Containers : gated `docker restart <name>`.
          - Validate · Evidence : gated submit-to-localmaxxing for the tag.
        Other contexts ignore it."""
        if self._active_mode == 2 and self._active_validate_tab() == "tab-evidence":
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
        # Operate · Containers ONLY.  [s] (restart) falls through here from
        # action_s_key without a sub-tab gate, so guard the WRITE itself — Doctor
        # (and Orchestration) are not container-write surfaces; Doctor is
        # read-only, and a stray [s] there must not pop a `docker restart` confirm.
        if self._active_mode != 1 or self._active_operate_tab() != "tab-containers":
            return
        con = self._selected_container()
        if con is None:
            self.notify(
                f"No container selected to {op}.", title="Containers", severity="warning", timeout=3
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
        """[X] in Operate · Containers: reconcile-gated `docker rm <name>`.

        Removing a container frees a GPU it held → the plan requires_reconcile,
        so it routes through the SAME ConfirmActionScreen → dispatch_action gate
        as stop.  rm of a live container needs Force (which adds -f)."""
        if self._active_mode != 1:
            return
        con = self._selected_container()
        if con is None:
            self.notify("No container selected to remove.", title="Containers", severity="warning", timeout=3)
            return
        plan = self._data.container_rm(con.name)
        self.push_screen(ConfirmActionScreen(plan))

    def action_context_t(self) -> None:
        """[t] reads `docker top` for the selected container (Operate · Containers).
        The Benchmarks sort-cycle role was retired with the Benchmarks tab (Fold 3)."""
        if self._active_mode == 1 and self._active_operate_tab() == "tab-containers":
            self._container_top()

    def _container_top(self) -> None:
        con = self._selected_container()
        if con is None:
            self.notify("No container selected.", title="Top", severity="warning", timeout=3)
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

        def _on_line(text: str) -> None:
            if live is not None:
                live.append_line(text)

        await self._data.run_validation(
            kind,
            model=self._target_model or None,
            url=self._target_url or None,
            slug=slug,
            on_line=_on_line,
        )
        self.notify(f"{kind} launched.", title="Validate", severity="information", timeout=4)

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
        if self._active_mode != 2 or self._active_validate_tab() != "tab-evidence":
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
        """[F] in the lane's ③ Gate tab: launch the ~43-min FULL validation battery
        (report.sh --full).  PRODUCER-only (gated in check_action).  CONFIRM-gated
        (heavy + long-running, and it needs a model serving), then bg-streamed into
        the ③ Gate LivePane.  It uses the SERVING model and does NOT claim a GPU
        (requires_confirm=True, requires_reconcile=False) → NEVER auto-fired."""
        if self._active_mode != 2 or self._active_validate_tab() != "tab-run":
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
        """[B] (Operate): stage the OUTWARD submit-to-localmaxxing for the most
        recent benched run tag.  This is the ONLY outward write of the three
        share-back affordances — it keeps its confirm + network gate (the
        existing submit_bench ActionPlan: requires_confirm + network=True).  The
        network is mocked in tests; NEVER auto-fired."""
        if self._active_mode != 1:
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
        """[s] in Validate · Evidence: stage the gated submit-to-localmaxxing for
        the selected run tag.  OUTWARD-FACING NETWORK WRITE — confirm-gated,
        NEVER auto-fired; the network is mocked in tests."""
        if self._active_mode != 2 or self._active_validate_tab() != "tab-evidence":
            return
        try:
            tag = self.query_one("#validate-evidence-pane", ValidateEvidencePane).selected_tag()
        except Exception:
            tag = None
        if tag is None:
            self.notify("No run tag selected.", title="Evidence", severity="warning", timeout=3)
            return
        plan = self._data.submit_bench(tag.tag)
        self.push_screen(ConfirmActionScreen(plan))

    # ── Operate · Orchestration: power-cap + prune (gated rig writes) ───────────────────

    def action_power_cap_toggle(self) -> None:
        """[c] in Operate · Orchestration: confirm-gated power-cap on/off.

        Reads the current cap state to decide the toggle direction (on→off /
        off→on), then routes the WRITE through the standard confirm gate.  A
        cap write is a rig mutation — NEVER auto-fired."""
        if self._active_mode != 1 or self._active_operate_tab() != "tab-orchestration":
            return
        self._toggle_power_cap()

    @work(group="power-cap-toggle")
    async def _toggle_power_cap(self) -> None:
        st = await self._data.power_cap_get()
        # If any GPU is below its default, treat the rig as "capped" → turn off.
        capped = any(
            g.limit_w is not None and g.default_w is not None and g.limit_w < g.default_w
            for g in st.gpus
        )
        target = "off" if capped else "on"
        plan = self._data.power_cap_set(target)
        self.push_screen(ConfirmActionScreen(plan))

    def action_power_cap_sweep(self) -> None:
        """[w] in Operate · Orchestration: confirm-gated power-cap sweep (heavy +
        mutating — runs benches at each cap).  NEVER auto-fired."""
        if self._active_mode != 1 or self._active_operate_tab() != "tab-orchestration":
            return
        plan = self._data.power_cap_sweep()
        self.push_screen(ConfirmActionScreen(plan))

    def action_prune_images(self) -> None:
        """[p] in Operate · Orchestration: confirm-gated image prune (DESTRUCTIVE —
        deletes unreferenced images).  NEVER auto-fired."""
        if self._active_mode != 1 or self._active_operate_tab() != "tab-orchestration":
            return
        plan = self._data.prune()
        self.push_screen(ConfirmActionScreen(plan))

    # ── Estate stop-all (Operate · Orchestration) ──────────────────────────────────────

    def action_estate_off(self) -> None:
        """[o] in Operate · Orchestration: gated estate-down (stop all)."""
        if self._active_mode != 1:
            return
        plan = self._data.estate_down()
        self.push_screen(ConfirmActionScreen(plan))

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
        if self._active_mode != 2:
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
            profile = (
                self.query_one("#lane-bring-profile-input", Input).value.strip()
                or "vllm/dual"
            )
        except Exception:
            return
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
        if self._active_mode != 2:
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
        if self._active_mode != 2:
            return
        if self._last_byo is None:
            self.notify(
                "No BYO model to promote — run a fit-check in Run · Bring-your-own first.",
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
            0: "#run-tabs",
            1: "#operate-tabs",
            2: "#validate-tabs",
        }
        tc_id = tab_widget_ids.get(self._active_mode, "")
        if not tc_id:
            return
        try:
            tc = self.query_one(tc_id, TabbedContent)
            panes = [p.id for p in tc.query(TabPane)]
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
        # Only respond to tabs that belong to the current mode's active panel.
        _mode_tabs: dict[int, set[str]] = {
            0: {"tab-catalog", "tab-byo"},
            1: {"tab-orchestration", "tab-containers", "tab-doctor"},
            # R3b-1: the producer lane's ordered stages ①→⑤.
            2: {"tab-bring", "tab-serve", "tab-run", "tab-evidence", "tab-promote"},
        }
        allowed_tabs = _mode_tabs.get(self._active_mode, set())
        if tab_id not in allowed_tabs:
            return
        # NOTE (R3b-1): the lane's ① Bring / ② Serve / ⑤ Promote stages are NOT in
        # this map on purpose — ① Bring's only focusable widget is an Input, which
        # would swallow the global digit (1/2/3) + bracket ([ ]) keys.  Leaving
        # focus on the tab bar there keeps those keys routed to the app; the user
        # Tab/clicks into the HF-repo input to type.
        _focus_map: dict[str, str] = {
            "tab-catalog":        "#catalog-table",
            "tab-run":            "#run-ladder-table",
            "tab-evidence":       "#evidence-table",
            "tab-orchestration":  "#scene-table",
            "tab-containers":     "#containers-table",
        }
        widget_id = _focus_map.get(tab_id, "")
        if widget_id:
            def _do_focus() -> None:
                try:
                    self.query_one(widget_id, DataTable).focus()
                except Exception:
                    pass
                # Entering Operate·Containers auto-loads the detail for the
                # already-highlighted container — switching tabs doesn't re-fire
                # RowHighlighted (the cursor was set when the table populated),
                # so trigger the load here.
                if tab_id == "tab-containers":
                    self._refresh_container_config()
                    self._load_active_drill_tab()
            self.call_after_refresh(_do_focus)

    # ── Widget event handlers ─────────────────────────────────────────────────────────

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """DataTable emits RowSelected when the user presses Enter (select_cursor).
        Route it to the app's primary action so focusing a DataTable doesn't break
        the ⏎ → primary_action contract that the tests (and help text) document."""
        event.stop()
        self.action_primary_action()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Operate · Containers: auto-load the drill detail for the highlighted
        container (lazydocker-style). Config is a local read → immediate; the
        active live tab (Logs / Top) is a docker read → debounced ~250ms so
        arrowing through the list doesn't spawn a subprocess per row."""
        try:
            if event.data_table.id != "containers-table":
                return
        except Exception:
            return
        if self._active_mode != 1 or self._active_operate_tab() != "tab-containers":
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
        tab = self._active_drill_tab()
        if tab == "drill-tab-logs":
            self.stream_container_logs(con.name)
        elif tab == "drill-tab-stats":
            self.read_container_top(con.name)

    def on_key(self, event) -> None:
        """App-level Esc: close an open filter (Run·Catalog) and refocus the
        table. Modal screens capture their own Esc (they have escape→dismiss
        bindings), so this only runs on the main screen — Esc otherwise no-ops
        and NEVER quits."""
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
        if bid == "byo-fit-btn":
            self._trigger_byo()
        elif bid == "lane-bring-fit-btn":
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
        elif event.input.id in ("byo-url-input", "byo-profile-input"):
            self._trigger_byo()
        elif event.input.id in ("lane-bring-url-input", "lane-bring-profile-input"):
            self._trigger_lane_bring()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "catalog-filter":
            try:
                self.query_one("#catalog-pane", CatalogPane).set_filter(event.value)
            except Exception:
                pass

    def _trigger_byo(self) -> None:
        try:
            repo = self.query_one("#byo-url-input", Input).value.strip()
            profile = self.query_one("#byo-profile-input", Input).value.strip() or "vllm/dual"
        except Exception:
            return
        if not repo:
            self.notify("Enter an HF repo (org/Model).", title="BYO", severity="warning", timeout=3)
            return
        self.run_byo_check(repo, profile)
