"""club3090 serve cockpit — main Textual application.

Phase 1: walking skeleton.
  - Full 4-mode nav (Discover / Serve / Estate / Validate)
  - Real Catalog DataTable populated from registry_variant_rows
  - All other panels are static placeholder stubs
  - No actions wired (Enter pops a "wired in Phase 3" toast)
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
)
from textual import work

from .registry import VariantRow, load_catalog_sync

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


# ── Help modal ────────────────────────────────────────────────────────────────


class HelpScreen(ModalScreen):
    """Help overlay showing keybindings and current phase status."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen > Vertical {
        width: 72;
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

    HELP_TEXT = """\
[bold]Keybindings[/bold]

  [cyan]1[/cyan]  Discover    [cyan]2[/cyan]  Serve    [cyan]3[/cyan]  Estate    [cyan]4[/cyan]  Validate
  [cyan]r[/cyan]  Refresh catalog (re-reads registry only)
  [cyan]⏎[/cyan]  Primary action (Phase 3 — shows "wired in Phase 3" notice)
  [cyan]?[/cyan]  This help
  [cyan]q[/cyan]  Quit

[bold]Phase 1 scope[/bold]

  This is the walking skeleton.  All navigation nodes are present and
  render something.  Only the Discover → Catalog tab is wired to real
  data (registry_variant_rows).  Serve / Estate / Validate and all
  action bindings are stubbed — they will be wired in Phase 3.

[bold]Stub columns in Catalog[/bold]

  fit · TPS · 8pk · source  are placeholder glyphs (·/—).
  slug · engine · status · ctx  come from the live registry.

[bold]Status glyphs[/bold]

  ✅ production   ⚠️  caveats   🧪 experimental
  🐣 incubating  👁️  preview   ⏸️  upstream-gated   🗑️  deprecated
"""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("club3090 serve cockpit — Help", classes="help-title")
            yield Static(self.HELP_TEXT)

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ── Placeholder panels ────────────────────────────────────────────────────────


class PlaceholderPanel(Static):
    """A static placeholder panel with a title and description text."""

    def __init__(self, title: str, body: str, **kwargs):
        markup = f"[bold]{title}[/bold]\n\n{body}"
        super().__init__(markup, **kwargs)


# ── Discover tab content ──────────────────────────────────────────────────────


class CatalogPane(Container):
    """Catalog tab: DataTable populated from the live registry."""

    DEFAULT_CSS = """
    CatalogPane {
        height: 1fr;
    }
    CatalogPane #catalog-status {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    CatalogPane DataTable {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Loading catalog…", id="catalog-status")
        table: DataTable = DataTable(id="catalog-table", zebra_stripes=True)
        table.cursor_type = "row"
        yield table

    def on_mount(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        table.add_columns("slug", "engine", "fit", "ctx", "TPS", "8pk", "status", "source")

    def populate(self, rows: list[VariantRow], error: str | None) -> None:
        """Fill the table with catalog rows (called from the worker result)."""
        status_label = self.query_one("#catalog-status", Label)
        table = self.query_one("#catalog-table", DataTable)
        table.clear()

        if error:
            status_label.update(f"[red]Catalog error:[/red] {error}")
            table.add_row("—", "—", "—", "—", "—", "—", "—", "—")
            return

        status_label.update(f"{len(rows)} variants loaded from registry")
        for r in rows:
            table.add_row(
                r.slug,
                r.engine,
                r.fit,           # stub: "·"
                r.ctx_label or "—",
                r.tps,           # stub: "—"
                r.quality_8pk,   # stub: "—"
                _status_glyph(r.status),
                r.source,        # stub: "·"
            )


# ── Mode switcher (left rail) ─────────────────────────────────────────────────


MODES = [
    ("Discover", "1"),
    ("Serve", "2"),
    ("Estate", "3"),
    ("Validate", "4"),
]


class ModeSwitcher(Static):
    """Left-rail mode selector.  Purely cosmetic in Phase 1 — navigation is
    driven by CockpitApp._active_mode and the 1–4 digit bindings."""

    DEFAULT_CSS = """
    ModeSwitcher {
        width: 16;
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
    """

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._active = 0

    def compose(self) -> ComposeResult:
        yield Label("Modes", classes="mode-title")
        for i, (name, digit) in enumerate(MODES):
            classes = "mode-item-active" if i == 0 else "mode-item"
            yield Label(f"▸ {name} [{digit}]" if i == 0 else f"  {name} [{digit}]",
                        id=f"mode-{i}", classes=classes)

    def set_active(self, index: int) -> None:
        """Update the visual highlight for the active mode."""
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


# ── Main application ──────────────────────────────────────────────────────────


class CockpitApp(App):
    """club3090 serve cockpit — Phase 1 walking skeleton."""

    TITLE = "club3090 cockpit"
    SUB_TITLE = "Phase 1 · walking skeleton"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("question_mark", "help", "Help", show=True),
        Binding("r", "refresh_catalog", "Refresh", show=True),
        Binding("1", "mode_discover", "Discover", show=True),
        Binding("2", "mode_serve", "Serve", show=True),
        Binding("3", "mode_estate", "Estate", show=True),
        Binding("4", "mode_validate", "Validate", show=True),
        Binding("enter", "primary_action", "Action", show=True),
    ]

    CSS = """
    #main-layout {
        height: 1fr;
    }
    #left-rail {
        width: 18;
        height: 1fr;
        padding: 0 0;
    }
    #content-area {
        width: 1fr;
        height: 1fr;
    }
    /* Mode content panels — only one visible at a time */
    .mode-panel {
        width: 1fr;
        height: 1fr;
        display: none;
    }
    .mode-panel.active {
        display: block;
    }
    /* Placeholder panel padding */
    PlaceholderPanel {
        padding: 1 2;
        color: $text-muted;
    }
    /* Estate GPU bar stub */
    #estate-gpu-stub {
        border: solid $primary;
        margin: 1 1;
        padding: 0 1;
        height: auto;
    }
    """

    def __init__(self, repo_root: Path, **kwargs):
        super().__init__(**kwargs)
        self._repo_root = repo_root
        self._active_mode = 0  # 0=Discover 1=Serve 2=Estate 3=Validate

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-layout"):
            with Vertical(id="left-rail"):
                yield ModeSwitcher(id="mode-switcher")
            with Container(id="content-area"):
                # Mode 0 — Discover
                with Container(id="panel-discover", classes="mode-panel active"):
                    with TabbedContent(id="discover-tabs"):
                        with TabPane("Catalog", id="tab-catalog"):
                            yield CatalogPane(id="catalog-pane")
                        with TabPane("Bring-your-own", id="tab-byo"):
                            yield PlaceholderPanel(
                                "Bring-your-own HF model",
                                "Future flow (Phase 3):\n\n"
                                "  1. Paste an HF repo slug or local path\n"
                                "  2. Run pull.sh --profile-like <engine-slug> --dry-run\n"
                                "  3. View the pull-gate verdict (arch supported? fits topology?)\n"
                                "  4. For in-scope arches: generate-compose.sh emits a\n"
                                "     reproduction compose as a runnable starting point\n\n"
                                "Routes A (HF safetensors) · B (local GGUF) · C (fine-tune of\n"
                                "a curated arch) will each surface the appropriate next step.\n\n"
                                "[dim]Not wired — Phase 3[/dim]",
                                id="byo-panel",
                            )

                # Mode 1 — Serve
                with Container(id="panel-serve", classes="mode-panel"):
                    yield PlaceholderPanel(
                        "Serve",
                        "Future flow (Phase 3):\n\n"
                        "  Select a slug from Discover → Catalog, press ⏎\n"
                        "  A plan-confirm modal surfaces:\n"
                        "    slug · engine · KV · fit · est VRAM · what it tears down\n"
                        "  Confirm (⏎) commits switch.sh <slug> and streams boot logs.\n"
                        "  --force is only offered when preflight explicitly needs it.\n\n"
                        "  --set-default / --clear-default are also wired here.\n"
                        "  setup.sh handles missing weights.\n\n"
                        "[dim]Not wired — Phase 3[/dim]",
                        id="serve-panel",
                    )

                # Mode 2 — Estate
                with Container(id="panel-estate", classes="mode-panel"):
                    with TabbedContent(id="estate-tabs"):
                        with TabPane("Orchestration", id="tab-orchestration"):
                            with ScrollableContainer():
                                yield Static(
                                    "[bold]GPU estate (stub)[/bold]",
                                    id="estate-gpu-stub",
                                )
                                yield Static(
                                    "  GPU0  ░░░░░░░░░░░░░░░░░░░░  1 MiB / 24 GiB  idle  ···W\n"
                                    "  GPU1  ░░░░░░░░░░░░░░░░░░░░  1 MiB / 24 GiB  idle  ···W\n"
                                    "  scene: [dim]off[/dim]",
                                    id="estate-gpu-bars",
                                )
                                yield PlaceholderPanel(
                                    "Orchestration",
                                    "Future wiring (Phase 3):\n\n"
                                    "  Live state via detect.py (docker-ps + nvidia-smi)\n"
                                    "  Doctor: health.sh runtime health panel\n"
                                    "  Scene switcher: gpu-mode --list-modes --json\n"
                                    "  estate_cli boot/down/validate via facade + worker thread\n"
                                    "  Every write goes through reconcile-before-write gate\n\n"
                                    "  Stub scene list:\n"
                                    "    [dim]serving:[/dim]  27b · gemma-int8 · deckard\n"
                                    "    [dim]studio:[/dim]   image-studio · video-studio\n"
                                    "    [dim]ops:[/dim]      off · prune · prune-all\n\n"
                                    "[dim]Not wired — Phase 3[/dim]",
                                )
                        with TabPane("Containers", id="tab-containers"):
                            yield PlaceholderPanel(
                                "Containers (lazydocker floor)",
                                "Future wiring (Phase 3):\n\n"
                                "  All stack containers grouped engines | services\n"
                                "  Columns: name · kind · status/health · uptime ·\n"
                                "           restarts · ports · GPU(s) · mapped slug/scene\n\n"
                                "  Stub container list:\n"
                                "    engines:   [dim](none running)[/dim]\n"
                                "    services:  [dim](probe not wired)[/dim]\n\n"
                                "  Drill (lazydocker tabs): Logs · Stats · Config\n"
                                "  Lifecycle: restart · stop (behind confirm)\n\n"
                                "[dim]Not wired — Phase 3[/dim]",
                            )

                # Mode 3 — Validate
                with Container(id="panel-validate", classes="mode-panel"):
                    with TabbedContent(id="validate-tabs"):
                        with TabPane("Run", id="tab-run"):
                            yield PlaceholderPanel(
                                "Validate · Run",
                                "Future wiring (Phase 4):\n\n"
                                "  Ladder: verify-full · verify-stress · bench ·\n"
                                "          quality-test · soak-test · rebench-full\n"
                                "  Plus:   quality-baseline (regression diff vs curated baseline)\n"
                                "          bench-agentic (multi-turn prefill stress)\n"
                                "          stream-toolcall-probe (silent streaming breakage check)\n\n"
                                "  Tuning gotchas inline (from BRING_YOUR_OWN.md §2):\n"
                                "    ctx ceiling ≠ advertised; NIAH-allocation-isn't-use;\n"
                                "    A/B at matched power (230↔370W lever);\n"
                                "    judge spec-dec on bench delta not acceptance rate\n\n"
                                "[dim]Not wired — Phase 4[/dim]",
                            )
                        with TabPane("Doctor", id="tab-doctor"):
                            yield PlaceholderPanel(
                                "Validate · Doctor",
                                "Future wiring (Phase 4):\n\n"
                                "  health.sh — runtime health (KV-pool, spec-dec firing, errors)\n"
                                "  diagnose-estate.sh — estate-config diagnosis\n"
                                "  diagnose-profile.sh — per-slug profile check (Tier-③)\n\n"
                                "[dim]Not wired — Phase 4[/dim]",
                            )
                        with TabPane("Benchmarks", id="tab-benchmarks"):
                            yield PlaceholderPanel(
                                "Validate · Benchmarks",
                                "Future wiring (Phase 4):\n\n"
                                "  Filterable explorer of BENCHMARKS.md + the structured\n"
                                "  measurement corpus (model / engine / topology / TPS / ctx / 8pk)\n"
                                "  Our self-hosted alternative to localmaxxing.\n\n"
                                "  submit-bench.sh — 'share to localmaxxing' action\n\n"
                                "[dim]Not wired — Phase 4[/dim]",
                            )
                        with TabPane("Evidence", id="tab-evidence"):
                            yield PlaceholderPanel(
                                "Validate · Evidence",
                                "Future wiring (Phase 4):\n\n"
                                "  Browse results/rebench/<tag>/ artifact tree\n"
                                "  (REPORT.md, soak logs, NIAH artifacts)\n"
                                "  report.sh / rebench-report.py — paste-ready triage report\n"
                                "  'export Results Card' action (RESULTS_CARD.md format)\n\n"
                                "[dim]Not wired — Phase 4[/dim]",
                            )
        yield Footer()

    # ── Mount / startup ───────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._load_catalog()

    @work(thread=True)
    def _load_catalog(self) -> None:
        """Load the catalog in a background thread (the only script call)."""
        rows, error = load_catalog_sync(self._repo_root)
        self.call_from_thread(self._apply_catalog, rows, error)

    def _apply_catalog(self, rows: list[VariantRow], error: str | None) -> None:
        try:
            pane = self.query_one("#catalog-pane", CatalogPane)
            pane.populate(rows, error)
        except Exception:
            pass

    # ── Mode switching ────────────────────────────────────────────────────────

    def _switch_mode(self, index: int) -> None:
        panel_ids = ["panel-discover", "panel-serve", "panel-estate", "panel-validate"]
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

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_mode_discover(self) -> None:
        self._switch_mode(0)

    def action_mode_serve(self) -> None:
        self._switch_mode(1)

    def action_mode_estate(self) -> None:
        self._switch_mode(2)

    def action_mode_validate(self) -> None:
        self._switch_mode(3)

    def action_refresh_catalog(self) -> None:
        """Re-read the catalog (the only real data source in Phase 1)."""
        try:
            pane = self.query_one("#catalog-pane", CatalogPane)
            pane.query_one("#catalog-status", Label).update("Refreshing catalog…")
        except Exception:
            pass
        self._load_catalog()

    def action_primary_action(self) -> None:
        """⏎ primary action — no-op in Phase 1, pops an informational toast."""
        self.notify(
            "Action not wired yet — all row/scene/container actions are wired in Phase 3.",
            title="Wired in Phase 3",
            severity="information",
            timeout=4,
        )

    def action_help(self) -> None:
        self.push_screen(HelpScreen())
