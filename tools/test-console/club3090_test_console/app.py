"""Main Textual application for the club3090 test console."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, Static, Button, RichLog, Select

from .detect import ServingTarget, detect_endpoint, detect_from_registry, match_target_to_registry, get_gpu_info
from .parsers import ParseEvent, TestType, Status
from .runner import TestConfig, TestRunner, RunState
from .widgets.target_pane import TargetPane
from .widgets.test_menu import TestMenuPane, TestEntry
from .widgets.live_pane import LivePane
from .widgets.history_view import HistoryScreen
from .widgets.manual_target import ManualTargetScreen


class ConfigScreen(ModalScreen[Optional[TestConfig]]):
    """Modal screen for configuring a test run."""

    DEFAULT_CSS = """
    ConfigScreen {
        align: center middle;
    }
    ConfigScreen > Vertical {
        width: 70;
        height: auto;
        max-height: 90%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    ConfigScreen #config-fields {
        height: auto;
        max-height: 1fr;
        overflow-y: auto;
    }
    ConfigScreen .config-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
    }
    ConfigScreen .config-hint {
        color: $text-muted;
        height: auto;
    }
    ConfigScreen Label {
        margin-top: 1;
    }
    ConfigScreen Select {
        margin-bottom: 1;
    }
    ConfigScreen Input {
        margin-bottom: 1;
    }
    ConfigScreen #button-bar {
        height: 3;
    }
    ConfigScreen Button {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, test_entry: TestEntry, current_config: Optional[TestConfig] = None):
        super().__init__()
        self.test_entry = test_entry
        self._config = current_config or TestConfig(test_type=test_entry.test_type)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Configure: {self.test_entry.display_name}", classes="config-title")
            yield Label(self._get_help_text(), classes="config-hint")
            with Vertical(id="config-fields"):
                yield from self._compose_config_fields()
            with Horizontal(id="button-bar"):
                yield Button("Run", variant="primary", id="btn-run")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def _get_help_text(self) -> str:
        match self.test_entry.test_type:
            case TestType.BENCH:
                return "Tune RUNS, WARMUPS, prompt selection, and thinking mode."
            case TestType.VERIFY_FULL:
                return "Optionally skip tool-call check or add a bench run."
            case TestType.VERIFY_STRESS:
                return "Control which stress probes to run (longctx, tool-prefill, ceiling)."
            case TestType.QUALITY:
                return "Choose tier, pack, thinking mode, and repeat count."
            case TestType.SOAK:
                return "Set session/turn counts and VRAM growth limits."
            case TestType.REBENCH:
                return "Configure 8-pack quality toggle, skips, and soak sizing."
            case _:
                return "Configure test parameters."

    def _compose_config_fields(self) -> ComposeResult:
        """Yield config fields based on test type."""
        tt = self.test_entry.test_type
        c = self._config

        # Helper for boolean select
        bool_options = [("No", False), ("Yes", True)]
        
        if tt == TestType.BENCH:
            yield Label("RUNS (measured runs per prompt):")
            yield Input(str(c.run_count), id="cfg-runs", type="integer")
            yield Label("WARMUPS:")
            yield Input(str(c.warmups), id="cfg-warmups", type="integer")
            yield Label("ONLY (prompt selection):")
            yield Select(
                [("Both (narrative + code)", "both"), ("Narrative only", "narr"), ("Code only", "code")],
                value=c.only,
                id="cfg-only"
            )
            yield Label("Enable thinking:")
            yield Select(bool_options, value=c.enable_thinking, id="cfg-thinking")

        elif tt == TestType.VERIFY_FULL:
            yield Label("Skip tools check:")
            yield Select(bool_options, value=c.skip_tools, id="cfg-skip-tools")
            yield Label("Run bench after:")
            yield Select(bool_options, value=c.run_bench, id="cfg-run-bench")

        elif tt == TestType.VERIFY_STRESS:
            yield Label("Skip long-context:")
            yield Select(bool_options, value=c.skip_longctx, id="cfg-skip-longctx")
            yield Label("Skip tool prefill:")
            yield Select(bool_options, value=c.skip_tool_prefill, id="cfg-skip-prefill")
            yield Label("Skip ceiling ladder:")
            yield Select(bool_options, value=c.skip_ceiling, id="cfg-skip-ceiling")

        elif tt == TestType.QUALITY:
            yield Label("Tier:")
            yield Select(
                [("Quick (2 packs)", "quick"), ("Medium (5 packs)", "medium"), 
                 ("Full (8 packs)", "full"), ("Reasoning", "reasoning")],
                value=c.quality_tier,
                id="cfg-tier"
            )
            yield Label("Pack ID (empty = all for tier):")
            yield Input(c.quality_pack, id="cfg-pack", placeholder="e.g., toolcall-15")
            yield Label("Enable thinking:")
            yield Select(bool_options, value=c.enable_thinking, id="cfg-thinking")
            yield Label("Repeat count:")
            yield Input(str(c.quality_repeat), id="cfg-repeat", type="integer")
            yield Label("Max tokens (0 = default):")
            yield Input(str(c.max_tokens) if c.max_tokens > 0 else "", id="cfg-max-tokens", 
                       type="integer", placeholder="auto")
            yield Label("Thinking max tokens (0 = default):")
            yield Input(str(c.thinking_max_tokens) if c.thinking_max_tokens > 0 else "", 
                       id="cfg-thinking-max-tokens", type="integer", placeholder="auto")

        elif tt == TestType.SOAK:
            yield Label("Mode:")
            yield Select(
                [("Fresh (cold start)", "fresh"), ("Continuous (warm)", "continuous"), ("Quick (short)", "quick")],
                value=c.soak_mode,
                id="cfg-mode"
            )
            yield Label("Sessions:")
            yield Input(str(c.soak_sessions), id="cfg-sessions", type="integer")
            yield Label("Turns per session:")
            yield Input(str(c.soak_turns), id="cfg-turns", type="integer")
            yield Label("Max VRAM growth (MiB):")
            yield Input(str(c.soak_max_growth), id="cfg-growth", type="integer")

        elif tt == TestType.REBENCH:
            yield Label("8-pack thinking:")
            yield Select(
                [("None (fast gates only)", ""), ("Off", "off"), ("On", "on"), ("Both (promotion gate)", "both")],
                value=c.rebench_8pack,
                id="cfg-8pack"
            )
            yield Label("Skip steps (CSV: verify-full,bench,...):")
            yield Input(",".join(c.rebench_skip), id="cfg-skip", placeholder="e.g., bench,soak")
            yield Label("Resume:")
            yield Select(bool_options, value=c.rebench_resume, id="cfg-resume")
            yield Label("Tag (empty = auto):")
            yield Input(c.rebench_tag, id="cfg-tag", placeholder="auto-generated")
            yield Label("SOAK_SESSIONS:")
            yield Input(str(c.soak_sessions), id="cfg-sessions", type="integer")
            yield Label("Max tokens (0 = default):")
            yield Input(str(c.max_tokens) if c.max_tokens > 0 else "", id="cfg-max-tokens", 
                       type="integer", placeholder="auto")
            yield Label("Thinking max tokens (0 = default):")
            yield Input(str(c.thinking_max_tokens) if c.thinking_max_tokens > 0 else "", 
                       id="cfg-thinking-max-tokens", type="integer", placeholder="auto")

    def _read_config(self) -> TestConfig:
        """Read values from widgets into config."""
        c = TestConfig(test_type=self.test_entry.test_type)
        tt = self.test_entry.test_type

        def get_select_val(id: str, default):
            try:
                widget = self.query_one(f"#{id}", Select)
                val = widget.value
                return val if val != Select.BLANK else default
            except Exception:
                return default

        def get_input_val(id: str, default: str = "") -> str:
            try:
                return self.query_one(f"#{id}", Input).value
            except Exception:
                return default

        def get_int(id: str, default: int = 0) -> int:
            try:
                val = get_input_val(id, str(default))
                return int(val) if val else default
            except ValueError:
                return default

        if tt == TestType.BENCH:
            c.run_count = get_int("cfg-runs", 5)
            c.warmups = get_int("cfg-warmups", 3)
            c.only = get_select_val("cfg-only", "both")
            c.enable_thinking = get_select_val("cfg-thinking", False)

        elif tt == TestType.VERIFY_FULL:
            c.skip_tools = get_select_val("cfg-skip-tools", False)
            c.run_bench = get_select_val("cfg-run-bench", False)

        elif tt == TestType.VERIFY_STRESS:
            c.skip_longctx = get_select_val("cfg-skip-longctx", False)
            c.skip_tool_prefill = get_select_val("cfg-skip-prefill", False)
            c.skip_ceiling = get_select_val("cfg-skip-ceiling", False)

        elif tt == TestType.QUALITY:
            c.quality_tier = get_select_val("cfg-tier", "medium")
            c.quality_pack = get_input_val("cfg-pack")
            c.enable_thinking = get_select_val("cfg-thinking", False)
            c.quality_repeat = get_int("cfg-repeat", 1)
            c.max_tokens = get_int("cfg-max-tokens", 0)
            c.thinking_max_tokens = get_int("cfg-thinking-max-tokens", 0)

        elif tt == TestType.SOAK:
            c.soak_mode = get_select_val("cfg-mode", "fresh")
            c.soak_sessions = get_int("cfg-sessions", 10)
            c.soak_turns = get_int("cfg-turns", 5)
            c.soak_max_growth = get_int("cfg-growth", 200)

        elif tt == TestType.REBENCH:
            c.rebench_8pack = get_select_val("cfg-8pack", "")
            skip_csv = get_input_val("cfg-skip")
            c.rebench_skip = [s.strip() for s in skip_csv.split(",") if s.strip()]
            c.rebench_resume = get_select_val("cfg-resume", False)
            c.rebench_tag = get_input_val("cfg-tag")
            c.soak_sessions = get_int("cfg-sessions", 10)
            c.max_tokens = get_int("cfg-max-tokens", 0)
            c.thinking_max_tokens = get_int("cfg-thinking-max-tokens", 0)

        return c

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            config = self._read_config()
            self.dismiss(config)
        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class HelpScreen(ModalScreen):
    """Help overlay showing keybindings."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen > Vertical {
        width: 70;
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

    HELP_TEXT = """
[bold]Keybindings[/bold]

  [cyan]↑/↓ or j/k[/cyan]    Move selection in test menu
  [cyan]Enter[/cyan]          Run selected test (default config)
  [cyan]c[/cyan]              Configure selected test
  [cyan]x[/cyan]              Stop current run
  [cyan]r[/cyan]              Re-detect serving target
  [cyan]m[/cyan]              Manual target override
  [cyan]f[/cyan]              Toggle log follow/scroll-lock
  [cyan]Tab[/cyan]            Cycle pane focus
  [cyan]1/2/3[/cyan]          Jump to pane (Target/Tests/Live)
  [cyan]?[/cyan]              Show this help
  [cyan]q[/cyan]              Quit

[bold]Status glyphs[/bold]

  [green]✓[/green] passed    [red]✗[/red] failed    [yellow]△[/yellow] partial/recall-miss
  [yellow]⊘[/yellow] skipped   [cyan]▶[/cyan] running   ◔ queued
"""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("club3090 test console — Help", classes="help-title")
            yield Static(self.HELP_TEXT)

    def action_dismiss(self) -> None:
        self.app.pop_screen()


class QuitConfirmScreen(ModalScreen[bool]):
    """Confirm quit when a test is running. Offers stop & quit or cancel."""

    DEFAULT_CSS = """
    QuitConfirmScreen {
        align: center middle;
    }
    QuitConfirmScreen > Vertical {
        width: 60;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    QuitConfirmScreen .quit-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }
    QuitConfirmScreen .quit-info {
        color: $text-muted;
        text-align: center;
        margin-bottom: 1;
    }
    QuitConfirmScreen Button {
        margin: 1 1;
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, test_name: str):
        super().__init__()
        self.test_name = test_name

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("⚠ Test Running", classes="quit-title")
            yield Label(f"{self.test_name} is still running.", classes="quit-info")
            yield Label("Choose an action:", classes="quit-info")
            with Horizontal():
                yield Button("Stop & Quit", variant="error", id="btn-stop-quit")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-stop-quit":
            self.dismiss(True)  # stop and quit
        elif event.button.id == "btn-cancel":
            self.dismiss(False)  # cancel

    def action_cancel(self) -> None:
        self.dismiss(False)


class TestConsoleApp(App):
    """The main club3090 test console application."""

    TITLE = "club3090 test console"
    CSS_PATH = "app.tcss"
    
    BINDINGS = [
        Binding("q", "safe_quit", "Quit", show=True),
        Binding("question_mark", "help", "Help", show=True),
        Binding("r", "redetect", "Re-detect", show=True),
        Binding("c", "config", "Configure", show=True),
        Binding("x", "stop", "Stop", show=True),
        Binding("h", "history", "History", show=True),
        Binding("f", "toggle_follow", "Follow", show=False),
        Binding("m", "manual_target", "Target", show=True),
        Binding("enter", "run_test", "Run", show=True),
    ]

    def __init__(self, repo_root: Path, **kwargs):
        super().__init__(**kwargs)
        self.repo_root = repo_root
        self.target: Optional[ServingTarget] = None
        self.registry_variants: list[dict] = []
        self.runner = TestRunner(repo_root)
        self._gpu_refresh_task: Optional[asyncio.Task] = None
        self._state_dir = Path.home() / ".local" / "state" / "club3090-test-console"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-layout"):
            with Vertical(id="left-rail"):
                yield TargetPane(id="target-pane")
                yield TestMenuPane(id="test-menu")
            yield LivePane(id="live-pane")
        yield Footer()

    CSS = """
    #main-layout {
        height: 1fr;
    }
    #left-rail {
        width: 38;
        height: 1fr;
    }
    #target-pane {
        height: auto;
        min-height: 8;
    }
    #test-menu {
        height: 1fr;
    }
    #live-pane {
        width: 1fr;
        height: 1fr;
    }
    """

    async def on_mount(self) -> None:
        """Initialize: detect target, load registry, set up runner."""
        self.runner.set_callbacks(
            on_event=self._on_run_event,
            on_line=self._on_run_line,
            on_complete=self._on_run_complete,
        )
        # Ensure state dir exists
        self._state_dir.mkdir(parents=True, exist_ok=True)

        # Start detection in background
        asyncio.create_task(self._initial_detect())

        # Set up elapsed timer update
        self.set_interval(1.0, self._update_elapsed_timer)

    async def on_unmount(self) -> None:
        """Clean up background tasks on app shutdown."""
        if self._gpu_refresh_task and not self._gpu_refresh_task.done():
            self._gpu_refresh_task.cancel()
            # Don't await - just cancel to avoid event loop cleanup issues

    async def _initial_detect(self) -> None:
        """Detect serving target and load registry."""
        # Detect endpoint
        self.target = await detect_endpoint()

        # Load registry for enrichment
        self.registry_variants = await detect_from_registry(str(self.repo_root))
        if self.target and self.registry_variants:
            self.target = match_target_to_registry(self.target, self.registry_variants)

        # Update the target pane
        self._update_target_pane()

        # Cancel old GPU refresh loop if exists, then start new one
        if self._gpu_refresh_task and not self._gpu_refresh_task.done():
            self._gpu_refresh_task.cancel()
            try:
                await self._gpu_refresh_task
            except asyncio.CancelledError:
                pass
        self._gpu_refresh_task = asyncio.create_task(self._gpu_refresh_loop())

    async def _gpu_refresh_loop(self) -> None:
        """Periodically refresh GPU stats."""
        try:
            while True:
                await asyncio.sleep(2)
                if self.target and self.target.is_active:
                    self.target.gpus = await get_gpu_info()
                    self._update_target_pane()
        except asyncio.CancelledError:
            pass

    def _update_elapsed_timer(self) -> None:
        """Update the elapsed timer in the live pane."""
        try:
            live = self.query_one("#live-pane", LivePane)
            live.update_elapsed_timer()
        except Exception:
            pass

    def _update_target_pane(self) -> None:
        """Update the target pane with current target info."""
        try:
            pane = self.query_one("#target-pane", TargetPane)
            pane.target = self.target
        except Exception:
            pass

    def _on_run_event(self, event: ParseEvent) -> None:
        """Called when the runner parses a structured event."""
        # Capture test_type at scheduling time before current_run may be cleared
        test_type = self.runner.current_run.test_type if self.runner.current_run else None
        self.call_later(self._handle_run_event, event, test_type)

    def _handle_run_event(self, event: ParseEvent, test_type: TestType | None = None) -> None:
        """Handle a run event on the UI thread."""
        try:
            live = self.query_one("#live-pane", LivePane)
            if test_type:
                live.process_event(event, test_type)
        except Exception:
            pass

    def _on_run_line(self, line: str) -> None:
        """Called for every stdout line from the runner."""
        self.call_later(self._handle_run_line, line)

    def _handle_run_line(self, line: str) -> None:
        """Handle a raw log line on the UI thread."""
        try:
            live = self.query_one("#live-pane", LivePane)
            live.append_line(line)
        except Exception:
            pass

    def _on_run_complete(self, state: RunState) -> None:
        """Called when a test run completes."""
        self.call_later(self._handle_run_complete, state)

    def _handle_run_complete(self, state: RunState) -> None:
        """Handle run completion on the UI thread."""
        # Update menu status
        try:
            menu = self.query_one("#test-menu", TestMenuPane)
            status = "passed" if state.verdict == "passed" else "failed"
            menu.set_status(state.test_type, status)
        except Exception:
            pass

        # Save run record
        self._save_run_record(state)

        # Show completion in log
        try:
            live = self.query_one("#live-pane", LivePane)
            elapsed = state.elapsed_s
            if state.verdict == "passed":
                live.append_line(f"\n[bold green]✓ {state.test_type.value} passed in {elapsed:.0f}s[/bold green]")
            else:
                live.append_line(f"\n[bold red]✗ {state.test_type.value} failed (rc={state.exit_code}) in {elapsed:.0f}s[/bold red]")
            if state.report_path:
                live.append_line(f"  report: {state.report_path}")
            if state.artifact_dir:
                live.append_line(f"  artifacts: {state.artifact_dir}")
        except Exception:
            pass

    def _save_run_record(self, state: RunState) -> None:
        """Persist a run record to disk."""
        runs_dir = self._state_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "test": state.test_type.value,
            "model": state.target.model if state.target else "",
            "url": state.target.url if state.target else "",
            "slug": state.target.slug if state.target else "",
            "started": state.started,
            "finished": state.finished,
            "exit_code": state.exit_code,
            "verdict": state.verdict,
            "elapsed_s": state.elapsed_s,
            "artifact_dir": state.artifact_dir,
            "report_path": state.report_path,
            "power_cap_w": (
                state.target.gpus[0].power_limit_w
                if state.target and state.target.gpus
                else None
            ),
        }

        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(state.started))
        path = runs_dir / f"{ts}-{state.test_type.value}.json"
        try:
            path.write_text(json.dumps(record, indent=2))
        except Exception:
            pass

    # ── Actions ──────────────────────────────────────────────────────────

    async def action_run_test(self) -> None:
        """Run the selected test."""
        menu = self.query_one("#test-menu", TestMenuPane)
        entry = menu.get_selected_entry()
        if not entry:
            return

        # Check if a run is active
        if self.runner.current_run:
            self.notify("A test is already running. Press x to stop first.", severity="warning")
            return

        # Check target
        if not self.target or not self.target.is_active:
            self.notify("No model serving. Start one with `gpu-mode <mode>`.", severity="error")
            return

        config = TestConfig(test_type=entry.test_type)
        await self._start_run(config)

    async def action_config(self) -> None:
        """Open config for the selected test."""
        menu = self.query_one("#test-menu", TestMenuPane)
        entry = menu.get_selected_entry()
        if not entry:
            return

        def on_config_dismiss(config: Optional[TestConfig]) -> None:
            if config:
                self.run_worker(self._start_run(config))

        self.push_screen(ConfigScreen(entry), callback=on_config_dismiss)

    async def _start_run(self, config: TestConfig) -> None:
        """Start a test run with the given config."""
        # Guard against starting a second run while one is already active
        if self.runner.current_run:
            self.notify("A test is already running. Press x to stop first.", severity="warning")
            return
        
        if not self.target:
            self.notify("No serving target detected.", severity="error")
            return

        # Update menu status
        try:
            menu = self.query_one("#test-menu", TestMenuPane)
            menu.set_status(config.test_type, "running")
        except Exception:
            pass

        # Set up live pane
        try:
            live = self.query_one("#live-pane", LivePane)
            live.set_run_header(config.test_type, self.target.model)
        except Exception:
            pass

        # Start the run
        await self.runner.start(config, self.target)

    async def action_stop(self) -> None:
        """Stop the current run."""
        if self.runner.current_run:
            test_type = self.runner.current_run.test_type
            orphans = await self.runner.cancel()
            
            if orphans:
                orphan_names = ' '.join(orphans)
                self.notify(
                    f"Test cancelled. ⚠ Orphaned benchlocal containers: {', '.join(orphans)}. "
                    f"Run: docker rm -f {orphan_names}",
                    severity="warning",
                    timeout=10,
                )
            else:
                self.notify("Test cancelled.", severity="warning")
            
            try:
                menu = self.query_one("#test-menu", TestMenuPane)
                menu.set_status(test_type, "idle")
            except Exception:
                pass
        else:
            self.notify("No active run.", severity="information")

    async def action_redetect(self) -> None:
        """Re-detect the serving target."""
        self.notify("Re-detecting...", severity="information")
        await self._initial_detect()
        if self.target and self.target.is_active:
            self.notify(f"Found: {self.target.model} on :{self.target.host_port}", severity="information")
        else:
            self.notify("No model serving.", severity="warning")

    def action_help(self) -> None:
        """Show the help screen."""
        self.push_screen(HelpScreen())

    def action_toggle_follow(self) -> None:
        """Toggle log follow mode."""
        try:
            live = self.query_one("#live-pane", LivePane)
            live.toggle_follow()
            follow_state = "on" if live._follow else "off"
            self.notify(f"Log follow: {follow_state}", severity="information")
        except Exception:
            pass

    async def action_manual_target(self) -> None:
        """Open manual target override."""
        variants = self.registry_variants
        if not variants:
            variants = await detect_from_registry(str(self.repo_root))

        def on_target_dismiss(result: Optional[ServingTarget]) -> None:
            if result:
                self.target = result
                self._update_target_pane()
                self.notify(f"Target set: {result.model} @ {result.url}", severity="information")

        self.push_screen(ManualTargetScreen(str(self.repo_root), variants), callback=on_target_dismiss)

    def action_history(self) -> None:
        """Open run history view."""
        self.push_screen(HistoryScreen(self._state_dir, self.repo_root))

    async def action_safe_quit(self) -> None:
        """Quit with confirmation if a test is running."""
        if not self.runner.current_run:
            # No active run, just quit
            self.exit()
            return

        # Show confirmation dialog
        test_name = self.runner.current_run.test_type.value
        
        def on_quit_choice(stop_and_quit: bool) -> None:
            if stop_and_quit:
                # Stop & quit
                self.run_worker(self._stop_and_quit())
            # else: cancel (do nothing)
        
        self.push_screen(QuitConfirmScreen(test_name), callback=on_quit_choice)

    async def _stop_and_quit(self) -> None:
        """Stop the current run and quit."""
        if self.runner.current_run:
            await self.runner.cancel()
        self.exit()
