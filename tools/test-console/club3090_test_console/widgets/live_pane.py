"""Live output pane — shows structured progress + raw log."""

from __future__ import annotations

import time
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, RichLog, Static

from ..parsers import (
    ParseEvent,
    TestType,
    BenchParser,
    VerifyParser,
    StressParser,
    QualityParser,
    SoakParser,
    RebenchParser,
)


class StructuredHeader(Static):
    """Structured progress display above the raw log."""

    DEFAULT_CSS = """
    StructuredHeader {
        width: 100%;
        height: auto;
        min-height: 3;
        max-height: 12;
        padding: 0 1;
        background: $boost;
    }
    StructuredHeader .header-title {
        text-style: bold;
        color: $accent;
    }
    StructuredHeader .progress-bar {
        height: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lines: list[str] = []

    def update_from_event(self, event: ParseEvent, test_type: TestType) -> None:
        """Update the structured display from a parsed event."""
        # Build display based on test type
        match test_type:
            case TestType.BENCH:
                self._update_bench(event)
            case TestType.VERIFY | TestType.VERIFY_FULL:
                self._update_verify(event)
            case TestType.VERIFY_STRESS:
                self._update_stress(event)
            case TestType.QUALITY:
                self._update_quality(event)
            case TestType.SOAK:
                self._update_soak(event)
            case TestType.REBENCH:
                self._update_rebench(event)

    def _update_bench(self, event: ParseEvent) -> None:
        if event.event_type == "bench_section":
            section = event.data.get("section", "").upper()
            self._lines.append(f"[bold cyan]═══ {section} ═══[/bold cyan]")
        elif event.event_type == "bench_run":
            run_type = event.data.get("type", "")
            run_num = event.data.get("run", 0)
            decode_tps = event.data.get("decode_tps", 0)
            ttft = event.data.get("ttft_ms", 0)
            if run_type == "run":
                self._lines.append(f"  run-{run_num}  decode [green]{decode_tps:.1f}[/green] TPS  ttft {ttft}ms")
        elif event.event_type == "summary_metric":
            metric = event.data.get("metric", "")
            mean = event.data.get("mean", 0)
            cv = event.data.get("cv", 0)
            self._lines.append(f"  [bold]{metric}[/bold]  mean=[cyan]{mean:.1f}[/cyan]  CV={cv:.1f}%")
            self._lines = self._lines[-20:]  # Keep last 20 lines
        self._refresh_display()

    def _update_verify(self, event: ParseEvent) -> None:
        if event.event_type == "verify_step":
            step = event.data.get("step", 0)
            total = event.data.get("total", 0)
            name = event.data.get("name", "")
            self._lines.append(f"[cyan][{step}/{total}][/cyan] {name}")
        elif event.event_type == "verify_check":
            glyph = event.data.get("glyph", "")
            msg = event.data.get("message", "")
            color = {"✓": "green", "✗": "red", "⊘": "yellow"}.get(glyph, "")
            self._lines.append(f"  [{color}]{glyph}[/{color}] {msg}")
            self._lines = self._lines[-20:]
        elif event.event_type == "verdict":
            status = event.data.get("status", "")
            msg = event.data.get("message", "")
            if status == "passed":
                self._lines.append(f"[bold green]✓ {msg}[/bold green]")
            else:
                self._lines.append(f"[bold red]✗ {msg}[/bold red]")
        self._refresh_display()

    def _update_stress(self, event: ParseEvent) -> None:
        if event.event_type == "stress_probe":
            probe = event.data.get("probe", 0)
            name = event.data.get("name", "")
            self._lines.append(f"[cyan][{probe}/8][/cyan] {name}")
        elif event.event_type == "niah_rung":
            rung = event.data.get("rung", 0)
            total = event.data.get("total", 0)
            target_k = event.data.get("target_k", 0)
            glyph = event.data.get("glyph", "")
            status = event.data.get("status", "")
            color = {"passed": "green", "partial": "yellow", "failed": "red", "skipped": "dim"}.get(status, "")
            self._lines.append(f"  rung {rung}/{total}  [{color}]{glyph} {target_k}K[/{color}]")
            self._lines = self._lines[-20:]
        elif event.event_type == "niah_token":
            tokens = event.data.get("tokens", 0)
            glyph = event.data.get("glyph", "")
            status = event.data.get("status", "")
            color = {"passed": "green", "partial": "yellow", "failed": "red", "skipped": "dim"}.get(status, "")
            self._lines.append(f"  [{color}]{glyph}[/] {tokens:,} tokens")
        elif event.event_type == "verdict":
            status = event.data.get("status", "")
            msg = event.data.get("message", "")
            color = "green" if status == "passed" else "red"
            self._lines.append(f"[bold {color}]{msg}[/bold {color}]")
        self._refresh_display()

    def _update_quality(self, event: ParseEvent) -> None:
        if event.event_type == "quality_scenario":
            num = event.data.get("num", 0)
            total = event.data.get("total", 0)
            scenario_id = event.data.get("scenario_id", "")
            passed = event.data.get("passed", False)
            elapsed = event.data.get("elapsed_s", 0)
            glyph = "✓" if passed else "✗"
            color = "green" if passed else "red"
            self._lines.append(f"  [{num}/{total}] {scenario_id} [{color}]{glyph}[/{color}] ({elapsed:.1f}s)")
            self._lines = self._lines[-25:]
        elif event.event_type == "quality_total":
            passed = event.data.get("passed", 0)
            total = event.data.get("total", 0)
            pct = (passed / total * 100) if total > 0 else 0
            self._lines.append(f"[bold]TOTAL: {passed}/{total} ({pct:.0f}%)[/bold]")
        self._refresh_display()

    def _update_soak(self, event: ParseEvent) -> None:
        if event.event_type == "soak_session":
            session = event.data.get("session", 0)
            total = event.data.get("total", 0)
            self._lines.append(f"[cyan]session {session}/{total}[/cyan]")
        elif event.event_type == "soak_turn":
            turn = event.data.get("turn", 0)
            total = event.data.get("total", 0)
            tps = event.data.get("decode_tps", 0)
            vram = event.data.get("vram_mib", 0)
            self._lines.append(f"  turn {turn}/{total}  [green]{tps:.1f}[/green] TPS  {vram}MiB")
            self._lines = self._lines[-20:]
        elif event.event_type == "verdict":
            verdict = event.data.get("verdict", "")
            color = "green" if verdict == "PASS" else "red"
            self._lines.append(f"[bold {color}]verdict: {verdict}[/bold {color}]")
        elif event.event_type == "soak_metric":
            key = event.data.get("key", "")
            value = event.data.get("value", "")
            self._lines.append(f"  {key}: {value}")
        self._refresh_display()

    def _update_rebench(self, event: ParseEvent) -> None:
        if event.event_type == "rebench_step_start":
            step = event.data.get("step", "")
            self._lines.append(f"[cyan]▶ {step} running…[/cyan]")
        elif event.event_type == "rebench_step_done":
            step = event.data.get("step", "")
            status = event.data.get("status", "")
            elapsed = event.data.get("elapsed_s", 0)
            if status == "passed":
                self._lines.append(f"[green]✓ {step} {elapsed}s[/green]")
            elif status == "failed":
                rc = event.data.get("rc", 0)
                self._lines.append(f"[red]✗ {step} {elapsed}s (rc={rc})[/red]")
            elif status == "skipped":
                self._lines.append(f"[yellow]⊘ {step} skipped[/yellow]")
        elif event.event_type == "rebench_complete":
            self._lines.append("[bold green]═══ rebench complete ═══[/bold green]")
        elif event.event_type == "rebench_report":
            path = event.data.get("path", "")
            self._lines.append(f"  report: {path}")
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the widget display."""
        try:
            content = "\n".join(self._lines[-25:]) if self._lines else "Waiting for output..."
            self.update(content)
        except Exception:
            pass

    def clear(self) -> None:
        """Clear the structured header."""
        self._lines = []
        self.update("")


class LivePane(Static):
    """Right pane showing structured progress + raw log."""

    DEFAULT_CSS = """
    LivePane {
        width: 1fr;
        height: 1fr;
        border: solid $primary;
    }
    LivePane .live-title {
        text-style: bold;
        color: $accent;
        dock: top;
        padding: 0 1;
        height: 1;
    }
    LivePane StructuredHeader {
        dock: top;
        height: auto;
        max-height: 12;
    }
    LivePane RichLog {
        height: 1fr;
        overflow-y: auto;
    }
    """

    current_test: reactive[Optional[TestType]] = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._follow = True
        self._run_start_time = None

    def compose(self) -> ComposeResult:
        yield Label("Live", classes="live-title")
        yield StructuredHeader(id="structured-header")
        yield RichLog(id="live-log", wrap=True, highlight=True, markup=True)

    def on_mount(self) -> None:
        log = self.query_one("#live-log", RichLog)
        log.write("[dim]Ready. Select a test and press Enter to run.[/dim]")

    def append_line(self, line: str) -> None:
        """Append a raw log line."""
        try:
            log = self.query_one("#live-log", RichLog)
            log.write(line)
            if self._follow:
                log.scroll_end(animate=False)
        except Exception:
            pass

    def process_event(self, event: ParseEvent, test_type: TestType) -> None:
        """Process a parsed event and update the structured header."""
        try:
            header = self.query_one("#structured-header", StructuredHeader)
            header.update_from_event(event, test_type)
        except Exception:
            pass

    def clear_log(self) -> None:
        """Clear the log and structured header."""
        try:
            log = self.query_one("#live-log", RichLog)
            log.clear()
            header = self.query_one("#structured-header", StructuredHeader)
            header.clear()
        except Exception:
            pass

    def set_run_header(self, test_type: TestType, model: str, elapsed: str = "") -> None:
        """Set the header for a new run."""
        self.current_test = test_type
        self._run_start_time = time.time()
        self.clear_log()
        try:
            log = self.query_one("#live-log", RichLog)
            log.write(f"[bold cyan]▶ {test_type.value}[/bold cyan]  model={model}  {elapsed}")
            log.write("─" * 60)
        except Exception:
            pass

    def update_elapsed_timer(self) -> None:
        """Update the elapsed timer display."""
        if not self._run_start_time:
            return
        elapsed_s = time.time() - self._run_start_time
        minutes = int(elapsed_s) // 60
        seconds = int(elapsed_s) % 60
        try:
            title = self.query_one(".live-title", Label)
            title.update(f"Live  [dim]elapsed {minutes:02d}:{seconds:02d}[/dim]")
        except Exception:
            pass

    def toggle_follow(self) -> None:
        """Toggle log follow/scroll-lock."""
        self._follow = not self._follow
