"""History view — browse past runs and existing artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, RichLog, Static, DataTable

from ..parsers import TestType


class RunRecord:
    """A single run record from history."""

    def __init__(self, data: dict):
        self.test = data.get("test", "unknown")
        self.model = data.get("model", "")
        self.url = data.get("url", "")
        self.slug = data.get("slug", "")
        self.started = data.get("started", 0)
        self.finished = data.get("finished", 0)
        self.exit_code = data.get("exit_code", -1)
        self.verdict = data.get("verdict", "unknown")
        self.elapsed_s = data.get("elapsed_s", 0)
        self.artifact_dir = data.get("artifact_dir", "")
        self.report_path = data.get("report_path", "")
        self.power_cap_w = data.get("power_cap_w")
        self.log_path = data.get("log_path", "")

    @property
    def timestamp_str(self) -> str:
        if self.started:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.started))
        return "unknown"

    @property
    def status_glyph(self) -> str:
        if self.verdict == "passed":
            return "[green]✓[/green]"
        elif self.verdict == "failed":
            return "[red]✗[/red]"
        return "[dim]?[/dim]"


class RunDetailScreen(ModalScreen):
    """Show details of a single run with report/log content."""

    DEFAULT_CSS = """
    RunDetailScreen {
        align: center middle;
    }
    RunDetailScreen > Vertical {
        width: 90%;
        height: 90%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    RunDetailScreen .detail-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        height: 1;
    }
    RunDetailScreen #detail-content {
        height: 1fr;
    }
    RunDetailScreen #button-bar {
        height: 3;
    }
    RunDetailScreen Button {
        width: 1fr;
    }
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, record: RunRecord, repo_root: Path):
        super().__init__()
        self.record = record
        self.repo_root = repo_root
        self._current_view = "summary"  # summary, report, log

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Run Details: {self.record.test}", classes="detail-title")
            yield RichLog(id="detail-content")
            with Horizontal(id="button-bar"):
                yield Button("Summary", variant="primary", id="btn-summary")
                if self.record.report_path:
                    yield Button("Report", variant="default", id="btn-report")
                if self.record.log_path:
                    yield Button("Log", variant="default", id="btn-log")
                yield Button("Close", variant="default", id="btn-close")

    def on_mount(self) -> None:
        self._show_summary()

    def _show_summary(self) -> None:
        """Display run summary."""
        self._current_view = "summary"
        content = self.query_one("#detail-content", RichLog)
        content.clear()
        r = self.record
        content.write(f"[bold]Test:[/bold]      {r.test}")
        content.write(f"[bold]Model:[/bold]     {r.model}")
        content.write(f"[bold]URL:[/bold]       {r.url}")
        content.write(f"[bold]Slug:[/bold]      {r.slug or '(none)'}")
        content.write(f"[bold]Started:[/bold]   {r.timestamp_str}")
        content.write(f"[bold]Elapsed:[/bold]   {r.elapsed_s:.1f}s")
        content.write(f"[bold]Exit code:[/bold] {r.exit_code}")
        content.write(f"[bold]Verdict:[/bold]   {r.status_glyph} {r.verdict}")
        if r.power_cap_w:
            content.write(f"[bold]Power cap:[/bold] {r.power_cap_w:.0f}W")
        if r.artifact_dir:
            content.write(f"\n[bold]Artifacts:[/bold] {r.artifact_dir}")
        if r.report_path:
            content.write(f"[bold]Report:[/bold]    {r.report_path}")
        if r.log_path:
            content.write(f"[bold]Log:[/bold]       {r.log_path}")

    def _show_report(self) -> None:
        """Load and display the report file."""
        if not self.record.report_path:
            return
        self._current_view = "report"
        content = self.query_one("#detail-content", RichLog)
        content.clear()

        report_path = Path(self.record.report_path)
        # Make relative to repo root if needed
        if not report_path.is_absolute():
            report_path = self.repo_root / report_path

        if report_path.exists():
            try:
                text = report_path.read_text()
                # Render markdown-ish content
                for line in text.split("\n"):
                    if line.startswith("# "):
                        content.write(f"[bold cyan]{line[2:]}[/bold cyan]")
                    elif line.startswith("## "):
                        content.write(f"[bold]{line[3:]}[/bold]")
                    elif line.startswith("### "):
                        content.write(f"[bold yellow]{line[4:]}[/bold yellow]")
                    elif line.startswith("- "):
                        content.write(f"  • {line[2:]}")
                    elif line.startswith("```"):
                        content.write("[dim]" + line + "[/dim]")
                    else:
                        content.write(line)
            except Exception as e:
                content.write(f"[red]Error reading report: {e}[/red]")
        else:
            content.write(f"[red]Report file not found: {report_path}[/red]")

    def _show_log(self) -> None:
        """Load and display the log file."""
        if not self.record.log_path:
            return
        self._current_view = "log"
        content = self.query_one("#detail-content", RichLog)
        content.clear()

        log_path = Path(self.record.log_path)
        if not log_path.is_absolute():
            log_path = self.repo_root / log_path

        if log_path.exists():
            try:
                text = log_path.read_text()
                # Show last 500 lines
                lines = text.split("\n")
                if len(lines) > 500:
                    content.write(f"[dim]... showing last 500 of {len(lines)} lines ...[/dim]\n")
                    lines = lines[-500:]
                for line in lines:
                    content.write(line)
            except Exception as e:
                content.write(f"[red]Error reading log: {e}[/red]")
        else:
            content.write(f"[red]Log file not found: {log_path}[/red]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close":
            self.app.pop_screen()
        elif event.button.id == "btn-summary":
            self._show_summary()
        elif event.button.id == "btn-report":
            self._show_report()
        elif event.button.id == "btn-log":
            self._show_log()

    def action_dismiss(self) -> None:
        self.app.pop_screen()


class HistoryScreen(ModalScreen):
    """Browse past runs from ~/.local/state and results/ artifacts."""

    DEFAULT_CSS = """
    HistoryScreen {
        align: center middle;
    }
    HistoryScreen > Vertical {
        width: 90%;
        height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    HistoryScreen .history-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
    }
    HistoryScreen DataTable {
        height: 1fr;
    }
    HistoryScreen .history-hint {
        text-align: center;
        color: $text-muted;
        height: 1;
    }
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, state_dir: Path, repo_root: Path):
        super().__init__()
        self.state_dir = state_dir
        self.repo_root = repo_root
        self.records: list[RunRecord] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Run History", classes="history-title")
            yield DataTable(id="history-table", cursor_type="row")
            yield Label("Press Enter or click a row to view details", classes="history-hint")
            yield Button("Close", variant="default", id="btn-close")

    def on_mount(self) -> None:
        self._load_records()
        table = self.query_one("#history-table", DataTable)
        table.add_columns("Time", "Test", "Model", "Verdict", "Elapsed", "Power")
        for i, record in enumerate(self.records):
            table.add_row(
                record.timestamp_str,
                record.test,
                record.model[:20],
                record.status_glyph,
                f"{record.elapsed_s:.0f}s",
                f"{record.power_cap_w:.0f}W" if record.power_cap_w else "-",
                key=str(i),
            )
        # Focus the table
        table.focus()

    def _load_records(self) -> None:
        """Load run records from state dir and results/ artifacts."""
        # Load from state dir
        runs_dir = self.state_dir / "runs"
        if runs_dir.exists():
            for path in sorted(runs_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(path.read_text())
                    # Infer log path from artifact dir
                    if "artifact_dir" in data and data["artifact_dir"]:
                        artifact_dir = Path(data["artifact_dir"])
                        test_name = data.get("test", "")
                        log_path = artifact_dir / f"{test_name}.log"
                        if log_path.exists():
                            data["log_path"] = str(log_path)
                    self.records.append(RunRecord(data))
                except Exception:
                    pass

        # Discover results/ artifacts
        results_dir = self.repo_root / "results"
        if results_dir.exists():
            # Rebench results
            for tag_dir in sorted(results_dir.glob("rebench/*"), reverse=True):
                if (tag_dir / "REPORT.md").exists():
                    # Find log files in the directory
                    log_files = list(tag_dir.glob("*.log"))
                    log_path = str(log_files[0]) if log_files else ""

                    self.records.append(RunRecord({
                        "test": "rebench-full",
                        "model": tag_dir.name.split("-")[0] if "-" in tag_dir.name else tag_dir.name,
                        "started": tag_dir.stat().st_mtime,
                        "verdict": "passed",
                        "artifact_dir": str(tag_dir),
                        "report_path": str(tag_dir / "REPORT.md"),
                        "log_path": log_path,
                    }))

            # Quality results
            for qdir in sorted(results_dir.glob("quality"), reverse=True):
                for json_file in qdir.glob("quality-*.json"):
                    try:
                        data = json.loads(json_file.read_text())
                        self.records.append(RunRecord({
                            "test": "quality",
                            "model": data.get("model", "unknown"),
                            "started": json_file.stat().st_mtime,
                            "verdict": "passed" if data.get("totals", {}).get("score", 0) >= 0.8 else "failed",
                            "artifact_dir": str(json_file.parent),
                            "log_path": "",
                        }))
                    except Exception:
                        pass

        # Sort by time descending
        self.records.sort(key=lambda r: r.started, reverse=True)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the table."""
        try:
            row_key = str(event.row_key.value)
            idx = int(row_key)
            if 0 <= idx < len(self.records):
                record = self.records[idx]
                self.app.push_screen(RunDetailScreen(record, self.repo_root))
        except (ValueError, IndexError):
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close":
            self.app.pop_screen()

    def action_dismiss(self) -> None:
        self.app.pop_screen()
