"""Test menu pane — lists available tests and their status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, ListItem, ListView, Static

from ..parsers import TestType


@dataclass
class TestEntry:
    """A test menu entry."""
    test_type: TestType
    display_name: str
    duration_hint: str
    description: str = ""


# The test catalog from spec Section 3
TEST_CATALOG = [
    TestEntry(TestType.VERIFY, "Smoke (verify)", "~15s", "Quick server reachability check"),
    TestEntry(TestType.VERIFY_FULL, "Functional (verify-full)", "~2min", "Full functional test suite"),
    TestEntry(TestType.BENCH, "Speed bench (TPS)", "~5min", "Throughput benchmark"),
    TestEntry(TestType.VERIFY_STRESS, "Stress / NIAH", "~15min", "Long-context + boundary tests"),
    TestEntry(TestType.QUALITY, "Quality packs", "5-90min", "Behavioral quality testing"),
    TestEntry(TestType.SOAK, "Soak / stability", "~20min", "Long-running stability test"),
    TestEntry(TestType.REBENCH, "★ FULL rebench (macro)", "~45min", "Complete test pipeline"),
]

STATUS_GLYPHS = {
    "idle": "○",
    "running": "▶",
    "passed": "✓",
    "failed": "✗",
    "skipped": "⊘",
    "queued": "◔",
}


class TestMenuPane(Static):
    """Left-bottom pane showing the test menu."""

    selected_index: reactive[int] = reactive(0)

    DEFAULT_CSS = """
    TestMenuPane {
        width: 100%;
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
    }
    TestMenuPane .pane-title {
        text-style: bold;
        color: $accent;
        dock: top;
    }
    TestMenuPane ListView {
        height: 1fr;
    }
    TestMenuPane ListView > ListItem {
        height: 2;
        padding: 0 1;
    }
    TestMenuPane ListView > ListItem.--highlight {
        background: $boost;
    }
    TestMenuPane .test-status {
        width: 3;
    }
    TestMenuPane .test-name {
        width: 1fr;
    }
    TestMenuPane .test-duration {
        width: 8;
        text-align: right;
        color: $text-muted;
    }
    """

    class TestSelected(Message):
        """Fired when a test is selected."""
        def __init__(self, entry: TestEntry) -> None:
            self.entry = entry
            super().__init__()

    class TestActivated(Message):
        """Fired when a test is activated (Enter pressed)."""
        def __init__(self, entry: TestEntry) -> None:
            self.entry = entry
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._entries = TEST_CATALOG
        self._statuses: dict[TestType, str] = {t.test_type: "idle" for t in TEST_CATALOG}

    def compose(self) -> ComposeResult:
        yield Label("Tests", classes="pane-title")
        with ListView(id="test-list"):
            for entry in self._entries:
                with ListItem():
                    yield Label(self._format_entry(entry), classes="test-entry")

    def _format_entry(self, entry: TestEntry) -> str:
        status = self._statuses.get(entry.test_type, "idle")
        glyph = STATUS_GLYPHS.get(status, "○")
        color = {
            "idle": "",
            "running": "[cyan]",
            "passed": "[green]",
            "failed": "[red]",
            "skipped": "[yellow]",
            "queued": "[dim]",
        }.get(status, "")
        end_color = f"[/{color.rstrip(']')}]" if color else ""
        # Strip brackets for rich markup
        if color:
            color_clean = color.rstrip("]").lstrip("[")
            end_clean = f"[/{color_clean}]"
        else:
            color_clean = ""
            end_clean = ""
        return f"{color}{glyph} {entry.display_name:<25} {entry.duration_hint}{end_clean}"

    def set_status(self, test_type: TestType, status: str) -> None:
        """Update the status of a test entry."""
        self._statuses[test_type] = status
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Refresh the list display."""
        try:
            list_view = self.query_one("#test-list", ListView)
            # Update labels
            for i, entry in enumerate(self._entries):
                items = list_view.children
                if i < len(items):
                    label = items[i].query_one(Label)
                    label.update(self._format_entry(entry))
        except Exception:
            pass

    def get_selected_entry(self) -> Optional[TestEntry]:
        """Get the currently selected test entry."""
        try:
            list_view = self.query_one("#test-list", ListView)
            idx = list_view.index
            if 0 <= idx < len(self._entries):
                return self._entries[idx]
        except Exception:
            pass
        return None

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list selection."""
        entry = self.get_selected_entry()
        if entry:
            self.post_message(self.TestSelected(entry))

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle list highlight change."""
        entry = self.get_selected_entry()
        if entry:
            self.post_message(self.TestSelected(entry))
