"""Manual target override screen."""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, DataTable

from ..detect import ServingTarget


class ManualTargetScreen(ModalScreen[Optional[ServingTarget]]):
    """Manual target override — pick registry slug or enter external URL."""

    DEFAULT_CSS = """
    ManualTargetScreen {
        align: center middle;
    }
    ManualTargetScreen > Vertical {
        width: 80;
        height: 40;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    ManualTargetScreen .manual-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
    }
    ManualTargetScreen .section-title {
        text-style: bold;
        margin-top: 1;
    }
    ManualTargetScreen DataTable {
        height: 10;
    }
    ManualTargetScreen Input {
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, repo_root: str, variants: list):
        super().__init__()
        self.repo_root = repo_root
        self.variants = variants

    def _get_field(self, v, name: str, default=""):
        """Get a field from either a VariantRow object or a dict."""
        if hasattr(v, name):
            return getattr(v, name)
        return v.get(name, default)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Manual Target Override", classes="manual-title")

            yield Label("Option 1: Select from registry", classes="section-title")
            yield DataTable(id="registry-table")

            yield Label("Option 2: External endpoint", classes="section-title")
            yield Label("URL:")
            yield Input(placeholder="http://192.168.1.50:8887", id="input-url")
            yield Label("Model name:")
            yield Input(placeholder="qwen3.6-27b-autoround", id="input-model")
            yield Label("Engine (vllm/llamacpp/sglang/other):")
            yield Input(placeholder="vllm", id="input-engine")

            yield Button("Use selected", variant="primary", id="btn-registry")
            yield Button("Use external", variant="primary", id="btn-external")
            yield Button("Cancel", variant="default", id="btn-cancel")

    def on_mount(self) -> None:
        table = self.query_one("#registry-table", DataTable)
        table.add_columns("Slug", "Model", "Engine", "Port", "Status")
        for v in self.variants[:50]:  # Limit to 50 rows
            table.add_row(
                self._get_field(v, "slug"),
                str(self._get_field(v, "model"))[:20],
                self._get_field(v, "engine"),
                str(self._get_field(v, "port")),
                self._get_field(v, "status"),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-registry":
            table = self.query_one("#registry-table", DataTable)
            if table.cursor_row is not None and table.cursor_row < len(self.variants):
                v = self.variants[table.cursor_row]
                port = self._get_field(v, "port", 8020)
                target = ServingTarget(
                    url=f"http://localhost:{port}",
                    model=self._get_field(v, "model"),
                    slug=self._get_field(v, "slug"),
                    engine=self._get_field(v, "engine"),
                    kv_format=self._get_field(v, "kvcalc_key"),
                    status=self._get_field(v, "status"),
                    host_port=port,
                    container=self._get_field(v, "container"),
                    health="serving",
                )
                self.dismiss(target)

        elif event.button.id == "btn-external":
            url = self.query_one("#input-url", Input).value
            model = self.query_one("#input-model", Input).value
            engine = self.query_one("#input-engine", Input).value or "other"
            if not url or not model:
                self.notify("URL and model are required", severity="error")
                return
            target = ServingTarget(
                url=url,
                model=model,
                engine=engine,
                container="none",
                health="serving",
            )
            self.dismiss(target)

        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
