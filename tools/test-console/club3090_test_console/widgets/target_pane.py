"""Target status pane widget — shows detected model, endpoint, GPU stats."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static

from ..detect import ServingTarget

STATUS_GLYPHS = {
    "serving": "● serving",
    "unreachable": "○ unreachable",
    "multiple": "⚠ multiple containers",
    "unknown": "? unknown",
}

STATUS_COLORS = {
    "serving": "green",
    "unreachable": "red",
    "multiple": "yellow",
    "unknown": "dim",
}

REGISTRY_STATUS_GLYPHS = {
    "production": "✅",
    "caveats": "⚠️",
    "experimental": "🧪",
    "incubating": "🐣",
    "preview": "👁️",
    "upstream-gated": "⏸️",
    "deprecated": "🗑️",
}


class TargetPane(Static):
    """Left-top pane showing serving target info."""

    target: reactive[ServingTarget | None] = reactive(None)

    DEFAULT_CSS = """
    TargetPane {
        width: 100%;
        height: auto;
        min-height: 8;
        border: solid $primary;
        padding: 0 1;
    }
    TargetPane .pane-title {
        text-style: bold;
        color: $accent;
        dock: top;
    }
    TargetPane .target-row {
        height: 1;
    }
    TargetPane .target-label {
        color: $text-muted;
    }
    TargetPane .no-target {
        color: $warning;
        text-style: italic;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Target", classes="pane-title")
        yield Label("Detecting...", classes="target-row", id="target-model")
        yield Label("", classes="target-row", id="target-slug")
        yield Label("", classes="target-row", id="target-engine")
        yield Label("", classes="target-row", id="target-url")
        yield Label("", classes="target-row", id="target-gpu0")
        yield Label("", classes="target-row", id="target-gpu1")

    def watch_target(self, target: ServingTarget | None) -> None:
        """Update display when target changes."""
        if target is None:
            self._show_no_target()
            return

        model_widget = self.query_one("#target-model", Label)
        slug_widget = self.query_one("#target-slug", Label)
        engine_widget = self.query_one("#target-engine", Label)
        url_widget = self.query_one("#target-url", Label)
        gpu0_widget = self.query_one("#target-gpu0", Label)
        gpu1_widget = self.query_one("#target-gpu1", Label)

        if not target.model:
            self._show_no_target()
            return

        # Model line
        status_glyph = STATUS_GLYPHS.get(target.health, STATUS_GLYPHS["unknown"])
        model_widget.update(f"Model  {target.model}")

        # Slug + registry status
        if target.slug:
            reg_glyph = REGISTRY_STATUS_GLYPHS.get(target.status, "")
            slug_widget.update(f"Slug   {target.slug}  {reg_glyph} {target.status}")
        else:
            slug_widget.update("")

        # Engine line
        tp_str = f"TP {target.tp}" if target.tp else ""
        kv_str = f"KV {target.kv_format}" if target.kv_format else ""
        engine_widget.update(f"Engine {target.engine}  {tp_str}  {kv_str}".strip())

        # URL + health
        color = STATUS_COLORS.get(target.health, "dim")
        url_widget.update(f"URL    :{target.host_port}  [{color}]{status_glyph}[/{color}]")

        # GPU lines
        if target.gpus:
            for i, gpu in enumerate(target.gpus[:2]):
                widget = gpu0_widget if i == 0 else gpu1_widget
                widget.update(
                    f"GPU{gpu.index} {gpu.mem_used_mib}/{gpu.mem_total_mib}M "
                    f"{gpu.utilization}% {gpu.power_draw_w:.0f}W/{gpu.power_limit_w:.0f}W "
                    f"{gpu.temp_c}°C"
                )
        else:
            gpu0_widget.update("")
            gpu1_widget.update("")

    def _show_no_target(self) -> None:
        model_widget = self.query_one("#target-model", Label)
        model_widget.update("[yellow]No model serving — run gpu-mode <mode>[/yellow]")
        for widget_id in ("#target-slug", "#target-engine", "#target-url", "#target-gpu0", "#target-gpu1"):
            self.query_one(widget_id, Label).update("")
