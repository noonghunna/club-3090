"""LivePane tail-buffer + placeholder contract (F4/F8).

Pure widget-object tests — no app mount needed: append_line buffers plain
text BEFORE touching the (unmounted) RichLog, so the [Y]-copy tail works
standalone.
"""

from __future__ import annotations

from club3090_tui_core.widgets.live_pane import LivePane


class TestLivePaneTailBuffer:
    def test_tail_collects_plain_text(self):
        lp = LivePane()
        lp.append_line("[green]✓[/green] step one")
        lp.append_line("run 3/5 narrative…")
        assert lp.tail_text() == "✓ step one\nrun 3/5 narrative…"
        # Markup is stripped — the tail is paste-ready plain text.
        assert "[green]" not in lp.tail_text()

    def test_unbuffered_lines_stay_out_of_tail(self):
        lp = LivePane()
        lp.append_line("[dim]▸ stopped · no live logs[/dim]", buffer=False)
        assert lp.tail_text() == ""
        lp.append_line("real output")
        assert lp.tail_text() == "real output"

    def test_clear_log_clears_tail(self):
        lp = LivePane()
        lp.append_line("old run line")
        lp.clear_log()
        assert lp.tail_text() == ""

    def test_tail_caps_line_count(self):
        lp = LivePane()
        for i in range(3000):
            lp.append_line(f"line {i}")
        # The buffer stays bounded and keeps the NEWEST lines.
        assert len(lp._raw_lines) <= lp._RAW_LINES_MAX
        assert lp.tail_text(lines=1) == "line 2999"

    def test_tail_limit_parameter(self):
        lp = LivePane()
        for i in range(10):
            lp.append_line(f"l{i}")
        assert lp.tail_text(lines=3) == "l7\nl8\nl9"

    def test_placeholder_is_constructor_owned(self):
        # F8 — hosts pass pane-specific idle copy; default is neutral (the old
        # hardcoded test-runner wording leaked into non-test-runner mounts).
        assert LivePane()._placeholder == "Ready."
        assert LivePane(placeholder="")._placeholder == ""
        assert LivePane(placeholder="logs stream here")._placeholder == "logs stream here"
