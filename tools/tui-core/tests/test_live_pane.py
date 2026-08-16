"""LivePane tail-buffer + placeholder contract (F4/F8).

Pure widget-object tests — no app mount needed: append_line buffers plain
text BEFORE touching the (unmounted) RichLog, so the [Y]-copy tail works
standalone.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Label, RichLog

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


class _Host(App):
    """Minimal host so the RichLog is mounted (scroll-follow needs layout)."""

    CSS = "LivePane { width: 1fr; height: 1fr; }"

    def compose(self) -> ComposeResult:
        yield LivePane(id="lp")


class TestLivePaneScrollFollow:
    async def test_scrolling_up_disables_auto_follow(self):
        async with _Host().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            lp = pilot.app.query_one("#lp", LivePane)
            log = pilot.app.query_one("#live-log", RichLog)
            for i in range(40):
                lp.append_line(f"line {i}")
                await pilot.pause()
            assert lp._follow is True
            log.scroll_up(immediate=True, animate=False)
            await pilot.pause()
            assert lp._follow is False
            assert log.auto_scroll is False  # write() must stop auto-scrolling too

    async def test_scrolling_to_bottom_re_enables(self):
        async with _Host().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            lp = pilot.app.query_one("#lp", LivePane)
            log = pilot.app.query_one("#live-log", RichLog)
            for i in range(40):
                lp.append_line(f"line {i}")
                await pilot.pause()
            log.scroll_up(immediate=True, animate=False)
            await pilot.pause()
            assert lp._follow is False
            log.scroll_end(immediate=True, animate=False)
            await pilot.pause()
            assert lp._follow is True
            assert log.auto_scroll is True

    async def test_append_while_scrolled_up_does_not_move_offset(self):
        async with _Host().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            lp = pilot.app.query_one("#lp", LivePane)
            log = pilot.app.query_one("#live-log", RichLog)
            for i in range(40):
                lp.append_line(f"line {i}")
                await pilot.pause()
            log.scroll_up(immediate=True, animate=False)
            await pilot.pause()
            before = log.scroll_y
            lp.append_line("line 40")
            await pilot.pause()
            assert log.scroll_y == before  # reading history is not yanked away

    async def test_set_title_updates_live_title_label(self):
        lp = LivePane()
        lp.set_title("whatever")  # unmounted — must not raise
        async with _Host().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            lp = pilot.app.query_one("#lp", LivePane)
            lp.set_title("Live  ●  [bold]following[/bold]")
            title = pilot.app.query_one(".live-title", Label)
            assert title.content == "Live  ●  [bold]following[/bold]"
