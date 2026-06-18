"""Tests for core SubprocessRunner and CoreRunState."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from club3090_tui_core.runner import SubprocessRunner, CoreRunState
from club3090_tui_core.parsers import ParseEvent


# ============================================================================
# CoreRunState tests
# ============================================================================

class TestCoreRunState:
    def test_elapsed_while_running(self):
        state = CoreRunState(run_type="bench", started=time.time() - 10)
        assert state.elapsed_s >= 9
        assert state.is_running is True
        assert state.is_finished is False

    def test_elapsed_after_finish(self):
        state = CoreRunState(run_type="bench", started=1000.0, finished=1060.0)
        assert state.elapsed_s == 60.0
        assert state.is_running is False
        assert state.is_finished is True

    def test_not_started(self):
        state = CoreRunState(run_type="bench")
        assert state.elapsed_s == 0
        assert state.is_running is False
        assert state.is_finished is False

    def test_initial_verdict_empty(self):
        state = CoreRunState(run_type="verify")
        assert state.verdict == ""
        assert state.events == []
        assert state.log_lines == []
        assert state.error == ""


# ============================================================================
# SubprocessRunner tests
# ============================================================================

class TestSubprocessRunner:
    def test_init(self):
        runner = SubprocessRunner(Path("/repo"))
        assert runner.repo_root == Path("/repo")
        assert runner.current_run is None
        assert runner._process is None
        assert runner.history == []

    def test_set_callbacks(self):
        runner = SubprocessRunner(Path("/repo"))
        on_event = MagicMock()
        on_line = MagicMock()
        on_complete = MagicMock()
        runner.set_callbacks(on_event=on_event, on_line=on_line, on_complete=on_complete)
        assert runner._on_event is on_event
        assert runner._on_line is on_line
        assert runner._on_complete is on_complete

    def test_set_callbacks_partial(self):
        runner = SubprocessRunner(Path("/repo"))
        on_event = MagicMock()
        runner.set_callbacks(on_event=on_event)
        assert runner._on_event is on_event
        assert runner._on_line is None
        assert runner._on_complete is None

    @pytest.mark.asyncio
    async def test_cancel_no_process(self):
        runner = SubprocessRunner(Path("/repo"))
        result = await runner.cancel()
        assert result == []

    @pytest.mark.asyncio
    async def test_start_raw_process_error(self):
        """When subprocess fails to start, state should reflect the error."""
        runner = SubprocessRunner(Path("/repo"))
        parser = MagicMock()
        completed = []

        def on_complete(state):
            completed.append(state)

        runner.set_callbacks(on_complete=on_complete)

        import unittest.mock as mock
        with mock.patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("binary not found"),
        ):
            state = await runner.start_raw(
                cmd=["nonexistent-binary"],
                env={},
                run_type="test",
                parser=parser,
            )

        assert state.exit_code == -1
        assert state.verdict == "failed"
        assert "binary not found" in state.error
        assert len(completed) == 1
