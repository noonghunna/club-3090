"""Generic subprocess streaming runner — shared core for both TUI apps."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .parsers import ParseEvent


@dataclass
class CoreRunState:
    """Generic run state not tied to test types."""
    run_type: str
    started: float = 0.0
    finished: float = 0.0
    exit_code: Optional[int] = None
    verdict: str = ""
    events: list[ParseEvent] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    error: str = ""
    artifact_dir: str = ""
    report_path: str = ""

    @property
    def elapsed_s(self) -> float:
        end = self.finished or time.time()
        return end - self.started if self.started else 0

    @property
    def is_running(self) -> bool:
        return self.started > 0 and self.finished == 0

    @property
    def is_finished(self) -> bool:
        return self.finished > 0


class SubprocessRunner:
    """Generic streaming subprocess runner — core engine for both TUI apps."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.current_run: Optional[CoreRunState] = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._cancel_event = asyncio.Event()
        self._on_event: Optional[Callable[[ParseEvent], None]] = None
        self._on_line: Optional[Callable[[str], None]] = None
        self._on_complete: Optional[Callable[[CoreRunState], None]] = None
        self.history: list[CoreRunState] = []

    def set_callbacks(
        self,
        on_event: Optional[Callable[[ParseEvent], None]] = None,
        on_line: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[CoreRunState], None]] = None,
    ):
        self._on_event = on_event
        self._on_line = on_line
        self._on_complete = on_complete

    async def start_raw(self, cmd: list[str], env: dict, run_type: str, parser) -> CoreRunState:
        """Start a subprocess with pre-built command. Returns CoreRunState."""
        state = CoreRunState(run_type=run_type, started=time.time())
        self.current_run = state
        self._cancel_event.clear()

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as e:
            state.error = str(e)
            state.finished = time.time()
            state.exit_code = -1
            state.verdict = "failed"
            if self._on_complete:
                self._on_complete(state)
            return state

        asyncio.create_task(self._read_output(state, parser))
        return state

    async def _read_output(self, state: CoreRunState, parser):
        proc = self._process
        try:
            while True:
                if self._cancel_event.is_set():
                    break
                try:
                    line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                state.log_lines.append(line)
                if self._on_line:
                    self._on_line(line)
                event = parser.parse_line(line)
                if event:
                    state.events.append(event)
                    if event.event_type == "rebench_report":
                        state.report_path = event.data.get("path", "")
                    elif event.event_type == "rebench_artifacts":
                        state.artifact_dir = event.data.get("dir", "")
                    elif event.event_type == "verdict":
                        state.verdict = "passed" if event.data.get("status") == "passed" else "failed"
                    if self._on_event:
                        self._on_event(event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            state.error = str(e)

        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

        state.exit_code = proc.returncode
        state.finished = time.time()
        if not state.verdict:
            state.verdict = "passed" if state.exit_code == 0 else "failed"

        self.history.append(state)
        self.current_run = None
        self._process = None

        if self._on_complete:
            self._on_complete(state)

    async def cancel(self) -> list[str]:
        """Cancel current run. Returns empty list (callers add orphan checks)."""
        if not self._process:
            return []
        self._cancel_event.set()
        proc = self._process
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                await proc.wait()
        return []
