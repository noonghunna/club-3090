"""Test runner — c3t's test-specific wrapper over the shared streaming core.

The generic subprocess-streaming engine (spawn + stdbuf + env-inject +
set_callbacks + parse loop + cancel) now lives in
``club3090_tui_core.runner.SubprocessRunner``.  This module keeps only the
test-specific pieces: ``TestType``/``TestConfig``/``_build_command`` and the
``RunState`` shape c3t's app + history records expect (which carries the
``test_type``/``config``/``target`` triple the core's generic ``CoreRunState``
deliberately does not).
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .detect import ServingTarget
from .parsers import ParseEvent, TestType, get_parser
from club3090_tui_core.runner import SubprocessRunner, CoreRunState


@dataclass
class TestConfig:
    """Configuration for a test run."""
    test_type: TestType
    # Test-specific tunables
    run_count: int = 5               # bench RUNS
    warmups: int = 3                 # bench WARMUPS
    only: str = "both"               # bench ONLY (both/narr/code)
    enable_thinking: bool = False    # bench/quality ENABLE_THINKING
    pp: bool = False                 # bench PP
    force_tokens: int = 0            # bench FORCE_TOKENS
    skip_tools: bool = False         # verify SKIP_TOOLS
    run_bench: bool = False          # verify-full --bench
    skip_longctx: bool = False       # verify-stress SKIP_LONGCTX
    skip_tool_prefill: bool = False  # verify-stress SKIP_TOOL_PREFILL
    skip_ceiling: bool = False       # verify-stress SKIP_CEILING
    quality_tier: str = "medium"     # quality --quick/--medium/--full/--reasoning
    quality_pack: str = ""           # quality --pack <id>
    quality_no_sandboxed: bool = False
    quality_sandboxed_only: bool = False
    quality_sampling_server: bool = False
    quality_repeat: int = 1
    max_tokens: int = 0              # quality MAX_TOKENS (0 = default)
    thinking_max_tokens: int = 0     # quality THINKING_MAX_TOKENS (0 = default)
    soak_mode: str = "fresh"         # soak --fresh/--continuous/--quick
    soak_sessions: int = 10
    soak_turns: int = 5
    soak_max_growth: int = 200
    soak_timeout: int = 1800
    rebench_8pack: str = ""          # "" or "off" or "on" or "both"
    rebench_skip: list[str] = field(default_factory=list)
    rebench_resume: bool = False
    rebench_tag: str = ""
    # External endpoint
    external_url: str = ""
    external_model: str = ""
    external_engine: str = ""


@dataclass
class RunState:
    """State of an active test run.

    Carries the test-specific ``test_type``/``config``/``target`` triple plus
    the generic run fields the core mirrors into it via callbacks.
    """
    test_type: TestType
    config: TestConfig
    target: ServingTarget
    started: float = 0.0
    finished: float = 0.0
    exit_code: Optional[int] = None
    verdict: str = ""              # passed/failed/unknown
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


class TestRunner:
    """Manages spawning + tracking of test subprocesses.

    A thin test-specific wrapper: it builds the command/env, delegates the
    actual spawn + stream + parse to a shared :class:`SubprocessRunner`, and
    mirrors the core's run-state back into c3t's richer :class:`RunState`.
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.current_run: Optional[RunState] = None
        self.history: list[RunState] = []
        self._on_event: Optional[Callable[[ParseEvent], None]] = None
        self._on_line: Optional[Callable[[str], None]] = None
        self._on_complete: Optional[Callable[[RunState], None]] = None
        # Shared streaming engine.
        self._core = SubprocessRunner(repo_root)

    def set_callbacks(
        self,
        on_event: Optional[Callable[[ParseEvent], None]] = None,
        on_line: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[RunState], None]] = None,
    ):
        """Set callback functions for events, lines, and completion."""
        self._on_event = on_event
        self._on_line = on_line
        self._on_complete = on_complete

    def _build_command(self, config: TestConfig) -> tuple[list[str], dict[str, str]]:
        """Build the command and environment for a test run."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        args: list[str] = []

        # Inject target
        target = self.current_run.target if self.current_run else ServingTarget()
        if target.url:
            env["URL"] = target.url
        if target.model:
            env["MODEL"] = target.model
        if target.container:
            env["CONTAINER"] = target.container

        # BENCHLOCAL_HERMES_RESOLVE_LOCALHOST for localhost quality/rebench
        if target.is_localhost and config.test_type in (TestType.QUALITY, TestType.REBENCH):
            env["BENCHLOCAL_HERMES_RESOLVE_LOCALHOST"] = "1"

        # External endpoint mode (for rebench)
        if config.external_url:
            env["PREFLIGHT_NO_AUTODETECT"] = "1"
            env["CONTAINER"] = "none"

        match config.test_type:
            case TestType.VERIFY:
                args = ["bash", "scripts/verify.sh"]
                if config.skip_tools:
                    env["SKIP_TOOLS"] = "1"

            case TestType.VERIFY_FULL:
                args = ["bash", "scripts/verify-full.sh"]
                if config.skip_tools:
                    env["SKIP_TOOLS"] = "1"
                if config.run_bench:
                    args.append("--bench")

            case TestType.BENCH:
                args = ["bash", "scripts/bench.sh"]
                env["RUNS"] = str(config.run_count)
                env["WARMUPS"] = str(config.warmups)
                env["ONLY"] = config.only
                if config.enable_thinking:
                    env["ENABLE_THINKING"] = "1"
                if config.pp:
                    env["PP"] = "1"
                if config.force_tokens:
                    env["FORCE_TOKENS"] = str(config.force_tokens)

            case TestType.VERIFY_STRESS:
                args = ["bash", "scripts/verify-stress.sh"]
                if config.skip_longctx:
                    env["SKIP_LONGCTX"] = "1"
                if config.skip_tool_prefill:
                    env["SKIP_TOOL_PREFILL"] = "1"
                if config.skip_ceiling:
                    env["SKIP_CEILING"] = "1"

            case TestType.QUALITY:
                args = ["bash", "scripts/quality-test.sh"]
                # Tier flag
                args.append(f"--{config.quality_tier}")
                if config.quality_pack:
                    args.extend(["--pack", config.quality_pack])
                if config.quality_no_sandboxed:
                    args.append("--no-sandboxed")
                if config.quality_sandboxed_only:
                    args.append("--sandboxed-only")
                if config.enable_thinking:
                    args.append("--enable-thinking")
                else:
                    args.append("--no-thinking")
                if config.quality_sampling_server:
                    args.append("--sampling-from-server")
                if config.quality_repeat > 1:
                    args.extend(["--repeat", str(config.quality_repeat)])
                if config.max_tokens > 0:
                    env["MAX_TOKENS"] = str(config.max_tokens)
                if config.thinking_max_tokens > 0:
                    env["THINKING_MAX_TOKENS"] = str(config.thinking_max_tokens)

            case TestType.SOAK:
                args = ["bash", "scripts/soak-test.sh"]
                match config.soak_mode:
                    case "fresh":
                        args.append("--fresh")
                    case "continuous":
                        args.append("--continuous")
                    case "quick":
                        args.append("--quick")
                env["SOAK_SESSIONS"] = str(config.soak_sessions)
                env["SOAK_TURNS"] = str(config.soak_turns)
                env["SOAK_MAX_GROWTH_MIB"] = str(config.soak_max_growth)
                env["SOAK_TIMEOUT_S"] = str(config.soak_timeout)

            case TestType.REBENCH:
                args = ["bash", "scripts/rebench-full.sh"]
                if config.rebench_8pack:
                    args.append(f"--with-8pack-thinking={config.rebench_8pack}")
                if config.rebench_skip:
                    args.append(f"--skip={','.join(config.rebench_skip)}")
                if config.rebench_resume:
                    args.append("--resume")
                if config.rebench_tag:
                    args.append(f"--tag={config.rebench_tag}")
                if config.external_url:
                    args.extend([
                        "--url", config.external_url,
                        "--model", config.external_model,
                        "--engine", config.external_engine or "other",
                    ])
                env["SOAK_SESSIONS"] = str(config.soak_sessions)
                env["SOAK_TURNS"] = str(config.soak_turns)
                if config.max_tokens > 0:
                    env["MAX_TOKENS"] = str(config.max_tokens)
                if config.thinking_max_tokens > 0:
                    env["THINKING_MAX_TOKENS"] = str(config.thinking_max_tokens)

        # Use stdbuf for line-buffered output
        full_cmd = ["stdbuf", "-oL", "-eL"] + args
        return full_cmd, env

    def _mirror(self, state: RunState, core_state: CoreRunState) -> None:
        """Copy the generic fields the core owns back into c3t's RunState."""
        state.started = core_state.started
        state.finished = core_state.finished
        state.exit_code = core_state.exit_code
        state.verdict = core_state.verdict
        state.events = core_state.events
        state.log_lines = core_state.log_lines
        state.error = core_state.error
        state.artifact_dir = core_state.artifact_dir
        state.report_path = core_state.report_path

    async def start(self, config: TestConfig, target: ServingTarget) -> RunState:
        """Start a test run via the shared streaming engine."""
        state = RunState(
            test_type=config.test_type,
            config=config,
            target=target,
            started=time.time(),
        )
        self.current_run = state

        cmd, env = self._build_command(config)
        parser = get_parser(config.test_type)

        # Wire the core's callbacks to c3t's, mirroring core state into our
        # RunState as it progresses so the app sees a single object.  The live
        # core state is reachable via self._core.current_run (set synchronously
        # inside start_raw, before the reader task runs).
        def _on_event(event: ParseEvent) -> None:
            if self._core.current_run is not None:
                self._mirror(state, self._core.current_run)
            if self._on_event:
                self._on_event(event)

        def _on_line(line: str) -> None:
            if self._on_line:
                self._on_line(line)

        def _on_complete(core_state: CoreRunState) -> None:
            self._mirror(state, core_state)
            self.history.append(state)
            self.current_run = None
            if self._on_complete:
                self._on_complete(state)

        self._core.set_callbacks(
            on_event=_on_event, on_line=_on_line, on_complete=_on_complete
        )

        core_state = await self._core.start_raw(
            cmd, env, config.test_type.value, parser
        )
        # Mirror the initial spawn result (covers the immediate-failure path,
        # where start_raw fires on_complete before returning).
        self._mirror(state, core_state)
        return state

    async def cancel(self) -> list[str]:
        """Cancel the current run. Returns orphaned benchlocal containers if any."""
        was_quality = (
            self.current_run is not None
            and self.current_run.test_type == TestType.QUALITY
        )

        await self._core.cancel()

        orphans: list[str] = []
        if was_quality:
            orphans = await self._check_benchlocal_orphans()
        return orphans

    async def _check_benchlocal_orphans(self) -> list[str]:
        """Check for orphaned benchlocal containers that may be squatting ports."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "ps", "--format", "{{.Names}}",
                "--filter", "name=benchlocal-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            containers = [name.strip() for name in stdout.decode().split("\n") if name.strip()]
            return containers
        except Exception:
            return []
