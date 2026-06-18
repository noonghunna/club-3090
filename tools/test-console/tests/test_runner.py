"""Tests for the runner module — subprocess management and env injection."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from club3090_test_console.runner import TestRunner, TestConfig, RunState
from club3090_test_console.detect import ServingTarget, GpuInfo
from club3090_test_console.parsers import TestType


# ============================================================================
# Test command building
# ============================================================================

class TestCommandBuilding:
    """Test that _build_command produces correct commands and env."""

    def _make_runner_with_target(self, target: ServingTarget) -> TestRunner:
        runner = TestRunner(repo_root=Path("/repo"))
        state = RunState(
            test_type=TestType.BENCH,
            config=TestConfig(test_type=TestType.BENCH),
            target=target,
        )
        runner.current_run = state
        return runner

    def test_bench_command(self):
        target = ServingTarget(url="http://localhost:8010", model="qwen3.6-27b", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(test_type=TestType.BENCH, run_count=3, warmups=2, only="narr")
        runner.current_run.config = config
        cmd, env = runner._build_command(config)

        assert "scripts/bench.sh" in cmd
        assert env["URL"] == "http://localhost:8010"
        assert env["MODEL"] == "qwen3.6-27b"
        assert env["CONTAINER"] == "vllm-test"
        assert env["RUNS"] == "3"
        assert env["WARMUPS"] == "2"
        assert env["ONLY"] == "narr"
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_bench_with_thinking(self):
        target = ServingTarget(url="http://localhost:8010", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(test_type=TestType.BENCH, enable_thinking=True)
        cmd, env = runner._build_command(config)
        assert env["ENABLE_THINKING"] == "1"

    def test_bench_with_force_tokens(self):
        target = ServingTarget(url="http://localhost:8010", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(test_type=TestType.BENCH, force_tokens=2000)
        cmd, env = runner._build_command(config)
        assert env["FORCE_TOKENS"] == "2000"

    def test_verify_full_command(self):
        target = ServingTarget(url="http://localhost:8010", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(test_type=TestType.VERIFY_FULL, skip_tools=True, run_bench=True)
        cmd, env = runner._build_command(config)

        assert "scripts/verify-full.sh" in cmd
        assert "--bench" in cmd
        assert env["SKIP_TOOLS"] == "1"

    def test_verify_stress_command(self):
        target = ServingTarget(url="http://localhost:8010", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(
            test_type=TestType.VERIFY_STRESS,
            skip_longctx=True,
            skip_tool_prefill=True,
        )
        cmd, env = runner._build_command(config)

        assert "scripts/verify-stress.sh" in cmd
        assert env["SKIP_LONGCTX"] == "1"
        assert env["SKIP_TOOL_PREFILL"] == "1"

    def test_quality_command(self):
        target = ServingTarget(url="http://localhost:8010", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(
            test_type=TestType.QUALITY,
            quality_tier="full",
            quality_pack="toolcall-15",
            quality_repeat=3,
        )
        cmd, env = runner._build_command(config)

        assert "scripts/quality-test.sh" in cmd
        assert "--full" in cmd
        assert "--pack" in cmd
        assert "toolcall-15" in cmd
        assert "--repeat" in cmd
        assert "3" in cmd
        assert env["BENCHLOCAL_HERMES_RESOLVE_LOCALHOST"] == "1"  # localhost

    def test_quality_non_localhost_no_hermes(self):
        target = ServingTarget(url="http://192.168.1.50:8887", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(test_type=TestType.QUALITY, quality_tier="medium")
        cmd, env = runner._build_command(config)

        assert "BENCHLOCAL_HERMES_RESOLVE_LOCALHOST" not in env

    def test_soak_command(self):
        target = ServingTarget(url="http://localhost:8010", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(
            test_type=TestType.SOAK,
            soak_mode="fresh",
            soak_sessions=20,
            soak_turns=10,
            soak_max_growth=300,
        )
        cmd, env = runner._build_command(config)

        assert "scripts/soak-test.sh" in cmd
        assert "--fresh" in cmd
        assert env["SOAK_SESSIONS"] == "20"
        assert env["SOAK_TURNS"] == "10"
        assert env["SOAK_MAX_GROWTH_MIB"] == "300"

    def test_rebench_command(self):
        target = ServingTarget(url="http://localhost:8010", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(
            test_type=TestType.REBENCH,
            rebench_8pack="both",
            rebench_skip=["soak"],
            rebench_tag="test-run",
        )
        cmd, env = runner._build_command(config)

        assert "scripts/rebench-full.sh" in cmd
        assert "--with-8pack-thinking=both" in cmd
        assert "--skip=soak" in cmd
        assert "--tag=test-run" in cmd

    def test_rebench_external_endpoint(self):
        target = ServingTarget(url="http://localhost:8010", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        config = TestConfig(
            test_type=TestType.REBENCH,
            external_url="http://192.168.1.50:8887",
            external_model="Qwen3.6-27B",
            external_engine="llama-cpp",
        )
        cmd, env = runner._build_command(config)

        assert "--url" in cmd
        assert "http://192.168.1.50:8887" in cmd
        assert "--model" in cmd
        assert "Qwen3.6-27B" in cmd
        assert "--engine" in cmd
        assert "llama-cpp" in cmd
        assert env["PREFLIGHT_NO_AUTODETECT"] == "1"
        assert env["CONTAINER"] == "none"

    def test_stdbuf_wrapping(self):
        """All commands should be wrapped in stdbuf for line-buffered output."""
        target = ServingTarget(url="http://localhost:8010", model="qwen", container="vllm-test")
        runner = self._make_runner_with_target(target)
        for tt in TestType:
            config = TestConfig(test_type=tt)
            cmd, _ = runner._build_command(config)
            assert cmd[0] == "stdbuf", f"{tt} command not wrapped in stdbuf"


# ============================================================================
# Test RunState
# ============================================================================

class TestRunState:
    def test_elapsed_while_running(self):
        import time
        state = RunState(
            test_type=TestType.BENCH,
            config=TestConfig(test_type=TestType.BENCH),
            target=ServingTarget(),
            started=time.time() - 10,
        )
        assert state.elapsed_s >= 9
        assert state.is_running is True
        assert state.is_finished is False

    def test_elapsed_after_finish(self):
        state = RunState(
            test_type=TestType.BENCH,
            config=TestConfig(test_type=TestType.BENCH),
            target=ServingTarget(),
            started=1000.0,
            finished=1060.0,
        )
        assert state.elapsed_s == 60.0
        assert state.is_running is False
        assert state.is_finished is True

    def test_not_started(self):
        state = RunState(
            test_type=TestType.BENCH,
            config=TestConfig(test_type=TestType.BENCH),
            target=ServingTarget(),
        )
        assert state.elapsed_s == 0
        assert state.is_running is False
