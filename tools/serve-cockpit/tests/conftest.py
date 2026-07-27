"""Test-suite safety net: no live subprocess, ever.

The cockpit is fully dependency-injected — tests construct ``CockpitData`` with
a ``FakeRunner`` (reads) and a ``FakeWriteRunner`` (writes), so no real process
should ever be spawned.  This autouse fixture turns that convention into an
enforced invariant: it monkeypatches the three real spawn points so that any
accidental escape (a forgotten seam, a regression that bypasses the fake)
raises loudly instead of shelling out / serving / switching on the host.

Specifically blocked for the duration of every test:
  - ``asyncio.create_subprocess_exec``       (RealRunner + core SubprocessRunner)
  - ``services.RealRunner.run``              (the live READ runner)
  - ``SubprocessRunner.start_raw``           (the live WRITE streamer)

A test that needs to assert "a write was attempted" must inject a
``FakeWriteRunner`` — which records ``start_raw`` without spawning — NOT call the
real runner.  This fixture guarantees the real one never runs.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

# Download plumbing must stay hermetic under test: DownloadLog must never touch
# the user's real ~/.config/club-3090, and download_preflight must never depend
# on the HOST's `hf` CLI / token state (a dev box with hf installed would pass
# where CI fails, and vice versa).  Explicit preflight/log tests override these
# per-test via monkeypatch.
os.environ.setdefault("C3_SKIP_DOWNLOAD_PREFLIGHT", "1")
os.environ.setdefault("C3_CONFIG_DIR", tempfile.mkdtemp(prefix="c3-tests-config-"))

from club3090_tui_core.runner import SubprocessRunner
from club3090_cockpit.services import RealRunner


@pytest.fixture(autouse=True)
def _no_live_subprocess(monkeypatch):
    """Hard-block every real subprocess spawn point during tests."""

    async def _blocked_exec(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError(
            "LIVE SUBPROCESS BLOCKED: a test tried to spawn a real process "
            f"({args[:2]}).  Inject a FakeRunner / FakeWriteRunner instead."
        )

    async def _blocked_real_run(self, cmd, *, cwd, timeout=30.0):  # pragma: no cover
        raise AssertionError(
            f"LIVE READ BLOCKED: RealRunner.run was invoked ({cmd[:2]}). "
            "Tests must inject a FakeRunner."
        )

    async def _blocked_start_raw(self, cmd, env, run_type, parser):  # pragma: no cover
        raise AssertionError(
            f"LIVE WRITE BLOCKED: SubprocessRunner.start_raw was invoked ({cmd[:2]}). "
            "Tests must inject a FakeWriteRunner — writes are never executed live."
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _blocked_exec)
    monkeypatch.setattr(RealRunner, "run", _blocked_real_run)
    monkeypatch.setattr(SubprocessRunner, "start_raw", _blocked_start_raw)
    yield
