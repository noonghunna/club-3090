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
import sys
import tempfile
from pathlib import Path

import pytest

# Download plumbing must stay hermetic under test: DownloadLog must never touch
# the user's real ~/.config/club-3090, and download_preflight must never depend
# on the HOST's `hf` CLI / token state (a dev box with hf installed would pass
# where CI fails, and vice versa).  Explicit preflight/log tests override these
# per-test via monkeypatch.
os.environ.setdefault("C3_SKIP_DOWNLOAD_PREFLIGHT", "1")
os.environ.setdefault("C3_CONFIG_DIR", tempfile.mkdtemp(prefix="c3-tests-config-"))
# The club-3090 settings store (club-3090#1466) — forced, not defaulted: a
# developer's exported CLUB3090_CONFIG_DIR must not leak into a test either.
# The per-test fixture below replaces it with a fresh directory.
os.environ["CLUB3090_CONFIG_DIR"] = tempfile.mkdtemp(prefix="c3-tests-club3090-config-")

# The checkout these tests live in. Its repo-root .env may hold the maintainer's
# real settings (the loader reads it last, as a legacy fallback).
_REAL_REPO_ROOT = Path(__file__).resolve().parents[3]


def pytest_configure(config):
    """Register the opt-in live-network marker (test_uiux_hf_search.py)."""
    config.addinivalue_line(
        "markers",
        "live_network: touches the live Hugging Face API — skipped unless "
        "CLUB3090_LIVE_NETWORK=1",
    )

from club3090_tui_core.runner import SubprocessRunner
from club3090_cockpit.services import RealRunner


# ── temp dirs never outlive their test on sys.path ────────────────────────────
# c3 puts a CockpitData's repo root on sys.path so `scripts.lib.*` imports work
# (services.py), and tests build apps on temporary repo roots. A temp root left
# there shadows the real tree for every later test in the process:
# TestScriptsImportable once left a seeded scripts/lib/profiles package on it, and
# 11 tests failed whenever test_services.py ran on its own (the full suite hid it by
# importing the real package first). After each test's fixtures are torn down —
# monkeypatch included, whatever the fixture order — drop any temp dir it added.
# The real repo root is left alone: later imports rely on it.
_TEMP_BASES = tuple({os.path.realpath(tempfile.gettempdir()), tempfile.gettempdir()})
_SYS_PATH_BEFORE: dict[str, list[str]] = {}


def _is_temp(entry: str) -> bool:
    return any(entry == b or entry.startswith(b + os.sep) for b in _TEMP_BASES)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _SYS_PATH_BEFORE[item.nodeid] = list(sys.path)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    before = set(_SYS_PATH_BEFORE.pop(item.nodeid, sys.path))
    sys.path[:] = [p for p in sys.path if p in before or not _is_temp(p)]


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


def _club_config():
    """The ONE settings loader (scripts/lib/club_config.py) of this checkout —
    the module object c3's settings_store uses, so patches here reach c3."""
    root = str(_REAL_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts.lib import club_config

    return club_config


def _is_real_repo_root(repo_root) -> bool:
    try:
        return Path(repo_root).resolve() == _REAL_REPO_ROOT
    except (OSError, TypeError, ValueError):
        return False


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path_factory, monkeypatch):
    """Tests never read or write the real club-3090 settings (club-3090#1466).

    c3 reads and writes settings through the one loader: the per-user store
    (``CLUB3090_CONFIG_DIR``: club3090.env + secrets.env) and, read last, the
    checkout's repo-root .env.  For every test:
      * ``CLUB3090_CONFIG_DIR`` and ``C3_CONFIG_DIR`` (c3-settings.json, logs)
        point at a fresh temporary directory;
      * the legacy .env of the checkout the tests run from is invisible (tests
        that use the real tree as repo root, to read the registry, must not
        pick up the maintainer's settings), and clearing a key there fails the
        test instead of rewriting it;
      * ``HF_TOKEN`` is restored afterwards (c3 applies a saved token to its own
        environment, as it does at launch).
    """
    d = tmp_path_factory.mktemp("club3090-config")
    monkeypatch.setenv("CLUB3090_CONFIG_DIR", str(d))
    monkeypatch.setenv("C3_CONFIG_DIR", str(d))

    cc = _club_config()
    real_layers = cc.layers
    real_unset = cc.unset_values

    def layers(repo_root=None, environ=None):
        out = real_layers(repo_root, environ)
        if repo_root is not None and _is_real_repo_root(repo_root):
            out = [layer for layer in out if layer[0] != cc.LEGACY_LABEL]
        return out

    def unset_values(keys, which="global", environ=None, repo_root=None):
        if repo_root is not None and _is_real_repo_root(repo_root):
            raise AssertionError("a test tried to rewrite the real checkout's .env")
        return real_unset(keys, which, environ, repo_root=repo_root)

    monkeypatch.setattr(cc, "layers", layers)
    monkeypatch.setattr(cc, "unset_values", unset_values)
    token = os.environ.get("HF_TOKEN")
    yield
    if token is None:
        os.environ.pop("HF_TOKEN", None)
    else:
        os.environ["HF_TOKEN"] = token
