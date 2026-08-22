"""First-run guide (fresh-rig onboarding overlay) — headless tests.

Covers the four acceptance behaviors:
  1. Fresh-state detection triggers ONCE: a catalog whose weights join is
     all-absent (≥1 ABSENT, none present/partial/downloading) with no local
     measurement corpus pops the FirstRunScreen after the first catalog load;
     dismissing it never re-fires within the session.
  2. "Skip — I know what I'm doing" persists ``first_run_seen`` in
     c3-settings.json (merged — other keys preserved) and the next launch
     never shows the guide again.
  3. Each step dispatches an EXISTING action: pick keys move the selection,
     [d] routes through start_download → the setup.sh WEIGHT_KEY download
     runner, [l]/⏎ deep-links the SAME reconcile-gated ConfirmActionScreen ⏎
     on the catalog row opens.
  4. A seeded-weights rig (any slug present/partial) and an all-UNKNOWN
     weights join (index missing) do NOT trigger.

No subprocess is ever spawned: FakeRunner reads + fake write/download runners.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from textual.screen import ModalScreen

from club3090_cockpit.app import (
    CatalogPane,
    ConfirmActionScreen,
    FirstRunScreen,
)
from tests.test_app_headless import (
    FakeWriteRunner,
    _settle,
    fake_responses,
    make_app,
    ok,
)

# ── Canned weights.py list --json (the weights_index contract) ──────────────────

WEIGHTS_LIST_JSON = json.dumps(
    [
        {
            "model": "qwen3.6-27b",
            "variant": "autoround-int4",
            "subdir": "qwen3.6-27b-autoround-int4",
            "hf_repo": "acme/qwen36-27b-int4",
            "size_gb": 16.2,
            "verify_glob": "*.safetensors",
            "status": "production",
        }
    ]
)


def fresh_responses(**extra) -> dict:
    """Standard fakes + a weights index that JOINS qwen variants: with no model
    dir on disk every joined slug reads ABSENT → the fresh-rig signal."""
    return fake_responses(**{"weights.py list --json": ok(WEIGHTS_LIST_JSON), **extra})


def make_fresh_app(tmp_path, *, responses=None, write_runner=None):
    """A CockpitApp pointed at an EMPTY model dir under tmp_path — the fresh rig."""
    app, runner, wr = make_app(
        responses=responses if responses is not None else fresh_responses(),
        repo_root=tmp_path,
        write_runner=write_runner,
    )
    app._data._model_dir = str(tmp_path / "weights")   # nothing downloaded
    return app, runner, wr


class RecordingDLRunner:
    """Stand-in download runner: records start_raw (cmd/env), returns an
    already-done run state so the poll loop exits immediately. Never spawns."""

    def __init__(self):
        self.started: list[dict] = []
        self.callbacks: dict = {}

    def set_callbacks(self, on_event=None, on_line=None, on_complete=None):
        self.callbacks = {
            "on_event": on_event, "on_line": on_line, "on_complete": on_complete,
        }

    async def start_raw(self, cmd, env, run_type, parser):
        self.started.append({"cmd": list(cmd), "env": dict(env), "run_type": run_type})
        done = threading.Event()
        done.set()
        return SimpleNamespace(done=done)


def shown_first_run(app) -> bool:
    try:
        return isinstance(app.screen, FirstRunScreen)
    except Exception:
        return False


# ===========================================================================
# 1 · Fresh-state detection triggers once
# ===========================================================================


class TestFreshDetection:
    @pytest.mark.asyncio
    async def test_all_absent_triggers_once(self, tmp_path):
        app, _, _ = make_fresh_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert shown_first_run(app)
            screen = app.screen
            # The hw-aware picks are the profile-template reps; the dual
            # canonical slug is among them AND pre-selected as the fit default.
            slugs = [o.slug for o in screen._options]
            assert "vllm/dual" in slugs
            assert len(slugs) <= 3
            assert screen._selected_slug() == "vllm/dual"
            # Dismiss (Esc — session-only close) → back to the main screen.
            await pilot.press("escape")
            await pilot.pause()
            assert not shown_first_run(app)
            # ONCE per session: a re-check after dismissal never re-pushes.
            entries = app.query_one("#catalog-pane", CatalogPane)._entries
            app._maybe_first_run(entries)
            await pilot.pause()
            assert not shown_first_run(app)

    @pytest.mark.asyncio
    async def test_seeded_weights_do_not_trigger(self, tmp_path):
        app, _, _ = make_fresh_app(tmp_path)
        # Seed the recommended slug's weights on disk (subdir + verify_glob hit).
        d = tmp_path / "weights" / "qwen3.6-27b-autoround-int4"
        d.mkdir(parents=True)
        (d / "model-00001.safetensors").write_bytes(b"x")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert not shown_first_run(app)

    @pytest.mark.asyncio
    async def test_partial_weights_do_not_trigger(self, tmp_path):
        app, _, _ = make_fresh_app(tmp_path)
        # Subdir exists but no verify_glob match → partial (bytes on disk).
        d = tmp_path / "weights" / "qwen3.6-27b-autoround-int4"
        d.mkdir(parents=True)
        (d / "scratch.txt").write_bytes(b"x")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert not shown_first_run(app)

    @pytest.mark.asyncio
    async def test_local_measurements_do_not_trigger(self, tmp_path):
        app, _, _ = make_fresh_app(tmp_path)
        rec = tmp_path / "results" / "measurement-records"
        rec.mkdir(parents=True)
        (rec / "r.jsonl").write_text(
            json.dumps({"_tag": "vllm/dual",
                        "measured_extensions": {"quality_8pk": "109/150"}}),
            encoding="utf-8",
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert not shown_first_run(app)

    @pytest.mark.asyncio
    async def test_unknown_join_does_not_trigger(self, tmp_path):
        # No canned weights.py response → the index degrades to {} → every
        # entry UNKNOWN ("can't tell") → NO trigger (also keeps hermetic
        # suites without a weights mock trigger-free).
        app, _, _ = make_app(repo_root=tmp_path)
        app._data._model_dir = str(tmp_path / "weights")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert not shown_first_run(app)


# ===========================================================================
# 2 · Skip persists the seen-flag (c3-settings.json)
# ===========================================================================


class TestSkipPersists:
    @pytest.mark.asyncio
    async def test_skip_persists_and_never_nags_again(self, monkeypatch, tmp_path):
        cfg = tmp_path / "cfg"
        monkeypatch.setenv("C3_CONFIG_DIR", str(cfg))
        from club3090_cockpit.__main__ import load_settings

        app, _, _ = make_fresh_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert shown_first_run(app)
            await pilot.press("s")          # Skip — I know what I'm doing
            await pilot.pause()
            assert not shown_first_run(app)
            assert load_settings().get("first_run_seen") is True

        # A SECOND launch honors the persisted flag — no guide, ever again.
        app2, _, _ = make_fresh_app(tmp_path)
        async with app2.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert not shown_first_run(app2)

    @pytest.mark.asyncio
    async def test_skip_merges_other_persisted_keys(self, monkeypatch, tmp_path):
        cfg = tmp_path / "cfg"
        monkeypatch.setenv("C3_CONFIG_DIR", str(cfg))
        from club3090_cockpit.__main__ import save_settings, load_settings

        save_settings({"model_dir": "/data/models", "hf_token": "tok"})
        app, _, _ = make_fresh_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("s")
            await pilot.pause()
        s = load_settings()
        assert s["first_run_seen"] is True
        assert s["model_dir"] == "/data/models"       # merge — nothing clobbered
        assert s["hf_token"] == "tok"

    @pytest.mark.asyncio
    async def test_escape_dismisses_without_persisting(self, monkeypatch, tmp_path):
        cfg = tmp_path / "cfg"
        monkeypatch.setenv("C3_CONFIG_DIR", str(cfg))
        from club3090_cockpit.__main__ import load_settings

        app, _, _ = make_fresh_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("escape")
            await pilot.pause()
            assert not shown_first_run(app)
        assert "first_run_seen" not in load_settings()   # nag allowed next launch


# ===========================================================================
# 3 · Steps dispatch EXISTING actions
# ===========================================================================


class TestStepsDispatchExistingActions:
    @pytest.mark.asyncio
    async def test_pick_keys_move_selection(self, tmp_path):
        app, _, _ = make_fresh_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            screen = app.screen
            assert isinstance(screen, FirstRunScreen)
            n = len(screen._options)
            if n > 1:
                await pilot.press("2")
                await pilot.pause()
                assert screen._sel == 1
                assert screen._selected_slug() == screen._options[1].slug
                await pilot.press("1")
                await pilot.pause()
                assert screen._selected_slug() == screen._options[0].slug
            else:
                await pilot.press("2")   # out-of-range pick is a safe no-op
                await pilot.pause()
                assert screen._sel == 0

    @pytest.mark.asyncio
    async def test_download_dispatches_start_download_weight_key(self, tmp_path):
        dl = RecordingDLRunner()
        app, _, _ = make_fresh_app(tmp_path)
        app._data._download_runner = dl
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert shown_first_run(app)
            await pilot.press("d")           # ② Download for the selected fit
            await pilot.pause()
            # Guide closed; the EXISTING download worker took over.
            assert not shown_first_run(app)
            await _settle(pilot)
        assert dl.started, "start_download must spawn via the download runner"
        call = dl.started[0]
        assert call["cmd"][:2] == ["bash", "scripts/setup.sh"]
        assert call["env"]["WEIGHT_KEY"] == "qwen3.6-27b:autoround-int4"

    @pytest.mark.asyncio
    async def test_launch_deep_links_serve_confirm(self, tmp_path):
        app, _, wr = make_fresh_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert shown_first_run(app)
            await pilot.press("l")           # ③ Launch the selected fit
            await pilot.pause()
            # The SAME reconcile-gated confirm ⏎ on the catalog row opens…
            assert isinstance(app.screen, ConfirmActionScreen)
            plan = app.screen._plan
            assert plan.cmd[:2] == ["bash", "scripts/switch.sh"]
            assert "vllm/dual" in plan.cmd
            # …and NOTHING executed: still behind the gate (no live write).
            assert wr.started == []
            # The staged entry is recorded exactly like a manual ⏎ stage.
            assert app._staged_entry.slug == "vllm/dual"

    @pytest.mark.asyncio
    async def test_enter_key_also_deep_links_serve_confirm(self, tmp_path):
        app, _, _ = make_fresh_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert shown_first_run(app)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)


# ===========================================================================
# 4 · Help pointer
# ===========================================================================


class TestHelpPointer:
    @pytest.mark.asyncio
    async def test_question_mark_opens_help_over_guide(self, tmp_path):
        from club3090_cockpit.app import HelpScreen

        app, _, _ = make_fresh_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert isinstance(app.screen, FirstRunScreen)
            await pilot.press("question_mark")
            await pilot.pause()
            # The standard help overlay opens OVER the (still-mounted) guide.
            assert isinstance(app.screen, HelpScreen)
