"""UI/UX Phase 3 — download preflight, jobs chip, route language, new bring."""

from __future__ import annotations

import pytest
from textual.widgets import Static, TabbedContent

from club3090_cockpit.app import LaneBringPane, LaneServePane, _byo_result_text
from club3090_cockpit.data import ByoResult
from tests.test_app_headless import _settle, make_app
from tests.test_uiux_phase1 import _route_c_byo


class TestPhase3RouteLanguage:
    def test_outcome_first_route_headings(self):
        a = _byo_result_text(ByoResult(
            repo="org/X", profile_like="vllm/dual", eligible=True,
            fit_verdict="fits-clean", route="A",
        ))
        assert "Needs a catalog profile" in a
        assert "route A" in a
        b = _byo_result_text(ByoResult(
            repo="org/X", profile_like="vllm/dual", eligible=True,
            fit_verdict="fits-constrained", route="B",
        ))
        assert "Local-only serve" in b
        c = _byo_result_text(_route_c_byo())
        assert "Can serve now" in c or "Servable" in c


class TestPhase3NewBring:
    @pytest.mark.asyncio
    async def test_new_bring_clears_arm(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app._last_byo = _route_c_byo()
            app._arm_serve_pane(app._last_byo)
            await pilot.pause()
            body = str(
                app.query_one("#lane-serve-pane", LaneServePane)
                .query_one("#lane-serve-body", Static)
                .render()
            )
            assert "Serving" in body
            app.action_new_bring()
            await pilot.pause()
            assert app._last_byo is None
            body2 = str(
                app.query_one("#lane-serve-pane", LaneServePane)
                .query_one("#lane-serve-body", Static)
                .render()
            )
            assert "Run ① Bring first" in body2 or "arm from ①" in body2


class TestPhase3JobsChip:
    @pytest.mark.asyncio
    async def test_jobs_chip_on_active_download(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app._bring_downloads = {
                "org/BigModel": {"handle": None, "profile_like": "vllm/dual"},
            }
            app._sync_jobs_chip()
            assert "downloading" in (app.sub_title or "").lower()
            app._bring_downloads.clear()
            app._sync_jobs_chip()
            assert "downloading" not in (app.sub_title or "").lower()


class TestPhase3DownloadPreflightSize:
    def test_bring_expected_size_from_inventory(self):
        from club3090_cockpit.data import ArtifactInventory, GgufVariant

        app, _, _ = make_app(surface="producer")
        inv = ArtifactInventory(
            repo="org/X",
            formats=["safetensors"],
            safetensors_size_gb=42.0,
        )
        app._last_inventory = inv
        assert app._bring_expected_size_gb() == 42.0
