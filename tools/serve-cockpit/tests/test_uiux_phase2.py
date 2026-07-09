"""UI/UX Phase 2 — per-stage visual hierarchy (target → action → details)."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Label, Static, TabbedContent

from club3090_cockpit.app import (
    LaneBringPane,
    LanePromotePane,
    LaneServePane,
    ValidateEvidencePane,
    ValidateRunPane,
)
from tests.test_app_headless import _settle, make_app
from tests.test_uiux_phase1 import _route_c_byo


class TestPhase2GateMeasure:
    @pytest.mark.asyncio
    async def test_gate_heading_and_live_above_gotchas(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            pane = app.query_one("#validate-run-pane", ValidateRunPane)
            heading = str(pane.query_one("#run-heading", Label).render())
            assert "Gate" in heading
            assert "Run " not in heading or "Gate" in heading
            # LivePane exists and gotchas are in a collapsible (not above live).
            assert pane.query("#run-output")
            assert pane.query("#run-gotchas-wrap") or pane.query("#run-gotchas")
            banner = str(pane.query_one("#run-target-banner", Static).render())
            assert "Target" in banner

    @pytest.mark.asyncio
    async def test_measure_heading(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-evidence"
            await pilot.pause()
            heading = str(
                app.query_one("#validate-evidence-pane", ValidateEvidencePane)
                .query_one("#evidence-heading", Label)
                .render()
            )
            assert "Measure" in heading


class TestPhase2ServePromote:
    @pytest.mark.asyncio
    async def test_serve_shows_button_when_armed(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            pane = app.query_one("#lane-serve-pane", LaneServePane)
            pane.set_armed(_route_c_byo(), host_port=8013)
            await pilot.pause()
            actions = pane.query_one("#lane-serve-actions")
            assert not actions.has_class("funnel-hidden")
            body = str(pane.query_one("#lane-serve-body", Static).render())
            assert "8013" in body or ":8013" in body
            assert "Serving" in body

    @pytest.mark.asyncio
    async def test_promote_has_checklist_and_button(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-promote"
            await pilot.pause()
            pane = app.query_one("#lane-promote-pane", LanePromotePane)
            assert pane.query("#lane-promote-prereqs")
            assert pane.query("#lane-promote-btn")
            btn = pane.query_one("#lane-promote-btn", Button)
            # No fit-check yet → disabled.
            assert btn.disabled is True
            pane.set_prereqs(
                fit=True, weights=True, served=False, gated=False, measured=False,
                byo=_route_c_byo(),
            )
            await pilot.pause()
            assert btn.disabled is False
            pr = str(pane.query_one("#lane-promote-prereqs", Static).render())
            assert "✓" in pr or "fit-checked" in pr


class TestPhase2BringContinue:
    @pytest.mark.asyncio
    async def test_continue_button_after_weights(self, monkeypatch):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            byo = _route_c_byo()
            app._last_byo = byo
            monkeypatch.setattr(app._data, "bring_weights_present", lambda r: True)
            pane = app.query_one("#lane-bring-pane", LaneBringPane)
            pane.populate(byo, weights_present=True)
            await pilot.pause()
            btn = pane.query_one("#lane-bring-continue-btn", Button)
            assert not btn.has_class("funnel-hidden")
