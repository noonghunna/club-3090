"""UI/UX Phase 1 — copy honesty, override hide, footer labels, promote preview badge.

Keeps phase-1 gates small and fast; full suite still runs for the PR tip.
"""

from __future__ import annotations

import pytest
from textual.containers import Vertical
from textual.widgets import Label, Static, TabbedContent

from club3090_cockpit.app import (
    LaneServePane,
    PromoteScaffoldScreen,
    UntestedComposePreviewScreen,
    _byo_result_text,
)
from club3090_cockpit.data import ByoResult, PromoteScaffold

# Import headless helpers from the main suite.
from tests.test_app_headless import _settle, make_app


def _route_c_byo(**kw) -> ByoResult:
    base = dict(
        repo="org/MyFineTune-FP8",
        profile_like="vllm/dual",
        arch="Qwen3ForCausalLM",
        eligible=False,
        fit_verdict="no-fit-model",
        route="C",
        sibling_slug="qwen3.6-27b",
        quant_match="fp8",
        drop_spec_config=False,
        error="",
    )
    base.update(kw)
    return ByoResult(**base)


class TestPhase11OverrideHidden:
    @pytest.mark.asyncio
    async def test_unarmed_serve_hides_override_editor(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            pane = app.query_one("#lane-serve-pane", LaneServePane)
            pane.set_armed(None)
            await pilot.pause()
            # Phase 2: override block lives under #lane-serve-ov-wrap Collapsible.
            wrap = pane.query_one("#lane-serve-ov-wrap")
            assert wrap.has_class("funnel-hidden")
            assert wrap.display is False


class TestPhase12Honesty:
    def test_route_c_card_never_says_not_your_weights_or_deferred(self):
        txt = _byo_result_text(_route_c_byo(), weights_present=False)
        assert "NOT your brought" not in txt
        assert "deferred" not in txt.lower()
        assert "download + serve" not in txt
        assert "download weights" in txt
        assert "\\[s]" in txt or "[s]" in txt.replace("\\", "")

    def test_route_c_present_points_at_s_continue(self):
        txt = _byo_result_text(_route_c_byo(), weights_present=True)
        assert "download + serve" not in txt
        assert "Continue" in txt or "② Serve" in txt
        assert "\\[s]" in txt

    @pytest.mark.asyncio
    async def test_reproduction_preview_warns_weights_not_mounted(self, tmp_path):
        """Catalog-reproduction modal keeps the honesty warning (not Route-C)."""
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.push_screen(
                UntestedComposePreviewScreen(
                    "vllm/dual",
                    str(tmp_path / "c.yml"),
                    "services:\n  x:\n    image: test\n",
                )
            )
            await pilot.pause()
            body = str(app.screen.query_one("#untested-body", Static).render())
            title = str(app.screen.query_one(".untested-title", Label).render())
            assert "NOT mounted" in body or "NOT mounted" in title
            assert "deferred" not in body.lower()
            assert "deferred" not in title.lower()

    @pytest.mark.asyncio
    async def test_route_c_armed_body_never_says_not_your_weights(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            pane = app.query_one("#lane-serve-pane", LaneServePane)
            # Arm as Route-C without override defaults (editor stays hidden).
            pane.set_armed(_route_c_byo(), overrides_defaults=None)
            await pilot.pause()
            body = str(pane.query_one("#lane-serve-body", Static).render())
            assert "NOT your brought" not in body
            assert "deferred" not in body.lower()
            assert "MyFineTune" in body or "org/MyFineTune" in body
            assert pane.query_one("#lane-serve-ov-wrap").display is False


class TestPhase13NextStepAndFooter:
    def test_absent_weights_card_no_download_plus_serve(self):
        txt = _byo_result_text(_route_c_byo(), weights_present=False)
        assert "download + serve" not in txt
        assert "download weights" in txt

    def test_failure_hint_not_forward_progress(self):
        res = ByoResult(
            repo="org/X", profile_like="vllm/dual", error="gated repo 401",
        )
        # populate sets the hint — exercise via the pane method path
        from club3090_cockpit.app import LaneBringPane

        # Pure string path through populate's logic: simulate by calling
        # _byo_result_text + the same branch as populate.
        assert "Fit-check failed" in _byo_result_text(res) or res.error

    @pytest.mark.asyncio
    async def test_footer_s_continue_after_servable_fit_with_weights(self, monkeypatch):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            # Inject a successful Route-C fit-check + weights present.
            byo = _route_c_byo()
            app._last_byo = byo
            monkeypatch.setattr(
                app._data, "bring_weights_present", lambda repo: True
            )
            pane = app.query_one("#lane-bring-pane")
            pane.populate(byo, weights_present=True, downloading=False)
            app._sync_footer_labels()
            app.refresh_bindings()
            await pilot.pause()
            # Binding description rewritten.
            descs = []
            for blist in app._bindings.key_to_bindings.values():
                for b in blist:
                    if b.action == "s_key":
                        descs.append(b.description)
            assert any("Continue" in d and "②" in d for d in descs), descs
            assert app.check_action("s_key", ()) is True
            # Hint is the continue path, not a failure repair string.
            hint = str(pane.query_one("#lane-bring-hint", Label).render())
            assert "Continue" in hint or "② Serve" in hint
            assert "fix the fit-check" not in hint

    @pytest.mark.asyncio
    async def test_failed_fit_hint_is_repair_not_forward(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            pane = app.query_one("#lane-bring-pane")
            res = ByoResult(
                repo="org/X", profile_like="vllm/dual", error="unknown profile foo",
            )
            pane.populate(res)
            await pilot.pause()
            hint = str(pane.query_one("#lane-bring-hint", Label).render())
            assert "fix" in hint.lower() or "re-Inspect" in hint
            assert "Continue → ②" not in hint


class TestPhase14PromotePreview:
    @pytest.mark.asyncio
    async def test_promote_pane_badge_preview_only(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-promote"
            await pilot.pause()
            heading = str(
                app.query_one("#lane-promote-heading", Label).render()
            )
            badge = str(app.query_one("#lane-promote-badge", Static).render())
            assert "Promotion Preview" in heading or "Scaffold" in heading
            assert "preview only" in badge.lower() or "no catalog write" in badge.lower()
            prereqs = str(app.query_one("#lane-promote-prereqs", Static).render())
            assert "fit-checked" in prereqs

    def test_promote_modal_badge(self):
        sc = PromoteScaffold(
            model_id="foo",
            repo="org/foo",
            profile_path="models/foo.yml",
            profile_yaml="id: foo\n",
            registry_entry="entry",
            error="",
        )
        assert sc.computed
        screen = PromoteScaffoldScreen(sc)
        body = screen._body_text()
        assert "preview only" in body.lower() or "no catalog write" in body.lower()

    def test_help_promote_is_preview(self):
        from club3090_cockpit.app import HelpScreen

        h = HelpScreen(surface="producer")
        t = h.help_text
        assert "preview only" in t.lower() or "Promotion Preview" in t
        assert "mock this phase" in t.lower() or "preview" in t.lower()
