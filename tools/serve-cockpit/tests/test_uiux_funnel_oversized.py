"""Bring funnel — oversized-artifact verdict + reachable custom-slug hatch.

Regression coverage for the 2026-07-09 dogfood (migtissera/Tess-4-27B, 54 GB bf16
on a 48 GB rig): the §2b size floor hid every slug, leaving only the ✎ sentinel
with (A) no "won't fit" explanation and (B) an unreachable custom-slug Input (the
sole sentinel is pre-selected under prevent(Select.Changed), so the Changed-gated
reveal never fires)."""

from __future__ import annotations

import pytest
from textual.widgets import Input, Label

from club3090_cockpit.app import LaneBringPane, PROFILE_CUSTOM_SENTINEL
from tests.test_app_headless import _settle, make_app


class TestCustomHatchReachable:
    @pytest.mark.asyncio
    async def test_sentinel_only_reveals_custom_input(self):
        """Bug B — when the only option is the sentinel, the custom Input must be
        revealed eagerly (the Changed-gated reveal can't fire on a sole option)."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            pane = app.query_one("#lane-bring-pane", LaneBringPane)
            pane.reveal_slug_stage([("✎ custom slug…", PROFILE_CUSTOM_SENTINEL)], None)
            await pilot.pause()
            custom = pane.query_one("#lane-bring-profile-custom", Input)
            assert not custom.has_class("profile-custom-hidden"), (
                "sentinel-only dropdown must reveal the custom-slug Input"
            )

    @pytest.mark.asyncio
    async def test_real_slugs_keep_custom_hidden(self):
        """The hatch stays hidden (until a genuine pick) whenever real slugs exist."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            pane = app.query_one("#lane-bring-pane", LaneBringPane)
            pane.reveal_slug_stage(
                [("dual/vllm/foo", "vllm/foo"),
                 ("✎ custom slug…", PROFILE_CUSTOM_SENTINEL)],
                "vllm/foo",
            )
            await pilot.pause()
            custom = pane.query_one("#lane-bring-profile-custom", Input)
            assert custom.has_class("profile-custom-hidden")


class TestOversizedVerdict:
    @pytest.mark.asyncio
    async def test_oversized_safetensors_shows_wont_fit(self, monkeypatch):
        """Bug A — a safetensors artifact bigger than every hostable topology gets
        an honest "won't fit" hint, not a silent empty dropdown."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            entries, _err = await app._data.load_catalog_rows()
            app._variants = [e.row for e in entries]
            # pin a 2×24 GB rig so the floor is deterministic
            monkeypatch.setattr(app, "_known_gpu_vram_gb", lambda: 24.0)
            monkeypatch.setattr(app, "_known_gpu_count", lambda: 2)
            app._reveal_funnel_slugs("safetensors", 54.0)  # bigger than 48 GB total
            await pilot.pause()
            hint = str(app.query_one("#lane-bring-hint", Label).render()).lower()
            assert "won't fit" in hint or "wont fit" in hint, hint
            # and the escape hatch is reachable in this state
            custom = app.query_one("#lane-bring-profile-custom", Input)
            assert not custom.has_class("profile-custom-hidden")

    @pytest.mark.asyncio
    async def test_normal_size_shows_slugs_no_wont_fit(self, monkeypatch):
        """A normal-size artifact keeps the recommended list (no false 'won't fit')."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            entries, _err = await app._data.load_catalog_rows()
            app._variants = [e.row for e in entries]
            monkeypatch.setattr(app, "_known_gpu_vram_gb", lambda: 24.0)
            monkeypatch.setattr(app, "_known_gpu_count", lambda: 2)
            app._reveal_funnel_slugs("safetensors", 15.0)
            await pilot.pause()
            hint = str(app.query_one("#lane-bring-hint", Label).render()).lower()
            assert "won't fit" not in hint and "wont fit" not in hint
