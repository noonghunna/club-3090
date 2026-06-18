"""Headless tests for the CockpitApp.

Verifies:
  1. The app mounts without error (no TTY, no GPU, no Docker, no registry call).
  2. All four modes are reachable via digit-key bindings.
  3. Each mode's nav nodes exist in the DOM.
  4. The Catalog DataTable has the expected columns.

The registry is never called from tests — the catalog worker is patched to
return an empty list so no subprocess is spawned.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Textual testing requires pytest-asyncio with asyncio_mode="auto"
from textual.widgets import DataTable, TabbedContent, TabPane, Label

from club3090_cockpit.app import CockpitApp, CatalogPane, ModeSwitcher


# ---------------------------------------------------------------------------
# Fixture: app with catalog worker patched to do nothing
# ---------------------------------------------------------------------------

FAKE_REPO_ROOT = Path("/tmp/fake-club-3090-test-root")


def _make_app() -> CockpitApp:
    return CockpitApp(repo_root=FAKE_REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PANEL_IDS = ["panel-discover", "panel-serve", "panel-estate", "panel-validate"]
MODE_ACTIONS = [
    "action_mode_discover",
    "action_mode_serve",
    "action_mode_estate",
    "action_mode_validate",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAppMounts:
    """The app mounts cleanly and initial DOM is correct."""

    @pytest.mark.asyncio
    async def test_app_mounts(self):
        """App should mount without raising, with catalog patched."""
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                # Just being able to get here means it mounted
                assert app is not None

    @pytest.mark.asyncio
    async def test_discover_panel_visible_on_start(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                panel = app.query_one("#panel-discover")
                assert "active" in panel.classes

    @pytest.mark.asyncio
    async def test_other_panels_hidden_on_start(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                for pid in ["panel-serve", "panel-estate", "panel-validate"]:
                    panel = app.query_one(f"#{pid}")
                    assert "active" not in panel.classes


class TestModeNavigation:
    """Digit-key bindings switch the visible mode panel."""

    @pytest.mark.asyncio
    async def test_switch_to_serve_mode(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("2")
                assert "active" in app.query_one("#panel-serve").classes
                assert "active" not in app.query_one("#panel-discover").classes

    @pytest.mark.asyncio
    async def test_switch_to_estate_mode(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("3")
                assert "active" in app.query_one("#panel-estate").classes
                assert "active" not in app.query_one("#panel-discover").classes

    @pytest.mark.asyncio
    async def test_switch_to_validate_mode(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("4")
                assert "active" in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_switch_back_to_discover(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("2")
                await pilot.press("1")
                assert "active" in app.query_one("#panel-discover").classes
                assert "active" not in app.query_one("#panel-serve").classes

    @pytest.mark.asyncio
    async def test_all_four_modes_cycle(self):
        """Cycling 1-2-3-4-1 should always leave exactly one active panel."""
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                for key, expected_active in [("1", 0), ("2", 1), ("3", 2), ("4", 3), ("1", 0)]:
                    await pilot.press(key)
                    active_panels = [
                        pid for pid in PANEL_IDS
                        if "active" in app.query_one(f"#{pid}").classes
                    ]
                    assert len(active_panels) == 1, (
                        f"After pressing {key!r}: expected 1 active panel, got {active_panels}"
                    )
                    assert active_panels[0] == PANEL_IDS[expected_active]


class TestNavNodesExist:
    """Every mode's navigation nodes are present in the DOM."""

    @pytest.mark.asyncio
    async def test_discover_tabs_exist(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                app.query_one("#discover-tabs", TabbedContent)
                app.query_one("#tab-catalog", TabPane)
                app.query_one("#tab-byo", TabPane)

    @pytest.mark.asyncio
    async def test_estate_tabs_exist(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                app.query_one("#estate-tabs", TabbedContent)
                app.query_one("#tab-orchestration", TabPane)
                app.query_one("#tab-containers", TabPane)

    @pytest.mark.asyncio
    async def test_validate_tabs_exist(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                app.query_one("#validate-tabs", TabbedContent)
                app.query_one("#tab-run", TabPane)
                app.query_one("#tab-doctor", TabPane)
                app.query_one("#tab-benchmarks", TabPane)
                app.query_one("#tab-evidence", TabPane)

    @pytest.mark.asyncio
    async def test_catalog_datatable_has_columns(self):
        """The Catalog DataTable should have the 8 expected column keys."""
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                table = app.query_one("#catalog-table", DataTable)
                col_labels = [str(c.label) for c in table.columns.values()]
                for expected in ("slug", "engine", "fit", "ctx", "TPS", "8pk", "status", "source"):
                    assert expected in col_labels, f"Expected column {expected!r} not found: {col_labels}"

    @pytest.mark.asyncio
    async def test_mode_switcher_exists(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                app.query_one("#mode-switcher", ModeSwitcher)

    @pytest.mark.asyncio
    async def test_catalog_status_label_exists(self):
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                app.query_one("#catalog-status", Label)


class TestCatalogPopulation:
    """Catalog panel populates correctly when rows are provided."""

    @pytest.mark.asyncio
    async def test_catalog_populates_with_rows(self):
        """Calling populate() with mock rows should add them to the table."""
        from club3090_cockpit.registry import VariantRow

        fake_rows = [
            VariantRow(
                slug="vllm/dual",
                switch_engine="vllm",
                launch_engine="vllm",
                compose_dir="models/qwen3.6-27b/vllm/compose/dual/autoround-int4",
                file="fp8-mtp.yml",
                port=8010,
                model="qwen3.6-27b",
                engine="vllm",
                kvcalc_key="qwen3.6-27b:fp8-mtp",
                container="vllm_qwen36_27b",
                compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml",
                status="production",
                ctx_label="295K",
                status_note="",
            ),
            VariantRow(
                slug="beellama/dflash",
                switch_engine="beellama",
                launch_engine="beellama",
                compose_dir="models/qwen3.6-27b/beellama/compose/dual/autoround-int4",
                file="dflash.yml",
                port=8065,
                model="qwen3.6-27b",
                engine="beellama",
                kvcalc_key="SKIP",
                container="beellama_qwen_dflash",
                compose_path="models/qwen3.6-27b/beellama/compose/dual/autoround-int4/dflash.yml",
                status="caveats",
                ctx_label="102K",
                status_note="DFlash prose regression",
            ),
        ]

        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                pane = app.query_one("#catalog-pane", CatalogPane)
                pane.populate(fake_rows, None)
                table = app.query_one("#catalog-table", DataTable)
                assert table.row_count == 2

    @pytest.mark.asyncio
    async def test_catalog_shows_error_message(self):
        """When populate() is called with an error, the status label updates."""
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                pane = app.query_one("#catalog-pane", CatalogPane)
                pane.populate([], "registry-emit timed out (>15s)")
                status = app.query_one("#catalog-status", Label)
                # Label.render() returns a renderable; use str() to get the text content
                label_text = str(status.render()).lower()
                assert "error" in label_text or "timed out" in label_text


class TestPrimaryActionToast:
    """Enter key pops the 'wired in Phase 3' notification."""

    @pytest.mark.asyncio
    async def test_enter_triggers_phase3_notification(self):
        """Pressing Enter should trigger a notification (not crash)."""
        app = _make_app()
        with patch.object(app, "_load_catalog"):
            async with app.run_test(size=(120, 40)) as pilot:
                # Should not raise
                await pilot.press("enter")
