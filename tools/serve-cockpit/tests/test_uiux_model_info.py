"""C6 mechanical UX wins — model-info popup [i], ★ default marker, degraded-catalog
banner, and the no-op notify on empty-catalog Explain / [i].

Runs headless via make_app (FakeRunner canned responses) — same pattern as the
other uiux phase files.
"""

from __future__ import annotations

import json

import pytest
from textual.widgets import DataTable, Label, Static

from club3090_cockpit.app import ModelInfoScreen
from club3090_cockpit.services import RunResult
from tests.test_app_headless import _settle, fake_responses, make_app, ok


# A registry payload that carries the C6 model-profile metadata (what
# registry-emit.sh _model() now emits) + a status_note on the variant + the
# curated defaults array (drives the ★ marker).
REGISTRY_JSON_C6 = json.dumps(
    {
        "defaults": [
            {"engine": "vllm", "model": "qwen3.6-27b", "slug": "vllm/dual",
             "source": "curated", "topology": "dual"},
        ],
        "profiles": {
            "models": {
                "qwen3.6-27b": {
                    "family": "qwen3_next",
                    "description": None,
                    "display_name": "Qwen 3.6 27B",
                    "active_params_b": 27.0,
                    "vision_capable": True,
                    "valid_tp": [1, 2],
                    "max_ctx": 262144,
                    "hf_repo": "Qwen/Qwen3.6-27B-FP8",
                    "weights": {},
                },
            },
        },
        "variants": [
            {
                "slug": "vllm/dual",
                "switch_engine": "vllm",
                "launch_engine": "vllm",
                "compose_dir": "models/qwen3.6-27b/vllm/compose/dual/autoround-int4",
                "file": "fp8-mtp.yml",
                "port": 8010,
                "model": "qwen3.6-27b",
                "engine": "vllm-stable",
                "kvcalc_key": "qwen3.6-27b:dual",
                "container": "vllm_qwen36_27b",
                "compose_path": "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml",
                "status": "caveats",
                "ctx_label": "262K",
                "configured_ctx": 262144,
                "status_note": "pin vLLM ≤ v0.24.x until the MoE regression lands",
                "source": "curated",
                "baseline": {
                    "narr_tps": 174.0, "code_tps": 42.0, "quality_8pk": "109/150",
                    "date": "2026-07-01", "engine_pin": "vllm/vllm-openai:v0.24.0",
                    "current_pin": "vllm/vllm-openai:v0.24.0", "stale": False,
                    "rig": "2x3090-pcie", "power_cap_w": [370, 420],
                    "submitted_by": "noonghunna",
                },
            },
        ],
    }
)

# Raw-tab fallback rows (registry_variant_rows shape) for the degraded-catalog
# path: the --json emit FAILED but the tab emitter still paints.
TAB_ROWS = (
    "VARIANT\tvllm/dual\tvllm\tvllm\tmodels/qwen3.6-27b/vllm/compose/dual/autoround-int4"
    "\tfp8-mtp.yml\t8010\tqwen3.6-27b\tvllm-stable\tqwen3.6-27b:dual"
    "\tvllm_qwen36_27b\tmodels/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml"
    "\tproduction\t262K\t\n"
)


def c6_responses(**extra) -> dict:
    return fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_C6), **extra})


def empty_catalog_responses() -> dict:
    return fake_responses(**{"registry-emit.sh --json": ok(json.dumps({"variants": []}))})


class TestModelInfoPopup:
    @pytest.mark.asyncio
    async def test_popup_renders_local_data(self):
        """[i] on a focused catalog row opens ModelInfoScreen built ONLY from
        already-loaded local data: display_name from the registry model profile,
        status_note from the variant row."""
        app, _, _ = make_app(responses=c6_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("i")
            await pilot.pause()
            assert isinstance(app.screen, ModelInfoScreen)
            body = str(app.screen.query_one("#model-info-body", Static).render())
            assert "Qwen 3.6 27B" in body          # display_name (profile metadata)
            assert "qwen3_next" in body            # family
            assert "27B active" in body            # active_params_b
            assert "yes" in body                   # vision_capable
            assert "Qwen/Qwen3.6-27B-FP8" in body  # hf_repo
            assert "caveats" in body               # status
            assert "pin vLLM ≤ v0.24.x" in body    # status_note
            assert "174/42" in body                # baseline TPS bar
            # Esc dismisses back to the app screen without touching panes.
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ModelInfoScreen)

    @pytest.mark.asyncio
    async def test_popup_noop_notifies_when_no_row(self):
        """[i] with an EMPTY catalog notifies instead of silently returning."""
        app, _, _ = make_app(responses=empty_catalog_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("i")
            await pilot.pause()
            assert not isinstance(app.screen, ModelInfoScreen)
            notifications = [n.message for n in app._notifications]
            assert any("No slug selected" in str(m) for m in notifications)

    @pytest.mark.asyncio
    async def test_explain_noop_notifies_when_no_row(self):
        """[e] with an EMPTY catalog notifies instead of silently returning."""
        app, _, _ = make_app(responses=empty_catalog_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("e")
            await pilot.pause()
            notifications = [n.message for n in app._notifications]
            assert any("No slug selected" in str(m) for m in notifications)


class TestDefaultMarker:
    @pytest.mark.asyncio
    async def test_default_slug_row_carries_star(self):
        """The row whose slug is the registry's curated default for its
        (model, topology) renders a ★ marker in the slug cell."""
        app, _, _ = make_app(responses=c6_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            table = app.query_one("#catalog-table", DataTable)
            row0 = [str(c) for c in table.get_row_at(0)]
            assert any("★" in c and "vllm/dual" in c for c in row0)

    @pytest.mark.asyncio
    async def test_non_default_row_has_no_star(self):
        """A slug absent from the defaults array gets no ★."""
        reg = json.loads(REGISTRY_JSON_C6)
        reg["defaults"] = []
        responses = fake_responses(**{"registry-emit.sh --json": ok(json.dumps(reg))})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            table = app.query_one("#catalog-table", DataTable)
            row0 = [str(c) for c in table.get_row_at(0)]
            assert not any("★" in c for c in row0)


class TestDegradedCatalogBanner:
    @pytest.mark.asyncio
    async def test_fallback_rows_surface_the_json_error(self):
        """When --json fails but the raw-tab fallback succeeds, the rows still
        paint AND the status line carries a one-line notice with the underlying
        error (never silently dropped)."""
        responses = fake_responses(
            **{
                # --json emit fails (empty stdout + rc!=0) → fallback path.
                "registry-emit.sh --json": RunResult(
                    returncode=1, stdout="", stderr="compose_registry exploded"
                ),
                # The raw-tab fallback still produces rows.
                "registry_variant_rows": ok(TAB_ROWS),
            }
        )
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            status = str(app.query_one("#catalog-status", Label).render())
            assert "registry JSON emit failed" in status
            assert "compose_registry exploded" in status
            # The rows still painted (reduced columns) — not the error screen.
            table = app.query_one("#catalog-table", DataTable)
            assert table.row_count >= 1

    @pytest.mark.asyncio
    async def test_total_failure_still_errors(self):
        """Both paths failing keeps the hard error (no rows, red Catalog error)."""
        responses = fake_responses(
            **{
                "registry-emit.sh --json": RunResult(
                    returncode=1, stdout="", stderr="emit died"
                ),
                "registry_variant_rows": RunResult(
                    returncode=1, stdout="", stderr="fallback died too"
                ),
            }
        )
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            status = str(app.query_one("#catalog-status", Label).render()).lower()
            assert "error" in status


class TestHelpScrollAndFooter:
    @pytest.mark.asyncio
    async def test_help_body_is_scrollable(self):
        """The help body lives in a VerticalScroll — a <50-row terminal scrolls
        instead of clipping the giant Static."""
        from textual.containers import VerticalScroll

        from club3090_cockpit.app import HelpScreen

        app, _, _ = make_app(responses=c6_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            scrolls = app.screen.query(VerticalScroll)
            content = str(app.screen.query("HelpScreen VerticalScroll Static").first().render())
            assert "Keybindings" in content

    def test_footer_brackets_symmetric(self):
        """[ and ] are both footer-visible (show=True) — no one-sided hide."""
        from club3090_cockpit.app import CockpitApp

        shows = {}
        for b in CockpitApp.BINDINGS:
            action = getattr(b, "action", None)
            if action in ("prev_subtab", "next_subtab"):
                shows[action] = bool(getattr(b, "show", True))
        assert shows == {"prev_subtab": True, "next_subtab": True}
