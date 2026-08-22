"""c3 Bring lane — HF search front-end ([f] / "Search HF" → SearchHFScreen).

Covers:
  1. ``scripts/lib/profiles/hf_search.py`` (the subprocess contract): URL
     shape, tag→format derivation, HF_TOKEN bearer, empty-is-valid, and
     non-zero-exit+stderr on failure.  The transport is monkeypatched —
     NO live network in the suite.
  2. The modal UX: canned FakeRunner rows render; ⏎ on a row dismisses
     returning the repo id which fills #lane-bring-url-input (+ refocus);
     a failed search notifies and leaves manual entry untouched.
  3. Binding hygiene: [f] is context-gated to mode 1 · tab-bring (hidden /
     inert elsewhere) and registered in Help + the command palette.

Pattern: tests/test_uiux_model_info.py (make_app + FakeRunner + pilot).
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.parse
from pathlib import Path

import pytest
from textual.coordinate import Coordinate
from textual.widgets import Button, DataTable, Input, Static, TabbedContent

from club3090_cockpit.app import (
    HelpScreen,
    SearchHFScreen,
    _PALETTE_COMMANDS,
    _PALETTE_PRODUCER_ONLY,
)
from tests.test_app_headless import (
    RunResult,
    _settle,
    fake_responses,
    make_app,
    ok,
)


def hf_responses(**extra) -> dict:
    return fake_responses(**{"hf_search.py": ok(HF_ROWS), **extra})


def hf_error_response() -> dict:
    return fake_responses(
        **{
            "hf_search.py": RunResult(
                returncode=1, stdout="", stderr="lookup huggingface.co: boom"
            ),
        }
    )


async def _enter_bring(pilot) -> None:
    """Enter the Bring & Validate lane (mode 1), land on ① Bring."""
    await pilot.press("2")
    try:
        pilot.app.query_one("#validate-tabs", TabbedContent).active = "tab-bring"
    except Exception:
        pass
    await _settle(pilot)


# The normalized row contract hf_search.py --json prints (what CockpitData
# .hf_search hands to the modal).  Raw hub-API shapes are covered by the CLI
# unit tests below.
HF_ROWS = json.dumps(
    [
        {
            "id": "unsloth/Qwen3-27B-GGUF",
            "downloads": 1234567,
            "likes": 301,
            "last_modified": "2026-08-01T10:00:00.000Z",
            "gguf": True,
            "safetensors": False,
            "pipeline_tag": "text-generation",
        },
        {
            "id": "org/Model-B",
            "downloads": 5321,
            "likes": 12,
            "last_modified": "2026-07-11T00:00:00.000Z",
            "gguf": False,
            "safetensors": True,
            "pipeline_tag": "",
        },
    ]
)


async def _search_in_modal(pilot, app, query: str = "qwen3") -> None:
    """Open the modal via [f], type the query, submit it, settle workers."""
    await pilot.press("f")
    await pilot.pause()
    inp = app.screen.query_one("#hf-search-input", Input)
    inp.value = query
    await pilot.press("enter")  # Input.Submitted → search worker
    await _settle(pilot)


# ===========================================================================
# 1. The subprocess CLI — scripts/lib/profiles/hf_search.py
# ===========================================================================


def _load_cli():
    """Import hf_search.py directly off the scripts tree (it is a standalone
    stdlib script, not part of the cockpit package)."""
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "lib" / "profiles" / "hf_search.py"
    assert path.exists(), f"missing CLI: {path}"
    spec = importlib.util.spec_from_file_location("c3_hf_search_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


RAW_API_PAYLOAD = [
    {
        # id wins over modelId; tags drive the format booleans.
        "id": "unsloth/Qwen3-27B-GGUF",
        "downloads": 1200000,
        "likes": 300,
        "lastModified": "2026-08-01T10:00:00.000Z",
        "tags": ["gguf", "text-generation"],
        "pipeline_tag": "text-generation",
    },
    {
        "modelId": "org/B",
        "downloads": None,
        "tags": ["safetensors"],
        "lastModified": "2026-07-11T00:00:00.000Z",
    },
]


class TestHfSearchCli:
    @pytest.fixture()
    def cli(self):
        return _load_cli()

    def test_url_contract(self, cli, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"], seen["timeout"] = req.full_url, timeout
            return _FakeResponse(b"[]")

        monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
        cli.search("qwen3 gguf", limit=7)
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(seen["url"]).query)
        assert qs["search"] == ["qwen3 gguf"]
        assert qs["limit"] == ["7"]
        assert qs["sort"] == ["downloads"]
        assert qs["direction"] == ["-1"]
        assert qs["full"] == ["false"]
        assert seen["timeout"] == cli._NET_TIMEOUT

    def test_rows_derive_formats_from_tags(self, cli, monkeypatch):
        monkeypatch.setattr(
            cli.urllib.request,
            "urlopen",
            lambda req, timeout=None: _FakeResponse(json.dumps(RAW_API_PAYLOAD).encode()),
        )
        rows = cli.search("anything")
        assert [r["id"] for r in rows] == ["unsloth/Qwen3-27B-GGUF", "org/B"]
        gguf_row, st_row = rows
        assert gguf_row["gguf"] is True and gguf_row["safetensors"] is False
        assert st_row["gguf"] is False and st_row["safetensors"] is True
        assert gguf_row["downloads"] == 1200000
        assert st_row["downloads"] == 0  # null coerced
        assert gguf_row["last_modified"].startswith("2026-08-01")

    def test_bearer_token_when_hf_token_set(self, cli, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["auth"] = req.headers.get("Authorization")
            return _FakeResponse(b"[]")

        monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setenv("HF_TOKEN", "hf_tok_123")
        cli.search("x")
        assert seen["auth"] == "Bearer hf_tok_123"
        monkeypatch.delenv("HF_TOKEN")
        cli.search("x")
        assert seen["auth"] is None

    def test_empty_result_is_valid(self, cli, monkeypatch, capsys):
        monkeypatch.setattr(
            cli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"[]")
        )
        rc = cli._main(["nothing-matches", "--json"])
        out = capsys.readouterr()
        assert rc == 0
        assert json.loads(out.out) == []
        assert not out.err

    def test_failure_nonzero_exit_and_stderr(self, cli, monkeypatch, capsys):
        def boom(req, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(cli.urllib.request, "urlopen", boom)
        rc = cli._main(["q", "--json"])
        out = capsys.readouterr()
        assert rc != 0
        assert "hf_search" in out.err
        assert out.out == ""  # stdout never carries partial output on failure

    def test_bad_limit_rejected(self, cli, capsys):
        assert cli._main(["q", "--limit", "0"]) != 0
        assert cli._main(["q", "--limit", "1000"]) != 0
        assert capsys.readouterr().err


# ===========================================================================
# 2. The modal UX
# ===========================================================================


class TestSearchHFModal:
    @pytest.mark.asyncio
    async def test_binding_opens_modal_and_renders_rows(self):
        app, runner, _ = make_app(responses=hf_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            await _search_in_modal(pilot, app)
            assert isinstance(app.screen, SearchHFScreen)
            table = app.screen.query_one("#hf-search-table", DataTable)
            assert table.row_count == 2
            assert table.get_cell_at(Coordinate(0, 0)) == "unsloth/Qwen3-27B-GGUF"
            status = str(app.screen.query_one("#hf-search-status", Static).render())
            assert "2 result(s)" in status
            # The Runner seam was used exactly once, with the CLI contract.
            hf_calls = [c for c in runner.calls if "hf_search.py" in " ".join(c)]
            assert len(hf_calls) == 1
            assert "scripts/lib/profiles/hf_search.py" in hf_calls[0]
            assert "--json" in hf_calls[0]

    @pytest.mark.asyncio
    async def test_enter_on_row_fills_bring_input_and_refocuses(self):
        app, _, _ = make_app(responses=hf_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            await _search_in_modal(pilot, app)
            # Results table is focused after a successful search; ⏎ picks row 0.
            assert isinstance(app.screen, SearchHFScreen)
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, SearchHFScreen)
            bring_inp = app.query_one("#lane-bring-url-input", Input)
            assert bring_inp.value == "unsloth/Qwen3-27B-GGUF"
            assert bring_inp.has_focus

    @pytest.mark.asyncio
    async def test_fill_button_dismisses_with_selected_id(self):
        app, _, _ = make_app(responses=hf_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            await _search_in_modal(pilot, app)
            # Move to the second row, then use the Fill button.
            table = app.screen.query_one("#hf-search-table", DataTable)
            table.move_cursor(row=1)
            app.screen.query_one("#hf-search-fill-btn", Button).press()
            await pilot.pause()
            assert not isinstance(app.screen, SearchHFScreen)
            assert app.query_one("#lane-bring-url-input", Input).value == "org/Model-B"

    @pytest.mark.asyncio
    async def test_escape_dismisses_without_touching_input(self):
        app, _, _ = make_app(responses=hf_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            await pilot.press("f")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, SearchHFScreen)
            assert app.query_one("#lane-bring-url-input", Input).value == ""

    @pytest.mark.asyncio
    async def test_error_notifies_and_manual_entry_unaffected(self):
        app, _, _ = make_app(responses=hf_error_response())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            # Manual entry first — it must survive the failed search verbatim.
            bring_inp = app.query_one("#lane-bring-url-input", Input)
            bring_inp.value = "my-org/My-Model"
            await _search_in_modal(pilot, app)
            notifications = [str(n.message) for n in app._notifications]
            assert any("HF search failed" in m for m in notifications)
            assert any("huggingface.co" in m for m in notifications)
            assert isinstance(app.screen, SearchHFScreen)  # modal stays open
            assert app.query_one("#lane-bring-url-input", Input).value == "my-org/My-Model"

    @pytest.mark.asyncio
    async def test_empty_results_is_a_valid_state(self):
        empty = json.dumps([])
        app, _, _ = make_app(
            responses=fake_responses(**{"hf_search.py": ok(empty)})
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            await _search_in_modal(pilot, app, query="zzz-no-such-model")
            notifications = [str(n.message) for n in app._notifications]
            assert not any("failed" in m for m in notifications)
            status = str(app.screen.query_one("#hf-search-status", Static).render())
            assert "No results" in status


# ===========================================================================
# 3. Binding / Help / palette registration hygiene
# ===========================================================================


class TestSearchHFRegistration:
    @pytest.mark.asyncio
    async def test_f_gated_to_mode1_tabbring(self):
        app, _, _ = make_app(responses=hf_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Merged mode 0 · Catalog: gated OFF — inert AND hidden.
            assert app.check_action("search_hf", ()) is False
            await pilot.press("f")
            await pilot.pause()
            assert not isinstance(app.screen, SearchHFScreen)

    @pytest.mark.asyncio
    async def test_f_gated_off_other_lane_tabs(self):
        app, _, _ = make_app(responses=hf_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await _settle(pilot)
            assert app.check_action("search_hf", ()) is False
            await pilot.press("f")
            await pilot.pause()
            assert not isinstance(app.screen, SearchHFScreen)

    @pytest.mark.asyncio
    async def test_check_action_true_on_bring_tab(self):
        app, _, _ = make_app(responses=hf_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            assert app.check_action("search_hf", ()) is True

    @pytest.mark.asyncio
    async def test_search_button_on_bring_pane(self):
        app, _, _ = make_app(responses=hf_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            btn = app.query_one("#lane-bring-search-btn", Button)
            btn.press()
            await pilot.pause()
            assert isinstance(app.screen, SearchHFScreen)

    def test_palette_registration(self):
        actions = {a for a, _t, _h in _PALETTE_COMMANDS}
        assert "search_hf" in actions
        # Producer-only: hidden on the consumer surface palette too.
        assert "search_hf" in _PALETTE_PRODUCER_ONLY

    @pytest.mark.asyncio
    async def test_help_teaches_the_key(self):
        app, _, _ = make_app(responses=hf_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            assert "search HF" in app.screen.help_text


# ===========================================================================
# Opt-in live check (skipped unless CLUB3090_LIVE_NETWORK=1)
# ===========================================================================


@pytest.mark.live_network
@pytest.mark.skipif(
    os.environ.get("CLUB3090_LIVE_NETWORK") != "1",
    reason="live Hugging Face API — opt in with CLUB3090_LIVE_NETWORK=1",
)
def test_live_network_hf_search():
    cli = _load_cli()
    rows = cli.search("qwen3", limit=3)
    assert isinstance(rows, list) and len(rows) == 3
    for r in rows:
        assert r["id"]
        assert isinstance(r["gguf"], bool)
