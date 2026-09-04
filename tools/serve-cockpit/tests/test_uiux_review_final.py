"""Final UX-review batch — key-hint markup, footer refresh, focus, narrow terminals.

Each test here pins a behaviour that was observably wrong in the shipped app.
Where a test is a *class* guard (rather than one site), it sweeps the source so
a future edit that reintroduces the bug anywhere reds the suite.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from textual.content import Content
from textual.widgets import DataTable, Static

sys.path.insert(0, str(Path(__file__).parent))

from test_app_headless import _renderable_text, _settle, make_app  # noqa: E402

import club3090_cockpit.app as app_mod  # noqa: E402


# ── #5 · key hints must survive Textual markup ──────────────────────────────

# Style tags this codebase genuinely uses, plus regex character classes and
# type names that happen to live inside brackets in a string literal.  Anything
# NOT in here that a markup parse deletes is a key hint being eaten.
_LEGIT_TAGS = {
    "bold", "cyan", "dim", "green", "red", "yellow", "italic", "underline",
    "orange1", "b",
    "/bold", "/cyan", "/dim", "/green", "/red", "/yellow", "/italic",
    "/underline", "/orange1", "/b",
    "\x00", "/\x00", "bold \x00", "/bold \x00",
    "A-Za-z0-9_", "A-Za-z_", "a-z0-9_", "0-9a-f", "Kk", "KM", "/-", "A-Za-z",
    "ByoResult", "GpuInfo",
}

# Verified NOT markup-rendered, so their brackets survive as typed:
#   app.py  — the Header sub_title is plain text (Textual does not parse markup
#             there); proved by rendering a Header with a "[k]" sub_title.
#   services.py — these strings land in a RichLog(markup=True), which uses
#             *rich* markup.  Rich's tag regex requires a leading lowercase
#             letter / # / / / @, so "[S]" survives there (Textual's parser,
#             which the Static/Label/Toast paths use, would eat it).
#             "[log]" / "[apply-swap]" ARE eaten in that pane, but they are
#             log-line prefixes teed to the on-disk log as well — escaping them
#             would write a literal backslash into the log file.  Left alone
#             deliberately; tracked in the PR description, not here.
_KNOWN_SAFE = {
    ("club3090_cockpit/app.py", "k"),
    ("club3090_cockpit/services.py", "S"),
    ("club3090_cockpit/services.py", "log"),
    ("club3090_cockpit/services.py", "apply-swap"),
}


def _string_nodes(tree: ast.AST):
    """Yield (lineno, text) for every markup-RENDERED string literal, with
    f-string expression slots collapsed to \\x00 so a tag split across an
    f-string boundary is still seen as one tag.

    Skips docstrings and `DEFAULT_CSS` / `CSS` blocks: neither is ever parsed as
    markup, and both legitimately contain bracketed key names in comments."""
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = n.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docs.add(id(body[0].value))
        # CSS blocks: `DEFAULT_CSS = """…"""` / `CSS = """…"""`
        if isinstance(n, ast.Assign):
            names = {t.id for t in n.targets if isinstance(t, ast.Name)}
            if names & {"DEFAULT_CSS", "CSS"}:
                docs.add(id(n.value))
    seen = set()
    for node in ast.walk(tree):
        if id(node) in docs:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "\x00"
                for v in node.values
            )
        else:
            continue
        key = (node.lineno, text)
        if key in seen:
            continue
        seen.add(key)
        yield node.lineno, text


@pytest.mark.parametrize(
    "rel", ["club3090_cockpit/app.py", "club3090_cockpit/data.py", "club3090_cockpit/services.py"]
)
def test_key_hints_are_escaped_from_textual_markup(rel):
    """Textual markup deletes `[d]`, `[l]`, `[t]`, `[Y]`, `[esc]`… — i.e. exactly
    the key hints the UI text exists to teach.  The fix is the `\\[x]` escape the
    codebase already uses; this sweeps for any site that forgot it."""
    path = Path(__file__).resolve().parents[1] / rel
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for lineno, text in _string_nodes(tree):
        if "[" not in text:
            continue
        try:
            plain = Content.from_markup(text).plain
        except Exception:
            plain = None
        for m in re.finditer(r"(?<!\\)\[([^\[\]]{1,10})\]", text):
            tag = m.group(1)
            if plain is not None and m.group(0) in plain:
                continue  # survived the parse — not a tag
            if tag in _LEGIT_TAGS or (rel, tag) in _KNOWN_SAFE:
                continue
            offenders.append(f"{rel}:{lineno}  [{tag}]  in {text.strip()[:70]!r}")
    assert not offenders, "unescaped key hints (use \\[x]):\n" + "\n".join(offenders)


@pytest.mark.asyncio
async def test_first_run_guide_shows_the_keys_it_teaches():
    """FirstRunScreen's numbered guide rendered as '2 · Download  — fetch its
    weights' — the `[d]` / `[D]` / `[l]` it exists to teach were deleted by the
    markup parser.  Asserts on the RENDERED card, not the source string."""
    from club3090_cockpit.app import FirstRunScreen, ProfileOption

    opts = [
        ProfileOption("Model A · single", "model-a/single", "single"),
        ProfileOption("Model B · dual", "model-b/dual", "dual"),
    ]
    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot)
        await app.push_screen(FirstRunScreen(opts, recommended="model-a/single", gpu_count=2))
        await _settle(pilot)
        body = _renderable_text(app.screen.query_one("#first-run-body", Static))
        for key in ("[d]", "[D]", "[l]"):
            assert key in body, f"{key} missing from the rendered first-run guide:\n{body}"


# ── #6 · ⑤ Promote's footer must follow the stage gate ──────────────────────


def _footer_keys(screen) -> str:
    """The footer as the user reads it — 'key description' per rendered key.

    Footer.render() is a Blank (it paints through child FooterKey widgets), so
    a str(render()) assertion would test nothing."""
    from textual.widgets._footer import FooterKey

    return " | ".join(f"{k.key_display} {k.description}" for k in screen.query(FooterKey))


def _scaffold():
    from club3090_cockpit.app import PromoteScaffold

    return PromoteScaffold(
        model_id="foo",
        repo="org/foo",
        profile_path="models/foo.yml",
        profile_yaml="id: foo\n",
        registry_entry="entry",
        error="",
    )


@pytest.mark.asyncio
async def test_promote_footer_gains_enter_when_required_edits_fill():
    """`on_input_changed` enabled the stage buttons but never called
    `refresh_bindings()`, so the footer stayed 'esc Close' after ⏎ had become
    valid — the action that just unlocked was invisible.

    Enablement and `refresh_bindings()` are two halves of one state change."""
    from club3090_cockpit.app import PromoteScaffoldScreen
    from textual.widgets import Button, Input

    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(pilot)
        await app.push_screen(PromoteScaffoldScreen(_scaffold()))
        await _settle(pilot)
        await _settle(pilot)
        screen = app.screen

        assert screen.query_one("#promote-stage-btn", Button).disabled is True
        assert "Write LOCAL layer" not in _footer_keys(screen)

        screen.query_one("#promote-display-input", Input).value = "Foo Model"
        screen.query_one("#promote-family-input", Input).value = "qwen"
        await _settle(pilot)
        await _settle(pilot)

        assert screen.query_one("#promote-stage-btn", Button).disabled is False
        assert "Write LOCAL layer" in _footer_keys(screen), _footer_keys(screen)


# ── #2 · ⏎ is advertised consistently, in one place ─────────────────────────


@pytest.mark.asyncio
async def test_enter_advertised_consistently_across_table_and_non_table_panes():
    """⏎ used to appear in the footer on non-table panes and VANISH on
    table-focused ones (Catalog, Orchestration and ③ Gate all boot with a
    DataTable focused, and DataTable's own `enter → select_cursor` binding is
    show=False and wins over the app's).  The key worked in all of them — only
    the advertisement was inconsistent.

    ⏎ is now advertised in exactly one place, the Modes rail's
    `#mode-action-hint`, in EVERY pane and with the pane's real verb."""
    from textual.widgets import Label
    from textual.widgets._footer import FooterKey

    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(pilot)
        await _settle(pilot)

        # Catalog: a DataTable owns focus (the case that used to lose ⏎).
        assert isinstance(app.focused, DataTable), type(app.focused).__name__
        hint = _renderable_text(app.query_one("#mode-action-hint", Label)).strip()
        assert hint == "⏎ Serve", hint
        assert "enter" not in {k.key for k in app.query(FooterKey)}

        # Bring & Validate: focus is NOT a DataTable — used to be the pane where
        # the footer DID show ⏎, i.e. the source of the inconsistency.
        await pilot.press("2")
        await _settle(pilot)
        await _settle(pilot)
        hint = _renderable_text(app.query_one("#mode-action-hint", Label)).strip()
        assert hint == "⏎ Fit-check", hint
        assert "enter" not in {k.key for k in app.query(FooterKey)}


@pytest.mark.asyncio
async def test_enter_still_works_on_a_table_focused_pane():
    """Dropping show=True is a DISPLAY change only — ⏎ on a catalog row must
    still reach the primary action (via DataTable.RowSelected)."""
    app, _, _ = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(pilot)
        await _settle(pilot)
        fired = []
        app.action_primary_action = lambda *a, **k: fired.append(True)  # type: ignore[method-assign]
        app.query_one("#catalog-table", DataTable).focus()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert fired, "⏎ no longer routes to the primary action"


# ── #4 · no keyboard trap after Inspect ─────────────────────────────────────


async def _reveal_bring_stage2(app, pilot, monkeypatch=None):
    """Drive ① Bring to the post-Inspect stage-2 reveal (no HF call)."""
    await pilot.press("2")
    await _settle(pilot)
    entries, _err = await app._data.load_catalog_rows()
    app._variants = [e.row for e in entries]
    app._known_gpu_vram_gb = lambda: 24.0        # type: ignore[method-assign]
    app._known_gpu_count = lambda: 2             # type: ignore[method-assign]
    app._reveal_funnel_slugs("safetensors", 15.0)
    await _settle(pilot)
    await _settle(pilot)


@pytest.mark.asyncio
async def test_inspect_focuses_the_fit_check_button_not_the_select():
    """After the stage-2 reveal the app focused the config Select while the
    footer and Modes rail both advertise "⏎ Fit-check".  ⏎ on a focused Select
    opens its dropdown, so the advertised primary action was unreachable from
    the focus the app had just set — a keyboard trap.

    The Fit-check button is the right target: the Select is one Shift+Tab away
    and already defaults to the ⭐ recommendation."""
    from textual.widgets import Label

    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        await _reveal_bring_stage2(app, pilot)

        assert app.focused is not None
        assert app.focused.id == "lane-bring-fit-btn", (
            f"focus landed on {app.focused.id!r}, so ⏎ does not do what the rail "
            f"advertises"
        )
        # ...and the advertisement it has to match is still Fit-check.
        hint = _renderable_text(app.query_one("#mode-action-hint", Label)).strip()
        assert hint == "⏎ Fit-check", hint


@pytest.mark.asyncio
async def test_enter_after_inspect_runs_fit_check():
    """The end-to-end claim behind the focus change: pressing the advertised ⏎
    from the focus the app sets must actually fit-check."""
    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        await _reveal_bring_stage2(app, pilot)
        fired = []
        app._trigger_lane_bring = lambda *a, **k: fired.append(True)  # type: ignore
        await pilot.press("enter")
        await _settle(pilot)
        assert fired, "⏎ from the post-Inspect focus did not run the fit-check"


@pytest.mark.asyncio
async def test_bring_result_card_is_on_screen_at_80x24():
    """At 80x24 the verdict card sat below the fold after Inspect, so the pane
    answered the user's question off-screen with nothing saying so."""
    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        await _reveal_bring_stage2(app, pilot)
        card = app.query_one("#lane-bring-result-card", Static)
        assert card.display
        bottom = card.region.y + card.region.height
        assert card.region.y >= 0 and bottom <= 24, (
            f"result card {card.region} is off-screen at 80x24"
        )


# ── #7 · Settings must be saveable at 80x24 ─────────────────────────────────


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (120, 40)])
@pytest.mark.asyncio
async def test_settings_save_affordance_is_on_screen(size):
    """SettingsScreen was `height: auto` with ~27 rows of content and no
    max-height, so on a 24-row terminal BOTH the hint line and the
    `^s Save / esc Cancel` Footer rendered below the bottom of the screen — a
    first-run user on a small terminal had no visible way to save.  The card was
    also `width: 84` on an 80-col screen.

    Geometry alone is not enough here (an in-bounds region can still be clipped),
    so this asserts the footer's rendered TEXT is present too."""
    from club3090_cockpit.app import SettingsScreen
    from textual.widgets import Footer

    width, height = size
    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=size) as pilot:
        await _settle(pilot)
        await app.push_screen(SettingsScreen("/mnt/models/huggingface", False))
        await _settle(pilot)
        await _settle(pilot)

        card = app.screen.query_one("Vertical")
        assert card.region.y + card.region.height <= height, (
            f"settings card {card.region} overflows a {width}x{height} screen"
        )
        assert card.region.x + card.region.width <= width, (
            f"settings card {card.region} is wider than {width} cols"
        )

        footer = app.screen.query_one(Footer)
        assert footer.region.y + footer.region.height <= height, (
            f"the Save/Cancel footer is off-screen at {width}x{height}: {footer.region}"
        )

        console = Console(width=width, no_color=True, legacy_windows=False)
        with console.capture() as cap:
            console.print(app.screen._compositor)
        rendered = cap.get()
        assert "Save" in rendered, f"no visible way to save at {width}x{height}"
        assert "Cancel" in rendered


# ── #15 · Explain must not show dashes for facts the row behind it displays ──


@pytest.mark.asyncio
async def test_explain_falls_back_to_the_catalog_row_when_explain_has_no_registry():
    """`switch.sh --explain` can answer without a `registry` block.  Reading it
    with a "—" default rendered "Model — / Engine — / Status —" directly on top
    of a catalog row that was showing the real values.  The CatalogEntry carries
    the same registry facts, so the modal falls back to them."""
    from club3090_cockpit.app import CatalogPane

    app, _, _ = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(pilot)
        await _settle(pilot)
        entry = app.query_one("#catalog-pane", CatalogPane).selected_entry()
        assert entry.model and entry.engine and entry.status  # not a vacuous test

        app.action_explain()
        await _settle(pilot)
        await _settle(pilot)
        screen = app.screen
        # explain answered, but with nothing structured in it
        screen.set_detail({"registry": {}}, None)
        await _settle(pilot)

        body = _renderable_text(screen.query_one("#explain-body", Static))
        assert entry.model in body, body
        assert entry.engine in body, body
        assert entry.status in body, body
        for field in ("Model", "Engine", "Status"):
            assert f"{field}   —" not in body and f"{field}  —" not in body, body


@pytest.mark.asyncio
async def test_explain_error_still_reports_what_the_caller_knows():
    """A hard explain failure used to render only "explain failed: …", hiding
    the three identity fields the catalog row behind it was displaying."""
    from club3090_cockpit.app import CatalogPane

    app, _, _ = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(pilot)
        await _settle(pilot)
        entry = app.query_one("#catalog-pane", CatalogPane).selected_entry()
        app.action_explain()
        await _settle(pilot)
        await _settle(pilot)
        screen = app.screen
        screen.set_detail(None, "switch.sh exited 1")
        await _settle(pilot)

        body = _renderable_text(screen.query_one("#explain-body", Static))
        assert "explain failed" in body
        assert entry.model in body, body
        assert entry.engine in body, body


# ── #16–#22 · narrow terminals (80x24 and 100x30 are real users) ────────────


def _screen_text(app, width: int) -> str:
    """The compositor as the user sees it.  NEVER collapse whitespace here — it
    breaks matches across box-drawing characters and has twice reported a
    working feature as broken."""
    console = Console(width=width, no_color=True, legacy_windows=False)
    with console.capture() as cap:
        console.print(app.screen._compositor)
    return cap.get()


@pytest.mark.asyncio
async def test_pane_headings_wrap_instead_of_truncating_at_80():
    """#17 — the four pane headings are `width: auto` Labels, so at 80 cols they
    were CUT, not wrapped: "① Bring — inspect an HF model, or bring a co".  Each
    lost the clause that says what the pane is for.  They are now in the same
    `width: 1fr` rule the hint lines already used."""
    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("2")
        await _settle(pilot)
        await _settle(pilot)
        heading = app.query_one("#lane-bring-heading")
        assert heading.region.x + heading.region.width <= 80, heading.region
        assert heading.region.height > 1, "heading did not wrap — still truncating"
        assert "compose you already have" in _screen_text(app, 80)


@pytest.mark.asyncio
async def test_catalog_hint_shows_the_sort_and_columns_keys_at_80():
    """#18 — `#catalog-hint` wraps (it is in the `width: 1fr` list) but was
    `height: 1`, which clipped every wrapped row but the first.  At 80 cols the
    line stopped at "[c]", so `[s] sort` and `[|] columns` never rendered at any
    width below ~159 — and neither key had another teaching surface."""
    app, _, _ = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        await _settle(pilot)
        rendered = _screen_text(app, 80)
        assert "[s] sort" in rendered, rendered
        assert "[|] columns" in rendered, rendered


@pytest.mark.asyncio
async def test_bring_action_buttons_fit_at_80():
    """#19 — the three action buttons were fixed at 16/20/auto with 2-col
    gutters: 58 columns of demand in the 42 the pane gets at 80 wide, so
    "Validate compose" rendered as "Va"."""
    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("2")
        await _settle(pilot)
        await _settle(pilot)
        row = app.query_one("#lane-bring-actions")
        for button in row.query("Button"):
            right = button.region.x + button.region.width
            assert right <= 80, f"{button.id} overflows to col {right}"
        assert "Validate" in _screen_text(app, 80)


@pytest.mark.asyncio
async def test_catalog_table_survives_a_start_at_80x24():
    """#21 — `#serve-live` was a fixed 13-row sibling; on a 24-row terminal it
    squeezed the catalog DataTable to height 1, so not even its header row
    survived.  Starting a model made the list you started it from vanish."""
    app, _, _ = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        await _settle(pilot)
        app.query_one("#serve-live").add_class("serving")   # what _serving_live_pane does
        await _settle(pilot)
        await _settle(pilot)
        table = app.query_one("#catalog-table", DataTable)
        assert table.region.height >= 4, (
            f"catalog table collapsed to {table.region.height} rows after a Start"
        )
        assert "slug" in _screen_text(app, 80)


@pytest.mark.asyncio
async def test_containers_hint_is_two_rows_and_the_log_pane_keeps_lines():
    """#20 — the run-on Containers hint wrapped to FOUR of the 24 rows at 80x24,
    which is what squeezed the Logs RichLog down to two lines."""
    from textual.widgets import RichLog
    from test_app_headless import _enter_operate

    app, _, _ = make_app(surface="producer")
    async with app.run_test(size=(80, 24)) as pilot:
        await _enter_operate(pilot, tab="tab-containers")
        await _settle(pilot)
        hint = app.query_one("#containers-hint")
        assert hint.region.height <= 2, f"hint takes {hint.region.height} rows"
        log = app.query_one("#drill-logs").query_one(RichLog)
        assert log.region.height >= 4, f"log pane is {log.region.height} rows"


@pytest.mark.parametrize(
    "width,must_show",
    [(80, ("Sort",)), (100, ("Sort",)), (120, ("Sort", "Refresh", "Settings"))],
)
@pytest.mark.asyncio
async def test_footer_keeps_the_context_key_at_every_width(width, must_show):
    """#16 — the footer renders bindings in declaration order and the six
    globals are declared first, so the key that is actually about the pane you
    are looking at was the one that got cut: "s Sort" → "s So", "s Submit" → "s
    ".  `r`/`S` are width-gated now (both are in Help AND the command palette),
    and the keys keep working at every width — only the advertisement is gated."""
    app, _, _ = make_app()
    async with app.run_test(size=(width, 30)) as pilot:
        await _settle(pilot)
        await _settle(pilot)
        footer_line = _screen_text(app, width).splitlines()[-1]
        for label in must_show:
            assert label in footer_line, f"{label!r} clipped at {width}: {footer_line!r}"
        if width < 120:
            assert "Refresh" not in footer_line
        # the gated keys still WORK — hidden is not disabled
        assert app.check_action("refresh", ()) is not False
        assert app.check_action("settings", ()) is not False


@pytest.mark.parametrize(
    "width,expected",
    [
        (80, ["slug", "ctx", "tps", "status"]),
        (100, ["model", "slug", "ctx", "tps", "status"]),
        (120, [k for k, _ in __import__("club3090_cockpit.app", fromlist=["x"])._CATALOG_COLUMNS]),
    ],
)
@pytest.mark.asyncio
async def test_catalog_narrow_column_default(width, expected):
    """#22 — with all 16 columns on, the table needs 151 cols and gets 46 at 80
    wide, so status/ctx/TPS (the columns you PICK on) sat ~100 columns
    off-screen right behind an unadvertised horizontal scroll.  Narrow widths
    now default to the decision columns; 120+ is unchanged."""
    from club3090_cockpit.app import CatalogPane

    app, _, _ = make_app()
    async with app.run_test(size=(width, 30)) as pilot:
        await _settle(pilot)
        await _settle(pilot)
        pane = app.query_one("#catalog-pane", CatalogPane)
        assert pane._visible_columns() == expected


@pytest.mark.asyncio
async def test_catalog_narrow_default_never_overrides_a_saved_picker_choice():
    """The narrow default is a DEFAULT.  A user who has used the [|] picker
    keeps exactly what they chose, at every width."""
    from club3090_cockpit.app import CatalogPane

    app, _, _ = make_app()
    app.catalog_columns_pref = {
        "order": [k for k, _ in __import__(
            "club3090_cockpit.app", fromlist=["x"]
        )._CATALOG_COLUMNS],
        "hidden": ["ctx"],
    }
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        await _settle(pilot)
        pane = app.query_one("#catalog-pane", CatalogPane)
        visible = pane._visible_columns()
        assert "ctx" not in visible          # their choice honoured
        assert "provider" in visible         # narrow default NOT applied on top
