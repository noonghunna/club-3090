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
    """Yield (lineno, text) for every non-docstring string literal, with
    f-string expression slots collapsed to \\x00 so a tag split across an
    f-string boundary is still seen as one tag."""
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
