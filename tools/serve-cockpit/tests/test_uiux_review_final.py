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
