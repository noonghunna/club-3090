#!/usr/bin/env python3
"""Apply / verify the sglang-effort-thinking-budget hooks in an SGLang source tree.

  patch_sglang.py apply  --root <sglang package dir>   patch (all three or none), then verify
  patch_sglang.py verify --root <sglang package dir>   AST-check the applied hooks, change nothing
  patch_sglang.py state  --root <sglang package dir>   print applied | absent | partial

Three edits, each anchored on exact text that must occur EXACTLY ONCE (no fuzzy matching; a
missing or repeated anchor refuses, naming the anchor and README.md "Re-anchoring"):

  protocol      openai/protocol.py      declare `thinking_token_budget: Optional[int]` on
                                        ChatCompletionRequest (pydantic drops unknown keys there)
  chat-hook     openai/serving_chat.py  one call between _process_messages and to_sampling_params
  anthropic     anthropic/serving.py    one call in place of the "budget_tokens ... is not
                                        enforced" warning

Every inserted line carries MARKER. All anchors are checked before any file is written, and
each file is replaced atomically (temp sibling + os.replace), so a refusal leaves the tree as it
was. Exit codes: 0 ok, 1 refused / verify failed, 2 usage.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PATCH_ID = "sglang-effort-thinking-budget"
MARKER = "# club3090: sglang-effort-thinking-budget v1"
GLUE = "club3090_sglang_effort_budget"
TAG = f"[{PATCH_ID}]"


@dataclass(frozen=True)
class Edit:
    name: str
    path: str          # relative to the sglang package dir
    anchor: str        # exact text, must occur once
    replacement: str   # what the anchor becomes


PROTOCOL_ANCHOR = (
    "    # Custom logit processor for advanced sampling control\n"
    "    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None\n"
    "    custom_params: Optional[Dict] = None\n"
)
CHAT_ANCHOR = (
    "        processed_messages = self._process_messages(request, is_multimodal)\n"
    "        # Build sampling parameters\n"
    "        sampling_params = request.to_sampling_params(\n"
)
ANTHROPIC_ANCHOR = (
    "            if anthropic_request.thinking.budget_tokens is not None:\n"
    "                logger.warning(\n"
    '                    "Anthropic thinking.budget_tokens=%d is accepted for "\n'
    '                    "SDK compatibility but the local backend has no "\n'
    '                    "equivalent hard-cap knob — the budget is not enforced",\n'
    "                    anthropic_request.thinking.budget_tokens,\n"
    "                )\n"
)

EDITS = (
    Edit(
        "protocol",
        "srt/entrypoints/openai/protocol.py",
        PROTOCOL_ANCHOR,
        PROTOCOL_ANCHOR + f"    thinking_token_budget: Optional[int] = None  {MARKER}\n",
    ),
    Edit(
        "chat-hook",
        "srt/entrypoints/openai/serving_chat.py",
        CHAT_ANCHOR,
        CHAT_ANCHOR.replace(
            "        # Build sampling parameters\n",
            f'        __import__("{GLUE}").apply_chat_budget(self, request, processed_messages)  {MARKER}\n'
            "        # Build sampling parameters\n",
            1,
        ),
    ),
    Edit(
        "anthropic",
        "srt/entrypoints/anthropic/serving.py",
        ANTHROPIC_ANCHOR,
        "            if anthropic_request.thinking.budget_tokens is not None:\n"
        f'                __import__("{GLUE}").apply_anthropic_budget(chat_request, '
        f"anthropic_request.thinking.budget_tokens)  {MARKER}\n",
    ),
)


class Refused(Exception):
    pass


def _read(root: Path, edit: Edit) -> str:
    path = root / edit.path
    if not path.is_file():
        raise Refused(f"{edit.name}: {edit.path} not found under {root} — wrong SGLang tree?")
    return path.read_text(encoding="utf-8")


def state(root: Path) -> str:
    present = [MARKER in _read(root, e) for e in EDITS]
    if all(present):
        return "applied"
    if not any(present):
        return "absent"
    have = ", ".join(e.name for e, p in zip(EDITS, present) if p)
    return f"partial ({have})"


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".club3090", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def apply(root: Path) -> str:
    st = state(root)
    if st == "applied":
        verify(root)
        return "already applied"
    if st != "absent":
        raise Refused(f"{st}: a half-patched tree. Recreate the container (docker compose up "
                      "--force-recreate) so it starts from the stock image, then retry")
    texts = {}
    for e in EDITS:
        text = _read(root, e)
        n = text.count(e.anchor)
        if n != 1:
            first = e.anchor.splitlines()[0].strip()
            what = "is missing" if n == 0 else f"occurs {n} times"
            raise Refused(f"{e.name}: the anchor in {e.path} {what} (first line: {first!r}). "
                          "SGLang changed this code; re-anchor per README.md \"Re-anchoring\" — "
                          "never loosen the match")
        texts[e] = text.replace(e.anchor, e.replacement, 1)
    for e, text in texts.items():
        compile(text, e.path, "exec")          # a broken edit refuses before anything is written
    for e, text in texts.items():
        _write_atomic(root / e.path, text)
    verify(root)
    return "patched"


# ---------------------------------------------------------------------------
# AST verification — the hook sits at the right call site, not just "somewhere in the file".
# ---------------------------------------------------------------------------

def _find(tree: ast.AST, cls: str, fn: str | None = None) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            if fn is None:
                return node
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == fn:
                    return item
            raise Refused(f"{cls}.{fn} not found")
    raise Refused(f"class {cls} not found")


def _glue_call(node: ast.AST, attr: str) -> ast.Call | None:
    """`__import__("<GLUE>").<attr>(...)` -> the Call, else None."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr):
        return None
    inner = node.func.value
    ok = (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
          and inner.func.id == "__import__" and len(inner.args) == 1
          and isinstance(inner.args[0], ast.Constant) and inner.args[0].value == GLUE)
    return node if ok else None


def _calls_named(tree: ast.AST, attr: str) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _glue_call(n, attr)]


def _assign_calling(stmt: ast.stmt, target: str, attr: str) -> bool:
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == target
            and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == attr)


def _names(args) -> list:
    return [a.id if isinstance(a, ast.Name) else ast.unparse(a) for a in args]


def check_protocol(text: str) -> None:
    cls = _find(ast.parse(text), "ChatCompletionRequest")
    fields = [s for s in cls.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
              and s.target.id == "thinking_token_budget"]
    if len(fields) != 1:
        raise Refused(f"protocol: ChatCompletionRequest declares thinking_token_budget {len(fields)} times, want 1")
    f = fields[0]
    if ast.unparse(f.annotation) != "Optional[int]" or not (isinstance(f.value, ast.Constant) and f.value.value is None):
        raise Refused(f"protocol: thinking_token_budget is `{ast.unparse(f)}`, want `Optional[int] = None`")
    tree = ast.parse(text)
    elsewhere = [c.name for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name != "ChatCompletionRequest"
                 and any(isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                         and s.target.id == "thinking_token_budget" for s in c.body)]
    if elsewhere:
        raise Refused(f"protocol: thinking_token_budget also declared on {elsewhere}")


def check_chat(text: str) -> None:
    tree = ast.parse(text)
    if len(_calls_named(tree, "apply_chat_budget")) != 1:
        raise Refused("chat-hook: expected exactly one apply_chat_budget call in serving_chat.py")
    fn = _find(tree, "OpenAIServingChat", "_convert_to_internal_request")
    body = fn.body
    i_pm = [i for i, s in enumerate(body) if _assign_calling(s, "processed_messages", "_process_messages")]
    i_sp = [i for i, s in enumerate(body) if _assign_calling(s, "sampling_params", "to_sampling_params")]
    i_hook = [i for i, s in enumerate(body) if isinstance(s, ast.Expr) and _glue_call(s.value, "apply_chat_budget")]
    if len(i_pm) != 1 or len(i_sp) != 1 or len(i_hook) != 1:
        raise Refused(f"chat-hook: _convert_to_internal_request has {len(i_pm)} _process_messages, "
                      f"{len(i_hook)} hook and {len(i_sp)} to_sampling_params statements at top level, want 1 each")
    if not (i_pm[0] + 1 == i_hook[0] and i_hook[0] + 1 == i_sp[0]):
        raise Refused(f"chat-hook: order is _process_messages@{i_pm[0]} hook@{i_hook[0]} "
                      f"to_sampling_params@{i_sp[0]}, want three consecutive statements in that order")
    args = _names(body[i_hook[0]].value.args)
    if args != ["self", "request", "processed_messages"] or body[i_hook[0]].value.keywords:
        raise Refused(f"chat-hook: called with {args}, want (self, request, processed_messages)")


def check_anthropic(text: str) -> None:
    tree = ast.parse(text)
    if len(_calls_named(tree, "apply_anthropic_budget")) != 1:
        raise Refused("anthropic: expected exactly one apply_anthropic_budget call in anthropic/serving.py")
    if "the budget is not enforced" in text:
        raise Refused("anthropic: the 'budget is not enforced' warning is still there")
    fn = _find(tree, "AnthropicServing", "_convert_to_chat_completion_request")
    body = fn.body
    i_build = [i for i, s in enumerate(body) if isinstance(s, ast.Assign) and len(s.targets) == 1
               and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "chat_request"
               and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name)
               and s.value.func.id == "ChatCompletionRequest"]
    i_think = [i for i, s in enumerate(body) if isinstance(s, ast.If)
               and ast.unparse(s.test) == "anthropic_request.thinking is not None"]
    if len(i_build) != 1 or len(i_think) != 1 or not i_build[0] < i_think[0]:
        raise Refused("anthropic: want `chat_request = ChatCompletionRequest(...)` followed by one "
                      "`if anthropic_request.thinking is not None:` block")
    block = body[i_think[0]].body
    i_budget = [i for i, s in enumerate(block) if isinstance(s, ast.If)
                and ast.unparse(s.test) == "anthropic_request.thinking.budget_tokens is not None"]
    i_enable = [i for i, s in enumerate(block) if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                and isinstance(s.value.func, ast.Attribute) and s.value.func.attr == "apply_reasoning_enabled"]
    if len(i_budget) != 1 or len(i_enable) != 1 or not i_budget[0] < i_enable[0]:
        raise Refused("anthropic: want one `budget_tokens is not None` branch before apply_reasoning_enabled")
    branch = block[i_budget[0]].body
    if len(branch) != 1 or not (isinstance(branch[0], ast.Expr) and _glue_call(branch[0].value, "apply_anthropic_budget")):
        raise Refused("anthropic: the budget_tokens branch must be exactly the one apply_anthropic_budget call")
    args = _names(branch[0].value.args)
    if args != ["chat_request", "anthropic_request.thinking.budget_tokens"]:
        raise Refused(f"anthropic: called with {args}, want (chat_request, anthropic_request.thinking.budget_tokens)")


CHECKS = {"protocol": check_protocol, "chat-hook": check_chat, "anthropic": check_anthropic}


def verify(root: Path) -> None:
    for e in EDITS:
        text = _read(root, e)
        if text.count(MARKER) != 1:
            raise Refused(f"{e.name}: {e.path} carries the marker {text.count(MARKER)} times, want 1")
        CHECKS[e.name](text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="patch_sglang.py", description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=("apply", "verify", "state"))
    p.add_argument("--root", required=True, type=Path, help="the sglang package dir (…/python/sglang)")
    a = p.parse_args(argv)
    try:
        if a.cmd == "state":
            print(state(a.root))
        elif a.cmd == "verify":
            verify(a.root)
            print(f"{TAG} verify: 3/3 hooks at their call sites")
        else:
            print(f"{TAG} {apply(a.root)}: 3/3 hooks verified")
    except Refused as exc:
        print(f"{TAG} REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
