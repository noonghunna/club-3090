#!/usr/bin/env python3
"""Patch B `vllm-effort-budget`: the one-call hook for the reasoning budget chosen by effort.

    patch_effort_budget.py copy     copy the engine-neutral module + the vLLM glue next to vllm
    patch_effort_budget.py patch    insert the call (marker-gated, strict anchor), then --verify
    patch_effort_budget.py verify   AST-check the installed hook (also the selftest's check)

The hook is ONE call, inserted just before the only

                sampling_params = request.to_sampling_params(

line of vllm/entrypoints/openai/chat_completion/serving.py (OpenAIServingChat.
_create_chat_completion, the non-beam branch). Chat completions and Anthropic /v1/messages
(AnthropicServingMessages subclasses OpenAIServingChat and goes through
create_chat_completion) both pass it. Strict anchors: the marker present -> verify and no-op;
the anchor missing or not unique -> refuse (exit 1) with the README's re-anchor steps. No
fuzzy matching: a hook in the wrong place would boot and silently not budget anything.

Stdlib only. Never imports vllm (find_spec locates it without running it).
"""
from __future__ import annotations

import ast
import importlib.util
import os
import py_compile
import shutil
import sys
from pathlib import Path

TAG = "[vllm-effort-budget]"
HERE = Path(__file__).resolve().parent
README = "models/qwen3.8-27b/vllm/patches/vllm-effort-budget/README.md"
MODULE_SRC = Path("/etc/club3090/effort_budget.py")      # scripts/lib/effort_budget.py, mounted by the compose
GLUE_SRC = HERE / "club3090_effort_budget_vllm.py"
MODULE_NAME = "club3090_effort_budget"
GLUE_NAME = "club3090_effort_budget_vllm"
TARGET_REL = Path("entrypoints/openai/chat_completion/serving.py")
CLASS = "OpenAIServingChat"
FUNC = "_create_chat_completion"
KWARGS_METHOD = "_effective_chat_template_kwargs"
MARKER = "[club3090 vllm-effort-budget]"
INDENT = " " * 16
ANCHOR = INDENT + "sampling_params = request.to_sampling_params(\n"
INSERT = (
    f"{INDENT}# {MARKER} a budget from the effort map when the request set none. See {README}\n"
    f'{INDENT}__import__("{GLUE_NAME}").apply_effort_budget(request, self.{KWARGS_METHOD}(request))\n'
)


def die(msg: str) -> "None":
    print(f"{TAG} REFUSE: {msg}", file=sys.stderr)
    print(f"{TAG} Fix: re-anchor the hook for this vLLM version — {README} ('Re-anchoring at a pin bump').",
          file=sys.stderr)
    sys.exit(1)


def vllm_dir() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        die("the vllm package is not importable in this container")
    return Path(spec.origin).resolve().parent


def cmd_copy() -> int:
    """Copy both modules on EVERY boot: a restarted container keeps its site-packages, so a
    marker-gated copy would keep serving an old module after the repo's copy changed."""
    site = vllm_dir().parent
    if not MODULE_SRC.is_file():
        die(f"{MODULE_SRC} is not mounted (the compose mounts scripts/lib/effort_budget.py there)")
    for src, name in ((MODULE_SRC, MODULE_NAME), (GLUE_SRC, GLUE_NAME)):
        dst = site / f"{name}.py"
        tmp = dst.with_suffix(".py.club3090-tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        try:
            py_compile.compile(str(dst), doraise=True)
        except py_compile.PyCompileError as exc:
            die(f"{dst} does not compile: {exc}")
    return 0


# ---------------------------------------------------------------------------
# AST checks
# ---------------------------------------------------------------------------

def _method(tree: ast.Module, cls: str, name: str):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    return item
    return None


def _is_to_sampling_params(stmt: ast.stmt) -> bool:
    if not (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)):
        return False
    f = stmt.value.func
    return (isinstance(f, ast.Attribute) and f.attr == "to_sampling_params"
            and isinstance(f.value, ast.Name) and f.value.id == "request")


def _is_hook(stmt: ast.stmt) -> bool:
    """`__import__("club3090_effort_budget_vllm").apply_effort_budget(request, self.<kwargs>(request))`."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    call = stmt.value
    f = call.func
    if not (isinstance(f, ast.Attribute) and f.attr == "apply_effort_budget" and isinstance(f.value, ast.Call)):
        return False
    imp = f.value
    if not (isinstance(imp.func, ast.Name) and imp.func.id == "__import__" and len(imp.args) == 1
            and isinstance(imp.args[0], ast.Constant) and imp.args[0].value == GLUE_NAME):
        return False
    if len(call.args) != 2 or not (isinstance(call.args[0], ast.Name) and call.args[0].id == "request"):
        return False
    kw = call.args[1]
    return (isinstance(kw, ast.Call) and isinstance(kw.func, ast.Attribute) and kw.func.attr == KWARGS_METHOD
            and isinstance(kw.func.value, ast.Name) and kw.func.value.id == "self"
            and len(kw.args) == 1 and isinstance(kw.args[0], ast.Name) and kw.args[0].id == "request")


def _stmt_lists(node: ast.AST):
    for child in ast.walk(node):
        for field in ("body", "orelse", "finalbody", "handlers"):
            seq = getattr(child, field, None)
            if isinstance(seq, list) and seq and all(isinstance(s, ast.stmt) for s in seq):
                yield seq


def structure_errors(source: str, *, patched: bool) -> list[str]:
    """Why `source` is not a serving.py this patch can hook (patched=False) or a correctly
    hooked one (patched=True). Empty list = fine."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{TARGET_REL} does not parse: {exc}"]
    errs = []
    if _method(tree, CLASS, KWARGS_METHOD) is None:
        errs.append(f"{CLASS}.{KWARGS_METHOD} is gone — the hook passes vLLM's merged chat-template kwargs "
                    "from it (it is what decides whether the request thinks)")
    func = _method(tree, CLASS, FUNC)
    if func is None:
        return errs + [f"{CLASS}.{FUNC} is gone"]
    sites = [(seq, i) for seq in _stmt_lists(func) for i, s in enumerate(seq) if _is_to_sampling_params(s)]
    if len(sites) != 1:
        return errs + [f"{CLASS}.{FUNC} has {len(sites)} `sampling_params = request.to_sampling_params(` "
                       "statements, expected exactly 1"]
    hooks = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and _is_hook(n)]
    if patched:
        seq, i = sites[0]
        if len(hooks) != 1:
            errs.append(f"expected exactly one apply_effort_budget call in {TARGET_REL}, found {len(hooks)}")
        elif i == 0 or not _is_hook(seq[i - 1]):
            errs.append(f"the apply_effort_budget call is not the statement right before "
                        f"`sampling_params = request.to_sampling_params(` in {CLASS}.{FUNC}")
    elif hooks:
        errs.append("an apply_effort_budget call is present without the marker")
    return errs


def cmd_verify(path: Path | None = None) -> int:
    path = path or vllm_dir() / TARGET_REL
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        print(f"{TAG} VERIFY FAIL: {path} carries no {MARKER} marker (not patched)", file=sys.stderr)
        return 1
    errs = structure_errors(text, patched=True)
    for e in errs:
        print(f"{TAG} VERIFY FAIL: {e}", file=sys.stderr)
    if not errs:
        print(f"{TAG} verify ok: the hook is the statement right before request.to_sampling_params( "
              f"in {CLASS}.{FUNC}")
    return 1 if errs else 0


def cmd_patch() -> int:
    path = vllm_dir() / TARGET_REL
    if not path.is_file():
        die(f"{path} does not exist — vLLM moved the chat serving module")
    text = path.read_text(encoding="utf-8")
    budgets = (os.environ.get("CLUB3090_REASONING_EFFORT_BUDGETS") or "").strip() or "off (no map: the hook sets nothing)"
    if MARKER in text:
        errs = structure_errors(text, patched=True)
        if errs:
            die(f"{path} carries the marker but the hook is wrong: " + "; ".join(errs))
        print(f"{TAG} already applied (marker present, verified); modules refreshed; map={budgets}")
        return 0
    n = text.count(ANCHOR)
    if n != 1:
        die(f"anchor {ANCHOR.strip()!r} (16-space indent) found {n} times in {path}, expected exactly 1")
    errs = structure_errors(text, patched=False)
    if errs:
        die("; ".join(errs))
    new = text.replace(ANCHOR, INSERT + ANCHOR, 1)
    errs = structure_errors(new, patched=True)
    if errs:
        die("the patched file would be wrong: " + "; ".join(errs))
    tmp = path.with_suffix(".py.club3090-tmp")
    tmp.write_text(new, encoding="utf-8")
    try:
        py_compile.compile(str(tmp), doraise=True)
    except py_compile.PyCompileError as exc:
        tmp.unlink(missing_ok=True)
        die(f"the patched file does not compile: {exc}")
    os.replace(tmp, path)
    print(f"{TAG} applied: {CLASS}.{FUNC} sets thinking_token_budget from the effort map when the "
          f"request set none (chat completions + /v1/messages); map={budgets}")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "copy":
        return cmd_copy()
    if cmd == "patch":
        return cmd_patch()
    if cmd == "verify":
        return cmd_verify(Path(argv[1]) if len(argv) > 1 else None)
    print(f"usage: {Path(__file__).name} copy|patch|verify [serving.py]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
