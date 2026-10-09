#!/usr/bin/env python3
"""Patch D `vllm-default-thinking-budget`: a server-side FLOOR for thinking_token_budget.

    patch_default_thinking_budget.py patch    validate the env, insert the floor (marker-gated), verify
    patch_default_thinking_budget.py verify   AST-check the installed floor (also the selftest's check)

vLLM honours `thinking_token_budget` only per request, and --override-generation-config
cannot set it (get_diff_sampling_param's allowlist excludes it). This makes
SamplingParams.__post_init__ fall back to $CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET when the
request set none, so EVERY entrypoint gets a budget: the Responses API, completions,
/v1/chat/completions/batch, and anything else that builds SamplingParams without passing
through patch B's hook. The compose's effort_budget.py shell-env exports the floor as the
budget of the compose's default effort (REASONING_EFFORT). An explicit request budget wins.

Ported from the bucko local layer's default-thinking-budget/apply.py (same anchor), with:
  * strict install-time validation (a non-integer floor refuses the boot here instead of
    raising in every SamplingParams());
  * no floor for trace-replay requests (`trace_decode_token_ids` refuses any budget);
  * an AST verify of the result, and refusal when the marker is present but wrong.

Strict anchor: the bucko three-line `self.thinking_token_budget = validate_thinking_token_budget(`
statement, exactly once in vllm/sampling_params.py. Missing or not unique -> exit 1 with the
README's re-anchor steps. Stdlib only; never imports vllm.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import py_compile
import sys
from pathlib import Path

TAG = "[vllm-default-thinking-budget]"
README = "models/qwen3.8-27b/vllm/patches/vllm-default-thinking-budget/README.md"
ENV = "CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET"
TARGET_REL = Path("sampling_params.py")
CLASS = "SamplingParams"
FUNC = "__post_init__"
MARKER = "[club3090 vllm-default-thinking-budget]"
ANCHOR = ("        self.thinking_token_budget = validate_thinking_token_budget(\n"
          "            self.thinking_token_budget\n"
          "        )\n")
INSERT = (
    f"        # {MARKER} server-side floor when the request set none. See {README}\n"
    "        if self.thinking_token_budget is None and not self.trace_decode_token_ids:\n"
    "            import os as _club3090_os\n"
    f"            _club3090_floor = _club3090_os.environ.get(\"{ENV}\", \"\").strip()\n"
    "            if _club3090_floor:\n"
    "                self.thinking_token_budget = validate_thinking_token_budget(int(_club3090_floor))\n"
)


def die(msg: str) -> "None":
    print(f"{TAG} REFUSE: {msg}", file=sys.stderr)
    print(f"{TAG} Fix: re-anchor the floor for this vLLM version — {README} ('Re-anchoring at a pin bump').",
          file=sys.stderr)
    sys.exit(1)


def vllm_dir() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        die("the vllm package is not importable in this container")
    return Path(spec.origin).resolve().parent


def floor_from_env() -> int | None:
    raw = (os.environ.get(ENV) or "").strip()
    if not raw:
        return None
    if not raw.isdigit() or not raw.isascii():
        print(f"{TAG} REFUSE: {ENV}={raw!r} is not a non-negative integer "
              "(effort_budget.py shell-env exports it; unset it for no floor)", file=sys.stderr)
        sys.exit(1)
    return int(raw)


# ---------------------------------------------------------------------------
# AST checks
# ---------------------------------------------------------------------------

def _method(tree, cls, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    return None


def _attr(node, obj, attr):
    return (isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Name) and node.value.id == obj)


def _is_validate(stmt) -> bool:
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and _attr(stmt.targets[0], "self", "thinking_token_budget")
            and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "validate_thinking_token_budget")


def _is_floor(stmt) -> bool:
    """`if self.thinking_token_budget is None and not self.trace_decode_token_ids:` whose
    body reads ENV and assigns self.thinking_token_budget."""
    if not (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.BoolOp) and isinstance(stmt.test.op, ast.And)
            and len(stmt.test.values) == 2):
        return False
    a, b = stmt.test.values
    if not (isinstance(a, ast.Compare) and _attr(a.left, "self", "thinking_token_budget")
            and isinstance(a.ops[0], ast.Is) and isinstance(a.comparators[0], ast.Constant)
            and a.comparators[0].value is None):
        return False
    if not (isinstance(b, ast.UnaryOp) and isinstance(b.op, ast.Not) and _attr(b.operand, "self", "trace_decode_token_ids")):
        return False
    body = ast.dump(ast.Module(body=stmt.body, type_ignores=[]))
    return ENV in body and "thinking_token_budget" in body


def structure_errors(source: str, *, patched: bool) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{TARGET_REL} does not parse: {exc}"]
    func = _method(tree, CLASS, FUNC)
    if func is None:
        return [f"{CLASS}.{FUNC} is gone"]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS)
    if not any(isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
               and s.target.id == "trace_decode_token_ids" for s in cls.body):
        return [f"{CLASS} has no trace_decode_token_ids field any more — the floor's guard reads it"]
    # The anchor statement sits directly in __post_init__'s body; the floor's own
    # validate_thinking_token_budget(int(...)) assignment is nested in its `if`, so count
    # only top-level statements.
    sites = [(func.body, i) for i, s in enumerate(func.body) if _is_validate(s)]
    if len(sites) != 1:
        return [f"{CLASS}.{FUNC} has {len(sites)} top-level `self.thinking_token_budget = "
                "validate_thinking_token_budget(...)` statements, expected exactly 1"]
    floors = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and _is_floor(n)]
    seq, i = sites[0]
    if patched:
        if len(floors) != 1:
            return [f"expected exactly one floor block in {TARGET_REL}, found {len(floors)}"]
        if i + 1 >= len(seq) or not _is_floor(seq[i + 1]):
            return [f"the floor is not the statement right after the validate_thinking_token_budget call in {CLASS}.{FUNC}"]
    elif floors:
        return ["a floor block is present without the marker"]
    return []


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
        print(f"{TAG} verify ok: the floor is the statement right after validate_thinking_token_budget "
              f"in {CLASS}.{FUNC}")
    return 1 if errs else 0


def cmd_patch() -> int:
    floor = floor_from_env()          # refuses a bad value before touching anything
    shown = f"floor={floor}" if floor is not None else f"floor unset ({ENV} empty: requests without a budget stay unbudgeted)"
    path = vllm_dir() / TARGET_REL
    if not path.is_file():
        die(f"{path} does not exist")
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        errs = structure_errors(text, patched=True)
        if errs:
            die(f"{path} carries the marker but the floor is wrong: " + "; ".join(errs))
        print(f"{TAG} already applied (marker present, verified); {shown}")
        return 0
    n = text.count(ANCHOR)
    if n != 1:
        die(f"anchor `self.thinking_token_budget = validate_thinking_token_budget(` (3-line, 8-space indent) "
            f"found {n} times in {path}, expected exactly 1")
    errs = structure_errors(text, patched=False)
    if errs:
        die("; ".join(errs))
    new = text.replace(ANCHOR, ANCHOR + INSERT, 1)
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
    print(f"{TAG} applied: SamplingParams without a thinking_token_budget get ${ENV}; {shown}")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "patch":
        return cmd_patch()
    if cmd == "verify":
        return cmd_verify(Path(argv[1]) if len(argv) > 1 else None)
    print(f"usage: {Path(__file__).name} patch|verify [sampling_params.py]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
