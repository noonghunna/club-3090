#!/usr/bin/env python3
"""Reasoning budget chosen by effort — the engine-neutral half.

A server-side default for a thinking model's reasoning budget, chosen from the effort the
request asks for: `low` gets a smaller budget than `xhigh`. vLLM and SGLang both enforce a
budget per request (vLLM `thinking_token_budget`, SGLang strict thinking's
`custom_params.thinking_budget`), but neither picks one from the effort, so a client that
sends only `reasoning_effort` — which is all the OpenAI API lets most coding agents send —
is unbounded. This module holds ALL the logic so the engine patches stay one inserted call
each and survive engine upgrades (re-anchoring = moving one line):

  * `shell-env` — run by the compose entrypoint. Reads the knobs, prints `export` lines on
    stdout and ONE readback line on stderr. Invalid knobs exit non-zero; it never falls back
    silently. Use it as
        _eb="$(python3 /etc/club3090/effort_budget.py shell-env ...)" || exit 1; eval "$_eb"
    never as a bare `eval "$(...)"` (an empty output would boot with no budgets, no error).
  * the library API (`budgets_from_env`, `budget_for`) — imported by the engine hooks.
  * `readback-budget` — used by the quality wrapper to learn the budget a server applies.

Precedence (both engines): an explicit per-request budget > map[effort in use] > the floor
(the budget for the compose's default effort, applied by the engines' own default paths).
The effort in use: the request's top-level `reasoning_effort`, else
`chat_template_kwargs.reasoning_effort`, else the server's default chat-template kwargs
(`reasoning_effort`, or `default_reasoning_effort` on our SGLang composes). Thinking off
(`enable_thinking` false, or effort `none`) gets no map budget.

Stdlib only; Python >= 3.8.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping, Optional

READBACK_TAG = "[effort-budget]"
READBACK_VERSION = "v1"

LEVELS = ("low", "medium", "xhigh")
# The Qwen3.8 template's own clamps (models/qwen3.8-27b/vllm/patches/qwen38-reasoning-effort-template).
ALIASES = {"minimal": "low", "high": "xhigh", "max": "xhigh"}

MAP_ENV = "CLUB3090_REASONING_EFFORT_BUDGETS"
FLOOR_ENV = {"vllm": "CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET", "sglang": "SGLANG_MAX_THINK_TOKENS"}
LEVEL_KNOBS = {"low": "THINKING_BUDGET_LOW", "medium": "THINKING_BUDGET_MEDIUM", "xhigh": "THINKING_BUDGET_XHIGH"}
OFF_KNOB = "THINKING_BUDGETS"
OFF_VALUES = ("off", "0", "false", "no", "none", "disabled")
SERVER_EFFORT_KEYS = ("reasoning_effort", "default_reasoning_effort")


class BudgetConfigError(ValueError):
    """A knob or map value that must not be used: the boot has to stop, not guess."""


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise BudgetConfigError(f"{name}={value!r} is not a token count")
    if isinstance(value, int):
        n = value
    else:
        text = str(value).strip()
        if not text.isdigit():
            raise BudgetConfigError(f"{name}={value!r} is not a non-negative integer")
        n = int(text)
    if n < 0:
        raise BudgetConfigError(f"{name}={value!r} is negative")
    return n


def _with_aliases(levels: Mapping[str, int]) -> dict:
    out = dict(levels)
    for alias, target in ALIASES.items():
        if target in levels:
            out[alias] = levels[target]
    return out


def _canonical_effort(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def build_map(env: Mapping[str, str], defaults: Mapping[str, Any]) -> Optional[dict]:
    """The budget map from the knobs (env wins over the compose's defaults), or None when off.

    An empty env value counts as unset. Raises BudgetConfigError on a bad value.
    """
    off = (env.get(OFF_KNOB) or "").strip().lower()
    if off in OFF_VALUES:
        return None
    if off and off not in ("on", "1", "true", "yes"):
        raise BudgetConfigError(f"{OFF_KNOB}={env.get(OFF_KNOB)!r}: use 'off' (or unset)")
    levels = {}
    for level in LEVELS:
        knob = LEVEL_KNOBS[level]
        raw = (env.get(knob) or "").strip()
        if raw:
            levels[level] = _non_negative_int(knob, raw)
        elif defaults.get(level) is not None:
            levels[level] = _non_negative_int(f"default {level}", defaults[level])
        else:
            raise BudgetConfigError(f"no budget for '{level}': set {knob} or pass --{level}")
    return _with_aliases(levels)


def parse_map(text: Optional[str]) -> dict:
    """Parse a map exported by shell-env. Empty/None → {} (no map). Raises on a malformed one."""
    if text is None or not str(text).strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise BudgetConfigError(f"{MAP_ENV} is not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise BudgetConfigError(f"{MAP_ENV} must be a JSON object, got {type(data).__name__}")
    return {str(k).strip().lower(): _non_negative_int(f"{MAP_ENV}[{k}]", v) for k, v in data.items()}


def budgets_from_env(environ: Optional[Mapping[str, str]] = None) -> dict:
    """The map the engine hooks use. Strict: a malformed map raises (the install check runs
    this at boot, so a bad value stops the server instead of 500-ing every request)."""
    return parse_map((environ if environ is not None else os.environ).get(MAP_ENV))


def resolve_effort(top_level: Any = None,
                   template_kwargs: Optional[Mapping[str, Any]] = None,
                   server_kwargs: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """The effort a request runs at: its own top-level field, its chat_template_kwargs, then the
    server's default chat-template kwargs. Lower-cased; aliases are NOT folded (the map has them)."""
    candidates = [top_level, (template_kwargs or {}).get("reasoning_effort")]
    candidates += [(server_kwargs or {}).get(k) for k in SERVER_EFFORT_KEYS]
    for value in candidates:
        effort = _canonical_effort(value)
        if effort is not None:
            return effort
    return None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def thinking_enabled(effort: Optional[str],
                     template_kwargs: Optional[Mapping[str, Any]] = None,
                     server_kwargs: Optional[Mapping[str, Any]] = None) -> bool:
    """False when the request runs with thinking off: effort `none`, or `enable_thinking` false
    (the request's own value wins over the server default). Unknown → True (thinking models
    think by default; a budget on a non-thinking turn is inert anyway)."""
    if effort == "none":
        return False
    for kwargs in (template_kwargs, server_kwargs):
        flag = _as_bool((kwargs or {}).get("enable_thinking"))
        if flag is not None:
            return flag
    return True


def budget_for(budgets: Mapping[str, int],
               top_level: Any = None,
               template_kwargs: Optional[Mapping[str, Any]] = None,
               server_kwargs: Optional[Mapping[str, Any]] = None) -> Optional[int]:
    """The map budget for one request, or None (no map, thinking off, or an unmapped effort).
    Callers apply it only when the request did not set its own budget."""
    if not budgets:
        return None
    effort = resolve_effort(top_level, template_kwargs, server_kwargs)
    if effort is None or not thinking_enabled(effort, template_kwargs, server_kwargs):
        return None
    return budgets.get(effort)


def readback_line(budgets: Optional[Mapping[str, int]], default_effort: Optional[str] = None,
                  floor: Optional[int] = None, engine: Optional[str] = None) -> str:
    head = f"{READBACK_TAG} {READBACK_VERSION}"
    if budgets is None:
        return f"{head} off" + (f" engine={engine}" if engine else "")
    parts = [head, "map=" + json.dumps(dict(budgets), sort_keys=True, separators=(",", ":"))]
    parts.append(f"default_effort={default_effort}")
    parts.append(f"floor={floor}")
    if engine:
        parts.append(f"engine={engine}")
    return " ".join(parts)


def parse_readback(text: str) -> Optional[dict]:
    """The LAST `[effort-budget] v1` line in `text` (docker logs span restarts) as
    {"off": bool, "map": {...}, "default_effort": str|None, "floor": int|None}; None if absent.
    A different version tag is ignored, so a future format can't be misread."""
    found = None
    for line in text.splitlines():
        idx = line.find(f"{READBACK_TAG} ")
        if idx < 0:
            continue
        fields = line[idx + len(READBACK_TAG):].split()
        if not fields or fields[0] != READBACK_VERSION:
            continue
        rest = fields[1:]
        if rest and rest[0] == "off":
            found = {"off": True, "map": {}, "default_effort": None, "floor": None}
            continue
        info = {"off": False, "map": {}, "default_effort": None, "floor": None}
        for field in rest:
            key, _, value = field.partition("=")
            if key == "map":
                info["map"] = parse_map(value)
            elif key == "default_effort":
                info["default_effort"] = None if value in ("", "None") else value
            elif key == "floor":
                info["floor"] = None if value in ("", "None") else _non_negative_int("floor", value)
        found = info
    return found


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def cmd_shell_env(args: argparse.Namespace) -> int:
    env = os.environ
    defaults = {"low": args.low, "medium": args.medium, "xhigh": args.xhigh}
    budgets = build_map(env, defaults)
    floor_env = FLOOR_ENV[args.engine]
    if budgets is None:
        # Non-empty stdout keeps the entrypoint's `|| exit 1` + eval pattern honest.
        print(": effort-budget off")
        print(f"unset {MAP_ENV} {floor_env}")
        print(readback_line(None, engine=args.engine), file=sys.stderr)
        return 0
    default_effort = _canonical_effort(args.default_effort or env.get("REASONING_EFFORT"))
    if default_effort is None:
        raise BudgetConfigError("no default effort: pass --default-effort (the compose's REASONING_EFFORT)")
    if default_effort not in budgets:
        raise BudgetConfigError(f"default effort {default_effort!r} has no budget (map: {sorted(budgets)})")
    floor = budgets[default_effort]
    payload = json.dumps(budgets, sort_keys=True, separators=(",", ":"))
    print(f"export {MAP_ENV}={_shell_quote(payload)}")
    print(f"export {floor_env}={floor}")
    print(readback_line(budgets, default_effort, floor, engine=args.engine), file=sys.stderr)
    return 0


def cmd_readback_budget(args: argparse.Namespace) -> int:
    text = sys.stdin.read() if args.line == "-" else args.line
    info = parse_readback(text)
    if info is None:
        print("unknown")
        return 2
    if info["off"]:
        print("off")
        return 0
    effort = _canonical_effort(args.effort) or info["default_effort"]
    if effort == "none":
        print("none")  # thinking off: no budget applies
        return 0
    budget = info["map"].get(effort) if effort else None
    print(budget if budget is not None else info["floor"] if info["floor"] is not None else "none")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="effort_budget.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("shell-env", help="compose entrypoint: export the map + floor, print the readback line")
    p.add_argument("--engine", choices=sorted(FLOOR_ENV), required=True)
    p.add_argument("--default-effort", help="the compose's default effort (REASONING_EFFORT)")
    for level in LEVELS:
        p.add_argument(f"--{level}", help=f"default budget for '{level}' ({LEVEL_KNOBS[level]} overrides)")
    p.set_defaults(func=cmd_shell_env)
    r = sub.add_parser("readback-budget", help="the budget a server applies, from its boot log")
    r.add_argument("--line", required=True, help="log text containing the readback line, or - for stdin")
    r.add_argument("--effort", help="the run's effort (default: the server's default effort)")
    r.set_defaults(func=cmd_readback_budget)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BudgetConfigError as exc:
        print(f"{READBACK_TAG} ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
