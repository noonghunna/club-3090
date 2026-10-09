"""club-3090 SGLang glue for the reasoning budget chosen by effort (patch sglang-effort-thinking-budget).

install.sh copies this file into the image's site-packages, next to `club3090_effort_budget` (a copy
of the repo's scripts/lib/effort_budget.py, which holds ALL the effort -> budget logic). Each patched
SGLang call site makes exactly ONE call into this module, so re-anchoring on an engine upgrade means
moving one line, never porting logic:

  apply_chat_budget(serving_chat, request, processed_messages)
      openai/serving_chat.py::_convert_to_internal_request, between `_process_messages` and
      `to_sampling_params` (chat completions AND /v1/messages, which converts through it).
      Precedence: custom_params.thinking_budget (SGLang's own field) > thinking_token_budget (the
      vLLM-named alias, declared on ChatCompletionRequest by this patch) > the map budget for the
      effort in use. A request with thinking off gets nothing (a budget would be inert there).
  apply_anthropic_budget(chat_request, budget_tokens)
      anthropic/serving.py::_convert_to_chat_completion_request, in place of SGLang's "budget_tokens
      is not enforced" warning: thinking.budget_tokens -> custom_params.thinking_budget.

Enforcement is SGLang's own strict thinking (--enable-strict-thinking): the grammar manager reads
custom_params.thinking_budget per request (constrained/grammar_manager.py). SGLANG_MAX_THINK_TOKENS
(exported by `effort_budget.py shell-env --engine sglang`) is the floor for anything that reaches no
hook (the Responses API, the native /generate endpoint).

A negative budget is passed through unchanged: SGLang treats any budget below 0 as unlimited.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import club3090_effort_budget as _eb

PATCH_ID = "sglang-effort-thinking-budget"
BUDGET_KEY = "thinking_budget"            # SGLang's custom_params key (grammar_manager.py)
ALIAS_FIELD = "thinking_token_budget"     # vLLM's request field name

# Under the sglang.* hierarchy so SGLang's own logging config formats and shows it.
logger = logging.getLogger("sglang.club3090_effort_budget")

_budgets: Optional[dict] = None
_strict: Optional[str] = None             # "on" | "off" | "unknown"; None = not looked up yet
_warned_unenforced = False


def budgets() -> dict:
    """The map from CLUB3090_REASONING_EFFORT_BUDGETS, parsed once ({} = no map). install.sh has
    already parsed it at boot, so a malformed map stops the container instead of failing here."""
    global _budgets
    if _budgets is None:
        _budgets = _eb.budgets_from_env()
    return _budgets


def reload(environ: Optional[Mapping[str, str]] = None) -> dict:
    """Re-read the map (and forget the strict-thinking lookup). For tests."""
    global _budgets, _strict, _warned_unenforced
    _budgets = _eb.budgets_from_env(environ)
    _strict = None
    _warned_unenforced = False
    return _budgets


def _strict_thinking() -> str:
    """Whether this server enforces budgets at all. Only used to keep log lines truthful."""
    global _strict
    if _strict is None:
        try:
            from sglang.srt.runtime_context import get_serving

            _strict = "on" if get_serving().enable_strict_thinking else "off"
        except Exception:  # an SGLang without this accessor: say nothing rather than guess
            _strict = "unknown"
    return _strict


def _has_explicit(custom_params: Any) -> bool:
    return isinstance(custom_params, dict) and custom_params.get(BUDGET_KEY) is not None


def _set_budget(obj: Any, budget: Any) -> int:
    # A new dict (never mutate the caller's), and a real int: the grammar manager ignores non-ints.
    params = dict(obj.custom_params or {})
    params[BUDGET_KEY] = int(budget)
    obj.custom_params = params
    return params[BUDGET_KEY]


def apply_chat_budget(serving_chat: Any, request: Any, processed_messages: Any) -> Optional[int]:
    """Hook 1. Sets request.custom_params.thinking_budget; returns the budget set, or None when the
    request is left alone (thinking off, its own budget, or no map budget for its effort)."""
    if not getattr(processed_messages, "require_reasoning", False):
        return None
    if _has_explicit(request.custom_params):
        return None
    alias = getattr(request, ALIAS_FIELD, None)
    if alias is not None:
        return _set_budget(request, alias)
    budget = _eb.budget_for(
        budgets(),
        request.reasoning_effort,
        request.chat_template_kwargs,
        getattr(serving_chat, "default_chat_template_kwargs", None),
    )
    if budget is None:
        return None
    global _warned_unenforced
    if not _warned_unenforced and _strict_thinking() == "off":
        _warned_unenforced = True
        logger.warning(
            "club-3090 %s: effort budgets are set but this server runs without "
            "--enable-strict-thinking, so SGLang will not enforce them",
            PATCH_ID,
        )
    return _set_budget(request, budget)


def apply_anthropic_budget(chat_request: Any, budget_tokens: Any) -> Optional[int]:
    """Hook 2. Maps /v1/messages thinking.budget_tokens onto custom_params.thinking_budget, where
    hook 1 then sees an explicit budget and leaves it alone (explicit wins over the map)."""
    if budget_tokens is None or _has_explicit(chat_request.custom_params):
        return None
    budget = _set_budget(chat_request, budget_tokens)
    if _strict_thinking() == "off":
        logger.warning(
            "Anthropic thinking.budget_tokens=%d is mapped to custom_params.thinking_budget but "
            "NOT enforced: the server runs without --enable-strict-thinking (club-3090 %s)",
            budget,
            PATCH_ID,
        )
    else:
        logger.debug(
            "Anthropic thinking.budget_tokens=%d -> custom_params.thinking_budget (club-3090 %s)",
            budget,
            PATCH_ID,
        )
    return budget
