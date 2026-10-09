"""vLLM glue for the reasoning budget chosen by effort (patch `vllm-effort-budget`).

install.sh copies this file next to the `vllm` package as `club3090_effort_budget_vllm`
and inserts ONE call to `apply_effort_budget` in OpenAIServingChat._create_chat_completion,
just before `request.to_sampling_params(`. All the policy lives in the engine-neutral
module (`club3090_effort_budget`, copied from scripts/lib/effort_budget.py); this file
only says what vLLM's view of the request is.

Why the effective kwargs and not the request's own fields: vLLM decides whether a request
thinks from the MERGED chat-template kwargs. `ChatCompletionRequest.build_chat_params`
copies a top-level `reasoning_effort` into the kwargs and, when the request's own kwargs
do not say, sets `enable_thinking = (effort != "none")`; `with_defaults` then fills the
rest from `--default-chat-template-kwargs`. So on a server booted with ENABLE_THINKING=false
a request that sends `reasoning_effort: xhigh` DOES think. Reading the request fields and
the server defaults separately would see the server's `enable_thinking: false` and give
that request no budget. `OpenAIServingChat._effective_chat_template_kwargs(request)` is
exactly what the chat template renders with, so the budget follows the prompt.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import club3090_effort_budget as _eb

# Parsed once per process. Strict: a malformed map raises here, and install.sh imports
# this module at boot with the same environment, so a bad map stops the boot instead of
# failing every request.
BUDGETS = _eb.budgets_from_env()


def topk_off_sampled(temperature: Any, top_p: Any, top_k: Any) -> bool:
    """True for a request vllm#58231 breaks once a budget fires: sampled (not greedy), top_p < 1
    and top-k explicitly off (0 or -1). Under spec decode the forced reasoning end then comes out
    as token 0 repeated to max_tokens. Such a request gets NO default budget — it reasons
    unbounded, as on stock vLLM — until the fix is in the pinned image (drop this guard then).
    A request that leaves top_k unset gets the server's default top_k (the composes set 20),
    and top_p unset counts as < 1 (the composes' samplers use 0.95)."""
    if top_k is None or top_k > 0:
        return False
    if temperature is not None and temperature < 1e-5:
        return False                       # greedy: vLLM forces top_p=1 / top_k off, unaffected
    return top_p is None or top_p < 1.0


def apply_effort_budget(request: Any,
                        effective_chat_template_kwargs: Optional[Mapping[str, Any]]) -> Optional[int]:
    """Set `request.thinking_token_budget` from the map when the request has none.

    An explicit budget always wins: the client's own `thinking_token_budget`, and
    `/v1/messages` `thinking.budget_tokens` (AnthropicServingMessages sets the field before
    it calls create_chat_completion). Returns the budget it set, or None.
    """
    if getattr(request, "thinking_token_budget", None) is not None:
        return None
    if topk_off_sampled(getattr(request, "temperature", None), getattr(request, "top_p", None),
                        getattr(request, "top_k", None)):
        return None
    budget = _eb.budget_for(BUDGETS, None, effective_chat_template_kwargs, None)
    if budget is not None:
        request.thinking_token_budget = budget
    return budget
