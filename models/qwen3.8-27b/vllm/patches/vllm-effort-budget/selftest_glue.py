#!/usr/bin/env python3
"""selftest.sh's in-image check of the glue against vLLM's REAL request objects.

Runs inside the pinned image (no GPU) after install.sh. For every case it builds a real
ChatCompletionRequest, resolves vLLM's own merged chat-template kwargs with the real
OpenAIServingChat._effective_chat_template_kwargs (on a stand-in `self` carrying only the
three attributes that method reads), asserts that resolution is what the case claims
(effort + thinking), then runs the installed glue exactly as the hook calls it and checks
the budget that reaches SamplingParams through the real to_sampling_params().
"""
from __future__ import annotations

import importlib
import json
import os
import sys

MAP = {"low": 64, "medium": 128, "xhigh": 256, "minimal": 64, "high": 256, "max": 256}
os.environ["CLUB3090_REASONING_EFFORT_BUDGETS"] = json.dumps(MAP)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat  # noqa: E402

import club3090_effort_budget_vllm as glue  # noqa: E402

THINK_ON = {"enable_thinking": True, "reasoning_effort": "low"}     # the composes' default kwargs
THINK_OFF = {"enable_thinking": False, "reasoning_effort": "low"}   # ENABLE_THINKING=false
MSG = [{"role": "user", "content": "hi"}]
fails = 0


class FakeServing:
    chat_template = None
    chat_template_content_format = "auto"

    def __init__(self, defaults):
        self.default_chat_template_kwargs = defaults


def check(label, server, req_kwargs, want_budget, want_effort=None, want_thinking=None):
    global fails
    req = ChatCompletionRequest(messages=MSG, **req_kwargs)
    eff = OpenAIServingChat._effective_chat_template_kwargs(FakeServing(server), req)
    problems = []
    if want_thinking is not None and bool(eff.get("enable_thinking", True)) != want_thinking:
        problems.append(f"vLLM resolves enable_thinking={eff.get('enable_thinking')!r}, case says {want_thinking}")
    if want_effort is not None and eff.get("reasoning_effort") != want_effort:
        problems.append(f"vLLM resolves reasoning_effort={eff.get('reasoning_effort')!r}, case says {want_effort!r}")
    glue.apply_effort_budget(req, eff)            # exactly the hook's call
    sp = req.to_sampling_params(1024, {})
    if sp.thinking_token_budget != want_budget:
        problems.append(f"SamplingParams.thinking_token_budget={sp.thinking_token_budget!r}, want {want_budget!r}")
    print(("  ✓ " if not problems else "  ✗ ") + label + ("" if not problems else ": " + "; ".join(problems)))
    fails += bool(problems)


print("--- glue vs vLLM's own request resolution (map low 64 / medium 128 / xhigh 256) ---")
check("top-level effort xhigh -> 256", THINK_ON, {"reasoning_effort": "xhigh"}, 256, "xhigh", True)
check("kwargs-only effort medium -> 128", THINK_ON, {"chat_template_kwargs": {"reasoning_effort": "medium"}},
      128, "medium", True)
check("top-level wins over kwargs (low vs xhigh) -> 64", THINK_ON,
      {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "xhigh"}}, 64, "low", True)
check("no effort -> the server default (low) -> 64", THINK_ON, {}, 64, "low", True)
check("alias high -> xhigh's 256", THINK_ON, {"reasoning_effort": "high"}, 256, "high", True)
check("alias max -> xhigh's 256", THINK_ON, {"reasoning_effort": "max"}, 256, "max", True)
check("alias minimal (kwargs) -> low's 64", THINK_ON, {"chat_template_kwargs": {"reasoning_effort": "minimal"}},
      64, "minimal", True)
check("thinking off (kwargs enable_thinking=false) -> no map budget", THINK_ON,
      {"chat_template_kwargs": {"enable_thinking": False}}, None, None, False)
check("effort none -> thinking off -> no map budget", THINK_ON, {"reasoning_effort": "none"}, None, "none", False)
check("explicit thinking_token_budget wins over the map", THINK_ON,
      {"reasoning_effort": "xhigh", "thinking_token_budget": 1000}, 1000)
check("explicit budget 0 is kept (0 is a budget, not unset)", THINK_ON,
      {"reasoning_effort": "xhigh", "thinking_token_budget": 0}, 0)
check("ENABLE_THINKING=false server + top-level xhigh: vLLM turns thinking ON -> 256", THINK_OFF,
      {"reasoning_effort": "xhigh"}, 256, "xhigh", True)
check("ENABLE_THINKING=false server + no effort -> thinking off -> no map budget", THINK_OFF, {}, None, "low", False)
check("ENABLE_THINKING=false server + kwargs enable_thinking=true -> server default effort 64", THINK_OFF,
      {"chat_template_kwargs": {"enable_thinking": True}}, 64, "low", True)
# Documented limitation (README): the request validator folds -1 ("unlimited") to None at
# parse time, so neither this hook nor the floor can tell it from an absent budget.
check("thinking_token_budget=-1 is parsed to None -> the map applies (documented)", THINK_ON,
      {"reasoning_effort": "xhigh", "thinking_token_budget": -1}, 256)

print("--- /v1/messages: AnthropicServingMessages sets the budget BEFORE the hook ---")
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest  # noqa: E402
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages  # noqa: E402

if not issubclass(AnthropicServingMessages, OpenAIServingChat):
    print("  ✗ AnthropicServingMessages no longer subclasses OpenAIServingChat")
    fails += 1
if "_create_chat_completion" in vars(AnthropicServingMessages) or "create_chat_completion" in vars(AnthropicServingMessages):
    print("  ✗ AnthropicServingMessages overrides (_)create_chat_completion — /v1/messages may bypass the hook")
    fails += 1


def anth(label, extra, want_budget):
    global fails
    a = AnthropicMessagesRequest(model="m", max_tokens=4096, messages=[{"role": "user", "content": "hi"}], **extra)
    req = AnthropicServingMessages.to_chat_completion_request(a)
    eff = OpenAIServingChat._effective_chat_template_kwargs(FakeServing(THINK_ON), req)
    glue.apply_effort_budget(req, eff)
    got = req.to_sampling_params(4096, {}).thinking_token_budget
    ok = got == want_budget
    print(("  ✓ " if ok else "  ✗ ") + label + ("" if ok else f": got {got!r}, want {want_budget!r}"))
    fails += not ok


anth("thinking.budget_tokens=2048 wins over the map", {"thinking": {"type": "enabled", "budget_tokens": 2048}}, 2048)
anth("no thinking block -> the server default effort's budget (64)", {}, 64)
anth("output_config.effort=xhigh -> 256", {"output_config": {"effort": "xhigh"}}, 256)

print("--- map off (THINKING_BUDGETS=off unsets the map) ---")
os.environ.pop("CLUB3090_REASONING_EFFORT_BUDGETS")
importlib.reload(glue)
check("map off: top-level xhigh gets no budget", THINK_ON, {"reasoning_effort": "xhigh"}, None)

print("--- a malformed map refuses at import (what install.sh relies on) ---")
os.environ["CLUB3090_REASONING_EFFORT_BUDGETS"] = '{"low": -1}'
try:
    importlib.reload(glue)
    print("  ✗ a negative budget in the map imported cleanly")
    fails += 1
except ValueError:
    print("  ✓ a negative budget in the map raises at import")

print(f"selftest_glue: {'FAIL' if fails else 'ok'} ({fails} failure(s))")
sys.exit(1 if fails else 0)
