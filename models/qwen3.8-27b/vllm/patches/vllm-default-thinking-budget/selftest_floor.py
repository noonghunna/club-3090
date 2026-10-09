#!/usr/bin/env python3
"""selftest.sh's in-image check of the floor (patch D) against vLLM's real objects.

Runs inside the pinned image (no GPU) after install.sh, with
CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET=64 exported by effort_budget.py shell-env:

  * SamplingParams: no budget -> the floor; an explicit budget (incl. 0) wins;
    trace-replay requests get none; the floor survives the engine's msgpack round trip.
  * The entrypoints patch B does not hook get the floor: Responses API, completions, and a
    chat request built without the hook (what /v1/chat/completions/batch does).
  * Inertness on a thinking-OFF request (the floor budgets those too): the pinned image's
    REAL Model-Runner-V2 kernel (v1/worker/gpu/sample/thinking_budget.py), run on the CPU
    under the Triton interpreter, does not force `</think>` when the prompt already holds a
    closed think block, with and without draft tokens — and DOES force it on a thinking-ON
    prompt past the budget (positive control).
  * No floor env -> stock behaviour.
"""
from __future__ import annotations

import os
import sys

# The caller exports it the way the compose does (effort_budget.py shell-env, default effort
# low = 64), so this also proves the entrypoint's export reaches SamplingParams.
if os.environ.get("CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET") != "64":
    sys.exit("selftest_floor: run with CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET=64 exported by shell-env")
os.environ["TRITON_INTERPRET"] = "1"   # before triton is imported: run kernels on the CPU

import msgspec  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import vllm.triton_utils as _tu  # noqa: E402

# Without a GPU vLLM swaps triton for a placeholder; hand it the real one (interpreter mode).
_tu.triton, _tu.tl = triton, tl

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.entrypoints.openai.completion.protocol import CompletionRequest  # noqa: E402
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.worker.gpu.sample.thinking_budget import apply_thinking_budget  # noqa: E402

fails = 0


def check(label, got, want):
    global fails
    ok = got == want
    print(("  ✓ " if ok else "  ✗ ") + label + ("" if ok else f": got {got!r}, want {want!r}"))
    fails += not ok


print("--- SamplingParams with CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET=64 ---")
check("no budget -> the floor", SamplingParams().thinking_token_budget, 64)
check("from_optional(None) -> the floor", SamplingParams.from_optional(thinking_token_budget=None).thinking_token_budget, 64)
check("explicit 300 wins", SamplingParams(thinking_token_budget=300).thinking_token_budget, 300)
check("explicit 0 wins (0 is a budget)", SamplingParams(thinking_token_budget=0).thinking_token_budget, 0)
check("-1 ('unlimited') folds to None, then the floor applies (documented)",
      SamplingParams(thinking_token_budget=-1).thinking_token_budget, 64)
check("trace replay gets no floor (it refuses any budget)",
      SamplingParams(trace_decode_token_ids=[1, 2]).thinking_token_budget, None)
rt = msgspec.msgpack.decode(msgspec.msgpack.encode(SamplingParams(thinking_token_budget=300)), type=SamplingParams)
check("msgpack round trip keeps an explicit budget", rt.thinking_token_budget, 300)
rt = msgspec.msgpack.decode(msgspec.msgpack.encode(SamplingParams()), type=SamplingParams)
check("msgpack round trip keeps the floor", rt.thinking_token_budget, 64)

print("--- entrypoints without patch B's hook get the floor ---")
r = ResponsesRequest(input="hi", reasoning={"effort": "high"})
check("Responses API (reasoning.effort=high) -> the floor", r.to_sampling_params(1024, {}).thinking_token_budget, 64)
check("completions -> the floor", CompletionRequest(prompt="hi").to_sampling_params(1024, {}).thinking_token_budget, 64)
c = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], reasoning_effort="xhigh")
check("a chat request built without the hook (the /v1/chat/completions/batch path) -> the floor",
      c.to_sampling_params(1024, {}).thinking_token_budget, 64)
c = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], thinking_token_budget=500)
check("…and its explicit budget still wins", c.to_sampling_params(1024, {}).thinking_token_budget, 500)

print("--- a budget on a thinking-OFF request is inert (Model Runner V2 kernel, Triton CPU interpreter) ---")
START, END = 1, 2   # stand-ins for <think> / </think>; the kernel matches id sequences


def forced_rows(committed, drafts, budget):
    """Rows whose logits the real kernel forces to END for one request with `drafts`."""
    vocab, cap, n = 32, 64, 1 + len(drafts)
    all_ids = torch.zeros((1, cap), dtype=torch.int32)
    all_ids[0, :len(committed)] = torch.tensor(committed, dtype=torch.int32)
    logits = torch.zeros((n, vocab), dtype=torch.float32)
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32)  # noqa: E731
    apply_thinking_budget(
        logits, i32([0]), i32([0] * n), i32([budget]), all_ids, i32([len(committed)]),
        i32([committed[-1]] + drafts), i32(list(range(n))),
        torch.full((1,), -1, dtype=torch.int32), torch.full((1,), -1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32), i32([START]), i32([END]), i32([END]))
    return [row for row in range(n) if logits[row, END] > 1e8]


off = [10, START, 11, END, 12]          # Qwen3.8 thinking-off prompt: <think>\n\n</think> already closed
on = [10, START]                        # thinking-on prompt: an open <think>
check("thinking-off, 8 tokens generated past a budget of 2 -> nothing forced",
      forced_rows(off + [20, 21, 22, 23, 24, 25, 26, 27], [], 2), [])
check("thinking-off with 3 draft tokens (spec decode) -> nothing forced",
      forced_rows(off + [20, 21, 22, 23], [24, 25, 26], 2), [])
check("positive control: thinking-on, 3 reasoning tokens past a budget of 2 -> </think> forced",
      forced_rows(on + [20, 21, 22], [], 2), [0])
check("positive control: thinking-on, 1 reasoning token, budget 2 -> not yet",
      forced_rows(on + [20], [], 2), [])
check("positive control with drafts: forced from the draft that crosses the budget",
      forced_rows(on + [20], [21, 22, 23], 2), [1, 2, 3])

print("--- no floor env -> stock behaviour ---")
os.environ.pop("CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET")
check("SamplingParams() without the env -> None", SamplingParams().thinking_token_budget, None)

print(f"selftest_floor: {'FAIL' if fails else 'ok'} ({fails} failure(s))")
sys.exit(1 if fails else 0)
