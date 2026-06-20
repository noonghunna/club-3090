# DiffusionGemma — streaming reasoning-parser `<|channel>thought` leak: root cause + fix

**Date:** 2026-06-19 · **Rig:** test777 (2× RTX 3090, PCIe, no NVLink) · **Endpoint:** `:8020`
**Image:** `vllm/vllm-openai:gemma@sha256:9c719fc0…` (dgemma branch `74b5964f`)
**Container:** `vllm-diffusiongemma-26b-a4b-fp8-tp2`

## Symptom

Agent / chat clients (the "hermes agent" interactive client) hitting diffusionGemma in
**streaming** mode with thinking enabled saw the literal sentinel `<|channel>thought`
rendered as raw text ("thought lines in chat"). Tool calls were *also* suspected, but
were ultimately found to work (see Regression check).

## Root cause

`vllm/reasoning/basic_parsers.py` → `BaseThinkingReasoningParser.extract_reasoning_streaming`
only strips the `<|channel>` start token when it can pair it with the end token in the
**same** delta:

```python
elif self.start_token_id in delta_token_ids:
    if self.end_token_id in delta_token_ids:
        # both in delta → start token stripped via .find()
        ...
    else:
        # start token in delta, NO end token yet
        return DeltaMessage(reasoning=delta_text)   # <-- leaks "<|channel>thought\n…"
```

In normal autoregressive decoding the `<|channel>` start token arrives as its own
single-token delta and is consumed by the "Skip single special tokens" guard.
**DiffusionGemma is a block-diffusion LM**: it denoises a whole ~256-token canvas and
emits it as **one SSE delta**, so `<|channel>thought\n…reasoning…` all arrive together
while the closing `<channel|>` lands in a *later* canvas — hitting exactly that leaky
`else` branch. The gemma4 subclass then only strips the `thought\n` label (not
`<|channel>`), so `startswith("thought\n")` is False and it re-emits the full text.

Non-streaming (`extract_reasoning`) was always clean — it partitions on the start token.

## Fix

Vendored full-file overlay (same delivery as the 3 existing dgemma image fixes — a
read-only bind-mount over the image's vllm package, NOT an install.sh diff-apply):

- `models/diffusiongemma-26b-a4b/vllm/patches/gemma-image-fixes/gemma4_reasoning_parser.py`
  → mounts over `…/vllm/reasoning/gemma4_reasoning_parser.py`

It strips a leaked leading `<|channel>` in `Gemma4ReasoningParser.extract_reasoning_streaming`
*before* the existing `thought\n`-label logic. The start token decodes atomically from one
token id (`skip_special_tokens=False`), so a `startswith` check is sufficient and never
matches genuine reasoning content. Wired in `compose/dual/fp8/base.yml`; UPSTREAM.md row
added (no upstream PR yet — cleanest general fix is in the base parser's "start in delta,
no end" branch).

## Validation (live, 2026-06-19)

Streaming, `enable_thinking:true`, train-overtake prompt (the original repro):

**BEFORE** (image's stock parser):
```
"delta":{"reasoning":"<|channel>thought\n*   Train A: Leaves at 3:00 PM …"}
```

**AFTER** (vendored overlay, `grep -c channel` = **0**):
```
"delta":{"reasoning":"*   Train 1: Leaves at 3:00 PM, speed = 60 mph.\n    *   Train 2: …"}
…
"finish_reason":"stop"
```
Reasoning is clean *and* fully intact (not over-stripped); content carries the answer
("…catches the first at **7:00 PM**").

**Regression check** — streaming + tools + `enable_thinking:true` (3-tool prompt):
```
channel hits: 0
tool names:  get_weather · get_weather · get_time
arguments:   populated (Tokyo / London / Paris)
finish_reason: tool_calls
```
No leak, tool calls + arguments intact.

## Notes

- The earlier "use streaming-off as a client mitigation" idea is **no longer needed** —
  streaming reasoning is now clean.
- At-depth tool-call misses (~11K–38K ctx, club-3090 #255) are a separate model/quant
  quality axis, not this parser bug.
- The overlay was validated live by `docker cp` + `docker restart` (preserves the live
  port/env); the persistent path is the `base.yml` bind-mount, picked up on next relaunch.
- **Rebase trigger:** if `:gemma` is re-pinned, re-diff `vllm/reasoning/gemma4_reasoning_parser.py`
  and re-apply the start-token strip (see the patch README).
