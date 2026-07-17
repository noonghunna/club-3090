# Google Gemma 4 Canonical Chat Template (vendored)

**Source:** `google/gemma-4-31B-it` @ main, template published **2026-07-09**
("fix: chat template — null handling, reasoning preservation, turn-tag balance").
Identical file ships in `google/gemma-4-12B-it` and `google/gemma-4-26B-A4B-it` —
one canonical template family-wide.
**sha256:** `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`

## Why vendored

The gemma vLLM composes previously pointed at the image-bundled
`/vllm-workspace/examples/tool_chat_template_gemma4.jinja`. That copy (as of
`vllm/vllm-openai:v0.25.1`) **predates Google's 2026-07-09 fix** and differs in
three functional ways, each an agent-loop failure class:

1. Injects an empty `<|channel>thought` block into *history* model-turns when
   thinking is off (phantom thought channels in replayed conversations).
2. `preserve_thinking` applied to every turn; canonical scopes it to
   **tool-call turns only** (reasoning-preservation fix).
3. Turn-closure emission ignores the next turn; canonical adds the
   `next_nt.found` guard (the turn-tag-balance fix for tool loops).

Vendoring (rather than trusting the image copy) pins the fixed bytes and makes
template provenance auditable — same rationale as the froggeric template for the
qwen3_5 family.

## Consumers

All gemma-family vLLM composes (31B, 12B, 26B-A4B, diffusiongemma) mount this
file and pass it via `--chat-template`. GGUF lanes keep their embedded
templates (minja compatibility unvalidated — do NOT point llama.cpp at this
file without a minja parse check first).

## Drop trigger

A future `vllm-stable` pin whose bundled example is byte-identical to (or
newer than) the canonical → repoint composes at the image path and retire this
dir. Check: `diff` this file against `/vllm-workspace/examples/tool_chat_template_gemma4.jinja`
in the new image.
