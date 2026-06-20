# DiffusionGemma — history chain-of-thought replay: root cause + fix

**Symptom (reported):** in long Hermes agent loops, DiffusionGemma sometimes narrates or
stops instead of issuing the next tool call — especially after several successful tool rounds
when prior assistant turns had `reasoning_content` populated.

**TL;DR root cause:** agent clients resend prior assistant `reasoning` / `reasoning_content`
on every API call. The stock vLLM `tool_chat_template_gemma4.jinja` re-rendered that history
as literal `<|channel>thought\n…\n<channel|>` blocks in the prompt. That puts off-distribution
CoT back into the model's input (Gemma / dLLM expects prior-turn thinking to stay out of
history). Same failure class as [llama.cpp 1b25994](https://github.com/potto007/llama.cpp/commit/1b2599478244e9b345bc9bc5f4f2e5ab7e2f0cca).

---

## How it was found

1. Confirmed Hermes `state.db` stores and resends `reasoning_content` on assistant turns
   after tool calls.
2. Rendered a multi-round history through the stock template inside the running `:gemma`
   container: a marker string `PRIOR_COT_XYZ123` placed in `reasoning_content` appeared in
   the rendered prompt as `<|channel>thought\nPRIOR_COT_XYZ123\n<channel|>`.
3. The stock guard `loop.index0 > ns_turn.last_user_idx` does **not** protect typical agent
   loops (one user message, many assistant→tool rounds) — every prior tool-call turn replays
   its CoT.

---

## Fix

Vendored overlay: `tool_chat_template_gemma4.jinja` — full stock copy minus the block that
re-injects history `reasoning` / `reasoning_content`. Current-turn thinking still comes from
`enable_thinking` + `<|think|>`; `strip_thinking()` on assistant `content` unchanged.

Mount: `/vllm-workspace/examples/tool_chat_template_gemma4.jinja` (same path vLLM reads for
`--chat-template` on the dual compose).

---

## Validation (test777, 2026-06-19)

| Check | Stock template | Overlay |
|---|---|---|
| Marker `PRIOR_COT_*` in rendered prompt | yes | **no** |
| `<\|channel>thought` blocks from history | ≥1 | **0** |
| Live multi-round (history + new user ask) | — | `finish_reason=tool_calls`, `get_weather({"location":"Boston"})` |
| Live empty-after-tools follow-up | — | `finish_reason=tool_calls`, not empty |
| Parser overlays still mounted | — | yes (reasoning + tool + template) |

Template render run inside container; live API from host against `127.0.0.1:8020`.

## Ablation vs PR #443 parsers (test777, 2026-06-19)

See [`diffusionGemma_ablation_pr443.md`](./diffusionGemma_ablation_pr443.md). Summary:

- **Template only** does **not** fix empty-after-tools (`054214` stays ~31% OFF / 56% ON).
- **Parsers only** (PR #443) fixes empties to **0%/0%** without template.
- **All six mounts** best for deep CoT-heavy sessions (`100411`: 4/4 tool calls vs 1/4 with template-only).
