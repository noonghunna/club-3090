# DiffusionGemma — empty assistant turn after tool calls: root cause + fix

**Symptom (reported):** with Hermes Agent, DiffusionGemma "stops chatting / cuts off
mid-sentence" during multi-tool work. Hermes logs `Empty response after tool calls —
nudging model to continue processing` and retries.

**TL;DR root cause:** under deep, degraded agentic context (long multi-round tool loops +
repeated tool errors) the model emits the tool-call delimiters with a **Python-style paren
body** instead of the canonical gemma4 brace body. The stock gemma4 tool parser only matches
the brace form, so in streaming it **silently swallows the whole block** → an assistant turn
with no content and no tool call (Hermes records `"(empty)"`).

---

## How it was found (real data, not synthetic)

Synthetic after-tool probes (shallow, 27K-deep, and 75-tool/30K-deep) were **62/62 clean** —
the bug does **not** reproduce in isolation. It only appears in real long Hermes sessions.

Pulled the actual failures from the Hermes `state.db` on the gateway host:

- Session `20260619_054214_b53bf8` (Home-Assistant agent, "play Bonobo – Migration on the
  office quads"): `in≈1.89M / out=4578` over **59 API calls** — chronic under-production.
- Two stored empties: `id=3039`, `id=3051` — `role=assistant, finish=stop, content="(empty)",
  reasoning=None`. (`"(empty)"` is Hermes' 7-char placeholder.)
- Trigger visible right before them: `ha_call_service` repeatedly returned
  `{"error": "Missing required parameters: domain and service"}` (a Hermes arg-mapping
  mismatch) → the model retried and **degraded**.

**Replaying the exact pre-empty context** (18.8KB system prompt + full history + the real
75-tool catalog) against the live endpoint reproduces it: the model emits e.g.

```
<|tool_call>call:ha_list_entities(domain='media_player')<tool_call|>
```

— **parentheses**, not gemma4's braces `call:fn{key:<|"|>val<|"|>}`.

## Mechanism (confirmed in parser source)

`vllm/tool_parsers/gemma4_tool_parser.py`:

- `tool_call_regex = <\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>` — **brace-only**.
- Streaming `_handle_tool_call_end` runs `findall` → on the paren body returns `[]` → logs
  *"Tool call end detected but no complete tool call parsed yet"* → **returns `None`**.
- `_extract_partial_call` also bails: `if "{" not in func_part: return None`.
- Net: the entire `<|tool_call>…<tool_call|>` block is consumed and **nothing is emitted** —
  no content delta, no tool_call delta → empty turn (`finish=stop`).

## Measured empty rate at the worst real context (streaming, 8 reps × 2 contexts)

| Mode | BEFORE | AFTER |
|---|---|---|
| Reasoning OFF | **7/16 (44%)** true empty (`content=0, tool_calls=0`) — matches stored `"(empty)"` | **0/16 (0%)** |
| Reasoning ON  | **5/16 (31%)** "reason-only" (thinks, then swallowed paren-call → empty to user) | **0/16 (0%)** |

After the fix, **32/32** malformed calls are recovered into proper `tool_calls`.

Reasoning ON was *modestly better* than OFF both before (31% vs 44%) and is kept on.

---

## Fix (vendored full-file overlay)

[`models/diffusiongemma-26b-a4b/vllm/patches/gemma-image-fixes/gemma4_tool_parser.py`]
mounted over `vllm/tool_parsers/gemma4_tool_parser.py`. Two additions, both gated behind
"the strict brace regex found **zero** matches" so canonical output is byte-for-byte
unchanged:

1. **`_recover_tool_calls`** — a delimiter-anchored (`<|tool_call>`…`<tool_call|>`) lenient
   parser. For each block it reads `call:<name>` then parses the body: `{…}` via the stock
   Gemma4 arg parser, `(…)` via a new `_parse_python_args` (handles `key='v'`, `key:"v"`,
   `key:{json}`, `key:[…]`, empty `()`; top-level-comma split that respects quotes/brackets;
   JSON → `ast.literal_eval` → string coercion).
2. **No-swallow fallback** (`_malformed_tool_call_fallback`) — if even recovery yields
   nothing, surface the raw block (delimiters stripped) as **content** instead of emitting
   nothing, so a turn is never silently empty.

Applied in both the streaming (`_handle_tool_call_end`) and non-streaming
(`extract_tool_calls`) paths.

### Validation (live, 2× RTX 3090, `:gemma` digest `9c719fc0…`)

- Offline unit test of the recovery helpers against the real malformed strings: all paren /
  empty-args / nested-JSON / multi-call / mixed-text cases recover; canonical brace
  identical.
- Real-session streaming replay: empty rate **44%/31% → 0%/0%** (32/32 recovered).
- Canonical brace tool-calling regression (fresh "weather in Tokyo" requests):
  **8/8 OK** (`get_weather({"city":"Tokyo"})`) streaming + non-streaming.

## Notes / follow-ups (separate from this fix)

- **Upstream trigger**: the `ha_call_service` `Missing required parameters: domain and
  service` arg-mapping error starts the degradation spiral — worth fixing Hermes-side.
- **Stale aux route**: two saved Hermes `request_dump`s are 404s for model
  `qwen3.6-27b-fp8` POSTed to this endpoint (which only serves `diffusiongemma-26b-a4b`) —
  an unrelated stale model reference.
- **Lowering max-tokens does NOT help** — every empty is `finish=stop`, never length-capped.
