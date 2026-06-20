# DiffusionGemma — invalid-JSON tool arguments (literal quotes in values): root cause + fix

**Symptom (reported):** after the empty-after-tool-calls fix landed, DiffusionGemma is "doing
a LOT better" but tool calls now fail with bad JSON. Hermes transcript:

```
⚡ tool_call mcp_ma_queue_get_active_queue  [tool_call 'arguments' is not valid JSON: Expe…]
⚡ tool_call "mcp_ma_queue_get_active_queue"  ['"mcp_ma_queue_get_active_queue"' is not a de…]
```

**TL;DR root cause:** under deep agentic context the model keeps the canonical gemma4 *braces*
but writes string values with **plain JSON/python quotes** — `call:fn{domain: "media_player"}`
— instead of gemma4's `<|"|>media_player<|"|>` special-token delimiters. The stock parser's
bare-value path **kept the surrounding quotes**, so the emitted JSON argument became the string
`"media_player"` *with literal quotes inside it*. Downstream that shows up as
`Invalid domain format: '"media_player"'`, quoted tool names, and `'arguments' is not valid
JSON`.

---

## How it was found (real data)

Dumped the most recent Hermes session `20260619_082023_7ec358` (Music-Assistant agent) from
`state.db` and printed each assistant `tool_calls` payload + the matching tool error. The
arguments were *valid JSON* but every string value carried embedded quotes:

| msg | emitted `arguments` | tool error |
|---|---|---|
| `id=3428` | `{… "domain": "\"media_player\"" …}` | `Invalid domain format: '"media_player"'` |
| `id=3411` | `{"name": "\"mcp_ma_queue_get_active_queue\""}` | `'"mcp_ma_…"' is not a deferrable tool` |
| `id=3409` | `{"player_id": "\"upc0bfbe8f6266\""}` | (wrong id — quotes baked in) |
| `id=3402` | `{"player_id": "upc0bfbe8f6266"}` | **clean** — this turn used `<|"|>` delimiters |

So the model is *inconsistent*: when it uses `<|"|>…<|"|>` the value is clean (`id=3402`);
when it uses plain `"…"` the quotes leak (`id=3409`). Same call, two encodings.

## Mechanism (parser source)

`_parse_gemma4_args` value dispatch had branches for `<|"|>…<|"|>`, `{…}`, `[…]`, and a
catch-all **bare** branch. A value like `"media_player"` is none of the first three, so it fell
to bare: scan-until-`,`/`}`/`]`, return the raw text **including the quotes** → `"media_player"`
→ `json.dumps` → `"\"media_player\""`. The bare scan also breaks on the first comma *inside* a
quoted string (`"a, b"` truncates to `"a`).

## Fix (vendored full-file overlay)

[`models/diffusiongemma-26b-a4b/vllm/patches/gemma-image-fixes/gemma4_tool_parser.py`]
mounted over `vllm/tool_parsers/gemma4_tool_parser.py`. Added a **plain-quoted-string branch**
to `_parse_gemma4_args` *and* `_parse_gemma4_array` (before the bare-value catch-all), plus a
quote-strip safety net in `_parse_gemma4_value`:

- quote-aware scan from the opening `"`/`'` to the matching close (honoring `\` escapes), so
  embedded commas/braces don't truncate the value;
- strip the surrounding quotes (a quoted value is always a string — no re-coercion);
- strip any leaked literal `<|"|>` delimiter from a value (the model sometimes mixes the two
  encodings at depth).

The canonical `<|"|>…<|"|>` delimiter branch is matched **first**, so well-formed brace calls
are byte-for-byte unchanged.

### Validation (live, 2× RTX 3090, `:gemma` digest `9c719fc0…`)

- **Offline unit test** on the exact failing strings: `domain: "media_player"` → `media_player`,
  full `ha_call_service` body clean, `q: "a, b, c"` preserved (comma inside quotes), mixed types
  (`count: 3, flag: true`) correct; canonical `<|"|>` path unchanged.
- **Real-session replay** (`20260619_082023`, streaming, contexts before the failing calls):
  embedded-quote rate on actual tool calls **~all-broken → 7/8 clean**. The 1 residual is the
  model emitting genuinely corrupted token soup at depth (`'tidal--…')}`, plus hallucinated
  player-ids that vary per rep) — a model-quality-at-depth axis the parser faithfully passes
  through, **not** a parser bug.
- **No regression:** empty-after-tools replay held **0% / 0%**; canonical "weather in Tokyo"
  values stayed clean (`{"city":"Tokyo"}`) — the only blips were intermittent model
  double-emission of an empty `{}` block (nondeterminism at temp 0.6, unrelated to this change).

## Notes / follow-ups (separate from this fix, model behavior — not parser)

- The model sometimes wraps a real call in Hermes' **`tool_call` deferred-dispatch meta-tool**
  with a *stringified* non-JSON `arguments` (`{"arguments": "{player_id: …}", "name": "…"}`) —
  Hermes then can't parse the inner string (`Expecting property name enclosed in double
  quotes`). It recovered by calling the tool directly. This is model confusion about the
  deferred-tool mechanism, not arg encoding.
- `ha_call_service` `domain`/`service` arg-mapping confusion persists (the model guesses the
  split) — a Hermes-side semantic, same as the empty-after-tools spiral trigger.
- These are the documented **at-depth quality** axis; chasing them with more aggressive parser
  heuristics would risk corrupting legitimate output.
