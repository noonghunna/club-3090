# DiffusionGemma — Ampere/TP + parser fixes for the official `vllm/vllm-openai:gemma` image

vLLM publishes an official **`vllm/vllm-openai:gemma`** image (pushed 2026-06-10,
same day as the DiffusionGemma blog) that has the dLLM arch baked in — it's a stock
vLLM build of the `dgemma` branch commit `74b5964f` (`DiffusionGemmaForBlockDiffusion`
registers natively; `transformers 5.10.2`). So we **pin that image** and no longer
sideload the model code.

But five fixes are **NOT upstream** (not in PR #45163), so they're not in `:gemma` —
vLLM builds/tests on H100/B200 and their recipe is TP=1, so neither our Ampere fp8
path nor our TP=2 path is exercised upstream (and the streaming reasoning leak + the
malformed-tool-call swallow only show on a block-diffusion decoder under deep agentic
context, which upstream's gemma parsers were never exercised against). The compose
**bind-mounts these 5 files** over the image's vllm package (`site_package_overlay`,
`wired_at: volumes`):

| File → mounts over | Fix |
|---|---|
| `marlin.py` → `vllm/model_executor/kernels/linear/scaled_mm/marlin.py` | sm_86 fp8 Marlin **sub-tile-K pad** (dense). Without it, `:gemma` dies in warmup: `Invalid thread config … num_bits=8 … max_shared_mem=101376` for K=352/1056 (can't tile in Ampere's 99 KB shared mem). |
| `marlin_utils_fp8.py` → `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py` | the FP8-MoE half of the K-pad. |
| `diffusion_gemma.py` → `vllm/model_executor/models/diffusion_gemma.py` | **TP-vocab soft-embed fix** (slice probs to the rank's vocab shard → local-embed matmul → TP all-reduce) + `:656` dtype cast. `:gemma`'s copy is the pristine branch file → hits the TP=2 vocab/dtype bug without this. |
| `gemma4_reasoning_parser.py` → `vllm/reasoning/gemma4_reasoning_parser.py` | **Streaming reasoning-parser fix.** The base `BaseThinkingReasoningParser.extract_reasoning_streaming` only strips the `<|channel>` start token when it can pair it with the end token in the *same* delta. DiffusionGemma decodes a whole ~256-token canvas per SSE delta, so `<\|channel>thought\n…reasoning…` arrives in one delta while the closing `<channel\|>` lands in a *later* canvas → the base parser's "start in delta, no end" branch returns `delta_text` verbatim, leaking the literal `<\|channel>thought\n` into the reasoning stream (agent/chat clients render it as raw "thoughts in chat"). The overlay strips a leaked leading `<\|channel>` in `Gemma4ReasoningParser.extract_reasoning_streaming` before its existing `thought\n`-label logic. Non-streaming (`extract_reasoning`) was already clean. See [`docs/UPSTREAM.md`](../../../../../../docs/UPSTREAM.md). |
| `gemma4_tool_parser.py` → `vllm/tool_parsers/gemma4_tool_parser.py` | **Malformed tool-call recovery + no-swallow fallback + plain-quoted-value handling.** The stock parser only understands the canonical brace form `<\|tool_call>call:name{key:<\|"\|>val<\|"\|>}<tool_call\|>` (regex `<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>`) with values delimited by the `<\|"\|>` special token. Under deep, degraded agentic context (long multi-round tool loops + repeated tool errors) DiffusionGemma drifts in two ways: **(a)** it emits the delimiters with a **paren / python-arg body** (`call:ha_list_entities(domain='media_player')`) → the brace regex matches nothing, `_handle_tool_call_end` returns `None`, the ENTIRE block is silently swallowed → an **empty assistant turn after tool calls**; **(b)** it keeps the canonical braces but writes values with **plain JSON/python quotes** (`call:fn{domain: "media_player"}`) instead of `<\|"\|>media_player<\|"\|>` → the stock bare-value path *kept the surrounding quotes*, so the emitted JSON arg became `"\"media_player\""` (literal quotes inside the string) → Hermes errors `Invalid domain format: '"media_player"'`, quoted tool names (`'"mcp_…"' is not a deferrable tool`), and `'arguments' is not valid JSON`. The overlay adds: `_recover_tool_calls` (delimiter-anchored lenient paren/python-arg parser) used **only when the strict regex finds zero matches**; a no-swallow fallback that surfaces the raw block as content if even recovery fails (never a silent empty); and **quote-aware string-value scanning** in `_parse_gemma4_args` / `_parse_gemma4_array` (+ a `_parse_gemma4_value` safety net) that parses plain-quoted values without truncating on embedded commas/braces, strips the surrounding quotes, and strips any leaked literal `<\|"\|>` delimiter from a value. The proper `<\|"\|>` delimiter path is matched first, so canonical brace output is byte-for-byte unchanged. See [`docs/UPSTREAM.md`](../../../../../../docs/UPSTREAM.md). |

Validated live on 2× RTX 3090 (2026-06-11): `:gemma` clean dies on the Marlin wall in
warmup; `:gemma` + the marlin/TP mounts boots, serves coherent output, 262K, ~177/180 TPS
typical (~1100 peak on low-entropy), 23.1 GB/card. The `gemma4_reasoning_parser.py` mount
was added 2026-06-19 after reproducing the streaming `<|channel>thought` leak live. The
`gemma4_tool_parser.py` mount was added 2026-06-19 after reproducing the empty-after-tool-calls
turn live by replaying real Hermes `state.db` sessions: the empty rate at the worst real
context dropped from **44% (reasoning off) / 31% (on) → 0% / 0%** (32/32 malformed calls
recovered into proper tool calls), with canonical brace tool-calling unchanged (8/8 OK).
See [`results-nvlink/diffusionGemma_empty_after_tools.md`](../../../../../../results-nvlink/diffusionGemma_empty_after_tools.md).
The **plain-quoted-value** half was added later the same day after the empties were fixed and
the *next* symptom surfaced — `'arguments' is not valid JSON` / `Invalid domain format:
'"media_player"'` from values that kept their literal quotes. Replaying the real failing
session (`20260619_082023`) the embedded-quote rate on actual tool calls dropped from
~all-broken → **7/8 clean** (the residual being the model hallucinating/garbling tokens at
depth, not the parser), empty-rate held at **0% / 0%**, canonical values stayed clean. See
[`results-nvlink/diffusionGemma_tool_args_json.md`](../../../../../../results-nvlink/diffusionGemma_tool_args_json.md).

## Provenance + rebase

The marlin + TP-vocab files are the surviving fixes from the original sideload overlay
(Codex's marlin-K-pad = the K-axis analogue of our PR #40361 sub-tile-N pad; + the
TP-vocab/dtype fix). The marlin pair were authored against a stock-nightly marlin
(identical stock==dgemma-branch), and apply cleanly onto `:gemma`'s newer base
(verified — no skew). `gemma4_reasoning_parser.py` and `gemma4_tool_parser.py` are
club-3090 fixes authored against `:gemma`'s own copies of those files (commit `74b5964f`);
each is the stock file plus the club-3090 delta (the leaked-start-token strip in
`extract_reasoning_streaming`; the malformed-tool-call recovery + no-swallow fallback in
the tool parser — see the table above). Both deltas are gated behind a "strict path found
nothing" check, so canonical output is unchanged.

**Rebase when `:gemma` is re-pinned** (vLLM may re-push the tag): pull the new image,
diff its `vllm/model_executor/.../marlin.py` / `marlin_utils_fp8.py` /
`models/diffusion_gemma.py` / `vllm/reasoning/gemma4_reasoning_parser.py` /
`vllm/tool_parsers/gemma4_tool_parser.py` against these, re-apply the K-pad + TP-vocab +
start-token-strip + tool-recovery + quote-aware-value deltas, re-validate (boot + serve + the
gate + a streaming thinking-on repro that asserts no `<|channel>` leak + a malformed-paren-call
replay that asserts 0% empty-after-tools + a plain-quoted-value replay that asserts no literal
quotes leak into argument JSON).

**Retire** each file independently when its fix lands upstream: the marlin K-pad (our
PR #40361 / an Ampere Marlin fix), the TP-vocab fix into `:gemma`, the reasoning leak when
the base streaming parser strips an unpaired start token, and the tool parser when upstream
recovers (or rejects-without-swallowing) malformed non-brace tool calls (see UPSTREAM.md).
Mount nothing once all five are upstream.
