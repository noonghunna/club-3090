# SGLang reasoning budget chosen by effort (`sglang-effort-thinking-budget`)

**What it does:** gives every chat request on the Qwen3.8-27B and ThinkingCap-Qwen3.8-27B `sgl/`
slugs a reasoning budget picked from the effort it runs at (`low` < `medium` < `xhigh`), unless the
request sends its own budget. SGLang enforces a per-request budget already — strict thinking's
`custom_params.thinking_budget` — but nothing picks one from the effort, so a client that sends only
`reasoning_effort` (all most coding agents can send) thinks without limit.

All the effort → budget logic lives in the engine-neutral `scripts/lib/effort_budget.py` (shared
with the vLLM patches, tested by `scripts/tests/test-effort-budget.sh`). This patch is only the
SGLang glue: one inserted call per call site, so an engine upgrade means moving lines, never porting
logic.

| Piece | Where | Role |
|---|---|---|
| `--enable-strict-thinking` | every wired compose | stock SGLang: enforces `custom_params.thinking_budget` per request (xgrammar backend) |
| `SGLANG_MAX_THINK_TOKENS` | exported by `effort_budget.py shell-env --engine sglang` | stock SGLang: the floor (= the budget of the compose's default effort) for anything no hook reaches |
| `club3090_sglang_effort_budget.py` | this dir → site-packages | the glue the two hooks call |
| `patch_sglang.py` | this dir | applies + AST-verifies the three edits |
| `install.sh` | this dir, run by the compose entrypoint | copies both modules into site-packages, parses the map, runs the patcher |
| `selftest.sh [image]` | this dir | offline drift guard (the `patches.yml` `drift_guard.check`) |

## Budgets (compose defaults)

| Effort | Budget (thinking tokens) |
|---|---|
| `low` (and `minimal`) | 4096 |
| `medium` | 16384 |
| `xhigh` (and `high`, `max`) | 32768 |

Maintainer-set 2026-10-09, the same for Qwen3.8-27B and ThinkingCap-Qwen3.8-27B. They are the
`--low/--medium/--xhigh` arguments of the `effort_budget.py shell-env` line in each compose.
`THINKING_BUDGET_LOW` / `THINKING_BUDGET_MEDIUM` / `THINKING_BUDGET_XHIGH` override them per
setting (empty = the default; a non-integer stops the boot), and `THINKING_BUDGETS=off` turns the
map and the floor off. The floor (`SGLANG_MAX_THINK_TOKENS`) is the budget of the compose's default
effort (`REASONING_EFFORT`, `low` unless set).

## Precedence (per request)

1. `custom_params.thinking_budget` — SGLang's own field. Untouched.
2. `thinking_token_budget` — the vLLM field name, accepted as an alias (declared on
   `ChatCompletionRequest` by this patch; stock SGLang silently drops it).
3. `/v1/messages` `thinking.budget_tokens` → `custom_params.thinking_budget` (so it ranks as 1).
4. The map budget for the effort in use: top-level `reasoning_effort` → `chat_template_kwargs.reasoning_effort`
   → the server default (`default_reasoning_effort` on our composes). Aliases `high`/`max` → xhigh,
   `minimal` → low.
5. The floor, `SGLANG_MAX_THINK_TOKENS`.

A request with thinking off (`enable_thinking: false`, effort `none`) gets no budget at all — it
would be inert there. A negative budget is passed through: SGLang treats any budget below 0 as
unlimited. A non-integer `thinking_token_budget` is a 400 (pydantic `Optional[int]`). An unknown
effort string never reaches the hook: SGLang's protocol already refuses it.

**Coverage:** chat completions and `/v1/messages` (which converts through the same function).
Structured-output requests (`response_format`, `tool_choice: required`) are covered too:
`grammar_manager.py` applies `custom_params.thinking_budget` on the grammar-cache hit, the
strict-only branch and the deferred branch after compilation. **Not hooked:** the Responses API
(`OpenAIServingResponses` builds its own sampling params) and the native `/generate` endpoint —
both get the floor. `/v1/messages` `output_config.effort` is removed by the LiteLLM gateway before
it reaches us, so those requests run at the server's default effort.

## The three edits (exact anchors, each must occur exactly once)

| Edit | File (under the `sglang` package) | Anchor | Becomes |
|---|---|---|---|
| `protocol` | `srt/entrypoints/openai/protocol.py` | in `ChatCompletionRequest`: `# Custom logit processor for advanced sampling control` + `custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None` + `custom_params: Optional[Dict] = None` | + `thinking_token_budget: Optional[int] = None` |
| `chat-hook` | `srt/entrypoints/openai/serving_chat.py` | in `_convert_to_internal_request`: `processed_messages = self._process_messages(request, is_multimodal)` + `# Build sampling parameters` + `sampling_params = request.to_sampling_params(` | + `__import__("club3090_sglang_effort_budget").apply_chat_budget(self, request, processed_messages)` between the two statements |
| `anthropic` | `srt/entrypoints/anthropic/serving.py` | in `_convert_to_chat_completion_request`: the `if anthropic_request.thinking.budget_tokens is not None:` branch with its `logger.warning(... "the budget is not enforced" ...)` | the branch's body becomes `__import__("club3090_sglang_effort_budget").apply_anthropic_budget(chat_request, anthropic_request.thinking.budget_tokens)` |

Every inserted line ends in `# club3090: sglang-effort-thinking-budget v1` (the marker).

The `anthropic` edit replaces SGLang's warning rather than adding beside it, because once the patch
is in, "the budget is not enforced" is false. The glue logs instead: a debug line normally, and a
WARNING that the budget is **not** enforced if the server runs without `--enable-strict-thinking`.
(The upstream comment above that branch still describes the unpatched behaviour; it is left as is to
keep the edit to one statement.)

**Why the protocol field.** `ChatCompletionRequest` has no pydantic `extra` setting (= ignore), so a
top-level `thinking_token_budget` is dropped at parse time — the request succeeds and the budget is
gone (selftest control 1 shows it on the stock image). Declaring the field is one line at one more
anchor; the alternatives (reading the raw body, flipping the model to `extra="allow"`) are more
invasive.

## Install behaviour

`install.sh` runs on every container start, after `shell-env` and before `sglang.launch_server`:

- locates the package with `importlib.util.find_spec("sglang")` — it never imports SGLang (~30 s);
- re-copies `club3090_effort_budget.py` / `club3090_sglang_effort_budget.py` into site-packages
  whenever their content differs, so a restart after `git pull` never runs stale logic;
- parses `CLUB3090_REASONING_EFFORT_BUDGETS` with the copied module: a malformed map stops the boot;
- patches only when the marker is absent; all three anchors are checked before any file is
  written, each file is replaced atomically, and the result is AST-verified. A half-patched tree
  (some markers present) refuses — recreate the container.
- prints one line: `[sglang-effort-thinking-budget] applied (hooks: …; modules: …; map: …)`.

It runs with `THINKING_BUDGETS=off` too (`map: off`): explicit budgets (`thinking_token_budget`,
`budget_tokens`, `custom_params`) are still honoured then; only the map and the floor go away.

⚠️ **`THINKING_BUDGETS=off` is not stock SGLang.** `--enable-strict-thinking` stays on, and strict
thinking also blocks `<tool_call>`, `</tool_call>`, `<|im_end|>` and `<|endoftext|>` while the model
is thinking (Qwen3 detector `think_excluded_tokens`; `</think>` is not blocked). Plan decision D4
accepts that if toolcall-15 / hermesagent-20 / cli-40 don't regress.

## Re-anchoring (an SGLang pin bump)

1. `bash models/qwen3.8-27b/sglang/patches/sglang-effort-thinking-budget/selftest.sh <new image>`.
   It needs Docker, no GPU. Green → done; the hooks carry over.
2. A refusal names the edit (`protocol` / `chat-hook` / `anthropic`) and says whether its anchor is
   missing or repeated. Read the new SGLang source at that place:
   - `chat-hook`: find `_convert_to_internal_request` (or its successor) and the last point where
     the request is final — after the server's default chat-template kwargs are merged in
     (`_process_messages`) and before the sampling params are built (`to_sampling_params`).
   - `anthropic`: find where `thinking.budget_tokens` is read after the chat request is built.
   - `protocol`: find `ChatCompletionRequest`'s `custom_params` field. If upstream now declares
     `thinking_token_budget` or `max_thinking_tokens` itself (sglang#36750), drop this edit.
3. Update the anchor text and the replacement in `patch_sglang.py` `EDITS`, and the AST checks
   (`check_*`) if the call site's shape changed. **Never loosen a match** (no regex, no fuzzy
   search): the anchor is what proves the hook sits where the logic assumes.
4. Re-run the selftest; then boot one slug and run `scripts/probe-effort-budget.sh` (the real check).

## Upstream status

**No upstream equivalent:** nothing in SGLang picks a budget from the effort. Related rows in
[`docs/UPSTREAM.md`](../../../../../docs/UPSTREAM.md) (SGLang section): sglang#36750 (`max_thinking_tokens`
on chat completions — an explicit-budget entry point, open), sglang#40634 (`--preferred-sampling-params`
`custom_params` clobbered by unset request fields — why the floor is an env var, open),
sglang#26330 / #27769 (custom logit processors bypassed under EAGLE v2 spec decode), sglang#20029
(a Qwen3.5 thinking-budget logit processor, closed unmerged), sglang#23953 (strict thinking,
merged — what this relies on).

- **Status:** `ours` (exception class, plan D8, decided 2026-10-09).
- **Criticality:** runaway reasoning is a measured defect — Flash-Next 2026-10-07: cap cut-offs on
  cli-40, −30% retry tokens recovered by a budget — and SGLang's shipped budget path is silently
  broken for these slugs (plan D8: the stock thinking-budget logit processor has the wrong
  think-end ids and is bypassed under spec decode), so strict thinking + this glue is the path.
- **Drop trigger:** upstream SGLang ships an equivalent (a per-request budget chosen by effort, or a
  native field covering both the explicit alias and the map) in the `sglang-stable` pin → drop the
  vendored copy in that pin bump. Posting the upstream equivalent is the drop path (plan D7: later,
  with the maintainer's go). If only sglang#36750 lands, drop the `protocol` edit and keep the rest.
