# vllm-effort-budget — a reasoning budget chosen by the request's effort (patch B)

**What.** vLLM enforces a reasoning budget per request (`thinking_token_budget`), but it never
picks one from `reasoning_effort`. A client that can only send an effort — which is all the
OpenAI API lets most coding agents send — reasons unbounded. This patch gives every chat
request without its own budget the budget the compose maps to its effort:
`THINKING_BUDGET_LOW` / `THINKING_BUDGET_MEDIUM` / `THINKING_BUDGET_XHIGH`
(`high`/`max` → xhigh, `minimal` → low).
Compose defaults, the same on Qwen3.8-27B and ThinkingCap: **low 4096 / medium 16384 / xhigh 32768**
tokens (maintainer-set 2026-10-09); the knobs override them, `THINKING_BUDGETS=off` turns it off.

**Why it is vendored (exception class, `patches.yml` `upstream.status: ours`).** Runaway
reasoning is a measured defect, not a tuning preference (Flash-Next 2026-10-07: cap cut-offs on
cli-40, −30 % retry tokens recovered by a budget). No upstream fix covers the
effort → budget mapping for kwargs-only and server-default effort (see *Upstream* below).
**Drop trigger:** upstream merges an equivalent, in the pinned release.

## How it works

| Piece | Where | Role |
|---|---|---|
| `scripts/lib/effort_budget.py` | repo, mounted at `/etc/club3090/effort_budget.py` | ALL the policy (engine-neutral, shared with SGLang). The compose entrypoint runs its `shell-env`, which exports `CLUB3090_REASONING_EFFORT_BUDGETS` (the map, JSON) and the floor, and prints `[effort-budget] v1 map=… default_effort=… floor=…` |
| `club3090_effort_budget_vllm.py` | this dir | vLLM glue: `apply_effort_budget(request, effective_kwargs)` |
| `patch_effort_budget.py` / `install.sh` | this dir, mounted at `/etc/club3090/effort-budget` | copies both modules next to the `vllm` package on every boot, validates the map, inserts the call |

**Precedence:** an explicit request budget > `map[effort in use]` > the floor (patch
[`vllm-default-thinking-budget`](../vllm-default-thinking-budget/README.md), = the budget of the
compose's `REASONING_EFFORT`). Effort in use: top-level `reasoning_effort` →
`chat_template_kwargs.reasoning_effort` → the server's `--default-chat-template-kwargs`. Thinking
off (`enable_thinking: false`, or effort `none`) → no map budget.

**vLLM's view, not the raw fields.** The glue reads `OpenAIServingChat._effective_chat_template_kwargs(request)`
— the merged kwargs the chat template actually renders with — instead of the request's fields and
the server defaults separately. The difference matters on a server booted with
`ENABLE_THINKING=false`: `ChatCompletionRequest.build_chat_params` sets
`enable_thinking = (reasoning_effort != "none")` when the request's own kwargs do not say, so a
request sending `reasoning_effort: xhigh` **does** think there, and must get the xhigh budget,
not the floor. `selftest_glue.py` asserts this against the real classes.

### The anchor (vLLM v0.31.0)

`vllm/entrypoints/openai/chat_completion/serving.py`, `OpenAIServingChat._create_chat_completion`,
non-beam branch — the only line of the file equal to (16-space indent)

```python
                sampling_params = request.to_sampling_params(
```

Two lines are inserted directly above it:

```python
                # [club3090 vllm-effort-budget] a budget from the effort map when the request set none. See models/qwen3.8-27b/vllm/patches/vllm-effort-budget/README.md
                __import__("club3090_effort_budget_vllm").apply_effort_budget(request, self._effective_chat_template_kwargs(request))
```

`self._effective_chat_template_kwargs` is defined on the same class (≈ line 191);
`self.default_chat_template_kwargs` (≈ line 155) feeds it.

### What reaches the hook (read from the v0.31.0 source)

| Endpoint | Path | Budget |
|---|---|---|
| `/v1/chat/completions` | `create_chat_completion` → `_create_chat_completion` | **map** (this hook) |
| `/v1/messages` | `AnthropicServingMessages` subclasses `OpenAIServingChat` and overrides neither `create_chat_completion` nor `_create_chat_completion`; `create_messages` → `to_chat_completion_request` → `_handle_thinking` sets `req.thinking_token_budget = thinking.budget_tokens` **before** it calls `create_chat_completion` | `thinking.budget_tokens` when sent (explicit wins), else the **map** (`output_config.effort` → `reasoning_effort` is honoured) |
| `/v1/chat/completions/batch` (`batch_serving.py`, `OpenAIServingChatBatch`) | its own loop calls `single_request.to_sampling_params(` directly — **not hooked** | the **floor** (patch D) |
| `/v1/responses`, `/v1/completions` | their own `to_sampling_params` | the **floor** (patch D) |

## Failure policy

| Situation | Result |
|---|---|
| marker present and the AST check passes | no-op, `[vllm-effort-budget] already applied …` (the modules are still re-copied: a restarted container keeps site-packages) |
| marker present but the hook is wrong | **boot refused** |
| anchor missing, or present more than once | **boot refused**, naming the anchor and pointing here |
| `_effective_chat_template_kwargs` or `_create_chat_completion` gone | **boot refused** |
| malformed map (`CLUB3090_REASONING_EFFORT_BUDGETS`) | **boot refused** before anything is patched |
| map off (`THINKING_BUDGETS=off`) | patched; the hook sets nothing |

No fuzzy matching: a hook in the wrong place would boot and silently budget nothing.

### Known limitation: `-1`

vLLM's request validator folds `thinking_token_budget: -1` ("unlimited") to `None` at parse time,
so neither this hook nor the floor can tell it from an absent budget — such a request gets the
map / floor. A client that wants no cap sends a large explicit budget (e.g. `1000000`); a server
without budgets is `THINKING_BUDGETS=off`.

## Re-anchoring at a pin bump

1. `bash models/qwen3.8-27b/vllm/patches/vllm-effort-budget/selftest.sh vllm/vllm-openai:<new tag>`.
   Green → nothing to do. It is the `drift_guard.check` in `patches.yml`.
2. Red on the anchor: open the new `vllm/entrypoints/openai/chat_completion/serving.py`, find
   where `OpenAIServingChat` turns a `ChatCompletionRequest` into `SamplingParams` for the
   **non-beam** path, and confirm `request` and `self` are in scope there. Update `ANCHOR` /
   `INDENT` (and, if the statement changed shape, `_is_to_sampling_params`) in
   `patch_effort_budget.py`. Keep exactly one anchor match.
3. Red on `_effective_chat_template_kwargs`: find what the new version renders the chat template
   with (merged request + server kwargs) and pass that instead — never the raw request fields
   (see *vLLM's view* above).
4. Check `/v1/messages` still funnels through the hooked function and still sets the explicit
   budget first (`vllm/entrypoints/anthropic/serving.py`); `selftest_glue.py` fails if the
   subclass starts overriding `(_)create_chat_completion`.
5. Re-run the selftest, then `scripts/probe-effort-budget.sh` against a booted slug with tiny
   budgets (`THINKING_BUDGET_LOW=64 THINKING_BUDGET_MEDIUM=128 THINKING_BUDGET_XHIGH=256`).

## Selftest

`bash selftest.sh [image]` (default `vllm/vllm-openai:v0.31.0`), no GPU, no network: install twice
(second a byte-identical no-op), the AST placement check plus a moved-hook positive control, the
glue against vLLM's real `ChatCompletionRequest` / `_effective_chat_template_kwargs` /
`to_sampling_params` and the Anthropic converter, the module's own `test-effort-budget.sh`, and
the negative controls (pristine file fails the AST check; malformed map, duplicated anchor and
removed anchor each refuse).

## Upstream

| Ref | Relation | Switch to native when |
|---|---|---|
| [vllm#55432](https://github.com/vllm-project/vllm/pull/55432) "Map reasoning effort to thinking budget" (open, `needs-rebase`) | the upstream equivalent for a **top-level** `reasoning_effort` only (`--reasoning-effort-budgets`, chat + batch chat). Not vendored: 5 files, already behind `main`, and our precedence would have had to patch exactly the code it changes | it merges in a pinned release **and** also resolves kwargs-only effort and the server-default effort. Then drop this hook and pass the map as that flag |

Tracked in `docs/UPSTREAM.md`. Related watch rows there: #58231 (budget + spec decode + top-p with
no top-k → token 0; the composes keep `top_k=20`), #58402 / #54467 (custom forced end strings;
we keep the plain `</think>`), #44676 (a tool call opened inside `<think>` counts against the budget).
