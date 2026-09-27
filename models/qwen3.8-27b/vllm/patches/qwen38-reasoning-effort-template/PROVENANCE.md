# qwen38-reasoning-effort-template

Vendored copy of Qwen3.8-27B's own `chat_template.jinja` with a **single semantic
change**: `reasoning_effort: "high"` is mapped onto `xhigh` instead of raising.

## The defect

Qwen3.8 renamed the conventional top reasoning rung. Its ladder is:

    low  <  medium  <  xhigh        (xhigh is also the default)

There is no `high`. The stock template hard-rejects anything else:

```jinja
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort ...') }}
```

Both OpenAI and Anthropic use `low` / `medium` / `high`, so a generic client sending
the standard top rung gets a Jinja `TemplateError` surfaced as **HTTP 500**.

The failure is silent in the worst way: `/v1/models` and `/health` both return 200
and the model is demonstrably serving, so the endpoint looks healthy while every
thinking-mode chat request fails. Observed on the reference rig against vLLM's
Anthropic-compatible `POST /v1/messages?beta=true` — the surface a drop-in
Claude-API client talks to.

## Scope: all twenty vLLM slugs, one identical template

The template is **byte-identical across every served Qwen3.8 checkpoint** — official
FP8, the AutoRound INT4 repack, and the NVFP4 export all ship the same 8,952-byte file
(sha256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`). Verified
by fetching each and comparing:

```bash
for r in Qwen/Qwen3.8-27B-FP8 Frozenlock/Qwen3.8-27B-int4-AutoRound RadixArk/Qwen3.8-27B-NVFP4; do
  curl -sL "https://huggingface.co/$r/resolve/main/chat_template.jinja" | sha256sum
done
```

So one vendored file is correct for every slug, and the quantiser is irrelevant.

**Re-verified 2026-08-21**, on two counts:

| checkpoint | `chat_template.jinja` sha256 | |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | `c3cf9e34…81041` | served |
| `Frozenlock/Qwen3.8-27B-int4-AutoRound` | `c3cf9e34…81041` | served (post-#1070 swap) |
| `RadixArk/Qwen3.8-27B-NVFP4` | `c3cf9e34…81041` | served |
| `syvai/Qwen3.8-27B-DFlash2-W4A16` | `f36668dd…ba59` | **drafter — never the render source** |

1. #1070 swapped the FAST tier's weights Avuja → Frozenlock, which would normally
   invalidate this file's provenance. It does not: the hash is unchanged, so the
   vendored copy is still the correct base.
2. #1072 added the DFlash2 super/ultra tiers, taking the slug count 8 → 20. The
   DFlash2 *drafter* repo ships a different template, but vLLM renders from the
   **served** model's tokenizer, so that file is deliberately not vendored.

## The change

A short mapping inserted before the validation. `high` maps to **`xhigh`**,
Qwen3.8's real top rung.

### Why `xhigh` and not `medium`

The original submission mapped `high` onto `medium`, reasoning that `medium` is
the un-nudged baseline while `xhigh` injects a "think carefully, validate
assumptions, consider alternatives" preamble that is slow and timeout-prone — and
that a Claude-API client sends `high` on *every* request, so mapping to the top
rung would push all agent traffic into the slowest mode on slugs shipping
`max_num_seqs=1`. That cost is real and documented in the compose headers.

It was changed to `xhigh` on merge, for three reasons:

1. **Bijection.** `low`/`medium`/`high` maps onto `low`/`medium`/`xhigh` with the
   ordering intact. Under the `medium` mapping, two distinct client values collapse
   onto one server behaviour and the top rung becomes **unreachable through
   standard OpenAI/Anthropic vocabulary at all**.
2. **These are not a scale with a middle.** `medium` has no template branch and
   injects nothing. So mapping the top rung there does not hand the caller "one
   rung down" — it hands them *no reasoning instruction whatsoever*, which is the
   opposite of what they asked for.
3. **The cost belongs in the default, not the vocabulary.** Traffic that sends no
   `reasoning_effort` is governed by the server default (`low`, set via
   `--default-chat-template-kwargs`) and is unaffected by this mapping. A caller
   that explicitly sends `high` has opted in. If `xhigh` is too slow for a given
   agent, the agent should send `medium` — the server should not silently
   reinterpret the word.

⚠️ The trade this accepts: clients that hardcode `high` land on the slow path.
That is deliberate, per-request, and reversible by the caller.

Everything else is byte-identical to upstream.

Regenerate the diff at any time:

```bash
diff models-cache/qwen3.8-27b-fp8/chat_template.jinja \
     models/qwen3.8-27b/vllm/patches/qwen38-reasoning-effort-template/chat_template.jinja
```

## Drop when

Upstream Qwen accepts `high` as an alias (costs them nothing), **or** vLLM's
Anthropic router stops forwarding a `high` it cannot know is unsupported. Re-vendor
from the model dir if Qwen ships a corrected template, and re-run the drift guard.

Source: `Qwen/Qwen3.8-27B-FP8` @ `chat_template.jinja`, fetched 2026-08-16,
re-diffed against the live upstream file 2026-08-21 — the only difference is the
seven-line alias block at lines 48–54.

## 2026-09-27 — SGLang slugs, and `default_reasoning_effort`

The SGLang Qwen3.8 and ThinkingCap-Qwen3.8 slugs now mount this template too
(ThinkingCap ships the same 8,952-byte file), and the template gains one line:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default(default_reasoning_effort|default('xhigh')) %}
```

**Why.** SGLang ≤ 0.5.20 lets a `--default-chat-template-kwargs`
`reasoning_effort` override the request's own
([sglang#38104](https://github.com/sgl-project/sglang/issues/38104), fix in
[sglang#38338](https://github.com/sgl-project/sglang/pull/38338), open):

1. `_convert_to_internal_request` pops `reasoning_effort` out of the request's
   `chat_template_kwargs` into `request.reasoning_effort`;
2. `_process_messages` merges the server defaults with `setdefault` — the key was
   just popped, so the default goes back in;
3. the render does `extra_template_kwargs.update(request.chat_template_kwargs)`,
   so the default overwrites the request's value.

Measured on `sgl/qwen38-27b-dual-fast`, 2026-09-27, prompt tokens for "hi":
`low`, `medium`, `xhigh` and even `high` — top-level or in
`chat_template_kwargs` — all rendered the same 41-token prompt (the server
default `low`); only `enable_thinking: false` changed it (13).

The SGLang composes therefore set `default_reasoning_effort` as their server
default, a key SGLang does not pop, and the template reads it only when the
request names no effort. vLLM composes never pass it: rendered against the stock
template for unset / `low` / `medium` / `xhigh` / an invalid value / thinking off,
the output is byte-identical.

**Drop the `default_reasoning_effort` line when** sglang#38338 (or an equivalent)
ships in the pinned SGLang image — then the SGLang composes can go back to a plain
`reasoning_effort` default. The `high` → `xhigh` alias stays under its own drop rule
above.

## 2026-09-27 — tool schemas rendered with sorted keys

One more change, in the tools block:

```jinja
{{- tool | tojson(sort_keys=True) }}      {#- was: tool | tojson #}
```

**Why.** Qwen3.8 renders the tool definitions *first* in the prompt. A client,
gateway or MCP server that re-serialises a schema with its keys in a different
order therefore changes the prompt at its very start, and the engine's prefix
cache misses the whole conversation. Measured on vLLM dual-fast: one tool's keys
reordered → **0 of 29.6K tokens** cached (an 18 s re-prefill). SGLang normalises
schemas itself and was unaffected. The same fix is described for Qwen3.8 on vLLM in
doug.sh's "vLLM KV cache for agents" write-up (cache share 78 % → 95 % there).

**Scope.** Only the order of keys *inside* each tool's JSON changes, recursively
(`function`/`type`, `description`/`name`/`parameters`, `properties`/`required`/`type`,
…); the tools' order, their content and everything outside the tools block are
unchanged. Rendered against the previous vendored file in both engines' images
(transformers 5.12 / 5.17): prompts without tools are byte-identical for every
effort; with tools, output differs only in key order; tool-call history renders
identically.

**Drop when** upstream Qwen's template sorts keys itself, or every client in use is
known to serialise schemas stably (omp's built-in tools do — verified).

## 2026-09-27 — a system message after the first renders in place (#1447)

The stock template raises for any system message that isn't first:

```jinja
{%- if message.role == "system" %}
    {%- if not loop.first %}
        {{- raise_exception('System message must be at the beginning.') }}
```

The vendored copy renders it where it sits instead, as its own system turn,
with the first system message's content rules (no images or video); an empty one
renders nothing:

```jinja
{%- set content = render_content(message.content, false, true)|trim %}
{%- if content %}
    {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
{%- endif %}
```

**Why.** Claude Code (2.1.x) sends its `# Environment` block as a `role: "system"`
message after the first user turn. LiteLLM forwards it in place (to the engine's
`/v1/responses` on an `openai/` route, and in place on chat completions too), and
vLLM hands it to the template, so **every Claude Code request 400'd on the vLLM
Qwen3.8 slugs**, the first one included. SGLang's request handling let it through.
Reported by paulp83 on a 2× 5090 rig; reproduced here with the Claude Code CLI
2.1.283 (#1447).

**Why in place, not hoisted into the leading system turn.** A hoist rewrites the
front of the prompt whenever a new system entry appears mid-session, and the engine
then re-prefills the whole conversation. In place, the prompt stays append-only.
The approach is froggeric's Qwen-Fixed-Chat-Templates (v22.x), which renders later
system messages the same way; only this rule is taken, not the rest of that
template.

**Scope.** Rendered against the previous vendored file (transformers-style Jinja):
96 input combinations without a later system message (6 conversation shapes × 8
effort / thinking settings × with and without tools) are byte-identical. Only a
conversation that has a system message after the first changes, from a raised
error to a rendered turn. `scripts/tests/test-qwen38-template.sh` checks this rule
and the three changes above.

**Measured** on `vllm/qwen38-27b-dual-fast` with the Claude Code CLI 2.1.283 through
the gateway: a three-file find-and-read task (tool calls) answered in 17 s, and two
`--continue` follow-ups in about 1 s each, with 99 % of their prompt tokens
(33,696 of 33,846) served from the prefix cache.

**Drop when** upstream Qwen's template accepts a system message after the first.

## 2026-09-27 — `minimal` → `low` and `max` → `xhigh`

Two more aliases beside `high` → `xhigh`:

```jinja
{%- if resolved_reasoning_effort == 'minimal' %}
    {%- set resolved_reasoning_effort = 'low' %}
{%- elif resolved_reasoning_effort == 'max' %}
    {%- set resolved_reasoning_effort = 'xhigh' %}
{%- endif %}
```

**Why.** The wider OpenAI-compatible vocabulary is `none / minimal / low / medium /
high / xhigh / max`. Hermes Agent sends it as a top-level `reasoning_effort` (and
clamps its own `ultra` to `max`). Measured through the gateway on both engines:
`none` turns thinking off — vLLM and SGLang both map it to
`enable_thinking=false` before the template runs — and `low` / `medium` / `high` /
`xhigh` work, but **`minimal` and `max` reached this template and 400'd**
("Unexpected reasoning effort …").

**Mapping.** Each clamps to the nearest real rung: `minimal` to `low`, the weakest
(never to "off" — that is `none`'s job), and `max` to `xhigh`, the top.

**Scope.** Every input that doesn't send `minimal` or `max` renders byte-identically
to the previous vendored file (64 combinations of conversation shape, effort /
thinking setting and tools). `test-qwen38-template` checks both aliases.

**Measured** on `vllm/qwen38-27b-dual-fast` (patched): all seven levels answer;
`minimal` renders the same 41-token prompt as `low`, `max` the same 53-token prompt
as `xhigh`. Hermes Agent with `--reasoning max` and `--reasoning minimal` answered.

**Drop when** upstream Qwen's template accepts these names — together with the `high`
alias above.
