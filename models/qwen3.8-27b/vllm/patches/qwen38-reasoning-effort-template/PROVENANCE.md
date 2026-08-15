# qwen38-reasoning-effort-template

Vendored copy of `Qwen/Qwen3.8-27B-FP8`'s own `chat_template.jinja` with a
**single semantic change**: `reasoning_effort: "high"` is aliased onto Qwen3.8's
top rung `xhigh` instead of raising.

## The defect

Qwen3.8 renamed the conventional top reasoning rung. Its ladder is:

    low  <  medium  <  xhigh        (xhigh is also the default)

There is no `high`. The stock template hard-rejects anything else:

```jinja
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort ...') }}
```

Both OpenAI and Anthropic use `low / medium / high`, so a generic client that
sends the standard top rung gets a Jinja `TemplateError` surfaced as **HTTP 500**,
not a degraded-but-working response. Observed on this stack 2026-08-14 against
vLLM's Anthropic-compatible `POST /v1/messages?beta=true` endpoint, which is
exactly the surface a drop-in Claude-API client talks to.

The failure is silent-ish in the worst way: `/v1/models` returns 200 and the
model is demonstrably healthy, so the endpoint looks up while every thinking-mode
chat request fails.

## The change

Six lines inserted before the validation (line 48). `high` maps to `xhigh` rather
than `medium` because `high` is the TOP rung in the standard 3-rung ladder and
`xhigh` is Qwen3.8's top rung — this preserves caller intent and matches the
template's own default. Everything else is byte-identical to the shipped template.

Regenerate the diff at any time:

```bash
diff models-cache/qwen3.8-27b-fp8/chat_template.jinja \
     models/qwen3.8-27b/vllm/patches/qwen38-reasoning-effort-template/chat_template.jinja
```

## Drop when

Upstream Qwen accepts `high` as an alias (costs them nothing) **or** vLLM's
Anthropic router stops forwarding a `high` it cannot know is unsupported. Re-vendor
from the model dir if Qwen ships a corrected template, and re-run the drift guard.

Source: `Qwen/Qwen3.8-27B-FP8` @ chat_template.jinja, fetched 2026-08-14.
