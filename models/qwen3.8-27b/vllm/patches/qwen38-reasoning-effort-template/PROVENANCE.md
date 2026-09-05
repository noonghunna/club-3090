# qwen38-reasoning-effort-template

Vendored copy of Qwen3.8-27B's own `chat_template.jinja` with a **single semantic
change**: `reasoning_effort: "high"` is mapped onto `medium` instead of raising.

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

Seven lines inserted before the validation. `high` maps to **`medium`**, the
un-nudged baseline — the rung with no template branch and no injected preamble.

`xhigh` was the other candidate, and it is the wrong one here. It is Qwen3.8's
top rung, so it looks like the intent-preserving choice for a caller asking for
the standard top rung — but it injects a "think carefully, validate assumptions,
consider alternatives" system preamble that is slow and timeout-prone, and a
Claude-API client sends `high` on **every** request. Mapping to the top rung
would therefore put all agent traffic into the slowest reasoning mode by
default, on slugs that ship `max_num_seqs=1`. medium keeps the request served
and un-nudged; a caller who genuinely wants the deep mode can still ask for
`xhigh` by name.

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
