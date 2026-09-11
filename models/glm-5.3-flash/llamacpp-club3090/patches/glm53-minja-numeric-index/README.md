# glm53-minja-numeric-index — GLM-5.3-Flash chat template, minja-compatible

**What it fixes:** every request carrying `tools` returned HTTP 400 on llama.cpp, and
`/props` reported `supports_tools: false`, `supports_tool_calls: false`.

```
Unable to generate parser for this template. Automatic parser generation failed:
While executing CallExpression at line 163, column 57 in source
```

Reported by @paulp83 on [club-3090#1250](https://github.com/noonghunna/club-3090/issues/1250)
(DevQuasar Q3_K_M, 2× 3090), and independently upstream on
[ggml-org/llama.cpp#27754](https://github.com/ggml-org/llama.cpp/pull/27754) (2026-09-07,
`/props` showing both tool caps false) — where it received no reply.

## Root cause: minja does not implement numeric dotted attribute access (`x.0`)

⚠️ **The line in the error message is the crash site, NOT the cause.** Two earlier readings of
this bug were wrong and are recorded here so nobody re-derives them:

1. ❌ *"minja cannot evaluate `.items()`"* — false. `models/templates/GLM-4.7-Flash.jinja`
   contains the identical `_args.items()` construct, is exercised by llama.cpp's own autoparser
   tests, and passes.
2. ❌ *"the caps probe under-detects `supports_object_arguments` because the template iterates
   rather than indexing"* — false. Adding a named `tool.function.name` access changes nothing;
   the probe never gets that far.

The actual chain:

| step | where | what happens |
|---|---|---|
| 1 | `common/jinja/parser.cpp` | on `.`, `parse_member_expression` calls `parse_primary_expression`, so `content.0` becomes a **non-computed** member whose property is the number literal `0` |
| 2 | `common/jinja/runtime.cpp` | a non-computed member requires an **identifier** → throws. Jinja2 instead treats `x.0` as `x[0]` |
| 3 | `common/jinja/caps.cpp` | the object-arguments probe throws → `return; // Nothing can be inferred` → `supports_object_arguments` stays false |
| 4 | `common/jinja/caps.cpp` | the string-arguments probe then runs and throws at `.items()` → sets `supports_tool_calls=false` **and** `supports_tools=false` |
| 5 | `common/chat.cpp` | `func_args_not_string()` is gated on `supports_object_arguments`, so it never runs → `arguments` stays a JSON **string** |
| 6 | `common/chat.cpp` | the autoparser render then hits `.items()` on that string → throws → wrapped as the 400 above |

GLM-4.7-Flash has **zero** `x.0` accesses, so its probe succeeds and it is unaffected. No
template in llama.cpp's `models/templates/` uses `ident.0.`, so this path is untested upstream.

## The patch — 5 lines

Four are `.0.` → `[0].`, which is **semantics-preserving** (Jinja2 defines `x.0` as `x[0]`).
The fifth guards the tools loop, because the caps probe passes `tools = [null]` and
`'function' in tool` trips on it.

```diff
-{% for tool in tools %}
+{% for tool in tools if tool %}
-...m.content.0.type == "tool_reference"...        +...m.content[0].type...
-...tr.output.0.type == "tool_reference"...        +...tr.output[0].type...
-...m.content.0.output is defined...               +...m.content[0].output...
-...entry.output.0.type == "tool_reference"...     +...entry.output[0].type...
```

Lines 39, 81, 85, 103, 230. `fix.diff` in this directory is the exact patch.

**The wire format is untouched** — tool calls still render
`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`. That matters: change
it and the generated parser stops matching what the model actually emits.

## Verification (no model load required)

```bash
# build once, CPU-only
cmake -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-template-analysis -j
./build/bin/llama-template-analysis --template-file chat_template.jinja
```

| capability | GGUF-embedded | **this override** |
|---|:--:|:--:|
| `supports_tools` | false | **true** |
| `supports_tool_calls` | false | **true** |
| `supports_parallel_tool_calls` | false | **true** |
| `supports_string_content` | false | **true** |

Also confirmed via `llama-debug-template-parser` (the real autoparser path): the patched template
yields `tool_mode: TAG_WITH_TAGGED`, `per_call_start '<tool_call>'`, and a full PEG parser + GBNF
with the `<arg_key>`/`<arg_value>` rules. And under reference Jinja2 3.1.2 (trim_blocks,
lstrip_blocks, loopcontrols), original ≡ patched byte-identical across 8 cases including the
two-call sort path, reasoning_content, typed content and list-of-outputs tool content.

## ⚠️ Not verified

- **No live boot.** GLM-5.3-Flash is ~140 GiB; this has not been served end-to-end with
  `--chat-template-file`. Everything above is static analysis plus the engine's own autoparser.
- The production pin is `server-cuda-b10236`; the analysis build is from a local checkout whose
  minja may differ. The error text and byte offset match exactly, so the mechanism holds, but the
  builds are not proven identical.

## Upstream

Not fixed upstream as of 2026-09-11. The right fix is in minja — either implement `x.0` as
`x[0]` per Jinja2, or have the caps probe degrade gracefully instead of inferring nothing when a
probe throws. Closest prior art is
[#28509](https://github.com/ggml-org/llama.cpp/issues/28509) (Gemma 4 misclassified
`supports_typed_content`), same subsystem, closed as completed.

⚠️ ggml-org's `AGENTS.md` bars autonomous agent contributions — this is reported upstream by the
maintainer, not from here.
