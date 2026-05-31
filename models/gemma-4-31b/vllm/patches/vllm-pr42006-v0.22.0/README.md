# vLLM PR #42006 — Gemma 4 MTP streaming multi-tool-call fix (lean diff-apply)

**Upstream PR:** [vllm-project/vllm#42006](https://github.com/vllm-project/vllm/pull/42006) (open)
**Target image:** pinned stock `vllm/vllm-openai:v0.22.0`
**Delivery:** boot-time diff-apply (`install_script`) — `install.sh`
**Carried by:** BOTH gemma duals (`vllm/gemma-bf16-mtp` + `vllm/gemma-int8-mtp`)

## What this fixes

On stock v0.22.0, a **streamed** chat response containing **2+ tool calls** drops the
`arguments` of every call except the last — the streaming gemma4 tool parser mishandles
the argument deltas across multiple tool-call segments. Non-streaming is unaffected.

**Confirmed live on this rig 2026-05-31** (stock v0.22.0, no overlay): a streamed
`get_weather(Tokyo)` + `get_weather(London)` + `get_time(Paris)` request returned
`get_weather()` / `get_weather()` (args lost) / `get_time({"city":"Paris"})` — only the
last call kept its args. The same request **non-streaming** returned all three with correct
args. #42006 adds `_extract_streaming_delta_segments` / `_split_delta_text_on_tool_tokens`
/ `_combine_delta_messages` to accumulate per-segment args correctly.

## Why both duals

`bf16-mtp` and `int8-mtp` both serve tool-bearing agent traffic with the same gemma4
parser, so both have the bug. (This *re-introduces* the only overlay on `bf16-mtp` — it
was briefly dropped 2026-05-31 on the "minimal surface" call, then brought back once the
bug was reproduced live.)

## Delivery

The PR's 2-hunk single-file diff (`pr42006-v0.22.0.patch`) applies cleanly to stock
v0.22.0's `tool_parsers/gemma4_tool_parser.py`. `install.sh` (idempotent, fail-loud)
applies it at boot before `vllm serve`. Drop when #42006 merges + lands in the pin.
Tracked in [`docs/UPSTREAM.md`](../../../../../../docs/UPSTREAM.md).
