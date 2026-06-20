# DiffusionGemma — Ampere/TP + agentic fixes for the official `vllm/vllm-openai:gemma` image

Production compose ([`base.yml`](../../compose/dual/fp8/base.yml)) bind-mounts **six files**
from this directory (no #45588 boot install — rolled back 2026-06-20 after streaming
tool-call regression; see [`patch_docs/diffusionGemma_streaming_tool_regression.md`](./patch_docs/diffusionGemma_streaming_tool_regression.md)).

| File → mounts over | Fix |
|---|---|
| `marlin.py` + `marlin_utils_fp8.py` | sm_86 fp8 Marlin K-pad (warmup wall at K=352/1056) |
| `diffusion_gemma.py` | TP-vocab soft-embed + dtype fix |
| `gemma4_reasoning_parser.py` | Streaming `<\|channel>` start-token strip |
| `gemma4_tool_parser.py` | Paren recovery + quote-aware args + no-swallow |
| `tool_chat_template_gemma4.jinja` | History CoT replay strip |

## Evidence (`patch_docs/`)

| Doc | What it covers |
|---|---|
| [`diffusionGemma_ablation_pr443.md`](./patch_docs/diffusionGemma_ablation_pr443.md) | Mount-matrix ablation for PR #443 (parsers vs template) |
| [`diffusionGemma_reasoning_stream_fix.md`](./patch_docs/diffusionGemma_reasoning_stream_fix.md) | Streaming `<\|channel>` leak repro + fix |
| [`diffusionGemma_empty_after_tools.md`](./patch_docs/diffusionGemma_empty_after_tools.md) | Paren-form tool calls → empty turn |
| [`diffusionGemma_tool_args_json.md`](./patch_docs/diffusionGemma_tool_args_json.md) | Plain-quoted values → invalid JSON args |
| [`diffusionGemma_history_cot_replay.md`](./patch_docs/diffusionGemma_history_cot_replay.md) | History CoT re-injection in chat template |
| [`diffusionGemma_streaming_tool_regression.md`](./patch_docs/diffusionGemma_streaming_tool_regression.md) | **#45588 ParserEngine rejected** — streaming tool leak |

vLLM #45588 ParserEngine experiment code lives on branch `experiment/dgemma-parser-engine-45588`
(not wired in production compose).
