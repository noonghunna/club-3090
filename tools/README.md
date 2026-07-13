# Tools

| Tool | Purpose |
|---|---|
| [`repack_prefix.py`](repack_prefix.py) ([docs](repack_prefix.md)) | Rename the `model.language_model.` ↔ `model.` prefix in safetensors checkpoint headers, stream-copying data bytes verbatim. Also re-prefixes `config.json` quantization keys. **For rolling your own AutoRound quant — not needed for shipped checkpoints.** |
| [`kv-calc.py`](kv-calc.py) | KV-cache calculator — estimate VRAM usage for a given model / ctx / batch. |
| [`bump-engine-nightlies.sh`](bump-engine-nightlies.sh) | Bump engine image tags to the latest nightly across all composes. |
| [`serve-cockpit/`](serve-cockpit/) | Textual TUI for inspecting and switching serving stacks. |
| [`charts/`](charts/) | Performance / VRAM chart generation scripts. |
| [`model-switch/`](model-switch/) | Helper scripts for model switching logic. |
| [`residency-instrument/`](residency-instrument/) | VRAM residency instrumentation. |
| [`test-console/`](test-console/) | Interactive test harnesses. |
| [`tui-core/`](tui-core/) | Shared TUI widgets. |
| [`grammar-eval/`](grammar-eval/) | Grammar evaluation tooling. |
