# Muse Glimmer 30B — Changelog

## 2026-08-14 — Initial community contribution (single-rig validation)

Added llama.cpp config for Muse Glimmer 30B with DFlash spec-decode on a single RTX 3090, validated on WSL2 Ubuntu (llama.cpp build 10349, commit 62bf73d25). Also runs on native Windows.

**Measured (RTX 3090, sm_86, streaming decode):**
- `dflash-131k`: 47.3 t/s long / 67.1 short (post-sweep, DFlash on) @ 131K ctx + vision (~20.9 GB VRAM)
- DFlash off: 35.5 t/s long (DFlash adds +10–25%)
- Prefill: ~860 tok/s @ 80–100K, ~679 tok/s @ 114K fresh

**Not yet run (N/A justifications):**
- verify-full.sh / verify-stress.sh: N/A — single-profile config validated by sustained use (multi-turn tool calling, vision, long-context) over multiple days. No Docker container to smoke-test.
- SOAK_MODE=continuous: N/A — same reason; however, multi-turn sessions up to 5 turns at ~220 ms/turn with growing cached_tokens confirm no accumulating-context cliff at this context size.
- bench.sh canonical run: N/A — script targets Docker containers. Numbers above are from `/metrics` during real workloads (n>30 requests).

**Key findings:**
- DFlash pays off on dense+SWA architectures (acceptance 0.14, mean length 3.16 → +10–25% net)
- DFlash regresses on BeeLlama/Qwen3.6 under agent workloads (tool marker suppression → 97→27 t/s)
- Draft KV must stay f16 (quantizing collapses acceptance to near-zero; upstream #25725, fix #25823 in build 10349)
- `--jinja` is required for tool calling + reasoning routing
- `reasoning_strength=low` gives ~2× speed without losing correctness

**Contributor:** @Isaac-opz (single rig, WSL2 Ubuntu on Windows 10)
