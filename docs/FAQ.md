# FAQ

Common questions about club-3090. If your question isn't here, open a [GitHub Discussion](https://github.com/noonghunna/club-3090/discussions) — most things end up in this doc eventually.

## Hardware

### Can I use a 4090 instead of a 3090?

Yes — 4090 (Ada, sm_89) is strictly better than 3090 (Ampere, sm_86) for everything we ship. Slightly different kernel paths but no patches needed. Caveats: vLLM Genesis patches are tested on Ampere; tools should still work but TPS scaling is untested. Open an issue with numbers if you bench it.

### Can I use a 5090?

Should work for vLLM (Blackwell adds new kernels but back-compat). The Marlin pad-sub-tile-n fork we mount targets Ampere edge cases — on Blackwell you can probably drop the `/opt/ai/vllm-src/` mount. Not validated yet. We'd love numbers from a 5090 rig — use the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) issue template.

### Do I need NVLink?

No. Our dual-card configs use PCIe-only, no NVLink. Custom all-reduce is disabled in the composes. NVLink would help dual-card TPS but it's not required, and the user has explicitly declined NVLink bridges as a default — adding the dependency would exclude most consumer rigs.

### Does this work on AMD / Intel / Apple Silicon?

vLLM: NVIDIA-only (CUDA). llama.cpp: yes — pick the right Docker image (`ghcr.io/ggml-org/llama.cpp:server-rocm` for AMD, `:server` for CPU-only, or build from source for Apple Silicon). Update the `image:` line in the compose. The flags (`--ngl`, `-fa on`, `--cache-type-k q4_0`) work identically across backends.

### Does this work on Windows / WSL2?

WSL2: yes, both engines. Make sure GPU passthrough is set up (`nvidia-smi` works inside WSL). Native Windows: vLLM doesn't support it; llama.cpp does — but use a native llama.cpp build, not Docker.

---

## Engine choice

### Why ship both vLLM and llama.cpp?

Different trades. vLLM is faster (51-89 TPS depending on config) and has full feature support (vision · tools · MTP spec-decode · streaming · reasoning), but its long-context configurations OOM on big tool returns (Cliff 1) or single prompts above 50-60K (Cliff 2). llama.cpp is slower (~21 TPS) but **passes every stress test cleanly** at full 262K context with vision and tools. For real-world tool-using agents (Claude Code, Cline, Hermes) that routinely send 25K+ tool messages, llama.cpp is the only single-card path that doesn't crash. See the launch frame: [vLLM dual = max throughput, llama.cpp single = max robustness](../README.md#tldr--what-this-is).

### Why not Ollama?

Ollama wraps llama.cpp with a different model registry and slightly easier UX. It's fine for chat. Two reasons we don't ship it:
1. Ollama doesn't expose all llama.cpp flags we need (`--cache-type-k q4_0`, `--mmproj`, `--spec-type ngram-mod`, custom `--parallel`).
2. Ollama's model registry doesn't have the exact Unsloth GGUF quants we ship (UD-Q3_K_XL).
You can run Ollama against an Unsloth GGUF manually, but at that point you've reimplemented our llama.cpp compose with a different wrapper.

### Why not LM Studio?

LM Studio is GUI-driven and great for hobbyist use. We ship CLI/Docker because:
- Reproducibility — pinned image SHAs + Genesis commit make exact bench runs across machines possible
- Headless deployment — homelab racks, dev backends
- Tool-call extraction across both engines on this exact model is non-trivial; LM Studio's defaults haven't been validated

Use LM Studio if you prefer a GUI and don't need the engineering. Use this repo if you want a tested config that another club-3090 user can match exactly.

### Why MTP and not EAGLE?

We tried EAGLE — it's blocked on Qwen3-Next (the family Qwen3.5/3.6 belong to) by DeltaNet hybrid attention's lack of KV rollback support in vLLM/SGLang. MTP works because it's a different protocol (multi-token prediction at draft-head level, not a separate draft model). See [INTERNALS.md "Speculative decoding"](../models/qwen3.6-27b/INTERNALS.md) for the full forensic chain. **Re-test triggers:** if vllm#39931 lands or DeltaNet rollback support arrives upstream, EAGLE becomes viable again.

### Why not GGUF on vLLM for this model?

Multiple gates blocked. Qwen3.6-27B GGUF on vLLM hits a chain of "fixed but-not-quite" issues — multimodal config routing, ParallelLMHead skip, the `Qwen35TensorProcessor._reverse_reorder_v_heads` weight loader producing garbage output on the 27B layout (transformers PR #45283 only validated on 0.8B). Tracked in [INTERNALS.md](../models/qwen3.6-27b/INTERNALS.md#qwen36-27b-gguf-on-vllm). Use llama.cpp for GGUF on this model.

### Why AutoRound INT4 not GPTQ / AWQ?

AutoRound (Lorbus) gave us +9% TPS over AWQ on this model. GPTQ has a similar quality bar but the AWQ + DFlash path failed (pad-Marlin × aux-layer interaction). AutoRound + Genesis + MTP is the production-validated path. AWQ is documented as a fallback for users who can't use AutoRound.

---

## Performance

### Why is single-card TPS lower than I expected?

Look at the [TPS chart](../README.md#measured-tps-at-a-glance) — single-card vLLM is 51-55 TPS narrative / 67-70 code at 48K, which beats most consumer-3090 numbers we've seen reported. If you're seeing materially lower, the most common causes are:
1. Power cap < 230 W (this rig benches at 230 W; 280 W gives ~+5%, 350 W ~+10%)
2. Wrong compose for your prompt shape (use `vllm/fast-chat.yml` for 20K chat, not `long-vision.yml`)
3. Genesis tree drift — `git pull origin main` between bench runs can change AL by ±15%. We pin to commit `bf667c7` for this reason.

### My TPS dropped after switching to 192K context. Why?

It shouldn't, much — we measured 50.93 TPS narr at 192K vs 50.53 at 32K (within variance) on `long-vision.yml`. If it dropped a lot, you're probably actually decoding *into* a long ctx (not just having KV pool reserved). Loaded-context decode is 2-4× cold short-prompt decode on any LLM. The TPS chart number is short-prompt cold; loaded numbers are in [BENCHMARKS.md](https://github.com/noonghunna/club-3090/blob/master/CHANGELOG.md).

### What's a "prefill cliff"?

VRAM-related OOM during prompt processing on single-card vLLM. Two cliffs documented:
- **Cliff 1** — FFN intermediate-buffer activation peak (138 MiB allocate, `intermediate_size × max-num-batched-tokens`). Fires on long-ctx single-card composes at high mem-util (>0.95) when the prefill batch needs the activation buffer. **Closes on `tools-text.yml`** (FP8 KV path) since 2026-04-29 via Genesis PN8 (frees ~900 MiB at boot — enough headroom for the activation peak). Still fires on TQ3 paths (`long-vision.yml`, `long-text.yml`) — PN8's quant-config propagation doesn't reach the activation buffer there. Documented at single-3090 #1.
- **Cliff 2** — DeltaNet GDN forward OOM at ~50-60K single-prompt regardless of mem-util, fires on long single prompts.
Both are in `fla.ops` and don't have file-replacement patches. Mitigation: `vllm/default` (48K + 0.92) for tools, dual-card or llama.cpp for long ctx.

### vllm#40914 keeps coming up — what is it?

Sandermage's K+1 verify routing PR for vLLM. When it lands, the spec-verify cost we're paying on Ampere SM 8.6 (~22 TPS narrative regression vs pre-bug substrate) closes. Our default on `dev205 + Genesis v7.62.x` will jump from ~51 narr to ~70 narr, matching what ampersandru measures on the older `dev21 + v7.13` cascade-prone substrate. We track it in [INTERNALS.md "Upstream tracker"](../models/qwen3.6-27b/INTERNALS.md).

### What's PN8?

A Genesis patch (`GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1`) added in v7.62.x — backport of vllm#40849 that makes the MTP draft head inherit the target model's online-quant config. We measured ~800-900 MiB freed on FP8+MTP single-card paths (`tools-text.yml`, `fast-chat.yml`), which **closes Cliff 1 on `tools-text.yml`**. No-op on TQ3 paths. Enabled by default in our two FP8 composes since 2026-04-29; opt-in elsewhere via the env var if you want to test.

---

## Setup

### `bash scripts/setup.sh qwen3.6-27b` is downloading 20+ GB. Where does it go?

`<repo>/models-cache/` by default. Override with `MODEL_DIR=/path/to/your/scratch bash scripts/setup.sh qwen3.6-27b`. See [`.env.example`](../.env.example) for all env vars.

### My GPU isn't card 0 — how do I change it?

`CUDA_VISIBLE_DEVICES=2 bash scripts/launch.sh --variant vllm/default` (substitute your card index). For dual-card, pass two: `CUDA_VISIBLE_DEVICES=2,3`. The compose files inherit env from your shell.

### Can I run multiple variants at once on the same machine?

You'd need different ports per variant. Edit `ports:` in the second compose (default is `8020:8000` → change to `8021:8000`). Watch VRAM — two configs simultaneously typically don't fit on 24 GB.

### Will this work behind Open WebUI?

Yes. Add a connection in Open WebUI's Settings → Connections → OpenAI: base URL `http://localhost:8020/v1`, any non-empty API key, model `qwen3.6-27b-autoround`. See [docs/EXAMPLES.md](EXAMPLES.md#open-webui).

### Will this work with VS Code GitHub Copilot LLM Gateway?

Yes, but you need a compose with **≥48K context**, not the 20K `fast-chat.yml` default — Copilot's LLM Gateway sends ~20K tokens of tool-schema preamble (50+ VS Code tools enumerated in a structured-outputs JSON schema) on every request, which alone exceeds fast-chat's 20K cap. Use `tools-text.yml` (75K + fp8 + PN8 enabled — Cliff 1 closed):

```bash
bash scripts/switch.sh vllm/tools-text
```

There's a second wrinkle: Copilot's LLM Gateway sometimes sends very low `max_tokens` (e.g. 64) on probe-style requests. With `tool_choice: required` (which Copilot enforces via `minItems: 1` on its structured-outputs schema), the model must emit a tool-call JSON that wraps a real argument like a file path — and 64 tokens isn't enough to fit `{"name": "read_file", "parameters": {"filePath": "/long/abs/path"}}`. The truncated JSON arrives at the gateway as "empty response." If you see this pattern, it's a client-side limit, not the server. Other OpenAI-compat clients (Cline / Continue.dev / Cursor) tend to send realistic max_tokens by default and don't hit this.

Background + debug-log analysis: [club-3090 #2](https://github.com/noonghunna/club-3090/issues/2).

---

## Community / contribution

### Can I add my benchmark numbers from a different rig?

Please do — open an issue using the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) template. We collect cross-rig data points in BENCHMARKS for community signal.

### Found a bug — what should I include?

The [bug report template](https://github.com/noonghunna/club-3090/issues/new?template=bug-report.yml) asks for the data we always need: `docker logs --tail 100`, `verify-full.sh` output, `nvidia-smi`, your compose variant, and the repo commit. Skipping these means the first reply will just ask for them, costing you a round-trip.

### How do I bump Genesis to a newer commit?

`GENESIS_PIN=<new-commit-sha> bash scripts/setup.sh qwen3.6-27b` and re-run `bash scripts/verify-full.sh` to confirm tools still work. Don't bump in production without re-running the verify suite — Genesis releases sometimes change spec-verify routing in ways that affect tool-call extraction.

---

## See also

- [README](../README.md) — top-level overview + quick start
- [models/qwen3.6-27b/README.md](../models/qwen3.6-27b/README.md) — model-specific variants + VRAM diagram
- [models/qwen3.6-27b/USE_CASES.md](../models/qwen3.6-27b/USE_CASES.md) — workload → recommended compose
- [models/qwen3.6-27b/INTERNALS.md](../models/qwen3.6-27b/INTERNALS.md) — engineering deep dive
- [docs/EXAMPLES.md](EXAMPLES.md) — Python / TS / curl client snippets
- [docs/HARDWARE.md](HARDWARE.md) — Ampere notes, NVLink, power caps
- [docs/GLOSSARY.md](GLOSSARY.md) — TPS / KV / MTP / TP / etc. plain-language definitions
