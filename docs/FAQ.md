# FAQ

Common questions about club-3090. If your question isn't here, open a [GitHub Discussion](https://github.com/noonghunna/club-3090/discussions) — most things end up in this doc eventually.

## Hardware

### Can I use a 4090 instead of a 3090?

Yes — 4090 (Ada, sm_89) is strictly better than 3090 (Ampere, sm_86) for everything we ship. Slightly different kernel paths but no patches needed. Caveats: vLLM Genesis patches are tested on Ampere; tools should still work but TPS scaling is untested. Open an issue with numbers if you bench it.

### Can I use a 5090?

Should work for vLLM (Blackwell adds new kernels but back-compat). The Marlin pad-sub-tile-n fork we mount targets Ampere edge cases — on Blackwell you can probably drop the `/opt/ai/vllm-src/` mount. Not validated yet. We'd love numbers from a 5090 rig — use the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) issue template.

### Do I need NVLink?

No for the 1× / 2× baseline. The measured dual-card configs were built to work on PCIe-only consumer rigs, and custom all-reduce remains disabled in those composes.

If NVLink bridges are already installed, use them by pinning the process to a real linked pair. On the inspected four-card host, GPUs `0,1` are one pair and `2,3` are the other; cross-pair links are `SYS`. See [`QUAD_CARD.md`](QUAD_CARD.md) for the topology-aware quad variants. Do not add NVLink as a new dependency for ordinary dual-card users.

### Does this work on AMD / Intel / Apple Silicon?

vLLM: NVIDIA-only (CUDA). llama.cpp: yes — pick the right Docker image (`ghcr.io/ggml-org/llama.cpp:server-rocm` for AMD, `:server` for CPU-only, or build from source for Apple Silicon). Update the `image:` line in the compose. The flags (`--ngl`, `-fa on`, `--cache-type-k q4_0`) work identically across backends.

### Does this work on Windows / WSL2?

WSL2: yes, both engines. Make sure GPU passthrough is set up (`nvidia-smi` works inside WSL). Native Windows: vLLM doesn't support it; llama.cpp does — but use a native llama.cpp build, not Docker.

---

## Engine choice

### Why ship both vLLM and llama.cpp?

Different trades. vLLM is faster (51-89 TPS depending on config) and has full feature support (vision · tools · MTP spec-decode · streaming · reasoning). As of 2026-04-30 PM, **Cliff 1 (the 25K-token tool-prefill OOM) is closed on every shipped vLLM single-card variant** via Genesis PN8 on the FP8 path and PN12 anchor sidecar on the TQ3 paths. The remaining caveat is **Cliff 2** — single prompts above 50-60K still OOM in DeltaNet GDN forward, and that's a different memory class with no upstream fix yet. llama.cpp is slower (~21 TPS) but **passes every stress test cleanly** at full 262K context with vision and tools — including the big single prompts that vLLM single-card can't yet handle. See the launch frame: [vLLM dual = max throughput, llama.cpp single = max robustness](../README.md#tldr--what-this-is).

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
2. Wrong compose for your prompt shape (use the `docker-compose.yml` 48K default for chat — don't pick `long-vision.yml` if you don't need 198K)
3. Genesis tree drift — `git pull origin main` between bench runs can change AL by ±15%. We pin to commit `bf667c7` for this reason.

### My TPS dropped after switching to 198K context. Why?

It shouldn't, much — we measured 50.93 TPS narr at 192K vs 50.53 at 32K (within variance) on `long-vision.yml` pre-fix; the new 198K + 0.98 config is in the same range. If it dropped a lot, you're probably actually decoding *into* a long ctx (not just having KV pool reserved). Loaded-context decode is 2-4× cold short-prompt decode on any LLM. The TPS chart number is short-prompt cold; loaded numbers are in [BENCHMARKS.md](https://github.com/noonghunna/club-3090/blob/master/CHANGELOG.md).

### What's a "prefill cliff"?

VRAM-related OOM during prompt processing on single-card vLLM. Two cliffs documented:
- **Cliff 1** — historical: FFN intermediate buffer (`SiluAndMul` output, 138 MiB at `max_num_batched_tokens=4128 × intermediate_size=17408 × 2 bytes`) fresh-allocated per layer. Plus a related FA2 softmax_lse cap-leak ([Dao-AILab/flash-attention#1011](https://github.com/Dao-AILab/flash-attention/issues/1011)). **Closed on every shipped vLLM single-card variant** as of 2026-04-30 PM: `tools-text.yml` via Genesis PN8 (frees ~900 MiB on FP8 path); `long-vision.yml` and `long-text.yml` via the PN12 anchor sidecar (PR #13 to Sandermage's repo) plus a local P104 FA softmax_lse clamp. Full diagnostic: [docs/CLIFFS.md](CLIFFS.md).
- **Cliff 2** — DeltaNet GDN forward OOM at ~50-60K single-prompt regardless of mem-util. Different memory class, lives in `fla.ops` upstream, no file-replacement patch yet. Tracked in [UPSTREAM.md](UPSTREAM.md). Mitigation: dual-card TP=2 (verified at 237K) or llama.cpp single-card (262K, different engine).

For the full deep dive — empirical bisection, root-cause walk-through, who-can-fix-it landscape, and what we could do at any difficulty level — see [docs/CLIFFS.md](CLIFFS.md).

### vllm#40914 keeps coming up — what is it?

Sandermage's K+1 verify routing PR for vLLM. When it lands, the spec-verify cost we're paying on Ampere SM 8.6 (~22 TPS narrative regression vs pre-bug substrate) closes. Our default on `0.20.1rc1.dev16+g7a1eb8ac2 + Genesis v7.65 dev tip` will jump from ~50 narr to ~70 narr, matching what ampersandru measures on the older `dev21 + v7.13` cascade-prone substrate. We track it in [INTERNALS.md "Upstream tracker"](../models/qwen3.6-27b/INTERNALS.md).

### What's PN8?

A Genesis patch (`GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1`) added in v7.62.x — backport of vllm#40849 that makes the MTP draft head inherit the target model's online-quant config. We measured ~800-900 MiB freed on the FP8+MTP single-card path (`tools-text.yml`), which **closes Cliff 1 there**. No-op on TQ3 paths. Enabled by default in `tools-text.yml` since 2026-04-29; opt-in elsewhere via the env var if you want to test.

---

## Setup

### `bash scripts/setup.sh qwen3.6-27b` is downloading 20+ GB. Where does it go?

`<repo>/models-cache/` by default. Override with `MODEL_DIR=/path/to/your/scratch bash scripts/setup.sh qwen3.6-27b`. See [`.env.example`](../.env.example) for all env vars.

### My GPU isn't card 0 — how do I change it?

`CUDA_VISIBLE_DEVICES=2 bash scripts/launch.sh --variant vllm/default` (substitute your card index). For dual-card, pass two: `CUDA_VISIBLE_DEVICES=2,3`. The compose files inherit env from your shell.

### Container fails to start: "Free memory ... is less than desired GPU memory utilization"

Looks like:

```
ValueError: Free memory on device cuda:0 (22.76/24.0 GiB) on startup
is less than desired GPU memory utilization (0.97, 23.28 GiB).
```

vLLM's startup check reserves `mem-util × total VRAM` of *currently-free* VRAM before booting. If something else on the GPU is holding memory (X11 / Wayland compositor, leftover container, Python process, browser GPU acceleration), the check fails. Most common on `tools-text.yml` (0.97) and the long-* variants (0.98 / 0.985).

Two fixes:
1. **Free the VRAM** (preferred). `nvidia-smi` shows what's holding it. Common: log out of GUI, stop a leftover container (`docker rm -f $(docker ps -aq --filter "name=vllm-")`), or kill orphaned `python` processes.
2. **Lower mem-util** in the compose. e.g. on `tools-text.yml`: drop `--gpu-memory-utilization 0.97` to `0.94` and reduce `--max-model-len` proportionally (75K → ~70K). Loses ~6K context but works on any rig.

The 0.97 / 0.98 / 0.985 defaults assume a headless rig with ≥23.3 GiB consistently free. If you're running a desktop session on the same card, `0.92`–`0.94` is the safer ceiling.

### Can I run multiple variants at once on the same machine?

You'd need different ports per variant. Set `PORT=9876` in `.env` (or pass inline: `PORT=9876 bash scripts/switch.sh vllm/default`) — every shipped compose now reads `${PORT}` for the host-side port mapping. Watch VRAM — two configs simultaneously typically don't fit on 24 GB.

### Will this work behind Open WebUI?

Yes. Add a connection in Open WebUI's Settings → Connections → OpenAI: base URL `http://localhost:8020/v1`, any non-empty API key, model `qwen3.6-27b-autoround`. See [docs/EXAMPLES.md](EXAMPLES.md#open-webui).

### Will this work with VS Code GitHub Copilot LLM Gateway?

Yes, but you need a compose with **≥48K context** — Copilot's LLM Gateway sends ~20K tokens of tool-schema preamble (50+ VS Code tools enumerated in a structured-outputs JSON schema) on every request, which alone consumes most of a small context budget. Use `tools-text.yml` (75K + fp8 + PN8 enabled — Cliff 1 closed):

```bash
bash scripts/switch.sh vllm/tools-text
```

There's a second wrinkle: Copilot's LLM Gateway sometimes sends very low `max_tokens` (e.g. 64) on probe-style requests. With `tool_choice: required` (which Copilot enforces via `minItems: 1` on its structured-outputs schema), the model must emit a tool-call JSON that wraps a real argument like a file path — and 64 tokens isn't enough to fit `{"name": "read_file", "parameters": {"filePath": "/long/abs/path"}}`. The truncated JSON arrives at the gateway as "empty response." If you see this pattern, it's a client-side limit, not the server. Other OpenAI-compat clients (Cline / Continue.dev / Cursor) tend to send realistic max_tokens by default and don't hit this.

**Server-side fix landed 2026-04-29:** the Genesis P68/P69 long-context tool-adherence patches were silently overriding `tool_choice: auto → required` and injecting "must use a tool" reminders whenever prompt > 8000 chars. That made greetings + clarifying questions stall on every IDE-agent setup (Cline, Cursor, OpenCode, and Copilot Gateway combined). We disabled both in `tools-text.yml`. Behavior now: greeting → plain-text reply ("Hello! How can I help you today?"); tool request → clean `read_file({"path": "..."})` call. P64 and PN8 stay enabled (real targeted bugfixes, no user-intent override).

Background + bisection: [club-3090 #2](https://github.com/noonghunna/club-3090/issues/2#issuecomment-4346345554).

---

## Community / contribution

### Can I add my benchmark numbers from a different rig?

Please do — open an issue using the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) template. We collect cross-rig data points in BENCHMARKS for community signal.

### Found a bug — what should I include?

The [bug report template](https://github.com/noonghunna/club-3090/issues/new?template=bug-report.yml) asks for the data we always need: `docker logs --tail 100`, `verify-full.sh` output, `nvidia-smi`, your compose variant, and the repo commit. Skipping these means the first reply will just ask for them, costing you a round-trip.

### How do I bump Genesis to a newer commit?

`GENESIS_PIN=<new-commit-sha> bash scripts/setup.sh qwen3.6-27b` and re-run `bash scripts/verify-full.sh` to confirm tools still work. Don't bump in production without re-running the verify suite — Genesis releases sometimes change spec-verify routing in ways that affect tool-call extraction.

---

## Troubleshooting

Quick recognition guide for common failure modes:

- **Container dies at boot with `GPTQ_MARLIN_MIN_THREAD_N (64) > out_features`** — dual-card vllm#40361 patch didn't apply. Confirm `/opt/ai/vllm-src/` exists with the patched marlin kernel files.
- **Container dies during DFlash boot** — vllm#40334 dtype mismatch. Verify the compose has `--dtype bfloat16`.
- **Tool calls return `<tool_call>` as plain text** — Genesis didn't apply. Check `Genesis Results: 27 applied` in logs (boot-time).
- **OOM during prefill at 60K+ tokens** — single-card Cliff 2 (DeltaNet GDN forward). Switch to lower max-model-len, dual-card, or llama.cpp + q4_0 KV.
- **OOM during prefill at 25K+ tool response** — historically Cliff 1 on TQ3 paths. **Closed since 2026-04-30 PM** via PN12 anchor sidecar on `long-vision.yml` / `long-text.yml`. If you're hitting it, check your compose has the sidecar wired in (`patch_pn12_ffn_pool_anchor.py` in entrypoint).
- **"Empty response" through VS Code Copilot LLM Gateway** — Copilot sends ~20K tokens of tool schemas + sometimes uses `max_tokens=64` which truncates tool-call JSON. Switch to `tools-text.yml` (75K) and check Copilot's max_tokens setting. See [#2](https://github.com/noonghunna/club-3090/issues/2) for full debug-log analysis.
- **Per-stream TPS lower than expected** — re-run `bench.sh` with 3+ warmups + 5 measured runs first. Run-to-run variance is ~5%.

If none match, open an issue with `docker logs <container> 2>&1 | tail -200` + `nvidia-smi` — see [`bug-report.yml` template](https://github.com/noonghunna/club-3090/issues/new?template=bug-report.yml).

---

## See also

- [README](../README.md) — top-level overview + quick start
- [docs/SINGLE_CARD.md](SINGLE_CARD.md) — 1× 3090 deployment menu
- [docs/DUAL_CARD.md](DUAL_CARD.md) — 2× 3090 deployment menu
- [models/qwen3.6-27b/README.md](../models/qwen3.6-27b/README.md) — model-specific reference
- [models/qwen3.6-27b/INTERNALS.md](../models/qwen3.6-27b/INTERNALS.md) — engineering deep dive
- [docs/EXAMPLES.md](EXAMPLES.md) — Python / TS / curl client snippets
- [docs/HARDWARE.md](HARDWARE.md) — Ampere notes, NVLink, power caps
- [docs/GLOSSARY.md](GLOSSARY.md) — TPS / KV / MTP / TP / etc. plain-language definitions
