# Single 3090 — what fits, how to run it

You have **one RTX 3090 (24 GB VRAM)**. This page is the front door for picking a config and knowing what to expect. The model-specific deep dives (quants, Genesis patches, engine internals) live elsewhere — links at the bottom.

---

## ⚠️ Critical — read first if you're running an agentic coding client

If your workload is **hermes / openhands / OpenCode / Cline / Roo / OpenClaw / Aider / Cursor with retained context**, single-card vLLM is **not safe** as of 2026-05-03. You will hit a hardware-physical cliff at ~21-26K accumulated multi-turn context regardless of which single-card vLLM variant you pick. Validated across all six shipped single-card vLLM composes.

Symptoms users report: "performance degrades after ~20 turns", "throughput drops to 0", "engine becomes unresponsive then 500s", "OOM after 4-5 turns". Same root cause — see [#41](https://github.com/noonghunna/club-3090/issues/41) for the full validation matrix.

**Two safe paths for these workloads:**

| Have | Run | Why it works |
|---|---|---|
| 2× 3090 (any topology, NVLink optional) | `bash scripts/switch.sh vllm/dual` | TP=2 splits the failing kernel's working set across both cards. Validated PASS at v2 continuous soak; 111+ TPS p50 decode. |
| 1× 3090 only | `bash scripts/switch.sh llamacpp/default` | Different engine, different GDN kernel, different memory allocator. Cliff doesn't exist on this path. 262K context, ~21 TPS — slower decode but cliff-immune. |

Neither acceptable? Two single-card vLLM mitigations: (a) cap session context at <15K via app-layer rolling summarization, OR (b) accept periodic engine restarts. **Mem-util tuning, MTP-off, and `max-num-batched-tokens` adjustments do not close it** — all tested. The cliff is in `chunk_gated_delta_rule_fwd`'s simultaneous live-tensor set (~500 MiB at T=4128) which doesn't fit alongside accumulated KV + model + workspace on a 24 GB card.

A Genesis sidecar fix (streaming refactor) is being filed with Sandermage. ETA 2-4 weeks if accepted. This section will be updated when it ships.

For workloads that **don't** accumulate context across turns (single-shot RAG, simple chat, batch processing), single-card vLLM is fine — see the table below.

---

## TL;DR — pick by workload

Five recommended options on Genesis v7.69 + vllm#35975 backport (2026-05-02 PM):

| What you're doing | Compose | Max ctx | Narr / Code TPS | VRAM (24 GB / card) |
|---|---|---|---|---|
| **Long ctx + vision** (chat, agents, image input) | [`long-vision.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.long-vision.yml) | **145K** | 50 / 66 | ~23.0 GB (mem-util 0.95) |
| **Long ctx, text-only — Balanced MTP** ⭐ (RAG, codebase, IDE agents, default) | [`long-text.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.long-text.yml) | **180K** | 50 / 67 | ~22.3 GB (mem-util 0.93) |
| **Long ctx, text-only — Max-context** (long single-shot RAG / codebase analysis) | [`long-text-no-mtp.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.long-text-no-mtp.yml) (NEW) | **200K** | TBD/TBD (slow decode, no MTP) | ~21.0 GB (mem-util 0.95) |
| **Bounded thinking** (coding agents, structured-CoT, cost-bounded thinking) — see [STRUCTURED_COT.md](STRUCTURED_COT.md) | [`bounded-thinking.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.bounded-thinking.yml) | **180K** | 50 / 66 | ~21.7 GB (mem-util 0.95) |
| **Bulletproof, no cliffs** (production service, unpredictable inputs) | [`llamacpp/default`](../models/qwen3.6-27b/llama-cpp/compose/docker-compose.yml) | **262K** | 21 / 21 | ~20 GB |
| **Small-context vLLM safe path** ([@stiggy2k16](https://github.com/noonghunna/club-3090/issues/43) data point) — IDE agents capped at <60K accumulated, when you need vLLM speed but llama.cpp is too slow | [`minimal.yml`](../models/qwen3.6-27b/vllm/compose/docker-compose.minimal.yml) at `--gpu-memory-utilization 0.95 --max-model-len 65536` | **64K** | ~32 / ~33 (no MTP) | ~22.4 GB |

Run via `bash scripts/launch.sh` (interactive) or `bash scripts/switch.sh <variant>`.

> ## ⚠️ Two cliffs to know — both same root kernel, different triggers
>
> ### Cliff 2a — single-prompt OOM (mostly closed on v7.69) ✅
>
> **Pre-v7.69:** vLLM single-card variants crashed on single prompts >~50K tokens.
>
> **Post-v7.69 + vllm#35975 + 0.93 mem-util (Balanced MTP) or MTP-off + 0.95 (Max-context):** **60K single-prompt now passes cleanly** (verified HTTP 200, recall correct, AL=4.00 on Balanced MTP). 90K is past wall-clock-feasible on this hardware. For prompts >60K, use `dual-turbo.yml` (TP=2 splits state) or `llamacpp/default` (262K, different engine).
>
> The fix combines [vllm#35975](https://github.com/vllm-project/vllm/pull/35975) (skip `inputs_embeds` GPU buffer for text-only models, frees ~444 MiB at boot) with mem-util tuning to free activation headroom for the late-stage 50 MiB allocation that previously fired the cliff.
>
> ### Cliff 2b — accumulated-context OOM under multi-turn agent traffic (NOT closed) ❌
>
> Same kernel, different trigger. The 50 MiB `chunk_fwd_o → torch.empty_like(v)` allocation also fails when accumulated multi-turn KV cache + GDN forward live-tensor cascade peaks above 24 GiB on a single card. Hits at **~21-26K accumulated tokens** across 4-5 turns of hermes/openhands/OpenCode/Cline/OpenClaw — not just at 50-60K single prompts. Validated 2026-05-03 across all six shipped single-card vLLM composes; only `vllm/dual` (TP=2) and `llamacpp/default` survive. See the critical warning at the top of this page.
>
> Why this isn't tunable at the config layer: Codex investigation showed the simultaneous live-set of `chunk_gated_delta_rule_fwd` is ~500 MiB at T=4128 (q/k/v/u/v_new/o/w/A/Ai/h tensors all alive at once). Adding accumulated KV + model + workspace + MTP draft puts the per-card peak above 24 GiB. The fix has to be at the kernel level — streaming/pooling these intermediates rather than holding them simultaneously. That's an upstream PR target, not a config knob.

> ### What was Cliff 1 mech B (now closed) ✅
>
> Earlier in 2026-05 we tracked an inductor compile-path FFN intermediate buffer leak ([club-3090 #16](https://github.com/noonghunna/club-3090/issues/16)) that crashed long-* variants on real IDE-agent prompts. **Closed 2026-05-02** via Genesis PN25 (Inductor-safe `silu_and_mul` opaque op) + PN30 (DS conv state dst-shaped temp fix). Both fixes ship by default in our composes; ChatGPT/Codex CLI cross-check helped land the PN30 dst-shaped temp variant. No user action needed — a fresh `bash scripts/setup.sh qwen3.6-27b` picks up the fixes automatically.

---

## Measured TPS on single 3090

![Qwen3.6-27B TPS — single 3090 configs](img/performance-single.png)

Bench protocol: 3 warm + 5 measured runs of the canonical narrative + code prompts on each config. Substrate: vLLM nightly `0.20.1rc1.dev16+g7a1eb8ac2` + Genesis v7.69 dev tip (commit `2db18df`), with local backports `patch_inputs_embeds_optional.py` (vllm#35975) and `patch_tolist_cudagraph.py`. llama.cpp mainline `0d0764dfd`, RTX 3090 sm_86 PCIe-only at 230 W. Per-config run-by-run + VRAM peaks: [models/qwen3.6-27b/CHANGELOG.md](../models/qwen3.6-27b/CHANGELOG.md).

---

## VRAM budget on 24 GB

![Per-card VRAM allocation, single-card configs](img/vram-budget-single.png)

What this says about single-card constraints:

- **Model weights** consume ~14 GB (AutoRound INT4 / GGUF Q3_K_XL). Half the card.
- **KV cache** is the next biggest line; its size depends on `--kv-cache-dtype` × ctx. fp8 ≈ 1 byte/token/(layer×head); TQ3 ≈ 0.4 bytes/token/(layer×head); fp16 ≈ 2 bytes/token/(layer×head).
- **Vision tower** (mmproj) costs ~0.5–1.0 GB extra when on.
- **Activations + cudagraph pools** is what's left. **2026-05-02 PM (v7.69 + vllm#35975 backport + Cliff 2 60K closed)** — settled at **180K + 0.93 (long-text Balanced MTP)**, **200K + 0.95 (long-text-no-mtp Max-context)**, **180K + 0.95 (bounded-thinking)**, and **145K + 0.95 (long-vision)** after v7.69 closed three v7.66/v7.68 regressions and the local #35975 backport freed ~444 MiB by skipping the text-only inputs_embeds GPU buffer. Both Cliff 1 mech B AND Cliff 2 60K are now CLOSED on TQ3 single-card. See [`docs/CLIFFS.md`](CLIFFS.md).

For the cross-card TP=2 picture, see [`DUAL_CARD.md`](DUAL_CARD.md).

---

## Pick a config

### Long ctx + vision — `long-vision.yml` ⭐

**Workload:** chat with images, vision-aware coding agents, multimodal RAG. Anything where the user might paste a screenshot.

145K + vision tower + TQ3 KV + DS layout + Genesis MTP n=3 + full v7.69 patch stack (PN12 + PN17 + PN25 + PN30 part3 + PN26b + P38B + P15B + PN33 + PN32 GDN chunked-prefill) at mem-util 0.95. `verify-stress.sh`: full 7/7 probes pass on text-only paths post-v7.69; vision tower's persistent ~1 GB tightens single-prompt envelope vs long-text. Code 66 / narr 50 TPS (n=5, CV 2-4%), AL 3.40-3.56.

### Long ctx, text-only — Balanced MTP — `long-text.yml` ⭐

**Workload:** RAG ingest, codebase analysis, book/document Q&A, IDE coding agents (Cline / OpenCode / Roo / Claude Code / Cursor), long conversations. **Default recommendation for steady-state agent + chat.**

180K + no vision + TQ3 KV + DS layout + MTP K=3 + same v7.69 patch stack + local vllm#35975 backport at mem-util 0.93. **60K single-prompt PASS @ 623s wall** (HTTP 200, recall correct, AL=4.00). Code 67 / narr 50 TPS (n=5, CV 2.6%), AL 3.34-3.51. **IDE-agent prompts AND big single prompts up to 60K both work cleanly here.**

### Long ctx, text-only — Max-context — `long-text-no-mtp.yml`

**Workload:** one-shot >50K input where you can wait, don't need MTP, and want maximum KV pool capacity.

200K + no vision + TQ3 KV + DS layout + MTP **off** + same v7.69 patch stack + local vllm#35975 backport at mem-util 0.95. **60K single-prompt PASS @ 537s wall** (HTTP 200, recall correct). Decode is slower without spec-decode (~33 narr / ~40 code TPS estimated; canonical bench pending). Use only when steady-state TPS isn't the priority and you need the extra KV headroom.

### Bulletproof / no cliffs — `llamacpp/default` ⭐

**Workload:** production service for unpredictable users. Inputs that might be 5K or might be 200K. Tool returns that might be 1K or might be 50K. Anywhere "predictable behavior" beats "peak TPS."

`bash scripts/switch.sh llamacpp/default`. Q3_K_XL (Unsloth dynamic) + q4_0 KV at 262K + vision (mmproj). Different attention library entirely (ggml-cuda, not FA2) → no Cliff 1 mechanism, no Cliff 2 mechanism. Trade is ~21 TPS (~2.5× slower than vLLM). Quant validated by [Benjamin Marie's Kaitchup eval](https://kaitchup.substack.com/p/summary-of-qwen36-gguf-evals-updating).

---

## Other variants in the repo (not recommended for shipping)

These exist for troubleshooting, niche workloads, or historical comparison. Not promoted as primary because the long-* variants now cover their use cases:

- **`docker-compose.yml`** — 48K + TQ3 + vision, mem-util 0.92. The "below both cliffs by definition" baseline (engine HTTP-400-rejects requests >48K, so Cliff 2 is unreachable). Useful when you want bulletproof error behavior on a specific small-ctx workload, or as a fast-boot diagnostic. Most users should pick `long-vision` or `llamacpp/default` instead.
- **`tools-text.yml`** — 75K + FP8 KV + PN8. Was the only Cliff-1-safe single-card path before PN12 anchor fix landed. FP8 KV is closer in quality to FP16 than TQ3 is, so kept around for accuracy-sensitive comparisons. Most IDE-agent workloads now run fine on `long-text.yml`.
- **`minimal.yml`** — 32K + FP8 + no Genesis + no spec-decode. Stripped-down stack for isolating "is this a Genesis bug?" questions. Half the throughput of any other variant.

## Watch list — Luce DFlash (not yet a recommendation)

Re-tested 2026-04-30 PM against [`Luce-Org/lucebox-hub`](https://github.com/Luce-Org/lucebox-hub) on Qwen3.6-27B Q4_K_M target + matched z-lab/Qwen3.6-27B-DFlash draft. **Closer to parity than 2026-04-22 — but several gaps still keep it off the recommended list:**

Measured TPS on this rig (RTX 3090, greedy, single-stream, n_gen=1000):

| Workload | Luce DFlash 3.6+3.6 (TQ3 KV, max_ctx=65K) | vLLM long-text 214K |
|---|---|---|
| Narrative essay | 37–47 TPS (mean ~40) | 50 TPS |
| Code (heap/LRU/AST) | 63–76 TPS (mean ~72) | 67 TPS |
| AL (code) | 5.9–7.1 | 3.4–3.8 (MTP) |

What works since 2026-04-22:
- ✅ **Tool calls** via `server_tools.py` — parses Qwen `<tool_call>` format → returns OpenAI `tool_calls[]`. The big server-UX gap from last bench is closed.
- ✅ **Streaming SSE** with `reasoning_content` deltas.
- ✅ **Daemon mode** with cache-reuse for fast cold starts.
- ✅ **Verify-stress 25K tool-prefill passes** at TQ3 KV + max_ctx=65K.

What still keeps it off the recommended list:
- ❌ **Greedy only** — `temperature` / `top_p` ignored. Real downside for creative-writing workloads.
- ❌ **3.6 draft under-trained** (z-lab snapshot 2026-04-26). Narrative AL ~3.7 vs code ~7.0; narr loses ~20% TPS to vLLM until training completes.
- ❌ **No vision** tower.
- ❌ **`enable_thinking` chat_template_kwargs handled differently** than vLLM — verify-full check 6 fails.
- ❌ **Prefill cliff at higher max_ctx** — 25K tool prefill OOMs in `fattn-chunked.cu` at Q8_0 KV + max_ctx=65K (TQ3 KV closes it). At max_ctx=131K + TQ3, the daemon subprocess crashes (broken-pipe to FastAPI) on 30K+ probes.
- ❌ **Build fragility** — fresh `git clone` of `dflash` main HEAD fails to compile (`ggml_turbo_wht` / `GGML_TYPE_TQ3_0` undefined) until you `git submodule update --init` after manual `git fetch` in `dflash/deps/llama.cpp`.
- ❌ **Daemon-mode "empty prompt" regression** — after streaming requests, subsequent requests sometimes return 0 tokens; needs server restart.

**Re-test trigger:** z-lab tags the Qwen3.6-27B-DFlash draft as training-complete OR Luce-Org publishes a tagged release with the daemon-mode bug fixed. Track in [`docs/UPSTREAM.md`](UPSTREAM.md#luce-dflash-luce-orglucebox-hub).

---

## What single-card can't do

| Want | Why not on 1× | What you'd need |
|---|---|---|
| 4 concurrent streams at 262K + vision | KV pool too small for 4 × full ctx | TP=2 (see DUAL_CARD.md) |
| Peak code TPS (>100 TPS on quicksort prompt) | DFlash N=5 needs head_size=256 + non-causal — vLLM head-dim split | TP=2 + DFlash |
| Single-prompt >60K tokens on vLLM | Cliff 2 (DeltaNet GDN forward), no fix yet | TP=2 OR llama.cpp 262K (different engine) |

---

## Common pitfalls (single-card specifics)

### Prefill cliffs

- **Cliff 1** — FFN intermediate-buffer activation peak (138 MiB allocate at `intermediate_size × max-num-batched-tokens`). Historically fired on long-ctx composes at >0.95 mem-util when prefill batch needed the buffer. **Closed on `tools-text.yml`** (FP8 KV path) since 2026-04-29 via Genesis PN8. **Closed on TQ3 paths** on v0.20 + Genesis v7.65+ since 2026-05-01 via PN12 + PN17 + P38B in-source hooks. **Mech B closed** 2026-05-02 (v7.66 + PN25 v3 + PN30 dst-shaped) — see [`docs/CLIFFS.md`](CLIFFS.md).
- **Cliff 2** — DeltaNet GDN forward OOM. **Closed at 60K single-prompt** as of 2026-05-02 PM via Genesis v7.69 (PN32 GDN chunked-prefill + P103 worker self-install) plus a local backport of vllm#35975 (skip text-only `inputs_embeds` GPU buffer, ~444 MiB freed). Two shippable variants: `long-text.yml` (Balanced MTP, 180K + 0.93) and `long-text-no-mtp.yml` (Max-context, 200K + 0.95). >60K still hits the 24 GB hardware-physical wall — for those prompts use dual-card TP=2 (verified 237K) or llama.cpp single-card (262K, different engine).

### VRAM peak vs idle

`nvidia-smi` at boot ≠ peak. Boot shows weights + KV pool reservation. **Peak adds activation buffers during prefill** — typically +500-1500 MiB. If `nvidia-smi` shows 23.5/24 GB at idle, you have ~500 MiB for prefill activations — not enough for the 138 MiB-class buffer at long ctx. Drop mem-util by 0.03 if you need the headroom.

### Tool-call extraction needs `--enable-auto-tool-choice`

vLLM ships this off by default. Our composes set `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice`. If you're rolling your own compose, both are required.

---

## Quick start

```bash
# 1. Setup (downloads model, clones Genesis, ~20 min cold)
bash scripts/setup.sh qwen3.6-27b

# 2. Pick + boot via wizard (asks engine + workload)
bash scripts/launch.sh

# 3. Or skip the wizard:
bash scripts/launch.sh --variant vllm/tools-text   # IDE agent path
bash scripts/launch.sh --variant llamacpp/default  # easy mode

# 4. Sanity test
curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":200}'

# 5. Switch later without re-running setup
bash scripts/switch.sh vllm/long-vision    # for example
bash scripts/switch.sh --list              # show all variants
```

---

## Models supported on single 3090

- **[Qwen3.6-27B](../models/qwen3.6-27b/)** — primary model. Quant choices, Genesis patches, engine internals all in the model directory.
- More models coming. As they're added, this section will list which single-card configs each one supports.

---

## Deep dives

- **[Model README](../models/qwen3.6-27b/)** — quant choices (AutoRound INT4 / GGUF Q3_K_XL), Genesis patch surface, what's working / what's not.
- **[INTERNALS.md](../models/qwen3.6-27b/INTERNALS.md)** — engineering rationale (Genesis P65/P66/PN8, Marlin pad fork, MTP, the cascade bug, upstream tracker).
- **[VRAM allocation diagram](../models/qwen3.6-27b/README.md#vram-allocation-across-configs)** — full per-config breakdown across single + dual.
- **[FAQ.md](FAQ.md)** — common questions (4090 / 5090 support, why MTP not EAGLE, Copilot Gateway, what's a cliff, etc.).
- **[EXAMPLES.md](EXAMPLES.md)** — Python / TS / curl client snippets + IDE connection settings.
- **[HARDWARE.md](HARDWARE.md)** — Ampere SM 8.6 specifics, NVLink (declined), power caps.
- **[DUAL_CARD.md](DUAL_CARD.md)** — when you need what single-card can't deliver.
