# Qwen3.6-27B

**Run [Qwen3.6-27B](https://huggingface.co/Qwen) — with vision and tool calling — on 1 or 2 RTX 3090s.** Full OpenAI-compatible API, drop-in replacement for ChatGPT/Claude in any tool that uses the OpenAI SDK.

> 👉 **For deployment options + workload-driven config picks**, see the hardware-axis pages:
> [`docs/SINGLE_CARD.md`](../../docs/SINGLE_CARD.md) (1× 3090) · [`docs/DUAL_CARD.md`](../../docs/DUAL_CARD.md) (2× 3090).
>
> This page is the **model-specific reference**: quants, what's working / not working, VRAM allocation, engine pointers.

---

## What this is

- **27B parameter dense LLM** with vision support (Qwen3-Next family — hybrid DeltaNet + standard attention)
- **Quant on this stack:** [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — INT4 weights with BF16 `mtp.fc` head preserved (lets vLLM use MTP spec-decode)
- **GGUF alternative:** [`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) — Q3_K_XL ⭐ (validated by [Marie's Kaitchup eval](https://kaitchup.substack.com/p/summary-of-qwen36-gguf-evals-updating)), Q4_K_M, Q5_K_S
- **Engines:** vLLM (full features) · llama.cpp (max context, lighter footprint) · SGLang (currently blocked, watch list)

---

## Quick start

The easiest entry is the wizard at the repo root, which asks engine + workload and boots the right compose:

```bash
bash scripts/setup.sh qwen3.6-27b
bash scripts/launch.sh
```

If you already know the variant you want, see [`docs/SINGLE_CARD.md`](../../docs/SINGLE_CARD.md) or [`docs/DUAL_CARD.md`](../../docs/DUAL_CARD.md) for the menu, then:

```bash
bash scripts/switch.sh vllm/tools-text   # for example
```

Sanity check after boot:

```bash
curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":200}'
```

---

## VRAM allocation across configs

How each config splits the 24 GB / card budget — weights, KV cache, vision tower, DFlash draft (where applicable), and activation/cudagraph headroom. Single-card (TP=1) on top, dual-card (TP=2, weights and KV halved across both GPUs) on bottom.

![Per-card VRAM allocation across single + dual configs](../../docs/img/vram-budget-combined.png)

As of 2026-05-02 (vLLM v0.20 + Genesis v7.66 dev tip + Cliff 1 mech B closed), single-card recommended options (see [`docs/SINGLE_CARD.md`](../../docs/SINGLE_CARD.md)):
- **`long-text.yml` — 180K text-only** at 0.95 mem-util. **IDE-agent recommended.** Cliff 1 mech B closed via PN25 v3 import-time + PN30 dst-shaped temp fix. Real OpenCode/Cline/Roo workloads work cleanly here. KV pool ~284K tokens.
- **`long-vision.yml` — 145K + vision** at 0.95 mem-util. Vision tower's persistent ~1 GB tightens activation budget further than long-text. Same patch stack.
- **`bounded-thinking.yml` — 180K text-only + structured-CoT grammar in reasoning** at 0.95 mem-util. Same patch stack as long-text plus `--structured-outputs-config.enable_in_reasoning true`. ~30× cheaper think output on coding workloads with **+4.3pp HE+ / +24pp LCB v6** vs FREE thinking. See [`docs/STRUCTURED_COT.md`](../../docs/STRUCTURED_COT.md).
- **`llamacpp/default` — 262K + vision** at ~21 TPS. Different engine, no cliffs anywhere — production-safe for unpredictable inputs.

The **single shipped limitation** on the vLLM single-card variants: Cliff 2 still fires on single prompts >50–60K (DeltaNet GDN forward OOM — architectural, not closed by v0.20 / v7.66 / our patches). Use llama.cpp single or dual-card (TP=2 splits state across cards) for one-shot big prompts. See [`docs/CLIFFS.md`](../../docs/CLIFFS.md).

Other variants (`docker-compose.yml` 48K · `tools-text.yml` 75K FP8 · `minimal.yml` 32K) are kept in the repo as fallbacks / diagnostics, not promoted as primary.

TP=2 unlocks **262K + 4 concurrent streams** on dual-card (`dual.yml`).

---

## What's working

- **Vision** — images in messages via OpenAI-compat format. Tower is small (~0.5–1.0 GB VRAM); each image consumes 640–1280 tokens at default resolution. Quality is good for charts / screenshots / natural images, less reliable for OCR on dense text. No image *generation* — this model is vision-input-only.
- **Tool calling** — `tools=[...]` + `tool_choice="auto"`, parsed cleanly into `tool_calls[]`. Genesis v7.62.x ships PN11 (Quentin-M's streaming-tool-call IndexError fix from vllm#41142).
- **Streaming** — SSE chunks add up to coherent text; tool-call deltas stream too.
- **Reasoning mode** — `chat_template_kwargs.enable_thinking=true` for chain-of-thought (vLLM extracts into `reasoning_content` field; llama.cpp emits inline).
- **Spec-decode** — MTP n=3 default on vLLM (~83% per-position-1 accept on code); DFlash N=5 on dual-card for code-heavy workloads.
- **All standard sampling** — temperature, top_p, top_k, repetition_penalty, JSON-mode, structured output.

## What's not working today

- **GGUF on vLLM** for Qwen3-Next family — not supported upstream. Use llama.cpp for GGUF on this model.
- **EAGLE spec-decode on hybrid attention** — DeltaNet rollback issue (cross-engine architectural). Watch upstream.
- **Single-card single prompts >50-60K** — Cliff 2 (DeltaNet GDN forward state). Lives in `fla.ops` upstream, no file-replacement patch. Watch [vllm#40914](https://github.com/vllm-project/vllm/pull/40914) and [FlashQLA](https://github.com/QwenLM/FlashQLA). Mitigation: dual-card (TP=2 splits state) or llama.cpp 262K. **Cliff 1 mech B** (inductor compile-path FFN intermediate buffer leak) — **closed since 2026-05-02** via PN25 v3 import-time backport + PN30 dst-shaped temp fix. Real IDE-agent prompts work cleanly. See [`docs/CLIFFS.md`](../../docs/CLIFFS.md).

---

## Quant decision

Why AutoRound INT4 over alternatives:

| Quant | Stack support | Bench | Trade-off |
|---|---|---|---|
| **Lorbus AutoRound INT4** ⭐ | vLLM `--quantization auto_round` | 51-89 TPS depending on config | +9% over AWQ (this model); BF16 MTP head preserved; required for MTP spec-decode |
| AWQ INT4 | vLLM `--quantization awq` | 38 TPS @ 8K | Works; slower; no spec-decode advantage |
| GPTQ INT4 (palmfuture) | vLLM `--quantization gptq` | 137 TPS @ 262K dual-card | Older path; AWQ + DFlash had a pad-Marlin × aux-layer bug we never reduced |
| GGUF Q3_K_XL (Unsloth dynamic) | llama.cpp only | 21 TPS @ 262K | One Docker line, no patches, no Cliff 1/2; ⭐ Marie's Kaitchup eval picks this as optimal accuracy/footprint |
| GGUF Q4_K_M | llama.cpp only | ~28 TPS measured 2026-04-23 (regression to ~21 today on current commit) | Heavier than Q3_K_XL; quality close |

For deeper rationale, comparison tables, and the patched-vLLM-source story (vllm#40361 Marlin pad-sub-tile-n), see [INTERNALS.md](INTERNALS.md).

---

## Genesis patch surface (vLLM)

The vLLM composes mount Sandermage's [Genesis tree](https://github.com/Sandermage/genesis-vllm-patches) and apply specific patches at boot. Currently pinned at commit `fc89395` (v7.66 dev tip, 2026-05-02) per `scripts/setup.sh`.

Active patches per compose (selected highlights — full env-var stack in each compose YAML):

| Patch | What it does | Composes |
|---|---|---|
| P4 | Hybrid TurboQuant support | TQ3 paths |
| P64 | Streaming tool-call edge case | all single-card vLLM with MTP |
| P65 | TurboQuant spec-CG downgrade (#40880 fix) | TQ3 paths |
| P66 | Cudagraph capture-size divisibility | TQ3 paths |
| P67 | TQ multi-query kernel | TQ3 paths |
| P68/P69 (50K char threshold) | Long-ctx tool-choice nudge — safe on real IDE-agent prompts since v7.65 raised default 8000 → 50000 | TQ3 paths |
| P98 | TQ WorkspaceManager revert (auto-skips on v0.20 due to drift marker; covered by `patch_workspace_lock_disable.py` sidecar) | TQ3 paths |
| P101 | TQ continuation 64-token slicing | TQ3 paths |
| P103 | FLA Cliff 2 chunked fwd_h+fwd_o orchestrator | TQ3 paths |
| **PN12** | FFN intermediate scratch pool — closes Cliff 1 mech B on TQ3 (Genesis-native on v0.20, no sidecar needed) | TQ3 paths |
| **PN17** | FA2 softmax_lse runtime clamp — closes Cliff 1 mech A | TQ3 paths |
| PN13 | CUDAGraphWrapper lambda-arity (vllm#41235 backport) | all |
| **P38B** | In-source `_continuation_prefill` hook — Genesis #14 fix for compile-path silent no-op | TQ3 paths |
| **P15B** | FA varlen `max_seqlen_k` clamp at TQ wrapper — Genesis #15 fix | TQ3 paths |
| **PN26b** | First public sparse-V Triton kernel for SM86 (Ampere consumer) — 27B-tuned BLOCK_KV=8 num_warps=4 threshold=0.01 | TQ3 paths |
| **PN8** | MTP draft online-quant propagation (vllm#40849) — closes Cliff 1 on FP8 path | **fp8 path** (`tools-text.yml`) |

Two local sidecars apply outside Genesis:
- **`patch_workspace_lock_disable.py`** — relaxes vllm#39226 strict assertion to one-shot WARNING. Sandermage's P98 covers same surface but auto-skips on v0.20 (drift-marker false-positive). Idempotent. Drop when Sandermage ships marker fix.
- **`patch_tolist_cudagraph.py`** — guards `.tolist()` calls during cudagraph capture. Unblocks TurboQuant KV + spec-decode + chunked-prefill on Qwen3-Next dense on Ampere. Drop when upstream fixes the continuation-prefill `.tolist()` sync.

Local sidecars dropped during 2026-05-01 v0.20 migration:
- `patch_pn12_ffn_pool_anchor.py` → covered natively by PN12 on v0.20
- `patch_pn12_compile_safe_custom_op.py` → covered by Genesis PN25
- `patch_fa_max_seqlen_clamp.py` → covered by PN17 + P15B

Dual-card composes (`dual.yml`, `dual-dflash*`) are **Genesis-less by design** — fp8 KV + TP=2 + 0.92 mem-util has plenty of headroom and doesn't trigger the cudagraph bugs Genesis was built to patch. `dual-turbo.yml` does mount Genesis (TQ3 path needs P65).

Forensic chain + per-patch attribution → [INTERNALS.md](INTERNALS.md).

---

## See also

- **[/docs/SINGLE_CARD.md](../../docs/SINGLE_CARD.md)** — 1× 3090 deployment menu (workloads → composes → TPS).
- **[/docs/DUAL_CARD.md](../../docs/DUAL_CARD.md)** — 2× 3090 deployment menu.
- **[INTERNALS.md](INTERNALS.md)** — engineering deep dive (Genesis patches, forensics, Marlin pad, DFlash, upstream tracker).
- **[CHANGELOG.md](CHANGELOG.md)** — dated history (combines single + dual timelines).
- **[/docs/EXAMPLES.md](../../docs/EXAMPLES.md)** — Python / TS / curl client snippets + Open WebUI / Cline / Cursor connection settings.
- **[vllm/](vllm/)** — vLLM-specific recipes (compose YAMLs are documented in their own headers).
- **[llama-cpp/](llama-cpp/)** — llama.cpp recipes (max context on single card, no prefill cliffs).
- **[sglang/](sglang/)** — SGLang status (currently blocked).
- **[/docs/engines/](../../docs/engines/)** — cross-model engine comparison + per-engine deep dives.
- **[/docs/HARDWARE.md](../../docs/HARDWARE.md)** — hardware notes (Ampere, NVLink, power).
