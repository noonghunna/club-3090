# Single 3090 — what fits, how to run it

You have **one RTX 3090 (24 GB VRAM)**. This is the front door for picking a config and knowing what to
expect. Deep dives (quants, Genesis patches, engine internals) and the failure forensics live elsewhere —
links at the bottom, and the archived watch-list/forensics in [`SINGLE_CARD.history.md`](SINGLE_CARD.history.md).

> **Want a model not listed here / any HF safetensors repo?** → [`PULL.md`](PULL.md): `scripts/pull.sh`
> evaluates any model against the KV math (honest, no download) and boots it if it passes.

---

## ⚠️ Read first if you run an agentic coding client

If your workload is **hermes / openhands / OpenCode / Cline / Roo / Aider / Cursor with retained context**,
**single-card vLLM is not safe**: you hit a hardware-physical cliff at **~21–26K accumulated multi-turn
context**, on every single-card vLLM variant (validated across all six, [#41](https://github.com/noonghunna/club-3090/issues/41)).
Symptoms: throughput drops to 0 / unresponsive / 500s / OOM after a few turns. It's a 24 GB wall in the
DeltaNet GDN kernel — not tunable at the config layer (mem-util, MTP-off, batched-tokens all tested). Full
mechanism: [`CLIFFS.md`](CLIFFS.md) · single-card forensics: [`SINGLE_CARD.history.md`](SINGLE_CARD.history.md).

**Two safe paths for these workloads:**

| Have | Run | Why it works |
|---|---|---|
| 2× 3090 (any topology) | `bash scripts/switch.sh vllm/dual` | TP=2 splits the failing kernel across both cards. Soak-PASS, 111+ TPS p50. |
| 1× 3090 only | `bash scripts/switch.sh llamacpp/default` | Different engine + GDN kernel + allocator — the cliff doesn't exist. 200K ctx, ~51/60 TPS. |

Workloads that **don't** accumulate context (single-shot RAG, simple chat, batch) run fine on single-card vLLM.

---

## TL;DR — pick by workload (Qwen3.6-27B, the primary single-card model)

> ⛔ The struck-through vLLM rows are pinned to a purged nightly ([#167](https://github.com/noonghunna/club-3090/issues/167))
> and won't boot until the next Genesis-compatible pin lands — use the **llama.cpp / ik_llama** rows (they work
> today and are cliff-immune). Detail on the blocked configs: [`SINGLE_CARD.history.md`](SINGLE_CARD.history.md).
>
> 🎯 **Don't want to choose?** `bash scripts/launch.sh` resolves the blessed single-card default — on one 3090
> that's **`ik-llama/iq4ks-mtp` (fastest + leanest)**, with **`llamacpp/default` as the cliff-immune
> alternative**. Pin your own with `switch.sh --set-default <slug>`.

| What you're doing | Compose | Max ctx | Narr / Code TPS | VRAM |
|---|---|---|---|---|
| ⛔ ~~Long ctx + vision, text-only, or bounded-thinking~~ _(4 vLLM configs blocked #167)_ | ~~`long-vision` / `long-text` / `long-text-no-mtp` / `bounded-thinking`~~ | ~~145–200K~~ | ~~50 / 66~~ | ~~21–23 GB~~ |
| **Bulletproof, no cliffs** (production, unpredictable inputs) | [`llamacpp/default`](../models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml) | **200K** | 52 / 61 | ~23 GB |
| **llama.cpp + MTP, fast + long ctx** ⭐ (IDE agents, Hermes, multi-turn) | [`llamacpp/mtp`](../models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml) | **200K** | **51 / 60** | ~22.5 GB |
| **llama.cpp + MTP + vision** (multimodal chat, screenshot-debug) | [`llamacpp/mtp-vision`](../models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp-vision.yml) | **49K** | **57 / 66** | ~20.5 GB |
| **ik_llama + IQ4_KS + MTP** ⭐ (fastest + leanest single-card) | [`iq4ks-mtp`](../models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp.yml) | **200K** | **~60 / ~69** | **~22 GB** |
| **ik_llama + IQ4_KS + MTP + vision** | [`iq4ks-mtp-vision`](../models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp-vision.yml) | **160K** | — | ~21 GB |
| **ik_llama + two-stage spec-dec** ⭐ (code-heavy; ngram + MTP, code +35%) | [`iq4ks-two-stage`](../models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/two-stage.yml) | **200K** | **~59 / ~98** | ~22 GB |

Run via `bash scripts/launch.sh` (interactive) or `bash scripts/switch.sh <variant>`.

---

## The recommended single-card recipes (Qwen3.6-27B)

The llama.cpp / ik_llama paths use the **ggml memory model — no Cliff 1, no Cliff 2** — so they're the
single-card recommendation today. All ship via pulled images (no source build).

- **`llamacpp/default`** ⭐ — Q4_K_M MTP GGUF (`unsloth/Qwen3.6-27B-MTP-GGUF`) + q4_0 KV + MTP `n=2`, **200K**
  (max-safe; 262K boots but walls ~125K). The no-friction, cliff-immune production pick. ~52/61 TPS. (Alias of
  `llamacpp/mtp`.)
- **`llamacpp/mtp`** ⭐ — same GGUF + `-ub 1024`, mainline build 9235 (PR #22673 MTP merged). ~51/60 TPS
  (~2.5× faster than default), **verify-stress 7/7 incl. 60K + 91K needle** — at this config the model walks
  past 91K cleanly (the Cliff-2 narrative is config-driven, not architectural). Quality 102/150 on the 8-pack
  beats every Qwen vLLM-dual; aider-polyglot 17/30 matches bf16-dual on half the hardware.
- **`llamacpp/mtp-vision`** — `llamacpp/mtp` + `--mmproj mmproj-F16.gguf`, **49K** speed-optimal default
  (~57/66 TPS text path). Push to **192K** with `UBATCH_SIZE=512 CTX_SIZE=196608` (~10% slower, verify-stress 7/7).
- **`iq4ks-mtp`** ⭐ — ubergarm `Qwen3.6-27B-MTP-IQ4_KS.gguf` (IQK imatrix, built-in MTP) + q4_0 KV +
  `-khad`/`-vhad` + MTP `n=2`, **200K**. At matched 370 W, **~18–20% faster than `llamacpp/mtp`** (~60/69 vs
  ~50/58), quality-tied (8-pack 103 vs 102), **~0.5–0.8 GB leaner** — the faster *and* leaner path. Engine:
  digest-pinned `ik-llama-cpp cu13-server`. Deep dive: [`engines/IK_LLAMA.md`](engines/IK_LLAMA.md).
- **`iq4ks-two-stage`** ⭐ — chains ngram self-spec (repeated code spans, near-zero compute) with the MTP head
  for novel tokens (ik PR #1789). At `ngram n_max=4` + `MTP n_max=3`: **~59 narr / ~98 code TPS** — narrative
  tied with `iq4ks-mtp`, **code +35%**. Lossless (8-pack 98 ≈ 100, aider 16 ≈ 19, within noise). 200K. **Use
  this for code; `iq4ks-mtp` for balanced/prose.**

The blocked TQ3+Genesis vLLM configs (`long-*`, `bounded-thinking`) and the not-recommended variants
(`tq3-mtp`, `tools-text`, `minimal`) are detailed in [`SINGLE_CARD.history.md`](SINGLE_CARD.history.md).

---

## Measured TPS + VRAM on 24 GB

![Qwen3.6-27B TPS — single 3090 configs](img/performance-single.png)

Bench: 3 warm + 5 measured of the canonical narrative + code prompts per config, RTX 3090 sm_86 PCIe at 230 W.
Per-config run-by-run + VRAM peaks: [`models/qwen3.6-27b/CHANGELOG.md`](../models/qwen3.6-27b/CHANGELOG.md).

![Per-card VRAM allocation, single-card configs](img/vram-budget-single.png)

- **Weights** ~14 GB (AutoRound INT4 / GGUF Q4_K_M) — half the card. **KV cache** is next (fp8 ≈ 1 ·
  TQ3 ≈ 0.4 · fp16 ≈ 2 bytes/token/layer×head). **Vision tower** (mmproj) +0.5–1.0 GB. **Activations +
  cudagraph pools** are what's left — the budget the cliffs fight over.
- `nvidia-smi` at idle ≠ peak: prefill adds +500–1500 MiB of activation buffers. If idle shows 23.5/24 GB,
  you have ~500 MiB for prefill — drop mem-util 0.03 if you need headroom.

For the cross-card TP=2 picture, see [`DUAL_CARD.md`](DUAL_CARD.md).

---

## Other models on a single 3090

Qwen3.6-27B is the production single-card model (recipes above). Other catalog models with single-card
composes — at varying maturity. **`bash scripts/switch.sh --list` is the authoritative, always-current matrix;**
this table is a snapshot:

| Model | Single-card pick | Status | Max ctx |
|---|---|---|---|
| **Qwen3.6-27B** ⭐ | `ik-llama/iq4ks-mtp` / `llamacpp/default` | ✅ Production | 200K |
| Qwen3.6-35B-A3B | `ik-llama/apex-fit-q8q5` (MoE, Q8/Q5 fit) | 🧪 Experimental | 197K |
| Gemma-4-12B | `vllm/gemma-12b-single-int8-mtp` (or beellama/llamacpp Q8-K-XL) | ⚠️ Caveats | 262K |
| Gemma-4-26B-A4B | `vllm/gemma-26ba4b-single` (AWQ + INT8 KV) | ⚠️ Caveats | 176K |
| Gemma-4-31B | `beellama/gemma-dflash` (Q4_K_S) | ⚠️ Caveats — 31B dense is tight on 24 GB | 128K |
| Ornith-1.0-9B | `ik-llama/ornith9b-single` (ngram) | 🧪 Experimental | 262K |
| VibeThinker-3B | single composes (3B fits easily) | 🐣 Incubating | — |

_Dual-card-only models (don't fit single): Ornith-1.0-35B, Qwen3.6-40B-Deckard, DiffusionGemma-26B-A4B — see
[`DUAL_CARD.md`](DUAL_CARD.md)._

---

## What single-card can't do

| Want | Why not on 1× | What you'd need |
|---|---|---|
| 4 concurrent streams at 262K + vision | KV pool too small for 4× full ctx | TP=2 ([`DUAL_CARD.md`](DUAL_CARD.md)) |
| Peak code TPS (>100 on quicksort) | DFlash N=5 needs head_size=256 + non-causal | TP=2 + DFlash |
| Single-prompt >60K on vLLM | Cliff 2 (DeltaNet GDN), no fix yet | TP=2, or llama.cpp 200K (different engine) |

---

## Common pitfalls (single-card)

- **Tool-call extraction needs `--enable-auto-tool-choice`** (off by default in vLLM). Our composes set
  `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice` — both required if you roll your own.
- **Desktop / sub-24 GB usable VRAM:** compose defaults assume a **headless** 3090. On a workstation driving a
  display (or a 4090 at ~23.5 GB usable), shrink the KV budget — prefer `MAX_MODEL_LEN` first (clean,
  predictable) over `GPU_MEMORY_UTILIZATION` (interacts with profiling). Stay in `0.85–0.92` for TQ3 KV;
  `0.80` is usually too aggressive. Validated on @laurimyllari's 4090.
  ```bash
  MAX_MODEL_LEN=90000 GPU_MEMORY_UTILIZATION=0.90 bash scripts/switch.sh vllm/minimal
  ```
- **Two llama.cpp / ik variants at once:** distinct default `container_name`s, but they share host port `8020`
  — give the second one `ESTATE_PORT` (two full-ctx servers won't co-reside on one 24 GB card; drop `CTX_SIZE`
  on one):
  ```bash
  bash scripts/switch.sh llamacpp/mtp
  ESTATE_PORT=8030 bash scripts/switch.sh llamacpp/mtp-vision
  ```

---

## Quick start

```bash
bash scripts/setup.sh qwen3.6-27b          # download model (~20 min cold)
bash scripts/launch.sh                     # wizard: asks model + GPUs, projects VRAM, boots the default
# or skip the wizard:
bash scripts/launch.sh --variant llamacpp/default
bash scripts/switch.sh --list              # all variants, all models
curl -sf http://localhost:8020/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":50}'
```

---

## Deep dives

- **[Model README](../models/qwen3.6-27b/)** + **[INTERNALS.md](../models/qwen3.6-27b/INTERNALS.md)** — quants, Genesis patch surface, MTP, the cascade bug.
- **[CLIFFS.md](CLIFFS.md)** — the cliff mechanisms (single source of truth) · **[SINGLE_CARD.history.md](SINGLE_CARD.history.md)** — single-card forensics + watch-list.
- **[FAQ.md](FAQ.md)** · **[EXAMPLES.md](EXAMPLES.md)** (client snippets) · **[HARDWARE.md](HARDWARE.md)** (Ampere, power caps) · **[DUAL_CARD.md](DUAL_CARD.md)**.
