# ik_llama.cpp — the advanced-quant engine

**Role on this stack:** the engine you reach for when you want **newer, higher-quality-per-bit quants** than mainline llama.cpp ships — specifically the **IQK imatrix family** (`IQ4_KS`, `IQ5_KS`, …) that exists *only* in this fork. It's a llama.cpp fork (ikawrakow), so it inherits llama.cpp's cliff-immune memory model and broad hardware support, then adds a co-designed quant + kernel stack on top.

> **In one line:** llama.cpp's robustness + fork-exclusive IQK quants + fused CUDA kernels → **MTP, clean to 262K on one card, quality on par with `llamacpp/mtp`** (8-pack 103 vs 102), at a **~0.5–0.8 GB leaner VRAM footprint**. It's also **~18–20% faster** than `llamacpp/mtp` on decode TPS at matched 370 W (~60 narr / ~69 code vs ~50 / ~58 on a 3090) — so it's the faster *and* leaner single-card path. The trade is a second engine to maintain + the IQK quant. Full matched-power write-up: [discussions/184](https://github.com/noonghunna/club-3090/discussions/184).

For *what the quants actually are* and how IQK compares to k-quants / i-quants / AWQ, see **[../QUANTIZATION.md](../QUANTIZATION.md)**. For the cross-engine overview see **[../INFERENCE_ENGINES.md](../INFERENCE_ENGINES.md)**.

---

## TL;DR

- **Image:** digest-pinned — `ghcr.io/ikawrakow/ik-llama-cpp@sha256:5f914f1c…` (a 2026-06-10 `cu13-server` build; cu13 = CUDA 13.x, matches our 13.2 host driver). **Composes pin the digest, not the rolling `:cu13-server` tag** — upstream retired the legacy speculative-decode flags (`-mtp` / `--draft-*` / `--spec-stage`) under the moving tag, which crash-looped MTP composes on a fresh pull (2026-06-13 flag churn); pinning a validated digest stops the recurrence. `docker pull …:cu13-server` fetches the current build; a `cu12` tag exists for older drivers.
- **Composes:** `models/qwen3.6-27b/ik-llama/compose/single/`:
  - `iq4ks-mtp.yml` — MTP-only text (default)
  - `iq4ks-mtp-vision.yml` — MTP + vision
  - `iq4ks-two-stage.yml` — ngram + MTP two-stage (code-optimized, experimental)
- **Model:** ubergarm `Qwen3.6-27B-MTP-IQ4_KS.gguf` (IQK imatrix quant, built-in MTP head).
- **Interface:** same `--jinja` + `--reasoning on|off` server contract as mainline llama.cpp — so the stack-wide thinking-off policy works unchanged.

```bash
MODEL_DIR=/your/models docker compose \
  -f models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp.yml up -d
curl http://localhost:8020/v1/models
```

---

## Why pick ik_llama.cpp over mainline llama.cpp?

Both are cliff-immune (same ggml flat allocator — see [LLAMA_CPP.md](LLAMA_CPP.md) "Why llama.cpp doesn't hit the prefill cliffs"). ik_llama adds, on top:

1. **IQK imatrix quants (fork-exclusive).** `IQ4_KS` / `IQ5_KS` use refined non-linear grids + an importance matrix + **kernels co-designed for those grids**. Net: better quality-per-bit than mainline `Q4_K_M`, and *faster* than mainline i-quants because the dequant path is hand-tuned. `IQ4_KS` is ~15.1 GB vs `Q4_K_M`'s ~17 GB on Qwen3.6-27B — smaller weights leave room for **262K context** on a single 24 GB card.
2. **Fused CUDA kernels** — `--merge-qkv` (fused QKV projection), `--merge-up-gate-experts` (MoE, no-op on dense), and a fast IQK dequant path.
3. **`-khad` / `-vhad` (Hadamard KV-cache)** — Hadamard transforms on K and V caches that improve quantized-KV accuracy (q4_0 + khad beats f16 perplexity on Qwen3.6-27B per ikawrakow's data). Zero VRAM cost.
4. **Two-stage spec-dec** (PR #1789) — chains ngram self-spec (zero VRAM, catches repeated code patterns) with MTP as fallback. Best of both worlds for code workloads.
5. **Hybrid-aware MTP** (PR #1774) — `--recurrent-ckpt-mode` pre-allocates VRAM for Qwen3.6's DeltaNet SSM state, avoiding costly recomputation on draft rejection.
6. **MoE-on-consumer tooling** — `-ser` (smart expert reduction) + on-the-fly MLA tensors make big-MoE-over-VRAM (DeepSeek/Kimi-class) practical, an alternative to ktransformers.
7. **Qwen3.x MTP on `main`** — MTP rides the fork's main branch (and mainline llama.cpp has since merged its own, [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), 2026-05-16) — so no PR-branch building on either.

The cost: it's a **fork** with no tagged releases (rolling `main`, smaller community) — a second engine image to track vs the clean `ggml-org/llama.cpp:server-cuda`. That's the trade for the IQK quants. Use mainline `llamacpp/mtp` for the conservative production path; use ik_llama when you want the cutting-edge quant.

---

## Pros

- **IQK imatrix quants** — fork-exclusive, best quality-per-bit available in the GGUF world.
- **Cliff-immune** (inherits llama.cpp's ggml allocator — no Cliff 1/2 GDN OOM).
- **262K context on a single 3090** with MTP (q4_0 KV ~5 GB; verified, verify-stress 7/7 incl. 91K needle).
- **Broad hardware** — CUDA (incl. CC 7.0 Volta), ROCm, Apple Metal, Intel, CPU (same as mainline).
- **MoE-over-VRAM** path (`-ser` + MLA) for models bigger than your cards.
- Same `--jinja` / `--reasoning` server contract as mainline → froggeric chat template + thinking-off both work (validated on this fork; see "Gotchas").

## Cons

- **A fork to track** — rolling `main`, no tags, smaller community; pin by image digest.
- **GGUF-only** — no safetensors / vLLM-class continuous batching (single-stream `-np 1` is the sweet spot here).
- **Tooling lag** — some mainline llama.cpp server flags differ or arrive later (e.g. `--alias`, `--no-mmproj-offload` are in source builds but not every published image tag).
- Not the production default — that's vLLM (dual, max TPS) and mainline llama.cpp (single, conservative).

---

## Quick recipe

### 1. Pull the engine image
```bash
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu13-server   # cu12-server for CUDA 12 drivers
```

### 2. Get an IQK GGUF
```bash
hf download ubergarm/Qwen3.6-27B-GGUF Qwen3.6-27B-MTP-IQ4_KS.gguf \
  --local-dir $MODEL_DIR/qwen3.6-27b-gguf/ubergarm-mtp-iq4ks
# Always SHA256-verify multi-GB GGUFs after download.
```

### 3. Launch

Easiest — via the variant wizard / `switch.sh` (these two are registered):
```bash
bash scripts/switch.sh ik-llama/iq4ks-mtp          # text, 262K
bash scripts/switch.sh ik-llama/iq4ks-mtp-vision   # + mmproj (vision), 160K
```

Or directly via compose:
```bash
# MTP-only (default):
MODEL_DIR=$MODEL_DIR docker compose \
  -f models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp.yml up -d

# Two-stage ngram+MTP (code-optimized, experimental):
MODEL_DIR=$MODEL_DIR docker compose \
  -f models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/two-stage.yml up -d
```
Defaults: q4_0 KV, 200K ctx (MTP) / 131K ctx (two-stage), MTP n=2, native template, thinking-off. Overrides:
- **Max context:** **200K** is the max-safe default (fills cleanly with ~1.1 GB margin). 262144 is the model's native max but *boots ≠ fills* — `UBATCH_SIZE=512 CTX_SIZE=262144` boots but crosses the agent-safety margin at high fill, so 200K is the recommended ceiling (see [`docs/CLIFFS.md`](../CLIFFS.md)).
- **Higher KV fidelity:** `KV_TYPE=q8_0` (caps ~131-200K — q8_0 KV @262K OOMs).
- **Reasoning on:** `REASONING=on` (pair with `MTP_DRAFT_N_MAX=5 DRAFT_P_MIN=0.5` — reasoning text drafts deeper).

---

## Tuning levers (ik-specific)

| Flag | What it does | Default here |
|---|---|---|
| `-ctk` / `-ctv` | KV cache quant (`q4_0` / `q8_0` / `f16`) — biggest VRAM lever | `q4_0` |
| `-khad` / `--k-cache-hadamard` | Hadamard transform on K-cache → better quantized-KV accuracy (no VRAM cost) | on |
| `-vhad` / `--v-cache-hadamard` | Hadamard transform on V-cache → modest additional accuracy gain (no VRAM cost) | on |
| `--merge-qkv` (`-mqkv`) | Fused QKV projection | on |
| `--merge-up-gate-experts` (`-muge`) | Fused MoE up/gate (no-op on dense models) | — |
| `-ser N,f` | Smart expert reduction (big-MoE) | — |
| `--multi-token-prediction` + `--draft-max` / `--draft-p-min` | Built-in MTP spec-decode (single-stage) | n=2 / p-min 0.0 |
| `--spec-stage` (two-stage) | Chain ngram self-spec + MTP fallback. Better for code than MTP alone. See `iq4ks-two-stage.yml` | — |
| `--recurrent-ckpt-mode` | Pre-allocate VRAM for hybrid (DeltaNet) SSM state on draft rejection. `auto` (default) or `per-step` | `auto` |
| `-ctk-first` / `-ctv-first` | Higher-precision KV for first N layers (early layers = more quality-sensitive) | off |
| `--parallel-tool-calls` (`-ptc`) | Allow multiple tool calls per response — **ik-exclusive**, not in mainline llama.cpp | on |
| `-b` / `-ub` | Batch / micro-batch — `-ub 512` unlocks higher ctx at ~2-4% TPS | 4096 / 1024 |

---

## Two-stage spec-dec (ngram + MTP)

PR #1789 (merged 2026-05-15) enables chaining a self-speculator (ngram-mod, zero VRAM) with MTP as fallback. The engine tries ngram first — great at matching repeated code patterns (imports, boilerplate, refactor targets). If ngram can't produce enough tokens, MTP takes over.

```bash
--spec-stage ngram-mod:n_max=64,n_min=2,spec-ngram-size-n=16
--spec-stage mtp:n_max=3,draft-p-min=0.0
```

**When two-stage wins over MTP-only:** code-heavy workloads with rich context (refactoring, test generation, boilerplate). **When MTP-only is better:** highly novel generation with no repeated patterns, or short-context chat where ngram has insufficient history.

Compose: `iq4ks-two-stage.yml`. Status: experimental (PR is 7 days old). Bench pending.

## Quality optimizations

ik_llama has several quality-relevant flags beyond what the default composes use:

- **`-khad` / `-vhad`** — already on in all composes. Zero VRAM cost, better-than-f16 KV quality.
- **`--recurrent-ckpt-mode auto`** — already on. Makes MTP rejections cheaper on Qwen3.6's hybrid DeltaNet layers.
- **`-ctk-first q8_0,N`** — use q8_0 K-cache for the first N layers, q4_0 for the rest. Early attention layers are more quality-sensitive. Adds ~0.5 GB for 8 layers. Worth trying if you see quality issues on specific tasks, but khad+q4_0 already beats f16 perplexity so marginal gain is small.
- **IQ5_KS quant** (~18.5 GB) — higher quality-per-bit than IQ4_KS (lower perplexity per the quant author's published PPL, ~6.93 vs ~6.97; we have **not** independently benched it), +3.8 GB over IQ4_KS. Fits a single 3090 at 131K ctx but not 262K. No MTP-bundled variant exists (would use built-in MTP from model config, untested). Treat as a quality-comparison candidate, not a daily driver.

## ik_llama-specific gotchas

- **froggeric chat template works here** (unlike mainline). The mainline `llama.cpp` note that froggeric "silently suppresses `--reasoning off`" is a *mainline* issue — on ik_llama, froggeric v19 + `--reasoning off` suppresses thinking cleanly **and** renders tool-calls correctly (validated 2026-05-21). However, native (GGUF-embedded) template **won** the A/B on the 8-pack (103 vs 99, toolcall tied 9/9), so the ik composes default to native. froggeric remains available for vLLM where it helps the qwen3_coder parser.
- **The published image lags source on a few flags.** `--alias` and `--no-mmproj-offload` exist in from-source builds but not the `cu13-server` tag we pull — don't copy a from-source config verbatim. Check `llama-server --help` in the container.
- **Single-stream is the regime.** `-np 1` — this isn't a continuous-batching server. For multi-tenant, use vLLM.
- **MoE flags are no-ops on dense models** (`--merge-up-gate-experts`, `-ser` do nothing on dense Qwen3.6-27B; they matter for the MoE catalog).

---

## Measured on this stack (Qwen3.6-27B, IQ4_KS + MTP, single 3090)

| Metric | Value | vs shipped `llamacpp/mtp` (Q4_K_M) |
|---|---|---|
| Decode TPS (narr / code) | **~60 / ~72** | **~18–20% faster** than `llamacpp/mtp` (~50 / ~59) at matched 370 W — confirmed by a power-cap-controlled A/B (ik leads at 230 W *and* 370 W); the earlier "tie" was a wrong-engine measurement artifact, see [#184](https://github.com/noonghunna/club-3090/discussions/184) |
| Max context (1× 3090) | **262K** (q4_0 KV) | tie (mainline also reaches 262K via `-ub 512`) |
| VRAM @ 262K | ~22.5 GB | **~0.5–0.8 GB leaner** (a second edge, alongside the TPS lead) |
| verify-stress | 7/7 (incl. 91K Cliff 2 needle) | parity |
| Quality 8-pack | **103/150** | ≈ tie (mainline 102) |
| toolcall-15 | 60% (native template) | tie — native won the A/B on **both** engines |

> Bench: canonical prompt, 3 warmup + measured runs, **set-and-readback 370 W same-card matched comparison**, q4_0 KV / MTP n=2 / thinking-off. The honest finding: **ik is ~18–20% faster on decode TPS** (quality + context tied, ~0.5–0.8 GB leaner) — faster *and* leaner. The 2026-05-22 "tie" was a wrong-engine measurement artifact (its "tie" number is exactly mainline@370, which ik produces at no power setting); corrected 2026-05-23 via a power-cap-controlled A/B + 5 independent ik runs all at ~70 code. See [#184](https://github.com/noonghunna/club-3090/discussions/184) + [../../BENCHMARKS.md](../../BENCHMARKS.md).
> 
> **Two-stage (ngram+MTP):** not yet benched. Compose exists (`iq4ks-two-stage.yml`), PR #1789 merged 2026-05-15. Expected to outperform MTP-only on code workloads with repeated patterns; bench pending.

---

## When ik_llama.cpp is the right pick

- You want the **best quality-per-bit GGUF** (IQK imatrix) on a single card.
- You want **262K context on one 3090** with MTP decode speed.
- You're running a **big MoE that doesn't fit VRAM** and want `-ser`/MLA instead of ktransformers.
- You're **experimenting with newer quants** generally — this is the stack's advanced-quant track.
- **Code-heavy workloads** with repeated patterns — two-stage ngram+MTP (`iq4ks-two-stage.yml`) catches boilerplate/refactors that pure MTP misses.

## When to use something else

- **Production multi-tenant / max dual-card TPS** → vLLM (`dual` / `dual-turbo`).
- **Conservative single-card, mainline image, no fork** → llama.cpp (`llamacpp/mtp`).
- **Apple Silicon** → either llama.cpp (Metal) or ik_llama (also Metal).

## See also
- [../QUANTIZATION.md](../QUANTIZATION.md) — what IQK / imatrix / k-quants actually are
- [LLAMA_CPP.md](LLAMA_CPP.md) — the mainline sibling (shared cliff-immunity)
- [../INFERENCE_ENGINES.md](../INFERENCE_ENGINES.md) — full cross-engine comparison
- [../../BENCHMARKS.md](../../BENCHMARKS.md) — measured TPS across engines/configs
