# Gemma 4 31B — on 2× RTX 3090

**Run [Gemma 4 31B](https://blog.google/technology/developers/gemma-4/) — with vision and tool calling — on 2× RTX 3090s, on stock vLLM v0.24.0 (overlay-free).**

> ⚠️ **Single-card boot OOMs on 24 GB Ampere** regardless of KV format. Needs ≥32 GB single-card (validated on RTX 5090 by [@apnar](https://github.com/noonghunna/club-3090/discussions/67#discussioncomment-16832042)).

---

## Deployment

See [`docs/DUAL_CARD.md`](../../docs/DUAL_CARD.md) for workload-driven config picks. TL;DR:

| Config | Max ctx | Decode TPS | Best for |
|--------|---------|----------------|----------|
| `vllm/gemma-31b-dual` (default) ⭐ | **224K** | ~59 | General-purpose, vision + tools — **stock vLLM v0.24.0, overlay-free** |

Run via:
```bash
bash scripts/launch.sh --variant vllm/gemma-31b-dual     # bf16 @224K, v0.24.0, overlay-free
```

> **v0.24.0 consolidation (2026-07-02):** the 31b is now a single overlay-free **bf16** dual slug on
> `vllm-stable`. The v0.22.0 composes (`gemma-int8-mtp` = 262K int8-PTH + PR #40391, `gemma-bf16-mtp`
> = 131K, `gemma-mtp-tp1`, `gemma-31b-qat-w4a16-dual`) are **deprecated** (`switch.sh --list --all`).
> MTP is off — Gemma-4 MTP × tool-calling is broken on v0.24.0 (vLLM #39043 / #42006). The 262K
> int8-PTH path returns overlay-free when PR #40391 merges upstream (on v0.24.0 int8-PTH allocates
> 262K but silently craters recall past ~32K, so bf16 @224K is the honest default).

---

## Models

- **Target:** [`Intel/gemma-4-31B-it-int4-AutoRound`](https://huggingface.co/Intel/gemma-4-31B-it-int4-AutoRound) (~21.2 GB, vision preserved)
- **Draft (MTP):** [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant) (0.5B / 927 MB BF16)

## Key details

| Aspect | Notes |
|--------|-------|
| **Quants** | Intel AutoRound INT4 |
| **KV** | bfloat16 @224K on stock v0.24.0 (overlay-free). The int8-PTH + PR #40391 262K path is deprecated (returns free when #40391 merges) |
| **Drafter** | none — MTP disabled on v0.24.0 (Gemma-4 MTP × tools broken, vLLM #39043 / #42006) |
| **Vision** | ✅ Yes |
| **Tools** | ✅ `--tool-call-parser gemma4` |
| **NVLink** | Auto-detected via `NVLINK_MODE` env var |

## Upstream tracker

- [vLLM PR #41745](https://github.com/vllm-project/vllm/pull/41745) — Gemma 4 MTP support (merged)
- [vLLM PR #40391](https://github.com/vllm-project/vllm/pull/40391) — INT8 PTH KV page-align (OPEN/unmerged). The deprecated `vllm/gemma-int8-mtp` (v0.22.0) vendors it for 262K; on v0.24.0 int8-PTH craters recall without it, so the default is bf16 (`vllm/gemma-31b-dual`). 262K int8-PTH returns when this merges.
- [Discussion #67](https://github.com/noonghunna/club-3090/discussions/67) — first Ampere consumer cross-rig data
