# Gemma 4 31B — on 2× RTX 3090

**Run [Gemma 4 31B](https://blog.google/technology/developers/gemma-4/) — with vision, tool calling, and MTP spec-decode — on 2× RTX 3090s.**

> ⚠️ **Single-card boot OOMs on 24 GB Ampere** regardless of KV format. Needs ≥32 GB single-card (validated on RTX 5090 by [@apnar](https://github.com/noonghunna/club-3090/discussions/67#discussioncomment-16832042)).

---

## Deployment

See [`docs/DUAL_CARD.md`](../../docs/DUAL_CARD.md) for workload-driven config picks. TL;DR:

| Config | Max ctx | Narr / Code TPS | Best for |
|--------|---------|----------------|----------|
| `vllm/gemma-bf16-mtp` (default) | 32K | 106 / 141 | General-purpose, vision + tools |
| `vllm/gemma-int8-mtp` | 98K default, 262K via `CTX=262144 MAX_NUM_SEQS=1` | 95 / 126 | Long-context via INT8 PTH KV (PR #40391) |
| `vllm/gemma-mtp-tp1` | 8K | community-provided 32 GB+ path | Single-card fp8 risk path; maintainer live check required on v0.21.0 |

Run via:
```bash
bash scripts/launch.sh --variant vllm/gemma-bf16-mtp     # MTP default
bash scripts/launch.sh --variant vllm/gemma-int8-mtp    # long-context
```

---

## Models

- **Target:** [`Intel/gemma-4-31B-it-int4-AutoRound`](https://huggingface.co/Intel/gemma-4-31B-it-int4-AutoRound) (~21.2 GB, vision preserved)
- **Draft (MTP):** [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant) (0.5B / 927 MB BF16)

## Key details

| Aspect | Notes |
|--------|-------|
| **Quants** | Intel AutoRound INT4 |
| **KV** | bfloat16 (32K) or INT8 PTH via vendored PR #40391 (262K) |
| **Drafter** | MTP n=3/4 (Google official assistant) |
| **Vision** | ✅ Yes |
| **Tools** | ✅ `--tool-call-parser gemma4` |
| **NVLink** | Auto-detected via `NVLINK_MODE` env var |

## Upstream tracker

- [vLLM PR #41745](https://github.com/vllm-project/vllm/pull/41745) — Gemma 4 MTP support (merged)
- [vLLM PR #40391](https://github.com/vllm-project/vllm/pull/40391) — INT8 PTH KV; vendored for `vllm/gemma-int8-mtp`
- [Discussion #67](https://github.com/noonghunna/club-3090/discussions/67) — first Ampere consumer cross-rig data
