# Multi-card (3+ GPUs) — derivation, constraints, scaling recipe

You have **three or more GPUs**. This page covers the one decision that matters most (replicate or tensor-parallel), the constraints that will stop a config booting, the two shipped 4-card baselines, and how to derive your own TP=N.

> **Two cards?** → [`DUAL_CARD.md`](DUAL_CARD.md). **One?** → [`SINGLE_CARD.md`](SINGLE_CARD.md).

## ⛔ This rig has 2 GPUs. Permanently.

Nothing above TP=2 is validated here, and nothing above TP=2 ever will be — that is the maintainer's hardware ceiling, not a backlog item. Filing "boot the 4-card slug" as pending work is a category error.

What you get instead: every shipped TP=4 config has been **cross-validated on independent community rigs** — [@Whamp](https://github.com/noonghunna/club-3090) (4× 3090), [@alanspires](https://github.com/noonghunna/club-3090/issues/127) (6× 3090 VFIO), [@ryanmpelletier](https://github.com/noonghunna/club-3090), [@MoppelMat](https://github.com/noonghunna/club-3090), [@alesha-pro](https://github.com/noonghunna/club-3090/discussions/773) — and the derivation recipe below is a three-line diff off the config we run daily.

**TP=8 has zero boots anywhere.** Those sections are arithmetic, and labelled as such.

If you're running 4+ GPUs you have more relevant hardware than we do. The constraints and the recipe are ours to give; the validation on your card count is the one thing only you can produce. `bash scripts/report.sh --full` — one command, ~35 min, and it becomes the next row in [`BENCHMARKS.md`](../BENCHMARKS.md) with your name on it.

---

## ⭐ Decide this first: replicate, or tensor-parallel?

> **If you read one section of this page, read this one.** Getting it wrong costs up to **3.4×** aggregate throughput, and no amount of TP tuning further down recovers it.

Everything else here reasons in the **TP dimension** — how do I split one model across N cards? For **aggregate** throughput that is frequently the wrong question. If the model fits on one card, running **N independent instances** beats splitting it N ways, by a lot.

Measured by [@alesha-pro](https://github.com/noonghunna/club-3090/discussions/773) on 4× 3090 PCIe, all arms matched at 220 W, peak aggregate tok/s:

| Model | TP=4, 1 instance | replicated | gain |
|---|--:|--:|--:|
| gemma-4-12B AWQ | 1,007 | **3,425** (TP=1 × 4) | **3.40×** |
| Qwen3.6-27B INT8-W8A8 | 358 | **741** (TP=2 × 2) | 2.07× |
| Qwen3.6-27B INT4 | 307 | **737** (TP=2 × 2) | 2.40× |

**The rule: latency wants max TP, throughput wants max instances.**

The trap is that **the same run flips winner** depending which number you read. AutoRound INT4, one rig, one sitting, one power cap:

| | TP=2 | TP=4 |
|---|--:|--:|
| single stream (wall) | 53.80 | **69.26** |
| aggregate peak @ c=64 | **473.1** | 294.8 |

TP=4 is **29% faster for one user and 38% slower for a full queue.** "Which TP should I run" has no answer without "for what workload".

**Our own concurrency sweep agrees from the other direction:** the dense 27B's batching knee is **N=2** — decode-aggregate *halves* by N=8 because per-stream falls faster than N grows. The 35B-A3B MoE holds ~250–270 flat to N=16. Past a low N, stop feeding one big engine and start adding engines.

**Before you replicate:** each instance needs its own full weight copy (the model must fit on one card *with* a useful KV pool), its own port, and its own KV pool — you lose the pooled KV that is the main reason to run TP=4 for long single prompts. Replication wins for **many short-to-medium requests**; TP wins for **one very long one**. No replication compose ships today — [`PODS.md`](PODS.md) is the closest thing.

---

## What scales at TP=4, and what doesn't

| Aspect | TP=1 | TP=2 (measured) | TP=4 (measured) | TP=8 (derived) |
|---|---|---|---|---|
| Per-card weight share | 100% (~14 GB) | 50% (~7 GB) | 25% (~3.5 GB) | 12.5% (~1.75 GB) |
| KV pool capacity | smallest | 2× | ~4× | ~8× |
| Per-card peak VRAM @262K | 23.5+ GB tight | 23.6 GB tight | **23.5 GB fp8** | ~10–12 GB |
| Cliff 2 single-prompt | fires at ~60K | doesn't fire (237K verified) | **passes 91K needle** | shouldn't fire |
| Per-stream TPS (PCIe) | baseline | ~same as TP=1 | **63 / 76 fp8** | lower still |
| Concurrent throughput | 1× | ~1.7–3.6× | **6.77× fp8 @262K** | derived ~3–12× |
| Marlin pad patch | not needed | required | required | required |

**More cards buy headroom, not per-stream speed.** On TP=4 the 24 GB-per-card pressure that drives Cliff 2 disappears — weights and KV both split.

⚠️ **"Per-stream TPS doesn't scale on PCIe" is a property of our recipe, not a law of topology.** Our TP=2 baseline already folds in ~1.85× of MTP drafter, so there's little left for extra cards to add. On a matched-recipe A/B (same rig, same sitting, @alesha-pro): **prefill @90K went 1,202 → 1,948 tok/s (+62%) and TTFT @90K 73.1 s → 45.6 s (−37.6%)**. Cards 3 and 4 buy **prefill and TTFT**, not decode.

⚠️ **Two asymmetries when comparing dual vs multi4 numbers:** TP=2 can run vLLM's custom all-reduce and TP≥3 cannot (engine-gated on PCIe — [#786](https://github.com/noonghunna/club-3090/issues/786)); and on **llama.cpp** (tensor-split, not TP) the same box gained only 3–13% going 2→4, with repeat spread wide enough to swamp it. On a GGUF engine, "more cards ≠ faster per stream" holds much more strongly.

---

## Valid TP values

vLLM asserts that the **attention** head count divides the TP size. KV heads do not have to divide it — when there are fewer KV heads than ranks, vLLM **replicates** them. Qwen3.6-27B and Qwen3.8-27B are identical here (`config.json` → `text_config`):

- **24 attention heads** (factors: 1, 2, 3, 4, 6, 8, 12, 24) — the binding constraint
- **4 KV heads** — shard cleanly to TP=4, then replicate above it
- head_dim 256

| GPUs | Valid TP | Attn/card | KV/card | Notes |
|---|---|--:|---|---|
| 1 | TP=1 | 24 | 4 | → [SINGLE_CARD.md](SINGLE_CARD.md) |
| 2 | TP=2 | 12 | 2 | → [DUAL_CARD.md](DUAL_CARD.md) |
| **3** | **TP=2 only** | — | — | 24 ÷ 3 = 8 attention heads is fine, but **4 KV heads across 3 ranks neither divides nor replicates evenly** — vLLM errors at boot. Use TP=2 with 1 idle card (`CUDA_VISIBLE_DEVICES=0,1`), or run 2 single-card stacks on different ports. |
| **4** | **TP=4** | 6 | 1 | The shipped multi-card tier. |
| **5** | **TP=4 + 1 idle** | — | — | ⚠️ **TP=5 does NOT work** — 24 ÷ 5 is not an integer. |
| **6 or 7** | **TP=4 + spares** | — | — | Neither divides 24 evenly alongside the KV constraint. |
| **8** | **TP=8** | 3 | replicated ×2 | 3 attention heads/card; the 4 KV heads replicate, so **per-card KV stays at the TP=4 figure** and only weights shard 8-way. |
| **10** | **TP=8 + 2 idle** | — | — | ⚠️ **TP=10 does NOT work** — 24 ÷ 10 is not an integer. |

**TP=3, 5, 6, 7, 9, 10 do NOT work.** vLLM errors at boot (*"number of attention heads must be divisible by tensor parallel size"*). Use the next-lower valid TP and leave the extras idle.

> ⚠️ Because KV heads replicate above TP=4, **going 4 → 8 cards does not shrink per-card KV** — it only shards weights. Budget accordingly.

### Picking which cards to use

On awkward counts, pick the **best-connected** pair or quad, not the first N. Inspect with `nvidia-smi topo -m` and prefer `NV#` > `PIX` > `PXB` > `PHB` > `SYS`, then pin with `CUDA_VISIBLE_DEVICES`. Full treatment of link classes, NUMA, slots and BIOS: [`PCIE_P2P.md`](PCIE_P2P.md).

⚠️ **TP across VRAM-mismatched cards is poor** — vLLM sizes the KV pool from the *smallest* card, so a 24 GB + 12 GB pair gives you two 12 GB cards. Compute-mismatched (same VRAM, different speed) is fine; the slower card just sets the pace.

---

## Shipped TP=4 baselines

| Slug | Weights | KV | Max ctx | Port | Notes |
|---|---|---|--:|--:|---|
| `vllm/qwen-27b-multi-fast` ⭐ | AutoRound INT4 | fp8 e4m3 | 262144 | 8014 | The primary 4-card config. Cross-rig: 74.76 / 90.83 TPS (@ryanmpelletier, all-x16). |
| `vllm/qwen-27b-multi-max` | FP8 | fp8 e4m3 | 262144 | 8015 | Highest weight fidelity. Cross-rig: 74.10 / 91.30 (@ryanmpelletier) · 79.23 / 101.61 (@MoppelMat). |
| `vllm/qwen38-27b-multi4-max` | official FP8 | bf16 | 262144 | 8092 | 🐣 Qwen3.8. **Never booted on 4 cards** — first community boot IS the validation. |
| `vllm/qwen38-27b-multi4-fast` | AutoRound INT4 + int8 act | fp8 e4m3 | 262144 | 8114 | 🧪 Qwen3.8. Never booted on 4 cards. |

### TP=8 slugs (never booted, anywhere)

| Slug | Weights | KV | Max ctx | Port | Notes |
|---|---|---|--:|--:|---|
| `vllm/qwen38-27b-multi8-max` | official FP8 | bf16 | 262144 | 8093 | 🐣 `--force`, hidden from `--list` |
| `vllm/qwen38-27b-multi8-fast` | AutoRound INT4 + int8 act | fp8 e4m3 | 262144 | 8097 | 🧪 `--force` |

⚠️ **TP=8 is not the same shard shape as TP=2/4.** With 4 KV heads, vLLM **replicates** them across rank pairs rather than splitting, so per-card KV stays at roughly the TP=4 figure — only the **weights** take the full 8-way split (~3.6 GiB/card vs ~7.2 at TP=4). The 8-card gain over 4 is weight headroom, bought with more all-reduce on a PCIe bus. On the qwen3.6 equivalent, single-stream decode was already ~flat from 2 → 4. Expect flat or worse; expect nothing quantitative until someone measures it.

The Qwen3.8 dual-card tier **has** been measured — `vllm/qwen38-27b-dual-max` at 67.4 / 85.8 TPS, [DUAL_CARD.md](DUAL_CARD.md). Its numbers transfer on topology but **not** on pool size or concurrency: since 2026-08-15 the multi slugs run bf16 KV while the dual runs fp8 e4m3.

⚠️ **Both qwen3.6 multi slugs are exposed to open [vllm#50021](https://github.com/vllm-project/vllm/pull/50021)** — a GDN spec-decode wild write that kills workers under sustained agent traffic. Mitigate with `SPEC=off`. Every measured row: [`BENCHMARKS.md`](../BENCHMARKS.md).

---

## Recipe — derive your own TP=N from `dual.yml`

Copy `dual/autoround-int4/fp8-mtp.yml` (`vllm/dual`, the tested 2-card baseline) or `multi4/autoround-int4/mtp.yml` (`vllm/qwen-27b-multi-fast`, the measured 4-card one) and change **three lines**:

```diff
  command:
    - --tensor-parallel-size
-   - "2"
+   - "4"      # must be a valid TP value from the table above
    - --max-num-seqs
-   - "2"
+   - "4"      # bump proportional to TP
    - --max-num-batched-tokens
-   - "8192"
+   - "16384"  # optionally bump for longer prefill chunks
```

Everything else stays: `--gpu-memory-utilization 0.92` (same per-card budget), `--kv-cache-dtype`, `--max-model-len 262144`, MTP n=3, and the Marlin pad patch mount — at higher TP *more* out-features get sub-tile-split, so the patch is more likely to be needed, not less.

Give it a distinct container name and port so it doesn't collide:

```yaml
container_name: vllm-qwen36-27b-octa
ports:
  - "${PORT:-8016}:8000"
```

⚠️ **PCIe-only with no NVLink means more ranks = more all-reduce on a slow bus.** Custom all-reduce stays disabled above TP=2 regardless — it's engine-gated, not a config choice.

---

## Cross-rig data we'd love

Anything on 3+ cards, in rough priority order:

1. **A TP=8 boot of anything.** Zero exist. The TP=8 rows above are arithmetic.
2. **A qwen3.8-27b multi4 or multi8 boot** — `vllm/qwen38-27b-multi4-max` has never run on any card count.
3. **A replication-vs-TP A/B on your rig** — the single highest-leverage number on this page, and it's cheap to produce.
4. **Odd card counts** (3, 5, 6) — does the "next-lower valid TP + idle cards" advice actually hold?

```bash
bash scripts/report.sh --full     # ~35 min, redacted, paste-ready
```

Report via the `numbers-from-your-rig` issue template.

---

---

## Keeping up

Slugs, defaults and measured numbers move faster than this page. For the latest:

- 📣 **[Announcements](https://github.com/noonghunna/club-3090/discussions/categories/announcements)** — new models and tiers land here first, with the numbers and the caveats. Recent: [Qwen3.8-27B](https://github.com/noonghunna/club-3090/discussions/993) · [the fast + NVFP4 tier](https://github.com/noonghunna/club-3090/discussions/1024).
- 💬 **[Discord](https://discord.gg/gzdfjhj5yN)** — synchronous Q&A, hardware questions, what people are actually running.
- 📋 **[Discussions](https://github.com/noonghunna/club-3090/discussions)** — cross-rig benchmark drops and "should I tune X" threads, searchable.
- ⚙️ **`bash scripts/switch.sh --list`** — the authoritative slug matrix for *your* machine. Always more current than any hand-written table, including the ones above.

## See also

- **[PCIE_P2P.md](PCIE_P2P.md)** — topology classes, NUMA, slots, BIOS, enabling P2P without NVLink.
- **[KV_MATH.md](KV_MATH.md)** — predicting per-card VRAM before you boot.
- **[PODS.md](PODS.md)** — running multiple instances on one host (the replication path).
- **[BENCHMARKS.md](../BENCHMARKS.md)** — every measured row, cross-rig, with attribution.
- **[DUAL_CARD.md](DUAL_CARD.md)** · **[SINGLE_CARD.md](SINGLE_CARD.md)** — the other topologies.
