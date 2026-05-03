# Multi-card (3+ GPUs) — derivation, constraints, scaling recipe

You have **3 or more GPUs** and want to know if club-3090 applies. Short
answer: yes, but we don't ship pre-baked configs because we can't
hardware-test them. This page explains what scales (and what doesn't)
when going beyond TP=2, the constraints to know, and the recipe for
deriving your own compose from `dual.yml`.

> **Honest disclaimer:** the maintainer rig is **2× RTX 3090 PCIe**.
> Everything below is derived from vLLM's documented tensor parallelism
> behavior + our measured TP=2 baseline + Marlin pad math. None of it is
> measured on 3+ card hardware locally. **If you have 4× / 8× hardware
> and run any of these configs, please share results via the
> [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml)
> issue template** — `bash scripts/report.sh --bench > my-rig.md`
> captures everything we'd want.

---

## TL;DR — what scales, what doesn't

| Aspect | TP=1 | TP=2 (measured) | TP=4 (derived) | TP=8 (derived) |
|---|---|---|---|---|
| Per-card weight share | 100% (~14 GB) | 50% (~7 GB) | 25% (~3.5 GB) | 12.5% (~1.75 GB) |
| KV pool capacity | smallest | 2× | ~4× | ~8× |
| Per-card peak VRAM (262K target) | 23.5+ GB tight | 23.6 GB tight | ~16-18 GB | ~10-12 GB |
| Cliff 2 single-prompt | fires at ~60K | doesn't fire (verified at 237K) | shouldn't fire | shouldn't fire |
| Per-stream TPS (PCIe-only) | baseline | ~same as TP=1 | likely lower | lower still |
| Concurrent throughput (multi-stream) | 1× | ~1.7-3.6× | derived ~2.5-7× | derived ~3-12× |
| Marlin pad-sub-tile-n patch | not needed | required | required | required |

**Two key takeaways:**

1. **More cards = much more headroom**, especially for long-context
   single-prompt workloads. On TP=4 the 24 GB-per-card pressure that
   drives Cliff 2 disappears entirely — weights and KV pool both split.
2. **Per-stream TPS doesn't scale** without NVLink. PCIe NCCL all-reduce
   overhead grows with TP count; per-stream decode at TP=4 may be
   *lower* than TP=2. Aggregate concurrent throughput still scales, but
   you don't get faster single-stream answers from more PCIe cards.

---

## Valid TP values for Qwen3.6-27B

vLLM's tensor parallelism splits attention heads across cards. The TP
value must divide both the attention head count AND the KV head count
cleanly. Qwen3.6-27B has:

- **80 attention heads** (factors: 1, 2, 4, 5, 8, 10, 16, 20, 40, 80)
- **5 KV heads** (factors: 1, 5)

The intersection — TP values that work — is **1, 2, 4, 5, 8, 10**. So:

| GPUs | Valid TP | Notes |
|---|---|---|
| 1 | TP=1 | Standard single-card. See [SINGLE_CARD.md](SINGLE_CARD.md). |
| 2 | TP=2 | Standard dual. See [DUAL_CARD.md](DUAL_CARD.md). |
| **3** | **TP=2 only** | TP=3 would split 5 KV heads as 5/3 = 1.67 per card — vLLM errors at boot. **Use TP=2 with 1 idle card** (set `CUDA_VISIBLE_DEVICES=0,1`), or run 2 single-card stacks on different ports. |
| **4** | **TP=4** | Each card gets 20 attention heads + 1.25 KV heads — vLLM splits with replication for fractional KV (handled internally). Production-viable if your rig has the slots + power + cooling. |
| **5** | **TP=5** | Theoretically valid (1 KV head per card, 16 attention heads per card). Unusual rig count; not common. |
| **6 or 7** | **TP=4 or TP=5 + spare cards** | TP=6/7 don't divide head count. Use TP=4 (idle 2-3 cards) or TP=5 (idle 1-2 cards). |
| **8** | **TP=8** | Datacenter-class. Each card gets 10 attention heads, splits KV heads via vLLM's internal handling. |
| **10** | **TP=10** | Server-class. Production-viable on data-center boards. |

**Critical: TP=3, TP=6, TP=7, TP=9 do NOT work.** vLLM errors at boot
("number of attention heads must be divisible by tensor parallel size").
If you have an awkward GPU count, use the next-lower valid TP and leave
the extras idle, or run separate stacks on different ports.

---

## Recipe — derive your config from `dual.yml`

`dual.yml` is the tested baseline. To scale to TP=N, copy it and change
**three lines**:

```diff
  command:
    - --tensor-parallel-size
-   - "2"
+   - "4"      # or 8, etc. — must be a valid TP value from the table above
    - --max-num-seqs
-   - "2"
+   - "4"      # bump proportional to TP — more cards = more concurrent streams
    - --max-num-batched-tokens
-   - "8192"
+   - "16384"  # optionally bump proportional to TP for longer prefill chunks
```

Everything else stays the same:
- `--gpu-memory-utilization 0.92` — same per-card budget
- `--kv-cache-dtype fp8_e5m2` — same KV class
- `--max-model-len 262144` — same target context (more cards = more
  total KV pool, but per-request max stays at 262K unless you raise it)
- `MTP n=3` spec-decode — same
- The Marlin pad-sub-tile-n patch mount stays — at higher TP, more
  out-features get sub-tile-split, so the patch is *more* likely to be
  needed, not less

Container name + port: pick something distinct so it doesn't collide
with your other variants:

```yaml
container_name: vllm-qwen36-27b-quad     # or octa
ports:
  - "${PORT:-8014}:8000"                  # 8010-8013 are dual variants
```

---

## What to expect on TP=4 (4× 3090 PCIe)

These are **derived expectations**, not measurements. Adjust to your
rig's actual numbers when you bench:

- **Boot time:** longer than TP=2 (more NCCL handshakes, more weight
  shards to load). 90s-3min cold boot typical for vLLM at this size.
- **Per-card VRAM:** roughly 16-18 GB at peak with `--max-model-len
  262144 + max-num-seqs 4 + fp8 KV`. Significant headroom vs TP=2's
  23.6 GB — you can push max-num-seqs higher or bump max-model-len if
  your prompts need it.
- **Per-stream decode TPS:** likely lower than TP=2's ~70 narr / ~89
  code (PCIe NCCL overhead). Could be ~50-65 narr / ~70-80 code as a
  starting estimate — verify with `bench.sh`.
- **Aggregate concurrent throughput:** higher than TP=2 across multiple
  streams. The KV pool 2× larger than TP=2 means more in-flight
  requests fit, and each card's compute is freed by smaller weight
  share.
- **Cliff 2:** doesn't apply. DeltaNet GDN forward state splits across
  4 cards — single-prompt at 262K should work without the per-card 24
  GB pressure that drives Cliff 2 on single-card.
- **NVLink bridges if available:** would substantially help per-stream
  TPS. PCIe-only at TP=4 is functional but compute-bound on NCCL.

---

## What to expect on TP=8 (8× 3090 / A6000)

Server-class setup. Most users at this scale are on rack hardware (DGX,
4U server chassis, dedicated cooling). The per-card pressure essentially
disappears:

- **Per-card peak VRAM:** ~10-12 GB. You have headroom to do almost
  anything — bump max-num-seqs to 8+, push max-model-len higher,
  experiment with TQ3 + Genesis stack from `dual-turbo.yml`.
- **Per-stream decode TPS:** without NVLink fabric, likely lower than
  TP=4. Server-class cards (A6000, A100) often have NVLink — that
  changes the per-stream calculus dramatically.
- **Aggregate throughput:** scales near-linearly with N if you have
  multi-stream load.

If you're on a server-class rig with NVLink: per-stream TPS could
approach 1.6-1.8× single-card vs the ~1.0× we see on PCIe TP=2. That
makes TP=8 with NVLink a meaningfully different regime than what we
measure.

---

## Cross-rig data we'd love

If you have 4× / 8× hardware and run any config, please share via
[Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml).
The single command that captures everything:

```bash
bash scripts/report.sh --bench > my-rig.md
```

Specifically interested in:

- **TP=4 on 4× 3090 PCIe** — does Cliff 2 disappear entirely as
  expected? What's per-stream TPS vs concurrent throughput?
- **TP=4 with NVLink topology** (e.g. NVLink across pairs) — how does
  per-stream TPS compare to PCIe-only TP=2?
- **TP=8 on 8× A6000 / A100** — first server-class data point we'd
  collect.
- **TP=4 on mixed cards** (e.g. 2× 3090 + 2× 4090, or 4× modded 3080
  20GB) — does vLLM's per-card weight balance handle asymmetric VRAM
  ceilings cleanly? Asymmetric setups need `--gpu-memory-utilization`
  tuned to the *smallest* card's free VRAM.

---

## Why we don't ship pre-baked configs

Three reasons:

1. **We can't hardware-test them.** Maintainer rig is 2× 3090. Pretending
   we tested `quad.yml` would set false confidence.
2. **Hardware combinations explode.** 4× 3090 vs 4× A5000 vs 4× A6000 vs
   2×3090 + 2×4090 vs 4× modded 3080 — each has different VRAM, NVLink
   topology, power profile, allreduce characteristics. A single
   "quad.yml" can't be optimal for all.
3. **Users at this scale are typically experienced.** If you have a
   workstation chassis or rack with 4-8 GPUs, you've already done the
   hardware homework. What you need from us is the methodology, the
   constraints, and the dial — not a hand-held tested compose.

If a community member contributes a tested compose for their specific
topology (with `verify-stress.sh` passing + `bench.sh` numbers), we'll
ship it under `models/qwen3.6-27b/vllm/compose/` with credit and a
header noting which rig validated it.

---

## See also

- [`SINGLE_CARD.md`](SINGLE_CARD.md) — 1× GPU baseline (where Cliff 2 lives)
- [`DUAL_CARD.md`](DUAL_CARD.md) — measured 2× GPU configs (your starting point for derivation)
- [`HARDWARE.md`](HARDWARE.md) — Ampere/Ada/Hopper notes, NVLink, power
- [`UPSTREAM.md`](UPSTREAM.md) — vLLM PRs we depend on (incl. our [#40361 Marlin pad-sub-tile-n](https://github.com/vllm-project/vllm/pull/40361) which becomes more relevant at higher TP)
- [`models/qwen3.6-27b/INTERNALS.md`](../models/qwen3.6-27b/INTERNALS.md) — head count + KV head structure (basis for the TP divisibility math above)
