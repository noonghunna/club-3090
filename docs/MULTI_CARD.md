# Multi-card (3+ GPUs) — derivation, constraints, scaling recipe

You have **3 or more GPUs** and want to know if club-3090 applies. Short
answer: yes. We ship one community-validated 4×3090 baseline and keep
other 3+ GPU configs as derivation recipes until someone measures them.
This page explains what scales (and what doesn't) when going beyond TP=2,
the constraints to know, and how to derive your own compose when `dual4.yml`
isn't your topology.

> **Validation note:** the maintainer rig is **2× RTX 3090 PCIe**, but
> Whamp's 4× RTX 3090 PCIe rig validated the TP=4 fp8/MTP baseline in
> [discussion #26](https://github.com/noonghunna/club-3090/discussions/26)
> on 2026-05-03. The TP=8+ sections remain derived expectations. If you
> have 4× / 8× hardware and run additional configs, please share results via
> the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml)
> issue template — `bash scripts/report.sh --full > my-rig.md` captures
> everything we'd want (verify + stress + soak-continuous + bench, ~35 min).

---

## TL;DR — what scales, what doesn't

| Aspect | TP=1 | TP=2 (measured) | TP=4 (measured) | TP=8 (derived) |
|---|---|---|---|---|
| Per-card weight share | 100% (~14 GB) | 50% (~7 GB) | 25% (~3.5 GB) | 12.5% (~1.75 GB) |
| KV pool capacity | smallest | 2× | ~4× | ~8× |
| Per-card peak VRAM (262K target) | 23.5+ GB tight | 23.6 GB tight | **23.5 GB fp8 / 22.0 GB DFlash** | ~10-12 GB |
| Cliff 2 single-prompt | fires at ~60K | doesn't fire (verified at 237K) | **passes 91K needle** | shouldn't fire |
| Per-stream TPS (PCIe-only) | baseline | ~same as TP=1 | **63/76 fp8, 64/104 DFlash** | lower still |
| Concurrent throughput (multi-stream) | 1× | ~1.7-3.6× | KV pre-check **6.77× fp8, 2.27× DFlash @ 262K** | derived ~3-12× |
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

## Shipped TP=4 baselines — `vllm/dual4` and `vllm/dual4-dflash`

For 4× RTX 3090 PCIe, start with the measured fp8/MTP compose:

```bash
bash scripts/switch.sh vllm/dual4
```

`docker-compose.dual4.yml` keeps the `dual.yml` fp8/MTP feature set and
changes TP/streams from 2 → 4. Validation on Whamp's 4× 3090 PCIe rig:

- boots at `max_model_len=262144`, `max_num_seqs=4`
- vLLM reports GPU KV cache size **483,200 tokens** and **6.77×** maximum concurrency for 262K-token requests
- `verify-full.sh` passes
- `verify-stress.sh` passes 7/7; probe 7 recalls **58,569-token** and **91,070-token** needles
- `bench.sh`: **63.01 narr / 76.25 code wall TPS**, peak **23,494 MiB/card**

Use the DFlash variant when code throughput matters more than stream count
and you can download the gated `z-lab/Qwen3.6-27B-DFlash` draft:

```bash
WITH_DFLASH_DRAFT=1 bash scripts/setup.sh qwen3.6-27b
bash scripts/switch.sh vllm/dual4-dflash
```

`docker-compose.dual4-dflash.yml` keeps full 262K context but uses FP16 KV
and admits two full-context streams:

- boots at `max_model_len=262144`, `max_num_seqs=2`
- vLLM reports GPU KV cache size **207,264 tokens** and **2.27×** maximum concurrency for 262K-token requests
- `verify-full.sh` passes
- `verify-stress.sh` passes 7/7; probe 7 recalls **58,570-token** and **91,070-token** needles
- `bench.sh`: **64.00 narr / 104.40 code wall TPS**, peak **21,960 MiB/card**
- DFlash AL during code bench: **4.43 / 4.37 / 4.35** last observed samples

Single-stream TPS is lower than the 2-card DFlash variants on PCIe-only
allreduce, so use TP=4 DFlash for full-262K code-heavy work and two admitted
streams — not as a replacement for the fastest 2-card short-prompt DFlash path.

## Recipe — derive your own config from `dual.yml`

`dual.yml` is the tested 2-card baseline and `dual4.yml` is the measured
4-card baseline. To scale to another TP=N, copy one of those and change
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
with your other variants. `dual4.yml` uses `vllm-qwen36-27b-dual4` and
port `8015`; reserve a different name/port for further experiments:

```yaml
container_name: vllm-qwen36-27b-octa
ports:
  - "${PORT:-8016}:8000"
```

---

## What we measured on TP=4 (4× 3090 PCIe)

Measured 2026-05-03 on Whamp's 4× RTX 3090 PCIe rig:

- **fp8/MTP boot time:** 355s cold after model/image cache populated.
- **fp8/MTP pre-check:** `max_model_len=262144`, `max_num_seqs=4`, GPU KV
  cache size 483,200 tokens, max concurrency 6.77× at 262K.
- **fp8/MTP VRAM:** 21,714 MiB idle after boot; 23,494 MiB/card peak
  during canonical bench.
- **fp8/MTP TPS:** 63.01 narrative / 76.25 code wall TPS.
- **MTP AL:** last three code-bench metrics showed mean acceptance length
  3.42 / 3.53 / 3.62.
- **DFlash boot time:** 375s cold after model/image cache populated.
- **DFlash pre-check:** `max_model_len=262144`, `max_num_seqs=2`, GPU KV
  cache size 207,264 tokens, max concurrency 2.27× at 262K.
- **DFlash VRAM:** 21,940 MiB idle after boot; 21,960 MiB/card peak during
  canonical bench.
- **DFlash TPS:** 64.00 narrative / 104.40 code wall TPS.
- **DFlash AL:** last three code-bench metrics showed mean acceptance length
  4.43 / 4.37 / 4.35.
- **Cliff 2:** canonical `verify-stress.sh` probe 7 passes at both large
  rungs on both TP=4 variants: ~58.6K tokens and 91K tokens recalled correctly.
- **Trade-off:** PCIe allreduce makes single-stream decode slower than
  TP=2, but TP=4 provides more full-context concurrency and the first
  published 4×3090 Cliff 2 boundary data.

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
bash scripts/report.sh --full > my-rig.md
```

That's verify-full + verify-stress 7/7 + SOAK_MODE=continuous + canonical bench in one ~35-min pass. Use `--bench` instead if you want bench numbers without the soak (faster but doesn't probe Cliff 2b).

Specifically interested in:

- **More TP=4 on 4× 3090 PCIe** — does your motherboard / power cap / PCIe
  topology match or beat Whamp's 63 / 76 TPS baseline? What's concurrent throughput?
- **TP=4 with NVLink topology** (e.g. NVLink across pairs) — how does
  per-stream TPS compare to PCIe-only TP=2?
- **TP=8 on 8× A6000 / A100** — first server-class data point we'd
  collect.
- **TP=4 on mixed cards** (e.g. 2× 3090 + 2× 4090, or 4× modded 3080
  20GB) — does vLLM's per-card weight balance handle asymmetric VRAM
  ceilings cleanly? Asymmetric setups need `--gpu-memory-utilization`
  tuned to the *smallest* card's free VRAM.

---

## Why we ship only one pre-baked 4-card config

We now ship `dual4.yml` because a community rig validated that exact
4× RTX 3090 PCIe topology with `verify-full.sh`, `verify-stress.sh`, and
`bench.sh`. We still avoid a broad matrix of untested 4+ GPU composes:

1. **Hardware combinations explode.** 4× 3090 vs 4× A5000 vs 4× A6000 vs
   2×3090 + 2×4090 vs 4× modded 3080 — each has different VRAM, topology,
   power profile, and allreduce characteristics.
2. **Variant count needs discipline.** A single measured fp8/MTP TP=4
   baseline is useful; a directory full of derived-but-unvalidated variants
   would create false confidence.
3. **Users at this scale are typically experienced.** If you have a
   workstation chassis or rack with 4-8 GPUs, you've already done the
   hardware homework. What you need from us is the methodology, the
   constraints, and one validated starting point.

If a community member contributes another tested compose for a specific
topology or workload (with `verify-stress.sh` passing + `bench.sh`
numbers), we'll ship it with credit and a header noting which rig
validated it.

---

## See also

- [`SINGLE_CARD.md`](SINGLE_CARD.md) — 1× GPU baseline (where Cliff 2 lives)
- [`DUAL_CARD.md`](DUAL_CARD.md) — measured 2× GPU configs (your starting point for derivation)
- [`HARDWARE.md`](HARDWARE.md) — Ampere/Ada/Hopper notes, NVLink, power
- [`UPSTREAM.md`](UPSTREAM.md) — vLLM PRs we depend on (incl. our [#40361 Marlin pad-sub-tile-n](https://github.com/vllm-project/vllm/pull/40361) which becomes more relevant at higher TP)
- [`models/qwen3.6-27b/INTERNALS.md`](../models/qwen3.6-27b/INTERNALS.md) — head count + KV head structure (basis for the TP divisibility math above)
