# Multi-card (3+ GPUs) — derivation, constraints, scaling recipe

You have **3 or more GPUs** and want to know if club-3090 applies. Short
answer: yes. We ship a community-validated 4×3090 fp8/MTP baseline (now the
`vllm/qwen-27b-multi-fast` slug) plus a matching FP8 + int8-PTH "max accuracy"
variant (`vllm/qwen-27b-multi-max`), and keep other 3+ GPU configs as derivation
recipes until someone measures them. This page explains what scales (and what
doesn't) when going beyond TP=2, the constraints to know, and how to derive your
own compose when the shipped `multi4` composes aren't your topology.

> **Model not in the configs here / want any HF safetensors repo?** → [`docs/PULL.md`](PULL.md): `scripts/pull.sh` evaluates any model against the KV math (honest, no download) and boots it if it passes. The recipes on this page are the measured/derivation path; both work.

> **Validation note:** the maintainer rig is **2× RTX 3090 PCIe**, but
> Whamp's 4× RTX 3090 PCIe rig validated the TP=4 fp8/MTP baseline (the config
> now shipped as `vllm/qwen-27b-multi-fast`) in
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
| Per-card peak VRAM (262K target) | 23.5+ GB tight | 23.6 GB tight | **23.5 GB fp8** (cross-rig) | ~10-12 GB |
| Cliff 2 single-prompt | fires at ~60K | doesn't fire (verified at 237K) | **passes 91K needle** | shouldn't fire |
| Per-stream TPS (PCIe-only) | baseline | ~same as TP=1 | **63/76 fp8** (cross-rig = `multi-fast`) | lower still |
| Concurrent throughput (multi-stream) | 1× | ~1.7-3.6× | KV pre-check **6.77× fp8 @ 262K** (cross-rig) | derived ~3-12× |
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

## Topology classification

The launcher classifies your selected hardware and emits strategy guidance
when the cards are not matched. You can run the classifier without booting
anything:

```bash
bash scripts/launch.sh --topology
```

Use `--gpus 0,1` or `--cards 2` with `--topology` if you only want advice
for a subset.

| Class | What it means | Example | Recommended |
|---|---|---|---|
| `single_card` | 1 GPU detected | 1x RTX 3090 | Use the largest single-card compose that fits (`vllm/default`, `vllm/long-text`, `llamacpp/default`). |
| `homogeneous` | All cards have matched VRAM and matched SM | 2x RTX 3090 | TP=N is the optimal default; use the shipped `vllm/dual*` (2-card) or `vllm/qwen-27b-multi-*` (4-card) composes. |
| `vram_matched_compute_mismatched` | Same VRAM, different compute tier | RTX 3090 + RTX 4090 | TP=N works correctly, but faster cards wait at NCCL allreduce. Estate planner is better for multi-model workloads. |
| `vram_mismatched` | Different VRAM sizes | RTX 3060 12 GB + RTX 3090 24 GB | Prefer llama.cpp `--tensor-split`, manual PP=N experiments, or estate planner. Avoid TP=N across the full mismatched set. |
| `heterogeneous_mixed` | Multiple VRAM and compute tiers | RTX 3060 + RTX 3090 + RTX 4090 | Manual selection. Run one model on the largest matched subset or use estate planner for separate endpoints. |

### Why TP=N is poor on VRAM-mismatched cards

Tensor parallelism splits weights evenly across cards. If one card has 24 GB
and another has 12 GB, TP=2 still puts roughly half the model on each card.
The smaller card becomes the hard ceiling for weights, KV cache, activations,
and fragmentation. For Qwen 3.6 27B INT4, that usually leaves too little KV
headroom to be useful.

For mismatched VRAM, the practical paths are:

- llama.cpp `--tensor-split` for weighted layer placement.
- PP=N as a manual vLLM flag flip (`--pipeline-parallel-size N`) when you are
  deliberately experimenting. club-3090 does not ship a PP compose today.
- Estate planner: `bash scripts/launch.sh --estate` runs different models on
  different card subsets without forcing one model across uneven VRAM.

### When compute-mismatched TP is fine

Matched VRAM with different SM, such as RTX 3090 + RTX 4090, is a different
trade-off. TP=2 works because both cards have enough memory for the same model
shard and KV budget. The cost is throughput: the faster card waits at NCCL
allreduce barriers, so effective pair speed caps near the slower card. You
preserve per-card VRAM capacity, but waste some compute on the faster card.

That is acceptable for one-model serving. If your goal is maximum aggregate
throughput from two different cards, estate planner usually wins because each
card runs its own model at full speed.

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

### Picking which cards to use on awkward counts

On a rig with 3 cards (or more, where you only want to use 2 for vLLM),
**which two you select matters** for both throughput and reliability.

```bash
# Inspect topology first — connectivity classes affect TP allreduce
nvidia-smi topo -m
```

The matrix shows pairwise links between GPUs. Best to worst for TP allreduce:

| Class | Meaning | Implication |
|---|---|---|
| `NV#` | NVLink-bonded | Fastest. We don't have it on consumer 3090s by default. |
| `PIX` | Same PCIe switch (one bridge hop) | Optimal on PCIe-only stacks. |
| `PXB` | Multiple PCIe bridges, no host bridge | Acceptable; slightly higher latency. |
| `PHB` | Crosses PCIe Host Bridge (the CPU) | Common on consumer boards; works but ~10-15% allreduce overhead vs PIX. |
| `SYS` | Crosses NUMA / SMP interconnect | Avoid for TP if there's a same-NUMA pair available. |

**Pick a same-switch pair (`PIX`) if your rig has one** — typically the two
slots wired into the same PCIe expander on workstation boards. On consumer
ATX, all GPUs usually traverse the host bridge (`PHB`) so it doesn't matter
much; on Threadripper / EPYC / dual-CPU server boards, NUMA topology can
make a measurable difference.

To run TP=2 on cards 1+2 (e.g., card 0 is reserved for ComfyUI / display):

```yaml
# In your override compose file
services:
  vllm-qwen36-27b-dual:
    environment:
      - CUDA_VISIBLE_DEVICES=1,2
```

Cross-rig data: [@lexhoefsloot](https://github.com/noonghunna/club-3090/issues/49)
runs TP=2 on host GPUs 1+2 of a 3× 3090 rig (same PCIe switch, `PHB` to
GPU 0) — that's the right pattern for a rig where GPU 0 is doing other
work or differs in topology.

---

## Shipped TP=4 baselines — `vllm/qwen-27b-multi-fast` and `vllm/qwen-27b-multi-max`

For 4× RTX 3090 PCIe, two TP=4 composes mirror the 2-card fast/max tiers (see
[DUAL_CARD.md](DUAL_CARD.md)). Each is **byte-identical to its 2-card sibling**
apart from `--tensor-parallel-size` and the gpu-count, so the duals are the
on-rig validation proxies (the maintainer rig is 2× 3090) — both ship
🧪 Experimental until re-confirmed on your 4-card host.

### `vllm/qwen-27b-multi-fast` — AutoRound INT4 + fp8 KV + MTP (the proven path)

```bash
bash scripts/switch.sh vllm/qwen-27b-multi-fast
```

`multi4/autoround-int4/mtp.yml` keeps the `dual.yml` (≡ `vllm/qwen-27b-dual-fast`)
fp8/MTP feature set and changes TP/streams from 2 → 4. This is the exact config
Whamp measured on a 4× 3090 PCIe rig — **cross-rig numbers** (the compose was
since renamed `fp8-mtp.yml` → `mtp.yml`; config unchanged):

- boots at `max_model_len=262144`
- vLLM reports GPU KV cache size **483,200 tokens** and **6.77×** maximum concurrency for 262K-token requests
- `verify-full.sh` passes
- `verify-stress.sh` passes 7/7; probe 7 recalls **58,569-token** and **91,070-token** needles
- `bench.sh`: **63.01 narr / 76.25 code wall TPS**, peak **23,494 MiB/card**

(Validation thread: [discussion #26](https://github.com/noonghunna/club-3090/discussions/26), 2026-05-03.)

### `vllm/qwen-27b-multi-max` — FP8 weights + int8-PTH KV (highest fidelity)

```bash
bash scripts/switch.sh vllm/qwen-27b-multi-max
```

`multi4/fp8/mtp.yml` mirrors the 2-card `vllm/qwen-27b-dual-max`: official **FP8**
weights (Marlin W8A16 on Ampere — memory win, no FP8 compute speedup) + **int8-PTH**
KV (8-bit, higher fidelity than fast's fp8_e5m2) + MTP n=3 @ 262K. Validated via the
dual-max proxy at TP=2 (KV pool **295K tok / 1.13×** @262K, MTP active) — the tight
2-card KV pool is exactly what TP=4 relieves. **No 4-card bench on this layout yet**,
and the quality A/B vs the fast tier is pending.

> **KV decode-at-depth applies here too ([#594](https://github.com/noonghunna/club-3090/pull/594)).** Because multi-max mirrors dual-max exactly (fp8 weights + int8-PTH KV), it inherits the same behaviour: int8-PTH KV is `TRITON_ATTN`-only and single-stream decode craters with context (measured on the dual proxy: 130.8→50.7 TPS @ 35K), while `KV_CACHE_DTYPE=fp8` (e4m3 → FlashInfer) keeps it flat (~115 @ 35K) at equal NIAH recall to 240K. **dual-max flipped to fp8 in [#594](https://github.com/noonghunna/club-3090/pull/594)** (all gates green: 8-pack 109 ties 107, decode 2.3× at depth, soak-continuous PASS / 0 growth / margin holds). **multi-max flips in a follow-up PR** — identical one-line env swap, bundled with the registry `kv_format` + rtx-3090 fp8_e4m3-compat sync.

> **DFlash on TP=4 was removed.** The former `vllm/dual4-dflash`
> (`multi4/autoround-int4/dflash.yml`) is gone — DFlash on Qwen3-Next vLLM is blocked
> by DeltaNet KV rollback ([vllm#39931](UPSTREAM.md), the same block as the dual-card
> path). DFlash-for-Qwen now lives on beellama (`beellama/qwen-dflash-dual`, v0.3.0 🧪);
> see [DUAL_CARD.md](DUAL_CARD.md).

## Recipe — derive your own config from `dual.yml`

The 2-card `dual/autoround-int4/fp8-mtp.yml` (`vllm/dual`) is the tested baseline
and `multi4/autoround-int4/mtp.yml` (`vllm/qwen-27b-multi-fast`) is the measured
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
with your other variants. `multi-fast` uses `vllm-qwen36-27b-multi4` on port
`8014` and `multi-max` uses `vllm-qwen36-27b-multi4-max` on port `8015`;
reserve a different name/port for further experiments:

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
- **Cliff 2:** canonical `verify-stress.sh` probe 7 passes at both large
  rungs: ~58.6K tokens and 91K tokens recalled correctly.
- **Trade-off:** PCIe allreduce makes single-stream decode slower than
  TP=2, but TP=4 provides more full-context concurrency and the first
  published 4×3090 Cliff 2 boundary data.

> The same 4×3090 run also measured a **now-removed** DFlash TP=4 variant
> (64.00 / 104.40 narr/code TPS, AL ~4.4, KV pool 207,264 tok / 2.27× @262K).
> That compose was dropped — DFlash on Qwen3-Next vLLM is blocked by DeltaNet KV
> rollback ([vllm#39931](UPSTREAM.md)) — so the numbers are kept only as historical
> cross-rig context; DFlash-for-Qwen now lives on beellama.

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

## Why we ship only a fast/max pair of pre-baked 4-card configs

We ship two TP=4 composes — `vllm/qwen-27b-multi-fast` (fp8/MTP) and
`vllm/qwen-27b-multi-max` (FP8 + int8-PTH) — mirroring the 2-card fast/max tiers.
The `multi-fast` config is here because a community rig validated that exact
4× RTX 3090 PCIe topology with `verify-full.sh`, `verify-stress.sh`, and
`bench.sh`; `multi-max` is its higher-fidelity sibling (validated via the
`dual-max` proxy at TP=2, 🧪 pending a 4-card bench). We still avoid a broad
matrix of untested 4+ GPU composes:

1. **Hardware combinations explode.** 4× 3090 vs 4× A5000 vs 4× A6000 vs
   2×3090 + 2×4090 vs 4× modded 3080 — each has different VRAM, topology,
   power profile, and allreduce characteristics.
2. **Variant count needs discipline.** A measured fp8/MTP TP=4 baseline
   (plus its FP8 fidelity sibling) is useful; a directory full of
   derived-but-unvalidated variants would create false confidence.
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
