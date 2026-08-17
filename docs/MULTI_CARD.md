# Multi-card (3+ GPUs) — derivation, constraints, scaling recipe

You have **3 or more GPUs** and want to know if club-3090 applies. Short
answer: yes. We ship a community-validated 4×3090 fp8/MTP baseline (now the
`vllm/qwen-27b-multi-fast` slug) plus a matching FP8 + fp8/e4m3 "max accuracy"
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
2. **Per-stream TPS doesn't scale *on our recipes*** — but read the scoping
   below before treating that as a law of PCIe.

> [!IMPORTANT]
> **⚠️ Scoped 2026-07-26.** This bullet used to read *"Per-stream TPS doesn't
> scale without NVLink. PCIe NCCL all-reduce overhead grows with TP count;
> per-stream decode at TP=4 may be lower than TP=2."* That states a **topology
> law**, and it isn't one — it's a property of the **recipe** we measure on.
>
> Every shipped config in the table above runs **MTP K=3 speculative decoding**.
> [@alesha-pro](https://github.com/noonghunna/club-3090/discussions/773) ran the
> same TP sweep on 4× 3090 with **no spec decoding at all** and got the opposite
> result. Decode tok/s, Qwen3.6-27B fp8, single stream:
>
> | | TP=2 | TP=4 | scaling |
> |---|--:|--:|--:|
> | **no spec** (@alesha-pro, 220 W, patched P2P) | 38.07 | 61.60 | **+62%** |
> | **MTP n=3** (our shipped configs) | 70.7 <sup>a</sup> | 74.76 <sup>b</sup> | **+5.7%** |
>
> <sup>a</sup> `vllm/qwen-27b-dual-fast`, v0.24.0 · <sup>b</sup> `vllm/qwen-27b-multi-fast`, @ryanmpelletier 4× 3090 all-x16, v0.24.0
>
> **Both are correct.** vLLM TP scaling is real; our TP=2 baseline just already
> has ~1.85× of drafter folded into it, so there's nothing left for cards 3 and 4
> to recover. The decisive number: **his TP=4 (61.60) lands 13% *below* our TP=2
> (70.7).**
>
> So the *practical* advice survives — **on a PCIe rig you buy single-stream speed
> with a drafter, not with more cards** — but the stated *mechanism* (all-reduce
> overhead growing with TP) is not what either dataset shows. Ours shows TP=4 ≈
> TP=2, not TP=4 < TP=2.
>
> **Confirmed by measurement 2026-07-27 — the matched-recipe pair.** The clean
> test this section was waiting for: same rig, same sitting, same 220 W on every
> card, same weights and image (v0.25.1), **MTP n=3 on *both* arms**,
> `switch.sh` + `bench.sh`, ~20 minutes apart
> ([@alesha-pro again](https://github.com/noonghunna/club-3090/discussions/773)):
>
> | | TP=2 `dual-fast` | TP=4 `multi-fast` | Δ |
> |---|--:|--:|--:|
> | narrative decode TPS | 79.26 (CV 4.1%) | 82.44 (CV 3.2%) | **+4.0%** |
> | code decode TPS | 106.50 (CV 6.2%) | 103.73 (CV 1.7%) | **−2.6%** |
> | prefill @10K tok/s | 1556 (CV 0.1%) | 2407 (CV 0.1%) | **+54.6%** |
> | prefill @90K tok/s | 1202 (CV 0.6%) | 1948 (CV 0.5%) | **+62.0%** |
> | TTFT @90K | 73.1 s | 45.6 s | **−37.6%** |
>
> Decode is flat — reproducing the cross-rig +5.7%/−2.9% above within ~2 points,
> from one rig in one sitting. Recipe-not-topology is now *measured*, not
> inferred. **And the number that changes the guidance: cards 3 and 4 buy
> prefill and TTFT, not decode.** A 90K-token prompt starts answering 27 seconds
> sooner on four cards; the tokens then stream no faster. (Same shape as the
> NVLink A/B in [#698](https://github.com/noonghunna/club-3090/issues/698) —
> decode +3–5%, prefill +35–49% — with a larger prefill term.) Two caveats
> carried from the report: the TP=2 arm runs vLLM's custom all-reduce and the
> TP=4 arm cannot (engine-gated at >2 PCIe GPUs —
> [#786](https://github.com/noonghunna/club-3090/issues/786); every
> dual-vs-multi4 comparison shares this asymmetry, recorded as Rig-cell field 4
> in BENCHMARKS), and the cross-*version* question is **resolved (2026-07-27)**:
> our first-party v0.25.1 re-run reproduces the v0.24.0 numbers (decode
> 68.5/92.1 vs 70.7/93.5, at 230 W vs stock caps — flat within power+CV
> noise), so the +11–16% by which the #773 rig's dual arm beats ours is
> **rig-side, not the engine** — their TP=2 runs custom all-reduce +
> transfer-verified P2P; ours runs neither. That difference matching the ~15%
> they measured for custom-AR alone is the strongest hint yet of what the
> interconnect stack is worth at TP=2.
>
> One engine caveat from the same report: this is a **vLLM** property. On the
> llama.cpp leg (tensor-split, not TP) the same box gained only 3–13% going 2→4,
> with a repeat spread wide enough to swamp it (Q8_0 gave 52.9 and 71.8 on
> identical settings). If you're on a GGUF engine, "more cards ≠ faster per
> stream" holds much more strongly.

   Aggregate concurrent throughput still scales — see **Replication vs
   tensor-parallel** below, which is usually the bigger lever.

---

## Replication vs tensor-parallel — the fork this page used to ignore

Everything else on this page reasons in the **TP dimension**: how do I split one
model across N cards? For **aggregate** throughput that is frequently the wrong
question. If the model fits on one card, running **N independent instances**
beats splitting it N ways — by a lot.

Measured by [@alesha-pro](https://github.com/noonghunna/club-3090/discussions/773)
on 4× 3090 PCIe, all arms matched at 220 W, peak aggregate tok/s
(`--max-num-seqs 256 --enable-chunked-prefill`):

| Model | TP=4, 1 instance | replicated | gain |
|---|--:|--:|--:|
| gemma-4-12B AWQ | 1,007 | **3,425** (TP=1 × 4) | **3.40×** |
| Qwen3.6-27B INT8-W8A8 | 358 | **741** (TP=2 × 2) | 2.07× |
| Qwen3.6-27B INT4 | 307 | **737** (TP=2 × 2) | 2.40× |

**The rule of thumb: latency wants max TP, throughput wants max instances.**
(@alesha-pro's phrasing.) Same cards, same weights, opposite answer depending on
which you're optimizing.

The trap is that **the same run flips winner** depending on which number you
read. AutoRound INT4 on that rig, one sitting, one power cap:

| | TP=2 | TP=4 |
|---|--:|--:|
| single stream (wall) | 53.80 | **69.26** |
| aggregate peak @ c=64 | **473.1** | 294.8 |

TP=4 is **29% faster for one user and 38% slower for a full queue.** So "which
TP should I run" has no answer without "for what workload".

**This agrees with what we measured from the concurrency side.** Our own sweep
(2026-07-10, 2× 3090, agent-shaped 16K prompts) found the dense 27B's batching
knee at **N=2** — decode-aggregate *halves* by N=8, because per-stream falls
faster than N grows. Meanwhile the 35B-A3B MoE holds ~250–270 flat to N=16.
Same lesson from the other direction: past a low N, stop feeding one big engine
and start adding engines. See the concurrency sweep in
[BENCHMARKS.md](../BENCHMARKS.md) and the FAQ entry on serving multiple coding
agents.

**Caveats before you replicate.** Each instance needs its own full weight copy
(so the model must fit on one card *with* a useful KV pool), its own port, and
its own KV pool — you lose the pooled-KV benefit that is the main reason to run
TP=4 for long-context single-prompt work. Replication is the answer for **many
short-to-medium requests**; TP is the answer for **one very long one**. We don't
ship a replication compose today — [PODS.md](PODS.md) is the closest thing.

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
| `single_card` | 1 GPU detected | 1x RTX 3090 | Run `bash scripts/switch.sh qwen3.6-27b/default` (resolves to `vllm/minimal`). See [SINGLE_CARD.md](SINGLE_CARD.md). |
| `homogeneous` | All cards have matched VRAM and matched SM | 2x RTX 3090 | TP=N is the optimal default; use the shipped `vllm/dual*` (2-card) or `vllm/qwen-27b-multi-*` (4-card) composes. |
| `vram_matched_compute_mismatched` | Same VRAM, different compute tier | RTX 3090 + RTX 4090 | TP=N works correctly, but faster cards wait at NCCL allreduce. Estate planner is better for multi-model workloads. |
| `vram_mismatched` | Different VRAM sizes | RTX 3060 12 GB + RTX 3090 24 GB | Prefer llama.cpp `--tensor-split`, manual PP=N experiments, or estate planner. Avoid TP=N across the full mismatched set. |
| `heterogeneous_mixed` | Multiple VRAM and compute tiers | RTX 3060 + RTX 3090 + RTX 4090 | Manual selection. Run one model on the largest matched subset or use estate planner for separate endpoints. |

> **Running several models at once** (the "estate planner / separate endpoints" rows above) is a **pod** workload — one model per GPU subset, managed with `pod.sh` or the c3 Operate tab. See **[PODS.md](PODS.md)**.

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

vLLM asserts that the **attention** head count divides the TP size. KV heads
do not have to divide it — when there are fewer KV heads than ranks, vLLM
**replicates** them. Qwen3.6-27B and Qwen3.8-27B are identical here
(`config.json` → `text_config`):

- **24 attention heads** (factors: 1, 2, 3, 4, 6, 8, 12, 24) — this is the binding constraint
- **4 KV heads** — shard cleanly to TP=4, then replicate above it
- head_dim 256

Valid TP values are therefore **1, 2, 4, 8** (and 12, 24 on datacenter boards):

| GPUs | Valid TP | Attn heads/card | KV heads/card | Notes |
|---|---|--:|---|---|
| 1 | TP=1 | 24 | 4 | Standard single-card. See [SINGLE_CARD.md](SINGLE_CARD.md). |
| 2 | TP=2 | 12 | 2 | Standard dual. See [DUAL_CARD.md](DUAL_CARD.md). |
| **3** | **TP=2 only** | — | — | 24 ÷ 3 = 8 attention heads is fine, but **4 KV heads across 3 ranks neither divides nor replicates evenly** — vLLM errors at boot. **Use TP=2 with 1 idle card** (`CUDA_VISIBLE_DEVICES=0,1`), or run 2 single-card stacks on different ports. |
| **4** | **TP=4** | 6 | 1 | The shipped multi-card tier. Production-viable if your rig has the slots + power + cooling. |
| **5** | **TP=4 + 1 idle** | — | — | ⚠️ **TP=5 does NOT work** — 24 ÷ 5 is not an integer. |
| **6 or 7** | **TP=4 + spare cards** | — | — | Neither divides 24 evenly alongside the KV constraint. Use TP=4 and leave the extras idle. |
| **8** | **TP=8** | 3 | replicated ×2 | Each card holds 3 attention heads; the 4 KV heads replicate, so **per-card KV stays at the TP=4 figure** and only weights shard 8-way. |
| **10** | **TP=8 + 2 idle** | — | — | ⚠️ **TP=10 does NOT work** — 24 ÷ 10 is not an integer. |

**Critical: TP=3, 5, 6, 7, 9, 10 do NOT work.** vLLM errors at boot
("number of attention heads must be divisible by tensor parallel size").
If you have an awkward GPU count, use the next-lower valid TP and leave
the extras idle, or run separate stacks on different ports.

> ⚠️ Because KV heads replicate above TP=4, **going from 4 to 8 cards does not
> shrink per-card KV** — it only shards weights further. Budget accordingly.

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

### `vllm/qwen-27b-multi-max` — FP8 weights + fp8/e4m3 KV (highest fidelity)

```bash
bash scripts/switch.sh vllm/qwen-27b-multi-max
```

`multi4/fp8/mtp.yml` mirrors the 2-card `vllm/qwen-27b-dual-max`: official **FP8**
weights (Marlin W8A16 on Ampere — memory win, no FP8 compute speedup) + **fp8/e4m3**
KV (1 byte/token, FlashInfer backend, scale=1.0 on this checkpoint) + MTP n=3 @ 262K.
**✅ Production** (2026-07-07) — validated on a clean v0.24.0 4-card rig:
[#584](https://github.com/noonghunna/club-3090/issues/584) (@ryanmpelletier, 4× 3090
x16) passed verify-full 9/9, verify-stress to 240K, soak-continuous, and the full 8-pack
at **111/150** (ties the dual-max proxy 109 → quality is TP-invariant), corroborated by a
2nd independent rig [#625](https://github.com/noonghunna/club-3090/issues/625) (@MoppelMat,
mixed x4/x8 lanes). The 4 cards relieve the dual-max proxy's tight **295K-tok / 1.13×** KV
pool → **6.77×** at TP=4; single-stream decode is ~flat vs the 2-card tier.

> **KV decode-at-depth applies here too ([#594](https://github.com/noonghunna/club-3090/pull/594), [#595](https://github.com/noonghunna/club-3090/pull/595)).** `multi-max` now mirrors `dual-max` exactly: FP8 weights + `KV_CACHE_DTYPE=fp8` (e4m3 → FlashInfer). The prior int8-PTH KV was `TRITON_ATTN`-only and cratered at depth on the dual proxy (130.8→50.7 TPS @ 35K); fp8/e4m3 stayed flat (~115 @ 35K), recalled NIAH to 240K, tied the 8-pack (109 vs 107), and passed soak. `multi-max` inherited that one-line KV flip in #595, and it was confirmed at TP=4 on [#584](https://github.com/noonghunna/club-3090/issues/584) (verify-full 9/9, stress to 240K, soak, 8-pack 111/150) + corroborated on [#625](https://github.com/noonghunna/club-3090/issues/625) — hence its promotion to ✅ Production.

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
`vllm/qwen-27b-multi-max` (FP8 + fp8/e4m3) — mirroring the 2-card fast/max tiers.
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
