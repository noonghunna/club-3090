# 1.22M-token KV pool on dual 3090 — TurboQuant 3-bit + MTP, the two paths we explored

**TL;DR.** On a dual RTX 3090 PCIe rig (no NVLink), TurboQuant 3-bit KV with built-in MTP (n=3) gives you a **1.22M-token KV pool at 262K max-model-len — 4.66× concurrency, 2× the INT8 PTH baseline pool**, at within ~5pp of the INT8 PTH quality on a 150-scenario quality suite and within noise on aider-polyglot-30 (18/30 vs 19/30 baseline). The catch: the working path requires Sander's Genesis modular patches today. The patch-only path (vendoring upstream PRs) is broken across every phase we measured.

This page is the user-facing "what we found, why it's interesting for dual-3090 local-maxxing, and what we learned along the way" writeup.

---

## The dream: more KV pool, same throughput, same model

What you actually run out of on a 2× 24 GB rig serving Qwen3.6-27B isn't compute — it's KV cache. Each token at 262K context is held twice (key + value), with one layer per hybrid block. The default fp8 KV gives you ~9× concurrency, but precision is brittle on long-context recall. INT8 PTH (per-token-head) is the production-safe baseline: 605K-token pool at 262K = 2.31× concurrency.

TurboQuant 3-bit (TQ3) is the dream-tier: ~3 bits/token instead of 8, **+134% pool capacity** (1.42M tokens without MTP, 1.22M with MTP) — **enough for 4-5 concurrent streams at full 262K context**, or for one stream and a long-running multi-agent workload with ~12× headroom at 100K-ctx turns.

**That's the actual prize for dual-3090 enthusiasts.** You don't want it because TPS goes up. You want it because *what you can do with the rig* changes: 4 concurrent users at 262K context, or 1 user with 12 simultaneous agent threads at 100K each. The pool is the bottleneck on local-AI workloads that any real product runs into, and TQ3 is the first thing on this stack that meaningfully changes it.

---

## What we found

### 1. KV pool capacity — TQ3 doubles the baseline

![KV pool by config](assets/tq3-mtp-genesis/01-kv-pool-by-config.png)

The two TQ3 bars dwarf the INT8 PTH and bf16 baselines. **The Genesis-backed TQ3+MTP row gives you 1.22M KV tokens at 262K ctx**, which is ~2× the Qwen INT8 PTH baseline and ~5× the bf16 (200K) variant. The patch-only TQ3 bar is slightly larger (1.42M, since it has no MTP overhead) but is non-functional — see chart 3.

### 2. The trade frontier — quality vs concurrency

![Quality vs concurrency](assets/tq3-mtp-genesis/02-quality-vs-concurrency.png)

Read the chart as: **how much quality you give up per extra concurrent stream**. The shaded region (top-right) is where Genesis TQ3+MTP sits — high concurrency, near-baseline quality. The cluster on the left is the bf16/INT8 PTH baselines (high quality, low concurrency). The lone bottom-right dot is the patch-only TQ3 attempt — same concurrency, quality collapsed.

The Genesis dot lands roughly on the same quality contour you'd extrapolate from the baselines, just twice as far right.

### 3. Per-phase verdict — broken vs working, in one table

![Verdict matrix](assets/tq3-mtp-genesis/03-verdict-matrix.png)

The two TQ3 rows tell the entire story. Patch-only: green TPS row (it generates *something* fast) but every quality column collapses — 18/150 quality, 0/30 aider, 34/100 silent-empty in soak. Genesis: every column lands at baseline-comparable (✓), at the same TQ3 KV pool capacity.

**That's the difference one patch (Genesis P67 — proper multi-query Triton kernel for spec-decode K+1 verify against compressed cache) makes.**

---

## The journey — what we tried, in order

This took about 2 weeks of probing. Highlights for the local-maxxing crowd:

### Path 1 — vendoring upstream PRs

The intuition was: TQ3+MTP is broken on stock vLLM, but there are 5+ open PRs touching the surface. Surely one of them is the fix?

We rebased and vendored every PR we could find that touched TQ + MTP + spec-decode K+1:

| PR | What it does | Result on our stack |
|---|---|---|
| [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) (our PR) | Marlin pad-sub-tile-n — unblocks TP=2 on AutoRound INT4 | Clean. Required. |
| [vllm#40798](https://github.com/vllm-project/vllm/pull/40798) | TQ decode workspace pre-allocation | Partially required; mounted the `gpu_model_runner.py` slice only |
| [vllm#40914](https://github.com/vllm-project/vllm/pull/40914) | K+1 spec-verify routing via synthetic `seq_lens` | **Negative.** Vendored on post-#41434 main; MTP acceptance hit 100% but outputs collapsed into `!`-floods. Dropping it improved verify-stress from 3/7 to 5/7. |
| [vllm#40792](https://github.com/vllm-project/vllm/pull/40792) | k8v4 GQA grouping kernel | Adjacent but doesn't implement K+1 multi-query verify |
| [vllm#42215](https://github.com/vllm-project/vllm/pull/42215) | TQ decode kernel warmup | Orthogonal to the multi-query bug |

We A/B'd across 4 TurboQuant precision tiers (3-bit, 4-bit, k8v4 = 8/4-bit) — the failure was **format-independent**. Every tier failed long-context needle recall with first-word repetition under MTP. The same TQ3 with MTP *disabled* (`tq3-nomtp.yml`) passes 7/7 verify-stress cleanly, so the bug is specifically the **MTP × TQ × multi-query interaction**, not TQ precision.

Then the [vllm.ai/blog/turboquant](https://vllm.ai/blog/turboquant) post made the upstream position explicit: **"TurboQuant supports only models with standard attention mechanisms (e.g. GQA) — models with sliding-window or hybrid attention are not yet supported."** Qwen3.6-27B is Qwen3-Next hybrid (DeltaNet + full-attention interleaved). So 4 of those 5 PRs wouldn't fix this anyway — there's no upstream multi-query TQ verify kernel that handles hybrid attention. Closed the patch-only investigation; tombstoned `dual/autoround-int4/tq3-mtp.yml`.

### Path 2 — Genesis P67

Sander's [Genesis](https://github.com/Sandermage/genesis-vllm-patches) v7.72.2 ships **P67: a proper multi-query Triton kernel for spec-decode K+1 verify against compressed TurboQuant cache**. That's the kernel the upstream PR landscape doesn't have. We'd been parking the Genesis-backed compose pending Sander's v7.73.x release; instead we ran the matched-config rebench on v7.72.2 with a pin downgrade and let the numbers speak.

One blocker hit along the way: Genesis v7.72.2's `KNOWN_GOOD_VLLM_PINS` allowlist doesn't include the canonical club-3090 nightly (`1acd67a79`, post-#41434 main). On that pin, `maybe_override_with_speculators` trips the transformers 5.8.0 `cached_file` regression and aborts at boot:

```
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/root/.cache/huggingface/qwen3.6-27b-autoround-int4'
```

Pin-downgraded `dual/autoround-int4/tq3-mtp-genesis.yml` to `nightly-01d4d1ad3` (= `0.20.2rc1.dev9`, allowlist-included), and the boot lit up green. Genesis P67 enabled on Ampere consumer (`[ON]` in the platform regime), P66 filtered cudagraph capture sizes for spec-decode `uniform_query_len=4` (kept [4, 8, 16]; removed [1, 2]).

---

## The result — apples-to-apples on dual 3090

Same nightly pin family, same TP=2, same `max-num-seqs=2`, same 262K context, same Qwen3.6-27B AutoRound INT4 weights. Just two different code paths for TQ3+MTP:

| Phase | Patch-only (broken) | Genesis P67 (working) | INT8 PTH baseline (ref) |
|---|---|---|---|
| Bench narr TPS | 98.7 (CV 36%) ⚠️ | **89.2** (CV 4%) | 85.0 |
| Bench code TPS | 104.5 (CV 27%) ⚠️ | **119.1** (CV 1%) | 121.1 |
| Verify-stress | 5/7 ✗ (10K/30K/60K/90K needles fail) | **7/7 ✓** (incl. 60K needle PASS) | 7/7 ✓ |
| Quality (8 packs, 150 scenarios) | 18/150 (12%) ⚠️ | **86/150 (57%)** | 94/150 (63%) |
| Soak silent-empty | 34/100 ⚠️ | **0/100 ✓** | 0/100 ✓ |
| Aider-polyglot-30 | 0/30 ⚠️ (full 2700s timeout) | **18/30** | 19/30 |
| KV pool @ 262K | 1.42M (broken) | **1.22M** | 605K |
| Concurrency | 5.41× (broken) | **4.66×** | 2.31× |
| Spec-decode AL | bimodal (100% / corrupt) | 3.50 sustained, [0.95, 0.84, 0.75] | n/a |

The Genesis row delivers near-baseline quality at **roughly 2× the concurrency**, with healthy spec-decode behaviour (clean diminishing per-position acceptance, no bimodal collapse).

---

## When to pick this — and when not to

Pick **`dual/autoround-int4/tq3-mtp-genesis.yml`** when **all three** of:

1. You're running on dual 3090 (TP=2) and want max concurrent streams or max effective long-context throughput.
2. You're OK with the Genesis modular patch stack as a dependency (Sander's [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches), v7.72.2 pinned).
3. The ~5pp quality drop on the 150-scenario suite (vs INT8 PTH) is acceptable for your workload — it's basically invisible on coding (-1 aider task) and visible on math reasoning.

Pick **`dual/autoround-int4/tq3-nomtp.yml`** when you want the **biggest possible KV pool** (1.73M / 6.59× concurrency) and don't need MTP throughput — this path is Genesis-free, vanilla upstream + marlin-pad only, and passes 7/7 verify-stress.

Pick **`dual/autoround-int4/int8.yml`** (Qwen INT8 PTH) when you want the **simplest, baseline-quality setup** with MTP and no Genesis dependency — 605K pool, 2.31× concurrency, 94/150 quality, 19/30 aider. Production-safe.

Pick **`dual/autoround-int4/turbo.yml`** (4-stream `max-num-seqs=4` production variant) for **multi-tenant serving** — same Genesis P67 stack, tuned for 4 concurrent streams at 262K. (This page benches the 2-seq matched-config sibling for the head-to-head.)

---

## What's the path forward without Genesis?

Genesis is great but a non-trivial dependency. The Genesis-free path is gated on **an upstream P67-equivalent landing**: a proper multi-query Triton kernel for spec-decode K+1 verify against compressed TurboQuant cache that handles hybrid attention. None of the open PRs do this today (we checked — see `docs/UPSTREAM.md` "TQ + MTP" section and the `dual/autoround-int4/tq3-mtp.yml` tombstone header).

Two upstream signals to watch:

1. **vllm.ai's TurboQuant blog** explicitly notes hybrid-attention support is "not yet supported" — an open scoping item, not a "won't fix".
2. **Sander has signalled he'll upstream P67** once Genesis matures past the v7.73.x rework currently in flight.

When either of those lands, the upstream-only path opens. Until then, `dual/autoround-int4/tq3-mtp-genesis.yml` is the working TQ+MTP path on dual 3090.

---

## Reproduce on your rig

```bash
# 1. Clone & set up
git clone https://github.com/noonghunna/club-3090
cd club-3090
bash scripts/setup.sh

# 2. Bring up the Genesis-backed TQ3+MTP compose (pin-downgraded to Genesis v7.72.2 known-good)
MODEL_DIR=/path/to/huggingface docker compose \
  -f models/qwen3.6-27b/vllm/compose/dual/autoround-int4/tq3-mtp-genesis.yml up -d

# 3. Run the full rebench incl. the 8-pack in both reasoning modes (~1.75-2 hr;
#    omit --with-8pack-thinking for the fast structural-gates-only run, #338)
URL=http://localhost:8015 TAG=my-tq3-mtp-genesis bash scripts/rebench-full.sh --with-8pack-thinking=both
```

Full results land in `results/rebench/<tag>/REPORT.md`. Cross-reference with our `results/rebench/qwen-tq3-mtp-genesis-2026-05-11/REPORT.md`.

---

## Acknowledgements

- **[Sander](https://github.com/Sandermage)** for [Genesis](https://github.com/Sandermage/genesis-vllm-patches) and specifically P67 — the multi-query verify kernel is the actual fix. The local rig wouldn't have a working TQ+MTP path without it.
- **[Tom (Lorbus)](https://huggingface.co/Lorbus)** for the Qwen3.6-27B AutoRound INT4 + BF16-preserved-MTP weights that this whole writeup is benched on.
- **[Lucy / lucebox-hub](https://github.com/lucebox-hub)** for DFlash, parallel to this — different attention path, different trade.
- **[Carnice / @intervitens](https://huggingface.co/intervitens)** for the Qwen3.6 calibration corpora work.

If you're on a 1× 3090 or hybrid setups, see [DUAL_CARD.md](./DUAL_CARD.md), [CLIFFS.md](./CLIFFS.md), and the per-model `learnings/` files for additional context.
