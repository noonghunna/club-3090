# Hardware notes

What this stack assumes about your hardware. True regardless of which model or engine you're running.

---

## Required

- **NVIDIA RTX 3090 (24 GB, Ampere SM 8.6)** — 1 or 2 cards.
- **PCIe Gen 4 slot** — Gen 3 works but allreduce on dual-card is slower (mild impact on multi-tenant; minimal impact on single-stream).
- **NVIDIA driver 580.x or newer** — for CUDA 13 runtime in vLLM nightly. `nvidia-smi` to check. Older drivers won't load CUDA 13 kernels.
- **Linux** (Ubuntu 22.04+ tested). vLLM is Linux + CUDA only. llama.cpp works on macOS / Windows but our recipes assume Linux paths.
- **Docker + NVIDIA Container Toolkit** for vLLM. llama.cpp doesn't need Docker.

---

## Other Ampere/Ada cards

The recipes are written against 3090 specifically but should work on:

| Card | VRAM | Compute capability | Notes |
|---|---|---|---|
| RTX 3090 | 24 GB | sm_86 | **Tested. Default target.** |
| RTX 3090 Ti | 24 GB | sm_86 | Should work; same VRAM, slightly higher TPS expected |
| **2× RTX 3080 modded 20 GB** | 20 GB / card (40 GB combined) | sm_86 | **Tested 2026-05-02 by [@troymroberts](https://github.com/troymroberts) ([#25](https://github.com/noonghunna/club-3090/discussions/25#discussioncomment-16787782))** at 200W/card power limit. `dual.yml` (TQ k8v4 KV + MTP K=3) boots at full 262K target with `gpu-memory-utilization=0.82` (down from shipped 0.95 — see note below). Available KV pool 5.2 GB/card, max concurrency 1.43×. verify-full 10/10 pass; bench 49 TPS wall single-stream, 210 TPS aggregate at n=8. First published SM86 / 40 GB combined data point outside the 3090 family. |
| RTX 4090 | 24 GB | sm_89 | Should work; ~30% faster decode (newer SMs); same memory characteristics |
| RTX 5090 | 32 GB | sm_120 | Untested; more VRAM relaxes the prefill cliffs but kernel paths might differ |
| RTX A5000 | 24 GB | sm_86 | **Sander's PROD class** for [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches). Identical SM and VRAM to 3090; should run identically. |
| RTX A6000 | 48 GB | sm_86 | Should work; double VRAM lets you skip the cliff workarounds (use Sandermage's reference defaults) |
| H100 SXM | 80 GB | sm_90 | Different beast; flash-attn 3 paths available; not what these recipes target |

**Won't work:** anything with <20 GB VRAM (3060, 3070, stock 3080, 3080 Ti). The 27B model in INT4 is ~18 GB — KV pool + activations push past 24 GB on smaller cards even with aggressive quantization. **Modded 20 GB 3080s do work** (see row above) — the mod gives them enough headroom for the 27B + TQ K8V4 KV path on TP=2, with `mem-util=0.82` to absorb cudagraph profiling overhead.

### Note for sub-24 GB cards

On 20 GB cards (modded 3080) the cudagraph-profiling overhead is a meaningful slice of available VRAM. Drop `--gpu-memory-utilization` to **0.82** (vs shipped 0.95 for 24 GB). vLLM nightly's `gpu_worker.py` reports the equivalent effective KV size in the boot log; tune to keep activation headroom for the ~15K tool-prefill peak (verify-full check 8). Credit: [@troymroberts](https://github.com/troymroberts).

**4090s with attached display — env-override the compose defaults.** Some 4090 rigs land at ~23.5 GB usable VRAM with X server + driver overhead, vs the headless 3090s the composes are calibrated for. Boot may fail with `No available memory for the cache blocks` at default `max-model-len`. Cross-rig data: @laurimyllari's 4090 single-card on `long-text.yml` needed `MAX_MODEL_LEN=90000` (down from 180K default) to fit cleanly ([disc #62](../../../noonghunna/club-3090/discussions/62) / [issue #71](../../../noonghunna/club-3090/issues/71)). Pattern:

```bash
MAX_MODEL_LEN=90000 bash scripts/switch.sh vllm/long-text
```

Same `MAX_MODEL_LEN` / `GPU_MEMORY_UTILIZATION` env overrides apply for any setup running vLLM alongside other GPU consumers on the same card. See [SINGLE_CARD.md "Running alongside a desktop"](SINGLE_CARD.md#running-alongside-a-desktop--sub-24-gb-usable-vram) for safe ranges.

**`dual-turbo.yml` on 20 GB Ampere — swap TQ3 KV → fp8_e5m2.** The shipped `dual-turbo.yml` uses `--kv-cache-dtype turboquant_3bit_nc` (the technique from [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874), ICLR 2026 — random rotation + scalar quantizers + 1-bit QJL transform on the residual; the paper claims absolute quality neutrality at 3.5 bits/channel). It's the right pick on 24 GB / 3090: smaller KV pool → more concurrency, and the 24 GB budget absorbs the dequant activation cost during the DeltaNet GDN forward. On 20 GB cards the trade flips: TQ3's activation peak (~1 GB/card more pressure than fp8 during the materialized block — see [PerfMamba arxiv 2511.22849](https://arxiv.org/html/2511.22849) for the underlying Mamba-2 block-state-materialization mechanism the GDN forward inherits) exceeds the per-card budget after TP=2 split, and Cliff 2 fires at 90K. **Override to `--kv-cache-dtype fp8_e5m2`** and you get the full 262K context working with verify-stress 7/7 PASS including 91K needles. Validated 2026-05-04 by [@efschu](https://github.com/noonghunna/club-3090/issues/47) on 2× 3080 modded 20 GB at 0.82 mem-util: bench 82.4 narr / 107.9 code TPS, full 257K-token auto-discovery needle PASS at 90% depth. Trade-off: fp8 KV is roomier per cached token but each token's KV state is larger, so concurrency at full ctx drops vs TQ3. Single-stream long-ctx works cleanly.

---

## NVLink

**Not required.** We've explicitly designed for PCIe-only consumer setups.

- 3090s have an NVLink connector but a **bridge has to be physically installed**. Most consumer setups don't have one. (Cost: ~$70-150 for a working 3-slot bridge if you wanted to add one.)
- Our composes set `NCCL_P2P_DISABLE=1` and avoid NVLink-dependent allreduce paths.
- **If you have NVLink installed and working**, single-stream TPS on dual-card will be ~1.6-1.8× single-card (vs ~1.05× without). Concurrent throughput scales similarly. Not a huge deal unless you really care about per-stream speed.

The user explicitly chose to operate without NVLink. Don't suggest adding one.

---

## Power

Production target: **330W per card** is the sweet spot — peak TPS/W efficiency and only ~5% TPS loss vs unrestricted stock (~388W).

Power lever:
```bash
sudo nvidia-smi -pm 1            # one-time: enable persistence mode
sudo nvidia-smi -pl 330 -i 0     # production default (recommended for both vLLM and llama.cpp)
sudo nvidia-smi -pl 230 -i 0     # thermal-constrained / quiet — see caveat below
```

Past 330W: diminishing returns (SM clocks saturate near 1.9 GHz on 3090s); 388W is actually *less* efficient than 330W on Qwen3.6's GDN-attention kernels.

**Caveat — 230W on llama.cpp + GDN models is more aggressive than it looks**: cross-rig data from [@syangsao](https://github.com/noonghunna/club-3090/issues/58#issuecomment-4388766174) (1× water-cooled 3090, llama.cpp + Qwen3.6 Q3_K_XL) shows **230W costs ~34% TPS** vs stock (25 vs 38 TPS) because the chunked_gated_delta_rule kernel is genuinely compute-bound on this model, not memory-bound. On vLLM + AutoRound the same cap costs less (~10-15%) because the kernel mix is GEMM-dominated. **Recommendation**: use 330W as the default cap on either engine. Drop to 230W only if you're more thermal-constrained than perf-constrained, and expect the larger penalty on llama.cpp.

**Cooling caveat**: the 388W stock numbers above are from a water-cooled rig (Alphacool Eiswolf 2 AIO 360mm) — that's what lets the card actually sustain full board power. On **air-cooled 3090s**, thermal throttling typically kicks in at ~80°C and drops effective power to ~310-340W under sustained decode load even with no software cap, so 388W → 330W gap mostly disappears — your "stock" was likely already 330W-equivalent. The 330W cap mainly helps liquid-cooled rigs by keeping the card cooler + quieter at near-zero perf cost; on air-cooled it's a soft no-op that just makes the throttling explicit.

For dual-card: combined power at 330W cap each = ~660W under heavy load — verify your PSU has at least 850W single rail. 230W cap each = ~460W combined for thermally-constrained builds.

### Cross-rig power-cap data (anchor points)

Run `sudo bash scripts/power-cap-sweep.sh --cooling air|water|aio` on a new rig to add a row. The script auto-detects the running container/model/URL, sweeps a configurable cap range, and emits a paste-ready markdown summary at `/tmp/power-cap-summary.md`. See [`scripts/power-cap-sweep.sh`](../scripts/power-cap-sweep.sh).

**Canonical cross-rig anchor command** (production-grade data — what to paste into [disc #86](https://github.com/noonghunna/club-3090/discussions/86) for a real cross-rig efficiency anchor):

```bash
sudo bash scripts/power-cap-sweep.sh \
  --cooling air|water|aio \
  --load-mode decode-concurrent \
  --concurrency auto \
  --bench-runs 3
```

Three flags matter for anchor data:
- **`--bench-runs 3`** — medians three batches per cap. Without this, single-batch variance can be 10-30%, making adjacent-cap deltas noise rather than signal.
- **`--concurrency auto`** — picks N via plateau-detection at the highest cap (selects the highest N where both TPS and draw improve >3% over previous N). Avoids both under-loading (smaller GPUs at default N=4 plateau early) and over-loading (concurrency contention drops TPS).
- **`--load-mode decode-concurrent`** — surfaces the curve on cards that don't saturate at single-stream decode (5090, larger Ada/Blackwell). 3090s often work fine at default `decode-single`, but `decode-concurrent` is safer cross-class.

**Default step-size is 10W.** Don't override unless you know why:
- `--step-size 10` (default) → ~30 caps × ~30 sec/cap = ~15-20 min total. The right resolution for finding the actual knee.
- `--step-size 50` → ~5-6 caps total. Quick smoke / single-rig sanity only — too coarse to pin down the efficiency knee for a cross-rig anchor.

| GPU | Cooling | Engine | Model | Cap | Narr TPS | Code TPS | TPS/W | Source |
|---|---|---|---|---:|---:|---:|---:|---|
| 3090 | water | llama.cpp default | Qwen3.6 27B Q3_K_XL | 230W | 25.15 | 24.86 | 0.109 | [@syangsao #58](https://github.com/noonghunna/club-3090/issues/58#issuecomment-4388766174) |
| 3090 | water | llama.cpp default | Qwen3.6 27B Q3_K_XL | **330W** ⭐ | 36.35 | 36.26 | 0.110 | [@syangsao #58](https://github.com/noonghunna/club-3090/issues/58#issuecomment-4388766174) |
| 3090 | water | llama.cpp default | Qwen3.6 27B Q3_K_XL | 388W (stock) | 38.23 | 37.97 | 0.098 | [@syangsao #58](https://github.com/noonghunna/club-3090/issues/58#issuecomment-4388766174) |
| 3090 | water | llama.cpp default | Qwen3.6 27B Q3_K_XL | **290W** ⭐ | 32.08 | 31.74 | **0.111** | @noonghunna (this rig, 18-cap 10W sweep) |
| 3090 | water | llama.cpp default | Qwen3.6 27B Q3_K_XL | 370W (stock) | 34.46 | 34.20 | 0.103 | same |
| 3090 | water | llama.cpp default | Qwen3.6 27B Q3_K_XL | 390W (max) | 35.84 | 35.66 | 0.092 | same |
| 4090 | air | llama.cpp default | Qwen3.6 27B Q3_K_XL | **260W** ⭐ | 48.41 | 48.43 | 0.186 | [@laurimyllari #62](https://github.com/noonghunna/club-3090/discussions/62#discussioncomment-16832066) |
| 4090 | air | llama.cpp default | Qwen3.6 27B Q3_K_XL | 280W | 49.54 | 49.10 | 0.177 | [@laurimyllari #62](https://github.com/noonghunna/club-3090/discussions/62#discussioncomment-16832066) |
| 4090 | air | llama.cpp default | Qwen3.6 27B Q3_K_XL | 300W | 50.26 | 50.02 | 0.168 | [@laurimyllari #62](https://github.com/noonghunna/club-3090/discussions/62#discussioncomment-16832066) |
| 4090 | air | llama.cpp default | Qwen3.6 27B Q3_K_XL | 450W (stock) | 52.28 | 52.22 | 0.116 | [@laurimyllari #62](https://github.com/noonghunna/club-3090/discussions/62#discussioncomment-16832066) |
| 5090 | air | vLLM default | Qwen3.6 27B AutoRound | **400W** ⭐ | 119.98 | 159.23 | 0.300 | [@apnar #62](https://github.com/noonghunna/club-3090/discussions/62#discussioncomment-16832685) |
| 5090 | air | vLLM default | Qwen3.6 27B AutoRound | 575W (near-stock) | 119.38 | 159.94 | 0.277 | [@apnar #62](https://github.com/noonghunna/club-3090/discussions/62#discussioncomment-16832685) |
| 5090 | air | vLLM `gemma-mtp` (TP=1) | Gemma 4 31B + MTP | **400W** ⭐ | 571.45 | 700.92 | **1.429** | [@apnar #86](https://github.com/noonghunna/club-3090/discussions/86#discussioncomment-16840610) |
| 5090 | air | vLLM `gemma-mtp` (TP=1) | Gemma 4 31B + MTP | 510W (peak narr) | 619.45 | 723.82 | 1.215 | same |
| 5090 | air | vLLM `gemma-mtp` (TP=1) | Gemma 4 31B + MTP | 600W (stock) | 600.65 | 756.67 | 1.103 | same |

⭐ = peak TPS/W efficiency on that rig.

#### Efficiency curves (10W resolution)

For rigs where we have full 10W-resolution sweeps, the curves below show TPS + TPS/W efficiency across the power envelope. These are the cross-rig anchor charts; sources + raw data are linked in each caption. To add your card class, run [`scripts/power-cap-sweep.sh`](../scripts/power-cap-sweep.sh) (canonical command above) and paste the output to [disc #86](https://github.com/noonghunna/club-3090/discussions/86).

![5090 + Gemma 4 + MTP power-cap efficiency curve (apnar)](img/power-cap-5090-gemma4.png)

*5090 air-cooled + Gemma 4 31B + MTP, 21-cap sweep at 10W resolution. Yellow callout: 400W sweet spot (1.43 TPS/W). Red-shaded: 530-600W = workload-limited, ~547W max actual draw regardless of cap. Source data: [disc #86](https://github.com/noonghunna/club-3090/discussions/86#discussioncomment-16840610) (@apnar). Source script: [`img/power-cap-5090-gemma4.py`](img/power-cap-5090-gemma4.py).*

![4090 + Qwen3.6-27B + llama.cpp power-cap efficiency curve (laurimyllari)](img/power-cap-4090-qwen36.png)

*4090 air-cooled + Qwen3.6-27B Q3_K_XL + llama.cpp default, 15-cap sweep at 10W resolution (260-400W). Yellow callout: 260W sweet spot (0.186 TPS/W) at 33% below the 4090's 450W stock TDP. **+8% TPS for +54% wattage** going from 260W to 400W — the 4090 is heavily workload-saturated on this single-stream decode path. Source data: [disc #62](https://github.com/noonghunna/club-3090/discussions/62#discussioncomment-16832066) (@laurimyllari). Source script: [`img/power-cap-4090-qwen36.py`](img/power-cap-4090-qwen36.py).*

![3090 + Qwen3.6-27B + llama.cpp power-cap efficiency curve (noonghunna)](img/power-cap-3090-qwen36.png)

*3090 water-cooled + Qwen3.6-27B Q3_K_XL + mainline llama.cpp, 18-cap sweep 200-390W (mixed bench shapes). Yellow callout: 290W sweet spot (0.111 TPS/W) at **78% of stock 370W TDP**. Orange-shaded zone 340-370W: firmware boost-state plateau where caps 340/350/360/370W all draw identical ~334W actual. At 380W cap, draw escapes to 361W; at 390W cap, draw reaches 388W — so the apparent "ceiling" at 334W isn't a hardware limit, it's a discrete boost-state behavior. Source script: [`img/power-cap-3090-qwen36.py`](img/power-cap-3090-qwen36.py).*

**Cross-rig pattern**: efficiency knee falls at **~60-85% of stock TDP** across consumer Ampere/Ada — start there for a new card class and zoom in. Ada (4090) is proportionally more aggressive than Ampere (3090) — 4090 cuts 33% of stock TDP for ~7% TPS loss; 3090 cuts 15% of stock for ~5% loss.

**5090 compute-saturation note**: @apnar's data shows the 5090 caps at ~430W actual draw on Qwen3.6-27B even when allowed up to 575W — the workload is compute-saturated, not power-saturated. So 400W cap delivers ~equal TPS to 575W. **Confirmed cross-workload on Gemma 4 31B + MTP**: 21-cap sweep at 10W resolution shows actual draw plateaus at ~547W beyond 530W cap (no thermal throttle, GPU temp peaked 66°C — compute / memory bandwidth limit, not thermal). **Same 400W sweet spot** despite ~5× different absolute TPS class. Pattern: the 5090 + consumer-air-cooled platform appears to have a workload-independent ~400W efficiency knee on this rig class.

### Interpreting "draw plateaued below cap" sweeps

If your sweep ends with the high-cap rows showing **actual draw < cap by 5-15%** (e.g. 547W actual at 600W cap on a 5090 with `decode-concurrent N=4`), it usually means one of:

| Pattern | Likely cause | What to try next |
|---|---|---|
| Draw plateau at high caps + GPU temp far below ~80°C | **Memory-bandwidth-bound decode** — the workload can't pull more power because HBM/GDDR throughput is the limit, not compute. Typical for decode-only workloads on Blackwell/Ada/Ampere. | `--load-mode prefill-heavy` (compute-bound; tests if compute can use the headroom) OR `--concurrency-stretch 4` (force more streams past plateau-detect's pick) |
| Draw plateau + GPU temp ≥ 78°C | **Thermal throttle** kicking in. Software cap not the limit; cooling is. | Improve cooling (better airflow, undervolt, water if available); not a software fix |
| Draw plateau across ALL caps (even low ones) + low GPU util | **Workload too small for card** (e.g. 7B model on 5090). Card is idling. | Use a bigger model OR `--concurrency 16+` to load it heavier |
| Draw plateau at high caps even with prefill-heavy + N=8+ | **Firmware/voltage cap**. The card has a sustained-power limit below the spec'd TDP regardless of workload. | None — that's the genuine ceiling for this card |

**Use `--concurrency-stretch N` to probe headroom**: if your sweep shows draw plateau and you want to know whether more concurrency would push power higher, run:

```bash
sudo bash scripts/power-cap-sweep.sh \
  --cooling air|water|aio \
  --load-mode decode-concurrent \
  --concurrency auto \
  --concurrency-stretch 4 \
  --bench-runs 3
```

This adds 4 streams past whatever plateau-detect picked. Per-stream TPS will drop; aggregate TPS may dip a few %; actual draw may rise — telling you whether the original plateau was a workload limit (draw stays flat) or a concurrency-contention artifact (draw rises). Default 0 = current behavior (no stretch).

**Discussion**: cross-rig power-cap data lives at [disc #86](https://github.com/noonghunna/club-3090/discussions/86). Drop your sweep there.

---

## VRAM ceilings (the cliffs)

This is model-specific but the **shapes apply across hybrid-attention models** (Qwen3-Next family, similar architectures):

- **Single 3090 (24 GB):** Cliff 1 (~25K-token tool prefills, FFN intermediate buffer) closed across all shipped variants since 2026-04-30 PM. **⚠️ Cliff 2 (~50-60K single prompts, DeltaNet GDN forward) regressed under Genesis v7.72.2** — PN59 streaming-GDN was advertised as the structural fix but doesn't engage on the chunked-prefill code path that 24 GB single-card configs are forced to take (`--max-num-batched-tokens 4128` populates `chunk_indices`/`chunk_offsets` which PN59's eligibility check rejects). `long-text.yml` / `long-text-no-mtp.yml` / `long-vision.yml` may OOM at >50K single-prompt context. Filed at [Sandermage/genesis-vllm-patches#22](https://github.com/Sandermage/genesis-vllm-patches/issues/22), pending Sander review. **Workarounds**: `dual.yml` / `dual-turbo.yml` (TP=2 escapes the cliff), or `llamacpp/default` (different engine, no Cliff 2). [See `docs/CLIFFS.md` for the full diagnostic.](CLIFFS.md)
- **Dual 3090 (48 GB combined):** TP=2 splits activation memory across cards. Cliffs are not active failure modes.

For visualization of how VRAM splits across single + dual configs, see [vram-budget-combined.svg](img/vram-budget-combined.svg) (or per-page: [single](img/vram-budget-single.svg) · [dual](img/vram-budget-dual.svg)).

---

## Disk

- **Per model**: ~20 GB for weights + Docker layers + scratch.
- **Per engine**: vLLM Docker image is ~9 GB. llama.cpp binary is ~50 MB.
- **For dual-card vLLM**: add ~2 GB for the patched vLLM source clone (`/opt/ai/vllm-src/`).

If you'll run multiple models, plan ~20 GB each.

---

## Things this stack doesn't support (hardware-wise)

- **macOS / Windows native** — Linux only (vLLM constraint). WSL2 works but with caveats (see below).
- **AMD GPUs** — vLLM has experimental ROCm support but we haven't validated. llama.cpp works on AMD via HIPBLAS.
- **Apple Silicon** — llama.cpp via Metal works for the model, but our recipes are Linux-x86-64 path-specific.
- **Intel GPUs** — llama.cpp via SYCL/oneAPI has support; not tested by us.

If you're on non-NVIDIA hardware, [`/docs/engines/LLAMA_CPP.md`](engines/LLAMA_CPP.md) is your starting point.

---

## Note for SM86 long-context single-stream TG

Independent cross-rig measurement ([sztlink/turboquant-cuda-bench, IQ4_NL repro 2026-04-27](https://github.com/sztlink/turboquant-cuda-bench/blob/main/bench/iq4nl-repro/results.md)) shows that on **SM86 (Ampere consumer, RTX 3090)** with `q8_0/turbo4` KV, single-stream token-generation rate degrades **~−71% from 0K to 131K context** vs ~−54% on **SM89 (Ada Lovelace, RTX 4090)** for the same model + KV config. The conclusion in their writeup: "SM86 has a weaker warp dispatch path for the turbo4 dequant kernel. The degradation is architectural, not model-specific." Source originally surfaced by [@lkaupp](https://github.com/ggml-org/llama.cpp/discussions/20969#discussioncomment-16733925) on a 3090.

Practical implication for our stack: even when you can fit the prompt + KV in VRAM (so Cliff 1 and Cliff 2 don't fire), **single-stream TG above ~32K-65K on SM86 pays a structural per-token rate tax** that you can't fix by changing KV format choice. The recommendation in [`docs/SINGLE_CARD.md`](SINGLE_CARD.md) to switch to llama.cpp at >60K context is partly about VRAM cliffs and partly about this dispatch-rate cliff — llama.cpp's CUDA kernels avoid the same warp-dispatch-overhead profile under Triton/AutoRound INT4.

If you're on **SM89+ hardware (RTX 4090 / 5090, A6000 Ada / Blackwell)**, the per-token rate at depth holds up materially better — the −54% drop at 131K on SM89 is still real but ~17 percentage points less severe than SM86. If long-context single-stream throughput is your primary workload, that's the hardware class to target.

---

## Note for WSL2 / Windows users

WSL2 inherits Windows' GPU timeout policy via WDDM (Windows Display Driver Model). Long-running CUDA kernels can trip **TDR (Timeout Detection and Recovery)** — Windows force-resets the GPU when a kernel exceeds the TDR delay (default 2 seconds), invalidating every CUDA allocation in flight. The signature in vLLM logs is:

```
RuntimeError: CUDA driver error: device not ready
  (during torch.ops._C.marlin_gemm.default or similar long kernel)
RuntimeError: !handles_.at(i) INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp
```

This is **not OOM** — the engine is being externally reset by the Windows display driver mid-kernel. Cross-rig validated by [@RossNE99](https://github.com/noonghunna/club-3090/issues/50) on a 2× 3090 WSL2 rig: 156K-token prompt triggers TDR reliably; smaller prompts pass.

### Fix — extend the TDR delay on the Windows host

From an Administrator PowerShell on the Windows host (not inside WSL):

```powershell
Set-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers" `
  -Name TdrDelay -Type DWord -Value 60 -Force

Set-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers" `
  -Name TdrDdiDelay -Type DWord -Value 60 -Force
```

Then **reboot Windows** (registry change only takes effect on boot). After reboot, retry the long-prompt path.

Three TDR options in increasing aggression:
- **`TdrDelay=60`** (recommended): give kernels up to 60 seconds before TDR triggers
- **`TdrLevel=0`** (more permissive): disable TDR entirely — appropriate for dedicated AI rigs where the GPU isn't your display adapter
- **Dual-boot Linux** (no TDR at all): sidesteps the whole class

### Fix — disable PyTorch `expandable_segments` if boot crashes at weight repack

A separate boot-time failure mode, distinct from TDR, surfaces with the same `device not ready` text:

```
File "/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py", line 1279, in gptq_marlin_repack
    return torch.ops._C.gptq_marlin_repack(...)
RuntimeError: CUDA driver error: device not ready
```

The engine logs `Loading weights took N seconds` cleanly, then dies ~1 second later inside `process_weights_after_loading` → `gptq_marlin_repack`. `CUDA_LAUNCH_BLOCKING=1` does not move the failure site, which rules out async-residual error from a prior kernel.

Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` resolves the crash. We haven't isolated the specific failing call — the suspicion is the `cuMemMap` virtual-memory API used internally by PyTorch's `expandable_segments:True` allocator, since (a) disabling that path fixes the crash and (b) JusefPol's NVLink boot-crash report (PR #31) responded to the same workaround. We list this section under WSL2 because that's where this PR's diagnostic rig hit it, but the failure mode itself is not strictly WSL2-only.

Known occurrences:

- JusefPol — NVLink-wired dual-3090 setups (PR #31). The `dual-nvlink*.yml` composes already hardcode `expandable_segments` off for that case.
- club-3090 issue (this PR, 2026-05-06) — single-card RTX 3090 Ti on WSL2, driver 596.36, vLLM nightly `01d4d1ad` (the v7.72.2-uplift pin).

#### Override

All single-card and PCIe dual-card composes now expose `PYTORCH_CUDA_ALLOC_CONF` as a `${...}` override knob. Drop a `.env` next to the compose file (or export the var in your shell):

```sh
# models/qwen3.6-27b/vllm/compose/.env
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

Then `docker compose up -d` as usual. No edits to tracked files needed.

#### Possible secondary effect on weight-load time

On the WSL2 rig where this was diagnosed, weight-load time on a fresh boot (caches cleared) was 32 sec with `expandable_segments:True` and 13 sec with `expandable_segments:False`. This is a single observation, not a controlled A/B (cache state, FS warmth and other factors weren't held constant), so treat it as suggestive rather than measured. If you're chasing boot-time latency on WSL2 and not crashing, the override is harmless to try.

### Additional WSL2 considerations

- vLLM auto-detects WSL2 and disables `pin_memory` (`Using 'pin_memory=False' as WSL is detected. This may slow down the performance.` in boot log) — expected behavior, can't be overridden cleanly.
- Host RAM: the 17 GB Q4 model checkpoint loads via paged read. If WSL2 host RAM is < 24 GB, vLLM will warn that auto-prefetch is disabled and fall back to slower load. Allocate at least 24 GB to WSL2 for clean model loading.

---

## Note for non-Docker / non-bare-metal runtimes

Proxmox VE, microk8s, podman, manual k8s — see [CONTAINER_RUNTIMES.md](CONTAINER_RUNTIMES.md) for environmental footnotes when the runtime / host stack diverges from the bare-metal Ubuntu + Docker baseline that the verify-* scripts target. Includes:

- The Proxmox VE 8.x / kernel 6.17.x asyncio crash class (parked, environmental, full elimination trail)
- microk8s integration (open invitation to PR an example manifest)
- podman compose env override pattern
