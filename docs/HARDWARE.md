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

Production target: **230W per card** (default cap, quiet, cool, stable).

Power lever:
```bash
sudo nvidia-smi -pl 230 -i 0    # production default
sudo nvidia-smi -pl 330 -i 0    # ~+10% mean TPS during heavy sessions
```

Past 330W: diminishing returns (SM clocks saturate near 1.9 GHz on 3090s).

For dual-card: combined power at 230W cap each = ~460W. Most modern 850W+ ATX PSUs handle this comfortably. If you push to 330W per card, you're at ~660W peak under heavy load — verify your PSU has at least 850W single rail.

---

## VRAM ceilings (the cliffs)

This is model-specific but the **shapes apply across hybrid-attention models** (Qwen3-Next family, similar architectures):

- **Single 3090 (24 GB):** Cliff 1 (~25K-token tool prefills, FFN intermediate buffer) closed across all shipped variants since 2026-04-30 PM. **Cliff 2 (~50-60K single prompts, DeltaNet GDN forward) closed at 60K** as of 2026-05-02 PM via Genesis v7.69 (PN32 + P103 worker self-install) plus a local backport of vllm#35975 — see `long-text.yml` (180K + MTP K=3, balanced) and `long-text-no-mtp.yml` (200K + no MTP, max-context). Both top out at 60K hardware-physical wall on 24 GB single-card. [See `docs/CLIFFS.md` for the full diagnostic.](CLIFFS.md)
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

- **macOS / Windows native** — Linux only (vLLM constraint). WSL2 might work but isn't tested.
- **AMD GPUs** — vLLM has experimental ROCm support but we haven't validated. llama.cpp works on AMD via HIPBLAS.
- **Apple Silicon** — llama.cpp via Metal works for the model, but our recipes are Linux-x86-64 path-specific.
- **Intel GPUs** — llama.cpp via SYCL/oneAPI has support; not tested by us.

If you're on non-NVIDIA hardware, [`/docs/engines/LLAMA_CPP.md`](engines/LLAMA_CPP.md) is your starting point.
