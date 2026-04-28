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
| RTX 4090 | 24 GB | sm_89 | Should work; ~30% faster decode (newer SMs); same memory characteristics |
| RTX 5090 | 32 GB | sm_120 | Untested; more VRAM relaxes the prefill cliffs but kernel paths might differ |
| RTX A6000 | 48 GB | sm_86 | Should work; double VRAM lets you skip the cliff workarounds (use Sandermage's reference defaults) |
| H100 SXM | 80 GB | sm_90 | Different beast; flash-attn 3 paths available; not what these recipes target |

**Won't work:** anything with <24 GB VRAM (3060, 3070, 3080 12 GB). The 27B model in INT4 is ~18 GB — KV pool + activations push past 24 GB on smaller cards even with aggressive quantization.

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

- **Single 3090 (24 GB):** Two activation-memory cliffs at ~25K-token tool prefills (TurboQuant scratch) and ~50-60K single prompts (DeltaNet GDN forward). [See model-specific INTERNALS.md for deep dive.](../models/qwen3.6-27b/INTERNALS.md)
- **Dual 3090 (48 GB combined):** TP=2 splits activation memory across cards. Cliffs are not active failure modes.

For visualization of how VRAM splits, see [vram-budget.svg](img/vram-budget.svg).

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
