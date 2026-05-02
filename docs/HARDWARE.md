# Hardware notes

What this stack assumes about your hardware. True regardless of which model or engine you're running.

---

## Required

- **NVIDIA RTX 3090 (24 GB, Ampere SM 8.6)** — 1, 2, or 4 cards.
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

**Not required for the 1× / 2× baseline.** The original measured dual-card substrate was PCIe-only, and those composes keep working on plain PCIe consumer setups.

- 3090s have an NVLink connector but a **bridge has to be physically installed**. Most consumer setups don't have one.
- The dual-card composes are still conservative and keep `--disable-custom-all-reduce`.
- On a four-card host with two existing NVLink pairs, use [`QUAD_CARD.md`](QUAD_CARD.md): the new quad composes keep TP groups inside the linked pairs instead of treating all four GPUs as a flat TP=4 group.
- If you run a dual-card variant on a four-card paired host, pin it to one physical pair with `CUDA_VISIBLE_DEVICES=0,1` or `CUDA_VISIBLE_DEVICES=2,3`.

Do not suggest adding NVLink to a PCIe-only rig. If bridges are already installed, the relevant optimization is GPU ordering and pair pinning.

---

## Power

Production target: **230W per card** (default cap, quiet, cool, stable).

Power lever:
```bash
sudo nvidia-smi -pl 230 -i 0    # production default
sudo nvidia-smi -pl 330 -i 0    # ~+10% mean TPS during heavy sessions
```

Past 330W: diminishing returns (SM clocks saturate near 1.9 GHz on 3090s).

For dual-card: combined power at 230W cap each = ~460W. For quad-card: ~920W at 230W each before CPU/platform draw. If you push to 330W per card, dual is ~660W and quad is ~1320W under heavy load — verify the PSU and cabling before raising caps.

---

## VRAM ceilings (the cliffs)

This is model-specific but the **shapes apply across hybrid-attention models** (Qwen3-Next family, similar architectures):

- **Single 3090 (24 GB):** Cliff 1 (~25K-token tool prefills, FFN intermediate buffer) closed across all shipped variants since 2026-04-30 PM via Genesis PN8 / PN12 anchor sidecar. Cliff 2 (~50-60K single prompts, DeltaNet GDN forward) still applies. [See `docs/CLIFFS.md` for the full diagnostic.](CLIFFS.md)
- **Dual 3090 (48 GB combined):** TP=2 splits activation memory across cards. Cliffs are not active failure modes.
- **Quad 3090 (96 GB combined):** use PP=2 × TP=2 or two independent TP=2 replicas on paired-NVLink hosts. `quad-pairs.yml` is measured and includes a router on `:8020`; `quad.yml` still needs local VRAM/TPS measurement before publishing numbers.

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
