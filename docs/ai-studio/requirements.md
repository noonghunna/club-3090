# AI Studio — requirements

What you need to self-host [Club 3090 AI Studio](README.md). Everything is **open-weight and runs
locally** — no cloud APIs, no accounts. Values are minimum / recommended; the reference rig is a
2× RTX 3090 workstation, but nothing here is specific to it.

## TL;DR

- **2× 24 GB NVIDIA GPUs** — Ampere (sm_86) or newer (RTX 3090 / 4090 / A5000-class). PCIe is fine,
  **no NVLink required**.
- **Linux + Docker** with the NVIDIA Container Toolkit; a driver new enough for the container CUDA (12.4+).
- **~120 GB disk** for the full model roster (less if you skip lanes); SSD recommended.
- **32 GB+ system RAM**.

A *subset* runs on a single 24 GB card — see **Single-GPU** below.

## GPU

| Lane group | Models | VRAM | Card(s) |
|---|---|---|---|
| **Director** | Qwen3.5-4B-Uncensored (prompt crafter, llama.cpp) | ~4.5 GB | one card; or CPU / the idle second card (configurable — see note) |
| **Image** | Ideogram-4 (~18.5 GB) · HiDream-O1 (~15 GB) · Chroma (~9 GB) · Z-Image (~7 GB) | up to ~18.5 GB | single card (GPU0), coexists with the director |
| **Video** | LTX-2.3 22B · Sulphur / 10Eros 22B · Wan2.2 14B | ~22 GB weights + ~7–14 GB compute | **both cards** — DisTorch donates the DiT weights to GPU1, compute runs on GPU0 |
| **Music / SFX** | ACE-Step (~8 GB) · Stable Audio | single card | GPU0 |
| **Premium voice** | Step-Audio-EditX | ~14 GB | a free card, **on-demand** (⊕ mutually exclusive with an active video render) |

**Why two cards:** the video DiTs (22B LTX / 14B Wan) plus their compute exceed one 24 GB card, so
they split across both via **DisTorch** — a *VRAM* split (weights stored on the second card, compute
on the first), not a compute split, so it works on **PCIe with no NVLink**. Image / audio / voice
each fit one card.

> **Director placement (a VRAM lever).** The director is a small, latency-tolerant helper — ~4.5 GB,
> **~1.4 s** to craft a prompt (measured, GPU), then idle. **Default: GPU0**, which coexists with every
> shipped default lane. It's also the swing factor on the single-card Wan ceiling: its 4.5 GB on GPU0
> caps the single-card 480p window at ~121 frames; **freeing it lifts that to 161** (measured).
> Relocate it to reclaim GPU0:
> - `STUDIO_DIRECTOR_GPU=1` → the second card. **Safe only when GPU1 has room** — the image lanes and
>   the Wan video lane (18 GB donor + 4.5 GB = 22.5 GB fits). **NOT** the LTX / Sulphur / 10Eros lanes:
>   they use GPU1 as their ~22 GB DisTorch donor, so a director there OOMs them.
> - `-ngl 0` → **CPU**: universally safe, frees the GPU entirely (~5 GB RAM), at a craft-latency cost
>   (tens of seconds for a 4B on CPU vs ~1.4 s on GPU — noticeable before a fast image lane, invisible
>   before a multi-minute video).
>
> Keep it on GPU0 for the snappy refine-by-reply UX; move it only to unlock an edge case (Ideogram
> 2048², single-window Wan >121 frames).

**Single-GPU (1× 24 GB):** image + music + SFX + the director run comfortably; **video is the
constraint** — a 22B DiT won't fit one card at full resolution. Treat **dual-card as recommended**
and single-card as "image + audio studio, video best-effort (short / low-res)."

## CPU + RAM

- **CPU:** a modern multi-core (8+ cores). Drives the **Kokoro** narration TTS (ONNX, CPU), the
  **orchestrator** (ffmpeg long-clip concat / mux), the **image-shim** proxy, and — optionally — the
  director when CPU-hosted.
- **RAM:** **32 GB** minimum, **64 GB+** comfortable. Add ~5 GB if the director runs on CPU.

## Disk

~**120 GB** for the full open-weight roster (GGUF / fp8). SSD recommended — the 18–22 GB video GGUFs
load faster. Per modality:

| Modality | Models | Disk |
|---|---|---|
| **Video** | LTX-2.3 + Sulphur + 10Eros (22.8 GB each) + Wan2.2 (18.7 GB) | ~87 GB |
| **Image** | Ideogram-4 (9 GB) + Z-Image (6 GB) + HiDream-O1 + Chroma | ~25–35 GB |
| **Audio** | ACE-Step (7.7 GB) + Stable Audio (4.9 GB) + Kokoro (0.3 GB) | ~13 GB |
| **Director** | Qwen3.5-4B-Uncensored GGUF | ~2.5 GB |
| **Shared** | text encoders (umt5, qwen3-4b, t5) + VAEs | ~15 GB |

Skip lanes you don't want — [`scripts/lib/studio-models.tsv`](../../scripts/lib/studio-models.tsv) is
the manifest, and each lane's weights are an independent download
(`services/comfyui/download_*.sh`). `bash services/comfyui/download_studio_models.sh` fetches the
whole roster (idempotent — only what's missing).

## Software

- **OS:** Linux (the images are CUDA Linux containers).
- **Docker** + **NVIDIA Container Toolkit** (GPU passthrough). No host CUDA toolkit needed — CUDA
  lives in the images.
- **NVIDIA driver:** recent enough for the container CUDA (12.4+; the reference rig runs a CUDA-13 driver).
- **Images (pulled / built on first bring-up):** ComfyUI (custom build, pinned commit + custom nodes
  for HiDream-O1, GGUF, and DisTorch multi-GPU), **llama.cpp** (the director), an **isolated**
  Step-Audio-EditX container (pinned `transformers==4.53.3`), **nginx** (gallery), **Open WebUI**.
- **No cloud APIs / accounts** — content capability lives in the open weights; the infra is content-neutral.

## Bring it up

```bash
bash services/comfyui/download_studio_models.sh   # fetch the roster (~120 GB, idempotent)
gpu-mode ai-studio                                 # ComfyUI (both cards) + director + sidecars + OWUI
```

Then open Open WebUI and pick a lane. Full per-lane detail in [image.md](image.md) /
[video.md](video.md) / [audio.md](audio.md); the service bundle is in
[`services/studio/README.md`](../../services/studio/README.md).
