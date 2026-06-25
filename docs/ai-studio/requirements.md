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
>
> One knob drives it: **`STUDIO_DIRECTOR_DEVICE=gpu0|gpu1|cpu`** in the rig `.env` — set it from
> **c3 → Settings → "Director placement"**, or edit the line directly. `gpu-mode` reads it on every
> scene launch and translates it to the container's `CUDA_VISIBLE_DEVICES` + `-ngl`:
> - **`gpu0`** (default) → GPU0, snappy refine-by-reply (~1.4 s craft).
> - **`gpu1`** → the second card. **Safe only when GPU1 has room** — the image lanes and the Wan video
>   lane (18 GB donor + 4.5 GB = 22.5 GB fits). **NOT** the LTX / Sulphur / 10Eros lanes: they use GPU1
>   as their ~22 GB DisTorch donor, so a director there OOMs them.
> - **`cpu`** → universally safe, frees the GPU entirely (model mmaps into ~5 GB system RAM, SSD-backed),
>   at a craft-latency cost (single-digit tok/s for a 4B on CPU vs ~1.4 s on GPU — noticeable before a
>   fast image lane, invisible before a multi-minute video). Cap its cores with `DIRECTOR_THREADS`
>   (default 8). **A CPU director is the always-on path:** it uses no GPU, so it survives scene switches
>   and stays live as the uncensored model in Open WebUI — `gpu-mode` only evicts a *GPU*-placed director
>   when a dual-card LLM scene needs the cards.
>
> The `chat` scene also starts the director (honoring the same knob) as the **supporting-infra home for
> Catalog models** — see [the chat scene + Catalog note](#chat-scene--the-catalog-support-layer) below.
>
> Keep it on GPU0 for the snappy refine-by-reply UX in AI-Studio; choose `cpu` for an always-on chat
> companion, or `gpu1` to unlock a GPU0 edge case (Ideogram 2048², single-window Wan >121 frames).

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

## Chat scene — the Catalog-support layer

The studio's front-of-house services aren't studio-only. The **`chat` scene** (`gpu-mode chat`) brings
up the same supporting infra — **Open WebUI** + **LiteLLM** + **Qdrant** (vector DB / document RAG) +
**SearXNG** (web search) + the **uncensored director** (`:8090`) — *without* a scene GPU model. It's the
home base for **Catalog models**: anything you launch ad-hoc outside a scene.

```bash
gpu-mode chat                                  # supporting infra + director, no scene LLM
bash scripts/switch.sh --owui <variant>        # launch any catalog model AND register it into OWUI
```

`switch.sh --owui` (also reachable from **c3 → Catalog → Serve**) starts the chosen variant and wires it
into Open WebUI as a direct connection, so it appears in the model picker alongside the director, with
**web search and document RAG already attached**. You get a full chat workstation for any model in the
catalog without authoring a scene for it.

The director's placement here honors the same **`STUDIO_DIRECTOR_DEVICE`** knob (c3 → Settings → Director
placement). Set it to **`cpu`** to keep the director **always-on**: it uses no GPU, so it persists across
scene switches and stays available in OWUI even while a dual-card LLM scene owns both cards.
