# Club 3090 AI Studio

A **chat-driven, all-modality, open-weight creative studio** that runs on a 2× RTX 3090
workstation — **text, image, video, and audio**, all behind one consistent flow in Open WebUI:

> **casual prompt → a "director" LLM crafts it → ComfyUI / a service renders → gallery link → reply to refine.**

Fully self-hosted, no cloud APIs. Uncensored lanes where the model allows. One director, one
gallery, one refine-by-reply UX across every modality.

| Deep-dive | Covers |
|---|---|
| **[image.md](image.md)** | HiDream-O1 (top quality) · Ideogram-4 (design/logo/text) · Chroma (uncensored) · the native-button shim |
| **[video.md](video.md)** | LTX-2.3 (video+audio) · Sulphur (uncensored) · 60 s+ chaining · the single-stage rule |
| **[audio.md](audio.md)** | Step-Audio-EditX (premium voice clone+edit) · Kokoro (narration) · ACE-Step (music) · Stable Audio (SFX) |

---

## The 8 lanes

Pick a lane in the OWUI model picker; the director crafts the right prompt shape for it.

| Lane | Model | Modality | License |
|---|---|---|---|
| 🎬 `Studio · LTX-2.3` | LTX-2.3 distilled 22B | video + synced audio | open |
| 🔓 `Studio · Sulphur` | Sulphur (LTX-2.3 dev FT) | video (uncensored) | open |
| ✨ `Studio · Image (HiDream-O1)` | HiDream-O1-Image-Dev-2604 | image — **top-quality / photoreal** (AA #1 single-model open-weight) | MIT |
| 🖼️ `Studio · Image` | Ideogram-4 fp8 | image — design / logo / text | open |
| 🔓 `Studio · Image (Chroma)` | Chroma1-HD fp8 | image (uncensored) | open |
| 🎵 `Studio · Music` | ACE-Step v1 3.5B | music — songs + instrumentals | open |
| 🔊 `Studio · SFX` | Stable Audio Open 1.0 | sound effects / ambience | open |
| 🎙️ `Studio · Voice` | Step-Audio-EditX 3B | premium voice — clone + emotion/style edit | **Apache** |

Plus **text** (chat / agentic) — the always-on qwen director itself. Video lanes can also mix a
**Kokoro voiceover** onto the clip (a directive in the message; see [audio.md](audio.md)).

## Architecture: lanes vs. modes

> A **lane** is anything light enough to coexist with the director on **GPU0** — it's just a pipe
> route, no GPU-mode switch. A **mode** (`gpu-mode <name>`) is for anything that needs **both
> GPUs** (video) or evicts the LLMs. Same trade-off as switching tools in a DAW/NLE on one box.

- **Lanes (coexist on GPU0):** all 3 image lanes, music, SFX — single-device, run in either mode.
- **Modes (need both cards / evict):** **video** (22B DiT split across both 3090s via DisTorch).
- **Premium voice** (Step-Audio-EditX, ~14 GB) and the future realtime voice agent run as their
  own services, brought up on demand.

**The hardware truth (measured):** during a video render GPU1 holds the 22B DiT (~22 GB donor) and
GPU0 does compute (~7–14 GB) **+** the ~4.6 GB director — so a ≤1024² image lane *also* fits on
GPU0 in `video-studio` with no switch. Heavy modalities are mode-switched, not simultaneous — a
workstation reality, framed like switching tools in a creative suite.

## Shared substrate (services)

| Service | Port | Role |
|---|---|---|
| **ComfyUI** | 8188 | the renderer (image/video/music/SFX lanes) |
| **Director** (`enhancer/`) | 8090 | qwen3.5-4b-uncensored — casual idea → crafted prompt; always-on, GPU0 ~4.6 GB |
| **Gallery** (`gallery/`) | 8189 | always-on nginx over the output dir — links survive ComfyUI down |
| **Orchestrator** (`orchestrator/`) | 8190 | long-clip chaining + ffmpeg mux (host-side, no GPU) |
| **Image shim** (`image-shim/`) | 8191 | ComfyUI reverse-proxy — crafts Ideogram JSON for the native 🖼️ button |
| **Studio TTS** (`tts/`) | 8192 | Kokoro-82M (CPU) voiceover + layer-aware ffmpeg mixdown |
| **Step-Voice** (`step-voice/`) | 8193 | Step-Audio-EditX premium voice (isolated, transformers 4.53.3, GPU, on-demand) |
| **`gpu-mode`** | — | the mode switcher (`video-studio` / `image-studio` / chat / off) |

The OWUI Studio pipe (`services/studio/build_studio_pipe.py` → `studio_pipe.py`) routes each lane
to the right backend and returns a gallery link. Install it once: **Admin → Functions → +**, paste
`studio_pipe.py`, enable.

## Why this is interesting

- **Fully open-weight + self-hosted** — no API, no cloud, no per-call cost; your data stays local.
- **Uncensored lanes where the model allows** — Sulphur (video), Chroma (image), the uncensored
  director — capability lives in the *weights*; the infrastructure is content-neutral.
- **One consistent director-driven UX** across text / image / video / audio.
- **Honest constraint as a feature:** heavy modalities are mode-switched, not simultaneous —
  lightweight combos (chat + a ≤1024² image + a voice) coexist.

## On the uncensored models

The Sulphur DiT, Chroma, and the director are uncensored fine-tunes — chosen so the creative lanes
don't refuse or sanitize. That capability is in the model weights; the infra is content-neutral. To
craft prompts through an **aligned** model instead, point the pipe's `chat_model` valve at e.g.
gemma-4-12b — the uncensored DiTs still render, only the prompt-writing changes.

## Bring it up

```bash
bash scripts/gpu-mode.sh video-studio   # ComfyUI (both cards) + director + gallery + orchestrator + shim + tts + OWUI
# image-only:  bash scripts/gpu-mode.sh image-studio
# premium voice (on demand):  docker compose -f services/studio/step-voice/docker-compose.yml up -d
```

Then open Open WebUI at `http://<your-host>:8080`, set the pipe's `browser_base` valve to your
host's LAN IP (`http://<your-host>:8189`), and pick a lane. Per-modality setup + model manifests
are in [image.md](image.md) / [video.md](video.md) / [audio.md](audio.md). The service bundle
itself is documented in [`services/studio/README.md`](../../services/studio/README.md).
