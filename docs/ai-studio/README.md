# Club 3090 AI Studio

A **chat-driven, open-weight creative studio** for **image, video, and audio** generation, running
on a 2× RTX 3090 workstation — all behind one consistent flow in Open WebUI:

> **casual prompt → a "director" LLM crafts it → ComfyUI / a service renders → gallery link → reply to refine.**

Fully self-hosted, no cloud APIs. Uncensored lanes where the model allows. One director, one
gallery, one refine-by-reply UX across the creative modalities.

> **Scope.** AI Studio is the *creative-generation* umbrella — **image · video · audio**. **Text**
> (chat / agentic LLM serving — the model catalog, engines, composes, KV, topology) is the **core
> rig stack**, documented separately in the global architecture, **not** a Studio modality. The
> qwen "**director**" used here is a small prompt-crafting *helper service*, not a chat lane — Open
> WebUI is just the shared front-end for both.

| Deep-dive | Covers |
|---|---|
| **[requirements.md](requirements.md)** | **Can I run this?** — GPU / CPU / RAM / disk / software, single-vs-dual-card, the director-placement VRAM lever |
| **[image.md](image.md)** | HiDream-O1 (top quality) · Ideogram-4 (design/logo/text) · Chroma + Z-Image (uncensored) · Krea 2 (aesthetic) · the native-button shim |
| **[video.md](video.md)** | LTX-2.3 (video+audio) · Sulphur / 10Eros (uncensored) · Wan2.2 (uncensored T2V) · 60 s+ chaining · the single-stage rule |
| **[audio.md](audio.md)** | Step-Audio-EditX (premium voice clone+edit) · Kokoro (narration) · ACE-Step (music) · Stable Audio (SFX) |

---

## The 12 lanes

Pick a lane in the OWUI model picker; the director crafts the right prompt shape for it.
They all live in the single **`ai-studio`** scene — `gpu-mode ai-studio` brings the whole
creative surface up; you switch *lanes* in OWUI, not gpu-mode *modes*.

| Lane | Model | Modality | License |
|---|---|---|---|
| 🎬 `Studio · Video (LTX-2.3)` | LTX-2.3 distilled 22B | video + synced audio | open |
| 🔓 `Studio · Video (Sulphur)` | Sulphur (LTX-2.3 dev FT) | video (uncensored) | open |
| 🔓 `Studio · Video (10Eros)` | 10Eros (LTX-2.3 dev FT) | video (uncensored) | open |
| 🔓 `Studio · Video (Wan2.2)` | Wan2.2-Rapid-AllInOne Mega v10 | video (uncensored, text→video) | Apache |
| ✨ `Studio · Image (HiDream-O1)` | HiDream-O1-Image-Dev-2604 | image — **top-quality / photoreal** (AA #1 single-model open-weight) | MIT |
| 🖼️ `Studio · Image` | Ideogram-4 fp8 | image — design / logo / text | open |
| 🔓 `Studio · Image (Chroma)` | Chroma1-HD fp8 | image (uncensored) | open |
| 🔓 `Studio · Image (Z-Image)` | Z-Image-Turbo fp8 | image — uncensored, **fast** (~25 s) | Apache |
| 🎨 `Studio · Image (Krea 2)` | Krea 2 Turbo fp8 | image — **aesthetic / stylized** (~40 s) | Krea |
| 🎵 `Studio · Music` | ACE-Step v1 3.5B | music — songs + instrumentals | open |
| 🔊 `Studio · SFX` | Stable Audio Open 1.0 | sound effects / ambience | open |
| 🎙️ `Studio · Voice` | Step-Audio-EditX 3B | premium voice — clone + emotion/style edit | **Apache** |

Video lanes can also mix a **Kokoro voiceover** onto the clip (a directive in the message; see
[audio.md](audio.md)). _(Text chat / agentic serving is the core rig stack, not a Studio lane — the
Studio only borrows a small qwen "director" to craft prompts.)_

## Architecture

```
                              Browser  —  Open WebUI :8080
                                 │  pick a lane · type an idea · reply to refine
                                 ▼
                    ┌──────────────────────────────────────────┐
                    │  Studio pipe  (OWUI Function)             │
                    │  routes the lane · returns gallery links  │
                    └───────┬──────────────────────────┬───────┘
                   craft (1)│                  render (2)│
                            ▼                            ▼
              ┌──────────────────────┐   ┌──────────────────────────────────────┐
              │ Director   :8090     │   │ Renderers                            │
              │ qwen3.5-4b · GPU0    │   │  • ComfyUI :8188 — image · video ·   │
              │ idea → crafted prompt│   │      music · SFX  (GPU0; video uses  │
              │ (JSON / prose /      │   │      both GPUs via DisTorch)         │
              │  tags / sound)       │   │  • step-voice :8193 — premium voice  │
              └──────────────────────┘   │      (isolated, transformers 4.53.3) │
              ┌──────────────────────┐   │  • studio-tts :8192 — Kokoro (CPU)   │
              │ image-shim :8191     │   │      voiceover ducked onto a clip    │
              │ proxy for OWUI's 🖼️  │   └──────────────────┬───────────────────┘
              │ button → JSON caption│   long video >15s → orchestrator :8190
              └──────────────────────┘        (chains ~10s segments + mux)
                                                           ▼
                                           ┌───────────────────────────┐
                                           │ Gallery :8189 (nginx)      │
                                           │ /output — survives ComfyUI │
                                           │ down; ▶️/🖼️/🎧 links in chat │
                                           └───────────────────────────┘
```

The qwen **director** crafts the right prompt shape per lane; **ComfyUI** renders image/video/music/SFX; the **step-voice** and **studio-tts** services handle premium + narration voice; the **orchestrator** chains long videos; everything lands in the always-on **gallery**. Text/LLM chat is the separate core stack (this is image/video/audio only).

### One scene, lanes inside it

> There's a **single** `ai-studio` gpu-mode scene now (it replaced the old separate
> `image-studio`/`video-studio` modes). `gpu-mode ai-studio` brings up ComfyUI on **both
> GPUs** + the director + all the sidecars; you pick image / video / audio / voice **as a
> lane in OWUI** — no gpu-mode switching between modalities. Same trade-off as switching
> tools in a DAW/NLE on one box, but it's all one workspace.

ComfyUI runs **one workflow at a time**, so the lanes time-share the cards:

- **GPU0 lanes (coexist with the director):** all 5 image lanes, music, SFX — single-device.
- **Both-GPU lane:** **video** (the 22B DiT splits across both 3090s via DisTorch).
- **GPU1 ⊕ video:** **premium voice** (Step-Audio-EditX, ~14 GB on GPU1) is on-demand and
  **mutually exclusive with an active video render** (both want GPU1) — c3 guards this.

**The hardware truth (measured):** during a video render GPU1 holds the 22B DiT (~22 GB donor) and
GPU0 does compute (~7–14 GB) **+** the ~4.6 GB director — so a ≤1024² image lane *also* fits on
GPU0 in `ai-studio` with no switch. Heavy modalities time-share (one ComfyUI queue), not
simultaneous — a workstation reality, framed like switching tools in a creative suite.

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
| **`gpu-mode`** | — | the mode switcher (`ai-studio` / chat / off) |

The OWUI Studio pipe (`services/studio/build_studio_pipe.py` → `studio_pipe.py`) routes each lane
to the right backend and returns a gallery link. Install it once: **Admin → Functions → +**, paste
`studio_pipe.py`, enable.

## Why this is interesting

- **Fully open-weight + self-hosted** — no API, no cloud, no per-call cost; your data stays local.
- **Uncensored lanes where the model allows** — Sulphur (video), Chroma (image), the uncensored
  director — capability lives in the *weights*; the infrastructure is content-neutral.
- **One consistent director-driven UX** across image / video / audio.
- **Honest constraint as a feature:** heavy modalities are mode-switched, not simultaneous —
  lightweight combos (chat + a ≤1024² image + a voice) coexist.

## On the uncensored models

The Sulphur DiT, Chroma, and the director are uncensored fine-tunes — chosen so the creative lanes
don't refuse or sanitize. That capability is in the model weights; the infra is content-neutral. To
craft prompts through an **aligned** model instead, point the pipe's `chat_model` valve at e.g.
gemma-4-12b — the uncensored DiTs still render, only the prompt-writing changes.

## Bring it up

**One command** (fresh clone → generating) — builds the ComfyUI image, downloads the full roster
(~120 GB), brings the scene up, and installs the OWUI Studio pipe:

```bash
bash scripts/setup-ai-studio.sh        # add --yes to skip the confirm; SKIP_BUILD / SKIP_DOWNLOAD / SKIP_DISK_CHECK / SKIP_PIPE to trim
```

**Where the models land:** everything goes under your **`MODEL_DIR`** (set in repo-root `.env`, e.g. via c3 Settings) — the HF/GGUF weights at `$MODEL_DIR`, and the ComfyUI tree at a `comfyui` sibling of it (e.g. `MODEL_DIR=/home/me/models` → ComfyUI at `/home/me/comfyui`). Override `COMFYUI_ROOT` / `COMFYUI_MODELS_DIR` to decouple them. Resuming a download that already pushed you under the free-space threshold? `SKIP_DISK_CHECK=1` bypasses the preflight (the roster pull is idempotent and only fetches what's missing).

**Already set up?** Just bring the scene up (or do it from c3 → Operate):

```bash
bash scripts/gpu-mode.sh ai-studio   # ComfyUI (both cards) + director + gallery + orchestrator + shim + tts + OWUI
# premium voice (on demand):  docker compose -f services/studio/step-voice/docker-compose.yml up -d
```

Then open Open WebUI at `http://<your-host>:8080`, **sign up** (first account = admin), set the pipe's
`browser_base` valve to your host's LAN IP (`http://<your-host>:8189`), and pick a lane. (If you signed
up *after* `setup-ai-studio.sh` ran, install the pipe with `bash services/studio/push-pipe-to-owui.sh`.)
Per-modality setup + model manifests are in [image.md](image.md) / [video.md](video.md) /
[audio.md](audio.md); requirements in [requirements.md](requirements.md). The service bundle itself is
documented in [`services/studio/README.md`](../../services/studio/README.md).
