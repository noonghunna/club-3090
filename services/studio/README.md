# Studio — chat-driven video + image generation (Open WebUI → ComfyUI)

A small layer that turns Open WebUI into a **text/image → video** and **text → image**
studio. You type a rough idea in chat; a "director" LLM crafts it into a professional
prompt; ComfyUI renders it on LTX-2.3 (video+audio), Sulphur (uncensored video), or
Ideogram-4 (image: graphic design / logo / photo / art). Full architecture, capabilities
and the measured length limits live in **[../../docs/VIDEO_STUDIO.md](../../docs/VIDEO_STUDIO.md)**.

## Pieces

| Path | What it is |
|---|---|
| `build_studio_pipe.py` | Generates `studio_pipe.py` — the Open WebUI **Function (pipe)** that drives ComfyUI. Run it, then install the output as a Function. |
| `workflows/ltx_distilled_distorch.json` | The validated **single-stage** ComfyUI graph (8-step, cfg 1) the pipe submits for video. DisTorch splits the 22B DiT across 2 GPUs. |
| `workflows/ideogram4.json` | The validated **Ideogram-4 fp8** image graph (DualModelGuider). Single-device GPU0 (~18.5 GB @1024²) — runs in either gpu-mode (no switch needed for image). |
| `studio_pipe.py` | Built artifact (committed for convenience; regenerate with the builder). |
| `gallery/` | `docker compose` for an always-on nginx media gallery (`:8189`) over ComfyUI's output dir — keeps generated media browsable + links alive even when ComfyUI is down. |
| `enhancer/` | `docker compose` for the "director" LLM (`:8090`, OpenAI-compatible). |
| `orchestrator/` | `docker compose` + Dockerfile for the long-clip engine (`:8190`): chains ~10 s segments into one combined video for requests >15 s. The pipe POSTs here when you ask for a length. |
| `image-shim/` | `docker compose` + Dockerfile for the native-button image shim (`:8191`): a transparent ComfyUI reverse-proxy that crafts an Ideogram-4 JSON caption (via the director) on `POST /prompt`, so OWUI's built-in 🖼️ image button renders instead of the "blocked by safety filter" placeholder. Point OWUI's `COMFYUI_BASE_URL` at it. See VIDEO_STUDIO.md "Native image button". |
| `extend_chain.py` | The same chaining as a standalone host CLI (handy for scripted long renders). |

## Install the pipe into Open WebUI

```bash
python3 build_studio_pipe.py            # writes studio_pipe.py
```

Then in Open WebUI: **Admin → Functions → +**, paste the contents of `studio_pipe.py`,
save, enable. Three models appear in the picker:

- `🎬 Studio · LTX-2.3` — video + audio (stock model)
- `🔓 Studio · Sulphur` — uncensored video lane
- `🖼️ Studio · Image` — Ideogram-4 (graphic design / logo / photo / art)

Set the pipe's **Valves** (gear icon on the function):
- `comfyui_url` → your ComfyUI (`http://host.docker.internal:8188` from the OWUI container)
- `chat_url` / `chat_model` → the director (`http://host.docker.internal:8090/v1`, `qwen3.5-4b-uncensored`)
- `browser_base` → the gallery at **your host's LAN IP** (e.g. `http://192.168.x.x:8189`) so returned video/image links open in your browser
- `frames` → default 241 (~10 s). Hard-capped at 361 (~15 s); see VIDEO_STUDIO.md for why.
- `image_width` / `image_height` / `image_steps` → image defaults (1024×1024, 20 steps). `image_max_edge` caps the long edge at 1024 so the image gen coexists with the director on GPU0 (2048² would OOM unless the director is stopped).

> **Why the image lane crafts a JSON prompt:** Ideogram-4 is trained on **structured JSON
> captions** and emits an "Image blocked by safety filter" placeholder for off-schema
> (plain-text) input — so the director outputs the JSON caption, not prose. Plain text
> sent straight to Ideogram-4 (e.g. Open WebUI's native 🖼️ image button via `imagegen.env`)
> hits that placeholder; use the **Studio · Image** lane, which crafts the JSON for you.

## Bring it up

`bash scripts/gpu-mode.sh video-studio` brings up ComfyUI (both GPUs) + the director +
the gallery + Open WebUI as a unit. Or start pieces individually:

```bash
docker compose -f services/studio/gallery/docker-compose.yml up -d     # always-on gallery
docker compose -f services/studio/enhancer/docker-compose.yml up -d    # director :8090
bash scripts/gpu-mode.sh comfyui                                       # ComfyUI :8188
```

## Use

Pick a Studio model, type a scene (or attach an image to animate). The director crafts
the prompt and it renders — you get a link to the clip or image. **Refine by just replying**
with what to change (video: "more moody", "make it night", "slower camera"; image:
"monochrome", "tighter crop", "flat vector style"); it evolves the previous prompt and
regenerates. No approval gate.

> Models (Sulphur, LTX-2.3 distilled, the director GGUF) are obtained separately — see
> the file manifest in [docs/VIDEO_STUDIO.md](../../docs/VIDEO_STUDIO.md).
