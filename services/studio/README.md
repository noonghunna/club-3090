# Studio — chat-driven video generation (Open WebUI → ComfyUI)

A small layer that turns Open WebUI into a **text/image → video** studio. You type a
rough idea in chat; a "director" LLM crafts it into a professional cinematic prompt;
ComfyUI renders it on LTX-2.3 (video+audio) or Sulphur (uncensored). Full architecture,
capabilities and the measured length limits live in **[../../docs/VIDEO_STUDIO.md](../../docs/VIDEO_STUDIO.md)**.

## Pieces

| Path | What it is |
|---|---|
| `build_studio_pipe.py` | Generates `studio_pipe.py` — the Open WebUI **Function (pipe)** that drives ComfyUI. Run it, then install the output as a Function. |
| `workflows/ltx_distilled_distorch.json` | The validated **single-stage** ComfyUI graph (8-step, cfg 1) the pipe submits. DisTorch splits the 22B DiT across 2 GPUs. |
| `studio_pipe.py` | Built artifact (committed for convenience; regenerate with the builder). |
| `gallery/` | `docker compose` for an always-on nginx media gallery (`:8189`) over ComfyUI's output dir — keeps generated media browsable + links alive even when ComfyUI is down. |
| `enhancer/` | `docker compose` for the "director" LLM (`:8090`, OpenAI-compatible). |
| `orchestrator/` | `docker compose` + Dockerfile for the long-clip engine (`:8190`): chains ~10 s segments into one combined video for requests >15 s. The pipe POSTs here when you ask for a length. |
| `extend_chain.py` | The same chaining as a standalone host CLI (handy for scripted long renders). |

## Install the pipe into Open WebUI

```bash
python3 build_studio_pipe.py            # writes studio_pipe.py
```

Then in Open WebUI: **Admin → Functions → +**, paste the contents of `studio_pipe.py`,
save, enable. Two models appear in the picker:

- `🎬 Studio · LTX-2.3` — video + audio (stock model)
- `🔓 Studio · Sulphur` — uncensored lane

Set the pipe's **Valves** (gear icon on the function):
- `comfyui_url` → your ComfyUI (`http://host.docker.internal:8188` from the OWUI container)
- `chat_url` / `chat_model` → the director (`http://host.docker.internal:8090/v1`, `qwen3.5-4b-uncensored`)
- `browser_base` → the gallery at **your host's LAN IP** (e.g. `http://192.168.x.x:8189`) so returned video links open in your browser
- `frames` → default 241 (~10 s). Hard-capped at 361 (~15 s); see VIDEO_STUDIO.md for why.

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
the prompt and it renders — you get a link to the clip. **Refine by just replying** with
what to change ("more moody", "make it night", "slower camera"); it evolves the previous
prompt and regenerates. No approval gate.

> Models (Sulphur, LTX-2.3 distilled, the director GGUF) are obtained separately — see
> the file manifest in [docs/VIDEO_STUDIO.md](../../docs/VIDEO_STUDIO.md).
