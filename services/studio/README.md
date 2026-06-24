# Studio — chat-driven video + image generation (Open WebUI → ComfyUI)

A small layer that turns Open WebUI into a **text/image → video** and **text → image**
studio. You type a rough idea in chat; a "director" LLM crafts it into a professional
prompt; ComfyUI renders it on LTX-2.3 (video+audio), Sulphur (uncensored video),
Ideogram-4 (image: graphic design / logo / photo / art), or Chroma (uncensored image). Full
architecture, capabilities and the measured length limits live in **[../../docs/ai-studio/video.md](../../docs/ai-studio/video.md)**.

## Pieces

| Path | What it is |
|---|---|
| `build_studio_pipe.py` | Generates `studio_pipe.py` — the Open WebUI **Function (pipe)** that drives ComfyUI. Run it, then install the output as a Function. |
| `workflows/ltx_distilled_distorch.json` | The validated **single-stage** ComfyUI graph (8-step, cfg 1) the pipe submits for video. DisTorch splits the 22B DiT across 2 GPUs. |
| `workflows/wan22_rapid.json` | The **Wan2.2-Rapid-AllInOne** Mega NSFW v10 Q8 GGUF video graph (14B, *uncensored*, text→video). umt5 encoder + Wan 2.1 VAE; the AllInOne merge bakes a 4-step distill in → single 4-step cfg=1 sampler. 832×480×81 @16fps (~3 min/clip). No synced audio (unlike LTX). |
| `workflows/ideogram4.json` | The validated **Ideogram-4 fp8** image graph (DualModelGuider). Single-device GPU0 (~18.5 GB @1024²) — runs in either gpu-mode (no switch needed for image). |
| `workflows/chroma1_hd.json` | The **Chroma1-HD fp8** image graph (Flux-based, de-distilled, *uncensored*). Natural-language prompt + negative + real CFG. Single-device GPU0 (~9 GB); reuses `t5xxl_fp16` + Flux `ae.safetensors`. |
| `workflows/z_image_turbo.json` | The **Z-Image-Turbo fp8** image graph (Alibaba 6B, Apache, *uncensored*). Natural-language prompt, Lumina2 encoder (`qwen_3_4b`), 8-step cfg=1 turbo. Single-device GPU0 (~7 GB, ~25 s) — the **fast** uncensored image lane; reuses Flux `ae.safetensors`. |
| `workflows/hidream_o1.json` | The **HiDream-O1-Image-Dev-2604 fp8** image graph (pixel-level unified transformer; AA #1 single-model open-weight T2I). Natural-language prompt, 28-step CFG-off, native **2048²** (~15 GB GPU0, ~3–4 min/image). **Needs the `HiDream_O1-ComfyUI` custom node** (no native ComfyUI support) — cloned by `services/comfyui/entrypoint.sh` (+ a transformers-5 compat patch); weights via `download_hidream_o1.sh`. |
| `workflows/ace_step_music.json` | The **ACE-Step v1 3.5B** music graph (tags + lyrics/`[instrumental]`, seconds-duration). Single-device GPU0 (~8 GB) — songs + instrumentals to `.mp3`. |
| `workflows/stable_audio_sfx.json` | The **Stable Audio Open 1.0** sound graph (natural-language, ≤47 s). Single-device GPU0 — SFX / ambience / textures to `.mp3`. |
| `studio_pipe.py` | Built artifact (committed for convenience; regenerate with the builder). |
| `gallery/` | `docker compose` for an always-on nginx media gallery (`:8189`) over ComfyUI's output dir — keeps generated media browsable + links alive even when ComfyUI is down. |
| `enhancer/` | `docker compose` for the "director" LLM (`:8090`, OpenAI-compatible). |
| `orchestrator/` | `docker compose` + Dockerfile for the long-clip engine (`:8190`): chains ~10 s segments into one combined video for requests >15 s. The pipe POSTs here when you ask for a length. |
| `image-shim/` | `docker compose` + Dockerfile for the native-button image shim (`:8191`): a transparent ComfyUI reverse-proxy that crafts an Ideogram-4 JSON caption (via the director) on `POST /prompt`, so OWUI's built-in 🖼️ image button renders instead of the "blocked by safety filter" placeholder. Point OWUI's `COMFYUI_BASE_URL` at it. See ai-studio/video.md "Native image button". |
| `tts/` | `docker compose` + Dockerfile for integrated voices (`:8192`): **Kokoro-82M** (ONNX, CPU) generates a voiceover and a **layer-aware ffmpeg mixdown** ducks it over the clip's native audio + loudness-normalizes. The pipe POSTs `/narrate` when the message has a `voiceover:`/`narration:` directive. No GPU. See ai-studio/video.md "Integrated audio". |
| `step-voice/` | `docker compose` + Dockerfile for the **premium voice** service (`:8193`): **Step-Audio-EditX** (3B, Apache) — zero-shot voice cloning + emotion/style/paralinguistic **editing**. **ISOLATED container** pinned to `transformers==4.53.3` (the version the model needs; conflicts with ComfyUI's 5.x), GPU (~14 GB bf16, pinned to a free card). The pipe POSTs `/clone`. On-demand (not always-on). Weights: `Step-Audio-EditX` + `Step-Audio-Tokenizer` under `models/Step-Audio/`. |
| `extend_chain.py` | The same chaining as a standalone host CLI (handy for scripted long renders). |
| `push-pipe-to-owui.sh` | Regenerate `studio_pipe.py` **and push it into the running Open WebUI function + reload**. OWUI stores the pipe code in its DB (not from the file), so after editing `build_studio_pipe.py` you must update the installed function — this does it in one command. `--no-reload` to skip the OWUI restart. |

## Install the pipe into Open WebUI

```bash
python3 build_studio_pipe.py            # writes studio_pipe.py
```

Then in Open WebUI: **Admin → Functions → +**, paste the contents of `studio_pipe.py`,
save, enable. Eleven models appear in the picker (naming format: `Studio · <Modality> (<Model> · <descriptor>)`):

- `🎬 Studio · Video (LTX-2.3)` — video + synced audio (stock model)
- `🔓 Studio · Video (Sulphur)` — uncensored video (LTX-2.3-22B-dev fine-tune)
- `🔓 Studio · Video (10Eros)` — uncensored video (LTX-2.3-native dev fine-tune; A/B vs Sulphur)
- `🔓 Studio · Video (Wan2.2)` — uncensored video, text→video (Wan2.2-Rapid Mega NSFW; no synced audio)
- `✨ Studio · Image (HiDream-O1)` — top-quality / photoreal stills (natural-language prompt)
- `🖼️ Studio · Image` — Ideogram-4 (graphic design / logo / photo / text)
- `🔓 Studio · Image (Chroma)` — uncensored stills (natural-language prompt)
- `🔓 Studio · Image (Z-Image)` — uncensored stills, **fast** (~25 s; natural-language prompt)
- `🎵 Studio · Music` — ACE-Step (songs + instrumentals)
- `🔊 Studio · SFX` — Stable Audio (sound effects + ambient)
- `🎙️ Studio · Voice` — Step-Audio-EditX premium voice (zero-shot clone + emotion/style)

> **Updating an already-installed pipe:** OWUI keeps the pipe **code in its DB**, not from the
> file — so regenerating `studio_pipe.py` alone won't take effect (the classic "stale function"
> trap). After any change to `build_studio_pipe.py`, run **`bash push-pipe-to-owui.sh`** (rebuilds
> + writes the new code into the OWUI `studio` function + restarts OWUI to reload it). First-time
> install is still the paste step above.

Set the pipe's **Valves** (gear icon on the function):
- `comfyui_url` → your ComfyUI (`http://host.docker.internal:8188` from the OWUI container)
- `chat_url` / `chat_model` → the director (`http://host.docker.internal:8090/v1`, `qwen3.5-4b-uncensored`)
- `browser_base` → the gallery at **your host's LAN IP** (e.g. `http://192.168.x.x:8189`) so returned video/image links open in your browser
- `frames` → default 241 (~10 s). Hard-capped at 361 (~15 s); see ai-studio/video.md for why.
- `image_width` / `image_height` / `image_steps` → image defaults (1024×1024, 20 steps). `image_max_edge` caps the long edge at 1024 so the image gen coexists with the director on GPU0 (2048² would OOM unless the director is stopped).

> **Why the image lane crafts a JSON prompt:** Ideogram-4 is trained on **structured JSON
> captions** and emits an "Image blocked by safety filter" placeholder for off-schema
> (plain-text) input — so the director outputs the JSON caption, not prose. Plain text
> sent straight to Ideogram-4 (e.g. Open WebUI's native 🖼️ image button via `imagegen.env`)
> hits that placeholder; use the **Studio · Image** lane, which crafts the JSON for you.

## Bring it up

`bash scripts/gpu-mode.sh ai-studio` brings up ComfyUI (both GPUs) + the director +
the gallery + Open WebUI as a unit. Or start pieces individually:

```bash
docker compose -f services/studio/gallery/docker-compose.yml up -d     # always-on gallery
docker compose -f services/studio/enhancer/docker-compose.yml up -d    # director :8090
docker compose -f services/comfyui/docker-compose.yml up -d            # ComfyUI :8188
```

## Use

Pick a Studio model, type a scene (or attach an image to animate). The director crafts
the prompt and it renders — you get a link to the clip or image. **Refine by just replying**
with what to change (video: "more moody", "make it night", "slower camera"; image:
"monochrome", "tighter crop", "flat vector style"); it evolves the previous prompt and
regenerates. No approval gate.

> Models (Sulphur, LTX-2.3 distilled, the director GGUF) are obtained separately — see
> the file manifest in [docs/ai-studio/video.md](../../docs/ai-studio/video.md).
