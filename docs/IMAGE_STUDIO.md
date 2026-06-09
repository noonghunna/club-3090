# Image Studio — local image generation + chat

A self-hosted bundle that gives you **text-to-image generation and an LLM chat in one
browser UI**, both running locally on your own GPUs:

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** runs **Ideogram-4** (fp8) for image generation.
- **[Open WebUI](https://github.com/open-webui/open-webui)** is the front-end — chat plus a 🖼️ image button that calls ComfyUI.
- **gemma-4-12b** (llama.cpp) is the default chat model, sized to **coexist** with image gen on a second GPU.

On a 2-GPU box the two run at once (image gen on GPU 0, chat on GPU 1). On a single GPU
they're mutually exclusive (see [Single-GPU](#single-gpu)).

---

## Architecture

```
                          Browser
                             │
                             ▼
              ┌──────────────────────────────────────────┐
              │  Open WebUI   :8080   (the front-end)     │
              │    • chat            • 🖼️  image button    │
              └───────┬──────────────────────┬────────────┘
            chat      │                       │   image gen
        (OpenAI API)  │                       │  (ComfyUI API)
                      ▼                       ▼
        ┌─────────────────────────┐   ┌───────────────────────────┐
        │ gemma-4-12b   :8069     │   │ ComfyUI        :8188       │
        │ llama.cpp               │   │ Ideogram-4 fp8 workflow    │
        │ GPU 1  ·  ~14 GB        │   │ GPU 0  ·  ~18.5 GB @1024²  │
        └────────────┬────────────┘   └─────────────┬─────────────┘
                     │  (also routed via            │  loads from
              LiteLLM :4000 gateway —               ▼
              + the rest of the LLM      models/{diffusion_models,
              catalog when their gpu-      text_encoders, vae}
              mode is up)                  (Ideogram-4 fp8 set, ~27 GB)
```

- **Two GPUs, concurrent.** Validated live: image gen ~18–22 GB on **GPU 0** *and* the chat
  model ~14 GB on **GPU 1** at the same time — sampled during a real generation.
- **Open WebUI** owns the UX: chat goes to a text model; the 🖼️ button POSTs the prompt to
  ComfyUI's API and pulls back the image.
- **Chat routing.** By default Open WebUI talks **directly** to gemma-4-12b (`:8069`) — a clean
  picker that only shows the model that's actually live in image-studio mode. gemma-4-12b is
  **also** registered on the **LiteLLM gateway** (`:4000`) alongside the full LLM catalog, so
  API clients / `benchlocal` can reach it the same way as every other model (the catalog's big
  models only respond when their own `gpu-mode` is up — they're GPU-mutex with ComfyUI).
- **ComfyUI** is a **fixed Docker service** (`services/comfyui`), pinned to a known-good ComfyUI
  commit; it lazy-loads the Ideogram-4 weights into VRAM only *during* a generation.

---

## Quickstart

```bash
bash scripts/setup-image-studio.sh
```

That builds the ComfyUI image, downloads the Ideogram-4 model set (~27 GB), and brings the
stack up via `gpu-mode image-studio`. Then open:

- **Open WebUI** → `http://<your-host>:8080` — start here (chat + 🖼️ image button)
- **ComfyUI** → `http://<your-host>:8188` — full node-graph control

### First run

1. **Create your account.** Open the Open WebUI URL and **sign up** — the *first* account
   created becomes the **admin**. (The setup script doesn't pre-set credentials; you choose
   them here. There's no hardcoded secret — Open WebUI generates its own per deployment.)
2. **Chat.** Pick **`gemma-4-12b…`** in the model selector (top of the chat) and talk to it.
3. **Generate an image.** Send a prompt, then click the **🖼️ image icon** on the assistant's
   reply — it renders via Ideogram-4 on ComfyUI and appears inline. Image generation is
   **already enabled and wired** (the setup sets `ENABLE_IMAGE_GENERATION` + the ComfyUI
   backend); you can inspect/tweak it under **Admin → Settings → Images** (steps, size, CFG).

> First image after a cold ComfyUI takes ~2 min (it loads ~20 GB of weights). Warm
> generations are ~70 s at 1024².

The script asks for confirmation before the heavy build/download; pass `--yes` (or `CI=1`)
to skip the prompt. Skip flags: `SKIP_DOWNLOAD=1` (weights already present),
`SKIP_BUILD=1` (image already built).

---

## Two front-ends — which to use

| Want… | Use | Why |
|---|---|---|
| **Easy** — type a prompt, get an image, chat | **Open WebUI** (`:8080`) | One box + the 🖼️ button. Your daily driver. |
| **Control** — steps, CFG, seed, structured prompts, img2img | **ComfyUI** (`:8188`) | The node graph. Drop in when you want to tune. |

In Open WebUI, image generation rides on the chat: send a prompt, then click the **🖼️
picture icon** on the assistant's reply to render it via Ideogram-4. (It won't appear in
the model selector — that's only for chat models.)

In ComfyUI, load the bundled **Ideogram 4** template (Workflow → Browse Templates → Image).
All model files are already in place, so it loads with no missing nodes.

---

## Modes (`gpu-mode`)

Image gen, video gen, and chat are **GPU-mutually-exclusive** at the heavy end (a video
model wants both cards; image + a small chat model fit on one card each). So the switcher
is a resource-mode manager:

| Mode | What it runs | GPUs |
|---|---|---|
| `gpu-mode image-studio` | ComfyUI/Ideogram-4 + gemma-4-12b chat + Open WebUI | GPU 0 (image) + GPU 1 (chat) |
| `gpu-mode comfyui` | ComfyUI only (all GPUs) — for video / large image jobs | all |
| `gpu-mode chat` | Open WebUI + LiteLLM (no local GPU model) | none |

Within ComfyUI, switch *what* you generate by loading a different workflow/template.

---

## VRAM by resolution (Ideogram-4 fp8, measured on one RTX 3090)

| Resolution | Peak VRAM | Time | Notes |
|---|---|---|---|
| 1024×1024 | ~18.5 GB | ~70 s warm | comfortable on a 24 GB card |
| 2048×2048 | ~21.8 GB (89%) | ~320 s | fits but tight — **batch size 1 only**; larger or batched → OOM |

It runs **single-device** — a second GPU doesn't speed up one generation. For routine
high-res, prefer **generate at 1024² then upscale** (higher quality and lower peak VRAM
than native 2048²).

---

## Chat model

Default is **gemma-4-12b** on the spare GPU (`:8069`), so chat and image gen run at the
same time. Open WebUI points **directly** at it — a clean picker showing only the model
that's actually live in image-studio mode.

It's **also** registered on the **LiteLLM gateway** (`:4000`, `model_name: gemma-4-12b`) next
to the rest of the catalog, so API clients reach it the same way as every other model. To
make Open WebUI itself route through LiteLLM (and see the whole catalog), swap to the
commented `OPENAI_API_BASE_URL` block in `services/openwebui/docker-compose.yml` — but note
LiteLLM's *larger* models are GPU-mutex with ComfyUI, so they only respond when their own
`gpu-mode` is up (you'd see them in the picker but they'd error in image-studio mode). That's
exactly why the bundle defaults to gemma-direct.

### Single-GPU

With one GPU, image gen and a local chat model can't run together. `gpu-mode image-studio`
detects this and starts ComfyUI only; for chat use `gpu-mode chat` (LiteLLM) or run a local
model while ComfyUI is down.

---

## Troubleshooting

**Image button missing / image gen not configured in Open WebUI.** Open WebUI's image
settings are *PersistentConfig* — the values in `services/openwebui/imagegen.env` apply only
on a **fresh data volume** (first boot). If you reused an existing `open-webui-data` volume,
set it manually: **Admin → Settings → Images** → Engine `ComfyUI`, Base URL
`http://host.docker.internal:8188`, then load the Ideogram-4 workflow (or recreate the volume).

**Out of memory at high resolution.** Drop back to 1024² (+ upscale), and keep batch size 1
at 2048². Ideogram-4 fp8 peaks ~21.8 GB at 2048² — there's little headroom on a 24 GB card.

**First generation is very slow.** Cold ComfyUI loads ~20 GB (two fp8 transformers + the
text encoder). The first request after boot warms it; subsequent ones are ~70 s.

**First ComfyUI boot takes minutes.** The entrypoint clones ComfyUI + custom nodes and
installs requirements on first run. Tail it: `sudo docker logs -f comfyui`.

---

## What's installed

- Image model: Ideogram-4 fp8 (`services/comfyui/download_ideogram4.sh`) — two transformers
  + Qwen3-VL-8B text encoder + flux2 VAE, in the ComfyUI models tree.
- ComfyUI built via `services/comfyui/Dockerfile`, **pinned to a known-good commit** (native
  Ideogram-4 support; `COMFYUI_REF=HEAD` to float). Entrypoint is mounted so the pin applies
  on `up` without a rebuild.
- Open WebUI pinned to `v0.9.6`; image-gen wired via `services/openwebui/imagegen.env`; secret
  key auto-generated per deployment.
- Chat: `models/gemma-4-12b/llama-cpp/compose/single/unsloth-q8kxl/base.yml` on the spare GPU,
  also routed on LiteLLM (`services/litellm/config.yaml`).

> **Video & audio generation** are planned follow-ons (this page will gain `video-studio` /
> `audio-studio` sections as they land).
