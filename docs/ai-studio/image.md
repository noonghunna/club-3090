# Image Studio — chat-driven stills (HiDream-O1 · Ideogram-4 · Chroma · Z-Image)

The image side of [Club 3090 AI Studio](README.md). Type a rough idea in Open WebUI; the
"director" LLM crafts it; ComfyUI renders a **still**. Four lanes share one pipe + one director,
all **single-device on GPU0** (they run in *either* `gpu-mode` — including alongside a video
render, since the video DiT's weights sit on GPU1). Pick by intent:

| Lane | Model | Best at | Prompt style | Notes |
|---|---|---|---|---|
| `✨ Image (HiDream-O1)` | HiDream-O1-Image-Dev-2604 fp8 | **top-quality general / photoreal** (AA #1 single-model open-weight) | natural language | native 2048², ~15 GB, ~3–4 min |
| `🖼️ Image` | Ideogram-4 fp8 | **design / logo / text / typography** | structured JSON (director-crafted) | safety-trained; ~18.5 GB @1024² |
| `🔓 Image (Chroma)` | Chroma1-HD fp8 | **uncensored** photoreal / illustration | natural language + negative + real CFG | ~9 GB; the "Sulphur for stills" |
| `🔓 Image (Z-Image)` | Z-Image-Turbo fp8 | **uncensored**, **fast** photoreal / general | natural language | ~7 GB, **~25 s** (8-step cfg=1); Lumina2 encoder; the quick uncensored lane |

Refine any of them by replying with the change (*"monochrome"*, *"tighter crop"*, *"at night"*,
*"flat vector style"*) — the director evolves the previous prompt and regenerates. No approval gate.

## Architecture

```
                          Browser
                             │  "a logo for a coffee shop" · "a photoreal red fox at dusk"
                             ▼
              ┌──────────────────────────────────────────────────┐
              │  Open WebUI   :8080   (the front-end)             │
              │  image lanes: ✨ HiDream-O1 · 🖼️ Ideogram-4 · 🔓 Chroma │
              └───────┬──────────────────────────────────┬───────┘
                  (1) │ craft the prompt              (2) │ render the still
                      ▼                                   ▼
        ┌──────────────────────────┐   ┌───────────────────────────┐
        │ Director   :8090         │   │ ComfyUI        :8188       │
        │ qwen3.5-4b · GPU0 ~4.6GB │   │ HiDream-O1 / Ideogram-4 /  │
        │ idea → Ideogram JSON     │   │ Chroma · single-device GPU0│
        │   caption · or HiDream / │   │ → .png                     │
        │   Chroma prose           │   └─────────────┬─────────────┘
        └──────────────────────────┘                 ▼
                                          ┌───────────────────────────┐
        OWUI's native 🖼️ button →          │ Gallery   :8189            │
        image-shim :8191 rewrites the     │ nginx over /output —       │
        plain text into an Ideogram       │ links survive ComfyUI down │
        JSON caption, then → ComfyUI      └─────────────┬─────────────┘
                                                        ▼  🖼️ link back in chat
                                                     Browser  (reply "monochrome" to refine)
```

All three image lanes are **single-device on GPU0** (the video DiT, when present, lives on GPU1) and coexist with the director. See [README.md](README.md) for the full studio substrate.

---

## ✨ HiDream-O1 (top-quality / photoreal)

The **`✨ Studio · Image (HiDream-O1)`** lane renders on **HiDream-O1-Image-Dev-2604 fp8** — a 9B
**pixel-level unified transformer** (Qwen3-VL backbone; no separate VAE / text encoder; the model
works directly in a pixel-and-token space). On [Artificial Analysis](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights)
it's the **#1 single-model open-weight** text-to-image (Elo 1189). It takes a **rich
natural-language prompt**, so the director crafts a vivid descriptive paragraph. The Dev-2604
build is **distilled**: 28-step, **CFG-off** (no negative prompt — everything lives in the
positive description). It renders at its **native 2048×2048** (the node snaps smaller requests
up), single-device on **GPU0 ~15 GB**, ~**3–4 min/image** on a 3090 (sdpa attention — flash-attn
isn't built for sm_86). Heaviest + slowest of the image lanes — the trade for top quality. Not
subject to `image_max_edge` (fixed 2048²).

> **No native ComfyUI support** (unlike Ideogram-4). Its nodes (HiDream O1 Model Loader /
> Conditioning / Sampler) come from the third-party **`Saganaki22/HiDream_O1-ComfyUI`** custom
> node, cloned by `services/comfyui/entrypoint.sh` (+ an idempotent transformers-5 compat patch);
> weights via `download_hidream_o1.sh`.

## 🖼️ Ideogram-4 (design / logo / photo / art)

The **`🖼️ Studio · Image`** lane renders on **Ideogram-4 fp8** — single-device on **GPU0**
(~18.5 GB @1024²), so it runs in either gpu-mode (no switch needed to make an image).

**The director crafts a JSON caption, not prose.** Ideogram-4 is trained on **structured JSON
captions** (`high_level_description`, a `style_description` block, and a
`compositional_deconstruction` with background + per-object elements). Hand it off-schema plain
text and it denoises to a gray **"Image blocked by safety filter"** placeholder — its built-in
fallback, *not* a real safety judgement (it fires on a plain "a red apple"). So the image director
outputs the JSON caption; the pipe validates it and falls back to wrapping your text in a minimal
caption if needed. Measured: plain text → 100% blocked; the same prompt as a JSON caption → clean
render (~80 s warm @1024²).

The lane is **category-aware**: the director infers logo / poster / UI-mockup / photo /
illustration and fills the JSON with the levers that matter (logos → vector/flat/negative
space/1–2 colours; photos → camera + lens, depth of field; etc.). Want visible text/lettering? Ask
for it in quotes. Defaults 1024×1024, 20 steps; the long edge is capped at `image_max_edge` (1024)
so it coexists with the director on GPU0 (2048² + director = OOM; raise the cap and stop the
director for 2K stills).

## 🔓 Chroma (uncensored)

The **`🔓 Studio · Image (Chroma)`** lane renders on **Chroma1-HD fp8** — a Flux-based,
de-distilled, *trained-uncensored* model (~9 GB, single-device GPU0). Unlike Ideogram, Chroma
takes a **rich natural-language prompt** (no JSON), supports a **negative prompt**, and uses **real
CFG** — so the director crafts a vivid descriptive paragraph (the uncensored qwen honours intent
without sanitising). The encoder (`t5xxl_fp16`) and VAE (Flux `ae.safetensors`) are shared with the
Flux ecosystem (already on disk), so only the Chroma DiT is model-specific. Defaults 1024×1024, 26
steps, cfg 3.5. The **uncensored stills lane** — Ideogram remains the choice for text/logos; Chroma
for unrestricted photoreal/illustration. Validated clean (~72–80 s warm).

> **Why a separate model instead of "un-censoring Ideogram":** Ideogram-4's safety is trained into
> the weights (no abliterated variant; diffusion abliteration isn't a drop-in). The image shim only
> removes Ideogram's *false-positive* blocking of neutral prompts — genuine moderation stays. So
> uncensored stills get their own model (Chroma), exactly as Sulphur is the uncensored video lane.
> Capability is in the weights; the infra is content-neutral.

## 🔓 Z-Image-Turbo (uncensored · fast)

The **`🔓 Studio · Image (Z-Image)`** lane renders on **Z-Image-Turbo fp8** — Alibaba's 6B,
**Apache-licensed**, permissively-trained text-to-image model (~7 GB, single-device GPU0). It's the
**fast** uncensored lane: an 8-step cfg=1 turbo schedule renders a coherent 1024² still in **~25 s**
(vs Chroma's ~75 s and HiDream's ~3–4 min). Native ComfyUI nodes — the text encoder is a Qwen3-4B
loaded via `CLIPLoader` type `lumina2`; VAE is the shared Flux `ae.safetensors`. Natural-language
prompt (the director crafts prose, same as Chroma). Use it when you want an uncensored still **now**;
reach for Chroma when you want real-CFG/negative control, HiDream for top-quality.

> **Krea 2 was evaluated for this slot and dropped.** Its weights are on disk, but its bespoke DiT
> (`txtfusion`/`tproj` tensors) isn't recognised by ComfyUI's local model detection — the only
> working path is the **cloud `Krea2ImageNode`**, which violates AI Studio's "fully self-hosted, no
> cloud APIs" rule. It's also aligned (not uncensored). Re-evaluate if ComfyUI adds local Krea2
> detection. Z-Image is the uncensored image pick instead.

## Native image button (via the image shim)

OWUI's built-in 🖼️ image button (on a chat message) also renders Ideogram-4 stills — but it sends
**plain text** to the image engine, which trips the same "blocked by safety filter" placeholder,
and OWUI's own image-prompt-generation can't help (it returns `{"prompt":"<string>"}` and nesting
the Ideogram JSON inside that string defeats the task models).

The fix is **`services/studio/image-shim/`** (`:8191`): a transparent **ComfyUI reverse-proxy**.
OWUI's `COMFYUI_BASE_URL` points at it (`imagegen.env`), with OWUI's image-prompt-generation turned
**off**. The shim proxies every ComfyUI call (incl. the `/ws` progress socket) straight through —
*except* `POST /prompt`, where it reads the plain-text prompt node, asks the director (qwen `:8090`)
for a rich Ideogram-4 JSON caption, and rewrites the node before forwarding. The escaping is done in
**Python** (reliable). Blast radius = image generation only — title/tag task-generation is untouched.

`gpu-mode ai-studio` starts the shim and the director. If the shim is down, point
`COMFYUI_BASE_URL` back at `:8188` (plain text then hits the placeholder) or use the
**Studio · Image lane**.

## VRAM / GPU split

All three image lanes render on **GPU0** and coexist with the ~4.6 GB director. Ideogram ~18.5 GB
@1024² + director ≈ 23 GB (fits; 2048² would OOM with the director resident). HiDream is fixed at
2048² (~15 GB) + director ≈ 20 GB. Because ComfyUI holds both cards in `ai-studio`, you can do
**video and a ≤1024² image in the same scene with no switch**.

## Models (obtain separately → `/mnt/models/comfyui/models/...`)

| File | ComfyUI dir | Lane |
|---|---|---|
| `HiDream-O1-Image-Dev-2604-FP8/` (drbaph; complete folder) | `diffusion_models/` | HiDream-O1 — needs `HiDream_O1-ComfyUI` node |
| `ideogram4_fp8_scaled.safetensors` (+ `_unconditional_`), `qwen3vl_8b_fp8_scaled`, `flux2-vae` | `diffusion_models/`, `text_encoders/`, `vae/` | Ideogram-4 |
| `Chroma1-HD-fp8mixed.safetensors` (Comfy-Org/Chroma1-HD_repackaged) | `diffusion_models/` | Chroma (uncensored) |
| `t5xxl_fp16.safetensors` + Flux `ae.safetensors` | `text_encoders/`, `vae/flux/` | Chroma (shared with Flux ecosystem) |
