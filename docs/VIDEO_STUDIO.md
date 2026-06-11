# Video Studio — chat-driven text/image → video (+ image) on 2× 3090

Type a rough idea in Open WebUI; a "director" LLM crafts it into a professional prompt;
ComfyUI renders it. Three lanes share one pipe and one director: **LTX-2.3** (video+audio),
**Sulphur** (an uncensored LTX-2.3 fine-tune), and **Ideogram-4** (image: graphic design /
logo / photo / art). Runs on the same 2× RTX 3090 box as the rest of the stack — video is
GPU-mutually-exclusive with the dual-card LLMs; the image lane runs on GPU0 in either mode.

This is the **P2 / video** sibling of [IMAGE_STUDIO.md](IMAGE_STUDIO.md); the **Image lane**
section below folds Ideogram-4 stills into the same chat-driven, director-crafted flow.

---

## Architecture

```
                          Browser
                             │  "a 40-second drone shot over a coastline"
                             ▼
              ┌──────────────────────────────────────────────┐
              │  Open WebUI   :8080   (the front-end)         │
              │  lanes: 🎬 LTX · 🔓 Sulphur · 🖼️ Image (Ideogram)│
              └───────┬────────────────────────────┬──────────┘
                  (1) │ craft the prompt        (2) │ render
                      ▼                             ▼
        ┌──────────────────────────┐   ┌───────────────────────────┐
        │ Director   :8090         │   │ ComfyUI        :8188       │
        │ qwen3.5-4b · llama.cpp   │   │ LTX-2.3 / Sulphur 22B GGUF │
        │ GPU0 · ~4.5 GB           │   │ DisTorch · BOTH 3090s      │
        │ casual idea → pro prompt │   │ → .mp4 (video + audio)     │
        └──────────────────────────┘   └─────────────┬─────────────┘
                                       long clip >15s │ (else straight to gallery)
                                                      ▼
                                        ┌───────────────────────────┐
                                        │ Orchestrator   :8190       │
                                        │ chain ~10s segments →      │
                                        │ one combined clip · no GPU │
                                        └─────────────┬─────────────┘
                                                      ▼
                                        ┌───────────────────────────┐
                                        │ Gallery   :8189            │
                                        │ nginx over /output —       │
                                        │ links survive ComfyUI down │
                                        └─────────────┬─────────────┘
                                                      ▼  ▶️ link back in chat
                                                   Browser   (reply "make it night" to refine)
```

- **Studio pipe** (`services/studio/build_studio_pipe.py` → `studio_pipe.py`): the OWUI
  Function. Video lanes (LTX, Sulphur) × two modes (text→video, image→video, auto-detected
  from whether you attach an image), plus an **Image lane** (Ideogram-4 stills) — see the
  *Image lane* section. One director, one gallery across all lanes.
- **Director** (`services/studio/enhancer/`): a small uncensored LLM that turns a casual
  line into a cinematic spec. Optional — falls back to your raw prompt if it's down.
- **ComfyUI** (`services/comfyui/`): the renderer. The 22B DiT is split across both cards
  by `UnetLoaderGGUFDisTorch2MultiGPU` (compute on GPU0, weights donated from GPU1).
- **Gallery** (`services/studio/gallery/`): always-on nginx serving ComfyUI's output dir,
  so media + links stay alive even when ComfyUI is stopped.

## Quickstart

No one-shot installer yet (the models are large + sourced separately) — three steps:

**1. Get the models.** Diffusion weights → `/mnt/models/comfyui/models/...` (see the
**Models** manifest near the end of this doc); the director GGUF →
`/mnt/models/huggingface/qwen3.5-4b-gguf/...`.

**2. Bring the stack up:**

```bash
bash scripts/gpu-mode.sh video-studio
```

Stops the GPU LLMs and starts ComfyUI (both cards) + director (`:8090`) + gallery
(`:8189`) + orchestrator (`:8190`) + Open WebUI. `gpu-mode off` (or any LLM mode) tears the
video model down again — it's GPU-mutex with the dual-card LLMs.

**3. Install the pipe into Open WebUI** (once):

```bash
python3 services/studio/build_studio_pipe.py     # writes services/studio/studio_pipe.py
```

In Open WebUI → **Admin → Functions → +**, paste `services/studio/studio_pipe.py`, save,
enable. Two models appear: **🎬 Studio · LTX-2.3** and **🔓 Studio · Sulphur**. Set the
pipe's **`browser_base`** valve to your host's LAN IP (`http://<your-host>:8189`) so the
returned video links open from your browser. Then open **Open WebUI** → `http://<your-host>:8080`.

### First run

1. **Create your account** — the first signup becomes admin (no hardcoded secret; Open WebUI
   generates its own per deployment).
2. **Pick a Studio lane** in the model selector — 🎬 LTX-2.3 (video + audio) or 🔓 Sulphur.
3. **Type a scene** — *"a fox padding through a neon alley at night"* — and send. The director
   crafts a cinematic prompt and it renders; you get a ▶️ link to the clip.
4. **Refine** by replying (*"more moody"*, *"make it night"*); for a **long clip**, include a
   duration (*"a 40-second…"*) and it auto-chains segments into one combined video.

> First render after a cold ComfyUI takes a few minutes (loads the 22B DiT + first-boot node
> deps). A 10 s clip is ~2.5 min warm; longer clips scale ~linearly per segment. See
> [How to prompt](#how-to-prompt) for what works best.

## The UX: craft-and-go, refine anytime (no approval gate)

1. **You** type something light — *"a fox in the city"*.
2. **The director** rewrites it into a full cinematographer's prompt (subject + action,
   camera/lens/movement, lighting + time of day, palette + mood, ambient sound) and it
   **renders immediately** — no "confirm?" step. The crafted prompt is shown above the
   video so you see what was generated.
3. **Refine** by just replying with the change — *"more moody"*, *"make it night"*,
   *"slower camera"*. The pipe carries the previous prompt forward (hidden marker in its
   reply) and the director **evolves** it rather than starting over. Or type a brand-new
   idea and it starts fresh — the director decides which.

Attach an **image** instead of (or with) text → it auto-routes to the **image→video**
lane (animates your still).

## How to prompt

**You don't write the cinematic prompt — the director does.** Give it the *intent* in a
line or two; it fills in camera, lens, lighting, palette, mood, and ambient sound. A
throwaway *"a fox in a neon city"* becomes a full shot. If you *do* want specific control,
just name it and the director keeps it — e.g. *"…top-down drone shot, golden hour, melancholic"*.

- **Length** — put a duration in the message: *"a **30-second** timelapse…"*, *"make it **1
  minute**"*. No duration → ~10 s. Over 15 s auto-chains ~10 s segments into one clip
  (capped ~120 s); each segment adds ~2.5 min of render time.
- **Lane** — pick the model: **🎬 LTX-2.3** (video + audio) or **🔓 Sulphur** (uncensored).
- **Image → video** — attach an image (optionally with a motion note like *"slow zoom in,
  leaves drifting"*); it animates your still.
- **Voiceover** — add *"voiceover: …"* / *"narration: '…'"* / *"say: …"* and a Kokoro voice is
  mixed over the clip (ducked under the ambient, normalized). See *Integrated audio*.
- **Refine** — just reply with the change: *"more moody"*, *"make it night"*, *"slower
  camera"*, *"add rain"*. It evolves the last prompt; a brand-new idea starts fresh.

**Works best:** one clear subject + one continuous camera move or action; slow / cinematic
/ ambient scenes; a defined mood or time of day — these render most cleanly, and chain
most seamlessly for long clips.

**Weaker / avoid:** fast or chaotic action (especially across long-clip segment joins — a
cut has no motion carry-over); lots of on-screen **text or logos**; many distinct subjects
or hard scene-cuts inside one segment; exact object counts. Keep one segment = one coherent
moment; use a longer duration (more segments) for a scene that needs to evolve.

**Examples**
- *"a hummingbird at a red flower, macro, soft morning light"* → a clean ~10 s macro shot.
- *"a 40-second drone flight over a foggy coastline at dawn, slow push forward"* → 4 chained
  segments → one combined ~40 s clip.
- then *"make it stormy, darker"* → re-crafts from that and regenerates.

## What it can generate

| | |
|---|---|
| **Modes** | text→video, image→video (attach an image) |
| **Audio** | yes — LTX-2.3 generates synced ambient audio |
| **Resolution** | Sulphur 1280×720 · LTX 768×512 (set in the workflow) |
| **Length** | default ~10 s; see the ceiling below |
| **Lanes** | `🎬 LTX-2.3` · `🔓 Sulphur` (video) · `✨ Image` (HiDream-O1) · `🖼️ Image` (Ideogram-4) · `🔓 Image` (Chroma) · `🎵 Music` (ACE-Step) · `🔊 SFX` (Stable Audio) — see *Image lanes* / *Music lane* / *SFX lane* |

### Length ceiling (measured on 2× 3090, 1280×720, frames = 24·seconds + 1)

A frame sweep on the single-stage Sulphur lane:

| Frames | Length | Result |
|--:|--:|---|
| 121 | ~5 s | crisp |
| 241 | ~10 s | **crisp — the default** |
| 361 | ~15 s | coherent end-to-end, but visibly lower-energy/softer |
| 481 | ~20 s | **collapses** — near-uniform/garbage frames the whole clip |

So the pipe **defaults to 241 (10 s)** and is **hard-capped at 361 (15 s)**: a 20 s
single-pass silently corrupts (it returns with no error, just unusable frames), so the
cap prevents you from hitting it by accident. **VRAM is not the limiter** — the weights sit
on GPU1 (~22 GB, fixed); longer clips only grow GPU0's latent (peaks ~14 GB, lots of
headroom). The wall is model coherence, not memory. Wall time scales ~linearly
(~2.5 min at 10 s, ~6.5 min at 15 s).

> Past ~15 s you **extend/chunk**: render segments ≤15 s, condition each on the previous
> segment's last frame, concatenate into one clip — see *Longer videos* below.

## Longer videos (60 s+)

Past the ~15 s single-pass ceiling, the studio **chains segments**: segment 1 is
text→video; each later segment is image→video conditioned on the **previous segment's last
frame**; all are ffmpeg-concatenated into one clip. Validated on 2× 3090 — the joins are
**visually seamless** (the last-frame conditioning carries the scene across each cut; a
slow camera move continues unbroken). Caveat: a single frame has no *velocity*, so **fast
action** can show a brief motion reset at a cut; slow/ambient scenes are clean (native LTX
temporal-extend would smooth fast cuts — future).

**In chat (default):** just ask for a length — *"a 40-second drone shot over a coastline"*.
The pipe parses the duration, the director crafts the prompt, and the **orchestrator**
(`services/studio/orchestrator/`, `:8190`) chains `ceil(seconds/10)` ~10 s segments and
returns **one combined video** (with live "segment k/N" progress). Capped at
`max_seconds` (default 120 s = 12 segments; each segment ~2.5 min to render). If the
orchestrator is down, the request falls back to a single capped clip.

**CLI (host):** the same chain is also a standalone tool —
`python3 services/studio/extend_chain.py "<prompt>" <n_segments> <frames_per_seg>`.

> Why a separate orchestrator: the OWUI pipe can't run ffmpeg or read the output dir, so
> the segment chaining + concat live in a tiny host-side service (ffmpeg + output access,
> no GPU). The pipe just submits a job and polls.

### The single-stage rule

Sulphur is a fine-tune of LTX-2.3-**dev**. The "official" dev recipe is 2-stage (a spatial
upscaler + a refine pass). On this hardware that 2-stage path renders a **diamond-lattice
mesh** over every frame. The fix — and what the pipe ships — is **single-stage**: splice
the distilled LoRA onto the base sampler, 8 steps, cfg 1, no upscaler. Clean output. The
workflow (`workflows/ltx_distilled_distorch.json`) already encodes this.

## Integrated audio (voices for video)

The video lanes already render a clip *with* native ambient audio. The Studio can add a
**voiceover/narration** on top: include a directive in your message and the pipe generates a
voice and mixes it over the clip.

- **Ask for it:** *"a fox padding through a neon alley at night, **voiceover: the city never
  sleeps, and neither do we**"* — or `narration: "..."`, or `say: ...`. The pipe pulls the
  spoken line out (so it doesn't pollute the video prompt), renders the clip, then narrates.
- **Engine:** **`studio-tts`** (`:8192`) — **Kokoro-82M** on **CPU** (never touches the GPUs),
  so it adds no VRAM pressure and runs after the render (voice ≈ 1–2 s of compute).
- **Layer-aware mixdown** (ffmpeg): the Kokoro voice is mixed over the clip's native audio, the
  **bed is ducked** under the voice (`sidechaincompress`), and the master is **loudness-normalized**
  (`loudnorm`, −16 LUFS); output audio is capped to the clip length. The mix stage is structured
  to accept more layers (generated music / SFX) later without a rewrite — only the voice layer is
  populated today (that's the planned audio-studio's job). Pick a voice with the `narrate_voice`
  valve (`af_heart`, `am_adam`, `bf_emma`, …); if `studio-tts` is unreachable the clip is returned
  silent (un-narrated).

> Lightweight "voices for video." A dedicated audio-studio (long-form TTS, ACE-Step music, SFX,
> full multi-layer mix) is the planned next phase.

## Music lane (ACE-Step)

The **🎵 Studio · Music** lane generates **songs + instrumentals** from a text idea on
**ACE-Step v1 (3.5B)** — a standalone audio track (not muxed onto video; that's the voiceover
path above).

- **Ask for it:** pick the 🎵 lane and describe the music — *"upbeat synthwave instrumental"*,
  *"a melancholic piano ballad about the sea"*, *"a lofi hip-hop beat to study to"*. Add a length
  (*"a 30-second…"*) or it defaults to ~60 s.
- **Director → tags + lyrics:** ACE-Step takes **tags** (genre / mood / instruments / tempo /
  vocal type) plus **lyrics** (`[verse]`/`[chorus]` structure) or `[instrumental]`. The director
  turns your idea into both — real singable lyrics for a song, or `[instrumental]` for a
  beat/score. Refine like everything else (*"more upbeat"*, *"add a sax solo"*, *"make it instrumental"*).
- **Single-device GPU0** (~8 GB, 50-step euler, cfg 5) — light enough to be a **lane** (coexists
  with the director on GPU0), not a separate mutex mode. ~6–16 s for a short clip on this rig.
  Output is an `.mp3` in the gallery. Uses the bundled ACE-Step ComfyUI nodes.

> First pieces of the broader audio-studio (music + SFX now; long-form TTS / a full multi-layer
> mix reusing the integrated-voices mixdown are the planned extensions).

## SFX lane (Stable Audio)

The **🔊 Studio · SFX** lane generates **sound effects, ambiences, and textures** from a text
description on **Stable Audio Open 1.0** — distinct from the Music lane (which is for
songs/instrumentals).

- **Ask for it:** pick the 🔊 lane and describe a sound — *"rain on a tin roof"*, *"sci-fi door
  whoosh"*, *"forest ambience with distant birds"*. Add a length (*"a 5-second…"*) or it defaults
  to ~10 s. **Capped at 47 s** (the model's max).
- **Director → sound prompt:** the director turns your idea into a concrete sound description
  (source, materials, acoustic space, motion). Refine like the rest (*"more distant"*, *"add
  reverb"*, *"heavier rain"*).
- **Single-device GPU0** — a lane like music. Output is an `.mp3` in the gallery. Reuses the
  bundled Stable Audio ComfyUI nodes (Stable Audio Open 1.0 + a T5-base encoder).

## Image lanes (HiDream-O1 quality · Ideogram-4 design · Chroma uncensored)

Three still-image lanes share the pipe and the director: **HiDream-O1** (top-quality general /
photoreal), **Ideogram-4** (design / logo / text), and **Chroma** (uncensored). All are
single-device GPU0 and run in either gpu-mode. Pick by intent — HiDream is the quality leader
for general images/photos; Ideogram is best at typography/logos but **safety-trained**; Chroma
is **uncensored** (the "Sulphur for stills") but weaker at crisp text.

### ✨ HiDream-O1 (top-quality / photoreal)

The **✨ Studio · Image (HiDream-O1)** lane renders on **HiDream-O1-Image-Dev-2604 fp8** — a 9B
**pixel-level unified transformer** (no separate VAE / text encoder; the model works directly in
a pixel-and-token space). On [Artificial Analysis](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights)
it's the **#1 single-model open-weight** text-to-image (Elo 1189). It takes a **rich
natural-language prompt** (Qwen3-VL text understanding), so the director crafts a vivid
descriptive paragraph. The Dev-2604 build is **distilled**: 28-step, **CFG-off** (no negative
prompt — everything lives in the positive description). It renders at its **native 2048×2048**
(the node snaps smaller requests up), single-device on **GPU0 ~15 GB**, ~**3–4 min/image** on a
3090 (sdpa attention — flash-attn isn't built for sm_86). That's the heaviest + slowest of the
image lanes — the trade for top quality. It coexists with the director (~15 + 4.6 ≈ 20 GB), but
not with a concurrent video render; not subject to `image_max_edge` (it's fixed at 2048²).

> **HiDream-O1 has no native ComfyUI support** — unlike Ideogram-4. Its nodes (HiDream O1 Model
> Loader / Conditioning / Sampler) come from the third-party **`Saganaki22/HiDream_O1-ComfyUI`**
> custom node, cloned by `services/comfyui/entrypoint.sh`; weights via `download_hidream_o1.sh`.
> The node needs PyTorch ≤ 2.8.x (our ComfyUI runs 2.7.0 — fine).

### 🖼️ Ideogram-4 (design / logo / photo / art)

The **🖼️ Studio · Image** lane shares the pipe and the director, but renders a **still** on
**Ideogram-4 fp8** instead of a video. It's single-device on **GPU0** (~18.5 GB @1024²), so
it runs in **either** gpu-mode — including alongside a video render in `video-studio` (the
DiT's weights sit on GPU1, GPU0 has room for the image + director). **No mode switch is
needed to make an image.**

**The director crafts a JSON caption, not prose.** Ideogram-4 is trained on **structured
JSON captions** (a `high_level_description`, a `style_description` block, and a
`compositional_deconstruction` with background + per-object elements). Hand it off-schema
plain text and it denoises to a gray **"Image blocked by safety filter"** placeholder — its
built-in fallback, *not* a real safety judgement (it fires on a plain "a red apple"). So the
image director outputs the JSON caption; the pipe validates it and falls back to wrapping
your text in a minimal caption if needed. Measured on this rig: plain text → 100% blocked;
the same prompt as a JSON caption → clean render (~80 s warm @1024²).

> **Open WebUI's native 🖼️ image button works too — via the image shim.** It used to hit the
> same trap (plain text → "blocked" placeholder). Now OWUI's `COMFYUI_BASE_URL` points at the
> **studio-image-shim** (`:8191`), a transparent ComfyUI reverse-proxy that asks the director
> to craft the JSON caption on `POST /prompt` before forwarding. See *Native image button* below.

The lane is **category-aware**: the director infers logo / poster / UI-mockup / photo /
illustration and fills the JSON with the levers that matter (logos → vector/flat/negative
space/1–2 colours; photos → camera + lens, depth of field; etc.). Want visible text/lettering?
Ask for it in quotes. Refine the same way as video — *"monochrome"*, *"tighter crop"*, *"flat
vector style"* — it evolves the prior caption. Defaults 1024×1024, 20 steps; the long edge is
capped at `image_max_edge` (1024) so the image gen coexists with the director on GPU0 (2048²
+ director = OOM; raise the cap and stop the director for 2K stills).

### 🔓 Chroma (uncensored)

The **🔓 Studio · Image (Chroma)** lane renders on **Chroma1-HD fp8** — a Flux-based,
de-distilled, *trained-uncensored* model (~9 GB, single-device GPU0). Unlike Ideogram, Chroma
takes a **rich natural-language prompt** (no JSON), supports a **negative prompt**, and uses
**real CFG** — so the director crafts a vivid descriptive paragraph (the uncensored qwen
honours intent without sanitising) rather than a JSON caption. The encoder (`t5xxl_fp16`) and
VAE (Flux `ae.safetensors`) are shared with the Flux ecosystem (already on disk), so only the
Chroma DiT is model-specific. Defaults 1024×1024, 26 steps, cfg 3.5; same `image_max_edge`
cap. This is the **uncensored stills lane** — Ideogram remains the choice for text/logos;
Chroma for unrestricted photoreal/illustration. Validated clean on this rig (~72–80 s warm).

> **Why two lanes instead of "un-censoring Ideogram":** Ideogram-4's safety is trained into
> the weights (no abliterated variant; diffusion abliteration isn't a drop-in). The image shim
> only removes Ideogram's *false-positive* blocking of neutral prompts — genuine moderation
> stays. So uncensored stills get their own model (Chroma), exactly as Sulphur is the
> uncensored video lane. Capability is in the weights; the infra is content-neutral.

### Native image button (via the image shim)

OWUI's built-in 🖼️ image button (on a chat message) also renders Ideogram-4 stills — but it
sends **plain text** to the image engine, which trips the same "blocked by safety filter"
placeholder, and OWUI's own image-prompt-generation can't help (it returns `{"prompt":"<string>"}`
and nesting the Ideogram JSON inside that string defeats the task models — escaping fails).

The fix is **`services/studio/image-shim/`** (`:8191`): a transparent **ComfyUI reverse-proxy**.
OWUI's `COMFYUI_BASE_URL` points at it (`imagegen.env`), with OWUI's image-prompt-generation
turned **off**. The shim proxies every ComfyUI call (incl. the `/ws` progress socket) straight
through — *except* `POST /prompt`, where it reads the plain-text prompt node, asks the director
(qwen `:8090`) for a rich Ideogram-4 JSON caption, and rewrites the node before forwarding. The
escaping is done in **Python** (reliable), so the model never has to produce nested JSON. Blast
radius = image generation only — title/tag/etc. task-generation is untouched (that's why it's a
ComfyUI proxy, not an OWUI task-model override).

`gpu-mode video-studio` and `image-studio` start the shim (and, in image-studio, the director it
needs). Validated on the rig: plain "a mountain landscape at sunrise" / "a serene lake" / a logo
all render clean through the shim. If the shim is down, point `COMFYUI_BASE_URL` back at `:8188`
(plain text will then hit the placeholder) or use the **Studio · Image lane**.

## VRAM / GPU split

Video and the dual-card LLMs are **mutually exclusive** (both want the GPUs). In video
mode: GPU1 holds the 22B DiT weights (~22 GB, DisTorch donor); GPU0 does compute (~7–14 GB)
**and** hosts the ~4 GB director — they coexist comfortably on one card. The **image lane**
also renders on GPU0 (~18.5 GB @1024² + the ~4 GB director ≈ 23 GB — fits; 2048² would OOM
with the director resident). Because ComfyUI holds both cards in `video-studio`, you can do
**video and ≤1024² image in the same mode with no switch** — only `image-studio`'s
gemma-12b chat or a 2048² still needs a `gpu-mode` change.

## Models (obtain separately → `/mnt/models/comfyui/models/...`)

| File | ComfyUI dir | Lane |
|---|---|---|
| `ltx-2.3-22b-distilled-1.1-Q8_0.gguf` | `unet/ltx2.3/distilled-1.1/` | LTX |
| `sulphur-2/sulphur_dev-Q8_0.gguf` | `unet/` | Sulphur |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | `loras/` | both (single-stage splice) |
| `ltx-2.3-22b-{distilled,dev}_{audio,video}_vae.safetensors` | `vae/` | LTX / Sulphur |
| `ltx-2.3-22b-{distilled,dev}_embeddings_connectors.safetensors` | `text_encoders/` | LTX / Sulphur |
| `ideogram4_fp8_scaled.safetensors` (+ `_unconditional_`), `qwen3vl_8b_fp8_scaled`, `flux2-vae` | `diffusion_models/`, `text_encoders/`, `vae/` | Ideogram-4 image |
| `Chroma1-HD-fp8mixed.safetensors` (Comfy-Org/Chroma1-HD_repackaged) | `diffusion_models/` | Chroma image (uncensored) |
| `HiDream-O1-Image-Dev-2604-FP8/` (drbaph; complete folder) | `diffusion_models/` | HiDream-O1 image (top-quality) — needs `HiDream_O1-ComfyUI` node |
| `t5xxl_fp16.safetensors` + Flux `ae.safetensors` | `text_encoders/`, `vae/flux/` | Chroma (shared with Flux ecosystem) |

Director GGUF (`Qwen3.5-4B-Uncensored-…`) → `/mnt/models/huggingface/qwen3.5-4b-gguf/…`.

Narration TTS (CPU): `kokoro-v1.0.onnx` + `voices-v1.0.bin` (kokoro-onnx GitHub release / [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)) → `/mnt/models/comfyui/models/tts/kokoro/`.

Music: `ace_step_v1_3.5b.safetensors` (ACE-Step v1 3.5B) → `models/checkpoints/ace-step-1.5/all_in_one/`.

SFX: `stable-audio-open-1.0.safetensors` (Comfy-Org repackaged) → `models/checkpoints/`; `t5-base.safetensors` → `models/text_encoders/`.

## On the uncensored models

The **Sulphur** DiT and the **director** are uncensored fine-tunes — chosen so the lane
doesn't refuse or sanitize creative prompts. That capability lives in the model weights;
the infrastructure here is content-neutral. To craft prompts through an **aligned** model
instead, point the pipe's `chat_model`/`chat_url` valves at e.g. gemma-4-12b — the Sulphur
DiT still renders uncensored, only the prompt-writing changes. (The text encoder is the
**stock** aligned gemma; for LTX it's not a meaningful censorship lever, so it's not
abliterated.)

## Follow-ups (not yet built)

- **Native temporal-extend** for smoother joins on fast-motion scenes (vs last-frame I2V).
- **Image→video long clips**: chaining currently starts from text (seg 1 = t2v); extend an
  attached image past 15 s is future.
- ~~**Fix Open WebUI's native 🖼️ image button**~~ ✅ done — the **image shim** (`:8191`) crafts
  the Ideogram JSON via the director on `POST /prompt` (see *Native image button*).
- **Uncensored stills**: Ideogram-4 is safety-trained (and the lane crafts to its schema), so
  the image lane is *aligned*. Uncensored *motion* is covered by the Sulphur video lane;
  uncensored *stills* would need a different image model (e.g. a `frames=1` render on an
  uncensored DiT) — not wired.
- Audio cross-fade at segment joins; a richer gallery (thumbnail grid vs file listing).
