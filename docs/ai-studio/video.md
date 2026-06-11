# Video Studio — chat-driven text/image → video on 2× 3090

The video side of [Club 3090 AI Studio](README.md). Type a rough idea in Open WebUI; a "director"
LLM crafts it into a professional prompt; ComfyUI renders it. Two video lanes share the pipe + the
director: **LTX-2.3** (video+audio) and **Sulphur** (an uncensored LTX-2.3 fine-tune). Video is
GPU-mutually-exclusive with the dual-card LLMs (it wants both 3090s).

> Sibling docs: **[image.md](image.md)** (HiDream-O1 / Ideogram-4 / Chroma stills) ·
> **[audio.md](audio.md)** (voices / music / SFX) · **[README.md](README.md)** (the overview +
> the full 8-lane matrix + shared services).

---

## Architecture

```
                          Browser
                             │  "a 40-second drone shot over a coastline"
                             ▼
              ┌──────────────────────────────────────────────┐
              │  Open WebUI   :8080   (the front-end)         │
              │  video lanes: 🎬 LTX-2.3 · 🔓 Sulphur          │
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

- **Studio pipe** (`services/studio/build_studio_pipe.py` → `studio_pipe.py`): the OWUI Function.
  Video lanes (LTX, Sulphur) × two modes (text→video, image→video, auto-detected from whether you
  attach an image). One director + one gallery across **all** lanes (image/audio too — see README).
- **Director** (`services/studio/enhancer/`): a small uncensored LLM that turns a casual line into
  a cinematic spec. Optional — falls back to your raw prompt if it's down.
- **ComfyUI** (`services/comfyui/`): the renderer. The 22B DiT is split across both cards by
  `UnetLoaderGGUFDisTorch2MultiGPU` (compute on GPU0, weights donated from GPU1).
- **Gallery** (`services/studio/gallery/`): always-on nginx serving ComfyUI's output dir, so media
  + links stay alive even when ComfyUI is stopped.

## Quickstart

No one-shot installer yet (the models are large + sourced separately) — three steps:

**1. Get the models.** Diffusion weights → `/mnt/models/comfyui/models/...` (see the **Models**
manifest below); the director GGUF → `/mnt/models/huggingface/qwen3.5-4b-gguf/...`.

**2. Bring the stack up:**

```bash
bash scripts/gpu-mode.sh video-studio
```

Stops the GPU LLMs and starts ComfyUI (both cards) + director (`:8090`) + gallery (`:8189`) +
orchestrator (`:8190`) + Open WebUI. `gpu-mode off` (or any LLM mode) tears the video model down
again — it's GPU-mutex with the dual-card LLMs.

**3. Install the pipe into Open WebUI** (once):

```bash
python3 services/studio/build_studio_pipe.py     # writes services/studio/studio_pipe.py
```

In Open WebUI → **Admin → Functions → +**, paste `services/studio/studio_pipe.py`, save, enable.
The Studio lanes appear in the model picker (see [README.md](README.md) for the full set). Set the
pipe's **`browser_base`** valve to your host's LAN IP (`http://<your-host>:8189`) so the returned
links open from your browser. Then open **Open WebUI** → `http://<your-host>:8080`.

### First run

1. **Create your account** — the first signup becomes admin (no hardcoded secret; Open WebUI
   generates its own per deployment).
2. **Pick a video lane** in the model selector — 🎬 LTX-2.3 (video + audio) or 🔓 Sulphur.
3. **Type a scene** — *"a fox padding through a neon alley at night"* — and send. The director
   crafts a cinematic prompt and it renders; you get a ▶️ link to the clip.
4. **Refine** by replying (*"more moody"*, *"make it night"*); for a **long clip**, include a
   duration (*"a 40-second…"*) and it auto-chains segments into one combined video.

> First render after a cold ComfyUI takes a few minutes (loads the 22B DiT + first-boot node deps).
> A 10 s clip is ~2.5 min warm; longer clips scale ~linearly per segment.

## The UX: craft-and-go, refine anytime (no approval gate)

1. **You** type something light — *"a fox in the city"*.
2. **The director** rewrites it into a full cinematographer's prompt (subject + action,
   camera/lens/movement, lighting + time of day, palette + mood, ambient sound) and it **renders
   immediately** — no "confirm?" step. The crafted prompt is shown above the video.
3. **Refine** by just replying with the change — *"more moody"*, *"make it night"*, *"slower
   camera"*. The pipe carries the previous prompt forward and the director **evolves** it. Or type
   a brand-new idea and it starts fresh — the director decides which.

Attach an **image** instead of (or with) text → it auto-routes to the **image→video** lane
(animates your still).

## How to prompt

**You don't write the cinematic prompt — the director does.** Give it the *intent* in a line or
two; it fills in camera, lens, lighting, palette, mood, and ambient sound. If you *do* want
specific control, just name it and the director keeps it — e.g. *"…top-down drone shot, golden
hour, melancholic"*.

- **Length** — put a duration in the message: *"a **30-second** timelapse…"*, *"make it **1
  minute**"*. No duration → ~10 s. Over 15 s auto-chains ~10 s segments into one clip (capped
  ~120 s); each segment adds ~2.5 min of render time.
- **Lane** — pick the model: **🎬 LTX-2.3** (video + audio) or **🔓 Sulphur** (uncensored).
- **Image → video** — attach an image (optionally with a motion note like *"slow zoom in, leaves
  drifting"*); it animates your still.
- **Voiceover** — add *"voiceover: …"* / *"narration: '…'"* / *"say: …"* and a Kokoro voice is mixed
  over the clip (ducked under the ambient, normalized). Details in [audio.md](audio.md).
- **Refine** — just reply with the change. It evolves the last prompt; a brand-new idea starts fresh.

**Works best:** one clear subject + one continuous camera move or action; slow / cinematic /
ambient scenes; a defined mood or time of day.

**Weaker / avoid:** fast or chaotic action (especially across long-clip segment joins — a cut has
no motion carry-over); lots of on-screen **text or logos**; many distinct subjects or hard
scene-cuts inside one segment; exact object counts.

**Examples**
- *"a hummingbird at a red flower, macro, soft morning light"* → a clean ~10 s macro shot.
- *"a 40-second drone flight over a foggy coastline at dawn, slow push forward"* → 4 chained
  segments → one combined ~40 s clip.
- then *"make it stormy, darker"* → re-crafts from that and regenerates.

## What it can generate

| | |
|---|---|
| **Modes** | text→video, image→video (attach an image) |
| **Audio** | yes — LTX-2.3 generates synced ambient audio; optional Kokoro voiceover ([audio.md](audio.md)) |
| **Resolution** | Sulphur 1280×720 · LTX 768×512 (set in the workflow) |
| **Length** | default ~10 s; see the ceiling below |
| **Video lanes** | `🎬 LTX-2.3` (video+audio) · `🔓 Sulphur` (uncensored) — image/audio lanes in [image.md](image.md) / [audio.md](audio.md), full matrix in [README.md](README.md) |

### Length ceiling (measured on 2× 3090, 1280×720, frames = 24·seconds + 1)

A frame sweep on the single-stage Sulphur lane:

| Frames | Length | Result |
|--:|--:|---|
| 121 | ~5 s | crisp |
| 241 | ~10 s | **crisp — the default** |
| 361 | ~15 s | coherent end-to-end, but visibly lower-energy/softer |
| 481 | ~20 s | **collapses** — near-uniform/garbage frames the whole clip |

So the pipe **defaults to 241 (10 s)** and is **hard-capped at 361 (15 s)**: a 20 s single-pass
silently corrupts (returns with no error, just unusable frames), so the cap prevents hitting it by
accident. **VRAM is not the limiter** — the weights sit on GPU1 (~22 GB, fixed); longer clips only
grow GPU0's latent (peaks ~14 GB). The wall is model coherence, not memory. Wall time scales
~linearly (~2.5 min at 10 s, ~6.5 min at 15 s).

> Past ~15 s you **extend/chunk**: render segments ≤15 s, condition each on the previous segment's
> last frame, concatenate into one clip — see *Longer videos* below.

## Longer videos (60 s+)

Past the ~15 s single-pass ceiling, the studio **chains segments**: segment 1 is text→video; each
later segment is image→video conditioned on the **previous segment's last frame**; all are
ffmpeg-concatenated into one clip. Validated on 2× 3090 — the joins are **visually seamless** (the
last-frame conditioning carries the scene across each cut). Caveat: a single frame has no
*velocity*, so **fast action** can show a brief motion reset at a cut; slow/ambient scenes are
clean (native LTX temporal-extend would smooth fast cuts — future).

**In chat (default):** just ask for a length — *"a 40-second drone shot over a coastline"*. The
pipe parses the duration, the director crafts the prompt, and the **orchestrator**
(`services/studio/orchestrator/`, `:8190`) chains `ceil(seconds/10)` ~10 s segments and returns
**one combined video** (with live "segment k/N" progress). Capped at `max_seconds` (default 120 s
= 12 segments; each ~2.5 min to render). If the orchestrator is down, it falls back to a single
capped clip.

**CLI (host):** the same chain is also a standalone tool —
`python3 services/studio/extend_chain.py "<prompt>" <n_segments> <frames_per_seg>`.

> Why a separate orchestrator: the OWUI pipe can't run ffmpeg or read the output dir, so the
> segment chaining + concat live in a tiny host-side service (ffmpeg + output access, no GPU).

### The single-stage rule

Sulphur is a fine-tune of LTX-2.3-**dev**. The "official" dev recipe is 2-stage (a spatial
upscaler + a refine pass). On this hardware that 2-stage path renders a **diamond-lattice mesh**
over every frame. The fix — and what the pipe ships — is **single-stage**: splice the distilled
LoRA onto the base sampler, 8 steps, cfg 1, no upscaler. Clean output. The workflow
(`workflows/ltx_distilled_distorch.json`) already encodes this.

## VRAM / GPU split

Video and the dual-card LLMs are **mutually exclusive** (both want the GPUs). In video mode: GPU1
holds the 22B DiT weights (~22 GB, DisTorch donor); GPU0 does compute (~7–14 GB) **and** hosts the
~4 GB director — they coexist on one card. Because ComfyUI holds both cards in `video-studio`, you
can also run a ≤1024² **image** lane in the same mode with no switch (it fits on GPU0 beside the
director). Full per-lane VRAM in [image.md](image.md) / [audio.md](audio.md) / [README.md](README.md).

## On the uncensored models

The **Sulphur** DiT and the **director** are uncensored fine-tunes — chosen so the lane doesn't
refuse or sanitize creative prompts. That capability lives in the model weights; the infrastructure
here is content-neutral. To craft prompts through an **aligned** model instead, point the pipe's
`chat_model`/`chat_url` valves at e.g. gemma-4-12b — the Sulphur DiT still renders uncensored, only
the prompt-writing changes. (The text encoder is the **stock** aligned gemma; for LTX it's not a
meaningful censorship lever, so it's not abliterated.)

## Models (video lanes — obtain separately → `/mnt/models/comfyui/models/...`)

| File | ComfyUI dir | Lane |
|---|---|---|
| `ltx-2.3-22b-distilled-1.1-Q8_0.gguf` | `unet/ltx2.3/distilled-1.1/` | LTX |
| `sulphur-2/sulphur_dev-Q8_0.gguf` | `unet/` | Sulphur |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | `loras/` | both (single-stage splice) |
| `ltx-2.3-22b-{distilled,dev}_{audio,video}_vae.safetensors` | `vae/` | LTX / Sulphur |
| `ltx-2.3-22b-{distilled,dev}_embeddings_connectors.safetensors` | `text_encoders/` | LTX / Sulphur |

Director GGUF (`Qwen3.5-4B-Uncensored-…`) → `/mnt/models/huggingface/qwen3.5-4b-gguf/…`. Image +
audio model manifests are in [image.md](image.md) / [audio.md](audio.md).

## Follow-ups (not yet built)

- **Native temporal-extend** for smoother joins on fast-motion scenes (vs last-frame I2V).
- **Image→video long clips**: chaining currently starts from text (seg 1 = t2v); extending an
  attached image past 15 s is future.
