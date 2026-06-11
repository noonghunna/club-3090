# Video Studio — chat-driven text/image → video on 2× 3090

Type a rough idea in Open WebUI; a "director" LLM crafts it into a professional cinematic
prompt; ComfyUI renders it to video (with audio) on **LTX-2.3**. A second lane uses
**Sulphur** (an uncensored LTX-2.3 fine-tune). Runs on the same 2× RTX 3090 box as the
rest of the stack — it's GPU-mutually-exclusive with the dual-card LLMs.

This is the **P2 / video** sibling of [IMAGE_STUDIO.md](IMAGE_STUDIO.md) (Ideogram-4 stills).

---

## Architecture

```
  You (browser)
      │  "a fox in a neon city"
      ▼
  Open WebUI  :8080  ──pick "🎬 Studio · LTX" or "🔓 Studio · Sulphur"
      │
      │ (1) Studio pipe (an OWUI Function) calls the director…
      ▼
  Director LLM  :8090   qwen3.5-4b-uncensored (llama.cpp, ~4 GB, GPU0)
      │  returns ONE rich cinematic prompt
      │ (2) …then submits a single-stage graph to ComfyUI
      ▼
  ComfyUI  :8188   LTX-2.3 / Sulphur 22B GGUF, DisTorch across BOTH 3090s
      │  renders .mp4 (video + audio) → /output/video/
      │
      │  (long clip >15s? the pipe instead POSTs to…)
      ▼
  Orchestrator :8190   chains ~10s segments (seg1 t2v → segN i2v from prev last frame),
      │                ffmpeg-concats → one clip; reports "segment k/N" (host-side, no GPU)
      ▼
  Gallery  :8189   always-on nginx over the output dir (survives ComfyUI down)
      │  ◀── the pipe returns a link here
      ▼
  You: ▶️ play / download   ·   reply "make it night" to refine
```

- **Studio pipe** (`services/studio/build_studio_pipe.py` → `studio_pipe.py`): the OWUI
  Function. Two lanes (LTX, Sulphur) × two modes (text→video, image→video, auto-detected
  from whether you attach an image).
- **Director** (`services/studio/enhancer/`): a small uncensored LLM that turns a casual
  line into a cinematic spec. Optional — falls back to your raw prompt if it's down.
- **ComfyUI** (`services/comfyui/`): the renderer. The 22B DiT is split across both cards
  by `UnetLoaderGGUFDisTorch2MultiGPU` (compute on GPU0, weights donated from GPU1).
- **Gallery** (`services/studio/gallery/`): always-on nginx serving ComfyUI's output dir,
  so media + links stay alive even when ComfyUI is stopped.

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

## What it can generate

| | |
|---|---|
| **Modes** | text→video, image→video (attach an image) |
| **Audio** | yes — LTX-2.3 generates synced ambient audio |
| **Resolution** | Sulphur 1280×720 · LTX 768×512 (set in the workflow) |
| **Length** | default ~10 s; see the ceiling below |
| **Lanes** | `🎬 LTX-2.3` (stock, video+audio) · `🔓 Sulphur` (uncensored) |

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

## VRAM / GPU split

Video and the dual-card LLMs are **mutually exclusive** (both want the GPUs). In video
mode: GPU1 holds the 22B DiT weights (~22 GB, DisTorch donor); GPU0 does compute (~7–14 GB)
**and** hosts the ~4 GB director — they coexist comfortably on one card. The 🖼️ Ideogram
image button (see IMAGE_STUDIO.md) also uses this ComfyUI but will swap the video model out
of VRAM, so alternating image/video costs a model reload each switch.

## Bring it up

```bash
bash scripts/gpu-mode.sh video-studio
```

Stops the GPU LLMs, starts ComfyUI (both cards) + the director (:8090) + the gallery
(:8189) + Open WebUI, then you generate from chat. `gpu-mode off` (or any LLM mode) tears
the video model down again. See [`services/studio/README.md`](../services/studio/README.md)
for installing the pipe into Open WebUI and per-piece startup.

## Models (obtain separately → `/mnt/models/comfyui/models/...`)

| File | ComfyUI dir | Lane |
|---|---|---|
| `ltx-2.3-22b-distilled-1.1-Q8_0.gguf` | `unet/ltx2.3/distilled-1.1/` | LTX |
| `sulphur-2/sulphur_dev-Q8_0.gguf` | `unet/` | Sulphur |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | `loras/` | both (single-stage splice) |
| `ltx-2.3-22b-{distilled,dev}_{audio,video}_vae.safetensors` | `vae/` | LTX / Sulphur |
| `ltx-2.3-22b-{distilled,dev}_embeddings_connectors.safetensors` | `text_encoders/` | LTX / Sulphur |

Director GGUF (`Qwen3.5-4B-Uncensored-…`) → `/mnt/models/huggingface/qwen3.5-4b-gguf/…`.

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
- **Uncensored stills**: a `frames=1` "image" intent on the Studio lane (the 🖼️ button uses
  Ideogram-4, which is aligned).
- Audio cross-fade at segment joins; a richer gallery (thumbnail grid vs file listing).
