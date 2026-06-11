# Audio Studio — voices · music · SFX (chat-driven, on 2× 3090)

The audio side of [Club 3090 AI Studio](README.md). Four **generate** lanes share the pipe + the
director, output `.mp3`/`.wav` into the gallery, and refine by reply like every other lane:

| Lane | Model | What | Where |
|---|---|---|---|
| `🎙️ Voice` | **Step-Audio-EditX** (3B, Apache) | premium **zero-shot voice clone** + emotion/style **editing** | isolated `step-voice` service `:8193` (GPU, on-demand) |
| *(narration)* | **Kokoro-82M** (Apache) | voiceover **mixed onto a video** clip | `studio-tts` service `:8192` (CPU) |
| `🎵 Music` | **ACE-Step v1 3.5B** | songs + instrumentals | ComfyUI lane, GPU0 |
| `🔊 SFX` | **Stable Audio Open 1.0** | sound effects / ambience / textures | ComfyUI lane, GPU0 |

> **Two pillars.** This doc covers the **Generate** pillar (synthesize audio). The mirror
> **Understand** pillar — transcribe (Whisper), diarize who-spoke-when (pyannote/WhisperX),
> separate overlapping voices (SepFormer) — feeds the future realtime voice-agent (calling /
> bookings) and is tracked in the private design notes, not yet built.

---

## 🎙️ Premium voice — Step-Audio-EditX (clone + edit)

The **`🎙️ Studio · Voice`** lane is the quality voice tier: **zero-shot voice cloning** (no
training — a 10–30 s reference clip + its transcript → speak any text in that voice) plus
iterative **audio editing** (emotion, speaking style, paralinguistics, speed). Step-Audio-EditX
is **3B, Apache-2.0** — the only commercially-licensed one of the top open TTS models (Fish
S2-Pro and Higgs v3 are research/non-commercial).

- **Ask for it:** pick the 🎙️ lane and type what you want spoken. It's cloned in the
  `voice_reference` voice (a bundled sample — `Narrator.wav` / `Narrator-UK.wav` / `Pirates.wav`
  — by default). Point `voice_reference` at your own clean clip to clone **your** voice.
- **No transcript needed:** if you don't give the reference's transcript, the service
  auto-transcribes it with Whisper first.
- **Why a separate service, not a ComfyUI lane:** Step-Audio-EditX hard-pins
  **`transformers==4.53.3`** (4.54+ produces *silent audio*, per the authors), which conflicts
  with ComfyUI's transformers 5.x (HiDream needs 5.x). So it runs in its **own isolated
  container** (`services/studio/step-voice/`, `:8193`) pinned to the exact version — guaranteed
  correct, zero conflict. The pipe POSTs `/clone`; the service writes a 24 kHz `.wav` to the gallery.
- **VRAM / serving:** ~14 GB bf16 on a pinned card (GPU1 by default), **on-demand** (not
  always-on — bring it up with `docker compose -f services/studio/step-voice/docker-compose.yml
  up -d`). An AWQ-4bit build (~3–4 GB) is the future light-deploy option. Validated: ~30 s to
  load, then a clip in seconds.

> Step-Audio-EditX is **generate-only** — it clones + edits speech, it does **not** diarize or
> separate multi-speaker recordings (that's the Understand pillar).

## Integrated audio — voices for video (Kokoro)

The video lanes already render a clip *with* native ambient audio. The Studio can add a
**voiceover/narration** on top: include a directive in your message and the pipe generates a
voice and **mixes it over the clip**.

- **Ask for it:** *"a fox padding through a neon alley at night, **voiceover: the city never
  sleeps, and neither do we**"* — or `narration: "..."`, or `say: ...`. The pipe pulls the spoken
  line out (so it doesn't pollute the video prompt), renders the clip, then narrates.
- **Engine:** **`studio-tts`** (`:8192`) — **Kokoro-82M** on **CPU** (never touches the GPUs),
  so it adds no VRAM pressure and runs after the render (voice ≈ 1–2 s of compute). Pick a voice
  with the `narrate_voice` valve (`af_heart`, `am_adam`, `bf_emma`, …).
- **Layer-aware mixdown** (ffmpeg): the Kokoro voice is mixed over the clip's native audio, the
  **bed is ducked** under the voice (`sidechaincompress`), and the master is **loudness-normalized**
  (`loudnorm`, −16 LUFS); output is capped to the clip length. The mix stage is structured to
  accept more layers (generated music / SFX) later without a rewrite.

> Kokoro is the **fast, zero-VRAM, low-latency** voice (and the right TTS for a future realtime
> voice agent); Step-Audio-EditX is the **premium / cloned / editable** voice. Two tiers, like
> the image lanes.

## 🎵 Music — ACE-Step

The **`🎵 Studio · Music`** lane generates **songs + instrumentals** from a text idea on
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
  with the director on GPU0), not a separate mutex mode. ~6–16 s for a short clip. Output is an
  `.mp3` in the gallery. Uses the bundled ACE-Step ComfyUI nodes.

## 🔊 SFX — Stable Audio

The **`🔊 Studio · SFX`** lane generates **sound effects, ambiences, and textures** from a text
description on **Stable Audio Open 1.0** — distinct from the Music lane (songs/instrumentals).

- **Ask for it:** pick the 🔊 lane and describe a sound — *"rain on a tin roof"*, *"sci-fi door
  whoosh"*, *"forest ambience with distant birds"*. Add a length (*"a 5-second…"*) or it defaults
  to ~10 s. **Capped at 47 s** (the model's max).
- **Director → sound prompt:** the director turns your idea into a concrete sound description
  (source, materials, acoustic space, motion). Refine like the rest (*"more distant"*, *"add
  reverb"*, *"heavier rain"*).
- **Single-device GPU0** — a lane like music. Output is an `.mp3` in the gallery. Reuses the
  bundled Stable Audio ComfyUI nodes (Stable Audio Open 1.0 + a T5-base encoder).

## Models (obtain separately → `/mnt/models/comfyui/models/...`)

| File | Dir | Lane |
|---|---|---|
| `kokoro-v1.0.onnx` + `voices-v1.0.bin` ([onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)) | `tts/kokoro/` | integrated voices (CPU) |
| `Step-Audio-EditX/` + `Step-Audio-Tokenizer/` (stepfun-ai) | `Step-Audio/` | premium voice (`step-voice` service) |
| `ace_step_v1_3.5b.safetensors` (ACE-Step v1 3.5B) | `checkpoints/ace-step-1.5/all_in_one/` | music |
| `stable-audio-open-1.0.safetensors` (Comfy-Org repackaged) + `t5-base.safetensors` | `checkpoints/`, `text_encoders/` | SFX |
