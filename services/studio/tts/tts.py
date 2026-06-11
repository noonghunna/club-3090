#!/usr/bin/env python3
"""Studio TTS + audio-mixdown service — integrated voices for video (Phase 3a).

The Studio's video lanes (LTX/Sulphur) already render a clip *with* native ambient audio.
This service adds a **voiceover/narration** layer on top, using **Kokoro-82M** (ONNX, CPU —
never touches the GPUs), and a **layer-aware ffmpeg mixdown**: it ducks the clip's native
audio under the voice (sidechain compression) and loudness-normalizes the master.

The mixdown is structured around an ordered layer list (bed → … → voice) so that future
layers (generated music / SFX from a dedicated audio-studio) slot in without a rewrite —
only the voice layer is populated today. See docs/ai-studio/video.md "Integrated audio".

Endpoints:
  GET  /tts/health
  POST /tts      {text, voice?, speed?}                 -> {wav: <file in OUTPUT_DIR>}
  POST /narrate  {video, text, voice?, speed?, duck_db?} -> {filename, subfolder}  (muxed clip)

Env:
  KOKORO_MODEL   path to kokoro-v1.0.onnx     (default /models/kokoro/kokoro-v1.0.onnx)
  KOKORO_VOICES  path to voices-v1.0.bin      (default /models/kokoro/voices-v1.0.bin)
  OUTPUT_DIR     ComfyUI output dir (mounted) (default /output)
  DEFAULT_VOICE  Kokoro voice id              (default af_heart)
  TTS_PORT       listen port                  (default 8192)
"""
import os, json, time, uuid, asyncio, subprocess, tempfile, shutil
from aiohttp import web
import soundfile as sf
from kokoro_onnx import Kokoro

KOKORO_MODEL = os.environ.get("KOKORO_MODEL", "/models/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES = os.environ.get("KOKORO_VOICES", "/models/kokoro/voices-v1.0.bin")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output").rstrip("/")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "af_heart")
PORT = int(os.environ.get("TTS_PORT", "8192"))

_kokoro = None
def kokoro():
    global _kokoro
    if _kokoro is None:
        _kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    return _kokoro


def _synth(text, voice, speed):
    """Kokoro TTS -> a temp wav path (24 kHz mono float)."""
    samples, sr = kokoro().create(text, voice=voice or DEFAULT_VOICE, speed=float(speed or 1.0), lang="en-us")
    fd, path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)
    sf.write(path, samples, sr)
    return path


def _has_audio(video_path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                              "stream=index", "-of", "csv=p=0", video_path],
                             capture_output=True, text=True, timeout=30)
        return bool(out.stdout.strip())
    except Exception:
        return False


def _duration(video_path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", video_path], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return None


def _mixdown(video_path, voice_wav, out_path, duck_db=12):
    """Layer-aware mixdown (Phase 3a = bed + voice).

    bed   = the clip's native audio (LTX/Sulphur ambient). Ducked under the voice.
    voice = the Kokoro narration, kept at full level.
    master is loudness-normalized (EBU R128, -16 LUFS). Output audio is capped to the
    video duration. Adding music/SFX later = more inputs into the same amix + their own
    levels before the duck — the structure doesn't change.
    """
    dur = _duration(video_path)
    common = ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0"]
    if dur:
        common += ["-t", f"{dur:.3f}"]
    if _has_audio(video_path):
        # ratio set from duck_db: a deeper duck = higher ratio. threshold low so quiet
        # ambient still ducks; release ~250ms so the bed swells back between phrases.
        ratio = max(2, min(20, round(duck_db / 1.5)))
        # apad the voice so the sidechain key spans the whole clip (sidechaincompress otherwise
        # ends when the shorter voice input runs out, truncating the bed). -t caps to video len.
        fc = (
            "[1:a]aresample=24000,apad,asplit=2[vk][vm];"
            "[0:a]aresample=48000[bed];"
            f"[bed][vk]sidechaincompress=threshold=0.03:ratio={ratio}:attack=5:release=250[duck];"
            "[duck][vm]amix=inputs=2:normalize=0:dropout_transition=0[mix];"
            "[mix]loudnorm=I=-16:TP=-1.5:LRA=11[a]"
        )
    else:
        fc = "[1:a]aresample=48000,loudnorm=I=-16:TP=-1.5:LRA=11[a]"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", voice_wav, "-filter_complex", fc, "-map", "[a]"] + common + [out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg mixdown failed: " + r.stderr[-600:])


async def health(request):
    return web.json_response({"ok": True, "model": os.path.basename(KOKORO_MODEL),
                              "default_voice": DEFAULT_VOICE,
                              "model_present": os.path.exists(KOKORO_MODEL)})


async def tts(request):
    b = await request.json()
    text = (b.get("text") or "").strip()
    if not text:
        return web.json_response({"error": "no text"}, status=400)
    loop = asyncio.get_event_loop()
    wav = await loop.run_in_executor(None, _synth, text, b.get("voice"), b.get("speed"))
    name = "studio_tts_" + uuid.uuid4().hex[:8] + ".wav"
    dst = os.path.join(OUTPUT_DIR, name)
    shutil.move(wav, dst)   # /tmp -> /output is cross-device; shutil.move copies+unlinks
    return web.json_response({"wav": name})


async def narrate(request):
    b = await request.json()
    text = (b.get("text") or "").strip()
    video = b.get("video") or ""
    if not text:
        return web.json_response({"error": "no text"}, status=400)
    sub = b.get("subfolder", "") or ""
    video_path = os.path.join(OUTPUT_DIR, sub, os.path.basename(video))
    if not os.path.exists(video_path):
        return web.json_response({"error": "video not found: " + video}, status=404)
    loop = asyncio.get_event_loop()
    voice_wav = None
    try:
        voice_wav = await loop.run_in_executor(None, _synth, text, b.get("voice"), b.get("speed"))
        out_name = "studio_narrated_" + uuid.uuid4().hex[:8] + ".mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        await loop.run_in_executor(None, _mixdown, video_path, voice_wav, out_path, int(b.get("duck_db", 12)))
        return web.json_response({"filename": out_name, "subfolder": ""})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if voice_wav and os.path.exists(voice_wav):
            os.remove(voice_wav)


def make_app():
    app = web.Application(client_max_size=32 * 1024 * 1024)
    app.router.add_get("/tts/health", health)
    app.router.add_post("/tts", tts)
    app.router.add_post("/narrate", narrate)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=PORT)
