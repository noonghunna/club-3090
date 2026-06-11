"""Studio Step-Voice service — premium voice (clone + emotion/style editing) via Step-Audio-EditX.

Isolated from ComfyUI: runs the model on its required transformers==4.53.3 in its own container.
Mirrors the Kokoro studio-tts service — an HTTP server that writes WAVs into the shared gallery
output dir, called by the OWUI Studio pipe's premium voice lane.

Endpoints:
  GET  /health                                              -> {ok, ready}
  POST /clone  {text|target_text, reference?, prompt_text?}  -> {filename, subfolder}
       zero-shot clone: speak `text` in the reference voice. `reference` = a bundled sample name
       (e.g. "Narrator.wav"), an absolute path, or a data: URI (user-attached voice). `prompt_text`
       is the transcript of the reference; if omitted, Whisper transcribes it automatically.
  POST /edit   {audio, source_text, edit_type, edit_info, generated_text?} -> {filename, subfolder}
       re-emote / restyle existing audio (edit_type: emotion|style|speed|paralinguistic|denoise|vad).
"""
import os, sys, time, base64, tempfile, traceback
import numpy as np
import soundfile as sf
import torch
from aiohttp import web

STEP_IMPL   = os.environ.get("STEP_IMPL", "/opt/step/step_audio_impl")
MODELS_DIR  = os.environ.get("STEP_MODELS_DIR", "/models")          # contains Step-Audio-EditX/ + Step-Audio-Tokenizer/
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "/output")
VOICE_DIR   = os.environ.get("VOICE_SAMPLES_DIR", "/opt/step/voice_samples")
PORT        = int(os.environ.get("STEP_VOICE_PORT", "8193"))
DEFAULT_REF = os.environ.get("STEP_DEFAULT_VOICE", "Narrator.wav")
TOKENIZER_ID = os.environ.get("STEP_TOKENIZER_ID",
                              "dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online")

sys.path.insert(0, STEP_IMPL)
from tokenizer import StepAudioTokenizer          # noqa: E402
from tts import StepAudioTTS                       # noqa: E402
from model_loader import ModelSource               # noqa: E402

print("[step-voice] loading tokenizer (funasr paraformer)…", flush=True)
_encoder = StepAudioTokenizer(os.path.join(MODELS_DIR, "Step-Audio-Tokenizer"),
                              model_source=ModelSource.LOCAL, funasr_model_id=TOKENIZER_ID)
print("[step-voice] loading Step-Audio-EditX (3B, transformers 4.53.3)…", flush=True)
_engine = StepAudioTTS(os.path.join(MODELS_DIR, "Step-Audio-EditX"), _encoder, model_source=ModelSource.LOCAL)
print("[step-voice] ready", flush=True)

_whisper = None
def _transcribe(path):
    """Auto-transcribe a reference clip when no prompt_text is supplied (clone needs the transcript)."""
    global _whisper
    if _whisper is None:
        import whisper
        _whisper = whisper.load_model(os.environ.get("WHISPER_MODEL", "base"))
    return (_whisper.transcribe(path).get("text") or "").strip()

def _materialize(ref):
    """Resolve a reference to a filesystem path: bundled sample name | absolute path | data: URI."""
    if not ref:
        ref = DEFAULT_REF
    if isinstance(ref, str) and ref.startswith("data:"):
        raw = base64.b64decode(ref.split(",", 1)[1])
        fd, p = tempfile.mkstemp(suffix=".wav"); os.write(fd, raw); os.close(fd)
        return p, True
    if os.path.isabs(ref) and os.path.isfile(ref):
        return ref, False
    return os.path.join(VOICE_DIR, ref), False

def _save(audio, sr, prefix):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.squeeze(np.asarray(audio))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fn = "%s_%08d.wav" % (prefix, int(time.time() * 1000) % 100000000)
    sf.write(os.path.join(OUTPUT_DIR, fn), audio, int(sr))
    return fn

async def health(_req):
    return web.json_response({"ok": True, "ready": _engine is not None})

async def clone(req):
    d = await req.json()
    target = (d.get("target_text") or d.get("text") or "").strip()
    if not target:
        return web.json_response({"error": "target_text (text to speak) is required"}, status=400)
    ref_path, tmp = _materialize(d.get("reference"))
    try:
        if not os.path.isfile(ref_path):
            return web.json_response({"error": "reference voice not found: %s" % ref_path}, status=400)
        prompt_text = (d.get("prompt_text") or "").strip() or _transcribe(ref_path)
        audio, sr = await _run(_engine.clone, ref_path, prompt_text, target)
        fn = _save(audio, sr, "step_voice")
        return web.json_response({"filename": fn, "subfolder": "", "prompt_text": prompt_text})
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if tmp:
            try: os.unlink(ref_path)
            except Exception: pass

async def edit(req):
    d = await req.json()
    ref_path, tmp = _materialize(d.get("audio"))
    try:
        if not os.path.isfile(ref_path):
            return web.json_response({"error": "audio to edit not found: %s" % ref_path}, status=400)
        src_text = (d.get("source_text") or "").strip() or _transcribe(ref_path)
        edit_type = d.get("edit_type", "emotion")
        edit_info = d.get("edit_info", "")
        generated_text = d.get("generated_text", src_text)
        audio, sr = await _run(_engine.edit, ref_path, src_text, edit_type, edit_info, generated_text)
        fn = _save(audio, sr, "step_voice_edit")
        return web.json_response({"filename": fn, "subfolder": ""})
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if tmp:
            try: os.unlink(ref_path)
            except Exception: pass

async def _run(fn, *args):
    # Step inference is blocking/CPU-GPU heavy — run off the event loop.
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, fn, *args)

app = web.Application(client_max_size=64 * 1024 * 1024)
app.add_routes([web.get("/health", health), web.post("/clone", clone), web.post("/edit", edit)])
if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=PORT)
