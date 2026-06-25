"""Studio Step-Voice service — premium voice (clone + emotion/style editing) via Step-Audio-EditX.

Isolated from ComfyUI: runs the model on its required transformers==4.53.3 in its own container.
Mirrors the Kokoro studio-tts service — an HTTP server that writes WAVs into the shared gallery
output dir, called by the OWUI Studio pipe's premium voice lane.

LAZY lifecycle (STEP_VOICE_LAZY=1, default): the model is ~14 GB on GPU and the ai-studio scene
shares GPU1 with the video DiT donor (~21.9 GB). So we DON'T load at boot — the container comes up
cheap (~0 GB), so the voice lane appears in OWUI as soon as ai-studio starts it, and the weights
load on the FIRST /clone or /edit. GPU1 is reclaimed two ways: an idle-unload timer
(STEP_VOICE_IDLE_UNLOAD_S, default 300 s) and an explicit POST /unload that the Studio pipe's video
lanes call before a render (deterministic voice⊕video mutex). Set STEP_VOICE_LAZY=0 to load at boot.

Endpoints:
  GET  /health                                              -> {ok, ready, loaded, lazy}
  POST /clone  {text|target_text, reference?, prompt_text?}  -> {filename, subfolder}
       zero-shot clone: speak `text` in the reference voice. `reference` = a bundled sample name
       (e.g. "Narrator.wav"), an absolute path, or a data: URI (user-attached voice). `prompt_text`
       is the transcript of the reference; if omitted, Whisper transcribes it automatically.
  POST /edit   {audio, source_text, edit_type, edit_info, generated_text?} -> {filename, subfolder}
       re-emote / restyle existing audio (edit_type: emotion|style|speed|paralinguistic|denoise|vad).
  POST /unload                                              -> {ok, unloaded}
       free the model from GPU (no-op if not loaded). Called before a video render to free GPU1.
"""
import os, sys, time, base64, tempfile, traceback, threading
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
LAZY          = os.environ.get("STEP_VOICE_LAZY", "1") != "0"
IDLE_UNLOAD_S = int(os.environ.get("STEP_VOICE_IDLE_UNLOAD_S", "300"))   # 0 = never idle-unload

sys.path.insert(0, STEP_IMPL)
from tokenizer import StepAudioTokenizer          # noqa: E402
from tts import StepAudioTTS                       # noqa: E402
from model_loader import ModelSource               # noqa: E402

# ── Lazy model lifecycle ───────────────────────────────────────────────────
_engine = None
_encoder = None
_load_lock = threading.Lock()      # serializes load/unload (both run in the request executor)
_last_used = 0.0

def _load():
    """Load tokenizer + Step-Audio-EditX (idempotent). Caller holds _load_lock."""
    global _engine, _encoder
    if _engine is not None:
        return _engine
    print("[step-voice] loading tokenizer (funasr paraformer)…", flush=True)
    _encoder = StepAudioTokenizer(os.path.join(MODELS_DIR, "Step-Audio-Tokenizer"),
                                  model_source=ModelSource.LOCAL, funasr_model_id=TOKENIZER_ID)
    print("[step-voice] loading Step-Audio-EditX (3B, transformers 4.53.3)…", flush=True)
    _engine = StepAudioTTS(os.path.join(MODELS_DIR, "Step-Audio-EditX"), _encoder, model_source=ModelSource.LOCAL)
    print("[step-voice] model ready", flush=True)
    return _engine

def _ensure_loaded():
    """Load on first use; refresh the idle clock. Runs inside the request executor (blocking)."""
    global _last_used
    with _load_lock:
        eng = _load()
    _last_used = time.time()
    return eng

def _unload():
    """Free the model from GPU. Returns True if something was freed."""
    global _engine, _encoder
    with _load_lock:
        if _engine is None and _encoder is None:
            return False
        _engine = None
        _encoder = None
    import gc; gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[step-voice] model unloaded (GPU freed)", flush=True)
    return True

def _idle_watch():
    """Background daemon: unload the model after IDLE_UNLOAD_S of inactivity to free GPU1."""
    if IDLE_UNLOAD_S <= 0:
        return
    tick = min(60, IDLE_UNLOAD_S)
    while True:
        time.sleep(tick)
        if _engine is not None and _last_used and (time.time() - _last_used) > IDLE_UNLOAD_S:
            print("[step-voice] idle %ds → unloading" % IDLE_UNLOAD_S, flush=True)
            _unload()

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

# ── blocking workers (run off the event loop; lazy-load the model first) ──
def _clone_sync(ref_path, prompt_text_in, target):
    eng = _ensure_loaded()
    prompt_text = prompt_text_in or _transcribe(ref_path)
    audio, sr = eng.clone(ref_path, prompt_text, target)
    return {"filename": _save(audio, sr, "step_voice"), "subfolder": "", "prompt_text": prompt_text}

def _edit_sync(ref_path, src_text_in, edit_type, edit_info, generated_text_in):
    eng = _ensure_loaded()
    src_text = src_text_in or _transcribe(ref_path)
    generated_text = generated_text_in or src_text
    audio, sr = eng.edit(ref_path, src_text, edit_type, edit_info, generated_text)
    return {"filename": _save(audio, sr, "step_voice_edit"), "subfolder": ""}

async def _run(fn, *args):
    # Step inference (and the lazy load) are blocking/CPU-GPU heavy — run off the event loop.
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, fn, *args)

async def health(_req):
    loaded = _engine is not None
    return web.json_response({"ok": True, "ready": loaded, "loaded": loaded, "lazy": LAZY})

async def clone(req):
    d = await req.json()
    target = (d.get("target_text") or d.get("text") or "").strip()
    if not target:
        return web.json_response({"error": "target_text (text to speak) is required"}, status=400)
    ref_path, tmp = _materialize(d.get("reference"))
    try:
        if not os.path.isfile(ref_path):
            return web.json_response({"error": "reference voice not found: %s" % ref_path}, status=400)
        return web.json_response(await _run(_clone_sync, ref_path, (d.get("prompt_text") or "").strip(), target))
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
        return web.json_response(await _run(
            _edit_sync, ref_path, (d.get("source_text") or "").strip(),
            d.get("edit_type", "emotion"), d.get("edit_info", ""), d.get("generated_text", "")))
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if tmp:
            try: os.unlink(ref_path)
            except Exception: pass

async def unload(_req):
    freed = await _run(_unload)
    return web.json_response({"ok": True, "unloaded": freed})

app = web.Application(client_max_size=64 * 1024 * 1024)
app.add_routes([web.get("/health", health), web.post("/clone", clone),
                web.post("/edit", edit), web.post("/unload", unload)])
if __name__ == "__main__":
    if not LAZY:
        print("[step-voice] STEP_VOICE_LAZY=0 → eager load at boot", flush=True)
        _ensure_loaded()
    else:
        print("[step-voice] lazy mode — model loads on first /clone or /edit "
              "(idle-unload %ss)" % (IDLE_UNLOAD_S or "off"), flush=True)
    threading.Thread(target=_idle_watch, daemon=True).start()
    web.run_app(app, host="0.0.0.0", port=PORT)
