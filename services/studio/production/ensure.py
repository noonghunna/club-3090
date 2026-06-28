"""ensure_lane — concrete admission checks per lane (NOT c3's UI-side guard).

The contract (design doc): verify services up → Step-Voice /unload before any video
lane → ComfyUI health → (the job-level lock is held by the caller). On the synthetic
backend everything is a no-op (no services to gate). The job-wide single-flight lock
lives in lock.py and is acquired once in run.py.
"""
from __future__ import annotations

import urllib.request

from . import comfy, config


class EnsureError(RuntimeError):
    pass


VIDEO_LANES = {"wan", "ltx", "sulphur", "10eros"}
COMFY_LANES = VIDEO_LANES | {"ace-step", "image"}


def _evict_voice() -> None:
    """Best-effort Step-Voice /unload before a video render (GPU1 mutex)."""
    try:
        req = urllib.request.Request(config.VOICE_URL + "/unload", data=b"",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # best-effort: the lazy service idle-unloads anyway


def ensure_lane(lane: str, *, backend: str) -> None:
    """Prepare for a render on `lane`. Raises EnsureError if a needed service is down."""
    if backend == "synthetic":
        return  # nothing to gate offline
    if lane in VIDEO_LANES:
        _evict_voice()
    if lane in COMFY_LANES:
        if not comfy.alive():
            raise EnsureError(
                f"ComfyUI not reachable at {config.COMFYUI_URL} — is the ai-studio scene up?"
            )


def ensure_tts(*, backend: str) -> None:
    if backend == "synthetic":
        return
    if not comfy.alive(config.TTS_URL, "/tts/health"):
        raise EnsureError(f"Kokoro TTS not reachable at {config.TTS_URL}")
