"""Runtime config — env-driven, host-side defaults.

The production CLI runs on the HOST (not inside the OWUI container), so the studio
backends are reached on `localhost` (studio_pipe.py uses host.docker.internal
because it runs in-container — do not copy that here).
"""
from __future__ import annotations

import os

# Studio service endpoints (host-side).
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8188").rstrip("/")
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8192").rstrip("/")
VOICE_URL = os.environ.get("VOICE_URL", "http://localhost:8193").rstrip("/")
# The director LLM (qwen3.5-4b-uncensored on llama.cpp) — the v0b planner.
DIRECTOR_URL = os.environ.get("DIRECTOR_URL", "http://localhost:8090/v1").rstrip("/")
DIRECTOR_MODEL = os.environ.get("DIRECTOR_MODEL", "qwen3.5-4b-uncensored")

# SearXNG (the rig's metasearch, JSON API) — optional web research for documentary briefs.
# Same instance OWUI uses (SEARXNG_QUERY_URL host.docker.internal:8088); host-side that's :8088.
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8088").rstrip("/")

# ComfyUI's host output root (mounted at /output in the studio containers). All
# lanes write here; we scope productions under `<root>/productions/<job_id>/`.
OUTPUT_ROOT = os.environ.get("COMFYUI_OUTPUT_DIR", "/mnt/models/comfyui/output")
PRODUCTIONS_DIR = os.path.join(OUTPUT_ROOT, "productions")

# The canonical (friendly-keyed) ComfyUI workflow graphs — shared with studio_pipe.
WORKFLOW_DIR = os.environ.get(
    "STUDIO_WORKFLOW_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflows"),
)

# Container names — used to pull artifacts a service wrote root-only (e.g. Kokoro
# TTS writes its WAV mode 0600, which the host CLI user can't read; `docker cp`
# from the container works because the docker daemon reads as root).
COMFY_CONTAINER = os.environ.get("STUDIO_COMFY_CONTAINER", "comfyui")
TTS_CONTAINER = os.environ.get("STUDIO_TTS_CONTAINER", "studio-tts")

# Network timeouts (seconds).
SUBMIT_TIMEOUT = int(os.environ.get("STUDIO_SUBMIT_TIMEOUT", "60"))
RENDER_TIMEOUT = int(os.environ.get("STUDIO_RENDER_TIMEOUT", "1800"))  # 30 min/clip ceiling
POLL_INTERVAL = float(os.environ.get("STUDIO_POLL_INTERVAL", "2.0"))
