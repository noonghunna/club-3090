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

# ComfyUI's host output root (mounted at /output in the studio containers). All
# lanes write here; we scope productions under `<root>/productions/<job_id>/`.
OUTPUT_ROOT = os.environ.get("COMFYUI_OUTPUT_DIR", "/mnt/models/comfyui/output")
PRODUCTIONS_DIR = os.path.join(OUTPUT_ROOT, "productions")

# The canonical (friendly-keyed) ComfyUI workflow graphs — shared with studio_pipe.
WORKFLOW_DIR = os.environ.get(
    "STUDIO_WORKFLOW_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflows"),
)

# Network timeouts (seconds).
SUBMIT_TIMEOUT = int(os.environ.get("STUDIO_SUBMIT_TIMEOUT", "60"))
RENDER_TIMEOUT = int(os.environ.get("STUDIO_RENDER_TIMEOUT", "1800"))  # 30 min/clip ceiling
POLL_INTERVAL = float(os.environ.get("STUDIO_POLL_INTERVAL", "2.0"))
