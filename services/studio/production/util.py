"""Small shared helpers."""
from __future__ import annotations

import hashlib
import os
import subprocess


class FFError(RuntimeError):
    pass


def sh(cmd: list[str], timeout: int = 600) -> str:
    """Run a subprocess, raise FFError with tail of stderr on failure. Returns stdout."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise FFError(f"{cmd[0]} failed ({r.returncode}): {(r.stderr or '')[-600:]}")
    return r.stdout


def last_frame(clip_path: str, out_png: str) -> str:
    """Extract a video's final frame to a PNG (the i2v-chain hand-off)."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    n = sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of",
            "default=nokey=1:noprint_wrappers=1", clip_path]).strip()
    try:
        idx = max(0, int(n) - 1)
    except ValueError:
        idx = 0
    sh(["ffmpeg", "-y", "-v", "error", "-i", clip_path,
        "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", out_png])
    return out_png


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()
