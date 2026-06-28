"""ffprobe-backed validators — transport-success != real success.

A rendered file existing is not enough (Ideogram placeholders, LTX collapse, audio
truncation all "succeed" at the file layer). These check the actual media: duration,
audio presence, dimensions, non-emptiness. v0a wires the cheap subset; richer checks
(placeholder / near-uniform-frame detection) are v1.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass


@dataclass
class ProbeResult:
    duration: float = 0.0
    has_audio: bool = False
    width: int = 0
    height: int = 0
    size_bytes: int = 0


def ffprobe(path: str) -> ProbeResult:
    """Probe a media file. Returns zeros if the file is missing/unreadable."""
    r = ProbeResult()
    if not (path and os.path.isfile(path)):
        return r
    r.size_bytes = os.path.getsize(path)
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,width,height",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        ).stdout
        data = json.loads(out or "{}")
    except Exception:
        return r
    try:
        r.duration = float(data.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        r.duration = 0.0
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio":
            r.has_audio = True
        if s.get("codec_type") == "video":
            r.width = int(s.get("width") or 0)
            r.height = int(s.get("height") or 0)
    return r


# -- individual checks: (name) -> (ok: bool, detail: str) ---------------------

def check(name: str, path: str, *, target_seconds: float = 0.0, tol: float = 1.5) -> tuple[bool, str]:
    p = ffprobe(path)
    if name == "non_empty":
        ok = p.size_bytes > 0
        return ok, f"{p.size_bytes} bytes"
    if name == "duration":
        if target_seconds <= 0:
            return p.duration > 0, f"{p.duration:.2f}s"
        ok = abs(p.duration - target_seconds) <= tol
        return ok, f"{p.duration:.2f}s vs target {target_seconds:.2f}s (±{tol})"
    if name == "no_audio_expected":
        return (not p.has_audio), ("has audio!" if p.has_audio else "silent (ok)")
    if name == "audio_present":
        return p.has_audio, ("audio present" if p.has_audio else "NO audio track")
    if name == "has_video":
        ok = p.width > 0 and p.height > 0
        return ok, f"{p.width}x{p.height}"
    return False, f"unknown validator {name!r}"


def run_all(names: list[str], path: str, *, target_seconds: float = 0.0) -> dict:
    """Run a list of validators against one artifact. Returns a result dict."""
    results = {}
    ok_all = True
    for n in names:
        ok, detail = check(n, path, target_seconds=target_seconds)
        results[n] = {"ok": ok, "detail": detail}
        ok_all = ok_all and ok
    return {"ok": ok_all, "checks": results}
