"""Minimal ComfyUI client — replicated from studio_pipe.py (no OWUI deps).

studio_pipe.py is a standalone OWUI pipe that can't be imported (it relies on the
OWUI async/event context), so we re-implement the three calls we need against the
HTTP API: submit a workflow graph, poll /history for completion, and a liveness
ping. Same patterns as studio_pipe `_submit` / `_await_output` / `_alive`.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import config


class ComfyError(RuntimeError):
    pass


def alive(base: str = config.COMFYUI_URL, path: str = "/system_stats", timeout: float = 0.6) -> bool:
    """True if the backend answers at all (an HTTP error still means 'up')."""
    try:
        urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def submit(wf: dict, base: str = config.COMFYUI_URL, client_id: str = "studio-production") -> str:
    """POST a workflow graph to /prompt, return its prompt_id."""
    req = urllib.request.Request(
        base.rstrip("/") + "/prompt",
        data=json.dumps({"prompt": wf, "client_id": client_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=config.SUBMIT_TIMEOUT))
    if r.get("node_errors"):
        raise ComfyError("ComfyUI node_errors: " + json.dumps(r["node_errors"])[:400])
    pid = r.get("prompt_id")
    if not pid:
        raise ComfyError("ComfyUI returned no prompt_id")
    return pid


_EXTS = {
    "video": (".mp4", ".webm", ".mkv"),
    "image": (".png", ".jpg", ".jpeg", ".webp"),
    "audio": (".mp3", ".flac", ".wav", ".opus"),
}
_KEYS = {"video": ("gifs", "videos", "images"),
         "image": ("images",),
         "audio": ("audio", "images")}


def pick_output(outputs: dict, want: str) -> tuple[str, str] | None:
    """Pick (filename, subfolder) for `want` from a /history `outputs` dict.

    PREFERS a saved `type="output"` artifact over a `temp` preview: some nodes (e.g.
    the HiDream sampler) emit a temp .jpg preview AND a SaveImage .png to /output —
    only the latter lands at OUTPUT_ROOT, so picking the first (temp) one →
    FileNotFoundError on copy. Returns None if nothing matches.
    """
    exts, keys = _EXTS[want], _KEYS[want]
    candidates = []   # (filename, subfolder, type)
    for node in (outputs or {}).values():
        for k in keys:
            for v in node.get(k, []) or []:
                fn = str(v.get("filename", ""))
                if fn.endswith(exts) or str(v.get("format", "")).startswith(want):
                    candidates.append((fn, v.get("subfolder", ""), v.get("type", "output")))
    if not candidates:
        return None
    fn, sub, _t = next((c for c in candidates if c[2] == "output"), candidates[0])
    return fn, sub


def await_output(
    pid: str,
    want: str,
    base: str = config.COMFYUI_URL,
    timeout: float = config.RENDER_TIMEOUT,
    poll: float = config.POLL_INTERVAL,
) -> tuple[str, str]:
    """Poll /history/{pid} until complete; return (filename, subfolder).

    `want` is 'video' | 'image' | 'audio'. Mirrors studio_pipe._await_output's
    output-node walk (see `pick_output` for the temp-vs-output preference).
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(poll)
        try:
            h = json.load(urllib.request.urlopen(base.rstrip("/") + "/history/" + pid, timeout=30))
        except Exception:
            continue
        if pid not in h:
            continue
        st = h[pid].get("status", {})
        if st.get("status_str") == "error":
            raise ComfyError(f"ComfyUI generation error for prompt {pid}")
        if not st.get("completed"):
            continue
        picked = pick_output(h[pid].get("outputs", {}), want)
        if picked is not None:
            return picked
        raise ComfyError(f"prompt {pid} completed but produced no {want} output")
    raise TimeoutError(f"ComfyUI timed out after {timeout:.0f}s for prompt {pid}")
