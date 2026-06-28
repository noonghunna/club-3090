"""Lane backends — `live` (real studio services) and `synthetic` (ffmpeg lavfi).

Backend methods take an `out_stem` (no extension) and return the ACTUAL path
written (extension chosen by the lane), so the executor stays agnostic to whether
a clip came from ComfyUI (.mp4), ACE (.mp3), Kokoro (.wav), or a synthetic source.

The synthetic backend exists so the WHOLE executor → validators → assemble →
manifest path runs offline (no GPU / ComfyUI / TTS) with real, ffprobe-able media —
that's how v0a is verified before it ever touches the rig.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from abc import ABC, abstractmethod

from . import comfy, config
from .util import sh


def _load_workflow(name: str) -> dict:
    path = os.path.join(config.WORKFLOW_DIR, name)
    with open(path) as f:
        return json.load(f)


# image lane -> ComfyUI workflow (continuity keyframes / reference stills, v0b-images).
# Only PROSE-prompt lanes belong here (keyframe prompts are prose): chroma/zimage/krea
# take T5 prose, hidream takes natural-language. Ideogram needs structured JSON → it's a
# title-card/design lane, NOT a continuity keyframe lane (see stack.KEYFRAME_LANES).
_IMAGE_WORKFLOWS = {
    "chroma": "chroma1_hd.json",
    "zimage": "z_image_turbo.json",
    "krea": "krea2.json",
    "hidream": "hidream_o1.json",
}


def _patch_chroma(wf: dict, prompt: str, width: int, height: int, seed: int) -> None:
    wf["pos"]["inputs"]["text"] = prompt
    wf["latent"]["inputs"]["width"] = width
    wf["latent"]["inputs"]["height"] = height
    wf["noise"]["inputs"]["noise_seed"] = seed       # custom sampler graph (noise node)


def _patch_ksampler(wf: dict, prompt: str, width: int, height: int, seed: int) -> None:
    wf["pos"]["inputs"]["text"] = prompt              # z-image / krea — standard ksampler
    wf["latent"]["inputs"]["width"] = width
    wf["latent"]["inputs"]["height"] = height
    wf["ksampler"]["inputs"]["seed"] = seed


def _hidream_dims(width: int, height: int) -> tuple[int, int]:
    """HiDream-O1's sampler requires each side >= 512, step 32 (it then snaps to its
    nearest patch-aligned resolution). Delivery dims (e.g. 832x480) have a sub-512
    side → 400. Scale up so the SHORT side reaches 512, snap both to /32, keeping the
    landscape aspect (the Wan i2v resize node downsizes the keyframe to the clip dims)."""
    scale = max(1.0, 512.0 / max(1, min(width, height)))
    w = max(512, int(round(width * scale / 32) * 32))
    h = max(512, int(round(height * scale / 32) * 32))
    return w, h


def _patch_hidream(wf: dict, prompt: str, width: int, height: int, seed: int) -> None:
    w, h = _hidream_dims(width, height)
    wf["cond"]["inputs"]["prompt"] = prompt           # HiDream-O1: prompt on cond, size+seed on sampler
    wf["sampler"]["inputs"]["width"] = w              # clamped to the node's >=512 /32 range; i2v resizes
    wf["sampler"]["inputs"]["height"] = h
    wf["sampler"]["inputs"]["seed"] = seed


# lane -> the node-patch that injects prompt/size/seed for that workflow graph.
_IMAGE_PATCH = {
    "chroma": _patch_chroma, "zimage": _patch_ksampler,
    "krea": _patch_ksampler, "hidream": _patch_hidream,
}


def _pull(src_host: str, dst: str, *, container: str, container_path: str) -> str:
    """Copy a service-written artifact into the production tree.

    Tries a plain host copy first (ComfyUI writes 0666). If the service wrote the
    file root-only (Kokoro TTS writes mode 0600), fall back to `docker cp` from the
    owning container — the docker daemon reads it as root. Either way the result is
    host-owned + 0644 so downstream ffmpeg can read it. Surfaced live on 2026-06-28;
    the cleaner systemic fix is for the TTS service to write 0644 (a v0b follow-up).
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.copyfile(src_host, dst)
    except PermissionError:
        subprocess.run(["docker", "cp", f"{container}:{container_path}", dst],
                       check=True, capture_output=True, text=True)
    try:
        os.chmod(dst, 0o644)
    except OSError:
        pass
    return dst


class StudioBackend(ABC):
    name = "abstract"

    @abstractmethod
    def render_video(self, *, prompt: str, width: int, height: int, frames: int,
                     steps: int, seed: int, fps: int, out_stem: str,
                     mode: str = "t2v", start_image: str | None = None) -> str: ...

    @abstractmethod
    def generate_image(self, *, prompt: str, width: int, height: int, seed: int,
                       lane: str, out_stem: str) -> str: ...

    @abstractmethod
    def make_music(self, *, tags: str, lyrics: str, seconds: float, steps: int,
                   cfg: float, seed: int, out_stem: str) -> str: ...

    @abstractmethod
    def tts(self, *, text: str, voice: str, speed: float, out_stem: str) -> str: ...


# --------------------------------------------------------------------------- live
class LiveBackend(StudioBackend):
    """Drives the real ComfyUI (:8188) + Kokoro TTS (:8192)."""
    name = "live"

    def _comfy_to(self, fn: str, sub: str, out_stem: str) -> str:
        src = os.path.join(config.OUTPUT_ROOT, sub, fn)
        ext = os.path.splitext(fn)[1] or ".bin"
        cpath = "/output/" + (f"{sub}/{fn}" if sub else fn)
        return _pull(src, out_stem + ext, container=config.COMFY_CONTAINER, container_path=cpath)

    def _upload_image(self, path: str) -> str:
        """Upload a host image to ComfyUI (/upload/image); return its name."""
        with open(path, "rb") as f:
            raw = f.read()
        ext = (os.path.splitext(path)[1].lstrip(".") or "png")
        fname = "studio_prod_input." + ext
        bnd = "----studioprodboundary7e3"
        body = (b"--" + bnd.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="image"; filename="' + fname.encode() + b'"\r\n'
                b"Content-Type: image/" + ext.encode() + b"\r\n\r\n" + raw + b"\r\n"
                b"--" + bnd.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
                b"--" + bnd.encode() + b"--\r\n")
        req = urllib.request.Request(config.COMFYUI_URL + "/upload/image", data=body,
                                     headers={"Content-Type": "multipart/form-data; boundary=" + bnd})
        return json.load(urllib.request.urlopen(req, timeout=60)).get("name", fname)

    def render_video(self, *, prompt, width, height, frames, steps, seed, fps, out_stem,
                     mode="t2v", start_image=None) -> str:
        if mode == "i2v":
            if not start_image:
                raise RuntimeError("i2v render requires a start_image")
            wf = _load_workflow("wan22_rapid_i2v.json")
            wf["loadimage"]["inputs"]["image"] = self._upload_image(start_image)
            wf["resize"]["inputs"]["width"] = width
            wf["resize"]["inputs"]["height"] = height
            wf["pos"]["inputs"]["text"] = prompt
            wf["i2v"]["inputs"]["width"] = width
            wf["i2v"]["inputs"]["height"] = height
            wf["i2v"]["inputs"]["length"] = frames
            wf["ksampler"]["inputs"]["steps"] = steps
            wf["ksampler"]["inputs"]["seed"] = seed
            wf["video"]["inputs"]["fps"] = float(fps)
        else:
            wf = _load_workflow("wan22_rapid.json")
            wf["pos"]["inputs"]["text"] = prompt
            wf["latent"]["inputs"]["width"] = width
            wf["latent"]["inputs"]["height"] = height
            wf["latent"]["inputs"]["length"] = frames
            wf["ksampler"]["inputs"]["steps"] = steps
            wf["ksampler"]["inputs"]["seed"] = seed
            wf["video"]["inputs"]["fps"] = float(fps)
        fn, sub = comfy.await_output(comfy.submit(wf), "video")
        return self._comfy_to(fn, sub, out_stem)

    def generate_image(self, *, prompt, width, height, seed, lane, out_stem) -> str:
        wf_name = _IMAGE_WORKFLOWS.get(lane)
        patch = _IMAGE_PATCH.get(lane)
        if not wf_name or not patch:
            raise RuntimeError(
                f"no production image workflow for keyframe lane {lane!r} "
                f"(have {sorted(_IMAGE_WORKFLOWS)})"
            )
        wf = _load_workflow(wf_name)
        patch(wf, prompt, width, height, seed)
        fn, sub = comfy.await_output(comfy.submit(wf), "image")
        return self._comfy_to(fn, sub, out_stem)

    def make_music(self, *, tags, lyrics, seconds, steps, cfg, seed, out_stem) -> str:
        wf = _load_workflow("ace_step_music.json")
        wf["pos"]["inputs"]["tags"] = tags
        wf["pos"]["inputs"]["lyrics"] = lyrics or "[instrumental]"
        wf["latent"]["inputs"]["seconds"] = float(seconds)
        wf["ksampler"]["inputs"]["steps"] = steps
        wf["ksampler"]["inputs"]["cfg"] = cfg
        wf["ksampler"]["inputs"]["seed"] = seed
        fn, sub = comfy.await_output(comfy.submit(wf), "audio")
        return self._comfy_to(fn, sub, out_stem)

    def tts(self, *, text, voice, speed, out_stem) -> str:
        body = {"text": text}
        if voice:
            body["voice"] = voice
        if speed:
            body["speed"] = speed
        req = urllib.request.Request(
            config.TTS_URL + "/tts", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        r = json.load(urllib.request.urlopen(req, timeout=config.RENDER_TIMEOUT))
        name = r.get("wav")
        if not name:
            raise RuntimeError(f"TTS returned no wav: {r}")
        src = os.path.join(config.OUTPUT_ROOT, name)
        return _pull(src, out_stem + ".wav",
                     container=config.TTS_CONTAINER, container_path="/output/" + name)


# ---------------------------------------------------------------------- synthetic
class SyntheticBackend(StudioBackend):
    """ffmpeg-lavfi stand-ins: real media, real durations — no GPU/services.

    Clips are silent solid-colour video (so `no_audio_expected` holds), narration is
    a sine WAV sized to word-count (so VO-timing logic is exercised), music is a low
    sine bed. Lets the full pipeline run + self-test offline.
    """
    name = "synthetic"

    def render_video(self, *, prompt, width, height, frames, steps, seed, fps, out_stem,
                     mode="t2v", start_image=None) -> str:
        dur = max(0.1, frames / float(fps or 16))
        dst = out_stem + ".mp4"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if mode == "i2v" and start_image:
            # synthetic i2v: a clip that literally BEGINS from the start frame (a
            # static hold), so chaining/hero continuity is visibly continuous offline
            # too — and the asset-DAG ordering is exercised for real.
            sh(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", start_image,
                "-t", f"{dur:.3f}", "-r", str(fps),
                "-vf", f"scale={width}:{height},format=yuv420p",
                "-c:v", "libx264", "-an", dst])
        else:
            colour = "0x%06x" % (abs(hash((prompt, seed))) % 0xFFFFFF)
            sh(["ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", f"color=c={colour}:s={width}x{height}:r={fps}",
                "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", dst])
        return dst

    def generate_image(self, *, prompt, width, height, seed, lane, out_stem) -> str:
        colour = "0x%06x" % (abs(hash((prompt, seed, "img"))) % 0xFFFFFF)
        dst = out_stem + ".png"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sh(["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c={colour}:s={width}x{height}", "-frames:v", "1", dst])
        return dst

    def make_music(self, *, tags, lyrics, seconds, steps, cfg, seed, out_stem) -> str:
        dst = out_stem + ".wav"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sh(["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"sine=frequency=180:duration={max(0.5, seconds):.3f}",
            "-c:a", "pcm_s16le", dst])
        return dst

    def tts(self, *, text, voice, speed, out_stem) -> str:
        words = max(1, len(text.split()))
        dur = max(0.6, words * 0.4 / float(speed or 1.0))
        dst = out_stem + ".wav"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sh(["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur:.3f}",
            "-c:a", "pcm_s16le", dst])
        return dst


def get_backend(name: str) -> StudioBackend:
    return {"live": LiveBackend, "synthetic": SyntheticBackend}[name]()
