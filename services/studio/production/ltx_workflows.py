"""LTX-family (LTX-2.3 · Sulphur · 10Eros) video workflow builder — ONE recipe, two callers.

Both the interactive Studio pipe (build_studio_pipe.py, at build time) and the Production
executor (lanes.py, at render time) build these ComfyUI graphs from the same recipe: a
DiT-GGUF swap on the proven LTX-distilled template + dev VAEs/connectors + the shared
distill LoRA for the uncensored dev lanes, plus i2v node-surgery. Single source of truth so
the interactive lanes and the production executor can't drift.

Lane traits (this rig, single-stage 8-step cfg=1):
  ltx      LTX-2.3-distilled — 768×512, native synced audio, no LoRA (already distilled)
  sulphur  LTX-2.3-22B dev fine-tune (uncensored) — 1280×720, + distill LoRA
  10eros   LTX-2.3-22B dev fine-tune (uncensored) — 1280×720, + distill LoRA
All are 24 fps. (In the multi-shot production pipeline the clip's own audio is unused — the
assembler builds the track from narration + music; see assemble.py.)
"""
from __future__ import annotations

import json
import os

from . import config

TEMPLATE = os.path.join(config.WORKFLOW_DIR, "ltx_distilled_distorch.json")
LTX_FPS = 24.0
_DISTILL_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

# lane -> (dit, audio_vae, video_vae, connectors, width, height, lora, i2v_longer_edge)
LANES: dict[str, tuple] = {
    "ltx": ("ltx2.3/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q8_0.gguf",
            "ltx-2.3-22b-distilled_audio_vae.safetensors",
            "ltx-2.3-22b-distilled_video_vae.safetensors",
            "ltx-2.3-22b-distilled_embeddings_connectors.safetensors", 768, 512, None, 768),
    "sulphur": ("sulphur-2/sulphur_dev-Q8_0.gguf",
                "ltx-2.3-22b-dev_audio_vae.safetensors",
                "ltx-2.3-22b-dev_video_vae.safetensors",
                "ltx-2.3-22b-dev_embeddings_connectors.safetensors", 1280, 720, _DISTILL_LORA, 1280),
    "10eros": ("10eros/10Eros_v1-Q8_0.gguf",
               "ltx-2.3-22b-dev_audio_vae.safetensors",
               "ltx-2.3-22b-dev_video_vae.safetensors",
               "ltx-2.3-22b-dev_embeddings_connectors.safetensors", 1280, 720, _DISTILL_LORA, 1280),
}


def is_ltx_family(lane: str) -> bool:
    return lane in LANES


def _build(dit, audio_vae, video_vae, connectors, width, height, frames=121, lora=None) -> dict:
    wf = json.load(open(TEMPLATE))
    wf["3"]["inputs"]["unet_name"] = dit
    wf["1"]["inputs"]["vae_name"] = audio_vae
    wf["2"]["inputs"]["vae_name"] = video_vae
    wf["47"]["inputs"]["clip_name2"] = connectors
    wf["7"]["inputs"]["width"] = width
    wf["7"]["inputs"]["height"] = height
    wf["8"]["inputs"]["scale_by"] = 1.0          # output res == base res (no upscaler stage)
    wf["10"]["inputs"]["value"] = frames
    if lora:                                     # distill LoRA -> dev DiT runs single-stage 8-step cfg=1
        wf["50"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["3", 0], "lora_name": lora, "strength_model": 1.0}}
        wf["18"]["inputs"]["model"] = ["50", 0]  # CFGGuider reads the LoRA'd model
    return wf


def _i2v_insert(wf: dict, base_longer_edge: int) -> dict:
    """LoadImage -> resize -> LTXVPreprocess -> LTXVImgToVideoInplace, conditioning the base
    video latent (node 14) on the start image (the keyframe/continuity anchor)."""
    wf = json.loads(json.dumps(wf))
    wf["100"] = {"class_type": "LoadImage", "inputs": {"image": "__STUDIO_IMAGE__"}}
    wf["101"] = {"class_type": "ResizeImagesByLongerEdge",
                 "inputs": {"images": ["100", 0], "longer_edge": base_longer_edge}}
    wf["102"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["101", 0], "img_compression": 35}}
    wf["103"] = {"class_type": "LTXVImgToVideoInplace",
                 "inputs": {"vae": ["2", 0], "image": ["102", 0], "latent": ["14", 0],
                            "strength": 1.0, "bypass": False}}
    wf["15"]["inputs"]["video_latent"] = ["103", 0]   # rewire concat: empty -> image-conditioned latent
    return wf


def build_workflows() -> dict:
    """Every LTX-family t2v + i2v graph, keyed '<lane>-<mode>' — for the interactive pipe's WF map."""
    out = {}
    for lane, (dit, avae, vvae, conn, w, h, lora, edge) in LANES.items():
        t2v = _build(dit, avae, vvae, conn, w, h, lora=lora)
        out[lane + "-t2v"] = t2v
        out[lane + "-i2v"] = _i2v_insert(t2v, edge)
    return out


def render_graph(lane: str, *, prompt: str, seconds: float, seed: int,
                 mode: str = "t2v", image_name: str | None = None,
                 negative: str = "") -> tuple[dict, int, int]:
    """A ready-to-submit LTX-family graph for the production executor + its (width, height).

    LTX renders at its OWN native res + 24 fps (NOT the Wan delivery dims) — frame count is
    sized from the requested seconds. seed makes the shot reproducible. i2v conditions on
    `image_name` (an already-uploaded ComfyUI input).
    """
    dit, avae, vvae, conn, w, h, lora, edge = LANES[lane]
    frames = max(1, round(seconds * LTX_FPS))
    wf = _build(dit, avae, vvae, conn, w, h, frames=frames, lora=lora)
    if mode == "i2v":
        wf = _i2v_insert(wf, edge)
        if image_name:
            wf["100"]["inputs"]["image"] = image_name
    wf["5"]["inputs"]["text"] = prompt          # node 5 = positive prompt encoder
    if negative:
        wf["6"]["inputs"]["text"] = negative     # node 6 = negative prompt encoder
    wf["16"]["inputs"]["noise_seed"] = int(seed) % 2147483647
    return wf, w, h
