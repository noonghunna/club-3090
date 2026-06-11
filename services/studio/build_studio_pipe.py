#!/usr/bin/env python3
"""Builds the unified OWUI Studio pipe: LTX/Sulphur lanes x T2V/I2V auto-route -> studio_pipe.py

Both lanes are SINGLE-STAGE (8-step cfg=1) built from the proven clean LTX-distilled graph.
Sulphur is a dev-based fine-tune, so the distill LoRA is spliced onto its base sampler to give
the same single-stage behaviour. The old 2-stage spatial-upscaler/refine path was DROPPED — on
2x 3090 it injected a diamond-lattice mesh artifact over every Sulphur frame (the single-stage
distilled path is clean; the 2-stage dev path is the bug). See docs/VIDEO_STUDIO.md.

Run `python3 build_studio_pipe.py` → writes studio_pipe.py next to it; install that as an
Open WebUI Function (Admin → Functions → +). See services/studio/README.md.
"""
import json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(_HERE, 'workflows', 'ltx_distilled_distorch.json')   # validated clean single-stage graph (video)
IMAGE_TEMPLATE = os.path.join(_HERE, 'workflows', 'ideogram4.json')          # validated Ideogram-4 fp8 graph (image, GPU0 single-device)
CHROMA_TEMPLATE = os.path.join(_HERE, 'workflows', 'chroma1_hd.json')         # Chroma1-HD fp8 graph (uncensored image, GPU0 single-device, natural-language prompt)
OUT_PATH = os.path.join(_HERE, 'studio_pipe.py')

def build(dit, audio_vae, video_vae, connectors, width, height, frames=121, lora=None):
    wf = json.load(open(TEMPLATE))
    wf['3']['inputs']['unet_name'] = dit
    wf['1']['inputs']['vae_name'] = audio_vae
    wf['2']['inputs']['vae_name'] = video_vae
    wf['47']['inputs']['clip_name2'] = connectors
    wf['7']['inputs']['width'] = width; wf['7']['inputs']['height'] = height
    wf['8']['inputs']['scale_by'] = 1.0          # output res == base res (no upscaler stage)
    wf['10']['inputs']['value'] = frames
    if lora:                                     # distill LoRA -> dev DiT runs single-stage 8-step cfg=1
        wf['50'] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["3", 0], "lora_name": lora, "strength_model": 1.0}}
        wf['18']['inputs']['model'] = ["50", 0]  # CFGGuider reads the LoRA'd model
    return wf

def i2v_insert(wf, base_longer_edge):
    """Insert LoadImage->resize->preprocess->ImgToVideoInplace, conditioning the base video latent (node 14) on an image."""
    wf = json.loads(json.dumps(wf))
    wf['100'] = {"class_type": "LoadImage", "inputs": {"image": "__STUDIO_IMAGE__"}}
    wf['101'] = {"class_type": "ResizeImagesByLongerEdge", "inputs": {"images": ["100", 0], "longer_edge": base_longer_edge}}
    wf['102'] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["101", 0], "img_compression": 35}}
    wf['103'] = {"class_type": "LTXVImgToVideoInplace",
                 "inputs": {"vae": ["2", 0], "image": ["102", 0], "latent": ["14", 0], "strength": 1.0, "bypass": False}}
    wf['15']['inputs']['video_latent'] = ["103", 0]   # rewire concat: empty latent -> image-conditioned latent
    return wf

LTX_DIT      = 'ltx2.3/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q8_0.gguf'
SUL_DIT      = 'sulphur-2/sulphur_dev-Q8_0.gguf'
DISTILL_LORA = 'ltx-2.3-22b-distilled-lora-384.safetensors'

# LTX: pre-distilled checkpoint, distilled VAEs/connectors, no LoRA (already single-stage). 768x512 proven (00016).
ltx_t2v = build(LTX_DIT,
                'ltx-2.3-22b-distilled_audio_vae.safetensors', 'ltx-2.3-22b-distilled_video_vae.safetensors',
                'ltx-2.3-22b-distilled_embeddings_connectors.safetensors', 768, 512)
# Sulphur: dev-based fine-tune -> dev VAEs/connectors + distill LoRA. 1280x720 single-stage proven (00030).
sul_t2v = build(SUL_DIT,
                'ltx-2.3-22b-dev_audio_vae.safetensors', 'ltx-2.3-22b-dev_video_vae.safetensors',
                'ltx-2.3-22b-dev_embeddings_connectors.safetensors', 1280, 720, lora=DISTILL_LORA)
WF = {
    "ltx-t2v": ltx_t2v,
    "ltx-i2v": i2v_insert(ltx_t2v, 768),
    "sulphur-t2v": sul_t2v,
    "sulphur-i2v": i2v_insert(sul_t2v, 1280),
    # Image lane: Ideogram-4 fp8 (DualModelGuider, native nodes). Single-device GPU0,
    # coexists with the director on GPU0 at <=1024^2 (2048^2 would OOM — capped in pipe).
    "image": json.load(open(IMAGE_TEMPLATE)),
    # Uncensored image lane: Chroma1-HD fp8 (Flux-based, de-distilled, trained uncensored).
    # Natural-language prompt + negative + real CFG (unlike Ideogram's JSON). Single-device GPU0.
    "chroma": json.load(open(CHROMA_TEMPLATE)),
}
WF_JSON = json.dumps(WF)

PIPE = r'''
"""
title: Studio (text/image -> video · image)
author: club-3090
description: Type a rough idea — the studio director (qwen) crafts it into a professional, artistic prompt and generates. Video lanes: LTX (video+audio) or Sulphur (uncensored), text->video or attach an image. Image lanes: Ideogram-4 (graphic design / logo / photo / art) or Chroma (uncensored). Refine anytime by just saying what to change.
required_open_webui_version: 0.5.0
version: 0.8.0
"""
# ── Pipeline defaults (this rig, 2x 3090, measured 2026-06-11) ──────────────────
#  Video lanes: ltx = LTX-2.3-distilled (video+audio) · sulphur = uncensored dev fine-tune
#    Render:  SINGLE-STAGE, 8-step, cfg=1 (no 2-stage upscaler — it injects mesh)
#    Res:     sulphur 1280x720 · ltx 768x512
#    Frames:  default 241 (=10s @24fps, crisp). Valve range to 361 (=15s, coherent).
#             HARD-CAPPED at 361 in _comfy — ~481/20s collapses to corrupted output.
#    VRAM:    weights on GPU1 (DisTorch donor ~21.9GB), compute on GPU0 (~14GB peak).
#  Image lanes (both single-device GPU0, run in EITHER gpu-mode; coexist w/ director ~4.6GB):
#    image  = Ideogram-4 fp8 (DualModelGuider, ~18.5GB) — STRUCTURED JSON caption; great at
#             text/logos/graphic design. SAFETY-TRAINED (blocks some content). 1024x1024, 20 steps.
#    chroma = Chroma1-HD fp8 (Flux-based, de-distilled, ~9GB) — NATURAL-LANGUAGE prompt + negative
#             + real CFG; trained UNCENSORED. The "Sulphur for stills." 1024x1024, 26 steps, cfg 3.5.
#    Both capped at image_max_edge (1024) so they coexist with the director on GPU0 (2048^2 = OOM).
#  Director: qwen3.5-4b-uncensored @ :8090 (GPU0); falls back to raw prompt if down.
# ────────────────────────────────────────────────────────────────────────────────
import json, time, base64, re, math, urllib.request, urllib.parse, asyncio
from pydantic import BaseModel, Field

WORKFLOWS = json.loads(r"""__WF_JSON__""")

class Pipe:
    class Valves(BaseModel):
        comfyui_url: str = Field(default="http://host.docker.internal:8188")
        chat_url: str = Field(default="http://host.docker.internal:8090/v1")
        chat_model: str = Field(default="qwen3.5-4b-uncensored")
        browser_base: str = Field(default="http://localhost:8189", description="Always-on media gallery (survives ComfyUI being down). Set to your host's LAN IP (e.g. http://192.168.x.x:8189) so the returned video links open from your browser.")
        enhance: bool = Field(default=True)
        timeout_s: int = Field(default=600)
        frames: int = Field(default=241, description="frames @24fps. 121=5s, 241=10s (default, crisp), 361=15s (max, coherent but softer). HARD-CAPPED at 361: ~481/20s collapses to corrupted output on this rig (measured 2026-06-11).")
        orchestrator_url: str = Field(default="http://host.docker.internal:8190", description="Studio orchestrator for long clips (>15s). Asked to chain ~10s segments into one combined video. If unreachable, long requests fall back to a single capped clip.")
        max_seconds: int = Field(default=120, description="Cap on requested long-clip length (segments = ceil(seconds/10), each ~2.5 min to render).")
        image_width: int = Field(default=1024, description="Image lane default width (Ideogram-4).")
        image_height: int = Field(default=1024, description="Image lane default height (Ideogram-4).")
        image_steps: int = Field(default=20, description="Image lane sampler steps (Ideogram-4).")
        image_max_edge: int = Field(default=1024, description="Cap on the image long edge. 1024 lets the image gen coexist with the director on GPU0 (~23GB); 2048 would OOM unless the director is stopped first.")
        chroma_steps: int = Field(default=26, description="Chroma (uncensored image lane) sampler steps.")
        chroma_cfg: float = Field(default=3.5, description="Chroma CFG scale (Chroma is de-distilled — real CFG + negative prompt, unlike Ideogram).")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [
            {"id": "ltx", "name": "\U0001F3AC Studio · LTX-2.3 (video+audio · text or image)"},
            {"id": "sulphur", "name": "\U0001F513 Studio · Sulphur (uncensored · text or image)"},
            {"id": "image", "name": "\U0001F5BC️ Studio · Image (Ideogram-4 · graphic / logo / photo / art)"},
            {"id": "chroma", "name": "\U0001F513 Studio · Image (Chroma · uncensored)"},
        ]

    def _extract_image(self, body):
        for m in reversed(body.get("messages", [])):
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        u = (part.get("image_url") or {}).get("url", "")
                        if u.startswith("data:"):
                            return u
            for u in (m.get("images") or []):
                if isinstance(u, str) and u.startswith("data:"):
                    return u
            return None
        return None

    def _upload_image(self, data_uri):
        head, b64 = data_uri.split(",", 1)
        ext = "png"
        if "image/" in head:
            ext = head.split("image/")[1].split(";")[0].split("+")[0] or "png"
        raw = base64.b64decode(b64)
        fname = "studio_input." + ext
        bnd = "----studioboundary7e3"
        body = (b"--" + bnd.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="image"; filename="' + fname.encode() + b'"\r\n'
                b"Content-Type: image/" + ext.encode() + b"\r\n\r\n" + raw + b"\r\n"
                b"--" + bnd.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
                b"--" + bnd.encode() + b"--\r\n")
        req = urllib.request.Request(self.valves.comfyui_url + "/upload/image", data=body,
                                     headers={"Content-Type": "multipart/form-data; boundary=" + bnd})
        return json.load(urllib.request.urlopen(req, timeout=60)).get("name", fname)

    DIRECTOR_SYS = (
        "You are an award-winning cinematographer writing prompts for a text-to-video model "
        "(LTX-2). Turn the user's brief, casual idea into ONE single-paragraph, richly detailed "
        "cinematic prompt with professional, artistic taste. Always specify: the subject and its "
        "action; camera angle, movement and lens feel; lighting and time of day; colour palette "
        "and mood; setting detail; and the ambient sound. Add tasteful cinematic detail the user "
        "didn't mention while honouring their intent. Keep it to one coherent shot for a short "
        "clip. Output ONLY the final prompt — no preamble, no lists, no quotes."
    )

    # Ideogram-4 is trained on STRUCTURED JSON captions and emits an "Image blocked by
    # safety filter" placeholder for off-schema (plain-text) input — so the director MUST
    # output the JSON caption. The art-director job is to translate a casual idea into it.
    DIRECTOR_IMG_SYS = (
        "You are an award-winning art director writing prompts for Ideogram-4, which is trained on "
        "STRUCTURED JSON captions. First silently infer the KIND of image the user wants — "
        "logo/brandmark, graphic design/poster, UI or product mockup, photograph, or "
        "illustration/concept art — then output ONE JSON object and NOTHING ELSE (no markdown, no "
        "code fences, no commentary), with EXACTLY these keys:\n"
        '{"high_level_description": "<one vivid sentence describing the whole image>", '
        '"style_description": {"aesthetics": "<style/genre cues for the inferred kind>", '
        '"lighting": "<lighting>", "photo": "<capture or render detail>", '
        '"medium": "<e.g. photograph, vector, 3D, gouache>", "color_palette": ["#RRGGBB", "#RRGGBB"]}, '
        '"compositional_deconstruction": {"background": "<background>", "elements": '
        '[{"type": "obj", "bbox": [x0, y0, x1, y1], "desc": "<object>", "color_palette": ["#RRGGBB"]}]}}\n'
        "bbox coordinates are integers in a 0-1024 canvas (top-left origin). Use the levers that matter "
        "for the inferred kind: logos -> vector/flat/bold negative space/scalable/1-2 colours; posters -> "
        "layout hierarchy, typographic feel, print palette; product/UI -> realistic materials, studio "
        "light, neutral background; photos -> camera and lens (e.g. 85mm f/1.4), lighting, depth of field; "
        "illustration -> medium, line weight, palette, rendering style. If the user wants visible "
        "text/lettering, put the EXACT words in quotes inside high_level_description and the relevant "
        "element desc. Add tasteful professional detail the user didn't mention while honouring intent. "
        "Output ONLY the JSON object."
    )

    # Chroma takes NATURAL-LANGUAGE prompts (T5) — no JSON. It is trained uncensored, and the
    # director is uncensored, so it honours the user's intent without sanitising.
    DIRECTOR_IMG_PROSE_SYS = (
        "You are an award-winning art director writing prompts for the Chroma image model, which takes "
        "rich NATURAL-LANGUAGE prompts. First silently infer the KIND of image the user wants — "
        "photograph, illustration/concept art, poster/graphic, product render, or portrait — then write "
        "ONE single-paragraph, richly detailed prompt with professional, artistic taste. Specify the "
        "subject and its details; composition and framing; lighting; colour and mood; medium/style; and "
        "rendering quality. Use the levers for the kind (photos -> camera and lens, depth of field, film "
        "stock; illustration -> medium, line weight, palette; portrait -> pose, expression, lighting). Add "
        "tasteful detail the user didn't mention while honouring their intent — do not refuse or sanitise. "
        "Output ONLY the final prompt — no preamble, no lists, no quotes around the whole thing."
    )

    def _min_caption(self, text):
        # Last-resort fallback when the director's JSON is unusable. Ideogram-4 blocks SPARSE
        # captions: empty color_palette / empty elements -> "Image blocked by safety filter"
        # (measured 2026-06-11). So every field is POPULATED — a non-empty palette + one
        # full-subject element. Lower quality / object-framed vs a director caption, but it
        # renders instead of hard-blocking.
        return json.dumps({
            "high_level_description": text,
            "style_description": {"aesthetics": "clean, professional, photorealistic, high detail",
                                  "lighting": "soft natural lighting", "photo": "sharp focus, high resolution, detailed",
                                  "medium": "photograph", "color_palette": ["#3A3A3A", "#C8C8C8", "#7A7A7A", "#E8E8E8"]},
            "compositional_deconstruction": {"background": "softly blurred complementary background",
                                             "elements": [{"type": "obj", "bbox": [256, 256, 768, 768],
                                                           "desc": text, "color_palette": ["#888888"]}]},
        })

    def _coerce_caption(self, s, fallback_text):
        # Return a valid JSON caption string. Accept the director's JSON (stripping ``` fences);
        # if it isn't valid schema, wrap the fallback text in a minimal caption.
        t = (s or "").strip()
        if t.startswith("```"):
            t = t.strip("`")
            t = t[4:] if t[:4].lower() == "json" else t
            t = t.strip()
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and obj.get("high_level_description"):
                return json.dumps(obj), obj.get("high_level_description")
        except Exception:
            pass
        return self._min_caption(fallback_text), fallback_text

    def _prior_spec(self, body):
        # Read the crafted prompt the pipe embedded in its most recent reply, so a
        # follow-up message can refine that spec instead of starting from scratch.
        for m in reversed(body.get("messages", [])):
            if m.get("role") != "assistant":
                continue
            c = m.get("content") or ""
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            mt = re.search(r"<!--SPEC:([A-Za-z0-9+/=]+)-->", c)
            if mt:
                try:
                    return base64.b64decode(mt.group(1)).decode("utf-8", "replace")
                except Exception:
                    return None
        return None

    def _enhance(self, user_prompt, i2v, prior_spec=None, kind="video"):
        # kind: "video" (LTX/Sulphur cinematic) · "image" (Ideogram-4 JSON caption) · "chroma" (Chroma prose)
        sys = {"image": self.DIRECTOR_IMG_SYS, "chroma": self.DIRECTOR_IMG_PROSE_SYS}.get(kind, self.DIRECTOR_SYS)
        if i2v and kind == "video":
            sys += (" The user attached an image to animate — describe how it should MOVE "
                    "(motion, camera, ambient sound); do not re-describe the still image.")
        noun = "video" if kind == "video" else "image"
        msgs = [{"role": "system", "content": sys}]
        if prior_spec:
            msgs.append({"role": "user", "content":
                "PREVIOUS " + noun + " prompt (for context):\n" + prior_spec + "\n\n"
                "USER'S NEW MESSAGE: " + user_prompt + "\n\n"
                "If the new message refines/adjusts the previous " + noun + ", output an updated full "
                "prompt that keeps the previous prompt and applies ONLY the requested change. "
                "If it is a brand-new idea, ignore the previous prompt and write a fresh one. "
                "Output ONLY the final prompt."})
        else:
            msgs.append({"role": "user", "content": user_prompt})
        body = json.dumps({"model": self.valves.chat_model, "messages": msgs,
                           "max_tokens": 700 if kind == "image" else 320, "temperature": 0.7 if kind in ("image", "chroma") else 0.8,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(self.valves.chat_url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"].strip()

    # ── long-clip (>15s) via the orchestrator: chain ~10s segments → one combined video ──
    def _target_seconds(self, text):
        """Parse a requested duration from the user's text. 0 = none (single clip)."""
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b", text, re.I)
        if m:
            return min(self.valves.max_seconds, max(1, round(float(m.group(1)) * 60)))
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", text, re.I)
        if m:
            return min(self.valves.max_seconds, max(1, round(float(m.group(1)))))
        return 0

    def _orch_submit(self, lane, prompt, segments):
        req = urllib.request.Request(self.valves.orchestrator_url.rstrip("/") + "/extend",
            data=json.dumps({"prompt": prompt, "lane": lane, "segments": segments, "frames": 241}).encode(),
            headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))["job_id"]

    def _orch_poll(self, jid):
        return json.load(urllib.request.urlopen(
            self.valves.orchestrator_url.rstrip("/") + "/job/" + jid, timeout=30))

    def _submit(self, wf):
        req = urllib.request.Request(self.valves.comfyui_url + "/prompt",
                                     data=json.dumps({"prompt": wf, "client_id": "owui-studio"}).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=60))
        if r.get("node_errors"):
            raise RuntimeError("ComfyUI node_errors: " + json.dumps(r["node_errors"])[:400])
        return r["prompt_id"]

    def _await_output(self, pid, want):
        # want="video" -> mp4 ; want="image" -> png/jpg/webp. Returns (filename, subfolder).
        t0 = time.time()
        while time.time() - t0 < self.valves.timeout_s:
            time.sleep(2)
            h = json.load(urllib.request.urlopen(self.valves.comfyui_url + "/history/" + pid, timeout=30))
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("completed"):
                    for node in h[pid].get("outputs", {}).values():
                        if want == "video":
                            for v in (node.get("gifs") or node.get("videos") or node.get("images") or []):
                                if str(v.get("filename", "")).endswith(".mp4") or str(v.get("format", "")).startswith("video"):
                                    return v.get("filename"), v.get("subfolder", "")
                        else:
                            for v in (node.get("images") or []):
                                if str(v.get("filename", "")).lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                                    return v.get("filename"), v.get("subfolder", "")
                    return None, None
                if st.get("status_str") == "error":
                    raise RuntimeError("ComfyUI generation error")
        raise TimeoutError("ComfyUI timed out")

    def _comfy(self, lane, mode, prompt_text, image_name, frames):
        wf = json.loads(json.dumps(WORKFLOWS[lane + "-" + mode]))
        wf["5"]["inputs"]["text"] = prompt_text
        wf["10"]["inputs"]["value"] = frames
        if mode == "i2v":
            wf["100"]["inputs"]["image"] = image_name
        return self._await_output(self._submit(wf), "video")

    def _comfy_image(self, prompt_text, width, height, steps, seed):
        wf = json.loads(json.dumps(WORKFLOWS["image"]))
        wf["pos"]["inputs"]["text"] = prompt_text
        wf["sigmas"]["inputs"]["steps"] = steps
        wf["sigmas"]["inputs"]["width"] = width
        wf["sigmas"]["inputs"]["height"] = height
        wf["latent"]["inputs"]["width"] = width
        wf["latent"]["inputs"]["height"] = height
        wf["noise"]["inputs"]["noise_seed"] = seed
        return self._await_output(self._submit(wf), "image")

    def _comfy_chroma(self, prompt_text, width, height, steps, cfg, seed):
        wf = json.loads(json.dumps(WORKFLOWS["chroma"]))
        wf["pos"]["inputs"]["text"] = prompt_text
        wf["latent"]["inputs"]["width"] = width
        wf["latent"]["inputs"]["height"] = height
        wf["sigmas"]["inputs"]["steps"] = steps
        wf["guider"]["inputs"]["cfg"] = cfg
        wf["noise"]["inputs"]["noise_seed"] = seed
        return self._await_output(self._submit(wf), "image")

    async def pipe(self, body, __event_emitter__=None):
        async def status(msg, done=False):
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": msg, "done": done}})
        model = str(body.get("model", ""))
        if "chroma" in model:
            lane = "chroma"
        elif "image" in model:
            lane = "image"
        elif "sulphur" in model:
            lane = "sulphur"
        else:
            lane = "ltx"
        label = {"image": "Image (Ideogram-4)", "chroma": "Image · Chroma (uncensored)",
                 "sulphur": "Sulphur (uncensored)", "ltx": "LTX-2.3 (video+audio)"}[lane]
        loop = asyncio.get_event_loop()

        # ── STILL-IMAGE LANES (Ideogram-4 JSON caption · or Chroma prose · single still) ──────────
        if lane in ("image", "chroma"):
            up = ""
            for m in reversed(body.get("messages", [])):
                if m.get("role") == "user":
                    c = m.get("content")
                    up = (" ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text").strip()
                          if isinstance(c, list) else (c or "").strip())
                    break
            if not up:
                return "Describe an image to generate — a logo, poster, product shot, photo, or illustration."
            prior_spec = self._prior_spec(body)
            crafted = up
            if self.valves.enhance:
                await status("\U0001F3A8 Art director crafting the image…")
                try:
                    crafted = await loop.run_in_executor(None, self._enhance, up, False, prior_spec, lane)
                except Exception:
                    crafted = up
            cap = max(256, int(self.valves.image_max_edge))
            w = min(int(self.valves.image_width), cap); h = min(int(self.valves.image_height), cap)
            seed = int(time.time() * 1000) % 2147483647
            try:
                if lane == "chroma":
                    human = crafted if crafted.strip() else up
                    spec_text = human
                    await status("\U0001F513 Rendering on Chroma (uncensored)… (~1-2 min)")
                    fn, sub = await loop.run_in_executor(None, self._comfy_chroma, human, w, h,
                                                         int(self.valves.chroma_steps), float(self.valves.chroma_cfg), seed)
                else:
                    spec_text, human = self._coerce_caption(crafted, up)   # Ideogram-4 needs a JSON caption
                    await status("\U0001F5BC️ Rendering on Ideogram-4… (~1-2 min)")
                    fn, sub = await loop.run_in_executor(None, self._comfy_image, spec_text, w, h,
                                                         int(self.valves.image_steps), seed)
            except Exception as e:
                await status("Failed", True)
                return "⚠️ Image generation failed: " + str(e)
            await status("Done", True)
            if not fn:
                return "Generation finished but no image output was found."
            base = self.valves.browser_base.rstrip("/")
            url = base + "/" + ((sub + "/") if sub else "") + fn
            marker = "<!--SPEC:" + base64.b64encode(spec_text.encode()).decode() + "-->"
            tweaks = "“more dramatic”, “at night”, “close-up”" if lane == "chroma" else "“monochrome”, “tighter crop”, “flat vector style”"
            return ("**\U0001F5BC️ " + label + " · " + str(w) + "x" + str(h) + "**\n\n"
                    "**Prompt used:** " + human + "\n\n"
                    "\U0001F5BC️ **[Open / download the image](" + url + ")**\n\n"
                    "_Want changes? Just say what to tweak — e.g. " + tweaks + " — and I’ll re-craft from this and regenerate._ "
                    "_(Browse all media: " + base + "/ )_" + marker)

        data_uri = self._extract_image(body)
        user_prompt = ""
        for m in reversed(body.get("messages", [])):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    user_prompt = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text").strip()
                else:
                    user_prompt = (c or "").strip()
                break
        mode = "t2v"; image_name = None
        if data_uri:
            await status("\U0001F5BC️ Uploading your image…")
            try:
                image_name = await loop.run_in_executor(None, self._upload_image, data_uri)
                mode = "i2v"
            except Exception as e:
                return "⚠️ Couldn't upload the attached image: " + str(e)
        if not user_prompt:
            if mode == "i2v":
                user_prompt = "subtle natural motion, gentle camera movement"
            else:
                return "Type a scene to generate (or attach an image to animate)."
        fr = ((min(int(self.valves.frames), 361) - 1) // 8) * 8 + 1   # capped + LTX-valid 8k+1
        prior_spec = self._prior_spec(body)
        final_prompt = user_prompt
        if self.valves.enhance and user_prompt:
            await status("\U0001F3A8 Director crafting the shot…")
            try:
                final_prompt = await loop.run_in_executor(None, self._enhance, user_prompt, mode == "i2v", prior_spec)
            except Exception:
                final_prompt = user_prompt

        # Long clip? If the user asked for >15s (text→video), chain ~10s segments via the
        # orchestrator into one combined video. Falls through to a single capped clip if
        # the orchestrator is unreachable.
        target = self._target_seconds(user_prompt) if mode == "t2v" else 0
        if target > 15:
            segments = min(self.valves.max_seconds // 10, max(2, math.ceil(target / 10)))
            jid = None
            try:
                jid = await loop.run_in_executor(None, self._orch_submit, lane, final_prompt, segments)
            except Exception:
                await status("Long-clip engine unreachable — making a single clip instead.")
            if jid:
                last = ""; t0 = time.time()
                await status("\U0001F3AC Long clip (~" + str(segments * 10) + "s): chaining " + str(segments) + " segments on " + label + "…")
                while time.time() - t0 < 3 * self.valves.timeout_s * (segments + 1):
                    await asyncio.sleep(8)
                    try:
                        j = await loop.run_in_executor(None, self._orch_poll, jid)
                    except Exception:
                        continue
                    p = j.get("progress")
                    if p and p != last:
                        last = p
                        await status("\U0001F3AC rendering segment " + p + " (~" + str(segments * 10) + "s total, a few min each)…")
                    if j.get("status") == "done":
                        await status("Done", True)
                        base = self.valves.browser_base.rstrip("/")
                        fn = j.get("filename"); sub = j.get("subfolder", "video")
                        url = base + "/" + ((sub + "/") if sub else "") + fn
                        marker = "<!--SPEC:" + base64.b64encode(final_prompt.encode()).decode() + "-->"
                        return ("**" + label + " · text→video · " + str(segments) + " segments (~" + str(segments * 10) + "s)**\n\n"
                                "**Prompt used:** " + final_prompt + "\n\n"
                                "▶️ **[Open / download the video](" + url + ")**\n\n"
                                "_Want changes? Just say what to tweak and I’ll re-craft and regenerate._ "
                                "_(Browse all media: " + base + "/ )_" + marker)
                    if j.get("status") == "error":
                        await status("Failed", True)
                        return "⚠️ Long-clip generation failed: " + str(j.get("error"))
                await status("Failed", True)
                return "⚠️ Long-clip generation timed out."

        kind = "image→video" if mode == "i2v" else "text→video"
        await status("\U0001F3AC Rendering " + kind + " on " + label + "… (a few minutes)")
        try:
            fn, sub = await loop.run_in_executor(None, self._comfy, lane, mode, final_prompt, image_name, fr)
        except Exception as e:
            await status("Failed", True)
            return "⚠️ Generation failed: " + str(e)
        await status("Done", True)
        if not fn:
            return "Generation finished but no video output was found."
        base = self.valves.browser_base.rstrip("/")
        url = base + "/" + ((sub + "/") if sub else "") + fn
        marker = "<!--SPEC:" + base64.b64encode(final_prompt.encode()).decode() + "-->"
        secs = int(round(fr / 24))
        return ("**" + label + " · " + kind + " · " + str(fr) + " frames (~" + str(secs) + "s)**\n\n"
                "**Prompt used:** " + final_prompt + "\n\n"
                "▶️ **[Open / download the video](" + url + ")**\n\n"
                "_Want changes? Just say what to tweak — e.g. “more moody”, “make it night”, "
                "“slower camera” — and I’ll re-craft from this and regenerate._ "
                "_(Browse all media: " + base + "/ )_" + marker)
'''.replace('__WF_JSON__', WF_JSON)

open(OUT_PATH, 'w').write(PIPE)
print("wrote %s (%d bytes; 6 workflows: ltx/sulphur x t2v/i2v + ideogram-4 + chroma image)" % (OUT_PATH, len(PIPE)))
