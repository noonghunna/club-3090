#!/usr/bin/env python3
"""Studio image shim — a transparent ComfyUI reverse-proxy that makes Open WebUI's
NATIVE 🖼️ image button work with Ideogram-4.

Why this exists
---------------
Ideogram-4 is trained on STRUCTURED JSON captions; plain text denoises to a gray
"Image blocked by safety filter" placeholder (an in-weights fallback). OWUI's native
image button sends the user's plain text straight into the ComfyUI workflow → blocked.
OWUI's image-prompt-generation LLM step can't help: it returns `{"prompt": "<string>"}`,
and nesting the Ideogram JSON inside that string defeats the task models (escaping fails).

Fix: point OWUI's `COMFYUI_BASE_URL` at this shim instead of ComfyUI directly, and turn
OWUI's image-prompt-generation OFF. The shim proxies EVERYTHING to real ComfyUI
(including the `/ws` progress socket) untouched, EXCEPT `POST /prompt`: there it reads the
plain-text prompt node, asks the Studio director (qwen, :8090) to craft a rich Ideogram-4
JSON caption (the escaping is done in Python — reliable, unlike asking the model), rewrites
the node, and forwards. Blast radius = image generation only; OWUI's title/tag/etc. task
generation is untouched (that's why this is a ComfyUI proxy, not a task-model override).

Env:
  COMFYUI_UPSTREAM   real ComfyUI base url      (default http://127.0.0.1:8188)
  DIRECTOR_URL       director OpenAI base + /v1  (default http://127.0.0.1:8090/v1)
  DIRECTOR_MODEL     director model id           (default qwen3.5-4b-uncensored)
  SHIM_PORT          listen port                 (default 8191)
"""
import os, json, asyncio, aiohttp
from aiohttp import web, WSMsgType

UPSTREAM = os.environ.get("COMFYUI_UPSTREAM", "http://127.0.0.1:8188").rstrip("/")
DIRECTOR_URL = os.environ.get("DIRECTOR_URL", "http://127.0.0.1:8090/v1").rstrip("/")
DIRECTOR_MODEL = os.environ.get("DIRECTOR_MODEL", "qwen3.5-4b-uncensored")
PORT = int(os.environ.get("SHIM_PORT", "8191"))

# Kept in sync with services/studio/build_studio_pipe.py DIRECTOR_IMG_SYS (image lane).
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


def _min_caption(text):
    # Populated fallback — sparse/empty captions hard-block on Ideogram-4 (measured).
    return json.dumps({
        "high_level_description": text,
        "style_description": {"aesthetics": "clean, professional, photorealistic, high detail",
                              "lighting": "soft natural lighting", "photo": "sharp focus, high resolution, detailed",
                              "medium": "photograph", "color_palette": ["#3A3A3A", "#C8C8C8", "#7A7A7A", "#E8E8E8"]},
        "compositional_deconstruction": {"background": "softly blurred complementary background",
                                         "elements": [{"type": "obj", "bbox": [256, 256, 768, 768],
                                                       "desc": text, "color_palette": ["#888888"]}]},
    })


def _coerce_caption(s, fallback_text):
    t = (s or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t[:4].lower() == "json" else t
        t = t.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and obj.get("high_level_description"):
            return json.dumps(obj)
    except Exception:
        pass
    return _min_caption(fallback_text)


async def _craft_caption(session, brief):
    """Ask the director for a rich Ideogram JSON caption; retry once on a flaky call, then
    fall back to a populated wrap (the director's first call after idle can miss)."""
    body = {"model": DIRECTOR_MODEL,
            "messages": [{"role": "system", "content": DIRECTOR_IMG_SYS},
                         {"role": "user", "content": brief}],
            "max_tokens": 700, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False}}
    last_err = None
    for attempt in range(3):
        try:
            async with session.post(DIRECTOR_URL + "/chat/completions", json=body,
                                    timeout=aiohttp.ClientTimeout(total=120)) as r:
                data = await r.json()
                raw = (data["choices"][0]["message"]["content"] or "").strip()
            # Only accept a VALID director caption; otherwise retry (don't settle for min_caption yet).
            try:
                obj = json.loads(raw[4:].strip() if raw[:4].lower() == "json" else raw.strip("`").strip()
                                 if raw.startswith("```") else raw)
                if isinstance(obj, dict) and obj.get("high_level_description"):
                    return json.dumps(obj)
            except Exception:
                pass
            last_err = "invalid director JSON"
        except Exception as e:
            last_err = repr(e)
        await asyncio.sleep(1.0)
    print(f"[shim] director unusable after retries ({last_err}) -> min_caption", flush=True)
    return _min_caption(brief)


def _find_text_node(wf):
    """Locate the prompt text node OWUI wrote into. Prefer 'pos'; else first CLIPTextEncode
    whose text is not already a JSON caption."""
    n = wf.get("pos")
    if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode":
        return "pos"
    for nid, node in wf.items():
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            return nid
    return None


async def prompt_handler(request):
    """Intercept ComfyUI /prompt: rewrite the plain-text prompt into an Ideogram caption."""
    raw_body = await request.read()
    try:
        body = json.loads(raw_body)
        wf = body.get("prompt", {})
        nid = _find_text_node(wf)
        if nid:
            cur = (wf[nid].get("inputs", {}) or {}).get("text", "")
            # Only craft if it's plain text (not already a JSON caption).
            if isinstance(cur, str) and cur.strip() and not cur.strip().startswith("{"):
                async with aiohttp.ClientSession() as s:
                    caption = await _craft_caption(s, cur.strip())
                wf[nid]["inputs"]["text"] = caption
                raw_body = json.dumps(body).encode()
                print(f"[shim] crafted Ideogram caption for {cur.strip()[:60]!r} ({len(caption)} chars)", flush=True)
    except Exception as e:
        print(f"[shim] /prompt rewrite error: {e!r}", flush=True)  # forward unchanged
    async with aiohttp.ClientSession() as s:
        async with s.post(UPSTREAM + "/prompt", data=raw_body,
                          headers={"Content-Type": "application/json"}) as r:
            body = await r.read()
            return web.Response(body=body, status=r.status,
                                headers={"Content-Type": r.headers.get("Content-Type", "application/json")})


async def ws_handler(request):
    """Relay the ComfyUI progress websocket (OWUI waits on it for completion)."""
    client_ws = web.WebSocketResponse()
    await client_ws.prepare(request)
    qs = request.query_string
    url = UPSTREAM.replace("http://", "ws://").replace("https://", "wss://") + "/ws" + (("?" + qs) if qs else "")
    try:
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url) as up:
                async def c2u():
                    async for m in client_ws:
                        if m.type == WSMsgType.TEXT: await up.send_str(m.data)
                        elif m.type == WSMsgType.BINARY: await up.send_bytes(m.data)
                        elif m.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED): break
                async def u2c():
                    async for m in up:
                        if m.type == WSMsgType.TEXT: await client_ws.send_str(m.data)
                        elif m.type == WSMsgType.BINARY: await client_ws.send_bytes(m.data)
                        elif m.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED): break
                await asyncio.gather(c2u(), u2c())
    except Exception:
        pass
    return client_ws


async def proxy_handler(request):
    """Transparent passthrough for every other ComfyUI endpoint (/history, /view, /upload, ...)."""
    url = UPSTREAM + request.raw_path
    data = await request.read()
    hdrs = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    async with aiohttp.ClientSession(auto_decompress=False) as s:
        async with s.request(request.method, url, data=data if data else None,
                             headers=hdrs, allow_redirects=False) as r:
            body = await r.read()
            out_hdrs = {k: v for k, v in r.headers.items()
                        if k.lower() not in ("content-encoding", "transfer-encoding", "content-length", "connection")}
            return web.Response(body=body, status=r.status, headers=out_hdrs)


async def health(request):
    return web.json_response({"ok": True, "upstream": UPSTREAM, "director": DIRECTOR_URL})


def make_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/shim/health", health)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/prompt", prompt_handler)
    app.router.add_route("*", "/{tail:.*}", proxy_handler)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=PORT)
