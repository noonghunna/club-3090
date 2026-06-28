"""Production service — a host HTTP wrapper around the production agent.

The OWUI `🎬 Studio · Production` lane is a thin client over this. The agent is
host-native (localhost lanes, host paths, ffmpeg, the docker-cp TTS read), so this
runs on the HOST (NOT in the OWUI container); OWUI reaches it via
`host.docker.internal:<port>`. stdlib-only (http.server) — no aiohttp dep.

Endpoints:
  POST /produce  {brief, video_lane?, keyframe_lane?, continuity?, music?, narration?,
                  voice?, shots?, backend?}  -> {job_id, stack}
                  (lanes default to 'auto'; an unknown/unwired lane 400s with the
                   renderable set — the stack is never silently picked)
  GET  /produce/health  -> {ok, active, video_lanes, keyframe_lanes, renders_today}
  GET  /job/<job_id>    -> {status: planning|rendering|done|error, phase, frac, title,
                            stack, stack_desc, final, gallery_url, error}

plan-then-execute: the 4B plans up front, the executor runs the lane sequence.
Single-flight (one production at a time) via ProductionLock — a 2nd /produce 409s.

Run:  python3 -m services.studio.production.server     (host; STUDIO_PRODUCTION_PORT=8195)
"""
from __future__ import annotations

import datetime
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, planner, registry
from .executor import run_production
from .lock import ProductionLock, ProductionLockHeld
from .stack import (
    KEYFRAME_LANES,
    VIDEO_LANES,
    ProductionStack,
    StackError,
    describe_stack,
    resolve_stack,
    wired_keyframe_lanes,
    wired_video_lanes,
)

PORT = int(os.environ.get("STUDIO_PRODUCTION_PORT", "8195"))
# Gallery serves OUTPUT_ROOT at /; set this to make `gallery_url` absolute for OWUI.
GALLERY_BASE = os.environ.get("STUDIO_GALLERY_BASE", "").rstrip("/")

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "production"


def _job_id(brief: str) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{_slug(brief)}-{stamp}-{os.urandom(2).hex()}"


def _set(job_id: str, **kw) -> None:
    with _LOCK:
        _JOBS.setdefault(job_id, {}).update(kw)


def _active() -> list[str]:
    with _LOCK:
        return [j for j, v in _JOBS.items() if v.get("status") in ("planning", "rendering")]


def _run_job(job_id: str, brief: str, stack: ProductionStack, shots: int, backend: str) -> None:
    """Background worker: plan then render, holding the production lock end-to-end."""
    lock = ProductionLock()
    try:
        lock.acquire()
    except ProductionLockHeld as e:
        _set(job_id, status="error", error=str(e))
        return
    try:
        prod_dir = os.path.join(config.PRODUCTIONS_DIR, job_id)
        os.makedirs(prod_dir, exist_ok=True)
        _set(job_id, status="planning", phase="director planning the brief", frac=0.02)
        plan, arts = planner.plan_from_brief(
            brief, registry.load(), n_shots=shots, stack=stack,
            prompts_dir=os.path.join(prod_dir, "prompts"),
        )
        _set(job_id, status="rendering", title=plan.project.title, phase="planned", frac=0.05)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary = run_production(
            plan, backend_name=backend, job_id=job_id, now_iso=now_iso,
            extra_artifacts=arts, stack=stack,
            progress_cb=lambda m, f: _set(job_id, phase=m, frac=f),
        )
        final = summary["final"]
        rel = os.path.relpath(final, config.OUTPUT_ROOT)
        _set(job_id, status="done", phase="done", frac=1.0, final=final,
             gallery_url=(GALLERY_BASE + "/" + rel) if GALLERY_BASE else rel)
    except planner.PlannerError as e:
        _set(job_id, status="error", error=f"planner: {e}")
    except Exception as e:
        _set(job_id, status="error", error=f"{type(e).__name__}: {e}")
    finally:
        lock.release()


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/produce/health":
            return self._json(200, {
                "ok": True, "director": config.DIRECTOR_MODEL, "active": _active(),
                "video_lanes": {k: dict(v, wired=v["wired"]) for k, v in VIDEO_LANES.items()},
                "keyframe_lanes": {k: v for k, v in KEYFRAME_LANES.items()},
                "renders_today": {"video": wired_video_lanes(), "keyframe": wired_keyframe_lanes()},
            })
        if self.path.startswith("/job/"):
            jid = self.path[len("/job/"):]
            with _LOCK:
                v = dict(_JOBS.get(jid) or {})
            return self._json(200 if v else 404, {"job_id": jid, **v} if v else {"error": "no such job"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/produce":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            b = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": f"bad body: {e}"})
        brief = (b.get("brief") or "").strip()
        if not brief:
            return self._json(400, {"error": "no brief"})
        # Resolve the operator-chosen stack; an unknown/unwired lane 400s with a clear
        # message (and the renderable set) — never silently coerced.
        try:
            stack = resolve_stack(
                video_lane=b.get("video_lane"), keyframe_lane=b.get("keyframe_lane"),
                continuity=b.get("continuity"), narration=b.get("narration", True),
                music=b.get("music", True), voice=b.get("voice", ""),
            )
        except StackError as e:
            return self._json(400, {"error": str(e), "renders_today": {
                "video": wired_video_lanes(), "keyframe": wired_keyframe_lanes()}})
        if _active():
            return self._json(409, {"error": "a production is already running", "job_id": _active()[0]})
        job_id = _job_id(brief)
        _set(job_id, status="planning", phase="queued", frac=0.0, brief=brief, title="…",
             stack=stack.to_dict(), stack_desc=describe_stack(stack))
        threading.Thread(
            target=_run_job,
            args=(job_id, brief, stack, int(b.get("shots", 3)), b.get("backend", "live")),
            daemon=True,
        ).start()
        return self._json(200, {"job_id": job_id, "stack": stack.to_dict()})


def main() -> None:
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[studio-production] serving on :{PORT}  (director {config.DIRECTOR_MODEL})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
