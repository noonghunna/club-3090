"""model-switch — a tiny host HTTP wrapper around scripts/switch.sh.

Only one model fits in VRAM at a time on a 1–2 GPU rig, so switching between
models (e.g. Qwen3.6-27B <-> Gemma-4-31B) for cross-model experiments means
tearing one down and booting another. This exposes that over HTTP so a harness
can POST /switch and block until the new model is serving.

It adds NO orchestration logic: `scripts/switch.sh` remains the single source of
truth (registry lookup, down->up, readiness). This is a thin wrapper — the same
role `tools/serve-cockpit` plays as a TUI, done as an HTTP endpoint. stdlib-only
(http.server), matching services/studio/*/server.py.

Endpoints (Bearer auth on all but /healthz, when a token is configured):
  GET  /healthz   -> {"ok": true}
  GET  /status    -> {"current_model", "ready", "port", "container"}
  GET  /models    -> {"available": [{"slug","model","status","port"}, ...]}
  POST /switch    -> body {"slug": "<registry-slug>"} OR {"model": "<model-id>"}
                     blocks until ready; 200 {"ok","slug","model","took_s"}
                     400 unknown/ambiguous · 401 bad token · 409 in-progress · 500 {ok:false,detail}

Config (env; systemd loads them from the repo-root .env):
  CLUB3090_API_TOKEN  control-endpoint bearer token (falls back to VLLM_API_KEY).
                      If neither is set, the endpoint is UNAUTHENTICATED (loopback only).
  MODEL_SWITCH_PORT   listen port (default 8099)
  MODEL_SWITCH_BIND   bind address (default 127.0.0.1)
  PORT                model http port used for the readiness probe / status (else the
                      target slug's registry default_port)
  SWITCH_SCRIPT       path to switch.sh (default <repo>/scripts/switch.sh; overridable for tests)

Run:  python3 tools/model-switch/server.py     (or the club3090-model-switch systemd unit)
"""
from __future__ import annotations

import functools
import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.lib.profiles.compose_registry import (  # noqa: E402
    COMPOSE_REGISTRY,
    curated_default_target,
    model_of_slug,
)

BIND = os.environ.get("MODEL_SWITCH_BIND", "127.0.0.1")
PORT = int(os.environ.get("MODEL_SWITCH_PORT", "8099"))
# Control-endpoint token: dedicated var first, then reuse VLLM_API_KEY so a
# secured rig needs no second secret. Empty -> endpoint is open (loopback only).
CONTROL_TOKEN = os.environ.get("CLUB3090_API_TOKEN") or os.environ.get("VLLM_API_KEY", "")
# Token for talking to the MODEL's own OpenAI API (/v1/models is auth-gated when set).
MODEL_TOKEN = os.environ.get("VLLM_API_KEY", "")
SWITCH_SCRIPT = os.environ.get("SWITCH_SCRIPT") or str(REPO_ROOT / "scripts" / "switch.sh")
SWITCH_TIMEOUT_S = int(os.environ.get("MODEL_SWITCH_TIMEOUT_S", "600"))
DOCKER_BIN = os.environ.get("DOCKER_BIN", "docker")  # overridable for tests
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"}

# Only one switch may run at a time (a switch tears down + boots a model).
_switch_lock = threading.Lock()

MODEL_IDS = sorted({e["model"] for e in COMPOSE_REGISTRY.values()})


class SwitchError(Exception):
    def __init__(self, code: int, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.payload = {"error": message, **extra}


def _topology() -> str:
    """dual/single from visible GPU count (overridable via CLUB3090_TOPOLOGY)."""
    forced = os.environ.get("CLUB3090_TOPOLOGY")
    if forced:
        return forced
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        n = 2
    return "single" if n <= 1 else ("dual" if n == 2 else "multi")


def _as_str(body: dict, key: str) -> str:
    """Return body[key] as a stripped string, or '' — 400 on a non-string value."""
    v = body.get(key)
    if v is None:
        return ""
    if not isinstance(v, str):
        raise SwitchError(400, f"{key!r} must be a string")
    return v.strip()


def resolve_slug(body) -> str:
    """Resolve a request body to a concrete registry slug, or raise SwitchError(400)."""
    if not isinstance(body, dict):
        raise SwitchError(400, "request body must be a JSON object")
    slug = _as_str(body, "slug")
    if slug:
        if slug not in COMPOSE_REGISTRY:
            raise SwitchError(400, f"unknown slug {slug!r}", available=sorted(COMPOSE_REGISTRY))
        return slug
    model = _as_str(body, "model")
    if model:
        if model not in MODEL_IDS:
            matches = [m for m in MODEL_IDS if m.startswith(model)]
            if len(matches) == 1:
                model = matches[0]
            elif len(matches) > 1:
                raise SwitchError(400, f"ambiguous model {model!r}", candidates=matches)
            else:
                raise SwitchError(400, f"unknown model {model!r}", available=MODEL_IDS)
        target = curated_default_target(model, _topology())
        if not target:
            raise SwitchError(400, f"no functional default slug for {model!r} at {_topology()}")
        return target
    raise SwitchError(400, "provide 'slug' or 'model'")


def _docker_ps() -> list[dict]:
    try:
        out = subprocess.run(
            [DOCKER_BIN, "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Ports}}"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            rows.append({"name": parts[0], "image": parts[1], "ports": parts[2]})
    return rows


def _running_model() -> tuple[str | None, int | None]:
    """(container_name, host_port) of the running model server, or (None, None)."""
    for r in _docker_ps():
        if "vllm" in r["image"] or "llama" in r["image"] or r["name"].startswith(
            ("vllm-", "llama-cpp-", "beellama-", "ik-llama-")
        ):
            # Ports like "127.0.0.1:8010->8000/tcp" or "0.0.0.0:8010->8000/tcp".
            for chunk in r["ports"].split(","):
                if "->" in chunk:
                    hostpart = chunk.split("->", 1)[0].strip()
                    try:
                        return r["name"], int(hostpart.rsplit(":", 1)[-1])
                    except ValueError:
                        continue
            return r["name"], None
    return None, None


@functools.lru_cache(maxsize=None)
def _compose_container(slug: str) -> str | None:
    """The default container_name a slug's compose creates (for same-config detection)."""
    entry = COMPOSE_REGISTRY.get(slug) or {}
    try:
        txt = (REPO_ROOT / entry.get("compose_path", "")).read_text()
    except OSError:
        return None
    m = re.search(r'container_name:\s*"?(?:\$\{[^:}]*:-)?([A-Za-z0-9._-]+)\}?"?', txt)
    return m.group(1) if m else None


def _slug_of_container(name: str | None) -> str | None:
    if not name:
        return None
    for slug in COMPOSE_REGISTRY:
        if _compose_container(slug) == name:
            return slug
    return None


def _get_json(url: str, token: str = "") -> dict | None:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read() or b"{}")
    except Exception:
        return None


def _is_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def status() -> dict:
    container, port = _running_model()
    if not container:
        return {"current_model": None, "current_slug": None, "ready": False, "port": None, "container": None}
    ready = _is_ready(port) if port else False
    current_model = None
    if port:
        data = _get_json(f"http://localhost:{port}/v1/models", MODEL_TOKEN)
        if data and data.get("data"):
            current_model = data["data"][0].get("id")
    return {"current_model": current_model, "current_slug": _slug_of_container(container),
            "ready": ready, "port": port, "container": container}


def do_switch(slug: str) -> dict:
    """Run switch.sh <slug>, blocking until ready. Raises SwitchError on failure."""
    entry = COMPOSE_REGISTRY[slug]
    port = os.environ.get("PORT") or str(entry["default_port"])
    env = dict(os.environ)
    # Probe the unauthenticated /health (not the auth-gated /v1/models) so readiness
    # works whether or not the model has VLLM_API_KEY set.
    env["READY_URL"] = f"http://localhost:{port}/health"
    env.setdefault("READY_TIMEOUT", str(SWITCH_TIMEOUT_S))
    t0 = time.time()
    try:
        p = subprocess.run(
            ["bash", SWITCH_SCRIPT, slug],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
            timeout=SWITCH_TIMEOUT_S + 60,
        )
    except subprocess.TimeoutExpired:
        raise SwitchError(500, "switch timed out", slug=slug)
    if p.returncode != 0:
        raise SwitchError(500, "switch failed", slug=slug, detail=(p.stderr or p.stdout or "")[-800:])
    return {"ok": True, "slug": slug, "model": model_of_slug(slug), "took_s": round(time.time() - t0, 1)}


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        if not CONTROL_TOKEN:
            return True  # open (warned at startup); loopback bind is the guard
        got = self.headers.get("Authorization", "")
        pfx = "Bearer "
        return got.startswith(pfx) and hmac.compare_digest(got[len(pfx):], CONTROL_TOKEN)

    def do_GET(self):
        if self.path == "/healthz":
            return self._json(200, {"ok": True})
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})
        if self.path == "/status":
            return self._json(200, status())
        if self.path == "/models":
            avail = [
                {"slug": s, "model": e["model"], "status": e["status"], "port": e["default_port"]}
                for s, e in sorted(COMPOSE_REGISTRY.items())
            ]
            return self._json(200, {"available": avail})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})
        if self.path != "/switch":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": f"bad body: {e}"})
        try:
            slug = resolve_slug(body)
        except SwitchError as e:
            return self._json(e.code, e.payload)
        # Fast no-op ONLY when the requested slug's compose is the exact one already
        # running + ready — compared by container, so a different quant/topology/engine
        # of the same model id still triggers a real switch (no silent skip).
        cur = status()
        if cur["ready"] and cur["container"] and _compose_container(slug) == cur["container"]:
            return self._json(200, {"ok": True, "slug": slug, "model": model_of_slug(slug),
                                    "status": "already-running", "took_s": 0})
        if not _switch_lock.acquire(blocking=False):
            return self._json(409, {"error": "a switch is already in progress"})
        try:
            result = do_switch(slug)
        except SwitchError as e:
            return self._json(e.code, {"ok": False, **e.payload})
        finally:
            _switch_lock.release()
        return self._json(200, result)

    def log_message(self, *a):  # quiet; systemd journal captures stderr
        pass


def main() -> None:
    if not CONTROL_TOKEN:
        if BIND not in LOOPBACK_HOSTS:
            raise SystemExit(
                f"model-switch: REFUSING to start — MODEL_SWITCH_BIND={BIND!r} is non-loopback and "
                "no CLUB3090_API_TOKEN/VLLM_API_KEY is set; that would expose the destructive "
                "/switch endpoint unauthenticated. Set a token, or bind to 127.0.0.1.")
        print("model-switch: WARNING — no CLUB3090_API_TOKEN/VLLM_API_KEY set; "
              "control endpoint is UNAUTHENTICATED (loopback only).", flush=True)
    srv = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"model-switch: serving on {BIND}:{PORT} (auth={'on' if CONTROL_TOKEN else 'OFF'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
