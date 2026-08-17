"""model-switch — a self-healing host HTTP control plane around scripts/switch.sh.

Only one model fits in VRAM at a time on a 1–2 GPU rig, so switching between models
means tearing one down and booting another. This exposes that over HTTP so a harness
can POST /switch and block until the new model is serving — and, because much of the
catalog is experimental, it also *keeps the desired model alive*:

  - hardware-aware /models (hides slugs that can't fit this GPU count),
  - explicit-consent force (non-production / oversized slugs need {"force": true}),
  - rollback if a switch fails (restore the previously-healthy model),
  - a background watchdog that re-launches a crashed/wedged model and, on a crash-loop,
    tears it down and marks the service `degraded` instead of thrashing forever.

It adds NO orchestration logic: scripts/switch.sh remains the single source of truth
(registry lookup, down->up, readiness). stdlib-only (http.server), matching
services/studio/*/server.py; the watchdog thread is the only new shape.

Endpoints: the authoritative list is the ROUTES table below (also served, unauthenticated,
at GET / for self-discovery). Bearer auth on all but /healthz and / when a token is configured.

Force: a slug is `requires_force` when its status is not production/caveats OR it needs
more GPUs than this host has. Switching to it needs an explicit {"force": true}; that
authorization is persisted so healing re-launches it the same way. FORCE=1 also bypasses
switch.sh's VRAM/SM/hardware-fit preflights, so it is deliberately a user act — never inferred.

Config (env; systemd loads them from the repo-root .env). All new vars are MODEL_SWITCH_*:
  CLUB3090_API_TOKEN / VLLM_API_KEY   control-endpoint bearer token (empty => open, loopback)
  MODEL_SWITCH_BIND (127.0.0.1) / MODEL_SWITCH_PORT (8099)
  MODEL_SWITCH_WATCHDOG (1)            background self-healing on/off
  MODEL_SWITCH_WATCH_INTERVAL_S (15)   watchdog poll period
  MODEL_SWITCH_HEAL_GRACE_S (60)       down time before healing a previously-healthy model
  MODEL_SWITCH_BOOT_GRACE_S (300)      unready time allowed while a model is booting
  MODEL_SWITCH_STABILITY_WINDOW_S (300) continuous health before the heal budget resets
  MODEL_SWITCH_HEAL_BUDGET_WINDOW_S (600) rolling window for the crash-loop budget
  MODEL_SWITCH_MAX_HEAL_FAILURES (3)   heals allowed in-window before degrading
  MODEL_SWITCH_STATE_DIR               desired-state dir (default $STATE_DIRECTORY else XDG)
  MODEL_SWITCH_GPU_COUNT               override GPU count (else nvidia-smi -L)
  PORT                                 model http port for readiness/status (else registry)
  SWITCH_SCRIPT / DOCKER_BIN           overridable for tests

Run:  python3 tools/model-switch/server.py     (or the club3090-model-switch systemd unit)
"""
from __future__ import annotations

import glob
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.lib.profiles.compose_registry import (  # noqa: E402
    COMPOSE_REGISTRY,
    FUNCTIONAL_STATUSES,
    curated_default_target,
    model_of_slug,
)

# ---- config --------------------------------------------------------------
BIND = os.environ.get("MODEL_SWITCH_BIND", "127.0.0.1")
PORT = int(os.environ.get("MODEL_SWITCH_PORT", "8099"))
TOKEN_ENV = ("CLUB3090_API_TOKEN", "VLLM_API_KEY")   # control-endpoint bearer-token sources, in order
CONTROL_TOKEN = next((v for v in (os.environ.get(k) for k in TOKEN_ENV) if v), "")
SWITCH_SCRIPT = os.environ.get("SWITCH_SCRIPT") or str(REPO_ROOT / "scripts" / "switch.sh")
SWITCH_TIMEOUT_S = int(os.environ.get("MODEL_SWITCH_TIMEOUT_S", "600"))
DOCKER_BIN = os.environ.get("DOCKER_BIN", "docker")
SETUP_SCRIPT = os.environ.get("SETUP_SCRIPT") or str(REPO_ROOT / "scripts" / "setup.sh")
WEIGHTS_READER = str(REPO_ROOT / "scripts" / "lib" / "profiles" / "weights.py")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"}


def _resolve_model_dir() -> str:
    """MODEL_DIR with setup.sh/switch.sh precedence: shell/systemd env wins, else the repo-root
    .env's MODEL_DIR= line, else <repo>/models-cache. A bare `python3 server.py` foreground run
    doesn't auto-load .env, so without this /pull would use models-cache while switch.sh (which
    loads .env) serves from the .env path — re-creating the missing-weights failure."""
    v = os.environ.get("MODEL_DIR")
    if v:
        return v
    try:
        for line in (REPO_ROOT / ".env").read_text().splitlines():
            s = line.strip()
            if s.startswith("MODEL_DIR=") and not s.startswith("#"):
                val = s.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    except OSError:
        pass
    return str(REPO_ROOT / "models-cache")


MODEL_DIR = _resolve_model_dir()

WATCHDOG_ENABLED = os.environ.get("MODEL_SWITCH_WATCHDOG", "1") == "1"
WATCH_INTERVAL_S = int(os.environ.get("MODEL_SWITCH_WATCH_INTERVAL_S", "15"))
HEAL_GRACE_S = int(os.environ.get("MODEL_SWITCH_HEAL_GRACE_S", "60"))
BOOT_GRACE_S = int(os.environ.get("MODEL_SWITCH_BOOT_GRACE_S", "300"))
STABILITY_WINDOW_S = int(os.environ.get("MODEL_SWITCH_STABILITY_WINDOW_S", "300"))
HEAL_BUDGET_WINDOW_S = int(os.environ.get("MODEL_SWITCH_HEAL_BUDGET_WINDOW_S", "600"))
MAX_HEAL_FAILURES = int(os.environ.get("MODEL_SWITCH_MAX_HEAL_FAILURES", "3"))


def _default_state_dir() -> Path:
    d = os.environ.get("MODEL_SWITCH_STATE_DIR") or os.environ.get("STATE_DIRECTORY")
    if d:
        return Path(d)
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "club3090-model-switch"


STATE_DIR = _default_state_dir()
STATE_FILE = STATE_DIR / "state.json"

MODEL_IDS = sorted({e["model"] for e in COMPOSE_REGISTRY.values()})

# ---- serialized state ----------------------------------------------------
# One lock serializes every mutation (/switch, /heal, /down, watchdog heal) and every
# state write. Reads (status/models) are lock-free snapshot reads.
_transition_lock = threading.Lock()


class _TryLock:
    """Non-blocking `with` for _transition_lock: `with _TryLock(lock) as got:` — `got` is True iff
    acquired, and the lock is released on exit ONLY if we acquired it. Replaces the repeated
    acquire(blocking=False) / try / finally-release boilerplate."""
    def __init__(self, lock):
        self._lock, self._held = lock, False

    def __enter__(self):
        self._held = self._lock.acquire(blocking=False)
        return self._held

    def __exit__(self, *exc):
        if self._held:
            self._lock.release()


_state: dict = {"desired_slug": None, "force_authorized": False, "degraded": False, "heal_history": []}

# in-memory watchdog timers (grace is re-timed after a restart; only heal_history persists)
_wd = {
    "launch_ts": 0.0, "last_healthy_ts": 0.0, "down_since": 0.0, "healthy_streak_start": 0.0,
    "last_action": None, "last_action_ts": 0.0, "last_error": None,
}
_wd_thread: threading.Thread | None = None


class SwitchError(Exception):
    def __init__(self, code: int, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.payload = {"error": message, **extra}


# ---- state persistence (atomic) ------------------------------------------
def _save_state() -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
        tmp.write_text(json.dumps(_state))
        os.replace(tmp, STATE_FILE)
    except OSError as e:
        _wd["last_error"] = f"state save failed: {e}"


def _set_state(**kw) -> None:
    """Update + persist state. Caller MUST hold _transition_lock."""
    _state.update(kw)
    _save_state()


def _load_state() -> None:
    data: dict = {}
    try:
        loaded = json.loads(STATE_FILE.read_text())
        if isinstance(loaded, dict):
            data = loaded
    except (OSError, ValueError):
        data = {}  # missing or corrupt -> empty
    slug = data.get("desired_slug")
    if slug is not None and slug not in COMPOSE_REGISTRY:
        _wd["last_error"] = f"dropped unknown persisted slug {slug!r}"
        slug = None
    _state["desired_slug"] = slug
    _state["force_authorized"] = bool(data.get("force_authorized", False)) if slug else False
    _state["degraded"] = bool(data.get("degraded", False))
    _state["heal_history"] = [
        float(t) for t in data.get("heal_history", []) if isinstance(t, (int, float))
    ]


def _pruned_heals(now: float | None = None) -> list[float]:
    now = now if now is not None else time.time()
    cutoff = now - HEAL_BUDGET_WINDOW_S
    return [t for t in _state.get("heal_history", []) if t >= cutoff]


# ---- topology / GPU eligibility ------------------------------------------
def _topology() -> str:
    """single/dual/multi family from GPU count (override CLUB3090_TOPOLOGY). For slug resolution."""
    forced = os.environ.get("CLUB3090_TOPOLOGY")
    if forced:
        return forced
    n = _gpu_count()
    if n is None:
        return "dual"
    return "single" if n <= 1 else ("dual" if n == 2 else "multi")


def _raw_topology(slug: str) -> str | None:
    """The concrete topology dir from compose_path (single/dual/multiN) — keeps the N."""
    entry = COMPOSE_REGISTRY.get(slug) or {}
    cp = entry.get("compose_path", "")
    if "/compose/" not in cp:
        return None
    return cp.split("/compose/", 1)[1].split("/", 1)[0]


def _required_gpus(slug: str) -> int | None:
    topo = _raw_topology(slug)
    if topo == "single":
        return 1
    if topo == "dual":
        return 2
    if topo and topo.startswith("multi"):
        m = re.match(r"multi(\d+)", topo)
        return int(m.group(1)) if m else None
    return None


_gpu_count_cache: int | None = None


def _gpu_count() -> int | None:
    """Visible GPU count. Override MODEL_SWITCH_GPU_COUNT; None if undetectable (fail-open)."""
    global _gpu_count_cache
    override = os.environ.get("MODEL_SWITCH_GPU_COUNT")
    if override:
        try:
            return int(override)
        except ValueError:
            return None
    if _gpu_count_cache is not None:
        return _gpu_count_cache
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        n = sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
        if n:
            _gpu_count_cache = n  # cache only a positive result
        return n or None
    except Exception:
        return None


def _gpu_eligible(slug: str) -> bool | None:
    """True/False fit, or None when either side is unknown (fail-open: don't block)."""
    req = _required_gpus(slug)
    cnt = _gpu_count()
    if req is None or cnt is None:
        return None
    return req <= cnt


def requires_force(slug: str) -> bool:
    """Consent needed? Non-functional status OR positively GPU-ineligible. Derived, never stored."""
    entry = COMPOSE_REGISTRY.get(slug) or {}
    if entry.get("status", "production") not in FUNCTIONAL_STATUSES:
        return True
    return _gpu_eligible(slug) is False


def _force_reason(slug: str) -> str:
    entry = COMPOSE_REGISTRY.get(slug) or {}
    if entry.get("status", "production") not in FUNCTIONAL_STATUSES:
        return f"status={entry.get('status')}"
    return f"needs {_required_gpus(slug)} GPUs, host has {_gpu_count()}"


# ---- slug resolution -----------------------------------------------------
def _as_str(body: dict, key: str) -> str:
    v = body.get(key)
    if v is None:
        return ""
    if not isinstance(v, str):
        raise SwitchError(400, f"{key!r} must be a string")
    return v.strip()


def resolve_slug(body) -> str:
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


# ---- docker observation --------------------------------------------------
def _parse_host_port(ports: str) -> int | None:
    for chunk in (ports or "").split(","):
        if "->" in chunk:
            hostpart = chunk.split("->", 1)[0].strip()
            try:
                return int(hostpart.rsplit(":", 1)[-1])
            except ValueError:
                continue
    return None


def _slug_from_config_file(cfgfile: str) -> str | None:
    """Map a compose config-files label (absolute path(s)) to the exact registry slug."""
    for part in (cfgfile or "").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            rel = os.path.relpath(os.path.realpath(p), str(REPO_ROOT)).replace(os.sep, "/")
        except (OSError, ValueError):
            continue
        for slug, entry in COMPOSE_REGISTRY.items():
            if entry.get("compose_path") == rel:  # exact normalized match, not suffix
                return slug
    return None


def _is_ready(port: int | None) -> bool:
    if not port:
        return False
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _observe() -> dict:
    """Observe the running model. state in {healthy, unready, absent, unknown}.

    'unknown' means docker itself is unreachable (never triggers healing); 'absent' means
    docker is fine but no model container is up.
    """
    base = {"slug": None, "container": None, "port": None, "restart_count": None, "started_at": None}
    try:
        ps = subprocess.run(
            [DOCKER_BIN, "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Ports}}"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return {**base, "state": "unknown"}
    if ps.returncode != 0:
        return {**base, "state": "unknown"}

    name = ports = None
    for line in ps.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        n, img, prts = parts[0], parts[1], parts[2]
        if "vllm" in img or "llama" in img or n.startswith(
            ("vllm-", "llama-cpp-", "beellama-", "ik-llama-", "sglang-")
        ):
            name, ports = n, prts
            break
    if not name:
        return {**base, "state": "absent"}

    try:
        insp = subprocess.run(
            [DOCKER_BIN, "inspect", "-f",
             '{{index .Config.Labels "com.docker.compose.project.config_files"}}\t'
             '{{.RestartCount}}\t{{.State.StartedAt}}', name],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return {**base, "state": "unknown"}
    if insp.returncode != 0:
        return {**base, "state": "unknown"}
    cfgfile, restart, started = (insp.stdout.strip().split("\t") + ["", "", ""])[:3]
    port = _parse_host_port(ports)
    try:
        rc = int(restart)
    except (TypeError, ValueError):
        rc = None
    return {"slug": _slug_from_config_file(cfgfile), "container": name, "port": port,
            "restart_count": rc, "started_at": started or None,
            "state": "healthy" if _is_ready(port) else "unready"}


def _uptime_s(started_at: str | None) -> int | None:
    if not started_at:
        return None
    try:
        dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        return max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
    except (ValueError, TypeError):
        return None


# ---- weights presence + download (reuses weights.py + setup.sh) ----------
# Mirrors serve-cockpit's Download action: a slug's weights = its core `weights_variant`
# PLUS every `weights_companions` (mmproj / DFlash the compose mounts). All must be on disk
# under MODEL_DIR/<subdir> for the slug to actually serve.
_weights_index_cache: dict | None = None


def _weights_index() -> dict:
    """(model, variant) -> {subdir, size_gb, verify_glob} from `weights.py list --json`. Cached."""
    global _weights_index_cache
    if _weights_index_cache is not None:
        return _weights_index_cache
    idx: dict = {}
    try:
        out = subprocess.run(["python3", WEIGHTS_READER, "list", "--json"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            for row in json.loads(out.stdout or "[]"):
                idx[(row.get("model"), row.get("variant"))] = {
                    "subdir": row.get("subdir") or "",
                    "size_gb": row.get("size_gb"),
                    "verify_glob": row.get("verify_glob") or "*.safetensors",
                }
    except Exception:
        pass
    _weights_index_cache = idx
    return idx


def _companion_keys(entry: dict) -> list[str]:
    """weights_companions as fully-qualified <model>:<variant> keys (qualify with model if bare)."""
    model = entry.get("model")
    return [c if ":" in c else f"{model}:{c}" for c in (entry.get("weights_companions") or []) if c]


def _slug_artifacts(slug: str) -> list[dict]:
    """Core weights variant + companions, each as {model, variant, subdir, size_gb, verify_glob}."""
    entry = COMPOSE_REGISTRY.get(slug) or {}
    model, core = entry.get("model"), entry.get("weights_variant")
    idx = _weights_index()
    arts = []
    keys = ([f"{model}:{core}"] if core else []) + _companion_keys(entry)
    for key in keys:
        m, v = key.split(":", 1)
        meta = idx.get((m, v)) or {"subdir": "", "size_gb": None, "verify_glob": "*"}
        arts.append({"model": m, "variant": v, **meta})
    return arts


def _weights_present(slug: str) -> bool:
    """True if every artifact's subdir exists under MODEL_DIR with a verify_glob match. Artifacts
    with no known subdir can't be verified here and are skipped (switch.sh preflight still guards)."""
    arts = _slug_artifacts(slug)
    if not arts:
        return True  # unknown slug shape -> don't block
    for a in arts:
        sub = a["subdir"]
        if not sub:
            continue
        d = os.path.join(MODEL_DIR, sub)
        if not os.path.isdir(d) or not glob.glob(os.path.join(d, a["verify_glob"])):
            return False
    return True


def _slug_total_size_gb(slug: str) -> float | None:
    """Summed download size, or None if ANY artifact size is unknown/variable (skip disk preflight)."""
    total = 0.0
    for a in _slug_artifacts(slug):
        s = a["size_gb"]
        if not isinstance(s, (int, float)):
            return None
        total += s
    return total


def _free_bytes(path: str) -> int | None:
    """Free bytes at path, walking up to the nearest existing ancestor first (path may not exist
    yet — mirrors preflight.sh's disk check)."""
    p = path
    while p and not os.path.isdir(p):
        p = os.path.dirname(p)
    try:
        return shutil.disk_usage(p or "/").free
    except OSError:
        return None


def _weights_missing_error(slug: str) -> SwitchError:
    core = (COMPOSE_REGISTRY.get(slug) or {}).get("weights_variant")
    model = model_of_slug(slug)
    return SwitchError(409, "weights_missing", slug=slug, model=model, model_dir=MODEL_DIR,
                       weight_key=f"{model}:{core}" if core else None,
                       pull=f'POST /pull {{"slug": "{slug}"}}')


# One download at a time; independent of _transition_lock (disk/network, not GPU) so a pull can
# run while a model serves. The pull thread releases _pull_lock when it finishes.
_pull_lock = threading.Lock()
_pull: dict = {"state": "idle", "model": None, "slug": None, "weight_key": None,
               "extra_keys": None, "model_dir": None, "started_ts": 0.0, "last_line": None, "error": None}


def _run_pull(slug: str) -> None:
    """Background: shell out to setup.sh (mirrors serve-cockpit) to fetch core + companions.
    Caller holds _pull_lock; this releases it when the download finishes."""
    try:
        entry = COMPOSE_REGISTRY[slug]
        model, core = entry["model"], entry["weights_variant"]
        comp = _companion_keys(entry)
        env = dict(os.environ)
        env["WEIGHT_KEY"] = f"{model}:{core}"
        env["MODEL_DIR"] = MODEL_DIR
        if comp:
            env["WEIGHT_EXTRA_KEYS"] = " ".join(comp)
        env.setdefault("HF_HOME", os.path.join(MODEL_DIR, ".cache", "huggingface"))
        _pull.update({"state": "downloading", "model": model, "slug": slug,
                      "weight_key": env["WEIGHT_KEY"], "extra_keys": env.get("WEIGHT_EXTRA_KEYS"),
                      "model_dir": MODEL_DIR, "started_ts": time.time(), "last_line": None, "error": None})
        try:
            p = subprocess.Popen(["bash", SETUP_SCRIPT, model], cwd=str(REPO_ROOT), env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:  # stream: keep last non-empty line as coarse progress
                ln = line.rstrip()
                if ln:
                    _pull["last_line"] = ln[-300:]
            rc = p.wait()
            if rc == 0:
                _pull["state"] = "ready"
            else:
                _pull["state"], _pull["error"] = "error", f"setup.sh exit {rc}: {_pull['last_line']}"
        except Exception as e:
            _pull["state"], _pull["error"] = "error", f"{type(e).__name__}: {e}"
    finally:
        _pull_lock.release()


# ---- switch primitive ----------------------------------------------------
def do_switch(slug: str, force: bool = False) -> dict:
    """Run switch.sh <slug>, blocking until ready. env FORCE=1 iff force. Raises SwitchError.

    Weights presence is checked HERE (the single choke point both perform_switch and the watchdog
    heal use) so a missing-weights model surfaces `weights_missing` instead of a doomed boot."""
    if not _weights_present(slug):
        raise _weights_missing_error(slug)
    entry = COMPOSE_REGISTRY[slug]
    port = os.environ.get("PORT") or str(entry["default_port"])
    env = dict(os.environ)
    env["READY_URL"] = f"http://localhost:{port}/health"  # unauth probe: survives VLLM_API_KEY
    env.setdefault("READY_TIMEOUT", str(SWITCH_TIMEOUT_S))
    if force:
        env["FORCE"] = "1"
    else:
        env.pop("FORCE", None)
    t0 = time.time()
    try:
        p = subprocess.run(
            ["bash", SWITCH_SCRIPT, slug], cwd=str(REPO_ROOT), env=env,
            capture_output=True, text=True, timeout=SWITCH_TIMEOUT_S + 60,
        )
    except subprocess.TimeoutExpired:
        raise SwitchError(500, "switch timed out", slug=slug)
    if p.returncode != 0:
        raise SwitchError(500, "switch failed", slug=slug, detail=(p.stderr or p.stdout or "")[-800:])
    return {"ok": True, "slug": slug, "model": model_of_slug(slug), "took_s": round(time.time() - t0, 1)}


def _reset_timers(launched: bool = False) -> None:
    if launched:
        _wd["launch_ts"] = time.time()
    _wd["down_since"] = 0.0
    _wd["healthy_streak_start"] = 0.0


# ---- switch + rollback (holds the lock) ----------------------------------
def perform_switch(target: str, force_requested: bool) -> tuple[dict, int]:
    if requires_force(target) and not force_requested:
        return ({"ok": False, "error": "slug requires explicit force consent",
                 "slug": target, "reason": _force_reason(target),
                 "hint": 'resend with {"force": true}'}, 400)

    obs = _observe()
    # no-op fast path: exact slug already healthy
    if obs["state"] == "healthy" and obs["slug"] == target:
        _set_state(desired_slug=target, force_authorized=force_requested, degraded=False, heal_history=[])
        _reset_timers(launched=True)
        return ({"ok": True, "slug": target, "model": model_of_slug(target),
                 "status": "already-running", "took_s": 0}, 200)

    rollback_to = obs["slug"] if obs["state"] == "healthy" else None
    rollback_force = _state.get("force_authorized", False) if rollback_to == _state.get("desired_slug") else False

    try:
        result = do_switch(target, force=force_requested)
    except SwitchError as e:
        if e.payload.get("error") == "weights_missing":
            # do_switch raised before any teardown — nothing to roll back; surface the pull hint.
            return ({"ok": False, **e.payload}, e.code)
        if rollback_to and rollback_to != target:
            try:
                do_switch(rollback_to, force=rollback_force)
                _set_state(desired_slug=rollback_to, force_authorized=rollback_force,
                           degraded=False, heal_history=[])
                _reset_timers(launched=True)
                return ({"ok": False, "error": str(e), "attempted": target,
                         "rolled_back_to": rollback_to, **e.payload}, e.code)
            except SwitchError as e2:
                return ({"ok": False, "error": str(e), "attempted": target,
                         "rollback_failed": str(e2), "detail": "rig may be empty", **e.payload}, e.code)
        return ({"ok": False, "error": str(e), "attempted": target, "rolled_back_to": None,
                 **e.payload}, e.code)

    _set_state(desired_slug=target, force_authorized=force_requested, degraded=False, heal_history=[])
    _reset_timers(launched=True)
    return (result, 200)


# ---- watchdog ------------------------------------------------------------
def _degrade(desired: str) -> None:
    """Caller holds the lock. Mark degraded + best-effort teardown to stop the crash-loop."""
    _set_state(degraded=True)
    _wd["last_action"], _wd["last_action_ts"] = f"degrade:{desired}", time.time()
    try:
        subprocess.run(["bash", SWITCH_SCRIPT, "--down"], cwd=str(REPO_ROOT),
                       env=dict(os.environ), capture_output=True, text=True, timeout=120)
    except Exception as e:  # teardown failure is logged, not retried (avoid a loop)
        _wd["last_error"] = f"teardown during degrade failed: {e}"


def _attempt_heal(desired: str, now: float) -> None:
    with _TryLock(_transition_lock) as got:
        if not got:
            return  # a user op or another heal is running; try next tick
        # Missing weights is permanent-until-pull, not a crash-loop: don't spend heal budget or
        # invoke switch.sh — degrade with the reason so /status shows it. Recover via /pull + /heal.
        if not _weights_present(desired):
            _set_state(degraded=True)
            _wd["last_action"], _wd["last_action_ts"] = f"blocked:weights_missing:{desired}", now
            _wd["last_error"] = f"weights_missing: {desired}"
            return
        recent = _pruned_heals(now)
        if len(recent) >= MAX_HEAL_FAILURES:
            _degrade(desired)
            return
        _set_state(heal_history=recent + [now])
        _wd["last_action"], _wd["last_action_ts"] = f"heal:{desired}", now
        try:
            do_switch(desired, force=_state.get("force_authorized", False))
            _reset_timers(launched=True)
        except SwitchError as e:
            _wd["last_error"] = f"heal failed: {e}"


def _watchdog_tick() -> None:
    desired = _state.get("desired_slug")
    if not desired or _state.get("degraded"):
        return
    obs = _observe()
    st, observed, now = obs["state"], obs["slug"], time.time()

    if st == "unknown":
        return  # docker outage: never take destructive action

    # Adopt any present container that isn't the desired one (external CLI switch, even
    # mid-boot). This makes the just-launched model the new intent, so we never restore
    # the old desired *under* a booting external launch.
    if observed and observed != desired:
        with _TryLock(_transition_lock) as got:
            if got:
                _set_state(desired_slug=observed, force_authorized=False)
                _reset_timers(launched=True)
        return

    if st == "healthy":
        if not _wd["healthy_streak_start"]:
            _wd["healthy_streak_start"] = now
        _wd["last_healthy_ts"], _wd["down_since"] = now, 0.0
        if now - _wd["healthy_streak_start"] >= STABILITY_WINDOW_S and _state.get("heal_history"):
            with _TryLock(_transition_lock) as got:
                if got:
                    _set_state(heal_history=[])
        return

    # st in {unready, absent} for the desired model
    _wd["healthy_streak_start"] = 0.0
    if not _wd["down_since"]:
        _wd["down_since"] = now
    booting = _wd["last_healthy_ts"] < _wd["launch_ts"]  # never healthy since last launch
    grace = BOOT_GRACE_S if booting else HEAL_GRACE_S
    anchor = _wd["launch_ts"] if booting else _wd["down_since"]
    if now - anchor >= grace:
        _attempt_heal(desired, now)


def _watchdog_safe_tick() -> None:
    """One watchdog tick with exception containment — a bad tick must never kill the thread."""
    try:
        _watchdog_tick()
    except Exception as e:
        _wd["last_error"] = f"watchdog tick: {type(e).__name__}: {e}"


def _watchdog_loop() -> None:
    while True:
        time.sleep(WATCH_INTERVAL_S)
        _watchdog_safe_tick()


# ---- payloads ------------------------------------------------------------
def status_payload() -> dict:
    obs = _observe()
    return {
        "current_slug": obs["slug"],
        "current_model": model_of_slug(obs["slug"]) if obs["slug"] else None,
        "desired_slug": _state.get("desired_slug"),
        "force_authorized": _state.get("force_authorized", False),
        "docker_state": obs["state"],
        "healthy": obs["state"] == "healthy",
        "port": obs["port"],
        "container": obs["container"],
        "restart_count": obs["restart_count"],
        "uptime_s": _uptime_s(obs["started_at"]),
        "gpu_count": _gpu_count(),
        "model_dir": MODEL_DIR,
        "download": dict(_pull),  # snapshot copy for a clean lock-free read
        "degraded": _state.get("degraded", False),
        "watchdog": {
            "enabled": WATCHDOG_ENABLED,
            "thread_alive": _wd_thread.is_alive() if _wd_thread else False,
            "heals_in_window": len(_pruned_heals()),
            "last_action": _wd["last_action"],
            "last_action_ts": _wd["last_action_ts"],
            "last_error": _wd["last_error"],
        },
    }


def models_payload(show_all: bool) -> dict:
    avail = []
    for slug, entry in sorted(COMPOSE_REGISTRY.items()):
        st = entry["status"]
        elig = _gpu_eligible(slug)
        row = {"slug": slug, "model": entry["model"], "status": st, "port": entry["default_port"],
               "topology": _raw_topology(slug), "gpu_eligible": elig,
               "requires_force": requires_force(slug),
               "recommended": st in FUNCTIONAL_STATUSES}
        if not show_all:
            if st in ("deprecated", "upstream-gated", "incubating"):
                continue
            if elig is False:  # None (unknown) stays visible under fail-open detection
                continue
        avail.append(row)
    return {"host_topology": _topology(), "gpu_count": _gpu_count(), "available": avail}


# ---- routes: single runtime source of truth for dispatch AND self-discovery ----
# Add/change an endpoint by editing ONE entry here (+ its h_* method). Dispatch, the 404
# set, the auth gate, and GET / all derive from this list, so they cannot drift apart.
ROUTES = [
    {"method": "GET", "path": "/healthz", "auth": False, "summary": "liveness",
     "handler": "h_healthz"},
    {"method": "GET", "path": "/", "auth": False, "summary": "this API manifest (self-discovery)",
     "handler": "h_discover"},
    {"method": "GET", "path": "/status", "auth": True,
     "summary": "current + desired model, health, watchdog", "handler": "h_status"},
    {"method": "GET", "path": "/models", "auth": True,
     "summary": "available models (+ recommended / status / requires_force / gpu_eligible)",
     "query": {"all": "1 -> include hidden (deprecated/upstream-gated/incubating) + GPU-ineligible"},
     "handler": "h_models"},
    {"method": "POST", "path": "/switch", "auth": True,
     "summary": "switch model; rolls back to the previous healthy model on failure",
     "body": {"slug": "registry slug (see /models)", "model": "model id -> its curated default slug",
              "force": "bool, optional; consent for non-production or GPU-oversized slugs"},
     "body_note": "provide at least one of slug|model; if both are given, slug wins (model ignored)",
     "handler": "h_switch"},
    {"method": "POST", "path": "/heal", "auth": True,
     "summary": "recover a model (or re-launch the current desired model)",
     "body": {"slug": "optional", "model": "optional", "force": "bool, optional"},
     "body_note": "omit slug+model to re-launch the current desired model; slug wins if both given",
     "handler": "h_heal"},
    {"method": "POST", "path": "/pull", "auth": True,
     "summary": "download a model's weights + companions (async; poll GET /status .download)",
     "body": {"slug": "registry slug (see /models)", "model": "model id (functional curated default only)"},
     "body_note": "slug|model like /switch, but {model} resolves ONLY to a functional curated "
                  "default — use slug for experimental/non-default entries. Returns 202 immediately; "
                  "poll GET /status .download for state (downloading|ready|error). Weights land under "
                  "MODEL_DIR (see /status.model_dir).",
     "handler": "h_pull"},
    {"method": "POST", "path": "/down", "auth": True,
     "summary": "stop the model + stand the watchdog down", "handler": "h_down"},
]


def discovery_payload() -> dict:
    """Self-description built from ROUTES — one call tells a client/agent the whole API.

    Per-route `auth` is the POLICY (protected when a token is configured); `auth.configured`
    reports whether auth is actually enforced right now (a token is set).
    """
    return {
        "service": "model-switch",
        "version": 1,
        "auth": {
            "scheme": "Bearer",
            "configured": bool(CONTROL_TOKEN),
            "token_env": list(TOKEN_ENV),
            "note": ("auth=true routes require the Bearer token WHEN configured; if neither env "
                     "var is set the service is unauthenticated (loopback only)"),
        },
        "endpoints": [{k: v for k, v in r.items() if k != "handler"} for r in ROUTES],
    }


# ---- HTTP handler --------------------------------------------------------
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
            return True
        got = self.headers.get("Authorization", "")
        pfx = "Bearer "
        return got.startswith(pfx) and hmac.compare_digest(got[len(pfx):], CONTROL_TOKEN)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _dispatch(self, method: str) -> None:
        route = self.path.split("?", 1)[0]
        entry = next((r for r in ROUTES if r["method"] == method and r["path"] == route), None)
        # Auth-first: unknown routes and auth:true routes require the token BEFORE we reveal a
        # 404. Only explicitly-open routes (/healthz, /) skip auth.
        if entry is None or entry["auth"]:
            if not self._authed():
                return self._json(401, {"error": "unauthorized"})
        if entry is None:
            return self._json(404, {"error": "not found"})
        getattr(self, entry["handler"])()

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    # -- handlers (one per ROUTES entry) --
    def h_healthz(self):
        self._json(200, {"ok": True})

    def h_discover(self):
        self._json(200, discovery_payload())

    def h_status(self):
        self._json(200, status_payload())

    def h_models(self):
        self._json(200, models_payload("all=1" in self.path))

    def h_switch(self):
        self._transition("/switch")

    def h_heal(self):
        self._transition("/heal")

    def h_down(self):
        self._handle_down()

    def h_pull(self):
        try:
            body = self._body()
        except Exception as e:
            return self._json(400, {"error": f"bad body: {e}"})
        try:
            target = resolve_slug(body)
        except SwitchError as e:
            return self._json(e.code, e.payload)
        if _weights_present(target):
            return self._json(200, {"state": "ready", "already": True, "slug": target,
                                    "model": model_of_slug(target), "model_dir": MODEL_DIR})
        # disk preflight (skip when total size is unknown/variable)
        need_gb = _slug_total_size_gb(target)
        free = _free_bytes(MODEL_DIR)
        if need_gb is not None and free is not None and free < need_gb * (1024 ** 3):
            return self._json(507, {"error": "insufficient_disk", "slug": target,
                                    "needed_gb": round(need_gb, 1), "free_gb": round(free / 1024 ** 3, 1),
                                    "model_dir": MODEL_DIR})
        if not _pull_lock.acquire(blocking=False):
            return self._json(409, {"error": "a download is already in progress",
                                    "download": dict(_pull)})
        started = False
        try:
            threading.Thread(target=_run_pull, args=(target,), name="model-switch-pull",
                             daemon=True).start()
            started = True
        finally:
            if not started:
                _pull_lock.release()
        self._json(202, {"state": "downloading", "slug": target, "model": model_of_slug(target),
                         "model_dir": MODEL_DIR, "needed_gb": need_gb})

    def _transition(self, route):
        try:
            body = self._body()
        except Exception as e:
            return self._json(400, {"error": f"bad body: {e}"})
        try:
            target = self._resolve_target(body, route)
        except SwitchError as e:
            return self._json(e.code, e.payload)
        force_requested = bool(isinstance(body, dict) and body.get("force"))
        with _TryLock(_transition_lock) as got:
            if not got:
                return self._json(409, {"error": "a transition is already in progress"})
            # /switch and /heal share one path: perform_switch clears degraded + heal_history on
            # every success, so a heal recovers a degraded service too. The only difference — heal
            # may omit a target and re-launch the desired model — is handled in _resolve_target.
            payload, code = perform_switch(target, force_requested)
        self._json(code, payload)

    def _resolve_target(self, body, route):
        # /heal with no slug/model -> recover the persisted desired
        if route == "/heal" and isinstance(body, dict) and not body.get("slug") and not body.get("model"):
            desired = _state.get("desired_slug")
            if not desired:
                raise SwitchError(400, "no desired model to heal; provide 'slug' or 'model'")
            return desired
        return resolve_slug(body)

    def _handle_down(self):
        with _TryLock(_transition_lock) as got:
            if not got:
                return self._json(409, {"error": "a transition is already in progress"})
            try:
                p = subprocess.run(["bash", SWITCH_SCRIPT, "--down"], cwd=str(REPO_ROOT),
                                   env=dict(os.environ), capture_output=True, text=True, timeout=180)
                if p.returncode != 0:
                    return self._json(500, {"ok": False, "error": "down failed",
                                            "detail": (p.stderr or p.stdout or "")[-400:]})
                _set_state(desired_slug=None, force_authorized=False, degraded=False)  # stand watchdog down
                return self._json(200, {"ok": True, "status": "down"})
            except Exception as e:
                return self._json(500, {"ok": False, "error": f"down failed: {e}"})

    def log_message(self, *a):
        pass


def main() -> None:
    global _wd_thread
    if not CONTROL_TOKEN and BIND not in LOOPBACK_HOSTS:
        raise SystemExit(
            f"model-switch: REFUSING to start — MODEL_SWITCH_BIND={BIND!r} is non-loopback and "
            "no CLUB3090_API_TOKEN/VLLM_API_KEY is set; that would expose the destructive "
            "/switch endpoint unauthenticated. Set a token, or bind to 127.0.0.1.")
    _load_state()
    # Adopt the currently-running model as desired if we have no persisted intent.
    if not _state.get("desired_slug"):
        obs = _observe()
        if obs["state"] == "healthy" and obs["slug"]:
            _state["desired_slug"] = obs["slug"]
            _save_state()
    if not CONTROL_TOKEN:
        print("model-switch: WARNING — no token set; control endpoint is UNAUTHENTICATED "
              "(loopback only).", flush=True)
    if WATCHDOG_ENABLED:
        _wd_thread = threading.Thread(target=_watchdog_loop, name="model-switch-watchdog", daemon=True)
        _wd_thread.start()
    srv = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"model-switch: serving on {BIND}:{PORT} (auth={'on' if CONTROL_TOKEN else 'OFF'}, "
          f"watchdog={'on' if WATCHDOG_ENABLED else 'off'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
