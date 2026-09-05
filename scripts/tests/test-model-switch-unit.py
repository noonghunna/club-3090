#!/usr/bin/env python3
"""Unit tests for the model-switch self-healing state machine.

Drives server.py's functions directly (monkeypatching `_observe` / `do_switch` / the
`--down` subprocess) so the watchdog, rollback, force-consent, crash-loop budget, and
persistence logic are tested DETERMINISTICALLY — no HTTP, no sleeps, no timing races.
The HTTP contract itself is covered by test-model-switch.sh.

Run: python3 scripts/tests/test-model-switch-unit.py   (invoked by test-model-switch.sh)
"""
import importlib.util
import os
import subprocess as _sp
import sys
import tempfile
from pathlib import Path

# Some tests stub `m.subprocess.run` — but that's the SHARED subprocess module, so the patch
# leaks across every later module (e.g. breaking weights.py list --json). Reset it per load().
_REAL_SP_RUN = _sp.run

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "tools" / "model-switch" / "server.py"
sys.path.insert(0, str(REPO))
import scripts.lib.profiles.compose_registry as reg  # noqa: E402

FAILS = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS.append(name)


def load(state_dir=None, **env):
    """Import a FRESH copy of server.py with a clean state dir + deterministic env."""
    _sp.run = _REAL_SP_RUN  # undo any prior test's global subprocess.run monkeypatch
    os.environ["MODEL_SWITCH_WATCHDOG"] = "0"
    os.environ["MODEL_SWITCH_GPU_COUNT"] = "2"
    os.environ["CLUB3090_TOPOLOGY"] = "dual"
    os.environ["MODEL_SWITCH_HEAL_GRACE_S"] = "0"
    os.environ["MODEL_SWITCH_BOOT_GRACE_S"] = "0"
    os.environ["MODEL_SWITCH_STABILITY_WINDOW_S"] = "1000"
    os.environ["MODEL_SWITCH_HEAL_BUDGET_WINDOW_S"] = "100000"
    os.environ["MODEL_SWITCH_MAX_HEAL_FAILURES"] = "3"
    os.environ["CLUB3090_API_TOKEN"] = ""   # hermetic auth: default no token (override via **env)
    os.environ["VLLM_API_KEY"] = ""
    os.environ.pop("MODEL_DIR", None)       # hermetic: resolve fresh (override via **env)
    os.environ.pop("SETUP_SCRIPT", None)
    os.environ["MODEL_SWITCH_STATE_DIR"] = state_dir or tempfile.mkdtemp()
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("ms_%d" % id(state_dir or object()), SERVER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Pick representative real slugs from the registry.
def _pick(m):
    func_dual = [s for s, e in reg.COMPOSE_REGISTRY.items()
                 if m._raw_topology(s) == "dual" and e["status"] in reg.FUNCTIONAL_STATUSES]
    exp_dual = [s for s, e in reg.COMPOSE_REGISTRY.items()
                if m._raw_topology(s) == "dual" and e["status"] not in reg.FUNCTIONAL_STATUSES]
    multi = [s for s in reg.COMPOSE_REGISTRY if (m._raw_topology(s) or "").startswith("multi")]
    return func_dual, exp_dual, multi


def obs(state, slug=None, port=8000):
    return {"state": state, "slug": slug, "container": "c", "port": port,
            "restart_count": 0, "started_at": None}


def recorder(m, fail_for=()):
    calls = []

    def do_switch(slug, force=False):
        calls.append((slug, force))
        if slug in fail_for:
            raise m.SwitchError(500, "switch failed", slug=slug)  # this module's class
        return {"ok": True, "slug": slug, "model": "m", "took_s": 0}

    return calls, do_switch


# --------------------------------------------------------------------------
def test_force_consent_and_propagation():
    m = load()
    func, exp, _ = _pick(m)
    A, E = func[0], (exp[0] if exp else None)
    m._observe = lambda: obs("absent")
    calls, m.do_switch = recorder(m)

    # production slug: no consent needed, no FORCE
    payload, code = m.perform_switch(A, force_requested=False)
    check("functional slug switches without force (200)", code == 200)
    check("functional slug not forced", (A, False) in calls)

    if E:
        # experimental slug: 400 without consent
        payload, code = m.perform_switch(E, force_requested=False)
        check("experimental slug needs consent (400)", code == 400 and "force" in payload.get("error", ""))
        # with consent: FORCE=1 propagated + persisted
        calls.clear()
        payload, code = m.perform_switch(E, force_requested=True)
        check("experimental slug switches with force (200)", code == 200)
        check("experimental slug forced (FORCE=1)", (E, True) in calls)
        check("force authorization persisted", m._state["force_authorized"] is True)


def test_gpu_eligibility():
    m = load()
    func, _, multi = _pick(m)
    check("dual slug eligible on 2 GPUs", m._gpu_eligible(func[0]) is True)
    check("dual slug not requires_force", m.requires_force(func[0]) is False)
    if multi:
        M = next((s for s in multi if (m._required_gpus(s) or 0) > 2), None)
        if M:
            check("multi>2 slug ineligible on 2 GPUs", m._gpu_eligible(M) is False)
            check("multi>2 slug requires_force", m.requires_force(M) is True)
    # fail-open: unknown gpu count -> eligible None, no forced consent on a functional slug
    m2 = load(MODEL_SWITCH_GPU_COUNT="")
    m2._gpu_count = lambda: None
    check("unknown gpu count -> eligible None (fail-open)", m2._gpu_eligible(func[0]) is None)


def test_rollback():
    m = load()
    func, _, _ = _pick(m)
    A, B = func[0], func[1]
    m._state["desired_slug"] = A
    m._observe = lambda: obs("healthy", slug=A)  # A currently healthy
    calls, m.do_switch = recorder(m, fail_for=(B,))
    payload, code = m.perform_switch(B, force_requested=False)
    check("failed switch returns 500", code == 500)
    check("rolled back to previous healthy slug", payload.get("rolled_back_to") == A)
    check("switch attempted B then restored A", [c[0] for c in calls] == [B, A])

    # rollback_failed: both fail -> honest 'rig may be empty'
    m._observe = lambda: obs("healthy", slug=A)
    _, m.do_switch = recorder(m, fail_for=(A, B))
    payload, code = m.perform_switch(B, force_requested=False)
    check("rollback_failed reported (may be empty)", "rollback_failed" in payload)


def test_models_hardware_filter():
    m = load()
    _, exp, multi = _pick(m)
    M = next((s for s in multi if (m._required_gpus(s) or 0) > 2), None)
    default = {r["slug"] for r in m.models_payload(False)["available"]}
    allrows = {r["slug"]: r for r in m.models_payload(True)["available"]}
    if M:
        check("multi>2 hidden from default /models", M not in default)
        check("multi>2 present under ?all=1", M in allrows)
        check("multi>2 flagged gpu_eligible false", allrows[M]["gpu_eligible"] is False)
    # incubating hidden by default
    inc = [s for s, e in reg.COMPOSE_REGISTRY.items() if e["status"] == "incubating"]
    if inc:
        check("incubating hidden from default /models", inc[0] not in default)


def test_slug_from_config_file_exact():
    m = load()
    func, _, _ = _pick(m)
    A = func[0]
    cp = reg.COMPOSE_REGISTRY[A]["compose_path"]
    abs_ok = str(REPO / cp)
    check("exact compose path resolves the slug", m._slug_from_config_file(abs_ok) == A)
    # a path that merely SHARES a suffix but isn't the repo file must NOT match
    check("non-repo suffix path does not match", m._slug_from_config_file("/somewhere/else/" + cp) is None)


def test_watchdog_adopt_external():
    m = load()
    func, _, _ = _pick(m)
    A, B = func[0], func[1]
    m._state["desired_slug"] = A
    m._observe = lambda: obs("healthy", slug=B)  # a DIFFERENT model is running (external switch)
    calls, m.do_switch = recorder(m)
    m._watchdog_tick()
    check("adopts externally-running slug as desired", m._state["desired_slug"] == B)
    check("adoption does not relaunch (no do_switch)", calls == [])


def test_watchdog_unknown_no_heal():
    m = load()
    func, _, _ = _pick(m)
    m._state["desired_slug"] = func[0]
    m._observe = lambda: obs("unknown")
    calls, m.do_switch = recorder(m)
    m._watchdog_tick()
    check("docker unknown -> no heal", calls == [])
    check("docker unknown -> not degraded", m._state["degraded"] is False)


def test_watchdog_heal_then_degrade():
    m = load()
    # This test is about the heal BUDGET, not about weights. do_switch() short-circuits on the
    # presence pre-check before any heal is attempted, so without this stub the assertions would
    # depend on the rig actually having A's weights under MODEL_DIR (passing here, failing in a
    # fresh clone / CI). weights_missing has its own coverage in
    # test_watchdog_weights_missing_degrades.
    m._weights_present = lambda slug: True
    func, _, _ = _pick(m)
    A = func[0]
    m._state["desired_slug"] = A
    m._wd["last_healthy_ts"] = 0.0
    m._wd["launch_ts"] = 0.0  # booting=False -> HEAL_GRACE(0)
    m._observe = lambda: obs("absent")           # desired is down
    calls, m.do_switch = recorder(m, fail_for=(A,))  # every heal fails
    downs = []
    _orig_run = m.subprocess.run

    def fake_run(cmd, **kw):
        if "--down" in cmd:
            downs.append(cmd)
            class R:  # noqa
                returncode = 0
                stdout = stderr = ""
            return R()
        return _orig_run(cmd, **kw)

    m.subprocess.run = fake_run
    for _ in range(3):
        m._watchdog_tick()
    check("heals up to the budget (3 attempts)", len(calls) == 3)
    check("not degraded before budget spent", m._state["degraded"] is False)
    m._watchdog_tick()  # 4th: budget spent
    check("degrades after budget spent", m._state["degraded"] is True)
    check("teardown (--down) called on degrade", len(downs) == 1)
    check("no 4th heal attempt", len(calls) == 3)


def test_watchdog_stability_reset():
    m = load()
    func, _, _ = _pick(m)
    A = func[0]
    m._state["desired_slug"] = A
    m._state["heal_history"] = [1.0, 2.0]
    m._observe = lambda: obs("healthy", slug=A)
    # STABILITY_WINDOW=1000 -> transient health must NOT reset the budget
    m._watchdog_tick()
    check("transient health keeps heal_history", m._state["heal_history"] == [1.0, 2.0])
    # window=0 -> health clears it
    m2 = load(MODEL_SWITCH_STABILITY_WINDOW_S="0")
    m2._state["desired_slug"] = A
    m2._state["heal_history"] = [1.0, 2.0]
    m2._observe = lambda: obs("healthy", slug=A)
    m2._watchdog_tick()
    check("stable health resets heal_history", m2._state["heal_history"] == [])


def test_persisted_heal_history_survives_restart():
    d = tempfile.mkdtemp()
    m = load(state_dir=d)
    m._weights_present = lambda slug: True   # budget persistence, not weights — see above
    func, _, _ = _pick(m)
    A = func[0]
    m._state["desired_slug"] = A
    m._wd["last_healthy_ts"] = m._wd["launch_ts"] = 0.0
    m._observe = lambda: obs("absent")
    _, m.do_switch = recorder(m, fail_for=(A,))
    m._watchdog_tick(); m._watchdog_tick()  # 2 failed heals, persisted
    check("2 heals recorded", len(m._state["heal_history"]) == 2)

    # "restart": fresh import, SAME state dir -> _load_state() (as main() does) must
    # rehydrate the budget so it survives across the daemon restart.
    m2 = load(state_dir=d)
    m2._weights_present = lambda slug: True
    m2._load_state()
    check("heal_history reloaded after restart", len(m2._state["heal_history"]) == 2)
    m2._state["desired_slug"] = A
    m2._wd["last_healthy_ts"] = m2._wd["launch_ts"] = 0.0
    m2._observe = lambda: obs("absent")
    calls, m2.do_switch = recorder(m2, fail_for=(A,))
    downs = []
    m2.subprocess.run = lambda cmd, **kw: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    m2._watchdog_tick()  # 3rd attempt (reaches budget=3)
    m2._watchdog_tick()  # budget spent -> degrade
    check("degrades at 3 across a restart (budget survived)", m2._state["degraded"] is True)


def test_state_load_corrupt_and_unknown():
    d = tempfile.mkdtemp()
    (Path(d) / "state.json").write_text("{ not json ]")
    m = load(state_dir=d)
    check("corrupt state -> desired None", m._state["desired_slug"] is None)
    d2 = tempfile.mkdtemp()
    (Path(d2) / "state.json").write_text('{"desired_slug": "__nope__", "force_authorized": true}')
    m2 = load(state_dir=d2)
    check("unknown persisted slug dropped", m2._state["desired_slug"] is None)
    check("force not carried for dropped slug", m2._state["force_authorized"] is False)


def test_routes_table_contract():
    m = load()
    expected = {("GET", "/healthz"), ("GET", "/"), ("GET", "/status"), ("GET", "/models"),
                ("POST", "/switch"), ("POST", "/heal"), ("POST", "/pull"), ("POST", "/down")}
    pairs = [(r["method"], r["path"]) for r in m.ROUTES]
    check("exactly the 8 expected (method,path) pairs", set(pairs) == expected)
    check("route pairs are unique (8)", len(pairs) == len(set(pairs)) == 8)
    check("every handler is a callable Handler method",
          all(callable(getattr(m.Handler, r["handler"], None)) for r in m.ROUTES))
    d = m.discovery_payload()
    check("no 'handler' key leaks into discovery endpoints",
          not any("handler" in e for e in d["endpoints"]))
    open_paths = {r["path"] for r in m.ROUTES if not r["auth"]}
    check("open (auth=False) routes are exactly /healthz and /", open_paths == {"/healthz", "/"})
    check("all other routes require auth",
          all(r["auth"] for r in m.ROUTES if r["path"] not in ("/healthz", "/")))
    # auth.configured reflects token presence — 3 hermetic cases (load() resets both vars)
    check("auth.configured False with no token", d["auth"]["configured"] is False)
    check("auth.configured True with CLUB3090_API_TOKEN",
          load(CLUB3090_API_TOKEN="tok1").discovery_payload()["auth"]["configured"] is True)
    check("auth.configured True via VLLM_API_KEY fallback",
          load(VLLM_API_KEY="tok2").discovery_payload()["auth"]["configured"] is True)


def test_models_recommended():
    m = load()
    rows = m.models_payload(True)["available"]  # ?all=1 -> includes lower-state entries
    bad = [r["slug"] for r in rows if r["recommended"] != (r["status"] in reg.FUNCTIONAL_STATUSES)]
    check("recommended == (status in FUNCTIONAL_STATUSES) for every row", bad == [])
    check("covers a recommended (production/caveats) row", any(r["recommended"] for r in rows))
    check("covers a non-recommended (experimental/…) row", any(not r["recommended"] for r in rows))


def _materialize(m, slug, model_dir, variants=None):
    """Create each artifact's subdir + a verify_glob-matching file (subset via `variants`)."""
    for a in m._slug_artifacts(slug):
        if not a["subdir"] or (variants is not None and a["variant"] not in variants):
            continue
        d = os.path.join(model_dir, a["subdir"])
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, (a["verify_glob"].replace("*", "x") or "x")), "w").close()


def test_model_dir_precedence():
    check("shell/systemd MODEL_DIR wins", load(MODEL_DIR="/tmp/msx-explicit").MODEL_DIR == "/tmp/msx-explicit")
    md = load().MODEL_DIR  # no env var -> repo .env's MODEL_DIR, else <repo>/models-cache
    check("fallback MODEL_DIR is a non-empty absolute path", bool(md) and os.path.isabs(md))


def test_weights_presence_and_companions():
    # Part A: a safetensors slug is always subdir-verifiable — guaranteed presence coverage.
    d = tempfile.mkdtemp()
    m = load(MODEL_DIR=d)
    check("safetensors slug absent -> not present", m._weights_present("vllm/dual") is False)
    _materialize(m, "vllm/dual", d)
    check("safetensors slug materialized -> present", m._weights_present("vllm/dual") is True)

    # Part B: companion counting — a slug whose core + companion are BOTH subdir-verifiable
    # (some GGUF cores aren't in the subdir index; skip cleanly if none qualify).
    def verifiable(sl):
        arts = m._slug_artifacts(sl)
        return len(arts) >= 2 and all(a["subdir"] for a in arts)
    cands = [s for s, e in reg.COMPOSE_REGISTRY.items() if e.get("weights_companions") and verifiable(s)]
    check("companions are modeled as extra artifacts",
          any(len(m._slug_artifacts(s)) >= 2 for s, e in reg.COMPOSE_REGISTRY.items() if e.get("weights_companions")))
    if not cands:
        check("(skip) no fully-subdir-verifiable companion slug in catalog", True)
        return
    d2 = tempfile.mkdtemp()
    m2 = load(MODEL_DIR=d2)
    s = cands[0]
    core = reg.COMPOSE_REGISTRY[s]["weights_variant"]
    check("nothing on disk -> not present", m2._weights_present(s) is False)
    _materialize(m2, s, d2, variants={core})  # core only
    check("core present but companion missing -> not present", m2._weights_present(s) is False)
    _materialize(m2, s, d2)  # core + companions
    check("core + companion present -> present", m2._weights_present(s) is True)


def test_do_switch_weights_missing_no_rollback():
    m = load()
    m._weights_present = lambda slug: False
    func = _pick(m)[0]
    A, B = func[0], func[1]
    raised = False
    try:
        m.do_switch(A)
    except m.SwitchError as e:
        raised = e.payload.get("error") == "weights_missing"
    check("do_switch raises weights_missing when absent", raised)
    # perform_switch uses the REAL do_switch, which short-circuits on the presence check before
    # switch.sh runs. A different model is healthy, so a *normal* failure would roll back — this
    # must NOT (nothing was torn down).
    m._observe = lambda: obs("healthy", slug=B)
    payload, code = m.perform_switch(A, force_requested=False)
    check("perform_switch surfaces weights_missing (409)", code == 409 and payload.get("error") == "weights_missing")
    check("no rollback on weights_missing", "rolled_back_to" not in payload and "rollback_failed" not in payload)


def test_watchdog_weights_missing_degrades():
    m = load()
    A = _pick(m)[0][0]
    m._state["desired_slug"] = A
    m._weights_present = lambda slug: False
    m._observe = lambda: obs("absent")
    m._wd["last_healthy_ts"] = m._wd["launch_ts"] = 0.0
    calls, m.do_switch = recorder(m)  # must NOT be called
    m._watchdog_tick()
    check("watchdog degrades on weights_missing", m._state["degraded"] is True)
    check("no heal budget burned / no switch attempted", m._state["heal_history"] == [] and calls == [])
    check("last_error notes weights_missing", "weights_missing" in (m._wd["last_error"] or ""))


def test_run_pull_env_and_state():
    d = tempfile.mkdtemp()
    envlog = os.path.join(d, "env.txt")
    stub = os.path.join(d, "setup.sh")
    with open(stub, "w") as f:
        f.write("#!/usr/bin/env bash\nenv > %r\nexit 0\n" % envlog)
    os.chmod(stub, 0o755)
    s = [sl for sl, e in reg.COMPOSE_REGISTRY.items() if e.get("weights_companions")][0]
    m = load(MODEL_DIR=d, SETUP_SCRIPT=stub)
    m._pull_lock.acquire()      # _run_pull assumes the caller holds it, then releases
    m._run_pull(s)
    check("pull reaches 'ready' after a clean setup.sh", m._pull["state"] == "ready")
    check("pull lock released after run", m._pull_lock.acquire(blocking=False))
    m._pull_lock.release()
    envtxt = open(envlog).read()
    check("child env has MODEL_DIR", ("MODEL_DIR=%s" % d) in envtxt)
    check("child env has WEIGHT_KEY", "\nWEIGHT_KEY=" in envtxt)
    check("child env has WEIGHT_EXTRA_KEYS (companion)", "\nWEIGHT_EXTRA_KEYS=" in envtxt)
    check("child env pins HF_HOME to the model disk",
          ("HF_HOME=%s" % os.path.join(d, ".cache", "huggingface")) in envtxt)


def test_disk_helpers():
    m = load()
    fb = m._free_bytes("/nonexistent/deeply/nested/path")
    check("_free_bytes walks up to an existing ancestor", isinstance(fb, int) and fb > 0)
    sz = m._slug_total_size_gb(_pick(m)[0][0])
    check("_slug_total_size_gb returns a number or None", sz is None or isinstance(sz, (int, float)))


def test_watchdog_exception_survives():
    m = load()
    m._state["desired_slug"] = _pick(m)[0][0]

    def boom():
        raise RuntimeError("kaboom")

    m._observe = boom
    m._watchdog_safe_tick()  # must NOT raise
    check("watchdog tick exception is contained", True)
    check("last_error records the exception", "kaboom" in (m._wd["last_error"] or ""))


def main():
    for fn in [
        test_force_consent_and_propagation, test_gpu_eligibility, test_rollback,
        test_models_hardware_filter, test_slug_from_config_file_exact,
        test_watchdog_adopt_external, test_watchdog_unknown_no_heal,
        test_watchdog_heal_then_degrade, test_watchdog_stability_reset,
        test_persisted_heal_history_survives_restart, test_state_load_corrupt_and_unknown,
        test_watchdog_exception_survives, test_routes_table_contract, test_models_recommended,
        test_model_dir_precedence, test_weights_presence_and_companions,
        test_do_switch_weights_missing_no_rollback, test_watchdog_weights_missing_degrades,
        test_run_pull_env_and_state, test_disk_helpers,
    ]:
        print(f"# {fn.__name__}")
        fn()
    print()
    if FAILS:
        print(f"test-model-switch-unit: FAIL ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("test-model-switch-unit: ok")


if __name__ == "__main__":
    main()
