"""c3's Launch settings form (club-3090#1465, phase 3d).

The form is the c3 twin of ``switch.sh --explain / --set / --unset``: it must
show the same value and the same source for every knob, refuse a value in the
resolver's own words, write only through the resolver and the per-slug store,
say that a change applies on the next launch, and mark the knobs a running
container was started with differently.

These tests drive the REAL resolver (``scripts/lib/launch_settings.py``) against
a fixture checkout (symlinks to this repo's scripts/ models/ tools/, plus an
empty legacy .env of its own) and a per-test settings store (conftest). Docker
is never touched: the running container comes from a FakeRunner. The one real
subprocess is ``switch.sh --explain`` (read-only) in the parity test.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from textual.widgets import DataTable, Static

from .test_app_headless import FakeRunner, _settle, fake_responses, make_app, ok

REPO = Path(__file__).resolve().parents[3]
SGL = "sgl/qwen38-27b-dual-fast"          # reads all six catalogued knobs, all via its env
VLL = "vllm/qwen38-27b-dual-fast"         # REASONING_EFFORT only on the command line
Q36 = "vllm/dual"                         # reads SPEC_N only
GLM = "llamacpp-club3090/glm53-flash-dual-iq3xxs-moecache"
NONE = "vllm/minimal"                     # reads no catalogued knob
SGL_COMPOSE = "models/qwen3.8-27b/sglang/compose/dual/autoround-int4/mtp.yml"
PIN = "CLUB3090_THINKING_QWEN3_8_27B"
KNOBS = ("KV_OFFLOAD_GB", "KV_OFFLOAD_DISK", "KV_OFFLOAD_DISK_GB", "ENABLE_THINKING",
         "REASONING_EFFORT", "SPEC_N")


@pytest.fixture(autouse=True)
def _clean_shell(monkeypatch):
    """Knobs, pins and the host-RAM seam come only from each test."""
    for k in (*KNOBS, "SPEC", PIN, "CLUB3090_MEMINFO_FILE", "HF_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    for k in list(os.environ):
        if k.startswith("CLUB3090_THINKING_"):
            monkeypatch.delenv(k, raising=False)


@pytest.fixture
def root(tmp_path) -> Path:
    """A checkout with this repo's code and its OWN (empty) legacy .env, so the
    maintainer's real .env never reaches either side of a comparison."""
    r = tmp_path / "root"
    r.mkdir()
    for d in ("scripts", "models", "tools"):
        (r / d).symlink_to(REPO / d)
    (r / ".env").write_text("", encoding="utf-8")
    return r


@pytest.fixture
def ls():
    from club3090_cockpit import launch_settings_store

    return launch_settings_store


def _cfg() -> Path:
    d = Path(os.environ["CLUB3090_CONFIG_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _files(club: str = "", secrets: str = "", slugs: dict | None = None) -> None:
    (_cfg() / "club3090.env").write_text(club, encoding="utf-8")
    (_cfg() / "secrets.env").write_text(secrets, encoding="utf-8")
    if slugs is not None:
        (_cfg() / "slugs.json").write_text(json.dumps({"version": 1, "slugs": slugs}), encoding="utf-8")


def _stored() -> dict:
    p = _cfg() / "slugs.json"
    return json.loads(p.read_text(encoding="utf-8"))["slugs"] if p.exists() else {}


def _mem(tmp_path: Path, gib: int) -> str:
    p = tmp_path / f"meminfo-{gib}"
    p.write_text(f"MemTotal:       {gib * 1024 * 1024} kB\nMemFree:        1 kB\n", encoding="utf-8")
    return str(p)


def _explain(root: Path, slug: str, extra_env: dict | None = None) -> dict:
    """`switch.sh --explain <slug> --json` of the fixture checkout — read-only
    (no docker, no compose; COMPOSE_BIN=: and a docker shim that refuses)."""
    shim = root.parent / "bin"
    shim.mkdir(exist_ok=True)
    (shim / "docker").write_text("#!/usr/bin/env bash\necho 'docker must not run' >&2\nexit 97\n",
                                 encoding="utf-8")
    (shim / "docker").chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if not k.endswith(("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "MASTER_KEY", "_KEY"))}
    env.update({"PATH": f"{shim}:{os.environ.get('PATH', '')}", "COMPOSE_BIN": ":",
                "CLUB3090_CARD": "rtx-3090", "PYTHONUTF8": "1"})
    env.update(extra_env or {})
    proc = subprocess.run(["bash", str(root / "scripts" / "switch.sh"), "--explain", slug, "--json"],
                          cwd=str(root), env=env, capture_output=True, text=True, encoding="utf-8",
                          timeout=120)
    assert proc.returncode == 0, proc.stderr[-800:]
    return json.loads(proc.stdout)["launch_settings"]


# ── the rows: values, sources, allowed values, refusals ─────────────────────


class TestRows:
    def test_every_layer_is_named_as_the_resolver_names_it(self, root, ls, monkeypatch):
        _files(club=f"KV_OFFLOAD_GB=32\nREASONING_EFFORT=medium\n{PIN}=off\n",
               secrets="SPEC_N=5\n", slugs={SGL: {"KV_OFFLOAD_GB": "64"}})
        monkeypatch.setenv("KV_OFFLOAD_DISK", "0")
        monkeypatch.setenv("CLUB3090_MEMINFO_FILE", "/nonexistent/meminfo")
        v = ls.view(SGL, root)
        assert v.available, v.reason
        got = {r.knob: (r.value, r.source) for r in v.knobs}
        assert got == {
            "ENABLE_THINKING": ("false", "model pin"),
            "KV_OFFLOAD_DISK": ("0", "shell"),
            "KV_OFFLOAD_DISK_GB": ("(unset)", "compose default"),
            "KV_OFFLOAD_GB": ("64", "this slug"),
            "REASONING_EFFORT": ("medium", "club3090.env"),
            "SPEC_N": ("<set, hidden>", "secrets.env"),
        }
        gb = v.knob("KV_OFFLOAD_GB")
        assert gb.overrides == [("club3090.env", "32")]
        assert gb.saved == "64"
        assert "at least 1.86" in gb.allowed
        assert v.knob("ENABLE_THINKING").detail == f"{PIN}=off from club3090.env"
        assert v.knob("REASONING_EFFORT").allowed == "low | medium | xhigh"
        assert v.knob("SPEC_N").secret and v.knob("REASONING_EFFORT").saved is None

    def test_a_refusal_is_filed_under_its_knobs(self, root, ls, monkeypatch):
        monkeypatch.setenv("KV_OFFLOAD_DISK", "1")          # needs KV_OFFLOAD_GB
        v = ls.view(SGL, root)
        msg = [e for e in v.errors if "needs KV_OFFLOAD_GB" in e]
        assert msg, v.errors
        assert v.knob("KV_OFFLOAD_DISK").errors == msg
        assert v.knob("KV_OFFLOAD_GB").errors == msg
        assert v.knob("SPEC_N").errors == []
        assert v.general_errors == []

    def test_only_the_knobs_the_slug_reads(self, root, ls):
        v = ls.view(Q36, root)
        assert [r.knob for r in v.knobs] == ["SPEC_N"]
        none = ls.view(NONE, root)
        assert none.available and none.knobs == []
        assert "KV_OFFLOAD_GB" in none.catalogued

    def test_an_unknown_slug_is_unavailable_not_a_crash(self, root, ls):
        v = ls.view("vllm/no-such-slug", root)
        assert not v.available and "unknown slug" in v.reason

    def test_what_c3_loaded_itself_is_not_the_shell(self, root, ls, monkeypatch):
        """c3 passes the keys it put in its own environment (switch.sh passes
        CLUB3090_CONFIG_SOURCE): such a key is its file's, not a shell export."""
        _files(club="SPEC_N=5\n")
        monkeypatch.setenv("SPEC_N", "5")
        assert ls.view(Q36, root).knob("SPEC_N").source == "shell"
        assert ls.view(Q36, root, loaded={"SPEC_N": "club3090.env"}).knob("SPEC_N").source == "club3090.env"

    def test_cockpit_reports_the_hf_token_it_applied(self, root, monkeypatch):
        from club3090_cockpit.services import CockpitData

        _files(secrets="HF_TOKEN=hf_fake_x\n")
        data = CockpitData(root, runner=FakeRunner({}))
        assert data.loaded_settings() == {}
        monkeypatch.setenv("HF_TOKEN", "hf_fake_x")
        data._hf_token_injected = True
        assert data.loaded_settings() == {"HF_TOKEN": "secrets.env"}
        monkeypatch.setenv("HF_TOKEN", "hf_fake_y")        # no longer the file's value
        assert data.loaded_settings() == {}


class TestParityWithSwitchExplain:
    """The form and `switch.sh --explain <slug>` agree value-for-value and
    source-for-source, with every layer in play."""

    @pytest.mark.parametrize("slug", [SGL, VLL, Q36, GLM, NONE])
    def test_rows_match_switch_explain(self, root, ls, monkeypatch, slug):
        _files(club=f"REASONING_EFFORT=xhigh\n{PIN}=off\nKV_OFFLOAD_DISK_GB=5\nSPEC_N=2\n",
               secrets="KV_OFFLOAD_GB=48\n",
               slugs={SGL: {"SPEC_N": "3", "REASONING_EFFORT": "medium"}, Q36: {"SPEC_N": "0"},
                      GLM: {"REASONING_EFFORT": "low"}})
        monkeypatch.setenv("KV_OFFLOAD_DISK", "1")
        monkeypatch.setenv("CLUB3090_MEMINFO_FILE", "/nonexistent/meminfo")
        cli = _explain(root, slug, {"KV_OFFLOAD_DISK": "1", "CLUB3090_MEMINFO_FILE": "/nonexistent/meminfo"})
        v = ls.view(slug, root)
        assert v.available and cli["available"]
        c3_rows = [(r.knob, r.value, r.source, r.detail, [list(o) for o in r.overrides], r.allowed, r.errors)
                   for r in v.knobs]
        cli_rows = [(k["knob"], k["value"], k["source"], k["detail"],
                     [[o["source"], o["value"]] for o in k["overrides"]], k["allowed"], k["errors"])
                    for k in cli["knobs"]]
        assert c3_rows == cli_rows
        assert (v.errors, v.warnings, v.unread, v.order) == (
            cli["errors"], cli["warnings"], cli["unread"], cli["order"])
        assert v.doc == cli
        if slug == SGL:                                  # the case is not trivially all-default
            assert {r.source for r in v.knobs} >= {"this slug", "model pin", "club3090.env",
                                                   "secrets.env", "shell"}

    def test_a_refusal_reads_like_switch_set(self, root, ls):
        """A value the form sends is refused with the sentence `switch.sh --set` prints."""
        ch = ls.save(SGL, {"REASONING_EFFORT": "high"}, root)
        assert not ch.ok and ch.changed == [] and _stored() == {}
        env = {k: v for k, v in os.environ.items() if not k.endswith(("TOKEN", "_KEY"))}
        env.update({"COMPOSE_BIN": ":", "PYTHONUTF8": "1"})
        proc = subprocess.run(["bash", str(root / "scripts" / "switch.sh"), "--set", SGL, "REASONING_EFFORT=high"],
                              cwd=str(root), env=env, capture_output=True, text=True, encoding="utf-8",
                              timeout=120)
        assert proc.returncode != 0
        assert ch.problems == ["REASONING_EFFORT=high: 'high' is not one of: low | medium | xhigh "
                               "(on sgl/qwen38-27b-dual-fast: the compose refuses it at boot)"]
        assert ch.problems[0] in proc.stderr


# ── writes: only through the resolver and the per-slug store ────────────────


class TestWrites:
    def test_save_then_remove_round_trip(self, root, ls, tmp_path, monkeypatch):
        monkeypatch.setenv("CLUB3090_MEMINFO_FILE", _mem(tmp_path, 256))
        ch = ls.save(SGL, {"KV_OFFLOAD_GB": "64"}, root)
        assert ch.ok and ch.changed == ["KV_OFFLOAD_GB"] and ch.path.endswith("slugs.json")
        assert _stored() == {SGL: {"KV_OFFLOAD_GB": "64"}}
        assert ls.view(SGL, root).knob("KV_OFFLOAD_GB").source == "this slug"
        ch = ls.remove(SGL, ["KV_OFFLOAD_GB"], root)
        assert ch.ok and ch.changed == ["KV_OFFLOAD_GB"] and _stored() == {}
        assert ls.view(SGL, root).knob("KV_OFFLOAD_GB").source == "compose default"

    def test_refusals_write_nothing(self, root, ls, tmp_path, monkeypatch):
        monkeypatch.setenv("CLUB3090_MEMINFO_FILE", _mem(tmp_path, 64))
        for slug, vals, want in (
            (Q36, {"KV_OFFLOAD_GB": "8"}, "vllm/dual doesn't read KV_OFFLOAD_GB"),
            (SGL, {"KV_OFFLOAD_GB": "48"}, "at most 31 GiB fits here"),
            (SGL, {"SPEC_N": "two"}, "'two' is not one of: a whole number"),
            (SGL, {"SPEC_N": "2", "REASONING_EFFORT": "max"}, "'max' is not one of"),
        ):
            ch = ls.save(slug, vals, root)
            assert not ch.ok and any(want in p for p in ch.problems), (vals, ch.problems)
        assert _stored() == {}

    def test_removing_nothing_saved_is_a_note_not_a_write(self, root, ls):
        ch = ls.remove(SGL, ["SPEC_N"], root)
        assert ch.ok and ch.changed == [] and ch.notes == [f"SPEC_N is not saved for {SGL} — nothing to remove"]
        assert not (_cfg() / "slugs.json").exists()

    def test_a_store_it_must_not_rewrite_is_an_error(self, root, ls):
        (_cfg() / "slugs.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ls.LaunchSettingsError, match="not valid JSON"):
            ls.save(SGL, {"SPEC_N": "2"}, root)
        v = ls.view(SGL, root)
        assert any("per-slug settings can't be read" in e for e in v.general_errors)


# ── the running container: drift, read-only ──────────────────────────────────


def _inspect_out(started: str, *entries: str) -> str:
    return "\n".join([json.dumps(started), *(json.dumps(e) for e in entries)]) + "\n"


class TestDrift:
    def test_containers_found_by_compose_label_from_any_checkout(self, ls):
        ps = ("sglang-qwen38-27b-mtp-dual\t/elsewhere/club-3090/" + SGL_COMPOSE + "\n"
              "other\t/elsewhere/club-3090/models/x/y.yml,/nonexistent/override.yml\n"
              "noise-without-label\t\n"
              "estate-copy\t/a/override.yml," + SGL_COMPOSE + "\n")
        assert ls.containers_for(SGL_COMPOSE, ps) == ["sglang-qwen38-27b-mtp-dual", "estate-copy"]

    def test_the_slug_label_decides_between_slugs_sharing_a_compose(self, ls):
        """vllm/dual and vllm/qwen-27b-dual-fast share one compose file; a container the
        launchers labelled (club3090.slug) belongs to the slug it names only. One
        without the label (started before it) still matches by compose."""
        compose = "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml"
        labelled = "vllm-qwen36-27b-dual\t/r/" + compose + ",/d/compose-labels/x.yml\tvllm/qwen-27b-dual-fast\n"
        assert ls.containers_for(compose, labelled, "vllm/qwen-27b-dual-fast") == ["vllm-qwen36-27b-dual"]
        assert ls.containers_for(compose, labelled, "vllm/dual") == []
        unlabelled = "vllm-qwen36-27b-dual\t/r/" + compose + "\t\n"
        assert ls.containers_for(compose, unlabelled, "vllm/dual") == ["vllm-qwen36-27b-dual"]
        assert ls.containers_for(compose, unlabelled, "vllm/qwen-27b-dual-fast") == ["vllm-qwen36-27b-dual"]

    def test_ps_format_carries_the_slug_label(self, ls):
        assert ls.PS_FORMAT.count("\t") == 2 and '"club3090.slug"' in ls.PS_FORMAT

    def test_inspect_template_names_only_the_knobs(self, ls):
        fmt = ls.inspect_format(["SPEC_N", "KV_OFFLOAD_GB"])
        assert '{{if eq (index $p 0) "SPEC_N" "KV_OFFLOAD_GB"}}' in fmt
        assert fmt.startswith("{{json .State.StartedAt}}")

    def test_parse_inspect(self, ls):
        started, env = ls.parse_inspect(_inspect_out("2026-09-28T18:02:42Z", "SPEC_N=3", "KV_OFFLOAD_GB",
                                                     "REASONING_EFFORT=", "X=a=b"))
        assert started == "2026-09-28T18:02:42Z"
        assert env == {"SPEC_N": "3", "KV_OFFLOAD_GB": None, "REASONING_EFFORT": "", "X": "a=b"}

    def test_each_knob_compared_where_the_truth_is_readable(self, root, ls):
        _files(slugs={SGL: {"SPEC_N": "2", "REASONING_EFFORT": "medium"}})
        v = ls.view(SGL, root)
        rc = ls.running_state(v, "c", _inspect_out(
            "t", "SPEC_N=2", "REASONING_EFFORT", "ENABLE_THINKING=true", "KV_OFFLOAD_GB=64"))
        st = {k: (r.status, r.shown) for k, r in rc.knobs.items()}
        assert st["SPEC_N"] == ("same", "2")
        assert st["REASONING_EFFORT"] == ("differs", "low")          # started unset → compose default low
        assert st["ENABLE_THINKING"] == ("same", "true")             # explicit value == compose default
        assert st["KV_OFFLOAD_GB"] == ("differs", "64")
        assert st["KV_OFFLOAD_DISK"] == ("same", "0")                # bare knob left out == unset
        assert rc.differing == ["KV_OFFLOAD_GB", "REASONING_EFFORT"]

    def test_command_line_knobs_are_not_guessed(self, root, ls):
        v = ls.view(VLL, root)
        eff = v.knob("REASONING_EFFORT")
        assert not eff.in_container_env
        rc = ls.running_state(v, "c", _inspect_out("t", "SPEC_N=4", "ENABLE_THINKING=true",
                                                   "KV_OFFLOAD_GB=", "KV_OFFLOAD_DISK="))
        assert rc.knobs["REASONING_EFFORT"].status == "unchecked"
        assert "command line" in rc.knobs["REASONING_EFFORT"].why
        assert rc.differing == []
        # A non-bare forwarded knob missing from the env: another compose version.
        rc = ls.running_state(v, "c", _inspect_out("t", "ENABLE_THINKING=true"))
        assert rc.knobs["SPEC_N"].status == "unknown"

    def test_a_secret_value_is_compared_but_never_shown(self, root, ls):
        _files(secrets="SPEC_N=909091\n")
        v = ls.view(SGL, root)
        rc = ls.running_state(v, "c", _inspect_out("t", "SPEC_N=909092"))
        r = rc.knobs["SPEC_N"]
        assert r.status == "differs" and "909091" not in r.why + r.shown and "909092" not in r.why + r.shown

    @pytest.mark.asyncio
    async def test_cockpit_reads_the_running_container_read_only(self, root):
        from club3090_cockpit.services import CockpitData

        _files(slugs={SGL: {"SPEC_N": "2"}})
        runner = FakeRunner({
            "docker ps": ok(f"sglang-qwen38-27b-mtp-dual\t/somewhere/{SGL_COMPOSE}\n"),
            "docker inspect": ok(_inspect_out("2026-09-28T18:02:42.18Z", "SPEC_N", "ENABLE_THINKING")),
        })
        data = CockpitData(root, runner=runner)
        v = await data.launch_settings(SGL)
        assert [rc.name for rc in v.running] == ["sglang-qwen38-27b-mtp-dual"]
        assert v.running[0].differing == ["SPEC_N"]
        inspect = [c for c in runner.calls if c[:2] == ["docker", "inspect"]]
        assert len(inspect) == 1
        fmt = inspect[0][inspect[0].index("--format") + 1]
        assert "{{range .Config.Env}}" in fmt and '"SPEC_N"' in fmt and "HF_TOKEN" not in fmt
        assert all(c[:2] in (["docker", "ps"], ["docker", "inspect"]) for c in runner.calls)


# ── the form (Textual pilot) ─────────────────────────────────────────────────


def _registry(*slugs: str) -> str:
    rows = {
        SGL: {"slug": SGL, "switch_engine": "sgl", "launch_engine": "sgl",
              "compose_dir": "models/qwen3.8-27b/sglang/compose", "file": "dual/autoround-int4/mtp.yml",
              "port": 8142, "model": "qwen3.8-27b", "engine": "sglang-stable", "kvcalc_key": "SKIP",
              "container": "sglang-qwen38-27b-mtp-dual", "compose_path": SGL_COMPOSE,
              "status": "production", "ctx_label": "262K", "configured_ctx": 262144,
              "status_note": "", "source": "curated"},
        NONE: {"slug": NONE, "switch_engine": "vllm", "launch_engine": "vllm",
               "compose_dir": "models/qwen3.6-27b/vllm/compose", "file": "single/autoround-int4/minimal.yml",
               "port": 8020, "model": "qwen3.6-27b", "engine": "vllm-stable",
               "kvcalc_key": "qwen3.6-27b:minimal", "container": "vllm-qwen36-27b-minimal",
               "compose_path": "models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml",
               "status": "production", "ctx_label": "33K", "configured_ctx": 32768,
               "status_note": "", "source": "curated"},
    }
    return json.dumps({"defaults": [], "profiles": {}, "variants": [rows[s] for s in slugs]})


async def _open_form(pilot):
    from club3090_cockpit.app import LaunchSettingsScreen

    await _settle(pilot)
    await pilot.press("E")
    await _settle(pilot)
    await _settle(pilot)
    assert isinstance(pilot.app.screen, LaunchSettingsScreen)
    return pilot.app.screen


def _text(screen, wid: str) -> str:
    return str(screen.query_one(wid, Static).render())


def _table_rows(screen) -> dict[str, list[str]]:
    t = screen.query_one("#ls-table", DataTable)
    return {str(t.get_row_at(i)[0]).split(" ")[0]: [str(c) for c in t.get_row_at(i)] for i in range(t.row_count)}


def _form_app(root, *slugs, running: str = "", inspect: str = ""):
    """The app on the fixture checkout. FakeRunner answers the FIRST key found in
    a command, so the form's own docker reads go first — keyed on text only they
    carry (the compose label, the knob filter)."""
    responses = {}
    if running:
        responses["com.docker.compose.project.config_files"] = ok(running)
    if inspect:
        responses["{{range .Config.Env}}"] = ok(inspect)
    responses.update(fake_responses(**{"registry-emit.sh --json": ok(_registry(*slugs))}))
    return make_app(repo_root=root, responses=responses)


class TestForm:
    @pytest.mark.asyncio
    async def test_E_opens_the_form_with_values_sources_and_the_next_launch_note(self, root):
        _files(club="REASONING_EFFORT=medium\n", slugs={SGL: {"SPEC_N": "2"}})
        app, runner, writes = _form_app(root, SGL)
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _open_form(pilot)
            assert "Changes apply on the next launch of this slug" in _text(screen, "#ls-banner")
            rows = _table_rows(screen)
            assert sorted(rows) == sorted(KNOBS)
            assert rows["SPEC_N"][1:3] == ["2", "this slug"]
            assert rows["REASONING_EFFORT"][1:4] == ["medium", "club3090.env", ""]
            assert rows["REASONING_EFFORT"][4] == "low | medium | xhigh"
            assert "Not running" in _text(screen, "#ls-running")
            hints = _text(screen, "#ls-hints")
            assert "shell > this slug > model pin > club3090.env" in hints
            assert "settings.sh set KEY=VALUE" in hints
        assert writes.started == []

    @pytest.mark.asyncio
    async def test_a_value_is_refused_in_the_resolvers_words_then_saved(self, root):
        app, _, writes = _form_app(root, SGL)
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _open_form(pilot)
            table = screen.query_one("#ls-table", DataTable)
            table.move_cursor(row=[r.knob for r in screen._view.knobs].index("REASONING_EFFORT"))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press(*"high", "enter")
            await _settle(pilot)
            await _settle(pilot)
            notes = _text(screen, "#ls-notes")
            assert "Not saved" in notes and "'high' is not one of: low | medium | xhigh" in notes
            assert _stored() == {}
            table.move_cursor(row=[r.knob for r in screen._view.knobs].index("REASONING_EFFORT"))
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press(*"xhigh", "enter")
            await _settle(pilot)
            await _settle(pilot)
            assert _stored() == {SGL: {"REASONING_EFFORT": "xhigh"}}
            assert _table_rows(screen)["REASONING_EFFORT"][1:3] == ["xhigh", "this slug"]
            assert "applies from the next launch" in _text(screen, "#ls-notes")
        assert writes.started == []

    @pytest.mark.asyncio
    async def test_x_removes_this_slugs_value(self, root):
        _files(club="SPEC_N=1\n", slugs={SGL: {"SPEC_N": "2"}})
        app, _, _ = _form_app(root, SGL)
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _open_form(pilot)
            screen.query_one("#ls-table", DataTable).move_cursor(
                row=[r.knob for r in screen._view.knobs].index("SPEC_N"))
            await pilot.pause()
            await pilot.press("x")
            await _settle(pilot)
            await _settle(pilot)
            assert _stored() == {}
            assert _table_rows(screen)["SPEC_N"][1:3] == ["1", "club3090.env"]

    @pytest.mark.asyncio
    async def test_a_dependency_refusal_shows_inline(self, root):
        _files(slugs={SGL: {"KV_OFFLOAD_DISK": "1"}})
        app, _, _ = _form_app(root, SGL)
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _open_form(pilot)
            rows = _table_rows(screen)
            assert rows["KV_OFFLOAD_DISK"][0].endswith("✗") and rows["KV_OFFLOAD_GB"][0].endswith("✗")
            assert "✗" not in rows["SPEC_N"][0]
            notes = _text(screen, "#ls-notes")
            assert "would be REFUSED" in notes and "KV_OFFLOAD_DISK=1 needs KV_OFFLOAD_GB set" in notes

    @pytest.mark.asyncio
    async def test_a_running_container_that_differs_is_marked(self, root):
        _files(slugs={SGL: {"SPEC_N": "2"}})
        app, _, _ = _form_app(root, SGL,
                              running=f"sglang-qwen38-27b-mtp-dual\t/elsewhere/{SGL_COMPOSE}\n",
                              inspect=_inspect_out("2026-09-28T18:02:42.18Z", "SPEC_N", "ENABLE_THINKING"))
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _open_form(pilot)
            running = _text(screen, "#ls-running")
            assert "sglang-qwen38-27b-mtp-dual" in running
            assert "1 setting differs from the next launch" in running and "SPEC_N" in running
            rows = _table_rows(screen)
            assert rows["SPEC_N"][3] == "≠ 4"          # started unset → its compose default 4
            assert rows["ENABLE_THINKING"][3] == "= true"

    @pytest.mark.asyncio
    async def test_a_slug_that_reads_no_knob_says_so(self, root):
        app, _, _ = _form_app(root, NONE)
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _open_form(pilot)
            assert screen.query_one("#ls-table", DataTable).row_count == 0
            assert "reads none of the catalogued launch settings" in _text(screen, "#ls-notes")

    @pytest.mark.asyncio
    async def test_escape_leaves_an_edit_then_closes(self, root):
        from club3090_cockpit.app import LaunchSettingsScreen

        app, _, _ = _form_app(root, SGL)
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _open_form(pilot)
            await pilot.press("enter")
            await pilot.pause()
            assert screen._editing
            await pilot.press("escape")
            await pilot.pause()
            assert not screen._editing and isinstance(app.screen, LaunchSettingsScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, LaunchSettingsScreen)
        assert _stored() == {}
