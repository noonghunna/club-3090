"""c3 reads and writes settings through the ONE loader (club-3090#1466).

c3 used to read the repo-root .env with private parsers, rewrite it in place,
and keep its own model dir + HF token in c3-settings.json that switch.sh never
saw. These tests drive c3's real code paths against a temporary settings store
(``CLUB3090_CONFIG_DIR`` → a per-test tmp dir, set by the autouse fixture in
conftest.py) and a temporary repo root:

  * a value that exists ONLY in club3090.env / secrets.env reaches c3;
  * c3's writes land in the store, and the repo .env is left alone;
  * c3-settings.json's model_dir / hf_token move into the store once, and never
    over a value the store already has;
  * `docker compose` gets the resolved settings as a temp env file that is
    removed after the run;
  * no token reaches c3's logs or notifications.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

import pytest

from club3090_cockpit import __main__ as M
from club3090_cockpit.services import MODEL_DIR as DEFAULT_MODEL_DIR
from club3090_cockpit.services import CockpitData
from club3090_tui_core.runner import CoreRunState

from .test_app_headless import make_app
from .test_services import _seed_service_dirs, full_runner

PIN = "CLUB3090_THINKING_QWEN3_8_27B"


@pytest.fixture(autouse=True)
def _clean_shell(monkeypatch):
    """The keys under test must come from the files, not the developer's shell."""
    for k in ("MODEL_DIR", "HF_TOKEN", "LANIP", "STUDIO_DIRECTOR_DEVICE", PIN, "C3_LOG"):
        monkeypatch.delenv(k, raising=False)


def _cfg() -> Path:
    return Path(os.environ["CLUB3090_CONFIG_DIR"])


def _write(name: str, text: str) -> Path:
    p = _cfg() / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _read(name: str) -> str:
    p = _cfg() / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir(exist_ok=True)
    return r


def _record_notifications(app) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def notify(message, *, title="", severity="information", timeout=None, **_kw):
        seen.append({"message": str(message), "title": title, "severity": severity})

    app.notify = notify  # type: ignore[method-assign]
    return seen


class _CaptureDownloads:
    def __init__(self):
        self.env = None

    def set_callbacks(self, **_kw):
        pass

    async def start_raw(self, cmd, env=None, run_type=None, parser=None):
        self.env = env
        st = CoreRunState(run_type=run_type or "download", started=time.time())
        st.done.set()
        return st


def _thinking_modal(repo_root):
    from club3090_cockpit.app import ConfirmActionScreen, ServeContext
    from club3090_cockpit.data import ActionPlan, CatalogEntry
    from club3090_cockpit.services import _variant_row_from_dict

    row = _variant_row_from_dict({
        "slug": "vllm/qwen38-27b-dual-max", "port": 8010, "model": "qwen3.8-27b",
        "sampler_profiles": {
            "instruct": {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5},
            "thinking": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
        },
    })
    m = ConfirmActionScreen.__new__(ConfirmActionScreen)
    m._plan = ActionPlan(kind="serve", cmd=["bash", "scripts/switch.sh", "vllm/qwen38-27b-dual-max"])
    m._serve_ctx = ServeContext(mode="start", entry=CatalogEntry(row=row))
    m._repo_root = repo_root
    m._act8_gate = None
    m._act8_on = False
    m._thinking = "inherit"
    m._sampler_reset = False
    m._reconcile = None
    return m


# ── reads ────────────────────────────────────────────────────────────────────


class TestReadsGoThroughTheLoader:
    @pytest.mark.asyncio
    async def test_store_only_values_reach_c3(self, tmp_path, monkeypatch):
        """Values that exist ONLY in the store (no repo .env, nothing in the
        shell) are what c3 uses — the weights root, the director placement, the
        LAN IP, the HF token in the download env and the preflight."""
        repo = _repo(tmp_path)
        _write("club3090.env", "MODEL_DIR=/store/models\nSTUDIO_DIRECTOR_DEVICE=cpu\nLANIP=10.1.2.3\n")
        _write("secrets.env", "HF_TOKEN=hf_store_only_tok\n")
        cap = _CaptureDownloads()
        cd = CockpitData(repo, runner=full_runner(), download_runner=cap)

        assert cd.weights_model_dir() == "/store/models"
        assert cd.director_device() == "cpu"
        assert cd.director_compose_env()["DIRECTOR_NGL"] == "0"
        assert cd.lan_ip() == "10.1.2.3"

        await cd.run_weights_download("qwen3.6-27b", "fp8")
        assert cap.env["MODEL_DIR"] == "/store/models"
        assert cap.env["HF_TOKEN"] == "hf_store_only_tok"

        monkeypatch.delenv("C3_SKIP_DOWNLOAD_PREFLIGHT", raising=False)
        monkeypatch.setattr(CockpitData, "_hf_cli_present", lambda self: True)
        monkeypatch.setattr(CockpitData, "_hf_token_file", lambda self: tmp_path / "no-token")
        _blockers, notes = cd.download_preflight()
        assert not any("no HF token" in n for n in notes), notes

    def test_shell_wins_over_the_store(self, tmp_path, monkeypatch):
        repo = _repo(tmp_path)
        _write("club3090.env", "MODEL_DIR=/store/models\nSTUDIO_DIRECTOR_DEVICE=cpu\n")
        monkeypatch.setenv("MODEL_DIR", "/shell/models")
        monkeypatch.setenv("STUDIO_DIRECTOR_DEVICE", "gpu1")
        cd = CockpitData(repo, runner=full_runner())
        assert cd.weights_model_dir() == "/shell/models"
        assert cd.director_device() == "gpu1"

    def test_store_wins_over_legacy_repo_env_which_is_still_read(self, tmp_path):
        repo = _repo(tmp_path)
        (repo / ".env").write_text("MODEL_DIR=/legacy/models\nSTUDIO_DIRECTOR_DEVICE=gpu1\n", encoding="utf-8")
        cd = CockpitData(repo, runner=full_runner())
        assert cd.weights_model_dir() == "/legacy/models"      # legacy fallback still works
        assert cd.director_device() == "gpu1"
        _write("club3090.env", "MODEL_DIR=/store/models\n")
        assert cd.weights_model_dir() == "/store/models"       # the store beats it
        assert cd.director_device() == "gpu1"

    def test_nothing_configured_uses_the_bundled_default(self, tmp_path):
        assert CockpitData(_repo(tmp_path), runner=full_runner()).weights_model_dir() == DEFAULT_MODEL_DIR

    def test_thinking_pin_is_read_from_the_store(self, tmp_path):
        repo = _repo(tmp_path)
        (repo / ".env").write_text(f"{PIN}=on\n", encoding="utf-8")
        _write("club3090.env", f"{PIN}=off\n")
        m = _thinking_modal(repo)
        assert m._thinking_persisted() == "off"
        assert f"persisted default: off ({PIN})" in "\n".join(m._thinking_card_lines())


# ── writes ───────────────────────────────────────────────────────────────────


class TestWritesGoToTheStore:
    def test_thinking_pin_write_lands_in_the_store(self, tmp_path):
        """[T] writes the pin to club3090.env; the repo .env (which still holds
        an old pin) is not touched, and the store's value is what applies."""
        repo = _repo(tmp_path)
        legacy = f"KEEP=1\nexport {PIN}=on\n"
        (repo / ".env").write_text(legacy, encoding="utf-8")
        m = _thinking_modal(repo)
        m.action_cycle_thinking()      # → on
        m.action_cycle_thinking()      # → off
        m.action_persist_thinking()
        assert f"{PIN}=off\n" in _read("club3090.env")
        assert (repo / ".env").read_text(encoding="utf-8") == legacy
        assert m._thinking_persisted() == "off"
        m.action_cycle_thinking()      # → inherit
        m.action_cycle_thinking()      # → on
        m.action_persist_thinking()    # upsert, no duplicate
        assert _read("club3090.env").count(f"{PIN}=") == 1
        assert f"{PIN}=on\n" in _read("club3090.env")

    @pytest.mark.asyncio
    async def test_settings_save_to_club3090_env_and_secrets_env(self, tmp_path):
        repo = _repo(tmp_path)
        app, _, _ = make_app()
        app._data.repo_root = repo
        seen = _record_notifications(app)
        async with app.run_test(size=(120, 40)):
            app.apply_settings(model_dir="/new/models", hf_token="hf_new_tok_1", director_device="gpu1")
            glob_env, secrets = _read("club3090.env"), _read("secrets.env")
            assert "MODEL_DIR=/new/models\n" in glob_env
            assert "STUDIO_DIRECTOR_DEVICE=gpu1\n" in glob_env
            assert "HF_TOKEN" not in glob_env
            assert "HF_TOKEN=hf_new_tok_1\n" in secrets
            assert stat.S_IMODE((_cfg() / "secrets.env").stat().st_mode) == 0o600
            assert not (repo / ".env").exists()                 # nothing written into the checkout
            saved = M.load_settings()
            assert "model_dir" not in saved and "hf_token" not in saved
            assert app._data.weights_model_dir() == "/new/models"
            assert app._data.director_device() == "gpu1"
            assert os.environ.get("HF_TOKEN") == "hf_new_tok_1"
        assert not any("hf_new_tok_1" in n["message"] for n in seen)

    @pytest.mark.asyncio
    async def test_refused_values_are_reported_and_not_written(self, tmp_path):
        """The writer refuses values bash / docker compose / systemd would read
        differently. c3 says which setting and why — never echoing a token —
        and keeps the current value."""
        repo = _repo(tmp_path)
        _write("club3090.env", "MODEL_DIR=/good/models\n")
        app, _, _ = make_app()
        app._data.repo_root = repo
        seen = _record_notifications(app)
        async with app.run_test(size=(120, 40)):
            app.apply_settings(model_dir="/models/$USER", hf_token="hf_bad'tok")
        assert _read("club3090.env") == "MODEL_DIR=/good/models\n"
        assert _read("secrets.env") == ""
        assert app._data.weights_model_dir() == "/good/models"
        errors = [n for n in seen if n["severity"] == "error"]
        assert errors, seen
        text = " ".join(n["message"] for n in seen)
        assert "MODEL_DIR" in text and "HF_TOKEN" in text
        assert "hf_bad" not in text
        assert os.environ.get("HF_TOKEN") is None

    def test_a_shell_token_stays_in_effect_and_c3_says_so(self, tmp_path, monkeypatch):
        repo = _repo(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "hf_shell_tok_1")
        app, _, _ = make_app()
        app._data.repo_root = repo
        seen = _record_notifications(app)
        app.apply_settings(model_dir="", hf_token="hf_saved_tok_1")
        assert "HF_TOKEN=hf_saved_tok_1\n" in _read("secrets.env")
        assert os.environ["HF_TOKEN"] == "hf_shell_tok_1"
        text = " ".join(n["message"] for n in seen)
        assert "shell" in text
        assert "hf_shell_tok_1" not in text and "hf_saved_tok_1" not in text


# ── the one-time fold-in of c3-settings.json ────────────────────────────────


class TestFoldIn:
    def _launch(self, repo, environ=None):
        app, _, _ = make_app()
        app._data.repo_root = repo
        env = {} if environ is None else environ
        M.apply_persisted_settings(app, env)
        return app, env

    def test_copies_once_and_keeps_c3_settings_json(self, tmp_path):
        repo = _repo(tmp_path)
        M.save_settings({"model_dir": "/old/models", "hf_token": "hf_old_tok_1", "catalog_sort": "name"})
        app, env = self._launch(repo)
        assert "MODEL_DIR=/old/models\n" in _read("club3090.env")
        assert "HF_TOKEN=hf_old_tok_1\n" in _read("secrets.env")
        assert stat.S_IMODE((_cfg() / "secrets.env").stat().st_mode) == 0o600
        assert app._data.weights_model_dir() == "/old/models"
        assert env.get("HF_TOKEN") == "hf_old_tok_1"
        notices = app._startup_notices
        assert len(notices) == 2 and all(sev == "information" for sev, _ in notices)
        assert not any("hf_old_tok_1" in msg for _, msg in notices)
        kept = M.load_settings()
        assert kept["model_dir"] == "/old/models" and kept["hf_token"] == "hf_old_tok_1"
        assert kept["catalog_sort"] == "name"
        # Second launch: nothing to move, nothing to say, the store is unchanged.
        before = (_read("club3090.env"), _read("secrets.env"))
        app2, _ = self._launch(repo)
        assert app2._startup_notices == []
        assert (_read("club3090.env"), _read("secrets.env")) == before

    def test_never_clobbers_the_store_and_says_so_once(self, tmp_path):
        repo = _repo(tmp_path)
        _write("club3090.env", "MODEL_DIR=/store/models\n")
        _write("secrets.env", "HF_TOKEN=hf_store_tok_1\n")
        M.save_settings({"model_dir": "/old/models", "hf_token": "hf_old_tok_2"})
        app, env = self._launch(repo)
        assert _read("club3090.env") == "MODEL_DIR=/store/models\n"
        assert _read("secrets.env") == "HF_TOKEN=hf_store_tok_1\n"
        assert app._data.weights_model_dir() == "/store/models"
        assert env.get("HF_TOKEN") == "hf_store_tok_1"
        msgs = [msg for sev, msg in app._startup_notices if sev == "warning"]
        assert len(msgs) == 2
        assert any("/store/models" in m for m in msgs)
        assert not any("hf_old_tok_2" in m or "hf_store_tok_1" in m for m in msgs)
        app2, _ = self._launch(repo)
        assert app2._startup_notices == []

    def test_a_legacy_repo_env_value_counts_as_the_store(self, tmp_path):
        repo = _repo(tmp_path)
        (repo / ".env").write_text("MODEL_DIR=/legacy/models\n", encoding="utf-8")
        M.save_settings({"model_dir": "/old/models"})
        app, _ = self._launch(repo)
        assert "MODEL_DIR" not in _read("club3090.env")
        assert app._data.weights_model_dir() == "/legacy/models"   # what switch.sh uses
        assert any(sev == "warning" for sev, _ in app._startup_notices)

    def test_a_setting_removed_later_is_not_brought_back(self, tmp_path):
        from scripts.lib import club_config   # the loader (conftest puts the root on sys.path)

        repo = _repo(tmp_path)
        M.save_settings({"model_dir": "/old/models"})
        self._launch(repo)
        club_config.unset_values(["MODEL_DIR"])
        app, _ = self._launch(repo)
        assert "MODEL_DIR" not in _read("club3090.env")
        assert app._data.weights_model_dir() == DEFAULT_MODEL_DIR

    def test_a_refused_copy_keeps_the_old_value_and_retries(self, tmp_path):
        repo = _repo(tmp_path)
        M.save_settings({"model_dir": "/odd/$dir"})
        app, _ = self._launch(repo)
        assert _read("club3090.env") == ""
        assert app._data.weights_model_dir() == "/odd/$dir"
        assert any(sev == "warning" for sev, _ in getattr(app, "_startup_notices", []))
        app2, _ = self._launch(repo)
        assert any(sev == "warning" for sev, _ in getattr(app2, "_startup_notices", []))

    def test_shell_values_still_win_after_the_fold_in(self, tmp_path, monkeypatch):
        repo = _repo(tmp_path)
        M.save_settings({"model_dir": "/old/models", "hf_token": "hf_old_tok_3"})
        monkeypatch.setenv("MODEL_DIR", "/shell/models")
        app, env = self._launch(repo, {"HF_TOKEN": "hf_shell_tok_2"})
        assert "MODEL_DIR=/old/models\n" in _read("club3090.env")     # saved for later launches
        assert app._data.weights_model_dir() == "/shell/models"
        assert env["HF_TOKEN"] == "hf_shell_tok_2"


# ── docker compose gets the resolved settings ───────────────────────────────


class TestMigrateNotice:
    """The launchers' one-time "your settings still live in this checkout" notice
    (club_config.migrate_notice, #1466) reaches c3 as a startup toast, shares the
    launchers' stamp (said once by whichever runs first), and carries no value."""

    def _launch(self, repo):
        app, _, _ = make_app()
        app._data.repo_root = repo
        M.apply_persisted_settings(app, {})
        return app

    @staticmethod
    def _said(app) -> list[str]:
        return [msg for _, msg in app._startup_notices if "still live in this checkout" in msg]

    def test_a_toast_once_shared_with_the_launchers(self, tmp_path):
        from club3090_cockpit import settings_store as S

        repo = _repo(tmp_path)
        (repo / ".env").write_text("THREADS=4\nHF_TOKEN=hf_notice_tok_1\n", encoding="utf-8")
        said = self._said(self._launch(repo))
        assert len(said) == 1
        assert "2 setting(s)" in said[0] and "settings.sh migrate" in said[0]
        assert "hf_notice_tok_1" not in said[0]
        assert self._said(self._launch(repo)) == []           # once
        assert S.loader().migrate_notice(repo) == []          # the launchers share the stamp

    def test_quiet_when_nothing_is_pending_or_after_migrate(self, tmp_path):
        from club3090_cockpit import settings_store as S

        repo = _repo(tmp_path)
        assert self._said(self._launch(repo)) == []           # no repo .env at all
        (repo / ".env").write_text("THREADS=4\n", encoding="utf-8")
        S.loader().migrate(repo)
        assert self._said(self._launch(repo)) == []

    def test_never_stops_c3(self, tmp_path, monkeypatch):
        from club3090_cockpit import settings_store as S

        def boom(*_a, **_kw):
            raise RuntimeError("no notice today")

        monkeypatch.setattr(S.loader(), "migrate_notice", boom)
        repo = _repo(tmp_path)
        (repo / ".env").write_text("THREADS=4\n", encoding="utf-8")
        assert self._said(self._launch(repo)) == []


class _EnvFileRunner:
    """A write runner that records what `docker compose --env-file` would read."""

    def __init__(self):
        self.cmd: list[str] = []
        self.env_file: Path | None = None
        self.text = ""
        self.mode = 0
        self.state: CoreRunState | None = None

    def set_callbacks(self, **_kw):
        pass

    async def start_raw(self, cmd, env, run_type, parser):
        self.cmd = list(cmd)
        if "--env-file" in cmd:
            self.env_file = Path(cmd[cmd.index("--env-file") + 1])
            self.text = self.env_file.read_text(encoding="utf-8")
            self.mode = stat.S_IMODE(self.env_file.stat().st_mode)
        self.keys_file = None
        for a in cmd:
            if a.startswith("CLUB3090_LITELLM_ROUTE_KEYS="):
                self.keys_file = Path(a.split("=", 1)[1])
                self.keys_text = self.keys_file.read_text(encoding="utf-8")
                self.keys_mode = stat.S_IMODE(self.keys_file.stat().st_mode)
        self.state = CoreRunState(run_type=run_type, started=time.time())
        return self.state


class TestComposeEnvFile:
    @pytest.mark.asyncio
    async def test_service_start_passes_the_settings_and_removes_the_file(self, tmp_path):
        repo = _repo(tmp_path)
        _seed_service_dirs(repo, ["litellm"])
        _write("club3090.env", "MODEL_DIR=/store/models\n")
        _write("secrets.env", "HF_TOKEN=hf_compose_tok_1\n")
        runner = _EnvFileRunner()
        cd = CockpitData(repo, runner=full_runner(), write_runner=runner)
        executed, _rec, state = await cd.execute_action(cd.service_start("litellm"))
        assert executed
        assert runner.env_file is not None, runner.cmd
        assert "MODEL_DIR='/store/models'" in runner.text
        assert "HF_TOKEN='hf_compose_tok_1'" in runner.text
        assert runner.mode == 0o600
        assert runner.cmd[-6:] == ["-f", "services/litellm/docker-compose.yml", "-p", "litellm", "up", "-d"]
        state.finished = time.time()
        state.done.set()
        for _ in range(20):
            if not runner.env_file.exists():
                break
            await asyncio.sleep(0.01)
        assert not runner.env_file.exists()

    @pytest.mark.asyncio
    async def test_nothing_configured_passes_no_env_file(self, tmp_path):
        repo = _repo(tmp_path)
        _seed_service_dirs(repo, ["litellm"])
        runner = _EnvFileRunner()
        cd = CockpitData(repo, runner=full_runner(), write_runner=runner)
        await cd.execute_action(cd.service_start("litellm"))
        assert runner.cmd[:3] == ["docker", "compose", "-f"], runner.cmd
        assert "--env-file" not in runner.cmd

    @pytest.mark.asyncio
    async def test_gateway_start_gets_only_its_route_keys_and_removes_the_file(self, tmp_path):
        """#1466 4c: the keys this rig's own gateway routes use are saved in
        secrets.env. The gateway gets only those (never the HF token), from a 0600
        file whose path rides `env CLUB3090_LITELLM_ROUTE_KEYS=…`, removed after."""
        repo = _repo(tmp_path)
        _seed_service_dirs(repo, ["litellm"])
        _write("litellm/config.local.yaml",
               "model_list:\n  - model_name: c\n    litellm_params:\n"
               "      model: openai/c\n      api_key: os.environ/RIG_KEY\n")
        _write("secrets.env", "HF_TOKEN=hf_route_tok_1\nRIG_KEY=sk-route-c3\n")
        runner = _EnvFileRunner()
        cd = CockpitData(repo, runner=full_runner(), write_runner=runner)
        executed, _rec, state = await cd.execute_action(cd.service_start("litellm"))
        assert executed
        assert runner.cmd[0] == "env" and runner.keys_file is not None, runner.cmd
        assert runner.keys_text == "RIG_KEY='sk-route-c3'\n"
        assert runner.keys_mode == 0o600
        assert "HF_TOKEN='hf_route_tok_1'" in runner.text          # compose still gets the settings
        assert not any("sk-route-c3" in a for a in runner.cmd)     # never on a command line
        state.finished = time.time()
        state.done.set()
        for _ in range(20):
            if not runner.keys_file.exists() and not runner.env_file.exists():
                break
            await asyncio.sleep(0.01)
        assert not runner.keys_file.exists()
        assert not runner.env_file.exists()

    @pytest.mark.asyncio
    async def test_other_services_get_no_route_keys(self, tmp_path):
        repo = _repo(tmp_path)
        _seed_service_dirs(repo, ["qdrant"])
        _write("litellm/config.local.yaml",
               "model_list:\n  - model_name: c\n    litellm_params:\n"
               "      model: openai/c\n      api_key: os.environ/RIG_KEY\n")
        _write("secrets.env", "RIG_KEY=sk-route-c3\n")
        runner = _EnvFileRunner()
        cd = CockpitData(repo, runner=full_runner(), write_runner=runner)
        await cd.execute_action(cd.service_start("qdrant"))
        assert runner.cmd[:2] == ["docker", "compose"], runner.cmd
        assert runner.keys_file is None

    @pytest.mark.asyncio
    async def test_legacy_repo_env_reaches_compose_through_the_loader(self, tmp_path):
        repo = _repo(tmp_path)
        _seed_service_dirs(repo, ["litellm"])
        (repo / ".env").write_text("MODEL_DIR=/legacy/models\n", encoding="utf-8")
        runner = _EnvFileRunner()
        cd = CockpitData(repo, runner=full_runner(), write_runner=runner)
        await cd.execute_action(cd.service_start("litellm"))
        assert runner.env_file is not None and runner.env_file.name != ".env"
        assert "MODEL_DIR='/legacy/models'" in runner.text


# ── secrets never reach the logs ────────────────────────────────────────────


class TestNoTokenInLogs:
    @pytest.mark.asyncio
    async def test_token_never_reaches_c3_logs(self, tmp_path):
        repo = _repo(tmp_path)
        _seed_service_dirs(repo, ["litellm"])
        M.save_settings({"hf_token": "hf_log_sentinel_1", "model_dir": "/m"})
        app, _, _ = make_app()
        app._data.repo_root = repo
        seen = _record_notifications(app)
        M.apply_persisted_settings(app, {"C3_LOG": "1"})           # session log ON + fold-in
        assert app._session_log is not None
        app.apply_settings(model_dir="", hf_token="hf_log_sentinel_2")
        app.apply_settings(model_dir="", hf_token="hf_log'sentinel_3")   # refused
        runner = _EnvFileRunner()
        app._data._write_runner = runner
        app._data.set_logging_enabled(True)
        await app._data.execute_action(app._data.service_start("litellm"))
        assert "hf_log_sentinel_2" in runner.text                  # compose did get it…
        app.configure_session_logging(False)
        logs = [p for p in (Path(os.environ["C3_CONFIG_DIR"]) / "logs").rglob("*") if p.is_file()]
        assert logs, "expected a session log and a run log"
        for p in logs:
            text = p.read_text(encoding="utf-8", errors="replace")
            assert "sentinel" not in text, p                        # …the logs didn't
        for n in seen + [{"message": m} for _, m in app._startup_notices]:
            assert "sentinel" not in n["message"]


def test_tests_never_see_the_real_config_dir():
    """conftest points CLUB3090_CONFIG_DIR and C3_CONFIG_DIR at a tmp dir for
    every test, so nothing here reads or writes the real ~/.config/club-3090."""
    real = Path.home() / ".config" / "club-3090"
    for var in ("CLUB3090_CONFIG_DIR", "C3_CONFIG_DIR"):
        assert os.environ.get(var), var
        assert Path(os.environ[var]).resolve() != real.resolve()
