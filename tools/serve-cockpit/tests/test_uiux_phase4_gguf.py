"""Phase 4 — GGUF route-G: fit + compose emit (no GPU serve).

Emit tests cover the *real* sibling shape (``command: >-`` folded scalar),
vision ``--mmproj`` rewrite, and sibling drafter-flag stripping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from club3090_cockpit.services import CockpitData


def _stub_registry(tmp_path: Path, slug: str, compose_rel: str, **entry_extra):
    """Install a minimal COMPOSE_REGISTRY stub for emit_gguf_compose imports."""
    fake_reg = {
        slug: {"compose_path": compose_rel, "engine": "llama-cpp-local", **entry_extra},
    }
    scripts = ModuleType("scripts")
    lib = ModuleType("scripts.lib")
    profiles = ModuleType("scripts.lib.profiles")
    cr = ModuleType("scripts.lib.profiles.compose_registry")
    cr.COMPOSE_REGISTRY = fake_reg
    sys.modules["scripts"] = scripts
    sys.modules["scripts.lib"] = lib
    sys.modules["scripts.lib.profiles"] = profiles
    sys.modules["scripts.lib.profiles.compose_registry"] = cr
    return fake_reg


def _write_sibling(tmp_path: Path, rel: str, command) -> Path:
    """Write a sibling compose. ``command`` may be a folded-style str or a list."""
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    # When command is a str, emit YAML folded scalar (the real ship shape).
    if isinstance(command, str):
        body = {
            "services": {
                "llm": {
                    "image": "ghcr.io/ggerganov/llama.cpp:server",
                    "command": command,
                    "ports": ["8080:8080"],
                    "volumes": [],
                }
            }
        }
        # force folded-scalar round-trip: dump then re-load to confirm str form
        text = yaml.safe_dump(body, default_flow_style=False, sort_keys=False)
        # yaml may dump str as plain; write explicit >- form for fidelity
        text = (
            "services:\n"
            "  llm:\n"
            "    image: ghcr.io/ggerganov/llama.cpp:server\n"
            "    command: >-\n"
            f"      {command}\n"
            "    ports:\n"
            "      - '8080:8080'\n"
            "    volumes: []\n"
        )
        path.write_text(text, encoding="utf-8")
    else:
        path.write_text(
            yaml.safe_dump({
                "services": {
                    "llm": {
                        "image": "ghcr.io/ggerganov/llama.cpp:server",
                        "command": list(command),
                        "ports": ["8080:8080"],
                        "volumes": [],
                    }
                }
            }, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
    return path


def _load_emitted_cmd(compose_path: str) -> list:
    doc = yaml.safe_load(Path(compose_path).read_text(encoding="utf-8"))
    svc = next(iter(doc["services"].values()))
    cmd = svc["command"]
    assert isinstance(cmd, list), f"emitted command must be a list, got {type(cmd)}"
    # Must not be per-character explosion.
    assert all(len(str(t)) > 0 for t in cmd)
    assert not (len(cmd) > 50 and all(len(str(t)) == 1 for t in cmd[:20]))
    return [str(x) for x in cmd]


class TestByoCheckGguf:
    def test_gguf_fit_not_unsupported_format(self, tmp_path: Path):
        data = CockpitData(tmp_path)
        res = data.byo_check_gguf(
            "org/Model-GGUF",
            "llama-cpp/q4km",
            quant="Q4_K_M",
            size_gb=12.0,
            card_vram_gb=24.0,
        )
        assert not res.error
        assert "unsupported" not in (res.error or "").lower()
        assert res.arch == "gguf"
        assert res.route == "G"
        assert res.quant_match == "Q4_K_M"
        assert res.fit_verdict == "fits-clean"
        assert res.eligible is True

    def test_gguf_wont_fit_huge_on_single(self, tmp_path: Path):
        data = CockpitData(tmp_path)
        res = data.byo_check_gguf(
            "org/Huge", "llama-cpp/q4km", quant="Q8_0", size_gb=40.0, card_vram_gb=24.0,
        )
        assert res.fit_verdict == "wont-fit"
        assert res.eligible is False
        assert res.route is None

    def test_gguf_large_fits_dual_budget(self, tmp_path: Path):
        """34 GB Q8 wrongly wont-fit on 24 GB; dual sibling → 48 GB budget."""
        data = CockpitData(tmp_path)
        # Stub registry path with /dual/ so topology_cards_for_profile → 2.
        _stub_registry(
            tmp_path,
            "llama-cpp/q8-dual",
            "models/m/llama-cpp/compose/dual/q8/serve.yml",
        )
        res = data.byo_check_gguf(
            "org/Huge",
            "llama-cpp/q8-dual",
            quant="Q8_0",
            size_gb=34.0,
            # card_vram_gb omitted → topology × 24
        )
        assert data.topology_cards_for_profile("llama-cpp/q8-dual") == 2
        assert res.fit_verdict in ("fits-clean", "fits-constrained")
        assert res.eligible is True
        assert "dual" in (res.note or "") or "48" in (res.note or "")

    def test_gguf_requires_quant(self, tmp_path: Path):
        data = CockpitData(tmp_path)
        res = data.byo_check_gguf("org/X", "llama-cpp/q4km", quant="", size_gb=5.0)
        assert res.error


class TestNormalizeCommand:
    def test_str_folded_scalar_shlex_not_chars(self, tmp_path: Path):
        data = CockpitData(tmp_path)
        raw = (
            "--host 0.0.0.0 --port 8080 --model /models/old.gguf "
            "--spec-type draft-mtp --spec-draft-n-max 2"
        )
        toks = data._normalize_compose_command(raw)
        assert toks[0] == "--host"
        assert "--model" in toks
        assert len(toks) < 30  # not 100+ single-char tokens
        assert not all(len(t) == 1 for t in toks)


def _emitted_volumes(compose_path: str) -> list:
    doc = yaml.safe_load(Path(compose_path).read_text(encoding="utf-8"))
    return next(iter(doc["services"].values())).get("volumes") or []


class TestEmitGgufCompose:
    # Brought files live UNDER MODEL_DIR (the pull dir), the sibling mounts
    # MODEL_DIR → /models, so route-G addresses them at /models/<rel> and mounts
    # MODEL_DIR ONCE — NOT the old fixed /models/brought.* + per-file mounts (which
    # died at boot with "read-only file system"; 2026-07-09 dogfood).
    def test_emit_list_form_still_works(self, tmp_path: Path, monkeypatch):
        rel = "models/m/llama-cpp/compose/single/q4/serve.yml"
        _write_sibling(
            tmp_path, rel,
            ["--host", "0.0.0.0", "--model", "/old/path.gguf"],
        )
        _stub_registry(tmp_path, "llama-cpp/q4km", rel)
        data = CockpitData(tmp_path)
        monkeypatch.setattr(data, "weights_model_dir", lambda: str(tmp_path))
        weights = tmp_path / "weights" / "model-Q4_K_M.gguf"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"gguf")
        res = data.emit_gguf_compose(
            "llama-cpp/q4km", str(weights), served_name="MyGGUF",
        )
        assert res["error"] == "", res
        cmd = _load_emitted_cmd(res["compose_path"])
        mi = cmd.index("--model") if "--model" in cmd else cmd.index("-m")
        assert cmd[mi + 1].startswith("/models/")
        assert cmd[mi + 1].endswith("model-Q4_K_M.gguf")
        assert "brought" not in cmd[mi + 1]

    def test_emit_single_models_mount_no_per_file(self, tmp_path: Path, monkeypatch):
        """The mount BUG: exactly ONE absolute MODEL_DIR → /models mount, and NO
        per-file mount into /models (that races the /models mount → read-only)."""
        rel = "models/m/llama-cpp/compose/single/q4/mtp-vision.yml"
        _write_sibling(
            tmp_path, rel,
            "--host 0.0.0.0 -m /models/${GGUF_FILE:-old.gguf} "
            "--mmproj /models/old-mmproj.gguf",
        )
        _stub_registry(tmp_path, "llama-cpp/q4-vision", rel)
        data = CockpitData(tmp_path)
        monkeypatch.setattr(data, "weights_model_dir", lambda: str(tmp_path))
        wdir = tmp_path / "pulls" / "repo"
        wdir.mkdir(parents=True)
        for n in ("main-Q4_K_M.gguf", "mmproj-F16.gguf", "mtp-Q4_K_M.gguf"):
            (wdir / n).write_bytes(b"x")
        res = data.emit_gguf_compose(
            "llama-cpp/q4-vision", str(wdir / "main-Q4_K_M.gguf"),
            served_name="V", mmproj_host_file=str(wdir / "mmproj-F16.gguf"),
            mtp_draft_host_file=str(wdir / "mtp-Q4_K_M.gguf"),
        )
        assert res["error"] == "", res
        vols = _emitted_volumes(res["compose_path"])
        models_mounts = [v for v in vols if ":/models:" in v or v.endswith(":/models")]
        assert len(models_mounts) == 1, vols
        assert models_mounts[0] == f"{tmp_path}:/models:ro"          # absolute, single
        assert not any("/models/brought" in v for v in vols)         # no per-file mounts
        # all three artifacts addressed under the single /models mount
        cmd = " ".join(_load_emitted_cmd(res["compose_path"]))
        assert "/models/pulls/repo/main-Q4_K_M.gguf" in cmd
        assert "/models/pulls/repo/mmproj-F16.gguf" in cmd
        assert "/models/pulls/repo/mtp-Q4_K_M.gguf" in cmd

    def test_emit_writes_to_runtime_dir_not_project(self, tmp_path: Path, monkeypatch):
        """The compose must NOT land beside the sibling in the project tree."""
        rel = "models/m/llama-cpp/compose/single/q4/serve.yml"
        _write_sibling(tmp_path, rel, ["--host", "0.0.0.0", "--model", "/old.gguf"])
        _stub_registry(tmp_path, "llama-cpp/q4km", rel)
        data = CockpitData(tmp_path)
        monkeypatch.setattr(data, "weights_model_dir", lambda: str(tmp_path))
        w = tmp_path / "w" / "m-Q4_K_M.gguf"
        w.parent.mkdir(parents=True)
        w.write_bytes(b"g")
        res = data.emit_gguf_compose("llama-cpp/q4km", str(w), served_name="X")
        assert res["error"] == "", res
        assert "/compose/single/q4/" not in res["compose_path"]      # not beside sibling
        assert "club3090/composes" in res["compose_path"]            # runtime dir

    def test_emit_folded_scalar_command_rewrites_model(self, tmp_path: Path, monkeypatch):
        """BLOCKER fix: real siblings use ``command: >-`` → str → must shlex.split."""
        rel = "models/m/ik-llama/compose/single/iq4/mtp.yml"
        _write_sibling(
            tmp_path, rel,
            "--host 0.0.0.0 --port 8080 --model /models/${GGUF_FILE:-old.gguf} "
            "-ngl 99 --ctx-size 200000 --spec-type mtp:n_max=2,p_min=0.0",
        )
        # Confirm yaml loads as str (the bug surface).
        loaded = yaml.safe_load((tmp_path / rel).read_text(encoding="utf-8"))
        raw_cmd = next(iter(loaded["services"].values()))["command"]
        assert isinstance(raw_cmd, str), "test setup must produce str command"

        _stub_registry(tmp_path, "ik-llama/iq4ks-mtp", rel)
        data = CockpitData(tmp_path)
        monkeypatch.setattr(data, "weights_model_dir", lambda: str(tmp_path))
        weights = tmp_path / "w" / "Tess-Q4_K_M.gguf"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"gguf")
        res = data.emit_gguf_compose(
            "ik-llama/iq4ks-mtp", str(weights), served_name="Tess",
        )
        assert res["error"] == "", res
        cmd = _load_emitted_cmd(res["compose_path"])
        assert len(cmd) < 40
        mi = cmd.index("--model") if "--model" in cmd else cmd.index("-m")
        assert cmd[mi + 1].startswith("/models/") and cmd[mi + 1].endswith("Tess-Q4_K_M.gguf")
        # Sibling built-in MTP --spec-type must not survive for a foreign model.
        assert not any(t.startswith("--spec-") for t in cmd)
        assert "mtp:n_max" not in " ".join(cmd)

    def test_emit_rewrites_mmproj_not_only_appends(self, tmp_path: Path, monkeypatch):
        rel = "models/m/llama-cpp/compose/single/q4/mtp-vision.yml"
        _write_sibling(
            tmp_path, rel,
            "--host 0.0.0.0 -m /models/old.gguf "
            "--mmproj /models/sibling-mmproj-F16.gguf "
            "--spec-type draft-mtp --spec-draft-n-max 2",
        )
        _stub_registry(tmp_path, "llama-cpp/q4km-vision", rel)
        data = CockpitData(tmp_path)
        monkeypatch.setattr(data, "weights_model_dir", lambda: str(tmp_path))
        weights = tmp_path / "w" / "main.gguf"
        mmproj = tmp_path / "w" / "brought-mmproj.gguf"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"gguf")
        mmproj.write_bytes(b"mm")
        res = data.emit_gguf_compose(
            "llama-cpp/q4km-vision",
            str(weights),
            served_name="Vis",
            mmproj_host_file=str(mmproj),
        )
        assert res["error"] == "", res
        cmd = _load_emitted_cmd(res["compose_path"])
        assert cmd.count("--mmproj") == 1
        mi = cmd.index("--mmproj")
        assert cmd[mi + 1].startswith("/models/") and cmd[mi + 1].endswith("brought-mmproj.gguf")
        assert "sibling-mmproj" not in " ".join(cmd)

    def test_emit_strips_sibling_drafter_flags(self, tmp_path: Path, monkeypatch):
        rel = "models/m/beellama/compose/single/q4/dflash.yml"
        _write_sibling(
            tmp_path, rel,
            "--host 0.0.0.0 --model /models/old.gguf "
            "--spec-draft-model /models/anbeeld-dflash-IQ4_XS.gguf "
            "--spec-type dflash --spec-dflash-cross-ctx 1024 --spec-draft-ngl all",
        )
        _stub_registry(tmp_path, "beellama/q4-dflash", rel)
        data = CockpitData(tmp_path)
        monkeypatch.setattr(data, "weights_model_dir", lambda: str(tmp_path))
        weights = tmp_path / "w" / "foreign.gguf"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"gguf")
        res = data.emit_gguf_compose(
            "beellama/q4-dflash", str(weights), served_name="Foreign",
        )
        assert res["error"] == "", res
        cmd = _load_emitted_cmd(res["compose_path"])
        joined = " ".join(cmd)
        assert "--spec-draft-model" not in cmd
        assert "--spec-type" not in cmd
        assert "anbeeld" not in joined
        assert "dflash" not in joined.lower()
        assert "--spec-dflash-cross-ctx" not in cmd
        assert "--spec-draft-ngl" not in cmd
        mi = cmd.index("--model") if "--model" in cmd else cmd.index("-m")
        assert cmd[mi + 1].startswith("/models/") and cmd[mi + 1].endswith("foreign.gguf")

    def test_emit_wires_brought_mtp_draft(self, tmp_path: Path, monkeypatch):
        rel = "models/m/llama-cpp/compose/single/q4/mtp.yml"
        _write_sibling(
            tmp_path, rel,
            "--host 0.0.0.0 --model /models/old.gguf "
            "--spec-type draft-mtp --spec-draft-n-max 2",
        )
        _stub_registry(tmp_path, "llama-cpp/q4-mtp", rel)
        data = CockpitData(tmp_path)
        monkeypatch.setattr(data, "weights_model_dir", lambda: str(tmp_path))
        weights = tmp_path / "w" / "Tess-Q4_K_M.gguf"
        mtp = tmp_path / "w" / "mtp-Q4_K_M.gguf"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"gguf")
        mtp.write_bytes(b"mtp")
        res = data.emit_gguf_compose(
            "llama-cpp/q4-mtp",
            str(weights),
            served_name="Tess",
            mtp_draft_host_file=str(mtp),
        )
        assert res["error"] == "", res
        cmd = _load_emitted_cmd(res["compose_path"])
        assert "--spec-draft-model" in cmd
        di = cmd.index("--spec-draft-model")
        assert cmd[di + 1].startswith("/models/") and cmd[di + 1].endswith("mtp-Q4_K_M.gguf")
        assert "--spec-type" in cmd
        assert "draft-mtp" in cmd
        # Only the brought draft path — not a sibling leftover.
        assert "old.gguf" not in " ".join(cmd)

    def test_emit_file_outside_model_dir_gets_own_mount(self, tmp_path: Path, monkeypatch):
        """A brought file NOT under MODEL_DIR → its own /brought/ mount, never /models."""
        rel = "models/m/llama-cpp/compose/single/q4/serve.yml"
        _write_sibling(tmp_path, rel, ["--host", "0.0.0.0", "--model", "/old.gguf"])
        _stub_registry(tmp_path, "llama-cpp/q4km", rel)
        data = CockpitData(tmp_path)
        # MODEL_DIR is one subtree; the weights live in a DIFFERENT subtree.
        monkeypatch.setattr(data, "weights_model_dir", lambda: str(tmp_path / "model-root"))
        (tmp_path / "model-root").mkdir()
        w = tmp_path / "elsewhere" / "m-Q4_K_M.gguf"
        w.parent.mkdir(parents=True)
        w.write_bytes(b"g")
        res = data.emit_gguf_compose("llama-cpp/q4km", str(w), served_name="X")
        assert res["error"] == "", res
        cmd = _load_emitted_cmd(res["compose_path"])
        mi = cmd.index("--model") if "--model" in cmd else cmd.index("-m")
        assert cmd[mi + 1].startswith("/brought/")                   # not /models/../
        vols = _emitted_volumes(res["compose_path"])
        assert any(str(w) in v and cmd[mi + 1] in v for v in vols)   # explicit mount
