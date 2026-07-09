"""Phase 4 — GGUF route-G: fit + compose emit (no GPU serve)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from club3090_cockpit.services import CockpitData
from club3090_cockpit.data import ByoResult


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

    def test_gguf_wont_fit_huge(self, tmp_path: Path):
        data = CockpitData(tmp_path)
        res = data.byo_check_gguf(
            "org/Huge", "llama-cpp/q4km", quant="Q8_0", size_gb=40.0, card_vram_gb=24.0,
        )
        assert res.fit_verdict == "wont-fit"
        assert res.eligible is False
        assert res.route is None

    def test_gguf_requires_quant(self, tmp_path: Path):
        data = CockpitData(tmp_path)
        res = data.byo_check_gguf("org/X", "llama-cpp/q4km", quant="", size_gb=5.0)
        assert res.error


class TestEmitGgufCompose:
    def test_emit_rewrites_model_and_engine_family(self, tmp_path: Path):
        # Minimal fake sibling compose + registry-like path layout.
        compose_dir = tmp_path / "models" / "m" / "llama-cpp" / "compose" / "single" / "q4"
        compose_dir.mkdir(parents=True)
        sibling = compose_dir / "serve.yml"
        sibling.write_text(
            yaml.safe_dump({
                "services": {
                    "llm": {
                        "image": "ghcr.io/ggerganov/llama.cpp:server",
                        "command": ["--host", "0.0.0.0", "--model", "/old/path.gguf"],
                        "ports": ["8080:8080"],
                        "volumes": [],
                    }
                }
            }),
            encoding="utf-8",
        )
        # Monkeypatch COMPOSE_REGISTRY via emit path: we pass profile that we inject.
        data = CockpitData(tmp_path)
        weights = tmp_path / "weights" / "model-Q4_K_M.gguf"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"gguf")

        # Inject a fake registry entry by patching the import inside the method.
        import club3090_cockpit.services as svc_mod

        fake_reg = {
            "llama-cpp/q4km": {
                "compose_path": str(sibling.relative_to(tmp_path)),
                "engine": "llama-cpp-local",
            }
        }

        class _FakeReg:
            @staticmethod
            def get(k):
                return fake_reg.get(k)

        # emit imports COMPOSE_REGISTRY from scripts — patch after ensuring path.
        import sys
        from types import ModuleType, SimpleNamespace

        # Provide a stub module tree for the import inside emit_gguf_compose.
        scripts = ModuleType("scripts")
        lib = ModuleType("scripts.lib")
        profiles = ModuleType("scripts.lib.profiles")
        cr = ModuleType("scripts.lib.profiles.compose_registry")
        cr.COMPOSE_REGISTRY = fake_reg
        sys.modules["scripts"] = scripts
        sys.modules["scripts.lib"] = lib
        sys.modules["scripts.lib.profiles"] = profiles
        sys.modules["scripts.lib.profiles.compose_registry"] = cr

        res = data.emit_gguf_compose(
            "llama-cpp/q4km", str(weights), served_name="MyGGUF",
        )
        assert res["error"] == "", res
        assert res["compose_path"]
        assert Path(res["compose_path"]).is_file()
        doc = yaml.safe_load(Path(res["compose_path"]).read_text(encoding="utf-8"))
        svc = next(iter(doc["services"].values()))
        cmd = [str(x) for x in svc["command"]]
        assert "--model" in cmd
        assert "/models/brought.gguf" in cmd
        assert any("brought.gguf" in str(v) for v in svc["volumes"])
        assert str(svc.get("container_name", "")).startswith("llama-brought-")
        # GGUF engine family (llama.cpp image), not vLLM.
        assert "vllm" not in str(svc.get("image", "")).lower()
