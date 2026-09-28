#!/usr/bin/env bash
# Offline contract tests for the prebuilt FlashAttention artifact consumer.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 - "$ROOT_DIR" <<'PY'
from dataclasses import make_dataclass
import hashlib
# Imported eagerly and deliberately: the patch block below replaces pathlib.Path
# with a function, and resolving the dotted target "importlib.metadata.version"
# would otherwise trigger this module's FIRST import at that moment. Its chain
# reaches importlib.resources._common, whose body runs
# `@as_file.register(pathlib.Path)` and raises TypeError on a non-class. Loading
# it here means the test does not depend on whether something else imported it.
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
import sysconfig
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

script = Path(sys.argv[1]) / "models/qwen3.8-27b/vllm/patches/fa2-fp8kv-sm86/install_artifact.py"
torch = ModuleType("torch")
torch.__version__ = "test-torch"
torch.version = SimpleNamespace(cuda="test-cuda")
torch._C = SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)
torch.cuda = SimpleNamespace(device_count=lambda: 2, get_device_capability=lambda i: (8, 6))
metadata = make_dataclass("FlashAttentionMetadata", [(name, object) for name in (
    "num_actual_tokens", "max_query_len", "query_start_loc", "seq_lens", "block_table", "slot_mapping", "causal")])
modules = {"torch": torch}
for name in ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends", "vllm.v1.kv_cache_layout"):
    modules[name] = ModuleType(name)
modules["vllm.v1.attention.backends"].flash_attn = SimpleNamespace(FlashAttentionMetadata=metadata)
modules["vllm.v1.kv_cache_layout"].KVCacheLayout = SimpleNamespace(LBNHC=SimpleNamespace(layer_view_order=(0, 2, 1, 3)))
abi = {"torch": torch.__version__, "cuda": torch.version.cuda,
       "python_soabi": sysconfig.get_config_var("SOABI"), "machine": platform.machine(),
       "system": platform.system(), "cxx11_abi": True}

for scenario in ("valid", "identity", "checksum", "abi", "compiled-sm", "unsupported-sm", "mixed-sm", "too-few-gpus", "path-escape", "missing-wheel", "native-sm90", "native-sm100"):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = root / "runtime.env"
        payload = {"fa2_fp8kv.so": b"kernel", "fa2_fp8kv_prefill.so": b"prefill", "plugin.whl": b"wheel"}
        if scenario == "missing-wheel":
            del payload["plugin.whl"]
        if scenario == "path-escape":
            payload["../escape"] = b"escape"
        manifest = {"schema": 1, "abi": dict(abi), "compiled_sm": ["8.6"],
                    "files": {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}}
        if scenario == "abi":
            manifest["abi"]["torch"] = "different-torch"
        if scenario == "compiled-sm":
            manifest["compiled_sm"] = ["8.9"]
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        artifact = root / identity
        artifact.mkdir()
        for name, value in payload.items():
            (artifact / name).write_bytes(value)
        manifest["artifact_id"] = "0" * 64 if scenario == "identity" else identity
        (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if scenario == "checksum":
            (artifact / "fa2_fp8kv.so").write_bytes(b"corrupt")
        capabilities = [(8, 6), (8, 6)]
        if scenario == "unsupported-sm":
            capabilities = [(7, 5), (7, 5)]
        elif scenario == "mixed-sm":
            capabilities[1] = (8, 9)
        elif scenario.startswith("native-sm"):
            capabilities = [(9, 0) if scenario.endswith("90") else (10, 0)] * 2
        torch.cuda.get_device_capability = lambda i: capabilities[i]
        torch.cuda.device_count = lambda: 1 if scenario == "too-few-gpus" else 2

        def local_path(value):
            return {"/opt/club3090/fa2-artifacts": root, "/etc/club3090/fa2-runtime.env": runtime}.get(value, Path(value))

        expected = {"identity": "identity mismatch", "checksum": "checksum mismatch", "abi": "ABI mismatch",
                    "compiled-sm": "no compiled kernel", "unsupported-sm": "No FP8 FlashAttention",
                    "mixed-sm": "homogeneous", "too-few-gpus": "selected CUDA devices",
                    "path-escape": "escapes", "missing-wheel": "exactly one plugin wheel"}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"FA2_ARTIFACT_ID": identity}), \
             patch.object(sys, "argv", [str(script)]), patch("pathlib.Path", local_path), \
             patch("subprocess.run") as install, patch("importlib.metadata.version", return_value="test-vllm"):
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as error:
                assert scenario in expected and expected[scenario] in str(error), (scenario, error)
                assert not install.called and not runtime.exists(), scenario
            else:
                assert scenario in ("valid", "native-sm90", "native-sm100"), scenario
                assert runtime.exists(), scenario
                if scenario == "valid":
                    command = install.call_args.args[0]
                    assert "--no-index" in command and "--no-deps" in command
                    assert str(artifact / "fa2_fp8kv.so") in runtime.read_text(encoding="utf-8")
                else:
                    assert not install.called, scenario
        print("PASS", scenario)

# Exercise the actual compose entrypoint: the shared detector's enabled flag
# covers PCIe P2P as well as NVLink, so it cannot alone select custom all-reduce.
import yaml
compose = script.parents[2] / "compose/dual/fp8/dflash2.yml"
body = yaml.safe_load(compose.read_text(encoding="utf-8"))["services"]["vllm-qwen38-27b-dual-ultramax"]["entrypoint"][2]
for name, enabled, level, disabled in (("pcie", 0, "", True), ("pcie-p2p", 1, "PHB", True), ("nvlink", 1, "NVL", False)):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        etc = root / "etc"
        etc.mkdir()
        (etc / "detect_nvlink.sh").write_text(
            f"_NVLINK_ENABLED={enabled}\nexport NCCL_P2P_LEVEL='{level}'\nexport PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512\n", encoding="utf-8")
        (etc / "fa2-runtime.env").write_text("", encoding="utf-8")
        for sub in set(re.findall(r"/etc/club3090/([\w.-]+)/install\.sh", body)):
            (etc / sub).mkdir()
            (etc / sub / "install.sh").write_text("exit 0\n", encoding="utf-8")
        # #1358: the envelope helper is sourced, not stubbed — run the real one.
        if "/etc/club3090/fa2/envelope.sh" in body:
            (etc / "fa2").mkdir(exist_ok=True)
            (etc / "fa2" / "envelope.sh").write_text(
                (script.parent / "envelope.sh").read_text(encoding="utf-8"), encoding="utf-8")
        engine = root / "vllm"
        engine.write_text('#!/usr/bin/env bash\nprintf "ALLOC=%s\\n" "$PYTORCH_CUDA_ALLOC_CONF"\nprintf "%s\\n" "$@"\n', encoding="utf-8")
        engine.chmod(0o755)
        environment = {"PATH": str(root) + ":" + os.environ["PATH"],
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        result = subprocess.run(["bash", "-c", body.replace("$$", "$").replace("/etc/club3090", str(etc)), "--"],
                                env=environment, capture_output=True, encoding="utf-8", check=True)
        lines = result.stdout.splitlines()
        assert ("--disable-custom-all-reduce" in lines) == disabled, (name, lines)
        allocator = "expandable_segments:True" if disabled else "max_split_size_mb:512"
        assert "ALLOC=" + allocator in lines, (name, lines)
        print("PASS interconnect", name)
PY
