#!/usr/bin/env python3
"""C4-rev promote.py executor tests — run against a THROWAWAY repo copy.

Every test builds a minimal repo root in tmp (scripts/lib + tools/tui-core +
models — everything registry-emit.sh --json and the compose_registry import
need) and drives promote.py's main() against it with --root.  NOTHING here ever
writes to the real checkout.

    pytest scripts/tests/test_promote.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.lib.profiles import promote  # noqa: E402

# Everything the post-write checks (compose_registry import + registry-emit.sh
REPO = Path(__file__).resolve().parents[2]
_COPY_TREES = ("scripts/lib", "tools/tui-core", "models")


@pytest.fixture()
def root(tmp_path):
    for rel in _COPY_TREES:
        shutil.copytree(
            REPO / rel,
            tmp_path / rel,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    return tmp_path


def _spec(*, mid="my-model", slug=None, layer_local=True):
    """A complete, valid LOCAL-layer spec (concrete values — the JSON local
    registry cannot carry <...> placeholders)."""
    quant = "autoround-int4"
    if layer_local:
        slug = slug or f"local/{mid}-dual-{quant}"
        compose_path = (
            f"scripts/lib/profiles-local/composes/{mid}/vllm/compose/dual/{quant}/base.yml"
        )
    else:
        slug = slug or f"vllm/{mid}-dual-{quant}"
        compose_path = f"models/{mid}/vllm/compose/dual/{quant}/base.yml"
    return {
        "model_id": mid,
        "display_name": "My Model",
        "family": "generic-dense",
        "arch": {
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attn_heads": 24,
            "num_kv_heads": 4,
            "head_dim_attn": 256,
            "max_ctx_supported": 131072,
            # compat.load_profiles requires this for a CORE write.
            "attention_k_eq_v": False,
        },
        "weights": {
            quant: {
                "path": f"{mid}-{quant}",
                "local_subdir": f"{mid}-{quant}",
                "size_gb": 12.5,
                "format": "autoround",
                "status": "incubating",
                "hf_repo": "org/model",
                "engine": "vllm",
                "kind": "main",
                "verify_glob": "*.safetensors",
            }
        },
        "default_weight_variant": quant,
        "compatible_drafters": [],
        "vision_capable": False,
        "compose": {
            "path": compose_path,
            "content": (
                "# Profile (at-a-glance):\n"
                "#   Status: 🐣 Incubating\n"
                "# ---\n"
                "services:\n"
                "  vllm:\n"
                "    image: vllm/vllm-openai:v0.27.1\n"
                "    container_name: vllm-my-model-dual\n"
                "    ports:\n"
                '      - "${PORT:-20242}:8000"\n'
            ),
        },
        "registry_entry": {
            "slug": slug,
            "kwargs": {
                "model": mid,
                "weights_variant": quant,
                "workload": "long-ctx-single",
                "engine": "vllm-stable",
                "drafter": None,
                "kv_format": "fp8_e5m2",
                "tp": 2,
                "max_ctx": 131072,
                "max_num_seqs": 2,
                "mem_util": 0.92,
                "compose_path": compose_path,
                "default_port": 20242,
                "kvcalc_key": f"{mid}:dual",
                "status": "incubating",
                "status_note": "test spec",
            },
        },
    }


def _run_cli(root, spec, *extra, env_extra=None):
    """Run promote.py as a SUBPROCESS against the tmp root (exercises the CLI
    shim + exit codes; the tmp root's own compose_registry is imported)."""
    f = root / "spec.json"
    f.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(os.environ)
    env.pop("C3_ALLOW_CORE_PROMOTE", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/lib/profiles/promote.py"),
            "--spec-file",
            str(f),
            "--root",
            str(root),
            *extra,
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
    )


LOCAL_ARTIFACTS = (
    "scripts/lib/profiles-local/models.d/my-model.yml",
    "scripts/lib/profiles-local/composes/my-model/vllm/compose/dual/autoround-int4/base.yml",
    "scripts/lib/profiles-local/registry.local.json",
)


class TestLocalLayer:
    def test_happy_path_writes_three_artifacts_and_ok(self, root):
        res = _run_cli(root, _spec(), "--layer", "local")
        assert res.returncode == 0, res.stderr + res.stdout
        assert "PROMOTE_OK local/my-model-dual-autoround-int4" in res.stdout
        for rel in LOCAL_ARTIFACTS:
            assert (root / rel).exists(), rel
        # NO core file was touched.
        assert not (root / "scripts/lib/profiles/models/my-model.yml").exists()
        reg_text = (root / "scripts/lib/profiles/compose_registry.py").read_text()
        assert "my-model" not in reg_text
        # The merged accessor sees the entry IN THE TMP ROOT.
        chk = subprocess.run(
            [
                sys.executable,
                "-c",
                "from scripts.lib.profiles.compose_registry import get_registry;"
                "e = get_registry()['local/my-model-dual-autoround-int4'];"
                "assert e['model'] == 'my-model' and e['status'] == 'incubating';"
                "assert e['pp'] == 1",  # wrapped through _entry → full shape
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        assert chk.returncode == 0, chk.stderr
        # The profile YAML parses with the right id.
        import yaml

        data = yaml.safe_load(
            (root / "scripts/lib/profiles-local/models.d/my-model.yml").read_text()
        )
        assert data["id"] == "my-model"
        assert data["num_hidden_layers"] == 40

    def test_collision_exit_3(self, root):
        first = _run_cli(root, _spec(), "--layer", "local")
        assert first.returncode == 0
        # Same slug again → refusal, nothing overwritten.
        second = _run_cli(root, _spec(), "--layer", "local")
        assert second.returncode == promote.EXIT_COLLISION
        assert "REFUSED" in second.stderr
        # A colliding MODEL ID (core model) refuses too.
        core_mid = next(
            Path(p).stem
            for p in (root / "scripts/lib/profiles/models").glob("*.yml")
        )
        coll = _run_cli(
            root, _spec(mid=core_mid, slug="local/other-dual-x"), "--layer", "local"
        )
        assert coll.returncode == promote.EXIT_COLLISION
        assert "CORE" in coll.stderr

    def test_refuses_slug_outside_local_namespace(self, root):
        res = _run_cli(root, _spec(slug="vllm/my-model-dual-x"), "--layer", "local")
        assert res.returncode == promote.EXIT_COLLISION
        assert "namespace" in res.stderr

    def test_refuses_compose_outside_layer(self, root):
        spec = _spec()
        spec["compose"]["path"] = "models/my-model/vllm/compose/dual/x/base.yml"
        res = _run_cli(root, spec, "--layer", "local")
        assert res.returncode == promote.EXIT_COLLISION
        assert "profiles-local" in res.stderr

    def test_dry_run_writes_nothing(self, root):
        res = _run_cli(root, _spec(), "--layer", "local", "--dry-run")
        assert res.returncode == 0
        assert "dry-run" in res.stdout
        for rel in LOCAL_ARTIFACTS:
            assert not (root / rel).exists(), rel
        assert not (root / "scripts/lib/profiles-local/models.d").exists()

    def test_status_forced_incubating(self, root):
        spec = _spec()
        spec["registry_entry"]["kwargs"]["status"] = "production"  # attempted bump
        res = _run_cli(root, spec, "--layer", "local")
        assert res.returncode == 0
        raw = json.loads(
            (root / "scripts/lib/profiles-local/registry.local.json").read_text()
        )
        assert raw["local/my-model-dual-autoround-int4"]["status"] == "incubating"


class TestCoreGate:
    def test_core_without_flag_exits_nonzero_and_writes_nothing(self, root):
        res = _run_cli(root, _spec(layer_local=False), "--layer", "core")
        assert res.returncode == promote.EXIT_COLLISION
        assert "maintainer-gated" in res.stderr
        assert not (root / "scripts/lib/profiles/models/my-model.yml").exists()
        assert "my-model" not in (
            root / "scripts/lib/profiles/compose_registry.py"
        ).read_text()

    def test_core_with_flag_writes_curated_catalog(self, root):
        res = _run_cli(
            root,
            _spec(layer_local=False),
            "--layer",
            "core",
            env_extra={"C3_ALLOW_CORE_PROMOTE": "1"},
        )
        assert res.returncode == 0, res.stderr + res.stdout
        assert "PROMOTE_OK vllm/my-model-dual-autoround-int4" in res.stdout
        assert (root / "scripts/lib/profiles/models/my-model.yml").exists()
        reg = (root / "scripts/lib/profiles/compose_registry.py").read_text()
        assert '"vllm/my-model-dual-autoround-int4": _entry(' in reg
        # And it still imports.
        chk = subprocess.run(
            [sys.executable, "-c", "import scripts.lib.profiles.compose_registry"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        assert chk.returncode == 0, chk.stderr

    def test_core_refuses_local_namespace_slug(self, root):
        res = _run_cli(
            root,
            _spec(),
            "--layer",
            "core",
            env_extra={"C3_ALLOW_CORE_PROMOTE": "1"},
        )
        assert res.returncode == promote.EXIT_COLLISION
        assert "LOCAL layer" in res.stderr


class TestInProcessMain:
    """The library entry point (what the c3 write plan shells out to)."""

    def test_main_returns_zero_and_prints_ok(self, root, capsys):
        rc = promote.main(["--spec-file", str(_write_spec(root)), "--root", str(root)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "PROMOTE_OK local/my-model-dual-autoround-int4" in out

    def test_main_refusal_exit_3(self, root, capsys):
        spec = _spec()
        spec["display_name"] = ""
        f = root / "bad.json"
        f.write_text(json.dumps(spec), encoding="utf-8")
        rc = promote.main(["--spec-file", str(f), "--root", str(root)])
        assert rc == promote.EXIT_COLLISION


def _write_spec(root):
    f = root / "spec.json"
    f.write_text(json.dumps(_spec()), encoding="utf-8")
    return f
