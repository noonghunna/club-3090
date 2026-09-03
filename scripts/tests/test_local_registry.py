#!/usr/bin/env python3
"""C4-rev compose_registry LOCAL-layer loader/merge tests.

Exercises load_local_registry() / get_registry() against a THROWAWAY repo copy
(subprocess imports resolve against the tmp root, never the real checkout).

    pytest scripts/tests/test_local_registry.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_COPY_TREES = ("scripts/lib", "tools/tui-core", "models")

# A complete _entry kwargs set (what promote.py --layer local writes).
GOOD_KWARGS = {
    "model": "my-model",
    "weights_variant": "autoround-int4",
    "workload": "long-ctx-single",
    "engine": "vllm-stable",
    "drafter": None,
    "kv_format": "fp8_e5m2",
    "tp": 2,
    "max_ctx": 131072,
    "max_num_seqs": 2,
    "mem_util": 0.92,
    "compose_path": (
        "scripts/lib/profiles-local/composes/my-model/vllm/compose/dual/autoround-int4/base.yml"
    ),
    "default_port": 20242,
    "kvcalc_key": "my-model:dual",
    "status": "incubating",
}


@pytest.fixture()
def root(tmp_path):
    for rel in _COPY_TREES:
        shutil.copytree(
            REPO / rel,
            tmp_path / rel,
            ignore=shutil.# #1142: skip root-owned container torch_compile caches under
            # models/ — copytree dies on them (Permission denied) on any
            # rig that has served a model. Pure build artifact.
            ignore_patterns("__pycache__", "cache"),
        )
    return tmp_path


def _write_local(root, payload):
    p = root / "scripts/lib/profiles-local/registry.local.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _py(root, snippet: str) -> subprocess.CompletedProcess:
    """Run a snippet with cwd = the TMP root so its scripts.lib import resolves
    to the COPY (this is the whole point — the real checkout stays pristine)."""
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(root),
        capture_output=True,
        text=True,
    )


class TestLoaderMerge:
    def test_no_local_layer_returns_core_identity(self, root):
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY, get_registry\n"
            "assert get_registry() is COMPOSE_REGISTRY\n"
            "print(len(COMPOSE_REGISTRY))\n",
        )
        assert r.returncode == 0, r.stderr
        assert int(r.stdout.strip()) > 0

    def test_local_entry_visible_via_get_registry(self, root):
        _write_local(root, {"local/my-model-dual-x": dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY, get_registry\n"
            "reg = get_registry()\n"
            "e = reg['local/my-model-dual-x']\n"
            "# wrapped through _entry → the FULL core row shape + defaults:\n"
            "assert e['pp'] == 1 and e['gpu_assignment_mode'] == 'contiguous'\n"
            "assert e['status'] == 'incubating'\n"
            "# and the CORE dict itself is untouched:\n"
            "assert 'local/my-model-dual-x' not in COMPOSE_REGISTRY\n"
            "assert len(reg) == len(COMPOSE_REGISTRY) + 1\n",
        )
        assert r.returncode == 0, r.stderr

    def test_local_entry_invisible_to_defaults_resolution(self, root):
        _write_local(root, {"local/my-model-dual-x": dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import (\n"
            "    DEFAULTS, ENGINE_PREFERENCE, RECOMMENDED_DEFAULT_MODELS,\n"
            "    curated_default_target, get_registry, model_set,\n"
            ")\n"
            "# C4-rev invariant: a local entry can NEVER become a default.\n"
            "assert 'my-model' not in model_set()\n"
            "assert not any(k[0] == 'my-model' for k in DEFAULTS)\n"
            "assert all('local/' not in s for s in DEFAULTS.values())\n"
            "assert 'my-model' not in RECOMMENDED_DEFAULT_MODELS\n"
            "assert curated_default_target('my-model', 'dual') is None\n",
        )
        assert r.returncode == 0, r.stderr

    def test_slug_collision_with_core_refused(self, root):
        # A local file claiming a CORE slug (even ignoring the namespace rule)
        # must fail LOUDLY.
        _write_local(root, {"vllm/minimal": dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        # The namespace rule OR the collision check refuses — both are loud.
        assert ("collides" in r.stderr) or ("namespace" in r.stderr)

    def test_model_id_collision_with_core_refused(self, root):
        kwargs = dict(GOOD_KWARGS, model="qwen3.6-27b")
        _write_local(root, {"local/qwen36-clone-dual-x": kwargs})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "core model" in r.stderr

    def test_non_local_namespace_refused(self, root):
        _write_local(root, {"curated/my-model-dual-x": dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "namespace" in r.stderr

    def test_bad_json_refused_loudly(self, root):
        _write_local(root, "{not json")
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "unreadable" in r.stderr

    def test_bad_kwargs_refused(self, root):
        # Unknown kwarg → _entry TypeError → LocalRegistryError.
        _write_local(root, {"local/my-model-dual-x": dict(GOOD_KWARGS, bogus_kwd=1)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "_entry kwargs" in r.stderr

    def test_bad_status_refused(self, root):
        _write_local(root, {"local/my-model-dual-x": dict(GOOD_KWARGS, status="nope")})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0

    def test_duplicate_local_slugs_refused(self, root):
        # JSON objects can't carry duplicate keys, but two slugs sharing one
        # MODEL id collide.
        kwargs2 = dict(GOOD_KWARGS, compose_path="scripts/lib/profiles-local/composes/other/x.yml")
        _write_local(
            root,
            {
                "local/my-model-dual-x": dict(GOOD_KWARGS),
                "local/my-model-single-x": kwargs2,
            },
        )
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "duplicate local model id" in r.stderr

    def test_runtime_lookup_helpers_see_local_entries(self, root):
        _write_local(root, {"local/my-model-dual-x": dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import model_of_slug, slug_topology\n"
            "assert model_of_slug('local/my-model-dual-x') == 'my-model'\n"
            "assert slug_topology('local/my-model-dual-x') == 'dual'\n",
        )
        assert r.returncode == 0, r.stderr
