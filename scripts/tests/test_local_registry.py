#!/usr/bin/env python3
"""C4-rev compose_registry LOCAL-layer loader/merge tests.

Exercises load_local_registry() / get_registry() against a THROWAWAY repo copy
(subprocess imports resolve against the tmp root, never the real checkout).

    pytest scripts/tests/test_local_registry.py

⚠️ Slug shape here is `<engine>/<name>` — the SAME shape curated rows use. The
old `local/` namespace was hard-cut in #1205; `local/` now refuses at load and
provenance lives in the `origin` field instead. See TestNamespaceHardCut.
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

# The canonical local slug under the post-#1205 shape.
LOCAL_SLUG = "vllm/my-model-dual-x"


@pytest.fixture()
def root(tmp_path):
    for rel in _COPY_TREES:
        shutil.copytree(
            REPO / rel,
            tmp_path / rel,
            # #1142: skip root-owned container torch_compile caches under
            # models/ — copytree dies on them (Permission denied) on any
            # rig that has served a model. Pure build artifact.
            #
            # ⚠️⚠️ `profiles-local` is skipped for a DIFFERENT and sharper reason
            # (#1316 follow-up): it is the REAL local layer of whoever is running
            # the suite. Copying it in makes these tests inherit that machine's
            # promoted models — so they pass on a clean checkout and fail on any
            # rig that has ever run `promote.py --layer local`, which is the very
            # feature under test. `_write_local` seeds what each test needs; the
            # real layer must never be visible. Note the FIRST test here asserts
            # "no local layer → core identity", which is precisely the assertion
            # a stray real layer silently destroys.
            ignore=shutil.ignore_patterns("__pycache__", "cache", "profiles-local"),
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
        _write_local(root, {LOCAL_SLUG: dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY, get_registry\n"
            "reg = get_registry()\n"
            f"e = reg['{LOCAL_SLUG}']\n"
            "# wrapped through _entry → the FULL core row shape + defaults:\n"
            "assert e['pp'] == 1 and e['gpu_assignment_mode'] == 'contiguous'\n"
            "assert e['status'] == 'incubating'\n"
            "# provenance is the origin field now, not the slug namespace (#1205):\n"
            "assert e['origin'] == 'local'\n"
            "# and the CORE dict itself is untouched:\n"
            f"assert '{LOCAL_SLUG}' not in COMPOSE_REGISTRY\n"
            "assert len(reg) == len(COMPOSE_REGISTRY) + 1\n",
        )
        assert r.returncode == 0, r.stderr

    def test_local_entry_invisible_to_defaults_resolution(self, root):
        _write_local(root, {LOCAL_SLUG: dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import (\n"
            "    DEFAULTS, ENGINE_PREFERENCE, RECOMMENDED_DEFAULT_MODELS,\n"
            "    curated_default_target, get_registry, model_set,\n"
            ")\n"
            "# C4-rev invariant: a local entry can NEVER become a default.\n"
            "assert 'my-model' not in model_set()\n"
            "assert not any(k[0] == 'my-model' for k in DEFAULTS)\n"
            "# ⚠️ post-#1205 a local slug is shaped like any other, so this can no\n"
            "# longer be checked by namespace prefix — name the slug explicitly.\n"
            f"assert '{LOCAL_SLUG}' not in set(DEFAULTS.values())\n"
            "assert 'my-model' not in RECOMMENDED_DEFAULT_MODELS\n"
            "assert curated_default_target('my-model', 'dual') is None\n",
        )
        assert r.returncode == 0, r.stderr

    def test_slug_collision_with_core_is_tolerated_and_marked(self, root):
        # ⚠️ NOT an error, deliberately (#1202). Refusing at load would let a
        # routine stack update — us shipping a curated slug whose name a user
        # already registered — break their working setup with no warning, and
        # take the WHOLE local layer down with it. Core wins the lookup; the
        # local row stays loaded and marked so `--list` can show what happened.
        _write_local(root, {"vllm/minimal": dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import (\n"
            "    COMPOSE_REGISTRY, get_registry, load_local_registry, local_entries,\n"
            ")\n"
            "reg = get_registry()\n"
            "# core wins the LOOKUP view:\n"
            "assert reg['vllm/minimal'] is COMPOSE_REGISTRY['vllm/minimal']\n"
            "assert reg['vllm/minimal']['origin'] != 'local'\n"
            "# but the row is still loaded, and marked:\n"
            "assert load_local_registry()['vllm/minimal']['shadowed_by_core'] is True\n"
            "# and the MANAGEMENT view surfaces it so the user can rename it:\n"
            "row = [r for r in local_entries() if r['slug'] == 'vllm/minimal']\n"
            "assert len(row) == 1 and row[0]['shadowed'] is True\n",
        )
        assert r.returncode == 0, r.stderr

    def test_model_id_collision_with_core_refused(self, root):
        kwargs = dict(GOOD_KWARGS, model="qwen3.6-27b")
        _write_local(root, {"vllm/qwen36-clone-dual-x": kwargs})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "core model" in r.stderr

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
        _write_local(root, {LOCAL_SLUG: dict(GOOD_KWARGS, bogus_kwd=1)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "_entry kwargs" in r.stderr

    def test_bad_status_refused(self, root):
        _write_local(root, {LOCAL_SLUG: dict(GOOD_KWARGS, status="nope")})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0

    def test_self_declared_origin_refused(self, root):
        # `origin` is stamped by the loader; a local file may not claim it.
        _write_local(root, {LOCAL_SLUG: dict(GOOD_KWARGS, origin="core")})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "origin" in r.stderr

    def test_duplicate_local_model_id_refused(self, root):
        # JSON objects can't carry duplicate keys, but two slugs sharing one
        # MODEL id collide.
        kwargs2 = dict(GOOD_KWARGS, compose_path="scripts/lib/profiles-local/composes/other/x.yml")
        _write_local(
            root,
            {
                LOCAL_SLUG: dict(GOOD_KWARGS),
                "vllm/my-model-single-x": kwargs2,
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
        _write_local(root, {LOCAL_SLUG: dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import model_of_slug, slug_topology\n"
            f"assert model_of_slug('{LOCAL_SLUG}') == 'my-model'\n"
            f"assert slug_topology('{LOCAL_SLUG}') == 'dual'\n",
        )
        assert r.returncode == 0, r.stderr


class TestNamespaceHardCut:
    """#1205 removed the `local/` namespace. These lock the new shape in."""

    def test_local_namespace_refused(self, root):
        # The OLD shape. Must refuse, and must say what to do instead —
        # a user upgrading across #1205 meets this message, not a stack trace.
        _write_local(root, {"local/my-model-dual-x": dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "namespace was removed" in r.stderr
        # the refusal names the replacement slug, not just the rule
        assert "vllm/my-model-dual-x" not in r.stderr  # engine is the user's call
        assert "'<engine>/" in r.stderr
        assert "my-model-dual-x" in r.stderr

    @pytest.mark.parametrize(
        "bad_slug",
        ["no-slash", "too/many/slashes", "/leading", "trailing/"],
    )
    def test_slug_must_be_engine_slash_name(self, root, bad_slug):
        _write_local(root, {bad_slug: dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "get_registry()\n",
        )
        assert r.returncode != 0
        assert "<engine>/<name>" in r.stderr

    def test_curated_shaped_slug_is_accepted(self, root):
        # The inverse of the old `test_non_local_namespace_refused`: a slug that
        # looks exactly like a curated one is now the CORRECT shape for a local
        # row. Only the `origin` field distinguishes them.
        _write_local(root, {"curated-looking/my-model-dual-x": dict(GOOD_KWARGS)})
        r = _py(
            root,
            "from scripts.lib.profiles.compose_registry import get_registry\n"
            "e = get_registry()['curated-looking/my-model-dual-x']\n"
            "assert e['origin'] == 'local'\n",
        )
        assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# extends.d — LOCAL extensions of CORE models (local_extensions.py)
# ---------------------------------------------------------------------------
CORE_MODEL = "qwen3.6-27b"
EXT_VARIANT = "test-ext-gguf"
EXT_COMPOSE = "scripts/lib/profiles-local/composes/qwen3.6-27b/vllm/compose/dual/test-ext-gguf/base.yml"
ATTACHED_KWARGS = dict(GOOD_KWARGS, model=CORE_MODEL, weights_variant=EXT_VARIANT,
                       compose_path=EXT_COMPOSE)


def _write_ext(root, data, name=CORE_MODEL):
    p = root / "scripts/lib/profiles-local/extends.d" / f"{name}.yml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")   # JSON is valid YAML
    return p


def _good_ext(**over):
    data = {
        "schema_version": 1,
        "extends": CORE_MODEL,
        "weights": {EXT_VARIANT: {"path": "test-ext", "size_gb": 1.0, "format": "gguf",
                                  "kind": "main", "status": "community-experimental"}},
        "valid_tp_add": [3],
    }
    data.update(over)
    return data


def _write_compose(root, rel=EXT_COMPOSE):
    src = REPO / "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml"
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


class TestCoreModelExtensions:
    def test_attach_without_extension_is_refused_and_names_the_fix(self, root):
        # Negative control for everything below: no extends.d file, old behaviour.
        _write_local(root, {"vllm/qwen36-ext-dual": dict(ATTACHED_KWARGS)})
        r = _py(root, "from scripts.lib.profiles.compose_registry import get_registry\nget_registry()\n")
        assert r.returncode != 0
        assert "core model" in r.stderr and "extends.d" in r.stderr

    def test_attach_with_extension_loads(self, root):
        _write_ext(root, _good_ext())
        _write_local(root, {"vllm/qwen36-ext-dual": dict(ATTACHED_KWARGS)})
        r = _py(root,
                "from scripts.lib.profiles.compose_registry import get_registry\n"
                "e = get_registry()['vllm/qwen36-ext-dual']\n"
                f"assert e['model'] == {CORE_MODEL!r} and e['origin'] == 'local', e\n")
        assert r.returncode == 0, r.stderr

    def test_several_slugs_attach_to_one_core_model(self, root):
        _write_ext(root, _good_ext())
        _write_local(root, {
            "vllm/qwen36-ext-dual": dict(ATTACHED_KWARGS),
            "vllm/qwen36-ext-single": dict(ATTACHED_KWARGS, tp=1, default_port=20243),
        })
        r = _py(root,
                "from scripts.lib.profiles.compose_registry import get_registry\n"
                "g = get_registry()\n"
                "assert 'vllm/qwen36-ext-dual' in g and 'vllm/qwen36-ext-single' in g\n")
        assert r.returncode == 0, r.stderr

    def test_duplicate_local_only_model_id_still_refused(self, root):
        # The one-slug rule still holds for models that exist only locally.
        _write_ext(root, _good_ext())
        kwargs2 = dict(GOOD_KWARGS, compose_path="scripts/lib/profiles-local/composes/other/x.yml")
        _write_local(root, {LOCAL_SLUG: dict(GOOD_KWARGS), "vllm/my-model-single-x": kwargs2})
        r = _py(root, "from scripts.lib.profiles.compose_registry import get_registry\nget_registry()\n")
        assert r.returncode != 0
        assert "duplicate local model id" in r.stderr

    def test_profiles_gain_the_variant_and_tp_and_keep_core_fields(self, root):
        _write_ext(root, _good_ext())
        r = _py(root,
                "from scripts.lib.profiles.compat import load_profiles\n"
                f"m = load_profiles().models[{CORE_MODEL!r}]\n"
                f"assert {EXT_VARIANT!r} in m.weights, list(m.weights)\n"
                "assert 'autoround-int4' in m.weights\n"
                "assert 3 in m.valid_tp and 2 in m.valid_tp, m.valid_tp\n"
                "assert m.display_name.startswith('Qwen'), m.display_name\n")
        assert r.returncode == 0, r.stderr

    def test_extension_cannot_override_a_core_variant(self, root):
        _write_ext(root, _good_ext(weights={"autoround-int4": {"path": "hijack", "format": "gguf"}}))
        r = _py(root, "from scripts.lib.profiles.compat import load_profiles\nload_profiles()\n")
        assert r.returncode != 0
        assert "only ADD" in r.stderr

    def test_extension_must_target_a_core_model(self, root):
        _write_ext(root, _good_ext(extends="no-such-model"), name="no-such-model")
        r = _py(root, "from scripts.lib.profiles.compat import load_profiles\nload_profiles()\n")
        assert r.returncode != 0
        assert "not a core model" in r.stderr

    def test_extension_may_not_set_other_fields(self, root):
        _write_ext(root, _good_ext(display_name="Hijacked"))
        r = _py(root, "from scripts.lib.profiles.compat import load_profiles\nload_profiles()\n")
        assert r.returncode != 0
        assert "unknown key" in r.stderr

    def test_extension_name_must_match_its_target(self, root):
        _write_ext(root, _good_ext(extends="gemma-4-31b"))   # file is qwen3.6-27b.yml
        r = _py(root, "from scripts.lib.profiles.compat import load_profiles\nload_profiles()\n")
        assert r.returncode != 0
        assert "must match" in r.stderr

    def test_models_d_profile_cannot_shadow_a_core_model(self, root):
        # Before extends.d this silently REPLACED the curated profile (local merged last).
        core = (root / "scripts/lib/profiles/models/qwen3.6-27b.yml").read_text(encoding="utf-8")
        local = root / "scripts/lib/profiles-local/models.d/qwen3.6-27b.yml"
        local.parent.mkdir(parents=True, exist_ok=True)
        line = next(l for l in core.splitlines() if l.startswith("display_name:"))
        local.write_text(core.replace(line, 'display_name: "HIJACKED"'), encoding="utf-8")
        r = _py(root,
                "from scripts.lib.profiles.compat import load_profiles\n"
                f"m = load_profiles().models[{CORE_MODEL!r}]\n"
                "assert m.display_name != 'HIJACKED', 'local models.d shadowed the core profile'\n")
        assert r.returncode == 0, r.stderr

    def test_weights_catalog_lists_the_variant(self, root):
        _write_ext(root, _good_ext())
        r = subprocess.run([sys.executable, "scripts/lib/profiles/weights.py", "list", "--json"],
                           cwd=str(root), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rows = json.loads(r.stdout)
        assert any(x["model"] == CORE_MODEL and x["variant"] == EXT_VARIANT for x in rows)

    def test_registry_emit_carries_the_variant_format(self, root):
        _write_ext(root, _good_ext())
        _write_compose(root)
        _write_local(root, {"vllm/qwen36-ext-dual": dict(ATTACHED_KWARGS)})
        r = subprocess.run(["bash", "scripts/lib/registry-emit.sh", "--json"],
                           cwd=str(root), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        row = [v for v in json.loads(r.stdout)["variants"] if v["slug"] == "vllm/qwen36-ext-dual"]
        assert row and row[0]["model"] == CORE_MODEL
        assert row[0]["weights_format"] == "gguf", row[0]["weights_format"]

    def test_demote_keeps_the_extension(self, root):
        _write_ext(root, _good_ext())
        _write_local(root, {"vllm/qwen36-ext-dual": dict(ATTACHED_KWARGS)})
        r = subprocess.run([sys.executable, "scripts/lib/profiles/demote.py", "--slug",
                            "vllm/qwen36-ext-dual", "--root", str(root), "--dry-run"],
                           cwd=str(root), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "extension   KEPT" in r.stderr
        assert (root / "scripts/lib/profiles/models/qwen3.6-27b.yml").is_file()

    def test_export_refuses_an_attached_slug_with_the_reason(self, root):
        _write_ext(root, _good_ext())
        r = _py(root,
                "from pathlib import Path\n"
                "from scripts.lib.profiles.export_pr import load_local_state, Refusal\n"
                "try:\n"
                f"    load_local_state(Path('.').resolve(), {CORE_MODEL!r})\n"
                "except Refusal as e:\n"
                "    assert 'CORE model extended' in str(e), e\n"
                "else:\n"
                "    raise SystemExit('no refusal')\n")
        assert r.returncode == 0, r.stderr

    def test_extends_dir_constants_agree(self, root):
        r = _py(root,
                "from scripts.lib.profiles.compose_registry import LOCAL_EXTENDS_REL\n"
                "from scripts.lib.profiles.local_extensions import EXTENDS_DIR_REL\n"
                "assert LOCAL_EXTENDS_REL == EXTENDS_DIR_REL\n")
        assert r.returncode == 0, r.stderr
