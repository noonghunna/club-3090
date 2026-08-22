#!/usr/bin/env python3
"""export_pr.py tests — LOCAL layer → ready-to-commit CORE PR bundle.

Runs the CLI as a SUBPROCESS against a throwaway repo copy (the same harness
as test_promote.py): a synthetic profiles-local/ layer is seeded in tmp, the
bundle lands under a second tmp dir, and NOTHING inside the repo root may be
touched by an export.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# compose_registry.py text is read for the patch context.
_COPY_TREES = ("scripts/lib",)


@pytest.fixture()
def root(tmp_path):
    # The throwaway repo lives in a SIBLING dir of tmp_path so the --out
    # bundle (tmp_path/bundle) is provably outside the exported repo.
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel in _COPY_TREES:
        shutil.copytree(
            REPO / rel,
            repo / rel,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    return repo


MID = "my-model"
QUANT = "autoround-int4"
LOCAL_SLUG = f"local/{MID}-dual-{QUANT}"
LOCAL_COMPOSE_REL = (
    f"scripts/lib/profiles-local/composes/{MID}/vllm/compose/dual/{QUANT}/base.yml"
)

COMPOSE_TEXT = (
    "# Profile (at-a-glance):\n"
    "#   Model:     My Model\n"
    "#   Status:    🐣 Incubating\n"
    "#   Caveats:   nothing boot-validated yet\n"
    "services:\n"
    "  vllm:\n"
    "    image: vllm/vllm-openai:v0.27.1\n"
    "    container_name: vllm-my-model-dual\n"
    '    ports:\n'
    '      - "${PORT:-20242}:8000"\n'
)


def _profile_text(**over) -> str:
    parts = [
        "schema_version: 1",
        f"id: {MID}",
        over.get("display_name", "display_name: My Model"),
        over.get("family", "family: generic-dense"),
        "num_hidden_layers: 40",
        "num_kv_heads: 4",
        "num_attention_heads: 24",
        "head_dim: 128",
        "weights:",
        f"  {QUANT}:",
        f"    path: {MID}-{QUANT}",
        f"    local_subdir: {MID}-{QUANT}",
        "    size_gb: 12.5",
        '    format: "autoround"',
        "    status: incubating",
        '    hf_repo: "org/model"',
        '    files: ["*.safetensors"]',
        '    engine: "vllm"',
        '    kind: "main"',
        '    verify_glob: "*.safetensors"',
        f"default_weight_variant: {QUANT}",
        "compatible_drafters: []",
        "vision_capable: false",
    ]
    return "\n".join(parts) + "\n"


def _entry_kwargs() -> dict:
    return {
        "model": MID,
        "weights_variant": QUANT,
        "workload": "long-ctx-single",
        "engine": "vllm-stable",
        "drafter": None,
        "kv_format": "fp8_e5m2",
        "tp": 2,
        "max_ctx": 131072,
        "max_num_seqs": 2,
        "mem_util": 0.92,
        "compose_path": LOCAL_COMPOSE_REL,
        "default_port": 20242,
        "kvcalc_key": f"{MID}:dual",
        "status": "incubating",
        "status_note": "community loop entry",
    }


def _seed_local(root: Path, *, profile_text=None, entry_kwargs=None):
    """Seed the gitignored LOCAL layer exactly as c3 ⑤ Promote writes it."""
    (root / "scripts/lib/profiles-local/models.d").mkdir(parents=True, exist_ok=True)
    (root / "scripts/lib/profiles-local/models.d" / f"{MID}.yml").write_text(
        profile_text or _profile_text(), encoding="utf-8"
    )
    comp_rel = (
        (entry_kwargs or {}).get("compose_path") or LOCAL_COMPOSE_REL
    )
    comp = root / comp_rel
    comp.parent.mkdir(parents=True, exist_ok=True)
    comp.write_text(COMPOSE_TEXT, encoding="utf-8")
    raw = {}
    reg = root / "scripts/lib/profiles-local/registry.local.json"
    if reg.exists():
        raw = json.loads(reg.read_text(encoding="utf-8"))
    raw[LOCAL_SLUG] = entry_kwargs or _entry_kwargs()
    reg.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _spec_file(root: Path) -> Path:
    f = root / "export-spec.json"
    payload = json.dumps({"model_id": MID})
    # Idempotent — a rewrite would bump the mtime the no-repo-touches test
    # snapshots around.
    if not f.exists() or f.read_text(encoding="utf-8") != payload + "\n":
        f.write_text(payload + "\n", encoding="utf-8")
    return f




def _run_cli(root, out, *extra, env_extra=None):
    env = dict(os.environ)
    env.pop("C3_ALLOW_CORE_PROMOTE", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/lib/profiles/export_pr.py"),
            "--spec-file", str(_spec_file(root)),
            "--root", str(root),
            "--out", str(out),
            *extra,
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
    )


class TestHappyPath:
    def test_exports_three_core_artifacts(self, root, tmp_path):
        _seed_local(root)
        out = tmp_path / "bundle"
        res = _run_cli(root, out)
        assert res.returncode == 0, res.stderr + res.stdout
        assert f"EXPORT_OK {out}" in res.stdout

        # Artifact 1 — the ModelProfile, copied verbatim to the CORE path.
        prof = out / "models" / f"{MID}.yml"
        assert prof.exists()
        assert prof.read_text(encoding="utf-8") == _profile_text()

        # Artifact 2 — the compose, translated to the CORE layout path
        # (profiles-local/composes/<rest> → models/<rest>).
        comp = out / "models" / MID / "vllm/compose/dual" / QUANT / "base.yml"
        assert comp.exists(), sorted(p.relative_to(out) for p in out.rglob("*"))
        assert comp.read_text(encoding="utf-8") == COMPOSE_TEXT

        # Artifact 3 — the compose_registry patch: core slug namespace
        # (local/ stripped → engine prefix), translated compose_path, the
        # _entry kwargs, and the setup-block notes.
        patch = out / "compose_registry.patch"
        assert patch.exists()
        text = patch.read_text(encoding="utf-8")
        assert "--- a/scripts/lib/profiles/compose_registry.py" in text
        assert "+    \"vllm/%s-dual-%s\": _entry(" % (MID, QUANT) in text
        assert f'"compose_path": ' in text or "compose_path=" in text
        assert f'"models/{MID}/vllm/compose/dual/{QUANT}/base.yml"' in text.replace(
            "'", '"'
        ) or f"models/{MID}/vllm/compose/dual/{QUANT}/base.yml" in text
        assert LOCAL_COMPOSE_REL not in text      # no LOCAL path leaks
        assert "setup:" in text                   # setup-block notes present
        assert "incubating" in text               # status guidance present
        # The patch must be APPLYABLE against a real checkout — "ready to
        # commit" means git apply --check passes, not merely that it parses.
        applied = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=str(root), capture_output=True, text=True,
        )
        assert applied.returncode == 0, applied.stdout + applied.stderr

    def test_export_touches_nothing_inside_the_repo(self, root, tmp_path):
        _seed_local(root)
        _spec_file(root)   # pre-write: the CLI must not touch it either
        out = tmp_path / "bundle"
        def _snap():
            return {
                p.relative_to(root): p.stat().st_mtime
                for p in root.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            }

        before = _snap()
        res = _run_cli(root, out)
        assert res.returncode == 0, res.stderr
        after = _snap()
        # registry.local.json is READ, never rewritten; compose_registry.py is
        # only diffed, never edited.
        changed = {k for k in after if before.get(k) != after[k]}
        assert not changed, changed

    def test_llama_family_defaults_kvcalc_skip(self, root, tmp_path):
        kw = _entry_kwargs()
        kw["engine"] = "llama-cpp-local"
        kw.pop("kvcalc_key")
        local_compose = (
            f"scripts/lib/profiles-local/composes/{MID}/llama-cpp/compose/single/"
            "unsloth-q4km/base.yml"
        )
        kw["compose_path"] = local_compose
        _seed_local(root, entry_kwargs=kw)
        (root / local_compose).write_text(COMPOSE_TEXT, encoding="utf-8")
        out = tmp_path / "bundle"
        res = _run_cli(root, out)
        assert res.returncode == 0, res.stderr + res.stdout
        patch = (out / "compose_registry.patch").read_text(encoding="utf-8")
        assert "kvcalc_key='SKIP'" in patch or 'kvcalc_key="SKIP"' in patch
        # llama.cpp FILESYSTEM dir → llamacpp slug prefix (ADDING_MODELS Step 3).
        assert f"+    \"llamacpp/{MID}-dual-{QUANT}\": _entry(" in patch
        assert (out / "models" / MID / "llama-cpp/compose/single"
                / "unsloth-q4km/base.yml").exists()


class TestRefusals:
    def test_incomplete_entry_refused_and_writes_nothing(self, root, tmp_path):
        # The scaffold's leftover <...> family placeholder — the exact state a
        # contributor exports too early.
        _seed_local(root, profile_text=_profile_text(family="family: <family-tag>"))
        out = tmp_path / "bundle"
        res = _run_cli(root, out)
        assert res.returncode == 3
        assert "REFUSED" in res.stderr
        assert "family" in res.stderr
        assert not out.exists() or not any(out.iterdir())

    def test_missing_local_model_refused(self, root, tmp_path):
        res = _run_cli(root, tmp_path / "bundle")   # nothing seeded
        assert res.returncode == 3
        assert "no LOCAL model profile" in res.stderr

    def test_vllm_without_kvcalc_key_refused(self, root, tmp_path):
        kw = _entry_kwargs()
        del kw["kvcalc_key"]
        _seed_local(root, entry_kwargs=kw)
        res = _run_cli(root, tmp_path / "bundle")
        assert res.returncode == 3
        assert "kvcalc_key" in res.stderr

    def test_gguf_without_verify_glob_refused(self, root, tmp_path):
        prof = _profile_text().replace(
            'verify_glob: "*.safetensors"', ""
        ).replace('format: "autoround"', 'format: "gguf"').replace(
            'files: ["*.safetensors"]', 'files: ["model-q4km.gguf"]'
        )
        _seed_local(root, profile_text=prof)
        res = _run_cli(root, tmp_path / "bundle")
        assert res.returncode == 3
        assert "verify_glob" in res.stderr

    def test_compose_without_status_header_refused(self, root, tmp_path):
        _seed_local(root)
        (root / LOCAL_COMPOSE_REL).write_text(
            "services:\n  vllm:\n    image: x\n", encoding="utf-8"
        )
        res = _run_cli(root, tmp_path / "bundle")
        assert res.returncode == 3
        assert "Status:" in res.stderr

    def test_missing_compose_on_disk_refused(self, root, tmp_path):
        _seed_local(root)
        (root / LOCAL_COMPOSE_REL).unlink()
        res = _run_cli(root, tmp_path / "bundle")
        assert res.returncode == 3
        assert "missing on disk" in res.stderr

    def test_no_local_entries_for_model_refused(self, root, tmp_path):
        _seed_local(root)
        reg = root / "scripts/lib/profiles-local/registry.local.json"
        reg.unlink()
        res = _run_cli(root, tmp_path / "bundle")
        assert res.returncode == 3
        assert "no registry.local.json entries" in res.stderr


class TestModes:
    def test_dry_run_writes_nothing(self, root, tmp_path):
        _seed_local(root)
        out = tmp_path / "bundle"
        res = _run_cli(root, out, "--dry-run")
        assert res.returncode == 0, res.stderr
        assert "dry-run — nothing written." in res.stdout
        assert "EXPORT_OK" not in res.stdout
        assert not out.exists() or not any(out.iterdir())

    def test_check_validates_without_writing(self, root, tmp_path):
        _seed_local(root)
        out = tmp_path / "bundle"
        res = _run_cli(root, out, "--check")
        assert res.returncode == 0, res.stderr
        assert f"CHECK_OK {MID}" in res.stdout
        assert not out.exists() or not any(out.iterdir())

    def test_check_lists_every_gap(self, root, tmp_path):
        _seed_local(
            root,
            profile_text=_profile_text(
                display_name="display_name: <Human-readable name>",
                family="family: <family-tag>",
            ),
        )
        res = _run_cli(root, tmp_path / "bundle", "--check")
        assert res.returncode == 3
        assert "display_name" in res.stderr and "family" in res.stderr

    def test_spec_env_input(self, root, tmp_path):
        _seed_local(root)
        env_extra = {"C3_EXPORT_SPEC": json.dumps({"model_id": MID})}
        res = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/lib/profiles/export_pr.py"),
                "--spec-env", "C3_EXPORT_SPEC",
                "--root", str(root),
                "--out", str(tmp_path / "bundle"),
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            env={**os.environ, **env_extra},
        )
        assert res.returncode == 0, res.stderr + res.stdout
