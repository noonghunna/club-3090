#!/usr/bin/env python3
"""Catalog-promotion executor — writes a NEW model into the catalog.

Plan contract C4-rev (c3 ⑤ Promote-to-catalog): the cockpit's gated write plan
runs this script with the promotion spec in the environment, then chains
``diagnose-profile.sh <slug>`` + ``preflight-add-model.sh <slug>``::

    python3 scripts/lib/profiles/promote.py --spec-env C3_PROMOTE_SPEC

The spec is the JSON dict the c3 scaffold computed (``compute_promote_scaffold``
→ ``PromoteScaffold.spec``), with the user's inline edits applied::

    {
      "model_id": "...",            # [a-z0-9][a-z0-9._-]*  — <id>.yml stem
      "display_name": "...",        # REQUIRED non-empty
      "family": "...",              # REQUIRED non-empty family tag
      "arch": {...},                # optional ModelProfile arch dims (may be partial)
      "weights": {"<variant>": {...}},          # REQUIRED — the weights MAP
      "default_weight_variant": "...",          # REQUIRED — key into weights
      "compatible_drafters": [...],             # optional
      "vision_capable": true|false|null,        # null → omitted
      "compose": {"path": "...", "content": "<yaml text>"},
      "registry_entry": {"slug": "<engine>/...", "kwargs": {...}},
      "setup": {...}                            # optional C1 `setup:` block
    }

TWO write targets (--layer, DEFAULT **local**):

**local** (the community path — C4-rev).  Writes ONLY inside the gitignored
``scripts/lib/profiles-local/`` layer; NO core catalog file is ever touched:
  1. ``scripts/lib/profiles-local/models.d/<model_id>.yml``
  2. the compose file at ``spec.compose.path`` (MUST live under
     ``scripts/lib/profiles-local/composes/``)
  3. a JSON MERGE into ``scripts/lib/profiles-local/registry.local.json``
     (plain-dict ``_entry`` kwargs — never a python-source edit).
The slug MUST carry the ``local/`` namespace prefix and the entry is written
with ``status="incubating"`` (hidden from ``switch.sh --list``, ``--force`` to
launch — see compose_registry.STATUS_VALUES).

**core** (maintainer-only).  Today's curated-catalog behavior:
  1. ``scripts/lib/profiles/models/<model_id>.yml``
  2. the compose file anywhere under ``models/``
  3. an anchored ``"<slug>": _entry(...)`` insert into
     ``scripts/lib/profiles/compose_registry.py``.
Double-gated: requires ``--layer core`` AND ``C3_ALLOW_CORE_PROMOTE=1`` in the
environment. Anything else refuses BEFORE writing — community users can never
reach the core catalog through this tool.

Refuses (exit 3, writes NOTHING) on any collision: existing model yml, existing
compose path, existing registry slug (core OR local), a bad model-id/slug shape,
or a namespace violation. Internal failures AFTER a write started exit 2 — there
is NO rollback; git is the rollback for core writes and "delete the file(s)" for
local ones (stated in every run's output). ``--dry-run`` prints the plan and
writes nothing.

Success marker (the c3 parser hook)::

    PROMOTE_OK <slug>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

# Repo-root default: this file lives at <root>/scripts/lib/profiles/promote.py.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]

_REGISTRY_REL = "scripts/lib/profiles/compose_registry.py"
_REGISTRY_YAML_REL = "scripts/lib/profiles/registry.yaml"
_MODELS_DIR_REL = "scripts/lib/profiles/models"
_EMIT_REL = "scripts/lib/registry-emit.sh"

# The C4-rev LOCAL layer (gitignored except its README — see its README.md).
_LOCAL_DIR_REL = "scripts/lib/profiles-local"
_LOCAL_MODELS_REL = f"{_LOCAL_DIR_REL}/models.d"
_LOCAL_COMPOSES_REL = f"{_LOCAL_DIR_REL}/composes"
_LOCAL_REGISTRY_REL = f"{_LOCAL_DIR_REL}/registry.local.json"
_LOCAL_SLUG_PREFIX = "local/"
# Local entries ALWAYS land here (the enum's pre-experimental rung: hidden from
# switch.sh --list, --force to launch). Not user-selectable — promote up the
# enum by editing registry.local.json once the model validates.
_LOCAL_FORCED_STATUS = "incubating"
# Maintainer gate for the CORE write path (C4-rev): both the flag AND --layer
# core must be present, or the core catalog is untouchable.
_CORE_GATE_ENV = "C3_ALLOW_CORE_PROMOTE"

_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*/[a-z0-9][a-z0-9._-]*$")

EXIT_COLLISION = 3
EXIT_INTERNAL = 2


class Refusal(Exception):
    """A collision / bad-shape refusal → exit 3, nothing written."""


class InternalError(Exception):
    """A post-write failure → exit 2 (git is the rollback)."""


def _info(msg: str) -> None:
    print(f"[promote] {msg}")




def _yaml_scalar(v: Any) -> str:
    """Render a scalar as a YAML token. json.dumps produces valid YAML for the
    string/number/bool shapes the spec carries (UTF-8 output, double-quoted
    escapes YAML accepts)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(v, ensure_ascii=False)


def _yaml_dump(data: Any, indent: int = 0) -> list[str]:
    """Minimal deterministic YAML emitter for the profile skeleton's shape
    (scalars / string-lists / one-level string→dict maps / one-level
    string→string maps). Key order = insertion order."""
    pad = " " * indent
    lines: list[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and v:
                lines.append(f"{pad}{k}:")
                lines.extend(_yaml_dump(v, indent + 2))
            elif isinstance(v, list) and v:
                lines.append(f"{pad}{k}:")
                for item in v:
                    lines.append(f"{pad}  - {_yaml_scalar(item)}")
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
    elif isinstance(data, list):
        for item in data:
            lines.append(f"{pad}- {_yaml_scalar(item)}")
    return lines


def render_profile_yaml(spec: dict) -> str:
    """The ModelProfile YAML for the spec — canonical ModelProfile schema keys
    (compat.ModelProfile: ``num_attn_heads`` / ``head_dim_attn`` /
    ``max_ctx_supported``). Only arch dims WITH values are emitted; the
    remaining family-specific extras are added by hand afterwards (same as
    every shipped profile)."""
    mid = spec["model_id"]
    arch = spec.get("arch") or {}
    weights = spec["weights"]

    root: dict[str, Any] = {
        "schema_version": 1,
        "id": mid,
        "display_name": spec["display_name"],
        "family": spec["family"],
    }
    # Arch dims — canonical schema names, only when the fact exists.
    for spec_key, yml_key in (
        ("hidden_size", "hidden_size"),
        ("num_hidden_layers", "num_hidden_layers"),
        ("num_attn_heads", "num_attn_heads"),
        ("num_kv_heads", "num_kv_heads"),
        ("head_dim_attn", "head_dim_attn"),
        ("max_ctx_supported", "max_ctx_supported"),
        # Required by compat.load_profiles for the CORE catalog — a core write
        # whose spec omits it fails the post-write registry-emit gate (exit 2),
        # which is the correct outcome: an incomplete profile never lands in
        # the curated catalog. Local-layer profiles bypass load_profiles.
        ("attention_k_eq_v", "attention_k_eq_v"),
    ):
        if arch.get(spec_key) is not None:
            root[yml_key] = arch[spec_key]
    if arch.get("valid_tp"):
        root["valid_tp"] = list(arch["valid_tp"])
    # compat.ModelProfile REQUIRES attention_k_eq_v — emit the conservative
    # default when the spec doesn't carry the fact (corrected by hand / flagged
    # by the post-write diagnose + kv-calc gates during validation).
    root["attention_k_eq_v"] = bool(arch.get("attention_k_eq_v"))

    root["weights"] = weights
    root["default_weight_variant"] = spec["default_weight_variant"]
    root["compatible_drafters"] = list(spec.get("compatible_drafters") or [])
    if spec.get("vision_capable") is not None:
        root["vision_capable"] = bool(spec["vision_capable"])
    if spec.get("setup"):
        root["setup"] = spec["setup"]

    where = (
        f"{_LOCAL_MODELS_REL}/{mid}.yml"
        if spec.get("_layer") == "local"
        else f"{_MODELS_DIR_REL}/{mid}.yml"
    )
    header = (
        f"# {spec['display_name']} — scaffolded by c3 Promote-to-catalog\n"
        f"# ({where} via scripts/lib/profiles/promote.py).\n"
        "# INCUBATING: fill family-specific arch extras + notes by hand, then\n"
        "# promote the status enum as the model validates (ADDING_MODELS.md).\n"
    )
    return header + "\n".join(_yaml_dump(root)) + "\n"




def _import_compose_registry(root: Path, *, merged: bool):
    """Import the CURRENT registry (pre-write) for the collision checks.

    merged=True returns the C4-rev merged view (core + local layer) so a local
    write refuses a slug that exists ANYWHERE; merged=False is the core dict
    (used by the core path, which additionally checks the local file below)."""
    sys.path.insert(0, str(root))
    try:
        if merged:
            from scripts.lib.profiles.compose_registry import get_registry  # noqa: E402

            return get_registry()
        from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402

        return COMPOSE_REGISTRY
    except Exception as exc:
        raise Refusal(f"compose_registry.py does not import: {exc}")


def _load_local_raw(root: Path) -> dict:
    """The RAW registry.local.json dict ({} when absent). Bad JSON is a
    refusal — we never overwrite a file we cannot parse."""
    path = root / _LOCAL_REGISTRY_REL
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal(f"{path} is unreadable — fix or remove it first: {exc}")
    if not isinstance(raw, dict):
        raise Refusal(f"{path}: expected a JSON object of {{slug: kwargs}}")
    return raw


def _append_registry_entry(root: Path, slug: str, kwargs: dict) -> None:
    """Merge the new entry into <root>'s registry.yaml (canonical rewrite).

    The old CORE path did an anchored textual insert into compose_registry.py
    — fragile by construction (brace anchors, comma repair, source parsing).
    Now: load the DATA file through the module's own reader, append the
    `_entry(**kwargs)` row, re-dump with the ONE shared deterministic writer,
    and atomically replace (temp sibling + os.replace — a failed encode can
    never truncate the catalog). Import-time validation of the result happens
    in the post-write checks; a kwargs problem surfaces THERE as exit 2."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from scripts.lib.profiles.compose_registry import (  # noqa: E402
            dump_registry_yaml,
            load_registry_data,
        )
    except Exception as exc:
        raise InternalError(f"compose_registry.py does not import: {exc}")
    path = root / _REGISTRY_YAML_REL
    data = load_registry_data(path)
    if slug in data["entries"]:
        raise Refusal(f"registry slug already exists in {_REGISTRY_YAML_REL}: {slug}")
    data["entries"][slug] = dict(kwargs)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(dump_registry_yaml(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise InternalError(f"could not rewrite {_REGISTRY_YAML_REL}: {exc}")


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _post_write_checks(root: Path, spec: dict, profile_path: Path) -> None:
    """Import-sanity + registry-emit re-check. Any failure exits 2 — the write
    already happened, so the message points at the rollback."""
    slug = spec["registry_entry"]["slug"]
    mid = spec["model_id"]
    rollback_hint = (
        f"revert with git: git checkout -- {_REGISTRY_YAML_REL}"
        if spec.get("_layer") == "core"
        else f"delete the written files ({_LOCAL_DIR_REL}/…)"
    )

    # 1) The mutated registry imports and carries the new slug (merged view —
    #    covers BOTH layers; the local JSON merge is visible via get_registry).
    chk = _run(
        [
            "python3",
            "-c",
            "from scripts.lib.profiles.compose_registry import get_registry; "
            f"assert {slug!r} in get_registry()",
        ],
        cwd=root,
    )
    if chk.returncode != 0:
        raise InternalError(
            "registry failed import-sanity after write:\n"
            + (chk.stderr.strip() or chk.stdout.strip())
            + f"\n[promote] NO ROLLBACK — {rollback_hint}"
        )

    # 2) The written profile parses as YAML with the right id.
    try:
        import yaml  # noqa: E402
    except Exception as exc:
        raise InternalError(f"PyYAML unavailable, cannot sanity-check the profile: {exc}")
    try:
        data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        assert data.get("id") == mid, f"profile id {data.get('id')!r} != {mid!r}"
    except Exception as exc:
        raise InternalError(
            f"written profile failed sanity: {exc}\n"
            f"[promote] NO ROLLBACK — delete {profile_path}"
        )

    # 3) The shared emitter sees the new slug end-to-end.
    emit = _run(["bash", _EMIT_REL, "--json", str(root)], cwd=root)
    if emit.returncode != 0:
        raise InternalError(
            "registry-emit.sh --json failed after write:\n"
            + (emit.stderr.strip() or emit.stdout.strip())[-800:]
            + f"\n[promote] NO ROLLBACK — {rollback_hint}"
        )
    try:
        payload = json.loads(emit.stdout)
        slugs = {v.get("slug") for v in payload.get("variants") or []}
        assert slug in slugs, f"slug {slug!r} absent from registry-emit output"
    except Exception as exc:
        raise InternalError(
            f"registry-emit re-check failed: {exc}\n"
            f"[promote] NO ROLLBACK — {rollback_hint}"
        )


def _check_common_shape(spec: Any) -> dict:
    """Spec-shape validation shared by BOTH layers. Raises Refusal."""
    if not isinstance(spec, dict):
        raise Refusal("spec is not a JSON object")

    mid = spec.get("model_id")
    if not isinstance(mid, str) or not _MODEL_ID_RE.match(mid or ""):
        raise Refusal(f"bad model_id {mid!r} — expected [a-z0-9][a-z0-9._-]*")
    for key in ("display_name", "family"):
        v = spec.get(key)
        if not isinstance(v, str) or not v.strip():
            raise Refusal(f"spec.{key} is required and must be a non-empty string")
    weights = spec.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise Refusal("spec.weights must be a non-empty map")
    dwv = spec.get("default_weight_variant")
    if dwv not in weights:
        raise Refusal(f"spec.default_weight_variant {dwv!r} is not a weights key")

    compose = spec.get("compose") or {}
    cpath = compose.get("path")
    ccontent = compose.get("content")
    if not isinstance(cpath, str) or not cpath:
        raise Refusal("spec.compose.path is required")
    if not isinstance(ccontent, str) or not ccontent.strip():
        raise Refusal("spec.compose.content is required (the compose YAML text)")
    entry = spec.get("registry_entry") or {}
    slug = entry.get("slug")
    kwargs = entry.get("kwargs")
    if not isinstance(slug, str) or not _SLUG_RE.match(slug or ""):
        raise Refusal(f"bad registry slug {slug!r} — expected <engine>/<name>")
    if not isinstance(kwargs, dict) or "model" not in kwargs or "compose_path" not in kwargs:
        raise Refusal("spec.registry_entry.kwargs must be a dict incl. model + compose_path")
    if kwargs.get("model") != mid:
        raise Refusal(f"registry kwargs.model {kwargs.get('model')!r} != model_id {mid!r}")
    return spec


def _refuse_if_path_escapes(rel: str, base_rel: str, what: str) -> None:
    """Refuse a spec-supplied relative path that escapes (or sits outside) the
    given layer directory — no traversal out of scripts/lib/profiles-local/."""
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise Refusal(f"{what} must be a repo-relative path without '..': {rel!r}")
    if not p.is_relative_to(Path(base_rel)):
        raise Refusal(f"{what} must live under {base_rel}/ (got {rel!r})")


def validate_spec(spec: Any, root: Path, layer: str) -> dict:
    """Shape + collision validation. Raises Refusal (exit 3) — nothing written."""
    spec = _check_common_shape(spec)
    spec["_layer"] = layer
    mid = spec["model_id"]
    slug = spec["registry_entry"]["slug"]
    kwargs = spec["registry_entry"]["kwargs"]
    cpath = spec["compose"]["path"]

    if layer == "local":
        # ── Namespace + containment (C4-rev): local writes NEVER leave the layer.
        if not slug.startswith(_LOCAL_SLUG_PREFIX):
            raise Refusal(
                f"local-layer slugs must carry the {_LOCAL_SLUG_PREFIX!r} namespace "
                f"(got {slug!r}); use --layer core for a curated engine slug"
            )
        _refuse_if_path_escapes(cpath, _LOCAL_COMPOSES_REL, "spec.compose.path")
        if kwargs.get("compose_path") != cpath:
            raise Refusal(
                f"registry kwargs.compose_path {kwargs.get('compose_path')!r} != "
                f"compose.path {cpath!r}"
            )
        # Force the incubating rung — a local entry never ships functional.
        kwargs["status"] = _LOCAL_FORCED_STATUS

        profile_path = root / _LOCAL_MODELS_REL / f"{mid}.yml"
        if profile_path.exists():
            raise Refusal(f"local model profile already exists: {profile_path}")
        compose_abs = root / cpath
        if compose_abs.exists():
            raise Refusal(f"compose path already exists: {compose_abs}")

        # Collide against EVERYTHING (core + local) via the merged accessor…
        registry = _import_compose_registry(root, merged=True)
        if slug in registry:
            raise Refusal(f"registry slug already exists: {slug}")
        # …and against core/local MODEL IDS (the loader refuses these loudly, so
        # refuse here where the message can name the exact fix).
        core_models = {e.get("model") for e in _import_compose_registry(root, merged=False).values()}
        if mid in core_models:
            raise Refusal(
                f"model_id {mid!r} collides with a CORE model — pick another id"
            )
        local_raw = _load_local_raw(root)
        if any(k.get("model") == mid for k in local_raw.values()):
            raise Refusal(f"local model id already exists in {_LOCAL_REGISTRY_REL}: {mid!r}")
        if slug in local_raw:
            raise Refusal(f"registry slug already exists in {_LOCAL_REGISTRY_REL}: {slug}")

        # Dry-wrap the kwargs EXACTLY like the loader will — a bad kwarg or a
        # bad status must refuse BEFORE anything is written.
        sys.path.insert(0, str(root))
        try:
            from scripts.lib.profiles.compose_registry import _entry  # noqa: E402

            _entry(**kwargs)
        except TypeError as exc:
            raise Refusal(f"registry kwargs are not valid _entry kwargs: {exc}")
        except ValueError as exc:
            raise Refusal(f"registry kwargs rejected by _entry: {exc}")
        return spec

    # ── layer == "core": maintainer-gated curated-catalog write ──────────────
    if os.environ.get(_CORE_GATE_ENV) != "1":
        raise Refusal(
            f"core-catalog writes are maintainer-gated: re-run with --layer core "
            f"AND {_CORE_GATE_ENV}=1 in the environment (community users: use the "
            f"default --layer local)"
        )
    if slug.startswith(_LOCAL_SLUG_PREFIX):
        raise Refusal(
            f"the {_LOCAL_SLUG_PREFIX!r} namespace belongs to the LOCAL layer — "
            "core slugs are <engine>/<name>"
        )
    profile_path = root / _MODELS_DIR_REL / f"{mid}.yml"
    if profile_path.exists():
        raise Refusal(f"model profile already exists: {profile_path}")
    compose_abs = root / cpath
    if compose_abs.exists():
        raise Refusal(f"compose path already exists: {compose_abs}")
    registry = _import_compose_registry(root, merged=True)
    if slug in registry:
        raise Refusal(f"registry slug already exists: {slug}")
    local_raw = _load_local_raw(root)
    if slug in local_raw:
        raise Refusal(f"registry slug already exists in {_LOCAL_REGISTRY_REL}: {slug}")
    return spec


def plan_lines(root: Path, spec: dict) -> list[str]:
    """The dry-run plan (also printed before a real run)."""
    mid = spec["model_id"]
    slug = spec["registry_entry"]["slug"]
    if spec["_layer"] == "local":
        return [
            f"plan (layer LOCAL — gitignored {_LOCAL_DIR_REL}/, root {root}):",
            f"  write {_LOCAL_MODELS_REL}/{mid}.yml",
            f"  write {spec['compose']['path']}",
            f"  merge entry into {_LOCAL_REGISTRY_REL} (JSON — no source edit)",
            f"  slug: {slug}  model: {mid}  variant: {spec['default_weight_variant']}"
            f"  status: {_LOCAL_FORCED_STATUS}",
            "  then: get_registry() import-sanity + registry-emit.sh --json re-check",
            "  NO core file is touched. Rollback: delete the written files.",
        ]
    return [
        f"plan (layer CORE — maintainer, root {root}):",
        f"  write {_MODELS_DIR_REL}/{mid}.yml",
        f"  write {spec['compose']['path']}",
        f"  merge entry into {_REGISTRY_YAML_REL} (canonical rewrite — no source edit)",
        f"  slug: {slug}  model: {mid}  variant: {spec['default_weight_variant']}",
        "  then: import-sanity + registry-emit.sh --json re-check",
        "  NO ROLLBACK — git is the rollback (dry-run writes nothing).",
    ]


def _write_local(root: Path, spec: dict) -> tuple[Path, Path]:
    """The THREE local-layer artifacts. Returns (profile_path, compose_path)."""
    mid = spec["model_id"]
    profile_path = root / _LOCAL_MODELS_REL / f"{mid}.yml"
    compose_abs = root / spec["compose"]["path"]

    # ── Write 1/3: the ModelProfile ─────────────────────────────────────────
    try:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(render_profile_yaml(spec), encoding="utf-8")
        _info(f"wrote {profile_path}")
    except OSError as exc:
        raise InternalError(f"could not write {profile_path}: {exc}")

    # ── Write 2/3: the compose ──────────────────────────────────────────────
    try:
        compose_abs.parent.mkdir(parents=True, exist_ok=True)
        compose_abs.write_text(spec["compose"]["content"], encoding="utf-8")
        _info(f"wrote {compose_abs}")
    except OSError as exc:
        raise InternalError(f"could not write {compose_abs}: {exc}")

    # ── Write 3/3: the JSON merge into registry.local.json ──────────────────
    reg_path = root / _LOCAL_REGISTRY_REL
    try:
        raw = _load_local_raw(root)
        raw[spec["registry_entry"]["slug"]] = spec["registry_entry"]["kwargs"]
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        reg_path.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _info(f"merged entry into {reg_path}")
    except OSError as exc:
        raise InternalError(f"could not rewrite {reg_path}: {exc}")
    return profile_path, compose_abs


def _write_core(root: Path, spec: dict) -> tuple[Path, Path]:
    """Today's curated-catalog write (maintainer-gated). Returns paths."""
    mid = spec["model_id"]
    profile_path = root / _MODELS_DIR_REL / f"{mid}.yml"
    compose_abs = root / spec["compose"]["path"]

    # ── Write 1/3: the ModelProfile ─────────────────────────────────────────
    try:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(render_profile_yaml(spec), encoding="utf-8")
        _info(f"wrote {profile_path}")
    except OSError as exc:
        raise InternalError(f"could not write {profile_path}: {exc}")

    # ── Write 2/3: the compose ──────────────────────────────────────────────
    try:
        compose_abs.parent.mkdir(parents=True, exist_ok=True)
        compose_abs.write_text(spec["compose"]["content"], encoding="utf-8")
        _info(f"wrote {compose_abs}")
    except OSError as exc:
        raise InternalError(f"could not write {compose_abs}: {exc}")

    # ── Write 3/3: the registry.yaml entry merge (data, not source) ─────────
    try:
        _append_registry_entry(
            root, spec["registry_entry"]["slug"], spec["registry_entry"]["kwargs"]
        )
        _info(f"merged entry into {_REGISTRY_YAML_REL}")
    except OSError as exc:
        raise InternalError(f"could not rewrite {_REGISTRY_YAML_REL}: {exc}")
    return profile_path, compose_abs


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="promote.py",
        description=(
            "Write a NEW model into the catalog (C4-rev executor). "
            "DEFAULT: the gitignored LOCAL layer — core is maintainer-gated."
        ),
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec-env", metavar="VAR", help="read the promotion spec JSON from this env var")
    g.add_argument("--spec-file", metavar="PATH", help="read the promotion spec JSON from this file")
    ap.add_argument(
        "--layer",
        choices=("local", "core"),
        default="local",
        help="write target: local (default — scripts/lib/profiles-local/, never "
        "touches core) or core (curated catalog; needs " + _CORE_GATE_ENV + "=1)",
    )
    ap.add_argument("--root", default=str(_DEFAULT_ROOT), help="repo root (default: this checkout)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    try:
        if args.spec_env:
            raw_spec = os.environ.get(args.spec_env)
            if not raw_spec:
                raise Refusal(f"env var {args.spec_env} is empty/unset — no spec to promote")
            spec = json.loads(raw_spec)
        else:
            spec = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
        spec = validate_spec(spec, root, args.layer)
    except Refusal as exc:
        print(f"[promote] REFUSED: {exc}", file=sys.stderr)
        return EXIT_COLLISION
    except json.JSONDecodeError as exc:
        print(f"[promote] REFUSED: spec is not valid JSON: {exc}", file=sys.stderr)
        return EXIT_COLLISION

    slug = spec["registry_entry"]["slug"]

    for ln in plan_lines(root, spec):
        _info(ln)
    if args.dry_run:
        _info("dry-run — nothing written.")
        return 0

    if spec["_layer"] == "local":
        profile_path, _compose_abs = _write_local(root, spec)
    else:
        profile_path, _compose_abs = _write_core(root, spec)

    # ── Post-write sanity ───────────────────────────────────────────────────
    _post_write_checks(root, spec, profile_path)
    _info(f"import-sanity + registry-emit re-check passed for {slug}")
    print(f"PROMOTE_OK {slug}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    try:
        raise SystemExit(main())
    except Refusal as exc:  # defensive: raised outside the validate window
        print(f"[promote] REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_COLLISION)
    except InternalError as exc:
        print(f"[promote] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_INTERNAL)
