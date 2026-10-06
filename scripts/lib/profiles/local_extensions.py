"""LOCAL extensions of CORE models: ``scripts/lib/profiles-local/extends.d/<core-id>.yml``.

WHY THIS EXISTS
---------------
A local slug used to need a local MODEL id: the registry loader refuses a local
entry whose ``model`` is a core id, and ``compat.load_profiles`` merges
``models.d`` AFTER core, so a ``models.d/<core-id>.yml`` would silently override
the curated profile. That refusal was the only thing keeping core profiles
intact, and it also meant a new engine or quant of a model we already ship (bucko
on Qwen3.8-Flash-Next, Strata on the same model) had to appear in c3 as a
separate "model" with a made-up id, instead of under the model it is.

An extension ATTACHES local slugs to a core model without redefining it:

    schema_version: 1
    extends: qwen3.8-flash-next     # a CORE model id; must equal this file's stem
    weights:                        # NEW variants only — a key core already has is refused
      ista-iq3s: {path: ..., size_gb: 83.6, format: gguf, ...}
    valid_tp_add: [1]               # optional; appended to the core valid_tp

ADDITIVE ONLY. Nothing here can change a field the curated profile sets: a
weights key that collides with core is an error, not an override, and there is no
other mergeable field. A local registry entry may then use the core model id
(``compose_registry.load_local_registry`` checks that this file exists), and
several local slugs may attach to one core model.

Readers that apply it (keep them in step — a reader that forgets it renders the
variant blank or refuses the slug): ``compat.load_profiles``, ``weights._load_models``
and ``registry-emit.sh``'s weights facet.

``extended_model_ids`` is stdlib-only on purpose: the registry loader sits on the
launcher's table path, which must run without PyYAML (#584). The YAML-reading
functions take the parsed dict, so each caller keeps its own loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

EXTENDS_DIR_REL = "scripts/lib/profiles-local/extends.d"
EXTENSION_KEYS = {"schema_version", "extends", "weights", "valid_tp_add", "notes"}


class ExtensionError(ValueError):
    """A malformed or non-additive extension. Callers decide loud vs skip."""


def extends_dir(repo_root: Path | str) -> Path:
    return Path(repo_root) / EXTENDS_DIR_REL


def extension_paths(repo_root: Path | str) -> list[Path]:
    d = extends_dir(repo_root)
    try:
        return sorted(d.glob("*.yml")) if d.is_dir() else []
    except OSError:
        return []


def extended_model_ids(repo_root: Path | str) -> set[str]:
    """Core model ids that have an extension file (file stems). Stdlib-only."""
    return {p.stem for p in extension_paths(repo_root)}


def validate(data: Any, path: Path, core_model_ids: Iterable[str],
             core_weights: dict[str, Any] | None = None,
             allowed_variant_keys: set[str] | None = None) -> dict[str, Any]:
    """Check one parsed extension and return it normalised.

    Raises ExtensionError for: a non-mapping, an unknown top-level key, ``extends``
    missing / not matching the file stem / not a core id, ``weights`` not a
    mapping of mappings, a variant key core already defines (the override this
    file type exists to make impossible), an unknown key inside a variant (when
    ``allowed_variant_keys`` is given), or a non-integer ``valid_tp_add``.
    """
    if not isinstance(data, dict):
        raise ExtensionError(f"{path}: expected a mapping")
    unknown = set(data) - EXTENSION_KEYS
    if unknown:
        raise ExtensionError(
            f"{path}: unknown key(s) {sorted(unknown)} — an extension may only set "
            f"{sorted(EXTENSION_KEYS)}; anything else would redefine the core profile"
        )
    target = data.get("extends")
    if not target or not isinstance(target, str):
        raise ExtensionError(f"{path}: 'extends:' must name a core model id")
    if target != path.stem:
        raise ExtensionError(
            f"{path}: 'extends: {target}' must match the file name ({path.stem}.yml)"
        )
    if target not in set(core_model_ids):
        raise ExtensionError(
            f"{path}: 'extends: {target}' is not a core model — extensions attach to "
            f"curated models only; a local-only model is a models.d/<id>.yml profile"
        )
    weights = data.get("weights") or {}
    if not isinstance(weights, dict) or not all(isinstance(v, dict) for v in weights.values()):
        raise ExtensionError(f"{path}: 'weights:' must map variant -> mapping")
    clash = sorted(set(weights) & set(core_weights or {}))
    if clash:
        raise ExtensionError(
            f"{path}: weights variant(s) {clash} already exist in core model "
            f"{target!r} — an extension can only ADD variants, never override one"
        )
    if allowed_variant_keys is not None:
        for variant, meta in weights.items():
            bad = set(meta) - set(allowed_variant_keys)
            if bad:
                raise ExtensionError(
                    f"{path}: weights.{variant} has unknown key(s) {sorted(bad)}"
                )
    tp_add = data.get("valid_tp_add") or []
    if not isinstance(tp_add, list) or not all(isinstance(x, int) and x > 0 for x in tp_add):
        raise ExtensionError(f"{path}: 'valid_tp_add' must be a list of positive integers")
    return {"extends": target, "weights": dict(weights), "valid_tp_add": list(tp_add)}


def merge_model_dict(core: dict[str, Any], ext: dict[str, Any]) -> dict[str, Any]:
    """A copy of a core model's raw YAML dict with the extension's additions.

    ``ext`` must come from ``validate`` (which already refused overlaps)."""
    out = dict(core)
    out["weights"] = {**dict(core.get("weights") or {}), **ext["weights"]}
    tp = list(core.get("valid_tp") or [])
    out["valid_tp"] = tp + [x for x in ext["valid_tp_add"] if x not in tp]
    return out
