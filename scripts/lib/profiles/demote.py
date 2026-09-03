#!/usr/bin/env python3
"""Local-layer REMOVAL — unregister a model that ``promote.py`` wrote.

``promote.py`` writes three things for a local model and says, in its own
docstring, that the rollback is "delete the file(s)". That instruction is the
whole reason this script exists: doing it by hand means deleting

  1. ``scripts/lib/profiles-local/models.d/<model_id>.yml``
  2. ``scripts/lib/profiles-local/composes/<model_id>/…``
  3. the slug's entry in ``scripts/lib/profiles-local/registry.local.json``

and **missing (3) is the failure that bites**: the files are gone but the merged
registry still advertises the slug, so every launcher offers a config whose
compose no longer exists. Removing (3) alone is just as bad in the other
direction — an orphaned compose tree nothing references.

    python3 scripts/lib/profiles/demote.py --slug local/my-model
    python3 scripts/lib/profiles/demote.py --slug local/my-model --dry-run

CORE IS NEVER TOUCHED. There is no ``--layer core``: a curated entry lives in
git-tracked files and version control is its removal tool. Asking to remove a
non-``local/`` slug is refused before anything is read.

Exit codes match promote.py: 0 success, 3 refusal (nothing written), 2 internal
failure after removal started.

Success marker (the c3 parser hook)::

    DEMOTE_OK <slug>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve()
_DEFAULT_ROOT = _HERE.parents[3]

_LOCAL_DIR_REL = "scripts/lib/profiles-local"
_LOCAL_MODELS_REL = f"{_LOCAL_DIR_REL}/models.d"
_LOCAL_COMPOSES_REL = f"{_LOCAL_DIR_REL}/composes"
_LOCAL_REGISTRY_REL = f"{_LOCAL_DIR_REL}/registry.local.json"

_LOCAL_NS = "local/"

EXIT_COLLISION = 3
EXIT_INTERNAL = 2


class Refusal(Exception):
    """Nothing has been removed; the caller can fix the input and retry."""


class InternalError(Exception):
    """Removal had started — state may be partial."""


def _info(msg: str) -> None:
    print(f"[demote] {msg}", file=sys.stderr)


def _load_local_raw(root: Path) -> dict:
    """The RAW registry.local.json dict ({} when absent)."""
    path = root / _LOCAL_REGISTRY_REL
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal(f"{path} is unreadable — fix or remove it first: {exc}")
    if not isinstance(data, dict):
        raise Refusal(f"{path} is not a JSON object")
    return data


def _model_id_for(slug: str, entry: dict) -> str:
    """The model_id behind a local slug.

    Prefer the registry entry's own `model` kwarg; fall back to the slug tail.
    Both are recorded by promote.py, and disagreeing about which files belong to
    a slug is exactly how a removal deletes the wrong tree."""
    mid = str(entry.get("model") or "").strip()
    return mid or slug[len(_LOCAL_NS):]


def plan_removal(root: Path, slug: str) -> dict:
    """What removing `slug` would touch. Pure: reads only, never writes."""
    if not slug.startswith(_LOCAL_NS):
        raise Refusal(
            f"{slug!r} is not a local slug. This tool only removes the "
            f"{_LOCAL_NS!r} layer; a curated catalog entry is git-tracked and "
            f"git is its removal tool."
        )
    raw = _load_local_raw(root)
    if slug not in raw:
        known = ", ".join(sorted(raw)) or "(none registered)"
        raise Refusal(f"{slug} is not in {_LOCAL_REGISTRY_REL}. Registered: {known}")

    entry = raw[slug] if isinstance(raw.get(slug), dict) else {}
    mid = _model_id_for(slug, entry)

    profile = root / _LOCAL_MODELS_REL / f"{mid}.yml"
    composes = root / _LOCAL_COMPOSES_REL / mid

    # Only ever remove a compose path that the registry entry actually points
    # into. A hand-edited compose_path outside the local composes tree (or
    # shared with another slug) is reported and LEFT ALONE rather than deleted.
    declared = str(entry.get("compose_path") or "")
    compose_warning = ""
    if declared:
        try:
            declared_abs = (root / declared).resolve()
            declared_abs.relative_to((root / _LOCAL_COMPOSES_REL).resolve())
        except Exception:
            compose_warning = (
                f"compose_path {declared!r} is outside {_LOCAL_COMPOSES_REL}/ — "
                f"left in place; remove it yourself if you no longer want it"
            )

    # Another slug sharing this model's tree means the tree is NOT ours to drop.
    others = [
        s for s, e in raw.items()
        if s != slug and isinstance(e, dict) and _model_id_for(s, e) == mid
    ]

    return {
        "slug": slug,
        "model_id": mid,
        "profile": profile,
        "composes": composes,
        "shared_with": others,
        "compose_warning": compose_warning,
        "registry": root / _LOCAL_REGISTRY_REL,
    }


def _describe(plan: dict) -> str:
    lines = [
        f"slug        {plan['slug']}",
        f"model id    {plan['model_id']}",
        f"registry    remove entry from {_LOCAL_REGISTRY_REL}",
    ]
    if plan["shared_with"]:
        lines.append(
            f"profile     KEPT — also used by {', '.join(plan['shared_with'])}"
        )
        lines.append("composes    KEPT — same reason")
    else:
        lines.append(
            f"profile     {'delete ' + str(plan['profile']) if plan['profile'].is_file() else '(already absent)'}"
        )
        lines.append(
            f"composes    {'delete ' + str(plan['composes']) + '/' if plan['composes'].is_dir() else '(already absent)'}"
        )
    if plan["compose_warning"]:
        lines.append(f"⚠  {plan['compose_warning']}")
    return "\n".join(f"[demote]   {ln}" for ln in lines)


def remove_local(root: Path, slug: str, *, dry_run: bool = False) -> dict:
    """Remove a local slug and (when unshared) its files. Returns the plan."""
    plan = plan_removal(root, slug)
    print(_describe(plan), file=sys.stderr)
    if dry_run:
        _info("--dry-run — nothing removed")
        return plan

    # Registry entry FIRST. If a later unlink fails, the catalog is already
    # consistent (no slug pointing at a half-deleted tree) — the opposite order
    # leaves exactly the dangling entry this tool exists to prevent.
    try:
        raw = _load_local_raw(root)
        raw.pop(slug, None)
        reg = plan["registry"]
        if raw:
            reg.write_text(
                json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        elif reg.is_file():
            # Last entry gone → remove the file rather than leave `{}` behind,
            # so `get_registry()` sees no local layer at all.
            reg.unlink()
        _info(f"unregistered {slug}")
    except OSError as exc:
        raise InternalError(f"could not rewrite {plan['registry']}: {exc}")

    if plan["shared_with"]:
        _info(
            f"kept {plan['model_id']} files — still used by "
            f"{', '.join(plan['shared_with'])}"
        )
        return plan

    try:
        if plan["profile"].is_file():
            plan["profile"].unlink()
            _info(f"removed {plan['profile']}")
        if plan["composes"].is_dir():
            shutil.rmtree(plan["composes"])
            _info(f"removed {plan['composes']}/")
    except OSError as exc:
        raise InternalError(
            f"unregistered {slug}, but could not remove its files: {exc}"
        )
    return plan


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Remove a model from the LOCAL layer (never the curated catalog)."
    )
    ap.add_argument("--slug", required=True, help="the local/<name> slug to remove")
    ap.add_argument("--root", default=str(_DEFAULT_ROOT), help="repo root")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, remove nothing")
    ap.add_argument("-y", "--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    try:
        if not args.yes and not args.dry_run and sys.stdin.isatty():
            plan = plan_removal(root, args.slug)
            print(_describe(plan), file=sys.stderr)
            try:
                reply = input(f"[demote] remove {args.slug}? [y/N] ").strip().lower()
            except EOFError:
                reply = ""
            if reply not in ("y", "yes"):
                print("[demote] aborted — nothing removed", file=sys.stderr)
                return EXIT_COLLISION
        remove_local(root, args.slug, dry_run=args.dry_run)
    except Refusal as exc:
        print(f"[demote] refused: {exc}", file=sys.stderr)
        return EXIT_COLLISION
    except InternalError as exc:
        print(f"[demote] FAILED: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
    if not args.dry_run:
        print(f"DEMOTE_OK {args.slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
