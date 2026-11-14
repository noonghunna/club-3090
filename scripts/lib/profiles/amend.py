#!/usr/bin/env python3
"""Local-layer AMENDMENT — rename a slug, or edit an entry's fields in place.

The local layer was create-and-delete only: ``promote.py`` refuses every "already
exists" case and ``demote.py`` only removes, so changing one field meant demoting
and re-promoting.

That is not merely inconvenient. #1202 decided a local slug colliding with a
newly-shipped curated one is **shadowed** — core wins the lookup, and the local
row is kept and MARKED "so the user can rename it". No rename existed, so the
remedy we advertised was unreachable. ``rename`` is that remedy; ``update`` is
the convenience half.

    RENAME_OK <old> -> <new>
    UPDATE_OK <slug>

⛔ CORE IS NEVER TOUCHED. Same guard as demote.py, and grounded the same way: a
slug is resolved in ``registry.local.json`` FIRST, and anything absent is refused
before a byte is read or written. That is stronger than a name test — a curated
slug can never appear in a file this layer is the sole writer of — and it fails
closed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:  # package context
    from scripts.lib.profiles import compose_header
    from scripts.lib.profiles.compose_facts import proposed_workload
except ImportError:  # direct-script context: this directory is on sys.path
    import compose_header
    from compose_facts import proposed_workload

_DEFAULT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_REGISTRY_REL = "scripts/lib/profiles-local/registry.local.json"
_LOCAL_COMPOSES_REL = "scripts/lib/profiles-local/composes"
_LOCAL_ENGINES_REL = "scripts/lib/profiles-local/engines.d"

EXIT_REFUSED = 3

# Fields a user may edit. `origin` is deliberately ABSENT: a local row that could
# set itself to "core" would hide from the very listings meant to mark it — the
# same anti-spoof reasoning as registration. `model` is absent too: it names the
# files on disk, so changing it is a move, not an edit.
# `offload` and `spec_method` are editable because a compose often CANNOT prove
# them: when the image's own entrypoint builds the command line, the backend name
# and the speculative config live inside the image, not the compose. The
# derivation says so rather than guessing (a wrong `offload` value has caused a
# real misdiagnosis before), which leaves stating it as the user's job.
# `kvcalc_key` names the kv-calc profile a vLLM-type slug projects VRAM with. A
# registration can't know it (it is per quant), and export_pr.py refuses a vLLM
# entry without one, so it has to be settable here.
# `weights_variant` picks another of the model's variants (it must exist, curated
# or local); `chat_template` is what same-model-defaults compares across a model's
# engines; `status_note` is the one-line why beside the status.
_EDITABLE = ("workload", "max_ctx", "max_num_seqs", "mem_util", "default_port",
             "kv_format", "tp", "drafter", "status", "offload", "spec_method",
             "kvcalc_key", "weights_variant", "chat_template", "status_note")
# Kept as strings even when they look numeric (a variant named `8`, a note `12`).
_STRING_KEYS = {"workload", "kv_format", "drafter", "status", "offload", "spec_method",
                "kvcalc_key", "weights_variant", "chat_template", "status_note"}


class Refusal(Exception):
    """Refused before anything was read or written."""


def _load(root: Path) -> dict[str, Any]:
    path = root / _LOCAL_REGISTRY_REL
    if not path.is_file():
        raise Refusal(
            f"no local layer at {_LOCAL_REGISTRY_REL} — nothing registered on this rig"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal(f"{path} is unreadable — fix or remove it first: {exc}") from exc
    if not isinstance(raw, dict):
        raise Refusal(f"{path} is not a JSON object")
    return raw


def _require_local(root: Path, slug: str) -> dict[str, Any]:
    """Resolve `slug` in the LOCAL layer, or refuse. The core-safety guard."""
    raw = _load(root)
    if slug in raw:
        return raw
    try:
        sys.path.insert(0, str(root))
        from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY

        is_core = slug in COMPOSE_REGISTRY
    except Exception:
        is_core = False
    if is_core:
        raise Refusal(
            f"{slug!r} is a CURATED catalog entry, not a local one. This tool only "
            f"amends the local layer; a curated entry is git-tracked and a PR is how "
            f"it changes."
        )
    known = ", ".join(sorted(raw)) or "(none registered)"
    raise Refusal(f"{slug} is not in {_LOCAL_REGISTRY_REL}. Registered: {known}")


def _check_new_slug(root: Path, raw: dict[str, Any], new: str) -> None:
    if new.count("/") != 1 or new.startswith("/") or new.endswith("/"):
        raise Refusal(f"{new!r} must be '<engine>/<name>'")
    if new in raw:
        raise Refusal(f"{new!r} is already registered locally")
    # A rename must re-run the SAME collision test registration does, or it is a
    # way to smuggle a shadowed slug back in through the side door.
    try:
        sys.path.insert(0, str(root))
        from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY

        if new in COMPOSE_REGISTRY:
            raise Refusal(
                f"{new!r} is a curated slug — renaming onto it would be shadowed "
                f"immediately (core wins the lookup). Pick another name."
            )
    except Refusal:
        raise
    except Exception:
        pass  # registry unreadable: the local-side checks above still applied


def _engine_exists(root: Path, engine: str) -> bool:
    """Is `engine` a known engine — curated, or in the local engines.d layer?"""
    if (root / _LOCAL_ENGINES_REL / f"{engine}.yml").is_file():
        return True
    try:
        sys.path.insert(0, str(root))
        from scripts.lib.profiles.compat import load_profiles

        return engine in load_profiles().engines
    except Exception:
        return False


def plan_rename(root: Path, old: str, new: str) -> dict[str, Any]:
    raw = _require_local(root, old)
    _check_new_slug(root, raw, new)
    entry = raw[old] if isinstance(raw.get(old), dict) else {}
    old_engine, new_engine = old.split("/", 1)[0], new.split("/", 1)[0]
    mid = str(entry.get("model") or "")
    moves = []
    if old_engine != new_engine:
        # The slug's namespace IS the engine (#1202) — so changing it changes
        # which engine the entry claims, and the `engine` KWARG must follow.
        # Leaving them to disagree looked harmless (nothing validates this path)
        # and would have quietly broken the one convention the design rests on.
        if not _engine_exists(root, new_engine):
            raise Refusal(
                f"no engine profile for {new_engine!r}. The slug's namespace is the "
                f"engine, so renaming into it would claim an engine that does not "
                f"exist. Register a model on it first (catalog.sh register --engine "
                f"{new_engine} --engine-type …), or rename within an existing engine."
            )
    if old_engine != new_engine and mid:
        # the engine segment is part of the compose path, so it has to follow
        src = root / _LOCAL_COMPOSES_REL / mid / old_engine
        dst = root / _LOCAL_COMPOSES_REL / mid / new_engine
        if src.is_dir():
            moves.append((src, dst))
    return {"old": old, "new": new, "entry": entry, "moves": moves,
            "old_engine": old_engine, "new_engine": new_engine, "raw": raw}


def do_rename(root: Path, old: str, new: str, *, dry_run: bool = False) -> dict[str, Any]:
    plan = plan_rename(root, old, new)
    if dry_run:
        return plan
    raw, entry = plan["raw"], dict(plan["entry"])
    for src, dst in plan["moves"]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        print(f"[amend] moved {src.relative_to(root)} -> {dst.relative_to(root)}")
    if plan["old_engine"] != plan["new_engine"]:
        entry["engine"] = plan["new_engine"]
        print(f"[amend] engine: {plan['old_engine']} -> {plan['new_engine']}")
    if plan["moves"]:
        cp = str(entry.get("compose_path") or "")
        marker = f"/{plan['old_engine']}/"
        if marker in cp:
            entry["compose_path"] = cp.replace(marker, f"/{plan['new_engine']}/", 1)
    raw[new] = entry
    del raw[old]
    (root / _LOCAL_REGISTRY_REL).write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[amend] registry.local.json: {old} -> {new}")
    return plan


def _model_variants(root: Path, model: str) -> Optional[set[str]]:
    """The weights variants a model has here: a local model's own, or a curated
    model's plus what its extends.d file adds. None when the model isn't found."""
    import yaml
    local = root / "scripts/lib/profiles-local/models.d" / f"{model}.yml"
    core = root / "scripts/lib/profiles/models" / f"{model}.yml"
    ext = root / "scripts/lib/profiles-local/extends.d" / f"{model}.yml"
    found = None
    for path in (local, core, ext):
        if path.is_file():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            found = (found or set()) | set((data.get("weights") or {}))
            if path == local:
                break
    return found


def _validate_entry(root: Path, slug: str, entry: dict[str, Any], edits: dict[str, Any]) -> list[str]:
    """Refuse an edit the loader would reject, BEFORE the write: a bad entry in
    registry.local.json fails get_registry() for every launcher. Returns notes."""
    notes: list[str] = []
    sys.path.insert(0, str(root))
    try:
        from scripts.lib.profiles.compose_registry import _entry, COMPOSE_REGISTRY
        _entry(**{k: v for k, v in entry.items() if k != "origin"})
    except (TypeError, ValueError) as exc:
        raise Refusal(f"the entry would not load: {exc}")
    if "weights_variant" in edits:
        have = _model_variants(root, str(entry.get("model") or ""))
        if have is not None and entry["weights_variant"] not in have:
            raise Refusal(f"{entry.get('model')} has no weights variant "
                          f"{entry['weights_variant']!r}. Its variants: {', '.join(sorted(have))}. "
                          f"A new one is added with catalog.sh register --weights-variant, or "
                          f"edited with catalog.sh update-weights")
        notes.append("the compose keeps its path; the directory named after the old variant "
                     "is only a convention")
    if "chat_template" in edits and edits["chat_template"]:
        known = {str(e.get("chat_template")) for e in COMPOSE_REGISTRY.values()
                 if e.get("chat_template")}
        if edits["chat_template"] not in known:
            notes.append(f"chat_template {edits['chat_template']!r} is not used by any curated "
                         f"slug (they use: {', '.join(sorted(known))})")
    return notes


def _reload_error(root: Path) -> Optional[str]:
    import subprocess
    chk = subprocess.run(
        [sys.executable, "-c",
         "from scripts.lib.profiles.compose_registry import get_registry; get_registry(); "
         "from scripts.lib.profiles.compat import load_profiles; load_profiles()"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8")
    return None if chk.returncode == 0 else (chk.stderr.strip() or chk.stdout.strip())[-800:]


def _header_for_status(root: Path, entry: dict[str, Any], status: str) -> Optional[tuple[Path, str]]:
    """(compose path, new text) when the entry's compose header must change to say
    `status`, else None. A compose with no profile header is left alone. A status
    that needs a Caveats line gets one when the header has none: the status_note, or
    a placeholder asking for the reason."""
    path = root / str(entry.get("compose_path") or "")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if compose_header.status_value(text) is None or compose_header.says(text, status):
        return None
    caveats = None
    if status in compose_header.NEEDS_CAVEATS and not compose_header.has_caveats(text):
        caveats = entry.get("status_note") or (
            f"set to {status} with catalog.sh update; say why here before exporting")
    try:
        return path, compose_header.set_status(text, compose_header.LABELS[status], caveats)
    except compose_header.HeaderError:
        return None


def do_update(root: Path, slug: str, edits: dict[str, Any], *,
              dry_run: bool = False) -> dict[str, Any]:
    raw = _require_local(root, slug)
    entry = dict(raw[slug] if isinstance(raw.get(slug), dict) else {})
    if not edits:
        raise Refusal(f"nothing to change — pass at least one of: {', '.join(_EDITABLE)}")
    bad = [k for k in edits if k not in _EDITABLE]
    if bad:
        raise Refusal(
            f"not editable: {', '.join(sorted(bad))}. Editable: {', '.join(_EDITABLE)}. "
            f"('origin' is stamped by the loader; 'model' names the files on disk, so "
            f"changing it is a move, not an edit.)"
        )
    edits = dict(edits)
    # Registration proposes `workload` from max_ctx (compose_facts.proposed_workload),
    # so a max_ctx correction re-applies that rule — when the stored workload is the
    # one the rule gave for the OLD max_ctx. One the user chose (any other value),
    # or one set in the same call, is left alone (#2028).
    if "max_ctx" in edits and "workload" not in edits:
        old_ctx, new_ctx = entry.get("max_ctx"), edits["max_ctx"]
        if (isinstance(old_ctx, int) and isinstance(new_ctx, int)
                and not isinstance(old_ctx, bool) and not isinstance(new_ctx, bool)):
            was, now = proposed_workload(old_ctx), proposed_workload(new_ctx)
            if entry.get("workload") == was and now != was:
                edits["workload"] = now
                print(f"[amend] note: workload follows max_ctx, as registration proposes "
                      f"it ({was} -> {now}); to keep another, pass --set workload=<name> "
                      f"with max_ctx")
    before = {k: entry.get(k) for k in edits}
    entry.update(edits)
    for k in [k for k, v in edits.items() if v is None and k in ("status_note", "chat_template")]:
        entry.pop(k, None)                    # `KEY=` / none removes an optional field
    notes = _validate_entry(root, slug, entry, edits)
    for n in notes:
        print(f"[amend] note: {n}")
    # The compose header carries the status too (test-compose-status-drift compares the
    # two once the slug is exported), so a status change rewrites its Status line.
    header = _header_for_status(root, entry, edits["status"]) if edits.get("status") else None
    if dry_run:
        if header:
            print(f"[amend] would set {header[0].name}'s header Status to "
                  f"{compose_header.LABELS[edits['status']]}")
        return {"slug": slug, "before": before, "after": edits}
    reg = root / _LOCAL_REGISTRY_REL
    old_text = reg.read_text(encoding="utf-8")
    raw[slug] = entry
    reg.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    err = _reload_error(root)
    if err:
        reg.write_text(old_text, encoding="utf-8")
        raise Refusal(f"the catalog no longer loads with this change, so it was put back:\n{err}")
    if header:
        path, text = header
        tmp = path.with_name(path.name + ".amend.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            reg.write_text(old_text, encoding="utf-8")
            raise Refusal(f"could not rewrite {path.name}'s header ({exc}); the entry was put back")
        print(f"[amend] {path.name}: header Status -> {compose_header.LABELS[edits['status']]}")
    for k in sorted(edits):
        print(f"[amend] {slug}: {k}: {before[k]!r} -> {edits[k]!r}")
    return {"slug": slug, "before": before, "after": edits}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rename or edit a LOCAL catalog entry (never the curated one)."
    )
    ap.add_argument("--slug", required=True, help="the local <engine>/<name> slug")
    ap.add_argument("--to", help="rename: the new <engine>/<name> slug")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help=f"update a field ({', '.join(_EDITABLE)}); repeatable")
    ap.add_argument("--root", default=str(_DEFAULT_ROOT), help="repo root")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    ap.add_argument("-y", "--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args(argv)

    if bool(args.to) == bool(args.set):
        print("[amend] pass exactly one of --to (rename) or --set (update)", file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    try:
        if args.to:
            plan = do_rename(root, args.slug, args.to, dry_run=args.dry_run)
            if args.dry_run:
                print(f"[amend] dry-run: {plan['old']} -> {plan['new']}")
                for src, dst in plan["moves"]:
                    print(f"[amend]   would move {src.relative_to(root)}")
                print("[amend] nothing written.")
                return 0
            print(f"RENAME_OK {args.slug} -> {args.to}")
            return 0

        edits: dict[str, Any] = {}
        for kv in args.set:
            if "=" not in kv:
                print(f"[amend] --set expects KEY=VALUE, got {kv!r}", file=sys.stderr)
                return 2
            k, v = kv.split("=", 1)
            if v.lower() in ("none", "null") or (v == "" and k in _STRING_KEYS):
                edits[k] = None
            elif k in _STRING_KEYS:
                edits[k] = v
            elif v.isdigit():
                edits[k] = int(v)
            elif v.replace(".", "", 1).isdigit():
                edits[k] = float(v)
            else:
                edits[k] = v
        res = do_update(root, args.slug, edits, dry_run=args.dry_run)
        if args.dry_run:
            for k in sorted(res["after"]):
                print(f"[amend] dry-run {res['slug']}: {k}: "
                      f"{res['before'][k]!r} -> {res['after'][k]!r}")
            print("[amend] nothing written.")
            return 0
        print(f"UPDATE_OK {args.slug}")
        return 0
    except Refusal as exc:
        print(f"[amend] refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
