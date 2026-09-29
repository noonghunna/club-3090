#!/usr/bin/env python3
"""slug_label — stamp a launched container with the registry slug it runs.

WHY
---
A measurement record, and c3's "running with these settings?" marker, need to know
which slug a running container is. The container name only says which COMPOSE it
came from, and two slugs can share one compose file (``vllm/dual`` and
``vllm/qwen-27b-dual-fast``; ``llamacpp/default`` and ``llamacpp/mtp``) — with
per-slug launch settings (#1465) they no longer even run the same way. So the
launchers add a tiny compose override at ``up`` time that labels every service
``club3090.slug=<slug>``; ``measurement_record.resolve_serving`` and c3 trust that
label first (#1477 follow-up). estate_cli does the same for pods in its own
per-instance override.

    python3 scripts/lib/slug_label.py override --slug SLUG --compose PATH [--root ROOT]

prints the override file's path, or nothing when no label should be stamped: an
unknown slug, or a slug whose registered compose is not PATH — a label must never
name a slug the container isn't running. It never fails the caller (exit 0).

The file lives in ``<data dir>/compose-labels/`` (club_config.data_dir: the same
path for every checkout), is only rewritten when its content changes, and stays
put: compose records its path in the container's config-files label. Written as
the invoking user, before any ``sudo docker compose`` (gpu-mode) reads it.
Standard library only: this is on the launcher path.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LABEL = "club3090.slug"
SUBDIR = "compose-labels"
_SERVICE_RE = re.compile(r"^  ([A-Za-z0-9._-]+):\s*(#.*)?$")


def service_names(compose_text: str) -> list[str]:
    """The service keys under the top-level ``services:`` block (2-space indent, as
    every club-3090 compose is written) — no YAML library on the launcher path."""
    out, inside = [], False
    for line in compose_text.splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            inside = line.rstrip().startswith("services:")
            continue
        if inside:
            m = _SERVICE_RE.match(line)
            if m:
                out.append(m.group(1))
    return out


def _registered_compose(slug: str, root: Path) -> Path | None:
    from scripts.lib.profiles.compose_registry import get_registry

    entry = get_registry(root).get(slug)
    if not entry or not entry.get("compose_path"):
        return None
    return (root / entry["compose_path"]).resolve()


def override_path(slug: str, environ=None) -> Path:
    from scripts.lib import club_config

    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-") or "slug"
    return club_config.data_dir(environ) / SUBDIR / f"{safe}.yml"


def write_override(slug: str, compose: Path, root: Path = REPO_ROOT, environ=None) -> Path | None:
    """The label override for ``slug`` launched from ``compose``, or None when no
    label should be stamped (see the module docstring)."""
    compose = Path(compose).resolve()
    if _registered_compose(slug, root) != compose:
        return None
    names = service_names(compose.read_text(encoding="utf-8", errors="replace"))
    if not names:
        return None
    body = [f"# Written by scripts/lib/slug_label.py: which registry slug this compose was",
            f"# launched as ({LABEL}); read by measurement records and c3.",
            "services:"]
    for name in names:
        body += [f"  {name}:", "    labels:", f'      {LABEL}: "{slug}"']
    text = "\n".join(body) + "\n"
    path = override_path(slug, environ)
    try:
        if path.read_text(encoding="utf-8") == text:
            return path
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="slug_label.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("override", help="write the label override; print its path (or nothing)")
    o.add_argument("--slug", required=True)
    o.add_argument("--compose", required=True, help="the compose file being launched")
    o.add_argument("--root", default=str(REPO_ROOT), help="repo root (its registry + local layer)")
    a = ap.parse_args(argv)
    try:
        path = write_override(a.slug, Path(a.compose), Path(a.root).resolve())
    except Exception as exc:  # noqa: BLE001 — a label must never stop a launch
        print(f"[slug-label] no label for {a.slug}: {exc}", file=sys.stderr)
        return 0
    if path:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
