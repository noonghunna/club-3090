#!/usr/bin/env python3
"""Repo-root ``.env`` loading for the python profile tools (#1142).

WHY THIS EXISTS. The shell launchers source ``<root>/.env`` (``set -a`` in
``launch.sh`` / ``setup.sh``), so users reasonably expect a value parked there to
reach the tooling. ``promote.py`` and ``export_pr.py`` are invoked DIRECTLY —
there is no shell wrapper anywhere in ``scripts/`` that runs them — so they read
only the real environment. A maintainer putting ``C3_ALLOW_CORE_PROMOTE=1`` in
``.env`` therefore got a **silent no-op**: no error, no effect, and a core
promote that kept refusing for no visible reason.

PRECEDENCE is the same as the shell path and as ``estate_cli.load_dotenv``: the
**real environment wins**; ``.env`` only fills keys that are unset. So exporting
a var still overrides the file, and nothing already set is ever clobbered.

Parsing is the union of the two hand-rolled readers already in the tree —
``estate_cli.load_dotenv`` (whole file, but no ``export`` prefix, no CR) and
``deriver._model_dir_from_env_or_dotenv`` (``export`` prefix + CR + the #599
non-UTF-8-locale ``errors="replace"`` read, but one key only). Those two callers
are deliberately left alone here; they can adopt this later.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_dotenv(root) -> dict[str, str]:
    """``<root>/.env`` → ``{KEY: VALUE}``. ``{}`` when absent or unreadable.

    Tolerates ``export KEY=v``, CRLF, ``#`` comments, blank lines and quoted
    values. Never raises: a malformed ``.env`` must not take down a promote."""
    out: dict[str, str] = {}
    try:
        text = (Path(root) / ".env").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip().rstrip("\r")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        out[key] = value.strip().strip('"').strip("'")
    return out


def apply_dotenv(root) -> list[str]:
    """Fill UNSET ``os.environ`` keys from ``<root>/.env``.

    Returns the keys actually injected (sorted) so a caller can SAY where a
    value came from — the point of the fix is that ``.env`` stops being silent,
    in both directions."""
    injected = []
    for key, value in parse_dotenv(root).items():
        if key not in os.environ:
            os.environ[key] = value
            injected.append(key)
    return sorted(injected)
