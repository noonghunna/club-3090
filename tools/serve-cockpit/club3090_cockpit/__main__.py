"""Entry point for the c3 command."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def config_path() -> Path:
    """The file the in-app Contribute door persists the chosen surface to.

    Honors the C3_CONFIG_DIR env override (tests point it at a tmp_path so they
    never touch the real ``~/.config``); otherwise falls back to
    ``~/.config/club-3090/``.  Returns ``<dir>/c3-surface.json``.
    """
    base = os.environ.get("C3_CONFIG_DIR")
    cfg_dir = Path(base) if base else Path.home() / ".config" / "club-3090"
    return cfg_dir / "c3-surface.json"


def load_surface_setting() -> "str | None":
    """Read the persisted surface setting, or None if absent / corrupt / invalid.

    Tolerant by design — a missing file, unreadable file, malformed JSON, or an
    unrecognised surface value all return None (the caller degrades to the
    default consumer surface) rather than crashing the launch.
    """
    path = config_path()
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    surface = data.get("surface") if isinstance(data, dict) else None
    if surface in ("consumer", "producer"):
        return surface
    return None


def save_surface_setting(surface: str) -> None:
    """Persist the chosen surface ("consumer"/"producer") for next launch.

    Best-effort — creates the config dir if needed; a write failure is swallowed
    (the toggle still takes effect for the current session, it just won't be
    remembered).  Invalid surfaces are ignored.
    """
    if surface not in ("consumer", "producer"):
        return
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump({"surface": surface}, fh)
    except OSError:
        pass


def resolve_surface(argv: list[str], environ: "os._Environ[str] | dict[str, str]") -> str:
    """Resolve the audience surface (R0/R4) from CLI args + env + persisted setting.

    Precedence (highest first):
      1. ``c3 --contribute`` (bare flag) or ``C3_SURFACE=producer`` (env) — an
         explicit per-launch opt-in into the producer surface.
      2. The PERSISTED setting (R4) — the surface the in-app Contribute door last
         saved via ``save_surface_setting`` (``c3-surface.json``).
      3. ``consumer`` (default — the clean Run + Operate UI).

    C3_SURFACE is normalised case- and whitespace-insensitively so
    ``Producer``/`` producer `` also work.  As of R4 the in-app Contribute door
    persists the chosen surface, and this resolver reads it (precedence 2) — an
    explicit flag/env on a given launch still wins over the persisted value.
    """
    if "--contribute" in argv:
        return "producer"
    if environ.get("C3_SURFACE", "").strip().lower() == "producer":
        return "producer"
    persisted = load_surface_setting()
    if persisted in ("consumer", "producer"):
        return persisted
    return "consumer"


def main() -> None:
    """Launch the serve cockpit TUI."""
    # Resolve repo root from this file's location:
    #   <repo>/tools/serve-cockpit/club3090_cockpit/__main__.py
    #   parents[0] = club3090_cockpit/
    #   parents[1] = serve-cockpit/
    #   parents[2] = tools/
    #   parents[3] = <repo root>
    # Override with C3_REPO_ROOT if installed outside the tree.
    env_root = os.environ.get("C3_REPO_ROOT")
    repo_root = Path(env_root) if env_root else Path(__file__).resolve().parents[3]

    if not (repo_root / "scripts").is_dir():
        print(
            f"Error: club-3090 repo root not found at {repo_root} "
            f"(no scripts/ dir).  Run via the repo tree, or set C3_REPO_ROOT.",
            file=sys.stderr,
        )
        sys.exit(1)

    from .app import CockpitApp

    # Surface (R0/R4): consumer (default — Run + Operate) vs producer (+ Bring &
    # Validate).  resolve_surface() precedence: explicit `c3 --contribute` /
    # C3_SURFACE=producer FIRST, else the surface the in-app Contribute door last
    # PERSISTED (R4 — c3-surface.json), else the clean consumer default.
    surface = resolve_surface(sys.argv, os.environ)
    app = CockpitApp(repo_root=repo_root, surface=surface)
    app.run()


if __name__ == "__main__":
    main()
