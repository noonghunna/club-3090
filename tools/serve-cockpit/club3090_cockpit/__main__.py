"""Entry point for the c3 command."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def config_path() -> Path:
    """The file the in-app [C] lean toggle persists the chosen surface to.

    Honors the C3_CONFIG_DIR env override (tests point it at a tmp_path so they
    never touch the real ``~/.config``); otherwise falls back to
    ``~/.config/club-3090/``.  Returns ``<dir>/c3-surface.json``.
    """
    base = os.environ.get("C3_CONFIG_DIR")
    cfg_dir = Path(base) if base else Path.home() / ".config" / "club-3090"
    return cfg_dir / "c3-surface.json"


def settings_path() -> Path:
    """The file the in-app Settings ([S]) persists MODEL_DIR / HF_TOKEN to —
    ``<C3_CONFIG_DIR or ~/.config/club-3090>/c3-settings.json`` (parallel to the
    surface file so the proven surface logic is untouched)."""
    base = os.environ.get("C3_CONFIG_DIR")
    cfg_dir = Path(base) if base else Path.home() / ".config" / "club-3090"
    return cfg_dir / "c3-settings.json"


def load_settings() -> dict:
    """Persisted user settings (``model_dir`` / ``hf_token``) — ``{}`` on a
    missing / unreadable / malformed file (tolerant; never crashes the launch)."""
    try:
        with settings_path().open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(settings: dict) -> None:
    """Persist user settings (the WHOLE dict — callers merge first).  Best-effort;
    a write failure is swallowed (settings still apply for the session)."""
    if not isinstance(settings, dict):
        return
    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(settings, fh)
    except OSError:
        pass


def apply_persisted_settings(app, environ) -> None:
    """Apply the persisted MODEL_DIR / HF_TOKEN to the app + process env before
    run().  MODEL_DIR points the weights-on-disk check at the user's models
    volume; HF_TOKEN flows into the download subprocess env.  An explicit env var
    already set WINS (don't overwrite a shell-provided token)."""
    s = load_settings()
    mdir = str(s.get("model_dir") or "").strip()
    if mdir:
        app._data._model_dir = mdir
    tok = str(s.get("hf_token") or "").strip()
    if tok and not environ.get("HF_TOKEN"):
        environ["HF_TOKEN"] = tok


def load_surface_setting() -> "str | None":
    """Read the persisted surface setting, or None if absent / corrupt / invalid.

    Tolerant by design — a missing file, unreadable file, malformed JSON, or an
    unrecognised surface value all return None (the caller degrades to the
    default FULL surface) rather than crashing the launch.
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
    """Persist the chosen surface for next launch.

    Values: ``"consumer"`` = LEAN, ``"producer"`` = FULL (see resolve_surface for
    the inversion).  Best-effort — creates the config dir if needed; a write
    failure is swallowed (the toggle still takes effect for the current session, it
    just won't be remembered).  Invalid surfaces are ignored.
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
    """Resolve the audience surface from CLI args + env + persisted setting.

    2-mode merge — surface INVERSION.  The app serves BOTH consumers and
    producers, and every producer is a consumer, so BOTH modes show by DEFAULT.
    Internal values are kept ("producer"/"consumer") to reuse the gate machinery,
    but the meaning is inverted:
      * ``"producer"`` = FULL  (default — Run & Operate + Bring & Validate)
      * ``"consumer"`` = LEAN  (the minimal rig view — hides Bring & Validate)

    Precedence (highest first):
      1. An explicit per-launch flag/env:
         * ``c3 --lean`` (bare flag) or ``C3_SURFACE=consumer`` (env) → LEAN.
         * ``c3 --contribute`` (kept as a harmless alias — already the default) or
           ``C3_SURFACE=producer`` (env) → FULL.
         An explicit lean opt-out wins over a redundant contribute alias.
      2. The PERSISTED setting — the surface the in-app [C] lean toggle last saved
         via ``save_surface_setting`` (``c3-surface.json``).
      3. ``producer`` (FULL — the default, both modes visible).

    C3_SURFACE is normalised case- and whitespace-insensitively so
    ``Consumer``/`` consumer `` also work.  An explicit flag/env on a given launch
    wins over the persisted value.
    """
    # Explicit per-launch flags/env first (lean opt-out beats the redundant alias).
    env_surface = environ.get("C3_SURFACE", "").strip().lower()
    if "--lean" in argv or env_surface == "consumer":
        return "consumer"
    if "--contribute" in argv or env_surface == "producer":
        return "producer"
    persisted = load_surface_setting()
    if persisted in ("consumer", "producer"):
        return persisted
    return "producer"


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

    # Surface (2-mode merge — inverted default): FULL ("producer", default — both
    # Run & Operate AND Bring & Validate) vs LEAN ("consumer" — the minimal rig
    # view, opt-in via `c3 --lean` / C3_SURFACE=consumer / the in-app [C] toggle).
    # resolve_surface() precedence: explicit per-launch flag/env FIRST, else the
    # surface the in-app [C] lean toggle last PERSISTED (c3-surface.json), else the
    # full default.
    surface = resolve_surface(sys.argv, os.environ)
    app = CockpitApp(repo_root=repo_root, surface=surface)
    # Apply persisted MODEL_DIR / HF_TOKEN (the in-app [S] Settings) before run.
    apply_persisted_settings(app, os.environ)
    app.run()


if __name__ == "__main__":
    main()
