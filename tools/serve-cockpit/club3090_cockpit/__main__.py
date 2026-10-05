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
    """c3's own preferences (logging, catalog columns / sort, first-run flag) —
    ``<C3_CONFIG_DIR or ~/.config/club-3090>/c3-settings.json`` (parallel to the
    surface file so the proven surface logic is untouched).  MODEL_DIR and
    HF_TOKEN used to live here too; they are club-3090 settings now (#1466, see
    ``settings_store``), and a copy left here is moved over once."""
    base = os.environ.get("C3_CONFIG_DIR")
    cfg_dir = Path(base) if base else Path.home() / ".config" / "club-3090"
    return cfg_dir / "c3-settings.json"


def load_settings() -> dict:
    """c3's persisted preferences — ``{}`` on a missing / unreadable / malformed
    file (tolerant; never crashes the launch)."""
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


def first_run_seen() -> bool:
    """Whether the first-run guide was dismissed with "Skip — I know what I'm
    doing" (``first_run_seen`` in c3-settings.json).  Tolerant like
    :func:`load_settings` — anything but an explicit ``true`` reads as unseen."""
    return load_settings().get("first_run_seen") is True


def save_first_run_seen() -> None:
    """Persist the first-run seen-flag (MERGE — every other persisted setting is
    preserved).  Best-effort like :func:`save_settings`; idempotent."""
    s = load_settings()
    if s.get("first_run_seen") is True:
        return
    s["first_run_seen"] = True
    save_settings(s)


def apply_persisted_settings(app, environ) -> None:
    """Apply the saved settings + c3's preferences to the app before run().

    MODEL_DIR and HF_TOKEN are club-3090 settings (#1466), read through the one
    loader with the precedence every script uses: the shell > club3090.env >
    secrets.env > the repo .env > the bundled default.  ``weights_model_dir``
    resolves MODEL_DIR itself, so c3 looks for weights exactly where switch.sh
    and setup.sh do.  The saved HF token is applied to ``environ`` when the shell
    hasn't set one, so c3's own children (downloads, HF search) get it.

    First, a model dir / HF token that an older c3 saved in c3-settings.json
    moves into the store once (``settings_store.fold_in_c3_settings``); what
    happened is queued for the UI as a toast, never with a token in it."""
    from .settings_store import fold_in_c3_settings, migrate_notice, stored

    s = load_settings()
    data = app._data
    notices, moved, fallbacks = fold_in_c3_settings(s, data.repo_root)
    if moved:
        save_settings(s)
    queue = getattr(app, "_startup_notices", None)
    if isinstance(queue, list):
        queue.extend(notices)
        # Settings still in the checkout (repo .env, gateway files): the same one-time
        # notice switch.sh / gpu-mode print, shown once by whichever runs first (#1466).
        text = migrate_notice(data.repo_root)
        if text:
            queue.append(("information", text))
    # A value the store refused keeps working in c3 for now (the warning above
    # says why) — the pre-#1466 behaviour, shell still first.
    if fallbacks.get("MODEL_DIR") and not str(environ.get("MODEL_DIR") or "").strip():
        data._model_dir = fallbacks["MODEL_DIR"]
    # HF_TOKEN: a shell-provided token wins; otherwise apply the saved one.
    if not environ.get("HF_TOKEN"):
        hit = stored("HF_TOKEN", data.repo_root)
        tok = (hit[1] if hit else "").strip() or fallbacks.get("HF_TOKEN", "")
        if tok:
            environ["HF_TOKEN"] = tok
            data._hf_token_injected = True
    # Catalog columns (#724): the [|] picker's persisted order/visibility —
    # applied via an app attribute (CatalogPane reads it on mount) so a
    # directly-constructed app (tests) always starts canonical.
    cols = s.get("catalog_columns")
    if isinstance(cols, dict):
        app.catalog_columns_pref = cols
    # Catalog sort ([s] cycle): the persisted last sort mode — applied via an
    # app attribute the same way (CatalogPane sanitizes on mount; unknown /
    # malformed values degrade to the group-by-model default).
    srt = s.get("catalog_sort")
    if isinstance(srt, str) and srt:
        app.catalog_sort_pref = srt
    # Catalog [w] downloaded-only: the persisted last choice (ON by default when
    # unset — CatalogPane applies the default, so only a saved bool is passed on).
    dl = s.get("catalog_downloaded_only")
    if isinstance(dl, bool):
        app.catalog_downloaded_only_pref = dl
    # Master logging: strict C3_LOG=1|0 shell override wins for this launch.
    # Invalid/absent values fall back to the persisted boolean, default OFF.
    from .session_logging import env_log_override

    env_override = env_log_override(environ)
    persisted_log = s.get("logging_enabled")
    enabled = env_override if env_override is not None else persisted_log is True
    app._c3_log_env_override = env_override is not None
    app.configure_session_logging(enabled)


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
    # Apply the saved settings (MODEL_DIR / HF_TOKEN, via the loader) and c3's
    # own preferences before run.
    apply_persisted_settings(app, os.environ)
    app.run()


if __name__ == "__main__":
    main()
