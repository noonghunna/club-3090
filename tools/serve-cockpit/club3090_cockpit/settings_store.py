"""c3's door to the club-3090 settings (club-3090#1466).

WHY THIS EXISTS
---------------
c3 used to read the repo-root ``.env`` with four private parsers, rewrite it
in place from two code paths, and keep its OWN copy of the model dir and HF
token in ``c3-settings.json``. ``switch.sh`` reads the settings, not c3's JSON,
so c3 could show weights as downloaded that ``switch.sh`` couldn't find.

Every setting c3 reads or writes now goes through the ONE loader,
``scripts/lib/club_config.py``, with the precedence every script uses:

    the shell > club3090.env > secrets.env > the repo .env (legacy, read last)

    read      get() / source() / stored()
    write     save()                       club3090.env, or secrets.env (0600)
    compose   compose_env_file()           0600 temp file for ``docker compose --env-file``
              route_keys_file()            0600 temp file of the gateway routes' keys
    fold-in   fold_in_c3_settings()        one-time move of c3-settings.json's
                                           model_dir / hf_token into the store
    notice    migrate_notice()             the launchers' one-time "settings still in
                                           this checkout" notice, as a toast

Nothing here parses or writes a settings file itself — it adapts the loader to
c3 (an error type whose message never carries the value, and the fold-in).
``scripts/tests/test-config-single-parser.sh`` fails any c3 file that goes back
to reading ``.env`` on its own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

_CC: Any = None


def _candidate_roots() -> list[Path]:
    """The checkout this c3 ships in, then ``C3_REPO_ROOT`` (an install outside
    the tree — the same override ``__main__`` honours)."""
    roots = [Path(__file__).resolve().parents[3]]
    env_root = os.environ.get("C3_REPO_ROOT")
    if env_root:
        roots.append(Path(env_root))
    return roots


def loader():
    """The ``scripts.lib.club_config`` module. The cockpit venv doesn't carry
    the repo root on ``sys.path`` (services.py and data.py insert it for the
    same reason), so put the first root that has the loader there first."""
    global _CC
    if _CC is None:
        for root in _candidate_roots():
            if (root / "scripts" / "lib" / "club_config.py").is_file():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                break
        from scripts.lib import club_config

        _CC = club_config
    return _CC


class SettingsError(Exception):
    """A write the store refused (a value bash, docker compose and systemd would
    read differently) or couldn't make (permissions). The message names the key
    and the reason — never the value, which may be a token."""


# ── reads ────────────────────────────────────────────────────────────────────


def get(key: str, repo_root, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The effective value of one setting — exactly what ``switch.sh`` launched
    from this environment resolves — or None when nothing sets it."""
    return loader().get(key, repo_root, environ)


def source(key: str, repo_root, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Where the effective value comes from: ``shell``, ``club3090.env``,
    ``secrets.env`` or ``repo .env``; None when nothing sets it."""
    cc = loader()
    env = os.environ if environ is None else environ
    if key in env:
        return cc.SHELL_LABEL
    hit = cc.resolve(repo_root, env).get(key)
    return hit[0] if hit else None


def stored(key: str, repo_root, environ: Optional[Mapping[str, str]] = None) -> Optional[tuple[str, str]]:
    """(file label, value) of the SAVED setting, ignoring the shell — what a
    launch gets when the shell doesn't set the key. None when no file sets it."""
    env = os.environ if environ is None else environ
    # Only what the loader needs to find the config dir; everything else in the
    # environment is left out so the shell can't mask the files.
    locate = {k: env[k] for k in ("CLUB3090_CONFIG_DIR", "XDG_CONFIG_HOME", "HOME")
              if k in env and k != key}
    return loader().resolve(repo_root, locate).get(key)


def store_path(*, secret: bool = False, environ: Optional[Mapping[str, str]] = None) -> Path:
    """club3090.env, or secrets.env for a secret — where save() writes."""
    return loader().target_path("secrets" if secret else "global", environ)


# ── writes ───────────────────────────────────────────────────────────────────


def save(values: Mapping[str, str], *, secret: bool = False,
         environ: Optional[Mapping[str, str]] = None) -> Path:
    """Store settings through the ONE writer (atomic, comments and order kept;
    secrets.env is created 0600). Returns the file written. Raises SettingsError."""
    cc = loader()
    which = "secrets" if secret else "global"
    try:
        return cc.set_values(dict(values), which, environ)
    except cc.ConfigError as e:
        raise SettingsError(str(e)) from None
    except OSError as e:
        raise SettingsError(
            f"couldn't write {cc.target_path(which, environ)}: {e.strerror or type(e).__name__}"
        ) from None


# ── docker compose ───────────────────────────────────────────────────────────


def compose_env_file(repo_root, environ: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """A 0600 temp file of every resolved setting, for ``docker compose
    --env-file``; the caller removes it. None when nothing is configured (the
    empty file is removed) — the same rule as gpu-mode.sh, so compose then reads
    the compose dir's own ``.env`` as it would without c3."""
    path = loader().write_compose_env_file(repo_root, environ=environ)
    try:
        if path.stat().st_size > 0:
            return path
    except OSError:
        pass
    try:
        path.unlink()
    except OSError:
        pass
    return None


def route_keys_file(repo_root, environ: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """A 0600 temp file of just the keys this rig's own gateway routes use, for the
    gateway compose's ``CLUB3090_LITELLM_ROUTE_KEYS`` — the same file gpu-mode.sh
    hands it (``scripts/lib/litellm_local.py``). The caller removes it. None when
    no route needs a saved key."""
    loader()                                  # puts the repo root on sys.path
    from scripts.lib import litellm_local

    return litellm_local.write_route_keys_file(repo_root, environ=environ)


def migrate_notice(repo_root, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The one-time "your settings still live in this checkout" notice that
    switch.sh / launch.sh / gpu-mode print (``club_config.migrate_notice``), as
    one toast; None when there is nothing to say or it was already shown (by any
    of them — they share one stamp). Never raises: a notice must not stop c3."""
    try:
        lines = loader().migrate_notice(repo_root, environ)
    except Exception:  # noqa: BLE001
        return None
    return _esc(" ".join(lines)) if lines else None


# ── the one-time fold-in of c3-settings.json (club-3090#1466) ───────────────
#
# c3's Settings used to save the model dir and HF token to c3-settings.json,
# which only c3 read. They now live in the store. On launch, a value c3 saved
# there moves into the store ONCE:
#   * the store (incl. the repo .env) has no value → copy it there;
#   * the store has the same value                → nothing to say;
#   * the store has a different value             → the store wins (it is what
#     switch.sh uses) and c3 says so.
# The key is then recorded in c3-settings.json under MOVED_KEY, so the notice
# isn't repeated and a setting the user later removes from the store isn't
# brought back from the old copy. c3-settings.json's own entries are never
# removed or rewritten. A copy the writer refuses is NOT recorded: c3 keeps using
# the old value for that launch (as before) and says why, every launch, until
# the value is fixed or set in [S] Settings.

MOVED_KEY = "moved_to_club3090_config"

# (c3-settings.json key, setting, secret?, label for the notice)
_FOLD_IN = (
    ("model_dir", "MODEL_DIR", False, "model dir"),
    ("hf_token", "HF_TOKEN", True, "HF token"),
)


def _esc(text: str) -> str:
    """Escape a dynamic value (a path) for a Textual notification."""
    return str(text).replace("[", "\\[")


def fold_in_c3_settings(settings: dict, repo_root,
                        environ: Optional[Mapping[str, str]] = None
                        ) -> tuple[list[tuple[str, str]], bool, dict[str, str]]:
    """Move c3-settings.json's ``model_dir`` / ``hf_token`` into the store once.

    Returns ``(notices, changed, fallbacks)``: notices are ``(severity, text)``
    for the UI (never a token value); ``changed`` means ``settings[MOVED_KEY]``
    was updated and the caller should save ``settings``; ``fallbacks`` maps a
    setting whose copy failed to the old value c3 keeps using for now."""
    raw = settings.get(MOVED_KEY)
    moved = [k for k in raw if isinstance(k, str)] if isinstance(raw, list) else []
    notices: list[tuple[str, str]] = []
    fallbacks: dict[str, str] = {}
    changed = False
    for jkey, key, secret, label in _FOLD_IN:
        if jkey in moved:
            continue
        old = str(settings.get(jkey) or "").strip()
        if not old:
            continue
        hit = stored(key, repo_root, environ)
        if hit is None:
            try:
                path = save({key: old}, secret=secret, environ=environ)
            except SettingsError as e:
                fallbacks[key] = old
                notices.append((
                    "warning",
                    f"c3 couldn't move your {label} from c3-settings.json into the club-3090 "
                    f"settings ({_esc(str(e))}). c3 still uses it, but switch.sh and setup.sh "
                    f"don't — set it again in \\[S] Settings.",
                ))
                continue
            notices.append((
                "information",
                f"Moved your {label} from c3-settings.json to {_esc(str(path))}, "
                f"where switch.sh and setup.sh read it too.",
            ))
        elif hit[1] != old:
            if secret:
                detail = f"the one in {hit[0]}"
            else:
                detail = f"{_esc(hit[1])} (from {hit[0]}) instead of {_esc(old)}"
            notices.append((
                "warning",
                f"Your {label} in c3-settings.json differs from your club-3090 settings. "
                f"c3 now uses {detail}, as switch.sh does. Change it in \\[S] Settings.",
            ))
        moved.append(jkey)
        changed = True
    if changed:
        settings[MOVED_KEY] = moved
    return notices, changed, fallbacks
