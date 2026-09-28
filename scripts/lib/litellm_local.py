#!/usr/bin/env python3
"""This rig's own LiteLLM gateway routes, and the keys they use (club-3090#1466, 4c).

WHERE THEY LIVE
---------------
In the club-3090 config dir (``club_config.config_dir()``: ``$CLUB3090_CONFIG_DIR``,
else ``${XDG_CONFIG_HOME:-~/.config}/club-3090``), next to the other settings, so
every checkout of the repo (a worktree too) serves the same routes:

    litellm/config.local.yaml   this rig's own routes: a cloud endpoint, a private
                                service. litellm-sync copies its ``model_list:``
                                entries into the runtime view the gateway serves.
    secrets.env (0600)          their keys, next to your other secrets. A route
                                names one as ``api_key: os.environ/<NAME>``; save it
                                with ``bash scripts/settings.sh set <NAME>=<key>``.

OLDER INSTALLS kept both in the checkout, gitignored: ``services/litellm/
config.local.yaml`` and ``services/litellm/local.env``. Both still work, so a rig
that never migrates serves the same routes with the same keys:

  * routes: the config-dir file is read when it exists, else the repo copy;
  * keys: the gateway compose still loads ``services/litellm/local.env`` itself,
    and the keys from your settings are loaded after it, so a saved key wins.

``bash scripts/settings.sh migrate`` copies both into the config dir (``migrate()``)
and never changes or deletes the repo files. Until then litellm-sync says so once
(``notice()``), and ``settings.sh show`` / ``path`` keep saying it.

HOW THE GATEWAY GETS THE KEYS
-----------------------------
Not from secrets.env itself: that file also holds the HF token and anything else
kept there, and the gateway needs none of it. ``write_route_keys_file()`` writes a
0600 temp file holding only the keys a route references (``os.environ/<NAME>`` in
the catalog ``config.yaml`` or in this rig's routes), with the values the one
loader resolves (the shell > club3090.env > secrets.env > the repo .env). Whatever
starts the gateway (gpu-mode, scripts/litellm-log.sh, c3) passes that file's PATH as
``CLUB3090_LITELLM_ROUTE_KEYS``, which the compose loads as an ``env_file``, and
deletes it after the compose call: compose reads an env_file only when it creates
the container. The values then live in the container's environment, as the ones
from local.env always did. A value is never printed and never on a command line.

Standard library only, like the rest of the switch path.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
import tempfile
from pathlib import Path

try:                                    # imported as scripts.lib.litellm_local (c3) …
    from scripts.lib import club_config
except ImportError:                     # … or run as a file from scripts/lib
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import club_config  # type: ignore

SUBDIR = "litellm"
ROUTES_FILE = "config.local.yaml"
LEGACY_ROUTES = Path("services/litellm") / ROUTES_FILE
LEGACY_KEYS = Path("services/litellm/local.env")
CATALOG = Path("services/litellm/config.yaml")
KEYS_VAR = "CLUB3090_LITELLM_ROUTE_KEYS"
# The compose sets these itself (`environment:` wins over an env_file), so they are
# never handed over as route keys.
COMPOSE_OWN = frozenset({"LITELLM_MASTER_KEY", "LITELLM_LOG"})
NOTICE_STAMP = ".notice-gateway-files-in-checkout"
MIGRATE_CMD = f"{club_config.SETTINGS_CMD} migrate"
_REF_RE = re.compile(r"os\.environ/([A-Za-z_][A-Za-z0-9_]*)")
_COMMENT_RE = re.compile(r"(?:^|\s)#")


# ── where the routes are ─────────────────────────────────────────────────────
def routes_path(environ=None) -> Path:
    """Where this rig's own routes are saved (whether or not the file exists)."""
    return club_config.config_dir(environ) / SUBDIR / ROUTES_FILE


def active_routes(root, environ=None, override: bool = True):
    """``(path, origin)`` of the routes file litellm-sync reads. origin is
    "override" (``C3_LITELLM_LOCAL_CONFIG``, a test seam — used even when the file
    is missing), "config" (the config dir), "legacy" (this checkout's
    services/litellm), or ``(None, "none")``."""
    env = os.environ if environ is None else environ
    if override and env.get("C3_LITELLM_LOCAL_CONFIG"):
        return Path(env["C3_LITELLM_LOCAL_CONFIG"]), "override"
    p = routes_path(env)
    if p.is_file():
        return p, "config"
    if root is not None and (Path(root) / LEGACY_ROUTES).is_file():
        return Path(root) / LEGACY_ROUTES, "legacy"
    return None, "none"


def display_path(path, root=None, environ=None) -> str:
    """A path as a person reads it: relative to the checkout, else ``~/…``."""
    env = os.environ if environ is None else environ
    p = Path(path)
    if root is not None:
        with contextlib.suppress(ValueError, OSError):
            return str(p.resolve().relative_to(Path(root).resolve()))
    home = env.get("HOME")
    if home:
        with contextlib.suppress(ValueError):
            return "~/" + str(p.relative_to(home))
    return str(p)


# ── which keys the gateway gets ──────────────────────────────────────────────
def _read(path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _refs(text: str) -> set:
    """``os.environ/<NAME>`` references outside YAML comments."""
    out = set()
    for line in text.splitlines():
        out.update(_REF_RE.findall(_COMMENT_RE.split(line, 1)[0]))
    return out


def referenced_keys(root, environ=None, override: bool = True) -> list:
    """The key names the gateway's routes read from its environment: the catalog
    config.yaml's and this rig's own routes' (``active_routes``), minus the ones
    the compose sets itself."""
    names = set()
    if root is not None:
        names |= _refs(_read(Path(root) / CATALOG))
    path, _origin = active_routes(root, environ, override)
    if path is not None:
        names |= _refs(_read(path))
    return sorted(names - COMPOSE_OWN)


def route_keys(root, environ=None) -> dict:
    """``{NAME: value}`` for each referenced key a settings file sets (the shell
    wins, as in every launch). A key no settings file sets is left to the legacy
    local.env, which the compose still loads."""
    res = club_config.resolve(root, environ)
    return {k: res[k][1] for k in referenced_keys(root, environ) if k in res}


def write_route_keys_file(root, environ=None):
    """Write the referenced keys (``route_keys``) to a new 0600 temp file that
    ``docker compose`` reads back exactly, and return its path; the caller removes
    it. ``None`` (and no file) when no route needs a saved key."""
    keys = route_keys(root, environ)
    if not keys:
        return None
    text = "".join(f"{k}={club_config.compose_env_quote(v)}\n" for k, v in keys.items())
    fd, name = tempfile.mkstemp(prefix="club3090-litellm-", suffix="-keys.env")   # created 0600
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(name, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(name)
        raise
    return Path(name)


def keys_missing_from_gateway(root, have_names, environ=None) -> list:
    """The keys a route uses that a gateway created with the variables ``have_names``
    lacks, though they are saved (in your settings, or the older local.env): a
    route added after the gateway started, say, or a key saved since. A restart
    can't add them — a container keeps the environment it was created with — so
    the caller says to recreate it. ``have_names`` are variable NAMES only."""
    wanted = set(referenced_keys(root, environ))
    saved = set(route_keys(root, environ))
    if root is not None:
        saved |= set(club_config.parse_env_file(Path(root) / LEGACY_KEYS))
    return sorted((wanted & saved) - set(have_names))


# ── migrate: copy the checkout's files into the config dir ───────────────────
def _copy_new(src: Path, dst: Path) -> None:
    """Copy src to dst (0600) atomically; never replace a dst that exists."""
    dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", dir=dst.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(src.read_bytes())
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        try:
            os.link(tmp, dst)                  # fails if dst appeared meanwhile
        except FileExistsError:
            raise
        except OSError:                        # no hard links on this filesystem
            if dst.exists():
                raise FileExistsError(str(dst))
            os.replace(tmp, dst)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def migrate(root, dry_run: bool = False, environ=None) -> dict:
    """Copy this checkout's gateway files into the config dir (part of
    ``settings.sh migrate``).

    * routes: services/litellm/config.local.yaml → <config dir>/litellm/
      config.local.yaml (0600), unless that file exists; then state is "same"
      or "differs" (the config-dir one is read).
    * keys: each key in services/litellm/local.env that a route references goes
      to secrets.env. A key already saved is skipped ("same"/"differs", values
      never reported: these are credentials); a value the writer refuses stays
      where it is ("refused", with the reason); a key no route references is
      left alone ("unreferenced").
    * The repo files are never modified or deleted — only read. The gateway
      compose still loads local.env, and the saved keys are loaded after it, so
      no key the gateway gets changes. dry_run writes nothing.

    Returns a JSON-serialisable dict: dry_run, config_dir, routes {from, to,
    state: none|copied|same|differs}, keys {from, found, copied, same, differs
    [{key, file}], refused [{key, reason}], unreferenced}."""
    env = os.environ if environ is None else environ
    root = Path(root)
    d = club_config.config_dir(env)
    src_r, dst_r, src_k = root / LEGACY_ROUTES, routes_path(env), root / LEGACY_KEYS
    routes = {"from": str(src_r), "to": str(dst_r), "state": "none"}
    keys = {"from": str(src_k), "found": src_k.is_file(), "copied": [], "same": [], "differs": [],
            "refused": [], "unreferenced": []}
    report = {"dry_run": bool(dry_run), "config_dir": str(d), "routes": routes, "keys": keys}

    if src_r.is_file():
        if dst_r.is_file():
            routes["state"] = "same" if src_r.read_bytes() == dst_r.read_bytes() else "differs"
        else:
            routes["state"] = "copied"
            if not dry_run:
                try:
                    _copy_new(src_r, dst_r)
                except FileExistsError:
                    routes["state"] = "same" if src_r.read_bytes() == dst_r.read_bytes() else "differs"

    if keys["found"]:
        # The routes the gateway reads once the copy above is done: the same
        # content, so the same names, whether or not this is a dry run.
        wanted = set(referenced_keys(root, env, override=False))
        stored = {k: (club_config.SECRETS_FILE, v)
                  for k, v in club_config.parse_env_file(d / club_config.SECRETS_FILE).items()}
        stored.update({k: (club_config.GLOBAL_FILE, v)
                       for k, v in club_config.parse_env_file(d / club_config.GLOBAL_FILE).items()})
        plan = {}
        for key, value in club_config.parse_env_file(src_k).items():
            if key not in wanted:
                keys["unreferenced"].append(key)
            elif key in stored:
                where, have = stored[key]
                if have == value:
                    keys["same"].append(key)
                else:
                    keys["differs"].append({"key": key, "file": where})
            else:
                try:
                    club_config.check_value(key, value)
                except club_config.ConfigError as e:
                    msg = str(e)
                    keys["refused"].append({"key": key, "reason": msg[len(key) + 2:] if msg.startswith(f"{key}: ") else msg})
                    continue
                plan[key] = value
        keys["copied"] = list(plan)
        if plan and not dry_run:
            club_config.set_values(plan, "secrets", env)
    return report


def migrate_lines(r: dict) -> list:
    """``settings.sh migrate``'s account of the gateway files (``[]`` when this
    checkout has none). Never a key's value."""
    routes, keys, dry = r["routes"], r["keys"], r["dry_run"]
    if routes["state"] == "none" and not keys["found"]:
        return []
    out = [f"This rig's own gateway routes and keys → {r['config_dir']}/"
           + ("   (dry run: nothing is written)" if dry else "")]
    state = routes["state"]
    if state == "copied":
        out.append(f"  {'would copy' if dry else 'copied'} {routes['from']} to {routes['to']}")
    elif state == "same":
        out.append(f"  {routes['from']}: already in {routes['to']}, same content")
    elif state == "differs":
        out.append(f"  {routes['from']}: {routes['to']} already exists with other content — that one is read, "
                   "the repo copy is not. Move any change across by hand.")
    if keys["found"]:
        src = keys["from"]
        if keys["copied"]:
            out.append(f"  {'would copy' if dry else 'copied'} route keys from {src} to "
                       f"{club_config.SECRETS_FILE}: {', '.join(keys['copied'])}")
        if keys["same"]:
            out.append(f"  already saved, same value: {', '.join(keys['same'])}")
        for x in keys["differs"]:
            out.append(f"  {x['key']}: already saved in {x['file']} with a different value — the saved one is "
                       "used (a key — values not shown)")
        for x in keys["refused"]:
            out.append(f"  {x['key']}: can't be stored ({x['reason']}) — left in {src}, which the gateway still loads")
        if keys["unreferenced"]:
            out.append(f"  no route uses {', '.join(keys['unreferenced'])} — left in {src}, "
                       "which the gateway still loads")
    if dry:
        return out
    # What the person can remove now.
    if state in ("copied", "same"):
        out.append(f"  {routes['from']} was not changed and is no longer read: you can delete it.")
    if keys["found"]:
        if keys["refused"] or keys["unreferenced"]:
            keep = [x["key"] for x in keys["refused"]] + keys["unreferenced"]
            out.append(f"  {keys['from']} was not changed. Keep it for {', '.join(keep)}; "
                       "the keys saved in your settings win over it.")
        else:
            out.append(f"  {keys['from']} was not changed. Every route key in it is saved in your settings, "
                       "which win over it: you can delete it.")
    return out


def pending(root, environ=None) -> list:
    """What migrate would still copy, one phrase each (``[]`` when nothing)."""
    if root is None:
        return []
    r = migrate(root, dry_run=True, environ=environ)
    out = []
    if r["routes"]["state"] == "copied":
        out.append(r["routes"]["from"])
    if r["keys"]["copied"]:
        out.append(f"{len(r['keys']['copied'])} route key(s) in {r['keys']['from']}")
    return out


def notice_lines(root, environ=None, include_pending: bool = True) -> list:
    """Why the checkout's gateway files still matter, for litellm-sync and
    ``settings.sh show`` (``[]`` when they don't). ``include_pending=False`` leaves
    out "still read from the checkout", which the launchers' one-time notice
    (club_config.migrate_notice) already says."""
    if root is None:
        return []
    env = os.environ if environ is None else environ
    out = []
    todo = pending(root, env) if include_pending else []
    if todo:
        out.append(f"This rig's own gateway routes still come from the checkout ({'; '.join(todo)}). "
                   f"They keep working; `{MIGRATE_CMD}` copies them to {club_config.config_dir(env)}/ "
                   "and leaves the repo files as they are.")
    legacy = Path(root) / LEGACY_ROUTES
    dst = routes_path(env)
    if legacy.is_file() and dst.is_file() and legacy.read_bytes() != dst.read_bytes():
        out.append(f"{legacy} is no longer read — {dst} is — and the two differ. "
                   "Move any change across, then delete the repo copy.")
    return out


def notice(root, quiet: bool = False, environ=None, stream=None) -> None:
    """Print ``notice_lines`` to stderr: every time unless quiet, and when quiet
    (switch.sh, gpu-mode) only the first time, remembered by a stamp file in the
    config dir. Silent when the routes come from C3_LITELLM_LOCAL_CONFIG: the
    checkout's files are not read then."""
    env = os.environ if environ is None else environ
    if env.get("C3_LITELLM_LOCAL_CONFIG"):
        return
    # Quiet runs come from switch.sh / gpu-mode, which print the combined one-time
    # migrate notice (club_config.migrate_notice) themselves; only the "two copies
    # differ" warning is left for this one.
    lines = notice_lines(root, env, include_pending=not quiet)
    if not lines:
        return
    d = club_config.config_dir(env)
    stamp = d / NOTICE_STAMP
    if quiet and stamp.exists():
        return
    for line in lines:
        print(f"[litellm-sync] NOTE: {line}", file=stream or sys.stderr)
    with contextlib.suppress(OSError):
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        stamp.touch()


def path_rows(root, environ=None) -> list:
    """``settings.sh path`` rows: (label, path, state)."""
    env = os.environ if environ is None else environ
    dst = routes_path(env)
    if dst.is_file():
        state = "exists; litellm-sync serves its routes"
    elif root is not None and (Path(root) / LEGACY_ROUTES).is_file():
        state = f"none; the gateway reads {Path(root) / LEGACY_ROUTES} instead ({MIGRATE_CMD} copies it)"
    else:
        state = "none (only needed for routes of your own)"
    rows = [("gateway routes", dst, state)]
    if root is not None and (Path(root) / LEGACY_KEYS).is_file():
        rows.append(("gateway keys (old)", Path(root) / LEGACY_KEYS,
                     f"exists; the gateway still loads it, after which the keys saved in {club_config.SECRETS_FILE} win"))
    return rows


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="litellm_local.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("route-keys-file",
                       help="write the keys this rig's gateway routes use to a new 0600 temp file and print its "
                            "path (prints nothing when no route needs a saved key); the caller removes it")
    k.add_argument("--root", required=True, help="the checkout whose gateway is started")
    a = ap.parse_args(argv)
    if a.cmd == "route-keys-file":
        try:
            p = write_route_keys_file(a.root)
        except (OSError, club_config.ConfigError) as e:
            print(f"[litellm] could not write the gateway's route keys: {e.__class__.__name__}: "
                  f"{getattr(e, 'strerror', None) or e}", file=sys.stderr)
            return 1
        if p is not None:
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
