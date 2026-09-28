#!/usr/bin/env python3
"""club-3090 configuration: the ONE Python loader and the ONE writer (club-3090#1466).

WHY THIS EXISTS
---------------
Until #1466 the repo-root ``.env`` was read five different ways (a line parser in
``switch.sh``, ``set -a; source`` in ``launch.sh``/``report.sh``/``setup.sh``,
``repo_dotenv.py``, ``docker compose --env-file`` and single-key greps), with five
different precedence rules and edge cases (duplicate keys: first wins in
``switch.sh``, last wins in ``repo_dotenv.py``). Settings now live per user, in
``${XDG_CONFIG_HOME:-~/.config}/club-3090/`` (override: ``CLUB3090_CONFIG_DIR``):

    club3090.env   global settings              (plain KEY=value)
    secrets.env    tokens and keys, mode 0600   (plain KEY=value)

and the repo-root ``.env`` is still read as a legacy fallback. ``club-config.sh``
is the bash twin of this module; ``scripts/tests/test-club-config.sh`` holds the
two to byte-identical output, and ``test-config-single-parser.sh`` fails any other
code that parses these files.

PRECEDENCE (highest first): the process environment ("shell") > club3090.env >
secrets.env > repo .env. A key set in the environment is never overridden, even
when it is set to the empty string.

PARSING (every file, identically):
  * whitespace (space, tab, CR, VT, FF) is trimmed from each line; blank lines and
    lines starting with ``#`` are skipped; an optional ``export `` prefix is dropped;
  * the key is the text before the first ``=``, trimmed, and must be a shell
    identifier (``[A-Za-z_][A-Za-z0-9_]*``) — anything else is skipped;
  * the value is the rest, trimmed; ONE matching pair of surrounding ``"`` or ``'``
    is removed. Nothing else: no expansion, no escapes, no inline comments;
  * within one file the LAST assignment of a key wins (as with ``source``,
    ``docker compose --env-file`` and systemd ``EnvironmentFile=``).

WRITING: only through ``set_values`` / ``unset_values`` (CLI: ``set`` / ``unset``).
Values are written as bare ``KEY=value``, the one form bash, ``docker compose
--env-file`` and systemd read identically, so values that those readers would
alter (quotes, ``$``, backslashes, `` #``, surrounding whitespace, newlines) are
refused. Comments and key order are kept; the file is replaced atomically under a
lock, and ``secrets.env`` is created 0600.

Standard library only: the launcher path must run on a bare ``python3``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

GLOBAL_FILE = "club3090.env"
SECRETS_FILE = "secrets.env"
LEGACY_LABEL = "repo .env"
SHELL_LABEL = "shell"

_WS = " \t\r\v\f"                      # == bash [[:space:]] in the C locale, minus newline
_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# Characters a bare KEY=value line cannot carry without some reader altering it.
_UNSAFE_VALUE_RE = re.compile(r"[\"'`$\\\n\r\x00]|[ \t]#|\A#")


def config_dir(environ=None) -> Path:
    """The per-user config directory (not created)."""
    env = os.environ if environ is None else environ
    if env.get("CLUB3090_CONFIG_DIR"):
        return Path(env["CLUB3090_CONFIG_DIR"])
    base = env.get("XDG_CONFIG_HOME") or os.path.join(env.get("HOME", "~"), ".config")
    return Path(base) / "club-3090"


def parse_env_file(path) -> dict[str, str]:
    """``KEY=value`` file → ``{KEY: value}`` under the rules above. ``{}`` when the
    file is absent or unreadable; never raises (a malformed file must not take
    down a launch)."""
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.split("\n"):
        line = raw.strip(_WS)
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip(_WS)
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip(_WS)
        if not _KEY_RE.match(key):
            continue
        value = value.strip(_WS)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def layers(repo_root=None, environ=None) -> list[tuple[str, Path]]:
    """The config files, HIGHEST precedence first, as (label, path)."""
    d = config_dir(environ)
    out = [(GLOBAL_FILE, d / GLOBAL_FILE), (SECRETS_FILE, d / SECRETS_FILE)]
    if repo_root is not None:
        out.append((LEGACY_LABEL, Path(repo_root) / ".env"))
    return out


def resolve(repo_root=None, environ=None) -> dict[str, tuple[str, str]]:
    """Every key any config file sets → (source label, effective value), sorted by
    key. The environment wins; otherwise the highest-precedence file that sets it."""
    env = os.environ if environ is None else environ
    merged: dict[str, tuple[str, str]] = {}
    for label, path in reversed(layers(repo_root, env)):     # lowest first; later overwrite
        for key, value in parse_env_file(path).items():
            merged[key] = (label, value)
    out = {}
    for key in sorted(merged):
        out[key] = (SHELL_LABEL, env[key]) if key in env else merged[key]
    return out


_SECRET_NAME_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|MASTER_KEY|_KEY)\Z")


def is_secret(key: str, source: str) -> bool:
    """A value that must never be printed: anything from secrets.env, or a name that
    looks like a credential wherever it came from (a token still sitting in .env)."""
    return source == SECRETS_FILE or bool(_SECRET_NAME_RE.search(key))


def redact(resolved: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    return {k: (s, ("<set, hidden>" if v else "<empty>") if is_secret(k, s) else v)
            for k, (s, v) in resolved.items()}


# A value that looks like it expected shell expansion: `$VAR`, `${VAR}` or a
# leading `~`. launch.sh, report.sh and setup.sh used to `source` the repo .env,
# which expanded these; every reader now takes values literally (as switch.sh and
# docker compose --env-file with a quoted value always did), so say so.
_EXPANSION_RE = re.compile(r"\$\{?[A-Za-z_]|\A~(/|\Z)")


def expansion_warnings(resolved: dict[str, tuple[str, str]]) -> list[str]:
    """One message per file value that looks like it expected expansion. Never
    includes the value, and skips secrets (a `$` in a password is just a `$`)."""
    return [f"[config] WARN: {k} (from {src}) contains '$VAR' or a leading '~'. Settings are read literally now, "
            f"not expanded the way 'source .env' did — write the full path."
            for k, (src, v) in resolved.items()
            if src != SHELL_LABEL and not is_secret(k, src) and _EXPANSION_RE.search(v)]


def load(repo_root=None, environ=None, warn=True) -> dict[str, str]:
    """Fill UNSET keys of the environment from the config files. Returns
    {key: source label} for the keys actually injected. Prints expansion warnings
    to stderr unless warn=False."""
    env = os.environ if environ is None else environ
    injected = {}
    res = resolve(repo_root, env)
    for key, (label, value) in res.items():
        if label != SHELL_LABEL:
            env[key] = value
            injected[key] = label
    if warn:
        for msg in expansion_warnings(res):
            print(msg, file=sys.stderr)
    return injected


def format_resolved(resolved: dict[str, tuple[str, str]]) -> str:
    """The parity format shared with club-config.sh: KEY<TAB>SOURCE<TAB>VALUE."""
    return "".join(f"{k}\t{src}\t{val}\n" for k, (src, val) in resolved.items())


def get(key: str, repo_root=None, environ=None):
    """The effective value of one setting (the environment wins), or None."""
    env = os.environ if environ is None else environ
    if key in env:
        return env[key]
    hit = resolve(repo_root, env).get(key)
    return hit[1] if hit else None


# ── handing settings to `docker compose --env-file` ──────────────────────────
# For callers that can't pass settings through the environment — `sudo docker
# compose` strips it (gpu-mode.sh) — the resolved values go in a temporary file.
# Measured on docker compose v5.5.1 (2026-09-28): a single-quoted value is fully
# literal ($, #, ", backslashes survive); inside double quotes \" \\ and \$ are
# escapes. So: single quotes, or double quotes with escapes when the value has a '.
def compose_env_quote(value: str) -> str:
    if "'" not in value:
        return f"'{value}'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'


def compose_env_text(resolved: dict[str, tuple[str, str]]) -> str:
    return "".join(f"{k}={compose_env_quote(v)}\n" for k, (_src, v) in resolved.items())


def write_compose_env_file(repo_root=None, out=None, environ=None) -> Path:
    """Write every resolved setting (the environment winning, as everywhere) to a
    0600 file docker compose reads back exactly. A new temp file unless `out`.
    The caller removes it. Empty when nothing is configured."""
    text = compose_env_text(resolve(repo_root, environ))
    if out is None:
        fd, name = tempfile.mkstemp(prefix="club3090-compose-", suffix=".env")
    else:
        name = str(out)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(name, 0o600)
    return Path(name)


# ── writer ───────────────────────────────────────────────────────────────────
class ConfigError(ValueError):
    pass


def check_key(key: str) -> None:
    if not _KEY_RE.match(key or ""):
        raise ConfigError(f"not a valid setting name: {key!r} (letters, digits and _; not starting with a digit)")


def check_value(key: str, value: str) -> None:
    if value != value.strip(_WS):
        raise ConfigError(f"{key}: value has leading or trailing whitespace, which readers would strip")
    m = _UNSAFE_VALUE_RE.search(value)
    if m:
        what = {"\n": "a newline", "\r": "a carriage return", "\x00": "a NUL byte"}.get(m.group(0), repr(m.group(0)))
        if m.group(0).endswith("#"):
            what = "a '#' that docker compose would read as a comment"
        raise ConfigError(f"{key}: value contains {what}; bash, docker compose and systemd would not all read it "
                          "the same way, so it can't be stored in this file")


@contextlib.contextmanager
def _locked(directory: Path):
    import fcntl
    with open(directory / ".lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _rewrite(path: Path, updates: dict[str, str], removals: set[str]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _locked(path.parent):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            lines, mode = [], (0o600 if path.name == SECRETS_FILE else 0o644)
        if lines and lines[-1] == "":
            lines.pop()
        pending = dict(updates)
        out = []
        for raw in lines:
            line = raw.strip(_WS)
            body = line[len("export "):].lstrip(_WS) if line.startswith("export ") else line
            key = body.partition("=")[0].strip(_WS) if "=" in body and not line.startswith("#") else None
            if key in removals:
                continue
            if key in updates:
                if key in pending:                      # first occurrence carries the new value
                    out.append(f"{key}={pending.pop(key)}")
                continue                                # later duplicates are dropped
            out.append(raw)
        out.extend(f"{k}={v}" for k, v in pending.items())
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(out) + ("\n" if out else ""))
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, mode)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


def target_path(which: str = "global", environ=None) -> Path:
    if which not in ("global", "secrets"):
        raise ConfigError(f"unknown config file {which!r} (global or secrets)")
    return config_dir(environ) / (GLOBAL_FILE if which == "global" else SECRETS_FILE)


def set_values(values: dict[str, str], which: str = "global", environ=None) -> Path:
    for k, v in values.items():
        check_key(k)
        check_value(k, v)
    path = target_path(which, environ)
    _rewrite(path, dict(values), set())
    return path


def unset_values(keys, which: str = "global", environ=None) -> Path:
    for k in keys:
        check_key(k)
    path = target_path(which, environ)
    if path.exists():
        _rewrite(path, {}, set(keys))
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="club_config.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dir", help="print the config directory")
    r = sub.add_parser("resolve", help="print every configured key: KEY<TAB>SOURCE<TAB>VALUE")
    r.add_argument("--root", help="repo root, to include its legacy .env")
    r.add_argument("--json", action="store_true")
    r.add_argument("--show-secrets", action="store_true",
                   help="print secret values (hidden by default: secrets.env, *TOKEN, *_KEY, ...)")
    s = sub.add_parser("set", help="store KEY=VALUE pairs")
    s.add_argument("--file", choices=("global", "secrets"), default="global")
    s.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
    g = sub.add_parser("get", help="print one setting's effective value (exit 1 if unset)")
    g.add_argument("key")
    g.add_argument("--root", help="repo root, to include its legacy .env")
    c = sub.add_parser("compose-env-file", help="write resolved settings for docker compose --env-file; print its path")
    c.add_argument("--root", help="repo root, to include its legacy .env")
    c.add_argument("--out", help="path to write (default: a new 0600 temp file)")
    u = sub.add_parser("unset", help="remove keys")
    u.add_argument("--file", choices=("global", "secrets"), default="global")
    u.add_argument("keys", nargs="+", metavar="KEY")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "dir":
            print(config_dir())
        elif a.cmd == "resolve":
            res = resolve(a.root)
            if not a.show_secrets:
                res = redact(res)
            if a.json:
                print(json.dumps({k: {"source": s_, "value": v} for k, (s_, v) in res.items()}, indent=2))
            else:
                sys.stdout.write(format_resolved(res))
        elif a.cmd == "get":
            v = get(a.key, a.root)
            if v is None:
                return 1
            print(v)
        elif a.cmd == "compose-env-file":
            print(write_compose_env_file(a.root, a.out))
        elif a.cmd == "set":
            vals = {}
            for p in a.pairs:
                if "=" not in p:
                    raise ConfigError(f"expected KEY=VALUE, got {p!r}")
                k, _, v = p.partition("=")
                vals[k] = v
            print(f"[config] saved {', '.join(vals)} to {set_values(vals, a.file)}", file=sys.stderr)
        elif a.cmd == "unset":
            print(f"[config] removed {', '.join(a.keys)} from {unset_values(a.keys, a.file)}", file=sys.stderr)
    except ConfigError as e:
        print(f"[config] ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
