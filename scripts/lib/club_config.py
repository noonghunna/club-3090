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

USER-FACING: ``bash scripts/settings.sh`` (show, get, set, unset, migrate, path,
compose-env-file) is the one command for people to see and change settings; it is
a thin wrapper around ``settings_main`` here. ``migrate`` copies the repo .env
into the store and never changes that file; it also copies this rig's own gateway
routes and their keys out of the checkout (``litellm_local.py``). report.sh's "Settings" section is
``settings_report_markdown`` (CLI: ``settings-report --root ROOT``), which has no
way to print a secret.

Standard library only: the launcher path must run on a bare ``python3``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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


# The per-user cache and data directories (#1466 phase 4). Same rule as config_dir:
# the override, else the XDG variable, else under HOME; not created, not expanded.
# Unlike the config dir they can be SAVED settings (settings.sh set CLUB3090_CACHE_DIR=…),
# so pass an environment the loader has filled (load()) when that matters.
#   cache_dir  compile caches, one subdirectory per engine image (engine_cache.py)
#   data_dir   the KV-offload disk tier's default home (<data_dir>/kv-offload)
def cache_dir(environ=None) -> Path:
    """The per-user cache directory (not created)."""
    env = os.environ if environ is None else environ
    if env.get("CLUB3090_CACHE_DIR"):
        return Path(env["CLUB3090_CACHE_DIR"])
    base = env.get("XDG_CACHE_HOME") or os.path.join(env.get("HOME", "~"), ".cache")
    return Path(base) / "club-3090"


def data_dir(environ=None) -> Path:
    """The per-user data directory (not created)."""
    env = os.environ if environ is None else environ
    if env.get("CLUB3090_DATA_DIR"):
        return Path(env["CLUB3090_DATA_DIR"])
    base = env.get("XDG_DATA_HOME") or os.path.join(env.get("HOME", "~"), ".local", "share")
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


def redact(resolved: dict[str, tuple[str, str]], secret_keys=()) -> dict[str, tuple[str, str]]:
    """Hide every secret value. `secret_keys` hides more keys whatever their
    winning source — settings.sh passes the keys secrets.env holds, so a value
    kept there stays hidden even when the shell or club3090.env overrides it."""
    extra = set(secret_keys)
    return {k: (s, ("<set, hidden>" if v else "<empty>") if is_secret(k, s) or k in extra else v)
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


def _rewrite(path: Path, updates: dict[str, str], removals: set[str], lock_dir: Path | None = None) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_dir = lock_dir or path.parent
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _locked(lock_dir):
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


def unset_values(keys, which: str = "global", environ=None, repo_root=None) -> list[Path]:
    """Remove keys from the chosen store file and, when `repo_root` is given, from
    that checkout's legacy .env as well. The repo .env is read last, so clearing a
    setting only in the store would let an old copy there come back into effect —
    e.g. a cleared model-default pin. Returns the files actually rewritten. The
    legacy .env keeps its comments, order and mode, and is never created; its lock
    lives in the config dir, so nothing is added to the checkout."""
    for k in keys:
        check_key(k)
    touched = []
    path = target_path(which, environ)
    if path.exists():
        _rewrite(path, {}, set(keys))
        touched.append(path)
    if repo_root is not None:
        legacy = Path(repo_root) / ".env"
        if legacy.is_file() and set(keys) & set(parse_env_file(legacy)):
            _rewrite(legacy, {}, set(keys), lock_dir=config_dir(environ))
            touched.append(legacy)
    return touched


# ── the settings command (scripts/settings.sh) and report.sh's section ──────
# settings.sh is a thin wrapper around settings_main(); every rule and message
# lives here, next to the loader, so the command, the launchers and c3 can't
# disagree about a setting. store_file_for / save_settings / unset_everywhere /
# migrate / pending_migration / settings_rows are plain functions for c3 to reuse.
SETTINGS_CMD = "bash scripts/settings.sh"


def _secret_keys(environ=None) -> set[str]:
    """The keys secrets.env holds. They are hidden whatever their name, and
    whichever layer wins."""
    return set(parse_env_file(target_path("secrets", environ)))


def store_file_for(key: str, environ=None) -> str:
    """Where `settings.sh set` saves `key`: "secrets" for a credential-looking name
    (is_secret) or a key secrets.env already holds (it was filed with the
    secrets; moving it to club3090.env would drop it out of the 0600 file), else
    "global"."""
    return "secrets" if is_secret(key, "") or key in _secret_keys(environ) else "global"


def save_settings(values: dict[str, str], environ=None):
    """`settings.sh set`. Every pair is checked before anything is written, so one
    refused value saves nothing. Each key goes to store_file_for(key). A key saved
    to secrets.env is then removed from club3090.env: that file is read first, so
    a copy there would keep the old value in effect (and keep a credential outside
    the 0600 file). Returns ({path: [keys saved]}, {path: [keys removed]})."""
    for k, v in values.items():
        check_key(k)
        check_value(k, v)
    groups: dict[str, dict[str, str]] = {"global": {}, "secrets": {}}
    for k, v in values.items():
        groups[store_file_for(k, environ)][k] = v
    saved, removed = {}, {}
    for which in ("global", "secrets"):
        if groups[which]:
            saved[set_values(groups[which], which, environ)] = list(groups[which])
    glob = target_path("global", environ)
    shadowed = [k for k in groups["secrets"] if k in parse_env_file(glob)]
    if shadowed:
        _rewrite(glob, {}, set(shadowed))
        removed[glob] = shadowed
    return saved, removed


def unset_everywhere(keys, repo_root=None, environ=None) -> dict:
    """`settings.sh unset`: remove keys from club3090.env, secrets.env and, with
    `repo_root`, that checkout's legacy .env — a copy left in any of them would
    still be read. Only files holding one of the keys are rewritten (the legacy
    .env as in unset_values: comments, order and mode kept, lock in the config
    dir, never created). Returns {path: [keys removed]}."""
    keys = list(dict.fromkeys(keys))
    for k in keys:
        check_key(k)
    files = [(target_path("global", environ), None), (target_path("secrets", environ), None)]
    if repo_root is not None:
        files.append((Path(repo_root) / ".env", config_dir(environ)))
    removed = {}
    for path, lock_dir in files:
        present = [k for k in keys if k in parse_env_file(path)]
        if present and path.is_file():
            _rewrite(path, {}, set(present), lock_dir=lock_dir)
            removed[path] = present
    return removed


def _gateway_files():
    """scripts/lib/litellm_local.py: this rig's own gateway routes and their keys,
    which `migrate`, `show` and `path` cover too (#1466, 4c). Imported late: it
    imports this module."""
    try:
        from scripts.lib import litellm_local
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import litellm_local  # type: ignore
    return litellm_local


def _reason(key: str, err: Exception) -> str:
    msg = str(err)
    return msg[len(key) + 2:] if msg.startswith(f"{key}: ") else msg


def migrate(repo_root, dry_run: bool = False, environ=None) -> dict:
    """Copy every setting in `repo_root`/.env into the store (`settings.sh migrate`).

    * A credential-looking name (is_secret) goes to secrets.env, the rest to
      club3090.env.
    * A key either store file already holds is skipped: the store wins, as it
      already does when both are read. `differs` lists the ones whose value
      differs, with secret values left out (None).
    * A value the writer refuses stays where it is (it is still read) and is
      listed in `refused` with the writer's reason, never the value.
    * The repo .env is never modified or deleted — only read. It stays the last
      layer read, so no effective value changes, and running migrate again
      copies nothing.
    * dry_run builds the same report and writes nothing.

    Returns a JSON-serialisable dict: repo_env, config_dir, found (the repo .env
    exists), dry_run, copied {club3090.env: [keys], secrets.env: [keys]} (what a
    dry run would copy), same [keys], differs [{key, file, secret, stored, repo}],
    refused [{key, reason}]."""
    legacy = Path(repo_root) / ".env"
    d = config_dir(environ)
    secrets = parse_env_file(d / SECRETS_FILE)
    stored = {k: (SECRETS_FILE, v) for k, v in secrets.items()}
    stored.update({k: (GLOBAL_FILE, v) for k, v in parse_env_file(d / GLOBAL_FILE).items()})
    report = {"repo_env": str(legacy), "config_dir": str(d), "found": legacy.is_file(),
              "dry_run": bool(dry_run), "copied": {GLOBAL_FILE: [], SECRETS_FILE: []},
              "same": [], "differs": [], "refused": []}
    plan: dict[str, dict[str, str]] = {"global": {}, "secrets": {}}
    for key, value in parse_env_file(legacy).items():
        if key in stored:
            where, have = stored[key]
            if have == value:
                report["same"].append(key)
            else:
                secret = is_secret(key, where) or key in secrets
                report["differs"].append({"key": key, "file": where, "secret": secret,
                                          "stored": None if secret else have,
                                          "repo": None if secret else value})
            continue
        try:
            check_value(key, value)
        except ConfigError as e:
            report["refused"].append({"key": key, "reason": _reason(key, e)})
            continue
        plan["secrets" if is_secret(key, "") else "global"][key] = value
    for which, label in (("global", GLOBAL_FILE), ("secrets", SECRETS_FILE)):
        report["copied"][label] = list(plan[which])
        if plan[which] and not dry_run:
            set_values(plan[which], which, environ)
    return report


def pending_migration(repo_root, environ=None) -> list[str]:
    """The keys in `repo_root`/.env that migrate would copy (not in the store yet,
    and a value the writer accepts)."""
    if repo_root is None:
        return []
    copied = migrate(repo_root, dry_run=True, environ=environ)["copied"]
    return copied[GLOBAL_FILE] + copied[SECRETS_FILE]


def migrate_hint(count: int, repo_root, environ=None) -> str:
    return (f"{count} setting(s) still live in {Path(repo_root) / '.env'} — "
            f"{SETTINGS_CMD} migrate moves them to {config_dir(environ)}/.")


MIGRATE_NOTICE_STAMP = ".notice-migrate"


def migrate_pending(repo_root, environ=None) -> tuple[list[str], str]:
    """Everything `settings.sh migrate` would still copy out of the checkout: the
    repo .env's settings and this rig's own gateway routes and keys
    (litellm_local). Returns ``(phrases, fingerprint)``: one phrase per source
    (names and counts, never a value) and a hash of the pending key NAMES, which
    changes when something new lands in the checkout. ``([], "")`` when nothing
    is pending."""
    if repo_root is None:
        return [], ""
    keys = pending_migration(repo_root, environ)
    gateway = _gateway_files().pending(repo_root, environ)
    phrases = ([f"{len(keys)} setting(s) in {Path(repo_root) / '.env'}"] if keys else []) + gateway
    if not phrases:
        return [], ""
    return phrases, hashlib.sha256("\n".join(sorted(keys) + gateway).encode()).hexdigest()


def migrate_notice(repo_root, environ=None) -> list[str]:
    """The one-time "your settings still live in this checkout" notice (#1466) that
    the launchers, gpu-mode and c3 show, as lines; ``[]`` when there is nothing to
    say. Settings in the checkout keep working (the repo .env is read last), so
    this only points at `settings.sh migrate`.

    Shown ONCE per set of pending items: the config dir's stamp file holds the
    fingerprint (a hash of key names, no values). It comes back when something new
    lands in the checkout and goes quiet once migrated. Nothing is shown when the
    stamp can't be written — a notice that can't remember it was shown would
    repeat on every launch; `settings.sh show` lists the same any time."""
    env = os.environ if environ is None else environ
    phrases, fingerprint = migrate_pending(repo_root, env)
    if not phrases:
        return []
    d = config_dir(env)
    stamp = d / MIGRATE_NOTICE_STAMP
    with contextlib.suppress(OSError):
        if stamp.read_text(encoding="utf-8").strip() == fingerprint:
            return []
    try:
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        stamp.write_text(fingerprint + "\n", encoding="utf-8")
    except OSError:
        return []
    return [f"Your settings still live in this checkout ({'; '.join(phrases)}). They keep working.",
            f"To share them with every checkout and worktree, copy them to {d}/: "
            f"{SETTINGS_CMD} migrate --dry-run, then {SETTINGS_CMD} migrate "
            "(a copy — the repo files are left as they are).",
            f"Shown once; {SETTINGS_CMD} show lists them any time."]


def settings_rows(repo_root=None, environ=None, show_secrets: bool = False) -> list[dict]:
    """Every configured setting as {key, value, source, secret}, sorted by key.
    Secret values (is_secret, or held in secrets.env) are redacted unless
    show_secrets."""
    env = os.environ if environ is None else environ
    res = resolve(repo_root, env)
    sk = _secret_keys(env)
    shown = res if show_secrets else redact(res, sk)
    return [{"key": k, "value": shown[k][1], "source": s, "secret": is_secret(k, s) or k in sk}
            for k, (s, _v) in res.items()]


def _shown(value: str) -> str:
    return "<empty>" if value == "" else value


def _md_code(text: str) -> str:
    text = text.replace("|", "\\|")
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def settings_report_markdown(repo_root, environ=None) -> str:
    """report.sh's "Settings" section body. Secret values are ALWAYS hidden: there
    is deliberately no way to ask this function for them. report.sh still pipes
    it through its own redact(), which scrubs paths, host and user."""
    env = os.environ if environ is None else environ
    rows = settings_rows(repo_root, env, show_secrets=False)
    d = config_dir(env)
    out = ["_Every configured setting, its effective value and where it comes from: the shell wins, "
           "then `club3090.env`, `secrets.env`, and the repo `.env` last. Secret values are always "
           f"hidden, with or without `--no-redact`. The same view: `{SETTINGS_CMD} show`._", "",
           f"- **Config dir:** `{d}`" + ("" if d.is_dir() else " (not created yet)")]
    if rows:
        out += ["", "| Setting | Value | Source |", "|---|---|---|"]
        out += [f"| `{r['key']}` | {_md_code(_shown(r['value']))} | {r['source']} |" for r in rows]
    else:
        out.append("- _No settings configured._")
    pending = pending_migration(repo_root, env)
    if pending:
        out += ["", f"- ⚠ {migrate_hint(len(pending), repo_root, env)}"]
    for line in _gateway_files().notice_lines(repo_root, env):
        out += ["", f"- ⚠ {line}"]
    return "\n".join(out) + "\n"


def settings_help(repo_root=None, environ=None) -> str:
    env = os.environ if environ is None else environ
    d = config_dir(env)
    legacy = Path(repo_root) / ".env" if repo_root else "<repo>/.env"
    return f"""usage: {SETTINGS_CMD} <command> [options]

See and change your club-3090 settings: the models path, tokens and keys, default
pins and the rest. switch.sh, launch.sh, setup.sh, gpu-mode.sh and c3 all read them.

commands:
  show [--json] [--show-secrets]
                     every configured setting: its effective value and where it
                     comes from. Secret values are hidden unless --show-secrets.
  get KEY            one setting's effective value (exit 1 if it is not set)
  set KEY=VALUE...   save settings. Credential-looking names (…TOKEN, …_KEY,
                     …SECRET, …PASSWORD) and keys already in secrets.env go to
                     secrets.env; everything else to club3090.env.
  unset KEY...       remove settings from club3090.env, secrets.env AND the repo
                     .env, and list every file changed
  migrate [--dry-run]
                     copy the settings in the repo .env into your config dir.
                     Keys you have already saved are skipped (the saved value
                     wins); values that can't be stored stay in the repo .env.
                     Also copies this rig's own gateway routes
                     (services/litellm/config.local.yaml → litellm/) and the keys
                     they use (services/litellm/local.env → secrets.env).
                     The repo files themselves are never changed or deleted.
  path               where your settings are stored, and which files exist
  caches [--remove-legacy]
                     the compile caches and the KV-offload disk tier: where they
                     are and how big; --remove-legacy asks, then deletes the old
                     caches this checkout kept inside the repo
  compose-env-file [--out PATH]
                     for running `docker compose` yourself: write every resolved
                     setting (the shell winning, as in a launch) to a 0600 file
                     for `docker compose --env-file`, and print its path. It holds
                     your secrets too — remove it when you are done.

Where a setting comes from — the first of these that sets it wins:
  shell          exported in your environment (export KEY=VALUE)
  club3090.env   {d / GLOBAL_FILE}
  secrets.env    {d / SECRETS_FILE}   (mode 0600: tokens and keys)
  repo .env      {legacy}   (older installs; still read, last)
The config dir is $CLUB3090_CONFIG_DIR if set, else ${{XDG_CONFIG_HOME:-~/.config}}/club-3090.

Saved settings apply the next time you launch; a model that is already running
keeps the settings it started with.

Values are stored exactly as typed, one KEY=VALUE per line: no quotes, and no
$VAR or ~ expansion. A value that bash, docker compose and systemd would read
differently (quotes, $, `, \\, ' #', leading or trailing spaces) is refused with
the reason, and nothing is saved.

examples:
  {SETTINGS_CMD} set MODEL_DIR=/data/models
  {SETTINGS_CMD} set HF_TOKEN=hf_xxx            # goes to secrets.env
  {SETTINGS_CMD} show
  {SETTINGS_CMD} unset MODEL_DIR
  {SETTINGS_CMD} migrate --dry-run
  f="$({SETTINGS_CMD} compose-env-file)"; docker compose --env-file "$f" -f <compose.yml> up -d; rm -f "$f"

exit codes: 0 done · 1 get: not set · 2 refused, couldn't write, or usage error
"""


def _shell_notes(keys, removed: bool = False) -> None:
    for k in keys:
        if k in os.environ:
            what = ("stays in effect from there until you `unset " + k + "` in that shell"
                    if removed else "and the shell wins: `unset " + k + "` there for the saved value to apply")
            print(f"[settings] note: {k} is also set in your shell environment, {what}.", file=sys.stderr)


def _settings_show(a, root) -> int:
    env = os.environ
    rows = settings_rows(root, env, a.show_secrets)
    pending = pending_migration(root, env)
    if a.json:
        print(json.dumps({
            "config_dir": str(config_dir(env)),
            "repo_env": str(Path(root) / ".env") if root else None,
            "settings": {r["key"]: {"value": r["value"], "source": r["source"], "secret": r["secret"]}
                         for r in rows},
            "still_in_repo_env": pending,
        }, indent=2))
        return 0
    notes = []
    if not rows:
        print(f"No settings saved yet. Save one with: {SETTINGS_CMD} set KEY=VALUE")
    else:
        vals = [_shown(r["value"]) for r in rows]
        kw = max(len("SETTING"), *(len(r["key"]) for r in rows))
        vw = min(48, max(len("VALUE"), *(len(v) for v in vals)))
        print(f"{'SETTING':<{kw}}  {'VALUE':<{vw}}  SOURCE")
        for r, v in zip(rows, vals):
            print(f"{r['key']:<{kw}}  {v:<{vw}}  {r['source']}")
        if not a.show_secrets and any(r["secret"] for r in rows):
            notes.append("Secret values are hidden; add --show-secrets to print them.")
    notes += _gateway_files().notice_lines(root, env)
    if pending:
        notes.append(migrate_hint(len(pending), root, env))
    if notes:
        print("\n" + "\n".join(notes))
    return 0


def _settings_get(a, root) -> int:
    v = get(a.key, root)
    if v is None:
        print(f"[settings] {a.key} is not set", file=sys.stderr)
        return 1
    print(v)
    return 0


def _settings_set(a, root) -> int:
    vals = {}
    for i, pair in enumerate(a.pairs, 1):
        if "=" not in pair:
            raise ConfigError(f"argument {i} is not KEY=VALUE (it has no '='; not printed, in case it is a secret)")
        k, _, v = pair.partition("=")
        vals[k] = v
    saved, removed = save_settings(vals)
    for path, keys in saved.items():
        print(f"saved {', '.join(keys)} to {path}")
    for path, keys in removed.items():
        print(f"removed {', '.join(keys)} from {path} (read before {SECRETS_FILE}, it would have kept the old value)")
    _shell_notes(vals)
    return 0


def _settings_unset(a, root) -> int:
    removed = unset_everywhere(a.keys, root)
    for path, keys in removed.items():
        print(f"removed {', '.join(keys)} from {path}")
    gone = {k for ks in removed.values() for k in ks}
    missing = [k for k in dict.fromkeys(a.keys) if k not in gone]
    if missing:
        print(f"{', '.join(missing)}: not saved in any settings file — nothing to remove")
    _shell_notes(a.keys, removed=True)
    return 0


def _settings_migrate(a, root) -> int:
    if root is None:
        raise ConfigError(f"migrate needs the repo root: run it as `{SETTINGS_CMD} migrate`")
    code = _settings_migrate_env(a, root)
    gf = _gateway_files()
    lines = gf.migrate_lines(gf.migrate(root, a.dry_run))
    if lines:
        print("\n" + "\n".join(lines))
    return code


def _settings_migrate_env(a, root) -> int:
    r = migrate(root, a.dry_run)
    src = r["repo_env"]
    if not r["found"]:
        print(f"There is no {src} — nothing to migrate from it.")
        return 0
    print(f"Settings in {src} → {r['config_dir']}/" + ("   (dry run: nothing is written)" if a.dry_run else ""))
    verb = "would copy to" if a.dry_run else "copied to"
    for label in (GLOBAL_FILE, SECRETS_FILE):
        if r["copied"][label]:
            print(f"  {verb} {label}: {', '.join(r['copied'][label])}")
    if r["same"]:
        print(f"  already saved, same value: {', '.join(r['same'])}")
    if r["differs"]:
        print("  already saved with a different value — the saved one stays in effect:")
        for x in r["differs"]:
            if x["secret"]:
                print(f"    {x['key']}: in {x['file']} (a secret — values not shown)")
            else:
                print(f"    {x['key']}: {x['file']} has {x['stored']!r}, the repo .env has {x['repo']!r}")
    if r["refused"]:
        print("  can't be stored — left in the repo .env, which is still read:")
        for x in r["refused"]:
            print(f"    {x['key']}: {x['reason']}")
    n = len(r["copied"][GLOBAL_FILE]) + len(r["copied"][SECRETS_FILE])
    if a.dry_run:
        print(f"Dry run: nothing was written. Run it without --dry-run to copy {n} setting(s)."
              if n else "Dry run: nothing to copy.")
    elif n:
        print(f"Copied {n} setting(s). {src} was not changed; it is still read, last, "
              "so every launch sees the same values as before.")
    else:
        print("Nothing to copy: every setting there is already saved" +
              (", or can't be stored." if r["refused"] else "."))
    return 0


def _settings_path(a, root) -> int:
    env = os.environ
    d = config_dir(env)
    how = ("from $CLUB3090_CONFIG_DIR" if env.get("CLUB3090_CONFIG_DIR")
           else "from $XDG_CONFIG_HOME" if env.get("XDG_CONFIG_HOME") else "the default")
    rows = [("config dir", d, ("exists" if d.is_dir() else "not created yet; the first save creates it") + f", {how}")]
    for label in (GLOBAL_FILE, SECRETS_FILE):
        p = d / label
        state = "not created yet"
        if p.is_file():
            state = "exists"
            if label == SECRETS_FILE:
                mode = p.stat().st_mode & 0o777
                state += ", mode 0600" if mode == 0o600 else f", mode {mode:04o}: tighten it with chmod 600 {p}"
        rows.append((label, p, state))
    if root is not None:
        legacy = Path(root) / ".env"
        rows.append((LEGACY_LABEL, legacy, "exists; read last, after the files above" if legacy.is_file()
                     else "none (only older installs have one)"))
    rows += _gateway_files().path_rows(root, env)
    # #1466 phase 4: where the launchers put compile caches and the KV-offload disk tier.
    # Both can be saved settings, so read them the way a launch does.
    loaded = dict(env)
    load(root, loaded, warn=False)
    for label, var, xdg, p, what in (
            ("cache dir", "CLUB3090_CACHE_DIR", "XDG_CACHE_HOME", cache_dir(loaded),
             f"compile caches, one folder per engine image; sizes: {SETTINGS_CMD} caches"),
            ("data dir", "CLUB3090_DATA_DIR", "XDG_DATA_HOME", data_dir(loaded),
             "the KV-offload disk tier's default home, <data dir>/kv-offload")):
        how = f"from ${var}" if loaded.get(var) else f"from ${xdg}" if loaded.get(xdg) else "the default"
        rows.append((label, p, f"{'exists' if p.is_dir() else 'not created yet'}, {how}; {what}"))
    w = max(len(r[0]) for r in rows) + 1
    for label, p, state in rows:
        print(f"{label + ':':<{w}}  {p}  ({state})")
    return 0


def _settings_compose_env_file(a, root) -> int:
    print(write_compose_env_file(root, a.out))
    return 0


def settings_main(argv) -> int:
    """`bash scripts/settings.sh …` → `club_config.py settings --root ROOT …`."""
    argv = list(argv)
    root = None
    if argv[:1] == ["--root"] and len(argv) >= 2:
        root, argv = argv[1], argv[2:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        (sys.stdout if argv else sys.stderr).write(settings_help(root))
        return 0 if argv else 2
    ap = argparse.ArgumentParser(prog=SETTINGS_CMD, add_help=False,
                                 usage=f"{SETTINGS_CMD} <command> [options]   (--help: every command)")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>", required=True)
    p = sub.add_parser("show", help="every configured setting, its effective value and source",
                       description="Every configured setting: its effective value and where it comes from "
                                   "(shell, club3090.env, secrets.env, repo .env).")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--show-secrets", action="store_true", help="print secret values too (hidden by default)")
    p = sub.add_parser("get", help="one setting's effective value (exit 1 if unset)",
                       description="Print one setting's effective value; exit 1 if it is not set.")
    p.add_argument("key", metavar="KEY")
    p = sub.add_parser("set", help="save settings",
                       description="Save settings. Credential-looking names and keys already in secrets.env "
                                   "go to secrets.env (mode 0600); everything else to club3090.env.")
    p.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
    p = sub.add_parser("unset", help="remove settings everywhere they are saved",
                       description="Remove settings from club3090.env, secrets.env and the repo .env.")
    p.add_argument("keys", nargs="+", metavar="KEY")
    p = sub.add_parser("migrate", help="copy the repo .env's settings into your config dir",
                       description="Copy the settings in the repo .env into your config dir. Keys already saved "
                                   "are skipped; values that can't be stored stay in the repo .env. The repo "
                                   ".env is never changed or deleted.")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    sub.add_parser("path", help="where your settings are stored",
                   description="The config dir and the settings files, and whether each exists.")
    p = sub.add_parser("compose-env-file", help="a 0600 settings file for docker compose --env-file",
                       description="Write every resolved setting (the shell winning, as in a launch) to a 0600 "
                                   "file that `docker compose --env-file` reads back exactly, and print its path. "
                                   "It holds your secrets too: remove it when you are done.")
    p.add_argument("--out", metavar="PATH", help="write this file instead of a new temporary one")
    a = ap.parse_args(argv)
    run = {"show": _settings_show, "get": _settings_get, "set": _settings_set, "unset": _settings_unset,
           "migrate": _settings_migrate, "path": _settings_path, "compose-env-file": _settings_compose_env_file}
    try:
        return run[a.cmd](a, root)
    except ConfigError as e:
        print(f"[settings] refused: {e}. Nothing was changed.", file=sys.stderr)
        return 2
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())   # `show | head`
        return 1
    except OSError as e:
        print(f"[settings] ERROR: couldn't write {e.filename or 'a settings file'}: {e.strerror or e}", file=sys.stderr)
        return 2


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["settings"]:                     # scripts/settings.sh
        return settings_main(argv[1:])
    if argv[:1] == ["settings-report"]:              # report.sh: settings-report [--root ROOT]
        root = argv[2] if argv[1:2] == ["--root"] and len(argv) > 2 else None
        sys.stdout.write(settings_report_markdown(root))
        return 0
    ap = argparse.ArgumentParser(prog="club_config.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dir", help="print the config directory")
    sub.add_parser("cache-dir", help="print the cache directory (compile caches)")
    sub.add_parser("data-dir", help="print the data directory (the KV-offload disk tier)")
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
    u.add_argument("--root", help="also remove the keys from this checkout's legacy .env")
    u.add_argument("keys", nargs="+", metavar="KEY")
    n = sub.add_parser("migrate-notice", help="print the one-time 'settings still in the checkout' notice to stderr")
    n.add_argument("--root", required=True, help="repo root")
    n.add_argument("--prefix", default="[club-3090]", help="line prefix, e.g. [switch]")
    mp = sub.add_parser("migrate-pending", help="print what `settings.sh migrate` would still copy, '; '-joined (empty: nothing)")
    mp.add_argument("--root", required=True, help="repo root")
    a = ap.parse_args(argv)
    if a.cmd == "migrate-pending":
        try:
            print("; ".join(migrate_pending(a.root)[0]))
        except Exception:   # noqa: BLE001 — setup.sh treats "can't tell" as "nothing pending"
            pass
        return 0
    if a.cmd == "migrate-notice":
        # Never fails a launch: any problem just means no notice this time.
        try:
            for line in migrate_notice(a.root):
                print(f"{a.prefix} NOTE: {line}", file=sys.stderr)
        except Exception:   # noqa: BLE001 — a notice must not break the caller
            pass
        return 0
    try:
        if a.cmd == "dir":
            print(config_dir())
        elif a.cmd == "cache-dir":
            print(cache_dir())
        elif a.cmd == "data-dir":
            print(data_dir())
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
            touched = unset_values(a.keys, a.file, repo_root=a.root)
            where = ", ".join(str(p) for p in touched) or "nowhere (not set)"
            print(f"[config] removed {', '.join(a.keys)} from {where}", file=sys.stderr)
    except ConfigError as e:
        print(f"[config] ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
