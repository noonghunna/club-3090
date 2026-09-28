#!/usr/bin/env python3
"""slug_settings — the per-slug launch-settings store (club-3090#1465, phase 3c).

One file next to the global settings, in the per-user config directory
(``${XDG_CONFIG_HOME:-~/.config}/club-3090/``, override ``CLUB3090_CONFIG_DIR``;
the directory comes from ``club_config.config_dir``, never recomputed here)::

    slugs.json   {"version": 1, "slugs": {"<slug>": {"KEY": "value", ...}, ...}}

What goes in it is decided by ``launch_settings.py`` (the resolver): only launch
knobs from ``profiles/launch-knobs.json`` that the slug's compose actually reads,
with values from that slug's catalogued domain. This module is the STORE: it
reads, checks the shape, and writes — nothing else.

RULES
  * values are literal strings under the same rules as the global writer
    (``club_config.check_key`` / ``check_value``, used read-only): no quotes, ``$``,
    backslashes, newlines, surrounding whitespace, and never the empty string —
    "not set" is expressed by removing the key;
  * credential-looking keys (``*_TOKEN``, ``*_KEY``, ``*PASSWORD``, … — the loader's
    own ``is_secret`` rule) are refused: secrets stay global, in the 0600
    ``secrets.env``. So the file never holds one, and it is created 0644 like
    ``club3090.env``; an existing file keeps its mode;
  * writes replace the file atomically (temporary file in the same directory +
    fsync + ``os.replace``) under an exclusive ``flock`` on ``<config dir>/.lock`` —
    the lock file the loader's writer uses — and the read-modify-write happens
    inside the lock, so concurrent writers never lose each other's keys;
  * ``version`` must be 1. A missing file is an empty store. A file that is not
    JSON, has the wrong shape, or carries a newer version raises ``StoreError``:
    readers must not guess at it, and writers never overwrite it. Unknown top-level
    keys are kept on rewrite and reported as warnings.

Standard library only: the launcher path runs on a bare ``python3``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:                                    # imported as scripts.lib.slug_settings …
    from scripts.lib import club_config
except ImportError:                     # … or run as a file from scripts/lib
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import club_config  # type: ignore

STORE_FILE = "slugs.json"
VERSION = 1
# Registry slugs look like `sgl/qwen38-27b-dual-fast`, `llamacpp-club3090/glm53-…`,
# local ones `local/<name>`: printable, no whitespace, no quoting surprises.
_SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+/-]*\Z")


class StoreError(ValueError):
    """The store can't be read or written as asked. The message names the file."""


@dataclass
class Store:
    path: Path
    exists: bool
    slugs: dict[str, dict[str, str]] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)       # unknown top-level keys, kept on rewrite
    warnings: list[str] = field(default_factory=list)


def store_path(environ=None) -> Path:
    return club_config.config_dir(environ) / STORE_FILE


def check_slug(slug: str) -> None:
    if not isinstance(slug, str) or not _SLUG_RE.match(slug):
        raise StoreError(f"not a valid slug name: {slug!r}")


def check_entry(key: str, value: str) -> None:
    """Refuse what the store can't hold. Raises StoreError naming the key (never
    echoing a value that might be a credential)."""
    try:
        club_config.check_key(key)
    except club_config.ConfigError as exc:
        raise StoreError(str(exc)) from None
    if club_config.is_secret(key, ""):
        raise StoreError(f"{key} looks like a credential; per-slug settings never hold secrets — "
                         "keep it global, in secrets.env (bash scripts/settings.sh set "
                         f"{key}=… stores it there)")
    if not isinstance(value, str):
        raise StoreError(f"{key}: value must be a string")
    if value == "":
        raise StoreError(f"{key}: empty value — remove the key instead of setting it to the empty string")
    try:
        club_config.check_value(key, value)
    except club_config.ConfigError as exc:
        raise StoreError(str(exc)) from None


def parse(text: str, path: Path) -> Store:
    """Text of slugs.json → Store. Raises StoreError on anything it can't interpret."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StoreError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno}) — "
                         "fix it or move it aside") from None
    if not isinstance(data, dict):
        raise StoreError(f"{path}: the top level must be a JSON object")
    ver = data.get("version")
    if ver != VERSION:
        if isinstance(ver, int) and not isinstance(ver, bool) and ver > VERSION:
            raise StoreError(f"{path}: written by a newer club-3090 (version {ver}); this checkout reads "
                             f"version {VERSION}. Update this checkout, or give it its own CLUB3090_CONFIG_DIR")
        raise StoreError(f"{path}: \"version\" must be {VERSION}, got {ver!r}")
    slugs = data.get("slugs", {})
    if not isinstance(slugs, dict):
        raise StoreError(f"{path}: \"slugs\" must be an object of slug -> {{KEY: value}}")
    out: dict[str, dict[str, str]] = {}
    for slug, entry in slugs.items():
        if not isinstance(entry, dict):
            raise StoreError(f"{path}: slugs[{slug!r}] must be an object of KEY -> value")
        for key, value in entry.items():
            if not isinstance(value, str):
                raise StoreError(f"{path}: slugs[{slug!r}][{key!r}] must be a string (write it as \"{value}\")")
        out[slug] = dict(entry)
    extra = {k: v for k, v in data.items() if k not in ("version", "slugs")}
    warnings = [f"{path}: unknown top-level key {k!r} (kept, not used)" for k in sorted(extra)]
    return Store(path=path, exists=True, slugs=out, extra=extra, warnings=warnings)


def read(environ=None) -> Store:
    """The store (empty when the file is absent). Raises StoreError when it exists
    but can't be read or interpreted."""
    path = store_path(environ)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Store(path=path, exists=False)
    except OSError as exc:
        raise StoreError(f"{path}: cannot read: {exc.strerror or exc}") from None
    except UnicodeDecodeError:
        raise StoreError(f"{path}: not UTF-8 text — fix it or move it aside") from None
    return parse(text, path)


def slug_values(slug: str, environ=None) -> dict[str, str]:
    return dict(read(environ).slugs.get(slug, {}))


def render(store: Store) -> str:
    doc = {"version": VERSION,
           "slugs": {s: dict(sorted(store.slugs[s].items())) for s in sorted(store.slugs) if store.slugs[s]}}
    doc.update(store.extra)
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


@contextlib.contextmanager
def _locked(directory: Path):
    """An exclusive flock on <config dir>/.lock — the lock file the loader's writer
    (club_config.py) takes, so the two writers never interleave in this directory."""
    import fcntl
    with open(directory / ".lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _update(mutate, environ=None) -> Store:
    """Read, mutate and atomically replace the store, all under the lock."""
    path = store_path(environ)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _locked(path.parent):
        store = read(environ)                     # raises on a file we must not overwrite
        mutate(store)
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o644
        fd, tmp = tempfile.mkstemp(prefix=f".{STORE_FILE}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(render(store))
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, mode)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return store


def set_values(slug: str, values: dict[str, str], environ=None) -> Path:
    """Save KEY=value pairs for one slug. Every pair is checked before anything is
    written: one bad pair saves nothing."""
    check_slug(slug)
    if not values:
        raise StoreError("nothing to save")
    for k, v in values.items():
        check_entry(k, v)

    def mutate(store: Store) -> None:
        store.slugs.setdefault(slug, {}).update(values)

    return _update(mutate, environ).path


def unset_values(slug: str, keys, environ=None) -> list[str]:
    """Remove keys from one slug's entry (the entry goes when it empties). Returns
    the keys that were actually stored. Never creates the file."""
    check_slug(slug)
    keys = list(keys)
    for k in keys:
        try:
            club_config.check_key(k)
        except club_config.ConfigError as exc:
            raise StoreError(str(exc)) from None
    if not store_path(environ).exists():
        return []
    removed: list[str] = []

    def mutate(store: Store) -> None:
        entry = store.slugs.get(slug, {})
        for k in keys:
            if k in entry:
                del entry[k]
                removed.append(k)
        if slug in store.slugs and not entry:
            del store.slugs[slug]

    _update(mutate, environ)
    return removed


# ── CLI (the store only — switch.sh --set/--unset go through launch_settings.py,
#    which checks each value against the slug's catalogued domain first) ─────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="slug_settings.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("path", help="print the store's path")
    sh = sub.add_parser("show", help="print the stored settings (all slugs, or one)")
    sh.add_argument("--slug")
    s = sub.add_parser("set", help="store KEY=VALUE pairs for a slug (no catalogue check: use switch.sh --set)")
    s.add_argument("slug")
    s.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
    u = sub.add_parser("unset", help="remove keys from a slug")
    u.add_argument("slug")
    u.add_argument("keys", nargs="+", metavar="KEY")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "path":
            print(store_path())
        elif a.cmd == "show":
            st = read()
            for w in st.warnings:
                print(f"[slug-settings] WARN: {w}", file=sys.stderr)
            view = st.slugs if a.slug is None else {a.slug: st.slugs.get(a.slug, {})}
            print(json.dumps(view, indent=2, sort_keys=True))
        elif a.cmd == "set":
            vals = {}
            for p in a.pairs:
                if "=" not in p:
                    raise StoreError(f"expected KEY=VALUE, got {p!r}")
                k, _, v = p.partition("=")
                vals[k] = v
            print(f"[slug-settings] saved {', '.join(vals)} for {a.slug} in {set_values(a.slug, vals)}",
                  file=sys.stderr)
        elif a.cmd == "unset":
            gone = unset_values(a.slug, a.keys)
            print(f"[slug-settings] removed {', '.join(gone) or 'nothing (not set)'} for {a.slug}", file=sys.stderr)
    except StoreError as exc:
        print(f"[slug-settings] ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[slug-settings] ERROR: cannot write {store_path()}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
