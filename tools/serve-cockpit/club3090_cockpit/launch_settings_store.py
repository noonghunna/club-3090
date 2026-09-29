"""c3's door to the launch settings (club-3090#1465, phase 3d).

WHY THIS EXISTS
---------------
``switch.sh --set / --unset / --explain`` read and write the per-slug launch
settings through ONE resolver, ``scripts/lib/launch_settings.py``: for every
catalogued knob a slug's compose reads, the value the next launch uses and the
layer it comes from —

    shell > this slug > model pin > club3090.env > secrets.env > repo .env > compose default

c3's "Launch settings" form (``LaunchSettingsScreen`` in app.py) shows and edits
the same thing. Everything here is the resolver's, called in-process:

    read      view()            resolve() + to_json() — the rows switch.sh --explain prints
    write     save() / remove() save_values() / remove_values() — switch.sh --set / --unset,
                                with the same refusals, word for word
    running   drift()           the running container's own environment (docker inspect,
                                read-only) against the next launch's values

Nothing here validates a value, parses a settings file or writes one: a value
the form sends is checked by the resolver, and a refusal is shown in its words.
``scripts/tests/test-config-single-parser.sh`` fails a c3 file that reads the
settings files itself.

THE SHELL LAYER
---------------
"shell" means a key in c3's environment that the settings loader did not put
there. c3 puts one key there itself — the saved HF token, applied at start-up
so its downloads get it (``services.CockpitData._hf_token_injected``) — and
says so through ``loaded`` (the resolver's ``--loaded KEY=FILE``, which
switch.sh fills from ``CLUB3090_CONFIG_SOURCE``). The serve plan runs
``switch.sh`` with c3's environment, so a knob exported in the shell c3 was
started from wins there too; the form says so.

DRIFT
-----
A running container keeps the settings it started with. For a knob the compose
forwards into the container's environment (``environment: - KNOB`` or
``- KNOB=${KNOB:-…}``), ``docker inspect`` shows the value it got, so the form
can compare it with the next launch's. A knob the compose only splices into the
command line (``${KNOB:-x}`` inside ``command:``) never reaches the environment:
those are shown as "can't check" rather than guessed. The inspect template
filters inside docker, so the rest of the container's environment (tokens
included) never reaches c3 or its logs.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

_LS: Any = None


def _candidate_roots() -> list[Path]:
    roots = [Path(__file__).resolve().parents[3]]
    env_root = os.environ.get("C3_REPO_ROOT")
    if env_root:
        roots.append(Path(env_root))
    return roots


def resolver():
    """The ``scripts.lib.launch_settings`` module (the one resolver). The cockpit
    venv doesn't carry the repo root on ``sys.path``; put the first root that has
    the resolver there, as ``settings_store.loader`` does for the loader."""
    global _LS
    if _LS is None:
        for root in _candidate_roots():
            if (root / "scripts" / "lib" / "launch_settings.py").is_file():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                break
        from scripts.lib import launch_settings

        _LS = launch_settings
    return _LS


class LaunchSettingsError(Exception):
    """The resolver can't answer for this slug (unknown slug, unusable catalogue,
    unreadable per-slug store, a write that failed). The message is its own."""


# ── the form's rows ──────────────────────────────────────────────────────────


@dataclass
class KnobRow:
    """One launch setting the slug's compose reads."""

    knob: str
    value: str                          # as switch.sh --explain prints it
    source: str                         # the layer it comes from (the resolver's name)
    detail: str = ""                    # e.g. the pin behind a "model pin" value
    overrides: list[tuple[str, str]] = field(default_factory=list)   # lower layers it beats
    compose_default: Optional[str] = None
    unset_means: str = ""
    allowed: Optional[str] = None
    enforced: Optional[str] = None
    errors: list[str] = field(default_factory=list)   # the next launch's refusals about it
    description: str = ""
    saved: Optional[str] = None         # this slug's own saved value, None when none
    in_container_env: bool = False      # docker inspect can show what a container got
    env_forms: list[str] = field(default_factory=list)   # how the compose forwards it (KnobUse)
    secret: bool = False                # value never shown (it comes from secrets.env)
    # The value the next launch passes ("" / None = unset → compose default). Kept
    # for the drift comparison only; never rendered for a secret.
    raw: Optional[str] = None


@dataclass
class RunningKnob:
    status: str                         # same | differs | unchecked | unknown
    shown: str = ""                     # what the container got, for display
    why: str = ""


@dataclass
class RunningContainer:
    name: str
    started_at: str = ""
    knobs: dict[str, RunningKnob] = field(default_factory=dict)
    error: str = ""

    @property
    def differing(self) -> list[str]:
        return sorted(k for k, r in self.knobs.items() if r.status == "differs")


@dataclass
class LaunchView:
    slug: str
    available: bool = True
    reason: str = ""
    order: list[str] = field(default_factory=list)
    store: str = ""
    engine_kind: str = ""
    compose_path: str = ""
    knobs: list[KnobRow] = field(default_factory=list)
    unread: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    catalogued: list[str] = field(default_factory=list)   # every knob the catalogue has
    doc: dict = field(default_factory=dict)               # the resolver's to_json()
    running: list[RunningContainer] = field(default_factory=list)
    running_error: str = ""

    def knob(self, name: str) -> Optional[KnobRow]:
        return next((k for k in self.knobs if k.knob == name), None)

    @property
    def general_errors(self) -> list[str]:
        """Refusals not about one knob (e.g. an unreadable slugs.json)."""
        mine = {e for k in self.knobs for e in k.errors}
        return [e for e in self.errors if e not in mine]


def view(slug: str, repo_root, environ: Optional[Mapping[str, str]] = None,
         loaded: Optional[Mapping[str, str]] = None) -> LaunchView:
    """The slug's launch settings as the next ``switch.sh <slug>`` from this
    environment would resolve them. ``loaded`` names the keys c3 itself put in
    the environment and the file each came from (they are not the shell).
    Blocking (the resolver reads the registry and asks engine-kind.sh): call it
    off the UI thread."""
    ls = resolver()
    env = os.environ if environ is None else environ
    try:
        cat = ls.lk.load_catalogue()
        res = ls.resolve(slug, repo_root, env, dict(loaded or {}), catalogue=cat)
    except (ls.ResolveError, ls.lk.CatalogueError) as exc:
        return LaunchView(slug=slug, available=False, reason=str(exc))
    doc = ls.to_json(res)
    try:
        saved = ls.slug_settings.read(env).slugs.get(slug, {})
    except ls.slug_settings.StoreError:
        saved = {}                      # the resolver already reports it (res.errors)
    rows: list[KnobRow] = []
    for k in doc["knobs"]:
        name = k["knob"]
        s = res.settings[name]
        use = res.uses.get(name)
        rows.append(KnobRow(
            knob=name,
            value=k["value"],
            source=k["source"],
            detail=k.get("detail") or "",
            overrides=[(o["source"], o["value"]) for o in k.get("overrides") or []],
            compose_default=k.get("compose_default"),
            unset_means=k.get("unset_means") or "",
            allowed=k.get("allowed"),
            enforced=k.get("enforced"),
            errors=list(k.get("errors") or []),
            description=str(cat["knobs"].get(name, {}).get("description", "")),
            saved=saved.get(name),
            in_container_env=bool(use is not None and use.forwarded),
            env_forms=use.env_forms() if use is not None else [],
            secret=ls.club_config.is_secret(name, s.source),
            raw=s.layers[0].value if s.layers else None,
        ))
    return LaunchView(
        slug=slug, order=list(doc["order"]), store=doc["store"],
        engine_kind=doc["engine_kind"], compose_path=res.compose_path, knobs=rows,
        unread=list(doc["unread"]), errors=list(doc["errors"]),
        warnings=list(doc["warnings"]), catalogued=ls.lk.knob_names(cat), doc=doc,
    )


# ── writes (switch.sh --set / --unset) ───────────────────────────────────────


@dataclass
class Change:
    """What a save or a removal did. ``problems`` non-empty = refused, nothing
    written; each is the resolver's own sentence."""

    slug: str
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    path: str = ""

    @property
    def ok(self) -> bool:
        return not self.problems


def _change(ch) -> Change:
    return Change(slug=ch.slug, problems=list(ch.problems), warnings=list(ch.warnings),
                  notes=list(ch.notes), changed=list(ch.changed), path=str(ch.path or ""))


def save(slug: str, values: Mapping[str, str], repo_root,
         environ: Optional[Mapping[str, str]] = None) -> Change:
    """Save KEY=VALUE for THIS slug (slugs.json), exactly as
    ``switch.sh --set <slug> KEY=VALUE`` does. Raises LaunchSettingsError when
    the resolver or the store can't be used at all."""
    ls = resolver()
    try:
        return _change(ls.save_values(slug, dict(values), repo_root, environ))
    except (ls.ResolveError, ls.slug_settings.StoreError) as exc:
        raise LaunchSettingsError(str(exc)) from None
    except OSError as exc:
        raise LaunchSettingsError(
            f"cannot write {ls.slug_settings.store_path(environ)}: {exc.strerror or exc}") from None


def remove(slug: str, keys, repo_root, environ: Optional[Mapping[str, str]] = None) -> Change:
    """Remove this slug's own saved values, as ``switch.sh --unset <slug> KEY``."""
    ls = resolver()
    try:
        return _change(ls.remove_values(slug, list(keys), repo_root, environ))
    except (ls.ResolveError, ls.slug_settings.StoreError) as exc:
        raise LaunchSettingsError(str(exc)) from None
    except OSError as exc:
        raise LaunchSettingsError(
            f"cannot write {ls.slug_settings.store_path(environ)}: {exc.strerror or exc}") from None


# ── the running container (read-only docker) ─────────────────────────────────

COMPOSE_FILES_LABEL = "com.docker.compose.project.config_files"
# The slug a launcher stamped on the container (scripts/lib/slug_label.py; estate
# pods too): tells apart two slugs that share one compose file.
SLUG_LABEL = "club3090.slug"
# `docker ps` rows: name <TAB> the compose file(s) the container was started from
# <TAB> its club3090.slug label ("" when it was started without one).
PS_FORMAT = ("{{.Names}}\t{{.Label \"" + COMPOSE_FILES_LABEL + "\"}}"
             "\t{{.Label \"" + SLUG_LABEL + "\"}}")


def containers_for(compose_path: str, ps_stdout: str, slug: Optional[str] = None) -> list[str]:
    """Running containers of this slug. A container carrying the ``club3090.slug``
    label is this slug's only when the label says so — two slugs can share a compose
    file (vllm/dual and vllm/qwen-27b-dual-fast), and per-slug settings make them
    run differently. One without it (started before the label, or by hand) is
    matched by the compose label, from any checkout of the repo."""
    want = compose_path.strip("/")
    out = []
    for line in (ps_stdout or "").splitlines():
        name, sep, rest = line.partition("\t")
        if not sep or not name.strip():
            continue
        files, _, label = rest.partition("\t")
        label = label.strip()
        if label and slug:
            if label == slug:
                out.append(name.strip())
            continue
        for f in files.split(","):
            f = f.strip()
            if f and (f == want or f.endswith("/" + want)):
                out.append(name.strip())
                break
    return out


def inspect_format(knobs) -> str:
    """A ``docker inspect --format`` template that prints the start time, then
    ONLY the named knobs' environment entries (JSON-quoted, one per line). The
    filter runs inside docker: nothing else in the environment is printed."""
    names = " ".join(json.dumps(k) for k in knobs)
    return ("{{json .State.StartedAt}}{{println}}"
            "{{range .Config.Env}}{{$p := split . \"=\"}}"
            f"{{{{if eq (index $p 0) {names}}}}}{{{{json .}}}}{{{{println}}}}{{{{end}}}}"
            "{{end}}")


def parse_inspect(stdout: str) -> tuple[str, dict[str, Optional[str]]]:
    """(started_at, {knob: value}) from :func:`inspect_format` output. A value is
    None for an entry without ``=`` — docker compose forwards an unset bare
    ``- KNOB`` that way — and a knob absent from the dict is not in the env."""
    started, env = "", {}
    for i, line in enumerate((stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if not isinstance(item, str):
            continue
        if i == 0:
            started = item
            continue
        k, sep, v = item.partition("=")
        env[k] = v if sep else None
    return started, env


def _effective(value: Optional[str], default: Optional[str]) -> Optional[str]:
    """What the compose ends up using: the value, or its default when unset."""
    return value if value not in (None, "") else default


def drift(row: KnobRow, env: dict[str, Optional[str]]) -> RunningKnob:
    """Compare one knob's next-launch value with what a running container got."""
    if not row.in_container_env:
        return RunningKnob("unchecked", "can't check",
                           "the compose passes it on the command line, not in the container's "
                           "environment, so what the container got can't be read")
    if row.knob in env:
        got = env[row.knob]
    elif row.env_forms and all(f == "bare" for f in row.env_forms):
        got = None                      # a bare `- KNOB` that was unset may be left out
    else:
        return RunningKnob("unknown", "?",
                           "not in the container's environment — it was started from another "
                           "version of this compose")
    started = _effective(got, row.compose_default)
    same = started == _effective(row.raw, row.compose_default)
    if row.secret:
        shown = "(hidden)"
    else:
        shown = started if started not in (None, "") else "(unset)"
    return RunningKnob("same" if same else "differs", shown,
                       "" if same else f"started with {shown}; the next launch uses {row.value}")


def running_state(view_: LaunchView, name: str, inspect_stdout: str) -> RunningContainer:
    """One running container of the slug, each knob compared (:func:`drift`)."""
    started, env = parse_inspect(inspect_stdout)
    rc = RunningContainer(name=name, started_at=started)
    for row in view_.knobs:
        rc.knobs[row.knob] = drift(row, env)
    return rc
