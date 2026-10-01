#!/usr/bin/env python3
"""launch_settings — the ONE resolver for launch settings (club-3090#1465, phases 3b/3c).

For one slug, every catalogued launch knob its compose READS (the knob catalogue
``profiles/launch-knobs.json`` + the compose scan in ``profiles/launch_knobs.py``)
gets an effective value and the layer it came from. Highest first (maintainer
decision 1 on #1466: the more specific setting wins, the shell wins over all):

    shell            exported in the environment of this one launch — a key the
                     settings loader did NOT put there (``--loaded`` names those)
    this slug        the per-slug store, <config dir>/slugs.json (slug_settings.py)
    model pin        CLUB3090_THINKING_<MODEL> = on|off → ENABLE_THINKING = true|false
                     (inherit, or anything else, adds nothing — switch.sh's rule since
                     the #1014 follow-up)
    club3090.env     the global settings, in the loader's own order:
    secrets.env        club3090.env > secrets.env > the checkout's legacy .env
    repo .env
    compose default  nothing sets it: the compose's own ``${KNOB:-x}`` applies

An EMPTY value from the shell or a settings file means "unset" (every catalogued
knob is read as ``${KNOB:-x}``, so empty == the compose default): it is not
validated, and an empty shell value still masks the saved layers, which is how
``KV_OFFLOAD_GB= bash scripts/switch.sh <slug>`` turns a saved tier off for one
launch. The per-slug store never holds an empty value.

VALIDATION (``check``; switch.sh runs it in check_variant, BEFORE the running slug
is torn down). Only a value that came from a layer is checked — never a compose
default:
  * the value must be in the slug's catalogued domain (the variant matching its
    engine kind, model and compose). With ``--force`` a value outside a domain the
    catalogue marks ``enforced: unverified`` or ``none`` is a warning instead: there
    is no evidence it breaks anything. ``boot`` / ``request`` domains always refuse;
  * the catalogue's ``requires`` rules (KV_OFFLOAD_DISK=1 needs KV_OFFLOAD_GB,
    KV_OFFLOAD_DISK_GB needs KV_OFFLOAD_DISK=1), unless the compose defaults alone
    break the rule;
  * host RAM: KV_OFFLOAD_GB × host_factor + 28 GiB must fit in MemTotal (see
    ``ram_error``), and on vLLM the tier must fit in the host's /dev/shm, which holds it
    (``shm_error``, #1503). Both run even with ``--force``. There is no disk-tier check: the
    disk tier is uncapped by default (discussion #1419), so nothing here invents a cap;
  * a corrupt, unreadable or newer-version slugs.json refuses too: the saved
    per-slug values can't be known, and launching without them would boot a config
    the user did not save.
WARNINGS (never refuse): a saved per-slug or global value, or a thinking pin, for a
knob this slug's compose doesn't read (it does nothing — #1465 gotcha 4); a per-slug
key that isn't a catalogued knob; a thinking pin that isn't on|off|inherit.

DELIVERY (``exports``): switch.sh exports every effective value from ``this slug``,
``model pin`` or a settings file right before ``docker compose up`` — overriding a
global value the loader exported, never the shell.

CALLERS: switch.sh runs the CLI below (check / exports / explain / set / unset). c3's
"Launch settings" form calls ``resolve`` / ``to_json`` / ``save_values`` /
``remove_values`` in-process (tools/serve-cockpit/club3090_cockpit/
launch_settings_store.py), so the form and ``switch.sh --explain`` / ``--set`` /
``--unset`` share every value, source and message.

Engine KIND comes from scripts/lib/engine-kind.sh (via launch_knobs_check.engine_kinds),
never classified here (#1282). Standard library only: this is on the launcher path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib import club_config, slug_settings  # noqa: E402
from scripts.lib.profiles import compose_registry as cr  # noqa: E402
from scripts.lib.profiles import launch_knobs as lk  # noqa: E402
from scripts.lib.profiles.launch_knobs_check import engine_kinds  # noqa: E402

SHELL = "shell"
SLUG = "this slug"
PIN = "model pin"
DEFAULT = "compose default"
GLOBAL_LABELS = (club_config.GLOBAL_FILE, club_config.SECRETS_FILE, club_config.LEGACY_LABEL)
ORDER = (SHELL, SLUG, PIN) + GLOBAL_LABELS + (DEFAULT,)

# Per-model pins that feed a knob. Only the thinking pin exists; on/off map exactly
# as switch.sh's apply_thinking_pin_env always did.
PIN_VALUES = {"on": "true", "off": "false"}

# ── host-RAM rule for the KV-offload RAM tier ─────────────────────────────────
# Refuse KV_OFFLOAD_GB when   tier × HOST_FACTOR[engine] + RAM_RESERVE_GIB > MemTotal
# (all GiB; KV_OFFLOAD_GB is GiB in every compose that reads it).
#  * MemTotal, not MemAvailable: the check runs BEFORE the running slug is torn
#    down, so whatever that slug holds now is free by boot time. The question is
#    "can this host hold the tier at all", not "is it free this second".
#  * RAM_RESERVE_GIB: the headroom preflight_lmcache_ram budgets for the serving
#    process of a 27B TP=2 slug plus the OS (scripts/preflight.sh). Every compose
#    that reads KV_OFFLOAD_GB is that shape (the Qwen3.8 27B dual MTP composes).
#  * HOST_FACTOR: vLLM pins exactly the tier (cpu_bytes_to_use = GiB × 2^30);
#    SGLang's host use runs above the setting — 74 GiB for a 64 GiB tier measured
#    (launch-knobs.json KV_OFFLOAD_GB caveat; docs/engines/SGLANG.md).
RAM_RESERVE_GIB = 28
HOST_FACTOR = {"vllm": 1.0, "sglang": 74 / 64}
MEMINFO_ENV = "CLUB3090_MEMINFO_FILE"      # test seam: a meminfo-format file instead of /proc/meminfo

# ── /dev/shm rule for vLLM's RAM tier (#1503) ─────────────────────────────────
# vLLM's OffloadingConnector keeps the RAM tier in a file under /dev/shm, in both modes the composes
# use (RAM only, and RAM + disk). Every compose that reads KV_OFFLOAD_GB on vLLM runs with
# `ipc: host`, so the limit is the HOST's /dev/shm (by default half of RAM) and the compose's
# shm_size does not apply. A 64 GiB tier on a 125 GiB host (63 GiB /dev/shm) passes the RAM rule
# above and then fails every worker with "Insufficient space in /dev/shm".
#  * Refuse a tier larger than /dev/shm's SIZE: stopping the running slug cannot free that.
#  * Warn when it is larger than what is FREE right now: the running slug's own tier is released
#    when it stops (after this check runs); anything else holding /dev/shm is not.
# SGLang's HiCache keeps its host pool in ordinary pinned memory, so the rule is vLLM-only.
SHM_PATH = "/dev/shm"
SHM_ENV = "CLUB3090_SHM_STATVFS"           # test seam: "<size GiB>:<free GiB>" instead of statvfs(/dev/shm)

ENFORCED_TEXT = {
    "boot": "the compose refuses it at boot",
    "request": "the server would boot, then fail every request that doesn't send its own value",
    "none": "nothing refuses it; the engine ignores or misreads it",
    "unverified": "the compose passes it through unchecked and nothing records what the engine then does",
}


class ResolveError(RuntimeError):
    """The slug, its compose, the catalogue or the registry can't be used."""


@dataclass
class Layer:
    source: str
    value: str
    detail: str = ""


@dataclass
class Setting:
    knob: str
    layers: list[Layer]                 # every layer that sets it, highest first
    default: str | None                 # compose fallback ("" = unset/off, None = varies)
    variant: dict | None
    unset_text: str = ""
    allowed: str | None = None          # the variant's domain in words (display only)
    errors: list[str] = field(default_factory=list)   # this knob's share of Resolution.errors

    @property
    def source(self) -> str:
        return self.layers[0].source if self.layers else DEFAULT

    @property
    def value(self) -> str | None:
        return self.layers[0].value if self.layers else self.default

    @property
    def is_unset(self) -> bool:
        return self.value in (None, "")


@dataclass
class Resolution:
    slug: str
    model: str
    engine_kind: str
    compose_path: str
    store_path: Path
    settings: dict[str, Setting] = field(default_factory=dict)
    unread: list[tuple[str, str, str]] = field(default_factory=list)   # (knob, source, value)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # The compose scan per catalogued knob (launch_knobs.KnobUse): how each knob
    # reaches the container — c3 reads a running container's environment for the
    # knobs the compose forwards there (KnobUse.forwarded).
    uses: dict = field(default_factory=dict)


# ── pieces ────────────────────────────────────────────────────────────────────
def _locating_env(env) -> dict:
    """Only what club_config needs to FIND the config dir — so the file layer is
    read from the files themselves, whatever the environment already carries."""
    return {k: env[k] for k in ("CLUB3090_CONFIG_DIR", "XDG_CONFIG_HOME", "HOME") if k in env}


def file_values(root, env) -> dict[str, tuple[str, str]]:
    """KEY → (file label, value) from the settings files (the loader's precedence)."""
    return club_config.resolve(root, _locating_env(env))


def shell_keys(env, loaded, files) -> set[str]:
    """Keys the environment sets that the settings loader did NOT put there. A key
    the loader injected counts as its file's, as long as the environment still
    carries that file's value (a later change in this process is the process's)."""
    loaded = loaded or {}
    return {k for k in env
            if not (k in loaded and k in files and files[k][1] == env[k])}


def facts(slug: str, root=ROOT, catalogue=None) -> tuple[dict, str, str, dict]:
    """(registry entry, engine kind, compose text, knob uses) for a slug."""
    root = Path(root)
    try:
        cat = catalogue or lk.load_catalogue()
        entry = cr.get_registry(root).get(slug)
    except (lk.CatalogueError, cr.LocalRegistryError) as exc:
        raise ResolveError(str(exc)) from None
    if entry is None:
        hint = ""
        if slug.endswith("/default"):
            hint = (" A <…>/default token isn't a slug: `bash scripts/switch.sh --explain "
                    f"{slug}` shows the slug it resolves to — name that one.")
        raise ResolveError(f"unknown slug {slug!r} — run: bash scripts/switch.sh --list.{hint}")
    try:
        kind = engine_kinds(root, [entry["engine"]])[entry["engine"]]
    except Exception as exc:                                     # noqa: BLE001 — bash missing, etc.
        raise ResolveError(f"cannot classify engine {entry['engine']!r}: {exc}") from None
    try:
        text = (root / entry["compose_path"]).read_text(encoding="utf-8")
        uses = lk.scan_compose_text(text, lk.knob_names(cat))
    except (OSError, ValueError) as exc:
        raise ResolveError(f"{entry['compose_path']}: cannot scan it for launch settings: {exc}") from None
    return entry, kind, text, uses


def read_meminfo_total_gib(env) -> float | None:
    path = env.get(MEMINFO_ENV) or "/proc/meminfo"
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024 * 1024)      # kB → GiB
    except (OSError, ValueError, IndexError):
        return None
    return None


def ram_error(value: str, engine_kind: str, env) -> tuple[str | None, str | None]:
    """(refusal, warning) for a KV_OFFLOAD_GB value on this host."""
    try:
        tier = float(value)
    except ValueError:
        return None, None                      # the domain check reports it
    total = read_meminfo_total_gib(env)
    if total is None:
        return None, f"cannot read MemTotal ({env.get(MEMINFO_ENV) or '/proc/meminfo'}); host-RAM check for KV_OFFLOAD_GB skipped"
    factor = HOST_FACTOR.get(engine_kind, 1.0)
    need = tier * factor + RAM_RESERVE_GIB
    if need <= total:
        return None, None
    cap = max(0.0, (total - RAM_RESERVE_GIB) / factor)
    over = f" × {factor:.2f} measured {engine_kind} host overhead" if factor != 1.0 else ""
    return (f"a {tier:g} GiB RAM tier needs ~{need:.0f} GiB of host RAM ({tier:g} GiB{over} + "
            f"{RAM_RESERVE_GIB} GiB for the serving process and the OS), but this host has "
            f"{total:.0f} GiB in total — at most {int(cap)} GiB fits here"), None


def read_shm_gib(env) -> tuple[float, float] | None:
    """(size, free) of /dev/shm in GiB, or None when it can't be read."""
    seam = env.get(SHM_ENV)
    if seam:
        try:
            size, free = (float(x) for x in seam.split(":"))
            return size, free
        except ValueError:
            return None
    try:
        st = os.statvfs(SHM_PATH)
    except OSError:
        return None
    return st.f_blocks * st.f_frsize / (1 << 30), st.f_bavail * st.f_frsize / (1 << 30)


def shm_error(value: str, engine_kind: str, env) -> tuple[str | None, str | None]:
    """(refusal, warning) for a vLLM KV_OFFLOAD_GB value against the host's /dev/shm (#1503)."""
    if engine_kind != "vllm":
        return None, None
    try:
        tier = float(value)
    except ValueError:
        return None, None                      # the domain check reports it
    got = read_shm_gib(env)
    if got is None:
        return None, f"cannot read {SHM_PATH}; /dev/shm check for KV_OFFLOAD_GB skipped"
    size, free = got
    enlarge = "sudo mount -o remount,size=<N>G /dev/shm (plus an fstab entry to keep it)"
    if tier > size:
        return (f"vLLM keeps the RAM tier in /dev/shm, and these composes use the host's (ipc: host), "
                f"which is {size:.1f} GiB here — at most {int(size)} GiB fits. Lower it, or enlarge "
                f"/dev/shm: {enlarge}"), None
    if tier > free:
        return None, (f"the KV_OFFLOAD_GB RAM tier is larger than the {free:.1f} of {size:.1f} GiB free in /dev/shm right now. "
                      f"The running slug's own tier is released when it stops; anything else in "
                      f"/dev/shm is not, and vLLM won't boot without the room")
    return None, None


def _fix_hint(slug: str, knob: str, source: str) -> str:
    if source == SHELL:
        return f"unset {knob} in your shell (or export a valid value)"
    if source == SLUG:
        return f"bash scripts/switch.sh --set {slug} {knob}=<value>   (or: --unset {slug} {knob})"
    if source == club_config.SECRETS_FILE:
        return f"python3 scripts/lib/club_config.py unset --file secrets {knob}"
    if source == club_config.LEGACY_LABEL:
        return f"python3 scripts/lib/club_config.py unset --root . {knob}   (it is in this checkout's .env)"
    if source == club_config.GLOBAL_FILE:
        return f"bash scripts/settings.sh set {knob}=<value>   (or: unset {knob})"
    return ""


def describe_source(s: Setting) -> str:
    return f"{s.source}: {s.layers[0].detail}" if s.layers and s.layers[0].detail else s.source


def shown(s: "Setting") -> str:
    """A setting's value as it may be printed in a message (secrets hidden)."""
    return display_value(s.knob, s.source, s.value)


def display_value(knob: str, source: str, value: str | None) -> str:
    if value is None:
        return "(varies)"
    if value == "":
        return "(unset)"
    if club_config.is_secret(knob, source):
        return "<set, hidden>"
    return value


def _refuse(res: Resolution, msg: str, *knobs: str) -> None:
    """A refusal of the next launch; also filed under each knob it is about."""
    res.errors.append(msg)
    for k in knobs:
        if k in res.settings:
            res.settings[k].errors.append(msg)


# ── the resolver ──────────────────────────────────────────────────────────────
def resolve(slug: str, root=ROOT, environ=None, loaded=None, force=False, catalogue=None) -> Resolution:
    """Every catalogued knob `slug` reads → its effective value and source, plus
    what the next launch of it would refuse or warn about. See the module doc."""
    env = os.environ if environ is None else environ
    root = Path(root)
    try:
        cat = catalogue or lk.load_catalogue()
    except lk.CatalogueError as exc:
        raise ResolveError(str(exc)) from None
    entry, kind, text, uses = facts(slug, root, cat)
    model = entry["model"]
    res = Resolution(slug=slug, model=model, engine_kind=kind, compose_path=entry["compose_path"],
                     store_path=slug_settings.store_path(env), uses=uses)
    files = file_values(root, env)
    shell = shell_keys(env, loaded, files)

    def origin(key: str) -> str | None:
        if key in shell:
            return SHELL
        return files[key][0] if key in files else None

    def current(key: str) -> str | None:
        return env[key] if key in env else (files[key][1] if key in files else None)

    saved: dict[str, str] = {}
    try:
        store = slug_settings.read(env)
        saved = store.slugs.get(slug, {})
        res.warnings += store.warnings
    except slug_settings.StoreError as exc:
        res.errors.append(f"per-slug settings can't be read: {exc}. Launching without them would boot "
                          "a config you did not save.")

    consumed = sorted(n for n, u in uses.items() if u.consumed)
    pin_key = cr.model_thinking_pin_key(model)
    pin_raw = current(pin_key)
    pin_state = (pin_raw or "").strip().lower()
    if pin_raw is not None and pin_state not in ("on", "off", "inherit", ""):
        res.warnings.append(f"{pin_key}={pin_raw} (from {origin(pin_key)}) is not on|off|inherit — ignored")

    for k in consumed:
        knob = cat["knobs"][k]
        layers: list[Layer] = []
        if k in shell:
            layers.append(Layer(SHELL, env[k]))
        if k in saved:
            layers.append(Layer(SLUG, saved[k]))
        if k == "ENABLE_THINKING" and pin_state in PIN_VALUES:
            layers.append(Layer(PIN, PIN_VALUES[pin_state], f"{pin_key}={pin_raw} from {origin(pin_key)}"))
        if k in files:
            layers.append(Layer(files[k][0], files[k][1]))
        vs = lk.variant_for(knob, kind, model, text)
        variant = vs[0] if len(vs) == 1 else None
        res.settings[k] = Setting(k, layers, uses[k].compose_default(), variant, knob.get("unset", ""),
                                  lk.domain_text(knob, variant))

    # Saved values this slug doesn't read: they do nothing (#1465 gotcha 4).
    for k, v in sorted(saved.items()):
        if k not in cat["knobs"]:
            res.warnings.append(f"{res.store_path} has {k} for {slug}, which is not a catalogued launch setting — "
                                f"ignored (remove it: bash scripts/switch.sh --unset {slug} {k})")
        elif k not in res.settings:
            res.unread.append((k, SLUG, v))
    for k, (label, v) in sorted(files.items()):
        if k in cat["knobs"] and k not in res.settings:
            res.unread.append((k, label, v))
    if pin_state in PIN_VALUES and "ENABLE_THINKING" not in res.settings:
        res.unread.append(("ENABLE_THINKING", f"{PIN} {pin_key}={pin_raw} from {origin(pin_key)}",
                           PIN_VALUES[pin_state]))

    # Values: each one a layer supplied, in the slug's catalogued domain.
    bad_value: set[str] = set()
    for k, s in res.settings.items():
        if s.source == DEFAULT:
            continue
        if s.value == "" and s.source != SLUG:
            continue                                   # empty == unset (module doc)
        if s.variant is None:
            n = len(lk.variant_for(cat["knobs"][k], kind, model, text))
            res.warnings.append(f"{k}={shown(s)} (from {describe_source(s)}): the catalogue has "
                                f"{'no value domain' if n == 0 else f'{n} value domains'} for {slug} "
                                f"({kind}, {model}) — not validated")
            continue
        why = lk.value_error(cat["knobs"][k], s.variant, s.value)
        if why is None:
            continue
        if shown(s) != s.value:                       # a secrets.env value: never echo it
            why = "not in the slug's catalogued domain"
        bad_value.add(k)
        enforced = s.variant.get("enforced", "unverified")
        msg = f"{k}={shown(s)} (from {describe_source(s)}): {why}; {ENFORCED_TEXT.get(enforced, enforced)}."
        hint = _fix_hint(slug, k, s.source)
        if force and enforced in ("unverified", "none"):
            res.warnings.append(msg + " Launching anyway (--force).")
        else:
            _refuse(res, msg + (f" Fix: {hint}" if hint else ""), k)

    # Dependencies between knobs, on the effective values (compose defaults included).
    eff = {k: (None if s.is_unset else s.value) for k, s in res.settings.items()}
    eff_shown = {k: (None if s.is_unset else shown(s)) for k, s in res.settings.items()}

    def said(k: str) -> str:
        s = res.settings.get(k)
        if s is None:
            return f"{k} is not read by {slug}"
        if s.is_unset:
            return f"{k} unset" + (f" (from {describe_source(s)})" if s.source != DEFAULT else "")
        return f"{k}={shown(s)} from {describe_source(s)}"

    for name, rule in lk.broken_requires(cat, eff):
        other = rule["knob"]
        if res.settings[name].source == DEFAULT and (other not in res.settings
                                                     or res.settings[other].source == DEFAULT):
            continue                                   # never refuse a compose default
        _refuse(res, f"{lk.requirement_text(name, rule, eff_shown)} ({said(name)}; {said(other)}).", name, other)

    # Host RAM for the RAM tier, then (vLLM) the host /dev/shm that holds it.
    s = res.settings.get("KV_OFFLOAD_GB")
    if s is not None and s.source != DEFAULT and not s.is_unset and "KV_OFFLOAD_GB" not in bad_value:
        for check_fn, hidden in ((ram_error, "more host RAM than this host has"),
                                 (shm_error, "more /dev/shm than this host has")):
            refusal, warning = check_fn(s.value, kind, env)
            if refusal and shown(s) != s.value:
                refusal = hidden                                  # a secrets.env value: never echo it
            if refusal:
                hint = _fix_hint(slug, "KV_OFFLOAD_GB", s.source)
                _refuse(res, f"KV_OFFLOAD_GB={shown(s)} (from {describe_source(s)}): {refusal}."
                        + (f" Fix: {hint}" if hint else ""), "KV_OFFLOAD_GB")
            if warning:
                res.warnings.append(warning)
            if refusal:
                break
    return res


# ── output ────────────────────────────────────────────────────────────────────
def unread_lines(res: Resolution) -> list[str]:
    return [f"saved {k}={display_value(k, src, v)} ({src}) is not read by {res.slug} — it has no effect there"
            for k, src, v in res.unread]


def to_json(res: Resolution) -> dict:
    return {
        "available": True,
        "order": list(ORDER),
        "store": str(res.store_path),
        "engine_kind": res.engine_kind,
        "knobs": [{
            "knob": k,
            "value": display_value(k, s.source, s.value),
            "source": s.source,
            "detail": s.layers[0].detail if s.layers else "",
            "overrides": [{"source": ly.source, "value": display_value(k, ly.source, ly.value)}
                          for ly in s.layers[1:]],
            "compose_default": s.default,
            "unset_means": s.unset_text,
            "enforced": (s.variant or {}).get("enforced"),
            "allowed": s.allowed,
            "errors": s.errors,
        } for k, s in sorted(res.settings.items())],
        "unread": [{"knob": k, "source": src, "value": display_value(k, src, v)} for k, src, v in res.unread],
        "errors": res.errors,
        "warnings": res.warnings,
    }


def _parse_loaded(items) -> dict[str, str]:
    out = {}
    for it in items or []:
        k, sep, label = it.partition("=")
        if sep:
            out[k] = label
    return out


def _pairs(items) -> dict[str, str]:
    out = {}
    for p in items:
        if "=" not in p:
            raise ResolveError(f"expected KEY=VALUE, got {p!r}")
        k, _, v = p.partition("=")
        out[k] = v
    return out


def cmd_check(a, env, loaded) -> int:
    res = resolve(a.slug, a.root, env, loaded, force=a.force)
    for w in res.warnings + unread_lines(res):
        print(f"{a.prefix} WARN: {w}", file=sys.stderr)
    if res.errors:
        print(f"{a.prefix} ERROR: launch settings for {a.slug} refused — nothing was torn down:", file=sys.stderr)
        for e in res.errors:
            print(f"{a.prefix}   - {e}", file=sys.stderr)
        print(f"{a.prefix}   Precedence: {' > '.join(ORDER)}.  See: bash scripts/switch.sh --explain {a.slug}",
              file=sys.stderr)
        return 1
    return 0


def cmd_exports(a, env, loaded) -> int:
    """NUL-separated records KEY, VALUE, LOG-LINE for switch.sh: every value a layer
    other than the shell supplies (the shell's own are already in the environment,
    and a compose default must stay unset). Shell values get a log line and an
    empty KEY so switch.sh prints them without exporting."""
    res = resolve(a.slug, a.root, env, loaded, force=True)
    out = sys.stdout.buffer
    for k, s in sorted(res.settings.items()):
        if s.source == DEFAULT:
            continue
        shown = display_value(k, s.source, s.value)
        beats = "".join(f"; overrides {ly.source}={display_value(k, ly.source, ly.value)}" for ly in s.layers[1:])
        line = f"launch setting {k}={shown}  ({describe_source(s)}{beats})"
        key = "" if s.source == SHELL else k
        for field_ in (key, s.value or "", line):
            out.write(field_.encode("utf-8") + b"\0")
    return 0


def _effective_lines(res: Resolution, keys) -> list[str]:
    out = []
    for k in keys:
        s = res.settings.get(k)
        if s is None:
            continue
        out.append(f"{k} on the next launch of {res.slug}: {display_value(k, s.source, s.value)} ({describe_source(s)})")
        if s.source == SHELL:
            out.append(f"  (your shell exports {k}, which wins over saved values for launches from this shell)")
    return out


def _catalogue() -> dict:
    try:
        return lk.load_catalogue()
    except lk.CatalogueError as exc:
        raise ResolveError(str(exc)) from None


@dataclass
class Change:
    """What saving or removing per-slug values did. ``problems`` non-empty means it
    was refused and nothing was written. switch.sh --set/--unset print it (cmd_set,
    cmd_unset); c3's launch-settings form shows the same words."""
    slug: str
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)     # keys saved / removed
    path: Path | None = None


def save_values(slug: str, vals: dict[str, str], root=ROOT, environ=None) -> Change:
    """Save KEY=VALUE pairs for one slug, each checked against the slug's catalogued
    domain, what its compose reads, the store's rules and the host-RAM rule. One bad
    pair saves nothing. Raises ResolveError (unknown slug, unusable catalogue),
    slug_settings.StoreError (a store it must not rewrite) or OSError (the write)."""
    env = os.environ if environ is None else environ
    ch = Change(slug)
    cat = _catalogue()
    entry, kind, text, uses = facts(slug, root, cat)
    consumed = sorted(n for n, u in uses.items() if u.consumed)
    for k, v in vals.items():
        if k not in cat["knobs"]:
            ch.problems.append(f"{k} is not a launch setting club-3090 knows (catalogued: {', '.join(lk.knob_names(cat))})")
            continue
        if k not in consumed:
            ch.problems.append(f"{slug} doesn't read {k} — its compose reads "
                               f"{', '.join(consumed) or 'no catalogued launch settings'}; a value saved for it would do nothing")
            continue
        try:
            slug_settings.check_entry(k, v)
        except slug_settings.StoreError as exc:
            ch.problems.append(str(exc))
            continue
        vs = lk.variant_for(cat["knobs"][k], kind, entry["model"], text)
        if len(vs) == 1:
            why = lk.value_error(cat["knobs"][k], vs[0], v)
            if why:
                enforced = vs[0].get("enforced", "unverified")
                ch.problems.append(f"{k}={v}: {why} (on {slug}: {ENFORCED_TEXT.get(enforced, enforced)})")
                continue
        else:
            ch.warnings.append(f"the catalogue has no single value domain for {k} on {slug} "
                               f"({kind}, {entry['model']}) — saved unchecked")
        if k == "KV_OFFLOAD_GB":
            for check_fn in (ram_error, shm_error):
                refusal, warning = check_fn(v, kind, env)
                if refusal:
                    ch.problems.append(f"KV_OFFLOAD_GB={v}: {refusal}")
                if warning:
                    ch.warnings.append(warning)
                if refusal:
                    break
    if not ch.problems:
        ch.path = slug_settings.set_values(slug, vals, env)
        ch.changed = list(vals)
    return ch


def remove_values(slug: str, keys, root=ROOT, environ=None) -> Change:
    """Remove keys saved for one slug. A saved key always goes, even one the slug no
    longer reads (the cleanup path for that warning); a typo, or a knob the slug
    doesn't read and nothing saved, is refused. Raises slug_settings.StoreError or
    OSError like save_values."""
    env = os.environ if environ is None else environ
    ch = Change(slug)
    keys = list(keys)
    store = slug_settings.read(env)                 # refuses a store it must not rewrite
    stored = store.slugs.get(slug, {})
    cat = _catalogue()
    consumed = None
    for k in keys:
        if k in stored:
            continue                                 # always removable, even a stale key
        if consumed is None:
            try:
                _, _, _, uses = facts(slug, root, cat)
            except ResolveError as exc:
                ch.problems.append(f"{exc} (and nothing is saved for it)")
                break
            consumed = {n for n, u in uses.items() if u.consumed}
        if k not in cat["knobs"]:
            ch.problems.append(f"{k} is not a launch setting club-3090 knows (catalogued: {', '.join(lk.knob_names(cat))})")
        elif k not in consumed:
            ch.problems.append(f"{slug} doesn't read {k}, and nothing is saved for it")
        else:
            ch.notes.append(f"{k} is not saved for {slug} — nothing to remove")
    if not ch.problems:
        ch.changed = slug_settings.unset_values(slug, [k for k in keys if k in stored], env) if stored else []
        ch.path = store.path
    return ch


def cmd_set(a, env, loaded) -> int:
    vals = _pairs(a.pairs)
    ch = save_values(a.slug, vals, a.root, env)
    for w in ch.warnings:
        print(f"{a.prefix} WARN: {w}", file=sys.stderr)
    if ch.problems:
        print(f"{a.prefix} ERROR: nothing saved for {a.slug}:", file=sys.stderr)
        for p in ch.problems:
            print(f"{a.prefix}   - {p}", file=sys.stderr)
        return 2
    print(f"{a.prefix} saved for {a.slug}: {', '.join(f'{k}={v}' for k, v in vals.items())}  ({ch.path})")
    print(f"{a.prefix} applies from the next launch of {a.slug}; a running container keeps the settings it started with.")
    res = resolve(a.slug, a.root, env, loaded)
    for line in _effective_lines(res, vals):
        print(f"{a.prefix} {line}")
    for e in res.errors:
        print(f"{a.prefix} WARN: the next launch of {a.slug} would be refused until this is fixed: {e}", file=sys.stderr)
    return 0


def cmd_unset(a, env, loaded) -> int:
    ch = remove_values(a.slug, a.keys, a.root, env)
    if ch.problems:
        print(f"{a.prefix} ERROR: nothing removed for {a.slug}:", file=sys.stderr)
        for p in ch.problems:
            print(f"{a.prefix}   - {p}", file=sys.stderr)
        return 2
    for n in ch.notes:
        print(f"{a.prefix} {n}")
    if ch.changed:
        print(f"{a.prefix} removed for {a.slug}: {', '.join(ch.changed)}  ({ch.path})")
        try:
            res = resolve(a.slug, a.root, env, loaded)
        except ResolveError:
            return 0                                 # a slug that left the registry: nothing to resolve
        for line in _effective_lines(res, ch.changed):
            print(f"{a.prefix} {line}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="launch_settings.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knob-names", help="print the catalogued launch knobs, one per line")
    for name, hlp in (("check", "refuse/warn for the next launch of a slug (exit 1 = refused)"),
                      ("exports", "NUL-separated KEY, VALUE, LINE records for switch.sh to export"),
                      ("explain", "the slug's launch settings as JSON"),
                      ("set", "save KEY=VALUE for a slug, checked against its catalogued domains"),
                      ("unset", "remove keys saved for a slug")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--root", default=str(ROOT))
        p.add_argument("--loaded", action="append", default=[], metavar="KEY=FILE",
                       help="a key the settings loader exported, and its file (switch.sh passes "
                            "CLUB3090_CONFIG_SOURCE); any other environment key counts as the shell")
        p.add_argument("--slug", required=True)
        p.add_argument("--prefix", default="[launch-settings]")
        if name == "check":
            p.add_argument("--force", action="store_true")
        if name == "set":
            p.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
        if name == "unset":
            p.add_argument("keys", nargs="+", metavar="KEY")
    a = ap.parse_args(argv)
    if a.cmd == "knob-names":
        try:
            print("\n".join(lk.knob_names(lk.load_catalogue())))
        except lk.CatalogueError as exc:
            print(f"[launch-settings] ERROR: {exc}", file=sys.stderr)
            return 2
        return 0
    env, loaded = os.environ, _parse_loaded(a.loaded)
    if a.cmd in ("check", "exports"):
        # A slug this resolver can't read (a local compose the knob scan refuses, a
        # broken catalogue) launched before #1465 and still does: say so, apply nothing.
        try:
            return cmd_check(a, env, loaded) if a.cmd == "check" else cmd_exports(a, env, loaded)
        except ResolveError as exc:
            if a.cmd == "check":
                print(f"{a.prefix} WARN: launch settings not resolved for {a.slug}, none applied: {exc}",
                      file=sys.stderr)
            return 0
    try:
        if a.cmd == "explain":
            try:
                doc = to_json(resolve(a.slug, a.root, env, loaded))
            except ResolveError as exc:
                doc = {"available": False, "reason": str(exc)}
            print(json.dumps(doc, indent=2, ensure_ascii=False))
            return 0
        if a.cmd == "set":
            return cmd_set(a, env, loaded)
        if a.cmd == "unset":
            return cmd_unset(a, env, loaded)
    except (ResolveError, slug_settings.StoreError) as exc:
        print(f"{a.prefix} ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"{a.prefix} ERROR: cannot write {slug_settings.store_path(env)}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sys.exit(main())
