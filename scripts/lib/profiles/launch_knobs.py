#!/usr/bin/env python3
"""launch_knobs — the launch-knob catalogue and a static per-compose scan (#1465, phase 3a).

Two halves, both STDLIB-ONLY (no PyYAML: the launcher path runs on a bare python3,
#584; this module is imported by the launcher-facing registry emit and by the #1465
resolver, scripts/lib/launch_settings.py):

1. ``launch-knobs.json`` (next to this file) DESCRIBES each launch setting a user
   may persist: its value domain, which engines read it, how it depends on other
   settings, and why. Hand-written; ``load_catalogue`` checks its shape.

2. ``scan_compose`` answers "which catalogued knobs does THIS compose read?" by
   scanning the compose TEXT. The compose is the source of truth for consumption
   (issue #1465 gotcha 4: a value set for a slug that doesn't read it does nothing,
   silently), so consumption is derived, never declared.

What counts as "reads" (``KnobUse.consumed``) — the two ways docker can deliver a
host value into a container, and nothing else (#1465 gotcha 5):

  * the knob is declared under a service's ``environment:`` so its host value is
    forwarded — ``- NAME``, ``- NAME=${NAME...}``, ``NAME:`` or ``NAME: ${NAME...}``;
  * the knob is interpolated by compose — ``${NAME}``, ``${NAME:-x}``, ``$NAME`` …
    (an ODD run of ``$``) — anywhere outside that environment entry.

A container-side read (``$${NAME}`` / ``$$NAME``, an EVEN run of ``$``) is recorded
but does NOT count: it only sees the value if the knob is also declared under
``environment:``. A compose that reads ``$${NAME}`` without declaring it is a dead
setting, which ``KnobUse.dead_read`` reports.

Comments are ignored: whole-line ``#`` comments (YAML ones, and shell ones inside a
``|`` block, which are dead code either way) and trailing `` #`` comments outside
quotes. Many composes mention knobs in comments only; counting those would claim
knobs the slug never reads.

This scan is a static claim. ``scripts/tests/test-launch-knobs.sh`` checks every
claim against ``docker compose config`` — the knob set to a sentinel must show up in
the rendered environment/command/entrypoint, and an unclaimed knob's sentinel must
not — so a scan bug cannot go green on a dead setting.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATALOGUE_PATH = HERE / "launch-knobs.json"
SCHEMA_VERSION = 1

TYPES = ("enum", "bool", "int", "size_gb", "string")
STATUSES = ("first-batch", "candidate")
# Where a value outside the domain is caught, if anywhere:
#   boot    — the compose's own entrypoint (or JSON splice) refuses it: the container exits.
#   request — the container boots; the failure surfaces per request (e.g. a chat
#             template raise_exception), so a bad default breaks every request that
#             doesn't send its own value.
#   none    — nothing refuses it; the engine ignores or misreads it.
#   unverified — the compose passes it through unchecked and the repo records no
#             evidence of what the engine/template then does. Say so rather than guess.
ENFORCED = ("boot", "request", "none", "unverified")
CHECK_KINDS = ("shell", "json_line", "passthrough")

# Implied full-match patterns for the numeric types (a variant's own `pattern`
# replaces them). These mirror the composes' shell `case` checks — see the
# catalogue's per-variant `source` for the exact line each was taken from.
_TYPE_PATTERN = {
    "int": r"[0-9]+",
    "size_gb": r"[0-9]+(?:\.[0-9]+)?",
}

_NAME_RX = r"[A-Za-z_][A-Za-z0-9_]*"


class CatalogueError(ValueError):
    """The catalogue file is malformed. The message names the knob and the field."""


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def load_catalogue(path: Path | str | None = None) -> dict:
    """Load and shape-check the catalogue. Raises CatalogueError on any problem."""
    p = Path(path) if path else CATALOGUE_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CatalogueError(f"{p}: cannot read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogueError(f"{p}: not valid JSON: {exc}") from exc
    errors = catalogue_errors(data)
    if errors:
        raise CatalogueError(f"{p}: " + "; ".join(errors))
    return data


def knob_names(catalogue: dict) -> list[str]:
    return sorted(catalogue["knobs"])


def _source_errors(where: str, src) -> list[str]:
    if not isinstance(src, dict) or not isinstance(src.get("file"), str) or not src.get("file"):
        return [f"{where}: source must be an object with a non-empty 'file'"]
    if not isinstance(src.get("what"), str) or not src["what"].strip():
        return [f"{where}: source needs a 'what' saying what that file does"]
    return []


def catalogue_errors(data) -> list[str]:
    """Every shape problem in the catalogue (empty list == valid).

    The schema is documented in launch-knobs.md; this is its enforcement.
    """
    errs: list[str] = []
    if not isinstance(data, dict):
        return ["top level must be a JSON object"]
    if data.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION}")
    knobs = data.get("knobs")
    if not isinstance(knobs, dict) or not knobs:
        return errs + ["'knobs' must be a non-empty object"]
    for name, k in knobs.items():
        w = f"knobs.{name}"
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            errs.append(f"{w}: env name must be UPPER_SNAKE")
        if not isinstance(k, dict):
            errs.append(f"{w}: must be an object")
            continue
        for fld in ("description", "unset"):
            if not isinstance(k.get(fld), str) or not k[fld].strip():
                errs.append(f"{w}.{fld}: required non-empty string")
        if isinstance(k.get("description"), str) and "\n" in k["description"]:
            errs.append(f"{w}.description: must be one line")
        if k.get("type") not in TYPES:
            errs.append(f"{w}.type: must be one of {TYPES}")
        if k.get("status") not in STATUSES:
            errs.append(f"{w}.status: must be one of {STATUSES}")
        engines = k.get("engines")
        if not isinstance(engines, list) or not engines or not all(isinstance(e, str) for e in engines):
            errs.append(f"{w}.engines: required non-empty list of engine kinds")
        variants = k.get("variants")
        if not isinstance(variants, list) or not variants:
            errs.append(f"{w}.variants: required non-empty list")
            variants = []
        for i, v in enumerate(variants):
            errs += _variant_errors(f"{w}.variants[{i}]", k, v)
        for i, r in enumerate(k.get("requires", []) or []):
            rw = f"{w}.requires[{i}]"
            if not isinstance(r, dict):
                errs.append(f"{rw}: must be an object")
                continue
            if r.get("knob") not in knobs:
                errs.append(f"{rw}.knob: {r.get('knob')!r} is not a catalogued knob")
            if not (r.get("when") == "set" or isinstance(r.get("when"), dict) and "equals" in r["when"]):
                errs.append(f"{rw}.when: must be \"set\" or {{\"equals\": <value>}}")
            if not (r.get("then") == "set" or isinstance(r.get("then"), dict) and "equals" in r["then"]):
                errs.append(f"{rw}.then: must be \"set\" or {{\"equals\": <value>}}")
            errs += _source_errors(rw, r.get("source"))
        for i, c in enumerate(k.get("caveats", []) or []):
            cw = f"{w}.caveats[{i}]"
            if not isinstance(c, dict) or not isinstance(c.get("text"), str) or not c["text"].strip():
                errs.append(f"{cw}: must be an object with a non-empty 'text'")
                continue
            errs += _source_errors(cw, c.get("source"))
        for i, it in enumerate(k.get("interacts", []) or []):
            iw = f"{w}.interacts[{i}]"
            if not isinstance(it, dict) or not isinstance(it.get("knob"), str) or not isinstance(it.get("effect"), str):
                errs.append(f"{iw}: must be an object with 'knob' and 'effect'")
                continue
            errs += _source_errors(iw, it.get("source"))
    return errs


def _variant_errors(w: str, knob: dict, v) -> list[str]:
    errs: list[str] = []
    if not isinstance(v, dict):
        return [f"{w}: must be an object"]
    m = v.get("match")
    if not isinstance(m, dict):
        errs.append(f"{w}.match: required object ({{}} matches every consumer)")
    else:
        for key in ("engine", "model"):
            if key in m and (not isinstance(m[key], list) or not m[key]):
                errs.append(f"{w}.match.{key}: must be a non-empty list")
        if "compose_contains" in m and not isinstance(m["compose_contains"], str):
            errs.append(f"{w}.match.compose_contains: must be a string")
        extra = set(m) - {"engine", "model", "compose_contains"}
        if extra:
            errs.append(f"{w}.match: unknown keys {sorted(extra)}")
    if v.get("enforced") not in ENFORCED:
        errs.append(f"{w}.enforced: must be one of {ENFORCED}")
    t = knob.get("type")
    vals = v.get("values")
    if t in ("enum", "bool"):
        if not isinstance(vals, list) or not vals or not all(isinstance(x, str) for x in vals):
            errs.append(f"{w}.values: an {t} knob needs a non-empty list of accepted strings")
        if t == "bool":
            sp = v.get("spelling")
            if not (isinstance(sp, dict) and set(sp) == {"on", "off"}
                    and isinstance(vals, list) and sp["on"] in vals and sp["off"] in vals):
                errs.append(f"{w}.spelling: a bool knob needs {{\"on\": v, \"off\": v}} drawn from values")
    elif vals is not None:
        errs.append(f"{w}.values: only enum/bool knobs take a values list")
    if "pattern" in v:
        try:
            re.compile(v["pattern"])
        except (re.error, TypeError) as exc:
            errs.append(f"{w}.pattern: not a valid regex: {exc}")
    for b in ("min", "max", "above"):
        if b in v and (isinstance(v[b], bool) or not isinstance(v[b], (int, float))):
            errs.append(f"{w}.{b}: must be a number")
    if any(b in v for b in ("min", "max", "above")) and t not in ("int", "size_gb"):
        errs.append(f"{w}: min/max/above only apply to int and size_gb knobs")
    ecc = v.get("every_consumer_contains", [])
    if not isinstance(ecc, list) or not all(isinstance(x, str) and x for x in ecc):
        errs.append(f"{w}.every_consumer_contains: must be a list of literal strings")
    aliases = v.get("aliases", {})
    if not isinstance(aliases, dict) or not all(isinstance(a, str) and isinstance(c, str) for a, c in aliases.items()):
        errs.append(f"{w}.aliases: must map accepted alias -> canonical value")
    elif isinstance(vals, list):
        for a, c in aliases.items():
            if a not in vals or c not in vals:
                errs.append(f"{w}.aliases: {a!r} -> {c!r} must both be in values")
    if "default" in v and not (v["default"] is None or isinstance(v["default"], str)):
        errs.append(f"{w}.default: must be a string, null (unset), or omitted (varies per compose)")
    errs += _source_errors(w, v.get("source"))
    for ev in v.get("evidence", []) or []:
        if not (isinstance(ev, dict) and isinstance(ev.get("file"), str) and isinstance(ev.get("contains"), str)):
            errs.append(f"{w}.evidence: entries need 'file' and 'contains'")
    checks = v.get("checks")
    if not isinstance(checks, list) or not checks:
        errs.append(f"{w}.checks: required non-empty list — how the guard proves this domain "
                    "against at least one real compose")
        return errs
    for ci, chk in enumerate(checks):
        cw = f"{w}.checks[{ci}]"
        if not isinstance(chk, dict):
            errs.append(f"{cw}: must be an object")
            continue
        if chk.get("kind") not in CHECK_KINDS:
            errs.append(f"{cw}.kind: must be one of {CHECK_KINDS}")
        if not isinstance(chk.get("compose"), str) or not chk["compose"].endswith(".yml"):
            errs.append(f"{cw}.compose: must name a real compose (.yml) path")
        if chk.get("kind") == "shell":
            for fld in ("from", "to"):
                if not isinstance(chk.get(fld), str):
                    errs.append(f"{cw}.{fld}: a shell check needs '{fld}' (line regex)")
            if "to_plus" in chk and not isinstance(chk["to_plus"], int):
                errs.append(f"{cw}.to_plus: must be an integer")
        if chk.get("kind") in ("json_line", "passthrough") and not isinstance(chk.get("line"), str):
            errs.append(f"{cw}.line: a {chk.get('kind')} check needs 'line' (line regex)")
        if "env" in chk and not (isinstance(chk["env"], dict)
                                 and all(isinstance(x, str) for x in chk["env"].values())):
            errs.append(f"{cw}.env: must map NAME -> string")
        for lst in ("accept", "reject", "stricter"):
            probes = chk.get(lst)
            if probes is None and lst == "stricter":
                continue
            if not isinstance(probes, list) or (lst != "stricter" and not probes):
                errs.append(f"{cw}.{lst}: required non-empty list of probes")
                continue
            for p in probes:
                if not (isinstance(p, str) or isinstance(p, dict)
                        and all(isinstance(x, str) for x in p.values())):
                    errs.append(f"{cw}.{lst}: probe {p!r} must be a value or a NAME -> value map")
        if chk.get("stricter") and v.get("enforced") != "boot":
            errs.append(f"{cw}.stricter: only meaningful when enforced=boot (otherwise use reject)")
    return errs


# ---------------------------------------------------------------------------
# Values: validation against a variant, dependency rules
# ---------------------------------------------------------------------------

def variant_for(knob: dict, engine_kind: str, model: str, compose_text: str = "") -> list[dict]:
    """The variants whose `match` covers (engine_kind, model, compose). A well-formed
    catalogue yields exactly one for every consuming slug; the guard enforces that.
    The engine KIND must come from scripts/lib/engine-kind.sh — never classify here."""
    out = []
    for v in knob["variants"]:
        m = v["match"]
        if "engine" in m and engine_kind not in m["engine"]:
            continue
        if "model" in m and model not in m["model"]:
            continue
        if "compose_contains" in m and m["compose_contains"] not in compose_text:
            continue
        out.append(v)
    return out


def value_error(knob: dict, variant: dict, value: str) -> str | None:
    """None when `value` is in the variant's domain, else the reason it is not.

    An empty string is never a value: "not set" is expressed by leaving the knob
    unset (the compose default then applies), never by setting it to ""."""
    if not isinstance(value, str):
        return f"value must be a string, got {type(value).__name__}"
    if value == "":
        return "empty — unset the knob instead of setting it to the empty string"
    t = knob["type"]
    vals = variant.get("values")
    pat = variant.get("pattern") or _TYPE_PATTERN.get(t)
    in_vals = isinstance(vals, list) and value in vals
    in_pat = bool(pat) and re.fullmatch(pat, value) is not None
    if not (in_vals or in_pat):
        return f"{value!r} is not one of: {_values_text(knob, variant)}"
    if in_pat and not in_vals and any(b in variant for b in ("min", "max", "above")):
        num = float(value)
        if "above" in variant and not num > variant["above"]:
            return f"{value} must be greater than {variant['above']:g}"
        if "min" in variant and num < variant["min"]:
            return f"{value} is below the minimum {variant['min']:g}"
        if "max" in variant and num > variant["max"]:
            return f"{value} is above the maximum {variant['max']:g}"
    return None


def _values_text(knob: dict, variant: dict) -> str:
    """The accepted spellings, as value_error names them: the listed values, the
    variant's own /pattern/, or the numeric type in words."""
    vals = variant.get("values")
    pat = variant.get("pattern") or _TYPE_PATTERN.get(knob["type"])
    allowed = " | ".join(vals) if vals else ""
    if pat and variant.get("pattern"):
        allowed = (allowed + " | " if allowed else "") + f"/{pat}/"
    elif pat:
        allowed = allowed or ("a whole number" if knob["type"] == "int" else "a number of GiB, e.g. 64 or 0.5")
    return allowed


def domain_text(knob: dict, variant: dict | None) -> str | None:
    """A variant's whole domain in words, for display (c3's launch-settings form,
    the resolver's JSON): the spellings value_error accepts, its aliases and its
    numeric bounds. None without a single variant. Display only — value_error is
    the check."""
    if variant is None:
        return None
    text = _values_text(knob, variant)
    aliases = variant.get("aliases") or {}
    if aliases:
        text += " (" + ", ".join(f"{a} = {c}" for a, c in aliases.items()) + ")"
    bounds = [f"{word} {variant[b]:g}" for b, word in (("above", "greater than"), ("min", "at least"),
                                                          ("max", "at most")) if b in variant]
    return text + (f"; {', '.join(bounds)}" if bounds else "")


def _cond_holds(cond, value: str | None) -> bool:
    if cond == "set":
        return value not in (None, "")
    return value == cond["equals"]


def broken_requires(catalogue: dict, values: dict[str, str | None]) -> list[tuple[str, dict]]:
    """(knob, rule) for every `requires` rule the effective values break (None/"" = unset).
    The resolver (scripts/lib/launch_settings.py) uses the pair to say where each
    side's value came from; dependency_errors() words the same list."""
    out = []
    for name, k in catalogue["knobs"].items():
        for r in k.get("requires", []) or []:
            if _cond_holds(r["when"], values.get(name)) and not _cond_holds(r["then"], values.get(r["knob"])):
                out.append((name, r))
    return out


def requirement_text(name: str, rule: dict, values: dict[str, str | None]) -> str:
    want = " set" if rule["then"] == "set" else f"={rule['then']['equals']}"
    return f"{name}={values.get(name)} needs {rule['knob']}{want}: {rule.get('why', '')}".rstrip(": ")


def dependency_errors(catalogue: dict, values: dict[str, str | None]) -> list[str]:
    """Broken `requires` rules for a set of effective knob values (None/"" = unset)."""
    return [requirement_text(name, r, values) for name, r in broken_requires(catalogue, values)]


# ---------------------------------------------------------------------------
# Static compose scan
# ---------------------------------------------------------------------------

@dataclass
class KnobUse:
    name: str
    # One entry per environment: declaration of NAME, as (line_no, form).
    #   bare           - NAME  /  NAME:        host value forwarded when set, absent otherwise
    #   interp         - NAME=${NAME}          forwarded; unset -> compose warns, sets ""
    #   empty-default  - NAME=${NAME:-}        forwarded; unset -> EMPTY STRING, not absent
    #   default        - NAME=${NAME:-x}       forwarded; unset -> x
    #   pinned         - NAME=<no ${NAME}>     a constant: the host value never arrives
    env: list[tuple[int, str]] = field(default_factory=list)
    interp: list[int] = field(default_factory=list)   # compose-side ${NAME...} outside its own env entry
    shell: list[int] = field(default_factory=list)    # container-side $${NAME...} / $$NAME
    assigned: list[int] = field(default_factory=list)  # the container script assigns NAME itself
    defaults: list[str] = field(default_factory=list)  # the `:-x` / `-x` defaults it is read with

    @property
    def forwarded(self) -> bool:
        return any(form != "pinned" for _, form in self.env)

    @property
    def consumed(self) -> bool:
        """The host value can reach the container (the emit's `knobs` field)."""
        return self.forwarded or bool(self.interp)

    @property
    def dead_read(self) -> bool:
        """The container reads $${NAME} but the compose never forwards it."""
        return bool(self.shell) and not self.forwarded and not self.assigned

    @property
    def dead_env(self) -> bool:
        """Forwarded into the container but nothing in the compose reads it."""
        return self.forwarded and not self.shell and not self.interp

    def env_forms(self) -> list[str]:
        return [form for _, form in self.env]

    def compose_default(self) -> str | None:
        """The default the compose falls back to when the knob is unset.

        Collected from every `${NAME:-x}` / `$${NAME:-x}` read EXCEPT the
        `NAME=${NAME:-}` forwarding entry (an empty passthrough is read back with
        the script's own `:-x`, so the script's default is the effective one).
        The MOST COMMON default wins — a status message such as
        `echo "cap $${KV_OFFLOAD_DISK_GB:-none} GiB"` carries a display default that
        is not the effective one; `default_outliers()` lists those. "" means
        "unset = off/none"; None means no read has a default, or a tie.
        """
        if not self.defaults:
            return None
        counts: dict[str, int] = {}
        for d in self.defaults:
            counts[d] = counts.get(d, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    def default_outliers(self) -> list[str]:
        best = self.compose_default()
        return sorted({d for d in self.defaults if d != best})


def strip_comment(line: str) -> str:
    """The line with its comment removed ("" for a whole-line comment).

    A `#` starts a comment when it is the first non-blank character, or when it
    follows whitespace outside single/double quotes (YAML and shell agree on that
    much). Quote state is per line — a comment inside an unbalanced quote is kept,
    which can only ADD a claim, and the delivery check catches a false claim."""
    s = line.lstrip()
    if s.startswith("#"):
        return ""
    in_s = in_d = False
    prev = " "
    i = 0
    while i < len(line):
        ch = line[i]
        if in_d and ch == "\\":
            prev = ch
            i += 2
            continue
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d and prev in " \t":
            return line[:i].rstrip()
        prev = ch
        i += 1
    return line


_ENV_KEY_RX = re.compile(r"^(\s*)environment:\s*$")
# environment: written as a flow collection, or pulled in through a YAML alias or
# merge key — the text scan cannot see those entries, so it refuses rather than
# silently under-claiming. No shipped compose uses them.
_ENV_FLOW_RX = re.compile(r"^\s*environment:\s*[\[{*&]")
_MERGE_RX = re.compile(r"^\s*<<\s*:")
_LIST_ITEM_RX = re.compile(r"^\s*-\s+(['\"]?)(" + _NAME_RX + r")(?:=(.*?))?\1\s*$")
_MAP_ITEM_RX = re.compile(r"^\s*(['\"]?)(" + _NAME_RX + r")\1\s*:\s*(.*?)\s*$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _env_value_form(name: str, value: str | None) -> str:
    if value is None:
        return "bare"
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    if v in ("null", "~"):          # map form `NAME: null` forwards like `NAME:`
        return "bare"
    if v == "":                     # `- NAME=` / `NAME: ""`: a constant empty string
        return "pinned"
    if re.fullmatch(r"\$\{" + re.escape(name) + r"\}", v):
        return "interp"
    own = re.fullmatch(r"\$\{" + re.escape(name) + r"(:?-)(.*)\}", v, re.S)
    if own:
        return "empty-default" if own.group(2) == "" else "default"
    # NAME=<something that interpolates NAME among other text> still forwards it.
    if _interp_refs(name, v):
        return "default"
    return "pinned"


def _dollar_refs(name: str, text: str) -> list[tuple[int, str | None]]:
    """(dollar-run length, default-or-None) for every `$..{NAME` / `$..NAME` in text."""
    out = []
    rx = re.compile(r"(\$+)(\{)?" + re.escape(name) + r"(?![A-Za-z0-9_])")
    for m in rx.finditer(text):
        run, braced = len(m.group(1)), bool(m.group(2))
        default = None
        if braced:
            rest = text[m.end():]
            dm = re.match(r"(:?-)", rest)
            if dm:
                default = _balanced_default(rest[len(dm.group(1)):])
        out.append((run, default))
    return out


def _balanced_default(rest: str) -> str | None:
    """The default text of `${NAME:-<this>}` up to its matching `}` ($$ -> $)."""
    depth = 0
    for i, ch in enumerate(rest):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return rest[:i].replace("$$", "$")
            depth -= 1
    return None


def _interp_refs(name: str, text: str) -> list[str | None]:
    return [d for run, d in _dollar_refs(name, text) if run % 2 == 1]


def scan_compose_text(text: str, names) -> dict[str, KnobUse]:
    """Per catalogued name, how this compose text uses it (see KnobUse)."""
    names = list(names)
    uses = {n: KnobUse(n) for n in names}
    lines = text.splitlines()
    code = [strip_comment(ln) for ln in lines]

    # 1. environment: blocks. Record each entry's line so step 2 can skip a knob's
    #    OWN forwarding entry (its ${NAME:-} is the forwarding, not a second read).
    own_entry: dict[int, str] = {}      # line index -> the knob name it declares
    i = 0
    while i < len(code):
        ln = code[i]
        if _ENV_FLOW_RX.match(ln):
            raise ValueError(f"line {i + 1}: environment: as a flow collection or YAML alias is not "
                             "supported by this scan; use a block list (- NAME) like every shipped compose")
        m = _ENV_KEY_RX.match(ln)
        if not m:
            i += 1
            continue
        base = len(m.group(1))
        j = i + 1
        while j < len(code):
            item = code[j]
            if not item.strip():
                j += 1
                continue
            # A block sequence may sit at the key's own indent (`environment:` /
            # `- NAME` aligned) — still part of the block; anything else ends it.
            if _indent(item) < base or (_indent(item) == base and not item.lstrip().startswith("- ")):
                break
            if _MERGE_RX.match(item):
                raise ValueError(f"line {j + 1}: a YAML merge key (<<:) inside environment: is not "
                                 "supported by this scan; list the entries in the service")
            lm = _LIST_ITEM_RX.match(item)
            mm = None if lm else _MAP_ITEM_RX.match(item)
            if lm:
                nm, val = lm.group(2), lm.group(3)
            elif mm:
                nm, val = mm.group(2), (mm.group(3) or None)
            else:
                j += 1
                continue
            if nm in uses:
                form = _env_value_form(nm, val)
                uses[nm].env.append((j + 1, form))
                own_entry[j] = nm
                if form == "default":
                    d = _interp_refs(nm, val or "")
                    uses[nm].defaults += [x for x in d if x is not None]
            j += 1
        i = j

    # 2. every other reference, line by line (comment-stripped).
    for idx, ln in enumerate(code):
        if not ln.strip():
            continue
        for nm in names:
            if nm not in ln:
                continue
            u = uses[nm]
            for run, default in _dollar_refs(nm, ln):
                if run % 2 == 1:
                    if own_entry.get(idx) == nm:
                        continue            # its own forwarding entry, counted in step 1
                    u.interp.append(idx + 1)
                else:
                    u.shell.append(idx + 1)
                if default is not None:
                    u.defaults.append(default)
            # `NAME=...` as a shell assignment at a statement start — not the env entry
            # `- NAME=`, and not text inside quotes (`echo "...; SPEC_N=0 disables"`).
            if own_entry.get(idx) != nm and not re.match(r"^\s*-\s", ln) and re.search(
                r"(?:^|;|&&|\|\||\bthen|\bdo|\belse)\s*(?:export\s+|local\s+|readonly\s+)?"
                + re.escape(nm) + r"=", _QUOTED_RX.sub('""', ln)
            ):
                u.assigned.append(idx + 1)
    return uses


_QUOTED_RX = re.compile(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'')


def scan_compose(path: Path | str, names) -> dict[str, KnobUse]:
    return scan_compose_text(Path(path).read_text(encoding="utf-8"), names)


def consumed_knobs(path: Path | str, names) -> list[str]:
    """Sorted catalogued knobs whose host value this compose can deliver."""
    return sorted(n for n, u in scan_compose(path, names).items() if u.consumed)


class SlugKnobs:
    """Per-compose cache for the registry emit: compose_path -> consumed knobs."""

    def __init__(self, root: Path | str, catalogue: dict | None = None):
        self.root = Path(root)
        self.names = knob_names(catalogue or load_catalogue())
        self._cache: dict[str, list[str] | None] = {}

    def for_compose(self, compose_path: str) -> list[str] | None:
        """None when the compose cannot be read or scanned (unknown != none)."""
        if compose_path not in self._cache:
            try:
                self._cache[compose_path] = consumed_knobs(self.root / compose_path, self.names)
            except (OSError, ValueError) as exc:
                print(f"[launch-knobs] WARN: {compose_path}: {exc}", file=sys.stderr)
                self._cache[compose_path] = None
        return self._cache[compose_path]


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Launch-knob catalogue: check it, or scan a compose.")
    ap.add_argument("--check", action="store_true", help="validate launch-knobs.json and exit")
    ap.add_argument("--catalogue", help="catalogue path (default: the one next to this file)")
    ap.add_argument("compose", nargs="*", help="compose file(s) to scan")
    a = ap.parse_args(argv)
    try:
        cat = load_catalogue(a.catalogue)
    except CatalogueError as exc:
        print(f"[launch-knobs] {exc}", file=sys.stderr)
        return 1
    if a.check and not a.compose:
        print(f"[launch-knobs] ok: {len(cat['knobs'])} knobs")
        return 0
    rc = 0
    for c in a.compose:
        try:
            uses = scan_compose(c, knob_names(cat))
        except (OSError, ValueError) as exc:
            print(f"{c}: {exc}", file=sys.stderr)
            rc = 1
            continue
        for n, u in sorted(uses.items()):
            if not (u.env or u.interp or u.shell):
                continue
            flags = [f for f, on in (("consumed", u.consumed), ("DEAD-READ", u.dead_read),
                                     ("dead-env", u.dead_env)) if on]
            print(f"{c}\t{n}\tenv={','.join(u.env_forms()) or '-'}\tinterp={len(u.interp)}"
                  f"\tshell={len(u.shell)}\tdefault={u.compose_default()!r}\t{' '.join(flags)}")
    return rc


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raise SystemExit(_main(sys.argv[1:]))
