#!/usr/bin/env bash
# Shared compose-registry shell emitter for switch.sh and launch.sh.
#
# Source this file, declare the destination arrays in the caller, then call
# derive_switch_variant_tables or derive_launch_variant_tables with ROOT_DIR.

registry_variant_rows() {
  local root="$1"
  python3 - "$root" <<'PY_EMIT'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# PyYAML is OPTIONAL on this path (the switch.sh/launch.sh table derivation):
# community rigs run minimal VMs without python3-yaml (#584 — ryan's Proxmox
# box), and the ONLY thing this block used yaml for is pulling container_name
# out of each compose — which the regex fallback below handles for our own
# compose files. The `--json` contract path (c3 cockpit / baselines join)
# still requires PyYAML. CLUB3090_EMIT_NO_YAML=1 forces the fallback so the
# no-yaml guard test can exercise it on rigs where yaml IS installed.
try:
    import yaml
except Exception:
    yaml = None
if os.environ.get("CLUB3090_EMIT_NO_YAML") == "1":
    yaml = None

# Non-UTF-8 locales (LC_ALL=C VMs, #599/#584) also break the WRITE side: a
# piped stdout defaults to the locale codec → UnicodeEncodeError printing the
# unicode in status notes. Reads were fixed in #599; pin the output too.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY, DEFAULTS  # noqa: E402


def die(key: str, message: str) -> None:
    print(f"__ERR__\t{key}\t{message}")


def launch_engine(key: str) -> str:
    prefix = key.split("/", 1)[0]
    return "llamacpp" if prefix in {"llamacpp", "ik-llama"} else prefix


def switch_engine(key: str) -> str:
    prefix = key.split("/", 1)[0]
    return "llamacpp" if prefix == "llamacpp" else prefix


# First non-comment `container_name:` line — the regex fallback for rigs
# without PyYAML. Our compose files are single-service (or first-service-wins,
# matching the yaml path's dict-order iteration), so this is equivalent for
# every checked-in compose; test-registry-emit-no-yaml asserts that parity.
_CONTAINER_RX = re.compile(r"""^\s*container_name:\s*(['"]?)(.+?)\1\s*$""", re.M)


def _unwrap_env_default(raw: str) -> str:
    match = re.fullmatch(r"\$\{[^}:]+:-(.+)\}", raw)
    return match.group(1) if match else raw


def container_name(compose_path: str) -> str:
    path = root / compose_path
    if yaml is None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"could not read compose yaml: {exc}") from exc
        m = _CONTAINER_RX.search(text)
        return _unwrap_env_default(m.group(2).strip()) if m else ""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"could not parse compose yaml: {exc}") from exc
    services = data.get("services") or {}
    for service in services.values():
        raw = service.get("container_name")
        if not raw:
            continue
        return _unwrap_env_default(str(raw))
    return ""


_CTX_ENV = re.compile(r"\$\{(?:MAX_MODEL_LEN|CTX_SIZE|CTX|MAX_CTX|N_CTX|MODEL_LEN)\s*:-\s*(\d+)\s*\}")
_CTX_FLAG = re.compile(r"(?:--max-model-len|--ctx-size|--n-ctx|(?<!\w)-c)\s*\n?\s*\"?(\d{3,})")


def compose_default_ctx(compose_path: str):
    """The ctx the compose serves by DEFAULT (its ${VAR:-N} fallback or flag literal)."""
    try:
        txt = (root / compose_path).read_text(encoding="utf-8")
    except Exception:
        return None
    m = _CTX_ENV.search(txt) or _CTX_FLAG.search(txt)
    return int(m.group(1)) if m else None


def ctx_label(entry) -> str:
    """Compact ctx label rounded to K. Single 'NK' when the registry max_ctx matches
    the compose's default ctx; 'NK/MK' (validated registry / compose default) when
    they drift — so the list surfaces registry<->compose context mismatches."""
    reg = entry.get("max_ctx")
    if not reg:
        return ""
    reg = int(reg)
    comp = compose_default_ctx(entry["compose_path"])
    label = f"{round(reg / 1000)}K"
    if comp is not None and comp != reg:
        label += f"/{round(comp / 1000)}K"
    return label


for key, entry in COMPOSE_REGISTRY.items():
    cp = entry["compose_path"]
    if "/compose/" not in cp:
        die(key, f"compose_path lacks /compose/: {cp}")
        continue
    dirpart, filepart = cp.split("/compose/", 1)
    compose_dir = f"{dirpart}/compose"
    try:
        cname = container_name(cp)
    except Exception as exc:
        die(key, str(exc))
        continue
    print(
        "\t".join(
            [
                "VARIANT",
                key,
                switch_engine(key),
                launch_engine(key),
                compose_dir,
                filepart,
                str(entry["default_port"]),
                str(entry["model"]),
                str(entry["engine"]),
                str(entry.get("kvcalc_key") or "SKIP"),
                cname,
                cp,
                str(entry.get("status") or "production"),
                # ctx label: 'NK' when registry max_ctx == compose default; 'NK/MK'
                # (validated / compose) on drift. BEFORE status_note so the free-text
                # note stays the LAST catch-all field.
                ctx_label(entry),
                # status_note is free text (may contain anything but a tab) — keep
                # it as the LAST field so each reader's catch-all final var can
                # absorb it without further splitting on its internal spaces.
                (str(entry.get("status_note") or "").replace("\t", " ")),
            ]
        )
    )

for (model, engine, topology), target in DEFAULTS.items():
    print("\t".join(["DEFAULT", model, engine, topology, target]))
PY_EMIT
}

derive_switch_variant_tables() {
  local root="$1" emit key switch_engine _launch_engine cdir cfile port _model _profile_engine _kvcalc container _compose_path status max_ctx status_note
  # Self-declare so every caller (switch.sh + test-switch-registry-parity) gets
  # proper assoc arrays without each having to declare them. VARIANT_CONTAINER
  # (slug -> container name) drives switch.sh's registry-derived orphan teardown.
  declare -gA VARIANT_CTX VARIANT_CONTAINER
  local _emit_err; _emit_err="$(mktemp)"
  if ! emit="$(registry_variant_rows "$root" 2>"$_emit_err")"; then
    echo "[switch] ERROR: could not derive variant tables from compose_registry.py" >&2
    [[ -s "$_emit_err" ]] && sed 's/^/[switch]   /' "$_emit_err" >&2
    rm -f "$_emit_err"
    exit 2
  fi
  rm -f "$_emit_err"
  while IFS=$'\t' read -r kind key switch_engine _launch_engine cdir cfile port _model _profile_engine _kvcalc container _compose_path status max_ctx status_note; do
    [[ -n "${kind:-}" ]] || continue
    case "$kind" in
      VARIANT)
        if [[ "$key" == "__ERR__" ]]; then
          echo "[switch] ERROR: registry entry not launchable: ${switch_engine} (${cdir})" >&2
          exit 2
        fi
        VARIANTS["$key"]="${switch_engine}|${cdir}|${cfile}"
        VARIANT_DEFAULT_PORT["$key"]="$port"
        VARIANT_STATUS["$key"]="${status:-production}"
        VARIANT_STATUS_NOTE["$key"]="${status_note:-}"
        VARIANT_CTX["$key"]="${max_ctx:-}"
        VARIANT_CONTAINER["$key"]="${container:-}"
        ;;
    esac
  done <<< "$emit"
  if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    echo "[switch] ERROR: derived an empty variant table from compose_registry.py" >&2
    exit 2
  fi
}

derive_launch_variant_tables() {
  local root="$1" emit key _switch_engine launch_engine cdir cfile port model profile_engine kvcalc container _compose_path status _max_ctx status_note
  local _emit_err; _emit_err="$(mktemp)"
  if ! emit="$(registry_variant_rows "$root" 2>"$_emit_err")"; then
    echo "[launch] ERROR: could not derive variant tables from compose_registry.py" >&2
    [[ -s "$_emit_err" ]] && sed 's/^/[launch]   /' "$_emit_err" >&2
    rm -f "$_emit_err"
    exit 2
  fi
  rm -f "$_emit_err"
  while IFS=$'\t' read -r kind key _switch_engine launch_engine cdir cfile port model profile_engine kvcalc container _compose_path status _max_ctx status_note; do
    [[ -n "${kind:-}" ]] || continue
    case "$kind" in
      VARIANT)
        if [[ "$key" == "__ERR__" ]]; then
          echo "[launch] ERROR: registry entry not launchable: ${launch_engine} (${cdir})" >&2
          exit 2
        fi
        LAUNCH_VARIANT_COMPOSE["$key"]="${cdir}/${cfile}"
        LAUNCH_VARIANT_MODEL["$key"]="$model"
        LAUNCH_VARIANT_ENGINE["$key"]="$launch_engine"
        LAUNCH_VARIANT_PROFILE_ENGINE["$key"]="$profile_engine"
        LAUNCH_VARIANT_KVCALC["$key"]="$kvcalc"
        LAUNCH_DEFAULT_PORT["$key"]="$port"
        LAUNCH_DEFAULT_CONTAINER["$key"]="$container"
        LAUNCH_VARIANT_STATUS["$key"]="${status:-production}"
        LAUNCH_VARIANT_STATUS_NOTE["$key"]="${status_note:-}"
        LAUNCH_VARIANT_ORDER+=("$key")
        ;;
    esac
  done <<< "$emit"
  if [[ ${#LAUNCH_VARIANT_COMPOSE[@]} -eq 0 ]]; then
    echo "[launch] ERROR: derived an empty variant table from compose_registry.py" >&2
    exit 2
  fi
}

registry_default_target() {
  local root="$1" model="$2" engine="$3" topology="$4"
  python3 - "$root" "$model" "$engine" "$topology" <<'PY_DEFAULT'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import DEFAULTS  # noqa: E402

model, engine, topology = sys.argv[2:5]
target = DEFAULTS.get((model, engine, topology))
if target:
    print(target)
    raise SystemExit(0)

available = [
    f"{m}/{e}/{t}->{v}"
    for (m, e, t), v in sorted(DEFAULTS.items())
    if m == model and e == engine
]
print(
    "no default for "
    f"model={model} engine={engine} topology={topology}. "
    "Available defaults: " + (", ".join(available) if available else "<none>"),
    file=sys.stderr,
)
raise SystemExit(1)
PY_DEFAULT
}

# --- PR-B: model-default resolver (the single injection point) ---------------
#
# model_default_target ROOT MODEL TOPOLOGY  →  resolved slug on stdout.
#
# Resolution precedence ladder (design §3); `--variant X` (caller-explicit) is
# handled by the callers BEFORE they reach here, so this implements:
#   user pin (.env CLUB3090_DEFAULT_<MODEL>)
#     ↓ else  community seam (community_default_target → None today)
#     ↓ else  curated: ENGINE_PREFERENCE[topology] → first functional DEFAULTS
#     ↓ else  degradation: notice + nearest-lower topology, then a clear message
#
# The pin is read from the *environment* (callers load .env first), so this
# stays a pure function of (env, registry). Diagnostics + warnings go to stderr;
# only the resolved slug goes to stdout. Returns non-zero with a clear message
# when no functional default exists at any topology (never crashes).
model_default_target() {
  local root="$1" model="$2" topology="$3"
  # Compute the .env pin key for this model, then read its value from the
  # environment (the caller has already loaded .env into the env).
  local pin_key pin_value
  pin_key="$(python3 - "$root" "$model" <<'PY_PINKEY'
import sys
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import model_default_pin_key  # noqa: E402
print(model_default_pin_key(sys.argv[2]))
PY_PINKEY
)"
  pin_value="${!pin_key:-}"

  python3 - "$root" "$model" "$topology" "$pin_value" "$pin_key" <<'PY_MODEL_DEFAULT'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import (  # noqa: E402
    COMPOSE_REGISTRY,
    FUNCTIONAL_STATUSES,
    community_default_target,
    curated_default_target,
    model_of_slug,
    slug_topology,
    _nearest_lower_topology,
    _topology_family,
)

model, topology, pin_value, pin_key = sys.argv[2:6]


def warn(msg: str) -> None:
    print(f"[default] {msg}", file=sys.stderr)


family = _topology_family(topology)

# 1) User pin (.env). Validate: slug exists · its model matches the key · its
#    topology matches the detected one · it is NOT (NA). Any failure → warn +
#    fall through to the curated path (never block a launch — §6).
if pin_value:
    entry = COMPOSE_REGISTRY.get(pin_value)
    if entry is None:
        warn(
            f"pinned default {pin_value!r} ({pin_key}) is not a known slug — "
            "ignoring the pin, using the curated default."
        )
    elif model_of_slug(pin_value) != model:
        warn(
            f"pinned default {pin_value!r} ({pin_key}) belongs to model "
            f"{model_of_slug(pin_value)!r}, not {model!r} — ignoring the pin."
        )
    elif slug_topology(pin_value) != family:
        warn(
            f"pinned default {pin_value!r} ({pin_key}) is a "
            f"{slug_topology(pin_value)} config but this rig is {family} — "
            "ignoring the pin, using the curated default for this topology."
        )
    elif entry.get("status", "production") not in FUNCTIONAL_STATUSES:
        warn(
            f"pinned default {pin_value!r} ({pin_key}) is "
            f"(NA: {entry.get('status')}) — not a reliable config; ignoring "
            "the pin, using the curated default."
        )
    else:
        print(pin_value)
        raise SystemExit(0)

# 2) Community-ranked rung — defined now, returns None today (§13.4). Inserted
#    between the user pin and the curated fallback.
community = community_default_target(model, family)
if community:
    print(community)
    raise SystemExit(0)

# 3) Curated fallback (§4) at the detected topology.
slug = curated_default_target(model, topology)
if slug:
    print(slug)
    raise SystemExit(0)

# 4) Degradation (§6): notice + nearest-lower topology, then a clear message.
fallback_topology = _nearest_lower_topology(topology)
while fallback_topology:
    slug = curated_default_target(model, fallback_topology)
    if slug:
        warn(
            f"no functional default for {model!r} on the detected "
            f"{topology} topology — falling back to the {fallback_topology} "
            f"default ({slug})."
        )
        print(slug)
        raise SystemExit(0)
    fallback_topology = _nearest_lower_topology(fallback_topology)

warn(
    f"no default for {model!r} on this topology ({topology}) — pick a config "
    "explicitly. Run: scripts/switch.sh --list"
)
raise SystemExit(1)
PY_MODEL_DEFAULT
}

# registry_variant_rows_json ROOT  →  one JSON object on stdout.
#
# Direct-invocation companion to the SOURCED registry_variant_rows tab emitter
# (which stays byte-identical — this is purely additive). Emits the registry
# variants + defaults + the profile catalog (engines / models / hardware /
# drafters) as a single JSON object for structured consumers (e.g. the cockpit
# TUI). The variants/defaults blocks are derived by feeding the EXISTING
# registry_variant_rows tab output through club3090_tui_core.registry's
# parse_variant_rows, so the field names/order stay locked to the shared
# dataclass; profiles are sourced via the EXISTING scripts.lib.profiles loaders
# (compat.load_profiles + compose_registry.DEFAULTS) — never re-derived here.
registry_variant_rows_json() {
  local root="$1"
  # Reuse the byte-identical tab emitter for the variants/defaults rows, then
  # hand both the rows and the root to the python serializer on stdin so the
  # JSON is built from the SAME parser the cockpit/c3t use.
  local rows
  if ! rows="$(registry_variant_rows "$root")"; then
    echo "[registry-emit] ERROR: registry_variant_rows failed" >&2
    return 2
  fi
  # Pass the tab rows via the environment (not stdin) so the heredoc below can
  # still serve as the python script's stdin.
  REGISTRY_TAB="$rows" python3 - "$root" <<'PY_JSON'
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path

# PyYAML is REQUIRED on the --json contract path (profiles + baselines join —
# load_profiles() below imports it too, so check FIRST and fail with the fix,
# not a bare ModuleNotFoundError traceback). The switch.sh/launch.sh table
# path runs stdlib-only (regex container_name fallback, #584) — only
# c3-cockpit consumers need PyYAML.
try:
    import yaml  # noqa: E402,F401
except Exception:
    print(
        "registry-emit --json requires PyYAML (profiles/baselines join).\n"
        "Fix: sudo apt install python3-yaml   (or: pip install pyyaml)",
        file=sys.stderr,
    )
    raise SystemExit(3)

# Pin output to UTF-8 regardless of locale (see the table-path note, #599/#584).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

# Reuse the shared registry-row parser from club3090_tui_core WITHOUT importing
# the package __init__ (it pulls httpx, not guaranteed in this interpreter).
# Load the submodule file directly; register it in sys.modules first so its
# @dataclass decorator resolves its own module namespace.
_reg_path = root / "tools" / "tui-core" / "club3090_tui_core" / "registry.py"
_spec = importlib.util.spec_from_file_location("_c3_tui_registry", _reg_path)
_tui_registry = importlib.util.module_from_spec(_spec)
sys.modules["_c3_tui_registry"] = _tui_registry
_spec.loader.exec_module(_tui_registry)

from scripts.lib.profiles.compat import load_profiles  # noqa: E402
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY, DEFAULTS  # noqa: E402
from scripts.lib.profiles.launch_compat import ProfileError, resolve_variant_pin  # noqa: E402

tab = os.environ.get("REGISTRY_TAB", "")

# --- profiles: sourced via the EXISTING loaders (never re-derived).  Loaded
#     HERE (not after the variants block) because the baselines join below
#     resolves per-slug current pins through the engine profiles. ---
profiles = load_profiles()

# --- baselines join (catalog-baselines slice 1): the shipped bar rows from
#     scripts/lib/profiles/baselines.yml, joined per-slug with a computed
#     staleness verdict.  THIS is the single point where measured display
#     numbers enter the contract — consumers never read baselines.yml (or
#     BENCHMARKS.md) directly. ---
import re as _re  # noqa: E402

_yaml = yaml  # required-import guard at the top of this block (#584)

_bl_path = root / "scripts" / "lib" / "profiles" / "baselines.yml"
_baselines = {}
if _bl_path.exists():
    _baselines = (_yaml.safe_load(_bl_path.read_text(encoding="utf-8")) or {}).get("baselines") or {}

# First `image:` default in the compose (handles both a bare literal and the
# ${ENGINE_IMAGE:-literal} env-fallback form) — the pin truth for engines with
# no docker-image install.spec (ik / llama.cpp pin per-compose or roll by policy).
_IMG_RE = _re.compile(r"^\s*image:\s*[\"']?(?:\$\{[A-Z_0-9]+:-)?([^\s}\"']+)\}?", _re.M)


def _compose_image_default(compose_path: str):
    try:
        txt = (root / compose_path).read_text(encoding="utf-8")
    except OSError:
        return None
    m = _IMG_RE.search(txt)
    return m.group(1) if m else None


def _current_pin(slug: str, compose_path: str):
    """The pin a launcher-started serve actually runs TODAY: the engine
    profile's docker-image spec when it has one (launchers inject it), else
    the compose's image default."""
    try:
        exports = resolve_variant_pin(profiles, slug)
        # Nightly pins export a bare SHA (VLLM_NIGHTLY_SHA) — not comparable to
        # an image string; fall through to the compose default for those.
        if "VLLM_NIGHTLY_SHA" not in exports:
            return next(iter(exports.values()))
    except ProfileError:
        pass
    return _compose_image_default(compose_path)


def _baseline_for(slug: str, compose_path: str):
    row = _baselines.get(slug)
    if not row:
        return None
    out = dict(row)
    cur = _current_pin(slug, compose_path)
    measured = row.get("engine_pin")
    # stale: true/false when both pins are known; null = undeterminable
    # (unpinned engine or unreadable compose) — badge only on TRUE.
    out["stale"] = (cur != measured) if (cur and measured) else None
    out["current_pin"] = cur
    # slice 3: cross-rig submissions ride the join with the SAME per-row
    # staleness verdict (an image pin is rig-independent, so the comparison
    # holds for foreign rigs).  A submission-only entry (no primary row) has
    # out["stale"] = None and carries only this map.
    subs = row.get("submissions")
    if isinstance(subs, dict):
        out["submissions"] = {
            rc: {
                **s,
                "stale": (cur != s.get("engine_pin"))
                if (cur and s.get("engine_pin")) else None,
            }
            for rc, s in subs.items()
        }
    return out

# --- variants: exactly the fields parse_variant_rows produces from the tab form,
#     trimmed to the contract's variant schema (+ 'source' default "curated"). ---
variants = []
for vr in _tui_registry.parse_variant_rows(tab):
    d = dataclasses.asdict(vr)
    variants.append(
        {
            "slug": d["slug"],
            "switch_engine": d["switch_engine"],
            "launch_engine": d["launch_engine"],
            "compose_dir": d["compose_dir"],
            "file": d["file"],
            "port": d["port"],  # int (parse_variant_rows coerces)
            "model": d["model"],
            "engine": d["engine"],
            "kvcalc_key": d["kvcalc_key"],
            "container": d["container"],
            "compose_path": d["compose_path"],
            "status": d["status"],
            "ctx_label": d["ctx_label"],
            # The EXACT numeric configured ctx (the registry max_ctx int behind
            # ctx_label — e.g. 262144 for "262K").  Sourced straight from
            # COMPOSE_REGISTRY so the cockpit's divergence badge compares the
            # probed running ctx against the slug's CONFIGURED ctx as an exact int,
            # never round-tripping through the colloquial ÷1000 label.
            "configured_ctx": (COMPOSE_REGISTRY.get(d["slug"], {}) or {}).get("max_ctx"),
            # Per-slug download artifacts BEYOND the core weights_variant — the
            # extra weight-variant keys (a DFlash draft / an mmproj vision
            # projector) the slug's compose mounts from a separate subdir.  The
            # cockpit Download action fetches these ALONGSIDE the core so the slug
            # actually serves; without them it reads "present" then fails to boot
            # for the missing companion.  Bare keys, scoped to the row's model.
            "weights_companions": list(
                (COMPOSE_REGISTRY.get(d["slug"], {}) or {}).get("weights_companions") or []
            ),
            # Drafter / vision facets (display + companion derivation): drafter is
            # the registry's per-slug spec-dec drafter id; vision is derived from
            # the vision-coding workload (there is no separate vision field).
            "drafter": (COMPOSE_REGISTRY.get(d["slug"], {}) or {}).get("drafter"),
            "vision": (
                (COMPOSE_REGISTRY.get(d["slug"], {}) or {}).get("workload") == "vision-coding"
            ),
            "status_note": d["status_note"],
            "source": "curated",
            # The shipped baseline row ("the bar") + computed staleness — the
            # ONLY measured-display source for consumers (replaces the c3-side
            # BENCHMARKS.md scrape).  None when the slug has no accepted row.
            "baseline": _baseline_for(d["slug"], d["compose_path"]),
        }
    )

# --- defaults: from the registry DEFAULTS map (model, engine, topology -> slug). ---
defaults = [
    {
        "model": model,
        "engine": engine,
        "topology": topology,
        "slug": slug,
        "source": "curated",
    }
    for (model, engine, topology), slug in DEFAULTS.items()
]

# (profiles loaded above, before the variants block — the baselines join needs it.)


def _engine(e):
    return {
        "image": (e.install or {}).get("spec"),
        "min_sm": e.min_sm,
        "supported_kv_formats": list(e.supported_kv_formats),
        "supported_weight_formats": list(e.supported_weight_formats),
        "supported_drafters": list(e.supported_drafters),
        "supported_model_families": list(e.supported_model_families),
    }


def _model(m):
    # model-level hf_repo: surface the default weight variant's hf_repo (the
    # canonical artifact); per-variant repos live under weights.<v>.hf_repo.
    default_meta = m.weights.get(m.default_weight_variant) or {}
    return {
        "family": m.family,
        "valid_tp": list(m.valid_tp),
        "max_ctx": m.max_ctx_supported,
        "hf_repo": default_meta.get("hf_repo"),
        "weights": m.weights,
    }


def _hardware(h):
    return {
        "vram_gb": h.vram_gb,
        "sm": h.sm,
        "supported_kv_formats": list(h.supported_kv_formats),
    }


def _drafter(dr):
    return {
        # the drafter profile's spec_method is its "type"; there is no separate
        # accept-rate field in the schema today (emit null so the key is stable).
        "type": dr.spec_method,
        "accept": None,
    }


payload = {
    "variants": variants,
    "defaults": defaults,
    "profiles": {
        "engines": {eid: _engine(e) for eid, e in profiles.engines.items()},
        "models": {mid: _model(m) for mid, m in profiles.models.items()},
        "hardware": {hid: _hardware(h) for hid, h in profiles.hardware.items()},
        "drafters": {did: _drafter(dr) for did, dr in profiles.drafters.items()},
    },
}

# default=str: baseline rows carry YAML-parsed datetime.date values.
json.dump(payload, sys.stdout, sort_keys=True, default=str)
sys.stdout.write("\n")
PY_JSON
}

# x_default_dispatch ROOT TOKEN TOPOLOGY MODEL  →  resolved slug on stdout.
#
# Parses an `X/default` token (design §13.1): if X is an engine name →
# engine-recommendation (registry_default_target on the given MODEL); else if X
# is a model-id → model_default_target (X overrides MODEL); else error. Both
# sets come from the registry and are disjoint. The caller passes the model to
# use for the engine-recommendation branch (its PRIMARY_MODEL / chosen model).
x_default_dispatch() {
  local root="$1" token="$2" topology="$3" model="$4" x
  x="${token%/default}"
  local kind
  kind="$(python3 - "$root" "$x" <<'PY_DISPATCH'
import sys
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import engine_set, model_set  # noqa: E402
x = sys.argv[2]
if x in engine_set():
    print("engine")
elif x in model_set():
    print("model")
else:
    print("unknown")
PY_DISPATCH
)"
  case "$kind" in
    engine)
      registry_default_target "$root" "$model" "$x" "$topology"
      ;;
    model)
      model_default_target "$root" "$x" "$topology"
      ;;
    *)
      echo "[default] ERROR: '${token}': '${x}' is neither a known engine nor a known model." >&2
      echo "[default]        Engines: $(python3 -c "import sys; sys.path.insert(0,'$root'); from scripts.lib.profiles.compose_registry import engine_set; print(' '.join(sorted(engine_set())))")" >&2
      echo "[default]        Models:  $(python3 -c "import sys; sys.path.insert(0,'$root'); from scripts.lib.profiles.compose_registry import model_set; print(' '.join(sorted(model_set())))")" >&2
      return 1
      ;;
  esac
}

# --- Direct-invocation entrypoint (additive) ---------------------------------
#
# This file is normally SOURCED for its functions (registry_variant_rows etc.).
# When run DIRECTLY (`bash scripts/lib/registry-emit.sh --json [ROOT]`) it emits
# the structured JSON catalog via registry_variant_rows_json. Sourcing the file
# never reaches this guard, so the sourced contract is unchanged.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
  case "${1:-}" in
    --json)
      # ROOT defaults to the repo root (two levels up from scripts/lib/).
      root="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
      registry_variant_rows_json "$root"
      ;;
    *)
      echo "usage: bash ${BASH_SOURCE[0]} --json [ROOT]" >&2
      exit 2
      ;;
  esac
fi
