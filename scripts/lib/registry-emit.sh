#!/usr/bin/env bash
# Shared compose-registry shell emitter for switch.sh and launch.sh.
#
# Source this file, declare the destination arrays in the caller, then call
# derive_switch_variant_tables or derive_launch_variant_tables with ROOT_DIR.

registry_variant_rows() {
  local root="$1"
  python3 - "$root" <<'PY_EMIT'
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

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


def container_name(compose_path: str) -> str:
    path = root / compose_path
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        raise RuntimeError(f"could not parse compose yaml: {exc}") from exc
    services = data.get("services") or {}
    for service in services.values():
        raw = service.get("container_name")
        if not raw:
            continue
        raw = str(raw)
        match = re.fullmatch(r"\$\{[^}:]+:-(.+)\}", raw)
        return match.group(1) if match else raw
    return ""


_CTX_ENV = re.compile(r"\$\{(?:MAX_MODEL_LEN|CTX_SIZE|CTX|MAX_CTX|N_CTX|MODEL_LEN)\s*:-\s*(\d+)\s*\}")
_CTX_FLAG = re.compile(r"(?:--max-model-len|--ctx-size|--n-ctx|(?<!\w)-c)\s*\n?\s*\"?(\d{3,})")


def compose_default_ctx(compose_path: str):
    """The ctx the compose serves by DEFAULT (its ${VAR:-N} fallback or flag literal)."""
    try:
        txt = (root / compose_path).read_text()
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
  local root="$1" emit key switch_engine _launch_engine cdir cfile port _model _profile_engine _kvcalc _container _compose_path status max_ctx status_note
  # Self-declare so every caller (switch.sh + test-switch-registry-parity) gets a
  # proper assoc array without each having to declare it.
  declare -gA VARIANT_CTX
  if ! emit="$(registry_variant_rows "$root" 2>/dev/null)"; then
    echo "[switch] ERROR: could not derive variant tables from compose_registry.py" >&2
    exit 2
  fi
  while IFS=$'\t' read -r kind key switch_engine _launch_engine cdir cfile port _model _profile_engine _kvcalc _container _compose_path status max_ctx status_note; do
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
  if ! emit="$(registry_variant_rows "$root" 2>/dev/null)"; then
    echo "[launch] ERROR: could not derive variant tables from compose_registry.py" >&2
    exit 2
  fi
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
