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
  local root="$1" emit key switch_engine _launch_engine cdir cfile port _model _profile_engine _kvcalc _container _compose_path status status_note
  if ! emit="$(registry_variant_rows "$root" 2>/dev/null)"; then
    echo "[switch] ERROR: could not derive variant tables from compose_registry.py" >&2
    exit 2
  fi
  while IFS=$'\t' read -r kind key switch_engine _launch_engine cdir cfile port _model _profile_engine _kvcalc _container _compose_path status status_note; do
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
        ;;
    esac
  done <<< "$emit"
  if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    echo "[switch] ERROR: derived an empty variant table from compose_registry.py" >&2
    exit 2
  fi
}

derive_launch_variant_tables() {
  local root="$1" emit key _switch_engine launch_engine cdir cfile port model profile_engine kvcalc container _compose_path status status_note
  if ! emit="$(registry_variant_rows "$root" 2>/dev/null)"; then
    echo "[launch] ERROR: could not derive variant tables from compose_registry.py" >&2
    exit 2
  fi
  while IFS=$'\t' read -r kind key _switch_engine launch_engine cdir cfile port model profile_engine kvcalc container _compose_path status status_note; do
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
