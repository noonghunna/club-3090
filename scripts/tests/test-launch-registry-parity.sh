#!/usr/bin/env bash
# PR-B — launch.sh tables are derived from compose_registry.py.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

declare -A REG_MODEL REG_ENGINE REG_PORT REG_KVCALC REG_CONTAINER REG_COMPOSE
eval "$(python3 - "$ROOT_DIR" <<'PY'
import re
import shlex
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY


def launch_engine(key):
    prefix = key.split('/', 1)[0]
    return 'llamacpp' if prefix in {'llamacpp', 'ik-llama'} else prefix


def container(path):
    data = yaml.safe_load((root / path).read_text()) or {}
    for service in (data.get('services') or {}).values():
        raw = service.get('container_name')
        if not raw:
            continue
        raw = str(raw)
        m = re.fullmatch(r'\$\{[^}:]+:-(.+)\}', raw)
        return m.group(1) if m else raw
    return ''

for key, entry in COMPOSE_REGISTRY.items():
    q = shlex.quote(key)
    print(f"REG_MODEL[{q}]={shlex.quote(str(entry['model']))}")
    print(f"REG_ENGINE[{q}]={shlex.quote(launch_engine(key))}")
    print(f"REG_PORT[{q}]={shlex.quote(str(entry['default_port']))}")
    print(f"REG_KVCALC[{q}]={shlex.quote(str(entry.get('kvcalc_key') or 'SKIP'))}")
    print(f"REG_CONTAINER[{q}]={shlex.quote(container(entry['compose_path']))}")
    print(f"REG_COMPOSE[{q}]={shlex.quote(str(entry['compose_path']))}")
PY
)"

declare -A LAUNCH_VARIANT_COMPOSE LAUNCH_VARIANT_MODEL LAUNCH_VARIANT_ENGINE LAUNCH_VARIANT_PROFILE_ENGINE LAUNCH_VARIANT_KVCALC LAUNCH_DEFAULT_PORT LAUNCH_DEFAULT_CONTAINER LAUNCH_VARIANT_STATUS LAUNCH_VARIANT_STATUS_NOTE
LAUNCH_VARIANT_ORDER=()
# shellcheck source=../lib/registry-emit.sh
source "$ROOT_DIR/scripts/lib/registry-emit.sh"
derive_launch_variant_tables "$ROOT_DIR"

fail=0
note() { echo "FAIL: $1" >&2; fail=1; }

for k in "${!REG_COMPOSE[@]}"; do
  [[ -n "${LAUNCH_VARIANT_COMPOSE[$k]:-}" ]] || note "registry '$k' missing from launch derived table"
done
for k in "${!LAUNCH_VARIANT_COMPOSE[@]}"; do
  [[ -n "${REG_COMPOSE[$k]:-}" ]] || note "launch ghost '$k' missing from registry"
done
for k in "${!REG_COMPOSE[@]}"; do
  [[ -n "${LAUNCH_VARIANT_COMPOSE[$k]:-}" ]] || continue
  [[ "${LAUNCH_VARIANT_COMPOSE[$k]}" == "${REG_COMPOSE[$k]}" ]] || note "$k compose mismatch"
  [[ "${LAUNCH_VARIANT_MODEL[$k]}" == "${REG_MODEL[$k]}" ]] || note "$k model mismatch"
  [[ "${LAUNCH_VARIANT_ENGINE[$k]}" == "${REG_ENGINE[$k]}" ]] || note "$k engine mismatch"
  [[ "${LAUNCH_DEFAULT_PORT[$k]}" == "${REG_PORT[$k]}" ]] || note "$k port mismatch"
  [[ "${LAUNCH_VARIANT_KVCALC[$k]}" == "${REG_KVCALC[$k]}" ]] || note "$k kvcalc mismatch"
  [[ "${LAUNCH_DEFAULT_CONTAINER[$k]}" == "${REG_CONTAINER[$k]}" ]] || note "$k container mismatch"
  [[ -f "$ROOT_DIR/${LAUNCH_VARIANT_COMPOSE[$k]}" ]] || note "$k compose file missing"
done

if [[ "$fail" -ne 0 ]]; then
  echo "[launch-registry-parity] FAIL" >&2
  exit 1
fi
echo "[launch-registry-parity] PASS: ${#REG_COMPOSE[@]} registered composes; launch tables match registry model/engine/path/port/container/kvcalc"
