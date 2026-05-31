#!/usr/bin/env bash
# CONTRACT-2b-ii — switch.sh ↔ compose_registry.py parity.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REG_EXPECTED="$(python3 - "$ROOT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY

for key, entry in COMPOSE_REGISTRY.items():
    engine_prefix = key.split("/", 1)[0]
    engine = "llamacpp" if engine_prefix == "llamacpp" else engine_prefix
    cp = entry["compose_path"]
    assert "/compose/" in cp, f"registry {key} compose_path lacks /compose/: {cp}"
    dirpart, filepart = cp.split("/compose/", 1)
    print(f"{key}\t{engine}|{dirpart}/compose|{filepart}\t{entry['default_port']}")
PY
)"

declare -A REG_MAP REG_PORT SW_MAP SW_PORT VARIANTS VARIANT_DEFAULT_PORT VARIANT_STATUS VARIANT_STATUS_NOTE
while IFS=$'\t' read -r key spec port; do
  [[ -n "$key" ]] || continue
  REG_MAP["$key"]="$spec"
  REG_PORT["$key"]="$port"
done <<< "$REG_EXPECTED"

# shellcheck source=../lib/registry-emit.sh
source "$ROOT_DIR/scripts/lib/registry-emit.sh"
derive_switch_variant_tables "$ROOT_DIR"
for key in "${!VARIANTS[@]}"; do
  SW_MAP["$key"]="${VARIANTS[$key]}"
  SW_PORT["$key"]="${VARIANT_DEFAULT_PORT[$key]}"
done

fail=0
note() { echo "FAIL: $1" >&2; fail=1; }

for k in "${!REG_MAP[@]}"; do
  [[ -n "${SW_MAP[$k]:-}" ]] || note "registered compose '$k' is NOT launchable via switch.sh"
done
for k in "${!SW_MAP[@]}"; do
  [[ -n "${REG_MAP[$k]:-}" ]] || note "switch.sh variant '$k' has NO compose_registry.py entry (ghost)"
done
for k in "${!REG_MAP[@]}"; do
  [[ -n "${SW_MAP[$k]:-}" ]] || continue
  [[ "${SW_MAP[$k]}" == "${REG_MAP[$k]}" ]] || note "variant '$k' spec drift: switch='${SW_MAP[$k]}' registry='${REG_MAP[$k]}'"
  [[ "${SW_PORT[$k]:-}" == "${REG_PORT[$k]}" ]] || note "variant '$k' default_port drift: switch='${SW_PORT[$k]:-<none>}' registry='${REG_PORT[$k]}'"
done
for k in "${!SW_MAP[@]}"; do
  IFS='|' read -r _eng cdir cfile <<< "${SW_MAP[$k]}"
  [[ -f "$ROOT_DIR/$cdir/$cfile" ]] || note "variant '$k' resolves to a missing compose file: $cdir/$cfile"
done

while IFS=$'\t' read -r relpath registered; do
  [[ -n "$relpath" ]] || continue
  [[ "$registered" == "1" ]] || note "single-card compose exists on disk but has no compose_registry.py entry: $relpath"
done < <(
  python3 - "$ROOT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY

registered_paths = {entry["compose_path"] for entry in COMPOSE_REGISTRY.values()}
for engine in ("vllm", "llama-cpp", "ik-llama"):
    for path in sorted(root.glob(f"models/*/{engine}/compose/single/*/*.yml")):
        rel = path.relative_to(root).as_posix()
        print(f"{rel}\t{1 if rel in registered_paths else 0}")
PY
)

if [[ "$fail" -ne 0 ]]; then
  echo "[switch-registry-parity] FAIL" >&2
  exit 1
fi
echo "[switch-registry-parity] PASS: ${#REG_MAP[@]} registered composes, all launchable; zero ghosts; spec+port parity"
