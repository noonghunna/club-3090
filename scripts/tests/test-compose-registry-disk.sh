#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
from pathlib import Path

from scripts.lib.profiles.compat import load_profiles
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY, DEFAULTS

root = Path.cwd()
profiles = load_profiles()
registry_paths = {Path(entry["compose_path"]) for entry in COMPOSE_REGISTRY.values()}
disk_paths = set(Path("models").glob("*/*/compose/*/*/*.yml"))

failures = []

def check(cond, msg):
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}")
        failures.append(msg)

check(len(COMPOSE_REGISTRY) == 56, f"registry has 56 entries (got {len(COMPOSE_REGISTRY)})")
check(len(disk_paths) == 56, f"disk has 56 compose files (got {len(disk_paths)})")
check(registry_paths <= disk_paths, "all registry compose_path values exist on disk")
parked_disk_only = disk_paths - registry_paths
check(
    all("/sglang/compose/" in f"/{path.as_posix()}" for path in parked_disk_only),
    "only parked SGLang archive composes are disk-only",
)
if parked_disk_only:
    print("INFO: disk-only parked composes: " + ", ".join(str(p) for p in sorted(parked_disk_only)))

for name, entry in sorted(COMPOSE_REGISTRY.items()):
    path = Path(entry["compose_path"])
    parts = path.parts
    check(path.exists(), f"{name}: compose_path exists")
    check(path.name not in {"docker-compose.yml", "default.yml"}, f"{name}: filename is descriptive")
    try:
        idx = parts.index("compose")
        topology, quant_slug, filename = parts[idx + 1:idx + 4]
    except (ValueError, IndexError):
        check(False, f"{name}: path follows compose/<topology>/<quant>/<file>.yml")
        continue
    check(topology in {"single", "dual", "multi4"}, f"{name}: topology segment valid")
    check(filename.endswith(".yml"), f"{name}: compose filename is .yml")
    check(quant_slug == entry["weights_variant"], f"{name}: quant slug matches weights_variant")
    model = profiles.models[entry["model"]]
    check(entry["weights_variant"] in model.weights, f"{name}: weights_variant exists in ModelProfile")

for key, name in sorted(DEFAULTS.items()):
    model, _engine, topology = key
    entry = COMPOSE_REGISTRY.get(name)
    check(entry is not None, f"DEFAULTS{key}: target exists")
    if entry is None:
        continue
    path_parts = Path(entry["compose_path"]).parts
    idx = path_parts.index("compose")
    check(entry["model"] == model, f"DEFAULTS{key}: model matches target")
    check(path_parts[idx + 1] == topology, f"DEFAULTS{key}: topology matches target path")

if failures:
    raise SystemExit(f"{len(failures)} registry/disk checks failed")
PY

echo "test-compose-registry-disk: ok"
