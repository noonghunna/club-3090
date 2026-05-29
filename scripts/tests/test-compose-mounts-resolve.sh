#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

root = Path.cwd().resolve()
compose_files = sorted(Path("models").glob("*/*/compose/*/*/*.yml"))
failures: list[str] = []

ENV_DEFAULT = re.compile(r"^\$\{[^}:]+:-(.+)\}$")

def host_source(source: str) -> str | None:
    match = ENV_DEFAULT.match(source)
    if match:
        return match.group(1)
    if source.startswith("${"):
        return None
    if source.startswith("/") or source.startswith("~"):
        return None
    if not source.startswith("."):
        return None
    return source

def split_volume(value: str) -> tuple[str, str | None]:
    # Compose short-form host paths in this repo do not contain ':' except as
    # the source/target separator. Preserve the first two fields and ignore ro/rw.
    parts = value.split(":")
    if len(parts) < 2:
        return value, None
    return parts[0], parts[1]

def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}")
        failures.append(msg)

for compose in compose_files:
    data = yaml.safe_load(compose.read_text()) or {}
    base = compose.parent.resolve()
    services = data.get("services", {}) or {}
    for service_name, service in sorted(services.items()):
        for volume in service.get("volumes", []) or []:
            if isinstance(volume, str):
                raw_source, target = split_volume(volume)
                source = host_source(raw_source)
            elif isinstance(volume, dict):
                raw_source = str(volume.get("source", ""))
                target = str(volume.get("target", ""))
                source = host_source(raw_source)
            else:
                continue
            if source is None:
                continue
            raw_path = base / source
            lexical = Path(os.path.abspath(raw_path))
            resolved = raw_path.resolve()
            label = f"{compose}:{service_name}:{target or raw_source}"
            check(str(lexical).startswith(str(root)), f"{label}: resolves inside repo")
            if "models-cache" in resolved.parts:
                check(resolved == root / "models-cache" or root / "models-cache" in resolved.parents, f"{label}: models-cache points at repo root")
            elif "cache" in resolved.parts:
                cache_idx = resolved.parts.index("cache")
                cache_root = Path(*resolved.parts[:cache_idx + 1])
                check(cache_root.exists(), f"{label}: cache root exists")
            elif "patches" in resolved.parts and "genesis" in resolved.parts:
                patches_idx = resolved.parts.index("patches")
                patches_root = Path(*resolved.parts[:patches_idx + 1])
                check(patches_root.exists(), f"{label}: patches root exists")
            else:
                check(resolved.exists(), f"{label}: source exists")

if failures:
    raise SystemExit(f"{len(failures)} mount-resolution checks failed")
PY

echo "test-compose-mounts-resolve: ok"
