#!/usr/bin/env bash
# Drift-guard for the slug health/availability flag (PR-A, design §12.5).
#
# Asserts that the registry `status` field and the compose-header `Status:`
# enum can't silently diverge again. Three checks:
#   (a) every registry `status` is in the canonical enum (STATUS_VALUES);
#   (b) every registered compose's profile-schema `Status:` header maps to the
#       enum (i.e. it carries a canonical leading emoji — no "Working (with
#       Genesis)"-style non-enum strings);
#   (c) the registry `status` is consistent with its compose header.
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
from pathlib import Path

from scripts.lib.profiles.compose_registry import (
    COMPOSE_REGISTRY,
    STATUS_VALUES,
    compose_header_status,
)

failures = []


def check(cond, msg):
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}")
        failures.append(msg)


enum = set(STATUS_VALUES)

for key, entry in sorted(COMPOSE_REGISTRY.items()):
    status = entry.get("status")
    # (a) registry status in the enum.
    check(status in enum, f"{key}: registry status {status!r} in enum")

    compose_path = Path(entry["compose_path"])
    if not compose_path.exists():
        check(False, f"{key}: compose file exists at {compose_path}")
        continue

    header = compose_header_status(compose_path.read_text(encoding="utf-8"))
    # (b) compose header maps to the enum (canonical leading emoji present).
    check(
        header in enum,
        f"{key}: compose Status header maps to enum (got {header!r} "
        f"from {compose_path.as_posix()})",
    )
    # (c) registry status consistent with the compose header.
    check(
        header == status,
        f"{key}: registry status ({status!r}) consistent with compose "
        f"header ({header!r})",
    )

if failures:
    raise SystemExit(f"{len(failures)} status-drift checks failed")
PY

echo "test-compose-status-drift: ok"
