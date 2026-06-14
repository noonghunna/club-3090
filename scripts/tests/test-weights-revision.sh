#!/usr/bin/env bash
# #319: weights-fetch schema should support an optional `revision:` pin
# (commit SHA / tag) per weights variant, round-tripping through weights.py as
# WEIGHT_REVISION. Unset must stay empty (track HEAD = today's behavior).
#
# In-process: imports weights.py and points PROFILE_ROOT at a throwaway fixture
# tree, so no real model entry is touched and the tracked models/ dir is never
# written to.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "weights", root / "scripts" / "lib" / "profiles" / "weights.py"
)
weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)

FIXTURE = """\
schema_version: 1
id: revfixture
display_name: Revision Fixture
weights:
  pinned:
    local_subdir: revfixture-pinned
    hf_repo: example/Repo
    revision: 65f69c7abc1234
    engine: vllm
    kind: main
  floating:
    local_subdir: revfixture-floating
    hf_repo: example/Repo
    engine: vllm
    kind: main
"""

failures = []
with tempfile.TemporaryDirectory() as tmp:
    models = Path(tmp) / "models"
    models.mkdir()
    (models / "revfixture.yml").write_text(FIXTURE, encoding="utf-8")
    weights.PROFILE_ROOT = Path(tmp)

    pinned = weights._recipe("revfixture", "pinned")
    if pinned.get("WEIGHT_REVISION") != "65f69c7abc1234":
        failures.append(
            f"pinned: WEIGHT_REVISION got {pinned.get('WEIGHT_REVISION')!r} "
            f"expected '65f69c7abc1234'"
        )

    floating = weights._recipe("revfixture", "floating")
    if floating.get("WEIGHT_REVISION") != "":
        failures.append(
            f"floating: WEIGHT_REVISION got {floating.get('WEIGHT_REVISION')!r} "
            f"expected '' (unset = track HEAD)"
        )

if failures:
    print("ASSERTION FAILED:", file=sys.stderr)
    for f in failures:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)

print("OK: weights.py round-trips optional revision: as WEIGHT_REVISION")
PY
