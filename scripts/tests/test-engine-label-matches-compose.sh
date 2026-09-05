#!/usr/bin/env bash
# Drift-guard: a registry entry's `engine` label must name the engine the
# compose ACTUALLY runs.
#
# Why this exists. When the moe-cache engine went v1.5 -> v1.6, the GLM and
# Qwen-flash composes were repointed to the new image digest but their registry
# `engine:` labels were not. Nine live slugs then advertised
# `llamacpp-club3090-v1.5` while running the v1.6 image — c3's catalogue showed
# v1.5, and any benchmark read back from the registry had the wrong provenance.
# Their deepseek-vision siblings got both halves of the same change, so this was
# a propagation gap, not a decision. Nothing caught it, which is why it sat.
#
# The rule, and why each clause is here rather than a simple equality test:
#
#   (1) SKIP deprecated entries. Retired engines legitimately keep stale pins as
#       history (7 beellama slugs + 1 deprecated vLLM one drift this way today);
#       failing on them would make the gate permanently red for no benefit.
#   (2) SKIP engines whose profile declares NO `install.spec`. That is by design
#       for ik-llama and llama-cpp-local — registry-emit.sh: "no docker-image
#       install.spec (ik / llama.cpp pin per-compose or roll by policy)". There
#       is nothing to compare against, and treating absence as a mismatch would
#       red every one of those entries.
#   (3) Otherwise the compose's first `image:` default must EQUAL the profile
#       spec.
#
# The two failure modes are NOT the same and the message says which:
#   - llama.cpp family: the compose is the pin truth (the launchers inject only
#     VLLM_IMAGE / BEELLAMA_IMAGE), so a mismatch means the LABEL IS FALSE about
#     what runs.
#   - vLLM: the launcher injects the profile's image and OVERRIDES the compose
#     default, so a mismatch means the compose's own default is misleading to a
#     human reading it — real, but a different bug.
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale decodes reads, stdout AND argv with the locale codec, which
# crashes the launcher/emit paths (#779). Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
import re
from pathlib import Path

import yaml

from scripts.lib.profiles.compose_registry import get_registry

ROOT = Path(".")
ENGINES = ROOT / "scripts" / "lib" / "profiles" / "engines"

# Matches a bare literal AND the ${ENGINE_IMAGE:-literal} env-fallback form —
# the same shape registry-emit.sh's _compose_image_default reads.
IMG = re.compile(r"^\s*image:\s*[\"']?(?:\$\{[A-Z_0-9]+:-)?([^\s}\"']+)\}?", re.M)

# Statuses that are historical rather than live. A retired engine keeping a
# stale pin is a record, not a defect.
SKIP_STATUS = {"deprecated"}

failures = []
checked = skipped_status = skipped_nospec = 0


def check(cond, msg):
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}")
        failures.append(msg)


_spec_cache = {}


def engine_spec(engine):
    """The engine profile's declared image, or None when it declares none."""
    if engine not in _spec_cache:
        p = ENGINES / f"{engine}.yml"
        spec = None
        if p.is_file():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            spec = (data.get("install") or {}).get("spec")
        _spec_cache[engine] = spec
    return _spec_cache[engine]


for slug, entry in sorted(get_registry().items()):
    engine = entry.get("engine")
    compose_path = entry.get("compose_path")
    if not engine or not compose_path:
        continue
    if entry.get("status") in SKIP_STATUS:
        skipped_status += 1
        continue

    spec = engine_spec(engine)
    if not spec:
        # (2) by design for ik / llama.cpp — nothing declared to compare against.
        skipped_nospec += 1
        continue

    f = ROOT / compose_path
    if not f.is_file():
        continue
    m = IMG.search(f.read_text(encoding="utf-8"))
    if not m:
        continue

    compose_img = m.group(1)
    checked += 1
    if compose_img == spec:
        continue

    if engine.startswith("vllm"):
        why = (
            "the launcher INJECTS the profile image, so the compose default is a "
            "misleading fallback (a human reading the compose sees the wrong tag)"
        )
    else:
        why = (
            "this engine pins PER-COMPOSE, so the compose wins and the registry "
            "label is FALSE about what actually runs"
        )
    check(
        False,
        f"{slug}: engine label {engine!r} declares {spec!r} but the compose pins "
        f"{compose_img!r} — {why}",
    )

print(
    f"\nchecked {checked} entries "
    f"({skipped_status} skipped: deprecated, {skipped_nospec} skipped: engine declares no spec)"
)
if failures:
    print(f"\n{len(failures)} engine-label mismatch(es)")
    raise SystemExit(1)
print("engine labels match the images their composes pin")
PY
