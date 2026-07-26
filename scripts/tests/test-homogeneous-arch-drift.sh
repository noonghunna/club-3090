#!/usr/bin/env bash
# Drift-guard for the mixed-architecture TP refusal (#762, #783).
#
# `scripts/preflight.sh` reads `# Requires-homogeneous-arch: true` from the
# COMPOSE HEADER. The registry carries the same fact as `requires_homogeneous_arch`.
# Two sources, one truth — so they can drift, and drift here is expensive: the
# failure mode is a user's 12-iteration crash-loop instead of a one-line refusal.
#
# That is exactly what happened. #762 shipped the guard on the single slug the
# bug report named; the 35B-A3B modelopt sibling has the same hardcoded-kernel
# property and stayed unguarded, crash-looping on mixed rigs until it surfaced
# in disc #768 Test 3 (#783). A per-slug guard with no class-level check will
# keep missing siblings.
#
# Three checks:
#   (a) registry True  -> compose header MUST carry the marker (the #783 gap);
#   (b) compose header -> registry MUST be True (no orphan headers preflight
#       honours but the catalog/c3 can't see);
#   (c) TP=1 slugs must NOT set it — there is no TP group to disagree, so a
#       flag there is a copy-paste error that would refuse a valid single-card
#       boot on a mixed rig.
#
# NOT asserted: that every nvfp4 slug is gated. The axis is the QUANT RUNTIME,
# not the weights_variant string — compressed-tensors exports (unsloth
# `nvfp4-fast`, migtissera tess) reach Marlin through the shared kernel
# selector and mixed-arch is solvable for them (disc #768 Test 6, DiffusionGemma).
# Gating by `weights_variant == "nvfp4"` would foreclose that. See #783.
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
import re
import sys
from pathlib import Path

from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY

# Matches what preflight.sh's compose_meta_get accepts: a header comment line
# `# Requires-homogeneous-arch: <value>`, case-insensitive key, anywhere in the
# leading comment block. Kept deliberately loose on whitespace so a reformat
# doesn't silently disarm the guard.
MARKER = re.compile(
    r"^\s*#\s*Requires-homogeneous-arch\s*:\s*(\S+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

TRUTHY = {"true", "yes", "1"}

failures = []
checked = 0
gated = []

for slug, entry in sorted(COMPOSE_REGISTRY.items()):
    rel = entry.get("compose_path")
    if not rel:
        continue
    path = Path(rel)
    if not path.is_file():
        # Missing compose files are the job of test-registry-json / the
        # existence gates; don't duplicate (or double-report) them here.
        continue

    checked += 1
    text = path.read_text(encoding="utf-8")
    m = MARKER.search(text)
    header_on = bool(m) and m.group(1).strip().lower() in TRUTHY
    registry_on = bool(entry.get("requires_homogeneous_arch", False))
    tp = entry.get("tp")

    # (a) registry says gated, compose header doesn't -> preflight WON'T refuse.
    #     This is the #783 bug: the fact was known, the enforcement point missed.
    if registry_on and not header_on:
        failures.append(
            f"{slug}: registry requires_homogeneous_arch=True but "
            f"{rel} has no `# Requires-homogeneous-arch: true` header — "
            f"preflight reads the HEADER, so mixed-arch boots are NOT refused. "
            f"This is the #783 failure mode."
        )

    # (b) header says gated, registry doesn't -> preflight refuses but the
    #     catalog/c3/compat layer has no idea, so the slug looks launchable.
    if header_on and not registry_on:
        failures.append(
            f"{slug}: {rel} carries `# Requires-homogeneous-arch: true` but the "
            f"registry entry does not set requires_homogeneous_arch=True — "
            f"preflight refuses while the catalog still advertises it."
        )

    # (c) TP=1 can't have a heterogeneous TP group. A flag here refuses a boot
    #     that would have worked.
    if (registry_on or header_on) and tp == 1:
        failures.append(
            f"{slug}: requires_homogeneous_arch set on a TP=1 slug (tp={tp}). "
            f"A single-card slug has no TP group to disagree across; this would "
            f"refuse a valid boot on a mixed rig."
        )

    if registry_on or header_on:
        gated.append(f"{slug} (tp={tp})")

if not checked:
    print("FAIL test-homogeneous-arch-drift: no registered composes found on disk")
    sys.exit(1)

if failures:
    print("FAIL test-homogeneous-arch-drift:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(
    f"PASS test-homogeneous-arch-drift: {checked} composes checked, "
    f"{len(gated)} arch-gated"
)
for g in gated:
    print(f"  gated: {g}")
PY
