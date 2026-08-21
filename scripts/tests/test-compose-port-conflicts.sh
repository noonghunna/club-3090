#!/usr/bin/env bash
# Gate: no two DISTINCT active slugs from DIFFERENT models share a default_port.
#
# Why: default_port is the host port the launcher/c3/estate bind a slug to. Two
# different models on the same port can't co-reside (GPU0+GPU1 small models), and
# it breaks slug<->port identity in c3 even on this one-model-at-a-time rig. Ports
# were historically assigned per-model with no global check, so overlaps crept in
# (qwen3.8's 8091-8112 block overlapped nemotron; iq4nl/iq4xs shared 8086 with
# inkling). This gate stops that regressing.
#
# Policy (three cases, only the third fails):
#   (a) ALIAS         — same compose_path under two slug names  -> OK
#   (b) SAME-MODEL    — variants of ONE model (mutually exclusive) -> OK
#   (c) CROSS-MODEL   — distinct models, distinct compose_paths  -> FAIL
#
# Deprecated slugs are excluded (they're not launchable). Known pre-existing
# cross-model collisions are allowlisted (ALLOW) and tracked for a separate
# hygiene PR; the gate ALSO fails if an ALLOW entry no longer collides, so the
# list can't silently rot (negative control).
set -euo pipefail

# Force Python UTF-8 mode (PEP 540) before the first python3 call (#779). Guarded
# by test-locale-utf8.sh; exported so children inherit it.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
import sys, collections
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY as R

# Pre-existing cross-model collisions, tracked for a separate port-hygiene PR.
# Burn these down and DELETE them from here — the gate fails if a listed port
# stops colliding, so a stale entry can't hide.
#   8020 — vllm/minimal (qwen3.6-27b) + llamacpp/tess-dual-mtp (tess-4-27b)
#   8032 — vllm/gemma-31b-dual (gemma-4-31b) + deepseek-flash-multi4 (deepseek-v4-flash)
ALLOW = {8020, 8032}

byport = collections.defaultdict(list)
for slug, e in R.items():
    if (e.get("status") or "").lower() == "deprecated":
        continue
    port = e.get("default_port")
    if port is None:
        continue
    byport[port].append((slug, e.get("model"), e.get("compose_path")))

conflicts = []
allow_still_colliding = set()
for port, entries in sorted(byport.items()):
    if len(entries) < 2:
        continue
    paths = {p for _, _, p in entries}
    models = {m for _, m, _ in entries}
    if len(paths) == 1:      # (a) alias — same compose_path
        continue
    if len(models) == 1:     # (b) same-model, mutually exclusive
        continue
    # (c) cross-model collision
    if port in ALLOW:
        allow_still_colliding.add(port)
        continue
    conflicts.append((port, entries))

fail = False

if conflicts:
    fail = True
    print("FAIL: cross-model default_port collisions (distinct models sharing a host port):")
    for port, entries in conflicts:
        print(f"  PORT {port}:")
        for slug, model, path in sorted(entries):
            print(f"    {slug}  [{model}]")
    print("Fix: reassign one side to a free default_port — registry default_port +")
    print("     the compose ${PORT:-NNNN} + any doc-table/port references. If it is a")
    print("     genuine, tracked pre-existing overlap, add the port to ALLOW with a note.")

stale = ALLOW - allow_still_colliding
if stale:
    fail = True
    print(f"FAIL: ALLOW lists port(s) {sorted(stale)} that no longer cross-model-collide.")
    print("      Remove them from ALLOW — the debt they tracked is paid.")

if fail:
    sys.exit(1)

active = sum(len(v) for v in byport.values())
print(f"OK: no un-allowlisted cross-model port collisions "
      f"({active} active slug-ports; ALLOW={sorted(ALLOW)} still valid).")
PY
