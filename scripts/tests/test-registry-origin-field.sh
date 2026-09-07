#!/usr/bin/env bash
# Guard: catalog provenance is a FIELD (`origin`), stamped by the loader, and
# core wins a slug collision (#1202 P2).
#
# Why a field and not the slug prefix: publishing a local recipe upstream later
# must flip a value and leave the slug alone. Encoding provenance in the NAME
# makes publish a RENAME, breaking every reference a user holds. export_pr.py
# already does that rename today (`core_slug = f"{ns}/{slug[len('local/'):]}"`).
#
# Three properties, each of which has failed or could fail silently:
#   1. every row carries `origin`; core rows say "core"
#   2. a local row cannot DECLARE its own origin (spoofing core would hide it)
#   3. core wins the merged view. This one is a real inversion: the merge was
#      `merged.update(local)`, i.e. LOCAL silently won. Unreachable while local
#      slugs sit in their own namespace, live the moment P3 lands.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
rc=0

out="$(python3 - <<'PY' 2>&1
import inspect, json, sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, ".")
from scripts.lib.profiles import compose_registry as cr

fail = []

# 1. core rows stamp origin=core
core_slug = next(iter(cr.COMPOSE_REGISTRY))
if cr.COMPOSE_REGISTRY[core_slug].get("origin") != "core":
    fail.append(f"core row {core_slug!r} origin={cr.COMPOSE_REGISTRY[core_slug].get('origin')!r}, want 'core'")
missing = [s for s, e in cr.COMPOSE_REGISTRY.items() if "origin" not in e]
if missing:
    fail.append(f"{len(missing)} core rows carry no origin field, e.g. {missing[0]!r}")

sig = inspect.signature(cr._entry)
required = [n for n, p in sig.parameters.items()
            if p.default is inspect._empty and p.kind is p.KEYWORD_ONLY]
tmpl = dict(cr.COMPOSE_REGISTRY[core_slug])
kw = {n: (tmpl[n] if n in tmpl else "composes/x/base.yml") for n in required}

root = Path(tempfile.mkdtemp())
try:
    (root / "scripts/lib/profiles-local").mkdir(parents=True)
    reg = root / "scripts/lib/profiles-local/registry.local.json"

    # 2. a valid local row is stamped local, not self-declared
    mine = dict(kw); mine["model"] = "guard-local-model"; mine["status"] = "experimental"
    reg.write_text(json.dumps({"local/guard": mine}))
    e = cr.load_local_registry(root)["local/guard"]
    if e.get("origin") != "local":
        fail.append(f"local row origin={e.get('origin')!r}, want 'local'")

    # 3. spoofing origin must be refused
    spoof = dict(mine); spoof["origin"] = "core"
    reg.write_text(json.dumps({"local/spoof": spoof}))
    try:
        cr.load_local_registry(root)
        fail.append("a local row was allowed to declare origin='core' (spoofable)")
    except cr.LocalRegistryError:
        pass

    # 4. core wins the merge
    real = cr.load_local_registry
    cr.load_local_registry = lambda r=None: {
        core_slug: {**tmpl, "origin": "local", "shadowed_by_core": True}
    }
    try:
        if cr.get_registry()[core_slug].get("origin") != "core":
            fail.append("a local row shadowed a CORE slug in the merged view")
    finally:
        cr.load_local_registry = real
finally:
    shutil.rmtree(root)

print("FAIL:" + "; ".join(fail) if fail else "OK")
PY
)"

case "$out" in
  OK) echo "  ok   origin stamped, unspoofable, core wins" ;;
  FAIL:*) echo "  FAIL ${out#FAIL:}"; rc=1 ;;
  *) echo "  FAIL guard could not run: $out"; rc=1 ;;
esac

# Source-level: the merge must not be a plain update(), which lets local win.
if command grep -qE "^\s*merged\.update\(local\)" scripts/lib/profiles/compose_registry.py; then
  echo "  FAIL get_registry uses merged.update(local) — local silently shadows core"; rc=1
else
  echo "  ok   merge is not a blind update(local)"
fi

[[ "$rc" == "0" ]] && echo "PASS: provenance is a field and core wins" || echo "FAIL: origin/precedence regression"
exit "$rc"
