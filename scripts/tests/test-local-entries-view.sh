#!/usr/bin/env bash
# Guard: `local_entries()` is a MANAGEMENT view of the local layer (#1153).
#
# Distinct from get_registry() on purpose. get_registry is the LOOKUP view, where
# core wins a collision — so a shadowed local row is absent from it BY DESIGN.
# That makes it the wrong source for a management UI: the row a user most needs
# to act on (registered, but unreachable by slug until renamed) is exactly the
# one the lookup view hides.
#
# It must also never raise. A management view that takes the cockpit down when
# the layer is malformed is worse than one that shows nothing.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
rc=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts/lib/profiles-local"
cp -r scripts/lib/profiles "$TMP/scripts/lib/" 2>/dev/null || true

out="$(python3 - "$TMP" <<'PY' 2>&1
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import (
    COMPOSE_REGISTRY, local_entries,
)

fail = []
root = Path(sys.argv[1])
reg = root / "scripts/lib/profiles-local/registry.local.json"

# 1. no layer -> empty, never an exception
if local_entries(root) != []:
    fail.append("expected [] with no local layer")

# 2. a MALFORMED layer must yield [] rather than raise
reg.write_text("{ this is not json", encoding="utf-8")
try:
    if local_entries(root) != []:
        fail.append("malformed layer did not yield []")
except Exception as exc:
    fail.append(f"malformed layer RAISED ({type(exc).__name__}) — would take the UI down")

# 3. a shadowed row must be VISIBLE here and marked, while get_registry hides it
core_slug = next(iter(COMPOSE_REGISTRY))
tmpl = dict(COMPOSE_REGISTRY[core_slug])
import inspect
from scripts.lib.profiles.compose_registry import _entry
req = [n for n, p in inspect.signature(_entry).parameters.items()
       if p.default is inspect._empty and p.kind is p.KEYWORD_ONLY]
kw = {n: (tmpl[n] if n in tmpl else "composes/x/base.yml") for n in req}
reg.write_text(json.dumps({
    "my-engine/mine": dict(kw, model="mine", status="experimental"),
    core_slug:        dict(kw, model="shadowed", status="experimental"),
}), encoding="utf-8")
rows = {e["slug"]: e for e in local_entries(root)}
if core_slug not in rows:
    fail.append("a SHADOWED row is missing from the management view — the row a "
                "user most needs to act on")
elif not rows[core_slug]["shadowed"]:
    fail.append("the shadowed row is present but not marked shadowed")
if rows.get("my-engine/mine", {}).get("shadowed"):
    fail.append("an unshadowed row was marked shadowed")
for k in ("engine", "model", "port", "max_ctx", "compose_path"):
    if k not in rows.get("my-engine/mine", {}):
        fail.append(f"management view omits {k!r}, needed to act on a row")

print("FAIL:" + "; ".join(fail) if fail else "OK")
PY
)"
case "$out" in
  OK) echo "  ok   management view: empty, malformed-safe, and shows shadowed rows" ;;
  FAIL:*) echo "  FAIL ${out#FAIL:}"; rc=1 ;;
  *) echo "  FAIL guard could not run: $out"; rc=1 ;;
esac

[[ "$rc" == "0" ]] && echo "PASS: local_entries is a usable management view" \
                   || echo "FAIL: local_entries regression"
exit "$rc"
