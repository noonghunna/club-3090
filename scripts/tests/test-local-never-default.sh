#!/usr/bin/env bash
# Guard: a LOCAL entry can never be resolved as a curated default (#1202 1f).
#
# The invariant holds today by construction, not by a check: `X/default` dispatch
# reads engine_set() / model_set(), and both derive from DEFAULTS and
# ENGINE_PREFERENCE, which bind to _CORE_*. get_registry() — the MERGED view every
# runtime consumer reads — is never consulted. compose_registry.py states it in a
# comment: "The local layer can NEVER leak into DEFAULTS / ENGINE_PREFERENCE /
# RECOMMENDED_DEFAULT_MODELS."
#
# ⚠️ WHY PIN SOMETHING THAT ALREADY HOLDS. It holds via three bindings plus a
# comment, and repointing COMPOSE_REGISTRY at the merged view would break it
# SILENTLY — a plausible refactor, since the merged view is what runtime reads.
# Before #1202 a local slug was `local/<name>`, so a leak would have been obvious
# on sight. Now local slugs are shape-identical to curated ones, so the same
# breakage is invisible. The change did not create the risk; it removed the thing
# that would have made the risk visible.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
rc=0

out="$(python3 - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from scripts.lib.profiles import compose_registry as cr

fail = []

# 1. The two dispatch sets must come from CORE, never from the merged view.
#    Feed get_registry a local-only engine and assert it stays invisible to them.
real = cr.load_local_registry
core_slug = next(iter(cr.COMPOSE_REGISTRY))
tmpl = dict(cr.COMPOSE_REGISTRY[core_slug])
cr.load_local_registry = lambda root=None: {
    "my-own-engine/my-own-model": {
        **tmpl, "origin": "local", "model": "my-own-model",
        "engine": "my-own-engine", "shadowed_by_core": False,
    }
}
try:
    merged = cr.get_registry()
    if "my-own-engine/my-own-model" not in merged:
        fail.append("fixture did not reach the merged view; the checks below prove nothing")
    if "my-own-engine" in cr.engine_set():
        fail.append("a LOCAL engine leaked into engine_set() — `my-own-engine/default` would resolve")
    if "my-own-model" in cr.model_set():
        fail.append("a LOCAL model leaked into model_set() — `my-own-model/default` would resolve")
    if any("my-own" in str(s) for s in cr.DEFAULTS.values()):
        fail.append("a LOCAL slug appears in DEFAULTS")
finally:
    cr.load_local_registry = real

# 2. The bindings themselves. If COMPOSE_REGISTRY or DEFAULTS is ever repointed at
#    the merged view, everything above silently starts passing for the wrong reason.
if cr.COMPOSE_REGISTRY is not cr._CORE_ENTRIES:
    fail.append("COMPOSE_REGISTRY is no longer _CORE_ENTRIES — the isolation is gone")
if cr.DEFAULTS is not cr._CORE_DEFAULTS:
    fail.append("DEFAULTS is no longer _CORE_DEFAULTS — a local row could become a default")

print("FAIL:" + "; ".join(fail) if fail else "OK")
PY
)"
case "$out" in
  OK) echo "  ok   local entries are invisible to default dispatch" ;;
  FAIL:*) echo "  FAIL ${out#FAIL:}"; rc=1 ;;
  *) echo "  FAIL guard could not run: $out"; rc=1 ;;
esac

# 3. The dispatcher must not consult the merged view at all.
if command grep -q "get_registry" scripts/lib/registry-emit.sh \
   && sed -n '/^x_default_dispatch()/,/^}/p' scripts/lib/registry-emit.sh | command grep -q "get_registry"; then
  echo "  FAIL x_default_dispatch consults get_registry (the MERGED view)"; rc=1
else
  echo "  ok   x_default_dispatch reads only core-derived sets"
fi

[[ "$rc" == "0" ]] && echo "PASS: a local entry can never be a curated default" \
                   || echo "FAIL: local/default isolation regression"
exit "$rc"
