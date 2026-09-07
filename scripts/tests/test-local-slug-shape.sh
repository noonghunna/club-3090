#!/usr/bin/env bash
# Guard: local slugs use the same '<engine>/<name>' shape as curated rows, the
# legacy 'local/' namespace is refused with guidance, and a core collision is
# SHADOWED (core wins, local row marked) rather than fatal. (#1202 P3)
#
# The namespace was not just cosmetic — it occupied the ENGINE slot, so a local
# model could not record which engine it runs, and a user on their own build had
# nowhere to put it. Removing it also removes the *rename on publish*:
# export_pr.py used to compute the core slug by stripping the prefix.
#
# The shadow case could not be tested before this change: the namespace check
# refused a non-'local/' slug before the collision test was reachable.
set -uo pipefail
# Non-UTF-8 locales break python3 reads/writes on this rig (#599/#584).
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
rc=0
out="$(python3 - <<'PY' 2>&1
import inspect, json, sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, ".")
from scripts.lib.profiles import compose_registry as cr

fail = []
sig = inspect.signature(cr._entry)
req = [n for n, p in sig.parameters.items()
       if p.default is inspect._empty and p.kind is p.KEYWORD_ONLY]
core_slug = next(iter(cr.COMPOSE_REGISTRY))
tmpl = dict(cr.COMPOSE_REGISTRY[core_slug])
kw = {n: (tmpl[n] if n in tmpl else "composes/x/base.yml") for n in req}

root = Path(tempfile.mkdtemp())
try:
    (root / "scripts/lib/profiles-local").mkdir(parents=True)
    reg = root / "scripts/lib/profiles-local/registry.local.json"

    def load(d):
        reg.write_text(json.dumps(d))
        try:
            return cr.load_local_registry(root), None
        except cr.LocalRegistryError as e:
            return None, str(e)

    row = lambda mid: dict(kw, model=mid, status="experimental")

    _, err = load({"local/mine": row("m1")})
    if not err:
        fail.append("legacy 'local/' slug was accepted")
    elif "engine" not in err:
        fail.append("legacy refusal does not tell the user the new shape")

    got, err = load({"my-llamacpp/mine": row("m2")})
    if err:
        fail.append(f"'<engine>/<name>' refused: {err[:60]}")
    elif got["my-llamacpp/mine"].get("origin") != "local":
        fail.append("accepted local row not stamped origin=local")

    _, err = load({"noslash": row("m3")})
    if not err:
        fail.append("a slug with no '/' was accepted")

    # The shadow path — core wins, local row loaded AND marked.
    got, err = load({core_slug: row("m4")})
    if err:
        fail.append(f"core collision still raises instead of shadowing: {err[:50]}")
    else:
        if not got[core_slug].get("shadowed_by_core"):
            fail.append("shadowed local row is not marked shadowed_by_core")
        if cr.get_registry(root)[core_slug].get("origin") != "core":
            fail.append("local row shadowed a CORE slug in the merged view")
finally:
    shutil.rmtree(root)

print("FAIL:" + "; ".join(fail) if fail else "OK")
PY
)"
case "$out" in
  OK) echo "  ok   shape enforced, legacy refused, core collision shadowed" ;;
  FAIL:*) echo "  FAIL ${out#FAIL:}"; rc=1 ;;
  *) echo "  FAIL guard could not run: $out"; rc=1 ;;
esac

# No file may still DERIVE locality by stripping the prefix (demote.py:88 and
# export_pr.py:282 both did; that arithmetic silently returns the wrong value).
# Only DERIVATION counts — an assignment or a return. Using the stripped tail
# inside a refusal message ("Use '<engine>/<name>' instead") is fine: those sit
# in branches that have just proven the prefix is present, and they compute
# nothing the code relies on.
_deriv='(^|[^#])[[:space:]]*(return|[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=)[^#]*\[len\((_LOCAL_SLUG_PREFIX|_LOCAL_NS)\):\]'
if command grep -rnE "$_deriv" scripts/lib/profiles/*.py >/dev/null 2>&1; then
  echo "  FAIL a file still DERIVES a value by stripping the local prefix:"
  command grep -rnE "$_deriv" scripts/lib/profiles/*.py | sed 's/^/       /'
  rc=1
else
  echo "  ok   no prefix-stripping arithmetic remains"
fi

[[ "$rc" == "0" ]] && echo "PASS: local slugs share the curated shape" || echo "FAIL: hard-cut regression"
exit "$rc"
