#!/usr/bin/env bash
# test-setup-slug-companions.sh — `setup.sh <slug>` must fetch everything the slug needs.
#
# WHY: a slug's `weights_companions` (a drafter / vision projector its compose mounts)
# reached setup.sh only through the serve-cockpit's Download action. A CLI download
# of the model got the core weights and left a slug that crash-loops on a missing
# drafter (#1247). `setup.sh <slug>` now resolves the slug through the registry.
# This guard checks, for one slug per distinct (model, variant, companions) set:
#   - the primary key is <model>:<weights_variant> (what the cockpit sends);
#   - every companion is queued for download (or is already the primary / the
#     model's always-fetched drafter), and resolves in the weights catalog;
# plus the overrides and the unknown-slug refusal, as negative controls.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fails=0
bad() { echo "FAIL: $*" >&2; fails=$((fails+1)); }
dump() { SETUP_DUMP_KEYS=1 MODEL_DIR="${TMPDIR:-/tmp}" bash scripts/setup.sh "$@" 2>&1; }
val() { command grep -m1 "^$2=" <<<"$1" | cut -d= -f2-; }

# One representative slug per distinct companion set, straight from the registry.
mapfile -t CASES < <(python3 - <<'PY'
import os, sys
sys.path.insert(0, os.path.join("scripts", "lib", "profiles"))
from compose_registry import get_registry
seen = {}
for slug, e in sorted(get_registry().items()):
    comps = [c for c in (e.get("weights_companions") or []) if c]
    if not comps:
        continue
    model = e["model"]
    q = tuple(c if ":" in c else f"{model}:{c}" for c in comps)
    seen.setdefault((model, e["weights_variant"], q), slug)
for (model, variant, q), slug in seen.items():
    print("\t".join([slug, model, f"{model}:{variant}", " ".join(q)]))
PY
)
[[ ${#CASES[@]} -gt 0 ]] || { echo "FAIL: no registry slug declares weights_companions — the search is wrong, not the tree" >&2; exit 1; }

for row in "${CASES[@]}"; do
  IFS=$'\t' read -r slug model primary comps <<<"$row"
  out="$(dump "$slug")" || { bad "$slug: setup.sh <slug> exited non-zero: $(tail -1 <<<"$out")"; continue; }
  [[ "$(val "$out" slug)" == "$slug" ]]       || bad "$slug: dump slug='$(val "$out" slug)'"
  [[ "$(val "$out" model)" == "$model" ]]     || bad "$slug: model='$(val "$out" model)', expected '$model'"
  [[ "$(val "$out" primary)" == "$primary" ]] || bad "$slug: primary='$(val "$out" primary)', expected '$primary'"
  [[ "$(val "$out" companions)" == "$comps" ]] || bad "$slug: companions='$(val "$out" companions)', expected '$comps' (the cockpit's keys)"
  queued=" $(val "$out" extras) $(val "$out" always_draft) $(val "$out" primary) "
  for c in $comps; do
    [[ "$queued" == *" $c "* ]] || bad "$slug: companion $c is not queued for download"
    python3 scripts/lib/profiles/weights.py entry "$c" >/dev/null 2>&1 || bad "$slug: companion $c does not resolve in the weights catalog"
  done
  extras="$(val "$out" extras)"
  for k in $extras; do
    [[ "$(command grep -o -w -- "$k" <<<"$extras" | wc -l)" -eq 1 ]] || bad "$slug: $k queued more than once"
  done
done

# --- negative controls -------------------------------------------------------
IQ=llamacpp/qwen38-27b-single-iq4xs
out="$(WEIGHT_EXTRA_KEYS= dump "$IQ")"
[[ -z "$(val "$out" extras)" ]] || bad "WEIGHT_EXTRA_KEYS= must skip the companions, got extras='$(val "$out" extras)'"
out="$(WEIGHT_KEY=qwen3.8-27b:unsloth-q8kxl dump "$IQ")"
[[ "$(val "$out" primary)" == "qwen3.8-27b:unsloth-q8kxl" ]] || bad "an explicit WEIGHT_KEY must win, got primary='$(val "$out" primary)'"
out="$(dump qwen3.8-27b)"
[[ -z "$(val "$out" slug)" && -z "$(val "$out" companions)" ]] || bad "a model name must not resolve as a slug"
if out="$(dump vllm/no-such-slug)"; then
  bad "an unknown slug must fail"
else
  command grep -q "not a model name or a known launch slug" <<<"$out" || bad "unknown slug: missing the refusal message"
fi

if [[ $fails -gt 0 ]]; then
  echo "$fails setup.sh slug-companion check(s) failed" >&2
  exit 1
fi
echo "PASS: ${#CASES[@]} distinct companion sets — setup.sh <slug> queues each slug's weights + companions (cockpit keys, catalog-resolvable, no duplicates); overrides win; an unknown slug is refused"
