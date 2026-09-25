#!/usr/bin/env bash
# test-replica-catalog-order — a replica model's slugs must appear in the SAME order as its base model's.
#
# WHY THIS TEST EXISTS
# --------------------
# c3's default catalog view (and `switch.sh --list`) groups by model and keeps each model's slugs in
# REGISTRY order — there is no secondary sort. The ThinkingCap replicas (#1395) were registered
# topology-first (all dual tiers, then multi4, then multi8, vLLM before SGLang) while qwen3.8-27b is
# engine → tier family (max… then fast…) → topology, so the same 29 lanes read in a different order
# under each model. Fixed 2026-09-25 by reordering the registry; this keeps a future replica slug from
# landing out of place. Each pair maps a replica slug to its base twin by a slug-token swap; every
# replica slug must have a twin, and the replica sequence must equal the base sequence restricted
# to those twins.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

# replica-model  base-model  replica-slug-token  base-slug-token
PAIRS=(
  "thinkingcap-qwen3.8-27b qwen3.8-27b thinkingcap38-27b qwen38-27b"
)

json="$(bash scripts/lib/registry-emit.sh --json 2>/dev/null)"
[[ -n "$json" ]] || { echo "✗ registry-emit.sh --json produced nothing" >&2; exit 1; }

fails=0
for pair in "${PAIRS[@]}"; do
  read -r replica base rtok btok <<<"$pair"
  out="$(printf '%s' "$json" | python3 -c '
import json, sys
replica, base, rtok, btok = sys.argv[1:5]
variants = json.load(sys.stdin)["variants"]
rep = [v["slug"] for v in variants if v.get("model") == replica]
bas = [v["slug"] for v in variants if v.get("model") == base]
if not rep or not bas:
    print(f"FAIL {replica} / {base}: model missing from the registry (replica {len(rep)}, base {len(bas)} slugs)")
    sys.exit()
twins = [s.replace(rtok, btok) for s in rep]
orphans = [r for r, t in zip(rep, twins) if t not in set(bas)]
for o in orphans:
    print(f"FAIL {o}: no {base} twin ({o.replace(rtok, btok)}) to take its order from")
want = [s for s in bas if s in set(twins)]
if not orphans and twins != want:
    for i, (got, exp) in enumerate(zip(twins, want)):
        if got != exp:
            print(f"FAIL {replica}: position {i + 1} is {got.replace(btok, rtok)}, its {base} twin order says {exp.replace(btok, rtok)}")
            break
print(f"OK {replica}: {len(rep)} slugs mirror {base}")
' "$replica" "$base" "$rtok" "$btok")"
  while IFS= read -r line; do
    case "$line" in
      FAIL*) echo "✗ ${line#FAIL }" >&2; fails=$((fails+1)) ;;
      OK*)   ok_line="${line#OK }" ;;
    esac
  done <<<"$out"
done

if [[ "$fails" -gt 0 ]]; then
  echo "test-replica-catalog-order: $fails failure(s)" >&2
  exit 1
fi
echo "test-replica-catalog-order: ok (${ok_line:-no pairs})"
