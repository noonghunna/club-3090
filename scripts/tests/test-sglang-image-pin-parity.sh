#!/usr/bin/env bash
# The sglang engine profile's install.spec is NOT injected by the launchers
# (launch.sh::export_variant_engine_pin handles vllm/* and beellama/* only), so the
# compose's own ${SGLANG_IMAGE:-...} default is what actually serves. That makes the
# two independent sources of truth, and a pin bump that edits only one is a silent
# no-op. This gate asserts they agree.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
FAIL=0

SPEC="$(python3 -c "
import yaml
print(yaml.safe_load(open('scripts/lib/profiles/engines/sglang-stable.yml'))['install']['spec'])
")" || { echo "FAIL: cannot read sglang-stable.yml install.spec"; exit 1; }

mapfile -t COMPOSES < <(python3 -c "
import sys; sys.path.insert(0,'scripts/lib/profiles')
from compose_registry import get_registry
for k,e in sorted(get_registry().items()):
    if k.startswith('sgl/'):
        print(e.get('compose_path') if isinstance(e,dict) else e.compose_path)
")
if [ "${#COMPOSES[@]}" -eq 0 ]; then
  echo "FAIL: no sgl/ slugs found — this gate must not pass vacuously"; exit 1
fi

for c in "${COMPOSES[@]}"; do
  [ -f "$c" ] || { echo "FAIL: compose missing: $c"; FAIL=1; continue; }
  got="$(command grep -oE 'image: \$\{SGLANG_IMAGE:-[^}]+\}' "$c" | head -1 | sed 's/.*:-//; s/}$//')"
  if [ -z "$got" ]; then
    echo "FAIL: $c has no \${SGLANG_IMAGE:-...} default"; FAIL=1
  elif [ "$got" != "$SPEC" ]; then
    echo "FAIL: $c pins '$got' but engines/sglang-stable.yml install.spec is '$SPEC'"
    echo "  Fix: bump BOTH — the engine profile spec is not injected for sgl/* slugs."
    FAIL=1
  fi
done

if [ "$FAIL" -eq 0 ]; then
  echo "PASS: ${#COMPOSES[@]} sgl composes all pin $SPEC, matching install.spec"
else
  echo "FAIL: sglang image pin parity"; exit 1
fi
