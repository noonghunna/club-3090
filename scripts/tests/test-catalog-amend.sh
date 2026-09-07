#!/usr/bin/env bash
# Guard: `catalog.sh rename` / `update` amend the LOCAL layer only (#1202 P7).
#
# `rename` is not a convenience. #1202 decided a local slug colliding with a
# newly-shipped curated one is SHADOWED — kept and marked "so the user can rename
# it" — and then shipped no rename, so the advertised remedy was unreachable.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
rc=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts" "$TMP/tools"
cp -r scripts/lib "$TMP/scripts/"; cp -r tools/tui-core "$TMP/tools/"

printf '{"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":65536}\n' > "$TMP/cfg.json"
cat > "$TMP/c.yml" <<'YML'
services:
  mine:
    image: ghcr.io/someone/their-own-engine:v1
    ports: ["${PORT:-8199}:8199"]
    command: >
      /app/llama-server -m /models/m.gguf -a amend-probe -c 65536 -ctk q8_0 -ts 1,1
YML
scripts/catalog.sh register --compose "$TMP/c.yml" --engine their-engine --engine-type llama.cpp \
  --weights "$TMP/cfg.json" --root "$TMP" -y >/dev/null 2>&1 \
  || { echo "  FAIL fixture register failed; nothing below proves anything"; exit 1; }
CORE="$(python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
print(next(iter(COMPOSE_REGISTRY)))")"
REG="$TMP/scripts/lib/profiles-local/registry.local.json"
before_core="$(find scripts/lib/profiles -type f | sort | xargs -r md5sum | md5sum)"

chk() { # name, expect-refusal(0/1), cmd...
  local name="$1" want="$2"; shift 2
  "$@" >/dev/null 2>&1; local code=$?
  if [[ "$want" == "1" && "$code" -eq 0 ]]; then echo "  FAIL $name was ACCEPTED"; rc=1
  elif [[ "$want" == "0" && "$code" -ne 0 ]]; then echo "  FAIL $name failed (exit $code)"; rc=1
  else echo "  ok   $name"; fi
}

chk "curated slug rename refused"      1 scripts/catalog.sh rename --slug "$CORE" --to mine/x --root "$TMP"
chk "curated slug update refused"      1 scripts/catalog.sh update --slug "$CORE" --set workload=fast-chat --root "$TMP"
chk "rename ONTO a curated slug refused" 1 scripts/catalog.sh rename --slug their-engine/amend-probe --to "$CORE" --root "$TMP"
chk "origin is not editable"           1 scripts/catalog.sh update --slug their-engine/amend-probe --set origin=core --root "$TMP"
chk "rename into an unknown engine refused" 1 scripts/catalog.sh rename --slug their-engine/amend-probe --to nope/amend-probe --root "$TMP"
chk "--dry-run rename"                 0 scripts/catalog.sh rename --slug their-engine/amend-probe --to their-engine/x --root "$TMP" --dry-run

if command grep -q "their-engine/amend-probe" "$REG"; then
  echo "  ok   --dry-run changed nothing"
else
  echo "  FAIL --dry-run mutated the registry"; rc=1
fi

chk "real rename"                      0 scripts/catalog.sh rename --slug their-engine/amend-probe --to their-engine/renamed --root "$TMP"
chk "real update"                      0 scripts/catalog.sh update --slug their-engine/renamed --set workload=fast-chat --root "$TMP"

# The slug's namespace IS the engine, so the two must never drift apart.
if python3 - "$REG" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
raise SystemExit(0 if all(k.split("/", 1)[0] == v.get("engine") for k, v in d.items()) else 1)
PY
then echo "  ok   slug namespace and entry.engine agree"
else echo "  FAIL slug namespace and entry.engine disagree after amend"; rc=1; fi

# The amended layer must still load, or the edit produced a broken catalog.
if (cd "$TMP" && python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.lib.profiles.compose_registry import load_local_registry
r = load_local_registry('.')
raise SystemExit(0 if r and next(iter(r.values())).get('origin') == 'local' else 1)") 2>/dev/null
then echo "  ok   amended layer still loads, origin preserved"
else echo "  FAIL amended layer does not load"; rc=1; fi

[[ "$before_core" == "$(find scripts/lib/profiles -type f | sort | xargs -r md5sum | md5sum)" ]] \
  || { echo "  FAIL the curated catalog changed"; rc=1; }

[[ "$rc" == "0" ]] && echo "PASS: rename/update amend only the local layer" \
                   || echo "FAIL: catalog amend regression"
exit "$rc"
