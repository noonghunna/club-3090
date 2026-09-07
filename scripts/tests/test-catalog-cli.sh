#!/usr/bin/env bash
# Guard: scripts/catalog.sh is the front door to the LOCAL layer (#1202 P4).
#
# Pins the four properties that make it safe to hand a stranger:
#   1. an engine that cannot be inferred REFUSES and names the fix, rather than
#      writing engine="unknown" into the catalog — that user (their own engine
#      build) is precisely who the local layer exists for;
#   2. everything auto-filled is PRINTED before the write. A compose is read
#      mechanically and cannot know whether `-ts 1,1` is a layer split or tensor
#      parallelism in the catalog's sense, so a wrong value nobody saw is worse
#      than a prompt;
#   3. register -> unregister is a clean round trip in a throwaway root;
#   4. `unregister` cannot reach the curated catalog.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$PWD"
rc=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts" "$TMP/tools"
cp -r scripts/lib "$TMP/scripts/"
cp -r tools/tui-core "$TMP/tools/"

# Arch dims come from the weights. A fabricated config.json exercises that path
# without depending on a multi-GB GGUF existing on a contributor's machine.
cat > "$TMP/config.json" <<'JSON'
{"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2}
JSON
W="$TMP/config.json"

C="$TMP/byo.yml"
cat > "$C" <<'YML'
services:
  mine:
    image: ghcr.io/someone/their-own-engine:v1
    ports:
      - "${PORT:-8199}:8199"
    command: >
      /app/llama-server -m /models/m.gguf -a byo-probe -c 65536 -ctk q8_0 -ts 1,1
YML

# 1. unknown engine -> refuse, and say what to do
out="$(scripts/catalog.sh register --compose "$C" --root "$TMP" --dry-run -y 2>&1)"; code=$?
if [[ "$code" -eq 0 ]]; then
  echo "  FAIL an uninferable engine was accepted (would write engine=unknown)"; rc=1
elif [[ "$out" != *"--engine"* ]]; then
  echo "  FAIL refusal does not name the fix (--engine)"; rc=1
else
  echo "  ok   uninferable engine refused, fix named"
fi

# 2. derived values are shown
out="$(scripts/catalog.sh register --compose "$C" --engine their-engine --engine-type llama.cpp --weights "$W" --root "$TMP" --dry-run -y 2>&1)"
missing=""
for k in slug model engine workload max_ctx kv_format tp port; do
  [[ "$out" == *"$k"* ]] || missing="$missing $k"
done
if [[ -n "$missing" ]]; then
  echo "  FAIL derived values not shown before write:$missing"; rc=1
else
  echo "  ok   derived values printed for confirmation"
fi
[[ "$out" == *"dry-run"* ]] || { echo "  FAIL --dry-run did not reach the executor"; rc=1; }

# 2b. arch is REQUIRED and must be refused BEFORE the write — promote reads it
# with .get, so a missing arch passes its validation and then dies in the
# post-write re-check, leaving the layer written and broken ("NO ROLLBACK").
out="$(scripts/catalog.sh register --compose "$C" --engine their-engine --engine-type llama.cpp --root "$TMP" --dry-run -y 2>&1)"; code=$?
if [[ "$code" -eq 0 ]]; then
  echo "  FAIL registered with no arch dims (post-write re-check would break the layer)"; rc=1
elif [[ "$out" != *"--weights"* ]]; then
  echo "  FAIL arch refusal does not name the fix (--weights)"; rc=1
else
  echo "  ok   missing arch refused BEFORE writing, fix named"
fi

# 2c. an unknown engine needs a declared lineage, never a guess: `type` has no
# enum validation, so a wrong value silently costs the fork drafter compat.
out="$(scripts/catalog.sh register --compose "$C" --engine their-engine --weights "$W" --root "$TMP" --dry-run -y 2>&1)"
if [[ "$out" == *"--engine-type"* ]]; then
  echo "  ok   unknown lineage refused, --engine-type named"
else
  echo "  FAIL unknown engine lineage was guessed rather than refused"; rc=1
fi

# 3. round trip
if scripts/catalog.sh register --compose "$C" --engine their-engine --engine-type llama.cpp --weights "$W" --root "$TMP" -y >/dev/null 2>&1; then
  REG="$TMP/scripts/lib/profiles-local/registry.local.json"
  if command grep -q "their-engine/byo-probe" "$REG" 2>/dev/null; then
    echo "  ok   register wrote their-engine/byo-probe"
  else
    echo "  FAIL slug missing from registry.local.json"; rc=1
  fi
  # the engine profile must exist, and carry EVIDENCE not assumptions
  EP="$TMP/scripts/lib/profiles-local/engines.d/their-engine.yml"
  if [[ -f "$EP" ]]; then
    if command grep -q "type: llama.cpp" "$EP" && command grep -q "q8_0" "$EP"; then
      echo "  ok   engine profile written with evidenced type + kv format"
    else
      echo "  FAIL engine profile missing declared lineage or the compose's kv format"; rc=1
    fi
    if command grep -qE "^(supported_drafters|features|supported_model_families):" "$EP"; then
      echo "  FAIL engine profile claims capabilities the compose does not evidence"; rc=1
    else
      echo "  ok   engine profile claims nothing unverified"
    fi
  else
    echo "  FAIL no local engine profile written for an unknown engine"; rc=1
  fi

  if scripts/catalog.sh unregister --slug their-engine/byo-probe --root "$TMP" -y >/dev/null 2>&1; then
    if [[ -f "$REG" ]] && command grep -q "their-engine/byo-probe" "$REG" 2>/dev/null; then
      echo "  FAIL unregister left the slug behind"; rc=1
    else
      echo "  ok   unregister removed it (clean round trip)"
    fi
  else
    echo "  FAIL unregister failed"; rc=1
  fi
else
  echo "  FAIL register failed in a throwaway root"; rc=1
fi

# 4. the curated catalog is unreachable from the front door too
CORE="$(python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
print(next(iter(COMPOSE_REGISTRY)))")"
before="$(find scripts/lib/profiles -type f | sort | xargs -r md5sum | md5sum)"
if scripts/catalog.sh unregister --slug "$CORE" -y >/dev/null 2>&1; then
  echo "  FAIL catalog.sh unregister ACCEPTED the curated slug $CORE"; rc=1
else
  echo "  ok   curated slug refused through the front door"
fi
[[ "$before" == "$(find scripts/lib/profiles -type f | sort | xargs -r md5sum | md5sum)" ]] \
  || { echo "  FAIL the curated catalog changed"; rc=1; }

[[ "$rc" == "0" ]] && echo "PASS: catalog.sh registers, unregisters, and cannot reach core" \
                   || echo "FAIL: catalog.sh regression"
exit "$rc"
