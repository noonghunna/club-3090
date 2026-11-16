#!/usr/bin/env bash
# test-qengine-kv-bits.sh — QEngine's opt-in 4-bit KV cache reaches the engine, and stays off by default.
#
# WHY: QEngine 1.3.6 reads QE_KV_BITS (the 27B, libqengine.so) and FLASH_KV_BITS (Flash-Next,
# libqflash.so) straight from the container's environment; 4 stores the KV in 4 bits instead of FP8.
# Docker forwards only the variables a compose declares in `environment:`, so an undeclared knob
# boots clean and does nothing. And a declaration with a value (`- QE_KV_BITS=${QE_KV_BITS:-}`)
# would set it EMPTY on every default boot. So this guard checks, for every QEngine compose of
# each model:
#   1. the bare declaration (`- QE_KV_BITS` / `- FLASH_KV_BITS`), on the right model's composes;
#   2. the DELIVERY PATH through `docker compose config`, which resolves the environment as `up`
#      does: the value 4 reaches the service, with a NEGATIVE CONTROL that unset leaves it absent.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fails=0
checked=0
bad() { echo "FAIL: $*" >&2; fails=$((fails+1)); }

# knob <var> <compose glob>
check_family() {
  local var="$1" glob="$2" f n=0
  for f in $glob; do
    [[ -f "$f" ]] || continue
    n=$((n+1)); checked=$((checked+1))
    command grep -Eq "^\s*- ${var}\s*$" "$f" \
      || bad "$f does not declare '- ${var}' bare (undeclared: docker never forwards it; with a value: empty on every boot)"
  done
  [[ $n -gt 0 ]] || bad "no compose matched $glob — the search is wrong, not the tree"
}
check_family QE_KV_BITS "models/qwen3.8-27b/qengine/compose/*/redhat-int4/mtp.yml"
check_family FLASH_KV_BITS "models/qwen3.8-flash-next/qengine/compose/*/qengine-int4/mtp.yml"

# Each knob belongs to its own engine: the 27B's libqengine.so reads QE_KV_BITS, Flash-Next's
# libqflash.so FLASH_KV_BITS. Declaring the other one would be a knob that does nothing.
for f in models/qwen3.8-27b/qengine/compose/*/redhat-int4/mtp.yml; do
  command grep -Eq '^\s*- FLASH_KV_BITS\s*$' "$f" && bad "$f declares FLASH_KV_BITS, which the 27B engine never reads"
done
for f in models/qwen3.8-flash-next/qengine/compose/*/qengine-int4/mtp.yml; do
  command grep -Eq '^\s*- QE_KV_BITS\s*$' "$f" && bad "$f declares QE_KV_BITS, which the Flash-Next engine never reads"
done

# --- 2. delivery path + NEGATIVE CONTROL -------------------------------------
eff() {   # eff <compose> <var> [value]: the value the service gets, or <absent>
  local f="$1" var="$2" val="${3:-}"
  env -u "$var" MODEL_DIR="${MODEL_DIR:-/tmp}" ${val:+"$var=$val"} \
    docker compose -f "$f" config --format json 2>/dev/null \
    | python3 -c 'import json,sys
e=next(iter(json.load(sys.stdin)["services"].values())).get("environment") or {}
v=e.get(sys.argv[1]); print("<absent>" if v is None else v)' "$var"
}
if command -v docker >/dev/null 2>&1; then
  delivered=0
  for pair in "QE_KV_BITS|models/qwen3.8-27b/qengine/compose/*/redhat-int4/mtp.yml" \
              "FLASH_KV_BITS|models/qwen3.8-flash-next/qengine/compose/*/qengine-int4/mtp.yml"; do
    var="${pair%%|*}"
    for f in ${pair#*|}; do
      base="$(eff "$f" "$var")"; on="$(eff "$f" "$var" 4)"
      [[ "$base" == "<absent>" ]] || bad "negative control — $f gives the service $var='$base' with it unset; expected it absent (FP8, the default)"
      [[ "$on" == "4" ]] || bad "delivery path — $var=4 reached $f's service as '$on', expected '4' (the knob is inert)"
      delivered=$((delivered+1))
    done
  done
else
  echo "NOTE: docker unavailable — static check only; the delivery-path leg did NOT run." >&2
fi

if [[ $fails -gt 0 ]]; then
  echo "$fails qengine kv-bits check(s) failed" >&2
  exit 1
fi
echo "PASS: test-qengine-kv-bits — $checked compose(s) declare their own engine's KV-bits knob bare; 4 -> 4 and unset -> absent through docker compose config (${delivered:-0} rendered)"
