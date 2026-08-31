#!/usr/bin/env bash
# test-compose-nparallel-knob.sh — `-np` must be operator-settable, not hardcoded.
#
# WHY: llama.cpp's `-np` sets the server's slot count. Shipped composes passed a
# literal `1`, and LLAMA_ARG_N_PARALLEL CANNOT rescue that: common/arg.cpp applies
# env vars (~L783) BEFORE parsing argv (~L808), so the command line always wins.
# The server therefore ran with one slot and concurrent requests queued serially —
# every "concurrency" measurement was really measuring queuing. Measured on GLM
# 2026-08-30: 4 streams gave 1.92x aggregate (17.18 -> 33.03 tok/s), so a
# single-slot default leaves ~2x on the table for multi-user serving.
#
# This is the third instance of one class (with THREADS not forwarded, and
# shmem_enabled vs enabled): a knob that exists upstream but is unreachable as
# shipped, where the surface looks right and the mechanism is inert. So this guard
# checks the DELIVERY PATH and carries a NEGATIVE CONTROL — asserting the literal
# is gone would pass on a compose that interpolates to the wrong thing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fails=0
checked=0

# --- 1. static: no compose may hardcode the slot count -----------------------
while IFS= read -r f; do
  [[ "$f" == *"/_archive/"* || "$f" == *.bak ]] && continue
  checked=$((checked+1))
  if ! grep -Eq "^\s*- '\\\$\{NPARALLEL:-[0-9]+\}'\s*$" "$f"; then
    echo "FAIL: $f hardcodes -np; use \"- '\${NPARALLEL:-1}'\" so operators can raise the slot count" >&2
    fails=$((fails+1))
  fi
done < <(grep -rl -e "- '-np'" models/*/*/compose --include='*.yml' 2>/dev/null || true)

[[ $checked -eq 0 ]] && { echo "FAIL: found no composes passing -np at all — the search is wrong, not the tree" >&2; exit 1; }

# --- 2. delivery path + NEGATIVE CONTROL -------------------------------------
# For a command arg, `docker compose config` resolution IS what reaches argv.
SAMPLE=models/glm-5.3-flash/llamacpp-club3090/compose/dual/unsloth-ud-iq3xxs/moecache.yml
if command -v docker >/dev/null 2>&1 && [[ -f "$SAMPLE" ]]; then
  eff() { MODEL_DIR="${MODEL_DIR:-/tmp}" ${1:+env NPARALLEL="$1"} docker compose -f "$SAMPLE" config 2>/dev/null \
            | grep -A1 -e '- -np' | tail -1 | sed -E 's/^[[:space:]]*-[[:space:]]*//' | tr -d '"'; }
  base=$(eff "");   set4=$(eff 4)
  if [[ "$base" != "1" ]]; then
    echo "FAIL: negative control — with NPARALLEL unset the effective -np is '$base', expected '1' (shipped behaviour must not change)" >&2
    fails=$((fails+1))
  fi
  if [[ "$set4" != "4" ]]; then
    echo "FAIL: delivery path — NPARALLEL=4 produced -np '$set4', expected '4' (the knob is inert)" >&2
    fails=$((fails+1))
  fi
else
  echo "NOTE: docker unavailable — static check only; the delivery-path leg did NOT run." >&2
fi

if [[ $fails -gt 0 ]]; then
  echo "$fails -np knob check(s) failed" >&2
  exit 1
fi
echo "PASS: $checked composes parameterise -np; NPARALLEL=4 -> 4 and unset -> 1 through docker compose config"
