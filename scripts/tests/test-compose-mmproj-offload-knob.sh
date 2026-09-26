#!/usr/bin/env bash
# test-compose-mmproj-offload-knob.sh — LLAMA_ARG_MMPROJ_OFFLOAD must reach llama-server
# on the composes that declare it, and must not break the default boot.
#
# WHY: #1380 fit q8_0 KV @215K on one card by keeping the vision projector in system
# RAM (`--no-mmproj-offload`). The compose had no way to ask for that: docker forwards
# only the vars declared in `environment:`, so the env var never arrived. Two ways to
# ship the knob look right and are wrong:
#   - `- LLAMA_ARG_MMPROJ_OFFLOAD=${LLAMA_ARG_MMPROJ_OFFLOAD:-}` sets it EMPTY on every
#     default boot, and llama.cpp rejects that ("invalid boolean value") before the
#     model loads, so the default boot dies.
#   - any `--mmproj-offload` / `--no-mmproj-offload` in `command:` wins over the env
#     (common/arg.cpp applies env vars before argv), so the knob boots clean and does
#     nothing.
# So this guard checks the declaration form, the argv, and the DELIVERY PATH, with a
# NEGATIVE CONTROL: unset must leave the container without the variable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fails=0
checked=0

# --- 1. static: bare declaration, and no argv flag that would override it ----
while IFS= read -r f; do
  [[ "$f" == *"/_archive/"* || "$f" == *.bak ]] && continue
  checked=$((checked+1))
  if ! command grep -Eq '^\s*- LLAMA_ARG_MMPROJ_OFFLOAD\s*$' "$f"; then
    echo "FAIL: $f declares LLAMA_ARG_MMPROJ_OFFLOAD with a value; use the bare '- LLAMA_ARG_MMPROJ_OFFLOAD' (an empty value kills the default boot)" >&2
    fails=$((fails+1))
  fi
  if command grep -Eq '^\s*[^#]*--(no-)?mmproj-offload' "$f"; then
    echo "FAIL: $f passes --mmproj-offload / --no-mmproj-offload in argv, which overrides LLAMA_ARG_MMPROJ_OFFLOAD" >&2
    fails=$((fails+1))
  fi
done < <(command grep -rlE '^\s*- LLAMA_ARG_MMPROJ_OFFLOAD' models/*/*/compose --include='*.yml' 2>/dev/null || true)

[[ $checked -eq 0 ]] && { echo "FAIL: found no composes declaring LLAMA_ARG_MMPROJ_OFFLOAD — the search is wrong, not the tree" >&2; exit 1; }

# --- 2. delivery path + NEGATIVE CONTROL -------------------------------------
# `docker compose config` resolves the container environment exactly as `up` does.
SAMPLE=models/qwen3.8-27b/llama-cpp/compose/single/unsloth-iq4xs/q4kv-vision.yml
if command -v docker >/dev/null 2>&1 && [[ -f "$SAMPLE" ]]; then
  eff() {
    env -u LLAMA_ARG_MMPROJ_OFFLOAD MODEL_DIR="${MODEL_DIR:-/tmp}" ${1:+LLAMA_ARG_MMPROJ_OFFLOAD="$1"} \
      docker compose -f "$SAMPLE" config --format json 2>/dev/null \
      | python3 -c 'import json,sys; e=next(iter(json.load(sys.stdin)["services"].values())).get("environment") or {}; v=e.get("LLAMA_ARG_MMPROJ_OFFLOAD"); print("<absent>" if v is None else v)'
  }
  base=$(eff "");   off=$(eff false)
  if [[ "$base" != "<absent>" ]]; then
    echo "FAIL: negative control — with the var unset the container gets LLAMA_ARG_MMPROJ_OFFLOAD='$base', expected it absent (the default boot must not change)" >&2
    fails=$((fails+1))
  fi
  if [[ "$off" != "false" ]]; then
    echo "FAIL: delivery path — LLAMA_ARG_MMPROJ_OFFLOAD=false reached the container as '$off', expected 'false' (the knob is inert)" >&2
    fails=$((fails+1))
  fi
else
  echo "NOTE: docker unavailable — static check only; the delivery-path leg did NOT run." >&2
fi

if [[ $fails -gt 0 ]]; then
  echo "$fails mmproj-offload knob check(s) failed" >&2
  exit 1
fi
echo "PASS: $checked compose(s) declare LLAMA_ARG_MMPROJ_OFFLOAD bare with no overriding argv; false -> false and unset -> absent through docker compose config"
