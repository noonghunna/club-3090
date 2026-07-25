#!/usr/bin/env bash
# test-compose-bind-host.sh — every compose that publishes a port MUST route the
# host side through ${BIND_HOST:-0.0.0.0}. This is the contract already emitted by
# scripts/lib/generate_compose.py and documented in docs/ADDING_MODELS.md; the
# hand-written composes had drifted off it.
#
# Why it matters: none of the engines we ship (vLLM, llama.cpp, ik_llama,
# beellama, SGLang) have auth. A compose with a bare "ports: - ${PORT}:8000"
# publishes on 0.0.0.0 with no way to narrow it, so an operator on a shared or
# untrusted LAN has to patch the file locally and re-patch after every update.
# With the knob present, BIND_HOST=127.0.0.1 in .env is enough.
#
# The default stays 0.0.0.0, so this guard does NOT assert a loopback default —
# only that the operator has the choice.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fails=0
checked=0

while IFS= read -r f; do
  # Walk the list items directly under the compose's `ports:` key.
  while IFS= read -r m; do
    [ -n "$m" ] || continue
    checked=$((checked+1))
    case "$m" in
      *BIND_HOST*) ;;
      *)
        echo "FAIL: $f publishes a port without a BIND_HOST knob:" >&2
        echo "        $(printf '%s' "$m" | sed 's/^[[:space:]]*//')" >&2
        echo "      expected the host side to route through \${BIND_HOST:-0.0.0.0}" >&2
        fails=$((fails+1))
        ;;
    esac
  done <<EOF
$(awk '
  /^[[:space:]]*ports:[[:space:]]*$/ { inports=1; next }
  inports {
    if ($0 ~ /^[[:space:]]*-[[:space:]]/) { print; next }
    inports=0
  }
' "$f")
EOF
done < <(grep -rlE '^[[:space:]]*ports:[[:space:]]*$' models --include='*.yml' 2>/dev/null | grep -v '/_archive/' | sort || true)

# The generator is the source of this contract — if it stops emitting the knob,
# every newly generated compose regresses silently.
grep -q 'BIND_HOST:-0.0.0.0' scripts/lib/generate_compose.py \
  || { echo "FAIL: generate_compose.py no longer emits the \${BIND_HOST:-0.0.0.0} prefix" >&2; fails=$((fails+1)); }

if [[ $fails -gt 0 ]]; then
  echo "$fails BIND_HOST check(s) failed" >&2
  exit 1
fi
echo "PASS: $checked published port mapping(s) across all live composes route through BIND_HOST"
