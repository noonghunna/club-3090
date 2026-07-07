#!/usr/bin/env bash
# pod.sh — GPU-pod management (#610 Phase A′).
#
# A "pod" = a named model on a chosen GPU set + port (an estate instance).
# This is the ergonomic front over the estate CLI (estate_cli.py owns the
# schema, validate_estate, kv-calc fit, and boot/down — ONE validation path
# shared with hand-written estate files and, later, the c3 wizard).
#
# Usage:
#   bash scripts/pod.sh create <name> --gpus 1,2 --slug vllm/dual [--port N]
#   bash scripts/pod.sh list [--json]
#   bash scripts/pod.sh status [--json]
#   bash scripts/pod.sh up <name>        # boot just this pod
#   bash scripts/pod.sh down <name>      # stop just this pod
#   bash scripts/pod.sh rm <name>        # remove from the estate file
#
# The artifact is the estate file (default scripts/lib/profiles/estate.yml;
# override with --file). GPU indices stay index-based in the file and are
# resolved to UUIDs at boot (#610 Phase A) so pods land on the cards they
# claimed on both container runtimes (classic nvidia + CDI/NixOS).
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ESTATE_CLI="${ROOT_DIR}/scripts/lib/profiles/estate_cli.py"

usage() {
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

cmd="${1:-}"
[[ -n "$cmd" ]] || { usage; exit 2; }
shift || true

case "$cmd" in
  create|list|status|rm)
    exec python3 "$ESTATE_CLI" "$cmd" "$@"
    ;;
  up)
    name="${1:-}"; shift || true
    [[ -n "$name" ]] || { echo "pod.sh up <name>" >&2; exit 2; }
    exec python3 "$ESTATE_CLI" boot --only "$name" "$@"
    ;;
  down)
    name="${1:-}"; shift || true
    [[ -n "$name" ]] || { echo "pod.sh down <name>" >&2; exit 2; }
    exec python3 "$ESTATE_CLI" down --only "$name" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "pod.sh: unknown command '$cmd'" >&2
    usage
    exit 2
    ;;
esac
