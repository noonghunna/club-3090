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

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"
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
