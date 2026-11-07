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
#   bash scripts/pod.sh create <name> --gpus 1 --slug <slug> --up   # create, then boot it
#   bash scripts/pod.sh list [--json]
#   bash scripts/pod.sh status [--json]
#   bash scripts/pod.sh up <name>        # boot just this pod
#   bash scripts/pod.sh down <name>      # stop just this pod
#   bash scripts/pod.sh rm <name>        # remove from the estate file
#
# The artifact is the estate file (default ~/.club3090/estate.yml; override
# with --file). Booting another file with --file also copies it to the
# default path, which c3 and report.sh read. GPU indices stay index-based
# in the file and are resolved to UUIDs at boot (#610 Phase A) so pods land
# on the cards they claimed on both container runtimes (classic nvidia +
# CDI/NixOS).
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"
# Dot decimals whatever the user's locale: under es_ES / de_DE mawk read "1.65" as 1 and bash printf rejected "9.0" (#1926, #159).
# LC_ALL would override LC_NUMERIC, so its value moves to the other categories first. Guarded by test-numeric-locale.sh.
[[ -z "${LC_ALL:-}" ]] || { for _lc in LANG LC_CTYPE LC_COLLATE LC_MESSAGES LC_TIME LC_MONETARY; do export "$_lc=$LC_ALL"; done; unset LC_ALL _lc; }; export LC_NUMERIC=C
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# ESTATE_HELPER points pod.sh at another estate CLI, as it does launch.sh: the tests'
# stub, so `create --up` is checked without booting a model.
ESTATE_CLI="${ESTATE_HELPER:-${ROOT_DIR}/scripts/lib/profiles/estate_cli.py}"

usage() {
  # The header comment only: every line from 2 up to the first non-comment line.
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}

cmd="${1:-}"
[[ -n "$cmd" ]] || { usage; exit 2; }
shift || true

case "$cmd" in
  create)
    # --up (#1939, c3's Catalog "Add as pod"): boot the new pod right after creating
    # it, as ONE command, so a caller gates one write on the pod's cards. A create
    # that fails boots nothing; a boot that fails leaves the pod in the file, down.
    up=0
    args=()
    file_args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --up) up=1; shift ;;
        --file)
          args+=("$1" "${2:-}"); file_args=(--file "${2:-}"); shift
          if [[ $# -gt 0 ]]; then shift; fi ;;
        --file=*) args+=("$1"); file_args=("$1"); shift ;;
        *) args+=("$1"); shift ;;
      esac
    done
    if [[ $up -eq 0 ]]; then
      exec python3 "$ESTATE_CLI" create "${args[@]}"
    fi
    name="${args[0]:-}"
    if [[ -z "$name" || "$name" == -* ]]; then
      echo "pod.sh create <name> … --up: give the pod's name first" >&2
      exit 2
    fi
    python3 "$ESTATE_CLI" create "${args[@]}" || exit $?
    exec python3 "$ESTATE_CLI" boot --only "$name" "${file_args[@]}"
    ;;
  list|status|rm)
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
