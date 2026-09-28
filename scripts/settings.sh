#!/usr/bin/env bash
# settings.sh — see and change your club-3090 settings (club-3090#1466).
#
#   bash scripts/settings.sh show [--json] [--show-secrets]
#   bash scripts/settings.sh get KEY
#   bash scripts/settings.sh set KEY=VALUE...
#   bash scripts/settings.sh unset KEY...
#   bash scripts/settings.sh migrate [--dry-run]
#   bash scripts/settings.sh path
#   bash scripts/settings.sh caches [--remove-legacy] [--yes]
#   bash scripts/settings.sh compose-env-file [--out PATH]
#   bash scripts/settings.sh --help
#
# A thin wrapper. The store, the precedence, the writer and every message live in
# scripts/lib/club_config.py (settings_main), the one loader every launcher and c3
# read settings through, so this command can never disagree with them about a
# setting. This checkout is passed as the repo root, whose legacy .env is read last.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"   # the command runs python3 (#779)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[settings] ERROR: python3 is required (it reads and writes the settings files)." >&2
  exit 2
fi
# `caches` (#1466 phase 4): the compile caches and the KV disk tier — sizes, and the old
# in-repo caches this checkout kept. Its logic lives next to the code that places them.
if [[ "${1:-}" == caches ]]; then
  shift
  exec python3 "${ROOT_DIR}/scripts/lib/engine_cache.py" caches --root "${ROOT_DIR}" "$@"
fi
exec python3 "${ROOT_DIR}/scripts/lib/club_config.py" settings --root "${ROOT_DIR}" "$@"
