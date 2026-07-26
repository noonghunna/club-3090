#!/usr/bin/env bash
#
# generate-compose.sh — v0.8.0 #141 compose generator (PR #147).
#
# Emits a minimal-reproduction docker-compose for an in-scope (non-Genesis,
# vLLM-only) profile. Mission (locked decision #2): reproduce + flag, NEVER
# repair. The engine image is NEVER rewritten; a failed-drift-guard patch is
# NEVER wired; --trust-remote-code (a governed slot, locked §88) is NEVER
# blind-passed for an in-scope profile.
#
# Usage:
#   scripts/generate-compose.sh --profile vllm/minimal [--out FILE]
#   scripts/generate-compose.sh --profile vllm/gemma-int8-mtp --accept-degraded
#   scripts/generate-compose.sh --model gemma-4-31b --engine vllm-gemma-stable
#       # convenience tuple: prints candidate --profile values, exits non-zero
#
# All decision logic lives in scripts/lib/generate_compose.py (this is a
# thin argv pass-through, matching the diagnose-profile.sh pattern).
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
exec python3 "${ROOT_DIR}/scripts/lib/generate_compose.py" "$@"
