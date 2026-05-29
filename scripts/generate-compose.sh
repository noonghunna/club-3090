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
#   scripts/generate-compose.sh --profile vllm/gemma-int8 --accept-degraded
#   scripts/generate-compose.sh --model gemma-4-31b --engine vllm-nightly-full
#       # convenience tuple: prints candidate --profile values, exits non-zero
#
# All decision logic lives in scripts/lib/generate_compose.py (this is a
# thin argv pass-through, matching the diagnose-profile.sh pattern).
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${ROOT_DIR}/scripts/lib/generate_compose.py" "$@"
