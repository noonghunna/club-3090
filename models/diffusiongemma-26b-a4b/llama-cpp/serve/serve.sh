#!/usr/bin/env bash
# ===========================================================================
# Launch the DiffusionGemma-26B-A4B OpenAI shim on the host.
#
# This is the WORKING serving path on a single 3090 today (Docker is not the
# primary path — see compose/single/q4-k-m/openai-shim.yml for the gated
# container variant). It runs the FastAPI shim, which spawns a patched
# `llama-diffusion-cli` (DG_NDJSON=1) as a resident backend.
#
# Prereqs:
#   - Build the patched CLI from llama.cpp draft PR #24423 + our patch
#     (see ../patches/README.md). Point DG_CLI_BIN at the binary.
#   - The Q4_K_M GGUF under models-cache/ (scripts pull it; see README).
#
# Usage:
#   DG_CLI_BIN=/path/to/llama-diffusion-cli bash serve.sh
#   # then: curl http://localhost:8060/v1/models
# ===========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"

# --- resolve inputs (override via env) -------------------------------------
DG_BUILD="${DG_BUILD:-$HOME/diffusiongemma-build/llama.cpp}"
export DG_CLI_BIN="${DG_CLI_BIN:-$DG_BUILD/build/bin/llama-diffusion-cli}"
export DG_GGUF="${DG_GGUF:-$REPO_ROOT/models-cache/diffusiongemma-26b-a4b-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
export DG_PORT="${DG_PORT:-8060}"
export DG_NGL="${DG_NGL:-99}"
export DG_CTX="${DG_CTX:-3072}"
export DG_MAXGEN="${DG_MAXGEN:-1024}"
export DG_MODEL_ID="${DG_MODEL_ID:-diffusiongemma-26b-a4b}"

[ -x "$DG_CLI_BIN" ] || { echo "ERROR: DG_CLI_BIN not executable: $DG_CLI_BIN" >&2;
  echo "       Build it (see ../patches/README.md) or set DG_CLI_BIN." >&2; exit 1; }
[ -f "$DG_GGUF" ]    || { echo "ERROR: GGUF not found: $DG_GGUF" >&2; exit 1; }

# --- shim venv (fastapi + uvicorn) -----------------------------------------
VENV="${DG_SHIM_VENV:-$HERE/.venv}"
if [ ! -d "$VENV" ]; then
  echo "[serve] creating shim venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "fastapi>=0.110" "uvicorn>=0.29" "pydantic>=2"
fi

echo "[serve] DiffusionGemma shim -> http://0.0.0.0:$DG_PORT/v1  (ctx=$DG_CTX, ngl=$DG_NGL)"
echo "[serve] backend: $DG_CLI_BIN"
echo "[serve] weights: $DG_GGUF"
exec "$VENV/bin/python" "$HERE/diffusion_openai_server.py"
