#!/usr/bin/env bash
# Downloads the Kokoro-82M ONNX voice model for the studio-tts sidecar (Kokoro
# narration / voiceover on the video lanes). CPU inference via the kokoro-onnx
# library (installed in the studio-tts image); these are the two model files it
# mounts at runtime.
#
# Source: the kokoro-onnx project's release assets (the kokoro-v1.0.onnx /
# voices-v1.0.bin format the library loads — NOT the onnx-community HF repo's
# onnx/model.onnx layout). ~330 MB total.
#
# Run:  ./download_kokoro.sh
#
# Lands files where the studio-tts compose mounts them (KOKORO_DIR):
#   tts/kokoro/kokoro-v1.0.onnx
#   tts/kokoro/voices-v1.0.bin
set -uo pipefail

# Mirrors the studio-tts compose default (${KOKORO_DIR:-/mnt/models/comfyui/models/tts/kokoro}).
# Resolve COMFYUI_MODELS_DIR from MODEL_DIR/.env so a standalone run matches
# setup-ai-studio.sh instead of falling back to the dev-rig /mnt path (issue #503).
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/comfyui-paths.sh"
c3_ensure_comfy_models_dir   # fail fast if the models dir isn't writable (#503)
ROOT="${KOKORO_DIR:-${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}/tts/kokoro}"
REL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
LOG_TS() { date +%H:%M:%S; }
log()  { echo "[$(LOG_TS)] $*"; }
step() { log ""; log "=== $* ==="; }

command -v curl >/dev/null 2>&1 || { echo "ERROR: 'curl' not found." >&2; exit 1; }
mkdir -p "$ROOT"

fetch() {  # <filename>  — resumable, fail on HTTP error
    local f="$1"
    curl -fL --retry 3 -C - -o "$ROOT/$f" "$REL/$f"
}

step "1/2  Kokoro ONNX model (~310 MB)"
fetch kokoro-v1.0.onnx

step "2/2  Kokoro voices pack (~27 MB)"
fetch voices-v1.0.bin

log ""
log "Done → $ROOT  (kokoro-v1.0.onnx + voices-v1.0.bin)"
