#!/usr/bin/env bash
# Downloads Stable Audio Open 1.0 + its T5-base text encoder for the studio 🔊 SFX lane
# (text → sound effects / ambience, ≤47 s). Lands them exactly where the studio SFX
# workflow loads them (services/studio/studio_pipe.py WORKFLOWS["sfx"]):
#   checkpoints/stable-audio-open-1.0.safetensors   (CheckpointLoaderSimple — model + VAE)
#   clip/t5-base.safetensors                        (CLIPLoader type=stable_audio — text enc)
#
# ~5.3 GB total. Single-device GPU0 — coexists with the qwen director.
# Idempotent: skips whichever file is already on disk.  Run:  ./download_stable_audio.sh
#
# Note: hf can't rename on download, so each file is fetched into a hidden staging dir
# and symlinked to the name the workflow expects (keeps hf's resume/idempotency intact).
set -uo pipefail

# Resolve COMFYUI_MODELS_DIR from MODEL_DIR/.env so a standalone run matches
# setup-ai-studio.sh instead of falling back to the dev-rig /mnt path (issue #503).
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/comfyui-paths.sh"
c3_ensure_comfy_models_dir   # fail fast if the models dir isn't writable (#503)
ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' (huggingface_hub CLI) not found. pip install -U huggingface_hub" >&2; exit 1; }
mkdir -p "$ROOT/checkpoints" "$ROOT/clip"

if [ -e "$ROOT/checkpoints/stable-audio-open-1.0.safetensors" ]; then
    log "Stable Audio Open model already present  (skip)"
else
    log "=== Stable Audio Open 1.0 model (~4.85 GB) ==="
    hf download stabilityai/stable-audio-open-1.0 model.safetensors \
        --local-dir "$ROOT/checkpoints/.stable-audio-open"
    ln -sf .stable-audio-open/model.safetensors "$ROOT/checkpoints/stable-audio-open-1.0.safetensors"
fi

if [ -e "$ROOT/clip/t5-base.safetensors" ]; then
    log "t5-base text encoder already present  (skip)"
else
    log "=== T5-base text encoder for stable_audio (~0.9 GB) ==="
    # google-t5/t5-base is the canonical T5-base Stable Audio Open uses; ComfyUI's
    # stable_audio CLIPLoader extracts the encoder from it.
    hf download google-t5/t5-base model.safetensors --local-dir "$ROOT/clip/.t5-base"
    ln -sf .t5-base/model.safetensors "$ROOT/clip/t5-base.safetensors"
fi

log "DONE — 🔊 SFX lane (Stable Audio Open + t5-base)"
ls -lhL "$ROOT/checkpoints/stable-audio-open-1.0.safetensors" "$ROOT/clip/t5-base.safetensors" 2>/dev/null \
  | awk '{print "  "$5"  "$NF}'
