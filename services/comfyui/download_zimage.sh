#!/usr/bin/env bash
# Downloads the Z-Image-Turbo (fp8) uncensored image lane into the ComfyUI models
# tree. Z-Image is Alibaba's 6B Apache-licensed text-to-image model — trained
# permissive, fast (8-step cfg=1 turbo schedule), ~7 GB on GPU0.
#
# ComfyUI HEAD has native Z-Image support — no custom node. The text encoder is a
# Qwen3-4B (loaded via CLIPLoader type "lumina2"); VAE is the flux-style ae.
#
# Run:  ./download_zimage.sh          (foreground)
#       nohup ./download_zimage.sh > /tmp/zimage-dl.log 2>&1 &   (background)
#
# Lands files where the z_image_turbo.json graph looks:
#   models/diffusion_models/z-image-turbo-fp8-e4m3fn.safetensors
#   models/text_encoders/qwen_3_4b_fp8_mixed.safetensors
#   models/vae/ae.safetensors
set -uo pipefail

# Resolve COMFYUI_MODELS_DIR from MODEL_DIR/.env so a standalone run matches
# setup-ai-studio.sh instead of falling back to the dev-rig /mnt path (issue #503).
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/comfyui-paths.sh"
c3_ensure_comfy_models_dir   # fail fast if the models dir isn't writable (#503)
ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
export HF_HUB_DISABLE_XET=1
LOG_TS() { date +%H:%M:%S; }
log()  { echo "[$(LOG_TS)] $*"; }
step() { log ""; log "=== $* ==="; }

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' (huggingface_hub CLI) not found. pip install -U huggingface_hub" >&2; exit 1; }
mkdir -p "$ROOT/diffusion_models" "$ROOT/text_encoders" "$ROOT/vae"

step "1/3  Z-Image-Turbo fp8 transformer (~6.15 GB)"
hf download T5B/Z-Image-Turbo-FP8 \
    z-image-turbo-fp8-e4m3fn.safetensors \
    --local-dir "$ROOT/diffusion_models"

step "2/3  Qwen3-4B fp8 text encoder (~5.63 GB) + flux-style VAE (~0.34 GB)"
# This repo nests under split_files/. Stage then symlink to where the loaders look.
hf download Comfy-Org/z_image \
    split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors \
    split_files/vae/ae.safetensors \
    --local-dir "$ROOT"
if [ -f "$ROOT/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors" ] && [ ! -e "$ROOT/text_encoders/qwen_3_4b_fp8_mixed.safetensors" ]; then
    ln -s ../split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors "$ROOT/text_encoders/qwen_3_4b_fp8_mixed.safetensors"
fi

step "3/3  VAE (ae.safetensors — flux-style autoencoder, shared)"
# ae.safetensors is the standard flux-family VAE (also used by the Chroma lane). Only
# create the link if nothing already provides it (don't clobber an existing flux ae).
if [ ! -e "$ROOT/vae/ae.safetensors" ] && [ -f "$ROOT/split_files/vae/ae.safetensors" ]; then
    ln -s ../split_files/vae/ae.safetensors "$ROOT/vae/ae.safetensors"
fi

step "DONE — Z-Image-Turbo set in $ROOT"
ls -lhL "$ROOT/diffusion_models/z-image-turbo-fp8-e4m3fn.safetensors" \
        "$ROOT/text_encoders/qwen_3_4b_fp8_mixed.safetensors" \
        "$ROOT/vae/ae.safetensors" 2>/dev/null | awk '{print "  "$5"  "$NF}'
