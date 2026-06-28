#!/usr/bin/env bash
# Downloads the Krea 2 Turbo (fp8) image lane into the ComfyUI models tree. Krea 2 is a
# 12B dense DiT; the "turbo" schedule is the fast few-step variant. ~13 GB DiT on GPU.
#
# Requires ComfyUI >= v0.26.0 — native Krea2 support landed in #14589 (the entrypoint pin
# is bumped to v0.26.0 in the same change as this script). Older pins only expose the
# CLOUD Krea2ImageNode (local load fails "Could not detect model type").
#
# The text encoder is Qwen3-VL-4B (NOT the qwen_3_4b that Z-Image uses); the VAE is the
# Qwen-Image VAE. All three ship from the official Comfy-Org/Krea-2 repackage, already
# nested in diffusion_models/ text_encoders/ vae/ — so they land where the loaders look.
#
# Run:  ./download_krea.sh          (foreground)
#       nohup ./download_krea.sh > /tmp/krea-dl.log 2>&1 &   (background)
#
# Lands files where the krea2 graph looks:
#   models/diffusion_models/krea2_turbo_fp8_scaled.safetensors
#   models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
#   models/vae/qwen_image_vae.safetensors
set -uo pipefail

# Resolve COMFYUI_MODELS_DIR from MODEL_DIR/.env so a standalone run matches
# setup-ai-studio.sh instead of falling back to the dev-rig /mnt path (issue #503).
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/comfyui-paths.sh"
ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
export HF_HUB_DISABLE_XET=1
LOG_TS() { date +%H:%M:%S; }
log()  { echo "[$(LOG_TS)] $*"; }
step() { log ""; log "=== $* ==="; }

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' (huggingface_hub CLI) not found. pip install -U huggingface_hub" >&2; exit 1; }
mkdir -p "$ROOT/diffusion_models" "$ROOT/text_encoders" "$ROOT/vae"

# Comfy-Org/Krea-2 nests files under diffusion_models/ text_encoders/ vae/ — download with
# the repo path into $ROOT so each lands in the matching subdir the loaders expect.
step "1/3  Krea 2 Turbo fp8 DiT (~13.14 GB)"
hf download Comfy-Org/Krea-2 \
    diffusion_models/krea2_turbo_fp8_scaled.safetensors \
    --local-dir "$ROOT"

step "2/3  Qwen3-VL-4B fp8 text encoder (~5.24 GB)"
hf download Comfy-Org/Krea-2 \
    text_encoders/qwen3vl_4b_fp8_scaled.safetensors \
    --local-dir "$ROOT"

step "3/3  Qwen-Image VAE (~0.25 GB)"
hf download Comfy-Org/Krea-2 \
    vae/qwen_image_vae.safetensors \
    --local-dir "$ROOT"

step "DONE — Krea 2 Turbo set in $ROOT"
ls -lhL "$ROOT/diffusion_models/krea2_turbo_fp8_scaled.safetensors" \
        "$ROOT/text_encoders/qwen3vl_4b_fp8_scaled.safetensors" \
        "$ROOT/vae/qwen_image_vae.safetensors" 2>/dev/null | awk '{print "  "$5"  "$NF}'
