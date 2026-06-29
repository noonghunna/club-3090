#!/usr/bin/env bash
set -uo pipefail
# Resolve COMFYUI_MODELS_DIR from MODEL_DIR/.env so a standalone run matches
# setup-ai-studio.sh instead of falling back to the dev-rig /mnt path (issue #503).
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/comfyui-paths.sh"
c3_ensure_comfy_models_dir   # fail fast if the models dir isn't writable (#503)
ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"

ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $*"; }

log "FLUX.2 1/4  DiT fp8mixed (~17 GB)"
hf download Comfy-Org/flux2-dev \
    split_files/diffusion_models/flux2_dev_fp8mixed.safetensors \
    --local-dir "$ROOT/_flux2_staging"

log "FLUX.2 2/4  text encoder Mistral-Small-3.1 fp8 (~14 GB)"
hf download Comfy-Org/flux2-dev \
    split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors \
    --local-dir "$ROOT/_flux2_staging"

log "FLUX.2 3/4  VAE"
hf download Comfy-Org/flux2-dev \
    split_files/vae/flux2-vae.safetensors \
    --local-dir "$ROOT/_flux2_staging"

log "FLUX.2 4/4  Turbo LoRA"
hf download Comfy-Org/flux2-dev \
    split_files/loras/Flux2TurboComfyv2.safetensors \
    --local-dir "$ROOT/_flux2_staging"

log "Moving into canonical paths"
mv -n  "$ROOT/_flux2_staging/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors" \
       "$ROOT/diffusion_models/flux2-dev/" 2>&1
mv -n  "$ROOT/_flux2_staging/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors" \
       "$ROOT/text_encoders/" 2>&1
mv -n  "$ROOT/_flux2_staging/split_files/vae/flux2-vae.safetensors" \
       "$ROOT/vae/flux2/" 2>&1
mv -n  "$ROOT/_flux2_staging/split_files/loras/Flux2TurboComfyv2.safetensors" \
       "$ROOT/loras/" 2>&1
rm -rf "$ROOT/_flux2_staging"

log "FLUX.2 download complete. Files:"
ls -lh "$ROOT/diffusion_models/flux2-dev/"  "$ROOT/text_encoders/mistral_3_small_flux2_fp8.safetensors" "$ROOT/vae/flux2/" "$ROOT/loras/Flux2TurboComfyv2.safetensors" 2>/dev/null
