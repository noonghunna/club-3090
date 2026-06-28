#!/usr/bin/env bash
# Downloads the Wan2.2-Rapid-AllInOne (Mega NSFW v10, Q8 GGUF) uncensored video lane
# into the ComfyUI models tree. Wan2.2 is a 14B text-to-video model; the "Rapid
# AllInOne" build merges the high+low-noise experts and bakes a 4-step distill LoRA
# in → single model, single 4-step cfg=1 sampler (~3 min/clip on 2× 3090).
#
# Apache-licensed weights; the Mega-v10 fine-tune is uncensored. Text->video only
# (no synced audio, unlike LTX). umt5 text encoder + Wan 2.1 VAE.
#
# Run:  ./download_wan.sh          (foreground)
#       nohup ./download_wan.sh > /tmp/wan-dl.log 2>&1 &   (background)
#
# Lands files where the wan22_rapid.json graph looks:
#   models/unet/wan-rapid/Mega-v10/wan2.2-rapid-mega-aio-nsfw-v10-Q8_0.gguf
#   models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
#   models/vae/wan_2.1_vae.safetensors
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
mkdir -p "$ROOT/unet/wan-rapid" "$ROOT/text_encoders" "$ROOT/vae"

step "1/3  Wan2.2-Rapid-AllInOne Mega NSFW v10 Q8 GGUF (~18.65 GB)"
hf download befox/WAN2.2-14B-Rapid-AllInOne-GGUF \
    Mega-v10/wan2.2-rapid-mega-aio-nsfw-v10-Q8_0.gguf \
    --local-dir "$ROOT/unet/wan-rapid"

step "2/3  umt5-xxl fp8 text encoder (~6.7 GB)"
hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
    --local-dir "$ROOT"
if [ -f "$ROOT/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" ] && [ ! -e "$ROOT/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" ]; then
    ln -s ../split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors "$ROOT/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
fi

step "3/3  Wan 2.1 VAE (~0.25 GB)"
hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/vae/wan_2.1_vae.safetensors \
    --local-dir "$ROOT"
if [ -f "$ROOT/split_files/vae/wan_2.1_vae.safetensors" ] && [ ! -e "$ROOT/vae/wan_2.1_vae.safetensors" ]; then
    ln -s ../split_files/vae/wan_2.1_vae.safetensors "$ROOT/vae/wan_2.1_vae.safetensors"
fi

step "DONE — Wan2.2-Rapid set in $ROOT"
ls -lhL "$ROOT/unet/wan-rapid/Mega-v10/wan2.2-rapid-mega-aio-nsfw-v10-Q8_0.gguf" \
        "$ROOT/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
        "$ROOT/vae/wan_2.1_vae.safetensors" 2>/dev/null | awk '{print "  "$5"  "$NF}'
