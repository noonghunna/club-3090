#!/usr/bin/env bash
# Downloads the Ideogram-4 (fp8) image-generation model set for the image-studio
# bundle into the ComfyUI models tree. ComfyUI HEAD has native Ideogram-4 support
# (Ideogram4Scheduler / DualModelGuider) — no custom node needed.
#
# Two fp8 transformers (main + unconditional) + the Qwen3-VL-8B text encoder +
# flux2 VAE. ~27 GB total. Validated single-3090: 1024² ~18.5 GB / ~70 s.
#
# Run:  ./download_ideogram4.sh          (foreground)
#       nohup ./download_ideogram4.sh > /tmp/ideogram4-dl.log 2>&1 &   (background)
#
# Lands files where ComfyUI's loaders look:
#   models/diffusion_models/ideogram4_fp8_scaled.safetensors
#   models/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors
#   models/text_encoders/qwen3vl_8b_fp8_scaled.safetensors
#   models/vae/flux2-vae.safetensors
set -uo pipefail

# Resolve COMFYUI_MODELS_DIR from MODEL_DIR/.env so a standalone run matches
# setup-ai-studio.sh instead of falling back to the dev-rig /mnt path (issue #503).
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/comfyui-paths.sh"
ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
LOG_TS() { date +%H:%M:%S; }
log()  { echo "[$(LOG_TS)] $*"; }
step() { log ""; log "=== $* ==="; }

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' (huggingface_hub CLI) not found. pip install -U huggingface_hub" >&2; exit 1; }
mkdir -p "$ROOT/diffusion_models" "$ROOT/text_encoders" "$ROOT/vae"

step "1/4  Ideogram-4 fp8 transformer (main, ~8.7 GB)"
hf download Comfy-Org/Ideogram-4 \
    diffusion_models/ideogram4_fp8_scaled.safetensors \
    --local-dir "$ROOT"

step "2/4  Ideogram-4 fp8 transformer (unconditional, ~8.7 GB)"
hf download Comfy-Org/Ideogram-4 \
    diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors \
    --local-dir "$ROOT"

step "3/4  Qwen3-VL-8B fp8 text encoder (~9.9 GB)"
hf download Comfy-Org/Qwen3-VL \
    text_encoders/qwen3vl_8b_fp8_scaled.safetensors \
    --local-dir "$ROOT"

step "4/4  flux2 VAE (~321 MB)"
# This repo nests the VAE under split_files/vae/. Stage then flatten to vae/.
hf download Comfy-Org/flux2-dev \
    split_files/vae/flux2-vae.safetensors \
    --local-dir "$ROOT"
if [ -f "$ROOT/split_files/vae/flux2-vae.safetensors" ] && [ ! -e "$ROOT/vae/flux2-vae.safetensors" ]; then
    ln -s ../split_files/vae/flux2-vae.safetensors "$ROOT/vae/flux2-vae.safetensors"
fi

step "DONE — Ideogram-4 set in $ROOT"
ls -lh "$ROOT/diffusion_models/"ideogram4_*fp8_scaled.safetensors \
       "$ROOT/text_encoders/qwen3vl_8b_fp8_scaled.safetensors" \
       "$ROOT/vae/flux2-vae.safetensors" 2>/dev/null | awk '{print "  "$5"  "$NF}'
