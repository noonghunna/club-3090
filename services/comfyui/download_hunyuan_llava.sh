#!/usr/bin/env bash
set -uo pipefail
# Resolve COMFYUI_MODELS_DIR from MODEL_DIR/.env so a standalone run matches
# setup-ai-studio.sh instead of falling back to the dev-rig /mnt path (issue #503).
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/comfyui-paths.sh"
c3_ensure_comfy_models_dir   # fail fast if the models dir isn't writable (#503)
ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
echo "[$(date +%H:%M:%S)] Downloading llava_llama3_fp8_scaled.safetensors (~9 GB)..."
hf download Comfy-Org/HunyuanVideo_repackaged \
    split_files/text_encoders/llava_llama3_fp8_scaled.safetensors \
    --local-dir "$ROOT/_hunyuan_staging"
mkdir -p "$ROOT/text_encoders"
[ -f "$ROOT/_hunyuan_staging/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors" ] && \
    mv -n "$ROOT/_hunyuan_staging/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors" "$ROOT/text_encoders/"
rm -rf "$ROOT/_hunyuan_staging"
echo "[$(date +%H:%M:%S)] Done."
ls -lh "$ROOT/text_encoders/llava_llama3_fp8_scaled.safetensors"
