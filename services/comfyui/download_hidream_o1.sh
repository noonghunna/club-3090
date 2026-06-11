#!/usr/bin/env bash
# Downloads the HiDream-O1-Image-Dev-2604 (fp8) image-generation model for the Studio
# image lane into the ComfyUI models tree.
#
# HiDream-O1 is a 9B *pixel-level unified transformer* (no separate VAE / text encoder —
# the whole HF folder is the model). It is NOT supported by ComfyUI natively; it needs the
# `Saganaki22/HiDream_O1-ComfyUI` custom node (cloned by services/comfyui/entrypoint.sh).
# The Dev-2604 build runs CFG-off, 28-step. ~8.8 GB weights, ~10 GB VRAM at 1024² — single
# RTX 3090, coexists with the Studio director on GPU0. AA #1 single-model open-weight T2I.
#
# Run:  ./download_hidream_o1.sh          (foreground)
#       nohup ./download_hidream_o1.sh > /tmp/hidream-o1-dl.log 2>&1 &   (background)
#
# Lands the complete model folder where the HiDream O1 loader looks (it scans
# diffusion_models/ for a folder matching a canonical name):
#   models/diffusion_models/HiDream-O1-Image-Dev-2604-FP8/{model.safetensors,config.json,...}
set -uo pipefail

ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
DEST="$ROOT/diffusion_models/HiDream-O1-Image-Dev-2604-FP8"   # folder name must match the loader's canonical choice
REPO="drbaph/HiDream-O1-Image-Dev-2604-FP8"

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' (huggingface_hub CLI) not found. pip install -U huggingface_hub" >&2; exit 1; }
mkdir -p "$DEST"

echo "=== HiDream-O1-Image-Dev-2604 fp8 (~8.8 GB, complete folder) ==="
# Whole repo into a single folder (model.safetensors + tokenizer/config/chat_template/etc).
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}" hf download "$REPO" --local-dir "$DEST"

echo "=== done -> $DEST ==="
ls -lh "$DEST"/model.safetensors 2>/dev/null || echo "WARN: model.safetensors missing — check the download"
