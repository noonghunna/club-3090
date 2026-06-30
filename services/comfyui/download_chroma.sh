#!/usr/bin/env bash
# Downloads the Chroma1-HD (fp8) uncensored image lane into the ComfyUI models
# tree. Chroma1-HD is a Flux-based, de-distilled, trained-uncensored text-to-image
# model (~9 GB on GPU0) — the "Sulphur for stills". Natural-language prompt +
# negative + real CFG, 26-step beta schedule.
#
# Needs no custom node (ComfyUI loads it via UNETLoader). Unlike HiDream-O1 it is
# NOT a single-folder model: the DiT, the t5xxl_fp16 text encoder, and the
# flux-family ae VAE are three separate files. The encoder + VAE are shared with
# the wider Flux ecosystem; this script fetches all three so the lane works on a
# fresh machine (the older setup relied on them already being on disk — issue #510).
#
# Run:  ./download_chroma.sh          (foreground)
#       nohup ./download_chroma.sh > /tmp/chroma-dl.log 2>&1 &   (background)
#
# Lands files where the chroma1_hd.json graph looks:
#   models/diffusion_models/Chroma1-HD-fp8mixed.safetensors
#   models/text_encoders/t5xxl_fp16.safetensors
#   models/vae/flux/ae.safetensors
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
mkdir -p "$ROOT/diffusion_models" "$ROOT/text_encoders" "$ROOT/vae/flux"

step "1/3  Chroma1-HD fp8mixed transformer (~9 GB)"
# Comfy-Org repackaged repo nests under split_files/. Stage then symlink to where the loader looks.
hf download Comfy-Org/Chroma1-HD_repackaged \
    split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors \
    --local-dir "$ROOT"
if [ -f "$ROOT/split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors" ] \
   && [ ! -e "$ROOT/diffusion_models/Chroma1-HD-fp8mixed.safetensors" ]; then
    ln -s ../split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors \
          "$ROOT/diffusion_models/Chroma1-HD-fp8mixed.safetensors"
fi

step "2/3  t5xxl_fp16 text encoder (~9.8 GB, shared Flux encoder)"
# Idempotent + shared: skip if another Flux lane already provided it.
if [ ! -e "$ROOT/text_encoders/t5xxl_fp16.safetensors" ]; then
    hf download comfyanonymous/flux_text_encoders \
        t5xxl_fp16.safetensors \
        --local-dir "$ROOT/text_encoders"
fi

step "3/3  Flux VAE (ae.safetensors — shared flux-family autoencoder, ~0.34 GB)"
# The graph looks for vae/flux/ae.safetensors. Reuse the staged flux-style ae if the
# z-image lane already fetched it; otherwise pull it (ungated, from the same repo).
if [ ! -f "$ROOT/split_files/vae/ae.safetensors" ]; then
    hf download Comfy-Org/z_image split_files/vae/ae.safetensors --local-dir "$ROOT"
fi
[ -e "$ROOT/vae/flux/ae.safetensors" ] || \
    ln -s ../../split_files/vae/ae.safetensors "$ROOT/vae/flux/ae.safetensors"

step "DONE — Chroma1-HD set in $ROOT"
ls -lhL "$ROOT/diffusion_models/Chroma1-HD-fp8mixed.safetensors" \
        "$ROOT/text_encoders/t5xxl_fp16.safetensors" \
        "$ROOT/vae/flux/ae.safetensors" 2>/dev/null | awk '{print "  "$5"  "$NF}'
