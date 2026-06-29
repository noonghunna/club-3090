#!/usr/bin/env bash
# Downloads the LTX-2.3 video model set for the video-studio bundle (the 🎬 LTX-2.3
# clean lane + the 🔓 Sulphur and 🔓 10Eros uncensored lanes — text/image → video +
# audio) into the ComfyUI models tree.
#
# The LTX-2.3 / director / gemma files were verified by byte-identical LFS-sha256
# match against their source repos (club-3090 audit). Sulphur and 10Eros are two
# LTX-2.3-22B-dev uncensored fine-tunes (shipped side by side so they can be compared);
# both run single-stage via the SHARED distill LoRA. Lands files exactly where the
# studio video workflows load them (services/studio/studio_pipe.py WORKFLOWS):
#   vae/ltx-2.3-22b-{distilled,dev}_{video,audio}_vae.safetensors
#   text_encoders/ltx-2.3-22b-{distilled,dev}_embeddings_connectors.safetensors
#   text_encoders/gemma_3_12B_it_fp8_scaled.safetensors        (LTX text encoder)
#   unet/ltx2.3/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q8_0.gguf   (🎬 LTX lane)
#   unet/sulphur-2/sulphur_dev-Q8_0.gguf                            (🔓 Sulphur lane)
#   unet/10eros/10Eros_v1-Q8_0.gguf                                 (🔓 10Eros lane)
#   loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors           (shared dev distill)
#
# ~82 GB total (two 22GB uncensored unets). HF_TOKEN MUST be set — the gemma encoder
# repo (Comfy-Org/ltx-2) is gated. Run:  ./download_video_models.sh  (or nohup … & for bg)
set -uo pipefail

# Resolve COMFYUI_MODELS_DIR from MODEL_DIR/.env so a standalone run matches
# setup-ai-studio.sh instead of falling back to the dev-rig /mnt path (issue #503).
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/comfyui-paths.sh"
c3_ensure_comfy_models_dir   # fail fast if the models dir isn't writable (#503)
ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
LOG_TS() { date +%H:%M:%S; }
log()  { echo "[$(LOG_TS)] $*"; }
step() { log ""; log "=== $* ==="; }

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' (huggingface_hub CLI) not found. pip install -U huggingface_hub" >&2; exit 1; }
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARN: HF_TOKEN not set — the gemma text-encoder repo (Comfy-Org/ltx-2) is gated and will 401." >&2
    echo "      export HF_TOKEN=hf_… (and accept the repo terms) before running, or that one file will fail." >&2
fi
mkdir -p "$ROOT/vae" "$ROOT/text_encoders" "$ROOT/unet/ltx2.3" "$ROOT/unet/sulphur-2" "$ROOT/unet/10eros" "$ROOT/loras"

step "1/7  LTX-2.3 VAEs — distilled + dev, video + audio (~1.3 GB)"
hf download unsloth/LTX-2.3-GGUF \
    vae/ltx-2.3-22b-distilled_video_vae.safetensors \
    vae/ltx-2.3-22b-distilled_audio_vae.safetensors \
    vae/ltx-2.3-22b-dev_video_vae.safetensors \
    vae/ltx-2.3-22b-dev_audio_vae.safetensors \
    --local-dir "$ROOT"

step "2/7  LTX-2.3 text-encoder connectors — distilled + dev (~2 GB)"
hf download unsloth/LTX-2.3-GGUF \
    text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
    text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors \
    --local-dir "$ROOT"

step "3/7  Gemma-3-12B fp8 LTX text encoder (~12 GB; GATED — needs HF_TOKEN)"
# Comfy-Org/ltx-2 nests it under split_files/; stage then flatten to text_encoders/.
hf download Comfy-Org/ltx-2 \
    split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors \
    --local-dir "$ROOT"
if [ -f "$ROOT/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" ] \
   && [ ! -e "$ROOT/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" ]; then
    ln -s ../split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors \
          "$ROOT/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"
fi

step "4/7  LTX-2.3 distilled-1.1 unet (Q8_0 GGUF, ~24 GB) — the 🎬 LTX lane"
hf download unsloth/LTX-2.3-GGUF \
    distilled-1.1/ltx-2.3-22b-distilled-1.1-Q8_0.gguf \
    --local-dir "$ROOT/unet/ltx2.3"

step "5/7  Sulphur-2 dev unet (Q8_0 GGUF, ~23 GB) — the 🔓 Sulphur uncensored lane"
hf download vantagewithai/Sulphur-2-Base-GGUF \
    sulphur_dev-Q8_0.gguf \
    --local-dir "$ROOT/unet/sulphur-2"

step "6/7  10Eros v1 unet (Q8_0 GGUF, ~23 GB) — the 🔓 10Eros uncensored lane (LTX-2.3-native)"
hf download vantagewithai/LTX2.3-10Eros-GGUF \
    10Eros_v1-Q8_0.gguf \
    --local-dir "$ROOT/unet/10eros"

step "7/7  LTX-2.3 distilled-1.1 LoRA-384 (~7.6 GB) — the shared dev-lane distill speed-up"
hf download Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
    --local-dir "$ROOT/loras"

step "DONE — LTX-2.3 / Sulphur / 10Eros video set in $ROOT"
ls -lh "$ROOT/unet/ltx2.3/distilled-1.1/"*.gguf \
       "$ROOT/unet/sulphur-2/"*.gguf \
       "$ROOT/unet/10eros/"*.gguf \
       "$ROOT/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors" 2>/dev/null \
  | awk '{print "  "$5"  "$NF}'
