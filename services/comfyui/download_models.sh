#!/usr/bin/env bash
# Downloads all weights/encoders/VAEs needed for the four ComfyUI workflows
# (nunchaku FLUX dev, nunchaku FLUX Kontext, HunyuanVideo, Wan2.2-Animate).
# Run in background:  nohup ./download_models.sh > /tmp/comfyui-downloads.log 2>&1 &
set -uo pipefail

ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
LOG_TS() { date +%H:%M:%S; }
log()  { echo "[$(LOG_TS)] $*"; }
step() { log ""; log "=== $* ==="; }

# Pattern: pass the repo, then the exact filename(s) as positional args.
# `--include` is only used when we want pattern globs and no positional args.

step "1/9  Nunchaku FLUX.1-dev (SVDQuant INT4)"
hf download nunchaku-ai/nunchaku-flux.1-dev \
    svdq-int4_r32-flux.1-dev.safetensors \
    --local-dir "$ROOT/diffusion_models/nunchaku-flux.1-dev"

step "2/9  Nunchaku FLUX.1-Kontext-dev (SVDQuant INT4)"
hf download nunchaku-ai/nunchaku-flux.1-kontext-dev \
    svdq-int4_r32-flux.1-kontext-dev.safetensors \
    --local-dir "$ROOT/diffusion_models/nunchaku-flux.1-kontext-dev"

step "3/9  HunyuanVideo T2V Q5_K_M GGUF"
# GGUF DiT files go in unet/ (city96 ComfyUI-GGUF Unet Loader convention).
hf download city96/HunyuanVideo-gguf \
    hunyuan-video-t2v-720p-Q5_K_M.gguf \
    --local-dir "$ROOT/unet/hunyuanvideo-gguf"

step "4/9  Wan2.2-Animate-14B Q5_K_M GGUF"
hf download QuantStack/Wan2.2-Animate-14B-GGUF \
    Wan2.2-Animate-14B-Q5_K_M.gguf \
    --local-dir "$ROOT/unet/wan2.2-animate-14b-gguf"

step "5a/9 FLUX text encoder: clip_l"
hf download comfyanonymous/flux_text_encoders \
    clip_l.safetensors \
    --local-dir "$ROOT/text_encoders"

step "5b/9 FLUX text encoder: t5xxl_fp16"
hf download comfyanonymous/flux_text_encoders \
    t5xxl_fp16.safetensors \
    --local-dir "$ROOT/text_encoders"

step "6/9  FLUX VAE (ae.safetensors from FLUX.1-schnell)"
hf download black-forest-labs/FLUX.1-dev \
    ae.safetensors \
    --local-dir "$ROOT/vae/flux"

step "7/9  HunyuanVideo VAE (bf16)"
hf download Kijai/HunyuanVideo_comfy \
    hunyuan_video_vae_bf16.safetensors \
    --local-dir "$ROOT/vae/hunyuanvideo"

step "8/9  HunyuanVideo LLaVA text encoder (Kijai bf16, ~16 GB)"
hf download Kijai/llava-llama-3-8b-text-encoder-tokenizer \
    --local-dir "$ROOT/text_encoders/llava-llama-3-8b"

step "9a/9 Wan2.2 text encoder umt5-xxl fp8"
hf download Kijai/WanVideo_comfy \
    umt5-xxl-enc-fp8_e4m3fn.safetensors \
    --local-dir "$ROOT/text_encoders"

step "9b/9 Wan2.2 VAE bf16"
hf download Kijai/WanVideo_comfy \
    Wan2_2_VAE_bf16.safetensors \
    --local-dir "$ROOT/vae/wan"

step "DONE. Disk usage:"
du -sh "$ROOT"/* 2>/dev/null | sort -h
log ""
log "All downloads complete."
