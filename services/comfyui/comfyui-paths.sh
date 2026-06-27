#!/usr/bin/env bash
# Shared ComfyUI path derivation — ONE knob (MODEL_DIR) configures the whole studio.
#
# Source this from any studio entry point (setup-ai-studio.sh, download_studio_models.sh,
# gpu-mode's ai-studio scene) so the download target, the disk-space check, and the
# container mounts all agree on where the ComfyUI tree lives.
#
# Rule: COMFYUI_ROOT is a "comfyui" SIBLING of the HF cache (MODEL_DIR) — matching the
# reference rig layout /mnt/models/{huggingface,comfyui}. A user who sets MODEL_DIR (e.g.
# in repo-root .env via c3 Settings) gets the ComfyUI tree alongside it automatically,
# instead of the download silently falling back to the rig's /mnt path and failing with
# "mkdir: Permission denied" (club-3090 sumo report, 2026-06-27).
#
# Explicitly-set COMFYUI_ROOT / COMFYUI_MODELS_DIR are always respected.

# 1. MODEL_DIR is configured in repo-root .env (c3 Settings writes it there). Pick it up
#    if the caller hasn't already exported it. Resolve this file's repo root from its own
#    location so it works whether sourced by an in-repo script or a symlinked launcher.
if [ -z "${MODEL_DIR:-}" ]; then
  _c3_paths_root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
  if [ -f "$_c3_paths_root/.env" ]; then
    MODEL_DIR="$(grep -E '^MODEL_DIR=' "$_c3_paths_root/.env" 2>/dev/null | tail -1 | cut -d= -f2-)"
    MODEL_DIR="${MODEL_DIR%\"}"; MODEL_DIR="${MODEL_DIR#\"}"   # strip optional surrounding quotes
  fi
  unset _c3_paths_root
fi

# 2. Derive (only when unset), then export so child processes + docker compose inherit them.
: "${MODEL_DIR:=/mnt/models/huggingface}"
: "${COMFYUI_ROOT:=$(dirname "$MODEL_DIR")/comfyui}"
: "${COMFYUI_MODELS_DIR:=${COMFYUI_ROOT}/models}"
export MODEL_DIR COMFYUI_ROOT COMFYUI_MODELS_DIR
