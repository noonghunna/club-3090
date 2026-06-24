#!/usr/bin/env bash
# Downloads the ACE-Step v1 3.5B all-in-one checkpoint for the studio 🎵 Music lane
# (text → music / song). Lands it exactly where the studio music workflow loads it
# (services/studio/studio_pipe.py WORKFLOWS["music"].ckpt):
#   checkpoints/ace-step-1.5/all_in_one/ace_step_v1_3.5b.safetensors
#
# ~7.7 GB. Single-device GPU0 (~8 GB) — coexists with the qwen director.
# Idempotent: skips if already on disk.  Run:  ./download_ace_step.sh
set -uo pipefail

ROOT="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' (huggingface_hub CLI) not found. pip install -U huggingface_hub" >&2; exit 1; }

target="$ROOT/checkpoints/ace-step-1.5/all_in_one/ace_step_v1_3.5b.safetensors"
if [ -e "$target" ]; then
    log "ACE-Step already present → $target  (skip)"
else
    mkdir -p "$ROOT/checkpoints/ace-step-1.5"
    log "=== ACE-Step v1 3.5B all-in-one (~7.7 GB) ==="
    hf download Comfy-Org/ACE-Step_ComfyUI_repackaged all_in_one/ace_step_v1_3.5b.safetensors \
        --local-dir "$ROOT/checkpoints/ace-step-1.5"
fi
log "DONE — 🎵 Music lane checkpoint"
ls -lhL "$target" 2>/dev/null | awk '{print "  "$5"  "$NF}'
