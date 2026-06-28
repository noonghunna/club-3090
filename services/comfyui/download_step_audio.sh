#!/usr/bin/env bash
# Downloads Step-Audio-EditX for the studio-step-voice sidecar (premium zero-shot
# voice clone + emotion/style editing — the 🎙️ Voice lane). Two repos: the model
# weights + its audio tokenizer, both from stepfun-ai (Apache-2.0). ~14 GB bf16.
#
# Run:  ./download_step_audio.sh
#       nohup ./download_step_audio.sh > /tmp/step-audio-dl.log 2>&1 &   (background)
#
# Lands where the step-voice compose mounts them (STEP_AUDIO_DIR → /models):
#   Step-Audio/Step-Audio-EditX/
#   Step-Audio/Step-Audio-Tokenizer/
set -uo pipefail

# Mirrors the step-voice compose default (${STEP_AUDIO_DIR:-/mnt/models/comfyui/models/Step-Audio}).
# Resolve COMFYUI_MODELS_DIR from MODEL_DIR/.env so a standalone run matches
# setup-ai-studio.sh instead of falling back to the dev-rig /mnt path (issue #503).
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/comfyui-paths.sh"
ROOT="${STEP_AUDIO_DIR:-${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}/Step-Audio}"
LOG_TS() { date +%H:%M:%S; }
log()  { echo "[$(LOG_TS)] $*"; }
step() { log ""; log "=== $* ==="; }

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' (huggingface_hub CLI) not found. pip install -U huggingface_hub" >&2; exit 1; }
mkdir -p "$ROOT"

step "1/2  Step-Audio-EditX weights (~14 GB)"
hf download stepfun-ai/Step-Audio-EditX --local-dir "$ROOT/Step-Audio-EditX"

step "2/2  Step-Audio-Tokenizer"
hf download stepfun-ai/Step-Audio-Tokenizer --local-dir "$ROOT/Step-Audio-Tokenizer"

log ""
log "Done → $ROOT  (Step-Audio-EditX/ + Step-Audio-Tokenizer/)"
