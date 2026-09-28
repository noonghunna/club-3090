#!/usr/bin/env bash
# Downloads the studio DIRECTOR GGUF (Qwen3.5-4B-Uncensored, the prompt crafter) + its
# vision mmproj → the HF weights root (MODEL_DIR).  Needed by EVERY ai-studio lane — it
# crafts the prompt behind each generation.  ~2.7 GB. Idempotent (hf skips present).
#   Run:  ./download_director.sh
set -uo pipefail

# Weights root: $MODEL_DIR env > saved settings (club-3090 config, then repo .env;
# club-3090#1466) > on-rig default.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # services/comfyui → repo root
# shellcheck source=../../scripts/lib/club-config.sh
. "$REPO_DIR/scripts/lib/club-config.sh"
md="${MODEL_DIR:-}"
[ -z "$md" ] && md="$(unset MODEL_DIR; club_config_get MODEL_DIR "$REPO_DIR" 2>/dev/null || true)"
md="${md:-/mnt/models/huggingface}"
command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' (huggingface_hub CLI) not found. pip install -U huggingface_hub" >&2; exit 1; }

dest="$md/qwen3.5-4b-gguf/hauhaucs-uncensored-q4km"
echo "[$(date +%H:%M:%S)] === Studio director GGUF (Qwen3.5-4B-Uncensored, ~2.7 GB) → $dest ==="
hf download HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive \
    Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
    mmproj-Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf \
    --local-dir "$dest"
echo "[$(date +%H:%M:%S)] DONE — director"
