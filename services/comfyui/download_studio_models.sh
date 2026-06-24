#!/usr/bin/env bash
# Download EVERYTHING the ai-studio scene needs — director + image + video + audio — by
# running each lane's download script.  Idempotent: every sub-script skips files already
# on disk, so this fetches exactly what's MISSING.  This is the "download all missing"
# entry point: c3's ai-studio setup modal launches it, and it's a handy CLI too.
#
# Does NOT build the ComfyUI image or start the scene — purely fetches weights.
# ~120 GB on a clean rig (only the missing models actually transfer).  Premium voice
# (🎙️ Step-Audio-EditX, ~14 GB) is opt-in via WITH_VOICE=1 (passed to the audio bundler).
#
# Mirrors scripts/lib/studio-models.tsv (the manifest the cockpit + gpu-mode check).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════ ai-studio models — download all (idempotent; skips what's already present) ════"
bash "$HERE/download_director.sh"
bash "$HERE/download_ideogram4.sh"
bash "$HERE/download_zimage.sh"
bash "$HERE/download_video_models.sh"
bash "$HERE/download_wan.sh"
bash "$HERE/download_audio_models.sh"
echo "════ ai-studio models — done ════"
