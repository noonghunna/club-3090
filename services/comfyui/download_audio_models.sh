#!/usr/bin/env bash
# Downloads the studio AUDIO-lane models in one go (for the unified ai-studio setup):
#   🎵 Music     — ACE-Step v1 3.5B            (download_ace_step.sh,    ~7.7 GB)
#   🔊 SFX       — Stable Audio Open + t5-base (download_stable_audio.sh, ~5.3 GB)
#   🗣️ Narration — Kokoro voices (CPU)          (download_kokoro.sh,      ~0.3 GB)
# Premium voice (🎙️ Step-Audio-EditX, ~14 GB GPU, the isolated on-demand step-voice
# service) is OPT-IN — set WITH_VOICE=1 to also fetch it (it isn't an always-on lane).
#
# ~13 GB core (+~14 GB with voice). Idempotent: each sub-script skips files already on
# disk.  Run:  ./download_audio_models.sh   (or WITH_VOICE=1 ./download_audio_models.sh)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "── Studio audio models (🎵 music · 🔊 SFX · 🗣️ narration) ──"
bash "$HERE/download_ace_step.sh"
bash "$HERE/download_stable_audio.sh"
bash "$HERE/download_kokoro.sh"

if [ "${WITH_VOICE:-0}" = "1" ]; then
    echo "── Premium voice (🎙️ Step-Audio-EditX, ~14 GB) — WITH_VOICE=1 ──"
    bash "$HERE/download_step_audio.sh"
else
    echo "  (premium voice 🎙️ skipped — set WITH_VOICE=1 to also fetch Step-Audio-EditX, ~14 GB)"
fi
echo "── Audio models done ──"
