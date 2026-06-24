#!/usr/bin/env bash
# DEPRECATED — image / video / audio are now ONE consolidated **AI Studio** scene
# (the separate `image-studio` / `video-studio` gpu-modes were retired). This name
# is kept only as a redirect to the canonical, full-roster setup:
#
#   scripts/setup-ai-studio.sh   (build + download the whole roster + bring up + install the OWUI pipe)
#
# All flags/env pass through. To pull just the video weights, run the per-lane
# downloader directly (services/comfyui/download_video_models.sh + download_wan.sh).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup-ai-studio.sh" "$@"
