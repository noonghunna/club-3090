#!/usr/bin/env bash
# One-shot setup for the club-3090 VIDEO-STUDIO bundle:
#   ComfyUI (LTX-2.3 🎬 + Sulphur/10Eros 🔓 video lanes, both GPUs) + the qwen director +
#   gallery/orchestrator/image-shim/tts sidecars + Open WebUI front-end.
#
# Usage:
#   bash scripts/setup-video-studio.sh           # build + download + bring up (asks to confirm)
#   bash scripts/setup-video-studio.sh --yes      # skip the confirmation prompt
#
# Env knobs:
#   SKIP_BUILD=1     skip the ComfyUI image build (already built)
#   SKIP_DOWNLOAD=1  skip the model pulls (already on disk)
#   ASSUME_YES=1     same as --yes (also auto-yes when not a TTY / under CI)
#   LANIP=<ip>       host IP shown in the final URLs (auto-detected otherwise)
#
# HF_TOKEN must be set — the gemma LTX text-encoder repo (Comfy-Org/ltx-2) is gated.
# Idempotent: re-pulls only what changed (hf skips present files), then brings the
# stack up via `gpu-mode video-studio`.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFYUI_DIR="$REPO_DIR/services/comfyui"
LANIP="${LANIP:-$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192\.168|10\.|172\.)' | head -1)}"
LANIP="${LANIP:-<host-ip>}"

ASSUME_YES="${ASSUME_YES:-}"
case "${1:-}" in
  -y|--yes) ASSUME_YES=1 ;;
  -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
[ -t 0 ] || ASSUME_YES=1
[ -n "${CI:-}" ] && ASSUME_YES=1

say()  { echo -e "\033[0;36m$*\033[0m"; }
warn() { echo -e "\033[1;33m$*\033[0m"; }
ok()   { echo -e "\033[0;32m$*\033[0m"; }

# --- 0. Preflight + plan ----------------------------------------------------
# shellcheck disable=SC1091
. "$REPO_DIR/scripts/preflight.sh" 2>/dev/null || true
MODEL_DIR_RESOLVED="$(grep -E '^MODEL_DIR=' "$REPO_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2-)"
MODEL_DIR_RESOLVED="${MODEL_DIR_RESOLVED:-/mnt/models/huggingface}"
if declare -f preflight_docker >/dev/null 2>&1; then
    preflight_docker || exit 1
    preflight_gpu 1  || exit 1                                   # both GPUs used (warned below if <2)
    [ -z "${SKIP_BUILD:-}" ]    && { preflight_disk / 32 || exit 1; }                      # comfyui-local image
    [ -z "${SKIP_DOWNLOAD:-}" ] && { preflight_disk /mnt/models/comfyui 90 || exit 1; }    # LTX/Sulphur/10Eros set (~82 GB)
    preflight_gpu_idle || true
else
    command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found." >&2; exit 1; }
fi
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
    command -v hf >/dev/null 2>&1 || {
        echo "[preflight] ERROR: 'hf' (huggingface_hub CLI) not found — needed for the weight download." >&2
        echo "            Fix: pip install -U huggingface_hub   (or SKIP_DOWNLOAD=1 if weights are present)." >&2
        exit 1; }
    [ -z "${HF_TOKEN:-}" ] && warn "[preflight] WARN: HF_TOKEN not set — the gemma LTX encoder (Comfy-Org/ltx-2) is gated; that file will 401."
fi
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)

say "═══ club-3090 video-studio setup ═══"
echo "  repo:  $REPO_DIR"
echo "  GPUs:  $NGPU"
echo ""
echo "  This will:"
[ -z "${SKIP_BUILD:-}" ]    && echo "    • build the ComfyUI image (comfyui-local) — pulls a ~9 GB CUDA base (one-time, slow)" \
                            || echo "    • (skip build — SKIP_BUILD set)"
[ -z "${SKIP_DOWNLOAD:-}" ] && echo "    • download the LTX-2.3 / Sulphur / 10Eros video set (~82 GB) + director + Kokoro" \
                            || echo "    • (skip download — SKIP_DOWNLOAD set)"
echo "    • start the bundle via 'gpu-mode video-studio':"
echo "        - ComfyUI (LTX-2.3 + Sulphur + 10Eros), both GPUs → port 8188"
echo "        - director + gallery + orchestrator + image-shim + tts sidecars"
echo "        - Open WebUI + LiteLLM + SearXNG (always-on)"
echo ""
if [ "${NGPU:-0}" -lt 2 ]; then
    warn "  ⚠ <2 GPUs — video-studio expects both cards for the 22B LTX/Sulphur/10Eros unets; it may OOM."
fi
if [ -z "$ASSUME_YES" ]; then
    printf "  Proceed? [y/N] "
    read -r reply
    case "$reply" in [yY]|[yY][eE][sS]) ;; *) echo "  aborted."; exit 0 ;; esac
fi

# --- 1. Build the ComfyUI image ---------------------------------------------
if [ -z "${SKIP_BUILD:-}" ]; then
    say "── [1/3] Building ComfyUI image (comfyui-local:latest) ──"
    (cd "$COMFYUI_DIR" && sudo docker compose build)
else
    echo "  (SKIP_BUILD set — skipping image build)"
fi

# --- 2. Download the model sets (video set + director + audio lanes) --------
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
    say "── [2/3] Downloading model sets (~76 GB; skip with SKIP_DOWNLOAD=1) ──"
    echo "  • LTX-2.3 / Sulphur / 10Eros video set (~82 GB)"
    bash "$COMFYUI_DIR/download_video_models.sh"
    # The director crafts the prompt for both video lanes (shared with image-studio).
    echo "  • Studio director GGUF (Qwen3.5-4B-Uncensored, ~2.7 GB + mmproj)"
    hf download HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive \
        Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
        mmproj-Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf \
        --local-dir "$MODEL_DIR_RESOLVED/qwen3.5-4b-gguf/hauhaucs-uncensored-q4km"
    # Audio lanes (🎵 music · 🔊 SFX · 🗣️ narration, incl. Kokoro voiceover) — shared
    # across the ai-studio scene.
    echo "  • Studio audio models (🎵 music · 🔊 SFX · 🗣️ narration; ~13 GB)"
    bash "$COMFYUI_DIR/download_audio_models.sh"
else
    echo "  (SKIP_DOWNLOAD set — skipping weight download)"
fi

# --- 3. Bring the stack up via gpu-mode -------------------------------------
say "── [3/3] Starting the bundle (gpu-mode ai-studio) ──"
bash "$REPO_DIR/scripts/gpu-mode.sh" ai-studio

# --- Done — onboarding -------------------------------------------------------
echo ""
ok "═══ Video-studio ready ═══"
echo "  Open WebUI:  http://$LANIP:8080   ← start here"
echo "  ComfyUI:     http://$LANIP:8188   ← optional: full node-graph control"
echo ""
say  "  Get started:"
echo "    1. Open the Open WebUI URL and SIGN UP (first account = admin)."
echo "    2. Pick a video lane in the model selector: '🎬 LTX-2.3' (video+audio) or"
echo "       '🔓 Sulphur' or '🔓 10Eros' (uncensored). Type an idea — the director crafts it + renders."
echo "    3. Attach an image for image→video; add 'voiceover: …' for a Kokoro narration."
echo ""
warn "  Notes:"
warn "    • First clip after a cold ComfyUI is slow (loads the 22B unet); subsequent ~faster."
warn "    • GPU-mutex with the dual-card LLMs — video-studio uses both cards."
warn "    • Clips default ~10 s, cap ~15 s. See docs/ai-studio/video.md."
