#!/usr/bin/env bash
# One-command setup for the club-3090 AI STUDIO — a self-hosted, open-weight
# image / video / audio studio behind Open WebUI. Builds + downloads + brings the
# stack up + installs the OWUI pipe, so a fresh clone goes straight to generating.
#
# Usage:
#   bash scripts/setup-ai-studio.sh           # build + download (~120 GB) + up + install pipe (asks to confirm)
#   bash scripts/setup-ai-studio.sh --yes     # skip the confirmation prompt
#
# Env knobs:
#   SKIP_BUILD=1     skip the ComfyUI image build (already built)
#   SKIP_DOWNLOAD=1  skip the ~120 GB roster pull (already on disk)
#   SKIP_PIPE=1      skip installing the OWUI Studio pipe (do it later)
#   ASSUME_YES=1     same as --yes (also auto-yes when not a TTY / under CI)
#   LANIP=<ip>       host IP shown in the final URLs (auto-detected otherwise)
#
# Idempotent: re-running rebuilds/re-pulls only what changed, installs-or-updates
# the pipe, then brings the stack up via `gpu-mode ai-studio`.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFYUI_DIR="$REPO_DIR/services/comfyui"
STUDIO_DIR="$REPO_DIR/services/studio"
LANIP="${LANIP:-$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192\.168|10\.|172\.)' | head -1)}"
LANIP="${LANIP:-<host-ip>}"

ASSUME_YES="${ASSUME_YES:-}"
case "${1:-}" in
  -y|--yes) ASSUME_YES=1 ;;
  -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
# Auto-yes when non-interactive (piped / nohup / CI) so we never hang on the prompt.
[ -t 0 ] || ASSUME_YES=1
[ -n "${CI:-}" ] && ASSUME_YES=1

say()  { echo -e "\033[0;36m$*\033[0m"; }
warn() { echo -e "\033[1;33m$*\033[0m"; }
ok()   { echo -e "\033[0;32m$*\033[0m"; }

# --- 0. Preflight + plan ----------------------------------------------------
# shellcheck disable=SC1091
. "$REPO_DIR/scripts/preflight.sh" 2>/dev/null || true
if declare -f preflight_docker >/dev/null 2>&1; then
    preflight_docker || exit 1                                   # docker + compose v2 + daemon
    preflight_gpu 1  || exit 1                                   # 1 GPU runs image+audio; video wants 2 (warned below)
    [ -z "${SKIP_BUILD:-}" ]    && { preflight_disk / 32 || exit 1; }                       # comfyui-local image + ~9 GB CUDA base
    [ -z "${SKIP_DOWNLOAD:-}" ] && { preflight_disk /mnt/models/comfyui 130 || exit 1; }    # ~120 GB studio roster
    preflight_gpu_idle || true                                   # soft warn if VRAM already in use
else
    command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found." >&2; exit 1; }
fi
if [ -z "${SKIP_DOWNLOAD:-}" ] && ! command -v hf >/dev/null 2>&1; then
    echo "[preflight] ERROR: 'hf' (huggingface_hub CLI) not found — needed for the weight download." >&2
    echo "            Fix: pip install -U huggingface_hub   (or SKIP_DOWNLOAD=1 if weights are present)." >&2
    exit 1
fi
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)

say "═══ club-3090 AI Studio setup ═══"
echo "  repo:  $REPO_DIR"
echo "  GPUs:  $NGPU"
echo ""
echo "  This will:"
[ -z "${SKIP_BUILD:-}" ]    && echo "    • build the ComfyUI image (comfyui-local) — pulls a ~9 GB CUDA base (one-time, slow)" \
                            || echo "    • (skip build — SKIP_BUILD set)"
[ -z "${SKIP_DOWNLOAD:-}" ] && echo "    • download the full studio roster (~120 GB: image · video · audio · director)" \
                            || echo "    • (skip download — SKIP_DOWNLOAD set)"
[ -z "${SKIP_PIPE:-}" ]     && echo "    • install/update the Open WebUI Studio pipe (the in-OWUI lane picker)" \
                            || echo "    • (skip pipe install — SKIP_PIPE set)"
echo "    • start the studio via 'gpu-mode ai-studio':"
echo "        - ComfyUI (image / video / music / SFX lanes) → both GPUs, port 8188"
echo "        - director (qwen3.5-4b-uncensored, prompt crafter) → GPU0, port 8090"
echo "        - gallery / orchestrator / image-shim / TTS + Open WebUI"
echo ""
if [ "${NGPU:-0}" -lt 2 ]; then
    warn "  ⚠ <2 GPUs detected — the video lanes (22B/14B DiTs) want both cards; image + audio run on one."
fi
if [ -z "$ASSUME_YES" ]; then
    printf "  Proceed? [y/N] "
    read -r reply
    case "$reply" in [yY]|[yY][eE][sS]) ;; *) echo "  aborted."; exit 0 ;; esac
fi

# --- 1. Build the ComfyUI image (clones a pinned ComfyUI + custom nodes on first boot) ---
if [ -z "${SKIP_BUILD:-}" ]; then
    say "── [1/4] Building ComfyUI image (comfyui-local:latest) ──"
    (cd "$COMFYUI_DIR" && sudo docker compose build)
else
    echo "  (SKIP_BUILD set — skipping image build)"
fi

# --- 2. Download the full roster (director · image · video · audio) ----------
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
    say "── [2/4] Downloading the studio roster (~120 GB; idempotent; skip with SKIP_DOWNLOAD=1) ──"
    bash "$COMFYUI_DIR/download_studio_models.sh"
else
    echo "  (SKIP_DOWNLOAD set — skipping weight download)"
fi

# --- 3. Bring the studio up via gpu-mode ------------------------------------
say "── [3/4] Starting the studio (gpu-mode ai-studio) ──"
bash "$REPO_DIR/scripts/gpu-mode.sh" ai-studio

# --- 4. Install the OWUI Studio pipe (install-if-absent; needs an OWUI admin) ---
if [ -z "${SKIP_PIPE:-}" ]; then
    say "── [4/4] Installing the Open WebUI Studio pipe ──"
    if bash "$STUDIO_DIR/push-pipe-to-owui.sh"; then
        ok "  Studio pipe installed/updated."
    else
        warn "  Pipe install skipped — Open WebUI likely has no admin account yet."
        warn "  Sign up at http://$LANIP:8080 (first account = admin), then run:"
        warn "    bash services/studio/push-pipe-to-owui.sh"
    fi
else
    echo "  (SKIP_PIPE set — install later: bash services/studio/push-pipe-to-owui.sh)"
fi

# --- Done — onboarding -------------------------------------------------------
echo ""
ok "═══ AI Studio ready ═══"
echo "  Open WebUI:  http://$LANIP:8080   ← start here"
echo "  ComfyUI:     http://$LANIP:8188   ← optional: full node-graph control"
echo "  Gallery:     http://$LANIP:8189   ← your renders (survive ComfyUI restarts)"
echo ""
say  "  Get started:"
echo "    1. Open the Open WebUI URL and SIGN UP — the FIRST account becomes the admin."
echo "       (If you signed up after this ran, install the pipe now:"
echo "        bash services/studio/push-pipe-to-owui.sh )"
echo "    2. On the 'Studio' function (gear icon), set the 'browser_base' valve to"
echo "       http://$LANIP:8189 so returned media links open from your browser."
echo "    3. Pick a lane in the model selector (🎬 Video · 🖼️ Image · 🎵 Audio), type an idea,"
echo "       and refine by just replying. Full guide: docs/ai-studio/README.md"
echo ""
warn "  Notes:"
warn "    • First render after a cold ComfyUI is slow (loads the model); warm is much faster."
warn "    • Video uses both GPUs; premium voice (Step-Audio) is on-demand + mutually exclusive with a video render."
warn "    • Requirements / per-lane deep-dives: docs/ai-studio/requirements.md · docs/ai-studio/{image,video,audio}.md"
