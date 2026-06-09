#!/usr/bin/env bash
# One-shot setup for the club-3090 IMAGE-STUDIO bundle:
#   ComfyUI (Ideogram-4 image gen) + Open WebUI front-end + gemma-4-12b chat,
#   coexisting on a 2-GPU box (ComfyUI -> GPU0, chat -> GPU1).
#
# Usage:
#   bash scripts/setup-image-studio.sh           # build + download + bring up (asks to confirm)
#   bash scripts/setup-image-studio.sh --yes      # skip the confirmation prompt
#
# Env knobs:
#   SKIP_BUILD=1     skip the ComfyUI image build (already built)
#   SKIP_DOWNLOAD=1  skip the ~27 GB Ideogram-4 weight pull (already on disk)
#   ASSUME_YES=1     same as --yes (also auto-yes when not a TTY / under CI)
#   LANIP=<ip>       host IP shown in the final URLs (auto-detected otherwise)
#
# Idempotent: re-running rebuilds/re-pulls only what changed, then brings the
# stack up via `gpu-mode image-studio`.
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
# Auto-yes when non-interactive (piped / nohup / CI) so we never hang on the prompt.
[ -t 0 ] || ASSUME_YES=1
[ -n "${CI:-}" ] && ASSUME_YES=1

say()  { echo -e "\033[0;36m$*\033[0m"; }
warn() { echo -e "\033[1;33m$*\033[0m"; }
ok()   { echo -e "\033[0;32m$*\033[0m"; }

# --- 0. Preflight + plan ----------------------------------------------------
# Reuse the repo's preflight library (docker / gpu / disk), then add image-studio
# specifics. Hard-fail before the long build/download if something's missing.
# shellcheck disable=SC1091
. "$REPO_DIR/scripts/preflight.sh" 2>/dev/null || true
MODEL_DIR_RESOLVED="$(grep -E '^MODEL_DIR=' "$REPO_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2-)"
MODEL_DIR_RESOLVED="${MODEL_DIR_RESOLVED:-/mnt/models/huggingface}"
if declare -f preflight_docker >/dev/null 2>&1; then
    preflight_docker || exit 1                                   # docker + compose v2 + daemon
    preflight_gpu 1  || exit 1                                   # >=1 GPU (coexist needs 2 — warned below)
    [ -z "${SKIP_BUILD:-}" ]    && { preflight_disk / 32 || exit 1; }                      # comfyui-local image (~27 GB) + base pull
    [ -z "${SKIP_DOWNLOAD:-}" ] && { preflight_disk /mnt/models/comfyui 30 || exit 1; }    # Ideogram-4 set (~27 GB)
    preflight_gpu_idle || true                                   # soft warn if VRAM already in use
else
    command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found." >&2; exit 1; }
fi
# Image-studio specifics (not in preflight.sh):
if [ -z "${SKIP_DOWNLOAD:-}" ] && ! command -v hf >/dev/null 2>&1; then
    echo "[preflight] ERROR: 'hf' (huggingface_hub CLI) not found — needed for the weight download." >&2
    echo "            Fix: pip install -U huggingface_hub   (or SKIP_DOWNLOAD=1 if weights are present)." >&2
    exit 1
fi
GEMMA_GGUF="$MODEL_DIR_RESOLVED/gemma-4-12b-gguf/unsloth-ud-q8kxl/gemma-4-12b-it-UD-Q8_K_XL.gguf"
GEMMA_MISSING=""
if [ ! -f "$GEMMA_GGUF" ]; then
    GEMMA_MISSING=1
    echo "[preflight] WARN:  gemma-4-12b GGUF not found at $GEMMA_GGUF" >&2
    echo "            The chat model won't start until it's present (image gen still works). Fetch it:" >&2
    echo "              hf download unsloth/gemma-4-12b-it-GGUF gemma-4-12b-it-UD-Q8_K_XL.gguf \\" >&2
    echo "                --local-dir \"$(dirname "$GEMMA_GGUF")\"" >&2
fi
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)

say "═══ club-3090 image-studio setup ═══"
echo "  repo:  $REPO_DIR"
echo "  GPUs:  $NGPU"
echo ""
echo "  This will:"
[ -z "${SKIP_BUILD:-}" ]    && echo "    • build the ComfyUI image (comfyui-local) — pulls a ~9 GB CUDA base (one-time, slow)" \
                            || echo "    • (skip build — SKIP_BUILD set)"
[ -z "${SKIP_DOWNLOAD:-}" ] && echo "    • download the Ideogram-4 model set (~27 GB) into the ComfyUI models tree" \
                            || echo "    • (skip download — SKIP_DOWNLOAD set)"
echo "    • start the bundle via 'gpu-mode image-studio':"
echo "        - ComfyUI / Ideogram-4   → GPU 0, port 8188"
echo "        - gemma-4-12b chat        → GPU 1, port 8069"
echo "        - Open WebUI + LiteLLM + SearXNG (always-on)"
echo ""
if [ "${NGPU:-0}" -lt 2 ]; then
    warn "  ⚠ <2 GPUs — image gen and a local chat model can't run at once (GPU-mutex)."
    warn "    The bundle will run ComfyUI image gen only; for chat use 'gpu-mode chat'."
fi
if [ -z "$ASSUME_YES" ]; then
    printf "  Proceed? [y/N] "
    read -r reply
    case "$reply" in [yY]|[yY][eE][sS]) ;; *) echo "  aborted."; exit 0 ;; esac
fi

# --- 1. Build the ComfyUI image (clones a pinned ComfyUI + nodes on first boot) ---
if [ -z "${SKIP_BUILD:-}" ]; then
    say "── [1/3] Building ComfyUI image (comfyui-local:latest) ──"
    (cd "$COMFYUI_DIR" && sudo docker compose build)
else
    echo "  (SKIP_BUILD set — skipping image build)"
fi

# --- 2. Download the Ideogram-4 model set (~27 GB) --------------------------
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
    say "── [2/3] Downloading Ideogram-4 model set (~27 GB; skip with SKIP_DOWNLOAD=1) ──"
    bash "$COMFYUI_DIR/download_ideogram4.sh"
else
    echo "  (SKIP_DOWNLOAD set — skipping weight download)"
fi

# --- 3. Bring the stack up via gpu-mode -------------------------------------
say "── [3/3] Starting the bundle (gpu-mode image-studio) ──"
bash "$REPO_DIR/scripts/gpu-mode.sh" image-studio

# --- Done — onboarding -------------------------------------------------------
echo ""
ok "═══ Image-studio ready ═══"
echo "  Open WebUI:  http://$LANIP:8080   ← start here (chat + image)"
echo "  ComfyUI:     http://$LANIP:8188   ← optional: full node-graph control"
echo ""
say  "  Get started:"
echo "    1. Open the Open WebUI URL and SIGN UP — the FIRST account you create"
echo "       becomes the admin. (No credentials are pre-set; you choose them here.)"
echo "    2. Pick the chat model 'gemma-4-12b…' in the top selector and chat normally."
echo "    3. Generate an image: send a prompt, then click the 🖼️ image icon on the"
echo "       reply — it renders via Ideogram-4 on ComfyUI. (Image gen is pre-wired."
echo "       Toggle/inspect it under Admin → Settings → Images.)"
echo ""
warn "  Notes:"
warn "    • First image after a cold ComfyUI is slow (~2 min, loads ~20 GB); warm ~70 s."
warn "    • 🖼️ button missing / image gen unconfigured? You reused an existing Open WebUI"
warn "      volume — the image-gen env only auto-applies on a FRESH volume. Set it in"
warn "      Admin → Settings → Images (Engine=ComfyUI, Base URL=http://host.docker.internal:8188)."
warn "    • 2048² fits but is tight (~22 GB, batch 1); prefer 1024² + upscale for routine high-res."
