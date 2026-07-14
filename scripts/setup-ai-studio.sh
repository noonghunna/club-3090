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
#   SKIP_DISK_CHECK=1 bypass the free-space preflight (resume an idempotent download)
#   SKIP_PIPE=1      skip installing the OWUI Studio pipe (do it later)
#   ASSUME_YES=1     same as --yes (also auto-yes when not a TTY / under CI)
#   LANIP=<ip>       host IP shown in the final URLs. Auto-detected + saved to .env on first run;
#                    pin it in .env (or via this env var) if it picks the wrong NIC / can't detect.
#   MODEL_DIR=<dir>  HF/GGUF cache root; the ComfyUI tree goes to a "comfyui" sibling
#                    of it (override COMFYUI_ROOT / COMFYUI_MODELS_DIR to decouple).
#
# Idempotent: re-running rebuilds/re-pulls only what changed, installs-or-updates
# the pipe, then brings the stack up via `gpu-mode ai-studio`.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFYUI_DIR="$REPO_DIR/services/comfyui"
STUDIO_DIR="$REPO_DIR/services/studio"
# Derive COMFYUI_ROOT / COMFYUI_MODELS_DIR from MODEL_DIR so the download target, the disk
# check, and the container mounts all agree (a user who set MODEL_DIR gets the ComfyUI tree
# alongside it instead of the rig's /mnt fallback). See services/comfyui/comfyui-paths.sh.
# shellcheck disable=SC1091
. "$COMFYUI_DIR/comfyui-paths.sh"
# LAN IP for the final URLs. Resolve via the shared helper: env / .env win, else auto-detect and
# PERSIST to .env (the source of truth) — or, if nothing detects, fall back to localhost and tell
# the user to set LANIP in .env. Keeps setup + gpu-mode from drifting. (#504, #512)
c3_resolve_lanip

ASSUME_YES="${ASSUME_YES:-}"
case "${1:-}" in
  -y|--yes) ASSUME_YES=1 ;;
  -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    [ -z "${SKIP_BUILD:-}${SKIP_DISK_CHECK:-}" ]    && { preflight_disk / 32 || exit 1; }                          # comfyui-local image + ~9 GB CUDA base
    [ -z "${SKIP_DOWNLOAD:-}${SKIP_DISK_CHECK:-}" ] && { preflight_disk "$COMFYUI_MODELS_DIR" 130 || exit 1; }      # ~120 GB roster → MODEL_DIR-derived path
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
# Open WebUI's host port 8080 is fixed (hardcoded in services/openwebui, and the
# pipe/register helpers all talk to :8080). If another service already holds it,
# OWUI silently fails to bind and the studio comes up half-working — and
# gpu-mode's start_service swallows that failure ('… || echo "failed"' returns
# 0). So gate the bring-up on 8080 being free, and verify OWUI afterwards,
# instead of reporting a false "ready" (#686).
if ! docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | grep -qE '^open-webui .*:8080->'; then
    _p8080=""
    if command -v ss >/dev/null 2>&1; then
        _p8080=$(ss -Hltn 2>/dev/null | awk '{print $4}' | grep -E ':8080$' | head -1)
    elif command -v lsof >/dev/null 2>&1; then
        _p8080=$(lsof -iTCP:8080 -sTCP:LISTEN -Pn 2>/dev/null | tail -n +2 | head -1)
    fi
    if [ -n "$_p8080" ]; then
        warn "  ✗ Port 8080 is already in use — Open WebUI can't bind it (listener: $_p8080)."
        echo "     OWUI's host port 8080 is fixed (baked into the compose + the pipe/register helpers)."
        echo "     Free 8080 first — e.g. move the other service (a llama-server router: add"
        echo "     '--port 8081' and point your client at the new port) — then re-run this script." >&2
        exit 1
    fi
fi

say "── [3/4] Starting the studio (gpu-mode ai-studio) ──"
bash "$REPO_DIR/scripts/gpu-mode.sh" ai-studio

# gpu-mode's start_service swallows a failed 'compose up' (#686), so trust the
# container STATE, not the exit code: a bind failure leaves open-webui missing
# or Exited/Restarting rather than Up.
_owui_status="$(docker ps -a --format '{{.Names}}\t{{.Status}}' 2>/dev/null | awk -F'\t' '$1=="open-webui"{print $2; exit}')"
case "$_owui_status" in
    Up*) ;;   # running (optionally "(healthy)") — good
    *)
        warn "  ✗ Open WebUI did not come up (status: ${_owui_status:-missing})."
        echo "     Most common cause: something else is bound to its host port 8080." >&2
        echo "     Inspect:  docker ps -a | grep open-webui" >&2
        echo "               docker logs open-webui 2>&1 | tail -20" >&2
        echo "     Fix the conflict, then re-run:  bash scripts/setup-ai-studio.sh" >&2
        exit 1 ;;
esac

# --- 4. Install the OWUI Studio pipe (install-if-absent; needs an OWUI admin) ---
# PIPE_OK drives the onboarding below: a fresh install has no OWUI admin yet (you create it by
# signing up), so the pipe install is EXPECTED to be skipped here — we then make "sign up THEN
# install the pipe" a prominent numbered step instead of a warning that scrolls past (#510).
PIPE_OK=0
if [ -z "${SKIP_PIPE:-}" ]; then
    say "── [4/4] Installing the Open WebUI Studio pipe ──"
    if bash "$STUDIO_DIR/push-pipe-to-owui.sh"; then
        ok "  Studio pipe installed/updated."
        PIPE_OK=1
    else
        warn "  Pipe install skipped — Open WebUI has no admin account yet (expected on a fresh install)."
        warn "  → see the highlighted step below: sign up first, then install the pipe."
    fi
else
    echo "  (SKIP_PIPE set — install later: bash services/studio/push-pipe-to-owui.sh)"
fi

# --- 4b. Wire the LLM catalog into OWUI as per-backend connections -----------
# OWUI hides models from an UNREACHABLE connection, so pointing it straight at each
# model's own port (instead of the always-up LiteLLM :4000 gateway) makes the chat
# picker scene-accurate: a model shows only while its gpu-mode scene is serving.
# Both helpers are idempotent + no-op when OWUI is down or the connection already
# (does not) exist — safe to re-run. Served names match the catalog (model IDs unchanged).
if [ -z "${SKIP_OWUI_WIRING:-}" ]; then
    say "── Wiring the LLM catalog into Open WebUI (per-backend, scene-accurate) ──"
    for port in 8090 8010 8051 8032 8038 8199; do
        bash "$REPO_DIR/scripts/lib/owui-register.sh" "$port" || true
    done
    # Drop the legacy LiteLLM :4000 gateway connection — it lists every catalog model
    # regardless of which scene is up (the masking this per-backend wiring replaces).
    bash "$REPO_DIR/scripts/lib/owui-unregister.sh" 4000 || true
else
    echo "  (SKIP_OWUI_WIRING set — leaving OWUI connections as-is)"
fi

# --- Done — onboarding -------------------------------------------------------
echo ""
ok "═══ AI Studio ready ═══"
echo "  Open WebUI:  http://$LANIP:8080   ← start here"
echo "  ComfyUI:     http://$LANIP:8188   ← optional: full node-graph control"
echo "  Gallery:     http://$LANIP:8189   ← your renders (survive ComfyUI restarts)"
echo ""
say  "  Get started:"
_n=1
echo "    $_n. Open the Open WebUI URL and SIGN UP — the FIRST account becomes the admin."; _n=$((_n + 1))
# The pipe couldn't install without an admin → make installing it a LOUD, unmissable step.
if [ "$PIPE_OK" != "1" ] && [ -z "${SKIP_PIPE:-}" ]; then
    echo ""
    warn "    $_n. ⚠ INSTALL THE STUDIO LANES — they were skipped because Open WebUI had no admin"
    warn "       account yet. After you sign up (step 1), run this ONE command:"
    warn ""
    warn "           bash services/studio/push-pipe-to-owui.sh"
    warn ""
    warn "       Then reload Open WebUI — the 🎬 Studio lanes appear in the model selector."
    echo ""
    _n=$((_n + 1))
fi
echo "    $_n. On the 'Studio' function (Admin → Functions → Studio), set the 'browser_base'"
echo "       valve to http://$LANIP:8189 so returned media links open from your browser."; _n=$((_n + 1))
echo "    $_n. Pick a lane in the model selector (🎬 Video · 🖼️ Image · 🎵 Audio), type an idea,"
echo "       and refine by just replying. Full guide: docs/ai-studio/README.md"
echo ""
warn "  Notes:"
warn "    • First render after a cold ComfyUI is slow (loads the model); warm is much faster."
warn "    • Video uses both GPUs; premium voice (Step-Audio) is on-demand + mutually exclusive with a video render."
warn "    • Requirements / per-lane deep-dives: docs/ai-studio/requirements.md · docs/ai-studio/{image,video,audio}.md"
