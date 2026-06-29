#!/usr/bin/env bash
# Shared ComfyUI path derivation — ONE knob (MODEL_DIR) configures the whole studio.
#
# Source this from any studio entry point (setup-ai-studio.sh, download_studio_models.sh,
# gpu-mode's ai-studio scene) so the download target, the disk-space check, and the
# container mounts all agree on where the ComfyUI tree lives.
#
# Rule: COMFYUI_ROOT is a "comfyui" SIBLING of the HF cache (MODEL_DIR) — matching the
# reference rig layout /mnt/models/{huggingface,comfyui}. Resolution order for MODEL_DIR:
#   1. an explicit env / .env value (c3 Settings writes it to repo-root .env), else
#   2. $HOME/models  → a USER-OWNED default, so a zero-config clone lands in a writable
#      tree ($HOME/comfyui/models) instead of the rig's /mnt path → no "mkdir: Permission
#      denied" (club-3090 #503; sumo report 2026-06-27). HOME-less shells keep /mnt.
#
# Explicitly-set COMFYUI_ROOT / COMFYUI_MODELS_DIR are always respected. This file is also
# the shared home for studio host helpers — c3_lan_ip / c3_ensure_comfy_models_dir (below).

# 1. MODEL_DIR is configured in repo-root .env (c3 Settings writes it there). Pick it up
#    if the caller hasn't already exported it. Resolve this file's repo root from its own
#    location so it works whether sourced by an in-repo script or a symlinked launcher.
#    (C3_PATHS_NO_ENV=1 skips the .env read — for tests / fully-explicit callers.)
if [ -z "${MODEL_DIR:-}" ] && [ -z "${C3_PATHS_NO_ENV:-}" ]; then
  _c3_paths_root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
  if [ -f "$_c3_paths_root/.env" ]; then
    MODEL_DIR="$(grep -E '^MODEL_DIR=' "$_c3_paths_root/.env" 2>/dev/null | tail -1 | cut -d= -f2-)"
    MODEL_DIR="${MODEL_DIR%\"}"; MODEL_DIR="${MODEL_DIR#\"}"   # strip optional surrounding quotes
  fi
  unset _c3_paths_root
fi

# 2. Default MODEL_DIR only when neither the env nor .env set it. Prefer a USER-OWNED location
#    so a zero-config clone never targets the reference rig's /mnt path (which a normal user
#    can't write → "mkdir: Permission denied", club-3090 #503). HOME-less contexts (some
#    CI/root shells) keep the legacy /mnt default. An explicit MODEL_DIR always wins.
if [ -z "${MODEL_DIR:-}" ]; then
  if [ -n "${HOME:-}" ]; then MODEL_DIR="$HOME/models"; else MODEL_DIR="/mnt/models/huggingface"; fi
fi

# 3. Derive (only when unset), then export so child processes + docker compose inherit them.
: "${COMFYUI_ROOT:=$(dirname "$MODEL_DIR")/comfyui}"
: "${COMFYUI_MODELS_DIR:=${COMFYUI_ROOT}/models}"
export MODEL_DIR COMFYUI_ROOT COMFYUI_MODELS_DIR


# --- shared studio host helpers (this is the lib every studio entry point sources) -----------

# Best-effort LAN IP for the user-facing URLs the launchers print. Prefers a routable private
# address (192.168/10) and DEPRIORITIZES docker bridges (172.16–31, where docker0 / compose
# networks live) so a docker host doesn't advertise a bridge IP instead of its LAN IP. Empty
# when none is found; callers fall back to localhost. Override everything with LANIP=. (#504)
c3_lan_ip() {
  local ips; ips="$(hostname -I 2>/dev/null | tr ' ' '\n')"
  { printf '%s\n' "$ips" | grep -E '^(192\.168|10)\.'                # routable LAN first
    printf '%s\n' "$ips" | grep -E '^172\.(1[6-9]|2[0-9]|3[01])\.'   # then 172.16/12 (incl docker)
  } 2>/dev/null | head -1
}

# Ensure COMFYUI_MODELS_DIR exists + is writable, else exit with an ACTIONABLE message instead
# of letting hf/mkdir cascade into a traceback. Call from download entry points. (#503)
c3_ensure_comfy_models_dir() {
  if mkdir -p "$COMFYUI_MODELS_DIR" 2>/dev/null && [ -w "$COMFYUI_MODELS_DIR" ]; then return 0; fi
  echo "ERROR: ComfyUI models dir is not writable: $COMFYUI_MODELS_DIR" >&2
  echo "       Set MODEL_DIR to a writable location and retry, e.g.:" >&2
  echo "         MODEL_DIR=\"\$HOME/models\" $(basename "${0:-this script}")" >&2
  echo "       (or set COMFYUI_MODELS_DIR directly; c3 Settings writes MODEL_DIR to repo .env)." >&2
  exit 1
}
