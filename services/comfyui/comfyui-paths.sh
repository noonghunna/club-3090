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
# Repo root (this file is services/comfyui/comfyui-paths.sh → ../.. = repo root). Exposed so the
# studio helpers below (c3_resolve_lanip) can find the repo-root .env to read/persist config.
C3_REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." 2>/dev/null && pwd || true)"
if [ -z "${C3_PATHS_NO_ENV:-}" ] && [ -n "$C3_REPO_ROOT" ]; then
  _c3_env="$C3_REPO_ROOT/.env"
  if [ -f "$_c3_env" ]; then
    # MODEL_DIR (the studio's one knob) — env wins over .env.
    if [ -z "${MODEL_DIR:-}" ]; then
      MODEL_DIR="$(grep -E '^MODEL_DIR=' "$_c3_env" 2>/dev/null | tail -1 | cut -d= -f2-)"
      MODEL_DIR="${MODEL_DIR%\"}"; MODEL_DIR="${MODEL_DIR#\"}"   # strip optional surrounding quotes
    fi
    # LANIP for the printed URLs — pin it here when auto-detect can't pick the right NIC, or on
    # hosts without `hostname -I` / `ip` (club-3090 #512). Env wins; auto-detect (c3_lan_ip) is the
    # fallback when neither sets it.
    if [ -z "${LANIP:-}" ]; then
      LANIP="$(grep -E '^LANIP=' "$_c3_env" 2>/dev/null | tail -1 | cut -d= -f2-)"
      LANIP="${LANIP%\"}"; LANIP="${LANIP#\"}"
      [ -n "$LANIP" ] && export LANIP
    fi
  fi
  unset _c3_env
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
  local ips=""
  # `hostname -I` is net-tools only; CachyOS / GNU inetutils lack it (club-3090 #512), where it
  # left LANIP empty → URLs fell back to localhost. Try it, then fall back to `ip` (portable).
  command -v hostname >/dev/null 2>&1 && ips="$(hostname -I 2>/dev/null || true)"
  if [ -z "${ips// /}" ] && command -v ip >/dev/null 2>&1; then
    ips="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)"
  fi
  # Prefer a routable LAN address, then a 172.16/12 (incl docker bridges); print at most one.
  # Never return non-zero — callers must not have to guard this with `|| true`.
  { printf '%s\n' $ips | grep -E '^(192\.168|10)\.' || true
    printf '%s\n' $ips | grep -E '^172\.(1[6-9]|2[0-9]|3[01])\.' || true
  } 2>/dev/null | head -1
  return 0
}

# Resolve LANIP for the printed URLs and PERSIST it to repo-root .env so it's stable + editable —
# the .env is the source of truth (club-3090 #512). Precedence: shell-env / .env (loaded above) >
# auto-detect. If auto-detected, write it to .env; if nothing can be detected, fall back to
# localhost and tell the user to set LANIP in .env. Sets + exports the global LANIP. Always 0.
c3_resolve_lanip() {
  if [ -n "${LANIP:-}" ]; then export LANIP; return 0; fi   # already pinned (env or .env)
  local ip env_file="${C3_REPO_ROOT:-.}/.env"
  ip="$(c3_lan_ip)"
  if [ -n "$ip" ]; then
    LANIP="$ip"; export LANIP
    if touch "$env_file" 2>/dev/null; then                  # materialize into .env (upsert)
      if grep -qE '^LANIP=' "$env_file" 2>/dev/null; then
        sed -i "s|^LANIP=.*|LANIP=$ip|" "$env_file" 2>/dev/null || true
      else
        printf 'LANIP=%s\n' "$ip" >> "$env_file" 2>/dev/null || true
      fi
      echo "  ✔ LAN IP $ip detected and saved to .env (edit it there if it picked the wrong NIC)." >&2
    fi
    return 0
  fi
  LANIP="localhost"; export LANIP                            # couldn't detect → ask the user
  echo "  ⚠ Couldn't auto-detect your LAN IP — browser media links will use 'localhost'." >&2
  echo "    Set it in $env_file so links open from other devices:   LANIP=<your-machine-ip>" >&2
  return 0
}

# Persist the resolved COMFYUI_ROOT to repo-root .env so the comfyui compose — launched as
# `sudo docker compose --env-file .env` — mounts the SAME tree the downloader writes into.
# Without this, COMFYUI_ROOT is derived in-shell but stripped by sudo AND absent from .env, so the
# compose's `${COMFYUI_ROOT:-/mnt/models/comfyui}/models` falls back to the /mnt default and mounts
# an EMPTY tree on any rig whose MODEL_DIR isn't the /mnt layout → ComfyUI's model dropdowns come up
# empty (loaders 400 with "not in []"; the HiDream node says "not installed"). club-3090 #510 + #530.
# Write-if-absent: never clobbers a hand-set COMFYUI_ROOT. No-op under C3_PATHS_NO_ENV / unwritable .env.
c3_persist_comfy_root() {
  [ -n "${C3_PATHS_NO_ENV:-}" ] && return 0
  local env_file="${C3_ENV_FILE:-${C3_REPO_ROOT:-.}/.env}"
  grep -qE '^COMFYUI_ROOT=' "$env_file" 2>/dev/null && return 0   # already pinned — respect it
  touch "$env_file" 2>/dev/null && printf 'COMFYUI_ROOT=%s\n' "$COMFYUI_ROOT" >> "$env_file" 2>/dev/null || true
  return 0
}

# Ensure COMFYUI_MODELS_DIR exists + is writable, else exit with an ACTIONABLE message instead
# of letting hf/mkdir cascade into a traceback. Call from download entry points. (#503)
c3_ensure_comfy_models_dir() {
  if mkdir -p "$COMFYUI_MODELS_DIR" 2>/dev/null && [ -w "$COMFYUI_MODELS_DIR" ]; then
    c3_persist_comfy_root   # pin COMFYUI_ROOT so the container mounts this same tree (#510/#530)
    return 0
  fi
  echo "ERROR: ComfyUI models dir is not writable: $COMFYUI_MODELS_DIR" >&2
  echo "       Set MODEL_DIR to a writable location and retry, e.g.:" >&2
  echo "         MODEL_DIR=\"\$HOME/models\" $(basename "${0:-this script}")" >&2
  echo "       (or set COMFYUI_MODELS_DIR directly; c3 Settings writes MODEL_DIR to repo .env)." >&2
  exit 1
}
