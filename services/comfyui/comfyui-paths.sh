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
      # `|| true`: a no-match grep returns 1, which pipefail propagates → the
      # assignment fails → a `set -e` caller (setup-ai-studio.sh) exits SILENTLY
      # before it can auto-detect. MODEL_DIR is usually present so it rarely bit;
      # LANIP (below) is usually absent, which is the #686 silent-no-op.
      MODEL_DIR="$(grep -E '^MODEL_DIR=' "$_c3_env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
      MODEL_DIR="${MODEL_DIR%\"}"; MODEL_DIR="${MODEL_DIR#\"}"   # strip optional surrounding quotes
    fi
    # LANIP for the printed URLs — pin it here when auto-detect can't pick the right NIC, or on
    # hosts without `hostname -I` / `ip` (club-3090 #512). Env wins; auto-detect (c3_lan_ip) is the
    # fallback when neither sets it.
    if [ -z "${LANIP:-}" ]; then
      LANIP="$(grep -E '^LANIP=' "$_c3_env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"   # || true: see MODEL_DIR note above (#686)
      LANIP="${LANIP%\"}"; LANIP="${LANIP#\"}"
      [ -n "$LANIP" ] && export LANIP
    fi
    # HF_TOKEN for the roster downloads (gated repos) — env wins; the repo .env is
    # where users naturally put it, and until now it was silently ignored by the
    # HOST-side hf calls (only the composes read .env) — MoppelMat had to env-prefix
    # the whole setup script (#686). Same read-then-export pattern as LANIP.
    if [ -z "${HF_TOKEN:-}" ]; then
      HF_TOKEN="$(grep -E '^HF_TOKEN=' "$_c3_env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
      HF_TOKEN="${HF_TOKEN%\"}"; HF_TOKEN="${HF_TOKEN#\"}"
      [ -n "$HF_TOKEN" ] && export HF_TOKEN
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
# COMFYUI_OUTPUT_DIR is where ComfyUI WRITES renders ($COMFYUI_ROOT/output) AND what every studio
# output consumer mounts (gallery :8189, orchestrator, tts, step-voice, production). It MUST derive
# from COMFYUI_ROOT — otherwise it falls back to the /mnt default and the gallery serves an EMPTY
# tree while ComfyUI writes renders elsewhere → 404 on generated media on any non-/mnt rig
# (club-3090 #510 follow-on; #531 pinned COMFYUI_ROOT for the *input* side but not the output side).
: "${COMFYUI_OUTPUT_DIR:=${COMFYUI_ROOT}/output}"
export MODEL_DIR COMFYUI_ROOT COMFYUI_MODELS_DIR COMFYUI_OUTPUT_DIR


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

# Persist the resolved COMFYUI_ROOT + COMFYUI_OUTPUT_DIR to repo-root .env so the studio composes —
# launched as `sudo docker compose --env-file .env` — mount the SAME trees this shell derives.
# Without this, the vars are derived in-shell but stripped by sudo AND absent from .env, so:
#   • `${COMFYUI_ROOT:-/mnt/models/comfyui}/models` falls back to /mnt → ComfyUI model dropdowns come
#     up EMPTY (loaders 400 "not in []"; HiDream node "not installed"). club-3090 #510 + #530.
#   • `${COMFYUI_OUTPUT_DIR:-/mnt/models/comfyui/output}` (gallery :8189 + orchestrator / tts /
#     step-voice / production) falls back to /mnt → generated renders 404 in the gallery / don't show
#     in OWUI, while ComfyUI writes them to $COMFYUI_ROOT/output. club-3090 #510 follow-on.
# Write-if-absent PER VAR: never clobbers a hand-set value, and — critically — adds COMFYUI_OUTPUT_DIR
# for users who already have COMFYUI_ROOT pinned from #531 (don't gate on ROOT presence). No-op under
# C3_PATHS_NO_ENV / unwritable .env.
c3_persist_comfy_root() {
  [ -n "${C3_PATHS_NO_ENV:-}" ] && return 0
  local env_file="${C3_ENV_FILE:-${C3_REPO_ROOT:-.}/.env}"
  touch "$env_file" 2>/dev/null || return 0
  grep -qE '^COMFYUI_ROOT='       "$env_file" 2>/dev/null || printf 'COMFYUI_ROOT=%s\n'       "$COMFYUI_ROOT"       >> "$env_file" 2>/dev/null || true
  grep -qE '^COMFYUI_OUTPUT_DIR=' "$env_file" 2>/dev/null || printf 'COMFYUI_OUTPUT_DIR=%s\n' "$COMFYUI_OUTPUT_DIR" >> "$env_file" 2>/dev/null || true
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


# Pre-create EVERY studio bind-mount source dir USER-OWNED before any `sudo docker`
# runs (#715 gap 1). The studio containers run as root under `sudo docker compose`,
# and Docker auto-creates a missing bind-mount source dir ROOT-OWNED — so the later
# non-sudo download step can't write to it, a mess the installer used to create
# itself and then blame the user for. mkdir -p as the CALLING user prevents the
# whole class. If a dir already exists root-owned (a previously-damaged install),
# prevention can't help — detect it and say exactly how to fix it instead.
# Call from setup-ai-studio.sh (before the build) AND gpu-mode's start_comfyui
# (before compose up). Idempotent; fails only on the damaged-install case.
c3_precreate_comfy_mounts() {
  local d bad=""
  # the comfyui compose's host-side mount sources (services/comfyui/docker-compose.yml)
  # + the shared HF cache root.
  for d in \
      "$COMFYUI_ROOT/ComfyUI" \
      "$COMFYUI_ROOT/models" \
      "$COMFYUI_ROOT/input" \
      "$COMFYUI_ROOT/output" \
      "$COMFYUI_ROOT/user" \
      "$COMFYUI_ROOT/pip-cache" \
      "$MODEL_DIR"; do
    mkdir -p "$d" 2>/dev/null || true
    [ -w "$d" ] || bad="$bad $d"
  done
  if [ -n "$bad" ]; then
    echo "ERROR: studio bind-mount dir(s) not writable by $(id -un):$bad" >&2
    echo "       (Usually root-owned leftovers from an earlier run where Docker" >&2
    echo "       auto-created them under sudo. Fix once:)" >&2
    echo "         sudo chown -R $(id -un):$(id -gn)$bad" >&2
    return 1
  fi
  return 0
}

# Hardened `hf` for every studio download script (#715 gap 2) — the same guards the
# main stack mandates, made repo-portable. Every download_*.sh sources this file (or
# inherits via `export -f` from download_studio_models.sh), so their bare `hf download`
# calls transparently gain:
#   • XET OFF by default   — the classic LFS path RESUMES from .incomplete; Xet restarts
#     from 0 on failure, so retries on it are lossy (caller may re-enable via env).
#   • a read timeout       — a dead socket errors instead of hanging forever.
#   • retry-until-success  — transient network errors resume mid-file (6 attempts).
#   • a false-DONE guard   — on a mid-run network loss, huggingface_hub can degrade to
#     "returning existing local_dir" and exit 0 with the payload still *.incomplete
#     (observed live 2026-07-16); rc=0 with leftover .incomplete files is treated as a
#     FAILURE and retried, not reported as success.
#   • a stall watchdog     — HF_HUB_DOWNLOAD_TIMEOUT only bounds SOCKET reads; hf_hub's
#     "Still waiting to acquire lock" wait is unbounded and hangs the whole install
#     (#726 — kodcuserkan, 3 attempts stuck on the same HiDream .lock). If dest bytes
#     stop growing for HF_FETCH_STALL_TIMEOUT (default 300s) the attempt is killed and
#     the retry loop takes over. Tune/disable: HF_FETCH_STALL_TIMEOUT=<secs|0>.
#   • stale-lock clearing  — before each attempt, *.lock files under the dest download
#     cache with NO live holder (fuser; age-only fallback) and ≥5 min age are removed,
#     so a lock orphaned by a killed run (SoftFileLock filesystems / dead holders)
#     can't wedge every future attempt (#726). A lock a LIVE process holds is spared.
# Non-download subcommands pass straight through.

# stale-lock sweep for c3_hf's retry loop (#726) — see the bullet above.
_c3_hf_clear_stale_locks() {
  local _cache="$1/.cache/huggingface/download" _lk _age_min
  [ -d "$_cache" ] || return 0
  # without fuser we can't see a live fcntl holder — demand a longer age instead
  if command -v fuser >/dev/null 2>&1; then _age_min=5; else _age_min=10; fi
  find "$_cache" -name '*.lock' -mmin "+$_age_min" 2>/dev/null | while IFS= read -r _lk; do
    if command -v fuser >/dev/null 2>&1 && fuser -s "$_lk" 2>/dev/null; then
      continue  # a live process holds it — a real concurrent download, leave it
    fi
    echo "[hf-fetch] clearing stale download lock (no live holder, >${_age_min}min): $_lk" >&2
    rm -f "$_lk" 2>/dev/null || true
  done
  return 0
}

hf() {
  if [ "${1:-}" != "download" ]; then command hf "$@"; return; fi
  # recover --local-dir for the false-DONE .incomplete scan (absent -> cache-mode, skip scan)
  local _dest="" _prev="" _a
  for _a in "$@"; do
    [ "$_prev" = "--local-dir" ] && _dest="$_a"
    _prev="$_a"
  done
  local _attempt=1 _max=6 _rc _hfpid _wpid _stall="${HF_FETCH_STALL_TIMEOUT:-300}"
  while :; do
    [ -n "$_dest" ] && _c3_hf_clear_stale_locks "$_dest"
    # setsid → own process GROUP, so the watchdog kill takes any subprocess the
    # downloader spawned with it (an orphaned child would keep holding the lock
    # AND the callers' stdout pipe). Fallback: plain background + single-pid kill.
    if command -v setsid >/dev/null 2>&1; then
      HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
      HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}" \
        setsid hf "$@" &   # exec via PATH — the real binary, not this function
    else
      HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
      HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}" \
        command hf "$@" &
    fi
    _hfpid=$!
    _wpid=""
    if [ -n "$_dest" ] && [ "$_stall" -gt 0 ] 2>/dev/null; then
      (
        _last=-1; _same=0
        while kill -0 "$_hfpid" 2>/dev/null; do
          sleep 15
          _cur=$(du -sb "$_dest" 2>/dev/null | cut -f1) || _cur=0
          if [ "${_cur:-0}" = "$_last" ]; then _same=$((_same + 15)); else _same=0; _last="${_cur:-0}"; fi
          if [ "$_same" -ge "$_stall" ]; then
            echo "[hf-fetch] no byte growth for ${_stall}s — killing the stalled attempt (lock-wait or dead peer); retrying" >&2
            echo "[hf-fetch] if another download is genuinely running, check: pgrep -af 'hf download'" >&2
            kill -- "-$_hfpid" 2>/dev/null || kill "$_hfpid" 2>/dev/null
            exit 0
          fi
        done
      ) &
      _wpid=$!
    fi
    wait "$_hfpid" && _rc=0 || _rc=$?
    [ -n "$_wpid" ] && { kill "$_wpid" 2>/dev/null; wait "$_wpid" 2>/dev/null; } || true
    if [ "$_rc" -eq 0 ]; then
      if [ -n "$_dest" ] && [ -d "$_dest" ] && \
         find "$_dest" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
        echo "[hf-fetch] rc=0 but *.incomplete remains under $_dest (offline-degrade false DONE) — retrying" >&2
        _rc=75  # EX_TEMPFAIL
      else
        return 0
      fi
    fi
    if [ "$_attempt" -ge "$_max" ]; then
      echo "[hf-fetch] giving up after $_max attempts (rc=$_rc)" >&2
      return "$_rc"
    fi
    echo "[hf-fetch] retry $_attempt/$_max in 8s (rc=$_rc) — the classic path resumes from .incomplete" >&2
    _attempt=$((_attempt + 1))
    sleep 8
  done
}
# Inherit into child bash processes (download_studio_models.sh runs each lane script
# as `bash download_X.sh`) so even the two children that don't source this file get it.
export -f hf 2>/dev/null || true
