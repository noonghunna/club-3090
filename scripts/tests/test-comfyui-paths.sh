#!/usr/bin/env bash
# Guards services/comfyui/comfyui-paths.sh — the shared ComfyUI path derivation.
# Regression guard for the sumo Discord report (2026-06-27): the ai-studio download
# ignored MODEL_DIR and hard-failed creating /mnt/models/comfyui/... ("mkdir: Permission
# denied") on any rig whose models don't live under /mnt.
#
# Asserts: COMFYUI_ROOT / COMFYUI_MODELS_DIR derive as a "comfyui" sibling of MODEL_DIR,
# stay backward-compatible with the reference rig's /mnt layout, and respect explicit
# overrides.
set -uo pipefail
HELPER="$(cd "$(dirname "$0")/../.." && pwd)/services/comfyui/comfyui-paths.sh"

[ -f "$HELPER" ] || { echo "FAIL: helper not found: $HELPER"; exit 1; }

fails=0
chk() {  # chk <desc> <expected> <actual>
  if [ "$2" = "$3" ]; then echo "  ok: $1"; else echo "  FAIL: $1 — expected '$2', got '$3'"; fails=$((fails+1)); fi
}
# Run the helper in a clean subshell (MODEL_DIR set so the .env lookup is skipped →
# deterministic) and echo "COMFYUI_ROOT|COMFYUI_MODELS_DIR".
derive() { env "$@" bash -c '. "'"$HELPER"'"; printf "%s|%s" "$COMFYUI_ROOT" "$COMFYUI_MODELS_DIR"'; }

# A — reference rig default (backward-compatible): /mnt/models/huggingface -> /mnt/models/comfyui
got="$(derive -u COMFYUI_ROOT -u COMFYUI_MODELS_DIR MODEL_DIR=/mnt/models/huggingface)"
chk "rig default root"   "/mnt/models/comfyui"        "${got%|*}"
chk "rig default models" "/mnt/models/comfyui/models" "${got#*|}"

# B — custom home cache (the sumo case): /home/u/models -> /home/u/comfyui
got="$(derive -u COMFYUI_ROOT -u COMFYUI_MODELS_DIR MODEL_DIR=/home/u/models)"
chk "home root"   "/home/u/comfyui"        "${got%|*}"
chk "home models" "/home/u/comfyui/models" "${got#*|}"

# C — explicit COMFYUI_ROOT respected (models follows it)
got="$(derive -u COMFYUI_MODELS_DIR COMFYUI_ROOT=/data/cr MODEL_DIR=/home/u/models)"
chk "explicit root respected" "/data/cr"        "${got%|*}"
chk "models follows root"     "/data/cr/models" "${got#*|}"

# D — explicit COMFYUI_MODELS_DIR respected
got="$(derive -u COMFYUI_ROOT COMFYUI_MODELS_DIR=/data/m MODEL_DIR=/x)"
chk "explicit models respected" "/data/m" "${got#*|}"

# E — zero-config default (no MODEL_DIR, .env skipped): a USER-OWNED $HOME tree, NOT /mnt (#503)
got="$(derive -u MODEL_DIR -u COMFYUI_ROOT -u COMFYUI_MODELS_DIR C3_PATHS_NO_ENV=1 HOME=/home/u)"
chk "zero-config root → \$HOME"   "/home/u/comfyui"        "${got%|*}"
chk "zero-config models → \$HOME" "/home/u/comfyui/models" "${got#*|}"

# F — HOME-less (CI/root) keeps the legacy /mnt default
got="$(derive -u MODEL_DIR -u COMFYUI_ROOT -u COMFYUI_MODELS_DIR -u HOME C3_PATHS_NO_ENV=1)"
chk "HOME-less models → /mnt legacy" "/mnt/models/comfyui/models" "${got#*|}"

# G — c3_lan_ip prefers a LAN address over a docker bridge, even when the bridge is listed first
lan="$(hostname() { echo '172.17.0.1 192.168.1.50 10.0.0.2'; }; . "$HELPER"; c3_lan_ip)"
chk "lan_ip prefers LAN over 172.x bridge" "192.168.1.50" "$lan"

if [ "$fails" -eq 0 ]; then echo "PASS: comfyui-paths derivation"; exit 0; else echo "FAIL: $fails assertion(s)"; exit 1; fi
