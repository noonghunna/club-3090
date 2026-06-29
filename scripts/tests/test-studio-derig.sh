#!/usr/bin/env bash
# Guards against dev-rig defaults leaking into the studio setup scripts.
# Regression guard for club-3090 #503 + #504 (sumo-dandan, 2026-06-28):
#   #503 — standalone services/comfyui/download_*.sh fell back to the rig path
#          /mnt/models/comfyui/models instead of deriving COMFYUI_MODELS_DIR from
#          MODEL_DIR → "mkdir: Permission denied" on a clean machine. Fix: each
#          download_*.sh sources comfyui-paths.sh before its ROOT= line.
#   #504 — gpu-mode.sh hard-coded the rig LAN IP 192.168.86.33 in its URL banners.
#          Fix: auto-detect LANIP (LANIP-overridable), like setup-ai-studio.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fails=0
chk() { if [ "$1" = ok ]; then echo "  ok: $2"; else echo "  FAIL: $2"; fails=$((fails+1)); fi; }

# 1. No script anywhere may hard-code the dev rig's LAN IP (this guard file excepted —
#    it names the IP in its own comment + grep pattern).
hits="$(grep -rl '192\.168\.86\.33' "$ROOT/scripts" "$ROOT/services" \
        --exclude="$(basename "$0")" 2>/dev/null || true)"
[ -z "$hits" ] && chk ok "no hardcoded rig IP 192.168.86.33" \
  || { chk fail "hardcoded rig IP still present in: $hits"; }

# 2. gpu-mode.sh derives LANIP (override + localhost fallback), no rig IP.
if grep -q 'LANIP="${LANIP:-' "$ROOT/scripts/gpu-mode.sh" \
   && grep -q 'LANIP:-localhost' "$ROOT/scripts/gpu-mode.sh"; then
  chk ok "gpu-mode.sh auto-detects LANIP with localhost fallback"
else
  chk fail "gpu-mode.sh missing LANIP auto-detect / localhost fallback"
fi

# 3. Every standalone download_*.sh that references the rig fallback also sources
#    comfyui-paths.sh (so a direct run derives COMFYUI_MODELS_DIR from MODEL_DIR).
for f in "$ROOT"/services/comfyui/download_*.sh; do
  grep -q 'COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models' "$f" || continue   # not a comfy-models downloader
  if grep -q 'comfyui-paths.sh' "$f"; then
    chk ok "sources comfyui-paths.sh: $(basename "$f")"
  else
    chk fail "$(basename "$f") uses the rig fallback but never sources comfyui-paths.sh"
  fi
done

# 4. Both launchers detect the LAN IP via the SHARED c3_lan_ip helper (no inline drift).
for s in scripts/gpu-mode.sh scripts/setup-ai-studio.sh; do
  if grep -q 'c3_lan_ip' "$ROOT/$s"; then chk ok "$s uses shared c3_lan_ip"
  else chk fail "$s does not use the shared c3_lan_ip helper"; fi
done

if [ "$fails" -eq 0 ]; then echo "PASS: studio scripts carry no dev-rig defaults"; exit 0
else echo "FAIL: $fails assertion(s)"; exit 1; fi
