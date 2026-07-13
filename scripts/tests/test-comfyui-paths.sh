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

# H — c3_persist_comfy_root pins the derived COMFYUI_ROOT into .env when absent (club-3090
#     #510/#530: the comfyui compose mounts ${COMFYUI_ROOT}/models via --env-file, so without this
#     the container falls back to the /mnt default and mounts an EMPTY tree on any non-/mnt rig).
_tmpenv="$(mktemp)"; : > "$_tmpenv"
env -u COMFYUI_ROOT -u COMFYUI_MODELS_DIR MODEL_DIR=/home/u/models C3_ENV_FILE="$_tmpenv" \
  bash -c '. "'"$HELPER"'"; c3_persist_comfy_root' >/dev/null 2>&1
chk "persist writes COMFYUI_ROOT when absent" "COMFYUI_ROOT=/home/u/comfyui" "$(grep '^COMFYUI_ROOT=' "$_tmpenv")"

# I — never clobbers a hand-set COMFYUI_ROOT already in .env (exactly one line, value unchanged)
printf 'COMFYUI_ROOT=/data/custom\n' > "$_tmpenv"
env -u COMFYUI_ROOT -u COMFYUI_MODELS_DIR MODEL_DIR=/home/u/models C3_ENV_FILE="$_tmpenv" \
  bash -c '. "'"$HELPER"'"; c3_persist_comfy_root' >/dev/null 2>&1
chk "persist respects an existing COMFYUI_ROOT" "/data/custom|1" \
  "$(grep '^COMFYUI_ROOT=' "$_tmpenv" | cut -d= -f2-)|$(grep -c '^COMFYUI_ROOT=' "$_tmpenv")"
rm -f "$_tmpenv"

# --- COMFYUI_OUTPUT_DIR (#510 follow-on): the gallery :8189 + orchestrator / tts / step-voice /
#     production mount ${COMFYUI_OUTPUT_DIR}; it MUST derive from COMFYUI_ROOT or ComfyUI writes
#     renders to $COMFYUI_ROOT/output while the gallery serves the empty /mnt default → 404.
out() { env "$@" bash -c '. "'"$HELPER"'"; printf "%s" "$COMFYUI_OUTPUT_DIR"'; }

# J — derives as $COMFYUI_ROOT/output (home case — where ComfyUI actually writes)
chk "output dir → \$COMFYUI_ROOT/output" "/home/u/comfyui/output" \
  "$(out -u COMFYUI_ROOT -u COMFYUI_OUTPUT_DIR MODEL_DIR=/home/u/models)"
# K — follows an explicit COMFYUI_ROOT
chk "output dir follows explicit root" "/data/cr/output" \
  "$(out -u COMFYUI_OUTPUT_DIR COMFYUI_ROOT=/data/cr MODEL_DIR=/x)"
# L — respects an explicit COMFYUI_OUTPUT_DIR
chk "explicit output dir respected" "/data/out" \
  "$(out -u COMFYUI_ROOT COMFYUI_OUTPUT_DIR=/data/out MODEL_DIR=/x)"

_tmpenv="$(mktemp)"
# M — c3_persist_comfy_root pins COMFYUI_OUTPUT_DIR when absent
: > "$_tmpenv"
env -u COMFYUI_ROOT -u COMFYUI_OUTPUT_DIR MODEL_DIR=/home/u/models C3_ENV_FILE="$_tmpenv" \
  bash -c '. "'"$HELPER"'"; c3_persist_comfy_root' >/dev/null 2>&1
chk "persist writes COMFYUI_OUTPUT_DIR when absent" "COMFYUI_OUTPUT_DIR=/home/u/comfyui/output" \
  "$(grep '^COMFYUI_OUTPUT_DIR=' "$_tmpenv")"

# N — THE migration case: COMFYUI_ROOT already pinned (from #531) but no OUTPUT_DIR → persist must
#     ADD OUTPUT_DIR (not bail on ROOT presence) and leave ROOT untouched.
printf 'COMFYUI_ROOT=/home/u/comfyui\n' > "$_tmpenv"
env -u COMFYUI_ROOT -u COMFYUI_OUTPUT_DIR MODEL_DIR=/home/u/models C3_ENV_FILE="$_tmpenv" \
  bash -c '. "'"$HELPER"'"; c3_persist_comfy_root' >/dev/null 2>&1
chk "adds OUTPUT_DIR when only ROOT was pinned" "COMFYUI_OUTPUT_DIR=/home/u/comfyui/output" \
  "$(grep '^COMFYUI_OUTPUT_DIR=' "$_tmpenv")"
chk "ROOT untouched in the migration case" "/home/u/comfyui|1" \
  "$(grep '^COMFYUI_ROOT=' "$_tmpenv" | cut -d= -f2-)|$(grep -c '^COMFYUI_ROOT=' "$_tmpenv")"

# O — never clobbers a hand-set COMFYUI_OUTPUT_DIR (exactly one line, unchanged)
printf 'COMFYUI_OUTPUT_DIR=/data/custom-out\n' > "$_tmpenv"
env -u COMFYUI_ROOT -u COMFYUI_OUTPUT_DIR MODEL_DIR=/home/u/models C3_ENV_FILE="$_tmpenv" \
  bash -c '. "'"$HELPER"'"; c3_persist_comfy_root' >/dev/null 2>&1
chk "persist respects an existing COMFYUI_OUTPUT_DIR" "/data/custom-out|1" \
  "$(grep '^COMFYUI_OUTPUT_DIR=' "$_tmpenv" | cut -d= -f2-)|$(grep -c '^COMFYUI_OUTPUT_DIR=' "$_tmpenv")"
rm -f "$_tmpenv"

# P (#686) — sourcing under a `set -e`+pipefail caller (setup-ai-studio.sh) MUST NOT
#     silently exit when .env lacks a LANIP line. The grep-no-match returned 1, pipefail
#     propagated it, the assignment failed, and set -e killed the caller BEFORE the
#     LAN-IP auto-detect could run → "no output at all". Copy the helper into a throwaway
#     repo root so C3_REPO_ROOT/.env (a MODEL_DIR-only .env) is controlled.
_reroot="$(mktemp -d)"; mkdir -p "$_reroot/services/comfyui"
cp "$HELPER" "$_reroot/services/comfyui/comfyui-paths.sh"
printf 'MODEL_DIR=%s/models\n' "$_reroot" > "$_reroot/.env"   # NO LANIP line
( cd "$_reroot" && bash -c 'set -euo pipefail; . services/comfyui/comfyui-paths.sh' ) >/dev/null 2>&1
chk "no silent set-e exit when .env lacks LANIP (#686)" "0" "$?"
rm -rf "$_reroot"

if [ "$fails" -eq 0 ]; then echo "PASS: comfyui-paths derivation"; exit 0; else echo "FAIL: $fails assertion(s)"; exit 1; fi
