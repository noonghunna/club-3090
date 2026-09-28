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
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
HELPER="$(cd "$(dirname "$0")/../.." && pwd)/services/comfyui/comfyui-paths.sh"
LOADER_SH="$(cd "$(dirname "$0")/../.." && pwd)/scripts/lib/club-config.sh"
LOADER_PY="$(cd "$(dirname "$0")/../.." && pwd)/scripts/lib/club_config.py"

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

# --- saving (club-3090#1466): c3_persist_comfy_root / c3_resolve_lanip write through the ONE
#     writer into club3090.env (CLUB3090_CONFIG_DIR), no longer into the repo .env. The helper
#     and the loader are copied into a throwaway repo root so its legacy .env is ours, and every
#     case gets a fresh settings dir. Nothing here reads or writes your real settings or .env.
_sroot="$(mktemp -d)"; mkdir -p "$_sroot/services/comfyui" "$_sroot/scripts/lib"
cp "$HELPER" "$_sroot/services/comfyui/comfyui-paths.sh"
cp "$LOADER_SH" "$LOADER_PY" "$_sroot/scripts/lib/"
SHELPER="$_sroot/services/comfyui/comfyui-paths.sh"
_n=0
fresh() { _n=$((_n+1)); CFG="$_sroot/cfg$_n"; rm -f "$_sroot/.env"; }   # a new, empty settings dir
saved() { grep "^$1=" "$CFG/club3090.env" 2>/dev/null | cut -d= -f2-; }   # value in club3090.env
nsaved() { grep -c "^$1=" "$CFG/club3090.env" 2>/dev/null || true; }
# persist [VAR=val ...] — source the helper with a clean studio env, then c3_persist_comfy_root.
# The loader is sourced FIRST, as gpu-mode.sh does: then the writer exists even under
# C3_PATHS_NO_ENV=1, so the "no writes" cases test the helper's own guard, not the loader's absence.
persist() {
  env -u COMFYUI_ROOT -u COMFYUI_MODELS_DIR -u COMFYUI_OUTPUT_DIR -u LANIP -u C3_PATHS_NO_ENV \
    MODEL_DIR=/srv/u/models CLUB3090_CONFIG_DIR="$CFG" "$@" \
    bash -c '. "$2"; . "$1"; c3_persist_comfy_root' _ "$SHELPER" "$_sroot/scripts/lib/club-config.sh" 2>&1
}

# H — c3_persist_comfy_root saves the derived COMFYUI_ROOT when absent (club-3090 #510/#530: the
#     comfyui compose mounts ${COMFYUI_ROOT}/models via --env-file, so without this the container
#     falls back to the /mnt default and mounts an EMPTY tree on any non-/mnt rig).
fresh; out="$(persist)"
chk "persist saves COMFYUI_ROOT to club3090.env when absent" "/srv/u/comfyui" "$(saved COMFYUI_ROOT)"
chk "…and says where" "yes" "$(case "$out" in *"to $CFG/club3090.env"*) echo yes;; *) echo "no: $out";; esac)"
chk "…and never writes the repo .env" "absent" "$([ -e "$_sroot/.env" ] && echo present || echo absent)"

# I — never clobbers a hand-set COMFYUI_ROOT already saved (exactly one line, value unchanged)
fresh; mkdir -p "$CFG"; printf 'COMFYUI_ROOT=/data/custom\n' > "$CFG/club3090.env"
persist >/dev/null
chk "persist respects a saved COMFYUI_ROOT" "/data/custom|1" "$(saved COMFYUI_ROOT)|$(nsaved COMFYUI_ROOT)"
# …and does not pin an output dir derived from a DIFFERENT root next to it (#510: renders 404).
chk "no OUTPUT_DIR saved beside a different saved root" "" "$(saved COMFYUI_OUTPUT_DIR)"

# I2 — a COMFYUI_ROOT in the LEGACY repo .env counts as saved too (still read), and that file
#      is left exactly as it was.
fresh; printf 'COMFYUI_ROOT=/data/legacy\n' > "$_sroot/.env"
persist >/dev/null
chk "a legacy .env COMFYUI_ROOT is not re-saved to club3090.env" "" "$(saved COMFYUI_ROOT)"
chk "…and the legacy .env is untouched" "COMFYUI_ROOT=/data/legacy" "$(cat "$_sroot/.env")"

# I3 — a COMFYUI_ROOT the CALLER exported for this run (≠ what MODEL_DIR derives) is not saved,
#      and neither is an output dir derived from it.
fresh; persist COMFYUI_ROOT=/data/shell >/dev/null
chk "a shell-only COMFYUI_ROOT is not saved" "|" "$(saved COMFYUI_ROOT)|$(saved COMFYUI_OUTPUT_DIR)"

# I4 — …but a value a PARENT studio script derived and exported (equal to the derivation) is
#      saved: that's how every download_*.sh reaches this (setup-ai-studio.sh exports first).
fresh; persist COMFYUI_ROOT=/srv/u/comfyui COMFYUI_OUTPUT_DIR=/srv/u/comfyui/output >/dev/null
chk "a parent-derived export is still saved" "/srv/u/comfyui|/srv/u/comfyui/output" "$(saved COMFYUI_ROOT)|$(saved COMFYUI_OUTPUT_DIR)"

# I5 — C3_PATHS_NO_ENV=1 writes nothing.
fresh; persist C3_PATHS_NO_ENV=1 >/dev/null
chk "C3_PATHS_NO_ENV=1 saves nothing" "absent" "$([ -e "$CFG/club3090.env" ] && echo present || echo absent)"

# I6 — a value the writer refuses is reported, not stored, and never fatal under set -e (#686).
fresh
out="$(env -u COMFYUI_ROOT -u COMFYUI_MODELS_DIR -u COMFYUI_OUTPUT_DIR -u C3_PATHS_NO_ENV \
  MODEL_DIR='/srv/u$x/models' CLUB3090_CONFIG_DIR="$CFG" \
  bash -c 'set -euo pipefail; . "$1"; c3_persist_comfy_root; echo SURVIVED' _ "$SHELPER" 2>&1)"
chk "a refused value doesn't stop a set -e caller" "yes" "$(case "$out" in *SURVIVED*) echo yes;; *) echo "no: $out";; esac)"
chk "…and the writer's reason is shown" "yes" "$(case "$out" in *"couldn't save COMFYUI_ROOT"*"value contains '\$'"*) echo yes;; *) echo "no: $out";; esac)"
chk "…and nothing is stored" "" "$(saved COMFYUI_ROOT)"
# An unwritable settings dir: same — reported, not fatal.
: > "$_sroot/not-a-dir"
out="$(env -u COMFYUI_ROOT -u COMFYUI_MODELS_DIR -u COMFYUI_OUTPUT_DIR -u C3_PATHS_NO_ENV \
  MODEL_DIR=/srv/u/models CLUB3090_CONFIG_DIR="$_sroot/not-a-dir/cfg" \
  bash -c 'set -euo pipefail; . "$1"; c3_persist_comfy_root; echo SURVIVED' _ "$SHELPER" 2>&1)"
chk "an unwritable settings dir doesn't stop a set -e caller, and says why" "yes" \
  "$(case "$out" in *"couldn't save COMFYUI_ROOT to"*"Not a directory"*SURVIVED*) echo yes;; *) echo "no: $out";; esac)"

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

# M — c3_persist_comfy_root saves COMFYUI_OUTPUT_DIR when absent
fresh; persist >/dev/null
chk "persist saves COMFYUI_OUTPUT_DIR when absent" "/srv/u/comfyui/output" "$(saved COMFYUI_OUTPUT_DIR)"

# N — THE migration case: COMFYUI_ROOT already saved (from #531 — here in the legacy repo .env)
#     but no OUTPUT_DIR → persist must ADD OUTPUT_DIR (not bail on ROOT presence) and leave ROOT
#     untouched.
fresh; printf 'COMFYUI_ROOT=/srv/u/comfyui\n' > "$_sroot/.env"
persist >/dev/null
chk "adds OUTPUT_DIR when only ROOT was saved" "/srv/u/comfyui/output" "$(saved COMFYUI_OUTPUT_DIR)"
chk "ROOT untouched in the migration case" "COMFYUI_ROOT=/srv/u/comfyui||" \
  "$(cat "$_sroot/.env")||$(saved COMFYUI_ROOT)"

# O — never clobbers a hand-set COMFYUI_OUTPUT_DIR (exactly one line, unchanged)
fresh; mkdir -p "$CFG"; printf 'COMFYUI_OUTPUT_DIR=/data/custom-out\n' > "$CFG/club3090.env"
persist >/dev/null
chk "persist respects a saved COMFYUI_OUTPUT_DIR" "/data/custom-out|1" \
  "$(saved COMFYUI_OUTPUT_DIR)|$(nsaved COMFYUI_OUTPUT_DIR)"

# --- LANIP (#512): an auto-detected IP is saved to club3090.env, write-if-absent.
lanip() {  # lanip [VAR=val ...] — resolve with `hostname -I` answering 10.20.30.40 (loader first, as above)
  env -u LANIP -u C3_PATHS_NO_ENV MODEL_DIR=/srv/u/models CLUB3090_CONFIG_DIR="$CFG" "$@" bash -c '
    hostname() { echo "172.17.0.1 10.20.30.40"; }
    . "$2"; . "$1"; c3_resolve_lanip; echo "LANIP=$LANIP"' _ "$SHELPER" "$_sroot/scripts/lib/club-config.sh" 2>&1
}
fresh; out="$(lanip)"
chk "an auto-detected LANIP is saved to club3090.env" "10.20.30.40" "$(saved LANIP)"
chk "…says where" "yes" "$(case "$out" in *"saved to $CFG/club3090.env"*) echo yes;; *) echo "no: $out";; esac)"
chk "…and never writes the repo .env" "absent" "$([ -e "$_sroot/.env" ] && echo present || echo absent)"
fresh; printf 'LANIP=10.9.9.9\n' > "$_sroot/.env"; out="$(lanip)"
chk "a LANIP in the legacy repo .env is used and not re-saved" "LANIP=10.9.9.9|" "$(printf '%s' "$out" | tail -n1)|$(saved LANIP)"
fresh; mkdir -p "$CFG"; printf 'LANIP=10.8.8.8\n' > "$CFG/club3090.env"; out="$(lanip)"
chk "a saved LANIP is used and left alone" "LANIP=10.8.8.8|1" "$(printf '%s' "$out" | tail -n1)|$(nsaved LANIP)"
fresh; out="$(lanip LANIP=10.7.7.7)"
chk "a shell LANIP is used and not saved" "LANIP=10.7.7.7|absent" "$(printf '%s' "$out" | tail -n1)|$([ -e "$CFG/club3090.env" ] && echo present || echo absent)"
fresh; out="$(lanip C3_PATHS_NO_ENV=1)"
chk "C3_PATHS_NO_ENV=1: detected, not saved, nothing said" "LANIP=10.20.30.40|absent" "$out|$([ -e "$CFG/club3090.env" ] && echo present || echo absent)"
fresh
out="$(env -u LANIP -u C3_PATHS_NO_ENV MODEL_DIR=/srv/u/models CLUB3090_CONFIG_DIR="$CFG" bash -c '
  hostname() { :; }; ip() { :; }   # neither finds an address
  . "$1"; c3_resolve_lanip; echo "LANIP=$LANIP"' _ "$SHELPER" 2>&1)"
chk "nothing detected → localhost, and the hint names club3090.env" "yes" \
  "$(case "$out" in *"Set LANIP=<your-machine-ip> in $CFG/club3090.env"*"LANIP=localhost") echo yes;; *) echo "no: $out";; esac)"
chk "…and nothing is saved" "absent" "$([ -e "$CFG/club3090.env" ] && echo present || echo absent)"
rm -rf "$_sroot"

# P (#686) — sourcing under a `set -e`+pipefail caller (setup-ai-studio.sh) MUST NOT
#     silently exit when .env lacks a LANIP line. The grep-no-match returned 1, pipefail
#     propagated it, the assignment failed, and set -e killed the caller BEFORE the
#     LAN-IP auto-detect could run → "no output at all". Copy the helper into a throwaway
#     repo root so C3_REPO_ROOT/.env (a MODEL_DIR-only .env) is controlled.
_reroot="$(mktemp -d)"; mkdir -p "$_reroot/services/comfyui"
cp "$HELPER" "$_reroot/services/comfyui/comfyui-paths.sh"
mkdir -p "$_reroot/scripts/lib" && cp "$LOADER_SH" "$LOADER_PY" "$_reroot/scripts/lib/"   # the helper reads settings through the loader (#1466)
printf 'MODEL_DIR=%s/models\n' "$_reroot" > "$_reroot/.env"   # NO LANIP line
( cd "$_reroot" && bash -c 'set -euo pipefail; . services/comfyui/comfyui-paths.sh' ) >/dev/null 2>&1
chk "no silent set-e exit when .env lacks LANIP (#686)" "0" "$?"
rm -rf "$_reroot"

# --- HF_TOKEN from .env (#686): the token in repo .env was silently ignored by the
#     HOST-side hf calls (only the composes read .env), forcing users to env-prefix
#     the whole setup script. The paths lib now reads + exports it (env wins).
_tkroot="$(mktemp -d)"; mkdir -p "$_tkroot/services/comfyui"
cp "$HELPER" "$_tkroot/services/comfyui/comfyui-paths.sh"
mkdir -p "$_tkroot/scripts/lib" && cp "$LOADER_SH" "$LOADER_PY" "$_tkroot/scripts/lib/"   # the helper reads settings through the loader (#1466)
printf 'HF_TOKEN=hf_dotenv_test\n' > "$_tkroot/.env"
chk "HF_TOKEN read from .env + exported (#686)" "hf_dotenv_test" \
  "$(env -u HF_TOKEN MODEL_DIR=/home/u/models bash -c '. "'"$_tkroot"'/services/comfyui/comfyui-paths.sh"; printf "%s" "${HF_TOKEN:-}"')"
chk "explicit env HF_TOKEN wins over .env" "hf_env_wins" \
  "$(env HF_TOKEN=hf_env_wins MODEL_DIR=/home/u/models bash -c '. "'"$_tkroot"'/services/comfyui/comfyui-paths.sh"; printf "%s" "${HF_TOKEN:-}"')"
rm -rf "$_tkroot"

if [ "$fails" -eq 0 ]; then echo "PASS: comfyui-paths derivation"; exit 0; else echo "FAIL: $fails assertion(s)"; exit 1; fi
