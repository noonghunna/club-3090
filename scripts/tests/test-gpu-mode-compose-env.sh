#!/usr/bin/env bash
# test-gpu-mode-compose-env — how gpu-mode hands settings to `sudo docker compose`
# (club-3090#1466).
#
# sudo strips the environment, so gpu-mode passes the resolved settings as a 0600
# --env-file. Two things must hold, and both used to fail:
#   1. The file is built FRESH for every compose call. A snapshot taken when gpu-mode
#      started missed anything saved during the run — start_comfyui saves
#      COMFYUI_ROOT and then starts ComfyUI, which then mounted the compose default.
#   2. Studio paths the user EXPORTED reach the containers through sudo, without
#      being saved. Paths comfyui-paths.sh merely DERIVED are not passed, so they can
#      never override a value saved in the settings.
#
# Offline: `docker` and `sudo` are PATH shims (prepended INLINE on each call) and
# CLUB3090_DIR is a scratch tree. No container is touched.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

mkdir -p "$T/bin" "$T/stub/config" "$T/stub/containers" "$T/club/services" "$T/tmp" "$T/cfg"
# Two support services that drifted, so `gpu-mode upgrade` recreates both: two compose calls.
for s in searxng spark-dashboard; do
  mkdir -p "$T/club/services/$s"; echo "services: {}" > "$T/club/services/$s/docker-compose.yml"
done
printf '{"name":"searxng","services":{"searxng":{"image":"searxng/searxng:new","container_name":"searxng"}}}\n' > "$T/stub/config/searxng.json"
printf '{"name":"spark-dashboard","services":{"spark-dashboard":{"image":"spark:new","container_name":"spark-dashboard"}}}\n' > "$T/stub/config/spark-dashboard.json"
printf 'searxng/searxng:old\ntrue\n\n\n' > "$T/stub/containers/searxng"
printf 'spark:old\ntrue\n\n\n' > "$T/stub/containers/spark-dashboard"
# The REAL comfyui-paths.sh, so gpu-mode derives and exports COMFYUI_ROOT as on a rig.
mkdir -p "$T/club/services/comfyui"; cp "$ROOT/services/comfyui/comfyui-paths.sh" "$T/club/services/comfyui/"

cat > "$T/bin/sudo" <<'EOF'
#!/usr/bin/env bash
# log the VAR=val assignments sudo would pass through, then run the command
while [[ "${1:-}" == *=* ]]; do echo "SUDO_ENV $1" >> "$STUB_LOG"; shift; done
exec "$@"
EOF
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  compose)
    if [[ " $* " == *" config "* ]]; then cat "$STUB_DIR/config/$(basename "$PWD").json"; exit 0; fi
    if [[ " $* " == *" up "* ]]; then
      n=$(( $(cat "$STUB_DIR/ups" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$STUB_DIR/ups"
      prev=""; for a in "$@"; do [[ "$prev" == --env-file ]] && cp "$a" "$STUB_DIR/envfile.$n"; prev="$a"; done
      echo "UP $n $(basename "$PWD")" >> "$STUB_LOG"
      # A setting saved between the two calls, as start_comfyui saves COMFYUI_ROOT.
      [[ $n -eq 1 ]] && printf 'MIDRUN=saved\n' >> "$MIDRUN_FILE"
    fi
    exit 0 ;;
  inspect)
    f="$STUB_DIR/containers/$2"; [ -f "$f" ] || exit 1
    case "$4" in *Config.Image*) sed -n 1p "$f" ;; *State.Running*) sed -n 2p "$f" ;; esac ;;
  *) exit 0 ;;
esac
EOF
chmod +x "$T/bin/sudo" "$T/bin/docker"
[[ "$(PATH="$T/bin:$PATH" command -v docker)" == "$T/bin/docker" && "$(PATH="$T/bin:$PATH" command -v sudo)" == "$T/bin/sudo" ]] \
  || { echo "shims not first on PATH — refusing to run" >&2; exit 1; }

run_upgrade() {  # [extra env assignments...]
  rm -f "$T/stub/ups" "$T/stub/envfile."* "$T/stub.log"
  printf 'FIRST=1\n' > "$T/cfg/club3090.env"
  env -u COMFYUI_ROOT -u COMFYUI_OUTPUT_DIR -u MODEL_DIR "$@" PATH="$T/bin:$PATH" TMPDIR="$T/tmp" \
    CLUB3090_DIR="$T/club" CLUB3090_CONFIG_DIR="$T/cfg" STUB_DIR="$T/stub" STUB_LOG="$T/stub.log" \
    MIDRUN_FILE="$T/cfg/club3090.env" HOME="$T/home" timeout 120 bash "$ROOT/scripts/gpu-mode.sh" upgrade --no-backup > "$T/out" 2>&1
}

# 1. a fresh env file per compose call
run_upgrade
ups="$(command grep -c '^UP ' "$T/stub.log" 2>/dev/null || echo 0)"
[[ "$ups" == 2 ]] && ok "gpu-mode upgrade recreated both drifted services (two compose calls)" \
  || { bad "expected 2 compose up calls, got $ups"; sed 's/^/      /' "$T/out" | tail -15 >&2; }
if [[ -f "$T/stub/envfile.1" && -f "$T/stub/envfile.2" ]]; then
  command grep -qxF "FIRST='1'" "$T/stub/envfile.1" && ! command grep -q MIDRUN "$T/stub/envfile.1" \
    && ok "call 1 got the settings as they were (FIRST, no MIDRUN yet)" || bad "call 1 env file: $(cat "$T/stub/envfile.1")"
  command grep -qxF "MIDRUN='saved'" "$T/stub/envfile.2" \
    && ok "call 2 sees a setting saved during the run — the file is built per call, not snapshotted" \
    || bad "call 2 missed the mid-run save: $(tr '\n' ' ' < "$T/stub/envfile.2")"
else
  bad "a compose call got no --env-file ($(ls "$T/stub" | command grep envfile | tr '\n' ' '))"
fi
left="$(ls -A "$T/tmp" | command grep -E '^club3090-compose-' || true)"
[[ -z "$left" ]] && ok "every per-call env file is removed" || bad "env files left behind: $left"

# 2. studio paths: exported ones go through sudo, derived ones don't
command grep -q '^SUDO_ENV COMFYUI_ROOT=' "$T/stub.log" \
  && bad "a DERIVED COMFYUI_ROOT was passed through sudo — it would override a saved one: $(command grep -m1 '^SUDO_ENV COMFYUI_ROOT' "$T/stub.log")" \
  || ok "COMFYUI_ROOT derived by comfyui-paths.sh is not passed through sudo"
run_upgrade COMFYUI_ROOT=/exported/comfyui
n="$(command grep -c '^SUDO_ENV COMFYUI_ROOT=/exported/comfyui$' "$T/stub.log" 2>/dev/null || echo 0)"
[[ "$n" == 2 ]] && ok "an EXPORTED COMFYUI_ROOT reaches every sudo docker compose call ($n/2)" \
  || bad "exported COMFYUI_ROOT reached $n of 2 calls"

[[ $fail -eq 0 ]] && echo "test-gpu-mode-compose-env: ok" || echo "test-gpu-mode-compose-env: FAIL"
exit $fail
