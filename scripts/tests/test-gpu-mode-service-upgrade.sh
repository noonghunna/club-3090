#!/usr/bin/env bash
# test-gpu-mode-service-upgrade — `gpu-mode status` reports support-service image
# drift, and `gpu-mode upgrade` recreates ONLY the running services behind their
# pin, backing up Qdrant's volume first.
#
# WHY THIS TEST EXISTS
# --------------------
# A `git pull` that bumps a pinned service image (#1436) changes nothing that is
# already running: only `docker compose up -d` recreates a container on the new
# image, and switch.sh / reboots / `docker restart` keep the old one. LiteLLM ran
# a six-month-old `main-latest` build unnoticed that way. The failure modes worth
# guarding are silent ones: upgrade touching a STOPPED service (starting things a
# mode deliberately stopped), recreating Qdrant BEFORE its backup (its storage
# migrates forward, one way), or status calling a drifted service current.
#
# Offline: `docker` and `sudo` are PATH shims driven by fixture files, and
# CLUB3090_DIR points at a scratch tree. No containers are touched.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
fail=0
bad() { echo "✗ $1" >&2; fail=1; }

mkdir -p "$T/bin" "$T/stub/config" "$T/stub/containers" "$T/club"
for s in openwebui litellm qdrant searxng spark-dashboard; do
  mkdir -p "$T/club/services/$s"
  echo "services: {}" > "$T/club/services/$s/docker-compose.yml"
done
# Pins, as `docker compose config --format json` would report them.
pin() { printf '{"name":"%s","services":{"%s":{"image":"%s","container_name":"%s"}}}\n' "$1" "$2" "$3" "$4" > "$T/stub/config/$1.json"; }
pin openwebui       open-webui      ghcr.io/open-webui/open-webui:v0.11.4       open-webui
pin litellm         litellm         ghcr.io/berriai/litellm:v1.100.3            litellm
pin qdrant          qdrant          qdrant/qdrant:v1.19.1                       qdrant
pin searxng         searxng         searxng/searxng:2026.9.25-12f8b6515         searxng
pin spark-dashboard spark-dashboard ghcr.io/niklasfrick/spark-dashboard:v0.14.0 spark-dashboard
# Containers: <image>\n<running>\n<volume mounted at /qdrant/storage>
ctr() { printf '%s\n%s\n%s\n%s\n' "$2" "$3" "${4:-}" "${5:-}" > "$T/stub/containers/$1"; }   # line 4: an env line
ctr litellm         ghcr.io/berriai/litellm:v1.100.3            true            # current
ctr qdrant          qdrant/qdrant:v1.18.0                       true qdrant_qdrant-data   # DRIFT
ctr searxng         searxng/searxng:2026.9.20-fdd8525b1         false           # stopped (and old)
ctr spark-dashboard ghcr.io/niklasfrick/spark-dashboard:v0.14.0 true            # current
# open-webui: no container at all

cat > "$T/bin/sudo" <<'EOF'
#!/usr/bin/env bash
# drop leading VAR=val assignments, like sudo would pass them through
while [[ "${1:-}" == *=* ]]; do shift; done
exec "$@"
EOF
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
echo "[$(basename "$PWD")] docker $*" >> "$STUB_LOG"
case "$1" in
  compose)
    if [[ " $* " == *" config "* ]]; then cat "$STUB_DIR/config/$(basename "$PWD").json"; fi
    exit 0 ;;
  inspect)
    f="$STUB_DIR/containers/$2"; [ -f "$f" ] || exit 1
    case "$4" in
      *Config.Image*)   sed -n 1p "$f" ;;
      *State.Running*)  sed -n 2p "$f" ;;
      *Mounts*)         sed -n 3p "$f" ;;
      *Config.Env*)     sed -n 4p "$f" ;;
    esac ;;
  run) printf 'fake-tar-stream' ;;
  *) exit 0 ;;
esac
EOF
chmod +x "$T/bin/sudo" "$T/bin/docker"
export STUB_DIR="$T/stub" CLUB3090_DIR="$T/club"
# Every gpu-mode call below carries PATH="$T/bin:$PATH" inline (the neutralizer
# test-tests-never-launch recognizes). Refuse to run at all unless that PATH
# really resolves docker and sudo to the shims: a missing shim would otherwise
# let `upgrade` recreate the rig's REAL services.
for b in docker sudo; do
  [ "$(PATH="$T/bin:$PATH" command -v "$b")" = "$T/bin/$b" ] || { echo "✗ $b shim not first on PATH — refusing to run gpu-mode" >&2; exit 1; }
done

# ── 1. upgrade recreates only the drifted running service, backup first ─────────
export STUB_LOG="$T/log1"; : > "$STUB_LOG"
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/gpu-mode.sh" upgrade 2>&1)"; rc=$?
[ "$rc" -eq 0 ] || bad "upgrade exited $rc: $out"
ups="$(command grep -E 'docker compose .* up -d' "$STUB_LOG" || true)"
[[ "$ups" == "[qdrant]"* ]] || bad "upgrade must recreate qdrant (drifted, running); compose up calls were: ${ups:-none}"
for s in litellm searxng openwebui spark-dashboard; do
  command grep -qE "^\[$s\] docker compose .* up -d" "$STUB_LOG" && bad "upgrade must NOT recreate $s (current, stopped or missing)"
done
stop_ln=$(command grep -n 'docker stop qdrant' "$STUB_LOG" | head -1 | cut -d: -f1)
run_ln=$(command grep -n 'docker run .*qdrant_qdrant-data:/data:ro' "$STUB_LOG" | head -1 | cut -d: -f1)
up_ln=$(command grep -n '^\[qdrant\] docker compose .* up -d' "$STUB_LOG" | head -1 | cut -d: -f1)
if [ -z "$stop_ln" ] || [ -z "$run_ln" ] || [ -z "$up_ln" ] || [ "$stop_ln" -gt "$run_ln" ] || [ "$run_ln" -gt "$up_ln" ]; then
  bad "qdrant must be stopped, backed up, THEN recreated (stop=$stop_ln backup=$run_ln up=$up_ln)"
fi
ls "$T/club/backups/"qdrant-data-*.tar.gz >/dev/null 2>&1 || bad "qdrant backup file not written under backups/"
[[ "$out" == *"searxng"*"not running"* ]] || bad "upgrade must report the stopped service as not running"

# ── 2. --no-backup skips the backup but still upgrades ──────────────────────────
export STUB_LOG="$T/log2"; : > "$STUB_LOG"; rm -rf "$T/club/backups"
PATH="$T/bin:$PATH" bash "$ROOT/scripts/gpu-mode.sh" upgrade --no-backup >/dev/null 2>&1
command grep -q 'docker run' "$STUB_LOG" && bad "--no-backup must not run a backup"
command grep -qE '^\[qdrant\] docker compose .* up -d' "$STUB_LOG" || bad "--no-backup must still recreate qdrant"

# ── 3. nothing drifted → nothing recreated ──────────────────────────────────────
ctr qdrant qdrant/qdrant:v1.19.1 true qdrant_qdrant-data
export STUB_LOG="$T/log3"; : > "$STUB_LOG"
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/gpu-mode.sh" upgrade 2>&1)"
command grep -qE 'docker compose .* up -d|docker stop|docker run' "$STUB_LOG" && bad "no drift must mean no stop/backup/recreate"
[[ "$out" == *"Nothing to upgrade."* ]] || bad "no drift must say 'Nothing to upgrade.'"

# ── 4. status shows the drift and points at upgrade ─────────────────────────────
ctr qdrant qdrant/qdrant:v1.18.0 true qdrant_qdrant-data
export STUB_LOG="$T/log4"; : > "$STUB_LOG"
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/gpu-mode.sh" status 2>&1)"
[[ "$out" == *"Service Images"* ]] || bad "status must print the Service Images section"
[[ "$out" == *"qdrant"*"running qdrant/qdrant:v1.18.0 → pinned qdrant/qdrant:v1.19.1"* ]] || bad "status must show qdrant's drift"
[[ "$out" == *"gpu-mode upgrade"* ]] || bad "status must point at 'gpu-mode upgrade' when something drifted"
command grep -qE 'docker compose .* up -d' "$STUB_LOG" && bad "status must never recreate anything"

# ── 5. service-images: the contract update.sh relies on ─────────────────────────
# update.sh prints the section only when it contains the upgrade hint, so the hint
# must appear with drift and must NOT appear without it.
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/gpu-mode.sh" service-images 2>&1)"
[[ "$out" == *"gpu-mode upgrade"* ]] || bad "service-images must carry the upgrade hint when a service drifted"
ctr qdrant qdrant/qdrant:v1.19.1 true qdrant_qdrant-data
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/gpu-mode.sh" service-images 2>&1)"
[[ "$out" == *"gpu-mode upgrade"* ]] && bad "service-images must NOT print the upgrade hint when nothing drifted (update.sh would nag)"
command grep -q 'update.sh' "$ROOT/scripts/update.sh" && command grep -q 'gpu-mode.sh" service-images' "$ROOT/scripts/update.sh" \
  || bad "update.sh must call 'gpu-mode.sh service-images' to report drift after a pull"

# ── 6. status warns while gateway request logging is on (scripts/litellm-log.sh) ─
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/gpu-mode.sh" status 2>&1)"
[[ "$out" == *"request logging is ON"* ]] && bad "status must NOT warn about request logging when it is off"
ctr litellm ghcr.io/berriai/litellm:v1.100.3 true "" "LITELLM_LOG=DEBUG"
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/gpu-mode.sh" status 2>&1)"
[[ "$out" == *"LiteLLM request logging is ON (LITELLM_LOG=DEBUG)"*"litellm-log.sh off"* ]] || bad "status must warn while gateway request logging is on"
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/gpu-mode.sh" service-images 2>&1)"
[[ "$out" == *"gpu-mode upgrade"* ]] && bad "the logging warning must not trip update.sh's upgrade-hint contract"

[ "$fail" -eq 0 ] && echo "test-gpu-mode-service-upgrade: ok (drift detection, upgrade scope, backup-before-recreate, --no-backup, status, update.sh contract, request-logging warning)"
exit "$fail"
