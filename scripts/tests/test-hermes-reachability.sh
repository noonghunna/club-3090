#!/usr/bin/env bash
# test-hermes-reachability — guards quality-test.sh's hermes container-reachability
# preflight (#960) and the diagnosis it prints when a container can't reach the
# endpoint (#1578), plus scripts/lib/listen-scope.sh which decides that diagnosis.
#
# hermesagent-20's sandbox reaches a loopback URL through host.docker.internal (the
# Docker bridge gateway). The preflight used to say "appears bound to loopback only"
# for EVERY failed probe — wrong for a server on 0.0.0.0 behind a host firewall, or
# one bound to a LAN address — and its bypass hint named a flag that doesn't exist
# (--no-sandbox) and an "unset" the wrapper immediately undoes.
#
#   A. listen_scope classifies listener sets (ss stubbed).
#   B. live leg: a real server on 127.0.0.1 reads "loopback", on 0.0.0.0 "wildcard"
#      (the real `ss`, skipped if absent).
#   C. the REAL wrapper, driven to the preflight with docker/ss/curl/benchlocal-cli
#      stubbed: the per-cause message, the exit code, a reachable endpoint passing,
#      the no-probe-image fallback, and both bypass hints actually bypassing.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

# Force Python UTF-8 mode (PEP 540) before the first python3 call (#779).
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
LIB="scripts/lib/listen-scope.sh"
WRAPPER="scripts/quality-test.sh"
fail() { echo "FAIL: $1" >&2; exit 1; }

tmp_bin="$(mktemp -d)"
tmp_work="$(mktemp -d)"
PIDS=()
before_list="$(mktemp)"; after_list="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before_list" || true
cleanup() {
  for p in "${PIDS[@]+"${PIDS[@]}"}"; do kill "$p" 2>/dev/null || true; done
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$after_list" || true
  comm -13 "$before_list" "$after_list" | xargs -r rm -f
  rm -rf "$tmp_bin" "$tmp_work"; rm -f "$before_list" "$after_list"
}
trap cleanup EXIT

bash -n "$LIB" || fail "bash -n: listen-scope.sh"
bash -n "$WRAPPER" || fail "bash -n: quality-test.sh"
echo "  ✓ syntax"

# ---- stubs ---------------------------------------------------------------------
# ss: prints the listener lines in $STUB_SS_FILE (what `ss -ltnH "sport = :P"` would).
cat > "${tmp_bin}/ss" <<'SS'
#!/usr/bin/env bash
[[ -f "${STUB_SS_FILE:-}" ]] && cat "$STUB_SS_FILE"
exit 0
SS
# docker: sandbox images present; the probe image present unless STUB_NO_PROBE_IMAGE=1;
# the probe `docker run` exits $STUB_PROBE_RC (and is recorded); bridge gateway 172.17.0.1.
cat > "${tmp_bin}/docker" <<'DOCKER'
#!/usr/bin/env bash
case "${1:-} ${2:-}" in
  "image inspect")
    case "${3:-}" in
      benchlocal-sandbox-*:latest)
        [[ "${4:-}" == "--format" ]] && date -u -d '+1 day' '+%Y-%m-%dT%H:%M:%S.000000000Z'
        exit 0 ;;
      curlimages/curl:latest) [[ "${STUB_NO_PROBE_IMAGE:-0}" == "1" ]] && exit 1; exit 0 ;;
    esac
    exit 1 ;;
  "network inspect") echo "172.17.0.1"; exit 0 ;;
esac
if [[ "${1:-}" == "run" ]]; then echo "run $*" >> "${STUB_DOCKER_LOG}"; exit "${STUB_PROBE_RC:-1}"; fi
exit 1
DOCKER
cat > "${tmp_bin}/curl" <<'CURL'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in */v1/models) printf '{"data":[{"id":"mock-model"}]}'; exit 0 ;; */props|*/get_model_info) exit 1 ;; esac; done
exit 0
CURL
cat > "${tmp_bin}/benchlocal-cli" <<'BL'
#!/usr/bin/env bash
json_out=""
for ((i=1; i<=$#; i++)); do
  case "${!i}" in list) echo hermesagent-20; exit 0 ;; --help) echo '--reasoning-effort'; exit 0 ;; esac
  if [[ "${!i}" == "--save-json" ]]; then j=$((i+1)); json_out="${!j}"; fi
done
echo "RAN $*" >> "${BENCHLOCAL_MOCK_LOG}"
if [[ -n "$json_out" ]]; then
  mkdir -p "$(dirname "$json_out")"
  echo '{"packs":[{"pack_id":"hermesagent-20","status":"ok","passed":5,"total":20,"score":0.25}]}' > "$json_out"
fi
echo "TOTAL (mock)"
exit 0
BL
chmod +x "${tmp_bin}"/{ss,docker,curl,benchlocal-cli}

# ---- A. listen_scope ------------------------------------------------------------
# shellcheck source=../lib/listen-scope.sh
source "$LIB"
scope_of() {  # listener lines...
  printf '%s\n' "$@" > "${tmp_work}/ss.lines"
  STUB_SS_FILE="${tmp_work}/ss.lines" PATH="${tmp_bin}:$PATH" listen_scope 8000 172.17.0.1
}
L() { echo "LISTEN 0 4096 $1 0.0.0.0:*"; }
check() { local got; got="$(scope_of "${@:3}")"; [[ "$got" == "$2" ]] || fail "listen_scope $1: expected $2, got $got"; echo "  ✓ A: $1 → $2"; }
check "127.0.0.1 only"                  loopback "$(L 127.0.0.1:8000)"
check "[::1] only"                      loopback "$(L '[::1]:8000')"
check "127.0.0.1 + [::1]"               loopback "$(L 127.0.0.1:8000)" "$(L '[::1]:8000')"
check "0.0.0.0"                         wildcard "$(L 0.0.0.0:8000)"
check "[::]"                            wildcard "$(L '[::]:8000')"
check "*"                               wildcard "$(L '*:8000')"
check "loopback + 0.0.0.0"              wildcard "$(L 127.0.0.1:8000)" "$(L 0.0.0.0:8000)"
check "the bridge address"              bridge   "$(L 172.17.0.1:8000)"
check "a LAN address only"              specific "$(L 192.168.1.50:8000)"
check "loopback + a LAN address"        specific "$(L 127.0.0.1:8000)" "$(L 192.168.1.50:8000)"
check "nothing listening"               none
got="$(PATH="${tmp_work}/nopath" listen_scope 8000 172.17.0.1)"
[[ "$got" == "unknown" ]] || fail "listen_scope without ss: expected unknown, got $got"
echo "  ✓ A: no ss on PATH → unknown"
printf '%s\n' "$(L '127.0.0.53%lo:8000')" > "${tmp_work}/ss.lines"
got="$(STUB_SS_FILE="${tmp_work}/ss.lines" PATH="${tmp_bin}:$PATH" listen_addrs 8000)"
[[ "$got" == "127.0.0.53" ]] || fail "listen_addrs must strip the %iface suffix: got '$got'"
echo "  ✓ A: listen_addrs strips :port and %iface"

# ---- B. live leg: the real ss ----------------------------------------------------
if command -v ss >/dev/null 2>&1; then
  for bind in 127.0.0.1 0.0.0.0; do
    python3 -c '
import http.server, sys
s = http.server.HTTPServer((sys.argv[1], 0), http.server.BaseHTTPRequestHandler)
print(s.server_address[1], flush=True); s.serve_forever()' "$bind" > "${tmp_work}/port.$bind" 2>/dev/null </dev/null &
    PIDS+=("$!")
    port=""; for _ in $(seq 1 50); do port="$(head -1 "${tmp_work}/port.$bind" 2>/dev/null || true)"; [[ -n "$port" ]] && break; sleep 0.1; done
    [[ -n "$port" ]] || fail "live leg: server on $bind did not start"
    want=loopback; [[ "$bind" == "0.0.0.0" ]] && want=wildcard
    got="$(listen_scope "$port" 172.17.0.1)"
    [[ "$got" == "$want" ]] || fail "live leg: a real server on $bind:$port read as '$got', expected '$want'"
    echo "  ✓ B: real ss — server bound $bind → $got"
  done
else
  echo "  - B: skipped (no ss on this host)"
fi

# ---- C. the real wrapper, driven to the preflight ---------------------------------
PORT=18080
run_wrapper() {  # extra env assignments...; sets OUT, RC
  : > "${tmp_work}/bl.log"; : > "${tmp_work}/docker.log"
  set +e
  OUT="$(env PATH="${tmp_bin}:$PATH" STUB_SS_FILE="${tmp_work}/ss.lines" STUB_DOCKER_LOG="${tmp_work}/docker.log" \
        BENCHLOCAL_MOCK_LOG="${tmp_work}/bl.log" PREFLIGHT_NO_AUTODETECT=1 \
        URL="http://127.0.0.1:${PORT}" MODEL=mock-model "$@" \
        bash "$WRAPPER" --pack hermesagent-20 --no-thinking --no-progress 2>&1)"
  RC=$?
  set -e
}
expect_fail() {  # label, then substrings the message must carry
  local label="$1"; shift
  [[ "$RC" == "2" ]] || { echo "$OUT" | tail -20 >&2; fail "C/$label: expected exit 2, got $RC"; }
  [[ ! -s "${tmp_work}/bl.log" ]] || fail "C/$label: benchlocal-cli ran after a failed preflight"
  for n in "$@"; do [[ "$OUT" == *"$n"* ]] || { echo "$OUT" | tail -20 >&2; fail "C/$label: message lacks: $n"; }; done
  [[ "$OUT" == *"--no-sandboxed, or BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=0"* ]] || fail "C/$label: bypass hint must name the real flag and =0"
  echo "  ✓ C: $label"
}

printf '%s\n' "$(L 127.0.0.1:${PORT})" > "${tmp_work}/ss.lines"
run_wrapper STUB_PROBE_RC=1
[[ -s "${tmp_work}/docker.log" ]] || fail "C: the container probe never ran — every assertion below would be vacuous"
expect_fail "loopback-only → loopback diagnosis + BIND_HOST / -p / --host fixes" \
  "listens on loopback only (127.0.0.1)" "BIND_HOST=127.0.0.1" "not -p 127.0.0.1:${PORT}:" "--host 0.0.0.0" \
  "URL=http://172.17.0.1:${PORT}"

printf '%s\n' "$(L 0.0.0.0:${PORT})" > "${tmp_work}/ss.lines"
run_wrapper STUB_PROBE_RC=1
expect_fail "0.0.0.0 but the probe fails → firewall, not the bind" \
  "the bind is NOT the problem" "ufw allow in on docker0 to any port ${PORT}"
[[ "$OUT" != *"listens on loopback only"* ]] || fail "C: a wildcard bind must not be diagnosed as loopback"

printf '%s\n' "$(L 192.168.1.50:${PORT})" > "${tmp_work}/ss.lines"
run_wrapper STUB_PROBE_RC=1
expect_fail "LAN address only → run against that address" "URL=http://192.168.1.50:${PORT}"

printf '%s\n' "$(L 0.0.0.0:${PORT})" > "${tmp_work}/ss.lines"
run_wrapper STUB_PROBE_RC=0
[[ "$RC" != "2" && "$OUT" == *"endpoint reachable from a container"* && -s "${tmp_work}/bl.log" ]] \
  || { echo "$OUT" | tail -15 >&2; fail "C: a reachable endpoint must pass the preflight and run (rc=$RC)"; }
echo "  ✓ C: positive control — probe succeeds → passes, benchlocal runs"

printf '%s\n' "$(L 127.0.0.1:${PORT})" > "${tmp_work}/ss.lines"
run_wrapper STUB_NO_PROBE_IMAGE=1
[[ ! -s "${tmp_work}/docker.log" ]] || fail "C: no probe image cached → no docker run expected"
expect_fail "no probe image → the listener check alone catches loopback" "listens on loopback only (127.0.0.1)"
printf '%s\n' "$(L 0.0.0.0:${PORT})" > "${tmp_work}/ss.lines"
run_wrapper STUB_NO_PROBE_IMAGE=1
[[ "$RC" != "2" && "$OUT" == *"endpoint reachable from a container"* ]] || fail "C: no probe image + wildcard bind should pass (rc=$RC)"
echo "  ✓ C: no probe image + wildcard bind → passes"

# The bypass hints must be TRUE: each one has to skip the failing preflight.
printf '%s\n' "$(L 127.0.0.1:${PORT})" > "${tmp_work}/ss.lines"
run_wrapper STUB_PROBE_RC=1 BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=0
[[ "$OUT" != *"NOT reachable from a container"* && ! -s "${tmp_work}/docker.log" ]] \
  || fail "C: BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=0 must bypass the preflight"
echo "  ✓ C: bypass hint 1 works — BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=0 skips the probe"
set +e
OUT="$(env PATH="${tmp_bin}:$PATH" STUB_SS_FILE="${tmp_work}/ss.lines" STUB_DOCKER_LOG="${tmp_work}/docker.log" \
      BENCHLOCAL_MOCK_LOG="${tmp_work}/bl.log" PREFLIGHT_NO_AUTODETECT=1 STUB_PROBE_RC=1 \
      URL="http://127.0.0.1:${PORT}" MODEL=mock-model bash "$WRAPPER" --full --no-sandboxed --no-thinking --no-progress 2>&1)"
set -e
[[ "$OUT" != *"NOT reachable from a container"* ]] || fail "C: --no-sandboxed must bypass the preflight"
echo "  ✓ C: bypass hint 2 works — --no-sandboxed skips the probe"
# And the old hint was false: unsetting the variable does NOT bypass (the wrapper re-sets it).
run_wrapper STUB_PROBE_RC=1
[[ "$RC" == "2" ]] || fail "C: with the variable unset the preflight must still run (the wrapper auto-sets it)"
echo "  ✓ C: 'unset' is not a bypass (why the hint changed)"

echo "test-hermes-reachability: ok"
