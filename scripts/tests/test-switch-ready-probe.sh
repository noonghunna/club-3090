#!/usr/bin/env bash
# Test for switch.sh wait_ready() crash detection + generation readiness probe.
#
# Two bugs this locks down:
#
#   #1099 — crash detection never fired under a restart policy. Docker reports
#           `.State.Running == true` for a container in restart backoff, so on
#           any compose with `restart: unless-stopped` (most of them) a dead
#           server was polled for the full READY_TIMEOUT while the display said
#           "still waiting". The check now keys on `.State.Status` (which does
#           say `restarting`) AND on RestartCount rising during the wait.
#           Secondary: llama.cpp / ik-llama emit none of the vLLM progress
#           strings, so those slugs showed a bare elapsed counter.
#
#   #1100 — "ready" was declared on a bound-but-cold server: `GET /v1/models`
#           answers the moment the HTTP server binds. One bounded max_tokens=1
#           completion now has to succeed first (READY_PROBE=0 opts out).
#
# Validates:
#   1.  Crash-loop under a restart policy (Status=restarting) fails fast.
#   2.  NEGATIVE CONTROL: a slow-but-healthy boot (Status=running, RestartCount
#       flat, endpoint binds late) is NOT flagged — a gate that fires on healthy
#       boots is worse than the bug.
#   3.  RestartCount rising while Status reads `running` (the between-restarts
#       sampling window) fails fast.
#   4.  A plain exited container still fails fast (old behaviour preserved).
#   5.  The last 30 log lines are dumped on every crash path.
#   6.  llama.cpp progress markers are surfaced.
#   7.  Generation probe: HTTP 5xx → boot FAILS (bound but cannot generate).
#   8.  Generation probe: transport failure / timeout → boot FAILS.
#   9.  Generation probe: HTTP 2xx with choices[] → ready.
#   10. Generation probe: HTTP 4xx → WARN only (graceful degrade, not a failure).
#   11. The probe uses the served model id resolved from /v1/models (no hardcode).
#   12. READY_PROBE=0 skips the probe entirely (no POST issued).
#
# Harness: extract ready_probe()/wait_ready() via sed, source them, and put mock
# `docker`, `curl` and `sleep` on PATH. Fully offline — no docker, no GPU.
set -uo pipefail

# wait_ready() parses the served-model id with python3 (via the sourced helper),
# so carry the repo's UTF-8-mode default here too (#779; test-locale-utf8.sh).
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0
FAIL=0
_dump() { echo "--- wait_ready output (rc=$2) ---" >&2; printf '%s\n' "$1" >&2; }
assert_contains() {
  local haystack="$1" needle="$2" label="$3" rc="${4:-}"
  if [[ "$haystack" == *"$needle"* ]]; then PASS=$((PASS + 1)); else
    echo "FAIL: ${label}: expected output to CONTAIN: ${needle}" >&2
    _dump "$haystack" "$rc"; FAIL=$((FAIL + 1))
  fi
}
assert_not_contains() {
  local haystack="$1" needle="$2" label="$3" rc="${4:-}"
  if [[ "$haystack" != *"$needle"* ]]; then PASS=$((PASS + 1)); else
    echo "FAIL: ${label}: expected output to NOT contain: ${needle}" >&2
    _dump "$haystack" "$rc"; FAIL=$((FAIL + 1))
  fi
}
assert_rc() {
  local got="$1" want="$2" label="$3" out="${4:-}"
  if [[ "$got" == "$want" ]]; then PASS=$((PASS + 1)); else
    echo "FAIL: ${label}: expected exit ${want}, got ${got}" >&2
    _dump "$out" "$got"; FAIL=$((FAIL + 1))
  fi
}

# --- Extract the functions under test (avoids running switch.sh's main) -------
HELPERS_FILE="$(mktemp --suffix=.sh)"
sed -n '/^ready_probe()/,/^}/p' scripts/switch.sh  > "$HELPERS_FILE"
sed -n '/^wait_ready()/,/^}/p'  scripts/switch.sh >> "$HELPERS_FILE"
if ! command grep -q '^wait_ready()' "$HELPERS_FILE" || ! command grep -q '^ready_probe()' "$HELPERS_FILE"; then
  echo "FAIL: could not extract ready_probe()/wait_ready() from scripts/switch.sh" >&2
  rm -f "$HELPERS_FILE"; exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() { rm -rf "$tmp_dir" "$HELPERS_FILE"; }
trap cleanup EXIT

# --- Mock `docker` ------------------------------------------------------------
# State is env-driven: DOCKER_MOCK_STATUS (.State.Status), a RestartCount
# SEQUENCE (successive polls, sticky at the last element), exit code and logs.
cat > "${tmp_dir}/docker" <<'EOF'
#!/usr/bin/env bash
sub="${1:-}"; shift || true
case "$sub" in
  ps)
    for n in ${DOCKER_MOCK_RUNNING:-}; do printf '%s\n' "$n"; done
    ;;
  inspect)
    fmt=""
    while [[ $# -gt 0 ]]; do
      case "$1" in -f|--format) fmt="${2:-}"; shift 2 || true ;; *) shift ;; esac
    done
    case "$fmt" in
      *State.Status*)   printf '%s\n' "${DOCKER_MOCK_STATUS:-running}" ;;
      *State.Running*)  printf '%s\n' "${DOCKER_MOCK_RUNNING_FLAG:-true}" ;;
      *State.ExitCode*) printf '%s\n' "${DOCKER_MOCK_EXIT:-0}" ;;
      *RestartCount*)
        read -r -a _seq <<< "${DOCKER_MOCK_RESTARTS:-0}"
        _i="$(cat "${MOCK_STATE}/rc_idx" 2>/dev/null || echo 0)"
        [[ "$_i" -lt "${#_seq[@]}" ]] || _i=$(( ${#_seq[@]} - 1 ))
        printf '%s\n' "${_seq[$_i]}"
        echo $(( _i + 1 )) > "${MOCK_STATE}/rc_idx"
        ;;
    esac
    ;;
  logs) printf '%s\n' "${DOCKER_MOCK_LOGS:-}" ;;
  *) : ;;
esac
exit 0
EOF
chmod +x "${tmp_dir}/docker"

# --- Mock `curl` --------------------------------------------------------------
# Three call shapes, all issued by wait_ready/ready_probe:
#   readiness poll   : -sf -o /dev/null --max-time 3 <models-url>
#   served-id fetch  : -sf --max-time 3 <models-url>          → JSON on stdout
#   generation probe : -s -o <file> -w '%{http_code}' -X POST <chat-url>
cat > "${tmp_dir}/curl" <<'EOF'
#!/usr/bin/env bash
is_post=0; out=""; payload=""; url=""
args=("$@"); i=0
while [[ $i -lt ${#args[@]} ]]; do
  case "${args[$i]}" in
    -X) [[ "${args[$((i+1))]:-}" == "POST" ]] && is_post=1; i=$((i+2)) ;;
    -o) out="${args[$((i+1))]:-}"; i=$((i+2)) ;;
    -d) payload="${args[$((i+1))]:-}"; i=$((i+2)) ;;
    -H|-w|--max-time) i=$((i+2)) ;;
    http*) url="${args[$i]}"; i=$((i+1)) ;;
    *) i=$((i+1)) ;;
  esac
done

if [[ $is_post -eq 1 ]]; then
  printf '%s' "$payload" > "${MOCK_STATE}/post_payload"
  printf '%s' "$url"     > "${MOCK_STATE}/post_url"
  [[ -n "$out" ]] && printf '%s' "${CURL_MOCK_PROBE_BODY:-}" > "$out"
  printf '%s' "${CURL_MOCK_PROBE_CODE:-200}"
  exit "${CURL_MOCK_PROBE_RC:-0}"
fi

if [[ "$out" == "/dev/null" ]]; then          # readiness poll
  n="$(cat "${MOCK_STATE}/polls" 2>/dev/null || echo 0)"
  n=$((n + 1)); echo "$n" > "${MOCK_STATE}/polls"
  [[ "$n" -ge "${CURL_MOCK_READY_AFTER:-1}" ]] && exit 0
  exit 22
fi
printf '{"data":[{"id":"%s"}]}' "${CURL_MOCK_SERVED:-club-test-model}"
exit 0
EOF
chmod +x "${tmp_dir}/curl"

# --- Mock `sleep`: keep the suite instant (wait_ready polls every 4s) ---------
printf '#!/usr/bin/env bash\nexit 0\n' > "${tmp_dir}/sleep"
chmod +x "${tmp_dir}/sleep"

export PATH="${tmp_dir}:$PATH"

# shellcheck source=/dev/null
source "$HELPERS_FILE"

VARIANT="ik-llama/iq4ks-mtp"
declare -A VARIANT_CONTAINER=(["ik-llama/iq4ks-mtp"]="ik-llama-qwen36-27b")
READY_URL="http://localhost:8020/v1/models"
READY_TIMEOUT=60
export DOCKER_MOCK_RUNNING="ik-llama-qwen36-27b"

run_wait_ready() {
  # Fresh mock state per scenario; run under switch.sh's own shell options.
  export MOCK_STATE="${tmp_dir}/state"
  rm -rf "$MOCK_STATE"; mkdir -p "$MOCK_STATE"
  OUT="$( set -euo pipefail; wait_ready 2>&1 )"; RC=$?
  return 0
}

# --- 1. Crash-loop under a restart policy (the #1099 bug) ---------------------
# Docker says Running=true the whole time — only Status/RestartCount betray it.
export DOCKER_MOCK_STATUS="restarting" DOCKER_MOCK_RUNNING_FLAG="true"
export DOCKER_MOCK_RESTARTS="29" DOCKER_MOCK_EXIT="1"
export DOCKER_MOCK_LOGS="ERROR boot-guard: rejected --ctx-size 262144 (needs 30 GiB)"
export CURL_MOCK_READY_AFTER=9999
run_wait_ready
assert_rc "$RC" 1 "crash-loop: fails fast instead of waiting out READY_TIMEOUT" "$OUT"
assert_contains "$OUT" "is not coming up"         "crash-loop: reports the container is not coming up" "$RC"
assert_contains "$OUT" "state=restarting"         "crash-loop: names the restarting state" "$RC"
assert_contains "$OUT" "boot-guard: rejected"     "crash-loop: dumps the log tail with the guard error" "$RC"
assert_not_contains "$OUT" "timeout — server not ready" "crash-loop: does not burn the full timeout" "$RC"

# --- 2. NEGATIVE CONTROL: slow-but-healthy boot ------------------------------
# Status stays `running`, RestartCount never moves, the endpoint binds on the
# 6th poll. Must reach ready — no crash verdict, no false positive.
export DOCKER_MOCK_STATUS="running" DOCKER_MOCK_RESTARTS="0" DOCKER_MOCK_EXIT="0"
export DOCKER_MOCK_LOGS="load_model: loading model from /models/qwen.gguf"
export CURL_MOCK_READY_AFTER=6
export CURL_MOCK_PROBE_CODE=200 CURL_MOCK_PROBE_RC=0
export CURL_MOCK_PROBE_BODY='{"choices":[{"message":{"content":"hi"}}]}'
run_wait_ready
assert_rc "$RC" 0 "negative control: slow-but-healthy boot is NOT flagged" "$OUT"
assert_contains "$OUT" "✓ ready"          "negative control: declares ready" "$RC"
assert_not_contains "$OUT" "ERROR"        "negative control: no crash verdict on a healthy boot" "$RC"
assert_not_contains "$OUT" "crash-looping" "negative control: no crash-loop verdict" "$RC"
# 6. llama.cpp progress marker surfaced (used to be vLLM-only strings)
assert_contains "$OUT" "load_model: loading model" "llama.cpp progress marker is surfaced" "$RC"
# 11. the probe targets the served id resolved from /v1/models, not a literal
assert_contains "$(cat "${MOCK_STATE}/post_payload" 2>/dev/null)" '"model":"club-test-model"' \
  "probe uses the served model id from /v1/models" "$RC"
assert_contains "$(cat "${MOCK_STATE}/post_url" 2>/dev/null)" "/v1/chat/completions" \
  "probe posts to the chat-completions endpoint" "$RC"
assert_contains "$OUT" "generation probe ok"  "probe reports success on a healthy engine" "$RC"

# --- 3. RestartCount rising while Status reads `running` ----------------------
# A crash-loop reads `running` in the window between two restarts; the counter
# is what makes it visible at an arbitrary sample point.
export DOCKER_MOCK_STATUS="running" DOCKER_MOCK_RESTARTS="0 0 3" DOCKER_MOCK_EXIT="1"
export DOCKER_MOCK_LOGS="CUDA error: out of memory"
export CURL_MOCK_READY_AFTER=9999
run_wait_ready
assert_rc "$RC" 1 "restart-count leg: rising RestartCount fails fast" "$OUT"
assert_contains "$OUT" "crash-looping"        "restart-count leg: names the crash loop" "$RC"
assert_contains "$OUT" "CUDA error"           "restart-count leg: dumps the log tail" "$RC"

# --- 4. Plain exited container (pre-existing behaviour preserved) -------------
export DOCKER_MOCK_STATUS="exited" DOCKER_MOCK_RESTARTS="0" DOCKER_MOCK_EXIT="137"
export DOCKER_MOCK_LOGS="killed"
run_wait_ready
assert_rc "$RC" 1 "exited container still fails fast" "$OUT"
assert_contains "$OUT" "state=exited"  "exited container: names the state" "$RC"
assert_contains "$OUT" "exit=137"      "exited container: surfaces the exit code" "$RC"

# --- 7. Generation probe: HTTP 5xx → the boot FAILS (#1100) -------------------
export DOCKER_MOCK_STATUS="running" DOCKER_MOCK_RESTARTS="0" DOCKER_MOCK_EXIT="0"
export DOCKER_MOCK_LOGS="AssertionError in sampler"
export CURL_MOCK_READY_AFTER=1
export CURL_MOCK_PROBE_CODE=500 CURL_MOCK_PROBE_RC=0
export CURL_MOCK_PROBE_BODY='{"error":"engine dead"}'
run_wait_ready
assert_rc "$RC" 1 "probe: bound but cannot generate (HTTP 500) fails the boot" "$OUT"
assert_contains "$OUT" "FAILED to generate"   "probe: says generation failed" "$RC"
assert_contains "$OUT" "AssertionError"       "probe: dumps the log tail" "$RC"
assert_not_contains "$OUT" "✓ ready"          "probe: does not declare ready on a broken engine" "$RC"

# --- 8. Generation probe: transport failure / timeout → FAILS ----------------
export CURL_MOCK_PROBE_CODE="000" CURL_MOCK_PROBE_RC=28 CURL_MOCK_PROBE_BODY=""
run_wait_ready
assert_rc "$RC" 1 "probe: transport failure / timeout fails the boot" "$OUT"
assert_contains "$OUT" "could not generate"     "probe: reports the transport failure" "$RC"
assert_contains "$OUT" "READY_PROBE=0"          "probe: names the opt-out in the error" "$RC"

# --- 10. Generation probe: HTTP 4xx → WARN only ------------------------------
# A different completion shape / missing chat template must NOT turn a working
# boot into a false failure.
export CURL_MOCK_PROBE_CODE=404 CURL_MOCK_PROBE_RC=0
export CURL_MOCK_PROBE_BODY='{"detail":"Not Found"}'
run_wait_ready
assert_rc "$RC" 0 "probe: HTTP 4xx degrades gracefully (not a boot failure)" "$OUT"
assert_contains "$OUT" "generation probe skipped" "probe: warns on 4xx" "$RC"
assert_contains "$OUT" "✓ ready"                  "probe: still declares ready on 4xx" "$RC"

# --- 12. READY_PROBE=0 skips the probe entirely ------------------------------
export CURL_MOCK_PROBE_CODE=500 CURL_MOCK_PROBE_RC=0
READY_PROBE=0 run_wait_ready
assert_rc "$RC" 0 "READY_PROBE=0: opt-out keeps the old behaviour" "$OUT"
assert_contains "$OUT" "probe skipped (READY_PROBE=0)" "READY_PROBE=0: says it skipped" "$RC"
if [[ -f "${MOCK_STATE}/post_payload" ]]; then
  echo "FAIL: READY_PROBE=0 still issued a completion request" >&2
  FAIL=$((FAIL + 1))
else
  PASS=$((PASS + 1))
fi

# --- Summary ------------------------------------------------------------------
echo "----------------------------------------"
echo "PASS: $PASS   FAIL: $FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
echo "OK: switch.sh crash detection (#1099) + generation readiness probe (#1100)"
