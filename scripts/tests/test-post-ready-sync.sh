#!/usr/bin/env bash
# Test for #1522: the gateway (and Open WebUI) get synced when a server becomes
# ready even though switch.sh did not wait for it.
#
# switch.sh syncs the gateway after teardown (which renders the old route away)
# and again after a WAITED ready. `--no-wait`, or a boot slower than
# READY_TIMEOUT, skipped the second sync and left the gateway with no local
# routes for a model that then came up fine.
#
# Validates:
#   1. post-ready-sync.sh runs owui-register.sh <port> and litellm-sync.sh
#      --quiet once the ready URL answers (with --owui), and only the gateway
#      sync without --owui.
#   2. It does not sync when the container is gone or stopped (exit 2), or when
#      nothing answers before --timeout (exit 3). Missing arguments: exit 64.
#   3. switch.sh's schedule_post_ready_sync() starts the waiter with the ready
#      URL, container, port and --owui, and with no container prints the manual
#      command instead of starting anything.
#   4. switch.sh's REAL wait_ready() timeout branch, on a container that is still
#      running, schedules the waiter and still exits 1.
#   5. The --no-wait path calls schedule_post_ready_sync.
#
# Harness: stub curl / docker on PATH, stub sync scripts beside a copy of the
# waiter, and a stub waiter under a fake ROOT_DIR for the switch.sh functions.
# Nothing is launched and nothing is left running.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0
FAIL=0
check() {   # check <label> <command…>
  local label="$1"; shift
  if "$@"; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); echo "  FAIL: ${label}"; fi
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# --- stubs -------------------------------------------------------------------
mkdir -p "$tmp/bin" "$tmp/lib"
cat > "$tmp/bin/curl" <<'EOF'
#!/usr/bin/env bash
# Answers once it has been called CURL_OK_AFTER times (never when unset).
n=$(( $(cat "$STUB_DIR/curl.count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$STUB_DIR/curl.count"
[[ -n "${CURL_OK_AFTER:-}" && "$n" -ge "$CURL_OK_AFTER" ]]
EOF
cat > "$tmp/bin/docker" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  "ps --format {{.Names}}") echo "${STUB_CONTAINER:-}" ;;
  *"{{.State.Status}}"*)    echo "${DOCKER_STATE-running}" ;;   # set-but-empty = a container that is gone
  *"{{.RestartCount}}"*)    echo 0 ;;
  *"{{.State.ExitCode}}"*)  echo 0 ;;
  logs*)                    : ;;
esac
EOF
cp scripts/lib/post-ready-sync.sh "$tmp/lib/"
for s in owui-register litellm-sync; do
  printf '#!/usr/bin/env bash\necho "%s $*" >> "$STUB_DIR/calls"\n' "$s" > "$tmp/lib/$s.sh"
done
chmod +x "$tmp/bin/"* "$tmp/lib/"*
export STUB_DIR="$tmp"

# A stub mistake must fail the test, not hang it: a short default ceiling, and a
# hard kill above that (a waiter left polling on the default 3600 s did happen).
export POST_READY_SYNC_TIMEOUT=10
run_waiter() {   # run_waiter <args…>; sets RC and CALLS
  rm -f "$tmp/calls" "$tmp/curl.count"
  PATH="$tmp/bin:$PATH" timeout 20 bash "$tmp/lib/post-ready-sync.sh" "$@" > "$tmp/waiter.out" 2>&1
  RC=$?
  CALLS="$(cat "$tmp/calls" 2>/dev/null || true)"
}

# --- 1. syncs once the URL answers ----------------------------------------------
CURL_OK_AFTER=3 run_waiter --url http://localhost:8122/v1/models --container c1 --port 8122 --owui --interval 0
check "1a: exit 0 once ready (got $RC)" test "$RC" -eq 0
check "1b: OWUI registered on the port" test "$(sed -n 1p <<<"$CALLS")" = "owui-register 8122"
check "1c: gateway synced, quietly" test "$(sed -n 2p <<<"$CALLS")" = "litellm-sync --quiet"
check "1d: waited for the 3rd poll" test "$(cat "$tmp/curl.count")" -eq 3

CURL_OK_AFTER=1 run_waiter --url http://localhost:8122/v1/models --container c1 --port 8122 --interval 0
check "1e: without --owui only the gateway is synced" test "$CALLS" = "litellm-sync --quiet"

# --- 2. no sync when it never comes up ------------------------------------------
DOCKER_STATE=exited run_waiter --url http://localhost:8122/v1/models --container c1 --port 8122 --owui --interval 0
check "2a: stopped container -> exit 2 (got $RC)" test "$RC" -eq 2
check "2b: stopped container -> no sync" test -z "$CALLS"

DOCKER_STATE="" run_waiter --url http://localhost:8122/v1/models --container gone --port 8122 --interval 0
check "2c: missing container -> exit 2 (got $RC)" test "$RC" -eq 2

run_waiter --url http://localhost:8122/v1/models --container c1 --port 8122 --timeout 0 --interval 0
check "2d: never answers -> exit 3 at the ceiling (got $RC)" test "$RC" -eq 3
check "2e: never answers -> no sync" test -z "$CALLS"

run_waiter --url http://localhost:8122/v1/models --port 8122
check "2f: missing --container -> exit 64 (got $RC)" test "$RC" -eq 64

# --- 3/4. switch.sh's own functions, against a stub waiter ----------------------
fake_root="$tmp/root"
mkdir -p "$fake_root/scripts/lib" "$tmp/data"
cat > "$fake_root/scripts/lib/post-ready-sync.sh" <<'EOF'
#!/usr/bin/env bash
echo "$*" > "$STUB_DIR/scheduled"
EOF
fns="$tmp/switch-fns.sh"
awk '/^schedule_post_ready_sync\(\) \{/,/^}/' scripts/switch.sh > "$fns"
awk '/^wait_ready\(\) \{/,/^}/' scripts/switch.sh >> "$fns"
check "3: both functions extracted from switch.sh" test "$(command grep -c '^}' "$fns")" -eq 2

wait_for_file() { local i; for i in $(seq 1 50); do [[ -s "$1" ]] && return 0; sleep 0.1; done; return 1; }

out="$(
  ROOT_DIR="$fake_root" READY_URL="http://localhost:8122/v1/models" OWUI_REGISTER=1
  club_config_data_dir() { echo "$tmp/data"; }
  source "$fns"
  schedule_post_ready_sync c1 2>&1
)"
wait_for_file "$tmp/scheduled"
sched="$(cat "$tmp/scheduled" 2>/dev/null || true)"
check "3a: waiter started with the ready URL, container, port and --owui" \
  test "$sched" = "--url http://localhost:8122/v1/models --container c1 --port 8122 --owui"
check "3b: says so, naming the port" command grep -q "synced in the background once :8122 answers" <<<"$out"
check "3c: log under the data dir" test -e "$tmp/data/logs/post-ready-sync-c1.log"

rm -f "$tmp/scheduled"
out="$(
  ROOT_DIR="$fake_root" READY_URL="http://localhost:8122/v1/models" OWUI_REGISTER=1
  club_config_data_dir() { echo "$tmp/data"; }
  source "$fns"
  schedule_post_ready_sync "" 2>&1
)"
sleep 0.3
check "3d: no container -> nothing started" test ! -e "$tmp/scheduled"
check "3e: no container -> prints the manual sync" command grep -q "bash scripts/lib/litellm-sync.sh" <<<"$out"

# The real wait_ready(): nothing answers, the container keeps running, and the
# timeout fires. sleep is a no-op so the 4 s poll step costs nothing.
rm -f "$tmp/scheduled" "$tmp/curl.count"
out="$(
  export PATH="$tmp/bin:$PATH" STUB_CONTAINER=c1
  ROOT_DIR="$fake_root" READY_URL="http://localhost:8122/v1/models" OWUI_REGISTER=0 READY_TIMEOUT=1 VARIANT=v1
  declare -A VARIANT_CONTAINER=([v1]=c1)
  club_config_data_dir() { echo "$tmp/data"; }
  sleep() { :; }
  source "$fns"
  ( wait_ready ) 2>&1   # its exit 1 must end only this inner subshell
  echo "rc=$?"
)"
wait_for_file "$tmp/scheduled"
check "4a: timeout still exits 1" command grep -q "^rc=1$" <<<"$out"
check "4b: the timeout line is unchanged" command grep -q "timeout — server not ready after 1s" <<<"$out"
check "4c: says the container is still booting" command grep -q "still booting" <<<"$out"
check "4d: waiter scheduled from the timeout branch (no --owui when off)" \
  test "$(cat "$tmp/scheduled" 2>/dev/null)" = "--url http://localhost:8122/v1/models --container c1 --port 8122"

# --- 5. the --no-wait path ---------------------------------------------------------
check "5: --no-wait schedules the waiter after the launch" \
  command grep -qE '^\[\[ \$WAIT -eq 0 \]\] && schedule_post_ready_sync "\$\{VARIANT_CONTAINER\[\$VARIANT\]:-\}"' scripts/switch.sh

echo "test-post-ready-sync: PASS ${PASS}, FAIL ${FAIL}"
[[ "$FAIL" -eq 0 ]]
