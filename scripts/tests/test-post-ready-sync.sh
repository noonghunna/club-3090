#!/usr/bin/env bash
# Test for #1522: the gateway (and Open WebUI) get synced when a server becomes
# ready even though switch.sh did not wait for it.
#
# On a switch, switch.sh advertises the model being loaded at the teardown
# (a retryable 500 while it boots) and renders the gateway accurately at
# ready. When it does not wait — `--no-wait`, or a boot slower than
# READY_TIMEOUT — this waiter owns the accurate render, prunes the gateway
# when the boot fails, and leaves the render alone when the container was
# removed by a later switch or --down (that run has already rendered).
#
# Validates:
#   1. post-ready-sync.sh runs owui-register.sh <port> and litellm-sync.sh
#      --quiet once the ready URL answers (with --owui), and only the gateway
#      sync without --owui.
#   2. It renders the gateway back to what serves (a prune, no OWUI) when the
#      container is stopped/dead (exit 2) or nothing answers before
#      --timeout (exit 3), and exits 2 -> 4 WITHOUT rendering when the
#      container is gone (a later switch/--down removed it and rendered).
#   3. switch.sh's schedule_post_ready_sync() starts the waiter with the ready
#      URL, container, port and --owui, and with no container prints the manual
#      command instead of starting anything.
#   4. switch.sh's REAL wait_ready() timeout branch, on a container that is still
#      running, schedules the waiter and still exits 1.
#   5. The --no-wait path calls schedule_post_ready_sync.
#   6. switch.sh's EXIT trap renders the gateway back to what serves on ANY
#      failed exit — a crash or a boot that outlasts READY_TIMEOUT — and not on
#      success; the timeout case prunes at the timeout, not deferred to the
#      waiter's ceiling.
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

# --- 2. failure paths: prune on stopped/timeout, leave-alone on gone -----------
DOCKER_STATE=exited run_waiter --url http://localhost:8122/v1/models --container c1 --port 8122 --owui --interval 0
check "2a: stopped container -> exit 2 (got $RC)" test "$RC" -eq 2
check "2b: stopped container -> gateway pruned" test "$CALLS" = "litellm-sync --quiet"

DOCKER_STATE="" run_waiter --url http://localhost:8122/v1/models --container gone --port 8122 --interval 0
check "2c: gone container -> exit 4 (got $RC)" test "$RC" -eq 4
check "2d: gone container -> NO render (a later switch/--down owns the gateway)" test -z "$CALLS"

run_waiter --url http://localhost:8122/v1/models --container c1 --port 8122 --timeout 0 --interval 0
check "2e: never answers -> exit 3 at the ceiling (got $RC)" test "$RC" -eq 3
check "2f: never answers -> gateway pruned" test "$CALLS" = "litellm-sync --quiet"

run_waiter --url http://localhost:8122/v1/models --port 8122
check "2g: missing --container -> exit 64 (got $RC)" test "$RC" -eq 64

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

# --- 6. the EXIT trap prunes on a failed exit — crash OR timeout -----------------
# The trap is defined in the main switch path (not an exported function), so
# extract it the same way as wait_ready and drive it against a stub litellm-sync.
# The key pin: it prunes on ANY non-zero exit, including the timeout case where a
# waiter is already scheduled — there is no "defer to the waiter" exception, so a
# boot that outlasts READY_TIMEOUT ends pruned at the timeout, not 3600 s later.
trap_fn="$tmp/trap-fn.sh"
awk '/^switch_gateway_render_on_fail\(\) \{/,/^}/' scripts/switch.sh > "$trap_fn"
check "6: trap function extracted from switch.sh" test -s "$trap_fn"

fake_root2="$tmp/root2"
mkdir -p "$fake_root2/scripts/lib"
printf '#!/usr/bin/env bash\necho "litellm-sync $* hint=${LITELLM_BOOTING_PORT:-unset}" >> "$STUB_DIR/trap-calls"\n' > "$fake_root2/scripts/lib/litellm-sync.sh"
chmod +x "$fake_root2/scripts/lib/litellm-sync.sh"

trap_calls() {   # trap_calls <true|false> [VAR=value]; echoes the trap's litellm-sync calls
  local setup="$1" extra="${2:-}"
  rm -f "$STUB_DIR/trap-calls"
  (
    export ROOT_DIR="$fake_root2"
    [[ -n "$extra" ]] && export "$extra"
    source "$trap_fn"
    "$setup"
    switch_gateway_render_on_fail
  ) >/dev/null 2>&1
  cat "$STUB_DIR/trap-calls" 2>/dev/null || true
}

check "6a: failed exit (rc=1) -> gateway pruned, no hint leak" \
  test "$(trap_calls false)" = "litellm-sync --quiet hint=unset"
check "6b: successful exit (rc=0) -> no prune" \
  test -z "$(trap_calls true)"
check "6c: timeout case (waiter scheduled, rc=1) -> still pruned, not deferred, no hint leak" \
  test "$(trap_calls false POST_READY_SYNC_SCHEDULED=1)" = "litellm-sync --quiet hint=unset"
check "6d: the trap is set on EXIT in the main switch path" \
  command grep -q '^trap switch_gateway_render_on_fail EXIT' scripts/switch.sh
# 6a/6c prove the extracted trap function carries no hint, but they drive it in a
# subshell that never sees main's environment: the real leak vector is switch.sh
# EXPORTING the hint (or running under `set -a`), which the trap and the detached
# waiter would both inherit — re-advertising the model that just failed, forever.
# That is only preventable statically.
check "6e: the booting hint is command-scoped, never exported" \
  test -z "$(command grep -nE '(export|declare -x)[[:space:]].*LITELLM_BOOTING_PORT' scripts/switch.sh)"
check "6f: switch.sh does not run under allexport (every assignment would leak)" \
  test -z "$(command grep -nE '^[[:space:]]*set[[:space:]]+-[a-zA-Z]*a' scripts/switch.sh)"

echo "test-post-ready-sync: PASS ${PASS}, FAIL ${FAIL}"
[[ "$FAIL" -eq 0 ]]
