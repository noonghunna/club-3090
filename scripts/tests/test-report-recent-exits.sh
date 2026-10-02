#!/usr/bin/env bash
# test-report-recent-exits — report.sh's "Recent failed boot attempts" must list
# only failures (#1525).
#
# A community report listed `fp8-vllm-qwen38-27b-multi4-ultramax-kernels-1`,
# exit code 0, as a failed boot. It is the FA2 fp8-KV kernel check: a
# `restart: "no"` service the engine waits on, which exits 0 by design once the
# kernel files check out. An engine that was stopped exits 0 as well. Neither is
# a failed boot, and reading one as such sends a contributor looking for a
# problem that is not there.
#
# Validates, against stub docker data:
#   1. A non-zero exit is reported as before: its own subsection, the exit code,
#      and the last log lines.
#   2. An exit-0 `restart: no` container is listed under "Exited cleanly" as a
#      finished one-shot setup step, with no failure subsection and no logs.
#   3. An exit-0 engine (restart unless-stopped) is listed there as stopped.
#   4. Exits older than 24 h and containers that are not ours are left out.
#   5. With only clean exits: "No failed boots in the last 24h", then the list.
#   6. With only old exits: the ">24h old" line, and no clean-exit list.
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
has()    { command grep -qF -- "$2" <<<"$1"; }
hasnt()  { ! command grep -qF -- "$2" <<<"$1"; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fn="$tmp/fn.sh"
awk '/^report_recent_exits\(\) \{/,/^}/' scripts/report.sh > "$fn"
check "0: report_recent_exits() extracted from report.sh" test "$(command grep -c '^}' "$fn")" -eq 1

# --- stub docker: containers come from $STUB_DIR/ps (name<TAB>image<TAB>status<TAB>id),
# per-id fields from $STUB_DIR/<id>.{finished,code,policy}.
mkdir -p "$tmp/bin"
cat > "$tmp/bin/docker" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  info) exit 0 ;;
  ps)   cat "$STUB_DIR/ps" ;;
  inspect)
    id="$2"; fmt="$4"
    case "$fmt" in
      *FinishedAt*)  cat "$STUB_DIR/$id.finished" ;;
      *ExitCode*)    cat "$STUB_DIR/$id.code" ;;
      *RestartPolicy*) cat "$STUB_DIR/$id.policy" ;;
    esac ;;
  logs) echo "log line from ${*: -1}" ;;
esac
EOF
chmod +x "$tmp/bin/docker"
export STUB_DIR="$tmp"

container() {   # container <name> <id> <exit code> <restart policy> <finished, date -d phrase>
  printf '%s\timg\tExited (%s)\t%s\n' "$1" "$3" "$2" >> "$tmp/ps"
  date -u -d "$5" '+%Y-%m-%dT%H:%M:%S.000000000Z' > "$tmp/$2.finished"
  echo "$3" > "$tmp/$2.code"
  echo "$4" > "$tmp/$2.policy"
}

run_section() {
  (
    export PATH="$tmp/bin:$PATH"
    section()    { printf '\n## %s\n\n' "$1"; }
    subsection() { printf '\n### %s\n\n' "$1"; }
    details()    { printf '<details><summary>%s</summary>\n' "$1"; cat; printf '</details>\n'; }
    redact()     { cat; }
    have()       { command -v "$1" >/dev/null 2>&1; }
    club_container_re_loose() { echo '^(vllm-|llama-cpp-|[a-z0-9-]+-vllm-)'; }
    # shellcheck source=/dev/null
    source "$fn"
    report_recent_exits
  )
}

# --- mixed set --------------------------------------------------------------------
: > "$tmp/ps"
container fp8-vllm-qwen38-27b-multi4-ultramax-kernels-1 k1 0 no             "11 minutes ago"
container vllm-qwen38-27b-dual-fast                     a1 1 unless-stopped "5 minutes ago"
container vllm-qwen38-27b-dual-max                      b1 0 unless-stopped "20 minutes ago"
container vllm-qwen38-27b-old                           o1 1 unless-stopped "30 hours ago"
container postgres-1                                    p1 1 always         "2 minutes ago"
out="$(run_section)"

check "1a: failed engine gets its own subsection with the exit code" \
  has "$out" '### `vllm-qwen38-27b-dual-fast` — exited 5 min ago (code 1)'
check "1b: failed engine's logs are attached" has "$out" "log line from a1"
check "2a: kernel check is NOT a failed-boot subsection" \
  hasnt "$out" '### `fp8-vllm-qwen38-27b-multi4-ultramax-kernels-1`'
check "2b: kernel check listed as a finished one-shot step" \
  has "$out" '- `fp8-vllm-qwen38-27b-multi4-ultramax-kernels-1` — exited 11 min ago, code 0: one-shot setup step (`restart: no`), finished'
check "2c: no logs for a clean exit" hasnt "$out" "log line from k1"
check "3: stopped engine listed as stopped" \
  has "$out" '- `vllm-qwen38-27b-dual-max` — exited 20 min ago, code 0: stopped (restart policy `unless-stopped`)'
check "4a: exits older than 24 h left out" hasnt "$out" "vllm-qwen38-27b-old"
check "4b: containers that are not ours left out" hasnt "$out" "postgres-1"
check "4c: clean exits sit under their own heading" has "$out" "### Exited cleanly (code 0) — not boot failures"
check "4d: a failure present, so no 'no failed boots' line" hasnt "$out" "No failed boots in the last 24h"

# --- only clean exits ---------------------------------------------------------------
: > "$tmp/ps"
container fp8-vllm-qwen38-27b-multi4-ultramax-kernels-1 k1 0 no "11 minutes ago"
out="$(run_section)"
check "5a: only clean exits -> 'No failed boots in the last 24h'" has "$out" "_No failed boots in the last 24h._"
check "5b: ...followed by the clean-exit list" has "$out" "one-shot setup step"
check "5c: ...and no failed-boot subsection" hasnt "$out" '### `fp8-vllm-qwen38-27b-multi4-ultramax-kernels-1`'

# --- only old exits ---------------------------------------------------------------
: > "$tmp/ps"
container vllm-qwen38-27b-old o1 1 unless-stopped "30 hours ago"
container vllm-qwen38-27b-old-clean o2 0 no "40 hours ago"
out="$(run_section)"
check "6a: only old exits -> the >24h line" has "$out" "all >24h old"
check "6b: ...and no clean-exit list" hasnt "$out" "Exited cleanly"

echo "test-report-recent-exits: PASS ${PASS}, FAIL ${FAIL}"
[[ "$FAIL" -eq 0 ]]
