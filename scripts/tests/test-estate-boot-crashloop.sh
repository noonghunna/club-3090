#!/usr/bin/env bash
# test-estate-boot-crashloop.sh — `pod.sh up` (estate_cli boot) must stop
# waiting once the container is crash-looping or has stopped (#1936).
#
# Under `restart: unless-stopped` docker reports `.State.Running == true` for a
# crash-looping container, so the old Running-only check never fired and the
# wait ran out its full 600 s on a container that died seconds in. The wait now
# uses switch.sh's rule (#1099): any state but `running` fails, and so does a
# RestartCount above the one at the start of the wait.
#
# Hardware-free: a fake `docker` on PATH replays scripted `docker inspect`
# answers, one per call; the endpoint probe is stubbed in-process.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
ok()  { echo "  PASS  $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL  $1"; FAIL=$((FAIL + 1)); }

# Fake docker. `inspect` answers from the next line of $FAKE_STATES, written
# `<status> <restart count> <exit code>` (the last line repeats); a line
# `MISSING` makes inspect fail like an unknown container. A `.State.Running`
# query gets what docker says: true while running AND while restarting.
# `logs` prints a stand-in traceback.
mkdir -p "$TMP_DIR/bin"
cat > "$TMP_DIR/bin/docker" <<'SH'
#!/usr/bin/env bash
case "$1" in
  inspect)
    n="$(cat "$FAKE_COUNTER" 2>/dev/null || echo 0)"
    total="$(wc -l < "$FAKE_STATES")"
    line_no=$(( n + 1 > total ? total : n + 1 ))
    echo $(( n + 1 )) > "$FAKE_COUNTER"
    line="$(sed -n "${line_no}p" "$FAKE_STATES")"
    if [[ "$line" == MISSING ]]; then
      echo "Error: No such object: $4" >&2
      exit 1
    fi
    if [[ "$3" == "{{.State.Running}}" ]]; then
      case "${line%% *}" in running|restarting) echo true ;; *) echo false ;; esac
    else
      echo "$line"
    fi
    ;;
  logs)
    echo "ValueError: invalid literal for int() with base 10: 'GPU-fake'"
    ;;
  *)
    echo "fake docker: unexpected call: $*" >&2
    exit 97
    ;;
esac
SH
chmod +x "$TMP_DIR/bin/docker"

# run_case <label> <wait fn> <timeout s> <ready on poll N, 0 = never> <state lines...>
# Prints the harness's output: RESULT=ok|raised, ELAPSED=<s>, then the error.
run_case() {
  local label="$1" fn="$2" timeout="$3" ready_on="$4"
  shift 4
  printf '%s\n' "$@" > "$TMP_DIR/states.$label"
  rm -f "$TMP_DIR/counter.$label"
  (
    cd "$ROOT_DIR"
    PATH="$TMP_DIR/bin:$PATH" \
    FAKE_STATES="$TMP_DIR/states.$label" FAKE_COUNTER="$TMP_DIR/counter.$label" \
    CLUB3090_ESTATE_POLL_INTERVAL=0.05 \
    python3 - "$fn" "$timeout" "$ready_on" <<'PY' 2>&1
import sys
import time

from scripts.lib.profiles import estate_cli as ec

fn, timeout, ready_on = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
polls = {"n": 0}


def fake_endpoint_ready(port):
    polls["n"] += 1
    return ready_on > 0 and polls["n"] >= ready_on


ec.endpoint_ready = fake_endpoint_ready
inst = ec.InstanceSpec(name="crashy", compose_name="vllm/gemma-12b-qat-w4a16-single", gpu_indices=(2,), port=8040)
start = time.monotonic()
try:
    getattr(ec, fn)(inst, timeout)
    print("RESULT=ok")
except ec.EstateCliError as exc:
    print("RESULT=raised")
    print(f"ERROR={exc}")
print(f"ELAPSED={time.monotonic() - start:.2f}")
PY
  )
}

expect() {   # expect <label> <output> <needle>...
  local label="$1" out="$2"
  shift 2
  local needle
  for needle in "$@"; do
    if [[ "$out" != *"$needle"* ]]; then
      bad "$label: missing '$needle'"
      echo "$out" | sed 's/^/        | /'
      return
    fi
  done
  ok "$label"
}

fast() {     # fast <label> <output>: the wait gave up well before its 60 s timeout
  local label="$1" out="$2" secs
  secs="$(printf '%s\n' "$out" | sed -n 's/^ELAPSED=//p')"
  if [[ -n "$secs" ]] && python3 -c "import sys; sys.exit(0 if float('$secs') < 5 else 1)"; then
    ok "$label: gave up after ${secs}s, not the timeout"
  else
    bad "$label: took ${secs:-?}s (timeout 60)"
  fi
}

echo "[1] crash-loop that reads 'running' at every poll: the counter catches it"
out="$(run_case loop wait_ready 60 0 "running 0 0" "running 0 0" "running 1 1")"
expect "crash-loop fails" "$out" "RESULT=raised" "club3090-crashy is not coming up" "crash-looping (restarts 0→1)" "invalid literal for int()" "bash scripts/pod.sh down crashy"
fast "crash-loop" "$out"

echo "[2] container between two restarts (state=restarting)"
out="$(run_case restarting wait_ready 60 0 "running 0 0" "restarting 1 1")"
expect "restarting fails" "$out" "RESULT=raised" "state=restarting, exit=1"
fast "restarting" "$out"

echo "[3] container exited (no restart policy)"
out="$(run_case exited wait_ready 60 0 "running 0 0" "exited 0 1")"
expect "exited fails" "$out" "RESULT=raised" "state=exited, exit=1"
fast "exited" "$out"

echo "[4] container gone"
out="$(run_case missing wait_ready 60 0 "running 0 0" "MISSING")"
expect "missing fails" "$out" "RESULT=raised" "state=missing"
fast "missing" "$out"

echo "[5] positive control: an earlier restart count that holds still is not a crash"
# `up -d` can keep an existing container whose RestartCount is already 2;
# the baseline is read, not assumed 0. Ready on the 4th poll.
out="$(run_case kept wait_ready 60 4 "running 2 0")"
expect "kept container boots" "$out" "RESULT=ok" "crashy ready after"

echo "[6] a slow boot still waits for the timeout"
out="$(run_case slow wait_ready 1 0 "running 0 0")"
expect "slow boot times out" "$out" "RESULT=raised" "timeout waiting for crashy after 1s"

echo "[7] the parallel path (wait_ready_quiet) uses the same rule"
out="$(run_case quietloop wait_ready_quiet 60 0 "running 0 0" "running 0 0" "running 1 1")"
expect "quiet crash-loop fails" "$out" "RESULT=raised" "crash-looping (restarts 0→1)"
fast "quiet crash-loop" "$out"
out="$(run_case quietkept wait_ready_quiet 60 3 "running 5 0")"
expect "quiet kept container boots" "$out" "RESULT=ok"

echo "[8] --parallel: the instance log gets the whole error, stdout the one-line summary"
printf '%s\n' "running 0 0" "restarting 1 1" > "$TMP_DIR/states.par"
rm -f "$TMP_DIR/counter.par"
out="$(
  cd "$ROOT_DIR"
  PATH="$TMP_DIR/bin:$PATH" \
  FAKE_STATES="$TMP_DIR/states.par" FAKE_COUNTER="$TMP_DIR/counter.par" \
  CLUB3090_ESTATE_POLL_INTERVAL=0.05 \
  python3 - "$TMP_DIR/crashy.log" <<'PY' 2>&1
import sys
from pathlib import Path

from scripts.lib.profiles import estate_cli as ec

ec.run_compose = lambda inst, action, log_path=None: None
ec.endpoint_ready = lambda port: False
log_path = Path(sys.argv[1])
log_path.touch()
inst = ec.InstanceSpec(name="crashy", compose_name="vllm/gemma-12b-qat-w4a16-single", gpu_indices=(2,), port=8040)
outcome = ec.boot_instance_parallel(1, 1, inst, 60, log_path)
print(f"OK={outcome.ok}")
print(f"SUMMARY_LINES={len(outcome.error.splitlines())}")
print("SUMMARY=" + outcome.error)
log = log_path.read_text(encoding="utf-8")
print(f"LOG_TAIL={'invalid literal for int()' in log}")
print(f"LOG_HINT={'bash scripts/pod.sh down crashy' in log}")
PY
)"
expect "parallel outcome fails with a one-line summary" "$out" "OK=False" "SUMMARY_LINES=1" "SUMMARY=container club3090-crashy is not coming up (state=restarting, exit=1)"
expect "parallel instance log has the log tail and the stop hint" "$out" "LOG_TAIL=True" "LOG_HINT=True"

echo
echo "PASS=$PASS FAIL=$FAIL"
[[ "$FAIL" -eq 0 ]]
