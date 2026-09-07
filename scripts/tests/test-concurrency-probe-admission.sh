#!/usr/bin/env bash
# test-concurrency-probe-admission — regression guard for the concurrency-probe
# admission race: a late `running < N` metrics sample (a request that finished
# before the sample was taken) must NOT be misread as the engine failing to
# admit all N streams. The honest admission signal is the PEAK running observed
# across the round's samples, and a genuine under-admission is only present
# when a request is actually observed WAITING while running < N.
#
# Fully offline: no server, no docker, no run_probe. It (a) unit-checks the
# admission rule the same way run_probe computes it, and (b) statically asserts
# the two call sites (the python verdict + the shell sweep early-stop) key off
# the peak + waiting, not the racy last sample.
set -euo pipefail

# Force Python UTF-8 mode (PEP 540) before the first python3 call (#779).
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT_DIR/scripts/concurrency-probe.sh"
LIB="$ROOT_DIR/scripts/lib/concurrency_probe.py"
fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. syntax
bash -n "$PROBE" || fail "bash -n: syntax error"
python3 -m py_compile "$LIB" || fail "py_compile concurrency_probe.py"
echo "  ✓ syntax"

# 2. the admission rule, exercised exactly as run_probe computes it:
#    admitted_ok = (run_peak is None) or (run_peak >= N)
#    (true -> the fit gate passes on admission; false -> FAIL — fit)
admitted_ok() {  # $1=N  $2=run_peak  -> echoes 1/0
  python3 -c '
import sys
N, rp = int(sys.argv[1]), sys.argv[2]
run_peak = None if rp == "-" else int(rp)
ok = (run_peak is None) or (run_peak >= N)
print(1 if ok else 0)
' "$1" "$2"
}
# N=1, request completes before the final metrics sample (last sample running=0):
# the peak was 1 (>= N), so it was admitted -> PASS.
[[ "$(admitted_ok 1 1)" == "1" ]] || fail "N=1 complete-before-sample must be admitted (PASS)"
# N=2, both admitted; one finishes before an intermediate sample (peak=2).
[[ "$(admitted_ok 2 2)" == "1" ]] || fail "N=2 one-finishes-before-sample must be admitted (PASS)"
# N=2, server genuinely only runs 1 (peak<N) whether or not the other queued:
# we cannot prove 2 were admitted -> FAIL.
[[ "$(admitted_ok 2 1)" == "0" ]] || fail "N=2 peak<2 (genuinely under-admitted) must FAIL"
# No engine stats at all (no container / no Running: lines): cannot prove
# under-admission, so it must not fail admission.
[[ "$(admitted_ok 2 -)" == "1" ]] || fail "missing engine stats must not fail admission"
echo "  ✓ admission rule: peak>=N admits; peak<N fails closed; late running<N is completion"

# 3. the python verdict + RESULT line key off the PEAK, not the last sample.
command grep -q 'run_peak is None' "$LIB" \
  || fail "run_probe must gate admission on the peak running (run_peak is None …)"
command grep -q 'run_peak >= N' "$LIB" \
  || fail "run_probe must treat run_peak >= N as admitted"
command grep -q 'admitted_ok' "$LIB" \
  || fail "run_probe must fold admitted_ok into the fit verdict"
command grep -q 'running_max' "$LIB" \
  || fail "RESULT line must carry running_max for the sweep driver"
command grep -q 'waiting_max' "$LIB" \
  || fail "RESULT line must carry waiting_max for the sweep driver"
echo "  ✓ python verdict + RESULT line use the peak, not the last sample"

# 4. the shell sweep early-stop keys off the PEAK running (running_max), not
#    the racy last-sample running= that produced the false FAIL — fit.
command grep -q 'running_max' "$PROBE" \
  || fail "sweep path must read running_max from the RESULT line"
command grep -qE 'running_max.*-lt.*"\$n"' "$PROBE" \
  || fail "sweep early-stop must compare the PEAK running against n"
echo "  ✓ sweep early-stop keys off the peak running, not the last sample"

echo "test-concurrency-probe-admission: ok"
