#!/usr/bin/env bash
# test-concurrency-probe-verdict — guards the two verdict rules in
# scripts/lib/concurrency_probe.py that produced the false `FAIL — fit` in #1211.
#
# ⚠️ It imports and calls the REAL functions, and statically asserts the REAL
# wiring. A guard that re-implements the rule inline and tests the copy proves
# nothing: PR #1212's version of this test passed unchanged with the admission
# logic gutted to `run_peak = None`, because it never touched run_probe. Every
# assertion here fails if the corresponding production line is reverted — that
# was verified by reverting each one.
#
# The two rules:
#   vram_ok(leak, growth)         — SYMMETRIC band. The old `0 <= leak` failed a
#                                   -1 MB delta, which is what #1211 actually hit.
#   admission_ok(run_peaks, n)    — judged on the PER-ROUND PEAK, never the racy
#                                   last sample (#1212's insight).
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT_DIR/scripts/concurrency-probe.sh"
LIB="$ROOT_DIR/scripts/lib/concurrency_probe.py"
fail() { echo "FAIL: $1" >&2; exit 1; }

bash -n "$PROBE" || fail "bash -n: syntax error"
python3 -m py_compile "$LIB" || fail "py_compile concurrency_probe.py"
echo "  ✓ syntax"

# --- the REAL functions, imported ------------------------------------------------
PYTHONPATH="$ROOT_DIR/scripts/lib" python3 - "$LIB" <<'PY' || fail "verdict rules"
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cp", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["cp"] = m
spec.loader.exec_module(m)

bad = []
def ck(cond, msg):
    if not cond: bad.append(msg)

# --- vram_ok: the #1211 case -----------------------------------------------------
# The reporter's own N=2 trace: cold 44302 -> warm 44303 -> final 44302.
leak_1211 = 44302 - 44303
ck(leak_1211 == -1, "fixture drift: the #1211 delta is -1 MB")
ck(m.vram_ok(leak_1211, 200) is True,
   "#1211: a -1 MB post-warm delta must PASS (it failed under `0 <= leak`)")
ck(m.vram_ok(0, 200) is True,     "a 0 MB delta must pass")
ck(m.vram_ok(199, 200) is True,   "growth inside the band must pass")
ck(m.vram_ok(201, 200) is False,  "a real leak above the band must FAIL")
# Bounded below: a container collapsing mid-probe must still fail, not read clean.
ck(m.vram_ok(-20924, 200) is False,
   "VRAM falling by the size of the model must FAIL (collapsed container)")
ck(m.vram_ok(-201, 200) is False, "below the band must FAIL")

# --- admission_ok: judged on the peak -------------------------------------------
ck(m.admission_ok([1, 1, 1, 1, 1], 1) is True,  "N=1 peak=1 is admitted")
ck(m.admission_ok([2, 2], 2) is True,           "N=2 peak=2 is admitted")
ck(m.admission_ok([0, 2, 0], 2) is True,        "a dip after the peak is completion")
ck(m.admission_ok([1, 1], 2) is False,          "peak<N fails closed")
ck(m.admission_ok([], 2) is True,               "no engine data must not fail admission")
ck(m.admission_ok([None, None], 2) is True,     "all-None must not fail admission")

if bad:
    for b in bad: print("  ⛔", b)
    sys.exit(1)
print("  ✓ vram_ok symmetric band (#1211) · admission_ok on the peak (#1212)")
PY

# --- the WIRING (a correct rule called with the wrong list is still the bug) -----
command grep -q 'clean_fit = (bad == 0) and vram_ok(leak, GROWTH)' "$LIB" \
  || fail "clean_fit must use vram_ok() — a literal '0 <= leak' reintroduces #1211"
command grep -qE 'admitted_ok = admission_ok\(\s*runmax_by_round' "$LIB" \
  || fail "admission_ok must be called with runmax_by_round (the PEAK list); \
run_by_round is each round's LAST sample and reintroduces the race one level up"
command grep -qE 'run_peak = max\(\[r for r in runmax_by_round' "$LIB" \
  || fail "run_peak must come from the peak list"
command grep -q 'runmax_by_round.append' "$LIB" \
  || fail "the per-round peak must actually be collected"
# Admission must NOT be folded into the fit verdict: that puts a sampling race
# into a verdict that a clean run has already earned.
command grep -qE 'clean_fit = .*admitted_ok' "$LIB" \
  && fail "admission must NOT be a term in clean_fit — a run where every request \
completed must not fail on when the metrics sample landed"
echo "  ✓ wiring: peak list, vram_ok, and admission kept out of clean_fit"

# --- the sweep early-stop keys off the peak --------------------------------------
command grep -qE 'running_max.*-lt.*"\$n"' "$PROBE" \
  || fail "sweep early-stop must compare the PEAK running against n"
command grep -q 'running_max=' "$PROBE" \
  || fail "sweep must parse running_max from the RESULT line"
echo "  ✓ sweep early-stop keys off the peak running"

echo "test-concurrency-probe-verdict: ok"
