#!/usr/bin/env bash
# test-report-exit-code — inner verdicts must reach report.sh's exit (#813).
#
# Committed to on #619: `report.sh --full` exited 0 while verify-stress printed
# "1 stress check(s) failed" inside. A --full chain whose exit code doesn't
# reflect inner verdicts is unsafe to automate against, which is the entire
# point of --full. It was caught only because @seanyourhighness read the inner
# verdicts, which most reporters reasonably won't.
#
# Contract asserted here:
#   0  every stage that ran passed, or no stage was requested
#   2  ADVISORY-only — the agent-safety VRAM margin at the context ceiling.
#      Recall is CORRECT there; what fails is headroom for sustained agent
#      load. Distinguishable so a caller can gate on hard failures alone.
#   1  hard failure — a correctness check failed, or a stage could not run
#   ...and a per-check summary names the failing check, so it is identifiable
#      from the exit path without reading every inner block.
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=fixtures/report-harness/report-env.sh
source "${ROOT_DIR}/scripts/tests/fixtures/report-harness/report-env.sh"

fail() { echo "FAIL: $1" >&2; exit 1; }
assert_contains() { [[ "$1" == *"$2"* ]] || { echo "FAIL: missing '$2' in report" >&2; echo "$1" >&2; exit 1; }; }
assert_not_contains() { [[ "$1" != *"$2"* ]] || { echo "FAIL: must NOT contain '$2'" >&2; echo "$1" >&2; exit 1; }; }
assert_rc() { [[ "$REPORT_RC" -eq "$1" ]] || fail "expected exit $1, got $REPORT_RC ($2)"; }

report_env_init "$ROOT_DIR"
trap report_env_cleanup EXIT

# The real verify-stress advisory, verbatim from the #619 run. It is printed by
# the ceiling-ladder check with a ⚠, NOT via fail() — so there is no ✗, and the
# script still exits with FAILED=1. That shape is exactly what makes the
# advisory distinguishable from a correctness failure.
ADVISORY_PASS='  ✓ ceiling ladder: all 2 rungs passed — fillable to 120320 tok (91% of n_ctx=131072)'
ADVISORY_WARN='    ⚠ VRAM margin thin at ceiling: 957 MB free < 1024 MB threshold'
ADVISORY_TAIL='1 stress check(s) failed. See hints above.'

all_pass() {
  report_stage_stub verify-full.sh   0 "All checks passed."
  report_stage_stub verify-stress.sh 0 "All stress / boundary checks passed."
  report_stage_stub soak-test.sh     0 "[soak]   verdict              PASS"
  report_stage_stub bench.sh         0 "decode_TPS mean=200.00"
  report_stage_stub bench-agentic.sh 0 "turn 12 ok"
}

report_stub_start

# ── 1. everything green → 0, and the summary says so ────────────────────────
all_pass
report_run --full --no-redact
assert_rc 0 "all stages passed"
assert_contains "$REPORT_OUT" "## Check summary"
assert_contains "$REPORT_OUT" "**Overall: PASS**"
assert_contains "$REPORT_OUT" "| verify-stress.sh | 0 | PASS |"

# ── 2. the #619 case: advisory-only inside verify-stress → exit 2 ───────────
all_pass
report_stage_stub verify-stress.sh 1 "$ADVISORY_PASS" "$ADVISORY_WARN" "$ADVISORY_TAIL"
report_run --full --no-redact
assert_rc 2 "advisory-only failure must be distinguishable"
assert_contains "$REPORT_OUT" "**Overall: ADVISORY**"
assert_contains "$REPORT_OUT" "| verify-stress.sh | 1 | ADVISORY |"
assert_contains "$REPORT_OUT" "agent-safety VRAM margin thin at ceiling"
# An advisory must not be reported as a correctness failure.
assert_not_contains "$REPORT_OUT" "**Overall: FAIL**"

# ── 3. --stress alone reaches the exit too, not just --full ─────────────────
all_pass
report_stage_stub verify-stress.sh 1 "$ADVISORY_PASS" "$ADVISORY_WARN" "$ADVISORY_TAIL"
report_run --stress --no-redact
assert_rc 2 "--stress alone must propagate the advisory"

# ── 4. a real correctness failure outranks the advisory → exit 1, named ────
all_pass
report_stage_stub verify-stress.sh 2 \
  '  ✗ ceiling ladder: wall at ~95000 tok — filled up to 90000 tok, then OOMed' \
  "$ADVISORY_WARN" \
  '2 stress check(s) failed. See hints above.'
report_run --full --no-redact
assert_rc 1 "a hard failure must outrank an advisory"
assert_contains "$REPORT_OUT" "**Overall: FAIL**"
assert_contains "$REPORT_OUT" "| verify-stress.sh | 2 | FAIL |"
# #813: "print a per-check summary line so the failing check is nameable from
# the exit path" — the check's own text must appear, not just a count.
assert_contains "$REPORT_OUT" "ceiling ladder: wall at ~95000 tok"

# ── 5. a failing non-stress stage propagates too ────────────────────────────
all_pass
report_stage_stub verify-full.sh 1 '  ✗ tool_calls[] empty — model did not call get_weather' '1 check(s) failed.'
report_run --verify --no-redact
assert_rc 1 "verify-full failure must propagate"
assert_contains "$REPORT_OUT" "tool_calls[] empty"

# ── 6. soak FAIL (no ✗ in its output) still counts as a hard failure ────────
all_pass
report_stage_stub soak-test.sh 1 "[soak]   verdict              FAIL" "[soak]   errors               7"
report_run --soak --no-redact
assert_rc 1 "soak FAIL must propagate"
assert_contains "$REPORT_OUT" "| soak-test.sh | 1 | FAIL |"

# ── 7. back-compat: no stage flags → no Check summary, exit 0 ──────────────
# The default hardware/stack report is what test-diagnose-estate.sh runs under
# `set -e`; it must keep exiting 0 and gain no new sections.
all_pass
report_run --no-redact
assert_rc 0 "the default report must still exit 0"
assert_not_contains "$REPORT_OUT" "## Check summary"
assert_not_contains "$REPORT_OUT" "## Engine liveness"

echo "PASS: test-report-exit-code"
