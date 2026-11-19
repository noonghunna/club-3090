#!/usr/bin/env bash
# test-report-size-notice — report.sh says when its output is too long for one GitHub post (#2103).
#
# GitHub takes at most 65,536 characters in an issue or comment; #2103's `report.sh --full` was
# over, and its reporter trimmed 93 log lines and the tail by hand to post it. report.sh now ends
# with a notice on STDERR, so it never lands in the saved report:
#   1. --full, stdout a file under the limit: says how long the file is and asks for it to be
#      attached as well as pasted;
#   2. stdout a file over the limit (REPORT_ISSUE_LIMIT lowered): a ⚠ warning naming the file and
#      its length, counted the way the test counts it;
#   3. --full, stdout a pipe (nothing to measure): the general rule, with the `> my-rig.md` form;
#   4. a run without --full under the limit: no notice at all;
#   5. a run without --full OVER the limit: the warning still fires;
#   and in every case the report itself carries no `[report]` line.
# ⚠️ Against the pre-#2103 tree this must FAIL (legs 1, 2, 3, 5).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"   # #779

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0
ok() { PASS=$((PASS+1)); }
no() { echo "FAIL: $1" >&2; FAIL=$((FAIL+1)); }
has()   { if [[ "$2" == *"$3"* ]]; then ok; else no "$1: missing '$3' in: $2"; fi; }
hasnt() { if [[ "$2" != *"$3"* ]]; then ok; else no "$1: must NOT contain '$3' in: $2"; fi; }

# shellcheck source=fixtures/report-harness/report-env.sh
source "${ROOT_DIR}/scripts/tests/fixtures/report-harness/report-env.sh"
report_env_init "$ROOT_DIR"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"; report_env_cleanup' EXIT

report_stage_stub verify-full.sh   0 "All checks passed."
report_stage_stub verify-stress.sh 0 "All stress / boundary checks passed."
report_stage_stub soak-test.sh     0 "[soak]   verdict              PASS"
report_stage_stub bench.sh         0 "decode_TPS mean=200.00"
report_stage_stub bench-agentic.sh 0 "turn 12 ok"
report_stub_start

run_file() {  # <stdout file> <stderr file> [report.sh args...]
  local out="$1" err="$2"; shift 2
  env PATH="${REPORT_FAKE_BIN}:$PATH" URL="${REPORT_STUB_URL:-}" HOME="${REPORT_ENV_DIR}/home" \
      REPORT_FULL_CALIBRATION=0 bash "${REPORT_FAKE_ROOT}/scripts/report.sh" "$@" > "$out" 2> "$err" || true
}
commas() { python3 -c 'import sys; print(f"{int(sys.argv[1]):,}")' "$1"; }

# --- 1. --full to a file, under the limit -------------------------------------------------------------
run_file "$OUT/my-rig.md" "$OUT/full.err" --full --no-redact
ERR="$(cat "$OUT/full.err")"; N="$(commas "$(wc -m < "$OUT/my-rig.md")")"
has   "--full, file: names the file and its length"  "$ERR" "[report] my-rig.md is ${N} characters, within GitHub's 65,536-character limit"
has   "--full, file: asks for the file as well"      "$ERR" "Attach the file to the issue as well as pasting it"
hasnt "--full, file: no warning under the limit"     "$ERR" "[report] ⚠"
hasnt "--full, file: the notice is not in the report" "$(cat "$OUT/my-rig.md")" "[report]"

# --- 2. a file over the limit ---------------------------------------------------------------------
REPORT_ISSUE_LIMIT=500 run_file "$OUT/my-rig.md" "$OUT/over.err" --full --no-redact
ERR="$(cat "$OUT/over.err")"; N="$(commas "$(wc -m < "$OUT/my-rig.md")")"
has   "over: the warning, the file and its length"   "$ERR" "[report] ⚠ my-rig.md is ${N} characters, more than GitHub takes in one issue or comment (500)"
has   "over: says to attach it"                      "$ERR" "Attach the file to the issue"
hasnt "over: not the under-the-limit line"           "$ERR" "within GitHub's"
hasnt "over: the notice is not in the report"        "$(cat "$OUT/my-rig.md")" "[report]"

# --- 3. --full to a pipe: nothing to measure ------------------------------------------------------------
env PATH="${REPORT_FAKE_BIN}:$PATH" URL="${REPORT_STUB_URL:-}" HOME="${REPORT_ENV_DIR}/home" REPORT_FULL_CALIBRATION=0 \
  bash "${REPORT_FAKE_ROOT}/scripts/report.sh" --full --no-redact 2> "$OUT/pipe.err" | cat > "$OUT/piped.md"
ERR="$(cat "$OUT/pipe.err")"
has   "--full, pipe: the general rule"               "$ERR" "A --full report can run past GitHub's 65,536-character limit"
has   "--full, pipe: how to save it"                 "$ERR" "bash scripts/report.sh --full > my-rig.md"
hasnt "--full, pipe: no length claimed"              "$ERR" " characters, within"
hasnt "--full, pipe: the notice is not in the report" "$(cat "$OUT/piped.md")" "[report]"

# --- 4. no --full, under the limit: quiet -------------------------------------------------------------
run_file "$OUT/bench.md" "$OUT/bench.err" --bench --no-redact
hasnt "--bench under the limit: no notice"           "$(cat "$OUT/bench.err")" "[report]"

# --- 5. no --full, over the limit: still warned ---------------------------------------------------------
REPORT_ISSUE_LIMIT=500 run_file "$OUT/plain.md" "$OUT/plain.err"
has   "no flags, over the limit: warned"             "$(cat "$OUT/plain.err")" "[report] ⚠ plain.md is"

echo "PASS: $PASS  FAIL: $FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
echo "test-report-size-notice: ok (file under / over the limit, pipe, quiet without --full, warned over it)"
