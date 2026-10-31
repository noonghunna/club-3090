#!/usr/bin/env bash
# test-rebench-soak-knobs — rebench-full.sh's soak step reaches soak-test.sh with the
# session and turn counts it documents.
#
# Why this exists: until 2026-10-31 the step passed SESSIONS= / TURNS=, names soak-test.sh
# never reads (its knobs are SOAK_SESSIONS / SOAK_TURNS). The halved default (10 sessions,
# 2026-05-11) never applied, so every rebench soak ran soak-test.sh's own 20, and on a
# ~30 tok/s CPU-offload model it ran out of its 1800 s cap at session 15 of 20 (#1857).
#
#   1. every variable the step hands soak-test.sh is one soak-test.sh reads;
#   2. the step itself, run with a stub run_step, gives soak-test.sh 10 sessions in fresh
#      mode, the caller's SOAK_SESSIONS / SOAK_TURNS when set, and soak-test.sh's own
#      preset in continuous mode.
#
# ⚠️ Against the pre-fix rebench-full.sh (REBENCH=<path>) this must FAIL.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RB="${REBENCH:-${ROOT}/scripts/rebench-full.sh}"
SOAK="${ROOT}/scripts/soak-test.sh"
fail() { echo "FAIL: test-rebench-soak-knobs — $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }

step="$(awk '/^# --- step 5: soak-test/{f=1} f && /^# --- final/{exit} f' "$RB")"
[[ -n "$step" ]] || fail "rebench-full.sh's step 5 (soak-test) not found"
command grep -q 'scripts/soak-test.sh' <<<"$step" || fail "step 5 no longer runs soak-test.sh — update this test"

# --- 1. names soak-test.sh actually reads --------------------------------------------
prefix="$(awk '/^URL=/{f=1} f; f && /run_step soak/{exit}' <<<"$step")"
names="$(command grep -oE '(^|[[:space:]])[A-Z][A-Z0-9_]*=' <<<"$prefix" | tr -d ' =' | sort -u)"
[[ -n "$names" ]] || fail "no variables found on the soak step's command"
for n in $names; do
  command grep -qE "\\\$\\{?${n}\\b" "$SOAK" || fail "the soak step passes ${n}=, which soak-test.sh never reads"
done
pass "every variable the soak step passes is one soak-test.sh reads: $(tr '\n' ' ' <<<"$names")"

# --- 2. what soak-test.sh is handed ---------------------------------------------------
# seen [VAR=value...] — run step 5 as rebench does, with run_step reporting the SOAK_*
# variables its child process would get.
seen() {
  env -u SOAK_SESSIONS -u SOAK_TURNS -u SOAK_MODE "$@" bash -c '
    set -euo pipefail
    URL=http://127.0.0.1:1 MODEL=m OUT_DIR=/nonexistent ROOT_DIR=/nonexistent
    run_step() { env | command grep -E "^SOAK_(MODE|SESSIONS|TURNS)=" | sort | tr "\n" " "; }
    eval "$1"' _ "$step"
}
check() {  # label, expected, [VAR=value...]
  local got; got="$(seen "${@:3}")"
  [[ "$got" == "$2" ]] || fail "$1: expected '$2', got '$got'"
  pass "$1 → $2"
}
check "fresh mode, nothing set" "SOAK_MODE=fresh SOAK_SESSIONS=10 SOAK_TURNS=5 "
check "the caller's SOAK_SESSIONS / SOAK_TURNS win" "SOAK_MODE=fresh SOAK_SESSIONS=20 SOAK_TURNS=3 " \
  SOAK_SESSIONS=20 SOAK_TURNS=3
check "continuous mode leaves the session count to soak-test.sh" "SOAK_MODE=continuous SOAK_SESSIONS= SOAK_TURNS=5 " \
  SOAK_MODE=continuous
# The continuous case hands over an EMPTY SOAK_SESSIONS: soak-test.sh must read it as unset.
command grep -qF 'SOAK_SESSIONS="${SOAK_SESSIONS:-5}"' "$SOAK" \
  || fail "soak-test.sh's continuous preset no longer treats an empty SOAK_SESSIONS as unset — the case above relies on it"
pass "soak-test.sh reads an empty SOAK_SESSIONS as its own preset (continuous: 5)"

echo "test-rebench-soak-knobs: ok"
