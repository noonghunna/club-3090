#!/usr/bin/env bash
# Test for switch.sh guard_sampler_env() — non-numeric TEMP/TEMPERATURE drop (#2059).
#
# Contract: TEMP/TEMPERATURE are the sampler-temperature knobs, and the composes
# interpolate them from switch.sh's own environment. A caller that exports TEMP
# as a filesystem path (AI-agent runtimes export TEMP/TMP/TMPDIR to scratch
# directories) used to reach vLLM as temperature=/path and crash-loop the
# container. guard_sampler_env(), called right after apply_launch_settings and
# before compose up, drops a NON-NUMERIC value with a one-line warning naming
# the variable and what it held; a numeric value keeps working as a knob.
#
# Validates:
#   1. TEMP=/some/path → warned by name+value, unset afterwards.
#   2. TEMPERATURE=abc → same treatment.
#   3. TEMP=0.7 / TEMP=1 / TEMP=9.5e-1 → kept, no warning (knob contract).
#   4. Empty/unset values → silent no-op.
#   5. The drop is visible to a child "compose": after the guard, a subprocess
#      inherits no TEMP — the interpolation source is clean. (The entrypoint's
#      sampler line itself is already pinned by test-compose-sampler-profiles.sh;
#      this pins the launcher side that feeds it.)
#
# Harness: extract guard_sampler_env() via sed, source it, assert on output and
# on the shell environment — same shape as test-switch-orphan-teardown.sh.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
unset TEMP TEMPERATURE 2>/dev/null || true
# hermetic: the very harness that runs CI-style tests may export TEMP as a path
# (#2059 is about that collision), so scrub it for this process and every child.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0
FAIL=0
assert_contains() {
  local haystack="$1" needle="$2" label="$3"
  if [[ "$haystack" == *"$needle"* ]]; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${label}: expected output to CONTAIN: ${needle}" >&2
    echo "--- guard output ---" >&2
    printf '%s\n' "$haystack" >&2
    FAIL=$((FAIL + 1))
  fi
}
assert_not_contains() {
  local haystack="$1" needle="$2" label="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${label}: expected output NOT to contain: ${needle}" >&2
    printf '%s\n' "$haystack" >&2
    FAIL=$((FAIL + 1))
  fi
}
assert_eq() {
  local got="$1" want="$2" label="$3"
  if [[ "$got" == "$want" ]]; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${label}: got '${got}' want '${want}'" >&2
    FAIL=$((FAIL + 1))
  fi
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
sed -n '/^guard_sampler_env() {/,/^}/p' scripts/switch.sh > "$TMP/guard.sh"
if ! grep -q '^guard_sampler_env() {' "$TMP/guard.sh"; then
  echo "FAIL: could not extract guard_sampler_env() from scripts/switch.sh" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$TMP/guard.sh"

echo "== 1. TEMP as a path is dropped, warned by name and value"
OUT="$(TEMP=/some/path bash -c 'source "$0" >/dev/null 2>"$1"; guard_sampler_env 2>&1; echo "child_TEMP=${TEMP:-UNSET}"' "$TMP/guard.sh" "$TMP/err1" 2>&1 || true)"
# run in a child shell so this test's own env stays clean:
OUT="$(TEMP=/some/path bash -c 'source "$0"; guard_sampler_env 2>&1; echo "|after=${TEMP:-UNSET}"' "$TMP/guard.sh")"
assert_contains "$OUT" "TEMP='/some/path'" "1a: warning names variable and value"
assert_contains "$OUT" "not a number — dropping it" "1b: warning says what it did"
assert_contains "$OUT" "|after=UNSET" "1c: TEMP unset in this process"

echo "== 2. TEMPERATURE as garbage is dropped the same way"
OUT="$(TEMPERATURE=not-a-number bash -c 'source "$0"; guard_sampler_env 2>&1; echo "|after=${TEMPERATURE:-UNSET}"' "$TMP/guard.sh")"
assert_contains "$OUT" "TEMPERATURE='not-a-number'" "2a: warning names variable and value"
assert_contains "$OUT" "|after=UNSET" "2b: TEMPERATURE unset"

echo "== 3. Numeric values survive (knob contract)"
for v in 0.7 1 9.5e-1; do
  OUT="$(TEMP="$v" bash -c 'source "$0"; guard_sampler_env 2>&1; echo "|after=${TEMP:-UNSET}"' "$TMP/guard.sh")"
  assert_not_contains "$OUT" "dropping it" "3-${v}a: no warning"
  assert_contains  "$OUT" "|after=$v"    "3-${v}b: value kept"
done

echo "== 4. Empty and unset are silent no-ops"
OUT="$(bash -c 'source "$0"; TEMP= guard_sampler_env 2>&1; echo "|silent"' "$TMP/guard.sh")"
assert_eq "$OUT" "|silent" "4a: empty TEMP silent"
OUT="$(bash -c 'source "$0"; guard_sampler_env 2>&1; echo "|silent"' "$TMP/guard.sh")"
assert_eq "$OUT" "|silent" "4b: nothing set → silent"

echo "== 5. The drop reaches a child compose process (interpolation source clean)"
OUT="$(TEMP=/some/path bash -c 'source "$0"; guard_sampler_env >/dev/null 2>&1; env | grep -c "^TEMP=" || true' "$TMP/guard.sh")"
assert_eq "${OUT:-0}" "0" "5: exported TEMP gone from the environment compose would inherit"

echo
if (( FAIL > 0 )); then
  echo "test-sampler-env-guard.sh: ${PASS} passed, ${FAIL} FAILED"
  exit 1
fi
echo "test-sampler-env-guard.sh: ${PASS} passed, 0 failed"
