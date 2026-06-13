#!/usr/bin/env bash
# Offline test for scripts/quality-baseline.sh (#252). Uses --dry-run so it never
# needs a live endpoint or benchlocal-cli — it only asserts that the wrapper resolves
# the correct quality-test.sh command + baseline path for each mode, and that the
# guard rails (required slug, valid mode, positive --repeat, missing-baseline) fire.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

QB="scripts/quality-baseline.sh"
# A throwaway slug that can never collide with a real (Phase-2) baseline filename.
FAKE_SLUG="test/fake-slug-252"
FAKE_FILE="results/baselines/test-fake-slug-252__no-thinking.json"

LAST_OUT=""
# rm -f never errors on a missing file and always returns 0, so the EXIT trap
# can't poison the script's exit code (the FAKE_FILE slug is unique to this test).
cleanup() { rm -f "$FAKE_FILE"; }
trap cleanup EXIT

assert_contains() {
  local haystack="$1" needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}
assert_not_contains() {
  local haystack="$1" needle="$2"
  if [[ "$haystack" == *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output NOT to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}
# run the wrapper, capture combined output, assert on the exit code.
expect_rc() {
  local want="$1"; shift
  local rc
  set +e
  LAST_OUT="$("$@" 2>&1)"
  rc=$?
  set -e
  if [[ "$rc" != "$want" ]]; then
    echo "ASSERTION FAILED: expected exit $want, got $rc for: $*" >&2
    echo "--- output ---" >&2
    echo "$LAST_OUT" >&2
    exit 1
  fi
}

# 1. capture, no-thinking (default mode) — dry-run resolves the canonical command.
expect_rc 0 bash "$QB" --slug "$FAKE_SLUG" --capture --dry-run
assert_contains "$LAST_OUT" "--full --no-thinking --repeat 3"
assert_contains "$LAST_OUT" "--save-json"
assert_contains "$LAST_OUT" "$FAKE_FILE"
assert_not_contains "$LAST_OUT" "--previous-result"

# 2. capture, enable-thinking — mode flag + filename both flip.
expect_rc 0 bash "$QB" --slug "$FAKE_SLUG" --mode enable-thinking --capture --dry-run
assert_contains "$LAST_OUT" "--full --enable-thinking --repeat 3"
assert_contains "$LAST_OUT" "results/baselines/test-fake-slug-252__enable-thinking.json"

# 3. --repeat override flows into the command.
expect_rc 0 bash "$QB" --slug "$FAKE_SLUG" --capture --repeat 5 --dry-run
assert_contains "$LAST_OUT" "--repeat 5"

# 4. passthrough args land after the resolved command.
expect_rc 0 bash "$QB" --slug "$FAKE_SLUG" --capture --dry-run --sampling-from-server
assert_contains "$LAST_OUT" "--sampling-from-server"

# 5. diff mode with NO baseline on disk → hard error (exit 1), even under --dry-run.
rm -f "$FAKE_FILE"
expect_rc 1 bash "$QB" --slug "$FAKE_SLUG" --dry-run
assert_contains "$LAST_OUT" "no baseline"

# 6. diff mode with a baseline present → resolves --previous-result, not --save-json.
mkdir -p results/baselines
echo '{}' > "$FAKE_FILE"
expect_rc 0 bash "$QB" --slug "$FAKE_SLUG" --dry-run
assert_contains "$LAST_OUT" "--previous-result"
assert_contains "$LAST_OUT" "$FAKE_FILE"
assert_not_contains "$LAST_OUT" "--save-json"
rm -f "$FAKE_FILE"

# 7. missing --slug → usage error (exit 2).
expect_rc 2 bash "$QB" --capture --dry-run
assert_contains "$LAST_OUT" "--slug"

# 8. invalid --mode → usage error (exit 2).
expect_rc 2 bash "$QB" --slug "$FAKE_SLUG" --mode sideways --dry-run
assert_contains "$LAST_OUT" "--mode must be"

# 9. non-numeric --repeat → usage error (exit 2).
expect_rc 2 bash "$QB" --slug "$FAKE_SLUG" --repeat zero --dry-run
assert_contains "$LAST_OUT" "--repeat must be"

echo "test-quality-baseline: ok"
