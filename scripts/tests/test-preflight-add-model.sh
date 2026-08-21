#!/usr/bin/env bash
set -euo pipefail
#
# Smoke test for scripts/preflight-add-model.sh (contract C5): the script
# exists, --help works, a missing/extra argument fails cleanly, and an unknown
# slug fails cleanly through the summary path (diagnose-profile goes red, the
# full-suite note still prints). The full green run on a healthy slug is done
# manually / by the lead — this test only guards the CLI contract so a broken
# entry point can't ship.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFLIGHT="${ROOT_DIR}/scripts/preflight-add-model.sh"
export PYTHONUTF8="${PYTHONUTF8:-1}"

fail() { echo "ASSERTION FAILED: $*" >&2; exit 1; }

[[ -f "${PREFLIGHT}" ]] || fail "scripts/preflight-add-model.sh does not exist"
[[ -x "${PREFLIGHT}" ]] || fail "scripts/preflight-add-model.sh is not executable"

# --help exits 0 and names the slug argument.
out="$(bash "${PREFLIGHT}" --help 2>&1)"
if [[ "$(echo "$out" | head -1)" != *"Usage:"* ]]; then
  fail "--help did not print usage: ${out}"
fi

# No slug -> usage error (exit 64), no traceback.
if out="$(bash "${PREFLIGHT}" 2>&1)"; then
  fail "missing-slug invocation unexpectedly succeeded"
fi
[[ "$out" == *"Usage:"* ]] || fail "missing-slug error lost the usage text"

# Unknown slug -> non-zero exit THROUGH the summary path (not a crash): the
# diagnose gate shows as failed and the full-suite-authoritative note prints.
set +e
out="$(bash "${PREFLIGHT}" nonexistent/slug-gone 2>&1)"
rc=$?
set -e
if [[ "$rc" == "0" ]]; then
  fail "unknown slug unexpectedly passed preflight"
fi
[[ "$out" == *"[preflight] diagnose-profile nonexistent/slug-gone"* ]] \
  || fail "unknown-slug run never invoked diagnose-profile on the slug"
[[ "$out" == *"gate(s) FAILED."* ]] || fail "unknown-slug run had no failure summary"
[[ "$out" == *"remains authoritative before commit"* ]] \
  || fail "full-suite note missing from unknown-slug run"

echo "test-preflight-add-model: ok"
