#!/usr/bin/env bash
# test-results-dir — runs whose checkout results/ can't be written still write their results.
#
# Why (#2006, D#2002): on a checkout owned by another account, soak-test.sh stopped at
# `mkdir: cannot create directory 'results/report-soak-…': Permission denied`, so report.sh's soak
# stage read as Overall FAIL with no soak run at all, and concurrency-probe.sh's sweep lost its card.
# scripts/lib/results-dir.sh's club_results_dir now falls back to the per-user data dir and says so.
#
# Covers: the helper itself (writable path kept, read-only path moved, neither → rc 1 with the
# original path), a real soak-test.sh run against a read-only results/ (the summary lands under the
# data dir, and the run says where), and that report.sh and concurrency-probe.sh resolve their
# output through the helper. ⚠️ Against the pre-fix tree the soak leg fails: soak-test.sh exits on
# the mkdir and writes nothing.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
fail() { echo "FAIL: test-results-dir — $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }

if [[ "$(id -u)" == "0" ]]; then
  echo "SKIP: test-results-dir — root ignores directory permissions, so a read-only results/ can't be staged"
  exit 0
fi

tmp="$(mktemp -d)"
cleanup() { chmod -R u+w "$tmp" 2>/dev/null || true; rm -rf "$tmp"; }

# --- 1. the helper --------------------------------------------------------------------------------
# shellcheck source=../lib/results-dir.sh
source "${ROOT_DIR}/scripts/lib/results-dir.sh"
mkdir -p "$tmp/ok/results" "$tmp/ro/results" "$tmp/data" "$tmp/data-ro"
chmod 555 "$tmp/ro/results" "$tmp/data-ro"

got="$(CLUB3090_DATA_DIR="$tmp/data" club_results_dir "$tmp/ok/results/run-a" "$tmp/ok" 2>"$tmp/err1")"
[[ "$got" == "$tmp/ok/results/run-a" && -d "$got" && ! -s "$tmp/err1" ]] \
  || fail "a writable path must be kept as it is, quietly (got '$got')"
pass "writable results/ is used as it is"

got="$(CLUB3090_DATA_DIR="$tmp/data" club_results_dir "$tmp/ro/results/run-b" "$tmp/ro" 2>"$tmp/err2")"
[[ "$got" == "$tmp/data/results/run-b" && -d "$got" ]] \
  || fail "a read-only results/ must fall back to <data dir>/results/run-b (got '$got')"
command grep -qF "isn't writable here; writing to $tmp/data/results/run-b instead" "$tmp/err2" \
  || fail "the fallback must say where it wrote: $(cat "$tmp/err2")"
pass "read-only results/ → <data dir>/results/<run>, with a note"

got="$(cd "$tmp/ro" && CLUB3090_DATA_DIR="$tmp/data" club_results_dir "results/run-c" "$tmp/ro" 2>/dev/null)"
[[ "$got" == "$tmp/data/results/run-c" ]] || fail "a relative path keeps its results/<run> shape (got '$got')"
pass "a relative path keeps its results/<run> shape"

set +e
got="$(CLUB3090_DATA_DIR="$tmp/data-ro" club_results_dir "$tmp/ro/results/run-d" "$tmp/ro" 2>/dev/null)"; rc=$?
set -e
[[ $rc -eq 1 && "$got" == "$tmp/ro/results/run-d" ]] \
  || fail "neither writable must return 1 with the original path (rc $rc, got '$got')"
pass "neither writable → rc 1, original path (the caller's mkdir reports it)"

# --- 2. a real soak run against a read-only results/ ----------------------------------------------
# shellcheck source=fixtures/soak-harness/soak-env.sh
source "${ROOT_DIR}/scripts/tests/fixtures/soak-harness/soak-env.sh"
soak_env_init
trap 'soak_env_cleanup; cleanup' EXIT
printf 'ok\nok\n' > "${SOAK_ENV_DIR}/plan"
SOAK_SESSIONS=1 SOAK_TURNS=2
export SOAK_SESSIONS SOAK_TURNS
soak_stub_start "${SOAK_ENV_DIR}/plan"
soak_run "$ROOT_DIR" "$tmp/ro/results/soak-x" CLUB3090_DATA_DIR="$tmp/data"
soak_stub_stop
[[ -f "$tmp/data/results/soak-x/summary.md" ]] \
  || fail "a soak against a read-only results/ must write its summary under the data dir (rc ${SOAK_RC}):
${SOAK_OUT}"
[[ "$SOAK_OUT" == *"isn't writable here; writing to $tmp/data/results/soak-x instead"* ]] \
  || fail "the soak run must say where it wrote"
[[ "$SOAK_OUT" == *"artifacts: $tmp/data/results/soak-x"* ]] || fail "the artifacts line must name the fallback"
pass "soak-test.sh: read-only results/ → summary under the data dir (#2006)"

# --- 3. the other two callers resolve through the helper ------------------------------------------
command grep -qE 'soak_run_dir="\$\(club_results_dir "results/report-soak-' scripts/report.sh \
  || fail "report.sh must resolve its soak dir with club_results_dir, so it reads the summary from where it went"
command grep -qE 'out_dir="\$\(club_results_dir "\$ROOT_DIR/results"' scripts/concurrency-probe.sh \
  || fail "concurrency-probe.sh must resolve its sweep card's dir with club_results_dir (D#2002)"
pass "report.sh and concurrency-probe.sh resolve their output through club_results_dir"

echo "PASS: test-results-dir"
