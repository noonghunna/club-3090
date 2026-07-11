#!/usr/bin/env bash
# Guard for scripts/rerun-failed-packs.sh: syntax, arg refusal, dry-run plan
# (pack extraction + thinking-mode derivation from a synthetic RunResult).
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${ROOT_DIR}/scripts/rerun-failed-packs.sh"

fail() { echo "ASSERTION FAILED: $1" >&2; exit 1; }

# 1. syntax
bash -n "$S" || fail "bash -n syntax check"

# 2. refuses missing/absent JSON with exit 2
set +e
bash "$S" >/dev/null 2>&1; [[ $? -eq 2 ]] || fail "no-arg should exit 2"
bash "$S" /nonexistent.json >/dev/null 2>&1; [[ $? -eq 2 ]] || fail "missing file should exit 2"
set -e

# 3. dry-run plan on a synthetic result: only the failing pack, correct mode
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/r.json" <<'JSON'
{
  "thinking_enabled": true,
  "packs": [
    {"pack_id": "toolcall-15", "scenarios": [
      {"id": "TC-01", "passed": true, "failure_mode": "passed"}]},
    {"pack_id": "reasonmath-15", "scenarios": [
      {"id": "RM-04", "passed": false, "failure_mode": "wrong_answer"},
      {"id": "RM-05", "passed": true, "failure_mode": "passed"}]}
  ]
}
JSON
out="$(RERUN_DRY=1 bash "$S" "$TMP/r.json" 2>&1)"
echo "$out" | command grep -q "failed packs: reasonmath-15$" || fail "plan should list only reasonmath-15 (got: $out)"
echo "$out" | command grep -q -- "--enable-thinking" || fail "plan should derive --enable-thinking from thinking_enabled=true"
echo "$out" | command grep -q -- "--previous-result $TMP/r.json" || fail "plan should pass --previous-result"
echo "$out" | command grep -qv "toolcall-15" || true
echo "$out" | command grep -q "\[dry\] quality-test.sh --pack toolcall-15" && fail "clean pack must not be re-run"

# 4. all-clean result → exit 0, nothing to re-run
cat > "$TMP/clean.json" <<'JSON'
{"thinking_enabled": false, "packs": [{"pack_id": "toolcall-15", "scenarios": [{"id": "TC-01", "passed": true}]}]}
JSON
out="$(bash "$S" "$TMP/clean.json" 2>&1)"
echo "$out" | command grep -q "nothing to re-run" || fail "clean result should short-circuit"

echo "test-rerun-failed-packs: PASS"
