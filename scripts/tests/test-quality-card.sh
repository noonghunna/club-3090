#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-quality-card — guards the #981/#983E/#987 contract on top of the
# promoted --report/--report-out flags:
#
#   1. Card-path emission: when --report is given, the wrapper points at the
#      Results Card v2 wherever it landed (written path, or "printed above"
#      when only --report was passed), and warns when a requested card was
#      not actually written.
#   2. Version-stamped Quality: suffix (#981/#983E): per-pack versions from
#      the results JSON in compact tc form, plus thinking mode, sampling
#      source and thinking validity — each stamp present ONLY when the JSON
#      carries it (old schema-v1 files keep emitting cleanly).
#   3. #983A --both-modes orchestration: exactly two stubbed inner runs
#      (--no-thinking then --enable-thinking), both pinned to
#      --sampling-from-server, per-leg cards namespaced, pass-through and
#      worst-leg exit code preserved, and conflicts refused.
#
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WRAPPER="scripts/quality-test.sh"

fail() { echo "✗ $1" >&2; exit 1; }

assert_contains() {
  local haystack="$1" needle="$2" label="${3:-$2}"
  if ! grep -qF -- "$needle" <<<"$haystack"; then
    echo "✗ expected to contain: $label" >&2
    echo "--- actual ---" >&2
    printf '%s\n' "$haystack" >&2
    exit 1
  fi
}

assert_not_contains() {
  local haystack="$1" needle="$2" label="${3:-$2}"
  if grep -qF -- "$needle" <<<"$haystack"; then
    echo "✗ expected NOT to contain: $label" >&2
    echo "--- actual ---" >&2
    printf '%s\n' "$haystack" >&2
    exit 1
  fi
}

tmp_bin="$(mktemp -d)"
tmp_log="$(mktemp)"
tmp_out="$(mktemp -d)"
before_list="$(mktemp)"
after_list="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before_list" || true
cleanup() {
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$after_list" || true
  comm -13 "$before_list" "$after_list" | xargs -r rm -f
  rm -rf "$tmp_bin" "$tmp_out"
  rm -f "$tmp_log"
}
trap cleanup EXIT

cat > "${tmp_bin}/curl" <<'MOCK_CURL'
#!/usr/bin/env bash
for arg in "$@"; do
  case "$arg" in
    */v1/models)
      printf '{"data":[{"id":"mock-model"}]}'
      exit 0
      ;;
  esac
done
exit 0
MOCK_CURL
chmod +x "${tmp_bin}/curl"

# Mock benchlocal-cli: logs its full argv, honours --save-json and
# --report-out (unless BENCHLOCAL_MOCK_SKIP_REPORT=1), parameterizes the
# results-JSON schema (BENCHLOCAL_MOCK_OLD_SCHEMA=1) and thinking_validity
# (BENCHLOCAL_MOCK_VALIDITY), and can fail the thinking-on arm
# (BENCHLOCAL_MOCK_FAIL_THINKING=1) to exercise leg-failure propagation.
cat > "${tmp_bin}/benchlocal-cli" <<'MOCK_BENCHLOCAL'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${BENCHLOCAL_MOCK_LOG}"
json_out=""
report_out=""
saw_thinking=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --save-json) json_out="${2:-}"; shift 2 ;;
    --report-out) report_out="${2:-}"; shift 2 ;;
    --enable-thinking) saw_thinking=1; shift ;;
    *) shift ;;
  esac
done
if [[ "${BENCHLOCAL_MOCK_FAIL_THINKING:-0}" == "1" && "$saw_thinking" == "1" ]]; then
  exit 4
fi
if [[ -n "$json_out" ]]; then
  mkdir -p "$(dirname "$json_out")"
  if [[ "${BENCHLOCAL_MOCK_OLD_SCHEMA:-0}" == "1" ]]; then
    cat > "$json_out" <<'JSON'
{"packs":[{"pack_id":"toolcall-15","status":"ok","passed":1,"total":1,"score":1.0}]}
JSON
  else
    validity="${BENCHLOCAL_MOCK_VALIDITY:-ok}"
    cat > "$json_out" <<JSON
{"thinking_mode":"force-off","sampling_source":"server",
 "thinking_validity":{"toolcall-15":{"status":"${validity}"}},
 "packs":[{"pack_id":"toolcall-15","status":"ok","passed":14,"total":15,"score":0.933,"version":"1.0.1"}]}
JSON
  fi
fi
if [[ -n "$report_out" && "${BENCHLOCAL_MOCK_SKIP_REPORT:-0}" != "1" ]]; then
  mkdir -p "$(dirname "$report_out")"
  printf '## Quality bench\n' > "$report_out"
fi
exit 0
MOCK_BENCHLOCAL
chmod +x "${tmp_bin}/benchlocal-cli"

qrun() {
  PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" PREFLIGHT_NO_AUTODETECT=1 \
    URL=http://mock MODEL=mock-model bash "$WRAPPER" "$@"
}

# ---------------------------------------------------------------------------
echo "--- 1. card-path emission ---"
card="${tmp_out}/card.md"
out="$(qrun --quick --report md --report-out "$card" 2>&1)"
assert_contains "$out" "[quality-test] report: Results Card v2 (md)"
assert_contains "$out" "[quality-test] report-out: $card"
assert_contains "$out" "[quality-test] Results Card v2 → $card"
[[ -f "$card" ]] || fail "mock did not write the card at $card"

# --report without --report-out: benchlocal prints the card itself; say so
out="$(qrun --quick --report md 2>&1)"
assert_contains "$out" "Results Card v2 printed above (--report md)"

# --report-out pointing at an unwritable/unwritten location: WARN, not silence
out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" \
  BENCHLOCAL_MOCK_SKIP_REPORT=1 PREFLIGHT_NO_AUTODETECT=1 \
  URL=http://mock MODEL=mock-model \
  bash "$WRAPPER" --quick --report md --report-out /nonexistent-dir-x/card.md 2>&1)"
assert_contains "$out" "WARN: --report-out /nonexistent-dir-x/card.md was not written"

# no --report at all: no card chatter (back-compat output surface)
out="$(qrun --quick 2>&1)"
assert_not_contains "$out" "Results Card v2"

# ---------------------------------------------------------------------------
echo "--- 2. version-stamped Quality: suffix ---"
out="$(qrun --quick 2>&1)"
qline="$(grep -m1 '^Quality:   ' <<<"$out")"
[[ -n "$qline" ]] || fail "no Quality: line emitted"
assert_contains "$qline" "(--quick, thinking OFF, sampling=server, validity=valid, packs tc1.0.1, "
assert_contains "$qline" "$(date +%Y-%m-%d))"
# score portion unchanged
assert_contains "$qline" "toolcall-15 14/15 (93%)"

# CONTAMINATED validity propagates loudly
out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" \
  BENCHLOCAL_MOCK_VALIDITY=contaminated PREFLIGHT_NO_AUTODETECT=1 \
  URL=http://mock MODEL=mock-model bash "$WRAPPER" --quick 2>&1)"
qline="$(grep -m1 '^Quality:   ' <<<"$out")"
assert_contains "$qline" "validity=CONTAMINATED"

# old schema-v1 JSON (no version/sampling/thinking keys): stamps stay absent
out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" \
  BENCHLOCAL_MOCK_OLD_SCHEMA=1 PREFLIGHT_NO_AUTODETECT=1 \
  URL=http://mock MODEL=mock-model bash "$WRAPPER" --quick 2>&1)"
qline="$(grep -m1 '^Quality:   ' <<<"$out")"
assert_contains "$qline" "(--quick,"
assert_not_contains "$qline" "packs " "no packs stamp without version data"
assert_not_contains "$qline" "sampling="
assert_not_contains "$qline" "validity="
assert_not_contains "$qline" "thinking "

# ---------------------------------------------------------------------------
echo "--- 3. --both-modes orchestration (stubbed inner runs) ---"
: > "$tmp_log"
card="${tmp_out}/both.md"
set +e
out="$(qrun --quick --both-modes --report md --report-out "$card" 2>&1)"
rc=$?
set -e
[[ "$rc" == "0" ]] || fail "--both-modes exited $rc on healthy legs"

# exactly two inner runs, correct order and flags
invocations=$(grep -c "^run --endpoint" "$tmp_log" || true)
[[ "$invocations" == "2" ]] || fail "expected 2 inner benchlocal runs, got $invocations"
leg1="$(grep "^run --endpoint" "$tmp_log" | sed -n 1p)"
leg2="$(grep "^run --endpoint" "$tmp_log" | sed -n 2p)"
assert_contains "$leg1" "--no-thinking"
assert_contains "$leg1" "--sampling-from-server"
assert_not_contains "$leg1" "--enable-thinking"
assert_contains "$leg2" "--enable-thinking"
assert_contains "$leg2" "--sampling-from-server"
assert_not_contains "$leg2" "--no-thinking"

# per-leg cards are namespaced, base name untouched
assert_contains "$out" "Results Card v2 → ${tmp_out}/both.thinking-off.md"
assert_contains "$out" "Results Card v2 → ${tmp_out}/both.thinking-on.md"
[[ -f "${tmp_out}/both.thinking-off.md" && -f "${tmp_out}/both.thinking-on.md" ]] \
  || fail "per-leg cards missing"

# pass-through survives into BOTH legs (re-inserted as real wrapper flags,
# before the `--`, so the leg's own flags stay wrapper-parsed)
: > "$tmp_log"
qrun --quick --both-modes -- --retry-runaways >/dev/null 2>&1
leg1="$(grep "^run --endpoint" "$tmp_log" | sed -n 1p)"
leg2="$(grep "^run --endpoint" "$tmp_log" | sed -n 2p)"
assert_contains "$leg1" "--retry-runaways"
assert_contains "$leg2" "--retry-runaways"

# conflicts refused
for bad in "--no-thinking" "--enable-thinking"; do
  set +e
  out="$(qrun --quick --both-modes $bad 2>&1)"
  rc=$?
  set -e
  [[ "$rc" == "2" ]] || fail "--both-modes $bad should exit 2, got $rc"
  assert_contains "$out" "--both-modes runs the no-thinking leg then the enable-thinking leg itself"
done

# worst-leg exit code propagates (thinking leg fails, off leg succeeds)
: > "$tmp_log"
set +e
out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" \
  BENCHLOCAL_MOCK_FAIL_THINKING=1 PREFLIGHT_NO_AUTODETECT=1 \
  URL=http://mock MODEL=mock-model \
  bash "$WRAPPER" --quick --both-modes 2>&1)"
rc=$?
set -e
[[ "$rc" == "4" ]] || fail "--both-modes should propagate worst leg (4), got $rc"
assert_contains "$out" "leg 1/2 (OFF) exited 0"
assert_contains "$out" "leg 2/2 (ON) exited 4"

# recursion guard: BOTH_MODES=1 env must not loop forever
: > "$tmp_log"
if ! timeout 60 env PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" \
  PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model BOTH_MODES=1 \
  bash "$WRAPPER" --quick >/dev/null 2>&1; then
  fail "BOTH_MODES=1 env invocation failed"
fi
invocations=$(grep -c "^run --endpoint" "$tmp_log" || true)
[[ "$invocations" == "2" ]] || fail "env BOTH_MODES=1 produced $invocations inner runs (recursion?)"

echo "test-quality-card: ok"
