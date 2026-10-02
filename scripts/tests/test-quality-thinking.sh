#!/usr/bin/env bash
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

tmp_bin="$(mktemp -d)"
tmp_log="$(mktemp)"
before_list="$(mktemp)"
after_list="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before_list" || true
cleanup() {
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$after_list" || true
  comm -13 "$before_list" "$after_list" | xargs -r rm -f
  rm -rf "$tmp_bin"
  rm -f "$tmp_log" "$before_list" "$after_list"
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
    */props)
      printf '{"reasoning":"on"}'
      exit 0
      ;;
  esac
done
exit 0
MOCK_CURL
chmod +x "${tmp_bin}/curl"

cat > "${tmp_bin}/benchlocal-cli" <<'MOCK_BENCHLOCAL'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${BENCHLOCAL_MOCK_LOG}"
# The clocks the wrapper hands benchlocal through the environment, not argv.
if [[ -n "${BENCHLOCAL_MOCK_ENV_LOG:-}" && "$*" != *"--help"* ]]; then
  printf 'turn=%s hermes=%s\n' "${BENCHLOCAL_MODEL_TURN_TIMEOUT:-}" "${BENCHLOCAL_HERMES_SUBPROCESS_TIMEOUT_S:-}" >> "$BENCHLOCAL_MOCK_ENV_LOG"
fi
if [[ "$*" == "run --help" && -n "${BENCHLOCAL_MOCK_HELP:-}" ]]; then
  printf '%s\n' "$BENCHLOCAL_MOCK_HELP"
  exit 0
fi
json_out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --save-json)
      json_out="${2:-}"
      shift 2
      ;;
    list)
      echo 'toolcall-15'
      exit 0
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -n "$json_out" ]]; then
  mkdir -p "$(dirname "$json_out")"
  cat > "$json_out" <<'JSON'
{"packs":[{"pack_id":"toolcall-15","status":"ok","passed":1,"total":1,"score":1.0}]}
JSON
fi
exit 0
MOCK_BENCHLOCAL
chmod +x "${tmp_bin}/benchlocal-cli"

out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model ENABLE_THINKING=1 THINKING_MAX_TOKENS=4096 bash scripts/quality-test.sh --quick 2>&1)"
assert_contains "$out" "[quality-test] thinking: enabled"
assert_contains "$out" "[quality-test] thinking max tokens: 4096 (applies to thinking-enabled packs)"
args="$(cat "$tmp_log")"
assert_contains "$args" "--enable-thinking"
assert_contains "$args" "--thinking-max-tokens 4096"


: > "$tmp_log"
out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model THINKING_MAX_TOKENS=8192 bash scripts/quality-test.sh --reasoning 2>&1)"
assert_contains "$out" "[quality-test] thinking max tokens: 8192 (applies to thinking-enabled packs)"
args="$(cat "$tmp_log")"
assert_contains "$args" "--reasoning"
assert_contains "$args" "--thinking-max-tokens 8192"
if [[ "$args" == *"--enable-thinking"* ]]; then
  echo "ASSERTION FAILED: quality-test forced --enable-thinking when only THINKING_MAX_TOKENS was set" >&2
  echo "$args" >&2
  exit 1
fi

: > "$tmp_log"
out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model bash scripts/quality-test.sh --quick 2>&1)"
assert_contains "$out" "WARN: server appears to have reasoning enabled"
args="$(cat "$tmp_log")"
if [[ "$args" == *"--enable-thinking"* ]]; then
  echo "ASSERTION FAILED: quality-test forwarded --enable-thinking while ENABLE_THINKING=0" >&2
  echo "$args" >&2
  exit 1
fi

# --- --max-tokens / MAX_TOKENS passthrough (overrides per-pack budget for BOTH arms) ---
# Lets a verbose model that self-truncates the deterministic packs be benched at a
# higher completion budget (benchlocal-cli already supports --max-tokens; this wires
# the club-3090 wrapper to forward it).
: > "$tmp_log"
out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model MAX_TOKENS=4096 bash scripts/quality-test.sh --quick 2>&1)"
assert_contains "$out" "[quality-test] max tokens: 4096 (overrides the per-pack completion budget for both arms)"
args="$(cat "$tmp_log")"
assert_contains "$args" "--max-tokens 4096"

# the --max-tokens flag form is equivalent to the MAX_TOKENS env var
: > "$tmp_log"
out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model bash scripts/quality-test.sh --quick --max-tokens 2048 2>&1)"
args="$(cat "$tmp_log")"
assert_contains "$args" "--max-tokens 2048"

# a non-integer --max-tokens is rejected (exit 2), like --thinking-max-tokens
if PATH="${tmp_bin}:$PATH" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model bash scripts/quality-test.sh --quick --max-tokens abc >/dev/null 2>&1; then
  echo "ASSERTION FAILED: --max-tokens accepted a non-integer" >&2
  exit 1
fi

# --- budget defaults (docs/RUN_EVALS.md "Budgets: set for you") ---
# Without flags the wrapper sends the published budgets instead of benchlocal's
# per-pack ~1024 tokens / 300s clocks; every value stays overridable, and
# --pack-budgets restores benchlocal's own.
tmp_env="$(mktemp)"
run_qt() {  # run_qt <env assignments...> -- <wrapper args...>
  local envs=() ; while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done; shift
  : > "$tmp_log"; : > "$tmp_env"
  # Start from none of the budget knobs set, whatever the calling shell exports.
  out="$(env -u MAX_TOKENS -u THINKING_MAX_TOKENS -u PACK_BUDGETS -u TIMEOUT_PER_CASE \
    -u BENCHLOCAL_MODEL_TURN_TIMEOUT -u BENCHLOCAL_HERMES_SUBPROCESS_TIMEOUT_S \
    "${envs[@]+"${envs[@]}"}" PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" BENCHLOCAL_MOCK_ENV_LOG="$tmp_env" \
    BENCHLOCAL_MOCK_HELP="--run-meta KEY=VALUE" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
    bash scripts/quality-test.sh "$@" 2>&1)"
  args="$(command grep -v -- '--help' "$tmp_log" || true)"
  envs_seen="$(cat "$tmp_env")"
}
assert_not_contains() {
  if [[ "$1" == *"$2"* ]]; then
    echo "ASSERTION FAILED: expected output NOT to contain: $2" >&2; echo "--- output ---" >&2; echo "$1" >&2; exit 1
  fi
}

# 1. nothing set -> the defaults, sent and recorded
run_qt -- --quick
assert_contains "$args" "--max-tokens 4096"
assert_contains "$args" "--thinking-max-tokens 16384"
assert_contains "$envs_seen" "turn=900 hermes=600"
assert_contains "$out" "budgets: max 4096 · thinking 16384 · model turn 900s · hermes episode 600s (wrapper default: max-tokens thinking-max-tokens model-turn hermes-episode)"
assert_contains "$args" "--run-meta budgets=max 4096 · thinking 16384"
assert_not_contains "$args" "--timeout-per-case"   # per-case clocks stay auto-scaled

# 2. every value is overridable, by flag and by env, and only the rest default
run_qt BENCHLOCAL_MODEL_TURN_TIMEOUT=1200 -- --quick --max-tokens 2048 --thinking-max-tokens 8192
assert_contains "$args" "--max-tokens 2048"
assert_contains "$args" "--thinking-max-tokens 8192"
assert_not_contains "$args" "--max-tokens 4096"
assert_contains "$envs_seen" "turn=1200 hermes=600"
assert_contains "$out" "(wrapper default: hermes-episode)"
run_qt MAX_TOKENS=3000 THINKING_MAX_TOKENS=9000 BENCHLOCAL_HERMES_SUBPROCESS_TIMEOUT_S=450 -- --quick
assert_contains "$args" "--max-tokens 3000"
assert_contains "$args" "--thinking-max-tokens 9000"
assert_contains "$envs_seen" "hermes=450"
run_qt -- --quick --timeout-per-case 900
assert_contains "$args" "--timeout-per-case 900"

# 3. --pack-budgets / PACK_BUDGETS=1 -> benchlocal's own budgets, nothing filled in
for form in "--pack-budgets" "PACK_BUDGETS=1"; do
  if [[ "$form" == --* ]]; then run_qt -- --quick "$form"
  else run_qt "$form" -- --quick; fi
  assert_not_contains "$args" "--max-tokens"
  assert_not_contains "$args" "--thinking-max-tokens"
  assert_contains "$envs_seen" "turn= hermes="
  assert_contains "$out" "budgets: benchlocal per-pack (--pack-budgets)"
done
# ...while an explicit value still goes through under --pack-budgets
run_qt -- --quick --pack-budgets --max-tokens 2048
assert_contains "$args" "--max-tokens 2048"
rm -f "$tmp_env"

echo "test-quality-thinking: ok"
