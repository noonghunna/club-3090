#!/usr/bin/env bash
set -euo pipefail

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
assert_contains "$out" "[quality-test] thinking max tokens: 4096"
args="$(cat "$tmp_log")"
assert_contains "$args" "--enable-thinking"
assert_contains "$args" "--thinking-max-tokens 4096"

: > "$tmp_log"
out="$(PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model bash scripts/quality-test.sh --quick 2>&1)"
assert_contains "$out" "WARN: server appears to have reasoning enabled"
args="$(cat "$tmp_log")"
if [[ "$args" == *"--enable-thinking"* ]]; then
  echo "ASSERTION FAILED: quality-test forwarded --enable-thinking while ENABLE_THINKING=0" >&2
  echo "$args" >&2
  exit 1
fi

echo "test-quality-thinking: ok"
