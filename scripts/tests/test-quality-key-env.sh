#!/usr/bin/env bash
# test-quality-key-env — quality-test.sh never puts the bearer key on a command line.
#
# Why this exists: the wrapper passed the key as `benchlocal-cli --api-key <key>` and as
# `curl -H "Authorization: Bearer <key>"`, where any user on the host reads it in the process list
# (/proc/<pid>/cmdline). Since #1883 the key can be the serving engine's own (a keyed Strata), not
# only a cloud key someone typed. It now reaches benchlocal-cli as BENCHLOCAL_API_KEY in its
# environment, and the probe's header comes from a process substitution, as engine-auth.sh sends it.
#
# Drives the REAL wrapper with curl and benchlocal-cli stubbed: each records its argv (and the stub
# curl the header file it was handed), so the assertions are on what the processes received.
# ⚠️ Against the pre-fix tree this must FAIL.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail() { echo "FAIL: test-quality-key-env — $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }
tmp="$(mktemp -d)"
before="$(mktemp)"; after="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before" || true
cleanup() {
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$after" || true
  comm -13 "$before" "$after" | xargs -r rm -f
  rm -rf "$tmp"; rm -f "$before" "$after"
}
trap cleanup EXIT
KEY="k-quality-test-secret-0001"
bin="$tmp/bin"; mkdir -p "$bin"

# curl: logs its argv; for `-H @file` it logs the file's content too. /v1/models answers a model list.
cat > "$bin/curl" <<'EOF'
#!/usr/bin/env bash
printf 'ARGV %s\n' "$*" >> "$STUB_LOG"
prev=""
for a in "$@"; do
  if [[ "$prev" == -H && "$a" == @* ]]; then printf 'HDRFILE %s\n' "$(cat "${a#@}" 2>/dev/null)" >> "$STUB_LOG"; fi
  prev="$a"
done
for a in "$@"; do case "$a" in */v1/models) printf '{"data":[{"id":"mock-model"}]}'; exit 0 ;; */props|*/get_model_info|*/v1/model) exit 22 ;; esac; done
exit 0
EOF
# benchlocal-cli: logs argv and the BENCHLOCAL_API_KEY it inherited; writes a minimal results JSON.
cat > "$bin/benchlocal-cli" <<'EOF'
#!/usr/bin/env bash
json_out=""
for ((i=1; i<=$#; i++)); do
  case "${!i}" in list) echo toolcall-15; exit 0 ;; --help) echo '--reasoning-effort'; exit 0 ;; esac
  if [[ "${!i}" == "--save-json" ]]; then j=$((i+1)); json_out="${!j}"; fi
done
printf 'BL_ARGV %s\n' "$*" >> "$STUB_LOG"
printf 'BL_ENV %s\n' "${BENCHLOCAL_API_KEY:-<unset>}" >> "$STUB_LOG"
if [[ -n "$json_out" ]]; then
  mkdir -p "$(dirname "$json_out")"
  echo '{"packs":[{"pack_id":"toolcall-15","status":"ok","passed":13,"total":15,"score":0.8667}]}' > "$json_out"
fi
echo "TOTAL (mock)"
EOF
printf '#!/usr/bin/env bash\nexit 1\n' > "$bin/docker"
chmod +x "$bin"/{curl,benchlocal-cli,docker}

: > "$tmp/log"
set +e
out="$(env PATH="$bin:$PATH" STUB_LOG="$tmp/log" PREFLIGHT_NO_AUTODETECT=1 URL="http://127.0.0.1:18080" MODEL=mock-model \
       API_KEY="$KEY" bash scripts/quality-test.sh --pack toolcall-15 --no-thinking --no-progress 2>&1)"; rc=$?
set -e
command grep -q '^BL_ARGV' "$tmp/log" || { echo "$out" | tail -15 >&2; fail "benchlocal-cli never ran (rc=$rc) — every assertion below would be vacuous"; }
! command grep -qF -- "$KEY" <(command grep -E '^(ARGV|BL_ARGV) ' "$tmp/log") \
  || { command grep -E '^(ARGV|BL_ARGV) ' "$tmp/log" | sed "s/$KEY/<KEY>/g" >&2; fail "the key is on a command line"; }
pass "the key is on no command line: neither curl's nor benchlocal-cli's"
command grep -qxF "BL_ENV $KEY" "$tmp/log" || fail "benchlocal-cli did not get the key as BENCHLOCAL_API_KEY"
pass "benchlocal-cli gets it as BENCHLOCAL_API_KEY in its environment"
command grep -qxF "HDRFILE Authorization: Bearer $KEY" "$tmp/log" || fail "the reachability probe did not send the key as a header"
pass "the reachability probe still sends it, as a header read from a file descriptor"

echo "test-quality-key-env: ok"
