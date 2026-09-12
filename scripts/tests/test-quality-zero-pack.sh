#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-quality-zero-pack — guards the #1269 / #1270 contract:
#
#   1. #1270 STRUCTURAL-ZERO GUARD (cause-agnostic, post-hoc): when ANY pack
#      comes back 0/N, quality-test.sh must NOT leave the bare TOTAL standing.
#      It prints both figures (valid subset + all packs), marks the all-packs
#      one "do not cite", and refuses to publish the per-rig quality record.
#      Trigger is the 0/N; latency is printed as corroboration only.
#   2. It must NOT fire on a pack that merely FAILED scenarios (verifier_fail
#      rows are the MODEL being wrong and keep counting), nor on a pack with
#      total == 0 (sandbox-unavailable / stubbed = never ran, not a zero score).
#   3. #1269: --no-thinking over a pack set containing hermesagent-20 prints a
#      loud, specific up-front notice naming the mechanism AND the reduced
#      denominator — and does NOT print it for a pack set without that pack.
#
# NEGATIVE CONTROLS (each of these FAILS against pre-#1269/#1270 quality-test.sh):
#   - case A: the guard block + both TOTAL figures + the record refusal
#   - case D: the every-pack-zeroed wording
#   - case E: cause attribution that does NOT blame thinking when thinking was
#             not forced off
#   - case F/J: the #1269 up-front notice
# And the false-positive controls (B, C, G, H, I) assert the guard/notice stay
# silent, so a guard that fires unconditionally cannot pass either.
#
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FAILED=0

assert_contains() {
  local label="$1" haystack="$2" needle="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED [$label]: expected output to contain:" >&2
    echo "  $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    FAILED=1
  fi
}

assert_not_contains() {
  local label="$1" haystack="$2" needle="$3"
  if [[ "$haystack" == *"$needle"* ]]; then
    echo "ASSERTION FAILED [$label]: expected output NOT to contain:" >&2
    echo "  $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    FAILED=1
  fi
}

tmp_bin="$(mktemp -d)"
tmp_work="$(mktemp -d)"
record_log="${tmp_work}/record-calls.log"
run_out="${tmp_work}/run.out"
before_list="$(mktemp)"
after_list="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before_list" || true
cleanup() {
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$after_list" || true
  comm -13 "$before_list" "$after_list" | xargs -r rm -f
  rm -rf "$tmp_bin" "$tmp_work"
  rm -f "$before_list" "$after_list"
}
trap cleanup EXIT

REAL_PYTHON3="$(command -v python3)"

cat > "${tmp_bin}/curl" <<'MOCK_CURL'
#!/usr/bin/env bash
for arg in "$@"; do
  case "$arg" in
    */v1/models) printf '{"data":[{"id":"mock-model"}]}'; exit 0 ;;
    */props|*/get_model_info) exit 1 ;;
  esac
done
exit 0
MOCK_CURL
chmod +x "${tmp_bin}/curl"

# benchlocal-cli mock: copies the fixture named by MOCK_RESULT_JSON to whatever
# --save-json path the wrapper chose. That is the real delivery path — the
# wrapper reads the same file a live run would.
cat > "${tmp_bin}/benchlocal-cli" <<'MOCK_BENCHLOCAL'
#!/usr/bin/env bash
json_out=""
argv=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --save-json) json_out="${2:-}"; shift 2 ;;
    list) echo 'toolcall-15'; exit 0 ;;
    --help) echo '--reasoning-effort'; exit 0 ;;
    *) shift ;;
  esac
done
printf '%s\n' "${argv[*]}" >> "${BENCHLOCAL_MOCK_LOG}"
if [[ -n "$json_out" && -n "${MOCK_RESULT_JSON:-}" ]]; then
  mkdir -p "$(dirname "$json_out")"
  cp "$MOCK_RESULT_JSON" "$json_out"
fi
echo "TOTAL (mock benchlocal scoreboard)"
exit 0
MOCK_BENCHLOCAL
chmod +x "${tmp_bin}/benchlocal-cli"

# python3 shim: records measurement_record.py invocations (the per-rig quality
# record IS a bare TOTAL, so the guard must stop it reaching the corpus) and
# forwards every other python3 call to the real interpreter untouched.
cat > "${tmp_bin}/python3" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == *measurement_record.py ]]; then
  printf '%s\n' "MEASUREMENT_RECORD \$*" >> "\${RECORD_MOCK_LOG}"
  exit 0
fi
exec "${REAL_PYTHON3}" "\$@"
EOF
chmod +x "${tmp_bin}/python3"

# docker mock: only the sandbox-image preflight calls docker in these runs
# (the hermes container-reachability probe needs a loopback URL, and URL=http://mock
# is not one). Report the three sandbox images present and freshly built so
# --full does not degrade to "sandbox packs will be SKIPPED".
cat > "${tmp_bin}/docker" <<'MOCK_DOCKER'
#!/usr/bin/env bash
if [[ "${1:-}" == "image" && "${2:-}" == "inspect" ]]; then
  case "${3:-}" in
    benchlocal-sandbox-*:latest)
      if [[ "${4:-}" == "--format" ]]; then
        date -u -d '+1 day' '+%Y-%m-%dT%H:%M:%S.000000000Z'
      fi
      exit 0
      ;;
  esac
  exit 1
fi
exit 1
MOCK_DOCKER
chmod +x "${tmp_bin}/docker"

# ---- fixtures ----------------------------------------------------------------
# @paulp83's #1253 shape: hermesagent-20 structurally zeroed at a p50 an order of
# magnitude below its healthy figure; the other 7 packs sum to 102/130.
pack_row() { # pack_row <id> <passed> <total> <p50> [status] [extra-json]
  printf '{"pack_id":"%s","passed":%s,"total":%s,"scenario_count":%s,"score":%s,"skipped":false,"status":"%s","latency":{"mean":%s,"p50":%s,"p95":%s}%s}' \
    "$1" "$2" "$3" "$3" "$(awk -v p="$2" -v t="$3" 'BEGIN{printf (t?p/t:0)}')" "${5:-ok}" "$4" "$4" "$4" "${6:-}"
}

write_result() { # write_result <path> <thinking_mode|-> <packs-json>
  local path="$1" tmode="$2" packs="$3" tm_field=""
  [[ "$tmode" != "-" ]] && tm_field=",\"thinking_mode\":\"${tmode}\""
  cat > "$path" <<JSON
{"schema_version":1,"runner_version":"0.9.9","endpoint":"http://mock","model":"mock-model",
 "mode":"full","started_at":"2026-09-12T00:00:00Z","finished_at":"2026-09-12T00:30:00Z",
 "thinking_enabled":false${tm_field},"warnings":[],"packs":[${packs}]}
JSON
}

SEVEN_VALID="$(pack_row toolcall-15 15 15 0.79),$(pack_row instructfollow-15 13 15 1.33),$(pack_row structoutput-15 15 15 2.07),$(pack_row dataextract-15 12 15 3.09),$(pack_row reasonmath-15 12 15 6.29),$(pack_row bugfind-15 11 15 5.66),$(pack_row cli-40 24 40 2.38)"

write_result "${tmp_work}/paul-off.json" force-off \
  "${SEVEN_VALID},$(pack_row hermesagent-20 0 20 2.40)"
write_result "${tmp_work}/paul-healthy.json" force-off \
  "${SEVEN_VALID},$(pack_row hermesagent-20 3 20 10.36)"
write_result "${tmp_work}/skipped-sandbox.json" force-off \
  "$(pack_row toolcall-15 15 15 0.79),{\"pack_id\":\"hermesagent-20\",\"passed\":0,\"total\":0,\"scenario_count\":20,\"score\":0.0,\"skipped\":true,\"status\":\"sandbox-unavailable\",\"latency\":null}"
write_result "${tmp_work}/all-zero.json" force-off \
  "$(pack_row toolcall-15 0 15 0.02),$(pack_row instructfollow-15 0 15 0.02)"
write_result "${tmp_work}/pack-defaults.json" pack-defaults \
  "${SEVEN_VALID},$(pack_row hermesagent-20 0 20 0.04)"
write_result "${tmp_work}/no-tmode.json" - \
  "${SEVEN_VALID},$(pack_row hermesagent-20 0 20 2.40)"

# run_wrapper <fixture> [KEY=VAL ...] -- [wrapper args ...]
run_wrapper() {
  local fixture="$1"; shift
  local extra_env=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do extra_env+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift
  : > "$record_log"
  : > "${tmp_work}/benchlocal-args.log"
  set +e
  env PATH="${tmp_bin}:$PATH" \
      BENCHLOCAL_MOCK_LOG="${tmp_work}/benchlocal-args.log" \
      RECORD_MOCK_LOG="$record_log" \
      MOCK_RESULT_JSON="$fixture" \
      PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
      ${extra_env[@]+"${extra_env[@]}"} \
      bash scripts/quality-test.sh "$@" > "$run_out" 2>&1
  local rc=$?
  set -e
  RUN_RC=$rc
  OUT="$(cat "$run_out")"
}

echo "--- case A: #1270 fires on a 0/N pack and refuses the bare TOTAL ---"
run_wrapper "${tmp_work}/paul-off.json" -- --quick
assert_contains A "$OUT" "STRUCTURAL-ZERO GUARD (#1270)"
assert_contains A "$OUT" "hermesagent-20 scored 0/20"
assert_contains A "$OUT" "p50 2.40s"
assert_contains A "$OUT" "TOTAL (valid subset)  102 / 130 (78%)"
assert_contains A "$OUT" "TOTAL (all packs)     102 / 150 (68%)   <- do not cite"
assert_contains A "$OUT" "verifier_fail rows are the MODEL being wrong and DO keep counting"
# thinking_mode: force-off + a thinking-default-on pack -> name #1269 as the first suspect
assert_contains A "$OUT" "check FIRST: thinking was forced OFF"
assert_contains A "$OUT" "(#1269)"
# the per-rig record is a bare TOTAL too: it must not be published
assert_contains A "$OUT" "per-rig quality record NOT written"
assert_not_contains A "$(cat "$record_log")" "MEASUREMENT_RECORD"

echo "--- case B: no 0/N pack -> guard silent, record published (false-positive control) ---"
run_wrapper "${tmp_work}/paul-healthy.json" -- --quick
assert_not_contains B "$OUT" "STRUCTURAL-ZERO GUARD"
assert_not_contains B "$OUT" "do not cite"
assert_not_contains B "$OUT" "per-rig quality record NOT written"
assert_contains B "$(cat "$record_log")" "MEASUREMENT_RECORD"
assert_contains B "$(cat "$record_log")" "--quality-8pk 105/150"

echo "--- case C: total==0 (sandbox-unavailable) is a SKIP, not a zero score ---"
run_wrapper "${tmp_work}/skipped-sandbox.json" -- --quick
assert_not_contains C "$OUT" "STRUCTURAL-ZERO GUARD"
assert_contains C "$(cat "$record_log")" "MEASUREMENT_RECORD"

echo "--- case D: every scored pack zeroed -> no valid subset to cite ---"
run_wrapper "${tmp_work}/all-zero.json" -- --quick
assert_contains D "$OUT" "STRUCTURAL-ZERO GUARD (#1270)"
assert_contains D "$OUT" "TOTAL (valid subset)  none — every scored pack returned 0/N"
assert_contains D "$OUT" "TOTAL (all packs)     0 / 30 (0%)   <- do not cite"

echo "--- case E: thinking NOT forced off -> do not blame thinking ---"
run_wrapper "${tmp_work}/pack-defaults.json" -- --quick
assert_contains E "$OUT" "STRUCTURAL-ZERO GUARD (#1270)"
assert_not_contains E "$OUT" "check FIRST: thinking was forced OFF"
assert_contains E "$OUT" "cause not determined here"
assert_contains E "$OUT" "endpoint reachable from a container"

echo "--- case E2: JSON without thinking_mode falls back to the wrapper's NO_THINKING ---"
run_wrapper "${tmp_work}/no-tmode.json" NO_THINKING=1 -- --quick
assert_contains E2 "$OUT" "check FIRST: thinking was forced OFF"

echo "--- case F: #1269 up-front notice on --full --no-thinking ---"
run_wrapper "${tmp_work}/paul-off.json" -- --full --no-thinking
assert_contains F "$OUT" "--no-thinking + hermesagent-20 is a KNOWN-RISKY combination (#1269)"
assert_contains F "$OUT" "NOT ATTEMPTING"
assert_contains F "$OUT" "denominator drops by its 20 scenarios (--full: 150 → 130)"
assert_contains F "$OUT" "instructfollow-15 / reasonmath-15 /"
assert_contains F "$OUT" "--no-sandboxed, or --medium"
assert_not_contains F "$OUT" "sandbox packs (BugFind / CLI / Hermes) will be SKIPPED"

echo "--- case G: --medium --no-thinking has no hermesagent-20 -> no notice ---"
run_wrapper "${tmp_work}/paul-healthy.json" -- --medium --no-thinking
assert_contains G "$OUT" "thinking: disabled for every pack"
assert_not_contains G "$OUT" "KNOWN-RISKY combination (#1269)"

echo "--- case H: --full WITHOUT --no-thinking -> no notice ---"
run_wrapper "${tmp_work}/paul-healthy.json" -- --full
assert_not_contains H "$OUT" "KNOWN-RISKY combination (#1269)"

echo "--- case I: --full --no-thinking --no-sandboxed drops hermes -> no notice ---"
run_wrapper "${tmp_work}/paul-healthy.json" -- --full --no-thinking --no-sandboxed
assert_contains I "$OUT" "thinking: disabled for every pack"
assert_not_contains I "$OUT" "KNOWN-RISKY combination (#1269)"

echo "--- case J: --pack hermesagent-20 --no-thinking -> notice ---"
run_wrapper "${tmp_work}/paul-off.json" -- --pack hermesagent-20 --no-thinking
assert_contains J "$OUT" "KNOWN-RISKY combination (#1269)"

echo "--- case K: an unreadable results JSON must say so, not pass silently ---"
: > "${tmp_work}/truncated.json"
printf '{"packs":[{"pack_id":"toolcall-15",' > "${tmp_work}/truncated.json"
run_wrapper "${tmp_work}/truncated.json" -- --quick
assert_contains K "$OUT" "structural-zero guard could not read"
assert_contains K "$OUT" "has NOT been checked for 0/N packs"

if [[ "$FAILED" != "0" ]]; then
  echo "FAIL: test-quality-zero-pack" >&2
  exit 1
fi
echo "PASS: test-quality-zero-pack (#1269 notice + #1270 structural-zero guard)"
