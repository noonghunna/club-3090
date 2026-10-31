#!/usr/bin/env bash
#
# Guard: verify-full's step 9 checks a llama.cpp drafter instead of skipping it.
#
# Why this exists: until 2026-10-15 the llama.cpp arm of check_mtp_acceptance was a
# bare `skip` ("vLLM-log-format-specific"), so on every llama.cpp slug a dead drafter
# passed verify-full. Speculative decoding rejects bad drafts, so the output stays
# correct and only decode slows: nothing else in verify-full can see it.
#
# This drives the REAL functions (scripts/lib/llamacpp-spec.sh, and step 9 extracted
# from scripts/verify-full.sh), with docker,
# curl and the pass/fail/skip helpers stubbed. The fixtures are the shapes measured on
# the reference rig: a wrapper shell whose script text contains "-md path." (the
# moe-cache Flash-Next compose), llama-server's own command line, and the per-request
# "draft acceptance = …" lines from mainline, PrismML and the moe-cache fork.
#
# ⚠️ Against the pre-fix tree this test must FAIL (the helpers don't exist and the
# llama.cpp arm skips). If it passes there, it is asserting the wrong thing.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VF="${VERIFY_FULL:-${ROOT}/scripts/verify-full.sh}"
LIB="${LLAMACPP_SPEC_LIB:-${ROOT}/scripts/lib/llamacpp-spec.sh}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
FAIL=0

ok()  { printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad() { printf "  \033[31m✗\033[0m %s\n" "$1"; printf "    expected=%s\n    got=%s\n" "$2" "$3"; FAIL=1; }
check() { [[ "$2" == "$3" ]] && ok "$1" || bad "$1" "$2" "$3"; }

# --- the functions under test: the shared lib + verify-full's step 9 -------------
# The helpers live in scripts/lib/llamacpp-spec.sh (shared with health.sh and report.sh);
# verify-full must source it, or step 9 calls functions that don't exist.
if ! command grep -qE '^source .*scripts/lib/llamacpp-spec\.sh' "$VF"; then
  bad "verify-full.sh sources scripts/lib/llamacpp-spec.sh" "a source line" "none"
fi
{ cat "$LIB" 2>/dev/null; awk '/^check_mtp_acceptance\(\) \{/,/^\}/' "$VF"; } > "${TMP}/fns.sh"
for fn in llamacpp_spec_args llamacpp_draft_rate check_mtp_acceptance; do
  if ! command grep -q "^${fn}() {" "${TMP}/fns.sh"; then
    bad "verify-full.sh defines ${fn}()" "a function body" "not found"
  fi
done
[[ "$FAIL" == 0 ]] || { echo "FAIL: test-verify-full-llamacpp-spec"; exit 1; }

# --- fixtures ----------------------------------------------------------------
# docker top -eo pid,args: a wrapper shell (its script text mentions -md) + the server
cat > "${TMP}/top-ngram" <<'EOF'
PID                 COMMAND
101                 sh -c set -e; # pass -md path. only when a drafter is set; exec /app/llama-server "$@"
102                 /app/llama-server --host 0.0.0.0 --port 8080 -m /models/x.gguf -c 204800 --spec-type ngram-mod --spec-ngram-mod-n-max 8 --jinja
EOF
cat > "${TMP}/top-mtp" <<'EOF'
PID                 COMMAND
201                 /app/llama-server --host 0.0.0.0 --port 8080 -m /models/x.gguf --spec-type draft-mtp --spec-draft-n-max 2 --jinja
EOF
cat > "${TMP}/top-md" <<'EOF'
PID                 COMMAND
301                 llama-server -m /models/target.gguf -md /models/draft.gguf --spec-draft-n-max 3
EOF
cat > "${TMP}/top-none" <<'EOF'
PID                 COMMAND
401                 sh -c # SPEC_N=0 removes --spec-type and -md; exec llama-server "$@"
402                 /app/llama-server --host 0.0.0.0 --port 8080 -m /models/x.gguf --jinja
EOF
cat > "${TMP}/top-specnone" <<'EOF'
PID                 COMMAND
501                 /app/llama-server -m /models/x.gguf --spec-type none
EOF
: > "${TMP}/log-empty"
printf '%s\n' "slot print_timing: id  0 | task 7 | draft acceptance = 0.95833 (  184 accepted /   192 generated)" > "${TMP}/log-good"
printf '%s\n' "slot print_timing: id  0 | task 9 | draft acceptance = 0.05000 (   10 accepted /   200 generated), mean len =  1.10" > "${TMP}/log-bad"
# #1857: an n-gram drafter's first request after a boot drafted ONE token, rejected — a 0/1 sample
printf '%s\n' "slot print_timing: id  0 | task 3 | draft acceptance = 0.00000 (    0 accepted /     1 generated), mean len =  1.00" > "${TMP}/log-tiny"

# --- 1: llamacpp_spec_args -----------------------------------------------------
source "${TMP}/fns.sh"
spec_args() { docker() { cat "$TOP"; }; TOP="$1" llamacpp_spec_args c; unset -f docker; }
check "spec args: n-gram server, wrapper's '-md path.' ignored" "--spec-type ngram-mod" "$(spec_args "${TMP}/top-ngram")"
check "spec args: MTP" "--spec-type draft-mtp" "$(spec_args "${TMP}/top-mtp")"
check "spec args: a draft model via -md" "-md /models/draft.gguf" "$(spec_args "${TMP}/top-md")"
check "spec args: no drafter (wrapper text mentions both flags)" "" "$(spec_args "${TMP}/top-none")"
check "spec args: --spec-type none is not a drafter" "" "$(spec_args "${TMP}/top-specnone")"

# --- 2: llamacpp_draft_rate ------------------------------------------------------
check "draft rate: one mainline line" "0.958 184 192 1" "$(llamacpp_draft_rate < "${TMP}/log-good")"
check "draft rate: moe-cache line with mean len" "0.050 10 200 1" "$(llamacpp_draft_rate < "${TMP}/log-bad")"
check "draft rate: lines summed, not averaged" "0.471 1457 3094 2" "$(printf '%s\n' \
  "draft acceptance = 0.95833 (  184 accepted /   192 generated)" \
  "draft acceptance = 0.43866 ( 1273 accepted /  2902 generated), mean len =  2.75" | llamacpp_draft_rate)"
check "draft rate: no lines -> nothing" "" "$(llamacpp_draft_rate < "${TMP}/log-empty")"

# --- 3: the step-9 decision, end to end ---------------------------------------------
# run_step <top fixture> <log fixture for request 1> <log fixture for request 2>
# prints "<verdict> rc=<n> requests=<n>"
run_step() {
  ( set +e
    ENGINE_KIND=llamacpp CONTAINER=c URL=http://127.0.0.1:1 MODEL=m
    THINK_OFF_STD='' THINK_OFF_KW='{"enable_thinking": false}'
    pass() { echo "PASS $1"; }; fail() { echo "FAIL $1"; }; skip() { echo "SKIP $1"; }
    container_is_real() { return 0; }
    sleep() { :; }
    curl() { echo x >> "${TMP}/reqs"; return 0; }
    docker() {
      case "$1" in
        top)  cat "$TOP_F" ;;
        logs) local n; n="$(wc -l < "${TMP}/reqs")"; if [[ "$n" -le 1 ]]; then cat "$LOG1"; else cat "$LOG2"; fi ;;
      esac
    }
    : > "${TMP}/reqs"
    TOP_F="$1" LOG1="$2" LOG2="$3"
    out="$(check_mtp_acceptance 2>&1)"; rc=$?
    v="$(printf '%s\n' "$out" | command grep -oE '^(PASS|FAIL|SKIP)' | head -1)"
    echo "${v:-NONE} rc=${rc} requests=$(wc -l < "${TMP}/reqs")"
  )
}
check "step 9: healthy MTP drafter -> PASS on one request" \
  "PASS rc=0 requests=1" "$(run_step "${TMP}/top-mtp" "${TMP}/log-good" "${TMP}/log-good")"
check "step 9: drafter rejected (5%) -> FAIL" \
  "FAIL rc=1 requests=1" "$(run_step "${TMP}/top-mtp" "${TMP}/log-bad" "${TMP}/log-bad")"
check "step 9: n-gram drafts nothing on request 1, then drafts -> PASS on request 2" \
  "PASS rc=0 requests=2" "$(run_step "${TMP}/top-ngram" "${TMP}/log-empty" "${TMP}/log-good")"
check "step 9: drafter configured but never drafts -> FAIL after 2 requests (the dead-drafter case)" \
  "FAIL rc=1 requests=2" "$(run_step "${TMP}/top-mtp" "${TMP}/log-empty" "${TMP}/log-empty")"
check "step 9: no drafter and no draft lines -> SKIP on one request" \
  "SKIP rc=0 requests=1" "$(run_step "${TMP}/top-none" "${TMP}/log-empty" "${TMP}/log-empty")"
check "step 9: LLAMACPP_ACCEPT_MIN raises the floor -> FAIL" \
  "FAIL rc=1 requests=1" "$(LLAMACPP_ACCEPT_MIN=0.99 run_step "${TMP}/top-mtp" "${TMP}/log-good" "${TMP}/log-good")"
# #1857: a sample too small to judge is no verdict — ask again, judge the second request.
check "step 9: n-gram drafts 1 token (rejected) on request 1, then drafts -> PASS on request 2 (#1857)" \
  "PASS rc=0 requests=2" "$(run_step "${TMP}/top-ngram" "${TMP}/log-tiny" "${TMP}/log-good")"
check "step 9: still too few drafts on request 2 -> FAIL (too few to judge), not a 0/1 verdict" \
  "FAIL rc=1 requests=2" "$(run_step "${TMP}/top-ngram" "${TMP}/log-tiny" "${TMP}/log-tiny")"
got="$( set +e
  ENGINE_KIND=llamacpp CONTAINER=c URL=http://127.0.0.1:1 MODEL=m THINK_OFF_STD='' THINK_OFF_KW='{}'
  pass() { echo "PASS $1"; }; fail() { echo "FAIL $1"; }; skip() { echo "SKIP $1"; }
  container_is_real() { return 0; }; sleep() { :; }; curl() { return 0; }
  docker() { case "$1" in top) cat "${TMP}/top-ngram" ;; logs) cat "${TMP}/log-tiny" ;; esac; }
  check_mtp_acceptance 2>&1 )"
[[ "$got" == *"too few to judge"* ]] && ok "step 9: the too-few FAIL says so (not 'acceptance 0.000 < 0.30')" \
  || bad "step 9: the too-few FAIL says so" "*too few to judge*" "$got"
check "step 9: LLAMACPP_DRAFT_MIN_SAMPLE=1 judges the one-token sample -> FAIL on request 1" \
  "FAIL rc=1 requests=1" "$(LLAMACPP_DRAFT_MIN_SAMPLE=1 run_step "${TMP}/top-ngram" "${TMP}/log-tiny" "${TMP}/log-good")"
check "step 9: an invalid LLAMACPP_DRAFT_MIN_SAMPLE -> FAIL before any request" \
  "FAIL rc=1 requests=0" "$(LLAMACPP_DRAFT_MIN_SAMPLE=x run_step "${TMP}/top-ngram" "${TMP}/log-good" "${TMP}/log-good")"

if [[ "$FAIL" == 0 ]]; then
  echo "PASS: test-verify-full-llamacpp-spec"
else
  echo "FAIL: test-verify-full-llamacpp-spec"; exit 1
fi
