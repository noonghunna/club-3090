#!/usr/bin/env bash
#
# Guard: verify-full's step 9 skips a Project Maya slug that runs no drafter, and still checks one that does.
#
# Why this exists: Maya's /metrics reports the DSpark drafter as engine.dspark, but its two engines spell "off"
# differently: MiniMax's maya-mm3 writes 0 (DSPARK=off), DeepSeek's maya-ds4 writes false (a pack with no
# drafter: strata-maya/deepseek41-flash-dual-ud-iq3-s). Step 9 matched only the 0, so the drafter-less DeepSeek
# slug failed verify-full with "no MTP drafts offered", a correct engine read as a broken one (2026-11-02).
#
# Hy4-Flash's maya-hy4 (Maya v1.1.68) reports engine.dspark 0 beside its MTP block, engine.mtp 1 (engine.speculation
# ["MTP"]): a skip on dspark 0 alone skipped a drafter that runs, and every Hy4 slug drafts (2026-11-20). With
# DRAFTING=off it reports mtp 0, which skips. Against the pre-fix verify-full the MTP-on case must FAIL.
#
# This drives the REAL check_mtp_acceptance (extracted from scripts/verify-full.sh) with ENGINE_KIND=strata, a
# stub strata_metrics.py that prints a /metrics snapshot, and curl / pass / fail / skip stubbed.
#
# ⚠️ Against the pre-fix verify-full this test must FAIL on the `false` case. If it passes there, it is asserting
# the wrong thing.
#
# A host-run Maya (CONTAINER=none, #1915): its drafts are only in its engine log, so with neither the container nor
# SERVER_LOG the check would read zero drafts and fail a working drafter. It skips then, naming SERVER_LOG, and runs
# with SERVER_LOG set.
#
# The trigger request's own response (#2053): Maya returns that request's draft counts in it under llama.cpp's names
# (timings.draft_n / draft_n_accepted), and the check reads them first. A host run with no log and a /metrics that
# names no maya_version used to read zero drafts and FAIL a working drafter (174 drafted, 143 kept); it passes from
# the timings now, a low rate or zero drafts there still fail, and a response without timings falls back to the log
# path above (so the SERVER_LOG skip still applies). Against the pre-fix verify-full the timings cases must FAIL.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VF="${VERIFY_FULL:-${ROOT}/scripts/verify-full.sh}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
FAIL=0
ok()  { printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad() { printf "  \033[31m✗\033[0m %s\n" "$1"; printf "    expected=%s\n    got=%s\n" "$2" "$3"; FAIL=1; }

awk '/^check_mtp_acceptance\(\) \{/,/^\}/' "$VF" > "${TMP}/fns.sh"
command grep -q '^check_mtp_acceptance() {' "${TMP}/fns.sh" \
  || { bad "verify-full.sh defines check_mtp_acceptance()" "a function body" "not found"; exit 1; }

# a stand-in for scripts/lib/strata_metrics.py: `snapshot` prints the fixture named by $SNAP
mkdir -p "${TMP}/root/scripts/lib"
cat > "${TMP}/root/scripts/lib/strata_metrics.py" <<'PY'
import os, sys
if sys.argv[1] == "snapshot":
    print(open(os.environ["SNAP"]).read())
    sys.exit(0)
if sys.argv[1] == "delta" and os.environ.get("DELTA"):      # the /metrics delta, when a case gives one
    print(os.environ["DELTA"].replace("\\n", "\n"))
    sys.exit(0)
sys.exit(1)
PY
printf '{"engine": {"maya_version": "1.1.48", "dspark": false}}' > "${TMP}/ds4-nodrafter.json"
printf '{"engine": {"maya_version": "1.1.30", "dspark": 0}}' > "${TMP}/mm3-off.json"
printf '{"engine": {"maya_version": "1.1.48", "dspark": true}}' > "${TMP}/ds4-drafter.json"
printf '{"engine": {"version": "0.1.55"}}' > "${TMP}/strata.json"
printf '{"engine": {"maya_version": "1.1.68", "engine_kind": "hy4", "dspark": 0, "mtp": 1, "speculation": ["MTP"]}}' > "${TMP}/hy4-mtp.json"
printf '{"engine": {"maya_version": "1.1.68", "engine_kind": "hy4", "dspark": 0, "mtp": 0, "speculation": []}}' > "${TMP}/hy4-off.json"
printf '{"engine": {"dspark": true}}' > "${TMP}/ds4-noversion.json"      # #2053: /metrics named no maya_version
printf '{"choices": [], "timings": {"draft_n": 174, "draft_n_accepted": 143}}' > "${TMP}/resp-drafts.json"
printf '{"choices": [], "timings": {"draft_n": 100, "draft_n_accepted": 10}}' > "${TMP}/resp-low.json"
printf '{"choices": [], "timings": {"draft_n": 0, "draft_n_accepted": 0}}' > "${TMP}/resp-zero.json"
printf '{"choices": [], "timings": {"predicted_n": 80}}' > "${TMP}/resp-notimings.json"

# step 9 for one snapshot: prints which helper ended it. REAL=1: the container is real (a club-3090 slug);
# LOG=<file>: SERVER_LOG for a host run
run() {
  (
    source "${TMP}/fns.sh"
    pass() { echo "PASS: $1"; }
    fail() { echo "FAIL: $1"; }
    skip() { echo "SKIP: $1"; }
    container_is_real() { [[ "${REAL:-0}" == 1 ]]; }
    # the trigger request: the response fixture named by $RESP, else a failure (a drafter that is on gets its
    # trigger request, which fails here: not a skip)
    curl() { [[ -n "${RESP:-}" ]] && { cat "$RESP"; return 0; }; return 7; }
    ENGINE_KIND=strata ROOT_DIR="${TMP}/root" URL=http://stub MODEL=m THINK_OFF_STD= THINK_OFF_KW='{}' CONTAINER=c \
      SERVER_LOG="${LOG:-}" SNAP="$1" check_mtp_acceptance 2>&1 | command grep -E '^(PASS|FAIL|SKIP):' | head -1
  )
}

got="$(run "${TMP}/ds4-nodrafter.json")"
[[ "$got" == SKIP:*"no drafter"* ]] && ok "maya-ds4, engine.dspark false: skipped (no drafter in the pack)" \
  || bad "maya-ds4 dspark false" "SKIP: …no drafter…" "$got"
got="$(run "${TMP}/mm3-off.json")"
[[ "$got" == SKIP:*"no drafter"* ]] && ok "maya-mm3, engine.dspark 0: skipped (DSPARK=off)" \
  || bad "maya-mm3 dspark 0" "SKIP: …no drafter…" "$got"
got="$(REAL=1 run "${TMP}/ds4-drafter.json")"
[[ "$got" == FAIL:*"acceptance-trigger request failed"* ]] && ok "engine.dspark true, in its container: not skipped (the trigger request runs)" \
  || bad "maya-ds4 dspark true, container" "FAIL: acceptance-trigger request failed" "$got"
got="$(RESP="${TMP}/resp-notimings.json" run "${TMP}/ds4-drafter.json")"
[[ "$got" == SKIP:*SERVER_LOG* ]] && ok "engine.dspark true, host run without SERVER_LOG, no timings in the response: skipped, naming SERVER_LOG" \
  || bad "maya-ds4 dspark true, host, no log, no timings" "SKIP: …SERVER_LOG…" "$got"
: > "${TMP}/maya-ds4.log"
got="$(LOG="${TMP}/maya-ds4.log" run "${TMP}/ds4-drafter.json")"
[[ "$got" == FAIL:*"acceptance-trigger request failed"* ]] && ok "engine.dspark true, host run with SERVER_LOG: not skipped" \
  || bad "maya-ds4 dspark true, host, SERVER_LOG" "FAIL: acceptance-trigger request failed" "$got"
got="$(run "${TMP}/strata.json")"
[[ "$got" == FAIL:*"acceptance-trigger request failed"* ]] && ok "Strata, no engine.dspark field: not skipped" \
  || bad "strata, no dspark field" "FAIL: acceptance-trigger request failed" "$got"

got="$(REAL=1 run "${TMP}/hy4-mtp.json")"
[[ "$got" == FAIL:*"acceptance-trigger request failed"* ]] && ok "maya-hy4, engine.dspark 0 with engine.mtp 1: not skipped (its MTP block drafts)" \
  || bad "maya-hy4 dspark 0, mtp 1" "FAIL: acceptance-trigger request failed" "$got"
got="$(REAL=1 RESP="${TMP}/resp-drafts.json" run "${TMP}/hy4-mtp.json")"
[[ "$got" == PASS:*"timings"* ]] && ok "maya-hy4, MTP on: checked from the response's timings" \
  || bad "maya-hy4 mtp on, timings" "PASS: … from the response's timings" "$got"
got="$(REAL=1 run "${TMP}/hy4-off.json")"
[[ "$got" == SKIP:*"no draft block"* ]] && ok "maya-hy4, DRAFTING=off (engine.mtp 0): skipped" \
  || bad "maya-hy4 mtp 0" "SKIP: …no draft block…" "$got"


echo "the trigger response's timings (#2053)"
got="$(RESP="${TMP}/resp-drafts.json" run "${TMP}/ds4-noversion.json")"
[[ "$got" == PASS:*"0.8218"*"174 drafted"*"timings"* ]] && ok "host run, no log, /metrics without maya_version: passes from the response's timings (143 of 174)" \
  || bad "#2053 shape" "PASS: draft acceptance 0.8218 … 174 drafted, from the response's timings" "$got"
got="$(RESP="${TMP}/resp-drafts.json" run "${TMP}/ds4-drafter.json")"
[[ "$got" == PASS:*"timings"* ]] && ok "host run, no log, with maya_version: checked from the timings, not skipped" \
  || bad "host maya with timings" "PASS: … from the response's timings" "$got"
got="$(REAL=1 RESP="${TMP}/resp-low.json" run "${TMP}/ds4-drafter.json")"
[[ "$got" == FAIL:*"0.1000"*"< 0.30"* ]] && ok "a low rate in the timings still fails (10 of 100)" \
  || bad "low acceptance" "FAIL: draft acceptance 0.1000 < 0.30 …" "$got"
got="$(REAL=1 RESP="${TMP}/resp-zero.json" run "${TMP}/ds4-drafter.json")"
[[ "$got" == FAIL:*"draft_n=0"* ]] && ok "zero drafts in the timings fails" \
  || bad "zero drafts" "FAIL: no MTP drafts offered … draft_n=0" "$got"
got="$(REAL=1 RESP="${TMP}/resp-notimings.json" DELTA='drafts_offered=50\naccept_rate=0.8000' run "${TMP}/strata.json")"
[[ "$got" == PASS:*"0.8000"*"from /metrics"* ]] && ok "no timings in the response: falls back to the /metrics delta" \
  || bad "fallback to /metrics" "PASS: draft acceptance 0.8000 … from /metrics" "$got"

if [[ "$FAIL" == 0 ]]; then echo "PASS: test-verify-full-maya-drafter-skip"; else echo "FAIL: test-verify-full-maya-drafter-skip"; exit 1; fi
