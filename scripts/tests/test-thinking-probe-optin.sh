#!/usr/bin/env bash
# Guard: every FUNCTIONAL check that grades response CONTENT must opt into the
# thinking effort-ladder probe (THINK_PROBE=1) before calling the detector.
#
# Without it the detector stops at the template SCAN, which proves only that a key
# is NAMED. For the GLM-5.3 family the scan's off-value `reasoning_effort: "none"`
# is not a valid level -- the template accepts only low|high and coerces anything
# else to MAX -- so "thinking off" becomes the strongest possible thinking request.
# The model then spends its whole budget reasoning and returns empty content, and
# every content-grading check reads that as a model failure. verify-stress.sh
# shipped without the opt-in and failed two community 4x3090 reports that way
# (#1128, #1129), scoring a healthy rig as broken.
#
# MEASUREMENT scripts (bench*, soak, concurrency) are deliberately NOT listed:
# the probe fires real requests and they must not perturb their own numbers.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
rc=0
for f in scripts/verify-full.sh scripts/verify.sh scripts/verify-stress.sh; do
  [[ -f "$f" ]] || { echo "  FAIL $f missing"; rc=1; continue; }
  if ! command grep -q 'preflight_detect_thinking_control' "$f"; then
    echo "  skip $f (does not use the detector)"; continue
  fi
  # THINK_PROBE=1 must appear BEFORE the detector call, not merely somewhere.
  probe_ln=$(command grep -n '^\s*THINK_PROBE=1' "$f" | head -1 | cut -d: -f1)
  call_ln=$(command grep -n '^\s*preflight_detect_thinking_control' "$f" | head -1 | cut -d: -f1)
  if [[ -z "$probe_ln" ]]; then
    echo "  FAIL $f calls the detector without THINK_PROBE=1 -> ladder skipped, invalid off-value used"
    rc=1
  elif [[ -z "$call_ln" || "$probe_ln" -ge "$call_ln" ]]; then
    echo "  FAIL $f sets THINK_PROBE=1 at line $probe_ln, at/after the call on $call_ln"
    rc=1
  else
    echo "  ok   $f (THINK_PROBE=1 on $probe_ln, detector on $call_ln)"
  fi
done
[[ "$rc" == "0" ]] && echo "PASS: content-grading checks opt into the effort ladder" || echo "FAIL: thinking-probe opt-in regression"
exit "$rc"
