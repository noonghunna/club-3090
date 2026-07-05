#!/usr/bin/env bash
# test-concurrency-probe — offline guards for scripts/concurrency-probe.sh (the
# #246 Phase 2 soak-validation tool). The live probe needs a running server;
# these check only what can be verified without one: syntax, the SWEEP-needs-SLUG
# refusal, and that SWEEP_DRY plans reboots WITHOUT calling switch.sh.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT_DIR/scripts/concurrency-probe.sh"
fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. syntax
bash -n "$PROBE" || fail "bash -n: syntax error"
echo "  ✓ syntax"

# 2. SWEEP without SLUG must refuse with exit 2 (a reboot target is mandatory —
#    vLLM can't hot-change max-num-seqs, so each N is a boot).
set +e
out="$(SWEEP="4 8" bash "$PROBE" 2>&1)"; rc=$?
set -e
[[ "$rc" == "2" ]] || fail "SWEEP without SLUG should exit 2, got $rc"
grep -q "SWEEP needs SLUG" <<<"$out" || fail "SWEEP-without-SLUG error message missing"
echo "  ✓ SWEEP refuses without SLUG (exit 2)"

# 3. SWEEP_DRY prints the plan for every N and must NOT boot the server (a dry
#    run never touches switch.sh). Assert on output: one [sweep:dry] line per N,
#    no [sweep] boot line, and the knee summary always prints.
out="$(SWEEP="4 8 12" SLUG=vllm/minimal SWEEP_DRY=1 bash "$PROBE" 2>&1)"
[[ "$(grep -c '\[sweep:dry\]' <<<"$out")" == "3" ]] || fail "SWEEP_DRY should print one plan line per N (3)"
grep -q '\[sweep\] boot' <<<"$out" && fail "SWEEP_DRY must not boot the server"
grep -q "sweep knee" <<<"$out" || fail "SWEEP should always print a knee summary"
echo "  ✓ SWEEP_DRY plans 3 reboots without booting"

echo "test-concurrency-probe: ok"
