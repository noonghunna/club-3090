#!/usr/bin/env bash
# test-spec-sweep — offline guards for scripts/spec-sweep.sh (the draft-depth
# n-sweep tool). The live sweep needs a running server; these check only what
# can be verified without one: syntax, the SWEEP_N refusal, engine resolution
# from the registry, the vLLM-needs-SLUG refusal, and that SWEEP_DRY plans
# arms WITHOUT booting or measuring.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP="$ROOT_DIR/scripts/spec-sweep.sh"
fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. syntax
bash -n "$SWEEP" || fail "bash -n: syntax error"
echo "  ✓ syntax"

# 2. SWEEP_N is mandatory (exit 2)
set +e
out="$(bash "$SWEEP" 2>&1)"; rc=$?
set -e
[[ "$rc" == "2" ]] || fail "no SWEEP_N should exit 2, got $rc"
grep -q "SWEEP_N is required" <<<"$out" || fail "SWEEP_N-required message missing"
echo "  ✓ refuses without SWEEP_N (exit 2)"

# 3. unknown slug refused (exit 2)
set +e
out="$(SWEEP_N="0 2" SLUG=vllm/definitely-not-real bash "$SWEEP" 2>&1)"; rc=$?
set -e
[[ "$rc" == "2" ]] || fail "unknown SLUG should exit 2, got $rc"
grep -q "unknown SLUG" <<<"$out" || fail "unknown-SLUG message missing"
echo "  ✓ refuses unknown SLUG (exit 2)"

# 4. SWEEP_DRY plans per-arm without booting/measuring — llama.cpp slug:
#    fast-path plan lines (per-request, no reboot), engine + URL resolved
#    from the registry (llamacpp/tess-dual-mtp -> :8020).
out="$(SWEEP_N="0 1 2" SLUG=llamacpp/tess-dual-mtp SWEEP_DRY=1 bash "$SWEEP" 2>&1)"
grep -q "engine=llamacpp" <<<"$out" || fail "llama.cpp slug should resolve engine=llamacpp"
grep -q "localhost:8020" <<<"$out" || fail "URL should derive from the slug's default_port (8020)"
[[ "$(grep -c 'sweep:dry' <<<"$out")" == "3" ]] || fail "SWEEP_DRY should print one plan line per n (3)"
grep -q "per-request speculative.n_max" <<<"$out" || fail "llama.cpp dry plan should be the per-request fast path"
grep -q "switch.sh" <<<"$out" && fail "llama.cpp SWEEP_DRY must not plan reboots"
echo "  ✓ llama.cpp SWEEP_DRY plans per-request arms, no reboots"

# 5. SWEEP_DRY — vLLM slug: reboot-per-arm plans, SPEC=off for the n=0 arm.
out="$(SWEEP_N="0 3" SLUG=vllm/dual SWEEP_DRY=1 bash "$SWEEP" 2>&1)"
grep -q "engine=vllm" <<<"$out" || fail "vllm/dual should resolve engine=vllm"
grep -q "SPEC=off switch.sh vllm/dual" <<<"$out" || fail "n=0 arm should plan SPEC=off"
grep -q "SPEC_N_MAX=3 switch.sh vllm/dual" <<<"$out" || fail "n=3 arm should plan SPEC_N_MAX=3"
echo "  ✓ vLLM SWEEP_DRY plans reboot-per-arm (SPEC=off baseline)"

# 6. vLLM without SLUG refused (exit 2) — can't reboot without a target.
#    (Engine defaults to llamacpp without a slug, so force via a vllm slug
#    minus SLUG isn't expressible — assert the guard directly instead.)
grep -q 'vLLM sweeps need SLUG' "$SWEEP" || fail "vLLM-needs-SLUG guard missing from script"
echo "  ✓ vLLM-needs-SLUG guard present"

echo "test-spec-sweep: ok"
