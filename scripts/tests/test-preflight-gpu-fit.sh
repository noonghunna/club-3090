#!/usr/bin/env bash
# Unit test for preflight_compose_gpu_fit — the free-VRAM-vs-util gate (club-3090 #535).
# vLLM aborts at boot when `free < util × total`; with restart:unless-stopped that becomes
# a silent 600s restart-loop. This gate must fail FAST + actionably instead. We mock
# nvidia-smi (a shell function satisfies `command -v`) and no-op sleep so the settle-retry
# doesn't actually wait.
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../preflight.sh
source "${ROOT_DIR}/scripts/preflight.sh"

FAILS=0
pass() { echo "  ok:   $1"; }
fail() { echo "  FAIL: $1"; FAILS=$((FAILS + 1)); }

# Instant retries — don't sleep through the ~10s settle window in tests.
sleep() { :; }
# Mocked GPU state: set MOCK_SMI to the CSV rows nvidia-smi should emit.
nvidia-smi() { printf '%s\n' "$MOCK_SMI"; }

# gemma-like compose: TP=2, aggressive 0.95 util.
tp2=$(mktemp)
cat > "$tp2" <<'YAML'
# Tensor-parallel: 2
services:
  x:
    command:
      - --gpu-memory-utilization
      - "${GPU_MEMORY_UTILIZATION:-0.95}"
YAML

# single-card compose: no TP header (defaults to 1 card), 0.92 util.
single=$(mktemp)
cat > "$single" <<'YAML'
services:
  x:
    command: ["--gpu-memory-utilization", "${GPU_MEMORY_UTILIZATION:-0.92}"]
YAML

# 1) FITS — both cards nearly empty (need ≈ 0.95×24576×0.98 = 22.9 GiB; 24000 free).
MOCK_SMI=$'0, 24576, 24000\n1, 24576, 24000'
if preflight_compose_gpu_fit "$tp2" 0 >/dev/null 2>&1; then pass "fits when both cards are free"; else fail "should fit when both cards free"; fi

# 2) INSUFFICIENT — the exact #535 numbers (free 22350/22060 < 22.9 GiB need) → hard fail.
MOCK_SMI=$'0, 24576, 22350\n1, 24576, 22060'
err=$(preflight_compose_gpu_fit "$tp2" 0 2>&1); rc=$?
[ "$rc" -ne 0 ] && pass "fails when a TP card is short (#535 repro)" || fail "should fail on the #535 numbers (rc=$rc)"
echo "$err" | grep -q "GPU_MEMORY_UTILIZATION=0.90" && pass "emits the actionable lower-util hint" || fail "missing actionable hint"
echo "$err" | grep -q "GPU 1 has" && pass "names the short card (GPU 1)" || fail "should name the worst card"

# 3) --force bypasses the gate (rc 0, with a WARN).
if preflight_compose_gpu_fit "$tp2" 1 >/dev/null 2>&1; then pass "--force bypasses"; else fail "--force should bypass"; fi

# 4) env GPU_MEMORY_UTILIZATION override lowers the need so the same VRAM now fits.
MOCK_SMI=$'0, 24576, 22350\n1, 24576, 22060'
if GPU_MEMORY_UTILIZATION=0.88 preflight_compose_gpu_fit "$tp2" 0 >/dev/null 2>&1; then pass "env util override (0.88) fits"; else fail "env override should fit"; fi

# 5) single-card only needs ONE good card — GPU0 busy, GPU1 free → OK.
MOCK_SMI=$'0, 24576, 5000\n1, 24576, 24000'
if preflight_compose_gpu_fit "$single" 0 >/dev/null 2>&1; then pass "single-card fits on the free card"; else fail "single-card should fit on the best card"; fi

# 6) no nvidia-smi → skip (never blocks a rig without a GPU query).
( unset -f nvidia-smi
  command_v_backup() { command -v nvidia-smi; }
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    preflight_compose_gpu_fit "$tp2" 0 >/dev/null 2>&1 && echo "  ok:   skips without nvidia-smi" || echo "  FAIL: should skip without nvidia-smi"
  else
    echo "  ok:   (nvidia-smi present on host — skip-case not exercised)"
  fi )

rm -f "$tp2" "$single"
if [ "$FAILS" -eq 0 ]; then echo "PASS test-preflight-gpu-fit"; exit 0; else echo "FAIL test-preflight-gpu-fit ($FAILS)"; exit 1; fi
