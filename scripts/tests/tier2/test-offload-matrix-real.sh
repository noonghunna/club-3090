#!/usr/bin/env bash
# test-offload-matrix-real — TIER 2. Needs a real GPU + a real model. NOT part of the
# default `scripts/tests/*.sh` sweep (it lives one directory down so the glob misses it),
# and additionally refuses to run without OFFLOAD_MATRIX_TIER2=1.
#
# ── Why this tier exists ──────────────────────────────────────────────────────
# Tiers 0 and 1 assert the sweep's logic against a FAKE server whose log strings are
# hard-coded from what the engine emitted when the fixture was written. That is exactly
# the thing that rots: an engine bump renames a log line, the fake keeps emitting the old
# one, tier 1 stays green, and the real sweep silently reports '-' for a whole column.
#
# So tier 2's job is NOT to re-test the logic. It is to prove the REAL engine still emits
# every string the scrapers depend on. If this fails and tier 1 passes, the fixture is
# stale -- fix `fixtures/offload-matrix/fake-llama-server`, not the sweep.
#
# Asserts structure, status and log-contract only. NEVER a throughput value: those are
# not portable across rigs, and asserting them would make this test a rig detector.
#
# Usage:
#   OFFLOAD_MATRIX_TIER2=1 MODEL=/path/to/model.gguf LLAMA_SERVER=/path/to/llama-server \
#     bash scripts/tests/tier2/test-offload-matrix-real.sh
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP="$ROOT_DIR/scripts/offload-matrix.sh"
REND="$ROOT_DIR/scripts/offload-matrix-render.py"
fail() { echo "FAIL: $1" >&2; exit 1; }
skip() { echo "SKIP: $1"; exit 0; }

[[ "${OFFLOAD_MATRIX_TIER2:-0}" == 1 ]] || skip "set OFFLOAD_MATRIX_TIER2=1 to run the real-GPU tier"
command -v nvidia-smi >/dev/null 2>&1 || skip "no nvidia-smi"
nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || skip "no usable GPU"
[[ -n "${MODEL:-}"  && -f "${MODEL:-}" ]] || skip "set MODEL=/path/to/model.gguf"
[[ -n "${LLAMA_SERVER:-}" && -x "${LLAMA_SERVER:-}" ]] || skip "set LLAMA_SERVER=/path/to/llama-server"

# A real server must not be up: the sweep boots its own per arm and would refuse (exit 2),
# and overriding that here would measure under contention.
if pgrep -x llama-server >/dev/null; then
  fail "a llama-server is already running — stop it first (this tier boots its own)"
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
# ⚠️⚠️ FIXTURE PORTS LIVE ABOVE THE PRODUCT PORT SPACE, ON PURPOSE.
# The registry allocates real slug `default_port`s across 8010-8199, so a
# fixture base inside that span silently collides: a slug parked on one of
# these ports makes this guard fail WHENEVER THAT MODEL IS SERVING, and the
# failure reads as a broken test rather than a port clash. Measured
# 2026-09-18 on test-bench-capture, whose 8147 base overlapped ELEVEN
# registry slugs; three suite failures were misdiagnosed as pre-existing
# before the cause was found. 18xxx is clear — the product never allocates
# above 8199. Keep the last three digits so the old base stays greppable.
PORT="${TEST_PORT:-18138}"
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "  booting ONE short real arm (ctx 4096, 2 rounds) on $NGPU GPU(s) ..."
MODEL="$MODEL" LLAMA_SERVER="$LLAMA_SERVER" \
OUT_DIR="$TMP" TSV="$TMP/r.tsv" PORT="$PORT" HOST=127.0.0.1 \
PRESET=smoke SWEEP_N=1 SWEEP_NMAX=0 SWEEP_CTX=4096 SWEEP_SHAPE=code \
SWEEP_CACHE="${TIER2_CACHE:-0}" REPS=1 ROUNDS=2 GEN=32 BOOT_TIMEOUT=600 \
INHERIT=0 \
  bash "$SWEEP" > "$TMP/sweep.log" 2>&1 || true

[[ -f "$TMP/r.tsv" ]] || { sed -n '1,40p' "$TMP/sweep.log" >&2; fail "no TSV produced"; }

# ── structural invariants (same ones tier 1 asserts, now on real output) ──
NCOL=$(head -1 "$TMP/r.tsv" | awk -F'\t' '{print NF}')
widths=$(awk -F'\t' '{print NF}' "$TMP/r.tsv" | sort -u | paste -sd,)
[[ "$widths" == "$NCOL" ]] || fail "ragged TSV on real output — widths [$widths], header $NCOL"
[[ "$(wc -l < "$TMP/r.tsv")" -ge 2 ]] || fail "header only, no data row"
status=$(awk -F'\t' 'NR==2{print $NF}' "$TMP/r.tsv")
echo "  ✓ real arm produced a full-width row (status=$status)"

if [[ "$status" != "OK" ]]; then
  sed -n '1,40p' "$TMP/sweep.log" >&2
  fail "real arm did not reach OK (status=$status) — read the log above before trusting any sweep"
fi

# ── THE POINT OF THIS TIER: the real engine still emits what the scrapers need ──
# Every pattern below is one the sweep greps out of the arm log. A miss here with tier 1
# green means the FIXTURE is stale, not the sweep.
arm_log=$(ls -t "$TMP"/*.log 2>/dev/null | grep -v sweep.log | head -1)
[[ -n "$arm_log" ]] || arm_log="$TMP/sweep.log"
miss=0
check() {  # $1=label  $2=ERE
  if grep -qE "$2" "$arm_log" 2>/dev/null; then
    echo "    ✓ $1"
  else
    echo "    ✗ $1   (pattern: $2)"; miss=1
  fi
}
echo "  log contract — real engine vs what the scrapers expect:"
check "context size (ctx_slot)"      'n_ctx_seq *= *[0-9]+|n_ctx_per_seq *= *[0-9]+'
check "decode timing (strm)"         'eval time'
check "tokens-per-second figure"     '[0-9.]+ tokens per second'
(( miss == 0 )) || fail "the real engine no longer emits a scraped pattern — update the fake-llama-server fixture to match, then re-check tier 1"

# cache-specific contract only when the cache was actually requested
if [[ "${TIER2_CACHE:-0}" != 0 ]]; then
  echo "  expert-cache contract:"
  check "cache enabled line"  '\[moe-cache\] enabled:'
  check "pool-init slots="    'slots=[0-9]+'
  check "hits=H/T"            'hits=[0-9]+/[0-9]+'
  (( miss == 0 )) || fail "expert-cache log strings drifted — update the fixture"
  # one pool-init line per device, the invariant behind the CACHE_DISABLED fix
  pool_lines=$(grep -c 'slots=' "$arm_log")
  [[ "$pool_lines" -ge "$NGPU" ]] || fail \
    "only $pool_lines pool line(s) for $NGPU device(s) — a per-device budget failure should have been caught as CACHE_DISABLED, not OK"
  echo "    ✓ $pool_lines pool line(s) for $NGPU device(s)"
fi

# ── renderer over genuinely-produced rows ──
for view in --perf --bw --cache; do
  python3 "$REND" "$TMP/r.tsv" "$view" >/dev/null 2>"$TMP/err" || { cat "$TMP/err" >&2; fail "renderer crashed on $view over real rows"; }
done
echo "  ✓ renderer handles real rows"

echo "test-offload-matrix-real: ok"
