#!/usr/bin/env bash
#
# Guard: the >500 tok/s ceiling is an artifact guard for CLIENT TIMING and must
# not delete ENGINE-reported rates (club-3090#1290).
#
# Why: soak-helper filtered every decode rate through `0 < t <= 500`. That guards
# a real artifact — a client-timed turn where ttft ≈ wall divides by a tiny
# window and yields an absurd rate. But #1268 added engine-reported rates
# (prom-tpot / sglang-log / llamacpp-log), computed by the ENGINE over its own
# decode steps, which structurally cannot divide by a tiny window. Those rows
# landed in the same pool and were silently dropped from p50 and retention.
#
# 500 is reachable: a passing soak on the reference rig reported p50 269.8 on an
# instrument that reads 2-3x bench.sh, and 2x5090 + spec-decode clears it.
#
# ⚠️ Must FAIL against the pre-fix helper, or it is asserting the wrong thing.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H="${ROOT}/scripts/soak-helper.py"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
FAIL=0
bad() { echo "FAIL: $1 — expected $2, got $3" >&2; FAIL=1; }
ok()  { echo "  ✓ $1"; }

hdr='session_id,turn_id,t_ms,vram_mib,ttft_ms,decode_tps,completion_tokens,status,error,decode_basis,decode_source'

run_summary() {  # $1 = csv path
  python3 "$H" summary "$1" "$TMP/out.md" 1000 200 0 1 1 2>/dev/null || true
  cat "$TMP/out.md" 2>/dev/null
}

# --- 1: an ENGINE row above the ceiling must be COUNTED -----------------------
{ echo "$hdr"
  echo "1,1,1000,1000,100,640.0,800,200,,engine,prom-tpot"
  echo "1,2,1000,1000,100,660.0,800,200,,engine,prom-tpot"
} > "$TMP/engine_fast.csv"
out="$(run_summary "$TMP/engine_fast.csv")"
p50="$(command grep -oE 'p50 decode TPS[^|]*\| *[0-9.]+' <<<"$out" | command grep -oE '[0-9.]+$' | head -1)"
if [[ -z "$p50" || "$p50" == "0.00" ]]; then
  bad "engine rows >500 counted" "a non-zero p50 (both rows are 640/660)" "${p50:-<none>}"
else
  ok "engine-reported rates above the ceiling are counted (p50=$p50)"
fi

# --- 2: and the summary SAYS it counted them ---------------------------------
if command grep -qiE 'ENGINE-reported turn\(s\) exceeded' <<<"$out"; then
  ok "summary discloses that over-ceiling engine rows were counted"
else
  bad "over-ceiling disclosure" "a line naming the counted engine rows" "absent"
fi

# --- 3: CLIENT-timed artifact still filtered (do not regress the guard) ------
{ echo "$hdr"
  echo "1,1,1000,1000,100,250.0,800,200,,decode,"
  echo "1,2,1000,1000,999,9000.0,800,200,,decode,"
} > "$TMP/client_artifact.csv"
out2="$(run_summary "$TMP/client_artifact.csv")"
p50b="$(command grep -oE 'p50 decode TPS[^|]*\| *[0-9.]+' <<<"$out2" | command grep -oE '[0-9.]+$' | head -1)"
if [[ "$p50b" == "250.00" ]]; then
  ok "client-timed divide-by-tiny artifact still excluded (p50=$p50b)"
else
  bad "client artifact still filtered" "250.00 (the 9000 row excluded)" "${p50b:-<none>}"
fi

# --- 4: and that exclusion is no longer silent -------------------------------
if command grep -qiE 'client-timed turn\(s\) exceeded' <<<"$out2"; then
  ok "client-timed exclusion is disclosed, not silent"
else
  bad "client exclusion disclosure" "a line naming the excluded value" "absent"
fi

if [[ $FAIL -ne 0 ]]; then echo "FAIL: test-soak-engine-rate-ceiling" >&2; exit 1; fi
echo "PASS: test-soak-engine-rate-ceiling (club-3090#1290)"
