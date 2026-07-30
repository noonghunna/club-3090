#!/usr/bin/env bash
# test-offload-matrix-mocked — Tier 1 guards for scripts/offload-matrix.sh.
#
# Runs the REAL sweep against a fake llama-server (scripts/tests/fixtures/) that
# emits the exact log lines the script scrapes and serves the two endpoints it
# calls. No GPU, no model, no engine — but the whole boot -> drive -> scrape ->
# emit -> render path executes, which is where the silent-wrongness defects live.
#
# Each case asserts a STATUS and then the row invariants. A harness that reports
# a healthy status for an unhealthy run is the failure mode that matters here;
# throughput values are deliberately never asserted.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP="$ROOT_DIR/scripts/offload-matrix.sh"
FAKE="$ROOT_DIR/scripts/tests/fixtures/fake-llama-server.py"
export PYTHONUTF8="${PYTHONUTF8:-1}"   # repo rule: locale must not decide python decoding
fail() { echo "FAIL: $1" >&2; [[ -n "${LOG:-}" && -f "${LOG:-}" ]] && tail -20 "$LOG" >&2; exit 1; }
# The suite boots a FAKE server and never touches a GPU, so the sweep's
# "a llama-server is already running" VRAM-contention refusal has no bearing on
# it — without this override the whole suite goes red on any box that happens to
# be serving a model, which is most dev rigs.
export ALLOW_CONCURRENT_SERVER=1
TMP="$(mktemp -d)"
# NOTE the bracket in the pattern below. An unbracketed pkill -f matches the
# killing command's OWN command line and kills this shell (exit 144), so the
# trap never finishes and the fixture it should reap survives. Bracketing the
# first character makes the pattern unable to match itself.
# SIGKILL, not SIGTERM: the `probefail` fixture ignores SIGTERM on purpose, so a
# TERM-only cleanup would leave the very process this suite asserts about.
trap 'rm -rf "$TMP"; pkill -9 -f "[f]ake-llama-server" >/dev/null 2>&1 || true' EXIT
touch "$TMP/model.gguf"

# pick a port nothing is using, so a stray local server can't poison the run
PORT=8891; while ss -ltn 2>/dev/null | grep -q ":$PORT "; do PORT=$((PORT+1)); done

CTX_SEQ=32768
run_case() {  # $1=scenario; each case gets its own ctx so arms stay distinguishable
  local sc="$1"; shift
  CTX_SEQ=$((CTX_SEQ + 1024))
  local out="$TMP/$sc"; rm -rf "$out"
  LOG="$TMP/$sc.log"
  env FAKE_SCENARIO="$sc" FAKE_NDEV=2 NGPU=2 \
      MODEL="$TMP/model.gguf" LLAMA_SERVER="$FAKE" INHERIT=0 OUT_DIR="$out" PORT="$PORT" \
      LAYERS_TOTAL=48 N_EXPERT=256 REPS=1 ROUNDS=2 GEN=8 BOOT_TIMEOUT=40 \
      SWEEP_OFFLOAD=19 SWEEP_N=1 SWEEP_NMAX=0 SWEEP_CTX="$CTX_SEQ" SWEEP_CACHE=8192 \
      SWEEP_ADMIT="1/64" SWEEP_SHAPE=copy "$@" \
      bash "$SWEEP" > "$LOG" 2>&1 || true
  STATUS="$(awk -F'\t' 'NR>1{print $NF}' "$out/offload-matrix-results.tsv" 2>/dev/null | head -1)"
  TSV="$out/offload-matrix-results.tsv"
}

# ---- row invariants, asserted after every case -------------------------------
invariants() {
  local tsv="$1" ncol row n
  ncol=$(head -1 "$tsv" | awk -F'\t' '{print NF}')
  while IFS= read -r row; do
    n=$(awk -F'\t' '{print NF}' <<<"$row")
    (( n == ncol )) || fail "V1 field count: row has $n fields, header has $ncol"
  done < <(tail -n +2 "$tsv")
  # V2: status=OK implies non-zero throughput
  awk -F'\t' -v OK=1 'NR>1 && $NF=="OK" && ($14=="-" || $14+0==0){exit 1}' "$tsv" \
    || fail "V2: a row with status=OK has zero/absent throughput"
  # V3: bypass>0 implies status != OK
  awk -F'\t' 'NR>1 && $(NF-2)+0>0 && $NF=="OK"{exit 1}' "$tsv" \
    || fail "V3: bypass>0 but status=OK"
}

# 1. healthy — both devices pool, arm is usable
run_case healthy
[[ "$STATUS" == "OK" ]] || fail "healthy scenario should be OK, got '$STATUS'"
invariants "$TSV"
echo "  ✓ healthy: OK + invariants hold"

# 2. PARTIAL — only CUDA0 pools. The engine STILL prints "[moe-cache] enabled:",
#    so a detector keying on that line reports OK for a half-uncached run. This
#    is the defect found live on 2026-07-30; it must not regress.
run_case partial
[[ "$STATUS" == "CACHE_PARTIAL" ]] || fail "per-device pool failure should be CACHE_PARTIAL, got '$STATUS'"
grep -q "only 1 of 2 devices" "$LOG" || fail "CACHE_PARTIAL should say how many devices got a pool"
invariants "$TSV"
echo "  ✓ partial-pool caught as CACHE_PARTIAL (not OK)"

# 3. no budget anywhere — nothing pools at all
run_case nobudget
[[ "$STATUS" == "CACHE_DISABLED" ]] || fail "no pools should be CACHE_DISABLED, got '$STATUS'"
grep -qi "reserve" "$LOG" || fail "CACHE_DISABLED should name the reserve as the remedy"
invariants "$TSV"
echo "  ✓ no-pool caught as CACHE_DISABLED with remedy"

# 4. bypass warning present — pools exist but decodes decline them
run_case bypass
[[ "$STATUS" == "INVALID_BYPASS" ]] || fail "bypass warning should be INVALID_BYPASS, got '$STATUS'"
invariants "$TSV"
echo "  ✓ cache bypass caught as INVALID_BYPASS"

# 5. zero tokens — endpoints answer but nothing streams. Must never read OK.
run_case notokens
[[ "$STATUS" == "NO_TOKENS" ]] || fail "zero tokens should be NO_TOKENS, got '$STATUS'"
invariants "$TSV"
echo "  ✓ zero-token run caught as NO_TOKENS"

# 6. the renderer must exclude every non-OK arm from its conclusions
{ head -1 "$TMP/healthy/offload-matrix-results.tsv"; \
  for f in "$TMP"/*/offload-matrix-results.tsv; do tail -n +2 "$f"; done; } > "$TMP/all.tsv"
out="$(python3 "$ROOT_DIR/scripts/offload-matrix-render.py" "$TMP/all.tsv" 2>&1)" \
  || fail "renderer crashed on the combined mocked TSV"
for s in CACHE_PARTIAL CACHE_DISABLED INVALID_BYPASS NO_TOKENS; do
  grep -q "$s" <<<"$out" || fail "renderer did not surface $s"
done
grep -q "NOT OK" <<<"$out" || fail "renderer must list non-OK arms separately"
echo "  ✓ renderer surfaces every failure status and separates them"

# PIDs of fixture servers still alive. Three guards, each one a real footgun:
#   * bracketed pattern, so the pgrep cannot match its own command line;
#   * comm filter, because the bracket trick does NOT save a CALLER whose command
#     line merely quotes the pattern (a wrapper shell running this suite matched);
#   * `|| true`, because pgrep exits 1 on no-match and under `set -e` a bare
#     assignment from a failing substitution aborts the suite instead of asserting.
strays() {
  local p out=""
  for p in $(pgrep -f "[f]ake-llama-server" 2>/dev/null || true); do
    [[ -r "/proc/$p/comm" ]] || continue
    if [[ "$(</proc/$p/comm)" == python3* ]]; then out+="$p "; fi
  done
  printf '%s' "$out"
}

# 7. PROBE LEAK — the layer probe boots a REAL server, so a probe that gives up
#    without reaping it leaves that server holding the model's VRAM, and every
#    arm launched afterwards measures a starved machine while reporting OK
#    (observed live 2026-07-30, ~22 GiB held on both cards). The fixture never
#    prints n_layer, so the probe must time out, AND it ignores SIGTERM the way a
#    server still inside model load does — so `kill; wait` alone both hangs and
#    leaks. Only an escalating reap on every exit path passes this.
[[ -z "$(strays)" ]] || fail "a fixture from an earlier case is still running: $(strays)"
LOG="$TMP/probefail.log"
set +e
timeout 60 env FAKE_SCENARIO=probefail MODEL="$TMP/model.gguf" LLAMA_SERVER="$FAKE" \
  INHERIT=0 OUT_DIR="$TMP/probefail" PORT="$PORT" LAYERS_TOTAL= PROBE_TIMEOUT=3 \
  SWEEP_OFFLOAD=19 bash "$SWEEP" > "$LOG" 2>&1
rc=$?
set -e
(( rc != 124 )) || fail "probe never returned: it waits on a child that ignores SIGTERM"
[[ "$rc" == "2" ]] || fail "a probe that cannot read n_layer should exit 2, got $rc"
grep -q "LAYERS_TOTAL" "$LOG" || fail "probe failure should tell the operator to set LAYERS_TOTAL"
left="$(strays)"
if [[ -n "$left" ]]; then
  pkill -9 -f "[f]ake-llama-server" >/dev/null 2>&1 || true
  fail "failed probe LEAKED its server (pid(s): $left) — it would hold VRAM for the whole run"
fi
echo "  ✓ failed probe reaps its server and exits actionably (no leaked VRAM holder)"

echo "test-offload-matrix-mocked: ok (Tier 1 — mocked server, no GPU)"
