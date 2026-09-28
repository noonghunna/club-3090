#!/usr/bin/env bash
# test-bench-card — guards for scripts/lib/card.sh and bench.sh's CARD= toggle.
#
# The card is a DECISION RECORD: it gets pasted into an issue and argued from. So
# the failure that matters is not "the renderer crashed", it is "the card stated
# something the run did not measure". These assertions are therefore mostly about
# what the card must REFUSE to say:
#   * human judgements stay `<fill: …>` placeholders — never guessed;
#   * an A/B whose arms differ in a held-constant value renders CONFOUNDED at the
#     TOP, not buried in a table nobody reads;
#   * an unparseable baseline fails LOUDLY rather than rendering half a delta;
#   * a resource-cost row is reported, not scored as good or bad.
#
# Plus the round-trip that the whole A/B rests on: the record bench.sh builds
# in-process and the record card.sh parses back out of a saved log must AGREE.
# If they drift, an A/B silently compares two different measurements.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUTF8="${PYTHONUTF8:-1}"   # repo rule: locale must not decide python decoding
BENCH="$ROOT_DIR/scripts/bench.sh"
CARDLIB="$ROOT_DIR/scripts/lib/card.sh"
FIX="$ROOT_DIR/scripts/tests/fixtures/offload-matrix"
CFIX="$ROOT_DIR/scripts/tests/fixtures/bench-card"
# ⚠️⚠️ FIXTURE PORTS LIVE ABOVE THE PRODUCT PORT SPACE, ON PURPOSE.
# The registry allocates real slug `default_port`s across 8010-8199, so a
# fixture base inside that span silently collides: a slug parked on one of
# these ports makes this guard fail WHENEVER THAT MODEL IS SERVING, and the
# failure reads as a broken test rather than a port clash. Measured
# 2026-09-18 on test-bench-capture, whose 8147 base overlapped ELEVEN
# registry slugs; three suite failures were misdiagnosed as pre-existing
# before the cause was found. 18xxx is clear — the product never allocates
# above 8199. Keep the last three digits so the old base stays greppable.
PORT_BASE="${TEST_PORT:-18171}"
fail() { echo "FAIL: $1" >&2; exit 1; }

for f in "$BENCH" "$CARDLIB" "$CFIX/baseline.log" "$CFIX/baseline-mismatched.log" \
         "$CFIX/live-offload-run.log" \
         "$FIX/fake-llama-server" "$FIX/fake-nvidia-smi"; do
  [[ -f "$f" ]] || fail "missing fixture or script: $f"
done

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null; true' EXIT
mkdir -p "$TMP/bin"
install -m 0755 "$FIX/fake-nvidia-smi"   "$TMP/bin/nvidia-smi"
install -m 0755 "$FIX/fake-llama-server" "$TMP/bin/fake-llama-server"
: > "$TMP/model.gguf"

# ===========================================================================
# TIER 0
# ===========================================================================
echo "  tier 0: source guards"
bash -n "$CARDLIB" || fail "bash -n: syntax error in card.sh"
python3 - "$CARDLIB" <<'PY' || fail "card.sh's embedded python does not parse"
import ast, re, sys
src = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"<<'PY'\n(.*?)\nPY\n", src, re.S)
assert m, "no embedded python found in card.sh"
ast.parse(m.group(1))
PY
if command grep -nE '^[^#]*\b(pgrep|pkill) +-[a-zA-Z]*f' "$CARDLIB"; then
  fail "card.sh uses pgrep/pkill -f"
fi
bare=$(command grep -nE '(^|[|;&(]|\$\()\s*grep ' "$CARDLIB" || true)
[[ -z "$bare" ]] || fail "card.sh uses bare grep (ugrep shim): $bare"
# JSON is #249's job; if that boundary blurs, the two producers drift.
command grep -q 'OUT OF SCOPE: machine-parseable JSON' "$CARDLIB" \
  || fail "card.sh must record that JSON output belongs to #249, not here"
echo "  ✓ syntax, no pgrep -f, command grep, #249 boundary recorded"

# ===========================================================================
# TIER 1a — renderer units against fixture records
# ===========================================================================
echo "  tier 1a: renderer units"
# shellcheck source=../lib/card.sh
source "$CARDLIB"

REC="$TMP/rec.kv"
cat > "$REC" <<'EOF'
meta.date=2026-08-01
meta.endpoint=http://127.0.0.1:8030
meta.mode=chat
fp.model=deepseek-v4-flash
fp.ctx=65536
fp.slots=1
fp.ftype=MXFP4 MoE
fp.drafter=none
fp.kv=q8_0 (source: argv)
fp.argv=model=/m/x.gguf offload=blk.*=CPU threads=24 ubatch=2048 ngl=99 split=1,1
fp.moe=moe_cache_cap=20480 RESERVE_MB=2048 ADMIT_AFTER=1 THROTTLE=64 MAX_BATCH=8 STATS=1000
fp.offload=DETECTED via argv -ot
proto.warmups=3
proto.runs=5
proto.force_tokens=1000
proto.thinking=0
proto.env=RUNS=5 FORCE_TOKENS=1000
shape.narrative.n=5
shape.narrative.n_usable=5
shape.narrative.decode_mean=18.43
shape.narrative.decode_cv=0.6
shape.narrative.wall_mean=18.20
shape.narrative.wall_cv=0.7
shape.narrative.ttft_mean_ms=507
shape.narrative.ttft_std_ms=30
prefill.prefill-10k.n=3
prefill.prefill-10k.tps_mean=127.20
prefill.prefill-10k.tps_cv=0.1
prefill.prefill-10k.ttft_ms=81000
gpu.vram_idle=23100
gpu.vram_peak=23800
gpu.vram_post=23100
gpu.leak=0
cache.CUDA0.marginal=48.0
cache.CUDA0.cumulative=29.6
cache.CUDA0.cum_start=19.3
cache.CUDA0.slots=4038
cache.CUDA1.marginal=58.0
cache.CUDA1.cumulative=35.3
cache.CUDA1.cum_start=23.3
cache.CUDA1.slots=2500
integrity.status=OK
integrity.health=PASS (all health counters zero)
integrity.swap=PASS (no serving-process pages in swap)
integrity.divergence=2.1
EOF

snap="$(card_render snapshot "$REC")" || fail "snapshot render failed"
command grep -q '^## 📊 bench.sh — deepseek-v4-flash' <<<"$snap" || fail "snapshot title wrong"
for want in "**Config:**" "**Env:** RUNS=5 FORCE_TOKENS=1000" "**Protocol:** 3 warm + 5 measured" \
            "| narrative | 5 | 18.43 | 0.6% |" "| prefill-10k | 3 | 127.20 |" \
            "### Integrity" "**One-line verdict:**"; do
  command grep -qF "$want" <<<"$snap" || fail "snapshot missing: $want"
done
# an offload run renders the OFFLOAD BLOCK, per device, never averaged
command grep -q '^### CPU offload + expert cache' <<<"$snap" \
  || fail "an offload run must render the CPU offload + expert cache block"
command grep -q '| CUDA0 | 4038 slots' <<<"$snap" || fail "per-device pool row missing: $snap"
command grep -q '| CUDA0 |.*\*\*48.0%\*\*' <<<"$snap" || fail "CUDA0 marginal hits missing from the table"
command grep -q '| CUDA1 |.*\*\*58.0%\*\*' <<<"$snap" \
  || fail "the CUDA0/CUDA1 asymmetry was averaged away on the card"
command grep -q 'Quote the \*\*marginal\*\* rate' <<<"$snap" \
  || fail "the card must tell the reader WHICH hit rate to quote"
command grep -qF 'The cache config IS the arm identity' <<<"$snap" \
  || fail "the cache-config line must explain why pool size alone is not the arm"
command grep -q 'skips.*= admission throttling' <<<"$snap" \
  || fail "skips must be footnoted as a load signal, not a defect"
# auto-verified integrity items carry evidence, not just a tick
command grep -q '\[x\] capture status: OK' <<<"$snap" || fail "status not on the integrity panel"
command grep -q '\[x\] swap check — PASS' <<<"$snap" || fail "swap check not on the integrity panel"
command grep -q '\[x\] cache hard-failure counters zero' <<<"$snap" || fail "health counters missing"
command grep -q '\[x\] n-usable == n for every shape — narrative 5/5' <<<"$snap" || fail "n-usable missing"
command grep -q '\[x\] VRAM returned after the run — leak delta 0 MiB' <<<"$snap" || fail "leak delta missing"
echo "  ✓ snapshot: header, shape + prefill tables, per-device cache, integrity panel"

# Human judgement is NEVER guessed.
for ph in "<fill: config-slug>" "<fill: engine+tag>" "<fill: power cap>" "<fill: confirm>" \
          "<fill: what this run establishes"; do
  command grep -qF "$ph" <<<"$snap" || fail "snapshot must keep the placeholder: $ph"
done
echo "  ✓ snapshot: every human judgement stays an explicit <fill: …> placeholder"

# A failing run must be visibly unquotable.
sed -e 's/^integrity.status=OK/integrity.status=NO_TOKENS/' \
    -e 's/^integrity.health=PASS.*/integrity.health=FAIL — CUDA1: fill_fail=3/' \
    -e 's/^shape.narrative.n_usable=5/shape.narrative.n_usable=1/' "$REC" > "$TMP/bad.kv"
badsnap="$(card_render snapshot "$TMP/bad.kv")" || fail "snapshot render failed on a bad run"
command grep -q '\[ \] capture status: NO_TOKENS' <<<"$badsnap" || fail "a failed status must not be ticked"
command grep -q 'do not quote a throughput number' <<<"$badsnap" || fail "NO_TOKENS must be called out on the card"
command grep -q '\[ \] cache hard-failure counters zero .* — FAIL' <<<"$badsnap" || fail "failing health must not be ticked"
command grep -q 'not trustworthy' <<<"$badsnap" || fail "n-usable shortfall must discredit the CV on the card"
command grep -q '1/5 ⚠️' <<<"$badsnap" || fail "the shape table must flag the usable-run shortfall"
echo "  ✓ snapshot: a failed run renders unticked, flagged, and explicitly unquotable"

# ===========================================================================
# TIER 1b — A/B
# ===========================================================================
echo "  tier 1b: A/B render"
ab="$(KNOB=moe-cache EXPECTATION="bigger pool should raise decode TPS" \
      BOOT_POLICY="boot-per-arm x1" card_render ab "$REC" "$CFIX/baseline.log")" \
  || fail "A/B render failed"
command grep -q '^## ⚖️ bench.sh A/B — moe-cache' <<<"$ab" || fail "A/B title should carry the declared knob"
command grep -qF "**Pre-registered expectation:** bigger pool should raise decode TPS" <<<"$ab" \
  || fail "EXPECTATION not rendered"
command grep -qF "**Boot policy:** boot-per-arm x1" <<<"$ab" || fail "BOOT_POLICY not rendered"
command grep -q 'decode narrative | 17.30 | 18.43 | +6.5% | outside band — real improvement' <<<"$ab" \
  || fail "delta row wrong: $(command grep 'decode narrative' <<<"$ab")"
command grep -q 'TTFT narrative (ms) | 612.00 | 507.00 | -17.2% | outside band — real improvement' <<<"$ab" \
  || fail "TTFT must be scored lower-is-better: $(command grep 'TTFT narrative' <<<"$ab")"
command grep -q 'prefill 10k | 119.40 | 127.20' <<<"$ab" || fail "prefill delta row missing"
# resource rows are REPORTED, not scored — spending pool to buy throughput may be
# exactly the intended trade, and only the human knows the VRAM budget.
command grep -q 'pool slots CUDA0 | 2000.00 | 4038.00 | +101.9% | \*\*+101.9% resource spend\*\*' <<<"$ab" \
  || fail "resource-cost row should report spend, not call it an improvement"
command grep -qv 'pool slots CUDA0.*improvement' <<<"$ab" || true
command grep -q 'pool slots CUDA0.*improvement' <<<"$ab" \
  && fail "a resource row must NOT be scored as an improvement"
command grep -qF "**Decision:** <fill: ADOPT / REJECT / PARK>" <<<"$ab" || fail "decision must stay a placeholder"
echo "  ✓ A/B: Δ table, lower-is-better TTFT, resource rows reported not scored"

# the declared knob is attributed, everything else held constant
command grep -q '🔧 the declared knob' <<<"$ab" || fail "the declared knob should be attributed in the invariants table"
command grep -q 'CONFOUNDED' <<<"$ab" && fail "a run differing ONLY in the declared knob must not be confounded"
echo "  ✓ A/B: declaring KNOB attributes that difference instead of confounding the card"

# noise band: derived and SAID SO
command grep -qE 'Noise band:\*\* ±1\.2% \(derived: 2x the worst decode CV across both arms' <<<"$ab" \
  || fail "the derived band must state how it was established: $(command grep 'Noise band' <<<"$ab")"
ab2="$(NOISE_BAND=20 KNOB=moe-cache card_render ab "$REC" "$CFIX/baseline.log")"
command grep -q 'Noise band:\*\* ±20.0% (NOISE_BAND env' <<<"$ab2" || fail "explicit NOISE_BAND not honoured"
command grep -q 'decode narrative | 17.30 | 18.43 | +6.5% | within band — no signal' <<<"$ab2" \
  || fail "a wider band must reclassify the same delta as no-signal"
echo "  ✓ A/B: band derived from CVs and stated, or taken from NOISE_BAND — and it changes verdicts"

# ===========================================================================
# TIER 1c — the teeth: confounded arms, and unparseable baselines
# ===========================================================================
echo "  tier 1c: confound + failure paths"
conf="$(card_render ab "$REC" "$CFIX/baseline-mismatched.log")" || fail "confounded A/B failed to render"
# The banner must be at the TOP — buried it is worthless.
[[ "$(head -1 <<<"$conf")" == "> ## ⛔ CONFOUNDED" ]] \
  || fail "the CONFOUNDED banner must be the FIRST line, got: $(head -1 <<<"$conf")"
# 5: the four the fixture deliberately changes (model, ctx, KV, threads/ubatch in
# the argv) plus the moe-cache config, which already differed from the record.
command grep -q 'differ in 5 value(s)' <<<"$conf" || fail "confound count wrong: $(command grep 'differ in' <<<"$conf")"
# prefix match: some labels carry a parenthetical (e.g. the moe-cache knob list)
for f in model "served ctx" "KV cache type" "moe-cache config"; do
  command grep -q "| $f.*⛔ \*\*CONFOUND\*\*" <<<"$conf" || fail "invariant '$f' not flagged as a confound"
done
command grep -q '<fill: knob>' <<<"$conf" || fail "with no KNOB declared the knob must stay a placeholder"
echo "  ✓ confounded arms: banner is line 1, every mismatched invariant flagged"

# A baseline that cannot be parsed must FAIL, not render half a delta.
for bad in "/nonexistent/path.log" "$TMP/garbage.log" ""; do
  [[ "$bad" == "$TMP/garbage.log" ]] && echo "this is not a bench log at all" > "$bad"
  if out="$(card_render ab "$REC" "$bad" 2>&1)"; then
    fail "an unparseable baseline ('$bad') rendered anyway:\n$out"
  fi
  command grep -q 'baseline unparseable:' <<<"$out" \
    || fail "an unparseable baseline must say 'baseline unparseable: <why>', got: $out"
done
# ...and the reason must be specific enough to act on.
out="$(card_render ab "$REC" "$TMP/garbage.log" 2>&1 || true)"
command grep -qE 'baseline unparseable:.*(CONFIG FINGERPRINT|summary)' <<<"$out" \
  || fail "the unparseable reason must name what was missing, got: $out"
echo "  ✓ unparseable/missing baselines fail loudly, naming what was missing"

# ===========================================================================
# TIER 1d — end-to-end through bench.sh, incl. the round-trip
# ===========================================================================
echo "  tier 1d: end-to-end + record round-trip"
SRV_PID=""; SRV_LOG=""
start_server() {   # see test-bench-capture.sh for why $! is not usable here
  local scen="$1" port="$2"; shift 2
  SRV_LOG="$TMP/server-$port.log"
  local pidfile="$TMP/server-$port.pid"; rm -f "$pidfile"
  env FAKE_SCENARIO="$scen" FAKE_NGPU=2 FAKE_TOKENS=40 FAKE_PID_FILE="$pidfile" "$@" \
      GGML_CUDA_MOE_CACHE_RESERVE_MB=2048 GGML_CUDA_MOE_CACHE_ADMIT_AFTER=1 \
      GGML_CUDA_MOE_CACHE_THROTTLE=64 GGML_CUDA_MOE_CACHE_MAX_BATCH=8 \
      GGML_CUDA_MOE_CACHE_STATS=1000 \
      "$TMP/bin/fake-llama-server" -m "$TMP/model.gguf" \
        -ot 'blk.(1|3|5).ffn_.*_exps.weight=CPU' -t 24 -ub 2048 -ct q8_0 -ngl 99 -ts 1,1 \
        --moe-cache 8192 --host 127.0.0.1 --port "$port" > "$SRV_LOG" 2>&1 &
  local i
  for i in $(seq 1 40); do
    curl -sf -m 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    sleep 0.25
  done
  SRV_PID="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$SRV_PID" && -r "/proc/$SRV_PID/cmdline" ]] || fail "fake server pid unreadable — run not hermetic"
  sleep 2
}
run_bench() {   # $1=port $2=out [extra env ...]
  local port="$1" out="$2"; shift 2
  start_server healthy "$port"
  PATH="$TMP/bin:$PATH" PREFLIGHT_NO_AUTODETECT=1 CONTAINER=none \
    SERVER_LOG="$SRV_LOG" SERVER_PID="$SRV_PID" URL="http://127.0.0.1:$port" \
    MODEL=fake-model RUNS=2 WARMUPS=1 PREFILL_PROBE=0 \
    env "$@" timeout 300 bash "$BENCH" > "$out" 2>"${out}.err" || true
  kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=""
}

# CARD unset => no card. The byte-identical guarantee lives in
# test-bench-capture.sh; this asserts the toggle itself is genuinely off.
run_bench "$((PORT_BASE))" "$TMP/nocard.out"
command grep -q 'bench.sh — ' "$TMP/nocard.out" && fail "CARD unset must not render a card"
command grep -q 'BENCH CARD' "$TMP/nocard.out" && fail "CARD unset must not emit the card section"
echo "  ✓ CARD unset renders no card at all"

run_bench "$((PORT_BASE+1))" "$TMP/snap.out" CARD=snapshot CARD_RECORD_OUT="$TMP/live-rec.kv"
command grep -q '## 📊 bench.sh — fake-model' "$TMP/snap.out" \
  || { tail -30 "$TMP/snap.out" >&2; fail "CARD=snapshot did not render a card"; }
command grep -q '^```markdown$' "$TMP/snap.out" || fail "the card must be a fenced markdown block, ready to paste"
command grep -qF '<fill: config-slug>' "$TMP/snap.out" || fail "placeholders must survive the real run"
command grep -q 'KV cache type\|KV q8_0' "$TMP/snap.out" || fail "fingerprint did not reach the card"
command grep -q '\[x\] capture status: OK' "$TMP/snap.out" || fail "integrity panel did not reach the card"
echo "  ✓ CARD=snapshot renders a fenced card from a real run, fingerprint + integrity intact"

# THE ROUND-TRIP. bench.sh builds the record in-process; card.sh parses one back
# out of a saved log. An A/B compares one of each, so they must agree — otherwise
# every delta is between two subtly different measurements.
[[ -s "$TMP/live-rec.kv" ]] || fail "CARD_RECORD_OUT did not write the in-process record"
card_parse_log "$TMP/snap.out" > "$TMP/parsed-rec.kv" || fail "card.sh cannot parse bench.sh's own output"
python3 - "$TMP/live-rec.kv" "$TMP/parsed-rec.kv" <<'PY' || exit 1
import sys


def load(p):
    d = {}
    for ln in open(p, encoding="utf-8"):
        if "=" in ln:
            k, v = ln.rstrip("\n").split("=", 1)
            d[k.strip()] = v.strip()
    return d


live, parsed = load(sys.argv[1]), load(sys.argv[2])
# Only fields the LOG can carry are comparable; the record also holds a few the
# printed output does not render (that asymmetry is fine and expected).
shared = sorted(set(live) & set(parsed))
assert len(shared) >= 12, f"only {len(shared)} shared fields — the parser is missing most of the record: {shared}"
bad = {k: (live[k], parsed[k]) for k in shared if live[k] != parsed[k]}
assert not bad, "in-process record and parsed-from-log record DISAGREE:\n" + "\n".join(
    f"  {k}: in-process={v[0]!r} parsed={v[1]!r}" for k, v in bad.items())
must = {"fp.model", "fp.kv", "fp.moe", "integrity.status"}
assert must <= set(shared), f"round-trip does not cover {sorted(must - set(shared))}"
print(f"  ✓ record round-trip: {len(shared)} fields agree between in-process and parsed-from-log")
PY

# A real saved run used as a BASELINE — the actual user workflow.
run_bench "$((PORT_BASE+2))" "$TMP/ab.out" CARD=ab BASELINE="$TMP/snap.out" KNOB=nothing
command grep -q '## ⚖️ bench.sh A/B' "$TMP/ab.out" \
  || { tail -30 "$TMP/ab.out" "$TMP/ab.out.err" >&2; fail "CARD=ab did not render against a real saved log"; }
command grep -q 'decode narrative' "$TMP/ab.out" || fail "A/B against a real log produced no delta row"
echo "  ✓ CARD=ab works against a previous run's saved log (the real workflow)"

# A bad BASELINE must fail the RUN loudly, not print a broken card.
run_bench "$((PORT_BASE+3))" "$TMP/abbad.out" CARD=ab BASELINE=/nonexistent/nope.log
command grep -q 'baseline unparseable' "$TMP/abbad.out" "$TMP/abbad.out.err" \
  || fail "a bad BASELINE must surface 'baseline unparseable' to the operator"
command grep -q 'summary \[narrative\]' "$TMP/abbad.out" \
  || fail "a bad BASELINE must not cost the operator the bench run itself"
echo "  ✓ bad BASELINE fails loudly without discarding the run's own measurements"

# A capture notice must never end up INSIDE a labelled field. Seen live:
#   'KV cache type  : capture: KV cache type unavailable (…)'
for f in "$TMP"/*.out; do
  if command grep -nE '^\s+[A-Za-z][A-Za-z /-]+\s+:\s+capture: ' "$f"; then
    fail "$(basename "$f"): a degradation notice was rendered inside a labelled field"
  fi
done
echo "  ✓ degradation notices never appear inside a labelled field"

# ===========================================================================
# TIER 1e — the offload block's conditionality, against a REAL run
# ===========================================================================
echo "  tier 1e: offload-block conditionality"

# (a) A real, redacted bare-metal CPU-offload run. A renderer that only ever sees
#     hand-written records drifts from what bench.sh actually prints.
real="$(card_parse_log "$CFIX/live-offload-run.log" > "$TMP/real.kv" && card_render snapshot "$TMP/real.kv")" \
  || fail "could not render a card from the real offload run fixture"
command grep -q '^### CPU offload + expert cache' <<<"$real" \
  || fail "the real offload run must render the offload block"
for want in "**Offload:** DETECTED via argv -ot" "**RSS** 139.2 GiB / 235.9 GiB total" \
            "**Cache config:** moe_cache_cap=20480" "| CUDA0 | 3038 slots · 12911 MiB · mxfp4" \
            "**PCIe** (GPU0 gen=4/4 width=16/16)" "**RAM path:** derived miss demand"; do
  command grep -qF "$want" <<<"$real" || fail "real-run card missing: $want"
done
# per-device values survive to the card, including the ones that only differ per card
command grep -q '| CUDA0 |.*\*\*60.3%\*\*.*| 42171 | 13955 |' <<<"$real" \
  || fail "CUDA0 row lost evictions/skips: $(command grep '| CUDA0 |' <<<"$real")"
command grep -q '| CUDA1 |.*\*\*80.4%\*\*' <<<"$real" || fail "CUDA1 marginal hits missing"
command grep -q 'decode rx 231 / 1941 MB/s' <<<"$real" || fail "decode PCIe phase missing"
command grep -q 'prefill rx 3031 / 28182 MB/s' <<<"$real" || fail "prefill PCIe phase missing"
# the ~13x decode-vs-prefill gap is the whole reason for the phase split
command grep -q 'Prefill PCIe traffic is lopsided: GPU0 5978 MB/s vs GPU1 85 MB/s' <<<"$real" \
  || fail "a 70x per-device prefill asymmetry must be called out"
command grep -q 'computed over the \*\*decode window\*\*' <<<"$real" \
  || fail "the RAM demand must state WHICH window it was computed over"
# An old log (written before the health-verdict correction) must still parse —
# users A/B against saved logs from previous versions.
command grep -q '^proto.warmups=' "$TMP/real.kv" \
  || fail "protocol counts not recovered from the real log"
echo "  ✓ real offload run: block filled from an actual bench.sh log (pools, PCIe phases, RAM window)"

# (b) A run with NO offload must render the GENERIC cache line and no block.
sed -e 's/^fp.offload=.*/fp.offload=not detected (moe-cache captures skipped)/' "$REC" > "$TMP/nooff.kv"
noff="$(card_render snapshot "$TMP/nooff.kv")" || fail "no-offload snapshot render failed"
command grep -q '### CPU offload + expert cache' <<<"$noff" \
  && fail "a run with no offload must NOT render the offload block"
command grep -q '\*\*Cache (if the engine has one):\*\*' <<<"$noff" \
  || fail "the non-offload case must keep the generic cache line"
command grep -q 'CUDA0 hits 19.3% → 29.6%' <<<"$noff" || fail "generic cache line lost its per-device values"
echo "  ✓ no-offload run: generic cache line, offload block absent"

# (b2) PCIe capture is UNCONDITIONAL (item 10), so a non-offload run must still
#      show it — on a multi-card TP run the all-reduce traffic IS the interesting
#      part, and that is precisely the run with no offload block to carry it.
{
  sed -e 's/^fp.offload=.*/fp.offload=not detected (moe-cache captures skipped)/' "$REC"
  printf '%s\n' \
    'pcie.link=GPU0 gen=4/4 width=16/16' \
    'pcie.link_pct=31.4' \
    'pcie.decode.all.rx_mean=812' \
    'pcie.decode.all.rx_peak=8410' \
    'pcie.prefill.all.rx_mean=1904' \
    'pcie.prefill.all.rx_peak=15220'
} > "$TMP/nooff-pcie.kv"
np="$(card_render snapshot "$TMP/nooff-pcie.kv")" || fail "no-offload+PCIe snapshot render failed"
command grep -q '### CPU offload + expert cache' <<<"$np" && fail "still no offload block expected here"
command grep -qF '**PCIe:** GPU0 gen=4/4 width=16/16' <<<"$np" \
  || fail "a non-offload card must still render PCIe (capture is unconditional): $np"
command grep -q 'decode rx 812 / 8410 MB/s mean/peak' <<<"$np" || fail "PCIe means/peaks missing"
command grep -q '31.4% of practical link' <<<"$np" || fail "link utilisation missing from the slim line"
command grep -q 'all-reduce path' <<<"$np" \
  || fail "the non-offload PCIe note should say what the traffic IS on a TP run"
# a run with no PCIe data at all must not render an empty stub
command grep -q '\*\*PCIe:\*\*' <<<"$noff" && fail "no PCIe data must render NO PCIe line, not an empty one"
echo "  ✓ non-offload card keeps PCIe (unconditional capture), and omits it entirely when absent"

# (c) A moe-cache config mismatch is a CONFOUND, exactly like a model mismatch.
sed -e 's/moe_cache_cap=8192/moe_cache_cap=4096/' "$CFIX/baseline.log" > "$TMP/moediff.log"
mc="$(card_render ab "$REC" "$TMP/moediff.log")" || fail "moe-mismatch A/B failed to render"
[[ "$(head -1 <<<"$mc")" == "> ## ⛔ CONFOUNDED" ]] \
  || fail "a moe-cache config mismatch must confound the A/B"
command grep -q '| moe-cache config.*⛔ \*\*CONFOUND\*\*' <<<"$mc" \
  || fail "the moe-cache config row must be flagged"
echo "  ✓ moe-cache config mismatch confounds the A/B"

# (d) With offload in BOTH arms, the Δ table gains the boot-comparable cache rows.
ab3="$(KNOB=moe-cache card_render ab "$REC" "$CFIX/baseline.log")"
command grep -q 'marginal hits CUDA0 (%) | 41.00 | 48.00' <<<"$ab3" \
  || fail "the A/B must compare MARGINAL hit rates per device: $(command grep 'marginal hits' <<<"$ab3")"
command grep -q 'marginal hits CUDA1 (%)' <<<"$ab3" || fail "per-device marginal rows must not be collapsed"
echo "  ✓ A/B gains per-device marginal-hit rows when both arms offloaded"

echo "test-bench-card: ok"
