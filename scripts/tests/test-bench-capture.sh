#!/usr/bin/env bash
# test-bench-capture — guards for scripts/lib/capture.sh and bench.sh's capture layer.
#
# Extends the #835 mocked-scrape pattern: TIER 0 checks that can be made from the
# source text alone, TIER 1 unit tests of every scrape against fixture logs with
# the REAL line shapes, and TIER 1 end-to-end runs of the actual bench against the
# fake llama-server.
#
# WHAT THIS SUITE IS FOR
#   The capture layer exists because a bench harness fails by printing a PLAUSIBLE
#   WRONG NUMBER, not by crashing. Every assertion below corresponds to a defect
#   that has actually shipped: a 0.00 TPS printed next to a healthy TTFT; a
#   cumulative hit rate that says a bigger pool is worse; an '[moe-cache] enabled:'
#   banner that hides a device with no pool (#824); a drafter reporting 0.992
#   acceptance while firing on a quarter of the requests.
#
# So: assert STRUCTURE, STATUS and WARNINGS — never a throughput value. Numbers
# from a fake server mean nothing.
#
# HERMETIC BY CONSTRUCTION — both are load-bearing:
#   SERVER_PID=<pid>           capture.sh otherwise runs `pgrep -x llama-server`
#                              and would adopt a production server's argv/env.
#   PREFLIGHT_NO_AUTODETECT=1  bench.sh otherwise adopts whatever container is up.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUTF8="${PYTHONUTF8:-1}"   # repo rule: locale must not decide python decoding
BENCH="$ROOT_DIR/scripts/bench.sh"
LIB="$ROOT_DIR/scripts/lib/capture.sh"
FIX="$ROOT_DIR/scripts/tests/fixtures/offload-matrix"
PORT_BASE="${TEST_PORT:-8147}"
fail() { echo "FAIL: $1" >&2; exit 1; }

for f in "$BENCH" "$LIB" "$FIX/fake-llama-server" "$FIX/fake-nvidia-smi"; do
  [[ -f "$f" ]] || fail "missing fixture or script: $f"
done

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null; true' EXIT
mkdir -p "$TMP/bin"
install -m 0755 "$FIX/fake-nvidia-smi"   "$TMP/bin/nvidia-smi"
install -m 0755 "$FIX/fake-llama-server" "$TMP/bin/fake-llama-server"
: > "$TMP/model.gguf"

# ===========================================================================
# TIER 0 — source-text guards
# ===========================================================================
echo "  tier 0: source guards"

bash -n "$BENCH" || fail "bash -n: syntax error in bench.sh"
bash -n "$LIB"   || fail "bash -n: syntax error in capture.sh"
python3 -m py_compile "$FIX/fake-llama-server" || fail "py_compile: fake-llama-server"
# bench.sh's measurement core is a python heredoc; a syntax error in it would only
# surface at run time, against a real model.
python3 - "$BENCH" <<'PY' || fail "python heredoc in bench.sh does not parse"
import ast, re, sys
src = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"<< 'PYEOF'\n(.*?)\nPYEOF\n", src, re.S)
assert m, "no PYEOF heredoc found in bench.sh"
ast.parse(m.group(1))
PY
# Inline `python3 -c '...'` snippets are the sneakiest syntax hazard in this
# codebase: a single quote INSIDE the single-quoted shell string silently ends
# it, python then gets a truncated program, and a `2>/dev/null || true` swallows
# the error completely — the capture just never prints. Parse every one of them.
python3 - "$BENCH" "$LIB" <<'PY' || fail "an inline python3 -c snippet does not parse"
import ast, re, sys
total = 0
for path in sys.argv[1:]:
    src = open(path, encoding="utf-8").read()
    for m in re.finditer(r"python3 -c '((?:[^']|\n)*?)'\s*[\"$\n]", src):
        body = m.group(1)
        if not body.strip():
            continue
        total += 1
        try:
            ast.parse(body)
        except SyntaxError as e:
            sys.exit(f"{path}: inline python3 -c does not parse ({e}):\n{body[:400]}")
        if "'" in body:
            sys.exit(f"{path}: inline python3 -c contains a single quote — it would "
                     f"terminate the shell string early:\n{body[:400]}")
assert total >= 3, f"only found {total} inline snippets — did the extraction regex rot?"
print(f"  ✓ {total} inline python3 -c snippets parse and are quote-safe")
PY
echo "  ✓ syntax (bash -n x2, py_compile, embedded-python ast)"

# `pgrep -f` / `pkill -f` match against the FULL command line, which includes the
# harness running the script — they have killed the calling shell (exit 144).
for f in "$LIB" "$BENCH"; do
  # `^[^#]*` keeps this to CODE — the rule itself is documented in comments in
  # both files, and matching those would make the guard unwritable.
  if command grep -nE '^[^#]*\b(pgrep|pkill) +-[a-zA-Z]*f' "$f"; then
    fail "$(basename "$f") uses pgrep/pkill -f — self-match footgun; use pgrep -x and kill by pid"
  fi
done
echo "  ✓ no pgrep/pkill -f (self-match footgun)"

# #809: the decode window must never be floored to an epsilon and divided by.
# `max(wall - ttft, 1e-6)` is the exact expression that printed
# `decode_TPS=690000000.00` on a block-diffusion model (#822).
# `^[^#]*` keeps this to CODE — the defect is quoted verbatim in the comment that
# explains it, and matching that would make the guard unwritable.
if command grep -nE '^[^#]*max\(wall ?- ?ttft, ?1e-6\)' "$BENCH"; then
  fail "bench.sh still floors the decode window to 1e-6 and divides by it (#809)"
fi
command grep -q 'DEGEN_WINDOW_FRAC' "$BENCH" \
  || fail "bench.sh must guard the decode window by RATIO (#809)"
echo "  ✓ no divide-by-epsilon decode window (#809)"

# #817: bench.sh has TWO print sites for the engine-log PP scrape. Both must go
# through the lib's gate — a copy of the thresholds in the python heredoc is
# exactly how the two callers of a shared scrape drifted apart before.
command grep -q 'cap_pp_plausible' "$LIB" || fail "capture.sh must define the PP plausibility gate (#817)"
command grep -q 'cap_pp_plausible' "$BENCH" || fail "bench.sh must CALL the lib gate, not reimplement it (#817)"
# `^[^#]*` keeps this to CODE — the header block names the three knobs so an
# operator can find them, which is documentation, not a second implementation.
for thresh in CAP_PP_MIN_PROMPT_TOKS CAP_PP_MIN_TPS CAP_PP_MAX_CV; do
  command grep -qE "^[^#]*${thresh}" "$BENCH" \
    && fail "bench.sh carries a copy of the PP threshold $thresh — it must live only in the lib"
done
[[ "$(command grep -c '^cap_pp_plausible()' "$LIB")" == "1" ]] \
  || fail "cap_pp_plausible must be defined exactly once"
echo "  ✓ PP plausibility policy defined once, in the lib (#817)"

# grep is shimmed to ugrep in interactive shells on some rigs, which breaks exact
# output matching. Every scrape in the lib must use `command grep`.
bare=$(command grep -nE '(^|[|;&(]|\$\()\s*grep ' "$LIB" || true)
[[ -z "$bare" ]] || fail "capture.sh uses bare grep (ugrep shim breaks exact matching):
$bare"
echo "  ✓ capture.sh uses 'command grep' throughout"

# The degradation contract: ONE line, in a fixed shape, whenever a source is
# missing. A capture that cannot run must be visibly absent, never silently zero.
command grep -q 'capture: ${thing} unavailable (${why})' "$LIB" \
  || fail "capture.sh's degradation notice no longer matches the documented shape"
echo "  ✓ degradation notice keeps the 'capture: <thing> unavailable (<why>)' shape"

# The deferral EXPIRED — offload-matrix.sh adopted the lib. Assert the adoption is
# recorded and that the old deferral note is gone, so nobody re-reads a stale
# "duplication is accepted" note as licence to fork a scrape again.
command grep -q 'adoption is DEFERRED' "$LIB" \
  && fail "the deferral note outlived the adoption — offload-matrix.sh now consumes the lib"
command grep -q 'BOTH scripts consume this lib' "$LIB" \
  || fail "capture.sh must record that both callers consume it (and that copies were DELETED)"
command grep -q 'Fork the statistic, never the scrape' "$LIB" \
  || fail "the lib must record the rule that keeps the two callers from re-forking"
echo "  ✓ lib records the completed adoption (both callers, copies deleted)"

# Sourcing the lib must have no side effects — no output, no exit, and safe to
# source twice (bench.sh and a future sweep may both pull it in).
out=$(bash -c "set -euo pipefail; source '$LIB'; source '$LIB'; echo READY" 2>&1)
[[ "$out" == "READY" ]] || fail "sourcing capture.sh is not side-effect-free / idempotent: $out"
echo "  ✓ capture.sh sources cleanly and idempotently"

command grep -q 'source "${ROOT_DIR}/scripts/lib/capture.sh"' "$BENCH" \
  || fail "bench.sh does not source the shared capture lib"
echo "  ✓ bench.sh consumes the shared lib"

# ===========================================================================
# TIER 1a — unit tests of every scrape, against the REAL line shapes
# ===========================================================================
echo "  tier 1a: scrape units"

# Two snapshots of a log whose hit rate CHANGES: a cold-fill phase followed by a
# steady state. Constructed so cumulative and marginal DISAGREE — the whole reason
# the marginal rate exists (item 3b). Also carries the measured per-device
# asymmetry (~48% / ~58% on a layer-split MoE), which averaging would destroy.
cat > "$TMP/start.log" <<'EOF'
llama_context: n_ctx_seq = 262144
CUDA_Host model buffer size = 98304.00 MiB
[moe-cache] enabled: reserve=1536 MiB admit=1/64
[moe-cache] CUDA0 pool[0]: type=mxfp4 expert=4352 KiB slots=3038 total=12911 MiB
[moe-cache] CUDA0 pool[1]: type=mxfp4 expert=4352 KiB slots=1000 total=4250 MiB
[moe-cache] CUDA1 pool[0]: type=mxfp4 expert=4352 KiB slots=2500 total=10625 MiB
[moe-cache] CUDA0 hits=10000/200000 evictions=5 skips=0 fill-fail=0 dispatch-fail=0 collect-fail=0
[moe-cache] CUDA1 hits=12000/200000 evictions=6 skips=0 fill-fail=0 dispatch-fail=0 collect-fail=0
EOF
cp "$TMP/start.log" "$TMP/end.log"
cat >> "$TMP/end.log" <<'EOF'
prompt eval time =  5000.00 ms /  10000 tokens (    0.50 ms per token,  2000.00 tokens per second)
       eval time = 10000.00 ms /    300 tokens (   33.33 ms per token,    30.00 tokens per second)
       eval time =   500.00 ms /     16 tokens (   31.25 ms per token,    32.00 tokens per second)
draft acceptance = 0.992 (  248 accepted /   250 generated)
[moe-cache] CUDA0 hits=58000/300000 evictions=25 skips=0 fill-fail=0 dispatch-fail=0 collect-fail=0
[moe-cache] CUDA1 hits=70000/300000 evictions=26 skips=0 fill-fail=3 dispatch-fail=0 collect-fail=0
EOF

# shellcheck source=../lib/capture.sh
source "$LIB"

# --- census: a device can hold MULTIPLE pools; slots and bytes AGGREGATE ------
cen=$(cap_moe_parse census "$TMP/end.log") || fail "census scrape returned nothing"
command grep -q 'CUDA0 pools=2 slots=4038 total_mib=17161' <<<"$cen" \
  || fail "census must SUM every pool on a device (3038+1000=4038 slots), got:
$cen"
command grep -q 'CUDA1 pools=1 slots=2500' <<<"$cen" || fail "census lost CUDA1: $cen"
command grep -q 'expert_kib=4352' <<<"$cen" || fail "census did not read expert size: $cen"
echo "  ✓ pool census aggregates multiple pools per device, keeps devices separate"

# --- #824: detection must count DEVICES WITH A POOL, not pool lines ----------
pd=$(cap_moe_parse pooldev "$TMP/end.log") || fail "pooldev scrape returned nothing"
command grep -q 'devices=2' <<<"$pd" || fail "pooldev device count wrong: $pd"
# The generalised trap: two pools on ONE device makes pool_lines >= NGPU while a
# whole device holds nothing. A count-of-lines detector passes that; only a
# count-of-devices detector catches it.
cat > "$TMP/halfcached.log" <<'EOF'
[moe-cache] enabled: reserve=1536 MiB admit=1/64
[moe-cache] CUDA0 has no cache budget after 1536 MiB reserve
[moe-cache] CUDA1 pool[0]: type=mxfp4 expert=4352 KiB slots=3038 total=12911 MiB
[moe-cache] CUDA1 pool[1]: type=mxfp4 expert=4352 KiB slots=1000 total=4250 MiB
EOF
pd2=$(cap_moe_parse pooldev "$TMP/halfcached.log") || fail "pooldev failed on half-cached log"
command grep -q 'devices=1' <<<"$pd2" \
  || fail "half-cached log must report devices=1 (only CUDA1 has a pool), got: $pd2"
command grep -q 'pool_lines=2' <<<"$pd2" || fail "expected pool_lines=2 in: $pd2"
command grep -q 'enabled=1' <<<"$pd2" \
  || fail "the 'enabled:' banner must still be reported — that it prints in this state IS the #824 hole"
echo "  ✓ #824: half-cached run reports devices=1 while pool_lines=2 and enabled=1"

# --- marginal vs cumulative MUST disagree ------------------------------------
cap_moe_parse counters "$TMP/start.log" > "$TMP/c0" || fail "counters scrape (start) failed"
cap_moe_parse counters "$TMP/end.log"   > "$TMP/c1" || fail "counters scrape (end) failed"
mar=$(cap_marginal_rates "$TMP/c0" "$TMP/c1") || fail "marginal-rate derivation failed"
command grep -q 'CUDA0 marginal=48.0% cumulative=19.3%' <<<"$mar" \
  || fail "CUDA0 marginal rate wrong (expect 48.0% marginal vs 19.3% cumulative), got:
$mar"
command grep -q 'CUDA1 marginal=58.0% cumulative=23.3%' <<<"$mar" \
  || fail "CUDA1 marginal rate wrong (expect 58.0% vs 23.3%), got:
$mar"
# If these ever agree the fixture stopped exercising the defect this exists for.
python3 - <<PY || fail "marginal and cumulative agree — the fixture no longer covers item 3b"
import re, sys
t = """$mar"""
for ln in t.splitlines():
    m = re.search(r"marginal=([\d.]+)% cumulative=([\d.]+)%", ln)
    if m and abs(float(m.group(1)) - float(m.group(2))) < 5:
        sys.exit(1)
PY
echo "  ✓ marginal rate derived per device, and disagrees with cumulative (48/58 vs 19/23)"

# --- ABSENCE OF MEASUREMENT != a measured zero (club-3090 #1105/#1106) --------
# Two field reports rendered `cumulative=0.0% dhits=0 dlookups=0` and were read
# (by us) as "the cache did nothing". Both were actually NO-MEASUREMENT runs:
#   #1106 the parsed `hits=` line came from a session whose container had EXITED
#         HOURS EARLIER — the counters never moved during the bench.
#   #1105 the cache never allocated at all.
# The engine's own comment names the class: "an ABSENCE OF MEASUREMENT reported
# as data". Guard both halves.
#
CF_STATS="$ROOT_DIR/models/deepseek-v4-flash-0731/llamacpp-club3090/compose/dual/unsloth-q8-kxl/moecache.yml"
# NEGATIVE CONTROL FIRST: identical snapshots => lookups do not advance.
cat > "$TMP/stale0.kv" <<'EOF'
CUDA0 hits=0 total=12 evictions=0 skips=0 admission=12 fill_fail=0 dispatch_fail=0 collect_fail=0
EOF
cp "$TMP/stale0.kv" "$TMP/stale1.kv"
stale=$(cap_marginal_rates "$TMP/stale0.kv" "$TMP/stale1.kv")   || fail "marginal derivation must still succeed on a stale pair"
command grep -q 'dlookups=0' <<<"$stale"   || fail "a stale (non-advancing) counter pair must report dlookups=0, got: $stale"
# The renderer's decision hinges on summed dlookups — assert the arithmetic the
# guard in bench.sh performs, so a change to the field name breaks HERE.
dlk=$(printf '%s\n' "$stale" | awk '{
  for (i = 1; i <= NF; i++) if ($i ~ /^dlookups=/) { split($i, a, "="); t += a[2] }
} END { print t + 0 }')
[[ "$dlk" == "0" ]] || fail "summed dlookups on a stale pair must be 0, got: $dlk"

# POSITIVE CONTROL: an advancing pair must NOT trip the guard, or the warning
# would fire on every healthy run and be trained away as noise.
dlk_live=$(printf '%s\n' "$mar" | awk '{
  for (i = 1; i <= NF; i++) if ($i ~ /^dlookups=/) { split($i, a, "="); t += a[2] }
} END { print t + 0 }')
[[ "$dlk_live" -gt 0 ]] || fail "the LIVE fixture must advance lookups, got: $dlk_live"

# bench.sh must actually branch on it, and must not print a rate headline when
# there is no measurement.
command grep -q 'NO MEASUREMENT IN THIS RUN' "$ROOT_DIR/scripts/bench.sh"   || fail "bench.sh lost the no-measurement branch"
command grep -q 'UNSCOPED — may belong to another session' "$ROOT_DIR/scripts/bench.sh"   || fail "bench.sh must label unscoped counters rather than presenting them as this run's"
command grep -q 'MOE_STATS=200 AND LLAMA_ARG_LOG_VERBOSITY=4' "$ROOT_DIR/scripts/bench.sh" || fail "bench.sh must name BOTH knobs - STATS alone emits nothing (measured 2026-08-26: STATS=200 at default verbosity gave 0 [moe-cache] lines)"
# The branch must NOT assert CACHE_DISABLED from an absence. cap_status_classify
# turns cache_ok=0 into that status, which claims the cache is OFF when all we
# know is that this run did not measure it - the same error the branch prevents.
# Assert the CODE, not the comment: scan the no-measurement branch BODY for a
# live CAP_CACHE_OK=0. Grepping the explanatory comment would pass even if the
# assignment came back — a gate guarding prose instead of behaviour.
python3 - "$ROOT_DIR/scripts/bench.sh" <<'PYEOF' || fail "the no-measurement branch must not flip CAP_CACHE_OK (it renders as CACHE_DISABLED, asserting the cache is OFF when it was merely unmeasured)"
import sys
src = open(sys.argv[1], encoding="utf-8").read()
i = src.index("NO MEASUREMENT IN THIS RUN")
start = src.rindex('if [[ "${_dlk:-0}" -eq 0 ]]; then', 0, i)
body = src[start:src.index("raw (UNSCOPED", i)]
live = [l for l in body.splitlines()
        if "CAP_CACHE_OK=0" in l and not l.lstrip().startswith("#")]
sys.exit(1 if live else 0)
PYEOF
# And the compose must document the co-requirement, or the next reader re-runs
# the same dead-feature boot we just did.
command grep -q 'THIS VAR ALONE IS NOT ENOUGH' "$CF_STATS" || fail "moecache.yml must document that verbosity 4 is ALSO required"
echo "  ✓ no-measurement guard: stale pair flagged, live pair untouched, bench.sh branches"

# --- the compose must be ABLE to deliver the telemetry ------------------------
# Root cause of the whole class: the var was absent from `environment:`, so docker
# never forwarded it and NO caller could switch the instrument on. A regex over
# the compose is the only cheap guard — the delivery path is docker's, not ours.
# EVERY compose that enables the cache, not just the one that was being debugged:
# the var was absent from ALL of them, and a gate that pins one path would have
# gone green while four composes stayed unreachable.
mapfile -t CACHE_COMPOSES < <(command grep -rl -- '--moe-cache' "$ROOT_DIR"/models/*/*/compose/*/*/*.yml 2>/dev/null || true)
[[ ${#CACHE_COMPOSES[@]} -ge 5 ]] \
  || fail "expected >=5 moe-cache composes, found ${#CACHE_COMPOSES[@]} — did the glob or the layout change?"
for CF in "${CACHE_COMPOSES[@]}"; do
  # Must be DECLARED (docker forwards only what is declared) and OVERRIDABLE.
  # ⚠️ The default is deliberately 0 as of 2026-08-26: telemetry is OPT-IN so
  # instrumentation does not run when nothing reads it. (An earlier comment here
  # claimed a ~6% decode cost; RETRACTED — run-to-run variance on the SHIPPED
  # image alone was 8-9%, and the arithmetic puts the counters at ~1e-4 % of
  # wall time.) Do NOT "fix" this back to a non-zero default.
  command grep -qE '^\s+- GGML_CUDA_MOE_CACHE_STATS=\$\{MOE_STATS:-[0-9]+\}' "$CF" \
    || fail "$(basename "$(dirname "$CF")")/$(basename "$CF"): needs a declared, overridable GGML_CUDA_MOE_CACHE_STATS=\${MOE_STATS:-N}"
  command grep -qE '^\s+- GGML_CUDA_MOE_CACHE_STATS=\$\{MOE_STATS:-0\}' "$CF" \
    || fail "$(basename "$(dirname "$CF")")/$(basename "$CF"): telemetry must default OFF (0) — it costs ~6% decode when live"
  command grep -qE '^\s+- LLAMA_ARG_LOG_VERBOSITY\s*$' "$CF" \
    || fail "$(basename "$(dirname "$CF")")/$(basename "$CF"): needs the LLAMA_ARG_LOG_VERBOSITY passthrough — STATS alone emits NOTHING (verbosity <4 drops every [moe-cache] line)"
done
echo "  ✓ all ${#CACHE_COMPOSES[@]} moe-cache composes declare both telemetry knobs"
CF="$ROOT_DIR/models/deepseek-v4-flash-0731/llamacpp-club3090/compose/dual/unsloth-q8-kxl/moecache.yml"
if [[ -f "$CF" ]]; then
  command grep -qE '^\s+- GGML_CUDA_MOE_CACHE_STATS' "$CF"     || fail "moecache.yml must declare GGML_CUDA_MOE_CACHE_STATS — undeclared vars are NOT forwarded by docker"
  # Non-empty default required: engine `stats_every` defaults to 0, so a bare
  # passthrough would leave every community report with no telemetry again.
  command grep -qE '^\s+- GGML_CUDA_MOE_CACHE_STATS=\$\{MOE_STATS:-0\}' "$CF" || fail "$(basename "$(dirname "$CF")")/$(basename "$CF"): telemetry must default OFF (MOE_STATS:-0) — opt-in, so instrumentation does not run when nothing reads it"
  echo "  ✓ compose declares the stats var with a non-zero default (delivery path exists)"
fi

# --- health counters: all-zero is the pass condition (item 3c) ---------------
[[ "$(cap_health_verdict "$TMP/c0")" == PASS* ]] || fail "clean counters should PASS"
v=$(cap_health_verdict "$TMP/c1")
[[ "$v" == FAIL* ]] || fail "a non-zero fill-fail must FAIL the health panel, got: $v"
command grep -q 'CUDA1' <<<"$v" || fail "health verdict must name the offending device: $v"
# ⚠ `skips` is NOT a hard failure. It counts admission throttling (insert queue
#   exhausted under the configured THROTTLE), so under a low ADMIT_AFTER a non-zero
#   value is normal steady-state pressure. A live run on a HEALTHY server showed
#   skips at 0.5% of misses — the original "all counters zero" spec would have
#   failed it. Report it as load, never as a defect.
cat > "$TMP/skips.kv" <<'EOF'
CUDA0 hits=100 total=200 evictions=5 skips=13955 admission=4000 fill_fail=0 dispatch_fail=0 collect_fail=0
CUDA1 hits=120 total=200 evictions=6 skips=2330 admission=900 fill_fail=0 dispatch_fail=0 collect_fail=0
EOF
v=$(cap_health_verdict "$TMP/skips.kv")
[[ "$v" == PASS* ]] || fail "non-zero skips with zero hard failures must PASS, got: $v"
command grep -q 'informational' <<<"$v" || fail "skips must still be REPORTED as a load metric: $v"
command grep -q 'skips=13955' <<<"$v" || fail "the informational tail must carry the value: $v"
command grep -q 'THROTTLE tuning signal, not a defect' <<<"$v" \
  || fail "the verdict must say why skips is not a defect: $v"
# ...and each hard-failure counter on its own must still fail.
for hard in fill_fail dispatch_fail collect_fail; do
  printf 'CUDA0 hits=1 total=2 skips=99 %s=1\n' "$hard" > "$TMP/hard.kv"
  [[ "$(cap_health_verdict "$TMP/hard.kv")" == FAIL* ]] \
    || fail "a non-zero $hard must FAIL the health panel"
done
echo "  ✓ health: hard failures FAIL, skips/admission report as load (the corrected pass condition)"

# --- acceptance WITH fire rate ----------------------------------------------
acc=$(cap_moe_parse acceptance "$TMP/end.log") || fail "acceptance scrape returned nothing"
command grep -q 'fired=1' <<<"$acc" || fail "acceptance fire count wrong: $acc"
command grep -q 'mean=0.992' <<<"$acc" || fail "acceptance value wrong: $acc"
command grep -q 'accepted=248 drafted=250' <<<"$acc" \
  || fail "drafts-attempted must be captured alongside acceptance: $acc"
# BOTH statistics, because the two callers need different ones: bench.sh quotes the
# MEAN over its run, while offload-matrix's TSV `accept` column has always meant the
# arm's LAST logged value. Forking the statistic in the lib is what stops the two
# from re-forking the whole scrape.
command grep -q 'last=0.992' <<<"$acc" || fail "acceptance must expose last= for the sweep: $acc"
cat >> "$TMP/end.log" <<'EOF'
draft acceptance = 0.500 (  100 accepted /   200 generated)
EOF
acc2=$(cap_moe_parse acceptance "$TMP/end.log")
command grep -q 'last=0.500' <<<"$acc2" || fail "last= must track the FINAL value: $acc2"
command grep -q 'mean=0.746' <<<"$acc2" || fail "mean= must average both values: $acc2"
echo "  ✓ acceptance exposes last= (sweep) and mean= (bench) from one scrape"

# --- timings: prefill and decode must not be confused; short runs filtered ----
tim=$(cap_moe_parse timings "$TMP/end.log") || fail "timings scrape returned nothing"
# The two prefill populations (short canonical prompts vs deep haystacks) differ by
# ~4x, so they are reported separately rather than averaged into one meaningless mean.
command grep -q '^prefill_deep(>=2000tok) n=1 mean=2000.00' <<<"$tim" \
  || fail "deep-prefill timing mis-parsed: $tim"
command grep -q '^prefill_short(<2000tok) n=0' <<<"$tim" \
  || fail "the short-prefill population must be reported even when empty: $tim"
command grep -q '^decode n=2' <<<"$tim" || fail "decode timing mis-parsed: $tim"
tim2=$(CAP_MIN_DECODE_TOKENS=32 cap_moe_parse timings "$TMP/end.log")
command grep -q '^decode n=1 ' <<<"$tim2" \
  || fail "the 16-token completion should be excluded at floor=32: $tim2"
command grep -q 'dropped_short_decode n=1 floor=32' <<<"$tim2" \
  || fail "dropping a short completion must be REPORTED, not silent: $tim2"
echo "  ✓ timings split prefill/decode and report (not hide) short-completion filtering"

# --- always-on fingerprint pieces -------------------------------------------
ARGV="python3 /x/llama-server -m /models/m.gguf -ot blk.(1|2).ffn_up_exps.weight=CPU -t 24 -ub 2048 -ct q8_0 -ngl 99 -ts 1,1 --moe-cache 8192"
fp=$(cap_argv_fingerprint "$ARGV") || fail "argv fingerprint returned nothing"
for want in "kv=q8_0" "threads=24" "ngl=99" "split=1,1" "ubatch=2048"; do
  command grep -q -- "$want" <<<"$fp" || fail "argv fingerprint missing $want: $fp"
done
[[ "$(cap_kv_type "$ARGV")" == "q8_0 (source: argv)" ]] || fail "KV type not read from argv"
# With no -ctk/-ctv the llama.cpp default is f16 — a fact the reader can act on,
# where "unavailable" is not. Only inferred once the engine family is identified.
CAP_LOG="$TMP/end.log" v="$(cap_kv_type 'llama-server -m /m/x.gguf -ngl 99' || true)"
[[ "$v" == "f16 (engine default; no -ctk/-ctv in argv)" ]] \
  || fail "a llama.cpp run with no KV flag should report the engine default, got: '$v'"
CAP_LOG="" CAP_PROPS_TRIED=1 CAP_PROPS="" v="$(cap_kv_type '' 2>/dev/null || echo UNAVAILABLE)"
[[ "$v" == "UNAVAILABLE" ]] || fail "with no argv and no identifiable engine, KV must stay unavailable, got: '$v'"
command grep -q 'argv -ot' <<<"$(cap_offload_detected "$ARGV")" || fail "-ot must trigger offload detection"
command grep -q 'moe_cache_cap=8192' <<<"$(cap_moe_cache_config "$ARGV")" \
  || fail "--moe-cache cap must be captured (the census self-limits below it)"
# --n-cpu-moe is the other offload spelling, and a CUDA_Host buffer line is the
# third signal for a server whose argv we cannot read.
command grep -q 'n-cpu-moe' <<<"$(cap_offload_detected 'llama-server --n-cpu-moe 20')" \
  || fail "--n-cpu-moe must trigger offload detection"
CAP_LOG="$TMP/end.log" && command grep -q 'CUDA_Host' <<<"$(cap_offload_detected '')" \
  || fail "a CUDA_Host model buffer line must trigger offload detection"
echo "  ✓ argv/env fingerprint: KV type, offload signals (x3), moe-cache cap"

# --- derived RAM demand + status enum + divergence ---------------------------
# 94001 misses x 4352 KiB / 1024 / 60 s = 6658 MB/s
[[ "$(cap_ram_rd_mbps 94001 4352 60)" == "6658" ]] || fail "derived ram_rd arithmetic changed"
cap_ram_rd_mbps 0 4352 60 >/dev/null 2>&1 && fail "ram_rd must refuse to invent a value from zero misses"
[[ "$(cap_status_classify 0 0 0 0 1 0)"     == NO_TOKENS ]]      || fail "zero tokens must be NO_TOKENS"
[[ "$(cap_status_classify 500 0 0 0 1 0)"   == NO_TOKENS ]]      || fail "zero TPS must be NO_TOKENS"
[[ "$(cap_status_classify 500 30 2 0 1 0)"  == REQ_ERRORS ]]     || fail "request errors must outrank OK"
[[ "$(cap_status_classify 500 30 0 1 0 0)"  == CACHE_DISABLED ]] || fail "unallocated requested cache must not read OK"
[[ "$(cap_status_classify 500 30 0 0 1 3)"  == INVALID_BYPASS ]] || fail "bypass>0 must not read OK"
[[ "$(cap_status_classify 500 30 0 0 1 0)"  == OK ]]             || fail "a clean run must read OK"
# NO_TOKENS outranks the cache states: a run with no tokens says nothing about a cache.
[[ "$(cap_status_classify 0 0 0 1 0 5)"     == NO_TOKENS ]]      || fail "NO_TOKENS must outrank the cache states"
# ...and REQ_ERRORS outranks NO_TOKENS in turn: a run that errored EXPLAINS its own
# zero, and reporting NO_TOKENS there sends the reader hunting a scrape bug that is
# not the cause. This is offload-matrix's original precedence, adopted verbatim so
# per-arm rows keep their meaning.
[[ "$(cap_status_classify 0 0 3 0 1 0)"     == REQ_ERRORS ]]     || fail "REQ_ERRORS must outrank NO_TOKENS"
[[ "$(cap_status_classify 0 0 3 1 0 5)"     == REQ_ERRORS ]]     || fail "REQ_ERRORS must outrank everything"
[[ "$(cap_divergence_pct 30 32)" == "6.2" ]] || fail "divergence arithmetic changed"
echo "  ✓ status enum precedence, derived-demand arithmetic, divergence"

# --- PP plausibility gate against the REAL reported samples (#817) -----------
# Every row below is a figure a human actually pasted into an issue as data.
pp_case() {   # $1=expect(plausible|suppress) $2=mean $3=cv $4=prompt_toks $5=source
  local out rc
  # `set -e` aborts on a failing command substitution in an assignment, and this
  # gate signals "suppress" with a non-zero exit BY DESIGN.
  out="$(cap_pp_plausible "$2" "$3" "$4")" && rc=0 || rc=$?
  if [[ "$1" == "plausible" ]]; then
    (( rc == 0 )) || fail "$5: mean=$2 CV=$3% prompt=$4 should PRINT, gate said: $out"
    [[ -z "$out" ]] || fail "$5: a plausible sample must print no reason, got: $out"
  else
    (( rc == 1 )) || fail "$5: mean=$2 CV=$3% prompt=$4 must be SUPPRESSED — it was printed as data"
    [[ -n "$out" ]] || fail "$5: a suppressed sample must say WHY (a number that vanishes silently is its own defect)"
  fi
}
pp_case suppress  2.00    55.9  15     "#769 — 65-char prompt"
pp_case suppress  6.88    19.7  15     "#822 [code] shape"
pp_case suppress  7.60    40.9  15     "#822 [narrative] shape"
pp_case suppress  6.07    49.5  10000  "#822 prefill-10k windowed — long prompt, absurd rate"
pp_case suppress  1127.30 171.9 90000  "#817 windowed-in-prefill — passes mean AND depth, CV is the only tell"
pp_case plausible 9968.67 14.0  90000  "#822 prefill-90k windowed — the one that IS a rate"
pp_case plausible 3950.40 0.0   9876   "PP fallback shape"
# ...and the CV condition must be load-bearing on its own, or the #817 case above
# would print: it clears both the depth and the rate floor.
[[ "$(CAP_PP_MAX_CV=1000 cap_pp_plausible 1127.30 171.9 90000; echo rc=$?)" == "rc=0" ]] \
  || fail "the CV condition is not what suppresses the windowed-in-prefill sample"
echo "  ✓ PP gate: every figure pasted as data in #769/#822 is suppressed, with a reason"

# ===========================================================================
# TIER 1b — end-to-end bench.sh against the fake server
# ===========================================================================
echo "  tier 1b: end-to-end bench runs (no GPU, no model)"

SRV_PID=""
SRV_LOG=""
# ⚠ TWO hermeticity traps here, both of which produced a run that measured the
# RIG'S REAL SERVER instead of the fake:
#   1. NEVER call this through a command substitution — `$( )` is a subshell, so
#      SRV_PID would not propagate out.
#   2. NEVER trust `$!` for the server pid. The shebang plus the `env` prefix
#      re-execs, and the pid bash recorded can already be gone while the server
#      runs under another. The fixture writes its OWN pid to FAKE_PID_FILE.
# Either way, capture.sh with an empty/stale SERVER_PID falls back to
# `pgrep -x llama-server` and adopts a production server's argv and env.
start_server() {   # $1=scenario $2=port [extra env as VAR=VAL ...]
  local scen="$1" port="$2"; shift 2
  SRV_LOG="$TMP/server-$scen-$port.log"
  local pidfile="$TMP/server-$scen-$port.pid"
  rm -f "$pidfile"
  env FAKE_SCENARIO="$scen" FAKE_NGPU=2 FAKE_TOKENS=40 FAKE_PID_FILE="$pidfile" "$@" \
      GGML_CUDA_MOE_CACHE_RESERVE_MB=1536 GGML_CUDA_MOE_CACHE_ADMIT_AFTER=1/64 \
      GGML_CUDA_MOE_CACHE_MAX_BATCH=8 GGML_CUDA_MOE_CACHE_STATS=1000 \
      "$TMP/bin/fake-llama-server" -m "$TMP/model.gguf" \
        -ot 'blk.(1|3|5).ffn_.*_exps.weight=CPU' -t 24 -ub 2048 -ct q8_0 -ngl 99 -ts 1,1 \
        --moe-cache 8192 --host 127.0.0.1 --port "$port" > "$SRV_LOG" 2>&1 &
  local i
  for i in $(seq 1 40); do
    curl -sf -m 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    sleep 0.25
  done
  curl -sf -m 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
    || { sed -n '1,15p' "$SRV_LOG" >&2; fail "$scen: fake server never became ready on $port"; }
  SRV_PID="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$SRV_PID" && -r "/proc/$SRV_PID/cmdline" ]] \
    || fail "$scen: fake server did not report a readable pid — the run would not be hermetic"
  # let the periodic cache stats emit at least one cumulative sample first, so the
  # marginal derivation has a genuine "before" to difference against
  sleep 2
}

run_bench() {   # $1=scenario $2=port $3=outfile [extra bench env ...]
  local scen="$1" port="$2" out="$3"; shift 3
  start_server "$scen" "$port"
  PATH="$TMP/bin:$PATH" PREFLIGHT_NO_AUTODETECT=1 CONTAINER=none \
    SERVER_LOG="$SRV_LOG" SERVER_PID="$SRV_PID" URL="http://127.0.0.1:$port" \
    MODEL=fake-model RUNS=2 WARMUPS=1 PREFILL_PROBE=0 \
    env "$@" timeout 300 bash "$BENCH" > "$out" 2>"${out}.err" || true
  kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=""
  # The fingerprint must describe the server this run actually drove. If the pid
  # ever fails to reach capture.sh again, this catches it on the next run instead
  # of silently reporting a production config.
  # Skipped when the run deliberately disabled /proc inspection (SERVER_PID=none):
  # there is no pid to adopt, so there is nothing to get wrong.
  if ! command grep -q 'serving pid    : unknown' "$out" \
     && command grep -q 'moe-cache cfg' "$out" \
     && ! command grep -q 'moe_cache_cap=8192' "$out"; then
    command grep -m1 'moe-cache cfg' "$out" >&2
    fail "$scen: fingerprint did not describe the FAKE server — the run was not hermetic"
  fi
}

# --- 1. healthy: the always-on captures all fire ----------------------------
run_bench healthy "$((PORT_BASE))" "$TMP/healthy.out"
H="$TMP/healthy.out"
for want in \
  "CONFIG FINGERPRINT" "KV cache type  : q8_0" "CPU offload    : DETECTED" \
  "moe-cache cfg  : moe_cache_cap=8192" "RESERVE_MB=1536" "ADMIT_AFTER=1/64" \
  "SYSTEM RAM" "mem_total_gib=" "swap check" \
  "CAPTURE: PCIe" "CAPTURE: VRAM" "CAPTURE: EXPERT CACHE" "CAPTURE: INTEGRITY" \
  "serving argv   :" "served ctx     :"; do
  command grep -q "$want" "$H" || { sed -n '1,40p' "$H" >&2; fail "healthy run missing: $want"; }
done
command grep -q 'status: OK' "$H" || fail "healthy run should classify OK: $(command grep -A2 'CAPTURE: INTEGRITY' "$H")"
# per-device values must survive into the OUTPUT, not just detection (item 3d)
command grep -q 'CUDA0 marginal=' "$H" || fail "per-device marginal rate missing from output"
command grep -q 'CUDA1 marginal=' "$H" || fail "the CUDA0/CUDA1 split was averaged away — item 3d"
command grep -q 'CUDA0 pools=' "$H" || fail "per-device pool census missing from output"
# #1137: these are TWO different failures and used to produce one message. If the
# derivation was skipped, say which input was empty — do not report it as missing
# caveat TEXT, which points at formatting and hides the real cause.
if command grep -q 'derived host-RAM read demand: NOT DERIVED' "$H"; then
  fail "the RAM derivation was SKIPPED, so its caveat is legitimately absent: $(
        command grep -m1 'NOT DERIVED' "$H")"
fi
command grep -q 'DERIVED, NOT A PERF COUNTER' "$H" \
  || fail "the derived RAM figure must carry its 'not a counter' caveat"
command grep -q 'MARGINAL' "$H" || fail "the marginal-vs-cumulative caveat is missing"
echo "  ✓ healthy: fingerprint + RAM + PCIe + VRAM + per-device cache + OK status"

# --- 2. NO_TOKENS: the loud warn, and the engine-side cross-check ------------
run_bench no_tokens "$((PORT_BASE+1))" "$TMP/notok.out"
N="$TMP/notok.out"
command grep -q 'status: NO_TOKENS' "$N" \
  || { sed -n '/INTEGRITY/,$p' "$N" >&2; fail "a run that produced no tokens must classify NO_TOKENS"; }
command grep -q 'NO_TOKENS: the run measured' "$N" || fail "NO_TOKENS must warn LOUDLY, not just set a status"
command grep -q 'SCRAPE failure, not a' "$N" \
  || fail "the NO_TOKENS warning must name the failure class (scrape, not a slow model)"
command grep -q 'Do NOT quote any throughput number' "$N" || fail "NO_TOKENS must tell the reader not to quote the run"
echo "  ✓ no_tokens: NO_TOKENS status + loud warning naming the failure class"

# --- 3. engine-side cross-check divergence (item 1b) ------------------------
command grep -q 'CAPTURE: ENGINE-SIDE TIMINGS' "$H" || fail "engine-side timing section missing"
command grep -q 'cross-check: client-observed' "$H" \
  || fail "SERVER_LOG was set — the engine-side cross-check must run (item 4 + item 1b)"
command grep -qE 'divergence' "$H" || fail "divergence must be reported"
# the fake's engine-side rate is deliberately far from the client-observed one
command grep -q '⚠ WARN: >5% divergence' "$H" \
  || fail "a >5% client-vs-engine divergence must WARN: $(command grep -A2 'cross-check' "$H")"
echo "  ✓ engine-side scrape from SERVER_LOG + >5% divergence warning"

# --- 4. short EOS: n-usable per shape (item 2) ------------------------------
run_bench short_eos "$((PORT_BASE+2))" "$TMP/short.out"
S="$TMP/short.out"
command grep -q 'n-usable' "$S" || fail "n-usable must be reported per shape"
command grep -q 'EOSed well below max_tokens' "$S" \
  || fail "runs EOSing far below max_tokens must WARN — a 5-run shape silently becomes n=1"
command grep -q 'treat the CV as unreliable' "$S" || fail "the short-EOS warning must discredit the CV"
command grep -q '⚠ n-usable' "$S" || fail "n-usable belongs on the integrity panel too"
echo "  ✓ short_eos: n-usable per shape + warning that the CV is untrustworthy"

# --- 5. acceptance WITH fire rate (item 11a) --------------------------------
run_bench low_fire "$((PORT_BASE+3))" "$TMP/fire.out"
F="$TMP/fire.out"
command grep -q 'CAPTURE: DRAFT ACCEPTANCE' "$F" || fail "acceptance section missing"
command grep -qE 'mean=0\.99' "$F" || fail "high acceptance value not reported: $(command grep -A3 'DRAFT ACCEPT' "$F")"
command grep -q 'fire rate: fired on' "$F" \
  || fail "acceptance without a fire rate is a deception — both must be reported"
command grep -q 'acceptance WITHOUT a fire rate is a deception' "$F" \
  || fail "a drafter idle on most requests must WARN: $(command grep -A6 'DRAFT ACCEPT' "$F")"
# The fire rate is a fraction of requests, so it can never exceed 100%.
python3 - "$F" <<'PY' || fail "fire rate exceeded 100% — numerator and denominator come from different windows"
import re, sys
for ln in open(sys.argv[1], encoding="utf-8"):
    m = re.search(r"fire rate: fired on (\d+) of (\d+) requests .*?\(([\d.]+)%\)", ln)
    if m:
        assert float(m.group(3)) <= 100.0, ln
        assert int(m.group(1)) <= int(m.group(2)), ln
PY
echo "  ✓ low_fire: high acceptance + low fire rate BOTH reported, and warned about"

# --- 6. graceful degradation: none of the sources present -------------------
# docker-less, GPU-less, no server log, no readable pid. bench.sh must still run
# end to end and say exactly what it could not capture.
run_bench healthy "$((PORT_BASE+4))" "$TMP/degraded.out" \
  SERVER_LOG= SERVER_PID=none PATH="/usr/bin:/bin"
D="$TMP/degraded.out"
command grep -q 'CONFIG FINGERPRINT' "$D" || fail "degraded run did not complete its fingerprint section"
command grep -q 'CAPTURE: INTEGRITY' "$D" \
  || { tail -25 "$D" "$D.err" >&2; fail "degraded run did not reach the end — degradation is not graceful"; }
command grep -qE '^capture: .* unavailable \(.*\)$' "$D" \
  || fail "a missing source must produce a 'capture: <thing> unavailable (<why>)' line"
# ...and each missing thing exactly once, never in a loop.
python3 - "$D" <<'PY' || fail "a degradation notice was printed more than once"
import collections, re, sys
c = collections.Counter(
    re.match(r"capture: (.*?) unavailable", ln).group(1)
    for ln in open(sys.argv[1], encoding="utf-8") if ln.startswith("capture: "))
dupes = {k: v for k, v in c.items() if v > 1}
assert not dupes, dupes
PY
command grep -q 'no CPU offload detected' "$D" \
  || fail "with no argv and no log, the moe-cache capture must be SKIPPED with a reason"
echo "  ✓ degradation: run completes, one notice per missing source, cache capture skipped with a reason"

# --- 7. the historical output shape is untouched ----------------------------
# Two shipped tests parse bench.sh's BENCH_MOCK output. The capture layer is
# additive by contract; if the mock branch ever grows a capture line, they break.
for kind in vllm llamacpp; do
  out=$(PREFLIGHT_NO_AUTODETECT=1 CONTAINER=none ENGINE_KIND="$kind" BENCH_MOCK=1 bash "$BENCH" 2>/dev/null)
  command grep -q '^capture:' <<<"$out" && fail "BENCH_MOCK output ($kind) grew a capture line"
  command grep -q 'CAPTURE:' <<<"$out" && fail "BENCH_MOCK output ($kind) grew a capture section"
  command grep -qE '(NARRATIVE|PROMPT-PROCESSING)' <<<"$out" || fail "BENCH_MOCK output ($kind) lost its canonical block"
done
echo "  ✓ BENCH_MOCK output shape unchanged (test-submit-bench / test-measurement-record)"

# --- 8. CAPTURE=0 disables the whole layer ----------------------------------
run_bench healthy "$((PORT_BASE+5))" "$TMP/off.out" CAPTURE=0
command grep -q 'CAPTURE:' "$TMP/off.out" && fail "CAPTURE=0 must suppress the capture sections"
command grep -q 'summary \[narrative\]' "$TMP/off.out" || fail "CAPTURE=0 broke the normal bench output"
echo "  ✓ CAPTURE=0 leaves the pre-capture output shape exactly as it was"

# --- 9. ENDPOINT: chat is the historical default; completion is the new mode ---
run_bench healthy "$((PORT_BASE+6))" "$TMP/chat.out" ONLY=narr RUNS=1 WARMUPS=0
command grep -q 'mode=chat' "$TMP/chat.out" || fail "ENDPOINT must default to chat (the historical path)"
command grep -q 'endpoint=/v1/chat/completions' "$SRV_LOG" \
  || fail "the default run did not drive /v1/chat/completions"
run_bench healthy "$((PORT_BASE+7))" "$TMP/comp.out" ONLY=narr RUNS=1 WARMUPS=0 ENDPOINT=completion
command grep -q 'mode=completion' "$TMP/comp.out" || fail "ENDPOINT=completion not reflected in the fingerprint"
command grep -q 'endpoint=/v1/completions' "$SRV_LOG" \
  || fail "ENDPOINT=completion did not actually drive the raw completions endpoint"
command grep -q 'status: OK' "$TMP/comp.out" \
  || fail "raw completions must still count tokens (choices[0].text, not a delta)"
echo "  ✓ ENDPOINT: chat default preserved, completion drives the raw endpoint and still counts tokens"

# --- 10. haystacks sized by MEASURED tokens, not words (item 7) --------------
# The fake tokenizer emits 1.375 tok/word, so the old word-count sizing would
# overshoot a 2000-token target by ~37% — exactly the "40K that tokenized to 55K".
run_bench healthy "$((PORT_BASE+8))" "$TMP/ratio.out" \
  ONLY=narr RUNS=1 WARMUPS=0 PREFILL_PROBE=1 PREFILL_DEPTHS=2000 PREFILL_RUNS=1
python3 - "$TMP/ratio.out" <<'PY' || fail "prefill warm-up haystack is not sized by a measured token ratio"
import re, sys
txt = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"warm-1\s+wall=.*?prompt_toks=\s*(\d+)", txt)
assert m, "no prefill warm-up line found:\n" + txt[-1500:]
got = int(m.group(1))
# within 10% of the 2000-token target; the word heuristic lands ~37% over
assert 1800 <= got <= 2200, f"warm-up haystack was {got} tokens for a 2000-token target"
PY
echo "  ✓ prefill warm-up lands on its token target (word-count sizing would overshoot ~37%)"

# --- 11. STREAM triad ceiling is opt-in, and reports how it was measured -----
command grep -q 'RAM BANDWIDTH CEILING' "$TMP/healthy.out" \
  && fail "the STREAM calibration must be OPT-IN (it is not free)"
run_bench healthy "$((PORT_BASE+9))" "$TMP/stream.out" ONLY=narr RUNS=1 WARMUPS=0 STREAM_CALIB=1
if command grep -q 'RAM BANDWIDTH CEILING' "$TMP/stream.out"; then
  command grep -qE 'triad ceiling  : [0-9.]+ GB/s  \([0-9]+ concurrent workers' "$TMP/stream.out" \
    || fail "the triad ceiling must state its worker count — a single-threaded number is not a host ceiling"
  echo "  ✓ STREAM_CALIB=1 reports a multi-worker sustained ceiling (and is off by default)"
else
  # numpy-less rigs are a supported configuration, not a failure.
  command grep -q 'capture: STREAM triad ceiling unavailable' "$TMP/stream.out" \
    || fail "STREAM_CALIB=1 produced neither a ceiling nor a degradation notice"
  echo "  ✓ STREAM_CALIB=1 degrades cleanly where numpy is absent (and is off by default)"
fi

# --- 12. canvas granularity: no divide-by-epsilon decode rate (#809) --------
# A block-diffusion model denoises a whole canvas in parallel and the endpoint
# emits ~one chunk per canvas, so TTFT == wall and the decode window is zero
# width. master divided by 1e-6 and printed 8- and 9-digit "throughputs"
# (decode_TPS=690000000.00, shape mean=2728143.37 CV=223.6% — #822, real).
#
# The realism bound below is the assertion that matters: NO decode figure
# anywhere in the output, per-run or aggregate, may exceed it.
REALISM_BOUND=100000

assert_no_absurd_decode() {   # $1=file $2=context
  python3 - "$1" "$2" "$REALISM_BOUND" <<'PY' || fail "an absurd decode figure was printed"
import re, sys
path, ctx, bound = sys.argv[1], sys.argv[2], float(sys.argv[3])
bad = []
for ln in open(path, encoding="utf-8"):
    for m in re.finditer(r"decode_TPS[= ]\s*(?:mean=\s*)?([0-9]+(?:\.[0-9]+)?)", ln):
        if float(m.group(1)) > bound:
            bad.append(ln.rstrip())
if bad:
    sys.exit(f"{ctx}: decode figure above the realism bound ({bound:.0f}):\n" + "\n".join(bad))
PY
}

run_bench canvas "$((PORT_BASE+10))" "$TMP/canvas.out" ONLY=narr
C="$TMP/canvas.out"
assert_no_absurd_decode "$C" "canvas run"
command grep -q 'decode_TPS=   n/a' "$C" \
  || { command grep -m2 'run-' "$C" >&2; fail "a zero-width decode window must print n/a, never a number"; }
command grep -q 'single-block emission' "$C" \
  || fail "the n/a must name WHY the window is unmeasurable"
command grep -q '⚠ CANVAS GRANULARITY' "$C" \
  || fail "an all-degenerate shape must be classified canvas-granularity (#809)"
command grep -q 'HEADLINE number is wall_TPS' "$C" \
  || fail "the canvas marker must name wall_TPS as the headline"
command grep -q 'DERIVED, NOT MEASURED' "$C" \
  || fail "with no measurable run the decode figure is derived and must say so"
command grep -q '⚠ decode-win : narrative' "$C" \
  || fail "the integrity panel must state the unmeasurable-window split"
# ...and the run must not be mistaken for a failed one. A canvas run's decode mean
# is DERIVED, so it is excluded from the engine cross-check — which used to leave
# the status enum with no TPS at all and classify a healthy run NO_TOKENS.
command grep -q 'status: NO_TOKENS' "$C" \
  && { command grep -A3 'CAPTURE: INTEGRITY' "$C" >&2; fail "a healthy canvas run must not classify NO_TOKENS"; }
# The record producer refuses to write a measured record without a parseable
# `decode_TPS mean=` line, so suppressing the column outright would break it.
command grep -qE '^\s+decode_TPS\s+mean=' "$C" \
  || fail "the decode_TPS summary line must stay parseable (measurement_record.py keys off it)"
echo "  ✓ canvas: n/a per run, derived+labelled aggregate, canvas marker, no absurd figure"

# --- 13. the #822 SHAPE: some runs degenerate, some measurable --------------
# Four plausible runs and one 690000000.00 is what produced mean=2728143.37.
# The degenerate run must be EXCLUDED from the statistic, and the exclusion said
# out loud — a quietly-dropped sample is its own defect.
run_bench canvas_mixed "$((PORT_BASE+11))" "$TMP/mixed.out" ONLY=narr
M="$TMP/mixed.out"
assert_no_absurd_decode "$M" "canvas_mixed run"
command grep -qE 'decode-window  unmeasurable on [0-9]+/[0-9]+ run\(s\)' "$M" \
  || fail "excluding a run from decode_TPS must be stated, not silent"
command grep -q 'decode_TPS above is over the' "$M" \
  || fail "the summary must say how many runs the decode statistic actually rests on"
python3 - "$M" <<'PY' || fail "the mixed shape did not produce a finite, interpretable decode CV"
import re, sys
txt = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"^\s+decode_TPS\s+mean=\s*([\d.]+)\s+std=\s*([\d.]+)\s+CV=\s*([\d.]+)%", txt, re.M)
assert m, "no decode_TPS summary line"
mean, cv = float(m.group(1)), float(m.group(3))
# master on this same scenario: mean=81584.17 CV=140.7%
assert mean < 100000, f"decode mean {mean} is not interpretable"
assert cv < 100, f"decode CV {cv}% — the degenerate run is still in the sample"
PY
echo "  ✓ canvas_mixed: degenerate run excluded from the statistic, and the exclusion is stated"

# --- 14. AR models are untouched, and the guard is not overridable ----------
# The healthy (autoregressive) run must carry NONE of the canvas machinery.
for unwanted in 'decode_TPS=   n/a' 'CANVAS GRANULARITY' 'decode-win :' 'DERIVED, NOT MEASURED'; do
  command grep -qF "$unwanted" "$H" \
    && fail "an autoregressive run grew canvas output: $unwanted"
done
echo "  ✓ AR run carries none of the canvas machinery (output unchanged)"

# DECODE_GRANULARITY=canvas declares the class on a model whose runs look AR.
run_bench healthy "$((PORT_BASE+12))" "$TMP/declared.out" ONLY=narr RUNS=1 WARMUPS=0 \
  DECODE_GRANULARITY=canvas
command grep -q 'declared: DECODE_GRANULARITY=canvas' "$TMP/declared.out" \
  || fail "DECODE_GRANULARITY=canvas must declare the class without waiting for auto-detection"
# ...and declaring `token` must NOT buy back the divide-by-epsilon. The per-run
# guard is unconditional: there is no decode rate to print, whatever the label.
run_bench canvas "$((PORT_BASE+13))" "$TMP/forced.out" ONLY=narr RUNS=1 WARMUPS=0 \
  DECODE_GRANULARITY=token
assert_no_absurd_decode "$TMP/forced.out" "canvas run with DECODE_GRANULARITY=token"
command grep -q 'CANVAS GRANULARITY' "$TMP/forced.out" \
  && fail "DECODE_GRANULARITY=token must suppress the canvas classification"
echo "  ✓ DECODE_GRANULARITY declares the class, and cannot re-enable the epsilon divide"

# --- 15. PP scrape end-to-end: suppressed where meaningless (#817) ----------
# The canonical narrative prompt is ~11 tokens on this fixture. The engine's
# windowed prompt-throughput average over it is not a prefill rate and must not
# render as one — `PP tok/s mean=2.00 CV=55.9%` reached three community reports.
run_bench healthy "$((PORT_BASE+14))" "$TMP/pp.out" ONLY=narr RUNS=2 WARMUPS=1 \
  PREFILL_PROBE=1 PREFILL_DEPTHS=2000 PREFILL_RUNS=2
P="$TMP/pp.out"
command grep -q 'PP tok/s       n/a (engine-log scrape suppressed:' "$P" \
  || { command grep -n 'PP tok/s' "$P" >&2; fail "the short-prompt PP scrape must be suppressed (#817)"; }
command grep -q 'a prompt this short cannot fill one' "$P" \
  || fail "the suppression must state WHY, not just vanish"
command grep -q 'client-side `prefill tok/s`' "$P" \
  || fail "the suppression must point at the number that IS trustworthy"
# The short-prompt shape must not print a numeric PP line at all.
python3 - "$P" <<'PY' || fail "a short-prompt shape still printed a numeric PP tok/s"
import re, sys
blocks = re.split(r"^=== summary \[([^\]]+)\] \(n=\d+\) ===$", open(sys.argv[1], encoding="utf-8").read(), flags=re.M)
for name, body in zip(blocks[1::2], blocks[2::2]):
    if name.startswith("prefill-"):
        continue
    bad = re.search(r"^\s+PP tok/s\s+mean=", body, re.M)
    assert not bad, f"shape [{name}] printed a numeric PP tok/s line: {bad.group(0)}"
PY
# ...while the DEEP probe, where the figure can be a rate, still prints — labelled.
command grep -q 'PP tok/s (engine log, windowed — indicative only) mean=' "$P" \
  || { command grep -n 'windowed' "$P" >&2; fail "a plausible windowed scrape must still print, labelled indicative"; }
echo "  ✓ PP scrape: suppressed with a reason on short prompts, printed+labelled at depth"

# --- 16. the CV condition fires at depth too (#817's second named case) -----
# min 13 / max 8044 across the sampled windows: passes the depth and rate floors,
# and is still not a rate.
# Exported, not passed through run_bench: this one configures the SERVER, which
# start_server launches from the ambient environment.
export FAKE_PP_JITTER=13,8044,13,8044
run_bench healthy "$((PORT_BASE+15))" "$TMP/ppcv.out" ONLY=narr RUNS=1 WARMUPS=1 \
  PREFILL_PROBE=1 PREFILL_DEPTHS=2000 PREFILL_RUNS=2
unset FAKE_PP_JITTER
command grep -q 'PP tok/s (engine log, windowed) n/a (suppressed: scraped CV' "$TMP/ppcv.out" \
  || { command grep -n 'windowed' "$TMP/ppcv.out" >&2; fail "a CV>100% windowed sample must be suppressed at depth (#817)"; }
command grep -q 'is the client-side measurement for this depth' "$TMP/ppcv.out" \
  || fail "the depth suppression must point at the client-side prefill line"
echo "  ✓ windowed-in-prefill scrape suppressed on CV, pointing at the client-side number"

# --- 17. a PLAUSIBLE short-shape scrape keeps the parseable line shape -------
# measurement_record.py and rebench-report.py both key on `^\s+PP tok/s\s+mean=`.
# The `(engine log, windowed — indicative only)` label is appended as a SUFFIX so
# the number stays where the parsers look for it.
LONGP="$(python3 -c 'print("calibration filler sentence with stable token shape. " * 200)')"
run_bench healthy "$((PORT_BASE+16))" "$TMP/pplong.out" ONLY=narr RUNS=1 WARMUPS=1 \
  PREFILL_PROBE=0 "PROMPT_NARR=$LONGP"
python3 - "$TMP/pplong.out" <<'PY' || fail "a plausible PP line no longer matches the shipped parsers"
import re, sys
txt = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"^\s+PP tok/s\s+mean=\s*([0-9.]+)\s+std=\s*([0-9.]+)\s+CV=\s*([0-9.]+)%", txt, re.M)
assert m, "no parseable PP tok/s line on a long-prompt shape:\n" + "\n".join(
    l for l in txt.splitlines() if "PP tok/s" in l)
assert "indicative only" in m.string[m.start():m.string.find("\n", m.start())], \
    "the plausible line must still carry its 'indicative only' qualifier"
PY
echo "  ✓ a plausible PP line keeps the shape measurement_record/rebench-report parse"

# --- 18. --quick is a PRESET over existing knobs, and nothing more (#832) ----
# Deliberately NOT run through run_bench: that helper exports RUNS/WARMUPS/
# PREFILL_PROBE itself, which the preset would (correctly) refuse to overwrite,
# so the test would prove nothing. This one sets only what hermeticity needs.
QUICK_EXPANSION='WARMUPS=1 RUNS=1 ONLY=narr PREFILL_PROBE=0 QUIET=1'
command grep -qF "QUICK_PRESET=($QUICK_EXPANSION)" "$BENCH" \
  || fail "the --quick preset array drifted from the documented expansion (#832)"
bash "$BENCH" --help > "$TMP/help.out" 2>&1 || fail "--help must exit 0"
command grep -qF "$QUICK_EXPANSION" "$TMP/help.out" \
  || fail "--help must print the exact knob set --quick expands to"
for want in "NOT CANONICAL" "NO CV" "comparable across boots" "BENCHMARKS.md row" \
            ">=2 BOOTS per arm"; do
  command grep -qF "$want" "$TMP/help.out" \
    || fail "--help is missing the guardrail text: $want"
done
echo "  ✓ --help documents --quick and every guardrail the issue asks for"

run_bench_bare() {   # $1=scenario $2=port $3=outfile $4=script args (may be "") [VAR=VAL ...]
  local scen="$1" port="$2" out="$3" argstr="$4"; shift 4
  local -a bargs=(); read -r -a bargs <<< "$argstr"
  start_server "$scen" "$port"
  PATH="$TMP/bin:$PATH" PREFLIGHT_NO_AUTODETECT=1 CONTAINER=none \
    SERVER_LOG="$SRV_LOG" SERVER_PID="$SRV_PID" \
    URL="http://127.0.0.1:$port" MODEL=fake-model \
    env "$@" timeout 300 bash "$BENCH" ${bargs[@]+"${bargs[@]}"} > "$out" 2>"${out}.err" || true
  kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=""
}

run_bench_bare healthy "$((PORT_BASE+17))" "$TMP/quick.out" "--quick"
Q="$TMP/quick.out"
command grep -q '⚠ QUICK MODE — NOT CANONICAL' "$Q" || fail "--quick must print a non-canonical banner"
command grep -qF "preset applied : $QUICK_EXPANSION" "$Q" \
  || { command grep -n 'preset applied' "$Q" >&2; fail "--quick did not expand to exactly the documented knob set"; }
# ...and each knob must actually have taken effect, not merely been announced.
command grep -q '=== warmups (1) ===' "$Q"  || fail "--quick did not set WARMUPS=1"
command grep -q '=== measured (1) ===' "$Q" || fail "--quick did not set RUNS=1"
command grep -q 'summary \[narrative\]' "$Q" || fail "--quick lost the narrative shape"
command grep -q 'summary \[code\]' "$Q"      && fail "--quick did not set ONLY=narr"
command grep -q 'PREFILL-' "$Q"              && fail "--quick did not set PREFILL_PROBE=0"
command grep -qE '^  (run|warm)-[0-9]+ ' "$Q" && fail "--quick did not set QUIET=1 (per-run lines printed)"
# The banner must survive a tail: a reader who scrolls to the summary and copies
# it never sees the opening one.
tail -8 "$Q" | command grep -q 'QUICK MODE' \
  || fail "the non-canonical warning must be repeated at the END of the run"
echo "  ✓ --quick expands to exactly the documented set, and every knob took effect"

# --- 19. --quick guardrails: no card, explicit env wins, QUICK=1 parity -----
if out="$(CARD=snapshot bash "$BENCH" --quick 2>&1)"; then
  fail "CARD under --quick must be refused (a card from n=1 data has a zero noise band)"
fi
command grep -q 'incompatible' <<<"$out" || fail "the CARD refusal must say why: $out"
command grep -q 'noise band' <<<"$out" \
  || fail "the CARD refusal must name the degenerate-noise-band reason"
# An explicitly-set knob wins over the preset AND is reported — silently
# overriding what the operator typed makes the run a third protocol.
run_bench_bare healthy "$((PORT_BASE+18))" "$TMP/quickov.out" "--quick" RUNS=3
command grep -q 'your env kept  : RUNS=3 (preset wanted 1)' "$TMP/quickov.out" \
  || { command grep -n 'preset\|env kept' "$TMP/quickov.out" >&2; fail "an explicit knob must win over the preset and be REPORTED"; }
command grep -q '=== measured (3) ===' "$TMP/quickov.out" || fail "the explicit RUNS=3 did not take effect"
# QUICK=1 must be the same thing as --quick (the issue offers both spellings).
run_bench_bare healthy "$((PORT_BASE+19))" "$TMP/quickenv.out" "" QUICK=1
command grep -qF "preset applied : $QUICK_EXPANSION" "$TMP/quickenv.out" \
  || fail "QUICK=1 must behave exactly like --quick"
# Unknown arguments are rejected rather than silently ignored.
bash "$BENCH" --not-a-flag >/dev/null 2>&1 && fail "an unknown argument must be rejected"
echo "  ✓ --quick guardrails: card refused, explicit env wins + reported, QUICK=1 parity"

# --- 20. --quick is additive: a default run carries none of it --------------
for unwanted in 'QUICK MODE' 'preset applied'; do
  command grep -qF "$unwanted" "$H" && fail "a default run grew --quick output: $unwanted"
done
out=$(PREFLIGHT_NO_AUTODETECT=1 CONTAINER=none BENCH_MOCK=1 bash "$BENCH" --quick 2>/dev/null)
command grep -q 'QUICK MODE' <<<"$out" \
  && fail "BENCH_MOCK output must stay byte-identical even under --quick (two shipped tests parse it)"
echo "  ✓ --quick is additive: default runs and BENCH_MOCK output are untouched"

# --- 21. interconnect: all three layers land in the artifact (#805) ---------
# report.sh has reported the three-layer P2P state since #789; bench.sh did not,
# so every BENCHMARKS Rig cell's field 4 (resolved custom-AR state) was
# hand-transcribed from a different run and went stale silently.
#
# The healthy run above is CONTAINER=none against a host-mode fake llama-server
# — i.e. exactly the degradation case the issue calls out: there is no engine
# all-reduce to report, and saying "off" would read as a misconfiguration.
command grep -q '=== Interconnect (three layers) ===' "$H" \
  || { command grep -n 'GPU state' "$H" >&2; fail "the bench artifact must self-document its interconnect state (#805)"; }
for layer in 'layer 1  driver P2P grant :' 'layer 2  NCCL use         :' 'layer 3  engine custom-AR :'; do
  command grep -qF "$layer" "$H" || fail "interconnect block is missing '$layer' — all THREE layers are the point"
done
# Layer 3 must degrade cleanly in host-mode / llama.cpp, per the issue's spec.
command grep -qE 'layer 3  engine custom-AR : .*custom-AR n/a' "$H" \
  || { command grep -F 'layer 3' "$H" >&2; fail "a host-mode/llama.cpp run must read 'custom-AR n/a', never 'off' (there is no AR kernel to disable)"; }
# ...and layers 1-2 must still report on that same run — degrading layer 3 must
# not take the whole block down with it.
command grep -qE 'layer 1  driver P2P grant : (GRANTED|REFUSED|NVLink|n/a)' "$H" \
  || fail "layer 1 must still report in host mode"
# The block is a FOOTER fact, not a measurement: BENCH_MOCK output is parsed by
# two shipped tests and must stay byte-identical.
out=$(PREFLIGHT_NO_AUTODETECT=1 CONTAINER=none BENCH_MOCK=1 bash "$BENCH" 2>/dev/null)
command grep -q 'Interconnect' <<<"$out" \
  && fail "BENCH_MOCK output must not grow the interconnect block (test-submit-bench parses it)"
# A CAPTURE=0 run keeps it too — the block is not part of the capture layer, and
# a harness that turned capture off still wants to know what it ran under.
run_bench healthy "$((PORT_BASE+20))" "$TMP/icap0.out" CAPTURE=0
command grep -q '=== Interconnect (three layers) ===' "$TMP/icap0.out" \
  || fail "CAPTURE=0 must keep the interconnect block (it is a footer fact, not a capture)"
echo "  ✓ interconnect: three layers reported, layer 3 degrades to n/a in host mode, BENCH_MOCK untouched"

echo "test-bench-capture: ok"
