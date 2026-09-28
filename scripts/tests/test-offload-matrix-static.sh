#!/usr/bin/env bash
# test-offload-matrix-static — TIER 0 guards for scripts/offload-matrix.sh and its
# renderer. Nothing here boots a server or touches a GPU; these are the checks that
# can be made from the source text alone, and they cover the whole class of defect
# this harness has already shipped: rows whose column count silently disagrees with
# the header, and a renderer that reads columns the driver never writes.
#
# The harness exists to produce numbers that DECIDE serving configs. Its failure mode
# is not a crash — it is a plausible wrong number. So these assert structure, never
# throughput values. See issue #824.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"   # repo rule: locale must not decide python decoding
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP="$ROOT_DIR/scripts/offload-matrix.sh"
REND="$ROOT_DIR/scripts/offload-matrix-render.py"
LIB="$ROOT_DIR/scripts/lib/capture.sh"
fail() { echo "FAIL: $1" >&2; exit 1; }

[[ -f "$SWEEP" ]] || fail "missing $SWEEP"
[[ -f "$REND"  ]] || fail "missing $REND"
[[ -f "$LIB"   ]] || fail "missing $LIB — the sweep consumes the shared capture layer"

# 1. both halves parse
bash -n "$SWEEP" || fail "bash -n: syntax error in offload-matrix.sh"
python3 -m py_compile "$REND" || fail "py_compile: syntax error in the renderer"
echo "  ✓ syntax (bash -n + py_compile)"

# 2. the header is the ONLY definition of column count, and status is last.
#    emit_row derives width from it; anything that hand-counts columns is the bug
#    this suite exists to prevent.
NCOL=$(python3 - "$SWEEP" <<'PY'
import re,sys
src=open(sys.argv[1],encoding="utf-8").read()
m=re.search(r"HDR=\$'(.*?)'",src,re.S) or sys.exit("no HDR")
f=m.group(1).split("\\t")
assert f[-1]=="status", f"header must end with status, ends with {f[-1]}"
assert len(f)==len(set(f)), "duplicate column name in header"
print(len(f))
PY
) || fail "header malformed"
[[ "$NCOL" -ge 30 ]] || fail "header collapsed to $NCOL columns — expected the full row"
echo "  ✓ header well-formed: $NCOL unique columns, ends with 'status'"

# 3. exactly ONE writer appends rows, and it is emit_row. A second hand-rolled
#    printf is how the 32-specifiers-for-33-arguments defect shipped: bash re-runs
#    a short format string for the surplus args, so every arm wrote a truncated row
#    PLUS a junk row containing only the status.
appends=$(grep -cE '>> *"\$TSV"' "$SWEEP")
[[ "$appends" == "1" ]] || fail "expected exactly 1 append to \$TSV (inside emit_row), found $appends"
grep -qE '^emit_row\(\)' "$SWEEP" || fail "emit_row not defined"
echo "  ✓ single row emitter (1 append site)"

# 4. emit_row itself: pads short rows to header width, and REFUSES to write an
#    over-long row rather than corrupting the TSV. Extracted and exercised in
#    isolation so this is a real unit test, not a grep.
emit_src=$(awk '/^emit_row\(\)/,/^}/' "$SWEEP")
[[ -n "$emit_src" ]] || fail "could not extract emit_row"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
(
  set -euo pipefail
  TSV="$tmp/t.tsv"; NCOL=5
  eval "$emit_src"
  emit_row OK a b                       # short: must pad
  emit_row OK a b c d                   # exact: must not pad
  emit_row OK a b c d e f 2>/dev/null && exit 3   # over-long: must refuse (non-zero)
  exit 0
) || fail "emit_row accepted an over-long row (would corrupt the TSV)"
widths=$(awk -F'\t' '{print NF}' "$tmp/t.tsv" | sort -u)
[[ "$widths" == "5" ]] || fail "emit_row produced ragged rows: widths [$widths], expected all 5"
[[ "$(awk -F'\t' '{print $NF}' "$tmp/t.tsv" | sort -u)" == "OK" ]] || fail "status must be the last field"
[[ "$(awk -F'\t' 'NR==1{print $3}' "$tmp/t.tsv")" == "-" ]] || fail "short rows must pad with '-'"
echo "  ✓ emit_row pads short rows, refuses over-long rows, keeps status last"

# 5. every status the renderer explains must be one the driver can actually emit,
#    and vice versa. Drift in either direction means an arm renders with no
#    explanation, or the renderer documents a state that cannot happen.
#
#    The vocabulary now has TWO sources: the sweep still emits its own early-exit
#    states directly (BOOT_FAIL, TIMEOUT, PORT_BUSY, ...), while the four shared
#    run-quality states (NO_TOKENS / REQ_ERRORS / CACHE_DISABLED / INVALID_BYPASS)
#    are classified by scripts/lib/capture.sh, which bench.sh uses too. Scanning
#    both is the point — one enum, one precedence, two callers.
python3 - "$SWEEP" "$REND" "$LIB" <<'PY' || exit 1
import re,sys
sweep=open(sys.argv[1],encoding="utf-8").read()
rend =open(sys.argv[2],encoding="utf-8").read()
lib  =open(sys.argv[3],encoding="utf-8").read()
def states(src):
    return (set(re.findall(r'status="?([A-Z_]{3,})"?',src))
            | set(re.findall(r'emit_row ([A-Z_]{3,})',src)))
emitted=states(sweep) | states(lib)
emitted.discard("OK")                      # OK is the default, never explained
known=set(re.findall(r'"([A-Z_]{3,})":',rend))
if emitted-known: sys.exit(f"FAIL: driver emits statuses the renderer cannot explain: {sorted(emitted-known)}")
if known-emitted: sys.exit(f"FAIL: renderer explains statuses the driver never emits: {sorted(known-emitted)}")
print(f"  ✓ status vocabulary agrees both ways ({len(emitted)} non-OK states, sweep + lib)")
PY

# 6. every column the renderer reads must exist in the header. One view once
#    crashed with KeyError and another silently rendered '-' forever because they
#    referenced columns the driver never wrote. Derived cells are '_'-prefixed and
#    'reps' is synthesised at merge time; everything else must be a real column.
python3 - "$SWEEP" "$REND" <<'PY' || exit 1
import re,sys
sweep=open(sys.argv[1],encoding="utf-8").read()
rend =open(sys.argv[2],encoding="utf-8").read()
hdr=set(re.search(r"HDR=\$'(.*?)'",sweep,re.S).group(1).split("\\t"))
def tup(name):
    m=re.search(rf"^{name}\s*=\s*\((.*?)\)",rend,re.S|re.M)
    return set(re.findall(r'"([a-z_]+)"',m.group(1))) if m else set()
refs=set()
refs|=tup("NUM"); refs|=tup("KEY"); refs|=tup("INTCOL")
# Scan ONLY inside the VIEWS dict. A whole-file scan also matches plain view-name
# tuples like `if view in ("bw","all")`, which are not column references at all.
views=re.search(r"^VIEWS\s*=\s*\{(.*?)^\}",rend,re.S|re.M)
if not views: sys.exit("FAIL: no VIEWS dict in renderer")
refs|=set(re.findall(r'\("[^"]*",\s*"([a-z_][a-z_0-9]*)"\)',views.group(1)))
synthetic={"reps"}
missing={c for c in refs if not c.startswith("_")} - hdr - synthetic
if missing: sys.exit(f"FAIL: renderer reads columns absent from the header: {sorted(missing)}")
print(f"  ✓ every renderer column resolves ({len(refs)} referenced)")
PY

# 7. the pairing key MUST include shape. Workload shape flips the SIGN of a
#    speculation result on this stack (+81% copy vs +3.9% prose), so pairing a
#    copy-heavy spec arm against a prose baseline manufactures a gain that does
#    not exist. This is the one renderer bug that produces a confident wrong number.
python3 - "$REND" <<'PY' || exit 1
import re,sys
rend=open(sys.argv[1],encoding="utf-8").read()
key=re.search(r"^KEY\s*=\s*\((.*?)\)",rend,re.S|re.M)
if not key: sys.exit("FAIL: no KEY tuple in renderer")
cols=set(re.findall(r'"([a-z_]+)"',key.group(1)))
for must in ("shape","offload","ctx_slot","cache_mb","pcache"):
    if must not in cols: sys.exit(f"FAIL: pairing KEY missing '{must}' — spec arms would mispair")
print(f"  ✓ pairing key is shape-aware ({len(cols)} dimensions)")
PY

# 8. pool_slots must sum EVERY pool with no tail/head. This looks like a bug and
#    has been reported as one; it is NOT. Each pool contributes its own slots, so
#    summing them all is correct — on a 4-pool run, 742+337+558+279 = 1916, the
#    reported figure. Adding a `tail -N` would silently DROP two pools and roughly
#    halve the number. Asserted so nobody "fixes" it back into a defect.
#    (The slots now come off the shared lib's per-device census rather than a raw
#    log grep; the invariant is unchanged and so is this guard.)
pool_line=$(grep -nE 'pool=\$\(' "$SWEEP" | head -1 | cut -d: -f1)
[[ -n "$pool_line" ]] || fail "could not find the pool_slots extraction"
pool_expr=$(sed -n "${pool_line}p" "$SWEEP")
grep -q 'slots=' <<<"$pool_expr"            || fail "pool extraction no longer reads slots="
grep -qE '\| *(tail|head) ' <<<"$pool_expr" && fail "pool_slots must NOT tail/head — it would drop pools (see #824)"
echo "  ✓ pool_slots sums all pool-init lines (no tail — the reported 'bug' is a false positive)"

# 9. the renderer's own docstring states the row width; keep it honest, since it is
#    what a reader trusts before running anything.
doc_n=$(grep -oE 'full row is [0-9]+ columns' "$REND" | grep -oE '[0-9]+' || true)
if [[ -n "$doc_n" ]]; then
  [[ "$doc_n" == "$NCOL" ]] || fail "renderer docstring says $doc_n columns, header has $NCOL"
  echo "  ✓ renderer docstring column count matches the header ($NCOL)"
fi

# 10. THE #824 DETECTOR, GENERALISED. The half-cached check must compare the count
#     of DEVICES THAT HOLD A POOL against the device count — never the count of
#     pool LINES. A device can hold several pools (`pool[0]`, `pool[1]`), so a
#     line-count detector reads pool_lines >= NGPU and passes an arm where an
#     entire device got no cache at all. That is exactly the state #824 was filed
#     for, one level deeper, and the sweep carried the weaker form until it adopted
#     the shared lib. Guarded here so it cannot regress to line-counting.
grep -qE 'pool_devs *< *\$\{NGPU' "$SWEEP" \
  || fail "the half-cached detector must compare DEVICES-with-a-pool against NGPU"
if grep -qE 'pool_lines *< *\$\{?NGPU' "$SWEEP"; then
  fail "the detector is comparing pool LINE count to device count — a multi-pool device defeats that (#824 generalised)"
fi
grep -q 'cap_moe_parse pooldev' "$SWEEP" \
  || fail "the sweep should get its allocation census from the shared lib, not a private grep"
echo "  ✓ half-cached detection counts DEVICES with a pool, not pool lines (#824 generalised)"

# 11. the sweep must consume the shared capture layer, and must not have kept a
#     private copy of a scrape the lib owns. Two copies is how a fixed defect
#     comes back — it is the reason this adoption happened at all.
grep -q 'source "\$_OM_LIB"' "$SWEEP" || fail "the sweep no longer sources scripts/lib/capture.sh"
for fn in cap_moe_parse cap_status_classify cap_vram_used cap_dmon_start cap_cpu_snap cap_ram_rd_mbps; do
  grep -q "$fn" "$SWEEP" || fail "the sweep stopped using $fn — did a private copy come back?"
done
# the private helpers the adoption deleted must stay deleted
grep -qE '^cpu_snap\(\)' "$SWEEP" && fail "cpu_snap() came back — use cap_cpu_snap from the lib"
grep -q 'adoption is DEFERRED' "$LIB" && fail "the deferral note outlived the adoption"
echo "  ✓ sweep consumes the shared lib; the private copies stayed deleted"

echo "test-offload-matrix-static: ok"
