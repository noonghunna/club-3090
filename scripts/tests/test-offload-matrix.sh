#!/usr/bin/env bash
# test-offload-matrix — Tier 0 (static) guards for scripts/offload-matrix.sh and
# its renderer. No server boot, no GPU, no model: everything here is verifiable
# from the sources plus a synthetic TSV.
#
# The harness produces numbers that decide serving configs, so its failure mode
# is a plausible WRONG number rather than a crash. Six defects of that shape
# shipped during development (see #824). Each check below guards one of them.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP="$ROOT_DIR/scripts/offload-matrix.sh"
RENDER="$ROOT_DIR/scripts/offload-matrix-render.py"
export PYTHONUTF8="${PYTHONUTF8:-1}"   # repo rule: locale must not decide python decoding
fail() { echo "FAIL: $1" >&2; exit 1; }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# 1. syntax, both halves
bash -n "$SWEEP" || fail "bash -n: syntax error in offload-matrix.sh"
python3 -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$RENDER" \
  || fail "renderer: syntax error"
echo "  ✓ syntax (sweep + renderer)"

# 2. MODEL is mandatory and the refusal is actionable (not a traceback)
set +e; out="$(MODEL= INHERIT=0 OUT_DIR="$TMP/out" bash "$SWEEP" 2>&1)"; rc=$?; set -e
[[ "$rc" == "2" ]] || fail "missing MODEL should exit 2, got $rc"
grep -q "MODEL=" <<<"$out" || fail "missing-MODEL error should name the MODEL variable"
echo "  ✓ refuses without MODEL (exit 2, actionable)"

# 3. COLUMN PARITY — guards the defect where the TSV printf carried 32 format
#    specifiers for 33 arguments: bash re-ran the format for the surplus arg, so
#    every arm wrote a short row PLUS a junk status-only row. Header is the only
#    place the width may be defined.
hdr_n="$(python3 - "$SWEEP" <<'PY'
import re,sys
s=open(sys.argv[1],encoding="utf-8").read()
print(len(re.search(r"HDR=\$'([^']+)'",s).group(1).split(r"\t")))
PY
)"
row_n="$(python3 - "$SWEEP" <<'PY'
import re,sys
s=open(sys.argv[1],encoding="utf-8").read()
blk=re.search(r'emit_row "\$status" \\\n((?:.*\\\n)*.*)\n',s).group(1)
print(len(re.findall(r'"[^"]*"', blk)))
PY
)"
[[ -n "$hdr_n" && -n "$row_n" ]] || fail "could not extract header/emit_row widths"
(( row_n == hdr_n - 1 )) || fail "emit_row passes $row_n values; header holds $hdr_n (expected $((hdr_n-1)) + status)"
grep -q 'NCOL=\$(awk' "$SWEEP" || fail "row width must be derived from \$HDR, not hand-counted"
echo "  ✓ column parity: header $hdr_n = $row_n values + status, width derived from \$HDR"

# 4. STATUS-ENUM PARITY — every status the sweep can emit must be explained by
#    the renderer. This has already caught real drift (CACHE_DISABLED shipped
#    with no renderer annotation within an hour of being added).
missing="$(python3 - "$SWEEP" "$RENDER" <<'PY'
import re,sys
sw=open(sys.argv[1],encoding="utf-8").read(); rd=open(sys.argv[2],encoding="utf-8").read()
emitted={m for m in re.findall(r'status="([A-Z_]{4,})"',sw)} | {m for m in re.findall(r'emit_row ([A-Z_]{4,})',sw)}
known={m for m in re.findall(r'"([A-Z_]{4,})":',rd)}
print(" ".join(sorted(emitted-known)))
PY
)"
[[ -z "$missing" ]] || fail "renderer does not annotate status(es): $missing"
echo "  ✓ status-enum parity (sweep ↔ renderer)"

# 5. PAIRING KEY INCLUDES shape — workload shape FLIPS THE SIGN of speculative
#    results, so pairing a spec arm against a baseline of another shape
#    manufactures a gain that does not exist.
grep -qE '^KEY *= *\(' "$RENDER" || fail "renderer should define an explicit pairing KEY"
python3 - "$RENDER" <<'PY' || exit 1
import re,sys
k=re.search(r'^KEY *= *\(([^)]*)\)',open(sys.argv[1],encoding="utf-8").read(),re.M).group(1)
need={"offload","N","ctx_req","cache_mb","admit","throttle","shape","pcache"}
have={x.strip().strip('"\'') for x in k.split(",") if x.strip()}
missing=need-have
if missing: print(f"FAIL: pairing KEY missing {sorted(missing)}",file=sys.stderr); sys.exit(1)
PY
echo "  ✓ pairing key: shape, pcache, ctx_req (sign-flip / prefill-skew / unified-KV mispairs)"

# 6. ENGINE-SCOPE GUARD — the tool emits llama.cpp flags and scrapes llama.cpp
#    log lines, so it must refuse a binary that is not one. vLLM's CPU offload
#    measures a different mechanism entirely and would produce numbers that look
#    valid and mean nothing.
touch "$TMP/fake.gguf"
set +e
out="$(MODEL="$TMP/fake.gguf" INHERIT=0 PLAN=1 LAYERS_TOTAL=48 LLAMA_SERVER=/bin/true \
       OUT_DIR="$TMP/out" bash "$SWEEP" 2>&1)"; rc=$?
set -e
[[ "$rc" == "2" ]] || fail "non-llama.cpp server should exit 2, got $rc"
grep -qi "llama.cpp" <<<"$out" || fail "engine-scope refusal should say what it accepts"
echo "  ✓ refuses a non-llama.cpp server (exit 2)"

# 7. PLAN=1 enumerates arms and boots nothing. Needs a binary that passes the
#    scope probe above, so this doubles as the minimal fake-server stub the
#    Tier 1 suite will grow from: --help must advertise the flags the tool emits.
cat > "$TMP/fake-llama-server" <<'STUB'
#!/usr/bin/env bash
if [[ " $* " == *" --help "* ]]; then
  echo "  -ngl,  --n-gpu-layers N"
  echo "  -ot,   --override-tensor PATTERN"
  echo "         --moe-cache N"
  exit 0
fi
exit 0
STUB
chmod +x "$TMP/fake-llama-server"
out="$(MODEL="$TMP/fake.gguf" INHERIT=0 PLAN=1 LAYERS_TOTAL=48 LLAMA_SERVER="$TMP/fake-llama-server" \
       OUT_DIR="$TMP/out" SWEEP_OFFLOAD="19 28" SWEEP_N="1 4" bash "$SWEEP" 2>&1)" || fail "PLAN=1 should exit 0"
grep -qiE 'arm|stage|plan' <<<"$out" || fail "PLAN=1 should enumerate the planned sequence"
grep -q '\[OK\]' <<<"$out" && fail "PLAN=1 must not produce measured arms"
echo "  ✓ PLAN=1 enumerates without booting"

# 8. Renderer survives a mixed TSV (OK + every failure status) and never
#    silently drops a non-OK arm into the conclusions.
python3 - "$SWEEP" "$TMP/t.tsv" <<'PY'
import re,sys
hdr=re.search(r"HDR=\$'([^']+)'",open(sys.argv[1],encoding="utf-8").read()).group(1).split(r"\t")
i={k:n for n,k in enumerate(hdr)}
def row(**kw):
    r=["-"]*len(hdr)
    for k,v in kw.items(): r[i[k]]=str(v)
    return "\t".join(r)
rows=[row(arm="A",rep=1,shape="copy",offload=19,N=1,ctx_slot=262144,drafter="none",nmax=0,
          cache_mb=8192,admit=1,throttle=64,agg=34.0,strm=34.4,status="OK"),
      row(arm="B",rep=1,shape="copy",offload=19,N=1,ctx_slot=262144,drafter="ngram-mod",nmax=3,
          cache_mb=8192,admit=1,throttle=64,agg=60.0,strm=60.5,status="OK"),
      row(arm="C",rep=1,shape="copy",offload=28,N=8,drafter="none",nmax=0,status="BOOT_FAIL"),
      row(arm="D",rep=1,shape="copy",offload=19,N=1,drafter="ngram-mod",nmax=3,status="CACHE_DISABLED")]
open(sys.argv[2],"w",encoding="utf-8").write("\t".join(hdr)+"\n"+"\n".join(rows)+"\n")
PY
out="$(python3 "$RENDER" "$TMP/t.tsv" 2>&1)" || fail "renderer crashed on a mixed TSV"
grep -q "NOT OK" <<<"$out" || fail "renderer must list non-OK arms separately"
grep -q "CACHE_DISABLED" <<<"$out" || fail "renderer must surface CACHE_DISABLED"
grep -qE '\+7[0-9]\.[0-9]%' <<<"$out" || fail "renderer should pair B against A and show the delta"
echo "  ✓ renderer handles mixed TSV; non-OK arms excluded from conclusions"

# 9. The suite must not pollute the repo: the sweep mkdir's $PWD/offload-matrix-out
#    unless OUT_DIR is set, and a test that leaves artifacts behind is a bad test.
[[ -e "$ROOT_DIR/offload-matrix-out" ]] && fail "test leaked offload-matrix-out into the repo"
echo "  ✓ no artifacts left in the repo"

echo "test-offload-matrix: ok (Tier 0 — static, no server)"
