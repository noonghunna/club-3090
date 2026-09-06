#!/usr/bin/env bash
# Guard: the ceiling-ladder verdict chain in verify-stress.sh must classify every
# combination of ladder outcomes, and must NOT fall through to the "engine may
# have crashed early" catch-all for cases the ladder body deliberately produces.
#
# Why this exists: the chain shipped without a branch for "recall missed, nothing
# failed, nothing passed" — the exact state the ladder body creates when it breaks
# on a first-rung recall miss ("break out of the ladder, and pass the probe"). Two
# community 4x3090 GLM reports (#1128/#1129) were scored as hard FAILs telling the
# reporter to go looking for an engine crash that never happened, on a rig whose
# every other check passed. A ladder verdict that can say "crashed" when nothing
# crashed is worse than no verdict, so each state is pinned here.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SRC="scripts/verify-stress.sh"
[[ -f "$SRC" ]] || { echo "FAIL: $SRC not found"; exit 1; }

START='if [[ "$any_fail" == "0" && "$any_pass" == "1" ]]; then'
END='fail "ceiling ladder: no rung succeeded and none were skipped"'
s=$(command grep -n -F "$START" "$SRC" | head -1 | cut -d: -f1)
e=$(command grep -n -F "$END"   "$SRC" | head -1 | cut -d: -f1)
[[ -n "$s" && -n "$e" && "$e" -gt "$s" ]] || { echo "FAIL: could not locate the verdict chain (anchors moved?)"; exit 1; }
CHAIN=$(mktemp); trap 'rm -f "$CHAIN"' EXIT
sed -n "${s},$((e+2))p" "$SRC" > "$CHAIN"

# Refuse a vacuous run: the extract must actually be the chain.
lines=$(wc -l < "$CHAIN")
[[ "$lines" -ge 20 ]] || { echo "FAIL: extracted only $lines lines — not the verdict chain"; exit 1; }
for needle in 'any_sizing_error' 'any_skipped' 'ceiling ladder'; do
  command grep -q -- "$needle" "$CHAIN" || { echo "FAIL: extract missing '$needle' — wrong region"; exit 1; }
done

rc=0
check() {
  local name="$1" expect="$2" setup="$3"
  local out verdict
  out=$( set +e
    pass() { echo "PASS|$1"; return 0; }
    fail() { echo "FAIL|$1"; return 1; }
    skip() { echo "SKIP|$1"; return 0; }
    any_pass=0 any_fail=0 any_skipped=0 any_recall_miss=0 any_recall_empty=0 any_sizing_error=0
    first_fail_tokens=0 first_recall_miss_tokens=94000 first_recall_miss_pct=46
    last_pass_tokens=0 last_pass_pct=0 rung_count=5 ceiling_start=95000 n_ctx=204800
    vram_after_all=0 vram_before_all=0 vram_margin_mb=800 tok_per_scale_unit=63
    LOG_CMD="docker logs X"
    eval "$setup"
    source "$CHAIN" 2>&1 | head -1 )
  verdict="${out%%|*}"
  if [[ "$verdict" == "$expect" ]]; then
    echo "  ok   $name -> $verdict"
  else
    echo "  FAIL $name -> got '$verdict', want '$expect'"; rc=1
  fi
}

echo "verify-stress ceiling-ladder verdict states:"
# The reported regression: model returned no content, system filled the rung.
check "empty content, system OK   " SKIP 'any_recall_miss=1; any_recall_empty=1'
# Genuine wrong answer on the first rung — the ladder body's documented intent.
check "recall miss, system OK     " PASS 'any_recall_miss=1'
# NEGATIVE CONTROLS — these must keep failing.
check "true crash (nothing ran)   " FAIL 'true'
check "OOM wall after passes      " FAIL 'any_pass=1; any_fail=1'
check "recall miss then OOM       " FAIL 'any_recall_miss=1; any_fail=1'
check "all rungs rejected (400)   " FAIL 'any_skipped=1'
check "filler sizing error        " FAIL 'any_sizing_error=1'
check "clean sweep, all rungs pass" PASS 'any_pass=1'

[[ "$rc" == "0" ]] && echo "PASS: all ladder verdict states classified" || echo "FAIL: ladder verdict regression"
exit "$rc"
