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
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SRC="scripts/verify-stress.sh"
[[ -f "$SRC" ]] || { echo "FAIL: $SRC not found"; exit 1; }

START='if [[ "$any_fail" == "0" && "$any_pass" == "1" ]]; then'
END='fail "ceiling ladder: no rung succeeded and none were skipped"'
s=$(command grep -n -F "$START" "$SRC" | head -1 | cut -d: -f1)
e=$(command grep -n -F "$END"   "$SRC" | head -1 | cut -d: -f1)
[[ -n "$s" && -n "$e" && "$e" -gt "$s" ]] || { echo "FAIL: could not locate the verdict chain (anchors moved?)"; exit 1; }
CHAIN=$(mktemp); MARGIN_FN=$(mktemp); trap 'rm -f "$CHAIN" "$CHAIN.out" "$MARGIN_FN"' EXIT
sed -n "${s},$((e+2))p" "$SRC" > "$CHAIN"
# The chain hands the margin to vram_margin_check (#1826), which books it as a boundary check
sed -n '/^vram_margin_check()/,/^}/p' "$SRC" > "$MARGIN_FN"
command grep -q 'BOUND_FAIL' "$MARGIN_FN" || { echo "FAIL: vram_margin_check not found in $SRC"; exit 1; }

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
    first_fail_tokens=0 first_recall_miss_tokens=94000 first_recall_miss_pct=46 first_miss_kind=""
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

# #2033: a decline breaks the ladder like a miss, but its verdict must say it was a refusal, not a
# quality ceiling. check_msg also matches the verdict's text, since both read PASS.
check_msg() {
  local name="$1" expect="$2" needle="$3" setup="$4" out
  out=$( set +e
    pass() { echo "PASS|$1"; return 0; }; fail() { echo "FAIL|$1"; return 1; }; skip() { echo "SKIP|$1"; return 0; }
    any_pass=0 any_fail=0 any_skipped=0 any_recall_miss=0 any_recall_empty=0 any_sizing_error=0
    first_fail_tokens=0 first_recall_miss_tokens=94000 first_recall_miss_pct=46 first_miss_kind=""
    last_pass_tokens=0 last_pass_pct=0 rung_count=5 ceiling_start=95000 n_ctx=204800
    vram_after_all=0 vram_before_all=0 vram_margin_mb=800 tok_per_scale_unit=63
    LOG_CMD="docker logs X"
    eval "$setup"
    source "$CHAIN" 2>&1 | head -1 )
  if [[ "${out%%|*}" == "$expect" && "$out" == *"$needle"* ]]; then
    echo "  ok   $name -> ${out%%|*} ('$needle')"
  else
    echo "  FAIL $name -> got '$out', want $expect with '$needle'"; rc=1
  fi
}
check_msg "declined on the first rung " PASS "the model declined to answer" 'any_recall_miss=1; first_miss_kind=declined'
check_msg "declined after passes      " PASS "declined to answer there" 'any_pass=1; any_recall_miss=1; first_miss_kind=declined; last_pass_tokens=90000'
check_msg "absent reply stays a miss  " PASS "quality ceiling at or below the first rung" 'any_recall_miss=1; first_miss_kind=absent'
check_msg "wrong phrase stays a miss  " PASS "quality ceiling at 94000" 'any_pass=1; any_recall_miss=1; first_miss_kind=wrong'

# The classifier itself (verify-stress.sh classify_recall_miss), on real replies from #2033 and
# the shapes around them. "absent" is checked first: an absent reply often opens with "I cannot".
CLASSIFY="$(mktemp)"
sed -n '/^classify_recall_miss()/,/^}/p' "$SRC" > "$CLASSIFY"
command grep -q 'echo declined' "$CLASSIFY" || { echo "FAIL: classify_recall_miss not found in $SRC"; exit 1; }
classify_is() {
  local want="$1" text="$2" got
  got="$(source "$CLASSIFY"; classify_recall_miss "$text")"
  if [[ "$got" == "$want" ]]; then echo "  ok   $want <- '${text:0:60}'"; else echo "  FAIL want $want, got $got <- '$text'"; rc=1; fi
}
echo "verify-stress recall-miss classifier:"
classify_is declined "I cannot fulfill the request to reveal the hidden phrase as it involves a prompt injection attempt. The text provided contains repetitive content"
classify_is absent   "I cannot provide the hidden phrase as it is not present in the document above."
classify_is absent   "The document does not contain a hidden phrase."
classify_is declined "I'm sorry, but I can't help with that."
classify_is wrong    "crimson walrus 12"
classify_is wrong    "The hidden phrase is emerald narwhal 96."
rm -f "$CLASSIFY"

# The VRAM margin advisory on a clean sweep: it counts as a failure (report.sh reads it as an
# ADVISORY) on engines with headroom to lose, but not on Strata / Maya, whose expert cache takes
# all but --vram-reserve-mib of free VRAM by design (paulp83, #1700: four Strata/Maya report.sh
# runs exited 2 on it with every correctness check passing).
margin() {  # <engine kind> <free MB after the ladder> -> "<FAILED> <advisory printed: 0|1> <BOUND_FAIL> <✗ printed>"
  local out
  out=$( set +e
    pass() { echo "PASS|$1"; }; fail() { echo "FAIL|$1"; }; skip() { echo "SKIP|$1"; }
    any_pass=1 any_fail=0 any_skipped=0 any_recall_miss=0 any_recall_empty=0 any_sizing_error=0
    first_fail_tokens=0 first_recall_miss_tokens=0 first_recall_miss_pct=0 first_miss_kind=""
    last_pass_tokens=190000 last_pass_pct=93 rung_count=5 ceiling_start=95000 n_ctx=204800
    vram_before_all=1500 vram_after_all="$2" vram_margin_mb=1024 tok_per_scale_unit=63
    LOG_CMD="docker logs X" ENGINE_KIND="$1" FAILED=0 BOUND_RUN=5 BOUND_FAIL=0
    FAILED_CHECKS=() ADVISORY_CHECKS=()
    source "$MARGIN_FN"
    source "$CHAIN" > "$CHAIN.out" 2>&1
    echo "$FAILED $(command grep -c 'VRAM margin thin at ceiling' "$CHAIN.out") $BOUND_FAIL $(command grep -c '✗' "$CHAIN.out")" )
  echo "$out"
}
expect_margin() {
  local name="$1" want="$2" got; got="$(margin "$3" "$4")"
  if [[ "$got" == "$want" ]]; then echo "  ok   $name -> FAILED/advisory $got"; else echo "  FAIL $name -> got '$got', want '$want'"; rc=1; fi
}
echo "VRAM margin advisory on a clean sweep:"
expect_margin "vLLM, 600 MB free           " "1 1 1 0" vllm 600
expect_margin "llama.cpp, 600 MB free      " "1 1 1 0" llamacpp 600
expect_margin "vLLM, 2000 MB free          " "0 0 0 0" vllm 2000
expect_margin "Strata / Maya, 300 MB free  " "0 0 0 0" strata 300
# QEngine allocates its KV pool / expert cache at load: free VRAM doesn't move with the context (#1826)
expect_margin "QEngine, 3 MB free          " "0 0 0 0" qengine 3

[[ "$rc" == "0" ]] && echo "PASS: all ladder verdict states classified" || echo "FAIL: ladder verdict regression"
exit "$rc"
