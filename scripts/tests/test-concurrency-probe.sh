#!/usr/bin/env bash
# test-concurrency-probe — offline guards for scripts/concurrency-probe.sh (the
# #246 Phase 2 soak-validation tool) and scripts/lib/concurrency_probe.py.
# The live probe needs a running server; these check only what can be verified
# without one: syntax, SWEEP-needs-SLUG, SWEEP_DRY reboot plans, --sweep dry
# plans (no SLUG, no reboot), planner clips, and the card renderer.
set -euo pipefail

# Force Python UTF-8 mode (PEP 540) before the first python3 call (#779).
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT_DIR/scripts/concurrency-probe.sh"
LIB="$ROOT_DIR/scripts/lib/concurrency_probe.py"
fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. syntax
bash -n "$PROBE" || fail "bash -n: syntax error"
python3 -m py_compile "$LIB" || fail "py_compile concurrency_probe.py"
echo "  ✓ syntax"

# 2. SWEEP without SLUG must refuse with exit 2 (a reboot target is mandatory —
#    vLLM can't hot-change max-num-seqs, so each N is a boot).
set +e
out="$(SWEEP="4 8" bash "$PROBE" 2>&1)"; rc=$?
set -e
[[ "$rc" == "2" ]] || fail "SWEEP without SLUG should exit 2, got $rc"
command grep -q "SWEEP needs SLUG" <<<"$out" || fail "SWEEP-without-SLUG error message missing"
echo "  ✓ SWEEP refuses without SLUG (exit 2)"

# 3. SWEEP_DRY prints the plan for every N and must NOT boot the server (a dry
#    run never touches switch.sh). Assert on output: one [sweep:dry] line per N,
#    no [sweep] boot line, and the knee summary always prints.
out="$(SWEEP="4 8 12" SLUG=vllm/minimal SWEEP_DRY=1 bash "$PROBE" 2>&1)"
[[ "$(command grep -c '\[sweep:dry\]' <<<"$out")" == "3" ]] || fail "SWEEP_DRY should print one plan line per N (3)"
command grep -q '\[sweep\] boot' <<<"$out" && fail "SWEEP_DRY must not boot the server"
command grep -q "sweep knee" <<<"$out" || fail "SWEEP should always print a knee summary"
echo "  ✓ SWEEP_DRY plans 3 reboots without booting"

# 4. --sweep --dry does NOT need SLUG and must not plan switch.sh reboots.
out="$(bash "$PROBE" --sweep --dry --n 1,2,4,8 --ctx 1k,4k,16k 2>&1)"
[[ $? == 0 ]] || fail "--sweep --dry should exit 0 without SLUG"
command grep -q "switch.sh" <<<"$out" && fail "--sweep --dry must not plan switch.sh reboots"
command grep -q "SWEEP needs SLUG" <<<"$out" && fail "--sweep must not require SLUG"
command grep -q "1K" <<<"$out" || fail "--sweep --dry should list 1K"
command grep -q "4K" <<<"$out" || fail "--sweep --dry should list 4K"
command grep -q "16K" <<<"$out" || fail "--sweep --dry should list 16K"
echo "  ✓ --sweep --dry plans a live matrix without SLUG"

# 5. --sweep + --validate refused
set +e
out="$(bash "$PROBE" --sweep --validate --dry 2>&1)"; rc=$?
set -e
[[ "$rc" == "2" ]] || fail "--sweep --validate should exit 2, got $rc"
command grep -q "cannot combine" <<<"$out" || fail "--sweep --validate message missing"
echo "  ✓ --sweep --validate refused"

# 6. planner clips: slots, max-len, KV pool
plan="$(
  CTX_SWEEP="1k 4k 8k 16k 32k" N_LIST="1 2 4 8 16 32" GEN_TOKENS=256 \
  KV_TOKENS=262144 SERVED_SLOTS=8 SERVED_MAX_LEN=32768 \
  python3 "$LIB" --plan
)"
command grep -q "32" <<<"$plan" && command grep -q "served slots=8" <<<"$plan" \
  || fail "plan should drop N>8 with a slots note"
command grep -q "32K" <<<"$plan" || fail "32K should remain when max-len=32K"
# 32K × N=8 = 8*(32768+256)=264192 > 262144 → skip
command grep -q "32K" <<<"$plan" || true
tsv="$(
  CTX_SWEEP="1k 4k 8k 16k 32k" N_LIST="1 2 4 8 16 32" GEN_TOKENS=256 \
  KV_TOKENS=262144 SERVED_SLOTS=8 SERVED_MAX_LEN=32768 \
  python3 "$LIB" --plan-tsv
)"
command grep -q $'skip\t32768\t8\t' <<<"$tsv" || fail "32K N=8 should skip on KV"
command grep -q $'run\t32768\t4\t' <<<"$tsv" || fail "32K N=4 should run (fits KV)"
command grep -q $'run\t16384\t8\t' <<<"$tsv" || fail "16K N=8 should run"
command grep -q $'skip\t1024\t16\t' <<<"$tsv" && fail "1K N=16 should be dropped by slots, not appear"
echo "  ✓ planner clips slots / max-len / KV"

plan_ml="$(
  CTX_SWEEP="1k 4k 16k 32k" N_LIST="1 2 4" GEN_TOKENS=256 \
  SERVED_MAX_LEN=16384 \
  python3 "$LIB" --plan
)"
command grep -q "32K" <<<"$plan_ml" && command grep -q "max-model-len" <<<"$plan_ml" \
  || fail "32K should be dropped when max-len=16K"
echo "  ✓ planner drops ctx above served max-model-len"

# 7. card renderer: vs 1-stream + SWEET on 16K knee
card="$(python3 "$LIB" --card <<'JSON'
{
  "model": "qwen3.6-27b",
  "slug": "vllm/qwen-27b-dual-fast",
  "spec": "MTP n=3",
  "gpus": "2× RTX 3090",
  "kv_tokens": 210000,
  "slots": 8,
  "served_max_len": 262144,
  "engine": "vllm",
  "gen_tokens": 256,
  "cache": "shared 75%",
  "command": "bash scripts/concurrency-probe.sh --sweep",
  "rows": [
    {"ctx": 1024, "n": 1, "strm": 87.3, "agg": 87, "ttft_s": 0.1, "vram_gb": 38.2, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 1024, "n": 8, "strm": 38.0, "agg": 241, "ttft_s": 0.4, "vram_gb": 39.8, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 1, "strm": 87.3, "agg": 87, "ttft_s": 0.8, "vram_gb": 38.2, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 2, "strm": 51.8, "agg": 104, "ttft_s": 1.4, "vram_gb": 39.1, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 4, "strm": 22.1, "agg": 88, "ttft_s": 3.2, "vram_gb": 40.4, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 8, "strm": 6.8, "agg": 54, "ttft_s": 12.0, "vram_gb": 42.1, "clean": 0, "pass": 0, "skip": null},
    {"ctx": 32768, "n": 8, "skip": "N*(ctx+gen) > KV_TOKENS"}
  ]
}
JSON
)"
command grep -q "club-3090" <<<"$card" || fail "card should say club-3090"
command grep -q "vs 1-stream" <<<"$card" || fail "card should label vs 1-stream"
command grep -q "1.20" <<<"$card" || fail "16K N=2 should be 104/87 ≈ 1.20×"
command grep -q "SWEET" <<<"$card" || fail "card should print SWEET"
command grep -q "N=2 @ 16K" <<<"$card" || fail "SWEET should star the 16K aggregate peak (N=2, 104 tok/s)"
command grep -q "github.com" <<<"$card" && fail "card should not billboard the repo URL"
command grep -q "=== recommend ===" <<<"$card" || fail "card should end with a compose recommendation"
command grep -q "MAX_NUM_SEQS=2" <<<"$card" || fail "keep-full-ctx rec should be MAX_NUM_SEQS=2 (16K peak)"
command grep -q "MAX_NUM_SEQS=8" <<<"$card" || fail "max-agg rec should be MAX_NUM_SEQS=8 (1K peak)"
command grep -q "MAX_MODEL_LEN=1024" <<<"$card" || fail "max-agg rec should set MAX_MODEL_LEN to the short ctx"
command grep -q "keep" <<<"$card" || fail "full-ctx rec should keep served max-model-len"
echo "  ✓ card renderer"

# 8. prompt shapes: cold salts first, shared salts after the prefix
cold="$(python3 -c 'import sys; sys.path.insert(0,"'"$ROOT_DIR"'/scripts/lib"); from concurrency_probe import prompt_text; print(prompt_text(0,1,1024,"cold",0.75,256)[:40])')"
[[ "$cold" == \[probe\ s0\ r1\]* ]] || fail "cold prompt must start with per-stream/round salt, got: $cold"
shared="$(python3 -c 'import sys; sys.path.insert(0,"'"$ROOT_DIR"'/scripts/lib"); from concurrency_probe import prompt_text; p=prompt_text(0,1,1024,"shared",0.75,256); print("SALT" if p.lstrip().startswith("[probe") else "PREFIX")')"
[[ "$shared" == "PREFIX" ]] || fail "shared prompt must NOT start with the salt"
echo "  ✓ prompt cache shapes"

# 9. KV log parse
kv="$(python3 -c 'import sys; sys.path.insert(0,"'"$ROOT_DIR"'/scripts/lib"); from concurrency_probe import parse_kv_tokens_text; print(parse_kv_tokens_text("GPU KV cache size: 210,000 tokens"))')"
[[ "$kv" == "210000" ]] || fail "parse_kv_tokens_text, got $kv"
echo "  ✓ KV log parse"

echo "test-concurrency-probe: ok"
