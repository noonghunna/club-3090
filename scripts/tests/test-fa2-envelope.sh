#!/usr/bin/env bash
# test-fa2-envelope.sh — fp8-KV FA2 composes must stay inside the kernel's envelope (#1358).
#
# WHY: the fa2-fp8kv-sm86 kernel refuses (fatally, killing the engine) a context past
# max_k = 262144 tokens, so with fp8 KV a --max-model-len above it must refuse to
# start. It also refuses a query chunk past max_q = 2048, and until 2026-11-11
# envelope.sh capped the per-request prefill chunk at 2048 whenever fp8 KV ran with
# --max-num-seqs > 1. The artifact's v0.1.0 (in the fork build the composes pin)
# splits long causal prefill in a mixed batch itself, so that cap is GONE: measured on
# 2x 3090 with MAX_NUM_SEQS=2 and threshold 4096 (36K-token prompts beside a decode,
# DFlash2 off and on), while the pre-v0.1.0 artifact crashed on the same workload.
# This guard now holds the opposite of what it used to: the chunk passes through.
#
# Legs: (1) the helper's decisions on synthetic argument lists; (2) every compose
# that mounts the plugin calls it before `exec vllm serve`; (3) the DELIVERY PATH:
# render each such compose through `docker compose config` with fp8 KV and 2 seqs,
# assert the real argv passes through unchanged (and that a >2048 chunk reaches
# vLLM somewhere, so a cap creeping back in reds), then render it with
# MAX_MODEL_LEN=300000 and assert the helper refuses, with a NEGATIVE CONTROL that the
# override reaches argv.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HELPER=models/qwen3.8-27b/vllm/patches/fa2-fp8kv-sm86/envelope.sh
# shellcheck source=../../models/qwen3.8-27b/vllm/patches/fa2-fp8kv-sm86/envelope.sh
source "$HELPER"

fails=0
fail() { echo "FAIL: $*" >&2; fails=$((fails+1)); }

# The per-request chunk an argument list lets through: the threshold when > 0,
# never more than the batch budget. Mirrors vLLM's scheduler, independently of the helper.
chunk_of() {
  python3 - "$@" <<'PY'
import sys
a = sys.argv[1:]
def val(flag):
    out = None
    for i, x in enumerate(a):
        if x == flag and i + 1 < len(a):
            out = a[i + 1]
        elif x.startswith(flag + "="):
            out = x.split("=", 1)[1]
    return out
t, b = val("--long-prefill-token-threshold"), val("--max-num-batched-tokens")
t = int(t) if t and t.isdigit() and int(t) > 0 else None
b = int(b) if b and b.isdigit() and int(b) > 0 else None
c = min(x for x in (t, b) if x is not None) if (t or b) else None
print("unbounded" if c is None else c)
PY
}

# --- 1. helper decisions -------------------------------------------------------
case_ok() { # <label> <expect-rc> <expect-chunk|same> args...
  local label="$1" want_rc="$2" want="$3"; shift 3
  local rc=0; fa2_envelope "$@" 2>/dev/null || rc=$?
  if [[ "$rc" != "$want_rc" ]]; then fail "helper [$label]: rc=$rc, want $want_rc"; return; fi
  [[ "$want_rc" != 0 ]] && return
  if [[ "$want" == same ]]; then
    [[ "${FA2_ARGS[*]}" == "$*" ]] || fail "helper [$label]: args changed but should not: ${FA2_ARGS[*]}"
  else
    local got; got="$(chunk_of "${FA2_ARGS[@]}")"
    [[ "$got" == "$want" ]] || fail "helper [$label]: chunk $got, want $want (${FA2_ARGS[*]})"
  fi
}
# formerly capped to 2048; since the v0.1.0 artifact they pass through unchanged
case_ok "fp8, 2 seqs, threshold 4096"       0 same --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096
case_ok "fp8, 2 seqs, threshold 0"          0 same --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --max-num-batched-tokens 8192 --long-prefill-token-threshold 0
case_ok "fp8, 2 seqs, no threshold flag"    0 same --kv-cache-dtype fp8 --max-num-seqs 2 --max-num-batched-tokens 8192
case_ok "fp8, seqs unset (vLLM default>1)"  0 same --kv-cache-dtype fp8_e4m3 --long-prefill-token-threshold 4096
case_ok "fp8, --flag=value forms"           0 same --kv-cache-dtype=fp8_e4m3 --max-num-seqs=4 --long-prefill-token-threshold=4096
case_ok "fp8, 1 seq (shipped dual)"         0 same --kv-cache-dtype fp8_e4m3 --max-num-seqs 1 --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096
case_ok "fp8, 2 seqs, budget 2048 (ultramax)" 0 same --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --max-num-batched-tokens 2048 --long-prefill-token-threshold 0
case_ok "fp8, 2 seqs, budget under threshold" 0 same --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --max-num-batched-tokens 1024 --long-prefill-token-threshold 4096
case_ok "bf16 (stock FlashAttention)"       0 same --kv-cache-dtype bfloat16 --max-num-seqs 2 --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096
case_ok "fp8, max-model-len past max_k"     1 -    --kv-cache-dtype fp8_e4m3 --max-num-seqs 1 --max-model-len 300000
case_ok "fp8=, max-model-len= past max_k"   1 -    --kv-cache-dtype=fp8 --max-num-seqs 2 --max-model-len=262145
case_ok "fp8, max-model-len at max_k"       0 same --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --max-model-len 262144
case_ok "bf16, max-model-len past max_k"    0 same --kv-cache-dtype bfloat16 --max-model-len 300000
# nothing moves: arguments around the ones the helper reads keep their order
fa2_envelope --model m --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --long-prefill-token-threshold 4096 --port 8000 2>/dev/null
[[ "${FA2_ARGS[*]}" == "--model m --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --long-prefill-token-threshold 4096 --port 8000" ]] \
  || fail "helper [pass-through]: ${FA2_ARGS[*]}"

# --- 2. every plugin compose calls the helper before exec ----------------------
composes=()
while IFS= read -r f; do composes+=("$f"); done < <(
  grep -rl -e 'patches/fa2-fp8kv-sm86:/etc/club3090/fa2' models/*/*/compose --include='*.yml' 2>/dev/null | sort)
[[ ${#composes[@]} -gt 0 ]] || { echo "FAIL: found no compose mounting the fa2 plugin — the search is wrong, not the tree" >&2; exit 1; }
for f in "${composes[@]}"; do
  python3 - "$f" <<'PY' || fail "$f: does not apply the fa2 envelope before exec vllm serve"
import sys
lines = [l.strip() for l in open(sys.argv[1], encoding="utf-8")]
def at(text):
    return next((i for i, l in enumerate(lines) if l == text), -1)
src = at("source /etc/club3090/fa2/envelope.sh")
call = at('fa2_envelope "$$@" || exit 1')
setp = at('set -- "$${FA2_ARGS[@]}"')
first_exec = next((i for i, l in enumerate(lines) if l.startswith("exec vllm serve")), -1)
ok = -1 < src < call < setp < first_exec
sys.exit(0 if ok else 1)
PY
done

# --- 3. delivery path + negative control ---------------------------------------
render() { # <compose> [ENV=VALUE ...] -> the rendered command, one argument a line
  local f="$1"; shift
  env MODEL_DIR=/tmp KV_CACHE_DTYPE=fp8_e4m3 MAX_NUM_SEQS=2 SPEC_N=0 "$@" \
    docker compose -f "$f" config --format json 2>/dev/null \
    | python3 -c 'import json,sys; s=json.load(sys.stdin)["services"]; print("\n".join(next(v["command"] for v in s.values() if v.get("command"))))'
}
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  over_2048=0 long_reached=0 fp8_composes=0
  for f in "${composes[@]}"; do
    raw=()
    while IFS= read -r a; do raw+=("$a"); done < <(render "$f")
    if [[ ${#raw[@]} -eq 0 ]]; then fail "$f: could not render its command"; continue; fi
    kv=""; for ((i=0; i<${#raw[@]}; i++)); do [[ "${raw[i]}" == --kv-cache-dtype ]] && kv="${raw[i+1]}"; done
    if ! fa2_envelope "${raw[@]}" 2>/dev/null; then fail "$f: helper refused the rendered argv"; continue; fi
    [[ "${FA2_ARGS[*]}" == "${raw[*]}" ]] || fail "$f: the helper changed the rendered argv: ${FA2_ARGS[*]}"
    after="$(chunk_of "${FA2_ARGS[@]}")"
    [[ "$kv" == fp8* && ( "$after" == unbounded || "$after" -gt 2048 ) ]] && over_2048=$((over_2048+1))
    # max_k through the delivery path: the same compose with a context past the kernel's
    long=()
    while IFS= read -r a; do long+=("$a"); done < <(render "$f" MAX_MODEL_LEN=300000)
    for ((i=0; i<${#long[@]}; i++)); do [[ "${long[i]}" == --max-model-len && "${long[i+1]}" == 300000 ]] && long_reached=$((long_reached+1)); done
    if [[ "$kv" == fp8* ]]; then
      fp8_composes=$((fp8_composes+1))
      if printf '%s\n' "${long[@]}" | command grep -qx 300000 && fa2_envelope "${long[@]}" 2>/dev/null; then
        fail "$f: fp8 KV with --max-model-len 300000 starts (kernel max_k 262144)"
      fi
    fi
  done
  [[ $fp8_composes -gt 0 ]] || fail "no compose rendered with fp8 KV — the KV_CACHE_DTYPE override no longer reaches argv"
  # The chunk cap is gone: somewhere a >2048 chunk must reach vLLM with fp8 + 2 seqs, or
  # a cap crept back in (or the overrides stopped reaching argv).
  [[ $over_2048 -gt 0 ]] || fail "no fp8 compose lets a >2048 prefill chunk through with MAX_NUM_SEQS=2 — is a chunk cap back?"
  # Negative control for the max_k leg: the override must reach argv, or the refusal proves nothing.
  [[ $long_reached -gt 0 ]] || fail "negative control: MAX_MODEL_LEN=300000 reached no compose's argv"
else
  echo "NOTE: docker compose unavailable — the delivery-path leg did NOT run." >&2
fi

if [[ $fails -gt 0 ]]; then
  echo "$fails fa2 envelope check(s) failed" >&2
  exit 1
fi
echo "PASS: fa2 envelope — helper cases, ${#composes[@]} composes wired, fp8 + MAX_NUM_SEQS=2 passes through unchanged and a context past max_k refuses, through docker compose config"
