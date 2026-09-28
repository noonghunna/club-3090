#!/usr/bin/env bash
# test-fa2-envelope.sh — fp8-KV FA2 composes must stay inside the kernel's envelope (#1358).
#
# WHY: the fa2-fp8kv-sm86 split-KV kernel refuses (fatally, killing the engine) any
# batch whose longest query exceeds 2048 tokens. The plugin avoids that kernel for
# prefill only when a step holds ONE request, so with fp8 KV and --max-num-seqs > 1
# a prefill chunk above 2048 — vLLM's --long-prefill-token-threshold, or the whole
# --max-num-batched-tokens budget when the threshold is 0 — crashes it. The multi*
# composes shipped threshold 4096 / seqs 2, and dual-ultrafast's own concurrency
# recipe (SPEC_N=0, raise MAX_NUM_SEQS) reaches the same state. envelope.sh caps the
# chunk at 2048 in exactly that case.
#
# Legs: (1) the helper's decisions on synthetic argument lists; (2) every compose
# that mounts the plugin calls it before `exec vllm serve`; (3) the DELIVERY PATH:
# render each such compose with the risky overrides through `docker compose config`,
# run the real argv through the helper, and assert the chunk that reaches vLLM is
# <= 2048 — with a NEGATIVE CONTROL showing the same argv WITHOUT the helper is over.
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
case_ok "fp8, 2 seqs, threshold 4096"       0 2048 --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096
case_ok "fp8, 2 seqs, threshold 0"          0 2048 --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --max-num-batched-tokens 8192 --long-prefill-token-threshold 0
case_ok "fp8, 2 seqs, no threshold flag"    0 2048 --kv-cache-dtype fp8 --max-num-seqs 2 --max-num-batched-tokens 8192
case_ok "fp8, seqs unset (vLLM default>1)"  0 2048 --kv-cache-dtype fp8_e4m3 --long-prefill-token-threshold 4096
case_ok "fp8, --flag=value forms"           0 2048 --kv-cache-dtype=fp8_e4m3 --max-num-seqs=4 --long-prefill-token-threshold=4096
case_ok "fp8, 1 seq (shipped dual)"         0 same --kv-cache-dtype fp8_e4m3 --max-num-seqs 1 --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096
case_ok "fp8, 2 seqs, budget 2048 (ultramax)" 0 same --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --max-num-batched-tokens 2048 --long-prefill-token-threshold 0
case_ok "fp8, 2 seqs, budget under threshold" 0 same --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --max-num-batched-tokens 1024 --long-prefill-token-threshold 4096
case_ok "bf16 (stock FlashAttention)"       0 same --kv-cache-dtype bfloat16 --max-num-seqs 2 --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096
case_ok "fp8, max-model-len past max_k"     1 -    --kv-cache-dtype fp8_e4m3 --max-num-seqs 1 --max-model-len 300000
case_ok "bf16, max-model-len past max_k"    0 same --kv-cache-dtype bfloat16 --max-model-len 300000
# the threshold is REPLACED, not duplicated, and nothing else moves
fa2_envelope --model m --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --long-prefill-token-threshold 4096 --port 8000 2>/dev/null
[[ "${FA2_ARGS[*]}" == "--model m --kv-cache-dtype fp8_e4m3 --max-num-seqs 2 --long-prefill-token-threshold 2048 --port 8000" ]] \
  || fail "helper [replace in place]: ${FA2_ARGS[*]}"

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
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  over_raw=0
  for f in "${composes[@]}"; do
    raw=()
    while IFS= read -r a; do raw+=("$a"); done < <(
      MODEL_DIR=/tmp KV_CACHE_DTYPE=fp8_e4m3 MAX_NUM_SEQS=2 SPEC_N=0 \
        docker compose -f "$f" config --format json 2>/dev/null \
        | python3 -c 'import json,sys; s=json.load(sys.stdin)["services"]; print("\n".join(next(v["command"] for v in s.values() if v.get("command"))))')
    if [[ ${#raw[@]} -eq 0 ]]; then fail "$f: could not render its command"; continue; fi
    kv=""; for ((i=0; i<${#raw[@]}; i++)); do [[ "${raw[i]}" == --kv-cache-dtype ]] && kv="${raw[i+1]}"; done
    before="$(chunk_of "${raw[@]}")"
    if ! fa2_envelope "${raw[@]}" 2>/dev/null; then fail "$f: helper refused the rendered argv"; continue; fi
    after="$(chunk_of "${FA2_ARGS[@]}")"
    [[ "$before" == unbounded || "$before" -gt 2048 ]] && over_raw=$((over_raw+1))
    if [[ "$kv" == fp8* ]]; then
      seqs=""; for ((i=0; i<${#FA2_ARGS[@]}; i++)); do [[ "${FA2_ARGS[i]}" == --max-num-seqs ]] && seqs="${FA2_ARGS[i+1]}"; done
      if [[ "$seqs" != 1 && ( "$after" == unbounded || "$after" -gt 2048 ) ]]; then
        fail "$f: fp8 KV, max-num-seqs $seqs, chunk $after reaches vLLM (limit 2048)"
      fi
    fi
  done
  # Negative control: without the helper the risky overrides DO exceed the envelope
  # somewhere, or this leg proves nothing.
  [[ $over_raw -gt 0 ]] || fail "negative control: no compose exceeds 2048 without the helper — the overrides no longer reach argv"
else
  echo "NOTE: docker compose unavailable — the delivery-path leg did NOT run." >&2
fi

if [[ $fails -gt 0 ]]; then
  echo "$fails fa2 envelope check(s) failed" >&2
  exit 1
fi
echo "PASS: fa2 envelope — helper cases, ${#composes[@]} composes wired, fp8 + MAX_NUM_SEQS=2 renders a <=2048 chunk through docker compose config"
