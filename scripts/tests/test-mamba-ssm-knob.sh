#!/usr/bin/env bash
# test-mamba-ssm-knob — the Qwen3.8-family vLLM composes must expose a WORKING SSM-state dtype knob.
#
# WHY THIS TEST EXISTS
# --------------------
# These composes shipped `MAMBA_CACHE_DTYPE=bfloat16`, documented as "store conv+ssm state in bf16".
# It never moved the SSM state: for Qwen3.5-family models vLLM (Qwen3_5ForConditionalGenerationConfig)
# copies the HF config's `mamba_ssm_dtype` (float32) into `--mamba-ssm-cache-dtype` whenever that is
# `auto`, so `--mamba-cache-dtype` alone only changes the tiny conv state. A clean boot, a plausible
# log line, and a knob that did nothing for the part that matters.
#
# `MAMBA_SSM_CACHE_DTYPE` → `--mamba-ssm-cache-dtype` is the lever that works (+11.2% KV pool on the
# dual-fast tier, measured 2026-09-24). This guard keeps it wired, keeps the comment honest, and runs
# each compose's REAL entrypoint block to prove the default and the override resolve as documented:
#   every compose (14): default bfloat16 (2026-09-24), `auto` reverts to the model's float32.
#   the CUDA-graph memory estimate is off by default on the two dual-fast composes ONLY.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fails=0
fail() { echo "✗ $*" >&2; fails=$((fails+1)); }

DUAL_FAST=(
  models/qwen3.8-27b/vllm/compose/dual/autoround-int4/mtp.yml
  models/thinkingcap-qwen3.8-27b/vllm/compose/dual/autoround-int4/mtp.yml
)
mapfile -t FILES < <(command grep -rlE 'MAMBA_CACHE_DTYPE' models/*/vllm/compose --include=*.yml | command grep -v _archive | sort)
[[ ${#FILES[@]} -gt 0 ]] || { echo "✗ no compose carries MAMBA_CACHE_DTYPE — the file set this guard covers is gone" >&2; exit 1; }

# The real block: from `MAMBA_ARGS=()` to the ssm-dtype line, compose `$$` unescaped to `$`.
block_of() { awk '/MAMBA_ARGS=\(\)/{f=1} f{print} f&&/mamba-ssm-cache-dtype|ssm cache dtype/&&/echo/{exit}' "$1" | sed 's/\$\$/$/g'; }
resolve() {  # resolve <file> [VAR=VAL] → prints the resulting MAMBA_ARGS, one per line
  local f="$1"; shift
  env -i PATH="$PATH" "$@" bash -c "set -u; $(block_of "$f")"$'\n''printf "%s\n" "${MAMBA_ARGS[@]+"${MAMBA_ARGS[@]}"}"' 2>/dev/null
}
is_dual_fast() { local d; for d in "${DUAL_FAST[@]}"; do [[ "$d" == "$1" ]] && return 0; done; return 1; }

for f in "${FILES[@]}"; do
  command grep -qE '^      - MAMBA_SSM_CACHE_DTYPE=\$\{MAMBA_SSM_CACHE_DTYPE:-\}$' "$f" \
    || fail "$f: MAMBA_SSM_CACHE_DTYPE not declared in environment: (docker would not forward it)"
  command grep -q -- '--mamba-ssm-cache-dtype' "$f" || fail "$f: --mamba-ssm-cache-dtype not wired in the entrypoint"
  command grep -q 'store conv+ssm state in bf16' "$f" && fail "$f: still claims MAMBA_CACHE_DTYPE moves the SSM state"
  unset_args="$(resolve "$f")"
  set_args="$(resolve "$f" MAMBA_SSM_CACHE_DTYPE=float16)"
  auto_args="$(resolve "$f" MAMBA_SSM_CACHE_DTYPE=auto)"
  [[ "$set_args" == *$'--mamba-ssm-cache-dtype\nfloat16'* ]] || fail "$f: an explicit MAMBA_SSM_CACHE_DTYPE=float16 did not reach the flag (got: ${set_args//$'\n'/ })"
  [[ "$auto_args" != *'--mamba-ssm-cache-dtype'* ]] || fail "$f: MAMBA_SSM_CACHE_DTYPE=auto still passed the flag (auto must defer to the model)"
  [[ "$unset_args" == *$'--mamba-ssm-cache-dtype\nbfloat16'* ]] || fail "$f: default is not bfloat16 (got: ${unset_args//$'\n'/ })"
done
for d in "${DUAL_FAST[@]}"; do [[ -f "$d" ]] || fail "dual-fast compose missing: $d"; done

# --- the CUDA-graph memory ESTIMATE: off by default on the two dual-fast composes ONLY -------------
# v0.30.0 reserves 0.5 GiB for graphs that take 0.1 GiB; off, plus bf16 SSM, is what clears two
# 262K sessions on dual-fast (1.98x -> 2.09x, full verify-stress passed). Everywhere else the
# estimate stays on — it is vLLM's own OOM accounting, and removing it is a per-tier decision.
for d in "${DUAL_FAST[@]}"; do
  command grep -qE '^      - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=\$\{VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0\}$' "$d" \
    || fail "$d: graph-estimate default (\${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}) missing"
  if docker compose version >/dev/null 2>&1; then
    off="$(env -u VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS docker compose -f "$d" config 2>/dev/null | command grep -oE 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "[0-9]"')"
    on="$(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 docker compose -f "$d" config 2>/dev/null | command grep -oE 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "[0-9]"')"
    [[ "$off" == *'"0"' ]] || fail "$d: renders '$off' by default, expected \"0\""
    [[ "$on" == *'"1"' ]] || fail "$d: =1 override renders '$on', expected \"1\" (the opt-out must work)"
  fi
done
while IFS= read -r f; do
  is_dual_fast "$f" && continue
  fail "$f: turns the CUDA-graph memory estimate off — only the dual-fast tier does, by measured decision"
done < <(command grep -rlE 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=\$\{VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0\}' models/*/vllm/compose --include=*.yml | command grep -v _archive)

if [[ "$fails" -gt 0 ]]; then
  echo "test-mamba-ssm-knob: $fails failure(s) across ${#FILES[@]} composes" >&2
  exit 1
fi
echo "test-mamba-ssm-knob: ok (${#FILES[@]} composes default bfloat16; ${#DUAL_FAST[@]} dual-fast with the graph estimate off)"
