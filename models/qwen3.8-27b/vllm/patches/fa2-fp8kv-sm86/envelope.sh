#!/usr/bin/env bash
# fa2_envelope — keep vLLM's arguments inside the FA2 fp8-KV kernel's envelope (#1358).
#
# Sourced by every compose that mounts this patch, right before `exec vllm serve`:
#
#   source /etc/club3090/fa2/envelope.sh
#   fa2_envelope "$@" || exit 1
#   set -- "${FA2_ARGS[@]}"
#
# WHY. The plugin's kernel has two bounds (`fp8_attn.cu`: max_q <= 2048 && max_k <=
# 262144, "Sequence bounds exceed this port's supported envelope"), and breaking
# either is fatal to the engine.
#   - max_k: a context past 262144 tokens. Nothing in the plugin routes around it, so
#     a --max-model-len above it would fail at the first request that long.
#   - max_q: a query chunk past 2048 tokens. Until the artifact's v0.1.0 this helper
#     also capped vLLM's per-request prefill chunk at 2048 (--long-prefill-token-
#     threshold) whenever fp8 KV ran with --max-num-seqs > 1: the plugin took the
#     bounded prefill route only for a batch of ONE request, so a longer chunk next to
#     another request reached the capped kernel. v0.1.0 (Anton's, 2026-10-04; carried
#     by our fork's build) splits long causal prefill in a mixed batch into per-request
#     paged calls of at most 2048 query tokens itself (backend.py:
#     prefill_query_starts -> PagedPrefill.forward_batch), so the cap is gone.
#     MEASURED 2026-11-11 on 2x 3090, vllm/qwen38-27b-dual-ultrafast at MAX_NUM_SEQS=2
#     and threshold 4096, with this cap removed: 36K-token needle prompts prefilled
#     beside a running decode, and two at once, passed with DFlash2 off and on, 0
#     restarts; the old artifact (731d1942, v0.32.0) on the same workload crashed with
#     the message above. The draft model's non-causal attention never reaches max_q at
#     these settings: its queries do not grow with the prefill chunk.
#
# WHAT. Only when the KV cache is fp8 (the plugin hands bf16 to stock FlashAttention,
# where none of this applies): --max-model-len > 262144 -> refuse to start. Every
# other argument passes through unchanged.
#
# Output: the argument list in the array FA2_ARGS. Returns 1 on refusal.

FA2_MAX_K=262144

fa2_envelope() {
  FA2_ARGS=("$@")
  local kv="" model_len=""
  local -a args=("$@")
  local i n=${#args[@]} a
  for ((i = 0; i < n; i++)); do
    a="${args[i]}"
    case "$a" in
      --kv-cache-dtype=*) kv="${a#*=}" ;;
      --kv-cache-dtype) kv="${args[i+1]:-}" ;;
      --max-model-len=*) model_len="${a#*=}" ;;
      --max-model-len) model_len="${args[i+1]:-}" ;;
    esac
  done

  case "$kv" in
    fp8|fp8_e4m3) ;;
    *) return 0 ;;   # bf16 / auto: stock FlashAttention, no envelope
  esac

  if [[ "$model_len" =~ ^[0-9]+$ ]] && (( model_len > FA2_MAX_K )); then
    echo "[fa2] --max-model-len ${model_len} exceeds the fp8-KV kernel's max_k ${FA2_MAX_K};" \
      "it would fail at the first request that long (#1358). Lower MAX_MODEL_LEN or use bf16 KV." >&2
    return 1
  fi
  return 0
}
