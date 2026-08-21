#!/usr/bin/env bash
# DFLASH15 opt-in patches for vLLM 0.27.1.
set -euo pipefail
DIR=/etc/club3090/dflash15
VLLM=/usr/local/lib/python3.12/dist-packages/vllm

apply_patch() {
  local file=$1 marker_file=$2 marker=$3
  if grep -qs "$marker" "$VLLM/$marker_file"; then
    echo "[dflash15] $(basename "$file") already applied" >&2
    return
  fi
  if (cd "$VLLM" && patch -p1 --forward --batch < "$DIR/$file" >/tmp/dflash15-$(basename "$file").log 2>&1); then
    echo "[dflash15] applied $(basename "$file")" >&2
  else
    echo "[dflash15] FAILED $(basename "$file") — refusing boot" >&2
    tail -30 "/tmp/dflash15-$(basename "$file").log" >&2
    exit 1
  fi
}

# Hybrid KV grouping keeps the drafter's sliding-window layers from padding
# every target layer; the explicit CUDAGraph reserve makes the KV budget honest.
apply_patch hybrid-kv-groups-v2-cudagraph.patch v1/core/kv_cache_utils.py 'def _prefer_padding_sliding_window_buckets'

# The main Qwen3.5 model needs the quantized embedding path. The stock file
# already contains an unrelated quant_config use elsewhere, so anchor on the
# embed_tokens prefix added by this patch.
apply_patch qwen3_5-embed-quant.patch model_executor/models/qwen3_5.py 'prefix=maybe_prefix(prefix, "embed_tokens")'
apply_patch spec-decode-attn.patch v1/attention/backends/flash_attn.py "split-KV Triton attention"
apply_patch dflash2-lookup-drafting.patch v1/worker/gpu/spec_decode/dflash2/lookup.py "def suffix_lookup"
