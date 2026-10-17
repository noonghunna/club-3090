#!/usr/bin/env bash
# qwen4exp-qsa-fp8kv-ampere installer: runs before `vllm serve`. See patch_qsa_fp8kv_ampere.py.
# No effect at runtime unless the KV cache is fp8 and the GPU is below sm_89.
set -u
export PYTHONUTF8="${PYTHONUTF8:-1}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$here/patch_qsa_fp8kv_ampere.py" patch || exit 1
