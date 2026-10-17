#!/usr/bin/env bash
# qwen4exp-ple-fp8-bytecopy installer: runs before `vllm serve`. See patch_ple_fp8_bytecopy.py.
set -u
export PYTHONUTF8="${PYTHONUTF8:-1}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$here/patch_ple_fp8_bytecopy.py" patch || exit 1
