#!/usr/bin/env bash
# qwen4exp-pp-ple-rank0 installer: runs in the compose entrypoint before `vllm serve`.
# Lets Qwen3.8-Flash-Next run with --pipeline-parallel-size > 1 on stock vLLM.
# See patch_pp_ple_rank0.py for what it changes and why. Anchor missing -> exit 1.
set -u
export PYTHONUTF8="${PYTHONUTF8:-1}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$here/patch_pp_ple_rank0.py" patch || exit 1
