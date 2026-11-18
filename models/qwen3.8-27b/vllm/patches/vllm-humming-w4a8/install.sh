#!/usr/bin/env bash
# W4A8 on vLLM v0.34.0's humming backend (int4 AutoRound weights, int8 activations).
#
# The compose entrypoint runs this only when W4A8 selects `--quantization humming`.
# apply.py patches the installed vLLM in place:
#   1. fallback init   — vllm#68194, the fix from upstream PR vllm#68202
#   2. placeholder     — vllm#68195, ours (no upstream PR yet)
#   3. activation log  — one line per worker naming the effective a_dtype
# Details and drop triggers: README.md here and the `vllm-humming-w4a8` entry
# in scripts/lib/profiles/patches.yml.
#
# Failure policy: any failure stops the boot. Without fix 1 humming cannot load
# our checkpoints; without fix 2 it doubles the weight memory and 262K context
# no longer fits. Serving W4A16 instead is a W4A8=0 decision, not a fallback.
set -u
export PYTHONUTF8="${PYTHONUTF8:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! python3 "$HERE/apply.py"; then
    echo "[humming-w4a8] FATAL: the humming fixes did not apply (drifted VLLM_IMAGE?)." >&2
    echo "[humming-w4a8] Refusing to serve W4A8. Boot with W4A8=0 for W4A16 instead." >&2
    exit 1
fi
