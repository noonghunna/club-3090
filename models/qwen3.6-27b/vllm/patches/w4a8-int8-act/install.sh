#!/usr/bin/env bash
# W4A8 (int4 weights + int8 activations) enablement patches — #609.
#
# Applies two pure-Python fixes to the container's installed vLLM so the STOCK
# serve-time toggle `VLLM_MARLIN_INPUT_DTYPE=int8` works on this model's
# auto-round checkpoints:
#   patch-inc-w4a8.py       — vLLM's `inc` route (all auto-round ckpts) drops the
#                             toggle's input_dtype; 3-line thread-through.
#   patch-negscale-fold.py  — the int8 kernel reads group scales UNSIGNED;
#                             AutoRound emits ~50% negative scales → garbage.
#                             Folds the sign into the weight codes at load
#                             (+ asym-checkpoint guards). Details: #609.
#
# BOTH patches are NO-OPS while VLLM_MARLIN_INPUT_DTYPE is unset (the fold is
# is_a_8bit-gated; the hook assigns None), so this runs unconditionally with
# zero behavior change until the knob is flipped. Idempotent, anchor-based;
# anchors verified byte-stable on vLLM v0.24.0 & v0.25.1.
#
# Failure policy: if the anchors drift (custom VLLM_IMAGE), fail the BOOT only
# when the user actually asked for int8 — a silent half-apply would mean the
# knob silently does nothing (or worse); without the knob the patches are
# unnecessary, so warn and serve normally.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if python3 "$HERE/patch-inc-w4a8.py" && python3 "$HERE/patch-negscale-fold.py"; then
    exit 0
fi
if [ -n "${VLLM_MARLIN_INPUT_DTYPE:-}" ]; then
    echo "[w4a8] FATAL: VLLM_MARLIN_INPUT_DTYPE is set but the W4A8 patches failed to apply" >&2
    echo "[w4a8] (custom VLLM_IMAGE with drifted anchors?) — refusing to serve a half-wired int8 config." >&2
    exit 1
fi
echo "[w4a8] WARN: W4A8 patches did not apply (custom image?) — int8 activations unavailable on this boot." >&2
exit 0
