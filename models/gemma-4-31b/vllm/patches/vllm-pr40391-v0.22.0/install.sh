#!/bin/bash
# ===========================================================================
# vLLM PR #40391 — Gemma 4 per-token-head KV cache page-size alignment
# Lean overlay for the PINNED stock vllm/vllm-openai:v0.22.0 image.
#
# Delivers the patch as a ~240-line diff + one new helper file, applied to the
# stock site-package at container boot — NOT as 7 full-module file mounts (the
# old vllm-pr40391-rebased/ approach, which vendored ~13K lines of v0.22.0 vLLM
# source and would silently revert any other v0.22.0.x change to those files on
# a future re-pin). This carries ONLY the genuine #40391 delta.
#
# Idempotent: no-ops cleanly if #40391 is already present (e.g. a future image
# where it has merged upstream). Fails LOUD if the diff doesn't apply (so a
# silent-wrong-overlay never ships).
#
# Drop this overlay entirely when PR #40391 merges upstream + lands in a pinned
# release:  gh api repos/vllm-project/vllm/pulls/40391 --jq '.state, .merged_at'
# ===========================================================================
set -euo pipefail

VLLM=/usr/local/lib/python3.12/dist-packages/vllm
SITE=/usr/local/lib/python3.12/dist-packages
PATCHDIR=/etc/club3090/pr40391

# --- upstream-merged / already-applied detection (no-op cleanly) ------------
if [ -f "$VLLM/v1/worker/kv_cache_shape_utils.py" ] \
   && grep -q "kv_cache_page_size_padded" "$VLLM/model_executor/layers/attention/attention.py" 2>/dev/null; then
  echo "[pr40391] #40391 already present in this image — skipping overlay."
  exit 0
fi

# --- new file (added by the PR; not a diff target) --------------------------
cp "$PATCHDIR/kv_cache_shape_utils.py" "$VLLM/v1/worker/kv_cache_shape_utils.py"
echo "[pr40391] installed kv_cache_shape_utils.py"

# --- apply the corrected #40391 diff onto stock v0.22.0 ---------------------
# Paths in the patch are a/vllm/... b/vllm/... ; -p1 strips the leading a/, so
# apply from the site-packages dir (the parent of vllm/).
cd "$SITE"
if patch -p1 --forward --batch --reject-file=/tmp/pr40391.rej < "$PATCHDIR/pr40391-v0.22.0.patch"; then
  echo "[pr40391] diff applied cleanly."
else
  echo "[pr40391] FATAL: #40391 diff did not apply to this image — refusing to boot a half-patched engine." >&2
  echo "[pr40391] rejects: $(cat /tmp/pr40391.rej 2>/dev/null | head -40)" >&2
  exit 1
fi
