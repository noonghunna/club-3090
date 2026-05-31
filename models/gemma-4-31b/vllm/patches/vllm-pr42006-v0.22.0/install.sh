#!/bin/bash
# ===========================================================================
# vLLM PR #42006 — Gemma 4 MTP streaming multi-tool-call fix (lean diff-apply)
#
# Without it, on STOCK v0.22.0 a streamed response containing 2+ tool calls
# drops the ARGUMENTS of every call except the last (e.g. an agent streaming
# get_weather(Tokyo)+get_weather(London) gets two argument-less calls).
# Confirmed live on this rig 2026-05-31: stock streaming → args lost on calls
# 0..n-1; non-streaming → all args correct. So it's a streaming-parser bug.
#
# Single-file boot-time diff onto the pinned stock v0.22.0 gemma4 tool parser.
# Carried by BOTH gemma duals (bf16-mtp + int8-mtp share the bug). Idempotent;
# fail-loud. Drop when #42006 merges upstream + lands in the pinned release:
#   gh api repos/vllm-project/vllm/pulls/42006 --jq '.state, .merged_at'
# ===========================================================================
set -euo pipefail

VLLM=/usr/local/lib/python3.12/dist-packages/vllm
SITE=/usr/local/lib/python3.12/dist-packages
PATCHDIR=/etc/club3090/pr42006

# already-applied / upstream-merged detection (no-op cleanly)
if grep -q "_extract_streaming_delta_segments\|_split_delta_text_on_tool_tokens" \
     "$VLLM/tool_parsers/gemma4_tool_parser.py" 2>/dev/null; then
  echo "[pr42006] streaming multi-tool fix already present — skipping overlay."
  exit 0
fi

cd "$SITE"
if patch -p1 --forward --batch --reject-file=/tmp/pr42006.rej < "$PATCHDIR/pr42006-v0.22.0.patch"; then
  echo "[pr42006] streaming multi-tool fix applied cleanly."
else
  echo "[pr42006] FATAL: #42006 diff did not apply to this image — refusing to boot." >&2
  cat /tmp/pr42006.rej >&2 2>/dev/null || true
  exit 1
fi
