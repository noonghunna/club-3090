#!/usr/bin/env bash
# club-3090 DFlash2 backport installer — runs in the container entrypoint before
# serve. Vendors vLLM PR #52816 (syv-ai v0.27.1 backport): adds the
# DFlash2DraftModel architecture (grouped depthwise conv + candidate selector) so
# the EXTERNAL DFlash2 block drafter serves via --speculative-config method=dflash.
# Idempotent; refuses boot (exit 1) on apply failure so a re-pinned image cannot
# silently serve unpatched while the compose claims DFlash2 support.
set -u
DIR=/etc/club3090/dflash2
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
python3 "$DIR/patch_topk_4arg_compat.py" || exit 1
if python3 "$DIR/_check_applied.py" "$DIR/dflash2-backport.patch" "$VLLM" 2>/dev/null; then
  echo "[dflash2] PR#52816 backport already present — skipping" >&2; exit 0
fi
if ( cd "$VLLM" && patch -p1 --forward --batch < "$DIR/dflash2-backport.patch" >/tmp/dflash2.patch.log 2>&1 ); then
  echo "[dflash2] applied vLLM PR#52816 backport (DFlash2DraftModel)" >&2
else
  echo "[dflash2] FAILED to apply dflash2-backport.patch — refusing boot:" >&2
  tail -20 /tmp/dflash2.patch.log >&2; exit 1
fi
