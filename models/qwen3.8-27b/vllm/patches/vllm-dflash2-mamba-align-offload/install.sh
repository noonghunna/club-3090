#!/usr/bin/env bash
# club-3090 DFlash2 mamba-align offload fix installer — runs in the container
# entrypoint before serve. Ports upstream vLLM commit fa5017a5 ("Scope
# prefix-cache last-block drop to eagle-family drafters (DFlash2 offload fix)")
# which is NOT yet in vllm/vllm-openai:v0.29.0 (lands on main 2026-09-10, one
# day after the 0.29.0 cut). Without it, DFlash2 + a KV connector (LMCache or
# native offload) gets the EAGLE volatile-trailing-block drop on the mamba
# group, so the final block-aligned mamba snapshot never materializes and
# prefix-cache/offload lookups converge to 0 (vllm#53505).
# Idempotent; refuses boot (exit 1) on anchor drift so a re-pinned image cannot
# silently serve with offloaded cache hits broken.
set -u
DIR=/etc/club3090/dflash2-mamba-align
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
if python3 "$DIR/_check_applied.py" "$DIR/dflash2-mamba-align-offload.patch" "$VLLM" 2>/dev/null; then
  echo "[dflash2-mamba-align] fa5017a5 port already present — skipping" >&2; exit 0
fi
if ( cd "$VLLM" && patch -p1 --forward --batch < "$DIR/dflash2-mamba-align-offload.patch" >/tmp/dflash2-mamba-align.patch.log 2>&1 ); then
  echo "[dflash2-mamba-align] applied fa5017a5 port (DFlash2 mamba-align offload fix)" >&2
else
  echo "[dflash2-mamba-align] FAILED to apply dflash2-mamba-align-offload.patch — refusing boot:" >&2
  tail -20 /tmp/dflash2-mamba-align.patch.log >&2; exit 1
fi
