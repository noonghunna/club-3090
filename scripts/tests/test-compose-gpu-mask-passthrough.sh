#!/usr/bin/env bash
# test-compose-gpu-mask-passthrough.sh — every compose that selects GPUs via
# the NVIDIA_VISIBLE_DEVICES env line MUST also pass CUDA_VISIBLE_DEVICES
# through (#610): CDI runtimes (NixOS, nvidia-ctk cdi) ignore
# NVIDIA_VISIBLE_DEVICES entirely, so the CUDA-level mask is the only thing
# that pins cards there. launch.sh --gpus exports both as GPU UUIDs
# (renumbering-proof on the classic runtime too). A new compose that copies
# the NVIDIA line but forgets the CUDA passthrough silently regresses CDI
# rigs — this guard reds instead.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fails=0
while IFS= read -r f; do
  # bare passthrough form: "- CUDA_VISIBLE_DEVICES" (no =value; unset → absent)
  if ! grep -Eq '^\s*- CUDA_VISIBLE_DEVICES\s*$' "$f"; then
    echo "FAIL: $f sets NVIDIA_VISIBLE_DEVICES but does not pass CUDA_VISIBLE_DEVICES through (#610 CDI regression)" >&2
    fails=$((fails+1))
  fi
done < <(grep -rlE '^\s*- NVIDIA_VISIBLE_DEVICES=' models/*/*/compose --include='*.yml' 2>/dev/null | grep -v '/_archive/' || true)

# launch.sh must still carry the UUID resolution (the other half of #610).
grep -q 'query-gpu=uuid' scripts/launch.sh \
  || { echo "FAIL: launch.sh lost the --gpus index→UUID resolution (#610)" >&2; fails=$((fails+1)); }

if [[ $fails -gt 0 ]]; then
  echo "$fails GPU-mask passthrough check(s) failed" >&2
  exit 1
fi
echo "PASS: all NVIDIA_VISIBLE_DEVICES composes pass CUDA_VISIBLE_DEVICES through; launch.sh UUID resolution present"
