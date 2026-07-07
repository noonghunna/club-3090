#!/usr/bin/env bash
# test-gpu-select.sh — the shared GPU-UUID resolver (#610 Phase A) has TWO
# implementations that MUST agree: scripts/lib/gpu-select.sh (bash, for
# launch.sh) and resolve_gpu_uuids() in estate_cli.py (python, for the estate
# boot path). This guard asserts:
#   1. both resolve the SAME indices to the SAME UUID csv (real nvidia-smi);
#   2. both fall back to empty/None identically when a resolver can't run
#      (fake-GPU test mode / no nvidia-smi);
#   3. gpu_select_export sets both CUDA_ and NVIDIA_VISIBLE_DEVICES;
#   4. the launcher sources the lib (no re-inlined resolver drift).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=../lib/gpu-select.sh
source "$ROOT/scripts/lib/gpu-select.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }

# (4) launch.sh sources the lib rather than re-inlining the resolver.
grep -q 'scripts/lib/gpu-select.sh' scripts/launch.sh \
  || fail "launch.sh no longer sources gpu-select.sh (resolver drift risk)"
grep -q 'gpu_select_export' scripts/launch.sh \
  || fail "launch.sh no longer calls gpu_select_export"

# (2) fallback parity: with no real GPUs to resolve, both yield empty/None.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bash_out="$(gpu_select_indices_to_uuids "0,1")"
  [[ -z "$bash_out" ]] || fail "bash resolver should be empty without nvidia-smi (got '$bash_out')"
fi
# python resolver under the fake-GPU seam must return None (empty print).
py_fake="$(CLUB3090_FAKE_GPUS='0:RTX 3090:24576:8.6' python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.lib.profiles.estate_cli import resolve_gpu_uuids
v = resolve_gpu_uuids([0, 1])
print("" if v is None else v)
PY
)"
[[ -z "$py_fake" ]] || fail "python resolver should be None under CLUB3090_FAKE_GPUS (got '$py_fake')"

# (3) gpu_select_export exports both masks (UUID when resolvable, else index).
( gpu_select_export "0" "test" >/dev/null 2>&1
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -n "${NVIDIA_VISIBLE_DEVICES:-}" ]] \
    || { echo "gpu_select_export did not set both masks" >&2; exit 1; }
  [[ "${CUDA_VISIBLE_DEVICES}" == "${NVIDIA_VISIBLE_DEVICES}" ]] \
    || { echo "masks disagree: cuda=${CUDA_VISIBLE_DEVICES} nvidia=${NVIDIA_VISIBLE_DEVICES}" >&2; exit 1; }
) || fail "gpu_select_export contract"

# (1) real-hardware parity: only when nvidia-smi is present with >=1 GPU.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  n="$(nvidia-smi -L | grep -c '^GPU ' || echo 0)"
  if [[ "$n" -ge 1 ]]; then
    bash_uuids="$(gpu_select_indices_to_uuids "0")"
    py_uuids="$(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.lib.profiles.estate_cli import resolve_gpu_uuids
v = resolve_gpu_uuids([0])
print("" if v is None else v)
PY
)"
    [[ "$bash_uuids" == "$py_uuids" ]] \
      || fail "bash/python resolver disagree for GPU 0: bash='$bash_uuids' python='$py_uuids'"
    [[ "$bash_uuids" == GPU-* ]] \
      || fail "resolver returned a non-UUID for GPU 0: '$bash_uuids'"
  fi
fi

echo "PASS: gpu-select.sh + estate_cli resolver agree; launcher sources the lib; export sets both masks"
