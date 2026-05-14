#!/bin/bash
# NVLink auto-detection + override. Sources NVLINK_MODE from env (default: auto).
# Exports: _NVLINK_ENABLED (0 or 1), sets NCCL/PYTORCH env vars accordingly.
# Designed for dual-card (2x GPU) setups. Skips detection on >2 GPUs.

NVLINK_MODE="${NVLINK_MODE:-auto}"

case "$NVLINK_MODE" in
  force_on)
    _NVLINK_ENABLED=1
    echo "[nvlink] NVLINK_MODE=force_on — enabling NVLink mode"
    ;;
  force_off)
    _NVLINK_ENABLED=0
    echo "[nvlink] NVLINK_MODE=force_off — forcing PCIe mode"
    ;;
  auto)
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c 'GPU' || echo 0)
    if [ "$GPU_COUNT" -gt 2 ]; then
      _NVLINK_ENABLED=0
      echo "[nvlink] $GPU_COUNT GPUs detected — skipping NVLink detection (dual-card only)"
    elif [ "$GPU_COUNT" -eq 2 ]; then
      LINK=$(nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/{print $3}')
      if [[ "$LINK" =~ ^NV[0-9]+$ ]]; then
        _NVLINK_ENABLED=1
        echo "[nvlink] detected NVLink ($LINK) between GPU0-GPU1 — enabling NVLink mode"
      else
        _NVLINK_ENABLED=0
        echo "[nvlink] PCIe topology ($LINK) — using PCIe mode"
      fi
    else
      _NVLINK_ENABLED=0
      echo "[nvlink] $GPU_COUNT GPU(s) — skipping NVLink detection"
    fi
    ;;
  *)
    echo "[nvlink] ERROR: invalid NVLINK_MODE=$NVLINK_MODE (must be auto|force_on|force_off)" >&2
    exit 1
    ;;
esac

# Apply environment overrides based on detection result
if [ "$_NVLINK_ENABLED" -eq 1 ]; then
  export NCCL_P2P_LEVEL=NVL
  unset NCCL_P2P_DISABLE 2>/dev/null || true
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
  echo "[nvlink] NVLink ENABLED — NCCL_P2P_LEVEL=NVL, custom all-reduce ON, expandable_segments OFF"
else
  export NCCL_P2P_DISABLE=1
  unset NCCL_P2P_LEVEL 2>/dev/null || true
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
  echo "[nvlink] NVLink DISABLED — NCCL_P2P_DISABLE=1, custom all-reduce OFF, expandable_segments ON"
fi
