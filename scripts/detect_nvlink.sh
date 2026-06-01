#!/bin/bash
# NVLink / PCIe-P2P detection + override. Sources NVLINK_MODE from env (default: auto).
# Exports: _NVLINK_ENABLED (0/1 — "fast P2P interconnect available → custom all-reduce ON")
# and sets NCCL/PYTORCH env vars accordingly.
# Handles 2-GPU setups (single NVLink bridge) and N-GPU setups (e.g. 2 bridges on 4 cards).
#
# NVLINK_MODE values:
#   auto       — detect NVLink via nvidia-smi topo (default). No NVLink => P2P off.
#   force_on   — assert NVLink present (NCCL_P2P_LEVEL=NVL).
#   force_off  — no P2P at all (NCCL_P2P_DISABLE=1).
#   pcie_p2p   — PCIe P2P WITHOUT NVLink (e.g. a patched consumer-GPU driver — the
#                tinygrad/geohot P2P patch). Sets NCCL_P2P_LEVEL=PHB (or your own
#                NCCL_P2P_LEVEL), leaves P2P ENABLED, and turns custom all-reduce ON.
#                Explicit opt-in (like force_on) — auto can't tell the patch is loaded.
#                See club-3090 #290.

NVLINK_MODE="${NVLINK_MODE:-auto}"
_P2P_LEVEL=NVL   # NCCL_P2P_LEVEL used when _NVLINK_ENABLED=1 (overridden by pcie_p2p)

case "$NVLINK_MODE" in
  force_on)
    _NVLINK_ENABLED=1
    echo "[nvlink] NVLINK_MODE=force_on — enabling NVLink mode"
    ;;
  force_off)
    _NVLINK_ENABLED=0
    echo "[nvlink] NVLINK_MODE=force_off — forcing PCIe mode (P2P off)"
    ;;
  pcie_p2p)
    # Explicit opt-in for PCIe P2P (no NVLink) — e.g. a patched consumer-GPU driver.
    _NVLINK_ENABLED=1
    _P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
    echo "[nvlink] NVLINK_MODE=pcie_p2p — forcing PCIe P2P (NCCL_P2P_LEVEL=$_P2P_LEVEL, custom all-reduce ON)"
    ;;
  auto)
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c 'GPU' || echo 0)
    if [ "$GPU_COUNT" -gt 2 ]; then
      # Check topology matrix for any NVLink connections (e.g. 2 bridges on 4 cards).
      if nvidia-smi topo -m 2>/dev/null | grep -qP '\bNV[0-9]+\b'; then
        _NVLINK_ENABLED=1
        echo "[nvlink] $GPU_COUNT GPUs detected — NVLink found, enabling NVLink mode"
      else
        _NVLINK_ENABLED=0
        echo "[nvlink] $GPU_COUNT GPUs detected — no NVLink found, using PCIe mode"
      fi
    elif [ "$GPU_COUNT" -eq 2 ]; then
      LINK=$(nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/{print $3}')
      if [[ "$LINK" =~ ^NV[0-9]+$ ]]; then
        _NVLINK_ENABLED=1
        echo "[nvlink] detected NVLink ($LINK) between GPU0-GPU1 — enabling NVLink mode"
      else
        _NVLINK_ENABLED=0
        echo "[nvlink] PCIe topology ($LINK) — using PCIe mode (no NVLink; for patched-driver PCIe P2P set NVLINK_MODE=pcie_p2p)"
      fi
    else
      _NVLINK_ENABLED=0
      echo "[nvlink] $GPU_COUNT GPU(s) — skipping NVLink detection"
    fi
    ;;
  *)
    echo "[nvlink] ERROR: invalid NVLINK_MODE=$NVLINK_MODE (must be auto|force_on|force_off|pcie_p2p)" >&2
    exit 1
    ;;
esac

# Apply environment overrides based on detection result.
# _NVLINK_ENABLED=1 means a fast P2P interconnect is available (NVLink OR patched PCIe
# P2P) — P2P stays on and the compose entrypoint enables custom all-reduce. The level is
# NVL for NVLink, PHB (or the user's value) for pcie_p2p.
if [ "$_NVLINK_ENABLED" -eq 1 ]; then
  export NCCL_P2P_LEVEL="${_P2P_LEVEL:-NVL}"
  unset NCCL_P2P_DISABLE 2>/dev/null || true
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
  echo "[nvlink] P2P ENABLED — NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL, custom all-reduce ON, expandable_segments OFF"
else
  export NCCL_P2P_DISABLE=1
  unset NCCL_P2P_LEVEL 2>/dev/null || true
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
  echo "[nvlink] P2P DISABLED — NCCL_P2P_DISABLE=1, custom all-reduce OFF, expandable_segments ON"
fi
