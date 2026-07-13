#!/bin/bash
# NVLink / PCIe-P2P detection + override. Sources NVLINK_MODE from env (default: auto).
# Exports: _NVLINK_ENABLED (0/1 — "fast P2P interconnect available → custom all-reduce ON")
# and sets NCCL/PYTORCH env vars accordingly.
# Handles 2-GPU setups (single NVLink bridge) and N-GPU setups (e.g. 2 bridges on 4 cards).
#
# NVLINK_MODE values:
#   auto       — detect a fast P2P interconnect via nvidia-smi (default): NVLink (topo -m)
#                OR, failing that, PCIe P2P that `nvidia-smi topo -p2p r` reports as OK
#                between all pairs — i.e. a patched consumer-GPU driver (tinygrad/geohot/
#                aikitoria) on a P2P-capable layout (shared root complex / switch). Neither
#                => P2P off. NOTE: stock GeForce drivers software-disable P2P (report CNS),
#                and cards on separate root complexes can't P2P — both correctly stay off.
#   force_on   — assert NVLink present (NCCL_P2P_LEVEL=NVL).
#   force_off  — no P2P at all (NCCL_P2P_DISABLE=1).
#   pcie_p2p   — FORCE the PCIe P2P config on (NCCL_P2P_LEVEL=PHB or your own,
#                custom all-reduce ON), bypassing auto-detect. It ALSO probes
#                `topo -p2p` and reports honestly: "P2P ENABLED" when the driver
#                confirms peer access, "P2P REQUESTED (UNVERIFIED)" + a warning
#                when it doesn't (a closed GeForce driver refuses P2P; the open
#                kernel modules / a patched module + a P2P-capable board enable
#                it). The config is forced either way — this is the escape hatch
#                for a rig whose probe is stricter than reality — but we never
#                claim "engaged" without evidence. See club-3090 #290, #688.

NVLINK_MODE="${NVLINK_MODE:-auto}"
_P2P_LEVEL=NVL   # NCCL_P2P_LEVEL used when _NVLINK_ENABLED=1 (overridden by pcie_p2p)

# True (0) when nvidia-smi reports working P2P between ALL GPU pairs — e.g. a patched
# consumer-GPU driver (NVIDIA's stock driver software-disables P2P → reports "CNS") on a
# P2P-capable PCIe layout. Parses `topo -p2p r`: a data row carries the self-"X" (header /
# legend rows don't, so they're skipped); ANY off-diagonal cell that isn't OK => unavailable.
_pcie_p2p_available() {
  nvidia-smi topo -p2p r 2>/dev/null | awk '
    $1 ~ /^GPU[0-9]+$/ {
      hasX = 0
      for (i = 2; i <= NF; i++) if ($i == "X") hasX = 1
      if (!hasX) next                                  # header row (no self-X) — skip
      rows++
      for (i = 2; i <= NF; i++) if ($i != "X" && $i != "OK") bad = 1
    }
    END { exit (rows > 0 && !bad) ? 0 : 1 }
  '
}

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
    # Explicit opt-in for PCIe P2P (no NVLink) — a patched module OR a driver/board
    # that grants peer access (some server boards + the open kernel modules do).
    # Force the config on either way (the escape hatch for a rig whose topo -p2p
    # probe is stricter than reality), but VERIFY peer access and flag it honestly
    # when the driver didn't grant it — else we'd falsely report "P2P engaged" on a
    # closed/stock driver that silently refuses it (club-3090 #688).
    _NVLINK_ENABLED=1
    _P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
    if _pcie_p2p_available; then
      echo "[nvlink] NVLINK_MODE=pcie_p2p — forcing PCIe P2P; driver confirms peer access (nvidia-smi topo -p2p: OK) — NCCL_P2P_LEVEL=$_P2P_LEVEL, custom all-reduce ON"
    else
      _P2P_UNVERIFIED=1
      echo "[nvlink] WARNING: NVLINK_MODE=pcie_p2p set, but nvidia-smi topo -p2p does NOT report peer access as OK — the driver likely refused P2P (a closed GeForce driver disables it; the open kernel modules or a patched module + a P2P-capable board are what enable it). Forcing the NCCL/all-reduce config on as requested, but NCCL will silently fall back → throughput ≈ P2P-off. Verify with: nvidia-smi topo -p2p rw. Guide: docs/PCIE_P2P.md" >&2
    fi
    ;;
  auto)
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c 'GPU' || echo 0)
    if [ "$GPU_COUNT" -gt 2 ]; then
      # Check topology matrix for any NVLink connections (e.g. 2 bridges on 4 cards).
      if nvidia-smi topo -m 2>/dev/null | grep -qP '\bNV[0-9]+\b'; then
        _NVLINK_ENABLED=1
        echo "[nvlink] $GPU_COUNT GPUs detected — NVLink found, enabling NVLink mode"
      elif _pcie_p2p_available; then
        _NVLINK_ENABLED=1; _P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
        echo "[nvlink] $GPU_COUNT GPUs — no NVLink, but nvidia-smi reports P2P=OK (patched driver / P2P-capable layout) — auto-enabling PCIe P2P (NCCL_P2P_LEVEL=$_P2P_LEVEL, custom all-reduce ON)"
      else
        _NVLINK_ENABLED=0
        echo "[nvlink] $GPU_COUNT GPUs detected — no NVLink, no P2P — using PCIe mode"
      fi
    elif [ "$GPU_COUNT" -eq 2 ]; then
      LINK=$(nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/{print $3}')
      if [[ "$LINK" =~ ^NV[0-9]+$ ]]; then
        _NVLINK_ENABLED=1
        echo "[nvlink] detected NVLink ($LINK) between GPU0-GPU1 — enabling NVLink mode"
      elif _pcie_p2p_available; then
        _NVLINK_ENABLED=1; _P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
        echo "[nvlink] PCIe topology ($LINK) but nvidia-smi reports P2P=OK (patched driver / shared root complex) — auto-enabling PCIe P2P (NCCL_P2P_LEVEL=$_P2P_LEVEL, custom all-reduce ON)"
      else
        _NVLINK_ENABLED=0
        echo "[nvlink] PCIe topology ($LINK), P2P not available (topo -p2p: no OK) — using PCIe mode (no P2P; for a patched driver on a P2P-capable layout this auto-enables, or set NVLINK_MODE=pcie_p2p to force)"
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
  # custom all-reduce is ON here. expandable_segments backs allocations with a
  # cuMemMap VA range, and cudaIpcGetMemHandle on that range fails during graph-
  # buffer registration (custom_all_reduce.cuh "invalid argument") — so it MUST
  # be off on this path. Dual composes inject
  # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,... for the PCIe path, so a
  # plain ${VAR:-default} would keep that crashing value. Strip ONLY the
  # expandable_segments token and preserve any other knobs the user set
  # (max_split_size_mb, garbage_collection_threshold, ...). See docs/UPSTREAM.md.
  _alloc="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
  _alloc="$(printf '%s' "$_alloc" | sed -E 's/(^|,)expandable_segments:[^,]*//g; s/^,+//; s/,+$//; s/,+/,/g')"
  [ -n "$_alloc" ] || _alloc="max_split_size_mb:512"
  export PYTORCH_CUDA_ALLOC_CONF="$_alloc"
  if [ "${_P2P_UNVERIFIED:-0}" -eq 1 ]; then
    # pcie_p2p forced, but topo -p2p didn't confirm peer access. Config is applied;
    # engagement is NOT proven. Say so — never print "ENABLED" without evidence (#688).
    echo "[nvlink] P2P REQUESTED (UNVERIFIED) — NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL + custom all-reduce configured as forced, but peer access is UNCONFIRMED (topo -p2p ≠ OK; see the warning above). If the driver refused P2P, NCCL falls back and throughput ≈ P2P-off — verify with nvidia-smi topo -p2p rw. expandable_segments stripped (PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF)"
  else
    echo "[nvlink] P2P ENABLED — NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL, custom all-reduce ON, expandable_segments stripped (PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF)"
  fi
else
  export NCCL_P2P_DISABLE=1
  unset NCCL_P2P_LEVEL 2>/dev/null || true
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
  echo "[nvlink] P2P DISABLED — NCCL_P2P_DISABLE=1, custom all-reduce OFF, expandable_segments ON"
fi

unset -f _pcie_p2p_available 2>/dev/null || true   # don't leak the probe into the sourcing shell
