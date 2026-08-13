# Requires -Version 5.1
#
# detect_nvlink.ps1 - NVLink / PCIe-P2P detection + override
#
# This script now delegates to nvlink-env.ps1 for the actual detection logic,
# keeping a thin wrapper for backward compatibility.
#
# Usage:
#   .\detect_nvlink.ps1
#   $env:NVLINK_MODE="force_on"; .\detect_nvlink.ps1
#   $env:NVLINK_MODE="force_off"; .\detect_nvlink.ps1
#   $env:NVLINK_MODE="pcie_p2p"; .\detect_nvlink.ps1
#
# Exports: _NVLINK_ENABLED (0/1), NCCL_P2P_LEVEL, NCCL_P2P_DISABLE

$ErrorActionPreference = "Stop"
$ScriptName = "detect_nvlink"

# Delegate to the shared nvlink-env module
. "$PSScriptRoot\nvlink-env.ps1"

# Output results (nvlink-env already prints its own summary, but we keep
# the old format for backward compatibility with scripts that parse output)
Write-Host ""
Write-Host "=== detect_nvlink summary ==="
Write-Host "_NVLINK_ENABLED=$script:NVLINK_ENABLED"
Write-Host "NCCL_P2P_LEVEL=$($script:NCCL_P2P_LEVEL)"
Write-Host "NCCL_P2P_DISABLE=$($script:NCCL_P2P_DISABLE)"
Write-Host "PYTORCH_CUDA_ALLOC_CONF=$($script:PYTORCH_CUDA_ALLOC_CONF)"
