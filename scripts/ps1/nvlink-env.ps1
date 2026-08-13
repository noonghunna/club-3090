#Requires -Version 5.1
#
# nvlink-env.ps1 — Shared NVLink environment configuration module
#
# This script centralizes NVLink/P2P environment variable detection so that
# multiple scripts can share consistent settings without re-running detection.
#
# Usage:
#   . "$PSScriptRoot\nvlink-env.ps1"
#   # After sourcing, these variables are available:
#   #   $NVLINK_ENABLED    - 0 or 1
#   #   $NCCL_P2P_LEVEL    - "NVL", "PHB", or "NVLL"
#   #   $NCCL_P2P_DISABLE  - "1" if disabled, unset if enabled
#   #   $PYTORCH_CUDA_ALLOC_CONF - CUDA allocation config
#   #   $GPU_COUNT         - Number of GPUs detected
#
#   # Or call explicitly:
#   . "$PSScriptRoot\nvlink-env.ps1" -Apply
#
#   # Override mode:
#   $env:NVLINK_MODE="force_on"; . "$PSScriptRoot\nvlink-env.ps1"

param(
    [switch]$Apply
)

$ScriptName = "nvlink-env"

function Log { param($Msg) Write-Host "[$ScriptName] $Msg" }

# Get GPU count
$GPU_COUNT = 0
try {
    $gpuLines = nvidia-smi -L 2>$null
    if ($gpuLines) {
        $GPU_COUNT = ($gpuLines -split "`n" | Where-Object { $_ -match 'GPU' }).Count
    }
} catch { }

$_NvlinkEnabled = 0
$_P2PLevel = "NVL"
$_NvlinkPartial = 0

function Test-PcieP2pAvailable {
    try {
        $p2p = nvidia-smi topo -p2p r 2>$null
        if (-not $p2p) { return $false }
        $hasX = $false
        $bad = $false
        foreach ($line in $p2p -split "`n") {
            if ($line -match '^GPU\d+') {
                $hasX = $line -match 'X'
                if (-not $hasX) { continue }
                if ($line -match '(?<!X)(?!X)([^X\s]|$)') { $bad = $true }
            }
        }
        return ($hasX -and -not $bad)
    } catch { return $false }
}

function Get-ArClaim {
    if ($GPU_COUNT -gt 2 -and $_P2PLevel -ne "NVL") {
        return "custom all-reduce engine-gated (vLLM disables its custom kernel at >2 PCIe-only GPUs)"
    } elseif ($GPU_COUNT -gt 2 -and $_NvlinkPartial -eq 1) {
        return "custom all-reduce engine-gated (NVLink mesh not fully connected)"
    } else {
        return "custom all-reduce ON"
    }
}

$NvlinkMode = if ($env:NVLINK_MODE) { $env:NVLINK_MODE } else { "auto" }

switch ($NvlinkMode) {
    "force_on" {
        $_NvlinkEnabled = 1
        Log "NVLINK_MODE=force_on - enabling NVLink mode"
    }
    "force_off" {
        $_NvlinkEnabled = 0
        Log "NVLINK_MODE=force_off - forcing PCIe mode (P2P off)"
    }
    "pcie_p2p" {
        $_NvlinkEnabled = 1
        $_P2PLevel = if ($env:NCCL_P2P_LEVEL) { $env:NCCL_P2P_LEVEL } else { "PHB" }
        if (Test-PcieP2pAvailable) {
            Log "NVLINK_MODE=pcie_p2p - forcing PCIe P2P; driver confirms peer access - NCCL_P2P_LEVEL=$_P2PLevel, $(Get-ArClaim)"
        } else {
            $_P2PUnverified = 1
            Log "WARNING: NVLINK_MODE=pcie_p2p set, but nvidia-smi topo -p2p does NOT report peer access as OK"
        }
    }
    "auto" {
        if ($GPU_COUNT -gt 2) {
            try {
                $topo = nvidia-smi topo -m 2>$null
                if ($topo -match '\bNV[0-9]+\b') {
                    $_NvlinkEnabled = 1
                    # Check full mesh
                    $topoLines = $topo -split "`n"
                    $partial = $false
                    foreach ($line in $topoLines) {
                        if ($line -match '^GPU\d+' -and $line -match 'X') {
                            if ($line -match '(?<!X)(?!X)([^X\s]|$)') { $partial = $true }
                        }
                    }
                    if ($partial) {
                        $_NvlinkPartial = 1
                        Log "$GPU_COUNT GPUs - NVLink found but mesh NOT fully connected (pairwise bridges)"
                    } else {
                        Log "$GPU_COUNT GPUs - NVLink full mesh, enabling NVLink mode"
                    }
                } elseif (Test-PcieP2pAvailable) {
                    $_NvlinkEnabled = 1
                    $_P2PLevel = if ($env:NCCL_P2P_LEVEL) { $env:NCCL_P2P_LEVEL } else { "PHB" }
                    Log "$GPU_COUNT GPUs - no NVLink, but P2P=OK - auto-enabling PCIe P2P"
                } else {
                    $_NvlinkEnabled = 0
                    Log "$GPU_COUNT GPUs - no NVLink, no P2P - using PCIe mode"
                }
            } catch {
                $_NvlinkEnabled = 0
            }
        } elseif ($GPU_COUNT -eq 2) {
            try {
                $topo = nvidia-smi topo -m 2>$null
                $linkMatch = $topo | Select-String -Pattern 'GPU0\s+\w+\s+(NV\d+)'
                if ($linkMatch.Matches[0].Groups[1].Value -match '^NV\d+$') {
                    $_NvlinkEnabled = 1
                    Log "detected NVLink ($($linkMatch.Matches[0].Groups[1].Value)) between GPU0-GPU1"
                } elseif (Test-PcieP2pAvailable) {
                    $_NvlinkEnabled = 1
                    $_P2PLevel = if ($env:NCCL_P2P_LEVEL) { $env:NCCL_P2P_LEVEL } else { "PHB" }
                    Log "PCIe topology but P2P=OK - auto-enabling PCIe P2P"
                } else {
                    $_NvlinkEnabled = 0
                    Log "PCIe topology, P2P not available - using PCIe mode"
                }
            } catch {
                $_NvlinkEnabled = 0
            }
        } else {
            $_NvlinkEnabled = 0
            Log "$GPU_COUNT GPU(s) - skipping NVLink detection"
        }
    }
    default {
        Write-Error "[$ScriptName] invalid NVLINK_MODE=$NvlinkMode (must be auto|force_on|force_off|pcie_p2p)"
        exit 1
    }
}

# Apply environment overrides
if ($_NvlinkEnabled -eq 1) {
    $env:NCCL_P2P_LEVEL = $_P2PLevel
    if ($env:PSModulePath) { $null = Remove-Item env:NCCL_P2P_DISABLE -ErrorAction SilentlyContinue }
    # Strip expandable_segments
    $alloc = if ($env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF } else { "max_split_size_mb:512" }
    $alloc = ($alloc -replace '(^|,)expandable_segments:[^,]*', '').TrimStart(',').TrimEnd(',')
    if (-not $alloc) { $alloc = "max_split_size_mb:512" }
    $env:PYTORCH_CUDA_ALLOC_CONF = $alloc

    # BAR1 check for PCIe P2P
    if ($_P2PLevel -ne "NVL") {
        try {
            $bar1Info = nvidia-smi -q -d MEMORY 2>$null
            $bar1Bad = ""
            $idx = 0; $sect = ""
            $fb = @{}; $bar1 = @{}
            foreach ($line in $bar1Info -split "`n") {
                if ($line -match '^GPU ') { $idx++; $sect = ""; continue }
                if ($line -match 'FB Memory Usage') { $sect = "fb"; continue }
                if ($line -match 'BAR1 Memory Usage') { $sect = "bar1"; continue }
                if ($line -match 'Memory Usage') { $sect = ""; continue }
                if ($sect -and $line -match 'Total\s+(\d+)') {
                    $val = [int]$Matches[1]
                    if ($sect -eq "fb") { $fb[$idx] = $val } else { $bar1[$idx] = $val }
                    $sect = ""
                }
            }
            $badCards = @()
            foreach ($i in $fb.Keys) {
                if ($fb[$i] -gt 0 -and $bar1.ContainsKey($i) -and $bar1[$i] -gt 0 -and $bar1[$i] * 2 -lt $fb[$i]) {
                    $badCards += "GPU$($i-1) BAR1=$($bar1[$i])MiB VRAM=$($fb[$i])MiB"
                }
            }
            if ($badCards.Count -gt 0) {
                Log "WARNING: BAR1 is far smaller than VRAM on: $($badCards -join ', ')"
            }
        } catch { }
    }

    if ($_P2PUnverified) {
        Log "P2P REQUESTED (UNVERIFIED) - NCCL_P2P_LEVEL=$env:NCCL_P2P_LEVEL + custom all-reduce configured as forced"
    } else {
        Log "P2P ENABLED - NCCL_P2P_LEVEL=$env:NCCL_P2P_LEVEL, $(Get-ArClaim)"
    }
} else {
    $env:NCCL_P2P_DISABLE = "1"
    if ($env:PSModulePath) { Remove-Item env:NCCL_P2P_LEVEL -ErrorAction SilentlyContinue }
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"
    Log "P2P DISABLED - NCCL_P2P_DISABLE=1, custom all-reduce OFF, expandable_segments ON"
}

# Export variables for use by calling scripts
# These are available as script-scoped variables when sourced
$script:NVLINK_ENABLED = $_NvlinkEnabled
$script:NCCL_P2P_LEVEL = $env:NCCL_P2P_LEVEL
$script:NCCL_P2P_DISABLE = $env:NCCL_P2P_DISABLE
$script:PYTORCH_CUDA_ALLOC_CONF = $env:PYTORCH_CUDA_ALLOC_CONF
$script:GPU_COUNT = $GPU_COUNT

# Print summary if run directly (not sourced)
if ($Apply -or ($PSScriptRoot -eq $PWD.Path -or $MyInvocation.MyCommand.Name -eq "nvlink-env.ps1")) {
    Write-Host ""
    Write-Host "=== nvlink-env summary ==="
    Write-Host "NVLINK_ENABLED=$($_NvlinkEnabled)"
    Write-Host "GPU_COUNT=$GPU_COUNT"
    Write-Host "NCCL_P2P_LEVEL=$($env:NCCL_P2P_LEVEL)"
    Write-Host "NCCL_P2P_DISABLE=$($env:NCCL_P2P_DISABLE)"
    Write-Host "PYTORCH_CUDA_ALLOC_CONF=$($env:PYTORCH_CUDA_ALLOC_CONF)"
}