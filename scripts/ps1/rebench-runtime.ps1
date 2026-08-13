# Requires -Version 5.1
#
# rebench-runtime.ps1 — runtime tier of rebench: verify-full + bench + verify-stress + soak
# (skipping quality legs)
#
# Usage:
#   .\rebench-runtime.ps1 [-Tag <tag>] [--skip <phases>]
#
# This is a thin preset over rebench-full.ps1: it injects
# --skip quality-full,quality-thinking and merges any --skip you pass.

$ErrorActionPreference = "Stop"
$ScriptName = "rebench-runtime"
$ScriptPath = $PSScriptRoot

param(
    [string]$Tag,
    [string]$Skip
)
. "$PSScriptRoot\get-model.ps1"

$BaseSkip = "quality-full,quality-thinking"
$UserSkip = if ($Skip) { $Skip } else { "" }
$FullSkip = if ($UserSkip) { "$BaseSkip,$UserSkip" } else { $BaseSkip }

Write-Host "[$ScriptName] runtime tier (verify-full + bench + verify-stress + soak); --skip=$FullSkip"

# Delegate to rebench-full with skip
$RebenchFull = Join-Path $ScriptPath "rebench-full.ps1"
if (Test-Path $RebenchFull) {
    $args = @()
    if ($Tag) { $args += "-Tag", $Tag }
    $args += "--skip", $FullSkip
    & $RebenchFull @args
} else {
    Write-Error "rebench-full.ps1 not found at $RebenchFull"
    exit 1
}
