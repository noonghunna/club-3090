# Requires -Version 5.1
#
# quality.ps1 — quick quality test (2 packs, ~5-10 min)
#
# This is a thin wrapper around quality-full.ps1 that passes --quick.
# For the full 8-pack test, use quality-full.ps1 instead.
#
# Usage:
#   .\quality.ps1
#   .\quality.ps1 -Model qwen-8010
#   .\quality.ps1 -EnableThinking

param(
    [string]$Model,
    [string]$Url,
    [switch]$EnableThinking,
    [switch]$NoThinking,
    [string]$ApiKey
)

$ErrorActionPreference = "Stop"
$ScriptPath = Join-Path $PSScriptRoot "quality-full.ps1"

if (-not (Test-Path $ScriptPath)) {
    Write-Error "quality-full.ps1 not found at $ScriptPath"
    exit 1
}

# Use splatting to properly pass switch parameters
$qtParams = @{
    Quick = $true
}
if ($Model) { $qtParams.Model = $Model }
if ($Url) { $qtParams.Url = $Url }
if ($EnableThinking) { $qtParams.EnableThinking = $true }
if ($NoThinking) { $qtParams.NoThinking = $true }
if ($ApiKey) { $qtParams.ApiKey = $ApiKey }

& $ScriptPath @qtParams
exit $LASTEXITCODE
