# Requires -Version 5.1
#
# rebench-full.ps1 — canonical rebench against the currently-running model
# (a fail-fast verify-full preflight + bench + verify-stress + soak + quality)
#
# Usage:
#   .\rebench-full.ps1 [-Tag <tag>] [--skip <phases>] [--with-8pack-thinking <off|on|both>] [--resume] [--dry-run]
#
# This is the full pipeline: verify-full → bench → bench-agentic → concurrency-probe → verify-stress → quality → soak

$ErrorActionPreference = "Stop"
$ScriptName = "rebench-full"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot

param(
    [string]$Tag,
    [string]$Skip,
    [string]$With8PackThinking,
    [switch]$Resume,
    [switch]$DryRun,
    [switch]$Full
)
. "$PSScriptRoot\get-model.ps1"

function Log { param($Msg) Write-Host "[$ScriptName] $Msg" }
function Die { param($Msg); Write-Error "[$ScriptName] ERROR: $Msg"; exit 1 }

# Default tag
if (-not $Tag) { $Tag = "rebench-$(Get-Date -Format 'yyyyMMdd-HHmmss')" }

$TagDir = Join-Path $RepoRoot "ps1-results/rebench/$Tag"
New-Item -ItemType Directory -Force -Path $TagDir | Out-Null

# Auto-detect endpoint
$Url = "http://localhost:8010"
$Model = ""
$Container = ""

# Try to detect from running containers
try {
    $models = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 15 -ErrorAction SilentlyContinue
    if ($models -and $models.data -and $models.data.Count -gt 0) {
        $Model = $models.data[0].id
    }
} catch { }

# Try to detect container
try {
    $containers = docker ps --format '{{.Names}}' 2>$null | Select-String -Pattern 'vllm-' -ErrorAction SilentlyContinue
    if ($containers) { $Container = $containers[0].ToString().Trim() }
} catch { }

Log "Tag: $Tag"
Log "URL: $Url"
Log "Model: $Model"
Log "Container: $Container"

# Parse skip list
$SkipList = @("verify-full", "bench", "verify-stress", "soak", "quality-full", "quality-thinking")
if ($Skip) {
    $skipParts = $Skip -split ','
    foreach ($s in $skipParts) {
        $s = $s.Trim()
        $SkipList = $SkipList | Where-Object { $_ -ne $s }
    }
}
if (-not $Full) {
    $SkipList = $SkipList | Where-Object { $_ -ne "soak" }
}

Log "Phases to run: $($SkipList -join ', ')"

# Run each phase
$ScriptPath = $PSScriptRoot
$Scripts = @{
    "verify-full" = "verify-full.ps1"
    "bench" = "bench.ps1"
    "bench-agentic" = "bench-agentic.ps1"
    "concurrency-probe" = "concurrency-probe.ps1"
    "verify-stress" = "verify-stress.ps1"
    "quality-full" = "quality-test.ps1"
    "quality-thinking" = "quality-test.ps1"
    "soak" = "soak-test.ps1"
}

foreach ($phase in $SkipList) {
    if ($Scripts.ContainsKey($phase)) {
        $scriptFile = Join-Path $ScriptPath $Scripts[$phase]
        if (Test-Path $scriptFile) {
            Log "Running phase: $phase"
            $args = @("-Tag", $Tag)
            if ($Resume) { $args += "--resume" }
            if ($phase -eq "quality-full" -and $With8PackThinking) {
                $args += "--no-thinking"
            }
            if ($phase -eq "quality-thinking" -and $With8PackThinking) {
                $args += "--enable-thinking"
            }
            if ($Container) { $args += "--container", $Container }
            if ($Model) { $args += "--model", $Model }
            if ($Url) { $args += "--url", $Url }
            
            if ($DryRun) {
                Log "  [dry-run] would run: .\$($Scripts[$phase]) $($args -join ' ')"
            } else {
                & $scriptFile @args *>&1
                if ($LASTEXITCODE -ne 0) {
                    Log "Phase $phase exited with code $LASTEXITCODE"
                }
            }
        } else {
            Log ("Phase " + $phase + ": script not found at " + $scriptFile + ", skipping")
        }
    } else {
        Log ("Phase " + $phase + ": not in this script's registry")
    }
}

# Generate _internal.json summary
$InternalJson = @{
    bench = @{
        narrative = @{ decode_tps_mean = 0; ttft_ms_mean = 0 }
        code = @{ decode_tps_mean = 0; ttft_ms_mean = 0 }
    }
    tag = $Tag
    date = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
}

if (Test-Path (Join-Path $TagDir "bench.log")) {
    $benchLog = Get-Content (Join-Path $TagDir "bench.log") -Raw
    # Try to extract TPS from bench log
    $narrMatch = $benchLog | Select-String -Pattern "=== summary \[narrative\] \(n=\d+\) ===" -Context 0, 10
    if ($narrMatch) {
        $tpsMatch = $narrMatch.Context.PostContext[0] | Select-String -Pattern "prefill tok/s.*mean=([0-9.]+)"
        if ($tpsMatch) {
            $InternalJson.bench.narrative.decode_tps_mean = [double]$tpsMatch.Matches[0].Groups[1].Value
        }
    }
}

$InternalJson | ConvertTo-Json -Depth 10 | Out-File (Join-Path $TagDir "_internal.json") -Encoding UTF8
Log "Wrote _internal.json to $TagDir"
