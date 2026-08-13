# Requires -Version 5.1
#
# run-benchmarks.ps1 — orchestrates the full benchmark pipeline:
#   1. verify-full (preflight)
#   2. bench (narrative + code TPS)
#   3. verify-stress (long-context needle + prefill-OOM)
#   4. soak-test (accumulating-context endurance)
#   5. report (generate report from results)
#
# Usage:
#   scripts/run-benchmarks.ps1                          # run all phases
#   scripts/run-benchmarks.ps1 --skip soak              # skip soak test
#   scripts/run-benchmarks.ps1 --skip quality           # skip quality (default)
#   scripts/run-benchmarks.ps1 --tag my-tag             # tag for reports
#   scripts/run-benchmarks.ps1 --help                   # show help
#
# Env vars passed through to individual scripts:
#   URL, MODEL, CONTAINER, RUNS, WARMUPS, etc.

param(
    [string]$Tag,
    [string]$Skip,
    [switch]$Help
)
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path | Select-String '^#( |$)' | ForEach-Object { $_.Line.Substring(2) }
    exit 0
}

$ROOT_DIR = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
$SCRIPTS_DIR = Split-Path $MyInvocation.MyCommand.Path -Parent

$SKIP_LIST = @()
if ($Skip) { $SKIP_LIST = $Skip.Split(",") | ForEach-Object { $_.Trim() } }

function Should-Skip {
    param([string]$Phase)
    return $SKIP_LIST -contains $Phase
}

function Run-Phase {
    param([string]$Phase, [string]$Script, [string]$Args)
    
    if (Should-Skip $Phase) {
        Write-Host "[skip] $Phase" -ForegroundColor Gray
        return
    }

    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  PHASE: $Phase" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""

    $scriptPath = Join-Path $SCRIPTS_DIR $Script
    if (-not (Test-Path $scriptPath)) {
        Write-Host "[error] Script not found: $scriptPath" -ForegroundColor Red
        return
    }

    try {
        $envVars = @()
        if ($env:URL) { $envVars += "URL='$env:URL'" }
        if ($env:MODEL) { $envVars += "MODEL='$env:MODEL'" }
        if ($env:CONTAINER) { $envVars += "CONTAINER='$env:CONTAINER'" }
        if ($env:RUNS) { $envVars += "RUNS='$env:RUNS'" }
        if ($env:WARMUPS) { $envVars += "WARMUPS='$env:WARMUPS'" }
        if ($Tag) { $envVars += "TAG='$Tag'" }
        
        $envDesc = if ($envVars.Count -gt 0) { " [env: $($envVars -join ', ')]" } else { "" }
        Write-Host "[running] .\ps1\$Script $Args$envDesc" -ForegroundColor White

        $fullArgs = $Args
        if ($Tag) { $fullArgs += " --tag $Tag" }
        
        $result = & powershell -NoProfile -ExecutionPolicy Bypass -File $scriptPath @($fullArgs.Split(' ') | Where-Object { $_ }) 2>&1
        Write-Host $result
        Write-Host ""
        Write-Host "[done] $Phase" -ForegroundColor Green
    } catch {
        Write-Host ("[failed] " + $Phase + ": " + $_) -ForegroundColor Red
    }
}

# Phase 0: Verify-full (preflight)
Run-Phase "verify-full" "verify-full.ps1" ""

# Phase 1: Bench
Run-Phase "bench" "bench.ps1" ""

# Phase 2: Verify-stress
Run-Phase "verify-stress" "verify-stress.ps1" ""

# Phase 3: Soak test
Run-Phase "soak-test" "soak-test.ps1" ""

# Phase 4: Report
Run-Phase "report" "report.ps1" ""

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ALL PHASES COMPLETE" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  - Check results/ for output files"
Write-Host "  - Run report.ps1 to generate a summary"
Write-Host "  - Submit with submit-bench.ps1"
