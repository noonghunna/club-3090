# Requires -Version 5.1
#
# rerun-failed-packs.ps1 — re-test ONLY the scenarios that failed in a prior
# quality run, and report which failures reproduced vs flipped (flakes).
#
# Usage:
#   .\rerun-failed-packs.ps1 <result.json> [--repeat N] [extra quality-test.ps1 args]

$ErrorActionPreference = "Stop"
$ScriptName = "rerun-failed-packs"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ResultJson,

    [string]$Repeat,
    [string[]]$ExtraArgs
)
. "$PSScriptRoot\get-model.ps1"

if (-not (Test-Path $ResultJson)) {
    Write-Error "result JSON not found: $ResultJson"
    Write-Error "  Fix: pass a saved RunResult (e.g. ps1-results/quality/quality-<ts>.json)"
    exit 2
}

# Parse failed scenarios
$JsonData = Get-Content $ResultJson -Raw -Encoding UTF8 | ConvertFrom-Json
$Selections = @()
$Failures = @()

foreach ($pack in $JsonData.packs) {
    $pid = if ($pack.pack_id) { $pack.pack_id } else { "?" }
    foreach ($scenario in $pack.scenarios) {
        if ($scenario.passed -ne $true) {
            $sid = if ($scenario.id) { $scenario.id } else { "?" }
            $Selections += "$pid/$sid"
            $failureMode = if ($scenario.failure_mode) { $scenario.failure_mode } else { "fail" }
            $Failures += ($pid + "/" + $sid + ":" + $failureMode)
        }
    }
}

if ($Selections.Count -eq 0) {
    Write-Host "✓ no failed scenarios in $ResultJson — nothing to re-run."
    exit 0
}

# Determine thinking mode from original run
$modeFlag = if ($JsonData.thinking_enabled) { "--enable-thinking" } else { "--no-thinking" }

$N_Sel = $Selections.Count
Write-Host "[$ScriptName] original run: $ResultJson  (mode: $($modeFlag -replace '--', ''))"
Write-Host "[$ScriptName] failed scenarios ($N_Sel): $($Selections -join ' ')"

# Create scenarios file
$SelFile = [System.IO.Path]::GetTempFileName()
$Selections | Out-File -FilePath $SelFile -Encoding UTF8
trap { Remove-Item $SelFile -ErrorAction SilentlyContinue }

$RerunJson = "$ResultJson.rerun.json"

# Run quality-test.ps1 with the failed scenarios
$QtScript = Join-Path $PSScriptRoot "quality-test.ps1"
if (-not (Test-Path $QtScript)) {
    Write-Error "quality-test.ps1 not found at $QtScript"
    exit 1
}

$CmdArgs = @("--scenarios-file", $SelFile, $modeFlag, "--previous-result", $ResultJson, "--incremental", "--save-json", $RerunJson)
if ($ExtraArgs) { $CmdArgs += $ExtraArgs }

Write-Host "[$ScriptName] running: .\$($QtScript -replace '\.ps1$','') $($CmdArgs -join ' ')"
& $QtScript @CmdArgs *>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "quality-test exited with code $LASTEXITCODE"
}

# Parse rerun results and produce verdict
if (Test-Path $RerunJson) {
    $RerunData = Get-Content $RerunJson -Raw -Encoding UTF8 | ConvertFrom-Json

    $Repro = @()
    $Fixed = @()

    foreach ($pack in $RerunData.packs) {
        foreach ($scenario in $pack.scenarios) {
            $key = "$($pack.pack_id)/$($scenario.id)"
            if ($scenario.passed -eq $true) {
                $Fixed += $key
            } else {
                $failureMode = if ($scenario.failure_mode) { $scenario.failure_mode } else { "fail" }
                $Repro += "$key ($failureMode)"
            }
        }
    }

    Write-Host ""
    Write-Host "════ rerun verdict ════"
    Write-Host "REPRODUCED ($($Repro.Count)) — likely real:"
    foreach ($t in $Repro) { Write-Host "  ✗ $t" }
    Write-Host "FIXED on re-run ($($Fixed.Count)) — flake / environment:"
    foreach ($t in $Fixed) { Write-Host "  ↺ $t" }
    Write-Host ""
    Write-Host "Note: single re-run separates 'stable' from 'flaky', not 'model' from"
    Write-Host "'harness'. For flakiness RATES, add --repeat 3 (cheap at scenario"
    Write-Host "granularity) and read benchlocal's >=50% aggregation in the delta output."
    Write-Host "The rerun JSON is a PARTIAL selection result — not a pack total."
} else {
    Write-Error "✗ no rerun JSON at $RerunJson — the selection run did not complete"
    exit 1
}

Remove-Item $SelFile -ErrorAction SilentlyContinue
