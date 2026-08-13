# Requires -Version 5.1
#
# quality-baseline.ps1 - diff a fresh quality 8-pack run against the
# curated baseline for a (slug, thinking-mode), or capture/refresh that baseline.
#
# Usage:
#   .\quality-baseline.ps1 -Slug vllm/qwen-35b-a3b-dual
#   .\quality-baseline.ps1 -Slug vllm/qwen-35b-a3b-dual -Mode enable-thinking
#   .\quality-baseline.ps1 -Slug vllm/qwen-35b-a3b-dual -Capture
#   .\quality-baseline.ps1 -Slug vllm/qwen-35b-a3b-dual -DryRun

param(
    [string]$Slug,
    [string]$Mode,
    [switch]$Capture,
    [int]$Repeat,
    [switch]$DryRun,
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$ScriptName = "quality-baseline"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot
. "$PSScriptRoot\get-model.ps1"

# Defaults
if (-not $Mode) { $Mode = "no-thinking" }
if (-not $Repeat) { $Repeat = 3 }

function Log { param($Msg) Write-Host "[$ScriptName] $Msg" }
function Die { param($Msg); Write-Error "[$ScriptName] ERROR: $Msg"; exit 1 }

if (-not $Slug) {
    Write-Error "--slug <registry-slug> is required (e.g. vllm/qwen-35b-a3b-dual)"
    exit 2
}

if (@("no-thinking", "enable-thinking") -notcontains $Mode) {
    Die "--mode must be no-thinking | enable-thinking (got: '$Mode')"
}

if ($Repeat -lt 1) {
    Die "--repeat must be a positive integer"
}

$slugSafe = $Slug -replace '/', '-'
$baselineDir = Join-Path $RepoRoot "ps1-results/baselines"
$baselineFile = Join-Path $baselineDir "${slugSafe}__${Mode}.json"

$qtScript = Join-Path $PSScriptRoot "quality-test.ps1"
if (-not (Test-Path $qtScript)) {
    Die "quality-test.ps1 not found at $qtScript"
}

$modeFlag = if ($Mode -eq "enable-thinking") { "--enable-thinking" } else { "--no-thinking" }

if ($Capture) {
    New-Item -ItemType Directory -Force -Path $baselineDir | Out-Null
    $cmd = @("--full", $modeFlag, "--repeat", $Repeat.ToString(), "--save-json", $baselineFile)
    if ($ExtraArgs) { $cmd += $ExtraArgs }
    Log "CAPTURE → $baselineFile  (n=$Repeat, mode=$Mode)"
    if ($DryRun) {
        Log "(dry-run) would run: .\$($qtScript -replace '\.ps1$','') $($cmd -join ' ')"
        exit 0
    }
    & $qtScript @cmd 2>&1
} else {
    if (-not (Test-Path $baselineFile)) {
        Write-Error "no baseline for slug='$Slug' mode='$Mode':"
        Write-Error "    $baselineFile"
        Write-Error "  Capture one first:  .\quality-baseline.ps1 -Slug '$Slug' -Mode $Mode -Capture"
        exit 1
    }

    $cmd = @("--full", $modeFlag, "--repeat", $Repeat.ToString(), "--previous-result", $baselineFile)
    if ($ExtraArgs) { $cmd += $ExtraArgs }
    Log "DIFF vs $baselineFile  (mode=$Mode)"
    if ($DryRun) {
        Log "(dry-run) would run: .\$($qtScript -replace '\.ps1$','') $($cmd -join ' ')"
        exit 0
    }
    & $qtScript @cmd 2>&1
}
