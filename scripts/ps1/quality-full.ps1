# Requires -Version 5.1
#
# quality-test.ps1 — behavioral quality bench against a running compose
#
# Usage:
#   .\quality-test.ps1
#   .\quality-test.ps1 -Quick
#   .\quality-test.ps1 -Full
#   .\quality-test.ps1 -Pack "toolcall-15"
#   .\quality-test.ps1 -EnableThinking

param(
    [switch]$Quick,
    [switch]$Medium,
    [switch]$Full,
    [switch]$Reasoning,
    [string]$Pack,
    [string[]]$Scenario,
    [string]$ScenariosFile,
    [switch]$Incremental,
    [string]$Resume,
    [switch]$AllowPartial,
    [string]$Model,
    [int]$TimeoutPerCase,
    [string]$SandboxLogDir,
    [switch]$Progress,
    [switch]$NoProgress,
    [switch]$SamplingFromServer,
    [switch]$EnableThinking,
    [switch]$NoThinking,
    [int]$ThinkingMaxTokens,
    [int]$MaxTokens,
    [int]$Repeat,
    [string]$PreviousResult,
    [string]$SaveJson,
    [string]$ApiKey,
    [switch]$NoSandboxed,
    [switch]$SandboxedOnly,
    [switch]$ListPacks,
    [string]$Url
)

$ErrorActionPreference = "Stop"
$ScriptName = "quality-test"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

function Log { param($Msg) Write-Host "[$ScriptName] $Msg" }
function Die { param($Msg); Write-Error "[$ScriptName] ERROR: $Msg"; exit 1 }

# Defaults: quality-full defaults to --full; quality (wrapper) passes -Quick
$mode = "--full"
if ($Quick) { $mode = "--quick" }
elseif ($Full) { $mode = "--full" }
elseif ($Reasoning) { $mode = "--reasoning" }
elseif ($Medium) { $mode = "--medium" }

# Auto-detect endpoint and model (consistent with other scripts)
$url = if ($Url) { $Url } else { "http://localhost:8010" }
$model = $Model
$modelExplicit = $false

# Auto-detect model from running engine
if (-not $model) {
    try {
        $models = Invoke-RestMethod -Uri "$url/v1/models" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($models -and $models.data -and $models.data.Count -gt 0) {
            $model = $models.data[0].id
            $modelExplicit = $true
        }
    } catch { $model = $DETECTED_MODEL }
}

# If still no model, probe endpoint reachability first
if (-not $model) {
    try {
        $probe = Invoke-WebRequest -Uri "$url/v1/models" -TimeoutSec 5 -ErrorAction SilentlyContinue -UseBasicParsing
        if ($probe) {
            $models = $probe.Content | ConvertFrom-Json
            if ($models -and $models.data -and $models.data.Count -gt 0) {
                $model = $models.data[0].id
                $modelExplicit = $true
            }
        }
    } catch {
        Write-Error "endpoint $url not responding - set MODEL=<name> or provide -Url"
        exit 1
    }
}

# Check benchlocal-cli
$benchlocalExists = Get-Command benchlocal-cli -ErrorAction SilentlyContinue
if (-not $benchlocalExists) {
    $pipExists = Get-Command pip -ErrorAction SilentlyContinue
    $pip3Exists = Get-Command pip3 -ErrorAction SilentlyContinue
    $pythonExists = Get-Command python -ErrorAction SilentlyContinue
    $python3Exists = Get-Command python3 -ErrorAction SilentlyContinue

    $pipMsg = ""
    if (-not $pipExists -and -not $pip3Exists) {
        $pipMsg = "`n  pip/pip3 not found on `$PATH - install Python first"
    }

    Write-Error @"
benchlocal-cli not found on `$PATH

Install it (one-time):
  pip install git+https://github.com/noonghunna/benchlocal-cli.git

Or from a local checkout:
  pip install -e /path/to/benchlocal-cli

Verify installation:
  benchlocal-cli list
"@
    if ($pipMsg) { Write-Error $pipMsg }
    exit 127
}

# Verify benchlocal-cli is functional
try {
    benchlocal-cli list --help > $null 2>&1
} catch {
    Write-Error "benchlocal-cli found but not functional: $($_.Exception.Message)"
    exit 127
}

if ($ListPacks) {
    benchlocal-cli list
    exit 0
}

# Reachability probe
try {
    $probe = Invoke-WebRequest -Uri "$url/v1/models" -TimeoutSec 8 -ErrorAction SilentlyContinue -UseBasicParsing
    if (-not $probe) {
        Write-Error "endpoint $url not responding"
        exit 1
    }
} catch {
    if ($ApiKey) {
        try {
            $headers = @{ "Authorization" = "Bearer $ApiKey" }
            $probe = Invoke-WebRequest -Uri "$url/v1/models" -Headers $headers -TimeoutSec 8 -ErrorAction SilentlyContinue -UseBasicParsing
        } catch { }
    }
    if (-not $probe) {
        Write-Error "endpoint $url not responding (no HTTP response on /v1/models)"
        Write-Error "  bring up a compose first: bash scripts/launch.sh"
        exit 1
    }
}

# Build benchlocal-cli command — always include --endpoint and --model
# Also always save JSON results to ps1-results/quality/
$qualityDir = Join-Path $RepoRoot "ps1-results/quality"
New-Item -ItemType Directory -Force -Path $qualityDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$defaultSaveJson = Join-Path $qualityDir "quality-${timestamp}.json"

$cmdArgs = @($mode, "--endpoint", $url, "--model", $model, "--save-json", $defaultSaveJson)

if ($Pack) { $cmdArgs += "--pack"; $cmdArgs += $Pack }
if ($Scenario) { foreach ($s in $Scenario) { $cmdArgs += "--scenario"; $cmdArgs += $s } }
if ($ScenariosFile) { $cmdArgs += "--scenarios-file"; $cmdArgs += $ScenariosFile }
if ($Incremental) { $cmdArgs += "--incremental" }
if ($Resume) { $cmdArgs += "--resume"; $cmdArgs += $Resume }
if ($AllowPartial) { $cmdArgs += "--allow-partial" }
if ($TimeoutPerCase) { $cmdArgs += "--timeout-per-case"; $cmdArgs += $TimeoutPerCase.ToString() }
if ($SandboxLogDir) { $cmdArgs += "--sandbox-log-dir"; $cmdArgs += $SandboxLogDir }
if ($Progress) { $cmdArgs += "--progress" }
if ($NoProgress) { $cmdArgs += "--no-progress" }
if ($SamplingFromServer) { $cmdArgs += "--sampling-from-server" }
if ($EnableThinking) { $cmdArgs += "--enable-thinking" }
if ($NoThinking) { $cmdArgs += "--no-thinking" }
if ($ThinkingMaxTokens) { $cmdArgs += "--thinking-max-tokens"; $cmdArgs += $ThinkingMaxTokens.ToString() }
if ($MaxTokens) { $cmdArgs += "--max-tokens"; $cmdArgs += $MaxTokens.ToString() }
if ($Repeat) { $cmdArgs += "--repeat"; $cmdArgs += $Repeat.ToString() }
if ($PreviousResult) { $cmdArgs += "--previous-result"; $cmdArgs += $PreviousResult }
if ($SaveJson) { $cmdArgs += "--save-json"; $cmdArgs += $SaveJson }
if ($ApiKey) { $cmdArgs += "--api-key"; $cmdArgs += $ApiKey }
if ($NoSandboxed) { $cmdArgs += "--no-sandboxed" }
if ($SandboxedOnly) { $cmdArgs += "--sandboxed-only" }

Log "Running: benchlocal-cli run $($cmdArgs -join ' ')"

# Execute benchlocal-cli (requires 'run' subcommand)
# Do NOT use 2>&1 — benchlocal-cli outputs to stdout and PowerShell misinterprets it as errors
$benchOutput = benchlocal-cli run @cmdArgs
$LASTEXITCODE = if ($benchOutput) { 0 } else { 1 }
$benchOutput
exit $LASTEXITCODE
