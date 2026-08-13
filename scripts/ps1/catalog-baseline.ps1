# Requires -Version 5.1
#
# catalog-baseline.ps1 - induct a validated rebench run as a slug's SHIPPED
# baseline row in baselines.yml
#
# Usage:
#   .\catalog-baseline.ps1 <slug> -FromTag <rebench-tag>
#   .\catalog-baseline.ps1 <slug> -FromBundle <tgz|dir> -Source <url> -SubmittedBy <handle>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Slug,

    [string]$FromTag,
    [string]$TagDir,
    [string]$Source,
    [string]$EnginePin,
    [string]$Rig,
    [string]$Power,
    [string]$SubmittedBy,
    [switch]$TpsOnly,
    [switch]$DryRun,
    [string]$BaselinesFile = "scripts/lib/profiles/baselines.yml",
    [string]$FromBundle
)

$ErrorActionPreference = "Stop"
$ScriptName = "catalog-baseline"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

function Log { param($Msg) Write-Host "[$ScriptName] $Msg" }
function Die { param($Msg); Write-Error "[$ScriptName] ERROR: $Msg"; exit 1 }

if (-not $Slug -or $Slug.StartsWith("--")) {
    Write-Error "usage: catalog-baseline.ps1 <slug> -FromTag <rebench-tag> [--dry-run]"
    exit 2
}

$FromBundle = $null
$Tag = ""
$tagDirResolved = ""

if ($FromBundle) {
    if (-not $Source) { Die "bundle mode: --source is required" }
    if (-not $SubmittedBy) { Die "bundle mode: --submitted-by is required" }

    # Extract bundle
    $bundleDir = if (Test-Path $FromBundle -PathType Container) {
        $FromBundle
    } else {
        $tmpDir = [System.IO.Path]::GetTempPath() + "catalog-baseline-$$"
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
        $extractCmd = "tar xzf `"$FromBundle`" -C $tmpDir"
        Invoke-Expression $extractCmd 2>$null
        if ($LASTEXITCODE -ne 0) {
            Die "tar extraction failed (exit code $LASTEXITCODE)"
        }
        $tmpDir
    }

    # Find tag dirs (dirs with _internal.json)
    $tagDirs = Get-ChildItem -Path $bundleDir -Recurse -Filter "_internal.json" | Select-Object -ExpandProperty DirectoryName
    if ($tagDirs.Count -eq 0) { Die "no tag dir (_internal.json) inside the bundle" }

    if ($tagDirs.Count -eq 1) {
        $tagDirResolved = $tagDirs[0]
    } else {
        if (-not $FromTag) {
            Write-Error "bundle carries $($tagDirs.Count) tags - select one with -FromTag"
            exit 2
        }
        $tagDirResolved = ($tagDirs | Where-Object { (Split-Path $_ -Leaf) -eq $FromTag })[0]
        if (-not $tagDirResolved) { Die "tag $FromTag not in the bundle" }
    }
    $Tag = Split-Path $tagDirResolved -Leaf
} else {
    if (-not $FromTag -and -not $TagDir) { Die "-FromTag or -TagDir is required" }
    if (-not $SubmittedBy) { $SubmittedBy = $env:USERNAME }
    $tagDirResolved = if ($TagDir) { $TagDir } else { Join-Path $RepoRoot "ps1-results/rebench/$FromTag" }
    $Tag = Split-Path $tagDirResolved -Leaf
}

# Validate artifacts
$verifyLog = Join-Path $tagDirResolved "verify-full.log"
if (-not (Test-Path $verifyLog)) { Die "verify-full gate not covered" }
$verifyText = Get-Content $verifyLog -Raw -Encoding UTF8
if (-not ($verifyText -match "All checks passed")) { Die "verify-full gate not all-pass" }

$internalPath = Join-Path $tagDirResolved "_internal.json"
if (-not (Test-Path $internalPath)) { Die "_internal.json missing" }

$internal = Get-Content $internalPath -Raw -Encoding UTF8 | ConvertFrom-Json
$bench = $internal.bench
if (-not $bench) { Die "no bench section in _internal.json" }

$narr = $bench.narrative
$code = $bench.code
if (-not $narr -or -not $code) { Die "bench missing narrative or code" }

# Get n_runs from bench.log
$nRuns = 0
$benchLog = Join-Path $tagDirResolved "bench.log"
if (Test-Path $benchLog) {
    $blText = Get-Content $benchLog -Raw -Encoding UTF8
    $matches = [regex]::Matches($blText, '=== summary \[(?:narrative|code)\] \(n=(\d+)\) ===')
    if ($matches.Count -gt 0) {
        $ns = @()
        foreach ($m in $matches) { $ns += [int]$m.Groups[1].Value }
        $nRuns = ($ns | Measure-Object -Minimum).Minimum
    }
}
if ($nRuns -gt 0 -and $nRuns -lt 3) { Die "bench gate: $nRuns measured runs (< 3)" }
if ($nRuns -gt 0 -and $nRuns -lt 5) { Log "bench ran n=$nRuns (below canonical n=5)" }

# Quality check
$qOff = $null; $qOn = $null
$qualPath = Join-Path $tagDirResolved "quality-full.json"
if (Test-Path $qualPath) {
    $qData = Get-Content $qualPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $p = ($qData.packs | Measure-Object -Property passed -Sum).Sum
    $t = ($qData.packs | Measure-Object -Property total -Sum).Sum
    $qOff = "$p/$t"
}
$qOnPath = Join-Path $tagDirResolved "quality-full-thinking.json"
if (Test-Path $qOnPath) {
    $qData = Get-Content $qOnPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $p = ($qData.packs | Measure-Object -Property passed -Sum).Sum
    $t = ($qData.packs | Measure-Object -Property total -Sum).Sum
    $qOn = "$p/$t"
}

if (-not $qOff -and -not $qOn -and -not $TpsOnly) {
    Die "quality gate not covered (re-run with --tps-only to induct TPS-only row)"
}

# NIAH ladder
$ctxTokens = $null
$stressLog = Join-Path $tagDirResolved "verify-stress.log"
if (Test-Path $stressLog) {
    $stText = Get-Content $stressLog -Raw -Encoding UTF8
    $m = [regex]::Match($stText, 'all (\d+) rungs passed - fillable to (\d+) tok')
    if ($m.Success) {
        $ctxTokens = [int]$m.Groups[2].Value
    }
}

# Provenance
$enginePinResolved = $EnginePin
$rigResolved = $Rig
$powerResolved = $Power

if (-not $enginePinResolved) {
    # Try to resolve from compose registry
    Log "WARNING: engine pin not resolved - pass --engine-pin explicitly"
    $enginePinResolved = "unknown"
}

if (-not $rigResolved) {
    try {
        $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($gpuInfo) {
            $names = $gpuInfo -split "`n" | Where-Object { $_.Trim() }
            $short = ($names[0] -replace 'NVIDIA\s+(GeForce\s+)?(RTX\s+)?', '').Trim().ToLower().Replace(' ', '')
            $rigResolved = "$($names.Count)x${short}-pcie"
        }
    } catch { }
}
if (-not $rigResolved) { Die "could not derive --rig" }

if (-not $powerResolved) {
    try {
        $pwrInfo = nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>$null
        if ($pwrInfo) {
            $caps = $pwrInfo -split "`n" | Where-Object { $_.Trim() } | ForEach-Object { [math]::Round([double]($_.Trim())) }
            $powerResolved = ($caps -join ',')
        }
    } catch { }
}
if (-not $powerResolved) { Die "could not derive --power-cap-w" }

# Build row
$rowDate = (Get-Date).Date.ToString("yyyy-MM-dd")
$narrTps = [math]::Round($narr.decode_tps_mean, 2)
$codeTps = [math]::Round($code.decode_tps_mean, 2)
$ttft = if ($narr.ttft_ms_mean) { [math]::Round($narr.ttft_ms_mean) } else { $null }

$comment = "# inducted by catalog-baseline.ps1 from rebench tag $Tag ($rowDate);"
$evidence = "quality both arms"
if ($qOff -and $qOn) { $evidence = "quality both arms" }
elseif ($qOff -or $qOn) { $evidence = "quality one arm" }
else { $evidence = "TPS-ONLY (no quality)" }
if ($ctxTokens) { $evidence += " · NIAH ladder" }

$fields = @(
    $comment
    "# evidence: verify-full pass · bench n=$nRuns · $evidence"
    "narr_tps: $narrTps"
    "code_tps: $codeTps"
    if ($ttft) { "ttft_ms: $ttft" }
    if ($qOff) { "quality_8pk: `"$qOff`"" }
    if ($qOn) { "quality_8pk_think_on: `"$qOn`"" }
    if ($ctxTokens) { "ctx_validated: { tokens: $ctxTokens, niah: `"$([math]::Round($ctxTokens/1000))K`" }" }
    "date: $rowDate"
    "engine_pin: `"$enginePinResolved`""
    "rig: `"$rigResolved`""
    "power_cap_w: [$($powerResolved -replace ',', ', ')]"
    "source_tag: `"$Tag`""
    "submitted_by: `"$SubmittedBy`""
    "tier: submitted"
)

$rowText = "  ${Slug}:" + ($fields | ForEach-Object { "    $_" }) -join "`n"

if ($DryRun) {
    Write-Host ""
    Write-Host $rowText
    Write-Host ""
    Write-Host "[$ScriptName] DRY RUN - row for $Slug NOT written"
    exit 0
}

# Write to baselines file
$blPath = Join-Path $RepoRoot $BaselinesFile
if (Test-Path $blPath) {
    $old = Get-Content $blPath -Raw -Encoding UTF8
    # Simple append before gap list footer
    $new = $old + "`n" + $rowText + "`n"
    $new | Out-File -FilePath $blPath -Encoding UTF8
    Log "Updated $blPath with row for $Slug"
} else {
    Log "baselines.yml not found at $blPath - would write:"
    Write-Host $rowText
}
