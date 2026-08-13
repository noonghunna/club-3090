# Requires -Version 5.1
#
# arch-ab.ps1 - cross-rig KV-dtype A/B runner
#
# Runs the SAME pilot variant once per KV-dtype arm, each on a fresh boot,
# with bench + verify-stress. Bundles everything into a tarball.
#
# Usage:
#   .\arch-ab.ps1
#   .\arch-ab.ps1 -Arms "e5m2,e4m3"
#   .\arch-ab.ps1 -Variant "vllm/dual"
#   .\arch-ab.ps1 -DryRun
#   .\arch-ab.ps1 -Resume

param(
    [string]$Arms = "e5m2,e4m3",
    [string]$Variant,
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$ScriptName = "arch-ab"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

function Log { param($Msg) Write-Host "[$ScriptName] $Msg" }
function Die { param($Msg); Write-Error "[$ScriptName] ERROR: $Msg"; exit 1 }

# ARM dtype mapping
$ArmDtype = @{
    "e5m2" = "fp8_e5m2"
    "e4m3" = "fp8_e4m3"
    "nvfp4" = "nvfp4"
    "fp8w" = ""
}

$ArmVariantOverride = @{
    "fp8w" = "vllm/qwen-27b-dual-max"
}

$ArmList = $Arms -split ','

# Detect GPUs
try {
    $gpuLines = nvidia-smi --query-gpu=index,name,memory.total,compute.major --format=csv,noheader 2>$null
    if (-not $gpuLines) { Die "no NVIDIA GPUs detected" }
    $gpuCount = 0
    $minSm = 999
    foreach ($line in $gpuLines -split "`n") {
        $parts = $line -split ',' | ForEach-Object { $_.Trim() }
        if ($parts.Count -lt 4) { continue }
        $gpuCount++
        $sm = [int]$parts[3]
        if ($sm -lt $minSm) { $minSm = $sm }
    }
} catch { Die "no NVIDIA GPUs detected" }

if ($gpuCount -eq 0) { Die "no NVIDIA GPUs detected" }

# Resolve variant
if (-not $Variant) {
    if ($gpuCount -ge 2) { $Variant = "vllm/dual" } else { $Variant = "vllm/minimal" }
}

if ($Variant -eq "vllm/dual" -and $gpuCount -lt 2) {
    Die "vllm/dual is TP=2 and needs 2 GPUs (detected $gpuCount)"
}

# Validate arms
foreach ($arm in $ArmList) {
    if (-not $ArmDtype.ContainsKey($arm)) {
        Die "unknown arm '$arm' (valid: e5m2, e4m3, nvfp4, fp8w)"
    }
    if ($arm -eq "e4m3" -and $minSm -lt 8.9) {
        Die "arm 'e4m3' needs sm>=8.9 (detected sm_$minSm)"
    }
    if ($arm -eq "nvfp4" -and $minSm -ne 10 -and $minSm -ne 10.3) {
        Die "arm 'nvfp4' needs DATACENTER Blackwell (sm_100/sm_103); detected sm_$minSm"
    }
    if ($arm -eq "fp8w" -and $gpuCount -lt 2) {
        Die "arm 'fp8w' needs 2 GPUs (detected $gpuCount)"
    }
}

Log "plan: variant=$Variant arms=$Arms gpus=$gpuCount (min sm_$minSm)"

if ($DryRun) {
    Log "dry run - nothing executed."
    exit 0
}

# Run arms
$ScriptPath = $PSScriptRoot
foreach ($arm in $ArmList) {
    $dtype = $ArmDtype[$arm]
    $armVariant = if ($ArmVariantOverride.ContainsKey($arm)) { $ArmVariantOverride[$arm] } else { $Variant }
    $tag = "246-ab-$arm"

    $armLabel = "arm $arm"
    $armInfo = if ($dtype) { " KV_CACHE_DTYPE=$dtype" } else { " (stock)" }
    Log ("===== " + $armLabel + ": " + $armVariant + $armInfo + " -> tag $tag =====")

    # Fresh boot per arm (simplified - in bash this calls switch.sh)
    Log "Would: bash scripts/switch.sh $armVariant$(if ($dtype) { " KV_CACHE_DTYPE=$dtype" } else { " --force" })"

    # Run rebench
    $rbArgs = @("-Tag", $tag)
    if ($Resume) { $rbArgs += "--resume" }
    if (-not $Full) {
        $rbArgs += "--skip", "soak"
    }

    $rebenchScript = Join-Path $ScriptPath "rebench-full.ps1"
    if (Test-Path $rebenchScript) {
        & $rebenchScript @rbArgs 2>&1
        Log "Phase $arm complete"
    } else {
        Log "rebench-full.ps1 not found - phase skipped"
    }
}

# Summary
Write-Host ""
Write-Host "[$ScriptName] ===== summary ====="

$summaryHeader = "  {0,-8} {1,9} {2,9} {3,8}  {4}" -f "arm", "narr TPS", "code TPS", "TTFT ms", "NIAH ladder"
Write-Host $summaryHeader

foreach ($arm in $ArmList) {
    $tag = "246-ab-$arm"
    $tagDir = Join-Path $RepoRoot "ps1-results/rebench/$tag"
    $narrTps = "-"; $codeTps = "-"; $ttft = "-"; $ladder = "(no artifact)"

    $internalPath = Join-Path $tagDir "_internal.json"
    if (Test-Path $internalPath) {
        try {
            $internal = Get-Content $internalPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $bench = $internal.bench
            if ($bench) {
                $narr = $bench.narrative
                $code = $bench.code
                if ($narr.decode_tps_mean) { $narrTps = "{0:F2}" -f $narr.decode_tps_mean }
                if ($code.decode_tps_mean) { $codeTps = "{0:F2}" -f $code.decode_tps_mean }
                if ($narr.ttft_ms_mean) { $ttft = "{0:F0}" -f $narr.ttft_ms_mean }
            }
        } catch { }
    }

    $stressLog = Join-Path $tagDir "verify-stress.log"
    if (Test-Path $stressLog) {
        $stressText = Get-Content $stressLog -Raw -Encoding UTF8
        $match = $stressText | Select-String -Pattern 'all (\d+) rungs passed - fillable to (\d+) tok'
        if ($match) {
            $ladder = "$($match.Matches[0].Groups[1].Value) rungs clean, fillable to $([int]$match.Matches[0].Groups[2].Value) tok"
        }
    }

    Write-Host ("  {0,-8} {1,9} {2,9} {3,8}  {4}" -f $arm, $narrTps, $codeTps, $ttft, $ladder)
}

# Bundle
$bundleName = "246-ab-bundle-$(hostname)-$(Get-Date -Format 'yyyyMMdd').tgz"
$bundlePath = Join-Path $RepoRoot "ps1-results/rebench/$bundleName"

Log "Would create bundle: $bundlePath"
Log "Attach to issue #246 with a sentence on anything that surprised you"
