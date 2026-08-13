# Requires -Version 5.1
#
# power-cap-sweep.ps1 - Power-cap A/B sweep for cross-rig efficiency-knee data
#
# Usage:
#   .\power-cap-sweep.ps1
#   .\power-cap-sweep.ps1 -StepSize 20
#   .\power-cap-sweep.ps1 -Caps "260,280,300"
#   .\power-cap-sweep.ps1 -LoadMode "decode-single"
#   .\power-cap-sweep.ps1 -Cooling "air"
#
# ⚠ Requires elevated privileges for nvidia-smi -pl

$ErrorActionPreference = "Stop"
$ScriptName = "power-cap-sweep"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot

param(
    [int]$GpuIndex,
    [switch]$GpuIndexSet,
    [string]$Gpus,
    [string]$Caps,
    [int]$Reset,
    [string]$Cooling,
    [int]$StepSize,
    [string]$LoadMode,
    [string]$Concurrency,
    [int]$BenchRuns,
    [int]$MaxConcurrencyProbe,
    [double]$LoadTarget,
    [int]$ConcurrencyStretch,
    [int]$TargetCapSeconds,
    [int]$DecodeSingleWarmups,
    [int]$DecodeSingleMaxTokensNarr,
    [int]$DecodeSingleMaxTokensCode,
    [string]$TargetPrefillSeconds,
    [int]$PrefillCalibrationRepeats,
    [switch]$IncludeCommit,
    [switch]$NoReset
)
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

# Defaults
if (-not $GpuIndex) { $GpuIndex = 0 }
if (-not $Reset) { $Reset = 1 }
if (-not $Cooling) { $Cooling = "unspecified" }
if (-not $StepSize) { $StepSize = 10 }
if (-not $LoadMode) { $LoadMode = "decode-single" }
if (-not $Concurrency) { $Concurrency = "4" }
if (-not $BenchRuns) { $BenchRuns = 1 }
if (-not $MaxConcurrencyProbe) { $MaxConcurrencyProbe = 16 }
if (-not $LoadTarget) { $LoadTarget = 0.92 }
if (-not $ConcurrencyStretch) { $ConcurrencyStretch = 0 }
if (-not $TargetCapSeconds) { $TargetCapSeconds = 10 }
if (-not $DecodeSingleWarmups) { $DecodeSingleWarmups = 3 }
if (-not $DecodeSingleMaxTokensNarr) { $DecodeSingleMaxTokensNarr = 500 }
if (-not $DecodeSingleMaxTokensCode) { $DecodeSingleMaxTokensCode = 400 }
if (-not $TargetPrefillSeconds) { $TargetPrefillSeconds = "10" }
if (-not $PrefillCalibrationRepeats) { $PrefillCalibrationRepeats = 1000 }

function Log { param($Msg) Write-Host "[$ScriptName] $Msg" }
function Die { param($Msg); Write-Error "[$ScriptName] ERROR: $Msg"; exit 1 }

# Validate load mode
if (@("decode-single", "decode-concurrent", "prefill-heavy") -notcontains $LoadMode) {
    Die "--load-mode must be one of: decode-single, decode-concurrent, prefill-heavy"
}

# Validate cooling
if (@("air", "water", "aio", "unspecified") -notcontains $Cooling) {
    Die "--cooling must be one of: air, water, aio (or omit for 'unspecified')"
}

if ($Cooling -eq "unspecified") {
    Log "warn: --cooling not specified. Cooling class is essential for interpreting the efficiency knee."
}

if ($NoReset) { $Reset = 0 }

# Check elevated
if ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent().IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) -eq $false) {
    Write-Warning "[$ScriptName] not running as admin - nvidia-smi -pl may fail"
}

# Auto-detect endpoint
$Url = "http://localhost:8010"
$Model = ""
$Container = ""

try {
    $models = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($models -and $models.data -and $models.data.Count -gt 0) {
        $Model = $models.data[0].id
    }
} catch { }

# Resolve GPU indices
$gpuIndices = @()
if ($Gpus) {
    $gpuIndices = ($Gpus -split ',' | ForEach-Object { $_.Trim() })
} elseif ($GpuIndexSet) {
    $gpuIndices = @($GpuIndex.ToString())
} elseif ($Container) {
    try {
        $nvd = docker inspect $Container 2>$null | Select-String -Pattern 'NVIDIA_VISIBLE_DEVICES=([^"]+)'
        if ($nvd) { $gpuIndices = @($nvd.Matches[0].Groups[1].Value -split ',') }
    } catch { }
}

if ($gpuIndices.Count -eq 0) {
    try {
        $allGpus = nvidia-smi --query-gpu=index --format=csv,noheader 2>$null
        $gpuIndices = ($allGpus -split "`n" | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() })
    } catch { }
}

if ($gpuIndices.Count -eq 0) { Die "could not determine which GPU(s) to sweep" }

$primaryGpu = $gpuIndices[0]
$gpuCount = $gpuIndices.Count
Log "GPUs: $($gpuIndices -join ',')"

# Capture envelopes
$minLimit = $null; $maxLimit = $null; $stockTdp = $null
$initArr = @{}; $minArr = @{}; $maxArr = @{}; $defaultArr = @{}; $nameArr = @{}; $vramArr = @{}

foreach ($idx in $gpuIndices) {
    try {
        $df = nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits -i $idx | Select-String -Pattern '.' | Select-Object -First 1
        $mn = nvidia-smi --query-gpu=power.min_limit --format=csv,noheader,nounits -i $idx | Select-String -Pattern '.' | Select-Object -First 1
        $mx = nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits -i $idx | Select-String -Pattern '.' | Select-Object -First 1
        $lim = nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i $idx | Select-String -Pattern '.' | Select-Object -First 1
        $name = nvidia-smi --query-gpu=name --format=csv,noheader -i $idx | Select-String -Pattern '.' | Select-Object -First 1
        $vram = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $idx | Select-String -Pattern '.' | Select-Object -First 1

        $dfVal = [double]($df.ToString().Trim())
        $mnVal = [double]($mn.ToString().Trim())
        $mxVal = [double]($mx.ToString().Trim())
        $limVal = [double]($lim.ToString().Trim())

        $initArr[$idx] = $limVal
        $minArr[$idx] = $mnVal
        $maxArr[$idx] = $mxVal
        $defaultArr[$idx] = $dfVal
        $nameArr[$idx] = $name.ToString().Trim()
        $vramArr[$idx] = [int]($vram.ToString().Trim())

        if (-not $minLimit -or $mnVal -gt $minLimit) { $minLimit = $mnVal }
        if (-not $maxLimit -or $mxVal -lt $maxLimit) { $maxLimit = $mxVal }
        if (-not $stockTdp -or $dfVal -lt $stockTdp) { $stockTdp = $dfVal }
    } catch {
        Log "WARN: could not capture envelope for GPU $idx"
    }
}

if (-not $minLimit) { $minLimit = 100 }
if (-not $maxLimit) { $maxLimit = 400 }

Log "Power envelope: ${minLimit}W - ${maxLimit}W (stock TDP: ${stockTdp}W)"

# Generate caps
$capList = @()
if ($Caps) {
    $capList = ($Caps -split ',' | ForEach-Object { [int]($_.Trim()) })
} else {
    for ($w = [math]::Ceiling($minLimit); $w -le $maxLimit; $w += $StepSize) {
        $capList += $w
    }
}

Log "Caps to sweep: $($capList.Count) values from $($capList[0])W to $($capList[-1])W"

# Restore function
function Restore-Gpus {
    if ($Reset -ne 1) { return }
    foreach ($idx in $gpuIndices) {
        if ($initArr.ContainsKey($idx)) {
            try {
                nvidia-smi -pl $initArr[$idx] -i $idx 2>$null
            } catch { }
        }
    }
}

# Trap for cleanup
$originalHandler = $ErrorActionPreference
trap {
    Restore-Gpus
    $ErrorActionPreference = $originalHandler
    throw
}

# Run sweep
$summaryRows = @()

foreach ($cap in $capList) {
    Log "Setting power cap: ${cap}W"
    try {
        foreach ($idx in $gpuIndices) {
            nvidia-smi -pl $cap -i $idx 2>$null
        }
    } catch {
        Log "WARN: failed to set cap $cap on GPU $idx"
    }

    # Run benchmark for this cap
    $logFile = "$env:TEMP\power-cap-${cap}w.log"
    $narrTps = 0; $codeTps = 0; $combinedW = 0; $maxTemp = 0; $smClk = 0; $memClk = 0

    if ($LoadMode -eq "decode-single") {
        # Run bench.sh for this cap
        $benchScript = Join-Path $RepoRoot "scripts/bench.sh"
        if (Test-Path $benchScript) {
            Log "Running bench.sh at ${cap}W..."
            try {
                # TODO: This calls bash scripts/bench.sh — breaks "Windows-native, no WSL".
                # Full rewrite to PowerShell is needed for proper Windows support.
                $benchOutput = bash "$benchScript" --save-json "$logFile" 2>&1
                # Parse TPS from output (simplified)
                $match = $benchOutput | Select-String -Pattern 'prefill tok/s.*mean=([0-9.]+)'
                if ($match) { $narrTps = [double]$match.Matches[0].Groups[1].Value }
            } catch {
                Log "WARN: bench.sh failed for cap $cap"
            }
        }
    } elseif ($LoadMode -eq "prefill-heavy") {
        Log "Running prefill-heavy benchmark at ${cap}W..."
        # Placeholder - would run a prefill-heavy benchmark
    } elseif ($LoadMode -eq "decode-concurrent") {
        Log "Running decode-concurrent benchmark at ${cap}W..."
        # Placeholder - would run concurrent benchmark
    }

    # Sample GPU stats
    try {
        $smi = nvidia-smi --query-gpu=index,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,power.state --format=csv,noheader -i $primaryGpu 2>$null
        if ($smi) {
            $parts = $smi -split ',' | ForEach-Object { $_.Trim() }
            if ($parts.Count -ge 7) {
                $combinedW = [double]$parts[2]
                $maxTemp = [int]$parts[3]
                $smClk = [int]$parts[4]
                $memClk = [int]$parts[5]
            }
        }
    } catch { }

    $combinedWAll = 0
    $tempMax = 0
    foreach ($idx in $gpuIndices) {
        try {
            $smi = nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv,noheader,nounits -i $idx 2>$null
            if ($smi) {
                $pwr = [double]($smi -split "`n" | Select-Object -First 1 | Select-String -Pattern '.' | Select-Object -First 1)
                $temp = [int]($smi -split "`n" | Select-Object -First 1 | Select-String -Pattern '.' | Select-Object -First 1)
                $combinedWAll += $pwr
                if ($temp -gt $tempMax) { $tempMax = $temp }
            }
        } catch { }
    }

    $summaryRows += [PSCustomObject]@{
        Cap = $cap
        NarrTps = [math]::Round($narrTps, 2)
        CodeTps = [math]::Round($codeTps, 2)
        CombinedW = [math]::Round($combinedWAll, 1)
        MaxTemp = $tempMax
        SmClk = $smClk
        MemClk = $memClk
    }

    Log "  cap=${cap}W combined=${combinedWAll}W temp=${tempMax}°C narr_tps=${narrTps}"

    # Wait for thermal stability
    Start-Sleep -Seconds 2
}

# Restore caps
Restore-Gpus

# Summary
Write-Host ""
Write-Host "=== power-cap-sweep summary ==="
$summaryHeader = "  {0,6} {1,9} {2,9} {3,10} {4,9} {5,6} {6,8} {7,11}" -f "Cap", "Narr TPS", "Code TPS", "Combined W", "Max°C", "SM", "Mem", "Pwr%"
Write-Host $summaryHeader
foreach ($row in $summaryRows) {
    Write-Host ("  {0,6} {1,9} {2,9} {3,10,1} {4,9,0} {5,6} {6,8} {7,11}" -f
        "$($row.Cap)W", $row.NarrTps, $row.CodeTps, $row.CombinedW, $row.MaxTemp, $row.SmClk, $row.MemClk, "-")
}

Write-Host ""
Write-Host "Sweep complete. Cooling class: $Cooling"
Write-Host "Report written to: /tmp/power-cap-summary.md (placeholder)"
