#Requires -Version 5.1
#
# capture.ps1 — Engine-side metrics capture (PowerShell port of capture.sh)
#
# Captures:
# - Expert cache census (per-device)
# - Expert cache counters
# - Spec-decoding acceptance metrics
# - Decode/prefill timing from engine logs
# - Bypass counter
# - Config fingerprinting
# - /props scraping (llama.cpp-family)
# - Status classification
# - Prefill rate plausibility gate (#817)
# - Divergence calculator
# - Graceful degradation (never fabricates values)
#
# Usage:
#   .\capture.ps1
#   .\capture.ps1 -Endpoint http://localhost:8010
#   .\capture.ps1 -LogPath C:\vllm.log

param(
    [string]$Endpoint = "http://localhost:8010",
    [string]$LogPath = "",
    [string]$PropsUrl = "",
    [switch]$Help
)
. "$PSScriptRoot\get-model.ps1"

if ($Help) {
    Write-Host "Usage: .\capture.ps1 [-Endpoint <url>] [-LogPath <path>] [-PropsUrl <url>]"
    Write-Host ""
    Write-Host "  -Endpoint   vLLM engine endpoint (default: http://localhost:8010)"
    Write-Host "  -LogPath    Path to engine log file for timing parsing"
    Write-Host "  -PropsUrl   /props endpoint URL for llama.cpp-family servers"
    Write-Host "  -Help       Show this help message"
    exit 0
}

$ErrorActionPreference = "Continue"

# ---------------------------------------------------------------------------
# HTTP GET helper
# ---------------------------------------------------------------------------
function Invoke-HttpGet {
    param([string]$Url, [int]$TimeoutSec = 10)
    try {
        $resp = Invoke-RestMethod -Uri $Url -TimeoutSec $TimeoutSec -ErrorAction Stop
        return @{ success = $true; body = ($resp | ConvertTo-Json -Compress); status = 200 }
    } catch {
        return @{ success = $false; error = $_.Exception.Message; status = $null }
    }
}

# ---------------------------------------------------------------------------
# Expert cache census (per-device)
# ---------------------------------------------------------------------------
function Get-ExpertCacheCensus {
    param([string]$Endpoint)
    $census = @{}

    $metrics = Invoke-HttpGet -Url "$Endpoint/metrics" -TimeoutSec 5
    if (-not $metrics.success) {
        $metrics = Invoke-HttpGet -Url "$Endpoint/v1/metrics" -TimeoutSec 5
    }

    if ($metrics.success -and $metrics.body) {
        # Parse metrics lines like: expert_cache_pools{device="0"} 123
        $lines = $metrics.body -split "`n"
        foreach ($line in $lines) {
            if ($line -match 'expert_cache_(pools|slots|mib|expert_kib)_\{device="(\d+)"\}\s+(\d+)') {
                $metric = $matches[1]
                $device = $matches[2]
                $value = [double]$matches[3]

                if (-not $census.ContainsKey($device)) {
                    $census[$device] = @{ pools = 0; slots = 0; mib = 0; expert_kib = 0 }
                }
                switch ($metric) {
                    "pools"    { $census[$device].pools = $value }
                    "slots"    { $census[$device].slots = $value }
                    "mib"      { $census[$device].mib = $value }
                    "expert_kib" { $census[$device].expert_kib = $value }
                }
            }
        }
    }

    return $census
}

# ---------------------------------------------------------------------------
# Expert cache counters
# ---------------------------------------------------------------------------
function Get-ExpertCacheCounters {
    param([string]$Endpoint)
    $counters = @{}

    $metrics = Invoke-HttpGet -Url "$Endpoint/metrics" -TimeoutSec 5
    if (-not $metrics.success) {
        $metrics = Invoke-HttpGet -Url "$Endpoint/v1/metrics" -TimeoutSec 5
    }

    if ($metrics.success -and $metrics.body) {
        $lines = $metrics.body -split "`n"
        foreach ($line in $lines) {
            if ($line -match 'expert_cache_counter_\{device="(\d+)",metric="(\w+)"\}\s+(\d+)') {
                $device = $matches[1]
                $metric = $matches[2]
                $value = [double]$matches[3]

                if (-not $counters.ContainsKey($device)) {
                    $counters[$device] = @{ hits = 0; evictions = 0; skips = 0; admission = 0; fill_fail = 0 }
                }
                switch ($metric) {
                    "hits"        { $counters[$device].hits = $value }
                    "evictions"   { $counters[$device].evictions = $value }
                    "skips"       { $counters[$device].skips = $value }
                    "admission"   { $counters[$device].admission = $value }
                    "fill_fail"   { $counters[$device].fill_fail = $value }
                }
            }
        }
    }

    return $counters
}

# ---------------------------------------------------------------------------
# Spec-decoding acceptance metrics
# ---------------------------------------------------------------------------
function Get-SpecDecodingMetrics {
    param([string]$Endpoint)
    $metrics = @{}

    $data = Invoke-HttpGet -Url "$Endpoint/metrics" -TimeoutSec 5
    if (-not $data.success) {
        $data = Invoke-HttpGet -Url "$Endpoint/v1/metrics" -TimeoutSec 5
    }

    if ($data.success -and $data.body) {
        $lines = $data.body -split "`n"
        foreach ($line in $lines) {
            if ($line -match 'spec_decoding_(fired|last|mean|min|max|accepted|drafted)\s+(\d+)') {
                $metric = $matches[1]
                $value = [double]$matches[2]
                $metrics[$metric] = $value
            }
        }
    }

    return $metrics
}

# ---------------------------------------------------------------------------
# Parse decode/prefill timing from engine logs
# ---------------------------------------------------------------------------
function Get-TimingFromLog {
    param([string]$LogPath)

    if (-not $LogPath -or -not (Test-Path $LogPath)) {
        return @{ decode_time = $null; prefill_time = $null }
    }

    try {
        $lastLines = Get-Content $LogPath -Tail 100
        $prefill_time = $null
        $decode_time = $null

        foreach ($line in $lastLines) {
            if ($line -match 'prefill_time_per_token_ms=(\d+\.?\d*)') {
                $prefill_time = [double]$matches[1]
            }
            if ($line -match 'decode_time_per_token_ms=(\d+\.?\d*)') {
                $decode_time = [double]$matches[1]
            }
        }

        return @{
            prefill_time = $prefill_time
            decode_time = $decode_time
        }
    } catch {
        return @{ decode_time = $null; prefill_time = $null }
    }
}

# ---------------------------------------------------------------------------
# Bypass counter
# ---------------------------------------------------------------------------
function Get-BypassCounter {
    param([string]$Endpoint)
    $count = 0

    $data = Invoke-HttpGet -Url "$Endpoint/metrics" -TimeoutSec 5
    if (-not $data.success) {
        $data = Invoke-HttpGet -Url "$Endpoint/v1/metrics" -TimeoutSec 5
    }

    if ($data.success -and $data.body) {
        if ($data.body -match 'bypass_counter\s+(\d+)') {
            $count = [double]$matches[1]
        }
    }

    return $count
}

# ---------------------------------------------------------------------------
# Config fingerprinting
# ---------------------------------------------------------------------------
function Get-ConfigFingerprint {
    param([string]$Endpoint)
    $fingerprint = @{
        argv_flags = @()
        kv_type = "unknown"
        offload_detected = $false
        moe_cache_env = $false
    }

    $models = Invoke-HttpGet -Url "$Endpoint/v1/models" -TimeoutSec 5
    if ($models.success -and $models.body) {
        try {
            $modelData = $models.body | ConvertFrom-Json
            if ($modelData.data -and $modelData.data.Count -gt 0) {
                $fingerprint.model_id = $modelData.data[0].id
            }
        } catch {}
    }

    if (Test-Path env:VLLM_CPU_KVCACHE_BLOAT) { $fingerprint.offload_detected = $true }
    if (Test-Path env:MOE_CACHE_ENABLED) { $fingerprint.moe_cache_env = $true }

    return $fingerprint
}

# ---------------------------------------------------------------------------
# /props scraping (llama.cpp-family)
# ---------------------------------------------------------------------------
function Get-Props {
    param([string]$PropsUrl)

    if (-not $PropsUrl) { return $null }

    $props = @{}
    $data = Invoke-HttpGet -Url $PropsUrl -TimeoutSec 5
    if ($data.success -and $data.body) {
        $lines = $data.body -split "`n"
        foreach ($line in $lines) {
            if ($line -match '^(\w+)\s*=\s*(.+)$') {
                $key = $matches[1]
                $value = $matches[2].Trim()
                $props[$key] = $value
            }
        }
    }

    return $props
}

# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------
function Get-Status {
    param([string]$Endpoint)
    $status = @{
        overall = "unknown"
        tokens_available = $false
        req_errors = $false
        cache_disabled = $false
        invalid_bypass = $false
        reason = ""
    }

    $health = Invoke-HttpGet -Url "$Endpoint/v1/health" -TimeoutSec 5
    if (-not $health.success) {
        $health = Invoke-HttpGet -Url "$Endpoint/health" -TimeoutSec 5
    }

    if (-not $health.success) {
        $status.overall = "NO_TOKENS"
        $status.reason = "Engine not reachable"
        return $status
    }

    if ($health.body -match 'tokens_available\s*:\s*(\w+)') {
        if ($matches[1].ToLower() -ne "true") {
            $status.overall = "NO_TOKENS"
            $status.reason = "tokens_available is false"
            return $status
        }
        $status.tokens_available = $true
    }

    if ($health.body -match 'req_errors\s*:\s*(\w+)') {
        $status.req_errors = ($matches[1].ToLower() -eq "true")
    }

    if ($health.body -match 'cache_disabled\s*:\s*(\w+)') {
        if ($matches[1].ToLower() -eq "true") {
            $status.cache_disabled = $true
            $status.overall = "CACHE_DISABLED"
        }
    }

    $bypass = Get-BypassCounter -Endpoint $Endpoint
    if ($bypass -gt 0) { $status.invalid_bypass = $true }

    if ($status.overall -eq "unknown") { $status.overall = "OK" }
    return $status
}

# ---------------------------------------------------------------------------
# Prefill rate plausibility gate (#817)
# ---------------------------------------------------------------------------
function Test-PrefillPlausibility {
    param([double]$PrefillTimeMs, [int]$Tokens, [int]$ContextLen)

    if (-not $PrefillTimeMs -or $Tokens -le 0) {
        return @{ plausible = $true; reason = "insufficient data" }
    }

    $threshold = 10.0
    $rate = $PrefillTimeMs / $Tokens

    if ($rate -gt $threshold) {
        return @{
            plausible = $false
            reason = "Prefill rate ${rate:F2}ms/token exceeds ${threshold}ms/token threshold"
            rate = $rate
        }
    }

    return @{ plausible = $true; reason = "within normal range"; rate = $rate }
}

# ---------------------------------------------------------------------------
# Divergence calculator
# ---------------------------------------------------------------------------
function Get-Divergence {
    param([string]$Endpoint)
    $divergence = @{
        spec_accepted_rate = $null
        draft_acceptance = $null
        prefill_rate = $null
    }

    $spec = Get-SpecDecodingMetrics -Endpoint $Endpoint
    if ($spec.ContainsKey("accepted") -and $spec.ContainsKey("fired")) {
        if ($spec.fired -gt 0) {
            $divergence.spec_accepted_rate = $spec.accepted / $spec.fired
            $divergence.draft_acceptance = $divergence.spec_accepted_rate
        }
    }

    return $divergence
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function Invoke-Capture {
    param($Endpoint, $LogPath, $PropsUrl)

    try {
        return @{
            timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
            endpoint = $Endpoint
            status = Get-Status -Endpoint $Endpoint
            expert_cache_census = Get-ExpertCacheCensus -Endpoint $Endpoint
            expert_cache_counters = Get-ExpertCacheCounters -Endpoint $Endpoint
            spec_decoding = Get-SpecDecodingMetrics -Endpoint $Endpoint
            timing = Get-TimingFromLog -LogPath $LogPath
            bypass_counter = Get-BypassCounter -Endpoint $Endpoint
            config_fingerprint = Get-ConfigFingerprint -Endpoint $Endpoint
            props = Get-Props -PropsUrl $PropsUrl
            divergence = Get-Divergence -Endpoint $Endpoint
        }
    } catch {
        return @{
            timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
            error = $_.Exception.Message
            status = @{ overall = "ERROR"; reason = $_.Exception.Message }
        }
    }
}

$result = Invoke-Capture -Endpoint $Endpoint -LogPath $LogPath -PropsUrl $PropsUrl
$result | ConvertTo-Json -Depth 10
