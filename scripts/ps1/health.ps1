# Requires -Version 5.1
#
# health.ps1 — operational health check for the running club-3090 server.
#
# Different from verify-full.ps1: that one tests functionality (does
# tool calling work? does long ctx recall correctly?). This one tells
# you runtime state: is the container up, what's the KV pool look
# like, is spec-decode actually firing, any recent errors?
#
# Usage:
#   scripts/health.ps1
#   scripts/health.ps1 --watch          # refresh every 5s (Ctrl-C to stop)
#   $env:URL="http://localhost:8010"; scripts/health.ps1
#
# Env:
#   URL          API base. Default: http://localhost:8010
#   CONTAINER    Target a specific named container instead of auto-matching.
#                Default: unset -> auto-match any recognized engine-prefix
#                container (vllm-/llama-cpp-/ik-llama-/sglang-/beellama-).
#   LOG_LINES    How many log lines to scan for AL/errors. Default: 200
#   WATCH_INTERVAL seconds between refreshes for --watch. Default: 5

param(
    [switch]$Watch,
    [switch]$Help
)
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path | Select-String '^#( |$)' | ForEach-Object { $_.Line.Substring(2) }
    exit 0
}

$env:PYTHONUTF8 = "1"

$URL = if ($env:URL) { $env:URL } else { "http://localhost:8010" }
$CONTAINER = if ($env:CONTAINER) { $env:CONTAINER } else { "" }
$LOG_LINES = if ($env:LOG_LINES) { [int]$env:LOG_LINES } else { 200 }
$WATCH_INTERVAL = if ($env:WATCH_INTERVAL) { [int]$env:WATCH_INTERVAL } else { 5 }

$ENGINE_PREFIX_RE = '^(vllm-|llama-cpp-|ik-llama-|sglang-|beellama-)'

function Write-OK { Write-Host "  [OK] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "  [WARN] $args" -ForegroundColor Yellow }
function Write-Fail { Write-Host "  [FAIL] $args" -ForegroundColor Red }
function Write-Dim { Write-Host "  $args" -ForegroundColor DarkGray }

function Run-Health {
    Write-Host ""
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "club-3090 health check ($timestamp)"
    Write-Host "Endpoint: $URL"
    Write-Host ""

    # 1. Server reachable
    try {
        $modelsJson = Invoke-WebRequest -Uri "$URL/v1/models" -UseBasicParsing -TimeoutSec 5
        Write-OK "API reachable on /v1/models"
    } catch {
        Write-Fail "API not reachable at $URL - is the container running?"
        Write-Host ""
        Write-Host "  -> scripts/switch.ps1 --list   # show available variants" -ForegroundColor DarkGray
        Write-Host "  -> scripts/launcher.ps1          # boot one with the wizard" -ForegroundColor DarkGray
        return $false
    }

    # Detect served model name + engine
    try {
        $modelsObj = $modelsJson.Content | ConvertFrom-Json
        $modelData = if ($modelsObj.data -and $modelsObj.data.Count -gt 0) { $modelsObj.data[0] } else { @{} }
        $modelName = if ($modelData.id) { $modelData.id } else { "unknown" }
        
        $engine = "vLLM"
        if ($modelsJson.Content -match "llamacpp") { $engine = "llama.cpp" }
        
        Write-OK "Serving model: $modelName  (engine: $engine)"
    } catch {
        Write-Warn "Could not parse model info"
        $modelName = "unknown"
        $engine = "vLLM"
    }

    # 2. Container
    $foundContainer = ""
    if ($CONTAINER) {
        try {
            $containers = docker ps --format '{{.Names}}' 2>$null
            $foundContainer = ($containers | Where-Object { $_ -eq $CONTAINER }) | Select-Object -First 1
        } catch { $foundContainer = "" }
    } else {
        try {
            $containers = docker ps --format '{{.Names}}' 2>$null
            $foundContainer = ($containers | Where-Object { $_ -match $ENGINE_PREFIX_RE }) | Select-Object -First 1
        } catch { $foundContainer = "" }
    }

    if (-not $foundContainer) {
        Write-Warn "No matching container running on this host (server may be on another machine, or running as a host process)"
    } else {
        try {
            $inspect = docker inspect $foundContainer 2>$null
            $containerId = ""
            $statusStr = ""
            $started = ""
            
            if ($inspect) {
                $parsed = $inspect | ConvertFrom-Json
                if ($parsed[0].Id) { $containerId = $parsed[0].Id.Substring(0, 12) }
                if ($parsed[0].State) { $statusStr = $parsed[0].State.Status }
                if ($parsed[0].State.StartedAt) { $started = $parsed[0].State.StartedAt }
            }
            
            if ($statusStr -eq "running") {
                $uptime = "?"
                if ($started) {
                    try {
                        $startDt = [DateTimeOffset]::Parse($started).UtcDateTime
                        $now = [DateTimeOffset]::Now.UtcDateTime
                        $diff = ($now - $startDt).TotalSeconds
                        if ($diff -lt 60) { $uptime = "$([math]::Floor($diff))s" }
                        elseif ($diff -lt 3600) { $uptime = "$([math]::Floor($diff/60))m$([math]::Floor($diff%60))s" }
                        else { $uptime = "$([math]::Floor($diff/3600))h$([math]::Floor(($diff%3600)/60))m" }
                    } catch { $uptime = "?" }
                }
                Write-OK "Container $foundContainer ($containerId) - up $uptime"
            } else {
                Write-Fail "Container $foundContainer status: $statusStr"
            }
        } catch {
            Write-Warn "Could not inspect container"
        }
    }

    # 3. VRAM
    Write-Host ""
    Write-Host "GPU VRAM:"
    try {
        $smi = nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($smi) {
            $smi | ForEach-Object {
                $parts = $_.Split(',') | ForEach-Object { $_.Trim() }
                if ($parts.Count -ge 6) {
                    $idx = $parts[0]
                    $name = $parts[1]
                    $used = [double]$parts[2]
                    $total = [double]$parts[3]
                    $util = $parts[4]
                    $temp = $parts[5]
                    $pct = if ($total -gt 0) { $used / $total * 100 } else { 0 }
                    $bar = ""
                    $filled = [math]::Floor($pct / 5)
                    for ($i = 0; $i -lt $filled; $i++) { $bar += "█" }
                    for ($i = $bar.Length; $i -lt 20; $i++) { $bar += "·" }
                    Write-Host "  GPU $idx ($name):  [$bar] $used / $total MiB ($([math]::Round($pct,0))%)  util=${util}%  temp=${temp}C"
                }
            }
        } else {
            Write-Dim "(nvidia-smi not available or no output)"
        }
    } catch {
        Write-Dim "(nvidia-smi not available)"
    }

    # 4. Engine-specific runtime state
    Write-Host ""
    if ($foundContainer) {
        try {
            $logs = docker logs --tail $LOG_LINES $foundContainer 2>&1
            
            if ($engine -eq "vLLM") {
                Write-Host "vLLM runtime (last $LOG_LINES log lines):"
                
                $kvLine = ($logs | Select-String 'GPU KV cache usage: [0-9.]+%' | Select-Object -Last 1).Line
                if ($kvLine) {
                    $kvMatch = $kvLine -match 'GPU KV cache usage: ([0-9.]+)%'
                    if ($kvMatch) {
                        Write-OK "KV cache: $($Matches[1])%"
                    }
                } else {
                    Write-Dim "KV cache: no recent usage line in logs"
                }
                
                $alLines = $logs | Select-String 'Mean acceptance length: [0-9.]+' | Select-Object -Last 5
                if ($alLines) {
                    $vals = @()
                    $alLines | ForEach-Object {
                        if ($_ -match 'Mean acceptance length: ([0-9.]+)') {
                            $vals += [double]$Matches[1]
                        }
                    }
                    if ($vals.Count -gt 0) {
                        $avg = ($vals | Measure-Object -Average).Average
                        $csv = ($vals -join ",")
                        Write-OK "MTP/Spec-decode: AL last 5 = $avg  ($csv)"
                    }
                } else {
                    Write-Dim "Spec-decode: no recent acceptance metric in logs (server may be idle)"
                }
                
                $tputLine = ($logs | Select-String 'Avg generation throughput: [0-9.]+ tokens/s' | Select-Object -Last 1).Line
                if ($tputLine) {
                    if ($tputLine -match 'Avg generation throughput: ([0-9.]+) tokens/s') {
                        Write-OK "Last gen throughput: $($Matches[1]) tokens/s"
                    }
                }
            } else {
                Write-Host "llama.cpp runtime (last $LOG_LINES log lines):"
                $slotLines = $logs | Select-String 'update_slots: all slots are idle|prompt processing|n_tokens =' | Select-Object -Last 3
                if ($slotLines) {
                    Write-OK "Slot activity (recent):"
                    $slotLines | ForEach-Object { Write-Host "      $_" }
                } else {
                    Write-Dim "No slot activity in last $LOG_LINES lines (server may be idle)"
                }
            }

            # 5. Recent errors
            Write-Host ""
            Write-Host "Recent errors / warnings (last $LOG_LINES log lines):"
            $errors = $logs | Select-String 'ERROR|CRITICAL|Traceback|OutOfMemory|CUDA error|Failed' | Where-Object { $_.Line -notmatch 'INFO' } | Select-Object -Last 5
            
            if (-not $errors) {
                Write-OK "no errors logged"
            } else {
                $errCount = ($errors | Measure-Object).Count
                Write-Fail "$errCount error/warning line(s) - last 5:"
                $errors | Select-Object -First 5 | ForEach-Object { Write-Host "      $_" }
            }
        } catch {
            Write-Warn "Could not read container logs"
        }
    }

    Write-Host ""
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  health check complete"
    return $true
}

if ($Watch) {
    while ($true) {
        Clear-Host
        $null = Run-Health
        Write-Host ""
        Write-Host "Refresh every ${WATCH_INTERVAL}s - Ctrl-C to stop"
        Start-Sleep -Seconds $WATCH_INTERVAL
    }
} else {
    $result = Run-Health
    if (-not $result) { exit 1 }
}
