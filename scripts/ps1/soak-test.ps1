#Requires -Version 5.1
#
# soak-test.ps1 - Multi-turn VRAM accretion / runtime stability test
# (PowerShell port of soak-test.sh)
#
# Tests raw per-request VRAM accretion across sessions, or continuous
# multi-turn agentic traffic that ramps to ~22-25K accumulated context.
# ~10-30 min depending on config.
#
# Usage:
#   .\soak-test.ps1
#   .\soak-test.ps1 -Continuous
#   .\soak-test.ps1 -Quick
#   .\soak-test.ps1 -Help

param(
    [string]$Url,
    [string]$Model,
    [string]$Container,
    [switch]$Continuous,
    [switch]$Quick,
    [switch]$Help
)
. "$PSScriptRoot\get-model.ps1"

$ErrorActionPreference = "Continue"

if ($Help) {
    Write-Host "soak-test.ps1 - multi-turn VRAM-accretion + Cliff 2b validation"
    Write-Host ""
    Write-Host "USAGE: .\soak-test.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "MODES"
    Write-Host "  (default)         fresh mode: 20 sessions × 5 turns (~10-25 min)"
    Write-Host "                    Tests raw per-request VRAM accretion."
    Write-Host "  -Continuous       Cliff 2b detector: 5 sessions × 5 turns,"
    Write-Host "                    ramping context to ~22-25K accumulated tokens."
    Write-Host "                    The only test that catches the multi-turn"
    Write-Host "                    accumulating-context cliff."
    Write-Host "  -Quick            8 sessions × 5 turns, fresh mode (~5-8 min)"
    Write-Host ""
    Write-Host "ENV (advanced - auto-detected by default)"
    Write-Host "  URL                 OpenAI endpoint. Default: http://localhost:8010"
    Write-Host "  MODEL               Served model. Default: first id from /v1/models"
    Write-Host "  SOAK_SESSIONS       Override session count (default: 20 fresh, 5 continuous)"
    Write-Host "  SOAK_TURNS          Override turn count (default: 5)"
    Write-Host "  SOAK_MAX_GROWTH_MIB VRAM-growth fail threshold (default: 200)"
    Write-Host "  SOAK_TIMEOUT_S      Hard wall-clock cap (default: 1800)"
    Write-Host "  CONTAINER           Container name (default: auto-detect)"
    exit 0
}

# ---------------------------------------------------------------------------
# Defaults / auto-detect
# ---------------------------------------------------------------------------
$ROOT = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$ROOT = Split-Path $ROOT -Parent

$SOAK_MODE = if ($Continuous) { "continuous" } else { "fresh" }
$SOAK_SESSIONS = if ($Quick) { 8 } elseif ($Continuous) { 5 } else { 20 }
$SOAK_TURNS = 5
if ($env:SOAK_MAX_GROWTH_MIB) { $SOAK_MAX_GROWTH_MIB = [int]$env:SOAK_MAX_GROWTH_MIB } else { $SOAK_MAX_GROWTH_MIB = 200 }
if ($env:SOAK_TIMEOUT_S) { $SOAK_TIMEOUT_S = [int]$env:SOAK_TIMEOUT_S } else { $SOAK_TIMEOUT_S = 1800 }
if ($env:SOAK_REQ_TIMEOUT_S) { $SOAK_REQ_TIMEOUT_S = [int]$env:SOAK_REQ_TIMEOUT_S } else { $SOAK_REQ_TIMEOUT_S = 600 }

if ($SOAK_MODE -eq "continuous") {
    if ($env:SOAK_SESSIONS) { $SOAK_SESSIONS = [int]$env:SOAK_SESSIONS } else { $SOAK_SESSIONS = 5 }
    if ($env:SOAK_TURNS) { $SOAK_TURNS = [int]$env:SOAK_TURNS } else { $SOAK_TURNS = 5 }
} elseif ($Quick) {
    if ($env:SOAK_SESSIONS) { $SOAK_SESSIONS = [int]$env:SOAK_SESSIONS } else { $SOAK_SESSIONS = 8 }
    if ($env:SOAK_TURNS) { $SOAK_TURNS = [int]$env:SOAK_TURNS } else { $SOAK_TURNS = 5 }
} else {
    if ($env:SOAK_SESSIONS) { $SOAK_SESSIONS = [int]$env:SOAK_SESSIONS } else { $SOAK_SESSIONS = 20 }
    if ($env:SOAK_TURNS) { $SOAK_TURNS = [int]$env:SOAK_TURNS } else { $SOAK_TURNS = 5 }
}

# Auto-detect URL
if (-not $Url) {
    $Url = if ($env:URL) { $env:URL } else { "http://localhost:8010" }
}

# Auto-detect model
if (-not $Model) {
    try {
        $resp = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction Stop
        $Model = $resp.data[0].id
    } catch { $Model = $DETECTED_MODEL }
}

# Auto-detect container
if (-not $Container) {
    try {
        $containers = docker ps --format '{{.Names}}|{{.Ports}}' 2>$null |
            Select-String -Pattern '([0-9]{1,3}\.){3}[0-9]{1,3}:[0-9]+->(8000|8080|30000)/tcp' |
            ForEach-Object { $_.Line }
        if ($containers) {
            $named = $containers | Select-String -Pattern '^(vllm-|llama-cpp-|ik-llama-|sglang-|beellama-)' | Select-Object -First 1
            if ($named) {
                $Container = ($named.Line -split '\|')[0]
            } else {
                $Container = ($containers[0] -split '\|')[0]
            }
        }
    } catch { $Container = "none" }
}
if (-not $Container) { $Container = "none" }

$HOST_MODE = ($Container -eq "none")

# Output directory
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$SOAK_OUTPUT = if ($env:SOAK_OUTPUT) { $env:SOAK_OUTPUT } else { $ROOT + "\ps1-results\soak-$timestamp" }
$SOAK_OUTPUT = $SOAK_OUTPUT -replace '\\', '/'
New-Item -ItemType Directory -Force -Path $SOAK_OUTPUT > $null
New-Item -ItemType Directory -Force -Path (Join-Path $SOAK_OUTPUT "requests") > $null
New-Item -ItemType Directory -Force -Path (Join-Path $SOAK_OUTPUT "responses") > $null
New-Item -ItemType Directory -Force -Path (Join-Path $SOAK_OUTPUT "states") > $null

$TURN_LOG = Join-Path $SOAK_OUTPUT "turn-log.csv"
$GPU_LOG = Join-Path $SOAK_OUTPUT "gpu-log.csv"
$SUMMARY_MD = Join-Path $SOAK_OUTPUT "summary.md"

# CSV headers
"session_id,turn_id,t_ms,vram_mib,ttft_ms,decode_tps,completion_tokens,status,error,decode_basis" | Out-File -FilePath $TURN_LOG -Encoding utf8
"session_id,turn_id,gpu_index,memory_used_mib,utilization_gpu_pct" | Out-File -FilePath $GPU_LOG -Encoding utf8

function Log { param($Msg); Write-Host "[soak] $Msg" }
function Die { param($Msg); Log "ERROR: $Msg"; exit 2 }

Log "host mode: CONTAINER=$Container"
Log "running soak test against $Url (model=$Model, container=$Container)"
Log "mode=$SOAK_MODE sessions=$SOAK_SESSIONS turns=$SOAK_TURNS max_growth=${SOAK_MAX_GROWTH_MIB}MiB timeout=${SOAK_TIMEOUT_S}s"
Log "output=$SOAK_OUTPUT"

# ---------------------------------------------------------------------------
# VRAM helper
# ---------------------------------------------------------------------------
function Get-VramMib {
    try {
        $output = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        $total = 0
        foreach ($line in $output -split "`n") {
            $val = $line.Trim().Replace(",", "").Trim()
            if ($val -match '^\d+$') { $total += [int]$val }
        }
        return $total
    } catch { return 0 }
}

# ---------------------------------------------------------------------------
# Append GPU snapshot
# ---------------------------------------------------------------------------
function Append-GpuSnapshot {
    param($session, $turn)
    try {
        $output = nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null
        $output -split "`n" | ForEach-Object {
            $parts = $_ -split ',' | ForEach-Object { $_.Trim() }
            if ($parts.Count -ge 3) {
                "$session,$turn,$($parts[0]),$($parts[1]),$($parts[2])" | Out-File -FilePath $GPU_LOG -Append -Encoding utf8
            }
        }
    } catch {}
}

# ---------------------------------------------------------------------------
# Capture state
# ---------------------------------------------------------------------------
function Capture-State {
    param($label)
    try {
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits > (Join-Path $SOAK_OUTPUT "nvidia-smi-${label}.csv") 2>$null
    } catch {}
    if (-not $HOST_MODE) {
        try {
            docker stats --no-stream --format '{{json .}}' "$Container" 2>$null | Out-File -FilePath (Join-Path $SOAK_OUTPUT "docker-stats-${label}.jsonl") -Encoding utf8
        } catch {}
    }
}

# ---------------------------------------------------------------------------
# Run a single request
# ---------------------------------------------------------------------------
function Invoke-SoakRequest {
    param($Endpoint, $Model, $RequestBody, $TimeoutSec, $OutputFile, $Mode)

    $result = @{
        status = "error"
        t_ms = 0
        ttft_ms = 0
        decode_tps = "0.0"
        completion_tokens = 0
        error = ""
        decode_basis = "decode"
    }

    try {
        $t0 = Get-Date
        $ttft = $null
        $contentParts = @()
        $usage = $null
        $timings = $null
        $finish = "n/a"
        $reader = $null
        $response = $null

        $bytes = [System.Text.Encoding]::UTF8.GetBytes($RequestBody)

        # Use HttpWebRequest for true streaming (WebClient.UploadData buffers everything)
        $request = [System.Net.HttpWebRequest]::Create("$Endpoint/v1/chat/completions")
        $request.Method = "POST"
        $request.ContentType = "application/json"
        $request.Timeout = $TimeoutSec * 1000
        $request.AllowWriteStreamBuffering = $false
        $request.ContentLength = $bytes.Length
        $requestStream = $request.GetRequestStream()
        $requestStream.Write($bytes, 0, $bytes.Length)
        $requestStream.Close()

        $response = $request.GetResponse()
        $stream = $response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader $stream

        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine().Trim()
            if (-not $line.StartsWith("data: ")) { continue }
            $payload = $line.Substring(6)
            if ($payload -eq "[DONE]") { break }
            try {
                $chunk = $payload | ConvertFrom-Json
                # Extract usage from any chunk (it only appears in the last chunk with empty choices)
                if ($chunk.usage) { $usage = $chunk.usage }
                if ($chunk.timings) { $timings = $chunk.timings }
                if ($chunk.choices) {
                    foreach ($ch in $chunk.choices) {
                        $delta = if ($ch.delta) { $ch.delta } else { @{}}
                        if ($delta.content) {
                            if (-not $ttft) { $ttft = ((Get-Date) - $t0).TotalSeconds }
                            $contentParts += $delta.content
                        }
                        if ($delta.reasoning_content) { $contentParts += $delta.reasoning_content }
                        if ($ch.finish_reason) { $finish = $ch.finish_reason }
                    }
                }
            } catch { continue }
        }

        $reader.Close()
        $stream.Close()
        $response.Close()

        $totalWall = ((Get-Date) - $t0).TotalSeconds
        $totalContent = -join $contentParts
        $compTokens = if ($usage -and $usage.completion_tokens) { [int]$usage.completion_tokens } else { 0 }
        $promptTokens = if ($usage -and $usage.prompt_tokens) { [int]$usage.prompt_tokens } else { 0 }

        $ttftMs = if ($ttft) { [Math]::Round($ttft * 1000) } else { 0 }
        $tMs = [Math]::Round($totalWall * 1000)

        # Decode TPS
        $decodeTps = "0.0"
        $decodeBasis = "decode"
        $decodeWindow = $totalWall

        if ($compTokens -gt 0 -and $decodeWindow -gt 0.01) {
            $decodeTps = [Math]::Round($compTokens / $decodeWindow, 1)
            # Canvas detection: if response arrived in ~1 chunk, it's canvas
            if ($contentParts.Count -le 2 -and $decodeWindow -lt 0.1) {
                $decodeBasis = "wall"
            }
        } elseif ($compTokens -gt 0 -and $totalWall -gt 0) {
            $decodeTps = [Math]::Round($compTokens / $totalWall, 1)
            $decodeBasis = "wall"
        }

        $result.status = "ok"
        $result.t_ms = $tMs
        $result.ttft_ms = $ttftMs
        $result.decode_tps = $decodeTps
        $result.completion_tokens = $compTokens
        $result.decode_basis = $decodeBasis

        # Save metrics
        $metrics = @{
            status = $result.status
            t_ms = $result.t_ms
            ttft_ms = $result.ttft_ms
            decode_tps = $result.decode_tps
            completion_tokens = $result.completion_tokens
            finish_reason = $finish
            decode_basis = $result.decode_basis
        } | ConvertTo-Json

        $metrics | Out-File -FilePath $OutputFile -Encoding utf8

    } catch {
        $result.status = "error"
        $result.error = $_.Exception.Message
        $result | ConvertTo-Json | Out-File -FilePath $OutputFile -Encoding utf8
    }

    return $result
}

# ---------------------------------------------------------------------------
# Generate request body
# ---------------------------------------------------------------------------
function New-SoakRequest {
    param($Model, $Session, $Turn, $Mode, $StateFile)

    if ($Mode -eq "continuous") {
        # Continuous mode: build on accumulated context from state file
        # Matches bash soak-helper.py CONTINUOUS_TURNS pattern with growing
        # tool-call context that ramps to ~22-25K accumulated tokens by turn 5.
        $state = @{}
        if (Test-Path $StateFile) {
            try { $state = Get-Content $StateFile | ConvertFrom-Json } catch {}
        }

        # Multi-turn prompts that grow context — matches bash fixture pattern
        $turnShapes = @(
            # Turn 1: simple question
            "What is the capital of France? One short sentence."
            # Turn 2: explanation request
            "Explain quantum entanglement in simple terms."
            # Turn 3: coding task
            "Write a Python function to compute Fibonacci numbers iteratively."
            # Turn 4: comparison question
            "What are the main differences between REST and GraphQL?"
            # Turn 5: summary request
            "Summarize the plot of Hamlet in 3 sentences."
        )

        $shape = $turnShapes[($Turn - 1) % $turnShapes.Count]

        $messages = @()
        if ($state.messages) {
            $messages += $state.messages
        }
        $messages += @{ role = "user"; content = $shape }

        # Match bash: vary max_tokens across turns (220-2000), use tools
        $maxTokens = switch ($Turn) {
            1 { 350 }; 2 { 350 }; 3 { 400 }; 4 { 500 }; default { 1500 }
        }

        $req = @{
            model = $Model
            messages = $messages
            max_tokens = $maxTokens
            temperature = 0.6
            stream = $true
            stream_options = @{ include_usage = $true }
        }

        # Add tools for turns 2+ (matches bash tool_choice pattern)
        if ($Turn -ge 2) {
            $req["tools"] = @(
                @{
                    type = "function"
                    function = @{
                        name = "read_file"
                        description = "Read a UTF-8 text file."
                        parameters = @{
                            type = "object"
                            properties = @{ path = @{ type = "string"; description = "Path to read." } }
                            required = @("path")
                        }
                    }
                },
                @{
                    type = "function"
                    function = @{
                        name = "grep"
                        description = "Search files for a text pattern."
                        parameters = @{
                            type = "object"
                            properties = @{
                                pattern = @{ type = "string" }
                                dir = @{ type = "string" }
                            }
                            required = @("pattern", "dir")
                        }
                    }
                }
            )
            $req["tool_choice"] = "auto"
        }

        $req
    } else {
        # Fresh mode: independent conversation each turn, but with richer prompts
        # matching bash fresh-mode fixture pattern (tool calls, growing context)
        $turnShapes = @(
            "What is the capital of France? One short sentence."
            "Explain quantum entanglement in simple terms."
            "Write a Python function to compute Fibonacci numbers iteratively."
            "What are the main differences between REST and GraphQL?"
            "Summarize the plot of Hamlet in 3 sentences."
        )

        $shape = $turnShapes[($Turn - 1) % $turnShapes.Count]

        # Match bash: vary max_tokens (220-2000), add tools for deeper turns
        $maxTokens = switch ($Turn) {
            1 { 220 }; 2 { 320 }; 3 { 700 }; 4 { 900 }; default { 2000 }
        }

        $messages = @(@{ role = "user"; content = $shape })

        # Add system prompt + tools for turns 3+ (matches bash tool-use pattern)
        if ($Turn -ge 3) {
            $messages = @(
                @{ role = "system"; content = "You are a concise coding assistant. Prefer tools when file contents or command output would materially change the answer." },
                @{ role = "user"; content = $shape }
            )
        }

        $req = @{
            model = $Model
            messages = $messages
            max_tokens = $maxTokens
            temperature = 0.6
            stream = $true
            stream_options = @{ include_usage = $true }
        }

        if ($Turn -ge 3) {
            $req["tools"] = @(
                @{
                    type = "function"
                    function = @{
                        name = "read_file"
                        description = "Read a UTF-8 text file."
                        parameters = @{
                            type = "object"
                            properties = @{ path = @{ type = "string"; description = "Path to read." } }
                            required = @("path")
                        }
                    }
                }
            )
            $req["tool_choice"] = "auto"
        }

        $req
    }
}

# ---------------------------------------------------------------------------
# Main soak loop
# ---------------------------------------------------------------------------
$START_SECONDS = $SECONDS
$BOOT_VRAM_MIB = $null
$BASELINE_SESSION = 0
$TURNS_RUN = 0
$TIMED_OUT = 0

$modelsJson = Join-Path $SOAK_OUTPUT "models.json"
try {
    Invoke-WebRequest -Uri "$Url/v1/models" -TimeoutSec 10 -OutFile $modelsJson -ErrorAction Stop
} catch { Die "no response from $Url/v1/models" }

# Capture baseline VRAM
Capture-State "baseline"

foreach ($session in 1..$SOAK_SESSIONS) {
    Log "session ${session}/${SOAK_SESSIONS}"
    $sessionErrors = 0
    $stateFile = Join-Path $SOAK_OUTPUT "states/state-s${session}.json"

    if ($SOAK_MODE -eq "continuous") {
        $state = @{}
        if (Test-Path $stateFile) {
            try { $state = Get-Content $stateFile | ConvertFrom-Json } catch {}
        }
        $state.session_id = $session
        $state.messages = @()
        $state | ConvertTo-Json | Out-File -FilePath $stateFile -Encoding utf8
    }

    foreach ($turn in 1..$SOAK_TURNS) {
        # Timeout check
        if (($SECONDS - $START_SECONDS) -ge $SOAK_TIMEOUT_S) {
            $TIMED_OUT = 1
            Log "timeout reached before session=${session} turn=${turn}"
            break
        }

        $reqFile = Join-Path $SOAK_OUTPUT "requests/s${session}-t${turn}.json"
        $metricsFile = Join-Path $SOAK_OUTPUT "responses/s${session}-t${turn}.metrics.json"

        # Build request
        $req = New-SoakRequest -Model $Model -Session $session -Turn $turn -Mode $SOAK_MODE -StateFile $stateFile
        # PS5.1 ConvertTo-Json -Compress doesn't handle nested hashtables (tools).
        # Use -Depth 10 for requests with tools, -Compress for simple ones.
        if ($req.tools) {
            $req | ConvertTo-Json -Depth 10 | Out-File -FilePath $reqFile -Encoding utf8
        } else {
            $req | ConvertTo-Json -Compress | Out-File -FilePath $reqFile -Encoding utf8
        }

        # Execute
        $reqBody = if ($req.tools) { $req | ConvertTo-Json -Depth 10 } else { $req | ConvertTo-Json -Compress }
        $result = Invoke-SoakRequest -Endpoint $Url -Model $Model -RequestBody $reqBody -TimeoutSec $SOAK_REQ_TIMEOUT_S -OutputFile $metricsFile -Mode $SOAK_MODE

        # Update state for continuous mode
        if ($SOAK_MODE -eq "continuous") {
            $state = Get-Content $stateFile | ConvertFrom-Json
            $msg = @{ role = "user"; content = $req.messages[$req.messages.Count - 1].content }
            $state.messages += $msg
            if ($result.status -eq "ok") {
                $state.messages += @{ role = "assistant"; content = "response" }
            }
            $state | ConvertTo-Json | Out-File -FilePath $stateFile -Encoding utf8
        }

        # VRAM snapshot
        $vram = Get-VramMib
        Append-GpuSnapshot -session $session -turn $turn

        # Append to turn log
        "$session,$turn,$($result.t_ms),$vram,$($result.ttft_ms),$($result.decode_tps),$($result.completion_tokens),$($result.status),$($result.error),$($result.decode_basis)" | Out-File -FilePath $TURN_LOG -Append -Encoding utf8

        $TURNS_RUN++
        if ($result.status -eq "error") { $sessionErrors++ }

        Log "  turn ${turn}/${SOAK_TURNS}: status=$($result.status) wall=$($result.t_ms)ms ttft=$($result.ttft_ms)ms decode_tps=$($result.decode_tps) vram=${vram}MiB"
    }

    # Capture warm baseline at end of first clean session
    if (-not $BOOT_VRAM_MIB) {
        if ($sessionErrors -eq 0) {
            $BOOT_VRAM_MIB = Get-VramMib
            $BASELINE_SESSION = $session
            Log "warm baseline after session ${session}: ${BOOT_VRAM_MIB} MiB"
        } else {
            Log "session ${session} had ${sessionErrors} errored turn(s) - NOT anchoring the warm VRAM baseline here"
        }
    }
}

# If no clean session, mark VRAM as unmeasurable
if (-not $BOOT_VRAM_MIB) {
    $BOOT_VRAM_MIB = 0
    $BASELINE_SESSION = 0
    if ($TURNS_RUN -eq 0) { $TIMED_OUT = 1; Log "no completed turns; writing inconclusive summary" }
    else { Log "no error-free session completed - VRAM growth + oscillation are UNMEASURABLE" }
}

# Generate summary
Capture-State "final"

# Build summary markdown
$summary = @"
# Soak Test Summary

- **Endpoint:** $Url
- **Model:** $Model
- **Mode:** $SOAK_MODE
- **Sessions:** $SOAK_SESSIONS
- **Turns per session:** $SOAK_TURNS
- **Total turns:** $TURNS_RUN
- **Max growth threshold:** ${SOAK_MAX_GROWTH_MIB} MiB
- **Timed out:** $TIMED_OUT
- **VRAM baseline session:** $BASELINE_SESSION
- **Baseline VRAM:** $(if ($BOOT_VRAM_MIB) { "${BOOT_VRAM_MIB} MiB" } else { "UNMEASURABLE" })
- **Output:** $SOAK_OUTPUT

## Turn Log
$(Get-Content $TURN_LOG | Select-Object -First 20)

## Artifacts
- turn-log.csv
- gpu-log.csv
- requests/
- responses/
- states/
"@

$summary | Out-File -FilePath $SUMMARY_MD -Encoding utf8

# Verdict
$verdict = "PASS"
$exitCode = 0

if ($TIMED_OUT) {
    $verdict = "INCONCLUSIVE"
    $exitCode = 2
} elseif ($BASELINE_SESSION -eq 0) {
    $verdict = "INCONCLUSIVE"
    $exitCode = 2
} else {
    # Check VRAM growth
    $vramGrowth = (Get-VramMib) - $BOOT_VRAM_MIB
    if ($vramGrowth -gt $SOAK_MAX_GROWTH_MIB) {
        $verdict = "FAIL"
        $exitCode = 1
        Log "VRAM growth ${vramGrowth} MiB exceeds threshold ${SOAK_MAX_GROWTH_MIB} MiB"
    }
}

Log ""
Log "=========================================================="
Log "  Soak Test: $verdict"
Log "=========================================================="
Log "  artifacts: $SOAK_OUTPUT"

if ($verdict -eq "PASS") {
    Write-Host "  PASS - no failure signal fired" -ForegroundColor Green
} elseif ($verdict -eq "FAIL") {
    Write-Host "  FAIL - failure signal detected" -ForegroundColor Red
} else {
    Write-Host "  INCONCLUSIVE - see notes above" -ForegroundColor Yellow
}

exit $exitCode
