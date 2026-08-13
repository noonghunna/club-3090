#Requires -Version 5.1
#
# verify-full.ps1 - Full-functional test (PowerShell port of verify-full.sh)
# PS5.1 compatible - no ?? operator, no ternary, no backtick-newline in strings
#
# Usage:
#   .\verify-full.ps1
#   .\verify-full.ps1 -Bench
#   $env:MTP_ACCEPT_MIN = '1.8'; .\verify-full.ps1

param(
    [string]$Url,
    [string]$Model,
    [string]$Container,
    [switch]$Bench
)
. "$PSScriptRoot\get-model.ps1"

$ErrorActionPreference = "Continue"

# Null-coalescing helper (replaces ?? which is PS7+)
function nco { param($a, $b); if ($a -eq $null) { $b } else { $a } }

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
$PS1_DIR = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$PS1_DIR = Resolve-Path $PS1_DIR
$SCRIPTS_ROOT = Split-Path $PS1_DIR -Parent
$RESULTS_DIR = if ($env:RESULTS_DIR) { $env:RESULTS_DIR } else { Join-Path $SCRIPTS_ROOT "ps1-results" }
$TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"
$RUN_DIR = Join-Path $RESULTS_DIR "verify-full-$TIMESTAMP"
New-Item -ItemType Directory -Force -Path $RUN_DIR | Out-Null
$LOG_FILE = Join-Path $RUN_DIR "verify-full.log"
function Write-Log { param($Text); $Text | Out-File -FilePath $LOG_FILE -Append -Encoding utf8; Write-Host $Text }

# ---------------------------------------------------------------------------
# Defaults / auto-detect
# ---------------------------------------------------------------------------
$ROOT = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$ROOT = Split-Path $ROOT -Parent

# Collect check results for summary.json
$Checks = @()

if (-not $Url) {
    $Url = nco $env:URL "http://localhost:8010"
}
if (-not $Model) {
    try {
        $resp = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction Stop
        $Model = $resp.data[0].id
    } catch { $Model = $DETECTED_MODEL }
}
if (-not $Container) { $Container = "vllm-8010" }

$FAILED = 0
$ENGINE_KIND = "unknown"

function Write-Check { param($Label, $Color = "White"); Write-Log "  $Label" }
function Pass { param($Msg); Write-Log "  [PASS] $Msg" -NoNewline; Write-Host "" -NoNewline; Write-Host "" -ForegroundColor Green; $Checks += @{ name = $Msg; status = "pass"; detail = ""; time = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss") }; return $true }
function Fail { param($Msg, $Hint=""); Write-Log "  [FAIL] $Msg" -NoNewline; Write-Host "" -NoNewline; Write-Host "" -ForegroundColor Red; if ($Hint) { Write-Log "       -> $Hint" -NoNewline; Write-Host "" -NoNewline; Write-Host "" -ForegroundColor Yellow }; $Checks += @{ name = $Msg; status = "fail"; detail = $Hint; time = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss") }; return $false }
function Skip { param($Msg); Write-Log "  [SKIP] $Msg" -NoNewline; Write-Host "" -NoNewline; Write-Host "" -ForegroundColor DarkYellow; $Checks += @{ name = $Msg; status = "skip"; detail = ""; time = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss") } }

# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------
function Detect-Engine {
    try {
        $null = Invoke-RestMethod -Uri "$Url/props" -TimeoutSec 3 -ErrorAction Stop
        return "llamacpp"
    } catch {}
    try {
        $fp = nco ((Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -TimeoutSec 300 -ContentType 'application/json' -Body (@{
            model = $Model; messages = @(@{ role = "user"; content = "hi" }); max_tokens = 1
        } | ConvertTo-Json -Depth 10) | ConvertFrom-Json).system_fingerprint) ""
        if ($fp -match '^vllm-') { return "vllm" }
        if ($fp -match '^sglang-') { return "sglang" }
    } catch {}
    if ($Container -match '^vllm-') { return "vllm" }
    if ($Container -match '^llama-cpp-') { return "llamacpp" }
    return "unknown"
}

$ENGINE_KIND = Detect-Engine

Write-Log ""
Write-Log "=========================================================="
Write-Log "  verify-full.ps1 - Full Functional Test"
Write-Log "=========================================================="
Write-Log "  endpoint:  $Url"
Write-Log "  model:     $Model"
Write-Log "  engine:    $ENGINE_KIND"
Write-Log ""

# ---------------------------------------------------------------------------
# 1. Server reachable
# ---------------------------------------------------------------------------
Write-Log "[1/9] Server reachable on /v1/models ..."
try {
    $null = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction Stop
    if (Pass "server is serving") {} else { $FAILED++ }
} catch {
    Fail "no response from $Url/v1/models" "Start the stack: docker compose up -d; docker logs -f $Container"
    $FAILED++
}

# ---------------------------------------------------------------------------
# 2. Genesis patches applied (vLLM-only)
# ---------------------------------------------------------------------------
Write-Log "[2/9] Genesis patches applied ..."
if ($ENGINE_KIND -in @("llamacpp", "sglang")) {
    Skip "$ENGINE_KIND engine - Genesis is vLLM-only, not applicable"
} else {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Skip "docker not in PATH (host engine build?)"
    } elseif (-not (docker inspect "$Container" -ErrorAction SilentlyContinue)) {
        Skip "container '$Container' not found (host engine build? CONTAINER=none for host endpoints)"
    } else {
        $dockerLogs = docker logs "$Container" 2>&1
        if ($dockerLogs -match '\[Genesis\] FAILED') {
            Fail "Genesis apply_all reported FAILED patch(es)" "Inspect: docker logs $Container 2>&1 | grep -E 'Genesis.*FAILED' | head"
            $FAILED++
        } elseif ($dockerLogs -match 'apply_all elapsed') {
            Pass "Genesis patches applied (apply_all completed clean)"
        } elseif ($dockerLogs -match '\[Genesis\] applied:') {
            Pass "Genesis patches applied (partial log - apply_all may still be running)"
        } else {
            Skip "no Genesis marker in logs (container restarted, or Genesis not loaded)"
        }
    }
}

# ---------------------------------------------------------------------------
# 3. Cold-start warmup
# ---------------------------------------------------------------------------
Write-Log "[warmup] priming engine (cold cudagraph/JIT, up to 180s, not scored) ..."
$warmupOk = $false
try {
    $null = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -TimeoutSec 180 -ContentType 'application/json' -Body (@{
        model = $Model; messages = @(@{ role = "user"; content = "ping" }); max_tokens = 1
        temperature = 0.0; chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 10 -ErrorAction Stop)
    Write-Log "[warmup] engine warm"
    $warmupOk = $true
} catch {
    Write-Log "[warmup] warmup request did not return in 180s - [4/9] will surface a real outage if present"
}

# ---------------------------------------------------------------------------
# 4. Basic completion - Paris sanity
# ---------------------------------------------------------------------------
Write-Log "[3/9] Basic completion - capital of France ..."
try {
    $resp = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -TimeoutSec 300 -ContentType 'application/json' -Body (@{
        model = $Model; messages = @(@{ role = "user"; content = "What is the capital of France? One short sentence." });
        max_tokens = 30; temperature = 0.6; chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 10 -ErrorAction Stop)
    $content = nco $resp.choices[0].message.content ""
    if ($content -match '(?i)Paris') {
        Pass "reply contains 'Paris'"
    } else {
        $preview = $content.Substring(0, [Math]::Min(80, $content.Length))
        Fail "reply didn't mention Paris: $preview" "Model may be loading badly or wrong chat template."
        $FAILED++
    }
} catch {
    Fail "completion request failed" "Check docker logs $Container"
    $FAILED++
}

# ---------------------------------------------------------------------------
# 5. Tool calling
# ---------------------------------------------------------------------------
Write-Log "[4/9] Tool calling ..."
if ($env:SKIP_TOOLS -eq "1") {
    Skip "SKIP_TOOLS=1 (expected for default config - see README Known issue)"
} else {
    try {
        $resp = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -TimeoutSec 300 -ContentType 'application/json' -Body (@{
            model = $Model; messages = @(@{ role = "user"; content = "What is the weather in San Francisco? Use the get_weather tool." });
            tools = @( @{ type = "function"; function = @{
                name = "get_weather"; description = "Get weather for a city.";
                parameters = @{ type = "object"; properties = @{ city = @{ type = "string" } }; required = @("city") }
            }} );
            tool_choice = "auto"; max_tokens = 200; temperature = 0.3;
            chat_template_kwargs = @{ enable_thinking = $false }
        } | ConvertTo-Json -Depth 10 -ErrorAction Stop)

        $msg = $resp.choices[0].message
        $toolCalls = $msg.tool_calls
        $content = nco $msg.content ""

        if ($content -match '<\|tool_call\|>') {
            Fail "model emitted <|tool_call|> as inline text (tool_calls[] empty)" "Known issue: MTP x TurboQuant incompat. Use docker-compose.tools.yml or .tools-text.yml."
            $FAILED++
        } elseif ($toolCalls -and $toolCalls.Count -gt 0) {
            $hasWeather = $false
            foreach ($tc in $toolCalls) {
                if ($tc.function.name -eq "get_weather") { $hasWeather = $true; break }
            }
            if ($hasWeather) {
                Pass "tool_calls[] populated with get_weather"
            } else {
                Fail "unexpected tool_calls structure" "Raw: $($resp | ConvertTo-Json -Depth 10)"
                $FAILED++
            }
        } else {
            Fail "no tool_calls in response" "Raw: $($resp | ConvertTo-Json -Depth 10)"
            $FAILED++
        }
    } catch {
        Fail "tool-call request failed" "Check docker logs"
        $FAILED++
    }
}

# ---------------------------------------------------------------------------
# 6. Streaming (SSE) - non-tool prompt
# ---------------------------------------------------------------------------
Write-Log "[5/9] Streaming (SSE) ..."
try {
    $requestBody = @{
        model = $Model; messages = @(@{ role = "user"; content = "Write a three-sentence haiku about debugging." });
        max_tokens = 120; temperature = 0.6; stream = $true;
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 10

    $webClient = New-Object System.Net.WebClient
    $webClient.Headers.Add("Content-Type", "application/json")
    $webClient.Headers.Add("Accept", "text/event-stream")
    $streamBytes = $webClient.UploadData("$Url/v1/chat/completions", "POST", [System.Text.Encoding]::UTF8.GetBytes($requestBody))
    $streamText = [System.Text.Encoding]::UTF8.GetString($streamBytes)

    $text = ""
    $chunks = 0
    foreach ($line in $streamText -split "`n") {
        $line = $line.Trim()
        if (-not $line.StartsWith("data: ")) { continue }
        $payload = $line.Substring(6)
        if ($payload -eq "[DONE]") { break }
        try {
            $chunk = $payload | ConvertFrom-Json
            $delta = nco $chunk.choices[0].delta.content ""
            if ($delta) { $text += $delta; $chunks++ }
        } catch { continue }
    }

    if (-not $text -or $chunks -eq 0) {
        Fail "no streaming content received ($chunks chunks)" "Streaming broken - check that vLLM isn't buffering."
        $FAILED++
    } elseif ($chunks -lt 5) {
        Fail "suspiciously few chunks ($chunks) for 120 max_tokens" "SSE may be buffering. Final text: $($text.Substring(0, [Math]::Min(120, $text.Length)))"
        $FAILED++
    } elseif ($text.Length -lt 20) {
        Fail "streamed text too short ($($text.Length) chars)" "Content: $text"
        $FAILED++
    } else {
        $preview = $text.Substring(0, [Math]::Min(80, $text.Length)).Replace("`n", " ")
        Pass "streamed $chunks chunks, $($text.Length) chars: $preview..."
    }
} catch {
    Fail "streaming request failed" "Check docker logs"
    $FAILED++
}

# ---------------------------------------------------------------------------
# 7. Streaming tool-calls (thinking-on)
# ---------------------------------------------------------------------------
Write-Log "[6/9] Streaming tool-calls (thinking-on) ..."
if ($env:SKIP_TOOLS -eq "1") {
    Skip "SKIP_TOOLS=1 (expected for default config - see README Known issue)"
} else {
    try {
        $requestBody = @{
            model = $Model; messages = @(@{ role = "user"; content = "What is the weather in San Francisco? Use the get_weather tool." });
            tools = @( @{ type = "function"; function = @{
                name = "get_weather"; description = "Get weather for a city.";
                parameters = @{ type = "object"; properties = @{ city = @{ type = "string" } }; required = @("city") }
            }} );
            tool_choice = "auto"; max_tokens = 256; temperature = 0.3; stream = $true;
            chat_template_kwargs = @{ enable_thinking = $true }
        } | ConvertTo-Json -Depth 10

        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("Content-Type", "application/json")
        $webClient.Headers.Add("Accept", "text/event-stream")
        $streamBytes = $webClient.UploadData("$Url/v1/chat/completions", "POST", [System.Text.Encoding]::UTF8.GetBytes($requestBody))
        $streamText = [System.Text.Encoding]::UTF8.GetString($streamBytes)

        $content = ""
        $toolName = ""
        $finish = ""

        foreach ($line in $streamText -split "`n") {
            $line = $line.Trim()
            if (-not $line.StartsWith("data: ")) { continue }
            $payload = $line.Substring(6)
            if ($payload -eq "[DONE]") { break }
            try {
                $chunk = $payload | ConvertFrom-Json
                foreach ($ch in $chunk.choices) {
                    $delta = nco $ch.delta @{}
                    if ($delta.content) { $content += $delta.content }
                    if ($delta.tool_calls) {
                        foreach ($tc in $delta.tool_calls) {
                            if ($tc.function.name) { $toolName += $tc.function.name }
                        }
                    }
                    if ($ch.finish_reason) { $finish = $ch.finish_reason }
                }
            } catch { continue }
        }

        if ($toolName -and $finish -eq "tool_calls" -and $content -notmatch '<\|tool_call\|>') {
            Pass "streamed delta.tool_calls ($toolName) + finish_reason=tool_calls, no <|tool_call|> leak"
        } elseif ($content -match '<\|tool_call\|>') {
            Fail "tool-call DROPPED over streaming - <|tool_call|> leaked into delta.content" "club-3090#145 / vLLM#39056 streaming class."
            $FAILED++
        } else {
            Fail "no streamed tool-call (finish=$finish)" "Raw head: $($streamText.Substring(0, [Math]::Min(200, $streamText.Length)))"
            $FAILED++
        }
    } catch {
        Fail "streaming tool-call request failed" "Check docker logs"
        $FAILED++
    }
}

# ---------------------------------------------------------------------------
# 8. Thinking / reasoning mode
# ---------------------------------------------------------------------------
Write-Log "[7/9] Thinking / reasoning mode ..."
try {
    $resp = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -TimeoutSec 300 -ContentType 'application/json' -Body (@{
        model = $Model; messages = @(@{ role = "user"; content = "What is 2+2? One-line answer." });
        max_tokens = 4000; temperature = 0.3;
        chat_template_kwargs = @{ enable_thinking = $true }
    } | ConvertTo-Json -Depth 10 -ErrorAction Stop)

    $msg = $resp.choices[0].message
    $reasoning = nco (nco $msg.reasoning $msg.reasoning_content) ""
    $content = nco $msg.content ""
    $finish = nco $resp.choices[0].finish_reason "n/a"
    $rLen = $reasoning.Length
    $cLen = $content.Length

    if ($rLen -eq 0) {
        Fail "reasoning field empty (thinking mode didn't engage)" "May indicate Genesis Patch 12 didn't land or chat_template_kwargs not honored."
        $FAILED++
    } elseif ($cLen -eq 0 -and $finish -eq "length") {
        Pass "reasoning $rLen chars (model kept thinking, hit max_tokens before finishing - Qwen3.6 is verbose; thinking IS extracting correctly)"
        Write-Log "    [dim]reasoning head: $($reasoning.Substring(0, [Math]::Min(60, $rLen)))..."
    } elseif ($cLen -eq 0) {
        Fail "reasoning present but content empty, finish=$finish (not length)" "Likely genuine stall - finish_reason should be length if it's just verbosity."
        $FAILED++
    } elseif ($rLen -lt 50) {
        Fail "reasoning suspiciously short ($rLen chars)" "reasoning: $($reasoning.Substring(0, [Math]::Min(60, $rLen)))"
        $FAILED++
    } else {
        Pass "reasoning $rLen chars, content $cLen chars (finish=$finish)"
        Write-Log "    [dim]reasoning: $($reasoning.Substring(0, [Math]::Min(60, $rLen)))..."
        Write-Log "    [dim]content:  $($content.Substring(0, [Math]::Min(60, $cLen)))..."
    }
} catch {
    Fail "thinking request failed" "Check docker logs"
    $FAILED++
}

# ---------------------------------------------------------------------------
# 9. Output quality / cascade detection
# ---------------------------------------------------------------------------
Write-Log "[8/9] Output quality / cascade detection (2K-token completion) ..."
try {
    $resp = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -TimeoutSec 300 -ContentType 'application/json' -Body (@{
        model = $Model; messages = @(@{ role = "user"; content = "Write a detailed 1500-word essay explaining how transformer attention works. Cover: query/key/value projections, scaled dot-product attention, softmax, multi-head attention, positional encodings, and a brief comparison with RNN-based attention." });
        max_tokens = 2000; temperature = 0.6;
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 10 -ErrorAction Stop)

    $content = nco $resp.choices[0].message.content ""
    $finish = nco $resp.choices[0].finish_reason "n/a"
    $clen = $content.Length

    if ($clen -eq 0) {
        Fail "empty completion (finish=$finish)" "Likely silent generation failure"
        $FAILED++
    } elseif ($content -match '<\|tool_call\|>') {
        Fail "MTP x TurboQuant cascade - <|tool_call|> emitted in normal text" "Genesis P64/P65 not active or compose using broken MTP path."
        $FAILED++
    } else {
        $lines = @($content -split "`n" | Where-Object { $_.Trim() })
        $maxRepeat = 0
        $curLine = ""
        $curCount = 0
        foreach ($line in $lines) {
            if ($line -eq $curLine) {
                $curCount++
                if ($curCount -gt $maxRepeat) { $maxRepeat = $curCount }
            } else {
                $curLine = $line
                $curCount = 1
            }
        }

        if ($maxRepeat -ge 5) {
            Fail "repetitive degeneracy - line repeats $maxRepeatx consecutively" "Sampling collapsed (stale-draft? sampler bug?). Check finish_reason=$finish."
            $FAILED++
        } else {
            $words = [regex]::Matches($content.ToLower(), "[a-z']+") | ForEach-Object { $_.Value }
            $sample = $words | Select-Object -First 200
            if ($sample.Count -gt 0) {
                $unique = ($sample | Sort-Object -Unique).Count
                $variety = $unique / $sample.Count
            } else { $variety = 0 }

            if ($variety -ge 0.30) {
                Pass "output OK - $clen chars, variety=$variety, max_line_repeat=$maxRepeat, finish=$finish"
            } else {
                Fail "low lexical variety ($variety, threshold 0.30)" "Possible degenerate output. clen=$clen, finish=$finish"
                $FAILED++
            }
        }
    }
} catch {
    Fail "output quality request failed" "Check docker logs $Container"
    $FAILED++
}

# ---------------------------------------------------------------------------
# 10. MTP acceptance length
# ---------------------------------------------------------------------------
Write-Log "[9/9] MTP acceptance length threshold ..."
if ($ENGINE_KIND -in @("llamacpp", "sglang")) {
    Skip "$ENGINE_KIND engine - MTP acceptance check is vLLM-log-format-specific"
} elseif (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Skip "docker not in PATH (host engine build?)"
} elseif (-not (docker inspect "$Container" -ErrorAction SilentlyContinue)) {
    Skip "container '$Container' not found (CONTAINER=none for host endpoints)"
} else {
    try {
        $null = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -TimeoutSec 300 -ContentType 'application/json' -Body (@{
            model = $Model; messages = @(@{ role = "user"; content = "Count from 1 to 80, one number per line." });
            max_tokens = 500; temperature = 0.0;
            chat_template_kwargs = @{ enable_thinking = $false }
        } | ConvertTo-Json -Depth 10 -ErrorAction Stop)
    } catch {
        Fail "metrics-trigger request failed" "Check docker logs"
        $FAILED++
    }

    Start-Sleep -Seconds 3

    $recent = docker logs --tail 200 "$Container" 2>&1 | Select-String -Pattern "SpecDecoding|acceptance length|spec_decode" -AllMatches | ForEach-Object { $_.Line } | Select-Object -Last 3
    if (-not $recent) {
        Skip "no SpecDecoding metrics in logs (compose may not have spec-decode enabled)"
    } else {
        $al = $null
        foreach ($line in $recent) {
            if ($line -match '(\d+\.\d+)') {
                $al = [double]$matches[1]
                break
            }
        }

        if (-not $al) {
            Skip "couldn't parse AL from metrics"
        } else {
            $acceptMin = [double](nco $env:MTP_ACCEPT_MIN "2.0")
            if ($al -ge $acceptMin) {
                Pass "MTP acceptance length = $al (>= $acceptMin - spec-decode contributing)"
            } else {
                Fail "MTP acceptance length = $al (< $acceptMin - below this profile's floor)" "Check MTP routing and the profile's measured acceptance/throughput evidence."
                $FAILED++
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Log ""
if ($FAILED -eq 0) {
    Write-Log "  All checks passed. Stack is ready for full-functionality use."
} else {
    Write-Log "  $FAILED check(s) failed. See hints above."
}

# ---------------------------------------------------------------------------
# Write summary.json
# ---------------------------------------------------------------------------
$passCount = ($Checks | Where-Object { $_.status -eq "pass" }).Count
$failCount = ($Checks | Where-Object { $_.status -eq "fail" }).Count
$summary = @{
    checks = $Checks
    overall = @{
        pass = $passCount
        fail = $failCount
        total = $Checks.Count
    }
    engine_kind = $ENGINE_KIND
    endpoint = $Url
    model = $Model
    timestamp = $TIMESTAMP
}
$summary | ConvertTo-Json -Depth 5 | Out-File -FilePath (Join-Path $RUN_DIR "summary.json") -Encoding utf8

# ---------------------------------------------------------------------------
# Optional: run bench.sh after all checks pass
# ---------------------------------------------------------------------------
if ($Bench -and $FAILED -eq 0) {
    Write-Log ""
    Write-Log "  --bench: running scripts/bench-full.ps1"
    $scriptDir = Join-Path $ROOT "scripts"
    $env:URL = $Url
    $env:MODEL = $Model
    $env:CONTAINER = $Container
    & "$scriptDir\bench-full.ps1"
}

exit $FAILED
