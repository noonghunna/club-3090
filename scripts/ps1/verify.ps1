# Requires -Version 5.1
#
# Post-setup smoke test — confirms the stack is healthy before you start using it.
#
# Runs four checks, each short-circuits on failure with an actionable hint:
#   1. Server responds on /v1/models
#   2. Genesis patches applied cleanly (tool_call fix is the fragile one)
#   3. Basic text completion works (Paris sanity)
#   4. Tool calling works end-to-end (request includes tools -> response has tool_calls[])
#
# If check 4 fails but checks 1-3 pass, your Genesis tool_call patch didn't apply
# and you're on a vLLM nightly that drifted past our pinned digest. See the README
# troubleshooting section.
#
# Usage:
#   scripts/verify.ps1
#   scripts/verify.ps1 --watch          # refresh every 5s (Ctrl-C to stop)
#   $env:URL="http://localhost:8010"; scripts/verify.ps1
#
# Env:
#   URL          API base. Default: http://localhost:8010
#   MODEL        Served model name. Default: auto-detected
#   CONTAINER    Docker container name for log scraping. Default: vllm-8010

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

$ROOT_DIR = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent

$URL = if ($env:URL) { $env:URL } else { "http://localhost:8010" }
$MODEL = if ($env:MODEL) { $env:MODEL } else { $DETECTED_MODEL }
$CONTAINER = if ($env:CONTAINER) { $env:CONTAINER } else { "vllm-8010" }

function Show-Check { Write-Host "  [1/4] Server reachable on /v1/models ..." }
function Pass-Check { Write-Host "  [OK] $args" -ForegroundColor Green }
function Fail-Check { param($msg, $hint); Write-Host "  [FAIL] $msg" -ForegroundColor Red; Write-Host "    -> $hint" -ForegroundColor Yellow; exit 1 }

function Run-Verify {
    Write-Host "Running smoke test against ${URL} (model=${MODEL}, container=${CONTAINER})"
    Write-Host ""

    # 1. Server reachable
    Show-Check
    try {
        $null = Invoke-WebRequest -Uri "$URL/v1/models" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        Pass-Check "server is serving"
    } catch {
        Fail-Check "no response from ${URL}/v1/models" "Start the stack: cd compose && docker compose up -d ; docker logs -f ${CONTAINER}"
    }

    # 2. Genesis patches applied cleanly
    Write-Host "[2/4] Genesis patches applied ..."
    $dockerAvailable = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerAvailable) {
        Write-Host "  (skipped - docker not in PATH, cannot read container logs)"
    } else {
        try {
            $null = docker inspect $CONTAINER 2>&1 | Out-Null
        } catch {
            Write-Host "  (skipped - container '${CONTAINER}' not found; if your container has a different name, set CONTAINER=...)"
            return
        }
        $logs = docker logs $CONTAINER 2>&1
        if ($logs -match '\[Genesis\] FAILED') {
            Fail-Check "Genesis apply_all reported FAILED patch(es)" "Inspect: docker logs ${CONTAINER} 2>&1 | grep -E 'Genesis.*FAILED' | head"
        } elseif ($logs -match 'apply_all elapsed') {
            Pass-Check "Genesis patches applied (apply_all completed clean)"
        } elseif ($logs -match '\[Genesis\] applied:') {
            Pass-Check "Genesis patches applied (apply_all may still be running)"
        } else {
            Write-Host "  (warn - no Genesis marker in logs; container may have been restarted. Continuing.)"
        }
    }

    # 3. Basic completion - Paris sanity
    Write-Host "[3/4] Basic completion - capital of France ..."
    try {
        $body = @{
            model = $MODEL
            messages = @(@{role="user"; content="What is the capital of France? Reply in one short sentence."})
            max_tokens = 30
            temperature = 0.6
            chat_template_kwargs = @{enable_thinking = $false}
        } | ConvertTo-Json -Depth 4
        $resp = Invoke-WebRequest -Uri "$URL/v1/chat/completions" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 30
        $content = $resp.Content | ConvertFrom-Json
        $reply = $content.choices[0].message.content
        if ($reply -match "(?i)Paris") {
            Pass-Check "reply contains 'Paris': $($reply.Substring(0, [Math]::Min(70, $reply.Length)))..."
        } else {
            Fail-Check "reply didn't mention Paris: $($reply.Substring(0, [Math]::Min(80, $reply.Length)))" "Model may be loading badly or using wrong chat template. Check docker logs ${CONTAINER}."
        }
    } catch {
        Fail-Check "completion request failed" "Check docker logs ${CONTAINER}"
    }

    # 4. Tool calling end-to-end
    Write-Host "[4/4] Tool calling - model should populate tool_calls[] ..."
    try {
        $toolBody = @{
            model = $MODEL
            messages = @(@{role="user"; content="What is the weather in San Francisco right now? Use the get_weather tool."})
            tools = @(
                @{
                    type = "function"
                    function = @{
                        name = "get_weather"
                        description = "Get the current weather for a given city."
                        parameters = @{
                            type = "object"
                            properties = @{
                                city = @{type = "string"; description = "City name"}
                                units = @{type = "string"; enum = @("celsius", "fahrenheit")}
                            }
                            required = @("city")
                        }
                    }
                }
            )
            tool_choice = "auto"
            max_tokens = 200
            temperature = 0.3
            chat_template_kwargs = @{enable_thinking = $false}
        } | ConvertTo-Json -Depth 6
        $toolResp = Invoke-WebRequest -Uri "$URL/v1/chat/completions" -Method POST -ContentType "application/json" -Body $toolBody -UseBasicParsing -TimeoutSec 60
        $toolJson = $toolResp.Content | ConvertFrom-Json
        $tc = $toolJson.choices[0].message.tool_calls

        if ($null -eq $tc) {
            $content = $toolJson.choices[0].message.content
            if ($content -match "<\?") {
                Fail-Check "model emitted inline text (tool_calls[] is empty)" "Genesis Patch 12 (Qwen3 tool_call fix) did not apply. Re-check the container logs and pin the image digest. README section Troubleshooting has the full chain."
            } else {
                Fail-Check "model answered without invoking the tool" "May be a model-behavior issue (it chose not to call) rather than a patch issue. Try rephrasing the prompt or lowering temperature. Raw content: $($content.Substring(0, [Math]::Min(200, $content.Length)))"
            }
        } elseif ($tc -is [Array] -and $tc.Count -gt 0) {
            $firstFunc = $tc[0].function
            if ($firstFunc.name -eq "get_weather") {
                Pass-Check "tool_calls[] populated, includes get_weather:"
                $tc | ConvertTo-Json -Depth 3 | ForEach-Object { Write-Host "      $_" }
            } else {
                Fail-Check "unexpected tool_calls structure" "Raw: $($tc | ConvertTo-Json -Depth 2)"
            }
        } else {
            Fail-Check "unexpected tool_calls structure" "Raw: $($tc | ConvertTo-Json -Depth 2)"
        }
    } catch {
        Fail-Check "tool-call request failed" "Check docker logs ${CONTAINER}"
    }

    Write-Host ""
    Write-Host "All checks passed. Stack is ready for use."
}

if ($Watch) {
    while ($true) {
        Clear-Host
        Run-Verify
        Write-Host ""
        Write-Host "Refresh every 5s - Ctrl-C to stop"
        Start-Sleep -Seconds 5
    }
} else {
    Run-Verify
}
