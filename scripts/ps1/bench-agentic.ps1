# Requires -Version 5.1
#
# bench-agentic.ps1 - Agentic prefill stress benchmark
#
# Simulates a multi-turn coding-agent session and measures TTFT + decode TPS
# as context accumulates over N turns of tool calls.
#
# Usage:
#   .\bench-agentic.ps1
#   $env:SESSIONS=3; .\bench-agentic.ps1
#   $env:TURNS=10; .\bench-agentic.ps1
#   $env:QUIET=1; .\bench-agentic.ps1

param(
    [string]$Url,
    [string]$Model,
    [string]$Container,
    [int]$Sessions,
    [int]$Turns,
    [int]$Quiet
)

$ErrorActionPreference = "Stop"
$ScriptName = "bench-agentic"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

# Defaults
if (-not $Url) { $Url = "http://localhost:8010" }
if (-not $Sessions) { $Sessions = 2 }
if (-not $Turns) { $Turns = 12 }
if (-not $Quiet) { $Quiet = 0 }

$FixturePath = Join-Path $PSScriptRoot "fixtures\agentic-bench-fixture.json"
if (-not (Test-Path $FixturePath)) {
    Write-Error "fixture not found: $FixturePath"
    exit 1
}

# Auto-detect endpoint
if ($Url -eq "http://localhost:8010") {
    try {
        $models = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($models -and $models.data -and $models.data.Count -gt 0) {
            $Url = $Url  # keep it
        }
    } catch { }
}

# Auto-detect model
if (-not $Model) {
    if ($env:MODEL) {
        $Model = $env:MODEL
    } else {
        try {
            $models = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($models -and $models.data -and $models.data.Count -gt 0) {
                $Model = $models.data[0].id
            }
        } catch { }
    }
}

if (-not $Model) {
    Write-Error "could not detect model name from $Url/v1/models - set MODEL=<name>"
    exit 1
}

Write-Host "[$ScriptName] URL=$Url MODEL=$Model SESSIONS=$Sessions TURNS=$Turns"

# Run the Python core
$FixtureJson = Get-Content $FixturePath -Raw -Encoding UTF8
$FixtureData = $FixtureJson | ConvertFrom-Json

# Handle both array format and object.with.turns format
if ($FixtureData.PSObject.Properties['turns']) {
    $FixtureData = $FixtureData.turns
}

# Cap to requested turns
$FixtureData = $FixtureData[0..[math]::Min($Turns - 1, $FixtureData.Count - 1)]

$SystemPrompt = @"
You are an autonomous coding assistant working inside a Python repository.
The user is investigating a performance regression. When file contents,
search results, or command output would materially change your answer,
call the appropriate tool - don't speculate. After each tool call,
briefly state what you learned and what your next planned step is.
Keep responses concise (under 100 words); defer to tools for raw data.

Repository layout:
  scripts/         - bench, verify, soak, launch helper scripts
  models/          - per-model compose configs + patches
  docs/            - architecture and cliff notes
  BENCHMARKS.md    - measured performance numbers
  CHANGELOG.md     - version history
"@

$Tools = @(
    @{ type = "function"; "function" = @{
        name = "Read"; description = "Read a UTF-8 file from the repository."
        parameters = @{ type = "object"; properties = @{ path = @{ type = "string" } } }
    }},
    @{ type = "function"; "function" = @{
        name = "Bash"; description = "Execute a shell command and return stdout+stderr."
        parameters = @{ type = "object"; properties = @{ command = @{ type = "string" } } }
    }},
    @{ type = "function"; "function" = @{
        name = "Edit"; description = "Apply a string replacement edit to a file."
        parameters = @{ type = "object"; properties = @{ path = @{ type = "string" }; pattern = @{ type = "string" } } }
    }},
    @{ type = "function"; "function" = @{
        name = "Write"; description = "Write or overwrite a file."
        parameters = @{ type = "object"; properties = @{ path = @{ type = "string" } } }
    }},
    @{ type = "function"; "function" = @{
        name = "Grep"; description = "Search for a regex pattern across the codebase."
        parameters = @{ type = "object"; properties = @{ pattern = @{ type = "string" } } }
    }},
    @{ type = "function"; "function" = @{
        name = "LS"; description = "List files in a directory."
        parameters = @{ type = "object"; properties = @{ path = @{ type = "string" } } }
    }},
    @{ type = "function"; "function" = @{
        name = "TodoRead"; description = "Read the current task/todo list."
        parameters = @{ type = "object"; properties = @{} }
    }},
    @{ type = "function"; "function" = @{
        name = "TodoWrite"; description = "Create or update a task/todo list."
        parameters = @{ type = "object"; properties = @{} }
    }},
    @{ type = "function"; "function" = @{
        name = "WebSearch"; description = "Search the web for information."
        parameters = @{ type = "object"; properties = @{ query = @{ type = "string" } } }
    }},
    @{ type = "function"; "function" = @{
        name = "WebFetch"; description = "Fetch a URL and return the HTML/text."
        parameters = @{ type = "object"; properties = @{ url = @{ type = "string" } } }
    }}
)

function Run-Turn {
    param($Messages, $FixtureTurn, $SessionId, $TurnIdx)

    $UserMsg = $FixtureTurn.user_msg
    $ToolResult = $FixtureTurn.tool_result

    $Messages += @{ role = "user"; content = $UserMsg }

    $Body = @{
        model = $Model
        messages = $Messages
        tools = $Tools
        tool_choice = "required"
        max_tokens = 600
        temperature = 0.3
        stream = $true
        stream_options = @{ include_usage = $true }
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 10

    $tSend = Get-Date
    $ttft = $null
    $completionTokens = 0
    $promptTokens = 0
    $contentParts = @()
    $toolCallsAcc = @{}

    $request = [System.Net.WebRequest]::Create("$Url/v1/chat/completions")
    $request.Method = "POST"
    $request.ContentType = "application/json"
    $request.Timeout = 600000

    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($Body)
    $request.ContentLength = $bodyBytes.Length
    $requestStream = $request.GetRequestStream()
    $requestStream.Write($bodyBytes, 0, $bodyBytes.Length)
    $requestStream.Close()

    try {
        $response = $request.GetResponse()
        $reader = New-Object System.IO.StreamReader $response.GetResponseStream()
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine().Trim()
            if (-not $line.StartsWith("data: ")) { continue }
            $payload = $line.Substring(6)
            if ($payload -eq '[DONE]') { break }
            try {
                $chunk = $payload | ConvertFrom-Json
            } catch { continue }

            $choices = $chunk.choices
            if ($choices) {
                $delta = $choices[0].delta
                if (-not $ttft -and ($delta.content -or $delta.tool_calls)) {
                    $ttft = ((Get-Date) - $tSend).TotalSeconds
                }
                if ($delta.content) { $contentParts += $delta.content }
                if ($delta.tool_calls) {
                    foreach ($tc in $delta.tool_calls) {
                        $idx = if ($tc.index) { $tc.index } else { 0 }
                        $slot = $toolCallsAcc[$idx]
                        if (-not $slot) { $slot = @{ id = ""; name = ""; args = "" }; $toolCallsAcc[$idx] = $slot }
                        if ($tc.id) { $slot.id = $tc.id }
                        if ($tc.function.name) { $slot.name = $tc.function.name }
                        if ($tc.function.arguments) { $slot.args += $tc.function.arguments }
                    }
                }
            }
            if ($chunk.usage) {
                $completionTokens = if ($null -eq $chunk.usage.completion_tokens) { $completionTokens } else { $chunk.usage.completion_tokens }
                $promptTokens = if ($null -eq $chunk.usage.prompt_tokens) { $promptTokens } else { $chunk.usage.prompt_tokens }
            }
        }
        $reader.Close()
        $response.Close()
    } catch {
        Write-Warning "Turn $TurnIdx error: $($_.Exception.Message)"
        return @{ ttft_ms = 0; wall_ms = 0; decode_tps = 0; wall_tps = 0; prompt_tokens = 0; completion_tokens = 0; tool_calls = 0; tool_call_missed = $true }
    }

    $wall = ((Get-Date) - $tSend).TotalSeconds
    if (-not $ttft) { $ttft = $wall }

    # Reconstruct assistant message
    $toolCallsResponse = @()
    foreach ($idx in $toolCallsAcc.Keys | Sort-Object) {
        $slot = $toolCallsAcc[$idx]
        if ($slot.name) {
            $toolCallsResponse += @{
                id = if ($slot.id) { $slot.id } else { "call_t${TurnIdx}_s${SessionId}_$idx" }
                type = "function"
                "function" = @{ name = $slot.name; arguments = "{}" }
            }
        }
    }

    $toolCallMissed = ($toolCallsResponse.Count -eq 0)
    if ($toolCallMissed) {
        $toolCallsResponse = @(@{
            id = "call_t${TurnIdx}_s${SessionId}_synthetic"
            type = "function"
            "function" = @{ name = "Read"; arguments = "{}" }
        })
    }

    $assistantMsg = @{ role = "assistant"; tool_calls = $toolCallsResponse }
    if ($contentParts.Count -gt 0) { $assistantMsg.content = ($contentParts -join "") }
    $Messages += $assistantMsg

    # Inject fixture tool result
    $Messages += @{
        role = "tool"
        tool_call_id = $toolCallsResponse[0].id
        content = $ToolResult
    }

    $decode_s = $wall - $ttft
    $degenerate = ($wall -le 0) -or ($decode_s -le 0) -or ($decode_s -lt 0.05 * $wall)

    if ($completionTokens -le 0) {
        $decodeTps = 0.0
    } elseif ($degenerate) {
        $decodeTps = $null
    } else {
        $decodeTps = $completionTokens / $decode_s
    }

    return @{
        ttft_ms = [math]::Round($ttft * 1000, 0)
        wall_ms = [math]::Round($wall * 1000, 0)
        decode_tps = $decodeTps
        wall_tps = if ($wall -gt 0) { $completionTokens / $wall } else { 0.0 }
        completion_tokens = $completionTokens
        prompt_tokens = $promptTokens
        tool_calls = $toolCallsResponse.Count
        tool_call_missed = $toolCallMissed
    }
}

# Run sessions
$PerTurnMetrics = @()
$ToolCallMisses = 0

for ($session = 1; $session -le $Sessions; $session++) {
    Write-Host ""
    Write-Host ("=" * 72)
    $totalChars = ($FixtureData | ForEach-Object { $_.chars }).Sum
    Write-Host "SESSION $session/$Sessions - $Turns turns, context grows to ~$([math]::Floor($totalChars/4)) tokens"
    Write-Host ("=" * 72)

    $Messages = @(@{ role = "system"; content = $SystemPrompt })

    $turnHeader = "  {0,-5} {1,10} {2,9} {3,11} {4,13}" -f "Turn", "Prompt tok", "TTFT ms", "Decode TPS", "Result chars"
    $turnDash = "  {0,-5} {1,-10} {2,-9} {3,-11} {4,-13}" -f "-----", "----------", "---------", "-----------", "-------------"
    Write-Host $turnHeader
    Write-Host $turnDash

    for ($turnIdx = 0; $turnIdx -lt $Turns; $turnIdx++) {
        $fixtureTurn = $FixtureData[$turnIdx]
        try {
            $m = Run-Turn -Messages $Messages -FixtureTurn $fixtureTurn -SessionId $session -TurnIdx $turnIdx
            $PerTurnMetrics += @{ turn = $turnIdx; metrics = $m }

            if ($m.tool_call_missed) { $ToolCallMisses++ }

            if (-not $Quiet) {
                $miss = if ($m.tool_call_missed) { "  WARNING tool-call miss (synthetic result injected)" } else { "" }
                $dcol = if ($null -eq $m.decode_tps) { "n/a" } else { "{0,11}" -f [math]::Round($m.decode_tps, 1) }
                $line = "  {0,-5} {1,10} {2,9} {3} {4,13}{5}" -f ($turnIdx+1), $m.prompt_tokens, $m.ttft_ms, $dcol, $m.completion_tokens, $miss
                Write-Host $line
            }
        } catch {
            Write-Host "  turn $($turnIdx+1): FAIL - $($_.Exception.Message)"
            break
        }
    }
}

# Summary
Write-Host ""
Write-Host ("=" * 72)
Write-Host "SUMMARY - multi-turn prefill stress ($Sessions session(s) x $Turns turns)"
Write-Host ("=" * 72)

if ($ToolCallMisses) {
    $turnsRun = $PerTurnMetrics.Count
    Write-Host "  tool-call misses: $ToolCallMisses/$turnsRun turns"
}

$summaryHeader = "  {0,-5} {1,10} {2,9} {3,6} {4,11}  {5}" -f "Turn", "Prompt tok", "TTFT ms", "s ms", "Decode TPS", "Notes"
$summaryDash = "  {0,-5} {1,-10} {2,-9} {3,-6} {4,-11}  {5}" -f "-----", "----------", "---------", "------", "-----------", "-----------------------------------"
Write-Host $summaryHeader
Write-Host $summaryDash

# Calculate per-turn stats
$activeTurns = $PerTurnMetrics.Count
for ($turnIdx = 0; $turnIdx -lt $activeTurns; $turnIdx++) {
    $turnData = $PerTurnMetrics[$turnIdx]
    $m = $turnData.metrics
    $meanTtft = $m.ttft_ms
    $meanPtok = $m.prompt_tokens
    $meanTps = $m.decode_tps

    if ($null -eq $meanTps) {
        $tcol = "n/a"
        $line = "  {0,-5} {1,10} {2,9} {3,6} {4}  {5}" -f ($turnIdx+1), $meanPtok, $meanTtft, 0, $tcol, "decode window unmeasurable"
        Write-Host $line
    } else {
        $formattedTps = [math]::Round($meanTps, 1).ToString().PadLeft(11)
        $line = "  {0,-5} {1,10} {2,9} {3,6} {4}  {5}" -f ($turnIdx+1), $meanPtok, $meanTtft, 0, $formattedTps, ""
        Write-Host $line
    }
}
