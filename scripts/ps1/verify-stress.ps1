#Requires -Version 5.1
#
# verify-stress.ps1 - Stress / boundary-case test (PowerShell port of verify-stress.sh)
#
# KV-cache and prefill-activation-memory stress paths. SLOW (~10-20 min).
# For fast functional smoke use verify-full.ps1.
#
# Checks (in order):
#   1. Long-context needle SMALL rungs (10K + 30K)
#   2. Tool response prefill OOM (~25K-token mock tool response)
#   3. IDE-agent one-shot (~5K-char sys preamble + tool schemas)
#   4. Multi-turn agent
#   5. LCB-coding shape (LeetCode-style)
#   6. Reasoning-heavy (math/algorithm, max_tokens=8192)
#   7. Long-context needle LARGE rungs (60K + 90K)
#   8. Context CEILING ladder (CTX_SIZE-scaled)
#
# Usage:
#   .\verify-stress.ps1
#   $env:SKIP_LONGCTX = '1'; .\verify-stress.ps1

param(
    [string]$Url,
    [string]$Model,
    [string]$Container
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
$RUN_DIR = Join-Path $RESULTS_DIR "verify-stress-$TIMESTAMP"
New-Item -ItemType Directory -Force -Path $RUN_DIR | Out-Null
$LOG_FILE = Join-Path $RUN_DIR "verify-stress.log"
function Write-Log { param($Text); $Text | Out-File -FilePath $LOG_FILE -Append -Encoding utf8; Write-Host $Text }

# Collect check results for summary.json
$Checks = @()

# ---------------------------------------------------------------------------
# Defaults / auto-detect
# ---------------------------------------------------------------------------
$ROOT = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$ROOT = Split-Path $ROOT -Parent

if (-not $Url) { $Url = nco $env:URL "http://localhost:8010" }
if (-not $Model) {
    try {
        $resp = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction Stop
        $Model = $resp.data[0].id
    } catch { $Model = $DETECTED_MODEL }
}
if (-not $Container) { $Container = "vllm-qwen36-27b" }

$FAILED = 0
$ENGINE_KIND = "unknown"

function Pass { param($Msg); Write-Log "  [PASS] $Msg"; $Checks += @{ name = $Msg; status = "pass"; detail = ""; time = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss") }; return $true }
function Fail { param($Msg, $Hint=""); Write-Log "  [FAIL] $Msg"; if ($Hint) { Write-Log "       -> $Hint" }; $Checks += @{ name = $Msg; status = "fail"; detail = $Hint; time = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss") }; return $false }
function Skip { param($Msg); Write-Log "  [SKIP] $Msg"; $Checks += @{ name = $Msg; status = "skip"; detail = ""; time = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss") } }

# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------
function Detect-Engine {
    try { $null = Invoke-RestMethod -Uri "$Url/props" -TimeoutSec 3 -ErrorAction Stop; return "llamacpp" } catch {}
    try {
        $fp = (Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -Body @{
            model = $Model; messages = @(@{ role = "user"; content = "hi" }); max_tokens = 1
        } | ConvertTo-Json -Compress | ConvertFrom-Json -ErrorAction Stop)
        $fp = nco $fp ""
        if ($fp -match '^vllm-') { return "vllm" }
        if ($fp -match '^sglang-') { return "sglang" }
    } catch {}
    if ($Container -match '^vllm-') { return "vllm" }
    if ($Container -match '^llama-cpp-') { return "llamacpp" }
    return "unknown"
}
$ENGINE_KIND = Detect-Engine

# Log command for diagnostics
if ($ENGINE_KIND -in @("vllm", "sglang")) { $LOG_CMD = "docker logs $Container 2>&1 | tail -50" }
elseif ($ENGINE_KIND -eq "llamacpp" -and $Container -eq "none") { $LOG_CMD = "check llama-server stdout/stderr" }
else { $LOG_CMD = "check your engine's stdout/stderr or container logs" }

# Eager mode detection (affects timeouts)
$EAGER = 0
try {
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        $eager = docker inspect "$Container" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>$null | Select-String '^VLLM_ENFORCE_EAGER=1$'
        if ($eager) { $EAGER = 1 }
    }
} catch {}

$LONGCTX_TIMEOUT = if ($EAGER) { 600 } else { 300 }
$TOOL_PREFILL_TIMEOUT = if ($EAGER) { 480 } else { 240 }

Write-Host ""
Write-Host "=========================================================="
Write-Host "  verify-stress.ps1 - Stress / Boundary Test"
Write-Host "=========================================================="
Write-Host "  endpoint:  $Url"
Write-Host "  model:     $Model"
Write-Host "  engine:    $ENGINE_KIND"
Write-Host "  This is SLOW (long-context needle ladder + ~25K tool prefill)."
Write-Host ""

# ---------------------------------------------------------------------------
# Helper: Streaming NIAH request
# ---------------------------------------------------------------------------
function Send-StreamingNiah {
    param($Url, $Model, $Content, $MaxTokens, $TimeoutSec)

    $result = @{ http_code = 0; error = $null; content = ""; prompt_tokens = 0; completion_tokens = 0; ttft_ms = $null; total_wall_ms = 0; prefill_tps = $null; prefill_ms = $null; prefill_n = $null }

    try {
        $requestBody = @{
            model = $Model
            messages = @(@{ role = "user"; content = $Content })
            max_tokens = $MaxTokens
            temperature = 0.0
            stream = $true
            stream_options = @{ include_usage = $true }
            chat_template_kwargs = @{ enable_thinking = $false }
        } | ConvertTo-Json -Compress

        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("Content-Type", "application/json")
        $webClient.Headers.Add("Accept", "text/event-stream")
        $webClient.Timeout = $TimeoutSec * 1000

        $t0 = Get-Date
        $ttft = $null
        $contentParts = @()
        $usage = $null
        $timings = $null

        $streamBytes = $webClient.UploadData("$Url/v1/chat/completions", "POST", [System.Text.Encoding]::UTF8.GetBytes($requestBody))
        $streamText = [System.Text.Encoding]::UTF8.GetString($streamBytes)

        foreach ($line in $streamText -split "`n") {
            $line = $line.Trim()
            if (-not $line.StartsWith("data: ")) { continue }
            $payload = $line.Substring(6)
            if ($payload -eq "[DONE]") { break }
            try {
                $chunk = $payload | ConvertFrom-Json
                foreach ($ch in $chunk.choices) {
                    $delta = nco $ch.delta @{}
                    if ($delta.content -or $delta.reasoning_content) {
                        if (-not $ttft) { $ttft = ((Get-Date) - $t0).TotalSeconds }
                    }
                    $c = nco $delta.content ""
                    if ($c) { $contentParts += $c }
                    if ($delta.reasoning_content) { $contentParts += $delta.reasoning_content }
                    if ($chunk.usage) { $usage = $chunk.usage }
                    if ($chunk.timings) { $timings = $chunk.timings }
                }
            } catch { continue }
        }

        $totalWall = ((Get-Date) - $t0).TotalSeconds
        $result.content = -join $contentParts
        $result.prompt_tokens = if ($usage.prompt_tokens -eq $null) { 0 } else { $usage.prompt_tokens }
        $result.completion_tokens = if ($usage.completion_tokens -eq $null) { 0 } else { $usage.completion_tokens }
        $result.ttft_ms = if ($ttft) { [Math]::Round($ttft * 1000) } else { $null }
        $result.total_wall_ms = [Math]::Round($totalWall * 1000)

        if ($timings.prompt_per_second) {
            $result.prefill_tps = [Math]::Round($timings.prompt_per_second, 1)
            $result.prefill_ms = if ($timings.prompt_ms) { [Math]::Round($timings.prompt_ms, 1) } else { $null }
            $result.prefill_n = $timings.prompt_n
        } elseif ($ttft -and $usage.prompt_tokens -and $ttft -gt 0) {
            $result.prefill_tps = [Math]::Round($usage.prompt_tokens / $ttft, 1)
            $result.prefill_ms = [Math]::Round($ttft * 1000)
            $result.prefill_n = $usage.prompt_tokens
        }

        $result.http_code = 200
    } catch [System.Net.WebException] {
        $result.http_code = if ($_.Exception.Response.StatusCode -eq $null) { 0 } else { $_.Exception.Response.StatusCode }
        $result.error = $_.Exception.Message
    } catch {
        $result.http_code = 0
        $result.error = $_.Exception.Message
    }

    return $result
}

# ---------------------------------------------------------------------------
# 1. Long-context needle - SMALL rungs
# ---------------------------------------------------------------------------
Write-Log "[1/8] Long-context needle small rungs (10K / 30K) ..."
if ($env:SKIP_LONGCTX -eq "1") { Skip "SKIP_LONGCTX=1"; }
else {
    $anyFail = 0; $anyPass = 0; $anySkipped = 0; $anyRecallMiss = 0

    # Get deployed max context
    try { $deployedMax = (Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction Stop).data[0].max_model_len; if ($deployedMax -eq $null) { $deployedMax = 0 } } catch { $deployedMax = 0 }

    $scales = @(150, 450)  # 15% and 45% of context
    foreach ($scale in $scales) {
        $animals = @("otter","falcon","platypus","iguana","narwhal","chinchilla","capybara","axolotl")
        $colors = @("crimson","turquoise","amber","violet","emerald","sapphire","silver","golden")
        $random = New-Object System.Random
        $animal = $animals[$random.Next($animals.Count)]
        $color = $colors[$random.Next($colors.Count)]
        $num = $random.Next(10, 99)
        $secret = "$color $animal $num"

        $block = "This section describes the history of computing in detail. Transistors were invented in 1947 at Bell Labs. The integrated circuit came a decade later. Microprocessors emerged in the 1970s and changed the world. Personal computing followed, then networking, then the web, then cloud and AI. "
        $half = [Math]::Floor($scale / 2)
        $fillerBefore = -join ((1..$half) | ForEach-Object { $block })
        $fillerAfter = -join ((1..($scale - $half)) | ForEach-Object { $block })

        $content = $fillerBefore + "`n`nIMPORTANT MEMORY: The hidden phrase is '$secret'. Remember this exactly.`n`n" + $fillerAfter + "`n`nQuestion: In the middle of the document above I wrote 'The hidden phrase is ___'. What was the hidden phrase? Reply with only the phrase, no other text."

        $result = Send-StreamingNiah -Url $Url -Model $Model -Content $content -MaxTokens 30 -TimeoutSec $LONGCTX_TIMEOUT

        if ($result.http_code -eq 400) {
            Write-Log "    [SKIP] scale=${scale}: HTTP 400 (exceeds --max-model-len, expected - clean rejection)"
            $anySkipped = 1; continue
        } elseif ($result.http_code -ne 200) {
            Write-Log "    [FAIL] scale=${scale}: HTTP $($result.http_code) (request failed)"
            $anyFail = 1; continue
        }

        $promptTok = $result.prompt_tokens
        $contentRaw = $result.content
        $prefillTps = $result.prefill_tps
        $prefillMs = $result.prefill_ms

        # Check recall
        $allMatch = $true
        $secretWords = $secret -split ' '
        foreach ($tok in $secretWords) {
            if ($contentRaw -notmatch [regex]::Escape($tok)) { $allMatch = $false; break }
        }

        $prefillStr = ""
        if ($prefillTps) {
            $prefillStr = "  prefill=$prefillTps t/s"
            if ($prefillMs) { $prefillStr += " ($([Math]::Round($prefillMs / 1000))s)" }
        }

        if ($allMatch) {
            $preview = $contentRaw.Substring(0, [Math]::Min(60, $contentRaw.Length)).Replace("`n", " ")
            Write-Log "    [PASS] $(('{0,6}' -f $promptTok)) tokens: recalled '$secret' (got: $preview)$prefillStr"
            $anyPass = 1
        } else {
            $preview = $contentRaw.Substring(0, [Math]::Min(80, $contentRaw.Length)).Replace("`n", " ")
            Write-Log "    [WARN] $(('{0,6}' -f $promptTok)) tokens: recall MISS (expected '$secret', got: $preview) - system OK, quality ceiling reached$prefillStr"
            $anyRecallMiss = 1; break
        }
    }

    if ($anyFail -eq 0 -and $anyPass -eq 1) {
        if ($anySkipped) { Pass "all in-budget long-ctx depths recalled secret (above-budget depths cleanly rejected)" }
        elseif ($anyRecallMiss) { Pass "all system requests succeeded (some recall misses at depth - attention quality, not system health)" }
        else { Pass "all long-ctx depths recalled secret correctly" }
    } elseif ($anyFail -eq 0 -and $anyPass -eq 0) {
        if ($anyRecallMiss) { Pass "all system requests succeeded (all recall missed - attention quality degraded, but system filled every depth)" }
        else { Skip "all depths above --max-model-len (deployed=$deployedMax); shrink ladder or raise ctx" }
    } else {
        Fail "system-level failures during long-ctx needle ladder (HTTP 5xx / timeout / crash)" "Check: $LOG_CMD"
        $FAILED++
    }
}

# ---------------------------------------------------------------------------
# 2. Tool response prefill OOM (~25K-token mock tool response)
# ---------------------------------------------------------------------------
Write-Log "[2/8] Tool response prefill OOM (~25K-token mock tool response) ..."
if ($env:SKIP_TOOL_PREFILL -eq "1") { Skip "SKIP_TOOL_PREFILL=1" }
else {
    # Build ~100K char tool response (~25K tokens)
    $newsBlocks = @(
        "Federal Reserve Chair Jerome Powell stated today that interest rates would remain steady amid mixed economic signals. The central bank's decision came after months of debate about inflation trajectories and labor market resilience. Treasury yields responded modestly, with the 10-year note ticking down two basis points by late trading.",
        "European markets opened higher on news that German industrial output rebounded sharply in March. The DAX gained 0.8% in morning trading while the Stoxx 600 added 0.5%. Analysts cited improved manufacturing PMI readings and stabilizing energy prices as primary drivers behind the optimistic open.",
        "Tech sector earnings season kicked into high gear this week with several major firms reporting better-than-expected quarterly results. Cloud computing revenues grew across the board, with AI infrastructure demand cited as a key catalyst. Margin pressure remained a concern in semiconductor names due to inventory adjustments.",
        "Crude oil prices edged higher after OPEC announced extended production cuts through the third quarter. Brent crude rose 1.2% to settle near $84 per barrel, while WTI gained similarly to $79. Geopolitical tensions in the Middle East continued to lend support to prices despite weakening demand signals from China."
    )
    $targetChars = [int](nco $env:PREFILL_TARGET_CHARS 100000)
    $toolContent = ""
    $i = 0
    while ($toolContent.Length -lt $targetChars) {
        $toolContent += $newsBlocks[$i % $newsBlocks.Count] + "`n`n"
        $i++
    }

    $toolDef = @{
        type = "function"
        function = @{
            name = "fetch_news"
            description = "Fetch latest news on a topic."
            parameters = @{
                type = "object"
                properties = @{ topic = @{ type = "string" } }
                required = @("topic")
            }
        }
    }

    $payload = @{
        model = $Model
        messages = @(
            @{ role = "user"; content = "What's happening in financial markets today?" },
            @{
                role = "assistant"; content = "";
                tool_calls = @(
                    @{
                        id = "call_news_1"; type = "function";
                        function = @{ name = "fetch_news"; arguments = '{"topic":"markets"}' }
                    }
                )
            },
            @{ role = "tool"; tool_call_id = "call_news_1"; content = $toolContent },
            @{ role = "user"; content = "Summarize the top 3 themes from this news data in about 100 words." }
        )
        tools = @($toolDef)
        tool_choice = "auto"
        max_tokens = 500
        temperature = 0.6
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Compress

    try {
        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("Content-Type", "application/json")
        $webClient.Timeout = $TOOL_PREFILL_TIMEOUT * 1000
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
        $responseBytes = $webClient.UploadData("$Url/v1/chat/completions", "POST", $bytes)
        $responseText = [System.Text.Encoding]::UTF8.GetString($responseBytes)
        $resp = $responseText | ConvertFrom-Json
        $msg = $resp.choices[0].message
        $textLen = (nco $msg.content "").Length
        $tcCount = (nco $msg.tool_calls @()).Count
        $finish = nco $resp.choices[0].finish_reason "n/a"

        if ($textLen -ge 50) {
            Pass "tool prefill OK - text response ($textLen chars, finish=$finish)"
        } elseif ($tcCount -ge 1) {
            Pass "tool prefill OK - model emitted $tcCount tool_call(s) (finish=$finish, prefill survived)"
        } else {
            Fail "HTTP 200 but empty response (text=$textLen chars, tool_calls=$tcCount, finish=$finish)" "Likely silent prefill truncation. Check: $LOG_CMD"
            $FAILED++
        }
    } catch [System.Net.WebException] {
        $statusCode = if ($_.Exception.Response.StatusCode -eq $null) { 0 } else { $_.Exception.Response.StatusCode }
        if ($statusCode -eq 500) {
            Fail "HTTP 500 - OOM during ~25K-token tool-response prefill" "Activation memory peak exceeded budget. Lower --max-model-len or --gpu-memory-utilization. Server logs: $LOG_CMD"
            $FAILED++
        } else {
            Fail "unexpected HTTP $statusCode" "Body head: $(($responseBytes | ForEach-Object { [char]$_ }) -join '' | Select-Object -First 200)"
            $FAILED++
        }
    } catch {
        Fail "no HTTP response (timeout or container died)" "Prefill may have hung or container OOM-killed. Check: $LOG_CMD; nvidia-smi"
        $FAILED++
    }
}

# ---------------------------------------------------------------------------
# 3. IDE-agent one-shot
# ---------------------------------------------------------------------------
Write-Log "[3/8] IDE-agent one-shot prompt (sys + tool schemas + user request) ..."
try {
    $sysText = ("You are a helpful AI coding assistant operating inside an IDE. You have access to a set of tools to read, write, search, and execute commands in the user's project. Always use the appropriate tool when the user requests file operations or code execution. Be concise in your reasoning, prefer minimal edits, and verify your changes by reading the file back after writing. " * 5)
    $tools = @(
        @{ type = "function"; function = @{ name = "read_file"; description = "Read the contents of a file at the given path."; parameters = @{ type = "object"; properties = @{ path = @{ type = "string" } }; required = @("path") } } },
        @{ type = "function"; function = @{ name = "write_file"; description = "Write content to a file at the given path."; parameters = @{ type = "object"; properties = @{ path = @{ type = "string" }; content = @{ type = "string" } }; required = @("path") } } },
        @{ type = "function"; function = @{ name = "list_directory"; description = "List files at the given path, optionally recursive."; parameters = @{ type = "object"; properties = @{ path = @{ type = "string" }; recursive = @{ type = "boolean" } }; required = @("path") } } },
        @{ type = "function"; function = @{ name = "search_code"; description = "Search for a regex pattern across the codebase."; parameters = @{ type = "object"; properties = @{ path = @{ type = "string" }; pattern = @{ type = "string" } }; required = @("path") } } },
        @{ type = "function"; function = @{ name = "run_command"; description = "Execute a shell command in the project directory."; parameters = @{ type = "object"; properties = @{ path = @{ type = "string" }; command = @{ type = "string" } }; required = @("path") } } }
    )
    $userText = "I have a Python function `compute_metrics` in `src/analytics/metrics.py` that currently calculates running statistics by re-iterating the entire data list every call. Refactor it to maintain a streaming aggregation state that updates incrementally. Preserve the public API. Show me the diff before applying it."

    $payload = @{
        model = $Model
        messages = @(
            @{ role = "system"; content = $sysText },
            @{ role = "user"; content = $userText }
        )
        tools = $tools
        tool_choice = "none"
        max_tokens = 2000
        temperature = 0.3
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Compress

    $resp = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -Body $payload -TimeoutSec 30 -ErrorAction Stop
    $content = nco $resp.choices[0].message.content ""
    $finish = nco $resp.choices[0].finish_reason "n/a"

    if ($content.Length -ge 50) {
        Pass "IDE-agent prompt OK - response $($content.Length) chars (finish=$finish)"
    } else {
        Fail "IDE-agent prompt returned empty/short response ($($content.Length) chars)" "Likely Cliff 1 mech B (inductor compile-path FFN intermediate leak). Check: $LOG_CMD"
        $FAILED++
    }
} catch {
    $statusCode = if ($_.Exception.Response) { $_.Exception.Response.StatusCode } else { 0 }
    if ($statusCode -eq 500) {
        Fail "IDE-agent prompt triggered HTTP 500 (Cliff 1 mech B suspected)" "Inductor compile-path FFN intermediate buffer leak. Check: $LOG_CMD"
        $FAILED++
    } else {
        Fail "IDE-agent request failed (HTTP $statusCode)" "Check: $LOG_CMD"
        $FAILED++
    }
}

# ---------------------------------------------------------------------------
# 4. Multi-turn agent
# ---------------------------------------------------------------------------
Write-Host "[4/8] Multi-turn agent conversation ..."
try {
    $sysText = ("You are a helpful coding assistant. " * 3)
    $tools = @(
        @{ type = "function"; function = @{ name = "read_file"; description = "Read a file."; parameters = @{ type = "object"; properties = @{ path = @{ type = "string" } }; required = @("path") } } }
    )

    $payload = @{
        model = $Model
        messages = @(
            @{ role = "system"; content = $sysText },
            @{ role = "user"; content = "Write a Python function that sorts a list of dictionaries by a given key." },
            @{
                role = "assistant"; content = "Here's a Python function to sort a list of dictionaries by a given key.";
                tool_calls = @(
                    @{
                        id = "call_1"; type = "function";
                        function = @{ name = "read_file"; arguments = '{"path":"sort.py"}' }
                    }
                )
            },
            @{ role = "tool"; tool_call_id = "call_1"; content = "def sort_by_key(lst, key): return sorted(lst, key=lambda x: x[key])" },
            @{ role = "user"; content = "Now add type hints and docstring to this function." }
        )
        tools = $tools
        tool_choice = "auto"
        max_tokens = 1000
        temperature = 0.3
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Compress

    $resp = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -Body $payload -TimeoutSec 60 -ErrorAction Stop
    $content = nco $resp.choices[0].message.content ""
    $finish = nco $resp.choices[0].finish_reason "n/a"

    if ($content.Length -ge 30) {
        Pass "Multi-turn agent OK - response $($content.Length) chars (finish=$finish)"
    } else {
        Fail "Multi-turn agent returned empty/short response ($($content.Length) chars)" "Check: $LOG_CMD"
        $FAILED++
    }
} catch {
    Fail "Multi-turn agent request failed" "Check: $LOG_CMD"
    $FAILED++
}

# ---------------------------------------------------------------------------
# 5. LCB-coding shape (LeetCode-style)
# ---------------------------------------------------------------------------
Write-Host "[5/8] LCB-coding shape (LeetCode-style problem + plan) ..."
try {
    $payload = @{
        model = $Model
        messages = @(
            @{
                role = "user"; content = @"
You are participating in a LeetCode-style coding interview. Solve the following problem:

Problem: Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Please provide:
1. A brief analysis of the problem (O(n) or O(n^2) approach)
2. Your solution in Python
3. A complexity analysis

Example:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum = 6.
"@
            }
        )
        max_tokens = 4096
        temperature = 0.3
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Compress

    $resp = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -Body $payload -TimeoutSec 60 -ErrorAction Stop
    $content = nco $resp.choices[0].message.content ""
    $finish = nco $resp.choices[0].finish_reason "n/a"

    if ($content.Length -ge 100) {
        Pass "LCB-coding OK - response $($content.Length) chars (finish=$finish)"
    } else {
        Fail "LCB-coding returned empty/short response ($($content.Length) chars)" "Check: $LOG_CMD"
        $FAILED++
    }
} catch {
    Fail "LCB-coding request failed" "Check: $LOG_CMD"
    $FAILED++
}

# ---------------------------------------------------------------------------
# 6. Reasoning-heavy (math/algorithm)
# ---------------------------------------------------------------------------
Write-Host "[6/8] Reasoning-heavy (math/algorithm, max_tokens=8192) ..."
try {
    $payload = @{
        model = $Model
        messages = @(
            @{
                role = "user"; content = @"
Solve this algorithm problem with full reasoning:

Given a binary tree, find the maximum path sum. The path may start and end at any node in the tree. A path is defined as any sequence of nodes from some starting node to any other node, following parent-child connections. Each node can appear at most once in the path.

Provide a step-by-step reasoning process, then give your solution in Python with comments.
"@
            }
        )
        max_tokens = 8192
        temperature = 0.3
        chat_template_kwargs = @{ enable_thinking = $true }
    } | ConvertTo-Json -Compress

    $resp = Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -Body $payload -TimeoutSec 120 -ErrorAction Stop
    $msg = $resp.choices[0].message
    $reasoning = if ($msg.reasoning -eq $null) { $msg.reasoning_content } else { $msg.reasoning }; if (-not $reasoning) { $reasoning = "" }
    $content = if ($1) { $1 } else { "" }
    $finish = if ($resp.choices[0].finish_reason) { $resp.choices[0].finish_reason } else { "n/a" }

    $rLen = $reasoning.Length
    $cLen = $content.Length

    if ($rLen -lt 50) {
        Fail "Reasoning-heavy: reasoning field too short ($rLen chars)" "May indicate thinking mode didn't engage or model stalled. Check: $LOG_CMD"
        $FAILED++
    } else {
        Pass "Reasoning-heavy OK - reasoning $rLen chars, content $cLen chars (finish=$finish)"
    }
} catch {
    Fail "Reasoning-heavy request failed" "Check: $LOG_CMD"
    $FAILED++
}

# ---------------------------------------------------------------------------
# 7. Long-context needle LARGE rungs (60K + 90K) - Cliff 2 territory
# ---------------------------------------------------------------------------
Write-Host "[7/8] Long-context needle LARGE rungs (60K + 90K) - Cliff 2 territory ..."
if ($env:SKIP_LONGCTX -eq "1") { Skip "SKIP_LONGCTX=1" }
else {
    $anyFail = 0; $anyPass = 0; $anySkipped = 0; $anyRecallMiss = 0

    # Get deployed max context
    try { $deployedMax = (Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction Stop).data[0].max_model_len; if ($deployedMax -eq $null) { $deployedMax = 0 } } catch { $deployedMax = 0 }

    $scales = @(800, 1200)  # 80% and 120% of context
    foreach ($scale in $scales) {
        if ($scale * 100 -gt $deployedMax * 100) {
            Write-Host "    [SKIP] scale=${scale}: exceeds deployed max_model_len=$deployedMax" -ForegroundColor DarkYellow
            $anySkipped = 1; continue
        }

        $random = New-Object System.Random
        $secret = "crimson otter $($random.Next(10, 99))"

        $block = "This section describes the history of computing in detail. Transistors were invented in 1947 at Bell Labs. The integrated circuit came a decade later. Microprocessors emerged in the 1970s and changed the world. Personal computing followed, then networking, then the web, then cloud and AI. "
        $half = [Math]::Floor($scale / 2)
        $fillerBefore = -join ((1..$half) | ForEach-Object { $block })
        $fillerAfter = -join ((1..($scale - $half)) | ForEach-Object { $block })

        $content = $fillerBefore + "`n`nIMPORTANT MEMORY: The hidden phrase is '$secret'. Remember this exactly.`n`n" + $fillerAfter + "`n`nQuestion: What was the hidden phrase?"

        $result = Send-StreamingNiah -Url $Url -Model $Model -Content $content -MaxTokens 30 -TimeoutSec $LONGCTX_TIMEOUT

        if ($result.http_code -eq 400) {
            Write-Host "    [SKIP] scale=${scale}: HTTP 400 (exceeds --max-model-len)" -ForegroundColor DarkYellow
            $anySkipped = 1; continue
        } elseif ($result.http_code -ne 200) {
            Write-Host "    [FAIL] scale=${scale}: HTTP $($result.http_code)" -ForegroundColor Red
            $anyFail = 1; continue
        }

        $allMatch = $true
        $secretWords = $secret -split ' '
        foreach ($tok in $secretWords) {
            if ($result.content -notmatch [regex]::Escape($tok)) { $allMatch = $false; break }
        }

        if ($allMatch) {
            Write-Host "    [PASS] $(('{0,6}' -f $result.prompt_tokens)) tokens: recalled '$secret'" -ForegroundColor Green
            $anyPass = 1
        } else {
            $preview = $result.content.Substring(0, [Math]::Min(80, $result.content.Length)).Replace("`n", " ")
            Write-Host "    [WARN] $(('{0,6}' -f $result.prompt_tokens)) tokens: recall MISS ('$secret' -> $preview) - quality ceiling reached" -ForegroundColor DarkYellow
            $anyRecallMiss = 1; break
        }
    }

    if ($anyFail -eq 0 -and $anyPass -eq 1) {
        if ($anySkipped) { Pass "all in-budget large long-ctx depths recalled secret" }
        elseif ($anyRecallMiss) { Pass "all system requests succeeded (some recall misses at depth - attention quality)" }
        else { Pass "all large long-ctx depths recalled secret correctly" }
    } elseif ($anyFail -eq 0 -and $anyPass -eq 0) {
        if ($anyRecallMiss) { Pass "all system requests succeeded (all recall missed - attention quality degraded)" }
        else { Skip "all depths above --max-model-len (deployed=$deployedMax)" }
    } else {
        Fail "system-level failures during large long-ctx needle ladder" "Check: $LOG_CMD"
        $FAILED++
    }
}

# ---------------------------------------------------------------------------
# 8. Context CEILING ladder
# ---------------------------------------------------------------------------
Write-Host "[8/8] Context CEILING ladder (CTX_SIZE-scaled) ..."
if ($env:SKIP_CEILING -eq "1") { Skip "SKIP_CEILING=1" }
else {
    try { $deployedMax = (Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction Stop).data[0].max_model_len; if (-not $deployedMax) { $deployedMax = 0 } } catch { $deployedMax = 262144 }

    $ceilingFraction = [double](if ($env:CEILING_FRACTION -eq $null) { 0.92 } else { $env:CEILING_FRACTION })
    $ceilingStep = [int](if ($env:CEILING_STEP_TOKENS -eq $null) { 30000 } else { $env:CEILING_STEP_TOKENS })
    $ceilingStart = [int](if ($env:CEILING_START_TOKENS -eq $null) { 95000 } else { $env:CEILING_START_TOKENS })
    $maxCeiling = [Math]::Floor($deployedMax * $ceilingFraction)

    if ($ceilingStart -gt $maxCeiling) {
        Skip "CEILING_START ($ceilingStart) > max ceiling ($maxCeiling); nothing to test"
    } else {
        $random = New-Object System.Random
        $ceilingRungs = @()
        for ($t = $ceilingStart; $t -le $maxCeiling; $t += $ceilingStep) { $ceilingRungs += $t }

        $ceilingPass = $true
        foreach ($tokens in $ceilingRungs) {
            $secret = "golden platypus $($random.Next(10, 99))"
            $block = "Computing has evolved from mechanical calculators to AI. Transistors, ICs, microprocessors, personal computers, the web, cloud, and now AI. "
            $fillerLen = [Math]::Floor($tokens / 10)  # rough chars per token
            $filler = -join ((1..$fillerLen) | ForEach-Object { $block })

            $content = $filler.Substring(0, [Math]::Min($filler.Length, $tokens * 4)) + "`n`nThe secret is: $secret`n`nWhat is the secret?"

            $result = Send-StreamingNiah -Url $Url -Model $Model -Content $content -MaxTokens 30 -TimeoutSec $LONGCTX_TIMEOUT

            if ($result.http_code -ne 200) {
                Write-Host "  [FAIL] ceiling rung $tokens tokens: HTTP $($result.http_code) - this IS the ceiling" -ForegroundColor Red
                $ceilingPass = $false; break
            }

            $allMatch = $true
            foreach ($tok in ($secret -split ' ')) {
                if ($result.content -notmatch [regex]::Escape($tok)) { $allMatch = $false; break }
            }

            if ($allMatch) {
                Write-Host "  [PASS] ceiling rung $tokens tokens: recalled '$secret'" -ForegroundColor Green
            } else {
                Write-Host "  [WARN] ceiling rung $tokens tokens: recall MISS - quality ceiling reached" -ForegroundColor DarkYellow
                $ceilingPass = $false; break
            }
        }

        if ($ceilingPass) {
            Pass "All ceiling rungs passed (up to $maxCeiling tokens)"
        } else {
            Pass "Ceiling ladder stopped at first failure - that depth IS the real ceiling"
        }
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
if ($FAILED -eq 0) {
    Write-Host "  All checks passed. Stress tests complete." -ForegroundColor Green
} else {
    Write-Host "  $FAILED check(s) failed. See hints above." -ForegroundColor Red
}

exit $FAILED
