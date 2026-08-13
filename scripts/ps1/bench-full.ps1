# Requires -Version 5.1
#
# Canonical bench against the running vLLM service.
#   - Runs both the canonical narrative AND code prompts in one invocation.
#   - 3 warmup + N measured runs per prompt (default 5 narrative + 5 code).
#   - per-run: wall time, TTFT (via streaming), completion tokens,
#     wall_TPS (= comp / wall), decode_TPS (= comp / (wall - TTFT))
#   - per-prompt summary: mean / std / CV for both TPS metrics + mean TTFT
#
# Usage:
#   scripts/bench.ps1
#   $env:ONLY="narr"; scripts/bench.ps1
#   $env:RUNS=10; scripts/bench.ps1
#   $env:FORCE_TOKENS=4000; scripts/bench.ps1   # fixed 4000-tok output
#
# Env vars:
#   URL                Endpoint. Default: http://localhost:8010
#   MODEL              Served model name. Default: auto-detected
#   CONTAINER          Container for log scraping. Default: vllm-qwen36-27b
#   RUNS               Measured runs per prompt. Default: 5
#   WARMUPS            Warm-up runs (shared across both). Default: 3
#   PROMPT_NARR        Override narrative prompt
#   PROMPT_CODE        Override code prompt
#   MAX_TOKENS_NARR    Default: 1000
#   MAX_TOKENS_CODE    Default: 800
#   ONLY               "narr", "code", or "both". Default: both
#   QUIET              Set to 1 to skip per-run lines (just print summary)
#   ENABLE_THINKING    Set to 1 to enable thinking mode. Default: 0
#   FORCE_TOKENS       Force exact output tokens. Default: 0 (model decides)
#   QUICK              Set to 1 for quick directional mode (1 warmup, 1 run, narr only)
#
# Exit: 0 on success, 1 on failure

param(
    [switch]$Help,
    [string]$Tag
)
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path | Select-String '^#( |$)' | ForEach-Object { $_.Line.Substring(2) }
    exit 0
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
$LogInfo = Init-Logging -ScriptName "bench-full" -Tag $Tag
$LogFilePath = $LogInfo.LogFile
$SummaryFile = (Join-Path $LogInfo.RunDir "summary.json")

# Override Write-Log to also handle NoNewline for compatibility
function Write-Log { param($Text, $NoNewline); if ($NoNewline) { Write-Host -NoNewline $Text } else { Write-Host $Text } ; $Text | Out-File -FilePath $LogFilePath -Append -Encoding utf8 }

$env:PYTHONUTF8 = "1"

# Auto-detect running container + port
$ROOT_DIR = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
$URL = $env:URL
$MODEL = $env:MODEL
$CONTAINER = $env:CONTAINER
$RUNS = [int]$env:RUNS
$WARMUPS = [int]$env:WARMUPS
$MAX_TOKENS_NARR = [int]$env:MAX_TOKENS_NARR
$MAX_TOKENS_CODE = [int]$env:MAX_TOKENS_CODE
$PROMPT_NARR = $env:PROMPT_NARR
$PROMPT_CODE = $env:PROMPT_CODE
$ONLY = $env:ONLY
$QUIET = [int]$env:QUIET
$ENABLE_THINKING = [int]$env:ENABLE_THINKING
$FORCE_TOKENS = [int]$env:FORCE_TOKENS
$QUICK = [int]$env:QUICK

# Defaults
if (-not $URL) { $URL = "http://localhost:8010" }
if (-not $MODEL) { $MODEL = $DETECTED_MODEL }
if (-not $CONTAINER) { $CONTAINER = "vllm-qwen36-27b" }
if (-not $RUNS) { $RUNS = 5 }
if (-not $WARMUPS) { $WARMUPS = 3 }
if (-not $MAX_TOKENS_NARR) { $MAX_TOKENS_NARR = 1000 }
if (-not $MAX_TOKENS_CODE) { $MAX_TOKENS_CODE = 800 }
if (-not $PROMPT_NARR) { $PROMPT_NARR = "Write a detailed 800-word essay explaining transformer attention." }
if (-not $PROMPT_CODE) { $PROMPT_CODE = "Write a Python implementation of quicksort with comments explaining each step." }
if (-not $ONLY) { $ONLY = "both" }
if (-not $QUIET) { $QUIET = 0 }
if (-not $ENABLE_THINKING) { $ENABLE_THINKING = 0 }
if (-not $FORCE_TOKENS) { $FORCE_TOKENS = 0 }
if (-not $QUICK) { $QUICK = 0 }

# Quick mode preset
if ($QUICK -eq 1) {
    if (-not $env:WARMUPS) { $WARMUPS = 1 }
    if (-not $env:RUNS) { $RUNS = 1 }
    if (-not $env:ONLY) { $ONLY = "narr" }
    if (-not $env:QUIET) { $QUIET = 1 }
    Write-Host "==============================================================================" -ForegroundColor Yellow
    Write-Host "QUICK MODE - NOT CANONICAL. Directional signal only." -ForegroundColor Yellow
    Write-Host "==============================================================================" -ForegroundColor Yellow
    Write-Host ""
}

# Check server is reachable
try {
    $null = Invoke-WebRequest -Uri "$URL/v1/models" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
} catch {
    Write-Host "ERROR: service not reachable at ${URL}/v1/models" -ForegroundColor Red
    Write-Host "  Start with: cd compose && docker compose up -d" -ForegroundColor Yellow
    exit 1
}

function Invoke-BenchRun {
    param(
        [string]$Label,
        [string]$Prompt,
        [int]$MaxTokens
    )

    Write-Host ""
    Write-Host "========== ${Label} (prompt=$($Prompt.Length) chars, max_tokens=$MaxTokens) ==========" -ForegroundColor Cyan

    # Warmup runs
    Write-Host "=== warmups ($WARMUPS) ==="
    $warmupResults = @()
    for ($i = 0; $i -lt $WARMUPS; $i++) {
        try {
            $result = Run-Request $Prompt $MaxTokens
            $line = Format-Result "warm-$($i+1)" $result.wall $result.ttft $result.completionTokens
            if ($QUIET -eq 0) { Write-Host $line }
            $warmupResults += $result
        } catch {
            Write-Host "  warm-$($i+1)  FAIL: $_" -ForegroundColor Red
        }
    }

    # Measured runs
    Write-Host ""
    Write-Host "=== measured ($RUNS) ==="
    $walls = @()
    $decodes = @()
    $ttfts = @()
    $toks = @()
    $degenCount = 0
    $errors = 0

    for ($i = 0; $i -lt $RUNS; $i++) {
        try {
            $result = Run-Request $Prompt $MaxTokens
            $line = Format-Result "run-$($i+1)" $result.wall $result.ttft $result.completionTokens
            if ($QUIET -eq 0) { Write-Host $line }

            # Progress to stderr (so stdout stays parseable)
            $rate = if ($result.decodeTPS -ne $null) {
                "{0:N2} decode TPS" -f $result.decodeTPS
            } else {
                "{0:N2} wall TPS (no decode window)" -f $result.wallTPS
            }
            $runLabel = "${Label} run $($i+1)/$RUNS"
            $toksInfo = "$($result.completionTokens) tok in $([math]::Round($result.wall,1))s"
            $perfInfo = "($rate, ttft $([math]::Round($result.ttft*1000))ms)"
            Write-Host ("[bench] " + $runLabel + ": " + $toksInfo + " " + $perfInfo) -ForegroundColor DarkGray

            $walls += $result.wallTPS
            $ttfts += $result.ttft
            $toks += $result.completionTokens

            if ($result.decodeTPS -eq $null) { $degenCount++ }
            else { $decodes += $result.decodeTPS }
        } catch {
            $errors++
            Write-Host "  run-$($i+1)  FAIL: $_" -ForegroundColor Red
        }
    }

    if ($errors -gt 0) {
        Write-Host "  WARNING: $errors run(s) failed" -ForegroundColor Yellow
    }

    if ($walls.Count -eq 0) {
        Write-Host "  No successful runs to summarize." -ForegroundColor Red
        return
    }

    # Summary
    $n = $walls.Count
    $meanWall = ($walls | Measure-Object -Average).Average
    $sumSqWall = 0
    foreach ($w in $walls) { $sumSqWall += [math]::Pow($w - $meanWall, 2) }
    $stdWall = if ($n -gt 1) { [math]::Sqrt($sumSqWall / $n) } else { 0 }
    $cvWall = if ($meanWall -gt 0) { $stdWall / $meanWall * 100 } else { 0 }

    if ($decodes.Count -gt 0) {
        $meanDecode = ($decodes | Measure-Object -Average).Average
    } else {
        $meanDecode = $null
    }
    $sumSqDecode = 0
    foreach ($d in $decodes) { $sumSqDecode += [math]::Pow($d - $meanDecode, 2) }
    $stdDecode = if ($decodes.Count -gt 1) { [math]::Sqrt($sumSqDecode / $decodes.Count) } else { 0 }
    $cvDecode = if ($null -ne $meanDecode -and $meanDecode -gt 0) { $stdDecode / $meanDecode * 100 } else { $null }

    $meanTTFT = ($ttfts | Measure-Object -Average).Average
    $minTTFT = ($ttfts | Measure-Object -Minimum).Minimum
    $maxTTFT = ($ttfts | Measure-Object -Maximum).Maximum

    Write-Host ""
    Write-Host "=== summary [$Label] (n=$n) ==="
    Write-Host "  wall_TPS       mean=$([math]::Round($meanWall,2))   std=$([math]::Round($stdWall,2))   CV=$([math]::Round($cvWall,1))%   min=$([math]::Round(($walls | Measure-Object -Minimum).Minimum,2))   max=$([math]::Round(($walls | Measure-Object -Maximum).Maximum,2))"
    if ($null -ne $meanDecode) {
        Write-Host "  decode_TPS     mean=$([math]::Round($meanDecode,2))   std=$([math]::Round($stdDecode,2))   CV=$([math]::Round($cvDecode,1))%   min=$([math]::Round(($decodes | Measure-Object -Minimum).Minimum,2))   max=$([math]::Round(($decodes | Measure-Object -Maximum).Maximum,2))"
    } else {
        Write-Host "  decode_TPS     n/a (no measurable decode window in any run)"
    }
    $sumSqTTFT = 0
    foreach ($t in $ttfts) { $sumSqTTFT += [math]::Pow($t - $meanTTFT, 2) }
    $ttftStd = if ($n -gt 1) { [math]::Sqrt($sumSqTTFT / $n) } else { 0 }
    Write-Host "  TTFT           mean=$([math]::Round($meanTTFT*1000,0))ms  std=$([math]::Round($ttftStd*1000,0))ms  min=$([math]::Round($minTTFT*1000,0))ms  max=$([math]::Round($maxTTFT*1000,0))ms"
}

function Run-Request {
    param(
        [string]$Prompt,
        [int]$MaxTokens
    )

    $mt = if ($FORCE_TOKENS -gt 0) { $FORCE_TOKENS } else { $MaxTokens }
    $body = @{
        model = $MODEL
        messages = @(@{role="user"; content=$Prompt})
        max_tokens = $mt
        temperature = 0.6
        top_p = 0.95
        stream = $true
        stream_options = @{include_usage = $true}
    }

    if ($ENABLE_THINKING -eq 1) {
        $body.chat_template_kwargs = @{enable_thinking = $true}
    }

    if ($FORCE_TOKENS -gt 0) {
        $body.min_tokens = $FORCE_TOKENS
        $body.ignore_eos = $true
    }

    $jsonBody = $body | ConvertTo-Json -Depth 4
    $tSend = Get-Date
    $ttft = $null
    $completionTokens = 0
    $promptTokens = 0

    $request = [System.Net.HttpWebRequest]::Create("$URL/v1/chat/completions")
    $request.Method = "POST"
    $request.ContentType = "application/json"
    $request.Timeout = 600000

    $streamBytes = [System.Text.Encoding]::UTF8.GetBytes($jsonBody)
    $requestStream = $request.GetRequestStream()
    $requestStream.Write($streamBytes, 0, $streamBytes.Length)
    $requestStream.Close()

    try {
        $response = $request.GetResponse()
        $responseStream = $response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($responseStream, [System.Text.Encoding]::UTF8)

        while ($true) {
            $line = $reader.ReadLine()
            if ($null -eq $line) { break }
            if (-not $line.StartsWith("data: ")) { continue }
            $payload = $line.Substring(6)
            if ($payload -eq "[DONE]") { break }

            try {
                $chunk = $payload | ConvertFrom-Json
                $choices = $chunk.choices
                if ($choices) {
                    $delta = $choices[0].delta
                    $content = if ($delta) { $delta.content } else { $choices[0].text }
                    if ($content -and $null -eq $ttft) {
                        $ttft = ((Get-Date) - $tSend).TotalSeconds
                    }
                }
                $usage = $chunk.usage
                if ($usage) {
                    $completionTokens = $usage.completion_tokens
                    $promptTokens = $usage.prompt_tokens
                }
            } catch {
                # Skip malformed JSON chunks
            }
        }
        $reader.Close()
        $responseStream.Close()
        $response.Close()
    } catch {
        throw "Request failed: $_"
    }

    $tEnd = Get-Date
    $wall = ($tEnd - $tSend).TotalSeconds
    if ($null -eq $ttft) { $ttft = $wall }
    if ($promptTokens -eq 0) { $promptTokens = [Math]::Max(1, ($Prompt.Split(" ").Count)) }

    $wallTPS = if ($wall -gt 0) { $completionTokens / $wall } else { 0 }
    $decodeWindow = $wall - $ttft
    $decodeTPS = $null
    $degen = $false

    if ($decodeWindow -gt 0 -and $decodeWindow -ge 0.05 * $wall) {
        $decodeTPS = $completionTokens / $decodeWindow
    } else {
        $degen = $true
    }

    return @{
        wall = $wall
        ttft = $ttft
        completionTokens = $completionTokens
        promptTokens = $promptTokens
        wallTPS = $wallTPS
        decodeTPS = $decodeTPS
        degen = $degen
    }
}

function Format-Result {
    param([string]$Label, [double]$Wall, [double]$TTFT, [int]$TokS)

    $wallTPS = if ($Wall -gt 0) { $TokS / $Wall } else { 0 }
    $decodeWindow = $Wall - $TTFT
    $ttftMs = [math]::Round($TTFT * 1000, 0)
    $wallRounded = [math]::Round($Wall, 2)
    $wallTPSRounded = [math]::Round($wallTPS, 2)
    $toksStr = $TokS.ToString().PadLeft(4)

    if ($decodeWindow -gt 0 -and $decodeWindow -ge 0.05 * $Wall) {
        $decodeTPS = $TokS / $decodeWindow
        $decodeTPSRounded = [math]::Round($decodeTPS, 2)
        $line = "  {0,-10} wall={1,6}s  ttft={2,4}ms  toks={3,4}  wall_TPS={4,6}  decode_TPS={5,6}" -f $Label, $wallRounded, $ttftMs, $toksStr, $wallTPSRounded, $decodeTPSRounded
    } else {
        $decPct = if ($Wall -gt 0) { $decodeWindow / $Wall * 100 } else { 0 }
        $decPctRounded = [math]::Round($decPct, 1)
        $decWinRounded = [math]::Round($decodeWindow, 3)
        $line = "  {0,-10} wall={1,6}s  ttft={2,4}ms  toks={3,4}  wall_TPS={4,6}  decode_TPS=n/a (decode window {5,5}s = {6,3}% of wall)" -f $Label, $wallRounded, $ttftMs, $toksStr, $wallTPSRounded, $decWinRounded, $decPctRounded
    }
    return $line
}

# Run the benchmarks
if ($ONLY -eq "narr" -or $ONLY -eq "both") {
    Invoke-BenchRun "NARRATIVE" $PROMPT_NARR $MAX_TOKENS_NARR
}

if ($ONLY -eq "code" -or $ONLY -eq "both") {
    Invoke-BenchRun "CODE" $PROMPT_CODE $MAX_TOKENS_CODE
}

Write-Host ""
Write-Host "bench.ps1 complete."
Write-Host "To save results: scripts/bench.ps1 | tee bench-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
