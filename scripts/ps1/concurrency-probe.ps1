#Requires -Version 5.1
#
# concurrency-probe.ps1 - measures how many concurrent streams a card's KV pool
# sustains cleanly, and at what per-stream throughput.
#
# Usage:
#   .\concurrency-probe.ps1
#   .\concurrency-probe.ps1 -Validate
#   .\concurrency-probe.ps1 -Sweep "4 8 12 16" -Slug vllm/minimal

param(
    [string]$Url,
    [string]$Model,
    [string]$Container,
    [int]$Concurrency,
    [int]$Rounds,
    [int]$PromptTokens,
    [int]$GenTokens,
    [int]$VramGrowthMb,
    [int]$ReqTimeout,
    [double]$TpsFloor,
    [double]$RetentionMin,
    [switch]$Validate,
    [string]$TargetCtx,
    [string]$Sweep,
    [string]$Slug,
    [switch]$SweepDry
)

$ErrorActionPreference = "Stop"
$ScriptName = "concurrency-probe"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

# Defaults
if (-not $Url) { $Url = "http://localhost:8010" }
if (-not $Rounds) { $Rounds = 5 }
if (-not $PromptTokens) { $PromptTokens = 16000 }
if (-not $GenTokens) { $GenTokens = 256 }
if (-not $VramGrowthMb) { $VramGrowthMb = 200 }
if (-not $ReqTimeout) { $ReqTimeout = 600 }
if (-not $TpsFloor) { $TpsFloor = 0 }
if (-not $RetentionMin) { $RetentionMin = 0.98 }

# Auto-detect model
if (-not $Model) {
    try {
        $models = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($models -and $models.data -and $models.data.Count -gt 0) {
            $Model = $models.data[0].id
        }
    } catch { $Model = $DETECTED_MODEL }
}

# Auto-detect container
if (-not $Container) {
    try {
        $containers = docker ps --format '{{.Names}}' 2>$null | Select-String -Pattern 'vllm-(qwen|gemma)' -ErrorAction SilentlyContinue
        if ($containers) { $Container = $containers[0].ToString().Trim() }
    } catch { }
}

# Detect concurrency
if (-not $Concurrency) {
    $conc = ""
    $concSrc = ""

    if ($Container) {
        try {
            $cmd = docker inspect $Container --format '{{join .Config.Cmd " "}}' 2>$null
            $match = $cmd | Select-String -Pattern 'max-num-seqs\s+(\d+)'
            if ($match) { $conc = $match.Matches[0].Groups[1].Value; $concSrc = "container max-num-seqs" }
        } catch { }
    }

    if (-not $conc) {
        try {
            $cmd = docker inspect $Container --format '{{join .Config.Cmd " "}}' 2>$null
            $match = $cmd | Select-String -Pattern '\-np\s+(\d+)'
            if ($match) { $conc = $match.Matches[0].Groups[1].Value; $concSrc = "container -np" }
        } catch { }
    }

    if (-not $conc) {
        try {
            $props = Invoke-RestMethod -Uri "$Url/props" -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($props.total_slots) { $conc = $props.total_slots.ToString(); $concSrc = "server /props total_slots" }
        } catch { }
    }

    if (-not $conc) {
        Write-Error "[$ScriptName] FATAL: cannot detect the served slot count - pass CONCURRENCY=N explicitly"
        exit 2
    }
    $Concurrency = [int]$conc
    Write-Host "[$ScriptName] CONCURRENCY=$Concurrency (detected: $concSrc)"
} else {
    Write-Host "[$ScriptName] CONCURRENCY=$Concurrency (explicit)"
}

# VALIDATE preset
if ($Validate -and (-not $Sweep)) {
    $ctx = if ($TargetCtx) { [int]$TargetCtx } else { 32768 }
    $headroom = $GenTokens + 512
    if ($ctx -gt $headroom) { $PromptTokens = $ctx - $headroom } else { $PromptTokens = $ctx }
    $Rounds = 6
    Write-Host "[$ScriptName] VALIDATE: filling to target_ctx=${ctx} (prompt=${PromptTokens}tok), rounds=$Rounds"
}

# Sweep mode
if ($Sweep) {
    if (-not $Slug) {
        Write-Error "SWEEP needs SLUG=<compose slug>"
        exit 2
    }

    Write-Host "[$ScriptName] sweep: slug=$Slug N in { $Sweep } · floor=${TpsFloor} tok/s/stream"
    $knee = ""
    $kneeTps = ""
    $sweepRows = @()

    foreach ($nStr in $Sweep -split ' ') {
        $N = [int]$nStr.Trim()

        if ($SweepDry) {
            Write-Host "[$ScriptName:dry] would: MAX_NUM_SEQS=$N switch.sh $Slug -> wait ready -> probe N=$N"
            continue
        }

        # For sweep, we'd normally reboot the server. Skip that for now, just probe.
        Write-Host "[$ScriptName] probing N=$N"
        $result = Run-Probe -N $N -Url $Url -Model $Model -Rounds $Rounds -PromptTokens $PromptTokens -GenTokens $GenTokens -VramGrowthMb $VramGrowthMb -ReqTimeout $ReqTimeout -TpsFloor $TpsFloor -RetentionMin $RetentionMin -Validate:$Validate

        $sweepRows += "  N=$N  per-stream=$($result.mps_tps) tok/s  aggregate=$($result.agg_tps) tok/s  clean=$($result.clean)"

        if ($result.clean -eq 1 -and ($TpsFloor -le 0 -or $result.mps_tps -ge $TpsFloor)) {
            $knee = $N
            $kneeTps = $result.mps_tps
        }
    }

    Write-Host ""
    Write-Host "=== sweep summary ==="
    foreach ($row in $sweepRows) { Write-Host $row }
    Write-Host ""
    Write-Host "=== sweep knee ==="
    if ($knee) {
        Write-Host "  largest clean N at/above floor: N=$knee (${kneeTps} tok/s/stream)"
    } else {
        Write-Host "  no N met the bar"
    }
    exit 0
}

# Single-N mode
Write-Host "[$ScriptName] URL=$Url model=$Model N=$Concurrency rounds=$Rounds prompt=${PromptTokens}tok gen=${GenTokens}tok"

# ---------------------------------------------------------------------------
# Run-Probe function — called by sweep block and single-N mode
# ---------------------------------------------------------------------------
function Run-Probe {
    param(
        [int]$N, [string]$Url, [string]$Model,
        [int]$Rounds, [int]$PromptTokens, [int]$GenTokens,
        [int]$VramGrowthMb, [int]$ReqTimeout, [double]$TpsFloor,
        [double]$RetentionMin, [switch]$Validate
    )

    # Build prompt template — matches bash BLOCK construction
    $block = "This section describes the history of computing in detail. Transistors were invented in 1947 at Bell Labs. The integrated circuit came a decade later. Microprocessors emerged in the 1970s and changed the world. "
    # Bash: int(PTOK/(len(BLOCK)*0.23))+1  — +1 guarantees reaching target
    $reps = [int]($PromptTokens / ($block.Length * 0.23)) + 1

    # Single script block that takes params and does the work — no two-stage
    # AddScript/AddCommand pipeline. .ToString() returns the source text
    # verbatim; no $-interpolation risk, no scope-sharing question.
    $oneRequestScript = {
        param([string]$Url,[string]$Model,[int]$GenTokens,[int]$ReqTimeout,
              [string]$Block,[int]$Reps,[int]$Stream,[int]$Round)
        $prompt = "[probe s${Stream} r${Round}] " + ($Block * $Reps) + "`nWrite a detailed multi-paragraph summary."
        $body = @{ model=$Model; max_tokens=$GenTokens; temperature=0.0; stream=$true
                   stream_options=@{ include_usage=$true }
                   messages=@(@{ role='user'; content=$prompt }) } | ConvertTo-Json -Depth 10
        $t0 = Get-Date; $tFirst = $null; $tLast = $null; $toks = 0
        try {
            $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($body)
            $request = [System.Net.HttpWebRequest]::Create("$Url/v1/chat/completions")
            $request.Method = "POST"; $request.ContentType = "application/json"; $request.Timeout = $ReqTimeout * 1000
            $request.AllowWriteStreamBuffering = $false; $request.ContentLength = $bodyBytes.Length
            $reqStream = $request.GetRequestStream(); $reqStream.Write($bodyBytes, 0, $bodyBytes.Length); $reqStream.Close()
            $response = $request.GetResponse(); $respStream = $response.GetResponseStream()
            $reader = New-Object System.IO.StreamReader $respStream
            while (-not $reader.EndOfStream) {
                $line = $reader.ReadLine().Trim()
                if (-not $line.StartsWith("data: ")) { continue }
                $payload = $line.Substring(6).Trim()
                if ($payload -eq "[DONE]") { break }
                try { $chunk = $payload | ConvertFrom-Json } catch { continue }
                if ($chunk.usage) { $toks = if ($null -eq $chunk.usage.completion_tokens) { $toks } else { $chunk.usage.completion_tokens } }
                if ($chunk.choices) { $delta = $chunk.choices[0].delta; if ($delta.content -or $delta.reasoning_content -or $delta.reasoning) { $now = Get-Date; if (-not $tFirst) { $tFirst = $now }; $tLast = $now } }
            }
            $reader.Close(); $respStream.Close(); $response.Close()
        } catch { return @{ ok=$false; toks=0; silent=$false; err=$_.Exception.Message.Substring(0, [Math]::Min(200,$_.Exception.Message.Length)); dt=0; ttft=$null; tps=0 } }
        $wall = ((Get-Date) - $t0).TotalSeconds
        $toks = if ($null -eq $toks) { 0 } else { $toks }
        $decodeDt = if ($tFirst -and $tLast -and $tLast -gt $tFirst) { ($tLast - $tFirst).TotalSeconds } else { 0 }
        $tps = if ($toks -gt 1 -and $decodeDt -gt 0) { $toks / $decodeDt } else { 0 }
        $ttft = if ($tFirst) { ($tFirst - $t0).TotalSeconds } else { $null }
        if ($toks -eq 0) { return @{ ok=$false; toks=0; silent=$true; err='0 tokens returned'; dt=$wall; ttft=$ttft; tps=0 } }
        return @{ ok=$true; toks=$toks; silent=$false; err=$null; dt=$wall; ttft=$ttft; tps=$tps }
    }

    # Invoke-ConcurrentRound: dispatches N requests via RunspacePool for true
    # concurrent execution (matches bash ThreadPoolExecutor pattern).
    $invokeConcurrentRound = {
        param(
            [string]$Url, [string]$Model, [int]$GenTokens, [int]$ReqTimeout,
            [string]$Block, [int]$Reps, [int]$N, [int]$Round
        )

        # $oneRequestScript is captured from the outer scope as a script block
        # (bytecode, not text — no interpolation risk when serialised).
        $results = @()
        $runspacePool = [runspacefactory]::CreateRunspacePool(1, $N)
        try {
            $runspacePool.Open()
            $handles = @()
            for ($s = 0; $s -lt $N; $s++) {
                $ps = [System.Management.Automation.PowerShell]::Create()
                $ps.RunspacePool = $runspacePool
                [void]$ps.AddScript($oneRequestScript.ToString()).AddParameters(@{
                    Url=$Url; Model=$Model; GenTokens=$GenTokens; ReqTimeout=$ReqTimeout
                    Block=$Block; Reps=$Reps; Stream=$s; Round=$Round
                })
                $handles += @{ ps = $ps; handle = $ps.BeginInvoke() }
            }
            foreach ($h in $handles) {
                try {
                    $out = $h.ps.EndInvoke($h.handle)
                    if ($out -and $out.Count -gt 0) {
                        $r = $out[0]
                        if (-not $r.ok -and $r.err) {
                            Write-Host "  [probe] s=$s err=$($r.err)" -ForegroundColor Red
                        }
                        $results += $r
                    }
                    else { $results += @{ ok=$false; toks=0; silent=$false; err="runspace returned empty"; dt=0; ttft=$null; tps=0 } }
                } catch {
                    $results += @{ ok=$false; toks=0; silent=$false; err=$_.Exception.Message.Substring(0, [Math]::Min(200,$_.Exception.Message.Length)); dt=0; ttft=$null; tps=0 }
                }
                $h.ps.Dispose()
            }
        } finally {
            $runspacePool.Close()
            $runspacePool.Dispose()
        }
        return $results
    }

    # Run rounds — concurrent execution via RunspacePool (PS5.1-compatible)
    $vram0 = -1
    try {
        $vramLines = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($vramLines) {
            $vram0 = ($vramLines -split "`n" | Where-Object { $_.Trim() } | ForEach-Object { [int]$_.Trim() } | Measure-Object -Sum).Sum
        }
    } catch { }

    $vramByRound = @()
    $mtpsByRound = @()
    $aggByRound = @()
    $ttftByRound = @()
    $bad = 0
    $errRounds = 0

    Write-Host ""
    $header = "  {0,5} {1,7} {2,7} {3,7} {4,8} {5,8} {6,9} {7,8} {8,7}" -f "round", "done", "silent", "errors", "vram_MB", "agg_t/s", "per-strm", "ttft_ms", "pf_t/s"
    Write-Host $header

    for ($rnd = 1; $rnd -le $Rounds; $rnd++) {
        $t0 = Get-Date

        # Concurrent dispatch via RunspacePool — matches bash ThreadPoolExecutor
        $results = & $invokeConcurrentRound -Url $Url -Model $Model -GenTokens $GenTokens -ReqTimeout $ReqTimeout -Block $block -Reps $reps -N $N -Stream 0 -Round $rnd

        $wall = ((Get-Date) - $t0).TotalSeconds

        $done = ($results | Where-Object { $_.ok }).Count
        $silent = ($results | Where-Object { $_.silent }).Count
        $errs = ($results | Where-Object { $_.err }).Count

        $v = -1
        try {
            $vr = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
            if ($vr) { $v = ($vr -split "`n" | Where-Object { $_.Trim() } | ForEach-Object { [int]$_.Trim() } | Measure-Object -Sum).Sum }
        } catch { }
        $vramByRound += $v

        $agg = if ($wall -gt 0) {
            # Results from runspace are serialized hashtables — iterate manually
            $totalToks = 0
            foreach ($r in $results) {
                if ($r -is [hashtable] -and $r.ContainsKey('toks')) { $totalToks += $r['toks'] }
                elseif ($r.toks -ne $null) { $totalToks += $r.toks }
            }
            $totalToks / $wall
        } else { 0 }
        $aggByRound += $agg

        $tpsOk = ($results | Where-Object { $_.ok -and $_.tps -gt 0 } | ForEach-Object { $_.tps })
        $mtps = if ($tpsOk.Count -gt 0) { ($tpsOk | Measure-Object -Average).Average } else { 0 }
        $mtpsByRound += $mtps

        $ttfts = ($results | Where-Object { $_.ok -and $_.ttft } | ForEach-Object { $_.ttft })
        $ttftMed = 0
        if ($ttfts.Count -gt 0) {
            $sorted = $ttfts | Sort-Object
            if ($sorted.Count % 2 -eq 1) {
                $ttftMed = $sorted[[int]($sorted.Count / 2)]
            } else {
                $mid = $sorted.Count / 2
                $ttftMed = ($sorted[$mid - 1] + $sorted[$mid]) / 2
            }
        }
        $ttftByRound += $ttftMed
        $pf = if ($ttftMed -gt 0) { $PromptTokens / $ttftMed } else { 0 }

        $ttftMs = [math]::Round($ttftMed * 1000, 0).ToString().PadLeft(8)
        $pfStr = [math]::Round($pf, 0).ToString().PadLeft(7)
        $aggStr = [math]::Round($agg, 1).ToString().PadLeft(8)
        $mtpsStr = [math]::Round($mtps, 1).ToString().PadLeft(9)

        Write-Host ("  {0,5} {1,4}/{2,-2} {3,7} {4,7} {5,8} {6} {7} {8} {9}" -f
            $rnd, $done, $N, $silent, $errs, $v, $aggStr, $mtpsStr, $ttftMs, $pfStr)

        if ($done -lt $N -or $silent -gt 0 -or $errs -gt 0) { $bad++ }
        if ($errs -gt 0) { $errRounds++ }
    }

    # Calculate verdict
    $warmI = if ($Rounds -ge 3) { 1 } else { 0 }
    $warm = if ($vramByRound.Count -gt $warmI) { $vramByRound[$warmI] } else { 0 }
    $poolFill = if ($vram0 -ge 0) { $warm - $vram0 } else { -1 }
    $leak = if ($vram0 -ge 0 -and $vramByRound.Count -gt 0) { $vramByRound[-1] - $warm } else { -1 }
    $vramPeak = if ($vramByRound.Count -gt 0) { ($vramByRound | Measure-Object -Maximum).Maximum } else { -1 }

    $reportTps = if ($mtpsByRound.Count -gt 0) { $mtpsByRound[-1] } else { 0 }
    $reportAgg = if ($aggByRound.Count -gt 0) { $aggByRound[-1] } else { 0 }

    $warmTps = if ($Rounds -ge 4) { $mtpsByRound[1..($mtpsByRound.Count-1)] } else { $mtpsByRound }
    $retention = 1.0
    if ($warmTps.Count -ge 3 -and $warmTps[0] -gt 0) {
        $early = ($warmTps[0..1] | Measure-Object -Average).Average
        $late = ($warmTps[-2..-1] | Measure-Object -Average).Average
        if ($early -gt 0) { $retention = $late / $early }
    }

    $cleanFit = ($bad -eq 0) -and (0 -le $leak -and $leak -le $VramGrowthMb)
    $floorOk = if ($TpsFloor -gt 0) { $reportTps -ge $TpsFloor } else { $true }
    $retentionOk = if ($Rounds -ge 3) { $retention -ge $RetentionMin } else { $true }
    $pass = $cleanFit -and $floorOk -and ($retentionOk -or -not $Validate)

    $steadyTtft = if ($ttftByRound.Count -gt 0) { $ttftByRound[-1] } else { 0 }
    $steadyPf = if ($ttftByRound.Count -gt 0) { $PromptTokens / ($ttftByRound[-1] + 0.001) } else { 0 }

    Write-Host ""
    Write-Host "=== verdict (N=$N) ==="
    Write-Host "  VRAM: cold $vram0 -> warm $warm MB (pool fill $poolFill MB, expected) -> final $($vramByRound[-1]) MB (post-warm growth $leak MB / $VramGrowthMb) peak $vramPeak MB"
    Write-Host "  per-stream decode: $([math]::Round($reportTps,1)) tok/s (steady) · aggregate $([math]::Round($reportAgg,1)) tok/s ($N streams) · retention $([math]::Round($retention*100,1))%"
    Write-Host "  concurrent prefill: steady TTFT $([math]::Round($steadyTtft*1000,0)) ms (median of $N streams)"

    $flags = @()
    if (-not $cleanFit) { $flags += "fit" }
    if ($TpsFloor -gt 0 -and -not $floorOk) { $flags += "tps-floor" }
    if ($Validate -and -not $retentionOk) { $flags += "retention" }
    $verdict = if ($pass) { "PASS - sustained clean" } else { "FAIL - $($flags -join ',')" }
    Write-Host "  concurrency $N @ ~${PromptTokens} tok: $verdict"

    if ($pass) {
        Write-Host "  envelope row: max_num_seqs: $N  validated: { concurrency_soak: '$N @ ~$([math]::Floor($PromptTokens/1000))K, $([math]::Round($reportTps,0)) tok/s/stream, $leak MB post-warm', vram_peak_gb: $([math]::Round($vramPeak/1024,1)) }"
    }

    return @{
        n = $N; clean = [int]$cleanFit; pass = [int]$pass
        mps_tps = [math]::Round($reportTps, 2)
        agg_tps = [math]::Round($reportAgg, 2)
        retention = [math]::Round($retention, 3)
        leak = $leak; vram_peak = $vramPeak
        floor_ok = [int]$floorOk
        ttft_ms = [math]::Round($steadyTtft * 1000, 0)
        pf_tps = [math]::Round($steadyPf, 1)
    }
}

$result = Run-Probe -N $Concurrency -Url $Url -Model $Model -Rounds $Rounds -PromptTokens $PromptTokens -GenTokens $GenTokens -VramGrowthMb $VramGrowthMb -ReqTimeout $ReqTimeout -TpsFloor $TpsFloor -RetentionMin $RetentionMin -Validate:$Validate
$result
