#Requires -Version 5.1
#
# club-3090 Interactive Launcher - Phase-based menu with status tracking.
# Groups scripts by workflow phase, shows completion status,
# runs with progress animations.
#
# Phases (ordered for testing a local LLM endpoint):
#   1. Pre-flight (verify, health)
#   2. GPU Config (NVLink, PCIe-P2P)
#   3. Verify (full, stress)
#   4. Benchmark (bench-full, all-in-one)
#   5. Soak Test (VRAM accretion, TPS retention)
#   6. Advanced (concurrency, power-cap)
#   7. Specialized (agentic, arch-ab)
#   8. Quality (test, baseline, rerun-failed)
#   9. Submission (rebench-full, rebench-runtime, submit-bench)
#  10. Reporting (report, catalog-baseline)
#  11. Tools (check-syntax, check-issues, verify-ours)
#  12. System (capture)
#
# Status tracking persists in .\scripts\status.json.
# Each script can be run individually or in groups.

param(
    [switch]$Reset
)

$ErrorActionPreference = "Continue"

# ---------------------------------------------------------------------------
# Resolve script directory relative to this file (portable across paths)
# ---------------------------------------------------------------------------
$SCRIPTS_DIR = Split-Path $MyInvocation.MyCommand.Path -Parent
if (-not $SCRIPTS_DIR) { $SCRIPTS_DIR = $PSScriptRoot }
if (-not $SCRIPTS_DIR) { $SCRIPTS_DIR = (Get-Location).Path }

$ROOT = $SCRIPTS_DIR
$STATUS_FILE = Join-Path $ROOT "status.json"

# ---------------------------------------------------------------------------
# Reset status tracking if --Reset is passed
# ---------------------------------------------------------------------------
if ($Reset) {
    if (Test-Path $STATUS_FILE) {
        Remove-Item $STATUS_FILE -Force
        Write-Host "  Status tracking cleared." -ForegroundColor Green
    } else {
        Write-Host "  No status file to clear." -ForegroundColor DarkGray
    }
    exit 0
}

# ---------------------------------------------------------------------------
# Model auto-detection on launch — query the running server and cache the name
# Result is written to model.json so all child scripts can read it.
# ---------------------------------------------------------------------------
$MODEL_CACHE_FILE = Join-Path $ROOT "model.json"
function Detect-Model {
    # Use cached value if available
    if (Test-Path $MODEL_CACHE_FILE) {
        $cached = Get-Content $MODEL_CACHE_FILE | ConvertFrom-Json
        if ($cached.model) { return $cached.model }
    }
    # Probe the server
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8010/v1/models" -TimeoutSec 5
        $model = $resp.data[0].id
        @{ model = $model } | ConvertTo-Json | Set-Content $MODEL_CACHE_FILE
        Write-Host "`n  Detected model: $model" -ForegroundColor Cyan
        return $model
    } catch {
        Write-Host "  Warning: could not detect model from server" -ForegroundColor Yellow
        return $null
    }
}
$DETECTED_MODEL = Detect-Model
if ($DETECTED_MODEL) {
    $env:MODEL = $DETECTED_MODEL
}

# Status tracking -----------------------------------------------------------
function Get-Status {
    param([string]$ScriptName)
    if (Test-Path $STATUS_FILE) {
        $status = Get-Content $STATUS_FILE | ConvertFrom-Json
        if ($status.PSObject.Properties.Name -contains $ScriptName) {
            return $status.$ScriptName
        }
    }
    return @{ status = "pending"; started = $null; completed = $null; duration = $null }
}

function Set-Status {
    param([string]$ScriptName, [string]$Status, [double]$Duration = 0)
    $data = @{}
    if (Test-Path $STATUS_FILE) {
        $data = @{}
        $json = Get-Content $STATUS_FILE | ConvertFrom-Json
        foreach ($prop in $json.PSObject.Properties) {
            $data[$prop.Name] = $prop.Value
        }
    }
    $data[$ScriptName] = @{
        status = $Status
        started = if ($Status -eq "running") { Get-Date -Format "yyyy-MM-ddTHH:mm:ss" } else { $data[$ScriptName].started }
        completed = if ($Status -eq "completed") { Get-Date -Format "yyyy-MM-ddTHH:mm:ss" } else { $data[$ScriptName].completed }
        duration = if ($Status -eq "completed") { [math]::Round($Duration, 1) } else { $data[$ScriptName].duration }
    }
    $data | ConvertTo-Json -Depth 3 | Set-Content $STATUS_FILE -Encoding UTF8
}

function Get-StatusIcon {
    param([string]$Status)
    switch ($Status) {
        "completed" { return "\u2705" }
        "running"   { return "\u231b" }
        "failed"    { return "\u274c" }
        default     { return "\u23f8" }
    }
}

# Animation helpers ----------------------------------------------------------
function Show-Progress {
    param([string]$Message, [switch]$NoAnim)
    if ($NoAnim) {
        Write-Host ""
        Write-Host "  $Message" -ForegroundColor Cyan
        return
    }
    $frames = @("\u280b", "\u2819", "\u2819", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2827", "\u280f")
    $i = 0
    while ($true) {
        $frame = $frames[$i % $frames.Count]
        Write-Host "`r  $frame $Message" -NoNewline
        Start-Sleep -Milliseconds 100
        $i++
    }
}

function Stop-Progress {
    param([string]$FinalMessage)
    Write-Host "`r  \u2705 $FinalMessage" -ForegroundColor Green
}

# Script runner --------------------------------------------------------------
function Invoke-Script {
    param(
        [string]$ScriptPath,
        [string]$ScriptName,
        [string]$Args = "",
        [string]$Runner = "auto"
    )

    if (-not (Test-Path $ScriptPath)) {
        Write-Host "  Warning: Script not found: $ScriptPath" -ForegroundColor Yellow
        return $false
    }

    if ($Runner -eq "auto") {
        $ext = [System.IO.Path]::GetExtension($ScriptPath).ToLower()
        if ($ext -eq ".ps1") { $Runner = "powershell" }
        elseif ($ext -eq ".sh") { $Runner = "bash" }
        else { $Runner = "powershell" }
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Set-Status -ScriptName $ScriptName -Status "running"

    $progressMsg = "Running $ScriptName..."
    if ($Runner -eq "bash") { $progressMsg = "Running $ScriptName (via bash)..." }

    $exitCode = 0
    $output = ""

    try {
        if ($Runner -eq "pwsh") {
            $output = & $ScriptPath @($Args.Split(' ') | Where-Object { $_ }) 2>&1 | Out-String
            $exitCode = $LASTEXITCODE
        }
        elseif ($Runner -eq "powershell") {
            $output = & powershell -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @($Args.Split(' ') | Where-Object { $_ }) *>&1 | Out-String
            $exitCode = $LASTEXITCODE
        }
        elseif ($Runner -eq "bash") {
            $bashPath = Get-Command bash -ErrorAction SilentlyContinue
            if ($bashPath) {
                $output = & $bashPath.FullName "-c" "cd `"$ROOT`" && bash `"$ScriptPath`" $Args" 2>&1 | Out-String
                $exitCode = $LASTEXITCODE
            }
            else {
                Write-Host "  Warning: bash not found in PATH" -ForegroundColor Yellow
                $exitCode = 1
            }
        }
        elseif ($Runner -eq "cmd") {
            $output = cmd /c "cd /d `"$ROOT`" && $ScriptPath $Args" 2>&1 | Out-String
            $exitCode = $LASTEXITCODE
        }
    }
    catch {
        $output = $_.Exception.Message
        $exitCode = 1
    }

    $sw.Stop()
    Stop-Progress "$ScriptName completed in $($sw.Elapsed.TotalSeconds.ToString('0.0'))s"

    if ($exitCode -eq 0) {
        Set-Status -ScriptName $ScriptName -Status "completed" -Duration $sw.Elapsed.TotalSeconds
        Write-Host "  OK Exit code: 0" -ForegroundColor Green
    }
    else {
        Set-Status -ScriptName $ScriptName -Status "failed" -Duration $sw.Elapsed.TotalSeconds
        Write-Host "  FAIL Exit code: $exitCode" -ForegroundColor Red
    }

    return ($exitCode -eq 0)
}

# Menu definitions -----------------------------------------------------------
function Get-ScriptList {
    param([string]$Phase)

    $scripts = @{
        "Pre-flight" = @(
            @{ Name = "Verify"; Script = "verify.ps1"; Runner = "powershell"; Desc = "Smoke test: reachability, completion, tool calling" }
            ,@{ Name = "Health"; Script = "health.ps1"; Runner = "powershell"; Desc = "Runtime health check of running server" }
        )
        "GPU Config" = @(
            @{ Name = "Detect NVLink"; Script = "detect_nvlink.ps1"; Runner = "powershell"; Desc = "NVLink/PCIe-P2P detection and configuration" }
        )
        "Verify" = @(
            @{ Name = "Verify Full"; Script = "verify-full.ps1"; Runner = "powershell"; Desc = "Functional tests: reachability, tool calling, quality" }
            ,@{ Name = "Verify Stress"; Script = "verify-stress.ps1"; Runner = "powershell"; Desc = "Stress tests: needles, OOM, ceiling ladder" }
        )
        "Benchmark" = @(
            @{ Name = "Bench Full"; Script = "bench-full.ps1"; Runner = "powershell"; Desc = "Full benchmark: TPS, TTFT, engine metrics (narrative + code)" }
            ,@{ Name = "All-in-One"; Script = "all-in-one-comprehensive.ps1"; Runner = "powershell"; Desc = "All-in-one: verify -> bench -> stress -> soak -> report" }
        )
        "Soak Test" = @(
            @{ Name = "Soak Test"; Script = "soak-test.ps1"; Runner = "powershell"; Desc = "Runtime soak test: VRAM accretion, TPS retention" }
        )
        "Advanced" = @(
            @{ Name = "Concurrency Probe"; Script = "concurrency-probe.ps1"; Runner = "powershell"; Desc = "Concurrency stress testing" }
            ,@{ Name = "Power Cap Sweep"; Script = "power-cap-sweep.ps1"; Runner = "powershell"; Desc = "Power cap and performance sweep" }
        )
        "Specialized" = @(
            @{ Name = "Bench Agentic"; Script = "bench-agentic.ps1"; Runner = "powershell"; Desc = "Agentic benchmark: multi-turn tool use" }
            ,@{ Name = "Arch A/B"; Script = "arch-ab.ps1"; Runner = "powershell"; Desc = "Architecture A/B comparison" }
        )
        "Quality" = @(
            @{ Name = "Quality Full"; Script = "quality-full.ps1"; Runner = "powershell"; Desc = "Quality testing with sandboxed packs" }
            ,@{ Name = "Quality Baseline"; Script = "quality-baseline.ps1"; Runner = "powershell"; Desc = "Quality baseline comparison" }
            ,@{ Name = "Rerun Failed"; Script = "rerun-failed-packs.ps1"; Runner = "powershell"; Desc = "Re-test failed quality scenarios" }
        )
        "Submission" = @(
            @{ Name = "Rebench Full"; Script = "rebench-full.ps1"; Runner = "powershell"; Desc = "Full benchmark re-run for submission" }
            ,@{ Name = "Rebench Runtime"; Script = "rebench-runtime.ps1"; Runner = "powershell"; Desc = "Runtime-only rebench (skip quality)" }
            ,@{ Name = "Submit Bench"; Script = "submit-bench.ps1"; Runner = "powershell"; Desc = "Submit benchmark results" }
        )
        "Reporting" = @(
            @{ Name = "Report"; Script = "report.ps1"; Runner = "powershell"; Desc = "Triage report: hardware, GPU state, verdicts" }
            ,@{ Name = "Catalog Baseline"; Script = "catalog-baseline.ps1"; Runner = "powershell"; Desc = "Induct baseline into catalog" }
        )
        "Tools" = @(
            @{ Name = "Check Syntax"; Script = "check-syntax.ps1"; Runner = "powershell"; Desc = "Verify syntax of all ps1 files" }
            ,@{ Name = "Check Issues"; Script = "check-issues.ps1"; Runner = "powershell"; Desc = "Scan for PS5.1 compatibility issues" }
            ,@{ Name = "Verify Ours"; Script = "verify-ours.ps1"; Runner = "powershell"; Desc = "Targeted syntax checker for ported scripts" }
        )
        "System" = @(
            @{ Name = "Capture"; Script = "capture.ps1"; Runner = "powershell"; Desc = "Capture/backup (engine-side metrics)" }
        )
    }

    if ($Phase) { return $scripts[$Phase] }
    return $scripts
}

# Display functions ----------------------------------------------------------
function Show-Header {
    Clear-Host
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "       club-3090 Interactive Launcher                     " -ForegroundColor Cyan
    Write-Host "       Native Windows PowerShell Port                     " -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Phase: $Phase" -ForegroundColor Gray
    Write-Host ""
}

# ---------------------------------------------------------------------------
# STATUS OVERVIEW — internal loop, runs scripts when user selects a phase
# ---------------------------------------------------------------------------
function Show-StatusOverview {
    Clear-Host
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "       Status Overview                                  " -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host ""

    $allPhases = @("Pre-flight", "GPU Config", "Verify", "Benchmark", "Soak Test", "Advanced", "Specialized", "Quality", "Submission", "Reporting", "Tools", "System")
    $total = 0; $completed = 0; $failed = 0; $pending = 0

    foreach ($phase in $allPhases) {
        $scripts = Get-ScriptList -Phase $phase
        if (-not $scripts) { continue }
        foreach ($s in $scripts) {
            $total++
            $status = Get-Status -ScriptName $s.Name
            switch ($status.status) {
                "completed" { $completed++ }
                "failed"    { $failed++ }
                default     { $pending++ }
            }
        }
    }

    Write-Host "  Total scripts: $total" -ForegroundColor White
    Write-Host "  Completed:     $completed" -ForegroundColor Green
    Write-Host "  Failed:        $failed" -ForegroundColor Red
    Write-Host "  Pending:       $pending" -ForegroundColor DarkGray
    Write-Host ""

    $progress = if ($total -gt 0) { [math]::Round(($completed / $total) * 100) } else { 0 }
    $barLength = 30
    $filled = [math]::Round($barLength * $progress / 100)
    $bar = [string]::new('x', $filled) + [string]::new('-', $barLength - $filled)
    Write-Host "  [$bar] $progress%" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "  Phase breakdown:" -ForegroundColor Gray
    Write-Host "  ----------------------------------------------------" -ForegroundColor DarkGray
    foreach ($phase in $allPhases) {
        $scripts = Get-ScriptList -Phase $phase
        if (-not $scripts) { continue }
        $phaseCompleted = 0; $phaseTotal = $scripts.Count
        foreach ($s in $scripts) {
            $status = Get-Status -ScriptName $s.Name
            if ($status.status -eq "completed") { $phaseCompleted++ }
        }
        $phaseProgress = if ($phaseTotal -gt 0) { [math]::Round(($phaseCompleted / $phaseTotal) * 100) } else { 0 }
        $icon = if ($phaseCompleted -eq $phaseTotal) { "\u2705" } elseif ($phaseCompleted -gt 0) { "\u231b" } else { "\u23f8" }
        $pColor = if ($phaseProgress -eq 100) { "Green" } elseif ($phaseProgress -gt 0) { "Yellow" } else { "DarkGray" }
        Write-Host "  $icon $($phase.PadRight(15)) $phaseCompleted/$phaseTotal ($phaseProgress%)" -ForegroundColor $pColor
    }
    Write-Host ""

    Write-Host "  [1] Go to Pre-flight" -ForegroundColor White
    Write-Host "  [2] Go to GPU Config" -ForegroundColor White
    Write-Host "  [3] Go to Verify" -ForegroundColor White
    Write-Host "  [4] Go to Benchmark" -ForegroundColor White
    Write-Host "  [5] Go to Soak Test" -ForegroundColor White
    Write-Host "  [6] Go to Advanced" -ForegroundColor White
    Write-Host "  [7] Go to Specialized" -ForegroundColor White
    Write-Host "  [8] Go to Quality" -ForegroundColor White
    Write-Host "  [9] Go to Submission" -ForegroundColor White
    Write-Host "  [10] Go to Reporting" -ForegroundColor White
    Write-Host "  [11] Go to Tools" -ForegroundColor White
    Write-Host "  [12] Go to System" -ForegroundColor White
    Write-Host "  [0] Back to main menu" -ForegroundColor Red
    Write-Host ""

    # Internal loop — stays here until user presses 0
    while ($true) {
        Write-Host "  Enter choice: " -NoNewline
        $choice = Read-Host
        switch ($choice) {
            "1"  { $currentPhase = "Pre-flight"; Show-PhaseMenu -Phase $currentPhase; break }
            "2"  { $currentPhase = "GPU Config"; Show-PhaseMenu -Phase $currentPhase; break }
            "3"  { $currentPhase = "Verify"; Show-PhaseMenu -Phase $currentPhase; break }
            "4"  { $currentPhase = "Benchmark"; Show-PhaseMenu -Phase $currentPhase; break }
            "5"  { $currentPhase = "Soak Test"; Show-PhaseMenu -Phase $currentPhase; break }
            "6"  { $currentPhase = "Advanced"; Show-PhaseMenu -Phase $currentPhase; break }
            "7"  { $currentPhase = "Specialized"; Show-PhaseMenu -Phase $currentPhase; break }
            "8"  { $currentPhase = "Quality"; Show-PhaseMenu -Phase $currentPhase; break }
            "9"  { $currentPhase = "Submission"; Show-PhaseMenu -Phase $currentPhase; break }
            "10" { $currentPhase = "Reporting"; Show-PhaseMenu -Phase $currentPhase; break }
            "11" { $currentPhase = "Tools"; Show-PhaseMenu -Phase $currentPhase; break }
            "12" { $currentPhase = "System"; Show-PhaseMenu -Phase $currentPhase; break }
            "0"  { return }
            default { Write-Host "  Invalid choice" -ForegroundColor Yellow }
        }
    }
}

# ---------------------------------------------------------------------------
# PHASE MENU — internal loop, runs scripts when user selects an option
# ---------------------------------------------------------------------------
function Show-PhaseMenu {
    param([string]$Phase)

    $scripts = Get-ScriptList -Phase $Phase
    if (-not $scripts) {
        Write-Host "  No scripts found for phase '$Phase'" -ForegroundColor Yellow
        return
    }

    while ($true) {
        Clear-Host
        Write-Host ""
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "       Phase: $Phase                                    " -ForegroundColor Cyan
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host ""

        Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
        Write-Host "  |  Scripts in this phase:                    " -ForegroundColor DarkGray
        Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
        Write-Host ""

        foreach ($i in 0..($scripts.Count - 1)) {
            $s = $scripts[$i]
            $status = Get-Status -ScriptName $s.Name
            $icon = Get-StatusIcon -Status $status.status
            $duration = if ($status.duration) { " (${status.duration}s)" } else { "" }
            $color = switch ($status.status) {
                "completed" { "Green" }
                "running"   { "Yellow" }
                "failed"    { "Red" }
                default     { "DarkGray" }
            }

            $name = if ($s.Name) { $s.Name } else { "unnamed" }
            $desc = if ($s.Desc) { $s.Desc } else { "" }
            Write-Host "  [$($i+1)] $icon $($name.PadRight(20)) $desc" -ForegroundColor $color
            if ($duration) { Write-Host "       Duration: $duration" -ForegroundColor DarkGray }
        }

        Write-Host ""
        Write-Host "  [0] Back to main menu" -ForegroundColor White
        Write-Host "  [x] Run all scripts in this phase" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Enter choice: " -NoNewline
        $choice = Read-Host

        if ($choice -eq "x") {
            Run-PhaseAll -Phase $Phase
        }
        elseif ($choice -eq "0") {
            return
        }
        elseif ([int]::TryParse($choice, [ref]$null)) {
            $idx = [int]$choice - 1
            if ($idx -ge 0 -and $idx -lt $scripts.Count) {
                $s = $scripts[$idx]
                Write-Host ""
                Write-Host "  Running $($s.Name): $($s.Desc)" -ForegroundColor Cyan
                $success = Invoke-Script -ScriptPath (Join-Path $SCRIPTS_DIR $s.Script) -ScriptName $s.Name -Runner $s.Runner
                Write-Host ""
                if ($success) {
                    Write-Host "  Press any key to return to menu..." -NoNewline
                }
                else {
                    Write-Host "  Press any key to continue..." -NoNewline
                }
                $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            }
            else {
                Write-Host "  Invalid choice" -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "  Invalid choice" -ForegroundColor Yellow
        }
    }
}

# Run all scripts in a phase -------------------------------------------------
function Run-PhaseAll {
    param([string]$Phase)

    $scripts = Get-ScriptList -Phase $Phase
    if (-not $scripts) {
        Write-Host "  No scripts found for phase '$Phase'" -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "  Running all scripts in phase '$Phase'..." -ForegroundColor Cyan
    Write-Host ""

    $results = @{}
    foreach ($s in $scripts) {
        Write-Host ""
        Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
        Write-Host "  |  Running: $($s.Name)                      " -ForegroundColor DarkGray
        Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray

        $success = Invoke-Script -ScriptPath (Join-Path $SCRIPTS_DIR $s.Script) -ScriptName $s.Name -Runner $s.Runner

        if ($success) {
            $results[$s.Name] = "PASS"
            Write-Host "  OK $($s.Name) completed successfully" -ForegroundColor Green
        }
        else {
            $results[$s.Name] = "FAIL"
            Write-Host "  FAIL $($s.Name) failed" -ForegroundColor Red
        }
    }

    Write-Host ""
    Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
    Write-Host "  |  Phase Complete - Results:                 " -ForegroundColor DarkGray
    Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
    Write-Host ""

    $passCount = 0; $failCount = 0
    foreach ($r in $results.GetEnumerator()) {
        if ($r.Value -eq "PASS") {
            Write-Host "  \u2705 $($r.Key)" -ForegroundColor Green
            $passCount++
        }
        else {
            Write-Host "  \u274c $($r.Key)" -ForegroundColor Red
            $failCount++
        }
    }

    Write-Host ""
    $rColor = if ($failCount -eq 0) { "Green" } else { "Red" }
    Write-Host "  Results: $passCount passed, $failCount failed" -ForegroundColor $rColor
    Write-Host ""
    Write-Host "  Press any key to return to menu..." -NoNewline
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Main loop ------------------------------------------------------------------
function Main {
    $currentPhase = ""
    $running = $true

    while ($running) {
        Clear-Host
            Write-Host ""
            Write-Host "==========================================================" -ForegroundColor Cyan
            Write-Host "       club-3090 Interactive Launcher                     " -ForegroundColor Cyan
            Write-Host "       Native Windows PowerShell Port                     " -ForegroundColor Cyan
            Write-Host "==========================================================" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
            Write-Host "  |  Main Menu                               " -ForegroundColor DarkGray
            Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
            Write-Host ""
            Write-Host "  [1] Status Overview (all scripts + completion tracking)" -ForegroundColor White
            Write-Host "  [2] Pre-flight (verify, health)" -ForegroundColor White
            Write-Host "  [3] GPU Config (NVLink, PCIe-P2P)" -ForegroundColor White
            Write-Host "  [4] Verify (full, stress)" -ForegroundColor White
            Write-Host "  [5] Benchmark (bench-full, all-in-one)" -ForegroundColor White
            Write-Host "  [6] Soak Test (VRAM, TPS retention)" -ForegroundColor White
            Write-Host "  [7] Advanced (concurrency, power-cap)" -ForegroundColor White
            Write-Host "  [8] Specialized (agentic, arch-ab)" -ForegroundColor White
            Write-Host "  [9] Quality (test, baseline, rerun-failed)" -ForegroundColor White
            Write-Host "  [10] Submission (rebench, submit)" -ForegroundColor White
            Write-Host "  [11] Reporting (report, catalog)" -ForegroundColor White
            Write-Host "  [12] Tools (check-syntax, check-issues)" -ForegroundColor White
            Write-Host "  [13] System (capture)" -ForegroundColor White
            Write-Host "  [0] Exit" -ForegroundColor Red
            Write-Host ""
            Write-Host "  Enter choice: " -NoNewline
            $choice = Read-Host

            switch ($choice) {
                "0"  { $running = $false }
                "1"  { Show-StatusOverview }
                "2"  { $currentPhase = "Pre-flight"; Show-PhaseMenu -Phase $currentPhase }
                "3"  { $currentPhase = "GPU Config"; Show-PhaseMenu -Phase $currentPhase }
                "4"  { $currentPhase = "Verify"; Show-PhaseMenu -Phase $currentPhase }
                "5"  { $currentPhase = "Benchmark"; Show-PhaseMenu -Phase $currentPhase }
                "6"  { $currentPhase = "Soak Test"; Show-PhaseMenu -Phase $currentPhase }
                "7"  { $currentPhase = "Advanced"; Show-PhaseMenu -Phase $currentPhase }
                "8"  { $currentPhase = "Specialized"; Show-PhaseMenu -Phase $currentPhase }
                "9"  { $currentPhase = "Quality"; Show-PhaseMenu -Phase $currentPhase }
                "10" { $currentPhase = "Submission"; Show-PhaseMenu -Phase $currentPhase }
                "11" { $currentPhase = "Reporting"; Show-PhaseMenu -Phase $currentPhase }
                "12" { $currentPhase = "Tools"; Show-PhaseMenu -Phase $currentPhase }
                "13" { $currentPhase = "System"; Show-PhaseMenu -Phase $currentPhase }
                default { Write-Host "  Invalid choice" -ForegroundColor Yellow }
            }
        }
    }

# Start
Main
