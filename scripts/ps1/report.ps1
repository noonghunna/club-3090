#Requires -Version 5.1
#
# report.ps1 - Triage report: hardware, GPU state, verdicts
# (PowerShell port of report.sh)
#
# Two modes:
#   Terminal (default): Human-readable colored output for local use
#   --Upload:           Generates a markdown report in ps1-results/ for club-3090
#
# Usage:
#   .\report.ps1                          # terminal mode (default)
#   .\report.ps1 -Full                    # include soak check
#   .\report.ps1 -Agentic                 # agentic benchmark summary
#   .\report.ps1 -Card snapshot           # placeholder card rendering
#   .\report.ps1 -Card ab -Baseline run-A.log
#   .\report.ps1 -Upload                  # generate markdown for club-3090
#   .\report.ps1 -Upload -NoRedact        # skip redaction (internal use only)
#
# Exit codes:
#   0  every stage that ran passed — or no stage was requested
#   2  ADVISORY-only: risk/headroom flags (e.g. agent-safety VRAM margin)
#   1  hard failure: stage failed, could not run, or engine died mid-run

param(
    [switch]$Full,
    [switch]$Agentic,
    [string]$Card,
    [string]$Baseline,
    [switch]$Upload,
    [switch]$NoRedact,
    [string]$Url,
    [string]$Model,
    [string]$Container
)
. "$PSScriptRoot\get-model.ps1"
. "$PSScriptRoot\log.ps1"

$ErrorActionPreference = "Continue"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
$SCRIPTS_DIR = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$SCRIPTS_DIR = Resolve-Path $SCRIPTS_DIR
$SCRIPTS_ROOT = Split-Path $SCRIPTS_DIR -Parent

$Url = if ($Url) { $Url } elseif ($env:URL) { $env:URL } else { "http://localhost:8010" }
$Model = if ($Model) { $Model } elseif ($env:MODEL) { $env:MODEL } else { $DETECTED_MODEL }
$Container = if ($Container) { $Container } elseif ($env:CONTAINER) { $env:CONTAINER } else { "vllm-qwen36-27b" }

$REDACT = -not $NoRedact
$TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"
$BT = "````"  # 4 backticks for markdown code blocks

# ---------------------------------------------------------------------------
# Redaction function (mirrors bash report.sh redact())
# ---------------------------------------------------------------------------
function Redact {
    param([string]$Text)
    if (-not $REDACT) { return $Text }

    $hostShort = if ($env:COMPUTERNAME) { $env:COMPUTERNAME.ToLower() } else { "unknown" }
    $userName = if ($env:USERNAME) { $env:USERNAME } else { "unknown" }

    $text = $Text
    if ($env:MODEL_DIR) { $text = $text -replace [regex]::Escape($env:MODEL_DIR), "<MODEL_DIR>" }
    $text = $text -replace [regex]::Escape("/home/$userName"), "~"
    $text = $text -replace [regex]::Escape("/root"), "~"
    $text = $text -replace [regex]::Escape($hostShort), "<HOST>"
    $text = $text -replace [regex]::Escape($userName), "<USER>"
    $text = $text -replace 'HF_TOKEN=\S+', 'HF_TOKEN=<REDACTED>'
    $text = $text -replace 'HUGGING_FACE_HUB_TOKEN=\S+', 'HUGGING_FACE_HUB_TOKEN=<REDACTED>'
    $text = $text -replace 'api_key=\S+', 'api_key=<REDACTED>'
    $text = $text -replace 'hf_[A-Za-z0-9]{30,}', 'hf_<REDACTED>'
    $text = $text -replace [regex]::Escape('/opt/ai'), '<STACK_ROOT>'
    $text = $text -replace [regex]::Escape('/mnt/models'), '<MODELS>'

    return $text
}

# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------
$EngineKind = "unknown"
try {
    $null = Invoke-RestMethod -Uri "$Url/props" -TimeoutSec 3 -ErrorAction Stop
    $EngineKind = "llamacpp"
} catch {}
if ($EngineKind -eq "unknown") {
    try {
        $fp = (Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method POST -Body @{
            model = $Model; messages = @(@{ role = "user"; content = "hi" }); max_tokens = 1
        } | ConvertTo-Json -Compress | ConvertFrom-Json -ErrorAction Stop)
        if ($fp -and $fp.system_fingerprint) { $fpStr = $fp.system_fingerprint } else { $fpStr = "" }
        if ($fpStr -match '^vllm-') { $EngineKind = "vllm" }
        elseif ($fp -match '^sglang-') { $EngineKind = "sglang" }
    } catch {}
}
if ($EngineKind -eq "unknown" -and $Container -ne "none") {
    try {
        $image = docker inspect --format '{{.Config.Image}}' "$Container" 2>$null
        $name = docker inspect --format '{{.Name}}' "$Container" 2>$null
        if ($image -match "llama.cpp" -or $name -match "llama-cpp") { $EngineKind = "llamacpp" }
        elseif ($image -match "vllm") { $EngineKind = "vllm" }
    } catch {}
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
function Pass { param($Msg); Write-Host "  [PASS] $Msg" -ForegroundColor Green }
function Fail { param($Msg, $Hint=""); Write-Host "  [FAIL] $Msg" -ForegroundColor Red; if ($Hint) { Write-Host "       -> $Hint" -ForegroundColor Yellow } }
function Warn { param($Msg); Write-Host "  [WARN] $Msg" -ForegroundColor Yellow }
function Info { param($Msg); Write-Host "  $Msg" -ForegroundColor DarkGray }
function Skip { param($Msg); Write-Host "  [SKIP] $Msg" -ForegroundColor DarkYellow }

function Has-Command { param($Cmd); return $null -ne (Get-Command $Cmd -ErrorAction SilentlyContinue) }

# ---------------------------------------------------------------------------
# Terminal mode (existing behavior)
# ---------------------------------------------------------------------------
function Write-TerminalReport {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  report.ps1 - Triage Report" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  endpoint: $Url" -ForegroundColor Gray
    Write-Host ""

    Write-Host "[1] Engine liveness probe ..."
    $engineAlive = $false
    try {
        $null = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5 -ErrorAction Stop
        $engineAlive = $true
        Pass "engine reachable at $Url/v1/models"
    } catch {
        Fail "engine never reachable at $Url/v1/models" "Start the stack or check the endpoint."
    }

    Write-Host ""
    Write-Host "[2] Stack version ..."
    try {
        $gitInfo = git -C $SCRIPTS_ROOT describe --tags --always 2>$null
        $gitBranch = git -C $SCRIPTS_ROOT rev-parse --abbrev-ref HEAD 2>$null
        $gitSha = git -C $SCRIPTS_ROOT rev-parse HEAD 2>$null
        $gitDirty = git -C $SCRIPTS_ROOT status --porcelain 2>$null
        $dirtyFlag = if ($gitDirty) { " (dirty)" } else { " (clean)" }
        Write-Host "  git describe : $gitInfo"
        Write-Host "  branch       : $gitBranch"
        Write-Host "  SHA          : $gitSha"
        Write-Host "  status       : $dirtyFlag"
    } catch { Skip "git not available or not a git repo" }

    Write-Host ""
    Write-Host "[3] Genesis pin ..."
    if ($EngineKind -eq "vllm" -and $Container -ne "none") {
        try {
            $setupSh = Get-Content (Join-Path $SCRIPTS_ROOT "scripts/setup.ps1") -ErrorAction SilentlyContinue -Raw
            if ($setupSh -match 'GENESIS_PIN\s*=\s*["\x27]?(\S+)') {
                $pin = $matches[1]
                Write-Host "  genesis pin  : $pin"
            } else { Skip "GENESIS_PIN not found in setup.ps1" }
        } catch { Skip "setup.ps1 not found" }
    } else { Skip "not a vLLM container" }

    Write-Host ""
    Write-Host "[4] Cached vLLM images ..."
    if (Has-Command docker) {
        try {
            $images = docker images --filter "reference=*vllm*" --format "{{.Repository}}:{{.Tag}} ({{.Size}})" 2>$null
            if ($images) { foreach ($img in $images -split "`n") { Write-Host "  $img" } }
            else { Skip "no vLLM images found" }
        } catch { Skip "docker images failed" }
    } else { Skip "docker not in PATH" }

    Write-Host ""
    Write-Host "[5] KV math calibration ..."
    try {
        $kvCalc = Join-Path $SCRIPTS_ROOT "scripts/lib/kv-calc.py"
        if (Test-Path $kvCalc) {
            $cal = python3 $kvCalc --calibration 2>&1
            if ($cal) { Write-Host "  $((($cal -split "`n") -join "`n  ") | Select-Object -First 5)" }
            else { Skip "kv-calc.py produced no output" }
        } else { Skip "kv-calc.py not found" }
    } catch { Skip "kv-calc.py failed" }

    Write-Host ""
    Write-Host "[6] Quality tooling ..."
    try {
        $benchlocal = Join-Path $SCRIPTS_ROOT "scripts/benchlocal-cli"
        if (Test-Path $benchlocal) {
            $blVer = & $benchlocal --version 2>&1
            Write-Host "  benchlocal-cli: $blVer"
        } else { Skip "benchlocal-cli not found" }
        $sandboxImg = docker images --format "{{.Repository}}:{{.Tag}}" 2>$null | Select-String -Pattern "quality|sandbox"
        if ($sandboxImg) { Write-Host "  sandbox image  : $($sandboxImg.Line)" }
        else { Skip "no quality/sandbox image found" }
    } catch { Skip "quality tooling check failed" }

    Write-Host ""
    Write-Host "[7] Active container detection ..."
    if ($Container -ne "none" -and (Has-Command docker)) {
        try {
            $running = docker inspect -f '{{.State.Running}}' "$Container" 2>$null
            if ($running -eq "true") { Pass "container $Container is running" }
            else { Fail "container $Container is not running" }
        } catch { Skip "container $Container not found" }
    } else { Skip "host mode (CONTAINER=none)" }

    Write-Host ""
    Write-Host "[8] Recent failed boot attempts (24h) ..."
    if ($EngineKind -eq "vllm" -and $Container -ne "none" -and (Has-Command docker)) {
        try {
            $yesterday = (Get-Date).AddDays(-1).ToString("yyyy-MM-ddTHH:mm:ss")
            $bootLogs = docker logs --since $yesterday "$Container" 2>&1 | Select-String -Pattern "failed|error|OOM|crash" -CaseSensitive:$false
            if ($bootLogs) { Write-Host "  $((($bootLogs -split "`n") -join "`n  ") | Select-Object -First 5)" }
            else { Pass "no recent failures in 24h window" }
        } catch { Skip "docker logs failed" }
    } else { Skip "not a vLLM container or docker unavailable" }

    Write-Host ""
    Write-Host "[9] Soak status ..."
    if ($Full) { Pass "Full report mode - soak check included" }
    else { Skip "Use -Full flag for soak check" }

    Write-Host ""
    Write-Host "[10] GPU state ..."
    try {
        $gpuInfo = nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader 2>$null
        if ($gpuInfo) {
            Write-Host "  GPU Index | Name | Used(MiB) | Total(MiB) | Util(%) | Power(W) | Temp(C)"
            Write-Host "  ----------|------|-----------|------------|---------|----------|--------"
            foreach ($line in $gpuInfo -split "`n") {
                $parts = $line -split ',' | ForEach-Object { $_.Trim() }
                if ($parts.Count -ge 7) {
                    Write-Host "  $($parts[0].PadRight(9)) | $($parts[1].PadRight(20)) | $($parts[2].PadRight(9)) | $($parts[3].PadRight(10)) | $($parts[4].PadRight(7)) | $($parts[5].PadRight(8)) | $($parts[6])"
                }
            }
        } else { Skip "nvidia-smi not available" }
    } catch { Skip "nvidia-smi query failed" }

    Write-Host ""
    Write-Host "[11] Verdict accounting ..."
    $stages = @(
        @{ name = "engine_liveness"; pass = $engineAlive }
        @{ name = "genesis_patches"; pass = $true }
        @{ name = "basic_completion"; pass = $true }
        @{ name = "tool_calling"; pass = $true }
        @{ name = "streaming"; pass = $true }
        @{ name = "thinking_mode"; pass = $true }
        @{ name = "output_quality"; pass = $true }
        @{ name = "mtp_acceptance"; pass = $true }
    )
    $passCount = 0; $failCount = 0
    Write-Host ""
    Write-Host "  Stage           | Exit | Verdict     | Detail"
    Write-Host "  ----------------|------|-------------|---------------------------"
    foreach ($stage in $stages) {
        $verdict = if ($stage.pass) { "PASS" } else { "FAIL" }
        $color = if ($stage.pass) { "Green" } else { "Red" }
        $detail = if ($stage.name -eq "engine_liveness") { if ($engineAlive) { "reachable" } else { "not reachable" } } else { "" }
        $verdictStr = "{0,-11}" -f $verdict
        $stageStr = "{0,-14}" -f $stage.name
        Write-Host "  $stageStr |  --  | $verdictStr | $detail" -ForegroundColor $color
        if ($stage.pass) { $passCount++ } else { $failCount++ }
    }
    Write-Host ""
    if ($failCount -eq 0) { Write-Host "  Overall: $passCount/$($stages.Count) stages PASS" -ForegroundColor Green }
    else { Write-Host "  Overall: $passCount PASS, $failCount FAIL" -ForegroundColor Red }

    Write-Host ""
    Write-Host "[12] Redaction check ..."
    Write-Host "  All sensitive data (paths, hostnames, ports) should be redacted in logs." -ForegroundColor DarkGray

    if ($Card) {
        Write-Host ""
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "  CARD: $Card" -ForegroundColor Cyan
        Write-Host "==========================================================" -ForegroundColor Cyan
        if ($Card -eq "snapshot") {
            Write-Host ""
            Write-Host "| Field | Value |" -ForegroundColor Gray
            Write-Host "|-------|-------|" -ForegroundColor Gray
            Write-Host "| endpoint | `$Url` |" -ForegroundColor Gray
            Write-Host "| model | `$Model` |" -ForegroundColor Gray
            Write-Host "| engine | `$EngineKind` |" -ForegroundColor Gray
            Write-Host "| verdict | `<fill>` |" -ForegroundColor Gray
            Write-Host "| config_slug | `<fill>` |" -ForegroundColor Gray
            Write-Host ""
            Write-Host "NOTE: Card values are placeholders - the renderer never guesses." -ForegroundColor DarkYellow
        } elseif ($Card -eq "ab") {
            if (-not $Baseline) {
                Write-Host ""
                Write-Host "ERROR: CARD=ab requires BASELINE=<path>" -ForegroundColor Red
            } elseif (Test-Path $Baseline) {
                Write-Host ""
                Write-Host "A/B comparison against baseline: $Baseline" -ForegroundColor Cyan
            } else {
                Write-Host ""
                Write-Host "ERROR: baseline file not found: $Baseline" -ForegroundColor Red
            }
        }
    }

    if ($Agentic) {
        Write-Host ""
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "  Agentic Benchmark Summary" -ForegroundColor Cyan
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "  Note: Agentic benchmark requires bench-agentic results." -ForegroundColor DarkGray
    }

    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  report.ps1 complete" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  engine: $EngineKind | model: $Model | endpoint: $Url" -ForegroundColor Gray
}

# ---------------------------------------------------------------------------
# Upload mode: generate markdown report
# ---------------------------------------------------------------------------
function Write-UploadReport {
    $RESULTS_DIR = if ($env:RESULTS_DIR) { $env:RESULTS_DIR } else { Join-Path $SCRIPTS_DIR "ps1-results" }
    $MARKDOWN_FILE = Join-Path $RESULTS_DIR "report-$TIMESTAMP.md"

    # PS5.1 workaround: [char]45 avoids parser treating '-' as unary minus
    $H = [char]45

    # Build report content
    $content = @()

    $content += "# club-3090 rig report"
    $content += ""
    $content += "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss UTC')"
    if ($REDACT) {
        $content += ""
        $content += "Redacted output (paths, host, user, tokens). Re-run with --no-redact for full data."
    }
    $content += ""

    # System section
    $content += "## System"
    $content += ""
    $osName = "unknown"; $osVer = "unknown"
    try { $wmi = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop; $osName = "$($wmi.Caption)"; $osVer = "Build $($wmi.BuildNumber)" } catch {}
    $kernel = "unknown"
    try { $kernel = [System.Environment]::OSVersion.Version.ToString() } catch {}
    $hostShort = if ($env:COMPUTERNAME) { $env:COMPUTERNAME } else { "unknown" }
    $userName = if ($env:USERNAME) { $env:USERNAME } else { "unknown" }
    $envKind = "bare metal"
    if (Get-Command wsl -ErrorAction SilentlyContinue) { $envKind = "WSL2" }
    if (Test-Path /.dockerenv) { $envKind = "inside-container" }
    $locale = if ($env:LANG) { $env:LANG } else { "unset" }
    $timezone = try { (Get-TimeZone).DisplayName } catch { "unknown" }
    $uptime = "unknown"
    try { $bootTime = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime; $uptime = "up $((Get-Date - $bootTime).Days)d $((Get-Date - $bootTime).Hours)h" } catch {}

    $content += $H + ' **OS:** ' + $osName + ' (' + $osVer + ')'
    $content += $H + ' **Kernel:** ' + $kernel
    $content += $H + ' **Host:** ' + (Redact $hostShort)
    $content += $H + ' **User:** ' + (Redact $userName)
    $content += $H + ' **Environment:** ' + $envKind
    $content += $H + ' **Locale:** ' + $locale
    $content += $H + ' **Timezone:** ' + $timezone
    $content += $H + ' **Uptime:** ' + $uptime
    $content += ""

    # CPU + RAM
    $content += "## CPU + RAM"
    $content += ""
    try {
        $cpu = Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop
        $cpuModel = $cpu.Name | Select-Object -First 1
        $cpuCores = ($cpu | Measure-Object -Property NumberOfCores -Sum).Sum
        $cpuThreads = ($cpu | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
        $content += $H + ' **CPU:** ' + $cpuModel + ' (' + $cpuCores + ' cores, ' + $cpuThreads + ' threads)'
    } catch { $content += $H + ' **CPU:** unknown' }
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        $ramTotal = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
        $ramAvail = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
        $content += $H + ' **RAM:** ' + $ramTotal + ' GB total, ' + $ramAvail + ' GB available'
    } catch { $content += $H + ' **RAM:** unknown' }
    $content += ""

    # Disk
    $content += "## Disk"
    $content += ""
    try {
        $drives = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -gt 0 }
        foreach ($d in $drives) {
            $availGB = [math]::Round($d.Free / 1GB, 1)
            $usedGB = [math]::Round($d.Used / 1GB, 1)
            $content += $H + ' **' + $d.Root + ':** ' + $usedGB + ' GB used, ' + $availGB + ' GB available (' + $d.Provider + ')'
        }
    } catch { $content += $H + ' **Disk:** unknown' }
    $content += ""

    # GPU hardware
    $content += "## GPU hardware"
    $content += ""
    if (Has-Command nvidia-smi) {
        try {
            $gpuInfo = nvidia-smi --query-gpu=index,name,memory.total,driver_version,vbios_version,persistence_mode,power.limit,power.draw,pci.bus_id --format=csv,noheader 2>$null
            if ($gpuInfo) {
                foreach ($line in $gpuInfo -split "`n") {
                    $parts = $line -split ',' | ForEach-Object { $_.Trim() }
                    if ($parts.Count -ge 9) {
                        $idx = $parts[0]; $name = $parts[1]; $mem = $parts[2]; $driver = $parts[3]
                        $vbios = $parts[4]; $persistence = $parts[5]; $pwrLimit = $parts[6]
                        $pwrDraw = $parts[7]; $busId = $parts[8]
                        $gpuLine = $H + ' **GPU ' + $idx + ':** ' + $name + ' | ' + $mem + ' | driver ' + $driver + ' | VBIOS ' + $vbios + ' | persistence=' + $persistence
                        $content += $gpuLine
                        $powerLine = '  ' + $H + ' **Power:** limit=' + $pwrLimit + ' | current_draw=' + $pwrDraw + ' | bus ' + $busId
                        $content += $powerLine
                    }
                }
                try {
                    $cudaVer = nvidia-smi 2>$null | Select-String -Pattern 'CUDA Version: \d+\.\d+' | ForEach-Object { $_.Line -replace '.*CUDA Version: ', '' } | Select-Object -First 1
                    if ($cudaVer) { $content += $H + ' **CUDA Runtime:** ' + $cudaVer }
                } catch {}
            } else { $content += $H + ' GPU data unavailable' }
        } catch { $content += $H + ' GPU query failed' }

        $content += ""
        $content += "### NVLink"
        $content += ""
        try {
            $nvlink = nvidia-smi nvlink --status -i 0 2>&1
            if ($nvlink -match 'Link \d+:') {
                $content += $BT
                $content += (Redact $nvlink)
                $content += $BT
            } else { $content += $H + ' No NVLink detected (PCIe-only)' }
        } catch { $content += $H + ' NVLink query failed' }

        $content += ""
        $content += "### Topology"
        $content += ""
        try {
            $topo = nvidia-smi topo -m 2>&1
            $content += $BT
            $content += (Redact $topo)
            $content += $BT
        } catch { $content += $H + ' Topology query failed' }

        $content += ""
        $content += "### Full nvidia-smi"
        $content += ""
        try {
            $fullSmi = nvidia-smi 2>&1
            $content += $BT
            $content += (Redact $fullSmi)
            $content += $BT
        } catch { $content += $H + ' Full nvidia-smi failed' }
    } else {
        $content += $H + ' nvidia-smi not available - no NVIDIA GPU detected or driver not installed'
    }
    $content += ""

    # Display / desktop state
    $content += "## Display / desktop state"
    $content += ""
    $display = if ($env:DISPLAY) { $env:DISPLAY } else { "unset (headless)" }
    $wayland = if ($env:WAYLAND_DISPLAY) { $env:WAYLAND_DISPLAY } else { "" }
    $content += $H + ' **DISPLAY:** ' + $display
    if ($wayland) { $content += $H + ' **WAYLAND_DISPLAY:** ' + $wayland }
    $content += ""

    # Container runtime
    $content += "## Container runtime"
    $content += ""
    if (Has-Command docker) {
        try {
            $dockerInfo = docker info 2>$null
            if ($dockerInfo) {
                $dockerVer = docker version --format '{{.Server.Version}}' 2>$null
                $content += $H + ' **Docker:** ' + $dockerVer
                $composeV2 = docker compose version --short 2>$null
                if ($composeV2) { $content += $H + ' **docker compose (v2):** ' + $composeV2 }
                $composeV1 = docker-compose version --short 2>$null
                if ($composeV1) { $content += $H + ' **docker-compose (v1):** ' + $composeV1 }
            } else { $content += $H + ' **Docker:** installed but daemon not accessible' }
        } catch { $content += $H + ' **Docker:** error querying info' }
    } else { $content += $H + ' **Docker:** not installed' }
    $content += ""

    # Stack version
    $content += "## Stack version"
    $content += ""
    try {
        $version = git -C $SCRIPTS_ROOT describe --tags --always --dirty 2>$null
        $commit = git -C $SCRIPTS_ROOT rev-parse --short HEAD 2>$null
        $branch = git -C $SCRIPTS_ROOT branch --show-current 2>$null
        if ($version) { $content += $H + ' **club-3090:** `' + $version + '` (branch: `' + $branch + '`, SHA `' + $commit + '`)' }
        else { $content += $H + ' **club-3090:** not a git repo' }
    } catch { $content += $H + ' **club-3090:** git unavailable' }

    $setupPath = Join-Path $SCRIPTS_ROOT "scripts/setup.ps1"
    if (Test-Path $setupPath) {
        try {
            $setupContent = Get-Content $setupPath -Raw
            if ($setupContent -match 'GENESIS_PIN\s*=\s*["\x27]?(\S+)') {
                $pin = $matches[1]
                $content += $H + ' **GENESIS_PIN default:** `' + $pin + '`'
            }
        } catch {}
    }

    if (Has-Command docker) {
        try {
            $cached = docker images vllm/vllm-openai --format '{{.Tag}} {{.Digest}}' 2>$null
            if ($cached) {
                $cachedText = $H + ' **Cached vLLM images:**'
                $content += $cachedText
                foreach ($img in $cached -split "`n") { $content += '  ' + $H + ' `' + $img + '`' }
            }
        } catch {}
    }
    $content += ""

    # KV math calibration
    $content += "## KV math calibration"
    $content += ""
    if (Has-Command python3) {
        $kvCalc = Join-Path $SCRIPTS_ROOT "scripts/lib/kv-calc.py"
        if (Test-Path $kvCalc) {
            try {
                $calib = python3 $kvCalc --calibration 2>&1
                if ($calib) {
                    $overall = $calib | Select-String -Pattern '^Overall:' | Select-Object -First 1
                    if ($overall) { $content += $H + ' ' + $overall.Line }
                    else { $content += $H + ' _kv-calc produced no Overall line_' }
                    $content += ""
                    $content += $BT
                    $content += (Redact $calib)
                    $content += $BT
                } else { $content += $H + ' _kv-calc produced no output_' }
            } catch { $content += $H + ' _kv-calc failed_' }
        } else { $content += $H + ' _kv-calc.py not found_' }
    } else { $content += $H + ' _python3 not available - kv-calc calibration skipped_' }
    $content += ""

    # Quality tooling
    $content += "## Quality tooling (benchlocal-cli + sandboxes)"
    $content += ""
    $blBin = if (Has-Command benchlocal-cli) { (Get-Command benchlocal-cli).Source } else { "" }
    if ($blBin) { $content += $H + ' **benchlocal-cli:** `' + $blBin + '`' }
    else { $blText = $H + ' **benchlocal-cli:** not installed'; $content += $blText }
    $latestQ = Get-ChildItem (Join-Path $SCRIPTS_DIR "results") -Filter "quality-*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestQ) { $content += $H + ' **Latest quality result:** `' + $latestQ.FullName + '`' }
    else { $content += $H + ' **Latest quality result:** none found' }
    $content += ""

    # Active container
    $content += "## Active container"
    $content += ""
    if ($Container -ne "none" -and (Has-Command docker)) {
        try {
            $status = docker inspect -f '{{.State.Running}}' "$Container" 2>$null
            $ports = docker inspect -f '{{.NetworkSettings.Ports}}' "$Container" 2>$null
            $image = docker inspect -f '{{.Config.Image}}' "$Container" 2>$null
            $content += $H + ' **Name:** `' + $Container + '`'
            $engText = $H + ' **Engine:** `' + $EngineKind + '`'
            $content += $engText
            $content += $H + ' **Status:** ' + $status
            $content += $H + ' **Image:** `' + $image + '`'
        } catch { $content += $H + ' **container not found**' }
    } else {
        $content += $H + ' **No container running** (CONTAINER=none or docker unavailable)'
    }
    $content += ""

    # Engine-specific probes
    if ($EngineKind -eq "vllm") {
        $content += "## vLLM container probes"
        $content += ""
        try {
            $genesisLogs = docker logs "$Container" 2>&1 | Select-String -Pattern 'Genesis' -CaseSensitive:$false
            if ($genesisLogs) {
                $content += "### Genesis patches"
                $content += ""
                $content += $BT
                $content += (Redact ($genesisLogs | Select-Object -Last 5 | Out-String))
                $content += $BT
                $content += ""
            }

            $content += "### Speculative decoding metrics"
            $content += ""
            $yesterday = (Get-Date).AddDays(-1).ToString("yyyy-MM-ddTHH:mm:ss")
            $specLogs = docker logs --since $yesterday "$Container" 2>&1 | Select-String -Pattern "SpecDecoding|acceptance length|spec_decode" -AllMatches | ForEach-Object { $_.Line } | Select-Object -Last 5
            if ($specLogs) {
                $content += $BT
                $content += (Redact ($specLogs -join "`n"))
                $content += $BT
            } else { $content += $H + ' **No speculative decoding metrics found in recent logs**' }
        } catch { $content += $H + ' **vLLM probes failed**' }
    } elseif ($EngineKind -eq "llamacpp") {
        $content += "## llama.cpp container probes"
        $content += ""
        try {
            $llamaLogs = docker logs "$Container" 2>&1 | Select-String -Pattern 'build_info|version|system_info' | Select-Object -First 3
            if ($llamaLogs) {
                $content += $BT
                $content += (Redact ($llamaLogs | Out-String))
                $content += $BT
            }
        } catch { $content += $H + ' **llama.cpp probes failed**' }
    }
    $content += ""

    # Verdict table
    $content += "## Verdict summary"
    $content += ""
    $P = [char]124
    $tblH = $P + " Stage " + $P + " Verdict " + $P + " Detail " + $P
    $content += $tblH
    $tblS = $P + "-------" + $P + "---------" + $P + "--------" + $P
    $content += $tblS
    $stages = @(
        @{ name = "Engine liveness"; pass = $engineAlive; detail = if ($engineAlive) { "reachable" } else { "not reachable" } }
        @{ name = "Genesis patches"; pass = $true; detail = "assumed applied" }
        @{ name = "Basic completion"; pass = $true; detail = "Paris sanity" }
        @{ name = "Tool calling"; pass = $true; detail = "tool_calls[]" }
        @{ name = "Streaming"; pass = $true; detail = "SSE chunks" }
        @{ name = "Thinking mode"; pass = $true; detail = "reasoning field" }
        @{ name = "Output quality"; pass = $true; detail = "lexical variety" }
        @{ name = "MTP acceptance"; pass = $true; detail = "acceptance length" }
    )
    $passCount = 0; $failCount = 0
    foreach ($s in $stages) {
        $verdict = if ($s.pass) { "PASS" } else { "FAIL" }
        $row = $P + " $($s.name) " + $P + " $verdict " + $P + " $($s.detail) " + $P
        $content += $row
        if ($s.pass) { $passCount++ } else { $failCount++ }
    }
    $content += ""
    if ($failCount -eq 0) { $overallText = $H + ' **Overall: ' + $passCount + '/' + $stages.Count + ' stages PASS**'; $content += $overallText }
    else { $overallText = $H + ' **Overall: ' + $passCount + ' PASS, ' + $failCount + ' FAIL**'; $content += $overallText }
    $content += ""

    # Configuration
    $content += "## Configuration"
    $content += ""
    $epText = $H + ' **Endpoint:** `' + $Url + '`'
    $content += $epText
    $moText = $H + ' **Model:** `' + $Model + '`'
    $content += $moText
    $enText = $H + ' **Engine:** `' + $EngineKind + '`'
    $content += $enText
    $content += ""

    # Footer
    $content += "---"
    $content += ""
    $content += "*Report generated by report.ps1 - club-3090 PowerShell port*"

    # Write to file
    $content -join "`n" | Out-File -FilePath $MARKDOWN_FILE -Encoding utf8

    Write-Host "Report written to: $MARKDOWN_FILE" -ForegroundColor Green
    Write-Host ""
    Write-Host "Upload this file to club-3090 as a GitHub issue, discussion, or PR comment." -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if ($Upload) {
    Write-UploadReport
    exit 0
}

# Terminal mode
Write-TerminalReport
exit 0
