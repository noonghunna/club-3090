# ---------------------------------------------------------------------------
# log.ps1 — Shared logging helper for all benchmark scripts
# Dot-source this at the top of any script that needs file logging.
#
# Usage:
#   . "$PSScriptRoot\log.ps1"
#   Write-Log "message"           # prints to stdout + appends to log file
#   Write-Log "message" -NoNewline  # suppress trailing newline
#
# Creates: results/<script>-<timestamp>/script.log + summary.json
# ---------------------------------------------------------------------------

function Write-Log {
    param(
        [string]$Text,
        [switch]$NoNewline
    )

    if ($NoNewline) {
        Write-Host -NoNewline $Text
    } else {
        Write-Host $Text
    }

    if ($LogFilePath) {
        $Text | Out-File -FilePath $LogFilePath -Append -Encoding utf8
    }
}

function Init-Logging {
    param(
        [string]$ScriptName,
        [string]$Tag
    )

    $PS1_DIR = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    $PS1_DIR = Resolve-Path $PS1_DIR
    $SCRIPTS_ROOT = Split-Path $PS1_DIR -Parent
    $RESULTS_DIR = if ($env:RESULTS_DIR) { $env:RESULTS_DIR } else { Join-Path $SCRIPTS_ROOT "ps1-results" }
    $TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"

    if ($Tag) {
        $RUN_DIR = Join-Path $RESULTS_DIR "$Tag"
    } else {
        $RUN_DIR = Join-Path $RESULTS_DIR "$ScriptName-$TIMESTAMP"
    }

    New-Item -ItemType Directory -Force -Path $RUN_DIR | Out-Null
    $script:LogFilePath = Join-Path $RUN_DIR "$ScriptName.log"

    # Create summary.json skeleton
    $Summary = @{
        script   = $ScriptName
        tag      = if ($Tag) { $Tag } else { "none" }
        date     = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        checks   = @()
        status   = "running"
    }
    $Summary | ConvertTo-Json -Depth 5 | Out-File -FilePath (Join-Path $RUN_DIR "summary.json") -Encoding utf8

    return @{
        RunDir  = $RUN_DIR
        LogFile = $script:LogFilePath
        Summary = $Summary
    }
}

function Update-Summary {
    param(
        [string]$Name,
        [string]$Status,
        [string]$Detail = "",
        [object]$Data = $null
    )

    if (-not $SummaryFile) { return }

    $runDir = Split-Path $SummaryFile -Parent
    $summary = Get-Content $SummaryFile | ConvertFrom-Json

    # Add check entry
    $check = @{
        name   = $Name
        status = $Status
        detail = $Detail
        time   = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    }
    if ($Data) { $check.data = $Data }

    if (-not $summary.checks) { $summary | Add-Member -NotePropertyName "checks" -NotePropertyValue @() }
    $summary.checks += $check

    $summary | ConvertTo-Json -Depth 5 | Out-File -FilePath $SummaryFile -Encoding utf8
}

function Finalize-Summary {
    param(
        [string]$Status = "completed"
    )

    if (-not $SummaryFile) { return }

    $summary = Get-Content $SummaryFile | ConvertFrom-Json
    $summary.status = $Status
    $summary.completed = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    $summary | ConvertTo-Json -Depth 5 | Out-File -FilePath $SummaryFile -Encoding utf8
}
