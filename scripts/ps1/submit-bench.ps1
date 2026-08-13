# Requires -Version 5.1
#
# submit-bench.ps1 — Generate or submit a BENCHMARKS.md row from results/rebench/<tag>/
#
# Usage:
#   .\submit-bench.ps1 -Tag <tag>
#   .\submit-bench.ps1 -Tag <tag> -AutoSubmit
#   .\submit-bench.ps1 -Tag <tag> -AutoSubmit -AsPr
#
# Produces: results/rebench/<tag>/BENCHMARKS-row.md
# Auto-submit: creates a GitHub issue or PR via gh CLI

param(
    [Parameter(Mandatory=$true)]
    [string]$Tag,

    [switch]$AutoSubmit,
    [switch]$AsPr,
    [string]$Section
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\get-model.ps1"

$ScriptName = "submit-bench"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$TagDir = Join-Path $RepoRoot "ps1-results/rebench/$Tag"

function Log { param($Msg) Write-Host "[$ScriptName] $Msg" }
function Die { param($Msg); Write-Error "[$ScriptName] ERROR: $Msg"; exit 1 }

# Validate tag dir and required artifacts
if (-not (Test-Path $TagDir)) { Die "tag dir not found: $TagDir" }

$RequiredFiles = @("_internal.json", "container-config.json", "rig.txt")
foreach ($f in $RequiredFiles) {
    $path = Join-Path $TagDir $f
    if (-not (Test-Path $path)) { Die "missing required artifact: $path" }
}

# Check for REPORT.md (required)
$reportPath = Join-Path $TagDir "REPORT.md"
if (-not (Test-Path $reportPath)) {
    # Also check bench.log as alternative
    $logPath = Join-Path $TagDir "bench.log"
    if (-not (Test-Path $logPath)) {
        Die "missing required artifact: REPORT.md or bench.log"
    }
}

# ---- Generate the BENCHMARKS row ----
# The row format is a markdown table row. We extract key metrics from
# _internal.json and produce a formatted row.

$InternalPath = Join-Path $TagDir "_internal.json"
$Internal = Get-Content $InternalPath -Raw -Encoding UTF8 | ConvertFrom-Json

$Bench = $Internal.bench
if (-not $Bench) { Die "_internal.json has no 'bench' section" }

$Narr = $Bench.narrative
$Code = $Bench.code
if (-not $Narr -or -not $Code) { Die "bench section missing narrative or code" }

# Extract metrics
$narrTps = if ($Narr.decode_tps_mean) { $Narr.decode_tps_mean.ToString("F2") } else { "N/A" }
$codeTps = if ($Code.decode_tps_mean) { $Code.decode_tps_mean.ToString("F2") } else { "N/A" }
$ttft = if ($Narr.ttft_ms_mean) { [math]::Round($Narr.ttft_ms_mean) } else { "N/A" }

# Get rig info from rig.txt
$RigText = Get-Content $reportPath -Raw -Encoding UTF8
$RigShort = ""
if ($RigText) {
    # Extract GPU info from rig.txt
    $gpuMatch = $RigText | Select-String -Pattern "GPU \d+: (.+?) \(" -AllMatches
    if ($gpuMatch) {
        $RigShort = ($gpuMatch.Matches[0].Groups[1].Value).Trim()
    }
}

# Build the row
$Row = "| $RigShort | $narrTps | $codeTps | ${ttft}ms | | | |"

$OutputPath = Join-Path $TagDir "BENCHMARKS-row.md"
$Row | Out-File -FilePath $OutputPath -Encoding UTF8

Log "Generated BENCHMARKS row for tag: $Tag"
Log "Wrote: $OutputPath"
Write-Host ""

# ---- Determine section ----
$SectionOverride = $Section
if (-not $SectionOverride) {
    # Default to the first section in BENCHMARKS.md that has a table
    $BenchmarksPath = Join-Path $RepoRoot "BENCHMARKS.md"
    if (Test-Path $BenchmarksPath) {
        $BenchmarksContent = Get-Content $BenchmarksPath -Raw -Encoding UTF8
        # Find first ## or ### section that has a markdown table
        $lines = $BenchmarksContent -split "`n"
        $currentSection = ""
        foreach ($line in $lines) {
            if ($line -match '^#{1,3} (.+)') {
                $currentSection = $Matches[1].Trim()
            }
            if ($line.StartsWith("|") -and $line.Contains("|") -and $currentSection) {
                $SectionOverride = $currentSection
                break
            }
        }
    }
}
if (-not $SectionOverride) { $SectionOverride = "Single-Card" }

Log "Section: $SectionOverride"

# ---- Auto-submit flow ----
if (-not $AutoSubmit) {
    Write-Host @"
Inspect at $OutputPath. Three ways to land it (recommended order):

  1. Issue + maintainer integrates (preferred — vetting before merge):
       .\submit-bench.ps1 -Tag $Tag -AutoSubmit
     Or, no-gh-needed:
       https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml
       — paste the contents of $OutputPath + $TagDir/rig.txt into the body

  2. Direct PR (advanced — for contributors who know the matrix structure):
       .\submit-bench.ps1 -Tag $Tag -AutoSubmit -AsPr
     Note: matrix is hand-curated; direct PRs may get redirected to an
     issue thread for context-gathering before merge.

  3. Manual edit (zero tools):
       Paste the row from $OutputPath into BENCHMARKS.md via the GitHub web editor.
"@
    exit 0
}

# Check gh CLI
$ghExists = Get-Command gh -ErrorAction SilentlyContinue
if (-not $ghExists) { Die "'gh' not found. Install GitHub CLI or submit manually." }

try { gh auth status | Out-Null } catch { Die "not authed with gh. Run: gh auth login" }

# Get GitHub user
$ghUser = ""
try {
    $ghUser = gh api user --jq .login 2>$null | Select-String -Pattern '.' | Select-Object -First 1
    if ($ghUser) { $ghUser = $ghUser.ToString().Trim() }
} catch { $ghUser = "" }

if (-not $ghUser) { Die "could not resolve GitHub user for row attribution" }

# Build PR/Issue title
$RigShortName = if ($RigShort) { $RigShort } else { $Tag }
$BranchUser = ($ghUser -replace '[^a-zA-Z0-9_.-]', '')
$BranchTag = ($Tag -replace '[^a-zA-Z0-9_.-]', '')
$Branch = "bench/${BranchUser}-${BranchTag}"

$PrTitle = "bench(matrix): @$ghUser $RigShortName"
$IssueTitle = "[bench] @$ghUser $RigShortName"

# Write PR or Issue body
$BodyFile = Join-Path $TagDir "PR-body.md"
if ($AsPr) {
    # PR body
    @"
## Rig bench submission

### New row

$Row

### Full results

See \`ps1-results/rebench/$Tag/REPORT.md\`.
"@ | Out-File -FilePath $BodyFile -Encoding UTF8
} else {
    # Issue body
    @"
**Compose / section**: \`$SectionOverride\`

**Rig**:

```text
$(Get-Content (Join-Path $TagDir "rig.txt") -Raw)
```

**Proposed BENCHMARKS.md row**:

$Row

**Full report**: \`ps1-results/rebench/$Tag/REPORT.md\`

**Generated row file**: \`ps1-results/rebench/$Tag/BENCHMARKS-row.md\`
"@ | Out-File -FilePath $BodyFile -Encoding UTF8
}

# Execute submit
if ($AsPr) {
    Log "Creating PR: $PrTitle"
    git fetch origin master 2>$null
    git switch -c $Branch origin/master 2>$null
    if ($LASTEXITCODE -ne 0) {
        git switch -c $Branch 2>$null
    }
    # Insert row into BENCHMARKS.md
    $BenchmarksPath = Join-Path $RepoRoot "BENCHMARKS.md"
    if (Test-Path $BenchmarksPath) {
        $bmContent = Get-Content $BenchmarksPath -Raw -Encoding UTF8
        $bmLines = $bmContent -split "`n"
        $sectionLine = $bmLines | Where-Object { $_ -match "^(##|###) $([regex]::Escape($SectionOverride))" }
        if ($sectionLine) {
            $sectionIdx = $bmLines.IndexOf($sectionLine)
            # Find the table after this section
            $tableStart = $null
            for ($i = $sectionIdx + 1; $i -lt $bmLines.Count; $i++) {
                if ($bmLines[$i] -match '^#') { break }
                if ($bmLines[$i].StartsWith("|")) { $tableStart = $i; break }
            }
            if ($tableStart) {
                # Find insert point (end of table)
                $insertAt = $tableStart
                for ($i = $tableStart; $i -lt $bmLines.Count; $i++) {
                    if ($bmLines[$i] -match '^#') { break }
                    if ($bmLines[$i].StartsWith("|")) { $insertAt = $i + 1 }
                }
                $newLines = $bmLines[0..($insertAt-1)] + $Row + $bmLines[$insertAt..($bmLines.Count-1)]
                $tmpFile = "$BenchmarksPath.submit-bench.tmp"
                ($newLines -join "`n") | Out-File -FilePath $tmpFile -Encoding UTF8
                Move-Item $tmpFile $BenchmarksPath -Force
                git add $BenchmarksPath
                git commit -m "$PrTitle" 2>$null
            }
        }
    }
    git push -u origin $Branch 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "git push to origin failed (exit code $LASTEXITCODE)"
        exit 1
    }
    $prUrl = gh pr create -t "$PrTitle" -F $BodyFile 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "gh pr create failed (exit code $LASTEXITCODE)"
        exit 1
    }
    Log "Opened PR: $prUrl"
} else {
    $issueUrl = gh issue create -t "$IssueTitle" -F $BodyFile -l bench-contribution 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "gh issue create failed (exit code $LASTEXITCODE)"
        exit 1
    }
    Log "Opened issue: $issueUrl"
}
