# check-issues.ps1 — scan all ps1 files for common PS5.1 compatibility issues

$FailList = @(
    "arch-ab.ps1", "beellama-pin-bump.ps1", "bench.ps1",
    "catalog-baseline.ps1", "detect_nvlink.ps1",
    "power-cap-sweep.ps1", "quality-baseline.ps1", "rebench-full.ps1",
    "rerun-failed-packs.ps1", "run-benchmarks.ps1", "verify-full.ps1", "verify-stress.ps1"
)

$Dir = "G:/scripts/ps1"

foreach ($name in $FailList) {
    $path = Join-Path $Dir $name
    if (-not (Test-Path $path)) { continue }
    $content = Get-Content $path -Raw -Encoding UTF8

    # Check for ${var:pattern} syntax (not standard PS5.1)
    $braceColon = $false
    foreach ($line in $content -split "`n") {
        if ($line -match '\$\{[^}]+:') {
            Write-Host "${name}: dollar-brace-colon pattern found: $($line.Trim())"
            $braceColon = $true
            break
        }
    }

    # Check for /dev/null
    $devNull = 0
    foreach ($line in $content -split "`n") {
        $devNull += ($line -split '/dev/null').Count - 1
    }
    if ($devNull -gt 0) {
        Write-Host "${name}: /dev/null occurrences: $devNull"
    }

    # Check for bash-style $((...)) arithmetic subshell
    # Only flag actual bash arithmetic: $((1+2)) or $((var))
    # NOT $('(' which is a valid PS subexpression with a string
    # Strategy: find all "$((" occurrences, then check if the char after is NOT a quote
    $subshell = 0
    $pos = 0
    while ($true) {
        $dollarIdx = $content.IndexOf('$',$pos)
        if ($dollarIdx -lt 0) { break }
        if ($dollarIdx + 2 -lt $content.Length -and $content[$dollarIdx+1] -eq '(' -and $content[$dollarIdx+2] -eq '(') {
            # Check what follows — if it's a quote, it's $('(' which is valid PS
            if ($dollarIdx + 3 -lt $content.Length -and $content[$dollarIdx+3] -eq "'") {
                # This is a PS subexpression $('(' — skip
            } else {
                $subshell++
            }
        }
        $pos = $dollarIdx + 1
    }
    if ($subshell -gt 0) {
        Write-Host "${name}: bash-dbl-paren arithmetic subshell occurrences: $subshell"
    }
}

Write-Host "Check complete."
