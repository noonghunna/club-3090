# check-issues.ps1 — scan all ps1 files for common PS5.1 compatibility issues
# Dynamically discovers all .ps1 files in the script directory

$Dir = $PSScriptRoot
$ps1Files = Get-ChildItem -Path $Dir -Filter "*.ps1" | Select-Object -ExpandProperty Name

# Known problematic patterns to flag
$issuesFound = @()

foreach ($name in $ps1Files) {
    $path = Join-Path $Dir $name
    if (-not (Test-Path $path)) { continue }
    $content = Get-Content $path -Raw -Encoding UTF8

    # Check for ${var:pattern} syntax (not standard PS5.1)
    $hasBraceColon = $false
    foreach ($line in $content -split "`n") {
        if ($line -match '\$\{[^}]+:') {
            Write-Host "${name}: dollar-brace-colon pattern found: $($line.Trim())"
            $hasBraceColon = $true
            break
        }
    }

    # Check for /dev/null (Unix path, not valid in Windows PowerShell)
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

    # Check for bash-style $() command substitution that might be problematic
    # PowerShell uses backticks for subexpression, not $()
    # Note: $() IS valid in PS5.1 for subexpression, so we only flag bash-specific patterns
    $bashPatterns = @()
    foreach ($line in $content -split "`n") {
        if ($line -match '^\s*echo\s') { $bashPatterns += "echo (use Write-Host or Write-Output)" }
        if ($line -match '^\s*export\s') { $bashPatterns += "export (use `$env: in PS)" }
        if ($line -match '^\s*local\s') { $bashPatterns += "local keyword (not valid in PS)" }
        if ($line -match '^\s*function\s+\w+\s*\(\s*\)\s*\{') { $bashPatterns += "bash-style function () { } (use function Name in PS)" }
    }
    if ($bashPatterns.Count -gt 0) {
        $uniquePatterns = $bashPatterns | Select-Object -Unique
        Write-Host "${name}: potential bash patterns found: $($uniquePatterns -join ', ')"
    }
}

Write-Host "Check complete. Scanned $($ps1Files.Count) files."
