$ErrorActionPreference = "Stop"
$SCRIPTS_DIR = Split-Path $MyInvocation.MyCommand.Path -Parent
if (-not $SCRIPTS_DIR) { $SCRIPTS_DIR = $PSScriptRoot }
$pass = 0
$fail = 0
$files = Get-ChildItem "$SCRIPTS_DIR/*.ps1" | Sort-Object Name
foreach ($file in $files) {
    $errors = $null
    $tokens = $null
    try {
        [System.Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$errors)
        if ($errors.Count -gt 0) {
            Write-Host ("FAIL: " + $file.Name)
            $fail = $fail + 1
        } else {
            Write-Host ("OK: " + $file.Name)
            $pass = $pass + 1
        }
    } catch {
        Write-Host ("FAIL: " + $file.Name + " - " + $_.Exception.Message)
        $fail = $fail + 1
    }
}
Write-Host ""
Write-Host ("Summary: " + $pass + " passed, " + $fail + " failed")
