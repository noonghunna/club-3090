$ErrorActionPreference = "Stop"
$pass = 0
$fail = 0
$files = @(
    "arch-ab.ps1", "beellama-pin-bump.ps1", "bench-agentic.ps1", "bench-full.ps1",
    "capture.ps1", "catalog-baseline.ps1", "concurrency-probe.ps1", "detect_nvlink.ps1",
    "engine-pin-bump.ps1", "health.ps1", "launcher.ps1", "power-cap-sweep.ps1",
    "quality-baseline.ps1", "quality-test.ps1", "rebench-full.ps1", "rebench-runtime.ps1",
    "report.ps1", "rerun-failed-packs.ps1", "run-benchmarks.ps1", "soak-test.ps1",
    "submit-bench.ps1", "verify-full.ps1", "verify-stress.ps1", "verify.ps1"
)

foreach ($f in $files) {
    $e = $null
    $t = $null
    try {
        $null = [System.Management.Automation.Language.Parser]::ParseFile($f, [ref]$t, [ref]$e)
        if ($e.Count -gt 0) {
            $msg = $e[0].Message
            Write-Host ("FAIL: " + $f + " - " + $msg)
            $fail = $fail + 1
        } else {
            Write-Host ("OK: " + $f)
            $pass = $pass + 1
        }
    } catch {
        Write-Host ("FAIL: " + $f + " - " + $_.Exception.Message)
        $fail = $fail + 1
    }
}

Write-Host ""
Write-Host ("Summary: " + $pass + " passed, " + $fail + " failed")
