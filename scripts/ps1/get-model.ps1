# ---------------------------------------------------------------------------
# get-model.ps1 — Shared model detection for all benchmark scripts
# ---------------------------------------------------------------------------
# Sources: $env:MODEL > model.json cache > /v1/models probe > fallback
# Dot-source this file in any script to get $DETECTED_MODEL.
#
# Usage: . "$PSScriptRoot\get-model.ps1"

$MODEL_CACHE = Join-Path $PSScriptRoot "model.json"

function Get-Model {
    # 1. Explicit env var wins
    if ($env:MODEL) { return $env:MODEL }

    # 2. Cached detection
    if (Test-Path $MODEL_CACHE) {
        $cached = Get-Content $MODEL_CACHE | ConvertFrom-Json
        if ($cached.model) { return $cached.model }
    }

    # 3. Probe the server
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8010/v1/models" -TimeoutSec 5
        $model = $resp.data[0].id
        @{ model = $model } | ConvertTo-Json | Set-Content $MODEL_CACHE
        return $model
    } catch {}

    # 4. Last resort fallback
    return "qwen-8010"
}

$DETECTED_MODEL = Get-Model
