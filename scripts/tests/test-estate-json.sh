#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

assert_json_keys() {
  # assert_json_keys <json> <jq-filter> <expected>
  local json="$1"
  local filter="$2"
  local expected="$3"
  local got
  got="$(printf '%s' "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print($filter)")"
  if [[ "$got" != "$expected" ]]; then
    echo "ASSERTION FAILED: $filter expected [$expected] got [$got]" >&2
    echo "--- output ---" >&2
    echo "$json" >&2
    exit 1
  fi
}

assert_valid_json() {
  if ! printf '%s' "$1" | python3 -c "import json,sys; json.load(sys.stdin)" >/dev/null 2>&1; then
    echo "ASSERTION FAILED: output is not valid JSON" >&2
    echo "--- output ---" >&2
    echo "$1" >&2
    exit 1
  fi
}

export CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6,1:RTX_3090:24576:8.6'

GOOD="${TMP_DIR}/estate-good.yml"
cat > "$GOOD" <<'YAML'
schema_version: 1
created: 2026-05-14T00:00:00Z
rig:
  hardware_id: rtx-3090
  gpu_count: 2
  nvlink_active: false
estate:
  - name: qwen-left
    compose: llamacpp/default
    gpus: [0]
    port: 8110
  - name: qwen-right
    compose: llamacpp/default
    gpus: [1]
    port: 8120
YAML

CLI="${ROOT_DIR}/scripts/lib/profiles/estate_cli.py"

# --- report-state --json: valid JSON with the expected top-level keys ---
out="$(python3 "$CLI" report-state --file "$GOOD" --json 2>&1)"
assert_valid_json "$out"
assert_json_keys "$out" "d['profile_schema_version']" "1"
assert_json_keys "$out" "sorted(d.keys())==['active_estate','calibration','canonical_scenarios','compose_registry_entries','profile_counts','profile_schema_version']" "True"
assert_json_keys "$out" "d['active_estate']['present']" "True"
assert_json_keys "$out" "d['active_estate']['instance_count']" "2"
assert_json_keys "$out" "d['active_estate']['instances'][0]['name']" "qwen-left"
assert_json_keys "$out" "d['active_estate']['instances'][0]['compose']" "llamacpp/default"
assert_json_keys "$out" "d['active_estate']['instances'][0]['gpus']==[0]" "True"
# F7 — per-instance LIVENESS: estate.yml is a plan; report-state probes docker
# per instance. The test instances are never booted → running is False (probed,
# down) or None (docker unavailable in this env) — NEVER True. Consumers treat
# only running==False as "not a claim" (fail closed on None/missing).
assert_json_keys "$out" "d['active_estate']['instances'][0]['running'] in (False, None)" "True"
assert_json_keys "$out" "d['active_estate']['instances'][0]['container']" "club3090-qwen-left"
assert_json_keys "$out" "d['active_estate']['running_count']" "0"
assert_json_keys "$out" "sorted(d['profile_counts'].keys())==['drafters','engines','hardware','models','workloads']" "True"

# report-state --json with a missing estate file
MISSING="${TMP_DIR}/does-not-exist.yml"
out="$(python3 "$CLI" report-state --file "$MISSING" --json 2>&1)"
assert_valid_json "$out"
assert_json_keys "$out" "d['active_estate']['present']" "False"

# --- diagnose --json: valid JSON with expected check structure ---
out="$(python3 "$CLI" diagnose "$GOOD" --json 2>&1)"
assert_valid_json "$out"
assert_json_keys "$out" "sorted(d.keys())==['checks','estate_file','live','summary','valid']" "True"
assert_json_keys "$out" "d['summary']" "GREEN"
assert_json_keys "$out" "d['valid']" "True"
assert_json_keys "$out" "d['live']" "False"
assert_json_keys "$out" "d['checks']['schema']['ok']" "True"
assert_json_keys "$out" "d['checks']['schema']['schema_version']" "1"
assert_json_keys "$out" "d['checks']['registry']['ok']" "True"
assert_json_keys "$out" "len(d['checks']['per_instance_fits'])" "2"
assert_json_keys "$out" "d['checks']['per_instance_fits'][0]['name']" "qwen-left"
assert_json_keys "$out" "d['checks']['cross_checks']['ok']" "True"
assert_json_keys "$out" "len(d['checks']['calibration'])" "2"
assert_json_keys "$out" "d['checks']['live']['checked']" "False"

# diagnose --json on a missing-compose estate (still valid JSON, RED summary)
MISSING_COMPOSE="${TMP_DIR}/estate-missing-compose.yml"
cat > "$MISSING_COMPOSE" <<'YAML'
schema_version: 1
created: 2026-05-14T00:00:00Z
rig:
  hardware_id: rtx-3090
  gpu_count: 2
  nvlink_active: false
estate:
  - name: missing
    compose: vllm/not-real
    gpus: [0]
    port: 8110
YAML
if out="$(python3 "$CLI" diagnose "$MISSING_COMPOSE" --json 2>&1)"; then
  echo "ASSERTION FAILED: missing-compose diagnose --json unexpectedly returned 0" >&2
  echo "$out" >&2
  exit 1
fi
assert_valid_json "$out"
assert_json_keys "$out" "d['checks']['registry']['ok']" "False"
assert_json_keys "$out" "d['checks']['registry']['missing']==['vllm/not-real']" "True"
assert_json_keys "$out" "d['summary']" "RED"

# --- HARD: absence of --json is byte-identical to today's human output ---
human_report="$(python3 "$CLI" report-state --file "$GOOD" 2>&1)"
assert_contains "$human_report" "## Profile state"
assert_contains "$human_report" "qwen-left: llamacpp/default, GPUs [0], port 8110"

human_diag="$(python3 "$CLI" diagnose "$GOOD" 2>&1)"
assert_contains "$human_diag" "[1/6] Estate file parses + schema_version supported"
assert_contains "$human_diag" "Triage summary: GREEN"

echo "test-estate-json: ok"
