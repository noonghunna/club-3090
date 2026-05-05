#!/usr/bin/env bash
#
# Soak test - runtime VRAM accretion / multi-turn traffic validation.
#
# Run before shipping config, Genesis, vLLM, or memory-policy changes that can
# pass verify-full.sh and verify-stress.sh but still accrete VRAM under
# repeated agent turns. This is intentionally slow and not part of launch.
#
# Scope:
#   - Single-stream multi-turn agent traffic, no concurrency stress.
#   - Watches VRAM growth, engine liveness, TTFT growth, and decode TPS
#     retention across sessions.
#   - Read-only against the running deployment.
#
# Time budget:
#   Default SOAK_SESSIONS=20 x SOAK_TURNS=5, capped by SOAK_TIMEOUT_S=1800.
#   Expect 10-30 minutes depending on config.
#
# Usage:
#   CONTAINER=vllm-qwen36-27b-long-vision bash scripts/soak-test.sh
#   ENDPOINT=http://localhost:8020 SOAK_SESSIONS=8 bash scripts/soak-test.sh
#
# Env:
#   CONTAINER              Running container. Default: first vllm-qwen36-27b*
#                          container from `docker ps`.
#   ENDPOINT / URL         OpenAI endpoint. Default: mapped container port for
#                          8000/tcp, falling back to http://localhost:8020.
#   MODEL                  Served model. Default: first id from /v1/models.
#   SOAK_MODE              "fresh" (default) — each turn is an independent
#                          conversation; tests raw VRAM accretion across
#                          requests. "continuous" — each session is one
#                          multi-turn agentic conversation that ramps to
#                          ~22-25K accumulated context by turn 5; tests the
#                          context-accumulation accretion class that bit
#                          club-3090#41 (hermes/openhands traffic).
#   SOAK_SESSIONS          Independent sessions. Default: 20.
#   SOAK_TURNS             Turns per session, max 5 fixture shapes. Default: 5.
#                          (Continuous mode requires SOAK_TURNS=5 — the
#                          turn shapes are designed to ramp; partial
#                          sessions don't reach the target context size.)
#   SOAK_MAX_GROWTH_MIB    Fail if max VRAM growth exceeds this after warm
#                          baseline. Default: 200 MiB.
#   SOAK_TIMEOUT_S         Hard wall-clock cap. Default: 1800 seconds.
#   SOAK_REQ_TIMEOUT_S     Per-request timeout. Default: 600 seconds.
#   SOAK_OUTPUT            Output dir. Default: results/soak-YYYYmmdd-HHMMSS.
#
# Outputs:
#   results/<run>/baseline.json
#   results/<run>/turn-log.csv
#   results/<run>/gpu-log.csv
#   results/<run>/summary.md
#
# Exit codes:
#   0 pass
#   1 fail
#   2 inconclusive / timeout / preflight could not run

set -euo pipefail

SOAK_SESSIONS="${SOAK_SESSIONS:-20}"
SOAK_TURNS="${SOAK_TURNS:-5}"
SOAK_MODE="${SOAK_MODE:-fresh}"
SOAK_MAX_GROWTH_MIB="${SOAK_MAX_GROWTH_MIB:-200}"
SOAK_TIMEOUT_S="${SOAK_TIMEOUT_S:-1800}"
SOAK_REQ_TIMEOUT_S="${SOAK_REQ_TIMEOUT_S:-600}"
SOAK_OUTPUT="${SOAK_OUTPUT:-results/soak-$(date +%Y%m%d-%H%M%S)}"

case "$SOAK_MODE" in
  fresh|continuous) ;;
  *) echo "ERROR: SOAK_MODE='${SOAK_MODE}' — must be 'fresh' or 'continuous'." >&2; exit 2 ;;
esac
if [[ "$SOAK_MODE" == "continuous" && "$SOAK_TURNS" -ne 5 ]]; then
  echo "ERROR: continuous mode requires SOAK_TURNS=5 (got ${SOAK_TURNS}). Turn shapes are designed to ramp; partial runs don't reach target context size." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="${REPO_ROOT}/scripts/soak-helper.py"
cd "$REPO_ROOT"

log() { printf '[soak] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 2; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"; }

need curl
need docker
need nvidia-smi
need python3
[[ -x "$HELPER" || -f "$HELPER" ]] || die "missing helper: $HELPER"

auto_container() {
  docker ps --format '{{.Names}}' 2>/dev/null \
    | grep -E '^(vllm-qwen36-27b|vllm-gemma-4-31b)' \
    | head -1 || true
}

endpoint_from_container() {
  local container="$1"
  local mapped port
  mapped="$(docker port "$container" 8000/tcp 2>/dev/null | head -1 || true)"
  if [[ -n "$mapped" ]]; then
    port="${mapped##*:}"
    [[ "$port" =~ ^[0-9]+$ ]] && { printf 'http://localhost:%s\n' "$port"; return 0; }
  fi
  printf 'http://localhost:8020\n'
}

vram_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '{gsub(/ /, "", $1); sum += $1} END {print sum + 0}'
}

append_gpu_snapshot() {
  local session="$1"
  local turn="$2"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v s="$session" -v t="$turn" '{
        for (i = 1; i <= NF; i++) gsub(/^ +| +$/, "", $i)
        printf "%s,%s,%s,%s,%s\n", s, t, $1, $2, $3
      }' >> "$GPU_LOG" || true
}

capture_state() {
  local label="$1"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
    --format=csv,noheader,nounits > "${SOAK_OUTPUT}/nvidia-smi-${label}.csv" 2>/dev/null || true
  docker stats --no-stream --format '{{json .}}' "$CONTAINER" \
    > "${SOAK_OUTPUT}/docker-stats-${label}.jsonl" 2>/dev/null || true
}

finish() {
  local rc=$?
  capture_state "final"
  log "artifacts: ${SOAK_OUTPUT}"
  exit "$rc"
}
trap finish EXIT
trap 'log "interrupted"; exit 2' INT TERM

CONTAINER="${CONTAINER:-$(auto_container)}"
[[ -n "$CONTAINER" ]] || die "no running vllm-qwen36-27b* container found; set CONTAINER=..."
docker inspect "$CONTAINER" >/dev/null 2>&1 || die "container '$CONTAINER' not found"
[[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" == "true" ]] \
  || die "container '$CONTAINER' is not running"

ENDPOINT="${ENDPOINT:-${URL:-$(endpoint_from_container "$CONTAINER")}}"
mkdir -p "$SOAK_OUTPUT"

MODELS_JSON="${SOAK_OUTPUT}/models.json"
curl -sf -m 10 "${ENDPOINT}/v1/models" -o "$MODELS_JSON" \
  || die "no response from ${ENDPOINT}/v1/models"
MODEL="${MODEL:-$(python3 "$HELPER" model "$MODELS_JSON")}"

TURN_LOG="${SOAK_OUTPUT}/turn-log.csv"
GPU_LOG="${SOAK_OUTPUT}/gpu-log.csv"
SUMMARY_MD="${SOAK_OUTPUT}/summary.md"
REQUEST_DIR="${SOAK_OUTPUT}/requests"
RESPONSE_DIR="${SOAK_OUTPUT}/responses"
STATE_DIR="${SOAK_OUTPUT}/states"
mkdir -p "$REQUEST_DIR" "$RESPONSE_DIR" "$STATE_DIR"

printf 'session_id,turn_id,t_ms,vram_mib,ttft_ms,decode_tps,completion_tokens,status,error\n' > "$TURN_LOG"
printf 'session_id,turn_id,gpu_index,memory_used_mib,utilization_gpu_pct\n' > "$GPU_LOG"

capture_state "baseline"
python3 "$HELPER" baseline "$SOAK_OUTPUT" "$CONTAINER" "$ENDPOINT" "$MODEL" \
  "$SOAK_SESSIONS" "$SOAK_TURNS" "$SOAK_MAX_GROWTH_MIB"

log "running soak test against ${ENDPOINT} (model=${MODEL}, container=${CONTAINER})"
log "mode=${SOAK_MODE} sessions=${SOAK_SESSIONS} turns=${SOAK_TURNS} max_growth=${SOAK_MAX_GROWTH_MIB}MiB timeout=${SOAK_TIMEOUT_S}s"
log "output=${SOAK_OUTPUT}"

START_SECONDS="$SECONDS"
BOOT_VRAM_MIB=""
TIMED_OUT=0

for session in $(seq 1 "$SOAK_SESSIONS"); do
  log "session ${session}/${SOAK_SESSIONS}"
  state_file="${STATE_DIR}/state-s${session}.json"
  if [[ "$SOAK_MODE" == "continuous" ]]; then
    python3 "$HELPER" init-session "$state_file" "$session"
  fi
  for turn in $(seq 1 "$SOAK_TURNS"); do
    if (( SECONDS - START_SECONDS >= SOAK_TIMEOUT_S )); then
      TIMED_OUT=1
      log "timeout reached before session=${session} turn=${turn}"
      break 2
    fi

    req_file="${REQUEST_DIR}/s${session}-t${turn}.json"
    metrics_file="${RESPONSE_DIR}/s${session}-t${turn}.metrics.json"
    if [[ "$SOAK_MODE" == "continuous" ]]; then
      python3 "$HELPER" request-continuous "$MODEL" "$state_file" "$turn" "$req_file"
    else
      python3 "$HELPER" request "$MODEL" "$session" "$turn" "$req_file"
    fi
    python3 "$HELPER" run "$ENDPOINT" "$req_file" "$SOAK_REQ_TIMEOUT_S" "$metrics_file"
    if [[ "$SOAK_MODE" == "continuous" ]]; then
      python3 "$HELPER" ingest "$state_file" "$metrics_file" "$turn"
    fi

    vram="$(vram_mib)"
    append_gpu_snapshot "$session" "$turn"
    python3 "$HELPER" append-log "$TURN_LOG" "$session" "$turn" "$vram" "$metrics_file"

    read -r status t_ms ttft_ms decode_tps < <(python3 "$HELPER" metric "$metrics_file")
    log "  turn ${turn}/${SOAK_TURNS}: status=${status} wall=${t_ms}ms ttft=${ttft_ms}ms decode_tps=${decode_tps} vram=${vram}MiB"
  done

  # Capture warm baseline at END of session 1 — after all 5 turn shapes have
  # run once and prefix cache has filled. Real accretion is measured FROM
  # this baseline across sessions 2-N, so cache-fill (typically +500-1500
  # MiB on the first 12K-char tool-result paste) doesn't false-positive.
  # Calibration validated 2026-05-03 on long-text @ 0.93 + 180K — sessions
  # 2-10 stayed flat at session-1-end VRAM, confirming the test discriminates
  # cache fill from accretion correctly.
  if [[ -z "$BOOT_VRAM_MIB" ]]; then
    BOOT_VRAM_MIB="$(vram_mib)"
    log "warm baseline after session 1: ${BOOT_VRAM_MIB} MiB"
  fi
done

if [[ -z "$BOOT_VRAM_MIB" ]]; then
  BOOT_VRAM_MIB="$(vram_mib)"
  TIMED_OUT=1
  log "no completed turns; writing inconclusive summary"
fi

set +e
python3 "$HELPER" summary "$TURN_LOG" "$SUMMARY_MD" "$BOOT_VRAM_MIB" \
  "$SOAK_MAX_GROWTH_MIB" "$TIMED_OUT" "$SOAK_SESSIONS"
rc=$?
set -e
exit "$rc"
