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
# PASS verdict semantics:
#   PASS = no failure signal fired on the test sample. Specifically:
#     - silent_empty turns: 0 (no HTTP 200 + 0 completion tokens)
#     - max VRAM growth: under SOAK_MAX_GROWTH_MIB (default 200 MiB)
#     - TPS retention: first-5 vs last-5 median >= 98%
#     - request errors / stream interruptions: 0
#   PASS does NOT mean:
#     - "Patches in this compose's overlay set are doing useful work."
#       PASS-on-patched is consistent with patches working OR with patches
#       not being load-bearing for this workload + topology. Cliff 2 / 2b
#       mitigations target single-card 24 GB pressure; TP=2 (dual.yml)
#       structurally escapes Cliff 2 regardless of which patches load.
#     - "Deeper-context workloads will also pass." Continuous mode ramps
#       to ~22-25K accumulated tokens by turn 5; it does not push to
#       model max_ctx. Longer-context regimes can still fail.
#     - "The configuration is optimally tuned." Soak detects failures,
#       not whether perf is on the table.
#   For patch attribution, run the same soak on the same compose with
#   the overlay bind-mounts stripped (or on a baseline image) and compare
#   metrics. See https://github.com/noonghunna/club-3090/issues/140.
#
# Time budget:
#   Default SOAK_SESSIONS=20 x SOAK_TURNS=5, capped by SOAK_TIMEOUT_S=1800.
#   Expect 10-30 minutes depending on config.
#
# Usage (preferred):
#   bash scripts/soak-test.sh                         # default fresh mode (20 sessions × 5 turns)
#   bash scripts/soak-test.sh --continuous            # Cliff 2b detector (5 sessions × 5 turns ramping ctx)
#   bash scripts/soak-test.sh --quick                 # 8 sessions × 5 turns, fresh mode (~5-8 min)
#   bash scripts/soak-test.sh --help                  # full help
#
# Auto-detect: container + endpoint + model are sniffed from `docker ps` and
# the running endpoint's /v1/models. Override via env vars if needed.
#
# Env (advanced — flags above cover the common cases):
#   CONTAINER              Running container. Default: the first vllm-/llama-cpp-/
#                          ik-llama-/beellama-/sglang- container with an engine-port
#                          mapping (8000/8080/30000) from `docker ps` — model-agnostic.
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

usage() {
  cat <<'EOF'
soak-test.sh — multi-turn VRAM-accretion + Cliff 2b validation

USAGE
  bash scripts/soak-test.sh [MODE]

MODES
  (default)         fresh mode: 20 sessions × 5 turns, ~10-25 min
                    Tests raw per-request VRAM accretion.
  --continuous      Cliff 2b detector: 5 sessions × 5 turns, ramping context
                    to ~22-25K accumulated tokens. **The only test that
                    catches the multi-turn accumulating-context cliff** that
                    bit hermes/openhands traffic on long-* configs.
  --quick           8 sessions × 5 turns, fresh mode (~5-8 min)
  --fresh           Explicit fresh mode (same as default)

OPTIONS
  -h, --help        Show this help

ENV (advanced — auto-detected by default)
  CONTAINER         Running container. Default: the first vllm-/llama-cpp-/
                    ik-llama-/beellama-/sglang- container with an engine-port
                    mapping (8000/8080/30000) from docker ps — model-agnostic.
                    Use CONTAINER=none for host-mode engines (e.g. llama.cpp
                    host build).
  ENDPOINT / URL    OpenAI endpoint. Default: mapped container port → fallback
                    http://localhost:8020.
  MODEL             Served model. Default: first id from /v1/models.
  SOAK_SESSIONS     Override session count.
  SOAK_TURNS        Override turn count.
  SOAK_MAX_GROWTH_MIB   VRAM-growth fail threshold. Default: 200.
  SOAK_TIMEOUT_S    Hard wall-clock cap. Default: 1800.

EXAMPLES
  bash scripts/soak-test.sh --continuous          # Cliff 2b detector
  bash scripts/soak-test.sh --quick               # fast smoke
  CONTAINER=vllm-gemma-4-31b-mtp bash scripts/soak-test.sh --continuous
  CONTAINER=none ENDPOINT=http://localhost:8030 bash scripts/soak-test.sh

NOTES
  Soak-continuous is the only test that surfaces Cliff 2b under
  multi-turn accumulating-context traffic on single-card configs.
  If you're filing a bench contribution, run with --continuous and
  paste the [soak] summary alongside your bench numbers.
  See docs/CLIFFS.md for context.

PASS VERDICT — WHAT IT DOES AND DOES NOT MEAN
  PASS = no failure signal on the test sample (silent_empty=0, VRAM
  growth under threshold, TPS retention >= 98%, zero errors).
  PASS does NOT validate that patches in the compose's overlay set
  are load-bearing for the workload — topology alone (e.g. TP=2)
  can sidestep the failure mode patches target. For patch attribution,
  re-run the same soak with overlays stripped and compare.
  Full discussion: docs/CLIFFS.md and issue #140.

EOF
}

# --- arg parsing -------------------------------------------------------------
# Set defaults (env vars override; flags override env vars; --help short-circuits)
MODE_FLAG=""
QUICK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --continuous)  MODE_FLAG="continuous"; shift ;;
    --fresh)       MODE_FLAG="fresh"; shift ;;
    --quick)       QUICK=1; MODE_FLAG="${MODE_FLAG:-fresh}"; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "✗ unknown argument: $1" >&2
                   echo "  run 'bash scripts/soak-test.sh --help' for usage." >&2
                   exit 2 ;;
  esac
done

SOAK_MODE="${SOAK_MODE:-${MODE_FLAG:-fresh}}"
if [[ "$SOAK_MODE" == "continuous" ]]; then
  # Continuous mode requires the ramping turn shape; sessions=5 is the
  # standard cross-rig cadence (matches what BENCHMARKS rows cite).
  SOAK_SESSIONS="${SOAK_SESSIONS:-5}"
  SOAK_TURNS="${SOAK_TURNS:-5}"
elif [[ "$QUICK" == "1" ]]; then
  SOAK_SESSIONS="${SOAK_SESSIONS:-8}"
  SOAK_TURNS="${SOAK_TURNS:-5}"
else
  SOAK_SESSIONS="${SOAK_SESSIONS:-20}"
  SOAK_TURNS="${SOAK_TURNS:-5}"
fi
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
soft_need() { command -v "$1" >/dev/null 2>&1; }

need curl
need nvidia-smi
need python3
[[ -x "$HELPER" || -f "$HELPER" ]] || die "missing helper: $HELPER"

# docker is soft-required: only needed for container-mode tracking
# (docker stats + docker logs scrape). Host engines (e.g. llama.cpp host
# build, see #85, #87) use CONTAINER=none and run without docker.
HAVE_DOCKER=0
if soft_need docker; then HAVE_DOCKER=1; fi
if [[ "${CONTAINER:-}" == "none" ]]; then
  HOST_MODE=1
elif [[ "$HAVE_DOCKER" == "0" ]]; then
  log "docker not in PATH — running in host mode (CONTAINER=none implied)"
  HOST_MODE=1
  CONTAINER="none"
else
  HOST_MODE=0
fi

auto_container() {
  # Detect a running inference container by its ENGINE-INTERNAL port mapping
  # (vLLM 8000 / llama.cpp 8080 / sglang 30000), NOT a hardcoded model-name
  # allowlist — so ANY shipped compose is found regardless of model
  # (diffusiongemma-26b-a4b, gemma-4-26b-a4b, gemma-4-12b, a BYO container, …).
  # Re-syncs with the canonical preflight.sh::preflight_autodetect_endpoint, which
  # this function is meant to mirror: the model allowlist had drifted and silently
  # failed soak auto-detect for every model not in the list (#405: a shipped
  # diffusiongemma-26b-a4b compose — same bug class as the #310 preflight fix).
  # Among port matches, prefer a recognised club-3090 engine-family prefix;
  # otherwise take the first. The `|| true` is load-bearing under set -euo pipefail.
  local lines name
  lines=$(docker ps --format '{{.Names}}|{{.Ports}}' 2>/dev/null \
    | grep -E '([0-9]{1,3}\.){3}[0-9]{1,3}:[0-9]+->(8000|8080|30000)/tcp' || true)
  [[ -z "$lines" ]] && return 0
  name=$(printf '%s\n' "$lines" \
    | grep -E '^(vllm-|llama-cpp-|ik-llama-|sglang-|beellama-)' | head -1 || true)
  [[ -z "$name" ]] && name=$(printf '%s\n' "$lines" | head -1)
  printf '%s\n' "${name%%|*}"
}

endpoint_from_container() {
  local container="$1"
  local mapped port internal
  # vllm maps internal 8000, llama.cpp / ik_llama map 8080, sglang maps 30000.
  for internal in 8000 8080 30000; do
    mapped="$(docker port "$container" "${internal}/tcp" 2>/dev/null | head -1 || true)"
    if [[ -n "$mapped" ]]; then
      port="${mapped##*:}"
      [[ "$port" =~ ^[0-9]+$ ]] && { printf 'http://localhost:%s\n' "$port"; return 0; }
    fi
  done
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
  if [[ "$HOST_MODE" == "0" ]]; then
    docker stats --no-stream --format '{{json .}}' "$CONTAINER" \
      > "${SOAK_OUTPUT}/docker-stats-${label}.jsonl" 2>/dev/null || true
  fi
}

finish() {
  local rc=$?
  capture_state "final"
  log "artifacts: ${SOAK_OUTPUT}"
  exit "$rc"
}
trap finish EXIT
trap 'log "interrupted"; exit 2' INT TERM

if [[ "$HOST_MODE" == "1" ]]; then
  log "host mode: CONTAINER=none — skipping docker checks (URL must be set or auto-detected)"
  CONTAINER="none"
else
  CONTAINER="${CONTAINER:-$(auto_container)}"
  [[ -n "$CONTAINER" ]] || die "no running club-3090 inference container found (looked for an engine-port mapping 8000/8080/30000 on a vllm-/llama-cpp-/ik-llama-/beellama-/sglang- container); set CONTAINER=... or CONTAINER=none for host engines"
  docker inspect "$CONTAINER" >/dev/null 2>&1 || die "container '$CONTAINER' not found (use CONTAINER=none for host engine builds)"
  [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" == "true" ]] \
    || die "container '$CONTAINER' is not running"
fi

if [[ "$HOST_MODE" == "1" ]]; then
  # Host mode: URL must be set explicitly (or fall back to localhost:8020).
  # We can't sniff a port from a container that doesn't exist.
  ENDPOINT="${ENDPOINT:-${URL:-http://localhost:8020}}"
else
  ENDPOINT="${ENDPOINT:-${URL:-$(endpoint_from_container "$CONTAINER")}}"
fi
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
