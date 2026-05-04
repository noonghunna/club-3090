#!/usr/bin/env bash
#
# Boot a vLLM compose variant with the residency sidecar mounted, run the v2
# continuous soak, and join the raw instrumentation rows to soak turn rows.
#
# Defaults target the known Cliff 2 reproducer:
#   VARIANT=vllm/long-text SOAK_MODE=continuous SOAK_SESSIONS=2 SOAK_TURNS=5

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

COMPOSE_BIN="${COMPOSE_BIN:-docker compose}"
VARIANT="${VARIANT:-vllm/long-text}"
SOAK_OUTPUT="${SOAK_OUTPUT:-${ROOT_DIR}/results/residency-$(date +%Y%m%d-%H%M%S)}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
RESIDENCY_ENGINE_STEP_INTERVAL="${RESIDENCY_ENGINE_STEP_INTERVAL:-1}"
RESIDENCY_WORKER_STEP_INTERVAL="${RESIDENCY_WORKER_STEP_INTERVAL:-1}"
RESIDENCY_SLOW_EVERY_N="${RESIDENCY_SLOW_EVERY_N:-1}"
RESIDENCY_KEEP_SERVER="${RESIDENCY_KEEP_SERVER:-0}"
RESIDENCY_DRY_RUN="${RESIDENCY_DRY_RUN:-0}"
RESIDENCY_EMPTY_CACHE_ON_IDLE="${RESIDENCY_EMPTY_CACHE_ON_IDLE:-0}"

case "$VARIANT" in
  vllm/long-text)
    COMPOSE_DIR="${ROOT_DIR}/models/qwen3.6-27b/vllm/compose"
    COMPOSE_FILE="docker-compose.long-text.yml"
    SERVICE="vllm-qwen36-27b-long-text"
    DEFAULT_PORT="8020"
    ;;
  vllm/long-text-no-mtp)
    COMPOSE_DIR="${ROOT_DIR}/models/qwen3.6-27b/vllm/compose"
    COMPOSE_FILE="docker-compose.long-text-no-mtp.yml"
    SERVICE="vllm-qwen36-27b-long-text-no-mtp"
    DEFAULT_PORT="8021"
    ;;
  vllm/tools-text)
    COMPOSE_DIR="${ROOT_DIR}/models/qwen3.6-27b/vllm/compose"
    COMPOSE_FILE="docker-compose.tools-text.yml"
    SERVICE="vllm-qwen36-27b-tools-text"
    DEFAULT_PORT="8020"
    ;;
  vllm/dual)
    COMPOSE_DIR="${ROOT_DIR}/models/qwen3.6-27b/vllm/compose"
    COMPOSE_FILE="docker-compose.dual.yml"
    SERVICE="vllm-qwen36-27b-dual"
    DEFAULT_PORT="8010"
    ;;
  *)
    echo "ERROR: unsupported VARIANT='${VARIANT}'." >&2
    echo "Supported: vllm/long-text, vllm/long-text-no-mtp, vllm/tools-text, vllm/dual" >&2
    exit 2
    ;;
esac

COMPOSE_PATH="${COMPOSE_DIR}/${COMPOSE_FILE}"
[[ -f "$COMPOSE_PATH" ]] || { echo "ERROR: missing compose file: $COMPOSE_PATH" >&2; exit 2; }

log() { printf '[residency-soak] %s\n' "$*"; }

extract_max_batched_tokens() {
  awk '
    /--max-num-batched-tokens/ {
      getline
      gsub(/["[:space:]-]/, "", $0)
      print
      exit
    }
  ' "$COMPOSE_PATH"
}

MAX_BATCHED="$(extract_max_batched_tokens || true)"
if [[ "$MAX_BATCHED" =~ ^[0-9]+$ && "$MAX_BATCHED" -lt 4128 ]]; then
  echo "ERROR: ${COMPOSE_FILE} has --max-num-batched-tokens ${MAX_BATCHED}." >&2
  echo "Qwen3-Next Mamba/GDN cache align block_size is 4128; this boot will fail before serving." >&2
  echo "Restore 4128 before running the residency pilot." >&2
  exit 2
fi

mkdir -p "$SOAK_OUTPUT"
SOAK_OUTPUT="$(cd "$SOAK_OUTPUT" && pwd)"
RAW_CSV="${SOAK_OUTPUT}/residency-log.raw.csv"
TURN_CSV="${SOAK_OUTPUT}/residency-turns.csv"
OVERLAY="${SOAK_OUTPUT}/docker-compose.residency-overlay.yml"
SERVER_LOG="${SOAK_OUTPUT}/container.log"
SOAK_DIR="${SOAK_OUTPUT}/soak"

cat > "$OVERLAY" <<YAML
services:
  ${SERVICE}:
    volumes:
      - ${ROOT_DIR}/tools/residency-instrument:/residency:ro
      - ${SOAK_OUTPUT}:/data
    environment:
      - PYTHONPATH=/residency
      - RESIDENCY_LOG_PATH=/data/residency-log.raw.csv
      - GENESIS_RESIDENCY_LOG=/data/residency-log.raw.csv
      - RESIDENCY_ENGINE_STEP_INTERVAL=${RESIDENCY_ENGINE_STEP_INTERVAL}
      - RESIDENCY_WORKER_STEP_INTERVAL=${RESIDENCY_WORKER_STEP_INTERVAL}
      - RESIDENCY_SLOW_EVERY_N=${RESIDENCY_SLOW_EVERY_N}
      - RESIDENCY_EMPTY_CACHE_ON_IDLE=${RESIDENCY_EMPTY_CACHE_ON_IDLE}
YAML

cleanup() {
  local rc=$?
  docker logs "$SERVICE" > "$SERVER_LOG" 2>&1 || true
  if [[ "$RESIDENCY_KEEP_SERVER" != "1" ]]; then
    (cd "$COMPOSE_DIR" && ${COMPOSE_BIN} -f "$COMPOSE_FILE" -f "$OVERLAY" down) >/dev/null 2>&1 || true
  fi
  log "artifacts: ${SOAK_OUTPUT}"
  exit "$rc"
}
trap cleanup EXIT
trap 'log "interrupted"; exit 2' INT TERM

if [[ "$RESIDENCY_DRY_RUN" == "1" ]]; then
  log "dry-run: rendering merged compose config"
  (cd "$COMPOSE_DIR" && ${COMPOSE_BIN} -f "$COMPOSE_FILE" -f "$OVERLAY" config) \
    > "${SOAK_OUTPUT}/docker-compose.residency.config.yml"
  log "dry-run complete"
  exit 0
fi

log "bringing down existing club-3090 containers"
PREFLIGHT_NO_FETCH=1 bash "${ROOT_DIR}/scripts/switch.sh" --down

log "booting ${VARIANT} with residency overlay"
(cd "$COMPOSE_DIR" && ${COMPOSE_BIN} -f "$COMPOSE_FILE" -f "$OVERLAY" up -d)

READY_URL="${READY_URL:-http://localhost:${PORT:-${DEFAULT_PORT}}/v1/models}"
ENDPOINT="${ENDPOINT:-${READY_URL%/v1/models}}"
log "waiting for ${READY_URL}"
elapsed=0
while ! curl -sf -o /dev/null --max-time 3 "$READY_URL"; do
  state="$(docker inspect -f '{{.State.Running}}' "$SERVICE" 2>/dev/null || echo false)"
  if [[ "$state" != "true" ]]; then
    echo "ERROR: container ${SERVICE} exited before ready." >&2
    docker logs --tail 80 "$SERVICE" >&2 || true
    exit 2
  fi
  sleep 4
  elapsed=$((elapsed + 4))
  if (( elapsed >= READY_TIMEOUT )); then
    echo "ERROR: timed out waiting for ${READY_URL}" >&2
    docker logs --tail 80 "$SERVICE" >&2 || true
    exit 2
  fi
  if (( elapsed % 30 == 0 )); then
    log "${elapsed}s elapsed, still waiting"
  fi
done

log "running soak; raw residency CSV=${RAW_CSV}"
set +e
CONTAINER="$SERVICE" \
ENDPOINT="$ENDPOINT" \
SOAK_MODE="${SOAK_MODE:-continuous}" \
SOAK_SESSIONS="${SOAK_SESSIONS:-2}" \
SOAK_TURNS="${SOAK_TURNS:-5}" \
SOAK_OUTPUT="$SOAK_DIR" \
bash "${ROOT_DIR}/scripts/soak-test.sh"
SOAK_RC=$?
set -e

if [[ -f "$RAW_CSV" && -f "${SOAK_DIR}/turn-log.csv" ]]; then
  log "joining raw residency rows to per-turn CSV"
  python3 "${ROOT_DIR}/tools/residency-instrument/instrument.py" join \
    --residency-csv "$RAW_CSV" \
    --turn-log "${SOAK_DIR}/turn-log.csv" \
    --output "$TURN_CSV"
else
  log "raw CSV or turn log missing; skipping joined CSV"
fi

exit "$SOAK_RC"
