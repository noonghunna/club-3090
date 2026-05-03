#!/usr/bin/env bash
#
# Switch between club-3090 compose variants.
#
# Brings down whatever's currently running, brings up the new variant,
# and (optionally) waits for the server to report ready on /v1/models.
# Stateless — re-run any time you want a different config.
#
# Usage:
#   bash scripts/switch.sh <variant>           # switch + tail until ready
#   bash scripts/switch.sh <variant> --no-wait # switch and return immediately
#   bash scripts/switch.sh --list              # show all variants
#   bash scripts/switch.sh --down              # just bring down whatever's up
#
# Variant names (engine/file, file is the docker-compose.<file>.yml stem):
#
#   Single-card vLLM:
#     vllm/default            48K + TQ3 + MTP + vision + tools (recommended)
#     vllm/long-vision        198K + TQ3 + vision (cliff-safe; Cliff 2 single-prompt >50K still applies)
#     vllm/long-text          180K + TQ3 + MTP + text-only (Balanced MTP — 60K single-prompt closed via v7.69 + #35975)
#     vllm/long-text-no-mtp   200K + TQ3 + no MTP + text-only (Max-context — same Cliff 2 closure, more KV pool, slower decode)
#     vllm/bounded-thinking   180K + TQ3 + structured-CoT grammar in reasoning (~30× cheaper think, +24pp LCB v6)
#     vllm/tools-text         75K + fp8 + MTP + text-only (IDE agents — Cline / Cursor)
#     vllm/minimal            32K + fp8 (no Genesis, no spec-decode, simplest)
#
#   Dual-card vLLM (TP=2):
#     vllm/dual             262K + fp8 + 2 streams + vision (recommended dual)
#     vllm/dual-turbo       262K + TQ3 + 4 streams + vision (multi-tenant)
#     vllm/dual-dflash      185K + FP16 + DFlash N=5 + vision (peak code TPS)
#     vllm/dual-dflash-noviz 200K + FP16 + DFlash N=5 + no vision (peak code, max ctx)
#     vllm/dual-nvlink      262K + fp8 + 2 streams + vision (REQUIRES NVLink bridge — community/experimental)
#
#   Single-card llama.cpp:
#     llamacpp/default      Q3_K_XL + 262K + q4_0 KV + vision (max ctx, no cliffs)
#     llamacpp/concurrent   Q3_K_XL + 192K pool + 4 parallel slots + vision
#
# Env overrides (rarely needed):
#   COMPOSE_BIN     Default: "docker compose" (set to e.g. "podman compose" if needed)
#   READY_URL       Default: http://localhost:8020/v1/models
#   READY_TIMEOUT   Default: 600 (seconds — longer for cold cudagraph capture)

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_BIN="${COMPOSE_BIN:-docker compose}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"

# Load .env if present, so PORT / MODEL_DIR / etc. flow through to docker
# compose AND to the ready-URL probe below.
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

# Per-variant default port (matches each compose's "${PORT:-XXXX}:8000"
# fallback). Used when neither $PORT nor $READY_URL is set explicitly.
declare -A VARIANT_DEFAULT_PORT=(
  [vllm/default]=8020
  [vllm/long-vision]=8020
  [vllm/long-text]=8020
  [vllm/long-text-no-mtp]=8021
  [vllm/bounded-thinking]=8020
  [vllm/tools-text]=8020
  [vllm/minimal]=8020
  [vllm/dual]=8010
  [vllm/dual-turbo]=8011
  [vllm/dual-dflash]=8012
  [vllm/dual-dflash-noviz]=8013
  [vllm/dual-nvlink]=8014
  [llamacpp/default]=8020
  [llamacpp/concurrent]=8020
)

# variant -> "engine|compose_dir|file"  (file relative to compose_dir)
declare -A VARIANTS=(
  [vllm/default]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.yml"
  [vllm/long-vision]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.long-vision.yml"
  [vllm/long-text]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.long-text.yml"
  [vllm/long-text-no-mtp]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.long-text-no-mtp.yml"
  [vllm/bounded-thinking]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.bounded-thinking.yml"
  [vllm/tools-text]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.tools-text.yml"
  [vllm/minimal]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.minimal.yml"
  [vllm/dual]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.dual.yml"
  [vllm/dual-turbo]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.dual-turbo.yml"
  [vllm/dual-dflash]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.dual-dflash.yml"
  [vllm/dual-dflash-noviz]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.dual-dflash-noviz.yml"
  [vllm/dual-nvlink]="vllm|models/qwen3.6-27b/vllm/compose|docker-compose.dual-nvlink.yml"
  [llamacpp/default]="llamacpp|models/qwen3.6-27b/llama-cpp/compose|docker-compose.yml"
  [llamacpp/concurrent]="llamacpp|models/qwen3.6-27b/llama-cpp/compose|docker-compose.concurrent.yml"
)

# Container name patterns we'll bring down — covers all current composes.
RUNNING_PATTERN="^(vllm-qwen36-27b|llama-cpp-qwen36-27b|vllm-qwen36-27b-bounded-thinking|vllm-qwen36-27b-long-text-no-mtp)"

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

list_variants() {
  echo "Available variants:"
  for v in "${!VARIANTS[@]}"; do
    IFS='|' read -r eng dir file <<< "${VARIANTS[$v]}"
    echo "  ${v}  →  ${dir}/${file}"
  done | sort
  exit 0
}

down_running() {
  local running
  running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E "$RUNNING_PATTERN" || true)
  if [[ -z "$running" ]]; then
    echo "[switch] no club-3090 container running"
    return
  fi
  for c in $running; do
    echo "[switch] bringing down: ${c}"
    # find the compose dir from the container's labels — fallback to direct stop
    local lbl_dir lbl_file
    lbl_dir=$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.project.working_dir"}}' "$c" 2>/dev/null || true)
    lbl_file=$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.project.config_files"}}' "$c" 2>/dev/null || true)
    if [[ -n "$lbl_dir" && -n "$lbl_file" ]]; then
      (cd "$lbl_dir" && ${COMPOSE_BIN} -f "$lbl_file" down) || docker stop "$c" >/dev/null
    else
      docker stop "$c" >/dev/null
    fi
  done
}

up_variant() {
  local v="$1"
  if [[ -z "${VARIANTS[$v]:-}" ]]; then
    echo "ERROR: unknown variant '${v}'." >&2
    echo "Run: bash scripts/switch.sh --list" >&2
    exit 1
  fi
  IFS='|' read -r eng dir file <<< "${VARIANTS[$v]}"
  local full_dir="${ROOT_DIR}/${dir}"
  if [[ ! -f "${full_dir}/${file}" ]]; then
    echo "ERROR: compose file missing at ${full_dir}/${file}" >&2
    exit 1
  fi

  # Pre-up sanity: warn if the on-disk Genesis tree is out of sync with
  # the GENESIS_PIN declared in setup.sh (catches "user pulled latest but
  # didn't re-run setup.sh") AND if the repo itself is behind origin/master
  # (catches "user cloned weeks ago, never pulled"). Both soft-warn via
  # stderr and don't block boot. See club-3090#32 for the original case.
  if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
    # shellcheck source=preflight.sh
    source "${ROOT_DIR}/scripts/preflight.sh"
    preflight_genesis_pin "${ROOT_DIR}" || true
    preflight_repo_drift "${ROOT_DIR}" || true
  fi

  echo "[switch] bringing up: ${v}  (${dir}/${file})"
  (cd "${full_dir}" && ${COMPOSE_BIN} -f "${file}" up -d)
}

resolve_ready_url() {
  # Precedence: $READY_URL (full override) → $PORT (port only, host=localhost)
  # → per-variant default port from VARIANT_DEFAULT_PORT.
  local variant="$1"
  if [[ -n "${READY_URL:-}" ]]; then
    return 0
  fi
  local port="${PORT:-${VARIANT_DEFAULT_PORT[$variant]:-8020}}"
  READY_URL="http://localhost:${port}/v1/models"
}

wait_ready() {
  echo "[switch] waiting for ${READY_URL} (timeout ${READY_TIMEOUT}s)..."
  local elapsed=0 step=4
  until curl -sf -o /dev/null --max-time 3 "${READY_URL}"; do
    sleep $step
    elapsed=$((elapsed + step))
    if [[ $elapsed -ge $READY_TIMEOUT ]]; then
      echo "[switch] timeout — server not ready after ${READY_TIMEOUT}s" >&2
      echo "[switch] tail logs:  docker logs --tail 100 \$(docker ps --format '{{.Names}}' | grep qwen36-27b | head -1)" >&2
      exit 1
    fi
    [[ $((elapsed % 30)) -eq 0 ]] && echo "[switch]   ${elapsed}s elapsed, still waiting..."
  done
  echo "[switch] ✓ ready"
}

# --- arg parsing ---
WAIT=1
case "${1:-}" in
  -h|--help|"") usage ;;
  --list) list_variants ;;
  --down) down_running; exit 0 ;;
esac

VARIANT="$1"
shift || true
for arg in "$@"; do
  case "$arg" in
    --no-wait) WAIT=0 ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

resolve_ready_url "${VARIANT}"
down_running
up_variant "${VARIANT}"
[[ $WAIT -eq 1 ]] && wait_ready
echo "[switch] done. Try:  curl -s ${READY_URL%/v1/models}/v1/models | jq ."
