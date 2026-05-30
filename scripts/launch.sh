#!/usr/bin/env bash
#
# Interactive launcher for club-3090 — pick model + GPUs, project the
# VRAM budget, boot the right compose, run verify-full to confirm it's serving.
#
# For first-run users coming in from the README. If you already know
# what you want, use `scripts/switch.sh <variant>` directly.
#
# Usage:
#   bash scripts/launch.sh                              # interactive model/GPU wizard
#   bash scripts/launch.sh --variant <name>             # skip wizard, boot directly
#   bash scripts/launch.sh --estate                     # multi-model estate wizard
#   bash scripts/launch.sh --estate-file <path>          # boot an existing estate plan
#   bash scripts/launch.sh --estate-file <path> --parallel --parallel-jobs 3 --parallel-stagger 30
#   bash scripts/launch.sh --validate-estate <path>      # validate estate.yml, no boot
#   bash scripts/launch.sh --down-estate <path>          # stop estate instances
#   bash scripts/launch.sh --topology                    # print GPU topology advisory, no boot
#   bash scripts/launch.sh --model qwen3.6-27b --gpus 0,1
#   bash scripts/launch.sh --engine vllm --cards 1      # deprecated; prefer --gpus
#   bash scripts/launch.sh --workload long-ctx-single    # profile-aware filter
#   bash scripts/launch.sh --stable                      # stable engine profiles only
#   bash scripts/launch.sh --tp 2 --pp 1                # override vLLM parallelism
#   bash scripts/launch.sh --no-projection              # skip kv-calc budget projection
#   bash scripts/launch.sh --no-verify                  # skip post-launch verify-full
#   bash scripts/launch.sh --no-preflight               # skip docker/GPU pre-flight
#
# The wizard marks variants that don't fit the detected GPUs; direct
# --variant keeps the power-user path and delegates final gating to switch.sh.
# All flags accept the same names as `switch.sh --list` produces.
# Examples:
#   bash scripts/launch.sh --variant vllm/default
#   bash scripts/launch.sh --variant llamacpp/default
#   bash scripts/launch.sh --variant vllm/dual
#
# Env vars:
#   NVLINK_MODE=auto|force_on|force_off — NVLink auto-detection for dual-card composes
#   auto (default): detects NVLink via nvidia-smi topo -m
#   force_on: assume NVLink bridge present, set NVLink env vars
#   force_off: force PCIe-only path even if NVLink detected

set -euo pipefail

# The VRAM-budget block formats dot-decimal numbers emitted by kv-calc.py JSON
# (e.g. "9.0") with `printf "%.2f"`. Under a comma-decimal LC_NUMERIC locale
# (de_DE, fr_FR, …) bash printf rejects the dot — `printf: 9.0: invalid number`
# / `Ungültige Zahl` — and the launcher aborts at the budget print (#159).
# Force C numeric parsing for the whole script; LC_CTYPE/encoding is left
# untouched so UTF-8 UI glyphs (—, ×, ⚠) still render.
export LC_NUMERIC=C

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SWITCH="${SWITCH:-${ROOT_DIR}/scripts/switch.sh}"
VERIFY="${VERIFY:-${ROOT_DIR}/scripts/verify-full.sh}"
LAUNCH_PROFILE="${LAUNCH_PROFILE:-${ROOT_DIR}/scripts/lib/profiles/launch_compat.py}"
ESTATE_HELPER="${ESTATE_HELPER:-${ROOT_DIR}/scripts/lib/profiles/estate_cli.py}"
if [[ -z "${MODEL_DIR:-}" && -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/.env"
  set +a
fi
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models-cache}"
# shellcheck source=preflight.sh
source "${ROOT_DIR}/scripts/preflight.sh"

# --- arg parsing ---
ENGINE=""
WORKLOAD_ID=""
DRAFTER_ID="__unset__"
WEIGHTS_VARIANT=""
STABLE_ONLY=0
ESTATE_MODE=0
ESTATE_APPEND=0
ESTATE_REPLACE=""
ESTATE_FILE=""
VALIDATE_ESTATE=""
DOWN_ESTATE=""
ONLY_NAMES=""
TOPOLOGY_ONLY=0
PARALLEL_BOOT=0
PARALLEL_JOBS=""
PARALLEL_STAGGER=""
CARDS=""
VARIANT=""
MODEL_NAME=""
GPU_ARG=""
TP_OVERRIDE=""
PP_OVERRIDE=""
PARALLELISM="auto"
SKIP_VERIFY=0
SKIP_PREFLIGHT=0
SKIP_PROJECTION=0
VERBOSE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --single) ESTATE_MODE=0; shift ;;
    --estate) ESTATE_MODE=1; shift ;;
    --append) ESTATE_APPEND=1; ESTATE_MODE=1; shift ;;
    --replace) ESTATE_REPLACE="$2"; ESTATE_MODE=1; shift 2 ;;
    --estate-file) ESTATE_FILE="$2"; shift 2 ;;
    --validate-estate) VALIDATE_ESTATE="$2"; shift 2 ;;
    --down-estate) DOWN_ESTATE="$2"; shift 2 ;;
    --only) ONLY_NAMES="$2"; shift 2 ;;
    --topology) TOPOLOGY_ONLY=1; SKIP_PREFLIGHT=1; shift ;;
    --parallel) PARALLEL_BOOT=1; shift ;;
    --parallel-jobs) PARALLEL_BOOT=1; PARALLEL_JOBS="$2"; shift 2 ;;
    --parallel-stagger) PARALLEL_BOOT=1; PARALLEL_STAGGER="$2"; shift 2 ;;
    --engine)  ENGINE="$2"; shift 2 ;;
    --workload) WORKLOAD_ID="$2"; shift 2 ;;
    --drafter) DRAFTER_ID="$2"; shift 2 ;;
    --weights-variant) WEIGHTS_VARIANT="$2"; shift 2 ;;
    --stable)  STABLE_ONLY=1; shift ;;
    --cards)   CARDS="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --model)   MODEL_NAME="$2"; shift 2 ;;
    --gpus)    GPU_ARG="$2"; shift 2 ;;
    --tp)      TP_OVERRIDE="$2"; shift 2 ;;
    --pp)      PP_OVERRIDE="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --no-projection) SKIP_PROJECTION=1; shift ;;
    --no-verify) SKIP_VERIFY=1; shift ;;
    --no-preflight) SKIP_PREFLIGHT=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    --*) echo "Unknown flag: $1"; exit 1 ;;
    *)
      if [[ -n "$VARIANT" ]]; then
        echo "ERROR: multiple variants supplied: '${VARIANT}' and '$1'" >&2
        exit 1
      fi
      VARIANT="$1"; shift ;;
  esac
done

if [[ -n "$VALIDATE_ESTATE" ]]; then
  python3 "$ESTATE_HELPER" validate --file "$VALIDATE_ESTATE"
  exit $?
fi
if [[ -n "$DOWN_ESTATE" ]]; then
  _estate_down_cmd=(python3 "$ESTATE_HELPER" down --file "$DOWN_ESTATE")
  [[ -n "$ONLY_NAMES" ]] && _estate_down_cmd+=(--only "$ONLY_NAMES")
  "${_estate_down_cmd[@]}"
  exit $?
fi

# --- pre-flight ---
if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
  echo "[preflight] checking environment..."
  preflight_docker || exit 1
  preflight_gpu 1  || exit 1
  preflight_gpu_idle
  preflight_running
  preflight_genesis_pin "${ROOT_DIR}"
  preflight_repo_drift "${ROOT_DIR}"
  echo "[preflight] ok."
  echo ""
fi

ask() {
  # ask "prompt" "default" -> echoes user input or default
  local p="$1" d="${2:-}" reply
  if [[ -n "$d" ]]; then
    read -rp "$p [${d}]: " reply
    echo "${reply:-$d}"
  else
    read -rp "$p: " reply
    echo "$reply"
  fi
}

read_or_interrupt() {
  local prompt="$1"
  local __reply_var="$2"
  local reply
  if ! read -rp "$prompt" reply; then
    echo "" >&2
    echo "  EOF on stdin — wizard needs interactive input. Use --variant <name> to skip." >&2
    kill -INT $$
    exit 1
  fi
  printf -v "$__reply_var" '%s' "$reply"
}

choose() {
  # choose "prompt" "label1" "value1" "label2" "value2" ... -> echoes chosen value
  local prompt="$1"; shift
  local i=1 labels=() values=()
  while [[ $# -gt 0 ]]; do
    labels+=("$1"); values+=("$2"); shift 2
  done
  echo "" >&2
  echo "$prompt" >&2
  for l in "${labels[@]}"; do
    printf "  %d) %s\n" "$i" "$l" >&2
    i=$((i+1))
  done
  while true; do
    local pick
    if ! read -rp "Choice [1-${#labels[@]}]: " pick; then
      echo "" >&2
      echo "  EOF on stdin — wizard needs interactive input. Use --variant <name> to skip." >&2
      kill -INT $$
      exit 1
    fi
    if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= ${#labels[@]} )); then
      echo "${values[$((pick-1))]}"
      return
    fi
    echo "  invalid — pick a number 1-${#labels[@]}" >&2
  done
}

declare -A LAUNCH_VARIANT_COMPOSE=()
declare -A LAUNCH_VARIANT_MODEL=()
declare -A LAUNCH_VARIANT_ENGINE=()
declare -A LAUNCH_VARIANT_PROFILE_ENGINE=()
declare -A LAUNCH_VARIANT_KVCALC=()
declare -A LAUNCH_DEFAULT_PORT=()
declare -A LAUNCH_DEFAULT_CONTAINER=()
declare -A LAUNCH_VARIANT_STATUS=()
declare -A LAUNCH_VARIANT_STATUS_NOTE=()
LAUNCH_VARIANT_ORDER=()
PRIMARY_MODEL="${PRIMARY_MODEL:-qwen3.6-27b}"
# shellcheck source=lib/registry-emit.sh
source "${ROOT_DIR}/scripts/lib/registry-emit.sh"
derive_launch_variant_tables "${ROOT_DIR}"

variant_hw_status() {
  local variant="$1"
  local rel="${LAUNCH_VARIANT_COMPOSE[$variant]:-}"
  if [[ -z "$rel" ]]; then
    printf 'ok|fits your rig'
    return 0
  fi

  local compose_file="${ROOT_DIR}/${rel}"
  if [[ ! -f "$compose_file" ]]; then
    printf 'unknown|compose metadata unavailable'
    return 2
  fi
  compose_hw_compose_status "$compose_file" 2>/dev/null || true
}

choose_variant() {
  # choose_variant "prompt" "default-variant" "label1" "value1" ...
  local prompt="$1" default_variant="$2"
  shift 2

  local i labels=() values=() statuses=() eligible=()
  while [[ $# -gt 0 ]]; do
    labels+=("$1")
    values+=("$2")
    statuses+=("$(variant_hw_status "$2")")
    shift 2
  done

  local default_idx=""
  for i in "${!values[@]}"; do
    if [[ "${values[$i]}" == "$default_variant" && "${statuses[$i]}" == ok\|* ]]; then
      default_idx=$((i + 1))
      break
    fi
  done
  if [[ -z "$default_idx" ]]; then
    for i in "${!values[@]}"; do
      if [[ "${statuses[$i]}" == ok\|* || "${statuses[$i]}" == unknown\|* ]]; then
        default_idx=$((i + 1))
        break
      fi
    done
  fi

  echo "" >&2
  echo "$prompt" >&2
  for i in "${!labels[@]}"; do
    local status="${statuses[$i]}"
    local state="${status%%|*}"
    local reason="${status#*|}"
    local marker="✓"
    case "$state" in
      ok) marker="✓" ;;
      unknown) marker="?" ;;
      *) marker="✗" ;;
    esac
    if [[ -n "$default_idx" && $((i + 1)) -eq "$default_idx" ]]; then
      printf "  %d) %s  %s %s  [default]\n" "$((i + 1))" "${labels[$i]}" "$marker" "$reason" >&2
    else
      printf "  %d) %s  %s %s\n" "$((i + 1))" "${labels[$i]}" "$marker" "$reason" >&2
    fi
  done

  if [[ -z "$default_idx" ]]; then
    echo "ERROR: no eligible variants in this menu. Use scripts/switch.sh --force <variant> to attempt anyway." >&2
    exit 1
  fi

  while true; do
    local pick
    if ! read -rp "Choice [1-${#labels[@]}, default ${default_idx}]: " pick; then
      echo "" >&2
      echo "  EOF on stdin — wizard needs interactive input. Use --variant <name> to skip." >&2
      kill -INT $$
      exit 1
    fi
    pick="${pick:-$default_idx}"
    if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= ${#labels[@]} )); then
      local status="${statuses[$((pick - 1))]}"
      if [[ "$status" == no\|* ]]; then
        echo "  That variant won't run on your detected rig: ${status#*|}" >&2
        echo "  Pick another, or use:  bash scripts/switch.sh --force ${values[$((pick - 1))]}" >&2
        continue
      fi
      echo "${values[$((pick - 1))]}"
      return
    fi
    echo "  invalid — pick a number 1-${#labels[@]}" >&2
  done
}

model_label() {
  case "$1" in
    qwen3.6-27b) echo "Qwen 3.6 27B" ;;
    gemma-4-31b) echo "Gemma 4 31B" ;;
    *) echo "$1" ;;
  esac
}

normalize_model_name() {
  case "$1" in
    qwen3.6-27b|qwen3.6-27b-gguf) echo "qwen3.6-27b" ;;
    gemma-4-31b|gemma-4-31b-awq|gemma-4-31b-gguf) echo "gemma-4-31b" ;;
    *) echo "$1" ;;
  esac
}

MODEL_ORDER=()
declare -A MODEL_ENGINES=()

add_installed_model_engine() {
  local model="$1" engine="$2"
  if [[ -z "${MODEL_ENGINES[$model]:-}" ]]; then
    MODEL_ORDER+=("$model")
    MODEL_ENGINES[$model]="$engine"
  elif [[ ",${MODEL_ENGINES[$model]}," != *",${engine},"* ]]; then
    MODEL_ENGINES[$model]="${MODEL_ENGINES[$model]},${engine}"
  fi
}

detect_installed_models() {
  MODEL_ORDER=()
  MODEL_ENGINES=()
  [[ -d "${MODEL_DIR}/qwen3.6-27b-autoround-int4" ]] && add_installed_model_engine "qwen3.6-27b" "vllm"
  [[ -d "${MODEL_DIR}/qwen3.6-27b-gguf" ]] && add_installed_model_engine "qwen3.6-27b" "llamacpp"
  [[ -d "${MODEL_DIR}/gemma-4-31b-autoround-int4" ]] && add_installed_model_engine "gemma-4-31b" "vllm"
  [[ -d "${MODEL_DIR}/gemma-4-31b-it-AWQ-4bit" ]] && add_installed_model_engine "gemma-4-31b" "vllm"
  [[ -d "${MODEL_DIR}/gemma-4-31b-gguf" ]] && add_installed_model_engine "gemma-4-31b" "llamacpp"
  return 0
}

model_has_engine() {
  local model="$1" engine="$2"
  [[ ",${MODEL_ENGINES[$model]:-}," == *",${engine},"* ]]
}

engine_hint() {
  case "$1" in
    vllm,llamacpp|llamacpp,vllm) echo "vLLM + llama.cpp engines available" ;;
    vllm) echo "vLLM only" ;;
    llamacpp) echo "llama.cpp only" ;;
    *) echo "$1" ;;
  esac
}

choose_model() {
  detect_installed_models
  if [[ -n "$MODEL_NAME" ]]; then
    MODEL_NAME="$(normalize_model_name "$MODEL_NAME")"
    if [[ -z "${MODEL_ENGINES[$MODEL_NAME]:-}" ]]; then
      echo "[launch] ERROR: ${MODEL_NAME} is not installed under ${MODEL_DIR}." >&2
      echo "[launch]        Run: bash scripts/setup.sh ${MODEL_NAME}" >&2
      echo "[launch]        Already have weights elsewhere? Point MODEL_DIR at them, e.g.:" >&2
      echo "[launch]          echo 'MODEL_DIR=/path/to/your/models' >> .env   # launch.sh, switch.sh + docker compose all read it" >&2
      exit 1
    fi
    return
  fi
  if [[ "${#MODEL_ORDER[@]}" -eq 0 ]]; then
    echo "[launch] ERROR: no supported model weights found under ${MODEL_DIR}." >&2
    echo "[launch]        Run: bash scripts/setup.sh" >&2
    echo "[launch]        Already have weights elsewhere? Point MODEL_DIR at them, e.g.:" >&2
    echo "[launch]          echo 'MODEL_DIR=/path/to/your/models' >> .env   # launch.sh, switch.sh + docker compose all read it" >&2
    exit 1
  fi
  if [[ "${#MODEL_ORDER[@]}" -eq 1 ]]; then
    MODEL_NAME="${MODEL_ORDER[0]}"
    echo "[launch] using installed model: $(model_label "$MODEL_NAME")" >&2
    return
  fi
  echo "" >&2
  echo "[launch] Installed models:" >&2
  local i=1 model
  for model in "${MODEL_ORDER[@]}"; do
    printf "  %d) %-16s (%s)\n" "$i" "$(model_label "$model")" "$(engine_hint "${MODEL_ENGINES[$model]}")" >&2
    i=$((i + 1))
  done
  while true; do
    local pick
    read_or_interrupt "Choice [1-${#MODEL_ORDER[@]}]: " pick
    if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= ${#MODEL_ORDER[@]} )); then
      MODEL_NAME="${MODEL_ORDER[$((pick - 1))]}"
      return
    fi
    echo "  invalid — pick a number 1-${#MODEL_ORDER[@]}" >&2
  done
}

GPU_LINES=""
CARD_INDICES=()
CARD_NAMES=()
CARD_MEM_MIB=()
CARD_SM=()
MIN_VRAM_GB=0
MAX_VRAM_GB=0
HET_VRAM_MIXED=0
SELECTED_GPU_CSV=""
SELECTED_VRAM_SUMMARY=""

gpu_exists() {
  local want="$1" idx name mem_mib sm
  while IFS=$'\t' read -r idx name mem_mib sm; do
    [[ "$idx" == "$want" ]] && return 0
  done <<< "$GPU_LINES"
  return 1
}

gpu_is_busy() {
  local want="$1" busy
  while IFS= read -r busy; do
    [[ "$busy" == "$want" ]] && return 0
  done <<< "$(compose_hw_in_use_gpus 2>/dev/null || true)"
  return 1
}

append_selected_gpu() {
  local want="$1" idx name mem_mib sm
  while IFS=$'\t' read -r idx name mem_mib sm; do
    if [[ "$idx" == "$want" ]]; then
      CARD_INDICES+=("$idx")
      CARD_NAMES+=("$name")
      CARD_MEM_MIB+=("$mem_mib")
      CARD_SM+=("$sm")
      return 0
    fi
  done <<< "$GPU_LINES"
  return 1
}

select_gpus_from_arg() {
  local arg="$1"
  CARD_INDICES=()
  CARD_NAMES=()
  CARD_MEM_MIB=()
  CARD_SM=()
  local available=() idx name mem_mib sm
  while IFS=$'\t' read -r idx name mem_mib sm; do
    [[ -z "$idx" ]] && continue
    if ! gpu_is_busy "$idx"; then
      available+=("$idx")
    fi
  done <<< "$GPU_LINES"
  if [[ "$arg" == "all" ]]; then
    [[ "${#available[@]}" -gt 0 ]] || { echo "[launch] ERROR: no available NVIDIA GPUs detected." >&2; exit 1; }
    for idx in "${available[@]}"; do append_selected_gpu "$idx"; done
    return
  fi
  IFS=',' read -ra _launch_gpu_tokens <<< "$arg"
  for idx in "${_launch_gpu_tokens[@]}"; do
    idx="$(_compose_meta_trim "$idx")"
    [[ -z "$idx" ]] && continue
    gpu_exists "$idx" || { echo "[launch] ERROR: requested GPU ${idx}, but it was not detected." >&2; exit 1; }
    append_selected_gpu "$idx"
  done
}

summarize_selected_vram() {
  local parts=() i gb
  MIN_VRAM_GB=0
  MAX_VRAM_GB=0
  HET_VRAM_MIXED=0
  for i in "${!CARD_INDICES[@]}"; do
    gb="$(compose_hw_vram_gb "${CARD_MEM_MIB[$i]}")"
    parts+=("${gb} GB")
    if [[ "$MIN_VRAM_GB" -eq 0 || "$gb" -lt "$MIN_VRAM_GB" ]]; then MIN_VRAM_GB="$gb"; fi
    if [[ "$gb" -gt "$MAX_VRAM_GB" ]]; then MAX_VRAM_GB="$gb"; fi
  done
  [[ "$MIN_VRAM_GB" != "$MAX_VRAM_GB" ]] && HET_VRAM_MIXED=1
  local joined="" part
  for part in "${parts[@]}"; do
    [[ -n "$joined" ]] && joined="${joined} + "
    joined="${joined}${part}"
  done
  SELECTED_VRAM_SUMMARY="$joined"
  printf '%s' "$joined"
}

choose_gpus() {
  GPU_LINES="$(compose_hw_detect_gpus 2>/dev/null || true)"
  [[ -n "$GPU_LINES" ]] || { echo "[launch] ERROR: no NVIDIA GPUs detected." >&2; exit 1; }
  local available=() idx name mem_mib sm state
  echo "" >&2
  echo "[launch] Detected GPUs:" >&2
  while IFS=$'\t' read -r idx name mem_mib sm; do
    [[ -z "$idx" ]] && continue
    state="available"
    if gpu_is_busy "$idx"; then
      state="in-use (skipped)"
    else
      available+=("$idx")
    fi
    printf "  GPU %s: %s (%s GB, sm_%s) — %s\n" "$idx" "${name#NVIDIA }" "$(compose_hw_vram_gb "$mem_mib")" "${sm/./}" "$state" >&2
  done <<< "$GPU_LINES"

  if [[ -n "$GPU_ARG" ]]; then
    select_gpus_from_arg "$GPU_ARG"
  elif [[ -n "$CARDS" ]]; then
    [[ "$CARDS" =~ ^[0-9]+$ && "$CARDS" -ge 1 ]] || { echo "[launch] ERROR: --cards expects a positive integer." >&2; exit 1; }
    (( CARDS <= ${#available[@]} )) || { echo "[launch] ERROR: --cards ${CARDS} requested, but only ${#available[@]} GPU(s) are available." >&2; exit 1; }
    local i
    for ((i = 0; i < CARDS; i++)); do append_selected_gpu "${available[$i]}"; done
  else
    case "${#available[@]}" in
      0) echo "[launch] ERROR: no available NVIDIA GPUs detected." >&2; exit 1 ;;
      1) select_gpus_from_arg "${available[0]}" ;;
      2)
        local pick
        read_or_interrupt "Use GPU ${available[0]}, GPU ${available[1]}, or both? [both]: " pick
        pick="${pick:-both}"
        case "$pick" in
          both|all) select_gpus_from_arg "${available[0]},${available[1]}" ;;
          "${available[0]}"|"${available[1]}") select_gpus_from_arg "$pick" ;;
          *) echo "[launch] ERROR: invalid GPU choice: $pick" >&2; exit 1 ;;
        esac
        ;;
      *)
        local default_csv pick
        default_csv="$(IFS=','; echo "${available[*]}")"
        read_or_interrupt "Which GPU(s)? (comma-separated indices, or 'all') [all]: " pick
        pick="${pick:-all}"
        [[ "$pick" == "all" ]] && pick="$default_csv"
        select_gpus_from_arg "$pick"
        ;;
    esac
  fi
  [[ "${#CARD_INDICES[@]}" -gt 0 ]] || { echo "[launch] ERROR: no GPUs selected." >&2; exit 1; }
  SELECTED_GPU_CSV="$(IFS=','; echo "${CARD_INDICES[*]}")"
  summarize_selected_vram >/dev/null
  echo "[launch] selected GPU(s): ${SELECTED_GPU_CSV} (${SELECTED_VRAM_SUMMARY})" >&2
}

selected_gpu_profile_spec() {
  local parts=() i
  for i in "${!CARD_INDICES[@]}"; do
    parts+=("${CARD_INDICES[$i]}|${CARD_NAMES[$i]}|${CARD_MEM_MIB[$i]}|${CARD_SM[$i]}")
  done
  local joined
  joined="$(IFS=';'; echo "${parts[*]}")"
  printf '%s' "$joined"
}

select_topology_gpus() {
  GPU_LINES="$(compose_hw_detect_gpus 2>/dev/null || true)"
  [[ -n "$GPU_LINES" ]] || return 1
  CARD_INDICES=()
  CARD_NAMES=()
  CARD_MEM_MIB=()
  CARD_SM=()

  if [[ -n "$GPU_ARG" && "$GPU_ARG" != "all" ]]; then
    IFS=',' read -ra _launch_topology_tokens <<< "$GPU_ARG"
    local idx
    for idx in "${_launch_topology_tokens[@]}"; do
      idx="$(_compose_meta_trim "$idx")"
      [[ -z "$idx" ]] && continue
      gpu_exists "$idx" || { echo "[launch] ERROR: requested GPU ${idx}, but it was not detected." >&2; exit 1; }
      append_selected_gpu "$idx"
    done
  elif [[ -n "$CARDS" ]]; then
    [[ "$CARDS" =~ ^[0-9]+$ && "$CARDS" -ge 1 ]] || { echo "[launch] ERROR: --cards expects a positive integer." >&2; exit 1; }
    local idx name mem_mib sm selected=0
    while IFS=$'\t' read -r idx name mem_mib sm; do
      [[ -z "$idx" ]] && continue
      append_selected_gpu "$idx"
      selected=$((selected + 1))
      (( selected >= CARDS )) && break
    done <<< "$GPU_LINES"
    (( selected == CARDS )) || { echo "[launch] ERROR: --cards ${CARDS} requested, but only ${selected} GPU(s) were detected." >&2; exit 1; }
  else
    local idx name mem_mib sm
    while IFS=$'\t' read -r idx name mem_mib sm; do
      [[ -z "$idx" ]] && continue
      append_selected_gpu "$idx"
    done <<< "$GPU_LINES"
  fi

  [[ "${#CARD_INDICES[@]}" -gt 0 ]] || return 1
  SELECTED_GPU_CSV="$(IFS=','; echo "${CARD_INDICES[*]}")"
  summarize_selected_vram >/dev/null
  return 0
}

print_topology_advisory() {
  local output
  output="$(python3 "$LAUNCH_PROFILE" topology --gpu-spec "$(selected_gpu_profile_spec)" --format wizard 2>&1)" || {
    echo "$output" >&2
    exit 2
  }
  if [[ -n "$output" ]]; then
    echo "$output" >&2
  fi
}

print_topology_and_exit() {
  local output
  if ! select_topology_gpus; then
    echo "Detected hardware:"
    echo "  no NVIDIA GPUs detected"
    echo ""
    echo "Topology class: unavailable"
    echo ""
    echo "For details, see docs/MULTI_CARD.md."
    exit 0
  fi
  output="$(python3 "$LAUNCH_PROFILE" topology --gpu-spec "$(selected_gpu_profile_spec)" --format standalone 2>&1)" || {
    echo "$output" >&2
    exit 0
  }
  echo "$output"
  exit 0
}

launch_nvlink_active() {
  if [[ "${#CARD_INDICES[@]}" -ne 2 ]]; then
    printf '0'
    return
  fi
  if [[ "${NVLINK_MODE:-auto}" == "force_on" ]]; then
    printf '1'
    return
  fi
  if [[ "${NVLINK_MODE:-auto}" == "force_off" ]]; then
    printf '0'
    return
  fi
  (
    # shellcheck source=detect_nvlink.sh
    source "${ROOT_DIR}/scripts/detect_nvlink.sh" >/dev/null 2>&1 || true
    printf '%s' "${_NVLINK_ENABLED:-0}"
  )
}

valid_tp_values() {
  case "$1" in
    qwen3.6-27b) echo "1 2 4" ;;
    gemma-4-31b) echo "1 2 4 8 16" ;;
    *) echo "1" ;;
  esac
}

tp_is_valid_for_model() {
  local model="$1" tp="$2" v
  for v in $(valid_tp_values "$model"); do [[ "$v" == "$tp" ]] && return 0; done
  return 1
}

largest_valid_tp_for_cards() {
  local model="$1" cards="$2" v best=1
  for v in $(valid_tp_values "$model"); do
    if (( v <= cards && cards % v == 0 && v > best )); then best="$v"; fi
  done
  echo "$best"
}

TP_VALUE=""
PP_VALUE=""

pick_parallelism() {
  local cards="${#CARD_INDICES[@]}"
  local tp_set=0 pp_set=0
  [[ -n "$TP_OVERRIDE" ]] && tp_set=1
  [[ -n "$PP_OVERRIDE" ]] && pp_set=1
  [[ -z "$TP_OVERRIDE" || "$TP_OVERRIDE" =~ ^[0-9]+$ ]] || { echo "[launch] ERROR: --tp expects an integer." >&2; exit 1; }
  [[ -z "$PP_OVERRIDE" || "$PP_OVERRIDE" =~ ^[0-9]+$ ]] || { echo "[launch] ERROR: --pp expects an integer." >&2; exit 1; }
  case "$PARALLELISM" in auto|tp|pp) ;; *) echo "[launch] ERROR: --parallelism expects auto, tp, or pp." >&2; exit 1 ;; esac

  if (( cards == 1 )); then
    TP_VALUE="${TP_OVERRIDE:-1}"
    PP_VALUE="${PP_OVERRIDE:-1}"
  elif (( tp_set == 1 && pp_set == 1 )); then
    TP_VALUE="$TP_OVERRIDE"; PP_VALUE="$PP_OVERRIDE"
  elif (( tp_set == 1 )); then
    TP_VALUE="$TP_OVERRIDE"
    (( cards % TP_VALUE == 0 )) || { echo "[launch] ERROR: --tp ${TP_VALUE} does not divide selected GPU count ${cards}." >&2; exit 1; }
    PP_VALUE=$(( cards / TP_VALUE ))
  elif (( pp_set == 1 )); then
    PP_VALUE="$PP_OVERRIDE"
    (( cards % PP_VALUE == 0 )) || { echo "[launch] ERROR: --pp ${PP_VALUE} does not divide selected GPU count ${cards}." >&2; exit 1; }
    TP_VALUE=$(( cards / PP_VALUE ))
  elif [[ "$PARALLELISM" == "pp" ]]; then
    TP_VALUE=1; PP_VALUE="$cards"
  elif [[ "$PARALLELISM" == "tp" ]]; then
    TP_VALUE="$cards"; PP_VALUE=1
  elif (( HET_VRAM_MIXED == 1 )); then
    TP_VALUE=1; PP_VALUE="$cards"
    echo "[launch] ${cards} GPUs selected (${MIN_VRAM_GB} GB + ${MAX_VRAM_GB} GB — heterogeneous)." >&2
    echo "[launch] Recommended: pipeline parallel PP=${PP_VALUE} to avoid bottlenecking on the smallest card." >&2
  else
    TP_VALUE="$(largest_valid_tp_for_cards "$MODEL_NAME" "$cards")"
    PP_VALUE=$(( cards / TP_VALUE ))
  fi

  (( TP_VALUE * PP_VALUE == cards )) || { echo "[launch] ERROR: TP × PP must equal selected GPU count (${TP_VALUE} × ${PP_VALUE} != ${cards})." >&2; exit 1; }
  if ! tp_is_valid_for_model "$MODEL_NAME" "$TP_VALUE"; then
    echo "[launch] ERROR: $(model_label "$MODEL_NAME") num_kv_heads does not divide TP=${TP_VALUE}." >&2
    echo "[launch]        Valid TP values: $(valid_tp_values "$MODEL_NAME")" >&2
    exit 1
  fi
  if (( cards > 1 && PP_VALUE == 1 )); then
    echo "[launch] Tensor parallel TP=${TP_VALUE} (PP=1)." >&2
  elif (( PP_VALUE > 1 )); then
    echo "[launch] Pipeline parallel PP=${PP_VALUE}, TP=${TP_VALUE}." >&2
    echo "[launch] WARN: pipeline parallel is experimental on this stack — no benchmarks yet." >&2
  fi
}

variant_min_gpu_count() {
  local rel="${LAUNCH_VARIANT_COMPOSE[$1]:-}" value
  [[ -n "$rel" && -f "${ROOT_DIR}/${rel}" ]] || { echo 1; return; }
  value="$(compose_meta_get "${ROOT_DIR}/${rel}" requires-min-gpu-count || true)"
  echo "${value:-1}"
}

variant_min_vram_gb() {
  local rel="${LAUNCH_VARIANT_COMPOSE[$1]:-}" value
  [[ -n "$rel" && -f "${ROOT_DIR}/${rel}" ]] || { echo 0; return; }
  value="$(compose_meta_get "${ROOT_DIR}/${rel}" requires-min-vram-gb || true)"
  echo "${value:-0}"
}

variant_engine_available() {
  local variant="$1" engine
  engine="${LAUNCH_VARIANT_ENGINE[$variant]:-vllm}"
  [[ -z "$ENGINE" || "$ENGINE" == "$engine" ]] || return 1
  model_has_engine "$MODEL_NAME" "$engine"
}

variant_engine_installed() {
  local variant="$1" engine
  engine="${LAUNCH_VARIANT_ENGINE[$variant]:-vllm}"
  model_has_engine "$MODEL_NAME" "$engine"
}

variant_survives_filter() {
  local variant="$1"
  [[ "${LAUNCH_VARIANT_MODEL[$variant]:-}" == "$MODEL_NAME" ]] || return 1
  variant_engine_available "$variant" || return 1
  local min_gpu min_vram engine
  min_gpu="$(variant_min_gpu_count "$variant")"
  min_vram="$(variant_min_vram_gb "$variant")"
  engine="${LAUNCH_VARIANT_ENGINE[$variant]:-vllm}"
  [[ "$engine" == "llamacpp" && "${#CARD_INDICES[@]}" -ne 1 ]] && return 1
  (( min_gpu <= ${#CARD_INDICES[@]} )) || return 1
  (( min_vram == 0 || min_vram <= MIN_VRAM_GB )) || return 1
  return 0
}

profile_filter_candidates() {
  local variant_csv candidate_text line use_runtime=0 nvlink_flag=()
  variant_csv="$(IFS=','; echo "${LAUNCH_VARIANT_ORDER[*]}")"
  [[ -n "$TP_OVERRIDE" || -n "$PP_OVERRIDE" ]] && use_runtime=1
  [[ "$(launch_nvlink_active)" == "1" ]] && nvlink_flag=(--nvlink-active)

  local cmd=(
    python3 "$LAUNCH_PROFILE" filter-candidates
    --variants "$variant_csv"
    --model "$MODEL_NAME"
    --gpu-spec "$(selected_gpu_profile_spec)"
    --tp "$TP_VALUE"
    --pp "$PP_VALUE"
    --drafter "$DRAFTER_ID"
  )
  [[ -n "$ENGINE" ]] && cmd+=(--engine "$ENGINE")
  [[ -n "$WORKLOAD_ID" ]] && cmd+=(--workload "$WORKLOAD_ID")
  [[ -n "$WEIGHTS_VARIANT" ]] && cmd+=(--weights-variant "$WEIGHTS_VARIANT")
  [[ "$STABLE_ONLY" -eq 1 ]] && cmd+=(--stable)
  [[ "$use_runtime" -eq 1 ]] && cmd+=(--use-runtime-parallelism)
  [[ "$VERBOSE" -eq 1 ]] && cmd+=(--verbose)
  cmd+=("${nvlink_flag[@]}")

  if ! candidate_text="$("${cmd[@]}")"; then
    echo "$candidate_text" >&2
    exit 2
  fi

  CANDIDATE_VARIANTS=()
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    if variant_engine_installed "$line"; then
      CANDIDATE_VARIANTS+=("$line")
    fi
  done <<< "$candidate_text"
}

suggest_default_variant() {
  local cards="${#CARD_INDICES[@]}"
  if [[ "$MODEL_NAME" == "qwen3.6-27b" ]]; then
    if [[ "$ENGINE" == "llamacpp" ]] || { ! model_has_engine "$MODEL_NAME" "vllm" && model_has_engine "$MODEL_NAME" "llamacpp"; }; then
      echo "llamacpp/default"
    elif (( cards >= 4 )); then
      echo "vllm/dual4"
    elif (( cards >= 2 )); then
      echo "vllm/dual"
    else
      # Single card: llamacpp/default is the recommended path — full 262K, cliff-immune,
      # and no purged-nightly dependency. The old vllm/long-text suggestion is dead
      # (#167 image purge + single-card Cliff 2b); vLLM single-card users can still pick
      # vllm/tools-text explicitly.
      echo "llamacpp/default"
    fi
  else
    if (( cards >= 2 )); then echo "vllm/gemma-mtp"; else echo "vllm/gemma-mtp-tp1"; fi
  fi
}


launch_topology_from_selected() {
  local cards="${#CARD_INDICES[@]}"
  case "$cards" in
    0|1) printf 'single' ;;
    2) printf 'dual' ;;
    4) printf 'multi4' ;;
    *) printf 'multi%s' "$cards" ;;
  esac
}

resolve_launch_default_variant() {
  local variant="$1" engine topology target
  if [[ "$variant" =~ ^([^/]+)/(single|dual|multi[0-9]+)/default$ ]]; then
    engine="${BASH_REMATCH[1]}"
    topology="${BASH_REMATCH[2]}"
  elif [[ "$variant" =~ ^([^/]+)/default$ ]]; then
    engine="${BASH_REMATCH[1]}"
    if [[ "${#CARD_INDICES[@]}" -eq 0 ]]; then
      select_topology_gpus || {
        echo "[launch] ERROR: cannot autodetect topology for '${variant}' because no GPUs were detected." >&2
        exit 1
      }
    fi
    topology="$(launch_topology_from_selected)"
  else
    printf '%s' "$variant"
    return 0
  fi
  MODEL_NAME="${MODEL_NAME:-$PRIMARY_MODEL}"
  MODEL_NAME="$(normalize_model_name "$MODEL_NAME")"
  if ! target="$(registry_default_target "$ROOT_DIR" "$MODEL_NAME" "$engine" "$topology")"; then
    echo "[launch] ERROR: cannot resolve default variant '${variant}' for model ${MODEL_NAME}." >&2
    exit 1
  fi
  printf '%s' "$target"
}

no_fit_guidance() {
  echo "[launch] Selected GPU budget: ${MIN_VRAM_GB} GB minimum per card." >&2
  echo "" >&2
  echo "No shipped model variant fits this GPU selection:" >&2
  echo "  Qwen 3.6 27B (INT4): needs >=20 GB" >&2
  echo "  Gemma 4 31B (INT4):  needs >=32 GB single-card or 2x24 GB" >&2
  echo "" >&2
  echo "Your options:" >&2
  echo "  1) Combine with another GPU." >&2
  echo "  2) Try llama.cpp with a smaller GGUF. See docs/SINGLE_CARD.md#sub-16gb" >&2
  echo "  3) Re-run with --gpus to pick a different card set." >&2
  exit 2
}

gemma_single_24gb_guidance() {
  echo "[launch] Selected: Gemma 4 31B on GPU ${SELECTED_GPU_CSV} (${MIN_VRAM_GB} GB)." >&2
  echo "" >&2
  echo "Gemma 4 31B does not fit on a single ${MIN_VRAM_GB} GB card today." >&2
  echo "Reason: vLLM's Gemma 4 single-card path needs >=32 GB; use TP=2 on 2x24 GB." >&2
  echo "" >&2
  echo "Your options:" >&2
  echo "  1) Re-run with --gpus 0,1 for TP=2 if you have two 24 GB cards." >&2
  echo "  2) Use Qwen 3.6 27B for single-card: bash scripts/launch.sh --model qwen3.6-27b" >&2
  exit 2
}

json_field() {
  local field="$1"
  python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get(sys.argv[1], ""))' "$field"
}

kv_projection() {
  local variant="$1"
  [[ "$SKIP_PROJECTION" -eq 0 ]] || return 0
  local mapping="${LAUNCH_VARIANT_KVCALC[$variant]:-}"
  if [[ -z "$mapping" || "$mapping" == "SKIP" ]]; then
    echo "[launch] KV projection only available for vLLM variants today." >&2
    return 0
  fi
  local kv_model="${mapping%%:*}" kv_compose="${mapping#*:}" kv_json status
  if kv_json="$("${ROOT_DIR}/tools/kv-calc.py" --model "$kv_model" --compose "$kv_compose" --vram "$MIN_VRAM_GB" --tp "$TP_VALUE" --json 2>&1)"; then
    status=0
  else
    status=$?
  fi
  if [[ "$kv_json" != \{* ]]; then
    echo "[launch] WARN: kv-calc failed for ${variant}: ${kv_json}" >&2
    return 0
  fi

  local verdict weights kv_pool activation overhead drafter total budget pct
  verdict="$(json_field verdict <<< "$kv_json")"
  weights="$(json_field weights_gb <<< "$kv_json")"
  kv_pool="$(json_field kv_pool_actual_gb <<< "$kv_json")"
  activation="$(json_field activation_gb <<< "$kv_json")"
  overhead="$(json_field cudagraph_overhead_gb <<< "$kv_json")"
  drafter="$(json_field drafter_gb <<< "$kv_json")"
  total="$(json_field total_gb <<< "$kv_json")"
  budget="$(json_field budget_gb <<< "$kv_json")"
  pct="$(json_field pct_of_vram <<< "$kv_json")"

  echo "" >&2
  echo "[launch] Suggested: ${variant} ($(model_label "$MODEL_NAME") on GPU ${SELECTED_GPU_CSV}, TP=${TP_VALUE} PP=${PP_VALUE})" >&2
  echo "" >&2
  echo "VRAM budget — per card (~${MIN_VRAM_GB} GB):" >&2
  printf "  Weights/TP=%s:       %.2f GB\n" "$TP_VALUE" "$weights" >&2
  printf "  KV pool:            %.2f GB\n" "$kv_pool" >&2
  printf "  Activations:        %.2f GB\n" "$activation" >&2
  printf "  Cudagraph + NCCL:   %.2f GB\n" "$overhead" >&2
  if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) > 0.01 else 1)' "$drafter"; then
    printf "  Drafter:            %.2f GB\n" "$drafter" >&2
  fi
  echo "  --------------" >&2
  printf "  Predicted peak:     %.2f GB  %s (%.0f%% of %.2f GB engine budget)\n" "$total" "$verdict" "$pct" "$budget" >&2
  if (( PP_VALUE > 1 )); then
    echo "  Note: PP is not modelled in projection; real per-card weights should be lower." >&2
  fi
  python3 -c 'import json,sys; data=json.load(sys.stdin); [print("  Note: " + n) for n in data.get("notes", [])]' <<< "$kv_json" >&2
  if [[ "$verdict" == "FAIL" || "$status" -ne 0 ]]; then
    echo "" >&2
    echo "[launch] Projection says this variant will not fit. Pick another GPU set or model." >&2
    exit 2
  fi
}

validate_selected_variant() {
  [[ "${#CARD_INDICES[@]}" -gt 0 ]] || return 0
  [[ -n "$TP_VALUE" && -n "$PP_VALUE" ]] || return 0
  [[ -x "$LAUNCH_PROFILE" || -f "$LAUNCH_PROFILE" ]] || {
    echo "[launch] ERROR: profile launch helper missing: ${LAUNCH_PROFILE}" >&2
    exit 2
  }

  local project_flag="--project-vram" nvlink_flag=()
  [[ "$SKIP_PROJECTION" -eq 1 ]] && project_flag="--no-project-vram"
  [[ "$(launch_nvlink_active)" == "1" ]] && nvlink_flag=(--nvlink-active)

  local cmd=(
    python3 "$LAUNCH_PROFILE" validate-variant
    --variant "$VARIANT"
    --gpu-spec "$(selected_gpu_profile_spec)"
    --tp "$TP_VALUE"
    --pp "$PP_VALUE"
    "$project_flag"
  )
  [[ "$VERBOSE" -eq 1 ]] && cmd+=(--verbose)
  cmd+=("${nvlink_flag[@]}")

  "${cmd[@]}"
}

export_variant_engine_pin() {
  local variant="$1" output line key value
  [[ "$variant" == vllm/* ]] || return 0
  if ! output="$(python3 "$LAUNCH_PROFILE" resolve-variant-pin --variant "$variant" --format shell 2>&1)"; then
    echo "$output" >&2
    exit 2
  fi
  while IFS='=' read -r key value; do
    [[ -n "$key" ]] || continue
    case "$key" in
      VLLM_NIGHTLY_SHA) export VLLM_NIGHTLY_SHA="$value" ;;
      VLLM_IMAGE) export VLLM_IMAGE="$value" ;;
      *) echo "[launch] ERROR: unexpected engine pin export: $key" >&2; exit 2 ;;
    esac
  done <<< "$output"
}

if [[ "$ESTATE_MODE" -eq 1 || -n "$ESTATE_FILE" ]]; then
  if [[ "$ESTATE_MODE" -eq 1 ]]; then
    _estate_cmd=(python3 "$ESTATE_HELPER" wizard)
    [[ -n "$ESTATE_FILE" ]] && _estate_cmd+=(--file "$ESTATE_FILE")
    [[ "$ESTATE_APPEND" -eq 1 ]] && _estate_cmd+=(--append)
    [[ -n "$ESTATE_REPLACE" ]] && _estate_cmd+=(--replace "$ESTATE_REPLACE")
  else
    _estate_cmd=(python3 "$ESTATE_HELPER" boot --file "$ESTATE_FILE")
    [[ -n "$ONLY_NAMES" ]] && _estate_cmd+=(--only "$ONLY_NAMES")
    [[ "$PARALLEL_BOOT" -eq 1 ]] && _estate_cmd+=(--parallel)
    [[ -n "$PARALLEL_JOBS" ]] && _estate_cmd+=(--parallel-jobs "$PARALLEL_JOBS")
    [[ -n "$PARALLEL_STAGGER" ]] && _estate_cmd+=(--parallel-stagger "$PARALLEL_STAGGER")
  fi
  "${_estate_cmd[@]}"
  exit $?
fi

if [[ "$TOPOLOGY_ONLY" -eq 1 ]]; then
  print_topology_and_exit
fi

# --- wizard ---
if [[ -n "$VARIANT" ]]; then
  VARIANT="$(resolve_launch_default_variant "$VARIANT")"
fi
if [[ -z "$VARIANT" ]]; then
  echo "" >&2
  echo "club-3090 launcher — pick model, GPU set, and serving variant." >&2
  echo "(Use --variant <name> next time to skip the wizard.)" >&2
  choose_model
  choose_gpus
  print_topology_advisory
  pick_parallelism
  if [[ "$MODEL_NAME" == "gemma-4-31b" && "${#CARD_INDICES[@]}" -eq 1 && "$MIN_VRAM_GB" -lt 32 ]]; then
    gemma_single_24gb_guidance
  fi

  profile_filter_candidates
  [[ "${#CANDIDATE_VARIANTS[@]}" -gt 0 ]] || no_fit_guidance

  VARIANT="$(suggest_default_variant)"
  if [[ " ${CANDIDATE_VARIANTS[*]} " != *" ${VARIANT} "* ]]; then
    VARIANT="${CANDIDATE_VARIANTS[0]}"
  fi
  echo "[launch] model: $(model_label "$MODEL_NAME")" >&2
  if (( HET_VRAM_MIXED == 1 && TP_VALUE > 1 )); then
    echo "[launch] Note: heterogeneous TP is bottlenecked by the smallest selected card (${MIN_VRAM_GB} GB)." >&2
  fi
  if (( PP_VALUE > 1 )) && [[ "$VARIANT" == vllm/* ]]; then
    echo "[launch] WARN: PP + vLLM drafter/spec-decode paths are experimental on this stack." >&2
  fi
  kv_projection "$VARIANT"
  validate_selected_variant

  other_variants=()
  for candidate in "${CANDIDATE_VARIANTS[@]}"; do
    [[ "$candidate" == "$VARIANT" ]] && continue
    other_variants+=("$candidate")
  done
  if [[ "${#other_variants[@]}" -gt 0 ]]; then
    echo "[launch] Other variants that fit this selection: ${other_variants[*]}" >&2
  fi
else
  if [[ -n "$GPU_ARG" ]]; then
    choose_gpus
  fi
  if [[ -n "$TP_OVERRIDE" || -n "$PP_OVERRIDE" ]]; then
    if [[ "${#CARD_INDICES[@]}" -eq 0 ]]; then
      TP_VALUE="${TP_OVERRIDE:-1}"
      PP_VALUE="${PP_OVERRIDE:-1}"
    else
      MODEL_NAME="${LAUNCH_VARIANT_MODEL[$VARIANT]:-qwen3.6-27b}"
      summarize_selected_vram >/dev/null
      pick_parallelism
    fi
  fi
fi

# --- launch + verify ---
echo ""
echo "[launch] selected variant: ${VARIANT}"
# Surface the lifecycle health flag (PR-A) before handing off. The actual gate
# (caveats notice / (NA) → require --force) lives in switch.sh up_variant, which
# this script delegates to; this is just a heads-up so the user isn't surprised.
_launch_status="${LAUNCH_VARIANT_STATUS[$VARIANT]:-production}"
_launch_status_note="${LAUNCH_VARIANT_STATUS_NOTE[$VARIANT]:-}"
case "$_launch_status" in
  production) ;;
  caveats)
    echo "[launch] NOTE: this is a ⚠️ production-with-caveats config.${_launch_status_note:+  ${_launch_status_note}}"
    ;;
  *)
    echo "[launch] WARNING: this is a (NA: ${_launch_status}) config — not a reliable path.${_launch_status_note:+  ${_launch_status_note}}"
    echo "[launch]          switch.sh will require --force (FORCE=1) to bring it up."
    ;;
esac
echo ""
if [[ -n "$SELECTED_GPU_CSV" ]]; then
  export CUDA_VISIBLE_DEVICES="$SELECTED_GPU_CSV"
  export NVIDIA_VISIBLE_DEVICES="$SELECTED_GPU_CSV"
fi
if [[ -n "$TP_VALUE" ]]; then
  export TP="$TP_VALUE"
fi
if [[ -n "$PP_VALUE" ]]; then
  export PP="$PP_VALUE"
fi
export_variant_engine_pin "$VARIANT"
"$SWITCH" "$VARIANT"

ENDPOINT_PORT="${PORT:-${LAUNCH_DEFAULT_PORT[$VARIANT]:-8020}}"
ENDPOINT_URL="http://localhost:${ENDPOINT_PORT}"
ENDPOINT_CONTAINER="${CONTAINER:-${LAUNCH_DEFAULT_CONTAINER[$VARIANT]:-vllm-qwen36-27b}}"

if [[ $SKIP_VERIFY -eq 1 ]]; then
  echo "[launch] --no-verify — skipping verify-full.sh"
else
  echo ""
  echo "[launch] running verify-full.sh against the new server (URL=${ENDPOINT_URL}, CONTAINER=${ENDPOINT_CONTAINER})..."
  echo ""
  URL="$ENDPOINT_URL" CONTAINER="$ENDPOINT_CONTAINER" bash "$VERIFY" || {
    echo ""
    echo "[launch] some checks failed — see hints above. Common cases:"
    echo "  - 'reasoning field empty' on llama.cpp = expected (parser gap, not a bug)"
    echo "  - 'Genesis patches' / 'MTP acceptance' skipped on llama.cpp = expected (vLLM-only checks)"
    exit 1
  }
fi

echo ""
echo "[launch] done. Endpoint: ${ENDPOINT_URL}"
echo "[launch] sample request:"
echo "  curl -sf ${ENDPOINT_URL}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen3.6-27b-autoround\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":200}'"
echo ""
echo "[launch] switch later with:  bash scripts/switch.sh <variant>"
echo "[launch] list variants:      bash scripts/switch.sh --list"
