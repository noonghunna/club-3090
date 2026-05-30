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
#   bash scripts/switch.sh --force <variant>   # skip hardware/free-VRAM preflight
#   bash scripts/switch.sh --list              # show all variants
#   bash scripts/switch.sh --down              # just bring down whatever's up
#
# Variant names are derived from the compose registry (the single source of
# truth); `bash scripts/switch.sh --list` is authoritative. A representative
# subset (engine/file, file is the docker-compose.<file>.yml stem):
#
#   Single-card vLLM:
#     vllm/default            48K + TQ3 + MTP + vision + tools (recommended)
#     vllm/long-vision        198K + TQ3 + vision (cliff-safe; Cliff 2 single-prompt >50K still applies)
#     vllm/long-text          180K + TQ3 + MTP + text-only (Balanced MTP — 60K single-prompt closed via v7.69 + #35975)
#     vllm/long-text-no-mtp   200K + TQ3 + no MTP + text-only (Max-context — same Cliff 2 closure, more KV pool, slower decode)
#     vllm/bounded-thinking   180K + TQ3 + structured-CoT FSM in reasoning (recommended grammar: DeepSeek scratchpad — 87.4% combined HE+/LCB v6)
#     vllm/tools-text         75K + fp8 + MTP + text-only (IDE agents — Cline / Cursor)
#     vllm/minimal            32K + fp8 (no Genesis, no spec-decode, simplest)
#
#   Dual-card vLLM (TP=2):
#     vllm/dual             262K + fp8 + 2 streams + vision (recommended dual)
#     vllm/dual4            262K + fp8 + 4 streams + vision (4× 3090 PCIe baseline)
#     vllm/dual4-dflash     262K + FP16 + DFlash N=5 + 2 streams + vision (4× 3090 code)
#     vllm/dual-turbo       262K + TQ3 + 4 streams + vision (multi-tenant)
#     vllm/dual-dflash      185K + FP16 + DFlash N=5 + vision (peak code TPS)
#     vllm/dual-dflash-noviz 200K + FP16 + DFlash N=5 + no vision (peak code, max ctx)
#     (NVLink is auto-detected at boot by every dual compose — no separate
#      nvlink-* variant. Force it with NVLINK_MODE=force_on if auto-detect misses.)
#     vllm/gemma-mtp        Gemma-4-31B + Google MTP drafter (32K, bf16 KV, vision — community/experimental, pre-merge)
#
#   Single-card llama.cpp:
#     llamacpp/default      alias for llamacpp/mtp (Q4_K_M MTP, no vision)
#     llamacpp/mtp          Q4_K_M MTP + 200K (max-safe @ -ub 512; 131K @ -ub 1024 faster prefill) + q4_0 KV (fast ~60 TPS code; no vision; cliff-immune)
#     llamacpp/bounded-thinking Q4_K_M MTP + 200K + reasoning on + per-request GBNF grammar
#     llamacpp/mtp-vision   Q4_K_M MTP + 150K @ 1M-px + q4_0 KV + mmproj (multimodal; 4M-px = override, lower ctx)
#   Single-card ik_llama (IQ4_KS — ~0.5-0.8 GB leaner; best for VRAM-tight / WSL):
#     ik-llama/iq4ks-mtp         IQ4_KS MTP + 200K + q4_0 KV (own image: ikawrakow/ik-llama-cpp)
#     ik-llama/iq4ks-mtp-vision  IQ4_KS MTP + 160K @ 1M-px + q4_0 KV + mmproj (multimodal; 4M-px = override, lower ctx)
#
# Env overrides (rarely needed):
#   COMPOSE_BIN     Default: "docker compose" (set to e.g. "podman compose" if needed)
#   CLUB3090_GPU    Single-card GPU index override, e.g. "1" on a hetero rig
#   FORCE           Set to 1 to skip hardware/free-VRAM preflight
#   READY_URL       Default: http://localhost:8020/v1/models
#   READY_TIMEOUT   Default: 600 (seconds — longer for cold cudagraph capture)

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_BIN="${COMPOSE_BIN:-docker compose}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
LAUNCH_PROFILE="${LAUNCH_PROFILE:-${ROOT_DIR}/scripts/lib/profiles/launch_compat.py}"

# Load .env if present, so PORT / MODEL_DIR / etc. flow through to docker
# compose AND to the ready-URL probe below.
#
# Precedence matches docker compose (and launch.sh): a variable already set in
# the shell environment WINS over the .env file — so `export MODEL_DIR=…` is no
# longer clobbered by a stale .env entry (#425). We parse line-by-line instead
# of `source` (a) to honour that precedence per-variable and (b) to tolerate
# CRLF line endings from Windows editors (#187). Values are taken literally
# (no shell expansion), matching docker compose's own .env semantics.
if [[ -f "${ROOT_DIR}/.env" ]]; then
  while IFS= read -r _env_line || [[ -n "$_env_line" ]]; do
    _env_line="${_env_line#"${_env_line%%[![:space:]]*}"}"   # strip leading whitespace
    _env_line="${_env_line%$'\r'}"                           # strip trailing CR (CRLF .env)
    [[ -z "$_env_line" || "$_env_line" == '#'* ]] && continue
    _env_line="${_env_line#export }"
    _env_key="${_env_line%%=*}"
    [[ "$_env_key" == "$_env_line" || -z "$_env_key" ]] && continue   # no '=' on the line
    [[ -n "${!_env_key+x}" ]] && continue                    # already set in env → shell wins
    _env_val="${_env_line#*=}"
    _env_val="${_env_val#\"}"; _env_val="${_env_val%\"}"     # strip surrounding double quotes
    _env_val="${_env_val#\'}"; _env_val="${_env_val%\'}"     # strip surrounding single quotes
    export "${_env_key}=${_env_val}"
  done < "${ROOT_DIR}/.env"
  unset _env_line _env_key _env_val
fi

# Surface the resolved MODEL_DIR + its source so the precedence is unambiguous
# (the exact confusion behind #425 / #187). Unset → the compose's built-in
# default applies; preflight_compose_deps notes that case.
if [[ -n "${MODEL_DIR:-}" ]]; then
  echo "[switch] MODEL_DIR=${MODEL_DIR}"
fi

# Variant tables are DERIVED from the single source of truth
# (scripts/lib/profiles/compose_registry.py COMPOSE_REGISTRY).
declare -A VARIANT_DEFAULT_PORT=()
declare -A VARIANTS=()
declare -A VARIANT_STATUS=()
declare -A VARIANT_STATUS_NOTE=()
# shellcheck source=lib/registry-emit.sh
source "${ROOT_DIR}/scripts/lib/registry-emit.sh"
derive_switch_variant_tables "${ROOT_DIR}"

# Container name patterns we'll bring down — covers all current composes
# AND any vllm/llama-cpp container we don't formally know about (catches
# locally-built variants and one-off `docker run` instances that would
# otherwise pin GPU memory invisibly to switch.sh).
RUNNING_PATTERN="^(vllm-|llama-cpp-)"


PRIMARY_MODEL="${PRIMARY_MODEL:-qwen3.6-27b}"

switch_topology_from_gpus() {
  local selector="${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}" count=0
  if [[ -n "$selector" && "$selector" != "all" && "$selector" != "void" ]]; then
    IFS=',' read -ra _switch_gpu_tokens <<< "$selector"
    local token
    for token in "${_switch_gpu_tokens[@]}"; do
      token="${token//[[:space:]]/}"
      [[ -n "$token" ]] && count=$((count + 1))
    done
  elif command -v nvidia-smi >/dev/null 2>&1; then
    count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')"
  else
    count=1
  fi
  case "$count" in
    0|1) printf 'single' ;;
    2) printf 'dual' ;;
    4) printf 'multi4' ;;
    *) printf 'multi%s' "$count" ;;
  esac
}

resolve_default_variant() {
  local variant="$1" engine topology target
  if [[ "$variant" =~ ^([^/]+)/(single|dual|multi[0-9]+)/default$ ]]; then
    engine="${BASH_REMATCH[1]}"
    topology="${BASH_REMATCH[2]}"
  elif [[ "$variant" =~ ^([^/]+)/default$ ]]; then
    engine="${BASH_REMATCH[1]}"
    topology="$(switch_topology_from_gpus)"
  else
    printf '%s' "$variant"
    return 0
  fi
  if ! target="$(registry_default_target "$ROOT_DIR" "$PRIMARY_MODEL" "$engine" "$topology")"; then
    echo "ERROR: cannot resolve default variant '${variant}' for primary model ${PRIMARY_MODEL}." >&2
    exit 1
  fi
  printf '%s' "$target"
}

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# Map a registry status word to the marker shown in --list and to launch
# gating. `production` → unmarked; `caveats` → "(caveats)"; the (NA) set
# (experimental/preview/upstream-gated/deprecated) → "(NA: <word>)".
status_marker() {
  case "$1" in
    production|"") printf '' ;;
    caveats)       printf '(caveats)' ;;
    *)             printf '(NA: %s)' "$1" ;;
  esac
}

list_variants() {
  # Grouped by model · topology so each slug's binding is visible at a glance.
  # VARIANTS stores "<engine>|<dir>|<file>" where
  #   dir  = models/<model>/<engine>/compose      → model is dir field 2
  #   file = <topology>/<quant>/<serving>.yml      → topology/quant/serving
  # (the registry emitter splits compose_path on "/compose/", so dir stops at
  #  /compose and the topology+quant live in file). Engine is the slug prefix.
  # The trailing column is the health marker derived from the registry status.
  echo "Available variants — grouped by model · topology (right column: <quant>/<serving>.yml):"
  echo "  Health: unmarked = production · (caveats) = works w/ documented limits · (NA: …) = needs --force"
  # Counts: supported models · total variants · health split.
  local _prod=0 _cav=0 _na=0
  declare -A _seen_models=()
  for v in "${!VARIANTS[@]}"; do
    IFS='|' read -r _e _d _f <<< "${VARIANTS[$v]}"
    IFS=/ read -ra _ds <<< "$_d"; _seen_models["${_ds[1]:-?}"]=1
    case "${VARIANT_STATUS[$v]:-production}" in
      production) _prod=$((_prod + 1)) ;;
      caveats)    _cav=$((_cav + 1)) ;;
      *)          _na=$((_na + 1)) ;;
    esac
  done
  echo "  Models: ${#_seen_models[@]} · variants: ${#VARIANTS[@]} (${_prod} production · ${_cav} caveats · ${_na} NA)"
  {
    for v in "${!VARIANTS[@]}"; do
      IFS='|' read -r eng dir file <<< "${VARIANTS[$v]}"
      IFS=/ read -ra dseg <<< "$dir"    # dseg[1] = model
      IFS=/ read -ra fseg <<< "$file"   # fseg[0]=topology fseg[1]=quant fseg[2]=serving
      topo="${fseg[0]:-unknown}"
      case "$topo" in
        single) rank=1 ;;
        dual)   rank=2 ;;
        multi*) rank=3 ;;
        *)      rank=9 ;;
      esac
      marker="$(status_marker "${VARIANT_STATUS[$v]:-production}")"
      printf '%s\t%d\t%s\t%s\t%s/%s\t%s\n' \
        "${dseg[1]:-?}" "$rank" "$topo" "$v" "${fseg[1]:-?}" "${fseg[2]:-${file}}" "$marker"
    done
  } | sort -t$'\t' -k1,1 -k2,2n -k4,4 | awk -F'\t' '
    { rows[NR] = $0; cnt[$1]++ }
    END {
      for (i = 1; i <= NR; i++) {
        split(rows[i], f, "\t")
        if (f[1] != m) { printf "\n%s  (%d variants)\n", f[1], cnt[f[1]]; m = f[1]; t = "" }
        tl = (f[3] == t ? "" : f[3]); t = f[3]
        printf "  %-8s %-34s %-36s %s\n", tl, f[4], f[5], f[6]
      }
    }
  '
  echo
  echo "Switch to one:  bash scripts/switch.sh <variant>"
  echo "Or via wizard:  bash scripts/launch.sh   (or: launch.sh --variant <variant>)"
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

gpu_preflight() {
  # Catch the "switch.sh said no club-3090 container running but GPU is
  # still pinned at 22 GiB and the new container OOMs at boot" failure
  # mode. down_running() only catches docker containers we manage; this
  # function catches anything else (out-of-band vllm/ollama/training
  # processes, exited containers that didn't release GPU memory cleanly,
  # etc.). Skip with FORCE=1 if you know what you're doing.
  if [[ "${FORCE:-0}" == "1" ]]; then
    echo "[switch] FORCE=1 — skipping GPU pre-flight"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  # Free MiB per GPU. Tolerate small overhead (driver, X server) — abort
  # if any selected GPU has <80% of its total memory free.
  local mem_query
  mem_query=$(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null) || return
  local selector="${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  local selector_specific=0
  if [[ -n "$selector" && "$selector" != "all" && "$selector" != "void" ]]; then
    selector_specific=1
  fi
  local bad=0
  while IFS=',' read -r idx free total; do
    free=$(echo "$free" | tr -d ' ')
    total=$(echo "$total" | tr -d ' ')
    idx=$(echo "$idx" | tr -d ' ')
    [[ -z "$free" || -z "$total" ]] && continue
    if [[ "$selector_specific" -eq 1 && ",${selector}," != *",${idx},"* ]]; then
      continue
    fi
    # Require ≥80% free. Compose default gpu-memory-utilization is 0.92.
    local need=$(( total * 80 / 100 ))
    if [[ "$free" -lt "$need" ]]; then
      if [[ "$bad" -eq 0 ]]; then
        echo "[switch] ERROR: GPU memory pre-flight failed." >&2
        echo "[switch]        Something is still pinning GPU memory after down_running()." >&2
        echo "[switch]        Per-GPU state (free / total MiB; need ≥80% free):" >&2
      fi
      echo "[switch]          GPU $idx: $free / $total MiB free  (need ≥ $need)" >&2
      bad=1
    fi
  done <<< "$mem_query"

  if [[ "$bad" -eq 1 ]]; then
    echo "[switch]" >&2
    echo "[switch]        Holding processes:" >&2
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$apps" ]]; then
      while IFS= read -r line; do
        echo "[switch]          $line" >&2
      done <<< "$apps"
    else
      echo "[switch]          (nvidia-smi shows no compute apps — likely a zombie process or driver state)" >&2
    fi
    echo "[switch]" >&2
    echo "[switch]        Common fixes:" >&2
    echo "[switch]          docker ps -a | grep -E 'vllm|llama'       # find stopped containers" >&2
    echo "[switch]          docker rm \$(docker ps -aq --filter status=exited)" >&2
    echo "[switch]          fuser -v /dev/nvidia*                     # find host process holding the device" >&2
    echo "[switch]" >&2
    echo "[switch]        Override (skip this check):  FORCE=1 bash scripts/switch.sh ${VARIANT}" >&2
    exit 1
  fi
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
      *) echo "[switch] ERROR: unexpected engine pin export: $key" >&2; exit 2 ;;
    esac
  done <<< "$output"
  if [[ -n "${VLLM_IMAGE:-}" ]]; then
    if [[ -n "${VLLM_NIGHTLY_SHA:-}" ]]; then
      echo "[switch] vLLM image override: ${VLLM_IMAGE} (profile nightly SHA ${VLLM_NIGHTLY_SHA})"
    else
      echo "[switch] vLLM image: ${VLLM_IMAGE}"
    fi
  else
    echo "[switch] vLLM nightly SHA: ${VLLM_NIGHTLY_SHA:-unset}"
  fi
}

status_gate() {
  # Lifecycle gate (PR-A health flag). production → launch silently;
  # caveats → launch with a one-line notice; the (NA) set
  # (experimental/preview/upstream-gated/deprecated) → warn + require --force.
  local v="$1" status note
  status="${VARIANT_STATUS[$v]:-production}"
  note="${VARIANT_STATUS_NOTE[$v]:-}"
  case "$status" in
    production)
      ;;
    caveats)
      echo "[switch] NOTE: '${v}' is ⚠️ production-with-caveats.${note:+  ${note}}"
      ;;
    *)
      if [[ "${FORCE:-0}" != "1" ]]; then
        echo "[switch] ERROR: '${v}' is (NA: ${status}) — not a reliable config.${note:+  ${note}}" >&2
        echo "[switch]        It is surfaced for visibility, but won't launch without an explicit override." >&2
        echo "[switch]        Re-run with --force if you know what you're doing:" >&2
        echo "[switch]          bash scripts/switch.sh --force ${v}" >&2
        exit 1
      fi
      echo "[switch] WARNING: forcing (NA: ${status}) variant '${v}'.${note:+  ${note}}"
      ;;
  esac
}

up_variant() {
  local v="$1"
  if [[ -z "${VARIANTS[$v]:-}" ]]; then
    echo "ERROR: unknown variant '${v}'." >&2
    echo "Run: bash scripts/switch.sh --list" >&2
    exit 1
  fi
  status_gate "$v"
  IFS='|' read -r eng dir file <<< "${VARIANTS[$v]}"
  local full_dir="${ROOT_DIR}/${dir}"
  if [[ ! -f "${full_dir}/${file}" ]]; then
    echo "ERROR: compose file missing at ${full_dir}/${file}" >&2
    exit 1
  fi

  # Pre-up sanity:
  #  - genesis_pin: warn if on-disk Genesis tree differs from GENESIS_PIN in setup.sh
  #  - repo_drift: warn if local HEAD is behind origin/master
  #  - compose_deps: HARD error if compose mounts a model dir that doesn't exist on host
  #    (catches the "you didn't WITH_DFLASH_DRAFT=1 then tried dual-dflash-noviz" case;
  #     see club-3090#37 — this is the canonical fix raphael / snoby asked for)
  #  - kv_format_hint: soft warn if VRAM class needs --kv-cache-dtype override (#47)
  if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
    # shellcheck source=preflight.sh
    source "${ROOT_DIR}/scripts/preflight.sh"
    preflight_genesis_pin "${ROOT_DIR}" || true
    preflight_repo_drift "${ROOT_DIR}" || true
    preflight_compose_deps "${full_dir}/${file}" || exit 1
    if [[ "$eng" == "vllm" ]]; then
      preflight_compose_hardware "${full_dir}/${file}" "$v" "${FORCE:-0}" || exit 1
    fi
    preflight_kv_format_hint "${full_dir}/${file}" || true
  fi
  gpu_preflight

  echo "[switch] bringing up: ${v}  (${dir}/${file})"
  export_variant_engine_pin "$v"
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
  # Find the container we just brought up so we can detect crashes mid-boot
  # AND surface stage progress markers from its logs while we wait.
  local container
  container=$(docker ps --format '{{.Names}}' 2>/dev/null \
    | grep -E '^(vllm-qwen36-27b|llama-cpp-qwen36-27b|vllm-gemma-4-31b)' | head -1)

  if [[ -z "$container" ]]; then
    # Compose started but no container is up — almost always a syntax error
    # or env-var issue caught before vLLM even started.
    echo "[switch] ERROR: no container running after 'compose up' — boot failed before vLLM started." >&2
    echo "[switch]        Run 'docker compose -f <file> logs' for the compose-level error." >&2
    exit 1
  fi

  echo "[switch] waiting for ${READY_URL} (container=${container}, timeout ${READY_TIMEOUT}s)..."
  local elapsed=0 step=4 last_marker=""
  until curl -sf -o /dev/null --max-time 3 "${READY_URL}"; do
    # CRASH DETECTION: if the container died, dump tail and exit fast — don't
    # silently burn through the full timeout on a dead server.
    local state
    state=$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || echo missing)
    if [[ "$state" != "true" ]]; then
      local exit_code
      exit_code=$(docker inspect -f '{{.State.ExitCode}}' "$container" 2>/dev/null || echo "?")
      echo "[switch] ERROR: container '${container}' is no longer running (state=${state}, exit=${exit_code})." >&2
      echo "[switch]        Last 30 log lines:" >&2
      docker logs --tail 30 "$container" 2>&1 | sed 's/^/[switch]   | /' >&2
      echo "[switch]        Full logs:  docker logs ${container}" >&2
      exit 1
    fi

    sleep $step
    elapsed=$((elapsed + step))

    # PROGRESS SIGNAL: surface boot-stage markers so users see WHAT vLLM is
    # doing, not just that it's "still waiting". The grep is selective — one
    # line per phase transition, not raw log streaming.
    local marker
    marker=$(docker logs --tail 50 "$container" 2>&1 | grep -oE \
      'Genesis Results: .* applied|Resolved architecture: \w+|Loading weights|Compilation finished|Memory profiling|Capturing CUDA graphs|Application startup complete' \
      | tail -1 || true)
    if [[ -n "$marker" && "$marker" != "$last_marker" ]]; then
      echo "[switch]   ${elapsed}s — ${marker}"
      last_marker="$marker"
    elif [[ $((elapsed % 30)) -eq 0 ]]; then
      echo "[switch]   ${elapsed}s elapsed, still waiting..."
    fi

    if [[ $elapsed -ge $READY_TIMEOUT ]]; then
      echo "[switch] timeout — server not ready after ${READY_TIMEOUT}s" >&2
      echo "[switch] tail logs:  docker logs --tail 100 ${container}" >&2
      exit 1
    fi
  done
  echo "[switch] ✓ ready (${elapsed}s)"
}

# --- arg parsing ---
WAIT=1
FORCE="${FORCE:-0}"
VARIANT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --list) list_variants ;;
    --down) down_running; exit 0 ;;
    --no-wait) WAIT=0 ;;
    --force) FORCE=1 ;;
    --*) echo "Unknown flag: $1"; exit 1 ;;
    *)
      if [[ -n "$VARIANT" ]]; then
        echo "ERROR: multiple variants supplied: '${VARIANT}' and '$1'" >&2
        exit 1
      fi
      VARIANT="$1"
      ;;
  esac
  shift
done

[[ -n "$VARIANT" ]] || usage
VARIANT="$(resolve_default_variant "$VARIANT")"

resolve_ready_url "${VARIANT}"
down_running
up_variant "${VARIANT}"
[[ $WAIT -eq 1 ]] && wait_ready
echo "[switch] done. Try:  curl -s ${READY_URL%/v1/models}/v1/models | jq ."
