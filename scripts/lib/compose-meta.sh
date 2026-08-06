#!/usr/bin/env bash
#
# Tiny parser for hardware metadata stored as compose header comments.
#
# Expected form:
#   # Requires-min-vram-gb: 24
#   # Requires-min-gpu-count: 2
#   # Tensor-parallel: 2
#   # Requires-sm: 9.0+
#
# This intentionally does not parse YAML. These fields are comments so that
# older docker compose versions and direct `docker compose -f ... up` flows keep
# working unchanged.

_compose_meta_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

_compose_meta_norm_key() {
  local key="$1"
  key="$(_compose_meta_trim "$key")"
  key="${key//_/-}"
  key="${key// /-}"
  printf '%s' "$key" | tr '[:upper:]' '[:lower:]'
}

_compose_meta_wants_key() {
  local requested="$(_compose_meta_norm_key "$1")"
  local candidate="$(_compose_meta_norm_key "$2")"

  case "$requested" in
    min-vram-gb) requested="requires-min-vram-gb" ;;
    min-gpu-count) requested="requires-min-gpu-count" ;;
    tp) requested="tensor-parallel" ;;
    sm) requested="requires-sm" ;;
  esac

  [[ "$candidate" == "$requested" ]]
}

compose_meta_get() {
  local compose_file="$1"
  local field="$2"

  [[ -f "$compose_file" ]] || return 1

  local line key value
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] || continue
    line="${line#*\#}"
    [[ "$line" == *:* ]] || continue
    key="${line%%:*}"
    value="${line#*:}"
    if _compose_meta_wants_key "$field" "$key"; then
      _compose_meta_trim "$value"
      return 0
    fi
  done < "$compose_file"

  return 1
}

compose_hw_sm_to_int() {
  local sm="$1"
  sm="${sm%%+}"
  sm="${sm//sm_/}"
  sm="${sm//SM_/}"
  sm="${sm// /}"
  [[ -z "$sm" ]] && { echo 0; return; }

  local major minor
  if [[ "$sm" == *.* ]]; then
    major="${sm%%.*}"
    minor="${sm#*.}"
  else
    major="$sm"
    minor="0"
  fi
  major="${major//[^0-9]/}"
  minor="${minor//[^0-9]/}"
  [[ -z "$major" ]] && major=0
  [[ -z "$minor" ]] && minor=0
  if [[ "${#minor}" -eq 1 ]]; then
    minor=$(( minor * 10 ))
  else
    minor="${minor:0:2}"
    [[ -z "$minor" ]] && minor=0
  fi
  echo $(( major * 100 + minor ))
}

compose_hw_vram_gb() {
  local mib="$1"
  echo $(( (mib + 1023) / 1024 ))
}

compose_hw_detect_gpus() {
  if [[ "${_COMPOSE_HW_GPU_CACHE_SET:-0}" == "1" ]]; then
    [[ -n "${_COMPOSE_HW_GPU_CACHE:-}" ]] || return 1
    printf '%s\n' "${_COMPOSE_HW_GPU_CACHE}"
    return 0
  fi

  if [[ -n "${CLUB3090_FAKE_GPUS:-}" ]]; then
    local fake parsed_fake="" f_idx f_name f_mem_mib f_sm
    IFS=',' read -ra _compose_fake_gpus <<< "${CLUB3090_FAKE_GPUS}"
    for fake in "${_compose_fake_gpus[@]}"; do
      IFS=':' read -r f_idx f_name f_mem_mib f_sm <<< "$fake"
      f_idx="$(_compose_meta_trim "${f_idx:-}")"
      f_name="$(_compose_meta_trim "${f_name:-}")"
      f_name="${f_name//_/ }"
      f_mem_mib="$(_compose_meta_trim "${f_mem_mib:-}")"
      f_sm="$(_compose_meta_trim "${f_sm:-}")"
      [[ -z "$f_idx" || -z "$f_mem_mib" ]] && continue
      parsed_fake+="${f_idx}"$'\t'"${f_name}"$'\t'"${f_mem_mib}"$'\t'"${f_sm}"$'\n'
    done
    parsed_fake="${parsed_fake%$'\n'}"
    _COMPOSE_HW_GPU_CACHE_SET=1
    _COMPOSE_HW_GPU_CACHE="$parsed_fake"
    [[ -n "$parsed_fake" ]] || return 1
    printf '%s\n' "$parsed_fake"
    return 0
  fi

  command -v nvidia-smi >/dev/null 2>&1 || return 1

  local query idx name mem_mib sm rest
  query="$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null)" || return 1
  [[ -n "$query" ]] || return 1

  local parsed=""
  while IFS=',' read -r idx name mem_mib sm rest; do
    idx="$(_compose_meta_trim "$idx")"
    name="$(_compose_meta_trim "$name")"
    mem_mib="$(_compose_meta_trim "$mem_mib")"
    sm="$(_compose_meta_trim "$sm")"
    [[ -z "$idx" || -z "$mem_mib" ]] && continue
    parsed+="${idx}"$'\t'"${name}"$'\t'"${mem_mib}"$'\t'"${sm}"$'\n'
  done <<< "$query"

  parsed="${parsed%$'\n'}"
  _COMPOSE_HW_GPU_CACHE_SET=1
  _COMPOSE_HW_GPU_CACHE="$parsed"
  [[ -n "$parsed" ]] || return 1
  printf '%s\n' "$parsed"
}

compose_hw_in_use_gpus() {
  # Returns GPU indices with non-trivial active compute work. Best-effort:
  # primary path maps compute-app UUIDs back to GPU indices; memory.used is
  # the fallback for drivers that do not expose compute app UUIDs.
  if [[ -n "${CLUB3090_FAKE_BUSY_GPUS:-}" ]]; then
    printf '%s\n' "${CLUB3090_FAKE_BUSY_GPUS//,/$'\n'}" | sed '/^$/d'
    return 0
  fi
  if [[ -n "${CLUB3090_FAKE_GPUS:-}" ]]; then
    return 0
  fi

  command -v nvidia-smi >/dev/null 2>&1 || return 0

  local uuid_query apps line uuid idx
  uuid_query="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null || true)"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$uuid_query" && -n "$apps" ]]; then
    while IFS=',' read -r uuid _pid; do
      uuid="$(_compose_meta_trim "$uuid")"
      [[ -z "$uuid" ]] && continue
      while IFS=',' read -r idx line; do
        idx="$(_compose_meta_trim "$idx")"
        line="$(_compose_meta_trim "$line")"
        if [[ "$line" == "$uuid" ]]; then
          printf '%s\n' "$idx"
        fi
      done <<< "$uuid_query"
    done <<< "$apps" | sort -u
    return 0
  fi

  local mem_used_lines used
  mem_used_lines="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  while IFS=',' read -r idx used; do
    idx="$(_compose_meta_trim "$idx")"
    used="$(_compose_meta_trim "$used")"
    [[ -z "$idx" || -z "$used" ]] && continue
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > 1024 )); then
      printf '%s\n' "$idx"
    fi
  done <<< "$mem_used_lines"
}

compose_hw_summary() {
  local gpu_lines
  gpu_lines="$(compose_hw_detect_gpus 2>/dev/null || true)"
  if [[ -z "$gpu_lines" ]]; then
    printf 'no NVIDIA GPUs detected'
    return 0
  fi

  local count=0 first_name="" first_gb="" mixed=0 idx name mem_mib sm
  while IFS=$'\t' read -r idx name mem_mib sm; do
    [[ -z "$idx" ]] && continue
    local gb
    gb="$(compose_hw_vram_gb "$mem_mib")"
    name="${name#NVIDIA }"
    name="${name#GeForce }"
    count=$((count + 1))
    if [[ -z "$first_name" ]]; then
      first_name="$name"
      first_gb="$gb"
    elif [[ "$name" != "$first_name" || "$gb" != "$first_gb" ]]; then
      mixed=1
    fi
  done <<< "$gpu_lines"

  if (( count == 0 )); then
    printf 'no NVIDIA GPUs detected'
  elif (( mixed == 0 )); then
    if (( count == 1 )); then
      printf '1× %s, %s GB' "$first_name" "$first_gb"
    else
      printf '%d× %s, %s GB each' "$count" "$first_name" "$first_gb"
    fi
  else
    local parts=()
    while IFS=$'\t' read -r idx name mem_mib sm; do
      [[ -z "$idx" ]] && continue
      name="${name#NVIDIA }"
      name="${name#GeForce }"
      parts+=("${name}, $(compose_hw_vram_gb "$mem_mib") GB")
    done <<< "$gpu_lines"
    local joined=""
    for part in "${parts[@]}"; do
      if [[ -z "$joined" ]]; then
        joined="$part"
      else
        joined="${joined} + ${part}"
      fi
    done
    printf '%s' "$joined"
  fi
}

compose_hw_requirement_text() {
  local min_vram_gb="$1"
  local min_gpu_count="$2"
  local requires_sm="${3:-}"

  local req
  if [[ "$min_gpu_count" == "1" ]]; then
    req="${min_vram_gb} GB+"
  else
    req="${min_gpu_count}× ${min_vram_gb} GB"
  fi
  if [[ -n "$requires_sm" && "$requires_sm" != "0.0" ]]; then
    req="${req}, sm_${requires_sm%%+}+"
  fi
  printf '%s' "$req"
}

compose_hw_compose_status() {
  local compose_file="$1"
  local min_vram_gb min_gpu_count requires_sm

  min_vram_gb="$(compose_meta_get "$compose_file" requires-min-vram-gb || true)"
  min_gpu_count="$(compose_meta_get "$compose_file" requires-min-gpu-count || true)"
  requires_sm="$(compose_meta_get "$compose_file" requires-sm || true)"

  if [[ -z "$min_vram_gb" || -z "$min_gpu_count" ]]; then
    printf 'unknown|metadata unavailable'
    return 2
  fi

  requires_sm="${requires_sm:-0.0}"
  local required_sm_int
  required_sm_int="$(compose_hw_sm_to_int "$requires_sm")"

  local gpu_lines
  gpu_lines="$(compose_hw_detect_gpus 2>/dev/null || true)"
  if [[ -z "$gpu_lines" ]]; then
    printf 'no|no NVIDIA GPUs detected'
    return 1
  fi

  local eligible_count=0 idx name mem_mib sm gb sm_int
  while IFS=$'\t' read -r idx name mem_mib sm; do
    [[ -z "$idx" ]] && continue
    gb="$(compose_hw_vram_gb "$mem_mib")"
    sm_int="$(compose_hw_sm_to_int "$sm")"
    if (( gb >= min_vram_gb && sm_int >= required_sm_int )); then
      eligible_count=$((eligible_count + 1))
    fi
  done <<< "$gpu_lines"

  if (( eligible_count >= min_gpu_count )); then
    printf 'ok|fits your rig'
    return 0
  fi

  printf 'no|needs %s (your rig: %s)' \
    "$(compose_hw_requirement_text "$min_vram_gb" "$min_gpu_count" "$requires_sm")" \
    "$(compose_hw_summary)"
  return 1
}

compose_hw_compose_eligible() {
  local status
  status="$(compose_hw_compose_status "$1" 2>/dev/null || true)"
  [[ "$status" == ok\|* ]]
}

compose_hw_model_status() {
  local repo_root="$1"
  local model="$2"
  local candidates=()
  local friendly_need=""

  case "$model" in
    qwen3.6-27b)
      candidates=(
        "${repo_root}/models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml"
      )
      friendly_need="needs 20 GB+ VRAM (24 GB recommended)"
      ;;
    gemma-4-31b)
      candidates=(
        "${repo_root}/models/gemma-4-31b/vllm/compose/dual/autoround-int4/bf16-mtp.yml"
        "${repo_root}/models/gemma-4-31b/vllm/compose/dual/autoround-int4/int8.yml"
        "${repo_root}/models/gemma-4-31b/vllm/compose/single/autoround-int4/fp8-mtp.yml"
      )
      friendly_need="needs 32 GB+ on single card OR 2× 24 GB"
      ;;
    *)
      printf 'no|unknown model: %s' "$model"
      return 1
      ;;
  esac

  local file status
  for file in "${candidates[@]}"; do
    [[ -f "$file" ]] || continue
    status="$(compose_hw_compose_status "$file" 2>/dev/null || true)"
    if [[ "$status" == ok\|* ]]; then
      printf 'ok|fits your rig'
      return 0
    fi
  done

  printf 'no|%s (your rig: %s)' "$friendly_need" "$(compose_hw_summary)"
  return 1
}

# ---------------------------------------------------------------------------
# resolve_offload_residency <compose_file>
#
# Sizes CPU-offload RESIDENCY from DETECTED per-device VRAM and exports OT_G0..N,
# which the offload composes expand into their leading `-ot` slots.
#
# Division of labour (matches how VLLM_IMAGE is handled): profiles hold policy,
# LAUNCHERS resolve and inject, preflight gates. Nothing is hardcoded in a compose,
# because the right count is card-dependent — a 24 GB-tuned regex either wastes VRAM
# on a 32 GB card or OOMs a smaller one.
#
# ⚠️ THE 0.55 CALIBRATION IS LOAD-BEARING. Naive free-VRAM ÷ bundle-size
#    OVERESTIMATES BY ~2x because it ignores the compute-buffer reserve (measured
#    3948+3697 MiB at -ub 4096, plus ~8-9 GB more when the DSpark drafter attaches).
#    Predicted 4 layers for Q8 on 2x24 GB; measured 2. Predicted 11 for IQ2; measured 6.
#    Shipping the naive number hands users a config that dies on first prefill.
#
# ⚠️ A LAYER'S EXPERTS MUST SIT ON THE CARD OWNING ITS DENSE TENSORS, or every token
#    pays a cross-PCIe hop. With `-sm layer` over N cards, layer i lives on card
#    floor(i*N/L) — so each card draws its resident layers from its OWN range.
#
# Emits nothing (leaving the composes' no-op defaults in place) when VRAM cannot be
# read or the header is absent. Degrading to all-experts-CPU is always safe; it is
# the config that runs anywhere.
# ---------------------------------------------------------------------------
resolve_offload_residency() {
  local compose_file="$1"
  [[ -f "$compose_file" ]] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0

  local bundle; bundle="$(compose_meta_get "$compose_file" cpu-offload-bundle-mib || true)"
  [[ "$bundle" =~ ^[0-9]+$ ]] || return 0          # not a residency-capable compose
  local layers; layers="$(compose_meta_get "$compose_file" cpu-offload-moe-layers || true)"
  [[ "$layers" =~ ^[0-9]+$ ]] || return 0
  local reserve; reserve="$(compose_meta_get "$compose_file" cpu-offload-gpu-reserve-mib || true)"
  [[ "$reserve" =~ ^[0-9]+$ ]] || reserve=18000

  local -a totals=()
  while read -r m; do [[ "$m" =~ ^[0-9]+$ ]] && totals+=("$m"); done \
    < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
  local n="${#totals[@]}"
  (( n >= 2 )) || return 0

  local per_card=$(( layers / n ))
  local i card_free fit lo hi start rule count
  for (( i=0; i<n; i++ )); do
    card_free=$(( totals[i] - reserve ))
    (( card_free < 0 )) && card_free=0
    # calibrated, not naive — see the warning above
    fit=$(( card_free * 55 / 100 / bundle ))
    (( fit > per_card )) && fit=$per_card
    (( fit < 0 )) && fit=0
    if (( fit == 0 )); then continue; fi          # leave this card's no-op default
    lo=$(( i * layers / n ))                      # this card's own layer range
    rule=""
    for (( count=0; count<fit; count++ )); do
      start=$(( lo + count ))
      rule="${rule}${rule:+|}${start}"
    done
    export "OT_G${i}=blk\.(${rule})\.ffn_(gate|up|down)_exps\.weight=CUDA${i}"
  done
  return 0
}
