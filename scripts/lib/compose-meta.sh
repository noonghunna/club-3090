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

# _compose_meta_registry_file <repo_root> <slug> <fallback-abs-path>
#
# Absolute path of SLUG's registered compose (registry_lookup_compose_path)
# when the registry resolves AND the file exists on disk; else FALLBACK
# verbatim. Used by compose_hw_model_status so its probe composes cannot drift
# from the catalog.
_compose_meta_registry_file() {
  local rel=""
  if declare -F registry_lookup_compose_path >/dev/null 2>&1; then
    rel="$(registry_lookup_compose_path "$2" 2>/dev/null || true)"
  fi
  if [[ -n "$rel" && -f "$1/$rel" ]]; then
    printf '%s\n' "$1/$rel"
  else
    printf '%s\n' "$3"
  fi
}

compose_hw_model_status() {
  local repo_root="$1"
  local model="$2"
  local candidates=()
  local friendly_need=""

  # Registry-first candidate resolution, sourced LAZILY: compose-meta.sh is
  # pulled in by switch.sh / preflight.sh / setup.sh, and none of those should
  # pay the ~1s registry emit unless THIS display path actually runs. The
  # literals below stay as fallbacks when the registry can't be consulted.
  if ! declare -F registry_lookup_compose_path >/dev/null 2>&1 \
     && [[ -f "${repo_root}/scripts/lib/registry-lookup.sh" ]]; then
    # shellcheck source=registry-lookup.sh
    source "${repo_root}/scripts/lib/registry-lookup.sh"
  fi
  REGISTRY_LOOKUP_ROOT="${repo_root}"

  case "$model" in
    qwen3.6-27b)
      candidates=(
        "$(_compose_meta_registry_file "$repo_root" vllm/minimal "${repo_root}/models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml")"
      )
      friendly_need="needs 20 GB+ VRAM (24 GB recommended)"
      ;;
    gemma-4-31b)
      candidates=(
        "$(_compose_meta_registry_file "$repo_root" vllm/gemma-bf16-mtp "${repo_root}/models/gemma-4-31b/vllm/compose/dual/autoround-int4/bf16-mtp.yml")"
        "$(_compose_meta_registry_file "$repo_root" vllm/gemma-int8-mtp "${repo_root}/models/gemma-4-31b/vllm/compose/dual/autoround-int4/int8.yml")"
        "$(_compose_meta_registry_file "$repo_root" vllm/gemma-mtp-tp1 "${repo_root}/models/gemma-4-31b/vllm/compose/single/autoround-int4/fp8-mtp.yml")"
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
# ⚠️ THE FIT MODEL IS ADDITIVE, CALIBRATED FROM FIELD FAILURES — NOT a fraction.
#    fit = (FREE_i − reserve − first_card_extra[i==0] − margin) / bundle
#    An earlier ×0.55 multiplicative guard was correct on the 24 GB cards it was
#    calibrated on and left ~6 GB/card idle on 32 GB cards (community-measured:
#    worth +19% decode — #931). The overhead it absorbed is ADDITIVE (dense split
#    + drafter half + compute buffers + KV don't scale with card size), so the
#    model now subtracts it explicitly:
#      reserve           per-compose header (true per-card engine cost, DRAFTER-FREE)
#      draft_reserve     per-card VRAM the DRAFT MODEL will take (default 0 — see below)
#      first_card_extra  card 0 carries the larger drafter half + compute buffer
#                        (measured 660 MiB on 2x5090; default 768)
#
#    ⚠️ WHY draft_reserve IS ITS OWN TERM (#953). `reserve` is a STATIC per-compose
#    constant, and it used to be documented as already covering the "drafter half".
#    It cannot: the drafter loads AFTER this resolves, so FREE_i is read before the
#    draft model allocates, and a static number set before that cost was measured
#    silently understates it. On DeepSeek-Flash-Q8 (2x24 GB) the constant said
#    18000 MiB while the true per-card cost measured 19586-20028 — which is exactly
#    enough to flip fit from 0 to 1 on card 0. The sizer then granted a bundle the
#    engine could not place, and the experts silently stayed on CPU while the RAM
#    gate priced a bundle that never landed.
#    Defaults to 0, so every compose WITHOUT the header is bit-identical to before
#    and the #931 calibration points keep reproducing untouched.
#      margin            deep-prefill spike headroom. #931 brackets it: a card at
#                        556 MiB free DIED on a ~90K prefill; 896 MiB survived a
#                        full 188K NIAH ladder. Default 1024 (RESIDENCY_MARGIN_MB).
#    Sizing from FREE (not total) makes desktops / other consumers fall out
#    automatically and yields per-card asymmetric counts (7+8 on #931's rig) with
#    zero special-casing. Calibration points this must keep reproducing:
#    Q8-dual 1/card + IQ2-dual 3/card on 2x24 GB · IQ2 7 (desktop) + 8 (bare) on
#    2x32 GB (#931) · Q8-multi4 2/card on 4x24 GB (milano) · 0 on 16 GB cards.
#    The grant is floor-conservative: a borderline card may get one bundle fewer
#    than a hand-tuned pin — OT_G<i> overrides exist for exactly that.
#
# ⚠️ A LAYER'S EXPERTS MUST SIT ON THE CARD OWNING ITS DENSE TENSORS, or every token
#    pays a cross-PCIe hop. With `-sm layer` over N cards, layer i lives on card
#    floor(i*N/L) — so each card draws its resident layers from its OWN range.
#
# Emits nothing (leaving the composes' no-op defaults in place) when VRAM cannot be
# read or the header is absent. Degrading to all-experts-CPU is always safe; it is
# the config that runs anywhere.
# ---------------------------------------------------------------------------

# _offload_fit_count <free_mib> <reserve_mib> <bundle_mib> <per_card_cap>
#
# The per-card fit arithmetic, extracted so the residency INJECTOR and the RAM
# GATE (offload_residency_grant_mib → preflight_cpu_offload_ram) can never
# disagree about how many bundles a card holds. The additive model lives HERE
# and nowhere else. <free_mib> is the card's DETECTED free VRAM; <reserve_mib>
# is the caller-adjusted per-card engine cost (first-card extra already added).
# RESIDENCY_MARGIN_MB (default 1024) is the deep-prefill spike headroom — #931
# measured the bracket: 556 MiB free died at ~90K, 896 MiB survived 188K.
_offload_fit_count() {
  local free="$1" reserve="$2" bundle="$3" per_card="$4"
  local margin="${RESIDENCY_MARGIN_MB:-1024}"
  [[ "$margin" =~ ^[0-9]+$ ]] || margin=1024
  local avail=$(( free - reserve - margin ))
  (( avail < 0 )) && avail=0
  local fit=$(( avail / bundle ))
  (( fit > per_card )) && fit=$per_card
  printf '%s' "$fit"
}

# _offload_layers_for_card <card_i> <n_cards> <first_moe_layer> <moe_layers> <fit>
#
# ⭐ OUTER-EDGE SELECTION. Card 0 counts UP from the first MoE layer; the last card
# counts DOWN from the last. Middle cards work outward from their range centre.
#
# Why not "first layer of this card's nominal range": that lands exactly ON the
# -sm layer split point, which we do NOT know (the engine reports buffer sizes, not
# per-layer device assignment). Land on the wrong side and the bundle sits on a card
# that does not own the layer's dense tensors -- every token then pays a cross-PCIe
# hop, the precise pathology explicit device pinning exists to avoid. Working from
# the outer edges is correct for ANY split.
#
# Prints the pipe-separated layer list for the -ot regex. Shared with the RAM gate
# so the gate subtracts EXACTLY the bundles that will be pinned — count the entries
# here rather than trusting `fit` (out-of-range layers are dropped, never emitted).
_offload_layers_for_card() {
  local i="$1" n="$2" first="$3" layers="$4" fit="$5"
  local last=$(( first + layers - 1 ))
  local rule="" count lay
  for (( count=0; count<fit; count++ )); do
    if (( i == 0 )); then                       # first card: up from the bottom
      lay=$(( first + count ))
    elif (( i == n - 1 )); then                 # last card: down from the top
      lay=$(( last - count ))
    else                                        # middle: outward from the centre
      lay=$(( first + (2*i + 1) * layers / (2*n) + (count % 2 == 0 ? count/2 : -(count/2 + 1)) ))
    fi
    (( lay < first || lay > last )) && continue
    rule="${rule}${rule:+|}${lay}"
  done
  printf '%s' "$rule"
}

# _offload_rule_layer_count <ot_rule>
#
# Prints the number of layers named in an OT_G-style rule's `blk\.(a|b|c)\.`
# group; 0 if the rule doesn't have that shape. 0-on-unparseable is the safe
# direction: a malformed user rule most likely matches nothing at boot, so the
# RAM gate should price that card at worst case, not at what the user intended.
_offload_rule_layer_count() {
  local rule="$1" group
  case "$rule" in
    *'blk\.('*')'*) ;;
    *) printf '0'; return 0 ;;
  esac
  group="${rule#*'blk\.('}"
  group="${group%%')'*}"
  [[ -z "$group" ]] && { printf '0'; return 0; }
  local -a lays
  IFS='|' read -ra lays <<<"$group"
  printf '%s' "${#lays[@]}"
}

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
  # First MoE layer. Models with a DENSE PREFIX (Laguna=1, Inkling=2) have no experts
  # in their leading layers; a rule naming one silently matches NOTHING, so the user
  # gets fewer resident layers than we think we granted and host RAM errs UNSAFE.
  local first; first="$(compose_meta_get "$compose_file" cpu-offload-first-moe-layer || true)"
  [[ "$first" =~ ^[0-9]+$ ]] || first=0
  local extra; extra="$(compose_meta_get "$compose_file" cpu-offload-first-card-extra-mib || true)"
  [[ "$extra" =~ ^[0-9]+$ ]] || extra=768
  # Per-card DRAFT-MODEL VRAM (#953). 0 when absent => no behaviour change for any
  # compose that does not declare it. Overridable for A/B via RESIDENCY_DRAFT_MB.
  local draft; draft="$(compose_meta_get "$compose_file" cpu-offload-draft-reserve-mib || true)"
  [[ "$draft" =~ ^[0-9]+$ ]] || draft=0
  # ⚠️ plain `if`, NOT `[[ ]] && assign`: the latter returns non-zero when the
  # condition is false, and every caller runs under `set -e` — that aborts the
  # launcher mid-resolve. Same trap documented in preflight.sh.
  if [[ "${RESIDENCY_DRAFT_MB:-}" =~ ^[0-9]+$ ]]; then draft="$RESIDENCY_DRAFT_MB"; fi

  local -a frees=()
  while read -r m; do [[ "$m" =~ ^[0-9]+$ ]] && frees+=("$m"); done \
    < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  local n="${#frees[@]}"
  (( n >= 2 )) || return 0

  local per_card=$(( layers / n ))
  local i fit rule var applied="" res_i
  for (( i=0; i<n; i++ )); do
    res_i=$(( reserve + draft + (i == 0 ? extra : 0) ))
    # An explicit OT_G<i> from the user/env ALWAYS WINS and is never clobbered —
    # same contract as THREADS (resolve_offload_threads). This is the supported
    # way to pin more residency than the sizer grants (the grant is deliberately
    # floor-conservative — club-3090 #931). The RAM gate prices the user's
    # ACTUAL pin (offload_residency_grant_mib counts the rule's layers), so the
    # two stay coherent.
    var="OT_G${i}"
    if [[ -n "${!var:-}" ]]; then
      # Sanity: a pin that exceeds the card's free space beyond the engine
      # reserve is a certain boot OOM. Warn loudly, don't block: the override
      # exists to out-judge us. First community hit: 8/card of Q8's 3264 MiB
      # bundles — an IQ2-sized recipe applied to the fat quant (#931).
      local ucount upin_mib
      ucount="$(_offload_rule_layer_count "${!var}")"
      upin_mib=$(( ucount * bundle ))
      if (( upin_mib > frees[i] - res_i )); then
        echo "[residency] WARN: OT_G${i} pins ${ucount} bundles = ~$(( upin_mib / 1024 )) GiB of experts, but card ${i}" >&2
        echo "            has only ~$(( (frees[i] > res_i ? frees[i] - res_i : 0) / 1024 )) GiB free beyond the reserve (bundle=${bundle} MiB on THIS quant —" >&2
        echo "            bundle sizes differ per quant tier; a layer count sized for one tier over-pins another)." >&2
        echo "            Expect a boot OOM; reduce the pin." >&2
      fi
      applied="${applied}${applied:+ · }card${i}: USER pin, ${ucount} bundles"
      continue
    fi
    fit="$(_offload_fit_count "${frees[i]}" "$res_i" "$bundle" "$per_card")"
    if (( fit < 1 )); then                        # leave this card's no-op default
      applied="${applied}${applied:+ · }card${i}: none (0 fit)"
      continue
    fi
    rule="$(_offload_layers_for_card "$i" "$n" "$first" "$layers" "$fit")"
    if [[ -z "$rule" ]]; then
      applied="${applied}${applied:+ · }card${i}: none (0 fit)"
      continue
    fi
    export "OT_G${i}=blk\.(${rule})\.ffn_(gate|up|down)_exps\.weight=CUDA${i}"
    applied="${applied}${applied:+ · }card${i}: auto ${fit} bundles (blk ${rule})"
  done
  # Say what will actually be pinned and WHO decided it. Three community
  # debugging rounds in two days (#931 twice, milano's multi4 boot) needed
  # docker-inspect forensics to answer exactly this — one boot line ends that.
  [[ -n "$applied" ]] && echo "[residency] ${applied}" >&2
  return 0
}

# ---------------------------------------------------------------------------
# offload_residency_grant_mib <compose_file>
#
# Prints the TOTAL MiB of expert bundles resolve_offload_residency will pin onto
# THIS rig's GPUs — same headers, same detection, same calibrated fit, same layer
# selection — so preflight's RAM gate can subtract bytes that will NOT be in host
# RAM. The compose's CPU-Offload-Host-RAM-GB header MUST therefore stay the
# ALL-experts-on-CPU worst case: the gate does the subtraction itself, and a
# header with residency pre-baked would double-count it and under-gate (a 4x16 GB
# rig fits ZERO bundles and truly needs the full worst case — the exact shape the
# multi4 header briefly shipped before this function existed).
#
# Prints 0 whenever the injector would emit nothing (no VRAM readable, <2 cards,
# not a residency-capable compose): "assume nothing resident" keeps the gate at
# the worst case, which is the safe direction.
# ---------------------------------------------------------------------------
offload_residency_grant_mib() {
  local compose_file="$1"
  [[ -f "$compose_file" ]] || { printf '0'; return 0; }
  command -v nvidia-smi >/dev/null 2>&1 || { printf '0'; return 0; }

  local bundle; bundle="$(compose_meta_get "$compose_file" cpu-offload-bundle-mib || true)"
  [[ "$bundle" =~ ^[0-9]+$ ]] || { printf '0'; return 0; }
  local layers; layers="$(compose_meta_get "$compose_file" cpu-offload-moe-layers || true)"
  [[ "$layers" =~ ^[0-9]+$ ]] || { printf '0'; return 0; }
  local reserve; reserve="$(compose_meta_get "$compose_file" cpu-offload-gpu-reserve-mib || true)"
  [[ "$reserve" =~ ^[0-9]+$ ]] || reserve=18000
  local first; first="$(compose_meta_get "$compose_file" cpu-offload-first-moe-layer || true)"
  [[ "$first" =~ ^[0-9]+$ ]] || first=0
  local extra; extra="$(compose_meta_get "$compose_file" cpu-offload-first-card-extra-mib || true)"
  [[ "$extra" =~ ^[0-9]+$ ]] || extra=768
  # Per-card DRAFT-MODEL VRAM (#953). 0 when absent => no behaviour change for any
  # compose that does not declare it. Overridable for A/B via RESIDENCY_DRAFT_MB.
  local draft; draft="$(compose_meta_get "$compose_file" cpu-offload-draft-reserve-mib || true)"
  [[ "$draft" =~ ^[0-9]+$ ]] || draft=0
  # ⚠️ plain `if`, NOT `[[ ]] && assign`: the latter returns non-zero when the
  # condition is false, and every caller runs under `set -e` — that aborts the
  # launcher mid-resolve. Same trap documented in preflight.sh.
  if [[ "${RESIDENCY_DRAFT_MB:-}" =~ ^[0-9]+$ ]]; then draft="$RESIDENCY_DRAFT_MB"; fi

  local -a frees=()
  local m
  while read -r m; do [[ "$m" =~ ^[0-9]+$ ]] && frees+=("$m"); done \
    < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  local n="${#frees[@]}"
  (( n >= 2 )) || { printf '0'; return 0; }

  local per_card=$(( layers / n ))
  local i fit rule total_mib=0 var ucount res_i
  local -a lays
  for (( i=0; i<n; i++ )); do
    res_i=$(( reserve + draft + (i == 0 ? extra : 0) ))
    # A user-set OT_G<i> is what will ACTUALLY be pinned (the injector never
    # clobbers it) — price ITS layer count, not the auto fit, so the gate and
    # the boot describe the same config. First hit in the wild: a 123 GB box
    # gated on the auto grant (~12 GB) while the user was pinning 4x that
    # (#931). Unparseable user rule counts 0 = worst case, the safe direction.
    var="OT_G${i}"
    if [[ -n "${!var:-}" ]]; then
      ucount="$(_offload_rule_layer_count "${!var}")"
      total_mib=$(( total_mib + ucount * bundle ))
      continue
    fi
    fit="$(_offload_fit_count "${frees[i]}" "$res_i" "$bundle" "$per_card")"
    (( fit < 1 )) && continue
    rule="$(_offload_layers_for_card "$i" "$n" "$first" "$layers" "$fit")"
    [[ -z "$rule" ]] && continue
    IFS='|' read -ra lays <<<"$rule"
    total_mib=$(( total_mib + ${#lays[@]} * bundle ))
  done
  printf '%s' "$total_mib"
}

# ---------------------------------------------------------------------------
# resolve_offload_threads <compose_file>
#
# Exports THREADS = nproc/2 for CPU-offload composes, so `-t` tracks the RIG
# instead of a number that happened to suit the reference box.
#
# ⚠️ WHY nproc/2 AND NOT nproc: the offloaded decode path is SYNCHRONIZATION-bound,
#    not compute-bound — ~40 sequential GPU<->CPU handoffs per token with a thread
#    barrier at each. Past the knee, extra threads add barrier contention faster
#    than they add streaming, so throughput INVERTS rather than plateauing:
#    measured on 35B-A3B, -t 48 burned 94% CPU to deliver 31% of peak (-69% vs
#    -t 16). llama.cpp's own default is worse still (-31%, stack finding).
#
# ⚠️ HONEST SCOPE: on DeepSeek-V4-Flash the knee is FLAT — 16 / 24 measured
#    10.18 / 10.67, i.e. indistinguishable (tuning matrix A0/A1, 2026-08-06). So
#    this is shipped for ROBUSTNESS, not for a measured speedup on this model: a
#    hardcoded 24 oversubscribes a 4-core box and under-uses a 64-core one. Do not
#    quote a TPS gain for it.
#
# Why the launcher and not the compose: these composes carry NO entrypoint, so
# there is no shell to run `nproc` in. Adding `bash -c` would pull them into the
# `$$`-escaping regime that test-compose-nvlink-escape polices. Same division of
# labour as OT_G*: profiles hold policy, launchers resolve, preflight gates.
#
# An explicit THREADS from the user/env always wins and is never clobbered.
# ---------------------------------------------------------------------------
resolve_offload_threads() {
  local compose_file="$1"
  [[ -f "$compose_file" ]] || return 0
  [[ -n "${THREADS:-}" ]] && return 0            # user override wins
  declare -F is_cpu_offload_compose >/dev/null 2>&1 || return 0
  is_cpu_offload_compose "$compose_file" || return 0

  local n; n="$(nproc 2>/dev/null || echo 0)"
  [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )) || return 0
  local t=$(( n / 2 ))
  (( t < 1 )) && t=1                             # single-core boxes still get 1
  export THREADS="$t"
  echo "[preflight] cpu-offload: threads=${t} (nproc/2 of ${n}) — offloaded decode is sync-bound; more threads invert"
}
