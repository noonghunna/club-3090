#!/usr/bin/env bash
#
# Pre-flight checks library. Sourced by setup.sh and launch.sh — not run
# directly. Functions return 0 on success, 1 on failure (caller decides
# whether to exit). Soft warnings print and return 0.
#
# Functions:
#   preflight_docker          — docker binary + 'docker compose' subcommand work
#   preflight_gpu [min]       — nvidia-smi works, GPU detected, count >= min
#   preflight_disk <path> <gb>— free space at path covers <gb> gigabytes
#   preflight_gpu_idle        — warn if GPUs have significant VRAM already in use
#   preflight_running         — warn if a club-3090 container is already up
#   preflight_repo_drift      — warn if local HEAD is behind origin/master
#   preflight_compose_hardware— check compose VRAM/GPU-count/SM metadata
#
# Style: each function prints one or more "[preflight] ..." lines.
# Hard failures get a one-line ERROR + a "Fix:" hint.

# Avoid double-sourcing.

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"
[[ -n "${_PREFLIGHT_LOADED:-}" ]] && return 0
_PREFLIGHT_LOADED=1
_PREFLIGHT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${_PREFLIGHT_DIR}/lib/compose-meta.sh" ]]; then
  # shellcheck source=lib/compose-meta.sh
  source "${_PREFLIGHT_DIR}/lib/compose-meta.sh"
fi

preflight_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[preflight] ERROR: 'docker' not found in PATH." >&2
    echo "            Fix: install Docker — https://docs.docker.com/engine/install/" >&2
    return 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    echo "[preflight] ERROR: 'docker compose' subcommand not available." >&2
    echo "            Fix: install Docker Compose v2 plugin (legacy 'docker-compose' is unsupported)." >&2
    return 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[preflight] ERROR: 'docker info' failed — daemon not running or no permission." >&2
    echo "            Fix: 'sudo systemctl start docker'  OR  add your user to the 'docker' group" >&2
    echo "                 ('sudo usermod -aG docker \$USER' + log out/in)." >&2
    return 1
  fi
  echo "[preflight] docker:  $(docker --version | awk '{print $3}' | tr -d ',') (compose v2 ok)"
  return 0
}

preflight_gpu() {
  local min_count="${1:-1}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[preflight] ERROR: 'nvidia-smi' not found — no NVIDIA driver detected." >&2
    echo "            Fix: install NVIDIA driver R550+ (CUDA 12.4+)." >&2
    return 1
  fi
  local gpu_lines
  gpu_lines=$(nvidia-smi -L 2>/dev/null || true)
  local gpu_count
  gpu_count=$(echo "$gpu_lines" | grep -c '^GPU ' || true)
  if [[ "$gpu_count" -lt "$min_count" ]]; then
    echo "[preflight] ERROR: needs ${min_count} GPU(s), found ${gpu_count}." >&2
    if [[ "$gpu_count" -eq 0 ]]; then
      echo "            Fix: confirm 'nvidia-smi' lists your GPU(s); check driver/PCIe wiring." >&2
    else
      echo "            Fix: pick a single-card variant, or install/wire the second GPU." >&2
    fi
    return 1
  fi
  echo "[preflight] gpu:     ${gpu_count}× detected"
  echo "$gpu_lines" | sed 's/^/[preflight]            /'
  # GPU arch class (display-only, #246). The launchers inject arch-aware env
  # (KV dtype) from the hardware profiles; this banner just names the class.
  local _cap _arch_class
  _cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
  if [[ -n "$_cap" ]]; then
    case "$_cap" in
      8.6|8.7)      _arch_class="ampere" ;;
      8.9)          _arch_class="ada" ;;
      9.*)          _arch_class="hopper" ;;
      1[0-9].*)     _arch_class="blackwell" ;;
      *)            _arch_class="unknown" ;;
    esac
    if [[ "$_arch_class" == "ampere" || "$_arch_class" == "unknown" ]]; then
      echo "[preflight] arch:    ${_arch_class} (sm_${_cap}) — compose defaults apply (no arch-aware override)"
    else
      echo "[preflight] arch:    ${_arch_class} (sm_${_cap}) — arch-aware KV defaults active for pilot slugs (#246)"
    fi
  fi
  # Interconnect capability (display-only; the engagement VERDICT lives in
  # report.sh — pre-boot there is no container to audit). Silent on
  # single-GPU / stock-PCIe rigs so the line is always signal.
  if [[ "$gpu_count" -ge 2 ]]; then
    local _p2p_cap=""
    # shellcheck source=lib/p2p-state.sh
    source "$(dirname "${BASH_SOURCE[0]}")/lib/p2p-state.sh" 2>/dev/null && \
      _p2p_cap="$(p2p_host_capability "$gpu_count")"
    case "$_p2p_cap" in
      nvlink)
        if [[ "${NVLINK_MODE:-auto}" == "force_off" ]]; then
          echo "[preflight] p2p:     NVLink bridge detected but NVLINK_MODE=force_off — the bridge will sit idle this boot (~15% decode, BENCHMARKS #77)"
        else
          echo "[preflight] p2p:     NVLink bridge detected — launcher auto-engages it (NVLINK_MODE=${NVLINK_MODE:-auto})"
        fi ;;
      pcie_p2p)
        echo "[preflight] p2p:     driver reports PCIe P2P available — launcher auto-engages it (patched-driver path, NVLINK_MODE=${NVLINK_MODE:-auto}; docs/PCIE_P2P.md)" ;;
    esac
  fi
  # Cross-rig friendliness: surface a hint when 4090 / 5090 cards are
  # detected. Composes run cross-rig but per-class gotchas (ctx derate,
  # VRAM envelope, SM-gated kernels) live in the FAQ — easier to catch
  # the hint here than for a user to discover it after a confusing run.
  if echo "$gpu_lines" | grep -qE "RTX 4090"; then
    echo "[preflight] note:    4090 detected → docs/FAQ.md#can-i-use-a-4090-instead-of-a-3090 (ctx ceiling ~15–20% lower than headless 3090)"
  fi
  if echo "$gpu_lines" | grep -qE "RTX 5090"; then
    echo "[preflight] note:    5090 detected → docs/FAQ.md#can-i-use-a-5090 (32 GB envelope unlocks single-card configs)"
  fi
  # nvidia-container-toolkit check — needed for docker GPU access.
  if ! docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
    echo "[preflight] WARN:  Docker doesn't list the 'nvidia' runtime. If 'docker compose up' fails" >&2
    echo "                   with 'unknown runtime' or 'could not select device driver', install:" >&2
    echo "                   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/" >&2
  fi
  return 0
}

_preflight_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

_preflight_csv_token() {
  local value="$1"
  value="$(_preflight_trim "$value")"
  printf '%s' "$value"
}

# UUID→index normalization (#610): launch.sh --gpus exports GPU UUIDs (the
# runtime-agnostic form — CDI ignores NVIDIA_VISIBLE_DEVICES, and indices
# renumber inside classic-runtime containers). Preflight's checks are all
# host-index-based, so translate any GPU-xxxx token back to its host index
# here — ONE choke point for every selector consumer. Unknown tokens pass
# through untouched (they fail the match downstream with the existing error).
_preflight_selector_normalize() {
  local selector="$1" token out="" idx
  [[ "$selector" != *GPU-* ]] && { printf '%s' "$selector"; return 0; }
  local uuid_map
  uuid_map="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null || true)"
  IFS=',' read -ra _pf_norm_tokens <<< "$selector"
  for token in "${_pf_norm_tokens[@]}"; do
    token="$(_preflight_trim "$token")"
    if [[ "$token" == GPU-* && -n "$uuid_map" ]]; then
      idx="$(awk -F', *' -v u="$token" '$2 == u {print $1; exit}' <<< "$uuid_map")"
      [[ -n "$idx" ]] && token="$idx"
    fi
    out="${out:+${out},}${token}"
  done
  printf '%s' "$out"
}

_preflight_selector() {
  local raw=""
  if [[ -n "${CLUB3090_GPU:-}" ]]; then
    raw="${CLUB3090_GPU}"
  elif [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "all" && "${NVIDIA_VISIBLE_DEVICES}" != "void" ]]; then
    raw="${NVIDIA_VISIBLE_DEVICES}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "all" && "${CUDA_VISIBLE_DEVICES}" != "void" ]]; then
    raw="${CUDA_VISIBLE_DEVICES}"
  fi
  [[ -n "$raw" ]] && _preflight_selector_normalize "$raw"
}

_preflight_selector_is_specific() {
  local selector="${1:-}"
  [[ -n "$selector" && "$selector" != "all" && "$selector" != "void" ]]
}

_preflight_selector_allows_index() {
  local selector="$1"
  local idx="$2"
  local token

  if ! _preflight_selector_is_specific "$selector"; then
    return 0
  fi

  IFS=',' read -ra _preflight_selector_tokens <<< "$selector"
  for token in "${_preflight_selector_tokens[@]}"; do
    token="$(_preflight_trim "$token")"
    [[ "$token" == "$idx" ]] && return 0
  done
  return 1
}

_preflight_selector_first_numeric() {
  local selector="$1"
  local token

  IFS=',' read -ra _preflight_selector_tokens <<< "$selector"
  for token in "${_preflight_selector_tokens[@]}"; do
    token="$(_preflight_trim "$token")"
    if [[ "$token" =~ ^[0-9]+$ ]]; then
      printf '%s' "$token"
      return 0
    fi
  done
  return 1
}

_preflight_sm_to_int() {
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

_preflight_vram_gb() {
  local mib="$1"
  echo $(( (mib + 1023) / 1024 ))
}

_preflight_hardware_suggestions() {
  local variant="${1:-}"

  echo "[preflight]" >&2
  echo "[preflight] Suggested next steps:" >&2
  echo "[preflight]   - Pick a compose that matches the detected GPU VRAM/topology." >&2
  if [[ "$variant" == vllm/gemma-mtp-tp1 ]]; then
    echo "[preflight]   - vllm/gemma-mtp-tp1 is DEPRECATED (no fp8 KV path for Gemma 4 on Ampere sm_86)." >&2
    echo "[preflight]   - Single 24 GB card: no functional Gemma-4-31B single config (beellama retired 2026-07-27) — nearest: bash scripts/switch.sh vllm/gemma-12b-single-int8-mtp (12B), or run the 31B dual" >&2
    echo "[preflight]   - On 2x 24 GB cards, use:  bash scripts/switch.sh vllm/gemma-31b-dual" >&2
  fi
  # ⚠️ Both suggestions here were stale/dead and are repointed 2026-08-12:
  #   - beellama/dflash: the beellama engine was RETIRED 2026-07-27 (all 10 slugs
  #     deprecated) — this line had been advertising a --force-only slug as "the
  #     single-card default" ever since. Pre-existing bug, fixed here.
  #   - llamacpp/default: deprecated 2026-08-12 with every other llama.cpp +
  #     ik-llama single-card qwen slug.
  # vllm/minimal is now the only FUNCTIONAL single-card qwen path (32K, no vision).
  echo "[preflight]   - On a single 24 GB card, start with:  bash scripts/switch.sh vllm/minimal  (single-card default; 32K ctx, no vision)" >&2
  echo "[preflight]   - Explicit bypass:  bash scripts/switch.sh --force ${variant:-<variant>}" >&2
}

# preflight_compose_hardware <compose_file> [variant] [force]
#
# Reads compose header metadata and checks the target host before docker compose
# starts. This is intentionally conservative:
#   - Missing metadata warns and allows the boot.
#   - TP=1 composes auto-select the largest eligible GPU unless the user set
#     CLUB3090_GPU, CUDA_VISIBLE_DEVICES, or NVIDIA_VISIBLE_DEVICES.
#   - TP>=2 composes hard-fail only on insufficient GPU count or hard SM gates;
#     heterogeneous VRAM below the requested floor warns because advanced users
#     may be validating sub-24 GB configs with tuned memory-utilization.
preflight_compose_hardware() {
  local compose_file="$1"
  local variant="${2:-}"
  local force="${3:-${FORCE:-0}}"

  if [[ "${PREFLIGHT_NO_HARDWARE:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "$force" == "1" || "${FORCE:-0}" == "1" ]]; then
    echo "[preflight] hardware: skipped (--force/FORCE=1)"
    return 0
  fi
  if [[ ! -f "$compose_file" ]]; then
    echo "[preflight] ERROR: compose file not found: $compose_file" >&2
    return 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[preflight] WARN:  nvidia-smi not found; skipping compose hardware metadata check." >&2
    return 0
  fi
  if ! declare -F compose_meta_get >/dev/null 2>&1; then
    echo "[preflight] WARN:  compose metadata parser unavailable; skipping hardware metadata check." >&2
    return 0
  fi

  local min_vram_gb min_gpu_count tp requires_sm
  min_vram_gb="$(compose_meta_get "$compose_file" requires-min-vram-gb || true)"
  min_gpu_count="$(compose_meta_get "$compose_file" requires-min-gpu-count || true)"
  tp="$(compose_meta_get "$compose_file" tensor-parallel || true)"
  requires_sm="$(compose_meta_get "$compose_file" requires-sm || true)"
  # Requires-homogeneous-arch: true -> this compose has an ARCH-GATED path
  # (e.g. NVFP4/MXFP8 activation kernels) that torch.compile emits per rank.
  # A weight-only fallback can be valid per card and still be invalid across
  # ranks, so mixed compute capabilities must be REFUSED, not warned. (#762)
  local requires_homog_arch
  requires_homog_arch="$(compose_meta_get "$compose_file" requires-homogeneous-arch || true)"

  if [[ -z "$min_vram_gb" || -z "$min_gpu_count" || -z "$tp" ]]; then
    echo "[preflight] WARN:  compose has no hardware metadata; allowing boot: $compose_file" >&2
    return 0
  fi

  requires_sm="${requires_sm:-0.0}"
  local required_sm_int
  required_sm_int="$(_preflight_sm_to_int "$requires_sm")"

  local gpu_query
  gpu_query="$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$gpu_query" ]]; then
    echo "[preflight] WARN:  could not query GPU VRAM/SM via nvidia-smi; skipping hardware metadata check." >&2
    return 0
  fi

  local selector
  selector="$(_preflight_selector || true)"

  local total_count=0 selected_count=0 eligible_count=0 selected_below_vram=0 selected_below_sm=0
  local best_idx="" best_name="" best_mib=0 best_sm=""
  # Architecture heterogeneity across the SELECTED cards (#762). The
  # pre-existing HET check in launch.sh keys on VRAM only, so a 3090+4090
  # pair (equal 24 GB, sm_86 vs sm_89) carried no architecture signal.
  local sel_sm_list=""
  local first_idx="" first_name="" first_mib=0 first_sm=""
  local idx name mem_mib sm rest vram_gb sm_int

  while IFS=',' read -r idx name mem_mib sm rest; do
    idx="$(_preflight_csv_token "$idx")"
    name="$(_preflight_csv_token "$name")"
    mem_mib="$(_preflight_csv_token "$mem_mib")"
    sm="$(_preflight_csv_token "$sm")"
    [[ -z "$idx" || -z "$mem_mib" ]] && continue
    total_count=$(( total_count + 1 ))
    _preflight_selector_allows_index "$selector" "$idx" || continue

    selected_count=$(( selected_count + 1 ))
    if [[ -z "$first_idx" ]]; then
      first_idx="$idx"
      first_name="$name"
      first_mib="$mem_mib"
      first_sm="$sm"
    fi

    vram_gb="$(_preflight_vram_gb "$mem_mib")"
    sm_int="$(_preflight_sm_to_int "$sm")"
    case " ${sel_sm_list} " in
      *" ${sm} "*) : ;;
      *) sel_sm_list="${sel_sm_list}${sm} " ;;
    esac

    if (( vram_gb < min_vram_gb )); then
      selected_below_vram=1
    fi
    if (( sm_int < required_sm_int )); then
      selected_below_sm=1
    fi

    if (( vram_gb >= min_vram_gb && sm_int >= required_sm_int )); then
      eligible_count=$(( eligible_count + 1 ))
      if (( mem_mib > best_mib )); then
        best_idx="$idx"
        best_name="$name"
        best_mib="$mem_mib"
        best_sm="$sm"
      fi
    fi
  done <<< "$gpu_query"

  if (( total_count == 0 )); then
    echo "[preflight] ERROR: no NVIDIA GPUs detected." >&2
    _preflight_hardware_suggestions "$variant"
    return 1
  fi
  if (( selected_count == 0 )); then
    echo "[preflight] ERROR: GPU selector '${selector}' did not match any detected GPU index." >&2
    _preflight_hardware_suggestions "$variant"
    return 1
  fi

  local requires_sm_display="${requires_sm%%+}"
  local sm_label=""
  if (( required_sm_int > 0 )); then
    sm_label=", sm_${requires_sm_display}+"
  fi

  if (( tp <= 1 )); then
    if _preflight_selector_is_specific "$selector"; then
      local first_vram_gb first_sm_int
      first_vram_gb="$(_preflight_vram_gb "$first_mib")"
      first_sm_int="$(_preflight_sm_to_int "$first_sm")"
      if (( first_vram_gb < min_vram_gb || first_sm_int < required_sm_int )); then
        echo "[preflight] ERROR: ${variant:-compose} requires one GPU with >=${min_vram_gb} GB VRAM${sm_label}." >&2
        echo "[preflight]        Explicit selector '${selector}' starts with GPU ${first_idx}: ${first_name}, ${first_vram_gb} GB, sm_${first_sm}." >&2
        _preflight_hardware_suggestions "$variant"
        return 1
      fi
      export CLUB3090_GPU="${CLUB3090_GPU:-$selector}"
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$selector}"
      export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-$selector}"
      echo "[preflight] hardware: ${variant:-compose} TP=1 requires >=${min_vram_gb} GB${sm_label}; using explicit GPU ${first_idx} (${first_vram_gb} GB, sm_${first_sm})"
      return 0
    fi

    if (( eligible_count == 0 )); then
      echo "[preflight] ERROR: ${variant:-compose} requires one GPU with >=${min_vram_gb} GB VRAM${sm_label}; none found." >&2
      echo "[preflight]        Detected GPUs:" >&2
      while IFS=',' read -r idx name mem_mib sm rest; do
        idx="$(_preflight_csv_token "$idx")"
        name="$(_preflight_csv_token "$name")"
        mem_mib="$(_preflight_csv_token "$mem_mib")"
        sm="$(_preflight_csv_token "$sm")"
        [[ -z "$idx" || -z "$mem_mib" ]] && continue
        echo "[preflight]          GPU ${idx}: ${name}, $(_preflight_vram_gb "$mem_mib") GB, sm_${sm}" >&2
      done <<< "$gpu_query"
      _preflight_hardware_suggestions "$variant"
      return 1
    fi

    export CLUB3090_GPU="$best_idx"
    export CUDA_VISIBLE_DEVICES="$best_idx"
    export NVIDIA_VISIBLE_DEVICES="$best_idx"
    echo "[preflight] hardware: ${variant:-compose} TP=1 requires >=${min_vram_gb} GB${sm_label}; auto-selected GPU ${best_idx} ($(_preflight_vram_gb "$best_mib") GB, sm_${best_sm})"
    return 0
  fi

  if (( selected_count < min_gpu_count )); then
    echo "[preflight] ERROR: ${variant:-compose} requires ${min_gpu_count} visible GPU(s) for TP=${tp}; found ${selected_count}." >&2
    _preflight_hardware_suggestions "$variant"
    return 1
  fi
  if (( selected_below_sm == 1 )); then
    echo "[preflight] ERROR: ${variant:-compose} requires sm_${requires_sm_display}+ on visible GPUs." >&2
    while IFS=',' read -r idx name mem_mib sm rest; do
      idx="$(_preflight_csv_token "$idx")"
      name="$(_preflight_csv_token "$name")"
      mem_mib="$(_preflight_csv_token "$mem_mib")"
      sm="$(_preflight_csv_token "$sm")"
      _preflight_selector_allows_index "$selector" "$idx" || continue
      echo "[preflight]          GPU ${idx}: ${name}, $(_preflight_vram_gb "$mem_mib") GB, sm_${sm}" >&2
    done <<< "$gpu_query"
    _preflight_hardware_suggestions "$variant"
    return 1
  fi
  if (( selected_below_vram == 1 )); then
    echo "[preflight] WARN:  ${variant:-compose} requires >=${min_vram_gb} GB per visible GPU for TP=${tp}, but at least one selected GPU is smaller." >&2
    echo "[preflight]        Continuing because TP>=2 sub-24 GB rigs may use tuned gpu-memory-utilization/KV settings." >&2
  fi

  # --- Mixed-architecture TP guard (#762, @paulp83) -------------------------
  # Reported: NVFP4 + MXFP8 activations on a 5090 (sm_120) + 3090 Ti (sm_86)
  # pair. Weight loading succeeded via the Marlin W4A16 weight-only fallback,
  # then torch.compile AOT emitted `tl.float8e4nv` activation kernels the
  # Ampere rank cannot compile:
  #     ValueError("type fp8e4nv not supported in this architecture")
  # Worker_TP1 died; Worker_TP0 hung on the shared-memory broadcast.
  #
  # Why the SM floor did not catch it: fallback_sm replaces required_sm as the
  # hard floor (compat.py), so both cards individually cleared 7.5. But a
  # weight-only fallback is a property of the TP GROUP, not of one card -- on a
  # homogeneous sub-sm_90 pair the fallback is the only path taken and works
  # (live-validated 2x3090 sm_86, 2026-07-11); on a MIXED pair the faster rank
  # takes the native activation path and the slower one cannot follow.
  local sel_sm_count
  sel_sm_count="$(printf '%s\n' ${sel_sm_list} | grep -c . || true)"
  if (( tp > 1 && sel_sm_count > 1 )); then
    if [[ "${requires_homog_arch,,}" == "true" || "${requires_homog_arch,,}" == "yes" ]]; then
      echo "[preflight] ERROR: ${variant:-compose} requires a HOMOGENEOUS GPU architecture for TP=${tp}." >&2
      echo "[preflight]        Selected cards span compute capabilities: ${sel_sm_list% }" >&2
      while IFS=',' read -r idx name mem_mib sm rest; do
        idx="$(_preflight_csv_token "$idx")"
        name="$(_preflight_csv_token "$name")"
        mem_mib="$(_preflight_csv_token "$mem_mib")"
        sm="$(_preflight_csv_token "$sm")"
        [[ -z "$idx" ]] && continue
        _preflight_selector_allows_index "$selector" "$idx" || continue
        echo "[preflight]          GPU ${idx}: ${name}, $(_preflight_vram_gb "$mem_mib") GB, sm_${sm}" >&2
      done <<< "$gpu_query"
      echo "[preflight]        This compose has an arch-gated activation path (e.g. NVFP4/MXFP8)." >&2
      echo "[preflight]        A weight-only fallback can be valid per card and still be invalid" >&2
      echo "[preflight]        across ranks: torch.compile emits activation kernels per rank, and" >&2
      echo "[preflight]        the lower-capability rank cannot compile them (club-3090 #762)." >&2
      echo "[preflight]        Options:" >&2
      echo "[preflight]          - run on a HOMOGENEOUS pair (all same sm_), or" >&2
      echo "[preflight]          - use the single-card slug on the higher-capability GPU, or" >&2
      echo "[preflight]          - try VLLM_ENFORCE_EAGER=1 to skip torch.compile AOT (~20-30% TPS), or" >&2
      echo "[preflight]          - for genuinely heterogeneous serving see https://github.com/efschu/shvllm" >&2
      _preflight_hardware_suggestions "$variant"
      return 1
    fi
    echo "[preflight] WARN:  mixed GPU architectures selected for TP=${tp} (sm: ${sel_sm_list% })." >&2
    echo "[preflight]        Mixed-arch tensor parallelism is NOT validated on this stack: ranks do" >&2
    echo "[preflight]        not compute identically, and arch-gated kernels can fail on the lower" >&2
    echo "[preflight]        card mid-compile. Prefer a homogeneous pair (club-3090 #762)." >&2
  fi
  # -------------------------------------------------------------------------

  echo "[preflight] hardware: ${variant:-compose} TP=${tp} requires ${min_gpu_count} GPU(s), >=${min_vram_gb} GB each${sm_label}; ${selected_count} visible GPU(s) detected"
  return 0
}

# preflight_lmcache_ram <compose_file>
# Guards LMCache composes against over-allocating CPU RAM for the L1 KV cache.
# Runs REGARDLESS of --force — host-choke prevention is NOT optional: an over-sized
# --l1-size-gb (100 GB on a 94 GB rig) once exhausted RAM and forced a server reboot
# (club-3090 #133). No-op for composes without an
# `LMCache-l1-gb` metadata header, so it's safe to call on every launch.
preflight_lmcache_ram() {
  local compose_file="$1"
  [[ -f "$compose_file" ]] || return 0
  declare -F compose_meta_get >/dev/null 2>&1 || return 0

  local l1_hdr
  l1_hdr="$(compose_meta_get "$compose_file" lmcache-l1-gb || true)"
  [[ -z "$l1_hdr" ]] && return 0   # not an LMCache compose

  # Env override (LMCACHE_L1_GB) wins over the header default — the guard must track
  # what the container will actually request.
  local l1="${LMCACHE_L1_GB:-$l1_hdr}"
  if ! [[ "$l1" =~ ^[0-9]+$ ]]; then
    echo "[preflight] WARN:  LMCACHE_L1_GB='${l1}' is not an integer; skipping LMCache RAM check." >&2
    return 0
  fi

  local reserve=28   # GB headroom for the vLLM process (27B @262K TP=2) + OS
  local need=$(( l1 + reserve ))

  if [[ ! -r /proc/meminfo ]]; then
    echo "[preflight] WARN:  cannot read /proc/meminfo; skipping LMCache RAM check." >&2
    return 0
  fi
  local kb avail_gb
  kb="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
  avail_gb=$(( kb / 1024 / 1024 ))

  if (( avail_gb < need )); then
    local suggest=$(( avail_gb > reserve ? avail_gb - reserve : 8 ))
    echo "[preflight] ERROR: LMCache --l1-size-gb=${l1} needs ~${need} GB RAM (l1 ${l1} + ~${reserve} GB vLLM+OS), but only ${avail_gb} GB is available." >&2
    echo "            (Guard against the l1-too-large host-OOM: a 100 GB cache on this 94 GB rig forced a reboot.)" >&2
    echo "            Fix: lower the cache — LMCACHE_L1_GB=${suggest} bash scripts/switch.sh ... — or free RAM." >&2
    return 1
  fi

  # Soft: SHM must be >= l1 or LMCache silently falls back to slow pickle serialization.
  local shm_gb
  shm_gb="$(grep -oE 'shm_size:[[:space:]]*"?[0-9]+' "$compose_file" | grep -oE '[0-9]+' | head -1 || true)"
  if [[ -n "$shm_gb" ]] && (( shm_gb < l1 )); then
    echo "[preflight] WARN:  shm_size (${shm_gb}g) < LMCACHE_L1_GB (${l1}) — LMCache SHM will fall back to slow pickle." >&2
    echo "            Fix: raise shm_size in the compose to >= ${l1}g." >&2
  fi

  echo "[preflight] lmcache: RAM ok — l1=${l1} GB needs ~${need} GB, ${avail_gb} GB available"

  # L2 disk tier (optional) — soft-warn on low free space at the L2 host dir. L2 is OFF unless
  # LMCACHE_L2=1 or LMCACHE_L2_ADAPTER is set; it grows ~33 GB per full 262K session (~131 KB/token,
  # measured) and is unbounded, so a small disk can fill. Warn, don't fail — the user opted in.
  if [[ -n "${LMCACHE_L2_ADAPTER:-}" || "${LMCACHE_L2:-0}" == "1" ]]; then
    local l2dir="${LMCACHE_KV_DIR:-}"
    if [[ -z "$l2dir" ]]; then
      # default host dir = repo-root lmcache-kv/ (this compose sits 6 levels below the repo root)
      l2dir="$(cd "$(dirname "$compose_file")/../../../../../.." 2>/dev/null && pwd)/lmcache-kv"
    fi
    local chkdir="$l2dir"; [[ -d "$chkdir" ]] || chkdir="$(dirname "$l2dir")"
    local l2_avail_gb
    l2_avail_gb="$(df -BG "$chkdir" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')"
    if [[ -n "$l2_avail_gb" ]] && (( l2_avail_gb < 50 )); then
      echo "[preflight] WARN:  LMCache L2 enabled, only ${l2_avail_gb} GB free at ${l2dir}." >&2
      echo "            L2 is unbounded and grows ~33 GB per full 262K session — free space or point LMCACHE_L2_ADAPTER at a larger SSD." >&2
    elif [[ -n "$l2_avail_gb" ]]; then
      echo "[preflight] lmcache: L2 disk ok — ${l2_avail_gb} GB free at ${l2dir}"
    fi
  fi
  return 0
}

preflight_disk() {
  local path="$1"
  local need_gb="$2"
  # Walk up to find an existing parent (path may not exist yet).
  while [[ -n "$path" && ! -d "$path" ]]; do
    path="$(dirname "$path")"
  done
  local avail_kb
  avail_kb=$(df -Pk "$path" 2>/dev/null | awk 'NR==2 {print $4}')
  if [[ -z "$avail_kb" ]]; then
    echo "[preflight] WARN:  could not check free space at ${path}" >&2
    return 0
  fi
  local avail_gb=$(( avail_kb / 1024 / 1024 ))
  if [[ "$avail_gb" -lt "$need_gb" ]]; then
    echo "[preflight] ERROR: only ${avail_gb} GB free at ${path}, need ~${need_gb} GB." >&2
    echo "            Fix: free space, or set MODEL_DIR=<path-on-larger-volume> and re-run." >&2
    return 1
  fi
  echo "[preflight] disk:    ${avail_gb} GB free at ${path} (need ~${need_gb} GB)"
  return 0
}

preflight_gpu_idle() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local mem_used_lines
  mem_used_lines=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
  [[ -z "$mem_used_lines" ]] && return 0
  local warned=0
  while IFS=, read -r idx used; do
    used=$(echo "$used" | tr -d ' ')
    # Threshold: 1 GiB. Below that is desktop / X server / kernel modules — fine.
    if [[ "$used" -gt 1024 ]]; then
      if [[ $warned -eq 0 ]]; then
        echo "[preflight] WARN:  GPU(s) already have significant VRAM in use:" >&2
        warned=1
      fi
      echo "[preflight]            GPU $idx: ${used} MiB in use" >&2
    fi
  done <<< "$mem_used_lines"
  if [[ $warned -eq 1 ]]; then
    echo "[preflight]        Boot may OOM. Free VRAM with 'nvidia-smi' / 'docker stop ...' first." >&2
  fi
  return 0
}

# preflight_compose_gpu_fit <compose_file> <force>
#   HARD-fail (unless force=1) when the GPUs lack enough FREE VRAM for the compose's
#   gpu_memory_utilization. vLLM aborts at boot if `free < util × total`; with
#   restart:unless-stopped it then restart-loops, so `switch.sh` waits the full
#   READY_TIMEOUT (600s) for an endpoint that never comes. This converts that silent
#   timeout into an instant, actionable error — the failure @stage-chuk hit switching
#   ai-studio → gemma on a desktop rig (club-3090 #535): GNOME on the GPUs (~1.3 GiB/card)
#   leaves < the 0.95 budget free. We mirror vLLM's own check (util × total × 0.98 ≈ its
#   mem_get_info total, i.e. nvidia-smi total minus ~2% driver-reserved) and briefly retry,
#   because `docker compose down` returns before CUDA actually releases a torn-down scene.
preflight_compose_gpu_fit() {
  local compose="$1" force="${2:-0}"
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  [[ -f "$compose" ]] || return 0

  # Effective util: an env GPU_MEMORY_UTILIZATION override wins over the compose default
  # (`${GPU_MEMORY_UTILIZATION:-<X>}`), so the gate matches what vLLM will actually use.
  local util_default util
  util_default=$(grep -oE 'GPU_MEMORY_UTILIZATION:-[0-9.]+' "$compose" | head -1 | sed 's/.*-//')
  util="${GPU_MEMORY_UTILIZATION:-$util_default}"
  case "$util" in ''|*[!0-9.]*) return 0 ;; esac   # unknown / non-numeric → can't gate

  # Cards this compose uses (TP / min-gpu-count header; default 1).
  local need_cards
  need_cards=$(grep -oiE '#[[:space:]]*(Tensor-parallel|Requires-min-gpu-count):[[:space:]]*[0-9]+' "$compose" \
               | grep -oE '[0-9]+' | sort -rn | head -1)
  [[ -z "$need_cards" ]] && need_cards=1

  # Settle window (~10s): a just-`down`ed scene's VRAM lags docker's return.
  local attempt result idx fg ng fits
  for attempt in 1 2 3 4 5; do
    result=$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, -v util="$util" -v n="$need_cards" '
          { gi[NR]=$1+0; f[NR]=$3+0; d[NR]=util*$2*0.98 }
          END {
            if (NR==0) { print "-1 0 0 1"; exit }          # no cards visible → skip
            for (i=1;i<=NR;i++) ord[i]=i                    # sort card rows by free desc
            for (i=1;i<=NR;i++) for (j=i+1;j<=NR;j++) if (f[ord[j]]>f[ord[i]]) { t=ord[i]; ord[i]=ord[j]; ord[j]=t }
            k=(n<NR ? n : NR); fits=1; wf=1e18; wn=0; wi=-1  # check the k best cards; report the worst
            for (i=1;i<=k;i++){ c=ord[i]; if (f[c]<d[c]) fits=0; if (f[c]<wf){ wf=f[c]; wn=d[c]; wi=gi[c] } }
            printf "%d %.1f %.1f %d", wi, wf/1024, wn/1024, fits
          }')
    read -r idx fg ng fits <<< "$result"
    [[ "$fits" == "1" ]] && return 0
    [[ "$attempt" -lt 5 ]] && sleep 2
  done

  echo "[preflight] ERROR: GPU ${idx} has ${fg} GiB free, but this config needs ~${ng} GiB/card" >&2
  echo "[preflight]        (gpu_memory_utilization=${util}, ${need_cards} card(s) for TP). Something else is holding VRAM —" >&2
  echo "[preflight]        a desktop on the GPU, another model, or a scene that isn't fully stopped (check: docker ps / nvidia-smi)." >&2
  echo "[preflight]        Fix: free that VRAM, or lower the ceiling —" >&2
  echo "[preflight]             GPU_MEMORY_UTILIZATION=0.90 bash scripts/switch.sh <variant>" >&2
  echo "[preflight]        — then retry.  (Bypass this check with --force.)" >&2
  if [[ "$force" == "1" ]]; then
    echo "[preflight] WARN:  --force set — launching anyway; vLLM may still abort at the free-memory check." >&2
    return 0
  fi
  return 1
}

preflight_running() {
  command -v docker >/dev/null 2>&1 || return 0
  local running
  running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^(vllm-qwen36-27b|llama-cpp-qwen36-27b|ik-llama-qwen36-27b|vllm-gemma-4-31b)' || true)
  if [[ -n "$running" ]]; then
    echo "[preflight] note:    a club-3090 container is already running:"
    echo "$running" | sed 's/^/[preflight]            /'
    echo "[preflight]          'switch.sh' will bring it down before booting the new variant."
  fi
  return 0
}

# preflight_repo_drift — warn if local HEAD is behind origin/master.
# Catches the most common stale-setup pattern: user cloned weeks ago, master
# has moved (compose changes, vendored patch updates, engine pin bumps),
# they re-run their compose, hit a stale config, and file an issue we
# already solved on master.
#
# Behavior:
#   - Skips silently if not in a git repo, on a non-master branch, or if
#     PREFLIGHT_NO_FETCH=1 (offline rigs / CI / forks tracking elsewhere).
#   - Runs 'git fetch --quiet origin master' (~1-2s online).
#   - Compares local HEAD vs origin/master. Behind > 0 → WARN with the
#     count + last-fetch age + the one-line fix command.
#   - Returns 0 always; soft-warning only.
preflight_repo_drift() {
  local repo_root="${1:-${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

  # Fast bail-outs — silent.
  [[ "${PREFLIGHT_NO_FETCH:-0}" == "1" ]] && return 0
  [[ -d "${repo_root}/.git" ]] || return 0
  command -v git >/dev/null 2>&1 || return 0

  # Only check on master — on a feature branch, "behind master" is expected
  # state, not drift. Forks / contributors live there.
  local current_branch
  current_branch=$(git -C "$repo_root" rev-parse --abbrev-ref HEAD 2>/dev/null)
  [[ "$current_branch" == "master" ]] || return 0

  # Verify origin remote points at noonghunna/club-3090. If they've forked
  # and re-pointed origin elsewhere, we don't know what's "behind."
  local origin_url
  origin_url=$(git -C "$repo_root" config --get remote.origin.url 2>/dev/null)
  [[ "$origin_url" == *"noonghunna/club-3090"* ]] || return 0

  # Fetch silently. 5s timeout so we don't hang on flaky networks.
  if ! timeout 5 git -C "$repo_root" fetch --quiet origin master 2>/dev/null; then
    # Network failure / timeout — don't make this fatal or even noisy.
    return 0
  fi

  local behind
  behind=$(git -C "$repo_root" rev-list --count HEAD..origin/master 2>/dev/null)
  [[ -z "$behind" || "$behind" == "0" ]] && return 0

  # Last-fetch age. FETCH_HEAD's mtime is the cleanest proxy.
  local fetch_head="${repo_root}/.git/FETCH_HEAD"
  local age_str=""
  if [[ -f "$fetch_head" ]]; then
    local now mtime age_sec
    now=$(date +%s)
    mtime=$(stat -c %Y "$fetch_head" 2>/dev/null || stat -f %m "$fetch_head" 2>/dev/null)
    if [[ -n "$mtime" ]]; then
      age_sec=$(( now - mtime ))
      if (( age_sec < 60 )); then age_str="just now"
      elif (( age_sec < 3600 )); then age_str="${age_sec}s ago"  # < 1h, surface seconds
      elif (( age_sec < 86400 )); then age_str="$(( age_sec / 3600 ))h ago"
      else age_str="$(( age_sec / 86400 ))d ago"; fi
    fi
  fi

  echo "[preflight] WARN:  Your club-3090 checkout is ${behind} commit(s) behind origin/master." >&2
  [[ -n "$age_str" ]] && echo "[preflight]          (last origin fetch: ${age_str})" >&2
  echo "[preflight]        Easy upgrade:  bash scripts/update.sh" >&2
  echo "[preflight]        (Will refuse if you have local edits — commit or stash first.)" >&2
  echo "[preflight]        Skip this check:  PREFLIGHT_NO_FETCH=1 bash scripts/launch.sh" >&2
  return 0
}

# preflight_hf_token — verify HF_TOKEN is set; warn if not.
#
# Soft warning (returns 0) — Qwen3.6-27B is T&C-gated on HuggingFace, so
# missing HF_TOKEN will cause `hf download` to fail with a generic error
# later. Surfacing the issue early saves a round-trip.
#
# Skip via: PREFLIGHT_NO_HF_TOKEN=1
preflight_hf_token() {
  if [[ "${PREFLIGHT_NO_HF_TOKEN:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[preflight] WARNING: HF_TOKEN is not set in the environment." >&2
    echo "[preflight]          Qwen3.6-27B is T&C-gated on HuggingFace; downloads will fail without a token." >&2
    echo "[preflight]          Fix: visit https://huggingface.co/settings/tokens, create a read token," >&2
    echo "[preflight]               accept the model T&C at https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct" >&2
    echo "[preflight]               (and any other Qwen3-Next variant you'll use)," >&2
    echo "[preflight]               then export HF_TOKEN=hf_... in your shell or .env file." >&2
    return 0
  fi
  # Sanity check token format — HF tokens start with hf_ and are 30+ chars
  if [[ ! "${HF_TOKEN}" =~ ^hf_ ]] || [[ "${#HF_TOKEN}" -lt 30 ]]; then
    echo "[preflight] WARNING: HF_TOKEN doesn't look like a valid HF token (expected 'hf_...' format, 30+ chars)." >&2
    echo "[preflight]          If downloads fail later, regenerate at https://huggingface.co/settings/tokens" >&2
  fi
  return 0
}

# preflight_compose_deps <compose_file> — verify any model directories the compose
# expects to mount actually exist on the host. Catches the "you set up the repo
# but didn't WITH_DFLASH_DRAFT=1, then tried to launch dual-dflash-noviz" failure
# mode. See club-3090#37 for the canonical case.
#
# Hard error (returns 1) — refuses to proceed if a required model dir is missing.
# Skip via: PREFLIGHT_NO_COMPOSE_DEPS=1
#
# Also runs the #1042 weight-shard preflight: for every model directory the
# compose mounts that DOES exist, verify the weight files its
# model.safetensors.index.json references (or, failing that, its numbered
# -000NN-of-000NN GGUF parts) are actually on disk. Catches the interrupted
# re-fetch / partial rsync / manual `hf download` divergence that otherwise
# surfaces as a 53 KB vLLM traceback naming the absent shard near the bottom
# (club-3090#1042). Existence + count only — never hashes (setup.sh owns
# integrity). Bypassed by FORCE=1 (--force) or PREFLIGHT_NO_SHARD_CHECK=1.
_preflight_compose_model_dir() {
  local compose_file="$1"
  local model_dir

  if [[ -n "${MODEL_DIR:-}" ]]; then
    model_dir="${MODEL_DIR}"
  else
    local root_dir="${ROOT_DIR:-}"
    if [[ -z "$root_dir" ]]; then
      root_dir="$(cd -- "${_PREFLIGHT_DIR}/.." && pwd)"
    fi
    model_dir="${root_dir}/models-cache"
    echo "[preflight] MODEL_DIR not set — defaulting to ${model_dir}" >&2
  fi

  # Resolve relative paths against the compose location. Do not require the
  # directory to already exist; this function is often called before download.
  if [[ "$model_dir" == ../* ]] || [[ "$model_dir" == ./* ]]; then
    local compose_dir
    compose_dir="$(cd -- "$(dirname -- "$compose_file")" && pwd)"
    model_dir="${compose_dir}/${model_dir}"
  fi
  printf '%s' "$model_dir"
}

# _preflight_compose_flag_paths <flag-alternation> <compose-file>... — emit every
# `/models/...` value passed to one of the given flags, in BOTH compose spellings.
#
# ⚠️ Compose command args come in two forms, and matching only one is a SILENT
# hole — the guard returns green for a compose it never actually read:
#
#     command: -m /models/x.gguf          # inline: flag + value on one line
#     command:                            # list: flag and value are SEPARATE items
#       - '-m'
#       - '/models/x.gguf'
#
# Every extractor here was inline-only, so all three DeepSeek-Flash composes
# (list form) were invisible: the path came back empty, the qwen fallback took
# over, and `switch.sh` validated *Qwen's* files while launching DeepSeek. That
# is how paulp83 got three container restarts and a cryptic in-container
# "failed to open GGUF file" instead of one clear "weights missing" (#913).
# _preflight_escapes_mount <mount_root> <file> — true when <file> is reachable on
# the HOST only by following a symlink out of <mount_root>.
#
# ⚠️ THE HOST AND THE CONTAINER DISAGREE, AND THE HOST IS THE OPTIMISTIC ONE.
# `[[ -f ]]` follows symlinks, so a model dir symlinked onto another disk reads as
# PRESENT here — while a Docker bind mount does NOT follow links that leave the
# mounted tree, so the container gets "No such file or directory" and the server
# exits. Preflight then looks like it passed and the failure surfaces as a cryptic
# in-container error, which is exactly the confusion this guard exists to prevent.
#
# Not exotic: these weights are 85-151 GB, so the users most likely to run them are
# the ones whose model disk filled up and who symlinked a model dir elsewhere. This
# rig does precisely that, and it is how the trap was found (2026-08-07).
_preflight_escapes_mount() {
  local root="$1" file="$2" rroot rfile
  command -v realpath >/dev/null 2>&1 || return 1     # cannot tell -> do not cry wolf
  rroot="$(realpath -m "$root" 2>/dev/null)"  || return 1
  rfile="$(realpath -m "$file" 2>/dev/null)"  || return 1
  [[ -n "$rroot" && -n "$rfile" ]] || return 1
  [[ "$rfile" == "$rroot"/* ]] && return 1            # stays inside the mount: fine
  return 0                                            # escapes: container will not see it
}

_preflight_compose_flag_paths() {
  local flags="$1"
  shift
  [[ $# -gt 0 ]] || return 0

  # (a) inline — flag and value on the same line.
  grep -hoE -- "(^|[[:space:]])(${flags})[[:space:]]+/models/[^[:space:]]+" "$@" 2>/dev/null \
    | awk '{print $NF}' || true

  # (b) YAML list — the value is the NEXT list item after the flag item.
  # `q` carries the single quote so the program stays single-quotable in bash.
  awk -v flagre="^(${flags})$" -v q="'" '
    FNR == 1 { pending = 0 }                       # never span two files
    {
      line = $0
      sub(/^[[:space:]]*-[[:space:]]*/, "", line)  # strip the list dash
      gsub("^[\"" q "]|[\"" q "],?$", "", line)    # strip quotes / trailing comma
      if (pending && line ~ /^\/models\//) { print line; pending = 0; next }
      pending = (line ~ flagre) ? 1 : 0
    }' "$@" 2>/dev/null || true
}

_preflight_compose_path_default() {
  local value="$1"
  value="$(_preflight_trim "$value")"
  value="${value#\"}"
  value="${value%\"}"
  value="${value#\'}"
  value="${value%\'}"
  value="${value%,}"

  # Compose files commonly use ${VAR:-default/path}. Presence checks should use
  # the path the compose will use by default; explicit env overrides are handled
  # by callers for user-facing knobs such as GGUF_FILE.
  value="$(printf '%s' "$value" | sed -E 's#\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]*)\}#\1#g')"
  value="$(printf '%s' "$value" | sed -E 's#\$\{MODEL_DIR[^}]*\}/?##g')"
  value="${value#/models/}"
  value="${value#/root/.cache/huggingface/}"
  # Strip a trailing `}` left when a path is the DEFAULT inside an outer
  # ${VAR:-/root/.cache/huggingface/<path>} — the path grep anchors mid-expansion
  # so it captures `<path>}`. A real model subdir never ends in `}`.
  value="${value%\}}"
  printf '%s' "$value"
}

_preflight_compose_vllm_subdir() {
  local value
  value="$(_preflight_compose_path_default "$1")"
  value="${value%%/*}"
  printf '%s' "$value"
}

_preflight_missing_rel() {
  local model_dir="$1"
  local item="$2"
  local item_path="${item%% (*}"
  local rel="${item_path#${model_dir}/}"
  printf '%s' "$rel"
}

_preflight_list_has() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    [[ "$value" == "$needle" ]] && return 0
  done
  return 1
}

_preflight_hf_cli_available() {
  command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1
}

_preflight_setup_root() {
  if [[ -n "${ROOT_DIR:-}" ]]; then
    printf '%s' "$ROOT_DIR"
  else
    cd -- "${_PREFLIGHT_DIR}/.." && pwd
  fi
}

_preflight_weights_reader() {
  local root_dir
  root_dir="$(_preflight_setup_root)"
  printf '%s' "${root_dir}/scripts/lib/profiles/weights.py"
}

_preflight_weight_recipe_for_path() {
  local rel="$1"
  local reader env_lines
  reader="$(_preflight_weights_reader)"
  command -v python3 >/dev/null 2>&1 || return 1
  [[ -f "$reader" ]] || return 1
  env_lines="$(python3 "$reader" lookup "$rel" 2>/dev/null)" || return 1
  eval "$env_lines"
}

_preflight_weight_recipe_for_key() {
  local key="$1"
  local reader env_lines
  reader="$(_preflight_weights_reader)"
  command -v python3 >/dev/null 2>&1 || return 1
  [[ -f "$reader" ]] || return 1
  env_lines="$(python3 "$reader" entry "$key" 2>/dev/null)" || return 1
  eval "$env_lines"
}

_preflight_weight_hf_command() {
  local model_dir_expr="${1:-\$MODEL_DIR}"
  [[ -n "${WEIGHT_REPO:-}" ]] || return 1
  # Mirror an optional revision pin (#319) so the manual fallback fetches the
  # same bytes setup.sh would. Unset -> no flag (track HEAD).
  local rev=""
  [[ -n "${WEIGHT_REVISION:-}" ]] && rev=" --revision ${WEIGHT_REVISION}"
  if [[ -n "${WEIGHT_FILES:-}" ]]; then
    printf 'hf download %s %s%s --local-dir %s/%s' \
      "$WEIGHT_REPO" "$WEIGHT_FILES" "$rev" "$model_dir_expr" "$WEIGHT_SUBDIR"
  else
    printf 'hf download %s%s --local-dir %s/%s' \
      "$WEIGHT_REPO" "$rev" "$model_dir_expr" "$WEIGHT_SUBDIR"
  fi
}

_preflight_weight_setup_command() {
  [[ -n "${WEIGHT_SETUP_MODEL:-}" ]] || return 1
  if [[ -n "${WEIGHT_SETUP_ENV:-}" ]]; then
    printf '%s bash scripts/setup.sh %s' "$WEIGHT_SETUP_ENV" "$WEIGHT_SETUP_MODEL"
  else
    printf 'bash scripts/setup.sh %s' "$WEIGHT_SETUP_MODEL"
  fi
}

_preflight_weight_hint_keys() {
  local model_dir="$1"
  shift
  local item rel key
  local keys=()

  for item in "$@"; do
    rel="$(_preflight_missing_rel "$model_dir" "$item")"
    if _preflight_weight_recipe_for_path "$rel"; then
      key="$WEIGHT_KEY"
      if ! _preflight_list_has "$key" "${keys[@]}"; then
        keys+=("$key")
        printf '%s\n' "$key"
      fi
    fi
  done
}

_preflight_print_weight_hints() {
  local model_dir="$1"
  shift
  local key any_hint=0 setup_cmd hf_cmd

  echo "[preflight]   If weights are already elsewhere, export MODEL_DIR=/path/to/models and retry." >&2
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    _preflight_weight_recipe_for_key "$key" || continue
    any_hint=1
    echo "[preflight]" >&2
    echo "[preflight]   ${WEIGHT_LABEL:-$key}:" >&2
    if hf_cmd="$(_preflight_weight_hf_command '$MODEL_DIR' 2>/dev/null)"; then
      echo "[preflight]     ${hf_cmd}" >&2
    fi
    if setup_cmd="$(_preflight_weight_setup_command 2>/dev/null)"; then
      echo "[preflight]     or: MODEL_DIR=${model_dir} ${setup_cmd}" >&2
    fi
    if [[ -n "${WEIGHT_MANUAL_NOTE:-}" ]]; then
        # Trim for terminal display. These notes are maintainer-facing and long by
        # design (median 554 chars, worst 2,273) — dumping one whole buries the
        # "Fix:" lines directly above it, which are what the user needs. Same class
        # as the switch.sh status_note dump fixed in #1041; reported on Discord
        # 2026-08-17, where a missing-weights preflight printed ~2.3 KB of note.
        _wmn="${WEIGHT_MANUAL_NOTE}"
        if (( ${#_wmn} > 220 )); then
          _wmn="${_wmn:0:220}"
          _wmn="${_wmn% *}… [truncated — full note: scripts/lib/profiles/models/*.yml]"
        fi
        echo "[preflight]     note: ${_wmn}" >&2
        unset _wmn
    fi
  done < <(_preflight_weight_hint_keys "$model_dir" "$@")

  if [[ "$any_hint" != "1" ]]; then
    echo "[preflight]   Check the compose header for its model-specific hf download command." >&2
  fi
}

_preflight_offer_fetch_missing() {
  local compose_file="$1"
  local model_dir="$2"
  shift 2

  [[ "${PREFLIGHT_NO_FETCH_PROMPT:-0}" != "1" ]] || return 1
  [[ -t 0 && -t 1 ]] || return 1
  _preflight_hf_cli_available || return 1

  local key setup_cmd answer root_dir
  local keys=()
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    _preflight_weight_recipe_for_key "$key" || continue
    [[ -n "${WEIGHT_SETUP_MODEL:-}" ]] || continue
    [[ -n "${WEIGHT_REPO:-}" ]] || continue
    if ! _preflight_list_has "$key" "${keys[@]}"; then
      keys+=("$key")
    fi
  done < <(_preflight_weight_hint_keys "$model_dir" "$@")

  [[ ${#keys[@]} -gt 0 ]] || return 1

  echo "[preflight]" >&2
  read -r -p "[preflight] Fetch missing weights now with scripts/setup.sh? [y/N]: " answer
  [[ "$answer" =~ ^[Yy]$ ]] || return 1

  root_dir="$(_preflight_setup_root)"
  for key in "${keys[@]}"; do
    _preflight_weight_recipe_for_key "$key" || continue
    local env_args=("MODEL_DIR=${model_dir}" "WEIGHT_KEY=${key}")
    echo "[preflight] fetching ${WEIGHT_LABEL:-$key} ..." >&2
    env "${env_args[@]}" bash "${root_dir}/scripts/setup.sh" "${WEIGHT_SETUP_MODEL}"
  done

  PREFLIGHT_NO_FETCH_PROMPT=1 preflight_compose_deps "$compose_file"
}

# _preflight_shard_scan <dir> — #1042 weight-shard presence scan.
#
# Prints one "<kind>:<filename>" line per ABSENT weight file, kind being:
#   safetensors — named in <dir>/model.safetensors.index.json's weight_map
#   gpart       — one numbered part of a -000NN-of-000NN GGUF part-set
# Empty output == nothing missing. Existence + count ONLY, never hashes:
# setup.sh owns integrity (sha256 per fetch), and re-hashing ~30 GB of
# weights on every launch is exactly what #1042 rules out.
#
# Skips cleanly — empty output, exit 0 — when there is no index AND no
# GGUF part-pattern: not every checkout has either (single-file safetensors
# or single-file GGUF), and an absent index is explicitly NOT an error.
_preflight_shard_scan() {
  local dir="$1"
  command -v python3 >/dev/null 2>&1 || return 0
  [[ -d "$dir" ]] || return 0
  python3 - "$dir" <<'PY'
import json, os, re, sys

d = sys.argv[1]
entries = set(os.listdir(d))

# Case 1 — sharded safetensors: diff weight_map values against disk.
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.isfile(idx):
    weight_map = None
    try:
        with open(idx, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            weight_map = data.get("weight_map")
    except (OSError, ValueError):
        weight_map = None
    if isinstance(weight_map, dict):
        for name in sorted({str(v) for v in weight_map.values()}):
            if name not in entries:
                print(f"safetensors:{name}")
    else:
        # A present-but-unparseable index usually means the download that
        # wrote it was interrupted — but this check owns ABSENT shards, not
        # index health; warn softly and let the engine report its own error.
        print(f"[preflight] WARN: unparseable {idx} — skipping the shard check", file=sys.stderr)
    raise SystemExit(0)

# Case 2 — no index, but numbered GGUF parts (-00001-of-000NN): every
# prefix-group must have ALL parts 1..NN on disk.
part = re.compile(r"^(?P<prefix>.+)-(?P<num>\d+)-of-(?P<total>\d+)\.gguf$")
groups = {}
for entry in sorted(entries):
    m = part.match(entry)
    if m:
        groups[m.group("prefix")] = (
            int(m.group("total")),
            max(len(m.group("num")), len(m.group("total"))),
        )
for prefix, (total, width) in sorted(groups.items()):
    for n in range(1, total + 1):
        name = f"{prefix}-{n:0{width}d}-of-{total:0{width}d}.gguf"
        if name not in entries:
            print(f"gpart:{name}")
PY
}

preflight_compose_deps() {
  local compose_file="$1"
  if [[ "${PREFLIGHT_NO_COMPOSE_DEPS:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$compose_file" ]]; then
    echo "[preflight] ERROR: compose file not found: $compose_file" >&2
    return 1
  fi

  local model_dir
  model_dir="$(_preflight_compose_model_dir "$compose_file")"

  local compose_files=("$compose_file")
  local compose_dir extends_file
  compose_dir="$(cd -- "$(dirname -- "$compose_file")" && pwd)"
  while IFS= read -r extends_file; do
    extends_file="$(_preflight_trim "$extends_file")"
    extends_file="${extends_file#\"}"
    extends_file="${extends_file%\"}"
    extends_file="${extends_file#'}"
    extends_file="${extends_file%'}"
    [[ -n "$extends_file" ]] || continue
    [[ "$extends_file" == /* ]] || extends_file="${compose_dir}/${extends_file}"
    [[ -f "$extends_file" ]] && compose_files+=("$extends_file")
  done < <(grep -hE '^[[:space:]]*file:[[:space:]]*[^#[:space:]]+' "$compose_file" \
    | sed -E 's/^[[:space:]]*file:[[:space:]]*//' || true)

  local missing=()
  local escaped=()
  # Model dirs that exist and are worth a #1042 shard scan (deduped below):
  # HF subdirs with a config.json, GGUF/drafter/mmproj parent dirs, SGLang
  # ${MODEL_DIR}/... volume dirs.
  local shard_dirs=()
  local shard_note=0

  # Engine detection: llama.cpp composes mount ${MODEL_DIR}:/models and pass
  # `-m /models/<path>` or `--model /models/<path>`; vLLM composes mount
  # ${MODEL_DIR}:/root/.cache/huggingface and pass
  # `/root/.cache/huggingface/<subdir>`.
  local is_llamacpp=0
  # beellama.cpp (ghcr.io/{anbeeld/beellama.cpp,noonghunna/beellama-cpp}) is a
  # llama.cpp-family server: it mounts ${MODEL_DIR}:/models and passes
  # `-m /models/<path>` (+ `--spec-draft-model /models/<path>` for DFlash/MTP),
  # so it belongs on the GGUF presence path, NOT the vLLM HF-cache path.
  if grep -qhE 'image:.*(ggml-org/llama\.cpp|ikawrakow/ik-llama|beellama)' "${compose_files[@]}"; then
    is_llamacpp=1
  fi

  if [[ $is_llamacpp -eq 1 ]]; then
    local gguf_paths=()
    local draft_paths=()
    local mmproj_paths=()
    local token path

    # Target weights: -m / --model
    while IFS= read -r token; do
      path="$(_preflight_compose_path_default "$token")"
      [[ -n "$path" ]] && gguf_paths+=("$path")
    done < <(_preflight_compose_flag_paths '-m|--model' "${compose_files[@]}")

    # Speculative drafter: beellama --spec-draft-model, llama.cpp -md/--model-draft.
    # A missing drafter GGUF otherwise surfaces only as a cryptic in-container
    # "failed to open GGUF file" crash (see #288 beellama onboarding reports).
    while IFS= read -r token; do
      path="$(_preflight_compose_path_default "$token")"
      [[ -n "$path" ]] && draft_paths+=("$path")
    done < <(_preflight_compose_flag_paths '--spec-draft-model|--model-draft|-md' "${compose_files[@]}")

    while IFS= read -r token; do
      path="$(_preflight_compose_path_default "$token")"
      [[ -n "$path" ]] && mmproj_paths+=("$path")
    done < <(_preflight_compose_flag_paths '--mmproj' "${compose_files[@]}")

    # Env overrides mirror the compose knobs (GGUF_FILE / DRAFT_FILE / MMPROJ_FILE),
    # each replacing only its own path class.
    if [[ -n "${GGUF_FILE:-}" ]]; then
      gguf_paths=("$GGUF_FILE")
    fi
    if [[ -n "${DRAFT_FILE:-}" && ${#draft_paths[@]} -gt 0 ]]; then
      draft_paths=("$DRAFT_FILE")
    fi
    if [[ -n "${MMPROJ_FILE:-}" && ${#mmproj_paths[@]} -gt 0 ]]; then
      mmproj_paths=("$MMPROJ_FILE")
    fi

    for path in "${gguf_paths[@]}"; do
      if [[ ! -f "${model_dir}/${path}" ]]; then
        missing+=("${model_dir}/${path} (llama.cpp GGUF weights)")
      else
        _preflight_escapes_mount "$model_dir" "${model_dir}/${path}" && escaped+=("${model_dir}/${path}")
        shard_dirs+=("$(dirname -- "${model_dir}/${path}")")
      fi
    done
    for path in "${draft_paths[@]}"; do
      if [[ ! -f "${model_dir}/${path}" ]]; then
        missing+=("${model_dir}/${path} (speculative drafter GGUF)")
      else
        _preflight_escapes_mount "$model_dir" "${model_dir}/${path}" && escaped+=("${model_dir}/${path}")
        shard_dirs+=("$(dirname -- "${model_dir}/${path}")")
      fi
    done
    for path in "${mmproj_paths[@]}"; do
      if [[ ! -f "${model_dir}/${path}" ]]; then
        missing+=("${model_dir}/${path} (vision projector)")
      else
        _preflight_escapes_mount "$model_dir" "${model_dir}/${path}" && escaped+=("${model_dir}/${path}")
        shard_dirs+=("$(dirname -- "${model_dir}/${path}")")
      fi
    done
  else
    local seen_subdirs=" "
    local subdir

    # vLLM path — collect every in-container HF model path the compose names:
    # main `--model` entries and JSON `--speculative-config` draft models.
    while IFS= read -r token; do
      subdir="$(_preflight_compose_vllm_subdir "$token")"
      [[ -n "$subdir" ]] || continue
      if [[ "$seen_subdirs" != *" ${subdir} "* ]]; then
        seen_subdirs+="${subdir} "
        if [[ ! -f "${model_dir}/${subdir}/config.json" ]]; then
          missing+=("${model_dir}/${subdir}/config.json (HF model)")
        else
          # Present enough to scan — the #1042 shard check runs on it below.
          shard_dirs+=("${model_dir}/${subdir}")
        fi
      fi
    # Char-class must NOT exclude `:` or `}` — model paths can be
    # `/root/.cache/huggingface/${MODEL_SUBDIR:-default}` (vLLM) and excluding
    # those truncated the token to `${MODEL_SUBDIR` before _preflight_compose_path_default
    # could resolve the `:-default`, causing a false "missing" (the gemma-4-12b
    # MODEL_SUBDIR/SPEC_MODEL_SUBDIR composes). Stop only at real delimiters
    # (quote / whitespace / comma); the `${VAR:-default}` resolver runs downstream.
    done < <(grep -hv '^[[:space:]]*#' "${compose_files[@]}" 2>/dev/null | grep -oE '/root/\.cache/huggingface/[^"'\''[:space:],]+' || true)

    # Experimental SGLang composes mount individual MODEL_DIR subdirectories to
    # /models/target and /models/drafter instead of using the HF cache mount.
    while IFS= read -r token; do
      path="$(_preflight_compose_path_default "$token")"
      path="${path%%:*}"
      [[ -n "$path" ]] || continue
      if [[ ! -e "${model_dir}/${path}" ]]; then
        missing+=("${model_dir}/${path} (MODEL_DIR volume path)")
      elif [[ -d "${model_dir}/${path}" ]]; then
        shard_dirs+=("${model_dir}/${path}")
      fi
    done < <(grep -hoE '\$\{MODEL_DIR[^}]*\}/[^"[:space:]]+' "${compose_files[@]}" || true)
  fi

  # Present on the host, unreachable from the container. Reported SEPARATELY from
  # `missing`, because the fix is completely different: the bytes are already
  # downloaded and telling the user to fetch them again would be wrong.
  if [[ ${#escaped[@]} -gt 0 ]]; then
    echo "[preflight] ERROR: compose '$compose_file' points at files that exist on the host" >&2
    echo "            but resolve OUTSIDE \$MODEL_DIR through a symlink:" >&2
    local _e
    for _e in "${escaped[@]}"; do
      echo "[preflight]   ${_e}" >&2
      echo "[preflight]     -> $(realpath -m "$_e" 2>/dev/null)" >&2
    done
    echo "[preflight]" >&2
    echo "[preflight] A Docker bind mount does NOT follow symlinks that leave the mounted" >&2
    echo "[preflight] tree, so the container sees a dangling link and the server exits with" >&2
    echo "[preflight] \"failed to open GGUF file (No such file or directory)\". The weights" >&2
    echo "[preflight] are fine — do NOT re-download them." >&2
    echo "[preflight]" >&2
    echo "[preflight] Fix (either one):" >&2
    echo "[preflight]   MODEL_DIR=<the directory the symlink points into> bash scripts/switch.sh ..." >&2
    echo "[preflight]   or replace the symlink with a real directory / bind mount under \$MODEL_DIR" >&2
    echo "[preflight] Skip this check:  PREFLIGHT_NO_COMPOSE_DEPS=1 bash scripts/switch.sh ..." >&2
    return 1
  fi

  # ── #1042 weight-shard preflight ─────────────────────────────────────────
  # The dirs exist; now verify the weight FILES they need are on disk.
  # Behind the same guard as the other preflight checks: --force (FORCE=1,
  # switch.sh) bypasses it deliberately, as does PREFLIGHT_NO_SHARD_CHECK=1.
  if [[ "${FORCE:-0}" != "1" && "${PREFLIGHT_NO_SHARD_CHECK:-0}" != "1" ]]; then
    local _sd _shard _seen_shard_dirs=" "
    for _sd in ${shard_dirs[@]+"${shard_dirs[@]}"}; do
      [[ "$_seen_shard_dirs" != *" ${_sd} "* ]] || continue
      _seen_shard_dirs+="$_sd "
      while IFS= read -r _shard; do
        [[ -n "$_shard" ]] || continue
        case "$_shard" in
          safetensors:*)
            missing+=("${_sd}/${_shard#safetensors:} (referenced by model.safetensors.index.json)")
            shard_note=1
            ;;
          gpart:*)
            missing+=("${_sd}/${_shard#gpart:} (missing numbered GGUF part)")
            shard_note=1
            ;;
        esac
      done < <(_preflight_shard_scan "$_sd")
    done
  fi

  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi

  echo "[preflight] ERROR: compose '$compose_file' expects model files that aren't on host." >&2
  for item in "${missing[@]}"; do
    echo "[preflight]   missing: ${item}" >&2
  done
  echo "[preflight]" >&2
  if [[ "$shard_note" == "1" ]]; then
    echo "[preflight] The index/part-referenced entries above are absent from disk — the download is incomplete." >&2
    echo "[preflight] (Existence + count are checked, never hashes.) The re-fetch is resumable:" >&2
  fi

  echo "[preflight] Fix:" >&2
  _preflight_print_weight_hints "$model_dir" "${missing[@]}"
  if _preflight_offer_fetch_missing "$compose_file" "$model_dir" "${missing[@]}"; then
    return 0
  fi
  echo "[preflight] Skip this check:  PREFLIGHT_NO_COMPOSE_DEPS=1 bash scripts/switch.sh ..." >&2
  return 1
}

# preflight_kv_format_hint <compose_file> — soft warning if the target compose
# uses a KV format known to be sub-optimal for the user's VRAM class.
#
# Specifically: dual-turbo.yml uses turboquant_3bit_nc which trips Cliff 2 at 90K
# on 20 GB Ampere even on TP=2. See docs/HARDWARE.md + #47 for the cross-rig data.
#
# Soft warning (returns 0). Skip via: PREFLIGHT_NO_KV_HINT=1
preflight_kv_format_hint() {
  local compose_file="$1"
  if [[ "${PREFLIGHT_NO_KV_HINT:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$compose_file" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  # Detect smallest VRAM among selected/visible cards (the TP-split ceiling).
  local min_vram_mib="" mem_query selector idx mem_mib
  selector="$(_preflight_selector || true)"
  mem_query="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
  while IFS=',' read -r idx mem_mib; do
    idx="$(_preflight_csv_token "$idx")"
    mem_mib="$(_preflight_csv_token "$mem_mib")"
    [[ -z "$idx" || -z "$mem_mib" ]] && continue
    _preflight_selector_allows_index "$selector" "$idx" || continue
    if [[ -z "$min_vram_mib" || "$mem_mib" -lt "$min_vram_mib" ]]; then
      min_vram_mib="$mem_mib"
    fi
  done <<< "$mem_query"
  if [[ -z "$min_vram_mib" ]] || [[ "$min_vram_mib" -ge 24000 ]]; then
    return 0   # 24 GB+ cards — TQ3 is the right pick, no hint needed
  fi

  local vram_gb=$((min_vram_mib / 1024))

  # Only fire on TQ3-using composes — that's where the 20 GB swap matters.
  if grep -qE -- '--kv-cache-dtype[[:space:]]*\n?[[:space:]]*-?[[:space:]]*turboquant_3bit_nc' "$compose_file" 2>/dev/null; then
    :
  elif grep -qE 'turboquant_3bit_nc' "$compose_file"; then
    :
  else
    return 0   # not a TQ3 compose
  fi

  echo "[preflight] HINT: smallest GPU has ~${vram_gb} GB VRAM and target compose uses TurboQuant 3-bit KV." >&2
  echo "[preflight]       On <24 GB Ampere, TQ3's activation peak during DeltaNet GDN forward exceeds" >&2
  echo "[preflight]       the per-card budget after TP split, and Cliff 2 fires at ~90K." >&2
  echo "[preflight]       Override with --kv-cache-dtype fp8_e5m2 in the compose file." >&2
  echo "[preflight]       Cross-rig validation: docs/HARDWARE.md + club-3090#47" >&2
  echo "[preflight]       Predict your config:  bash tools/kv-calc.py --compose <name> --vram ${vram_gb} --kv-format <fp8_e5m2|turboquant_3bit_nc>" >&2
  return 0
}

# autodetect_endpoint — discover the running club-3090 container + its host port.
#
# Caller-controlled: the bench / verify scripts default URL=http://localhost:8020
# and CONTAINER=vllm-qwen36-27b. That assumption breaks when the user is running
# a different variant (e.g. dual-turbo on 8011, dual-dflash on 8012, etc.) and
# silently makes verify-full / bench / verify-stress emit false negatives because
# they're hitting an empty port. Reported by sudepo on club-3090#52.
#
# Behaviour:
#   - If $URL or $CONTAINER is already set in the environment, it WINS — never
#     overwritten. This preserves explicit override behaviour.
#   - Otherwise, scan `docker ps` for a club-3090-pattern container and extract
#     its host port from the port-mapping. Print one [autodetect] line so the
#     user knows what we picked.
#   - If nothing is detected (no container running, docker unavailable), the
#     hardcoded defaults stand — same behaviour as before this helper existed.
#
# Outputs (mutates env in caller's scope when sourced):
#   URL          — http://localhost:<port> if detected
#   CONTAINER    — running container name if detected
#
# Skip via: PREFLIGHT_NO_AUTODETECT=1
preflight_autodetect_endpoint() {
  if [[ "${PREFLIGHT_NO_AUTODETECT:-0}" == "1" ]]; then
    return 0
  fi
  command -v docker >/dev/null 2>&1 || return 0

  local explicit_url="${URL:-}"
  local explicit_container="${CONTAINER:-}"
  if [[ -n "$explicit_url" && -n "$explicit_container" ]]; then
    return 0   # both already set — caller knows what they're doing
  fi

  # Detect a running inference container by its ENGINE-INTERNAL port mapping
  # (vLLM 8000 / llama.cpp 8080 / sglang 30000), NOT a hardcoded model-name
  # allowlist — so any compose is found regardless of model: gemma-4-12b,
  # qwen-35b-a3b, beellama, a BYO container, etc. (#310: the old allowlist only
  # knew qwen36-27b / gemma-4-31b, so everything else silently fell back to 8020).
  # Among matches, prefer a recognised club-3090 engine-family prefix; otherwise
  # take the first. Users running endpoint-first via `--url` bypass this entirely
  # (PREFLIGHT_NO_AUTODETECT=1 set there).
  #
  # The `|| true` is load-bearing: grep -E returns 1 when nothing matches, which
  # under `set -euo pipefail` in the caller would silently abort rebench-full.sh
  # before its own "endpoint not responding" path. Empty = the no-container case.
  local engine_lines found_line
  engine_lines=$(docker ps --format '{{.Names}}|{{.Ports}}' 2>/dev/null \
    | grep -E '([0-9]{1,3}\.){3}[0-9]{1,3}:[0-9]+->(8000|8080|30000)/tcp' || true)
  if [[ -z "$engine_lines" ]]; then
    return 0   # nothing serving on an engine port; defaults stand
  fi
  # Prefer a recognised club-3090 engine-family prefix when several match.
  found_line=$(printf '%s\n' "$engine_lines" \
    | grep -E '^(vllm-|llama-cpp-|ik-llama-|sglang-|beellama-)' | head -1 || true)
  [[ -z "$found_line" ]] && found_line=$(printf '%s\n' "$engine_lines" | head -1)
  # Several inference containers up → we picked one; tell the user how to override.
  if [[ "$(printf '%s\n' "$engine_lines" | grep -c .)" -gt 1 ]]; then
    echo "[autodetect] multiple inference containers running; picked '${found_line%%|*}' — set CONTAINER=/URL= to override" >&2
  fi

  local detected_name detected_port
  detected_name="${found_line%%|*}"
  # Extract host port from "0.0.0.0:8011->8000/tcp", "[::]:8011->8000/tcp",
  # or "127.0.0.1:8011->8000/tcp" forms (BIND_HOST=127.0.0.1 produces the last).
  # llama-cpp container maps to internal 8080, vllm to 8000, sglang to 30000.
  detected_port=$(echo "${found_line#*|}" \
    | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}:[0-9]+->(8000|8080|30000)/tcp' \
    | head -1 \
    | sed -E 's|^[^:]+:([0-9]+)->.*|\1|')

  # Apply, but only fields the user didn't already set explicitly.
  if [[ -z "$explicit_container" && -n "$detected_name" ]]; then
    CONTAINER="$detected_name"
  fi
  if [[ -z "$explicit_url" && -n "$detected_port" ]]; then
    URL="http://localhost:${detected_port}"
  fi

  # One-line surface so the user sees what we chose.
  if [[ -z "$explicit_url" || -z "$explicit_container" ]]; then
    local note=""
    [[ -z "$explicit_container" ]] && note="container=${CONTAINER}"
    [[ -z "$explicit_url" ]] && note="${note:+$note }url=${URL}"
    echo "[autodetect] using running ${note}  (skip: PREFLIGHT_NO_AUTODETECT=1)" >&2
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Resolve the SERVED model name from the running endpoint's /v1/models when the
# caller hasn't set MODEL explicitly. Mirrors what soak-test.sh / bench-agentic.sh
# / quality-test.sh already do — so verify-full / verify-stress / bench / verify
# send the model the server actually serves instead of a hardcoded qwen default.
#
# Why this exists (#372): report.sh autodetects container + URL + engine but NOT
# the served model, so the verify/bench scripts fell back to MODEL=qwen3.6-27b-
# autoround. Against a non-qwen vLLM endpoint (e.g. gemma-4-26b-a4b-awq) every
# request 404'd ("The model `qwen3.6-27b` does not exist"). llama.cpp
# ignores the request's model field, so the same wrong default silently "worked"
# there (#371) — which is exactly what masked the bug.
#
# Sets MODEL in the caller's scope. No-op when MODEL is already set (an explicit
# env/flag value always wins — critical for llama-swap / multi-model endpoints
# where /v1/models returns the first, often wrong, registered model), when there
# is no URL to query, or when the endpoint isn't reachable (the caller's own
# reachability check then surfaces the real outage). Callers keep their own
# last-resort literal after this, so behaviour is unchanged when detection no-ops.
preflight_autodetect_model() {
  [[ -n "${MODEL:-}" ]] && return 0
  local url="${1:-${URL:-}}"
  [[ -n "$url" ]] || return 0
  command -v curl >/dev/null 2>&1 || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  local detected
  detected="$(curl -sf -m 5 "${url%/}/v1/models" 2>/dev/null \
    | python3 -c "import json,sys
try:
    d = json.load(sys.stdin).get('data', [])
    print(d[0]['id'] if d else '')
except Exception:
    print('')" 2>/dev/null || true)"
  if [[ -n "$detected" ]]; then
    MODEL="$detected"
    echo "[autodetect] served model='${MODEL}' (from ${url%/}/v1/models; set MODEL= to override)" >&2
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Resolve WHICH chat_template_kwargs key controls reasoning on the served model.
# Sets THINK_CONTROL / THINK_OFF_KW / THINK_ON_KW in the caller's scope.
#
# The key is NOT universal, and the whole script layer had the Qwen one baked in:
#   Qwen3.x + most families → {"enable_thinking": false|true}
#   Inkling (TML)           → {"reasoning_effort": "none"|"high"} — a DIAL
#                             (none/minimal/low/medium/high/xhigh/max, or a
#                             float), defaulting to 0.9 "high", NOT a boolean
#   neither                 → {} (caller decides whether to add budget headroom)
#
# Why this exists (found on Inkling-Small 2026-08-12): an unrecognised kwarg is
# silently IGNORED — no error, no warning. The model then reasons at full effort,
# and a short-budget check spends its entire allowance on the reasoning preamble
# before emitting a single content token. verify-full's [3/9] and [5/9] failed
# structurally on such a model, reporting "Model may be loading badly or wrong
# chat template" — sending you to debug a template that was in fact correct.
#
# ⚠️ Both switches must go in chat_template_kwargs. llama-server does NOT map the
# top-level OpenAI-standard `reasoning_effort` request parameter into the
# template; passing it there is silently ignored (measured, same session).
#
# ⚠️ THINK_*_KW hold FINAL JSON text, not backslash-escaped source. Callers
# interpolate them into a double-quoted payload, and bash processes \" escapes
# BEFORE parameter expansion — an escaped value reaches curl with its
# backslashes intact and every request 400s.
#
# Detection order is deliberate: enable_thinking is checked FIRST so that a
# template supporting both keeps the exact request shape it has today. This can
# only add coverage, never change an existing model's result.
#
# No-ops when THINK_CONTROL is already set, or when there's no URL / curl /
# python3 — callers keep working defaults. Overridden by VERIFY_THINK_OFF /
# VERIFY_THINK_ON (plain JSON objects).
_preflight_probe_thinking_key() {
  # Echo the content length a trivial question returns under the given kwargs.
  # A working off-switch answers in a few tokens; an ignored one burns the whole
  # budget reasoning and returns empty content.
  local url="$1" model="$2" kwargs="$3"
  curl -sf -m 90 "${url%/}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${model}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}], \"max_tokens\": 24, \"temperature\": 0.0, \"chat_template_kwargs\": ${kwargs}}" 2>/dev/null \
    | python3 -c "import sys,json; print(len((json.load(sys.stdin)['choices'][0]['message'].get('content') or '').strip()))" 2>/dev/null \
    || echo 0
}

_preflight_probe_thinking_reasoning_both() {
  # Same as the sibling below, but ALSO sends the TOP-LEVEL `reasoning_effort`
  # field alongside the kwargs — i.e. the exact shape verify-full builds from
  # THINK_OFF_STD + THINK_OFF_KW. Exists to detect models where the two DISAGREE.
  local url="$1" model="$2" kwargs="$3" effort="$4"
  curl -sf -m 90 "${url%/}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${model}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}], \"max_tokens\": 96, \"reasoning_effort\": \"${effort}\", \"chat_template_kwargs\": ${kwargs}}" 2>/dev/null \
    | python3 -c "import sys,json; print(len((json.load(sys.stdin)['choices'][0]['message'].get('reasoning_content') or '')))" 2>/dev/null \
    || echo 0
}

_preflight_probe_thinking_reasoning() {
  # Echo the REASONING length a trivial question returns under the given kwargs.
  # The sibling probe measures CONTENT — right for proving a value turns thinking
  # OFF, useless for proving one turns it ON. max_tokens stays tiny so a max-effort
  # model cannot burn minutes here (GLM at effort=max reasons ~6,050 tokens on a
  # real prompt; capped at 24 it just emits a preamble).
  local url="$1" model="$2" kwargs="$3"
  curl -sf -m 90 "${url%/}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${model}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}], \"max_tokens\": 96, \"temperature\": 0.0, \"chat_template_kwargs\": ${kwargs}}" 2>/dev/null \
    | python3 -c "import sys,json; print(len((json.load(sys.stdin)['choices'][0]['message'].get('reasoning_content') or '').strip()))" 2>/dev/null \
    || echo 0
}

preflight_detect_thinking_control() {
  [[ -n "${THINK_CONTROL:-}" ]] && return 0
  local url="${1:-${URL:-}}"
  local model="${2:-${MODEL:-}}"
  THINK_CONTROL="enable_thinking"   # safe default = today's behaviour
  if [[ -n "$url" ]] && command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    local tmpl
    # 1. Exact — read the live chat template when the engine exposes it
    #    (llama.cpp /props). Free, and needs no inference.
    tmpl="$(curl -sf -m 5 "${url%/}/props" 2>/dev/null \
      | python3 -c "import sys,json; print(json.load(sys.stdin).get('chat_template') or '')" 2>/dev/null || true)"
    if [[ -n "$tmpl" ]]; then
      case "$tmpl" in
        *enable_thinking*)  THINK_CONTROL="enable_thinking"  ;;
        *reasoning_effort*) THINK_CONTROL="reasoning_effort" ;;
        *)                  THINK_CONTROL="none"             ;;
      esac
      # 1b. VERIFY the switch is HONOURED, not merely MENTIONED.
      #
      # The scan above proves a template NAMES a key. It does not prove the model
      # OBEYS it, and those differ in practice:
      #   - GLM-5.3-Flash (2026-08-29) names `reasoning_effort` but has FORCED
      #     thinking (vendor: thinking.type=disabled is an ERROR on 5.3/5.3-FLASH).
      #     The scan set THINK_CONTROL=reasoning_effort, so TOK_SCALE stayed 1 and
      #     verify-full handed a forced-thinking model a 30-TOKEN budget. 4/9 failed
      #     — [3] basic, [5] streaming, [7] thinking, [8] quality — every
      #     empty-content failure being budget exhaustion, while tool-calling passed
      #     TWICE. The engine was fine; the hint ("Model may be loading badly or
      #     wrong chat template") pointed at the model. That is exactly the failure
      #     this file's header describes, reached via the SCAN branch rather than
      #     the probe branch it was written to cover.
      #   - An unsupported ENUM VALUE fails the same silent way: `reasoning_effort:
      #     none` is not valid for the GLM-5.3 family and is dropped without error.
      #
      # Same opt-in and the same load-bearing reachability gate as the probe branch
      # below: MEASUREMENT scripts never reach this (THINK_PROBE unset), and an
      # unreachable endpoint falls through to the scan's answer rather than being
      # misread as "no switch" and inflating token budgets 64x.
      #
      # ASYMMETRIC ON PURPOSE: a switch that WORKS keeps the scan's control
      # untouched, so every model passing today keeps its exact request shape
      # (scenario 3's no-regression promise holds). Only a demonstrably-IGNORED
      # switch is downgraded to `none` — which is what makes the caller widen its
      # token budgets instead of blaming the model.
      if [[ "$THINK_CONTROL" != "none" ]] && [[ "${THINK_PROBE:-0}" == "1" ]] \
         && [[ -n "$model" ]] && curl -sf -m 5 "${url%/}/v1/models" >/dev/null 2>&1; then
        local _off_probe_kw=""
        case "$THINK_CONTROL" in
          enable_thinking)  _off_probe_kw='{"enable_thinking": false}'   ;;
          reasoning_effort) _off_probe_kw='{"reasoning_effort": "none"}' ;;
        esac
        if [[ -n "$_off_probe_kw" ]] \
           && [[ "$(_preflight_probe_thinking_key "$url" "$model" "$_off_probe_kw")" == "0" ]]; then
          # ⚠️ "this VALUE did not work" is NOT "this KEY does not exist" — the
          # distinction caused a real misdiagnosis. `reasoning_effort: none` is
          # INVALID on the GLM-5.3 family (vendor lists max|high|low only) and is
          # silently ignored, so probing ONLY `none` declared GLM switch-less.
          # Downstream that set THINK_OFF_KW={} — no switch on any check — and
          # verify-full's [8] then spent its entire budget reasoning and failed with
          # "empty completion", blaming the model rather than the probe value.
          # So before giving up on an effort DIAL, try the other documented minimum.
          # `low` is valid for GLM-5.3/5.3-Flash and is a listed level in llama.cpp's
          # own --reasoning-effort help (minimal|low|medium|high|xhigh|max).
          # ⚠️ WALK A LADDER, DO NOT HARDCODE A LEVEL. Valid levels differ per
          # family and the dial is not always monotonic:
          #   GLM-5.3       — `none` INVALID (max|high|low only); `high` measured
          #                   beside `low` (11 ch vs 0), so `high` is NOT "on".
          #   Qwen3.8-27B   — template FORCES xhigh and remaps high -> xhigh.
          # llama.cpp's own --reasoning-effort help lists the level set:
          #   minimal | low | medium | high | xhigh | max
          # OFF ladder: first level that lets a 24-token reply produce CONTENT.
          # ON  ladder: first level that produces REASONING.
          # A model whose first candidate works is unaffected — the ladder stops there.
          if [[ "$THINK_CONTROL" == "reasoning_effort" ]]; then
            local _lvl
            for _lvl in minimal low medium; do
              if [[ "$(_preflight_probe_thinking_key "$url" "$model" "{\"reasoning_effort\": \"${_lvl}\"}")" != "0" ]]; then
                THINK_EFFORT_OFF_VALUE="$_lvl"; break
              fi
            done
            if [[ -z "${THINK_EFFORT_OFF_VALUE:-}" ]]; then
              THINK_CONTROL="none"
            else
              # ⚠️ THE BAR MUST MATCH THE CONSUMER'S. verify-full [7] fails a level
              # whose reasoning is <50 chars ("suspiciously short"). An earlier
              # revision accepted ANY non-zero reasoning, so the ladder blessed
              # GLM's `high` on ~11 chars and [7] then REJECTED the value the probe
              # had just chosen — probe and check disagreeing about what "thinking
              # is on" means. 50 is that consumer's threshold; the probe budget
              # above is sized to clear it comfortably (96 tok >> 50 chars).
              local _min_reasoning=50 _rlen
              for _lvl in high xhigh max; do
                _rlen="$(_preflight_probe_thinking_reasoning "$url" "$model" "{\"reasoning_effort\": \"${_lvl}\"}")"
                if [[ "$_rlen" =~ ^[0-9]+$ ]] && (( _rlen >= _min_reasoning )); then
                  THINK_EFFORT_ON_VALUE="$_lvl"; break
                fi
              done
            fi
          else
            THINK_CONTROL="none"
          fi
        fi
      fi
    elif [[ "${THINK_PROBE:-0}" == "1" ]] && [[ -n "$model" ]] && curl -sf -m 5 "${url%/}/v1/models" >/dev/null 2>&1; then
      # 2. Behavioural probe for engines that don't expose the template (vLLM,
      #    SGLang). At most two tiny requests, and only when step 1 no-ops.
      #
      # ⚠️ OPT-IN via THINK_PROBE=1, and deliberately so: this fires REAL
      # inference requests. Functional checks (verify / verify-full) opt in
      # because one extra request is harmless there. MEASUREMENT scripts
      # (bench, soak, power-cap-sweep, quality-test) must NOT, for two reasons:
      # it puts uncontrolled requests on the server before warmup, and against
      # a scripted/mocked endpoint it consumes responses the run expects —
      # which is exactly how it broke the soak fixtures, silently shifting
      # every turn's response by two.
      #
      # ⚠️ The /v1/models reachability gate above is load-bearing. The probe
      # reads "empty content" as "this switch is ignored", and a request that
      # never reached the server also returns empty — so without the gate an
      # unreachable or still-warming endpoint is misread as "model has no
      # reasoning switch", which both changes the request shape and (in
      # verify-full) inflates token budgets by 64×. Unreachable must fall
      # through to the safe default instead, and let the caller's own
      # reachability check surface the real outage.
      if [[ "$(_preflight_probe_thinking_key "$url" "$model" '{"enable_thinking": false}')" != "0" ]]; then
        THINK_CONTROL="enable_thinking"
      elif [[ "$(_preflight_probe_thinking_key "$url" "$model" '{"reasoning_effort": "none"}')" != "0" ]]; then
        THINK_CONTROL="reasoning_effort"
      else
        THINK_CONTROL="none"
      fi
    fi
  fi
  # THINK_*_STD is the OpenAI-standard TOP-LEVEL parameter, emitted as a JSON
  # fragment (trailing comma included) to sit alongside the kwargs object.
  #
  # `reasoning_effort` IS the standard, and Thinking Machines' own API takes it
  # top-level (tinker-docs "compatible-apis/openai"): none/minimal/low/medium/
  # high/xhigh or a float in [0.0, 0.99], default 0.9. We send it — but we
  # cannot send it ALONE, because llama.cpp only half-implements it:
  #
  #   server-common.cpp: if (reasoning_effort == "none") inputs.enable_thinking = false;
  #                      // other reasoning_effort values are model-specific and not yet handled
  #
  # i.e. the one handled value is mapped onto `enable_thinking`, a template
  # variable this family does NOT read, and every other value is dropped
  # silently. So on llama.cpp the chat_template_kwargs copy is what actually
  # reaches the template. Sending both is verified non-conflicting (measured
  # 2026-08-12: none → 10 tok / 0 reasoning; high → 45 tok / 152 chars) and
  # makes the request correct on engines that DO implement the standard.
  # Drop the kwargs fallback once llama.cpp forwards the value into the template.
  # THINK_*_EFFORT are the same top-level value as a PLAIN string (empty when the
  # model has no effort dial), for consumers that build their payload in Python
  # rather than by string-splicing JSON — they set req["reasoning_effort"] from
  # it directly instead of parsing the fragment above.
  THINK_OFF_STD=''
  THINK_ON_STD=''
  THINK_EFFORT_OFF_VALUE="${THINK_EFFORT_OFF_VALUE:-}"
  THINK_EFFORT_ON_VALUE="${THINK_EFFORT_ON_VALUE:-}"
  THINK_OFF_EFFORT=''
  THINK_ON_EFFORT=''
  case "$THINK_CONTROL" in
    reasoning_effort)
      # The OFF value is whatever the probe PROVED works — `none` by default, `low`
      # on families where `none` is not a valid level (GLM-5.3). Hardcoding `none`
      # here is what silently disabled the switch on those models.
      _eoff="${THINK_EFFORT_OFF_VALUE:-none}"
      _eon="${THINK_EFFORT_ON_VALUE:-high}"
      THINK_OFF_KW="{\"reasoning_effort\": \"${_eoff}\"}"
      THINK_ON_KW="{\"reasoning_effort\": \"${_eon}\"}"
      THINK_OFF_STD="\"reasoning_effort\": \"${_eoff}\", "
      THINK_ON_STD="\"reasoning_effort\": \"${_eon}\", "
      THINK_OFF_EFFORT="${_eoff}"
      THINK_ON_EFFORT="${_eon}"
      # ⚠️⚠️ DOES THE TOP-LEVEL FIELD DEFEAT THE KWARG? Probe, never assume.
      # Some models read `reasoning_effort` from the CHAT TEMPLATE and ignore
      # `enable_thinking` — which is what llama.cpp translates the TOP-LEVEL
      # field into (server-common.cpp). On those, sending BOTH makes the
      # top-level one win and the kwarg never takes effect, so a caller that
      # believes it disabled thinking gets a reasoning-only reply.
      # MEASURED on Inkling-Small 2026-08-29 (identical prompt, 30-tok budget):
      #     kwarg only      ->   0 ch reasoning, correct content
      #     top-level only  -> 132 ch reasoning, EMPTY content
      #     BOTH            -> 132 ch reasoning, EMPTY content  (== top-level)
      # verify-full sends BOTH, so [3]/[5] failed on a healthy model.
      # Keep the top-level fragment ONLY where it does no harm.
      if [[ -n "${url:-}" && -n "${model:-}" ]]; then
        local _r_kw _r_both
        _r_kw="$(_preflight_probe_thinking_reasoning "$url" "$model" "$THINK_OFF_KW")"
        _r_both="$(_preflight_probe_thinking_reasoning_both "$url" "$model" "$THINK_OFF_KW" "$_eoff")"
        if [[ "$_r_kw" =~ ^[0-9]+$ && "$_r_both" =~ ^[0-9]+$ ]] \
           && (( _r_kw < 20 )) && (( _r_both >= 20 )); then
          THINK_OFF_STD=''
          THINK_PAYLOAD_NOTE="kwargs-only (top-level field defeats the kwarg: ${_r_kw}ch vs ${_r_both}ch)"
        fi
      fi ;;
    none)
      THINK_OFF_KW='{}'
      THINK_ON_KW='{}' ;;
    *)
      THINK_OFF_KW='{"enable_thinking": false}'
      THINK_ON_KW='{"enable_thinking": true}' ;;
  esac
  # An explicit override is the full statement of intent — it replaces the
  # detected kwargs AND suppresses the top-level fragment, so the two can't
  # disagree in the same request.
  [[ -n "${VERIFY_THINK_OFF:-}" ]] && { THINK_OFF_KW="${VERIFY_THINK_OFF}"; THINK_OFF_STD=''; THINK_OFF_EFFORT=''; THINK_CONTROL="${THINK_CONTROL} (off overridden)"; }
  [[ -n "${VERIFY_THINK_ON:-}"  ]] && { THINK_ON_KW="${VERIFY_THINK_ON}";   THINK_ON_STD='';  THINK_ON_EFFORT='';  THINK_CONTROL="${THINK_CONTROL} (on overridden)"; }
  # THINK_FRAG_* is the COMPLETE request fragment as one JSON object — the kwargs
  # plus, when applicable, the top-level standard parameter. Python consumers
  # splat it into their payload in a single uniform line:
  #   **json.loads(os.environ.get("THINK_FRAG_OFF") or '{"chat_template_kwargs": {"enable_thinking": false}}')
  # ⚠️ Build these with plain assignments, NEVER `X="$(cond && printf …)"`. Under
  # `set -e` (every caller) a command substitution whose last command fails makes
  # the ASSIGNMENT return non-zero, which aborts the function mid-way — leaving
  # THINK_* half-set and taking the calling script down with it. That is not
  # theoretical: it broke 10 test suites in one commit.
  local _off_extra='' _on_extra=''
  if [[ -n "$THINK_OFF_EFFORT" ]]; then _off_extra=", \"reasoning_effort\": \"${THINK_OFF_EFFORT}\""; fi
  if [[ -n "$THINK_ON_EFFORT"  ]]; then _on_extra=", \"reasoning_effort\": \"${THINK_ON_EFFORT}\""; fi
  THINK_FRAG_OFF="{\"chat_template_kwargs\": ${THINK_OFF_KW}${_off_extra}}"
  THINK_FRAG_ON="{\"chat_template_kwargs\": ${THINK_ON_KW}${_on_extra}}"
  # Announce ONLY when the answer is non-default. For every model that already
  # used the Qwen key this function is now completely silent, so output-drift
  # guards (test-soak-decode-basis compares stdout against origin/master) and
  # anything else parsing these logs stay byte-identical. A surprising answer is
  # worth a line; confirming the status quo is not.
  if [[ "$THINK_CONTROL" != "enable_thinking" ]]; then
    echo "[autodetect] thinking-control='${THINK_CONTROL}' off=${THINK_OFF_KW} (set VERIFY_THINK_OFF/ON to override)" >&2
  fi
  return 0
}

# ── #633 — ik-llama cu13/cu12 driver-aware image selection ────────────────────
# The pinned cu13 ik-llama image has a CUDA 13.2 runtime; on a driver whose
# supported CUDA < 13.2 the forward-compat path fails on GeForce (CUDA error
# 804) → silent CPU fallback → segfault crash-loop, with no actionable hint
# (launch just times out after 600 s). Auto-pick the cu12 sibling build (same
# build 4574, CUDA 12.6, backward-compatible with the 13.0 driver — validated on
# a 580.159 rig, #633) unless the user pinned IK_LLAMA_IMAGE. ik-llama only.
IK_LLAMA_CU13_MIN_CUDA="13.2"
IK_LLAMA_CU12_FALLBACK="${IK_LLAMA_CU12_FALLBACK:-ghcr.io/ikawrakow/ik-llama-cpp:cu12-server-4574}"

# _cuda_ge A B → 0 (true) iff major.minor A >= B
_cuda_ge() {
  local a1="${1%%.*}" b1="${2%%.*}" a2 b2
  a2="${1#*.}"; [[ "$a2" == "$1" ]] && a2=0
  b2="${2#*.}"; [[ "$b2" == "$2" ]] && b2=0
  (( 10#${a1:-0} > 10#${b1:-0} )) && return 0
  (( 10#${a1:-0} < 10#${b1:-0} )) && return 1
  (( 10#${a2:-0} >= 10#${b2:-0} ))
}

# Driver's max supported CUDA (major.minor), or "" if undetectable.
_driver_cuda_version() {
  local v
  v="$(nvidia-smi --query 2>/dev/null | grep -m1 -oE 'CUDA Version[[:space:]]*:[[:space:]]*[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || true)"
  [[ -z "$v" ]] && v="$(nvidia-smi 2>/dev/null | grep -m1 -oE 'CUDA Version:?[[:space:]]*[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || true)"
  printf '%s' "$v"
}

# preflight_single_card_util <compose_file> [variant]
# Advisory WARN (never blocks; runs even under --force) when the user raises
# GPU_MEMORY_UTILIZATION above the compose's validated default on a SINGLE-CARD
# (TP<=1) config. On one GPU a higher util shrinks the free VRAM a large
# tool-response prefill needs for its activation peak, so vLLM OOMs mid-prefill
# even though boot succeeds (verify-stress step 2/8). Two 5090 testers hit this
# by setting util=0.92 on the nvfp4 slug whose validated default is 0.85 (#617).
# No-op unless the user overrode the env AND the compose ships a
# `GPU_MEMORY_UTILIZATION:-<default>` (so it never fires for non-vLLM engines,
# dual/multi-card, or a plain default run).
preflight_single_card_util() {
  local compose_file="$1" variant="${2:-}"
  [[ -f "$compose_file" ]] || return 0
  local user_util="${GPU_MEMORY_UTILIZATION:-}"
  [[ -n "$user_util" ]] || return 0                 # only when explicitly overridden

  local tp=""
  if declare -F compose_meta_get >/dev/null 2>&1; then
    tp="$(compose_meta_get "$compose_file" tensor-parallel 2>/dev/null || true)"
  fi
  [[ "$tp" =~ ^[0-9]+$ ]] || return 0               # unknown topology → stay quiet
  (( tp <= 1 )) || return 0                          # single-card only

  local default_util
  default_util="$(command grep -oE 'GPU_MEMORY_UTILIZATION:-[0-9.]+' "$compose_file" | head -1 | sed -E 's/.*:-//')"
  [[ -n "$default_util" ]] || return 0               # compose ships no util default

  if awk -v u="$user_util" -v d="$default_util" 'BEGIN{exit !((u+0) > (d+0))}'; then
    echo "[preflight] WARN:  GPU_MEMORY_UTILIZATION=${user_util} exceeds ${variant:-this config}'s validated single-card default of ${default_util}." >&2
    echo "[preflight]        On one GPU a higher util steals the headroom a large tool-response prefill needs for its" >&2
    echo "[preflight]        activation peak — vLLM can OOM mid-prefill (verify-stress step 2/8) even though boot succeeds." >&2
    echo "[preflight]        Fix: grow context via MAX_MODEL_LEN and keep GPU_MEMORY_UTILIZATION <= ${default_util}. (#617)" >&2
  fi
  return 0
}

preflight_ik_llama_image() {
  local variant="${1:-}"
  [[ "$variant" == ik-llama/* ]] || return 0
  if [[ -n "${IK_LLAMA_IMAGE:-}" ]]; then
    echo "[preflight] ik-llama image pinned (.env/shell): ${IK_LLAMA_IMAGE}"
    return 0
  fi
  local drv; drv="$(_driver_cuda_version)"
  [[ -z "$drv" ]] && return 0                    # undetectable → leave compose default (cu13)
  _cuda_ge "$drv" "$IK_LLAMA_CU13_MIN_CUDA" && return 0   # >=13.2 → cu13 runtime OK
  export IK_LLAMA_IMAGE="$IK_LLAMA_CU12_FALLBACK"
  echo "[preflight] ⚠ driver CUDA ${drv} < ${IK_LLAMA_CU13_MIN_CUDA} (the cu13 ik-llama pin's runtime)." >&2
  echo "[preflight]   cu13 would forward-compat-fail on GeForce (error 804) → CPU fallback → crash loop (#633)." >&2
  echo "[preflight]   Auto-selected the cu12 sibling build (same build, CUDA 12.6, backward-compatible):" >&2
  echo "[preflight]     IK_LLAMA_IMAGE=${IK_LLAMA_CU12_FALLBACK}" >&2
  echo "[preflight]   Pin IK_LLAMA_IMAGE in .env to override." >&2
}

# ---------------------------------------------------------------------------
# CPU-offload guards.
#
# MARKER-SCOPED, never model-scoped. Model-keyed guards rot: every new offload
# model needs an update and the one that gets forgotten is the one that bites.
# Keying on the offload flags means every future offload model inherits these
# for free. Same convention as preflight_lmcache_ram(), which keys on a header.
#
# ⭐ Our OWN composes are correct by construction — we author and measure them.
# The entire value of these guards is in configs we DON'T control: user edits,
# community composes, someone adapting our pattern to a different MoE.
# ---------------------------------------------------------------------------

# is_cpu_offload_compose <compose_file>
# True when the compose actually offloads experts to host RAM.
#
# ⚠️ Match `=CPU` SPECIFICALLY, not the presence of `-ot`. Our offload composes
#    carry two-to-four `-ot` rules that are `=CUDA0`/`=CUDA1`/... RESIDENCY —
#    GPU-side placement, the exact opposite of offload. A naive "has -ot" test
#    false-positives on a fully GPU-resident config.
is_cpu_offload_compose() {
  local compose_file="$1"
  [[ -f "$compose_file" ]] || return 1
  # ⚠️ Must work for BOTH compose command forms. The list form puts every arg on its
  # own line ("      - '-ot'" / "      - '...=CPU'"), so a same-line regex silently
  # misses it — which is exactly what happened when these composes moved to list form
  # (Docker Compose rejects `(gate|up|down)` in a folded string). Match per-token.
  command grep -qE -- "=CPU|--cpu-moe|(^|[[:space:]'\"])-cmoe([[:space:]'\"]|$)|--n-cpu-moe|(^|[[:space:]'\"])-ncmoe([[:space:]'\"]|$)" \
    "$compose_file"
}

# preflight_offload_split_mode <compose_file>
# Refuses split modes that are measured-catastrophic under CPU offload.
#
# GUARD AND ADVISE — never auto-switch the mode. detect_nvlink.sh's auto mode is
# the cautionary tale in this repo: it enables P2P on a DETECTED GRANT ALONE, and
# that is what took the reference rig to a hang. Auto-selecting a split mode from
# a detected property repeats that one layer up.
preflight_offload_split_mode() {
  local compose_file="$1"
  [[ -f "$compose_file" ]] || return 0
  is_cpu_offload_compose "$compose_file" || return 0   # not an offload compose → no-op

  local sm="${SPLIT_MODE:-}"
  # same-line form (folded command:)
  [[ -z "$sm" ]] && sm="$(command grep -oE -- '(--split-mode|[[:space:]]-sm)[[:space:]]+[a-z]+' "$compose_file" | head -1 | awk '{print $NF}')"
  # list form: the VALUE is the next list item after the flag
  # list form: each arg is its own list item, so the VALUE is the NEXT line.
  # A same-line regex silently misses it — the exact regression that shipped when
  # these composes moved to list form (Compose rejects `(gate|up|down)` when folded).
  if [[ -z "$sm" ]]; then
    sm="$(awk -F"['\" ]+" '
      /^[[:space:]]*-[[:space:]]+.?(--split-mode|-sm).?[[:space:]]*$/ { want=1; next }
      want { for (i=1;i<=NF;i++) if ($i ~ /^(layer|row|tensor|none)$/) { print $i; exit } want=0 }
    ' "$compose_file" | head -1)"
  fi
  [[ -z "$sm" ]] && return 0

  if [[ "$sm" == "tensor" || "$sm" == "row" ]]; then
    echo "[preflight] ERROR: --split-mode ${sm} with CPU expert offload is measured-catastrophic." >&2
    echo "            Offloaded prefill: 305 t/s vs 2031 t/s on --split-mode layer (-85%)." >&2
    echo "            Offloaded decode:  32.75 vs 47.14." >&2
    echo "            With speculative decoding it SIGSEGVs on the second request while the" >&2
    echo "            container still reports healthy — the worst possible failure shape." >&2
    echo "            Why: under offload the GPU is idle ~87% of the time (SM 12.8%) waiting on" >&2
    echo "            sequential CPU handoffs. TP parallelises GPU compute — which is NOT the" >&2
    echo "            bottleneck — and adds an all-reduce that scales with batch size, so prefill" >&2
    echo "            pays it ~1000x harder than decode. The experts are not even on the GPUs." >&2
    echo "            Fix: use --split-mode layer (the llama.cpp default for multi-GPU)." >&2
    return 1
  fi

  # Uneven VRAM: TP wants an even split, and offload composes pin per-device rules.
  if [[ "$sm" == "layer" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    local sizes; sizes="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -u | wc -l)"
    if [[ "${sizes:-1}" -gt 1 ]]; then
      echo "[preflight] NOTE: mixed GPU memory sizes detected — residency is sized per device," >&2
      echo "            so asymmetric counts are expected and fine on --split-mode layer." >&2
    fi
  fi
  return 0
}

# preflight_cpu_offload_ram <compose_file>
# Guards an offload compose against a host that cannot hold the experts.
#
# Reads the compose's `CPU-Offload-Host-RAM-GB:` header, which declares the
# WORST CASE (all experts on CPU), then subtracts the expert bundles the launcher
# will pin onto THIS rig's GPUs (offload_residency_grant_mib — the launcher's own
# arithmetic, so gate and boot cannot disagree). Every pinned bundle is bytes NOT
# in host RAM, which is why a 4-card rig needs less than a 2-card one and a
# big-VRAM rig less than the header. When VRAM is unreadable the grant is 0 and
# the gate degrades to the worst case — the safe direction.
#
# ⚠️ The header MUST stay the all-experts-on-CPU number. Pre-baking expected
#    residency into it double-counts once the gate subtracts, and under-gates
#    rigs whose cards fit fewer bundles than the header assumed (a 4x16 GB rig
#    fits ZERO and truly needs the full worst case).
#
# ⚠️ The number is MEASURED, not computed. The quant name does not give you the
#    byte size (UD-Q8_K_XL is really MXFP4 experts + BF16 attention), and on
#    Unsloth *Dynamic* quants the per-layer bundles are not even uniform. Do not
#    "simplify" this into a formula.
preflight_cpu_offload_ram() {
  local compose_file="$1"
  [[ -f "$compose_file" ]] || return 0
  declare -F compose_meta_get >/dev/null 2>&1 || return 0

  local need_hdr
  need_hdr="$(compose_meta_get "$compose_file" cpu-offload-host-ram-gb || true)"
  [[ -z "$need_hdr" ]] && return 0                 # not an offload compose we size
  if ! [[ "$need_hdr" =~ ^[0-9]+$ ]]; then
    echo "[preflight] WARN:  CPU-Offload-Host-RAM-GB='${need_hdr}' is not an integer; skipping." >&2
    return 0
  fi

  # Residency-adjusted need for THIS rig. Integer division floors the subtraction,
  # which errs conservative (subtracts slightly less than granted).
  local grant_mib=0 grant_gb=0 need_gb="$need_hdr"
  if declare -F offload_residency_grant_mib >/dev/null 2>&1; then
    grant_mib="$(offload_residency_grant_mib "$compose_file" 2>/dev/null || printf '0')"
    [[ "$grant_mib" =~ ^[0-9]+$ ]] || grant_mib=0
    if (( grant_mib > 0 )); then
      grant_gb=$(( grant_mib / 1024 ))
      need_gb=$(( need_hdr - grant_gb ))
      (( need_gb < 1 )) && need_gb=1
    fi
  fi

  if [[ ! -r /proc/meminfo ]]; then
    echo "[preflight] WARN:  cannot read /proc/meminfo; skipping CPU-offload RAM check." >&2
    return 0
  fi
  local kb total_gb avail_gb
  kb="$(awk '/^MemTotal:/{print $2}' /proc/meminfo)";     total_gb=$(( kb / 1024 / 1024 ))
  kb="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"; avail_gb=$(( kb / 1024 / 1024 ))

  if (( total_gb < need_gb )); then
    echo "[preflight] ERROR: this compose offloads experts to host RAM and needs ~${need_gb} GB" >&2
    if (( grant_gb > 0 )); then
      echo "            on this rig (${need_hdr} GB worst case, minus ~${grant_gb} GB of experts" >&2
      echo "            residency keeps on your GPUs), but the machine has only ${total_gb} GB TOTAL." >&2
    else
      echo "            (worst case: all experts on CPU), but the machine has only ${total_gb} GB TOTAL." >&2
    fi
    echo "            It cannot run here." >&2
    echo "            This is a hard gate, not a tuning knob: below it the box thrashes or OOMs." >&2
    echo "            Fix: use a lower-bit tier (the IQ2 slug needs ~86 GB), add RAM, or pick a" >&2
    echo "            model that fits VRAM." >&2
    # ⚠️ "more cards => less host RAM" holds ONLY for composes that can pin expert
    # bundles back onto the GPUs, i.e. ones carrying the residency headers. The
    # moe-cache composes pin NOTHING (their -ot is an unconditional =CPU catch-all;
    # the expert cache is a VRAM-side copy, so the CPU master buffer stays whole).
    # Printing the hint there sends a RAM-short user shopping for GPUs that cannot
    # help. Key off the same header the sizer uses, not the model name.
    if [[ "$(compose_meta_get "$compose_file" cpu-offload-bundle-mib || true)" =~ ^[0-9]+$ ]]; then
      echo "            On a rig with more cards the same model needs LESS host RAM, because" >&2
      echo "            residency moves expert bytes onto the GPUs." >&2
    else
      echo "            NOTE: more GPUs will NOT lower this compose's host-RAM need — it keeps" >&2
      echo "            every expert on the CPU regardless of card count." >&2
    fi
    return 1
  fi
  if (( avail_gb < need_gb )); then
    echo "[preflight] ERROR: needs ~${need_gb} GB of host RAM; only ${avail_gb} GB is AVAILABLE" >&2
    echo "            (of ${total_gb} GB total). Something else is holding memory." >&2
    echo "            Fix: stop other services and retry — 'docker ps' is the usual culprit." >&2
    return 1
  fi
  if (( grant_gb > 0 )); then
    echo "[preflight] cpu-offload: RAM ok — needs ~${need_gb} GB on this rig (${need_hdr} GB worst" \
         "case, residency keeps ~${grant_gb} GB of experts on GPU), ${avail_gb} GB available"
  else
    echo "[preflight] cpu-offload: RAM ok — needs ~${need_hdr} GB (worst case), ${avail_gb} GB available"
  fi
  return 0
}
