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
#   preflight_genesis_pin     — warn if on-disk Genesis tree differs from setup.sh's pin
#   preflight_repo_drift      — warn if local HEAD is behind origin/master
#   preflight_compose_hardware— check compose VRAM/GPU-count/SM metadata
#
# Style: each function prints one or more "[preflight] ..." lines.
# Hard failures get a one-line ERROR + a "Fix:" hint.

# Avoid double-sourcing.
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

_preflight_selector() {
  if [[ -n "${CLUB3090_GPU:-}" ]]; then
    printf '%s' "${CLUB3090_GPU}"
  elif [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "all" && "${NVIDIA_VISIBLE_DEVICES}" != "void" ]]; then
    printf '%s' "${NVIDIA_VISIBLE_DEVICES}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "all" && "${CUDA_VISIBLE_DEVICES}" != "void" ]]; then
    printf '%s' "${CUDA_VISIBLE_DEVICES}"
  fi
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
    echo "[preflight]   - Single 24 GB card, use:  bash scripts/switch.sh beellama/gemma-dflash" >&2
    echo "[preflight]   - On 2x 24 GB cards, use:  bash scripts/switch.sh vllm/gemma-bf16-mtp" >&2
  fi
  echo "[preflight]   - On a single 24 GB card, start with:  bash scripts/switch.sh beellama/dflash  (single-card default)" >&2
  echo "[preflight]   - For maximum compatibility, use:  bash scripts/switch.sh llamacpp/default" >&2
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

  echo "[preflight] hardware: ${variant:-compose} TP=${tp} requires ${min_gpu_count} GPU(s), >=${min_vram_gb} GB each${sm_label}; ${selected_count} visible GPU(s) detected"
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

# preflight_genesis_pin — warn if scripts/setup.sh's declared GENESIS_PIN
# differs from the on-disk Genesis tree HEAD. This catches the
# "user pulled the repo but didn't re-run setup.sh" failure mode where
# vLLM boots against an outdated Genesis tree (mysterious patch failures
# at runtime). Sourceable; soft-warning only — caller decides whether
# to abort. Returns 0 always; emits a [preflight] WARN line on mismatch.
preflight_genesis_pin() {
  local repo_root="${1:-${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
  local setup_script="${repo_root}/scripts/setup.sh"
  local genesis_dir="${repo_root}/models/qwen3.6-27b/vllm/patches/genesis"

  # If setup.sh isn't here we're in a weird state — skip silently.
  [[ -f "$setup_script" ]] || return 0
  # If Genesis hasn't been cloned yet, this isn't a mismatch — it's a
  # missing-setup case. Skip; setup.sh will handle it on first run.
  [[ -d "${genesis_dir}/.git" ]] || return 0

  # Parse `GENESIS_PIN="${GENESIS_PIN:-<default>}"` to extract the default.
  local declared_pin
  declared_pin=$(grep -E '^GENESIS_PIN=' "$setup_script" 2>/dev/null | head -1 \
    | sed -E 's/.*:-([^}]+)\}.*/\1/; t; s/.*=//' \
    | tr -d '"' | tr -d "'")
  [[ -z "$declared_pin" ]] && return 0

  # Get on-disk HEAD short SHA (matches setup.sh's `git rev-parse --short HEAD`).
  local ondisk_pin
  ondisk_pin=$(cd "$genesis_dir" && git rev-parse --short HEAD 2>/dev/null)
  [[ -z "$ondisk_pin" ]] && return 0

  # Compare. setup.sh declares short-form pins (e.g. 2db18df); on-disk
  # short SHA from git rev-parse --short matches that form. If declared
  # pin is full-length, take its prefix matching ondisk's length.
  local declared_short="${declared_pin:0:${#ondisk_pin}}"

  if [[ "$declared_short" != "$ondisk_pin" ]]; then
    echo "[preflight] WARN:  Genesis tree out of sync with setup.sh's declared pin." >&2
    echo "[preflight]          declared (scripts/setup.sh): ${declared_pin}" >&2
    echo "[preflight]          on-disk (genesis/.git HEAD): ${ondisk_pin}" >&2
    echo "[preflight]        This usually means you pulled latest club-3090 but" >&2
    echo "[preflight]        didn't re-run setup.sh. vLLM may boot against an" >&2
    echo "[preflight]        outdated Genesis tree, causing mysterious patch" >&2
    echo "[preflight]        failures at runtime (see #32 for an example)." >&2
    echo "[preflight]        Fix:  bash scripts/setup.sh qwen3.6-27b" >&2
  fi
  return 0
}

# preflight_repo_drift — warn if local HEAD is behind origin/master.
# Catches the most common stale-setup pattern: user cloned weeks ago, master
# has moved (Genesis pin bumps, compose changes, vendored patch updates),
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
  echo "[preflight]        Master may have new configs, patches, or Genesis pin bumps." >&2
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
  if [[ -n "${WEIGHT_FILES:-}" ]]; then
    printf 'hf download %s %s --local-dir %s/%s' \
      "$WEIGHT_REPO" "$WEIGHT_FILES" "$model_dir_expr" "$WEIGHT_SUBDIR"
  else
    printf 'hf download %s --local-dir %s/%s' \
      "$WEIGHT_REPO" "$model_dir_expr" "$WEIGHT_SUBDIR"
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
      echo "[preflight]     note: ${WEIGHT_MANUAL_NOTE}" >&2
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
    done < <(grep -hoE -- '(^|[[:space:]])(-m|--model)[[:space:]]+/models/[^[:space:]]+' "${compose_files[@]}" \
      | awk '{print $NF}' || true)

    # Speculative drafter: beellama --spec-draft-model, llama.cpp -md/--model-draft.
    # A missing drafter GGUF otherwise surfaces only as a cryptic in-container
    # "failed to open GGUF file" crash (see #288 beellama onboarding reports).
    while IFS= read -r token; do
      path="$(_preflight_compose_path_default "$token")"
      [[ -n "$path" ]] && draft_paths+=("$path")
    done < <(grep -hoE -- '(^|[[:space:]])(--spec-draft-model|--model-draft|-md)[[:space:]]+/models/[^[:space:]]+' "${compose_files[@]}" \
      | awk '{print $NF}' || true)

    while IFS= read -r token; do
      path="$(_preflight_compose_path_default "$token")"
      [[ -n "$path" ]] && mmproj_paths+=("$path")
    done < <(grep -hoE -- '(^|[[:space:]])--mmproj[[:space:]]+/models/[^[:space:]]+' "${compose_files[@]}" \
      | awk '{print $NF}' || true)

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
      fi
    done
    for path in "${draft_paths[@]}"; do
      if [[ ! -f "${model_dir}/${path}" ]]; then
        missing+=("${model_dir}/${path} (speculative drafter GGUF)")
      fi
    done
    for path in "${mmproj_paths[@]}"; do
      if [[ ! -f "${model_dir}/${path}" ]]; then
        missing+=("${model_dir}/${path} (vision projector)")
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
      fi
    done < <(grep -hoE '\$\{MODEL_DIR[^}]*\}/[^"[:space:]]+' "${compose_files[@]}" || true)
  fi

  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi

  echo "[preflight] ERROR: compose '$compose_file' expects model files that aren't on host." >&2
  for item in "${missing[@]}"; do
    echo "[preflight]   missing: ${item}" >&2
  done
  echo "[preflight]" >&2
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
# request 404'd ("The model `qwen3.6-27b-autoround` does not exist"). llama.cpp
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
