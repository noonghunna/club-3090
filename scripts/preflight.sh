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
#
# Style: each function prints one or more "[preflight] ..." lines.
# Hard failures get a one-line ERROR + a "Fix:" hint.

# Avoid double-sourcing.
[[ -n "${_PREFLIGHT_LOADED:-}" ]] && return 0
_PREFLIGHT_LOADED=1

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
  # nvidia-container-toolkit check — needed for docker GPU access.
  if ! docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
    echo "[preflight] WARN:  Docker doesn't list the 'nvidia' runtime. If 'docker compose up' fails" >&2
    echo "                   with 'unknown runtime' or 'could not select device driver', install:" >&2
    echo "                   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/" >&2
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

preflight_running() {
  command -v docker >/dev/null 2>&1 || return 0
  local running
  running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^(vllm-qwen36-27b|llama-cpp-qwen36-27b)' || true)
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
