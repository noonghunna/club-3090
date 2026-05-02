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
#   preflight_model_dir <root> <subdir>
#                             — model directory exists and contains config/weights
#   preflight_gpu_idle        — warn if GPUs have significant VRAM already in use
#   preflight_running         — warn if a club-3090 container is already up
#
# Style: each function prints one or more "[preflight] ..." lines.
# Hard failures get a one-line ERROR + a "Fix:" hint.
#
# Env:
#   CLUB3090_NVIDIA_SMI_SUDO=1  Use `sudo -n nvidia-smi` for hosts where
#                               nvidia-smi access is gated behind sudo.

# Avoid double-sourcing.
[[ -n "${_PREFLIGHT_LOADED:-}" ]] && return 0
_PREFLIGHT_LOADED=1

preflight_nvidia_smi() {
  if [[ "${CLUB3090_NVIDIA_SMI_SUDO:-0}" == "1" ]]; then
    sudo -n nvidia-smi "$@"
  else
    nvidia-smi "$@"
  fi
}

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
  gpu_lines=$(preflight_nvidia_smi -L 2>/dev/null || true)
  local gpu_count
  gpu_count=$(echo "$gpu_lines" | grep -c '^GPU ' || true)
  if [[ "$gpu_count" -lt "$min_count" ]]; then
    echo "[preflight] ERROR: needs ${min_count} GPU(s), found ${gpu_count}." >&2
    if [[ "$gpu_count" -eq 0 ]]; then
      echo "            Fix: confirm 'nvidia-smi' lists your GPU(s); check driver/PCIe wiring." >&2
      echo "                 If this host requires sudo for nvidia-smi, set CLUB3090_NVIDIA_SMI_SUDO=1." >&2
    else
      echo "            Fix: pick a variant matching the detected GPU count, or install/wire the missing GPU(s)." >&2
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

preflight_model_dir() {
  local model_root="$1"
  local model_subdir="$2"
  local model_path="${model_root%/}/${model_subdir}"

  if [[ ! -d "$model_path" ]]; then
    echo "[preflight] ERROR: model directory missing: ${model_path}" >&2
    echo "            Fix: set MODEL_DIR to the parent directory containing ${model_subdir}," >&2
    echo "                 or run: MODEL_DIR=${model_root} bash scripts/setup.sh qwen3.6-27b" >&2
    return 1
  fi
  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "[preflight] ERROR: ${model_path}/config.json missing." >&2
    echo "            Fix: re-run setup, or point MODEL_DIR at the complete HF model cache parent." >&2
    return 1
  fi
  local has_weights=0
  local weight
  for weight in "${model_path}"/*.safetensors; do
    if [[ -f "$weight" ]]; then
      has_weights=1
      break
    fi
  done
  if [[ "$has_weights" != "1" ]]; then
    echo "[preflight] ERROR: no safetensors weights found in ${model_path}." >&2
    echo "            Fix: re-run setup, or point MODEL_DIR at the complete HF model cache parent." >&2
    return 1
  fi
  echo "[preflight] model:   ${model_path}"
  return 0
}

preflight_gpu_idle() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local mem_used_lines
  mem_used_lines=$(preflight_nvidia_smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
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
