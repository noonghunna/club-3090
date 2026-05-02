#!/usr/bin/env bash
# scripts/report.sh — paste-ready triage report for club-3090
#
# Run when filing a bug report, sharing cross-rig benchmark data, or replying
# to a triage thread. Captures hardware, OS, GPU, container runtime, stack
# version, and active container state in markdown ready to paste into a GitHub
# issue or discussion.
#
# Usage:
#   bash scripts/report.sh                   # default: hardware + stack + boot log highlights (~2 sec)
#   bash scripts/report.sh --verify          # adds verify-full.sh output (~1-2 min)
#   bash scripts/report.sh --bench           # adds bench.sh output (~3 min)
#   bash scripts/report.sh --full            # both verify + bench (~5 min)
#   bash scripts/report.sh --no-redact       # disable path/host/user redaction
#   bash scripts/report.sh --container NAME  # override container auto-detection
#   bash scripts/report.sh > my-rig.md       # capture for paste
#
# By default, paths under user homes, hostnames, usernames, and HF tokens are
# redacted. Use --no-redact for internal sharing only.

set -uo pipefail

DO_VERIFY=0
DO_BENCH=0
REDACT=1
CONTAINER=""

print_help() {
  sed -n '2,/^set/p' "$0" | sed 's/^# \?//' | head -n -1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify) DO_VERIFY=1; shift ;;
    --bench) DO_BENCH=1; shift ;;
    --full) DO_VERIFY=1; DO_BENCH=1; shift ;;
    --no-redact) REDACT=0; shift ;;
    --container) CONTAINER="${2:-}"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; echo "Try: bash scripts/report.sh --help" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOST_SHORT="$(hostname -s 2>/dev/null || echo unknown)"
USER_NAME="${USER:-$(whoami 2>/dev/null || echo unknown)}"

redact() {
  if [[ $REDACT -eq 1 ]]; then
    sed \
      -e "s|/home/${USER_NAME}|~|g" \
      -e "s|/root|~|g" \
      -e "s|${HOST_SHORT}|<HOST>|g" \
      -e "s|${USER_NAME}|<USER>|g" \
      -e 's|HF_TOKEN=[^ "]*|HF_TOKEN=<REDACTED>|g' \
      -e 's|HUGGING_FACE_HUB_TOKEN=[^ "]*|HUGGING_FACE_HUB_TOKEN=<REDACTED>|g' \
      -e 's|api_key=[^ "]*|api_key=<REDACTED>|gi' \
      -e 's|hf_[A-Za-z0-9]\{30,\}|hf_<REDACTED>|g'
  else
    cat
  fi
}

section() { printf '\n## %s\n\n' "$1"; }
subsection() { printf '\n### %s\n\n' "$1"; }

details() {
  local summary="$1"
  printf '<details><summary>%s</summary>\n\n```\n' "$summary"
  cat
  printf '```\n\n</details>\n'
}

have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

cat <<EOF
# club-3090 rig report

Generated: $(date -u +'%Y-%m-%d %H:%M:%S UTC')
EOF

if [[ $REDACT -eq 1 ]]; then
  printf '\n_Redacted output (paths, host, user, tokens). Re-run with `--no-redact` for full data._\n'
fi

# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

section "System"
{
  os_name="unknown"
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    os_name="${PRETTY_NAME:-${NAME:-unknown}}"
  fi
  echo "- **OS:** $os_name"
  echo "- **Kernel:** $(uname -r)"

  # Environment detection
  env_kind="bare metal"
  if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
    env_kind="WSL2"
    if grep -qE 'WSL2' /proc/version 2>/dev/null; then
      env_kind="WSL2 (kernel reports WSL2)"
    fi
  elif have systemd-detect-virt && [[ "$(systemd-detect-virt 2>/dev/null)" != "none" ]]; then
    env_kind="$(systemd-detect-virt 2>/dev/null) (virtualized)"
  elif [[ -r /.dockerenv ]]; then
    env_kind="inside-container (unusual for this script)"
  fi
  echo "- **Environment:** $env_kind"

  echo "- **Locale:** ${LANG:-unset}"
  echo "- **Timezone:** $(date +%Z)"
  echo "- **Uptime:** $(uptime -p 2>/dev/null || echo unknown)"
} | redact

# ---------------------------------------------------------------------------
# CPU + RAM
# ---------------------------------------------------------------------------

section "CPU + RAM"
{
  if have lscpu; then
    cpu_model=$(lscpu 2>/dev/null | awk -F: '/Model name/ {sub(/^ */, "", $2); print $2; exit}')
    cpu_cores=$(lscpu 2>/dev/null | awk -F: '/^CPU\(s\):/ {gsub(/ /, "", $2); print $2; exit}')
    echo "- **CPU:** ${cpu_model:-unknown} (${cpu_cores:-?} threads)"
  else
    echo "- **CPU:** lscpu not available"
  fi

  if have free; then
    ram_total=$(free -h 2>/dev/null | awk '/^Mem:/ {print $2}')
    ram_avail=$(free -h 2>/dev/null | awk '/^Mem:/ {print $7}')
    echo "- **RAM:** ${ram_total} total, ${ram_avail} available"
    swap_total=$(free -h 2>/dev/null | awk '/^Swap:/ {print $2}')
    [[ "$swap_total" != "0B" && -n "$swap_total" ]] && echo "- **Swap:** $swap_total"
  fi
} | redact

# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

section "Disk"
{
  declare -a checked_paths=()
  add_disk_row() {
    local p="$1"
    [[ -z "$p" || ! -d "$p" ]] && return
    for seen in "${checked_paths[@]:-}"; do
      [[ "$seen" == "$p" ]] && return
    done
    checked_paths+=("$p")
    local fs avail
    fs=$(df -T "$p" 2>/dev/null | awk 'NR==2 {print $2}')
    avail=$(df -h "$p" 2>/dev/null | awk 'NR==2 {print $4}')
    echo "- **$p:** ${avail:-?} available, ${fs:-?} filesystem"
  }

  add_disk_row "${MODEL_DIR:-}"
  add_disk_row "$REPO_ROOT/models-cache"
  add_disk_row "/mnt/models/huggingface"

  if have docker && docker info >/dev/null 2>&1; then
    docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
    [[ -n "$docker_root" ]] && add_disk_row "$docker_root"
  fi
} | redact

# ---------------------------------------------------------------------------
# GPU hardware
# ---------------------------------------------------------------------------

section "GPU hardware"
if ! have nvidia-smi; then
  echo "_nvidia-smi not available — no NVIDIA GPU detected or driver not installed_"
else
  {
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,vbios_version,persistence_mode,power.limit,power.default_limit,power.max_limit,power.draw \
      --format=csv,noheader 2>/dev/null \
      | while IFS=, read -r idx name memtotal driver vbios persistence pwr_limit pwr_default pwr_max pwr_draw; do
          # trim leading spaces from CSV fields
          idx="${idx# }"; name="${name# }"; memtotal="${memtotal# }"
          driver="${driver# }"; vbios="${vbios# }"; persistence="${persistence# }"
          pwr_limit="${pwr_limit# }"; pwr_default="${pwr_default# }"
          pwr_max="${pwr_max# }"; pwr_draw="${pwr_draw# }"

          # Flag if user has capped below default
          power_note=""
          # Strip " W" suffix and convert to int for comparison
          pwr_limit_w="${pwr_limit% W}"; pwr_limit_w="${pwr_limit_w%.*}"
          pwr_default_w="${pwr_default% W}"; pwr_default_w="${pwr_default_w%.*}"
          if [[ "$pwr_limit_w" =~ ^[0-9]+$ ]] && [[ "$pwr_default_w" =~ ^[0-9]+$ ]]; then
            if [[ "$pwr_limit_w" -lt "$pwr_default_w" ]]; then
              power_note=" ⚠ user-capped below default"
            elif [[ "$pwr_limit_w" -gt "$pwr_default_w" ]]; then
              power_note=" (overclocked above default)"
            fi
          fi

          echo "- **GPU $idx:** $name | $memtotal | driver $driver | VBIOS $vbios | persistence=$persistence"
          echo "  - **Power:** limit=${pwr_limit} (default=${pwr_default}, max=${pwr_max}) | current_draw=${pwr_draw}${power_note}"
        done

    cuda_ver=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1 | awk '{print $3}')
    [[ -n "$cuda_ver" ]] && echo "- **CUDA Runtime (per driver):** $cuda_ver"

    # Persistence mode + ECC summary
    ecc_status=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    [[ -n "$ecc_status" ]] && echo "- **ECC mode:** $ecc_status (3090s don't have ECC; expect N/A)"
  } | redact

  subsection "NVLink"
  if nvidia-smi nvlink --status -i 0 2>/dev/null | grep -qE 'Link [0-9]+:'; then
    nvidia-smi nvlink --status 2>&1 | redact | details "NVLink link status"
  else
    echo "_No NVLink detected (PCIe-only)_"
  fi

  subsection "Topology"
  nvidia-smi topo -m 2>&1 | redact | details "PCIe / GPU topology matrix"

  subsection "Full nvidia-smi"
  nvidia-smi 2>&1 | redact | details "Full nvidia-smi output"
fi

# ---------------------------------------------------------------------------
# Display / desktop state
# ---------------------------------------------------------------------------

section "Display / desktop state"
{
  if [[ -n "${DISPLAY:-}" ]]; then
    echo "- **\$DISPLAY:** ${DISPLAY} (X11 / Wayland session present)"
  else
    echo "- **\$DISPLAY:** unset (headless)"
  fi
  [[ -n "${WAYLAND_DISPLAY:-}" ]] && echo "- **\$WAYLAND_DISPLAY:** ${WAYLAND_DISPLAY}"

  compositor=""
  for proc in Xorg Xwayland weston gnome-shell kwin sway hyprland mutter; do
    if pgrep -x "$proc" >/dev/null 2>&1; then
      compositor="$compositor $proc"
    fi
  done
  if [[ -n "$compositor" ]]; then
    echo "- **Display processes running:**$compositor"
  else
    echo "- **Display processes running:** none detected"
  fi

  if have nvidia-smi; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | while IFS=, read -r idx used; do
          idx="${idx# }"; used="${used# }"
          if [[ "$used" =~ ^[0-9]+$ ]] && [[ "$used" -gt 100 ]]; then
            echo "- **GPU $idx idle VRAM:** ${used} MiB ⚠ something is using this GPU (display, browser, container)"
          else
            echo "- **GPU $idx idle VRAM:** ${used} MiB ✓"
          fi
        done
  fi
} | redact

# ---------------------------------------------------------------------------
# Container runtime
# ---------------------------------------------------------------------------

section "Container runtime"
{
  if have docker; then
    if docker info >/dev/null 2>&1; then
      docker_ver=$(docker version --format '{{.Server.Version}}' 2>/dev/null)
      echo "- **Docker:** ${docker_ver:-unknown}"

      if docker compose version >/dev/null 2>&1; then
        compose_ver=$(docker compose version --short 2>/dev/null)
        echo "- **docker compose (v2):** ${compose_ver:-unknown}"
      elif have docker-compose; then
        compose_ver=$(docker-compose version --short 2>/dev/null)
        echo "- **docker-compose (v1):** ${compose_ver:-unknown}"
      fi

      if have nvidia-ctk; then
        nvct_ver=$(nvidia-ctk --version 2>&1 | head -1 | awk '{print $NF}')
        echo "- **NVIDIA Container Toolkit:** ${nvct_ver:-unknown}"
      elif have nvidia-container-toolkit; then
        nvct_ver=$(nvidia-container-toolkit --version 2>&1 | head -1 | awk '{print $NF}')
        echo "- **NVIDIA Container Toolkit:** ${nvct_ver:-unknown}"
      fi
    else
      echo "- **Docker:** installed but daemon not accessible"
    fi
  else
    echo "- **Docker:** not installed"
  fi
} | redact

# ---------------------------------------------------------------------------
# Stack version
# ---------------------------------------------------------------------------

section "Stack version"
{
  if [[ -d .git ]]; then
    commit=$(git rev-parse --short HEAD 2>/dev/null)
    branch=$(git branch --show-current 2>/dev/null)
    echo "- **club-3090:** \`${commit:-unknown}\` (branch: \`${branch:-detached}\`)"
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
      echo "- **Working tree:** ⚠ has uncommitted changes (run \`git status\` to inspect)"
    fi
  else
    echo "- **club-3090:** not a git repo"
  fi

  if [[ -f scripts/setup.sh ]]; then
    # Parse `GENESIS_PIN="${GENESIS_PIN:-<default>}"` — extract just the default value
    genesis_pin=$(grep -E '^GENESIS_PIN=' scripts/setup.sh 2>/dev/null | head -1 \
      | sed -E 's/.*:-([^}]+)\}.*/\1/; t; s/.*=//' \
      | tr -d '"' | tr -d "'")
    [[ -n "$genesis_pin" ]] && echo "- **GENESIS_PIN default:** \`$genesis_pin\` (per scripts/setup.sh)"
    # Override from env if set
    [[ -n "${GENESIS_PIN:-}" ]] && echo "- **GENESIS_PIN env override:** \`$GENESIS_PIN\`"
  fi

  if have docker && docker info >/dev/null 2>&1; then
    cached=$(docker images vllm/vllm-openai --format '{{.Tag}} {{.Digest}} {{.CreatedSince}}' 2>/dev/null | head -3)
    if [[ -n "$cached" ]]; then
      echo "- **Cached vLLM images:**"
      echo "$cached" | while read -r tag digest age rest; do
        echo "  - tag \`$tag\` digest \`$digest\` ($age $rest)"
      done
    fi
  fi
} | redact

# ---------------------------------------------------------------------------
# Active container
# ---------------------------------------------------------------------------

section "Active container"
if [[ -z "$CONTAINER" ]] && have docker && docker info >/dev/null 2>&1; then
  CONTAINER=$(docker ps --format '{{.Names}}' --filter 'name=vllm-qwen36' 2>/dev/null | head -1)
  [[ -z "$CONTAINER" ]] && CONTAINER=$(docker ps --format '{{.Names}}' --filter 'name=vllm-' 2>/dev/null | head -1)
fi

if [[ -z "$CONTAINER" ]]; then
  echo "_No vLLM container running. Start one with \`bash scripts/launch.sh\` and re-run for the full report._"
else
  {
    status=$(docker ps --filter "name=$CONTAINER" --format '{{.Status}}' 2>/dev/null | head -1)
    ports=$(docker ps --filter "name=$CONTAINER" --format '{{.Ports}}' 2>/dev/null | head -1)
    image=$(docker ps --filter "name=$CONTAINER" --format '{{.Image}}' 2>/dev/null | head -1)
    echo "- **Name:** \`$CONTAINER\`"
    echo "- **Status:** ${status:-unknown}"
    echo "- **Ports:** ${ports:-unknown}"
    echo "- **Image:** \`${image:-unknown}\`"
  } | redact

  subsection "Container Python / CUDA versions"
  {
    # vLLM version + Torch CUDA build vs host driver mismatch is one of the
    # rare failure modes that image SHA pinning doesn't catch. Quick docker
    # exec to surface what the container actually sees.
    py_versions=$(docker exec "$CONTAINER" python3 -c \
      'import torch, sys; print(f"torch={torch.__version__} torch_cuda_build={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")' \
      2>&1)
    if [[ -n "$py_versions" ]] && [[ "$py_versions" != *"Error"* ]] && [[ "$py_versions" != *"error"* ]]; then
      echo "- **PyTorch:** \`${py_versions}\`"
    else
      echo "- **PyTorch:** (could not query — \`docker exec\` failed or torch not importable)"
    fi

    vllm_ver=$(docker exec "$CONTAINER" python3 -c 'import vllm; print(vllm.__version__)' 2>&1)
    if [[ -n "$vllm_ver" ]] && [[ "$vllm_ver" != *"Error"* ]] && [[ "$vllm_ver" != *"error"* ]]; then
      echo "- **vLLM:** \`${vllm_ver}\`"
    else
      echo "- **vLLM:** (could not query)"
    fi

    # Container's view of the GPUs — should match host driver, but if NVIDIA
    # Container Toolkit is mis-configured this surfaces the mismatch.
    cuda_in_container=$(docker exec "$CONTAINER" nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>&1 | head -4)
    if [[ -n "$cuda_in_container" ]] && [[ "$cuda_in_container" != *"Error"* ]] && [[ "$cuda_in_container" != *"command not found"* ]]; then
      echo "- **nvidia-smi inside container:**"
      echo '  ```'
      echo "$cuda_in_container" | sed 's/^/  /'
      echo '  ```'
    fi
  } | redact

  subsection "Boot log highlights"
  {
    genesis_results=$(docker logs "$CONTAINER" 2>&1 | grep -E '\[INFO:genesis\.apply_all\] (Genesis|✅) Results' | tail -1)
    if [[ -n "$genesis_results" ]]; then
      echo "**Genesis patches applied:**"
      echo '```'
      echo "$genesis_results" | sed 's/.*Genesis Results: /Genesis Results: /'
      echo '```'
      echo
    fi

    sidecar_status=$(docker logs "$CONTAINER" 2>&1 | grep -E '^\[(tolist_cudagraph_fix|inputs_embeds_optional|workspace_lock_disable|pn25_genesis_register_fix|pn30_dst_shaped_temp_fix|fa_max_seqlen_clamp|pn12_ffn_pool_anchor|pn12_compile_safe_custom_op)\]' | head -10)
    if [[ -n "$sidecar_status" ]]; then
      echo "**Local sidecar application:**"
      echo '```'
      echo "$sidecar_status"
      echo '```'
      echo
    fi

    kv_pool=$(docker logs "$CONTAINER" 2>&1 | grep -E 'Available KV cache memory|GPU KV cache size:|Maximum concurrency for' | tail -3)
    if [[ -n "$kv_pool" ]]; then
      echo "**KV pool sizing:**"
      echo '```'
      echo "$kv_pool"
      echo '```'
      echo
    fi

    # Engine config — the line containing "non-default args" or "Initializing a V1 LLM engine"
    # captures every important CLI flag (max_model_len, mem_util, kv dtype, spec config, etc.)
    engine_config=$(docker logs "$CONTAINER" 2>&1 | grep -E 'non-default args:|Initializing a V1 LLM engine' | head -2)
    if [[ -n "$engine_config" ]]; then
      echo "**Engine config (CLI flags + engine init):**"
      echo '```'
      echo "$engine_config"
      echo '```'
      echo
    fi

    boot_errors=$(docker logs "$CONTAINER" 2>&1 | grep -E '^(WARNING|ERROR|CRITICAL)' | tail -5)
    if [[ -n "$boot_errors" ]]; then
      echo "**Recent warnings/errors (last 5):**"
      echo '```'
      echo "$boot_errors"
      echo '```'
    fi
  } | redact

  subsection "Full boot log (first 200 lines)"
  docker logs "$CONTAINER" 2>&1 | head -200 | redact | details "First 200 lines of docker logs"
fi

# ---------------------------------------------------------------------------
# Optional: verify-full
# ---------------------------------------------------------------------------

if [[ $DO_VERIFY -eq 1 ]]; then
  section "verify-full.sh output"
  if [[ -f scripts/verify-full.sh ]]; then
    bash scripts/verify-full.sh 2>&1 | redact | details "verify-full output"
  else
    echo "_scripts/verify-full.sh not found_"
  fi
fi

# ---------------------------------------------------------------------------
# Optional: bench
# ---------------------------------------------------------------------------

if [[ $DO_BENCH -eq 1 ]]; then
  section "bench.sh output"
  if [[ -f scripts/bench.sh ]]; then
    bash scripts/bench.sh 2>&1 | redact | details "bench output (3 warmups + 5 measured per prompt)"
  else
    echo "_scripts/bench.sh not found_"
  fi
fi

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

cat <<'EOF'

---

_Generated by `bash scripts/report.sh`. Add `--verify` for verify-full output, `--bench` for canonical TPS bench, `--full` for both. Use `--no-redact` to disable redaction (internal sharing only)._
EOF
