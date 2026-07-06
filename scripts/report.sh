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
#   bash scripts/report.sh --stress          # adds verify-stress.sh 7/7 output (~5-10 min)
#   bash scripts/report.sh --soak            # adds SOAK_MODE=continuous summary (~25 min) — catches Cliff 2b
#   bash scripts/report.sh --bench           # adds bench.sh output (~3 min)
#   bash scripts/report.sh --agentic         # adds bench-agentic.sh curve-shape output (~8 min estimate)
#   bash scripts/report.sh --full            # ALL five: verify + stress + soak + bench + agentic (~43 min estimate, the canonical "everything" pass for cross-rig contributions)
#   bash scripts/report.sh --studio          # adds AI Studio container log tails (ComfyUI + director + …) — for image/video/audio generation bugs (~2 sec)
#   bash scripts/report.sh --no-redact       # disable path/host/user redaction
#   bash scripts/report.sh --container NAME  # override container auto-detection
#   bash scripts/report.sh --full-calibration  # kv-calc matrix for ALL models (default: only the running model; skipped on llama.cpp/ik_llama)
#   bash scripts/report.sh > my-rig.md       # capture for paste
#
# Why --soak is its own flag:
#   verify-full + verify-stress + bench all PASS on configs that FAIL the
#   multi-turn continuous soak (Cliff 2b at ~25K accumulated tokens). Until
#   the upstream fix lands, soak is the only test that catches the agentic-
#   workload failure mode. See docs/CLIFFS.md.
#
# By default, paths under user homes, hostnames, usernames, and HF tokens are
# redacted. Use --no-redact for internal sharing only.

set -uo pipefail

DO_VERIFY=0
DO_STRESS=0
DO_SOAK=0
DO_BENCH=0
DO_AGENTIC=0
DO_STUDIO=0
REDACT=1
CONTAINER=""
# KV-calc calibration is scoped to the running model by default (#168). Set to 1
# (flag or REPORT_FULL_CALIBRATION=1) to emit the full catalog-wide matrix.
FULL_CALIBRATION="${REPORT_FULL_CALIBRATION:-0}"

print_help() {
  sed -n '2,/^set/p' "$0" | sed 's/^# \?//' | head -n -1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify) DO_VERIFY=1; shift ;;
    --stress) DO_STRESS=1; shift ;;
    --soak) DO_SOAK=1; shift ;;
    --bench) DO_BENCH=1; shift ;;
    --agentic) DO_AGENTIC=1; shift ;;
    --studio) DO_STUDIO=1; shift ;;
    --full) DO_VERIFY=1; DO_STRESS=1; DO_SOAK=1; DO_BENCH=1; DO_AGENTIC=1; shift ;;
    --no-redact) REDACT=0; shift ;;
    --container) CONTAINER="${2:-}"; shift 2 ;;
    --full-calibration) FULL_CALIBRATION=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; echo "Try: bash scripts/report.sh --help" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# KV-calc calibration helpers (engine/model detection + per-model filter, #168).
source "$REPO_ROOT/scripts/lib/report_calib.sh"
# shellcheck source=lib/p2p-state.sh
source "$REPO_ROOT/scripts/lib/p2p-state.sh"

# Pick up a saved MODEL_DIR (and other config) from the repo .env — same as
# launch.sh / switch.sh, and what setup.sh writes there. An explicit exported
# MODEL_DIR still wins. This makes the Disk section report the user's real
# models path instead of falling back to the hardcoded mount below.
if [[ -z "${MODEL_DIR:-}" && -f "${REPO_ROOT}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
fi

HOST_SHORT="$(hostname -s 2>/dev/null || echo unknown)"
USER_NAME="${USER:-$(whoami 2>/dev/null || echo unknown)}"

redact() {
  if [[ $REDACT -eq 1 ]]; then
    # Mask the literal MODEL_DIR value first (if exported) so an arbitrary models
    # path — /data/..., /srv/... — is caught before the prefix rules below.
    { if [[ -n "${MODEL_DIR:-}" ]]; then sed -e "s|${MODEL_DIR}|<MODEL_DIR>|g"; else cat; fi; } | sed \
      -e "s|/home/${USER_NAME}|~|g" \
      -e "s|/root|~|g" \
      -e "s|${HOST_SHORT}|<HOST>|g" \
      -e "s|${USER_NAME}|<USER>|g" \
      -e 's|HF_TOKEN=[^ "]*|HF_TOKEN=<REDACTED>|g' \
      -e 's|HUGGING_FACE_HUB_TOKEN=[^ "]*|HUGGING_FACE_HUB_TOKEN=<REDACTED>|g' \
      -e 's|api_key=[^ "]*|api_key=<REDACTED>|gi' \
      -e 's|hf_[A-Za-z0-9]\{30,\}|hf_<REDACTED>|g' \
      -e 's|/opt/ai|<STACK_ROOT>|g' \
      -e 's|/mnt/[a-z]/Users/[^ /]*|/mnt/<DRIVE>/Users/<REDACTED>|g' \
      -e 's|/mnt/models|<MODELS>|g'
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
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,vbios_version,persistence_mode,power.limit,power.default_limit,power.max_limit,power.draw,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
      --format=csv,noheader 2>/dev/null \
      | while IFS=, read -r idx name memtotal driver vbios persistence pwr_limit pwr_default pwr_max pwr_draw bus_id pcie_gen_cur pcie_gen_max pcie_width_cur pcie_width_max; do
          # trim leading spaces from CSV fields
          idx="${idx# }"; name="${name# }"; memtotal="${memtotal# }"
          driver="${driver# }"; vbios="${vbios# }"; persistence="${persistence# }"
          pwr_limit="${pwr_limit# }"; pwr_default="${pwr_default# }"
          pwr_max="${pwr_max# }"; pwr_draw="${pwr_draw# }"
          bus_id="${bus_id# }"; pcie_gen_cur="${pcie_gen_cur# }"; pcie_gen_max="${pcie_gen_max# }"
          pcie_width_cur="${pcie_width_cur# }"; pcie_width_max="${pcie_width_max# }"

          # Flag if user has capped below default
          power_note=""
          pwr_limit_w="${pwr_limit% W}"; pwr_limit_w="${pwr_limit_w%.*}"
          pwr_default_w="${pwr_default% W}"; pwr_default_w="${pwr_default_w%.*}"
          if [[ "$pwr_limit_w" =~ ^[0-9]+$ ]] && [[ "$pwr_default_w" =~ ^[0-9]+$ ]]; then
            if [[ "$pwr_limit_w" -lt "$pwr_default_w" ]]; then
              power_note=" ⚠ user-capped below default"
            elif [[ "$pwr_limit_w" -gt "$pwr_default_w" ]]; then
              power_note=" (overclocked above default)"
            fi
          fi

          # Flag if PCIe lane width is below max — that's hardware-level (slot
          # has fewer lanes wired, riser cables, BIOS bifurcation, etc.) and
          # affects model load speed + per-card all-reduce bandwidth.
          # NOTE: pcie.link.gen.current drops to Gen 1 at idle for power
          # saving — that's normal, not a degradation. Re-check under load if
          # you want the actual negotiated gen. Width is hardware-fixed.
          pcie_note=""
          if [[ -n "$pcie_width_cur" && -n "$pcie_width_max" && "$pcie_width_cur" != "$pcie_width_max" ]]; then
            pcie_note=" ⚠ slot is narrower than GPU capability — affects load + all-reduce bandwidth"
          fi

          echo "- **GPU $idx:** $name | $memtotal | driver $driver | VBIOS $vbios | persistence=$persistence"
          echo "  - **Power:** limit=${pwr_limit} (default=${pwr_default}, max=${pwr_max}) | current_draw=${pwr_draw}${power_note}"
          echo "  - **PCIe:** x${pcie_width_cur} lanes negotiated (GPU max x${pcie_width_max}, Gen up to ${pcie_gen_max}) | bus ${bus_id}${pcie_note}"
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

  # lspci-based PCIe/P2P detail. nvidia-smi reports negotiated gen/width but
  # cannot show trained link state vs capability side-by-side, ACS state on the
  # upstream bridge, or the real PCIe topology tree — the three things that
  # actually decide whether GPU↔GPU P2P engages (see issues #137, #351).
  subsection "PCIe / P2P detail (lspci)"
  if ! have lspci; then
    # Fallback: nvidia-smi topo -p2p doesn't need pciutils and shows P2P capability
    if have nvidia-smi && nvidia-smi topo -p2p rw >/dev/null 2>&1; then
      echo "_lspci not available (pciutils not installed) — showing P2P capability matrix instead._"
      echo
      nvidia-smi topo -p2p rw | redact
    else
      echo "_lspci not available (pciutils not installed) — skipping PCIe/P2P detail._"
    fi
  else
    # sudo lspci -vvv is needed for full capability blocks (ACS lives in the
    # extended config space, root-only). Degrade gracefully if sudo is
    # unavailable / non-interactive — non-sudo lspci still shows LnkSta.
    LSPCI_CMD=(lspci)
    SUDO_NOTE=""
    if [[ $EUID -ne 0 ]]; then
      if have sudo && sudo -n true 2>/dev/null; then
        LSPCI_CMD=(sudo lspci)
      else
        SUDO_NOTE="_Note: sudo unavailable/non-interactive — running lspci without root; ACS capability lines may be incomplete (LnkSta still accurate)._"
      fi
    fi

    {
      [[ -n "$SUDO_NOTE" ]] && { echo "$SUDO_NOTE"; echo; }

      echo "# lspci -t  (PCIe topology tree)"
      lspci -t 2>&1
      echo

      # Per NVIDIA VGA / 3D-controller function: trained link state vs
      # capability + ACS state. Filter to the four load-bearing lines only —
      # never dump the full -vvv block (keeps the report compact + redaction-safe).
      # ACS (ACSCap/ACSCtl) lives on the UPSTREAM PCIe port, not the GPU
      # endpoint — and ACS-redirect on that bridge is exactly what blocks P2P
      # (issues #137, #351) — so for each GPU we also dump its upstream bridge.
      dump_func() {
        local slot="$1" label="$2"
        echo "# lspci -vvv -s ${slot}  (${label}: LnkCap/LnkSta/ACSCap/ACSCtl)"
        "${LSPCI_CMD[@]}" -vvv -s "$slot" 2>/dev/null \
          | grep -E '^[[:space:]]*(LnkCap|LnkSta|ACSCap|ACSCtl):' \
          || echo "  (no matching LnkCap/LnkSta/ACSCap/ACSCtl lines)"
        echo
      }
      while read -r slot _; do
        [[ -z "$slot" ]] && continue
        dump_func "$slot" "GPU function"
        # Resolve the upstream bridge via sysfs (../.. of the device node).
        bridge=""
        if [[ -e "/sys/bus/pci/devices/${slot}" ]]; then
          bridge="$(basename "$(readlink -f "/sys/bus/pci/devices/${slot}/../" 2>/dev/null)" 2>/dev/null)"
        fi
        if [[ "$bridge" =~ ^[0-9a-fA-F]{4}: ]]; then
          dump_func "$bridge" "upstream bridge of ${slot}"
        else
          echo "  (could not resolve upstream bridge for ${slot} — ACS state for P2P may be elsewhere in the tree)"
          echo
        fi
      done < <(lspci -D 2>/dev/null | grep -iE 'VGA compatible controller.*NVIDIA|3D controller.*NVIDIA')

      echo "# lspci -nnk | grep -A3 -i nvidia  (driver binding + device IDs)"
      lspci -nnk 2>/dev/null | grep -A3 -i nvidia 2>/dev/null \
        || echo "  (no NVIDIA functions found)"
    } 2>&1 | redact | details "lspci PCIe/P2P detail (LnkSta / ACS / topology)"
  fi

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
    # Check if a club-3090 container is running (lightweight — full detection is later)
    # NB: top-level (not in a function) — plain assignment, not `local`.
    our_container=""
    if have docker && docker info >/dev/null 2>&1; then
      our_container=$(docker ps --format '{{.Names}}' --filter 'name=vllm-' --filter 'name=llama-cpp-' --filter 'name=beellama-' --filter 'name=club3090-' --filter 'name=ik-llama-' 2>/dev/null | head -1)
    fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | while IFS=, read -r idx used; do
          idx="${idx# }"; used="${used# }"
          if [[ "$used" =~ ^[0-9]+$ ]] && [[ "$used" -gt 100 ]]; then
            if [[ -n "$our_container" ]]; then
              echo "- **GPU $idx idle VRAM:** ${used} MiB (held by running \`${our_container}\`)"
            else
              echo "- **GPU $idx idle VRAM:** ${used} MiB ⚠ something is using this GPU (display, browser, container)"
            fi
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
    # Prefer `git describe` for a human-readable version (e.g. v0.6.2-3-ge299e70,
    # "3 commits past v0.6.2 at SHA e299e70"). Falls back to raw SHA if no tags
    # are reachable (shallow clone, fresh repo).
    version=$(git describe --tags --always --dirty 2>/dev/null)
    commit=$(git rev-parse --short HEAD 2>/dev/null)
    branch=$(git branch --show-current 2>/dev/null)
    echo "- **club-3090:** \`${version:-${commit:-unknown}}\` (branch: \`${branch:-detached}\`, SHA \`${commit:-unknown}\`)"
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
# Profile state
# ---------------------------------------------------------------------------

if [[ -x scripts/lib/profiles/estate_cli.py || -f scripts/lib/profiles/estate_cli.py ]]; then
  python3 scripts/lib/profiles/estate_cli.py report-state 2>&1 | redact || true
fi

# ---------------------------------------------------------------------------
# KV math calibration
# ---------------------------------------------------------------------------
# When a user files a VRAM-OOM or context-ceiling bug, the maintainer's first
# question is "does kv-calc still agree with measured reality?" — a calibration
# failure means the projection model has drifted from the actual VRAM cost of a
# compose, so any "predicted PASS" verdict can't be trusted. Surface the
# verdict line + any FAIL rows here so a triage reply can immediately see
# whether to trust kv-calc projections for this user's config.

# Engine + model detection for kv-calc scoping (#168). Resolve the active
# container (explicit --container wins; else first running club-3090 container),
# then map it to a kv-calc engine family + model id via scripts/lib/report_calib.sh.
_calib_container="${CONTAINER:-}"
if [[ -z "$_calib_container" ]] && have docker && docker info >/dev/null 2>&1; then
  _calib_container=$(docker ps --format '{{.Names}}' --filter 'name=vllm-' --filter 'name=llama-cpp-' --filter 'name=beellama-' --filter 'name=club3090-' --filter 'name=ik-llama-' 2>/dev/null | head -1)
fi
CALIB_ENGINE_KIND="${ENGINE_KIND:-$(calib_engine_for_container "$_calib_container")}"
CALIB_MODEL_ID="$(calib_model_for_container "$_calib_container")"

if have python3 && [[ -f tools/kv-calc.py ]]; then
  section "KV math calibration"
  # kv-calc is vLLM-memory-model-coupled; skip on the ggml engines (llama.cpp + ik_llama).
  if [[ "$CALIB_ENGINE_KIND" == "llamacpp" ]]; then
    echo "- _kv-calc calibration is vLLM-specific — skipped on the llama.cpp / ik_llama engine (ggml uses a different allocator)._"
  elif ! python3 -c 'import yaml' 2>/dev/null; then
    # Item 1: graceful-degrade when PyYAML is missing
    echo "- _kv-calc calibration skipped — PyYAML not installed (\`pip install pyyaml\`)._"
  else
    # #168: scope to the running model by default; --full-calibration (or
    # REPORT_FULL_CALIBRATION=1) restores the catalog-wide matrix. Falls back to
    # the full matrix when the model can't be resolved.
    calib_scope=""
    if [[ "$FULL_CALIBRATION" != "1" && -n "$CALIB_MODEL_ID" ]]; then
      calib_scope="$CALIB_MODEL_ID"
      echo "- _Scoped to the running model \`${CALIB_MODEL_ID}\` — pass \`--full-calibration\` for all calibrated models._"
    fi
    calib_output=$(python3 tools/kv-calc.py --calibration 2>&1 | calib_filter_model_section "$calib_scope" || true)
    overall=$(echo "$calib_output" | grep -E '^Overall:' | head -1)
    fail_rows=$(echo "$calib_output" | grep -E '\bFAIL\b' || true)
    {
      if [[ -n "$overall" ]]; then
        echo "- ${overall}"
      else
        echo "- _kv-calc --calibration produced no Overall line; see output below._"
      fi
      if [[ -n "$fail_rows" ]]; then
        echo "- ⚠ Failing rows:"
        echo '```'
        echo "$fail_rows"
        echo '```'
        echo "- Math model is mis-calibrated against measured reality for the rows above. Any kv-calc projection on this checkout should be treated as suspect until the calibration anchors / formulas are reconciled."
      else
        echo "- No FAIL rows. kv-calc projections should agree with measured VRAM within the ±1.5 GB error band."
      fi
    } | redact
    echo "$calib_output" | redact | details "Full kv-calc --calibration output"
  fi
fi

# ---------------------------------------------------------------------------
# Quality tooling (benchlocal-cli + sandboxes)
# ---------------------------------------------------------------------------
# Triage for "my quality run skipped packs / scored weird": is benchlocal-cli
# installed, how fresh, are the sandbox images built, and do they PREDATE the
# CLI (the rebuilt-CLI-stale-sandboxes incident class)? All best-effort — a
# rig without any of this still produces a report.

section "Quality tooling (benchlocal-cli + sandboxes)"
{
  bl_bin=$(command -v benchlocal-cli 2>/dev/null || true)
  if [[ -z "$bl_bin" ]]; then
    echo "- **benchlocal-cli:** not installed (quality-test.sh needs it — \`pip install git+https://github.com/noonghunna/benchlocal-cli.git\`)"
  else
    bl_mtime=$(stat -c %Y "$bl_bin" 2>/dev/null || echo 0)
    bl_when=$([[ "$bl_mtime" -gt 0 ]] && date -d "@${bl_mtime}" +%F 2>/dev/null || echo "unknown")
    # Version via the CLI's own interpreter (works for pip-from-git AND
    # editable-checkout installs; console-script shebang points at the env).
    bl_py=$(head -1 "$bl_bin" 2>/dev/null | sed 's/^#!//')
    bl_ver=$([[ -x "$bl_py" ]] && "$bl_py" -c 'import importlib.metadata as m; print(m.version("benchlocal-cli"))' 2>/dev/null || true)
    echo "- **benchlocal-cli:** \`${bl_bin}\` (version: \`${bl_ver:-unknown}\`, installed/updated: ${bl_when})"
    if have docker && docker info >/dev/null 2>&1; then
      stale_any=0
      echo "- **Sandbox images** (needed by the --full sandboxed packs):"
      for img in benchlocal-sandbox-bugfind benchlocal-sandbox-cli benchlocal-sandbox-hermes benchlocal-sandbox-aider-polyglot; do
        created=$(docker image inspect "${img}:latest" --format '{{.Created}}' 2>/dev/null || true)
        if [[ -z "$created" ]]; then
          echo "  - \`${img}\`: ✗ not built"
          continue
        fi
        cdate=$(date -d "$created" +%F 2>/dev/null || echo "$created")
        cepoch=$(date -d "$created" +%s 2>/dev/null || echo 0)
        mark=""
        if [[ "$cepoch" -gt 0 && "$bl_mtime" -gt 0 && "$cepoch" -lt "$bl_mtime" ]]; then
          mark="  ⚠ OLDER than the installed CLI — rebuild if the update touched sandbox sources"
          stale_any=1
        fi
        echo "  - \`${img}\`: built ${cdate}${mark}"
      done
      [[ "$stale_any" == "1" ]] && echo "- **Rebuild:** \`bash <benchlocal-cli-checkout>/tools/build-sandboxes.sh\` (heuristic — an unrelated reinstall also trips it)"
    else
      echo "- **Sandbox images:** docker unavailable — cannot inspect (sandboxed packs need Docker)"
    fi
  fi
  latest_q=$(ls -t results/quality/quality-*.json 2>/dev/null | head -1)
  if [[ -n "$latest_q" ]]; then
    echo "- **Latest quality result:** \`${latest_q}\` ($(date -d "@$(stat -c %Y "$latest_q")" +%F 2>/dev/null || echo '?'))"
  else
    echo "- **Latest quality result:** none found under results/quality/"
  fi
} | redact

# ---------------------------------------------------------------------------
# Active container
# ---------------------------------------------------------------------------

section "Active container"
# Engine-agnostic auto-detection: try vllm-* first (most common on this stack),
# fall back to llama-cpp-* (the alternate engine we ship). User can override
# with CONTAINER=... env var for non-standard naming (microk8s deployments,
# host engine builds via CONTAINER=none, etc.).
if [[ -z "$CONTAINER" ]] && have docker && docker info >/dev/null 2>&1; then
  CONTAINER=$(docker ps --format '{{.Names}}' --filter 'name=vllm-qwen36' 2>/dev/null | head -1)
  [[ -z "$CONTAINER" ]] && CONTAINER=$(docker ps --format '{{.Names}}' --filter 'name=vllm-' 2>/dev/null | head -1)
  [[ -z "$CONTAINER" ]] && CONTAINER=$(docker ps --format '{{.Names}}' --filter 'name=llama-cpp-' 2>/dev/null | head -1)
  [[ -z "$CONTAINER" ]] && CONTAINER=$(docker ps --format '{{.Names}}' --filter 'name=beellama-' 2>/dev/null | head -1)
  [[ -z "$CONTAINER" ]] && CONTAINER=$(docker ps --format '{{.Names}}' --filter 'name=club3090-' 2>/dev/null | head -1)
fi

# Engine class — drives which probes run inside the container body. Inferred
# from container name; user can override with ENGINE_KIND=vllm|llamacpp env var.
case "${ENGINE_KIND:-}" in
  vllm|llamacpp|unknown) ;;  # respect user override
  *)
    case "$CONTAINER" in
      vllm-*)      ENGINE_KIND="vllm" ;;
      llama-cpp-*) ENGINE_KIND="llamacpp" ;;
      club3090-*)
        container_image=$(docker ps --filter "name=$CONTAINER" --format '{{.Image}}' 2>/dev/null | head -1)
        case "$container_image" in
          *llama.cpp*|*llama-cpp*) ENGINE_KIND="llamacpp" ;;
          *vllm*)                  ENGINE_KIND="vllm" ;;
          *)                       ENGINE_KIND="unknown" ;;
        esac ;;
      *)           ENGINE_KIND="unknown" ;;
    esac ;;
esac

if [[ -z "$CONTAINER" ]]; then
  echo "_No vLLM, llama.cpp, or estate container running. Start one with \`bash scripts/launch.sh\` and re-run for the full report._"
else
  {
    status=$(docker ps --filter "name=$CONTAINER" --format '{{.Status}}' 2>/dev/null | head -1)
    ports=$(docker ps --filter "name=$CONTAINER" --format '{{.Ports}}' 2>/dev/null | head -1)
    image=$(docker ps --filter "name=$CONTAINER" --format '{{.Image}}' 2>/dev/null | head -1)
    # Digest + OCI labels (load-bearing when tag is rolling, e.g. llama.cpp
    # `:server-cuda`). Without these, a bug report can't reproduce the bytes.
    image_digest=$(docker inspect "$CONTAINER" --format '{{.Image}}' 2>/dev/null | head -1)
    image_revision=$(docker inspect "$CONTAINER" --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' 2>/dev/null)
    image_version=$(docker inspect "$CONTAINER" --format '{{ index .Config.Labels "org.opencontainers.image.version" }}' 2>/dev/null)
    image_source=$(docker inspect "$CONTAINER" --format '{{ index .Config.Labels "org.opencontainers.image.source" }}' 2>/dev/null)
    echo "- **Name:** \`$CONTAINER\`"
    echo "- **Engine:** \`${ENGINE_KIND}\`"
    echo "- **Status:** ${status:-unknown}"
    echo "- **Ports:** ${ports:-unknown}"
    echo "- **Image:** \`${image:-unknown}\`"
    [[ -n "$image_digest" ]]   && echo "- **Image digest:** \`${image_digest}\`"
    [[ -n "$image_version"  && "$image_version"  != "<no value>" ]] && echo "- **Build tag (OCI version):** \`${image_version}\`"
    [[ -n "$image_revision" && "$image_revision" != "<no value>" ]] && echo "- **Upstream commit (OCI revision):** \`${image_revision}\`"
    [[ -n "$image_source"   && "$image_source"   != "<no value>" ]] && echo "- **Upstream source:** ${image_source}"
  } | redact

  # Engine-specific probes from this point. vLLM container has Python +
  # PyTorch + Genesis markers; llama.cpp container ships a stripped C++
  # binary with no Python — different probe set.

  # Engine-specific subsections. vLLM container has Python + PyTorch + Genesis
  # markers; llama.cpp container ships a stripped C++ binary with no Python
  # exec available — different probe set.

  if [[ "$ENGINE_KIND" == "llamacpp" ]]; then
    # ---- llama.cpp probe set ----
    subsection "Container engine state (llama.cpp)"
    {
      # llama-server prints its version + build flags on startup. Grep the
      # boot log for the version banner instead of trying to docker exec
      # (the llama-cpp image doesn't ship interactive shell utilities).
      llama_version=$(docker logs "$CONTAINER" 2>&1 | grep -E '^build_info:|^version:|^system_info:' | head -3)
      if [[ -n "$llama_version" ]]; then
        echo "**llama-server version + build:**"
        echo '```'
        echo "$llama_version"
        echo '```'
        echo
      fi

      # Loaded model + ctx + KV type — surfaces model identity from boot log.
      model_loaded=$(docker logs "$CONTAINER" 2>&1 | grep -E 'load_model:|llama_model_load_from_file_impl:|llama_kv_cache_init:|llama_init_from_model:' | head -8)
      if [[ -n "$model_loaded" ]]; then
        echo "**Model load + KV cache init:**"
        echo '```'
        echo "$model_loaded"
        echo '```'
        echo
      fi

      # llama.cpp doesn't have Genesis / vLLM SpecDecoding metrics. Skip
      # those grep patterns. Capture warnings/errors only.
      boot_errors=$(docker logs "$CONTAINER" 2>&1 | grep -iE '^(warn|error|fatal|abort)|panic|core dumped' | tail -5)
      if [[ -n "$boot_errors" ]]; then
        echo "**Recent warnings/errors (last 5):**"
        echo '```'
        echo "$boot_errors"
        echo '```'
      fi
    } | redact

    subsection "Full boot log (first 200 lines)"
    docker logs "$CONTAINER" 2>&1 | head -200 | redact | details "First 200 lines of docker logs"

  else
  # ---- vLLM probe set (default for engine=vllm or unknown) ----
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
    # Interconnect / P2P ENGAGEMENT — the runtime truth. The GPU "Topology" /
    # "PCIe / P2P detail" sections above report P2P *capability* (can it?); this
    # reports whether P2P is actually ON for the running serving container.
    # detect_nvlink.sh emits an [nvlink] decision trail at boot stating the
    # resolved NCCL_P2P_LEVEL + custom-all-reduce state. Grep the WHOLE log (not
    # head -200) so a late line on a 3-4 GPU boot isn't missed, and fall back to
    # the live container env. ALWAYS prints something so a reviewer never has to
    # guess whether P2P was engaged (the gap that forced asks on #446 / #488).
    nvlink_boot=$(docker logs "$CONTAINER" 2>&1 | grep -E '\[nvlink\]' | head -8)
    p2p_env=$(docker exec "$CONTAINER" env 2>/dev/null | grep -E '^(NCCL_P2P|NVLINK_MODE|NCCL_CUMEM)=' | sort)
    echo "**Interconnect / P2P engagement:**"
    if [[ -n "$nvlink_boot" || -n "$p2p_env" ]]; then
      echo '```'
      [[ -n "$nvlink_boot" ]] && echo "$nvlink_boot"
      [[ -n "$p2p_env" ]] && { echo "# resolved container env:"; echo "$p2p_env"; }
      echo '```'
    else
      echo "_No \`[nvlink]\` boot line or NCCL_P2P/NVLINK_MODE env found — P2P engagement undetermined (single-GPU, a non-NCCL engine like llama.cpp, or an entrypoint predating detect_nvlink.sh)._"
    fi
    # Cross-referenced VERDICT (capability x engagement — the #488/#158 matrix).
    # Silent on single-GPU / no-capability rigs so the OK/WARN/INFO line is
    # always signal, never boilerplate.
    _p2p_verdict_line="$(p2p_verdict "$(p2p_gpu_count)" "$(p2p_host_capability)" \
      "$(printf '%s\n%s' "$nvlink_boot" "$p2p_env" | p2p_classify_engagement)")"
    [[ -n "$_p2p_verdict_line" ]] && { echo; echo "**Interconnect verdict:** ${_p2p_verdict_line}"; }
    echo

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
  fi  # end of vLLM/llamacpp engine branch
fi  # end of "if no container running"

# ---------------------------------------------------------------------------
# Recent failed boot attempts
# ---------------------------------------------------------------------------
# Capture exited vLLM/llama.cpp containers from the last 24h. Most valuable
# diagnostic data for boot-failure scenarios — without this, contributors hit
# "no container running" and have to manually paste docker logs ad-hoc.
# Engine-agnostic: matches both vllm-* and llama-cpp-* container patterns.

section "Recent failed boot attempts"
if ! have docker; then
  echo "_docker not available — skipping._"
elif ! docker info >/dev/null 2>&1; then
  echo "_docker daemon unreachable — skipping._"
else
  # Get exited containers matching club-3090 engine patterns. `docker ps -a`
  # without a time filter; we'll filter to last 24h via the FinishedAt field.
  exited_lines=$(docker ps -a \
    --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.ID}}' \
    --filter 'status=exited' 2>/dev/null \
    | grep -E '^(vllm-|llama-cpp-)' || true)

  if [[ -z "$exited_lines" ]]; then
    echo "_No recently-exited vLLM or llama.cpp containers found._"
  else
    found_recent=0
    while IFS=$'\t' read -r ex_name ex_image ex_status ex_id; do
      [[ -z "$ex_name" ]] && continue
      # Cutoff: last 24h. docker inspect gives ISO-8601 FinishedAt.
      finished_at=$(docker inspect "$ex_id" --format '{{.State.FinishedAt}}' 2>/dev/null || echo "")
      exit_code=$(docker inspect "$ex_id" --format '{{.State.ExitCode}}' 2>/dev/null || echo "?")
      [[ -z "$finished_at" ]] && continue

      # Skip containers that exited >24h ago (epoch comparison)
      finished_epoch=$(date -d "$finished_at" +%s 2>/dev/null || echo 0)
      cutoff_epoch=$(date -d '24 hours ago' +%s 2>/dev/null || echo 0)
      [[ "$finished_epoch" -lt "$cutoff_epoch" ]] && continue

      found_recent=1
      relative_when=$(date -d "$finished_at" '+%Y-%m-%dT%H:%M:%SZ (%s seconds ago)' 2>/dev/null || echo "$finished_at")
      # Format relative_when nicely: how many minutes ago?
      mins_ago=$(( ($(date +%s) - finished_epoch) / 60 ))
      if [[ $mins_ago -lt 60 ]]; then
        relative_label="${mins_ago} min ago"
      else
        hrs_ago=$(( mins_ago / 60 ))
        rem_mins=$(( mins_ago % 60 ))
        relative_label="${hrs_ago}h ${rem_mins}min ago"
      fi

      subsection "\`$ex_name\` — exited $relative_label (code $exit_code)"
      {
        echo "- **Name:** \`$ex_name\`"
        echo "- **Image:** \`$ex_image\`"
        echo "- **Status:** $ex_status"
        echo "- **Exit code:** $exit_code"
        echo "- **Finished at:** $finished_at"
      } | redact

      docker logs --tail 80 "$ex_id" 2>&1 | redact | details "Last 80 log lines from \`$ex_name\`"
    done <<< "$exited_lines"

    if [[ "$found_recent" == "0" ]]; then
      echo "_Exited vLLM/llama.cpp containers exist but all >24h old — likely not relevant to current investigation._"
    fi
  fi
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
# Optional: verify-stress
# ---------------------------------------------------------------------------

if [[ $DO_STRESS -eq 1 ]]; then
  section "verify-stress.sh output"
  if [[ -f scripts/verify-stress.sh ]]; then
    bash scripts/verify-stress.sh 2>&1 | redact | details "verify-stress output (7 boundary checks incl. Cliff 2 needle recall)"
  else
    echo "_scripts/verify-stress.sh not found_"
  fi
fi

# ---------------------------------------------------------------------------
# Optional: soak-continuous (catches Cliff 2b — the only test that does)
# ---------------------------------------------------------------------------

if [[ $DO_SOAK -eq 1 ]]; then
  section "soak-test.sh (SOAK_MODE=continuous) output"
  if [[ -f scripts/soak-test.sh ]]; then
    soak_run_dir="results/report-soak-$(date +%Y%m%d-%H%M%S)"
    SOAK_MODE=continuous SOAK_SESSIONS=5 SOAK_TURNS=5 SOAK_OUTPUT="$soak_run_dir" \
      SOAK_TIMEOUT_S="${SOAK_TIMEOUT_S:-1800}" \
      bash scripts/soak-test.sh 2>&1 | redact | details "soak-test stdout (5-session × 5-turn ramping conversation, ~25 min)"
    if [[ -f "$soak_run_dir/summary.md" ]]; then
      echo
      echo "**Soak summary** (\`$soak_run_dir/summary.md\`):"
      echo
      redact < "$soak_run_dir/summary.md"
    else
      echo "_soak summary.md not produced — check stdout above_"
    fi
  else
    echo "_scripts/soak-test.sh not found_"
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
# Soak-not-run reminder — fired when --bench (or partial) was used without
# --soak/--full. Cross-rig bench rows want the soak verdict; without it we
# can't say if Cliff 2b is open on this rig class.
# ---------------------------------------------------------------------------

if [[ $DO_BENCH -eq 1 && $DO_SOAK -eq 0 ]]; then
  section "Soak status"
  cat <<'EOF'
> ⚠️ **Soak: not included in this report.**
>
> This run used `--bench` (or `--verify`/`--stress` only) — the soak-continuous
> test was skipped. Cross-rig bench contributions on club-3090 want the soak
> verdict so we can tell whether Cliff 2b is open on your rig class.
>
> Run soak separately and paste its output as a follow-up:
>
> ```bash
> bash scripts/soak-test.sh --continuous   # auto-detects endpoint + container
> ```
>
> Takes ~25 min. The `[soak]` summary block (verdict, max VRAM growth, silent-empty %, TPS retention) is what ends up in the bench-template's "Soak verdict" dropdown. See [docs/CLIFFS.md](https://github.com/noonghunna/club-3090/blob/master/docs/CLIFFS.md) for context.
EOF
fi

if [[ $DO_AGENTIC -eq 1 ]]; then
  section "bench-agentic.sh output"
  if [[ -f scripts/bench-agentic.sh ]]; then
    SESSIONS=1 bash scripts/bench-agentic.sh 2>&1 | redact | details "bench-agentic output (1 session x 12 default turns, curve-shape estimate; ~8 min estimate)"
  else
    echo "_scripts/bench-agentic.sh not found_"
  fi
fi

# ---------------------------------------------------------------------------
# AI Studio container logs (--studio) — opt-in: the ComfyUI / director / orchestrator
# log tails that diagnose an image/video/audio GENERATION failure (e.g. a "ComfyUI
# generation error" in a lane). Off by default — verbose + only relevant for studio bugs.
# ---------------------------------------------------------------------------
if [[ $DO_STUDIO -eq 1 ]]; then
  section "AI Studio logs (--studio)"
  echo "_Container log tails for image/video/audio generation bugs. ComfyUI carries the workflow"
  echo "execution trace (the actual generation error). Redacted; pass \`--no-redact\` for full paths._"
  _studio_found=0
  # ComfyUI first (the generation engine — longest tail), then the studio sidecars.
  for c in comfyui studio-director studio-orchestrator studio-image-shim studio-tts studio-step-voice studio-gallery; do
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$c"; then
      _studio_found=1
      _tail=200; [[ "$c" == comfyui ]] && _tail=400
      _running=$(docker ps --filter "name=^${c}$" --format '{{.Status}}' 2>/dev/null | head -1)
      # strip ANSI colour codes (ComfyUI logs are coloured) so the pasted block reads cleanly
      docker logs --tail "$_tail" "$c" 2>&1 | sed -E 's/\x1b\[[0-9;]*[mK]//g' | redact \
        | details "$c — ${_running:-not running} (last $_tail lines)"
    fi
  done
  [[ $_studio_found -eq 0 ]] && echo "_No AI Studio containers found. Bring the studio up (\`gpu-mode ai-studio\` or \`bash scripts/setup-ai-studio.sh\`), reproduce the failure, then re-run with \`--studio\`._"
fi

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

cat <<'EOF'

---

_Generated by `bash scripts/report.sh`. Flags: `--verify` (verify-full), `--stress` (verify-stress 7/7 incl. Cliff 2 needles), `--soak` (SOAK_MODE=continuous, catches Cliff 2b), `--bench` (canonical TPS), `--agentic` (multi-turn TTFT/decode curve-shape, ~8 min estimate), `--studio` (AI Studio / ComfyUI container log tails — for generation bugs), `--full` (all five, ~43 min estimate). Use `--no-redact` to disable redaction (internal sharing only)._
EOF
