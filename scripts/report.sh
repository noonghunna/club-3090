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
# Exit codes (#813, committed to on #619):
#   0  every stage that ran passed — or no stage was requested
#   2  ADVISORY-only: a check flagged headroom/risk rather than incorrectness
#      (today: verify-stress's agent-safety VRAM margin at the context ceiling —
#      recall is correct there, the margin for sustained agent load is not)
#   1  hard failure: a stage failed a correctness check, could not run, or the
#      engine died mid-run and the remaining stages were skipped
# The "Check summary" section names the failing check, so it is identifiable
# from the exit path without reading every inner block.
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

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

DO_VERIFY=0
DO_STRESS=0
DO_SOAK=0
DO_BENCH=0
DO_AGENTIC=0
DO_STUDIO=0
REDACT=1
CONTAINER="${CONTAINER:-}"
# KV-calc calibration is scoped to the running model by default (#168). Set to 1
# (flag or REPORT_FULL_CALIBRATION=1) to emit the full catalog-wide matrix.
FULL_CALIBRATION="${REPORT_FULL_CALIBRATION:-0}"

print_help() {
  sed -n '2,/^set/p' "$0" | sed 's/^# \?//' | head -n -1
}

# Preserve the invocation so the "re-run with sudo" hint can echo back the SAME
# flags, rather than a bare command the user has to reconstruct.
REPORT_ARGS="$*"

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

# Root-gated captures — warn UP FRONT, not per-section.
#
# Some of the most decision-relevant hardware facts are root-only: DIMM
# population + configured memory speed (dmidecode), and ACS state on the
# upstream bridge (lspci extended config space). Without them a report looks
# complete while missing the fields that actually explain CPU-offload
# throughput and whether GPU↔GPU P2P can engage.
#
# This is deterministic at start-up, so say it BEFORE the sections rather than
# leaving a reader to notice three "needs root" lines scattered through a long
# paste — which they will not, because reports get truncated from the bottom.
if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
  _root_gated=()
  command -v dmidecode >/dev/null 2>&1 && _root_gated+=("DIMM count / size / configured memory speed")
  command -v lspci >/dev/null 2>&1     && _root_gated+=("PCIe ACS capability state (LnkSta is still accurate)")
  if [[ ${#_root_gated[@]} -gt 0 ]]; then
    printf '\n> ⚠️ **Incomplete — running without root.** These fields were skipped:\n'
    for _g in "${_root_gated[@]}"; do printf '> - %s\n' "$_g"; done
    printf '>\n> For a complete report re-run with sudo:\n>\n> ```bash\n> sudo bash scripts/report.sh %s\n> ```\n' "${REPORT_ARGS:-}"
    printf '>\n> Worth doing before filing a CPU-offload issue: memory channels and speed are the\n> variables that set throughput on those configs, and they cannot be read without root.\n'
  fi
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

  # Motherboard / BIOS — from sysfs DMI (readable WITHOUT root; serials/UUIDs
  # are root-only in sysfs so they are never exposed here). #690.
  _dmi() {
    local v; v="$(cat "/sys/class/dmi/id/$1" 2>/dev/null)"
    v="${v//$'\n'/ }"; v="${v#"${v%%[![:space:]]*}"}"; v="${v%"${v##*[![:space:]]}"}"
    # drop common OEM placeholder junk so it degrades to "(not exposed)"
    case "$v" in
      "To Be Filled By O.E.M."|"To be filled by O.E.M."|"Default string"|      "System manufacturer"|"System Product Name"|"System Version"|      "Not Applicable"|"None"|"N/A"|"Unknown"|"0.0.0"|"") v="" ;;
    esac
    printf '%s' "$v"
  }
  _mb_vendor="$(_dmi board_vendor)"; _mb_name="$(_dmi board_name)"
  _mb_product="$(_dmi product_name)"; _mb_bios="$(_dmi bios_version)"; _mb_biosdate="$(_dmi bios_date)"
  _mb="$_mb_vendor${_mb_vendor:+${_mb_name:+ }}$_mb_name"
  if [[ -n "$_mb" ]]; then
    [[ -n "$_mb_product" && "$_mb_product" != "$_mb" ]] && _mb="$_mb  (system: $_mb_product)"
    echo "- **Motherboard:** $_mb"
  elif [[ -n "$_mb_product" ]]; then
    echo "- **Motherboard:** (board DMI not exposed; system: $_mb_product)"
  else
    echo "- **Motherboard:** (not exposed)"
  fi
  [[ -n "$_mb_bios" ]] && echo "- **BIOS:** ${_mb_bios}${_mb_biosdate:+ ($_mb_biosdate)}"

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

# _report_mem_bandwidth — MEASURED memory bandwidth, because the rated figure is
# blind to the failure that matters.
#
# ⚠️ Firmware's "Configured Memory Speed" does NOT move when a channel is lost.
# This rig read 3200 MT/s both before and after a DIMM went missing, while real
# STREAM Triad fell ~100 -> 59.9 GB/s (2026-08-07). The spec number is exactly
# the one that cannot detect the problem; the measured one is the only witness.
# Our own learnings already say "the bandwidth number must be MEASURED, not spec"
# — this implements it.
#
# Cheap and honest: ~1.5 GB, a few hundred ms, skipped when RAM is tight or no
# compiler exists, and it SAYS SO rather than omitting the line (a missing row
# reads as "nothing to report"). Disable with REPORT_NO_BANDWIDTH=1.
_report_mem_bandwidth() {
  [[ "${REPORT_NO_BANDWIDTH:-0}" == "1" ]] && return 0
  local cc=""
  for c in cc gcc clang; do command -v "$c" >/dev/null 2>&1 && { cc="$c"; break; }; done
  if [[ -z "$cc" ]]; then
    echo "- **Memory bandwidth:** not measured (no C compiler) — install \`gcc\` for a STREAM Triad figure"
    return 0
  fi
  local free_mb; free_mb=$(free -m 2>/dev/null | awk '/^Mem:/{print $7}')
  if [[ -n "$free_mb" && "$free_mb" -lt 4096 ]]; then
    echo "- **Memory bandwidth:** not measured (only ${free_mb} MiB available; probe needs ~1.5 GB)"
    return 0
  fi
  local d; d="$(mktemp -d 2>/dev/null)" || return 0
  cat > "$d/t.c" <<'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#define N (64L*1024*1024)          /* 3 x 512 MiB */
static double *a,*b,*c; static int NT;
static void *w(void *p){ long id=(long)p, lo=N*id/NT, hi=N*(id+1)/NT;
  for(long i=lo;i<hi;i++) c[i]=a[i]+3.0*b[i]; return 0; }
int main(int argc,char**argv){
  NT=atoi(argv[1]); if(NT<1)NT=1;
  a=malloc(N*8); b=malloc(N*8); c=malloc(N*8);
  if(!a||!b||!c){ printf("0\n"); return 1; }
  for(long i=0;i<N;i++){a[i]=1.0;b[i]=2.0;c[i]=0.0;}
  double best=0; pthread_t th[512];
  for(int r=0;r<3;r++){
    struct timespec s,e; clock_gettime(CLOCK_MONOTONIC,&s);
    for(long i=0;i<NT;i++) pthread_create(&th[i],0,w,(void*)i);
    for(long i=0;i<NT;i++) pthread_join(th[i],0);
    clock_gettime(CLOCK_MONOTONIC,&e);
    double dt=(e.tv_sec-s.tv_sec)+(e.tv_nsec-s.tv_nsec)/1e9;
    double gbs=(3.0*N*8)/dt/1e9; if(gbs>best) best=gbs;
  }
  printf("%.1f\n",best); return 0;
}
CEOF
  local gbs=""
  if "$cc" -O2 -o "$d/t" "$d/t.c" -lpthread >/dev/null 2>&1; then
    gbs="$("$d/t" "$(nproc 2>/dev/null || echo 4)" 2>/dev/null)"
  fi
  rm -rf "$d"
  if [[ -n "$gbs" && "$gbs" != "0" ]]; then
    echo "- **Memory bandwidth (measured):** ${gbs} GB/s STREAM Triad — *this*, not the rated MT/s above, is what sets CPU-offload decode throughput"
  else
    echo "- **Memory bandwidth:** probe failed to build/run — rated speed above is NOT a substitute"
  fi
}

# ---------------------------------------------------------------------------
# CPU + RAM
# ---------------------------------------------------------------------------

section "CPU + RAM"
{
  # ⚠️ This section is load-bearing for the CPU-offload slugs. Their throughput
  # is set by the HOST — memory channels, memory speed, physical core count —
  # not by the GPU, which is why those slugs publish no first-party numbers and
  # ask readers for theirs instead. Reporting only "model name + thread count"
  # made that ask unanswerable: physical cores are not derivable from the model
  # string on most vendors, and channels/speed were not collected at all.
  if have lscpu; then
    # LC_ALL=C — lscpu TRANSLATES its field labels, so every lookup below would
    # silently return empty on a non-English locale (same class as #779).
    _lscpu="$(LC_ALL=C lscpu 2>/dev/null)"
    _lscpu_f() { printf '%s\n' "$_lscpu" | grep -m1 -E "^$1:" | sed -E 's/^[^:]*:[[:space:]]*//'; }

    cpu_model="$(_lscpu_f 'Model name')"
    cpu_threads="$(_lscpu_f 'CPU\(s\)')"
    cpu_sockets="$(_lscpu_f 'Socket\(s\)')"
    cpu_percore="$(_lscpu_f 'Core\(s\) per socket')"
    cpu_tpc="$(_lscpu_f 'Thread\(s\) per core')"
    cpu_numa="$(_lscpu_f 'NUMA node\(s\)')"
    cpu_hyp="$(_lscpu_f 'Hypervisor vendor')"

    echo "- **CPU:** ${cpu_model:-unknown}"
    cpu_phys=""
    [[ -n "$cpu_sockets" && -n "$cpu_percore" ]] && cpu_phys=$(( cpu_sockets * cpu_percore ))
    echo "- **CPU topology:** ${cpu_sockets:-?} socket(s) × ${cpu_percore:-?} core(s) = ${cpu_phys:-?} physical, ${cpu_tpc:-?} thread(s)/core → ${cpu_threads:-?} logical"
    [[ -n "$cpu_numa" ]] && echo "- **NUMA nodes:** $cpu_numa"

    # nproc honours cgroup/affinity limits and is what the offload `-t` default
    # halves — so a divergence from lscpu's total changes the thread count the
    # launcher picks, and is worth surfacing rather than silently disagreeing.
    if have nproc; then
      cpu_nproc="$(nproc 2>/dev/null)"
      if [[ -n "$cpu_nproc" && -n "$cpu_threads" && "$cpu_nproc" != "$cpu_threads" ]]; then
        echo "- **nproc:** $cpu_nproc ⚠️ differs from lscpu's $cpu_threads — cgroup/affinity limited; offload \`-t\` derives from this"
      else
        echo "- **nproc:** ${cpu_nproc:-?}"
      fi
    fi
    [[ -n "$cpu_hyp" ]] && echo "- **Virtualized:** yes — $cpu_hyp ($(_lscpu_f 'Virtualization type'))"
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

  # DIMM population + configured speed. Needs root, so it is best-effort — but
  # ALWAYS print a line: an omitted row is indistinguishable from "nothing to
  # report", and a reader must be able to tell "unknown" from "not collected".
  _dmi=""
  if have dmidecode; then
    if [[ $EUID -eq 0 ]]; then
      _dmi="$(dmidecode -t memory 2>/dev/null)"
    elif sudo -n true 2>/dev/null; then
      _dmi="$(sudo -n dmidecode -t memory 2>/dev/null)"
    fi
  fi
  if [[ -n "$_dmi" ]]; then
    printf '%s\n' "$_dmi" | awk '
      BEGIN { RS = ""; FS = "\n" }
      /Memory Device/ {
        slots++; size = ""; spd = ""; ch = ""
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^[[:space:]]*Size:/)                    { s = $i; sub(/^[^:]*:[[:space:]]*/, "", s); size = s }
          if ($i ~ /^[[:space:]]*Configured Memory Speed:/) { s = $i; sub(/^[^:]*:[[:space:]]*/, "", s); spd  = s }
          if ($i ~ /^[[:space:]]*Bank Locator:/)            { s = $i; sub(/^[^:]*:[[:space:]]*/, "", s); ch   = s }
        }
        # ⚠️ CHANNEL identity, not just a slot tally. Slots != channels — boards
        # commonly carry 2 DIMMs per channel, so "7 of 8 slots" can mean four fully
        # populated channels, a completely different bandwidth story. It is CHANNEL
        # population that sets the interleave, and the interleave is what moves GB/s.
        if (ch != "") { if (!(ch in seen_ch)) { seen_ch[ch] = 1; nch++ } }
        if (size != "" && size !~ /No Module/) {
          pop++; sizes[size]++
          if (ch != "") full_ch[ch] = 1
          if (spd != "" && spd !~ /Unknown/) speeds[spd]++
        } else if (ch != "") { empty_list = empty_list (empty_list ? ", " : "") ch }
      }
      END {
        if (slots == 0) { print "- **Memory config:** dmidecode reported no memory devices"; exit }
        for (s in sizes)  desc = desc (desc ? " + " : "") sizes[s] "× " s
        for (s in speeds) sp   = sp   (sp   ? ", "   : "") s
        nfull = 0; for (c in full_ch) nfull++
        printf "- **Memory config:** %d of %d slot(s) populated — %s%s\n", pop, slots, desc,
               (sp != "" ? " @ " sp " (rated — NOT measured; see bandwidth below)" : " (speed not reported by firmware)")
        if (nch > 0)
          printf "- **Memory channels:** %d of %d populated%s\n", nfull, nch,
                 (empty_list != "" ? " — EMPTY: " empty_list : "")
        if (nfull < nch)
          print "  - ⚠️ _An unpopulated channel forces a coarser interleave. Measured on this stack: losing ONE of eight channels cost ~**30% of bandwidth** (~100 → 69.8 GB/s idle; 59.9 under load) — 2.4× worse than the 12.5% a naive share implies._"
      }'
    _report_mem_bandwidth
  elif have dmidecode; then
    echo "- **Memory config:** not collected (needs root) — run \`sudo dmidecode -t memory\` for DIMM count / size / configured speed"
    _report_mem_bandwidth
  else
    echo "- **Memory config:** not collected (\`dmidecode\` not installed)"
  fi

  # Host memory POLICY. Load-bearing for cross-rig comparison of CPU-offload
  # slugs: their experts live in Shmem (CUDA pinned host memory), so
  # `shmem_enabled` — NOT `enabled` — governs whether they get huge pages.
  # ⚠️ Report COVERAGE, not the setting: a knob reading `always` can still yield
  # 0% under fragmentation, and `enabled=always` alone moves AnonHugePages (the
  # heap) while the experts stay on 4 KiB pages. FIRST-TOKEN LATENCY only
  # (measured -56% TTFT when fixed) — never a throughput claim.
  if [[ -r /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/lib/thp.sh" 2>/dev/null || true
    if declare -F thp_setting >/dev/null 2>&1; then
      echo "- **THP policy:** enabled=$(thp_setting enabled) · shmem_enabled=$(thp_setting shmem_enabled) · defrag=$(thp_setting defrag)"
      _sh="$(thp_meminfo_kib Shmem)"; _shp="$(thp_meminfo_kib ShmemHugePages)"
      _anon="$(thp_meminfo_kib AnonHugePages)"; _pmd="$(thp_meminfo_kib ShmemPmdMapped)"
      _cov="$(thp_shmem_coverage_pct)"
      if [[ -n "$_cov" ]]; then
        echo "- **THP coverage (Shmem):** ${_cov}% — ShmemHugePages $(( ${_shp:-0} / 1048576 )) GiB of Shmem $(( ${_sh:-0} / 1048576 )) GiB"
        [[ "$_cov" -lt 50 ]] && echo "  - ⚠️ _Low coverage: CPU-offloaded experts are on 4 KiB pages. Costs first-token latency (~-56% TTFT when fixed); throughput is unaffected._"
      else
        echo "- **THP coverage (Shmem):** n/a — no large Shmem allocation resident (load a model to measure)"
      fi
      echo "- **THP detail:** ShmemPmdMapped ${_pmd:-?} kB · AnonHugePages ${_anon:-?} kB"
      echo "  - _AnonHugePages is the process heap, NOT the experts — it climbs even when the model stays on 4 KiB pages._"
    fi
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
    # "Could P2P be on here, and is it worth it?" — the question a 2-card
    # PCIe rig actually has, which every other line in this report leaves
    # unanswered (the verdict below needs a RUNNING container; this doesn't).
    # Tiered + achievability-gated on purpose: see p2p_opportunity_hint.
    _p2p_count="$(p2p_gpu_count)"
    _p2p_cap="$(p2p_host_capability "$_p2p_count" 2>/dev/null || echo none)"
    if [[ "$_p2p_cap" == "pcie_p2p" ]]; then
      # ⚠️ The hint below deliberately says NOTHING once peer access is already
      # granted ("already covered"). That left a rig with working P2P showing no
      # interconnect status at all — and a reader cannot tell "nothing to say"
      # from "never ran". State it positively instead, with the caveat that
      # matters most here: a grant is not proof.
      echo
      echo "ℹ interconnect: ${_p2p_count} GPUs, no NVLink, but \`topo -p2p\` reports peer access **enabled on every pair** — P2P is granted on this host."
      echo
      echo "  ⚠️ A grant is not proof it works. \`topo -p2p OK\` reflects the driver's *permission*, not a completed transfer: a dual-5090 pair read OK and then hung inside \`ncclCommInitRank\` (#873). Only an actual peer transfer settles it — \`bash scripts/p2p-validate.sh\` runs a real all-reduce, which copy-then-sync benchmarks cannot substitute for. If a multi-GPU engine hangs at init, suspect this first."
    else
      _p2p_hint="$(p2p_opportunity_hint "$_p2p_count" "$_p2p_cap" 2>/dev/null || true)"
      [[ -n "$_p2p_hint" ]] && { echo; echo "$_p2p_hint"; }
    fi
  fi

  subsection "Topology"
  nvidia-smi topo -m 2>&1 | redact | details "PCIe / GPU topology matrix"

  # Peer-access matrix — ALWAYS, not just when lspci is missing.
  #
  # This used to be emitted only as an lspci fallback, so any rig with pciutils
  # installed (i.e. most) filed a report containing no P2P verdict at all — the
  # one command that answers "is peer access on?" was the one we skipped. It is
  # a single cheap call and it is the PRIMARY signal; the lspci ACS/LnkSta dump
  # below is supporting evidence, not a substitute for it.
  if have nvidia-smi && [[ "$(p2p_gpu_count)" -ge 2 ]]; then
    if nvidia-smi topo -p2p rw >/dev/null 2>&1; then
      nvidia-smi topo -p2p rw 2>&1 | redact | details "GPU peer-access matrix (topo -p2p rw)"
    fi
  fi

  # lspci-based PCIe/P2P detail. nvidia-smi reports negotiated gen/width but
  # cannot show trained link state vs capability side-by-side, ACS state on the
  # upstream bridge, or the real PCIe topology tree — the three things that
  # actually decide whether GPU↔GPU P2P engages (see issues #137, #351).
  subsection "PCIe / P2P detail (lspci)"
  if ! have lspci; then
    # The peer-access matrix is emitted unconditionally under Topology above, so
    # there is nothing to fall back to here any more — just say what is missing.
    echo "_lspci not available (pciutils not installed) — skipping ACS / LnkSta detail. The peer-access matrix under **Topology** above is unaffected; install \`pciutils\` for the link-state and ACS evidence._"
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
    # PRECISE source — the metadata version is frozen at install time and
    # fixes are pushed without bumping it, so it alone can't identify the
    # code. pip records the truth in direct_url.json: a git install carries
    # the exact commit; an editable install carries the checkout dir → git
    # describe (path itself withheld from the public report).
    bl_src=$([[ -x "$bl_py" ]] && "$bl_py" - <<'PYEOF' 2>/dev/null
import importlib.metadata as m, json, subprocess
try:
    raw = m.distribution("benchlocal-cli").read_text("direct_url.json") or ""
    d = json.loads(raw)
except Exception:
    d = {}
vcs = (d.get("vcs_info") or {}).get("commit_id")
if vcs:
    print(f"git@{vcs[:9]}")
elif (d.get("dir_info") or {}).get("editable") and str(d.get("url", "")).startswith("file://"):
    path = d["url"][7:]
    try:
        desc = subprocess.run(["git", "-C", path, "describe", "--tags", "--always", "--dirty"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        print(f"{desc} (editable checkout)" if desc else "editable checkout")
    except Exception:
        print("editable checkout")
PYEOF
    )
    echo "- **benchlocal-cli:** \`${bl_bin}\` (version: \`${bl_ver:-unknown}\`${bl_src:+, source: \`${bl_src}\`}, installed/updated: ${bl_when})"
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

if [[ "$CONTAINER" == "none" ]]; then
  # Host engine build: no container to introspect. Docker ships a built-in
  # network object named "none", so `docker inspect none` succeeds (exit 0) on
  # any Docker host (club-3090#1067) — guard it so the body can't inspect a phantom.
  echo "_Host engine build (CONTAINER=none) — no container to introspect; container-scoped probes skipped._"
elif [[ -z "$CONTAINER" ]]; then
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
    # vLLM's runtime custom-AR veto (world>2 without NVLink — its gate never
    # consults peer access). Fed to the classifier so the verdict can't claim
    # "custom all-reduce ON" that vLLM already vetoed (#786).
    vllm_ar_gate=$(docker logs "$CONTAINER" 2>&1 | grep -m1 'Custom allreduce is disabled' || true)
    echo "**Interconnect / P2P engagement:**"
    if [[ -n "$nvlink_boot" || -n "$p2p_env" || -n "$vllm_ar_gate" ]]; then
      echo '```'
      [[ -n "$nvlink_boot" ]] && echo "$nvlink_boot"
      [[ -n "$vllm_ar_gate" ]] && { echo "# vLLM runtime:"; echo "$vllm_ar_gate"; }
      [[ -n "$p2p_env" ]] && { echo "# resolved container env:"; echo "$p2p_env"; }
      echo '```'
    else
      echo "_No \`[nvlink]\` boot line or NCCL_P2P/NVLINK_MODE env found — P2P engagement undetermined (single-GPU, a non-NCCL engine like llama.cpp, or an entrypoint predating detect_nvlink.sh)._"
    fi
    # Cross-referenced VERDICT (capability x engagement — the #488/#158 matrix).
    # Silent on single-GPU / no-capability rigs so the OK/WARN/INFO line is
    # always signal, never boilerplate.
    _p2p_verdict_line="$(p2p_verdict "$(p2p_gpu_count)" "$(p2p_host_capability)" \
      "$(printf '%s\n%s\n%s' "$nvlink_boot" "$vllm_ar_gate" "$p2p_env" | p2p_classify_engagement)")"
    [[ -n "$_p2p_verdict_line" ]] && { echo; echo "**Interconnect verdict:** ${_p2p_verdict_line}"; }
    # Kernel-module flavor — the WHY behind a P2P result on GeForce cards. A
    # proprietary (closed) module refuses P2P; the open modules can grant it, with
    # `topo -p2p rw` above the functional proof. Only meaningful multi-GPU.
    # We report open-vs-proprietary (detectable); we do NOT claim to fingerprint
    # the aikitoria patch — it's metadata-identical to stock nvidia-open.
    if [[ "$(p2p_gpu_count)" -ge 2 ]]; then
      case "$(p2p_driver_flavor)" in
        proprietary) echo; echo "**NVIDIA kernel module:** proprietary (closed) — refuses P2P on GeForce; the open kernel modules (\`nvidia-open\`, or a patched fork) are what enable it. A \`CNS\` in \`topo -p2p rw\` above is this. See docs/PCIE_P2P.md." ;;
        open)        echo; echo "**NVIDIA kernel module:** open (\`Dual MIT/GPL\`) — P2P-capable on GeForce; whether it's granted is the \`topo -p2p rw\` result above (\`OK\` = engaged, \`CNS\` = board/layout still refusing). Metadata can't tell stock \`nvidia-open\` from a patched fork — the topo result is the proof." ;;
      esac
      # Transfer-verified P2P (#786, read-only tier folded from #787): report
      # vLLM's functional-check cache when a boot ever ran with
      # VLLM_SKIP_P2P_CHECK=0 — host first, then the serving container.
      # Absence is normal (the check is off by default upstream); when present
      # it upgrades the verdict above from driver-asserted to measured.
      _tc_json=""; _tc_src=""
      _tc_f="$(p2p_transfer_cache_file)"
      if [[ -n "$_tc_f" ]]; then
        _tc_json="$(cat "$_tc_f" 2>/dev/null)"; _tc_src="${_tc_f/#$HOME/\~}"
      fi
      if [[ -z "$_tc_json" ]]; then
        _tc_f="$(docker exec "$CONTAINER" sh -c 'ls -t /root/.cache/vllm/gpu_p2p_access_cache_for_*.json 2>/dev/null | head -1' 2>/dev/null || true)"
        [[ -n "$_tc_f" ]] && { _tc_json="$(docker exec "$CONTAINER" cat "$_tc_f" 2>/dev/null)"; _tc_src="container:${_tc_f}"; }
      fi
      if [[ -n "$_tc_json" ]]; then
        read -r _tc_ok _tc_total <<<"$(printf '%s' "$_tc_json" | p2p_transfer_cache_parse)" || true
        _tc_line="$(p2p_transfer_verdict "${_tc_ok:-0}" "${_tc_total:-0}" "$_tc_src")"
        [[ -n "$_tc_line" ]] && { echo; echo "**Transfer check:** ${_tc_line}"; }
      fi
    fi
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
# Stage gating + engine liveness (#830)
# ---------------------------------------------------------------------------
# Every stage used to run unconditionally, so a mid-run engine death produced N
# more identical `<details>` blocks, each reading like its own fault:
#
#   ## bench.sh output
#   ERROR: service not reachable at http://localhost:8099/v1/models
#     Start with: cd compose && docker compose up -d
#   ## bench-agentic.sh output
#   ERROR: service not reachable at http://localhost:8099/v1/models
#     Start with: bash scripts/launch.sh
#
# That is one dead engine wearing two faults, and the "just start it" hints are
# actively wrong — the service WAS started, it crashed (#827). So we probe the
# endpoint between stages and, once it stops answering, skip the rest with the
# causal order stated instead of running each one into the same wall.
#
# Deliberately NOT done: aborting on a stage's own failure. A thin-VRAM-margin
# advisory is no reason to skip soak, and the point of --full is a complete
# artifact for cross-rig comparison. Bailing early would have made #827's report
# LESS useful — the soak crash is the finding. Only loss of the endpoint gates.

STAGE_ENDPOINT=""
ENGINE_PROBE=0        # 1 = we have an endpoint we can meaningfully probe
ENGINE_DEAD=0
ENGINE_DEAD_AFTER=""
ENGINE_UP_AT_START=0
LAST_STAGE_RUN=""

# ---------------------------------------------------------------------------
# Stage verdict accounting (#813) — inner verdicts must reach the outer exit.
# ---------------------------------------------------------------------------
# Committed to on #619: `report.sh --full` exited 0 while verify-stress printed
# "1 stress check(s) failed" inside. A --full chain whose exit code doesn't
# reflect inner verdicts is unsafe to automate against, which is the entire
# point of --full. @seanyourhighness caught it only by reading inner verdicts,
# which most reporters reasonably won't.
#
# Exit contract:
#   0  every stage that ran passed (or no stage was requested)
#   2  ADVISORY-only failure — a check fired that flags headroom or risk rather
#      than incorrectness. Today that is verify-stress's agent-safety VRAM
#      margin: recall is CORRECT at the ceiling, what fails is the margin for
#      sustained agent load. Distinguishable so a caller can gate on hard
#      failures alone.
#   1  hard failure — a stage failed a correctness check, could not run, or the
#      engine died mid-run.
#
# The advisory classifier reads the stage's own output rather than its exit
# code, because verify-stress exits with the COUNT of failed checks and does not
# distinguish the classes. `✗` is emitted only by fail(); the margin advisory
# prints `⚠ VRAM margin thin at ceiling`. So: nonzero exit + no `✗` + a margin
# line == advisory-only. Read from the RAW output, before redaction.
STAGE_ROWS=()
STAGE_WORST=0         # 0 clean · 2 advisory · 1 hard
STAGE_RAW_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_RAW_DIR"' EXIT

# Escalate to the worst verdict seen. Hard (1) outranks advisory (2).
stage_escalate() {
  case "$1" in
    1) STAGE_WORST=1 ;;
    2) [[ $STAGE_WORST -eq 1 ]] || STAGE_WORST=2 ;;
  esac
}

# stage_record <flag-key> <exit-code> <raw-output-file|"">
stage_record() {
  local key="$1" rc="$2" raw="${3:-}" verdict detail advisory=0 hard=0
  if [[ "$rc" -eq 0 ]]; then
    verdict="PASS"; detail="—"
  else
    if [[ -n "$raw" && -f "$raw" ]]; then
      hard="$(command grep -c '✗' "$raw" 2>/dev/null || true)"
      advisory="$(command grep -c 'VRAM margin thin at ceiling' "$raw" 2>/dev/null || true)"
    fi
    if [[ "${hard:-0}" -eq 0 && "${advisory:-0}" -gt 0 ]]; then
      verdict="ADVISORY"
      detail="agent-safety VRAM margin thin at ceiling (recall correct; headroom is not)"
      stage_escalate 2
    else
      verdict="FAIL"
      if [[ -n "$raw" && -f "$raw" && "${hard:-0}" -gt 0 ]]; then
        # Name the failing checks so they're identifiable from the exit path.
        detail="$(sed -E 's/\x1b\[[0-9;]*[mK]//g' "$raw" | command grep '✗' \
          | sed -E 's/^[[:space:]]*✗[[:space:]]*//' | head -3 | paste -sd';' - \
          | sed -E 's/;/ · /g')"
        [[ -n "$detail" ]] || detail="see the block above"
      else
        detail="see the block above"
      fi
      stage_escalate 1
    fi
  fi
  # `|` is the row separator AND the markdown cell separator — a check message
  # carrying one would split the row in both places.
  detail="${detail//|/ / }"
  STAGE_ROWS+=("$(stage_label "$key")|${rc}|${verdict}|${detail}")
}

# stage_skipped <flag-key> <reason>
stage_skipped() {
  STAGE_ROWS+=("$(stage_label "$1")|-|SKIPPED|$2")
  stage_escalate 1
}

# Flag key -> the script the stage runs, for human-readable causal statements.
stage_label() {
  case "$1" in
    verify)  echo "verify-full.sh" ;;
    stress)  echo "verify-stress.sh" ;;
    soak)    echo "soak-test.sh" ;;
    bench)   echo "bench.sh" ;;
    agentic) echo "bench-agentic.sh" ;;
    *)       echo "$1" ;;
  esac
}

# Resolve the engine endpoint the same way preflight.sh / soak-test.sh do: by
# the container's ENGINE-INTERNAL port mapping (vLLM 8000 / llama.cpp 8080 /
# sglang 30000), never a model-name allowlist. Mirrors, rather than sources,
# preflight.sh — that file executes checks at source time.
resolve_stage_endpoint() {
  if [[ -n "${URL:-}" ]]; then printf '%s\n' "${URL%/}"; return 0; fi
  if [[ -n "${ENDPOINT:-}" ]]; then printf '%s\n' "${ENDPOINT%/}"; return 0; fi
  have docker || return 0
  [[ -n "$CONTAINER" ]] || return 0
  local internal mapped port
  for internal in 8000 8080 30000; do
    mapped="$(docker port "$CONTAINER" "${internal}/tcp" 2>/dev/null | head -1 || true)"
    if [[ -n "$mapped" ]]; then
      port="${mapped##*:}"
      [[ "$port" =~ ^[0-9]+$ ]] && { printf 'http://localhost:%s\n' "$port"; return 0; }
    fi
  done
  return 0
}

engine_alive() {
  [[ $ENGINE_PROBE -eq 1 ]] || return 0   # can't probe → never claim it's dead
  curl -sf -m 5 "${STAGE_ENDPOINT}/v1/models" >/dev/null 2>&1
}

# Called BEFORE each stage. Returns 1 when the stage must be skipped.
stage_guard() {
  local stage="$1"
  if [[ $ENGINE_DEAD -eq 1 ]]; then
    printf '_SKIPPED — endpoint unreachable since **%s** (the engine appears to have crashed there). ' "$ENGINE_DEAD_AFTER"
    printf 'This stage was not run, so it contributes no evidence: it would only have reproduced the same '
    printf 'connection failure. Container logs are above. Re-run `bash scripts/report.sh --%s` once the service is back._\n' "$stage"
    stage_skipped "$stage" "endpoint unreachable since ${ENGINE_DEAD_AFTER}"
    return 1
  fi
  if [[ $ENGINE_PROBE -eq 1 ]] && ! engine_alive; then
    ENGINE_DEAD=1
    ENGINE_DEAD_AFTER="$(stage_label "${LAST_STAGE_RUN:-an earlier stage}")"
    printf '_SKIPPED — the endpoint stopped answering after **%s**. ' "$ENGINE_DEAD_AFTER"
    printf 'The engine did not survive that stage; every remaining stage is skipped rather than run into the '
    printf 'same wall. Container logs are above. Re-run `bash scripts/report.sh --%s` once the service is back._\n' "$stage"
    stage_skipped "$stage" "endpoint died during ${ENGINE_DEAD_AFTER}"
    return 1
  fi
  LAST_STAGE_RUN="$stage"
  return 0
}

if [[ $DO_VERIFY -eq 1 || $DO_STRESS -eq 1 || $DO_SOAK -eq 1 || $DO_BENCH -eq 1 || $DO_AGENTIC -eq 1 ]]; then
  STAGE_ENDPOINT="$(resolve_stage_endpoint)"
  if [[ -n "$STAGE_ENDPOINT" ]] && have curl; then
    ENGINE_PROBE=1
    if engine_alive; then
      ENGINE_UP_AT_START=1
    else
      # Never up. This IS the "just start it" case, and saying so once beats
      # saying it once per stage.
      ENGINE_DEAD=1
      ENGINE_DEAD_AFTER="before any stage ran (the endpoint was never reachable)"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Optional: verify-full
# ---------------------------------------------------------------------------

if [[ $DO_VERIFY -eq 1 ]]; then
  section "verify-full.sh output"
  if [[ ! -f scripts/verify-full.sh ]]; then
    echo "_scripts/verify-full.sh not found_"
  elif stage_guard verify; then
    # tee the RAW output aside so stage_record can classify + name the failing
    # checks; redaction would keep the markers but the tee is where the exit
    # path gets its evidence. PIPESTATUS[0] is the stage's own code.
    bash scripts/verify-full.sh 2>&1 | tee "${STAGE_RAW_DIR}/verify" | redact | details "verify-full output"
    stage_record verify "${PIPESTATUS[0]}" "${STAGE_RAW_DIR}/verify"
  fi
fi

# ---------------------------------------------------------------------------
# Optional: verify-stress
# ---------------------------------------------------------------------------

if [[ $DO_STRESS -eq 1 ]]; then
  section "verify-stress.sh output"
  if [[ ! -f scripts/verify-stress.sh ]]; then
    echo "_scripts/verify-stress.sh not found_"
  elif stage_guard stress; then
    bash scripts/verify-stress.sh 2>&1 | tee "${STAGE_RAW_DIR}/stress" | redact | details "verify-stress output (7 boundary checks incl. Cliff 2 needle recall)"
    stage_record stress "${PIPESTATUS[0]}" "${STAGE_RAW_DIR}/stress"
  fi
fi

# ---------------------------------------------------------------------------
# Optional: soak-continuous (catches Cliff 2b — the only test that does)
# ---------------------------------------------------------------------------

if [[ $DO_SOAK -eq 1 ]]; then
  section "soak-test.sh (SOAK_MODE=continuous) output"
  if [[ ! -f scripts/soak-test.sh ]]; then
    echo "_scripts/soak-test.sh not found_"
  elif stage_guard soak; then
    soak_run_dir="results/report-soak-$(date +%Y%m%d-%H%M%S)"
    SOAK_MODE=continuous SOAK_SESSIONS=5 SOAK_TURNS=5 SOAK_OUTPUT="$soak_run_dir" \
      SOAK_TIMEOUT_S="${SOAK_TIMEOUT_S:-1800}" \
      bash scripts/soak-test.sh 2>&1 | tee "${STAGE_RAW_DIR}/soak" | redact | details "soak-test stdout (5-session × 5-turn ramping conversation, ~25 min)"
    stage_record soak "${PIPESTATUS[0]}" "${STAGE_RAW_DIR}/soak"
    if [[ -f "$soak_run_dir/summary.md" ]]; then
      echo
      echo "**Soak summary** (\`$soak_run_dir/summary.md\`):"
      echo
      redact < "$soak_run_dir/summary.md"
    else
      echo "_soak summary.md not produced — check stdout above_"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Optional: bench
# ---------------------------------------------------------------------------

if [[ $DO_BENCH -eq 1 ]]; then
  section "bench.sh output"
  if [[ ! -f scripts/bench.sh ]]; then
    echo "_scripts/bench.sh not found_"
  elif stage_guard bench; then
    bash scripts/bench.sh 2>&1 | tee "${STAGE_RAW_DIR}/bench" | redact | details "bench output (3 warmups + 5 measured per prompt)"
    stage_record bench "${PIPESTATUS[0]}" "${STAGE_RAW_DIR}/bench"
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
  if [[ ! -f scripts/bench-agentic.sh ]]; then
    echo "_scripts/bench-agentic.sh not found_"
  elif stage_guard agentic; then
    SESSIONS=1 bash scripts/bench-agentic.sh 2>&1 | tee "${STAGE_RAW_DIR}/agentic" | redact | details "bench-agentic output (1 session x 12 default turns, curve-shape estimate; ~8 min estimate)"
    stage_record agentic "${PIPESTATUS[0]}" "${STAGE_RAW_DIR}/agentic"
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
# Engine liveness verdict (#830) — ONE verdict, not one per skipped stage.
# ---------------------------------------------------------------------------

if [[ $ENGINE_DEAD -eq 1 ]]; then
  section "Engine liveness"
  if [[ $ENGINE_UP_AT_START -eq 0 ]]; then
    cat <<EOF
> ❌ **The endpoint was never reachable — no stage ran.**
>
> \`${STAGE_ENDPOINT}/v1/models\` did not answer before the first stage, so the
> stages were skipped rather than each reporting the same connection failure.
>
> Start the service and re-run:
>
> \`\`\`bash
> bash scripts/launch.sh          # or: bash scripts/switch.sh <variant>
> \`\`\`
EOF
  else
    cat <<EOF
> ❌ **The engine died during this run — remaining stages were skipped.**
>
> The endpoint answered at the start and stopped answering after **${ENGINE_DEAD_AFTER}**.
> That is ONE fault, not one per stage: the later stages were skipped instead of
> each reproducing the same unreachable-endpoint error, which reads like
> additional independent failures and sends triage the wrong way.
>
> **The service was running and crashed — do not "just start it" without reading
> why.** The container log sections above carry the actual cause. Common ones on
> this stack: an MTP drafter emitting out-of-range draft token IDs under
> sustained multi-turn load (\`SPEC_N=3\` is the known-good workaround, see #758),
> or an OOM at high accumulated context.
>
> Re-run the skipped stages individually once the service is back.
EOF
  fi
fi

# ---------------------------------------------------------------------------
# Check summary (#813) — the failing check must be nameable from the exit path.
# ---------------------------------------------------------------------------

if [[ ${#STAGE_ROWS[@]} -gt 0 ]]; then
  section "Check summary"
  echo "| Stage | Exit | Verdict | Detail |"
  echo "|---|---:|---|---|"
  for row in "${STAGE_ROWS[@]}"; do
    IFS='|' read -r _s _rc _v _d <<< "$row"
    printf '| %s | %s | %s | %s |\n' "$_s" "$_rc" "$_v" "$_d"
  done | redact
  echo
  case "$STAGE_WORST" in
    0) echo "**Overall: PASS** — every stage that ran passed. \`report.sh\` exits 0." ;;
    2) echo "**Overall: ADVISORY** — no correctness check failed, but an advisory fired (headroom / risk, not incorrectness). \`report.sh\` exits **2**, so a caller can gate on hard failures alone." ;;
    *) echo "**Overall: FAIL** — at least one stage failed a check, could not run, or was skipped. \`report.sh\` exits **1**." ;;
  esac
fi

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

cat <<'EOF'

---

_Generated by `bash scripts/report.sh`. Flags: `--verify` (verify-full), `--stress` (verify-stress 7/7 incl. Cliff 2 needles), `--soak` (SOAK_MODE=continuous, catches Cliff 2b), `--bench` (canonical TPS), `--agentic` (multi-turn TTFT/decode curve-shape, ~8 min estimate), `--studio` (AI Studio / ComfyUI container log tails — for generation bugs), `--full` (all five, ~43 min estimate). Use `--no-redact` to disable redaction (internal sharing only). Exit code: 0 all-clear · 2 advisory-only · 1 hard failure._
EOF

exit "$STAGE_WORST"
