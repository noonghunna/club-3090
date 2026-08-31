#!/usr/bin/env bash
# thp.sh — transparent-hugepage facts for CPU-offload MoE slugs.
#
# WHY: with `-ot …exps=CPU` the expert weights are allocated by CUDA as PINNED
# HOST memory, which the kernel accounts as **Shmem**, not anonymous memory. So
# `transparent_hugepage/enabled` — the knob everyone reaches for — governs only
# the process heap and does nothing for the model. On a default Ubuntu
# (`enabled=madvise`, `shmem_enabled=never`) a ~105 GiB working set runs on
# ~27M × 4 KiB pages against a ~2K-entry L2 TLB. There is no error and no
# visible symptom.
#
# ⚠️ THE TRAP: setting `enabled=always` alone is NOT enough and LOOKS like it
# worked — AnonHugePages climbs (the heap) while the experts stay on 4 KiB
# pages. `shmem_enabled` is the knob that matters, and the only honest check is
# **ShmemHugePages ÷ Shmem**, because a knob reading `always` can still yield 0%
# under fragmentation.
#
# ⚠️ SCOPE: this is a FIRST-TOKEN LATENCY effect, not throughput. Measured
# 2026-08-30 on GLM-5.3-Flash: short-prompt TTFT ~593 → ~262 ms (-56%), while
# decode was -0.2% and prefill +0.9%/-2.7% — i.e. noise. Never word a hint as a
# tok/s promise.
#
# Paths are overridable so the guard test can drive fixtures without root.
THP_SYSFS_DIR="${THP_SYSFS_DIR:-/sys/kernel/mm/transparent_hugepage}"
THP_MEMINFO="${THP_MEMINFO:-/proc/meminfo}"

# thp_setting <knob> -> the bracketed active value, or "n/a"
thp_setting() {
  local f="$THP_SYSFS_DIR/$1"
  [[ -r "$f" ]] || { echo "n/a"; return; }
  sed -n 's/.*\[\([a-z_+]*\)\].*/\1/p' "$f" 2>/dev/null | head -1 | grep . || echo "n/a"
}

# thp_meminfo_kib <Key> -> kB value, or empty
thp_meminfo_kib() {
  [[ -r "$THP_MEMINFO" ]] || return 0
  awk -v k="$1:" '$1==k {print $2; exit}' "$THP_MEMINFO" 2>/dev/null
}

# thp_shmem_coverage_pct -> integer percent, or empty when there is nothing to judge.
# Empty is NOT zero: with no large Shmem allocation resident there is no model
# loaded, so a 0% reading would be a false alarm rather than a finding.
thp_shmem_coverage_pct() {
  local sh shp
  sh="$(thp_meminfo_kib Shmem)"; shp="$(thp_meminfo_kib ShmemHugePages)"
  [[ -n "$sh" && -n "$shp" ]] || return 0
  # below ~1 GiB of Shmem there is no offloaded model to speak of
  [[ "$sh" -lt 1048576 ]] && return 0
  awk -v a="$shp" -v b="$sh" 'BEGIN{ printf "%d", (a*100)/b }'
}
