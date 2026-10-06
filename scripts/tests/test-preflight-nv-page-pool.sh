#!/usr/bin/env bash
# test-preflight-nv-page-pool.sh — the NVIDIA driver page-pool hint in switch.sh.
#
# WHY THIS EARNS ITS KEEP: the hint fires on a HOST condition (a pool left by a
# stopped CUDA-VMM slug) that a normal test rig never has, so it would rot unnoticed.
# The failure modes that would do real damage are the false ones:
#   • blaming NVIDIA for memory something else holds (pools off, ZFS ARC, hugetlb)
#   • firing on every host because the accounting forgot a category
#   • BLOCKING a boot over what is only a hint
#   • being wired AFTER the host-RAM gates it exists to explain
#
# Fixture-driven: preflight_nvidia_page_pool reads NV_POOL_MEMINFO / NV_POOL_PARAMS /
# NV_POOL_ARCSTATS, so every case is an end-to-end run against a synthetic host.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
fail=0
ok()  { echo "  ok   — $1"; }
bad() { echo "  FAIL — $1" >&2; fail=1; }
echo "== test-preflight-nv-page-pool =="

# shellcheck source=/dev/null
source scripts/lib/compose-meta.sh 2>/dev/null
# shellcheck source=/dev/null
source scripts/preflight.sh 2>/dev/null

declare -F preflight_nvidia_page_pool >/dev/null 2>&1 \
  && ok "preflight_nvidia_page_pool defined" || { bad "preflight_nvidia_page_pool missing"; exit 1; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
GIB=$(( 1024 * 1024 ))   # kB per GiB
TOTAL=$(( 224 * GIB ))

# mkmem <name> <unaccounted_kB> [hugetlb_kB] [mode]
#   mode=hugetlb  (default) reports the hugetlb pool in the `Hugetlb:` line
#   mode=legacy   omits `Hugetlb:` and reports HugePages_Total x Hugepagesize instead
# Every other category carries a realistic value; MemFree absorbs the rest, so the
# only memory outside every category is exactly <unaccounted_kB>.
mkmem() {
  local name="$1" gap="$2" huge="${3:-0}" mode="${4:-hugetlb}"
  local buffers=102400 cached=$(( 142 * GIB )) anon=$(( 5 * GIB )) slab=$(( 3 * GIB ))
  local kstack=51200 ptables=204800 vmalloc=524288 percpu=102400
  local free=$(( TOTAL - gap - huge - buffers - cached - anon - slab - kstack - ptables - vmalloc - percpu ))
  {
    echo "MemTotal:       $TOTAL kB"
    echo "MemFree:        $free kB"
    echo "MemAvailable:   $(( free + cached )) kB"
    echo "Buffers:        $buffers kB"
    echo "Cached:         $cached kB"
    echo "AnonPages:      $anon kB"
    echo "Shmem:          20480 kB"
    echo "KReclaimable:   $(( 2 * GIB )) kB"
    echo "Slab:           $slab kB"
    echo "SReclaimable:   $(( 2 * GIB )) kB"
    echo "KernelStack:    $kstack kB"
    echo "PageTables:     $ptables kB"
    echo "SecPageTables:  0 kB"
    echo "VmallocUsed:    $vmalloc kB"
    echo "Percpu:         $percpu kB"
    if [[ "$mode" == "legacy" ]]; then
      echo "HugePages_Total: $(( huge / 2048 ))"
      echo "Hugepagesize:       2048 kB"
    else
      echo "HugePages_Total: $(( huge / 2048 ))"
      echo "Hugepagesize:       2048 kB"
      echo "Hugetlb:        $huge kB"
    fi
  } > "$TMP/$name.meminfo"
}
mkparams() { # <name> <EnableSystemMemoryPools value | "absent">
  {
    echo "ResmanDebugLevel: 4294967295"
    [[ "$2" == "absent" ]] || echo "EnableSystemMemoryPools: $2"
    echo "RegistryDwords: \"RMForceStaticBar1=1\""
  } > "$TMP/$1.params"
}
mkarc() { # <name> <arc size in bytes>
  printf '13 1 0x01 147 39984 3307392399 1234567890\nname                            type data\nhits                            4    123\nsize                            4    %s\n' \
    "$2" > "$TMP/$1.arcstats"
}
# run <meminfo> <params> [arcstats] -> stderr text
run() {
  NV_POOL_MEMINFO="$TMP/$1.meminfo" NV_POOL_PARAMS="$TMP/$2.params" \
  NV_POOL_ARCSTATS="$TMP/${3:-none}.arcstats" \
    preflight_nvidia_page_pool 2>&1 >/dev/null
}
rc_of() {
  NV_POOL_MEMINFO="$TMP/$1.meminfo" NV_POOL_PARAMS="$TMP/$2.params" \
  NV_POOL_ARCSTATS="$TMP/${3:-none}.arcstats" \
    preflight_nvidia_page_pool >/dev/null 2>&1; echo $?
}

mkmem   pool58   $(( 58 * GIB ))           # ~58 GiB pool after a CUDA-VMM slug stopped
mkmem   fresh    $(( 1 * GIB ))            # freshly loaded driver
mkmem   edge     $(( 4 * GIB - 1024 ))     # just under the threshold
mkmem   arena    $(( 1 * GIB )) $(( 13 * GIB ))          # exl3 hugetlb arena, nothing else
mkmem   arenaleg $(( 1 * GIB )) $(( 13 * GIB )) legacy   # same, kernel without Hugetlb:
mkparams on      529
mkparams off     0
mkparams nokey   absent
mkarc   arc58    $(( 58 * 1024 * 1024 * 1024 ))
mkarc   arc10    $(( 10 * 1024 * 1024 * 1024 ))

# --- POSITIVE: the condition this exists for ----------------------------------
out="$(run pool58 on)"
printf '%s' "$out" | command grep -q 'echo 2 | sudo tee /proc/sys/vm/drop_caches' \
  && ok "58 GiB pool: prints the drop_caches line" || bad "silent on a 58 GiB pool"
printf '%s' "$out" | command grep -q '~62 GB of host RAM' \
  && ok "reports the gap in decimal GB like the RAM gates (58 GiB -> ~62 GB)" \
  || bad "wrong size in the hint: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | command grep -q 'echo 3' \
  && bad "hint must drop slab only (2), never the page cache (3)" \
  || ok "hint drops slab only, keeps the page cache"
out="$(run pool58 on arc10)"
printf '%s' "$out" | command grep -q '~51 GB of host RAM' \
  && ok "subtracts a 10 GiB ZFS ARC and still fires (~51 GB)" \
  || bad "ARC subtraction wrong: $(printf '%s' "$out" | head -1)"

# --- NEGATIVE CONTROLS: other memory must not be blamed on NVIDIA --------------
[[ -z "$(run fresh on)" ]] && ok "quiet on a freshly loaded driver (1 GiB)" \
  || bad "false alarm at 1 GiB"
[[ -z "$(run edge on)" ]] && ok "quiet just under the 4 GiB threshold" \
  || bad "fires below the threshold"
[[ -z "$(run pool58 off)" ]] && ok "quiet when EnableSystemMemoryPools=0 (no pool to blame)" \
  || bad "blames a pool the driver does not keep"
[[ -z "$(run pool58 nokey)" ]] && ok "quiet when the driver has no pool parameter" \
  || bad "fires on a driver without pools"
[[ -z "$(NV_POOL_MEMINFO="$TMP/pool58.meminfo" NV_POOL_PARAMS="$TMP/missing.params" NV_POOL_ARCSTATS="$TMP/none.arcstats" \
         preflight_nvidia_page_pool 2>&1)" ]] \
  && ok "quiet with no /proc/driver/nvidia (WSL, no NVIDIA driver)" \
  || bad "fires without an NVIDIA driver"
[[ -z "$(run pool58 on arc58)" ]] && ok "quiet when the ZFS ARC explains the gap" \
  || bad "blames NVIDIA for the ZFS ARC"
[[ -z "$(run arena on)" ]] && ok "quiet with a 13 GiB hugetlb pool (counted, not unaccounted)" \
  || bad "reads a hugetlb arena as a driver pool"
[[ -z "$(run arenaleg on)" ]] && ok "quiet with hugetlb reported only as HugePages_Total x size" \
  || bad "legacy hugetlb accounting missed"

# --- it must NEVER block --------------------------------------------------------
for c in "pool58 on" "fresh on" "pool58 off" "pool58 on arc10"; do
  # shellcheck disable=SC2086
  [[ "$(rc_of $c)" == "0" ]] || bad "returned non-zero on '$c' — a hint must never block a boot"
done
ok "returns 0 on every host state (never blocks)"

# --- wiring: switch.sh must call it after the teardown, BEFORE the RAM gates ---
body="$(awk '/^up_variant\(\) \{/{f=1} f{print} f && /^\}/{exit}' scripts/switch.sh)"
ln_of() { printf '%s\n' "$body" | command grep -n -m1 "$1" | cut -d: -f1; }
hint="$(ln_of 'preflight_nvidia_page_pool')"
lm="$(ln_of 'preflight_lmcache_ram ')"
cpu="$(ln_of 'preflight_cpu_offload_ram ')"
if [[ -n "$hint" && -n "$lm" && -n "$cpu" ]] && (( hint < lm && hint < cpu )); then
  ok "up_variant runs the hint before the LMCache and CPU-offload RAM gates"
else
  bad "up_variant wiring wrong (hint=${hint:-missing} lmcache=${lm:-?} cpu-offload=${cpu:-?})"
fi
printf '%s\n' "$body" | command grep -q 'preflight_nvidia_page_pool || true' \
  && ok "switch.sh cannot fail the launch on the hint" || bad "switch.sh call is not '|| true'"

[[ $fail -eq 0 ]] && echo "test-preflight-nv-page-pool: ok" || echo "test-preflight-nv-page-pool: FAILED" >&2
exit $fail
