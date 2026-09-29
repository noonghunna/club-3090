#!/usr/bin/env bash
# test-report-membw — report.sh's memory-bandwidth line reads the same way on any rig.
#
# The first probe read ~90 GB/s on an 8-channel EPYC that does ~160 (plain stores paid
# a read-for-ownership STREAM doesn't count, threads were created inside the timed
# region, nothing was pinned). scripts/lib/membw_probe.c fixes that for every rig, and
# scripts/lib/membw.sh adds the rated peak when dmidecode names the channels.
#
#   1. the plan, on fake /sys trees: one thread per PHYSICAL core, the 1/4-1/2-all sweep,
#      L3 counted per cache id (a VM's shape too), arrays >= 4x L3, capped by free memory.
#   2. a real (tiny) measurement prints the documented fields.
#   3. the rated peak from dmidecode: AMD desktop, Intel desktop, 8-channel EPYC — and
#      nothing (never a guess) when a DIMM doesn't name its channel or its speed.
#   4. the report line: % of rated, and the VM / busy-host / capped / plain-store notes.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
CC=""; for c in cc gcc clang; do command -v "$c" >/dev/null 2>&1 && { CC="$c"; break; }; done
[[ -n "$CC" ]] || { echo "test-report-membw: SKIP (no C compiler)"; exit 0; }
"$CC" -O2 -pthread -o "$T/probe" scripts/lib/membw_probe.c 2>"$T/cc.log" || { echo "✗ probe does not compile: $(head -5 "$T/cc.log")" >&2; exit 1; }

# fake /sys/devices/system/cpu: mkcpu <root> <cpu> <package> <core> <l3 size> <l3 id> <l3 shared list>
mkcpu() {
  local d="$1/cpu$2"
  mkdir -p "$d/topology" "$d/cache/index0" "$d/cache/index3"
  echo "$3" > "$d/topology/physical_package_id"; echo "$4" > "$d/topology/core_id"
  echo 1 > "$d/cache/index0/level"; echo 32K > "$d/cache/index0/size"
  echo 3 > "$d/cache/index3/level"; echo "$5" > "$d/cache/index3/size"
  echo "$6" > "$d/cache/index3/id"; echo "$7" > "$d/cache/index3/shared_cpu_list"
}
plan() { MEMBW_SYSFS_ROOT="$1" MEMBW_MEMINFO="${2:-$T/mem-big}" MEMBW_PLAN_ONLY=1 "$T/probe"; }
printf 'MemTotal: 268435456 kB\nMemAvailable: 201326592 kB\n' > "$T/mem-big"

echo "1. the plan"
S="$T/smt"; for c in 0 1 2 3 4 5 6 7; do mkcpu "$S" $c 0 $((c % 4)) 32768K 0 0-7; done
got="$(plan "$S")"
[[ "$got" == *"cpus=8 cores=4 "* && "$got" == *"threads=1,2,4 "* && "$got" == *"pins=1:0|2:0,2|4:0,1,2,3 "* ]] \
  && ok "8 CPUs on 4 SMT cores: one pinned thread per physical core (never a sibling); tries 1, 2, 4" || bad "SMT plan: $got"
# two sockets x 8 cores, numbered socket by socket (cpus 0-7 on package 0, 8-15 on 1),
# each socket two 4-core dies (L3 ids 0-3): a half run must reach both sockets, all dies
S="$T/2s"; for c in $(seq 0 15); do mkcpu "$S" $c $((c / 8)) $((c % 8)) 32768K $((c / 4)) "$((c / 4 * 4))-$((c / 4 * 4 + 3))"; done
got="$(plan "$S")"
[[ "$got" == *"cores=16 "* && "$got" == *"pins=4:0,4,8,12|8:0,2,4,6,8,10,12,14|16:"* ]] \
  && ok "2 sockets x 2 dies: 4 threads land one per die, 8 threads spread over both sockets (not packed on socket 0)" || bad "2-socket spread: $got"
# interleaved numbering (some dual-socket Intel: even CPUs on socket 0, odd on socket 1).
# A plain stride over CPU order would land every thread on socket 0; ordered by
# (socket, die, core) first, 4 threads get 2 per socket.
S="$T/2si"; for c in $(seq 0 15); do mkcpu "$S" $c $((c % 2)) $((c / 2)) 32768K $((c % 2)) "$((c % 2))-15:2"; done
got="$(plan "$S")"
[[ "$got" == *"pins=4:0,8,1,9|8:0,4,8,12,1,5,9,13|16:"* ]] \
  && ok "interleaved socket numbering (even/odd): 4 threads get 2 per socket, not 4 on socket 0" || bad "interleaved: $got"
got="$(plan "$T/smt")"
[[ "$got" == *"l3_mib=32 array_mib=256 "* ]] && ok "a 32 MiB L3 still gets 256 MiB arrays (the floor)" || bad "SMT sizing: $got"
S="$T/vcache"; for c in $(seq 0 63); do mkcpu "$S" $c 0 $c 98304K $((c / 8)) "$((c / 8 * 8))-$((c / 8 * 8 + 7))"; done
got="$(plan "$S")"
[[ "$got" == *"cores=64 "* && "$got" == *"threads=16,32,64 "* && "$got" == *"l3_mib=768 array_mib=1024 "* ]] \
  && ok "8 x 96 MiB L3 (V-Cache class): arrays grow to 1024 MiB (3 x >= 4x L3)" || bad "V-Cache sizing: $got"
S="$T/vm"; for c in $(seq 0 31); do mkcpu "$S" $c 0 $c 16384K $c 0-31; done
got="$(plan "$S")"
[[ "$got" == *"l3_mib=512 "* ]] && ok "a VM that lists every vCPU as sharing but gives each its own cache id: counted per id (as lscpu does)" || bad "VM L3: $got"
S="$T/nopology"; mkdir -p "$S/cpu0" "$S/cpu1"
got="$(plan "$S")"
[[ "$got" == *"cpus=2 cores=2 "* ]] && ok "no topology files: every CPU counts as a core (no crash)" || bad "no topology: $got"
printf 'MemAvailable: 1048576 kB\n' > "$T/mem-1g"
got="$(plan "$T/smt" "$T/mem-1g")"
[[ "$got" == *"array_mib=170 capped=1"* ]] && ok "1 GiB free: arrays capped to half of it, and it says so" || bad "cap: $got"
printf 'MemAvailable: 307200 kB\n' > "$T/mem-300m"
got="$(plan "$T/smt" "$T/mem-300m")"
[[ "$got" == "skip=low-memory"* ]] && ok "300 MiB free: skipped, with the reason" || bad "low memory: $got"

echo "2. a real (tiny) measurement"
got="$(MEMBW_ARRAY_MIB=16 MEMBW_REPS=1 "$T/probe")"
if [[ "$got" =~ ^triad=([0-9.]+)\ read=([0-9.]+)\ threads=[0-9]+\ cores=[0-9]+\ array_mib=16\ l3_mib=[0-9]+\ capped=0\ nt=([01])$ ]] \
   && awk -v t="${BASH_REMATCH[1]}" -v r="${BASH_REMATCH[2]}" 'BEGIN { exit !(t > 0 && r > 0) }'; then
  ok "prints triad / read / threads / cores / sizes / nt ($got)"
else
  bad "measurement line: $got"
fi
case "$(uname -m)" in x86_64|amd64)
  [[ "$got" == *" nt=1" ]] && ok "x86-64: non-temporal stores in use" || bad "x86 without non-temporal stores: $got" ;;
esac

echo "3. the rated peak from dmidecode"
# shellcheck source=../lib/membw.sh
source scripts/lib/membw.sh
dimm() {  # <locator> <bank locator> <size> <speed>
  printf '\n\nHandle 0x%04x, DMI type 17\nMemory Device\n\tSize: %s\n\tLocator: %s\n\tBank Locator: %s\n\tConfigured Memory Speed: %s\n\n' \
    "$RANDOM" "$3" "$1" "$2" "$4"
}
amd="$(for ch in A B; do for n in 1 2; do dimm "DIMM_${ch}${n}" "P0 CHANNEL $ch" "32 GB" "3200 MT/s"; done; done)"
[[ "$(club_membw_rated <<<"$amd")" == "51.2 2 3200" ]] && ok "AMD desktop, 4 DIMMs on 2 channels: 51.2 GB/s (2 x 3200 x 8)" || bad "AMD desktop: '$(club_membw_rated <<<"$amd")'"
intel="$(for c in 0 1; do for n in 0 1; do dimm "Controller${c}-ChannelA-DIMM${n}" "BANK $((c * 2 + n))" "16 GB" "5600 MT/s"; done; done)"
[[ "$(club_membw_rated <<<"$intel")" == "89.6 2 5600" ]] && ok "Intel desktop, channel in the Locator (BANK n is per slot): 89.6 GB/s" || bad "Intel desktop: '$(club_membw_rated <<<"$intel")'"
epyc="$(for ch in A B C D E F G H; do dimm "DIMM_${ch}1" "P0 CHANNEL $ch" "32 GB" "3200 MT/s"; done)"
[[ "$(club_membw_rated <<<"$epyc")" == "204.8 8 3200" ]] && ok "8-channel EPYC: 204.8 GB/s" || bad "EPYC: '$(club_membw_rated <<<"$epyc")'"
empty="$epyc$(dimm "DIMM_H2" "P0 CHANNEL H" "No Module Installed" "Unknown")"
[[ "$(club_membw_rated <<<"$empty")" == "204.8 8 3200" ]] && ok "an empty slot is ignored" || bad "empty slot: '$(club_membw_rated <<<"$empty")'"
mixed="$(dimm DIMM_A1 "P0 CHANNEL A" "32 GB" "3200 MT/s")$(dimm DIMM_B1 "P0 CHANNEL B" "32 GB" "2933 MT/s")"
[[ "$(club_membw_rated <<<"$mixed")" == "46.9 2 2933" ]] && ok "mixed speeds: the slowest sets the peak" || bad "mixed: '$(club_membw_rated <<<"$mixed")'"
vm="$(dimm "DIMM 0" "Not Specified" "233824 MB" "Unknown")"
[[ -z "$(club_membw_rated <<<"$vm")" ]] && ok "a VM's single virtual DIMM (no channel, no speed): no peak" || bad "VM: '$(club_membw_rated <<<"$vm")'"
nochan="$(dimm DIMM_A1 "BANK 0" "32 GB" "3200 MT/s")$(dimm DIMM_A2 "BANK 1" "32 GB" "3200 MT/s")"
[[ -z "$(club_membw_rated <<<"$nochan")" ]] && ok "slots without a channel name: no peak (slots are not channels)" || bad "no channel: '$(club_membw_rated <<<"$nochan")'"

echo "4. the report line"
line() { MEMBW_VIRT="${VIRT:-none}" MEMBW_LOADAVG="${LOAD:-0.10}" MEMBW_FAKE_RESULT="$1" club_membw_lines "${2:-}"; }
R="triad=159.3 read=146.2 threads=32 cores=32 array_mib=684 l3_mib=512 capped=0 nt=1"
got="$(line "$R" "$epyc")"
[[ "$got" == *"159.3 GB/s STREAM Triad · 146.2 GB/s read (best with 32 thread(s) on 32 physical core(s); 3 × 684 MiB arrays)"* \
   && "$got" == *"= **78% of 204.8 GB/s rated** (8 channel(s) × 3200 MT/s × 8 B;"* ]] \
  && ok "measured + % of rated in one line" || bad "line: $got"
got="$(line "$R")"
[[ "$got" != *"% of"* && "$(wc -l <<<"$got")" == 1 ]] && ok "no dmidecode: no percentage, no notes on a quiet bare-metal host" || bad "no-dmi line: $got"
got="$(VIRT=kvm line "$R")"
[[ "$got" == *"inside a VM (\`kvm\`)"* ]] && ok "says when it was measured inside a VM" || bad "VM note: $got"
got="$(LOAD=2.57 line "$R")"
[[ "$got" == *"host was busy while measuring (load average 2.57)"* ]] && ok "warns when the host was busy (load >= 1)" || bad "load note: $got"
got="$(line "${R/capped=0/capped=1}")"
[[ "$got" == *"limited by free memory"* ]] && ok "says when the arrays were capped" || bad "capped note: $got"
got="$(line "${R/nt=1/nt=0}")"
[[ "$got" == *"plain stores"* ]] && ok "says when Triad used plain stores" || bad "nt note: $got"
got="$(line "skip=low-memory avail_mib=300")"
[[ "$got" == *"not measured (only 300 MiB"* ]] && ok "a skip says why" || bad "skip: $got"
got="$(line "triad=0.0 read=0.0")"
[[ "$got" == *"probe failed"* ]] && ok "a failed run says so (never a zero)" || bad "failed: $got"
command grep -qF '_report_mem_bandwidth "$_dmi"' scripts/report.sh && command grep -qF 'club_membw_lines "${1:-}"' scripts/report.sh \
  && ok "report.sh hands its dmidecode text to the new probe" || bad "report.sh is not wired to membw.sh"

[ "$fail" -eq 0 ] && echo "test-report-membw: ok" || { echo "test-report-membw: FAIL"; exit 1; }
