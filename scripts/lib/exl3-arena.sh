#!/usr/bin/env bash
# exl3-arena.sh — page state of an ExLlamaV3 CPU-MoE server's expert arena (club-3090#1542).
#
# Sourced by report.sh and bench.sh. Whether the CPU-resident experts sit on 2 MiB pages is ~19%
# of decode on exl3, and the default arena reaches them only through a one-shot background
# collapse that can fail silently, so a boot can serve on 4 KiB pages and a bench reads 10-20%
# low as if the engine had regressed. The deterministic fix is opt-in in the exl3 composes:
# EXL3_MOE_PINNED_ARENA=1 EXL3_MOE_ARENA_HUGE=2m plus reserved vm.nr_hugepages.
#
# System AnonHugePages and report.sh's ShmemHugePages line do NOT answer this: the first is
# dominated by the parent's heap, and the default arena is anonymous memory, not Shmem.

# The reader runs under python3 (in the container); UTF-8 mode before the first call (#779).
export PYTHONUTF8="${PYTHONUTF8:-1}"

# exl3_arena_state <container> — print one key=value line from scripts/lib/exl3_arena.py run
# inside the container (verdict=2MiB|4KiB|none). Prints nothing and returns 1 when the container
# can't be read.
exl3_arena_state() {
  local c="${1:-}" here out
  [[ -n "$c" ]] || return 1
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  out="$(docker exec -i "$c" python3 - < "$here/exl3_arena.py" 2>/dev/null)" || return 1
  [[ "$out" == *verdict=* ]] || return 1
  printf '%s\n' "$out"
}

# exl3_arena_explain <state line> — one human sentence for a report or a bench capture.
exl3_arena_explain() {
  case "${1:-}" in
    *verdict=2MiB*) echo "experts on 2 MiB pages — decode numbers from this boot are comparable" ;;
    *verdict=4KiB*) echo "experts on 4 KiB pages — decode reads up to ~19% LOW on this boot (club-3090#1542); use EXL3_MOE_PINNED_ARENA=1 EXL3_MOE_ARENA_HUGE=2m, or wait for the collapse and re-measure" ;;
    *verdict=none*) echo "no expert arena resident yet (still loading, or not a CPU-MoE config)" ;;
    *) echo "arena state unreadable" ;;
  esac
}
