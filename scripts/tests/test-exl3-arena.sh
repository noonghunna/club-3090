#!/usr/bin/env bash
# test-exl3-arena — the exl3 expert-arena page state is readable, reported, and switchable.
#
# WHY THIS EXISTS
# ---------------
# On exl3 CPU-MoE, whether the CPU-resident experts sit on 2 MiB pages is ~19% of decode
# (club-3090#1542: 62.5 / 83.2 tok/s on 4 KiB vs 74.6 / 88.4 on 2 MiB, same container). The
# default arena reaches 2 MiB only through a one-shot background collapse that can fail silently,
# so a boot can serve on 4 KiB pages and its bench reads like an engine regression. The composes
# silently dropped the opt-in fix (EXL3_MOE_PINNED_ARENA / EXL3_MOE_ARENA_HUGE were never
# forwarded, and the pinned arena needs an unlimited memlock), and nothing reported the state:
# report.sh's THP lines read Shmem and system AnonHugePages, which miss this arena or are
# dominated by the parent's heap.
#
# Checks: the arena reader on fixture /proc trees (the parent's heap must NOT count), the
# docker-exec wrapper, both cpumoe composes' env + ulimits, the entrypoint's [arena] branch run
# for real, and that report.sh / bench.sh call the reader.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
READER="$ROOT_DIR/scripts/lib/exl3_arena.py"
fail() { echo "FAIL: $1" >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# mkproc <tree> <pid> <cmdline> <Rss kB> <hugetlb kB> <AnonHugePages kB> <ShmemPmdMapped kB>
mkproc() {
  mkdir -p "$1/$2"
  printf '%s' "$3" | tr ' ' '\0' > "$1/$2/cmdline"
  printf 'Rss:            %s kB\nShared_Hugetlb: %s kB\nPrivate_Hugetlb: 0 kB\nAnonHugePages:  %s kB\nShmemPmdMapped: %s kB\n' \
    "$4" "$5" "$6" "$7" > "$1/$2/smaps_rollup"
}
PARENT="python3 main.py --cpu-moe-threads 28"
WORKER="/opt/venv/bin/python3 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5)"
G=1048576   # kB per GiB

# 1. default arena, collapse never landed: workers on 4 KiB. The parent carries 32 GiB of THP on
#    its heap; counting it would read "2MiB" (the exact misreading #1542 warns about).
T="$WORK/cold"; mkproc "$T" 7 "$PARENT" $((40*G)) 0 $((32*G)) 0
mkproc "$T" 255 "$WORKER" $((1*G)) 0 0 0; mkproc "$T" 382 "$WORKER" $((13*G)) 0 0 0
got="$(EXL3_ARENA_PROC="$T" python3 "$READER")"
[[ "$got" == *"workers=2"* && "$got" == *"verdict=4KiB"* ]] || fail "cold arena (4 KiB workers, huge parent heap) should read 4KiB: got '$got'"

# 2. default arena after the collapse (or THP enabled=always): anonymous huge pages in the workers.
T="$WORK/anon"; mkproc "$T" 7 "$PARENT" $((40*G)) 0 $((32*G)) 0
mkproc "$T" 382 "$WORKER" $((14*G)) 0 $((25*G/2)) 0
got="$(EXL3_ARENA_PROC="$T" python3 "$READER")"
[[ "$got" == *"verdict=2MiB"* && "$got" == *"anon_thp_gib=12.5"* ]] || fail "collapsed anonymous arena should read 2MiB: got '$got'"

# 3. pinned arena on hugetlbfs (measured on the reference rig: Rss 1.8 GiB, hugetlb 12.2 GiB).
T="$WORK/pinned"; mkproc "$T" 7 "$PARENT" $((40*G)) $((13*G)) $((32*G)) 0
mkproc "$T" 255 "$WORKER" $((8*G/10)) $((2*G/10)) 0 0; mkproc "$T" 382 "$WORKER" $((1*G)) $((12*G)) 0 0
got="$(EXL3_ARENA_PROC="$T" python3 "$READER")"
[[ "$got" == *"verdict=2MiB"* && "$got" == *"hugetlb_gib=12.2"* ]] || fail "pinned hugetlb arena should read 2MiB with 12.2 GiB hugetlb: got '$got'"

# 4. no workers (not CPU-MoE, or still loading).
T="$WORK/none"; mkproc "$T" 7 "$PARENT" $((40*G)) 0 $((32*G)) 0
got="$(EXL3_ARENA_PROC="$T" python3 "$READER")"
[[ "$got" == "workers=0 verdict=none" ]] || fail "no spawn workers should read verdict=none: got '$got'"
echo "  ✓ arena reader: 4KiB / 2MiB (anonymous, hugetlb) / none; the parent's heap is not counted"

# 5. the wrapper report.sh and bench.sh call, through a docker stub that runs `python3 -` locally.
STUB="$WORK/bin"; mkdir -p "$STUB"
cat > "$STUB/docker" <<'DOCK'
#!/usr/bin/env bash
[ "$1" = exec ] && [ "$3" = good ] && exec python3 -
exit 1
DOCK
chmod +x "$STUB/docker"
# shellcheck source=../lib/exl3-arena.sh
. "$ROOT_DIR/scripts/lib/exl3-arena.sh"
got="$(PATH="$STUB:$PATH" EXL3_ARENA_PROC="$WORK/cold" exl3_arena_state good)"
[[ "$got" == *"verdict=4KiB"* ]] || fail "exl3_arena_state through docker exec: got '$got'"
[[ "$(exl3_arena_explain "$got")" == *"19%"* ]] || fail "a 4KiB verdict should explain the decode cost"
if PATH="$STUB:$PATH" exl3_arena_state broken >/dev/null; then fail "an unreadable container must return non-zero"; fi
echo "  ✓ exl3_arena_state / exl3_arena_explain via docker exec"

# 6. both cpumoe composes forward the knobs (bare names) and lift memlock for the pinned arena.
for c in "$ROOT_DIR"/models/qwen3.8-flash-next/exllamav3/compose/dual/exl3-*/cpumoe.yml; do
  python3 - "$c" <<'PY' || fail "compose $(basename "$(dirname "$c")") is missing the arena env or memlock ulimit"
import sys, yaml
svc = next(iter(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["services"].values()))
env = svc.get("environment") or []
for name in ("EXL3_MOE_PINNED_ARENA", "EXL3_MOE_ARENA_HUGE", "EXL3_MOE_ARENA_DEBUG"):
    assert name in env, f"{name} not forwarded (bare form)"
assert (svc.get("ulimits") or {}).get("memlock") == {"soft": -1, "hard": -1}, "memlock not unlimited"
PY
done
echo "  ✓ cpumoe composes forward EXL3_MOE_* and set memlock unlimited"

# 7. the entrypoint's [arena] branch, run for real with python3 stubbed (records its argv).
C305="$ROOT_DIR/models/qwen3.8-flash-next/exllamav3/compose/dual/exl3-3.05bpw/cpumoe.yml"
ENTRY="$(python3 -c 'import sys,yaml; s=next(iter(yaml.safe_load(open(sys.argv[1]))["services"].values())); print(s["entrypoint"][2].replace("$$","$"))' "$C305")"
PYSTUB="$WORK/py"; mkdir -p "$PYSTUB"; printf '#!/usr/bin/env bash\necho "PY $*"\n' > "$PYSTUB/python3"; chmod +x "$PYSTUB/python3"
run_entry() { env -i PATH="$PYSTUB:/usr/bin:/bin" "$@" bash -c "$ENTRY" -- --host 0.0.0.0 2>&1; }
out="$(run_entry)"
[[ "$out" == *"[arena] default (unpinned)"* && "$out" == *"PY main.py"* ]] || fail "default boot: no [arena] default line or no exec: $out"
out="$(run_entry EXL3_MOE_PINNED_ARENA=1 EXL3_MOE_ARENA_HUGE=2m)"
[[ "$out" == *"[arena] pinned, hugetlbfs 2m: host has"* && "$out" == *"PY main.py"* ]] || fail "pinned 2m boot: $out"
set +e; out="$(run_entry EXL3_MOE_PINNED_ARENA=1 EXL3_MOE_ARENA_HUGE=4k)"; rc=$?; set -e
[[ "$rc" == 64 && "$out" == *"must be 2m or 1g"* && "$out" != *"PY main.py"* ]] || fail "EXL3_MOE_ARENA_HUGE=4k must refuse with 64 before exec (rc=$rc): $out"
echo "  ✓ entrypoint [arena]: default / pinned 2m / bad value refused before exec"

# 8. the reports read it.
command grep -q 'exl3_arena_state "$CONTAINER"' "$ROOT_DIR/scripts/report.sh" || fail "report.sh no longer reads the exl3 arena"
command grep -q 'CAPTURE: EXL3 EXPERT ARENA' "$ROOT_DIR/scripts/bench.sh" || fail "bench.sh lost its exl3 arena capture"
command grep -q -- '--cpu-moe-split-experts\* \]\]' "$ROOT_DIR/scripts/bench.sh" || fail "bench.sh's arena capture is no longer gated on exl3 CPU-MoE"
echo "  ✓ report.sh + bench.sh read the arena"

echo "test-exl3-arena: ok"
