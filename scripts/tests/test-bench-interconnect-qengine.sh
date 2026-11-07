#!/usr/bin/env bash
# test-bench-interconnect-qengine — bench.sh's interconnect block reports what QEngine says it chose
# (#1846). QEngine runs its own collectives and links no NCCL, so the vLLM / SGLang classifier found no
# evidence in its log. Its two libraries word it differently, and the block must not claim more than a
# line says:
#   1. layer 2 is n/a (no NCCL), never "no NCCL_P2P* in the serving environment"
#   2. the 27B (libqengine) with peer access: "P2P: decode all-reduce GPU to GPU", plus its prefill mode
#   3. Flash-Next (libqflash) with peer access prints no decode line: the block says it isn't stated
#   4. without peer access (both libraries' own format strings) and a refused pair: said, with a ⚠
#   5. CONTAINER=none: n/a, no log to read
#   6. QEngine 1.3.5's own decode-path line (QE_DECODE_COMM) wins over the 27B's "GPU to GPU" line, which
#      it still prints when peer access opens, also when the choice is host memory (real 1.3.5 lines)
#   7. 1.3.5.3 rewords that line to "P2P: peer access between all N GPUs" (real 1.3.5.3 lines)
#   8. 1.3.5.4's Flash-Next "[qflash] parallel mode" lines, which now name decode's path, change nothing
#      (real 1.3.5.4 lines)
# Hermetic: the function is extracted from bench.sh. The log reader is the real p2p_qengine_transport
# from lib/p2p-state.sh (report.sh reads QEngine through it too, #1862); the other p2p_* helpers and
# docker are stubs.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"   # #779
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fail() { echo "FAIL: $1" >&2; exit 1; }
ok() { echo "  ✓ $1"; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
sed -n '/^bench_interconnect_block()/,/^}/p' "$ROOT_DIR/scripts/bench.sh" > "$TMP/block.sh"
command grep -q 'QEngine runs its own collectives' "$TMP/block.sh" || fail "the QEngine branch is not in bench_interconnect_block"
mkdir -p "$TMP/bin"
cat > "$TMP/bin/docker" <<'SH'
#!/usr/bin/env bash
case "$1" in
  inspect) exit 0 ;;
  logs) printf '%b' "${QE_LOG:-}" ;;
esac
exit 0
SH
chmod +x "$TMP/bin/docker"

block() {  # <container> <log> -> the block's output
  QE_LOG="$2" CONTAINER="$1" PATH="$TMP/bin:$PATH" bash -c '
    source "$2"
    p2p_gpu_count() { echo 2; }; p2p_host_capability() { echo pcie_p2p; }; p2p_driver_flavor() { echo open; }
    ENGINE_KIND=qengine; source "$1"; bench_interconnect_block' _ "$TMP/block.sh" "$ROOT_DIR/scripts/lib/p2p-state.sh" 2>&1
}

out="$(block qengine-qwen38-27b-dual 'P2P: decode all-reduce GPU to GPU (peer access between all 2 GPUs)\n[comm] prefill transfers: copy engine (QE_COMM=auto, auto)\n')"
[[ "$out" == *"layer 2  NCCL use         : n/a — QEngine runs its own collectives (no NCCL)"* ]] || fail "layer 2: $out"
[[ "$out" == *"layer 3  engine transport : decode all-reduce GPU to GPU (peer access); prefill transfers: copy engine (QE_COMM=auto)"* ]] \
  || fail "27B with peer access: $out"
[[ "$out" != *"engine custom-AR :"* && "$out" != *"NCCL_P2P"* ]] || fail "no vLLM / SGLang classifier row for QEngine: $out"
ok "27B with peer access: layer 2 n/a, decode GPU to GPU, prefill on the copy engine"

out="$(block qengine-qwen38-flash-next-dual '[comm] topology: numa -1,-1 (1 node), P2P rank 0-1:0\n[comm] prefill transfers: P2P SM loads / stores, pairs 1<-0:pull 0<-1:pull (FLASH_COMM=auto, auto: P2P tax 1.78 (cap 1.30, 1 NUMA node), CE tax 1.03, P2P 13 CE 13 host 9 GB/s)\n')"
[[ "$out" == *"layer 3  engine transport : decode path not stated in the log; prefill transfers: P2P SM loads / stores (FLASH_COMM=auto)"* ]] \
  || fail "Flash-Next with peer access: $out"
ok "Flash-Next with peer access: decode path not claimed, prefill P2P SM loads / stores"

out="$(block qengine-qwen38-27b-dual 'P2P: enabling peer access 0->1 failed (peer access is not supported between these two devices), host path\nQE_COMM: collectives through pinned host memory (no peer access)\n[comm] prefill transfers: host-staged (QE_COMM=auto, auto)\n')"
[[ "$out" == *"engine transport : no peer access: collectives through pinned host memory; prefill transfers: host-staged (QE_COMM=auto); ⚠ P2P: enabling peer access 0->1 failed"* ]] \
  || fail "27B without peer access: $out"
out="$(block qengine-qwen38-flash-next-dual '[flash] no peer access: attention exchanges through pinned host memory (slower than P2P)\n')"
[[ "$out" == *"engine transport : no peer access: collectives through pinned host memory"* ]] || fail "Flash-Next without peer access: $out"
ok "without peer access (both libraries) and a refused pair: said, with a ⚠"

out="$(block none 'P2P: decode all-reduce GPU to GPU (peer access between all 2 GPUs)\n')"
[[ "$out" == *"engine transport : n/a — no container to read QEngine's log from"* ]] || fail "CONTAINER=none: $out"
ok "CONTAINER=none: n/a"

P2P_LINE='P2P: decode all-reduce GPU to GPU (peer access between all 2 GPUs)\n'
PRE27='[comm] prefill transfers: copy engine (QE_COMM=auto, auto)\n'
ANNOUNCE='[comm] decode collectives: auto, timed on real decode steps at the end of the start-up (P2P SM stores vs pinned host memory); the rule alone would pick P2P (the prefill rule keeps P2P: P2P kept: tax over the cap, under the 1 NUMA node cap)\n'
out="$(block qengine-qwen38-27b-dual "${P2P_LINE}${ANNOUNCE}${PRE27}"'[comm] decode collectives (all-reduce, all-gather): pinned host memory (QE_DECODE_COMM=host, forced)\n')"
[[ "$out" == *"engine transport : decode collectives pinned host memory (QE_DECODE_COMM=host); prefill transfers: copy engine (QE_COMM=auto)"* ]] \
  || fail "1.3.5 27B, QE_DECODE_COMM=host beside the GPU-to-GPU line: $out"
ok "1.3.5 27B forced to host memory: the decode-collectives line wins over 'GPU to GPU'"
out="$(block qengine-qwen38-27b-dual "${P2P_LINE}${ANNOUNCE}${PRE27}"'[comm] decode collectives (QE_DECODE_COMM=auto, real decode steps, median of 4 interleaved rounds of 12): single sequence: P2P SM stores (P2P 15.75, host memory 17.11 ms/step, host faster in 0/4); batched (4 sequences): P2P SM stores (P2P 23.46, host memory 33.57 ms/step, host faster in 0/4); calibration 8731 ms\n')"
[[ "$out" == *"engine transport : decode collectives P2P SM stores (QE_DECODE_COMM=auto); prefill transfers: copy engine (QE_COMM=auto)"* ]] \
  || fail "1.3.5 27B, auto timed: $out"
out="$(block qengine-qwen38-27b-dual "${P2P_LINE}${ANNOUNCE}${PRE27}"'[comm] decode collectives (QE_DECODE_COMM=auto, from the start-up cache, measured 2026-11-06 23:22:49): single sequence: P2P SM stores (P2P 15.75, host memory 17.11 ms/step); batched: P2P SM stores (P2P 23.46, host memory 33.57 ms/step); QE_DECODE_COMM=remeasure measures again\n')"
[[ "$out" == *"engine transport : decode collectives P2P SM stores (QE_DECODE_COMM=auto); prefill"* ]] || fail "1.3.5 27B, auto from the cache: $out"
ok "1.3.5 27B auto, timed and from the start-up cache: P2P SM stores"
out="$(block qengine-qwen38-flash-next-dual "${ANNOUNCE}"'[comm] prefill transfers: P2P SM loads / stores, pairs 1<-0:pull 0<-1:pull (FLASH_COMM=auto, auto: P2P tax 1.69)\n[comm] decode collectives (QE_DECODE_COMM=auto, real decode steps, median of 4 interleaved rounds of 12): single sequence: P2P SM stores (P2P 23.90, host memory 28.62 ms/step, host faster in 0/4); batched (4 sequences): P2P SM stores (P2P 63.67, host memory 69.16 ms/step, host faster in 0/4); calibration 9020 ms\n')"
[[ "$out" == *"engine transport : decode collectives P2P SM stores (QE_DECODE_COMM=auto); prefill transfers: P2P SM loads / stores (FLASH_COMM=auto)"* ]] \
  || fail "1.3.5 Flash-Next, auto timed: $out"
ok "1.3.5 Flash-Next auto: its decode path is stated now"
# the two paths differ (auto picks per batch shape; the format of the real lines above)
out="$(block qengine-qwen38-27b-dual "${P2P_LINE}${PRE27}"'[comm] decode collectives (QE_DECODE_COMM=auto, real decode steps, median of 4 interleaved rounds of 12): single sequence: pinned host memory (P2P 30.10, host memory 21.40 ms/step, host faster in 4/4); batched (4 sequences): P2P SM stores (P2P 40.00, host memory 52.00 ms/step, host faster in 0/4); calibration 9000 ms\n')"
[[ "$out" == *"engine transport : decode collectives: one sequence pinned host memory, batched P2P SM stores (QE_DECODE_COMM=auto)"* ]] \
  || fail "1.3.5 auto with different paths per shape: $out"
out="$(block qengine-qwen38-27b-dual "${P2P_LINE}${ANNOUNCE}"'[comm] decode collectives (QE_DECODE_COMM=auto): calibration could not run a decode step, P2P SM stores (120 ms)\n')"
[[ "$out" == *"engine transport : decode collectives P2P SM stores (QE_DECODE_COMM=auto, not timed)"* ]] || fail "1.3.5 auto, calibration failed: $out"
ok "1.3.5 auto: per-shape paths, and a calibration that couldn't run"

# 7. 1.3.5.3's real 27B lines: the peer-access line no longer names a decode path
out="$(block qengine-qwen38-27b-dual 'P2P: peer access between all 2 GPUs (the paths: [comm] prefill transfers, [comm] decode collectives)\n[comm] decode collectives: auto, timed on real decode steps at the end of the start-up (P2P SM stores vs pinned host memory); the rule alone would pick P2P (the prefill rule keeps P2P: P2P kept: tax over the cap, under the 1 NUMA node cap)\n[comm] prefill transfers: copy engine (QE_COMM=auto, auto)\n[comm] decode collectives (QE_DECODE_COMM=auto, real decode steps, median of 4 interleaved rounds of 12): single sequence: P2P SM stores (P2P 15.67, host memory 17.33 ms/step, host faster in 0/4); batched (4 sequences): P2P SM stores (P2P 23.23, host memory 33.41 ms/step, host faster in 0/4); calibration 8713 ms\n')"
[[ "$out" == *"engine transport : decode collectives P2P SM stores (QE_DECODE_COMM=auto); prefill transfers: copy engine (QE_COMM=auto)"* ]] \
  || fail "1.3.5.3 27B, reworded peer-access line: $out"
ok "1.3.5.3 27B: the reworded peer-access line, decode path from its own line"
# 8. 1.3.5.4's real Flash-Next lines: the parallel-mode lines name decode's path (bpx03/QEngine#19); the
#    decode-collectives line still decides, and "peer access yes" is not read as "no peer access"
out="$(block qengine-qwen38-flash-next-dual '[comm] decode collectives: auto, timed on real decode steps at the end of the start-up (P2P SM stores vs pinned host memory); the rule alone would pick P2P (the prefill rule keeps P2P: P2P kept: tax over the cap, under the 1 NUMA node cap)\n[comm] prefill transfers: P2P SM loads / stores, pairs 1<-0:pull 0<-1:pull (FLASH_COMM=auto, auto: P2P tax 1.71 (median of 5 runs, 1.67-1.73; cap 1.30, 1 NUMA node cap 1.90 on a narrow link, severe 3.00, 1 NUMA node, narrowest link x8), CE tax 1.03, P2P 13 CE 13 host 9 GB/s: P2P kept: tax over the cap, under the 1 NUMA node cap)\n[qflash] parallel mode: tp (auto: peer access between the cards: TP); probe: 2 cards, peer access yes, TP prefill exchanges P2P, decode collectives timed at the end of the start-up (model: +0.000 ms/token, +0.0 ms/step), host link 13.5 GB/s\n[comm] decode collectives (QE_DECODE_COMM=auto, real decode steps, median of 4 interleaved rounds of 12): single sequence: P2P SM stores (P2P 23.85, host memory 28.59 ms/step, host faster in 0/4); batched (4 sequences): P2P SM stores (P2P 63.58, host memory 69.07 ms/step, host faster in 0/4); calibration 19811 ms\n[qflash] parallel mode tp, decode collectives: single sequence P2P SM stores, batched P2P SM stores\n')"
[[ "$out" == *"engine transport : decode collectives P2P SM stores (QE_DECODE_COMM=auto); prefill transfers: P2P SM loads / stores (FLASH_COMM=auto)"* ]] \
  || fail "1.3.5.4 Flash-Next, parallel-mode lines naming decode: $out"
ok "1.3.5.4 Flash-Next: the parallel-mode lines change nothing"
echo "PASS test-bench-interconnect-qengine"
