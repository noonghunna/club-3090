#!/usr/bin/env bash
# test-report-gpu-attribution — report.sh says WHO holds each card's VRAM, per
# card (#1537). The old loop credited the one running club container with every
# card that had memory in use. Stubs nvidia-smi, docker and /proc; no GPU needed.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FAIL=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; FAIL=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"
CID_OURS=$(printf 'a%.0s' {1..64}); CID_OTHER=$(printf 'b%.0s' {1..64})

# nvidia-smi: GPU table and compute apps come from files the cases write.
cat > "$T/bin/nvidia-smi" <<EOF
#!/usr/bin/env bash
case "\$*" in
  *--query-gpu=index,uuid,memory.used*) cat "$T/gpus" ;;
  *--query-compute-apps=gpu_uuid,pid,process_name*) cat "$T/apps" 2>/dev/null ;;
esac
EOF
# docker: ps maps ids to names; exec answers the container's own compute-apps view.
cat > "$T/bin/docker" <<EOF
#!/usr/bin/env bash
case "\$1" in
  ps)   printf '%s sglang-cmp170hx\n%s vllm-other\n' "$CID_OURS" "$CID_OTHER" ;;
  exec) cat "$T/inner" 2>/dev/null ;;
esac
EOF
chmod +x "$T/bin/nvidia-smi" "$T/bin/docker"
proc_pid() { mkdir -p "$T/proc/$1"; echo "0::/system.slice/docker-$2.scope" > "$T/proc/$1/cgroup"; }

run() {  # run <our_container>
  ( export PATH="$T/bin:$PATH" GPU_ATTR_PROC="$T/proc"
    source "$ROOT_DIR/scripts/lib/gpu-select.sh"
    gpu_vram_report_lines "$1" )
}
line_for() { command grep -F -- "**GPU $2 idle VRAM:**" <<<"$1"; }

# --- Case 1: the #1537 shape. Ours runs on GPU 1; another container's vLLM
#     workers hold GPUs 0 and 2. Each card must name ITS holder.
printf '0, GPU-0, 20000\n1, GPU-1, 21000\n2, GPU-2, 20500\n' > "$T/gpus"
printf 'GPU-0, 101, VLLM::Worker_TP0\nGPU-2, 102, VLLM::Worker_TP1\nGPU-1, 201, sglang::scheduler\n' > "$T/apps"
proc_pid 101 "$CID_OTHER"; proc_pid 102 "$CID_OTHER"; proc_pid 201 "$CID_OURS"
out="$(run sglang-cmp170hx)"
l0="$(line_for "$out" 0)"; l1="$(line_for "$out" 1)"; l2="$(line_for "$out" 2)"
[[ "$l1" == *'held by running `sglang-cmp170hx`'* ]] && ok "GPU 1 → our container" || bad "GPU 1: $l1"
[[ "$l0" == *'held by `vllm-other`'* && "$l0" != *'held by running `sglang'* ]] && ok "GPU 0 → the other container, not ours" || bad "GPU 0: $l0"
[[ "$l2" == *'held by `vllm-other`'* && "$l2" == *'not `sglang-cmp170hx`'* ]] && ok "GPU 2 → the other container, says not ours" || bad "GPU 2: $l2"

# --- Case 2: a process outside any container falls back to its name.
printf 'GPU-0, 301, python3\n' > "$T/apps"; mkdir -p "$T/proc/301"; echo "0::/user.slice" > "$T/proc/301/cgroup"
printf '0, GPU-0, 5000\n' > "$T/gpus"
out="$(run "")"
[[ "$(line_for "$out" 0)" == *'held by `python3`'* ]] && ok "non-container process named by its process name" || bad "case 2: $out"

# --- Case 3: host shows NO processes (some VMs / runtimes); the container's own
#     view places ours on GPU 1. GPU 0 has memory and no owner → flagged, never ours.
printf '0, GPU-0, 3000\n1, GPU-1, 21000\n' > "$T/gpus"; : > "$T/apps"; echo "GPU-1" > "$T/inner"
out="$(run sglang-cmp170hx)"
[[ "$(line_for "$out" 1)" == *'held by running `sglang-cmp170hx`'* ]] && ok "fallback: container view places ours on GPU 1" || bad "case 3 GPU 1: $out"
[[ "$(line_for "$out" 0)" == *'⚠ something is using this GPU'*'not `sglang-cmp170hx`'* ]] && ok "unowned card flagged, not credited to ours" || bad "case 3 GPU 0: $out"

# --- Case 4: an idle card is a ✓ whatever else is running.
printf '0, GPU-0, 12\n' > "$T/gpus"; : > "$T/apps"; : > "$T/inner"
[[ "$(line_for "$(run sglang-cmp170hx)" 0)" == *'12 MiB ✓'* ]] && ok "idle card is ✓" || bad "case 4"

# --- Negative control: report.sh must call the per-card function, not carry its
#     own loop again (the #1537 bug lived in report.sh, not in a library).
if command grep -q 'gpu_vram_report_lines "\$our_container"' "$ROOT_DIR/scripts/report.sh" \
   && ! command grep -q 'held by running \\`${our_container}\\`' "$ROOT_DIR/scripts/report.sh"; then
  ok "report.sh delegates to gpu_vram_report_lines"
else
  bad "report.sh does not use gpu_vram_report_lines (or still has its own per-container line)"
fi

(( FAIL )) && { echo "test-report-gpu-attribution: FAIL" >&2; exit 1; }
echo "test-report-gpu-attribution: ok"
