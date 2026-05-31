#!/usr/bin/env bash
# Test: power-cap-sweep.sh multi-GPU logic (pure functions, mocked nvidia-smi/docker).
#
# Covers the bug-prone logic of the multi-GPU sweep redesign:
#   1. resolve_gpu_indices  — --gpus > --gpu > container-detect > all-GPUs fallback
#   2. gpu_indices_from_container — read NVIDIA_VISIBLE_DEVICES / device requests
#   3. aggregate_gpu_sample — collapse N per-GPU sampler lines to one synthetic
#      line: SUM power, MAX util/temp, OR throttle (keeps downstream untouched)
#   4. capture_envelopes    — per-GPU INIT/STOCK arrays + intersected MIN/MAX range
#   5. restore_gpus         — RESET=1 -> stock per GPU; --no-reset -> captured per GPU
#
# The end-to-end sweep (setting caps, measuring) needs a real dual-GPU rig; this
# guards the logic that is wrong/subtle, with mocks. Functions are extracted with
# sed (the repo pattern) so the script's main body doesn't run.
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); }
no()  { echo "FAIL: $1" >&2; FAIL=$((FAIL+1)); }
eq()  { if [[ "$2" == "$3" ]]; then ok; else no "$1: expected '$3', got '$2'"; fi; }

# --- Extract the functions under test -----------------------------------------
HELPERS="$(mktemp --suffix=.sh)"
for fn in gpu_indices_from_container resolve_gpu_indices aggregate_gpu_sample capture_envelopes restore_gpus; do
  sed -n "/^${fn}()/,/^}/p" scripts/power-cap-sweep.sh >> "$HELPERS"
done

tmp="$(mktemp -d)"
cleanup() { rm -rf "$tmp" "$HELPERS"; }
trap cleanup EXIT

# --- Mock nvidia-smi: 2 GPUs; query values + record -pl calls -----------------
cat > "$tmp/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
declare -A MINV=([0]=100 [1]=100) MAXV=([0]=400 [1]=420) DEFV=([0]=370 [1]=390) LIMV=([0]=300 [1]=310)
args="$*"
# set power limit: -pl <W> -i <idx>
if [[ "$args" == *"-pl "* ]]; then
  w=""; idx=""; prev=""
  for a in "$@"; do case "$prev" in -pl) w="$a";; -i) idx="$a";; esac; prev="$a"; done
  echo "PL idx=$idx w=$w" >> "$NVSMI_LOG"; echo "set to ${w} W"; exit 0
fi
if [[ "$1" == "-L" ]]; then echo "GPU 0: Mock"; echo "GPU 1: Mock"; exit 0; fi
if [[ "$1" == "-pm" ]]; then exit 0; fi
# query: find -i index (absent for the all-indices listing)
idx=""; prev=""; has_i=0
for a in "$@"; do [[ "$prev" == "-i" ]] && { idx="$a"; has_i=1; }; prev="$a"; done
if [[ "$args" == *"--query-gpu=index"* && "$has_i" -eq 0 ]]; then printf '0\n1\n'; exit 0; fi
case "$args" in
  *power.default_limit*) echo "${DEFV[$idx]}";;
  *power.min_limit*)     echo "${MINV[$idx]}";;
  *power.max_limit*)     echo "${MAXV[$idx]}";;
  *power.limit*)         echo "${LIMV[$idx]}";;
  *name*)                echo "Mock GPU";;
  *memory.total*)        echo "24576";;
  *)                     echo "";;
esac
EOF
chmod +x "$tmp/nvidia-smi"

# --- Mock docker: inspect returns NVIDIA_VISIBLE_DEVICES for a known container -
cat > "$tmp/docker" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "inspect" ]]; then
  # Env listing form used by gpu_indices_from_container.
  echo "PATH=/usr/bin"
  echo "NVIDIA_VISIBLE_DEVICES=${MOCK_NVD:-0,1}"
  exit 0
fi
exit 0
EOF
chmod +x "$tmp/docker"

export PATH="$tmp:$PATH"
export NVSMI_LOG="$tmp/nvsmi.log"

# shellcheck source=/dev/null
source "$HELPERS"

# --- 1. resolve_gpu_indices precedence ----------------------------------------
out="$(GPUS="0,1" GPU_INDEX_SET=0 GPU_INDEX=0 CONTAINER=none resolve_gpu_indices)";        eq "resolve --gpus explicit" "$out" "0,1"
out="$(GPUS="" GPU_INDEX_SET=1 GPU_INDEX=1 CONTAINER=none resolve_gpu_indices)";            eq "resolve --gpu single"    "$out" "1"
out="$(GPUS="" GPU_INDEX_SET=0 GPU_INDEX=0 CONTAINER=vllm MOCK_NVD=0,1 resolve_gpu_indices 2>/dev/null)"; eq "resolve from container" "$out" "0,1"
out="$(GPUS="" GPU_INDEX_SET=0 GPU_INDEX=0 CONTAINER=none resolve_gpu_indices 2>/dev/null)"; eq "resolve fallback all GPUs" "$out" "0,1"

# --- 2. gpu_indices_from_container 'all' -> empty (caller falls back) ----------
out="$(MOCK_NVD=all gpu_indices_from_container vllm)";  eq "container NVD=all -> empty" "$out" ""

# --- 3. aggregate_gpu_sample: SUM power, MAX util/temp, OR throttle ------------
agg="$(printf '%s\n%s\n' \
  '0, 95, 320.50, 70, 1800, 9501, P2, Active, Not Active' \
  '1, 88, 310.00, 72, 1755, 9501, P2, Not Active, Not Active' | aggregate_gpu_sample)"
eq "aggregate -> summed/max/or" "$agg" "agg, 95, 630.50, 72, 1800, 9501, P2, Active, Not Active"
# single GPU passes through (sum == that GPU)
agg1="$(printf '%s\n' '0, 90, 300.00, 65, 1700, 9501, P2, Not Active, Not Active' | aggregate_gpu_sample)"
eq "aggregate single GPU" "$agg1" "agg, 90, 300.00, 65, 1700, 9501, P2, Not Active, Not Active"

# --- 4. capture_envelopes: intersected range + per-GPU arrays -----------------
declare -A INIT_ARR=() STOCK_ARR=()
GPU_INDICES=(0 1)
capture_envelopes
eq "MIN_LIMIT = max(per-gpu min)" "$MIN_LIMIT" "100"
eq "MAX_LIMIT = min(per-gpu max)" "$MAX_LIMIT" "400"
eq "INIT_ARR[0] = current limit"  "${INIT_ARR[0]}" "300"
eq "INIT_ARR[1] = current limit"  "${INIT_ARR[1]}" "310"
eq "STOCK_ARR[0] = default limit" "${STOCK_ARR[0]}" "370"
eq "STOCK_ARR[1] = default limit" "${STOCK_ARR[1]}" "390"

# --- 5. restore_gpus: stock (default) vs captured (--no-reset) -----------------
: > "$NVSMI_LOG"
RESET=1 restore_gpus
got="$(sort "$NVSMI_LOG" | tr '\n' ';')"
eq "RESET=1 restores stock per GPU" "$got" "PL idx=0 w=370;PL idx=1 w=390;"
: > "$NVSMI_LOG"
RESET=0 restore_gpus
got="$(sort "$NVSMI_LOG" | tr '\n' ';')"
eq "--no-reset restores captured per GPU" "$got" "PL idx=0 w=300;PL idx=1 w=310;"

echo "----------------------------------------"
echo "PASS: $PASS  FAIL: $FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
echo "OK: power-cap-sweep multi-GPU logic"
