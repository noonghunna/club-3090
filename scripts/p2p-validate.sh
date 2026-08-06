#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# p2p-validate.sh — does this rig actually complete NCCL collectives?
#
# The tier BELOW `nvidia-smi topo -p2p` and copy benchmarks. Those can all pass
# on a rig where every collective deadlocks or silently returns wrong data:
# a copy test synchronizes AFTER the transfer, while NCCL's kernels stay
# resident and must observe each other's writes mid-flight — so copy-then-sync
# can never validate the contract NCCL actually depends on.
#
# Runs a tiny all-reduce twice — once with P2P as your driver has it, once with
# NCCL_P2P_DISABLE=1 as a control — and checks the RESULT VALUES, not just that
# it finished. Read-only; changes nothing.
#
#   bash scripts/p2p-validate.sh
#   bash scripts/p2p-validate.sh --gpus 0,3 --timeout 120
#   bash scripts/p2p-validate.sh --image ghcr.io/…   # pick the torch container
#
# Exit: 0 healthy · 2 hang · 3 wrong data · 4 inconclusive · 5 unusable
# ---------------------------------------------------------------------------
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"   # #779 — python3 under a C/POSIX locale mangles non-ASCII

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PAYLOAD="${ROOT_DIR}/scripts/lib/p2p_collective_check.py"
GPUS="${GPUS:-0,1}"
TIMEOUT_S="${TIMEOUT_S:-90}"
IMAGE="${IMAGE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)    GPUS="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --image)   IMAGE="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 64 ;;
  esac
done

[[ -r "$PAYLOAD" ]] || { echo "ERROR: payload not found at $PAYLOAD" >&2; exit 5; }

# ---- 1. host python with torch? then no container needed ----
if python3 -c 'import torch' >/dev/null 2>&1; then
  echo "[p2p-validate] using host python3 (torch present)"
  GPUS="$GPUS" TIMEOUT_S="$TIMEOUT_S" exec python3 "$PAYLOAD"
fi

# ---- 2. otherwise run inside a container that has torch ----
command -v docker >/dev/null 2>&1 || {
  echo "ERROR: no host torch and no docker. Install torch, or run the payload yourself:" >&2
  echo "       python3 $PAYLOAD" >&2
  exit 5
}

if [[ -z "$IMAGE" ]]; then
  # Any vLLM/SGLang image ships torch + NCCL. Prefer whatever is already local so
  # the check never triggers a multi-GB pull.
  IMAGE="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
           | grep -E 'vllm|sglang' | grep -v '<none>' | head -1)"
fi
if [[ -z "$IMAGE" ]]; then
  echo "ERROR: no local vLLM/SGLang image found and no --image given." >&2
  echo "       Pass one you already have:  bash scripts/p2p-validate.sh --image <img>" >&2
  exit 5
fi

echo "[p2p-validate] host has no torch — running inside: $IMAGE"
echo "[p2p-validate] GPUs=$GPUS timeout=${TIMEOUT_S}s (read-only; nothing is changed)"

# --ipc=host: NCCL/torch shared-memory transports need more than docker's 64 MB default.
docker run --rm --gpus all --ipc=host \
  -e GPUS="$GPUS" -e TIMEOUT_S="$TIMEOUT_S" \
  -v "${PAYLOAD}:/p2p_collective_check.py:ro" \
  --entrypoint python3 "$IMAGE" /p2p_collective_check.py
exit $?
