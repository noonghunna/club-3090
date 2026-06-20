#!/usr/bin/env bash
# detect_nvlink.sh must strip expandable_segments when P2P/custom-AR is ON even
# if compose pre-injected PYTORCH_CUDA_ALLOC_CONF (vllm#42609 / club-3090 boot crash).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DETECT="${ROOT_DIR}/scripts/detect_nvlink.sh"

run_case() {
  local mode="$1" pre="$2" expect_substr="$3" expect_absent="${4:-}"
  local out alloc
  out="$(PYTORCH_CUDA_ALLOC_CONF="$pre" NVLINK_MODE="$mode" bash -c "source '$DETECT' >/dev/null; printf '%s' \"\$PYTORCH_CUDA_ALLOC_CONF\"")"
  alloc="$out"
  if [[ "$alloc" != *"$expect_substr"* ]]; then
    echo "[detect-nvlink-alloc] FAIL mode=$mode pre='$pre' got='$alloc' expected substring '$expect_substr'" >&2
    exit 1
  fi
  if [[ -n "$expect_absent" && "$alloc" == *"$expect_absent"* ]]; then
    echo "[detect-nvlink-alloc] FAIL mode=$mode pre='$pre' got='$alloc' must not contain '$expect_absent'" >&2
    exit 1
  fi
}

# NVLink / pcie_p2p: strip expandable_segments from compose default injection.
run_case force_on "expandable_segments:True,max_split_size_mb:512" "max_split_size_mb:512" "expandable_segments"
run_case pcie_p2p "expandable_segments:True,max_split_size_mb:256" "max_split_size_mb:512" "expandable_segments"

# PCIe: preserve compose default when unset.
out="$(NVLINK_MODE=force_off bash -c "source '$DETECT' >/dev/null; printf '%s' \"\$PYTORCH_CUDA_ALLOC_CONF\"")"
if [[ "$out" != *"expandable_segments"* ]]; then
  echo "[detect-nvlink-alloc] FAIL force_off default missing expandable_segments: '$out'" >&2
  exit 1
fi

echo "[detect-nvlink-alloc] PASS: P2P paths strip expandable_segments; PCIe path keeps it"
