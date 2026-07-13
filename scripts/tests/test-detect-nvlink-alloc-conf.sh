#!/usr/bin/env bash
# detect_nvlink.sh must DROP expandable_segments on the P2P / custom-all-reduce
# path (it crashes custom_all_reduce IPC graph-buffer registration — see
# docs/UPSTREAM.md → vllm#42609) while PRESERVING every other alloc-conf knob the
# compose / user set (max_split_size_mb, garbage_collection_threshold, ...). On
# the PCIe path (custom AR off) expandable_segments is kept untouched.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DETECT="${ROOT_DIR}/scripts/detect_nvlink.sh"

fail=0

# Resolve PYTORCH_CUDA_ALLOC_CONF as detect_nvlink.sh would leave it.
# $1 = NVLINK_MODE, $2 = pre-set PYTORCH_CUDA_ALLOC_CONF ("__UNSET__" to leave unset).
resolve() {
  local mode="$1" pre="$2"
  # 2>/dev/null: pcie_p2p's honest "unverified" warning (no mocked nvidia-smi here)
  # goes to stderr — irrelevant to the alloc-conf assertion, so keep output clean.
  if [[ "$pre" == "__UNSET__" ]]; then
    NVLINK_MODE="$mode" bash -c "unset PYTORCH_CUDA_ALLOC_CONF; source '$DETECT' >/dev/null 2>&1; printf '%s' \"\${PYTORCH_CUDA_ALLOC_CONF:-}\""
  else
    PYTORCH_CUDA_ALLOC_CONF="$pre" NVLINK_MODE="$mode" bash -c "source '$DETECT' >/dev/null 2>&1; printf '%s' \"\${PYTORCH_CUDA_ALLOC_CONF:-}\""
  fi
}

# Exact-match assertion (catches dangling/duplicate commas, not just substrings).
assert_exact() {
  local mode="$1" pre="$2" want="$3" got
  got="$(resolve "$mode" "$pre")"
  if [[ "$got" != "$want" ]]; then
    echo "[detect-nvlink-alloc] FAIL mode=$mode pre='$pre' got='$got' want='$want'" >&2
    fail=1
  fi
}

# Substring present + absent assertion (mode-parameterized robustness check).
assert_contains_absent() {
  local mode="$1" pre="$2" present="$3" absent="$4" got
  got="$(resolve "$mode" "$pre")"
  if [[ "$got" != *"$present"* ]]; then
    echo "[detect-nvlink-alloc] FAIL mode=$mode pre='$pre' got='$got' missing '$present'" >&2
    fail=1
  fi
  if [[ "$got" == *"$absent"* ]]; then
    echo "[detect-nvlink-alloc] FAIL mode=$mode pre='$pre' got='$got' must not contain '$absent'" >&2
    fail=1
  fi
}

# --- P2P / custom-all-reduce paths (force_on = NVLink, pcie_p2p = patched driver) ---

# Compose default injection: strip expandable_segments, keep the rest.
assert_exact force_on "expandable_segments:True,max_split_size_mb:512" "max_split_size_mb:512"

# A user's non-default max_split_size_mb must SURVIVE (not silently reset to 512).
assert_exact pcie_p2p "expandable_segments:True,max_split_size_mb:256" "max_split_size_mb:256"

# expandable_segments mid-list: both neighbors preserved, no dangling/double commas.
assert_exact force_on \
  "max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9" \
  "max_split_size_mb:512,garbage_collection_threshold:0.9"

# expandable_segments:False is also stripped (any value), neighbor preserved.
assert_exact force_on "expandable_segments:False,max_split_size_mb:512" "max_split_size_mb:512"

# Only expandable_segments set -> nothing left -> fall back to the NVLink default.
assert_exact force_on "expandable_segments:True" "max_split_size_mb:512"

# Unset entirely on a P2P rig -> the NVLink default, no expandable_segments.
assert_exact force_on "__UNSET__" "max_split_size_mb:512"

# Belt-and-suspenders: expandable_segments must never survive a P2P path.
assert_contains_absent force_on "expandable_segments:True,max_split_size_mb:256" "max_split_size_mb:256" "expandable_segments"
assert_contains_absent pcie_p2p "expandable_segments:True,max_split_size_mb:256" "max_split_size_mb:256" "expandable_segments"

# --- PCIe path (custom AR off) keeps expandable_segments untouched ---

# Compose default flows through unchanged.
assert_exact force_off "expandable_segments:True,max_split_size_mb:512" "expandable_segments:True,max_split_size_mb:512"

# Unset on PCIe -> the script's expandable_segments-on default.
assert_contains_absent force_off "__UNSET__" "expandable_segments" "max_split_size_mb:99999"

if [[ "$fail" -ne 0 ]]; then
  echo "[detect-nvlink-alloc] FAILED" >&2
  exit 1
fi
echo "[detect-nvlink-alloc] PASS: P2P strips ONLY expandable_segments (other knobs preserved); PCIe keeps it"
