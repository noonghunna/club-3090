#!/usr/bin/env bash
# gpu-select.sh — runtime-agnostic GPU selection helpers (#610 Phase A).
#
# ONE mechanism for pinning specific GPUs on BOTH container runtimes:
#   - classic nvidia runtime: honors NVIDIA_VISIBLE_DEVICES but RENUMBERS the
#     exposed set inside the container (host GPUs 1,2 -> 0,1) — so an
#     index-based CUDA mask points at the wrong / a nonexistent card;
#   - CDI runtimes (NixOS, nvidia-ctk cdi): IGNORE NVIDIA_VISIBLE_DEVICES
#     entirely (devices come from the compose's CDI device_ids) — only a
#     CUDA-level mask pins cards there.
# GPU UUIDs sidestep both: the classic runtime exposes-by-UUID, and
# CUDA_VISIBLE_DEVICES masks-by-UUID regardless of exposure order.
#
# Source this file; it defines functions only (no side effects on source).
# Shared by scripts/launch.sh (--gpus) and — via the parallel python resolver
# in estate_cli.py — the estate boot path. test-gpu-select asserts the two
# resolvers agree.

# gpu_select_indices_to_uuids "<idx_csv>"
#   Resolve a comma-separated list of host GPU indices to their UUIDs.
#   Echoes the UUID csv on success; echoes NOTHING (empty) on any failure
#   (no nvidia-smi, an index that doesn't resolve) so callers fall back to
#   raw indices — the pre-UUID behavior.
gpu_select_indices_to_uuids() {
  local idx_csv="$1" out="" idx uuid
  [[ -z "$idx_csv" ]] && return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local IFS=','
  read -ra _gpu_sel_idx <<< "$idx_csv"
  for idx in "${_gpu_sel_idx[@]}"; do
    idx="${idx//[[:space:]]/}"
    [[ -z "$idx" ]] && continue
    uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$idx" 2>/dev/null | head -1 | tr -d '[:space:]')"
    if [[ "$uuid" != GPU-* ]]; then
      # Any index that fails to resolve -> abandon the whole set (partial
      # UUID pinning is worse than the index fallback).
      printf '' && return 0
    fi
    out="${out:+${out},}${uuid}"
  done
  printf '%s' "$out"
}

# gpu_select_export "<idx_csv>" ["<log_prefix>"]
#   Resolve the indices to UUIDs and EXPORT both CUDA_VISIBLE_DEVICES and
#   NVIDIA_VISIBLE_DEVICES (UUIDs when resolvable, raw indices otherwise).
#   Emits a one-line note to stderr when UUID-pinned. The launcher's #611
#   inline block, factored out.
gpu_select_export() {
  local idx_csv="$1" log_prefix="${2:-gpu}" uuids
  [[ -z "$idx_csv" ]] && return 0
  uuids="$(gpu_select_indices_to_uuids "$idx_csv")"
  if [[ -n "$uuids" ]]; then
    export CUDA_VISIBLE_DEVICES="$uuids"
    export NVIDIA_VISIBLE_DEVICES="$uuids"
    echo "[${log_prefix}] GPU selection ${idx_csv} → UUID-pinned (runtime-agnostic: classic nvidia + CDI, #610)" >&2
  else
    export CUDA_VISIBLE_DEVICES="$idx_csv"
    export NVIDIA_VISIBLE_DEVICES="$idx_csv"
  fi
}

# gpu_select_container_uuids "<container>"
#   The GPU UUIDs the container's COMPUTE PROCESSES are actually running on —
#   the runtime-agnostic ground truth of placement. Uses --query-compute-apps
#   (NOT --query-gpu): under CDI the container sees ALL cards via nvidia-smi,
#   but only RUNS on the CUDA-masked set, so the compute-apps view is what
#   reflects the real pinning. Echoes a sorted, unique UUID csv (empty if no
#   compute apps yet / nvidia-smi unavailable in the container).
gpu_select_container_uuids() {
  local container="$1"
  [[ -z "$container" ]] && return 0
  docker exec "$container" nvidia-smi \
      --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
    | tr -d '[:space:]' | grep '^GPU-' | sort -u | paste -sd, -
}

# gpu_select_assert_placement "<container>" "<requested_uuid_csv>" ["<log_prefix>"]
#   Post-boot placement assertion (#610 Phase A): compare where the model
#   ACTUALLY landed against what was requested. A loud warning on mismatch
#   instead of silent wrong-card serving — the failure mode that opened #610.
#   Non-fatal (returns 0): a warning, never a boot abort. Skips silently when
#   the request wasn't UUID-based or placement can't be read.
gpu_select_assert_placement() {
  local container="$1" requested="$2" log_prefix="${3:-gpu}" actual
  [[ -z "$container" || -z "$requested" ]] && return 0
  # Only meaningful when the request is UUIDs (a specific selection).
  [[ "$requested" == GPU-* ]] || return 0
  actual="$(gpu_select_container_uuids "$container")"
  [[ -z "$actual" ]] && return 0  # no compute apps yet / can't read — don't cry wolf
  # Requested must be a subset of actual (actual may list more if the app
  # spans them; a mismatch is a requested UUID that isn't running).
  local IFS=',' u miss=""
  read -ra _gpu_req <<< "$requested"
  for u in "${_gpu_req[@]}"; do
    [[ -z "$u" ]] && continue
    case ",${actual}," in
      *",${u},"*) : ;;
      *) miss="${miss:+${miss} }${u}" ;;
    esac
  done
  if [[ -n "$miss" ]]; then
    echo "[${log_prefix}] ⚠ PLACEMENT MISMATCH: requested GPU(s) not running the model — ${miss}" >&2
    echo "[${log_prefix}]   requested: ${requested}" >&2
    echo "[${log_prefix}]   actual:    ${actual}" >&2
    echo "[${log_prefix}]   On a CDI runtime this usually means CUDA_VISIBLE_DEVICES didn't reach the container (see docs/HARDWARE.md → Pinning specific GPUs)." >&2
  else
    echo "[${log_prefix}] ✓ placement verified — model running on the requested GPU(s)" >&2
  fi
  return 0
}
