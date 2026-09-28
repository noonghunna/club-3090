# cuda-ordinals.sh — sourced INSIDE engine containers (mounted at
# /etc/club3090/cuda-ordinals.sh). Not executable on its own.
#
# WHY THIS EXISTS (club-3090#1462, @ensingerphilipp)
# --------------------------------------------------
# Estate launches pin GPUs by UUID (CUDA_VISIBLE_DEVICES=GPU-…,GPU-…), which is the
# right namespace for choosing physical cards. SGLang's custom all-reduce parses
# CUDA_VISIBLE_DEVICES with int() (custom_all_reduce_utils.py, cuda_vmm_utils.py —
# still true in v0.5.20; sglang#25996 closed unmerged), fails, logs one warning and
# falls back to NCCL — while the boot args still say disable_custom_all_reduce=False.
#
# club3090_cuda_ordinals rewrites a UUID list into the container's own GPU indices,
# in the order given, resolved against what THIS container can see (nvidia-smi
# inside it), not assumed to be 0..N-1. It also sets CUDA_DEVICE_ORDER=PCI_BUS_ID:
# SGLang looks each ordinal up with nvmlDeviceGetHandleByIndex (PCI order), and CUDA
# counts FASTEST_FIRST by default — the two must agree for an index to name a GPU.
#
# It never fails the boot: unset or already-numeric input is left alone, and anything
# it cannot resolve is left exactly as it was, with a warning. Always returns 0.

club3090_cuda_ordinals() {
  local cvd="${CUDA_VISIBLE_DEVICES:-}" map entry idx out=""
  [ -n "$cvd" ] || return 0
  case "$cvd" in
    *[!0-9,]*) ;;          # contains something other than digits and commas
    *) return 0 ;;         # already ordinals
  esac
  if ! command -v nvidia-smi >/dev/null 2>&1 \
     || ! map="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)" \
     || [ -z "$map" ]; then
    echo "[gpus] CUDA_VISIBLE_DEVICES=$cvd left as is: cannot list this container's GPUs (nvidia-smi). SGLang custom all-reduce cannot parse UUIDs and will fall back to NCCL (club-3090#1462)." >&2
    return 0
  fi
  local IFS=','
  for entry in $cvd; do
    entry="${entry//[[:space:]]/}"
    case "$entry" in
      "" ) continue ;;
      *[!0-9]*) idx="$(printf '%s\n' "$map" | awk -F', *' -v u="$entry" '$2 == u { print $1; exit }')" ;;
      *) idx="$entry" ;;
    esac
    if [ -z "$idx" ]; then
      echo "[gpus] CUDA_VISIBLE_DEVICES=$cvd left as is: '$entry' is not a GPU this container can see (MIG instances and partial UUIDs are not remapped). SGLang custom all-reduce will fall back to NCCL (club-3090#1462)." >&2
      return 0
    fi
    out="${out:+$out,}$idx"
  done
  [ -n "$out" ] || return 0
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$out"
  echo "[gpus] CUDA_VISIBLE_DEVICES $cvd -> $out (this container's ordinals, CUDA_DEVICE_ORDER=PCI_BUS_ID; club-3090#1462)" >&2
  return 0
}
