#!/usr/bin/env bash
# scripts/lib/report_calib.sh — helpers for report.sh's "KV math calibration"
# section (club-3090 #168). Pure functions: sourcing this file has no side
# effects, so it can be unit-tested directly (see scripts/tests/test-report-calib.sh).
#
# Why these exist:
#   - kv-calc's prediction model is vLLM-memory-model-coupled, so its
#     calibration is not a valid sanity-check on the llama.cpp / ik_llama
#     (ggml) engines — those must be skipped.
#   - `kv-calc.py --calibration` always prints the full catalog (all 4 models);
#     a bug-reporter should see only the model they're actually running.

# Map a running container name to its kv-calc engine family.
# Echoes: vllm | llamacpp | unknown
# "llamacpp" intentionally covers BOTH mainline llama.cpp and ik_llama — both
# use the ggml allocator, so kv-calc's vLLM memory model applies to neither.
calib_engine_for_container() {
  case "$1" in
    vllm-*)                 echo "vllm" ;;
    llama-cpp-*|ik-llama-*) echo "llamacpp" ;;
    sglang-*|sgl-*)         echo "sglang" ;;   # club-3090#1261
    *)                      echo "unknown" ;;
  esac
}

# Map a running container name to its kv-calc model id (a MODEL_SPECS key in
# tools/kv-calc.py, which is also the `== <id> ==` section header in
# `--calibration` output). Echoes the model id, or "" if unrecognized.
# Order matters: the more specific MoE names are matched before the dense ones.
calib_model_for_container() {
  case "$1" in
    *qwen36-35b-a3b*)  echo "qwen3.6-35b-a3b" ;;
    *gemma-4-26b-a4b*) echo "gemma-4-26b-a4b" ;;
    *gemma-4-31b*)     echo "gemma-4-31b" ;;
    *qwen36-27b*)      echo "qwen3.6-27b" ;;
    *)                 echo "" ;;
  esac
}

# Filter `kv-calc.py --calibration` output (stdin) to a single model's section.
# Keeps everything before the first "== " header (banner + legend), the matching
# "== <model> ==" block, and any trailing "Overall:" line; drops other models.
# Arg 1: model id. If empty, passes stdin through unchanged (full matrix).
calib_filter_model_section() {
  local model="$1"
  if [[ -z "$model" ]]; then cat; return; fi
  awk -v target="== ${model} ==" '
    BEGIN { before_first = 1 }
    /^== / { before_first = 0; in_section = ($0 == target) }
    {
      if (before_first)       { print; next }   # banner + legend
      if ($0 ~ /^Overall:/)   { print; next }   # global verdict line
      if (in_section)         { print }         # the wanted section only
    }
  '
}
