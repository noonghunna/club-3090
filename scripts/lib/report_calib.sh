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
# Delegates to the canonical resolver (club-3090#1282) — do NOT re-inline the
# prefix arms here; scripts/lib/engine-kind.sh is the single source of truth.
# Dot decimals whatever the user's locale: under es_ES / de_DE mawk read "1.65" as 1 and bash printf rejected "9.0" (#1926, #159).
# LC_ALL would override LC_NUMERIC, so its value moves to the other categories first. Guarded by test-numeric-locale.sh.
[[ -z "${LC_ALL:-}" ]] || { for _lc in LANG LC_CTYPE LC_COLLATE LC_MESSAGES LC_TIME LC_MONETARY; do export "$_lc=$LC_ALL"; done; unset LC_ALL _lc; }; export LC_NUMERIC=C
calib_engine_for_container() {
  if ! declare -F engine_kind_from_container >/dev/null 2>&1; then
    # shellcheck source=engine-kind.sh
    source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/engine-kind.sh"
  fi
  engine_kind_from_container "$1"
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
# ⚠️⚠️ The `Overall:` line is GLOBAL — it is the verdict across every calibrated
# model, not the scoped one. Printing it unconditionally while the target model
# has NO section produced a false clean: report.sh read the borrowed verdict,
# found no FAIL rows (because no rows were emitted at all), and asserted "No FAIL
# rows. kv-calc projections should agree with measured VRAM" for a model that was
# never evaluated. Reported by foureight84 on discussion #1076: `report.sh --full`
# scored 8/8 on a qwen3.8-27b multi4 boot while calibrating only qwen3.6-27b /
# 35b-a3b / agentworld / gemma-4-31b rows, because multi4-ultramax has no measured
# BENCHMARKS.md anchor.
#
# Emits a machine-readable marker when the scoped model has no section, so the
# caller can refuse to report a verdict it did not earn.
CALIB_NO_SECTION_MARKER="__CALIB_NO_SECTION__"

calib_filter_model_section() {
  local model="$1"
  if [[ -z "$model" ]]; then cat; return; fi
  awk -v target="== ${model} ==" -v marker="__CALIB_NO_SECTION__" '
    BEGIN { before_first = 1; seen = 0 }
    /^== / { before_first = 0; in_section = ($0 == target); if (in_section) seen = 1 }
    {
      if (before_first)       { print; next }   # banner + legend
      if ($0 ~ /^Overall:/)   { overall = $0; next }  # HELD, not printed yet
      if (in_section)         { print }         # the wanted section only
    }
    END {
      # Only surface the global verdict when this model actually contributed rows.
      if (seen) { if (overall != "") print overall }
      else      { print marker }
    }
  '
}
