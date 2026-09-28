#!/usr/bin/env bash
# Unit test for scripts/lib/report_calib.sh (club-3090 #168):
# container→engine, container→model, and the per-model calibration filter.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/scripts/lib/report_calib.sh"

fail=0
check() {  # check <label> <expected> <actual>
  if [[ "$2" != "$3" ]]; then
    echo "FAIL: $1 — expected [$2], got [$3]" >&2
    fail=1
  fi
}

# --- container → engine family ------------------------------------------------
check "engine vllm"          "vllm"     "$(calib_engine_for_container vllm-qwen36-27b-dual)"
check "engine llama.cpp"     "llamacpp" "$(calib_engine_for_container llama-cpp-qwen36-27b)"
check "engine llama.cpp-vis" "llamacpp" "$(calib_engine_for_container llama-cpp-qwen36-27b-vision)"
check "engine ik_llama"      "llamacpp" "$(calib_engine_for_container ik-llama-qwen36-27b)"        # #168: was 'unknown' → ran calibration
check "engine ik_llama-2stg" "llamacpp" "$(calib_engine_for_container ik-llama-qwen36-27b-two-stage)"
check "engine unknown"       "unknown"  "$(calib_engine_for_container some-other-container)"
check "engine empty"         "unknown"  "$(calib_engine_for_container '')"

# --- container → kv-calc model id --------------------------------------------
check "model qwen-27b dense"  "qwen3.6-27b"     "$(calib_model_for_container vllm-qwen36-27b-dual-turbo)"
check "model qwen-27b llama"  "qwen3.6-27b"     "$(calib_model_for_container llama-cpp-qwen36-27b)"
check "model qwen-27b ik"     "qwen3.6-27b"     "$(calib_model_for_container ik-llama-qwen36-27b-vision)"
check "model qwen-35b-a3b"    "qwen3.6-35b-a3b" "$(calib_model_for_container vllm-qwen36-35b-a3b-dual)"
check "model gemma-31b"       "gemma-4-31b"     "$(calib_model_for_container vllm-gemma-4-31b-mtp)"
check "model gemma-26b-a4b"   "gemma-4-26b-a4b" "$(calib_model_for_container vllm-gemma-4-26b-a4b-awq-tp2)"
check "model unknown"         ""                "$(calib_model_for_container mystery-box)"

# --- per-model section filter -------------------------------------------------
fixture=$'========================================================================================\nCalibration — predicted per-card VRAM vs measured BENCHMARKS rows\n========================================================================================\n\n  Predicted = weights + activation + overhead.\n\n== qwen3.6-27b ==\n  dual          19.91 GB   PASS\n  Verdict accuracy: 11/11 (100%)\n\n== gemma-4-31b ==\n  gemma-dual-int8   22.80 GB   TIGHT\n  Verdict accuracy: 4/4 (100%)\n\nOverall: 17/17 (100%)'

scoped="$(printf '%s\n' "$fixture" | calib_filter_model_section qwen3.6-27b)"
case "$scoped" in
  *"== qwen3.6-27b =="*) ;;
  *) echo "FAIL: filter dropped the target section header" >&2; fail=1 ;;
esac
case "$scoped" in
  *"gemma-4-31b"*) echo "FAIL: filter leaked the gemma section" >&2; fail=1 ;;
esac
case "$scoped" in
  *"Calibration — predicted"*) ;;
  *) echo "FAIL: filter dropped the banner/legend" >&2; fail=1 ;;
esac
case "$scoped" in
  *"Overall: 17/17"*) ;;
  *) echo "FAIL: filter dropped the Overall line" >&2; fail=1 ;;
esac
check "filter keeps target rows" "1" "$(printf '%s\n' "$scoped" | grep -c 'dual          19.91')"

# empty model id → passthrough (full matrix)
full="$(printf '%s\n' "$fixture" | calib_filter_model_section '')"
check "empty filter = passthrough" "$(printf '%s\n' "$fixture" | wc -l)" "$(printf '%s\n' "$full" | wc -l)"

# --- #1076: a model with NO section must not borrow the global verdict ---------
# foureight84 hit this on a qwen3.8-27b multi4 boot: report.sh --full printed
# "Overall: 8/8" and "No FAIL rows" while calibrating only qwen3.6-27b /
# 35b-a3b / agentworld / gemma-4-31b rows, because multi4-ultramax has no
# measured BENCHMARKS anchor. A clean verdict for a model nothing checked.

# POSITIVE CONTROL — the bug itself. Asserted first: a gate that cannot detect
# the defect it was written for is theatre.
absent="$(printf '%s\n' "$fixture" | calib_filter_model_section qwen3.8-27b)"
check "absent model emits the marker" "1" "$(printf '%s\n' "$absent" | grep -c '__CALIB_NO_SECTION__')"
check "absent model does NOT borrow Overall" "0" "$(printf '%s\n' "$absent" | grep -cE '^Overall:')"
check "absent model leaks no foreign rows" "0" "$(printf '%s\n' "$absent" | grep -c '19.91')"

# NEGATIVE CONTROL — a present model keeps its rows AND the global verdict, and
# is not flagged. Suppressing both would be worse than the bug.
check "present model keeps Overall" "1" "$(printf '%s\n' "$scoped" | grep -cE '^Overall:')"
check "present model is not flagged" "0" "$(printf '%s\n' "$scoped" | grep -c '__CALIB_NO_SECTION__')"

# passthrough (empty id) must be unaffected by the marker logic
check "passthrough not flagged" "0" "$(printf '%s\n' "$full" | grep -c '__CALIB_NO_SECTION__')"

if [[ "$fail" -ne 0 ]]; then
  echo "test-report-calib: FAILED" >&2
  exit 1
fi
echo "test-report-calib: ok"
