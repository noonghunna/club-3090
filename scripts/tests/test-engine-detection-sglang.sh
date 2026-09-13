#!/usr/bin/env bash
#
# Guard: an `sglang-*` container must resolve to ENGINE_KIND=sglang everywhere.
#
# Why this exists (club-3090#1261, @paulp83): a hand-rolled SGLang NVFP4 run
# printed `engine=unknown`, so verify-full's acceptance check fell through the
# `case "$ENGINE_KIND"` dispatch into the vLLM branch, found no "SpecDecoding
# metrics" line, and SKIPPED. The DFlash2 drafter was never verified alive —
# which is exactly the sglang#39087 false-clean (a garbage-drafting drafter
# leaves output correct and only collapses decode, silently).
#
# The cause was NOT the acceptance check added in #1249. It was detection:
# every container-name fallback listed `vllm-*` and `llama-cpp-*` and simply
# had no `sglang-*` arm, even though rebench-full.sh and club3090-env.sh both
# already use that prefix. So the #1249 SGLang branch was unreachable by
# auto-detection and had likely never fired.
#
# ⚠️ This test must FAIL against the pre-fix tree. If it passes before the fix,
# it is asserting the wrong thing.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
FAIL=0

ok()  { printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad() { printf "  \033[31m✗\033[0m %s\n" "$1"; printf "    expected=%s got=%s\n" "$2" "$3"; FAIL=1; }
check() { [[ "$2" == "$3" ]] && ok "$1" || bad "$1" "$2" "$3"; }

# --- 1 & 2: detect_engine() in verify-full.sh and verify-stress.sh -----------
# Extract just the function and drive it with a dead URL so hints 1 and 2 fail
# immediately (connection refused, not a timeout) and the container-name
# fallback — the arm that was missing — is what actually decides.
DEAD_URL="http://127.0.0.1:1"

for script in verify-full.sh verify-stress.sh; do
  # club-3090#1282: detect_engine now DELEGATES the prefix arms to the canonical
  # resolver, so the extracted body needs the lib in scope. Sourcing it here is
  # not a weakening of the control — the arms under test are the lib's, and this
  # test still drives the real extracted function against a dead URL.
  { echo "source '${ROOT}/scripts/lib/engine-kind.sh'"; \
    awk '/^detect_engine\(\) \{/,/^\}/' "${ROOT}/scripts/${script}"; } > "${TMP}/fn.sh"
  if [[ ! -s "${TMP}/fn.sh" ]]; then
    bad "${script}: detect_engine() not extractable" "a function body" "empty"
    continue
  fi
  # shellcheck disable=SC1090
  got="$(URL="$DEAD_URL" MODEL=x CONTAINER=sglang-qwen38-27b-nvfp4-dflash2-dual \
         bash -c "source '${TMP}/fn.sh'; detect_engine" 2>/dev/null)"
  check "${script}: sglang-* container → sglang" "sglang" "$got"

  # Negative control: the arms that already worked must keep working, and an
  # unrecognised name must still be honestly 'unknown' rather than defaulting.
  got="$(URL="$DEAD_URL" MODEL=x CONTAINER=vllm-qwen38-27b \
         bash -c "source '${TMP}/fn.sh'; detect_engine" 2>/dev/null)"
  check "${script}: vllm-* still → vllm" "vllm" "$got"
  got="$(URL="$DEAD_URL" MODEL=x CONTAINER=totally-unrecognised \
         bash -c "source '${TMP}/fn.sh'; detect_engine" 2>/dev/null)"
  check "${script}: unknown name still → unknown" "unknown" "$got"
done

# --- 3: calib_engine_for_container() in lib/report_calib.sh -----------------
# shellcheck source=../lib/report_calib.sh
source "${ROOT}/scripts/lib/report_calib.sh"
check "report_calib: sglang-* → sglang" "sglang" "$(calib_engine_for_container sglang-qwen38-27b-dual-fast)"
check "report_calib: vllm-* → vllm"     "vllm"   "$(calib_engine_for_container vllm-qwen38-27b)"
check "report_calib: junk → unknown"    "unknown" "$(calib_engine_for_container some-other-container)"

# --- 4: report.sh must RESPECT an explicit ENGINE_KIND=sglang override ------
# Its override guard reads `case "${ENGINE_KIND:-}" in vllm|llamacpp|unknown)`,
# so `sglang` fell through to the re-derive branch and was silently discarded.
if command grep -qE '^\s*vllm\|llamacpp\|sglang\|unknown\)' "${ROOT}/scripts/report.sh"; then
  ok "report.sh: honours ENGINE_KIND=sglang override"
else
  bad "report.sh: honours ENGINE_KIND=sglang override" "sglang in the override guard" "absent"
fi

# --- 5: bench.sh detection must RESOLVE sglang ------------------------------
# ⚠️ This used to grep bench.sh for a literal 'sglang' arm. club-3090#1282 moved
# the arms into scripts/lib/engine-kind.sh, so a text assertion would now fail
# on a correct tree — it was asserting the IMPLEMENTATION SHAPE, not the
# behaviour. Assert the outcome instead: bench.sh must source the canonical
# resolver, and that resolver must answer sglang for both evidence forms
# bench.sh actually has (image ref and container name).
if command grep -q 'lib/engine-kind.sh' "${ROOT}/scripts/bench.sh"; then
  ok "bench.sh: sources the canonical engine resolver"
else
  bad "bench.sh: sources the canonical engine resolver" "a source of lib/engine-kind.sh" "absent"
fi
got="$(bash -c "source '${ROOT}/scripts/lib/engine-kind.sh'; engine_kind_from_image lmsysorg/sglang:v0.5.19" 2>/dev/null)"
check "bench.sh path: sglang image → sglang" "sglang" "$got"
got="$(bash -c "source '${ROOT}/scripts/lib/engine-kind.sh'; engine_kind_from_container sglang-qwen38" 2>/dev/null)"
check "bench.sh path: sglang-* container → sglang" "sglang" "$got"

echo
if [[ "$FAIL" == "0" ]]; then
  printf "\033[32mtest-engine-detection-sglang: PASS\033[0m\n"
else
  printf "\033[31mtest-engine-detection-sglang: FAIL\033[0m\n"
fi
exit "$FAIL"
