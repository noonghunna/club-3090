#!/usr/bin/env bash
# test-thinking-pin.sh — the persisted per-model THINKING pin (#1014 follow-up).
#
# Contract:
#   - compose_registry.model_thinking_pin_key normalizes exactly like
#     model_default_pin_key: qwen3.6-27b → CLUB3090_THINKING_QWEN3_6_27B
#   - switch.sh thinking_pin_state resolves the (already-loaded) env value to
#     on | off | inherit; unknown/empty → inherit
#   - apply_thinking_pin_env injects ENABLE_THINKING=true/false for on/off and
#     NOTHING for inherit; an ENABLE_THINKING the shell already exports wins
#     (#425 precedence — the .env pin is file-tier defaulting, never an override)
#   - the launch path actually calls apply_thinking_pin_env (wiring seam)
#
# Hermetic: functions are extracted from switch.sh (the proven
# test-switch-orphan-teardown.sh pattern) so the script's main never runs; the
# .env side is exercised through the environment switch.sh loads at startup.
set -uo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail=0
note() { echo "FAIL: $1" >&2; fail=1; }
assert_eq() {
  local got="$1" want="$2" msg="$3"
  [[ "$got" == "$want" ]] || note "${msg}: got '${got}' want '${want}'"
}

# --- key normalization (compose_registry.model_thinking_pin_key) -------------
key="$(python3 -c "import sys; sys.path.insert(0,'$ROOT_DIR'); from scripts.lib.profiles.compose_registry import model_thinking_pin_key; print(model_thinking_pin_key('qwen3.6-27b'))")"
assert_eq "$key" "CLUB3090_THINKING_QWEN3_6_27B" "thinking pin key normalization"
key2="$(python3 -c "import sys; sys.path.insert(0,'$ROOT_DIR'); from scripts.lib.profiles.compose_registry import model_thinking_pin_key, model_default_pin_key; print(model_thinking_pin_key('a_b.c-d'))")"
assert_eq "$key2" "CLUB3090_THINKING_A_B_C_D" "thinking pin key non-alnum → _"
defkey="$(python3 -c "import sys; sys.path.insert(0,'$ROOT_DIR'); from scripts.lib.profiles.compose_registry import model_default_pin_key; print(model_default_pin_key('qwen3.6-27b'))")"
assert_eq "$defkey" "CLUB3090_DEFAULT_QWEN3_6_27B" "default pin key unchanged by the refactor"

# --- extract the switch.sh functions under test ------------------------------
HELPERS_FILE="$(mktemp --suffix=.sh)"
cleanup() { rm -f "$HELPERS_FILE"; }
trap cleanup EXIT
for fn in thinking_pin_key_for thinking_pin_state apply_thinking_pin_env; do
  sed -n "/^${fn}()/,/^}/p" scripts/switch.sh >> "$HELPERS_FILE"
done
# shellcheck source=/dev/null
source "$HELPERS_FILE"

# thinking_pin_key_for shells out to compose_registry via ROOT_DIR.
assert_eq "$(thinking_pin_key_for qwen3.6-27b)" "CLUB3090_THINKING_QWEN3_6_27B" \
  "thinking_pin_key_for mirrors compose_registry"

# --- thinking_pin_state: env resolution per value ----------------------------
KEY=CLUB3090_THINKING_QWEN3_6_27B
assert_eq "$(env "$KEY=on"   bash -c 'source "$1"; thinking_pin_state qwen3.6-27b' _ "$HELPERS_FILE")" "on"      "pin on → on"
assert_eq "$(env "$KEY=off"  bash -c 'source "$1"; thinking_pin_state qwen3.6-27b' _ "$HELPERS_FILE")" "off"     "pin off → off"
assert_eq "$(env "$KEY=inherit" bash -c 'source "$1"; thinking_pin_state qwen3.6-27b' _ "$HELPERS_FILE")" "inherit" "pin inherit → inherit"
assert_eq "$(env "$KEY=ON"   bash -c 'source "$1"; thinking_pin_state qwen3.6-27b' _ "$HELPERS_FILE")" "on"      "pin ON → on (case-insensitive)"
assert_eq "$(env "$KEY=bogus" bash -c 'source "$1"; thinking_pin_state qwen3.6-27b' _ "$HELPERS_FILE")" "inherit" "pin bogus → inherit (degrade, never crash)"
assert_eq "$(bash -c 'source "$1"; thinking_pin_state qwen3.6-27b' _ "$HELPERS_FILE")" "inherit" "pin unset → inherit"

# --- apply_thinking_pin_env: launch-env injection per value ------------------
out="$(env "$KEY=on" bash -c '
  source "$1"
  declare -A VARIANTS=( ["vllm/dual"]="vllm|models/qwen3.6-27b/vllm/compose|dual/quant/serving.yml" )
  apply_thinking_pin_env vllm/dual
  printf "%s\n" "${ENABLE_THINKING-<unset>}"
' _ "$HELPERS_FILE")"
assert_eq "$(tail -n1 <<<"$out")" "true" "apply: pin on → ENABLE_THINKING=true"

out="$(env "$KEY=off" bash -c '
  source "$1"
  declare -A VARIANTS=( ["vllm/dual"]="vllm|models/qwen3.6-27b/vllm/compose|dual/quant/serving.yml" )
  apply_thinking_pin_env vllm/dual
  printf "%s\n" "${ENABLE_THINKING-<unset>}"
' _ "$HELPERS_FILE")"
assert_eq "$(tail -n1 <<<"$out")" "false" "apply: pin off → ENABLE_THINKING=false (explicit)"

for v in inherit bogus; do
  out="$(env "$KEY=$v" bash -c '
    source "$1"
    declare -A VARIANTS=( ["vllm/dual"]="vllm|models/qwen3.6-27b/vllm/compose|dual/quant/serving.yml" )
    apply_thinking_pin_env vllm/dual
    printf "%s\n" "${ENABLE_THINKING-<unset>}"
  ' _ "$HELPERS_FILE")"
  assert_eq "$(tail -n1 <<<"$out")" "<unset>" "apply: pin $v → nothing injected"
done
out="$(bash -c '
  source "$1"
  declare -A VARIANTS=( ["vllm/dual"]="vllm|models/qwen3.6-27b/vllm/compose|dual/quant/serving.yml" )
  apply_thinking_pin_env vllm/dual
  printf "%s\n" "${ENABLE_THINKING-<unset>}"
' _ "$HELPERS_FILE")"
assert_eq "$(tail -n1 <<<"$out")" "<unset>" "apply: no pin → nothing injected"

# Shell wins (#425): an exported ENABLE_THINKING is never overridden by the pin.
for pin in on off; do
  out="$(env "$KEY=$pin" ENABLE_THINKING=true bash -c '
    source "$1"
    declare -A VARIANTS=( ["vllm/dual"]="vllm|models/qwen3.6-27b/vllm/compose|dual/quant/serving.yml" )
    apply_thinking_pin_env vllm/dual
    printf "%s" "${ENABLE_THINKING}"
  ' _ "$HELPERS_FILE")"
  assert_eq "$(tail -n1 <<<"$out")" "true" "apply: shell ENABLE_THINKING=true beats pin $pin"
done

# --- wiring: the launch path calls the applier -------------------------------
command grep -q 'apply_thinking_pin_env "\$v"' scripts/switch.sh \
  || note "up_variant does not call apply_thinking_pin_env (pin would never apply)"

[[ $fail -eq 0 ]] && echo "test-thinking-pin: ok" || echo "test-thinking-pin: FAIL"
exit $fail
