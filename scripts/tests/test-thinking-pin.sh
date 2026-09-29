#!/usr/bin/env bash
# test-thinking-pin.sh — the persisted per-model THINKING pin (#1014 follow-up).
#
# Contract:
#   - compose_registry.model_thinking_pin_key normalizes exactly like
#     model_default_pin_key: qwen3.6-27b → CLUB3090_THINKING_QWEN3_6_27B
#   - the pin is one layer of the launch-settings resolver (scripts/lib/launch_settings.py,
#     #1465): on → ENABLE_THINKING=true, off → false (case-insensitive), inherit / unknown /
#     empty / unset → no layer (the compose default applies)
#   - precedence: an ENABLE_THINKING exported in the shell or saved for the slug wins over
#     the pin; the pin wins over a GLOBAL ENABLE_THINKING (maintainer decision 1, #1466)
#   - the launch line names where the pin came from: the settings file holding it
#     (club3090.env, or the legacy repo .env), or the shell
#   - the launch path applies the resolved settings (wiring seam)
#
# Hermetic: a fixture checkout (symlinks to this repo's code, its own .env) and a settings
# dir of our own — never the real settings or .env. Precedence against every other layer
# lives in test-launch-settings.sh.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

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

# --- the pin through the resolver ---------------------------------------------
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
# a launch writes the slug-label override (data dir) and cache dirs: never into your real ones
export CLUB3090_DATA_DIR="$T/data" CLUB3090_CACHE_DIR="$T/cache"
mkdir -p "$T/cfg" "$T/root"
for d in scripts models; do ln -s "$ROOT_DIR/$d" "$T/root/$d"; done
: > "$T/root/.env"
KEY=CLUB3090_THINKING_QWEN3_8_27B
SLUG=vllm/qwen38-27b-dual-fast          # a qwen3.8-27b slug whose compose reads ENABLE_THINKING
# thinking <club3090.env text> [VAR=val …] → "SOURCE|VALUE|DETAIL" of ENABLE_THINKING
thinking() {
  local club="$1"; shift
  printf '%s' "$club" > "$T/cfg/club3090.env"
  env -u "$KEY" -u ENABLE_THINKING CLUB3090_CONFIG_DIR="$T/cfg" "$@" \
    python3 scripts/lib/launch_settings.py explain --root "$T/root" --slug "$SLUG" \
    | python3 -c 'import json,sys; k={x["knob"]: x for x in json.load(sys.stdin)["knobs"]}["ENABLE_THINKING"]; print(k["source"], k["value"], k["detail"], sep="|")'
}
assert_eq "$(thinking "$KEY=on"$'\n' | cut -d'|' -f1-2)"      "model pin|true"        "pin on → ENABLE_THINKING=true"
assert_eq "$(thinking "$KEY=off"$'\n' | cut -d'|' -f1-2)"     "model pin|false"       "pin off → ENABLE_THINKING=false (explicit)"
assert_eq "$(thinking "$KEY=ON"$'\n' | cut -d'|' -f1-2)"      "model pin|true"        "pin ON → true (case-insensitive)"
assert_eq "$(thinking "$KEY=inherit"$'\n' | cut -d'|' -f1-2)" "compose default|true"  "pin inherit → nothing injected"
assert_eq "$(thinking "$KEY=bogus"$'\n' | cut -d'|' -f1-2)"   "compose default|true"  "pin bogus → nothing injected (degrade, never crash)"
assert_eq "$(thinking "" | cut -d'|' -f1-2)"                  "compose default|true"  "no pin → nothing injected"
# Shell wins (9a27de83): an exported ENABLE_THINKING is never overridden by the pin.
for pin in on off; do
  assert_eq "$(thinking "$KEY=$pin"$'\n' ENABLE_THINKING=true | cut -d'|' -f1-2)" "shell|true" "shell ENABLE_THINKING=true beats pin $pin"
done
# The pin beats a GLOBAL ENABLE_THINKING (it used to lose: the loader's export looked like the shell).
assert_eq "$(thinking "$KEY=off"$'\n'"ENABLE_THINKING=true"$'\n' | cut -d'|' -f1-2)" "model pin|false" "pin off beats a global ENABLE_THINKING=true"

# --- where the pin came from (#1466) -------------------------------------------
out="$(thinking "$KEY=on"$'\n' | cut -d'|' -f3)"
[[ "$out" == "$KEY=on from club3090.env" ]] || note "pin from club3090.env not labelled so: '$out'"
printf '%s=off\n' "$KEY" > "$T/root/.env"
out="$(thinking "" | cut -d'|' -f3)"
[[ "$out" == "$KEY=off from repo .env" ]] || note "pin from the legacy repo .env not labelled so: '$out'"
out="$(thinking "" "$KEY=on" | cut -d'|' -f2-3)"
[[ "$out" == "true|$KEY=on from shell" ]] || note "a pin exported in the shell (beating the file's off) not labelled so: '$out'"
: > "$T/root/.env"

# --- wiring: the launch path applies the resolved settings ---------------------
command grep -q 'apply_launch_settings "\$v"' scripts/switch.sh \
  || note "up_variant does not call apply_launch_settings (the pin would never apply)"
command grep -q '_launch_settings check --slug "\$v"' scripts/switch.sh \
  || note "check_variant does not run the launch-settings check"

[[ $fail -eq 0 ]] && echo "test-thinking-pin: ok" || echo "test-thinking-pin: FAIL"
exit $fail
