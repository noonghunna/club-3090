#!/usr/bin/env bash
# PR-B — model-default resolver + user-pinnable defaults.
#
# Exercises the shared resolver (registry-emit.sh model_default_target /
# x_default_dispatch) + switch.sh --set-default/--clear-default round-trip:
#   - curated walk picks the first FUNCTIONAL DEFAULTS slug per ENGINE_PREFERENCE
#   - (NA) candidates are skipped (never auto-default a broken config)
#   - X/default dispatch: engine name → engine rec; model-id → model default;
#     unknown → error; precedence is explicit
#   - .env pin overrides; invalid / (NA) / topology-mismatch pin → warn + fall
#     back to curated (never blocks)
#   - degradation: no functional default at the detected topology → notice +
#     nearest-lower topology, else a clear "pick explicitly" message (no crash)
#   - community seam returns None today → skipped
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail=0
note() { echo "FAIL: $1" >&2; fail=1; }

assert_eq() {
  local got="$1" want="$2" msg="$3"
  [[ "$got" == "$want" ]] || note "${msg}: got '${got}' want '${want}'"
}
assert_contains() {
  local hay="$1" needle="$2" msg="$3"
  [[ "$hay" == *"$needle"* ]] || note "${msg}: '${hay}' lacks '${needle}'"
}

# shellcheck source=../lib/registry-emit.sh
source "$ROOT_DIR/scripts/lib/registry-emit.sh"

# --- curated walk (no pin) ---------------------------------------------------
# qwen3.6-27b: single → beellama (ranked #1 in ENGINE_PREFERENCE; promoted to a
# functional `caveats` DEFAULTS entry 2026-05-30, so the resolver now picks it
# ahead of ik-llama); dual → vllm; multi4 → vllm.
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b single 2>/dev/null)" \
  "beellama/dflash" "qwen single curated"
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b dual 2>/dev/null)" \
  "vllm/dual" "qwen dual curated"
out="$(model_default_target "$ROOT_DIR" qwen3.6-27b multi4 2>&1)"
assert_contains "$out" "falling back to the dual default" "qwen multi4 degradation notice (no multi4 vLLM slug post-#327)"
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b multi4 2>/dev/null)" \
  "vllm/dual" "qwen multi4 degrades to dual slug"
# gemma-4-31b dual → vllm/gemma-int8-mtp (full-ctx default; gemma-mtp is the stable fallback).
assert_eq "$(model_default_target "$ROOT_DIR" gemma-4-31b dual 2>/dev/null)" \
  "vllm/gemma-int8-mtp" "gemma dual curated"

# --- (NA) skip + graceful degradation ---------------------------------------
# gemma-4-31b single → beellama/gemma-dflash. vllm/gemma-mtp-tp1 is upstream-gated/(NA),
# so the resolver SKIPS it and picks the next functional engine — beellama, #1 in
# ENGINE_PREFERENCE[single] (promoted to the single-card default 2026-05-30). This
# exercises the (NA)-skip-then-fall-through path with a real fallback present.
assert_eq "$(model_default_target "$ROOT_DIR" gemma-4-31b single 2>/dev/null)" \
  "beellama/gemma-dflash" "gemma single curated (vllm NA → beellama)"
# qwen3.6-35b-a3b single: preview-only → (NA) → no functional default at single.
if model_default_target "$ROOT_DIR" qwen3.6-35b-a3b single >/dev/null 2>&1; then
  note "qwen-35b-a3b single unexpectedly resolved (all candidates are (NA))"
fi
# multi4 with no multi default → notice + nearest-lower (dual).
out="$(model_default_target "$ROOT_DIR" gemma-4-31b multi4 2>&1)"
assert_contains "$out" "falling back to the dual default" "gemma multi4 degradation notice"
assert_eq "$(model_default_target "$ROOT_DIR" gemma-4-31b multi4 2>/dev/null)" \
  "vllm/gemma-int8-mtp" "gemma multi4 degrades to dual slug"

# --- X/default dispatch ------------------------------------------------------
# engine name → engine recommendation (back-compat).
assert_eq "$(x_default_dispatch "$ROOT_DIR" vllm/default single qwen3.6-27b 2>/dev/null)" \
  "vllm/minimal" "vllm/default single → vllm/minimal (Genesis tq3-mtp deprecated 2026-05-31)"
assert_eq "$(x_default_dispatch "$ROOT_DIR" ik-llama/default single qwen3.6-27b 2>/dev/null)" \
  "ik-llama/iq4ks-mtp" "ik-llama/default engine dispatch"
# model-id → model default (model token overrides the passed model).
assert_eq "$(x_default_dispatch "$ROOT_DIR" qwen3.6-27b/default single qwen3.6-27b 2>/dev/null)" \
  "beellama/dflash" "qwen3.6-27b/default model dispatch"
# unknown → error, lists both sets.
if out="$(x_default_dispatch "$ROOT_DIR" bogus/default single qwen3.6-27b 2>&1)"; then
  note "bogus/default unexpectedly resolved to '${out}'"
else
  assert_contains "$out" "neither a known engine nor a known model" "unknown dispatch error"
fi
# engines + models are disjoint (precedence is unambiguous).
disjoint="$(python3 -c "import sys; sys.path.insert(0,'$ROOT_DIR'); from scripts.lib.profiles.compose_registry import engine_set, model_set; print('ok' if not (engine_set() & model_set()) else 'overlap')")"
assert_eq "$disjoint" "ok" "engine/model namespaces disjoint"

# --- .env pin: override + validation -----------------------------------------
# The resolver reads the pin from the *environment* (callers load .env first),
# so the pin is exercised by exporting the key in a subshell.
PIN=CLUB3090_DEFAULT_QWEN3_6_27B
# Valid pin on matching topology → honoured.
( export "$PIN=vllm/dual"; assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b dual 2>/dev/null)" "vllm/dual" "valid pin honoured" )
# (NA) pin → warn + fall back to curated.
( export "$PIN=ik-llama/prism-pro-dq-dual"
  out="$(model_default_target "$ROOT_DIR" qwen3.6-27b dual 2>&1 1>/dev/null)"
  slug="$(model_default_target "$ROOT_DIR" qwen3.6-27b dual 2>/dev/null)"
  assert_contains "$out" "(NA: experimental)" "(NA) pin warns"
  assert_eq "$slug" "vllm/dual" "(NA) pin falls back to curated" )
# wrong-model pin → warn + fall back.
( export "$PIN=vllm/gemma-bf16-mtp"
  out="$(model_default_target "$ROOT_DIR" qwen3.6-27b dual 2>&1 1>/dev/null)"
  slug="$(model_default_target "$ROOT_DIR" qwen3.6-27b dual 2>/dev/null)"
  assert_contains "$out" "belongs to model 'gemma-4-31b'" "wrong-model pin warns"
  assert_eq "$slug" "vllm/dual" "wrong-model pin falls back" )
# topology-mismatch pin → warn + fall back to the detected topology's curated.
( export "$PIN=vllm/dual"
  out="$(model_default_target "$ROOT_DIR" qwen3.6-27b single 2>&1 1>/dev/null)"
  slug="$(model_default_target "$ROOT_DIR" qwen3.6-27b single 2>/dev/null)"
  assert_contains "$out" "this rig is single" "topology-mismatch pin warns"
  assert_eq "$slug" "beellama/dflash" "topology-mismatch pin falls back to single curated" )
# unknown-slug pin → warn + fall back.
( export "$PIN=vllm/nope"
  out="$(model_default_target "$ROOT_DIR" qwen3.6-27b dual 2>&1 1>/dev/null)"
  assert_contains "$out" "not a known slug" "unknown-slug pin warns" )

# --- community seam: returns None today → skipped ----------------------------
community="$(python3 -c "import sys; sys.path.insert(0,'$ROOT_DIR'); from scripts.lib.profiles.compose_registry import community_default_target; print(community_default_target('qwen3.6-27b','dual'))")"
assert_eq "$community" "None" "community_default_target stub returns None"
# Sanity: with no pin, the resolver result equals the curated walk (i.e. the
# community rung is currently transparent / skipped).
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b dual 2>/dev/null)" \
  "$(python3 -c "import sys; sys.path.insert(0,'$ROOT_DIR'); from scripts.lib.profiles.compose_registry import curated_default_target; print(curated_default_target('qwen3.6-27b','dual'))")" \
  "community rung skipped (resolver == curated when no pin)"

# --- .env pin key normalization (design §13.2) -------------------------------
key="$(python3 -c "import sys; sys.path.insert(0,'$ROOT_DIR'); from scripts.lib.profiles.compose_registry import model_default_pin_key; print(model_default_pin_key('qwen3.6-27b'))")"
assert_eq "$key" "CLUB3090_DEFAULT_QWEN3_6_27B" "pin key normalization"

# --- switch.sh --set-default / --clear-default round-trip --------------------
# --set-default / --clear-default write ROOT_DIR/.env. ROOT_DIR is derived from
# the script's own BASH_SOURCE, so the round-trip is exercised against the repo
# .env, saved + restored around the test (it's gitignored either way).
SAVED_ENV=""
if [[ -f "$ROOT_DIR/.env" ]]; then SAVED_ENV="$(mktemp)"; cp "$ROOT_DIR/.env" "$SAVED_ENV"; fi
cleanup() {
  if [[ -n "$SAVED_ENV" ]]; then cp "$SAVED_ENV" "$ROOT_DIR/.env"; rm -f "$SAVED_ENV";
  else rm -f "$ROOT_DIR/.env"; fi
}
trap cleanup EXIT

rm -f "$ROOT_DIR/.env"
bash "$ROOT_DIR/scripts/switch.sh" --set-default vllm/dual >/dev/null 2>&1
grep -q "^CLUB3090_DEFAULT_QWEN3_6_27B=vllm/dual$" "$ROOT_DIR/.env" \
  || note "--set-default did not write the pin key/value"
# Resolve through the script's loaded .env on a dual rig → honour the pin.
out="$(NVIDIA_VISIBLE_DEVICES=0,1 bash "$ROOT_DIR/scripts/switch.sh" --defaults 2>&1)"
assert_contains "$out" "vllm/dual" "set-default reflected in --defaults"
assert_contains "$out" "[pin]" "set-default marked as [pin] in --defaults"
# Clear → key removed, round-trips.
bash "$ROOT_DIR/scripts/switch.sh" --clear-default qwen3.6-27b >/dev/null 2>&1
if grep -q "CLUB3090_DEFAULT_QWEN3_6_27B" "$ROOT_DIR/.env" 2>/dev/null; then
  note "--clear-default did not remove the pin key"
fi
# Invalid slug → rejected, no .env write.
rm -f "$ROOT_DIR/.env"
if bash "$ROOT_DIR/scripts/switch.sh" --set-default vllm/not-a-real-slug >/dev/null 2>&1; then
  note "--set-default accepted an unknown slug"
fi
[[ -f "$ROOT_DIR/.env" ]] && note "--set-default wrote .env for an unknown slug"

if [[ "$fail" -ne 0 ]]; then
  echo "[model-default-resolver] FAIL" >&2
  exit 1
fi
echo "test-model-default-resolver: ok"
