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

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

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
# qwen3.6-27b: single → vllm. ⚠️ 2026-08-12: ik-llama was removed from every walk
# (same treatment as beellama on 2026-07-27) once its last functional slug,
# ik-llama/iq4ks-mtp, was deprecated; every llama.cpp single-card qwen slug went
# with it, so llamacpp has no functional single target either. The walk is now
# ["llamacpp", "vllm"] and lands on vllm/minimal. dual → vllm; multi4 → vllm.
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b single 2>/dev/null)" \
  "vllm/minimal" "qwen single curated (ik-llama + llamacpp single slugs retired 2026-08-12)"
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b dual 2>/dev/null)" \
  "vllm/dual" "qwen dual curated"
out="$(model_default_target "$ROOT_DIR" qwen3.6-27b multi4 2>&1)"
assert_contains "$out" "falling back to the dual default" "qwen multi4 degradation notice (no multi4 vLLM slug post-#327)"
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b multi4 2>/dev/null)" \
  "vllm/dual" "qwen multi4 degrades to dual slug"
# gemma-4-31b dual → vllm/gemma-31b-dual (bf16 @224K, stock v0.24.0, overlay-free; the v0.22.0
# int8-PTH/bf16-mtp composes are now deprecated — see the v0.24.0 consolidation).
assert_eq "$(model_default_target "$ROOT_DIR" gemma-4-31b dual 2>/dev/null)" \
  "vllm/gemma-31b-dual" "gemmadual curated"

# --- (NA) skip + graceful degradation ---------------------------------------
# gemma-4-31b single: NO functional default since the 2026-07-27 beellama
# retirement (its vllm singles are deprecated too) — the resolver honestly
# degrades to pick-explicitly. This exercises the every-candidate-(NA) path.
if model_default_target "$ROOT_DIR" gemma-4-31b single >/dev/null 2>&1; then
  note "gemma-4-31b single unexpectedly resolved (no functional single since the beellama retirement)"
fi
assert_contains "$(model_default_target "$ROOT_DIR" gemma-4-31b single 2>&1)" \
  "pick a config explicitly" "gemma single curated → pick-explicitly (beellama retired)"
# qwen3.6-35b-a3b single: preview-only → (NA) → no functional default at single.
if model_default_target "$ROOT_DIR" qwen3.6-35b-a3b single >/dev/null 2>&1; then
  note "qwen-35b-a3b single unexpectedly resolved (all candidates are (NA))"
fi
# multi4 with no multi default → notice + nearest-lower (dual).
out="$(model_default_target "$ROOT_DIR" gemma-4-31b multi4 2>&1)"
assert_contains "$out" "falling back to the dual default" "gemma multi4 degradation notice"
assert_eq "$(model_default_target "$ROOT_DIR" gemma-4-31b multi4 2>/dev/null)" \
  "vllm/gemma-31b-dual" "gemmamulti4 degrades to dual slug"

# --- arch-gate: beellama DFlash default steers off non-sm_86 (#693) ----------
# ⚠️ Largely MOOT for qwen since 2026-08-12: with ik-llama and llamacpp holding no
# functional single-card qwen slug, every arch resolves to vllm/minimal, so these
# rows now assert arch-INVARIANCE rather than a steer. They are kept because the
# helper itself (warn_if_default_arch_gated) is still live for other models, and
# because a future functional ik/llamacpp single slug must re-establish the steer.
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b single 8.6 2>/dev/null)" \
  "vllm/minimal" "qwen single sm_8.6 → vllm/minimal (ik-llama retired from the walk 2026-08-12)"
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b single 8.9 2>/dev/null)" \
  "vllm/minimal" "qwen single sm_8.9 (Ada) → vllm/minimal; the #693 ik-llama steer is moot, ik-llama has no functional slug"
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b single 12.0 2>/dev/null)" \
  "vllm/minimal" "qwen single sm_12.0 (Blackwell) → vllm/minimal"
assert_eq "$(model_default_target "$ROOT_DIR" qwen3.6-27b single '' 2>/dev/null)" \
  "vllm/minimal" "qwen single sm unknown → vllm/minimal"
# gemma single has no other single default → off-arch degrades to pick-explicitly.
if model_default_target "$ROOT_DIR" gemma-4-31b single 8.6 >/dev/null 2>&1; then
  note "gemma single sm_8.6 unexpectedly resolved (beellama retired, no functional single)"
fi
if model_default_target "$ROOT_DIR" gemma-4-31b single 8.9 >/dev/null 2>&1; then
  note "gemma single sm_8.9 should have NO default (beellama gated, no fallback)"
fi
assert_contains "$(model_default_target "$ROOT_DIR" gemma-4-31b single 8.9 2>&1)" \
  "pick a config explicitly" "gemma single sm_8.9 → pick-explicitly"
# X/default dispatch threads the sm too.
assert_eq "$(x_default_dispatch "$ROOT_DIR" qwen3.6-27b/default single qwen3.6-27b 8.9 2>/dev/null)" \
  "vllm/minimal" "qwen/default X-dispatch on sm_8.9 → vllm/minimal"
assert_eq "$(x_default_dispatch "$ROOT_DIR" qwen3.6-27b/default single qwen3.6-27b 8.6 2>/dev/null)" \
  "vllm/minimal" "qwen/default X-dispatch on sm_8.6 → vllm/minimal"

# --- helpers: primary_sm_from_gpu_spec + warn_if_default_arch_gated -----------
assert_eq "$(primary_sm_from_gpu_spec '0|NVIDIA GeForce RTX 4090|24564|8.9;1|x|24564|8.9')" \
  "8.9" "primary_sm_from_gpu_spec extracts the first GPU sm"
assert_eq "$(primary_sm_from_gpu_spec '')" "" "primary_sm_from_gpu_spec empty → empty"
warn_out="$(warn_if_default_arch_gated "$ROOT_DIR" beellama/dflash 8.9 2>&1 >/dev/null)"
assert_contains "$warn_out" "arch-gate" "warn fires for beellama/dflash on sm_8.9"
assert_contains "$warn_out" "vllm/minimal" "warn recommends the functional single-card qwen slug"
assert_contains "$(warn_if_default_arch_gated "$ROOT_DIR" beellama/gemma-dflash 8.9 2>&1 >/dev/null)" \
  "No validated single-card default" "gemma warn → no-fallback message"
assert_eq "$(warn_if_default_arch_gated "$ROOT_DIR" beellama/dflash 8.6 2>&1)" \
  "" "warn silent for beellama/dflash on sm_8.6 (on-arch)"
assert_eq "$(warn_if_default_arch_gated "$ROOT_DIR" vllm/minimal 8.9 2>&1)" \
  "" "warn silent for a non-gated slug"

# --- X/default dispatch ------------------------------------------------------
# engine name → engine recommendation (back-compat).
assert_eq "$(x_default_dispatch "$ROOT_DIR" vllm/default single qwen3.6-27b 2>/dev/null)" \
  "vllm/minimal" "vllm/default single → vllm/minimal (Genesis tq3-mtp deprecated 2026-05-31)"
# ik-llama/default must now FAIL LOUDLY: every ik-llama qwen slug was deprecated
# 2026-08-12, and the DIRECT engine lookup does not status-filter — so the row was
# removed rather than repointed, and this must error instead of handing back a
# --force-only slug.
if out="$(x_default_dispatch "$ROOT_DIR" ik-llama/default single qwen3.6-27b 2>&1)"; then
  fail "ik-llama/default should error (no functional ik-llama single qwen slug), got: $out"
else
  # engine_set() = DEFAULTS engines ∪ ENGINE_PREFERENCE engines. Removing ik-llama
  # from BOTH drops it out of that set, so the token is rejected as unknown rather
  # than resolving-then-failing. Same shape as beellama/default since 2026-07-27.
  assert_contains "$out" "is neither a known engine nor a known model" \
    "ik-llama/default errors after the 2026-08-12 retirement (out of engine_set)"
fi
# model-id → model default (model token overrides the passed model).
assert_eq "$(x_default_dispatch "$ROOT_DIR" qwen3.6-27b/default single qwen3.6-27b 2>/dev/null)" \
  "vllm/minimal" "qwen3.6-27b/default model dispatch"
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
  # prism-pro-dq-dual is `deprecated` (since #956), not `experimental` — this
  # assertion was stale on master and failed there independently of this change.
  assert_contains "$out" "(NA: deprecated)" "(NA) pin warns"
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
  assert_eq "$slug" "vllm/minimal" "topology-mismatch pin falls back to single curated" )
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
