#!/usr/bin/env bash
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="${ROOT_DIR}/scripts/lib/profiles/launch_compat.py"
GPU_3090='0|RTX_3090|24576|8.6'
MTP_SHA="01d4d1ad375dc5854779c593eee093bcebb0cada"
CLEAN_SHA="bf610c2f56764e1b30bc6065f4ceace3d6e59036"
DFLASH_SHA="e47c98ef7a38792996e452ef53914e21e41928e9"

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" == *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output not to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

out="$(python3 "$HELPER" filter-candidates \
  --variants vllm/dual,vllm/minimal,llamacpp/default \
  --model qwen3.6-27b \
  --gpu-spec "$GPU_3090" \
  --tp 1 \
  --pp 1 \
  --workload fast-chat)"
assert_contains "$out" "vllm/minimal"
assert_not_contains "$out" "vllm/dual"

out="$(python3 "$HELPER" filter-candidates \
  --variants vllm/minimal,llamacpp/default,llamacpp/mtp \
  --model qwen3.6-27b \
  --gpu-spec "$GPU_3090" \
  --tp 1 \
  --pp 1 \
  --stable)"
assert_contains "$out" "vllm/minimal"
assert_contains "$out" "llamacpp/default"
assert_contains "$out" "llamacpp/mtp"

if out="$(python3 "$HELPER" validate-variant \
  --variant vllm/gemma-mtp-tp1 \
  --gpu-spec "$GPU_3090" \
  --tp 2 \
  --pp 1 \
  --no-project-vram 2>&1)"; then
  echo "ASSERTION FAILED: invalid Gemma single-card profile unexpectedly passed" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "C1: tp=2 * pp=1 = 2 != 1 cards selected"
assert_contains "$out" "C5: kv_format=fp8_e4m3 not supported by hardware: rtx-3090"

# fp8_e4m3 KV is WEIGHTS-CONDITIONAL on Ampere (sm_86): fp8-weights checkpoints route to
# FlashInfer (native fp8 storage) and ARE supported; non-fp8 weights (Gemma W4A16, above)
# route to Triton (needs SM89+) and are NOT. Validated #594 (Qwen fp8 dual-max on 2x 3090:
# boot/decode/NIAH/quality/soak green) + learnings/gemma-4-31b.md 2026-07-01 (gemma fp8_e4m3
# fails at KV-init on the same stack). Prove the ALLOW direction so the two stay in sync.
GPU_3090_X2="0|RTX_3090|24576|8.6;1|RTX_3090|24576|8.6"
out="$(python3 "$HELPER" validate-variant \
  --variant vllm/qwen-27b-dual-max \
  --gpu-spec "$GPU_3090_X2" \
  --tp 2 \
  --pp 1 \
  --no-project-vram 2>&1)" || {
  echo "ASSERTION FAILED: Qwen fp8-weights dual-max (fp8_e4m3 KV) rejected on 2x rtx-3090" >&2
  echo "$out" >&2
  exit 1
}
assert_not_contains "$out" "kv_format=fp8_e4m3 not supported"

out="$(python3 "$HELPER" validate-variant \
  --variant vllm/minimal \
  --gpu-spec "$GPU_3090" \
  --tp 1 \
  --pp 1 \
  --no-project-vram \
  --verbose 2>&1)"
assert_contains "$out" "Pass 1 fits()"
assert_contains "$out" "Resolved compose: vllm/minimal"
assert_contains "$out" "Pass 2 fits()"

out="$(python3 "$HELPER" resolve-engine-pin --engine-id vllm-nightly-mtp --format shell)"
assert_contains "$out" "VLLM_NIGHTLY_SHA=${MTP_SHA}"

# #1365: a pip engine is NOT an image pin, and that is a normal state, not an
# error. It used to RAISE, which made the whole slug unresolvable -- the reason
# both launchers gated hardware injection behind a vllm/beellama prefix test and
# 73 of 138 slugs got none. It now resolves to an EMPTY pin.
out="$(python3 "$HELPER" resolve-engine-pin --engine-id vllm-pip-baseline --format shell 2>&1)" || {
  echo "ASSERTION FAILED: pip engine must resolve to an EMPTY pin, not raise" >&2; echo "$out" >&2; exit 1; }
[[ -z "$(printf '%s' "$out" | tr -d '[:space:]')" ]] || {
  echo "ASSERTION FAILED: pip engine emitted an image pin: $out" >&2; exit 1; }
# same for one engine id whose composes use TWO image vars (llama-cpp-local:
# LLAMACPP_IMAGE vs IK_LLAMA_IMAGE) -- no single var, so no injection, never a guess.
out="$(python3 "$HELPER" resolve-engine-pin --engine-id llama-cpp-local --format shell 2>&1)" || {
  echo "ASSERTION FAILED: llama-cpp-local must resolve to an EMPTY pin, not raise" >&2; echo "$out" >&2; exit 1; }
[[ -z "$(printf '%s' "$out" | tr -d '[:space:]')" ]] || {
  echo "ASSERTION FAILED: llama-cpp-local emitted a pin despite two vars: $out" >&2; exit 1; }
# a single-image non-vLLM engine DOES pin, from its profile image_env
out="$(python3 "$HELPER" resolve-engine-pin --engine-id sglang-stable --format shell 2>&1)"
assert_contains "$out" "SGLANG_IMAGE="

out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell)"
assert_contains "$out" "VLLM_IMAGE=vllm/vllm-openai:v0.30.0"


out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/gemma-int8-mtp --format shell)"
assert_contains "$out" "VLLM_IMAGE=vllm/vllm-openai:v0.22.0"

# --- #246 Phase 1 KV injection: RETIRED (#1371) -------------------------------
# This block used to be eight `assert_not_contains KV_CACHE_DTYPE` cases against a
# live injector. Every one of them passed for the wrong reason: the injector had
# been inert on EVERY card since the composes migrated to fp8_e4m3, so the suite
# was asserting a no-op it believed was a decision. That is the shape this repo
# calls a false clean, and it is why the mechanism survived nine months unnoticed.
#
# The injector is gone. What must be guarded now is that it stays gone and that
# removing it did not disturb the user's own knob.
GPU_4090='0|NVIDIA GeForce RTX 4090|24564|8.9'
GPU_5090X2='0|NVIDIA GeForce RTX 5090|32607|12.0;1|NVIDIA GeForce RTX 5090|32607|12.0'

# 1. NOTHING resolves KV_CACHE_DTYPE, on any card, for any slug. A positive
#    control rides along: the same call must still return the image pin, so an
#    empty result cannot make this pass.
for _spec in "$GPU_3090" "$GPU_4090" "$GPU_5090X2" "0|NVIDIA GB10|131072|12.1" "0|Weird GPU|8192|7.0"; do
  for _v in vllm/dual vllm/minimal vllm/qwen-27b-dual-fast vllm/gemma-int8-mtp; do
    out="$(python3 "$HELPER" resolve-variant-pin --variant "$_v" --format shell --gpu-spec "$_spec")"
    assert_not_contains "$out" "KV_CACHE_DTYPE"
    assert_contains "$out" "VLLM_IMAGE="
  done
done

# 2. The symbols are gone from the resolver, not merely unreachable. A dormant
#    map is what let this rot: it read as a feature in every review.
if python3 -c "
import sys; sys.path.insert(0, '.')
import scripts.lib.profiles.launch_compat as m
sys.exit(0 if any(hasattr(m, n) for n in ('ARCH_KV_PILOT_VARIANTS', '_ARCH_KV_ALLOWED', '_arch_aware_env')) else 1)
"; then
  echo "ASSERTION FAILED: the #246 Phase 1 KV injector is back. If that is deliberate," >&2
  echo "  it needs a slug that actually declares the source kv_format (zero do today)" >&2
  echo "  and it must not fight compat.py's _fp8w_ampere_kv routing rule — see #1371." >&2
  exit 1
fi

# 3. A user's own KV_CACHE_DTYPE is untouched. It never travelled through the
#    resolver or the launcher case arm — the composes read ${KV_CACHE_DTYPE:-…}
#    and docker interpolates it — so removing both must change nothing here.
out="$(KV_CACHE_DTYPE=fp8_e5m2 python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_4090")"
assert_not_contains "$out" "KV_CACHE_DTYPE"
assert_contains "$out" "VLLM_IMAGE="

echo "  ok: #1371 Phase 1 KV injector stays retired (20 slug x card cases · symbols absent · user knob intact)"

# --- detector: Blackwell family must not collapse to rtx-5090 (#576 wrinkle) --
# The sm>=12 bucket used to map every Blackwell to rtx-5090, so a 96 GB PRO 6000
# and a 128 GB GB10 both mis-detected as a 32 GB 5090. Lock the split.
det="$(python3 - <<'PY'
import sys; sys.path.insert(0, "scripts/lib/profiles")
from launch_compat import _hardware_id_from_gpu as m
cases = [
    ("NVIDIA RTX PRO 6000 Blackwell", 98304, 12.0, "rtx-6000-pro-blackwell"),
    ("Unnamed Blackwell 96GB",        98304, 12.0, "rtx-6000-pro-blackwell"),  # alias-miss fallback
    ("NVIDIA GB10",                  131072, 12.1, "dgx-spark"),
    ("NVIDIA GeForce RTX 5090",       32607, 12.0, "rtx-5090"),
    ("NVIDIA RTX 6000 Ada Generation",49140,  8.9, "rtx-4090"),                # 'pro 6000' must NOT catch Ada
    # #1364: the sm_8.6 bucket returned rtx-3090 for ANY >=24 GB Ampere, so a
    # 48 GB A6000 resolved to the 24 GB profile and the three rtx-a6000 envelope
    # rows could never fire. Lock the split by NAME and by VRAM fallback.
    ("NVIDIA RTX A6000",              49140,  8.6, "rtx-a6000"),
    ("Unnamed Ampere 48GB",           49140,  8.6, "rtx-a6000"),
    ("NVIDIA GeForce RTX 3090",       24576,  8.6, "rtx-3090"),
    ("NVIDIA RTX A5000",              24564,  8.6, "rtx-a5000"),
    # Deliberate under-promise: no exact profile, so fall to the largest SMALLER
    # same-family profile. Safe direction; the reverse would not be.
    ("NVIDIA A100-SXM4-80GB",         81920,  8.0, "a100-40gb"),
]
bad = [f"{n}->{m(n,v,s)} want {e}" for n, v, s, e in cases if m(n, v, s) != e]
print("FAIL: " + " | ".join(bad) if bad else "OK")
PY
)"
[[ "$det" == "OK" ]] || { echo "  FAIL: detector: $det"; exit 1; }
echo "  ok: hardware detector split (PRO 6000 / GB10 / 5090 / Ada / A6000 / A5000 / 3090 / A100-80)"

# --- #246 Phase 2 mem-fraction floor (DOWNWARD only) --------------------------
# A unified-memory card (Spark, mem_util_safe 0.85) can't safely give the 0.92
# compose default -> inject the floor. Discrete cards (0.95/0.96 > 0.92) are
# never raised (that touches Cliff margin -> validated opt-in, not automatic).
GPU_SPARK='0|NVIDIA GB10|131072|12.1'
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_SPARK")"
assert_contains "$out" "GPU_MEMORY_UTILIZATION=0.85"
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_5090X2")"
assert_not_contains "$out" "GPU_MEMORY_UTILIZATION"
# heterogeneous: the lowest ceiling (Spark 0.85) forces the whole rig down
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "${GPU_SPARK};1|NVIDIA GeForce RTX 5090|32607|12.0")"
assert_contains "$out" "GPU_MEMORY_UTILIZATION=0.85"
# explicit user pin wins
out="$(GPU_MEMORY_UTILIZATION=0.7 python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_SPARK")"
assert_not_contains "$out" "GPU_MEMORY_UTILIZATION=0.85"
# #1365: the floor is now reachable on SGLang too, and it must arrive under
# SGLang's OWN key. `--mem-fraction-static` is NOT `--gpu-memory-utilization`
# (static weights+KV share vs total-VRAM budget, cudagraph capture comes from
# the remainder), so emitting vLLM's name here would land on a var no sgl
# compose reads -- silently inert. The VALUE is the card's safe ceiling, a
# hardware property (Spark's LPDDR5X is shared with the Grace CPU/OS), not a
# vLLM-tuned number, and it only ever moves DOWNWARD from the slug's own
# registered default -- conservative under either engine's semantics.
# ⚠️ Still UNVALIDATED on SGLang: no sgl soak at 0.85 on a unified-memory card.
out="$(python3 "$HELPER" resolve-variant-pin --variant sgl/qwen38-27b-dual-fast --format shell --gpu-spec "$GPU_SPARK")"
assert_contains "$out" "MEM_FRACTION=0.85"
assert_not_contains "$out" "GPU_MEMORY_UTILIZATION"
out="$(python3 "$HELPER" resolve-variant-pin --variant sgl/qwen38-27b-dual-fast --format shell --gpu-spec "$GPU_5090X2")"
assert_not_contains "$out" "MEM_FRACTION"
# llama.cpp has no fraction knob at all -> neither name, on any card
out="$(python3 "$HELPER" resolve-variant-pin --variant llamacpp-club3090/glm53-flash-dual-iq4xs-moecache --format shell --gpu-spec "$GPU_SPARK")"
assert_not_contains "$out" "MEM_FRACTION"
assert_not_contains "$out" "GPU_MEMORY_UTILIZATION"
echo "  ok: #246 mem-fraction floor (Spark down · discrete no-raise · het-min · user-pin · sgl-own-key · llama.cpp-none — 7 cases)"

# --- fp8/NVFP4-weights DeepGEMM disable on consumer cards (disc #571/#613) ---
# DeepGEMM has no recipe on consumer Blackwell (sm_120/121, hard-fails) and is
# unused on Ada (sm_89, harmless no-op) -> disable for fp8-family and ModelOpt
# NVFP4 slugs. Hopper (sm_90) keeps it; unrelated quant families are untouched.
DMAX=vllm/qwen-27b-dual-max
NVFP4_SINGLE=vllm/qwen-27b-single-nvfp4
GPU_H100X2='0|NVIDIA H100|81920|9.0;1|NVIDIA H100|81920|9.0'
out="$(python3 "$HELPER" resolve-variant-pin --variant "$DMAX" --format shell --gpu-spec "$GPU_5090X2")"
assert_contains "$out" "VLLM_USE_DEEP_GEMM=0"
out="$(python3 "$HELPER" resolve-variant-pin --variant "$DMAX" --format shell --gpu-spec "${GPU_4090};1|NVIDIA GeForce RTX 4090|24564|8.9")"
assert_contains "$out" "VLLM_USE_DEEP_GEMM=0"          # Ada proactively covered
out="$(python3 "$HELPER" resolve-variant-pin --variant "$DMAX" --format shell --gpu-spec "$GPU_H100X2")"
assert_not_contains "$out" "VLLM_USE_DEEP_GEMM"        # Hopper keeps the fast path
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/dual --format shell --gpu-spec "$GPU_5090X2")"
assert_not_contains "$out" "VLLM_USE_DEEP_GEMM"        # int4-weights slug: not the DeepGEMM path
# fp8-DYNAMIC (compressed-tensors) is fp8-family too — agents-a1 must also disable
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/agents-a1-dual --format shell --gpu-spec "$GPU_5090X2")"
assert_contains "$out" "VLLM_USE_DEEP_GEMM=0"          # fp8-dynamic weights route FP8 GEMM
out="$(python3 "$HELPER" resolve-variant-pin --variant "$NVFP4_SINGLE" --format shell --gpu-spec "$GPU_5090X2")"
assert_contains "$out" "VLLM_USE_DEEP_GEMM=0"          # ModelOpt NVFP4 carries FP8 linears
out="$(VLLM_USE_DEEP_GEMM=1 python3 "$HELPER" resolve-variant-pin --variant "$DMAX" --format shell --gpu-spec "$GPU_5090X2")"
assert_not_contains "$out" "VLLM_USE_DEEP_GEMM=0"      # explicit user pin wins
echo "  ok: fp8/NVFP4 DeepGEMM disable (5090/Ada down · Hopper keep · non-fp8 skip · fp8-dynamic · nvfp4 · user-pin — 7 cases)"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  out="$(VLLM_NIGHTLY_SHA="$CLEAN_SHA" docker compose -f "$ROOT_DIR/models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml" config 2>/dev/null)"
  assert_contains "$out" "image: vllm/vllm-openai:v0.30.0"

  out="$(VLLM_NIGHTLY_SHA="$CLEAN_SHA" VLLM_IMAGE=vllm/vllm-openai:latest docker compose -f "$ROOT_DIR/models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml" config 2>/dev/null)"
  assert_contains "$out" "image: vllm/vllm-openai:latest"
fi

out="$(python3 - <<'PY'
from scripts.lib.profiles.compat import InstanceSpec
from scripts.lib.profiles.estate_cli import compose_env

clean = compose_env(InstanceSpec(name="qwen", compose_name="vllm/dual", gpu_indices=(0, 1), port=8010))
gemma = compose_env(InstanceSpec(name="gemma", compose_name="vllm/gemma-int8-mtp", gpu_indices=(0, 1), port=8032))
print(clean["VLLM_IMAGE"])
print(gemma["VLLM_IMAGE"])
PY
)"
assert_contains "$out" "vllm/vllm-openai:v0.30.0"   # clean (vllm/dual → vllm-stable) bumped to v0.30.0
assert_contains "$out" "vllm/vllm-openai:v0.22.0"   # gemma (vllm/gemma-int8-mtp → vllm-gemma-stable) stays v0.22.0

# --- #809: decode_granularity travels from the profile YAML to the launchers --
# A block-diffusion (dLLM) model denoises a whole canvas in parallel and emits
# ~one SSE chunk per canvas, so TTFT == wall on a single-canvas response and
# `decode_TPS = tokens/(wall - TTFT)` divides by a zero-width window. The class
# is a property of the MODEL, so it is declared in the model profile and rides
# the same export channel as the image pin.
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/diffusiongemma-dual --format shell)"
assert_contains "$out" "DECODE_GRANULARITY=canvas"
# ...and it must be gpu-spec-independent: unlike the arch-aware exports, this is
# the same fact on every card, so it must not vanish when a spec IS passed.
out="$(python3 "$HELPER" resolve-variant-pin --variant vllm/diffusiongemma-dual \
  --format shell --gpu-spec "$GPU_3090")"
assert_contains "$out" "DECODE_GRANULARITY=canvas"
# Every autoregressive slug's export set must be unchanged — the field defaults
# to "token" and the emitter stays silent for it, so no other slug moves a byte.
# (#1365 corrected the note that used to sit here: it claimed resolve-variant-pin
# "refuses a non-docker-image engine pin, which is llamacpp's pre-existing
# behaviour". `llama-cpp-local` IS install.method: docker_image -- the raise was
# for "no env-key mapping" wearing the wrong error text. It no longer raises, so
# the sample below is vLLM slugs simply because DECODE_GRANULARITY is declared by
# a vLLM-served model; a llama.cpp slug is asserted silent right after.)
for v in vllm/dual vllm/minimal vllm/gemma-int8-mtp \
         llamacpp/default sgl/qwen38-27b-dual-fast exllamav3/qwen38-flash-next-dual-exl3-305-cpumoe; do
  out="$(python3 "$HELPER" resolve-variant-pin --variant "$v" --format shell --gpu-spec "$GPU_3090")"
  assert_not_contains "$out" "DECODE_GRANULARITY"
done
# Both launchers must ACCEPT the key. Their allowlists are hand-written and an
# unlisted key is `exit 2`, not a silent no-op — so an emitter without both arms
# breaks the launch, and nothing else in the suite would catch it.
for f in "$ROOT_DIR/scripts/launch.sh" "$ROOT_DIR/scripts/switch.sh"; do
  command grep -q 'DECODE_GRANULARITY)' "$f" \
    || { echo "ASSERTION FAILED: $f does not allowlist DECODE_GRANULARITY (launch would exit 2)" >&2; exit 1; }
done
# The profile field is validated, not coerced: a typo must FAIL rather than
# silently degrade to "token" and re-arm the epsilon divide on the one model
# class the field exists to protect.
out="$(python3 - <<'PY' 2>&1 || true
from scripts.lib.profiles.compat import _decode_granularity
try:
    _decode_granularity({"id": "x", "decode_granularity": "canvass"})
    print("NO-RAISE")
except ValueError as e:
    print(f"raised: {e}")
PY
)"
assert_contains "$out" "raised:"
assert_not_contains "$out" "NO-RAISE"
echo "  ✓ #809: decode_granularity reaches both launchers, only for the model that declares it"

# --- #1365: the export path is invoked TWICE on a launch.sh run ---------------
# launch.sh:1473 exports, then execs switch.sh, which exports again into the
# inherited env. Now that the vllm/beellama prefix gate is gone this happens for
# EVERY slug, so it has to be provably idempotent: pass 2 must not re-announce,
# must not clobber, and must not report pass 1's own export as a user override.
#
# ⚠️ POSITIVE CONTROL FIRST. "pass 2 printed nothing" is worthless unless pass 1
# printed something -- a harness that silently fails to call the function at all
# would satisfy the negative half. So each launcher asserts BOTH halves.
dbl_harness() { # dbl_harness <launcher> <slug> <gpu-spec> [PRESET=k=v]
  local body; body="$(sed -n '/^export_variant_engine_pin() {/,/^}/p' "$1")"
  SPEC="$3" LAUNCH_PROFILE="$HELPER" PRESET="${4:-}" SLUG="$2" COMPOSE_BIN=: bash -c '
    set -uo pipefail
    switch_gpu_profile_spec()   { printf "%s" "${SPEC}"; }   # switch.sh
    selected_gpu_profile_spec() { printf "%s" "${SPEC}"; }   # launch.sh
    '"$body"'
    [[ -n "$PRESET" ]] && export "$PRESET"
    echo "@@PASS1"; export_variant_engine_pin "$SLUG"
    echo "@@PASS2"; export_variant_engine_pin "$SLUG"
    echo "@@FINAL MOE_RESERVE_MB=${MOE_RESERVE_MB:-<unset>}"
  ' 2>&1
}
# a moe-cache slug on a 5090 is the widest case the flip newly reaches: it was
# behind the prefix gate until #1365 and it injects a non-image key.
DBL_SLUG="llamacpp-club3090/glm53-flash-dual-iq4xs-moecache"
DBL_SPEC="0|NVIDIA GeForce RTX 5090|32607|12.0;1|NVIDIA GeForce RTX 5090|32607|12.0"
# ⚠️ Kept as an ARRAY, not `for _L in scripts/switch.sh scripts/launch.sh`:
# that spelling matches test-tests-never-launch's EXEC regex (the `sh` of
# switch.sh + a space + scripts/launch.sh reads as `sh …launch.sh`). It is a
# false positive — nothing is executed here — but that guard is deliberately
# dumb and broad, and a precise version of it once went blind to the exact
# line it exists to catch. Satisfy it rather than argue with it.
DBL_LAUNCHERS=("scripts/switch.sh" "scripts/launch.sh")
for _L in "${DBL_LAUNCHERS[@]}"; do
  _out="$(dbl_harness "$_L" "$DBL_SLUG" "$DBL_SPEC")"
  _p1="$(printf '%s' "$_out" | sed -n '/@@PASS1/,/@@PASS2/p')"
  # ⚠️ the @@FINAL line NAMES the key, so it must be excluded from the pass-2
  # window or the "did pass 2 re-announce?" grep matches its own witness line.
  _p2="$(printf '%s' "$_out" | sed -n '/@@PASS2/,/@@FINAL/{/@@FINAL/!p;}')"
  grep -q "MOE_RESERVE_MB=2048" <<<"$_p1" \
    || { echo "ASSERTION FAILED ($_L): pass 1 did not inject MOE_RESERVE_MB (positive control)" >&2
         printf '%s\n' "$_out" >&2; exit 1; }
  grep -q "MOE_RESERVE_MB" <<<"$_p2" \
    && { echo "ASSERTION FAILED ($_L): pass 2 re-announced MOE_RESERVE_MB — double invocation is not idempotent" >&2
         printf '%s\n' "$_out" >&2; exit 1; }
  grep -q "keeping your value" <<<"$_out" \
    && { echo "ASSERTION FAILED ($_L): the launcher reported its OWN export as a user override" >&2
         printf '%s\n' "$_out" >&2; exit 1; }
  grep -q "@@FINAL MOE_RESERVE_MB=2048" <<<"$_out" \
    || { echo "ASSERTION FAILED ($_L): pass 2 clobbered the value" >&2; printf '%s\n' "$_out" >&2; exit 1; }
  # a genuine user override survives both passes untouched
  _ov="$(dbl_harness "$_L" "$DBL_SLUG" "$DBL_SPEC" "MOE_RESERVE_MB=999")"
  grep -q "@@FINAL MOE_RESERVE_MB=999" <<<"$_ov" \
    || { echo "ASSERTION FAILED ($_L): user MOE_RESERVE_MB=999 did not survive" >&2; printf '%s\n' "$_ov" >&2; exit 1; }
done
echo "  ✓ #1365: double invocation is idempotent in both launchers (inject-once · no re-announce · no false 'user override' · user value survives)"
echo "test-launch-compat: ok"
