#!/usr/bin/env bash
# test-cpu-moe-split-fit — guards `resolve_cpu_moe_split` (#1366 / #1361 step 7).
#
# REDs on: a calibration point that stops reproducing (an engine-pin bump or a
# CTX/KV change silently invalidates the reserve constant) · a rig-size change
# that does NOT move the split (the #1360 defect: a 2x32 GB rig running the
# 2x24 GB expert count and leaving ~16 GB unused) · an explicit pin being
# clobbered · the >50% selection-rule warning going silent · the resolver firing
# on a compose that never asked for it.
#
# ⚠️ The constants are CALIBRATED, not derived, so this test is the only thing
# that notices when they rot. It pins BOTH tiers at the anchor rig; if you change
# an engine pin, CTX or KV_TYPE, re-measure and update the header AND this file.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=scripts/lib/compose-meta.sh
source "${ROOT_DIR}/scripts/lib/compose-meta.sh"

FAIL=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; FAIL=1; }

C305="models/qwen3.8-flash-next/exllamav3/compose/dual/exl3-3.05bpw/cpumoe.yml"
C405="models/qwen3.8-flash-next/exllamav3/compose/dual/exl3-4.05bpw/cpumoe.yml"

# CPU_MOE_FREE_MIB is the test seam: the fit must be reproducible WITHOUT a GPU,
# or this guard only runs on one rig and rots everywhere else.
split_for() { # split_for <compose> <free-list> [PIN]
  local out
  out="$(CPU_MOE_FREE_MIB="$2" MOE_SPLIT="${3:-}" bash -c '
    source scripts/lib/compose-meta.sh
    resolve_cpu_moe_split "$1" >/dev/null 2>&1
    printf "%s" "${MOE_SPLIT:-unset}"' _ "$1")"
  printf '%s' "$out"
}
warn_for() { # warn_for <compose> <free-list> [PIN]
  CPU_MOE_FREE_MIB="$2" MOE_SPLIT="${3:-}" bash -c '
    source scripts/lib/compose-meta.sh
    resolve_cpu_moe_split "$1" 2>&1 >/dev/null' _ "$1"
}

# --- 1. the calibration anchors -----------------------------------------------
# 2x24 GB, free 24,123 MiB/card, exl3 1.5.0, ctx 204800, cache-mode Q4.
# 3.05@144 and 4.05@256 are BOOT-VERIFIED fit points, not preferences:
# learnings/exllamav3-engine.md has the ladder (144 fits on 1.5.0, 1.5.1 needs
# 160) and the per-card VRAM at each.
ANCHOR="24123 24123"
got="$(split_for "$C305" "$ANCHOR")"
[[ "$got" == "144" ]] && ok "3.05 tier reproduces the 2x24 GB anchor (144)" \
                      || bad "3.05 anchor: want 144, got ${got}"
got="$(split_for "$C405" "$ANCHOR")"
[[ "$got" == "256" ]] && ok "4.05 tier reproduces the 2x24 GB anchor (256)" \
                      || bad "4.05 anchor: want 256, got ${got}"

# --- 2. NEGATIVE CONTROL ------------------------------------------------------
# A guard that only asserts "144" proves nothing unless a wrong input changes it.
# Shrink the rig by one card's worth and the split MUST rise.
got="$(split_for "$C305" "24123")"
[[ "$got" =~ ^[0-9]+$ && "$got" -gt 144 ]] \
  && ok "negative control: one card instead of two raises the split (${got} > 144)" \
  || bad "negative control: one card gave '${got}', expected >144 — the fit is not reading free VRAM"

# --- 3. the #1360 defect ------------------------------------------------------
# The bug that started this: a 2x32 GB rig ran the 2x24 GB expert count. A larger
# rig MUST put fewer experts on the CPU. This is the assertion that would have
# caught #1360 before a user did.
small="$(split_for "$C405" "24123 24123")"
big="$(split_for "$C405" "32100 32100")"
[[ "$big" =~ ^[0-9]+$ && "$big" -lt "$small" ]] \
  && ok "#1360: 2x32 GB gets a LOWER split than 2x24 GB (${big} < ${small})" \
  || bad "#1360 regression: 2x32 GB gave '${big}' vs 2x24 GB '${small}' — the hardcoded-count defect is back"

# --- 4. an explicit pin always wins -------------------------------------------
# Same contract as THREADS and OT_G<i>: the override exists to out-judge us.
got="$(split_for "$C305" "$ANCHOR" 300)"
[[ "$got" == "300" ]] && ok "explicit MOE_SPLIT=300 is kept, not clobbered" \
                      || bad "explicit pin: want 300, got ${got}"
warn_for "$C305" "$ANCHOR" 300 | grep -q "would be 144" \
  && ok "a pin still reports the fit it overrode (the actual diagnostic)" \
  || bad "a pin hid the fit — 'my pin vs what the rig holds' is the whole question"

# --- 5. the selection-rule warning --------------------------------------------
# exllamav3.yml: over ~50% CPU-resident the model belongs on an engine with a
# real expert cache. GLM-5.3-Flash was rejected on exactly this basis at 80%.
# Nothing told you that you were at the threshold before this line existed.
warn_for "$C305" "24123" | grep -q "REAL EXPERT CACHE" \
  && ok ">50% CPU-resident warns and cites the engine profile's own rule" \
  || bad ">50% warning is silent — a single-card rig lands at ~64% and hears nothing"
warn_for "$C405" "$ANCHOR" | grep -q "exactly 50%" \
  && ok "the 4.05 tier is flagged as sitting exactly ON the 50% threshold" \
  || bad "the 4.05 anchor is 256/512 = exactly 50% and must say so"
warn_for "$C305" "$ANCHOR" | grep -q "REAL EXPERT CACHE" \
  && bad "the >50% warning fired at 28% — it is not reading the fraction" \
  || ok "no >50% warning at the 28% anchor (the warning discriminates)"

# --- 6. extrapolation honesty -------------------------------------------------
# One calibration point per tier means every other rig is an extrapolation, and
# the risky direction is DOWN: too small a split crash-loops with "Insufficient
# VRAM in split" WHILE THE PORT STAYS OPEN (RestartCount is the only signal).
warn_for "$C405" "32100 32100" | grep -q "EXTRAPOLATED" \
  && ok "a rig far from the anchor is labelled an extrapolation" \
  || bad "no extrapolation warning off-anchor — an unmeasured number reads as measured"
warn_for "$C405" "$ANCHOR" | grep -q "EXTRAPOLATED" \
  && bad "extrapolation warning fired AT the calibration anchor" \
  || ok "no extrapolation warning at the anchor itself"

# --- 7. no-op everywhere else -------------------------------------------------
# Marker-scoped like the other resolvers: a compose without CPU-MoE-* headers
# must come back untouched, or this injector leaks into 136 other slugs.
other="models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml"
[[ -f "$other" ]] || other="$(command grep -rl 'image:' models/*/vllm/compose/dual/*/*.yml 2>/dev/null | head -1)"
if [[ -n "$other" && -f "$other" ]]; then
  got="$(split_for "$other" "$ANCHOR")"
  [[ "$got" == "unset" ]] && ok "no-op on a compose without CPU-MoE-* headers ($(basename "$(dirname "$other")")/$(basename "$other"))" \
                          || bad "resolver fired on an unrelated compose, set MOE_SPLIT=${got}"
fi

# --- 8. the value has somewhere to land ---------------------------------------
# Exporting a var the compose never reads is a silent no-op. Verified live once
# with `docker compose config` (unset -> 144, MOE_SPLIT=137 -> 137); this keeps
# the wiring from being deleted.
for c in "$C305" "$C405"; do
  env_name="$(compose_meta_get "$c" cpu-moe-split-env)"
  grep -q "\${${env_name}:-" "$c" \
    && ok "$(basename "$(dirname "$c")") consumes \${${env_name}:- in its argv" \
    || bad "$(basename "$(dirname "$c")") declares Split-Env=${env_name} but never reads it"
done

# --- 9. the declared expert size is in the right ballpark ---------------------
# Expert-KiB is READ FROM THE TENSOR TABLE, not computed -- trellis quants carry
# overhead the nominal bpw understates. But a typo (a factor of 10, a transposed
# digit) must not pass, so bound it against 3*hidden*moe_inter*bpw/8.
#   3 * 2560 * 640 * bpw / 8 / 1024 KiB
for pair in "$C305:3.05" "$C405:4.05"; do
  c="${pair%:*}"; bpw="${pair##*:}"
  declared="$(compose_meta_get "$c" cpu-moe-expert-kib)"
  nominal="$(python3 -c "print(round(3*2560*640*${bpw}/8/1024))")"
  ratio="$(python3 -c "print(round(${declared}/${nominal},3))")"
  ok_ratio="$(python3 -c "print(1.0 <= ${declared}/${nominal} <= 1.15)")"
  [[ "$ok_ratio" == "True" ]] \
    && ok "${bpw}bpw Expert-KiB=${declared} is ${ratio}x the nominal ${nominal} (tensor-table overhead)" \
    || bad "${bpw}bpw Expert-KiB=${declared} vs nominal ${nominal} (ratio ${ratio}) — outside 1.00-1.15, likely a typo"
done

if (( FAIL )); then echo "test-cpu-moe-split-fit: FAIL" >&2; exit 1; fi
echo "test-cpu-moe-split-fit: ok"
