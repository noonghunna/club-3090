#!/usr/bin/env bash
#
# arch-ab.sh — cross-rig KV-dtype A/B runner for issue #246 (lean tier).
#
# Runs the SAME pilot variant once per KV-dtype arm, each arm on a fresh
# symmetric boot, with the protocol bench + NIAH recall ladder — and bundles
# everything into one tarball to attach to the issue. Deliberately NO quality
# packs and NO soak: quality validation is consolidated at promotion time
# (issue #246 test plan), so a full run is ~25-30 min/arm, not hours.
#
# Usage:
#   bash scripts/arch-ab.sh                          # arms e5m2,e4m3; variant by GPU count
#   bash scripts/arch-ab.sh --arms e5m2,e4m3,nvfp4   # Blackwell (sm>=10) opt-in third arm
#   bash scripts/arch-ab.sh --variant vllm/dual      # force the variant
#   bash scripts/arch-ab.sh --dry-run                # print the plan, run nothing
#   bash scripts/arch-ab.sh --resume                 # skip arms/steps with artifacts
#   bash scripts/arch-ab.sh --full                   # + soak + 8-pack (long; NOT the ask)
#
# Variant auto-pick: 2+ GPUs -> vllm/dual (TP=2), 1 GPU -> vllm/minimal.
# Arms:
#   e5m2   fp8_e5m2 KV on the pilot variant (control — today's shipped default)
#   e4m3   fp8_e4m3 KV on the pilot variant (native FP8 KV compute on sm_89+)
#   nvfp4  nvfp4 KV on the pilot variant (Blackwell-only, UNVALIDATED — refuse
#          below sm 10.0)
#   fp8w   vllm/qwen-27b-dual-max STOCK (FP8 weights + int8-PTH KV, no dtype
#          override; dual-rig only) — measures the native-FP8-WEIGHTS lift on
#          sm_89+ vs Ampere's Marlin-dequant path. NOTE: fp8 KV is rejected on
#          FP8-weights checkpoints, which is why the KV arms can't run there.
# KV arms are EXPLICIT KV_CACHE_DTYPE pins, so what ran is never ambiguous
# regardless of launcher-injection state.
#
# Outputs: results/rebench/246-ab-<arm>/ per arm + a single
# results/rebench/246-ab-bundle-<host>-<date>.tgz to attach to issue #246.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=lib/compose-meta.sh
source "${ROOT_DIR}/scripts/lib/compose-meta.sh"

die() { echo "[arch-ab] ERROR: $*" >&2; exit 1; }

ARMS="e5m2,e4m3"
VARIANT=""
DRY_RUN=0
RESUME=0
FULL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms)    ARMS="$2"; shift 2 ;;
    --arms=*)  ARMS="${1#*=}"; shift ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --variant=*) VARIANT="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume)  RESUME=1; shift ;;
    --full)    FULL=1; shift ;;
    -h|--help) awk 'NR>1 && /^set -euo/{exit} NR>1{sub(/^# ?/,""); print}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

# KV arms pin a dtype on the pilot variant; the fp8w arm swaps the VARIANT
# (stock config, no pin) and asserts its registry KV format is live instead.
declare -A ARM_DTYPE=([e5m2]=fp8_e5m2 [e4m3]=fp8_e4m3 [nvfp4]=nvfp4 [fp8w]="")
declare -A ARM_VARIANT_OVERRIDE=([fp8w]="vllm/qwen-27b-dual-max")
declare -A ARM_ASSERT_DTYPE=([fp8w]="int8_per_token_head")

# --- detect GPUs -------------------------------------------------------------
GPU_LINES="$(compose_hw_detect_gpus 2>/dev/null || true)"
[[ -n "$GPU_LINES" ]] || die "no NVIDIA GPUs detected (nvidia-smi missing or empty)"
GPU_COUNT=0
MIN_SM=999
while IFS=$'\t' read -r idx name mem sm; do
  [[ -z "$idx" ]] && continue
  GPU_COUNT=$((GPU_COUNT + 1))
  awk -v a="$sm" -v b="$MIN_SM" 'BEGIN{exit !(a<b)}' && MIN_SM="$sm"
done <<< "$GPU_LINES"

# --- resolve variant + validate arms ----------------------------------------
if [[ -z "$VARIANT" ]]; then
  if (( GPU_COUNT >= 2 )); then VARIANT="vllm/dual"; else VARIANT="vllm/minimal"; fi
fi
[[ "$VARIANT" == "vllm/dual" && "$GPU_COUNT" -lt 2 ]] \
  && die "vllm/dual is TP=2 and needs 2 GPUs (detected ${GPU_COUNT}); use --variant vllm/minimal"

IFS=',' read -ra ARM_LIST <<< "$ARMS"
[[ "${#ARM_LIST[@]}" -ge 1 ]] || die "no arms given"
for arm in "${ARM_LIST[@]}"; do
  [[ -n "${ARM_DTYPE[$arm]+x}" ]] || die "unknown arm '${arm}' (valid: e5m2, e4m3, nvfp4, fp8w)"
  if [[ "$arm" == "e4m3" ]]; then
    # vLLM hard-rejects fp8_e4m3 KV below SM 8.9 at boot — without this guard
    # an Ampere rig sits in switch.sh's ready-wait until the 10-min timeout.
    awk -v a="$MIN_SM" 'BEGIN{exit !(a>=8.9)}' \
      || die "arm 'e4m3' needs sm>=8.9 (fp8_e4m3 KV is boot-rejected on Ampere); detected min sm_${MIN_SM}. On a 3090-class rig there is no arch delta to measure — this A/B is for Ada/Blackwell rigs (run --arms e5m2 if you just want the control numbers)"
  fi
  if [[ "$arm" == "nvfp4" ]]; then
    awk -v a="$MIN_SM" 'BEGIN{exit !(a>=10.0)}' \
      || die "arm 'nvfp4' needs Blackwell (sm>=10.0); detected min sm_${MIN_SM}. Drop it: --arms e5m2,e4m3"
  fi
  if [[ "$arm" == "fp8w" && "$GPU_COUNT" -lt 2 ]]; then
    die "arm 'fp8w' runs vllm/qwen-27b-dual-max (TP=2) and needs 2 GPUs (detected ${GPU_COUNT})"
  fi
done

# --- plan --------------------------------------------------------------------
echo "[arch-ab] plan: variant=${VARIANT}  arms=${ARMS}  gpus=${GPU_COUNT} (min sm_${MIN_SM})"
echo "[arch-ab]       per arm: fresh boot -> verify-full + bench + verify-stress$( ((FULL)) && echo ' + soak + 8-pack both (--full)' || echo ' (soak/quality skipped — lean tier)')"
for arm in "${ARM_LIST[@]}"; do
  arm_variant="${ARM_VARIANT_OVERRIDE[$arm]:-$VARIANT}"
  if [[ -n "${ARM_DTYPE[$arm]}" ]]; then
    echo "[arch-ab]       arm ${arm}: ${arm_variant} + KV_CACHE_DTYPE=${ARM_DTYPE[$arm]} -> results/rebench/246-ab-${arm}/"
  else
    echo "[arch-ab]       arm ${arm}: ${arm_variant} STOCK (no dtype pin) -> results/rebench/246-ab-${arm}/"
  fi
done
if (( DRY_RUN )); then
  echo "[arch-ab] dry run — nothing executed."
  exit 0
fi

# --- container names for the live-dtype assertion (registry-derived) ---------
declare -A VARIANT_DEFAULT_PORT=() VARIANTS=() VARIANT_STATUS=() VARIANT_STATUS_NOTE=() VARIANT_CONTAINER=()
# shellcheck source=lib/registry-emit.sh
source "${ROOT_DIR}/scripts/lib/registry-emit.sh"
derive_switch_variant_tables "${ROOT_DIR}"

# --- run arms ----------------------------------------------------------------
for arm in "${ARM_LIST[@]}"; do
  dtype="${ARM_DTYPE[$arm]}"
  arm_variant="${ARM_VARIANT_OVERRIDE[$arm]:-$VARIANT}"
  assert_dtype="${ARM_ASSERT_DTYPE[$arm]:-$dtype}"
  container="${VARIANT_CONTAINER[$arm_variant]:-}"
  [[ -n "$container" ]] || die "no container mapping for ${arm_variant} in the registry"
  tag="246-ab-${arm}"
  echo ""
  echo "[arch-ab] ===== arm ${arm}: ${arm_variant}$( [[ -n "$dtype" ]] && echo " KV_CACHE_DTYPE=${dtype}" || echo " (stock)") -> tag ${tag} ====="
  # Fresh symmetric boot per arm (same settle path for every arm). KV arms use
  # an explicit env pin so what ran is unambiguous on any launcher version;
  # the fp8w arm runs stock (its checkpoint REJECTS fp8 KV — see header).
  if [[ -n "$dtype" ]]; then
    KV_CACHE_DTYPE="$dtype" bash scripts/switch.sh "$arm_variant"
  else
    bash scripts/switch.sh "$arm_variant"
  fi
  # Assert the expected dtype is actually live before spending bench time on it.
  running_cmd="$(docker inspect "$container" --format '{{join .Config.Cmd " "}}' 2>/dev/null || true)"
  case "$running_cmd" in
    *"--kv-cache-dtype ${assert_dtype}"*) echo "[arch-ab] verified: container runs --kv-cache-dtype ${assert_dtype}" ;;
    *) die "arm ${arm}: container '${container}' is NOT running --kv-cache-dtype ${assert_dtype} — aborting before benching the wrong config" ;;
  esac
  rb_args=(--tag "$tag")
  stress_fast=0
  if (( FULL )); then
    rb_args+=(--with-8pack-thinking=both)
  else
    rb_args+=(--skip soak)
    # A/B tier: STRESS_FAST halves verify-stress's deep-prefill request count
    # (large fresh needles skipped + ladder capped at ~3 rungs) while keeping
    # the fillable-to proof, ceiling recall, and VRAM margin.
    stress_fast=1
  fi
  (( RESUME )) && rb_args+=(--resume)
  STRESS_FAST="$stress_fast" bash scripts/rebench-full.sh "${rb_args[@]}"
  # A margin-advisory rc=1 with a clean ladder scares first-time runners into
  # aborting — say explicitly that it is not a failure.
  if command grep -q "rungs passed" "results/rebench/${tag}/verify-stress.log" 2>/dev/null \
     && command grep -q "VRAM margin thin at ceiling" "results/rebench/${tag}/verify-stress.log" 2>/dev/null; then
    echo "[arch-ab] note: verify-stress exited nonzero on the VRAM-margin ADVISORY only —"
    echo "[arch-ab]       the NIAH ladder is CLEAN. Expected on tightly-packed configs; carry on."
  fi
done

# --- summary + bundle ---------------------------------------------------------
echo ""
echo "[arch-ab] ===== summary ====="
ARMS="$ARMS" python3 - <<'PY'
import json
import os
import re
from pathlib import Path

arms = [a for a in os.environ["ARMS"].split(",") if a]
print(f"{'arm':<8} {'narr TPS':>9} {'code TPS':>9} {'TTFT ms':>8}  NIAH ladder")
for arm in arms:
    tag = Path(f"results/rebench/246-ab-{arm}")
    try:
        bench = json.loads((tag / "_internal.json").read_text()).get("bench") or {}
        narr = (bench.get("narrative") or {}).get("decode_tps_mean")
        code = (bench.get("code") or {}).get("decode_tps_mean")
        ttft = (bench.get("narrative") or {}).get("ttft_ms_mean")
    except OSError:
        narr = code = ttft = None
    ladder = "(no verify-stress artifact)"
    try:
        m = re.search(r"all (\d+) rungs passed — fillable to (\d+) tok",
                      (tag / "verify-stress.log").read_text(errors="replace"))
        if m:
            ladder = f"{m.group(1)} rungs clean, fillable to {int(m.group(2)):,} tok"
            mm = re.search(r"VRAM margin thin at ceiling: (\d+) MB free < (\d+) MB",
                           (tag / "verify-stress.log").read_text(errors="replace"))
            if mm:
                ladder += f" (margin advisory: {mm.group(1)} MB < {mm.group(2)} MB — not a failure)"
        else:
            ladder = "LADDER DID NOT PASS — check verify-stress.log"
    except OSError:
        pass
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "—"
    print(f"{arm:<8} {fmt(narr):>9} {fmt(code):>9} {fmt(ttft):>8}  {ladder}")
PY

# Rig triage report (fast mode + kv-calc calibration matrix; path/host/user
# redaction ON) — the cross-rig context that makes A/B variance interpretable
# (PCIe lane width, driver, power caps, WSL-vs-bare-metal, NVLink state), plus
# predicted-vs-actual VRAM calibration on this card class (Phase 2 input).
# Deliberately NOT --full: that re-runs the whole test battery (~43 min)
# against only the LAST arm's container — the arms already carry that data.
echo "[arch-ab] capturing rig report (redacted, + kv-calc calibration) ..."
bash scripts/report.sh --full-calibration > results/rebench/246-ab-rig-report.md 2>/dev/null \
  || echo "[arch-ab] WARN: report.sh failed — bundle will ship without the rig report" >&2

bundle="results/rebench/246-ab-bundle-$(hostname)-$(date +%Y%m%d).tgz"
tar czf "$bundle" \
  $(for arm in "${ARM_LIST[@]}"; do echo "results/rebench/246-ab-${arm}"; done) \
  $( [[ -s results/rebench/246-ab-rig-report.md ]] && echo results/rebench/246-ab-rig-report.md )
echo ""
echo "[arch-ab] bundle written: ${bundle}"
echo "[arch-ab] -> attach that ONE file to the #246 cross-rig test thread (link pinned on"
echo "[arch-ab]    https://github.com/noonghunna/club-3090/issues/246)"
echo "[arch-ab]    plus a sentence on anything that surprised or annoyed you — the friction"
echo "[arch-ab]    report is as valuable as the numbers."
