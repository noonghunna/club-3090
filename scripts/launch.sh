#!/usr/bin/env bash
#
# Interactive launcher for club-3090 — pick engine + workload, boot the
# right compose, run verify-full to confirm it's serving.
#
# For first-run users coming in from the README. If you already know
# what you want, use `scripts/switch.sh <variant>` directly.
#
# Usage:
#   bash scripts/launch.sh                              # interactive wizard
#   bash scripts/launch.sh --variant <name>             # skip wizard, boot directly
#   bash scripts/launch.sh --engine vllm --cards 1      # partial flags, ask the rest
#   bash scripts/launch.sh --no-verify                  # skip post-launch verify-full
#   bash scripts/launch.sh --no-preflight               # skip docker/GPU pre-flight
#
# All flags accept the same names as `switch.sh --list` produces.
# Examples:
#   bash scripts/launch.sh --variant vllm/default
#   bash scripts/launch.sh --variant llamacpp/default
#   bash scripts/launch.sh --variant vllm/dual

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SWITCH="${SWITCH:-${ROOT_DIR}/scripts/switch.sh}"
VERIFY="${VERIFY:-${ROOT_DIR}/scripts/verify-full.sh}"
# shellcheck source=preflight.sh
source "${ROOT_DIR}/scripts/preflight.sh"

# --- arg parsing ---
ENGINE=""
CARDS=""
VARIANT=""
SKIP_VERIFY=0
SKIP_PREFLIGHT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)  ENGINE="$2"; shift 2 ;;
    --cards)   CARDS="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --no-verify) SKIP_VERIFY=1; shift ;;
    --no-preflight) SKIP_PREFLIGHT=1; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# --- pre-flight ---
if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
  echo "[preflight] checking environment..."
  preflight_docker || exit 1
  preflight_gpu 1  || exit 1
  preflight_gpu_idle
  preflight_running
  preflight_genesis_pin "${ROOT_DIR}"
  preflight_repo_drift "${ROOT_DIR}"
  echo "[preflight] ok."
  echo ""
fi

ask() {
  # ask "prompt" "default" -> echoes user input or default
  local p="$1" d="${2:-}" reply
  if [[ -n "$d" ]]; then
    read -rp "$p [${d}]: " reply
    echo "${reply:-$d}"
  else
    read -rp "$p: " reply
    echo "$reply"
  fi
}

choose() {
  # choose "prompt" "label1" "value1" "label2" "value2" ... -> echoes chosen value
  local prompt="$1"; shift
  local i=1 labels=() values=()
  while [[ $# -gt 0 ]]; do
    labels+=("$1"); values+=("$2"); shift 2
  done
  echo "" >&2
  echo "$prompt" >&2
  for l in "${labels[@]}"; do
    printf "  %d) %s\n" "$i" "$l" >&2
    i=$((i+1))
  done
  while true; do
    local pick
    read -rp "Choice [1-${#labels[@]}]: " pick
    if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= ${#labels[@]} )); then
      echo "${values[$((pick-1))]}"
      return
    fi
    echo "  invalid — pick a number 1-${#labels[@]}" >&2
  done
}

# --- wizard ---
# Flow: cards → workload → auto-pick engine. Newcomers can answer "how
# many GPUs" and "what do I want to do" but rarely "vLLM or llama.cpp" —
# so the engine is derived from the workload pick, not asked first.
# Manual --engine override filters the workload list to that engine.
if [[ -z "$VARIANT" ]]; then
  echo ""
  echo "club-3090 launcher — let's pick the right config for your workload."
  echo "(Use --variant <name> next time to skip the wizard.)"
  echo ""

  # Step 1 — cards.
  if [[ -z "$CARDS" ]]; then
    CARDS=$(choose "How many RTX 3090s?" \
      "1× 3090 (24 GB)"           "1" \
      "2× 3090 (PCIe / no NVLink)" "2")
  fi

  # Re-validate now that we know the requirement (preflight already ran
  # with min=1; bump to actual count). Skip if --no-preflight.
  if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
    preflight_gpu "$CARDS" || exit 1
  fi

  # Step 2 — workload, filtered by cards (and --engine override if set).
  # Each option's value is "engine/file" so engine is implied by the pick.
  if [[ "$CARDS" == "1" ]]; then
    # Primary recommended options first; diagnostic / niche variants in
    # an "Other" group at the end. The only single-card limitation users
    # need to know: vLLM single-card crashes on a single prompt >50K
    # (Cliff 2). For unpredictable inputs, use llamacpp/default.
    if [[ -z "$ENGINE" || "$ENGINE" == "vllm" ]]; then
      VLLM_OPTS=(
        "Long ctx + vision (145K + vision, MTP) — recommended for chat/agents"        "vllm/long-vision"
        "Long ctx, text only — Balanced MTP (180K, MTP K=3) — recommended IDE-agent"  "vllm/long-text"
        "Long ctx, text only — Max-context (200K, no MTP) — one-shot >50K prompts"    "vllm/long-text-no-mtp"
        "Bounded thinking (180K, structured-CoT FSM — recommended grammar: DeepSeek scratchpad, 87.4% combined HE+/LCB v6)"  "vllm/bounded-thinking"
      )
    else
      VLLM_OPTS=()
    fi
    if [[ -z "$ENGINE" || "$ENGINE" == "llamacpp" ]]; then
      LLAMA_OPTS=(
        "Bulletproof, no cliffs (262K + vision, ~21 TPS) — production-safe"        "llamacpp/default"
      )
    else
      LLAMA_OPTS=()
    fi
    # Diagnostic / niche fallbacks — shown last so they don't dominate the menu
    if [[ -z "$ENGINE" || "$ENGINE" == "vllm" ]]; then
      VLLM_FALLBACK_OPTS=(
        "[fallback] Default 48K + vision (Cliff 2 unreachable; fast boot)"         "vllm/default"
        "[fallback] tools-text 75K FP8 (FP8 KV alternative for accuracy compare)"  "vllm/tools-text"
        "[fallback] minimal 32K (no Genesis, no spec-decode — diagnostic stack)"   "vllm/minimal"
      )
    else
      VLLM_FALLBACK_OPTS=()
    fi
    if [[ -z "$ENGINE" || "$ENGINE" == "llamacpp" ]]; then
      LLAMA_FALLBACK_OPTS=(
        "[fallback] llamacpp/concurrent (4 parallel slots, 192K pool, vision)"     "llamacpp/concurrent"
      )
    else
      LLAMA_FALLBACK_OPTS=()
    fi
    VARIANT=$(choose "What's your main workload?" \
      "${VLLM_OPTS[@]}" "${LLAMA_OPTS[@]}" \
      "${VLLM_FALLBACK_OPTS[@]}" "${LLAMA_FALLBACK_OPTS[@]}")
  elif [[ "$CARDS" == "2" ]]; then
    if [[ -n "$ENGINE" && "$ENGINE" != "vllm" ]]; then
      echo "ERROR: --engine ${ENGINE} not supported on 2× cards (no llama.cpp dual recipe yet)." >&2
      exit 1
    fi
    VARIANT=$(choose "What's your dual-card priority?" \
      "Balanced default — 262K + vision + 2 streams (recommended)" "vllm/dual" \
      "Multi-tenant — 4 concurrent streams @ 262K, TQ3 KV"         "vllm/dual-turbo" \
      "Peak code TPS with vision (185K, DFlash N=5)"               "vllm/dual-dflash" \
      "Peak code TPS no vision (200K, DFlash N=5)"                 "vllm/dual-dflash-noviz")
  else
    echo "ERROR: --cards ${CARDS} unsupported (expected 1 or 2)." >&2
    exit 1
  fi

  # Step 3 — explain the auto-picked engine.
  echo ""
  case "$VARIANT" in
    llamacpp/*)
      echo "[wizard] picked llama.cpp — chosen because no prefill cliffs at 262K and the simplest"
      echo "[wizard]   serving path. Trade-off: ~21 TPS vs 70+ on vLLM. Right call for long-prompt"
      echo "[wizard]   robustness, frontier context, or anyone who wants the simplest setup."
      ;;
    vllm/*)
      echo "[wizard] picked vLLM — chosen for spec-decode (MTP), tool-call extraction, and best TPS."
      echo "[wizard]   Watch out for the prefill cliffs (Cliff 1 = 25K+ tool returns; Cliff 2 = 50-60K"
      echo "[wizard]   single prompts on TQ3) — see docs/SINGLE_CARD.md for the safe-config map."
      ;;
  esac
fi

# --- launch + verify ---
echo ""
echo "[launch] selected variant: ${VARIANT}"
echo ""
"$SWITCH" "$VARIANT"

# Resolve the actual endpoint port + container name the same way switch.sh
# does: explicit $PORT / $CONTAINER > per-variant default. Mirrors
# VARIANT_DEFAULT_PORT in switch.sh — keep in sync if you add a new variant.
declare -A LAUNCH_DEFAULT_PORT=(
  [vllm/default]=8020
  [vllm/long-vision]=8020
  [vllm/long-text]=8020
  [vllm/long-text-no-mtp]=8021
  [vllm/bounded-thinking]=8020
  [vllm/tools-text]=8020
  [vllm/minimal]=8020
  [vllm/dual]=8010
  [vllm/dual-turbo]=8011
  [vllm/dual-dflash]=8012
  [vllm/dual-dflash-noviz]=8013
  [vllm/dual-nvlink]=8014
  [llamacpp/default]=8020
  [llamacpp/concurrent]=8020
)
declare -A LAUNCH_DEFAULT_CONTAINER=(
  [vllm/default]=vllm-qwen36-27b
  [vllm/long-vision]=vllm-qwen36-27b-long-vision
  [vllm/long-text]=vllm-qwen36-27b-long-text
  [vllm/long-text-no-mtp]=vllm-qwen36-27b-long-text-no-mtp
  [vllm/bounded-thinking]=vllm-qwen36-27b-bounded-thinking
  [vllm/tools-text]=vllm-qwen36-27b
  [vllm/minimal]=vllm-qwen36-27b-minimal
  [vllm/dual]=vllm-qwen36-27b-dual
  [vllm/dual-turbo]=vllm-qwen36-27b-dual-turbo
  [vllm/dual-dflash]=vllm-qwen36-27b-dual-dflash
  [vllm/dual-dflash-noviz]=vllm-qwen36-27b-dual-dflash-noviz
  [vllm/dual-nvlink]=vllm-qwen36-27b-dual-nvlink
  [llamacpp/default]=llama-cpp-qwen36-27b
  [llamacpp/concurrent]=llama-cpp-qwen36-27b-concurrent
)
ENDPOINT_PORT="${PORT:-${LAUNCH_DEFAULT_PORT[$VARIANT]:-8020}}"
ENDPOINT_URL="http://localhost:${ENDPOINT_PORT}"
ENDPOINT_CONTAINER="${CONTAINER:-${LAUNCH_DEFAULT_CONTAINER[$VARIANT]:-vllm-qwen36-27b}}"

if [[ $SKIP_VERIFY -eq 1 ]]; then
  echo "[launch] --no-verify — skipping verify-full.sh"
else
  echo ""
  echo "[launch] running verify-full.sh against the new server (URL=${ENDPOINT_URL}, CONTAINER=${ENDPOINT_CONTAINER})..."
  echo ""
  URL="$ENDPOINT_URL" CONTAINER="$ENDPOINT_CONTAINER" bash "$VERIFY" || {
    echo ""
    echo "[launch] some checks failed — see hints above. Common cases:"
    echo "  - 'reasoning field empty' on llama.cpp = expected (parser gap, not a bug)"
    echo "  - 'Genesis patches' / 'MTP acceptance' skipped on llama.cpp = expected (vLLM-only checks)"
    exit 1
  }
fi

echo ""
echo "[launch] done. Endpoint: ${ENDPOINT_URL}"
echo "[launch] sample request:"
echo "  curl -sf ${ENDPOINT_URL}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen3.6-27b-autoround\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":200}'"
echo ""
echo "[launch] switch later with:  bash scripts/switch.sh <variant>"
echo "[launch] list variants:      bash scripts/switch.sh --list"
