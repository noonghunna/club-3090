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
#
# All flags accept the same names as `switch.sh --list` produces.
# Examples:
#   bash scripts/launch.sh --variant vllm/default
#   bash scripts/launch.sh --variant llamacpp/default
#   bash scripts/launch.sh --variant vllm/dual

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SWITCH="${ROOT_DIR}/scripts/switch.sh"
VERIFY="${ROOT_DIR}/scripts/verify-full.sh"

# --- arg parsing ---
ENGINE=""
CARDS=""
VARIANT=""
SKIP_VERIFY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)  ENGINE="$2"; shift 2 ;;
    --cards)   CARDS="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --no-verify) SKIP_VERIFY=1; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

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
if [[ -z "$VARIANT" ]]; then
  echo ""
  echo "club-3090 launcher — pick a config and we'll boot it."
  echo "(Use --variant <name> next time to skip the wizard.)"
  echo ""

  if [[ -z "$ENGINE" ]]; then
    ENGINE=$(choose "Which engine?" \
      "vLLM — full features (vision + tools + spec-decode), max TPS, has prefill cliffs at long ctx" "vllm" \
      "llama.cpp — max context (262K), no prefill cliffs, simpler setup, slower (~21 TPS)" "llamacpp")
  fi

  if [[ "$ENGINE" == "vllm" ]]; then
    if [[ -z "$CARDS" ]]; then
      CARDS=$(choose "How many RTX 3090s?" \
        "1× 3090 (24 GB)"           "1" \
        "2× 3090 (PCIe, no NVLink)" "2")
    fi
    if [[ "$CARDS" == "1" ]]; then
      VARIANT=$(choose "What's your main workload?" \
        "Chat / coding agent (≤48K, tools work, recommended)"        "vllm/default" \
        "Long context with vision (192K — UNSAFE with 25K+ tool returns: Cliff 1)" "vllm/long-vision" \
        "Long text-only (205K — same Cliff 1 caveat)"                "vllm/long-text" \
        "Pure chat, max TPS (20K, fp8 KV)"                           "vllm/fast-chat" \
        "Long single prompts, text-only (75K, fp8 KV)"               "vllm/tools-text" \
        "Minimal — no Genesis, no spec-decode (32K)"                 "vllm/minimal")
    elif [[ "$CARDS" == "2" ]]; then
      VARIANT=$(choose "Dual-card priority?" \
        "Balanced default (262K + vision, 2 streams, fp8 KV)"        "vllm/dual" \
        "Multi-tenant (4 concurrent streams @ 262K, TQ3 KV)"         "vllm/dual-turbo" \
        "Peak code TPS with vision (185K, DFlash N=5)"               "vllm/dual-dflash" \
        "Peak code TPS no vision (200K, DFlash N=5)"                 "vllm/dual-dflash-noviz")
    fi
  elif [[ "$ENGINE" == "llamacpp" ]]; then
    VARIANT=$(choose "llama.cpp mode?" \
      "Single slot, full 262K context + vision (recommended)"      "llamacpp/default" \
      "4 parallel slots @ 192K pool + vision (multi-tenant)"       "llamacpp/concurrent")
  else
    echo "ERROR: unknown engine '${ENGINE}' (expected: vllm | llamacpp)" >&2
    exit 1
  fi
fi

# --- launch + verify ---
echo ""
echo "[launch] selected variant: ${VARIANT}"
echo ""
"$SWITCH" "$VARIANT"

if [[ $SKIP_VERIFY -eq 1 ]]; then
  echo "[launch] --no-verify — skipping verify-full.sh"
else
  echo ""
  echo "[launch] running verify-full.sh against the new server..."
  echo ""
  bash "$VERIFY" || {
    echo ""
    echo "[launch] some checks failed — see hints above. Common cases:"
    echo "  - 'reasoning field empty' on llama.cpp = expected (parser gap, not a bug)"
    echo "  - 'Genesis patches' / 'MTP acceptance' skipped on llama.cpp = expected (vLLM-only checks)"
    exit 1
  }
fi

echo ""
echo "[launch] done. Endpoint: http://localhost:8020"
echo "[launch] sample request:"
echo "  curl -sf http://localhost:8020/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen3.6-27b-autoround\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":30}'"
echo ""
echo "[launch] switch later with:  bash scripts/switch.sh <variant>"
echo "[launch] list variants:      bash scripts/switch.sh --list"
