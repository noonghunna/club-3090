#!/usr/bin/env bash
#
# Model-aware one-shot setup for club-3090.
#
#   bash scripts/setup.sh                # interactive model picker in a TTY
#   bash scripts/setup.sh <model-name>   # scripted/CI positional form
#
# Currently supported model families are listed in usage(). Exact repositories,
# files, and local subdirectories live in scripts/lib/profiles/models/*.yml.
#
# What it does (per supported model):
#   - clones Sandermage/genesis-vllm-patches into models/<model>/vllm/patches/genesis
#     (vLLM-only; skip with SKIP_GENESIS=1 if you only need llama.cpp / SGLang;
#     Gemma 4 doesn't fetch Genesis at all yet)
#   - downloads model weights into $MODEL_DIR with SHA256 verification
#     against HF x-linked-etag
#   - downloads the always-required drafter (Gemma 4: MTP "assistant"; Qwen3.6:
#     no always-required drafter — DFlash is optional via WITH_DFLASH_DRAFT=1)
#
# Env vars (optional):
#   MODEL_DIR           Where to place model weights. Default: <repo>/models-cache
#   WEIGHTS             'autoround' (default, vLLM INT4) or 'gguf' (llama.cpp /
#                       ik_llama). gguf fetches the Q4_K_M MTP GGUF + mmproj for
#                       the llamacpp/* + ik-llama/* composes (qwen3.6-27b only),
#                       and skips Genesis. Use this if you're serving via
#                       llama.cpp/ik_llama rather than vLLM.
#   HF_TOKEN            HF token (public models, usually unnecessary)
#   SKIP_MODEL          Set to 1 to skip the model download step
#   SKIP_GENESIS        Set to 1 to skip cloning Genesis patches
#   WITH_DFLASH_DRAFT   Set to 1 to ALSO download the model family's DFlash
#                       drafter when one is registered in profiles/models/*.yml.
#                       Default: 0.
#                       Note: draft model is still under training as of
#                       2026-04-26; bench numbers in DUAL_CARD.md were
#                       measured against that snapshot. AL improvements
#                       expected when z-lab tags training-complete.
#   PREFLIGHT_DISK_GB   Required free space at MODEL_DIR (default: 25, or
#                       28 if WITH_DFLASH_DRAFT=1)
#
# Idempotent: safe to re-run — skips steps already done.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_READER="${ROOT_DIR}/scripts/lib/profiles/weights.py"

usage() {
  echo "Usage: $0 <model-name>"
  echo "       $0              # interactive model picker in a TTY"
  echo ""
  echo "Run with no model name in a normal terminal to open the hardware-aware"
  echo "model picker. Use the positional form in scripts/CI to skip prompts."
  echo ""
  echo "Supported model names:"
  echo "  qwen3.6-27b"
  echo "  qwen3.6-35b-a3b"
  echo "  gemma-4-31b"
  echo "  gemma-4-26b-a4b"
  echo ""
  echo "Exact catalog entry fetch: WEIGHT_KEY=<registry-key> $0 <model-name>"
}

model_label() {
  case "$1" in
    qwen3.6-27b) echo "Qwen 3.6 27B" ;;
    qwen3.6-35b-a3b) echo "Qwen 3.6 35B-A3B" ;;
    gemma-4-31b) echo "Gemma 4 31B" ;;
    gemma-4-26b-a4b) echo "Gemma 4 26B-A4B" ;;
    diffusiongemma-26b-a4b) echo "DiffusionGemma 26B-A4B (dLLM)" ;;
    *) echo "$1" ;;
  esac
}

load_weight_recipe() {
  local key="$1"
  local env_lines
  command -v python3 >/dev/null 2>&1 || {
    echo "ERROR: python3 is required to read profile weight recipes." >&2
    exit 1
  }
  env_lines="$(python3 "${WEIGHTS_READER}" entry "$key" 2>/dev/null)" || {
    echo "ERROR: could not resolve weight recipe '${key}' from scripts/lib/profiles/models/*.yml." >&2
    echo "       Install python3-yaml/PyYAML if missing, or check the catalog key." >&2
    exit 1
  }
  eval "$env_lines"
  if [[ -n "${WEIGHT_MODEL:-}" && "${WEIGHT_MODEL}" != "${MODEL_NAME}" ]]; then
    echo "ERROR: weight recipe '${key}' belongs to ${WEIGHT_MODEL}, not ${MODEL_NAME}." >&2
    exit 1
  fi
  if [[ -z "${WEIGHT_REPO:-}" ]]; then
    echo "ERROR: weight recipe '${key}' has no direct download recipe." >&2
    [[ -n "${WEIGHT_MANUAL_NOTE:-}" ]] && echo "       ${WEIGHT_MANUAL_NOTE}" >&2
    exit 1
  fi
  MODEL_REPO="${WEIGHT_REPO}"
  MODEL_REVISION="${WEIGHT_REVISION:-}"
  MODEL_SUBDIR="${WEIGHT_SUBDIR}"
  GGUF_FILES="${WEIGHT_FILES}"
  VERIFY_GLOB="${WEIGHT_VERIFY_GLOB:-*.safetensors}"
  echo "[model]   ${WEIGHT_KEY} -> ${MODEL_REPO} ${WEIGHT_FILES} -> ${MODEL_SUBDIR}"
}

model_picker_line() {
  local idx="$1" model="$2" size="$3" status mark reason
  status="$(compose_hw_model_status "$ROOT_DIR" "$model" 2>/dev/null || true)"
  reason="${status#*|}"
  if [[ "$status" == ok\|* ]]; then
    mark="✓"
  else
    mark="✗"
  fi
  printf "  %s. %-14s (%s)  %s %s\n" "$idx" "$(model_label "$model")" "$size" "$mark" "$reason"
}

pick_model_interactive() {
  # shellcheck source=lib/compose-meta.sh
  source "${ROOT_DIR}/scripts/lib/compose-meta.sh"

  echo "[setup] Which model to download?" >&2
  echo "" >&2
  model_picker_line "1" "qwen3.6-27b" "~14 GB AutoRound INT4" >&2
  model_picker_line "2" "gemma-4-31b" "~21 GB AutoRound INT4 + drafter" >&2
  echo "  3. Both           (~30 GB total)  downloads both model families" >&2
  echo "" >&2
  while true; do
    local pick
    read -rp "Choice [1-3]: " pick
    case "$pick" in
      1) echo "qwen3.6-27b"; return ;;
      2) echo "gemma-4-31b"; return ;;
      3) echo "both"; return ;;
      *) echo "  ! invalid — pick 1, 2, or 3" >&2 ;;
    esac
  done
}

# ---------- Model dispatch ----------
case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

# setup.sh takes a SINGLE positional (the model). It DOWNLOADS WEIGHTS — it does
# NOT select or boot a serving config. A stray second arg (commonly a launch slug
# like `vllm/int8`) used to be silently ignored, which let users believe it did
# something — see issue #250, where the slug was dropped and the real failure
# (a purged-nightly compose pin) got mis-attributed to it. Reject it loudly and
# point at the launch path. (`both` recurses with a single arg, so it's unaffected.)
if [[ $# -gt 1 ]]; then
  echo "ERROR: setup.sh takes a single model name; got extra argument(s): ${*:2}" >&2
  echo "       setup.sh only DOWNLOADS WEIGHTS for a model — e.g. bash scripts/setup.sh ${1}" >&2
  echo "       To LAUNCH a serving config (a slug such as 'vllm/gemma-int8-mtp'), use:" >&2
  echo "         bash scripts/launch.sh --variant <slug>      # or: bash scripts/switch.sh <slug>" >&2
  echo "       See the slugs available for a model:  bash scripts/switch.sh --list" >&2
  exit 64
fi

MODEL_NAME="${1:-}"
if [[ -z "${MODEL_NAME}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    MODEL_NAME="$(pick_model_interactive)"
  else
    usage
    echo ""
    echo "(Interactive picker available in a TTY shell. Use the positional form in scripts/CI.)"
    exit 1
  fi
fi

if [[ "${MODEL_NAME}" == "both" ]]; then
  # Resolve MODEL_DIR once in the parent by reusing the normal prompt below,
  # then recurse through the positional form for each model.
  SETUP_BOTH_MODE=1
  MODEL_NAME="qwen3.6-27b"
else
  SETUP_BOTH_MODE=0
fi

# The profile YAMLs are the only source of download recipes. setup.sh only
# maps friendly setup knobs (MODEL_NAME, WEIGHTS, WITH_*) to profile keys.
ALWAYS_DRAFT_KEY=""
DFLASH_KEY=""
PRIMARY_WEIGHT_KEY=""
EXTRA_WEIGHT_KEYS=()
# Genesis is opt-in: nothing in the shipped catalog requires it anymore (its last
# unique feature on this stack — turboquant_3bit_nc KV — is now provided by beellama
# KVarN; #182). The clone only fires if a user explicitly sets NEEDS_GENESIS=1 (e.g.
# to revive an archived TQ3 compose). The per-model dispatch below no longer flips it on.
NEEDS_GENESIS="${NEEDS_GENESIS:-0}"

case "${MODEL_NAME}" in
  qwen3.6-27b)
    PRIMARY_WEIGHT_KEY="qwen3.6-27b:autoround-int4"
    DFLASH_KEY="qwen3.6-27b:dflash"
    # Genesis no longer auto-requested: qwen3.6-27b.yml declares requires_genesis:false
    # and the live autoround composes run overlay-free on vllm-stable v0.22.0 (#182).
    ;;
  qwen3.6-35b-a3b)
    PRIMARY_WEIGHT_KEY="qwen3.6-35b-a3b:autoround-int4"
    ;;
  gemma-4-31b)
    PRIMARY_WEIGHT_KEY="gemma-4-31b:autoround-int4"
    ALWAYS_DRAFT_KEY="gemma-4-31b:assistant"
    DFLASH_KEY="gemma-4-31b:dflash"
    ;;
  gemma-4-26b-a4b)
    PRIMARY_WEIGHT_KEY="gemma-4-26b-a4b:autoround-int4-mixed"
    ;;
  diffusiongemma-26b-a4b)
    # dLLM, fp8 only (no autoround variant). Default WEIGHTS=autoround is a no-op
    # here — the fp8 key set below is what's fetched.
    PRIMARY_WEIGHT_KEY="diffusiongemma-26b-a4b:fp8"
    ;;
  *)
    echo "ERROR: unsupported model '${MODEL_NAME}'."
    echo "Supported: qwen3.6-27b, qwen3.6-35b-a3b, gemma-4-31b, gemma-4-26b-a4b, diffusiongemma-26b-a4b"
    echo "(To add a new model, extend the model dispatch in scripts/setup.sh and profiles/models/*.yml)"
    exit 1
    ;;
esac

# ---------- Weights format / exact registry entry ----------
# WEIGHTS selects a common weight variant for the model family. WEIGHT_KEY is
# the exact catalog-entry path used by preflight's fetch-now flow.
WEIGHTS="${WEIGHTS:-autoround}"
GGUF_FILES=""
VERIFY_GLOB="*.safetensors"

if [[ -n "${WEIGHT_KEY:-}" ]]; then
  PRIMARY_WEIGHT_KEY="${WEIGHT_KEY}"
elif [[ "${WEIGHTS}" == "gguf" ]]; then
  case "${MODEL_NAME}" in
    qwen3.6-27b)
      PRIMARY_WEIGHT_KEY="qwen3.6-27b:unsloth-q4km"
      EXTRA_WEIGHT_KEYS+=("qwen3.6-27b:gguf_mmproj_f16")
      NEEDS_GENESIS=0
      ;;
    *)
      echo "ERROR: WEIGHTS=gguf is only wired for qwen3.6-27b right now." >&2
      echo "       Use WEIGHT_KEY=<model>:<variant> for exact catalog entries." >&2
      exit 1 ;;
  esac
elif [[ "${WEIGHTS}" == "iq4ks" ]]; then
  case "${MODEL_NAME}" in
    qwen3.6-27b)
      PRIMARY_WEIGHT_KEY="qwen3.6-27b:ubergarm-iq4ks"
      NEEDS_GENESIS=0
      ;;
    *)
      echo "ERROR: WEIGHTS=iq4ks is only wired for qwen3.6-27b." >&2
      exit 1 ;;
  esac
elif [[ "${WEIGHTS}" == "awq" ]]; then
  case "${MODEL_NAME}" in
    gemma-4-31b) PRIMARY_WEIGHT_KEY="gemma-4-31b:awq" ;;
    gemma-4-26b-a4b) PRIMARY_WEIGHT_KEY="gemma-4-26b-a4b:awq" ;;
    *)
      echo "ERROR: WEIGHTS=awq is only wired for gemma-4-31b and gemma-4-26b-a4b." >&2
      exit 1 ;;
  esac
elif [[ "${WEIGHTS}" == "fp8" ]]; then
  case "${MODEL_NAME}" in
    qwen3.6-27b)
      PRIMARY_WEIGHT_KEY="qwen3.6-27b:fp8"
      NEEDS_GENESIS=0
      ;;
    *)
      echo "ERROR: WEIGHTS=fp8 is only wired for qwen3.6-27b." >&2
      exit 1 ;;
  esac
elif [[ "${WEIGHTS}" == "qwopus-coder" ]]; then
  # Qwopus3.6-27B-Coder Q5_K_M GGUF (embedded MTP head) for beellama/qwopus-coder.
  case "${MODEL_NAME}" in
    qwen3.6-27b)
      PRIMARY_WEIGHT_KEY="qwen3.6-27b:qwopus-coder-mtp-q5km"
      NEEDS_GENESIS=0
      ;;
    *)
      echo "ERROR: WEIGHTS=qwopus-coder is only wired for qwen3.6-27b." >&2
      exit 1 ;;
  esac
elif [[ "${WEIGHTS}" == "carnice-v2" ]]; then
  # Carnice-V2-27B Q5_K_M GGUF (embedded MTP head) for beellama/carnice-v2-single-q5km-mtp.
  case "${MODEL_NAME}" in
    qwen3.6-27b)
      PRIMARY_WEIGHT_KEY="qwen3.6-27b:carnice-v2-q5km"
      NEEDS_GENESIS=0
      ;;
    *)
      echo "ERROR: WEIGHTS=carnice-v2 is only wired for qwen3.6-27b." >&2
      exit 1 ;;
  esac
elif [[ "${WEIGHTS}" != "autoround" ]]; then
  echo "ERROR: WEIGHTS='${WEIGHTS}' not recognized (use 'autoround', 'awq', 'fp8', 'qwopus-coder', 'carnice-v2', 'gguf', or 'iq4ks')." >&2
  exit 1
fi

if [[ "${WITH_ASSISTANT_DRAFT:-0}" == "1" ]]; then
  case "${MODEL_NAME}" in
    gemma-4-31b) ALWAYS_DRAFT_KEY="gemma-4-31b:assistant" ;;
    gemma-4-26b-a4b) ALWAYS_DRAFT_KEY="gemma-4-26b-a4b:assistant" ;;
    *)
      echo "ERROR: WITH_ASSISTANT_DRAFT=1 is only wired for Gemma models." >&2
      exit 1 ;;
  esac
fi

load_weight_recipe "${PRIMARY_WEIGHT_KEY}"

# ---------- MODEL_DIR resolution ----------
# Order of precedence:
#   1. MODEL_DIR already exported in the calling shell  → use as-is
#   2. .env at repo root sets MODEL_DIR                  → source it
#   3. Interactive prompt (only if stdin is a TTY)       → ask user
#   4. Silent fallback to <repo>/models-cache            → in-repo default
#
# The prompt only fires for fresh users on a TTY who haven't set anything.
# CI / scripted runs (no TTY) get the silent fallback, preserving prior behavior.

# Step 2: source repo-root .env if present (lets a saved choice persist)
if [[ -z "${MODEL_DIR:-}" && -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck source=/dev/null
  set -a; source "${ROOT_DIR}/.env"; set +a
fi

# Step 3: prompt if still unset + interactive
if [[ -z "${MODEL_DIR:-}" && -t 0 && -t 1 ]]; then
  echo ""
  echo "Where should I put model weights?"
  echo "  Models are large (Qwen3.6-27B AutoRound: ~14 GB; Gemma 4 31B: ~21 GB)."
  echo "  This dir lives outside the git tree — pick a location with sufficient free space."
  echo ""
  echo "  1) ${ROOT_DIR}/models-cache  (in-repo, default — pollutes git tree)"
  echo "  2) ${HOME}/models             (recommended for cross-rig — outside repo)"
  echo "  3) custom path"
  echo ""
  while true; do
    read -rp "Choice [1-3] (or set MODEL_DIR env var to skip): " pick
    case "${pick}" in
      1) MODEL_DIR="${ROOT_DIR}/models-cache"; break ;;
      2) MODEL_DIR="${HOME}/models"; break ;;
      3)
        read -rp "  Enter absolute path: " custom
        if [[ "${custom}" =~ ^/ ]]; then
          MODEL_DIR="${custom}"; break
        else
          echo "  ! must be an absolute path (start with /)" >&2
        fi
        ;;
      *) echo "  ! invalid — pick 1, 2, or 3" >&2 ;;
    esac
  done
  echo ""

  # Offer to persist the choice so future runs skip the prompt
  read -rp "Save MODEL_DIR=${MODEL_DIR} to .env so we skip this next time? [Y/n]: " save
  if [[ "${save:-y}" =~ ^[Yy]$ || -z "${save:-}" ]]; then
    if [[ -f "${ROOT_DIR}/.env" ]]; then
      # Update existing .env (replace MODEL_DIR= line if present, else append)
      if grep -qE "^MODEL_DIR=" "${ROOT_DIR}/.env"; then
        sed -i "s|^MODEL_DIR=.*|MODEL_DIR=${MODEL_DIR}|" "${ROOT_DIR}/.env"
      else
        echo "MODEL_DIR=${MODEL_DIR}" >> "${ROOT_DIR}/.env"
      fi
    else
      echo "MODEL_DIR=${MODEL_DIR}" > "${ROOT_DIR}/.env"
    fi
    echo "  → saved. (.env is gitignored.)"
  else
    echo "  → not saved. Set MODEL_DIR=... when re-running, or you'll get this prompt again."
  fi
  echo ""
fi

# Step 4: silent fallback (preserves prior behavior for non-TTY contexts)
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models-cache}"
if [[ "${SETUP_BOTH_MODE:-0}" == "1" ]]; then
  export MODEL_DIR
  echo "[setup] downloading both supported models into ${MODEL_DIR}"
  echo ""
  bash "$0" qwen3.6-27b
  echo ""
  bash "$0" gemma-4-31b
  echo ""
  echo "[setup] ✓ Both models downloaded."
  echo "[setup] Next: bash scripts/launch.sh"
  exit 0
fi
GENESIS_DIR="${ROOT_DIR}/models/${MODEL_NAME}/vllm/patches/genesis"

cd "${ROOT_DIR}"

# ---------- Pre-flight checks ----------
# Catches the common "first-run failures": missing docker, no GPU visible,
# disk too small for the ~14 GB AutoRound int4 download. Fails fast with
# actionable hints rather than mid-download or first-boot crash.
# shellcheck source=preflight.sh
source "${ROOT_DIR}/scripts/preflight.sh"

# Required disk: model is ~14 GB on disk; 25 GB gives buffer for download
# temp files + safetensors + tokenizer/config. Add ~3 GB if also pulling
# the DFlash draft (~1.75 GB packed + buffer for download tempfiles).
if [[ "${WITH_DFLASH_DRAFT:-0}" == "1" ]]; then
  PREFLIGHT_DISK_GB="${PREFLIGHT_DISK_GB:-28}"
else
  PREFLIGHT_DISK_GB="${PREFLIGHT_DISK_GB:-25}"
fi

echo "[preflight] checking environment..."
# docker is soft-warn for setup.sh — this script only fetches genesis + models,
# no docker invocations until you actually `docker compose up` later. Hard-failing
# blocks non-docker container-runtime users (microk8s / podman / k8s / manual)
# from running setup at all (club-3090 disc #48). launch.sh keeps the hard check
# because it actually invokes docker.
preflight_docker || echo "[preflight] WARN:  docker unavailable — setup will continue (genesis + model fetch don't need docker), but you'll need a working container runtime before 'docker compose up'."
preflight_gpu 1  || exit 1
preflight_disk "${MODEL_DIR}" "${PREFLIGHT_DISK_GB}" || exit 1
preflight_hf_token  # soft-warn only; downloads will surface the hard failure
echo "[preflight] ok."
echo ""

# ---------- WSL2 detection — auto-configure .env for known WSL2 boot crash ----------
# WSL2 + driver 596.36 + vLLM nightly hit a `gptq_marlin_repack` boot crash
# with `cudaErrorNotReady`. Workaround is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
# (PR #84). The compose default is `expandable_segments:True,max_split_size_mb:512`
# which works on bare-metal Linux but fails on WSL2 — so we auto-create a .env
# override here on detected WSL2 systems. Cross-rig validated by @timxx (issue #60),
# @easel, and others. Safe no-op on bare-metal (only runs when /proc/version
# contains "microsoft").
COMPOSE_DIR="${ROOT_DIR}/models/${MODEL_NAME}/vllm/compose"
if [[ -f /proc/version ]] && grep -qi microsoft /proc/version 2>/dev/null; then
  ENV_FILE="${COMPOSE_DIR}/.env"
  if [[ -d "${COMPOSE_DIR}" ]]; then
    if [[ ! -f "${ENV_FILE}" ]]; then
      cat > "${ENV_FILE}" <<'EOF'
# WSL2 boot-crash workaround — see PR #84 + issue #60.
# vLLM + WSL2 + driver 596.36 hit `gptq_marlin_repack` cudaErrorNotReady on boot
# with the default `expandable_segments:True`. This override fixes it.
# Auto-created by scripts/setup.sh on detected WSL2 systems. Safe to delete
# on bare-metal Linux (the compose default works there).
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
EOF
      echo "[wsl2] detected WSL2 — created ${ENV_FILE} with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
      echo "[wsl2] this fixes the known gptq_marlin_repack boot crash on WSL2 + driver ≥596.36 (issue #60)."
    elif ! grep -q "expandable_segments:False" "${ENV_FILE}"; then
      echo "[wsl2] WARN: detected WSL2 but ${ENV_FILE} exists without the expandable_segments:False override."
      echo "[wsl2]       If vLLM fails to boot with cudaErrorNotReady, add:"
      echo "[wsl2]         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
      echo "[wsl2]       See PR #84 / issue #60 for context."
    else
      echo "[wsl2] detected WSL2 — ${ENV_FILE} already has the expandable_segments:False override. ✓"
    fi
  fi
fi

# ---------- Tool checks ----------
need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required tool '$1' not found in PATH." >&2
    exit 1
  }
}
need git
need curl
need sha256sum

echo "Setup root:   ${ROOT_DIR}"
echo "Model dir:    ${MODEL_DIR}"

# ---------- Genesis patches (opt-in only) ----------
# Genesis is no longer fetched by default. No shipped compose requires it: its one
# unique feature on this stack — turboquant_3bit_nc KV — is now covered by beellama
# KVarN, and the Genesis-anchored vLLM engines were deprecated when the catalog moved
# to stock vllm/vllm-openai:v0.22.0 (their TQ3/Genesis composes live in compose/_archive/).
# We are not reviving Genesis in the near term (#182).
#
# The pin below is retained ONLY for an explicit revival: running
#   NEEDS_GENESIS=1 bash scripts/setup.sh qwen3.6-27b
# clones Sandermage's tree at this commit so an archived TQ3 compose can be brought back.
# Bumping it is parked on Sander's next *stable* Genesis tag (the in-flight memory-rework
# WIP supersedes the 7b9fd319 skeleton); any bump requires re-running verify-stress.sh
# against the revived compose. Override with GENESIS_PIN=<ref> if reviving against another.
GENESIS_PIN="${GENESIS_PIN:-7b9fd319}"

if [[ "${NEEDS_GENESIS:-1}" != "1" ]]; then
  echo "[genesis] ${MODEL_NAME} doesn't use Genesis — skipping clone."
elif [[ "${SKIP_GENESIS:-0}" != "1" ]]; then
  if [[ -d "${GENESIS_DIR}/.git" ]]; then
    echo "[genesis] Already cloned at ${GENESIS_DIR} — fetching + checking out ${GENESIS_PIN} ..."
    (cd "${GENESIS_DIR}" && git fetch origin && git checkout "${GENESIS_PIN}" 2>&1 | tail -3)
  else
    echo "[genesis] Cloning Sandermage/genesis-vllm-patches at ${GENESIS_PIN} ..."
    # Full clone (commit SHAs aren't reachable via --branch + --depth 1).
    git clone https://github.com/Sandermage/genesis-vllm-patches.git "${GENESIS_DIR}"
    (cd "${GENESIS_DIR}" && git checkout "${GENESIS_PIN}")
  fi

  # v7.14+ layout sanity check
  if [[ ! -d "${GENESIS_DIR}/vllm/_genesis" ]]; then
    echo "ERROR: genesis tree at ${GENESIS_PIN} missing vllm/_genesis package." >&2
    echo "       Re-run with GENESIS_PIN=<other-ref> to try a different version." >&2
    exit 1
  fi
  echo "[genesis] Pinned to ${GENESIS_PIN} ($(cd "${GENESIS_DIR}" && git rev-parse --short HEAD))"

  # v7.69 ships PN25 + PN30 + PN34 directly (Sander's accept-and-fold of our
  # cross-rig sidecars). Local patch_pn25_genesis_register_fix.py +
  # patch_pn30_dst_shaped_temp_fix.py + patch_workspace_lock_disable.py are
  # now redundant. Sidecar Python files retained in vllm/patches/ for
  # rollback if any v7.69 patch regresses on your config.
else
  echo "[genesis] SKIP_GENESIS=1 — not cloning."
fi

# ---------- Model download ----------
if [[ "${SKIP_MODEL:-0}" == "1" ]]; then
  echo "[model]   SKIP_MODEL=1 — not downloading."
  exit 0
fi

_hf_download_repo() {
  local repo="$1"
  local subdir="$2"
  local files="${3:-}"
  local revision="${4:-}"
  # Optional commit-SHA / tag pin (#319). Empty -> track HEAD (today's behavior).
  local rev_args=()
  [[ -n "$revision" ]] && rev_args=(--revision "$revision")
  mkdir -p "${MODEL_DIR}/${subdir}"
  if command -v hf >/dev/null 2>&1; then
    echo "[model]   Using 'hf download' (hf_transfer if available) ..."
    # files is intentionally word-split: empty -> whole repo; non-empty -> selected files.
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
      hf download "$repo" ${files} "${rev_args[@]}" --local-dir "${MODEL_DIR}/${subdir}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[model]   Using 'huggingface-cli download' ..."
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
      huggingface-cli download "$repo" ${files} "${rev_args[@]}" --local-dir "${MODEL_DIR}/${subdir}"
  else
    echo "ERROR: neither 'hf' nor 'huggingface-cli' found. Install with:" >&2
    echo "  pip install 'huggingface-hub[hf_transfer]'" >&2
    echo "or:" >&2
    echo "  uv tool install --with hf_transfer huggingface-hub" >&2
    exit 1
  fi
}

_verify_downloaded_files() {
  local repo="$1"
  local subdir="$2"
  local verify_glob="$3"
  # Pin the etag lookup to the same revision we downloaded (#319). Empty -> main
  # HEAD; a stale pin would otherwise etag-check against a newer HEAD and FAIL.
  local revision="${4:-main}"
  local fail=0 count=0 f expected actual

  echo "[verify]  Checking SHA256 of every ${verify_glob} against HF x-linked-etag (rev: ${revision}) ..."
  cd "${MODEL_DIR}/${subdir}"
  for f in ${verify_glob}; do
    [[ -f "$f" ]] || continue
    count=$((count + 1))
    expected="$(curl -sfI "https://huggingface.co/${repo}/resolve/${revision}/$f" \
      | grep -i '^x-linked-etag:' | tr -d '"\r' | awk '{print $NF}' || true)"
    actual="$(sha256sum "$f" | awk '{print $1}')"
    if [[ -z "$expected" ]]; then
      printf "  %-50s SKIP (no etag)\n" "$f"
    elif [[ "$expected" == "$actual" ]]; then
      printf "  %-50s OK\n" "$f"
    else
      printf "  %-50s FAIL  exp=%.12s  act=%.12s\n" "$f" "$expected" "$actual"
      fail=$((fail + 1))
    fi
  done
  cd "${ROOT_DIR}"

  if [[ "$fail" != "0" ]]; then
    echo "[verify]  ${fail} file(s) failed SHA check." >&2
    echo "          Delete ${MODEL_DIR}/${subdir} and re-run setup.sh." >&2
    exit 1
  fi
  if [[ "$count" == "0" ]]; then
    echo "[verify]  No ${verify_glob} found in ${MODEL_DIR}/${subdir} — download may have failed." >&2
    exit 1
  fi
  echo "[done]    ${count} file(s) SHA-verified in ${subdir}."
}

download_weight_key() {
  local key="$1"
  load_weight_recipe "$key"
  echo "[model]   Downloading ${WEIGHT_LABEL:-$key} ..."
  _hf_download_repo "$WEIGHT_REPO" "$WEIGHT_SUBDIR" "$WEIGHT_FILES" "${WEIGHT_REVISION:-}"
  _verify_downloaded_files "$WEIGHT_REPO" "$WEIGHT_SUBDIR" "$WEIGHT_VERIFY_GLOB" "${WEIGHT_REVISION:-main}"
}

VERIFY_GLOB="${VERIFY_GLOB_OVERRIDE:-*.safetensors}"
_hf_download_repo "${MODEL_REPO}" "${MODEL_SUBDIR}" "${GGUF_FILES}" "${MODEL_REVISION:-}"
_verify_downloaded_files "${MODEL_REPO}" "${MODEL_SUBDIR}" "${VERIFY_GLOB}" "${MODEL_REVISION:-main}"

for extra_key in "${EXTRA_WEIGHT_KEYS[@]}"; do
  download_weight_key "$extra_key"
done

echo ""
[[ -d "${GENESIS_DIR}/.git" ]] && echo "          Genesis pinned at ${GENESIS_PIN} ($(cd "${GENESIS_DIR}" && git rev-parse --short HEAD))."
echo ""

# ---------- Optional / companion draft models ----------
if [[ -n "${ALWAYS_DRAFT_KEY:-}" ]] && [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  echo "[draft]   downloading required companion drafter ${ALWAYS_DRAFT_KEY} ..."
  download_weight_key "${ALWAYS_DRAFT_KEY}"
  echo ""
fi

if [[ "${WITH_DFLASH_DRAFT:-0}" == "1" ]] && [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  if [[ -z "${DFLASH_KEY:-}" ]]; then
    echo "ERROR: WITH_DFLASH_DRAFT=1 is not wired for ${MODEL_NAME}." >&2
    exit 1
  fi
  echo "[dflash]  WITH_DFLASH_DRAFT=1 — downloading ${DFLASH_KEY} ..."
  download_weight_key "${DFLASH_KEY}"
  echo ""
else
  echo "[dflash]  Skipping DFlash draft model. Set WITH_DFLASH_DRAFT=1 to fetch it when a matching compose requires it."
fi

echo ""

if [[ "${WITH_PRISM_EAGLE3:-0}" == "1" ]] && [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  case "${MODEL_NAME}" in
    qwen3.6-27b)
      download_weight_key qwen3.6-27b:prism_eagle3
      ;;
    *)
      echo "ERROR: WITH_PRISM_EAGLE3=1 is only wired for qwen3.6-27b." >&2
      exit 1 ;;
  esac
  echo ""
fi

# Note: vllm#40361 Marlin pad-sub-tile-n patched files are vendored in-repo
# at models/qwen3.6-27b/vllm/patches/vllm-marlin-pad/. Dual-card composes
# mount them via repo-relative paths — no host filesystem dependency, no
# clone needed. (Previous design required cloning a fork to /opt/ai/engines/vllm/primary/;
# refactored 2026-05-03 to vendor the two files in-repo, fixing #37.)

# Per-model "next steps" — different composes / served-model-name / port between models.
SETUP_MODEL_DISPLAY="$(model_label "${MODEL_NAME}")"
case "${MODEL_NAME}" in
  qwen3.6-27b)
    SAMPLE_CONTAINER="vllm-qwen36-27b"
    SAMPLE_COMPOSE_FLAGS_DUAL=" -f dual/autoround-int4/fp8-mtp.yml"
    SAMPLE_PORT="8020"
    SAMPLE_MODEL_NAME="qwen3.6-27b-autoround"
    NEXT_STEPS_NOTE="Or dual-card vLLM (Marlin patched files already vendored in-repo):
  cd models/${MODEL_NAME}/vllm/compose && docker compose -f dual/autoround-int4/fp8-mtp.yml up -d"
    ;;
  gemma-4-31b)
    SAMPLE_CONTAINER="vllm-gemma-4-31b-mtp"
    # Gemma 4 ships specific composes — pick MTP as the canonical default.
    # Use scripts/switch.sh which auto-selects the right compose by variant.
    SAMPLE_COMPOSE_FLAGS_DUAL=""
    SAMPLE_PORT="8030"
    SAMPLE_MODEL_NAME="gemma-4-31b-autoround"
    NEXT_STEPS_NOTE="Available variants:
  bash scripts/switch.sh vllm/gemma-bf16-mtp        # MTP drafter, TP=2, port 8030 (dual-card)
  bash scripts/switch.sh beellama/gemma-dflash # DFlash, single-card default, port 8061"
    ;;
  gemma-4-26b-a4b)
    SAMPLE_CONTAINER="vllm-gemma-4-26b-a4b"
    SAMPLE_COMPOSE_FLAGS_DUAL=""
    SAMPLE_PORT="8035"
    SAMPLE_MODEL_NAME="gemma-4-26b-a4b"
    NEXT_STEPS_NOTE="Available variants:
  bash scripts/switch.sh vllm/gemma-26b-awq
  WITH_ASSISTANT_DRAFT=1 bash scripts/setup.sh gemma-4-26b-a4b  # fetch MTP assistant if using awq-mtp"
    ;;
  diffusiongemma-26b-a4b)
    SAMPLE_CONTAINER="vllm-diffusiongemma-26b-a4b-fp8-tp2"
    SAMPLE_COMPOSE_FLAGS_DUAL=""
    SAMPLE_PORT="8042"
    SAMPLE_MODEL_NAME="diffusiongemma-26b-a4b"
    NEXT_STEPS_NOTE="🧪 experimental dLLM (vLLM's first), dual-card. Launch needs --force (non-functional status):
  bash scripts/switch.sh --force vllm/diffusiongemma-dual
  # or: gpu-mode dgemma   (stops other GPU models, serves on :8199)"
    ;;
  qwen3.6-35b-a3b)
    SAMPLE_CONTAINER="vllm-qwen36-35b-a3b"
    SAMPLE_COMPOSE_FLAGS_DUAL=""
    SAMPLE_PORT="8040"
    SAMPLE_MODEL_NAME="qwen3.6-35b-a3b-autoround"
    NEXT_STEPS_NOTE="Preview variants:
  bash scripts/switch.sh vllm/qwen35-preview"
    ;;
esac

echo "[setup] ✓ ${SETUP_MODEL_DISPLAY} downloaded."
echo "[setup] Next: bash scripts/launch.sh"
echo ""
echo "Next — single-card vLLM (default):"
if [[ "${MODEL_NAME}" == "gemma-4-31b" ]]; then
  echo "  bash scripts/switch.sh vllm/gemma-bf16-mtp"
  echo "  docker logs -f ${SAMPLE_CONTAINER}"
else
  echo "  cd models/${MODEL_NAME}/vllm/compose && docker compose up -d"
  echo "  docker logs -f ${SAMPLE_CONTAINER}"
fi
echo ""
echo "${NEXT_STEPS_NOTE}"
echo ""
echo "Sanity test (after 'Application startup complete'):"
echo "  curl -sf http://localhost:${SAMPLE_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${SAMPLE_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":200}'"
