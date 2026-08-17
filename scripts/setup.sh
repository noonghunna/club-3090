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
#   - downloads model weights into $MODEL_DIR with SHA256 verification against
#     the hub's published x-linked-etag (no-redirect resolve HEAD via
#     scripts/lib/profiles/hf_fetch.py). A file the hub publishes no hash for is
#     reported UNVERIFIED and EXCLUDED from the verified count — "verified"
#     never attaches to a size-only check (#857)
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

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

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
  echo "  qwen3.8-27b"
  echo "  gemma-4-26b-a4b"
  echo "  deepseek-v4-flash-0731"
  echo ""
  echo "Exact catalog entry fetch: WEIGHT_KEY=<registry-key> $0 <model-name>"
}

model_label() {
  case "$1" in
    qwen3.6-27b) echo "Qwen 3.6 27B" ;;
    qwen3.6-35b-a3b) echo "Qwen 3.6 35B-A3B" ;;
    deepseek-v4-flash-0731) echo "DeepSeek-V4-Flash-0731 (284B MoE, CPU offload)" ;;
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
  echo "       To LAUNCH a serving config (a slug such as 'vllm/gemma-31b-dual'), use:" >&2
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
  qwen3.8-27b)
    # Defaults to the fp8 "max accuracy" tier — what DUAL_CARD.md leads with, and the
    # only qwen3.8 config with measured numbers (67.4 / 85.8 tok/s, verify-full PASS,
    # 2026-08-17). DUAL-card: 30.9 GB of weights do not fit one 24 GB card.
    # Single card: WEIGHT_KEY=qwen3.8-27b:unsloth-iq4nl (131K, llama.cpp).
    #
    # Excluded from this dispatch until 2026-08-17 on the grounds that "both its slugs
    # are incubating and nothing has booted". All three parts expired: 10 slugs exist,
    # 6 are experimental, and dual-max booted + verify-full + benched. The docs had
    # meanwhile been rewritten to lead with it, so users were sent to a model that
    # setup.sh refused BY NAME (Discord, 2026-08-17). Re-check if statuses regress.
    PRIMARY_WEIGHT_KEY="qwen3.8-27b:fp8"
    ;;
  deepseek-v4-flash-0731)
    # Defaults to the IQ2 REACH tier (~85 GB on disk, ~86 GB host RAM) rather than
    # Q8 (~151 GB / ~146 GB host RAM): the reach tier is the one most rigs can
    # actually run, and a wrong guess here costs the user a 151 GB download.
    # Q8 instead:  WEIGHT_KEY=deepseek-v4-flash-0731:unsloth-q8-kxl scripts/setup.sh deepseek-v4-flash-0731
    PRIMARY_WEIGHT_KEY="deepseek-v4-flash-0731:unsloth-iq2-xxs"
    # ⚠️ NOT optional: both slugs pass -md and will not boot without the drafter.
    # The main GGUF has no embedded nextn head, so DSpark is the only spec path.
    ALWAYS_DRAFT_KEY="deepseek-v4-flash-0731:dspark"
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
    # An unknown friendly model name is fatal ONLY when WEIGHT_KEY is NOT set.
    # WEIGHT_KEY is the authoritative "exact catalog entry" fetch-now flow (used
    # by preflight + the serve-cockpit Download action): the recipe fully
    # specifies <model>:<variant>, and load_weight_recipe() validates that the
    # key's model matches MODEL_NAME — so the hardcoded per-family dispatch isn't
    # needed for an arbitrary catalog entry. Leave PRIMARY_WEIGHT_KEY empty here;
    # the WEIGHT_KEY override below sets it.
    if [[ -z "${WEIGHT_KEY:-}" ]]; then
      echo "ERROR: unsupported model '${MODEL_NAME}'."
      echo "Supported: qwen3.6-27b, qwen3.6-35b-a3b, qwen3.8-27b, gemma-4-31b, gemma-4-26b-a4b, diffusiongemma-26b-a4b, deepseek-v4-flash-0731"
      echo "(To add a new model, extend the model dispatch in scripts/setup.sh and profiles/models/*.yml,"
      echo " or pass WEIGHT_KEY=<model>:<variant> to fetch an exact catalog entry directly.)"
      exit 1
    fi
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

# ---------- Companion artifacts (cockpit Download) ----------
# The serve-cockpit Download action reads the slug's `weights_companions` from the
# registry (a DFlash draft model / mmproj vision projector its compose mounts from
# a separate subdir) and passes them as a space/comma-separated WEIGHT_EXTRA_KEYS
# list of fully-qualified <model>:<variant> keys.  Fetch them ALONGSIDE the core
# so a downloaded slug actually serves — otherwise it reads "present" then fails
# to boot for the missing companion.  Each is a normal catalog entry pulled by the
# EXTRA_WEIGHT_KEYS loop below (after the SKIP_MODEL guard), with its own SHA verify.
if [[ -n "${WEIGHT_EXTRA_KEYS:-}" ]]; then
  read -ra _COMPANION_KEYS <<< "${WEIGHT_EXTRA_KEYS//,/ }"
  for _ck in "${_COMPANION_KEYS[@]}"; do
    [[ -n "${_ck}" ]] && EXTRA_WEIGHT_KEYS+=("${_ck}")
  done
  [[ -n "${_COMPANION_KEYS[*]:-}" ]] && echo "[model]   + companion(s): ${_COMPANION_KEYS[*]}"
fi

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

# Required disk — derived from what this run will actually FETCH.
#
# This used to be a hardcoded 25 GB (28 with a DFlash draft), sized when
# setup.sh served exactly one ~14 GB model. That constant is wrong in both
# directions once the catalog holds large weights, and the under-gate is the
# dangerous one:
#
#   too LOW  — DeepSeek-Flash IQ2 is 85 GiB and Q8 is 151 GiB. A user with
#              30 GB free PASSES the gate, then runs out mid-download. The
#              check gives false assurance exactly where it matters most.
#   too HIGH — with the weights already on disk nothing needs fetching, yet
#              the gate still demanded 25 GB and hard-exited (found by
#              actually running setup.sh, 2026-08-07).
#
# So: sum `size_gb` over the keys this run would download, counting only the
# ones NOT already present, and add headroom for download temp files. An
# explicit PREFLIGHT_DISK_GB still wins — it is the documented escape hatch.
_disk_need_gb() {
  local total=0 key subdir size present
  for key in "$@"; do
    [[ -n "$key" ]] || continue
    # Subshell: don't let one entry's WEIGHT_* leak into the next iteration.
    read -r size subdir < <(
      env_lines="$(python3 "${ROOT_DIR}/scripts/lib/profiles/weights.py" entry "$key" 2>/dev/null)" || exit 0
      eval "$env_lines"
      printf '%s %s\n' "${WEIGHT_SIZE_GB:-0}" "${WEIGHT_VERIFY_GLOB:-}|${WEIGHT_SUBDIR:-}"
    )
    [[ -n "${size:-}" ]] || continue
    local glob="${subdir%%|*}" dir="${subdir##*|}"
    # Already on disk -> costs nothing. Same presence test the verifier uses,
    # so "present" here means the same thing it means at verify time.
    present=0
    if [[ -n "$dir" && -d "${MODEL_DIR}/${dir}" ]]; then
      # shellcheck disable=SC2086  # glob must stay unquoted to expand
      compgen -G "${MODEL_DIR}/${dir}/${glob}" >/dev/null 2>&1 && present=1
    fi
    [[ "$present" == "1" ]] && continue
    total=$(( total + ${size%%.*} ))
  done
  # Headroom for partial-download temp files, plus a floor so a fully-present
  # re-run still refuses to proceed on a volume with nothing left at all.
  if [[ "$total" -gt 0 ]]; then
    echo $(( total + 10 ))
  else
    echo 5
  fi
}

_DISK_KEYS=("${PRIMARY_WEIGHT_KEY:-}" "${ALWAYS_DRAFT_KEY:-}")
[[ "${WITH_DFLASH_DRAFT:-0}" == "1" ]] && _DISK_KEYS+=("${MODEL_NAME}:dflash")
PREFLIGHT_DISK_GB="${PREFLIGHT_DISK_GB:-$(_disk_need_gb "${_DISK_KEYS[@]}")}"

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

# Ensure an HF download CLI ('hf', or legacy 'huggingface-cli') is available.
# If neither is present, offer a CONSENT-GATED, ISOLATED install (uv tool / pipx
# — lands on PATH, no changes to system Python). We deliberately NEVER run
# `pip --break-system-packages` or `sudo apt` on the user's behalf; those stay
# copy-paste the user owns. Returns 0 when a CLI is available (already or after
# an accepted install); prints the working manual options + returns 1 otherwise.
# Honors CLUB3090_ASSUME_YES=1 for non-interactive opt-in. Fixes the PEP 668
# `externally-managed-environment` wall on Ubuntu 24.04 / WSL, where the old
# bare `pip install` hint could not run.
ensure_hf_cli() {
  command -v hf >/dev/null 2>&1 && return 0
  command -v huggingface-cli >/dev/null 2>&1 && return 0

  # First ISOLATED installer available (never touches system Python).
  local installer="" cmd=""
  if command -v uv >/dev/null 2>&1; then
    installer="uv";   cmd="uv tool install --with hf_transfer huggingface-hub"
  elif command -v pipx >/dev/null 2>&1; then
    installer="pipx"; cmd="pipx install 'huggingface-hub[hf_transfer]'"
  fi

  if [[ -n "$installer" ]]; then
    local go=""
    if [[ "${CLUB3090_ASSUME_YES:-0}" == "1" ]]; then
      go=1
      echo "[model] hf CLI missing — CLUB3090_ASSUME_YES=1, installing via ${installer} (isolated)." >&2
    elif [[ -t 0 ]]; then
      echo "[model] The Hugging Face CLI ('hf') is required to download weights, but it's not installed." >&2
      echo "[model] I can install it for you, isolated (on your PATH, no changes to system Python):" >&2
      echo "[model]     ${cmd}" >&2
      local reply=""
      read -rp "[model] Install it now? [Y/n] " reply
      case "${reply}" in ""|[Yy]|[Yy][Ee][Ss]) go=1 ;; esac
    fi
    if [[ -n "$go" ]]; then
      echo "[model] Installing: ${cmd}" >&2
      if eval "$cmd" >&2; then
        # Freshly-installed console scripts land in ~/.local/bin (pipx + uv's
        # default tool bin) — make them visible to THIS run, no shell restart.
        export PATH="${HOME}/.local/bin:${PATH}"
        hash -r 2>/dev/null || true
        if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
          echo "[model] hf CLI installed and on PATH." >&2
          return 0
        fi
        echo "[model] Installed, but 'hf' isn't on PATH in this shell yet — restart your terminal (or run 'pipx ensurepath' / add ~/.local/bin to PATH) and re-run setup.sh." >&2
        return 1
      fi
      echo "[model] Automatic install did not complete — use a manual option below." >&2
    fi
  fi

  # No isolated installer, declined, or install failed → the working manual set.
  {
    echo "ERROR: the Hugging Face CLI ('hf') is required but not installed."
    echo "  Recommended — isolated, puts 'hf' on your PATH (works on Ubuntu 24.04 / WSL):"
    echo "    sudo apt install -y pipx && pipx install 'huggingface-hub[hf_transfer]' && pipx ensurepath"
    echo "    (then restart your shell and re-run this command)"
    echo "  Or, with uv:"
    echo "    uv tool install --with hf_transfer huggingface-hub"
    echo "  Quick override — modifies SYSTEM Python (only on a dedicated box / WSL):"
    echo "    pip install --break-system-packages 'huggingface-hub[hf_transfer]'"
  } >&2
  return 1
}

_hf_download_repo() {
  local repo="$1"
  local subdir="$2"
  local files="${3:-}"
  local revision="${4:-}"
  # Optional commit-SHA / tag pin (#319). Empty -> track HEAD (today's behavior).
  local rev_args=()
  [[ -n "$revision" ]] && rev_args=(--revision "$revision")
  mkdir -p "${MODEL_DIR}/${subdir}"
  # Guarantee a download CLI (consent-gated isolated install if missing).
  ensure_hf_cli || exit 1
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
    # Unreachable: ensure_hf_cli returned 0 so one of the above resolves.
    echo "ERROR: hf CLI unexpectedly unavailable after ensure_hf_cli." >&2
    exit 1
  fi
}

# _hf_remote_meta <repo> <revision> <file> -> "<sha256> <size>" on stdout
#
# Delegates to scripts/lib/profiles/hf_fetch.py::resolve_head (#855) rather than
# re-implementing the lookup here. Two reasons that are not style preferences:
#
#   1. The HEAD must NOT follow redirects. The old `curl -sfI` did, and on any
#      Xet-backed repo the 302 lands on the CAS bridge, whose response carries
#      the CAS *blob* ETag and NO `x-linked-etag` — so the canonical sha256,
#      which lives on the FIRST hop, read as "not published" on every modern
#      repo. The SKIP was the normal outcome, not a rare edge (#857).
#   2. One implementation means the downloader and setup.sh cannot disagree
#      about what "verified" means. They already did, and the weaker surface
#      was the one printing the reassuring line.
#
# Prints "<sha256|-> <size|->" and returns 0 when the hub answered at all;
# returns non-zero only when the lookup itself failed. A literal "-" in the hash
# field means the hub published NO sha256 for this file — the caller MUST treat
# that as UNVERIFIED, never as a pass, however good the size looks.
_hf_remote_meta() {
  local repo="$1" revision="$2" file="$3"
  command -v python3 >/dev/null 2>&1 || return 1
  [[ -f "${ROOT_DIR}/scripts/lib/profiles/hf_fetch.py" ]] || return 1
  python3 - "$repo" "$revision" "$file" <<PY 2>/dev/null
import sys
sys.path.insert(0, "${ROOT_DIR}/scripts/lib/profiles")
import hf_fetch
m = hf_fetch.resolve_head(sys.argv[1], sys.argv[3], revision=sys.argv[2],
                          token=hf_fetch._token())
if m.sha256 is None and m.size is None:
    raise SystemExit(1)          # the hub told us nothing at all
print(f"{m.sha256 or '-'} {m.size if m.size is not None else '-'}")
PY
}

# Verify the downloaded artifacts against the hub's published hashes.
#
# #857 — the word "verified" only ever attaches to a HASH check. Before this,
# a file whose etag lookup came back empty printed `SKIP (no etag)`, was not
# counted as a failure, and was still included in the `N file(s) SHA-verified`
# total. Combined with the redirect-following lookup above, on a Xet-backed repo
# that meant the reassuring line was printed for files nothing had checked —
# the same failure shape as the historical "DONE (hash-verified)" incident.
#
# Now: hash-checkable files are verified; everything else is UNVERIFIED (size-
# only at best) and is EXCLUDED from the verified count, with the split stated.
_verify_downloaded_files() {
  local repo="$1"
  local subdir="$2"
  local verify_glob="$3"
  # Pin the etag lookup to the same revision we downloaded (#319). Empty -> main
  # HEAD; a stale pin would otherwise etag-check against a newer HEAD and FAIL.
  local revision="${4:-main}"
  local fail=0 verified=0 unverified=0 total=0
  local f meta expected exp_size actual local_size

  echo "[verify]  Checking SHA256 of every ${verify_glob} against the hub's published"
  echo "          x-linked-etag (no-redirect resolve HEAD, rev: ${revision}) ..."
  cd "${MODEL_DIR}/${subdir}"
  for f in ${verify_glob}; do
    [[ -f "$f" ]] || continue
    total=$((total + 1))
    expected=""; exp_size=""
    if meta="$(_hf_remote_meta "$repo" "$revision" "$f")"; then
      read -r expected exp_size <<<"$meta"
      [[ "$expected" == "-" ]] && expected=""
      [[ "$exp_size" == "-" ]] && exp_size=""
    fi
    if [[ -z "$expected" ]]; then
      # No published hash -> this file CANNOT be verified. Size is the only
      # cross-check available and it is not a verification: a truncated-then-
      # padded or silently-corrupted file matches on size (the exact incident
      # this rule exists for). Say so, and keep it out of the verified count.
      local_size="$(stat -c '%s' "$f" 2>/dev/null || echo "?")"
      if [[ -n "$exp_size" && "$exp_size" == "$local_size" ]]; then
        printf "  %-50s UNVERIFIED (no published hash; size matches: %s bytes — NOT a verification)\n" \
          "$f" "$local_size"
      elif [[ -n "$exp_size" ]]; then
        printf "  %-50s FAIL  size mismatch  exp=%s  act=%s\n" "$f" "$exp_size" "$local_size"
        fail=$((fail + 1))
        continue
      else
        printf "  %-50s UNVERIFIED (hub published neither hash nor size)\n" "$f"
      fi
      unverified=$((unverified + 1))
      continue
    fi
    actual="$(sha256sum "$f" | awk '{print $1}')"
    if [[ "$expected" == "$actual" ]]; then
      printf "  %-50s OK\n" "$f"
      verified=$((verified + 1))
    else
      printf "  %-50s FAIL  exp=%.12s  act=%.12s\n" "$f" "$expected" "$actual"
      fail=$((fail + 1))
    fi
  done
  cd "${ROOT_DIR}"

  if [[ "$fail" != "0" ]]; then
    echo "[verify]  ${fail} file(s) failed their integrity check." >&2
    echo "          Delete ${MODEL_DIR}/${subdir} and re-run setup.sh." >&2
    exit 1
  fi
  if [[ "$total" == "0" ]]; then
    echo "[verify]  No ${verify_glob} found in ${MODEL_DIR}/${subdir} — download may have failed." >&2
    exit 1
  fi
  # The count in this line is HASH-VERIFIED FILES ONLY. Anything the hub does
  # not publish a hash for is reported separately and loudly — a reader must
  # never be able to read "N file(s) SHA-verified" and conclude N == total when
  # it does not.
  echo "[done]    ${verified}/${total} file(s) SHA-verified in ${subdir}."
  if [[ "$unverified" != "0" ]]; then
    echo "[verify]  ⚠ ${unverified}/${total} file(s) UNVERIFIED — the hub published no sha256 for them,"
    echo "            so nothing checked their contents (a size match is not a verification)."
    echo "            Re-fetch through the resilience ladder for a hash-gated pull:"
    echo "              python3 scripts/lib/profiles/hf_fetch.py ${repo} \\"
    echo "                --local-dir ${MODEL_DIR}/${subdir} --verify-in-place"
  fi
}

download_weight_key() {
  local key="$1"
  load_weight_recipe "$key"
  echo "[model]   Downloading ${WEIGHT_LABEL:-$key} ..."
  _hf_download_repo "$WEIGHT_REPO" "$WEIGHT_SUBDIR" "$WEIGHT_FILES" "${WEIGHT_REVISION:-}"
  _verify_downloaded_files "$WEIGHT_REPO" "$WEIGHT_SUBDIR" "$WEIGHT_VERIFY_GLOB" "${WEIGHT_REVISION:-main}"
}

# #634 — honour the PRIMARY recipe's verify_glob (set by load_weight_recipe →
# line 100, e.g. "*.gguf" for GGUF keys), NOT a re-hardcoded *.safetensors, or
# every GGUF primary fetch (WEIGHTS=gguf/iq4ks) fails verify despite a good
# download.  An explicit VERIFY_GLOB_OVERRIDE still wins.
VERIFY_GLOB="${VERIFY_GLOB_OVERRIDE:-${VERIFY_GLOB}}"
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
#
# ⚠️ Initialise every SAMPLE_* here. This case has no default arm, so a model
# without one left these UNBOUND and `set -u` killed the script on the final
# echo — AFTER a fully successful download and verify, so the user saw exit 1
# on a run that had actually worked. (DeepSeek-Flash hit exactly this; found by
# running setup.sh for real, 2026-08-07.) Empty is the honest value: the echo
# block below degrades to a generic hint rather than printing a wrong one.
SAMPLE_CONTAINER=""
SAMPLE_COMPOSE_FLAGS_DUAL=""
SAMPLE_PORT=""
SAMPLE_MODEL_NAME=""
SAMPLE_LAUNCH_HINT=""
NEXT_STEPS_NOTE=""
SETUP_MODEL_DISPLAY="$(model_label "${MODEL_NAME}")"
case "${MODEL_NAME}" in
  qwen3.6-27b)
    SAMPLE_CONTAINER="vllm-qwen36-27b"
    SAMPLE_COMPOSE_FLAGS_DUAL=" -f dual/autoround-int4/fp8-mtp.yml"
    SAMPLE_PORT="8020"
    SAMPLE_MODEL_NAME="qwen3.6-27b"
    NEXT_STEPS_NOTE="Or dual-card vLLM (Marlin patched files already vendored in-repo):
  cd models/${MODEL_NAME}/vllm/compose && docker compose -f dual/autoround-int4/fp8-mtp.yml up -d"
    ;;
  gemma-4-31b)
    SAMPLE_CONTAINER="vllm-gemma-4-31b-mtp"
    # Gemma 4 ships specific composes — pick MTP as the canonical default.
    # Use scripts/switch.sh which auto-selects the right compose by variant.
    SAMPLE_COMPOSE_FLAGS_DUAL=""
    SAMPLE_PORT="8030"
    SAMPLE_MODEL_NAME="gemma-4-31b"
    NEXT_STEPS_NOTE="Available variants:
  bash scripts/switch.sh vllm/gemma-31b-dual        # bf16 KV, TP=2, port 8032 (dual-card, v0.24.0 overlay-free)
  bash scripts/switch.sh vllm/gemma-31b-dual # dual (single-card 31B has no functional config since the beellama retirement)"
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
  deepseek-v4-flash-0731)
    # llama.cpp only — there is no vLLM compose for this model, so the generic
    # "single-card vLLM" line below MUST be overridden or it prints a path that
    # does not exist. Defaults track the IQ2 reach tier (setup.sh's default
    # weight key); the Q8 sibling serves on 8030.
    SAMPLE_CONTAINER="llama-cpp-deepseek-flash-iq2"
    SAMPLE_COMPOSE_FLAGS_DUAL=""
    SAMPLE_PORT="8031"
    SAMPLE_MODEL_NAME="deepseek-v4-flash"
    SAMPLE_LAUNCH_HINT="  bash scripts/switch.sh --force llamacpp/deepseek-flash-dual-iq2"
    NEXT_STEPS_NOTE="🐣 incubating — launch needs --force (non-functional by default).
  Quality tier instead (needs ~146 GB host RAM vs ~86 GB, serves on 8030):
  bash scripts/switch.sh --force llamacpp/deepseek-flash-dual-q8
  Expect 8-10 min to load (--no-mmap, and the weights are large)."
    ;;
  qwen3.8-27b)
    # The generic "single-card vLLM" line below MUST be overridden (the #914 class).
    # ⚠️ NOTE 2026-08-14: vLLM composes NOW EXIST for this model —
    # vllm/qwen38-27b-dual-max and vllm/qwen38-27b-multi4-max. The override is STILL
    # correct, but for a different reason than originally written: both vLLM slugs are
    # DUAL/MULTI-card and 🐣 Incubating, so a single-card next-step line must not point
    # at them. Only the old "no vLLM compose exists" rationale was stale.
    # ⚠️ 2026-08-17: this model IS now in the dispatch + "Supported:" lists — the
    # old exclusion rationale expired (see the dispatch arm above). The sample lines
    # below still track the SINGLE-card IQ4_NL tier because they advertise a one-card
    # next step; the dispatch default is the dual-card fp8 tier (serves on 8091).
    SAMPLE_CONTAINER="llama-cpp-qwen38-27b-single"
    SAMPLE_COMPOSE_FLAGS_DUAL=""
    SAMPLE_PORT="8086"
    SAMPLE_MODEL_NAME="qwen3.8-27b"
    SAMPLE_LAUNCH_HINT="  bash scripts/switch.sh --force llamacpp/qwen38-27b-single-iq4nl"
    NEXT_STEPS_NOTE="🧪/🐣 — every qwen3.8 slug needs --force; the incubating ones are
  also hidden from 'switch.sh --list' (reveal with --list --all).
  MEASURED (dual fp8, 2026-08-17): vllm/qwen38-27b-dual-max on 2x 3090 — 67.4 narr /
  85.8 code tok/s, verify-full PASS. Still ungated: no verify-stress, no NIAH fill
  depth, no soak, no quality run.
  Dual-card max-context tier instead (needs BOTH 3090s — 31.5 GB of weights
  does not fit one card; serves on 8087):
  WEIGHT_KEY=qwen3.8-27b:unsloth-q8kxl bash scripts/setup.sh qwen3.8-27b
  bash scripts/switch.sh --force llamacpp/qwen38-27b-dual-q8kxl"
    ;;
esac

echo "[setup] ✓ ${SETUP_MODEL_DISPLAY} downloaded."
echo "[setup] Next: bash scripts/launch.sh"
echo ""
# A model arm may set SAMPLE_LAUNCH_HINT to override the launch line. Required
# for models with no vLLM compose — the generic form below would otherwise print
# a `models/<model>/vllm/compose` path that does not exist.
if [[ -n "${SAMPLE_LAUNCH_HINT:-}" ]]; then
  echo "Next — launch it:"
  echo "${SAMPLE_LAUNCH_HINT}"
  echo "  docker logs -f ${SAMPLE_CONTAINER}"
elif [[ "${MODEL_NAME}" == "gemma-4-31b" ]]; then
  echo "Next — single-card vLLM (default):"
  echo "  bash scripts/switch.sh vllm/gemma-31b-dual"
  echo "  docker logs -f ${SAMPLE_CONTAINER}"
else
  echo "Next — single-card vLLM (default):"
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
