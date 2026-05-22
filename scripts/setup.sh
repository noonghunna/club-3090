#!/usr/bin/env bash
#
# Model-aware one-shot setup for club-3090.
#
#   bash scripts/setup.sh                # interactive model picker in a TTY
#   bash scripts/setup.sh <model-name>   # scripted/CI positional form
#
# Currently supported:
#   qwen3.6-27b   →  Lorbus/Qwen3.6-27B-int4-AutoRound + Genesis patches
#   gemma-4-31b   →  Intel/gemma-4-31B-it-int4-AutoRound + Google MTP "assistant"
#                    drafter (no Genesis — not yet integrated upstream as of v7.72.2)
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
#   WITH_DFLASH_DRAFT   Set to 1 to ALSO download z-lab/Qwen3.6-27B-DFlash
#                       (~1.75 GB; required ONLY for dual-dflash.yml /
#                       dual-dflash-noviz.yml composes). Default: 0.
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

usage() {
  echo "Usage: $0 <model-name>"
  echo "       $0              # interactive model picker in a TTY"
  echo ""
  echo "Run with no model name in a normal terminal to open the hardware-aware"
  echo "model picker. Use the positional form in scripts/CI to skip prompts."
  echo ""
  echo "Supported model names:"
  echo "  qwen3.6-27b"
  echo "  gemma-4-31b"
}

model_label() {
  case "$1" in
    qwen3.6-27b) echo "Qwen 3.6 27B" ;;
    gemma-4-31b) echo "Gemma 4 31B" ;;
    *) echo "$1" ;;
  esac
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

# ALWAYS_DRAFT_REPO + ALWAYS_DRAFT_SUBDIR: a drafter that this model REQUIRES
# (vs the optional WITH_DFLASH_DRAFT path). Empty for Qwen3.6 (no required
# drafter); Google MTP drafter for Gemma 4 (canonical recipe per Gemma 4 docs +
# our dual.yml compose).
ALWAYS_DRAFT_REPO=""
ALWAYS_DRAFT_SUBDIR=""

# Per-model dflash drafter info (overrides defaults set later for non-qwen3.6).
DFLASH_REPO_OVERRIDE=""
DFLASH_SUBDIR_OVERRIDE=""

case "${MODEL_NAME}" in
  qwen3.6-27b)
    MODEL_REPO="Lorbus/Qwen3.6-27B-int4-AutoRound"
    MODEL_SUBDIR="qwen3.6-27b-autoround-int4"
    NEEDS_GENESIS=1
    ;;
  gemma-4-31b)
    MODEL_REPO="Intel/gemma-4-31B-it-int4-AutoRound"
    MODEL_SUBDIR="gemma-4-31b-autoround-int4"
    # Gemma 4 isn't Genesis-integrated yet — Sander's roadmap (disc #19) lists
    # Gemma 4 as a follow-up. Until v7.73.x or later integrates it, skip Genesis
    # entirely on this path.
    NEEDS_GENESIS=0
    # Google ships the MTP drafter with the canonical Gemma 4 recipe; our
    # dual.yml compose requires it. Always-fetch (no opt-in flag).
    ALWAYS_DRAFT_REPO="google/gemma-4-31B-it-assistant"
    ALWAYS_DRAFT_SUBDIR="gemma-4-31b-it-assistant"
    # DFlash drafter is z-lab/gemma-4-31B-it-DFlash (different repo than
    # Qwen3.6's z-lab/Qwen3.6-27B-DFlash).
    DFLASH_REPO_OVERRIDE="z-lab/gemma-4-31B-it-dflash"
    DFLASH_SUBDIR_OVERRIDE="gemma-4-31b-it-dflash"
    ;;
  *)
    echo "ERROR: unsupported model '${MODEL_NAME}'."
    echo "Supported: qwen3.6-27b, gemma-4-31b"
    echo "(To add a new model, extend the case dispatch in scripts/setup.sh)"
    exit 1
    ;;
esac

# ---------- Weights format (autoround vLLM default, or gguf for llama.cpp/ik_llama) ----------
# WEIGHTS=gguf swaps the AutoRound vLLM repo above for the llama.cpp GGUF and
# downloads just the file(s) it needs (a GGUF repo holds many quants). The
# GGUF/llama.cpp path doesn't use Genesis. Default is unchanged (autoround).
WEIGHTS="${WEIGHTS:-autoround}"
GGUF_FILES=""   # empty → download whole repo dir (autoround); non-empty → specific files
if [[ "${WEIGHTS}" == "gguf" ]]; then
  case "${MODEL_NAME}" in
    qwen3.6-27b)
      MODEL_REPO="unsloth/Qwen3.6-27B-GGUF"
      MODEL_SUBDIR="qwen3.6-27b-gguf/unsloth-mtp-q4km"
      GGUF_FILES="Qwen3.6-27B-Q4_K_M.gguf mmproj-F16.gguf"  # Q4_K_M MTP + vision mmproj
      NEEDS_GENESIS=0
      ;;
    *)
      echo "ERROR: WEIGHTS=gguf is only wired for qwen3.6-27b right now." >&2
      echo "       Use WEIGHTS=autoround (default), or download the GGUF manually." >&2
      exit 1 ;;
  esac
  echo "[model]   WEIGHTS=gguf → ${MODEL_REPO} (${GGUF_FILES}) → ${MODEL_SUBDIR}"
elif [[ "${WEIGHTS}" != "autoround" ]]; then
  echo "ERROR: WEIGHTS='${WEIGHTS}' not recognized (use 'autoround' or 'gguf')." >&2
  exit 1
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

# ---------- Genesis patches ----------
# We track Sandermage's tree at HEAD and rely on tagged commits / SHA pinning
# in the compose files for reproducibility. The repo layout changed substantially
# between v7.13 (monolithic patch_genesis_unified.py shim) and v7.14 (modular
# vllm/_genesis package + per-patch env opts). Newer composes mount the package;
# the legacy compose still references the v7.13 shim.
# Pin Genesis to the exact commit our published numbers were measured against.
# Currently pointing at v7.69 dev tip (commit 2db18df, 2026-05-02 PM). Bumped
# from v7.66 (fc89395) for the v7.69 patch set, which addresses the 3
# regressions the v7.68 cross-rig retest found ([club-3090#19] and our
# v7.68-cliff2-test branch summary):
#   - F1 (PN30 part3 drift-marker bug) — fixed via specific marker
#     `[Genesis PN30 v7.68 dst-shaped]` so part3 idempotency check no longer
#     collides with part1+2 markers in the same file.
#   - F2 (P103 setattr lost on `exec vllm serve`) — fixed via self-install
#     hook text-patched into chunk.py end-of-file. Survives any startup
#     mechanism (workers, fork, spawn, exec). The "rebound at 0 caller sites"
#     log message in v7.68 was misleading — internal callers DID get the
#     setattr in the entrypoint shell process, but `exec` replaced the image
#     and lost it. v7.69 hook fires every time chunk.py imports.
#   - F3 (PN32 v1 chunked at wrong level) — rewritten as PN32 v2 to patch
#     `_forward_core` directly + thread initial_state via prior chunk's
#     last_recurrent_state. Composes with P103: v2 chunks the OUTER FLA
#     call, P103 chunks INSIDE the FLA inner h tensor.
# Recommended Cliff 2 closure env bundle for single-24GB-GPU:
#   GENESIS_ENABLE_P103=1                          (close inner FLA h tensor)
#   GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1      (close outer FLA call buffer)
#   GENESIS_PN32_GDN_CHUNK_SIZE=8192               (default)
#   GENESIS_PN32_GDN_CHUNK_THRESHOLD=16384         (default)
#   GENESIS_FLA_FWD_H_MAX_T=16384                  (P103 default)
# v7.69 also retains v7.68's accept-and-fold of our 3 cross-rig sidecars:
# PN25 v7.68 (TP=1 worker-spawn registration), PN30 v7.68 (DS conv-state
# layout dst-shaped temp), PN34 (vllm#39226 runtime workspace_lock).
# Pinned to dev SHA 2db18df because v7.69 is feature-complete on dev pending
# our retest validation; if clean, Sander will tag stable.
# Bumping GENESIS_PIN requires re-running verify-stress.sh against your composes
# to confirm the new commit works on your config.
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

mkdir -p "${MODEL_DIR}/${MODEL_SUBDIR}"

# Prefer `hf` CLI if available (faster with hf_transfer); fall back to curl.
download_via_hf() {
  echo "[model]   Using 'hf download' (hf_transfer if available) ..."
  # ${GGUF_FILES} is intentionally unquoted: empty (autoround) → whole-repo
  # download as before; set (gguf) → just those file(s) word-split as args.
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
    hf download "${MODEL_REPO}" ${GGUF_FILES} --local-dir "${MODEL_DIR}/${MODEL_SUBDIR}"
}

if command -v hf >/dev/null 2>&1; then
  download_via_hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  echo "[model]   Using 'huggingface-cli download' ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
    huggingface-cli download "${MODEL_REPO}" ${GGUF_FILES} --local-dir "${MODEL_DIR}/${MODEL_SUBDIR}"
else
  echo "ERROR: neither 'hf' nor 'huggingface-cli' found. Install with:" >&2
  echo "  pip install 'huggingface-hub[hf_transfer]'" >&2
  echo "or:" >&2
  echo "  uv tool install --with hf_transfer huggingface-hub" >&2
  exit 1
fi

# ---------- SHA verification ----------
VERIFY_GLOB="*.safetensors"; [[ "${WEIGHTS}" == "gguf" ]] && VERIFY_GLOB="*.gguf"
echo "[verify]  Checking SHA256 of every ${VERIFY_GLOB} against HF x-linked-etag ..."
cd "${MODEL_DIR}/${MODEL_SUBDIR}"

fail=0
count=0
for f in ${VERIFY_GLOB}; do
  [[ -f "$f" ]] || continue
  count=$((count + 1))
  expected="$(curl -sfI "https://huggingface.co/${MODEL_REPO}/resolve/main/$f" \
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
  echo "[verify]  ${fail} shard(s) failed SHA check." >&2
  echo "          Delete ${MODEL_DIR}/${MODEL_SUBDIR} and re-run setup.sh." >&2
  exit 1
fi

if [[ "$count" == "0" ]]; then
  echo "[verify]  No ${VERIFY_GLOB} found in ${MODEL_DIR}/${MODEL_SUBDIR} — download may have failed." >&2
  exit 1
fi

echo ""
echo "[done]    ${count} file(s) SHA-verified."
[[ -d "${GENESIS_DIR}/.git" ]] && echo "          Genesis pinned at ${GENESIS_PIN} ($(cd "${GENESIS_DIR}" && git rev-parse --short HEAD))."
echo ""

# ---------- Optional DFlash draft model ----------
# Required ONLY for `dual/dflash.yml` / `dual-dflash-noviz.yml`.
# vLLM `method:"dflash"` spec-decode loads this as the draft. The compose
# expects it at <MODEL_DIR>/qwen3.6-27b-dflash/ (~1.75 GB / card after load).
#
# Caveat: as of 2026-04-26, z-lab/Qwen3.6-27B-DFlash is still under training.
# Published bench in docs/DUAL_CARD.md (82 narr / 125 code TPS on dual-3090)
# was measured against the 2026-04-26 snapshot at peak code-prompt conditions.
# Real agent traffic (mixed code + narrative + tool schemas) will see lower
# AL until z-lab tags training-complete. See docs/UPSTREAM.md for the watch
# entry and re-test trigger.
# Per-model DFlash repo + subdir defaults. Override (Gemma 4) set in the case
# dispatch above.
DFLASH_REPO="${DFLASH_REPO_OVERRIDE:-z-lab/Qwen3.6-27B-DFlash}"
DFLASH_SUBDIR="${DFLASH_SUBDIR_OVERRIDE:-qwen3.6-27b-dflash}"

# Always-required drafter (Gemma 4 MTP "assistant"). Empty for models without
# a canonical drafter shipped alongside the target weights (Qwen3.6).
if [[ -n "${ALWAYS_DRAFT_REPO}" ]] && [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  if [[ -d "${MODEL_DIR}/${ALWAYS_DRAFT_SUBDIR}" ]] \
     && find "${MODEL_DIR}/${ALWAYS_DRAFT_SUBDIR}" -name "*.safetensors" -print -quit | grep -q .; then
    echo "[draft]   ${MODEL_DIR}/${ALWAYS_DRAFT_SUBDIR} already has weights — skipping."
  else
    echo "[draft]   Downloading required drafter ${ALWAYS_DRAFT_REPO} ..."
    mkdir -p "${MODEL_DIR}/${ALWAYS_DRAFT_SUBDIR}"
    if command -v hf >/dev/null 2>&1; then
      HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
        hf download "${ALWAYS_DRAFT_REPO}" --local-dir "${MODEL_DIR}/${ALWAYS_DRAFT_SUBDIR}"
    elif command -v huggingface-cli >/dev/null 2>&1; then
      HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
        huggingface-cli download "${ALWAYS_DRAFT_REPO}" --local-dir "${MODEL_DIR}/${ALWAYS_DRAFT_SUBDIR}"
    else
      echo "[draft]   ERROR: neither 'hf' nor 'huggingface-cli' available — cannot download drafter." >&2
      exit 1
    fi
    echo "[draft]   Downloaded ${ALWAYS_DRAFT_REPO} to ${MODEL_DIR}/${ALWAYS_DRAFT_SUBDIR}"
  fi
  echo ""
fi

if [[ "${WITH_DFLASH_DRAFT:-0}" == "1" ]] && [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  echo "[dflash]  WITH_DFLASH_DRAFT=1 — downloading ${DFLASH_REPO} ..."
  mkdir -p "${MODEL_DIR}/${DFLASH_SUBDIR}"
  if command -v hf >/dev/null 2>&1; then
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
      hf download "${DFLASH_REPO}" --local-dir "${MODEL_DIR}/${DFLASH_SUBDIR}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
      huggingface-cli download "${DFLASH_REPO}" --local-dir "${MODEL_DIR}/${DFLASH_SUBDIR}"
  else
    echo "[dflash]  ERROR: neither 'hf' nor 'huggingface-cli' available — cannot download DFlash draft." >&2
    exit 1
  fi
  echo "[dflash]  Downloaded ${DFLASH_REPO} to ${MODEL_DIR}/${DFLASH_SUBDIR}"
elif [[ -d "${MODEL_DIR}/${DFLASH_SUBDIR}" ]]; then
  echo "[dflash]  ${MODEL_DIR}/${DFLASH_SUBDIR} already exists — using existing draft."
else
  echo "[dflash]  Skipping DFlash draft model. Set WITH_DFLASH_DRAFT=1 to fetch"
  echo "          ${DFLASH_REPO} (~1.75 GB; required only for dual-dflash composes)."
fi
echo ""

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
    SAMPLE_COMPOSE_FLAGS_DUAL=" -f dual/docker-compose.yml"
    SAMPLE_PORT="8020"
    SAMPLE_MODEL_NAME="qwen3.6-27b-autoround"
    NEXT_STEPS_NOTE="Or dual-card vLLM (Marlin patched files already vendored in-repo):
  cd models/${MODEL_NAME}/vllm/compose && docker compose -f dual/docker-compose.yml up -d"
    ;;
  gemma-4-31b)
    SAMPLE_CONTAINER="vllm-gemma-4-31b-mtp"
    # Gemma 4 ships specific composes — pick MTP as the canonical default.
    # Use scripts/switch.sh which auto-selects the right compose by variant.
    SAMPLE_COMPOSE_FLAGS_DUAL=""
    SAMPLE_PORT="8030"
    SAMPLE_MODEL_NAME="gemma-4-31b-autoround"
    NEXT_STEPS_NOTE="Available variants:
  bash scripts/switch.sh vllm/gemma-mtp        # MTP drafter, TP=2, port 8030 (recommended)
  bash scripts/switch.sh vllm/gemma-mtp-tp1    # MTP drafter, TP=1 (single-card; upstream-blocked on Ampere fp8)
  bash scripts/switch.sh vllm/gemma-dflash     # DFlash drafter, TP=2, port 8032 (requires WITH_DFLASH_DRAFT=1)"
    ;;
esac

echo "[setup] ✓ ${SETUP_MODEL_DISPLAY} downloaded."
echo "[setup] Next: bash scripts/launch.sh"
echo ""
echo "Next — single-card vLLM (default):"
if [[ "${MODEL_NAME}" == "gemma-4-31b" ]]; then
  echo "  bash scripts/switch.sh vllm/gemma-mtp"
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
