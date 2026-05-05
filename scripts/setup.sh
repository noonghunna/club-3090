#!/usr/bin/env bash
#
# Model-aware one-shot setup for club-3090.
#
#   bash scripts/setup.sh <model-name>
#
# Currently supported:
#   qwen3.6-27b   →  Lorbus/Qwen3.6-27B-int4-AutoRound + Genesis patches
#
# What it does (per supported model):
#   - clones Sandermage/genesis-vllm-patches into models/<model>/vllm/patches/genesis
#     (vLLM-only; skip with SKIP_GENESIS=1 if you only need llama.cpp / SGLang)
#   - downloads model weights into $MODEL_DIR with SHA256 verification
#     against HF x-linked-etag
#
# Env vars (optional):
#   MODEL_DIR           Where to place model weights. Default: <repo>/models-cache
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

# ---------- Model dispatch ----------
MODEL_NAME="${1:-}"
if [[ -z "${MODEL_NAME}" ]]; then
  echo "Usage: $0 <model-name>"
  echo ""
  echo "Supported model names:"
  echo "  qwen3.6-27b"
  exit 1
fi

case "${MODEL_NAME}" in
  qwen3.6-27b)
    MODEL_REPO="Lorbus/Qwen3.6-27B-int4-AutoRound"
    MODEL_SUBDIR="qwen3.6-27b-autoround-int4"
    NEEDS_GENESIS=1
    ;;
  *)
    echo "ERROR: unsupported model '${MODEL_NAME}'."
    echo "Supported: qwen3.6-27b"
    echo "(To add a new model, extend the case dispatch in scripts/setup.sh)"
    exit 1
    ;;
esac

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models-cache}"
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

if [[ "${SKIP_GENESIS:-0}" != "1" ]]; then
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
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
    hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}/${MODEL_SUBDIR}"
}

if command -v hf >/dev/null 2>&1; then
  download_via_hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  echo "[model]   Using 'huggingface-cli download' ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
    huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_DIR}/${MODEL_SUBDIR}"
else
  echo "ERROR: neither 'hf' nor 'huggingface-cli' found. Install with:" >&2
  echo "  pip install 'huggingface-hub[hf_transfer]'" >&2
  echo "or:" >&2
  echo "  uv tool install --with hf_transfer huggingface-hub" >&2
  exit 1
fi

# ---------- SHA verification ----------
echo "[verify]  Checking SHA256 of every *.safetensors against HF x-linked-etag ..."
cd "${MODEL_DIR}/${MODEL_SUBDIR}"

fail=0
count=0
for f in *.safetensors; do
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
  echo "[verify]  No .safetensors found in ${MODEL_DIR}/${MODEL_SUBDIR} — download may have failed." >&2
  exit 1
fi

echo ""
echo "[done]    ${count} shards SHA-verified."
[[ -d "${GENESIS_DIR}/.git" ]] && echo "          Genesis pinned at ${GENESIS_PIN} ($(cd "${GENESIS_DIR}" && git rev-parse --short HEAD))."
echo ""

# ---------- Optional DFlash draft model ----------
# Required ONLY for `docker-compose.dual-dflash.yml` / `dual-dflash-noviz.yml`.
# vLLM `method:"dflash"` spec-decode loads this as the draft. The compose
# expects it at <MODEL_DIR>/qwen3.6-27b-dflash/ (~1.75 GB / card after load).
#
# Caveat: as of 2026-04-26, z-lab/Qwen3.6-27B-DFlash is still under training.
# Published bench in docs/DUAL_CARD.md (82 narr / 125 code TPS on dual-3090)
# was measured against the 2026-04-26 snapshot at peak code-prompt conditions.
# Real agent traffic (mixed code + narrative + tool schemas) will see lower
# AL until z-lab tags training-complete. See docs/UPSTREAM.md for the watch
# entry and re-test trigger.
DFLASH_REPO="z-lab/Qwen3.6-27B-DFlash"
DFLASH_SUBDIR="qwen3.6-27b-dflash"
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
# clone needed. (Previous design required cloning a fork to /opt/ai/vllm-src/;
# refactored 2026-05-03 to vendor the two files in-repo, fixing #37.)

echo "Next — single-card vLLM (default):"
echo "  cd models/${MODEL_NAME}/vllm/compose && docker compose up -d"
echo "  docker logs -f vllm-qwen36-27b"
echo ""
echo "Or dual-card vLLM (Marlin patched files already vendored in-repo):"
echo "  cd models/${MODEL_NAME}/vllm/compose && docker compose -f docker-compose.dual.yml up -d"
echo ""
echo "Sanity test (after 'Application startup complete'):"
echo "  curl -sf http://localhost:8020/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen3.6-27b-autoround\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":200}'"
