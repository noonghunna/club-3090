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
#   PREFLIGHT_DISK_GB   Required free space at MODEL_DIR (default: 25)
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
# temp files + safetensors + tokenizer/config.
PREFLIGHT_DISK_GB="${PREFLIGHT_DISK_GB:-25}"

echo "[preflight] checking environment..."
preflight_docker || exit 1
preflight_gpu 1  || exit 1
preflight_disk "${MODEL_DIR}" "${PREFLIGHT_DISK_GB}" || exit 1
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
# Currently pointing at v7.65 dev tip (commit d89a089, 2026-05-01 PM). Bumped
# from v7.64 (64dd18b) for the v7.65 patch set:
#   - P38B / P15B — close the Cliff 1 mech B cascade (issues #14 + #15) via
#     compile-safe in-source hook + FA varlen workspace clamp.
#   - PN25 — Inductor-safe silu_and_mul opaque op (replaces our local
#     patch_pn12_compile_safe_custom_op.py — now removed).
#   - PN26b — Genesis-original sparse-V Triton kernel for SM86 (Ampere
#     consumer). First sparse-V kernel in any public tree for SM86. Default
#     ON in v0.20+ composes (BLOCK_KV=8 num_warps=4 threshold=0.01 per
#     Sandermage's 27B-specific tuning).
#   - PN28 — merge_attn_states NaN guard backport (vllm#39148).
#   - Cliff 8 hardening (partial_apply_warnings counter in boot summary).
# Pinned to dev SHA d89a089 because v7.65 is feature-complete on dev but not
# yet tagged; SHA pin is immutable.
# Bumping GENESIS_PIN requires re-running verify-full.sh against your composes
# to confirm the new commit works on your config.
GENESIS_PIN="${GENESIS_PIN:-d89a089}"

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

  # PN25 worker-spawn registration fix (genesis-vllm-patches#16) — local
  # backport while upstream PR cycle plays out. Idempotent: re-running is a
  # no-op once the markers are present. Safe to remove after Sandermage ships
  # the import-time custom-op registration fix.
  if [[ -f "${ROOT_DIR}/models/qwen3.6-27b/vllm/patches/patch_pn25_genesis_register_fix.py" ]]; then
    (cd "${ROOT_DIR}" && python3 models/qwen3.6-27b/vllm/patches/patch_pn25_genesis_register_fix.py) || {
      echo "[genesis] WARN: PN25 register fix did not apply cleanly. PN25 may not work in workers." >&2
    }
  fi
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
echo "Next — single-card vLLM (default):"
echo "  cd models/${MODEL_NAME}/vllm/compose && docker compose up -d"
echo "  docker logs -f vllm-qwen36-27b"
echo ""
echo "For dual-card composes, you ALSO need the Marlin pad fork mounted at"
echo "/opt/ai/vllm-src/ (vLLM PR #40361 — open upstream, drops out when it lands):"
echo "  sudo mkdir -p /opt/ai && sudo chown \$USER /opt/ai"
echo "  git clone -b marlin-pad-sub-tile-n https://github.com/noonghunna/vllm.git /opt/ai/vllm-src"
echo ""
echo "Then:"
echo "  cd models/${MODEL_NAME}/vllm/compose && docker compose -f docker-compose.dual.yml up -d"
echo ""
echo "Sanity test (after 'Application startup complete'):"
echo "  curl -sf http://localhost:8020/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen3.6-27b-autoround\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":200}'"
