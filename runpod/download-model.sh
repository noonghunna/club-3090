#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="Lorbus/Qwen3.6-27B-int4-AutoRound"
MODEL_DIR="${MODEL_DIR:-/workspace/models/qwen3.6-27b-autoround-int4}"
HF_TOKEN="${HF_TOKEN:-}"

if [ -f "${MODEL_DIR}/config.json" ]; then
    echo "[download-model] Model already at ${MODEL_DIR} — skipping download."
    exit 0
fi

echo "[download-model] Downloading ${MODEL_REPO} to ${MODEL_DIR} ..."
mkdir -p "${MODEL_DIR}"

download() {
    if command -v hf >/dev/null 2>&1; then
        HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
            hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
            huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
    else
        echo "[download-model] Installing huggingface-hub..."
        python3 -m pip install -q "huggingface-hub[hf_transfer]"
        HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
            huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
    fi
}

download

echo "[download-model] Verifying .safetensors files..."
cd "${MODEL_DIR}"
fail=0
count=0
for f in *.safetensors; do
    [[ -f "$f" ]] || continue
    count=$((count + 1))
    expected=$(curl -sfI "https://huggingface.co/${MODEL_REPO}/resolve/main/$f" 2>/dev/null \
        | grep -i '^x-linked-etag:' | tr -d '"\r' | awk '{print $NF}' || true)
    if [ -z "$expected" ]; then
        printf "  %-50s SKIP (no etag)\n" "$f"
    else
        actual=$(sha256sum "$f" | awk '{print $1}')
        if [ "$expected" = "$actual" ]; then
            printf "  %-50s OK\n" "$f"
        else
            printf "  %-50s FAIL\n" "$f"
            fail=$((fail + 1))
        fi
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "[download-model] ERROR: ${fail} shard(s) failed SHA verification. Delete ${MODEL_DIR} and retry."
    exit 1
fi
echo "[download-model] ${count} shards verified OK."
echo "[download-model] Done."
