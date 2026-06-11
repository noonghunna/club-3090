#!/usr/bin/env bash
# Bootstraps ComfyUI + custom nodes into the mounted /workspace/ComfyUI volume on first run,
# pulls latest on subsequent runs, then launches the ComfyUI server on :8188.
set -euo pipefail

# Optional GPU pin: confine ComfyUI to specific card(s) so a chat LLM can use the rest
# (the `gpu-mode image-studio` 2-card split passes "0"). Only export when non-empty —
# CUDA_VISIBLE_DEVICES="" would hide ALL GPUs.
if [ -n "${COMFYUI_CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$COMFYUI_CUDA_VISIBLE_DEVICES"
    echo "[bootstrap] CUDA_VISIBLE_DEVICES pinned to $CUDA_VISIBLE_DEVICES"
fi

COMFY_ROOT="/workspace/ComfyUI"
PIP_OPTS="--no-input --retries 10 --timeout 60"

clone_or_update() {
    local url="$1"
    local dest="$2"
    if [ ! -d "$dest/.git" ]; then
        echo "[bootstrap] git clone $url -> $dest"
        git clone --depth 1 "$url" "$dest"
    else
        echo "[bootstrap] git pull $dest"
        git -C "$dest" fetch --depth 1 origin || true
        git -C "$dest" reset --hard "@{u}" 2>/dev/null || git -C "$dest" pull --ff-only || true
    fi
}

# 0. Make models/input/output/user dirs writable by the host user (uid 1000) too,
#    so host-side hf download / file moves work after container has touched them as root.
chmod -R a+rwX /workspace/ComfyUI/models /workspace/ComfyUI/input /workspace/ComfyUI/output /workspace/ComfyUI/user 2>/dev/null || true

# 0b. Claim the pip cache for the container user. The cache is a bind mount (so it
#     persists across recreates), but it's host-owned (uid 1000) while this container
#     runs as root — and pip DISABLES a cache dir it doesn't own, even a world-writable
#     one (the "sudo without -H" safety check). A disabled cache means every recreate
#     re-downloads all node deps (~10 min: matplotlib/opencv/insightface/nunchaku-149MB).
#     chown it to the running uid so pip actually persists wheels; 0777 mode is preserved
#     so the host user can still read/clear it.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR"
chown -R "$(id -u):$(id -g)" "$PIP_CACHE_DIR" 2>/dev/null || true

# 1. ComfyUI core — PINNED to a known-good commit (reproducible; this commit has
#    native Ideogram-4 support and is validated on this stack). Floating on HEAD would
#    let an upstream change silently break users. Set COMFYUI_REF=HEAD (or 'latest') to
#    float on origin/master instead (needed for bleeding-edge models — re-pin after
#    validating). Bump this default in the same change that re-validates image gen.
COMFYUI_REF="${COMFYUI_REF:-cb9f6394160808f7d25163f6cc2ea300c6841ef9}"
checkout_comfy_ref() {  # $1 = repo dir
    local r="$1"
    if [ "$COMFYUI_REF" = "HEAD" ] || [ "$COMFYUI_REF" = "latest" ]; then
        echo "[bootstrap] ComfyUI: floating on origin/master (COMFYUI_REF=$COMFYUI_REF)"
        git -C "$r" fetch --depth 1 origin 2>/dev/null || true
        git -C "$r" reset --hard origin/master 2>/dev/null || git -C "$r" pull --ff-only || true
    else
        echo "[bootstrap] ComfyUI: pinned to ${COMFYUI_REF:0:12}"
        git -C "$r" fetch --depth 1 origin "$COMFYUI_REF" 2>/dev/null || git -C "$r" fetch origin 2>/dev/null || true
        git -C "$r" checkout -q -f "$COMFYUI_REF" 2>/dev/null \
          || git -C "$r" reset --hard "$COMFYUI_REF" 2>/dev/null \
          || echo "[bootstrap] WARN: could not checkout $COMFYUI_REF (offline?) — using current tree"
    fi
}
if [ ! -f "$COMFY_ROOT/main.py" ]; then
    echo "[bootstrap] Cloning ComfyUI into temp..."
    rm -rf /tmp/_comfy_clone
    # Full clone (not --depth 1) so an arbitrary pinned commit is checkout-able.
    git clone https://github.com/comfyanonymous/ComfyUI.git /tmp/_comfy_clone
    if [ "$COMFYUI_REF" != "HEAD" ] && [ "$COMFYUI_REF" != "latest" ]; then
        git -C /tmp/_comfy_clone checkout -q -f "$COMFYUI_REF" 2>/dev/null \
          || echo "[bootstrap] WARN: pinned ref $COMFYUI_REF not found in fresh clone — using HEAD"
    fi
    mkdir -p "$COMFY_ROOT"
    # Copy everything except dirs that exist as bind mounts (models, input, output, user)
    cp -an /tmp/_comfy_clone/. "$COMFY_ROOT/"
    rm -rf /tmp/_comfy_clone
elif [ -d "$COMFY_ROOT/.git" ]; then
    checkout_comfy_ref "$COMFY_ROOT"
else
    echo "[bootstrap] ComfyUI present without .git; skipping pin/update."
fi

cd "$COMFY_ROOT"

# 2. Core requirements (don't fail container if a transient pull fails — already-installed deps stay)
echo "[bootstrap] Installing ComfyUI core requirements..."
pip install $PIP_OPTS -r requirements.txt || echo "[bootstrap] WARN: core pip install had errors; continuing"

# 3. Custom nodes
NODES="$COMFY_ROOT/custom_nodes"
mkdir -p "$NODES"

clone_or_update https://github.com/Comfy-Org/ComfyUI-Manager.git "$NODES/ComfyUI-Manager"
clone_or_update https://github.com/city96/ComfyUI-GGUF.git           "$NODES/ComfyUI-GGUF"
clone_or_update https://github.com/mit-han-lab/ComfyUI-nunchaku.git  "$NODES/ComfyUI-nunchaku"
clone_or_update https://github.com/kijai/ComfyUI-WanVideoWrapper.git "$NODES/ComfyUI-WanVideoWrapper"
clone_or_update https://github.com/kijai/ComfyUI-HunyuanVideoWrapper.git "$NODES/ComfyUI-HunyuanVideoWrapper"
clone_or_update https://github.com/kijai/ComfyUI-KJNodes.git         "$NODES/ComfyUI-KJNodes"
clone_or_update https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$NODES/ComfyUI-VideoHelperSuite"
clone_or_update https://github.com/pollockjj/ComfyUI-MultiGPU.git "$NODES/ComfyUI-MultiGPU"
# HiDream-O1 image lane: HiDream-O1 has NO native ComfyUI support, so its nodes come from this
# third-party repo (HiDream O1 Model Loader / Conditioning / Sampler). Reviewed source: no
# subprocess/eval/network beyond an opt-in HF model downloader (unused — weights are pre-fetched
# by download_hidream_o1.sh). Its requirements (torch/transformers/diffusers/...) are floor-pins
# already satisfied by the base image, so the loop below is a no-op and does NOT touch torch 2.7.0.
clone_or_update https://github.com/Saganaki22/HiDream_O1-ComfyUI.git "$NODES/HiDream_O1-ComfyUI"
# transformers 5.x compat: the HiDream node calls create_causal_mask(input_embeds=, cache_position=),
# but transformers 5.x renamed input_embeds->inputs_embeds and dropped cache_position from that
# signature. Idempotently fix the one call site so the model runs on our image's transformers.
# (No-op once upstream fixes it or if the file changes.) Drift_guard: matched-block replace.
HIDREAM_CCM="$NODES/HiDream_O1-ComfyUI/hidream_o1/models/qwen3_vl_transformers.py"
if [ -f "$HIDREAM_CCM" ]; then
    python3 - "$HIDREAM_CCM" <<'PYEOF' || echo "[bootstrap] WARN: HiDream create_causal_mask patch failed"
import sys
p = sys.argv[1]; s = open(p).read()
old = ("            input_embeds=inputs_embeds,\n"
       "            attention_mask=attention_mask,\n"
       "            cache_position=cache_position,\n")
new = ("            inputs_embeds=inputs_embeds,\n"
       "            attention_mask=attention_mask,\n")
if old in s:
    open(p, "w").write(s.replace(old, new))
    print("[bootstrap] patched HiDream create_causal_mask for transformers 5.x")
else:
    print("[bootstrap] HiDream create_causal_mask patch: no-op (already patched / upstream-fixed)")
PYEOF
fi

for d in "$NODES"/*/; do
    if [ -f "$d/requirements.txt" ]; then
        echo "[bootstrap] Installing requirements for $(basename "$d")..."
        pip install $PIP_OPTS -r "$d/requirements.txt" || echo "[bootstrap] WARN: requirements failed for $(basename "$d")"
    fi
done

# 4. Nunchaku — pin v1.2.0 wheel for torch 2.7 + cu12.8 (sm_86 supported).
#    NOTE: PyPI 'nunchaku' is a different stats package; do not pip install nunchaku without the URL.
NUNCHAKU_WHL="https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.0/nunchaku-1.2.0+torch2.7-cp311-cp311-linux_x86_64.whl"
if ! python3 -c "import nunchaku; assert nunchaku.__file__.find('site-packages/nunchaku/') >= 0" 2>/dev/null; then
    echo "[bootstrap] Installing nunchaku 1.2.0 from prebuilt wheel..."
    pip uninstall -y nunchaku 2>/dev/null || true
    pip install $PIP_OPTS "$NUNCHAKU_WHL" || echo "[bootstrap] WARN: nunchaku wheel install failed"
else
    echo "[bootstrap] nunchaku already installed."
fi

# 5. Launch ComfyUI
echo "[bootstrap] Starting ComfyUI on 0.0.0.0:8188"
exec python main.py --listen 0.0.0.0 --port 8188 "$@"
