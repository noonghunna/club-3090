#!/bin/bash
# GPU/RAM Mode Switcher for AI Inference Stack
# Manages Docker containers to avoid GPU/RAM contention on dual-3090 setup
# Location: club-3090/scripts/gpu-mode.sh (symlinked to /usr/local/bin/gpu-mode)

set -e

# club-3090 is the canonical repo (qwen36-dual-3090 + /opt/ai/compose/<svc>
# both deprecated 2026-05-10 — supporting services moved into services/).
CLUB3090_DIR="/opt/ai/github/club-3090"
COMPOSE_BASE="$CLUB3090_DIR/services"
# Post-PR-A (<quant>/ layer): dual composes live under <topology>/<quant>/.
# Point each var at the quant dir so `compose_at` cd's into it — mount-safe,
# the same invocation switch.sh uses (project dir = compose-file dir).
DUAL_27B_DIR="$CLUB3090_DIR/models/qwen3.6-27b/vllm/compose/dual/autoround-int4"
GEMMA_DUAL_DIR="$CLUB3090_DIR/models/gemma-4-31b/vllm/compose/dual/autoround-int4"
GEMMA_DUAL_AWQ_DIR="$CLUB3090_DIR/models/gemma-4-31b/vllm/compose/dual/awq"
# Qwen3.6-40B-Deckard: uncensored dense 40B, Q6_K GGUF + embedded MTP head,
# layer-split across both cards (llama.cpp). Dual-only — see `gpu-mode deckard`.
DECKARD_DIR="$CLUB3090_DIR/models/qwen3.6-40b-deckard/llama-cpp/compose/dual/piehsoft-q6k"
# DiffusionGemma 26B-A4B — vLLM's first dLLM (TP=2, official :gemma image + 3 fix-mounts).
# Dual-only (both cards). See `gpu-mode dgemma`. 🧪 experimental.
DGEMMA_DIR="$CLUB3090_DIR/models/diffusiongemma-26b-a4b/vllm/compose/dual/fp8"
# Image-studio chat brain: gemma-4-12b single-card (llama.cpp), pinned to the spare GPU
# so it coexists with ComfyUI image gen (different card). See `gpu-mode image-studio`.
GEMMA_12B_DIR="$CLUB3090_DIR/models/gemma-4-12b/llama-cpp/compose/single/unsloth-q8kxl"

# Estate planner state file (v0.7.0+). Instances booted via launch.sh --estate
# or --estate-file are tracked here and persist via Docker `restart:
# unless-stopped`, so they DO survive a plain mode-switch unless explicitly
# torn down via launch.sh --down-estate. mode_off uses this path to clean
# them up alongside the older vLLM/Gemma/ComfyUI services.
ESTATE_YAML="${HOME}/.club3090/estate.yml"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Standard supporting services living under $CLUB3090_DIR/services.
# Ollama dropped 2026-05-10 — we route Qwen/Gemma through LiteLLM directly
# instead. Compose dir kept at services/ollama/ for manual spin-up if needed.
SERVICES=(openwebui litellm qdrant searxng)

# Run a docker compose command in any directory, with optional -f override.
# Args: <dir> <action> [compose_file]
#
# Always passes --env-file $CLUB3090_DIR/.env when that file exists, so
# ${MODEL_DIR} (and other repo-level vars) resolve correctly regardless of
# which compose dir we're cd'd into. Without this, docker compose only
# auto-loads .env from the compose file's own directory and falls back to
# the relative-path default `../../../../../models-cache` (mostly empty).
#
# stderr is preserved (no 2>/dev/null) so real errors surface.
compose_at() {
    local dir=$1
    local action=$2
    local file=${3:-docker-compose.yml}
    if [ -f "$dir/$file" ]; then
        local env_args=()
        if [ -f "$CLUB3090_DIR/.env" ]; then
            env_args=(--env-file "$CLUB3090_DIR/.env")
        fi
        (cd "$dir" && sudo docker compose "${env_args[@]}" -f "$file" $action)
    fi
}

# Like compose_at, but injects per-invocation env assignments that survive `sudo`.
# `sudo docker compose` sanitizes the caller's environment, so vars set in the shell
# don't reach compose interpolation — pass them as leading `VAR=val` args to the
# command instead (`sudo VAR=val docker compose ...`).
# Args: <dir> <action> <file> [VAR=val ...]
compose_at_env() {
    local dir=$1 action=$2 file=$3; shift 3
    local envs=("$@")
    if [ -f "$dir/$file" ]; then
        local env_args=()
        if [ -f "$CLUB3090_DIR/.env" ]; then
            env_args=(--env-file "$CLUB3090_DIR/.env")
        fi
        (cd "$dir" && sudo "${envs[@]}" docker compose "${env_args[@]}" -f "$file" $action)
    fi
}

# Standard service helpers (look in $COMPOSE_BASE/<service>)
compose_cmd() {
    compose_at "$COMPOSE_BASE/$1" "$2"
}

start_service() {
    printf "  ${GREEN}▲${NC} Starting %-12s" "$1..."
    compose_cmd "$1" "up -d" && echo "done" || echo "failed"
}

stop_service() {
    printf "  ${RED}▼${NC} Stopping %-12s" "$1..."
    compose_cmd "$1" "down" && echo "done" || echo "skipped"
}

# Project-specific helpers
start_27b_dual_mtp() {
    printf "  ${GREEN}▲${NC} Starting 27b-dual-mtp..."
    compose_at "$DUAL_27B_DIR" "up -d" fp8-mtp.yml && echo "done" || echo "failed"
}
stop_27b_dual_mtp() {
    printf "  ${RED}▼${NC} Stopping 27b-dual-mtp..."
    compose_at "$DUAL_27B_DIR" "down" fp8-mtp.yml && echo "done" || echo "skipped"
}

start_27b_dual_dflash() {
    printf "  ${GREEN}▲${NC} Starting 27b-dual-dflash..."
    compose_at "$DUAL_27B_DIR" "up -d" dflash.yml && echo "done" || echo "failed"
}
stop_27b_dual_dflash() {
    printf "  ${RED}▼${NC} Stopping 27b-dual-dflash..."
    compose_at "$DUAL_27B_DIR" "down" dflash.yml && echo "done" || echo "skipped"
}

start_27b_dual_dflash_noviz() {
    printf "  ${GREEN}▲${NC} Starting 27b-dflash-noviz..."
    compose_at "$DUAL_27B_DIR" "up -d" dflash-noviz.yml && echo "done" || echo "failed"
}
stop_27b_dual_dflash_noviz() {
    printf "  ${RED}▼${NC} Stopping 27b-dflash-noviz..."
    compose_at "$DUAL_27B_DIR" "down" dflash-noviz.yml && echo "done" || echo "skipped"
}

start_27b_dual_turbo() {
    printf "  ${GREEN}▲${NC} Starting 27b-dual-turbo..."
    compose_at "$DUAL_27B_DIR" "up -d" turbo.yml && echo "done" || echo "failed"
}
stop_27b_dual_turbo() {
    printf "  ${RED}▼${NC} Stopping 27b-dual-turbo..."
    compose_at "$DUAL_27B_DIR" "down" turbo.yml && echo "done" || echo "skipped"
}

# Stop every 27b serving variant before starting a new one
stop_all_27b() {
    stop_27b_dual_mtp
    stop_27b_dual_dflash
    stop_27b_dual_dflash_noviz
    stop_27b_dual_turbo
}

# --- ComfyUI (image / video generation) -------------------------------------
# GPU-bound — mutex with all vLLM / SGLang / llama-server LLM serving.
start_comfyui() {
    printf "  ${GREEN}▲${NC} Starting comfyui..."
    compose_at "$COMPOSE_BASE/comfyui" "up -d" && echo "done" || echo "failed"
}
stop_comfyui() {
    printf "  ${RED}▼${NC} Stopping comfyui..."
    compose_at "$COMPOSE_BASE/comfyui" "down" && echo "done" || echo "skipped"
}

# Video-studio sidecars: the always-on media gallery (:8189) and the prompt
# "director" LLM (:8090, GPU0). See services/studio/ + docs/VIDEO_STUDIO.md.
start_studio_gallery() {
    printf "  ${GREEN}▲${NC} Starting studio-gallery (:8189)..."
    compose_at "$COMPOSE_BASE/studio/gallery" "up -d" && echo "done" || echo "failed"
}
start_studio_director() {
    printf "  ${GREEN}▲${NC} Starting studio-director (:8090, GPU0)..."
    compose_at "$COMPOSE_BASE/studio/enhancer" "up -d" && echo "done" || echo "failed"
}
start_studio_orchestrator() {
    printf "  ${GREEN}▲${NC} Starting studio-orchestrator (:8190, long-clip chaining)..."
    compose_at "$COMPOSE_BASE/studio/orchestrator" "up -d --build" && echo "done" || echo "failed"
}
# Native-button image shim (:8191): ComfyUI reverse-proxy that crafts Ideogram-4 JSON
# captions (via the director) so OWUI's native 🖼️ image button renders instead of hitting
# the "blocked by safety filter" placeholder. Needs the director (:8090) up.
start_studio_image_shim() {
    printf "  ${GREEN}▲${NC} Starting studio-image-shim (:8191, native-button Ideogram captions)..."
    compose_at "$COMPOSE_BASE/studio/image-shim" "up -d --build" && echo "done" || echo "failed"
}

# ComfyUI pinned to GPU 0 (image-studio split — leaves the other card for the chat LLM).
start_comfyui_gpu0() {
    printf "  ${GREEN}▲${NC} Starting comfyui (GPU0)..."
    compose_at_env "$COMPOSE_BASE/comfyui" "up -d" docker-compose.yml COMFYUI_CUDA_VISIBLE_DEVICES=0 \
        && echo "done" || echo "failed"
}

# --- gemma-4-12b chat brain (llama.cpp single-card) — image-studio's coexisting LLM ---
# Pinned to the spare GPU (1) so it runs alongside ComfyUI on GPU0. Serves OpenAI API :8069.
start_gemma_12b_chat() {
    printf "  ${GREEN}▲${NC} Starting gemma-4-12b-chat (GPU1)..."
    compose_at_env "$GEMMA_12B_DIR" "up -d" base.yml ESTATE_GPUS=1 CTX_SIZE=32768 PORT=8069 \
        && echo "done" || echo "failed"
}
stop_gemma_12b_chat() {
    printf "  ${RED}▼${NC} Stopping gemma-4-12b-chat..."
    compose_at_env "$GEMMA_12B_DIR" "down" base.yml ESTATE_GPUS=1 CTX_SIZE=32768 PORT=8069 \
        && echo "done" || echo "skipped"
}

# --- Gemma 4 31B dual-card serving variants ---------------------------------
start_gemma_mtp() {
    printf "  ${GREEN}▲${NC} Starting gemma-mtp..."
    compose_at "$GEMMA_DUAL_DIR" "up -d" bf16-mtp.yml && echo "done" || echo "failed"
}
stop_gemma_mtp() {
    printf "  ${RED}▼${NC} Stopping gemma-mtp..."
    compose_at "$GEMMA_DUAL_DIR" "down" bf16-mtp.yml && echo "done" || echo "skipped"
}

start_gemma_int8() {
    printf "  ${GREEN}▲${NC} Starting gemma-int8..."
    compose_at "$GEMMA_DUAL_DIR" "up -d" int8.yml && echo "done" || echo "failed"
}
stop_gemma_int8() {
    printf "  ${RED}▼${NC} Stopping gemma-int8..."
    compose_at "$GEMMA_DUAL_DIR" "down" int8.yml && echo "done" || echo "skipped"
}

# Stop every Gemma serving variant before starting a new one
stop_all_gemma() {
    stop_gemma_mtp
    stop_gemma_int8
}

start_deckard() {
    printf "  ${GREEN}▲${NC} Starting deckard-40b..."
    compose_at "$DECKARD_DIR" "up -d" mtp.yml && echo "done" || echo "failed"
}
stop_deckard() {
    printf "  ${RED}▼${NC} Stopping deckard-40b..."
    compose_at "$DECKARD_DIR" "down" mtp.yml && echo "done" || echo "skipped"
}
start_diffusiongemma() {
    printf "  ${GREEN}▲${NC} Starting diffusiongemma-26b-a4b (dLLM)..."
    # PORT=8199 = the dual-card "active big model" slot (shared with deckard).
    compose_at_env "$DGEMMA_DIR" "up -d" base.yml PORT=8199 && echo "done" || echo "failed"
}
stop_diffusiongemma() {
    printf "  ${RED}▼${NC} Stopping diffusiongemma-26b-a4b..."
    compose_at_env "$DGEMMA_DIR" "down" base.yml PORT=8199 && echo "done" || echo "skipped"
}

show_status() {
    echo ""
    echo -e "${CYAN}═══ Service Status ═══${NC}"
    sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null
    echo ""
    echo -e "${CYAN}═══ Active Model(s) ═══${NC}"
    # Check ports in priority order: 8010, 8012, 8020, 11434, 4000
    if curl -sf -m 2 http://localhost:8010/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8010/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} 27b-dual-mtp @ :8010    → ${m:-unknown} (MTP n=3 + fp8 + 262K + vision)"
    fi
    if curl -sf -m 2 http://localhost:8012/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8012/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} 27b-dflash @ :8012      → ${m:-unknown} (DFlash N=5 + 185K + vision)"
    fi
    if curl -sf -m 2 http://localhost:8013/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8013/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} 27b-dflash-noviz @ :8013 → ${m:-unknown} (DFlash N=5 + 200K, no vision)"
    fi
    if curl -sf -m 2 http://localhost:8011/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8011/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} 27b-turbo @ :8011       → ${m:-unknown} (TurboQuant_3bit_nc + MTP n=3 + v7.14, 4-stream concurrency)"
    fi
    # :8020 = llama.cpp single-card. llamacpp/default + llamacpp/mtp share the
    # base container llama-cpp-qwen36-27b (same compose, collapsed 2026-05-22);
    # llamacpp/mtp-vision now defaults to llama-cpp-qwen36-27b-vision (#169).
    # All still match the llama-cpp-* prefix used for detection below.
    if curl -sf -m 2 http://localhost:8020/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8020/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} llamacpp/single @ :8020 → ${m:-unknown} (llama.cpp single-card)"
    fi
    if curl -sf -m 2 http://localhost:8030/v1/models >/dev/null 2>&1; then
        local m container engine_tag
        m=$(curl -sf -m 2 http://localhost:8030/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        # Detect engine via container name on the port (was hardcoded to "gemma-mtp"
        # / Gemma description; post-v0.8.3, llamacpp/mtp-vision also lands on :8030).
        container=$(sudo docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/:8030->/ {print $1; exit}')
        if [[ "$container" == llama-cpp-* ]]; then
            engine_tag="llamacpp/mtp-vision @ :8030 → ${m:-unknown} (Q4_K_M + MTP + vision, 49K)"
        else
            engine_tag="gemma-mtp @ :8030          → ${m:-unknown} (Gemma 4 31B + MTP n=3 + bf16 KV + 32K)"
        fi
        echo -e "  ${GREEN}▶${NC} $engine_tag"
    fi
    if curl -sf -m 2 http://localhost:8032/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8032/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} gemma-int8 @ :8032        → ${m:-unknown} (INT8 PTH KV)"
    fi
    if curl -sf -m 2 http://localhost:8069/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8069/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} gemma-4-12b @ :8069      → ${m:-unknown} (image-studio chat brain, GPU1, llama.cpp)"
    fi
    if curl -sf -m 2 http://localhost:8188/ >/dev/null 2>&1; then
        echo -e "  ${GREEN}▶${NC} ComfyUI @ :8188          → image/video generation (GPU-bound, mutex with LLM)"
    fi
    if curl -sf -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:11434/api/tags | python3 -c "import sys,json;d=json.load(sys.stdin);mdls=[x['name'] for x in d.get('models',[])];print(f'{len(mdls)} models available' if mdls else 'none loaded')" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} Ollama @ :11434         → ${m:-unknown}"
    fi
    if curl -sf -m 2 -H "Authorization: Bearer sk-litellm-master-key" http://localhost:4000/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 -H "Authorization: Bearer sk-litellm-master-key" http://localhost:4000/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} LiteLLM @ :4000         → ${m:-unknown}"
    fi
    if ! curl -sf -m 2 http://localhost:8010/v1/models >/dev/null 2>&1 \
      && ! curl -sf -m 2 http://localhost:8011/v1/models >/dev/null 2>&1 \
      && ! curl -sf -m 2 http://localhost:8012/v1/models >/dev/null 2>&1 \
      && ! curl -sf -m 2 http://localhost:8013/v1/models >/dev/null 2>&1 \
      && ! curl -sf -m 2 http://localhost:8030/v1/models >/dev/null 2>&1 \
      && ! curl -sf -m 2 http://localhost:8032/v1/models >/dev/null 2>&1 \
      && ! curl -sf -m 2 http://localhost:8033/v1/models >/dev/null 2>&1 \
      && ! curl -sf -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e "  ${YELLOW}(no inference endpoint responding)${NC}"
    fi
    echo ""
    echo -e "${CYAN}═══ GPU Status ═══${NC}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
    # One-line power-cap state: enforced vs default per card (see 'gpu-mode power-cap').
    nvidia-smi --query-gpu=index,power.limit,power.default_limit --format=csv,noheader,nounits 2>/dev/null \
      | while IFS=',' read -r gi lim def; do
            gi="${gi// /}"; lim="${lim// /}"; def="${def// /}"
            if awk "BEGIN{exit !($lim < $def)}"; then
                echo -e "  power cap: GPU ${gi} ${YELLOW}${lim}W${NC} (capped; default ${def}W)"
            else
                echo -e "  power cap: GPU ${gi} ${GREEN}${lim}W${NC} (uncapped; default ${def}W)"
            fi
        done || true
    echo ""
    echo -e "${CYAN}═══ RAM Status ═══${NC}"
    free -h | head -2
    echo ""
    echo -e "${CYAN}═══ Disk Status ═══${NC}"
    df -h / /mnt/models 2>/dev/null | tail -2
    echo ""
    echo -e "${CYAN}═══ Docker Disk ═══${NC}"
    sudo docker system df 2>/dev/null | head -5 || echo "(docker not running)"
    local docker_dir_size tmp_size
    docker_dir_size=$(sudo du -sh /var/lib/docker 2>/dev/null | cut -f1)
    tmp_size=$(sudo du -sh /tmp 2>/dev/null | cut -f1)
    echo ""
    echo "  /var/lib/docker (on /):  ${docker_dir_size:-?}"
    echo "  /tmp (on /):             ${tmp_size:-?}"
    echo ""
}

mode_prune() {
    echo -e "${CYAN}═══ Docker prune (safe) ═══${NC}"
    echo "Removes images not referenced by any container (running OR stopped)."
    echo "Does NOT touch build cache or volumes — use 'prune-all' for those."
    echo ""
    echo "${CYAN}── Before ──${NC}"
    sudo docker system df 2>/dev/null | head -5
    echo ""
    sudo docker image prune -a -f 2>&1 | tail -10
    echo ""
    echo "${CYAN}── After ──${NC}"
    sudo docker system df 2>/dev/null | head -5
}

mode_prune_all() {
    echo -e "${CYAN}═══ Docker prune (aggressive) ═══${NC}"
    echo "Removes:"
    echo "  - images not referenced by any container"
    echo "  - all build cache (kept ≤5 GB)"
    echo "  - dangling networks"
    echo "Does NOT remove volumes (qdrant-data, openwebui-data are safe)."
    echo ""
    echo "${CYAN}── Before ──${NC}"
    sudo docker system df 2>/dev/null | head -5
    echo ""
    echo "${YELLOW}Pruning images...${NC}"
    sudo docker image prune -a -f 2>&1 | tail -3
    echo ""
    echo "${YELLOW}Pruning networks...${NC}"
    sudo docker network prune -f 2>&1 | tail -3
    echo ""
    echo "${YELLOW}Pruning build cache (keeping 5 GB)...${NC}"
    sudo docker buildx prune -f --keep-storage 5GB 2>&1 | tail -3
    echo ""
    echo "${CYAN}── After ──${NC}"
    sudo docker system df 2>/dev/null | head -5
}

mode_chat() {
    echo -e "${CYAN}═══ Switching to CHAT mode ═══${NC}"
    echo "Starting: Open WebUI, LiteLLM, Qdrant, SearXNG"
    echo "Stopping: all GPU-served model containers (Qwen + Gemma)"
    echo ""
    stop_all_27b
    stop_deckard
    stop_all_gemma
    stop_comfyui
    start_service openwebui
    start_service litellm
    start_service qdrant
    start_service searxng
    echo ""
    echo -e "${GREEN}Chat mode active.${NC} Open WebUI: http://192.168.86.33:8080"
}

mode_27b() {
    echo -e "${CYAN}═══ Switching to 27B dual-card MTP mode (default) ═══${NC}"
    echo "Starting: Qwen3.6-27B MTP n=3 + fp8 KV + 262K + vision + 2 streams (TP=2)"
    echo "Port: 8010 | Container: vllm-qwen36-27b-dual"
    echo "Stopping: Ollama, other 27B variants"
    echo ""
    stop_service ollama
    stop_all_gemma
    stop_comfyui
    stop_27b_dual_dflash
    stop_27b_dual_dflash_noviz
    stop_27b_dual_turbo
    start_27b_dual_mtp
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}27B dual-card MTP mode active.${NC} API: http://192.168.86.33:8010"
    echo -e "${YELLOW}Per-stream: 68 narr / 89 code TPS short, 36 TPS @ 100K, 28 TPS @ 200K warm.${NC}"
    echo -e "${YELLOW}2 concurrent streams. KV pool 168K, max concurrency 2.36× at full 262K.${NC}"
    echo -e "${YELLOW}Vision + tools + thinking + 262K ctx all working. Boot ~3-4 min.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-qwen36-27b-dual${NC}"
}

mode_gemma() {
    echo -e "${CYAN}═══ Switching to Gemma 4 31B MTP mode (bf16 fallback) ═══${NC}"
    echo "Starting: Gemma 4 31B (Intel AutoRound INT4) + MTP n=3 + bf16 KV + 32K + vision (TP=2)"
    echo "Port: 8030 | Container: vllm-gemma-4-31b-mtp"
    echo "Stopping: Ollama, all 27B Qwen variants, other Gemma variants"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_deckard
    stop_gemma_int8
    start_gemma_mtp
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}Gemma 4 31B MTP mode active.${NC} API: http://192.168.86.33:8030"
    echo -e "${YELLOW}109 narr / 141 code TPS (AL 3.05 / 3.99). 32K ctx (BF16 ceiling).${NC}"
    echo -e "${YELLOW}For 262K ctx use 'gemma' (the default — INT8 PTH KV). Boot ~2-3 min.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-gemma-4-31b-mtp${NC}"
}

mode_gemma_dflash() {
    echo -e "${CYAN}═══ Switching to Gemma 4 31B DFlash mode ═══${NC}"
    echo "Starting: Gemma 4 31B + z-lab DFlash drafter (TP=2, :8032)"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_deckard
    stop_gemma_mtp
    stop_gemma_int8
    stop_gemma_dflash_int8
    stop_gemma_awq
    start_gemma_dflash
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}Gemma 4 31B DFlash mode active.${NC} API: http://192.168.86.33:8032"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-gemma-4-31b-dflash${NC}"
}

mode_gemma_int8() {
    echo -e "${CYAN}═══ Switching to Gemma 4 31B INT8-PTH mode (dual default, long ctx) ═══${NC}"
    echo "Starting: Gemma 4 31B + INT8 PTH KV + 262K ctx (TP=2, :8032)"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_deckard
    stop_gemma_mtp
    stop_gemma_dflash
    stop_gemma_dflash_int8
    stop_gemma_awq
    start_gemma_int8
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}Gemma 4 31B INT8 PTH mode active.${NC} API: http://192.168.86.33:8032"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-gemma-4-31b-mtp-int8${NC}"
}
mode_deckard() {
    echo -e "${CYAN}═══ Switching to DECKARD-40B mode (uncensored, dual-card) ═══${NC}"
    echo "Starting: Qwen3.6-40B-Deckard Q6_K + MTP n=2 + q8_0 KV + 128K ctx (llama.cpp, :8199)"
    echo "Stopping: all other GPU models (Deckard layer-splits across both cards)"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_all_gemma
    stop_gemma_12b_chat
    stop_comfyui
    start_deckard
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    # Deckard isn't in the LiteLLM gateway config, so wire it into Open WebUI
    # directly as an OpenAI connection (reuses switch.sh --owui's helper).
    # Best-effort: a no-op if OWUI isn't running / not ready yet.
    if [ -x "$CLUB3090_DIR/scripts/lib/owui-register.sh" ]; then
        bash "$CLUB3090_DIR/scripts/lib/owui-register.sh" 8199 || true
    fi
    echo ""
    echo -e "${GREEN}Deckard-40B mode active.${NC} API: http://192.168.86.33:8199  (model: deckard-40b)"
    echo -e "${YELLOW}MTP n=2: ~36 narr / 46 code TPS · 128K ctx @ q8_0 KV · uncensored, text-only.${NC}"
    echo -e "${YELLOW}First boot ~1-2 min (31 GB GGUF load + 128K KV alloc across both cards).${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f llama-cpp-deckard-40b${NC}"
}

mode_gemma_dflash_int8() {
    echo -e "${CYAN}═══ Switching to Gemma 4 31B DFlash + INT8 PTH mode ═══${NC}"
    echo "Starting: Gemma 4 31B + DFlash + INT8 PTH KV (TP=2, :8032). Requires vllm#42102."
    echo ""
    stop_service ollama
    stop_all_27b
    stop_deckard
    stop_gemma_mtp
    stop_gemma_dflash
    stop_gemma_int8
    stop_gemma_awq
    start_gemma_dflash_int8
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}Gemma 4 31B DFlash + INT8 mode active.${NC} API: http://192.168.86.33:8032"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-gemma-4-31b-dflash-int8${NC}"
}

mode_gemma_awq() {
    echo -e "${CYAN}═══ Switching to Gemma 4 31B AWQ-4bit mode ═══${NC}"
    echo "Starting: Gemma 4 31B AWQ-4bit (TP=2, :8033)"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_deckard
    stop_gemma_mtp
    stop_gemma_dflash
    stop_gemma_int8
    stop_gemma_dflash_int8
    start_gemma_awq
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}Gemma 4 31B AWQ mode active.${NC} API: http://192.168.86.33:8033"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-gemma-4-31b-awq${NC}"
}

mode_diffusiongemma() {
    echo -e "${CYAN}═══ Switching to DiffusionGemma 26B-A4B mode (dLLM, dual-card) 🧪 ═══${NC}"
    echo "Starting: DiffusionGemma 26B-A4B (vLLM's first diffusion LM) — official :gemma image"
    echo "          + 3 Ampere/TP fix-mounts, fp8, TP=2, 262K (vLLM, :8199)"
    echo "Stopping: all other GPU models (DiffusionGemma uses both cards, TP=2)"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_all_gemma
    stop_gemma_12b_chat
    stop_comfyui
    stop_deckard
    start_diffusiongemma
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    # Not in the LiteLLM gateway config → wire into Open WebUI directly (reuses
    # switch.sh --owui's helper). Best-effort: no-op if OWUI isn't up yet.
    if [ -x "$CLUB3090_DIR/scripts/lib/owui-register.sh" ]; then
        bash "$CLUB3090_DIR/scripts/lib/owui-register.sh" 8199 || true
    fi
    echo ""
    echo -e "${GREEN}DiffusionGemma mode active.${NC} API: http://192.168.86.33:8199  (model: diffusiongemma-26b-a4b)"
    echo -e "${YELLOW}🧪 experimental — block-parallel dLLM; SSE streams a whole canvas per chunk.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-diffusiongemma-26b-a4b-fp8-tp2${NC}"
}

mode_comfyui() {
    echo -e "${CYAN}═══ Switching to ComfyUI mode (image / video gen) ═══${NC}"
    echo "Starting: ComfyUI :8188"
    echo "Stopping: all GPU-bound LLM serving (Qwen + Gemma)"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_deckard
    stop_all_gemma
    stop_gemma_12b_chat
    start_comfyui
    echo ""
    echo -e "${GREEN}ComfyUI mode active.${NC} UI: http://192.168.86.33:8188"
    echo -e "${YELLOW}First boot ~2-3 min while entrypoint clones ComfyUI + custom nodes.${NC}"
    echo -e "${YELLOW}GPU-bound, mutex with vLLM/SGLang. No LiteLLM routing (ComfyUI is non-OpenAI).${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f comfyui${NC}"
}

mode_video_studio() {
    echo -e "${CYAN}═══ Switching to VIDEO-STUDIO mode (text/image → video) ═══${NC}"
    echo "Starting: ComfyUI :8188 (both GPUs) + director :8090 + gallery :8189 + Open WebUI"
    echo "Stopping: all GPU-bound LLM serving (Qwen + Gemma + DiffusionGemma)"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_deckard
    stop_all_gemma
    stop_gemma_12b_chat
    stop_diffusiongemma
    start_comfyui
    start_studio_director
    start_studio_gallery
    start_studio_orchestrator
    start_studio_image_shim
    start_service openwebui
    start_service litellm
    start_service searxng
    echo ""
    echo -e "${GREEN}Video-studio mode active.${NC}"
    echo -e "  Open WebUI:  http://192.168.86.33:8080   (pick 🎬 Studio · LTX or 🔓 Studio · Sulphur)"
    echo -e "  Gallery:     http://192.168.86.33:8189   (all generated media; survives ComfyUI down)"
    echo -e "  ComfyUI:     http://192.168.86.33:8188   (full node graph / control)"
    echo -e "${YELLOW}First ComfyUI boot can take a few min (clones + node deps). The 22B DiT splits across both 3090s (DisTorch).${NC}"
    echo -e "${YELLOW}GPU-mutex with the dual-card LLMs. Clips default ~10s, cap ~15s — see docs/VIDEO_STUDIO.md.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f comfyui${NC}"
}

mode_image_studio() {
    echo -e "${CYAN}═══ Switching to IMAGE-STUDIO mode (image gen + chat, 2-card split) ═══${NC}"
    echo "Starting: ComfyUI/Ideogram-4 on GPU0 + gemma-4-12b chat on GPU1 + Open WebUI"
    echo "Stopping: all dual-card LLM serving (Qwen + Gemma-31B)"
    echo ""
    local ngpu
    ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    stop_service ollama
    stop_all_27b
    stop_deckard
    stop_all_gemma
    if [ "${ngpu:-0}" -lt 2 ]; then
        echo -e "${YELLOW}⚠ Only ${ngpu:-0} GPU detected — image gen + a local chat model can't coexist"
        echo -e "  (both are GPU-resident). Starting ComfyUI image gen only.${NC}"
        echo -e "  ${YELLOW}For chat: use 'gpu-mode chat' (LiteLLM) or run gemma-4-12b when ComfyUI is down.${NC}"
        start_comfyui   # default (all GPUs) — single-card box, nothing to split
    else
        start_comfyui_gpu0
        start_gemma_12b_chat
        start_studio_director       # qwen director on GPU0 (~4.6GB) — crafts Ideogram JSON for the image shim
        start_studio_image_shim     # native 🖼️ button -> clean Ideogram images (see docs/VIDEO_STUDIO.md)
    fi
    start_service openwebui
    start_service litellm
    start_service searxng
    echo ""
    echo -e "${GREEN}Image-studio mode active.${NC}"
    echo -e "  Open WebUI:  http://192.168.86.33:8080   (chat + 🖼️ image button)"
    echo -e "  ComfyUI:     http://192.168.86.33:8188   (full node graph / control)"
    if [ "${ngpu:-0}" -ge 2 ]; then
        echo -e "  Chat model:  gemma-4-12b @ :8069 (GPU1) — OpenWebUI default"
    fi
    echo -e "${YELLOW}First ComfyUI boot ~2-3 min (clones HEAD + nodes); first image ~2 min cold / ~70 s warm.${NC}"
    echo -e "${YELLOW}If the OpenWebUI image button is missing on an existing volume: Admin → Settings → Images.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f comfyui | sudo docker logs -f llama-cpp-gemma4-12b${NC}"
}

mode_bigmodel() {
    echo -e "${CYAN}═══ Switching to BIG MODEL mode ═══${NC}"
    echo "Stopping ALL containers to maximize RAM + VRAM..."
    echo ""
    stop_all_27b
    stop_deckard
    stop_all_gemma
    stop_comfyui
    for svc in "${SERVICES[@]}"; do
        stop_service "$svc"
    done
    echo ""
    echo "Dropping filesystem caches..."
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    echo ""
    echo -e "${CYAN}═══ Available Resources ═══${NC}"
    echo -e "VRAM:"
    nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader 2>/dev/null
    echo -e "RAM:"
    free -h | grep Mem | awk '{print "  Free: "$4" / Total: "$2}'
    echo ""
    echo -e "${GREEN}Big model mode active.${NC} All containers stopped, max RAM+VRAM available."
    echo ""
    echo -e "Example: run a custom GGUF with llama-server:"
    echo -e "  llama-server --model /mnt/models/gguf/<file>.gguf \\"
    echo -e "    --n-gpu-layers 99 --ctx-size 32768 --host 0.0.0.0 --port 8001"
}

stop_estate() {
    # Tear down any estate-managed instances (launch.sh --estate-file or --estate
    # bookings persist via Docker `restart: unless-stopped`). No-op if no estate
    # plan exists or launch.sh is unavailable.
    if [[ ! -f "$ESTATE_YAML" ]]; then
        return 0
    fi
    if ! command -v bash >/dev/null 2>&1 || [[ ! -x "$CLUB3090_DIR/scripts/launch.sh" ]]; then
        return 0
    fi
    if ! python3 -c "import yaml; d=yaml.safe_load(open('$ESTATE_YAML')); raise SystemExit(0 if d and d.get('estate') else 1)" 2>/dev/null; then
        return 0  # empty/missing estate list
    fi
    printf "  ${RED}▼${NC} Stopping estate-managed instances..."
    if bash "$CLUB3090_DIR/scripts/launch.sh" --down-estate "$ESTATE_YAML" >/dev/null 2>&1; then
        echo "done"
    else
        echo "skipped (no instances or already down)"
    fi
}

# --- GPU power-cap controls -------------------------------------------------
# The rig normally runs both 3090s capped at 230W (quieter / cooler — see the
# systemd unit below). The cap suppresses benchmark TPS, so maintainers need a
# quick way to uncap to the hardware default for a true-TPS bench, then re-cap.
#
# `nvidia-power-cap.service` is the single source of truth for the 230W value
# AND re-applies it on every boot (Type=oneshot, RemainAfterExit=yes, enabled).
# So `power-cap on` *restarts* that unit — `restart` (not `start`) is required:
# the unit is already `active` from boot, and `systemctl start` on an
# already-active RemainAfterExit oneshot is a no-op (it won't re-run ExecStart,
# so the cap wouldn't actually re-apply after a `power-cap off`). `restart`
# stops it (clearing RemainAfterExit) then re-runs both `-pl 230` ExecStart
# lines. `power-cap off` reads each card's Default Power Limit from nvidia-smi
# (370W on GPU 0, 420W on GPU 1 here — they differ, so we never hardcode) and
# applies it. `off` is session-scoped: a reboot OR a driver reload re-applies
# 230W via the service. We never disable the service.
POWER_CAP_SERVICE="nvidia-power-cap.service"

# Print per-GPU enforced / default / min / max power limits (one row per card).
powercap_show() {
    echo -e "${CYAN}═══ GPU Power Limits ═══${NC}"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "  ${RED}✗ nvidia-smi not found${NC} — cannot read power limits."
        return 1
    fi
    if ! nvidia-smi \
        --query-gpu=index,power.limit,power.default_limit,power.min_limit,power.max_limit \
        --format=csv 2>/dev/null; then
        echo -e "  ${RED}✗ nvidia-smi query failed${NC} — driver loaded?"
        return 1
    fi
}

# Echo the current enforced limit per GPU (used after on/off to confirm effect).
powercap_echo_enforced() {
    local line
    while IFS= read -r line; do
        echo -e "  ${GREEN}▶${NC} GPU ${line%%,*} enforced limit:${line#*,} W"
    done < <(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits 2>/dev/null)
}

mode_powercap() {
    local action="${1:-status}"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${RED}✗ nvidia-smi not found.${NC} Install the NVIDIA driver / utils first." >&2
        exit 1
    fi
    case "$action" in
        on)
            echo -e "${CYAN}═══ Re-applying GPU power cap (230W) ═══${NC}"
            echo "Restarting ${POWER_CAP_SERVICE} (the boot-time 230W enforcer)."
            # restart, not start — the unit is already active from boot, so
            # `start` is a no-op on a RemainAfterExit oneshot (won't re-run -pl).
            if sudo systemctl restart "$POWER_CAP_SERVICE" 2>/dev/null; then
                echo -e "${GREEN}Power cap re-applied via systemd.${NC}"
            else
                # Fallback: service missing/disabled — apply 230W directly.
                echo -e "${YELLOW}systemctl restart failed; falling back to direct nvidia-smi -pl 230.${NC}" >&2
                if ! { sudo nvidia-smi -i 0 -pl 230 && sudo nvidia-smi -i 1 -pl 230; }; then
                    echo -e "${RED}✗ Failed to set 230W cap.${NC} Check sudo + driver state with: nvidia-smi -q -d POWER" >&2
                    exit 1
                fi
            fi
            powercap_echo_enforced
            ;;
        off)
            echo -e "${CYAN}═══ Uncapping GPUs to hardware default ═══${NC}"
            # Read each card's Default Power Limit — they can differ (370 vs 420
            # here), and nvidia-smi has no "reset" flag, so we pass the value.
            local idx def rc=0 applied=0
            while IFS=',' read -r idx def; do
                idx="${idx// /}"
                def="${def// /}"
                [ -z "$idx" ] && continue
                echo "  Setting GPU ${idx} → ${def} W (default)..."
                if ! sudo nvidia-smi -i "$idx" -pl "$def" >/dev/null 2>&1; then
                    echo -e "  ${RED}✗ Failed to set GPU ${idx} to ${def} W${NC} (sudo? driver?)." >&2
                    rc=1
                else
                    applied=1
                fi
            done < <(nvidia-smi --query-gpu=index,power.default_limit \
                --format=csv,noheader,nounits 2>/dev/null)
            if [ "$applied" -eq 0 ]; then
                echo -e "${RED}✗ No GPUs updated.${NC} Check: nvidia-smi -q -d POWER" >&2
                exit 1
            fi
            echo -e "${GREEN}Uncapped to default.${NC} ${YELLOW}Session-scoped — a reboot or driver"
            echo -e "reload re-applies 230W via ${POWER_CAP_SERVICE}. Run 'gpu-mode power-cap on' to re-cap now.${NC}"
            powercap_echo_enforced
            [ "$rc" -eq 0 ] || exit 1
            ;;
        status)
            powercap_show
            ;;
        *)
            echo -e "${RED}Unknown power-cap action:${NC} $action" >&2
            echo "Usage: gpu-mode power-cap <on|off|status>" >&2
            exit 1
            ;;
    esac
}

mode_off() {
    echo -e "${CYAN}═══ Stopping ALL services ═══${NC}"
    stop_all_27b
    stop_deckard
    stop_diffusiongemma
    stop_all_gemma
    stop_comfyui
    stop_estate
    for svc in "${SERVICES[@]}"; do
        stop_service "$svc"
    done
    echo ""
    echo -e "${GREEN}All services stopped.${NC}"
}

usage() {
    echo ""
    echo -e "${CYAN}GPU Mode Switcher${NC} — AI Inference Stack Manager"
    echo ""
    echo "Usage: gpu-mode <mode>"
    echo ""
    echo "Modes:"
    echo "  chat               Ollama + Open WebUI + LiteLLM + Qdrant (browser chat, no GPU model)"
    echo ""
    echo "  Qwen 3.6 27B (dual 3090, TP=2):"
    echo "  27b                ⭐ DEFAULT — Qwen3.6-27B MTP + fp8 + 262K + vision + 2 streams (:8010)"
    echo ""
    echo "  Gemma 4 31B (dual 3090, TP=2):"
    echo "  gemma              ⭐ DEFAULT — Gemma 4 31B INT8 PTH KV + 262K + vision (:8032)"
    echo "  gemma-int8         alias for 'gemma' (INT8 PTH KV; 98K default, CTX=262144 MAX_NUM_SEQS=1 for native 262K)"
    echo "  gemma-mtp          bf16 KV fallback — 32K, stock vLLM v0.22.0, no overlay (:8030)"
    echo ""
    echo "  Qwen 3.6 40B Deckard (uncensored, dual 3090, llama.cpp):"
    echo "  deckard            Q6_K + MTP n=2 + q8_0 KV + 128K ctx (:8199) — text-only, both cards"
    echo "  dgemma             DiffusionGemma 26B-A4B dLLM 🧪 (:8199) — fp8, 262K, both cards (run 'off' before switching)"
    echo ""
    echo "  Image / Video Gen:"
    echo "  image-studio       ⭐ Ideogram-4 image gen (GPU0) + gemma-4-12b chat (GPU1) + Open WebUI"
    echo "                     — chat + image coexist on 2 cards (alias: imagestudio)"
    echo "  comfyui            ComfyUI :8188 only, all GPUs (FLUX/Hunyuan/Wan; mutex with LLM)"
    echo "  video-studio       text/image→video (LTX-2.3 / Sulphur) — ComfyUI both GPUs +"
    echo "                     director (:8090) + gallery (:8189) + Open WebUI (alias: videostudio)"
    echo ""
    echo "  bigmodel           Stop everything, max RAM+VRAM for one-off llama-server / custom workloads"
    echo "  off                Stop all services"
    echo "  status             Show running services, GPU, RAM, disk, Docker disk"
    echo ""
    echo "  GPU power cap (both 3090s; normally capped at 230W for quiet/cool operation):"
    echo "  power-cap on       Re-apply the 230W cap (via nvidia-power-cap.service)"
    echo "  power-cap off      Uncap to hardware default for a true-TPS bench"
    echo "                     (session-scoped — a reboot / driver reload re-caps at 230W)"
    echo "  power-cap status   Show per-GPU enforced / default / min / max power limits"
    echo "                     (alias: powercap)"
    echo ""
    echo "  Maintenance:"
    echo "  prune              docker image prune -a (safe — only unreferenced images)"
    echo "  prune-all          + build cache (keep 5 GB) + dangling networks (volumes safe)"
    echo ""
}

case "${1:-}" in
    chat)               mode_chat ;;
    27b)                mode_27b ;;
    gemma)              mode_gemma_int8 ;;
    gemma-int8)         mode_gemma_int8 ;;
    gemma-mtp)          mode_gemma ;;
    deckard)            mode_deckard ;;
    diffusiongemma|dgemma) mode_diffusiongemma ;;
    comfyui)            mode_comfyui ;;
    image-studio|imagestudio) mode_image_studio ;;
    video-studio|videostudio) mode_video_studio ;;
    bigmodel)           mode_bigmodel ;;
    off)                mode_off ;;
    status)             show_status ;;
    power-cap|powercap) mode_powercap "${2:-status}" ;;
    prune)              mode_prune ;;
    prune-all)          mode_prune_all ;;
    *)                  usage ;;
esac
