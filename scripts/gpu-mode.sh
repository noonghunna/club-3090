#!/bin/bash
# GPU/RAM Mode Switcher for AI Inference Stack
# Manages Docker containers to avoid GPU/RAM contention on dual-3090 setup
# Location: club-3090/scripts/gpu-mode.sh (symlinked to /usr/local/bin/gpu-mode)

set -e

# club-3090 is the canonical repo (qwen36-dual-3090 + /opt/ai/compose/<svc>
# both deprecated 2026-05-10 — supporting services moved into services/).
CLUB3090_DIR="/opt/ai/github/club-3090"
COMPOSE_BASE="$CLUB3090_DIR/services"
DUAL_27B_DIR="$CLUB3090_DIR/models/qwen3.6-27b/vllm/compose/dual"
GEMMA_DUAL_DIR="$CLUB3090_DIR/models/gemma-4-31b/vllm/compose/dual"

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
    compose_at "$DUAL_27B_DIR" "up -d" docker-compose.yml && echo "done" || echo "failed"
}
stop_27b_dual_mtp() {
    printf "  ${RED}▼${NC} Stopping 27b-dual-mtp..."
    compose_at "$DUAL_27B_DIR" "down" docker-compose.yml && echo "done" || echo "skipped"
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

# --- Gemma 4 31B dual-card serving variants ---------------------------------
start_gemma_mtp() {
    printf "  ${GREEN}▲${NC} Starting gemma-mtp..."
    compose_at "$GEMMA_DUAL_DIR" "up -d" docker-compose.yml && echo "done" || echo "failed"
}
stop_gemma_mtp() {
    printf "  ${RED}▼${NC} Stopping gemma-mtp..."
    compose_at "$GEMMA_DUAL_DIR" "down" docker-compose.yml && echo "done" || echo "skipped"
}

start_gemma_dflash() {
    printf "  ${GREEN}▲${NC} Starting gemma-dflash..."
    compose_at "$GEMMA_DUAL_DIR" "up -d" dflash.yml && echo "done" || echo "failed"
}
stop_gemma_dflash() {
    printf "  ${RED}▼${NC} Stopping gemma-dflash..."
    compose_at "$GEMMA_DUAL_DIR" "down" dflash.yml && echo "done" || echo "skipped"
}

start_gemma_int8() {
    printf "  ${GREEN}▲${NC} Starting gemma-int8..."
    compose_at "$GEMMA_DUAL_DIR" "up -d" int8.yml && echo "done" || echo "failed"
}
stop_gemma_int8() {
    printf "  ${RED}▼${NC} Stopping gemma-int8..."
    compose_at "$GEMMA_DUAL_DIR" "down" int8.yml && echo "done" || echo "skipped"
}

start_gemma_dflash_int8() {
    printf "  ${GREEN}▲${NC} Starting gemma-dflash-int8..."
    compose_at "$GEMMA_DUAL_DIR" "up -d" dflash-int8.yml && echo "done" || echo "failed"
}
stop_gemma_dflash_int8() {
    printf "  ${RED}▼${NC} Stopping gemma-dflash-int8..."
    compose_at "$GEMMA_DUAL_DIR" "down" dflash-int8.yml && echo "done" || echo "skipped"
}

start_gemma_awq() {
    printf "  ${GREEN}▲${NC} Starting gemma-awq..."
    compose_at "$GEMMA_DUAL_DIR" "up -d" awq.yml && echo "done" || echo "failed"
}
stop_gemma_awq() {
    printf "  ${RED}▼${NC} Stopping gemma-awq..."
    compose_at "$GEMMA_DUAL_DIR" "down" awq.yml && echo "done" || echo "skipped"
}

# Stop every Gemma serving variant before starting a new one
stop_all_gemma() {
    stop_gemma_mtp
    stop_gemma_dflash
    stop_gemma_int8
    stop_gemma_dflash_int8
    stop_gemma_awq
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
    if curl -sf -m 2 http://localhost:8030/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8030/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} gemma-mtp @ :8030       → ${m:-unknown} (Gemma 4 31B + MTP n=3 + bf16 KV + 32K)"
    fi
    if curl -sf -m 2 http://localhost:8032/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8032/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} gemma-{dflash|int8} @ :8032 → ${m:-unknown}"
    fi
    if curl -sf -m 2 http://localhost:8033/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8033/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} gemma-awq @ :8033        → ${m:-unknown} (AWQ-4bit)"
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

mode_27b_dflash() {
    echo -e "${CYAN}═══ Switching to 27B DFlash mode (with vision) ═══${NC}"
    echo "Starting: Qwen3.6-27B DFlash N=5 + 185K + vision (TP=2, single stream)"
    echo "Port: 8012 | Container: vllm-qwen36-27b-dual-dflash"
    echo "Stopping: Ollama, other 27B variants"
    echo ""
    stop_service ollama
    stop_all_gemma
    stop_comfyui
    stop_27b_dual_mtp
    stop_27b_dual_dflash_noviz
    stop_27b_dual_turbo
    start_27b_dual_dflash
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}27B DFlash mode active.${NC} API: http://192.168.86.33:8012"
    echo -e "${YELLOW}78 narr / 128 code TPS single-stream — fastest single-user with vision.${NC}"
    echo -e "${YELLOW}185K ctx + vision + tools. Single concurrent stream (KV pool 66,912 tokens).${NC}"
    echo -e "${YELLOW}Requires --dtype bfloat16 (vLLM PR #40334 workaround). Boot ~5-6 min.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-qwen36-27b-dual-dflash${NC}"
}

mode_27b_dflash_noviz() {
    echo -e "${CYAN}═══ Switching to 27B DFlash mode (text-only, max ctx) ═══${NC}"
    echo "Starting: Qwen3.6-27B DFlash N=5 + 200K, no vision (TP=2, single stream)"
    echo "Port: 8013 | Container: vllm-qwen36-27b-dual-dflash-noviz"
    echo "Stopping: Ollama, other 27B variants"
    echo ""
    stop_service ollama
    stop_all_gemma
    stop_comfyui
    stop_27b_dual_mtp
    stop_27b_dual_dflash
    stop_27b_dual_turbo
    start_27b_dual_dflash_noviz
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}27B DFlash text-only mode active.${NC} API: http://192.168.86.33:8013"
    echo -e "${YELLOW}77 narr / 124 code TPS single-stream. 200K ctx (no vision tower → +15K vs vision variant).${NC}"
    echo -e "${YELLOW}Use when ctx > vision: long codebases, large RAG. Boot ~5-6 min.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-qwen36-27b-dual-dflash-noviz${NC}"
}

mode_27b_turbo() {
    echo -e "${CYAN}═══ Switching to 27B TurboQuant mode (MTP + 4-stream + 262K) ═══${NC}"
    echo "Starting: Qwen3.6-27B TurboQuant_3bit_nc + MTP n=3 + Genesis v7.14 (TP=2)"
    echo "Port: 8011 | Container: vllm-qwen36-27b-dual-turbo"
    echo "Stopping: Ollama, other 27B variants"
    echo ""
    stop_service ollama
    stop_all_gemma
    stop_comfyui
    stop_27b_dual_mtp
    stop_27b_dual_dflash
    stop_27b_dual_dflash_noviz
    start_27b_dual_turbo
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}27B TurboQuant mode active.${NC} API: http://192.168.86.33:8011"
    echo -e "${YELLOW}58 narr / 69 code TPS per-stream. 262K ctx + 4-stream concurrency (KV pool 1.5M tokens, 9× fp8).${NC}"
    echo -e "${YELLOW}Use for multi-agent serving at long ctx. Vision + tools + thinking + recall all working.${NC}"
    echo -e "${YELLOW}Per-stream slower than fp8 default (P65 cudagraph downgrade); aggregate higher at ≥3 streams.${NC}"
    echo -e "${YELLOW}Boot ~6-8 min (Genesis apply + compile + cudagraph capture).${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-qwen36-27b-dual-turbo${NC}"
}

mode_gemma() {
    echo -e "${CYAN}═══ Switching to Gemma 4 31B MTP mode (default) ═══${NC}"
    echo "Starting: Gemma 4 31B (Intel AutoRound INT4) + MTP n=3 + bf16 KV + 32K + vision (TP=2)"
    echo "Port: 8030 | Container: vllm-gemma-4-31b-mtp"
    echo "Stopping: Ollama, all 27B Qwen variants, other Gemma variants"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_gemma_dflash
    stop_gemma_int8
    stop_gemma_dflash_int8
    stop_gemma_awq
    start_gemma_mtp
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}Gemma 4 31B MTP mode active.${NC} API: http://192.168.86.33:8030"
    echo -e "${YELLOW}109 narr / 141 code TPS (AL 3.05 / 3.99). 32K ctx (BF16 ceiling).${NC}"
    echo -e "${YELLOW}For 262K ctx use 'gemma-int8' (INT8 PTH KV). Boot ~2-3 min.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-gemma-4-31b-mtp${NC}"
}

mode_gemma_dflash() {
    echo -e "${CYAN}═══ Switching to Gemma 4 31B DFlash mode ═══${NC}"
    echo "Starting: Gemma 4 31B + z-lab DFlash drafter (TP=2, :8032)"
    echo ""
    stop_service ollama
    stop_all_27b
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
    echo -e "${CYAN}═══ Switching to Gemma 4 31B INT8-PTH mode (long ctx) ═══${NC}"
    echo "Starting: Gemma 4 31B + INT8 PTH KV + 262K ctx (TP=2, :8032)"
    echo ""
    stop_service ollama
    stop_all_27b
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

mode_gemma_dflash_int8() {
    echo -e "${CYAN}═══ Switching to Gemma 4 31B DFlash + INT8 PTH mode ═══${NC}"
    echo "Starting: Gemma 4 31B + DFlash + INT8 PTH KV (TP=2, :8032). Requires vllm#42102."
    echo ""
    stop_service ollama
    stop_all_27b
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

mode_comfyui() {
    echo -e "${CYAN}═══ Switching to ComfyUI mode (image / video gen) ═══${NC}"
    echo "Starting: ComfyUI :8188"
    echo "Stopping: all GPU-bound LLM serving (Qwen + Gemma)"
    echo ""
    stop_service ollama
    stop_all_27b
    stop_all_gemma
    start_comfyui
    echo ""
    echo -e "${GREEN}ComfyUI mode active.${NC} UI: http://192.168.86.33:8188"
    echo -e "${YELLOW}First boot ~2-3 min while entrypoint clones ComfyUI + custom nodes.${NC}"
    echo -e "${YELLOW}GPU-bound, mutex with vLLM/SGLang. No LiteLLM routing (ComfyUI is non-OpenAI).${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f comfyui${NC}"
}

mode_bigmodel() {
    echo -e "${CYAN}═══ Switching to BIG MODEL mode ═══${NC}"
    echo "Stopping ALL containers to maximize RAM + VRAM..."
    echo ""
    stop_all_27b
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

mode_off() {
    echo -e "${CYAN}═══ Stopping ALL services ═══${NC}"
    stop_all_27b
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
    echo "  27b-turbo          Qwen3.6-27B TurboQuant_3bit_nc + MTP + v7.14 + 262K + 4-stream concurrency (:8011)"
    echo "  27b-dflash         Qwen3.6-27B DFlash + 185K + vision (:8012, 1 stream — fastest with vision)"
    echo "  27b-dflash-noviz   Qwen3.6-27B DFlash + 200K, no vision (:8013, 1 stream — fastest max ctx)"
    echo ""
    echo "  Gemma 4 31B (dual 3090, TP=2):"
    echo "  gemma              Gemma 4 31B + MTP n=3 + bf16 KV + 32K + vision (:8030)"
    echo "  gemma-dflash       Gemma 4 31B + DFlash drafter (:8032)"
    echo "  gemma-int8         Gemma 4 31B + INT8 PTH KV + 262K ctx (:8032)"
    echo "  gemma-dflash-int8  Gemma 4 31B + DFlash + INT8 PTH KV (:8032, requires vllm#42102)"
    echo "  gemma-awq          Gemma 4 31B AWQ-4bit (:8033)"
    echo ""
    echo "  Image / Video Gen (mutex with all LLM modes — GPU-bound):"
    echo "  comfyui            ComfyUI :8188 (FLUX, HunyuanVideo, Wan2.2-Animate)"
    echo ""
    echo "  bigmodel           Stop everything, max RAM+VRAM for one-off llama-server / custom workloads"
    echo "  off                Stop all services"
    echo "  status             Show running services, GPU, RAM, disk, Docker disk"
    echo ""
    echo "  Maintenance:"
    echo "  prune              docker image prune -a (safe — only unreferenced images)"
    echo "  prune-all          + build cache (keep 5 GB) + dangling networks (volumes safe)"
    echo ""
}

case "${1:-}" in
    chat)               mode_chat ;;
    27b)                mode_27b ;;
    27b-turbo)          mode_27b_turbo ;;
    27b-dflash)         mode_27b_dflash ;;
    27b-dflash-noviz)   mode_27b_dflash_noviz ;;
    gemma)              mode_gemma ;;
    gemma-dflash)       mode_gemma_dflash ;;
    gemma-int8)         mode_gemma_int8 ;;
    gemma-dflash-int8)  mode_gemma_dflash_int8 ;;
    gemma-awq)          mode_gemma_awq ;;
    comfyui)            mode_comfyui ;;
    bigmodel)           mode_bigmodel ;;
    off)                mode_off ;;
    status)             show_status ;;
    prune)              mode_prune ;;
    prune-all)          mode_prune_all ;;
    *)                  usage ;;
esac
