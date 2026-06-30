#!/bin/bash
# GPU/RAM Mode Switcher for AI Inference Stack
# Manages Docker containers to avoid GPU/RAM contention on dual-3090 setup
# Location: club-3090/scripts/gpu-mode.sh (symlinked to /usr/local/bin/gpu-mode)

set -e

# Repo root: auto-detected from this script's real location (resolving the
# /usr/local/bin/gpu-mode symlink on the reference rig), so the script is portable to
# any clone. Override with CLUB3090_DIR=... if needed.
CLUB3090_DIR="${CLUB3090_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
COMPOSE_BASE="$CLUB3090_DIR/services"
# ComfyUI/studio paths derive from MODEL_DIR (see services/comfyui/comfyui-paths.sh) so the
# ai-studio scene's compose mounts + missing-model check match wherever the user keeps models.
if [ -f "$COMPOSE_BASE/comfyui/comfyui-paths.sh" ]; then
    # shellcheck disable=SC1091
    . "$COMPOSE_BASE/comfyui/comfyui-paths.sh"
fi
# Post-PR-A (<quant>/ layer): dual composes live under <topology>/<quant>/.
# Point each var at the quant dir so `compose_at` cd's into it — mount-safe,
# the same invocation switch.sh uses (project dir = compose-file dir).
DUAL_27B_DIR="$CLUB3090_DIR/models/qwen3.6-27b/vllm/compose/dual/autoround-int4"
GEMMA_DUAL_DIR="$CLUB3090_DIR/models/gemma-4-31b/vllm/compose/dual/autoround-int4"
# Gemma 4 12B (gemma4_unified arch) — AutoRound INT8 weights + bf16 KV, single-card vLLM on
# the EPHEMERAL vllm/vllm-openai:gemma4-unified arch-preview image (:8038, served gemma-4-12b-int8).
GEMMA_12B_DIR="$CLUB3090_DIR/models/gemma-4-12b/vllm/compose/single/autoround-int8"
# Qwen3.6-35B-A3B MoE (3B active / 35B total) — AutoRound INT4 + fp8 KV, 262K + vision (TP=2).
A3B_DUAL_DIR="$CLUB3090_DIR/models/qwen3.6-35b-a3b/vllm/compose/dual/autoround-int4"
# Qwen3.6-40B-Deckard: uncensored dense 40B, Q6_K GGUF + embedded MTP head,
# layer-split across both cards (llama.cpp). Dual-only — see `gpu-mode deckard`.
DECKARD_DIR="$CLUB3090_DIR/models/qwen3.6-40b-deckard/llama-cpp/compose/dual/piehsoft-q6k"
# DiffusionGemma 26B-A4B — vLLM's first dLLM (TP=2, official :gemma image + 3 fix-mounts).
# Dual-only (both cards); served via the catalog slug vllm/diffusiongemma-dual (the dgemma scene was removed). 🧪
DGEMMA_DIR="$CLUB3090_DIR/models/diffusiongemma-26b-a4b/vllm/compose/dual/fp8"
# (gemma-4-12b chat brain retired from the studio 2026-06-23 — ai-studio uses the qwen
#  director only. Weights kept on disk; gemma stays a core model via `gpu-mode gemma-31b`.)

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

# LAN IP for the URLs printed by the mode banners below — via the shared c3_lan_ip helper
# (prefers LAN over docker bridges; LANIP=-overridable), so it can't drift from setup-ai-
# studio.sh. Falls back to 'localhost' so a fresh clone never prints the rig's IP (#504).
LANIP="${LANIP:-$(c3_lan_ip 2>/dev/null || true)}"
LANIP="${LANIP:-localhost}"

# Standard supporting services living under $CLUB3090_DIR/services.
# Ollama removed 2026-06-22 (dropped from serving 2026-05-10 — Qwen/Gemma route
# through LiteLLM directly; the unused services/ollama/ compose dir was deleted).
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

# Qwen3.6-35B-A3B (MoE, 3B active / 35B total) dual-card vLLM — production slug
# vllm/qwen-35b-a3b-dual (AutoRound INT4 + fp8 KV + 262K + vision, :8051).
start_35b_a3b_dual() {
    printf "  ${GREEN}▲${NC} Starting 35b-a3b-dual..."
    compose_at "$A3B_DUAL_DIR" "up -d" fp8.yml && echo "done" || echo "failed"
}
stop_35b_a3b_dual() {
    printf "  ${RED}▼${NC} Stopping 35b-a3b-dual..."
    compose_at "$A3B_DUAL_DIR" "down" fp8.yml && echo "done" || echo "skipped"
}

# Gemma 4 12B single-card vLLM (gemma4_unified arch-preview image, AutoRound INT8 + MTP n=2).
start_gemma_12b() {
    printf "  ${GREEN}▲${NC} Starting gemma-12b..."
    compose_at "$GEMMA_12B_DIR" "up -d" mtp.yml && echo "done" || echo "failed"
}
stop_gemma_12b() {
    printf "  ${RED}▼${NC} Stopping gemma-12b..."
    compose_at "$GEMMA_12B_DIR" "down" mtp.yml && echo "done" || echo "skipped"
}

# --- ComfyUI (image / video generation) -------------------------------------
# GPU-bound — mutex with all vLLM / SGLang / llama-server LLM serving.
start_comfyui() {
    printf "  ${GREEN}▲${NC} Starting comfyui..."
    # Pin COMFYUI_ROOT into repo-root .env so the compose's `--env-file` mounts the SAME tree the
    # downloads went into (not the /mnt default) on any rig whose MODEL_DIR isn't /mnt — #510/#530.
    type c3_persist_comfy_root >/dev/null 2>&1 && c3_persist_comfy_root || true
    compose_at "$COMPOSE_BASE/comfyui" "up -d" && echo "done" || echo "failed"
}
stop_comfyui() {
    printf "  ${RED}▼${NC} Stopping comfyui..."
    compose_at "$COMPOSE_BASE/comfyui" "down" && echo "done" || echo "skipped"
}

# Video-studio sidecars: the always-on media gallery (:8189) and the prompt
# "director" LLM (:8090, GPU0). See services/studio/ + docs/ai-studio/video.md.
start_studio_gallery() {
    printf "  ${GREEN}▲${NC} Starting studio-gallery (:8189)..."
    compose_at "$COMPOSE_BASE/studio/gallery" "up -d" && echo "done" || echo "failed"
}
# Director placement lever — STUDIO_DIRECTOR_DEVICE (gpu0|gpu1|cpu) from the rig .env, set by
# c3's Settings "Director placement" field (default gpu0). gpu0 = ~4.6 GB on GPU0 (fast craft,
# coexists with the image lanes); gpu1 = GPU1 (NOT during a video render — GPU1 is the DiT donor);
# cpu = frees GPU0 for long single-card video, but craft is ~single-digit tok/s. See video.md.
_director_device() {
    local d=gpu0
    if [ -f "$CLUB3090_DIR/.env" ]; then
        local v
        v=$(grep -E '^STUDIO_DIRECTOR_DEVICE=' "$CLUB3090_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d "\"' ")
        [ -n "$v" ] && d="$v"
    fi
    echo "$d"
}
start_studio_director() {
    local dev ngl cvd gpu label think_args
    dev=$(_director_device)
    # think_args empty on GPU: thinking stays ON (fast there, richer craft; the trace lands in
    # reasoning_content so `content` is still clean). On CPU a full <think> trace dominates the
    # ~14 tok/s latency, so disable it — '--reasoning off' sets the template's enable_thinking=false
    # (this 'Aggressive' fine-tune ignores /no_think and --reasoning-budget 0, but honors this).
    think_args=""
    case "$dev" in
        cpu)  ngl=0;  cvd="";  gpu=0; think_args="--jinja --reasoning off"
              label="CPU — thinking OFF for latency, ~14 tok/s craft" ;;
        gpu1) ngl=99; cvd=1;   gpu=1; label=":8090, GPU1" ;;
        *)    ngl=99; cvd=0;   gpu=0; label=":8090, GPU0" ;;
    esac
    printf "  ${GREEN}▲${NC} Starting studio-director ($label)..."
    compose_at_env "$COMPOSE_BASE/studio/enhancer" "up -d" docker-compose.yml \
        "DIRECTOR_NGL=$ngl" "STUDIO_DIRECTOR_CUDA=$cvd" "STUDIO_DIRECTOR_GPU=$gpu" "DIRECTOR_THINK_ARGS=$think_args" \
        && echo "done" || echo "failed"
}
stop_studio_director() {
    printf "  ${RED}▼${NC} Stopping studio-director..."
    compose_at "$COMPOSE_BASE/studio/enhancer" "down" && echo "done" || echo "skipped"
}
# Free a GPU-resident director before a GPU-heavy LLM scene claims the cards. A CPU-placed
# director uses no GPU, so it STAYS UP across scenes — the always-on uncensored model in OWUI.
_director_evict_if_gpu() {
    [ "$(_director_device)" != "cpu" ] && stop_studio_director
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
# Integrated-voices TTS + audio mixdown (:8192, Kokoro on CPU). The pipe POSTs /narrate after a
# video render to mix a voiceover over the clip's native audio (ducked + normalized). No GPU.
start_studio_tts() {
    printf "  ${GREEN}▲${NC} Starting studio-tts (:8192, Kokoro voices + mixdown, CPU)..."
    compose_at "$COMPOSE_BASE/studio/tts" "up -d --build" && echo "done" || echo "failed"
}
# Premium voice (:8193, Step-Audio-EditX, GPU1, isolated transformers 4.53.3). LAZY: the container
# comes up cheap (~0 GB) so the OWUI voice lane appears as soon as ai-studio starts it; the ~14 GB
# model loads on the FIRST /clone and is freed again by idle-unload (300s) + the pipe's POST /unload
# before a video render. Stopped wherever ComfyUI is (freeing GPU1 for a dual-card LLM scene).
# `--build` picks up server.py edits (layer-cached → fast when unchanged).
start_step_voice() {
    printf "  ${GREEN}▲${NC} Starting step-voice (:8193, lazy — model loads on first use, GPU1)..."
    compose_at "$COMPOSE_BASE/studio/step-voice" "up -d --build" && echo "done" || echo "failed"
}
stop_step_voice() {
    printf "  ${RED}▼${NC} Stopping step-voice..."
    compose_at "$COMPOSE_BASE/studio/step-voice" "down" && echo "done" || echo "skipped"
}

# (start_comfyui_gpu0 + the gemma-4-12b chat brain were retired with the
#  image-studio→ai-studio consolidation 2026-06-23 — ComfyUI now always spans both GPUs.)

# --- Gemma 4 31B dual-card serving variants ---------------------------------
# (start_gemma_mtp + the gemma-mtp/gemma-int8 scenes were removed — only 'gemma'
#  (INT8 PTH KV, 262K, :8032) remains.  stop_gemma_mtp is KEPT: stop_all_gemma
#  (→ mode_off) defensively tears down a stray bf16 gemma container.)
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
    stop_gemma_12b
}

start_deckard() {
    printf "  ${GREEN}▲${NC} Starting deckard-40b..."
    compose_at "$DECKARD_DIR" "up -d" mtp.yml && echo "done" || echo "failed"
}
stop_deckard() {
    printf "  ${RED}▼${NC} Stopping deckard-40b..."
    compose_at "$DECKARD_DIR" "down" mtp.yml && echo "done" || echo "skipped"
}
# (start_diffusiongemma + the diffusiongemma SCENE were removed — the model stays
#  serveable via the catalog slug vllm/diffusiongemma-dual.  stop_diffusiongemma is
#  KEPT: mode_off + the studio modes call it to defensively clear a stray dgemma
#  container off the cards.)
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
    if curl -sf -m 2 http://localhost:8051/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8051/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} 35b-a3b-dual @ :8051    → ${m:-unknown} (MoE 3B/35B + fp8 + 262K + vision)"
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
    if curl -sf -m 2 http://localhost:8038/v1/models >/dev/null 2>&1; then
        local m
        m=$(curl -sf -m 2 http://localhost:8038/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(x['id'] for x in d.get('data',[])))" 2>/dev/null)
        echo -e "  ${GREEN}▶${NC} gemma-12b @ :8038        → ${m:-unknown} (gemma4_unified, INT8 + bf16 KV + MTP, single card)"
    fi
    if curl -sf -m 2 http://localhost:8090/v1/models >/dev/null 2>&1; then
        echo -e "  ${GREEN}▶${NC} studio director @ :8090  → qwen3.5-4b-uncensored (prompt crafter, GPU0, llama.cpp)"
    fi
    if curl -sf -m 2 http://localhost:8188/ >/dev/null 2>&1; then
        echo -e "  ${GREEN}▶${NC} ComfyUI @ :8188          → image/video generation (GPU-bound, mutex with LLM)"
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
      && ! curl -sf -m 2 http://localhost:8038/v1/models >/dev/null 2>&1 \
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
    echo "Starting: Open WebUI + LiteLLM + Qdrant + SearXNG + uncensored director (:8090)"
    echo "The supporting-infra home: launch ANY catalog model with 'switch.sh --owui <variant>'"
    echo "and it appears in OWUI here (web search + document RAG included)."
    echo "Stopping: all GPU-served scene models (Qwen + Gemma)."
    echo ""
    stop_all_27b
    stop_deckard
    stop_all_gemma
    stop_comfyui
    stop_step_voice
    start_service openwebui
    start_service litellm
    start_service qdrant
    start_service searxng
    start_studio_director
    echo ""
    echo -e "${GREEN}Chat mode active.${NC} Open WebUI: http://$LANIP:8080"
    echo -e "${YELLOW}Director placement: $(_director_device) (change in c3 Settings · Director placement).${NC}"
    echo -e "${YELLOW}Plug in catalog models: bash scripts/switch.sh --owui <variant>  (registers it into OWUI here).${NC}"
}

mode_27b() {
    echo -e "${CYAN}═══ Switching to 27B dual-card MTP mode (default) ═══${NC}"
    echo "Starting: Qwen3.6-27B MTP n=3 + fp8 KV + 262K + vision + 2 streams (TP=2)"
    echo "Port: 8010 | Container: vllm-qwen36-27b-dual"
    echo "Stopping: other 27B variants"
    echo ""
    stop_all_gemma
    stop_deckard
    stop_35b_a3b_dual
    stop_comfyui
    stop_step_voice
    _director_evict_if_gpu
    stop_27b_dual_dflash
    stop_27b_dual_dflash_noviz
    stop_27b_dual_turbo
    start_27b_dual_mtp
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}27B dual-card MTP mode active.${NC} API: http://$LANIP:8010"
    echo -e "${YELLOW}Per-stream: 68 narr / 89 code TPS short, 36 TPS @ 100K, 28 TPS @ 200K warm.${NC}"
    echo -e "${YELLOW}2 concurrent streams. KV pool 168K, max concurrency 2.36× at full 262K.${NC}"
    echo -e "${YELLOW}Vision + tools + thinking + 262K ctx all working. Boot ~3-4 min.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-qwen36-27b-dual${NC}"
}

mode_35b_a3b() {
    echo -e "${CYAN}═══ Switching to 35B-A3B dual-card mode (MoE, 262K + vision) ═══${NC}"
    echo "Starting: Qwen3.6-35B-A3B AutoRound INT4 + fp8 KV + 262K + vision (TP=2)"
    echo "Port: 8051 | Container: vllm-qwen36-35b-a3b-dual"
    echo "Stopping: all other GPU models"
    echo ""
    stop_all_27b
    stop_all_gemma
    stop_deckard
    stop_comfyui
    stop_step_voice
    _director_evict_if_gpu
    stop_diffusiongemma
    start_35b_a3b_dual
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}35B-A3B dual-card mode active.${NC} API: http://$LANIP:8051"
    echo -e "${YELLOW}MoE: 3B active / 35B total — ~178/174 TPS, 262K ctx, vision. Boot ~3-4 min.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-qwen36-35b-a3b-dual${NC}"
}

mode_gemma_12b() {
    echo -e "${CYAN}═══ Switching to Gemma 4 12B mode (single-card, gemma4_unified) ═══${NC}"
    echo "Starting: Gemma 4 12B AutoRound INT8 + bf16 KV + MTP n=2 (single card, :8038)"
    echo "Port: 8038 | Container: vllm-gemma-4-12b-int8-mtp"
    echo "Stopping: all other GPU models"
    echo ""
    stop_all_27b
    stop_all_gemma
    stop_deckard
    stop_35b_a3b_dual
    stop_comfyui
    stop_step_voice
    _director_evict_if_gpu
    stop_diffusiongemma
    start_gemma_12b
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}Gemma 4 12B mode active.${NC} API: http://$LANIP:8038"
    echo -e "${YELLOW}gemma4_unified arch-preview image (EPHEMERAL tag — pin a digest before prod). Single card; the other GPU is free.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-gemma-4-12b-int8-mtp${NC}"
}

# (mode_gemma — the bf16 'gemma-mtp' fallback scene — was removed; only the INT8
#  'gemma' scene (mode_gemma_int8) remains.  The bf16 compose is still serveable
#  via its catalog slug if needed.)

mode_gemma_int8() {
    echo -e "${CYAN}═══ Switching to Gemma 4 31B INT8-PTH mode (dual default, long ctx) ═══${NC}"
    echo "Starting: Gemma 4 31B + INT8 PTH KV + 262K ctx (TP=2, :8032)"
    echo ""
    stop_all_27b
    stop_deckard
    stop_35b_a3b_dual
    stop_gemma_12b
    stop_gemma_mtp
    stop_comfyui             # TP=2 needs both cards — clear the studio lanes too
    stop_step_voice
    _director_evict_if_gpu
    stop_gemma_int8          # clean re-switch; the dflash/awq gemma scenes were pruned
    start_gemma_int8
    start_service litellm
    start_service qdrant
    start_service openwebui
    start_service searxng
    echo ""
    echo -e "${GREEN}Gemma 4 31B INT8 PTH mode active.${NC} API: http://$LANIP:8032"
    echo -e "${YELLOW}Tail: sudo docker logs -f vllm-gemma-4-31b-mtp-int8${NC}"
}
mode_deckard() {
    echo -e "${CYAN}═══ Switching to DECKARD-40B mode (uncensored, dual-card) ═══${NC}"
    echo "Starting: Qwen3.6-40B-Deckard Q6_K + MTP n=2 + q8_0 KV + 128K ctx (llama.cpp, :8199)"
    echo "Stopping: all other GPU models (Deckard layer-splits across both cards)"
    echo ""
    stop_all_27b
    stop_all_gemma
    stop_35b_a3b_dual
    stop_comfyui
    stop_step_voice
    _director_evict_if_gpu
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
    echo -e "${GREEN}Deckard-40B mode active.${NC} API: http://$LANIP:8199  (model: deckard-40b)"
    echo -e "${YELLOW}MTP n=2: ~36 narr / 46 code TPS · 128K ctx @ q8_0 KV · uncensored, text-only.${NC}"
    echo -e "${YELLOW}First boot ~1-2 min (31 GB GGUF load + 128K KV alloc across both cards).${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f llama-cpp-deckard-40b${NC}"
}

# (mode_diffusiongemma removed — DiffusionGemma is a niche dLLM; its gpu-mode scene
#  was redundant with the catalog slug vllm/diffusiongemma-dual, which is the way
#  to serve it.  'off' still tears down any running dgemma container.)

# Verify a studio scene's MODELS are on disk BEFORE starting its containers —
# else they boot with no weights (silent failure: the director llama.cpp has no
# GGUF, ComfyUI has no checkpoint).  gpu-mode only STARTS the bundle; the models
# are fetched by setup-image-studio.sh.  This mirrors preflight_compose_deps for
# the LLM composes — it CHECKS + points at the installer, never auto-downloads.
# Both the CLI and the cockpit reach the studio scenes through here, so one gate
# protects both.  Skip: STUDIO_NO_PREFLIGHT=1.
preflight_studio_models() {
    if [ "${STUDIO_NO_PREFLIGHT:-0}" = "1" ]; then return 0; fi
    # Drive the check off the SHARED manifest (scripts/lib/studio-models.tsv) — the SAME
    # file c3 reads — so "what ai-studio needs" is defined once (director · image · video
    # · audio) and the two surfaces can't drift.  director = HARD requirement (no lane can
    # craft without it); every other lane's models missing is a WARNING (studio still boots).
    local manifest="$CLUB3090_DIR/scripts/lib/studio-models.tsv"
    if [ ! -f "$manifest" ]; then
        echo -e "${YELLOW}[preflight] studio manifest missing ($manifest) — skipping model check.${NC}" >&2
        return 0
    fi
    # Roots: weights (director, MODEL_DIR from .env) · comfy (image/video/audio tree).
    local model_dir comfy_models
    model_dir="$(grep -E '^MODEL_DIR=' "$CLUB3090_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2-)"
    model_dir="${model_dir:-/mnt/models/huggingface}"
    comfy_models="${COMFYUI_MODELS_DIR:-/mnt/models/comfyui/models}"
    local director_missing=0 warns=() modality label root rel size installer base
    while IFS=$'\t' read -r modality label root rel size installer; do
        case "$modality" in ''|\#*) continue ;; esac        # skip blank / comment rows
        [ -n "$rel" ] || continue
        if [ "$root" = "weights" ]; then base="$model_dir"; else base="$comfy_models"; fi
        [ -e "$base/$rel" ] && continue                      # model present
        if [ "$modality" = "director" ]; then
            director_missing=1
            echo -e "${RED}[preflight] ai-studio director GGUF not on disk — no lane can craft a prompt:${NC}" >&2
            echo "  - $label  → $base/$rel" >&2
            echo -e "${YELLOW}[preflight] Fix:  bash $installer${NC}" >&2
        else
            warns+=("$modality: $label → bash $installer")
        fi
    done < "$manifest"
    if [ "$director_missing" = "1" ]; then
        echo -e "${YELLOW}[preflight] Skip: STUDIO_NO_PREFLIGHT=1 gpu-mode ai-studio${NC}" >&2
        return 1
    fi
    local w
    for w in "${warns[@]}"; do
        echo -e "${YELLOW}[preflight] lane models absent — $w${NC}" >&2
    done
    return 0
}

mode_ai_studio() {
    preflight_studio_models ai-studio || return 1
    echo -e "${CYAN}═══ Switching to AI-STUDIO mode (image · video · audio · voice) ═══${NC}"
    echo "Starting: ComfyUI :8188 (both GPUs) + director :8090 + gallery/orchestrator/image-shim/tts + Open WebUI"
    echo "Stopping: all GPU-bound LLM serving (Qwen + Gemma + DiffusionGemma)"
    echo ""
    stop_all_27b
    stop_deckard
    stop_35b_a3b_dual
    stop_all_gemma
    stop_diffusiongemma
    start_comfyui
    start_studio_director
    start_studio_gallery
    start_studio_orchestrator
    start_studio_image_shim
    start_studio_tts
    start_step_voice
    start_service openwebui
    start_service litellm
    start_service qdrant
    start_service searxng
    echo ""
    echo -e "${GREEN}AI-studio mode active.${NC} — one scene; pick the lane in Open WebUI."
    echo -e "  Open WebUI:  http://$LANIP:8080   (image · video · music · SFX · voice lanes)"
    echo -e "  Gallery:     http://$LANIP:8189   (all generated media; survives ComfyUI down)"
    echo -e "  ComfyUI:     http://$LANIP:8188   (full node graph / control)"
    echo -e "${YELLOW}First ComfyUI boot can take a few min (clones + node deps). Video DiT splits across both 3090s (DisTorch); image/audio lanes run on GPU0 beside the director.${NC}"
    echo -e "${YELLOW}GPU-mutex with the dual-card LLMs. Premium voice (step-audio-editx) is on-demand on GPU1 — mutually exclusive with an active video render.${NC}"
    echo -e "${YELLOW}Tail: sudo docker logs -f comfyui${NC}"
}

# ('bigmodel' was removed — it was ~redundant with 'off' (both stop everything).
#  Its only useful extra, the freed VRAM/RAM readout, is folded into mode_off
#  below; the cache-drop was marginal (Linux auto-reclaims page cache, and it does
#  nothing for VRAM) and the llama-server hint lives in the docs.)

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
# The rig normally runs both 3090s capped at 250W (quieter / cooler — see the
# systemd unit below). The cap suppresses benchmark TPS, so maintainers need a
# quick way to uncap to the hardware default for a true-TPS bench, then re-cap.
#
# `nvidia-power-cap.service` is the single source of truth for the 250W value
# AND re-applies it on every boot (Type=oneshot, RemainAfterExit=yes, enabled).
# So `power-cap on` *restarts* that unit — `restart` (not `start`) is required:
# the unit is already `active` from boot, and `systemctl start` on an
# already-active RemainAfterExit oneshot is a no-op (it won't re-run ExecStart,
# so the cap wouldn't actually re-apply after a `power-cap off`). `restart`
# stops it (clearing RemainAfterExit) then re-runs both `-pl 250` ExecStart
# lines. `power-cap off` reads each card's Default Power Limit from nvidia-smi
# (370W on GPU 0, 420W on GPU 1 here — they differ, so we never hardcode) and
# applies it. `off` is session-scoped: a reboot OR a driver reload re-applies
# 250W via the service. We never disable the service.
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
    # A numeric action = an explicit CUSTOM wattage applied to both cards (the
    # serve-cockpit power-cap menu's "custom" option).  Validated against each
    # card's [min,max] range; session-scoped like `off` (the boot service still
    # re-applies 250W on reboot/reload).
    if [[ "$action" =~ ^[0-9]+$ ]]; then
        echo -e "${CYAN}═══ Setting custom GPU power cap (${action}W) ═══${NC}"
        local cidx cmin cmax crc=0 capplied=0
        while IFS=',' read -r cidx cmin cmax; do
            cidx="${cidx// /}"; cmin="${cmin%%.*}"; cmin="${cmin// /}"
            cmax="${cmax%%.*}"; cmax="${cmax// /}"
            [ -z "$cidx" ] && continue
            if [ -n "$cmin" ] && [ -n "$cmax" ] && { [ "$action" -lt "$cmin" ] || [ "$action" -gt "$cmax" ]; }; then
                echo -e "  ${RED}✗ GPU ${cidx}: ${action}W out of range [${cmin},${cmax}]W${NC}" >&2
                crc=1; continue
            fi
            echo "  Setting GPU ${cidx} → ${action} W..."
            if ! sudo nvidia-smi -i "$cidx" -pl "$action" >/dev/null 2>&1; then
                echo -e "  ${RED}✗ Failed to set GPU ${cidx} to ${action} W${NC} (sudo? driver?)." >&2
                crc=1
            else
                capplied=1
            fi
        done < <(nvidia-smi --query-gpu=index,power.min_limit,power.max_limit \
            --format=csv,noheader,nounits 2>/dev/null)
        if [ "$capplied" -eq 0 ]; then
            echo -e "${RED}✗ No GPUs updated.${NC} Check the value is within range: nvidia-smi -q -d POWER" >&2
            exit 1
        fi
        echo -e "${GREEN}Custom cap ${action}W applied.${NC} ${YELLOW}Session-scoped — a reboot or driver"
        echo -e "reload re-applies 250W via ${POWER_CAP_SERVICE}.${NC}"
        powercap_echo_enforced
        [ "$crc" -eq 0 ] || exit 1
        return
    fi
    case "$action" in
        on)
            echo -e "${CYAN}═══ Re-applying GPU power cap (250W) ═══${NC}"
            echo "Restarting ${POWER_CAP_SERVICE} (the boot-time 250W enforcer)."
            # restart, not start — the unit is already active from boot, so
            # `start` is a no-op on a RemainAfterExit oneshot (won't re-run -pl).
            if sudo systemctl restart "$POWER_CAP_SERVICE" 2>/dev/null; then
                echo -e "${GREEN}Power cap re-applied via systemd.${NC}"
            else
                # Fallback: service missing/disabled — apply 250W directly.
                echo -e "${YELLOW}systemctl restart failed; falling back to direct nvidia-smi -pl 250.${NC}" >&2
                if ! { sudo nvidia-smi -i 0 -pl 250 && sudo nvidia-smi -i 1 -pl 250; }; then
                    echo -e "${RED}✗ Failed to set 250W cap.${NC} Check sudo + driver state with: nvidia-smi -q -d POWER" >&2
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
            echo -e "reload re-applies 250W via ${POWER_CAP_SERVICE}. Run 'gpu-mode power-cap on' to re-cap now.${NC}"
            powercap_echo_enforced
            [ "$rc" -eq 0 ] || exit 1
            ;;
        status)
            powercap_show
            ;;
        *)
            echo -e "${RED}Unknown power-cap action:${NC} $action" >&2
            echo "Usage: gpu-mode power-cap <on|off|status|WATTS>" >&2
            exit 1
            ;;
    esac
}

mode_off() {
    echo -e "${CYAN}═══ Stopping ALL services ═══${NC}"
    stop_all_27b
    stop_deckard
    stop_35b_a3b_dual
    stop_diffusiongemma
    stop_all_gemma
    stop_comfyui
    stop_step_voice
    stop_studio_director     # full off stops even a CPU/always-on director
    stop_estate
    for svc in "${SERVICES[@]}"; do
        stop_service "$svc"
    done
    echo ""
    echo -e "${GREEN}All services stopped.${NC}"
    # Freed-resource readout (folded in from the removed 'bigmodel' scene) — handy
    # before a one-off manual workload (a custom llama-server GGUF, an experiment).
    echo ""
    echo -e "${CYAN}═══ Free resources ═══${NC}"
    echo -e "VRAM:"
    nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader 2>/dev/null
    echo -e "RAM:"
    free -h | grep Mem | awk '{print "  Free: "$4" / Total: "$2}'
}

# --- Scene catalog (`--list-modes [--json]`) --------------------------------
# Machine-readable mirror of the dispatch case + what each mode_* function
# actually starts. Each row: name (the dispatch keyword), group, description,
# services (logical service/container names the mode brings up), ports, gpus.
# Groups follow the contract: serving / studio / ops. This is hand-maintained
# alongside the dispatch case below — keep them in lockstep when adding a mode.
#
# Format is a TSV heredoc (name \t group \t description \t services \t ports \t
# gpus) so the data lives in one readable place; we render it to a plain table
# or to JSON via python3 (already a hard dep of show_status). The plain render
# is the default; --json emits the [{name,group,description,services,ports,gpus}]
# array the contract specifies. services/ports are comma-joined in the TSV and
# split into JSON arrays; gpus is the human GPU-usage note.
list_modes_data() {
    # name<TAB>group<TAB>description<TAB>services<TAB>ports<TAB>gpus
    cat <<'TSV'
chat	ops	Open WebUI + LiteLLM + Qdrant + SearXNG + uncensored director — supporting-infra home for catalog models	openwebui,litellm,qdrant,searxng,studio-director	8080,4000,8090	none
qwen27b	models	Qwen3.6-27B MTP n=3 + fp8 KV + 262K + vision (TP=2) — default	vllm-qwen36-27b-dual,litellm,qdrant,openwebui,searxng	8010,8080,4000	both
qwen35b-a3b	models	Qwen3.6-35B-A3B MoE (3B active / 35B total) AutoRound INT4 + fp8 KV + 262K + vision (TP=2)	vllm-qwen36-35b-a3b-dual,litellm,qdrant,openwebui,searxng	8051,8080,4000	both
gemma-31b	models	Gemma 4 31B INT8 PTH KV + 262K + vision (TP=2) — dual default	vllm-gemma-4-31b-mtp-int8,litellm,qdrant,openwebui,searxng	8032,8080,4000	both
gemma12b	models	Gemma 4 12B AutoRound INT8 + bf16 KV + MTP n=2 (gemma4_unified arch-preview, single-card)	vllm-gemma-4-12b-int8-mtp,litellm,qdrant,openwebui,searxng	8038,8080,4000	1
deckard	models	Qwen3.6-40B-Deckard Q6_K + MTP n=2 + q8_0 KV + 128K (llama.cpp, dual)	llama-cpp-deckard-40b,litellm,qdrant,openwebui,searxng	8199,8080,4000	both
ai-studio	studio	image · video · audio · voice — ComfyUI both GPUs + qwen director + sidecars + Open WebUI (pick the lane in OWUI)	comfyui,studio-director,studio-gallery,studio-orchestrator,studio-image-shim,studio-tts,studio-step-voice,openwebui,litellm,qdrant,searxng	8188,8090,8189,8190,8191,8192,8193,8080,4000,6333	both
off	ops	Stop all services	all-stopped		none
power-cap	ops	GPU power-cap controls (on/off/status; both 3090s, 250W default cap)			both
prune	ops	docker image prune -a (safe — only unreferenced images)			none
prune-all	ops	+ build cache (keep 5 GB) + dangling networks (volumes safe)			none
TSV
}

list_modes() {
    local as_json=0
    [[ "${1:-}" == "--json" ]] && as_json=1
    if [[ "$as_json" -eq 1 ]]; then
        list_modes_data | python3 -c '
import sys, json
rows = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    parts = line.split("\t")
    # pad to 6 fields so trailing-empty columns survive split
    while len(parts) < 6:
        parts.append("")
    name, group, desc, services, ports, gpus = parts[:6]
    rows.append({
        "name": name,
        "group": group,
        "description": desc,
        "services": [s for s in services.split(",") if s],
        "ports": [p for p in ports.split(",") if p],
        "gpus": gpus,
    })
print(json.dumps(rows, indent=2))
'
    else
        echo ""
        echo -e "${CYAN}═══ Scene Catalog ═══${NC}"
        local cur_group=""
        while IFS=$'\t' read -r name group desc services ports gpus; do
            [[ -z "$name" ]] && continue
            if [[ "$group" != "$cur_group" ]]; then
                cur_group="$group"
                echo ""
                echo -e "${CYAN}[$group]${NC}"
            fi
            printf "  %-16s %s\n" "$name" "$desc"
        done < <(list_modes_data)
        echo ""
        echo -e "${YELLOW}--list-modes --json for the machine-readable catalog.${NC}"
    fi
}

usage() {
    echo ""
    echo -e "${CYAN}GPU Mode Switcher${NC} — AI Inference Stack Manager"
    echo ""
    echo "Usage: gpu-mode <mode>"
    echo ""
    echo "Modes:"
    echo "  chat               Open WebUI + LiteLLM + Qdrant + SearXNG + uncensored director (:8090) — catalog-model home"
    echo ""
    echo "  Qwen 3.6 27B (dual 3090, TP=2):"
    echo "  qwen27b            ⭐ DEFAULT — Qwen3.6-27B MTP + fp8 + 262K + vision + 2 streams (:8010) — alias: 27b"
    echo ""
    echo "  Qwen 3.6 35B-A3B MoE (dual 3090, TP=2):"
    echo "  qwen35b-a3b        AutoRound INT4 + fp8 KV + 262K + vision (:8051) — alias: 35b-a3b, a3b, 35b"
    echo ""
    echo "  Gemma 4 31B (dual 3090, TP=2):"
    echo "  gemma-31b          ⭐ DEFAULT — Gemma 4 31B INT8 PTH KV + 262K + vision (:8032) — alias: gemma"
    echo ""
    echo "  Gemma 4 12B (single 3090, gemma4_unified arch-preview):"
    echo "  gemma12b           AutoRound INT8 + bf16 KV + MTP n=2 (:8038, served gemma-4-12b-int8) — alias: gemma-12b"
    echo ""
    echo "  Qwen 3.6 40B Deckard (uncensored, dual 3090, llama.cpp):"
    echo "  deckard            Q6_K + MTP n=2 + q8_0 KV + 128K ctx (:8199) — text-only, both cards"
    echo ""
    echo "  AI Studio (image · video · audio · voice — one scene, pick the lane in OWUI):"
    echo "  ai-studio          ⭐ ComfyUI both GPUs + qwen director (:8090) + gallery/orchestrator/tts"
    echo "                     + Open WebUI (:8080) — image/video/music/SFX/voice lanes (alias: aistudio)"
    echo ""
    echo "  off                Stop all services"
    echo "  status             Show running services, GPU, RAM, disk, Docker disk"
    echo ""
    echo "  GPU power cap (both 3090s; normally capped at 250W for quiet/cool operation):"
    echo "  power-cap on       Re-apply the 250W cap (via nvidia-power-cap.service)"
    echo "  power-cap off      Uncap to hardware default for a true-TPS bench"
    echo "                     (session-scoped — a reboot / driver reload re-caps at 250W)"
    echo "  power-cap <WATTS>  Apply a custom cap to both cards (e.g. 'power-cap 280';"
    echo "                     validated against each card's [min,max]; session-scoped)"
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
    qwen27b|27b)        mode_27b ;;
    qwen35b-a3b|35b-a3b|a3b|35b)  mode_35b_a3b ;;
    gemma-31b|gemma)    mode_gemma_int8 ;;
    gemma12b|gemma-12b) mode_gemma_12b ;;
    deckard)            mode_deckard ;;
    ai-studio|aistudio)       mode_ai_studio ;;
    off)                mode_off ;;
    status)             show_status ;;
    power-cap|powercap) mode_powercap "${2:-status}" ;;
    prune)              mode_prune ;;
    prune-all)          mode_prune_all ;;
    --list-modes)       list_modes "${2:-}" ;;
    *)                  usage ;;
esac
