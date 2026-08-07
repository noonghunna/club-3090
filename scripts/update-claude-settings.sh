#!/usr/bin/env bash
# Auto-detect the running club-3090 endpoint and update Claude Code settings.
#
# Used as a `before-turn` hook in ~/.claude/settings.json so Claude Code
# always points at the correct port and model name, regardless of which
# compose you switched to.
#
# Usage:
#   bash scripts/update-claude-settings.sh
#
# Exit 0 on success (settings updated or already correct).
# Exit 0 with no-op message if no club-3090 container is running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SETTINGS_FILE="${HOME}/.claude/settings.json"

# --- Find the running club-3090 container ---------------------------------
# Look for containers with names matching our convention (vllm-*, llamacpp-*,
# sglang-*, beellama-*, ik-llama-*) that are currently running.
CONTAINER_ID=""
CONTAINER_NAME=""

while IFS=' ' read -r cid cname; do
  if [[ -n "$cid" ]]; then
    CONTAINER_ID="$cid"
    CONTAINER_NAME="$cname"
    break
  fi
done < <(docker ps --filter 'status=running' \
  --format '{{.ID}} {{.Names}}' 2>/dev/null \
  | grep -E '(vllm-|llamacpp-|sglang-|beellama-|ik-llama-|llama-cpp-)' \
  | head -1 || true)

if [[ -z "$CONTAINER_ID" ]]; then
  echo "[claude-settings] No club-3090 container running. Settings unchanged."
  exit 0
fi

# --- Extract the host port ------------------------------------------------
# Port binding format from docker ps: "0.0.0.0:8077->8000/tcp"
PORT_BINDING=$(docker ps --filter "id=${CONTAINER_ID}" \
  --format '{{.Ports}}' 2>/dev/null)

HOST_PORT=$(echo "$PORT_BINDING" | sed -n 's/^[^:]*:\([0-9]*\)->.*/\1/p' | head -1)

if [[ -z "$HOST_PORT" ]]; then
  echo "[claude-settings] Could not parse port from binding: ${PORT_BINDING}"
  exit 0
fi

# --- Extract the host IP --------------------------------------------------
HOST_IP=$(echo "$PORT_BINDING" | sed -n 's/^\([0-9.]*\):.*/\1/p' | head -1)
if [[ -z "$HOST_IP" || "$HOST_IP" == "0.0.0.0" ]]; then
  HOST_IP="localhost"
fi

# --- Detect the model name ------------------------------------------------
# Query the /v1/models endpoint via the host IP:port and pick the most
# specific model ID (longest). vLLM registers multiple aliases (e.g.,
# qwen3.6-27b, qwen3.6-27b-autoround, qwen3.6-27b-nvfp4).
#
# Retry with backoff because the engine may still be booting when the
# SessionStart hook fires. Don't write settings until the endpoint is
# healthy — leave last-good values intact.
MODEL_NAME=""

BASE_URL="http://${HOST_IP}:${HOST_PORT}"

# Retry loop: 5 attempts, 2s / 3s / 4s / 5s / 5s backoff (max ~19s total)
for i in 1 2 3 4 5; do
  if [[ $i -gt 1 ]]; then
    sleep $((i + 1))
  fi
  RESPONSE=$(curl -s --max-time 5 "${BASE_URL}/v1/models" 2>/dev/null || true)
  if [[ -n "$RESPONSE" ]]; then
    MODEL_NAME=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [m['id'] for m in data.get('data', []) if 'inst' not in m['id']]
    if models:
        print(max(models, key=len))
    else:
        print(data.get('data', [{}])[0].get('id', ''))
except:
    print('')
" 2>/dev/null || true)
    if [[ -n "$MODEL_NAME" ]]; then
      break
    fi
  fi
done

# Fallback: extract model name from container command line
if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME=$(docker inspect --format '{{.Config.Cmd}}' "$CONTAINER_ID" 2>/dev/null \
    | sed -n 's/.*served-model-name[[:space:]]*\+\([^- ]\+\).*/\1/p' | head -1 || true)
fi

# Fallback: derive from container name
if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME=$(echo "$CONTAINER_NAME" | sed 's/^\(vllm\|llamacpp\|sglang\|beellama\|ik-llama\|llama-cpp\)-//' \
    | sed 's/qwen36/qwen3.6/g' \
    | sed 's/gemma4/gemma-4/g')
fi

if [[ -z "$MODEL_NAME" ]]; then
  echo "[claude-settings] Could not detect model name for ${CONTAINER_NAME}. Settings unchanged."
  exit 0
fi

# --- Update settings.json -------------------------------------------------
python3 "$ROOT_DIR/scripts/update-claude-settings.py" "$SETTINGS_FILE" "$BASE_URL" "$MODEL_NAME"

exit 0