#!/usr/bin/env bash
#
# health.sh — operational health check for the running club-3090 server.
#
# Different from verify-full.sh: that one tests *functionality* (does
# tool calling work? does long ctx recall correctly?). This one tells
# you *runtime state*: is the container up, what's the KV pool look
# like, is spec-decode actually firing, any recent errors?
#
# Read this any time you want a quick "is it healthy right now" answer
# before pointing more traffic at the endpoint.
#
# Usage:
#   bash scripts/health.sh
#   bash scripts/health.sh --watch          # refresh every 5s (Ctrl-C to stop)
#   URL=http://localhost:8030 bash scripts/health.sh
#
# Env:
#   URL          API base. Default: http://localhost:8020
#   LOG_LINES    How many log lines to scan for AL/errors. Default: 200
#   WATCH_INTERVAL seconds between refreshes for --watch. Default: 5

set -uo pipefail

URL="${URL:-http://localhost:8020}"
LOG_LINES="${LOG_LINES:-200}"
WATCH_INTERVAL="${WATCH_INTERVAL:-5}"

# Color helpers
if [[ -t 1 ]]; then
  C_OK="\033[0;32m"; C_WARN="\033[1;33m"; C_FAIL="\033[0;31m"; C_DIM="\033[2m"; C_RST="\033[0m"
else
  C_OK=""; C_WARN=""; C_FAIL=""; C_DIM=""; C_RST=""
fi

ok()   { printf "  ${C_OK}✓${C_RST} %s\n" "$1"; }
warn() { printf "  ${C_WARN}⚠${C_RST} %s\n" "$1"; }
fail() { printf "  ${C_FAIL}✗${C_RST} %s\n" "$1"; }
dim()  { printf "  ${C_DIM}%s${C_RST}\n" "$1"; }

probe() {
  echo ""
  echo "club-3090 health check  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "Endpoint: ${URL}"
  echo ""

  # 1. Server reachable
  local models_json status
  if models_json=$(curl -sf --max-time 5 "${URL}/v1/models" 2>/dev/null); then
    ok "API reachable on /v1/models"
  else
    fail "API not reachable at ${URL} — is the container running?"
    echo ""
    echo "  → bash scripts/switch.sh --list   # show available variants"
    echo "  → bash scripts/launch.sh           # boot one with the wizard"
    return 1
  fi

  # Detect served model name + engine
  local model_name engine
  model_name=$(echo "$models_json" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('data',[{}])[0].get('id','unknown'))" 2>/dev/null)
  if echo "$models_json" | grep -qi "owned_by.*llamacpp"; then
    engine="llama.cpp"
  else
    engine="vLLM"
  fi
  ok "Serving model: ${model_name}  (engine: ${engine})"

  # 2. Container — find any qwen36-27b container
  local container container_id status_str started uptime
  container=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^(vllm-qwen36-27b|llama-cpp-qwen36-27b)' | head -1)
  if [[ -z "$container" ]]; then
    warn "No matching container running on this host (server may be on another machine, or running as a host process)"
    container=""
  else
    container_id=$(docker inspect --format '{{.Id}}' "$container" 2>/dev/null | cut -c1-12)
    status_str=$(docker inspect --format '{{.State.Status}}' "$container" 2>/dev/null)
    started=$(docker inspect --format '{{.State.StartedAt}}' "$container" 2>/dev/null)
    if [[ "$status_str" == "running" ]]; then
      uptime=$(python3 -c "
import datetime, sys
t = sys.argv[1].split('.')[0].rstrip('Z') + '+00:00'
diff = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(t)
s = int(diff.total_seconds())
if s < 60: print(f'{s}s')
elif s < 3600: print(f'{s//60}m{s%60:02d}s')
else: print(f'{s//3600}h{(s%3600)//60:02d}m')
" "$started" 2>/dev/null || echo "?")
      ok "Container ${container} (${container_id}) — up ${uptime}"
    else
      fail "Container ${container} status: ${status_str}"
    fi
  fi

  # 3. VRAM
  echo ""
  echo "GPU VRAM:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
               --format=csv,noheader,nounits 2>/dev/null | \
      awk -F', ' '{
        used_pct = $3/$4*100
        bar = ""; for(i=0;i<int(used_pct/5);i++) bar=bar"█"
        for(i=length(bar);i<20;i++) bar=bar"·"
        printf "  GPU %s (%s):  [%s] %5d / %5d MiB (%.0f%%)  util=%s%%  temp=%s°C\n", $1, $2, bar, $3, $4, used_pct, $5, $6
      }'
  else
    dim "(nvidia-smi not available)"
  fi

  # 4. Engine-specific runtime state from logs
  echo ""
  if [[ -n "$container" ]]; then
    local logs
    logs=$(docker logs --tail "${LOG_LINES}" "$container" 2>&1)

    if [[ "$engine" == "vLLM" ]]; then
      # KV cache % from latest "Engine 000" line
      echo "vLLM runtime (last ${LOG_LINES} log lines):"
      local kv_line
      kv_line=$(echo "$logs" | grep -oE 'GPU KV cache usage: [0-9.]+%' | tail -1 || true)
      if [[ -n "$kv_line" ]]; then
        ok "KV cache: ${kv_line#GPU KV cache usage: }"
      else
        dim "KV cache: no recent usage line in logs"
      fi
      # Last 5 SpecDecoding accept rates (AL)
      local al_lines
      al_lines=$(echo "$logs" | grep -oE 'Mean acceptance length: [0-9.]+' | tail -5 || true)
      if [[ -n "$al_lines" ]]; then
        local al_avg
        al_avg=$(echo "$al_lines" | awk '{ s += $4; n++ } END { if (n) printf "%.2f", s/n; else print "n/a" }')
        ok "MTP/Spec-decode: AL last 5 = ${al_avg}  ($(echo "$al_lines" | awk '{print $4}' | tr '\n' ',' | sed 's/,$//'))"
      else
        dim "Spec-decode: no recent acceptance metric in logs (server may be idle)"
      fi
      # Recent throughput
      local tput
      tput=$(echo "$logs" | grep -oE 'Avg generation throughput: [0-9.]+ tokens/s' | tail -1 || true)
      [[ -n "$tput" ]] && ok "Last gen throughput: ${tput#Avg generation throughput: }"
    else
      # llama.cpp
      echo "llama.cpp runtime (last ${LOG_LINES} log lines):"
      local slot_state
      slot_state=$(echo "$logs" | grep -E 'update_slots: all slots are idle|prompt processing|n_tokens =' | tail -3 || true)
      if [[ -n "$slot_state" ]]; then
        ok "Slot activity (recent):"
        echo "$slot_state" | sed 's/^/      /'
      else
        dim "No slot activity in last ${LOG_LINES} lines (server may be idle)"
      fi
      # Decode throughput
      local llcpp_tps
      llcpp_tps=$(echo "$logs" | grep -oE 'eval time =[^,]*\(.*tokens per second\)' | tail -3 || true)
      if [[ -n "$llcpp_tps" ]]; then
        echo "  Recent decode rates:"
        echo "$llcpp_tps" | tail -3 | sed 's/^/      /'
      fi
    fi

    # 5. Recent errors / warnings
    echo ""
    echo "Recent errors / warnings (last ${LOG_LINES} log lines):"
    local errs
    errs=$(echo "$logs" | grep -E 'ERROR|CRITICAL|Traceback|OutOfMemory|CUDA error|Failed' | grep -v 'INFO' | tail -5 || true)
    if [[ -z "$errs" ]]; then
      ok "no errors logged"
    else
      fail "$(echo "$errs" | wc -l) error/warning line(s) — last 5:"
      echo "$errs" | sed 's/^/      /' | head -5
    fi
  fi

  echo ""
  echo "$(date '+%H:%M:%S')  health check complete"
  return 0
}

# --- arg parsing ---
case "${1:-}" in
  -h|--help)
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  --watch)
    while true; do
      clear
      probe || true
      echo ""
      echo "Refresh every ${WATCH_INTERVAL}s — Ctrl-C to stop"
      sleep "$WATCH_INTERVAL"
    done
    ;;
  *)
    probe
    ;;
esac
