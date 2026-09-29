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
#   URL          API base. Default: the running inference container's port, found
#                the way verify.sh / bench.sh find it (preflight_autodetect_endpoint:
#                any container publishing an engine port); with nothing running, the
#                registry-derived port of qwen3.6-27b's default slug (currently :8020).
#   CONTAINER    Target a specific named container instead of auto-matching.
#                Default: the container autodetect found, else auto-match any
#                recognized engine-prefix container (vllm-/llama-cpp-/ik-llama-/
#                sglang-/beellama-). PREFLIGHT_NO_AUTODETECT=1 skips the autodetect.
#   LOG_LINES    How many log lines to scan for AL/errors. Default: 200
#   WATCH_INTERVAL seconds between refreshes for --watch. Default: 5

set -uo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# The running container + its port, exactly as verify.sh finds them (URL= and
# CONTAINER= still win). Without this, health.sh probed qwen3.6-27b's default port
# and reported "API not reachable" while another slug was serving on its own port.
if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
  # shellcheck source=preflight.sh
  source "${ROOT_DIR}/scripts/preflight.sh"
  preflight_autodetect_endpoint
fi
# With nothing running, the default endpoint follows the registry's curated
# DEFAULTS walk for qwen3.6-27b instead of a hand-maintained :8020/:8010 literal
# that drifts from the catalog.
# The trailing literal is only a last resort when the registry can't be consulted.
_DEFAULT_ENDPOINT_PORT=""
if [[ -f "${ROOT_DIR}/scripts/lib/registry-lookup.sh" ]]; then
  # shellcheck source=lib/registry-lookup.sh
  source "${ROOT_DIR}/scripts/lib/registry-lookup.sh"
  REGISTRY_LOOKUP_ROOT="${ROOT_DIR}"
  _DEFAULT_ENDPOINT_PORT="$(registry_lookup_default_port qwen3.6-27b 2>/dev/null || true)"
fi
URL="${URL:-http://localhost:${_DEFAULT_ENDPOINT_PORT:-8020}}"
CONTAINER="${CONTAINER:-}"
LOG_LINES="${LOG_LINES:-200}"
WATCH_INTERVAL="${WATCH_INTERVAL:-5}"

# Container matcher used when CONTAINER= is unset: any inference container the
# REGISTRY knows, plus a prefix arm for estate/ad-hoc instances that rename their
# container. Registry-derived so a new engine is covered the moment its slug
# lands.
#
# ⚠️ This was a hand-written list — '^(vllm-|llama-cpp-|ik-llama-|sglang-|beellama-)'
# — and it had already fallen behind: exl3 serves as `tabbyapi-*`, so on a rig
# running it health.sh reported no engine container over a healthy server. The
# same list in report.sh was missing sglang- as well. #281 fixed exactly this in
# switch.sh by deriving the set from the registry; the other copies were never
# converted. See scripts/lib/club-containers.sh.
# shellcheck source=lib/club-containers.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/club-containers.sh"
# Which engine it is: the decision is engine-kind.sh's (#1282); health.sh only
# gathers the evidence (the container's image and name, /v1/models owned_by).
# shellcheck source=lib/engine-kind.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/engine-kind.sh"
ENGINE_PREFIX_RE="$(club_container_re)"

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

  # The container — CONTAINER= (set by the user, or by the autodetect above) if
  # set, else any recognized engine container. Found before the engine is named,
  # because its image is the best evidence of which engine this is.
  local container container_id status_str started uptime
  if [[ -n "$CONTAINER" ]]; then
    # Exact-name match for the user-specified container.
    container=$(docker ps --format '{{.Names}}' 2>/dev/null | command grep -Fx "$CONTAINER" | head -1)
  else
    container=$(docker ps --format '{{.Names}}' 2>/dev/null | command grep -E "$ENGINE_PREFIX_RE" | head -1)
  fi

  # Detect served model name + engine (image → container name → owned_by).
  local model_name kind="unknown" engine
  model_name=$(echo "$models_json" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('data',[{}])[0].get('id','unknown'))" 2>/dev/null)
  if [[ -n "$container" ]]; then
    kind=$(engine_kind_from_image "$(docker inspect --format '{{.Config.Image}}' "$container" 2>/dev/null)")
    [[ "$kind" == unknown ]] && kind=$(engine_kind_from_container "$container")
  fi
  if [[ "$kind" == unknown ]]; then
    kind=$(engine_kind_from_owned_by "$(echo "$models_json" | python3 -c "import sys,json;d=json.load(sys.stdin);print((d.get('data') or [{}])[0].get('owned_by',''))" 2>/dev/null)")
  fi
  case "$kind" in
    vllm)      engine="vLLM" ;;
    llamacpp)  engine="llama.cpp" ;;
    sglang)    engine="SGLang" ;;
    exllamav3) engine="ExLlamaV3" ;;
    *)         engine="unknown" ;;
  esac
  ok "Serving model: ${model_name}  (engine: ${engine})"

  # 2. Container
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
      kv_line=$(echo "$logs" | command grep -oE 'GPU KV cache usage: [0-9.]+%' | tail -1 || true)
      if [[ -n "$kv_line" ]]; then
        ok "KV cache: ${kv_line#GPU KV cache usage: }"
      else
        dim "KV cache: no recent usage line in logs"
      fi
      # Last 5 SpecDecoding accept rates (AL)
      local al_lines
      al_lines=$(echo "$logs" | command grep -oE 'Mean acceptance length: [0-9.]+' | tail -5 || true)
      if [[ -n "$al_lines" ]]; then
        local al_avg
        al_avg=$(echo "$al_lines" | awk '{ s += $4; n++ } END { if (n) printf "%.2f", s/n; else print "n/a" }')
        ok "MTP/Spec-decode: AL last 5 = ${al_avg}  ($(echo "$al_lines" | awk '{print $4}' | tr '\n' ',' | sed 's/,$//'))"
      else
        dim "Spec-decode: no recent acceptance metric in logs (server may be idle)"
      fi
      # Recent throughput
      local tput
      tput=$(echo "$logs" | command grep -oE 'Avg generation throughput: [0-9.]+ tokens/s' | tail -1 || true)
      [[ -n "$tput" ]] && ok "Last gen throughput: ${tput#Avg generation throughput: }"
    elif [[ "$engine" == "SGLang" ]]; then
      # SGLang logs one line per decode / prefill batch, e.g.
      #   Decode batch, #running-req: 1, #full token: 0, full token usage: 0.00, …,
      #   accept len: 1.37, accept rate: 0.09, cuda graph: True,
      #   gen throughput (token/s): 1.35, #queue-req: 0
      echo "SGLang runtime (last ${LOG_LINES} log lines):"
      local usage decode_lines
      usage=$(echo "$logs" | command grep -oE '(full )?token usage: [0-9.]+' | tail -1 | command grep -oE '[0-9.]+$' || true)
      if [[ -n "$usage" ]]; then
        ok "KV cache: $(awk -v u="$usage" 'BEGIN { printf "%.0f", u * 100 }')%  (token usage ${usage})"
      else
        dim "KV cache: no recent batch line in logs"
      fi
      decode_lines=$(echo "$logs" | command grep -E 'Decode batch' || true)
      local al_lines
      al_lines=$(echo "$decode_lines" | command grep -oE 'accept len: [0-9.]+' | tail -5 || true)
      if [[ -n "$al_lines" ]]; then
        ok "MTP/Spec-decode: accept len last 5 = $(echo "$al_lines" | awk '{ s += $3; n++ } END { if (n) printf "%.2f", s/n }')  ($(echo "$al_lines" | awk '{print $3}' | tr '\n' ',' | sed 's/,$//'))"
      else
        dim "Spec-decode: no recent decode batch in logs (server idle, or no drafter)"
      fi
      local tput reqs
      tput=$(echo "$decode_lines" | command grep -oE 'gen throughput \(token/s\): [0-9.]+' | tail -1 | command grep -oE '[0-9.]+$' || true)
      [[ -n "$tput" ]] && ok "Last gen throughput: ${tput} tokens/s"
      reqs=$(echo "$logs" | command grep -E '(Decode|Prefill) batch' | tail -1 | command grep -oE '#(running|queue)-req: [0-9]+' | tr '\n' ' ' || true)
      [[ -n "$reqs" ]] && dim "Requests at the last batch: ${reqs% }"
    elif [[ "$engine" == "ExLlamaV3" || "$engine" == "unknown" ]]; then
      echo "${engine} runtime:"
      dim "runtime details aren't parsed for this engine — see: docker logs --tail ${LOG_LINES} ${container}"
    else
      # llama.cpp
      echo "llama.cpp runtime (last ${LOG_LINES} log lines):"
      local slot_state
      slot_state=$(echo "$logs" | command grep -E 'update_slots: all slots are idle|prompt processing|n_tokens =' | tail -3 || true)
      if [[ -n "$slot_state" ]]; then
        ok "Slot activity (recent):"
        echo "$slot_state" | sed 's/^/      /'
      else
        dim "No slot activity in last ${LOG_LINES} lines (server may be idle)"
      fi
      # Decode throughput
      local llcpp_tps
      llcpp_tps=$(echo "$logs" | command grep -oE 'eval time =[^,]*\(.*tokens per second\)' | tail -3 || true)
      if [[ -n "$llcpp_tps" ]]; then
        echo "  Recent decode rates:"
        echo "$llcpp_tps" | tail -3 | sed 's/^/      /'
      fi
    fi

    # 5. Recent errors / warnings
    echo ""
    echo "Recent errors / warnings (last ${LOG_LINES} log lines):"
    local errs
    errs=$(echo "$logs" | command grep -E 'ERROR|CRITICAL|Traceback|OutOfMemory|CUDA error|Failed' | command grep -v 'INFO' | tail -5 || true)
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
