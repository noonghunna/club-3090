#!/usr/bin/env bash
# served-slots.sh — the ONE served-slot-count detector (#1577). Sourced by
# rebench-full.sh (its concurrency step) and concurrency-probe.sh (#818).
#
# rebench-full.sh used to carry a private, narrower copy of the probe's detector
# with no SGLang sources, so an SGLang endpoint reached by --url fell back to the
# N=1 control while concurrency-probe.sh would have found its slot count.
#
#   served_slots URL [CONTAINER]
#     prints "<N><TAB><source>" for the first source that answers, nothing if none.
#     Order (the probe's #818 order — the container's own flags first, they are
#     what the engine was started with):
#       1. container --max-num-seqs          (vLLM)
#       2. container -np                     (llama.cpp)
#       3. container --max-running-requests  (SGLang)
#       4. GET /get_server_info max_running_requests  (SGLang, set via env)
#       5. GET /props total_slots            (llama.cpp family)
#     vLLM reports no slot count over HTTP outside dev mode (/server_info needs
#     VLLM_SERVER_DEV_MODE=1), so a vLLM endpoint reached by --url with no container
#     prints nothing. Callers must SAY so rather than run a short ladder silently.
#
#   concurrency_slots_marker FILE SLOTS SOURCE RUNGS OVERRIDE
#     writes the record rebench-full.sh's summary and rebench-report.py read.
#   concurrency_slots_note FILE
#     prints the one-line warning for a run whose N>1 rungs were skipped, or nothing.

export PYTHONUTF8="${PYTHONUTF8:-1}"   # #779: python3 below must not decode with the locale codec

_slots_container_cmd() {
  local c="${1:-}"
  [[ -n "$c" && "$c" != "none" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  docker inspect "$c" --format '{{join .Config.Cmd " "}}' 2>/dev/null || true
}

# _slots_flag PATTERN CMD — the integer after a flag (space- or =-separated).
_slots_flag() {
  command grep -oE -- "$1[ =]+[0-9]+" <<<"$2" | command grep -oE '[0-9]+$' | head -1 || true
}

# _slots_json_int URL KEY — a top-level integer (booleans don't count) or nothing.
_slots_json_int() {
  curl -s -m 3 "$1" 2>/dev/null | python3 -c '
import json, sys
try:
    v = json.load(sys.stdin).get(sys.argv[1], "")
except Exception:
    v = ""
print(v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else "")
' "$2" 2>/dev/null || true
}

served_slots() {
  local url="${1:-}" container="${2:-}" cmd n
  cmd="$(_slots_container_cmd "$container")"
  if [[ -n "$cmd" ]]; then
    n="$(_slots_flag 'max-num-seqs' "$cmd")"
    [[ -n "$n" ]] && { printf '%s\t%s\n' "$n" "container max-num-seqs"; return 0; }
    n="$(_slots_flag '\-np' "$cmd")"
    [[ -n "$n" ]] && { printf '%s\t%s\n' "$n" "container -np"; return 0; }
    n="$(_slots_flag 'max-running-requests' "$cmd")"
    [[ -n "$n" ]] && { printf '%s\t%s\n' "$n" "container max-running-requests"; return 0; }
  fi
  [[ -n "$url" ]] || return 0
  n="$(_slots_json_int "${url}/get_server_info" max_running_requests)"
  [[ -n "$n" ]] && { printf '%s\t%s\n' "$n" "server max_running_requests"; return 0; }
  n="$(_slots_json_int "${url}/props" total_slots)"
  [[ -n "$n" ]] && { printf '%s\t%s\n' "$n" "server /props total_slots"; return 0; }
  return 0
}

concurrency_slots_marker() {
  local file="$1" slots="$2" source="$3" rungs="$4" override="$5"
  {
    echo "slots=${slots}"
    echo "source=${source}"
    echo "rungs=${rungs}"
    echo "override=${override}"
  } > "$file"
}

concurrency_slots_note() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  command grep -qx 'slots=undetected' "$file" || return 0
  command grep -qx 'override=0' "$file" || return 0
  echo "concurrency: N=1 only — the served slot count could not be read (vLLM over --url exposes none); the N=2/4 rungs did not run. Re-run with CONCURRENCY_RUNGS='1 2 4' (rungs above the server's real slot count measure queue wait, not concurrency)."
}
