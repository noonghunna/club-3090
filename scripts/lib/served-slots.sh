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
#       3b. container log "non-default args: {... 'max_num_seqs': N ...}"  (a vLLM-family engine that
#          starts vLLM from its own launcher: HyperQwen's MAX_SEQS, bucko's entrypoint)
#       4. container log "N sequence slots"  (QEngine: an env setting, QE_MAX_SEQS on the 27B and
#          FLASH_MAX_SEQS on Flash-Next, and none over HTTP; only its boot log says what it set, #1846)
#       5. GET /get_server_info max_running_requests  (SGLang, set via env)
#       6. GET /props total_slots            (llama.cpp family)
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

# _slots_vllm_log CONTAINER — the max_num_seqs a vLLM-family container's own start-up dump states, or nothing.
# A vLLM derivative that starts vLLM from its own launcher (HyperQwen from MAX_SEQS, bucko from a bash -c
# entrypoint) leaves --max-num-seqs out of the container's Cmd, and vLLM reports no slot count over HTTP, so
# the probe refused them. vLLM logs every non-default engine argument at start-up: "non-default args: {...
# 'max_num_seqs': 8, ...}". Read only for a container the engine resolver calls vllm; the last such line wins
# (a restarted container keeps its earlier boots' lines). A vLLM left at its default prints none: nothing.
_slots_vllm_log() {
  local c="${1:-}" kind
  [[ -n "$c" && "$c" != "none" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  # shellcheck source=engine-kind.sh
  declare -F engine_kind_from_container >/dev/null || source "$(dirname "${BASH_SOURCE[0]}")/engine-kind.sh"
  kind="$(engine_kind_from_container "$c")"
  [[ "$kind" == unknown ]] && kind="$(engine_kind_from_image "$(docker inspect "$c" --format '{{.Config.Image}}' 2>/dev/null || true)")"
  [[ "$kind" == vllm ]] || return 0
  docker logs "$c" 2>&1 | command grep -oE "non-default args: .*'max_num_seqs': [0-9]+" | tail -1 \
    | command grep -oE '[0-9]+$' || true
}

# _slots_qengine_log CONTAINER — the slot count a QEngine container's boot log states, or nothing.
# The 27B engine says "KV pool: N tokens shared by 32 sequence slots (max context …)", Flash-Next
# "[qflash] 16 sequence slots over a pool …". Read only for a QEngine image: the words are its own. The
# last boot's line wins: a restarted container keeps its earlier boots' lines.
_slots_qengine_log() {
  local c="${1:-}"
  [[ -n "$c" && "$c" != "none" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  [[ "$(docker inspect "$c" --format '{{.Config.Image}}' 2>/dev/null || true)" == *qengine* ]] || return 0
  docker logs "$c" 2>&1 | command grep -oE '[0-9]+ sequence slots' | tail -1 | command grep -oE '^[0-9]+' || true
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
  # Outside the block above: these engines' composes set an entrypoint and no command, so their Cmd is empty
  n="$(_slots_vllm_log "$container")"
  [[ -n "$n" ]] && { printf '%s\t%s\n' "$n" "container log (vLLM non-default args)"; return 0; }
  n="$(_slots_qengine_log "$container")"
  [[ -n "$n" ]] && { printf '%s\t%s\n' "$n" "container log (QEngine)"; return 0; }
  [[ -n "$url" ]] || return 0
  n="$(_slots_json_int "${url}/get_server_info" max_running_requests)"
  [[ -n "$n" ]] && { printf '%s\t%s\n' "$n" "server max_running_requests"; return 0; }
  n="$(_slots_json_int "${url}/props" total_slots)"
  [[ -n "$n" ]] && { printf '%s\t%s\n' "$n" "server /props total_slots"; return 0; }
  return 0
}

# slot_setting_for_compose <compose.yml> — the env setting that compose reads for its slot count, or
# nothing: the first of these it reads as a setting, `${NAME…}` or a bare `- NAME` passthrough, with comment
# lines ignored. concurrency-probe.sh's SWEEP sets it per N (#1931); it used to set MAX_NUM_SEQS for every
# engine, which SGLang, llama.cpp and QEngine never read. MAX_NUM_SEQS comes first: HyperQwen's compose reads
# it and maps it to its own MAX_SEQS. Strata, Project Maya and TabbyAPI composes read none of them.
SLOT_SETTINGS="MAX_NUM_SEQS MAX_RUNNING_REQUESTS NPARALLEL NP FLASH_MAX_SEQS QE_MAX_SEQS"
slot_setting_for_compose() {
  local f="${1:-}" k body
  [[ -n "$f" && -r "$f" ]] || return 0
  body="$(command grep -vE '^[[:space:]]*#' "$f" || true)"
  for k in $SLOT_SETTINGS; do
    # single-quoted pieces: in double quotes `\$` would reach grep as `$`, an end-of-line anchor
    if command grep -qE '\$\{'"$k"'(:-[^}]*)?\}|^[[:space:]]*-[[:space:]]*'"$k"'(=|[[:space:]]*$)' <<<"$body"; then
      echo "$k"; return 0
    fi
  done
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
