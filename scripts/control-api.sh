#!/usr/bin/env bash
# control-api.sh — start/stop/status the model-switch HTTP control plane (dev / non-systemd path).
#
# The PRODUCTION path is the systemd unit (scripts/systemd/club3090-model-switch.service), which
# reads the same repo-root .env. This wrapper is the equivalent for a foreground rig without
# systemd: it loads .env, launches tools/model-switch/server.py detached, and manages a pidfile +
# log — so you never hand-type the env incantation. All config still comes from .env + the
# server's own MODEL_SWITCH_* defaults.
#
# Usage:
#   bash scripts/control-api.sh start      # launch (detached); waits for /healthz
#   bash scripts/control-api.sh stop
#   bash scripts/control-api.sh restart
#   bash scripts/control-api.sh status     # UP/DOWN (+ /status JSON if a token is set)
#   bash scripts/control-api.sh logs       # tail -f the server log
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for the server we launch below. Repo sources are full of
# unicode (— × → ⚠), and without this a rig on a real non-UTF-8 locale (de_DE.iso88591 and
# friends) decodes reads, stdout AND argv with the locale codec, which crashes the launcher/emit
# paths the server shells out to (#779). Exported, so the detached server and every switch.sh /
# setup.sh it spawns inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load repo-root .env, shell env winning (matches switch.sh precedence) so the daemon, downloads,
# and serving all agree on VLLM_API_KEY / PORT / MODEL_DIR.
if [[ -f "$ROOT_DIR/.env" ]]; then
  while IFS= read -r _line; do
    _line="${_line%%#*}"; _line="${_line#export }"
    [[ "$_line" == *=* ]] || continue
    _k="${_line%%=*}"; _k="${_k//[[:space:]]/}"; _v="${_line#*=}"
    [[ "$_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!_k:-}" ]]; then                         # shell env wins
      _v="${_v#"${_v%%[![:space:]]*}"}"; _v="${_v%"${_v##*[![:space:]]}"}"
      _v="${_v%\"}"; _v="${_v#\"}"; _v="${_v%\'}"; _v="${_v#\'}"
      export "$_k=$_v"
    fi
  done < "$ROOT_DIR/.env"
fi

PORT_API="${MODEL_SWITCH_PORT:-8099}"
BIND="${MODEL_SWITCH_BIND:-127.0.0.1}"
export MODEL_SWITCH_PORT="$PORT_API" MODEL_SWITCH_BIND="$BIND"
export MODEL_SWITCH_WATCHDOG="${MODEL_SWITCH_WATCHDOG:-1}"
export CLUB3090_API_TOKEN="${CLUB3090_API_TOKEN:-${VLLM_API_KEY:-}}"   # server + our status curl
TOKEN="$CLUB3090_API_TOKEN"

RUN_DIR="${MODEL_SWITCH_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/club3090-model-switch}"
export MODEL_SWITCH_STATE_DIR="$RUN_DIR"
mkdir -p "$RUN_DIR"
PIDFILE="$RUN_DIR/server.pid"
LOG="$RUN_DIR/server.log"
BASE="http://$BIND:$PORT_API"

_healthz() { curl -s -o /dev/null -w '%{http_code}' "$BASE/healthz" 2>/dev/null; }
# `|| true`: with `set -o pipefail`, grep finding no listener (port free) would otherwise make
# this exit non-zero and trip `set -e` in callers.
_listener_pid() { ss -tlnHp "sport = :$PORT_API" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true; }
_running_pid() {
  local p; p="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then echo "$p"; else _listener_pid; fi
}

start() {
  local p; p="$(_running_pid)"
  if [[ -n "$p" ]]; then echo "already running (pid $p) on $BASE"; return 0; fi
  echo "starting model-switch on $BASE  (watchdog=$MODEL_SWITCH_WATCHDOG, MODEL_DIR=${MODEL_DIR:-<repo>/models-cache})"
  nohup python3 tools/model-switch/server.py >"$LOG" 2>&1 < /dev/null &
  echo "$!" > "$PIDFILE"
  disown 2>/dev/null || true
  for _ in $(seq 1 40); do
    [[ "$(_healthz)" == "200" ]] && { echo "up (pid $(cat "$PIDFILE"))   log: $LOG"; return 0; }
    sleep 0.25
  done
  echo "started but /healthz isn't answering — check $LOG" >&2
  return 1
}

stop() {
  local p; p="$(_running_pid)"
  if [[ -z "$p" ]]; then echo "not running"; rm -f "$PIDFILE"; return 0; fi
  kill "$p" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "stopped (pid $p)"
}

status() {
  if [[ "$(_healthz)" != "200" ]]; then echo "DOWN  ($BASE)"; return 1; fi
  echo "UP    ($BASE, pid $(_running_pid))"
  if [[ -n "$TOKEN" ]]; then
    curl -s -H "Authorization: Bearer $TOKEN" "$BASE/status" \
      | { command -v jq >/dev/null 2>&1 && jq . || cat; }
  fi
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  logs)    tail -n "${LINES:-40}" -f "$LOG" ;;
  *) echo "usage: bash scripts/control-api.sh {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
