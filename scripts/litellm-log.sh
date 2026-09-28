#!/usr/bin/env bash
# litellm-log.sh — turn the LiteLLM gateway's request logging on to troubleshoot,
# and back off. OFF is the default and what every normal (re)start gives you.
#
#   bash scripts/litellm-log.sh status   # is it on?
#   bash scripts/litellm-log.sh on       # log every request as FORWARDED to the engine
#   bash scripts/litellm-log.sh off      # back to the default
#   docker logs -f litellm               # read it
#
# OFF (default): one access line per request (method, path, status) and errors.
# No prompt or reply content is written anywhere.
# ON (LITELLM_LOG=DEBUG): each request exactly as the gateway forwarded it — the
# engine URL, every parameter (reasoning_effort, chat_template_kwargs, tools, …),
# the full messages — and the engine's raw reply. That is what answers "did the
# client's setting reach the engine?" and "what did the engine get?".
# ⚠️ ON writes full prompts and replies into the container log. Turn it off when
# done; the log is capped at 3 × 50 MB and goes when the container is recreated.
#
# Each switch recreates the gateway container (~10 s, like any route change) from
# the compose file it was started from, so its config mount does not move. It
# stays on across the route-sync restarts `switch.sh` does, and any fresh start
# of the service (gpu-mode, `docker compose up`) comes back OFF.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"   # club_config_compose_env_file runs python3 (#779)
CONTAINER="litellm"
LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib"
# shellcheck source=lib/club-config.sh
. "$LIB_DIR/club-config.sh"

case "${1:-status}" in
  on)     WANT="${2:-DEBUG}" ;;
  off)    WANT="" ;;
  status) WANT="__status__" ;;
  -h|--help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "litellm-log: usage: $0 on [DEBUG|INFO] | off | status" >&2; exit 2 ;;
esac
case "$WANT" in ""|__status__|DEBUG|INFO) ;; *) echo "litellm-log: level must be DEBUG or INFO (got '$WANT')" >&2; exit 2 ;; esac

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "litellm-log: no '$CONTAINER' container — start the gateway first (gpu-mode, or docker compose up in services/litellm)." >&2
  exit 1
fi
current="$(docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^LITELLM_LOG=//p' | head -1)"

if [[ "$WANT" == "__status__" ]]; then
  if [[ -n "$current" ]]; then
    echo "LiteLLM request logging: ON (LITELLM_LOG=$current) — full prompts are being logged; turn off with: bash $0 off"
  else
    echo "LiteLLM request logging: off (default)"
  fi
  exit 0
fi
if [[ "$current" == "$WANT" ]]; then
  echo "litellm-log: already ${WANT:-off} — nothing to do."
  exit 0
fi

# Recreate from the SAME compose project the running gateway came from — not this
# checkout's services/litellm: run from a worktree, that would re-mount the
# worktree's config.runtime.yaml and the gateway would serve stale routes.
workdir="$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}')"
files="$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}')"
project="$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "com.docker.compose.project"}}')"
if [[ -z "$workdir" || -z "$files" ]]; then
  echo "litellm-log: '$CONTAINER' was not started by docker compose — can't recreate it safely." >&2
  exit 1
fi
compose_args=(-p "$project")
IFS=',' read -ra cfgs <<<"$files"
for f in "${cfgs[@]}"; do compose_args+=(-f "$f"); done
# The recreate must see the same settings gpu-mode started it with — above all
# LITELLM_MASTER_KEY from secrets.env (club-3090#1467). Without them the compose
# falls back to the public default key and every client carrying the per-install
# key gets `400 No connected db.`. Settings of the checkout the gateway came from;
# the shell still wins (LITELLM_LOG below rides the environment).
gw_root="$(cd "$workdir/../.." 2>/dev/null && pwd || true)"
CLUB3090_COMPOSE_ENV_FILE="$(club_config_compose_env_file "$gw_root" 2>/dev/null || true)"
if [[ -n "$CLUB3090_COMPOSE_ENV_FILE" && -s "$CLUB3090_COMPOSE_ENV_FILE" ]]; then
  compose_args+=(--env-file "$CLUB3090_COMPOSE_ENV_FILE")
elif [[ -n "$CLUB3090_COMPOSE_ENV_FILE" ]]; then
  rm -f "$CLUB3090_COMPOSE_ENV_FILE"; CLUB3090_COMPOSE_ENV_FILE=""
fi
# …and the keys this rig's own routes use, as gpu-mode starts it (club-3090#1466): a
# 0600 temp file of just those keys, whose path the compose reads from
# CLUB3090_LITELLM_ROUTE_KEYS. Without it a route whose key is saved in secrets.env
# would come back without one. Empty when no route needs a saved key.
ROUTE_KEYS=""
if [[ -n "$gw_root" ]]; then
  ROUTE_KEYS="$(python3 "$LIB_DIR/litellm_local.py" route-keys-file --root "$gw_root" || true)"
fi
if [[ -n "$ROUTE_KEYS" ]]; then
  export CLUB3090_LITELLM_ROUTE_KEYS="$ROUTE_KEYS"
else
  unset CLUB3090_LITELLM_ROUTE_KEYS
fi
trap 'rm -f ${CLUB3090_COMPOSE_ENV_FILE:+"$CLUB3090_COMPOSE_ENV_FILE"} ${ROUTE_KEYS:+"$ROUTE_KEYS"}' EXIT

if [[ -n "$WANT" ]]; then
  export LITELLM_LOG="$WANT"
else
  unset LITELLM_LOG
fi
echo "litellm-log: recreating the gateway with request logging ${WANT:-off} …"
(cd "$workdir" && docker compose "${compose_args[@]}" up -d --force-recreate "$CONTAINER") >/dev/null

for _ in $(seq 1 45); do
  curl -sf -o /dev/null http://127.0.0.1:4000/health/liveliness && break
  sleep 2
done
now="$(docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^LITELLM_LOG=//p' | head -1)"
if [[ "$now" != "$WANT" ]]; then
  echo "litellm-log: the gateway came back with LITELLM_LOG='${now}', expected '${WANT}'." >&2
  exit 1
fi
if [[ -n "$WANT" ]]; then
  echo "LiteLLM request logging: ON ($WANT). Read it with: docker logs -f $CONTAINER"
  echo "⚠️  Full prompts and replies are now logged — turn it off when done: bash $0 off"
else
  echo "LiteLLM request logging: off (default)."
fi
