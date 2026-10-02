#!/usr/bin/env bash
# post-ready-sync.sh — sync the gateway (and Open WebUI) once a server answers.
#
#   post-ready-sync.sh --url <ready url> --container <name> --port <port>
#                      [--owui] [--timeout S] [--interval S]
#
# switch.sh syncs the LiteLLM gateway at two points: after teardown (which
# renders the OLD model's route away) and after the new server is ready. The
# second sync only ran when switch.sh itself waited for ready, so `--no-wait`,
# or a boot slower than READY_TIMEOUT, left the gateway with no local routes
# for a model that came up fine minutes later (#1522). switch.sh now starts
# this script, detached, in exactly those two cases.
#
# It polls the ready URL, and when the server answers runs the same two steps
# switch.sh runs after a waited launch: owui-register.sh <port> (with --owui)
# and litellm-sync.sh --quiet. It does NOT run switch.sh's generation probe,
# so "the gateway was synced" is not "generation was verified".
#
# Exits without syncing when the container is gone or has stopped (a failed
# boot leaves nothing to route to; the teardown sync already removed the old
# route), and after --timeout seconds (default 3600) as a ceiling for a boot
# that never comes up. Exit codes: 0 synced, 2 container gone, 3 timed out.
set -uo pipefail

URL="" CONTAINER="" PORT="" OWUI=0
TIMEOUT="${POST_READY_SYNC_TIMEOUT:-3600}" INTERVAL="${POST_READY_SYNC_INTERVAL:-5}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)       URL="${2:-}"; shift 2 ;;
    --container) CONTAINER="${2:-}"; shift 2 ;;
    --port)      PORT="${2:-}"; shift 2 ;;
    --owui)      OWUI=1; shift ;;
    --timeout)   TIMEOUT="${2:-}"; shift 2 ;;
    --interval)  INTERVAL="${2:-}"; shift 2 ;;
    *) echo "post-ready-sync: unknown argument '$1'" >&2; exit 64 ;;
  esac
done
[[ -n "$URL" && -n "$CONTAINER" && -n "$PORT" ]] || {
  echo "post-ready-sync: --url, --container and --port are required" >&2; exit 64; }

LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
say() { printf '%s [post-ready-sync] %s\n' "$(date -Is)" "$*"; }

say "waiting for ${URL} (container ${CONTAINER}, up to ${TIMEOUT}s)"
start=$SECONDS
until curl -sf -o /dev/null --max-time 3 "$URL"; do
  # `restarting` keeps waiting: under a restart policy a slow first boot can
  # read that way, and the ceiling bounds a container that never settles.
  state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || true)"
  case "$state" in
    running|restarting|created) ;;
    *) say "container ${CONTAINER} is ${state:-gone}: not syncing"; exit 2 ;;
  esac
  if (( SECONDS - start >= TIMEOUT )); then
    say "no answer from ${URL} after ${TIMEOUT}s: not syncing"
    exit 3
  fi
  sleep "$INTERVAL"
done

say "${URL} answers after $((SECONDS - start))s: syncing"
if [[ "$OWUI" -eq 1 ]]; then
  bash "${LIB_DIR}/owui-register.sh" "$PORT" || say "owui-register failed (exit $?)"
fi
bash "${LIB_DIR}/litellm-sync.sh" --quiet || say "litellm-sync failed (exit $?)"
say "done"
exit 0
