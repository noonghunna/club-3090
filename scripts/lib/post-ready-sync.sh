#!/usr/bin/env bash
# post-ready-sync.sh — sync the gateway (and Open WebUI) once a server answers.
#
#   post-ready-sync.sh --url <ready url> --container <name> --port <port>
#                      [--owui] [--timeout S] [--interval S]
#
# On a switch, switch.sh advertises the model being loaded at the teardown
# (LITELLM_BOOTING_PORT — a retryable 500 while it boots, not a routeless
# window), then renders the gateway accurately once the new server is ready.
# This script is that accurate render for the cases switch.sh does not wait for
# itself: `--no-wait`, or a boot slower than READY_TIMEOUT (#1522). It is also
# the prune when the boot never comes up.
#
# It polls the ready URL, and when the server answers runs the same two steps
# switch.sh runs after a waited launch: owui-register.sh <port> (with --owui)
# and litellm-sync.sh --quiet. It does NOT run switch.sh's generation probe,
# so "the gateway was synced" is not "generation was verified".
#
# When the boot fails instead — the container is stopped/dead, or nothing
# answers before --timeout (default 3600, the ceiling for a boot that never
# comes up) — it renders the gateway back to what actually answers (a prune):
# a failed boot must not leave the model it was loading advertised. A
# container that is GONE is the exception: it was removed by a later
# switch.sh or --down, which rendered the gateway itself, so pruning here
# would drop that run's route and reopen the routeless window. Exit codes:
# 0 synced, 2 container stopped/dead (pruned), 4 container gone (no render),
# 3 timed out (pruned).
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
    "") # Gone, not merely down: removed by a later switch.sh or --down, which
        # has already rendered the gateway (optimistically for its own model,
        # or pruned for --down). Pruning here would drop that run's route for
        # the rest of its boot — the same routeless 400 window the optimistic
        # teardown render exists to close. Leave the render to that run.
        say "container ${CONTAINER} is gone (removed by a later switch or --down, which rendered the gateway itself): leaving the render alone"
        exit 4 ;;
    *) say "container ${CONTAINER} is ${state}: rendering the gateway back to what serves"
       bash "${LIB_DIR}/litellm-sync.sh" --quiet || say "litellm-sync failed (exit $?)"
       exit 2 ;;
  esac
  if (( SECONDS - start >= TIMEOUT )); then
    say "no answer from ${URL} after ${TIMEOUT}s: rendering the gateway back to what serves"
    bash "${LIB_DIR}/litellm-sync.sh" --quiet || say "litellm-sync failed (exit $?)"
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
