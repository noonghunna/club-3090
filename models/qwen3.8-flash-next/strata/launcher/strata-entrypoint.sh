#!/bin/sh
# Strata entrypoint for the qwen3.8-flash-next composes (Strata 0.2+, on upstream's own image); every
# topology mounts this file. It writes club-3090's launch settings into the run config setup made,
# then runs the image's own entrypoint, which does the rest: `strata launch` with the settings below,
# setup when the model is new or CONTEXT / KV / the cards changed (the model files are kept), serving.
#
#   CONTEXT, KV            passed to `strata launch` (setup owns them: they size the KV and expert cache)
#   STRATA_START_GPUS      the cards to start on, as the container sees them: "0,1" (the dual compose's
#                          default, a layer split) or one number (single card)
#   API_KEY                the key every request must send (Strata 0.2+ serves nothing but /health
#                          without one); the composes set club-3090's default unless STRATA_API_KEY is set
#   REASONING_EFFORT, ENABLE_THINKING, THINKING_BUDGET, SPEC_N,
#   ROPE_SCALING, ROPE_SCALE,
#   TEMP, TOP_P, TOP_K, MIN_P, PRESENCE_PENALTY, REPEAT_PENALTY, SERVED_NAME
#                          written into the run config by strata_config.py (mounted next to this file;
#                          its header maps each to Strata's key). Unset = Strata's own default; a bad
#                          value stops the boot. ROPE_SCALING/ROPE_SCALE extend RoPE past the model's
#                          trained 262144 (see the CONTEXT note above: raising CONTEXT re-runs setup,
#                          which picks yarn itself; these knobs override that at every boot).
#
# Its one argument is the model family, fixed by the compose (not a setting, so a saved value can never
# point a slug at the other model's files): qwen (ISTA-DASLab's GSQ-RCO of the original, the default
# when absent) or swift (a Swift 1.5 GSQ-RCO, here SC117's abliterated one). The weights are our model
# folder, mounted read-only at /gguf; packs, the MTP layer and the run configs go to /data.
#
# The first start for a model in a topology (its own folder on the data volume) runs setup before there
# is a run config to write the settings into, so it serves on a port nothing outside reaches
# (SETUP_PORT) until setup is done; then the settings go in and Strata starts for real on 8080. A later
# setup (after CONTEXT or KV changed) keeps the keys it doesn't write itself and our --spec too; once
# the server answers, what it runs from is checked anyway, and when a setting didn't survive, Strata
# restarts once to read it.
set -eu
umask 0002                   # group-writable, so the launcher's user can clear /data

FAMILY="${1:-qwen}"
case "$FAMILY" in
  qwen)  tag="iq3_s" ;;
  swift) tag="swift-iq3_s" ;;
  *) echo "[strata-recipe] the family argument must be qwen or swift, not '$FAMILY'" >&2; exit 1 ;;
esac
PY=/opt/strata/terminal/python/bin/python3   # the image's python (it has no other)
knobs=/opt/club3090/strata_config.py
inst="/data/install/strata-$tag.json"        # setup's run config; `strata launch` copies it at every start
mark="/data/club3090/strata-$tag.applied"    # its sha256 as we last left it
export CLUB3090_STRATA_SETUP_SPEC_FILE="/data/club3090/strata-$tag.setup-spec"   # setup's own --spec

"$PY" "$knobs" validate || exit 1            # a bad setting stops here, before any setup
mkdir -p /data/club3090

# the image entrypoint's own settings (docs/DOCKER.md upstream)
export MODEL=IQ3_S FAMILY MODEL_PATH=/gguf VISION=off
export GPUS="${STRATA_START_GPUS:-0,1}"
export CONTEXT="${CONTEXT:-131072}"
export PRESET="club-$FAMILY"

SETUP_PORT=8090                              # the first start's setup serves here, unpublished

apply() {
  "$PY" "$knobs" apply "$inst" "$inst"
  sha256sum "$inst" > "$mark"
}
healthy() {  # <port>
  "$PY" -c 'import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:%s/health" % sys.argv[1], timeout=3)
except Exception:
    sys.exit(1)' "$1"
}
start() {  # <port>: run the image's entrypoint until it serves on <port> (or exit with it if it stops)
  PORT="$1" /opt/strata/entrypoint.sh run &
  child=$!
  trap 'kill -TERM "$child" 2>/dev/null; wait "$child"; exit $?' TERM INT
  while kill -0 "$child" 2>/dev/null && ! healthy "$1"; do sleep 5; done
  if ! kill -0 "$child" 2>/dev/null; then wait "$child"; exit $?; fi
}
stop() {
  kill -TERM "$child"
  wait "$child" || true
  trap - TERM INT
}

if [ ! -f "$inst" ]; then
  echo "[strata-recipe] first start for this model in this topology: setup first, on port $SETUP_PORT (not published)"
  start "$SETUP_PORT"
  stop
fi
if [ -f "$inst" ]; then
  apply
else
  echo "[strata-recipe] WARNING: setup left no $inst, so club-3090's settings are not applied" >&2
fi

start "${PORT:-8080}"
if [ -f "$inst" ] && ! sha256sum -c "$mark" >/dev/null 2>&1; then
  # setup ran during this start anyway: apply again, and restart once if what Strata runs from changed
  before="$("$PY" "$knobs" effective "$inst")"
  apply
  if [ "$before" != "$("$PY" "$knobs" effective "$inst")" ]; then
    echo "[strata-recipe] setup ran during this start and dropped a setting: restarting Strata once to read it"
    stop
    exec /opt/strata/entrypoint.sh run
  fi
fi
wait "$child"
