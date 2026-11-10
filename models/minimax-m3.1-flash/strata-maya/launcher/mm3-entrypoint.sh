#!/bin/sh
# Project Maya entrypoint for the minimax-m3.1-flash strata-maya composes (Maya's maya-mm3 engine); every compose
# mounts this file and mm3_prepare.py / mm3_settings.py beside it. The image (strata-maya-club3090) has no entrypoint
# of its own and Maya's own setup (maya.sh) is not used: the weights come from our model directory, mounted
# read-only at /gguf (Maya's GGUF quant) and /companions/dspark (the DSpark drafter), and the pack is built on /data.
#
#   SETUP_ONLY=1        check the files, build the pack, write the config, then exit (so the boot that gets
#                       benched is not the one that did the one-time work)
#   REPACK=1            build the pack again (the engine's prefill calibration for this PC is kept)
#   VERIFY=0            skip the one-time SHA-256 check of the files
#   MAYA_GPUS           the cards to start on, as the container sees them: "0" (single), "0,1" (dual, a layer
#                       split: maya-mm3's tensor parallel is its EXL3 path's); the compose sets it
#   CONTEXT             --max-context (default 204800; Maya's own setup defaults to 65536 and recommends more
#                       where memory holds it: the model reaches 1M, Maya has run 256K end to end). Every
#                       position's K / V is pinned in RAM: 63,360 bytes a token at INT8, ~13.0 GB at 204800;
#                       mm3_settings.py validates it (>= 4096)
#   REASONING_EFFORT, ENABLE_THINKING, THINKING_BUDGET, TEMP, TOP_P, TOP_K, MIN_P, PRESENCE_PENALTY,
#   REPEAT_PENALTY, SERVED_NAME, LOOP_GUARD, KV, DSPARK, THREADS
#                       written into maya-mm3.json at every boot by mm3_prepare.py through mm3_settings.py
#                       (its header maps each to Maya's key); unset = the defaults below. A bad value stops
#                       the boot before the SHA-256 check.
#
# The server's defaults for requests that name none, with nothing set: effort auto (MiniMax's adaptive thinking,
# as Maya's own setup writes it) and sampling 1.0 / 0.95 (the model card's), the name minimax-m3.1-flash, an INT8
# K / V cache, the DSpark drafter off (Maya's own default until a speed calibration says it pays) and a CPU lane
# of half the CPUs (nproc / 2). A request's own reasoning_effort / chat_template_kwargs win, and so do the Chat
# settings saved from Maya's web page (maya-mm3.shared-settings.json next to the config).
#
# Its one argument is the quant (minimax-maya-s), fixed by the compose, never a setting, so a saved value can't
# point a slug at another quant's pack. Each quant has its own folder on /data.
set -eu
umask 0002                   # group-writable, so the launcher's user can clear /data
TAG="${1:?the quant (minimax-maya-s) is this script's argument}"
here="$(dirname "$0")"

/opt/maya/.venv/bin/python "$here/mm3_prepare.py" --tag "$TAG" --context "${CONTEXT:-204800}" \
  --gpus "${MAYA_GPUS:-0,1}" --verify "${VERIFY:-1}" --repack "${REPACK:-0}"

if [ "${SETUP_ONLY:-0}" = "1" ]; then
  echo "[maya-recipe] setup done: /data/$TAG"
  exit 0
fi
cd /opt/maya
exec /opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine strata --config "/data/$TAG/maya-mm3.json" \
  --host 0.0.0.0 --port 8080
