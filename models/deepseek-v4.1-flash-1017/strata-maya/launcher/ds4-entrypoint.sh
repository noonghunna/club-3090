#!/bin/sh
# Project Maya entrypoint for the deepseek-v4.1-flash-1017 strata-maya composes (Maya's maya-ds4 engine); every
# compose mounts this file and ds4_prepare.py / ds4_settings.py beside it. The image (strata-maya-club3090) has no
# entrypoint of its own and Maya's own setup (maya.sh) is not used: the weights come from our model directory,
# mounted read-only at /gguf (the GGUF) and, for a quant with a drafter, /dspark (the DSpark drafter), and the
# pack is built on /data.
#
#   SETUP_ONLY=1        check the files, build the pack, write the config, then exit (so the boot that gets
#                       benched is not the one that did the one-time work)
#   REPACK=1            build the pack again (the learned expert profile is kept)
#   VERIFY=0            skip the one-time SHA-256 check of the files
#   MAYA_GPUS           the cards to start on, as the container sees them: "0,1" (dual); the compose sets it
#   MAYA_TP             --tp: 1 (the compose default) on two GPUs is a layer split, each GPU its own layers with
#                       whole experts; 2 is tensor parallel (each expert split between the cards: on two 3090s,
#                       in a hand-run test at 8K context, 2026-10-25: decode +9% / +5%, a 6K prompt -35%)
#   CONTEXT             --max-context (default 65536: Maya's own DeepSeek default until v1.1.30, which picks 256K
#                       where the PC's VRAM and RAM hold the model)
#   REASONING_EFFORT, ENABLE_THINKING, THINKING_BUDGET, TEMP, TOP_P, TOP_K, MIN_P, PRESENCE_PENALTY,
#   REPEAT_PENALTY, SERVED_NAME, THREADS
#                       written into maya-ds4.json at every boot by ds4_prepare.py through ds4_settings.py
#                       (its header maps each to Maya's key); unset = the defaults below. A bad value stops
#                       the boot before the SHA-256 check.
#
# The server's defaults for requests that name none, with nothing set: effort high and sampling 1.0 / 1.0 (as
# Maya's own setup writes them for DeepSeek), the name deepseek-v4.1-flash, and a CPU lane of half the CPUs
# (nproc / 2). A request's own reasoning_effort / chat_template_kwargs win, and so do the Chat
# settings saved from Maya's web page (maya.shared-settings.json next to the config).
#
# Its one argument is the quant (ista-iq2-xxs, ud-iq3-s or maya-xs), fixed by the compose, never a setting, so a saved
# value can't point a slug at another quant's pack. Each quant has its own folder on /data.
set -eu
umask 0002                   # group-writable, so the launcher's user can clear /data
TAG="${1:?the quant (ista-iq2-xxs, ud-iq3-s or maya-xs) is this script's argument}"
here="$(dirname "$0")"

/opt/maya/.venv/bin/python "$here/ds4_prepare.py" --tag "$TAG" --context "${CONTEXT:-65536}" \
  --gpus "${MAYA_GPUS:-0,1}" --tp "${MAYA_TP:-1}" --verify "${VERIFY:-1}" --repack "${REPACK:-0}"

if [ "${SETUP_ONLY:-0}" = "1" ]; then
  echo "[maya-recipe] setup done: /data/$TAG"
  exit 0
fi
cd /opt/maya
exec /opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine strata --config "/data/$TAG/maya-ds4.json" \
  --host 0.0.0.0 --port 8080
