#!/bin/sh
# Project Maya entrypoint for the inkling-2-mini strata-maya compose (Maya's maya-ik2 engine, v1.1.72 on); the compose
# mounts this file and ik2_prepare.py / ik2_settings.py beside it. The image (strata-maya-club3090) has no entrypoint of
# its own and Maya's own setup (maya.sh) is not used: the weights come from our model directory, each file mounted
# read-only into /gguf under the name Maya's setup gives it (unsloth's UD-Q4_K_XL GGUF and the model's drafter, as
# unsloth's GGUF of it), and the tokenizer and the config are written on /data.
#
#   SETUP_ONLY=1        check the files, export the tokenizer, write the config, then exit (so the boot that gets
#                       benched is not the one that did the one-time work)
#   REPACK=1            export the tokenizer again
#   VERIFY=0            skip the one-time SHA-256 check of the files
#   MAYA_GPUS           the card to start on, as the container sees it: "0" (the compose sets it)
#   CONTEXT             --max-context (default 262144, the maintainer's cap for Inkling-2-Mini; the model reaches 1M,
#                       and from v1.1.74 maya-ik2 moves experts to RAM past what the card holds). ik2_settings.py
#                       validates it (>= 4096)
#   REASONING_EFFORT, TEMP, TOP_P, TOP_K, MIN_P, PRESENCE_PENALTY, REPEAT_PENALTY, SERVED_NAME, LOOP_GUARD, KV,
#   DRAFTING, THREADS
#                       written into maya-ik2.json at every boot by ik2_prepare.py through ik2_settings.py (its header
#                       maps each to Maya's key); unset = the defaults below. A bad value stops the boot before the
#                       SHA-256 check.
#
# The server's defaults for requests that name none, with nothing set: effort high (thinkingmachines' template; none
# turns thinking off) and sampling 1.0 / 1.0, the name inkling-2-mini, drafting auto (maya-ik2 measures once
# which of no drafting, the prompt lookup and the drafter decodes fastest, and keeps it on /data), the cache in FP8.
# A request's own reasoning_effort / chat_template_kwargs win, and so do the Chat settings saved from Maya's web page.
# Text only: pictures need Maya's maya-vision encoder, which the image does not build. One request at a time.
#
# Its one argument is the quant (ik2-ud-q4_k_xl), fixed by the compose, never a setting, so a saved value can't point
# the slug at other files. Each quant has its own folder on /data.
set -eu
umask 0002                   # group-writable, so the launcher's user can clear /data
TAG="${1:?the quant (ik2-ud-q4_k_xl) is the argument to this script}"
here="$(dirname "$0")"

/opt/maya/.venv/bin/python "$here/ik2_prepare.py" --tag "$TAG" --context "${CONTEXT:-262144}" \
  --gpus "${MAYA_GPUS:-0}" --verify "${VERIFY:-1}" --repack "${REPACK:-0}"

if [ "${SETUP_ONLY:-0}" = "1" ]; then
  echo "[maya-recipe] setup done: /data/$TAG"
  exit 0
fi
cd /opt/maya
exec /opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine maya --config "/data/$TAG/maya-ik2.json" \
  --host 0.0.0.0 --port 8080
