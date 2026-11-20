#!/bin/sh
# Project Maya entrypoint for the hy4-flash strata-maya composes (Maya's maya-hy4 engine); every compose mounts this
# file and hy4_prepare.py / hy4_settings.py beside it. The image (strata-maya-club3090) has no entrypoint of its own and
# Maya's own setup (maya.sh) is not used: the weights come from our model directory, each file mounted read-only into
# /gguf under the name Maya's setup gives it (the quant's model shard, the shared n-gram table, Tencent's MTP block, the
# vision tower), and the pack is built on /data.
#
#   SETUP_ONLY=1        check the files, build the pack, write the config, then exit (so the boot that gets
#                       benched is not the one that did the one-time work)
#   REPACK=1            build the pack again
#   VERIFY=0            skip the one-time SHA-256 check of the files
#   MAYA_GPUS           the cards to start on, as the container sees them: "0" (single), "0,1" (dual), "0,1,2"
#                       (multi3), "0,1,2,3" (multi4); maya-hy4 runs tensor parallel over the largest group of equal
#                       cards and splits the layers where none fits. The compose sets it
#   CONTEXT             --max-context (default 32768: the longest context Maya has run its quants end to end at; the
#                       model reaches 1M). The cache has no RAM path in this engine: every position sits in VRAM, so a
#                       longer context leaves fewer experts there. hy4_settings.py validates it (>= 4096)
#   REASONING_EFFORT, ENABLE_THINKING, THINKING_BUDGET, TEMP, TOP_P, TOP_K, MIN_P, PRESENCE_PENALTY,
#   REPEAT_PENALTY, SERVED_NAME, LOOP_GUARD, KV, DRAFTING, VISION, THREADS, HY4_EG_POPULATE
#                       written into maya-hy4.json at every boot by hy4_prepare.py through hy4_settings.py (its header
#                       maps each to Maya's key); unset = the defaults below. A bad value stops the boot before the
#                       SHA-256 check.
#
# The server's defaults for requests that name none, with nothing set: thinking high (Hy4-Flash thinks at high or not
# at all) and sampling 0.9 / 1.0 (Tencent's), the name hy4-flash, drafting with the MTP block on, pictures on, the
# engine's own KV rows (auto) and a CPU lane of half the CPUs (nproc / 2). A request's own reasoning_effort /
# chat_template_kwargs win, and so do the Chat settings saved from Maya's web page. One request at a time.
#
# Its one argument is the quant (hy4-maya-l, ...), fixed by the compose, never a setting, so a saved value can't point
# a slug at another quant's pack. Each quant has its own folder on /data.
set -eu
umask 0002                   # group-writable, so the launcher's user can clear /data
TAG="${1:?the quant (hy4-maya-xs, hy4-maya-s, hy4-maya-m or hy4-maya-l) is the argument to this script}"
here="$(dirname "$0")"

/opt/maya/.venv/bin/python "$here/hy4_prepare.py" --tag "$TAG" --context "${CONTEXT:-32768}" \
  --gpus "${MAYA_GPUS:-0,1}" --verify "${VERIFY:-1}" --repack "${REPACK:-0}"

if [ "${SETUP_ONLY:-0}" = "1" ]; then
  echo "[maya-recipe] setup done: /data/$TAG"
  exit 0
fi
cd /opt/maya
exec /opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine maya --config "/data/$TAG/maya-hy4.json" \
  --host 0.0.0.0 --port 8080
