#!/bin/sh
# Project Maya entrypoint for the glm-5.3-flash strata-maya composes; every topology mounts this file
# and maya_prepare.py beside it. The image (strata-maya-club3090) has no entrypoint of its own and Maya's
# own setup (maya.sh) is not used: the weights come from our model directory, mounted read-only at
# /gguf, and the pack is built on /data.
#
#   SETUP_ONLY=1        link, check and pack, write the config, then exit (so the boot that gets
#                       benched is not the one that did the one-time work)
#   REPACK=1            build the pack again
#   VERIFY=0            skip the one-time SHA-256 check of the shards
#   MAYA_GPUS           the cards to start on, as the container sees them: "0,1" (dual, a layer
#                       split), "0" (single), "0,1,2,3" (multi4); the compose sets it
#   CONTEXT             --max-context (default 131072); maya_settings.py validates it (>= 4096), so a bad
#                       value stops the boot like any other launch setting
#   MAYA_PCIE_SHARE     the engine's STRATA_GLM_PCIE_SHARE (the share of each token's RAM-tier experts
#                       sent over PCIe instead of computed on the CPU), set only when non-empty: what
#                       Maya's own `maya.sh --calibrate` tunes. Unset: the engine's own choice
#   MAYA_CPU_LANE       the engine's STRATA_GLM_CPU_LANE (CPU threads per card), the same way
#   REASONING_EFFORT, ENABLE_THINKING, THINKING_BUDGET, KV,
#   TEMP, TOP_P, TOP_K, MIN_P, PRESENCE_PENALTY, REPEAT_PENALTY, SERVED_NAME
#                       written into maya.json at every boot by maya_prepare.py through maya_settings.py
#                       (its header maps each to Maya's key); unset = the defaults below. A bad value
#                       stops the boot before the SHA-256 check.
#
# The server's defaults for requests that name none, with nothing set: effort max (as the other
# GLM-5.3-Flash slugs; Maya's own setup writes high), temperature 1.0 / top_p 0.95 (GLM-5.3's vendor
# sampling), Maya's own 32K thinking budget and the name glm-5.3-flash. A request's own
# reasoning_effort / chat_template_kwargs win, and so do the Chat settings saved from Maya's web page
# (maya.shared-settings.json next to the config).
#
# Its one argument is the quant (maya-s | maya-s24 | maya-m | maya-l), fixed by the compose, never a
# setting, so a saved value can't point a slug at another quant's pack. Each quant has its own folder
# on /data.
# serve/server.py passes its environment on to the engine, which is how MAYA_PCIE_SHARE /
# MAYA_CPU_LANE reach it.
set -eu
umask 0002                   # group-writable, so the launcher's user can clear /data
TAG="${1:?the quant (maya-s, maya-s24, maya-m or maya-l) is this script's argument}"
here="$(dirname "$0")"

/opt/maya/.venv/bin/python "$here/maya_prepare.py" --tag "$TAG" --context "${CONTEXT:-131072}" \
  --gpus "${MAYA_GPUS:-0}" --verify "${VERIFY:-1}" --repack "${REPACK:-0}"

if [ "${SETUP_ONLY:-0}" = "1" ]; then
  echo "[maya-recipe] setup done: /data/$TAG"
  exit 0
fi
# exported only when set: an empty value would reach the engine as "" rather than as unset
if [ -n "${MAYA_PCIE_SHARE:-}" ]; then export STRATA_GLM_PCIE_SHARE="$MAYA_PCIE_SHARE"; fi
if [ -n "${MAYA_CPU_LANE:-}" ]; then export STRATA_GLM_CPU_LANE="$MAYA_CPU_LANE"; fi
cd /opt/maya
exec /opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine strata --config "/data/$TAG/maya.json" \
  --host 0.0.0.0 --port 8080
