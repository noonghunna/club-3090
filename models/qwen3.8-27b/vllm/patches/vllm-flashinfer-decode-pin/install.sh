#!/usr/bin/env bash
# FlashInfer decode workspace-buffer unpin installer — runs in the container
# entrypoint before `vllm serve`.
#
# Fixes the MTP c>=4 Xid 31 VIRT_READ crash (vllm#40756) on this W4A8+MTP
# hybrid-GDN config. Unpins the flashinfer decode workspace buffer (the one the
# MTP drafter re-plans K-1x/step) so the async plan copy-out can't read a
# stale plan.
#
# - Idempotent: marker-gated, safe to run multiple times.
# - No-op if flashinfer is not installed (a non-FlashInfer backend config).
# - Hard-fails (exit 2) on drift (a pin_memory outside a workspace buffer) so
#   the compose refuses to serve a half-patched state.
# - Inert on the SPEC_N=0 (no drafter) path: the race needs the drafter to
#   re-plan the decode wrapper.
#
# Env (all optional, read by the patcher):
#   FI_PINQ_LIB_ALL=1   also unpin prefill/sparse/pod (validated full mirror;
#                       default 0 = decode-only minimal fix).
#   FI_PINQ_LIB_ROOT=   override the flashinfer package dir (default auto-detect).
set -u
python3 /etc/club3090/flashinfer-decode-pin/patch_flashinfer_decode_pin.py
