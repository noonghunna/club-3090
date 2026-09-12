#!/usr/bin/env bash
# Vendored vllm#51581 installer — runs in the container entrypoint before serve.
# Idempotent; refuses boot (exit 1) on anchor drift, because an unpatched fused-KV
# path on a quantized drafter can CORRUPT SILENTLY rather than fail loudly.
set -u
python3 /etc/club3090/dflash-dense-kv/patch_dflash_dense_kv.py
