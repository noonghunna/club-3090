#!/usr/bin/env bash
# Vendored vllm#48375 installer — runs in the container entrypoint before serve.
# Idempotent; refuses boot (exit 1) on anchor drift so a re-pinned image can't
# silently serve unpatched while the compose claims the fix. Inert at runtime
# under the shipped --no-enable-prefix-caching default.
set -u
python3 /etc/club3090/pr48375/patch_mamba_drop_eagle_block.py
