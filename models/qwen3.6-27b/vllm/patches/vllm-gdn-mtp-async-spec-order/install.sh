#!/usr/bin/env bash
# club-3090 GDN+MTP async spec-order fix installer — runs in the container
# entrypoint before serve. Idempotent; refuses boot (exit 1) on anchor drift so a
# re-pinned image can't silently serve unpatched while the compose claims the fix.
# Inert unless spec decode is active (the guarded event is None otherwise).
set -u
python3 /etc/club3090/gdn-async-order/patch_gdn_mtp_async_spec_order.py
