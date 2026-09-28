#!/usr/bin/env bash
# test-compose-sglang-nccl-p2p-knob — every multi-GPU compose that launches an SGLang
# server forwards NCCL_P2P_DISABLE into the container.
#
# WHY THIS TEST EXISTS
# --------------------
# SGLang has no interconnect detection: on a driver that grants PCIe P2P (a patched
# module, some server boards, the VM route in docs/PCIE_P2P.md §4a) NCCL uses the peer
# path automatically. NCCL_P2P_DISABLE=1 is the only way back to host-staged
# transfers — and docker forwards a variable only when the compose lists it. Until
# 2026-09-28 just 1 of 26 SGLang composes did, so the documented off-switch silently
# did nothing on the rest. Single-card composes are exempt: there is no GPU↔GPU path.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Forwarded = listed under environment: — bare (`- NCCL_P2P_DISABLE`), with a value
# (`- NCCL_P2P_DISABLE=…`), or in map form (`NCCL_P2P_DISABLE: …`). A mention in a
# comment does not count.
forwards() { command grep -qE '^[[:space:]]+(- )?NCCL_P2P_DISABLE([[:space:]]*$|=|:)' "$1"; }

# Negative control first: a compose without the entry must be caught, or the scan
# below proves nothing.
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
printf 'services:\n  x:\n    environment:\n      - SPEC_N\n      # NCCL_P2P_DISABLE mentioned only in a comment\n' > "$tmp"
if forwards "$tmp"; then echo "✗ self-test: the check accepted a compose that does not forward NCCL_P2P_DISABLE" >&2; exit 1; fi

fail=0; n=0
while IFS= read -r f; do
  case "$f" in */compose/single/*) continue ;; esac
  n=$((n + 1))
  forwards "$f" || { echo "✗ $f launches SGLang on >1 GPU without forwarding NCCL_P2P_DISABLE" >&2; fail=1; }
done < <(command grep -rlE 'sglang\.launch_server|sglang serve' models --include='*.yml' | sort)
[[ $n -gt 0 ]] || { echo "✗ found no multi-GPU SGLang composes — the scan itself is broken" >&2; exit 1; }
[[ $fail -eq 0 ]] && echo "test-compose-sglang-nccl-p2p-knob: ok ($n multi-GPU SGLang composes forward NCCL_P2P_DISABLE)"
exit $fail
