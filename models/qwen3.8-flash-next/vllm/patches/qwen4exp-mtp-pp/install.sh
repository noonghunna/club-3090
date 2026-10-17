#!/usr/bin/env bash
# qwen4exp-mtp-pp installer: runs before `vllm serve`. Applies mtp-pp.diff to the image's vllm.
# Idempotent (marker = the patch's own comment). Fails the boot if it does not apply cleanly.
set -u
TAG="[qwen4exp-mtp-pp]"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
site="$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')" || exit 1
target="$site/vllm/models/qwen4_exp/nvidia/mtp.py"
if grep -q "Under PP > 1 the drafter always keeps an input embedding of its own" "$target"; then
  echo "$TAG already applied"; exit 0
fi
if ! (cd "$site" && patch -p1 --batch --forward --dry-run < "$here/mtp-pp.diff" >/dev/null); then
  echo "$TAG REFUSE: mtp-pp.diff does not apply to this vLLM" >&2
  (cd "$site" && patch -p1 --batch --forward --dry-run < "$here/mtp-pp.diff" | grep -E "FAILED|fuzz") >&2
  exit 1
fi
(cd "$site" && patch -p1 --batch --forward --no-backup-if-mismatch < "$here/mtp-pp.diff" >/dev/null) || exit 1
python3 -m py_compile "$target" || exit 1
echo "$TAG applied: MTP drafter works with --pipeline-parallel-size > 1"
