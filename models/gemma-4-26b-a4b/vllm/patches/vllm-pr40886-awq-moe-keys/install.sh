#!/usr/bin/env bash
# Install vLLM PR #40886 — compressed-tensors AWQ MoE key remapping for
# Gemma 4 26B-A4B. Without this patch, `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
# fails to load with KeyError on `moe.gate_up_proj_packed` (or similar)
# because vLLM's `gemma4.py::_weight_iterator` doesn't handle the
# `_packed`/`_scale` suffix on MoE expert weights.
#
# PR: https://github.com/vllm-project/vllm/pull/40886
# Head commit at time of vendor: 652819dad0bf9bbb0436d6660822e7aff30c3ff0
# Author tested on: RTX 3090 24 GB (same SKU as ours), vLLM 0.19.1
#
# WHY a Python anchor-based patcher instead of full-file replacement:
# The PR's diff is +23 / -0 — pure insertion before an existing branch
# in `_weight_iterator`. Anchor-based insertion is robust to upstream
# drift around the function (vLLM nightly may add unrelated logic
# elsewhere in gemma4.py without breaking our patch).
#
# Idempotent: the patcher checks for a sentinel comment before inserting,
# so re-running it on an already-patched file is a no-op.
#
# Drop when: vLLM PR #40886 merges upstream AND our engine pin bumps
# past the merge commit.

set -euo pipefail

# Container's vLLM install path. Override via env if vLLM moves.
GEMMA4_PY="${CLUB3090_PR40886_TARGET:-/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py}"

if [ ! -f "$GEMMA4_PY" ]; then
  echo "[club3090/pr40886] ERROR: $GEMMA4_PY not found; aborting overlay install" >&2
  exit 1
fi

python3 - <<'PY'
import os
import re
import sys

target = os.environ.get(
    "CLUB3090_PR40886_TARGET",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py",
)

SENTINEL = "# PATCH: AWQ compressed-tensors key remapping (club3090/pr40886)"

PATCH_BLOCK = '''                # PATCH: AWQ compressed-tensors key remapping (club3090/pr40886)
                if "moe.gate_up_proj_packed" in name and weight.dim() == 3:
                    mid = weight.size(1) // 2
                    for e in range(weight.size(0)):
                        base = name.replace("moe.", f"moe.experts.{e}.")
                        yield base.replace("gate_up_proj_packed", "gate_proj.weight_packed"), weight[e, :mid]
                        yield base.replace("gate_up_proj_packed", "up_proj.weight_packed"), weight[e, mid:]
                    continue
                if "moe.gate_up_proj_scale" in name and weight.dim() == 3:
                    mid = weight.size(1) // 2
                    for e in range(weight.size(0)):
                        base = name.replace("moe.", f"moe.experts.{e}.")
                        yield base.replace("gate_up_proj_scale", "gate_proj.weight_scale"), weight[e, :mid]
                        yield base.replace("gate_up_proj_scale", "up_proj.weight_scale"), weight[e, mid:]
                    continue
                if "moe.down_proj_packed" in name and weight.dim() == 3:
                    for e in range(weight.size(0)):
                        yield name.replace("moe.", f"moe.experts.{e}.").replace("down_proj_packed", "down_proj.weight_packed"), weight[e]
                    continue
                if "moe.down_proj_scale" in name and weight.dim() == 3:
                    for e in range(weight.size(0)):
                        yield name.replace("moe.", f"moe.experts.{e}.").replace("down_proj_scale", "down_proj.weight_scale"), weight[e]
                    continue
'''

# Insert immediately above the existing `if "moe.gate_up_proj" in name and weight.dim() == 3:`
# branch inside `_weight_iterator`. This is the line the upstream PR puts the patch above.
ANCHOR_RE = re.compile(
    r'^(?P<indent>[ \t]+)(?P<line>if "moe\.gate_up_proj" in name and weight\.dim\(\) == 3:)',
    re.MULTILINE,
)

with open(target, "r", encoding="utf-8") as f:
    src = f.read()

if SENTINEL in src:
    print(f"[club3090/pr40886] {target}: sentinel present, patch already applied; no-op", file=sys.stderr)
    sys.exit(0)

m = ANCHOR_RE.search(src)
if not m:
    print(
        f"[club3090/pr40886] ERROR: anchor 'if \"moe.gate_up_proj\" in name and weight.dim() == 3:' "
        f"not found in {target}. vLLM nightly may have changed gemma4.py — overlay needs re-anchoring.",
        file=sys.stderr,
    )
    sys.exit(1)

# Insertion point: start of the matched line
insert_at = m.start()
patched = src[:insert_at] + PATCH_BLOCK + src[insert_at:]

with open(target, "w", encoding="utf-8") as f:
    f.write(patched)

print(f"[club3090/pr40886] {target}: PR #40886 (AWQ MoE key remapping) applied", file=sys.stderr)
PY

echo "[club3090/pr40886] install complete" >&2
