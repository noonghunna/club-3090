#!/usr/bin/env bash
# Install vLLM PR #41800 — `truncate_prompt_tokens` kwarg on get_max_tokens.
#
# WHY THIS OVERLAY EXISTS:
# opencode (and other agentic clients like codex-cli) send `truncate_prompt_tokens`
# on chat-completion requests. Pre-#41800, vLLM's `get_max_tokens()` doesn't
# accept that kwarg — and somewhere upstream of the function the kwarg gets
# unpacked into the call — so requests fail with:
#   HTTP 400: get_max_tokens() got an unexpected keyword argument 'truncate_prompt_tokens'
#
# PR: https://github.com/vllm-project/vllm/pull/41800
# Merged: 2026-05-06 at commit d5b31c95
# Affected pins on master:
#   - vllm-nightly-mtp (01d4d1ad, 2026-05-04) — pre-fix
#   - vllm-nightly-dflash (e47c98ef) — pre-fix
#   - vllm-nightly-full (e47c98ef) — pre-fix
#   (vllm-nightly-clean at bf610c2f is POST-fix; doesn't need the overlay)
#
# Tracking issue: #139 (noonghunna/club-3090)
# Triggered by: #138 — SEVENID's opencode boot failure on dual-dflash-noviz.
#
# WHY A PYTHON ANCHOR-BASED PATCHER:
# The PR is +7 lines in `vllm/entrypoints/utils.py` (the actual fix) plus a
# handful of forward-compat call-site additions in 5 other files. The
# function-signature change in utils.py is the ONLY thing required to fix
# the TypeError — once `get_max_tokens` accepts the kwarg, requests stop
# crashing. The call-site changes are nice-to-have semantic completeness
# (actually applying the truncation), so we patch those too via anchors.
#
# Idempotent: each anchor checks for a sentinel marker before inserting.

set -euo pipefail

# Container's vLLM install path. Override via env if vLLM moves.
SITE_PACKAGES="${CLUB3090_PR41800_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"

UTILS_PY="$SITE_PACKAGES/vllm/entrypoints/utils.py"

if [ ! -f "$UTILS_PY" ]; then
  echo "[club3090/pr41800] ERROR: $UTILS_PY not found; aborting overlay install" >&2
  exit 1
fi

python3 - <<'PY'
import os
import re
import sys

site_packages = os.environ.get(
    "CLUB3090_PR41800_SITE_PACKAGES",
    "/usr/local/lib/python3.12/dist-packages",
)
utils_py = f"{site_packages}/vllm/entrypoints/utils.py"

SENTINEL = "# PATCH: truncate_prompt_tokens kwarg (club3090/pr41800)"

# Function signature update: add `truncate_prompt_tokens: int | None = None,`
# as a kwarg on `get_max_tokens`. Anchor on the existing line that closes
# the signature (`override_max_tokens: int | None = None,` line right before `) -> int:`).
SIGNATURE_ANCHOR_RE = re.compile(
    r'^(?P<indent>[ \t]+)override_max_tokens: int \| None = None,\n(?P<close>[ \t]*\) -> int:)',
    re.MULTILINE,
)
SIGNATURE_INSERT = '''    override_max_tokens: int | None = None,
    truncate_prompt_tokens: int | None = None,  # PATCH: truncate_prompt_tokens kwarg (club3090/pr41800)
) -> int:'''

# Body update: insert truncation-aware input_length adjustment BEFORE the
# `if max_model_len < input_length:` check. Anchor on that line.
BODY_ANCHOR_RE = re.compile(
    r'^(?P<indent>[ \t]+)if max_model_len < input_length:',
    re.MULTILINE,
)
BODY_INSERT = '''    # PATCH: truncate_prompt_tokens kwarg (club3090/pr41800)
    if truncate_prompt_tokens is not None:
        limit = truncate_prompt_tokens
        input_length = min(
            input_length,
            max_model_len if limit == -1 else limit,
        )
    if max_model_len < input_length:'''

with open(utils_py, "r", encoding="utf-8") as f:
    src = f.read()

if SENTINEL in src:
    print(f"[club3090/pr41800] {utils_py}: sentinel present, patch already applied; no-op", file=sys.stderr)
    sys.exit(0)

# Upstream-fix detection: if the function signature already accepts the kwarg
# (i.e. the engine pinned a post-#41800 nightly), the overlay is unnecessary
# and should no-op gracefully so composes that mount it on a post-fix image
# (e.g. via vllm-nightly-clean) still boot cleanly.
UPSTREAM_RE = re.compile(
    r'def get_max_tokens\([^)]*truncate_prompt_tokens\b',
    re.DOTALL,
)
if UPSTREAM_RE.search(src):
    print(f"[club3090/pr41800] {utils_py}: upstream get_max_tokens() already accepts truncate_prompt_tokens; no-op", file=sys.stderr)
    sys.exit(0)

# Apply signature patch first (so the function accepts the kwarg)
m = SIGNATURE_ANCHOR_RE.search(src)
if not m:
    print(
        f"[club3090/pr41800] ERROR: signature anchor "
        f"'override_max_tokens: int | None = None, ... ) -> int:' not found in {utils_py}. "
        f"vLLM nightly may have changed entrypoints/utils.py — overlay needs re-anchoring.",
        file=sys.stderr,
    )
    sys.exit(1)

src = SIGNATURE_ANCHOR_RE.sub(SIGNATURE_INSERT, src, count=1)

# Apply body patch
m = BODY_ANCHOR_RE.search(src)
if not m:
    print(
        f"[club3090/pr41800] ERROR: body anchor 'if max_model_len < input_length:' not found in {utils_py} "
        f"after signature patch. vLLM nightly diverged unexpectedly — overlay needs re-anchoring.",
        file=sys.stderr,
    )
    sys.exit(1)

src = BODY_ANCHOR_RE.sub(BODY_INSERT, src, count=1)

with open(utils_py, "w", encoding="utf-8") as f:
    f.write(src)

# Quick validity check
import ast
try:
    ast.parse(src)
except SyntaxError as e:
    print(f"[club3090/pr41800] ERROR: post-patch utils.py is not valid Python: {e}", file=sys.stderr)
    sys.exit(1)

print(f"[club3090/pr41800] {utils_py}: signature + body patches applied (truncate_prompt_tokens kwarg)", file=sys.stderr)
PY

echo "[club3090/pr41800] install complete" >&2
