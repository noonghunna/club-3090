"""
Disable strict WorkspaceManager.lock() semantics on vLLM v0.20+.

Background — see docs/UPSTREAM.md "vllm#39226":
  vLLM v0.20.0 added strict `WorkspaceManager.lock()` semantics: after
  lock, any allocation that grows the workspace raises `AssertionError`.
  The intent is to catch workspace-resize memory leaks. Side effect: on
  configs where `profile_run` doesn't exercise every workspace consumer
  (TurboQuant decode in our 27B + INT4 + MTP K=3 + TQ3 KV path), the
  workspace gets locked at 0 bytes and the first real call from that
  consumer raises.

This sidecar patches the file on disk to make `_ensure_workspace_size`
log a one-shot warning and grow the workspace anyway when locked,
instead of asserting. Behavior matches the pre-v0.20 path (workspace
was just resized as needed).

Why this is safe enough to ship:
  - The strict lock is a debugging aid, not a correctness guarantee. The
    underlying allocator was always allowed to grow workspace; the lock
    only added an assertion at the Python boundary.
  - Genesis P98 has a similar shape but its anchor doesn't match v0.20
    (auto-skips with "UNIFORM_SINGLE_TOKEN_DECODE marker → patch
    obsolete"); we still hit the lock on the path P98 was supposed to
    revert.
  - Sandermage's PROD configs (35B-A3B-FP8, 27B Lorbus fp8_e5m2) don't
    exercise the locked-grow path so they don't notice the assertion;
    our TurboQuant configs do.

Idempotent: marker `LOCAL workspace lock disable` — if present, skip.

Drop this when:
  - Upstream `profile_run` exercises TurboQuant decode (so workspace is
    sized correctly at lock time), OR
  - Upstream `WorkspaceManager` adds a per-callsite carve-out, OR
  - Sandermage's P98 ships a v0.20-aware anchor.
"""

import logging
import os
import re
import sys
from pathlib import Path

log = logging.getLogger("workspace_lock_disable")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

PATCH_TAG = "[workspace_lock_disable]"
PATCH_MARKER = "LOCAL workspace lock disable"


def _find_target() -> Path:
    import vllm
    return (
        Path(vllm.__file__).resolve().parent
        / "v1"
        / "worker"
        / "workspace.py"
    )


# Original (v0.20.0):
#     if self._locked:
#         raise AssertionError(
#             f"Workspace is locked but allocation from '{get_caller_info()}' "
#             f"requires {required_bytes / _MB:.2f} MB, current size is "
#             f"{current_size / _MB:.2f} MB. "
#             "Workspace growth is not allowed after locking."
#         )
#
# Replacement: log a one-shot warning at WARNING level (not INFO so it
# shows in default-log-level setups) and proceed with the grow path.
ORIGINAL_BLOCK = re.compile(
    r"            if self\._locked:\n"
    r"                raise AssertionError\(\n"
    r'                    f"Workspace is locked but allocation from \'\{get_caller_info\(\)\}\' "\n'
    r'                    f"requires \{required_bytes / _MB:\.2f\} MB, current size is "\n'
    r'                    f"\{current_size / _MB:\.2f\} MB\. "\n'
    r'                    "Workspace growth is not allowed after locking\."\n'
    r"                \)\n",
    re.MULTILINE,
)

REPLACEMENT_BLOCK = (
    "            if self._locked:\n"
    "                # LOCAL workspace lock disable — see club-3090\n"
    "                # patch_workspace_lock_disable.py docstring.\n"
    "                global _GENESIS_WORKSPACE_LOCK_WARNED\n"
    "                try:\n"
    "                    _already_warned = _GENESIS_WORKSPACE_LOCK_WARNED\n"
    "                except NameError:\n"
    "                    _already_warned = False\n"
    "                if not _already_warned:\n"
    "                    logger.warning(\n"
    "                        '[club-3090] Workspace lock violated by %s '\n"
    "                        '(%.2f MB needed, %.2f MB sized) — growing anyway. '\n"
    "                        'See vllm#39226. Future violations silenced.',\n"
    "                        get_caller_info(),\n"
    "                        required_bytes / _MB, current_size / _MB,\n"
    "                    )\n"
    "                    _GENESIS_WORKSPACE_LOCK_WARNED = True\n"
)


def main() -> int:
    target = _find_target()
    log.info("%s target: %s", PATCH_TAG, target)

    if not target.exists():
        log.warning("%s skip — target file missing", PATCH_TAG)
        return 0

    text = target.read_text()
    if PATCH_MARKER in text:
        log.info("%s already patched (idempotent skip)", PATCH_TAG)
        return 0

    if not ORIGINAL_BLOCK.search(text):
        log.warning(
            "%s skip — couldn't find lock-assertion anchor (workspace.py format drifted)",
            PATCH_TAG,
        )
        return 0

    text2 = ORIGINAL_BLOCK.sub(REPLACEMENT_BLOCK, text, count=1)
    if PATCH_MARKER not in text2:
        log.warning("%s skip — replacement block missing marker", PATCH_TAG)
        return 0

    target.write_text(text2)
    log.info("%s applied (lock-violation now logs WARNING, allocates anyway)", PATCH_TAG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
