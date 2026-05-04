"""Auto-load the Cliff 2 residency instrumentation inside vLLM containers.

This file is mounted via PYTHONPATH. Python imports ``sitecustomize`` on
startup, including multiprocessing worker children. We deliberately skip the
pre-serve Genesis patch application sidecars so instrumentation does not import
vLLM before those text patches have been applied.
"""

from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    return bool(
        os.environ.get("RESIDENCY_LOG_PATH")
        or os.environ.get("GENESIS_RESIDENCY_LOG")
    )


def _is_pre_serve_patch_process() -> bool:
    argv = " ".join(sys.argv)
    skip_markers = (
        "vllm._genesis.patches.apply_all",
        "patch_tolist_cudagraph.py",
        "patch_inputs_embeds_optional.py",
        "py_compile",
    )
    return any(marker in argv for marker in skip_markers)


if _enabled() and not _is_pre_serve_patch_process():
    try:
        import instrument

        instrument.install()
    except Exception as exc:  # pragma: no cover - must never block boot.
        sys.stderr.write(f"[residency] sitecustomize install failed: {exc!r}\n")
