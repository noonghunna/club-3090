#!/usr/bin/env python3
"""
FlashInfer decode workspace-buffer unpin — fixes the MTP c>=4 Xid 31 VIRT_READ
crash on the Qwen3.8-27B W4A8+MTP (hybrid GDN) FastAPI serve config.

ROOT CAUSE (vllm-project/vllm#40756, thread comments 2026-08-17 — brasrox
retraction + sempai SM86 corroboration):
  FlashInfer plan() reuses ONE pinned host buffer per wrapper and copies it out
  asynchronously with nothing guarding it. The MTP drafter re-plans the DECODE
  wrapper K-1 times per step. A stale plan feeds the split-KV merge
  (PersistentVariableLengthMergeStatesKernel) a garbage row (observed: row
  33,621 of an 85-row buffer) -> Xid 31 MMU fault at an unmapped VA. It needs
  >=4 genuinely concurrent in-flight MTP requests to lose the race; c<=3 and
  the SPEC_N=0 (no drafter) path never do.

FIX (validated):
  Unpin the flashinfer-library decode workspace buffer. A pinned buffer is
  safe against the async copy-out only when the re-plan cannot overlap it; the
  MTP drafter breaks that. pin_memory=False forces a synchronous host->device
  copy of the plan each step, closing the window. Validated on v0.27.1 /
  flashinfer 0.6.16.post3 / 2x RTX 3090 (SM86): c=4 survives 425s clean
  (120s OK=126 + 300s OK=345, ERR=0, zero new Xid) at FULL async speed (no
  CUDA_LAUNCH_BLOCKING, no MAX_NUM_SEQS=1 cap). decode.py alone is necessary AND
  sufficient (A/B: prefill/sparse/pod unpin without decode.py still crashed).

WHAT THIS SCRIPT DOES:
  Flips `pin_memory=True,` -> `pin_memory=False,` in the
  `_pin_memory_int_workspace_buffer` allocations of flashinfer/decode.py (the
  DECODE wrapper — the one MTP re-plans). That is the validated minimal fix and
  the default. Set FI_PINQ_LIB_ALL=1 to ALSO unpin prefill.py/sparse.py/pod.py
  (the full mirror; also validated, kept for defense in depth).

SAFETY (matches this repo's "never serve unpatched" convention):
  - Auto-detects the flashinfer install via `import flashinfer` (no hardcoded
    path). If flashinfer is not installed (a non-FlashInfer backend config),
    this NO-OPS and exits 0 — it must not break a compose that doesn't use it.
  - Guard: in each file it touches, the number of `pin_memory=True,` tokens MUST
    equal the number of `_pin_memory_int_workspace_buffer* = torch.empty(`
    allocations. If they don't match, a `pin_memory=True,` exists OUTSIDE a
    workspace buffer and the blanket flip is no longer safe -> hard-fail (exit
    2) so the compose refuses to boot unpatched/mispatched rather than serve a
    half-applied state.
  - Idempotent: skips a file already carrying the marker.
  - py_compiles every file it writes; hard-fails on any post-write syntax error.

Environment:
  FI_PINQ_LIB_ALL=1   also unpin prefill/sparse/pod (default 0 = decode-only).
  FI_PINQ_LIB_ROOT=   override the flashinfer package dir (default: auto-detect).
"""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

MARKER = "# [club3090] pin_memory False — MTP c>=4 stale-plan race fix (vllm#40756)"
TOKEN = "pin_memory=True,"
TOKEN_FLIPPED = "pin_memory=False,"
ALLOC_RE = re.compile(r"_pin_memory_int_workspace_buffer[a-z_]* = torch\.empty\(")
# decode.py is the necessary+sufficient file (validated). The rest are the
# optional full-mirror (defense in depth).
PRIMARY = ("decode.py",)
MIRROR = ("prefill.py", "sparse.py", "pod.py")


def die(msg: str) -> None:
    print("[flashinfer-decode-pin] ERROR:", msg, file=sys.stderr)
    sys.exit(2)


def find_flashinfer_root() -> Path | None:
    override = os.environ.get("FI_PINQ_LIB_ROOT", "").strip()
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    try:
        import flashinfer  # noqa: F401
    except Exception:  # noqa: BLE001 - not installed -> caller no-ops
        return None
    if not getattr(flashinfer, "__file__", None):
        return None
    return Path(flashinfer.__file__).resolve().parent


def patch_one(root: Path, name: str) -> None:
    path = root / name
    if not path.exists():
        print(f"[flashinfer-decode-pin] {name}: not present — skipping.")
        return
    src = io.open(path, encoding="utf-8").read()

    if MARKER in src:
        print(f"[flashinfer-decode-pin] {name}: already patched — skipping.")
        return

    pins = src.count(TOKEN)
    if pins == 0:
        print(f"[flashinfer-decode-pin] {name}: no pinned workspace buffers — skipping.")
        return
    allocs = len(ALLOC_RE.findall(src))
    if allocs != pins:
        die(
            f"{name}: {pins} pin_memory tokens but {allocs} "
            f"_pin_memory_int_workspace_buffer allocs. A pin_memory=True exists "
            f"outside a workspace buffer — refusing to blanket-flip (drift). "
            f"Aborting."
        )

    new_src = src.replace(TOKEN, TOKEN_FLIPPED)
    new_src = new_src.replace("    def __init__(", MARKER + "\n    def __init__(", 1)
    if new_src.count(TOKEN) != 0:
        die(f"{name}: token remains after replace — internal error.")

    io.open(path, "w", encoding="utf-8").write(new_src)
    import py_compile

    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as e:  # noqa: BLE001
        die(f"{name}: py_compile failed after patch: {e}")
    print(f"[flashinfer-decode-pin] {name}: {pins} pin_memory True->False (py_compile OK).")


def main() -> None:
    root = find_flashinfer_root()
    if root is None:
        print("[flashinfer-decode-pin] flashinfer not installed — nothing to patch (no-op).")
        return
    if os.environ.get("FI_PINQ_LIB_ALL", "0") == "1":
        targets = PRIMARY + MIRROR
        scope = "full mirror (decode+prefill+sparse+pod)"
    else:
        targets = PRIMARY
        scope = "decode-only (minimal validated fix)"
    print(f"[flashinfer-decode-pin] root={root} scope={scope}")
    for name in targets:
        patch_one(root, name)
    print("[flashinfer-decode-pin] done.")


if __name__ == "__main__":
    main()
