#!/usr/bin/env python3
"""Overlay `qwen4exp-ple-fp8-bytecopy`: let the pinned-RAM PLE lookup run on sm_80/sm_86.

    patch_ple_fp8_bytecopy.py patch    apply (marker-gated; a second run is a verified no-op)
    patch_ple_fp8_bytecopy.py verify   check the installed result

Stock vLLM's `_lookup_ple_embedding_from_pinned_kernel` (models/qwen4_exp/common/
ngram_embedding.py) is a pure row copy, but it is launched on the FP8 table as
float8_e4m3fn pointers. Triton on Ampere has no fp8e4nv type, so the kernel fails to
compile ("type fp8e4nv not supported in this architecture") and the engine dies in the
memory-profiling forward. Measured on 2x CMP 170HX (sm_80), vLLM v0.31.0, albucino FP8 PLE.

Fix: launch the kernel on int8 views of the weight and the output when they are FP8. The
bytes copied are identical; dequantization still happens later in torch. No effect on
BF16 tables or on archs with native fp8e4nv.

Anchor must appear exactly once. Missing or not unique -> exit 1. Stdlib only.
"""
from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path

TAG = "[qwen4exp-ple-fp8-bytecopy]"
MARKER = "# [club3090 qwen4exp-ple-fp8-bytecopy]"
TARGET = Path("models/qwen4_exp/common/ngram_embedding.py")
ANCHOR = (
    "            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](\n"
    "                self._uva_weight,\n"
    "                flat_ids,\n"
    "                output,\n"
)
REPLACEMENT = (
    f"            {MARKER} byte-copy FP8 rows: Triton has no fp8e4nv on sm_80/86\n"
    "            _club_w, _club_o = self._uva_weight, output\n"
    "            if _club_w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):\n"
    "                _club_w, _club_o = _club_w.view(torch.int8), output.view(torch.int8)\n"
    "            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](\n"
    "                _club_w,\n"
    "                flat_ids,\n"
    "                _club_o,\n"
)


def die(msg: str) -> None:
    print(f"{TAG} REFUSE: {msg}", file=sys.stderr)
    print(f"{TAG} Fix: re-anchor for this vLLM version "
          "(models/qwen3.8-flash-next/vllm/patches/qwen4exp-ple-fp8-bytecopy/README.md).", file=sys.stderr)
    sys.exit(1)


def main(mode: str) -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        die("the vllm package is not importable")
    path = Path(spec.origin).resolve().parent / TARGET
    if not path.is_file():
        die(f"{TARGET} not found")
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        if REPLACEMENT not in text:
            die(f"{TARGET} carries the marker but not the expected edit")
        print(f"{TAG} already applied")
        return
    if mode == "verify":
        die(f"{TARGET} is not patched")
    if text.count(ANCHOR) != 1:
        die(f"anchor found {text.count(ANCHOR)} times in {TARGET} (expected 1)")
    tmp = path.with_suffix(".py.club3090.tmp")
    tmp.write_text(text.replace(ANCHOR, REPLACEMENT), encoding="utf-8")
    try:
        py_compile.compile(str(tmp), doraise=True)
    except py_compile.PyCompileError as exc:
        tmp.unlink(missing_ok=True)
        die(f"patched file does not compile: {exc}")
    tmp.replace(path)
    print(f"{TAG} applied: pinned PLE lookup copies FP8 rows as int8")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("patch", "verify"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
