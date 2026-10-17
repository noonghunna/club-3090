#!/usr/bin/env python3
"""Overlay `qwen4exp-qsa-fp8kv-ampere`: FP8 (e4m3) KV cache for Qwen3.8-Flash-Next on sm_80/86.

    patch_qsa_fp8kv_ampere.py patch    apply (marker-gated; a second run is a verified no-op)
    patch_qsa_fp8kv_ampere.py verify   check the installed result

Stock vLLM's QSA sparse-attention kernel (models/qwen4_exp/nvidia/ops/qsa.py) reads an FP8
KV cache through float8_e4m3fn pointers. Triton has no fp8e4nv type below sm_89, so on
Ampere `--kv-cache-dtype fp8_e4m3` fails to compile `_qsa_sparse_paged_gqa_splitk_kernel`
("type fp8e4nv not supported in this architecture"). Measured on 2x CMP 170HX (sm_80),
vLLM v0.31.0, 2026-10-17.

Fix, below sm_89 only: hand the kernel the cache as uint8 and decode e4m3 in registers.
Shifting the 7 magnitude bits left by 7 puts the e4m3 exponent and mantissa in the fp16
fields under the wrong bias (7 vs 15); one multiply by 2^8 fixes it. Exact for all 254
finite codes, subnormals included (fp16 subnormals are not flushed on sm_80/86). The
kernel's K/V scale handling is unchanged (stock folds them on the host). sm_89+ keeps
the stock native-fp8 path.

Decode by halt95 (Merlin, github.com/halt95/qwen38-flash-next-3090s, Apache-2.0,
patches/0001-qsa-FP8-E4M3-KV-reader-for-sm_86-with-static-K-V-sca.patch), ported to the
stock v0.31.0 kernel by club-3090. Co-authored-by: halt95 <halt95@users.noreply.github.com>

Each anchor must match its expected count. Otherwise exit 1. Stdlib only.
"""
from __future__ import annotations

import importlib.util
import py_compile
import re
import sys
from pathlib import Path

TAG = "[qwen4exp-qsa-fp8kv-ampere]"
MARKER = "# [club3090 qwen4exp-qsa-fp8kv-ampere]"
TARGET = Path("models/qwen4_exp/nvidia/ops/qsa.py")

HELPERS = f'''{MARKER} Triton has fp8e4nv only on sm_89+.
@lru_cache(maxsize=1)
def _club_fp8_as_bytes() -> bool:
    """Below sm_89, read an FP8 KV cache as uint8 and decode e4m3 in the kernel."""
    return not current_platform.has_device_capability(89)


@triton.jit
def _club_decode_e4m3_fp16(b):
    # e4m3 magnitude bits << 7 land in the fp16 exponent/mantissa fields under bias 7
    # instead of 15; x256 fixes the bias. Exact for every finite code (halt95/Merlin).
    b = b.to(tl.uint16)
    bits = ((b & 0x80) << 8) | ((b & 0x7F) << 7)
    return bits.to(tl.float16, bitcast=True) * 256.0


'''

# (anchor, replacement, expected count)
EDITS = [
    ('@triton.jit(do_not_specialize=["num_rows", "num_requests"])\ndef _qsa_sparse_paged_gqa_splitk_kernel(',
     HELPERS + '@triton.jit(do_not_specialize=["num_rows", "num_requests"])\ndef _qsa_sparse_paged_gqa_splitk_kernel(', 1),
    ("    IS_FP8: tl.constexpr,\n",
     "    IS_FP8: tl.constexpr,\n    FP8_AS_BYTES: tl.constexpr = False,\n", 1),
    ("            keys = keys.to(query.dtype)\n",
     "            if FP8_AS_BYTES:\n                keys = _club_decode_e4m3_fp16(keys).to(query.dtype)\n"
     "            else:\n                keys = keys.to(query.dtype)\n", 1),
    ("            values = values.to(tl.float16)\n",
     "            if FP8_AS_BYTES:\n                values = _club_decode_e4m3_fp16(values)\n"
     "            else:\n                values = values.to(tl.float16)\n", 1),
    ("    is_fp8 = k_cache.dtype == torch.float8_e4m3fn\n",
     "    is_fp8 = k_cache.dtype == torch.float8_e4m3fn\n"
     f"    {MARKER} present e4m3 as bytes below sm_89\n"
     "    fp8_bytes = is_fp8 and _club_fp8_as_bytes()\n"
     "    if fp8_bytes:\n"
     "        k_cache = k_cache.view(torch.uint8)\n"
     "        v_cache = v_cache.view(torch.uint8)\n", 1),
    ("    cache_dtype = torch.float8_e4m3fn if is_fp8 else key_cache.dtype\n",
     "    fp8_bytes = is_fp8 and _club_fp8_as_bytes()\n"
     "    cache_dtype = (torch.uint8 if fp8_bytes else torch.float8_e4m3fn) if is_fp8 else key_cache.dtype\n", 1),
]
LAUNCH_RE = re.compile(r"^( +)IS_FP8=is_fp8,\n", re.M)


def die(msg: str) -> None:
    print(f"{TAG} REFUSE: {msg}", file=sys.stderr)
    print(f"{TAG} Fix: re-anchor for this vLLM version "
          "(models/qwen3.8-flash-next/vllm/patches/qwen4exp-qsa-fp8kv-ampere/README.md).", file=sys.stderr)
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
        if "_club_decode_e4m3_fp16(values)" not in text or text.count("FP8_AS_BYTES=fp8_bytes") != 2:
            die(f"{TARGET} carries the marker but not the expected edits")
        print(f"{TAG} already applied")
        return
    if mode == "verify":
        die(f"{TARGET} is not patched")
    for anchor, repl, n in EDITS:
        c = text.count(anchor)
        if c != n:
            die(f"anchor {anchor.strip()[:50]!r} found {c} times (expected {n})")
        text = text.replace(anchor, repl)
    if len(LAUNCH_RE.findall(text)) != 2:
        die("expected 2 kernel launches passing IS_FP8=is_fp8")
    text = LAUNCH_RE.sub(lambda m: f"{m.group(1)}IS_FP8=is_fp8,\n{m.group(1)}FP8_AS_BYTES=fp8_bytes,\n", text)
    tmp = path.with_suffix(".py.club3090.tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(tmp), doraise=True)
    except py_compile.PyCompileError as exc:
        tmp.unlink(missing_ok=True)
        die(f"patched file does not compile: {exc}")
    tmp.replace(path)
    print(f"{TAG} applied: FP8 KV read as bytes + e4m3 decode below sm_89")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("patch", "verify"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
