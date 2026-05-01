"""
Compile-safe PN12 — file-edit variant that survives spawn-multiprocessing (issue #16).

Problem: vLLM uses VLLM_WORKER_MULTIPROC_METHOD=spawn, which means engine
workers re-import vllm.model_executor.layers.activation from disk fresh.
Process-level monkey-patches on the main bash process don't reach them.

So we have to do what patch_pn12_ffn_pool_anchor.py does: regex-edit the
file on disk so each fresh subprocess gets the patched version on import.

What we patch:
  1. Inject an `import torch.library` + a `@torch.library.custom_op` block
     at the top of activation.py registering `club3090::pn12_silu_and_mul`.
     This is the OPAQUE op that Inductor cannot inline — it leaves a
     black-box call in the compiled graph and our pool-acquire runs at
     call time in eager Python.
  2. Replace SiluAndMul.forward_native body so it routes to the opaque op
     when GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1 + cuda + 2D input.

Why this matters: vLLM compile config uses `custom_ops=["none"]` under
Inductor, which makes SiluAndMul.__call__ dispatch to forward_native (not
forward_cuda). Inductor then traces the plain `F.silu(x[..., :d]) * x[..., d:]`
body and lowers it into a fused kernel that allocates the FFN intermediate
via `empty_strided_cuda((s18, 17408), ...)` ≈ 138 MiB. PN12's pool is
bypassed entirely. The earlier patch_pn12_ffn_pool_anchor.py only fixes
forward_cuda, so it doesn't reach this code path.

After this patch: forward_native calls the opaque op, the op body acquires
from FFNIntermediateCache._BUFFER_REGISTRY, and the 138 MiB allocation
is reused across layers instead of repeated per-call.

Idempotent: the patched marker is `LOCAL PN12 compile-safe custom op` —
if present in activation.py, this skip exits cleanly.
"""

import logging
import os
import re
import sys
from pathlib import Path

log = logging.getLogger("pn12_compile_safe")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

PATCH_TAG = "[pn12_compile_safe]"
PATCH_MARKER = "LOCAL PN12 compile-safe custom op"


def _find_target() -> Path:
    import vllm
    return (
        Path(vllm.__file__).resolve().parent
        / "model_executor"
        / "layers"
        / "activation.py"
    )


# The custom op + opaque-trace block we inject at the top of activation.py.
# It self-registers exactly once per Python process via a module-level guard,
# so multiple imports / re-imports don't double-register.
INJECTED_BLOCK = '''
# === LOCAL PN12 compile-safe custom op (club-3090 issue #16) ===
# Registers an opaque torch.library.custom_op that wraps the FFNIntermediateCache
# pool acquire + silu_and_mul kernel. Opaque to Inductor so the FFN
# intermediate buffer is reused from the pool instead of fresh-allocated
# during compiled forward.
#
# Also exposes _PN12_ENABLED as a module-level constant so forward_native can
# guard with a single if check that Dynamo treats as a static guard
# (constant-folded at trace time). Without this, Dynamo would compile BOTH
# branches of an `os.environ.get(...) == "1"` check; the else-branch's plain
# F.silu/mul lowers to empty_strided_cuda — defeating the whole patch.
import os as _PN12_os

_PN12_ENABLED = _PN12_os.environ.get("GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL") == "1"

if _PN12_ENABLED:
    import torch as _PN12_torch
    import torch.library as _PN12_lib

    if not hasattr(_PN12_torch.ops, "club3090") or not hasattr(
        _PN12_torch.ops.club3090, "pn12_silu_and_mul"
    ):
        try:
            from vllm._genesis.kernels.ffn_intermediate_cache import (
                FFNIntermediateCache as _PN12_Cache,
            )
            _PN12_silu_op = _PN12_torch.ops._C.silu_and_mul

            @_PN12_lib.custom_op(
                "club3090::pn12_silu_and_mul",
                mutates_args=(),
                device_types="cuda",
            )
            def _pn12_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
                d = x.shape[-1] // 2
                out = None
                if x.dim() == 2 and x.is_cuda:
                    try:
                        capturing = _PN12_torch.cuda.is_current_stream_capturing()
                        registry = getattr(_PN12_Cache, "_BUFFER_REGISTRY", None)
                        cached = (
                            registry.get((d, x.dtype, x.device))
                            if registry else None
                        )
                        if cached is not None and x.shape[0] <= cached.shape[0]:
                            out = cached[: x.shape[0]]
                        elif (
                            not capturing
                            and _PN12_Cache.is_production_eligible()
                        ):
                            out = _PN12_Cache.acquire_silu_out(
                                num_tokens=x.shape[0],
                                intermediate_size=d,
                                dtype=x.dtype,
                                device=x.device,
                            )
                    except Exception:
                        out = None
                if out is None:
                    out = _PN12_torch.empty(
                        x.shape[:-1] + (d,), dtype=x.dtype, device=x.device
                    )
                _PN12_silu_op(out, x)
                return out

            @_pn12_silu_and_mul.register_fake
            def _pn12_silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
                d = x.shape[-1] // 2
                return x.new_empty(x.shape[:-1] + (d,))
        except Exception:
            pass
# === END LOCAL PN12 compile-safe custom op ===
'''


# The replacement body for SiluAndMul.forward_native.
#
# Important: when PN12 is enabled, dispatch unconditionally to the opaque op.
# The original `if env_var and shape and is_cuda` guards moved to module-level
# constants so Dynamo treats them as static guards. A runtime guard on
# os.environ.get(...) inside forward_native causes Dynamo to compile BOTH
# branches; the else-branch's plain F.silu/mul lowers to empty_strided_cuda,
# defeating the patch. With _PN12_ENABLED as a module constant, Dynamo
# specializes on it at trace time — only one path compiles.
PATCHED_FORWARD_NATIVE = '''    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        # LOCAL PN12 compile-safe custom op route — see top of file.
        if _PN12_ENABLED:
            return torch.ops.club3090.pn12_silu_and_mul.default(x)
        # Fallback: original implementation (PN12 disabled)
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]
'''


def main() -> int:
    if os.environ.get("GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL") != "1":
        log.info("%s skip — GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL not set", PATCH_TAG)
        return 0

    target = _find_target()
    log.info("%s target: %s", PATCH_TAG, target)

    if not target.exists():
        log.warning("%s skip — target file missing", PATCH_TAG)
        return 0

    text = target.read_text()
    if PATCH_MARKER in text:
        log.info("%s already patched (idempotent skip)", PATCH_TAG)
        return 0

    # Step 1: inject the custom op registration block near the top, after
    # `import torch.nn.functional as F` (which we know exists in this file).
    F_IMPORT_RE = re.compile(r"^(import torch\.nn\.functional as F\n)", re.MULTILINE)
    if not F_IMPORT_RE.search(text):
        log.warning("%s skip — couldn't find 'import torch.nn.functional as F' anchor", PATCH_TAG)
        return 0
    text2 = F_IMPORT_RE.sub(r"\1" + INJECTED_BLOCK, text, count=1)
    if INJECTED_BLOCK not in text2:
        log.warning("%s skip — block injection failed", PATCH_TAG)
        return 0

    # Step 2: replace SiluAndMul.forward_native body. The current body is:
    #   @staticmethod
    #   def forward_native(x: torch.Tensor) -> torch.Tensor:
    #       """PyTorch-native implementation equivalent to forward()."""
    #       d = x.shape[-1] // 2
    #       return F.silu(x[..., :d]) * x[..., d:]
    FORWARD_NATIVE_RE = re.compile(
        r"    @staticmethod\n"
        r"    def forward_native\(x: torch\.Tensor\) -> torch\.Tensor:\n"
        r'        """PyTorch-native implementation equivalent to forward\(\)\."""\n'
        r"        d = x\.shape\[-1\] // 2\n"
        r"        return F\.silu\(x\[\.\.\., :d\]\) \* x\[\.\.\., d:\]\n",
        re.MULTILINE,
    )
    if not FORWARD_NATIVE_RE.search(text2):
        log.warning(
            "%s skip — couldn't find SiluAndMul.forward_native anchor (file format drifted)",
            PATCH_TAG,
        )
        return 0
    text3 = FORWARD_NATIVE_RE.sub(PATCHED_FORWARD_NATIVE, text2, count=1)
    if PATCHED_FORWARD_NATIVE.split("\n")[1] not in text3:
        log.warning("%s skip — forward_native replacement failed", PATCH_TAG)
        return 0

    target.write_text(text3)
    log.info("%s SiluAndMul.forward_native: applied", PATCH_TAG)
    log.info("%s patched %s", PATCH_TAG, target)
    log.info(
        "%s active: PN12 pool now reaches Inductor-compiled forward_native via opaque custom op",
        PATCH_TAG,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
