"""Genesis PN25 worker-spawn registration fix (setup-time patch).

Filed upstream as
[Sandermage/genesis-vllm-patches#16](https://github.com/Sandermage/genesis-vllm-patches/issues/16).
This is our local backport while the upstream PR cycle plays out.

Why setup-time, not boot-time
-----------------------------

The genesis tree is mounted read-only into the vLLM container; a boot-time
sidecar can't write to it. The patch needs to land in the genesis checkout
BEFORE the container starts, which means at setup time. setup.sh invokes
this script after `git checkout <pin>` so the fix lands on every fresh
setup.

The bug
-------

`_register_op_once()` checks an in-process `_op_registered` flag. When
vLLM workers spawn (vLLM uses spawn, not fork), they get a fresh Python
interpreter — module-level state resets, including `_op_registered=False`.
The C++ `torch.library` registry also doesn't survive across spawn.

Worker's first call to the patched `SiluAndMul.forward_native` body calls
`get_op_callable()` which calls `_register_op_once()`. The flag is False,
so it tries to re-decorate via `@torch.library.custom_op(...)`. That call
internally invokes `torch.library.infer_schema(fn)` to derive the op
schema from type hints. **Dynamo refuses to trace through `infer_schema`**
(it's marked as skipped). And worker's first `forward_native` call IS
inside a dynamo `aot_compile_fullgraph` trace during profile_run. Engine
dies at:

    torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
      Explanation: Dynamo developers have intentionally marked that the
      function `infer_schema` ... should not be traced.

The fix
-------

Do not register from the compiled forward path at all. In PyTorch 2.11,
calling a `torch._dynamo.disable` wrapped function from vLLM's
`fullgraph=True` compile path is itself an Unsupported graph break.

Instead, patch Genesis PN25 so `activation.py` registers and caches the
opaque op at module import time in each spawned worker:

    _GENESIS_PN25_SILU_AND_MUL_OP = get_op_callable()

Worker module import happens during model construction, before
`profile_run` enters `aot_compile_fullgraph`. The patched
`forward_native` then calls only that cached module global and never runs
`torch.library.custom_op(...)` while Dynamo is tracing.
"""
import os
import sys

CUSTOMOP_TARGET = (
    "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/kernels/"
    "silu_and_mul_customop.py"
)
WIRING_TARGET = (
    "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/wiring/hybrid/"
    "patch_N25_silu_inductor_safe_pool.py"
)

MARKER_CUSTOMOP = "# club-3090: PN25 trace-safe get_op_callable v3"
MARKER_WIRING = "club-3090: PN25 import-time op cache v3"

OLD_GET_OP_CALLABLE = '''def get_op_callable():
    """Return the registered op callable, or None if registration failed.

    Used by the PN25 wiring patch to populate the replacement body.
    Caller is responsible for graceful degradation on None.
    """
    if not _register_op_once():
        return None
    try:
        return torch.ops.genesis.silu_and_mul_pooled
    except (AttributeError, RuntimeError):
        return None
'''

NEW_GET_OP_CALLABLE = f'''def get_op_callable():
    """Return the registered op callable, or None if registration failed.

    {MARKER_CUSTOMOP}

    Intended path: activation.py imports this module and calls this function
    while each spawned worker imports activation.py, before Dynamo tracing.
    If a stale forward body calls this from inside fullgraph tracing, avoid
    torch.library registration there because Dynamo cannot trace Library
    construction or infer_schema.
    """
    if not _op_registered:
        try:
            compiler = getattr(torch, "compiler", None)
            if compiler is not None and compiler.is_compiling():
                log.debug(
                    "[PN25] get_op_callable called during torch.compile "
                    "before registration; returning None"
                )
                return None
        except Exception:
            pass
        try:
            dynamo = getattr(torch, "_dynamo", None)
            if dynamo is not None and dynamo.is_compiling():
                log.debug(
                    "[PN25] get_op_callable called during Dynamo tracing "
                    "before registration; returning None"
                )
                return None
        except Exception:
            pass

    if not _register_op_once():
        return None
    try:
        return torch.ops.genesis.silu_and_mul_pooled
    except (AttributeError, RuntimeError):
        return None
'''

OLD_FORWARD_REPLACEMENT = '''PN25_FORWARD_NATIVE_REPLACEMENT = (
    "    @staticmethod\\n"
    "    def forward_native(x: torch.Tensor) -> torch.Tensor:\\n"
    "        \\"\\"\\"PyTorch-native — Genesis PN25 routes through opaque\\n"
    "        custom op so torch.compile/Inductor cannot inline the FFN\\n"
    "        intermediate alloc; pool from FFNIntermediateCache instead.\\n"
    "        Falls back to vanilla math when registration unavailable.\\n"
    "        \\"\\"\\"\\n"
    "        try:\\n"
    "            from vllm._genesis.kernels.silu_and_mul_customop import (\\n"
    "                get_op_callable as _genesis_pn25_get_op,\\n"
    "            )\\n"
    "            _genesis_pn25_op = _genesis_pn25_get_op()\\n"
    "            if _genesis_pn25_op is not None:\\n"
    "                return _genesis_pn25_op(x)\\n"
    "        except Exception:  # pragma: no cover — defensive fallback\\n"
    "            pass\\n"
    "        d = x.shape[-1] // 2\\n"
    "        return F.silu(x[..., :d]) * x[..., d:]\\n"
)
'''

NEW_FORWARD_REPLACEMENT = '''PN25_FORWARD_NATIVE_REPLACEMENT = (
    "    @staticmethod\\n"
    "    def forward_native(x: torch.Tensor) -> torch.Tensor:\\n"
    "        \\"\\"\\"PyTorch-native — Genesis PN25 routes through an opaque\\n"
    "        custom op cached at activation.py import time, before vLLM's\\n"
    "        aot_compile_fullgraph trace. Do not register custom ops here.\\n"
    "        \\"\\"\\"\\n"
    "        _genesis_pn25_op = _GENESIS_PN25_SILU_AND_MUL_OP\\n"
    "        if _genesis_pn25_op is not None:\\n"
    "            return _genesis_pn25_op(x)\\n"
    "        d = x.shape[-1] // 2\\n"
    "        return F.silu(x[..., :d]) * x[..., d:]\\n"
)
'''

IMPORT_CONSTANTS = f'''PN25_IMPORT_ANCHOR = (
    "logger = init_logger(__name__)\\n"
    "\\n"
    "\\n"
)

PN25_IMPORT_REPLACEMENT = (
    "logger = init_logger(__name__)\\n"
    "\\n"
    "# [Genesis PN25] {MARKER_WIRING}. Register/cache the opaque\\n"
    "# silu_and_mul op while activation.py is imported in each spawned\\n"
    "# worker, before vLLM enters aot_compile_fullgraph profile_run.\\n"
    "try:\\n"
    "    from vllm._genesis.kernels.silu_and_mul_customop import (\\n"
    "        get_op_callable as _genesis_pn25_get_op_callable,\\n"
    "    )\\n"
    "    _GENESIS_PN25_SILU_AND_MUL_OP = _genesis_pn25_get_op_callable()\\n"
    "except Exception:\\n"
    "    _GENESIS_PN25_SILU_AND_MUL_OP = None\\n"
    "\\n"
    "\\n"
)

'''

SUBPATCH_ANCHOR = '''        sub_patches=[
            TextPatch(
                name="pN25_silu_and_mul_forward_native_opaque",
'''

SUBPATCH_REPLACEMENT = '''        sub_patches=[
            TextPatch(
                name="pN25_silu_and_mul_import_time_register",
                anchor=PN25_IMPORT_ANCHOR,
                replacement=PN25_IMPORT_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="pN25_silu_and_mul_forward_native_opaque",
'''


def _read(path: str) -> str | None:
    if not os.path.isfile(path):
        print(f"[pn25_register_fix] target not found: {path}", file=sys.stderr)
        return None
    with open(path, "r") as f:
        return f.read()


def _write(path: str, src: str) -> None:
    with open(path, "w") as f:
        f.write(src)


def _patch_customop() -> bool:
    src = _read(CUSTOMOP_TARGET)
    if src is None:
        return False
    if MARKER_CUSTOMOP in src:
        print("[pn25_register_fix v3] customop guard already applied")
        return True
    if OLD_GET_OP_CALLABLE not in src:
        print(
            "[pn25_register_fix v3] customop get_op_callable anchor not found",
            file=sys.stderr,
        )
        return False
    _write(CUSTOMOP_TARGET, src.replace(OLD_GET_OP_CALLABLE, NEW_GET_OP_CALLABLE, 1))
    print("[pn25_register_fix v3] patched get_op_callable trace guard")
    return True


def _patch_wiring() -> bool:
    src = _read(WIRING_TARGET)
    if src is None:
        return False

    changed = False
    if MARKER_WIRING not in src:
        if OLD_FORWARD_REPLACEMENT not in src:
            print(
                "[pn25_register_fix v3] forward replacement anchor not found",
                file=sys.stderr,
            )
            return False
        src = src.replace(OLD_FORWARD_REPLACEMENT, NEW_FORWARD_REPLACEMENT, 1)
        src = src.replace(
            NEW_FORWARD_REPLACEMENT,
            NEW_FORWARD_REPLACEMENT + "\n" + IMPORT_CONSTANTS,
            1,
        )
        changed = True

    if "pN25_silu_and_mul_import_time_register" not in src:
        if SUBPATCH_ANCHOR not in src:
            print(
                "[pn25_register_fix v3] subpatch insertion anchor not found",
                file=sys.stderr,
            )
            return False
        src = src.replace(SUBPATCH_ANCHOR, SUBPATCH_REPLACEMENT, 1)
        changed = True

    if changed:
        _write(WIRING_TARGET, src)
        print("[pn25_register_fix v3] patched PN25 activation import-time cache")
    else:
        print("[pn25_register_fix v3] wiring already applied")
    return True


def main() -> int:
    ok_customop = _patch_customop()
    ok_wiring = _patch_wiring()
    if not (ok_customop and ok_wiring):
        return 1
    print(
        "[pn25_register_fix v3] done — PN25 registers in spawned workers "
        "before fullgraph tracing; forward_native no longer registers in-graph"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
