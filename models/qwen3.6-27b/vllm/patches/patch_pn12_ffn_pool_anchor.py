"""
Local PN12 anchor repair for vLLM's SiluAndMul FFN intermediate pool patch.

Genesis PN12 is the right idea for Cliff 1 mechanism B, but the dev205+
activation.py layout drifted: the original Genesis anchor expects
SiluAndMul.forward_cuda to be followed directly by
@CustomOp.register("silu_and_mul_with_clamp"), while the current file has
the MulAndSilu section there instead. Genesis can therefore report PN12 as
"applied" while activation.py is still vanilla.

This sidecar applies the same env-gated pooled-output body with a local,
class-scoped regex anchor that matches the current SiluAndMul class without
editing Sandermage's Genesis tree.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger("pn12_ffn_pool_anchor_fix")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

PATCH_TAG = "[pn12_ffn_pool_anchor_fix]"
PATCH_MARKER = "LOCAL PN12 FFN intermediate pool anchor repair"


def _find_target() -> Path | None:
    try:
        import vllm

        candidate = (
            Path(vllm.__file__).resolve().parent
            / "model_executor"
            / "layers"
            / "activation.py"
        )
        if candidate.exists():
            return candidate
    except ImportError:
        pass

    for raw_path in (
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/activation.py",
        "/usr/lib/python3.12/site-packages/vllm/model_executor/layers/activation.py",
    ):
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate
    return None


SILU_FORWARD_RE = re.compile(
    r"^    def forward_cuda\(self, x: torch\.Tensor\) -> torch\.Tensor:\n"
    r"        d = x\.shape\[-1\] // 2\n"
    r"        output_shape = x\.shape\[:-1\] \+ \(d,\)\n"
    r"        out = torch\.empty\(output_shape, dtype=x\.dtype, device=x\.device\)\n"
    r"        self\.op\(out, x\)\n"
    r"        return out\n"
    r"\n"
    r"    def forward_xpu\(self, x: torch\.Tensor\) -> torch\.Tensor:\n"
    r"        return self\.forward_cuda\(x\)\n",
    re.MULTILINE,
)


SILU_FORWARD_REPLACEMENT = f"""    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # {PATCH_MARKER}.
        # Pool the [M, d] BF16 transient across layers instead of allocating
        # a fresh tensor per SiluAndMul call. Runtime behavior remains gated
        # by GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL.
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        try:
            from vllm._genesis.kernels.ffn_intermediate_cache import (
                FFNIntermediateCache as _LOCAL_PN12_Cache,
            )
            if _LOCAL_PN12_Cache.is_production_eligible() and x.dim() == 2:
                out = _LOCAL_PN12_Cache.acquire_silu_out(
                    num_tokens=x.shape[0],
                    intermediate_size=d,
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        except Exception:
            out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(x)
"""


def _class_body_bounds(src: str) -> tuple[int, int] | None:
    start = src.find('class SiluAndMul(CustomOp):')
    if start < 0:
        return None

    next_class = src.find("\n@CustomOp.register(", start + 1)
    if next_class < 0:
        return None
    return start, next_class


def _apply(src: str) -> tuple[str, str]:
    if PATCH_MARKER in src:
        return src, "skip-already-applied"

    # Genesis-side PN12 already applied (e.g., via a Genesis tree that has
    # the dev205+ anchor fix from PR #13). The pooled body is in place,
    # nothing for this sidecar to do.
    if "FFNIntermediateCache" in src and "Genesis PN12" in src:
        return src, "skip-genesis-pn12-applied"

    bounds = _class_body_bounds(src)
    if bounds is None:
        return src, "anchor-not-found-class"

    start, end = bounds
    class_body = src[start:end]
    patched_body, count = SILU_FORWARD_RE.subn(
        SILU_FORWARD_REPLACEMENT,
        class_body,
        count=1,
    )
    if count != 1:
        return src, "anchor-not-found-forward_cuda"

    return src[:start] + patched_body + src[end:], "applied"


def main() -> int:
    target = _find_target()
    if target is None:
        log.error("%s vLLM activation.py not found", PATCH_TAG)
        return 1

    log.info("%s target: %s", PATCH_TAG, target)
    src = target.read_text()
    patched, status = _apply(src)
    log.info("%s SiluAndMul.forward_cuda: %s", PATCH_TAG, status)

    if patched == src:
        if status in ("skip-already-applied", "skip-genesis-pn12-applied"):
            return 0
        log.error("%s no changes written; %s", PATCH_TAG, status)
        return 1

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(patched)
    os.replace(tmp_path, target)
    log.info("%s patched %s", PATCH_TAG, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
