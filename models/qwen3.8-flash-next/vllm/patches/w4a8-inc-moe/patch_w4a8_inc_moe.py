#!/usr/bin/env python3
"""Overlay `w4a8-inc-moe`: int8 activations (W4A8) for Marlin MoE experts.

    patch_w4a8_inc_moe.py patch    apply (marker-gated; a second run is a verified no-op)
    patch_w4a8_inc_moe.py verify   check the installed result

`w4a8-int8-act` (qwen3.6-27b/vllm/patches) makes VLLM_MARLIN_INPUT_DTYPE=int8 work for
the dense Marlin linears of AutoRound checkpoints. The fused-MoE Marlin path has its
own bugs; on stock v0.31.0 an AutoRound MoE model such as Qwen3.8-Flash-Next outputs
"!!!!" with int8 on, and a compressed-tensors MoE drafter loses most of its acceptance.
Fixed here, each found on 2x CMP 170HX (sm_80) and checked with a kernel-level test
against a torch reference:

1. INC never sets the MoE input dtype. The INC GPTQ MoE builder (vLLM <= 0.32:
   `_resolve_gptq_moe` in inc/schemes/inc_wna16_scheme.py; from 0.33:
   `INCWNA16MoEScheme._build_gptq_method` in inc/schemes/inc_wna16_moe.py) builds
   AutoGPTQMoEMethod directly, so its input_dtype stays None and the experts are packed
   for 16-bit activations, while MarlinExperts reads the env itself and feeds them int8.
   Fix: set input_dtype from get_marlin_input_dtype(), as AutoGPTQConfig.get_quant_method
   does.
2. Inverted asserts. AutoGPTQMoEMethod (quantization/auto_gptq.py) requires
   `size_bits == 8` under int8 activations, with the message "W8A8-INT8 is not
   supported by marlin kernel". The dense path (utils/marlin_utils.py) requires uint4b8
   under the same message. Fix: require 4-bit weights.
3. The int8 global scale never reaches the kernel. With int8 activations,
   `_process_weights_marlin` divides the group scales by a per-layer factor
   (marlin_act_int8_process_scales) and the method registers it as
   w13/w2_input_global_scale, but neither AutoGPTQMoEMethod nor
   CompressedTensorsWNA16MoEMethod puts it in the fused-MoE quant config, so every
   expert output is off by that factor. Fix: pass it as the a1/a2 gscale in both.
4. Negative scales. Marlin's int8 path stores group scales as unsigned int16, and
   AutoRound emits about half of them negative. In `_process_weights_marlin`, when the
   4-bit weights have negative scales (only possible when symmetric; AutoRound still
   ships qzeros), rewrite each such group as q -> min(16 - q, 15), s -> |s| (uint4b8:
   s * (q - 8) == |s| * ((16 - q) - 8)). Exact except q == 0 in a negative group, the
   same trade as the dense fold. Runs on the GPU in chunks of 8 experts.

No effect without int8 activations. Every anchor must appear exactly once (scoped ones:
the first match after a reference line); missing or not unique -> exit 1, and nothing is written
unless every anchor matches. The INC file is picked by release: inc_wna16_moe.py when it exists
(vLLM 0.33+), else inc_wna16_scheme.py, so one script serves the pin and its rollback. Stdlib only.
"""
from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path

TAG = "[w4a8-inc-moe]"
MARKER = "# [club3090 w4a8-inc-moe]"

ORACLE = "model_executor/layers/fused_moe/oracle/int_wna16.py"
# vLLM <= 0.32 builds the INC GPTQ MoE method in inc_wna16_scheme.py; 0.33 moved it into a class
# in inc_wna16_moe.py (same logic, one indent deeper). targets() picks whichever exists.
INC_V032 = "model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py"
INC_V033 = "model_executor/layers/quantization/inc/schemes/inc_wna16_moe.py"
GPTQ = "model_executor/layers/quantization/auto_gptq.py"
CT = "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"

FOLD_HELPER = f'''


{MARKER} fold negative group scales for int8-activation Marlin MoE
def _club_moe_negfold(qw: torch.Tensor, s: torch.Tensor, group_size: int):
    """qw [E, K/8, N] int32 (K-first, 8 nibbles LSB-first), s [E, K/g, N]."""
    e_total, kp, n = qw.shape
    k = kp * 8
    g = group_size if group_size > 0 else k
    shifts = torch.arange(0, 32, 4, device=qw.device, dtype=torch.int32).view(1, 1, 8, 1)
    out = torch.empty_like(qw)
    for e0 in range(0, e_total, 8):
        q = qw[e0:e0 + 8]
        c = q.shape[0]
        nib = (q.unsqueeze(2) >> shifts) & 0xF
        neg = (s[e0:e0 + 8] < 0).repeat_interleave(g, dim=1)[:, :k].reshape(c, kp, 8, n)
        nib = torch.where(neg, (16 - nib).clamp(max=15), nib)
        packed = torch.zeros((c, kp, n), dtype=torch.int32, device=qw.device)
        for i in range(8):
            packed |= nib[:, :, i, :] << (4 * i)
        out[e0:e0 + 8] = packed
    return out, s.abs()


def _process_weights_marlin(
'''

FOLD_CALL_ANCHOR = (
    "    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1\n"
    "\n"
    "    marlin_w13_qweight: torch.Tensor\n"
)
FOLD_CALL = (
    f"    {MARKER} fold negative scales (AutoRound ships qzeros even when symmetric,\n"
    "    # so gate on the sign: a negative scale only exists in a symmetric checkpoint)\n"
    "    if input_dtype == torch.int8 and num_bits == 4:\n"
    "        if bool((w13_scales < 0).any()):\n"
    "            w13_qweight, w13_scales = _club_moe_negfold(w13_qweight, w13_scales, group_size)\n"
    "        if bool((w2_scales < 0).any()):\n"
    "            w2_qweight, w2_scales = _club_moe_negfold(w2_qweight, w2_scales, group_size)\n"
    + FOLD_CALL_ANCHOR
)

# Each edit: (old, new, after) -- `after` (or None) is text the old must follow.
EDITS: dict[str, list[tuple[str, str, str | None]]] = {
    ORACLE: [
        ("\n\ndef _process_weights_marlin(\n", FOLD_HELPER, None),
        (FOLD_CALL_ANCHOR, FOLD_CALL, None),
    ],
    INC_V032: [
        ("    if use_marlin:\n"
         "        return AutoGPTQMoEMethod(\n",
         "    if use_marlin:\n"
         f"        {MARKER} thread the int8 activation dtype into the INC MoE method\n"
         "        from vllm.model_executor.layers.quantization.utils.marlin_utils import (\n"
         "            get_marlin_input_dtype as _club_gmid,\n"
         "        )\n"
         "        _club_m = AutoGPTQMoEMethod(\n",
         None),
        ("            layer.moe_config,\n"
         "        )\n"
         "\n"
         "    moe_config = MoeWNA16Config.from_config(\n",
         "            layer.moe_config,\n"
         "        )\n"
         "        _club_m.input_dtype = _club_gmid()\n"
         "        return _club_m\n"
         "\n"
         "    moe_config = MoeWNA16Config.from_config(\n",
         "        _club_m = AutoGPTQMoEMethod(\n"),
    ],
    INC_V033: [
        ("        if use_marlin:\n"
         "            return AutoGPTQMoEMethod(\n",
         "        if use_marlin:\n"
         f"            {MARKER} thread the int8 activation dtype into the INC MoE method\n"
         "            from vllm.model_executor.layers.quantization.utils.marlin_utils import (\n"
         "                get_marlin_input_dtype as _club_gmid,\n"
         "            )\n"
         "            _club_m = AutoGPTQMoEMethod(\n",
         None),
        # the AWQ builder below ends the same way: scoped to the first match after the GPTQ one
        ("                layer.moe_config,\n"
         "            )\n"
         "\n"
         "        moe_config = MoeWNA16Config.from_config(\n",
         "                layer.moe_config,\n"
         "            )\n"
         "            _club_m.input_dtype = _club_gmid()\n"
         "            return _club_m\n"
         "\n"
         "        moe_config = MoeWNA16Config.from_config(\n",
         "            _club_m = AutoGPTQMoEMethod(\n"),
    ],
    GPTQ: [
        ("            assert self.quant_config.quant_type.size_bits == 8, (\n"
         "                \"W8A8-INT8 is not supported by marlin kernel.\"\n",
         f"            {MARKER} was == 8 (inverted; int4 x int8 is the supported pair)\n"
         "            assert self.quant_config.quant_type.size_bits == 4, (\n"
         "                \"W8A8-INT8 is not supported by marlin kernel.\"\n",
         None),
        ("        assert not is_a_8bit or self.quant_config.quant_type.size_bits == 8, (\n",
         f"        {MARKER} was == 8 (inverted)\n"
         "        assert not is_a_8bit or self.quant_config.quant_type.size_bits == 4, (\n",
         None),
        ("        return gptq_marlin_moe_quant_config(\n",
         f"        {MARKER} carry the int8 input global scales to MarlinExperts\n"
         "        _club_cfg = gptq_marlin_moe_quant_config(\n",
         None),
        ("            w2_bias=getattr(layer, \"w2_bias\", None),\n"
         "        )\n"
         "\n"
         "    def apply(\n",
         "            w2_bias=getattr(layer, \"w2_bias\", None),\n"
         "        )\n"
         "        _club_cfg._a1.alpha_or_gscale = getattr(layer, \"w13_input_global_scale\", None)\n"
         "        _club_cfg._a2.alpha_or_gscale = getattr(layer, \"w2_input_global_scale\", None)\n"
         "        return _club_cfg\n"
         "\n"
         "    def apply(\n",
         "        _club_cfg = gptq_marlin_moe_quant_config(\n"),
    ],
    CT: [
        ("            w2_bias=getattr(layer, \"w2_bias\", None),\n"
         "            gemm1_clamp_limit=getattr(layer, \"swiglu_limit\", None),\n",
         "            w2_bias=getattr(layer, \"w2_bias\", None),\n"
         f"            {MARKER} carry the int8 input global scales to MarlinExperts\n"
         "            a1_gscale=getattr(layer, \"w13_input_global_scale\", None),\n"
         "            a2_gscale=getattr(layer, \"w2_input_global_scale\", None),\n"
         "            gemm1_clamp_limit=getattr(layer, \"swiglu_limit\", None),\n",
         None),
    ],
}


def targets(root: Path) -> list[str]:
    """The files this release needs patched: the INC one depends on the vLLM version."""
    inc = INC_V033 if (root / INC_V033).is_file() else INC_V032
    return [ORACLE, inc, GPTQ, CT]


def die(msg: str) -> None:
    print(f"{TAG} REFUSE: {msg}", file=sys.stderr)
    print(f"{TAG} Fix: re-anchor for this vLLM version "
          "(models/qwen3.8-flash-next/vllm/patches/w4a8-inc-moe/README.md).", file=sys.stderr)
    sys.exit(1)


def edit(text: str, rel: str) -> str:
    for old, new, after in EDITS[rel]:
        start = 0
        if after is not None:
            start = text.find(after)
            if start < 0:
                die(f"{rel}: '{after.strip()}' not found")
        if after is None:
            n = text.count(old)
            if n != 1:
                die(f"{rel}: anchor found {n} times, expected 1: {old.strip().splitlines()[0]!r}")
        i = text.find(old, start)  # scoped edits take the first match after `after`
        if i < 0:
            die(f"{rel}: anchor not found: {old.strip().splitlines()[0]!r}")
        text = text[:i] + new + text[i + len(old):]
    return text


def main(mode: str) -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        die("the vllm package is not importable")
    root = Path(spec.origin).resolve().parent
    # marker checks cover only this release's targets: on 0.33 inc_wna16_scheme.py stays unmarked
    files = {rel: root / rel for rel in targets(root)}
    for rel, p in files.items():
        if not p.is_file():
            die(f"{rel} not found")
    texts = {rel: p.read_text() for rel, p in files.items()}
    marked = {rel for rel, t in texts.items() if MARKER in t}
    if marked and marked != set(files):
        die(f"only some files carry the marker ({sorted(marked)}); restore them from the image")
    if marked:
        for rel, t in texts.items():
            if any(new not in t for _, new, _ in EDITS[rel]):
                die(f"{rel} carries the marker but not every edit")
    elif mode == "verify":
        die("not patched")
    else:
        out = {rel: edit(t, rel) for rel, t in texts.items()}  # all anchors checked first
        for rel, t in out.items():
            files[rel].write_text(t)
            py_compile.compile(str(files[rel]), doraise=True)
    print(f"{TAG} {'verified' if mode == 'verify' else 'applied'}: Marlin MoE experts take "
          "int8 activations (INC dtype, asserts, global scales, negative-scale fold)",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("patch", "verify"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
