#!/usr/bin/env python3
"""Vendored fix for vllm#51581 — dequantize qkv_proj before the DFlash fused-KV slice.

Upstream bug (vllm#51581, OPEN since 2026-08-09): DFlashQwen3Model builds its fused
KV buffers with

    kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]

`qwen3_dflash.py` has NO quantization awareness at all (zero references to
qweight / weight_packed / quant_method / scheme), so on a weight-quantized drafter
this either

  * raises `AttributeError: 'QKVParallelLinear' object has no attribute 'weight'`
    -- a hard EngineCore init failure, which is what v0.29.0 does with our shipped
    W4A16 drafter; or
  * silently CORRUPTS, when `.weight` exists but holds packed uint8/float8 rows:
    slicing those and handing them to F.linear produces garbage with no error.

This restores `_dense_kv_rows()`, which unpacks weight_packed/weight_scale and
dequantizes before the slice. It is the same fix the club-3090 dflash2 backport
carried; that backport ALSO vendored vllm#52816 (the DFlash2 model classes), which
IS native since v0.29.0 -- so this patch is the #52816 half stripped out and only
the still-unfixed #51581 half kept.

⚠️ Scope: compressed-tensors pack-quantized W4A16/W8A16 (symmetric, group) plus the
already-dense path. FP8 / NVFP4 / MXFP4 / GPTQ-AWQ drafters are NOT handled here --
gratex/Qwen3.8-27B-DFlash2-W4A16-g128-sym-GPTQ ships a broader `_dense_kv_rows`
covering those; borrow from it if we ever adopt an fp8 drafter.

Idempotent; anchor-checked; exits non-zero on drift so the entrypoint refuses to
boot a silently-unpatched configuration (a corrupt fused-KV path would otherwise
present as a merely slow drafter).
"""
import io
import py_compile
import sys
import tempfile

TARGET = ("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/"
          "qwen3_dflash.py")
MARKER = "_dense_kv_rows"
BUG = "        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]\n"
FIX = "        kv_weights = [_dense_kv_rows(a) for a in layers_attn]\n"
CLASS_ANCHOR = "@support_torch_compile\nclass DFlashQwen3Model"

HELPER = '''def _dense_kv_rows(attn: nn.Module) -> torch.Tensor:
    """Rows [q_size:] of the qkv projection as a dense bf16 matrix.

    club-3090 vendored fix for vllm#51581 (see patches.yml). For a
    compressed-tensors W4A16/W8A16 (pack-quantized, symmetric, group) qkv_proj this
    runs from load_weights, i.e. BEFORE the Marlin repack, so weight_packed /
    weight_scale are still in the plain checkpoint layout and can be dequantized
    here. An already-dense 2-D .weight short-circuits, so an unquantized drafter
    takes the stock path.
    """
    qkv = attn.qkv_proj
    w = getattr(qkv, "weight", None)
    if w is not None and w.dim() == 2:
        return w[attn.q_size:]
    packed, scale = qkv.weight_packed, qkv.weight_scale
    # (weight_shape holds only the last-loaded shard of a fused qkv; use the tensors.)
    out_f, in_f = int(packed.shape[0]), int(qkv.input_size)
    bits = 32 * packed.shape[1] // in_f
    from compressed_tensors.compressors.pack_quantized.base import unpack_from_int32
    q = unpack_from_int32(packed.data, bits, torch.Size([out_f, in_f]), packed_dim=1)
    group = in_f // scale.shape[1]
    dense = (q.to(torch.float32).reshape(out_f, in_f // group, group)
             * scale.to(torch.float32)[..., None]).reshape(out_f, in_f)
    out_dtype = scale.dtype if scale.dtype.is_floating_point else torch.bfloat16
    return dense.to(out_dtype)[attn.q_size:]


'''


def main() -> int:
    try:
        src = io.open(TARGET, encoding="utf-8").read()
    except OSError as e:
        print(f"[51581] REFUSE: cannot read target: {e}")
        return 1
    if MARKER in src:
        print("[51581] already patched (idempotent no-op)")
        return 0
    if src.count(BUG) != 1:
        print(f"[51581] REFUSE: anchor drift - expected exactly 1 fused-KV slice line, "
              f"found {src.count(BUG)}. Upstream may have fixed #51581; re-validate "
              f"before dropping this patch.")
        return 1
    if src.count(CLASS_ANCHOR) != 1:
        print(f"[51581] REFUSE: anchor drift - expected exactly 1 "
              f"'@support_torch_compile / class DFlashQwen3Model', "
              f"found {src.count(CLASS_ANCHOR)}")
        return 1
    out = src.replace(CLASS_ANCHOR, HELPER + CLASS_ANCHOR).replace(BUG, FIX)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(out)
        tmp = fh.name
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"[51581] REFUSE: patched file does not compile: {e}")
        return 1
    io.open(TARGET, "w", encoding="utf-8").write(out)
    print("[51581] applied: DFlash fused-KV slice now dequantizes qkv_proj "
          "(quantized drafters load instead of crashing/corrupting)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
