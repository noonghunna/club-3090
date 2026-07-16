"""club-3090 W4A8 guards — negative-scale fold + asym-checkpoint warning at load.

Injected at the chokepoint every Marlin route converges on
(MarlinLinearKernel.process_weights_after_loading — wNa16, gptq, awq, INC),
active ONLY when the 8-bit-activation path is engaged (is_a_8bit):

1. ASYMMETRIC (zero-point) weights + 8-bit activations → loud warning.
   avesed/vllm-ampere-optimized v0.3 documents garbage output for asym-weight
   W4A8; upstream vllm#24722 nominally supports the combination but it is
   unvalidated on this stack. W4A16 serving of asym checkpoints is untouched.

2. NEGATIVE group scales → warn + fix. Upstream's int8-act path quantizes
   group scales to int16 but the kernel reads them back unsigned
   (reinterpret_cast<uint16_t*>), so negative scales (valid in W4A16;
   AutoRound emits ~50%) corrupt every affected group. Fold at load: for
   groups with s<0, remap uint4b8 codes q -> clamp(16-q, 0, 15) (exact except
   q==0 -> 15: +7|s| instead of +8|s|, ~0.6% of weights on the Lorbus 27B)
   and set s -> |s|, BEFORE repack.

Idempotent, anchor-based, fails loudly on version skew.
"""
import importlib.util
import os

SENTINEL = "club-3090 W4A8 guards"

# Anchor 2: stock's pre-existing assert fires at the TOP of
# process_weights_after_loading — before our guard block — with a misleading
# message ("W8A8 is not supported"): what actually hits it is an ASYMMETRIC
# (zero-point) int4 checkpoint (weight_type uint4, not uint4b8) served with
# 8-bit activations. Replace the message with an actionable one.
# (Live-verified 2026-07-16: ct-format asym AWQ + int8 env → this assert.)
OLD_ASSERT = """        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "W8A8 is not supported by marlin kernel."
            )"""

NEW_ASSERT = """        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "club-3090 W4A8 guards: 8-bit activations "
                "(VLLM_MARLIN_INPUT_DTYPE) with weight type "
                f"{c.weight_type} are not supported on this Marlin path — "
                "ASYMMETRIC (zero-point) int4 checkpoints hit this. Drop "
                "VLLM_MARLIN_INPUT_DTYPE (asym checkpoints serve fine as "
                "W4A16) or use a POSITIVE-symmetric int4 checkpoint."
            )"""

OLD = """        def transform_w_q(x):"""

NEW = """        # club-3090 W4A8 guards — 8-bit-activation path only (this method is the
        # chokepoint every Marlin route converges on: wNa16, gptq, awq, INC).
        if is_a_8bit:
            from vllm.logger import init_logger as _c3_init_logger

            _c3_log = _c3_init_logger(__name__)

            # Guard 1: ASYMMETRIC (zero-point) weights + 8-bit activations.
            # avesed/vllm-ampere-optimized v0.3 documents garbage output for
            # asym-weight W4A8; upstream vllm#24722 nominally supports it but it
            # is unvalidated on this stack. Warn loudly instead of failing silent.
            if c.zero_points:
                _c3_log.warning_once(
                    "W4A8 (%s activations) with ASYMMETRIC zero-point weights: "
                    "known to produce degraded or garbage output "
                    "(avesed/vllm-ampere-optimized v0.3 release notes; upstream "
                    "vllm#24722 claims support but this combination is unvalidated "
                    "here). Verify output quality before trusting results — prefer "
                    "a POSITIVE-symmetric int4 checkpoint.",
                    str(c.act_type),
                )

            # Guard 2 + fix: NEGATIVE group scales. The int8-act kernel reads
            # int16-quantized group scales UNSIGNED, so negative scales (valid in
            # W4A16; AutoRound emits ~50%) corrupt every affected group. Fold the
            # sign into the uint4b8 codes (q -> 16-q, clamp 15) and take |s|.
            if c.weight_type == scalar_types.uint4b8:
                _w = getattr(layer, self.w_q_name)
                _s = getattr(layer, self.w_s_name)
                _neg = _s.data < 0
                if bool(_neg.any()):
                    # static text — a varying message (e.g. per-layer %) defeats
                    # warning_once dedup and spams one line per layer (252 on the
                    # 27B; live-verified 2026-07-16)
                    _c3_log.warning_once(
                        "W4A8: negative group scales detected (AutoRound-style "
                        "checkpoint) — folding sign into weight codes at load "
                        "(all affected layers; logged once). Exact except q==0 "
                        "codes in negative groups (clamped to q=15: +7|s| "
                        "instead of +8|s|)."
                    )
                    _qw = _w.data  # GPTQ packed [k/8, n] int32
                    _g = c.group_size if c.group_size != -1 else c.partition_weight_shape[0]
                    _rows = torch.arange(_qw.shape[0], device=_qw.device) * 8 // _g
                    _negrow = _neg[_rows]  # [k/8, n]
                    _out = torch.zeros_like(_qw)
                    for _i in range(8):
                        _nib = (_qw >> (4 * _i)) & 0xF
                        _folded = (16 - _nib).clamp(0, 15)
                        _nib = torch.where(_negrow, _folded, _nib)
                        _out |= _nib << (4 * _i)
                    _w.data = _out
                    _s.data = _s.data.abs()

        def transform_w_q(x):"""


def main():
    spec = importlib.util.find_spec("vllm")
    vllm_dir = os.path.dirname(spec.origin)
    target = os.path.join(vllm_dir, "model_executor/kernels/linear/mixed_precision/marlin.py")
    with open(target, encoding="utf-8") as f:
        s = f.read()
    if SENTINEL in s:
        print("[skip] already patched")
        return
    if "club-3090 negscale-fold" in s:
        raise SystemExit(
            "[FAIL] older negscale-fold-only version applied — restore "
            "marlin.py.negfold.bak first, then re-run"
        )
    for label, old in (("guard-block", OLD), ("asym-assert", OLD_ASSERT)):
        if s.count(old) != 1:
            raise SystemExit(f"[FAIL] {label} anchor found {s.count(old)}x, need exactly 1")
    if "import torch" not in s:
        raise SystemExit("[FAIL] no torch import in target")
    with open(target + ".negfold.bak", "w", encoding="utf-8") as f:
        f.write(s)
    s = s.replace(OLD_ASSERT, NEW_ASSERT).replace(OLD, NEW)
    with open(target, "w", encoding="utf-8") as f:
        f.write(s)
    print(f"[ok] patched {target} (asym-assert message + guard block)")


if __name__ == "__main__":
    main()
