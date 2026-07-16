"""club-3090 W4A8 patch — thread VLLM_MARLIN_INPUT_DTYPE through the INC (auto-round) path.

Upstream gap (vLLM v0.24.0): auto-round checkpoints are forced onto the `inc`
quantization path by INCConfig.override_quantization_method, but
inc/schemes/inc_wna16_linear.py::_build_gptq_method constructs AutoGPTQLinearMethod
directly and never assigns `input_dtype` — unlike AutoGPTQConfig.get_quant_method
(auto_gptq.py) which does `quant_method.input_dtype = get_marlin_input_dtype(prefix)`.
Result: VLLM_MARLIN_INPUT_DTYPE=int8 (the stock W4A16->W4A8 serve-time toggle) is
silently ignored for every auto-round checkpoint.

Idempotent, anchor-based, fails loudly on version skew (style: avesed's
w4a8_int_marlin_ampere.py installer).
"""
import importlib.util
import os
import sys

SENTINEL = "club-3090 W4A8"

OLD = """            return AutoGPTQLinearMethod(
                AutoGPTQConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=self.layer_config.group_size,
                    desc_act=False,
                    is_sym=self.layer_config.sym,
                    lm_head_quantized=False,
                    dynamic={},
                    full_config={},
                )
            )"""

NEW = """            # club-3090 W4A8: thread the stock VLLM_MARLIN_INPUT_DTYPE toggle
            # (auto_gptq.py's get_quant_method assigns input_dtype; this direct
            # construction path forgot to — so auto-round ckpts silently stay W4A16)
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                get_marlin_input_dtype,
            )

            method = AutoGPTQLinearMethod(
                AutoGPTQConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=self.layer_config.group_size,
                    desc_act=False,
                    is_sym=self.layer_config.sym,
                    lm_head_quantized=False,
                    dynamic={},
                    full_config={},
                )
            )
            method.input_dtype = get_marlin_input_dtype()
            return method"""


def main():
    spec = importlib.util.find_spec("vllm")
    vllm_dir = os.path.dirname(spec.origin)
    target = os.path.join(
        vllm_dir, "model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py"
    )
    with open(target, encoding="utf-8") as f:
        s = f.read()
    if SENTINEL in s:
        print("[skip] already patched")
        return
    if s.count(OLD) != 1:
        raise SystemExit(f"[FAIL] anchor found {s.count(OLD)}x, need exactly 1 (version skew?)")
    with open(target + ".bak", "w", encoding="utf-8") as f:
        f.write(s)
    with open(target, "w", encoding="utf-8") as f:
        f.write(s.replace(OLD, NEW))
    print(f"[ok] patched {target}")


if __name__ == "__main__":
    main()
