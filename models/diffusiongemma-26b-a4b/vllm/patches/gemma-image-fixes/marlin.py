# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    is_fp8_marlin_supported,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)

_MARLIN_K_PAD_MULTIPLE = 64


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class MarlinFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """
    FP8 Marlin kernel for GPUs that lack FP8 hardware support.
    Leverages the Marlin kernel for fast weight-only FP8 quantization.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        # Check if platform supports FP8 Marlin
        if not is_fp8_marlin_supported():
            return False, "FP8 Marlin requires compute capability 7.5 or higher"
        if envs.VLLM_BATCH_INVARIANT:
            return False, "FP8 Marlin not supported for batch invariant execution."
        if (
            compute_capability is not None
            and compute_capability >= 89
            and not envs.VLLM_TEST_FORCE_FP8_MARLIN
        ):
            return (
                False,
                "To apply FP8 Marlin on high-capability GPUs, please set "
                "VLLM_TEST_FORCE_FP8_MARLIN=1",
            )
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)
        self.marlin_input_dtype = None
        self.block_quant = self.config.weight_quant_key in {kFp8Static128BlockSym}
        self.size_k_first = not self.block_quant

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.block_quant:
            weight, weight_scale_inv = process_fp8_weight_block_strategy(
                layer.weight, layer.weight_scale_inv
            )
            # Update layer with new values
            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale_inv", weight_scale_inv.data)
        else:
            w_q, *_ = self._get_layer_params(layer)
            # Compressed tensors transposes the weight to (K, N)
            # for channel and tensor quant strategies.
            # So we can skip the transpose if the layout is
            # already (K, N).
            # TODO: Remove this check once the layouts have been
            # canonicalized to a standard (N, K) dimension. See issue
            # #33314 for more details.
            if w_q.shape != (
                layer.input_size_per_partition,
                layer.output_size_per_partition,
            ):
                # transpose the weights to (K,N)
                replace_parameter(
                    layer,
                    "weight",
                    w_q.t(),
                )

        part_size_k = layer.input_size_per_partition
        padded_size_k = _round_up(part_size_k, _MARLIN_K_PAD_MULTIPLE)
        if padded_size_k != part_size_k:
            if self.block_quant:
                raise RuntimeError(
                    "FP8 Marlin K-padding is only implemented for tensor/channel "
                    "W8A16 scales; block-FP8 scales need scale-row padding too."
                )

            part_size_n = layer.output_size_per_partition
            assert layer.weight.shape == (part_size_k, part_size_n)
            layer._marlin_orig_input_size_per_partition = part_size_k
            layer._marlin_k_pad = padded_size_k - part_size_k
            replace_parameter(
                layer,
                "weight",
                torch.nn.functional.pad(layer.weight, (0, 0, 0, layer._marlin_k_pad)),
            )
            layer.input_size_per_partition = padded_size_k
            logger.warning_once(
                "Padding FP8 Marlin K dimension from %d to %d for %s. "
                "Extra activation columns are zero-padded at apply time.",
                part_size_k,
                padded_size_k,
                layer.__class__.__name__,
            )
        else:
            layer._marlin_k_pad = 0

        layer.input_scale = None
        prepare_fp8_layer_for_marlin(
            layer, self.size_k_first, input_dtype=self.marlin_input_dtype
        )
        del layer.input_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.block_quant:
            weight_scale = layer.weight_scale_inv
        else:
            weight_scale = layer.weight_scale
        k_pad = getattr(layer, "_marlin_k_pad", 0)
        if k_pad:
            x = torch.nn.functional.pad(x, (0, k_pad))
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            input_dtype=self.marlin_input_dtype,
            bias=bias,
        )

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass
