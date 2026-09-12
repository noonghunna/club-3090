# SPDX-License-Identifier: Apache-2.0
"""SM86 FA2 attention with the stock vLLM 0.29.0 cache contract."""
import os
from dataclasses import dataclass

import torch
from fa2_prefill import Fa2Prefill
from paged_prefill import PagedPrefill
from vllm.platforms import current_platform
from vllm.v1.attention.backends import flashinfer as stock
from vllm.v1.kv_cache_layout import KVCacheLayout

torch.ops.load_library(os.environ["FA2_FP8KV_LIBRARY"])
torch.ops.load_library(os.environ["FA2_FP8KV_PREFILL_LIBRARY"])


@dataclass
class Metadata(stock.FlashInferMetadata):
    query_starts: torch.Tensor
    sequence_lengths: torch.Tensor
    pages: torch.Tensor
    max_query: int
    max_context: int
    prefill_context: int


class MetadataBuilder(stock.FlashInferMetadataBuilder):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return stock.AttentionCGSupport.UNIFORM_BATCH

    def _init_reorder_batch_threshold(self, reorder_batch_threshold=1,
                                    supports_spec_as_decode=False,
                                    supports_dcp_with_varlen=False):
        # FA2 handles the complete speculative block inside a fixed CUDA Graph.
        super()._init_reorder_batch_threshold(
            reorder_batch_threshold, supports_spec_as_decode=True,
            supports_dcp_with_varlen=False,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        if self.use_dcp:
            raise NotImplementedError("FA2 FP8 KV does not support DCP")
        decodes, prefills, _, _ = stock.split_decodes_and_prefills(
            common, decode_threshold=self.reorder_batch_threshold,
            require_uniform=True,
        )
        starts = common.query_start_loc_cpu[:common.num_reqs + 1]
        actual = int(starts[-1].item())
        decoded = int(starts[decodes].item())
        max_query = int((starts[1:] - starts[:-1]).max().item()) if common.num_reqs else 0
        context = 0
        if (common.causal and max_query > 64 and common.num_reqs == 1
                and common.seq_lens_cpu_upper_bound is not None):
            context = int(common.seq_lens_cpu_upper_bound[0].item())
        return Metadata(
            num_actual_tokens=actual, slot_mapping=common.slot_mapping,
            q_data_type_prefill=self.q_data_type_prefill,
            q_data_type_decode=self.q_data_type_decode,
            num_decodes=decodes, num_decode_tokens=decoded,
            num_prefills=prefills, num_prefill_tokens=actual - decoded,
            causal=common.causal, prefill=None, decode=None,
            use_cascade=False, cascade_wrapper=None,
            query_starts=common.query_start_loc[:common.num_reqs + 1],
            sequence_lengths=common.seq_lens[:common.num_reqs],
            pages=common.block_table_tensor[:common.num_reqs],
            max_query=max_query, max_context=self.model_config.max_model_len,
            prefill_context=context,
        )

    def use_cascade_attention(self, *args, **kwargs):
        return False


class Attention(stock.FlashInferImpl):
    @property
    def kv_cache_layout(self):
        # Backend advertises exactly this layout. In 0.29.0 the drafter's
        # copied CacheConfig misses the target's set_kv_cache_layout RPC.
        return KVCacheLayout.LBNHC

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if attn_metadata is None or attn_metadata.num_actual_tokens == 0:
            return output.fill_(0)
        if not (
            current_platform.is_device_capability(86)
            and (self.head_size, self.num_kv_heads) in ((256, 1), (256, 2), (128, 4))
            and query.dtype == torch.bfloat16 and self.dcp_world_size == 1
            and self.window_left in (-1, 2047) and not self.logits_soft_cap
            and self.kv_cache_dtype in ("fp8", "fp8_e4m3")
            and kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn)
            and stock.get_flashinfer_layout_string(self.kv_cache_layout) == "NHD"
        ):
            raise NotImplementedError("FA2 requires SM86, supported heads, BF16 Q and NHD E4M3 KV")
        assert output_scale is None and output_block_scale is None
        assert isinstance(attn_metadata, Metadata)
        count = attn_metadata.num_actual_tokens
        cache = kv_cache.view(torch.float8_e4m3fn).permute(
            *self.kv_cache_layout.layer_view_order
        )
        keys, values = cache.split(self.head_size, dim=-1)
        max_query = attn_metadata.max_query
        if (attn_metadata.prefill_context >= max_query > 64
                and attn_metadata.causal and self.window_left == -1
                and not torch.cuda.is_current_stream_capturing()):
            try:
                Fa2Prefill().forward(
                    query[:count], keys, values, attn_metadata.pages,
                    layer._k_scale, layer._v_scale,
                    attn_metadata.prefill_context, self.scale, output[:count],
                )
                stock.logger.info_once("FA2 prefill: bounded BF16 unpacking, persistent FP8 KV")
                return output
            except torch.OutOfMemoryError:
                stock.logger.warning_once("FA2 prefill workspace exhausted; using paged FA2")
            # Outside the except block so the failed workspace can be released.
            PagedPrefill().forward(
                query[:count], keys, values, attn_metadata.pages,
                layer._k_scale, layer._v_scale,
                attn_metadata.prefill_context, self.scale, output[:count],
            )
            return output
        grouped = (attn_metadata.causal and self.window_left == -1
                   and max_query * (query.shape[1] // self.num_kv_heads) <= 64)
        # These splits stay fixed across graph replays; lengths/pages are tensors.
        splits = (128 if attn_metadata.causal else 32) if max_query <= 64 else 1
        max_context = min(attn_metadata.max_context,
                          attn_metadata.pages.shape[1] * keys.shape[1])
        torch.ops.fa2_fp8kv.forward(
            query[:count], keys, values, output[:count],
            attn_metadata.query_starts, attn_metadata.sequence_lengths,
            attn_metadata.pages, layer._k_scale, layer._v_scale,
            max_query, max_context, attn_metadata.causal, self.window_left, -1,
            self.scale, splits, grouped,
        )
        return output


class Backend(stock.FlashInferBackend):
    @classmethod
    def get_supported_head_sizes(cls):
        return [128, 256]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype):
        return kv_cache_dtype in ("fp8", "fp8_e4m3")

    @staticmethod
    def get_impl_cls():
        return Attention

    @staticmethod
    def get_builder_cls():
        return MetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability):
        return capability.major == 8 and capability.minor == 6

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def supports_sink(cls):
        return False
