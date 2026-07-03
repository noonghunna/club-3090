r"""
Compile-time auto-tuning block: 

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/sn/csn5am67odvgbktv5sswroffzb7ra7iqib3rf45f3ermt2yglekn.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp29 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (((-2) + x0) % 3)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp7
    tmp10 = tl.load(in_ptr0 + (x1 + 2*ks0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert(((0 <= tl.broadcast_to(tmp14, [XBLOCK])) & (tl.broadcast_to(tmp14, [XBLOCK]) < 1048576)) | ~(tmp9 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp14, [XBLOCK]) < 1048576")
    tmp16 = tl.load(in_ptr1 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3)) + 64*tmp14), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.full([1], 1, tl.int64)
    tmp18 = tmp0 >= tmp17
    tmp19 = (((-1) + x0) % 3)
    tmp20 = tmp19 == tmp6
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr0 + (ks0 + x1), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK]) < 1048576)) | ~(tmp21 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK]) < 1048576")
    tmp28 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3)) + 64*tmp26), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp31 = tmp29 + tmp30
    tmp32 = tmp29 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp29)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp33 < 1048576")
    tmp35 = tl.load(in_ptr1 + (x0 + 64*tmp33), xmask).to(tl.float32)
    tmp36 = tl.where(tmp21, tmp28, tmp35)
    tmp37 = tl.where(tmp9, tmp16, tmp36)
    tmp38 = tl.load(in_ptr1 + (34 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3)) + 64*tmp14), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (33 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3)) + 64*tmp26), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr1 + (32 + x0 + 64*tmp33), xmask).to(tl.float32)
    tmp41 = tl.where(tmp21, tmp39, tmp40)
    tmp42 = tl.where(tmp9, tmp38, tmp41)
    tl.store(out_ptr0 + (x2), tmp37, xmask)
    tl.store(out_ptr1 + (x2), tmp42, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/ez/cezh2iyzzx2igt47kgvkvo64ovxjmg6jqrq2amryzutvf3nchgnn.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_per_fused_1 = async_compile.triton('triton_per_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (r0_1 + 128*((x0 % 16)) + 6144*(x0 // 16)), xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl.full([1, 1], 128.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1, 1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = -tmp17
    tmp19 = libdevice.exp(tmp18)
    tmp20 = tl.full([1, 1], 1.0, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = (tmp17 / tmp21)
    tmp23 = tmp15 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 128*x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/nr/cnrmtkpftgk7a7wgdyhfkavkqpgngnqpydzy6wvpoimg5ievmd54.py
# Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul, float_2, mul_1, reshape_1, float_3, silu, mul_2, to, flatten, linear], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.clone, aten._unsafe_view, aten.silu, aten.t, aten.mm]
# Source node to ATen node mapping:
#   add => add_22
#   flatten => view_3
#   float_1 => convert_element_type
#   float_2 => convert_element_type_1
#   float_3 => convert_element_type_2
#   linear => mm, permute
#   mean => mean
#   mul => mul_32
#   mul_1 => mul_35
#   mul_2 => mul_40
#   pow_1 => pow_1
#   reshape => view
#   reshape_1 => clone, view_1
#   rsqrt => rsqrt
#   silu => add_35, div, exp, neg
#   to => convert_element_type_3
# Graph fragment:
#   %convert_element_type_3 : Tensor "f16[16*s18, 128][128, 1]cuda:1" = PlaceHolder[target=convert_element_type_3]
#   %view : Tensor "f16[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 128]), kwargs = {})
#   %convert_element_type : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[16*s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_22 : Tensor "f32[16*s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[16*s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_32 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[128][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %mul_35 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %convert_element_type_1), kwargs = {})
#   %clone : Tensor "f16[s18, 16, 128][2048, 128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%arg2_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_1 : Tensor "f16[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [%mul_6, 128]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_2,), kwargs = {})
#   %exp : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_35 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, %add_35), kwargs = {})
#   %mul_40 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %div), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_40, torch.float16), kwargs = {})
#   %view_3 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_3, [%arg7_1, 2048]), kwargs = {})
#   %permute : Tensor "f16[2048, 2048][1, 2048]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg5_1, [1, 0]), kwargs = {})
#   %mm : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_3, %permute), kwargs = {})
#   return %buf2
triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128*((((x0 + 2048*x1) // 128) % (16*ks0))) + ((x0 % 128))), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/ao/cao6ytwzbe4uxdzi4zculnhrjkayvnde3v7knrdcojg6jqatgz7f.py
# Topologically Sorted Source Nodes: [float_4, add_1, fused_add_rms_norm_default, moe_forward_shared], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, vllm.moe_forward_shared]
# Source node to ATen node mapping:
#   add_1 => add_67
#   float_4 => convert_element_type_6
#   fused_add_rms_norm_default => add_tensor_4, add_tensor_5, convert_element_type_default_11, convert_element_type_default_8, convert_element_type_default_9, mean_dim_3, mul_tensor_6, mul_tensor_7, pow_tensor_scalar_3, rsqrt_default_3
#   moe_forward_shared => moe_forward_shared
# Graph fragment:
#   %all_reduce : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce]
#   %arg9_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg9_1]
#   %buf6 : Tensor "f32[s18, 1][1, s18]cuda:1" = PlaceHolder[target=buf6]
#   %arg8_1 : Tensor "f16[2048][1]cuda:1" = PlaceHolder[target=arg8_1]
#   %convert_element_type_6 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.float32), kwargs = {})
#   %add_67 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 1.0), kwargs = {})
#   %convert_element_type_default_8 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce, torch.float32), kwargs = {})
#   %convert_element_type_default_9 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_8, %convert_element_type_default_9), kwargs = {})
#   %pow_tensor_scalar_3 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_4, 2), kwargs = {})
#   %mean_dim_3 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_3, [-1], True), kwargs = {})
#   %add_tensor_5 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_3, 1e-06), kwargs = {})
#   %rsqrt_default_3 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_tensor_6 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_4, %rsqrt_default_3), kwargs = {})
#   %mul_tensor_7 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_6, %add_67), kwargs = {})
#   %convert_element_type_default_11 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_7, torch.float16), kwargs = {})
#   %moe_forward_shared : [num_users=2] = call_function[target=torch.ops.vllm.moe_forward_shared.default](args = (%convert_element_type_default_11, %convert_element_type_default_11, %convert_element_type_default_11, None, %arg10_1, 0), kwargs = {})
#   return %buf6,%buf7,%buf8,%buf9
triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 268439552}}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 2048.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp19 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp25, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp25, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 2048*x0), tmp25, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/tz/ctz6a4jvvt3fqr4r6s5jcwhifxzxninp3zhavn7gx6i5wkl4du6r.py
# Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_2 => add_83
# Graph fragment:
#   %getitem_2 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=getitem_2]
#   %getitem_3 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=getitem_3]
#   %add_83 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, %getitem_3), kwargs = {})
#   return %add_83
triton_poi_fused_add_4 = async_compile.triton('triton_poi_fused_add_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 134217728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/ju/cjupfiv3i26vwt6btkvylvkf2hf6cwobvwabplre2a4dorhr3dyn.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_default, float_5, add_3, fused_add_rms_norm_default_1, marlin_gemm], Original ATen: [vllm_ir.fused_add_rms_norm, aten._to_copy, aten.add, _C.marlin_gemm, aten.copy_]
# Source node to ATen node mapping:
#   add_3 => add_93
#   float_5 => convert_element_type_7
#   fused_add_rms_norm_default => add_tensor_4, convert_element_type_default_10, convert_element_type_default_8, convert_element_type_default_9
#   fused_add_rms_norm_default_1 => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_7, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
#   marlin_gemm => marlin_gemm
# Graph fragment:
#   %all_reduce_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce_1]
#   %all_reduce : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce]
#   %arg9_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg9_1]
#   %buf17 : Tensor "f32[s18, 1][1, s18]cuda:1" = PlaceHolder[target=buf17]
#   %arg11_1 : Tensor "f16[2048][1]cuda:1" = PlaceHolder[target=arg11_1]
#   %copy_ : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=copy_]
#   %convert_element_type_default_8 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce, torch.float32), kwargs = {})
#   %convert_element_type_default_9 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_8, %convert_element_type_default_9), kwargs = {})
#   %convert_element_type_default_10 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_4, torch.float16), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg11_1, torch.float32), kwargs = {})
#   %add_93 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_7, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce_1, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_10, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.float16), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_2), kwargs = {})
#   %mul_tensor_5 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_4, %add_93), kwargs = {})
#   %convert_element_type_default_7 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_5, torch.float16), kwargs = {})
#   %marlin_gemm : Tensor "f16[s18, 4608][4608, 1]cuda:1"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%convert_element_type_default_7, None, %arg12_1, None, %arg13_1, None, None, None, None, None, %arg14_1, 2814749767172868, %arg7_1, 4608, 2048, True, False, True, False), kwargs = {})
#   %copy_ : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg6_1, %all_reduce), kwargs = {})
#   return %buf17,%convert_element_type_default_6,%buf19,%buf34
triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5 = async_compile.triton('triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5', 'mutated_arg_names': ['out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 301993984}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 2048.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.full([1, 1], 1.0, tl.float32)
        tmp34 = tmp32 + tmp33
        tmp35 = tmp30 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp36, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 2048*x0), tmp17, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/hy/chyez7emnmnqavn4gzjc6eo2l3ngxuo7lmo656jdaqscfd7qbtcv.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 8)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 4608*x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/bj/cbjcfhyz453mv4nsfd2kj7nsvtfqfeleeqew3bleie4ghxopa5d5.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_7 = async_compile.triton('triton_red_fused_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_7', 'mutated_arg_names': [], 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_7(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = (xindex % 8)
        x1 = xindex // 8
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x0 + 4608*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(out_ptr0 + (x3), tmp4, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x4 = xindex
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_5 = r0_index
            tmp6 = tl.load(in_ptr0 + (4096 + r0_5 + 4608*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr1 + (x4), tmp10, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 4608), (4608, 1), device='cuda:1', dtype=torch.float16)
    arg_1 = rand_strided((8192, 8, 1), (8, 1, 65536), device='cuda:1', dtype=torch.float32)
    arg_2 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:1', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 65536, 8192,


def call(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        stream1 = get_raw_stream(1)
        triton_red_fused_7.run(*args, stream=stream1)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        return triton_red_fused_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/wb/cwb7fe4vcrvy27gznbwe4qyuykmfdqwhls62c5kmcgzmieqjm65j.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(xnumel_3, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 8)
        x2 = xindex // 512
        x3 = xindex // 64
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 32, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (512*x1 + 4608*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.full([1], 256.0, tl.float32)
        tmp9 = (tmp7 / tmp8)
        tmp10 = tl.full([1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp6 * tmp12
        tmp14 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (32*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tl.load(in_ptr0 + (32 + 512*x1 + 4608*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 * tmp12
        tmp25 = tl.load(in_ptr2 + (32 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 + tmp16
        tmp28 = tmp24 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (32*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tmp29 * tmp30
        tmp32 = tmp21 - tmp31
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp4, tmp32, tmp33)
        tmp35 = tmp0 >= tmp3
        tmp36 = tl.full([1], 64, tl.int64)
        tmp37 = tmp0 < tmp36
        tmp38 = tl.load(in_ptr0 + (32 + 512*x1 + 4608*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tl.load(in_ptr1 + (x3), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full([1], 256.0, tl.float32)
        tmp42 = (tmp40 / tmp41)
        tmp43 = tl.full([1], 1e-06, tl.float32)
        tmp44 = tmp42 + tmp43
        tmp45 = libdevice.rsqrt(tmp44)
        tmp46 = tmp39 * tmp45
        tmp47 = tl.load(in_ptr2 + (32 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tl.full([1], 1.0, tl.float32)
        tmp50 = tmp48 + tmp49
        tmp51 = tmp46 * tmp50
        tmp52 = tmp51.to(tl.float32)
        tmp53 = tl.load(in_ptr3 + (32*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tmp52 * tmp53
        tmp55 = tl.load(in_ptr0 + (512*x1 + 4608*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tmp56 * tmp45
        tmp58 = tl.load(in_ptr2 + ((-32) + x0), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tmp58.to(tl.float32)
        tmp60 = tmp59 + tmp49
        tmp61 = tmp57 * tmp60
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tl.load(in_ptr4 + (32*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp64 = tmp62 * tmp63
        tmp65 = tmp54 + tmp64
        tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
        tmp67 = tl.where(tmp35, tmp65, tmp66)
        tmp68 = tl.where(tmp4, tmp34, tmp67)
        tl.store(out_ptr0 + (x0 + 256*x3), tmp68, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x4 = (xindex % 192)
        x5 = ((xindex // 192) % 8)
        x6 = xindex // 1536
        x7 = xindex // 192
        tmp69 = tl.load(in_ptr0 + (64 + x4 + 512*x5 + 4608*x6), xmask).to(tl.float32)
        tmp71 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last')
        tmp78 = tl.load(in_ptr2 + (64 + x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp70 = tmp69.to(tl.float32)
        tmp72 = tl.full([1], 256.0, tl.float32)
        tmp73 = (tmp71 / tmp72)
        tmp74 = tl.full([1], 1e-06, tl.float32)
        tmp75 = tmp73 + tmp74
        tmp76 = libdevice.rsqrt(tmp75)
        tmp77 = tmp70 * tmp76
        tmp79 = tmp78.to(tl.float32)
        tmp80 = tl.full([1], 1.0, tl.float32)
        tmp81 = tmp79 + tmp80
        tmp82 = tmp77 * tmp81
        tmp83 = tmp82.to(tl.float32)
        tl.store(out_ptr1 + (x4 + 256*x7), tmp83, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x8 = (xindex % 64)
        x9 = xindex // 64
        tmp84 = x8
        tmp85 = tl.full([1], 0, tl.int64)
        tmp86 = tmp84 >= tmp85
        tmp87 = tl.full([1], 32, tl.int64)
        tmp88 = tmp84 < tmp87
        tmp89 = tl.load(in_ptr0 + (4096 + 4608*x9 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp90 = tmp89.to(tl.float32)
        tmp91 = tl.load(in_ptr5 + (x9), tmp88 & xmask, eviction_policy='evict_last', other=0.0)
        tmp92 = tl.full([1], 256.0, tl.float32)
        tmp93 = (tmp91 / tmp92)
        tmp94 = tl.full([1], 1e-06, tl.float32)
        tmp95 = tmp93 + tmp94
        tmp96 = libdevice.rsqrt(tmp95)
        tmp97 = tmp90 * tmp96
        tmp98 = tl.load(in_ptr6 + (x8), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tmp98.to(tl.float32)
        tmp100 = tl.full([1], 1.0, tl.float32)
        tmp101 = tmp99 + tmp100
        tmp102 = tmp97 * tmp101
        tmp103 = tmp102.to(tl.float32)
        tmp104 = tl.load(in_ptr3 + (32*x9 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp105 = tmp103 * tmp104
        tmp106 = tl.load(in_ptr0 + (4128 + 4608*x9 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp107 = tmp106.to(tl.float32)
        tmp108 = tmp107 * tmp96
        tmp109 = tl.load(in_ptr6 + (32 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp110 = tmp109.to(tl.float32)
        tmp111 = tmp110 + tmp100
        tmp112 = tmp108 * tmp111
        tmp113 = tmp112.to(tl.float32)
        tmp114 = tl.load(in_ptr4 + (32*x9 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp115 = tmp113 * tmp114
        tmp116 = tmp105 - tmp115
        tmp117 = tl.full(tmp116.shape, 0.0, tmp116.dtype)
        tmp118 = tl.where(tmp88, tmp116, tmp117)
        tmp119 = tmp84 >= tmp87
        tmp120 = tl.full([1], 64, tl.int64)
        tmp121 = tmp84 < tmp120
        tmp122 = tl.load(in_ptr0 + (4128 + 4608*x9 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp123 = tmp122.to(tl.float32)
        tmp124 = tl.load(in_ptr5 + (x9), tmp119 & xmask, eviction_policy='evict_last', other=0.0)
        tmp125 = tl.full([1], 256.0, tl.float32)
        tmp126 = (tmp124 / tmp125)
        tmp127 = tl.full([1], 1e-06, tl.float32)
        tmp128 = tmp126 + tmp127
        tmp129 = libdevice.rsqrt(tmp128)
        tmp130 = tmp123 * tmp129
        tmp131 = tl.load(in_ptr6 + (32 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp132 = tmp131.to(tl.float32)
        tmp133 = tl.full([1], 1.0, tl.float32)
        tmp134 = tmp132 + tmp133
        tmp135 = tmp130 * tmp134
        tmp136 = tmp135.to(tl.float32)
        tmp137 = tl.load(in_ptr3 + (32*x9 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp138 = tmp136 * tmp137
        tmp139 = tl.load(in_ptr0 + (4096 + 4608*x9 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp140 = tmp139.to(tl.float32)
        tmp141 = tmp140 * tmp129
        tmp142 = tl.load(in_ptr6 + ((-32) + x8), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp143 = tmp142.to(tl.float32)
        tmp144 = tmp143 + tmp133
        tmp145 = tmp141 * tmp144
        tmp146 = tmp145.to(tl.float32)
        tmp147 = tl.load(in_ptr4 + (32*x9 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp148 = tmp146 * tmp147
        tmp149 = tmp138 + tmp148
        tmp150 = tl.full(tmp149.shape, 0.0, tmp149.dtype)
        tmp151 = tl.where(tmp119, tmp149, tmp150)
        tmp152 = tl.where(tmp88, tmp118, tmp151)
        tl.store(out_ptr2 + (x8 + 256*x9), tmp152, xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_3
        x10 = (xindex % 192)
        x11 = xindex // 192
        tmp153 = tl.load(in_ptr0 + (4160 + x10 + 4608*x11), xmask).to(tl.float32)
        tmp155 = tl.load(in_ptr5 + (x11), xmask, eviction_policy='evict_last')
        tmp162 = tl.load(in_ptr6 + (64 + x10), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp154 = tmp153.to(tl.float32)
        tmp156 = tl.full([1], 256.0, tl.float32)
        tmp157 = (tmp155 / tmp156)
        tmp158 = tl.full([1], 1e-06, tl.float32)
        tmp159 = tmp157 + tmp158
        tmp160 = libdevice.rsqrt(tmp159)
        tmp161 = tmp154 * tmp160
        tmp163 = tmp162.to(tl.float32)
        tmp164 = tl.full([1], 1.0, tl.float32)
        tmp165 = tmp163 + tmp164
        tmp166 = tmp161 * tmp165
        tmp167 = tmp166.to(tl.float32)
        tl.store(out_ptr3 + (x10 + 256*x11), tmp167, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 4608), (4608, 1), device='cuda:1', dtype=torch.float16)
    arg_1 = rand_strided((8192, 8, 1), (8, 1, 65536), device='cuda:1', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:1', dtype=torch.float16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:1', dtype=torch.float16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:1', dtype=torch.float16)
    arg_5 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:1', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:1', dtype=torch.float16)
    arg_7 = rand_strided((8192, 8, 64), (2048, 256, 1), device='cuda:1', dtype=torch.float16)
    arg_8 = rand_strided((8192, 8, 192), (2048, 256, 1), device='cuda:1', dtype=torch.float16)
    arg_9 = rand_strided((8192, 1, 64), (256, 256, 1), device='cuda:1', dtype=torch.float16)
    arg_10 = rand_strided((8192, 1, 192), (256, 256, 1), device='cuda:1', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 4194304, 12582912, 524288, 1572864,


def call(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        stream1 = get_raw_stream(1)
        triton_poi_fused_8.run(*args, stream=stream1)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        return triton_poi_fused_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(1):
    stream1 = get_raw_stream(1)
stream1 = get_raw_stream(1)
arg18_1 = generate_example_value((3, 8192), (8193, 1), 'cuda:1', torch.int64, 0, (3, 8192))
arg17_1 = generate_example_value((1048576, 64), (64, 1), 'cuda:1', torch.float16, 0, (1048576, 64))
buf24 = generate_example_value((8192, 32), (32, 1), 'cuda:1', torch.float16, 0, (8192, 32))
buf25 = generate_example_value((8192, 32), (32, 1), 'cuda:1', torch.float16, 0, (8192, 32))
with torch.cuda._DeviceGuard(1):
    triton_poi_fused_0.run(arg18_1, arg17_1, buf24, buf25, 8193, 262144, stream=stream1)
del arg18_1, arg17_1

stream1 = get_raw_stream(1)
arg0_1 = generate_example_value((8192, 16, 128), (2048, 128, 1), 'cuda:1', torch.float16, 0, (8192, 16, 128))
arg3_1 = generate_example_value((128,), (1,), 'cuda:1', torch.float16, 0, (128,))
arg2_1 = generate_example_value((8192, 16, 128), (6144, 128, 1), 'cuda:1', torch.float16, 0, (8192, 16, 128))
buf1 = generate_example_value((131072, 128), (128, 1), 'cuda:1', torch.float16, 0, (131072, 128))
with torch.cuda._DeviceGuard(1):
    triton_per_fused_1.run(arg0_1, arg3_1, arg2_1, buf1, 131072, 128, stream=stream1)
del arg0_1, arg3_1, arg2_1

stream1 = get_raw_stream(1)
buf2 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
with torch.cuda._DeviceGuard(1):
    triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2.run(buf1, buf2, 8192, 16777216, stream=stream1)
del buf1, buf2

stream1 = get_raw_stream(1)
buf5 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
arg9_1 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
arg8_1 = generate_example_value((2048,), (1,), 'cuda:1', torch.float16, 0, (2048,))
buf7 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
buf8 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
buf9 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
with torch.cuda._DeviceGuard(1):
    triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3.run(buf5, arg9_1, arg8_1, buf7, buf8, buf9, 8192, 2048, stream=stream1)
del arg8_1, buf7, buf8, buf9

stream1 = get_raw_stream(1)
buf13 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
buf12 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
with torch.cuda._DeviceGuard(1):
    triton_poi_fused_add_4.run(buf13, buf12, 16777216, stream=stream1)
del buf13, buf12

stream1 = get_raw_stream(1)
buf15 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
arg11_1 = generate_example_value((2048,), (1,), 'cuda:1', torch.float16, 0, (2048,))
buf16 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
buf19 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
arg6_1 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
with torch.cuda._DeviceGuard(1):
    triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5.run(buf15, buf5, arg9_1, arg11_1, buf16, buf19, arg6_1, 8192, 2048, stream=stream1)
del buf5, arg9_1, buf15, arg11_1, buf16, buf19, arg6_1

stream1 = get_raw_stream(1)
buf21 = generate_example_value((8192, 4608), (4608, 1), 'cuda:1', torch.float16, 0, (8192, 4608))
buf22 = generate_example_value((8192, 8, 256), (2048, 256, 1), 'cuda:1', torch.float16, 0, (8192, 8, 256))
with torch.cuda._DeviceGuard(1):
    triton_poi_fused_6.run(buf21, buf22, 16777216, stream=stream1)
del buf22

stream1 = get_raw_stream(1)
buf23 = generate_example_value((8192, 8, 1), (8, 1, 65536), 'cuda:1', torch.float32, 0, (8192, 8, 1))
buf29 = generate_example_value((8192, 1, 1), (1, 8192, 8192), 'cuda:1', torch.float32, 0, (8192, 1, 1))
with torch.cuda._DeviceGuard(1):
    triton_red_fused_7.run(buf21, buf23, buf29, 65536, 8192, stream=stream1)

stream1 = get_raw_stream(1)
arg15_1 = generate_example_value((256,), (1,), 'cuda:1', torch.float16, 0, (256,))
arg16_1 = generate_example_value((256,), (1,), 'cuda:1', torch.float16, 0, (256,))
buf26 = generate_example_value((8192, 8, 64), (2048, 256, 1), 'cuda:1', torch.float16, 0, (8192, 8, 64))
buf27 = generate_example_value((8192, 8, 192), (2048, 256, 1), 'cuda:1', torch.float16, 0, (8192, 8, 192))
buf30 = generate_example_value((8192, 1, 64), (256, 256, 1), 'cuda:1', torch.float16, 0, (8192, 1, 64))
buf31 = generate_example_value((8192, 1, 192), (256, 256, 1), 'cuda:1', torch.float16, 0, (8192, 1, 192))
with torch.cuda._DeviceGuard(1):
    triton_poi_fused_8.run(buf21, buf23, arg15_1, buf24, buf25, buf29, arg16_1, buf26, buf27, buf30, buf31, 4194304, 12582912, 524288, 1572864, stream=stream1)
del buf24, buf25, buf21, buf23, buf29, arg15_1, arg16_1, buf26, buf27, buf30, buf31

"""
# AOT ID: ['3_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/sn/csn5am67odvgbktv5sswroffzb7ra7iqib3rf45f3ermt2yglekn.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp29 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (((-2) + x0) % 3)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp7
    tmp10 = tl.load(in_ptr0 + (x1 + 2*ks0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert(((0 <= tl.broadcast_to(tmp14, [XBLOCK])) & (tl.broadcast_to(tmp14, [XBLOCK]) < 1048576)) | ~(tmp9 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp14, [XBLOCK]) < 1048576")
    tmp16 = tl.load(in_ptr1 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3)) + 64*tmp14), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.full([1], 1, tl.int64)
    tmp18 = tmp0 >= tmp17
    tmp19 = (((-1) + x0) % 3)
    tmp20 = tmp19 == tmp6
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr0 + (ks0 + x1), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK]) < 1048576)) | ~(tmp21 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK]) < 1048576")
    tmp28 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3)) + 64*tmp26), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp31 = tmp29 + tmp30
    tmp32 = tmp29 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp29)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp33 < 1048576")
    tmp35 = tl.load(in_ptr1 + (x0 + 64*tmp33), xmask).to(tl.float32)
    tmp36 = tl.where(tmp21, tmp28, tmp35)
    tmp37 = tl.where(tmp9, tmp16, tmp36)
    tmp38 = tl.load(in_ptr1 + (34 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3)) + 64*tmp14), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (33 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3)) + 64*tmp26), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr1 + (32 + x0 + 64*tmp33), xmask).to(tl.float32)
    tmp41 = tl.where(tmp21, tmp39, tmp40)
    tmp42 = tl.where(tmp9, tmp38, tmp41)
    tl.store(out_ptr0 + (x2), tmp37, xmask)
    tl.store(out_ptr1 + (x2), tmp42, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/ez/cezh2iyzzx2igt47kgvkvo64ovxjmg6jqrq2amryzutvf3nchgnn.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_per_fused_1 = async_compile.triton('triton_per_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (r0_1 + 128*((x0 % 16)) + 6144*(x0 // 16)), xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl.full([1, 1], 128.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1, 1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = -tmp17
    tmp19 = libdevice.exp(tmp18)
    tmp20 = tl.full([1, 1], 1.0, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = (tmp17 / tmp21)
    tmp23 = tmp15 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 128*x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/nr/cnrmtkpftgk7a7wgdyhfkavkqpgngnqpydzy6wvpoimg5ievmd54.py
# Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul, float_2, mul_1, reshape_1, float_3, silu, mul_2, to, flatten, linear], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.clone, aten._unsafe_view, aten.silu, aten.t, aten.mm]
# Source node to ATen node mapping:
#   add => add_22
#   flatten => view_3
#   float_1 => convert_element_type
#   float_2 => convert_element_type_1
#   float_3 => convert_element_type_2
#   linear => mm, permute
#   mean => mean
#   mul => mul_32
#   mul_1 => mul_35
#   mul_2 => mul_40
#   pow_1 => pow_1
#   reshape => view
#   reshape_1 => clone, view_1
#   rsqrt => rsqrt
#   silu => add_35, div, exp, neg
#   to => convert_element_type_3
# Graph fragment:
#   %convert_element_type_3 : Tensor "f16[16*s18, 128][128, 1]cuda:1" = PlaceHolder[target=convert_element_type_3]
#   %view : Tensor "f16[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 128]), kwargs = {})
#   %convert_element_type : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[16*s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_22 : Tensor "f32[16*s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[16*s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_32 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[128][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %mul_35 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %convert_element_type_1), kwargs = {})
#   %clone : Tensor "f16[s18, 16, 128][2048, 128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%arg2_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_1 : Tensor "f16[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [%mul_6, 128]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_2,), kwargs = {})
#   %exp : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_35 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, %add_35), kwargs = {})
#   %mul_40 : Tensor "f32[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %div), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[16*s18, 128][128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_40, torch.float16), kwargs = {})
#   %view_3 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_3, [%arg7_1, 2048]), kwargs = {})
#   %permute : Tensor "f16[2048, 2048][1, 2048]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg5_1, [1, 0]), kwargs = {})
#   %mm : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_3, %permute), kwargs = {})
#   return %buf2
triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128*((((x0 + 2048*x1) // 128) % (16*ks0))) + ((x0 % 128))), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/ao/cao6ytwzbe4uxdzi4zculnhrjkayvnde3v7knrdcojg6jqatgz7f.py
# Topologically Sorted Source Nodes: [float_4, add_1, fused_add_rms_norm_default, moe_forward_shared], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, vllm.moe_forward_shared]
# Source node to ATen node mapping:
#   add_1 => add_67
#   float_4 => convert_element_type_6
#   fused_add_rms_norm_default => add_tensor_4, add_tensor_5, convert_element_type_default_11, convert_element_type_default_8, convert_element_type_default_9, mean_dim_3, mul_tensor_6, mul_tensor_7, pow_tensor_scalar_3, rsqrt_default_3
#   moe_forward_shared => moe_forward_shared
# Graph fragment:
#   %all_reduce : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce]
#   %arg9_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg9_1]
#   %buf6 : Tensor "f32[s18, 1][1, s18]cuda:1" = PlaceHolder[target=buf6]
#   %arg8_1 : Tensor "f16[2048][1]cuda:1" = PlaceHolder[target=arg8_1]
#   %convert_element_type_6 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.float32), kwargs = {})
#   %add_67 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 1.0), kwargs = {})
#   %convert_element_type_default_8 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce, torch.float32), kwargs = {})
#   %convert_element_type_default_9 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_8, %convert_element_type_default_9), kwargs = {})
#   %pow_tensor_scalar_3 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_4, 2), kwargs = {})
#   %mean_dim_3 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_3, [-1], True), kwargs = {})
#   %add_tensor_5 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_3, 1e-06), kwargs = {})
#   %rsqrt_default_3 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_tensor_6 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_4, %rsqrt_default_3), kwargs = {})
#   %mul_tensor_7 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_6, %add_67), kwargs = {})
#   %convert_element_type_default_11 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_7, torch.float16), kwargs = {})
#   %moe_forward_shared : [num_users=2] = call_function[target=torch.ops.vllm.moe_forward_shared.default](args = (%convert_element_type_default_11, %convert_element_type_default_11, %convert_element_type_default_11, None, %arg10_1, 0), kwargs = {})
#   return %buf6,%buf7,%buf8,%buf9
triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 268439552}}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 2048.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp19 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp25, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp25, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 2048*x0), tmp25, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/tz/ctz6a4jvvt3fqr4r6s5jcwhifxzxninp3zhavn7gx6i5wkl4du6r.py
# Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_2 => add_83
# Graph fragment:
#   %getitem_2 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=getitem_2]
#   %getitem_3 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=getitem_3]
#   %add_83 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, %getitem_3), kwargs = {})
#   return %add_83
triton_poi_fused_add_4 = async_compile.triton('triton_poi_fused_add_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 134217728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/ju/cjupfiv3i26vwt6btkvylvkf2hf6cwobvwabplre2a4dorhr3dyn.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_default, float_5, add_3, fused_add_rms_norm_default_1, marlin_gemm], Original ATen: [vllm_ir.fused_add_rms_norm, aten._to_copy, aten.add, _C.marlin_gemm, aten.copy_]
# Source node to ATen node mapping:
#   add_3 => add_93
#   float_5 => convert_element_type_7
#   fused_add_rms_norm_default => add_tensor_4, convert_element_type_default_10, convert_element_type_default_8, convert_element_type_default_9
#   fused_add_rms_norm_default_1 => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_7, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
#   marlin_gemm => marlin_gemm
# Graph fragment:
#   %all_reduce_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce_1]
#   %all_reduce : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce]
#   %arg9_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg9_1]
#   %buf17 : Tensor "f32[s18, 1][1, s18]cuda:1" = PlaceHolder[target=buf17]
#   %arg11_1 : Tensor "f16[2048][1]cuda:1" = PlaceHolder[target=arg11_1]
#   %copy_ : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=copy_]
#   %convert_element_type_default_8 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce, torch.float32), kwargs = {})
#   %convert_element_type_default_9 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_8, %convert_element_type_default_9), kwargs = {})
#   %convert_element_type_default_10 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_4, torch.float16), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg11_1, torch.float32), kwargs = {})
#   %add_93 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_7, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce_1, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_10, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.float16), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_2), kwargs = {})
#   %mul_tensor_5 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_4, %add_93), kwargs = {})
#   %convert_element_type_default_7 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_5, torch.float16), kwargs = {})
#   %marlin_gemm : Tensor "f16[s18, 4608][4608, 1]cuda:1"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%convert_element_type_default_7, None, %arg12_1, None, %arg13_1, None, None, None, None, None, %arg14_1, 2814749767172868, %arg7_1, 4608, 2048, True, False, True, False), kwargs = {})
#   %copy_ : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg6_1, %all_reduce), kwargs = {})
#   return %buf17,%convert_element_type_default_6,%buf19,%buf34
triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5 = async_compile.triton('triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5', 'mutated_arg_names': ['out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 301993984}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 2048.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.full([1, 1], 1.0, tl.float32)
        tmp34 = tmp32 + tmp33
        tmp35 = tmp30 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp36, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 2048*x0), tmp17, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/hy/chyez7emnmnqavn4gzjc6eo2l3ngxuo7lmo656jdaqscfd7qbtcv.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 8)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 4608*x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/bj/cbjcfhyz453mv4nsfd2kj7nsvtfqfeleeqew3bleie4ghxopa5d5.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_7 = async_compile.triton('triton_red_fused_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_7', 'mutated_arg_names': [], 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_7(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = (xindex % 8)
        x1 = xindex // 8
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x0 + 4608*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(out_ptr0 + (x3), tmp4, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x4 = xindex
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_5 = r0_index
            tmp6 = tl.load(in_ptr0 + (4096 + r0_5 + 4608*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr1 + (x4), tmp10, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 4608), (4608, 1), device='cuda:1', dtype=torch.float16)
    arg_1 = rand_strided((8192, 8, 1), (8, 1, 65536), device='cuda:1', dtype=torch.float32)
    arg_2 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:1', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 65536, 8192,


def call(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        stream1 = get_raw_stream(1)
        triton_red_fused_7.run(*args, stream=stream1)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        return triton_red_fused_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/wb/cwb7fe4vcrvy27gznbwe4qyuykmfdqwhls62c5kmcgzmieqjm65j.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(xnumel_3, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 8)
        x2 = xindex // 512
        x3 = xindex // 64
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 32, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (512*x1 + 4608*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.full([1], 256.0, tl.float32)
        tmp9 = (tmp7 / tmp8)
        tmp10 = tl.full([1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp6 * tmp12
        tmp14 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (32*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tl.load(in_ptr0 + (32 + 512*x1 + 4608*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 * tmp12
        tmp25 = tl.load(in_ptr2 + (32 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 + tmp16
        tmp28 = tmp24 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (32*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tmp29 * tmp30
        tmp32 = tmp21 - tmp31
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp4, tmp32, tmp33)
        tmp35 = tmp0 >= tmp3
        tmp36 = tl.full([1], 64, tl.int64)
        tmp37 = tmp0 < tmp36
        tmp38 = tl.load(in_ptr0 + (32 + 512*x1 + 4608*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tl.load(in_ptr1 + (x3), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full([1], 256.0, tl.float32)
        tmp42 = (tmp40 / tmp41)
        tmp43 = tl.full([1], 1e-06, tl.float32)
        tmp44 = tmp42 + tmp43
        tmp45 = libdevice.rsqrt(tmp44)
        tmp46 = tmp39 * tmp45
        tmp47 = tl.load(in_ptr2 + (32 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tl.full([1], 1.0, tl.float32)
        tmp50 = tmp48 + tmp49
        tmp51 = tmp46 * tmp50
        tmp52 = tmp51.to(tl.float32)
        tmp53 = tl.load(in_ptr3 + (32*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tmp52 * tmp53
        tmp55 = tl.load(in_ptr0 + (512*x1 + 4608*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tmp56 * tmp45
        tmp58 = tl.load(in_ptr2 + ((-32) + x0), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tmp58.to(tl.float32)
        tmp60 = tmp59 + tmp49
        tmp61 = tmp57 * tmp60
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tl.load(in_ptr4 + (32*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp64 = tmp62 * tmp63
        tmp65 = tmp54 + tmp64
        tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
        tmp67 = tl.where(tmp35, tmp65, tmp66)
        tmp68 = tl.where(tmp4, tmp34, tmp67)
        tl.store(out_ptr0 + (x0 + 256*x3), tmp68, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x4 = (xindex % 192)
        x5 = ((xindex // 192) % 8)
        x6 = xindex // 1536
        x7 = xindex // 192
        tmp69 = tl.load(in_ptr0 + (64 + x4 + 512*x5 + 4608*x6), xmask).to(tl.float32)
        tmp71 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last')
        tmp78 = tl.load(in_ptr2 + (64 + x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp70 = tmp69.to(tl.float32)
        tmp72 = tl.full([1], 256.0, tl.float32)
        tmp73 = (tmp71 / tmp72)
        tmp74 = tl.full([1], 1e-06, tl.float32)
        tmp75 = tmp73 + tmp74
        tmp76 = libdevice.rsqrt(tmp75)
        tmp77 = tmp70 * tmp76
        tmp79 = tmp78.to(tl.float32)
        tmp80 = tl.full([1], 1.0, tl.float32)
        tmp81 = tmp79 + tmp80
        tmp82 = tmp77 * tmp81
        tmp83 = tmp82.to(tl.float32)
        tl.store(out_ptr1 + (x4 + 256*x7), tmp83, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x8 = (xindex % 64)
        x9 = xindex // 64
        tmp84 = x8
        tmp85 = tl.full([1], 0, tl.int64)
        tmp86 = tmp84 >= tmp85
        tmp87 = tl.full([1], 32, tl.int64)
        tmp88 = tmp84 < tmp87
        tmp89 = tl.load(in_ptr0 + (4096 + 4608*x9 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp90 = tmp89.to(tl.float32)
        tmp91 = tl.load(in_ptr5 + (x9), tmp88 & xmask, eviction_policy='evict_last', other=0.0)
        tmp92 = tl.full([1], 256.0, tl.float32)
        tmp93 = (tmp91 / tmp92)
        tmp94 = tl.full([1], 1e-06, tl.float32)
        tmp95 = tmp93 + tmp94
        tmp96 = libdevice.rsqrt(tmp95)
        tmp97 = tmp90 * tmp96
        tmp98 = tl.load(in_ptr6 + (x8), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tmp98.to(tl.float32)
        tmp100 = tl.full([1], 1.0, tl.float32)
        tmp101 = tmp99 + tmp100
        tmp102 = tmp97 * tmp101
        tmp103 = tmp102.to(tl.float32)
        tmp104 = tl.load(in_ptr3 + (32*x9 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp105 = tmp103 * tmp104
        tmp106 = tl.load(in_ptr0 + (4128 + 4608*x9 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp107 = tmp106.to(tl.float32)
        tmp108 = tmp107 * tmp96
        tmp109 = tl.load(in_ptr6 + (32 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp110 = tmp109.to(tl.float32)
        tmp111 = tmp110 + tmp100
        tmp112 = tmp108 * tmp111
        tmp113 = tmp112.to(tl.float32)
        tmp114 = tl.load(in_ptr4 + (32*x9 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp115 = tmp113 * tmp114
        tmp116 = tmp105 - tmp115
        tmp117 = tl.full(tmp116.shape, 0.0, tmp116.dtype)
        tmp118 = tl.where(tmp88, tmp116, tmp117)
        tmp119 = tmp84 >= tmp87
        tmp120 = tl.full([1], 64, tl.int64)
        tmp121 = tmp84 < tmp120
        tmp122 = tl.load(in_ptr0 + (4128 + 4608*x9 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp123 = tmp122.to(tl.float32)
        tmp124 = tl.load(in_ptr5 + (x9), tmp119 & xmask, eviction_policy='evict_last', other=0.0)
        tmp125 = tl.full([1], 256.0, tl.float32)
        tmp126 = (tmp124 / tmp125)
        tmp127 = tl.full([1], 1e-06, tl.float32)
        tmp128 = tmp126 + tmp127
        tmp129 = libdevice.rsqrt(tmp128)
        tmp130 = tmp123 * tmp129
        tmp131 = tl.load(in_ptr6 + (32 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp132 = tmp131.to(tl.float32)
        tmp133 = tl.full([1], 1.0, tl.float32)
        tmp134 = tmp132 + tmp133
        tmp135 = tmp130 * tmp134
        tmp136 = tmp135.to(tl.float32)
        tmp137 = tl.load(in_ptr3 + (32*x9 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp138 = tmp136 * tmp137
        tmp139 = tl.load(in_ptr0 + (4096 + 4608*x9 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp140 = tmp139.to(tl.float32)
        tmp141 = tmp140 * tmp129
        tmp142 = tl.load(in_ptr6 + ((-32) + x8), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp143 = tmp142.to(tl.float32)
        tmp144 = tmp143 + tmp133
        tmp145 = tmp141 * tmp144
        tmp146 = tmp145.to(tl.float32)
        tmp147 = tl.load(in_ptr4 + (32*x9 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp148 = tmp146 * tmp147
        tmp149 = tmp138 + tmp148
        tmp150 = tl.full(tmp149.shape, 0.0, tmp149.dtype)
        tmp151 = tl.where(tmp119, tmp149, tmp150)
        tmp152 = tl.where(tmp88, tmp118, tmp151)
        tl.store(out_ptr2 + (x8 + 256*x9), tmp152, xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_3
        x10 = (xindex % 192)
        x11 = xindex // 192
        tmp153 = tl.load(in_ptr0 + (4160 + x10 + 4608*x11), xmask).to(tl.float32)
        tmp155 = tl.load(in_ptr5 + (x11), xmask, eviction_policy='evict_last')
        tmp162 = tl.load(in_ptr6 + (64 + x10), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp154 = tmp153.to(tl.float32)
        tmp156 = tl.full([1], 256.0, tl.float32)
        tmp157 = (tmp155 / tmp156)
        tmp158 = tl.full([1], 1e-06, tl.float32)
        tmp159 = tmp157 + tmp158
        tmp160 = libdevice.rsqrt(tmp159)
        tmp161 = tmp154 * tmp160
        tmp163 = tmp162.to(tl.float32)
        tmp164 = tl.full([1], 1.0, tl.float32)
        tmp165 = tmp163 + tmp164
        tmp166 = tmp161 * tmp165
        tmp167 = tmp166.to(tl.float32)
        tl.store(out_ptr3 + (x10 + 256*x11), tmp167, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 4608), (4608, 1), device='cuda:1', dtype=torch.float16)
    arg_1 = rand_strided((8192, 8, 1), (8, 1, 65536), device='cuda:1', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:1', dtype=torch.float16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:1', dtype=torch.float16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:1', dtype=torch.float16)
    arg_5 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:1', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:1', dtype=torch.float16)
    arg_7 = rand_strided((8192, 8, 64), (2048, 256, 1), device='cuda:1', dtype=torch.float16)
    arg_8 = rand_strided((8192, 8, 192), (2048, 256, 1), device='cuda:1', dtype=torch.float16)
    arg_9 = rand_strided((8192, 1, 64), (256, 256, 1), device='cuda:1', dtype=torch.float16)
    arg_10 = rand_strided((8192, 1, 192), (256, 256, 1), device='cuda:1', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 4194304, 12582912, 524288, 1572864,


def call(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        stream1 = get_raw_stream(1)
        triton_poi_fused_8.run(*args, stream=stream1)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        return triton_poi_fused_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1 = args
        args.clear()
        s59 = arg1_1
        s18 = arg7_1
        s7 = arg19_1
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            buf24 = empty_strided_cuda((s18, 32), (32, 1), torch.float16)
            buf25 = empty_strided_cuda((s18, 32), (32, 1), torch.float16)
            buf1 = empty_strided_cuda((16*s18, 128), (128, 1), torch.float16)
            # Unsorted Source Nodes: [], Original ATen: []
            triton_poi_fused_0_xnumel = 32*s18
            stream1 = get_raw_stream(1)
            triton_poi_fused_0.run(arg18_1, arg17_1, buf24, buf25, s7, triton_poi_fused_0_xnumel, stream=stream1)
            # Unsorted Source Nodes: [], Original ATen: []
            triton_per_fused_1_xnumel = 16*s18
            stream1 = get_raw_stream(1)
            triton_per_fused_1.run(arg0_1, arg3_1, arg2_1, buf1, triton_per_fused_1_xnumel, 128, stream=stream1)
            del arg0_1
            del arg17_1
            del arg18_1
            del arg2_1
            del arg3_1
            buf2 = empty_strided_cuda((s18, 2048), (2048, 1), torch.float16)
            # Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul, float_2, mul_1, reshape_1, float_3, silu, mul_2, to, flatten, linear], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.clone, aten._unsafe_view, aten.silu, aten.t, aten.mm]
            triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2_xnumel = 2048*s18
            stream1 = get_raw_stream(1)
            triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2.run(buf1, buf2, s18, triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2_xnumel, stream=stream1)
            buf3 = reinterpret_tensor(buf1, (s18, 2048), (2048, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul, float_2, mul_1, reshape_1, float_3, silu, mul_2, to, flatten, linear], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.clone, aten._unsafe_view, aten.silu, aten.t, aten.mm]
            extern_kernels.mm(buf2, reinterpret_tensor(arg5_1, (2048, 2048), (1, 2048), 0), out=buf3)
            del arg5_1
            del buf2
            # Topologically Sorted Source Nodes: [all_reduce], Original ATen: [vllm.all_reduce]
            buf4 = torch.ops.vllm.all_reduce.default(buf3, 'tp:0')
            del buf3
            buf5 = buf4
            del buf4
            buf7 = empty_strided_cuda((s18, 2048), (2048, 1), torch.float16)
            buf8 = empty_strided_cuda((s18, 2048), (2048, 1), torch.float16)
            buf9 = empty_strided_cuda((s18, 2048), (2048, 1), torch.float16)
            # Topologically Sorted Source Nodes: [float_4, add_1, fused_add_rms_norm_default, moe_forward_shared], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, vllm.moe_forward_shared]
            stream1 = get_raw_stream(1)
            triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_3.run(buf5, arg9_1, arg8_1, buf7, buf8, buf9, s18, 2048, stream=stream1)
            del arg8_1
            # Topologically Sorted Source Nodes: [float_4, add_1, fused_add_rms_norm_default, moe_forward_shared], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, vllm.moe_forward_shared]
            buf10 = torch.ops.vllm.moe_forward_shared.default(buf7, buf8, buf9, None, arg10_1, 0)
            del arg10_1
            del buf7
            del buf8
            del buf9
            buf11 = buf10[0]
            buf12 = buf10[1]
            del buf10
            buf13 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.add]
            triton_poi_fused_add_4_xnumel = 2048*s18
            stream1 = get_raw_stream(1)
            triton_poi_fused_add_4.run(buf13, buf12, triton_poi_fused_add_4_xnumel, stream=stream1)
            del buf12
            # Topologically Sorted Source Nodes: [add_2, all_reduce_1], Original ATen: [aten.add, vllm.all_reduce]
            buf14 = torch.ops.vllm.all_reduce.default(buf13, 'tp:0')
            del buf13
            buf15 = buf14
            del buf14
            buf16 = empty_strided_cuda((s18, 2048), (2048, 1), torch.float16)
            buf19 = empty_strided_cuda((s18, 2048), (2048, 1), torch.float16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_default, float_5, add_3, fused_add_rms_norm_default_1, marlin_gemm], Original ATen: [vllm_ir.fused_add_rms_norm, aten._to_copy, aten.add, _C.marlin_gemm, aten.copy_]
            stream1 = get_raw_stream(1)
            triton_red_fused__to_copy_add_copy__fused_add_rms_norm_marlin_gemm_5.run(buf15, buf5, arg9_1, arg11_1, buf16, buf19, arg6_1, s18, 2048, stream=stream1)
            del arg11_1
            del arg6_1
            del arg9_1
            del buf15
            buf18 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_default, float_5, add_3, fused_add_rms_norm_default_1, marlin_gemm], Original ATen: [vllm_ir.fused_add_rms_norm, aten._to_copy, aten.add, _C.marlin_gemm]
            buf20 = torch.ops._C.marlin_gemm.default(buf19, None, arg12_1, None, arg13_1, None, None, None, None, None, arg14_1, 2814749767172868, s18, 4608, 2048, True, False, True, False)
            del arg12_1
            del arg13_1
            del arg14_1
            del buf19
            buf21 = buf20
            del buf20
            buf22 = empty_strided_cuda((s18, 8, 256), (2048, 256, 1), torch.float16)
            buf23 = empty_strided_cuda((s18, 8, 1), (8, 1, 8*s18), torch.float32)
            buf29 = empty_strided_cuda((s18, 1, 1), (1, s18, s18), torch.float32)
            # Topologically Sorted Source Nodes: [split, view_2, chunk, reshape_6, reshape_5, rms_norm_default, view_5, rms_norm_default_1], Original ATen: [aten.split_with_sizes, aten.view, aten.split, aten.clone, vllm_ir.rms_norm]
            triton_poi_fused_6_xnumel = 2048*s18
            stream1 = get_raw_stream(1)
            triton_poi_fused_6.run(buf21, buf22, triton_poi_fused_6_xnumel, stream=stream1)
            # Topologically Sorted Source Nodes: [split, view_2, chunk, reshape_6, reshape_5, rms_norm_default, view_5, rms_norm_default_1], Original ATen: [aten.split_with_sizes, aten.view, aten.split, aten.clone, vllm_ir.rms_norm]
            triton_red_fused_7_xnumel_0 = 8*s18
            stream1 = get_raw_stream(1)
            triton_red_fused_7.run(buf21, buf23, buf29, triton_red_fused_7_xnumel_0, s18, stream=stream1)
            buf28 = empty_strided_cuda((s18, 8, 256), (2048, 256, 1), torch.float16)
            buf26 = reinterpret_tensor(buf28, (s18, 8, 64), (2048, 256, 1), 0)  # alias
            buf27 = reinterpret_tensor(buf28, (s18, 8, 192), (2048, 256, 1), 64)  # alias
            buf32 = empty_strided_cuda((s18, 1, 256), (256, 256, 1), torch.float16)
            buf30 = reinterpret_tensor(buf32, (s18, 1, 64), (256, 256, 1), 0)  # alias
            buf31 = reinterpret_tensor(buf32, (s18, 1, 192), (256, 256, 1), 64)  # alias
            # Topologically Sorted Source Nodes: [split, view_2, chunk, reshape_5, float_6, add_4, rms_norm_default, getitem_20, chunk_2, mul_3, mul_4, sub, mul_5, mul_6, add_6, cat, getitem_21, cat_1, view_5, float_7, add_5, rms_norm_default_1, getitem_24, chunk_3, mul_7, mul_8, sub_1, mul_9, mul_10, add_7, cat_2, getitem_25, cat_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, aten.clone, aten._to_copy, aten.add, vllm_ir.rms_norm, aten.slice, aten.unsqueeze, aten.mul, aten.sub, aten.cat]
            triton_poi_fused_8_xnumel_0 = 512*s18
            triton_poi_fused_8_xnumel_1 = 1536*s18
            triton_poi_fused_8_xnumel_2 = 64*s18
            triton_poi_fused_8_xnumel_3 = 192*s18
            stream1 = get_raw_stream(1)
            triton_poi_fused_8.run(buf21, buf23, arg15_1, buf24, buf25, buf29, arg16_1, buf26, buf27, buf30, buf31, triton_poi_fused_8_xnumel_0, triton_poi_fused_8_xnumel_1, triton_poi_fused_8_xnumel_2, triton_poi_fused_8_xnumel_3, stream=stream1)
            del arg15_1
            del arg16_1
            del buf23
            del buf24
            del buf25
            del buf29
            buf33 = empty_strided_cuda((s18, 2048), (2048, 1), torch.float16)
        return (buf32, reinterpret_tensor(buf21, (s18, 1, 256), (4608, 256, 1), 4352), buf28, reinterpret_tensor(buf33, (s18, 8, 256), (2048, 256, 1), 0), reinterpret_tensor(buf22, (s18, 2048), (2048, 1), 0), buf18, buf16, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 16, 128), (2048, 128, 1), device='cuda:1', dtype=torch.float16)
    arg1_1 = 8192
    arg2_1 = rand_strided((8192, 16, 128), (6144, 128, 1), device='cuda:1', dtype=torch.float16)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:1', dtype=torch.float16)
    arg4_1 = 8192
    arg5_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:1', dtype=torch.float16)
    arg6_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:1', dtype=torch.float16)
    arg7_1 = 8192
    arg8_1 = rand_strided((2048, ), (1, ), device='cuda:1', dtype=torch.float16)
    arg9_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:1', dtype=torch.float16)
    import pickle
    global arg10_1
    arg10_1 = pickle.loads(b'\x80\x04\x95c\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c)language_model.model.layers.2.mlp.experts\x94sb.')
    arg11_1 = rand_strided((2048, ), (1, ), device='cuda:1', dtype=torch.float16)
    arg12_1 = rand_strided((128, 18432), (18432, 1), device='cuda:1', dtype=torch.int32)
    arg13_1 = rand_strided((1, 4608), (4608, 1), device='cuda:1', dtype=torch.float16)
    arg14_1 = rand_strided((82, ), (1, ), device='cuda:1', dtype=torch.int32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:1', dtype=torch.float16)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:1', dtype=torch.float16)
    arg17_1 = rand_strided((1048576, 64), (64, 1), device='cuda:1', dtype=torch.float16)
    arg18_1 = rand_strided((3, 8192), (8193, 1), device='cuda:1', dtype=torch.int64)
    arg19_1 = 8193
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
