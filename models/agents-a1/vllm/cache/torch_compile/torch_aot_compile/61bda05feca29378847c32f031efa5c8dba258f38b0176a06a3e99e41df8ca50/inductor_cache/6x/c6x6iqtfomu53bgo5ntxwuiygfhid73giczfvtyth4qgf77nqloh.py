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


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/jm/cjmghpfjl6la5podcfjrnf77g4sjlmw7fzm2cfjeoccn2655f5rt.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul, marlin_gemm], Original ATen: [aten.view, aten.sigmoid, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   marlin_gemm => marlin_gemm
#   mul => mul_4
#   sigmoid => sigmoid
#   view => view
# Graph fragment:
#   %arg0_1 : Tensor "f16[s18, 8, 256][2048, 256, 1]cuda:1" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg2_1]
#   %view : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 2048]), kwargs = {})
#   %sigmoid : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   %marlin_gemm : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_4, None, %arg3_1, None, %arg4_1, None, None, None, None, None, %arg5_1, 2814749767172868, %arg6_1, 2048, 2048, True, False, True, False), kwargs = {})
#   return %buf0
triton_poi_fused_marlin_gemm_mul_sigmoid_view_0 = async_compile.triton('triton_poi_fused_marlin_gemm_mul_sigmoid_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_marlin_gemm_mul_sigmoid_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 134217728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_marlin_gemm_mul_sigmoid_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/je/cjebavhk65nhfcvoel4pwvsng7jf3tnpyclmdphkrzflfcp7lofn.py
# Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default, moe_forward_shared], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, vllm.moe_forward_shared]
# Source node to ATen node mapping:
#   add => add_30
#   float_1 => convert_element_type
#   fused_add_rms_norm_default => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
#   moe_forward_shared => moe_forward_shared
# Graph fragment:
#   %all_reduce : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce]
#   %arg9_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg9_1]
#   %buf5 : Tensor "f32[s18, 1][1, s18]cuda:1" = PlaceHolder[target=buf5]
#   %arg8_1 : Tensor "f16[2048][1]cuda:1" = PlaceHolder[target=arg8_1]
#   %convert_element_type : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.float32), kwargs = {})
#   %add_30 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_30), kwargs = {})
#   %convert_element_type_default_7 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.float16), kwargs = {})
#   %moe_forward_shared : [num_users=2] = call_function[target=torch.ops.vllm.moe_forward_shared.default](args = (%convert_element_type_default_7, %convert_element_type_default_7, %convert_element_type_default_7, None, %arg10_1, 0), kwargs = {})
#   return %buf5,%buf6,%buf7,%buf8
triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 268439552}}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/fz/cfzugeqomf7y7vrhatekxun4cdiahj5qxww4eefivnrl735fr2vz.py
# Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_1 => add_46
# Graph fragment:
#   %getitem_2 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=getitem_2]
#   %getitem_3 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=getitem_3]
#   %add_46 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, %getitem_3), kwargs = {})
#   return %add_46
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 134217728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/cp/ccpbzsub4elastnajeybt3zi7p2trow7jo4sdphzvzeq7wu5umph.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_default, float_2, add_2, fused_add_rms_norm_default_1], Original ATen: [vllm_ir.fused_add_rms_norm, aten._to_copy, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add_2 => add_56
#   float_2 => convert_element_type_1
#   fused_add_rms_norm_default => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6
#   fused_add_rms_norm_default_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %all_reduce_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce_1]
#   %all_reduce : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce]
#   %arg9_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg9_1]
#   %buf15 : Tensor "f32[s18, 1][1, s18]cuda:1" = PlaceHolder[target=buf15]
#   %arg11_1 : Tensor "f16[2048][1]cuda:1" = PlaceHolder[target=arg11_1]
#   %copy_ : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=copy_]
#   %convert_element_type_default_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.float16), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg11_1, torch.float32), kwargs = {})
#   %add_56 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce_1, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_56), kwargs = {})
#   %convert_element_type_default_3 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.float16), kwargs = {})
#   %copy_ : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg7_1, %all_reduce), kwargs = {})
#   return %buf15,%convert_element_type_default_3,%buf17
triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3 = async_compile.triton('triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 234885120}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp15 + tmp22
        tmp24 = tl.full([1, 1], 2048.0, tl.float32)
        tmp25 = (tmp12 / tmp24)
        tmp26 = tl.full([1, 1], 1e-06, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.full([1, 1], 1.0, tl.float32)
        tmp33 = tmp31 + tmp32
        tmp34 = tmp29 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 2048*x0), tmp35, r0_mask & xmask)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp16, r0_mask & xmask)
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
arg0_1 = generate_example_value((8192, 8, 256), (2048, 256, 1), 'cuda:1', torch.float16, 0, (8192, 8, 256))
arg2_1 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
buf0 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
with torch.cuda._DeviceGuard(1):
    triton_poi_fused_marlin_gemm_mul_sigmoid_view_0.run(arg0_1, arg2_1, buf0, 16777216, stream=stream1)
del arg0_1, arg2_1, buf0

stream1 = get_raw_stream(1)
buf4 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
arg9_1 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
arg8_1 = generate_example_value((2048,), (1,), 'cuda:1', torch.float16, 0, (2048,))
buf6 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
buf7 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
buf8 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
with torch.cuda._DeviceGuard(1):
    triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1.run(buf4, arg9_1, arg8_1, buf6, buf7, buf8, 8192, 2048, stream=stream1)
del arg8_1, buf6, buf7, buf8

stream1 = get_raw_stream(1)
buf12 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
buf11 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
with torch.cuda._DeviceGuard(1):
    triton_poi_fused_add_2.run(buf12, buf11, 16777216, stream=stream1)
del buf12, buf11

stream1 = get_raw_stream(1)
buf16 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
arg11_1 = generate_example_value((2048,), (1,), 'cuda:1', torch.float16, 0, (2048,))
arg7_1 = generate_example_value((8192, 2048), (2048, 1), 'cuda:1', torch.float16, 0, (8192, 2048))
with torch.cuda._DeviceGuard(1):
    triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3.run(buf16, buf4, arg9_1, arg11_1, arg7_1, 8192, 2048, stream=stream1)
del buf4, arg9_1, buf16, arg11_1, arg7_1

"""
# AOT ID: ['40_inference']
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


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/jm/cjmghpfjl6la5podcfjrnf77g4sjlmw7fzm2cfjeoccn2655f5rt.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul, marlin_gemm], Original ATen: [aten.view, aten.sigmoid, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   marlin_gemm => marlin_gemm
#   mul => mul_4
#   sigmoid => sigmoid
#   view => view
# Graph fragment:
#   %arg0_1 : Tensor "f16[s18, 8, 256][2048, 256, 1]cuda:1" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg2_1]
#   %view : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 2048]), kwargs = {})
#   %sigmoid : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   %marlin_gemm : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_4, None, %arg3_1, None, %arg4_1, None, None, None, None, None, %arg5_1, 2814749767172868, %arg6_1, 2048, 2048, True, False, True, False), kwargs = {})
#   return %buf0
triton_poi_fused_marlin_gemm_mul_sigmoid_view_0 = async_compile.triton('triton_poi_fused_marlin_gemm_mul_sigmoid_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_marlin_gemm_mul_sigmoid_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 134217728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_marlin_gemm_mul_sigmoid_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/je/cjebavhk65nhfcvoel4pwvsng7jf3tnpyclmdphkrzflfcp7lofn.py
# Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default, moe_forward_shared], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, vllm.moe_forward_shared]
# Source node to ATen node mapping:
#   add => add_30
#   float_1 => convert_element_type
#   fused_add_rms_norm_default => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
#   moe_forward_shared => moe_forward_shared
# Graph fragment:
#   %all_reduce : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce]
#   %arg9_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg9_1]
#   %buf5 : Tensor "f32[s18, 1][1, s18]cuda:1" = PlaceHolder[target=buf5]
#   %arg8_1 : Tensor "f16[2048][1]cuda:1" = PlaceHolder[target=arg8_1]
#   %convert_element_type : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.float32), kwargs = {})
#   %add_30 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_30), kwargs = {})
#   %convert_element_type_default_7 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.float16), kwargs = {})
#   %moe_forward_shared : [num_users=2] = call_function[target=torch.ops.vllm.moe_forward_shared.default](args = (%convert_element_type_default_7, %convert_element_type_default_7, %convert_element_type_default_7, None, %arg10_1, 0), kwargs = {})
#   return %buf5,%buf6,%buf7,%buf8
triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 268439552}}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/fz/cfzugeqomf7y7vrhatekxun4cdiahj5qxww4eefivnrl735fr2vz.py
# Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_1 => add_46
# Graph fragment:
#   %getitem_2 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=getitem_2]
#   %getitem_3 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=getitem_3]
#   %add_46 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, %getitem_3), kwargs = {})
#   return %add_46
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 134217728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/cp/ccpbzsub4elastnajeybt3zi7p2trow7jo4sdphzvzeq7wu5umph.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_default, float_2, add_2, fused_add_rms_norm_default_1], Original ATen: [vllm_ir.fused_add_rms_norm, aten._to_copy, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add_2 => add_56
#   float_2 => convert_element_type_1
#   fused_add_rms_norm_default => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6
#   fused_add_rms_norm_default_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %all_reduce_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce_1]
#   %all_reduce : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=all_reduce]
#   %arg9_1 : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=arg9_1]
#   %buf15 : Tensor "f32[s18, 1][1, s18]cuda:1" = PlaceHolder[target=buf15]
#   %arg11_1 : Tensor "f16[2048][1]cuda:1" = PlaceHolder[target=arg11_1]
#   %copy_ : Tensor "f16[s18, 2048][2048, 1]cuda:1" = PlaceHolder[target=copy_]
#   %convert_element_type_default_4 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.float16), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg11_1, torch.float32), kwargs = {})
#   %add_56 : Tensor "f32[2048][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%all_reduce_1, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_56), kwargs = {})
#   %convert_element_type_default_3 : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.float16), kwargs = {})
#   %copy_ : Tensor "f16[s18, 2048][2048, 1]cuda:1"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg7_1, %all_reduce), kwargs = {})
#   return %buf15,%convert_element_type_default_3,%buf17
triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3 = async_compile.triton('triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 234885120}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp15 + tmp22
        tmp24 = tl.full([1, 1], 2048.0, tl.float32)
        tmp25 = (tmp12 / tmp24)
        tmp26 = tl.full([1, 1], 1e-06, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.full([1, 1], 1.0, tl.float32)
        tmp33 = tmp31 + tmp32
        tmp34 = tmp29 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 2048*x0), tmp35, r0_mask & xmask)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp16, r0_mask & xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1 = args
        args.clear()
        s59 = arg1_1
        s18 = arg6_1
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            buf0 = empty_strided_cuda((s18, 2048), (2048, 1), torch.float16)
            # Topologically Sorted Source Nodes: [view, sigmoid, mul, marlin_gemm], Original ATen: [aten.view, aten.sigmoid, aten.mul, _C.marlin_gemm]
            triton_poi_fused_marlin_gemm_mul_sigmoid_view_0_xnumel = 2048*s18
            stream1 = get_raw_stream(1)
            triton_poi_fused_marlin_gemm_mul_sigmoid_view_0.run(arg0_1, arg2_1, buf0, triton_poi_fused_marlin_gemm_mul_sigmoid_view_0_xnumel, stream=stream1)
            del arg0_1
            del arg2_1
            # Topologically Sorted Source Nodes: [view, sigmoid, mul, marlin_gemm], Original ATen: [aten.view, aten.sigmoid, aten.mul, _C.marlin_gemm]
            buf1 = torch.ops._C.marlin_gemm.default(buf0, None, arg3_1, None, arg4_1, None, None, None, None, None, arg5_1, 2814749767172868, s18, 2048, 2048, True, False, True, False)
            del arg3_1
            del arg4_1
            del arg5_1
            buf2 = buf1
            del buf1
            # Topologically Sorted Source Nodes: [all_reduce], Original ATen: [vllm.all_reduce]
            buf3 = torch.ops.vllm.all_reduce.default(buf2, 'tp:0')
            buf4 = buf3
            del buf3
            buf6 = buf2; del buf2  # reuse
            buf7 = buf0; del buf0  # reuse
            buf8 = empty_strided_cuda((s18, 2048), (2048, 1), torch.float16)
            # Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default, moe_forward_shared], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, vllm.moe_forward_shared]
            stream1 = get_raw_stream(1)
            triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_1.run(buf4, arg9_1, arg8_1, buf6, buf7, buf8, s18, 2048, stream=stream1)
            del arg8_1
            # Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default, moe_forward_shared], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, vllm.moe_forward_shared]
            buf9 = torch.ops.vllm.moe_forward_shared.default(buf6, buf7, buf8, None, arg10_1, 0)
            del arg10_1
            del buf6
            del buf7
            del buf8
            buf10 = buf9[0]
            buf11 = buf9[1]
            del buf9
            buf12 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
            triton_poi_fused_add_2_xnumel = 2048*s18
            stream1 = get_raw_stream(1)
            triton_poi_fused_add_2.run(buf12, buf11, triton_poi_fused_add_2_xnumel, stream=stream1)
            del buf11
            # Topologically Sorted Source Nodes: [add_1, all_reduce_1], Original ATen: [aten.add, vllm.all_reduce]
            buf13 = torch.ops.vllm.all_reduce.default(buf12, 'tp:0')
            del buf12
            buf14 = buf13
            del buf13
            buf16 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_default, float_2, add_2, fused_add_rms_norm_default_1], Original ATen: [vllm_ir.fused_add_rms_norm, aten._to_copy, aten.add, aten.copy_]
            stream1 = get_raw_stream(1)
            triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3.run(buf16, buf4, arg9_1, arg11_1, arg7_1, s18, 2048, stream=stream1)
            del arg11_1
            del arg7_1
            del arg9_1
            del buf4
        return (buf16, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 8, 256), (2048, 256, 1), device='cuda:1', dtype=torch.float16)
    arg1_1 = 8192
    arg2_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:1', dtype=torch.float16)
    arg3_1 = rand_strided((128, 8192), (8192, 1), device='cuda:1', dtype=torch.int32)
    arg4_1 = rand_strided((1, 2048), (2048, 1), device='cuda:1', dtype=torch.float16)
    arg5_1 = rand_strided((82, ), (1, ), device='cuda:1', dtype=torch.int32)
    arg6_1 = 8192
    arg7_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:1', dtype=torch.float16)
    arg8_1 = rand_strided((2048, ), (1, ), device='cuda:1', dtype=torch.float16)
    arg9_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:1', dtype=torch.float16)
    import pickle
    global arg10_1
    arg10_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*language_model.model.layers.39.mlp.experts\x94sb.')
    arg11_1 = rand_strided((2048, ), (1, ), device='cuda:1', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
