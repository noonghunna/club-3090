# AOT ID: ['43_inference']
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


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/61bda05feca29378847c32f031efa5c8dba258f38b0176a06a3e99e41df8ca50/inductor_cache/6b/c6bekhk6lquqeezvidcdog2ffsuplczk2lpt2e6vifwuyonodzxh.py
# Topologically Sorted Source Nodes: [ge, lt, org_vocab_mask, ge_1, lt_1, added_vocab_mask, vocab_mask, mul, mul_1, valid_offset, sub_3, input_, invert], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.mul, aten.add, aten.sub, aten.bitwise_not]
# Source node to ATen node mapping:
#   added_vocab_mask => bitwise_and_1
#   ge => ge
#   ge_1 => ge_1
#   input_ => mul_6
#   invert => bitwise_not
#   lt => lt
#   lt_1 => lt_1
#   mul => mul
#   mul_1 => mul_2
#   org_vocab_mask => bitwise_and
#   sub_3 => sub_13
#   valid_offset => add_16
#   vocab_mask => bitwise_or
# Graph fragment:
#   %arg1_1 : Tensor "i32[s15][1]cuda:1" = PlaceHolder[target=arg1_1]
#   %ge : Tensor "b8[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg1_1, %arg2_1), kwargs = {})
#   %lt : Tensor "b8[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg1_1, %arg3_1), kwargs = {})
#   %bitwise_and : Tensor "b8[s15][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %lt), kwargs = {})
#   %ge_1 : Tensor "b8[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg1_1, %arg4_1), kwargs = {})
#   %lt_1 : Tensor "b8[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg1_1, %arg5_1), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[s15][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_1, %lt_1), kwargs = {})
#   %bitwise_or : Tensor "b8[s15][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and, %bitwise_and_1), kwargs = {})
#   %mul : Tensor "i64[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_and, %arg2_1), kwargs = {})
#   %mul_2 : Tensor "i64[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_and_1, %sub_8), kwargs = {})
#   %add_16 : Tensor "i64[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_2), kwargs = {})
#   %sub_13 : Tensor "i64[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %add_16), kwargs = {})
#   %mul_6 : Tensor "i64[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_or, %sub_13), kwargs = {})
#   %bitwise_not : Tensor "b8[s15][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%bitwise_or,), kwargs = {})
#   return %mul_6,%bitwise_not
triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0 = async_compile.triton('triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i64', 'out_ptr1': '*i1', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'ks4': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 462}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = ks0
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = ks2
    tmp7 = tmp0 >= tmp6
    tmp8 = ks3
    tmp9 = tmp0 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 | tmp10
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp0.to(tl.int64)
    tmp14 = tmp5.to(tl.int64)
    tmp15 = tmp14 * tmp1
    tmp16 = tmp10.to(tl.int64)
    tmp17 = ks0 + ks2 + ((-1)*ks1) + ((-1)*ks4)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tmp13 - tmp19
    tmp21 = tmp12 * tmp20
    tmp22 = tmp11 == 0
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        s15 = arg0_1
        s90 = arg2_1
        s87 = arg3_1
        s68 = arg4_1
        s53 = arg5_1
        s88 = arg6_1
        assert_size_stride(arg1_1, (s15, ), (1, ))
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            buf0 = empty_strided_cuda((s15, ), (1, ), torch.int64)
            buf1 = empty_strided_cuda((s15, ), (1, ), torch.bool)
            # Topologically Sorted Source Nodes: [ge, lt, org_vocab_mask, ge_1, lt_1, added_vocab_mask, vocab_mask, mul, mul_1, valid_offset, sub_3, input_, invert], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.mul, aten.add, aten.sub, aten.bitwise_not]
            stream1 = get_raw_stream(1)
            triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0.run(arg1_1, buf0, buf1, s90, s87, s68, s53, s88, s15, stream=stream1)
            del arg1_1
        return (buf0, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 21
    arg1_1 = rand_strided((21, ), (1, ), device='cuda:1', dtype=torch.int32)
    arg2_1 = 124160
    arg3_1 = 248320
    arg4_1 = 248320
    arg5_1 = 248320
    arg6_1 = 0
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
