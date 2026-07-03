
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'kernel_num_gb': 0.004390912, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
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


def get_args():
    arg_0 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = 8193
    return arg_0, arg_1, arg_2, arg_3, arg_4, 262144,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004390912
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
