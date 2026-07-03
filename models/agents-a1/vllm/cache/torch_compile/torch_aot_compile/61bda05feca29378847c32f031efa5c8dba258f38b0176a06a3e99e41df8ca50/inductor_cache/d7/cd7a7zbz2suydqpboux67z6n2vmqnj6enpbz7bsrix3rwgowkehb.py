
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
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 14, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'kernel_num_gb': 0.006324736, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4096 + 4608*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp20 = tl.load(in_ptr3 + (32*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tl.load(in_ptr0 + (4128 + 4608*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 * tmp12
    tmp25 = tl.load(in_ptr2 + (32 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 + tmp16
    tmp28 = tmp24 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.load(in_ptr4 + (32*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp21 - tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp4, tmp32, tmp33)
    tmp35 = tmp0 >= tmp3
    tmp36 = tl.full([1], 64, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tl.load(in_ptr0 + (4128 + 4608*x1 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tl.load(in_ptr1 + (x1), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp53 = tl.load(in_ptr3 + (32*x1 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp54 = tmp52 * tmp53
    tmp55 = tl.load(in_ptr0 + (4096 + 4608*x1 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56 * tmp45
    tmp58 = tl.load(in_ptr2 + ((-32) + x0), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp59 + tmp49
    tmp61 = tmp57 * tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tl.load(in_ptr4 + (32*x1 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp64 = tmp62 * tmp63
    tmp65 = tmp54 + tmp64
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp35, tmp65, tmp66)
    tmp68 = tl.where(tmp4, tmp34, tmp67)
    tl.store(out_ptr0 + (x0 + 256*x1), tmp68, xmask)


def get_args():
    arg_0 = rand_strided((8192, 4608), (4608, 1), device='cuda:1', dtype=torch.float16)
    arg_1 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:1', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:1', dtype=torch.float16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:1', dtype=torch.float16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:1', dtype=torch.float16)
    arg_5 = rand_strided((8192, 1, 64), (256, 256, 1), device='cuda:1', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 524288,


def call(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        stream1 = get_raw_stream(1)
        triton_.run(*args, stream=stream1)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006324736
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
