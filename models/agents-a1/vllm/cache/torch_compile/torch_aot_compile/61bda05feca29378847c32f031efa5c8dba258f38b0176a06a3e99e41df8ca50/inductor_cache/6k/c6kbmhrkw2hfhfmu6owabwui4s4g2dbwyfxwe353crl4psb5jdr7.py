
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
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
    arg_0 = rand_strided((8192, 4608), (4608, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((8192, 8, 1), (8, 1, 65536), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg_5 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float16)
    arg_7 = rand_strided((8192, 8, 64), (2048, 256, 1), device='cuda:0', dtype=torch.float16)
    arg_8 = rand_strided((8192, 8, 192), (2048, 256, 1), device='cuda:0', dtype=torch.float16)
    arg_9 = rand_strided((8192, 1, 64), (256, 256, 1), device='cuda:0', dtype=torch.float16)
    arg_10 = rand_strided((8192, 1, 192), (256, 256, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 4194304, 12582912, 524288, 1572864,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
