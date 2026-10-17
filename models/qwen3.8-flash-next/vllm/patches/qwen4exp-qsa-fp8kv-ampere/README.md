# qwen4exp-qsa-fp8kv-ampere

FP8 (e4m3) KV cache for Qwen3.8-Flash-Next on Ampere (sm_80 / sm_86) with stock vLLM.
FP8 KV halves the attention KV per token, which buys concurrency.

**Bug.** The QSA sparse-attention kernel (`vllm/models/qwen4_exp/nvidia/ops/qsa.py`,
`_qsa_sparse_paged_gqa_splitk_kernel`) reads the FP8 cache through `float8_e4m3fn`
pointers. Triton has no `fp8e4nv` below sm_89, so `--kv-cache-dtype fp8_e4m3` fails to
compile on Ampere:

    ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")

Hit on 2x CMP 170HX (sm_80), vLLM v0.31.0, 2026-10-17.

**Fix.** Below sm_89 only, pass the cache as `uint8` and decode e4m3 in registers:
`((b & 0x80) << 8) | ((b & 0x7F) << 7)` reinterpreted as fp16, times 256. Exact for all 254
finite codes, subnormals included. K is then cast to BF16 for the QK dot and V stays fp16
for the PV dot, exactly as the stock FP8 path does. Stock's host-folded K/V scales are
untouched. sm_89+ keeps the native path.

Only this kernel reads the main KV cache. It is written by vLLM's CUDA
`reshape_and_cache_flash`, and the QSA indexer caches are a separate setting
(`indexer_kv_dtype`, BF16 by default) that this overlay does not touch.

**Provenance.** The decode is halt95's (Merlin engine, `github.com/halt95/qwen38-flash-next-3090s`,
Apache-2.0, `patches/0001-qsa-FP8-E4M3-KV-reader-for-sm_86-with-static-K-V-sca.patch`),
validated there on sm_86. Ported to the stock v0.31.0 kernel by club-3090 (stock already
folds the scales, so only the two load sites change).
Co-authored-by: halt95 <halt95@users.noreply.github.com>

**Not included.** Merlin's static per-layer K/V scale calibration (`VLLM_QSA_KV_SCALES`).
Stock runs FP8 KV at scale 1.0 unless the checkpoint carries scales.

**Applies to** v0.31.0 and v0.32.0rc3 (anchor-checked; the installer refuses otherwise).
