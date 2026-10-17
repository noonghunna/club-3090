# qwen4exp-ple-fp8-bytecopy

Makes the stock pinned-RAM PLE path (`--engram-config '{"cpu_offload":true}'`) work with an
FP8 PLE table on Ampere (sm_80 / sm_86).

**Bug.** `_lookup_ple_embedding_from_pinned_kernel` in
`vllm/models/qwen4_exp/common/ngram_embedding.py` copies table rows, but it is launched on
`float8_e4m3fn` tensors. Triton on Ampere has no `fp8e4nv`, so the kernel fails to compile
and the engine dies in the memory-profiling forward:

    ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")

Hit on 2x CMP 170HX (sm_80), vLLM v0.31.0, albucino W4A16 + FP8 PLE, 2026-10-17.

**Fix.** Launch the kernel on `int8` views when the table is FP8. Same bytes, no change to
BF16 tables or to archs with native FP8. Dequantization still happens afterwards in torch.

**Upstream.** Not yet filed; add a `docs/UPSTREAM.md` row before filing.

**Re-anchoring.** One anchor (the kernel launch in `Qwen4ExpPLEPinnedHostEmbedding._lookup`).
Missing or not unique -> the installer exits 1 and the container does not start.
