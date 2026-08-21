# DFLASH15 opt-in patches

These patches extend the vendored vLLM 0.27.1 DFlash2 backport for the
`dual-dflash15-fast` compose. They are applied in the container entrypoint and
fail closed on anchor drift.

- `hybrid-kv-groups-v2-cudagraph.patch`: keeps the DFlash2 sliding-window bucket
  from padding every target KV group and reserves `VLLM_V2_CUDAGRAPH_MEM_MIB`
  before KV sizing. The canonical compose pins the pool with
  `--kv-cache-memory=9000000000`.
- `dflash2-lookup-drafting.patch`: lookup-augmented block drafting from
  [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090).
  It requires `--no-async-scheduling` because the lookup flags are consumed by
  the synchronous scheduler path.
- `spec-decode-attn.patch`: split-KV FlashAttention verify kernel for blocks
  longer than the trained seven-token drafter block, same upstream source.
- `qwen3_5-embed-quant.patch`: routes compressed-tensors Qwen3.5/3.8 token
  embeddings through vLLM's quantized embedding path; required by the complete
  `Qwen3.8-27B-W4A16-AutoRound-fast` target.
- `install.sh`: idempotent, fail-closed installer. The canonical ceiling profile
  is `SPEC_N=15`, `MAX_NUM_SEQS=1`, `MAX_MODEL_LEN=244320`, `CG_MAX=16`,
  `VLLM_V2_CUDAGRAPH_MEM_MIB=1900`, BF16 KV and FlashAttention. `MAX_NUM_SEQS=2`
  is intentionally a separate unvalidated variant.

The patches target the
  pinned `vllm/vllm-openai:v0.27.1` image and must be removed or refreshed when
  the upstream DFlash2 and Qwen embedding support lands in the image.

The lookup patch is lossless: it only changes proposal construction; vLLM's
rejection sampler remains responsible for the final distribution.
