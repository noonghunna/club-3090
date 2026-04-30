# Cliff 1 Mechanism B Attack Log

Branch: `cliff1-fa-clamp`  
Starting commit: `90a03ce`  
Scope: local `club-3090` patches only. No new commits or PRs to `Sandermage/genesis-vllm-patches`.

## Known Wall Before This Pass

- `205K` and `175K` both fail with the full stack (`P101 + P103 + P104 + PN12 + PN13 + --num-gpu-blocks-override 50`) using the same OOM signature:
  `empty_strided_cuda((s18, 17408))`, tried to allocate `138 MiB` with about `130 MiB` free.
- `--max-num-batched-tokens` cannot drop below `4128` on Qwen3-Next hybrid because Mamba cache align mode asserts `block_size <= max_num_batched_tokens`; `2048` already failed, so `3072` is intentionally skipped.
- `8192` batch tokens is expected to worsen the fresh FFN intermediate allocation, not help it.
- Lowering `max_model_len` alone does not buy activation headroom because vLLM spends freed memory back into KV unless the block override caps the KV pool.

## Task 1 - PN12 Anchor Health

Live check on the dev205+ long-text container copied:
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/activation.py`.

Result: PN12 was not actually applied. `SiluAndMul.forward_cuda` was still the vanilla body:

```python
out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
self.op(out, x)
return out
```

No `Genesis PN12` marker and no `FFNIntermediateCache` import were present in the live file.

Cause: Genesis PN12's anchor expects the next decorator after `SiluAndMul` to be `@CustomOp.register("silu_and_mul_with_clamp")`. In dev205+ the next section is `MulAndSilu`, so the text patch skips. PN12's `apply()` path reports any non-failed text-patcher result as `"applied"`, which masks the skip.

Local action: added `patch_pn12_ffn_pool_anchor.py`, a repo-local sidecar that patches only `SiluAndMul.forward_cuda` with the PN12 pooled-output body using a class-scoped anchor. It runs after Genesis in `docker-compose.long-text.yml`; runtime pooling remains env-gated by `GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1`.

Ephemeral container verification (no vLLM boot) showed:

```text
[pn12_ffn_pool_anchor_fix] SiluAndMul.forward_cuda: applied
```

## Local P104 Sidecar

The active Genesis checkout does not contain P104. To avoid a false-positive `GENESIS_ENABLE_FA_MAX_SEQLEN_CLAMP=1` run, added `patch_fa_max_seqlen_clamp.py` as a local sidecar and wired it into `docker-compose.long-text.yml` after Genesis.

Runtime behavior is still env-gated by `GENESIS_ENABLE_FA_MAX_SEQLEN_CLAMP=1`; the file patch itself is local and idempotent.

Ephemeral container verification (no vLLM boot) showed:

```text
[fa_max_seqlen_clamp] _flash_attn_varlen: applied
```

## Pending

- Reboot long-text with `GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1` and confirm the live `SiluAndMul.forward_cuda` body contains the local marker.
- Re-test 205K with the full stack after the PN12 anchor repair.
- If the repaired PN12 still fails, capture a CUDA memory snapshot and disambiguate `gate_up_proj` vs `SiluAndMul` allocation as the failure site.

## Final Verdict

Pending.
