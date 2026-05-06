# Gemma 4 Per-Head KV Overlay Comparison

Date: 2026-05-06

Environment: Gemma 4 31B AutoRound INT4 target, Google Gemma 4 MTP drafter,
TP=2 on RTX 3090, `--kv-cache-dtype int8_per_token_head`,
`MAX_MODEL_LEN=131072`, `GPU_MEMORY_UTILIZATION=0.95`.

## Summary

The Codex spec-level overlay remains the only locally validated bootable path,
but it is not shippable for long-context traffic because prior validation showed
severe decode-TPS decay and this follow-up found the likely structural cause:
generic `page_size_padded` handling uses a strided page view that vLLM itself
marks as unsafe for standard attention layouts. The current upstream PR #40391
shape cannot be tested as a worker-only overlay under the "do not change Gemma 4
model code" constraint; its strict unifier still needs the PR's model/attention
spec changes to pre-pad Gemma 4 global layers to a 1040-byte factor.

Combining Codex's generic unifier with PR #40391's worker-side logical-shape
view booted, but failed correctness: a simple "capital of France" request
returned corrupted text. That rules out the naive hybrid overlay.

## Results

| Metric | Codex spec-level | PR #40391 worker-runtime | Codex unifier + PR worker view |
|---|---:|---:|---:|
| Boot success | Yes | No, worker-only overlay | Yes |
| KV cache size | 247,186 tokens observed in prior run | N/A | 49,264 token log, but 1.89x concurrency log; not comparable |
| verify-full pass | Prior sanity/tool checks passed | Not run | Not run, smoke failed |
| Turn-1 decode TPS | 32.957 prior run | N/A | Not run |
| Turn-5 decode TPS | 9.784 prior run | N/A | Not run |
| TPS retention | 29.7% | N/A | N/A |
| Diff size | 1-file local overlay | Upstream PR #40391, but not worker-only | 5-file local diagnostic overlay |
| Upstream-track | No | Yes, if full PR is applied | No |
| Recommended | No, not for shipping long-context int8 | Not locally validated under constraints | No |

## Diagnosis

The per-token-head cache update kernel does not iterate directly over
`page_size_bytes`; it works from logical cache tensor strides and logical
`block_size/head_size`. The relevant code path is
`triton_reshape_and_cache_flash_per_token_head_quant`, which passes
`key_cache.stride(...)`, `value_cache.stride(...)`, and scale-cache strides into
the Triton kernel.

The suspect overhead/correctness site is therefore not a simple "dequantizes the
padded page" loop. It is the logical KV cache tensor shape/stride created when
`page_size_padded` is present. The existing generic strided-view path assumes the
block index is the first physical dimension; the source comment explicitly says
that is not true for standard attention backends whose shape starts with a K/V
dimension. PR #40391's later worker patch confirms this by switching standard
attention to a padded logical last dimension instead of the generic block-stride
view.

PR #40391 also includes model/attention changes that set
`kv_cache_page_size_padded` for Gemma 4 global per-token-head layers. Without
those changes, its `kv_cache_utils.py` remains strict and fails with the original
`NotImplementedError`.

## Commands Run

Worker-only PR #40391 overlay test:

```bash
MODEL_DIR=/mnt/models/huggingface docker compose \
  -f /opt/ai/github/club-3090/models/gemma-4-31b/vllm/compose/docker-compose.gemma-mtp.yml \
  -f /tmp/gemma4-pr40391-perheadkv.override.yml \
  up --force-recreate
```

Result: first attempt failed because the base `gpu_model_runner.py` override
removed Gemma4 MTP overlay compatibility:
`InputBatch.__init__() got an unexpected keyword argument 'is_spec_decode'`.
After rebasing onto the Gemma4 MTP overlay file, it failed at the original
strict page-size unifier:
`NotImplementedError: The page size of the layer is not divisible by the maximum page size`.

Hybrid diagnostic overlay test:

```bash
MODEL_DIR=/mnt/models/huggingface docker compose \
  -f /opt/ai/github/club-3090/models/gemma-4-31b/vllm/compose/docker-compose.gemma-mtp.yml \
  up --force-recreate
```

Result: engine booted with `Available KV cache memory: 11.45 GiB` and
`Application startup complete`, but smoke generation was corrupted:

```text
The capital1edistist de// deimon-,/,，,/ or, l/List or or orP,P orP
```

The container was stopped after the smoke failure.

## Verdict

Fix not found for shippable Gemma 4 `int8_per_token_head` on Ampere TP=2.

Recommended next step: either test the full upstream PR #40391 including its
Gemma 4 model/attention spec changes, or keep `bf16` KV as the default. The
worker-runtime idea is probably the right upstream direction, but it cannot be
validated as a worker-only local overlay in this tree.
