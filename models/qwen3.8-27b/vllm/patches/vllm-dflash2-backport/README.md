# vllm-dflash2-backport — DFlash2 external drafter support (vLLM PR #52816)

Vendors the **v0.27.1 backport** of upstream [vllm-project/vllm#52816](https://github.com/vllm-project/vllm/pull/52816)
*"[Spec Decode] DFlash2: local convolution + candidate selector"* (OPEN as of 2026-08-20;
+885/−44, 13 files). Backport authored by [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
(`patches/dflash2-backport.patch`) — **attribution: the backport is theirs, not ours.**

## What it adds
A new `DFlash2DraftModel` architecture (`qwen3_dflash2.py` + `v1/worker/gpu/spec_decode/dflash2/`),
carried alongside the existing `DFlashDraftModel` (untouched). Grouped **dynamic depthwise
convolution** inside each drafter block (a position sees the ones before it without another
backbone pass) + a **candidate selector**. Enables the external DFlash2 **block drafter**
(`incoai/Qwen3.8-27B-DFlash2`, requantized W4A16 by syv-ai → `syvai/Qwen3.8-27B-DFlash2-W4A16`,
~1.2 GB) via `--speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}'`.

## Why it matters here
An EXTERNAL drafter — it does NOT use the built-in MTP proposer, so it entirely sidesteps the
MTP-proposer-specific acceptance collapse (club-3090#1052). Measured by syv-ai on a W4A16 target
(RTX 3090, v0.27.1): ~3.34–3.65 accept-len at greedy (≈ the bf16 drafter), ~2.7 GB less read/step.

## Requirements (from the model card)
- `--attention-backend FLASH_ATTN` and `--kv-cache-dtype bfloat16` (the drafter's block attention
  is non-causal; bf16 KV). NOT fp8 KV.
- The drafter ships **no** embeddings/lm_head — it shares the target's. Target lm_head must be
  **bf16** (upstream refuses a non-bf16 lm_head for the candidate top-k); our FAST-tier target
  (`qwen3.8-27b-autoround-int4`, lm_head bf16) qualifies, so no extra lookup patch is needed.

## Delivery
Idempotent `install.sh` (mount at `/etc/club3090/dflash2`, call in the entrypoint before serve):
reverse-applied fingerprint check via `_check_applied.py`, else `patch -p1 --forward` from the
`vllm/` package dir. Refuses boot on failure. Applies CLEAN to stock `vllm/vllm-openai:v0.27.1`
(verified 2026-08-20).

## Upstream status / drop-when
PR #52816 OPEN. Drop this vendored patch when #52816 merges AND the `vllm-stable` pin moves past it.
