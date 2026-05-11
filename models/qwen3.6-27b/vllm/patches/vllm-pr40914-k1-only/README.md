# vLLM PR #40914 K+1-only overlay — manually rebased onto post-#41434 main

## Status: negative result on Qwen3.6-27B

Vendored 2026-05-11 to test whether PR #40914 could unblock TQ3 + MTP on
Qwen 3.6 27B without Genesis. It does **not** close the stack:

- With this overlay active, MTP acceptance stabilizes at AL=4.0 / ~100%,
  but outputs collapse into `!` floods and tool/multi-turn paths time out.
- Skipping `mtp.*` drafter layers does not fix the corruption.
- `--enforce-eager` plus the output-buffer epilogue fix does not fix it.
- Dropping this overlay improves verify-stress from 3/7 to 5/7, but
  TQ3/TQ4/k8v4 + MTP still fail long-context needles.

Keep this only as a re-test artifact. Do not mount it in shipping composes
unless upstream changes the PR into a true P67-equivalent multi-query
TurboQuant spec-decode fix.

## Source

- Upstream PR: <https://github.com/vllm-project/vllm/pull/40914>
  ("[Bugfix][Spec-Decode] TurboQuant K+1 spec-verify routing (fixes #40880)")
- Upstream head SHA: `0ee9b859bbb2bbb6e33a461e7fd1fee1fa4792cc` (2026-04-29)
- State at vendor time: OPEN, MERGEABLE.

## Why this is a manual rebase

PR #40914 was forked from main BEFORE merged PR #41434 ("Eliminate
GPU↔CPU syncs in attention impls", merged 2026-05-08). Its full
`turboquant_attn.py` overlay would REVERT #41434's `query_start_loc_cpu`
/ `seq_lens_cpu` CPU-resident-metadata fixes and reintroduce the
`.tolist()` cudagraph crash family.

We need ONLY the additive K+1 spec-verify dispatch block (~86 lines)
inserted between `num_decode_tokens = attn_metadata.num_decode_tokens`
and `if not attn_metadata.is_prefill:`. All other diff hunks from PR
#40914 are reverts of #41434 and must be skipped.

## What this overlay contains

`turboquant_attn.py` = current `origin/main` (post-#41434) + ONLY the
additive K+1 dispatch block from PR #40914 inserted at the right location.

Result:
- `query_start_loc_cpu` / `seq_lens_cpu` fields and accessors: PRESERVED
- `.tolist()` → CPU-resident metadata fix: PRESERVED
- K+1 spec-verify dispatch from #40914: ADDED
- `buf_holder=layer` call site: matches nightly's `triton_turboquant_decode.py`
  signature (which post-#41434 accepts that kwarg)

## What this was expected to fix

Bimodal MTP acceptance + "first word right then breaks" needle failure
on TQ3 + MTP. Symptom on the previous stack (PR #40798 partial only):

```
12:22:02  AL=3.73  accept 91.1%  per-position [0.962, 0.930, 0.841]  ← GOOD
12:22:32  AL=2.30  accept 43.4%  per-position [1.000, 0.303, 0.000]  ← BAD
12:23:42  AL=2.00  accept 33.2%  per-position [0.397, 0.306, 0.294]  ← VERY BAD
```

Verify-stress needle 10K/30K/60K/90K all failed with `expected 'golden
chinchilla 38' got 'golden '` pattern — first token recalled, then
attention diverged because the verify pass was attending only to
current-chunk K/V instead of prior cached compressed K/V.

This dispatch routes uniform K+1 spec-verify batches through the decode
kernel (which natively reads prior cached K/V). On our Qwen3.6-27B stack
this expectation did not hold; see the status note above.

## Drop trigger

```
gh api repos/vllm-project/vllm/pulls/40914 --jq '.state, .merged_at'
```

reports `MERGED` AND the merge commit is post-#41434 (i.e., the PR was
rebased before merge so it doesn't revert #41434).

## Verified on

- vLLM nightly: `1acd67a7` (includes merged #41434)
- Compose: `dual/int8-tq3.yml`
- Qwen 3.6 27B AutoRound INT4, TQ3 KV, MTP n=3, 262K × 2 streams
- 2026-05-11

## Round-2/3 local experiment: skip MTP drafter layers

Instrumentation captured the K+1 dispatch firing first on:

```text
mtp.layers.0.self_attn.attn
```

That is the MTP drafter path, not a target-model full-attention verify
layer. With this branch active there, acceptance stabilized at AL=4.0 /
~100%, but output degenerated into repeated `!`, meaning drafter and
target were agreeing on the same corrupt path.

This overlay now skips K+1 dispatch on `mtp.*` layers by default while
leaving the target verify path eligible. A/B escape hatch:

```bash
CLUB3090_TQ_K1_SKIP_MTP=0
```

Round 3 result: this did not fix the corruption. The target-side path is
also wrong for this stack.
