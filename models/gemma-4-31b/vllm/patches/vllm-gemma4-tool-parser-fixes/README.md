# Gemma 4 tool-parser stacked fixes — vLLM PR #42006 + PR #41991

**Authors:**
- [@whytem](https://github.com/whytem) — PR [#42006](https://github.com/vllm-project/vllm/pull/42006) (MTP streaming multi-tool calls)
- [@the-david-oy](https://github.com/the-david-oy) — PR [#41991](https://github.com/vllm-project/vllm/pull/41991) (infinite loop + array boundary in parser helpers)

**Vendored on:** 2026-05-08
**Stacked locally by:** noonghunna for club-3090 (post-Gemma4-MTP-merge nightly base)

## What these fix

Two independent tool-parser bugs that affect Gemma 4 + MTP serving paths:

### PR #42006 — MTP streaming multi-tool calls

Streaming SSE responses for multi-tool calls produced empty `tool_calls[]` in some
cases when MTP bundled the last param + closing `</function>` in the same delta.
Same class of bug as the Qwen3 tool-parser SSE-silence we fixed in
[club-3090 issue #72](https://github.com/noonghunna/club-3090/issues/72).

**Symptom**: opencode / Claude Code / Cline agents see "tool was called but didn't get
parameters" silently. Server returns 200 OK but tool_calls array is empty/malformed.

**Fix**: 142-line refactor of `_extract_streaming` in `gemma4_tool_parser.py`. Buffers
the partial state correctly across deltas. Handles MTP's bundled-token-output pattern.

### PR #41991 — Infinite loop + array boundary in parser helpers

`_parse_gemma4_args()` and `_parse_gemma4_array()` had two distinct bugs at edge cases:
- Infinite loop on certain malformed args strings (no progress check)
- Array index out of bounds when partial parse reaches end of buffer

**Symptom**: server hangs / 500s / silent crashes on tool calls with edge-case args.

**Fix**: 19-line addition of progress checks + bounds guards in the parsing helpers.

## Why stack them

Both PRs are filed within 1 day of each other, both target the same file, and
both fix real classes of agent-workflow bugs (silent SSE / hang / crash).
Same family as our shipped #72 fix for Qwen3 tool parser.

The two PRs touch **different lines** of `gemma4_tool_parser.py`:
- #41991 modifies lines around 200-300 (parsing helpers)
- #42006 modifies line 480+ (streaming extraction)

They stack cleanly with no merge conflict. Stacking lets us ship both fixes via
a single overlay file.

## Files vendored

```
tool_parsers/gemma4_tool_parser.py    # base + #41991 + #42006 stacked (919 lines vs 759 base)
```

Test files from both PRs not vendored — we don't run vLLM tests in our serving container.

## Validation status (as of vendor date)

- ✅ Both patches apply cleanly (no conflict, no fuzz)
- ✅ Final file Python-parses OK
- ⏳ **End-to-end validation pending** — needs verify-full's tool-calling check (8/8) to
  pass against a Gemma 4 + MTP container with this overlay mounted
- ⏳ Multi-tool streaming reproducer — fire a chat completion with multiple tool calls
  and confirm tool_calls[] is correctly populated end-to-end

## Drop trigger

When BOTH PRs merge to vLLM main:

```bash
gh api repos/vllm-project/vllm/pulls/42006 --jq '.state, .merged_at'
gh api repos/vllm-project/vllm/pulls/41991 --jq '.state, .merged_at'
```

Once both are merged + propagated to a nightly tag, this overlay can be removed entirely.

## Companion overlay

Used in tandem with [`../vllm-pr40391-rebased/`](../vllm-pr40391-rebased/) (Gemma 4
per-token-head fp8 KV) on the `docker-compose.gemma-mtp-fp8.yml` compose. Different
file surfaces (this is `tool_parsers/`, that is `model_executor/` + `v1/core/` + `v1/worker/`),
so they don't conflict.

## Composes that mount this

`docker-compose.gemma-mtp-fp8.yml` — the new fp8-KV variant on the gemma-mtp track.

Optionally consider mounting on `docker-compose.gemma-mtp.yml` and `gemma-mtp-tp1.yml` too
since these tool-parser fixes don't require fp8 KV — they help any Gemma 4 + MTP
streaming tool-call workflow. Hold pending validation that the patches don't introduce
regressions on the bf16 KV path.
