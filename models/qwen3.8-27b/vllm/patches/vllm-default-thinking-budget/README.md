# vllm-default-thinking-budget — a server-side floor for `thinking_token_budget` (patch D)

**What.** vLLM applies a reasoning budget only when a request sends `thinking_token_budget`, and
`--override-generation-config` cannot set one (`ModelConfig.get_diff_sampling_param`'s
`available_params` allowlist excludes it — vllm#54469). This patch makes
`SamplingParams.__post_init__` fall back to `$CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET` when the
request set none, so **every** entrypoint is bounded — including the ones patch
[`vllm-effort-budget`](../vllm-effort-budget/README.md) (B) does not hook: the Responses API,
`/v1/completions`, `/v1/chat/completions/batch`, and anything that builds `SamplingParams`
itself. The compose's `effort_budget.py shell-env` exports the floor as the budget of the
compose's default effort (`REASONING_EFFORT`), so a client that sends nothing gets the same
budget through B or through D. An explicit request budget always wins; B's map budget is set on
the request before `SamplingParams` exists, so D never overrides it.

**Why it is vendored (exception class, `patches.yml` `upstream.status: ours`).** Runaway
reasoning is a measured defect (Flash-Next 2026-10-07: cap cut-offs on cli-40, −30 % retry tokens
recovered by a budget), and the Responses API has no way to carry a budget at all through the
gateway (`thinking_token_budget` is dropped; `reasoning.effort` is forwarded). Ported from the
bucko local layer's validated `default-thinking-budget/apply.py`, same anchor. **Drop trigger:**
upstream ships a server-side default budget that also reaches the Responses API (see *Upstream*).

## The anchor (vLLM v0.31.0)

`vllm/sampling_params.py`, `SamplingParams.__post_init__` — the only occurrence of

```python
        self.thinking_token_budget = validate_thinking_token_budget(
            self.thinking_token_budget
        )
```

Inserted directly after it:

```python
        # [club3090 vllm-default-thinking-budget] server-side floor when the request set none. See models/qwen3.8-27b/vllm/patches/vllm-default-thinking-budget/README.md
        if self.thinking_token_budget is None and not self.trace_decode_token_ids:
            import os as _club3090_os
            _club3090_floor = _club3090_os.environ.get("CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET", "").strip()
            if _club3090_floor:
                self.thinking_token_budget = validate_thinking_token_budget(int(_club3090_floor))
```

Differences from the bucko original: the env is validated at **install** time (a non-integer
refuses the boot instead of raising in every `SamplingParams()`); trace-replay requests
(`trace_decode_token_ids`) get no floor, because vLLM refuses any budget on them; `-1` is no longer
special (shell-env never emits it; `THINKING_BUDGETS=off` unsets the env instead).

`__post_init__` runs in the API server (`to_sampling_params`) and again when the engine core
decodes the msgpack'd `SamplingParams`; both processes carry the env (exported before
`exec vllm serve`), and a non-`None` budget is left alone on decode.

## A budget on a thinking-OFF request is inert

The floor budgets thinking-off requests too (B skips them, D cannot tell). On Qwen3.8 a thinking-off
prompt already ends with a **closed** `<think>\n\n</think>` block, and both enforcement paths only
act inside an **open** think block (v0.31.0 source):

| Runner | Code | Why it is inert |
|---|---|---|
| Model Runner V2 (the runner these MTP / DFlash2 composes serve on, per their headers) | `v1/worker/gpu/sample/thinking_budget.py`: `_update_committed_marker_cache_kernel` finds the last reasoning-start and last natural reasoning-end positions over prompt + output; `_thinking_budget_kernel` line 327 `if last_start < 0 or last_start <= last_end: return` | the closed block puts `last_end` after `last_start`, so the kernel returns before forcing anything — at every step and every draft position — unless the model itself opens a new `<think>` |
| V1 | `v1/sample/thinking_budget_state.py` `_init_state_entry` line 194 `in_think = last_start > last_end` (prompt) → `False`; `_update_think_state` then looks for a start marker in the **output** only and returns while there is none (line 274) | same |

`selftest_floor.py` runs the pinned image's **real V2 kernels** on the CPU (Triton interpreter):
a closed think block with 8 tokens past a budget of 2, with and without draft tokens, forces
nothing; the positive controls (an open think block past the budget, also with drafts) force
`</think>` exactly where the budget is crossed. The live confirmation is
`scripts/probe-effort-budget.sh` (thinking off → no reasoning, no stray `</think>` in content).

## Costs and exposure (read before you rely on it)

- **Every request now carries a budget.** V2's `ThinkingBudgetState.apply` (line 113) returns
  early only when no request in the batch has one (`np.any(self.use_thinking_budget[...])`, line
  115); with the floor, both Triton kernels launch on every decode step. V1 takes the
  tracked-request sampling path (`has_tracked_requests()`). **The decode cost is UNMEASURED** —
  the plan's speed row covers SGLang only. Bench before claiming "free".
- **#58231 exposure widens.** With speculative decoding, a row that has `top_p < 1` and **no
  active top-k** goes through the split top-p kernel, which masks every token once the budget's
  `1e9` forcing logit is present → token 0 until `max_tokens`. Before D a client had to send a
  budget to reach it; now any request that overrides `top_k` to 0/−1 (with `top_p < 1`) can. The
  composes' server default keeps `top_k=20`; do not drop it while #58231 is open.
- **#44676.** Qwen3.5+ opens tool calls inside `<think>`; the budget keeps counting those tokens
  and can force `</think>` into the arguments. With production-size budgets this needs a very long
  in-think tool call; with the probe's tiny budgets it is easy to reach.
- **`-1` is not "unlimited" here.** vLLM folds `-1` to `None` before any hook sees it, so it gets
  the floor (selftest-asserted). Send a large explicit budget instead.
- **Requires a reasoning parser.** A budget on a server without `--reasoning-parser` is refused
  per request (`input_processor.py`). All 39 wired composes pass `--reasoning-parser qwen3`; do
  not wire this patch on one that does not.

## Failure policy

| Situation | Result |
|---|---|
| `CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET` not a non-negative integer | **boot refused** before anything is patched |
| empty / unset | patched; the floor does nothing (stock behaviour) |
| marker present and the AST check passes | no-op, `[vllm-default-thinking-budget] already applied …` |
| marker present but the floor is wrong | **boot refused** |
| anchor missing or present more than once; `__post_init__` or the `trace_decode_token_ids` field gone | **boot refused**, naming the anchor and pointing here |

## Re-anchoring at a pin bump

1. `bash models/qwen3.8-27b/vllm/patches/vllm-default-thinking-budget/selftest.sh vllm/vllm-openai:<new tag>`
   — the `drift_guard.check` in `patches.yml`. Green → nothing to do.
2. Red on the anchor: in the new `vllm/sampling_params.py`, find where `__post_init__` normalises
   `thinking_token_budget` (or where `SamplingParams` is finalised) and move `ANCHOR` in
   `patch_default_thinking_budget.py` there. The floor must run **after** the request value is
   validated and **before** `_verify_args()`.
3. Red on the inertness check: the enforcement kernels changed. Re-read
   `v1/worker/gpu/sample/thinking_budget.py` (and the V1 state) for how a closed think block in the
   prompt is treated before trusting the floor on thinking-off traffic; update `forced_rows()` in
   `selftest_floor.py` to the new kernel signature.
4. Check whether upstream now honours a server-default budget (vllm#54469) — if it reaches the
   Responses API too, drop this patch (see below).

## Selftest

`bash selftest.sh [image]` (default `vllm/vllm-openai:v0.31.0`), no GPU, no network: the compose's
own `shell-env` line exports the floor, install twice (second a byte-identical no-op), the AST
placement check plus a positive control, `selftest_floor.py` (floor / explicit wins / trace replay /
msgpack round trip / Responses / completions / an unhooked chat request / V2-kernel inertness with
positive controls), and the negative controls (pristine file fails the AST check; a non-integer
floor, a duplicated anchor and a removed anchor each refuse).

## Upstream

| Ref | Relation | Drop when |
|---|---|---|
| [vllm#54469](https://github.com/vllm-project/vllm/pull/54469) "Honor server-default thinking_token_budget" (open, `needs-rebase`) | not vendored; D covers it. It lets `--override-generation-config '{"thinking_token_budget": N}'` reach chat + completions `to_sampling_params` | it (or an equivalent) merges in a pinned release **and** reaches every entrypoint we budget, including the Responses API; then set the floor through that flag and drop D |

Tracked in `docs/UPSTREAM.md`, with the watch rows #58231, #58402 / #54467, #44676.
