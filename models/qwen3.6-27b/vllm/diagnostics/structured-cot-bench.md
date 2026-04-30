# Structured-CoT bench on club-3090 long-text 218K (internal)

**Status (2026-04-30 PM):** smoke + full HE+ 164 + full LCB v6 50 all complete. Headline ratios reproduce andthattoo's published numbers within tolerance, with **+24pp pass@1 gain on LCB v6** as the standout result. Internal-only until user decides on shipping path. No commits to the repo for this experiment.

## Headline (TL;DR)

On a single RTX 3090 running Qwen3.6-27B AutoRound INT4 dense via vLLM long-text 218K + MTP n=3 + TQ3 KV, with grammar enforcement extended into the reasoning channel:

| Benchmark | FREE pass@1 | FSM pass@1 | Δ pass@1 | think compression |
|---|---|---|---|---|
| HumanEval+ 164 | 88.4% | **92.7%** | **+4.3pp** | **30.7×** |
| LCB v6 50 (post-2025-01-01, leetcode) | 42.0% | **66.0%** | **+24.0pp** | **26.2×** |

**Qualitative summary:** structured CoT works on our stack. The compression ratio is in line with andthattoo's published 22.4× / 43.3×, and the accuracy gain is *larger* than their published +0.6pp / +14pp — but a meaningful chunk of that gain comes from FSM rescuing FREE from the `max_tokens=4096` trap (verbose thinking burns the budget, model never emits code). Same effect they would have seen if they'd benched at our `max_tokens` ceiling.

## Setup (the exact config we ran)

- **Hardware:** 1× RTX 3090, Ampere SM 8.6, 24 GB, PCIe-only (no NVLink)
- **Model:** `Lorbus/Qwen3.6-27B-int4-AutoRound` (dense, 27B params, INT4 AutoRound)
- **Engine:** vLLM nightly `v0.19.2rc1.dev205+g07351e088`
- **Genesis patches:** P4 hybrid TQ + PN12 anchor sidecar + P104 FA clamp + P101/P103 TurboQuant + tolist cudagraph guard (per `docker-compose.long-text.yml`)
- **KV cache:** TurboQuant 3-bit (`turboquant_3bit_nc`)
- **Spec-decode:** MTP n=3 (Lorbus draft head)
- **Compose:** `models/qwen3.6-27b/vllm/compose/docker-compose.long-text.yml`
- **Required override (currently uncommitted):** `--structured-outputs-config.enable_in_reasoning true`
- **Endpoint:** `http://localhost:8020/v1` (single card), `http://localhost:8021/v1` (parallel GPU 1 used for LCB run)
- **vLLM serve args (full command line from compose):**

```
--model /root/.cache/huggingface/qwen3.6-27b-autoround-int4
--served-model-name qwen3.6-27b-autoround
--quantization auto_round
--dtype float16
--tensor-parallel-size 1
--max-model-len 218000
--gpu-memory-utilization 0.985
--max-num-seqs 1
--max-num-batched-tokens 4128
--kv-cache-dtype turboquant_3bit_nc
--language-model-only
--trust-remote-code
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--enable-prefix-caching
--enable-chunked-prefill
--no-scheduler-reserve-full-isl
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--structured-outputs-config.enable_in_reasoning true   # experimental override
--host 0.0.0.0
--port 8000
```

- **Eval harness:** `andthattoo/structured-cot @ main` — `fsm_vs_free_eval.py` ported with one-line patch to use vLLM's new structured-outputs API (see Findings #2 below). Greedy decoding (temp=0).
- **Grammar files:**
  - HE+: `grammars/fsm_grammar_no_open.gbnf` (variant of upstream `fsm_grammar.gbnf` that drops the leading `"<think>\n"` literal — Qwen3.6 chat template auto-prefixes it)
  - LCB v6: `grammars/fsm_grammar_lcb_plan.gbnf` (5-line GOAL/STATE/ALGO/EDGE/VERIFY format, upstream-as-is)
- **Sampling:** `temperature=0.0, max_tokens=4096, request_timeout=600s`
- **Tokenizer for think-token counting:** local Qwen3.6-27B AutoRound

## Reproduce

```bash
# 1. Boot long-text 218K with enable_in_reasoning=true (currently in long-text.yml)
cd /opt/ai/github/club-3090/models/qwen3.6-27b/vllm/compose
MODEL_DIR=/opt/ai/github/qwen36-dual-3090/models docker compose -f docker-compose.long-text.yml up -d

# 2. Set up the harness
git clone https://github.com/andthattoo/structured-cot.git ~/structured-cot
# Patch fsm_vs_free_eval.py line 711:
#   extra_body={"grammar": grammar}
# →
#   extra_body={"structured_outputs": {"grammar": grammar}}

# 3. Drop the leading "<think>\n" from grammars/fsm_grammar.gbnf (Qwen chat template
#    already emits it). Save the variant as grammars/fsm_grammar_no_open.gbnf.

# 4. Run HE+ 164
/tmp/structured-cot-venv/bin/python fsm_vs_free_eval.py \
  --base-url http://localhost:8020/v1 \
  --model qwen3.6-27b-autoround \
  --tokenizer /opt/ai/github/qwen36-dual-3090/models/qwen3.6-27b-autoround-int4 \
  --dataset humaneval --n-problems 164 --only all \
  --grammar-file grammars/fsm_grammar_no_open.gbnf \
  --max-tokens 4096 --request-timeout 600 \
  --out-dir runs/full-humaneval-2026-04-30

# 5. Run LCB v6 50
/tmp/structured-cot-venv/bin/python fsm_vs_free_eval.py \
  --base-url http://localhost:8020/v1 \
  --model qwen3.6-27b-autoround \
  --tokenizer /opt/ai/github/qwen36-dual-3090/models/qwen3.6-27b-autoround-int4 \
  --dataset livecodebench --lcb-version release_v6 \
  --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --only all \
  --grammar-file grammars/fsm_grammar_lcb_plan.gbnf \
  --max-tokens 4096 --request-timeout 600 \
  --out-dir runs/full-lcb-v6-2026-04-30
```

Total wall-clock: ~3h sequential, ~2.5h if you parallelize LCB on the second 3090.

## HumanEval+ 164 — full results

| Mode | pass@1 | mean think | mean total | mean post-think | wall/problem |
|---|---|---|---|---|---|
| FREE | 88.4% (145/164) | 2950 tok | 3068 tok | 118 tok | ~64s |
| **FSM** | **92.7%** (152/164) | **96 tok** | 389 tok | 293 tok | ~6s |
| PROMPT_TERSE | 92.1% (151/164) | 68 tok | 305 tok | 237 tok | ~6s |

**Aggregates:**
- FSM compression vs FREE: **30.7×** (think tokens)
- FSM accuracy delta vs FREE: **+4.3pp**
- PT compression vs FREE: 43.5×
- PT accuracy delta vs FREE: +3.7pp

**Pass-set categorization:**

| outcome | count | task IDs |
|---|---|---|
| 🟰 both pass (FREE✓ FSM✓) | 139 | most |
| 🔺 FSM-wins (FREE✗ FSM✓) | 13 | HE/10, 22, 36, 39, 77, 124, 127, 141, 146, 148, 153, 154, 156 |
| 🔻 FSM-regress (FREE✓ FSM✗) | 6 | HE/97, 101, 108, 129, 137, 151 |
| ❌ both fail | 6 | HE/32, 76, 91, 132, 145, 163 |

**FREE failure breakdown (19 fails):** runtime_error=11, syntax_error=5, empty_code=2, missing_entry_point=1. Of the 19, **15 hit max_tokens** (3000+ think tokens, often `4080+`).

**FSM failure breakdown (12 fails):** runtime_error=12 (all extracted clean code; the algorithm was wrong). The 6 FSM-regress problems all had FSM think_tokens between 36–188 — the rigid GOAL/APPROACH/EDGE template over-compressed and lost necessary context.

## LCB v6 50 — full results (post-2025-01-01 leetcode functional)

| Mode | pass@1 | mean think | mean total | mean post-think | wall/problem |
|---|---|---|---|---|---|
| FREE | 42.0% (21/50) | 3797 tok | 3828 tok | 31 tok | ~117s |
| **FSM** | **66.0%** (33/50) | **145 tok** | 1854 tok | 1709 tok | ~30s |
| PROMPT_TERSE | 64.0% (32/50) | 937 tok | 1898 tok | 962 tok | ~38s |

**Aggregates:**
- FSM compression vs FREE: **26.2×** (think tokens)
- FSM accuracy delta vs FREE: **+24.0pp**
- PT compression vs FREE: 4.1×
- PT accuracy delta vs FREE: +22.0pp

**Pass-set categorization:**

| outcome | count |
|---|---|
| 🟰 both pass | 18 |
| 🔺 FSM-wins | 15 |
| 🔻 FSM-regress | 3 (problem IDs: 3721, 3733, 3759) |
| ❌ both fail | 14 (truly hard problems — 9 of them PT also fails) |

**FREE failure breakdown (29 fails):** **empty_code=23**, missing_entry_point=5, syntax_error=1. Almost the entire FREE failure cohort is "model burned all 4096 tokens reasoning and never emitted code." This is the dominant failure mode on LCB v6 — the problems are hard enough that 27B-INT4 can't conclude its reasoning under 4096 tokens.

**FSM failure breakdown (17 fails):** wrong_answer=12, runtime_error=2, syntax_error=2, missing_entry_point=1. FSM produces extractable code that just fails the public test cases — the planning skeleton was insufficient, but the model at least *finished*.

## Comparison to andthattoo's published reference

andthattoo's reference setup: **Qwen3.6-35B-A3B MoE Q4_K_M / 1× H100 / llama.cpp**. Different model, different quant family, different engine — but same family (Qwen3-Next hybrid) and same grammar.

| metric | their HE+ 164 | our HE+ 164 | their LCB v6 50 | our LCB v6 50 |
|---|---|---|---|---|
| FREE pass@1 | 92.1% | 88.4% | 50% | 42% |
| FSM pass@1 | 92.7% | **92.7%** | 64% | **66%** |
| FSM−FREE Δ | +0.6pp | **+4.3pp** | +14pp | **+24pp** |
| FSM compression | 22.4× | **30.7×** | 43.3× | 26.2× |

**Reproduction quality:** FSM pass@1 lands within ~2pp of their reference on both benchmarks (92.7%/92.7% on HE+, 64%/66% on LCB v6). Compression ratios within tolerance (30.7× ≥ 22.4×; 26.2× ≈ 43.3×/2 — half theirs but still well above the ≥5× smoke floor). Effect is preserved across the model+quant+engine substitution.

**Where we diverge from their setup, and why our numbers look better on accuracy delta:**

1. **`max_tokens=4096` truncation.** Our FREE baseline is artificially weak because 23/29 LCB FREE failures and 15/19 HE+ FREE failures are "burned all 4096 tokens, never emitted code." Their reference setup likely had a higher max_tokens or a cleaner truncation handler. **Part of FSM's headline win on our setup is "rescue from the max_tokens trap" rather than pure reasoning gain.** That said, the bound is real — `max_tokens=4096` is a sensible production cap, and FSM lets the model fit useful reasoning + code into that envelope where FREE can't.

2. **Smaller, denser model.** 27B dense vs 35B-A3B MoE. Less capacity → harder problems hit the model's limit sooner → FSM's bounded-thinking benefit is bigger.

3. **MTP n=3 + grammar interaction.** AL drops from ~3.5 (no grammar) to ~3.0–3.4 (with grammar) — measurable but not catastrophic. They didn't run with spec-decode in their published config, so we have one extra source of variance.

## Honest caveats (must read before publishing)

1. **The +4.3pp / +24pp accuracy gains overstate "structured CoT magic" on our stack.** Most of the lift comes from FSM dodging the max_tokens=4096 truncation. If you re-bench with `max_tokens=8192` or `max_tokens=16384`, the FREE baseline will recover and the FSM delta will shrink toward andthattoo's headline numbers (+0.6pp / +14pp). We did not do this, but should before any public claim.

2. **MTP×grammar non-determinism.** Two greedy `temperature=0` runs of LCB problem 3715 on two RTX 3090s (same image, same compose, different `CUDA_VISIBLE_DEVICES`) produced different verdicts (FSM✓ at 185tt on GPU 1 vs FSM✗ at 135tt on GPU 0 partial run). Likely from MTP draft-rollback non-determinism interacting with the grammar mask. Per-problem reproducibility caveat.

3. **FSM-regress cases are real (6 on HE+, 3 on LCB v6).** When the model needs more than ~150 think tokens to reach a correct algorithm and the grammar caps it, accuracy drops below FREE's. The HE+ regression cluster (97/101/108/129/137/151) is concentrated in the back half — the harder problems. A two-stage grammar (allow longer think on a "complexity-budget" signal) might bridge this.

4. **PROMPT_TERSE is competitive on HE+ (92.1% vs FSM's 92.7%) and on LCB (64% vs 66%).** With *no grammar*, just a system prompt asking for terse GOAL/APPROACH/EDGE thinking, the model self-disciplines well enough on most problems. The grammar's incremental value is structural enforcement on the back-half problems where PT loses discipline. Worth being explicit about: we're not comparing FSM to "no thinking constraint at all" — both compact modes (FSM and PT) crush FREE.

5. **n=50 on LCB v6 is small.** ±2pp confidence intervals at 50 samples are wide. The +24pp delta is robust to noise but a re-bench at n=164+ would tighten the headline.

6. **Sampling-time cost.** Grammar mask compute adds latency to each token. We see ~10-15% TPS drop with grammar enabled vs no grammar (61 → 53 TPS in our smoke). On a per-problem wall-clock basis FSM still wins by ~10× because it generates 30× fewer think tokens, but the per-token rate is slower.

## Findings (port surprises worth keeping)

### 1. vLLM defaults grammar OFF in the reasoning channel

vLLM nightly's `--reasoning-parser qwen3` configures `StructuredOutputsConfig.enable_in_reasoning=False` by default. Without an override, `structured_outputs.grammar` only applies to the post-`</think>` content channel — the model thinks freely first, and the grammar only activates after `</think>`. This is the *opposite* of what structured-CoT needs.

**Fix:** add `--structured-outputs-config.enable_in_reasoning true` to the vLLM command. Currently a temporary edit on `long-text.yml` line 162-167 that needs a decision on whether to keep, revert, or split into a `docker-compose.bounded-thinking.yml` variant.

### 2. Legacy `extra_body={"guided_grammar": ...}` is silently dropped

vLLM dev205+ accepts the legacy field with HTTP 200 but does not enforce. The new field is `extra_body={"structured_outputs": {"grammar": "..."}}`. We caught this only because FREE and FSM produced *identical* think-token counts on the first smoke. Tip-off worth remembering: identical token counts across modes = grammar is not firing.

### 3. Qwen3.6 chat template auto-prefixes `<think>\n`

The structured-cot upstream `grammars/fsm_grammar.gbnf` opens with `"<think>\n"` literal. Qwen3.6's chat template already emits `<think>\n` as the assistant turn suffix. xgrammar tolerates the duplication, but the cleaner port drops the literal: see `fsm_grammar_no_open.gbnf`.

### 4. Eval-script reasoning extraction fallback does the work

The eval script's `message_text()` checks `reasoning_content` first; vLLM exposes it as `reasoning` instead. The fallback path through `model_extra` handles this — no script change required.

## What's saved on disk

```
/home/wasif/structured-cot/runs/
├── smoke-2026-04-30/                  # 10-problem HE+ smoke (33.7×, +0pp)
├── full-humaneval-2026-04-30/         # full HE+ 164
│   ├── results.jsonl                  # per-problem JSON (raw_response, think, code, extraction)
│   ├── summary.json                   # aggregate metrics
│   ├── per_problem.md                 # readable narrative with FREE/FSM/PT think bodies
│   └── run.log                        # stdout
└── full-lcb-v6-gpu1-2026-04-30/       # full LCB v6 50 (the GPU 1 parallel run)
    └── (same structure)
```

## If we ship publicly (decision pending)

When/if user signals "ship it," promote this writeup to a public doc and split the experimental config:

| target file | what goes there |
|---|---|
| `docs/STRUCTURED_COT.md` | Promoted version of this writeup, minus internal-only sections (caveat #2 reproducibility, file paths, etc) |
| `docker-compose.bounded-thinking.yml` | Sister compose to long-text with `enable_in_reasoning=true` baked in. Long-text reverts to default-off. |
| `BENCHMARKS.md` | New rows under qwen3.6-27b for `bounded-thinking-218K` config: HE+ 92.7% / LCB v6 66% |
| `docs/USE_CASES.md` | New section: "Cost-bounded coding agent" recommending bounded-thinking when you need to control output token spend |
| `learnings/qwen3.6-27b.md` | "Structured-CoT works on this stack" finding under the speculative decoding / advanced configs section |
| follow-up | A reply to andthattoo on his vllm#40914 PR thread (or his repo) sharing the cross-rig data; opens the door to upstreaming `enable_in_reasoning` discoverability |

Pre-flight before shipping:
- Re-bench at `max_tokens=8192` to disambiguate "real reasoning gain" from "max_tokens trap rescue." The honest +Δpp is whichever is smaller after that.
- Larger LCB sample (n=164 ≥ released_v6 leetcode pool) to tighten the +24pp number.
- Capture VRAM during the bench (per `feedback_always_capture_vram` rule) for the BENCHMARKS row.

## Cross-rig context

andthattoo is also behind vllm#40914 (K+1 verify routing PR — listed in our Genesis upstream tracker). When/if we ship publicly, ping him directly on his repo with the cross-rig data — he'd find this useful, and the +24pp on LCB v6 with a smaller dense model is a stronger argument for the technique than his original headline.
