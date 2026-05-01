# Structured CoT on a single 3090 — bounded thinking that's actually cheaper

Qwen3.6-27B is a thinking model — the `<think>...</think>` block before the answer typically runs ~3000 tokens on coding problems. That's expensive at inference, and on a single RTX 3090 it routinely blows past `max_tokens=4096` before the model emits any code.

[`andthattoo/structured-cot`](https://github.com/andthattoo/structured-cot) showed that a GBNF grammar constraining the `<think>` block to a 3-line `GOAL/APPROACH/EDGE` plan compresses thinking ~22× on HumanEval+ and ~43× on LiveCodeBench v6 with no accuracy loss. They benched on Qwen3.6-35B-A3B MoE Q4_K_M / 1× H100 / llama.cpp.

We re-ran their bench on **our stack** — 27B INT4 dense on a single 3090 with vLLM, MTP n=3, and TurboQuant 3-bit KV — to test whether the technique transfers across the model+quant+engine substitution. It does, and on LiveCodeBench v6 the accuracy gain is bigger than theirs.

This page is the user-facing "what we found, why we shipped it, and how to use it" writeup.

## Bench results (full)

Full HumanEval+ 164 + LiveCodeBench v6 50 (post-2025-01-01 leetcode functional), greedy `temp=0`, `max_tokens=4096`. Three modes per problem:

- **FREE** — standard thinking-mode generation. No grammar.
- **FSM** — GBNF grammar enforced inside the `<think>` block via vLLM xgrammar.
- **PROMPT_TERSE** — system prompt asks for the same compact format. No grammar; control for "did the prompt do the work, or did the grammar?"

### HumanEval+ 164

| Mode | pass@1 | mean think tokens | wall/problem |
|---|---|---|---|
| FREE | 88.4% (145/164) | 2950 | ~64s |
| **FSM** | **92.7%** (152/164) | **96** | ~6s |
| PROMPT_TERSE | 92.1% (151/164) | 68 | ~6s |

**FSM vs FREE: 30.7× compression, +4.3pp pass@1.**

### LiveCodeBench v6 50

| Mode | pass@1 | mean think tokens | wall/problem |
|---|---|---|---|
| FREE | 42.0% (21/50) | 3797 | ~117s |
| **FSM** | **66.0%** (33/50) | **145** | ~30s |
| PROMPT_TERSE | 64.0% (32/50) | 937 | ~38s |

**FSM vs FREE: 26.2× compression, +24.0pp pass@1.**

### Comparison to the published reference

andthattoo's headline numbers were on Qwen3.6-35B-A3B MoE Q4_K_M / H100 / llama.cpp. Different model, different quant family, different engine — same Qwen3-Next family, same grammar.

| | their HE+ | ours HE+ | their LCB v6 | ours LCB v6 |
|---|---|---|---|---|
| FREE pass@1 | 92.1% | 88.4% | 50% | 42% |
| FSM pass@1 | 92.7% | **92.7%** | 64% | **66%** |
| FSM−FREE Δ | +0.6pp | **+4.3pp** | +14pp | **+24pp** |
| FSM compression | 22.4× | **30.7×** | 43.3× | 26.2× |

Our FSM pass@1 lands within 2pp of theirs on both benchmarks — the technique reproduces cleanly across the model+quant+engine substitution. Our accuracy *delta* is bigger because our FREE baseline is weaker (more on this in Caveats).

## When to pick this over the standard `long-text` 130K

Pick `bounded-thinking` when **all three** of:

1. You're calling the API with `extra_body={"structured_outputs": {...}}` (grammar / JSON / regex / etc).
2. You want bounded thinking cost as a structural guarantee (not just "the prompt asks nicely").
3. Your workload tolerates a ~10% per-token TPS hit in exchange for ~30× cheaper think output (per-problem wall-clock is faster, not slower).

Pick `long-text` (the regular variant) when none of those apply. The two composes are otherwise identical (same 130K context, same MTP n=3, same TQ3 KV, same patches). The only difference is one vLLM flag — `--structured-outputs-config.enable_in_reasoning true`, which is what makes grammar enforcement actually fire inside the `<think>` block on this stack.

## How to use it

### 1. Boot the compose

```bash
cd /path/to/club-3090
bash scripts/launch.sh --variant vllm/bounded-thinking
# Or directly:
cd models/qwen3.6-27b/vllm/compose
docker compose -f docker-compose.bounded-thinking.yml up -d
```

Endpoint: `http://localhost:8020/v1` (set `PORT=...` to override).

### 2. Send requests with a grammar

The grammar should constrain only the `<think>` block contents — Qwen3.6's chat template auto-prefixes `<think>\n` and the model emits `</think>\n\n` itself. A minimal HumanEval+-style grammar:

```gbnf
root  ::= think code
think ::= "GOAL: " line "APPROACH: " line "EDGE: " line "</think>\n\n"
line  ::= [^\n]+ "\n"
code  ::= [\x09\x0A\x0D\x20-\x7E]+
```

Python client (uses the OpenAI SDK, points at the local vLLM endpoint):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8020/v1", api_key="dummy")

GRAMMAR = open("grammars/fsm_grammar_no_open.gbnf").read()

r = client.chat.completions.create(
    model="qwen3.6-27b-autoround",
    messages=[
        {"role": "system", "content": "You are an expert Python programmer. Think only inside the required <think> block. After </think>, output runnable Python code directly."},
        {"role": "user", "content": "Write a function add(a, b) that returns the sum."},
    ],
    max_tokens=512,
    temperature=0.0,
    extra_body={"structured_outputs": {"grammar": GRAMMAR}},
)
m = r.choices[0].message
reasoning = (getattr(m, "model_extra", {}) or {}).get("reasoning")
print("think:", reasoning)
print("code :", m.content)
```

The reasoning channel will contain exactly `GOAL: ... APPROACH: ... EDGE: ...`; the content channel will have the runnable code.

### 3. (Optional) Reproduce the bench

```bash
git clone https://github.com/andthattoo/structured-cot.git ~/structured-cot
# Patch fsm_vs_free_eval.py line 711:
#   extra_body={"grammar": grammar}
# →
#   extra_body={"structured_outputs": {"grammar": grammar}}

# Drop the leading "<think>\n" from grammars/fsm_grammar.gbnf since the
# Qwen chat template emits it. Save the variant as fsm_grammar_no_open.gbnf.

cd ~/structured-cot
python fsm_vs_free_eval.py \
  --base-url http://localhost:8020/v1 \
  --model qwen3.6-27b-autoround \
  --tokenizer Qwen/Qwen3.6-27B \
  --dataset humaneval --n-problems 164 --only all \
  --grammar-file grammars/fsm_grammar_no_open.gbnf \
  --max-tokens 4096 --request-timeout 600
```

Runtime: ~3h for HE+ + LCB v6 sequentially on one card. About ~2.5h if you run LCB on a second card in parallel (we did — see the bench artifacts).

## Honest caveats

We caught these during the bench. They're not blockers but they shape how to read the headline.

### Most of the +Δpp gain is "max_tokens trap rescue", not pure reasoning gain

23 of FREE's 29 LCB v6 failures are `empty_code` — the model burned all 4096 tokens reasoning and never emitted code. 15 of FREE's 19 HE+ failures hit the same wall. FSM's bounded thinking dodges the trap, which is a real product win at production max_tokens caps but isn't "structured CoT made the model smarter."

If you re-bench at `max_tokens=8192`, the FREE baseline recovers and the FSM Δ shrinks toward andthattoo's published +0.6pp / +14pp. Compression stays the same (30×/26×). **The compression effect is the reliably-real win**; the accuracy gain is partly artifact of the production envelope.

### MTP × grammar non-determinism

Two greedy `temperature=0` runs of the same problem on two RTX 3090s (same image, same compose, different `CUDA_VISIBLE_DEVICES`) produced different verdicts on a small fraction of problems. Likely from MTP draft-rollback non-determinism interacting with the grammar mask. Per-problem reproducibility caveat — aggregate rates are stable.

### FSM-regress cases are real

6 problems on HE+ (HE/97, 101, 108, 129, 137, 151) and 3 on LCB v6: FREE solves them, FSM doesn't. Pattern is consistent — FSM produces 36–188 think tokens, the rigid `GOAL/APPROACH/EDGE` shape over-compresses and the model loses necessary context. A 2-stage grammar (allow longer thinking on a complexity-budget signal) might bridge this.

### PROMPT_TERSE is competitive

On HE+, PROMPT_TERSE pass@1 is 92.1% vs FSM's 92.7%. On LCB v6, 64% vs 66%. With *no grammar*, just a system prompt asking for the compact format, the model self-disciplines well enough on most problems. The grammar's incremental value is structural enforcement on hard problems where prompting alone fails. Worth being honest: we're not comparing FSM to "no constraint at all" — both compact modes (FSM and PT) crush FREE.

The case for FSM specifically is *guarantees* — the grammar can't be talked out of by an aggressive system prompt, can't drift on hard problems, can't exceed the bound. PT can't promise any of that.

### Sampling-time cost

Grammar mask compute adds ~10% per-token TPS (~61 → 53 TPS in our bench). Per-problem wall-clock is still ~10× faster than FREE because the output is 30× shorter, but per-token rate is slower.

## What's saved on disk

The full bench outputs (results.jsonl, summary.json, per-problem narrative with FREE/FSM/PT think bodies) are at `/home/wasif/structured-cot/runs/full-humaneval-2026-04-30/` and `/home/wasif/structured-cot/runs/full-lcb-v6-gpu1-2026-04-30/`. Internal diagnostics writeup with full setup details, vLLM CLI args, and port surprises is at [`models/qwen3.6-27b/vllm/diagnostics/structured-cot-bench.md`](../models/qwen3.6-27b/vllm/diagnostics/structured-cot-bench.md).

## Credit

[andthattoo/structured-cot](https://github.com/andthattoo/structured-cot) — the technique, the grammar files, the eval harness. We did the cross-rig port to vLLM (a one-line patch + the `enable_in_reasoning` flag dance) and re-benched on a smaller dense model with a different quant + spec-decode stack.
