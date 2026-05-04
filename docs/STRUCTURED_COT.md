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

## When to pick a `bounded-thinking-*` variant over the standard `long-text` 214K

Pick a bounded-thinking variant when **all three** of:

1. You're calling the API with `extra_body={"structured_outputs": {...}}` (grammar / JSON / regex / etc).
2. You want bounded thinking cost as a structural guarantee (not just "the prompt asks nicely").
3. Your workload tolerates a ~10% per-token TPS hit in exchange for ~7-30× cheaper think output (per-problem wall-clock is faster, not slower).

Pick `long-text` (the regular variant) when none of those apply. The bounded-thinking variants are otherwise identical to long-text (same 214K context, same MTP n=3, same TQ3 KV, same patches). The only difference is one vLLM flag — `--structured-outputs-config.enable_in_reasoning true`, which is what makes grammar enforcement actually fire inside the `<think>` block on this stack.

### One compose ships — `bounded-thinking.yml` with the DeepSeek scratchpad as the recommended grammar

After the Phase 3 grammar A/B (2026-05-04, full HE+ 164 + LCB v6 50, 5 grammars × 214 problems), the defensible finding is that **DeepSeek scratchpad is the only grammar that doesn't lose combined accuracy vs the original andthattoo G/A/E baseline while improving the harder LCB v6 slice** (LCB +4pp = 2 additional problems on n=50; design signal, not statistical proof). The +0.47pp combined edge (1 problem on n=214) is well below the noise floor and shouldn't be framed as a categorical accuracy win. The stronger evidence for DeepSeek is the LCB tie with Holiday + near-andthattoo HE+ retention, not the +1 combined number alone. The original andthattoo grammar remains excellent and ~4× tighter on think budget; we ship DeepSeek as the recommended default because preserving HE+ accuracy while gaining LCB headroom is the most defensible per-workload posture.

We ship one compose, with the DeepSeek scratchpad as the recommended grammar:

```bash
bash scripts/switch.sh vllm/bounded-thinking
# Then send requests with:
#   extra_body={"structured_outputs": {"grammar": <deepseek-scratchpad GBNF>}}
```

The grammar is at [`tools/grammar-eval/deepseek-scratchpad.gbnf`](../tools/grammar-eval/deepseek-scratchpad.gbnf).

#### Three grammars are available, all client-selectable

The compose is grammar-agnostic — the grammar is selected client-side via `extra_body={"structured_outputs": {"grammar": ...}}`. Three grammars are validated against the bench and live in `tools/grammar-eval/`:

| Grammar | File | HE+ | LCB v6 | Combined | Mean think | Best for |
|---|---|---:|---:|---:|---:|---|
| **DeepSeek scratchpad (recommended ⭐)** | [`deepseek-scratchpad.gbnf`](../tools/grammar-eval/deepseek-scratchpad.gbnf) | 93.9% | **66.0%** | **87.4%** | 541 | General default. Best LCB / harder algorithmic problems. PLAN/NOTE×0-15/VERDICT scaffolding. |
| andthattoo G/A/E (the original) | [`andthattoo/structured-cot`](https://github.com/andthattoo/structured-cot/blob/main/grammars/fsm_grammar.gbnf) | **94.5%** | 62.0% | 86.9% | **134** | Cost-bounded deployments where ~4× tighter think budget matters more than the +0.5pp combined gain. The originally-published technique. |
| Holiday tagline | [`holiday-tagline.gbnf`](../tools/grammar-eval/holiday-tagline.gbnf) | 92.7% | **66.0%** | 86.4% | **24** | Extreme-compression workloads — tagging, classification, log triage. 24-token thinking is the *point*. |

The combined-accuracy spread is within noise (87.4% / 86.9% / 86.4% over 214 problems = ±0.5pp — well below the n=214 confidence interval of ±3pp). What's *not* noise is the per-LCB +4pp win for DeepSeek and Holiday over the original, and the per-HE +0.6pp edge for andthattoo over DeepSeek (also borderline). The honest framing: **all three grammars are good; pick by workload-fit and think-budget cost rather than chasing fractional pass@1 gains**.

## How to use it

### 1. Boot the compose

```bash
cd /path/to/club-3090
bash scripts/switch.sh vllm/bounded-thinking
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

### FSM-regress cases — Phase 3 final (5-grammar A/B at 214 problems)

6 problems on HE+ (HE/97, 101, 108, 129, 137, 151) and 3 on LCB v6: FREE solves them, the original FSM doesn't. Phase 3 ran a 5-grammar A/B on the **full HE+ 164 + LCB v6 50 (n=214)** to see whether any alternative grammar net-beats the andthattoo G/A/E FSM at scale.

The five conditions:

- **FREE** — open thinking, no constraint. Reproduces the upstream baseline.
- **current** (now `andthattoo` G/A/E FSM) — the shipped grammar. GOAL/APPROACH/EDGE 3-line bounded-thinking FSM.
- **Holiday tagline** ([Holiday_Purpose_3166](https://www.reddit.com/r/LocalLLaMA/comments/1sx7w55/) on r/LocalLLaMA, Codex-translated to xgrammar): 5 ultra-short fields (`Q=verb / M=method / K=keywords / R=result-keywords / V=verdict`) with comma-separated free-token lists in K and R as a pressure-relief valve.
- **DeepSeek scratchpad** (proposed by DeepSeek when we ran the design problem past it): `PLAN: …` + 0–15 `NOTE: …` lines + `VERDICT: …`. Variable-count free-text notes bounded by explicit PLAN/VERDICT markers — pressure-relief via repeatable note count rather than Holiday's bounded-token-list approach.
- **PROMPT_TERSE** (existing andthattoo control): same `GOAL/APPROACH/EDGE` shape as the shipped FSM, **delivered as a system-prompt instruction with no grammar mask**.

#### Phase 3 final results (2026-05-04, 0 grammar violations across 1070 generations on andthattoo/Holiday/DeepSeek; 5/1070 = 0.47% on PROMPT_TERSE / FREE shape checks)

##### HumanEval+ 164

| Grammar | Pass@1 | Mean think | Compression vs FREE |
|---|---:|---:|---:|
| FREE | 90.2% (148) | 2983 | 1× |
| **andthattoo (G/A/E FSM)** | **94.5% (155)** | 92 | 32× |
| Holiday tagline | 92.7% (152) | **23** | **128×** |
| **DeepSeek scratchpad** | 93.9% (154) | 401 | 7× |
| PROMPT_TERSE | 92.1% (151) | 73 | 41× |

##### LiveCodeBench v6 50

| Grammar | Pass@1 | Mean think | Compression vs FREE |
|---|---:|---:|---:|
| FREE | 38.0% (19) | 3775 | 1× |
| andthattoo (G/A/E FSM) | 62.0% (31) | 270 | 14× |
| **Holiday tagline** | **66.0% (33)** | **24** | **155×** |
| **DeepSeek scratchpad** | **66.0% (33)** | 1002 | 3.8× |
| PROMPT_TERSE | 50.0% (25) | 984 | 3.8× |

##### Combined (n=214)

| Grammar | Pass@1 | Net vs andthattoo | Mean think |
|---|---:|---:|---:|
| FREE | 78.0% (167) | −19 | 3168 |
| **andthattoo (G/A/E FSM)** | 86.9% (186) | baseline | 134 |
| Holiday tagline | 86.4% (185) | −1 (10 rescued, 11 new fail) | **24** |
| **DeepSeek scratchpad** | **87.4% (187)** | **+1** (9 rescued, 8 new fail) | 541 |
| PROMPT_TERSE | 82.2% (176) | −10 (5 rescued, 15 new fail) | 286 |

##### Phase 1 reproducibility (sanity)

| | Phase 1 | Phase 3 |
|---|---:|---:|
| HE+ FSM Δ vs FREE | +4.3pp | +4.3pp ✓ |
| LCB v6 FSM Δ vs FREE | +24.0pp | +24.0pp ✓ |

**Exact match.** The bench harness is sound and the original results reproduce.

#### Phase 3 findings

1. **DeepSeek scratchpad is the only grammar that doesn't lose combined accuracy vs andthattoo while gaining on the harder LCB slice** (LCB +4pp = 2 problems / n=50, well within sampling noise). The +0.47pp combined edge (1 problem / n=214) is well below the noise floor and shouldn't be framed as a categorical win. What's defensible is the per-dataset pattern: ties andthattoo's HE+ within noise, ties Holiday's best LCB score. Both DeepSeek and Holiday landing at 66% on LCB suggests the common factor is a pressure-relief channel beyond the rigid 3-line shape, not DeepSeek-specific design.

2. **Holiday's extreme compression (24 mean tokens) ties DeepSeek on LCB v6 at 66%** — beats the original andthattoo grammar by 4pp on LCB despite paying 1.8pp on HE+. Strong fit for code-block-tight workloads (LeetCode-style problems, tagging, log triage). Combined is −1 net.

3. **PROMPT_TERSE loses on the long tail (−10 net combined), but the right framing is nuanced.** Phase 2's n=30 finding ("PROMPT_TERSE wins, FSM mask hurts") was subset-selection bias — the 30-problem set was weighted toward the 6 prior regressions, which over-represented PROMPT_TERSE's strength. At full HE+ + LCB, PROMPT_TERSE lands at 82.2% combined (92.1% HE+ / 50% LCB) vs the FSM grammars' 86-87%. So the defensible claim isn't "prompting never helps" — PROMPT_TERSE still rescues HE/151 and ties DeepSeek on the 6-problem prior-regression cluster. The right framing is *prompt-only shaping lacks the long-tail reliability needed across the full distribution, even when it preserves flexibility on rigid-FSM failures*.

4. **HE/151 is one example where all three FSM grammars fail.** Only FREE and PROMPT_TERSE pass it. andthattoo / Holiday / DeepSeek all fail. This is *weak* evidence of a failure class — one universal-FSM-fail can't prove that compact grammars are categorically incapable of representing this reasoning shape. It does show that all three tested FSM shapes excluded something useful (more scratch space, a final verification move, or the ability to choose reasoning shape). A different grammar (e.g. one with an explicit final-validation step) might plausibly rescue it; that's a future research question, not a settled architectural fact.

5. **The dataset matters more than the grammar.** All four grammars converge to within 4pp on HE+ (90.2–94.5%) but spread across **28pp on LCB v6** (38.0–66.0%). LCB amplifies grammar-design effects because problems are denser and longer-form-resistant.

#### Ship decision (2026-05-04)

We ship **one compose**, `bounded-thinking.yml`, with the DeepSeek scratchpad as the recommended grammar — the only grammar that doesn't lose combined accuracy vs the andthattoo G/A/E baseline while improving the harder LCB v6 slice (+4pp on LCB, n=50 = 2 problems; combined +0.47pp is below noise).

The compose is grammar-agnostic — all three validated grammars (DeepSeek, andthattoo G/A/E, Holiday tagline) are available client-side via `extra_body={"structured_outputs": {"grammar": ...}}`. Pick by workload:

- **DeepSeek scratchpad** — default. Strongest combined + best LCB.
- **andthattoo G/A/E** — when ~4× tighter think budget matters (cost-bounded deployments). The originally-published technique.
- **Holiday tagline** — when 24-token think is the point (tagging, classification, log triage). Wins LCB by 4pp over andthattoo too.

We chose to ship one compose rather than three sibling composes because:

1. The combined-accuracy spread across the three grammars is within noise (0.5pp over 214 problems).
2. Three near-identical composes in `switch.sh --list` creates a paradox-of-choice problem the data doesn't justify.
3. The compose itself is grammar-agnostic — selecting the grammar at the client is the natural locus of choice.

PROMPT_TERSE is **not shipped as a default** — at full HE+ + LCB, it lands at 82.2% combined (−10 net vs the FSM-enforced grammars), which is a real long-tail reliability gap. It still has its uses (rescues HE/151 + ties DeepSeek on the prior-regression cluster), so the framing is "FSM enforcement earns its keep across the full distribution," not "prompting never helps."

See [`tools/grammar-eval/`](../tools/grammar-eval/) for the harness, all four grammars, and the per-problem results.

### PROMPT_TERSE — Phase 3 disproved the Phase 2 hypothesis

On Phase 1's first run (n=164 HE+, n=50 LCB v6), PROMPT_TERSE landed at 92.1%/64% — close enough to the original FSM (92.7%/66%) to look competitive. Phase 2's n=30 subset bench then showed PROMPT_TERSE leading at 96.7% — but that was subset-selection bias (the 30 problems were weighted toward the 6 known FSM-regressions, which over-represented PROMPT_TERSE's strength).

**Phase 3 at full HE+ 164 + LCB v6 50 lands PROMPT_TERSE at 82.2% combined** (−10 net vs the FSM-enforced grammars). The right framing isn't "prompt-only never wins" but "prompt-only lacks the long-tail reliability needed across the full distribution" — PT still rescues HE/151 and ties DeepSeek on the 6-problem prior-regression cluster, so it's competitive on the rigid-FSM-fail class but loses on the broader tail. The case for FSM specifically is *guarantees* — the grammar can't be talked out of by an aggressive system prompt, can't drift on hard problems, can't exceed the bound. PT can't promise any of that, which is why FSM enforcement is the default ship.

### Sampling-time cost

Grammar mask compute adds ~10% per-token TPS (~61 → 53 TPS in our bench). Per-problem wall-clock is still ~10× faster than FREE because the output is 30× shorter, but per-token rate is slower.

## What's saved on disk

The full bench outputs (results.jsonl, summary.json, per-problem narrative with FREE/FSM/PT think bodies) are at `/home/wasif/structured-cot/runs/full-humaneval-2026-04-30/` and `/home/wasif/structured-cot/runs/full-lcb-v6-gpu1-2026-04-30/`. Internal diagnostics writeup with full setup details, vLLM CLI args, and port surprises is at [`models/qwen3.6-27b/vllm/diagnostics/structured-cot-bench.md`](../models/qwen3.6-27b/vllm/diagnostics/structured-cot-bench.md).

## Credit

[andthattoo/structured-cot](https://github.com/andthattoo/structured-cot) — the technique, the grammar files, the eval harness. We did the cross-rig port to vLLM (a one-line patch + the `enable_in_reasoning` flag dance) and re-benched on a smaller dense model with a different quant + spec-decode stack.
