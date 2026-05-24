# Grammar evaluation harness

Tools to A/B-test alternative bounded-thinking grammars against the originally-shipped `andthattoo/structured-cot` GOAL/APPROACH/EDGE grammar (in [`docs/STRUCTURED_COT.md`](../../docs/STRUCTURED_COT.md)). Specifically targeting [Holiday_Purpose_3166's tagline grammar](https://www.reddit.com/r/LocalLLaMA/comments/1sx7w55/) from r/LocalLLaMA.

**The hypothesis:** the K (keywords, 1-5 free tokens) and R (result-keywords, 1-5 free tokens) fields in Holiday's grammar give the model a pressure-relief valve that GOAL/APPROACH/EDGE's rigid 3-line shape lacks. We have 6 documented HE+ regression cases (HE/97, 101, 108, 129, 137, 151) where FSM under-thinks and FREE wins; if Holiday's grammar rescues even some of those without losing too much compression, it's a Pareto improvement worth shipping.

## Files

| File | Purpose |
|---|---|
| `deepseek-scratchpad.gbnf` | Recommended default grammar for bounded-thinking; **vLLM/xgrammar** (uses underscored rule names — does NOT parse on llama.cpp) |
| `deepseek-scratchpad.llamacpp.gbnf` | **llama.cpp variant** of the above — identical language, rule names with no underscores (llama.cpp GBNF is `[a-zA-Z0-9-]` only). Use this for the llama.cpp bounded-thinking compose. Validated on-rig 2026-05-24. |
| `holiday-tagline.gbnf` | Translated grammar, xgrammar-compatible |
| [`TRANSLATION.md`](TRANSLATION.md) | Translation decisions + Phase-1 smoke results |
| `smoke-test.py` | Phase-1 — verify the grammar parses + applies on vLLM |
| `subset-bench.py` | Phase-2 — 30-prompt HE+ A/B vs current grammar + FREE + PROMPT_TERSE |

## Serving notes

The recommended production/default grammar is the DeepSeek scratchpad. **The two engines need different files** (same language, different rule-name dialect): vLLM uses [`deepseek-scratchpad.gbnf`](deepseek-scratchpad.gbnf) as `extra_body={"structured_outputs": {"grammar": ...}}`; **llama.cpp uses [`deepseek-scratchpad.llamacpp.gbnf`](deepseek-scratchpad.llamacpp.gbnf)** (no underscores in rule names) as the request body `grammar` field — the underscored vLLM version silently fails to parse on llama.cpp and falls back to unconstrained generation. See [`docs/STRUCTURED_COT.md`](../../docs/STRUCTURED_COT.md) for engine-specific request examples + validation status.

## How to run

**Phase 1 — smoke (~3 min):**

```bash
python3 tools/grammar-eval/smoke-test.py --base-url http://localhost:8020/v1
# or with auto-boot:
python3 tools/grammar-eval/smoke-test.py --boot
```

5 prompts × tagline-grammar. Pass criterion: all 5 return successfully + match the BODY_RE regex. **Validated 2026-05-03 PM** — see TRANSLATION.md.

**Phase 2 — 30-prompt subset bench (~30-60 min):**

```bash
/tmp/structured-cot-venv/bin/python tools/grammar-eval/subset-bench.py \
  --base-url http://localhost:8020/v1
```

Subset = 6 known FSM-regress problems + 4 known FREE-regress + 20 random HE+. Compares: FREE vs current GOAL/APPROACH/EDGE vs Holiday tagline vs PROMPT_TERSE. Outputs `results/grammar-ab-<timestamp>/{results.jsonl, summary.md}`.

**Headline question phase 2 answers:** of the 6 known HE+ FSM regressions, how many does Holiday's grammar rescue? 0 = drop the experiment. 3+ = phase 3 (full HE+ 164 + LCB v6 50) is worth running.

**Phase 3 — full bench (only if phase 2 shows signal):**

Re-run the same harness against the full HumanEval+ 164 + LiveCodeBench v6 50. ~6-8 hours of compute. Updates the table in [`docs/STRUCTURED_COT.md`](../../docs/STRUCTURED_COT.md).

## Background

The brief that produced these tools is at `docs/diagnostics/grammar-eval-codex-brief.md` (gitignored). The companion LLM-agnostic design exercise prompt (for refining beyond Holiday's design) is at `docs/diagnostics/grammar-design-llm-prompt.md` (gitignored).
