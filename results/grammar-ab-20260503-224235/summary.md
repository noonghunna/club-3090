# Grammar A/B subset bench

- Endpoint: `http://localhost:8020/v1`
- Model: `qwen3.6-27b-autoround`
- Subset seed: `42`
- Target prior HE+ regressions: HE/97, HE/101, HE/108, HE/129, HE/137, HE/151
- Holiday target prior-regressions passed: **4/6**
- Holiday rescues vs current in this run: **1**
- Holiday new failures vs current in this run: **2**
- DeepSeek target prior-regressions passed: **5/6**
- DeepSeek rescues vs current in this run: **1**
- DeepSeek new failures vs current in this run: **1**

| Grammar | Pass@1 (30) | Mean think tokens | Median think tokens | Failures rescued vs current | New failures introduced |
|---|---:|---:|---:|---|---|
| FREE | 28/30 (93.3%) | 3036 | 3464 | HumanEval/101, HumanEval/151 | HumanEval/10, HumanEval/25 |
| GOAL/APPROACH/EDGE | 28/30 (93.3%) | 95 | 82 | baseline | baseline |
| Holiday tagline | 27/30 (90.0%) | 23 | 23 | HumanEval/101 | HumanEval/108, HumanEval/10 |
| DeepSeek scratchpad | 28/30 (93.3%) | 387 | 350 | HumanEval/101 | HumanEval/10 |
| PROMPT_TERSE | 29/30 (96.7%) | 75 | 59 | HumanEval/101, HumanEval/151 | HumanEval/108 |

## Target details

- Holiday target passes: HumanEval/97, HumanEval/101, HumanEval/129, HumanEval/137
- Holiday rescues vs current: HumanEval/101
- Holiday new failures vs current: HumanEval/108, HumanEval/10
- DeepSeek target passes: HumanEval/97, HumanEval/101, HumanEval/108, HumanEval/129, HumanEval/137
- DeepSeek rescues vs current: HumanEval/101
- DeepSeek new failures vs current: HumanEval/10

## Next step

Run phase 3: Holiday and DeepSeek rescued enough of the prior regression cluster to justify a full HE+164 + LCB v6 bench.
