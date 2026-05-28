# Quality testing on club-3090

Operational tests (`verify` / `verify-full` / `verify-stress` / `bench` / `soak-test`) tell you whether a compose **serves** correctly. They don't tell you whether the model **behaves** correctly — whether tool calls land on the right functions, whether instruction-follow constraints hold, whether structured-output stays valid JSON. A compose can pass every operational layer and still ship with degraded behavioral quality from quantization drift or a Genesis env-var flip.

`scripts/quality-test.sh` closes this gap. It wraps [`benchlocal-cli`](https://github.com/noonghunna/benchlocal-cli) — a CLI port of [BenchLocal](https://github.com/stevibe/BenchLocal) bench packs — and runs verifier-backed scenarios against the running compose endpoint.

## Where it sits in the pipeline

```
verify.sh         — fast smoke (15s,        "does it serve")
verify-full.sh    — functional (1-2min,     "does everything work")
verify-stress.sh  — boundary (5-10min,      "does it survive stress")
bench.sh          — throughput (3-5min,     "what's the TPS")
quality-test.sh   — behavioral (10-90min,   "does it produce useful output")  ← THIS
soak-test.sh      — stability (30-60min,    "does it stay healthy over time")
```

Each layer has a different question. Quality testing is the one that catches "passed every other gate but produces wrong tool calls or violates format constraints."

## What the packs measure

Five **deterministic** packs (verifier-backed, no LLM-as-judge — these run without Docker):

| Pack | Dimension | Why it matters for club-3090 users |
|---|---|---|
| **ToolCall-15** | Tool selection + argument correctness | IDE-agent traffic (Cline / OpenCode / Cursor) is 100% tool calls. Genesis env flips like P68/P69 cause silent-empty regressions that die here. |
| **InstructFollow-15** | Constraint-heavy instruction compliance | Catches "ignore the format constraint" drift from cudagraph mode changes or sampling tweaks. |
| **StructOutput-15** | JSON / YAML / markdown structure validity | Bounded-thinking, JSON tool args, FSM-constrained reasoning. |
| **ReasonMath-15** | Numeric reasoning | Code-reasoning correctness; Q4-quant drift surfaces here first. |
| **DataExtract-15** | Field-level extraction accuracy | RAG / document-Q&A workloads. |

Three **sandboxed** packs add execution-backed verification via Docker sandboxes. They're included in `--full` (need Docker; `--no-sandboxed` skips them):

| Pack | Verifier | Why it matters for club-3090 users |
|---|---|---|
| **BugFind-15** | Candidate-fix execution sandbox | Code-repair quality + trap-scenario discipline (no false "found a bug"). |
| **HermesAgent-20** | Multi-tool agent harness (browser / cron / memory / artifact mocks) | Multi-step agentic workflows — chained tool calls, recall, delegation. Closest proxy for IDE-agent stacks. |
| **CLI-40** | Linux command-exec sandbox | Shell/CLI agent tasks (terminal agents like Claude Code / opencode). |

A separate eval-expansion pack, **AiderPolyglot-30** (multi-language code editing across cpp/go/java/js/python/rust), runs *independently* — not bundled into `--quick`/`--medium`/`--full`. Drive it via `benchlocal-cli run --pack aider-polyglot-30 --enable-sandboxed-packs`, or as the `aider` leg of [`rebench-full.sh`](../scripts/rebench-full.sh).

The **reasoning suite** is also separate from `--full`; run it with `--reasoning` when you specifically want code/math/science reasoning signal under thinking-on pack defaults:

| Pack | Verifier | Why it matters for club-3090 users |
|---|---|---|
| **HumanEval+-30** | Code execution sandbox over HumanEval+ functional tests | Small Python coding tasks; catches code-reasoning regressions quickly. |
| **LiveCodeBench-v6-30** | Code execution sandbox over public LCB functional tests | Harder post-2025 coding tasks; exposes budget runaway and algorithmic failures. |
| **GSM-Symbolic-30** | Deterministic `answer_match` exact numeric scoring | Symbolic grade-school math without LLM-as-judge. |
| **GPQA-Diamond** | Deterministic `answer_match` exact letter scoring | Science QA placeholder; gated metadata-only until dataset access is materialized, so it reports `dataset-unavailable` instead of committing restricted data. |

## Modes

| Mode | Packs | Budget | When to run |
|---|---|---|---|
| `--quick` | ToolCall + InstructFollow (2) | ~10-15 min | Per-commit gate; pre-push smoke. The two packs that catch the highest-value regressions for IDE-agent users. No Docker. |
| `--medium` (default) | + StructOutput + DataExtract + ReasonMath (5) | ~25-30 min | Pre-release; pin bumps; new compose authoring. Generates the `Quality:` line for the compose schema. No Docker. |
| `--full` | + BugFind + HermesAgent + CLI (8) | ~45-60 min | Cross-rig comparison; quality A/B vs another quant. **The 3 added packs are Docker-sandboxed — needs Docker.** |
| `--reasoning` | HumanEval+ + LiveCodeBench v6 + GSM-Symbolic + GPQA-Diamond metadata (4) | ~30-90+ min | Dedicated reasoning/code suite. Thinking defaults on for all 4 packs; HumanEval+ and LCB need Docker. |

`--full` runs the sandbox packs by default. `--no-sandboxed` drops `--full` back to the 5-pack deterministic scope (no Docker); `--sandboxed-only` runs just the 3 sandbox packs. `--reasoning` is independent of `--full`; use it for the four reasoning packs, with GPQA skipped until gated data is available.

## Install (one-time)

```bash
pip install git+https://github.com/noonghunna/benchlocal-cli.git
```

Or for development from a local clone of benchlocal-cli:

```bash
pip install -e /path/to/benchlocal-cli
```

## Run

> **Live progress is on by default.** The wrapper forwards `--progress` to
> benchlocal-cli, so per-scenario `[N/M] <pack> <id> …` lines stream to stderr
> as the run advances. Long modes (`--full` ~30–40 min, `--reasoning`
> similar, `--pack aider-polyglot-30` ~25–30 min) otherwise go dark for the
> whole duration with no signal whether anything is wrong mid-run. Pass
> `--no-progress` (or `PROGRESS=0`) for CI / log-volume-sensitive contexts.

```bash
# default --medium against the auto-detected running compose
bash scripts/quality-test.sh

# faster mode (per-commit gate)
bash scripts/quality-test.sh --quick

# full mode (pin bumps, cross-rig comparison)
bash scripts/quality-test.sh --full

# dedicated reasoning suite (thinking-on pack defaults; code packs need Docker)
bash scripts/quality-test.sh --reasoning

# explicit endpoint override
URL=http://localhost:8011 bash scripts/quality-test.sh --quick

# suppress live [N/M] progress for CI / log-volume contexts
bash scripts/quality-test.sh --full --no-progress

# --full includes the 3 Docker-sandboxed packs by default (BugFind/HermesAgent/CLI) — needs Docker
bash scripts/quality-test.sh --full

# skip the sandbox packs (drops --full to the 5-pack deterministic scope, no Docker)
bash scripts/quality-test.sh --full --no-sandboxed

# run ONLY the 3 sandbox packs
bash scripts/quality-test.sh --sandboxed-only

# run individual reasoning packs
bash scripts/quality-test.sh --pack humaneval-plus-30 --enable-thinking --thinking-max-tokens 16384 --timeout-per-case 300
bash scripts/quality-test.sh --pack lcb-v6-30 --enable-thinking --thinking-max-tokens 16384 --timeout-per-case 300
bash scripts/quality-test.sh --pack gsm-symbolic-30
bash scripts/quality-test.sh --pack gpqa-diamond
```

Output:

1. **Markdown table to stdout** — paste-ready for BENCHMARKS quality rows
2. **JSON to `results/quality/quality-<timestamp>.json`** — full per-scenario detail for delta tracking
3. **One-liner suitable for the compose `Quality:` schema field** — paste into compose YAML header

Example output:

```
=== benchlocal-cli --medium  (endpoint: http://localhost:8020, model: qwen3.6-27b-autoround) ===

Pack                       | Pass / Total | Score | p50 latency | p95 latency | Status
ToolCall-15 (v1.0.1)       |   14 / 15    |  93%  |     8.2s    |     12.1s   | ✅
InstructFollow-15 (v1.0.0) |   13 / 15    |  87%  |    11.4s    |     17.8s   | ✅
StructOutput-15 (v1.0.0)   |   15 / 15    | 100%  |     6.9s    |      9.2s   | ✅
DataExtract-15 (v1.0.0)    |   12 / 15    |  80%  |     7.3s    |     10.5s   | ✅
ReasonMath-15 (v1.0.0)     |   11 / 15    |  73%  |    14.2s    |     22.6s   | ✅
─────────────────────────|──────────────|───────|─────────────|─────────────|──────
TOTAL                      |   65 / 75    |  87%  |             |             |

Failure breakdown:
  ToolCall-15           1 verifier_fail  (TC-07: wrong arg value for "filename")
  InstructFollow-15     2 verifier_fail  (IF-03 word-count, IF-09 citation-format)
  DataExtract-15        2 missing_field, 1 wrong_value
  ReasonMath-15         4 wrong_answer   (RM-03, RM-07, RM-09, RM-12)

==========================================================================
Quality: line for compose schema field (paste into compose YAML header):
==========================================================================
Quality:   ToolCall-15 14/15 (93%) · InstructFollow-15 13/15 (87%) · StructOutput-15 15/15 (100%) · DataExtract-15 12/15 (80%) · ReasonMath-15 11/15 (73%) (--medium, packs v1.0.x, 2026-05-09)
```

## Sampling & temperature

By default the packs sample at **temperature 0** (greedy) — deterministic and reproducible, so scores are comparable across rigs and across runs. This is the **canonical** baseline, and it's what regression tracking and cross-config ranking should use.

Two opt-in modes evaluate a model at a non-zero / model-recommended temperature instead. Both **tag the run non-canonical** (markdown header + saved JSON) and refuse to gate CI:

| Mode | What it does | When to use |
|---|---|---|
| `--sampling-from-server` | Omits all sampling params from requests, so the server applies its **compose-configured** defaults; reads them back from `/props` (llama.cpp) and records them. The compose is the single source of truth. | "Evaluate the model exactly as it's served." |
| `benchlocal-cli … --temperature N` (+ `--top-p` / `--top-k` / `--min-p` / `--repeat-penalty`) | Eval at sampling values you specify. Mutually exclusive with `--sampling-from-server`. | When you know the model's recommended temp and want it explicit and recorded. |

The composes ship **model-recommended sampling defaults** (Qwen3.6 `0.6`, Qwopus3.6 `0.8`, Gemma `1.0`), set via the `TEMP` / `TEMPERATURE` / `TOP_P` / `TOP_K` / `MIN_P` / `REPEAT_PENALTY` env (see [`.env.example`](../.env.example)). `--sampling-from-server` inherits whatever the running compose declares — so "serve at the recommended temp" and "eval at the recommended temp" stay in sync from one source.

```bash
# canonical (default): temp 0, reproducible — use for ranking + regression tracking
bash scripts/quality-test.sh --full

# evaluate at the model's served / recommended temperature (inherits the compose default)
bash scripts/quality-test.sh --full --sampling-from-server
SAMPLING_FROM_SERVER=1 bash scripts/rebench-full.sh

# or an explicit temperature, via benchlocal-cli directly.
# NB: invoking benchlocal-cli directly BYPASSES the wrapper's localhost guard. With a
# localhost endpoint + a sandboxed *agentic* pack (HermesAgent-20 runs the agent INSIDE
# the sandbox), you must set BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1 yourself — otherwise the
# in-sandbox agent can't reach the host model and hermes silently scores ~0/20.
# quality-test.sh sets this automatically for localhost URLs (see Limitations).
BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1 \
  benchlocal-cli run --full --endpoint http://localhost:8020 --model <name> --temperature 0.8
```

### Reasoning-on evals

Serving with a model's reasoning flag enabled is necessary but not sufficient: the request also has to send `chat_template_kwargs.enable_thinking=true`. `benchlocal-cli` honors each pack's `default_thinking` metadata, so the dedicated `--reasoning` suite defaults thinking on for all four packs while many format/extraction packs stay answer-only. Use `--enable-thinking` only when you want to force thinking on for every pack in a broader mode such as `--full`:

```bash
# dedicated reasoning suite; default thinking is on for these packs
bash scripts/quality-test.sh --reasoning --thinking-max-tokens 16384

# force thinking on for every full-suite pack
bash scripts/quality-test.sh --full --enable-thinking --thinking-max-tokens 16384

# full rebench: bench.sh + both quality-test.sh legs inherit it
ENABLE_THINKING=1 THINKING_MAX_TOKENS=16384 SAMPLING_FROM_SERVER=1 bash scripts/rebench-full.sh

# TPS bench only
ENABLE_THINKING=1 bash scripts/bench.sh
```

If `/props` or the running container suggests reasoning is enabled but the wrapper is not forcing thinking on globally, `quality-test.sh` / `bench.sh` print a warning; pack defaults still apply, and `--enable-thinking` forces every pack on. `--thinking-max-tokens` now passes through independently and only affects packs whose thinking gate resolves on. The default is 16K; hard LiveCodeBench items may still exhaust that budget, so compare with `benchlocal-cli run --reasoning --no-thinking` when diagnosing budget runaway.

**Why it matters:** a reasoning / exploratory fine-tune (e.g. Qwopus3.6, whose author recommends temp 0.75–1) is *under-represented* at temp 0 or with thinking disabled — greedy, thinking-off decoding collapses the path-exploration the fine-tune was trained for. But high temp and reasoning also *hurt* deterministic packs (DataExtract / StructOutput want exact, repeatable output), so read **per-pack deltas**, not just the total — and keep canonical temp-0 thinking-off as the bar for any apples-to-apples ranking.

## Compose `Quality:` schema field

Each compose's `Profile` header (per [`AGENTS.md`](../AGENTS.md)) can carry an optional `Quality:` line:

```yaml
# Profile (at-a-glance):
#   Model:     Qwen3.6-27B (Lorbus AutoRound INT4 + BF16 mtp.fc preserved)
#   Topology:  Dual 3090 PCIe (TP=2, no NVLink)
#   ...
#   Status:    ✅ Production
#   Quality:   ToolCall-15 14/15 (93%) · InstructFollow-15 13/15 (87%) · StructOutput-15 15/15 (100%) · DataExtract-15 12/15 (80%) (--medium, packs v1.0.x, 2026-05-09)
#   Best for:  General-purpose dual-card vision + tools + long-ctx default ⭐
```

The line documents what the compose was tested on. Cross-rig contributors running quality-test.sh against the same compose can paste their numbers as a sibling row in BENCHMARKS.md.

Compact format (one line) so the schema header doesn't bloat. Full per-scenario detail lives in the JSON saved by quality-test.sh, which can be diffed against past runs for regression tracking.

## What "passing" means

`quality-test.sh` does NOT enforce a hard pass/fail threshold. The script always exits 0 if the runner completes; you decide whether the scores are acceptable.

Suggested gates (informal, not enforced):

| Pack | Suggested floor | Notes |
|---|---|---|
| ToolCall-15 | ≥80% | Below this, IDE-agent users will report regressions |
| InstructFollow-15 | ≥80% | Below this, format-constraint workflows break |
| StructOutput-15 | ≥90% | JSON shape failures are visible immediately to users |
| DataExtract-15 | ≥75% | Slightly more tolerant; field-level scoring is granular |
| ReasonMath-15 | ≥60% | Reasoning quality varies more by quant; treat as informational |

For comparing a new pin / quant / config A/B against the previous version: a >10pp drop on any pack vs the previous baseline is a signal worth investigating before promoting `Status: ✅ Production`.

## What it doesn't replace

- **`bench.sh`** measures throughput, not quality. They're complementary.
- **`soak-test.sh`** measures stability over time. Quality + soak together catch "fast + correct + healthy."
- **NIAH (needle-in-haystack)** tests in `verify-stress.sh` measure long-context retrieval correctness — a different axis than tool-call / instruction-follow.

## Limitations

1. **Sandboxed packs need Docker** — BugFind / HermesAgent / CLI-40 run in Docker-hosted verifier sandboxes. On a host without Docker, run `--medium` (or `--full --no-sandboxed`) for the 5 deterministic packs.
2. **Sandboxed *agentic* packs need a container-reachable model URL** — HermesAgent-20 runs the agent *inside* the sandbox, so it calls the model over the network. A `localhost` / `127.x` / `[::1]` endpoint is the *container's* own loopback, not the host. `quality-test.sh` auto-detects this and exports `BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1` (rewrites the URL → `host.docker.internal` + adds `--add-host`). **If you bypass the wrapper and run `benchlocal-cli` directly against a localhost endpoint, set that env var yourself** — otherwise the in-sandbox agent never reaches the model and hermes silently scores ~0/20. Failure signature: uniform ~timeout-length per-scenario latencies + flat GPU (*not* `turn_count`, which is `0` for hermes regardless of engagement).
3. **Verifier translation is lossy in places** — the upstream BenchLocal evaluators have partial-credit branches we collapsed to pass/fail. See benchlocal-cli's [`docs/EXTRACTOR_NOTES.md`](https://github.com/noonghunna/benchlocal-cli/blob/master/docs/EXTRACTOR_NOTES.md) for the specific surfaces.
4. **Single-run sampling at temperature 0** — each scenario runs once, greedy, by default (see [Sampling & temperature](#sampling--temperature) for the non-canonical override modes). For non-determinism debugging, use `benchlocal-cli run --pack <id> --repeat N`.

For the full pipeline architecture + JSONL pack format, read [benchlocal-cli's docs](https://github.com/noonghunna/benchlocal-cli/tree/master/docs).

## Filing quality regressions

If `quality-test.sh` shows a meaningful regression (e.g., ToolCall-15 drops from 14/15 to 8/15 after a Genesis pin bump), file an issue with:

1. The compose name + the change that triggered it (Genesis pin bump? new quant? cudagraph mode?)
2. The full JSON output from `results/quality/`
3. The pre-change baseline JSON for diff
4. Output of `bash scripts/report.sh --bench` for context (vLLM image SHA, Genesis commit, hardware)

The JSON blobs include enough per-scenario detail to reproduce specific failing scenarios via `benchlocal-cli reproduce` (post-v0.2 subcommand) for upstream debugging.
