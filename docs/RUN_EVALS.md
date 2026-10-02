# Run the evals yourself

How to measure a slug on your own rig: behavioural quality (the **8-pack**) and operational health
(**verify, stress, soak, bench**). It works for any slug in the catalog. The engine-specific parts
(how to switch thinking off, which sampler the server serves) are called out where they differ.

Post what you get with the **Numbers from your rig** issue template. That's how community results
end up in [`BENCHMARKS.md`](../BENCHMARKS.md) and how slugs leave 🧪 experimental.

> Recipes in older announcement posts can be out of date (several changed after they were posted).
> Where a post and this page disagree, trust this page and the compose file's own header.

## 1. One-time setup

The 8-pack runs through [benchlocal-cli](https://github.com/noonghunna/benchlocal-cli). Three of the
eight packs run inside Docker sandbox images that you build once (~30 GB free; `docker system prune`
if tight):

```bash
git clone https://github.com/noonghunna/benchlocal-cli   # already have it? git -C benchlocal-cli pull
pip install -e ./benchlocal-cli
bash benchlocal-cli/tools/build-sandboxes.sh
```

**Update and rebuild before every eval run** (pull, then the last two lines again). Verifier and
harness fixes land often, some of them inside the sandbox images, and an old install scores
differently. Without the images the five deterministic packs still run; the three sandboxed ones
are skipped with a warning.

Slugs with an external drafter need it on disk too. The compose header and the slug's announcement
say which one (for example the Qwen3.8 DFlash2 tiers).

## 2. Operational health

With the slug running and the rig otherwise idle:

```bash
bash scripts/report.sh --full > my-rig.md     # ~43 min
```

It records your hardware (GPUs, power limit, PCIe link, driver) and the stack version, then runs
verify-full, the verify-stress ceiling ladder, soak-continuous, bench and the agentic curve.
Paste the file into the issue.

⚠️ **For throughput numbers, bench on a fresh boot as well.** `--full` runs bench after the stress
ladder and the soak on the same boot, and in that position we measured decode reading **25–27% low**
(prefill 8% low). Restart the container, wait for it to answer, then:

```bash
bash scripts/bench.sh
```

Don't also run `rebench-full.sh`: it re-runs the same gates.

## 3. Quality: the 8-pack, one leg per reasoning mode

Run each reasoning mode on **its own boot**, in both directions. The container keeps its settings
between runs, so skipping the reboot makes the second leg a silent repeat of the first. benchlocal
flags that (per-pack `thinking_validity` in the results JSON, and a warning in the summary).

**Find the slug's thinking switch in its compose header.** Across the catalog today:

| Engine | Default | Switch to instruct | Sampler follows the switch? |
|---|---|---|---|
| vLLM (Qwen3.8 composes) | thinking | `ENABLE_THINKING=false` | ✅ yes: the server serves the card's instruct row, **except `presence_penalty`** (vLLM drops it server-side) |
| SGLang (`sgl/…`) | thinking | `ENABLE_THINKING=false` | ❌ **no**: the server keeps the thinking row. Send the instruct row from the client (below) |
| llama.cpp (Qwen3.8, MiMo) | thinking | `INSTRUCT=1` | ✅ yes, including `presence_penalty` |
| composes without a switch | per compose | per compose | read the header |

Gated slugs (🧪 / 🐣) need forcing, and the form differs per script: `bash scripts/switch.sh --force <slug>`
or `FORCE=1 bash scripts/launch.sh --variant <slug>`. (`launch.sh … --force` fails with *Unknown flag*.)

### Thinking leg (the default on most thinking models)

```bash
bash scripts/switch.sh --force <slug>
REASONING_EFFORT=low bash scripts/quality-test.sh --full --enable-thinking --sampling-from-server
```

### Instruct leg

vLLM and llama.cpp, where the server serves the instruct row:

```bash
ENABLE_THINKING=false bash scripts/switch.sh --force <slug>   # llama.cpp: INSTRUCT=1 instead
bash scripts/quality-test.sh --full --no-thinking --sampling-from-server
```

SGLang, where it doesn't: send the row yourself, as benchlocal flags after `--`. Don't combine this
with `--sampling-from-server` (benchlocal refuses the mix), and don't use `--both-modes` on an `sgl/`
slug, which pins `--sampling-from-server` on both legs.

```bash
ENABLE_THINKING=false bash scripts/switch.sh --force <sgl-slug>
bash scripts/quality-test.sh --full --no-thinking \
  -- --temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0.0
```

**`presence_penalty`:** the Qwen3.8 card's instruct row sets it to 1.5. Neither vLLM nor SGLang can
apply it server-side, so our published instruct numbers on those engines ran without it. To score
the card's complete row, add `--extra-body '{"presence_penalty": 1.5}'` after `--` (it reaches the
request) and say so when you post, since that result isn't comparable with the published ones.

### Why each flag

Each of these exists because leaving it out produced a wrong number for us.

| Flag | Why |
|---|---|
| `--sampling-from-server` | Scores the compose's shipped sampler instead of the pack's `temperature=0`. Without it you measure greedy decoding. (Not on an SGLang instruct leg, above.) |
| `REASONING_EFFORT=low` | Qwen3.8 only; forwarded per request. Pin it so the run records what it measured. Effort changes both cost and score, so results at different efforts aren't comparable. Other models ignore it. |
| `--repeat 3` | For anything you'll quote: with `--sampling-from-server` every answer is sampled, so one draw isn't enough. |
| `URL=http://localhost:<port>` | Optional. The endpoint is auto-detected, but naming it avoids picking up something else you have running. |

### Budgets: set for you

`quality-test.sh` (and `rebench-full.sh`, which calls it) fills these in when you leave them out,
because benchlocal's own per-pack budgets cut long answers off into failures that look like wrong
answers. The run prints them (`[quality-test] budgets: …`) and records them with the results.

| Default | Why | Change it with |
|---|---|---|
| 4,096 completion tokens | benchlocal's ~1,024 truncates long answers into `token_limit` failures. | `--max-tokens N` / `MAX_TOKENS=N` |
| 16,384 tokens on thinking packs | The thinking leg needs the headroom. Set explicitly because benchlocal otherwise gives thinking packs the `--max-tokens` value. | `--thinking-max-tokens N` / `THINKING_MAX_TOKENS=N` |
| 900 s per sandbox model call | Caps one runner-owned model call (cli-40, bugfind-15). benchlocal's 300 s doesn't scale with speed: a 16,384-token answer below ~55 tok/s can't finish in it. | `BENCHLOCAL_MODEL_TURN_TIMEOUT=N` |
| 600 s per hermes episode | The hermes agent's clock is the one that doesn't scale with the token budget (benchlocal: 300 s). | `BENCHLOCAL_HERMES_SUBPROCESS_TIMEOUT_S=N` |

`--pack-budgets` runs benchlocal's own per-pack budgets instead, to compare with results taken that
way. Results from before 2026-10-02 used whichever budgets their command named, so check the command
before diffing scores across that date.

**Per-scenario timeouts are left to benchlocal**, which scales each scenario's clock with the token
budget and the rig's measured speed. Earlier versions of this page passed `--timeout-per-case 600`:
that switches the scaling off and shortens the thinking leg's clock, so leave it out unless you've
confirmed the scaled clock is wrong for your rig.

## 4. Read the results before the score

- **Failure modes first.** `timeout` and `token_limit` are budget artifacts; `verifier_fail` and
  `wrong_answer` are the model. A catastrophic-looking score is often a budget that was too small.
- **Runaways** (answers that never finished) are listed separately. Each is either a cap cutting off
  an answer still in progress or a model loop; the output tells you which.
- **Thinking validity.** A thinking leg where no response reasoned is `silent` (not a valid thinking
  run); under half is `sparse` (usually a model choosing not to think; the scores are mostly
  non-thinking). An instruct leg with any reasoning is `contaminated`.
- **A dead endpoint stops the pack** instead of retrying every scenario, and the summary says so
  (`endpoint-down`). Bring the server back and add `--resume <results.json>` to finish the run.
- **Spec-decode slugs: check the drafter is alive** before trusting a speed number. Look at the
  engine log's acceptance figure (vLLM `Mean acceptance length`, SGLang `accept len:`, llama.cpp
  `draft acceptance rate =`). An acceptance length near 1.0 means the drafter is doing nothing, and
  every functional test still passes.
- **Label anything non-default**: KV type, context, power limit, reasoning effort, sampler overrides.

## Other test commands

Everything above runs against the compose that's currently up. The individual tools:

```bash
bash scripts/bench.sh                                 # throughput: narrative + code, 3 warm-up + 5 measured
bash scripts/quality-test.sh --quick                  # 2 packs, ~5-10 min, no Docker
bash scripts/quality-test.sh                          # --medium: 5 packs, ~15-25 min, no Docker
bash scripts/quality-test.sh --full                   # the 8-pack (150 scenarios; needs the sandbox images)
bash scripts/quality-test.sh --pack aider-polyglot-30 # one named pack
bash scripts/quality-test.sh --reasoning              # HumanEval+/LCB/GPQA/GSM suite, separate from --full
```

`rebench-full.sh` runs the whole pipeline for one model and writes everything under
`results/rebench/<tag>/`. It replaces `report.sh --full` rather than adding to it:

```bash
bash scripts/rebench-full.sh                      # tag derived from the model
bash scripts/rebench-full.sh --skip soak,concurrency   # skip phases
bash scripts/rebench-full.sh --resume             # continue an interrupted run
bash scripts/rebench-full.sh --url http://HOST:PORT --model NAME --engine llama-cpp   # any OpenAI-compatible endpoint
```

## 5. Posting

Open **Numbers from your rig** (issue template) and attach: the `report.sh --full` file, the
fresh-boot `bench.sh` output, and the 8-pack results JSON for each leg you ran (the wrapper prints
its path). Say which slug, and anything you overrode.

---

More detail: [`QUALITY_TEST.md`](QUALITY_TEST.md) (the wrapper's full flag reference) ·
[`CLIFFS.md`](CLIFFS.md) (what soak-continuous is catching) · [`BENCHMARKS.md`](../BENCHMARKS.md).
