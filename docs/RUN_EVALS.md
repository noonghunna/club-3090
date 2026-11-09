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
| Strata (`strata/…`) | thinking | none: per request | ❌ **no**: the server decodes greedy. Send the card's row on both legs ([below](#engines-with-their-own-defaults)) |
| Project Maya (`strata-maya/…`) | thinking | none: per request | n/a: GLM's card has one row, which the server serves on both legs |
| bucko-vllm (`bucko-vllm/…`) | thinking | none: per request | ❌ **no**: the server keeps the thinking row |
| HyperQwen (`vllm-hyperqwen/…`) | thinking | none: per request | ❌ **no**: the server keeps the thinking row |
| ExLlamaV3 (`exllamav3/…`) | thinking | none: per request (`REASONING=false` only turns off the parser; leave it) | ❌ **no**: TabbyAPI's generic sampler. Send the card's row on both legs |
| PrismML llama.cpp (`llama-cpp-prism/…`) | thinking | `INSTRUCT=1` | ✅ yes |
| composes without a switch | per compose | per compose | read the header |

"None: per request" means there is no boot switch: benchlocal's `--no-thinking` turns thinking off
in each request. Those six engines also differ on effort and budget; read
[Engines with their own defaults](#engines-with-their-own-defaults) before running them.

Gated slugs (🧪 / 🐣) need forcing, and the form differs per script: `bash scripts/switch.sh --force <slug>`
or `FORCE=1 bash scripts/launch.sh --variant <slug>`. (`launch.sh … --force` fails with *Unknown flag*.)

### Thinking leg (the default on most thinking models)

```bash
bash scripts/switch.sh --force <slug>
REASONING_EFFORT=low bash scripts/quality-test.sh --full --enable-thinking
```

`low` is what the Qwen3.8 vLLM and SGLang composes serve. The engines in the next section serve
other efforts.

### Instruct leg

vLLM and llama.cpp, where the server serves the instruct row:

```bash
ENABLE_THINKING=false bash scripts/switch.sh --force <slug>   # llama.cpp: INSTRUCT=1 instead
bash scripts/quality-test.sh --full --no-thinking
```

SGLang, where it doesn't: send the row yourself, as benchlocal flags after `--`. Explicit sampler
flags switch the default server sampling off for that run (benchlocal refuses the mix). Don't use
`--both-modes` on an `sgl/` slug: both its legs run on the server's sampler.

```bash
ENABLE_THINKING=false bash scripts/switch.sh --force <sgl-slug>
bash scripts/quality-test.sh --full --no-thinking \
  -- --temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0.0
```

**`presence_penalty`:** the Qwen3.8 card's instruct row sets it to 1.5. Neither vLLM nor SGLang can
apply it server-side, so our published instruct numbers on those engines ran without it. To score
the card's complete row, send the whole row from the client, all after `--`:

```bash
bash scripts/quality-test.sh --full --no-thinking \
  -- --temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --extra-body '{"presence_penalty": 1.5}'
```

`--extra-body` on its own doesn't reach the request. It leaves server sampling on, and under server
sampling benchlocal drops every sampler key, `presence_penalty` included. The explicit sampler flags
are what turn server sampling off. Say so when you post, since that result isn't comparable with the
published ones.

### Engines with their own defaults

Strata, Project Maya (its GLM, DeepSeek and MiniMax engines), bucko-vllm, HyperQwen, ExLlamaV3,
PrismML's llama.cpp and QEngine each serve a different effort, and several of them don't serve the
card's sampler on every leg. The Qwen3.8 card's
rows, for where the table says to send one:
- **thinking:** `--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0`
- **instruct:** `--temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0.0`

| Engine | Thinking-leg effort (as served) | Efforts it accepts | Send from the client (after `--`) | Also |
|---|---|---|---|---|
| Strata | `xhigh` | low, medium, xhigh (`high` and `max` mean xhigh) | the thinking row on the thinking leg, the instruct row on the instruct leg | no server reasoning budget |
| Project Maya, GLM-5.3-Flash | **`max`** | low, high, max (`medium` means high and `xhigh` means max: GLM's levels are off, low, high and max). `none`, `off` or **`minimal`** turns thinking off, unlike the Qwen3.8 templates, where `minimal` means low | nothing: the server serves GLM's one row (temperature 1.0, top_p 0.95) | a 32K reasoning budget, clamped per request to leave room for the answer. Maya sizes its RAM tier from free memory, so drop the page cache before each boot |
| Project Maya, DeepSeek-V4.1-Flash (`maya-ds4`) | `high` | low, high, max (`medium` means high, `xhigh` means max), which DeepSeek's template turns into its own 1–100 effort budget. `none`, `off` or `minimal` turns thinking off | nothing: the server serves Maya's DeepSeek default (temperature 1.0, top_p 1.0) | the same 32K reasoning budget. One request at a time |
| Project Maya, MiniMax-M3.1-Flash (`maya-mm3`) | **`auto`** (the model decides whether to think) | auto, low, medium, high, xhigh, max: all five levels reach MiniMax's template. `none`, `off` or `minimal` turns thinking off | nothing: the server serves the model card's temperature 1.0, top_p 0.95 | the same 32K reasoning budget. Set `REASONING_EFFORT=auto` to score the served default: unset, benchlocal sends `high`. One request at a time |
| bucko-vllm | `xhigh` | **low, medium, xhigh only**: `high` or `max` fails every request | the instruct row on the instruct leg | a 16,384-token server reasoning budget, the same as quality-test's cap: run the thinking leg with `THINKING_MAX_TOKENS=20480` |
| HyperQwen | `low` | low, medium, xhigh (`high` and `max` mean xhigh) | the instruct row on the instruct leg | `REASONING_EFFORT` is also its boot setting, and the boot refuses `none` or `high`: set it on the `quality-test.sh` line only |
| ExLlamaV3 | `xhigh` | low, medium, xhigh | the thinking row on the thinking leg, the instruct row on the instruct leg | boot with your rig's thread count (`THREADS=28` on the reference rig; [EXLLAMAV3.md](engines/EXLLAMAV3.md)) |
| PrismML llama.cpp | **`medium`** | xhigh, medium, low, but **`low` behaves like xhigh** | nothing: `INSTRUCT=1` moves the server to the instruct row | a 32,768-token boot reasoning budget, which quality-test reads and raises its cap for |
| QEngine | `low` (the compose's default; QEngine's own template defaults to xhigh) | low, medium, xhigh (`high` and, from 1.3.4.3, `max` mean xhigh; `minimal` means low); another unknown effort becomes **medium** | from 1.3.5.3, nothing: a request that sends no sampler values gets the Qwen3.8 card's row for its mode (thinking 1.0 / 0.95 / 20 / 0, instruct 0.7 / 0.80 / 20 / 0 with presence 1.5; unmeasured upstream), and a `QE_PRESENCE_PENALTY` set at boot replaces the presence in both. Before 1.3.5.3: the thinking row on the 27B's thinking leg (it served 0.6 / 0.95 / 20 / 0) and the instruct row on both slugs' instruct leg | from 1.3.4.3 the effort map's budgets (4,096 / 16,384 / 32,768 for low / medium / xhigh), as on vLLM and SGLang. `REASONING_EFFORT` is also its boot setting, and the boot refuses anything but low, medium or xhigh: set it on the `quality-test.sh` line only. Its cross-turn loop guard is on by default: after three identical turns it adds a server note to the agent's latest tool result. `QE_LOOP_GUARD_TURNS=0` at boot turns it off ([QENGINE.md](engines/QENGINE.md)). From 1.3.5 it also catches short cycles of calls, and a guarded turn reasons even on the instruct leg (at most 2,048 reasoning tokens; `QE_LOOP_GUARD_THINK=0` turns that off), and sampled output differs from 1.3.4.3's at a fixed seed, so agent packs aren't strictly comparable across that upgrade. 1.3.5.3 changes the server's own sampler to the card's rows, so a run that sends none isn't comparable with 1.3.5's either |

For an engine with no server switch, both legs look like this (Strata shown):

```bash
# thinking leg
bash scripts/switch.sh --force <slug>
REASONING_EFFORT=xhigh bash scripts/quality-test.sh --full --enable-thinking \
  -- --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0

# instruct leg: reboot, and no REASONING_EFFORT
bash scripts/switch.sh --force <slug>
bash scripts/quality-test.sh --full --no-thinking \
  -- --temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0.0
```

- **Set `REASONING_EFFORT` on every thinking leg** to the table's value. Unset isn't neutral: where
  the server advertises an effort control (Project Maya), benchlocal then sends `high`, not the
  served `max` (GLM) or `auto` (MiniMax).
- **On a `--no-thinking` leg, `quality-test.sh` doesn't send it** (from 2026-11-09, and it says so
  in its output). Given an effort, benchlocal turns thinking off by sending `reasoning_effort: "none"`
  and no `enable_thinking: false`, and templates read that `none` differently: QEngine's renders it
  as `medium`, with thinking on ([#1966](https://github.com/noonghunna/club-3090/issues/1966)), and ExLlamaV3's and Swift-1.5's (below) reject it.
  Without it, benchlocal picks the off switch from the model's template, so a `REASONING_EFFORT`
  left in the environment for `--both-modes` or `rebench-full.sh` no longer reaches the off leg.
  Leave it unset on an instruct leg that doesn't pass `--no-thinking`, such as one that turns
  thinking off at boot.
- **Explicit sampler flags turn server sampling off** for that run, as on SGLang. Say which row you
  sent when you post.

### Models that differ on the same engines

Nanbeige4.3-33B (vLLM and SGLang) and Swift-1.5 (mainline llama.cpp) run on engines the tables above
cover, but their templates and cards differ from Qwen3.8's:

| Model (slugs) | Thinking leg | Instruct leg | Sampler | Also |
|---|---|---|---|---|
| Nanbeige4.3-33B (`vllm/nanbeige43-33b-dual`, `sgl/nanbeige43-33b-dual`) | thinking on by default. **No effort control**: its template reads `enable_thinking`, not `reasoning_effort`, so leave `REASONING_EFFORT` unset on both legs | `ENABLE_THINKING=false` at boot | the card's two rows are by scenario, not by mode: agentic and tool use 1.0 / 0.95 (what the vLLM compose serves), reasoning and chat 0.6 / 0.95 (`TEMP=0.6`); it names no top_k or min_p. **SGLang serves 1.0 / 1.0 to a request that sends none**, so send the row you choose from the client, and on vLLM too for the same row on both engines | no reasoning budget |
| Swift-1.5 (`llamacpp/qwen38-27b-swift15-iq3s-mtp`) | `medium` (the compose's default; Swift's own template defaults to xhigh): set `REASONING_EFFORT=medium` | `INSTRUCT=1` (or `ENABLE_THINKING=false`) at boot, which moves the sampler to the instruct row too | the Qwen3.8 card's rows: thinking 1.0 / 0.95 / 20 / 0 as served, instruct 0.7 / 0.80 / 20 / 0 with presence_penalty 1.5 after `INSTRUCT=1` | low, medium and xhigh (`high` and `max` mean xhigh, `minimal` means low, through the compose's template patch); **anything else, `none` included, is HTTP 500**, so never set `REASONING_EFFORT` on its instruct leg. A 32,768-token boot reasoning budget, which quality-test reads and raises its cap for |

### Why each flag

Each of these exists because leaving it out produced a wrong number for us.

| Flag | Why |
|---|---|
| *(no sampling flag)* | The server's sampler is the default (since 2026-10-09): it scores the compose's shipped sampler, not the packs' fixed one. `--pack-sampling` gives the reproducible baseline instead (a greedy instruct leg). On an SGLang instruct leg, send the row yourself (above). |
| `REASONING_EFFORT=low` | Forwarded per request on the thinking leg. Set it to the effort the slug serves, so the run records what it measured: `low` on the Qwen3.8 vLLM and SGLang composes, and the [table above](#engines-with-their-own-defaults) for the newer engines. Effort changes both cost and score, so results at different efforts aren't comparable. Qwen3.8 and GLM-5.3-Flash honour it; a model whose template has no effort control ignores it (Nanbeige4.3-33B: [below](#models-that-differ-on-the-same-engines)). Not sent on a `--no-thinking` leg (from 2026-11-09: there benchlocal would send `none` as the off switch, which QEngine renders as `medium`). |
| `--repeat 3` | For anything you'll quote: under the server's sampler (the default) every answer is sampled, so one draw isn't enough. |
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
