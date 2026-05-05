# Contributing to club-3090

Thanks for being here. This repo collects working recipes for serving big LLMs on RTX 3090s. Most of its value is in **measured numbers** and **tested configs** — not prose. The contributions that move the most needle are also the easiest: bench your rig, file the data.

---

## What kind of contributions work best

### ✅ Yes please

- **Numbers from your rig.** Different power caps, different motherboards, different models — we want all of it. Use the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) issue template (no PR needed). The template asks for `bash scripts/report.sh --full > my-rig.md` — one ~35-min pass captures hardware (incl. power caps + NVLink topology), stack version, verify-full + verify-stress 7/7, **SOAK_MODE=continuous summary (catches Cliff 2b)**, AND the canonical bench numbers. High-signal contributions land in `BENCHMARKS` with attribution.
- **Bug reports with the data we ask for.** The [bug report template](https://github.com/noonghunna/club-3090/issues/new?template=bug-report.yml) leads with `bash scripts/report.sh > my-rig.md` (add `--verify` to include verify-full output, `--soak` to also run SOAK_MODE=continuous if you suspect a multi-turn agent cliff) — single command captures the rig state we'd otherwise ask for individually (hardware, container state, Genesis patches, KV pool sizing, engine config). With that paste, the first reply is usually a fix or a clear next step instead of "can you send me…".
- **Bug reproductions / minimum repros for upstream issues.** vLLM / llama.cpp / Genesis bugs that affect this stack are most useful when they have a one-paragraph reduction. Drop them in an issue or open a draft PR adding a reproducer to `verify-stress.sh`.
- **New compose variants with measured numbers.** If you've found a config combination that beats one we ship — better TPS, lower VRAM, cleaner stress profile — open a PR with: (a) the `docker-compose.<name>.yml`, (b) `verify-full.sh` output passing, (c) `verify-stress.sh` output passing, (d) a `bench.sh` run (3 warm + 5 measured) showing the delta against the closest existing variant. Bonus points: a footer in the compose file explaining which existing variant you compared against and why this one is better for which workload.
- **New models.** Adding a model is a real lift but well-defined: clone the `models/qwen3.6-27b/` directory structure, populate the engine subdirs, follow the [canonical learnings template](https://github.com/noonghunna/club-3090/blob/master/CLAUDE.md) layout (this repo doesn't ship that file but the convention is documented in `models/qwen3.6-27b/INTERNALS.md`). Open an issue first to scope.
- **Patch experiments.** If you've found a working file-replacement patch for an upstream issue (Genesis-style), the PR shape we expect: a script or `.py` that does the patch idempotently, a `verify-full.sh` pass on a fresh container with the patch applied, and a CHANGELOG entry naming the upstream issue + tracking link. We're happy to ship "buggy-but-fast" patches as opt-in variants if they're cleanly fenced.
- **Doc improvements where there's a genuine clarity win.** "I read this section and was confused; here's what I tried first" is a great PR opener. We're tone-conservative (terse / technical / no marketing fluff), but not allergic to clearer writing.
- **Cross-link to your rig's own published numbers** (Reddit, blog, Twitter). Adding a row to BENCHMARKS or a footnote in the relevant doc with attribution is welcome.

### ❌ Not really

- **Doc style nitpicks.** Reformatting tables, changing list bullets, "this would read better as bullet points" — usually no. We optimize for "reader finds the answer" not "prose is consistent." Open an issue if you really want to make the case.
- **Untested config knobs.** "Adding `--foo-bar 42` because the vLLM docs mention it" without a before/after measurement on this hardware = no. Every flag in our composes is there because we measured it changing something. PRs that add or change flags need numbers.
- **Removing the two-routes framing or the cliffs language** without new data. The "vLLM dual = max throughput, llama.cpp single = max robustness" framing is editorial — built on stress-test findings (no Cliff 1, no Cliff 2 on llama.cpp). Not up for revision unless the underlying data changes (e.g., vllm#40914 lands and Cliff 1 closes).
- **Vendoring upstream packages.** We pin specific commits/SHAs of vLLM, Genesis, llama.cpp images — we don't fork them into the repo. PRs adding `vendor/` directories or copying upstream source get redirected.
- **Marketing-style README rewrites.** "What if we added emojis here?" type changes. The launch tweet handles the marketing surface; the README is for users who already clicked through.
- **Driveby PRs that don't run verify-full.** If a config change affects what the server does, run `bash scripts/verify-full.sh` on it. A passing verify is the entry ticket for compose / patch / script changes.

---

## Issues vs Discussions — where to file what

Two GitHub channels, two different shapes of conversation. Picking the right one keeps the right people seeing the right thing:

| You have | File as | Why |
|---|---|---|
| **A reproducible failure** (boot OOM, 500, silent-empty, stuck engine) — anything with a stack trace, error log, or `report.sh` dump | **[Issue](https://github.com/noonghunna/club-3090/issues/new/choose)** | Bug-tracking lives in issues — open/closed state, labels, assignees, searchable when others hit the same symptom. The [bug report template](https://github.com/noonghunna/club-3090/issues/new?template=bug-report.yml) leads with `bash scripts/report.sh > my-rig.md` so the rig context is in the issue, not split across replies |
| **A measured rig + bench** (cross-rig data, BENCHMARKS row request) | **[Numbers from your rig issue](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml)** | Same reasoning — structured one-shot data point lands cleanly as a closeable issue, ends up in BENCHMARKS with attribution |
| **A design question** ("should the launch wizard ask X?", "why does dual-turbo use TQ3 not k8v4?") | **[Discussion](https://github.com/noonghunna/club-3090/discussions)** | Open-ended, branchy, no single resolution state |
| **A welcome / first-impression / sharing your rig** | **[Discussion](https://github.com/noonghunna/club-3090/discussions)** | Tree-of-comments shape fits casual interaction; doesn't pollute the issue tracker |
| **Asking for help and not sure if it's a bug** | **Discussion**, until clarity emerges; then **fork the bug-shaped piece into an issue** with a cross-link | Avoid prematurely treating an unclear question as a bug; but once the bug is clear, move it to issues so the resolution state is trackable |

**Please don't post log dumps / `report.sh` output / stack traces inside discussions.** It buries the bug under non-bug content (and the reverse: a bug-shaped problem buried inside a "great success!" thread doesn't reach maintainers' issue queue). If a discussion thread accumulates a log dump or a reproducer, the maintainers may ask you to fork the bug-shaped piece into an issue with a cross-link back — not banishment, just keeping each channel doing what it's good at.

---

## Process for non-trivial changes

1. **Open an issue first** for anything bigger than a typo fix or a one-line measurement contribution. We'll either align on shape or explain why we'd land it differently — saves you a wasted afternoon.
2. **Branch off `master`**, work in your fork.
3. **Run the verify suite** before pushing:
   ```bash
   bash scripts/verify-full.sh    # fast functional smoke (~1-2 min)
   bash scripts/verify-stress.sh  # boundary tests (~5-10 min)
   ```
   For compose / patch / script changes, both should pass against your changes. Include the output in the PR description.
4. **Bench the relevant config** if you're touching anything that could move TPS:
   ```bash
   bash scripts/bench.sh
   ```
   Drop the run-by-run output in the PR — `wall_TPS`, `decode_TPS`, `TTFT`, MTP `AL` (where applicable). Mean + CV + n=5 minimum.
5. **Open the PR with a description that answers four questions:**
   - What problem does this solve? (One paragraph.)
   - What's the measured impact? (Numbers.)
   - What did you compare it against? (Specific existing variant or BENCHMARKS row.)
   - What's the trade-off? (TPS vs VRAM vs ctx vs feature support.)
   - The PR template walks you through these — fill it in, don't delete it.
6. **Sign-off:** if your patch involves upstream code (vLLM source, Genesis tree, llama.cpp), credit the upstream author in the docstring/CHANGELOG. We hold a high bar on attribution because Sandermage / the vLLM maintainers / the llama.cpp folks do most of the actual heavy lifting; we just package and bench.

---

## Submitting a new compose variant — full gate list

New compose files (`models/<model>/<engine>/compose/docker-compose.<name>.yml`) get a tighter checklist than other PRs because they ship as a "supported" path that other people boot blind. The PR template enumerates these — bullets here are the *why*:

**Single command captures all of (1)–(5) in one paste:**

```bash
bash scripts/report.sh --full > my-rig.md
```

That runs ~35 min and captures rig + verify-full + verify-stress + soak-continuous + bench. Paste the file contents as a PR comment.

Or run the steps individually if you'd rather:

1. **Rig report** — `bash scripts/report.sh > my-rig.md`, paste as a PR comment. Captures GENESIS_PIN, vLLM image SHA, container CUDA/Python, PCIe lanes per card, power caps, NVLink topology in one pass. Without it future readers can't tell whether your numbers are reproducible against their environment or rig-specific. **This is a merge gate**, not a nice-to-have.
2. **`verify-full.sh` PASS** — fast functional smoke. Confirms the variant boots and serves correctly on your rig.
3. **`verify-stress.sh` 7/7 PASS** — boundary tests including Cliff 2 needle recall (probe 7: 60K + 90K needles). Required for any variant claiming long-context support.
4. **`SOAK_MODE=continuous` summary (single-card variants: required)** — verify-stress catches single-prompt cliffs but **not** the multi-turn accumulating-context cliff (Cliff 2b) that fires at ~25K accumulated tokens on single-card vLLM paths. Soak-continuous is the only test that exercises that regime. See [docs/CLIFFS.md](docs/CLIFFS.md) for the byte-level explanation and [#41](https://github.com/noonghunna/club-3090/issues/41) for the validation matrix. Multi-card variants: strongly recommended (TP=2 and TP=4 escape Cliff 2b, but expectation isn't evidence — verify on your rig).
   ```bash
   SOAK_MODE=continuous SOAK_SESSIONS=5 SOAK_TURNS=5 \
     CONTAINER=<container-name> ENDPOINT=<http://localhost:port> \
     bash scripts/soak-test.sh
   ```
5. **`bench.sh` run** — 3 warmups + 5 measured runs of narrative + code prompts. Report `wall_TPS`, `decode_TPS`, `TTFT`, peak VRAM/card per run, MTP/DFlash AL where applicable.
6. **BENCHMARKS row** — under the appropriate model section, mirroring existing column shape (incl. `Rig` column). Attribution is automatic.
7. **CHANGELOG entry** — in `models/<model>/CHANGELOG.md`.

If any of (1)–(5) don't apply to your variant, say so explicitly in the PR ("N/A — short-prompt-only path; soak-continuous would not exercise the multi-turn regime"). "Forgot to run" gets the PR put on hold; "explained why N/A" gets it merged.

---

## Honesty and reproducibility — the ground rules

- **Don't claim a number you didn't measure.** "Should be ~80 TPS" is fine if labeled as estimate. "Is 80 TPS" needs a bench.
- **Always capture VRAM during benchmarks.** Per-card peak VRAM is a load-bearing piece of information for any TPS comparison. Skip it and the PR will be asked to add it.
- **Pin everything.** vLLM image SHA, Genesis commit, llama.cpp commit. If you bumped one between bench runs, say so.
- **Differentiate "we shipped" from "we measured."** If a config can run a workload at 70 TPS but only with a known-buggy patch combination, that's *measured* but not *shipped*. We label it accordingly in BENCHMARKS.

---

## License

Apache-2.0 (see [LICENSE](LICENSE)). By submitting a PR you agree your contribution is offered under that license.

---

## See also

- [README](README.md) — top-level overview
- [AGENTS.md](AGENTS.md) — guidance for AI coding agents (Claude Code, Cursor, etc.) working in this repo
- [docs/FAQ.md](docs/FAQ.md) — common questions
- [docs/EXAMPLES.md](docs/EXAMPLES.md) — client snippets
- [docs/UPSTREAM.md](docs/UPSTREAM.md) — every upstream issue / PR we depend on or have filed (file new issues here first)
- [models/qwen3.6-27b/INTERNALS.md](models/qwen3.6-27b/INTERNALS.md) — engineering deep dive (where most "why is X this way" answers live)
- [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/) — bug report + numbers-from-your-rig templates
