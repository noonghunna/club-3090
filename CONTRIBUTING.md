# Contributing to club-3090

Thanks for being here. This repo collects working recipes for serving big LLMs on RTX 3090s. Most of its value is in **measured numbers** and **tested configs** — not prose. The contributions that move the most needle are also the easiest: bench your rig, file the data.

---

## What kind of contributions work best

### ✅ Yes please

- **Numbers from your rig.** Different power caps, different motherboards, different models — we want all of it. Use the [Numbers from your rig](https://github.com/noonghunna/club-3090/issues/new?template=numbers-from-your-rig.yml) issue template (no PR needed). High-signal contributions land in `BENCHMARKS` with attribution.
- **Bug reports with the data we ask for.** The [bug report template](https://github.com/noonghunna/club-3090/issues/new?template=bug-report.yml) has the fields we always need: `docker logs --tail 100`, `verify-full.sh` output, `nvidia-smi`, your compose variant, and the repo commit. With those, the first reply is usually a fix or a clear next step instead of "can you send me…".
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
6. **Sign-off:** if your patch involves upstream code (vLLM source, Genesis tree, llama.cpp), credit the upstream author in the docstring/CHANGELOG. We hold a high bar on attribution because Sandermage / the vLLM maintainers / the llama.cpp folks do most of the actual heavy lifting; we just package and bench.

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
