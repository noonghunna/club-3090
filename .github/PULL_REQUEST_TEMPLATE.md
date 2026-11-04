<!--
Thanks for the PR. Most reviews stall on missing rig context — this template is
the data we'd ask for individually otherwise. Tick what applies; explain N/A
where it doesn't.

For typo / doc-only changes, ignore everything below the "Summary" section.
-->

> **Adding a slug or an engine?** Use its own template instead: add `?template=slug.md` or
> `?template=engine.md` to the compare URL, or run `gh pr create --template slug.md`. A new engine or
> a model we don't have starts with a proposal: [engine](https://github.com/noonghunna/club-3090/issues/new?template=engine-proposal.yml),
> [model](https://github.com/noonghunna/club-3090/issues/new?template=model-proposal.yml).

## Summary

<!-- One paragraph. What problem does this solve, what's the measured impact,
what existing variant did you compare against, what's the trade-off? -->

## Type of change

- [ ] New compose variant (`models/<model>/<engine-dir>/compose/<topology>/<quant>/<serving>.yml`)
- [ ] New patch / sidecar (`models/<model>/<engine>/patches/`)
- [ ] New script or tool (`scripts/`, `tools/`)
- [ ] New model (`models/<new-model>/`)
- [ ] Doc-only / typo
- [ ] Other (describe)

## Verification

- [ ] **Full rig + validation report attached** — single command captures everything:
  ```bash
  bash scripts/report.sh --full > my-rig.md
  ```
  Runs hardware + stack + boot log capture **plus** verify-full + verify-stress + SOAK_MODE=continuous + canonical bench in one ~35-min pass. Paste contents as a PR comment. See [docs/CLIFFS.md](../docs/CLIFFS.md) for why the soak-continuous step is load-bearing (catches Cliff 2b, which verify-stress doesn't).
- [ ] **Profile header complete** (new/changed compose) — `# Profile (at-a-glance):` block with `Status:` set to one enum value (`✅`/`⚠️`/`🧪`/`👁️`/`⏸️`/`🗑️`) and a `Caveats:` line if `⚠️`/`👁️`/`⏸️`/`🗑️`. Enforced by `test-compose-status-drift`; schema in [CLAUDE.md](../CLAUDE.md).
- [ ] **BENCHMARKS row added** — under the appropriate model section, mirroring existing column shape (incl. `Rig` column).
- [ ] **CHANGELOG entry added** in `models/<model>/CHANGELOG.md`.

If you'd rather run the steps separately:

- `bash scripts/report.sh > my-rig.md` (rig only, ~2 sec)
- `bash scripts/verify-full.sh` — fast functional smoke
- `bash scripts/verify-stress.sh` — boundary checks incl. Cliff 2 needles
- `SOAK_MODE=continuous SOAK_SESSIONS=5 SOAK_TURNS=5 bash scripts/soak-test.sh` — required for new single-card composes (catches Cliff 2b)
- `bash scripts/bench.sh` — canonical TPS (3 warmups + 5 measured)

### N/A justifications (if any boxes above are unchecked)

<!-- e.g. "N/A — short-prompt-only path; soak-continuous would not exercise the multi-turn regime"
       or "N/A — doc-only change, no compose touched" -->

## Cross-links

<!-- Issue this PR closes / contributes to. Upstream PRs / issues this depends on.
Sandermage Genesis tickets if relevant. -->

- Closes #
- Related upstream:
