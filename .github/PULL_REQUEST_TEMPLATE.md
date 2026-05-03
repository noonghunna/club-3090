<!--
Thanks for the PR. Most reviews stall on missing rig context — this template is
the data we'd ask for individually otherwise. Tick what applies; explain N/A
where it doesn't.

For typo / doc-only changes, ignore everything below the "Summary" section.
-->

## Summary

<!-- One paragraph. What problem does this solve, what's the measured impact,
what existing variant did you compare against, what's the trade-off? -->

## Type of change

- [ ] New compose variant (`models/<model>/<engine>/compose/docker-compose.<name>.yml`)
- [ ] New patch / sidecar (`models/<model>/<engine>/patches/`)
- [ ] New script or tool (`scripts/`, `tools/`)
- [ ] New model (`models/<new-model>/`)
- [ ] Doc-only / typo
- [ ] Other (describe)

## Verification

- [ ] **Rig report attached** — paste contents of `bash scripts/report.sh > my-rig.md` (or `--bench` / `--verify` if relevant) as a PR comment. Captures GENESIS_PIN, vLLM image SHA, container CUDA/Python, PCIe lanes, power caps, NVLink topology — everything we'd otherwise have to ask for one bullet at a time.
- [ ] **`bash scripts/verify-full.sh` PASSES** against this PR's compose. Output attached.
- [ ] **`bash scripts/verify-stress.sh` 7/7 PASSES** against this PR's compose. Output attached.

### For new compose variants ONLY

- [ ] **`SOAK_MODE=continuous` summary attached** — single-card variants: required (catches Cliff 2b at ~25K accumulated tokens, which `verify-stress` does not). Multi-card variants: strongly recommended.
  ```bash
  SOAK_MODE=continuous SOAK_SESSIONS=5 SOAK_TURNS=5 \
    CONTAINER=<container-name> ENDPOINT=<http://localhost:port> \
    bash scripts/soak-test.sh
  ```
  Paste the resulting `summary.md` as a PR comment. See [docs/CLIFFS.md](../docs/CLIFFS.md) for why soak-continuous is the only test that catches the multi-turn cliff, and [Issue #41](https://github.com/noonghunna/club-3090/issues/41) for the validation matrix.
- [ ] **`bash scripts/bench.sh` run included** — 3 warmups + 5 measured runs. Report `wall_TPS`, `decode_TPS`, `TTFT`, peak VRAM/card. MTP `AL` if applicable.
- [ ] **BENCHMARKS row added** — under the appropriate model section, mirroring existing column shape.
- [ ] **CHANGELOG entry added** in `models/<model>/CHANGELOG.md`.

### N/A justifications (if any boxes above are unchecked)

<!-- e.g. "N/A — short-prompt-only path; soak-continuous would not exercise the multi-turn regime"
       or "N/A — doc-only change, no compose touched" -->

## Cross-links

<!-- Issue this PR closes / contributes to. Upstream PRs / issues this depends on.
Sandermage Genesis tickets if relevant. -->

- Closes #
- Related upstream:
