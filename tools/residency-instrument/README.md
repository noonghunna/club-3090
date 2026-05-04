# residency-instrument — Cliff 2b activation-residency probe

Research-grade instrumentation harness used to investigate the multi-turn accumulating-context cliff (Cliff 2b) on single-card vLLM. **Not a stable user-facing tool** — preserved for archeological value and to let cross-rig contributors reproduce the residency findings underlying [docs/CLIFFS.md](../../docs/CLIFFS.md) Cliff 2b.

## Status

- **Built for**: [club-3090#41](https://github.com/noonghunna/club-3090/issues/41) (Cliff 2b investigation, 2026-05-03)
- **Origin**: Codex-authored against `docs/diagnostics/cliff2-followup-residency-instrumentation-brief.md` (gitignored)
- **SLA**: none. May need adaptation for future PyTorch / vLLM / Genesis pin combinations. If it breaks, file an issue and we'll triage on best-effort — this is research-grade tooling, not a shipped feature.

## What it does

Instruments three layers of the vLLM stack via a `sitecustomize.py` hook (auto-loaded by Python on container start):

1. **PyTorch caching allocator** — periodic snapshots of pool sizes, fragmentation, free segments
2. **Engine step boundaries** — request/response transitions, KV pool deltas across turns
3. **DeltaNet GDN forward call sites** — `vllm.attention.backends.turboquant_attn` allocation patterns at the activation peak

Output is CSV per request boundary; default lands in `results/residency-<timestamp>/` (gitignored). Joining instrumentation rows to soak-test turn rows produces the residency-vs-time picture that fed the Cliff 2b root-cause analysis ("fragmentation-dominated, not pool growth — PN12 stays flat at 137 MiB across the cliff fire").

The harness is **observational only** — it monkey-patches request/engine/worker boundaries to write snapshots, but does not change scheduling, memory policy, or kernels.

## How to run

```bash
# Defaults target the known Cliff 2b reproducer (long-text + 2-session × 5-turn soak)
bash tools/residency-instrument/run-instrumented-soak.sh

# Override target compose / soak shape:
VARIANT=vllm/bounded-thinking \
SOAK_SESSIONS=5 SOAK_TURNS=5 \
bash tools/residency-instrument/run-instrumented-soak.sh
```

The script:
1. Boots the target compose with `tools/residency-instrument/sitecustomize.py` mounted as a Python entry point via `PYTHONPATH`
2. Runs `SOAK_MODE=continuous bash scripts/soak-test.sh` against the booted endpoint
3. Joins raw instrumentation rows to soak turn rows
4. Writes the merged output to `results/residency-<timestamp>/`

## Output is local-only

`results/residency-*` is gitignored by repo policy. We don't auto-publish diagnostic outputs because they're large, rig-specific, and most of the analytical value lives in the issue thread + commit log + memory entries that distill what was learned. **If you investigate Cliff 2b on your own rig and want to share findings, the right surface is a comment on the relevant issue (e.g. #41) — paste the summary table, not the raw CSV.**

## Cross-links

- [docs/CLIFFS.md](../../docs/CLIFFS.md) — Cliff 2b mechanism explanation
- [#41](https://github.com/noonghunna/club-3090/issues/41) — investigation thread + cross-rig validation matrix
- [scripts/soak-test.sh](../../scripts/soak-test.sh) — the driver this harness wraps

## Don't run on production composes

The `sitecustomize.py` mount changes Python's startup behavior inside the container. It's safe for investigation runs but slows boot and adds CPU overhead per request. Use a dedicated compose for instrumentation; tear down + re-launch the regular variant when you're done.
