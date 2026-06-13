# Quality baseline corpus (#252)

Curated, **committed** quality 8-pack baselines — the trusted `n≥3` aggregates that
[`scripts/quality-baseline.sh`](../../scripts/quality-baseline.sh) diffs a fresh run
against to catch quality regressions on a config change (quant swap, pin bump, Genesis
env-flip, KV change).

This is distinct from the runtime **measurement-record** TPS corpus
(`scripts/lib/profiles/measurement_record.py`) — that tracks throughput; this tracks
*behavioral quality* (ToolCall / InstructFollow / StructOutput / DataExtract / …).

## What lives here

One JSON per `(registry-slug, thinking-mode)`, named:

```
<slug-with-slashes-as-dashes>__<mode>.json
```

- `<mode>` ∈ `no-thinking` | `enable-thinking`.
- `no-thinking` is **canonical** — temp-0, reproducible, the one a CI-style regression
  gate should diff against. `enable-thinking` is the non-canonical companion for
  reasoning-on configs.
- e.g. `vllm-qwen-35b-a3b-dual__no-thinking.json` ← slug `vllm/qwen-35b-a3b-dual`.

Each file is a benchlocal-cli `RunResult` JSON written with `--repeat N` (N≥3), so a
baseline is an **aggregate**, not a single run — run-to-run noise (~±5–7 / 150 on the
8-pack) doesn't read as a regression.

## How to use

```bash
# capture / refresh a baseline (needs a live endpoint for the slug)
bash scripts/quality-baseline.sh --slug vllm/qwen-35b-a3b-dual --capture

# diff a fresh run vs the committed baseline (the regression check)
bash scripts/quality-baseline.sh --slug vllm/qwen-35b-a3b-dual

# thinking-on variant
bash scripts/quality-baseline.sh --slug vllm/qwen-35b-a3b-dual --mode enable-thinking --capture
```

Endpoint/model are inherited by `quality-test.sh` (auto-detect, or `MODEL=`/`URL=`).
Extra args pass through to `quality-test.sh` → benchlocal-cli (e.g.
`--exit-on-regression` for a CI gate, `--sampling-from-server`). `--dry-run` prints the
resolved command without running.

**Refresh a baseline only deliberately** — a baseline that drifts upward silently to
track a regression defeats the purpose. Re-capture when the config legitimately changes
(new quant, intentional sampling change) and note why in the commit.

## Index

| Baseline file | Slug | Mode | Score (/150) | Captured | Notes |
|---|---|---|---|---|---|
| _(none yet — Phase 2 curates these against the live `RECOMMENDED_DEFAULT_MODELS` configs)_ | | | | | |
