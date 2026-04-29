# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, Copilot, Continue, etc.) working in this repo. Focused — only conventions an agent wouldn't infer from the code itself.

## Read first

Before making non-trivial changes:

- [`README.md`](README.md) — what the repo is, two-routes framing, repo layout
- [`docs/SINGLE_CARD.md`](docs/SINGLE_CARD.md) / [`docs/DUAL_CARD.md`](docs/DUAL_CARD.md) — pick-by-workload guidance + the cliffs they reference
- [`docs/UPSTREAM.md`](docs/UPSTREAM.md) — every upstream issue / PR we depend on or have filed (see "Upstream issues" section below for why this matters)
- [`models/qwen3.6-27b/INTERNALS.md`](models/qwen3.6-27b/INTERNALS.md) — DFlash forensics, AutoRound rationale, Marlin pad fork, MTP head
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — what kinds of PRs land cleanly + benchmark + verify protocol

## Hardware truths

- 2× RTX 3090 Ampere SM 8.6, PCIe-only, **no NVLink** (and we won't add it).
- Custom all-reduce must be disabled in vLLM/SGLang configs (PCIe topology breaks NVLink-assumed paths).
- No native FP8 compute; FP8 KV is a storage optimization only.
- Speculative decoding using EAGLE / DFlash is blocked on Qwen3-Next family (DeltaNet rollback). MTP works. See [`docs/UPSTREAM.md`](docs/UPSTREAM.md) — vllm#39931.

## Upstream issues — single source of truth

`docs/UPSTREAM.md` tracks every upstream issue / PR we depend on, have filed, or use as context. **Before filing a new upstream issue or referencing an existing one in code/docs, check this file.** When status changes (issue closed, PR merged, pin bumped), update the row.

This rule exists because we previously had upstream links scattered across CHANGELOG, INTERNALS, FAQ, and per-compose comment headers — and they drifted. The tracker file is the canonical place; cross-link to it from anywhere else.

When filing a fresh upstream issue from this work:
1. Add the row to `docs/UPSTREAM.md` first
2. Link the issue back to `noonghunna/club-3090` in the body (helps maintainers see affected user surface)
3. If the upstream eventually merges + propagates, update the row to ✅ Resolved and bump the relevant pin (Genesis commit, vLLM nightly, etc.) in the same commit / PR

## Conventions on this repo

### Bench protocol
3 warm + 5 measured runs. Canonical prompts: 800-word essay (narrative, max_tokens=1000) + quicksort code (max_tokens=800). `temperature=0.6, top_p=0.95, top_k=20`. Capture both wall-time TPS and engine-internal `gen throughput` from logs. **Always capture per-card peak VRAM** alongside TPS.

### Genesis opt-in env vars
Genesis ships ~50 env-gated patches. Some are **targeted bugfixes** (P64 streaming, PN8 memory savings, P3/P5/P6 KV); others are **behavioral mitigations** that silently rewrite the request (P68 = `tool_choice → required`, P69 = inject "must use tool" reminder). Behavioral mitigations need a streaming + large-prompt repro before shipping default-on. We learned this the hard way on 2026-04-29 — see [`docs/UPSTREAM.md`](docs/UPSTREAM.md) → Genesis #9 row + the [club-3090 #2 thread](https://github.com/noonghunna/club-3090/issues/2#issuecomment-4346740245).

If you're considering enabling a new Genesis env var by default in a shipped compose:
1. Read the patch's header docstring in `vllm/_genesis/wiring/`. Does it modify `request.tool_choice`, `request.messages`, or rewrite output?
2. If yes (behavioral): run a streaming repro with prompt > the patch's threshold + a casual user message ("hi") + `tool_choice: auto`. If `finish_reason=stop` with empty content, don't ship default-on.
3. Pure bugfixes (no behavioral override) are fine to ship default-on once they pass `verify-full.sh`.

### CHANGELOG
- `CHANGELOG.md` (cross-cutting) and `models/<name>/CHANGELOG.md` (per-model) are **append-only history**. Don't rewrite past entries even when a finding is superseded — add a new entry. The historical trail is load-bearing for "why did we do X."
- Old entries can reference files / patches that no longer exist. That's fine — leave them.

### Compose variants
- Each variant ships with a **header table** comparing it against sibling composes in the same directory. Update both directions when adding/removing variants.
- Keep the variant set lean. Three composes that overlap badly (similar TPS + similar context + same KV) is worse than two with a clean differentiation. Removed `fast-chat.yml` 2026-04-29 for exactly this reason.

### Documentation
- Don't create new docs proactively. Most non-obvious things belong in `INTERNALS.md`, `FAQ.md`, `SINGLE_CARD.md`, or `DUAL_CARD.md`. New top-level files only when there's a recurring search miss.
- Charts: source `.svg` + exported `.png` at retina resolution (≥1500px wide). Markdown embeds use `.png` (clicking opens a viewable image; SVG opens raw XML). Re-generate with `python3 tools/charts/gen-perf.py` and `gen-vram.py` after editing data.
- For any change that adds a footnote to "this depends on upstream X" — the answer is to link the row in `docs/UPSTREAM.md`, not to inline-cite the upstream URL.

### Tests
- `verify-full.sh` — fast functional smoke (~1–2 min). Runs on every compose change.
- `verify-stress.sh` — boundary cases (longctx ladder + tool-prefill OOM ~5–10 min). Runs on cliff-related changes.
- `bench.sh` — canonical TPS bench. Run when you change anything that could move TPS (compose flags, Genesis env vars, vLLM pin).

### Commits
- New commit per logical change. Don't amend published commits.
- Commit messages: subject ≤72 chars, imperative ("Disable P68/P69..." not "Disabled..."), optional body for "why."
- Don't push without local verify-full + verify-stress passing for the affected compose.

### Hooks / verification
- Pre-flight checks live in `scripts/preflight.sh` and run from `setup.sh` + `launch.sh`. They check docker / GPU / disk before any heavy work. If a check fails, print an actionable `Fix:` hint, never a cryptic mid-run crash.

## Things to NOT do

- Don't add NVLink suggestions. The user has explicitly declined.
- Don't recommend EAGLE / DFlash spec-decode on Qwen3-Next single-card. It's blocked by DeltaNet rollback (see [`docs/UPSTREAM.md`](docs/UPSTREAM.md) → vllm#39931). MTP works.
- Don't enable Genesis behavioral patches (P68/P69 class) by default. They override user intent. If a user wants them, they can flip the env var.
- Don't claim a TPS number you didn't measure. "Should be ~80" labeled as estimate is fine; "is 80" needs a bench.
- Don't compress historical CHANGELOG entries. Append-only.
- Don't scatter upstream issue links across multiple docs. Link to the row in `docs/UPSTREAM.md` instead.
