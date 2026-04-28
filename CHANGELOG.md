# Changelog (cross-cutting)

Changes that span the entire stack — engine version pins, script behavior, repo structure. Per-model dated history lives in `models/<name>/CHANGELOG.md`.

## 2026-04-28 — Repo created (consolidating + superseding old single + dual repos)

`club-3090` was created to replace two predecessor repos:

- `noonghunna/qwen36-27b-single-3090` — single-card Qwen3.6-27B recipe
- `noonghunna/qwen36-dual-3090` — dual-card Qwen3.6-27B recipe

Reasons for consolidation:
- **Engine-first organization** — most users decide "vLLM or llama.cpp" before "1 card or 2"; the new structure reflects that.
- **Model-agnostic scaffolding** — when we add Qwen3.5-27B / GLM-4.6 / Llama-3.x quants in the future, they slot into `models/<name>/` without restructuring.
- **Single source of truth** — one issue tracker, one Twitter/Reddit/HN URL, no confusion about where to file or read.

**Old repos** remain readable (not deleted, not archived yet) for:
- Existing issue threads that are still active (e.g., the prefill-OOM investigation in single-3090 #1)
- External links from Medium articles, Reddit posts, Twitter
- Historical context for users who landed there via search

The old repo READMEs now have prominent "moved to" banners pointing here. New issues should be filed against `noonghunna/club-3090`.

## See also

- `models/qwen3.6-27b/CHANGELOG.md` — model-specific history (was previously split across two repos; now in one timeline)
- Engine version pins are tracked in the per-engine compose / recipe files. Bumping a pinned vLLM nightly is a per-model change documented in the model's CHANGELOG.
