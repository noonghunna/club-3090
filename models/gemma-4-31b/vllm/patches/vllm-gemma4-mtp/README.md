# vLLM Gemma 4 MTP overlay (PR #41745)

Vendored Python files from [vllm-project/vllm#41745](https://github.com/vllm-project/vllm/pull/41745) — adds first-party MTP speculative-decoding support for Google's Gemma 4 "assistant" drafter family (released 2026-05-05).

The compose `docker-compose.gemma-mtp.yml` mounts these files RO over the stock nightly image's vLLM package paths, the same pattern as `models/qwen3.6-27b/vllm/patches/vllm-marlin-pad/`.

## Why this exists

PR #41745 is open + mergeable as of 2026-05-05 evening. Until it merges + propagates to a vLLM nightly tag, every cross-rig user wanting to run Gemma 4 + MTP needs these 7 modified Python files. Vendoring keeps this self-contained inside club-3090 so the compose works without external dependencies.

## Provenance

- Upstream branch: `lucianommartins/my-vllm` @ `lucianommartins/gemma4-mtp`
- Cloned to: `/opt/ai/github/lucianommartins-vllm/`
- File set: 7 files (6 modified, 1 new — `gemma4_mtp.py` + `gemma4.py`)
- Tracked: [PR #41745](https://github.com/vllm-project/vllm/pull/41745) + [docs/UPSTREAM.md](../../../../docs/UPSTREAM.md)

## When to drop this

When PR #41745 merges to vLLM main AND a vLLM `:nightly` tag rebuilds against that change (typically 24-48h after merge). At that point:

1. Bump the `image:` line in `docker-compose.gemma-mtp.yml` to the new nightly with a SHA dated AFTER the merge
2. Remove the entire `# vLLM PR #41745 overlay` volume block from the compose
3. Delete this entire patch directory (`rm -rf models/gemma-4-31b/vllm/patches/vllm-gemma4-mtp/`)
4. Update the [docs/UPSTREAM.md](../../../../docs/UPSTREAM.md) row from "🟡 Open" to "🟢 Landed"

## Companion: transformers 5.8.0 upgrade

The drafter checkpoint (`google/gemma-4-31B-it-assistant`) ships with `model_type: gemma4_assistant`, which only `transformers ≥ 5.8.0` recognizes. The vLLM `:nightly-01d4d1ad3` Docker image ships `transformers 5.7.0`, so the compose's entrypoint runs a `pip install --upgrade transformers==5.8.0` before exec'ing vllm serve. Drop that line when the vLLM nightly rebuilds against transformers ≥ 5.8.0.
