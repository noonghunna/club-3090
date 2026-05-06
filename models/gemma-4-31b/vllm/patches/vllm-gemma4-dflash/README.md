# vLLM Gemma 4 DFlash overlay (PR #41703)

Vendored Python files from [vllm-project/vllm#41703](https://github.com/vllm-project/vllm/pull/41703) — adds first-party **DFlash** (block-diffusion) speculative-decoding support for Gemma 4 + Qwen3.5 models, with the [`z-lab/gemma-4-31B-it-DFlash`](https://huggingface.co/z-lab/gemma-4-31B-it-dflash) drafter (2.9 GB BF16) as the canonical companion.

The compose `docker-compose.gemma-dflash.yml` mounts these files RO over the stock nightly image's vLLM package paths, same pattern as `vllm-gemma4-mtp/` and `models/qwen3.6-27b/vllm/patches/vllm-marlin-pad/`.

## Why this exists

PR #41703 is open + needs-rebase as of 2026-05-06 morning. The PR was authored against an older base than current `main` (pre-`SpecDecodeBaseProposer` refactor); a clean rebase onto `upstream/main` 5d0fd87038b was performed via Codex/ChatGPT delegation, with one manual fix on top:

- `_warn_if_multimodal` (PR's name) → `_raise_if_multimodal` (post-2026-04 main rename) — without this, the override doesn't take effect and DFlash rejects multimodal inputs with `NotImplementedError: Speculative Decoding does not support multimodal models`.

That fix is preserved in `v1/spec_decode/dflash.py` here as a self-contained marker until the PR is itself rebased upstream.

Vendoring keeps this self-contained inside club-3090 so the compose works without external dependencies. Cross-rig users who want DFlash on Gemma 4 don't need to clone the PR fork themselves.

## Provenance

- Upstream branch: `jianc99/dflash-gemma4-fix` (original PR head)
- Local rebase: `/opt/ai/github/jianc99-vllm-dflash-gemma4/` branch `dflash-rebased` — 6 PR commits cherry-picked onto upstream/main `5d0fd87038b`
- File set: 12 modified files (config, model, attention, scheduler, kv cache, spec decode, worker)
- Tracked: [PR #41703](https://github.com/vllm-project/vllm/pull/41703) + [docs/UPSTREAM.md](../../../../docs/UPSTREAM.md)

## DFlash vs MTP on Gemma 4 (TP=2, 2× 3090)

n-sweep on `num_speculative_tokens` (2026-05-06):

| n | Narr wall TPS | Code wall TPS | AL code |
|---|---:|---:|---:|
| 5 | 109 | 141 | 3.99 |
| 6 |  99 | 161 | 4.73 |
| **7 (shipped)** | **95** | **168** | **5.23** |
| 8 |  91 | 167 | 5.36 |
| 15 | 82 | 172 | 6.17 |

vs MTP at n=4: **109 narr / 142 code TPS**.

DFlash dominates MTP on **code (+18%)**; MTP wins on **narrative (+15%)**. They land in genuinely different operating regimes — block-diffusion's larger draft horizon helps deterministic code more than prose. Soak: PASS at n=7 (100 turns, 0 errors / 0 silent-empty / 0 MiB growth, 98.6% TPS retention, p50 55.78 TPS).

## When to drop this

When PR #41703 merges to vLLM main AND a vLLM `:nightly` tag rebuilds against that change. At that point:

1. Bump the `image:` line in `docker-compose.gemma-dflash.yml` to the new nightly with a SHA dated AFTER the merge
2. Remove the entire `# vLLM PR #41703 overlay` volume block from the compose
3. Delete this entire patch directory (`rm -rf models/gemma-4-31b/vllm/patches/vllm-gemma4-dflash/`)
4. Update the [docs/UPSTREAM.md](../../../../docs/UPSTREAM.md) row from "🟡 Open" to "🟢 Landed"

## Companion: transformers 5.8.0 upgrade

Same path as gemma-mtp — the drafter ships with a `model_type` only `transformers ≥ 5.8.0` recognizes, the nightly image ships 5.7.0, the compose entrypoint upgrades it at boot. Drop that line when the vLLM nightly rebuilds against transformers ≥ 5.8.0.

## Conflict with `vllm-gemma4-mtp/`

This overlay and `vllm-gemma4-mtp/` (PR #41745) modify overlapping files (`v1/spec_decode/eagle.py`, `v1/worker/gpu_model_runner.py`, `config/speculative.py`). Run only one variant at a time — `bash scripts/switch.sh` cleanly tears down the previous container before booting the next, so this is enforced operationally.
