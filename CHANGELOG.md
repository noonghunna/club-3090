# Changelog (cross-cutting)

Changes that span the entire stack — engine version pins, script behavior, repo structure. Per-model dated history lives in `models/<name>/CHANGELOG.md`.

## 2026-04-28 (post-launch) — wizard + switcher + health + examples + FAQ + VRAM diagram + Kaitchup citation

Polish pass after the launch tweet went live. All click-through paths now lead somewhere useful instead of "find the right yml file":

- **`scripts/launch.sh`** — interactive wizard. Asks engine → cards → workload, maps to variant, calls switch.sh, runs verify-full. Also accepts `--variant <name>` to skip the wizard, or partial flags (`--engine vllm --cards 1`) and prompts for the rest.
- **`scripts/switch.sh`** — stateless variant switcher. `--list` shows all 13 variants; `<variant>` brings down whatever's running (auto-discovers via docker labels) and brings up the new one, waits for `/v1/models`. `--down` stops without booting.
- **`scripts/health.sh`** — operational health probe. Different lens from verify-full: probes runtime state (KV cache %, recent MTP AL, recent gen TPS, container uptime, VRAM bars, last 5 errors). `--watch` refreshes every 5s. Auto-detects vLLM vs llama.cpp from `/v1/models` response.
- **`docs/EXAMPLES.md`** — one-stop client reference: curl sanity test, Python via openai SDK (chat / streaming / tool calls / vision / reasoning mode), Python via raw `requests` (no SDK), TypeScript / Node, plus connection settings for Open WebUI / Cline / Cursor / LiteLLM. Calls out the Cliff-1 risk for tool-using IDE clients.
- **`docs/FAQ.md`** — ~22 common questions absorbing repeat issue-tracker traffic: 4090/5090 support, NVLink, AMD/Intel/Apple, why both engines, why not Ollama/LM Studio, why MTP not EAGLE, why not GGUF on vLLM, why AutoRound, TPS expectations, what cliffs are, what `vllm#40914` will fix.
- **`docs/img/vram-budget-dual.svg`** + **`.png`** — per-card VRAM allocation diagram across 7 configs (3 single-card, 4 dual-card). Embedded in `models/qwen3.6-27b/README.md` after the variant tables. Visualizes the TP=2 unlock — each card holds half the weights and half the KV, which is why 262K + vision + 2 streams fits on dual but not single.
- **`models/qwen3.6-27b/llama-cpp/README.md`** — quant table marks UD-Q3_K_XL ⭐ as our default with citation to Benjamin Marie's [Kaitchup eval](https://kaitchup.substack.com/p/summary-of-qwen36-gguf-evals-updating). Independent third-party validation of the quant pick (their H100 sweep, our 3090 speed lens — complementary).
- **README** — TL;DR rewritten to lead with the two-routes frame (vLLM dual = max TPS, llama.cpp single = max robustness). Quick-start replaces "cd into compose dir + docker compose up" with `bash scripts/launch.sh`.

## 2026-04-28 — Genesis pin + .env.example + issue templates + perf chart + llama.cpp compose

- **`scripts/setup.sh`** — `GENESIS_PIN` switched from tag `v7.51-stable-2026-04-27` to commit `bf667c7` (Genesis HEAD as of 2026-04-27, semver "v7.54"). This is the exact tree our published TPS numbers were measured against; pinning to commit removes the doc-vs-runtime mismatch.
- **`.env.example`** added at repo root — documents `MODEL_DIR`, `HF_TOKEN`, `CUDA_VISIBLE_DEVICES`, `MEM_UTIL`, `MAX_MODEL_LEN`, `GENESIS_PIN`, `SKIP_GENESIS`, `URL`, `WARMUPS`, `RUNS` with defaults.
- **`.github/ISSUE_TEMPLATE/`** — bug-report template (requires `docker logs --tail 100`, `verify-full.sh` output, `nvidia-smi`, GPU config, compose variant, repo commit) + numbers-from-your-rig template (structured cross-rig TPS contributions). Q&A redirected to GitHub Discussions via `config.yml`.
- **`docs/performance.svg`** + **`docs/performance.png`** — TPS bar chart across 10 single + dual configs, embedded in top-level README.
- **`models/qwen3.6-27b/llama-cpp/compose/`** — two new docker compose files using `ghcr.io/ggml-org/llama.cpp:server-cuda`:
  - `docker-compose.yml` — single slot, 262K ctx, q4_0 KV, vision on (showcase)
  - `docker-compose.concurrent.yml` — 4 parallel slots, 192K ctx pool, vision on
- **First measured TPS for UD-Q3_K_XL on this stack:** 21.22 narr / 20.79 code @ 262K + vision (single 3090, q4_0 KV). Lower than 2026-04-23's 28.5 measurement on Q4_K_M — investigating mainline llama.cpp regression between commits `9ab47e7d8` and current `0d0764dfd`. ngram-mod path measured at 22.04 / 26.11 (+25% on code).

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
