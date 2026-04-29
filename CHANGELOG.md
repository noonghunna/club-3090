# Changelog (cross-cutting)

Changes that span the entire stack — engine version pins, script behavior, repo structure. Per-model dated history lives in `models/<name>/CHANGELOG.md`.

## 2026-04-29 — Add `docs/UPSTREAM.md` + `AGENTS.md` (consolidate upstream tracking)

- **`docs/UPSTREAM.md`** (new) — single source of truth for every upstream issue / PR we depend on, have filed, or use as workaround context. Categorized by upstream (vLLM, Genesis, fla-org, FlashQLA, llama.cpp, transformers, SGLang). Status emoji per row (🟢/🔵/🟡/🟠/🔴/⚫/✅/❌) + what-it-unblocks + local workaround. Replaces scattered cross-references in CHANGELOG / INTERNALS / FAQ / per-compose comment headers.
- **`AGENTS.md`** (new, repo root) — concise AI-coding-agent guidance. Includes the rule "before filing or referencing an upstream issue, check + update `docs/UPSTREAM.md`" so the tracker stays the canonical place. Also captures the Genesis-opt-in vetting rule from today's P68/P69 bisection (behavioral mitigations need streaming + large-prompt repro before shipping default-on).
- Cross-links from `README.md`, `CONTRIBUTING.md`, and `models/qwen3.6-27b/INTERNALS.md` "See also" sections updated to point at both new files.
- Convention going forward: file an upstream issue → add row to `UPSTREAM.md` → cross-link from any code/doc that depends on the workaround → update the row when status changes (don't delete; mark ✅/❌ for historical context).

## 2026-04-29 — Remove `fast-chat.yml` + extend P68/P69 disable to all Genesis-loading composes

- **`docker-compose.fast-chat.yml`** — deleted. Post-PN8, `fast-chat.yml` (20K, fp8, vision) and `docker-compose.yml` (48K, TQ3, vision) had effectively the same TPS (52/67 vs 50/67), so fast-chat's only remaining differentiator was "smaller context = ~3s faster boot." Not worth a maintained variant when 20K is also actively bad for IDE-agent users (Copilot Gateway tool-schema preamble alone hits 20K).
- **P68/P69 disable extended to every Genesis-loading compose** — `docker-compose.yml`, `long-vision.yml`, `long-text.yml`, `dual-turbo.yml` all had P68 + P69 enabled and would hit the same silent-stop bug on long agent prompts. Now disabled across the board with consistent inline comments. P64 and PN8 (where appropriate) stay enabled — they're targeted bugfixes, not behavioral overrides.
- `scripts/switch.sh` — dropped `vllm/fast-chat` from the variant map.
- `scripts/launch.sh` — wizard's vLLM workload list shortened from 6 → 5 entries.
- `docs/SINGLE_CARD.md`, `docs/FAQ.md`, `docs/engines/VLLM.md`, model README, vllm/README, patches/README, all sibling compose YAML "see also" tables — references removed or updated to point to `docker-compose.yml` (default) or `tools-text.yml` instead.
- Old CHANGELOG entries that mention `fast-chat.yml` are kept as-is (append-only history).

## 2026-04-29 — Disable Genesis P68/P69 in shipped composes (silent-stop bugfix)

- **`tools-text.yml`** + **`fast-chat.yml`** — `GENESIS_ENABLE_P68_AUTO_FORCE_TOOL` and `GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER` are now commented out (default-off). Caused silent finish_reason=stop with empty content + no tool_calls on greetings and clarifying questions when the prompt exceeded 8000 chars (the patches' threshold). Affected every realistic IDE-agent setup (Cline, Cursor, OpenCode, Copilot Gateway).
- **Bisected via cross-rig data on club-3090 issue #2** (HoodOG1 + tenitram). State A (all 4 env vars on) reproduced the silent stop. State B (P68 off, P69 on) still broken — model loops on "I cannot respond with plain text" then stops mid-reasoning. State D (both P68 and P69 off) — clean: greeting → "Hello! How can I help you today?", tool request → clean `read_file` call.
- **P64 and PN8 stay enabled** — both are real targeted bugfixes (P64 = qwen3coder MTP streaming early-return fix from kotori-yan vllm#39598 backport; PN8 = FP8+MTP draft online-quant memory savings, vllm#40849 backport). Neither overrides user intent.
- Mechanism: `vllm/_genesis/middleware/long_ctx_tool_adherence.py:227` — P68 silently sets `request.tool_choice = "required"` when prompt > 8000 chars; P69 appends "must use a tool" text to the user message. Either alone makes "hi there" + tools fail. Together they're worse.
- Threshold of 8000 chars is too low for IDE agents (typical context: 20-50K). We may file an upstream issue with Sandermage suggesting either raising the default or making opt-in+threshold mandatory-explicit.

## 2026-04-29 — UX polish: pre-flight checks + cards-first wizard + PNG embeds + per-page chart split

- **`scripts/preflight.sh`** (new) — sourceable library of `preflight_docker`, `preflight_gpu [min]`, `preflight_disk <path> <gb>`, `preflight_gpu_idle`, `preflight_running`. Each prints actionable `Fix:` hints on failure rather than a cryptic mid-run crash.
- **`scripts/setup.sh`** — runs pre-flight before any work: docker present, GPU detected, ≥25 GB free at `MODEL_DIR` (override via `PREFLIGHT_DISK_GB`). Catches the most common first-run footguns (no docker, no nvidia driver, full disk).
- **`scripts/launch.sh`** — runs pre-flight, plus inverts the wizard: now asks **cards → workload → auto-pick engine** instead of engine-first. New users can answer "how many GPUs do I have" and "what do I want to do" but rarely "vLLM or llama.cpp" — the engine falls out of the workload pick, with a one-paragraph explanation of why we chose it. `--engine vllm|llamacpp` overrides still work; `--no-preflight` skips the checks.
- **Embedded charts switched from SVG → PNG** in markdown (`README.md`, `docs/SINGLE_CARD.md`, `docs/DUAL_CARD.md`, `models/qwen3.6-27b/README.md`). Clicking an SVG on GitHub opens the raw XML; clicking a PNG opens a normal viewable image. Both files (.svg + .png at retina resolution) ship in the repo — SVG remains the editable source, PNG is what the docs link to. Convention: re-export PNG when SVG changes.
- **Charts split per GPU-count page.** SINGLE_CARD.md and DUAL_CARD.md now embed scoped charts (only their card count), instead of the combined diagrams which show both halves. Top-level README.md and the model README still show the combined views. New files in `docs/img/`: `performance-single.{svg,png}` (6 single configs), `performance-dual.{svg,png}` (4 dual configs), `vram-budget-single.{svg,png}`, `vram-budget-combined.{svg,png}` (renamed from `vram-budget-dual` which was actually combined). The `vram-budget-dual.{svg,png}` filename is reclaimed for genuinely-dual content.
- **`tools/charts/gen-perf.py`** + **`tools/charts/gen-vram.py`** (new) — matplotlib source for all chart files, idempotent re-generation. Run `python3 tools/charts/gen-perf.py` and `python3 tools/charts/gen-vram.py` after editing data. Use uv to bring matplotlib (`uv run --with matplotlib --with numpy python3 ...`).

## 2026-04-29 — Genesis bumped to v7.62.x + PN8 enabled on FP8 paths

- **`scripts/setup.sh`** — `GENESIS_PIN` bumped from `bf667c7` (v7.54) → `917519b` (v7.62.x release, 2026-04-29). Includes Sandermage's PN8 (MTP draft online-quant propagation, backport of vllm#40849), PN11 (Quentin-M streaming tool-call IndexError fix vllm#41142), per-GPU profile auto-recommendations, and TurboQuant k8v4 unlocked on hybrid GDN via P4+P98.
- **PN8 enabled on FP8 paths only** (`docker-compose.tools-text.yml` + `docker-compose.fast-chat.yml`). Cross-rig validation on 3090 measured ~800-900 MiB freed at boot, **Cliff 1 closes on `tools-text.yml`** (25K-token tool prefill no longer OOMs). TPS cost is real but small (~−5% narr / −7% code).
- **PN8 deliberately NOT enabled** on TQ3 paths (`docker-compose.yml` default 48K, `long-vision.yml`, `long-text.yml`). Tested 2026-04-29:
  - Default 48K + 0.92: PN8 is no-op (KV pool unchanged, plenty of activation headroom already)
  - long-vision 192K + 0.98: PN8 grows KV pool by 230 MiB and lifts engine ceiling 192K → 198K, but does NOT close Cliff 1 (the 138 MiB allocate is an FFN intermediate-buffer activation peak, not draft-model footprint)
  - long-text 205K + 0.98: PN8 has no effect (engine ceiling at 206K is gated by attention-block-size divisor, not KV)
- **PN8 not applied on dual-card configs** — `dual.yml` is intentionally Genesis-less (per its header), and adding Genesis structurally for one patch isn't worth it given dual.yml has no current cliff to close.
- Cross-rig data shared with Sandermage on [single-3090 #1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1#issuecomment-4343317153).

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
- **`docs/img/performance.svg`** + **`docs/img/performance.png`** — TPS bar chart across 10 single + dual configs, embedded in top-level README.
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
